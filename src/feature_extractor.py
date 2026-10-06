"""Extract and cache frozen-backbone features for the TEAM splits.

Input  : the frozen dataset pipeline (dataset_pipeline.py). Each item is a dict with
         'clip' (16, 3, 224, 224) uint8 and metadata 'video_id', 'class_name', 'group_id', 'label', 'split'.
Output : one cache file per (backbone, sampling, split):
             FEATURE_ROOT/<backbone>/<sampling>_T16/<split>.pt
         containing 'features' (N, D) float16, 'video_ids', 'class_names', 'group_ids', 'labels' and 'settings'.
Next   : few_shot.py / run_experiments.py read these files with load_split_features().

Why a cache: running a video backbone on ~12,900 videos takes hours on Colab, while the few-shot
experiments only need the resulting vectors. Features are computed once and reused by every experiment.

No split / sampling / resizing code lives here: all of it comes from the frozen dataset pipeline.
"""
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

import dataset_pipeline
from backbones import describe_backbone, encode_clips, load_backbone

SPLITS = ["train", "val", "test"]
SAMPLINGS = ["uniform", "random", "consecutive"]
NUM_FRAMES = 16
SPATIAL_PREPROCESSING = "resize_short_256_center_crop_224"
CHUNK_SIZE = 500

# Where cached features came from (stored in the cache settings and in the experiment cache key):
#   "v3_notebook_verified" : features from the original v3 notebook, verified against this pipeline (Section 3)
#   "dataset_pipeline"     : features extracted with this file
LEGACY_GENERATION = "v3_notebook_verified"
NEW_GENERATION = "dataset_pipeline"
COMPATIBILITY_MIN_COSINE = 0.999

# Dataset paths are supplied by run_experiments.py. `data_root` must point to the folder containing
# the UCF101 class folders; if it points one level above, an inner UCF-101 folder is detected automatically.
DATASET_KWARGS = {}

# What happened to every cache file during this run; reported by run_experiments.py.
CACHE_REPORT = {"reused": [], "reused_after_verification": [], "regenerated": [], "verification": []}

loaded_models = {}
_dataset_records = None
_dataset_root_used = None
_split_dir_used = None


# ---------------------------------------------------------------------------------------------------------
# 1. The only place that calls the frozen dataset pipeline
# ---------------------------------------------------------------------------------------------------------
def _resolve_dataset_root(path):
    root = Path(path)
    nested = root / "UCF-101"
    if nested.is_dir():
        root = nested
    return root


def _load_dataset_records():
    global _dataset_records, _dataset_root_used, _split_dir_used
    if "data_root" not in DATASET_KWARGS:
        raise ValueError("Missing dataset argument: --dataset_arg data_root=/path/to/UCF-101")

    data_root = _resolve_dataset_root(DATASET_KWARGS["data_root"])
    default_split_dir = Path(__file__).resolve().parents[1] / "splits" / "ucf"
    split_dir = Path(DATASET_KWARGS.get("split_dir", default_split_dir))

    if _dataset_records is None or data_root != _dataset_root_used or split_dir != _split_dir_used:
        _, class_to_idx = dataset_pipeline.build_class_mapping(data_root)
        train_records, val_records, test_records = dataset_pipeline.load_team_splits(
            split_dir, data_root, class_to_idx
        )
        _dataset_records = {"train": train_records, "val": val_records, "test": test_records}
        _dataset_root_used = data_root
        _split_dir_used = split_dir

    return _dataset_records


def make_dataset(split, sampling):
    """Build one dataset from the actual frozen dataset_pipeline.py interface."""
    if split not in SPLITS:
        raise ValueError(f"Unknown split '{split}', expected one of {SPLITS}")
    if sampling not in SAMPLINGS:
        raise ValueError(f"Unknown sampling '{sampling}', expected one of {SAMPLINGS}")
    records = _load_dataset_records()[split]
    return dataset_pipeline.UCF101ClipDataset(
        records=records,
        sampling_strategy=sampling,
        num_frames=NUM_FRAMES,
        split_name=split,
    )


def dataset_status():
    records = _load_dataset_records()
    missing = getattr(dataset_pipeline, "LAST_MISSING", {name: [] for name in SPLITS})
    return {
        "matched": {name: len(records[name]) for name in SPLITS},
        "missing": {name: len(missing.get(name, [])) for name in SPLITS},
        "total_matched": sum(len(records[name]) for name in SPLITS),
        "total_missing": sum(len(missing.get(name, [])) for name in SPLITS),
    }


# ---------------------------------------------------------------------------------------------------------
# 2. Cache files: paths, settings (= cache key), safe save / load
# ---------------------------------------------------------------------------------------------------------
def feature_path(feature_root, backbone, sampling, split, chunk_index=None):
    folder = Path(feature_root) / backbone / f"{sampling}_T{NUM_FRAMES}"
    if chunk_index is None:
        return folder / f"{split}.pt"
    return folder / "chunks" / f"{split}_c{chunk_index:03d}.pt"


def expected_settings(backbone, sampling, split):
    """Everything that changes the features. A cache file is reused only if its settings are equal to these."""
    settings = describe_backbone(backbone)            # backbone, weights, normalization, model_input, precision
    settings["sampling"] = sampling
    settings["num_frames"] = NUM_FRAMES
    settings["spatial_preprocessing"] = SPATIAL_PREPROCESSING
    settings["split"] = split
    return settings


def settings_match(saved, backbone, sampling, split):
    stored = saved.get("settings")
    if stored is None:
        return False
    for key, value in expected_settings(backbone, sampling, split).items():
        if stored.get(key) != value:
            return False
    return True


def save_atomically(obj, path):
    """Write to '<name>.tmp' then rename: an interrupted save never leaves a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    torch.save(obj, temporary_path)
    os.replace(temporary_path, path)


def load_or_none(path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        return torch.load(path, map_location="cpu")
    except Exception as error:
        print(f"[cache] unreadable {path} ({error}); treated as missing.")
        return None


def get_model(backbone, device):
    if backbone not in loaded_models:
        loaded_models[backbone] = load_backbone(backbone, device)
    return loaded_models[backbone]


def as_list(value):
    """Batch metadata arrives as a tensor (numbers) or a list (strings); return a plain Python list."""
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def normalize_video_id(video_id):
    """'Archery/v_Archery_g01_c01' and 'v_Archery_g01_c01' both become 'v_archery_g01_c01'."""
    return str(video_id).split("/")[-1].lower()


def dataset_video_ids(dataset):
    """Video IDs in dataset order without decoding frames."""
    return [normalize_video_id(record["video_id"]) for record in dataset.records]


def cache_matches_dataset(saved, dataset):
    """A feature file is reusable only when its row IDs exactly match the current dataset order."""
    if saved is None or "video_ids" not in saved:
        return False
    saved_ids = [normalize_video_id(video_id) for video_id in saved["video_ids"]]
    return saved_ids == dataset_video_ids(dataset)


# ---------------------------------------------------------------------------------------------------------
# 3. Old caches from the v3 notebook: verify once, then reuse
# ---------------------------------------------------------------------------------------------------------
def verify_legacy_cache(feature_root, backbone, sampling, device, num_videos=8, batch_size=4, num_workers=2):
    """Old cache files have no 'settings'. They are reused only if the new pipeline reproduces them.

    We take a few test videos from the frozen dataset pipeline, extract their features now, and compare them
    with the cached vectors of the same videos (cosine similarity). Same frames + same preprocessing give
    cosine ~1.0 (small FP16 differences only). If it matches and the video counts match, the old files are
    stamped with settings and reused; otherwise they are moved aside and regenerated.
    """
    legacy_test = load_or_none(feature_path(feature_root, backbone, sampling, "test"))
    row_of_video = {}
    for row, video_id in enumerate(legacy_test["video_ids"]):
        row_of_video[normalize_video_id(video_id)] = row

    dataset = make_dataset("test", sampling)
    positions = []
    for k in range(num_videos):
        positions.append(int(k * (len(dataset) - 1) / max(num_videos - 1, 1)))
    loader = DataLoader(Subset(dataset, positions), batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = get_model(backbone, device)
    similarities = []
    all_found = True
    for batch in loader:
        new_features = encode_clips(backbone, model, batch["clip"].to(device))
        for i, video_id in enumerate(as_list(batch["video_id"])):
            row = row_of_video.get(normalize_video_id(video_id))
            if row is None:
                all_found = False
                continue
            old_feature = legacy_test["features"][row].float().reshape(-1).to(device)
            similarities.append(F.cosine_similarity(new_features[i], old_feature, dim=0).item())

    min_cosine = min(similarities) if similarities else float("nan")
    passed = all_found and len(similarities) == num_videos and min_cosine >= COMPATIBILITY_MIN_COSINE
    CACHE_REPORT["verification"].append({"backbone": backbone, "sampling": sampling,
                                         "videos_checked": len(similarities), "min_cosine": min_cosine,
                                         "passed": passed})
    print(f"[verify] {backbone}/{sampling}: {len(similarities)} videos, min cosine = {min_cosine:.5f} "
          f"-> {'REUSE' if passed else 'REGENERATE'}")
    return passed


def stamp_or_retire_legacy_files(feature_root, backbone, sampling, device):
    """Handle old (settings-less) cache files of one (backbone, sampling) for all three splits."""
    legacy_splits = []
    for split in SPLITS:
        saved = load_or_none(feature_path(feature_root, backbone, sampling, split))
        if saved is not None and "settings" not in saved:
            legacy_splits.append(split)
    if not legacy_splits:
        return

    passed = "test" in legacy_splits and verify_legacy_cache(feature_root, backbone, sampling, device)
    for split in legacy_splits:
        path = feature_path(feature_root, backbone, sampling, split)
        saved = load_or_none(path)
        dataset = make_dataset(split, sampling)
        same_dataset = cache_matches_dataset(saved, dataset)
        if passed and same_dataset:
            # Old files store class names in 'labels' and group numbers in 'groups'.
            saved["class_names"] = list(saved["labels"])
            saved["group_ids"] = list(saved["groups"])
            settings = expected_settings(backbone, sampling, split)
            settings["generation"] = LEGACY_GENERATION
            saved["settings"] = settings
            save_atomically(saved, path)
            CACHE_REPORT["reused_after_verification"].append(f"{backbone}/{sampling}/{split}")
        else:
            os.replace(path, path.with_name(f"{split}.legacy.pt"))   # keep the old file, but never reuse it
            print(f"[cache] {path} moved to {split}.legacy.pt; it will be regenerated.")


# ---------------------------------------------------------------------------------------------------------
# 4. Extraction
# ---------------------------------------------------------------------------------------------------------
def extract_split(feature_root, split, sampling, backbones, device, batch_size=8, num_workers=2):
    """Extract features of one split with one sampling strategy for every backbone whose cache is missing.

    Each video is decoded once (by the dataset pipeline) and fed to all missing backbones.
    Work is saved every CHUNK_SIZE videos, so a Colab disconnection loses at most one chunk.
    """
    dataset = make_dataset(split, sampling)
    backbones_to_do = []
    for backbone in backbones:
        saved = load_or_none(feature_path(feature_root, backbone, sampling, split))
        if (saved is not None and settings_match(saved, backbone, sampling, split)
                and cache_matches_dataset(saved, dataset)):
            CACHE_REPORT["reused"].append(f"{backbone}/{sampling}/{split}")
        else:
            backbones_to_do.append(backbone)
    if not backbones_to_do:
        return

    num_chunks = math.ceil(len(dataset) / CHUNK_SIZE)
    for chunk_index in range(num_chunks):
        start = chunk_index * CHUNK_SIZE
        end = min(len(dataset), start + CHUNK_SIZE)

        backbones_missing = []
        for backbone in backbones_to_do:
            chunk = load_or_none(feature_path(feature_root, backbone, sampling, split, chunk_index))
            expected_chunk_ids = dataset_video_ids(dataset)[start:end]
            chunk_done = (
                chunk is not None
                and settings_match(chunk, backbone, sampling, split)
                and [normalize_video_id(v) for v in chunk.get("video_ids", [])] == expected_chunk_ids
            )
            if not chunk_done:
                backbones_missing.append(backbone)
        if not backbones_missing:
            continue

        loader = DataLoader(Subset(dataset, range(start, end)), batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=device.type == "cuda")
        features = {}
        for backbone in backbones_missing:
            features[backbone] = []
        metadata = {"video_ids": [], "class_names": [], "group_ids": [], "labels": []}

        for batch in tqdm(loader, desc=f"{split}/{sampling} chunk {chunk_index + 1}/{num_chunks}", leave=False):
            clips = batch["clip"].to(device, non_blocking=True)          # (B, 16, 3, 224, 224) uint8
            for backbone in backbones_missing:
                vectors = encode_clips(backbone, get_model(backbone, device), clips)
                features[backbone].append(vectors.cpu().half())         # (B, D), stored as float16
            metadata["video_ids"].extend(as_list(batch["video_id"]))
            metadata["class_names"].extend(as_list(batch["class_name"]))
            metadata["group_ids"].extend(as_list(batch["group_id"]))
            metadata["labels"].extend(as_list(batch["label"]))

        for backbone in backbones_missing:
            chunk = dict(metadata)
            chunk["features"] = torch.cat(features[backbone])
            chunk["settings"] = expected_settings(backbone, sampling, split)
            save_atomically(chunk, feature_path(feature_root, backbone, sampling, split, chunk_index))
        del loader

    # All chunks exist: join them into the final file of each backbone.
    for backbone in backbones_to_do:
        record = {"features": [], "video_ids": [], "class_names": [], "group_ids": [], "labels": []}
        for chunk_index in range(num_chunks):
            chunk = load_or_none(feature_path(feature_root, backbone, sampling, split, chunk_index))
            record["features"].append(chunk["features"])
            for key in ["video_ids", "class_names", "group_ids", "labels"]:
                record[key].extend(chunk[key])
        record["features"] = torch.cat(record["features"])
        settings = expected_settings(backbone, sampling, split)
        settings["generation"] = NEW_GENERATION
        record["settings"] = settings

        output_path = feature_path(feature_root, backbone, sampling, split)
        save_atomically(record, output_path)
        CACHE_REPORT["regenerated"].append(f"{backbone}/{sampling}/{split}")
        print(f"[cache] saved {output_path}  features={tuple(record['features'].shape)}")


def prepare_feature_cache(feature_root, backbones, samplings, device, batch_size=8, num_workers=2):
    """Make sure every (backbone, sampling, split) cache exists and is valid: verify old files, extract the rest."""
    for backbone in backbones:
        for sampling in samplings:
            stamp_or_retire_legacy_files(feature_root, backbone, sampling, device)
    for sampling in samplings:
        for split in ["test", "val", "train"]:
            extract_split(feature_root, split, sampling, backbones, device, batch_size, num_workers)


# ---------------------------------------------------------------------------------------------------------
# 5. Loading for the few-shot experiments
# ---------------------------------------------------------------------------------------------------------
def load_split_features(feature_root, backbone, sampling, split):
    """Return (features (N, D) float32, class_names, group_ids, generation) for one cache file.

    Rows keep the order stored in the file, so episodes are sampled exactly as when the file was created.
    """
    path = feature_path(feature_root, backbone, sampling, split)
    saved = load_or_none(path)
    if saved is None or not settings_match(saved, backbone, sampling, split):
        raise RuntimeError(f"Missing or stale feature cache: {path}. Run prepare_feature_cache() first.")

    features = saved["features"].float()
    if features.dim() == 3:
        features = features.mean(dim=1)              # old files store (N, 1, D): one clip per video
    if "valid" in saved:
        features[~saved["valid"]] = float("nan")     # old files flag unreadable videos (none in our runs)
    return features, list(saved["class_names"]), list(saved["group_ids"]), saved["settings"]["generation"]
