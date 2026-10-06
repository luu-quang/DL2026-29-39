from pathlib import Path
import random
import re
import zlib

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


NUM_FRAMES = 16
RESIZE_SHORTER_SIDE = 256
CROP_SIZE = 224
RANDOM_SEED_BASE = 1234
RANDOM_SEED_MULTIPLIER = 1000003


def build_class_mapping(dataset_root: Path):
    """Return sorted class names and class -> numeric label mapping."""
    classes = []
    for folder in Path(dataset_root).iterdir():
        if folder.is_dir():
            classes.append(folder.name)

    classes.sort()
    class_to_idx = {}
    for label, class_name in enumerate(classes):
        class_to_idx[class_name] = label

    return classes, class_to_idx


def scan_video_files(dataset_root: Path):
    """Index real UCF101 .avi files by lower-case filename stem."""
    videos_by_stem = {}

    # TEAM entries are matched by filename stem, not assumed to be valid local paths.
    for path in sorted(Path(dataset_root).rglob("*")):
        if not path.is_file() or path.suffix.lower() != ".avi":
            continue

        stem_key = path.stem.lower()
        if stem_key not in videos_by_stem:
            videos_by_stem[stem_key] = path

    if not videos_by_stem:
        raise ValueError(f"No .avi videos found under {dataset_root}.")

    return videos_by_stem


def parse_group_id(video_stem: str):
    """Parse gXX from a UCF101 name such as v_Basketball_g07_c01."""
    match = re.search(r"_g(\d+)_c\d+$", video_stem, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(
            f"Could not parse group_id from '{video_stem}'. Expected _gXX_cXX."
        )
    return int(match.group(1))


def load_team_split(split_file: Path, videos_by_stem, class_to_idx, missing=None):
    """Match one TEAM split file to available UCF101 videos by filename stem."""
    records = []

    with open(split_file, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            team_entry = line.split()[0]
            stem_key = Path(team_entry).stem.lower()

            if stem_key not in videos_by_stem:
                if missing is None:
                    raise FileNotFoundError(
                        f"Line {line_number} of {Path(split_file).name} has no matching video: '{stem_key}'."
                    )
                missing.append(team_entry)
                continue

            video_path = videos_by_stem[stem_key]
            video_id = video_path.stem
            class_name = video_path.parent.name

            if class_name not in class_to_idx:
                raise ValueError(f"Unknown class '{class_name}' for {video_path}.")

            records.append(
                {
                    "video_path": video_path,
                    "video_id": video_id,
                    "class_name": class_name,
                    "label": class_to_idx[class_name],
                    "group_id": parse_group_id(video_id),
                }
            )

    return records


def check_team_split_disjointness(train_records, val_records, test_records):
    """Fail if TEAM train/val/test share classes or source groups."""
    split_records = {
        "train": train_records,
        "val": val_records,
        "test": test_records,
    }

    class_sets = {}
    class_group_sets = {}

    for split_name, records in split_records.items():
        class_sets[split_name] = {record["class_name"] for record in records}
        class_group_sets[split_name] = {
            (record["class_name"], record["group_id"]) for record in records
        }

    for first_split, second_split in [
        ("train", "val"),
        ("train", "test"),
        ("val", "test"),
    ]:
        shared_classes = class_sets[first_split] & class_sets[second_split]
        if shared_classes:
            raise ValueError(
                f"{first_split}/{second_split} share classes: "
                f"{sorted(shared_classes)[:10]}"
            )

        shared_groups = (
            class_group_sets[first_split] & class_group_sets[second_split]
        )
        if shared_groups:
            raise ValueError(
                f"{first_split}/{second_split} share (class, group): "
                f"{sorted(shared_groups)[:10]}"
            )

    return True


LAST_MISSING = {"train": [], "val": [], "test": []}


def load_team_splits(split_dir: Path, dataset_root: Path, class_to_idx):
    """Load available TEAM train/val/test videos and verify split disjointness."""
    split_dir = Path(split_dir)
    videos_by_stem = scan_video_files(dataset_root)

    missing = {"train": [], "val": [], "test": []}
    train_records = load_team_split(
        split_dir / "trainlist.txt", videos_by_stem, class_to_idx, missing["train"]
    )
    val_records = load_team_split(
        split_dir / "vallist.txt", videos_by_stem, class_to_idx, missing["val"]
    )
    test_records = load_team_split(
        split_dir / "testlist.txt", videos_by_stem, class_to_idx, missing["test"]
    )

    check_team_split_disjointness(train_records, val_records, test_records)
    LAST_MISSING.clear()
    LAST_MISSING.update(missing)
    total_missing = sum(len(items) for items in missing.values())
    print(
        f"[dataset] matched train={len(train_records)}, val={len(val_records)}, test={len(test_records)}; "
        f"missing TEAM entries={total_missing}"
    )
    return train_records, val_records, test_records


def _short_video_indices(total_frames: int, num_frames: int):
    # Match the experiment: spread 16 positions over the available frames.
    return (
        np.linspace(0, total_frames - 1, num_frames)
        .round()
        .astype(np.int64)
    )


def uniform_frame_indices(total_frames: int, num_frames: int = NUM_FRAMES):
    """Take the center frame of each temporal segment."""
    if total_frames <= 0:
        raise ValueError("Video contains no readable frames.")
    if total_frames < num_frames:
        return _short_video_indices(total_frames, num_frames)

    segment = total_frames // num_frames
    indices = []
    for k in range(num_frames):
        indices.append(k * segment + (segment - 1) // 2)

    return np.array(indices, dtype=np.int64)


def random_frame_indices(
    total_frames: int,
    video_stem: str,
    num_frames: int = NUM_FRAMES,
):
    """Take one deterministic random frame from each temporal segment."""
    if total_frames <= 0:
        raise ValueError("Video contains no readable frames.")
    if total_frames < num_frames:
        return _short_video_indices(total_frames, num_frames)

    segment = total_frames // num_frames

    # Per-video seed makes random sampling reproducible across runs.
    checksum = zlib.crc32(video_stem.lower().encode("utf-8"))
    seed = RANDOM_SEED_BASE * RANDOM_SEED_MULTIPLIER + checksum
    rng = random.Random(seed)

    indices = []
    for k in range(num_frames):
        start = k * segment
        end = (k + 1) * segment - 1
        indices.append(rng.randint(start, end))

    return np.array(indices, dtype=np.int64)


def consecutive_frame_indices(total_frames: int, num_frames: int = NUM_FRAMES):
    """Take a centered consecutive block; repeat the last frame if too short."""
    if total_frames <= 0:
        raise ValueError("Video contains no readable frames.")

    start = max((total_frames - num_frames) // 2, 0)
    indices = np.arange(start, start + num_frames, dtype=np.int64)
    return np.clip(indices, 0, total_frames - 1)


def resize_shorter_side_and_center_crop(
    rgb_frame: np.ndarray,
    resize_shorter_side: int = RESIZE_SHORTER_SIDE,
    crop_size: int = CROP_SIZE,
):
    """Resize short side to 256 while keeping aspect ratio, then crop 224x224."""
    height, width = rgb_frame.shape[:2]

    if height <= width:
        new_height = resize_shorter_side
        new_width = round(width * resize_shorter_side / height)
    else:
        new_width = resize_shorter_side
        new_height = round(height * resize_shorter_side / width)

    resized = cv2.resize(
        rgb_frame,
        (new_width, new_height),
        interpolation=cv2.INTER_LINEAR,
    )

    top = (new_height - crop_size) // 2
    left = (new_width - crop_size) // 2
    return resized[top : top + crop_size, left : left + crop_size]


def read_and_preprocess_clip(video_path: Path, frame_indices):
    """Read selected RGB frames and return uint8 tensor (T, C, 224, 224)."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    processed_frames = []
    for frame_index in frame_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        success, frame = capture.read()
        if not success:
            capture.release()
            raise RuntimeError(
                f"Could not read frame {frame_index} from {video_path}"
            )

        # OpenCV is BGR; pretrained vision backbones expect RGB input.
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = resize_shorter_side_and_center_crop(frame)
        processed_frames.append(frame)

    capture.release()

    clip = torch.from_numpy(np.stack(processed_frames))
    # Dataset returns time-first, channel-first frames: (T,H,W,C) -> (T,C,H,W).
    return clip.permute(0, 3, 1, 2).contiguous()


class UCF101ClipDataset(Dataset):
    """Return one 16-frame clip plus metadata needed by the model/few-shot code."""

    def __init__(
        self,
        records,
        sampling_strategy: str = "uniform",
        num_frames: int = NUM_FRAMES,
        split_name: str = "",
    ):
        self.records = records
        self.sampling_strategy = sampling_strategy
        self.num_frames = num_frames
        self.split_name = split_name

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        video_path = record["video_path"]
        video_id = record["video_id"]

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()

        if self.sampling_strategy == "uniform":
            frame_indices = uniform_frame_indices(total_frames, self.num_frames)
        elif self.sampling_strategy == "random":
            frame_indices = random_frame_indices(
                total_frames,
                video_id,
                self.num_frames,
            )
        elif self.sampling_strategy == "consecutive":
            frame_indices = consecutive_frame_indices(
                total_frames,
                self.num_frames,
            )
        else:
            raise ValueError(
                "sampling_strategy must be 'uniform', 'random', or 'consecutive'."
            )

        clip = read_and_preprocess_clip(video_path, frame_indices)

        return {
            "clip": clip,
            "label": record["label"],
            "video_id": video_id,
            "video_path": str(video_path),
            "class_name": record["class_name"],
            "group_id": record["group_id"],
            "frame_indices": torch.tensor(frame_indices, dtype=torch.long),
            "split": self.split_name,
        }


def create_dataloader(
    records,
    sampling_strategy: str,
    split_name: str,
    batch_size: int = 4,
    shuffle: bool = False,
    num_workers: int = 0,
):
    """Build a DataLoader over UCF101ClipDataset."""
    dataset = UCF101ClipDataset(
        records=records,
        sampling_strategy=sampling_strategy,
        num_frames=NUM_FRAMES,
        split_name=split_name,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
