"""Run the final experiments: VideoMAE v2 (main model) vs R(2+1)D-18 (baseline), with ProtoHead.

Input  : the frozen dataset pipeline (through feature_extractor.py) and the feature cache on Drive.
Steps  : 1. smoke test (dataset -> both backbones -> one episode -> ProtoHead)
         2. prepare the feature cache (verify / reuse / regenerate)
         3. group-safety audit of the episode sampler
         4. experiments: backbone comparison, sampling sweep, way sweep, training-data fraction sweep
         5. CSV files, figures and handoff_report.json
Output : OUT_DIR/main_5way.csv, sampling_sweep.csv, way_sweep.csv, train_frac_sweep.csv,
         plot_way_sweep.png, plot_sampling.png, plot_train_frac.png, plot_protocol_gap.png, handoff_report.json

Usage from the repository root (Colab):
    python src/run_experiments.py --feature_root /content/drive/MyDrive/UCF101_Features \
        --runs_file /content/drive/MyDrive/UCF101_Features/results_team_split/runs.jsonl --out_dir results \
        --dataset_arg data_root=/content/drive/MyDrive/Dataset/UCF101

`data_root` may point directly to UCF-101 or to a parent folder containing UCF-101.
The TEAM split directory defaults to <repo>/splits/ucf; override it with --dataset_arg split_dir=/path if needed.

Every finished experiment is stored as one line in --runs_file. An experiment is recomputed only if no stored
run has exactly the same settings (including where its features came from), so re-running is cheap and safe.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

import feature_extractor
from backbones import BACKBONES, FEATURE_DIM, encode_clips, get_device
from few_shot import FewShotDataset, ProtoHead, audit_group_safety, train_and_evaluate

SAMPLINGS = ["uniform", "random", "consecutive"]
WAYS = [5, 10, 20]
SHOTS = [1, 5]
TRAIN_FRACS = [0.02, 0.1, 0.5, 1.0]

# Experiment settings: identical to the v3 notebook that produced the current results.
CONFIG = {
    "seed": 1234,
    "group_safe": True,            # training and validation episodes are group-safe; reported test result too
    "queries_train": 6,
    "queries_test": 1,
    "train_iters": 2000,
    "iters_per_epoch": 200,
    "val_every": 100,
    "val_tasks": 300,
    "test_tasks": 10000,
    "lr": 0.001,
    "lr_steps": [0, 3, 5, 7],
    "lr_factors": [1, 0.5, 0.1, 0.01],
    "max_epoch": 10,
}

# Runs saved by the v3 notebook only stored part of their settings in their key. The missing ones had these values.
LEGACY_RUN_DEFAULTS = {
    "frames": 16, "val_tasks": 300, "queries_train": 6, "queries_test": 1, "lr": 0.001, "val_every": 100,
    "iters_per_epoch": 200, "lr_steps": [0, 3, 5, 7], "lr_factors": [1, 0.5, 0.1, 0.01], "max_epoch": 10,
    "feature_generation": feature_extractor.LEGACY_GENERATION,
}

# Column names used by the v3 notebook -> clearer names used now ("standard_random" = group overlap allowed).
OLD_TO_NEW_COLUMNS = {
    "notrain": "notrain_standard_random", "notrain_strict": "notrain_group_safe",
    "trained": "trained_standard_random", "trained_conf": "trained_standard_random_ci95",
    "trained_strict": "trained_group_safe", "trained_strict_conf": "trained_group_safe_ci95",
}

RESULT_COLUMNS = ["backbone", "sampling", "way", "shot", "train_frac", "chance",
                  "notrain_standard_random", "notrain_group_safe",
                  "trained_standard_random", "trained_standard_random_ci95",
                  "trained_group_safe", "trained_group_safe_ci95",
                  "val_best", "best_iter", "group_safe_train", "n_train_per_class"]

REPORT = {"smoke_test_passed": False, "runs_reused": [], "runs_computed": []}


# ---------------------------------------------------------------------------------------------------------
# 1. Smoke test
# ---------------------------------------------------------------------------------------------------------
def smoke_test(device, num_videos=48, batch_size=8):
    """dataset -> DataLoader -> batch['clip'] -> VideoMAE v2 and R(2+1)D features -> one 2-way 1-shot episode
    -> ProtoHead -> prediction. Raises an error if any shape is wrong."""
    dataset = feature_extractor.make_dataset("test", "uniform")
    positions = []
    for k in range(num_videos):
        positions.append(int(k * (len(dataset) - 1) / (num_videos - 1)))   # spread over the split = many classes
    loader = DataLoader(Subset(dataset, positions), batch_size=batch_size, shuffle=False)

    models = {}
    for backbone in BACKBONES:
        models[backbone] = feature_extractor.get_model(backbone, device)

    features = {}
    for backbone in BACKBONES:
        features[backbone] = []
    class_names = []
    for batch_number, batch in enumerate(loader):
        clips = batch["clip"]
        assert clips.dtype == torch.uint8, f"clip dtype {clips.dtype}, expected torch.uint8"
        assert tuple(clips.shape[1:]) == (16, 3, 224, 224), f"clip shape {tuple(clips.shape)}"
        if batch_number == 0:
            print(f"[smoke] dataset batch: {tuple(clips.shape)} {clips.dtype}")
        clips = clips.to(device)
        for backbone in BACKBONES:
            vectors = encode_clips(backbone, models[backbone], clips)
            assert tuple(vectors.shape) == (clips.shape[0], FEATURE_DIM[backbone]), f"{backbone}: {tuple(vectors.shape)}"
            if batch_number == 0:
                print(f"[smoke] {backbone} features: {tuple(vectors.shape)}")
            features[backbone].append(vectors)
        class_names.extend(feature_extractor.as_list(batch["class_name"]))

    # one 2-way 1-shot episode: two classes with at least two videos (1 support + 1 query each)
    rows_of_class = {}
    for row, class_name in enumerate(class_names):
        rows_of_class.setdefault(class_name, []).append(row)
    episode_classes = []
    for class_name, rows in rows_of_class.items():
        if len(rows) >= 2:
            episode_classes.append(class_name)
    assert len(episode_classes) >= 2, "smoke test: not enough classes with two videos; increase num_videos"
    support_rows = [rows_of_class[episode_classes[0]][0], rows_of_class[episode_classes[1]][0]]
    query_rows = [rows_of_class[episode_classes[0]][1], rows_of_class[episode_classes[1]][1]]

    for backbone in BACKBONES:
        all_features = torch.cat(features[backbone])
        head = ProtoHead(FEATURE_DIM[backbone]).to(device)
        with torch.no_grad():
            logits = head(all_features[support_rows], all_features[query_rows], way=2, shot=1)
        predictions = logits.argmax(-1).cpu()
        accuracy = (predictions == torch.tensor([0, 1])).float().mean().item()
        print(f"[smoke] {backbone}: ProtoHead logits {tuple(logits.shape)}, predictions {predictions.tolist()}, "
              f"accuracy {accuracy:.2f}")
    REPORT["smoke_test_passed"] = True
    print("[smoke] PASSED")


# ---------------------------------------------------------------------------------------------------------
# 2. Few-shot data and the experiment cache
# ---------------------------------------------------------------------------------------------------------
loaded_data = {}


def get_data(feature_root, backbone, sampling, device):
    """{'train', 'val', 'test'} -> FewShotDataset for one (backbone, sampling); loaded once."""
    key = (backbone, sampling)
    if key not in loaded_data:
        data = {}
        for split in ["train", "val", "test"]:
            features, class_names, group_ids, generation = feature_extractor.load_split_features(
                feature_root, backbone, sampling, split)
            data[split] = FewShotDataset(features, class_names, group_ids, device)
            data["generation"] = generation
        loaded_data[key] = data
    return loaded_data[key]


generation_of_cache = {}


def feature_generation(feature_root, backbone, sampling):
    """Where the cached features of (backbone, sampling) came from; part of the experiment cache key."""
    key = (backbone, sampling)
    if key not in generation_of_cache:
        path = feature_extractor.feature_path(feature_root, backbone, sampling, "test")
        generation_of_cache[key] = feature_extractor.load_or_none(path)["settings"]["generation"]
    return generation_of_cache[key]


def run_settings(feature_root, backbone, sampling, way, shot, train_frac):
    settings = {"backbone": backbone, "sampling": sampling, "way": way, "shot": shot, "train_frac": train_frac,
                "group_safe": CONFIG["group_safe"], "frames": 16, "iters": CONFIG["train_iters"],
                "test_tasks": CONFIG["test_tasks"], "seed": CONFIG["seed"],
                "feature_generation": feature_generation(feature_root, backbone, sampling)}
    for key in ["val_tasks", "queries_train", "queries_test", "lr", "val_every", "iters_per_epoch",
                "lr_steps", "lr_factors", "max_epoch"]:
        settings[key] = CONFIG[key]
    return settings


def same_settings(saved_settings, wanted_settings):
    for key, value in wanted_settings.items():
        saved_value = saved_settings.get(key, LEGACY_RUN_DEFAULTS.get(key, "<missing>"))
        if saved_value != value:
            return False
    return True


def run_cached(runs_file, feature_root, device, backbone, sampling, way, shot, train_frac=1.0):
    """Return a stored result with identical settings, otherwise run the experiment and store it."""
    wanted = run_settings(feature_root, backbone, sampling, way, shot, train_frac)
    runs_file = Path(runs_file)
    if runs_file.exists():
        for line in open(runs_file):
            saved = json.loads(line)
            if "settings" in saved:
                saved_settings = saved["settings"]
            else:
                saved_settings = json.loads(saved["key"])         # runs written by the v3 notebook
            if same_settings(saved_settings, wanted):
                result = {}
                for name, value in saved.items():
                    result[OLD_TO_NEW_COLUMNS.get(name, name)] = value
                REPORT["runs_reused"].append(f"{backbone}/{sampling}/{way}w{shot}s/frac{train_frac}")
                return result

    data = get_data(feature_root, backbone, sampling, device)
    result = train_and_evaluate(data["train"], data["val"], data["test"], way, shot, train_frac, CONFIG, device)
    result["backbone"] = backbone
    result["sampling"] = sampling
    stored = dict(result)
    stored["settings"] = wanted
    runs_file.parent.mkdir(parents=True, exist_ok=True)
    with open(runs_file, "a") as f:
        f.write(json.dumps(stored, default=float) + "\n")
    REPORT["runs_computed"].append(f"{backbone}/{sampling}/{way}w{shot}s/frac{train_frac}")
    print(f"done: {backbone:12s} {sampling:11s} {result['way']:2d}-way {shot}-shot frac={train_frac:<4} "
          f"group_safe={result['trained_group_safe']:.1f}")
    return result


def to_table(rows):
    return pd.DataFrame(rows)[RESULT_COLUMNS]


# ---------------------------------------------------------------------------------------------------------
# 3. Figures (group-safe accuracy with 95 % confidence interval)
# ---------------------------------------------------------------------------------------------------------
def make_figures(out_dir, way_df, sampling_df, frac_df, main_df):
    fig, axes = plt.subplots(1, len(SHOTS), figsize=(5.5 * len(SHOTS), 4), sharey=True, squeeze=False)
    axes = axes[0]
    for ax, shot in zip(axes, SHOTS):
        for backbone in BACKBONES:
            data = way_df[(way_df.backbone == backbone) & (way_df.shot == shot)].sort_values("way")
            ax.errorbar(data.way, data.trained_group_safe, yerr=data.trained_group_safe_ci95, marker="o",
                        capsize=3, label=backbone)
        ax.set_title(f"{shot}-shot, uniform sampling")
        ax.set_xlabel("way")
        ax.set_xticks(WAYS)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("group-safe accuracy (%)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "plot_way_sweep.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(SHOTS), figsize=(5.5 * len(SHOTS), 4), sharey=True, squeeze=False)
    axes = axes[0]
    bar_width = 0.8 / len(SAMPLINGS)
    positions = np.arange(len(BACKBONES))
    for ax, shot in zip(axes, SHOTS):
        for j, sampling in enumerate(SAMPLINGS):
            data = sampling_df[(sampling_df.sampling == sampling) & (sampling_df.shot == shot)]
            data = data.set_index("backbone").loc[BACKBONES]
            ax.bar(positions + j * bar_width, data.trained_group_safe, bar_width, yerr=data.trained_group_safe_ci95,
                   capsize=2, label=sampling)
        ax.set_xticks(positions + bar_width * (len(SAMPLINGS) - 1) / 2)
        ax.set_xticklabels(BACKBONES)
        lowest = sampling_df[sampling_df.shot == shot].trained_group_safe.min()
        ax.set_ylim(max(0, lowest - 5), 100)
        ax.set_title(f"5-way {shot}-shot")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("group-safe accuracy (%)")
    axes[0].legend(title="sampling")
    fig.tight_layout()
    fig.savefig(out_dir / "plot_sampling.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for backbone in BACKBONES:
        data = frac_df[frac_df.backbone == backbone].sort_values("train_frac")
        ax.errorbar(data.train_frac, data.trained_group_safe, yerr=data.trained_group_safe_ci95, marker="o",
                    capsize=3, label=backbone)
    ax.set_xscale("log")
    ax.set_xticks(TRAIN_FRACS)
    ax.set_xticklabels(TRAIN_FRACS)
    ax.set_xlabel("fraction of training videos per class")
    ax.set_ylabel("group-safe accuracy (%)")
    ax.set_title("5-way 1-shot, uniform sampling")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "plot_train_frac.png", dpi=150)
    plt.close(fig)

    gap_df = main_df.copy()
    gap_df["gap"] = gap_df.trained_standard_random - gap_df.trained_group_safe
    gap_table = gap_df.pivot(index="backbone", columns="shot", values="gap").loc[BACKBONES]
    ax = gap_table.plot.bar(figsize=(6, 4), rot=0)
    ax.set_ylabel("standard_random - group_safe (pp)")
    ax.set_title("Accuracy difference between protocols (5-way, trained head)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "plot_protocol_gap.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_root", default="/content/drive/MyDrive/UCF101_Features")
    parser.add_argument("--runs_file", default="/content/drive/MyDrive/UCF101_Features/results_team_split/runs.jsonl")
    parser.add_argument("--out_dir", default="results")
    parser.add_argument("--dataset_arg", action="append", default=[],
                        help="dataset key=value (required: data_root; optional: split_dir), repeatable")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    for item in args.dataset_arg:
        key, value = item.split("=", 1)
        feature_extractor.DATASET_KWARGS[key] = value
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print("Device:", device)

    # 1. smoke test: nothing else runs if it fails
    smoke_test(device)

    # 2. feature cache: verify old files, reuse valid ones, extract what is missing
    feature_extractor.prepare_feature_cache(args.feature_root, BACKBONES, SAMPLINGS, device,
                                            args.batch_size, args.num_workers)

    # 3. group-safety audit (test split, main model, uniform)
    test_data = get_data(args.feature_root, BACKBONES[0], "uniform", device)["test"]
    overlap_percent = audit_group_safety(test_data, CONFIG["seed"])
    print("Group-safe sampler: no shared source group in 2000 class splits (assertion passed).")
    print(f"standard_random sampler: {overlap_percent:.1f}% of 5-shot class splits share a group between "
          f"support and query.")
    REPORT["standard_random_overlap_percent"] = overlap_percent

    def run(backbone, sampling, way, shot, train_frac=1.0):
        return run_cached(args.runs_file, args.feature_root, device, backbone, sampling, way, shot, train_frac)

    # 4. experiments
    rows = []
    for backbone in BACKBONES:
        for shot in SHOTS:
            rows.append(run(backbone, "uniform", 5, shot))
    main_df = to_table(rows)
    main_df.to_csv(out_dir / "main_5way.csv", index=False)

    rows = []
    for backbone in BACKBONES:
        for sampling in SAMPLINGS:
            for shot in SHOTS:
                rows.append(run(backbone, sampling, 5, shot))
    sampling_df = to_table(rows)
    sampling_df.to_csv(out_dir / "sampling_sweep.csv", index=False)

    rows = []
    for backbone in BACKBONES:
        for way in WAYS:
            for shot in SHOTS:
                rows.append(run(backbone, "uniform", way, shot))
    way_df = to_table(rows)
    way_df.to_csv(out_dir / "way_sweep.csv", index=False)

    rows = []
    for backbone in BACKBONES:
        for fraction in TRAIN_FRACS:
            rows.append(run(backbone, "uniform", 5, 1, fraction))
    frac_df = to_table(rows)
    frac_df.to_csv(out_dir / "train_frac_sweep.csv", index=False)

    # 5. figures, summary, report
    make_figures(out_dir, way_df, sampling_df, frac_df, main_df)
    for _, row in main_df.iterrows():
        print(f"{row.backbone:12s} 5-way {int(row.shot)}-shot | no-train(group-safe) {row.notrain_group_safe:.1f} | "
              f"trained(group-safe) {row.trained_group_safe:.1f} ± {row.trained_group_safe_ci95:.2f} | "
              f"standard_random {row.trained_standard_random:.1f}")

    REPORT["dataset"] = feature_extractor.dataset_status()
    REPORT["cache"] = feature_extractor.CACHE_REPORT
    with open(out_dir / "handoff_report.json", "w") as f:
        json.dump(REPORT, f, indent=2, default=float)
    print(f"Saved CSV files, figures and handoff_report.json to {out_dir}")


if __name__ == "__main__":
    main()
