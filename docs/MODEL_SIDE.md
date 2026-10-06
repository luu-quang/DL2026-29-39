# Model side — final handoff

Scope: VideoMAE v2 (main model) vs R(2+1)D-18 (baseline), shared ProtoHead. CLIP removed.
Pipeline: `dataset_pipeline.py` (frozen, dataset team) → `feature_extractor.py` → `few_shot.py` → `run_experiments.py`.
`backbones.py` is used by the extractor and the smoke test.

## Files

### backbones.py
- **Input:** `batch["clip"]`, uint8 `(B, 16, 3, 224, 224)`.
- **Main functions:** `load_backbone(name, device)` loads and freezes VideoMAE v2 or R(2+1)D-18; `encode_clips(name, model, clips)` converts to `(B, 3, 16, 224, 224)`, applies the model-specific normalisation (ImageNet for VideoMAE v2; Kinetics + trilinear 224→112 for R(2+1)D, unchanged from the existing experiment) and runs the frozen model in FP16; `describe_backbone(name)` returns the settings stored in the cache key.
- **Output:** VideoMAE v2 `(B, 768)`, R(2+1)D-18 `(B, 512)`, float32.
- **Next:** `feature_extractor.py`.

### feature_extractor.py
- **Input:** the frozen dataset pipeline via `make_dataset(split, sampling)`. The adapter uses the real dataset API: `build_class_mapping()` + `load_team_splits()` + `UCF101ClipDataset`.
- **Main functions:** `extract_split()` builds a DataLoader for one split and sampling, feeds `batch["clip"]` to every backbone whose cache is missing, saves a chunk every 500 videos, then a final file. `prepare_feature_cache()` runs all splits/samplings and handles old caches (`verify_legacy_cache()`, `stamp_or_retire_legacy_files()`). `load_split_features()` reads a cache file for the experiments.
- **Output:** `FEATURE_ROOT/<backbone>/<sampling>_T16/<split>.pt` with `features (N, D)` float16, `video_ids`, `class_names`, `group_ids`, `labels`, and `settings`.
- **Cache key (checked before reuse):** backbone, weights/checkpoint, normalisation, model input, precision, sampling, num_frames = 16, spatial preprocessing = resize_short_256_center_crop_224, split, plus the stored video IDs/count. `settings.generation` records whether the file is a verified v3-notebook cache or a new extraction.
- **Next:** `few_shot.py` / `run_experiments.py`.

### few_shot.py
- **Input:** features `(N, D)` + class name and group of each video.
- **Main functions:** `FewShotDataset.sample_episode()` builds N-way K-shot episodes; support and query never share a video. With `group_safe=True`, queries come only from source groups not used by the support (`pick_group_safe()`). `group_safe=False` is now called **standard_random** (group overlap allowed). Also contains `ProtoHead` (Linear → L2 norm → class prototypes → cosine → learnable scale → logits), `evaluate_head()` (mean accuracy + 95 % CI), `learning_rate_at()` (TEAM schedule), `train_and_evaluate()` (no-train test → 2000 episodic SGD iterations → best on validation → test), and `audit_group_safety()`.
- **Trainable parameters:** the pretrained backbones are frozen. Only ProtoHead's projection layer and learnable scale are trained.
- **Output:** one result dict per experiment.
- **Next:** `run_experiments.py`.

### run_experiments.py
- **Input:** cache folder, `runs.jsonl` (stored experiment results), dataset arguments.
- **Main functions:**
  1. `smoke_test()`: dataset → DataLoader → shape checks → both backbones → one 2-way 1-shot episode → ProtoHead prediction.
  2. `feature_extractor.prepare_feature_cache()`.
  3. Group-safety audit.
  4. `run_cached()` for every setting. A stored run is reused only if all its settings match, including `feature_generation`.
  5. `make_figures()`.
- **Output:**
  - CSV files: `main_5way.csv`, `sampling_sweep.csv`, `way_sweep.csv`, `train_frac_sweep.csv`.
  - Figures: `plot_way_sweep.png`, `plot_sampling.png`, `plot_train_frac.png`, `plot_protocol_gap.png`.
  - `handoff_report.json`: smoke test status, which caches were reused, verified or regenerated, and which runs were reused or recomputed.
- **Next:** CSV/figures → `results/`, README.

Result columns: `notrain_standard_random, notrain_group_safe, trained_standard_random, trained_standard_random_ci95, trained_group_safe, trained_group_safe_ci95` (renamed from `notrain, notrain_strict, trained, trained_conf, trained_strict, trained_strict_conf`; values unchanged). Report `trained_group_safe ± trained_group_safe_ci95`.

## Settings (unchanged from the v3 experiment)

seed 1234 · group-safe training/validation · 6 queries/class (train), 1 (test) · 2000 iterations · SGD lr 0.001, momentum 0.9, Nesterov, wd 5e-4 · LR × [1, 0.5, 0.1, 0.01] at epochs [0, 3, 5, 7], 200 it/epoch · validation every 100 it on 300 episodes (min(way, 10)-way) · 10 000 test episodes · way {5, 10, 20} · shot {1, 5} · train fraction {0.02, 0.1, 0.5, 1.0}.

## Legacy v3 cache compatibility

The compatibility logic remains in the code for old v3 caches, but the canonical full-dataset run uses a fresh feature root.

- The old local copy physically contained **12,902 / 13,320 UCF101 files**.
- Only **12,900** of those files matched TEAM entries and were used in the v3 experiment.
- The remaining 2 files had no TEAM entry and were never used.
- Therefore the v3 feature caches contain **12,900 TEAM-matched rows**.
- These IDs cannot match the complete **13,320-video** canonical split.
- The final run therefore uses fresh caches under `UCF101_Features_full13320`.
- v3 numbers are historical only and must not be mixed with final full-dataset results.

## Status report

| Question | Answer |
|---|---|
| Smoke test passed? | **YES.** End-to-end GPU smoke test passed on the complete official UCF101 dataset. TEAM matched train=9,154, val=1,421, test=2,745 with 0 missing. Dataset batch shape was `(4, 16, 3, 224, 224)` uint8; VideoMAE v2 produced `(4, 768)` and R(2+1)D-18 produced `(4, 512)`. Both successfully interfaced with ProtoHead. |
| Existing v3 caches reused in the canonical full run? | **NO.** The full run uses the fresh `UCF101_Features_full13320` root. |
| Full-dataset experiment? | **COMPLETE.** The canonical sweep finished on all 13,320 videos and produced the final CSV files, figures, and `handoff_report.json`. |
| Final result numbers? | **FINALIZED.** Reported results come only from the completed full-dataset CSVs in `results/`; old v3 results are historical only. |

## Confirmed dataset integration

- Complete official UCF101: **13,320 videos, 101 classes**.
- Fixed TEAM class-disjoint split: **train 9,154 / val 1,421 / test 2,745; 0 missing**.
- `feature_extractor.make_dataset()` uses `build_class_mapping()`, `load_team_splits()`, and `UCF101ClipDataset`.
- Dataset items contain `clip` uint8 `(16, 3, 224, 224)`, `video_id`, `class_name`, `group_id`, `label`, and `split`.
- New feature caches are reused only when their ordered video IDs exactly match the current dataset.
- Group-safe episodes require support and query source groups (`gXX`) to be disjoint within each class.
