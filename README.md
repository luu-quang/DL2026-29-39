# Few-Shot Video Action Recognition with Pretrained Video Models

This repository contains our Deep Learning final project on few-shot video action recognition.

The main question we study is:

> How well can a pretrained video model recognize a new action when only a small number of labeled examples are available?

Instead of training a large video model from scratch, we use pretrained video backbones as frozen feature extractors. A lightweight prototype-based head is then used for few-shot classification.

## Project pipeline

```text
UCF101 video
-> sample 16 frames
-> resize and center crop
-> frozen pretrained video backbone
-> video feature vector
-> N-way K-shot episode
-> class prototypes
-> scaled cosine similarity
-> predicted action
```

We compare two pretrained video models:

- **VideoMAE v2** - our main model
- **R(2+1)D-18** - our baseline

Both backbones remain frozen during the few-shot experiments.

## Research questions

We focus on three questions:

1. How does the pretrained backbone affect few-shot action recognition?
2. How do different frame-sampling strategies affect performance?
3. How much task-specific training data is needed when the video backbone is already pretrained?

## Dataset

We use the complete **UCF101** dataset, which contains 101 human action classes and 13,320 videos.

The project uses a fixed class-disjoint split from the TEAM project:

| Split | Videos |
| --- | ---: |
| Train | 9,154 |
| Validation | 1,421 |
| Test | 2,745 |
| **Total** | **13,320** |

With the complete UCF101 release, all TEAM split entries are available and the number of missing videos is 0.

More details are provided in `DATA.md`.

## Few-shot learning setup

An **N-way K-shot** episode contains:

- `N` action classes
- `K` labeled support examples for each class
- query videos that must be classified

For example, in **5-way 1-shot** learning, the model receives five possible classes and only one labeled support video for each class.

For each class, the support representations are averaged to form a prototype. Query videos are compared with these prototypes using cosine similarity.

Our main evaluation is **group-safe**. Support and query examples from the same action class cannot come from the same UCF101 source group (`gXX`). This reduces the chance that very similar clips from the same original recording appear on both sides of an episode.

## Models

### VideoMAE v2

The main backbone is:

```text
OpenGVLab/VideoMAEv2-Base
```

VideoMAE v2 produces a 768-dimensional feature representation for each video clip.

The pretrained backbone is frozen during the few-shot experiments.

### R(2+1)D-18

The baseline is torchvision R(2+1)D-18 pretrained on Kinetics-400.

It factorizes 3D convolution into spatial and temporal operations and produces a 512-dimensional video representation.

This backbone is also frozen.

## Frame sampling

Each video is represented using 16 frames.

We compare three strategies:

- `uniform` - frames are spread across the whole video
- `random` - one deterministic random frame is selected from each temporal segment
- `consecutive` - 16 consecutive frames are selected around the middle of the video

This lets us study whether the way temporal information is selected has a noticeable effect on few-shot recognition.

## Repository structure

```text
DL2026-29-39/
|-- README.md
|-- DATA.md
|-- requirements.txt
|-- splits/
|   `-- ucf/
|       |-- trainlist.txt
|       |-- vallist.txt
|       `-- testlist.txt
|-- src/
|   |-- dataset_pipeline.py
|   |-- backbones.py
|   |-- feature_extractor.py
|   |-- few_shot.py
|   `-- run_experiments.py
|-- notebooks/
|-- docs/
`-- results/
```

## What each source file does

### `src/dataset_pipeline.py`

Matches UCF101 videos with the TEAM split files, extracts source-group information, samples 16 frames, and performs the common video preprocessing.

### `src/backbones.py`

Loads VideoMAE v2 and R(2+1)D-18 and converts video clips into frozen feature vectors.

### `src/feature_extractor.py`

Extracts features and stores them in cache files. This prevents the pretrained backbones from processing the same videos again for every few-shot experiment.

### `src/few_shot.py`

Contains the episodic sampler, group-safe protocol, `ProtoHead`, training procedure, validation, and test evaluation.

### `src/run_experiments.py`

Runs the complete experiment pipeline and creates the result tables and figures.

## Installation

A CUDA-capable GPU is strongly recommended for feature extraction.

Install the required Python packages with:

```bash
pip install -r requirements.txt
```

## Running the experiments

From the repository root:

```bash
python src/run_experiments.py   --feature_root /path/to/UCF101_Features   --runs_file /path/to/UCF101_Features/results_team_split/runs.jsonl   --out_dir results   --dataset_arg data_root=/path/to/UCF-101
```

The pipeline performs:

```text
1. End-to-end smoke test
2. Feature extraction and cache validation
3. Group-safety audit
4. Backbone comparison
5. Frame-sampling comparison
6. N-way and K-shot experiments
7. Training-data fraction experiments
8. CSV and figure generation
```

## Experiment settings

The final experiment suite includes:

```text
Backbones:
- VideoMAE v2
- R(2+1)D-18

Sampling:
- uniform
- random
- consecutive

Ways:
- 5
- 10
- 20

Shots:
- 1
- 5

Training fractions:
- 2%
- 10%
- 50%
- 100%
```

## Output files

The experiment runner produces:

```text
results/
|-- main_5way.csv
|-- sampling_sweep.csv
|-- way_sweep.csv
|-- train_frac_sweep.csv
|-- plot_way_sweep.png
|-- plot_sampling.png
|-- plot_train_frac.png
|-- plot_protocol_gap.png
`-- handoff_report.json
```

These files are used to build the tables and figures in the final report.

## Validation status

The integrated pipeline has passed an end-to-end GPU smoke test on the complete UCF101 dataset.

The tested flow was:

```text
UCF101 video
-> 16 sampled RGB frames
-> tensor shape (16, 3, 224, 224)
-> VideoMAE v2 feature: 768 dimensions
-> R(2+1)D-18 feature: 512 dimensions
-> ProtoHead
-> cosine-similarity prediction
```

The smoke test is only used to verify that the complete pipeline works. Its tiny example episode is not treated as an experimental accuracy result.

Final performance numbers should come from the complete experiment run.

## Reproducibility

The project uses fixed seeds where deterministic sampling is needed.

The feature cache stores information about the backbone, pretrained weights, normalization, frame sampling, preprocessing, split, and video ordering. This helps prevent incompatible cached features from being reused silently.
