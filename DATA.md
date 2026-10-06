# Dataset

This project uses the **UCF101** human action-recognition dataset.

UCF101 contains 101 action classes and 13,320 short video clips. The actions include sports, body movements, musical activities, and everyday actions.

Official dataset page:

https://www.crcv.ucf.edu/research/data-sets/ucf101/

## Dataset used in this project

Our complete UCF101 setup contains:

| Split | Number of videos |
| --- | ---: |
| Train | 9,154 |
| Validation | 1,421 |
| Test | 2,745 |
| **Total** | **13,320** |

All 13,320 entries referenced by the TEAM split files are available in the complete dataset, so the current setup has **0 missing videos**.

## Train, validation, and test split

We do **not** create a new random train/validation/test split.

Instead, we use the fixed class-disjoint UCF101 split provided by the TEAM project.

TEAM split source:

https://github.com/leesb7426/TEAM/tree/master/splits/ucf

The split definitions are included in this repository:

```text
splits/ucf/
|-- trainlist.txt
|-- vallist.txt
`-- testlist.txt
```

The three sets contain different action classes.

This is useful for few-shot recognition because evaluation should measure the ability to recognize unseen action classes rather than only new videos from classes already observed during training.

## UCF101 source groups

UCF101 filenames include a source-group identifier.

For example:

```text
v_Basketball_g07_c01.avi
```

Here:

```text
g07 = source group 7
c01 = clip 1
```

Several clips with the same `gXX` value can come from the same original recording.

Our main few-shot evaluation therefore uses a **group-safe** protocol. Support and query examples from the same action class cannot use the same source group.

This helps reduce near-duplicate leakage inside an episode.

## Expected raw dataset structure

The project expects the normal UCF101 directory structure:

```text
UCF-101/
|-- ApplyEyeMakeup/
|   |-- v_ApplyEyeMakeup_g01_c01.avi
|   `-- ...
|-- Archery/
|-- Basketball/
|-- ...
`-- YoYo/
```

The video dataset itself is not stored in this GitHub repository because it is several gigabytes in size.

The dataset path is passed to the experiment runner with:

```text
--dataset_arg data_root=/path/to/UCF-101
```

## Dataset loading

The main dataset implementation is:

```text
src/dataset_pipeline.py
```

It performs the following steps:

1. scans the UCF101 directories for `.avi` files;
2. matches videos with the TEAM split files;
3. extracts the class name and source-group ID;
4. checks that train, validation, and test classes are disjoint;
5. samples 16 frames from each video;
6. decodes and preprocesses the selected frames;
7. returns the clip together with its metadata.

With the complete dataset, the loader reports:

```text
train = 9154
validation = 1421
test = 2745
missing TEAM entries = 0
```

## Frame sampling

Every video is represented by 16 frames.

Three sampling strategies are evaluated.

### Uniform

The video timeline is divided into temporal segments and a representative frame is selected from each segment.

This spreads the selected frames across the video.

### Random

The timeline is divided into temporal segments and one frame is selected randomly from each segment.

The random sampling is deterministic for each video so that the same experiment can be reproduced.

### Consecutive

A block of 16 consecutive frames is selected around the middle of the video.

This keeps local motion information but covers less of the full video duration.

## Common preprocessing

The common preprocessing pipeline is:

```text
AVI video
-> selected frames
-> BGR to RGB
-> resize shorter side to 256 pixels
-> center crop to 224 x 224
-> stack 16 frames
```

The dataset loader returns:

```text
shape = (16, 3, 224, 224)
dtype = uint8
```

### AVI decoding robustness

A small number of UCF101 AVI files report frame positions that cannot actually be decoded.

The loader normally reads the requested frame using direct random access. If this fails, the video is decoded sequentially. When a requested frame index is beyond the last decodable frame, the final readable frame is repeated.

This fallback is deterministic and preserves the requested 16-frame clip length instead of dropping the video or stopping the experiment.

Backbone-specific normalization is not performed inside the dataset loader.

It is applied later in `src/backbones.py` because VideoMAE v2 and R(2+1)D-18 use different pretrained-model normalization settings.

## Two levels of splitting

There are two different kinds of splitting in this project.

### Dataset-level split

The TEAM files define:

```text
train classes
validation classes
test classes
```

These class sets are disjoint.

### Episode-level split

Inside one few-shot episode, examples are separated into:

```text
support examples
query examples
```

The main protocol additionally requires support and query examples to come from different UCF101 source groups.

These two splitting steps serve different purposes and should not be confused.
