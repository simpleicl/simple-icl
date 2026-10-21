# SimpleICL

This repository is the official code release for the paper **In-context Robot Learning Made Simple: A Democratized Recipe for Manipulation Tasks**.

[![English](https://img.shields.io/badge/README-English-111111.svg)](./README.md)
[![中文](https://img.shields.io/badge/README-%E4%B8%AD%E6%96%87-d14836.svg)](./README_zh.md)

[![arXiv](https://img.shields.io/badge/arXiv-2609.38173-b31b1b.svg)](https://arxiv.org/pdf/2609.38173)
[![Project Page](https://img.shields.io/badge/Project_Page-SimpleICL-2ea44f.svg)](https://simpleicl.github.io/simpleicl/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97_Hugging_Face-Real--Robot_Data_Sample-FFD21E.svg)](https://huggingface.co/datasets/ly152/simpleicl-sample)
[![ModelScope](https://img.shields.io/badge/ModelScope-Real--Robot_Data_Sample-6247eb.svg)](https://www.modelscope.cn/datasets/ly985211/simpleicl-sample)

This repository covers the complete pipeline — SimpleICL policy training, Franka FR3 real-robot deployment, and the paired human-robot data collection recipe behind it:

```
setup → training → real-robot deployment & inference
          ↑
   paired data collection & processing
```

The code is organized into two sub-directories, each with its own environment:

| Sub-directory | Contents | Environment |
| --- | --- | --- |
| `icl_model/` | SimpleICL model, training, cache precomputation, policy server | `pip install -e .` |
| `data_collection/` | collection plan generation, data collection and post-processing, action replay self-check, policy inference client | `requirements.txt` |

---

## Table of Contents

1. [Installation](#1-installation)
2. [Training](#2-training)
3. [Real-Robot Deployment and Inference](#3-real-robot-deployment-and-inference)
4. [Data Collection](#4-data-collection)
5. [Data Processing and Replay](#5-data-processing-and-replay)
6. [End-to-End Smoke Test (Mock)](#6-end-to-end-smoke-test-mock)
7. [Acknowledgements](#7-acknowledgements)

---

## 1. Installation

The two sides can share one conda environment, or use separate ones.

### 1.1 Python environment

```bash
conda create -n simpleicl python=3.10 -y
conda activate simpleicl
pip install -U pip
```

Install the PyTorch build matching your machine first (this repository was
verified on `torch 2.7.1`). For example, CUDA 12.8:

```bash
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 \
  --extra-index-url https://download.pytorch.org/whl/cu128
```

### 1.2 Model-side dependencies

```bash
cd icl_model
pip install -e .
```

This installs the `simpleicl` package in editable mode (dependency list in
`pyproject.toml`).

### 1.3 Data-collection dependencies

```bash
cd data_collection
pip install -r requirements.txt
```

### 1.4 Optional components

```bash
# Collection GUI (data_collection/lerobot/gui.py)
pip install PyQt5

# Action re-timing (TOPP); without it the server streams raw action chunks
pip install toppra scipy
```

Real-robot collection additionally needs three hardware SDKs: `pylibfranka`
(Franka FR3), `pyspacemouse` (3D mouse), and `pyrealsense2` (RealSense
cameras). None of them is required for the mock pipeline (Section 6).

---

## 2. Training

All commands in this section run from `icl_model/`. Two things need to be
prepared before training: the data, and the model weights.

### 2.1 Data

Training data is a paired robot/human LeRobot dataset, which can be
collected with the process described in Sections 4–5.

We also release a sample dataset that can be used for training directly:

> **Hugging Face**: https://huggingface.co/datasets/ly152/simpleicl-sample

```bash
pip install -U "huggingface_hub[cli]"
hf download ly152/simpleicl-sample --repo-type dataset \
    --local-dir ../data_collection/outputs/data_sample
```

The download yields `pair_data.zip`; unzip it in place:

```bash
cd ../data_collection/outputs/data_sample
unzip pair_data.zip
```

After that, the robot side `robot/` and the human side `human/` sit
side by side under `data_sample/`. Fill the robot-side path
`../data_collection/outputs/data_sample/robot` into `data.train.dataset_dirs`
in `configs/task/robot_task.yaml`.

For data collected with Sections 4–5, fill its path into the same field.

### 2.2 Wan2.2 weights

The Wan2.2 snapshot must contain the DiT safetensor shards, Wan2.2_VAE.pth,
models_t5_umt5-xxl-enc-bf16.pth, and the google/umt5-xxl/ directory. Please
obtain these files from
[Wan-AI/Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) and
place them under `icl_model/checkpoints/`:

```
icl_model/checkpoints/
├── Wan-AI/
│   ├── Wan2.2-TI2V-5B/
│   │   ├── diffusion_pytorch_model-0000*-of-00003.safetensors
│   │   ├── Wan2.2_VAE.pth
│   │   └── models_t5_umt5-xxl-enc-bf16.pth
│   └── Wan2.1-T2V-1.3B/
│       └── google/umt5-xxl/
└── DiffSynth-Studio/Wan-Series-Converted-Safetensors/   
```

### 2.3 Start training

```bash
cd icl_model
bash scripts/train_zero1.sh N task=robot_task    # N = number of GPUs
```

That starts training. The training caches (text embeddings and
condition-video latents) are computed automatically on first launch and
automatically rebuilt whenever the datasets or the relevant settings change,
so nothing needs to be run manually.

Output directory: `runs/<task>/<run_id>/`

- `config.yaml`, `dataset_stats.json`
- `checkpoints/weights/step_XXXXXX.pt`
- `checkpoints/state/step_XXXXXX/` (used for `resume`)

Resume training:

```bash
bash scripts/train_zero1.sh N task=robot_task \
  resume=runs/robot_task/<run_id>/checkpoints/state/step_XXXXXX
```

---

## 3. Real-Robot Deployment and Inference

### 3.1 Start the server (icl_model/, model machine)

```bash
DIFFSYNTH_MODEL_BASE_PATH=checkpoints \
python experiments/fastwam_policy/server.py \
    --ckpt runs/robot_task/<run_id>/checkpoints/weights/step_XXXXXX.pt \
    --host 0.0.0.0 --port 5553 \
    --replan-steps 32 --num-inference-steps 10 --gpu 0
```

- `--video-ctx-path <demo.mp4>` sets a startup default condition video;
  otherwise the server waits for the client to send one via F3 recording.

### 3.2 Start the client (data_collection/, robot machine)

```bash
cd data_collection
PYTHONPATH="$(pwd)" python lerobot/scripts/franka_fastwam_client.py \
    --host <server-ip> --port 5553 \
    --task <task> \
    --control-time-s 60
```

Dry run without hardware:

```bash
env -u DISPLAY PYTHONPATH="$(pwd)" python lerobot/scripts/franka_fastwam_client.py \
    --host 127.0.0.1 --port 5553 \
    --task <task> --control-time-s 10 --fps 10 --mock
```

- During startup, **F3** records a high-camera human-hand demonstration as
  the `video_ctx` condition (press F3 again to stop); **F2** uses the
  server's startup default condition.
- **Esc / Ctrl-C** stops execution.

---

## 4. Data Collection

This section and the next describe how paired datasets like the sample one
are produced: plan generation, collection, and post-processing. Commands run
from `data_collection/`.

### 4.1 Generate the collection plan

```bash
python generate_plan.py -g 50 -o plan.csv
```

- `-g N`: N collection groups; each group = 3 tasks, and each task gets one
  `robot_` + one `human_` episode (the teleoperated arm and the human hand
  perform the same task back to back, forming one pair).
- Scene labels alternate as `scene1, scene1_alt, scene2, scene2_alt, ...`:
  every scene is collected under two placements. `sceneN` is the original
  placement; `sceneN_alt` perturbs the task-irrelevant factors — objects are
  moved to new positions, lighting is perturbed, and background color
  clutter is added. With this pairing, the base and alt placements of the
  same task serve as each other's condition references: the policy learns
  *what the task is* from the in-context demonstration while staying robust
  to object positions, lighting, and background — decoupling the task from
  irrelevant scene factors.
- `action` column naming contract: the `robot_` / `human_` prefix decides the
  pairing side; the `taskX-Y` suffix decides who pairs with whom.
- The `data_path` column is filled in automatically by the collection
  program after recording.

### 4.2 Configure hardware

Edit `FrankaRobotConfig` in
`data_collection/lerobot/common/robot_devices/robots/configs.py`:

| Field | Fill in | Notes |
| --- | --- | --- |
| `robot_ips` | IPs of the two Franka arms | currently `TODO_LEFT_ARM_IP` / `TODO_RIGHT_ARM_IP` placeholders |
| `mouse_paths` | SpaceMouse `/dev/hidraw*` nodes | currently `TODO_*` placeholders |
| `cameras[*].serial_number` | serial numbers of the three RealSenses | when `None`, taken in enumeration order |

The default camera resolution is 848×480, matching the training config's
`raw_shape: [3, 480, 848]`. Adjust `reset_joint_positions`,
`replay_joint_*`, etc. for your actual arms.

### 4.3 Collect

Option A: GUI

```bash
python lerobot/gui.py
```

Option B: command line

```bash
PYTHONPATH="$(pwd)" python lerobot/scripts/control_robot.py \
    --robot.type=franka \
    --control.type=record \
    --control.fps=30 \
    --control.repo_id=me/paired_demo \
    --control.root=outputs/date_id \
    --control.single_task=DUMMY \
    --control.warmup_time_s=2 \
    --control.episode_time_s=60 \
    --control.reset_time_s=2 \
    --control.num_episodes=24 \
    --control.video=false \
    --control.push_to_hub=false \
    --control.plan_csv=plan.csv
```

- `--control.root` is conventionally `outputs/<date>`, corresponding to
  `DATE_C` later.
- `--control.single_task` is only a placeholder; it gets overwritten by the
  plan's `action` column during processing.
- Keyboard: **F3** start/continue the next episode; **Left arrow** discard
  the current one; **Esc** stop.

---

## 5. Data Processing and Replay

Run the following commands from `data_collection/`.

### 5.1 Convert + paired split

```bash
DATE_C=date_id bash convert.sh
```

The script runs two steps:

1. `convert/convert_to_video.py`: PNG frames → mp4, written to
   `outputs/videos/<DATE_C>`; episode task names are rewritten from the
   plan's `action` column, and the `scene` column of `plan_filled.csv` is
   filled in.
2. `convert/split_dataset.py`: split by the `robot_` / `human_` prefix and
   pair up, producing:

```
outputs/split_new/<DATE_C>/
├── robot/                    # robot-arm side, keeps all 3 cameras
│   └── meta/
│       ├── conditions.json
│       ├── conditions_list.json
│       ├── scenes.json
│       └── plan_rows.json
├── human/                    # human-hand side, keeps cam_high only
├── pairing.csv
└── split.log
```

`robot/meta/conditions_list.json` lists, for every robot episode, the
reference human videos (paths relative to the split root) used as the
`video_ctx` condition during training; episodes sharing the same base scene
(trailing `_alt` stripped) and the same task share one set of references.
These relative paths keep working after the dataset directory is moved as a
whole.

Optional environment variables:

- `MOCK=1`: relax the checks designed for real data
  (`--no-dup-check --min-frames-robot 1 --min-frames-human 1 --motion-min 0`).
- `DROP_EPS="3,8-9"`: discard the given source episodes (dropped in pairs;
  ranges supported).

### 5.2 Replay (optional)

```bash
DATASET=outputs/date_id EPISODE=0 bash replay.sh          # real-robot replay
EXTRA="--fps 5" DATASET=outputs/date_id EPISODE=1 bash replay.sh
```

Replay logs are saved to `outputs/logfiles/replay-<dataset>-ep<N>.log`.

---

## 6. End-to-End Smoke Test (Mock)

**No hardware required** — walk the full pipeline from scratch: collection →
processing → 1 training step (with eval) → mock inference. Prerequisites:
Section 1 is done, and the Wan weights are in place (Section 2.2).

The smoke test is split into two parts:

| Part | Working directory | Contents |
| --- | --- | --- |
| Part A: data collection | `data_collection/` | plan generation, mock collection, conversion and paired split (6.1–6.2) |
| Part B: model side | `icl_model/` | 1 training step (with eval), server + mock inference (6.3–6.5) |

When Part A runs normally, it produces the paired robot / human datasets
under `data_collection/outputs/split_new/smoke/`, which Part B consumes.

### Part A: Data Collection (data_collection/)

#### 6.1 Generate the plan and collect in mock mode

```bash
cd data_collection

# 3 tasks x (robot + human) x 2 groups x (base, alt) = 24 episodes
python generate_plan.py -g 4 -o plan.csv

# mock collection
env -u DISPLAY PYTHONPATH="$(pwd)" python lerobot/scripts/control_robot.py \
    --robot.type=franka --robot.mock=true \
    --control.type=record --control.fps=30 \
    --control.repo_id=smoke/franka_mock --control.root=outputs/smoke \
    --control.single_task=DUMMY \
    --control.num_episodes=24 \
    --control.episode_time_s=1.0 --control.warmup_time_s=0 --control.reset_time_s=0 \
    --control.video=false --control.push_to_hub=false \
    --control.plan_csv=plan.csv \
    --control.dup_check_every=0
```

`--robot.mock=true` uses the constant-gray-frame mock cameras and the
built-in MockFranka arm (because of this, `dup_check_every=0` is required to
disable the duplicate-first-frame self-check — not needed for real
collection); `env -u DISPLAY` forces headless mode and skips keyboard waits.

#### 6.2 Convert, split, and optionally replay

```bash
DATE_C=smoke MOCK=1 bash convert.sh

# Optional: replay episode 0 (error is always 0 under mock)
env -u DISPLAY DATASET=outputs/smoke EPISODE=0 MOCK=1 bash replay.sh
```

### Part B: Model Side (icl_model/)

#### 6.3 Train 1 step (with eval)

Use the same launch script as Section 2.3, passing the smoke dataset and the
1-step schedule as command-line overrides:

```bash
cd icl_model
RUN_ID=smoke bash scripts/train_zero1.sh 1 task=robot_task \
  'data.train.dataset_dirs=[../data_collection/outputs/split_new/smoke/robot]' \
  data.train.text_embedding_cache_dir=cache/smoke/text_embeds \
  data.train.video_ctx_latent_cache_dir=cache/smoke/video_ctx_latents \
  batch_size=1 num_workers=0 num_epochs=1 max_steps=1 \
  eval_every=1 eval_num_inference_steps=2
```

One eval pass (val loss + inference + side-by-side mp4) runs automatically
after the single training step, written to `runs/robot_task/smoke/eval/`.
Outputs:

- Weights: `runs/robot_task/smoke/checkpoints/weights/step_000001.pt`
- Training state: `runs/robot_task/smoke/checkpoints/state/step_000001/`
- Config and stats: `runs/robot_task/smoke/config.yaml`, `dataset_stats.json`

#### 6.4 Start the policy server (icl_model/, terminal A)

```bash
DIFFSYNTH_MODEL_BASE_PATH=checkpoints \
python experiments/fastwam_policy/server.py \
    --ckpt runs/robot_task/smoke/checkpoints/weights/step_000001.pt \
    --host 127.0.0.1 --port 5553 \
    --replan-steps 8 --num-inference-steps 2 --gpu 0
```

The server locates `config.yaml` and `dataset_stats.json` automatically by
walking up from the checkpoint; the first load of the Wan2.2 components takes
about 1–2 minutes. Visualization outputs are written to
`runs/vis/<exp>_<run>_<step>/`:

- `obs_video/`: observation videos reported by the client (on by default;
  `--no-obs-video` disables, `--obs-video-dir` customizes the location)
- `video_ctx/`: the currently active condition video clip
  (`--no-video-ctx-dump` disables)
- `pred_video/`: predicted rollout videos, enabled with `--vis-video` (the
  server switches to `infer_joint`, decoding one extra video chunk per
  replan)

#### 6.5 Mock client inference (data_collection/, terminal B)

```bash
cd data_collection

# Simulate the client's F3 video_ctx recording: 2 s after startup, F3 is
# simulated to start recording; 5 s later F3 is simulated again to stop,
# producing a 5 s condition video that is sent to the server
env -u DISPLAY PYTHONPATH="$(pwd)" python lerobot/scripts/franka_fastwam_client.py \
    --host 127.0.0.1 --port 5553 \
    --task "task1-1" \
    --control-time-s 10 --fps 10 --mock \
    --simulate-video-ctx 5
```

`--simulate-video-ctx` mimics the real-robot flow: press **F3** to record a
human-hand demonstration, press **F3** again to stop and send it.

---

## 7. Acknowledgements

This repository used the following open-source projects during development.
We thank them for their work:

- **[FastWAM](https://github.com/yuantianyuan01/FastWAM)**: architectural
  reference for the model side. Its Wan2.2-TI2V-5B video backbone, ActionDiT
  action expert, and MoT mixed-attention design underpin the policy model in
  this repository.
- **[LeRobot](https://github.com/huggingface/lerobot)** (Hugging Face): the
  data-format and collection-pipeline code framework.
- **[Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)** (Wan-AI):
  the video foundation model.

We sincerely thank the authors and communities of these projects; their
excellent work has helped us greatly.
