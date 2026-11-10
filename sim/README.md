# SimpleICL-Sim

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97_Hugging_Face-Pretrained_models-FFD21E.svg)](https://huggingface.co/SimpleICL/SimpleICL)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97_Hugging_Face-ICL--Oriented_Training_Set-FFD21E.svg)](https://huggingface.co/datasets/SimpleICL/ICL-oriented-Training-Set)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97_Hugging_Face-Semantic_Discrimination_Test_Set-FFD21E.svg)](https://huggingface.co/datasets/SimpleICL/Semantic-Discrimination-Test-Set)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97_Hugging_Face-Out_Of_Distribution_Test_Set-FFD21E.svg)](https://huggingface.co/datasets/SimpleICL/Out-Of-Distribution-Test-Set)

Simulation data collection, training, and fixed-scene evaluation for three SimpleICL robot policies. Model, dataset, training, preprocessing, and evaluation code live in `simpleicl` package. The integrated simulator lives in `robotwin`.

## Installation

Use a dedicated Python 3.10 environment, separate from `real/`. Both sides
install a package named `simpleicl` and must never share an environment.

Before installing, make Git, a C++ compiler, CUDA Toolkit (`nvcc`), FFmpeg, Vulkan tools (`vulkaninfo`), and the NVIDIA driver available on the host. The installer checks these prerequisites; 

```bash
# From the unified repository root:
source "$(conda info --base)/etc/profile.d/conda.sh" &&
conda create -n simpleicl-sim python=3.10.12 -y &&
conda activate simpleicl-sim &&
cd sim &&
python scripts/install.py --stage all
```

All commands below run from `sim/` with this environment active. Assets,
datasets, and checkpoints are installed separately as described below.

## Resources

Code installation does not install assets, benchmark scenes, datasets, or model weights. Install or symlink them into this layout:

```text
assets/                         # RoboTwin assets: embodiments, objects, backgrounds, ...
checkpoints/
  Wan2.2-TI2V-5B/                # complete Wan-AI/Wan2.2-TI2V-5B snapshot
  action_init.pt                # generated below; training only
  action_only/{model.pt,dataset_stats.json}
  shared_ctx/{model.pt,dataset_stats.json}
  text_only/{model.pt,dataset_stats.json}
data/
  train_lerobot/                # converted lerobot training set
  eval_native/                  # Semantic Discrimination Test Set
  eval_ood/                     # Out Of Distribution Test Set
outputs/                        # collection, caches, training and evaluation results
```

The Wan snapshot must contain the DiT safetensor shards, `Wan2.2_VAE.pth`, `models_t5_umt5-xxl-enc-bf16.pth`, and `google/umt5-xxl/`. Obtain it from [Wan-AI/Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B). Assets follow the [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) layout.

Each model directory contains its weights and the normalization statistics from its training dataset. Evaluation composes the corresponding `configs/task/<model>.yaml`.

## Collect simulation data

Collection uses one pipeline for all task lists: an ALOHA demonstration at the original placement and successful ALOHA subjects with object pose perturbations. 

Each retained placement requires the demonstration and at least the configured number of successful subject variants. 

Choose the task pool with `--tasks-file` when generating the plan, then review the plan before collection. Plans fix tasks, scene seeds, search quotas, and visual variants.

```bash
# Generate a production plan, review it, then collect it.
python scripts/collect.py --groups 20 --seed 42 --generate-only --output outputs/collection/train
python scripts/collect.py --plan outputs/collection/train/plan.yaml --gpus 0,1 --output outputs/collection/train
# Use another task list with the same generation and collection settings.
python scripts/collect.py --tasks-file src/robotwin/task_config/eval_ood_list.yaml --groups 20 --seed 42 --generate-only --output outputs/collection/ood
# Review each generated plan, then collect it with --plan as above.
```

The default task settings are `src/robotwin/task_config/plan_gen_list.yaml`. 

YAML settings use `tasks` for all scene object providers and `required_tasks` for tasks to collect (defaults to `tasks`). Both collection and background tasks may provide scene objects. 

OOD settings include the seven OOD tasks plus the training task pool in `tasks`, with only the seven OOD tasks in `required_tasks`. 

Training settings use the training pool for both fields. `--tasks-file path/to/settings.yaml` replaces these settings. Alternatively, use `--tasks task_a,task_b`; the two options are mutually exclusive. 

All task lists use the same grouping, quota, perturbation, and visual-variant defaults. 

Generated groups record `tasks` for scene construction and `required_tasks` for collection. Every retained placement must succeed on all required tasks; `exclude_tasks` excludes every other task, including incidental static matches. Quotas count required tasks only; inspect its groups before collecting production data. 

Each GPU in `--gpus` independently collects one scene group, then takes the next pending group. Within each group, the source RoboTwin worker launcher handles ALOHA search, online quotas, and cached trajectory recording on that GPU. 

Raw scenes are written under `<output>/scenes/<scene_config>/<group>`, with completion records and separate logs for each group.

## Prepare training data

The converter selects perturbed ALOHA subjects where present, otherwise baseline ALOHA episodes. 

It limits subjects round-robin across placements and perturbations and uses the unperturbed ALOHA demonstration video. 

Actions are the next frame's joint state, with synchronized three-camera videos and instructions.

```bash
# Convert raw data to lerobot format
python scripts/prepare_data.py convert -- \
  --input data/raw_train --output data/train_lerobot \
  --workers 8 --seed 42
# Resume conversion after interruption:
python scripts/prepare_data.py convert -- \
  --input data/raw_train --output data/train_lerobot \
  --workers 8 --seed 42 --resume
# Generate action expert
python scripts/prepare_data.py action-init -- --dtype bfloat16
# Cache text instruction embedding
CUDA_VISIBLE_DEVICES=0 python scripts/prepare_data.py text --model action_only
CUDA_VISIBLE_DEVICES=0 python scripts/prepare_data.py text --model text_only
```

Action initialization is shared by all three models. Training needs embeddings for the dataset's task instructions. The converter reports skipped episodes and writes `meta/conversion_summary.json`. Check that report before training. `--max-episodes N` limits episodes **per task**, and `--video-codec libx264` selects H.264 instead of the default AV1 codec.

## Train

| Model alias | Conditioning | Training recipe |
| --- | --- | --- |
| `action_only` | Demonstration video and task instruction | [action_only.yaml](src/simpleicl/configs/task/action_only.yaml) |
| `shared_ctx` | Shared video/action context, demonstration video and task instruction | [shared_ctx.yaml](src/simpleicl/configs/task/shared_ctx.yaml) |
| `text_only` | Task instruction | [text_only.yaml](src/simpleicl/configs/task/text_only.yaml) |

```bash
python scripts/train.py --model action_only --gpus 0,1,2,3,4,5,6,7
python scripts/train.py --model shared_ctx --gpus 0,1,2,3,4,5,6,7
python scripts/train.py --model text_only --gpus 0,1,2,3,4,5,6,7
# Explicit paths and Hydra overrides:
python scripts/train.py --model shared_ctx --gpus 0 --dataset data/train_lerobot \
  --output outputs/train/action_only -- batch_size=1 num_workers=0
# Resume from an Accelerate trainer-state directory, not a weights-only .pt:
python scripts/train.py --model shared_ctx --gpus 0 \
  --resume outputs/train/action_only/checkpoints/state/step_002500
```

The three task recipes define the model architecture and training settings. Demonstration sampling defaults to a fixed request of 16 frames (17 after VAE alignment), with percentage sampling and the frame cap disabled. (We recommend to try enabling these features)

## Evaluate the benchmarks

```bash
python scripts/evaluate.py --model shared_ctx --benchmark native --gpus 0,1,2,3,4,5,6,7
python scripts/evaluate.py --model shared_ctx --benchmark ood --gpus 0,1,2,3,4,5,6,7
# The other models use the same benchmark interfaces:
python scripts/evaluate.py --model text_only --benchmark native --gpus 0,1,2,3,4,5,6,7
python scripts/evaluate.py --model action_only --benchmark ood --gpus 0,1,2,3,4,5,6,7
# Explicit scene and checkpoint directories (no benchmark or task list required):
python scripts/evaluate.py --model shared_ctx \
  --scene-dir data/eval_ood \
  --checkpoint checkpoints/shared_ctx/model.pt \
  --stats checkpoints/shared_ctx/dataset_stats.json \
  --gpus 0,1,2,3,4,5,6,7
# Optional arguments: cap episodes per task or select a task subset
  --episodes-per-task 10
  --task-list src/robotwin/task_config/eval_ood_list.yaml
```

Evaluation uses the training sampling settings directly; there is no separate candidate evaluation profile. 