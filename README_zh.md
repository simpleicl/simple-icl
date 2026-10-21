# SimpleICL

论文 **《In-context Robot Learning Made Simple: A Democratized Recipe for Manipulation Tasks》** 的官方代码仓库。

[![English](https://img.shields.io/badge/README-English-111111.svg)](./README.md)
[![中文](https://img.shields.io/badge/README-%E4%B8%AD%E6%96%87-d14836.svg)](./README_zh.md)

[![arXiv](https://img.shields.io/badge/arXiv-2609.38173-b31b1b.svg)](https://arxiv.org/pdf/2609.38173)
[![Project Page](https://img.shields.io/badge/Project_Page-SimpleICL-2ea44f.svg)](https://simpleicl.github.io/simpleicl/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97_Hugging_Face-%E7%9C%9F%E6%9C%BA%E6%95%B0%E6%8D%AE%E6%A0%B7%E4%BE%8B-FFD21E.svg)](https://huggingface.co/datasets/ly152/simpleicl-sample)
[![ModelScope](https://img.shields.io/badge/ModelScope-%E7%9C%9F%E6%9C%BA%E6%95%B0%E6%8D%AE%E6%A0%B7%E4%BE%8B-6247eb.svg)](https://www.modelscope.cn/datasets/ly985211/simpleicl-sample)

本仓库涵盖完整流程——SimpleICL 策略训练与 Franka FR3 真机部署，以及背后的人-机配对数据采集方案：

```
环境配置 → 训练 → 真机部署与推理
           ↑
    配对数据采集与处理
```

代码按用途分为两个子目录，各自独立配置环境：

| 子目录 | 内容 | 环境 |
| --- | --- | --- |
| `icl_model/` | SimpleICL 模型、训练、缓存预计算、策略 server | `pip install -e .` |
| `data_collection/` | 采集计划表生成、数据采集与后处理、动作回放自检、策略推理客户端 | `requirements.txt` |

---

## 目录

1. [安装](#1-安装)
2. [训练](#2-训练)
3. [真机部署与推理](#3-真机部署与推理)
4. [数据采集](#4-数据采集)
5. [数据处理与回放](#5-数据处理与回放)
6. [全链路冒烟测试（mock）](#6-全链路冒烟测试mock)
7. [致谢](#7-致谢)

---

## 1. 安装

两侧可以共用一个 conda 环境，也可以分开建。

### 1.1 Python 环境

```bash
conda create -n simpleicl python=3.10 -y
conda activate simpleicl
pip install -U pip
```

先安装与本机匹配的 PyTorch（本仓库在 `torch 2.7.1` 上验证）。例如
CUDA 12.8：

```bash
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 \
  --extra-index-url https://download.pytorch.org/whl/cu128
```

### 1.2 模型侧依赖

```bash
cd icl_model
pip install -e .
```

这会把 `simpleicl` 包以 editable 方式安装（依赖清单见 `pyproject.toml`）。

### 1.3 采集侧依赖

```bash
cd data_collection
pip install -r requirements.txt
```

### 1.4 可选组件

```bash
# 采集 GUI（data_collection/lerobot/gui.py）
pip install PyQt5

# 动作重定时（TOPP）；未安装时 server 直接下发原始动作块
pip install toppra scipy
```

真机采集额外需要三个硬件 SDK：`pylibfranka`（Franka FR3）、`pyspacemouse`
（三维鼠标）、`pyrealsense2`（RealSense 相机）。只跑 mock 流程（第 6 节）不需要。

---

## 2. 训练

以下命令均在 `icl_model/` 下执行。开始训练前需要准备两件事：数据和模型权重。

### 2.1 数据

训练数据为配对的 robot/human LeRobot 数据集，可根据第 4–5 节所述流程采集得到。

同时，我们发布了一份数据集样例，可以直接用于训练：

> **Hugging Face**: https://huggingface.co/datasets/ly152/simpleicl-sample

```bash
pip install -U "huggingface_hub[cli]"
hf download ly152/simpleicl-sample --repo-type dataset \
    --local-dir ../data_collection/outputs/data_sample
```

下载得到 `pair_data.zip`，就地解压：

```bash
cd ../data_collection/outputs/data_sample
unzip pair_data.zip
```

解压完成后，`data_sample/` 下机械臂侧 `robot/` 与人手侧 `human/` 并列。把
robot 侧路径 `../data_collection/outputs/data_sample/robot` 填到
`configs/task/robot_task.yaml` 的 `data.train.dataset_dirs`。

用第 4–5 节方法自行采集数据训练时，同样将数据路径填入该字段即可。

### 2.2 Wan2.2 权重

Wan2.2 快照需包含 DiT safetensor 分片、Wan2.2_VAE.pth、
models_t5_umt5-xxl-enc-bf16.pth 和 google/umt5-xxl/ 目录。请从
[Wan-AI/Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) 获取，
并放到 `icl_model/checkpoints/` 下：

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

### 2.3 启动训练

```bash
cd icl_model
bash scripts/train_zero1.sh N task=robot_task    # N = GPU 数
```

即可启动训练。训练所需的缓存（文本 embedding 与条件视频 latent）首次启动时自动
计算，数据集或相关设置变化后自动重建，无需手动运行。

输出目录：`runs/<task>/<run_id>/`

- `config.yaml`、`dataset_stats.json`
- `checkpoints/weights/step_XXXXXX.pt`
- `checkpoints/state/step_XXXXXX/`（用于 `resume`）

恢复训练：

```bash
bash scripts/train_zero1.sh N task=robot_task \
  resume=runs/robot_task/<run_id>/checkpoints/state/step_XXXXXX
```

---

## 3. 真机部署与推理

### 3.1 启动 server（icl_model/，模型机器）

```bash
DIFFSYNTH_MODEL_BASE_PATH=checkpoints \
python experiments/fastwam_policy/server.py \
    --ckpt runs/robot_task/<run_id>/checkpoints/weights/step_XXXXXX.pt \
    --host 0.0.0.0 --port 5553 \
    --replan-steps 32 --num-inference-steps 10 --gpu 0
```

- `--video-ctx-path <demo.mp4>` 可设置启动默认条件视频；否则等待客户端通过
  F3 录制下发。

### 3.2 启动客户端（data_collection/，机器人机器）

```bash
cd data_collection
PYTHONPATH="$(pwd)" python lerobot/scripts/franka_fastwam_client.py \
    --host <server-ip> --port 5553 \
    --task <task> \
    --control-time-s 60
```

无硬件试跑：

```bash
env -u DISPLAY PYTHONPATH="$(pwd)" python lerobot/scripts/franka_fastwam_client.py \
    --host 127.0.0.1 --port 5553 \
    --task <task> --control-time-s 10 --fps 10 --mock
```

- 启动阶段 **F3** 录制高位相机人手演示作为 `video_ctx` 条件（再按 F3 停止）；
  **F2** 使用 server 启动时的默认条件。
- 执行中 **Esc / Ctrl-C** 停止。

---

## 4. 数据采集

本节与下一节介绍样例数据集这类配对数据是如何生产的：计划表生成、采集与
后处理。以下命令在 `data_collection/` 下执行。

### 4.1 生成采集计划表

```bash
python generate_plan.py -g 50 -o plan.csv
```

- `-g N`：N 个采集组；每组 = 3 个任务，每个任务一条 `robot_` + 一条 `human_`
  （遥操作机械臂与人手先后完成同一任务，构成一对）。
- 组的场景标签按 `scene1, scene1_alt, scene2, scene2_alt, ...` 交替：同一场景
  在两种摆位下各采一组。`sceneN` 为原始摆位；`sceneN_alt` 在任务保持不变的
  前提下变换与任务无关的因素——移动物体位置、扰动光照、增加背景颜色干扰等。
  经此配对，同一任务的 base 与 alt 摆位互为条件参考，策略只从 in-context 演示
  中学习"任务是什么"，而对物体摆放位置、光照、背景保持鲁棒——把任务与无关
  场景因素解耦。
- `action` 列命名契约：`robot_` / `human_` 前缀决定配对归属，后缀 `taskX-Y`
  决定谁与谁配对。
- `data_path` 列由采集程序录制后自动回填。

### 4.2 配置硬件

编辑 `data_collection/lerobot/common/robot_devices/robots/configs.py` 中的
`FrankaRobotConfig`：

| 字段 | 需要填写 | 说明 |
| --- | --- | --- |
| `robot_ips` | 两台 Franka 的 IP | 当前为 `TODO_LEFT_ARM_IP` / `TODO_RIGHT_ARM_IP` 占位 |
| `mouse_paths` | SpaceMouse 的 `/dev/hidraw*` 节点 | 当前为 `TODO_*` 占位 |
| `cameras[*].serial_number` | 三路 RealSense 序列号 | `None` 时按枚举顺序取用 |

相机分辨率默认 848×480，与训练配置的 `raw_shape: [3, 480, 848]` 一致。
`reset_joint_positions`、`replay_joint_*` 等参数请按实际机械臂调整。

### 4.3 采集

方式 A：GUI

```bash
python lerobot/gui.py
```

方式 B：命令行

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

- `--control.root` 约定为 `outputs/<日期>`，后续 `DATE_C` 与之对应。
- `--control.single_task` 只是占位，处理阶段会被 `plan.csv` 的 `action` 列覆盖。
- 键盘：**F3** 开始/继续下一条；**左方向键** 丢弃当前条；
  **Esc** 停止。

---

## 5. 数据处理与回放

以下命令在 `data_collection/` 下执行。

### 5.1 转换 + 配对拆分

```bash
DATE_C=date_id bash convert.sh
```

脚本执行两步：

1. `convert/convert_to_video.py`：PNG 帧 → mp4，写到 `outputs/videos/<DATE_C>`；
   同时用 plan 的 `action` 列改写 episode 任务名，并补全 `plan_filled.csv` 的
   `scene` 列。
2. `convert/split_dataset.py`：按 `robot_` / `human_` 前缀拆分并配对，输出：

```
outputs/split_new/<DATE_C>/
├── robot/                    # 机械臂侧，保留 3 路相机
│   └── meta/
│       ├── conditions.json
│       ├── conditions_list.json
│       ├── scenes.json
│       └── plan_rows.json
├── human/                    # 人手侧，只保留 cam_high
├── pairing.csv
└── split.log
```

`robot/meta/conditions_list.json` 为每条 robot episode 列出可参考的 human 视频
（相对 split 根目录的路径），训练时作为 `video_ctx` 条件；同一 base 场景
（去掉尾部 `_alt`）且同一任务的 episode 共享一组参考。数据集目录整体搬移后
这些相对路径仍然有效。

可选环境变量：

- `MOCK=1`：放宽为 mock 数据设计的检查
  （`--no-dup-check --min-frames-robot 1 --min-frames-human 1 --motion-min 0`）。
- `DROP_EPS="3,8-9"`：丢弃指定源 episode（成对丢弃，支持区间）。

### 5.2 回放（可选）

```bash
DATASET=outputs/date_id EPISODE=0 bash replay.sh          # 真机回放
EXTRA="--fps 5" DATASET=outputs/date_id EPISODE=1 bash replay.sh
```

回放日志保存到 `outputs/logfiles/replay-<dataset>-ep<N>.log`。

---

## 6. 全链路冒烟测试（mock）

**不需要任何硬件**，从零走完采集 → 处理 → 训练 1 步（含 eval）→ mock 推理。
前提：完成第 1 节安装，并按 2.2 节放好 Wan 权重。

冒烟流程分为两部分：

| 部分 | 工作目录 | 内容 |
| --- | --- | --- |
| Part A：数据采集侧 | `data_collection/` | 计划表生成、mock 采集、转换与配对拆分（6.1–6.2） |
| Part B：模型侧 | `icl_model/` | 训练 1 步（含 eval）、server + mock 推理（6.3–6.5） |

Part A 正常运行后，会在 `data_collection/outputs/split_new/smoke/` 生成 robot /
human 两个配对数据集，供 Part B 使用。

### Part A：数据采集侧（data_collection/）

#### 6.1 生成计划表并 mock 采集

```bash
cd data_collection

# 3 个任务 x (robot + human) x 2组 x (base, alt) = 24 条 episode
python generate_plan.py -g 4 -o plan.csv

# mock 采集
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

`--robot.mock=true` 使用恒定灰度帧的 mock 相机与内置 MockFranka机械臂（为此，必须用 `dup_check_every=0`
关闭重复首帧自检。实际采集时不需这么做。）；`env -u DISPLAY`
强制 headless，自动跳过键盘等待。

#### 6.2 转换、拆分与可选回放

```bash
DATE_C=smoke MOCK=1 bash convert.sh

# 可选：回放第 0 条（mock 下误差恒为 0）
env -u DISPLAY DATASET=outputs/smoke EPISODE=0 MOCK=1 bash replay.sh
```

### Part B：模型侧（icl_model/）

#### 6.3 训练 1 步（含 eval）

用与 2.3 节相同的启动脚本，通过命令行复写指定 smoke 数据集与 1 步训练计划：

```bash
cd icl_model
RUN_ID=smoke bash scripts/train_zero1.sh 1 task=robot_task \
  'data.train.dataset_dirs=[../data_collection/outputs/split_new/smoke/robot]' \
  data.train.text_embedding_cache_dir=cache/smoke/text_embeds \
  data.train.video_ctx_latent_cache_dir=cache/smoke/video_ctx_latents \
  batch_size=1 num_workers=0 num_epochs=1 max_steps=1 \
  eval_every=1 eval_num_inference_steps=2
```

训练 1 步后自动执行一次 eval（val loss + 推理 + 拼接对比 mp4，写入
`runs/robot_task/smoke/eval/`）。产物：

- 权重：`runs/robot_task/smoke/checkpoints/weights/step_000001.pt`
- 训练状态：`runs/robot_task/smoke/checkpoints/state/step_000001/`
- 配置与统计：`runs/robot_task/smoke/config.yaml`、`dataset_stats.json`

#### 6.4 启动策略 server（icl_model/，终端 A）

```bash
DIFFSYNTH_MODEL_BASE_PATH=checkpoints \
python experiments/fastwam_policy/server.py \
    --ckpt runs/robot_task/smoke/checkpoints/weights/step_000001.pt \
    --host 127.0.0.1 --port 5553 \
    --replan-steps 8 --num-inference-steps 2 --gpu 0
```

server 会自动从 checkpoint 向上定位 `config.yaml` 与 `dataset_stats.json`；首次加载
Wan2.2 组件约 1–2 分钟。可视化输出统一写入
`runs/vis/<exp>_<run>_<step>/`：

- `obs_video/`：客户端上报的观测视频（默认开启，`--no-obs-video` 关闭，
  `--obs-video-dir` 自定义位置）
- `video_ctx/`：当前生效的条件视频 clip（`--no-video-ctx-dump` 关闭）
- `pred_video/`：预测 rollout 视频，加 `--vis-video` 开启（server 改用
  `infer_joint`，每个 replan 额外解码一个视频 chunk）

#### 6.5 mock 客户端推理（data_collection/，终端 B）

```bash
cd data_collection

# 模拟客户端 F3 录制 video_ctx：启动 2 s 后模拟按下 F3 开始录制，
# 5 s 后再次模拟按下 F3 停止，得到一段 5 s 的条件视频并发送给 server
env -u DISPLAY PYTHONPATH="$(pwd)" python lerobot/scripts/franka_fastwam_client.py \
    --host 127.0.0.1 --port 5553 \
    --task "task1-1" \
    --control-time-s 10 --fps 10 --mock \
    --simulate-video-ctx 5
```

`--simulate-video-ctx` 模拟真机推理时的流程：按 **F3**
录制人手演示、再按 **F3** 停止并发送。

---

## 7. 致谢

本仓库在开发中使用了以下开源项目，感谢他们的工作：

- **[FastWAM](https://github.com/yuantianyuan01/FastWAM)**：模型侧的架构参考。
  其 Wan2.2-TI2V-5B 视频骨干、ActionDiT 动作专家与 MoT 混合注意力设计为本仓库
  的策略模型提供了基础。
- **[LeRobot](https://github.com/huggingface/lerobot)**（Hugging Face）：数据格式
  与采集流程的代码框架。
- **[Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)**（Wan-AI）：
  视频基础模型。

我们衷心感谢这些项目的作者和社区。您们卓越的工作为我们提供了极大的帮助。
