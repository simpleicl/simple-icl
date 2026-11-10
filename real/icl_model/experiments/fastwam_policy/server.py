#!/usr/bin/env python3
"""FastWAM policy server for the Franka FR3 arm.

Model-side half of the two-process deployment. The robot-side client
(``lerobot/scripts/franka_fastwam_client.py``) speaks to this server over TCP::

    Franka arm (this repo)          <--TCP-->   FastWAM model (this repo)
      franka_fastwam_client.py                     server.py

This server instantiates a trained FastWAM (video+action) checkpoint and
answers the client's command interface:

    should_request_observation -> {"result": bool}
        True iff the next ``step`` must carry a fresh observation so we can
        re-infer / "replan": normally that means the internal action queue is
        drained, or -- under ``--async-infer`` -- that it is close enough to
        draining to start the next chunk while this one is still being handed out.
    configure {fps, max_vel, max_acc, gripper_dims, topp} -> {"topp": bool, "detail": str}
        The client reports its control period and the robot's per-joint velocity
        / acceleration limits, which enables TOPP re-timing of the queued chunks
        (see ``chunk_topp.py``). Optional: without it the raw chunk is queued
        verbatim, as it was before. The limits are the robot's, so they have to
        come from the client -- this server serves a Franka arm.
    step {instruction, state?, images?} -> {"action": ndarray | None}
        * On a replan step (queue empty) the client ships state+images; we build
          the mosaic input, denormalize the predicted chunk, re-time it (if
          configured) and refill the queue with ``replan_steps`` raw-qpos actions.
        * Otherwise we simply pop the next queued action.
    get_video_ctx_info -> {"has_video_ctx": bool, "num_frames": int, "size": [h,w]}
        Report the video_ctx geometry so the client can downsample its recorded
        demo to exactly ``num_frames`` before shipping it.
    set_video_ctx {frames?} -> {"status": "ok", "has_video_ctx": bool}
        Swap the video_ctx global condition (only if the checkpoint was trained
        with has_video_ctx_condition=True). ``frames`` is a [T,H,W,3] uint8 RGB
        array recorded by the client's F3 teleop demo; omit/None to restore the
        startup default (--video-ctx-path), which is what the client's F2 sends.
        The latents are cached and only (re)built right before inference when the
        source changed.
    reset  -> {"status": "ok"}         (clear the queue, bump episode)
    shutdown -> {"status": "ok"}       (close the connection)

Replanning: synchronous by default, ``--async-infer`` to overlap
-----------------------------------------------------------------
By default a replan is synchronous: the queue drains completely, the client ships
an observation and then *blocks* for the whole diffusion sample while the arm
holds its last target. At 30 Hz with ``--replan-steps 8`` the queue is 267 ms of
motion, so an inference of even half a second makes the arm move in bursts
separated by longer stalls, and that stop-go cadence reads as jitter no matter
how well the chunk itself is smoothed -- it is not something ``chunk_topp`` can
fix, because the command stream is *correct*, it just has holes in it.

``--async-infer`` removes the holes. ``should_request_observation`` goes true
while the queue still has runway, ``step`` hands the observation to a worker
thread and returns a queued action immediately, and the client keeps ticking at
full rate throughout. When the chunk lands, the rows describing ticks that have
meanwhile been executed are dropped (which is what aligns it with the queue) and
the rest is cross-faded in over ``--async-blend-steps`` -- a hard swap would
reintroduce a seam, since the two chunks disagree by however much the newer
observation changed the plan. See ``_merge_chunk``.

The one hard requirement is that a chunk must arrive before the queue it has to
cover runs out, i.e. inference must fit inside ``action_horizon`` control ticks
(32 ticks = 1.07 s at 30 Hz). Past that the server logs the shortfall in ticks
and the arm stalls again; lower ``--num-inference-steps`` or raise the horizon.

Preprocessing mirrors the training pipeline (``RobotVideoDataset._get`` /
``build_mosaic_from_images``): each camera frame is resized to its
``shape_meta.images[i].shape``, tiled into the ``concat_multi_camera`` mosaic
(top ``cam_high`` over ``cam_left_wrist``|``cam_right_wrist``), resized/
cropped to ``video_size`` and normalized to [-1, 1]. State is normalized and fed
back as ``proprio``; the predicted action chunk is denormalized to raw qpos.

Wire protocol: length-prefixed pickle (kept in sync with franka_fastwam_client.py).

All visualization output is written under one ckpt-derived root,
``<repo>/runs/vis/<exp_name>_<date_dir>_<step>/`` (see ``vis_root_from_ckpt``):
``pred_video/`` (--vis-video), ``obs_video/`` (client observations) and
``video_ctx/`` (video_ctx condition clips). The --*-dir flags override each.

Example
-------
DIFFSYNTH_MODEL_BASE_PATH=checkpoints python experiments/fastwam_policy/server.py \
    --ckpt runs/robot_task/<run_id>/checkpoints/weights/step_XXXXXX.pt \
    --host 0.0.0.0 --port 5553 \
    --replan-steps 32 --num-inference-steps 10 --gpu 0
"""

import argparse
import io
import logging
import math
import os
import pickle
import socket
import struct
import sys
import threading
import time
import traceback
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf

# --- import path: this repo root, src, and this package's own directory ------
project_root = Path(__file__).resolve().parents[2]
here = Path(__file__).resolve().parent
for _p in (project_root, project_root / "src", here):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("server")

from model_loading import (  # noqa: E402
    _MOSAIC_SIZES,
    denormalize_action,
    find_config_from_ckpt,
    find_dataset_stats_from_ckpt,
    load_model,
    normalize_state,
    prepare_video_ctx_latents,
)
from simpleicl.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT  # noqa: E402
from simpleicl.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json  # noqa: E402

from chunk_topp import TOPP_AVAILABLE, TOPP_IMPORT_ERROR, ChunkRetimer  # noqa: E402


# ---------------------------------------------------------------------------
# TCP helpers (length-prefixed pickle) -- kept in sync with the client.
# ---------------------------------------------------------------------------
def _send_msg(sock: socket.socket, obj: Any) -> None:
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(struct.pack(">I", len(data)) + data)


def _recv_exact(sock: socket.socket, n: int) -> Optional[bytes]:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


class _NumpyCompatUnpickler(pickle.Unpickler):
    """Load pickles across the numpy 1.x / 2.x boundary.

    numpy 2 moved its internals from ``numpy.core`` to ``numpy._core``, and the
    array reducers it pickles name the *new* path -- so a numpy-1 process fails
    to load a numpy-2 array with ``No module named 'numpy._core.numeric'``.
    Robot and model run on separate machines with separate envs, so retry the
    other spelling instead of forcing both to pin the same numpy major.
    """

    _ALIASES = (("numpy._core", "numpy.core"), ("numpy.core", "numpy._core"))

    def find_class(self, module: str, name: str) -> Any:
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError):
            for old, new in self._ALIASES:
                if module == old or module.startswith(old + "."):
                    return super().find_class(new + module[len(old):], name)
            raise


def _loads(payload: bytes) -> Any:
    return _NumpyCompatUnpickler(io.BytesIO(payload)).load()


def _recv_msg(sock: socket.socket) -> Any:
    header = _recv_exact(sock, 4)
    if header is None:
        return None
    length = struct.unpack(">I", header)[0]
    payload = _recv_exact(sock, length)
    if payload is None:
        return None
    return _loads(payload)


# ---------------------------------------------------------------------------
# Mosaic construction from in-memory camera frames.
# ---------------------------------------------------------------------------
def build_mosaic_from_images(
    images: Dict[str, np.ndarray],
    cam_keys: List[str],
    proc_sizes: List[List[int]],
    video_size: List[int],
    concat_mode: str,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Tile the client's camera frames the training (mosaic) way.

    Mirrors the training mosaic construction but consumes HWC uint8 RGB ndarrays
    (as ``capture_observation`` produces them) instead of video files. Returns a
    ``[1, 3, H, W]`` tensor in [-1, 1].
    """
    import torchvision.transforms.functional as TF
    from simpleicl.datasets.dataset_utils import (
        ResizeSmallestSideAspectPreserving,
        CenterCrop,
    )

    if concat_mode not in _MOSAIC_SIZES:
        raise ValueError(
            f"Unsupported mosaic concat mode '{concat_mode}'; expected one of "
            f"{list(_MOSAIC_SIZES)}."
        )
    if len(cam_keys) != 3:
        raise ValueError(f"mosaic requires exactly 3 camera keys, got {cam_keys}")
    missing = [k for k in cam_keys if k not in images]
    if missing:
        raise KeyError(
            f"observation is missing camera(s) {missing}; got {sorted(images)}"
        )

    sizes = _MOSAIC_SIZES[concat_mode]
    bl = TF.InterpolationMode.BILINEAR

    per_cam = []
    for i, key in enumerate(cam_keys):
        arr = np.ascontiguousarray(images[key])
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"camera '{key}' frame must be HWC RGB, got {arr.shape}")
        x = torch.from_numpy(arr).permute(2, 0, 1).contiguous().float() / 255.0
        # Processor val_transform: torchvision.transforms.Resize(shape_meta shape).
        x = TF.resize(x, size=list(proc_sizes[i]), interpolation=bl, antialias=True)
        per_cam.append(x)

    cam_top = TF.resize(per_cam[0], size=sizes["top"], interpolation=bl, antialias=True)
    cam_left = TF.resize(per_cam[1], size=sizes["wrist"], interpolation=bl, antialias=True)
    cam_right = TF.resize(per_cam[2], size=sizes["wrist"], interpolation=bl, antialias=True)
    bottom = torch.cat([cam_left, cam_right], dim=-1)   # [3, wrist_h, W]
    frame = torch.cat([cam_top, bottom], dim=-2)        # [3, 384, W]

    resize_t = ResizeSmallestSideAspectPreserving(
        args={"img_w": int(video_size[1]), "img_h": int(video_size[0])}
    )
    crop_t = CenterCrop(args={"img_w": int(video_size[1]), "img_h": int(video_size[0])})
    frame = resize_t(frame)
    frame = crop_t(frame)

    frame = frame * 2.0 - 1.0  # [-1, 1]
    return frame.unsqueeze(0).to(device=device, dtype=dtype)  # [1, 3, H, W]


# ---------------------------------------------------------------------------
# Observation recording: client camera frames -> one 2x2-grid frame.
# ---------------------------------------------------------------------------
def build_quad_grid(
    images: Dict[str, np.ndarray],
    order: List[str],
    tile_hw: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Tile up to 4 camera frames into a single 2x2 grid uint8 RGB frame.

    ``order`` fixes the slot assignment -- top-left, top-right, bottom-left,
    bottom-right -- so the layout stays stable across steps even if the client's
    dict ordering doesn't. Slots without a frame are filled black; the Franka
    setup ships 3 cameras, so the bottom-right tile is normally empty. ``tile_hw`` defaults to
    half the first available frame, making the grid roughly one camera's
    resolution; pass it explicitly to keep a video's frame size constant.
    """
    from PIL import Image

    frames = {k: np.ascontiguousarray(v) for k, v in images.items() if v is not None}
    slots = (list(order) + [""] * 4)[:4]
    ref = next((frames[k] for k in slots if k in frames), None)
    if ref is None:
        raise ValueError(f"no usable camera frame in {sorted(images)} for slots {order}")
    if tile_hw is None:
        tile_h, tile_w = max(2, ref.shape[0] // 2), max(2, ref.shape[1] // 2)
    else:
        tile_h, tile_w = int(tile_hw[0]), int(tile_hw[1])

    tiles = []
    for key in slots:
        arr = frames.get(key)
        if arr is None:
            tiles.append(np.zeros((tile_h, tile_w, 3), dtype=np.uint8))
            continue
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"camera '{key}' frame must be HWC RGB, got {arr.shape}")
        if arr.dtype != np.uint8:
            # Tolerate float frames ([0,1] or [0,255]) rather than dropping the tile.
            scale = 255.0 if float(np.nanmax(arr)) <= 1.0 else 1.0
            arr = np.clip(arr.astype(np.float32) * scale, 0, 255).astype(np.uint8)
        if arr.shape[:2] != (tile_h, tile_w):
            arr = np.asarray(
                Image.fromarray(arr).resize((tile_w, tile_h), Image.BILINEAR)
            )
        tiles.append(arr)

    top = np.concatenate(tiles[:2], axis=1)
    bottom = np.concatenate(tiles[2:], axis=1)
    return np.concatenate([top, bottom], axis=0)


# ---------------------------------------------------------------------------
# video_ctx global condition from in-memory frames (client teleop demo).
# ---------------------------------------------------------------------------
@torch.no_grad()
def build_video_ctx_latents_from_frames(
    model: torch.nn.Module,
    frames: np.ndarray,
    video_ctx_size: List[int],
    device: str,
    dtype: torch.dtype,
    num_frames: int,
) -> torch.Tensor:
    """Build the ``video_ctx`` condition from raw RGB frames.

    In-memory twin of ``prepare_video_ctx_latents``: instead of decoding a demo
    mp4 it consumes a ``[T, H, W, 3]`` uint8 RGB array (as the client records it
    from ``cam_high`` during a teleop demo), then applies the identical uniform
    temporal downsample -> resize/crop to ``video_ctx_size`` -> [-1, 1]
    normalize -> temporal pad (T % 4 == 1) -> condition-VAE encode.
    """
    arr = np.asarray(frames)
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(
            f"video_ctx frames must be [T, H, W, 3] RGB uint8, got shape {arr.shape}"
        )
    video = (
        torch.from_numpy(np.ascontiguousarray(arr)).permute(0, 3, 1, 2).float() / 255.0
    )  # [T, 3, H, W]

    if video.shape[0] > num_frames:
        indices = np.linspace(0, video.shape[0] - 1, num_frames, dtype=int)
        video = video[indices]
    if video.shape[0] < num_frames:
        pad = video[-1:].repeat(num_frames - video.shape[0], 1, 1, 1)
        video = torch.cat([video, pad], dim=0)

    target_h, target_w = int(video_ctx_size[0]), int(video_ctx_size[1])
    _, C, H, W = video.shape
    scale = max(target_w / W, target_h / H)
    new_h, new_w = round(H * scale), round(W * scale)
    video = torch.nn.functional.interpolate(
        video, size=(new_h, new_w), mode="bilinear", align_corners=False
    )
    crop_top = (new_h - target_h) // 2
    crop_left = (new_w - target_w) // 2
    video = video[:, :, crop_top:crop_top + target_h, crop_left:crop_left + target_w]

    video = video * 2.0 - 1.0
    video = video.permute(1, 0, 2, 3).unsqueeze(0)  # [1, C, T, H, W]

    T_va = video.shape[2]
    if T_va % 4 != 1:
        pad_t = (4 - (T_va - 1) % 4) % 4
        video = torch.nn.functional.pad(video, (0, 0, 0, 0, 0, pad_t), mode="replicate")

    video = video.to(device=device, dtype=dtype)
    return model._encode_cond_latents(video)


# ---------------------------------------------------------------------------
# Visualization output root (derived from the checkpoint path).
# ---------------------------------------------------------------------------
# Dirs that only describe *where* a checkpoint is stored, not *which* run it
# belongs to; they are skipped when building the tag.
_CKPT_CONTAINER_DIRS = {"weights", "checkpoints", "checkpoint", "ckpt", "ckpts", "models"}


def vis_root_from_ckpt(ckpt: str) -> Path:
    """``<repo>/runs/vis/<exp>_<date>_<step>`` for the given checkpoint.

    Checkpoints live at ``.../<exp_name>/<date_dir>/checkpoints/weights/step_N.pt``,
    so the tag keeps the experiment name, its run timestamp and the step, which
    is enough to tell two rollouts apart at a glance. Shallower layouts just
    contribute whatever ancestors they have.
    """
    path = Path(os.path.expanduser(str(ckpt))).resolve()
    ancestors = [d.name for d in path.parents if d.name and d.name not in _CKPT_CONTAINER_DIRS]
    tag = "_".join(list(reversed(ancestors[:2])) + [path.stem])
    return project_root / "runs" / "vis" / tag


# ---------------------------------------------------------------------------
# Replan bookkeeping.
# ---------------------------------------------------------------------------
# How long to wait on the inference worker before giving up on it. Generous: it
# is one diffusion sample, and the alternative to waiting is touching the model
# from two threads at once.
_INFER_WAIT_TIMEOUT_S = 120.0

# Slack on the measured inference latency when deciding how deep the queue has to
# be to dispatch (see `_prefetch_threshold`). Diffusion sample times are not
# constant -- GPU contention, a video_ctx re-encode -- and the cost of being 25%
# early is a slightly staler chunk, while the cost of being late is a stalled arm.
_PREFETCH_SAFETY = 1.25


def _format_retime_note(info: Dict[str, Any], n_wp: int) -> str:
    """One log fragment describing what the retimer did to a chunk.

    ``guard_bite`` / ``guard_vel`` are reported in every mode because they are the
    diagnostic that says whether the declared limits match the policy: if the
    clamp is routinely holding back 200% of the acceleration limit, the limits are
    the thing to fix, not the smoothing.
    """
    guard = " | guard: acc {:.0%}, vel {:.0%} of limit pre-clamp".format(
        info.get("guard_bite", 0.0), info.get("guard_vel", 0.0)
    )
    if info.get("mode") == "guard":
        return f" | topp: guard-only, anchor={info['anchor']}{guard}"
    if info.get("fallback"):
        return f" | topp: RAW ({info['fallback']}){guard}"
    note = (
        " | topp: {duration:.2f}s for {n} wp ({scale:.2f}x nominal, {consumed:.0%} "
        "queued), sd0={sd:.2f}, peak_v={peak:.0%}, anchor={anchor}".format(
            duration=info["duration"], n=n_wp, scale=info["time_scale"],
            consumed=info["consumed"], sd=info["sd_start"],
            peak=info["peak_vel"], anchor=info["anchor"],
        )
    )
    if info.get("note"):
        note += f" [{info['note']}]"
    return note + guard


# ---------------------------------------------------------------------------
# Policy: model + replan queue.
# ---------------------------------------------------------------------------
class FastWAMPolicy:
    """Instantiate a FastWAM checkpoint and serve raw-qpos actions with replan."""

    def __init__(self, args: argparse.Namespace) -> None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        cfg_path = Path(args.config) if args.config else find_config_from_ckpt(args.ckpt)
        if cfg_path is None or not Path(cfg_path).exists():
            raise FileNotFoundError(
                f"Could not locate config.yaml near {args.ckpt}; pass --config."
            )
        stats_path = (
            Path(args.dataset_stats) if args.dataset_stats
            else find_dataset_stats_from_ckpt(args.ckpt)
        )
        if stats_path is None or not Path(stats_path).exists():
            raise FileNotFoundError(
                f"Could not locate dataset_stats.json near {args.ckpt}; pass --dataset-stats."
            )
        logger.info("Using training config: %s", cfg_path)
        logger.info("Using dataset stats : %s", stats_path)
        self.cfg = OmegaConf.load(str(cfg_path))

        # ---- rollout geometry from the training config ----------------------
        self.video_size = list(self.cfg.data.train.get("video_size", [384, 448]))
        num_frames = int(self.cfg.data.train.num_frames)
        ratio = int(self.cfg.data.train.get("action_video_freq_ratio", 4))
        self.num_video_frames = (num_frames - 1) // ratio + 1
        self.action_horizon = (
            int(args.action_horizon) if args.action_horizon is not None
            else num_frames - 1
        )
        replan_steps = (
            int(args.replan_steps) if args.replan_steps is not None
            else self.num_video_frames - 1  # video frames, i.e. execute ~1 chunk
        )
        self.replan_steps = int(max(1, min(replan_steps, self.action_horizon)))

        # ---- camera / mosaic layout ----------------------------------------
        self.concat_mode = str(self.cfg.data.train.get("concat_multi_camera", None))
        self.cam_keys: Optional[List[str]] = None
        self.proc_sizes: Optional[List[List[int]]] = None
        if self.concat_mode in _MOSAIC_SIZES:
            self.cam_keys = [m["key"] for m in self.cfg.data.train.shape_meta.images]
            self.proc_sizes = [
                list(m["shape"][-2:]) for m in self.cfg.data.train.shape_meta.images
            ]
        else:
            logger.warning(
                "concat_multi_camera='%s' is not a mosaic mode; the first client "
                "camera will be used verbatim.", self.concat_mode,
            )

        # ---- model + processor ---------------------------------------------
        self.model = load_model(
            self.cfg, args.ckpt, self.device, skip_dit_load_override=args.skip_dit_load
        )
        self.dtype = self.model.torch_dtype

        from hydra.utils import instantiate
        self.processor = instantiate(self.cfg.data.train.processor).eval()
        self.processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(stats_path)))

        # Training may re-anchor action targets to the first observed state
        # (RelativeJointTransform); such chunks are shifted back by the replan
        # state in _infer_action_chunk before they are queued for the arm.
        self.action_state_transforms = (
            getattr(self.processor, "action_state_transforms", None) or []
        )
        if self.action_state_transforms:
            logger.info(
                "action_state_transforms active: %s | predicted chunks are "
                "re-anchored to the replan state before being queued",
                [type(t).__name__ for t in self.action_state_transforms],
            )

        # ---- inference knobs ------------------------------------------------
        self.raw_text = bool(args.raw_text)
        self.num_inference_steps = int(args.num_inference_steps)
        self.sigma_shift = args.sigma_shift
        self.seed = None if (args.seed is not None and int(args.seed) < 0) else args.seed
        self.rand_device = str(args.rand_device)
        self.tiled = bool(args.tiled)
        self.bgr_input = bool(args.bgr_input)

        # ---- visualization root --------------------------------------------
        # Everything we dump for inspection (dreamed video, observations,
        # video_ctx clips) lands under one ckpt-derived directory so rollouts of
        # different checkpoints never mix. Explicit --*-dir flags still win.
        self.vis_root = vis_root_from_ckpt(args.ckpt)
        logger.info("Visualization root: %s", self.vis_root)

        # ---- predicted-video visualization ---------------------------------
        # With --vis-video we call ``infer_joint`` instead of ``infer_action``
        # (same action output, plus the decoded video chunk) and dump one mp4
        # per replan so the dreamed rollout can be inspected.
        self.vis_video = bool(args.vis_video)
        self.vis_video_dir = (
            Path(args.vis_video_dir) if args.vis_video_dir
            else self.vis_root / "pred_video"
        )
        self._vis_chunk_count = 0
        if self.vis_video and not hasattr(self.model, "infer_joint"):
            raise ValueError("--vis-video requires a checkpoint exposing `infer_joint`.")

        # ---- observation recording -------------------------------------------
        # Every images payload the client ships (one per replan) is tiled into a
        # 2x2 grid and appended to one mp4 per episode. Frames are written
        # incrementally so a crash/kill still leaves a playable file.
        self.obs_video_dir = (
            None if args.no_obs_video
            else Path(args.obs_video_dir) if args.obs_video_dir
            else self.vis_root / "obs_video"
        )
        self.obs_video_fps = int(args.obs_video_fps)
        self._obs_writer: Optional[Any] = None
        self._obs_path: Optional[Path] = None
        self._obs_tile_hw: Optional[Tuple[int, int]] = None
        self._obs_frame_count = 0
        self._obs_failed = False

        # ---- video_ctx global condition ------------------------------------
        # The condition may come from two places: a demo video baked in at
        # startup (--video-ctx-path, the "default"), or one the client records
        # live via a teleop demo and ships with the `set_video_ctx` command.
        # We keep the startup default around so the client can fall back to it.
        self.has_video_ctx = (
            getattr(self.model, "video_expert", None) is not None
            and getattr(self.model.video_expert, "has_video_ctx_condition", False)
        )
        self.va_size = list(self.cfg.data.train.get("video_ctx_size", self.video_size))
        va_num_cfg = self.cfg.data.train.get("video_ctx_num_frames", None)
        self.va_num = int(va_num_cfg) if va_num_cfg else self.num_video_frames

        self.default_video_ctx_path = (
            Path(args.video_ctx_path).expanduser().resolve()
            if args.video_ctx_path else None
        )
        self.default_video_ctx_latents = None
        if self.has_video_ctx and args.video_ctx_path:
            logger.info("Building default video_ctx condition from %s", args.video_ctx_path)
            self.default_video_ctx_latents = prepare_video_ctx_latents(
                self.model, args.video_ctx_path, self.va_size,
                self.device, self.dtype, self.va_num,
            )
        elif self.has_video_ctx:
            logger.warning(
                "Model expects video_ctx but no --video-ctx-path given; waiting for "
                "the client to supply one (F3 teleop demo)."
            )
        # Active condition served to infer_action; starts as the startup default.
        # The condition is (re)built lazily right before inference: ``set_video_ctx``
        # only records the source and flips ``_video_ctx_dirty``; ``_ensure_video_ctx``
        # rebuilds + caches the latents when dirty, so a fixed condition is encoded
        # exactly once and re-encoded only when it actually changes.
        self.video_ctx_frames: Optional[np.ndarray] = None  # None -> startup default
        self.video_ctx_latents = self.default_video_ctx_latents
        self._video_ctx_dirty = False
        self.video_ctx_dump_dir = (
            None if args.no_video_ctx_dump
            else Path(args.video_ctx_dump_dir) if args.video_ctx_dump_dir
            else self.vis_root / "video_ctx"
        )
        self._video_ctx_dump_idx = 0

        # ---- replan state ---------------------------------------------------
        self.pending_actions: Deque[np.ndarray] = deque()
        self.episode_count = 0
        self.step_count = 0

        # ---- asynchronous replanning ----------------------------------------
        # Off by default: the synchronous path is the one every existing rollout
        # was recorded with. `_cv` guards the queue and everything below it,
        # because under --async-infer the worker thread splices into the queue
        # while the serving thread is popping from it.
        self.async_infer = bool(args.async_infer)
        self.async_blend_steps = max(0, int(args.async_blend_steps))
        self.async_prefetch_steps = int(args.async_prefetch_steps)
        self._cv = threading.Condition()
        self._infer_busy = False
        # Bumped whenever the rollout is cut (reset / configure / new video_ctx):
        # a chunk that lands with a stale generation was conditioned on a world
        # that no longer exists and is dropped rather than spliced.
        self._infer_generation = 0
        self._pop_count = 0  # actions handed to the client, ever (never reset)
        # What the client last received, and the velocity it received it at. The
        # retimer cannot answer this under async replanning: its continuity state
        # describes the end of a chunk the arm has not reached yet.
        self._emitted_q: Optional[np.ndarray] = None
        self._emitted_v: Optional[np.ndarray] = None
        # Inference latency measured in control ticks (EWMA) -- see _prefetch_threshold.
        self._infer_ticks_ewma: Optional[float] = None

        # ---- TOPP re-timing --------------------------------------------------
        # Built by the `configure` command, because the limits and the control
        # period belong to the robot, not to the checkpoint. Until then (or with
        # --no-topp) `retimer` stays None and the raw chunk is queued verbatim.
        self.topp_enabled = not bool(args.no_topp)
        self.topp_mode = str(args.topp_mode)
        self.topp_smooth_window = int(args.topp_smooth_window)
        self.topp_polyorder = int(args.topp_polyorder)
        self.topp_limit_margin = float(args.topp_limit_margin)
        self.topp_max_anchor_error = float(args.topp_max_anchor_error)
        self.retimer: Optional[ChunkRetimer] = None
        if not self.topp_enabled:
            logger.info("TOPP re-timing disabled (--no-topp).")
        elif not TOPP_AVAILABLE:
            logger.warning(
                "TOPP re-timing unavailable (%s); chunks will be queued raw. "
                "Install toppra + scipy to enable it.", TOPP_IMPORT_ERROR,
            )

        logger.info(
            "FastWAM franka policy ready | horizon=%d replan_steps=%d "
            "num_video_frames=%d infer_steps=%d cams=%s bgr_input=%s | "
            "replan=%s topp_mode=%s",
            self.action_horizon, self.replan_steps, self.num_video_frames,
            self.num_inference_steps, self.cam_keys, self.bgr_input,
            (f"async (blend={self.async_blend_steps}, prefetch="
             + (f"{self.async_prefetch_steps})" if self.async_prefetch_steps > 0 else "auto)")
             ) if self.async_infer else "sync",
            self.topp_mode if self.topp_enabled else "off",
        )

    # ---- preprocessing ------------------------------------------------------
    def _build_input_image(self, images: Dict[str, np.ndarray]) -> torch.Tensor:
        if self.cam_keys is not None:
            image = build_mosaic_from_images(
                images, self.cam_keys, self.proc_sizes, self.video_size,
                self.concat_mode, self.device, self.dtype,
            )
        else:
            # Non-mosaic model: take a single camera and resize to video_size.
            import torchvision.transforms.functional as TF
            key = next(iter(images))
            arr = np.ascontiguousarray(images[key])
            x = torch.from_numpy(arr).permute(2, 0, 1).contiguous().float() / 255.0
            x = TF.resize(
                x, size=[int(self.video_size[0]), int(self.video_size[1])],
                interpolation=TF.InterpolationMode.BILINEAR, antialias=True,
            )
            image = (x * 2.0 - 1.0).unsqueeze(0).to(device=self.device, dtype=self.dtype)
        if self.bgr_input:
            image = image[:, [2, 1, 0], :, :]
        return image

    # ---- inference ----------------------------------------------------------
    @torch.no_grad()
    def _infer_action_chunk(
        self, instruction: str, state: np.ndarray, images: Dict[str, np.ndarray]
    ) -> np.ndarray:
        self._ensure_video_ctx()  # (re)build the condition if it changed
        image = self._build_input_image(images)
        proprio = normalize_state(self.processor, np.asarray(state, dtype=np.float32))
        prompt = instruction if self.raw_text else DEFAULT_PROMPT.format(task=instruction)

        infer_kwargs: Dict[str, Any] = dict(
            prompt=prompt,
            input_image=image,
            action_horizon=self.action_horizon,
            proprio=proprio,
            num_inference_steps=self.num_inference_steps,
            sigma_shift=self.sigma_shift,
            seed=None if self.seed is None else int(self.seed),
            rand_device=self.rand_device,
            tiled=self.tiled,
        )
        if self.video_ctx_latents is not None:
            infer_kwargs["video_ctx_latents"] = self.video_ctx_latents

        if self.vis_video:
            # Jointly infer the action chunk and its video so the dreamed rollout
            # can be saved for this replan; the action output is unchanged.
            infer_kwargs["num_video_frames"] = int(self.num_video_frames)
            infer_kwargs["test_action_with_infer_action"] = False
            pred = self.model.infer_joint(**infer_kwargs)
            self._save_vis_video(pred.get("video"))
        else:
            pred = self.model.infer_action(**infer_kwargs)
        # state_vec undoes the training-time action_state_transforms (if any):
        # RelativeJointTransform-trained chunks are relative to the replan state.
        raw_action = denormalize_action(
            self.processor, pred["action"],
            state_vec=np.asarray(state, dtype=np.float32),
        )  # [horizon, D]
        return raw_action

    def _save_vis_video(self, frames: Optional[List[Any]]) -> None:
        """Persist one predicted video chunk per replan (best-effort)."""
        if frames is None:
            logger.warning("--vis-video enabled but infer_joint returned no video.")
            return
        try:
            from simpleicl.utils.video_io import save_mp4
            out_dir = self.vis_video_dir / f"episode{self.episode_count:03d}"
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / f"chunk_{self._vis_chunk_count:04d}.mp4"
            save_mp4(frames, str(path), fps=8)
            logger.info("Saved predicted video chunk -> %s", path)
        except Exception as exc:  # noqa: BLE001 - visualization is best-effort
            logger.warning("Failed to save vis_video chunk: %s", exc)
        self._vis_chunk_count += 1

    # ---- observation recording ----------------------------------------------
    def _obs_slot_order(self, images: Dict[str, np.ndarray]) -> List[str]:
        """2x2 slot order: the model's camera order first, then any extra keys."""
        order = [k for k in (self.cam_keys or []) if k in images]
        order += sorted(k for k in images if k not in order)
        return order[:4]

    def _record_obs_images(self, images: Dict[str, np.ndarray]) -> None:
        """Append the client's camera frames to this episode's 2x2-grid video."""
        if self.obs_video_dir is None or self._obs_failed or not images:
            return
        try:
            order = self._obs_slot_order(images)
            if self._obs_writer is None:
                # Lock the tile size on the first frame: an mp4 needs a constant
                # frame size, and cameras may differ in resolution.
                probe = build_quad_grid(images, order)
                self._obs_tile_hw = (probe.shape[0] // 2, probe.shape[1] // 2)
                import imageio
                self.obs_video_dir.mkdir(parents=True, exist_ok=True)
                self._obs_path = (
                    self.obs_video_dir / f"episode{self.episode_count:03d}_obs.mp4"
                )
                self._obs_writer = imageio.get_writer(
                    str(self._obs_path),
                    fps=max(self.obs_video_fps, 1),
                    codec="libx264",
                    format="FFMPEG",
                    pixelformat="yuv420p",
                )
                dropped = [k for k in sorted(images) if k not in order]
                logger.info(
                    "Recording observations %dx%d, slots=%s%s -> %s",
                    probe.shape[1], probe.shape[0], order,
                    f" (dropped {dropped})" if dropped else "", self._obs_path,
                )
                grid = probe
            else:
                grid = build_quad_grid(images, order, self._obs_tile_hw)
            self._obs_writer.append_data(grid)
            self._obs_frame_count += 1
        except Exception as exc:  # noqa: BLE001 - recording must never break a rollout
            logger.warning("Observation recording disabled after error: %s", exc)
            self._obs_failed = True

    def close_obs_video(self) -> None:
        """Finalize the current episode's observation mp4 (if one is open)."""
        if self._obs_writer is None:
            return
        try:
            self._obs_writer.close()
            logger.info(
                "Saved observation video (%d frames) -> %s",
                self._obs_frame_count, self._obs_path,
            )
        except Exception as exc:  # noqa: BLE001 - best-effort
            logger.warning("Failed to finalize observation video: %s", exc)
        finally:
            self._obs_writer = None
            self._obs_tile_hw = None
            self._obs_frame_count = 0

    # ---- TOPP re-timing -----------------------------------------------------
    def configure(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Accept the robot's control period + joint limits; build the retimer.

        Called once per rollout by the client right after it connects. Returns
        whether TOPP is actually active and a human-readable reason, so the
        client can log what it got rather than silently assuming.
        """
        # Swapping the retimer out from under a worker thread that is about to
        # re-time with it would splice a chunk under the previous robot's limits.
        self._abandon_inference("configure")
        if not spec.get("topp", True):
            self.retimer = None
            return {"topp": False, "detail": "the client asked for TOPP to be off"}
        if not self.topp_enabled:
            self.retimer = None
            return {"topp": False, "detail": "disabled on the server (--no-topp)"}
        if self.topp_mode == "full" and not TOPP_AVAILABLE:
            self.retimer = None
            return {
                "topp": False,
                "detail": f"toppra/scipy unavailable for --topp-mode full "
                          f"({TOPP_IMPORT_ERROR}); --topp-mode guard needs neither",
            }

        fps = float(spec.get("fps") or 0.0)
        if fps <= 0.0:
            self.retimer = None
            return {"topp": False, "detail": f"the client reported an unusable fps={fps!r}"}
        try:
            self.retimer = ChunkRetimer(
                dt=1.0 / fps,
                max_vel=spec["max_vel"],
                max_acc=spec["max_acc"],
                gripper_dims=spec.get("gripper_dims", ()),
                smooth_window=self.topp_smooth_window,
                smooth_polyorder=self.topp_polyorder,
                limit_margin=self.topp_limit_margin,
                max_anchor_error=self.topp_max_anchor_error,
                mode=self.topp_mode,
            )
        except Exception as exc:  # noqa: BLE001 - report back instead of dying
            self.retimer = None
            logger.exception("TOPP configure failed")
            return {"topp": False, "detail": f"could not build the retimer: {exc}"}

        detail = self.retimer.describe()
        logger.info("TOPP re-timing enabled | %s", detail)
        return {"topp": True, "detail": detail}

    def _retime(
        self, raw_action: np.ndarray, n_exec: int, state: np.ndarray
    ) -> Tuple[np.ndarray, str]:
        """Re-time the chunk if configured; never let it break the rollout."""
        if self.retimer is None:
            return raw_action[:n_exec], ""
        try:
            actions, info = self.retimer.retime(raw_action, n_exec, state=state)
        except Exception as exc:  # noqa: BLE001 - fall back to the raw chunk
            logger.exception("TOPP re-timing raised; queuing the raw chunk")
            return raw_action[:n_exec], f" | topp: RAW ({exc})"
        return actions, _format_retime_note(info, raw_action.shape[0])

    def _fill_queue(
        self, instruction: str, state: np.ndarray, images: Dict[str, np.ndarray]
    ) -> None:
        t0 = time.perf_counter()
        raw_action = self._infer_action_chunk(instruction, state, images)
        t_infer = time.perf_counter() - t0
        n_exec = min(self.replan_steps, raw_action.shape[0])
        actions, topp_note = self._retime(raw_action, n_exec, state)
        for action in actions:
            self.pending_actions.append(np.asarray(action, dtype=np.float32))
        logger.info(
            "[replan] infer %.2fs | queued %d actions%s",
            t_infer, len(actions), topp_note,
        )

    # ---- asynchronous replanning --------------------------------------------
    def _prefetch_threshold(self) -> int:
        """Queue length at or below which the next inference is dispatched.

        The useful unit here is control ticks, not seconds. A chunk that lands
        after the client has consumed ``k`` more actions is stale in its first
        ``k`` rows -- they describe ticks that already happened -- so to hand over
        without a gap the queue has to still be that deep when the chunk arrives.

        Which means the number to measure is the inference latency in ticks, which
        ``_observe_latency`` tracks. On top of it: one blend window (those samples
        are consumed by the cross-fade, not by the handover) and
        ``_PREFETCH_SAFETY`` of slack, because dispatching with exactly
        latency-plus-blend left in the queue lands the chunk with nothing to spare
        and any jitter in the sample time starves. Floored at ``replan_steps`` so
        we never replan less often than the synchronous path, capped below the
        horizon (dispatching with a full queue only yields a chunk whose useful
        part is shorter).
        """
        if self.async_prefetch_steps > 0:
            return min(self.async_prefetch_steps, self.action_horizon - 1)
        if self._infer_ticks_ewma is None:
            return self.action_horizon - 1  # unmeasured: dispatch as early as we can
        want = (
            int(math.ceil(_PREFETCH_SAFETY * self._infer_ticks_ewma))
            + self.async_blend_steps
        )
        return int(min(max(want, self.replan_steps), self.action_horizon - 1))

    def _observe_latency(self, t_infer: float, elapsed: int) -> None:
        """Record how long an inference took, in control ticks. Caller holds ``_cv``.

        Measured from the wall clock and the client's control period, *not* from
        how many actions went out while it ran. Those two agree only as long as
        the client keeps ticking -- and it stops ticking precisely when the queue
        starves, so the pop count reads short exactly when the estimate matters,
        which feeds back into a threshold too small to prevent the next starve.
        (Observed: a true 12-tick latency estimated as 0, then 8, 9, 10, ...,
        starving on every chunk the whole way up.) ``elapsed`` is still the right
        answer for *aligning* the chunk; it is the wrong one for predicting.

        Rises immediately and decays slowly: a latency spike has to widen the
        prefetch window now, while one lucky fast sample should not narrow it.
        """
        dt = None if self.retimer is None else self.retimer.dt
        sample = (t_infer / dt) if dt else float(max(elapsed, 0))
        if not np.isfinite(sample) or sample <= 0.0:
            return
        self._infer_ticks_ewma = (
            sample if self._infer_ticks_ewma is None
            else max(sample, 0.8 * self._infer_ticks_ewma + 0.2 * sample)
        )

    def _pop_locked(self) -> np.ndarray:
        """Hand out the next queued action. Caller holds ``self._cv``.

        Records what was emitted and the velocity it was emitted at: that is the
        seed the discrete guard needs when the next chunk is spliced in, and under
        async replanning it cannot be read off the retimer, whose continuity state
        describes the tail of a chunk the client never got to.
        """
        action = self.pending_actions.popleft()
        current = np.asarray(action, dtype=np.float64)
        dt = None if self.retimer is None else self.retimer.dt
        if dt and self._emitted_q is not None:
            self._emitted_v = (current - self._emitted_q) / dt
        self._emitted_q = current
        self._pop_count += 1
        self.step_count += 1
        return action

    def _splice_seed(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Where the committed stream will be when the cross-fade hands over.

        This is the anchor a TOPP re-time of the incoming chunk has to start from:
        not the last command emitted (the queue moves past that before the new
        chunk takes over) and not the end of the previous chunk (the client never
        reaches it). ``(None, None)`` when nothing is committed yet, which lets
        the retimer fall back to picking an anchor from the measured state.
        """
        queued = list(self.pending_actions)
        if not queued:
            return self._emitted_q, self._emitted_v
        n = int(min(max(self.async_blend_steps, 1), len(queued)))
        q = np.asarray(queued[n - 1], dtype=np.float64)
        prev = np.asarray(queued[n - 2], dtype=np.float64) if n >= 2 else self._emitted_q
        dt = None if self.retimer is None else self.retimer.dt
        vel = None if (prev is None or not dt) else (q - np.asarray(prev, dtype=np.float64)) / dt
        return q, vel

    def _dispatch_inference(
        self, instruction: str, state: np.ndarray, images: Dict[str, np.ndarray]
    ) -> None:
        """Start inferring a chunk in the background. Caller holds ``self._cv``.

        ``_pop_count`` is captured here because it is what dates the chunk: row
        ``i`` of the result predicts the action for rollout tick
        ``_pop_count + i``, and comparing against the count at arrival is how
        ``_merge_chunk`` knows which rows are already in the past.
        """
        self._infer_busy = True
        threading.Thread(
            target=self._infer_worker,
            args=(
                instruction,
                np.asarray(state, dtype=np.float32).copy(),
                images,
                self._infer_generation,
                self._pop_count,
            ),
            name="fastwam-infer",
            daemon=True,
        ).start()

    def _infer_worker(
        self,
        instruction: str,
        state: np.ndarray,
        images: Dict[str, np.ndarray],
        generation: int,
        pops_at_dispatch: int,
    ) -> None:
        """Infer off the serving thread, then splice under the lock."""
        raw_action = None
        t_infer = 0.0
        try:
            t0 = time.perf_counter()
            raw_action = self._infer_action_chunk(instruction, state, images)
            t_infer = time.perf_counter() - t0
        except Exception:  # noqa: BLE001 - a failed chunk must not end the rollout
            logger.exception("Async inference failed; the queue keeps draining.")
        with self._cv:
            try:
                if raw_action is None:
                    pass
                elif generation != self._infer_generation:
                    logger.info(
                        "Dropping a chunk from a rollout that has since been cut."
                    )
                else:
                    self._merge_chunk(raw_action, state, pops_at_dispatch, t_infer)
            except Exception:  # noqa: BLE001 - keep the old queue, keep serving
                logger.exception("Failed to splice an async chunk; keeping the queue.")
            finally:
                # Unconditionally, or `should_request_observation` never goes true
                # again and a starved `step` waits out its whole timeout.
                self._infer_busy = False
                self._cv.notify_all()

    def _merge_chunk(
        self,
        raw_action: np.ndarray,
        state: np.ndarray,
        pops_at_dispatch: int,
        t_infer: float,
    ) -> None:
        """Splice a fresh chunk into the queue that is still draining.

        ``raw_action[i]`` predicts rollout tick ``pops_at_dispatch + i``, so by the
        time it lands its first ``k = _pop_count - pops_at_dispatch`` rows describe
        ticks the client has already executed. Dropping them is what aligns the
        chunk with the queue: row ``k`` and ``pending_actions[0]`` are then two
        predictions of the *same* tick, one from an older observation and one from
        a newer.

        They are cross-faded rather than swapped. A hard swap is a step in the
        command stream -- the two chunks disagree by however much the newer
        observation changed the plan -- and that is exactly the seam the
        synchronous path pays at every replan. The raised-cosine ramp reaches 1.0
        at its last sample, so the join to the untouched remainder of the new chunk
        is seamless; the discontinuity that is left sits at the *head* of the ramp,
        where the guard below can absorb it.

        Caller holds ``self._cv``. The re-time/guard work runs under it too: it is
        milliseconds of numpy against a diffusion sample, and doing it outside
        would mean the queue could move between reading the splice point and
        writing the result.
        """
        raw_action = np.asarray(raw_action, dtype=np.float64)
        elapsed = int(self._pop_count - pops_at_dispatch)
        self._observe_latency(t_infer, elapsed)
        fresh = raw_action[max(elapsed, 0):]
        if fresh.shape[0] < 2:
            logger.warning(
                "Inference took %.2fs = %d ticks, outrunning the whole %d-step "
                "horizon: the chunk is entirely in the past and was dropped. Lower "
                "--num-inference-steps or raise --action-horizon, or the arm will "
                "keep stalling.", t_infer, elapsed, raw_action.shape[0],
            )
            return

        note = ""
        if self.retimer is not None and self.retimer.mode == "full":
            # Only `full` mode does anything here that the final guard would not:
            # re-parameterize time and filter the geometry. It has to start from
            # the point the cross-fade hands over at, not from the last command.
            splice_q, splice_v = self._splice_seed()
            retimed, info = self.retimer.retime(
                fresh, fresh.shape[0], state=state,
                anchor=splice_q, anchor_vel=splice_v,
            )
            fresh = np.asarray(retimed, dtype=np.float64)
            note = _format_retime_note(info, fresh.shape[0])

        queued = list(self.pending_actions)
        n_blend = int(min(self.async_blend_steps, len(queued), fresh.shape[0]))
        merged = np.array(fresh, dtype=np.float64, copy=True)
        if n_blend > 0:
            ramp = np.arange(1, n_blend + 1, dtype=np.float64)
            weight = 0.5 * (1.0 - np.cos(np.pi * ramp / n_blend))  # (0, 1], ends at 1
            prev = np.asarray(queued[:n_blend], dtype=np.float64)
            merged[:n_blend] = (
                (1.0 - weight)[:, None] * prev + weight[:, None] * merged[:n_blend]
            )
            # The gripper is not a joint: a cross-faded open/close is a half-open
            # command, which is worse than either end of the ramp. Hand it over at
            # once, the same reason the guard leaves it alone downstream.
            for i in (() if self.retimer is None else self.retimer.gripper_dims):
                merged[:n_blend, i] = fresh[:n_blend, i]

        # The limits have to hold on the stream that actually reaches the arm, and
        # that stream is this splice -- neither of the two chunks it was built from.
        seed_q = self._emitted_q
        seed_v = self._emitted_v
        if seed_q is None and state is not None:
            seed_q, seed_v = np.asarray(state, dtype=np.float64).reshape(-1), None
        if (
            self.retimer is not None
            and seed_q is not None
            and seed_q.shape[0] == self.retimer.dof
        ):
            guarded, ginfo = self.retimer.guard(merged, seed_q, seed_v)
            merged = np.asarray(guarded, dtype=np.float64)
            note += " | guard: acc {:.0%}, vel {:.0%} of limit pre-clamp".format(
                ginfo["guard_bite"], ginfo["guard_vel"]
            )

        self.pending_actions.clear()
        for action in merged:
            self.pending_actions.append(np.asarray(action, dtype=np.float32))
        logger.info(
            "[replan-async] infer %.2fs (%d ticks) | dropped %d stale, blended %d "
            "into %d queued -> %d queued (prefetch<=%d)%s",
            t_infer, elapsed, elapsed, n_blend, len(queued),
            len(self.pending_actions), self._prefetch_threshold(), note,
        )

    def _abandon_inference(self, why: str) -> None:
        """Invalidate any in-flight chunk and wait for the worker to let go.

        Bumping the generation is what discards the result; the wait is what makes
        the model safe to touch again. ``configure``, ``set_video_ctx`` and
        ``reset`` all mutate state the worker reads (the retimer, the video_ctx
        latents, the queue), and the model itself is not reentrant.

        A no-op when nothing is in flight, which is the synchronous path always.
        """
        with self._cv:
            self._infer_generation += 1
            if not self._infer_busy:
                return
            logger.info("Waiting for the in-flight inference to finish (%s).", why)
            if not self._cv.wait_for(
                lambda: not self._infer_busy, timeout=_INFER_WAIT_TIMEOUT_S
            ):
                logger.error(
                    "The inference worker did not finish within %.0fs (%s); "
                    "continuing without it.", _INFER_WAIT_TIMEOUT_S, why,
                )

    def _step_async(
        self,
        instruction: str,
        state: Optional[np.ndarray],
        images: Optional[Dict[str, np.ndarray]],
    ) -> Optional[np.ndarray]:
        """Dispatch a replan if it is due, then hand out an action immediately."""
        with self._cv:
            if (
                not self._infer_busy
                and state is not None
                and images is not None
                and len(self.pending_actions) <= self._prefetch_threshold()
            ):
                self._dispatch_inference(instruction, state, images)
            elif not self.pending_actions and not self._infer_busy:
                raise ValueError(
                    "Action queue is empty and nothing is in flight: step must "
                    "include state + images. Call should_request_observation first."
                )
            if not self.pending_actions:
                # The whole point of async replanning is not to get here: it means
                # inference did not fit inside the runway the queue had.
                logger.warning(
                    "Action queue starved after %d ticks; blocking on the "
                    "in-flight chunk. Inference needs ~%s ticks but the horizon "
                    "only covers %d.", self._pop_count,
                    "?" if self._infer_ticks_ewma is None
                    else f"{self._infer_ticks_ewma:.0f}", self.action_horizon,
                )
                self._cv.wait_for(
                    lambda: bool(self.pending_actions) or not self._infer_busy,
                    timeout=_INFER_WAIT_TIMEOUT_S,
                )
            if not self.pending_actions:
                logger.warning("No action generated; returning None.")
                return None
            return self._pop_locked()

    # ---- command handlers ---------------------------------------------------
    def should_request_observation(self) -> bool:
        if not self.async_infer:
            return not self.pending_actions
        with self._cv:
            # One inference at a time: the model is not reentrant, and a second
            # chunk in flight would be spliced against a queue it never saw.
            if self._infer_busy:
                return False
            return len(self.pending_actions) <= self._prefetch_threshold()

    def step(
        self,
        instruction: str,
        state: Optional[np.ndarray],
        images: Optional[Dict[str, np.ndarray]],
    ) -> Optional[np.ndarray]:
        if images is not None:
            self._record_obs_images(images)
        if self.async_infer:
            return self._step_async(instruction, state, images)
        if not self.pending_actions:
            if state is None or images is None:
                raise ValueError(
                    "Action queue is empty (replan step): step must include "
                    "state + images. Call should_request_observation first."
                )
            self._fill_queue(instruction, state, images)
        if not self.pending_actions:
            logger.warning("No action generated; returning None.")
            return None
        self.step_count += 1
        return self.pending_actions.popleft()

    def reset(self) -> None:
        self._abandon_inference("reset")
        self.close_obs_video()  # the finished episode gets its own file
        with self._cv:
            self.pending_actions.clear()
            # The command stream is broken, so nothing about it carries over.
            # `_pop_count` deliberately stays monotonic: it only ever appears in
            # differences, and the generation bump is what invalidates old chunks.
            self._emitted_q = None
            self._emitted_v = None
        self.episode_count += 1
        self.step_count = 0
        self._vis_chunk_count = 0
        if self.retimer is not None:
            # Dropping the queue breaks the command stream (the client pauses on
            # F3 and the arm holds its target), so the anchor and the carried
            # velocity no longer describe the robot: re-seed from the next state.
            self.retimer.reset()
        logger.info("reset -> episode %d", self.episode_count)

    def new_connection(self) -> None:
        """A fresh client owns a fresh rollout *and* a fresh TOPP configuration.

        The retimer holds the previous client's control period and joint limits.
        A different robot (or an older client that never sends ``configure``)
        must not inherit them, so drop it and wait to be told again.
        """
        # Before the retimer goes, so no worker is mid-splice when it does.
        self._abandon_inference("new connection")
        self.retimer = None
        self.reset()

    def set_video_ctx(self, frames: Optional[np.ndarray]) -> bool:
        """Record the video_ctx source; the latents are (re)built before inference.

        ``frames`` None -> restore the startup default (client pressed F2);
        ``frames`` a [T, H, W, 3] uint8 RGB array -> use the client's teleop demo
        (client pressed F3). This only stashes the source and marks it dirty;
        ``_ensure_video_ctx`` does the actual encode right before the next
        inference (and only if it changed). Returns whether the model consumes a
        video_ctx condition at all (so the client can log it).
        """
        if not self.has_video_ctx:
            logger.warning("set_video_ctx ignored: model has no video_ctx condition.")
            return False
        # A chunk already in flight was conditioned on the old clip, and the
        # worker may be inside `_ensure_video_ctx` right now.
        self._abandon_inference("video_ctx changed")
        self.video_ctx_frames = None if frames is None else np.asarray(frames)
        self._video_ctx_dirty = True
        logger.info(
            "video_ctx source updated (%s); will (re)build before next inference.",
            "startup default" if frames is None
            else f"{self.video_ctx_frames.shape[0]} client frames",
        )
        return True

    def _dump_video_ctx(self, frames: np.ndarray) -> None:
        """Persist the active video_ctx clip to disk (for inspection)."""
        if self.video_ctx_dump_dir is None:
            return
        try:
            from simpleicl.utils.video_io import save_video_to_mp4
            self.video_ctx_dump_dir.mkdir(parents=True, exist_ok=True)
            self._video_ctx_dump_idx += 1
            path = self.video_ctx_dump_dir / f"video_ctx_{self._video_ctx_dump_idx:03d}.mp4"
            save_video_to_mp4(np.asarray(frames), str(path), fps=8, value_range="[0,255]")
            logger.info("Saved video_ctx clip -> %s", path)
        except Exception as exc:  # noqa: BLE001 - dumping is best-effort
            logger.warning("Failed to dump video_ctx clip: %s", exc)

    def _ensure_video_ctx(self) -> None:
        """Build + cache the video_ctx latents if the source changed (lazy)."""
        if not self._video_ctx_dirty:
            return
        if self.video_ctx_frames is None:
            self.video_ctx_latents = self.default_video_ctx_latents
            logger.info(
                "video_ctx -> startup default (%s).",
                "loaded" if self.default_video_ctx_latents is not None else "none",
            )
        else:
            t0 = time.perf_counter()
            self.video_ctx_latents = build_video_ctx_latents_from_frames(
                self.model, self.video_ctx_frames, self.va_size,
                self.device, self.dtype, self.va_num,
            )
            self._dump_video_ctx(self.video_ctx_frames)
            logger.info(
                "video_ctx rebuilt from client demo: %d frames -> %d in %.2fs",
                self.video_ctx_frames.shape[0], self.va_num, time.perf_counter() - t0,
            )
        self._video_ctx_dirty = False


# ---------------------------------------------------------------------------
# TCP serving loop.
# ---------------------------------------------------------------------------
def _handle_request(policy: FastWAMPolicy, msg: Dict[str, Any]) -> Dict[str, Any]:
    cmd = msg.get("cmd")
    if cmd == "should_request_observation":
        return {"status": "ok", "result": policy.should_request_observation()}
    if cmd == "configure":
        result = policy.configure(msg)
        return {"status": "ok", **result}
    if cmd == "step":
        action = policy.step(
            instruction=str(msg.get("instruction", "")),
            state=msg.get("state"),
            images=msg.get("images"),
        )
        payload = None if action is None else np.asarray(action, dtype=np.float32)
        return {"status": "ok", "action": payload}
    if cmd == "get_video_ctx_info":
        return {
            "status": "ok",
            "has_video_ctx": policy.has_video_ctx,
            "num_frames": policy.va_num,
            "size": list(policy.va_size),
        }
    if cmd == "set_video_ctx":
        active = policy.set_video_ctx(msg.get("frames"))
        return {"status": "ok", "has_video_ctx": active}
    if cmd == "reset":
        policy.reset()
        return {"status": "ok"}
    if cmd == "shutdown":
        return {"status": "ok", "shutdown": True}
    raise ValueError(f"Unknown command: {cmd!r}")


def serve(policy: FastWAMPolicy, host: str, port: int) -> None:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(1)
    logger.info("FastWAM policy server listening on %s:%d", host, port)

    try:
        while True:
            conn, addr = server.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            logger.info("Client connected: %s", addr)
            # A fresh connection is a fresh rollout (and a fresh TOPP config).
            policy.new_connection()
            try:
                while True:
                    try:
                        msg = _recv_msg(conn)
                    except (pickle.UnpicklingError, EOFError, ImportError,
                            AttributeError, TypeError, ValueError) as exc:
                        # An undecodable frame was still consumed whole, so the
                        # stream stays aligned: report it and keep the (expensive
                        # to reload) model alive instead of dying on the message.
                        logger.exception("Failed to decode a client message; skipping it.")
                        _send_msg(conn, {
                            "status": "error",
                            "message": f"server could not unpickle the message: {exc}",
                            "traceback": traceback.format_exc(),
                        })
                        continue
                    if msg is None:
                        logger.info("Client disconnected: %s", addr)
                        break
                    try:
                        response = _handle_request(policy, msg)
                    except Exception as exc:  # noqa: BLE001 - report back to client
                        logger.exception("Error handling %s", msg.get("cmd"))
                        _send_msg(conn, {
                            "status": "error",
                            "message": str(exc),
                            "traceback": traceback.format_exc(),
                        })
                        continue
                    _send_msg(conn, response)
                    if response.get("shutdown"):
                        logger.info("Shutdown requested by client.")
                        break
            finally:
                # A closed connection ends the rollout: flush recorded media.
                policy.close_obs_video()
                conn.close()
    except KeyboardInterrupt:
        logger.info("Server interrupted; shutting down.")
    finally:
        policy.close_obs_video()
        server.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # networking
    p.add_argument("--host", default="0.0.0.0", help="Bind address (default: all interfaces).")
    p.add_argument("--port", type=int, default=5555)
    # model / data
    p.add_argument("--ckpt", required=True, help="Path to weights .pt.")
    p.add_argument("--config", default=None, help="Training config.yaml (auto-located near --ckpt).")
    p.add_argument("--dataset-stats", default=None, help="dataset_stats.json (auto-located near --ckpt).")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--skip-dit-load-from-pretrain", dest="skip_dit_load", default=None,
                   type=lambda s: str(s).strip().lower() in ("1", "true", "yes", "y"))
    # rollout / replan
    p.add_argument("--action-horizon", type=int, default=None,
                   help="Actions predicted per chunk (default: num_frames-1).")
    p.add_argument("--replan-steps", type=int, default=8,
                   help="Actions executed before re-inferring (default: 8). Under "
                        "--async-infer this is only a floor on how often a replan is "
                        "dispatched; the queue depth is set by the horizon.")
    # Asynchronous replanning: infer the next chunk while the current one is still
    # being handed out, and cross-fade the two. Off by default -- the synchronous
    # path is what every existing rollout was recorded with.
    p.add_argument("--async-infer", action="store_true",
                   help="Infer the next chunk on a worker thread instead of making the "
                        "client block for it, and splice it into the draining queue. "
                        "Removes the stall (and the stop-go jitter it causes) between "
                        "chunks; requires inference to fit inside --action-horizon "
                        "control ticks. Off by default.")
    p.add_argument("--async-blend-steps", type=int, default=4,
                   help="Ticks over which a newly arrived chunk is cross-faded into the "
                        "queued one (raised cosine). 0 hard-swaps, which puts the seam "
                        "back. Only used with --async-infer (default: 4).")
    p.add_argument("--async-prefetch-steps", type=int, default=0,
                   help="Queue length at or below which the next inference is "
                        "dispatched. 0 (default) measures the inference latency in "
                        "control ticks and derives it, which is usually what you want; "
                        "set it explicitly to pin the replan cadence.")
    # TOPP re-timing of the predicted chunk (see chunk_topp.py). The joint limits
    # and the control period come from the client's `configure` command; these
    # flags only shape / disable the re-timing.
    p.add_argument("--no-topp", action="store_true",
                   help="Queue the raw predicted chunk instead of re-timing it under "
                        "the client's joint velocity/acceleration limits.")
    p.add_argument("--topp-mode", choices=("full", "guard"), default="full",
                   help="`full` smooths the chunk and re-parameterizes its timing under "
                        "the client's limits. `guard` keeps the chunk's own timing -- one "
                        "waypoint per control tick, the speed the demos were recorded at "
                        "-- and only clamps the steps that break the limits. Switch to "
                        "`guard` when `full` logs a time_scale well above 1: that means "
                        "the declared limits are tighter than the demonstrated motion "
                        "and TOPP can only honour them by slowing everything down "
                        "(default: full).")
    p.add_argument("--topp-smooth-window", type=int, default=7,
                   help="Savitzky-Golay window (waypoints) applied to the chunk before "
                        "re-timing; <3 disables it. Filtering matters: TOPP on a raw "
                        "chunk brakes for the per-step noise and crawls (default: 7).")
    p.add_argument("--topp-polyorder", type=int, default=2,
                   help="Savitzky-Golay polynomial order (default: 2).")
    p.add_argument("--topp-limit-margin", type=float, default=0.9,
                   help="Scale applied to the client's limits before solving, to absorb "
                        "the few-percent overshoot of the spline between toppra's "
                        "gridpoints (default: 0.9).")
    p.add_argument("--topp-max-anchor-error", type=float, default=0.5,
                   help="If the last command is further than this (rad, worst joint) from "
                        "the measured state, re-seed the path from the measured state "
                        "instead of building on a pose the arm never reached; <=0 "
                        "disables the check (default: 0.5).")
    # diffusion sampler
    p.add_argument("--num-inference-steps", type=int, default=10)
    p.add_argument("--sigma-shift", type=float, default=None)
    p.add_argument("--seed", type=int, default=42, help="Diffusion seed (<0 -> None).")
    p.add_argument("--rand-device", default="cpu")
    p.add_argument("--tiled", action="store_true")
    p.add_argument("--raw-text", action="store_true",
                   help="Use the instruction verbatim (skip the DEFAULT_PROMPT template).")
    p.add_argument("--bgr-input", action="store_true",
                   help="Swap RGB->BGR before feeding the model (for cameras that "
                        "deliver BGR; leave off for RGB cameras).")
    p.add_argument("--video-ctx-path", default=None,
                   help="Demo video for the video_ctx global condition (only if the "
                        "checkpoint was trained with has_video_ctx_condition=True). "
                        "Used as the startup default that the client's F2 falls back to.")
    p.add_argument("--vis-video", action="store_true",
                   help="Infer with `infer_joint` instead of `infer_action` (same action "
                        "output, plus the dreamed video) and save one mp4 per replan "
                        "chunk under --vis-video-dir.")
    p.add_argument("--vis-video-dir", default=None,
                   help="Where --vis-video writes <dir>/episodeNNN/chunk_NNNN.mp4 "
                        "(default: <vis-root>/pred_video).")
    p.add_argument("--obs-video-dir", default=None,
                   help="Where to record the client's observations: every images "
                        "payload is tiled into a 2x2 grid and appended to "
                        "<dir>/episodeNNN_obs.mp4 (default: <vis-root>/obs_video).")
    p.add_argument("--no-obs-video", action="store_true",
                   help="Disable observation recording.")
    p.add_argument("--obs-video-fps", type=int, default=8,
                   help="Playback fps of the observation video (one frame per replan).")
    p.add_argument("--video-ctx-dump-dir", default=None,
                   help="Where each client-supplied video_ctx clip is dumped as an mp4 "
                        "(one file per change) for inspection "
                        "(default: <vis-root>/video_ctx).")
    p.add_argument("--no-video-ctx-dump", action="store_true",
                   help="Disable dumping of video_ctx clips.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    policy = FastWAMPolicy(args)
    serve(policy, args.host, args.port)


if __name__ == "__main__":
    main()