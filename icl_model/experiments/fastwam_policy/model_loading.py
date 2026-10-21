"""Checkpoint loading and action/state normalization helpers for the policy server."""

import importlib
import inspect
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

# Mosaic geometry per concat mode: the top camera fills [256, W] and each wrist
# camera fills [128, W//2], tiled into a [384, W] mosaic.
_MOSAIC_SIZES = {
    "realworld": {"top": [256, 448], "wrist": [128, 224]},
}


def find_config_from_ckpt(ckpt_path: str) -> Optional[Path]:
    """Walk up from *ckpt_path* to locate the training ``config.yaml``."""
    ckpt = Path(ckpt_path).resolve()
    for parent in ckpt.parents[:6]:
        candidate = parent / "config.yaml"
        if candidate.exists():
            return candidate
    return None


def find_dataset_stats_from_ckpt(ckpt_path: str) -> Optional[Path]:
    """Walk up from *ckpt_path* to locate ``dataset_stats.json``."""
    ckpt = Path(ckpt_path).resolve()
    for parent in ckpt.parents[:6]:
        candidate = parent / "dataset_stats.json"
        if candidate.exists():
            return candidate
    return None


def _mixed_precision_to_dtype(mixed_precision: str) -> torch.dtype:
    key = str(mixed_precision).strip().lower()
    return {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}.get(
        key, torch.bfloat16
    )


def load_model(
    cfg: DictConfig,
    ckpt: str,
    device: str,
    skip_dit_load_override: Optional[bool] = None,
) -> torch.nn.Module:
    """Instantiate the model from the training config and load the checkpoint.

    At serving time the text encoder is enabled because prompts are provided
    as raw text, and the standalone ActionDiT pretrain is skipped because the
    checkpoint already contains the action expert.

    ``skip_dit_load_from_pretrain`` is chosen from ``use_lora``: LoRA
    checkpoints only store adapters, so the frozen backbone must be loaded
    from the pretrained weights first. Full-finetune checkpoints contain the
    whole DiT, so the pretrained load can be skipped.
    """
    dtype = _mixed_precision_to_dtype(cfg.get("mixed_precision", "bf16"))

    kwargs = OmegaConf.to_container(cfg.model, resolve=True)
    target = kwargs.pop("_target_")
    mod_name, fn_name = target.rsplit(".", 1)
    create_fn = getattr(importlib.import_module(mod_name), fn_name)

    use_lora = kwargs.get("use_lora")
    if use_lora is None:
        use_lora = bool(cfg.get("use_lora", False))
    use_lora = bool(use_lora)

    if skip_dit_load_override is not None:
        skip_dit = bool(skip_dit_load_override)
    else:
        skip_dit = not use_lora

    overrides = {
        "load_text_encoder": True,
        "action_dit_pretrained_path": None,
        "use_lora": use_lora,
        "skip_dit_load_from_pretrain": skip_dit,
    }
    accepted = set(inspect.signature(create_fn).parameters)
    for key, val in overrides.items():
        if key in accepted:
            kwargs[key] = val
    kwargs = {k: v for k, v in kwargs.items() if k in accepted}

    logging.info(
        "Instantiating model via %s (dtype=%s, device=%s, use_lora=%s, "
        "skip_dit_load_from_pretrain=%s) ...",
        target, dtype, device, kwargs.get("use_lora"),
        kwargs.get("skip_dit_load_from_pretrain"),
    )
    model = create_fn(**kwargs, model_dtype=dtype, device=device)
    model.load_checkpoint(str(ckpt))
    logging.info("Loaded checkpoint: %s", ckpt)
    return model.to(device).eval()


@torch.no_grad()
def prepare_video_ctx_latents(
    model: torch.nn.Module,
    video_path: str,
    video_ctx_size: List[int],
    device: str,
    dtype: torch.dtype,
    num_frames: int,
) -> torch.Tensor:
    """Build the ``video_ctx`` condition from a demo video.

    Reads the whole video, uniformly downsamples it to ``num_frames``,
    resizes/crops to ``video_ctx_size``, normalizes to [-1, 1], pads the
    temporal dim to satisfy T % 4 == 1, then VAE-encodes the result.
    """
    import av

    path = Path(video_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"video_ctx source not found: {path}")

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"

        total = stream.frames if stream.frames and stream.frames > 0 else None
        if total is None and stream.duration and stream.average_rate and stream.time_base:
            total = int(float(stream.duration * stream.time_base) * float(stream.average_rate))
            if total <= 0:
                total = None

        needed = (
            set(np.linspace(0, total - 1, num_frames, dtype=int).tolist())
            if total is not None and total > num_frames
            else None
        )

        frames = []
        for i, frame in enumerate(container.decode(video=0)):
            if needed is None or i in needed:
                frames.append(torch.from_numpy(frame.to_ndarray(format="rgb24")).permute(2, 0, 1))
            if needed is not None and len(frames) >= num_frames:
                break

    video = torch.stack(frames, dim=0).float() / 255.0

    if needed is None and video.shape[0] > num_frames:
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


def normalize_state(processor, state_vec: np.ndarray) -> torch.Tensor:
    """Raw joint vector -> normalized proprio [1, proprio_dim]."""
    state_meta = processor.shape_meta["state"]
    if len(state_meta) != 1:
        raise ValueError("Expected exactly one merged state key in shape_meta['state'].")
    state_key = state_meta[0]["key"]
    batch = {"state": {state_key: torch.as_tensor(state_vec, dtype=torch.float32).unsqueeze(0)}}
    batch = processor.action_state_transform(batch)
    batch = processor.normalizer.forward(batch)
    return batch["state"][state_key]


def denormalize_action(
    processor,
    action_tensor: torch.Tensor,
    state_vec: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Normalized action [T,D] or [1,T,D] -> raw qpos actions [T,D].

    When the training config applies ``action_state_transforms`` (e.g. actions
    re-anchored to the first observed state), pass the replan-time ``state_vec``
    so the same transforms are undone.
    """
    if action_tensor.ndim == 2:
        action_tensor = action_tensor.unsqueeze(0)
    action_meta = processor.shape_meta["action"]
    if len(action_meta) != 1:
        raise ValueError("Expected exactly one merged action key in shape_meta['action'].")
    action_key = action_meta[0]["key"]
    normalizer = processor.normalizer.normalizers["action"][action_key]
    denorm = normalizer.backward(action_tensor.to(dtype=torch.float32, device="cpu"))
    if state_vec is not None:
        transforms = getattr(processor, "action_state_transforms", None) or []
        if transforms:
            state_meta = processor.shape_meta["state"]
            if len(state_meta) != 1:
                raise ValueError("Expected exactly one merged state key in shape_meta['state'].")
            state_key = state_meta[0]["key"]
            batch = {
                "action": {action_key: denorm},
                "state": {
                    state_key: torch.as_tensor(
                        state_vec, dtype=torch.float32
                    ).reshape(1, 1, -1)
                },
            }
            for trans in reversed(transforms):
                batch = trans.backward(batch)
            denorm = batch["action"][action_key]
    return denorm.numpy()[0]
