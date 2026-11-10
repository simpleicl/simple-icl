import os
from typing import Iterable

import imageio
import numpy as np
from PIL import Image

from .fs import ensure_dir


def _to_even_frame(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    pad_h = h % 2
    pad_w = w % 2
    if pad_h == 0 and pad_w == 0:
        return frame
    return np.pad(frame, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")


def save_mp4(frames: Iterable[Image.Image], path: str, fps: int = 8):
    ensure_dir(os.path.dirname(path) or ".")
    writer = imageio.get_writer(
        path,
        fps=max(fps, 1),
        codec="libx264",
        format="FFMPEG",
        pixelformat="yuv420p",
    )
    try:
        for frame in frames:
            arr = np.array(frame.convert("RGB"))
            writer.append_data(_to_even_frame(arr))
    finally:
        writer.close()

def save_video_to_mp4(
    video,
    path: str,
    fps: int = 8,
    value_range: str = "auto",
):
    """Save a video to MP4 from various input formats.

    Supports:
      - torch.Tensor: [T,H,W,C], [C,T,H,W], [B,C,T,H,W], [B,T,H,W,C]
      - np.ndarray:   [T,H,W,C], [C,T,H,W], [B,C,T,H,W], [B,T,H,W,C]
      - list[Image.Image] or list[np.ndarray]

    Args:
        video: Video data in any of the above formats.
        path: Output MP4 file path.
        fps: Frames per second.
        value_range: One of "auto", "[0,1]", "[-1,1]", "[0,255]".
            "auto" detects the range from min/max values.
    """
    import torch

    # --- Convert to list of uint8 RGB frames [H, W, 3] ---
    if isinstance(video, (list, tuple)):
        frames = []
        for item in video:
            if isinstance(item, Image.Image):
                frames.append(np.array(item.convert("RGB")))
            elif isinstance(item, np.ndarray):
                frames.append(item.astype(np.uint8) if item.dtype == np.uint8 else item)
            elif isinstance(item, torch.Tensor):
                arr = item.float().cpu().numpy()
                if arr.ndim == 3 and arr.shape[0] == 3:
                    arr = np.transpose(arr, (1, 2, 0))
                frames.append(arr)
            else:
                raise TypeError(f"Unsupported frame type in list: {type(item)}")
    elif isinstance(video, torch.Tensor):
        arr = video.float().cpu().numpy()
        frames = _tensor_to_frames(arr, value_range)
    elif isinstance(video, np.ndarray):
        frames = _ndarray_to_frames(video, value_range)
    else:
        raise TypeError(f"Unsupported video type: {type(video)}")

    if len(frames) == 0:
        raise ValueError("No frames to save.")

    # Normalize all frames to uint8 [H, W, 3]
    normalized_frames = []
    for frame in frames:
        if isinstance(frame, Image.Image):
            normalized_frames.append(np.array(frame.convert("RGB")))
        elif frame.dtype == np.uint8:
            # Already [0, 255]. The float heuristic below would mistake a dark
            # uint8 frame (max <= 10) for [0,1] data and scale it to white.
            normalized_frames.append(frame)
        else:
            f = frame.astype(np.float32)
            if f.max() <= 10.0:
                if f.min() < 0.0:
                    f = (f + 1.0) * 127.5
                else:
                    f = f * 255.0
            normalized_frames.append(np.clip(f, 0, 255).astype(np.uint8))

    pil_frames = [Image.fromarray(f) for f in normalized_frames]
    save_mp4(pil_frames, path, fps=fps)


def _detect_value_range(arr: np.ndarray) -> str:
    min_v, max_v = float(arr.min()), float(arr.max())
    if max_v > 10.0:
        return "[0,255]"
    if min_v < 0.0:
        return "[-1,1]"
    return "[0,1]"


def _normalize_to_thwc(arr: np.ndarray, value_range: str) -> np.ndarray:
    """Convert any common video array layout to [T, H, W, C=3] uint8."""
    if value_range == "auto":
        value_range = _detect_value_range(arr)

    f = arr.astype(np.float32, copy=False)
    if value_range == "[-1,1]":
        f = (f + 1.0) * 127.5
    elif value_range == "[0,1]":
        f = f * 255.0
    # "[0,255]" needs no scaling

    f = np.clip(f, 0, 255).astype(np.uint8)

    # Ensure [T, H, W, C]
    if f.ndim == 4:
        if f.shape[-1] == 3:
            return f  # already THWC
        if f.shape[0] == 3:
            return np.transpose(f, (1, 2, 3, 0))  # CTHW -> THWC
        if f.shape[1] == 3:
            return np.transpose(f, (0, 2, 3, 1))  # TCHW -> THWC
    raise ValueError(f"Cannot interpret array shape {arr.shape} as video frames.")


def _tensor_to_frames(arr: np.ndarray, value_range: str) -> list[np.ndarray]:
    """Convert numpy array (from tensor) to list of [H,W,3] frames."""
    if arr.ndim == 5:
        # [B, C, T, H, W] or [B, T, H, W, C] — take first batch
        arr = arr[0]
    thwc = _normalize_to_thwc(arr, value_range)
    return [thwc[t] for t in range(thwc.shape[0])]


def _ndarray_to_frames(arr: np.ndarray, value_range: str) -> list[np.ndarray]:
    """Convert numpy array to list of [H,W,3] frames."""
    if arr.ndim == 5:
        arr = arr[0]
    thwc = _normalize_to_thwc(arr, value_range)
    return [thwc[t] for t in range(thwc.shape[0])]
