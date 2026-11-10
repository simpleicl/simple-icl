"""Helpers for temporal sampling of whole-video conditioning inputs."""

import math
from typing import Optional


def validate_frame_percentage(value: float) -> float:
    """Validate and normalize a frame percentage represented as a fraction."""
    percentage = float(value)
    if not math.isfinite(percentage) or not 0.0 < percentage <= 1.0:
        raise ValueError(
            f"frame percentage must be finite and in (0, 1], got {value!r}"
        )
    return percentage


def align_frame_count_to_vae(
    frame_count: int,
    *,
    temporal_factor: int = 4,
    min_num_frames: int = 2,
    max_num_frames: Optional[int] = None,
) -> int:
    """Round a requested frame count to the nearest ``4n+1``-style length.

    The lower valid length is used only when it is strictly closer. Ties use
    the upper length so short clips keep at least ``min_num_frames`` frames.
    ``max_num_frames`` limits the result to a count that can be sampled without
    duplicating frames.
    """
    frame_count = int(frame_count)
    temporal_factor = int(temporal_factor)
    min_num_frames = int(min_num_frames)
    if frame_count < 1:
        raise ValueError(f"frame_count must be positive, got {frame_count}")
    if temporal_factor < 1:
        raise ValueError(
            f"temporal_factor must be positive, got {temporal_factor}"
        )
    if min_num_frames < 1:
        raise ValueError(f"min_num_frames must be positive, got {min_num_frames}")
    if max_num_frames is not None:
        max_num_frames = int(max_num_frames)
        if max_num_frames < 1:
            raise ValueError(
                f"max_num_frames must be positive, got {max_num_frames}"
            )

    first_valid = 1 + max(
        0, math.ceil((min_num_frames - 1) / temporal_factor)
    ) * temporal_factor
    last_valid = None
    if max_num_frames is not None:
        last_valid = 1 + ((max_num_frames - 1) // temporal_factor) * temporal_factor
        if last_valid < first_valid:
            raise ValueError(
                "No valid VAE frame count is available within the requested range: "
                f"min_num_frames={min_num_frames}, max_num_frames={max_num_frames}, "
                f"temporal_factor={temporal_factor}."
            )

    lower = 1 + ((frame_count - 1) // temporal_factor) * temporal_factor
    lower = max(lower, first_valid)
    if last_valid is not None and lower > last_valid:
        return last_valid
    if lower > frame_count:
        upper = lower
        return upper

    upper = lower if lower == frame_count else lower + temporal_factor
    if last_valid is not None and upper > last_valid:
        return lower
    lower_distance = frame_count - lower
    upper_distance = upper - frame_count
    return lower if lower_distance < upper_distance else upper


def resolve_video_frame_count(
    total_frames: Optional[int],
    *,
    frame_percentage: Optional[float] = None,
    num_frames: Optional[int] = None,
    max_num_frames: Optional[int] = None,
    min_num_frames: int = 2,
) -> Optional[int]:
    """Resolve the number of frames to retain from a video.

    ``frame_percentage`` takes precedence over the legacy fixed ``num_frames``
    setting. When metadata does not provide ``total_frames``, a percentage-based
    count is resolved after decoding the stream.
    """
    if min_num_frames < 1:
        raise ValueError(f"min_num_frames must be positive, got {min_num_frames}")
    if max_num_frames is not None:
        max_num_frames = int(max_num_frames)
        if max_num_frames < min_num_frames:
            raise ValueError(
                "max_num_frames must be >= min_num_frames, "
                f"got {max_num_frames} < {min_num_frames}"
            )

    if frame_percentage is not None:
        percentage = validate_frame_percentage(frame_percentage)
        if total_frames is None:
            return None
        total_frames = int(total_frames)
        if total_frames < 1:
            raise ValueError(f"total_frames must be positive, got {total_frames}")
        target = max(min_num_frames, int(math.ceil(total_frames * percentage)))
    elif num_frames is not None:
        target = int(num_frames)
        if target < min_num_frames:
            raise ValueError(
                f"num_frames must be >= {min_num_frames}, got {target}"
            )
    else:
        return None

    if max_num_frames is not None:
        target = min(target, max_num_frames)
    return target
