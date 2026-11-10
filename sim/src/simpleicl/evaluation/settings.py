"""Parse evaluation sampling settings and explicit policy arguments."""

from collections.abc import Mapping
from typing import Any


def resolve_video_sampling(
    cfg: Any, overrides: Mapping[str, Any] | None = None
) -> tuple[int | None, float | None, int | None]:
    """Parse values without replacing an explicit null with another default."""
    values = []
    for key, convert in (
        ("video_all_num_frames", int),
        ("video_all_frame_percentage", float),
        ("video_all_max_num_frames", int),
    ):
        if overrides is not None and key in overrides:
            value = overrides[key]
        else:
            value = cfg.EVALUATION.get(key)
        if value is None or (
            isinstance(value, str) and value.strip().lower() in {"", "none", "null"}
        ):
            values.append(None)
        else:
            values.append(convert(value))

    num_frames, percentage, max_frames = values
    if num_frames is None and percentage is None:
        raise ValueError(
            "video_all requires either video_all_num_frames or "
            "video_all_frame_percentage"
        )
    return num_frames, percentage, max_frames
