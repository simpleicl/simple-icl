#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import argparse
import csv
import glob
import json
import os, shutil
import sys
import tempfile
import traceback
from pathlib import Path

# build_conditions_list.py lives next to this script; the scene-column resolution rule is
# defined there once and reused by fill_plan_scenes below.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_conditions_list import resolve_scene_cells  # noqa: E402

# Heavy dependencies (datasets / lerobot) are only needed for the video-conversion path.
# They are imported lazily so lightweight helpers such as `update_episode_tasks_from_plan`
# still work in environments where the lerobot import chain is unavailable.
try:
    import numpy as np
    from datasets import load_dataset

    from lerobot.common.datasets.compute_stats import compute_episode_stats
    from lerobot.common.datasets.utils import (
        DEFAULT_IMAGE_PATH,
        DEFAULT_VIDEO_PATH,
        EPISODES_STATS_PATH,
        load_jsonlines,
        write_episode_stats,
        write_json,
    )
    from lerobot.common.datasets.video_utils import encode_video_frames, get_video_info
except ImportError as _import_exc:  # pragma: no cover - depends on runtime environment
    _CONVERT_IMPORT_ERROR = _import_exc
else:
    _CONVERT_IMPORT_ERROR = None


def _format_episode_dir(ep_idx: int) -> str:
    return f"episode_{ep_idx:06d}"


def _episode_image_paths(dataset_dir: Path, cam_key: str, ep_idx: int, num_frames: int) -> list[str]:
    return [
        str(dataset_dir / DEFAULT_IMAGE_PATH.format(image_key=cam_key, episode_index=ep_idx, frame_index=i))
        for i in range(num_frames)
    ]


def _compute_episode_stats(
    dataset_dir: Path, parquet_path: Path, features: dict, camera_keys: list[str], ep_idx: int
) -> dict:
    """Build the per-episode data dict and compute its stats.

    Non-image features are read from the episode parquet; image features are read from the
    frames stored on disk (which must still exist, i.e. before the images/ folder is removed).
    """
    episode_ds = load_dataset("parquet", data_files=str(parquet_path), split="train")
    num_frames = episode_ds.num_rows

    episode_data: dict = {}
    for key, ft in features.items():
        if ft["dtype"] in ["image", "video"]:
            # `compute_episode_stats` expects a list of image paths for visual features.
            episode_data[key] = _episode_image_paths(dataset_dir, key, ep_idx, num_frames)
        elif key in episode_ds.column_names:
            episode_data[key] = np.array(episode_ds[key])

    return compute_episode_stats(episode_data, features)


def _get_episode_parquet_path(dataset_dir: Path, data_path_template: str, ep_idx: int, chunk_size: int) -> Path:
    ep_chunk = ep_idx // chunk_size
    rel_path = data_path_template.format(episode_chunk=ep_chunk, episode_index=ep_idx)
    return dataset_dir / rel_path


def _get_episode_video_path(dataset_dir: Path, video_path_template: str, ep_idx: int, cam_key: str, chunk_size: int) -> Path:
    ep_chunk = ep_idx // chunk_size
    rel_path = video_path_template.format(
        episode_chunk=ep_chunk,
        video_key=cam_key,
        episode_index=ep_idx,
    )
    return dataset_dir / rel_path


class _Tee:
    """Write to several streams at once (e.g. the real stdout and a log file)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


class _TeeLogger:
    """Duplicate everything written to stdout/stderr into ``log_path`` for the context duration."""

    def __init__(self, log_path: Path):
        self.log_path = Path(log_path)
        self._file = None
        self._orig_stdout = None
        self._orig_stderr = None

    def __enter__(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.log_path, "a", encoding="utf-8")
        self._orig_stdout, self._orig_stderr = sys.stdout, sys.stderr
        sys.stdout = _Tee(self._orig_stdout, self._file)
        sys.stderr = _Tee(self._orig_stderr, self._file)
        return self

    def __exit__(self, exc_type, exc, tb):
        # Make sure a propagating exception is also recorded in the log.
        if exc is not None and self._file is not None:
            self._file.write("".join(traceback.format_exception(exc_type, exc, tb)))
        sys.stdout, sys.stderr = self._orig_stdout, self._orig_stderr
        if self._file is not None:
            self._file.flush()
            self._file.close()
        return False


def _is_truncated_image_error(exc: BaseException) -> bool:
    """Heuristically detect a truncated / incomplete image error (PIL / struct)."""
    tb_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))

    messages = []
    cur, seen = exc, set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        messages.append(f"{type(cur).__name__}: {cur}")
        cur = cur.__cause__ or cur.__context__

    haystack = (tb_text + "\n" + "\n".join(messages)).lower()
    markers = [
        "image file is truncated",
        "truncated",
        "buffer of at least",
        "unpack_from requires a buffer",
    ]
    return any(m in haystack for m in markers)


def _remove_truncated_images(img_dir: Path) -> list[str]:
    """Delete every frame in ``img_dir`` that PIL cannot fully decode; return their paths."""
    from PIL import Image

    img_dir = Path(img_dir)
    template = "frame_" + ("[0-9]" * 6) + ".png"
    removed: list[str] = []
    for p in sorted(glob.glob(str(img_dir / template))):
        ok = True
        try:
            with Image.open(p) as im:
                im.convert("RGB")  # force a full decode so truncation raises
        except Exception:
            ok = False
        if not ok:
            try:
                Path(p).unlink()
                removed.append(p)
            except OSError:
                pass
    return removed


def _write_first_frame_video(src_video: Path, dst_video: Path, fps: int) -> None:
    """Encode a single-frame placeholder video at ``dst_video`` from ``src_video``'s first frame."""
    import av

    src_video = Path(src_video)
    frame_image = None
    with av.open(str(src_video)) as container:
        for frame in container.decode(video=0):
            frame_image = frame.to_image()
            break
    if frame_image is None:
        raise RuntimeError(f"Could not decode any frame from {src_video}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        frame_image.save(tmp_dir / "frame_000000.png")
        encode_video_frames(imgs_dir=tmp_dir, video_path=dst_video, fps=fps, overwrite=True)


def _handle_encode_failure(
    exc: BaseException,
    img_dir: Path,
    video_path: Path,
    last_good_video_path: Path | None,
    images_error_dir: Path,
    cam_key: str,
    ep_idx: int,
    fps: int,
) -> Path | None:
    """Recover from an ``encode_video_frames`` failure.

    Returns the (possibly updated) ``last_good_video_path``: it is advanced to ``video_path``
    only when a valid video was actually (re-)encoded here.
    """
    # 1) Log the error.
    print(f"[convert] ERROR encoding video for cam='{cam_key}' episode={ep_idx}: {video_path}", file=sys.stderr)
    print("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)), file=sys.stderr)

    # 2) Save the scene: copy the offending image folder to images_error/.
    error_dir = images_error_dir / cam_key / _format_episode_dir(ep_idx)
    if error_dir.exists():
        shutil.rmtree(error_dir)
    error_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(img_dir, error_dir)
    print(f"[convert] Saved offending images to: {error_dir}", file=sys.stderr)

    # 3) Fallback: make video_path a 1-frame video from the previous video's first frame.
    if last_good_video_path is not None and Path(last_good_video_path).exists():
        try:
            _write_first_frame_video(last_good_video_path, video_path, fps)
            print(f"[convert] Wrote placeholder video from previous first frame: {video_path}", file=sys.stderr)
        except Exception as fe:
            print(f"[convert] Failed to write placeholder video: {fe}", file=sys.stderr)
    else:
        print("[convert] No previous video available for placeholder fallback.", file=sys.stderr)

    # 4) Special case: truncated / incomplete images. Clean img_dir and re-encode.
    #    This runs *after* saving the scene and logging above.
    if _is_truncated_image_error(exc):
        removed = _remove_truncated_images(img_dir)
        print(f"[convert] Removed {len(removed)} truncated image(s) from {img_dir}.", file=sys.stderr)
        for p in removed:
            print(f"[convert]   removed: {p}", file=sys.stderr)
        try:
            encode_video_frames(imgs_dir=img_dir, video_path=video_path, fps=fps, overwrite=True)
            print(f"[convert] Re-encoded video after removing truncated images: {video_path}", file=sys.stderr)
            return video_path
        except Exception as re_exc:
            print(f"[convert] Re-encode after cleaning still failed: {re_exc}", file=sys.stderr)
            print("".join(traceback.format_exception(type(re_exc), re_exc, re_exc.__traceback__)), file=sys.stderr)

    return last_good_video_path


def convert_image_dataset_to_video(src_dataset: Path, dst_dataset: Path, keep_images: bool = False) -> None:
    if _CONVERT_IMPORT_ERROR is not None:
        raise ImportError(
            "Video conversion requires the datasets/lerobot dependencies, which failed to import: "
            f"{_CONVERT_IMPORT_ERROR}"
        ) from _CONVERT_IMPORT_ERROR

    if not src_dataset.exists():
        raise FileNotFoundError(f"Source dataset not found: {src_dataset}")

    dst_existed = dst_dataset.exists()
    dst_dataset.mkdir(parents=True, exist_ok=True)

    # Capture all terminal output of the conversion into dst_dataset/convert.log.
    with _TeeLogger(dst_dataset / "convert.log"):
        if dst_existed:
            print(f"Warning: Destination already exists and will be overwritten: {dst_dataset}")
        _convert_src_to_dst(src_dataset, dst_dataset, keep_images)


# Top-level entries that are produced by the conversion (or too bulky to copy) and therefore
# must NOT be copied verbatim from src to dst:
#   - images:   read frame-by-frame from src; only re-encoded into videos/ (never copied).
#   - data:     rewritten per-episode into dst (image columns dropped).
#   - videos:   encoded into dst from the src frames.
#   - convert.log / images_error: conversion artifacts that belong to the destination only.
_CONVERT_PRODUCED_ENTRIES = {"images", "data", "videos", "convert.log", "images_error"}


def _populate_dst_auxiliary(src_dataset: Path, dst_dataset: Path) -> None:
    """Copy the small, non-converted supporting files (meta/, plan.csv, README, ...) into dst.

    The bulky/produced entries listed in ``_CONVERT_PRODUCED_ENTRIES`` are skipped: their content
    is read from ``src`` and written into ``dst`` by the conversion steps below.
    """
    for entry in sorted(src_dataset.iterdir()):
        if entry.name in _CONVERT_PRODUCED_ENTRIES:
            continue
        dst_entry = dst_dataset / entry.name
        if entry.is_dir():
            if dst_entry.exists():
                shutil.rmtree(dst_entry)
            shutil.copytree(entry, dst_entry)
        else:
            shutil.copy2(entry, dst_entry)


def _convert_src_to_dst(src_dataset: Path, dst_dataset: Path, keep_images: bool = False) -> None:
    info = json.loads((src_dataset / "meta" / "info.json").read_text(encoding="utf-8"))

    camera_keys = [k for k, ft in info["features"].items() if ft["dtype"] == "image"]
    if not camera_keys:
        raise ValueError("No image camera keys found. Dataset does not look like use_videos=False.")

    total_episodes = info["total_episodes"]
    fps = int(info["fps"])
    chunk_size = int(info["chunks_size"])
    episodes_meta = load_jsonlines(src_dataset / "meta" / "episodes.jsonl")
    episode_indices = [int(ep["episode_index"]) for ep in episodes_meta]

    # 0) Populate dst with the small supporting files (meta/, plan.csv, ...). info.json and
    #    episodes_stats.jsonl are (re)written below; everything else is used as-is.
    _populate_dst_auxiliary(src_dataset, dst_dataset)
    info_path = dst_dataset / "meta" / "info.json"

    # 1) Encode videos into dst from the src image folders.
    images_error_dir = dst_dataset / "images_error"
    last_good_video_path: Path | None = None
    for ep_idx in episode_indices:
        for cam_key in camera_keys:
            img_dir = src_dataset / "images" / cam_key / _format_episode_dir(ep_idx)
            if not img_dir.exists():
                raise FileNotFoundError(f"Missing image directory: {img_dir}")
            video_path = _get_episode_video_path(
                dataset_dir=dst_dataset,
                video_path_template=DEFAULT_VIDEO_PATH,
                ep_idx=ep_idx,
                cam_key=cam_key,
                chunk_size=chunk_size,
            )
            if os.path.exists(video_path):
                print(f"Warning: Video already exists: {video_path}")
                continue
            try:
                encode_video_frames(imgs_dir=img_dir, video_path=video_path, fps=fps, overwrite=True)
                last_good_video_path = video_path
            except Exception as exc:
                last_good_video_path = _handle_encode_failure(
                    exc=exc,
                    img_dir=img_dir,
                    video_path=video_path,
                    last_good_video_path=last_good_video_path,
                    images_error_dir=images_error_dir,
                    cam_key=cam_key,
                    ep_idx=ep_idx,
                    fps=fps,
                )

    # 2) Write each episode parquet into dst, removing image columns (read from src).
    for ep_idx in episode_indices:
        src_parquet = _get_episode_parquet_path(
            dataset_dir=src_dataset,
            data_path_template=info["data_path"],
            ep_idx=ep_idx,
            chunk_size=chunk_size,
        )
        dst_parquet = _get_episode_parquet_path(
            dataset_dir=dst_dataset,
            data_path_template=info["data_path"],
            ep_idx=ep_idx,
            chunk_size=chunk_size,
        )

        episode_ds = load_dataset("parquet", data_files=str(src_parquet), split="train")
        drop_cols = [k for k in camera_keys if k in episode_ds.column_names]
        if drop_cols:
            episode_ds = episode_ds.remove_columns(drop_cols)
        dst_parquet.parent.mkdir(parents=True, exist_ok=True)
        episode_ds.to_parquet(str(dst_parquet))

    # 3) Update info.json to video mode (written into dst).
    info["video_path"] = DEFAULT_VIDEO_PATH
    info["total_videos"] = total_episodes * len(camera_keys)
    for cam_key in camera_keys:
        info["features"][cam_key]["dtype"] = "video"
        info["features"][cam_key]["info"] = None

    if episode_indices:
        first_ep = episode_indices[0]
        for cam_key in camera_keys:
            video_path = _get_episode_video_path(
                dataset_dir=dst_dataset,
                video_path_template=DEFAULT_VIDEO_PATH,
                ep_idx=first_ep,
                cam_key=cam_key,
                chunk_size=chunk_size,
            )
            info["features"][cam_key]["info"] = get_video_info(video_path)

    write_json(info, info_path)

    # 3b) Compute per-episode stats (skipped during recording) and write episodes_stats.jsonl.
    # Image frames are read from src (which retains them); results are written into dst.
    stats_path = dst_dataset / EPISODES_STATS_PATH
    if stats_path.exists():
        stats_path.unlink()  # avoid stale/duplicate entries on re-runs
    for ep_idx in episode_indices:
        src_parquet = _get_episode_parquet_path(
            dataset_dir=src_dataset,
            data_path_template=info["data_path"],
            ep_idx=ep_idx,
            chunk_size=chunk_size,
        )
        ep_stats = _compute_episode_stats(
            dataset_dir=src_dataset,
            parquet_path=src_parquet,
            features=info["features"],
            camera_keys=camera_keys,
            ep_idx=ep_idx,
        )
        write_episode_stats(ep_idx, ep_stats, dst_dataset)

    if keep_images:
        src_images = src_dataset / "images"
        dst_images = dst_dataset / "images"
        if src_images.exists() and not dst_images.exists():
            shutil.copytree(src_images, dst_images)


def fill_plan_scenes(plan_csv: Path, out_csv: str | None = None) -> None:
    """Fill in the empty ``scene`` column of ``plan.csv`` in place (or to ``out_csv``).

    The per-row resolution rule lives in ``build_conditions_list.resolve_scene_cells``
    (shared with split_dataset.py so scenes.json and this CSV never disagree): a row's
    ``scene`` becomes the nearest non-empty one above it, and a placement marker
    ("_keep" / "_alt") is prefixed with the nearest real scene name above it.

    Rows whose own ``scene`` is already a full scene name are left untouched, so re-running
    this on an already-filled CSV is a no-op.
    """
    if not plan_csv.exists():
        print(f"[plan] plan.csv not found, skipping task update: {plan_csv}")
        return
    out_csv = Path(out_csv) if out_csv is not None else plan_csv

    # utf-8-sig strips (and, on write, restores) the BOM the CSV is exported with.
    with open(plan_csv, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    if not fieldnames or "scene" not in fieldnames:
        raise ValueError(f"plan.csv has no 'scene' column (columns: {fieldnames})")

    resolved = resolve_scene_cells([row.get("scene") for row in rows])

    n_filled = 0
    for row, new_scene in zip(rows, resolved):
        if new_scene != (row.get("scene") or "").strip():
            n_filled += 1
        row["scene"] = new_scene

    with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[plan] filled/updated {n_filled}/{len(rows)} scene cell(s) -> {out_csv}")


def _load_plan_action_map(plan_csv: Path) -> list[tuple[str, str]]:
    """Parse ``plan.csv`` into a list of (normalized parquet path, action) pairs.

    Only rows that carry both a ``data_path`` (parquet path) and an ``action`` are kept.
    Paths are normalized to POSIX form so they can be matched regardless of separator.
    """
    pairs: list[tuple[str, str]] = []
    # utf-8-sig strips the BOM that the CSV is exported with (﻿ before `index`).
    with open(plan_csv, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data_path = (row.get("data_path") or "").strip()
            action = (row.get("action") or "").strip()
            if not data_path or not action:
                continue
            norm = str(data_path).replace("\\", "/").strip("/")
            pairs.append((norm, action))
    return pairs


def _find_plan_action(plan_pairs: list[tuple[str, str]], dataset_name: str, rel_parquet: str) -> str | None:
    """Return the ``action`` for an episode whose parquet matches a ``data_path`` entry.

    An episode's parquet is identified by ``<dataset_name>/<rel_parquet>``.
    """
    key = f"{dataset_name}/{rel_parquet}".replace("\\", "/").strip("/")
    for norm, action in plan_pairs:
        if norm == key or norm.endswith("/" + key):
            return action
    return None


def update_episode_tasks_from_plan(dataset_dir: Path, plan_csv: Path | None = None) -> None:
    """Overwrite each episode's task with the matching ``action`` from ``plan.csv``.

    For every episode in ``meta/episodes.jsonl`` we locate its parquet file. If that parquet
    path is found in the plan's ``data_path`` column, the episode's ``tasks`` is replaced with the
    corresponding ``action``. ``meta/tasks.jsonl`` is then rebuilt from the resulting task strings.
    Episodes with no matching plan row keep their existing task.
    """
    dataset_dir = Path(dataset_dir)
    plan_csv = Path(plan_csv) if plan_csv is not None else dataset_dir / "plan.csv"
    if not plan_csv.exists():
        print(f"[plan] plan.csv not found, skipping task update: {plan_csv}")
        return

    info_path = dataset_dir / "meta" / "info.json"
    episodes_path = dataset_dir / "meta" / "episodes.jsonl"
    tasks_path = dataset_dir / "meta" / "tasks.jsonl"

    info = json.loads(info_path.read_text(encoding="utf-8"))
    chunk_size = int(info["chunks_size"])
    data_path_template = info["data_path"]
    dataset_name = dataset_dir.resolve().name

    plan_pairs = _load_plan_action_map(plan_csv)
    with open(episodes_path, encoding="utf-8") as f:
        episodes_meta = [json.loads(line) for line in f if line.strip()]

    n_matched = 0
    for ep in episodes_meta:
        ep_idx = int(ep["episode_index"])
        ep_chunk = ep_idx // chunk_size
        rel_parquet = data_path_template.format(episode_chunk=ep_chunk, episode_index=ep_idx)
        action = _find_plan_action(plan_pairs, dataset_name, rel_parquet)
        if action is None:
            print(f"[plan] episode {ep_idx}: no plan match for {rel_parquet}, keeping {ep.get('tasks')}")
            continue
        ep["tasks"] = [action]
        n_matched += 1
        print(f"[plan] episode {ep_idx}: task -> {action!r}")

    # Rewrite episodes.jsonl with the updated tasks.
    with open(episodes_path, "w", encoding="utf-8") as f:
        for ep in episodes_meta:
            f.write(json.dumps(ep, ensure_ascii=False) + "\n")

    # Rebuild tasks.jsonl from the (possibly updated) task strings, in first-seen order.
    task_to_index: dict[str, int] = {}
    for ep in episodes_meta:
        for task in ep.get("tasks", []):
            if task not in task_to_index:
                task_to_index[task] = len(task_to_index)
    with open(tasks_path, "w", encoding="utf-8") as f:
        for task, idx in task_to_index.items():
            f.write(json.dumps({"task_index": idx, "task": task}, ensure_ascii=False) + "\n")

    print(f"[plan] updated {n_matched}/{len(episodes_meta)} episode(s); {len(task_to_index)} task(s) in tasks.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert LeRobot image dataset (use_videos=False) to video dataset")
    parser.add_argument("--src_dataset", type=Path, help="Source dataset path, e.g. outputs/1015")
    parser.add_argument("--dst_dataset", type=Path, help="Destination dataset path, e.g. outputs/videos/1015")
    parser.add_argument(
        "--keep_images",
        action="store_true",
        help="Keep images folder in destination after encoding videos",
    )
    parser.add_argument(
        "--update_tasks_only",
        action="store_true",
        help="Only update meta/episodes.jsonl and meta/tasks.jsonl from plan.csv; skip video conversion.",
    )
    parser.add_argument(
        "--fill_scenes_only",
        action="store_true",
        help="Only fill the empty scene column of plan.csv (in place); skip everything else.",
    )

    args = parser.parse_args()
    dataset_dir = args.src_dataset
    if dataset_dir is None:
        parser.error("--src_dataset is required")
    plan_csv = dataset_dir / "plan.csv"
    if not plan_csv.exists():
        print(f"[plan] plan.csv not found, skipping task update: {plan_csv}")

    if args.fill_scenes_only:
        fill_plan_scenes(plan_csv=plan_csv, out_csv=str(plan_csv).replace(".csv", "_filled.csv"))
        return

    if args.update_tasks_only:
        update_episode_tasks_from_plan(dataset_dir=dataset_dir, plan_csv=plan_csv)
        return

    if args.src_dataset is None or args.dst_dataset is None:
        parser.error("--src_dataset and --dst_dataset are required")

    convert_image_dataset_to_video(
        src_dataset=args.src_dataset,
        dst_dataset=args.dst_dataset,
        keep_images=args.keep_images,
    )
    plan_csv = args.dst_dataset / "plan.csv"
    update_episode_tasks_from_plan(dataset_dir=args.dst_dataset, plan_csv=plan_csv)
    fill_plan_scenes(plan_csv=plan_csv, out_csv=str(plan_csv).replace(".csv", "_filled.csv"))


if __name__ == "__main__":
    main()