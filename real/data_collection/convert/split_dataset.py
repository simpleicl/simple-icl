#!/usr/bin/env python3
"""Split a LeRobot (v2.1) dataset into two independent, format-complete datasets
by episode task label.

Episodes are classified by their task label in episodes.jsonl:

    task starts with robot_ (robot-arm prefix)  -> <root>/robot
    task starts with human_ (human-hand prefix) -> <root>/human
    task starts with neither                    -> discarded

Pairing follows the source ordering: each robot episode is paired with the
human episode that immediately follows it (the `robot_`/`human_` prefix is
stripped before comparing the two sides). A pair
whose task suffixes disagree is flagged; an episode that cannot be paired is
discarded.

Explicit drops: source episode indices passed to --drop-episodes are discarded
outright, before any other filtering. As with the duplicate check, listing either
side of a pair drops the WHOLE pair.

Duplicate first frames: adjacent source episodes are compared on their first
frames. An episode is a duplicate of the previous one when ANY camera's first
frame is (near-)identical (mean abs pixel diff < --dup-threshold) -- a real
recording glitch, not the naturally-similar same-scene robot/human boundary.
If either episode of a pair is a duplicate, the WHOLE pair is dropped.

Each output is a self-contained LeRobot dataset: episodes are re-indexed to a
consecutive 0..N-1 range, parquet `episode_index`/`index` columns are rewritten,
video files are copied & renamed, and every meta json (info/tasks/episodes/
episodes_stats) is regenerated to match.

Data semantics (used by the validity checks):
    - robot: the robot arm performs the task -> action/state must MOVE.
    - human: a human performs the same task in the same scene while the
      arm is held still -> action is frozen and observation.state only drifts by
      small perturbations. A real motion here means the episode is mislabelled.

Checks (all reported as warnings, never silently dropped):
    1. human episodes must be nearly static (action frozen, state within a small
       perturbation tolerance); a non-perturbation change is flagged.
    2. any episode shorter than its per-kind minimum (--min-frames-robot /
       --min-frames-human) is flagged.
    3. extra integrity checks:
       - robot episodes are expected to move (a static one means the arm did not
         execute) -> flagged.
       - a robot task may name the arm that executes it ("left..." /
         "right..."), so the OTHER arm must stay nearly still: for a left task
         the second half of the action vector, for a right task the first half
         (the split is action_dim // 2, so it works for both 14- and 16-dim
         actions) -> flagged.
       - parquet <-> meta <-> video presence, and non-empty video files.
       - frame_index is a contiguous 0..len-1 range; parquet length matches
         meta length.
       - timestamp is consistent with frame_index / fps.
       - action / observation.state row width matches the feature shape.
       - task_index values exist in tasks.jsonl.

A robot/meta/scenes.json is also written, recording which scene each robot
episode belongs to, plus a robot/meta/plan_rows.json recording which row of the
original plan CSV it came from. Both are looked up from the source plan CSV: an
episode's pre-conversion parquet path is matched against the "data_path"
column (anchored on the source dataset name, since the plan may list several
datasets with overlapping episode indices) and that row's "scene" (resolved, so
placement markers like "_alt" become "<scene name>_alt") and row number are
recorded. An episode with no matching "data_path" row is a hard error (it should
have been filtered out earlier).

robot/meta/conditions_list.json is then built from scenes.json + plan_rows.json +
conditions.json (see build_conditions_list.py for the grouping rule).

Oversized condition groups: an episode whose conditions_list would hold more than
--max-conditions (default 3) entries is discarded (whole pair, as with every other
drop). Such a group means the grouping key did not discriminate -- usually a
scene cell that resolved to a bare "_alt"/"_keep" marker, leaving an empty base scene
so every such episode lands in one group. The list is predicted before anything is
written, so survivors are re-indexed once and all meta stays consistent.

All outputs (the robot/ and human/ datasets, split.log, pairing.csv) are written
under a single --root directory.

Usage:
    python split_dataset.py --src outputs/videos/1015 --root outputs/split_new/1015
"""

import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# build_conditions_list.py lives next to this script; reuse its logic verbatim
# for the post-split step (see build_robot_conditions_list below).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_conditions_list import (  # noqa: E402
    build_conditions_list as _build_conditions_list,
    load_episode_tasks as _load_episode_tasks,
    resolve_scene_cells as _resolve_scene_cells,
)


# ----------------------------------------------------------------------------- helpers
def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def parse_episode_list(tokens):
    """Turn --drop-episodes tokens into a set of source episode indices.

    Accepts space- or comma-separated values and inclusive `a-b` ranges, e.g.
    `19 883,955 10-12`.
    """
    out = set()
    for tok in tokens or []:
        for part in str(tok).split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                lo, hi = part.split("-", 1)
                out.update(range(int(lo), int(hi) + 1))
            else:
                out.add(int(part))
    return out


def int_stats(arr):
    """Per-column stats in LeRobot's episodes_stats shape (lists of length 1)."""
    arr = np.asarray(arr, dtype=np.float64)
    return {
        "min": [float(arr.min())],
        "max": [float(arr.max())],
        "mean": [float(arr.mean())],
        "std": [float(arr.std())],
        "count": [int(arr.size)],
    }


class Tee:
    """Mirror everything printed to stdout into a log file as well."""

    def __init__(self, stream, log_path):
        self.stream = stream
        self.log = open(log_path, "w")

    def write(self, data):
        self.stream.write(data)
        self.log.write(data)
        self.stream.flush()
        self.log.flush()

    def flush(self):
        self.stream.flush()
        self.log.flush()

    def close(self):
        self.log.close()


class Reporter:
    """Collects warnings and prints a grouped summary at the end."""

    def __init__(self):
        self.warnings = []  # (episode_label, message)

    def warn(self, label, msg):
        self.warnings.append((label, msg))
        print(f"  [WARN] {label}: {msg}")

    def summary(self):
        print("\n" + "=" * 78)
        if not self.warnings:
            print("VALIDATION SUMMARY: no warnings. ✓")
            print("=" * 78)
            return 0
        print(f"VALIDATION SUMMARY: {len(self.warnings)} warning(s)")
        print("=" * 78)
        for label, msg in self.warnings:
            print(f"  - {label}: {msg}")
        print("=" * 78)
        return len(self.warnings)


# ----------------------------------------------------------------------------- checks
def decode_video_frames(path):
    """Fully decode a video with PyAV, returning (ok, n_frames, err).

    Every frame is decoded so any corrupt packet raises here, exactly as the
    training-time torchvision->pyav fallback would. ok=False means the file
    could not be decoded at all.
    """
    try:
        import av
    except ImportError as e:  # pragma: no cover - env issue, not data issue
        return False, 0, f"pyav import failed: {e}"
    n = 0
    try:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            for _ in container.decode(stream):
                n += 1
    except Exception as e:  # noqa: BLE001 - need to catch all decode errors
        return False, n, f"{type(e).__name__}: {e}"
    if n == 0:
        return False, 0, "decoded 0 frames (empty/unreadable stream)"
    return True, n, ""


def task_active_arm(task):
    """Which arm a task label says executes it: "left" for left..., "right" for
    right...

    Returns None when the task starts with neither (the arm is then unknown and
    the idle-arm check cannot be applied).
    """
    if "left" in task and "right" in task:
        return "bimanual"
    if task.startswith("left"):
        return "left"
    if task.startswith("right"):
        return "right"
    return None


def check_episode(df, src, src_idx, meta_len, info, tasks_set, is_human,
                  video_keys, task, args, rep):
    """Validate one source episode's parquet before it is written out.

    `task` is the episode's task label (from episodes.jsonl); it names which arm
    executes, which drives the idle-arm static check for robot episodes.

    Returns True if the episode raised no warning (passed all checks), False
    otherwise -- the caller uses this to drop a whole pair when either side fails.
    """
    label = f"src ep{src_idx:03d} ({'human' if is_human else 'robot'})"
    n_warn_before = len(rep.warnings)
    fps = info["fps"]
    action_dim = info["features"]["action"]["shape"][0]
    state_dim = info["features"]["observation.state"]["shape"][0]

    n = len(df)

    # --- check 2: too short (per-kind min) ---
    min_frames = args.min_frames_human if is_human else args.min_frames_robot
    if n < min_frames:
        rep.warn(label, f"too short: {n} frames < min {min_frames}")

    # --- length matches meta ---
    if meta_len is not None and n != meta_len:
        rep.warn(label, f"parquet length {n} != meta length {meta_len}")

    # --- frame_index contiguous 0..n-1 ---
    fi = df["frame_index"].to_numpy()
    if not np.array_equal(fi, np.arange(n)):
        rep.warn(label, "frame_index is not a contiguous 0..len-1 range")

    # --- timestamp consistent with frame_index / fps ---
    ts = df["timestamp"].to_numpy()
    expected_ts = fi / fps
    max_drift = float(np.abs(ts - expected_ts).max()) if n else 0.0
    if max_drift > 1.0 / fps:  # more than one frame of drift
        rep.warn(label, f"timestamp drifts from frame_index/fps by {max_drift:.3f}s")

    # --- task_index validity ---
    bad_tasks = sorted(set(df["task_index"].tolist()) - tasks_set)
    if bad_tasks:
        rep.warn(label, f"task_index {bad_tasks} not present in tasks.jsonl")

    # --- action / state width ---
    action = np.stack(df["action"].to_numpy())
    state = np.stack(df["observation.state"].to_numpy())
    if action.shape[1] != action_dim:
        rep.warn(label, f"action width {action.shape[1]} != feature dim {action_dim}")
    if state.shape[1] != state_dim:
        rep.warn(label, f"state width {state.shape[1]} != feature dim {state_dim}")

    # --- video decode: every kept camera must fully decode (no corrupt packet)
    #     and yield exactly `n` frames (same length as action/parquet) ---
    src_chunk_name = f"chunk-{src_idx // 1000:03d}"
    for key in video_keys:
        vp = src / "videos" / src_chunk_name / key / f"episode_{src_idx:06d}.mp4"
        if not vp.exists() or vp.stat().st_size == 0:
            rep.warn(label, f"video missing/empty for {key}")
            continue
        ok, n_dec, err = decode_video_frames(vp)
        if not ok:
            rep.warn(label, f"video {key} failed to decode: {err}")
        elif n_dec != n:
            rep.warn(label,
                     f"video {key} decoded {n_dec} frames != action length {n}")

    # --- motion checks (per-joint peak-to-peak range) ---
    action_ptp = np.ptp(action, axis=0)
    state_ptp = np.ptp(state, axis=0)

    if is_human:
        # --- check 1: human episodes must be nearly static ---
        a_bad = np.where(action_ptp > args.static_action_tol)[0].tolist()
        s_bad = np.where(state_ptp > args.static_state_tol)[0].tolist()
        # action is judged NOT static only when at least two joints move beyond tol
        if len(a_bad) >= 2:
            rep.warn(
                label,
                f"action is NOT static (expected frozen): {len(a_bad)} joints "
                f"{a_bad} range up to {action_ptp.max():.3f} > tol "
                f"{args.static_action_tol}",
            )
        if len(s_bad) >= 2:
            rep.warn(
                label,
                f"observation.state moved beyond perturbation tol: joints {s_bad} "
                f"range up to {state_ptp.max():.3f} > tol {args.static_state_tol}",
            )
    else:
        # --- extra check: robot episodes should actually move ---
        if action_ptp.max() < args.motion_min:
            rep.warn(
                label,
                f"action barely moves (max range {action_ptp.max():.3f} < "
                f"{args.motion_min}); robot may not have executed",
            )

        # --- extra check: the arm the task does NOT name must stay still ---
        # The action vector is [left arm | right arm]; the halves are derived from
        # the actual width (14 -> 7+7, 16 -> 8+8), never hard-coded.
        width = action.shape[1]
        arm = task_active_arm(task or "")
        if arm is None:
            # informational only (goes to split.log via Tee, not the warning count)
            print(label, f"task '{task}' starts with neither 'left' nor "
                            "'right'; cannot tell which arm should stay still")
        elif width % 2:
            rep.warn(label, f"action width {width} is odd; cannot split it into "
                            "a left and a right half")
        else:
            half = width // 2
            if arm == "bimanual":
                if (action_ptp[:half].max() < args.motion_min or
                        action_ptp[half:].max() < args.motion_min):
                    rep.warn(
                        label,
                        f"bimanual task '{task}' requires both arms to move",
                    )
            else:
                offset = half if arm == "left" else 0
                idle_name = "right" if arm == "left" else "left"
                idle_ptp = action_ptp[offset:offset + half]
                bad = (np.where(idle_ptp > args.idle_arm_tol)[0] + offset).tolist()
                if bad:
                    rep.warn(
                        label,
                        f"task '{task}' uses the {arm} arm, but the {idle_name} half "
                        f"of action is NOT static: joints {bad} range up to "
                        f"{idle_ptp.max():.3f} > tol {args.idle_arm_tol}",
                    )

    return len(rep.warnings) == n_warn_before


def episode_passes_check(src, src_idx, is_human, info, ep_len, tasks_set,
                         video_keys, task, args, rep):
    """Load one source episode's parquet and run check_episode on it.

    Returns True if the episode passed (parquet present and no check warning).
    Used to pre-filter pairs before writing: a pair is dropped when either side
    fails here. `video_keys` are the cameras kept for this split, all of which
    must fully decode to the action length.
    """
    label = f"src ep{src_idx:03d} ({'human' if is_human else 'robot'})"
    src_chunk_name = f"chunk-{src_idx // 1000:03d}"
    src_parquet = src / "data" / src_chunk_name / f"episode_{src_idx:06d}.parquet"
    if not src_parquet.exists():
        rep.warn(label, "parquet file missing in source")
        return False
    df = pd.read_parquet(src_parquet)
    meta_len = ep_len.get(src_idx)
    return check_episode(df, src, src_idx, meta_len, info, tasks_set, is_human,
                         video_keys, task, args, rep)


# ----------------------------------------------------------------------------- split
def process_split(name, src_group, src, dst, info, tasks, ep_len, ep_stats,
                  tasks_set, is_human, video_keys, args, rep):
    """Build one output dataset from the given ordered source episode indices.

    `video_keys` is the list of camera keys to KEEP for this split (e.g. the
    human split drops the wrist cameras). Returns {src_idx: new_idx} for the
    episodes actually written, so the caller can build the pairing report.
    """
    print(f"\n### building {name}  ({len(src_group)} candidate episodes) -> {dst}")

    all_video_keys = [k for k, v in info["features"].items() if v["dtype"] == "video"]
    drop_keys = [k for k in all_video_keys if k not in video_keys]
    if drop_keys:
        print(f"  keeping cams {video_keys}; dropping {drop_keys}")
    meta_dir = dst / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    episodes_out, stats_out = [], []
    mapping = {}  # src_idx -> new_idx for episodes actually written
    new_idx = 0
    global_index = 0  # running global frame index within this split
    total_frames = 0

    for src_idx in src_group:
        src_chunk_id = src_idx // 1000
        src_chunk_name = f"chunk-{src_chunk_id:03d}"
        src_parquet = src / "data" / src_chunk_name / f"episode_{src_idx:06d}.parquet"
        label = f"src ep{src_idx:03d}"

        # --- integrity: parquet present ---
        if not src_parquet.exists():
            rep.warn(label, "parquet file missing in source; episode SKIPPED")
            continue

        # --- integrity: all videos present & non-empty ---
        src_videos = {}
        missing_video = False
        for key in video_keys:
            vp = src / "videos" / src_chunk_name / key / f"episode_{src_idx:06d}.mp4"
            if not vp.exists() or vp.stat().st_size == 0:
                rep.warn(label, f"video missing/empty for {key}; episode SKIPPED")
                missing_video = True
                break
            src_videos[key] = vp
        if missing_video:
            continue

        df = pd.read_parquet(src_parquet)

        n = len(df)

        # --- rewrite indices for the new dataset ---
        df = df.copy()
        df["episode_index"] = new_idx
        df["frame_index"] = np.arange(n, dtype=np.int64)
        df["index"] = np.arange(global_index, global_index + n, dtype=np.int64)
        dst_chunk_id = new_idx // 1000
        dst_chunk_name = f"chunk-{dst_chunk_id:03d}"
        dst_data_dir = dst / "data" / dst_chunk_name
        dst_data_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(dst_data_dir / f"episode_{new_idx:06d}.parquet", index=False)

        # --- copy / link videos under the new name ---
        for key in video_keys:
            dst_video_dir = dst / "videos" / dst_chunk_name / key
            dst_video_dir.mkdir(parents=True, exist_ok=True)
            dst_vp = dst_video_dir / f"episode_{new_idx:06d}.mp4"
            if args.link:
                if dst_vp.exists() or dst_vp.is_symlink():
                    dst_vp.unlink()
                dst_vp.symlink_to(src_videos[key].resolve())
            else:
                shutil.copy2(src_videos[key], dst_vp)

        # --- episodes.jsonl entry ---
        tasks_for_ep = next(
            (e["tasks"] for e in load_episodes_cache(src) if e["episode_index"] == src_idx),
            [tasks[t]["task"] for t in sorted(set(df["task_index"].tolist()))],
        )
        episodes_out.append(
            {"episode_index": new_idx, "tasks": tasks_for_ep, "length": n}
        )

        # --- episodes_stats.jsonl entry: reuse content stats, fix derived stats ---
        stats = json.loads(json.dumps(ep_stats[src_idx]["stats"]))  # deep copy
        for dk in drop_keys:  # drop stats for cameras not kept in this split
            stats.pop(dk, None)
        stats["episode_index"] = int_stats([new_idx] * n)
        stats["index"] = int_stats(np.arange(global_index, global_index + n))
        stats_out.append({"episode_index": new_idx, "stats": stats})

        print(f"  src ep{src_idx:03d} -> new ep{new_idx:03d}  ({n} frames)")
        mapping[src_idx] = new_idx
        new_idx += 1
        global_index += n
        total_frames += n

    # --- write meta ---
    write_jsonl(meta_dir / "episodes.jsonl", episodes_out)
    write_jsonl(meta_dir / "episodes_stats.jsonl", stats_out)
    write_jsonl(meta_dir / "tasks.jsonl", tasks)

    out_info = json.loads(json.dumps(info))  # deep copy
    for dk in drop_keys:  # remove dropped cameras from the feature schema
        out_info["features"].pop(dk, None)
    n_eps = len(episodes_out)
    out_info["total_episodes"] = n_eps
    out_info["total_frames"] = total_frames
    out_info["total_tasks"] = len(tasks)
    out_info["total_videos"] = n_eps * len(video_keys)
    out_info["total_chunks"] = max(1, math.ceil(n_eps / info["chunks_size"]))
    out_info["splits"] = {"train": f"0:{n_eps}"}
    with open(meta_dir / "info.json", "w") as f:
        json.dump(out_info, f, indent=4)

    print(f"  -> wrote {n_eps} episodes, {total_frames} frames")
    return mapping


# episodes.jsonl is small; cache it once per source dir to avoid re-reading.
_EP_CACHE = {}


def load_episodes_cache(src):
    src = str(src)
    if src not in _EP_CACHE:
        _EP_CACHE[src] = load_jsonl(Path(src) / "meta" / "episodes.jsonl")
    return _EP_CACHE[src]


# ----------------------------------------------------------------------------- classify / pair
ROBOT_PREFIX = "robot_"
HUMAN_PREFIX = "human_"

def episode_kind(tasks):
    """Classify an episode by its (first) task label.

    kind is "robot" when the task starts with robot_ (ROBOT_PREFIX), "human"
    when it starts with human_ (HUMAN_PREFIX), else None; the checks run in
    that order. suffix is the task text with that prefix stripped, used to
    match the two sides of a pair; for an unclassified task it is the task
    unchanged.
    """
    task = tasks[0] if tasks else ""
    if task.startswith(ROBOT_PREFIX):
        return "robot", task[len(ROBOT_PREFIX):]
    if task.startswith(HUMAN_PREFIX):
        return "human", task[len(HUMAN_PREFIX):]
    return None, task


def build_pairs(episodes, rep):
    """Pair episodes from episodes.jsonl by task label, in source order.

    Each robot episode is paired with the human episode that immediately follows
    it. Episodes with neither prefix are discarded; a robot/human that cannot be
    paired is discarded (both warned).

    Returns a list of (robot_src_idx, human_src_idx) tuples.
    """
    ordered = sorted(episodes, key=lambda e: e["episode_index"])
    pairs = []
    pending = None  # (robot_src_idx, suffix) awaiting its human partner
    for e in ordered:
        idx = e["episode_index"]
        kind, suffix = episode_kind(e.get("tasks", []))
        label = f"src ep{idx:03d}"
        if kind is None:
            rep.warn(label, f"task {e.get('tasks')} starts with neither "
                            f"'{ROBOT_PREFIX}' nor '{HUMAN_PREFIX}'; discarded")
            continue
        if kind == "robot":
            if pending is not None:
                rep.warn(f"src ep{pending[0]:03d}",
                         "robot episode has no following human partner; discarded")
            pending = (idx, suffix)
        else:  # human
            if pending is None:
                rep.warn(label,
                         "human episode has no preceding robot partner; discarded")
                continue
            p_idx, p_suffix = pending
            if p_suffix != suffix:
                rep.warn(f"pair (robot src{p_idx}/human src{idx})",
                         f"task suffix mismatch: robot '{p_suffix}' vs "
                         f"human '{suffix}'")
            pairs.append((p_idx, idx))
            pending = None
    if pending is not None:
        rep.warn(f"src ep{pending[0]:03d}",
                 "robot episode has no following human partner; discarded")
    return pairs


def first_frame(video_path):
    """Decode the first RGB frame (HxWx3 uint8) of a video, or None on failure."""
    import av
    try:
        with av.open(str(video_path)) as container:
            for frame in container.decode(video=0):
                return frame.to_ndarray(format="rgb24")
    except Exception:
        return None
    return None


def find_duplicates(src, video_keys, threshold, rep):
    """Flag episodes whose first frame duplicates the previous episode's.

    Adjacent source episodes (in episode_index order) are compared on the first
    frame of EVERY camera. An episode is flagged when any camera it shares with
    the previous episode is (near-)identical -- mean abs pixel diff below
    `threshold`.

    Returns the set of duplicate source episode indices.
    """
    ordered = sorted(e["episode_index"] for e in load_episodes_cache(src))
    dup = set()
    prev_frames = None   # {cam_key: first-frame ndarray} for the previous episode
    prev_idx = None
    for idx in ordered:
        chunk = f"chunk-{idx // 1000:03d}"
        print(f"processing episode {idx:03d} from chunk {chunk}")
        frames = {}
        for key in video_keys:
            vp = src / "videos" / chunk / key / f"episode_{idx:06d}.mp4"
            if vp.exists() and vp.stat().st_size > 0:
                fr = first_frame(vp)
                if fr is not None:
                    frames[key] = fr
                else:
                    rep.warn(f"src ep{idx:03d}", f"video {vp} is empty; discarded")

        if prev_frames:
            common = [k for k in video_keys if k in frames and k in prev_frames]
            if common:
                diffs = []
                for k in common:
                    a, b = prev_frames[k].astype(np.float32), frames[k].astype(np.float32)
                    d = float(np.abs(a - b).mean()) if a.shape == b.shape else float("inf")
                    diffs.append(d)
                if min(diffs) < threshold:  # any shared camera near-identical
                    dup.add(idx)
                    rep.warn(
                        f"src ep{idx:03d}",
                        f"first frame duplicates ep{prev_idx:03d} on any "
                        f"{len(common)} camera(s) (min mean abs diff "
                        f"{min(diffs):.4f} < {threshold})",
                    )

        prev_frames, prev_idx = frames, idx
    return dup


# ----------------------------------------------------------------------------- scenes
def resolve_plan_csv(src, plan_csv_arg):
    """Locate the plan CSV that carries the scene / data_path columns for `src`.

    Prefers the scene-filled variant (`plan_filled.csv`, produced by
    convert_to_video.py --fill_scenes_only) so every episode's row has a scene;
    falls back to the raw `plan.csv`. An explicit --plan-csv overrides both.
    """
    if plan_csv_arg is not None:
        return Path(plan_csv_arg)
    filled = src / "plan_filled.csv"
    if filled.exists():
        return filled
    return src / "plan.csv"


def load_plan_scene_map(plan_csv):
    """Parse `plan_csv` into (normalized data_path, scene, row number) triples.

    Only rows carrying a `data_path` (parquet path) are kept; the scene may be
    empty. Paths are normalized to POSIX form so they match regardless of
    separator.

    The scene column is resolved with `resolve_scene_cells` over *all* rows, so
    a row that only carries a placement marker ("_alt" / "_keep") becomes the
    full "<scene name><marker>" string. The row number is the 1-based position
    of the row among the CSV's data rows (i.e. its line in the original
    plan.csv, ignoring the header) -- kept for *every* row, including the ones
    without data_path, since the row distance in build_conditions_list is
    measured in original plan.csv rows.
    """
    triples = []
    # utf-8-sig strips the BOM that the CSV is exported with (﻿ before `index`).
    with open(plan_csv, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "data_path" not in reader.fieldnames \
                or "scene" not in reader.fieldnames:
            sys.exit(f"error: {plan_csv} lacks 'data_path'/'scene' columns "
                     f"(columns: {reader.fieldnames})")
        rows = list(reader)

    scenes = _resolve_scene_cells([row.get("scene") for row in rows])
    for row_no, (row, scene) in enumerate(zip(rows, scenes), start=1):
        data_path = (row.get("data_path") or "").strip()
        if not data_path:
            continue
        norm = data_path.replace("\\", "/").strip("/")
        triples.append((norm, scene, row_no))
    return triples


def find_plan_scene(plan_pairs, dataset_name, rel_parquet):
    """Return the (scene, row number) for an episode matching a data_path entry.

    An episode's parquet is identified by `<dataset_name>/<rel_parquet>`.
    """
    key = f"{dataset_name}/{rel_parquet}".replace("\\", "/").strip("/")
    for norm, scene, row_no in plan_pairs:
        if norm == key or norm.endswith("/" + key):
            return scene, row_no
    return None


def compute_scenes(src, robot_map, info, plan_csv_arg, rep=None):
    """Look up each robot output episode's scene / plan.csv row number.

    For every robot output episode we take its pre-conversion (source) parquet
    path and look it up in the plan CSV's `data_path` column; the matching
    row's (resolved) `scene` and its row number in the original plan.csv are
    recorded under the new episode index. An episode whose source parquet is
    NOT present in `data_path` is an error (such episodes should already have
    been filtered out upstream) -- all such episodes are collected and the run
    aborts. A matched-but-empty scene is warned about and recorded as "".

    Returns ({new_idx(str): scene}, {new_idx(str): row number}). `rep` may be None to
    compute quietly, as the pre-write conditions_list filter does (it runs before
    the episodes it warns about are known to survive).
    """
    plan_csv = resolve_plan_csv(src, plan_csv_arg)
    if not plan_csv.exists():
        sys.exit(f"error: plan csv not found for scenes.json: {plan_csv}")

    plan_pairs = load_plan_scene_map(plan_csv)
    dataset_name = src.resolve().name
    chunks_size = int(info["chunks_size"])
    data_path_template = info["data_path"]

    scenes = {}     # robot new episode (str) -> scene
    plan_rows = {}  # robot new episode (str) -> row number in the original plan.csv
    missing = []    # (new_idx, rel_parquet) for episodes with no plan row
    for src_idx, new_idx in sorted(robot_map.items(), key=lambda kv: kv[1]):
        rel_parquet = data_path_template.format(
            episode_chunk=src_idx // chunks_size, episode_index=src_idx)
        found = find_plan_scene(plan_pairs, dataset_name, rel_parquet)
        if found is None:
            missing.append((new_idx, f"{dataset_name}/{rel_parquet}"))
            continue
        scene, row_no = found
        if not scene and rep is not None:
            rep.warn(f"robot ep{new_idx:03d}",
                     f"matched plan row has empty scene ({dataset_name}/{rel_parquet})")
        scenes[str(new_idx)] = scene
        plan_rows[str(new_idx)] = row_no

    if missing:
        lines = "\n".join(f"    robot ep{n:03d}: {p}" for n, p in missing)
        sys.exit(
            f"error: {len(missing)} robot episode(s) have no matching data_path row "
            f"in {plan_csv}\n{lines}\n"
            "  (these should have been filtered out earlier; the plan and the "
            "source dataset are inconsistent)"
        )

    return scenes, plan_rows


def write_scenes(src, robot_out, robot_map, info, plan_csv_arg, rep):
    """Write `robot/meta/scenes.json` and `robot/meta/plan_rows.json`.

    Contents come from compute_scenes; plan_rows.json is what lets
    build_conditions_list apply the "no more than MAX_ROW_GAP rows apart in the
    original plan.csv" rule.
    """
    plan_csv = resolve_plan_csv(src, plan_csv_arg)
    scenes, plan_rows = compute_scenes(src, robot_map, info, plan_csv_arg, rep)

    meta_dir = robot_out / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    scenes_path = meta_dir / "scenes.json"
    ordered = {k: scenes[k] for k in sorted(scenes, key=int)}
    with open(scenes_path, "w", encoding="utf-8") as f:
        json.dump(ordered, f, ensure_ascii=False, indent=2)

    rows_path = meta_dir / "plan_rows.json"
    ordered_rows = {k: plan_rows[k] for k in sorted(plan_rows, key=int)}
    with open(rows_path, "w", encoding="utf-8") as f:
        json.dump(ordered_rows, f, ensure_ascii=False, indent=2)

    n_distinct = len(set(scenes.values()))
    print(f"scenes: {len(scenes)} robot episodes across {n_distinct} scene(s) "
          f"(from {plan_csv.name}) -> {scenes_path}")
    print(f"plan rows: {len(plan_rows)} robot episodes -> {rows_path}")


# ----------------------------------------------------------------------------- pairing
def write_pairing(root, robot_out, human_out, pairs, robot_map, human_map, rep,
                  cam_key="observation.images.cam_high"):
    """Write the robot<->human episode correspondence and flag broken pairs.

    `pairs` is the list of (robot_src, human_src) tuples that survived
    classification and duplicate filtering. A pair is broken when one side was
    dropped during writing (missing parquet/video, etc.); such pairs are warned
    about and marked so a reader never silently trusts an unpaired episode.

    The machine-readable pairing is emitted as `<robot_out>/meta/conditions.json`:
    a flat map from each robot (new) episode index to the root-relative path of
    its paired human `cam_key` video. Broken pairs are omitted from that map. A
    `<root>/pairing.csv` audit table is written alongside.
    """
    rows = []
    conditions = {}  # robot_new_episode (str) -> paired human cam_key video path
    for pair_id, (p_src, h_src) in enumerate(pairs):
        p_new = robot_map.get(p_src)
        h_new = human_map.get(h_src)
        if p_new is not None and h_new is not None:
            status = "ok"
            h_chunk = f"chunk-{h_new // 1000:03d}"
            h_video = (human_out / "videos" / h_chunk / cam_key
                       / f"episode_{h_new:06d}.mp4")
            # root-relative, so the split output stays relocatable
            conditions[str(p_new)] = str(h_video.relative_to(root))
        elif p_new is None and h_new is None:
            status = "missing_both"
        elif p_new is None:
            status = "missing_robot"
        else:
            status = "missing_human"
        if status != "ok":
            rep.warn(f"pair {pair_id} (robot src{p_src}/human src{h_src})",
                     f"broken pair: {status}")
        rows.append({
            "pair_index": pair_id,
            "robot_src_episode": p_src,
            "robot_new_episode": p_new if p_new is not None else "",
            "human_src_episode": h_src,
            "human_new_episode": h_new if h_new is not None else "",
            "status": status,
        })

    base = root / "pairing"

    # CSV (human-readable, mirrors the repo's plan.csv style)
    cols = ["pair_index", "robot_src_episode", "robot_new_episode",
            "human_src_episode", "human_new_episode", "status"]
    with open(base.with_suffix(".csv"), "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")

    # conditions.json under the robot dataset: {robot_ep: human cam_high video}
    # (ordered by robot episode index)
    cond_path = robot_out / "meta" / "conditions.json"
    cond_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = {k: conditions[k] for k in sorted(conditions, key=int)}
    with open(cond_path, "w") as f:
        json.dump(ordered, f, indent=2)

    ok = sum(1 for r in rows if r["status"] == "ok")
    print(f"\npairing: {ok}/{len(rows)} complete pairs -> {base}.csv")
    print(f"conditions ({cam_key}): {len(conditions)} robot episodes -> {cond_path}")


# ----------------------------------------------------------------------------- conditions_list
def drop_long_conditions_pairs(src, pairs, info, episodes, plan_csv_arg, max_len,
                               rep):
    """Drop pairs whose robot episode would get a conditions_list longer than max_len.

    A conditions_list groups the episodes sharing a base scene, a task and a plan
    row neighbourhood (see build_conditions_list.py). An oversized group means the
    grouping key is not discriminating -- typically a scene cell that resolved
    to a bare "_alt"/"_keep" marker, whose base scene is empty, so every such
    episode collapses into one group -- and every episode in it is discarded.

    The list is predicted here, BEFORE anything is written: only scenes, plan rows
    and tasks feed the grouping, none of which need the output dataset to exist.
    That way the survivors are re-indexed once by process_split and every meta file
    stays consistent (dropping episodes afterwards would mean re-indexing the whole
    dataset). The prediction is a pre-filter only; the final conditions_list.json
    is rebuilt from the episodes actually written.

    Returns the surviving pairs (whole pair goes, as with every other drop here).
    """
    robot_map = {p_src: i for i, (p_src, _) in enumerate(pairs)}
    scenes, plan_rows = compute_scenes(src, robot_map, info, plan_csv_arg)
    src_tasks = {e["episode_index"]: tuple(e.get("tasks", [])) for e in episodes}
    ep_tasks = {str(new): src_tasks.get(p_src, ()) for p_src, new in robot_map.items()}
    # the condition values are irrelevant to the grouping; only the sizes are read
    result = _build_conditions_list(scenes, {k: k for k in scenes}, ep_tasks,
                                    plan_rows)

    too_long = {int(k) for k, v in result.items() if len(v) > max_len}
    if not too_long:
        print(f"  conditions_list check: no episode exceeds {max_len} condition(s)")
        return pairs

    remaining = []
    for p_src, h_src in pairs:
        n_cond = len(result[str(robot_map[p_src])])
        if robot_map[p_src] in too_long:
            rep.warn(f"pair (robot src{p_src}/human src{h_src})",
                     f"conditions_list would hold {n_cond} condition(s) > "
                     f"{max_len} (scene '{scenes[str(robot_map[p_src])]}', "
                     f"task {list(src_tasks.get(p_src, ()))}); pair dropped")
            continue
        remaining.append((p_src, h_src))
    print(f"  conditions_list check: dropped {len(pairs) - len(remaining)} pair(s) "
          f"with more than {max_len} condition(s); {len(remaining)} pair(s) remain")
    return remaining


def build_robot_conditions_list(robot_out):
    """Run build_conditions_list.py's logic on the freshly-written robot dataset.

    scenes.json + plan_rows.json + conditions.json + episodes.jsonl must already
    exist under `robot_out/meta`; writes `robot_out/meta/conditions_list.json`.
    """
    meta = robot_out / "meta"
    scenes = json.loads((meta / "scenes.json").read_text(encoding="utf-8"))
    conditions = json.loads((meta / "conditions.json").read_text(encoding="utf-8"))
    plan_rows = json.loads((meta / "plan_rows.json").read_text(encoding="utf-8"))
    ep_tasks = _load_episode_tasks(meta / "episodes.jsonl")

    missing = [k for k in scenes if k not in conditions]
    if missing:
        sys.exit(f"error: {len(missing)} robot episode(s) in scenes.json are "
                 f"missing from conditions.json, e.g. {missing[:5]}")

    result = _build_conditions_list(scenes, conditions, ep_tasks, plan_rows)
    out = meta / "conditions_list.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                   encoding="utf-8")

    sizes = [len(v) for v in result.values()]
    print(f"conditions_list: {len(result)} robot episode(s) -> {out}")
    if sizes:
        print(f"  list length min/max/avg = {min(sizes)}/{max(sizes)}/"
              f"{sum(sizes) / len(sizes):.1f}")


# ----------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True,
                   help="source LeRobot dataset root (e.g. outputs/videos/1015)")
    p.add_argument("--root", required=True,
                   help="output root; robot/ and human/ datasets plus split.log "
                        "and pairing.csv are written under it (e.g. "
                        "outputs/split_new/1015)")
    p.add_argument("--dup-threshold", type=float, default=0.001,
                   help="mean abs first-frame pixel diff (0-255) below which an "
                        "episode is a duplicate of the previous one on ALL cameras")
    p.add_argument("--no-dup-check", action="store_true",
                   help="skip the duplicate-first-frame check (keep all pairs)")
    p.add_argument("--drop-episodes", nargs="*", default=[],
                   help="source episode indices to discard outright, e.g. "
                        "`--drop-episodes 19 883,955 10-12`. A pair is dropped "
                        "whole when either of its sides is listed here")
    p.add_argument("--min-frames-robot", type=int, default=45,
                   help="warn when a robot episode is shorter than this")
    p.add_argument("--min-frames-human", type=int, default=20,
                   help="warn when a human episode is shorter than this")
    p.add_argument("--static-action-tol", type=float, default=0.10,
                   help="max per-joint action range allowed for human (odd) episodes")
    p.add_argument("--static-state-tol", type=float, default=0.10,
                   help="max per-joint state range allowed as perturbation for human")
    p.add_argument("--motion-min", type=float, default=0.25,
                   help="min per-joint action range expected for robot (even) episodes")
    p.add_argument("--idle-arm-tol", type=float, default=0.30,
                   help="max per-joint action range allowed for the arm a robot "
                        "task does NOT name (second half for a left task, first "
                        "half for a right task)")
    p.add_argument("--keep-third-cam", action="store_true",
                   help="keep the third-person cam in the human split (default: dropped)")
    p.add_argument("--human-drop-cams", nargs="*",
                   default=["observation.images.cam_left_wrist",
                            "observation.images.cam_right_wrist"],
                   help="camera keys NOT stored in the human split (default: wrist cams)")
    p.add_argument("--max-conditions", type=int, default=3,
                   help="drop an episode (whole pair) whose conditions_list would "
                        "hold more than this many conditions; 0 or less disables "
                        "the check")
    p.add_argument("--plan-csv", default=None,
                   help="plan CSV with scene/data_path columns for scenes.json "
                        "(default: <src>/plan_filled.csv, else <src>/plan.csv)")
    p.add_argument("--link", action="store_true",
                   help="symlink videos instead of copying (saves disk)")
    p.add_argument("--overwrite", action="store_true",
                   help="remove existing output dirs before writing")
    args = p.parse_args()

    src = Path(args.src)
    if not (src / "meta" / "info.json").exists():
        sys.exit(f"error: {src} is not a LeRobot dataset (no meta/info.json)")

    # --- everything lands under a single root ---
    root = Path(args.root)
    robot_out, human_out = root / "robot", root / "human"
    log_path = root / "split.log"
    root.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(sys.__stdout__, log_path)
    print(f"log: {log_path}")

    info = json.load(open(src / "meta" / "info.json"))
    tasks = load_jsonl(src / "meta" / "tasks.jsonl")
    tasks_set = {t["task_index"] for t in tasks}
    episodes = load_episodes_cache(src)
    ep_len = {e["episode_index"]: e["length"] for e in episodes}
    ep_stats = {e["episode_index"]: e for e in load_jsonl(src / "meta" / "episodes_stats.jsonl")}

    rep = Reporter()

    print(f"source: {src}")
    print(f"  total episodes in meta: {len(ep_len)}")

    # --- classify by task label and pair robot<->human in source order ---
    pairs = build_pairs(episodes, rep)
    print(f"  paired {len(pairs)} robot<->human episode pair(s)")

    # --- drop the explicitly-listed source episodes (whole pair goes) ---
    # Applied AFTER pairing on purpose: build_pairs matches each robot with the
    # human that immediately follows it, so removing an episode beforehand would
    # silently re-pair its neighbours.
    drop_eps = parse_episode_list(args.drop_episodes)
    if drop_eps:
        unknown = sorted(drop_eps - set(ep_len))
        if unknown:
            rep.warn("--drop-episodes",
                     f"episodes not present in source (ignored): {unknown}")
        remaining = []
        for p_src, h_src in pairs:
            listed = [x for x in (p_src, h_src) if x in drop_eps]
            if listed:
                print(f"  pair (robot src{p_src}/human src{h_src}): episode(s) "
                      f"{listed} listed in --drop-episodes; pair dropped")
                continue
            remaining.append((p_src, h_src))
        print(f"  --drop-episodes {sorted(drop_eps)}: dropped "
              f"{len(pairs) - len(remaining)} pair(s); {len(remaining)} remain")
        pairs = remaining

    if args.keep_third_cam:
        all_video_keys = [k for k, v in info["features"].items() if v["dtype"] == "video"]
    else:
        all_video_keys = [k for k, v in info["features"].items() if v["dtype"] == "video" and k != "observation.images.cam_third"]
    human_video_keys = [k for k in all_video_keys if k not in args.human_drop_cams]

    # --- drop pairs whose robot or human first frame duplicates the previous ep ---
    if args.no_dup_check:
        dup_set = set()
        print("  duplicate first-frame check skipped (--no-dup-check)")
    else:
        print(f"  checking first frames across {len(all_video_keys)} camera(s) "
              f"(threshold {args.dup_threshold}) ...")
        dup_set = find_duplicates(src, all_video_keys, args.dup_threshold, rep)
        print(f"  {len(dup_set)} episode(s) flagged as duplicate first frame")
    kept_pairs = []
    for p_src, h_src in pairs:
        bad = [x for x in (p_src, h_src) if x in dup_set]
        if bad:
            rep.warn(f"pair (robot src{p_src}/human src{h_src})",
                     f"episode(s) {bad} flagged as duplicate first frame; "
                     "pair dropped")
            continue
        kept_pairs.append((p_src, h_src))
    if len(kept_pairs) != len(pairs):
        print(f"  dropped {len(pairs) - len(kept_pairs)} duplicate pair(s); "
              f"{len(kept_pairs)} pair(s) remain")

    # --- per-episode validity checks: drop a pair if either side fails ---
    print(f"  running per-episode checks on {len(kept_pairs)} pair(s) ...")
    checked_pairs = []
    ep_task = {e["episode_index"]: (e.get("tasks") or [""])[0] for e in episodes}
    for p_src, h_src in kept_pairs:
        p_ok = episode_passes_check(src, p_src, False, info, ep_len, tasks_set,
                                    all_video_keys, ep_task.get(p_src, ""),
                                    args, rep)
        h_ok = episode_passes_check(src, h_src, True, info, ep_len, tasks_set,
                                    human_video_keys, ep_task.get(h_src, ""),
                                    args, rep)
        if not (p_ok and h_ok):
            failed = [x for x, ok in ((p_src, p_ok), (h_src, h_ok)) if not ok]
            rep.warn(f"pair (robot src{p_src}/human src{h_src})",
                     f"episode(s) {failed} failed check_episode; pair dropped")
            continue
        checked_pairs.append((p_src, h_src))
    if len(checked_pairs) != len(kept_pairs):
        print(f"  dropped {len(kept_pairs) - len(checked_pairs)} pair(s) failing "
              f"per-episode checks; {len(checked_pairs)} pair(s) remain")
    kept_pairs = checked_pairs

    # --- drop pairs whose robot episode would carry too many conditions ---
    if args.max_conditions > 0:
        kept_pairs = drop_long_conditions_pairs(src, kept_pairs, info, episodes,
                                                args.plan_csv,
                                                args.max_conditions, rep)

    robot_group = [p for p, _ in kept_pairs]
    human_group = [h for _, h in kept_pairs]

    for out in (robot_out, human_out):
        if out.exists():
            if args.overwrite:
                shutil.rmtree(out)
            else:
                sys.exit(f"error: {out} already exists (use --overwrite)")

    robot_map = process_split("robot", robot_group, src, robot_out, info, tasks,
                              ep_len, ep_stats, tasks_set, is_human=False,
                              video_keys=all_video_keys, args=args, rep=rep)
    human_map = process_split("human", human_group, src, human_out, info, tasks,
                              ep_len, ep_stats, tasks_set, is_human=True,
                              video_keys=human_video_keys, args=args, rep=rep)

    print(f"\nwritten: robot={len(robot_map)} episodes, "
          f"human={len(human_map)} episodes")

    # --- pairing report: each robot episode pairs with its human partner ---
    write_pairing(root, robot_out, human_out, kept_pairs, robot_map, human_map, rep)

    # --- scenes.json: which scene each robot episode belongs to (from plan.csv) ---
    write_scenes(src, robot_out, robot_map, info, args.plan_csv, rep)

    # --- conditions_list.json for robot (needs scenes.json + conditions.json) ---
    build_robot_conditions_list(robot_out)

    n_warn = rep.summary()
    # non-zero exit if anything looked wrong, so callers/CI can catch it
    sys.exit(1 if n_warn else 0)


if __name__ == "__main__":
    main()
