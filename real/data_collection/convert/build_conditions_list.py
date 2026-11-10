#!/usr/bin/env python3
"""Build conditions_list.json from scenes.json + conditions.json + plan_rows.json.

Each robot episode has one paired reference video in conditions.json; this
script turns that into a "condition list" for every episode. An episode shares
a condition (the two appear in each other's lists) when all of **three** things
hold -- every episode trivially satisfies them with itself, which is why it
always comes first in its own list:

  1. its plan.csv row and the other's are no more than ``MAX_ROW_GAP`` (30) rows
     apart;
  2. both belong to the same base scene, i.e. the trailing placement marker
     (_keep/_alt etc.) is stripped, so scene1 and scene1_alt pair up (see
     generate_plan.py);
  3. their task (episodes.jsonl "tasks") is identical.

Row numbers come from plan_rows.json, {episode_index: plan.csv data row},
written by split_dataset.py; without that file the row-gap rule is skipped
(with a warning).

Usage:
    python convert/build_conditions_list.py outputs/split_new/1015/robot
    python convert/build_conditions_list.py outputs/split_new/1015/robot -o /tmp/cl.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# Max row distance in the original plan.csv for two episodes to share a condition.
MAX_ROW_GAP = 30

# Markers in the scene column that denote another placement of the same scene
# (a cell starting with one of them continues the current scene).
_SCENE_MARKERS = ("_keep", "_squeeze", "_alt", "_shuffle", "_disturb")

# Trailing placement-marker suffix stripped by ``base_scene``.
_SUFFIX_RE = re.compile(r"_(keep|squeeze|alt|shuffle|disturb[^;]*)$")

# Anchor-pass suffixes: a row ending in _small_alt attaches to the nearest
# earlier same-task row without it, while a row ending in _big_alt/_swap cuts
# the equivalence class (it becomes the new anchor).
_SMALL_MOVE_SUFFIX = "_small_alt"
_CUT_MOVE_SUFFIXES = ("_big_alt", "_swap")


def is_scene_marker(scene: str) -> bool:
    """Whether the scene description starts with a placement marker (i.e. the
    cell is only a marker, not a new scene name)."""
    return scene.startswith(_SCENE_MARKERS)


def base_scene(scene: str) -> str:
    """Strip the trailing placement-marker suffix to get the base scene."""
    return _SUFFIX_RE.sub("", scene.strip()).strip()


def is_shuffled(scene: str) -> bool:
    """True when the scene string ends with the _shuffle marker."""
    return scene.strip().endswith("_shuffle")


def scene_move_kind(scene: str) -> str:
    """Placement category of a scene string.

    small: ends with _small_alt (attaches to the previous same-task row without it);
    cut:   ends with _big_alt/_swap (cuts the equivalence class, new anchor);
    base:  anything else (merged via the base-scene rule).
    """
    s = (scene or "").strip()
    if s.endswith(_SMALL_MOVE_SUFFIX):
        return "small"
    if s.endswith(_CUT_MOVE_SUFFIXES):
        return "cut"
    return "base"


def resolve_scene_cells(cells: list[str]) -> list[str]:
    """Fill the plan.csv scene column so every row carries a full scene string.

    For each row, let ``A`` be its own non-empty scene, else the nearest
    non-empty scene above it. If ``A`` starts with a placement marker
    (``_SCENE_MARKERS``), the result is ``B + A`` where ``B`` is the nearest
    scene above that is not a marker itself; otherwise the result is ``A``.

    Re-running on an already-resolved column (e.g. ``plan_filled.csv``) is a
    no-op.
    """
    resolved: list[str] = []
    last_scene = ""       # nearest non-empty raw scene (candidate A for blank rows)
    last_base_scene = ""  # nearest non-empty non-marker scene (candidate B)

    for cell in cells:
        own = (cell or "").strip()

        if own:
            scene_a = own
            last_scene = own
            if not is_scene_marker(own):
                last_base_scene = own
        else:
            scene_a = last_scene

        if not scene_a:
            # nothing to fill from yet
            resolved.append(own)
        elif is_scene_marker(scene_a):
            resolved.append(last_base_scene + scene_a)
        else:
            resolved.append(scene_a)

    return resolved


def load_episode_tasks(episodes_path: Path) -> dict[str, tuple]:
    """Read episodes.jsonl; return {episode_index(str): tasks(tuple)}."""
    tasks: dict[str, tuple] = {}
    with episodes_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ep = json.loads(line)
            tasks[str(ep["episode_index"])] = tuple(ep.get("tasks", []))
    return tasks


def build_conditions_list(
    scenes: dict[str, str],
    conditions: dict[str, str],
    ep_tasks: dict[str, tuple],
    plan_rows: dict[str, int] | None = None,
    max_row_gap: int = MAX_ROW_GAP,
) -> dict[str, list[str]]:
    """Build the condition list for every episode (rules in the module docstring).

    ``plan_rows`` is {episode_index: plan.csv data row}; when None, the row-gap
    rule is not applied.
    """
    result: dict[str, list[str]] = {}

    # Pre-compute base scene / move kind / shuffled flag / plan row per episode,
    # iterating in numeric order.
    keys = sorted(scenes, key=lambda k: int(k))
    base = {k: base_scene(scenes[k]) for k in keys}
    kind = {k: scene_move_kind(scenes[k]) for k in keys}
    shuffled = {k: is_shuffled(scenes[k]) for k in keys}

    rows: dict[str, int] | None = None
    if plan_rows is not None:
        missing = [k for k in keys if k not in plan_rows]
        if missing:
            raise ValueError(
                f"{len(missing)} episode(s) are missing a plan row number, e.g. {missing[:5]}"
            )
        rows = {k: int(plan_rows[k]) for k in keys}

    # _small_alt rows anchor in plan-row order to the nearest preceding
    # same-task row that is not _small_alt. _big_alt/_swap rows are valid
    # anchors too and overwrite the task's anchor slot, cutting the class off
    # from earlier anchors; subsequent _small_alt rows attach to them instead.
    order = sorted(
        keys, key=lambda k: (rows[k] if rows is not None else int(k), int(k))
    )
    anchor: dict[str, str | None] = {}
    last_non_small: dict[tuple, str] = {}
    for k in order:
        if shuffled[k]:
            continue
        task = ep_tasks.get(k)
        if kind[k] == "small":
            anchor[k] = last_non_small.get(task)
        else:
            last_non_small[task] = k

    def related(j: str, k: str) -> bool:
        """Whether the two episodes share a condition (symmetric relation)."""
        if ep_tasks.get(j) != ep_tasks.get(k):
            return False
        if shuffled[j] or shuffled[k]:
            return False
        if rows is not None and abs(rows[j] - rows[k]) > max_row_gap:
            return False
        kj, kk = kind[j], kind[k]
        if kj == "base" and kk == "base":
            return base[j] == base[k]
        if kj == "small" and kk == "small":
            return anchor[j] is not None and anchor[j] == anchor[k]
        if kk == "small":
            return anchor[k] == j
        if kj == "small":
            return anchor[j] == k
        # Remaining cut vs base/cut combinations: the class is cut here.
        return False

    for k in keys:
        if shuffled[k]:
            # A shuffled scene keeps only its own condition.
            result[k] = [conditions[k]]
            continue

        # Collect every episode related to k (k first, the rest ascending by
        # episode index).
        members = [j for j in keys if j != k and related(j, k)]
        ordered = [k] + members
        result[k] = [conditions[j] for j in ordered]

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build conditions_list.json")
    parser.add_argument(
        "root",
        type=Path,
        help="dataset root (must contain meta/scenes.json, meta/conditions.json "
             "and meta/episodes.jsonl; meta/plan_rows.json recommended)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="output path; defaults to <root>/meta/conditions_list.json",
    )
    parser.add_argument(
        "--max-row-gap",
        type=int,
        default=MAX_ROW_GAP,
        help=f"max row distance in plan.csv (default {MAX_ROW_GAP})",
    )
    args = parser.parse_args()

    meta = args.root / "meta"
    scenes = json.loads((meta / "scenes.json").read_text(encoding="utf-8"))
    conditions = json.loads((meta / "conditions.json").read_text(encoding="utf-8"))
    ep_tasks = load_episode_tasks(meta / "episodes.jsonl")

    rows_path = meta / "plan_rows.json"
    if rows_path.exists():
        plan_rows = json.loads(rows_path.read_text(encoding="utf-8"))
    else:
        plan_rows = None
        print(f"warning: {rows_path} not found; skipping the plan.csv row-gap "
              "rule (re-run split_dataset.py to generate it)")

    missing = [k for k in scenes if k not in conditions]
    if missing:
        raise SystemExit(
            f"{len(missing)} episode(s) in scenes have no conditions entry, "
            f"e.g. {missing[:5]}"
        )

    result = build_conditions_list(
        scenes, conditions, ep_tasks, plan_rows, args.max_row_gap
    )

    out = args.output or (meta / "conditions_list.json")
    out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    sizes = [len(v) for v in result.values()]
    print(f"Wrote {out}")
    print(f"episodes: {len(result)}; list length min/max/avg = "
          f"{min(sizes)}/{max(sizes)}/{sum(sizes) / len(sizes):.1f}")


if __name__ == "__main__":
    main()
