#!/usr/bin/env python3
"""Generate deterministic scene plans using the scene task registry.

Accept groups only when all tasks can be constructed and statically matched
within the object limit. Quotas adapt to supported task counts, and group sizes
adapt to difficulty; --quota and --no-adaptive override this policy.
Search budgets are fixed at 200. --existing-plan balances task load and avoids
duplicate task sets across batches.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.scene import (  # noqa: E402
    OOD_EVAL_TASKS,
    all_task_names,
    candidate_tasks,
    mixed_scene_for_tasks,
)


ISOLATED_TASKS = frozenset(
    {
        "place_dual_shoes",
        *OOD_EVAL_TASKS,
    }
)
EXPERIMENTAL_TASKS = OOD_EVAL_TASKS
BLOCK_TASKS = frozenset(
    {
        "beat_block_hammer",
        "blocks_ranking_rgb",
        "blocks_ranking_size",
        "handover_block",
        "stack_blocks_three",
        "stack_blocks_two",
    }
)
# Each pair describes two task families that cannot share one physical layout.
PLACEMENT_CONFLICTS = (
    (frozenset({"lift_pot", "move_can_pot"}), frozenset({"place_cans_plasticbox"})),
    (frozenset({"open_laptop"}), frozenset({"place_cans_plasticbox"})),
    (
        frozenset({"place_burger_fries"}),
        frozenset({"beat_block_hammer", "dump_bin_bigbin", "lift_pot", "move_can_pot"}),
    ),
    (frozenset({"place_burger_fries"}), frozenset({"place_can_basket"})),
)
DEFAULT_SCENE_CONFIG = "dual_arm_multi_perturb"
DEFAULT_QUOTA = "2:4,3:3"
DEFAULT_MAX_SEARCH = 200
DEFAULT_MIN_OBJECTS = 5
DEFAULT_MAX_BLOCK_RATIO = 0.15

# Snapshot from TASK_DISTRIBUTION_AND_DIFFICULTY.md (2026-07-12).  Episodes
# are the dual-arm total and participation is the selected-placement ratio.
# Keeping this small table next to the generator makes the policy reproducible
# while allowing callers to replace it with a newer dashboard snapshot.
_TASK_STATS_ROWS = (
    ("insert_marker_into_pencup", "D3", 0, 0.000),
    ("swap_two_objects", "D3", 0, 0.000),
    ("hanging_mug", "D4", 0, 0.000),
    ("place_pot_on_scale", "D4", 0, 0.000),
    ("scan_object", "D3", 8, 0.040),
    ("put_object_cabinet", "D3", 8, 0.050),
    ("put_bottles_dustbin", "D3", 4, 0.059),
    ("lift_pot", "D3", 12, 0.065),
    ("move_pillbottle_pad", "D3", 12, 0.070),
    ("pick_dual_bottles", "D3", 32, 0.073),
    ("pick_diverse_bottles", "D3", 36, 0.083),
    ("place_can_basket", "D2", 24, 0.214),
    ("place_object_on_shoe", "D2", 0, 0.000),
    ("close_microwave", "D2", 0, 0.000),
    ("unstack_bowls_two", "D2", 0, 0.000),
    ("push_toycar_to_pad", "D2", 0, 0.000),
    ("lay_bottle_down", "D2", 0, 0.000),
    ("handover_block", "D2", 16, 0.222),
    ("grab_roller", "D2", 40, 0.222),
    ("dump_bin_bigbin", "D2", 32, 0.348),
    ("place_object_scale", "D1", 32, 0.381),
    ("place_object_upper_left", "D1", 0, 0.000),
    ("place_object_upper_right", "D1", 0, 0.000),
    ("place_object_lower_left", "D1", 0, 0.000),
    ("place_object_lower_right", "D1", 0, 0.000),
    ("close_laptop", "D1", 0, 0.000),
    ("place_object_between_objects", "D1", 0, 0.000),
    ("place_fan", "D1", 140, 0.417),
    ("place_object_stand", "D1", 36, 0.450),
    ("place_empty_cup", "D1", 76, 0.452),
    ("place_phone_stand", "D1", 56, 0.483),
    ("shake_bottle_horizontally", "D1", 356, 0.497),
    ("place_object_basket", "D1", 32, 0.500),
    ("blocks_ranking_size", "D1", 28, 0.538),
    ("adjust_bottle", "D1", 388, 0.542),
    ("turn_switch", "D1", 120, 0.545),
    ("move_can_pot", "D1", 76, 0.559),
    ("shake_bottle", "D1", 420, 0.587),
    ("place_bread_skillet", "D1", 36, 0.600),
    ("move_stapler_pad", "D1", 176, 0.611),
    ("place_bread_basket", "D1", 52, 0.619),
    ("stack_blocks_three", "D1", 40, 0.625),
    ("open_microwave", "D1", 108, 0.628),
    ("stack_bowls_three", "D0", 336, 0.656),
    ("place_a2b_left", "D0", 200, 0.658),
    ("place_container_plate", "D0", 84, 0.677),
    ("rotate_qrcode", "D0", 200, 0.685),
    ("handover_mic", "D0", 156, 0.750),
    ("place_shoe", "D0", 132, 0.767),
    ("stamp_seal", "D0", 228, 0.770),
    ("move_playingcard_away", "D0", 108, 0.771),
    ("open_laptop", "D0", 120, 0.811),
    ("stack_bowls_two", "D0", 428, 0.836),
    ("click_alarmclock", "D0", 300, 0.843),
    ("stack_blocks_two", "D0", 76, 0.864),
    ("blocks_ranking_rgb", "D0", 52, 0.867),
    ("place_a2b_right", "D0", 264, 0.868),
    ("place_cans_plasticbox", "D0", 64, 0.889),
    ("place_mouse_pad", "D0", 252, 0.900),
    ("beat_block_hammer", "D0", 76, 0.905),
    ("place_burger_fries", "D0", 92, 0.958),
    ("press_stapler", "D0", 292, 0.961),
    ("click_bell", "D0", 332, 0.976),
    ("place_dual_shoes", "D0", 52, 1.000),
)
TASK_STATS = {
    task: {
        "difficulty": difficulty,
        "episodes": episodes,
        "participation": participation,
    }
    for task, difficulty, episodes, participation in _TASK_STATS_ROWS
}
_DIFFICULTY_RANK = {"D0": 0, "D1": 1, "D2": 2, "D3": 3, "D4": 4}
_GROUP_CAP_BY_DIFFICULTY = {"D0": None, "D1": 5, "D2": 3, "D3": 2, "D4": 1}


def _normalise_tasks(tasks: Iterable[str] | None) -> list[str]:
    """Validate and deduplicate task names while preserving first occurrence."""
    if tasks is None:
        values = [task for task in all_task_names() if task not in EXPERIMENTAL_TASKS]
    elif isinstance(tasks, str):
        values = [item.strip() for item in tasks.split(",") if item.strip()]
    else:
        values = [str(task).strip() for task in tasks if str(task).strip()]
    known = set(all_task_names())
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(f"unknown task(s): {', '.join(unknown)}")
    return list(dict.fromkeys(values))


def _normalise_exclusions(exclude_tasks: Iterable[str] | None) -> set[str]:
    """Validate task exclusions and return their names as a set."""
    if exclude_tasks is None:
        return set()
    values = _normalise_tasks(exclude_tasks)
    return set(values)


def _task_signature(tasks: Iterable[str]) -> tuple[str, ...]:
    """Return the order-independent task signature used for de-duplication."""
    return tuple(sorted({str(task) for task in tasks if str(task).strip()}))


def _load_existing_metadata(
    paths: Iterable[str | Path] | None,
) -> tuple[set[str], set[int], Counter, set[tuple[str, ...]]]:
    """Read names, seeds, task load and signatures from existing plans."""
    names: set[str] = set()
    seeds: set[int] = set()
    task_counts: Counter = Counter()
    signatures: set[tuple[str, ...]] = set()
    for raw_path in paths or ():
        path = Path(raw_path)
        candidates = sorted(path.glob("*.y*ml")) if path.is_dir() else [path]
        for candidate in candidates:
            try:
                document = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
            except (OSError, yaml.YAMLError) as exc:
                raise ValueError(
                    f"could not read existing plan {candidate}: {exc}"
                ) from exc
            if not isinstance(document, dict):
                continue
            if document.get("group_seed") is not None:
                try:
                    seeds.add(int(document["group_seed"]))
                except (TypeError, ValueError):
                    pass
            for group in document.get("groups", []) or ():
                if isinstance(group, dict) and group.get("name"):
                    names.add(str(group["name"]))
                if isinstance(group, dict) and isinstance(group.get("tasks"), list):
                    tasks = [str(task) for task in group["tasks"] if str(task).strip()]
                    task_counts.update(tasks)
                    if tasks:
                        signatures.add(_task_signature(tasks))
                if isinstance(group, dict) and group.get("group_seed") is not None:
                    try:
                        seeds.add(int(group["group_seed"]))
                    except (TypeError, ValueError):
                        pass
    return names, seeds, task_counts, signatures


def _load_existing_names(paths: Iterable[str | Path] | None) -> set[str]:
    """Read group names from existing YAML plan files or directories."""
    names, _seeds, _task_counts, _signatures = _load_existing_metadata(paths)
    return names


def _task_profile(task: str) -> dict:
    """Return the report profile, with a conservative fallback for new tasks."""
    return TASK_STATS.get(
        task, {"difficulty": "D1", "episodes": 0, "participation": 0.0}
    )


def _difficulty(task: str) -> str:
    return str(_task_profile(task)["difficulty"])


def _difficulty_rank(task: str) -> int:
    return _DIFFICULTY_RANK.get(_difficulty(task), 1)


def _group_cap(tasks: Sequence[str], max_tasks: int, adaptive: bool) -> int:
    """Limit mixed-scene size so low-yield tasks get focused placements."""
    if not adaptive or not tasks:
        return max_tasks
    hardest = max(
        (_difficulty(task) for task in tasks), key=lambda level: _DIFFICULTY_RANK[level]
    )
    cap = _GROUP_CAP_BY_DIFFICULTY[hardest]
    return min(max_tasks, cap) if cap is not None else max_tasks


def _group_difficulty(tasks: Sequence[str]) -> str:
    return max(
        (_difficulty(task) for task in tasks), key=lambda level: _DIFFICULTY_RANK[level]
    )


def _adaptive_quota(
    tasks: Sequence[str], supported_task_count: int, fallback: str, max_search: int
) -> tuple[str, int]:
    """Derive quota from compatible task count and use the fixed search budget.

    Difficulty is intentionally not used to lower quota: a difficult task is
    still valuable whenever a multi-task placement succeeds.  Only scenes that
    support one task use a ``1:*`` bucket.
    """
    if not tasks:
        return fallback, DEFAULT_MAX_SEARCH
    if supported_task_count <= 1:
        quota = "1:2"
    elif supported_task_count == 2:
        quota = "2:4"
    elif supported_task_count == 3:
        quota = "2:4,3:3"
    else:
        # Match the high-coverage shape used by the existing large plans.
        quota = "2:3,3:3,4:1"
    return quota, DEFAULT_MAX_SEARCH


def _has_placement_conflict(tasks: Sequence[str]) -> bool:
    task_set = set(tasks)
    return any(
        task_set & left and task_set & right for left, right in PLACEMENT_CONFLICTS
    )


def _trial(
    tasks: Sequence[str], seed: int, max_objects: int
) -> tuple[object, str] | None:
    """Build and statically validate one candidate group."""
    if _has_placement_conflict(tasks):
        return None
    scene, scene_id, skipped = mixed_scene_for_tasks(tasks, seed=seed)
    if skipped or len(scene.objects) > max_objects:
        return None
    matched = {match.task_name for match in candidate_tasks(scene, list(tasks))}
    if not set(tasks) <= matched:
        return None
    return scene, scene_id


def _group_name(index: int, scene_id: str) -> str:
    """Return a pipeline-safe, globally unique group name."""
    safe = "".join(char if char.isalnum() or char in "_-" else "_" for char in scene_id)
    return f"{safe or 'scene'}"


def _assign_unique_names(
    groups: Sequence[dict], existing_names: Iterable[str] | None
) -> None:
    """Assign deterministic names while avoiding names from prior plans."""
    used = {str(name) for name in (existing_names or ()) if str(name).strip()}
    for index, group in enumerate(groups):
        base = _group_name(index, str(group.get("_scene_id", "scene")))
        candidate = base
        suffix = 1
        while candidate in used:
            candidate = f"{base}_v{suffix:02d}"
            suffix += 1
        group["name"] = candidate
        used.add(candidate)


def _group_defaults(
    tasks: Sequence[str],
    supported_task_count: int,
    quota: str,
    max_search: int,
    *,
    adaptive: bool = False,
) -> tuple[str, int]:
    """Apply known runtime constraints for special task groups."""
    if adaptive:
        quota, _ = _adaptive_quota(tasks, supported_task_count, quota, max_search)
    return quota, DEFAULT_MAX_SEARCH


def generate_plan(
    tasks: Iterable[str] | None = None,
    *,
    exclude_tasks: Iterable[str] | None = None,
    background_tasks: Iterable[str] | None = None,
    seed: int = 0,
    max_tasks: int = 6,
    max_objects: int = 6,
    min_objects: int = DEFAULT_MIN_OBJECTS,
    min_tasks: int = 2,
    count: int | None = None,
    scene_config: str = DEFAULT_SCENE_CONFIG,
    quota: str | None = None,
    max_search: int = DEFAULT_MAX_SEARCH,
    visual: bool = True,
    visual_variants: int = 2,
    adaptive: bool = True,
    existing_names: Iterable[str] | None = None,
    existing_seeds: Iterable[int] | None = None,
    existing_task_counts: Mapping[str, int] | None = None,
    existing_signatures: Iterable[Sequence[str]] | None = None,
    max_block_ratio: float = DEFAULT_MAX_BLOCK_RATIO,
    max_block_tasks: int | None = None,
) -> list[dict]:
    """Partition tasks into deterministic, statically valid scene groups.

    ``count`` requests an exact number of groups. In that mode tasks may be
    reused across groups; without it, every requested task is emitted once.
    """
    task_list = _normalise_tasks(tasks)
    excluded = _normalise_exclusions(exclude_tasks)
    task_list = [task for task in task_list if task not in excluded]
    background_list = _normalise_tasks(background_tasks or [])
    if max_tasks < 1:
        raise ValueError("max_tasks must be at least 1")
    if max_objects < 1:
        raise ValueError("max_objects must be at least 1")
    if min_objects < 1:
        raise ValueError("min_objects must be at least 1")
    if min_tasks < 1:
        raise ValueError("min_tasks must be at least 1")
    if min_tasks > max_tasks:
        raise ValueError("min_tasks cannot exceed max_tasks")
    if count is not None and count < 1:
        raise ValueError("count must be at least 1 when specified")
    if count is not None and not task_list:
        raise ValueError("at least one task is required when count is specified")
    if max_search < 1:
        raise ValueError("max_search must be at least 1")
    if not 0 <= max_block_ratio <= 1:
        raise ValueError("max_block_ratio must be between 0 and 1")
    if max_block_tasks is not None and max_block_tasks < 0:
        raise ValueError("max_block_tasks must be non-negative")
    if visual_variants < 1:
        raise ValueError("visual_variants must be at least 1")

    # A supplied quota is an explicit escape hatch for reproducibility.  The
    # default path uses the difficulty snapshot and adapts each group.
    adaptive = bool(adaptive and quota is None)
    base_quota = DEFAULT_QUOTA if quota is None else str(quota)
    target_objects = min(min_objects, max_objects)
    mandatory_block_tasks = len(set(task_list) & BLOCK_TASKS)
    if max_block_tasks is not None:
        block_budget = max_block_tasks
    elif count is not None:
        block_budget = math.ceil(count * max_block_ratio)
    else:
        # Partition mode emits every requested task once, so no extra cap is
        # needed beyond retaining the requested block-task coverage.
        block_budget = mandatory_block_tasks
    block_budget = max(block_budget, mandatory_block_tasks)
    non_block_tasks = [task for task in task_list if task not in BLOCK_TASKS]
    remaining = set(task_list)
    rng = random.Random(seed)
    task_use_counts = Counter(
        {str(task): int(count) for task, count in (existing_task_counts or {}).items()}
    )
    order: dict[str, int] = {}

    def refresh_order() -> None:
        """Shuffle tasks within each difficulty tier for this sampling cycle."""
        order.clear()
        cursor = 0
        for rank in sorted(_DIFFICULTY_RANK.values(), reverse=True):
            tier = [task for task in task_list if _difficulty_rank(task) == rank]
            rng.shuffle(tier)
            for task in tier:
                order[task] = cursor
                cursor += 1

    refresh_order()
    used_seeds = {int(value) for value in (existing_seeds or ())}
    next_seed = int(seed)
    while next_seed in used_seeds:
        next_seed += 1
    groups: list[dict] = []
    seen_signatures: set[tuple[str, ...]] = {
        _task_signature(signature) for signature in (existing_signatures or ())
    }
    block_used = 0

    target_count = count if count is not None else None
    while (target_count is not None and len(groups) < target_count) or (
        target_count is None and remaining
    ):
        # Exact-count plans are a sampling plan rather than a partition: once
        # every task has been used, refill the pool so later groups can reuse
        # tasks while retaining unique group seeds.
        if target_count is not None and not remaining:
            refill_tasks = (
                non_block_tasks
                if block_used >= block_budget and non_block_tasks
                else task_list
            )
            remaining = set(refill_tasks)
            refresh_order()
        if target_count is not None and block_used >= block_budget:
            non_block_remaining = {
                task for task in remaining if task not in BLOCK_TASKS
            }
            if not non_block_remaining and non_block_tasks:
                remaining = set(non_block_tasks)
                refresh_order()
        # Each cycle has a seeded permutation, so repeated task pools do not
        # replay the same group order or pairing pattern.
        eligible = remaining
        if block_used >= block_budget:
            non_block = {task for task in remaining if task not in BLOCK_TASKS}
            if non_block:
                eligible = non_block
        hardest_rank = max(_difficulty_rank(task) for task in eligible)
        tier = [task for task in eligible if _difficulty_rank(task) == hardest_rank]
        first = min(tier, key=lambda task: (task_use_counts[task], order[task], task))
        remaining.remove(first)
        current = [first]
        group_seed = next_seed
        used_seeds.add(group_seed)
        next_seed += 1
        while next_seed in used_seeds:
            next_seed += 1
        current_block_count = int(first in BLOCK_TASKS)

        if first not in ISOLATED_TASKS and _difficulty(first) != "D4":
            while len(current) < max_tasks:
                candidates: list[tuple[tuple, str, str]] = []
                base = _trial(current, group_seed, max_objects)
                if base is None:  # defensive; every registry task should stand alone
                    break
                base_objects = len(base[0].objects)
                if base_objects >= target_objects:
                    break
                candidate_pool = set(remaining)
                if target_count is not None and base_objects < target_objects:
                    refill_tasks = (
                        non_block_tasks
                        if block_used >= block_budget and non_block_tasks
                        else task_list
                    )
                    candidate_pool.update(refill_tasks)
                    candidate_pool.difference_update(current)
                if not candidate_pool:
                    break
                minimum_load = min(task_use_counts[task] for task in candidate_pool)
                zero_gain_candidates: list[tuple[tuple, str, str]] = []
                for candidate in sorted(
                    candidate_pool, key=lambda task: (order[task], task)
                ):
                    if candidate in ISOLATED_TASKS or _difficulty(candidate) == "D4":
                        continue
                    if candidate in BLOCK_TASKS:
                        if block_used + current_block_count + 1 > block_budget:
                            continue
                    trial = _trial(current + [candidate], group_seed, max_objects)
                    if trial is None:
                        continue
                    scene, scene_id = trial
                    object_count = len(scene.objects)
                    object_gain = object_count - base_objects
                    # Keep similarly difficult tasks together: this avoids
                    # burying D3/D2 tasks in a high-yield D0 mixed group.
                    tie = rng.random()
                    spread = abs(
                        _difficulty_rank(candidate) - _difficulty_rank(current[0])
                    )
                    signature = tuple(sorted(current + [candidate]))
                    repeated = int(signature in seen_signatures)
                    shortfall = max(0, target_objects - object_count)
                    reused = int(candidate not in remaining)
                    load_penalty = task_use_counts[candidate] - minimum_load
                    score = (
                        (
                            load_penalty,
                            repeated,
                            shortfall,
                            -object_count,
                            reused,
                            spread,
                            tie,
                            candidate,
                        ),
                        candidate,
                        scene_id,
                    )
                    if object_gain > 0:
                        candidates.append(score)
                    else:
                        zero_gain_candidates.append(score)
                if not candidates:
                    candidates = zero_gain_candidates
                if not candidates:
                    break
                _score, chosen, _scene_id = min(candidates)
                current.append(chosen)
                current_block_count += int(chosen in BLOCK_TASKS)
                remaining.discard(chosen)

        final = _trial(current, group_seed, max_objects)
        if final is None:
            # This should only be reachable after a future registry change.  A
            # clear error is safer than emitting a plan the pipeline cannot run.
            raise ValueError(f"could not build a valid scene for tasks: {current}")
        _scene, scene_id = final
        block_used += current_block_count
        signature = tuple(sorted(current))
        seen_signatures.add(signature)
        supported_task_count = len(candidate_tasks(_scene))
        group_quota, group_search = _group_defaults(
            current, supported_task_count, base_quota, max_search, adaptive=adaptive
        )
        groups.append(
            {
                "name": "",
                "tasks": current,
                "quota": str(group_quota),
                "max_search": int(group_search),
                "group_seed": int(group_seed),
                "difficulty": _group_difficulty(current),
                "visual": bool(visual),
                "visual_variants": int(visual_variants if visual else 1),
                "_object_count": len(_scene.objects),
                "_supported_task_count": supported_task_count,
                "_scene_id": scene_id,
            }
        )
        task_use_counts.update(current)

    if count is not None and len(groups) != count:
        raise ValueError(
            f"could not generate exactly {count} groups from {len(task_list)} tasks; "
            f"try increasing the task set or lowering --count"
        )

    # Repair a trailing singleton when the registry permits merging it into an
    # existing group.  Some tasks are intentionally isolated (for example the
    # dual-shoe task because it needs a special table height); those remain
    # singleton groups by design.
    if count is None and min_tasks > 1:
        index = 0
        while index < len(groups):
            group = groups[index]
            if (
                len(group["tasks"]) >= min_tasks
                or any(task in ISOLATED_TASKS for task in group["tasks"])
                or any(_difficulty(task) == "D4" for task in group["tasks"])
            ):
                index += 1
                continue
            choices = []
            for other_index, other in enumerate(groups):
                if other_index == index or any(
                    task in ISOLATED_TASKS for task in other["tasks"]
                ):
                    continue
                merged_tasks = other["tasks"] + group["tasks"]
                if len(merged_tasks) > _group_cap(merged_tasks, max_tasks, adaptive):
                    continue
                if adaptive:
                    if any(_difficulty(task) == "D4" for task in merged_tasks):
                        continue
                    levels = [_difficulty_rank(task) for task in merged_tasks]
                    if max(levels) - min(levels) > 1:
                        continue
                trial = _trial(merged_tasks, other["group_seed"], max_objects)
                if trial is None:
                    continue
                scene, scene_id = trial
                choices.append(
                    (
                        len(scene.objects),
                        len(merged_tasks),
                        other_index,
                        scene_id,
                        scene,
                    )
                )
            if not choices:
                index += 1
                continue
            _objects, _task_count, target_index, scene_id, scene = min(choices)
            target = groups[target_index]
            target["tasks"] = target["tasks"] + group["tasks"]
            target["_object_count"] = len(scene.objects)
            target["_supported_task_count"] = len(candidate_tasks(scene))
            target["_scene_id"] = scene_id
            target["quota"], target["max_search"] = _group_defaults(
                target["tasks"],
                target["_supported_task_count"],
                base_quota,
                max_search,
                adaptive=adaptive,
            )
            target["difficulty"] = _group_difficulty(target["tasks"])
            del groups[index]
            if target_index > index:
                target_index -= 1
            # Recheck from the beginning because removing a group changes the
            # candidate set and can make another singleton mergeable.
            index = 0

    # Object providers and collection eligibility are independent. This also
    # lets isolated OOD subjects receive background objects.
    allowed = set(task_list)
    scene_pool = list(dict.fromkeys(task_list + background_list))
    background_use = Counter()
    for group in groups:
        current = group["tasks"]
        scene, scene_id = _trial(current, group["group_seed"], max_objects)
        while len(current) < max_tasks and len(scene.objects) < target_objects:
            choices = []
            for task in scene_pool:
                if (task in current or "place_dual_shoes" in current
                        or task == "place_dual_shoes"):
                    continue
                if task in BLOCK_TASKS and block_used >= block_budget:
                    continue
                trial = _trial(current + [task], group["group_seed"], max_objects)
                if trial is None or len(trial[0].objects) <= len(scene.objects):
                    continue
                choices.append((background_use[task], rng.random(), task, trial))
            if not choices:
                break
            _, _, chosen, (scene, scene_id) = min(choices)
            current.append(chosen)
            background_use[chosen] += 1
            block_used += int(chosen in BLOCK_TASKS)
        collected = [task for task in current if task in allowed]
        group["required_tasks"] = collected
        # The collector and resume checks already honor exclude_tasks, including
        # incidental static matches outside the collection list.
        group["exclude_tasks"] = sorted(set(all_task_names()) - set(collected))
        group["_object_count"] = len(scene.objects)
        group["_supported_task_count"] = len(collected)
        group["_scene_id"] = scene_id
        group["quota"], group["max_search"] = _group_defaults(
            collected, len(collected), base_quota, max_search, adaptive=adaptive
        )
        group["difficulty"] = _group_difficulty(collected)

    _assign_unique_names(groups, existing_names)
    return groups


def plan_document(
    groups: Sequence[dict],
    *,
    scene_config: str = DEFAULT_SCENE_CONFIG,
    visual: bool = True,
    visual_variants: int = 2,
) -> dict:
    """Build the YAML document consumed by ``run_data_pipeline.load_plan``."""
    clean_groups = []
    for group in groups:
        clean_groups.append(
            {key: value for key, value in group.items() if not key.startswith("_")}
        )
    return {
        "scene_config": scene_config,
        "visual": bool(visual),
        "visual_variants": int(visual_variants if visual else 1),
        "generator": "script/generate_scene_plans.py",
        "groups": clean_groups,
    }


def _load_task_settings(path: str | Path) -> tuple[list[str], list[str]]:
    """Read scene task pools and collection targets, or a legacy text list.

    tasks supplies all scene object providers. required_tasks selects collection
    targets and defaults to tasks. Return targets and background-only providers.
    """
    path = Path(path).resolve()
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() not in {".yaml", ".yml"}:
        return [item.strip() for item in text.replace("\n", ",").split(",") if item.strip()], []
    try:
        settings = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid task settings in {path}: {exc}") from exc
    if not isinstance(settings, dict) or "tasks" not in settings:
        raise ValueError(f"{path}: expected a mapping with tasks")
    unknown = set(settings) - {"tasks", "required_tasks"}
    if unknown:
        raise ValueError(f"{path}: unknown task settings: {sorted(unknown)}")
    tasks = settings["tasks"]
    required = settings.get("required_tasks", tasks)
    for key, values in (("tasks", tasks), ("required_tasks", required)):
        if not isinstance(values, list) or any(not isinstance(v, str) or not v.strip() for v in values):
            raise ValueError(f"{path}: {key} must be a list of task names")
        if not values:
            raise ValueError(f"{path}: {key} must not be empty")
    tasks = _normalise_tasks(tasks)
    required = _normalise_tasks(required)
    missing = sorted(set(required) - set(tasks))
    if missing:
        raise ValueError(f"{path}: required_tasks must be drawn from tasks: {missing}")
    return required, [task for task in tasks if task not in set(required)]


def _parse_tasks(value: str | None, tasks_file: str | None) -> list[str] | None:
    if value and tasks_file:
        raise ValueError("use only one of --tasks and --tasks-file")
    if value:
        return [item.strip() for item in value.split(",") if item.strip()]
    if tasks_file:
        return _load_task_settings(tasks_file)[0]
    return None


def _parse_paths(values: Sequence[str] | None) -> list[Path]:
    """Normalize repeatable plan path arguments."""
    return [Path(value) for value in (values or ())]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=None, help="YAML output path; omit to print YAML"
    )
    parser.add_argument(
        "--tasks",
        default=None,
        help="comma-separated task names (default: all registered tasks)",
    )
    parser.add_argument(
        "--tasks-file",
        default=None,
        help="YAML collection/background settings or legacy text task list",
    )
    parser.add_argument(
        "--exclude-tasks", default=None, help="comma-separated task names to exclude"
    )
    parser.add_argument(
        "--existing-plan",
        action="append",
        default=[],
        help="existing YAML plan or directory; repeatable for name de-duplication",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Starting seed; if omitted, generated randomly for each CLI invocation"
    )
    parser.add_argument(
        "--count", type=int, default=None, help="exact number of groups to generate"
    )
    parser.add_argument("--max-tasks", type=int, default=6)
    parser.add_argument("--max-objects", type=int, default=6)
    parser.add_argument(
        "--min-objects",
        type=int,
        default=DEFAULT_MIN_OBJECTS,
        help="preferred minimum object count (default: 5; capped by --max-objects)",
    )
    parser.add_argument(
        "--min-tasks",
        type=int,
        default=2,
        help="minimum desired size (singletons remain valid when necessary)",
    )
    parser.add_argument("--scene-config", default=DEFAULT_SCENE_CONFIG)
    parser.add_argument(
        "--quota",
        default=None,
        help="fixed quota (default: adapt from compatible task count)",
    )
    parser.add_argument(
        "--max-search",
        type=int,
        default=DEFAULT_MAX_SEARCH,
        help="Legacy compatibility option; max_search in generated plans is always fixed at 200",
    )
    parser.add_argument(
        "--max-block-ratio",
        type=float,
        default=DEFAULT_MAX_BLOCK_RATIO,
        help="Maximum fraction of groups using block scene-building tasks in batch plans (default: 0.20)",
    )
    parser.add_argument(
        "--max-block-tasks",
        type=int,
        default=None,
        help="Absolute limit on block scene-building task groups in batch plans; takes precedence over the fraction",
    )
    parser.add_argument(
        "--no-adaptive",
        action="store_true",
        help="disable difficulty-based grouping/quota adaptation",
    )
    parser.add_argument("--no-visual", action="store_true")
    parser.add_argument("--visual-variants", type=int, default=2)
    args = parser.parse_args(argv)

    try:
        if args.tasks and args.tasks_file:
            raise ValueError("use only one of --tasks and --tasks-file")
        if args.tasks_file:
            tasks, background_tasks = _load_task_settings(args.tasks_file)
        else:
            tasks = _parse_tasks(args.tasks, None)
            background_tasks = []
        exclude_tasks = _parse_tasks(args.exclude_tasks, None)
        (existing_names, existing_seeds, existing_task_counts, existing_signatures) = (
            _load_existing_metadata(_parse_paths(args.existing_plan))
        )
        seed = (
            args.seed
            if args.seed is not None
            else random.SystemRandom().randrange(0, 2**31)
        )
        groups = generate_plan(
            tasks,
            exclude_tasks=exclude_tasks,
            background_tasks=background_tasks,
            seed=seed,
            max_tasks=args.max_tasks,
            max_objects=args.max_objects,
            min_objects=args.min_objects,
            min_tasks=args.min_tasks,
            count=args.count,
            scene_config=args.scene_config,
            quota=args.quota,
            max_search=args.max_search,
            max_block_ratio=args.max_block_ratio,
            max_block_tasks=args.max_block_tasks,
            visual=not args.no_visual,
            visual_variants=args.visual_variants,
            adaptive=not args.no_adaptive,
            existing_names=existing_names,
            existing_seeds=existing_seeds,
            existing_task_counts=existing_task_counts,
            existing_signatures=existing_signatures,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    document = plan_document(
        groups,
        scene_config=args.scene_config,
        visual=not args.no_visual,
        visual_variants=args.visual_variants,
    )
    rendered = yaml.safe_dump(document, sort_keys=False, allow_unicode=True)
    if args.output is None:
        sys.stdout.write(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"Generated {len(groups)} groups -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
