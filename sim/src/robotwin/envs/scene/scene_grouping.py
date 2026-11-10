"""Merge task requirements into a scene object set for collection."""

from __future__ import annotations

import re
import random
from collections import defaultdict

from .roles import TaskRequirement, SceneObjectSet
from .task_requirements import REGISTRY

# Pin validated model variants for geometry-sensitive tasks.
PREFERRED_MODEL_IDS: dict[str, int] = {"002_bowl": 3}


def _shortname(modelname: str) -> str:
    """Strip the numeric NNN_ asset prefix for scene IDs."""
    m = re.match(r"^\d+_(.*)$", modelname)
    return m.group(1) if m else modelname


def _auto_scene_id(counts: dict[str, int]) -> str:
    """Build a scene ID from sorted <short_name><count> entries, e.g. bottle2_bowl3."""
    return "_".join(f"{_shortname(m)}{counts[m]}" for m in sorted(counts))


def _is_primitive_task(req: TaskRequirement) -> bool:
    return any(r.kind == "primitive_box" for r in req.roles)


def _merge_object_set(dst: SceneObjectSet, src: SceneObjectSet) -> None:
    """Merge src into dst, assigning new oids to avoid cross-task collisions."""
    for o in src.objects:
        dst.add(
            o.modelname,
            kind=o.kind,
            model_id=o.model_id,
            box_spec=o.box_spec,
            place_region=o.place_region,
            binding_task=o.binding_task,
            binding_role=o.binding_role,
        )


def mixed_scene_for_tasks(task_names, scene_id: str | None = None, seed: int = 0):
    """Merge ordinary, block, and structured tasks; return objects, scene ID, and unknown tasks."""
    from .task_requirements import (
        infer_kind,
        build_primitive_scene,
        build_structured_scene,
        _STRUCTURED_LAYOUT,
    )

    rng = random.Random(seed)
    skipped: list[tuple[str, str]] = []
    primitive_tasks: list[str] = []
    structured_tasks: list[str] = []
    plain_required: dict[str, int] = {}

    for tn in task_names:
        req = REGISTRY.get(tn)
        if req is None:
            skipped.append((tn, "unknown"))
            continue
        if tn in _STRUCTURED_LAYOUT:
            structured_tasks.append(tn)
            continue
        if _is_primitive_task(req):
            primitive_tasks.append(tn)
            continue
        per_task: dict[str, int] = defaultdict(int)
        for role in req.roles:
            if len(role.accept) == 1:
                (m,) = tuple(role.accept)
            else:
                m = rng.choice(sorted(role.accept))
            per_task[m] += role.count
        for m, c in per_task.items():
            plain_required[m] = max(plain_required.get(m, 0), c)

    scene_set = SceneObjectSet()
    for m in sorted(plain_required):
        for _ in range(plain_required[m]):
            scene_set.add(m, kind=infer_kind(m), model_id=PREFERRED_MODEL_IDS.get(m))
    for tn in primitive_tasks:
        _merge_object_set(scene_set, build_primitive_scene(tn))
    for tn in structured_tasks:
        _merge_object_set(scene_set, build_structured_scene(tn))

    counts: dict[str, int] = defaultdict(int)
    for o in scene_set.objects:
        counts[o.modelname] += 1
    sid = scene_id or _auto_scene_id(counts)
    return scene_set, sid, skipped
