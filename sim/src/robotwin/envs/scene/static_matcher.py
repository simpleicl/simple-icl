"""Match scene objects to task roles.

Each role needs at least count objects with an accepted model name.
Return candidate tasks with concrete role-to-object bindings.
"""

from __future__ import annotations

from typing import NamedTuple

from .roles import TaskRequirement, RoleSpec, SceneObjectSet, ObjectInstance
from .task_requirements import REGISTRY


class RoleBinding(NamedTuple):
    """Objects assigned to one task role."""

    role: RoleSpec
    objects: list[ObjectInstance]  # len == role.count


class TaskMatch(NamedTuple):
    """A task and its matched role bindings."""

    task_name: str
    requirement: TaskRequirement
    bindings: list[RoleBinding]

    @property
    def bound_objects(self) -> set[str]:
        """Return bound object IDs to prevent duplicate assignments."""
        out = set()
        for b in self.bindings:
            out.update(o.oid for o in b.objects)
        return out


def match_task(task_req: TaskRequirement, scene: SceneObjectSet) -> TaskMatch | None:
    """Greedily bind available objects in role order.

    Return TaskMatch if every role has enough objects; otherwise return None.
    """
    available = {o.oid: o for o in scene.objects}  # oid → ObjectInstance
    bindings: list[RoleBinding] = []

    for role in task_req.roles:
        # Structured layouts reserve their role-specific instances. The owner task
        # must use its exact role; other tasks may not steal those anchors.
        reserved = [
            obj
            for obj in available.values()
            if obj.binding_task == task_req.task_name and obj.binding_role == role.name
        ]
        if role.requires_binding and len(reserved) < role.count:
            return None
        if reserved:
            candidates = reserved
        else:
            candidates = [
                obj
                for obj in available.values()
                if role.accepts(obj.modelname)
                and (obj.binding_task is None or obj.binding_task == task_req.task_name)
            ]
        if len(candidates) < role.count:
            return None

        selected = candidates[: role.count]
        bindings.append(RoleBinding(role=role, objects=selected))

        # Remove assigned objects so roles cannot share an instance.
        for obj in selected:
            del available[obj.oid]

    return TaskMatch(
        task_name=task_req.task_name, requirement=task_req, bindings=bindings
    )


def match_from_binding(
    task_req: TaskRequirement, scene: SceneObjectSet, binding: dict
) -> TaskMatch | None:
    """Rebuild and validate a persisted `{role: [oid...]}` task binding."""
    if not isinstance(binding, dict):
        return None
    by_oid = {obj.oid: obj for obj in scene.objects}
    used = set()
    bindings = []
    for role in task_req.roles:
        oids = binding.get(role.name)
        if not isinstance(oids, list) or len(oids) != role.count:
            return None
        reserved_oids = {
            obj.oid
            for obj in scene.objects
            if obj.binding_task == task_req.task_name and obj.binding_role == role.name
        }
        if role.requires_binding and len(reserved_oids) < role.count:
            return None
        if reserved_oids and set(oids) != reserved_oids:
            return None
        objects = []
        for oid in oids:
            obj = by_oid.get(oid)
            if (
                obj is None
                or oid in used
                or not role.accepts(obj.modelname)
                or (
                    obj.binding_task is not None
                    and (
                        obj.binding_task != task_req.task_name
                        or obj.binding_role != role.name
                    )
                )
            ):
                return None
            used.add(oid)
            objects.append(obj)
        bindings.append(RoleBinding(role=role, objects=objects))
    if set(binding) != {role.name for role in task_req.roles}:
        return None
    return TaskMatch(
        task_name=task_req.task_name, requirement=task_req, bindings=bindings
    )


def candidate_tasks(
    scene: SceneObjectSet, task_filter: list[str] | None = None
) -> list[TaskMatch]:
    """Return matching tasks sorted by name.

    Check task_filter when provided; otherwise check the full registry.
    """
    tasks = REGISTRY.values()
    if task_filter:
        tasks = [REGISTRY[tn] for tn in task_filter if tn in REGISTRY]

    matches: list[TaskMatch] = []
    for req in tasks:
        m = match_task(req, scene)
        if m:
            matches.append(m)

    return sorted(matches, key=lambda x: x.task_name)


def candidate_tasks_summary(
    scene: SceneObjectSet, task_filter: list[str] | None = None
) -> dict:
    """Summarize candidate tasks and the model names bound to each role."""
    matches = candidate_tasks(scene, task_filter)
    summary = {}
    for m in matches:
        bindings_view = {
            b.role.name: [o.modelname for o in b.objects] for b in m.bindings
        }
        summary[m.task_name] = bindings_view
    return summary
