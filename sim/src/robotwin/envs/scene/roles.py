"""Scene object and task-role data models without simulation dependencies.

RoleSpec describes accepted models, count, and kind; TaskRequirement groups roles.
ObjectInstance identifies one scene object; SceneObjectSet holds the object set.
Kinds are rigid assets, articulated URDFs, or generated primitive_box objects.
Role names identify the task environment's self.<name> actor handles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------- task side


@dataclass(frozen=True)
class RoleSpec:
    """A task's need for one object role."""

    name: str  # binding handle name (without "self.")
    accept: frozenset[str]  # accepted modelnames (expanded to concrete names)
    count: int = 1  # number of distinct objects for this role
    kind: str = "rigid"  # rigid | articulated | primitive_box
    graspable: bool = True  # whether the object is grasped (informational)
    requires_binding: bool = False  # only accept instances reserved for this task/role

    def accepts(self, modelname: str) -> bool:
        return modelname in self.accept


@dataclass(frozen=True)
class TaskRequirement:
    """All object-role needs of a task."""

    task_name: str
    roles: tuple[RoleSpec, ...]
    arms: str = "any"  # single | dual | any (informational)
    source: str = "task_info"  # task_info (auto) | override (manual)

    @property
    def total_objects(self) -> int:
        return sum(r.count for r in self.roles)

    @property
    def required_modelnames(self) -> set[str]:
        """Union of all modelnames the task may involve."""
        out: set[str] = set()
        for r in self.roles:
            out |= set(r.accept)
        return out


# ---------------------------------------------------------------- scene side


@dataclass(frozen=True)
class ObjectInstance:
    """A concrete object in a scene set."""

    oid: str  # scene-unique id, e.g. "bottle#0"
    modelname: str  # e.g. "001_bottle" / "sapien-block"
    kind: str = "rigid"
    model_id: Optional[int] = None  # asset sub-model id (may be None)
    # primitive_box only: {half_size:[x,y,z], color:[r,g,b], qpos:[w,x,y,z]}.
    # A block is a create_box prop whose geometry and colour are task semantics,
    # so they're carried per-instance and consumed by _place_box.
    box_spec: Optional[dict] = None
    # Structured placement: {xlim:[a,b], ylim:[a,b], ...} for "central container +
    # side objects" tasks; overrides the placer's default reachable region.
    place_region: Optional[dict] = None
    # A structured layout owns some instances for one task. This prevents a
    # same-model generic object from being bound in place of a side-specific one.
    binding_task: Optional[str] = None
    binding_role: Optional[str] = None


@dataclass
class SceneObjectSet:
    """All objects on a scene's table."""

    objects: list[ObjectInstance] = field(default_factory=list)

    def add(
        self,
        modelname: str,
        kind: str = "rigid",
        model_id: Optional[int] = None,
        oid: Optional[str] = None,
        box_spec: Optional[dict] = None,
        place_region: Optional[dict] = None,
        binding_task: Optional[str] = None,
        binding_role: Optional[str] = None,
    ) -> ObjectInstance:
        if oid is None:
            n = sum(1 for o in self.objects if o.modelname == modelname)
            oid = f"{modelname}#{n}"
        inst = ObjectInstance(
            oid=oid,
            modelname=modelname,
            kind=kind,
            model_id=model_id,
            box_spec=box_spec,
            place_region=place_region,
            binding_task=binding_task,
            binding_role=binding_role,
        )
        self.objects.append(inst)
        return inst

    @property
    def modelnames(self) -> list[str]:
        return [o.modelname for o in self.objects]

    @classmethod
    def from_modelnames(cls, names) -> "SceneObjectSet":
        """Build an object set from a list of modelnames (kind inferred)."""
        from .task_requirements import infer_kind

        s = cls()
        for nm in names:
            s.add(nm, kind=infer_kind(nm))
        return s
