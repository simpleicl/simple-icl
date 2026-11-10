"""Frozen scene descriptions for task-driven collection."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict

from .roles import SceneObjectSet
from .environment import DEFAULT_ENVIRONMENT

SCENE_SPEC_VERSION = 3


@dataclass
class SceneSpec:
    scene_id: str
    environment: dict = field(default_factory=lambda: dict(DEFAULT_ENVIRONMENT))
    objects: list[dict] = field(default_factory=list)
    supported_tasks: list[dict] = field(default_factory=list)
    request: dict = field(default_factory=dict)
    version: int = SCENE_SPEC_VERSION

    @classmethod
    def from_object_set(
        cls, scene_id: str, scene_set: SceneObjectSet, environment: dict | None = None
    ) -> "SceneSpec":
        """Build a serializable scene from object instances."""
        spec = cls(
            scene_id=scene_id,
            environment=dict(environment) if environment else dict(DEFAULT_ENVIRONMENT),
        )
        for o in scene_set.objects:
            entry = {
                "oid": o.oid,
                "modelname": o.modelname,
                "kind": o.kind,
                "model_id": o.model_id,
            }
            if o.box_spec is not None:
                entry["box_spec"] = o.box_spec
            if o.place_region is not None:
                entry["place_region"] = o.place_region
            if o.binding_task is not None:
                entry["binding_task"] = o.binding_task
            if o.binding_role is not None:
                entry["binding_role"] = o.binding_role
            spec.objects.append(entry)
        return spec

    def build_object_set(self) -> SceneObjectSet:
        """Rebuild the object set used for placement and matching."""
        s = SceneObjectSet()
        for o in self.objects:
            s.add(
                modelname=o["modelname"],
                kind=o.get("kind", "rigid"),
                model_id=o.get("model_id"),
                oid=o["oid"],
                box_spec=o.get("box_spec"),
                place_region=o.get("place_region"),
                binding_task=o.get("binding_task"),
                binding_role=o.get("binding_role"),
            )
        return s

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SceneSpec":
        return cls(
            scene_id=d["scene_id"],
            environment=dict(d.get("environment") or DEFAULT_ENVIRONMENT),
            objects=list(d.get("objects", [])),
            supported_tasks=list(d.get("supported_tasks", [])),
            request=dict(d.get("request", {})),
            version=d.get("version", SCENE_SPEC_VERSION),
        )

    @classmethod
    def load(cls, path) -> "SceneSpec":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
