"""Scene layer — pure-logic models plus the run-time placer.

Pure-logic pieces (no SAPIEN dependency):
  - task requirements (task -> required object roles)   task_requirements
  - static matching (object set -> candidate tasks + binding)  static_matcher
Run-time placement/binding lives in object_placer and _base_task.
"""

from .roles import RoleSpec, TaskRequirement, ObjectInstance, SceneObjectSet
from .task_requirements import (
    REGISTRY,
    get_requirement,
    all_task_names,
    OOD_EVAL_TASKS,
    PRIMITIVE_BLOCK,
    TASK_TABLE_Z_BASELINES,
    task_table_z_baseline,
)
from .static_matcher import (
    match_task,
    match_from_binding,
    candidate_tasks,
    candidate_tasks_summary,
)
from .object_placer import ObjectPlacer, PlacementError
from .pair_perturbation import (
    PAIR_PERTURBATION_DEFAULTS,
    PAIR_PERTURBATION_MULTI_DEFAULTS,
    PerturbationInfeasible,
    normalize_pair_perturbation,
    perturbation_seed,
    yaw_premultiply,
    sample_pair_offsets,
)
from .scene_spec import SceneSpec, SCENE_SPEC_VERSION
from .environment import sample_environment, DEFAULT_ENVIRONMENT
from .scene_grouping import mixed_scene_for_tasks

__all__ = [
    "RoleSpec",
    "TaskRequirement",
    "ObjectInstance",
    "SceneObjectSet",
    "REGISTRY",
    "get_requirement",
    "all_task_names",
    "OOD_EVAL_TASKS",
    "PRIMITIVE_BLOCK",
    "TASK_TABLE_Z_BASELINES",
    "task_table_z_baseline",
    "match_task",
    "match_from_binding",
    "candidate_tasks",
    "candidate_tasks_summary",
    "ObjectPlacer",
    "PlacementError",
    "PAIR_PERTURBATION_DEFAULTS",
    "PAIR_PERTURBATION_MULTI_DEFAULTS",
    "PerturbationInfeasible",
    "normalize_pair_perturbation",
    "perturbation_seed",
    "yaw_premultiply",
    "sample_pair_offsets",
    "SceneSpec",
    "SCENE_SPEC_VERSION",
    "sample_environment",
    "DEFAULT_ENVIRONMENT",
    "mixed_scene_for_tasks",
]
