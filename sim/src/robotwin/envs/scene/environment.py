"""Sample fixed scene settings without simulation dependencies.

Freeze table_z_bias once per scene to preserve reachability during verification.
Optional table_xy_bias overrides task-local offsets. A dedicated seeded RNG
preserves the global placement RNG; visual randomization happens during collection.
"""

from __future__ import annotations

import numpy as np

DEFAULT_ENVIRONMENT = {"table_z_bias": 0.0}


def sample_environment(
    generation_cfg: dict | None, seed: int = 0, table_z_bias: float = 0.0
) -> dict:
    """Sample fixed environment parameters per the generation config.

    Parameters
    ----------
    generation_cfg : the scene_config `generation` section (None/empty → default
                     clean environment). Recognized field: `random_table_height`.
    seed           : scene-level seed, for reproducibility across episodes.

    Returns an environment dict (storable into scene_spec.environment).
    """
    cfg = generation_cfg or {}
    rng = np.random.RandomState(seed)

    random_table_height = float(cfg.get("random_table_height", 0) or 0)
    # Matches _base_task: table_z_bias = uniform(-random_table_height, 0).
    table_z_bias += float(rng.uniform(low=-random_table_height, high=0))

    return {"table_z_bias": table_z_bias}


def apply_initial_physics(actors_by_oid, environment):
    """Apply frozen physical state to every task sharing a scene."""
    for oid, settings in (environment or {}).get("initial_physics", {}).items():
        actor = actors_by_oid[oid]
        if "joint_fraction" in settings:
            limits = actor.get_qlimits()
            qpos = actor.get_qpos()
            qpos[0] = (
                limits[0, 0]
                + (limits[0, 1] - limits[0, 0]) * settings["joint_fraction"]
            )
            actor.set_qpos(qpos)
        if "orientation" in settings or "z" in settings:
            pose = actor.get_pose()
            if "orientation" in settings:
                pose.q = settings["orientation"]
            if "z" in settings:
                pose.p = [*pose.p[:2], settings["z"]]
            actor.actor.set_pose(pose)
        if "kinematic" in settings:
            actor.set_kinematic(settings["kinematic"])
        if "mass" in settings:
            actor.set_mass(settings["mass"])
        if "properties" in settings:
            actor.set_properties(*settings["properties"])
