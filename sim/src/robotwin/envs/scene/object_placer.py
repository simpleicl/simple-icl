"""Scene-mode object placer — lays a SceneObjectSet's named objects onto the table.

Reuses the collision-aware cluttered engine (`rand_create_cluttered_actor` /
`rand_pose_cluttered` in `envs/utils/rand_create_cluttered_actor.py`) for
prohibited-area avoidance, per-object separation (size_dict), table bounds and
upright glb poses — but places the scene's named objects instead of random
distractors. Only invoked in scene mode by `_base_task`.
"""

from __future__ import annotations

import math

import numpy as np
import sapien
import transforms3d as t3d

from ..utils.rand_create_cluttered_actor import (
    check_overlap,
    cluttered_objects_info,
    rand_create_cluttered_actor,
)
from ..utils.create_actor import create_box, create_actor, create_sapien_urdf_obj
from .roles import SceneObjectSet, ObjectInstance


class PlacementError(RuntimeError):
    """Placement failed after retries; caller retries with a new seed."""


def snapshot_np_rng_state() -> dict:
    """Return JSON-safe RNG state for task-local random setup."""
    name, keys, pos, has_gauss, cached_gaussian = np.random.get_state()
    return {
        "name": name,
        "keys": [int(key) for key in keys],
        "pos": int(pos),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached_gaussian),
    }


def restore_np_rng_state(state: dict) -> None:
    """Restore a state saved by `snapshot_np_rng_state`."""
    np.random.set_state(
        (
            state["name"],
            np.asarray(state["keys"], dtype=np.uint32),
            int(state["pos"]),
            int(state["has_gauss"]),
            float(state["cached_gaussian"]),
        )
    )


# Default half-extent (metres) for a primitive_box without an explicit spec.
_DEFAULT_BOX_HALF = [0.02, 0.02, 0.02]

# Extra pad (metres) on a primitive block's avoidance radius, so neighbours stay
# far enough apart that the gripper doesn't hit an adjacent block.
_BLOCK_AVOID_PAD = 0.02

# Dense mixed scenes occasionally leave only a small valid region for the last
# object. Each attempt samples up to 100 poses in the shared cluttered helper.
_PLACEMENT_ATTEMPTS = 3
_LAYOUT_ATTEMPTS = 3

_MODEL_ID_UNSET = object()

# Pinned base qpos for orientation-sensitive objects. The cluttered engine's
# default upright pose suits symmetric objects; grasp/place tasks whose contact
# or functional point depends on orientation (bowl mouth up, top buttons,
# container openings) need a fixed base qpos. Unlisted objects keep the default
# upright pose + random yaw.
# TODO: derive from model_data orientation_point/transform_matrix to drop this table.
_PREFERRED_QPOS: dict[str, list] = {
    "002_bowl": [0.5, 0.5, 0.5, 0.5],  # bowl mouth up (for stacking)
    "048_stapler": [0.5, 0.5, 0.5, 0.5],  # press_stapler / move_stapler_pad
    "046_alarm-clock": [0.5, 0.5, 0.5, 0.5],  # click_alarmclock
    "050_bell": [0.5, 0.5, 0.5, 0.5],  # click_bell
    "018_microphone": [0.707, 0.707, 0, 0],  # handover_mic (fixed, no rotation)
    "040_rack": [-0.22, -0.22, 0.67, 0.67],  # hanging_mug rack (hook orientation)
    "021_cup": [0.5, 0.5, 0.5, 0.5],  # place_empty_cup / place_container_plate
    "019_coaster": [0.5, 0.5, 0.5, 0.5],  # place_empty_cup (static)
    "110_basket": [0.5, 0.5, 0.5, 0.5],  # place_can_basket (opening up)
    "106_skillet": [0, 0, 0.707, 0.707],  # place_bread_skillet
    "062_plasticbox": [0.5, 0.5, 0.5, 0.5],  # place_cans_plasticbox (opening up)
    "024_scanner": [0, 0, 0.707, 0.707],  # scan_object
    "112_tea-box": [0.5, 0.5, 0.5, 0.5],  # scan_object (scanned object)
    "076_breadbasket": [0.5, 0.5, 0.5, 0.5],  # place_bread_basket (opening up)
    "078_phonestand": [0.707, 0.707, 0, 0],  # place_phone_stand (static)
    "003_plate": [0.5, 0.5, 0.5, 0.5],  # place_container_plate (static)
    "072_electronicscale": [
        0.5,
        0.5,
        0.5,
        0.5,
    ],  # place_object_scale (functional point on top)
    "074_displaystand": [0.707, 0.707, 0, 0],  # place_object_stand
    "041_shoe": [0.707, 0.707, 0, 0],  # place_object_on_shoe
}

# Random rotation layered on the pinned base qpos. A few objects (stapler, alarm
# clock) need rotation about a specific local axis for a top-down
# graspable/pressable pose; unlisted objects stay fixed.
_PREFERRED_ROTATE_LIM: dict[str, list] = {
    "048_stapler": [0, math.pi, 0],
    "046_alarm-clock": [0, math.pi, 0],
}


# Placement spec for articulated (URDF) objects, which are created via
# rand_create_sapien_urdf_obj rather than the cluttered engine. Base qpos,
# placement region (usually behind the table for reach-in), rotation, candidate
# model_ids, avoidance radius. Joint state / mass / physics are task semantics,
# set in each task's bind_actors.
_URDF_PLACE_SPEC: dict[str, dict] = {
    # Mixed-scene fallback; lift_pot / move_can_pot override it below.
    "060_kitchenpot": {
        "qpos": [0.704141, 0, 0, 0.71006],
        "rotate_lim": [0, 0, math.pi / 8],
        "rotate_rand": True,
        "xlim": [-0.2, 0.2],
        "ylim": [0.05, 0.15],
        "zlim": [0.741],
        "model_ids": list(range(2)),
        "radius": 0.12,
        "fix_root_link": False,
    },
    "015_laptop": {
        "qpos": [0.7, 0, 0, 0.7],
        "rotate_lim": [0, 0, math.pi / 3],
        "rotate_rand": True,
        "xlim": [-0.05, 0.05],
        "ylim": [-0.1, 0.05],
        "zlim": [0.741],
        "model_ids": list(range(11)),
        "radius": 0.13,
    },
    "056_switch": {
        "qpos": [0.704141, 0, 0, 0.71006],
        "rotate_lim": [0, 0, math.pi / 4],
        "rotate_rand": True,
        "xlim": [-0.25, 0.25],
        "ylim": [0.0, 0.1],
        "zlim": [0.81, 0.84],
        "model_ids": list(range(8)),
        "radius": 0.1,
    },
    "044_microwave": {
        "qpos": [0.707, 0, 0, 0.707],
        "rotate_lim": [0, 0, 0],
        "rotate_rand": False,
        "xlim": [-0.12, -0.02],
        "ylim": [0.15, 0.2],
        "zlim": [0.8, 0.8],
        "model_ids": list(range(2)),
        "radius": 0.2,
    },
    "036_cabinet": {
        "qpos": [1, 0, 0, 1],
        "rotate_lim": [0, 0, math.pi / 16],
        "rotate_rand": True,
        "xlim": [-0.05, 0.05],
        "ylim": [0.155, 0.155],
        "zlim": [0.741],
        "model_ids": [46653],
        "radius": 0.2,
    },
}

# Per-task overrides for URDF objects shared across tasks with different
# placement (e.g. 060_kitchenpot in lift_pot / move_can_pot).
# `fix_root_link=False` = free articulation that can be lifted (the pot), vs the
# fixed-base laptop/switch/microwave/cabinet.
_URDF_PLACE_SPEC_BY_TASK: dict[str, dict] = {
    "close_laptop": {
        "modelname": "015_laptop",
        "qpos": [0.707, 0, 0, -0.707],
        "rotate_lim": [0, 0, 0],
        "rotate_rand": False,
        "xlim": [-0.06, -0.04],
        "ylim": [-0.02, 0.02],
        "zlim": [0.741],
        "model_ids": [10],
        "radius": 0.13,
    },
    "lift_pot": {
        "modelname": "060_kitchenpot",
        "qpos": [0.704141, 0, 0, 0.71006],
        "rotate_lim": [0, 0, math.pi / 8],
        "rotate_rand": True,
        "xlim": [-0.05, 0.05],
        "ylim": [-0.05, 0.05],
        "zlim": [0.741],
        "model_ids": list(range(2)),
        "radius": 0.12,
        "fix_root_link": False,
    },
    "move_can_pot": {
        "modelname": "060_kitchenpot",
        "qpos": [0, 0, 0, 1],
        "rotate_lim": [0, 0, math.pi / 8],
        "rotate_rand": True,
        "xlim": [0.0, 0.0],
        "ylim": [0.0, 0.0],
        "zlim": [0.741],
        "model_ids": list(range(7)),
        "radius": 0.12,
        "fix_root_link": False,
    },
    "place_pot_on_scale": {
        "modelname": "060_kitchenpot",
        "qpos": [0.704141, 0, 0, 0.71006],
        "rotate_lim": [0, 0, math.pi / 8],
        "rotate_rand": True,
        "xlim": [-0.02, 0.02],
        "ylim": [0.08, 0.12],
        "zlim": [0.741],
        "model_ids": [0],
        "radius": 0.12,
        "fix_root_link": False,
    },
}


class ObjectPlacer:
    """Places all objects of a `SceneObjectSet` onto a `task`'s table.

    Parameters
    ----------
    task : Base_Task instance. Uses its `scene`, `prohibited_area`,
           `table_xy_bias`, `table_z_bias`.
    """

    def __init__(self, task):
        self.task = task

    # ------------------------------------------------------------------ API
    def place_all(
        self,
        scene_set: SceneObjectSet,
        xlim=(-0.3, 0.3),
        ylim=(-0.2, 0.2),
        zlim=(0.74,),
    ) -> dict[str, "object"]:
        """Place every object; return {oid: Actor}.

        - Objects avoid each other via a local size_dict ([x, y, z, radius]) and
          avoid `task.prohibited_area` (usually empty here; prohibited areas are
          added after binding).
        - xlim/ylim offset by table_xy_bias, zlim by table_z_bias.
        - Default bounds are the dual-arm reachable workspace, not the wider
          distractor spread (±0.55/±0.3): that would place graspable objects out
          of reach and fail motion planning.
        - Raises PlacementError if any object exhausts its retries.
        """
        task = self.task
        xlim = [xlim[0] + task.table_xy_bias[0], xlim[1] + task.table_xy_bias[0]]
        ylim = [ylim[0] + task.table_xy_bias[1], ylim[1] + task.table_xy_bias[1]]
        zlim = list(np.array(zlim) + task.table_z_bias)

        # Fixed-base articulated objects and structured anchors define the layout;
        # place them before ordinary objects so all later placements avoid them.
        def placement_order(obj):
            if obj.kind == "articulated":
                return (0, 0.0)
            if obj.place_region is not None:
                return (1, 0.0)
            return (2, -self._radius_of(obj))

        ordered = sorted(scene_set.objects, key=placement_order)
        last_error = None
        for _ in range(_LAYOUT_ATTEMPTS):
            size_dict: list = []
            placed: dict[str, object] = {}
            try:
                for inst in ordered:
                    actor = self._place_one(inst, xlim, ylim, zlim, size_dict)
                    if actor is None:
                        raise PlacementError(
                            f"failed to place {inst.oid} ({inst.modelname}) after retries"
                        )
                    actor.set_name(inst.oid)
                    if getattr(actor, "model_id", None) is None:
                        setattr(actor, "model_id", inst.model_id)
                    setattr(actor, "oid", inst.oid)
                    placed[inst.oid] = actor

                    pose = actor.get_pose().p.tolist()
                    radius = self._radius_of(
                        inst, model_id=getattr(actor, "model_id", inst.model_id)
                    )
                    size_dict.append([pose[0], pose[1], pose[2], radius])
                return placed
            except PlacementError as e:
                last_error = e
                self._remove_placed(placed.values())
            except Exception:
                self._remove_placed(placed.values())
                raise
        raise last_error

    # ------------------------------------------------------------- dispatch
    def _place_one(self, inst: ObjectInstance, xlim, ylim, zlim, size_dict):
        # A place_region overrides the default region (container centred / objects
        # pinned to side bands). Rigid objects with a region use
        # _place_rigid_region (rand_pose), avoiding the cluttered engine's
        # y+radius<ylim[1] constraint on narrow bands.
        if inst.place_region is not None:
            r = inst.place_region
            # absolute=True: world-absolute coords, not offset by table_xy_bias
            # (for tasks with table_xy_bias != 0 whose play_once hardcodes
            # absolute coords, e.g. put_bottles_dustbin / dump_bin_bigbin).
            if r.get("absolute", False):
                rx = [r["xlim"][0], r["xlim"][1]]
                ry = [r["ylim"][0], r["ylim"][1]]
                rz = list(r["zlim"]) if "zlim" in r else zlim
            else:
                rx = [
                    r["xlim"][0] + self.task.table_xy_bias[0],
                    r["xlim"][1] + self.task.table_xy_bias[0],
                ]
                ry = [
                    r["ylim"][0] + self.task.table_xy_bias[1],
                    r["ylim"][1] + self.task.table_xy_bias[1],
                ]
                rz = (
                    list(np.array(r["zlim"]) + self.task.table_z_bias)
                    if "zlim" in r
                    else zlim
                )
            if inst.kind == "rigid":
                return self._place_rigid_region(
                    inst,
                    rx,
                    ry,
                    rz,
                    r.get("qpos"),
                    r.get("is_static", False),
                    r.get("convex", True),
                    size_dict,
                )
            xlim, ylim, zlim = rx, ry, rz
        if inst.kind == "rigid":
            return self._place_rigid(inst, xlim, ylim, zlim, size_dict)
        if inst.kind == "primitive_box":
            return self._place_box(inst, xlim, ylim, zlim, size_dict)
        if inst.kind == "articulated":
            return self._place_articulated(inst, xlim, ylim, zlim, size_dict)
        raise ValueError(f"unknown object kind: {inst.kind!r} for {inst.oid}")

    def _place_rigid_region(
        self, inst: ObjectInstance, xlim, ylim, zlim, qpos, is_static, convex, size_dict
    ):
        """Fixed/side-region placement for structured layouts.

        Uses rand_pose directly to support narrow side bands, while still
        respecting the radii of every object already placed in the scene.
        """
        from ..utils.rand_create_actor import rand_create_actor

        info = cluttered_objects_info.get(inst.modelname)
        model_id = inst.model_id
        if model_id is None:
            allowed_ids = (inst.place_region or {}).get("model_ids")
            if allowed_ids:
                model_id = int(np.random.choice(allowed_ids))
            elif info is not None:
                digit_ids = [int(i) for i in info["ids"] if str(i).isdigit()]
                model_id = int(np.random.choice(digit_ids)) if digit_ids else 0
            else:
                model_id = 0
        if qpos is None:
            qpos = _PREFERRED_QPOS.get(inst.modelname, [0.707107, 0.707107, 0, 0])
        from ..utils.rand_create_actor import rand_pose

        radius = self._radius_of(
            ObjectInstance(
                oid=inst.oid,
                modelname=inst.modelname,
                kind=inst.kind,
                model_id=model_id,
                box_spec=inst.box_spec,
                place_region=inst.place_region,
                binding_task=inst.binding_task,
                binding_role=inst.binding_role,
            )
        )
        pose = None
        for _ in range(100 * _PLACEMENT_ATTEMPTS):
            candidate = rand_pose(
                xlim=xlim, ylim=ylim, zlim=list(zlim), rotate_rand=False, qpos=qpos
            )
            padded_radius = radius + 0.005
            if any(
                check_overlap(padded_radius, candidate.p[0], candidate.p[1], area)
                for area in self.task.prohibited_area
            ):
                continue
            if all(
                np.hypot(old[0] - candidate.p[0], old[1] - candidate.p[1])
                > old[3] + padded_radius + 0.005
                for old in size_dict
            ):
                pose = candidate
                break
        if pose is None:
            return None
        actor = create_actor(
            scene=self.task.scene,
            pose=pose,
            modelname=inst.modelname,
            convex=convex,
            is_static=is_static,
            model_id=model_id,
        )
        if actor is not None:
            setattr(actor, "model_id", model_id)
        return actor

    def _place_rigid(self, inst: ObjectInstance, xlim, ylim, zlim, size_dict):
        info = cluttered_objects_info.get(inst.modelname)
        if info is None:
            raise PlacementError(
                f"{inst.modelname} not in object info table (no radius/z data)"
            )
        ids = info["ids"]
        model_id = inst.model_id
        if model_id is None:
            model_id = np.random.choice(ids)
        model_id = str(model_id)
        if model_id not in info["params"]:
            raise PlacementError(f"{inst.modelname} has no stable model_id {model_id}")
        params = info["params"][model_id]

        # Orientation-sensitive objects use their pinned qpos; others keep the
        # default upright pose + random yaw. Pinned objects default to no
        # rotation, but button-type objects need rotation about a specific axis
        # for a top-down graspable/pressable pose (_PREFERRED_ROTATE_LIM).
        pref_qpos = _PREFERRED_QPOS.get(inst.modelname)
        pref_rot = _PREFERRED_ROTATE_LIM.get(inst.modelname)
        if pref_qpos is None:
            rotate_rand, rotate_lim = True, [0, 0, math.pi]
        else:
            rotate_rand = pref_rot is not None
            rotate_lim = pref_rot if pref_rot is not None else [0, 0, math.pi]

        success, actor = False, None
        for _ in range(_PLACEMENT_ATTEMPTS):
            success, actor = rand_create_cluttered_actor(
                self.task.scene,
                xlim=xlim,
                ylim=ylim,
                zlim=zlim,
                modelname=inst.modelname,
                modelid=model_id,
                modeltype=info["type"],
                qpos=pref_qpos,
                rotate_rand=rotate_rand,
                rotate_lim=rotate_lim,
                size_dict=size_dict,
                obj_radius=params["radius"],
                z_offset=params["z_offset"],
                # z_max=0 disables rand_pose_cluttered's "no front-zone placement"
                # rule (y<radius and z_max>0.05 → forbidden), which keeps distractors
                # clear of the robot's near workspace. Scene objects are the task's
                # main objects (legacy places them in the y<0 front zone), so disable it.
                z_max=0,
                prohibited_area=self.task.prohibited_area,
            )
            if success and actor is not None:
                break
        if not success or actor is None:
            return None
        # rand_create_cluttered_actor picked the model_id; store it for bind.
        inst_model_id_int = int(model_id) if str(model_id).isdigit() else model_id
        setattr(actor, "model_id", inst_model_id_int)
        return actor

    def _place_box(self, inst: ObjectInstance, xlim, ylim, zlim, size_dict):
        from ..utils.rand_create_cluttered_actor import rand_pose_cluttered

        # Geometry / colour / orientation come from box_spec (task semantics);
        # fall back to a placeholder box when no spec given.
        spec = inst.box_spec or {}
        half = list(spec.get("half_size", _DEFAULT_BOX_HALF))
        qpos = list(spec.get("qpos", [1, 0, 0, 0]))
        boxtype = spec.get(
            "boxtype", "default"
        )  # "long" → 8 contact points (handover); "default" → 4
        color = spec.get("color")
        if color is None:
            color = [np.random.rand(), np.random.rand(), np.random.rand()]

        world_half_z = self._box_world_half_z(half, qpos)
        avoid_radius = self._box_avoid_radius(half, qpos)
        # Cubes take a small random yaw without changing meaning; non-cubes (e.g.
        # the handover long box) keep a fixed qpos for stable z.
        is_cube = half[0] == half[1] == half[2]
        success, pose = False, None
        for _ in range(_PLACEMENT_ATTEMPTS):
            success, pose = rand_pose_cluttered(
                xlim=np.array(xlim),
                ylim=np.array(ylim),
                zlim=np.array(zlim),
                rotate_rand=is_cube,
                rotate_lim=[0, 0, math.pi] if is_cube else [0, 0, 0],
                qpos=qpos,
                size_dict=size_dict,
                obj_radius=avoid_radius,
                z_offset=-world_half_z,  # sit the box base on the table (qpos-aware half height)
                z_max=0,  # disable the front-zone rule (blocks are the task objects)
                prohibited_area=self.task.prohibited_area,
            )
            if success and pose is not None:
                break
        if not success or pose is None:
            return None
        actor = create_box(
            self.task.scene,
            pose,
            half_size=half,
            color=tuple(color),
            name=inst.oid,
            boxtype=boxtype,
        )
        return actor

    def _place_articulated(self, inst: ObjectInstance, xlim, ylim, zlim, size_dict):
        from ..utils.rand_create_actor import rand_pose

        # Prefer a per-task override (shared model, different placement, e.g.
        # pot); otherwise the per-model spec.
        spec = _URDF_PLACE_SPEC_BY_TASK.get(getattr(self.task, "task_name", None))
        if spec is None or spec.get("modelname") != inst.modelname:
            spec = _URDF_PLACE_SPEC.get(inst.modelname)
        if spec is None:
            raise PlacementError(
                f"{inst.modelname} has no URDF placement spec (_URDF_PLACE_SPEC)"
            )
        if inst.place_region and not getattr(
            self.task, "scene_legacy_articulated_placement", False
        ):
            spec = {**spec, **inst.place_region}
            if "qpos" in inst.place_region and "rotate_rand" not in inst.place_region:
                spec["rotate_rand"] = False
        model_id = inst.model_id
        if model_id is None:
            model_id = int(np.random.choice(spec["model_ids"]))

        # Use rand_create_sapien_urdf_obj (rand_pose, no cluttered avoidance or
        # ±0.34 bound): large URDFs would always hit the cluttered bounds and
        # fail. The articulated object is the only/main object here, so no
        # cluttered avoidance is needed.
        ax = [
            spec["xlim"][0] + self.task.table_xy_bias[0],
            spec["xlim"][1] + self.task.table_xy_bias[0],
        ]
        ay = [
            spec["ylim"][0] + self.task.table_xy_bias[1],
            spec["ylim"][1] + self.task.table_xy_bias[1],
        ]
        az = list(np.array(spec.get("zlim", [0.741])) + self.task.table_z_bias)
        radius = spec.get("radius", 0.15)
        pose = None
        for _ in range(100 * _PLACEMENT_ATTEMPTS):
            candidate = rand_pose(
                xlim=ax,
                ylim=ay,
                zlim=az,
                rotate_rand=spec.get("rotate_rand", False),
                rotate_lim=spec.get("rotate_lim", [0, 0, 0]),
                qpos=spec["qpos"],
            )
            padded_radius = radius + 0.005
            if any(
                check_overlap(padded_radius, candidate.p[0], candidate.p[1], area)
                for area in self.task.prohibited_area
            ):
                continue
            if all(
                np.hypot(old[0] - candidate.p[0], old[1] - candidate.p[1])
                > old[3] + padded_radius + 0.005
                for old in size_dict
            ):
                pose = candidate
                break
        if pose is None:
            return None
        actor = create_sapien_urdf_obj(
            scene=self.task.scene,
            pose=pose,
            modelname=inst.modelname,
            modelid=model_id,
            fix_root_link=spec.get(
                "fix_root_link", True
            ),  # fixed base by default; pot liftable=False
        )
        if actor is None:
            return None
        setattr(actor, "model_id", model_id)
        return actor

    # -------------------------------------------------------------- helpers
    def _remove_placed(self, actors) -> None:
        for actor in reversed(list(actors)):
            raw = getattr(actor, "actor", actor)
            if isinstance(raw, sapien.physx.PhysxArticulation):
                self.task.scene.remove_articulation(raw)
            else:
                self.task.scene.remove_actor(raw)

    @staticmethod
    def _box_world_half_z(half, qpos) -> float:
        """World-frame vertical half-height (qpos-aware), used to sit the box base on the table."""
        R = t3d.quaternions.quat2mat(qpos)
        return float(
            abs(R[2, 0]) * half[0] + abs(R[2, 1]) * half[1] + abs(R[2, 2]) * half[2]
        )

    @staticmethod
    def _box_avoid_radius(half, qpos) -> float:
        """Avoidance radius: larger world-frame horizontal half-width + block separation pad."""
        R = t3d.quaternions.quat2mat(qpos)
        world_hx = (
            abs(R[0, 0]) * half[0] + abs(R[0, 1]) * half[1] + abs(R[0, 2]) * half[2]
        )
        world_hy = (
            abs(R[1, 0]) * half[0] + abs(R[1, 1]) * half[1] + abs(R[1, 2]) * half[2]
        )
        return float(max(world_hx, world_hy)) + _BLOCK_AVOID_PAD

    def _radius_of(self, inst: ObjectInstance, model_id=_MODEL_ID_UNSET) -> float:
        if inst.kind == "rigid" and "avoid_radius" in (inst.place_region or {}):
            radius = float(inst.place_region["avoid_radius"])
            if not np.isfinite(radius) or radius <= 0:
                raise ValueError(f"Invalid avoidance radius for {inst.oid}")
            return radius
        if inst.kind == "primitive_box":
            spec = inst.box_spec or {}
            half = list(spec.get("half_size", _DEFAULT_BOX_HALF))
            qpos = list(spec.get("qpos", [1, 0, 0, 0]))
            return self._box_avoid_radius(half, qpos)
        if inst.kind == "articulated":
            spec = _URDF_PLACE_SPEC.get(inst.modelname)
            return spec.get("radius", 0.15) if spec else 0.15
        info = cluttered_objects_info.get(inst.modelname)
        if info is None:
            return 0.05
        if model_id is _MODEL_ID_UNSET:
            model_id = inst.model_id
        if model_id is None:
            # Unspecified: use the model's largest radius for conservative avoidance.
            return max(p["radius"] for p in info["params"].values())
        if str(model_id) not in info["params"]:
            raise ValueError(
                f"No stable placement metadata for {inst.modelname}/base{model_id} "
                f"(oid={inst.oid}); available model_ids={sorted(info['params'])}"
            )
        return info["params"][str(model_id)]["radius"]
