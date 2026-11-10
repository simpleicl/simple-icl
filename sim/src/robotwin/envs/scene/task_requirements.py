"""Task requirement registry, built from task_metadata.json plus curated rules.

Covers the built-in tasks plus scene-only extensions. Sources:
  - actor_list in task_metadata.json (auto-parsed modelnames)
  - inline object pools for generalized tasks (curated from envs/*.py)
  - special handling for primitive tasks (blocks) and bowls

Generalized tasks expand a category pool (e.g. GRASPABLE_SMALL = the place_a2b
pool). Articulated tasks are marked kind="articulated". Blocks use the virtual
modelname "sapien-block"; bowls use "002_bowl".
"""

from __future__ import annotations

import json
from pathlib import Path


def get_all_tasks():
    return json.loads(Path(__file__).with_name("task_metadata.json").read_text())


from .roles import RoleSpec, TaskRequirement
from .ood_catalog import (
    TASKS as OOD_TASK_PROFILES,
    OBJECTS as OOD_OBJECTS,
    TARGETS as OOD_TARGETS,
    layout as ood_layout,
)

# ---------------------------------------------------------------- object pools

# Object pools for generalized tasks (extracted from envs/*.py).
GRASPABLE_SMALL = frozenset(
    [
        "047_mouse",
        "048_stapler",
        "050_bell",
        "057_toycar",
        "073_rubikscube",
        "075_bread",
        "077_phone",
        "081_playingcards",
        "086_woodenblock",
        "112_tea-box",
        "113_coffee-box",
        "107_soap",
    ]
)  # place_a2b_left/right pool


SHOE_OBJECTS = frozenset(set(GRASPABLE_SMALL) | {"021_cup"})

SCALE_OBJECTS = frozenset(
    ["047_mouse", "048_stapler", "050_bell"]
)  # place_object_scale

STAND_OBJECTS = frozenset(
    [
        "047_mouse",
        "048_stapler",
        "050_bell",
        "073_rubikscube",
        "057_toycar",
        "079_remotecontrol",
    ]
)  # place_object_stand


BASKET_OBJECTS = frozenset(
    [
        "081_playingcards",
        "057_toycar",
        # Paired scenes also accept cans and bottles; legacy load_actors uses the first two models.
        "071_can",
        "001_bottle",
    ]
)  # place_object_basket

CONTAINER_TYPES = frozenset(["002_bowl", "021_cup"])  # place_container_plate

# Fixed containers / tools
BASKET = "110_basket"
CAN = "071_can"
ELECTRONIC_SCALE = "072_electronicscale"
# place_object_stand creates 074_displaystand (067 has no asset dir); accept must
# match the actual scene object or the static matcher won't bind it.
DISPLAY_STAND = "074_displaystand"

OOD_EVAL_TASKS = frozenset({"close_laptop", *OOD_TASK_PROFILES})

TASK_TABLE_Z_BASELINES = {
    "place_dual_shoes": -0.1,
}


def task_table_z_baseline(task_name: str) -> float:
    return float(TASK_TABLE_Z_BASELINES.get(task_name, 0.0))


# Articulated URDF objects. 044_microwave = open_microwave's URDF;
# 060_kitchenpot = the lift_pot / move_can_pot pot.
ARTICULATED = frozenset(
    [
        "015_laptop",
        "016_oven",
        "036_cabinet",
        "009_kettle",
        "056_switch",
        "044_microwave",
        "060_kitchenpot",
    ]
)

# Primitive block: virtual modelname for create_box blocks (no asset dir).
PRIMITIVE_BLOCK = "sapien-block"

# Per-instance box specs for primitive block tasks (half_size / color / optional
# qpos). Blocks are create_box props whose geometry and colour are task
# semantics. List order == scene instance order == match_task binding order, so
# bind_actors gets the right block by index (ranking large→left, rgb red→left;
# handover box grasped / target placed).
_CUBE = 0.025  # standard cube half-extent (stack / ranking_rgb)
PRIMITIVE_BLOCK_SPECS = {
    "stack_blocks_two": [
        {"half_size": [_CUBE, _CUBE, _CUBE], "color": [1, 0, 0]},  # block1 red (bottom)
        {"half_size": [_CUBE, _CUBE, _CUBE], "color": [0, 1, 0]},  # block2 green
    ],
    "stack_blocks_three": [
        {"half_size": [_CUBE, _CUBE, _CUBE], "color": [1, 0, 0]},  # block1 red (bottom)
        {"half_size": [_CUBE, _CUBE, _CUBE], "color": [0, 1, 0]},  # block2 green
        {"half_size": [_CUBE, _CUBE, _CUBE], "color": [0, 0, 1]},  # block3 blue
    ],
    "blocks_ranking_rgb": [
        {"half_size": [_CUBE, _CUBE, _CUBE], "color": [1, 0, 0]},  # block1 red → left
        {
            "half_size": [_CUBE, _CUBE, _CUBE],
            "color": [0, 1, 0],
        },  # block2 green → middle
        {"half_size": [_CUBE, _CUBE, _CUBE], "color": [0, 0, 1]},  # block3 blue → right
    ],
    "blocks_ranking_size": [  # fixed sizes (large/medium/small) for the ranking order
        {
            "half_size": [0.031, 0.031, 0.031],
            "color": [0.8, 0.2, 0.2],
        },  # block1 large → left
        {"half_size": [0.025, 0.025, 0.025], "color": [0.2, 0.8, 0.2]},  # block2 medium
        {
            "half_size": [0.019, 0.019, 0.019],
            "color": [0.2, 0.2, 0.8],
        },  # block3 small → right
    ],
}


def infer_kind(modelname: str) -> str:
    """Infer object kind (rigid / articulated / primitive_box) for the placer."""
    if modelname in ARTICULATED:
        return "articulated"
    if modelname.startswith(
        "sapien-block"
    ):  # matches sapien-block / sapien-block1 / ...
        return "primitive_box"
    return "rigid"


# ---------------------------------------------------------------- registry


def _parse_task_info_roles(actor_list: dict | list) -> list[RoleSpec]:
    """Parse roles from a task_info actor_list."""
    if isinstance(actor_list, list):
        # Plain list form with no modelnames; handled by an override.
        return []

    roles: list[RoleSpec] = []
    for role_name, spec in actor_list.items():
        if not isinstance(spec, dict):
            continue
        # role_name is like "self.hammer"; drop the "self." prefix.
        name = role_name.replace("self.", "")
        modelname = spec.get("modelname")
        if modelname is None:
            # No modelname (e.g. target_pose/middle_pose) → not an object role.
            continue

        # Normalize sapien-block* variants → PRIMITIVE_BLOCK.
        if isinstance(modelname, str) and modelname.startswith("sapien-block"):
            modelname = PRIMITIVE_BLOCK

        accept = frozenset([modelname])
        kind = infer_kind(modelname)
        graspable = kind != "primitive_box"  # blocks are usually targets, not grasped

        roles.append(
            RoleSpec(name=name, accept=accept, count=1, kind=kind, graspable=graspable)
        )

    return roles


def _build_registry() -> dict[str, TaskRequirement]:
    """Build the task requirement registry."""
    registry: dict[str, TaskRequirement] = {}
    all_task_info = get_all_tasks()

    # Auto-parse task_info first.
    for key, task_dict in all_task_info.items():
        if key == "task_name":  # skip the global template
            continue
        task_name = task_dict.get("task_name")
        if not task_name or not isinstance(task_name, str):
            continue
        actor_list = task_dict.get("actor_list", [])
        roles = _parse_task_info_roles(actor_list)

        if roles:
            registry[task_name] = TaskRequirement(
                task_name=task_name, roles=tuple(roles)
            )

    # ------------------------------------------------------ manual overrides
    # Generalized tasks (object pools).
    registry["place_a2b_left"] = TaskRequirement(
        task_name="place_a2b_left",
        roles=(
            RoleSpec(name="object", accept=GRASPABLE_SMALL, count=1, kind="rigid"),
            RoleSpec(
                name="target_object", accept=GRASPABLE_SMALL, count=1, kind="rigid"
            ),
        ),
        arms="any",
        source="override",
    )
    registry["place_a2b_right"] = TaskRequirement(
        task_name="place_a2b_right",
        roles=(
            RoleSpec(name="object", accept=GRASPABLE_SMALL, count=1, kind="rigid"),
            RoleSpec(
                name="target_object", accept=GRASPABLE_SMALL, count=1, kind="rigid"
            ),
        ),
        arms="any",
        source="override",
    )

    registry["place_object_scale"] = TaskRequirement(
        task_name="place_object_scale",
        roles=(
            RoleSpec(name="object", accept=SCALE_OBJECTS, count=1, kind="rigid"),
            RoleSpec(
                name="scale",
                accept=frozenset([ELECTRONIC_SCALE]),
                count=1,
                kind="rigid",
                graspable=False,
            ),
        ),
        source="override",
    )

    registry["place_object_stand"] = TaskRequirement(
        task_name="place_object_stand",
        roles=(
            RoleSpec(name="object", accept=STAND_OBJECTS, count=1, kind="rigid"),
            RoleSpec(
                name="displaystand",
                accept=frozenset([DISPLAY_STAND]),
                count=1,
                kind="rigid",
                graspable=False,
            ),
        ),
        source="override",
    )

    registry["place_object_on_shoe"] = TaskRequirement(
        task_name="place_object_on_shoe",
        roles=(
            RoleSpec(name="object", accept=SHOE_OBJECTS, count=1, kind="rigid"),
            RoleSpec(
                name="shoe",
                accept=frozenset(["041_shoe"]),
                count=1,
                kind="rigid",
                graspable=False,
            ),
        ),
        arms="single",
        source="override",
    )

    for task_name in (
        "place_object_upper_left",
        "place_object_upper_right",
        "place_object_lower_left",
        "place_object_lower_right",
    ):
        registry[task_name] = TaskRequirement(
            task_name=task_name,
            roles=(
                RoleSpec(name="object", accept=GRASPABLE_SMALL, count=1, kind="rigid"),
            ),
            arms="single",
            source="override",
        )

    registry["close_laptop"] = TaskRequirement(
        task_name="close_laptop",
        roles=(
            RoleSpec(
                name="laptop",
                accept=frozenset(["015_laptop"]),
                kind="articulated",
                graspable=False,
            ),
        ),
        arms="single",
        source="override",
    )

    registry["place_object_basket"] = TaskRequirement(
        task_name="place_object_basket",
        roles=(
            RoleSpec(name="object", accept=BASKET_OBJECTS, count=1, kind="rigid"),
            RoleSpec(
                name="basket",
                accept=frozenset([BASKET]),
                count=1,
                kind="rigid",
                graspable=False,
            ),
        ),
        source="override",
    )

    registry["place_can_basket"] = TaskRequirement(
        task_name="place_can_basket",
        roles=(
            RoleSpec(name="can", accept=frozenset([CAN]), count=1, kind="rigid"),
            RoleSpec(
                name="basket",
                accept=frozenset([BASKET]),
                count=1,
                kind="rigid",
                graspable=False,
            ),
        ),
        source="override",
    )

    registry["place_container_plate"] = TaskRequirement(
        task_name="place_container_plate",
        roles=(
            RoleSpec(name="container", accept=CONTAINER_TYPES, count=1, kind="rigid"),
            RoleSpec(
                name="plate",
                accept=frozenset(["003_plate"]),
                count=1,
                kind="rigid",
                graspable=False,
            ),
        ),
        source="override",
    )

    # "Main object + local flat pad/target" tasks: the pad/target is a create_box
    # prop built in bind_actors (not a scene object), so only the main object is a
    # scene role. Overrides the auto-parsed sapien-block role so the task
    # classifies as concrete, not primitive.
    registry["move_pillbottle_pad"] = TaskRequirement(
        task_name="move_pillbottle_pad",
        roles=(
            RoleSpec(
                name="pillbottle",
                accept=frozenset(["080_pillbottle"]),
                count=1,
                kind="rigid",
            ),
        ),
        source="override",
    )
    registry["place_shoe"] = TaskRequirement(
        task_name="place_shoe",
        roles=(
            RoleSpec(
                name="shoe", accept=frozenset(["041_shoe"]), count=1, kind="rigid"
            ),
        ),
        source="override",
    )
    # beat_block_hammer: hammer is the scene object (fixed pose); block is a local
    # random target built in bind_actors. Only hammer is a scene role.
    registry["beat_block_hammer"] = TaskRequirement(
        task_name="beat_block_hammer",
        roles=(
            RoleSpec(
                name="hammer", accept=frozenset(["020_hammer"]), count=1, kind="rigid"
            ),
        ),
        source="override",
    )
    # place_dual_shoes: two shoes (side regions) are scene objects; shoe_box is a
    # fixed anchor built in bind_actors.
    registry["place_dual_shoes"] = TaskRequirement(
        task_name="place_dual_shoes",
        roles=(
            RoleSpec(
                name="left_shoe", accept=frozenset(["041_shoe"]), count=1, kind="rigid"
            ),
            RoleSpec(
                name="right_shoe", accept=frozenset(["041_shoe"]), count=1, kind="rigid"
            ),
        ),
        arms="dual",
        source="override",
    )
    # dump_bin_bigbin: the deskbin is 063_tabletrashbin (task_info mislabels it
    # 005_desk-bin); it is the scene object. dustbin anchor + 5 garbage spheres
    # are built in bind_actors.
    registry["dump_bin_bigbin"] = TaskRequirement(
        task_name="dump_bin_bigbin",
        roles=(
            RoleSpec(
                name="deskbin",
                accept=frozenset(["063_tabletrashbin"]),
                count=1,
                kind="rigid",
            ),
        ),
        source="override",
    )

    # Primitive blocks.
    registry["stack_blocks_two"] = TaskRequirement(
        task_name="stack_blocks_two",
        roles=(
            RoleSpec(
                name="block",
                accept=frozenset([PRIMITIVE_BLOCK]),
                count=2,
                kind="primitive_box",
            ),
        ),
        source="override",
    )
    registry["stack_blocks_three"] = TaskRequirement(
        task_name="stack_blocks_three",
        roles=(
            RoleSpec(
                name="block",
                accept=frozenset([PRIMITIVE_BLOCK]),
                count=3,
                kind="primitive_box",
            ),
        ),
        source="override",
    )
    registry["blocks_ranking_rgb"] = TaskRequirement(
        task_name="blocks_ranking_rgb",
        roles=(
            RoleSpec(
                name="block",
                accept=frozenset([PRIMITIVE_BLOCK]),
                count=3,
                kind="primitive_box",
            ),
        ),
        source="override",
    )
    registry["blocks_ranking_size"] = TaskRequirement(
        task_name="blocks_ranking_size",
        roles=(
            RoleSpec(
                name="block",
                accept=frozenset([PRIMITIVE_BLOCK]),
                count=3,
                kind="primitive_box",
            ),
        ),
        source="override",
    )

    # Bowls (002_bowl).
    registry["stack_bowls_two"] = TaskRequirement(
        task_name="stack_bowls_two",
        roles=(
            RoleSpec(
                name="bowl", accept=frozenset(["002_bowl"]), count=2, kind="rigid"
            ),
        ),
        source="override",
    )
    registry["stack_bowls_three"] = TaskRequirement(
        task_name="stack_bowls_three",
        roles=(
            RoleSpec(
                name="bowl", accept=frozenset(["002_bowl"]), count=3, kind="rigid"
            ),
        ),
        source="override",
    )

    # Additional evaluation objects use the existing role geometry and success criteria.

    for name, spec in OOD_TASK_PROFILES.items():
        models = [("object", OOD_OBJECTS[spec["object"]]["model"])]
        if spec["action"] == "place":
            models.append(("target", OOD_TARGETS[spec["target"]]["model"]))
        registry[name] = TaskRequirement(
            task_name=name,
            arms="single",
            source="override",
            roles=tuple(
                RoleSpec(
                    name=role,
                    accept=frozenset([model]),
                    requires_binding=True,
                    graspable=role == "object" and spec["action"] == "place",
                    kind="primitive_box" if model == PRIMITIVE_BLOCK else "rigid",
                )
                for role, model in models
            ),
        )
    # Task lists select tasks at the CLI boundary; the registry contains all
    # definitions and must remain independent of generation/evaluation splits.
    return registry


REGISTRY = _build_registry()


def get_requirement(task_name: str) -> TaskRequirement | None:
    return REGISTRY.get(task_name)


def all_task_names() -> list[str]:
    return sorted(REGISTRY.keys())


def build_primitive_scene(task_name: str):
    """Build a scene object set for a primitive-block task from PRIMITIVE_BLOCK_SPECS.

    One sapien-block per spec (carrying box_spec). Instance order matches the
    greedy binding order, so bind_actors gets the right block by index. Primitive
    blocks have no asset, so this bypasses from_modelnames / scene_grouping.
    """
    from .roles import SceneObjectSet

    specs = PRIMITIVE_BLOCK_SPECS.get(task_name)
    if specs is None:
        raise ValueError(f"no primitive block specs registered for task {task_name!r}")
    s = SceneObjectSet()
    for spec in specs:
        box_spec = dict(spec)
        region = box_spec.pop("place_region", None)
        role = box_spec.pop("binding_role", None)
        s.add(
            PRIMITIVE_BLOCK,
            kind="primitive_box",
            box_spec=box_spec,
            place_region=region,
            binding_task=task_name if role else None,
            binding_role=role,
        )
    return s


# ---------------------------------------------------------------- structured layouts
# "Central container + side objects" tasks: the container is pinned to the centre
# and side objects to left/right bands (random placement would offset the
# container out of one arm's reach). Each instance's place_region overrides the
# default region. List order == instance order == match_task binding order.
_STRUCTURED_LAYOUT = {
    "place_cans_plasticbox": [
        (
            "062_plasticbox",
            {"xlim": [0.0, 0.0], "ylim": [-0.15, -0.1], "qpos": [0.5, 0.5, 0.5, 0.5]},
        ),  # box centred
        (
            "071_can",
            {
                "xlim": [-0.25, -0.15],
                "ylim": [-0.15, -0.07],
                "qpos": [0.5, 0.5, 0.5, 0.5],
            },
        ),  # left can
        (
            "071_can",
            {
                "xlim": [0.15, 0.25],
                "ylim": [-0.15, -0.07],
                "qpos": [0.5, 0.5, 0.5, 0.5],
            },
        ),  # right can
    ],
    "place_burger_fries": [
        (
            "008_tray",
            {
                "xlim": [0.0, 0.0],
                "ylim": [-0.15, -0.1],
                "qpos": [0.706527, 0.706483, -0.0291356, -0.0291767],
                "is_static": True,
            },
        ),  # tray centred (static)
        (
            "006_hamburg",
            {
                "xlim": [-0.3, -0.25],
                "ylim": [-0.15, -0.07],
                "qpos": [0.5, 0.5, 0.5, 0.5],
            },
        ),  # left hamburger
        (
            "005_french-fries",
            {"xlim": [0.2, 0.3], "ylim": [-0.15, -0.07], "qpos": [1.0, 0.0, 0.0, 0.0]},
        ),  # right fries
    ],
    "beat_block_hammer": [
        # hammer at a fixed pose: place_region → rand_create_actor bypasses the
        # cluttered table (the asset can still be created). block is a local
        # random target built in bind_actors, so only hammer is here.
        (
            "020_hammer",
            {
                "xlim": [0.0, 0.0],
                "ylim": [-0.06, -0.06],
                "zlim": [0.783],
                "qpos": [0, 0, 0.995, 0.105],
                "model_id": 0,
            },
        ),
    ],
    "place_dual_shoes": [
        # two shoes pinned to left/right bands; shoe_box anchor built in bind_actors.
        # zlim follows table_z_bias (which includes -0.1) to sit on the table.
        (
            "041_shoe",
            {
                "xlim": [-0.3, -0.2],
                "ylim": [-0.1, 0.05],
                "zlim": [0.741],
                "qpos": [0.707, 0.707, 0, 0],
            },
        ),
        (
            "041_shoe",
            {
                "xlim": [0.2, 0.3],
                "ylim": [-0.1, 0.05],
                "zlim": [0.741],
                "qpos": [0.707, 0.707, 0, 0],
            },
        ),
    ],
    "dump_bin_bigbin": [
        # table_xy_bias=[0.3,0] but legacy deskbin uses rand_pose (no bias) →
        # absolute. deskbin (063_tabletrashbin) is the scene object (random, right
        # half, avoiding |x|<0.05); dustbin anchor + 5 garbage spheres built in bind_actors.
        (
            "063_tabletrashbin",
            {
                "xlim": [0.08, 0.2],
                "ylim": [-0.2, -0.05],
                "qpos": [0.651892, 0.651428, 0.274378, 0.274584],
                "absolute": True,
                "model_ids": [0, 3, 7, 8, 9, 10],
            },
        ),
    ],
}


_STRUCTURED_LAYOUT.update({name: ood_layout(name) for name in OOD_TASK_PROFILES})


def build_structured_scene(task_name: str):
    """Build a scene object set with place_region from _STRUCTURED_LAYOUT.

    One object per layout entry (carrying place_region). Instance order matches
    the greedy binding order, so container / left / right objects land in place.
    Same-name side objects (e.g. two cans) are separated by place_region.
    """
    from .roles import SceneObjectSet

    layout = _STRUCTURED_LAYOUT.get(task_name)
    if layout is None:
        raise ValueError(f"no structured layout registered for task {task_name!r}")
    s = SceneObjectSet()
    req = REGISTRY[task_name]
    role_by_model = {}
    for role in req.roles:
        for modelname in role.accept:
            role_by_model.setdefault(modelname, []).extend([role.name] * role.count)
    role_cursor = {modelname: 0 for modelname in role_by_model}

    for modelname, spec in layout:
        model_id = spec.get("model_id")
        region = {k: v for k, v in spec.items() if k not in {"model_id", "box_spec"}}
        roles = role_by_model.get(modelname, [])
        role_idx = role_cursor.get(modelname, 0)
        role_name = roles[role_idx] if role_idx < len(roles) else None
        role_cursor[modelname] = role_idx + 1
        s.add(
            modelname,
            kind=infer_kind(modelname),
            model_id=model_id,
            place_region=region,
            box_spec=spec.get("box_spec"),
            binding_task=task_name,
            binding_role=role_name,
        )
    return s
