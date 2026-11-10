"""Frozen object profiles and layouts for the seven-task OOD benchmark."""

from copy import deepcopy

OBJECTS = {
    "playingcards": dict(
        model="081_playingcards",
        ids=[0, 1, 2],
        source="place_object_basket",
        q=[0.707225, 0.706849, -0.0100455, -0.00982061],
        mass=0.01,
        grasp={},
        x=[-0.25, -0.20],
        y=[-0.1, 0.1],
    ),
    "cup": dict(
        model="021_cup",
        ids=[0],
        source="place_empty_cup",
        q=[0.5] * 4,
        grasp=dict(pre_grasp_dis=0.1),
        x=[-0.3, -0.15],
        y=[-0.2, 0.05],
    ),
    "mouse": dict(
        model="047_mouse",
        ids=[0, 1, 2],
        source="place_mouse_pad",
        q=[0, 0, 0.70710678, 0.70710678],
        mass=0.05,
        grasp=dict(pre_grasp_dis=0.1),
        x=[-0.25, -0.05],
        y=[-0.2, 0],
    ),
    "block": dict(
        model="sapien-block",
        ids=[0],
        source="stack_blocks_two",
        q=[1, 0, 0, 0],
        grasp=dict(pre_grasp_dis=0.09),
        box_spec=dict(half_size=[0.025] * 3, color=[1, 0, 0]),
        x=[-0.25, -0.10],
        y=[-0.15, 0.05],
    ),
    "toycar": dict(
        model="057_toycar",
        ids=list(range(6)),
        source="place_object_basket",
        q=[0.707225, 0.706849, -0.0100455, -0.00982061],
        mass=0.01,
        grasp=dict(pre_grasp_dis=0.1),
        x=[-0.25, -0.20],
        y=[-0.1, 0.1],
    ),
}

TARGETS = {
    "pad": dict(
        model="sapien-block",
        model_id=0,
        q=[1, 0, 0, 0],
        source="place_mouse_pad",
        box_spec=dict(half_size=[0.035, 0.065, 0.0005], color=[0, 0, 1]),
        static=True,
    ),
    "basket": dict(
        model="110_basket",
        model_id=0,
        q=[0.5] * 4,
        source="place_object_basket",
        mass=0.5,
    ),
    "bowl": dict(model="002_bowl", model_id=3, q=[0.5] * 4, source="stack_bowls_two"),
    "plate": dict(
        model="003_plate",
        model_id=0,
        q=[0.5] * 4,
        source="place_container_plate",
        mass=0.05,
    ),
    "plasticbox": dict(
        model="062_plasticbox",
        model_id=3,
        q=[0.5] * 4,
        source="place_cans_plasticbox",
        mass=0.05,
    ),
}

TASKS = {
    "place_toycar_plasticbox": {
        "action": "place",
        "object": "toycar",
        "target": "plasticbox",
    },
    "place_cup_basket": {"action": "place", "object": "cup", "target": "basket"},
    "touch_toycar": {"action": "touch", "object": "toycar"},
    "place_playingcards_plate": {
        "action": "place",
        "object": "playingcards",
        "target": "plate",
    },
    "place_block_pad": {"action": "place", "object": "block", "target": "pad"},
    "place_mouse_bowl": {"action": "place", "object": "mouse", "target": "bowl"},
}


def layout(name):
    task = TASKS[name]
    obj = OBJECTS[task["object"]]
    spec = dict(
        xlim=obj["x"], ylim=obj["y"], qpos=obj["q"], model_id=obj["ids"][0], convex=True
    )
    spec.update(deepcopy(obj.get("placement", {})))
    if "box_spec" in obj:
        spec["box_spec"] = deepcopy(obj["box_spec"])
    result = [(obj["model"], spec)]
    if task["action"] == "place":
        target = TARGETS[task["target"]]
        region = dict(
            xlim=[-0.04, 0] if obj["x"][0] < 0 else [0, 0.04],
            ylim=[-0.12, -0.08],
            qpos=target["q"],
            model_id=target["model_id"],
            convex=True,
        )
        if "box_spec" in target:
            region["box_spec"] = deepcopy(target["box_spec"])
            region["ylim"] = [-0.18, -0.02]
        result.append((target["model"], region))
    return result
