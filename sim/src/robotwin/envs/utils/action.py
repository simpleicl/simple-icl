from typing import Literal
from .transforms import _tolist
import numpy as np
import sapien


class ArmTag:
    _instances = {}

    def __new__(cls, value):
        if isinstance(value, ArmTag):
            return value
        if isinstance(value, str) and value in ["left", "right"]:
            value = value.lower()
            if value in cls._instances:
                return cls._instances[value]
            instance = super().__new__(cls)
            cls._instances[value] = instance
            return instance
        raise ValueError(f"Invalid arm tag: {value}. Must be 'left' or 'right'.")

    def __init__(self, value):
        if isinstance(value, str):
            self.arm = value.lower()

    @property
    def opposite(self):
        return ArmTag("right") if self.arm == "left" else ArmTag("left")

    def __eq__(self, other):
        if isinstance(other, ArmTag):
            return self.arm == other.arm
        if isinstance(other, str):
            return self.arm == other.lower()
        return False

    def __hash__(self):
        return hash(self.arm)

    def __repr__(self):
        return f"ArmTag('{self.arm}')"

    def __str__(self):
        return self.arm


class Action:
    arm_tag: ArmTag
    action: Literal["move", "gripper"]
    target_pose: list = None
    target_gripper_pos: float = None

    def __init__(
        self,
        arm_tag: ArmTag | Literal["left", "right"],
        action: Literal["move", "open", "close", "gripper"],
        target_pose: sapien.Pose | list | np.ndarray = None,
        target_gripper_pos: float = None,
        **args,
    ):
        self.arm_tag = ArmTag(arm_tag)
        if action != "move":
            if action == "open":
                self.action = "gripper"
                self.target_gripper_pos = (
                    target_gripper_pos if target_gripper_pos is not None else 1.0
                )
            elif action == "close":
                self.action = "gripper"
                self.target_gripper_pos = (
                    target_gripper_pos if target_gripper_pos is not None else 0.0
                )
            elif action == "gripper":
                self.action = "gripper"
            else:
                raise ValueError(
                    f"Invalid action: {action}. Must be 'open' or 'close'."
                )
            assert self.target_gripper_pos is not None, (
                "target_gripper_pos cannot be None for gripper action."
            )
        else:
            self.action = "move"
            assert target_pose is not None, (
                "target_pose cannot be None for move action."
            )
            self.target_pose = _tolist(target_pose)
        self.args = args

    def __str__(self):
        result = f"{self.arm_tag}: {self.action}"
        if self.action == "move":
            result += f"({self.target_pose})"
        else:
            result += f"({self.target_gripper_pos})"
        if self.args:
            result += f"    {self.args}"
        return result


def offset_move_actions_z(action_group, delta: float, count: int = 2) -> bool:
    """Offset the first planned move targets; return False if planning produced no actions."""
    try:
        actions = action_group[1]
        targets = [
            action.target_pose for action in actions[:count] if action.action == "move"
        ]
    except (AttributeError, IndexError, TypeError):
        return False
    if (
        len(actions) < count
        or len(targets) != count
        or any(len(target) < 3 for target in targets)
    ):
        return False
    for target in targets:
        target[2] += delta
    return True


def retime_arm_trajectory(trajectory, speed_scale: float):
    """Slow a planned arm trajectory while preserving its geometric path."""
    if trajectory is None or speed_scale == 1.0:
        return trajectory
    if not np.isfinite(speed_scale) or not 0.0 < speed_scale <= 1.0:
        raise ValueError("speed_scale must be in (0, 1]")

    positions = np.asarray(trajectory["position"])
    velocities = np.asarray(trajectory["velocity"])
    if len(positions) < 2:
        return trajectory

    sample_count = int(np.ceil((len(positions) - 1) / speed_scale)) + 1
    samples = np.linspace(0.0, len(positions) - 1, sample_count)
    lower = np.floor(samples).astype(int)
    upper = np.minimum(lower + 1, len(positions) - 1)
    blend = (samples - lower)[:, None]

    retimed = dict(trajectory)
    retimed["position"] = (
        positions[lower] * (1.0 - blend) + positions[upper] * blend
    ).astype(positions.dtype, copy=False)
    retimed["velocity"] = (
        (velocities[lower] * (1.0 - blend) + velocities[upper] * blend) * speed_scale
    ).astype(velocities.dtype, copy=False)
    return retimed
