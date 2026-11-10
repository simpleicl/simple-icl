# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
from dataclasses import dataclass, field

import draccus

from lerobot.common.robot_devices.cameras.configs import (
    CameraConfig,
    IntelRealSenseCameraConfig,
)


@dataclass
class RobotConfig(draccus.ChoiceRegistry, abc.ABC):
    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)


@RobotConfig.register_subclass("franka")
@dataclass
class FrankaRobotConfig(RobotConfig):
    """Config for a Franka FR3 arm teleoperated by a 3D SpaceMouse.

    Controller  : pyspacemouse 3D mouse -> a 6-DoF Cartesian twist command.
    Actuator    : pylibfranka Franka FR3 arm.

    The twist is integrated into a desired Cartesian pose and handed to
    libfranka's Cartesian pose controller, which internally solves inverse
    kinematics. The resulting *commanded joint positions* (``RobotState.q_d``)
    are what gets recorded as the ``action`` — i.e. the pose control signal is
    converted to a joint signal and that joint signal is the action.

    Multiple arms are supported by listing several ips / mouse paths (the order
    of ``arm_names``, ``robot_ips`` and ``mouse_paths`` must match).

    Operating mode (``control_mode``):
      * "teleop" -> SpaceMouse drives the arm, joint action is recorded (data
        collection via ``control_robot.py --control.type=record``).
      * "replay" -> ``send_action`` drives the arm via joint-position control
        (replaying a dataset with ``replay.sh``, or executing policy actions
        with ``lerobot/scripts/franka_fastwam_client.py``).
    """

    control_mode: str = "teleop"

    # One entry per arm; the three lists must be the same length and order.
    # The values shipped in this repo are placeholders: fill in your actual
    # IPs / device paths before real-robot recording (see README section 3).
    arm_names: list[str] = field(default_factory=lambda: ["left", "right"])
    robot_ips: list[str] = field(
        default_factory=lambda: ["TODO_LEFT_ARM_IP", "TODO_RIGHT_ARM_IP"]
    )
    # SpaceMouse hidraw device paths.
    mouse_paths: list[str] = field(
        default_factory=lambda: ["TODO_LEFT_MOUSE_PATH", "TODO_RIGHT_MOUSE_PATH"]
    )

    # Joint configuration each arm moves to before teleoperation starts.
    # One entry per arm, keyed by the arm name (must match `arm_names`).
    reset_joint_positions: dict[str, list[float]] = field(
        default_factory=lambda: {
            "left": [0.0, -0.3, 0.0, -2.8, 0.0, 2.5, 0.83],
            "right": [0.0, -0.3, 0.0, -2.8, 0.0, 2.5, 0.83],
        }
    )
    # Homing: duration of the cosine interpolation, then hold at the target until
    # the measured per-joint error is within tolerance (rad) or the settle window
    # elapses. Homing is only reported as reached once this convergence check passes.
    home_duration_s: float = 1.5
    home_joint_tolerance: float = 0.03
    home_settle_time_s: float = 5.0

    # Replay-mode target tracking: `reset_speed_factor` scales the per-joint
    # velocity cap, `replay_joint_acceleration` (rad/s^2) caps how fast that
    # velocity may change, and `replay_joint_jerk` (rad/s^3) caps how fast the
    # acceleration itself may change. The caps are what stop a jumped policy
    # target from being commanded as an instantaneous velocity step, which
    # libfranka aborts as a motion-generator discontinuity (reflex).
    reset_speed_factor: float = 0.08
    replay_joint_acceleration: float = 1.0
    replay_joint_jerk: float = 1000.0

    # SpaceMouse -> Cartesian twist shaping.
    dead_zone: float = 0.1
    max_linear_velocity: float = 0.30  # m/s
    max_angular_velocity: float = 0.55  # rad/s
    max_linear_acceleration: float = 0.4  # m/s^2
    max_angular_acceleration: float = 0.7  # rad/s^2

    # Keep the gripper pointing straight down.
    lock_gripper_vertical: bool = True

    # Collision behaviour thresholds passed to `robot.set_collision_behavior`.
    lower_torque_thresholds: list[float] = field(default_factory=lambda: [20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0])
    upper_torque_thresholds: list[float] = field(default_factory=lambda: [20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0])
    lower_force_thresholds: list[float] = field(default_factory=lambda: [20.0, 20.0, 20.0, 25.0, 25.0, 25.0])
    upper_force_thresholds: list[float] = field(default_factory=lambda: [20.0, 20.0, 20.0, 25.0, 25.0, 25.0])

    # Gripper: open/closed widths (m), grasp speed (m/s) and force (N).
    gripper_open_width: float = 0.075
    gripper_close_width: float = 0.005
    gripper_speed: float = 0.4
    gripper_force: float = 3.0
    # Skip the startup homing() calibration (the close-then-open sweep). Homing
    # keeps grasp/move widths accurate; only skip it if the hand is pre-calibrated.
    skip_gripper_homing: bool = False

    # Safety: max relative joint target magnitude for goal clamping in replay mode.
    max_relative_target: float | list[float] | None = None

    # Cameras configuration. The serial numbers select the physical cameras.
    # The serial numbers shipped in this repo are placeholders (None): fill in
    # your cameras' real serial numbers before real-robot recording (see
    # README section 3).
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "cam_high": IntelRealSenseCameraConfig(
                serial_number=None,  # TODO: cam_high serial number
                fps=30,
                width=848,
                height=480,
            ),
            "cam_left_wrist": IntelRealSenseCameraConfig(
                serial_number=None,  # TODO: cam_left_wrist serial number
                fps=30,
                width=848,
                height=480,
            ),
            "cam_right_wrist": IntelRealSenseCameraConfig(
                serial_number=None,  # TODO: cam_right_wrist serial number
                fps=30,
                width=848,
                height=480,
            ),
        }
    )

    # No-hardware mode: cameras emit constant test frames and the arm control
    # process is replaced by an idle echo loop (see frankarobot.py).
    mock: bool = False

    def __post_init__(self):
        n = len(self.arm_names)
        if not (len(self.robot_ips) == n and len(self.mouse_paths) == n):
            raise ValueError(
                "FrankaRobotConfig: `arm_names`, `robot_ips` and `mouse_paths` must have the same length "
                f"(got {n}, {len(self.robot_ips)}, {len(self.mouse_paths)})."
            )
        missing = [name for name in self.arm_names if name not in self.reset_joint_positions]
        if missing:
            raise ValueError(
                "FrankaRobotConfig: `reset_joint_positions` is missing entries for arm(s) "
                f"{missing}; it must define a joint configuration for every arm in `arm_names`."
            )
        if self.control_mode not in ("teleop", "replay"):
            raise ValueError(f"FrankaRobotConfig: unknown control_mode '{self.control_mode}'.")
        if self.replay_joint_acceleration <= 0.0:
            raise ValueError(
                "FrankaRobotConfig: `replay_joint_acceleration` must be > 0 rad/s^2 "
                f"(got {self.replay_joint_acceleration}); a non-positive cap freezes "
                "replay-mode tracking."
            )
        if self.replay_joint_jerk <= 0.0:
            raise ValueError(
                "FrankaRobotConfig: `replay_joint_jerk` must be > 0 rad/s^3 "
                f"(got {self.replay_joint_jerk}); a non-positive cap freezes "
                "replay-mode tracking."
            )
        if self.mock:
            for cam in self.cameras.values():
                if not cam.mock:
                    cam.mock = True
