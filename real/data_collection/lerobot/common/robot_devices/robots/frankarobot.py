"""FrankaRobot: teleoperate a Franka FR3 arm with a 3D SpaceMouse and record data.

Pipeline
--------
    pyspacemouse (3D mouse)  ->  6-DoF Cartesian twist
                             ->  integrated into a desired Cartesian pose
    pylibfranka (Franka FR3) ->  Cartesian pose controller solves inverse
                                 kinematics internally
                             ->  commanded joint positions (RobotState.q_d)

The commanded joint positions ``q_d`` are exactly the "joint signal sent to the
executor", and that is what we record as the ``action``. ``observation.state``
holds the *measured* joint positions ``q``.

libfranka requires an uninterrupted ~1 kHz realtime control loop, which is
incompatible with lerobot's ~30 Hz ``teleop_step``. So the SpaceMouse read loop,
the Franka control loop and the gripper loop each run in their own process and
communicate with the main process through shared-memory arrays. ``teleop_step``
merely samples the latest published joint values and reads the cameras.

The gripper dof follows the same commanded/measured split as the joints: the
``action`` carries the commanded open/close width, while ``observation.state``
carries the finger width read back from the hardware (see ``_gripper_proc`` for
the one caveat: it is not sampled during a blocking open/close motion).
"""

import ctypes
import logging
import multiprocessing as mp
import os
import time

import numpy as np
import torch

# Set FRANKA_DEBUG=1 to print SpaceMouse/gripper timing to localize latency.
_DEBUG = bool(os.environ.get("FRANKA_DEBUG"))

from lerobot.common.robot_devices.cameras.utils import make_cameras_from_configs
from lerobot.common.robot_devices.robots.configs import FrankaRobotConfig
from lerobot.common.robot_devices.utils import (
    RobotDeviceAlreadyConnectedError,
    RobotDeviceNotConnectedError,
)

# Degrees of freedom of a single Franka arm (7 joints), plus 1 gripper dof.
ARM_DOF = 7
GRIPPER_DOF = 1
ARM_TOTAL_DOF = ARM_DOF + GRIPPER_DOF

# Per-joint velocity ceiling (rad/s) the replay tracker is allowed to reach,
# before `reset_speed_factor` scales it. Exposed because anything that generates
# replay targets has to respect the same ceiling the tracker enforces -- e.g.
# `franka_fastwam_client.py` reports it to the policy server so the server's TOPP
# re-timing plans against the limits the tracker will actually apply.
REPLAY_MAX_JOINT_VEL = np.array([2.0, 2.0, 2.0, 2.0, 2.5, 2.5, 2.5])


def ensure_safe_goal_position(
    goal_pos: torch.Tensor, present_pos: torch.Tensor, max_relative_target: float | list[float]
) -> torch.Tensor:
    diff = goal_pos - present_pos
    max_rel = (
        torch.tensor(max_relative_target)
        if not isinstance(max_relative_target, torch.Tensor)
        else max_relative_target
    )
    safe_diff = torch.clamp(diff, -max_rel, max_rel)
    safe_goal = present_pos + safe_diff
    if not torch.allclose(goal_pos, safe_goal):
        logging.warning(
            "Relative goal position magnitude had to be clamped to be safe.\n"
            f"  requested: {diff}\n  clamped:   {safe_diff}"
        )
    return safe_goal


# ----------------------------------------------------------------------------
# Per-arm shared memory bundle (all values are plain float/bool, process-safe).
# ----------------------------------------------------------------------------
class _ArmShared:
    def __init__(self, reset_q: np.ndarray):
        # SpaceMouse -> Franka: 6-DoF twist [vx, vy, vz, wx, wy, wz] in [-1, 1].
        self.speed_command = mp.Array("f", [0.0] * 6)
        # Gripper open (True) / closed (False) toggle.
        self.open_command = mp.Value(ctypes.c_bool, True)
        # perf_counter timestamp (system-wide CLOCK_MONOTONIC, comparable across
        # processes) of the last gripper toggle — used only for latency debugging.
        self.gripper_cmd_ts = mp.Value("d", 0.0)
        # Gripper -> main: measured finger width (m). NaN means "no live reading"
        # (mock mode, or the state reader failed), in which case the main process
        # falls back to the commanded open/close width.
        self.measured_gripper_width = mp.Value("d", float("nan"))
        # Gripper -> main: the side the hand has actually *reached*, i.e.
        # `open_command` delayed by the blocking move/grasp call. It lags
        # `open_command` for the whole duration of a transition, so the main
        # process can wait for a commanded open/close to complete. Starts True:
        # homing() leaves the fingers open.
        self.gripper_open_state = mp.Value(ctypes.c_bool, True)
        # Franka -> main: commanded joint positions q_d (the recorded action).
        self.commanded_q = mp.Array("f", list(reset_q))
        # Franka -> main: measured joint positions q (observation.state).
        self.measured_q = mp.Array("f", list(reset_q))
        # main -> Franka (replay mode only): joint-position target.
        self.target_q = mp.Array("f", list(reset_q))
        # Franka -> main: True once the reset motion finished and control started.
        self.ready = mp.Value(ctypes.c_bool, False)
        # main -> Franka: F3 toggle. Starts True so the arm waits at the reset
        # pose until the first F3 (which also starts recording). Toggling to
        # False drives the arm; toggling back to True pauses + re-homes it.
        self.paused = mp.Value(ctypes.c_bool, True)
        # main -> children: keep the loops running.
        self.is_running = mp.Value(ctypes.c_bool, True)


# ----------------------------------------------------------------------------
# Child processes
# ----------------------------------------------------------------------------
def _spacemouse_proc(shared: _ArmShared, path: str):
    """Read the 3D mouse and publish a normalized twist + gripper toggle."""
    import pyspacemouse

    device = pyspacemouse.open_by_path(path)
    np_speed = np.frombuffer(shared.speed_command.get_obj(), dtype=np.float32)
    button_1_pressed = False
    try:
        next_t = time.perf_counter()
        while shared.is_running.value:
            state = device.read()
            with shared.speed_command.get_lock():
                np_speed[:] = np.array(
                    [state.y, -state.x, state.z, state.roll, state.pitch, -state.yaw],
                    dtype=np.float32,
                )
            # Button 1 toggles the gripper. (Stopping/re-homing is driven by the
            # keyboard F3 key in the main process, not by a mouse button.)
            if bool(state.buttons[1]) and not button_1_pressed:
                shared.open_command.value = not shared.open_command.value
                shared.gripper_cmd_ts.value = time.perf_counter()
                if _DEBUG:
                    print(f"[SpaceMouse] button1 -> open_command={shared.open_command.value} "
                          f"@ {shared.gripper_cmd_ts.value:.4f}", flush=True)
            button_1_pressed = bool(state.buttons[1])

            next_t += 0.001
            sleep_s = next_t - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
    finally:
        device.close()


def _franka_proc(shared: _ArmShared, ip: str, config: FrankaRobotConfig, reset_q: np.ndarray):
    """1 kHz Franka control loop.

    First moves to the reset joint configuration, then either follows the
    SpaceMouse twist through Cartesian pose control (teleop) or tracks the
    shared joint target (replay). In both cases it continuously publishes the
    commanded joints (``q_d``) and the measured joints (``q``).
    """
    import pylibfranka
    from pylibfranka import CartesianPose, ControllerMode, JointPositions, Robot
    from scipy.spatial.transform import Rotation as R

    reset_q = np.asarray(reset_q, dtype=np.float64)
    commanded_view = np.frombuffer(shared.commanded_q.get_obj(), dtype=np.float32)
    measured_view = np.frombuffer(shared.measured_q.get_obj(), dtype=np.float32)

    def publish(state):
        # q_d is the commanded/desired joint position (falls back to q).
        q_cmd = np.asarray(getattr(state, "q_d", state.q), dtype=np.float32)
        with shared.commanded_q.get_lock():
            commanded_view[:] = q_cmd
        with shared.measured_q.get_lock():
            measured_view[:] = np.asarray(state.q, dtype=np.float32)

    robot = Robot(ip, realtime_config=pylibfranka._pylibfranka.RealtimeConfig.kIgnore)
    robot.set_collision_behavior(
        config.lower_torque_thresholds,
        config.upper_torque_thresholds,
        config.lower_force_thresholds,
        config.upper_force_thresholds,
    )

    try:
        # Move to the reset pose, then report ready so `connect()` can return.
        if not _franka_home(shared, robot, config, publish, ControllerMode, JointPositions, reset_q):
            return
        shared.ready.value = True

        if config.control_mode == "replay":
            # Same fault/recover/resume contract as the teleop branch below: a
            # `ControlException` (collision reflex, commanded velocity/jerk
            # discontinuity, ...) aborts the libfranka motion but must not take
            # this process down. Letting it propagate is silent from the caller's
            # side -- `send_action` only writes shared memory and never checks the
            # child is alive, so the policy loop keeps streaming targets at nobody
            # while `observation.state` freezes at the last published q.
            #
            # Recovery is in-place: clear the fault, re-enter the tracker (which
            # re-seeds `current` from q_d and zeroes vel/acc, so the restart is
            # continuous by construction) and keep following the policy. No
            # re-home -- that would drive the arm away from the task mid-rollout.
            consecutive_errors = 0
            while shared.is_running.value:
                try:
                    _franka_replay_loop(
                        shared, robot, config, publish, ControllerMode, JointPositions, reset_q
                    )
                    break  # returned on its own -> shutdown was requested
                except Exception as e:  # noqa: BLE001
                    consecutive_errors += 1
                    logging.error(
                        f"[Franka@{ip}] replay control fault ({consecutive_errors}): {e}. "
                        "Recovering in place and resuming."
                    )
                    try:
                        robot.stop()
                    except Exception:  # noqa: BLE001
                        pass
                    recovered = _franka_recover(robot, ip)
                    # Resume holding where the reflex actually left the arm instead
                    # of stepping back toward the pre-fault target, which may sit a
                    # long way off (and is what we just collided on the way to).
                    # The next `send_action` overwrites this ~1 control period later.
                    _franka_hold_here(shared, robot, measured_view)
                    if not recovered:
                        # Fault still latched: back off so a permanent failure does
                        # not turn into a tight recover/restart loop on the FCI.
                        time.sleep(min(0.2 * consecutive_errors, 2.0))
            return

        # Teleop, F3-driven:
        #   F3 #1 -> start driving the arm (recording starts in the pipeline)
        #   F3 #2 -> pause driving + re-home (recording pauses)
        #   F3 #3 -> start driving again (recording resumes)
        # The arm starts paused at the reset pose, waiting for the first F3.
        consecutive_errors = 0
        while shared.is_running.value:
            # Wait for F3 to (re)start control.
            while shared.is_running.value and shared.paused.value:
                time.sleep(0.05)
            if not shared.is_running.value:
                break

            try:
                # Drive until F3 requests a pause (or shutdown).
                _franka_teleop_loop(shared, robot, config, publish, ControllerMode, CartesianPose, R)
                try:
                    robot.stop()
                except Exception:  # noqa: BLE001
                    pass
                if not shared.is_running.value:
                    break

                # F3 paused recording -> re-home immediately, then await next F3.
                if shared.paused.value:
                    print(f"[Franka@{ip}] paused & re-homing; press F3 again to resume recording.")
                    if not _franka_home(shared, robot, config, publish, ControllerMode, JointPositions, reset_q):
                        break
                consecutive_errors = 0
            except Exception as e:  # noqa: BLE001
                # A control fault (e.g. the arm was driven past a velocity /
                # acceleration limit or tripped the collision reflex) aborts the
                # libfranka motion. Recover in-place and resume from the current
                # pose instead of letting the control process die (which would
                # leave the arm frozen while the gripper keeps working).
                consecutive_errors += 1
                logging.error(
                    f"[Franka@{ip}] control fault ({consecutive_errors}): {e}. "
                    "Recovering and resuming."
                )
                try:
                    robot.stop()
                except Exception:  # noqa: BLE001
                    pass
                _franka_recover(robot, ip)
                # Only a *singular-pose* rejection needs special handling. A normal
                # control fault (velocity/acceleration limit, collision reflex, ...)
                # does NOT leave the arm at a singularity, so we keep the original
                # behaviour there: recover and resume Cartesian control in place.
                #
                # But if the arm was driven into a kinematic singularity, restarting
                # Cartesian pose control from that pose is rejected by the FR3
                # ("cannot start at singular pose") — trapping us in an endless
                # fault/recover/retry loop. Escape in JOINT space first: `_franka_home`
                # uses joint-position control, which has no singular-pose check, so it
                # can always drive the arm back to the (well-conditioned) reset
                # configuration; only then is it safe to resume Cartesian teleop.
                is_singular = "singular" in str(e).lower()
                if is_singular and shared.is_running.value and not shared.paused.value:
                    print(f"[Franka@{ip}] singular pose — re-homing (joint space) to escape it.")
                    try:
                        if not _franka_home(shared, robot, config, publish, ControllerMode, JointPositions, reset_q):
                            break
                    except Exception as home_err:  # noqa: BLE001
                        # Joint re-home itself faulted: recover and let the next
                        # loop iteration try again rather than killing the process.
                        logging.error(f"[Franka@{ip}] re-home after singular pose failed: {home_err}. Retrying.")
                        try:
                            robot.stop()
                        except Exception:  # noqa: BLE001
                            pass
                        _franka_recover(robot, ip)
                        continue
                # paused is unchanged. Non-singular fault: re-drive from the current
                # pose. Singular fault: we just re-homed, so re-drive from reset.
                # Faulted while paused: wait for F3.
    except KeyboardInterrupt:
        pass
    except Exception as e:  # noqa: BLE001
        logging.error(f"[Franka@{ip}] control loop error: {e}")
        if "Reflex" in str(e):
            logging.info(f"[Franka@{ip}] Attempting automatic recovery from Reflex mode...")
            recovered = _franka_recover(robot, ip)
            if recovered:
                logging.info(f"[Franka@{ip}] Successfully recovered from Reflex. Restarting control loop...")
                try:
                    robot.stop()
                except Exception:  # noqa: BLE001
                    pass
                # Re-enter the main control logic after successful recovery
                _franka_proc(shared, ip, config, reset_q)
                return
            else:
                logging.error(f"[Franka@{ip}] Failed to recover from Reflex mode. Stopping control process.")
    finally:
        try:
            robot.stop()
        except Exception:  # noqa: BLE001
            pass
        print(f"[Franka@{ip}] control process stopped.")


def _franka_recover(robot, ip: str) -> bool:
    """Clear a libfranka control fault so a new motion can be started.

    After a ``ControlException`` (limit violation / collision reflex), libfranka
    latches an error and refuses to start a new controller until it is cleared.
    The binding may spell this ``automatic_error_recovery`` or
    ``automaticErrorRecovery`` — try both.
    """
    for name in ("automatic_error_recovery", "automaticErrorRecovery"):
        fn = getattr(robot, name, None)
        if fn is not None:
            try:
                fn()
                return True
            except Exception as e:  # noqa: BLE001
                logging.error(f"[Franka@{ip}] automatic error recovery failed: {e}")
                return False
    logging.warning(
        f"[Franka@{ip}] no automatic error-recovery method on Robot; cannot clear fault."
    )
    return False


def _franka_hold_here(shared: _ArmShared, robot, measured_view: np.ndarray) -> None:
    """Point the replay target at the arm's current pose, so it holds still.

    Called after a fault is cleared: the target that was in flight is stale (the
    reflex stopped the arm somewhere short of it) and re-entering the tracker
    against it would immediately drive back along the path that just faulted.
    Prefers a fresh ``read_once`` and falls back to the last published measured
    joints, which may lag the reflex braking by a cycle but is always available.
    """
    try:
        q_now = np.asarray(robot.read_once().q, dtype=np.float32)
    except Exception:  # noqa: BLE001
        with shared.measured_q.get_lock():
            q_now = measured_view.copy()
    with shared.target_q.get_lock():
        np.frombuffer(shared.target_q.get_obj(), dtype=np.float32)[:] = q_now


def _franka_home(shared, robot, config, publish, ControllerMode, JointPositions, reset_q) -> bool:
    """Move to the reset joint configuration and confirm the arm got there.

    Phase 1 commands a cosine-smoothed trajectory from the current joints to
    ``reset_q``. Phase 2 keeps commanding ``reset_q`` and only declares the
    motion finished once the *measured* per-joint error is within
    ``home_joint_tolerance`` (converged) or the settle window elapses. This
    guards against the JointImpedance controller's tracking error, so a True
    return means the arm actually reached the target (a warning is logged if it
    only stopped because the settle window ran out).

    Returns True if homing completed, False if interrupted by shutdown.
    """
    reset_q = np.asarray(reset_q, dtype=np.float64)
    initial_q = np.asarray(robot.read_once().q, dtype=np.float64)
    motion_duration = config.home_duration_s
    tol = config.home_joint_tolerance
    settle_timeout = config.home_settle_time_s

    elapsed = 0.0
    settle_elapsed = 0.0
    active_control = robot.start_joint_position_control(ControllerMode.JointImpedance)

    def _finish(converged: bool, error: float):
        handle = JointPositions(list(reset_q))
        handle.motion_finished = True
        try:
            active_control.writeOnce(handle)
        except Exception:  # noqa: BLE001
            pass
        robot.stop()
        if not converged:
            logging.warning(
                f"[Franka] homing did not fully converge: max joint error "
                f"{error:.4f} rad > tol {tol:.4f} after a {settle_timeout:.1f}s settle window."
            )

    try:
        while shared.is_running.value:
            state, duration = active_control.readOnce()
            dt = duration.to_sec()
            publish(state)

            if elapsed < motion_duration:
                # Phase 1: cosine-smoothed interpolation initial_q -> reset_q.
                elapsed += dt
                progress = float(np.clip(elapsed / motion_duration, 0.0, 1.0))
                smooth = 0.5 * (1.0 - np.cos(np.pi * progress))
                desired_q = initial_q + (reset_q - initial_q) * smooth
                active_control.writeOnce(JointPositions(list(desired_q)))
                continue

            # Phase 2: hold at reset_q until the measured error converges (or the
            # settle window is exhausted).
            settle_elapsed += dt
            error = float(np.max(np.abs(np.asarray(state.q, dtype=np.float64) - reset_q)))
            if error < tol:
                _finish(True, error)
                return True
            if settle_elapsed >= settle_timeout:
                _finish(False, error)
                return True
            active_control.writeOnce(JointPositions(list(reset_q)))

        # Shutdown requested mid-motion.
        _finish(True, 0.0)
        return False
    except Exception:  # noqa: BLE001
        robot.stop()
        raise


def _franka_teleop_loop(shared, robot, config, publish, ControllerMode, CartesianPose, R):
    """SpaceMouse twist -> desired Cartesian pose -> Cartesian pose control."""
    velocity_scales = np.array(
        [
            config.max_linear_velocity,
            config.max_linear_velocity,
            config.max_linear_velocity,
            config.max_angular_velocity,
            config.max_angular_velocity,
            config.max_angular_velocity,
        ]
    )
    max_acceleration = np.array(
        [
            config.max_linear_acceleration,
            config.max_linear_acceleration,
            config.max_linear_acceleration,
            config.max_angular_acceleration,
            config.max_angular_acceleration,
            config.max_angular_acceleration,
        ]
    )
    dead_zone = config.dead_zone
    lock_gripper_vertical = config.lock_gripper_vertical

    active_control = robot.start_cartesian_pose_control(ControllerMode.JointImpedance)
    np_speed = np.frombuffer(shared.speed_command.get_obj(), dtype=np.float32)
    internal_speed = np.zeros(6, dtype=np.float32)
    desired_T = None
    shared.ready.value = True
    cartesian_pose = None
    try:
        # Runs until F3 requests a pause (shared.paused) or shutdown.
        while shared.is_running.value and not shared.paused.value:
            state, duration = active_control.readOnce()
            publish(state)
            if desired_T is None:
                desired_T = np.reshape(state.O_T_EE_d, (4, 4), order="F")
            dt = duration.to_sec()
            if dt <= 0.0:
                dt = 0.001

            with shared.speed_command.get_lock():
                cmd = np_speed.copy()
            # Dead-zone + rescale so the usable range stays [0, 1].
            cmd[np.abs(cmd) < dead_zone] = 0.0
            cmd = (cmd - dead_zone * np.sign(cmd)) / (1.0 - dead_zone)
            cmd = cmd * velocity_scales

            if lock_gripper_vertical:
                # Ignore roll/pitch (rotation about the world x/y axes) so the
                # tool z-axis keeps pointing straight down; only translation and
                # yaw rotation about the world vertical (z) axis remain. A pure
                # world-z rotation leaves the tool z-direction unchanged, so a
                # gripper that starts vertical stays vertical.
                cmd[3] = 0.0
                cmd[4] = 0.0

            # Acceleration-limited velocity integration.
            acceleration = (cmd - internal_speed) / dt
            acceleration = np.clip(acceleration, -max_acceleration, max_acceleration)
            internal_speed += acceleration * dt

            if lock_gripper_vertical:
                # Also clear the integrated x/y angular velocity so acceleration
                # limiting can't leak a residual tilt into the desired pose.
                internal_speed[3] = 0.0
                internal_speed[4] = 0.0

            delta_pose = internal_speed * dt
            delta_R = R.from_rotvec(delta_pose[3:])
            desired_p = desired_T[:3, 3] + delta_pose[:3]
            desired_R = delta_R * R.from_matrix(desired_T[:3, :3])
            desired_T = np.eye(4)
            desired_T[:3, :3] = desired_R.as_matrix()
            desired_T[:3, 3] = desired_p

            cartesian_pose = CartesianPose(np.reshape(desired_T, (16,), order="F"))
            active_control.writeOnce(cartesian_pose)
    finally:
        if cartesian_pose is not None:
            cartesian_pose.motion_finished = True
            try:
                active_control.writeOnce(cartesian_pose)
            except Exception:  # noqa: BLE001
                pass


def _franka_replay_loop(shared, robot, config, publish, ControllerMode, JointPositions, reset_q):
    """Track the shared joint target under velocity, acceleration and jerk limits.

    Clamping only the per-cycle position step is not enough: the first cycle
    after a target jump then commands a 0 -> v_max velocity step, i.e. an
    instantaneous v_max / 1 ms acceleration, and libfranka rejects the commanded
    trajectory with ``joint_motion_generator_velocity_discontinuity`` /
    ``..._acceleration_discontinuity`` (reflex, motion aborted). Note what those
    two names actually mean: libfranka finite-differences the *commanded*
    positions, so "velocity discontinuity" is the commanded **acceleration**
    over its limit (~7.5-20 rad/s^2 per joint) and "acceleration discontinuity"
    is the commanded **jerk** over its limit (~3750-10000 rad/s^3). Both firing
    together is the signature of a one-cycle step in the commanded velocity.

    Since ``send_action`` publishes a policy target that can sit a large jump
    away from the current pose, ramp through all three derivatives: brake into
    the target on a jerk-aware velocity profile, rate-limit the acceleration
    toward that profile, and rate-limit the acceleration's own change. Every
    commanded quantity is then continuous by construction.

    Two things this loop must *not* do, both of which trip the reflex:
      * hard-reset ``vel``/``acc`` to zero on some "close enough" condition.
        During replay the target is *moving*, so the steady state is
        ``err ~ 0`` with ``vel`` at the target's velocity — a deadband on
        ``|err|`` therefore fires while tracking well, and zeroing a 0.014 rad/s
        velocity in one 1 ms tick is 14 rad/s^2 / 1.4e4 rad/s^3, over both
        limits. Anti-overshoot belongs on the profile input (``v_des``), never
        on the integrator state.
      * seed ``current`` from the *desired* reset pose. The impedance controller
        leaves a residual error at the end of homing, so the first ``writeOnce``
        would be a position step. Seed from the arm's own commanded position
        (``q_d``) instead, the way `_franka_teleop_loop` seeds from ``O_T_EE_d``.
    """
    target_view = np.frombuffer(shared.target_q.get_obj(), dtype=np.float32)
    # Per-joint velocity cap (rad/s); `reset_speed_factor` scales the tracking speed.
    max_vel = REPLAY_MAX_JOINT_VEL * config.reset_speed_factor
    max_acc = float(config.replay_joint_acceleration)
    max_jerk = float(getattr(config, "replay_joint_jerk", 1000.0))
    # Jerk-aware braking: stopping from speed v needs v^2/(2a) + v*a/(2j) of
    # travel (the second term is the distance covered while the deceleration
    # ramps in). Inverting that for v given the remaining distance d yields
    #   v_brake = (sqrt(k^2 + 8*a*d) - k) / 2,  k = a^2 / j
    # which degenerates to the plain sqrt(2*a*d) profile as j -> inf. Using the
    # jerk-agnostic profile with a finite jerk cap would arrive at the target
    # with residual velocity, overshoot, and hunt around it.
    brake_k = max_acc * max_acc / max_jerk

    active_control = robot.start_joint_position_control(ControllerMode.JointImpedance)
    current = None  # seeded from the arm's commanded pose on the first cycle
    vel = np.zeros(len(reset_q), dtype=np.float64)
    acc = np.zeros_like(vel)
    shared.ready.value = True
    joint_positions = None
    try:
        while shared.is_running.value:
            state, duration = active_control.readOnce()
            publish(state)
            if current is None:
                current = np.asarray(getattr(state, "q_d", state.q), dtype=np.float64)
            # `duration` is the elapsed control time; clamp it so a late cycle
            # (or a bogus first reading) can't turn into a large `vel * dt`
            # position step inside a single 1 ms tick. Under-integrating just
            # makes the tracker lag slightly, which is always safe.
            dt = duration.to_sec()
            if not 0.0005 <= dt <= 0.002:
                dt = 0.001
            with shared.target_q.get_lock():
                target = target_view.astype(np.float64)
            err = target - current

            # Fastest velocity we can still brake out of within `err`, capped by
            # `max_vel` and by what would step past the target this cycle.
            v_brake = 0.5 * (np.sqrt(brake_k * brake_k + 8.0 * max_acc * np.abs(err)) - brake_k)
            v_des = np.sign(err) * np.minimum(np.minimum(v_brake, max_vel), np.abs(err) / dt)

            # Acceleration toward that profile, then jerk-limit its change.
            acc_des = np.clip((v_des - vel) / dt, -max_acc, max_acc)
            acc = acc + np.clip(acc_des - acc, -max_jerk * dt, max_jerk * dt)
            vel = np.clip(vel + acc * dt, -max_vel, max_vel)
            current = current + vel * dt
            joint_positions = JointPositions(list(current))
            active_control.writeOnce(joint_positions)
    finally:
        if joint_positions is not None:
            joint_positions.motion_finished = True
            try:
                active_control.writeOnce(joint_positions)
            except Exception:  # noqa: BLE001
                pass


def _gripper_proc(shared: _ArmShared, ip: str, config: FrankaRobotConfig):
    """Open/close the gripper following the shared toggle.

    Note on latency: ``grasp()``/``move()``/``homing()`` are *blocking* — they
    return only after the finger motion completes. The poll loop therefore can't
    service a new toggle until the current command finishes, so a button press
    that lands while a motion is in flight only reacts once that motion is done.
    Opening uses ``move()`` (pure positioning) rather than ``grasp()``: grasp is
    meant to close onto an object and waits to detect one (epsilon/force
    settling), which makes opening feel sluggish to react. Closing keeps
    ``grasp()`` so it applies force on the object.

    This process also publishes the *measured* finger width for
    ``observation.state``. It has to be done from here, on this same connection:
    the gripper server only serves one client, so opening a second connection
    just to read state makes the commands issued here fail. The consequence is
    that the width is sampled between commands only — it holds its last value for
    the duration of a blocking open/close and then jumps.
    """
    from pylibfranka import Gripper as FrankaGripper

    gripper = FrankaGripper(ip)
    if not getattr(config, "skip_gripper_homing", False):
        gripper.homing()  # calibration sweep: fully closes then opens
        print(f"[Gripper@{ip}] homed.")
    move_fn = getattr(gripper, "move", None)

    # Publishing the width is best-effort: it must never take the gripper down.
    # After a few consecutive failures, stop trying and leave NaN behind so the
    # main process falls back to the commanded width.
    width_errors = 0

    def publish_width():
        nonlocal width_errors
        if width_errors >= 5:
            return
        try:
            shared.measured_gripper_width.value = float(gripper.read_once().width)
            width_errors = 0
        except Exception as e:  # noqa: BLE001
            width_errors += 1
            shared.measured_gripper_width.value = float("nan")
            if width_errors >= 5:
                print(f"[Gripper@{ip}] width readback disabled after 5 failures ({e}); "
                      "observation.state falls back to the commanded width.", flush=True)

    def open_gripper():
        if move_fn is not None:
            move_fn(config.gripper_open_width, config.gripper_speed)
        else:
            gripper.grasp(config.gripper_open_width, config.gripper_speed, config.gripper_force, 0.005, 0.04)

    def close_gripper():
        gripper.grasp(config.gripper_close_width, config.gripper_speed, config.gripper_force, 0.005, 0.06)

    def _run(kind, fn):
        # Time both the dispatch latency (toggle -> command start) and the
        # blocking command duration, to localize where the reaction delay is.
        if _DEBUG:
            t0 = time.perf_counter()
            dispatch = t0 - shared.gripper_cmd_ts.value
            print(f"[Gripper@{ip}] {kind}: dispatch +{dispatch*1000:.1f} ms after toggle", flush=True)
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            # Don't let one refused command kill the process: that would leave the
            # gripper unresponsive for the rest of the session.
            print(f"[Gripper@{ip}] {kind} command failed: {e}", flush=True)
        if _DEBUG:
            print(f"[Gripper@{ip}] {kind}: command returned after {(time.perf_counter()-t0)*1000:.1f} ms",
                  flush=True)

    # Published (not process-local) so the main process can tell a commanded
    # toggle apart from a completed one -- it only flips once the blocking
    # move/grasp call has returned.
    shared.gripper_open_state.value = True
    try:
        while shared.is_running.value:
            if shared.open_command.value and not shared.gripper_open_state.value:
                _run("open", open_gripper)
                shared.gripper_open_state.value = True
            elif not shared.open_command.value and shared.gripper_open_state.value:
                _run("close", close_gripper)
                shared.gripper_open_state.value = False
            # Sampled only here, i.e. while no command is in flight; `read_once`
            # blocks for up to one gripper state period, so it goes after the
            # toggle check to keep button-press dispatch latency unchanged.
            publish_width()
            time.sleep(0.001)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            gripper.stop()
        except Exception:  # noqa: BLE001
            pass
        print(f"[Gripper@{ip}] process stopped.")


# ----------------------------------------------------------------------------
# Mock child processes (no hardware) — idle loops that echo commands.
# ----------------------------------------------------------------------------
def _mock_franka_proc(shared: _ArmShared, ip: str, config: FrankaRobotConfig, reset_q: np.ndarray):
    reset_q = np.asarray(reset_q, dtype=np.float32)
    commanded_view = np.frombuffer(shared.commanded_q.get_obj(), dtype=np.float32)
    measured_view = np.frombuffer(shared.measured_q.get_obj(), dtype=np.float32)
    target_view = np.frombuffer(shared.target_q.get_obj(), dtype=np.float32)
    with shared.commanded_q.get_lock():
        commanded_view[:] = reset_q
    with shared.measured_q.get_lock():
        measured_view[:] = reset_q
    shared.ready.value = True
    print(f"[MockFranka@{ip}] control process started.")
    while shared.is_running.value:
        # There is no mock gripper process, so echo the toggle here: a consumer
        # waiting for a commanded open/close to complete would otherwise wait
        # forever under --mock.
        shared.gripper_open_state.value = shared.open_command.value
        if config.control_mode == "replay":
            with shared.target_q.get_lock():
                tgt = target_view.copy()
            with shared.commanded_q.get_lock():
                commanded_view[:] = tgt
            with shared.measured_q.get_lock():
                measured_view[:] = tgt
        time.sleep(0.001)
    print(f"[MockFranka@{ip}] control process stopped.")


# ----------------------------------------------------------------------------
# Robot
# ----------------------------------------------------------------------------
class FrankaRobot:
    """Franka FR3 arm(s) teleoperated by a 3D SpaceMouse for data collection.

    Implements the interface expected by ``control_robot.py`` and the dataset
    recording pipeline (connect / teleop_step / disconnect, features).
    """

    def __init__(self, config: FrankaRobotConfig):
        self.config = config
        self.mock = config.mock
        self.robot_type = config.type
        self.is_connected = False
        self.logs: dict = {}

        self.arm_names = list(config.arm_names)
        # Per-arm reset joint configuration (keyed by arm name).
        self._reset_q = {
            arm: np.array(config.reset_joint_positions[arm], dtype=np.float32)
            for arm in self.arm_names
        }

        # Per-arm joint/gripper motor names, e.g. "main_joint_1" ... "main_gripper".
        self._motor_names: list[str] = []
        for arm in self.arm_names:
            self._motor_names.extend(f"{arm}_joint_{i}" for i in range(1, ARM_DOF + 1))
            self._motor_names.append(f"{arm}_gripper")

        self.cameras = make_cameras_from_configs(config.cameras)

        self._shared: dict[str, _ArmShared] = {}
        self._procs: list[mp.Process] = []

        # F3 keyboard toggle for start / pause+re-home / resume during teleop.
        # Starts True (paused): the first F3 starts control + recording.
        self._pause_listener = None
        self._paused_state = True

    # ------------------------------------------------------------------
    # Feature descriptors expected by the dataset / control pipeline
    # ------------------------------------------------------------------
    @property
    def leader_arms(self) -> dict:
        # Compatibility shim: control_utils iterates robot.leader_arms keys.
        return {name: None for name in self.arm_names}

    @property
    def follower_arms(self) -> dict:
        return {name: None for name in self.arm_names}

    @property
    def motor_features(self) -> dict:
        dof = len(self._motor_names)
        return {
            "action": {
                "dtype": "float32",
                "shape": (dof,),
                "names": list(self._motor_names),
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (dof,),
                "names": list(self._motor_names),
            },
        }

    @property
    def camera_features(self) -> dict:
        cam_ft = {}
        for cam_key, cam in self.cameras.items():
            cam_ft[f"observation.images.{cam_key}"] = {
                "shape": (cam.height, cam.width, cam.channels),
                "names": ["height", "width", "channels"],
                "info": None,
            }
        return cam_ft

    @property
    def features(self) -> dict:
        return {**self.motor_features, **self.camera_features}

    @property
    def has_camera(self) -> bool:
        return len(self.cameras) > 0

    @property
    def num_cameras(self) -> int:
        return len(self.cameras)

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------
    def connect(self):
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                "FrankaRobot is already connected. Do not run `robot.connect()` twice."
            )

        for arm, ip, mouse in zip(self.config.arm_names, self.config.robot_ips, self.config.mouse_paths):
            reset_q = self._reset_q[arm]
            shared = _ArmShared(reset_q)
            self._shared[arm] = shared

            if self.mock:
                self._procs.append(
                    mp.Process(target=_mock_franka_proc, args=(shared, ip, self.config, reset_q), daemon=True)
                )
            else:
                self._procs.append(
                    mp.Process(target=_franka_proc, args=(shared, ip, self.config, reset_q), daemon=True)
                )
                self._procs.append(
                    mp.Process(target=_gripper_proc, args=(shared, ip, self.config), daemon=True)
                )
                # SpaceMouse only drives the arm during teleoperation.
                if self.config.control_mode == "teleop":
                    self._procs.append(
                        mp.Process(target=_spacemouse_proc, args=(shared, mouse), daemon=True)
                    )

        print(f"Starting {len(self._procs)} Franka control process(es)...")
        for p in self._procs:
            p.start()

        # Wait for every arm to finish its reset motion and enter control.
        print("Waiting for arm(s) to reach reset configuration...")
        for arm, shared in self._shared.items():
            while not shared.ready.value:
                if not any(p.is_alive() for p in self._procs):
                    raise RuntimeError(f"Franka control process for arm '{arm}' died during startup.")
                time.sleep(0.05)

        for name in self.cameras:
            self.cameras[name].connect()

        # Start F3 listener for teleop start / pause+re-home / resume.
        # The arm waits at reset until the first F3.
        if self.config.control_mode == "teleop":
            self._paused_state = True
            self._pause_listener = self._start_pause_listener()

        self.is_connected = True
        print("FrankaRobot connected.")

    def _start_pause_listener(self):
        """Listen for F3 in the main process and toggle the arm control state.

        This is designed to stay in phase with the ``record`` pipeline, which
        also binds F3 and alternates start/stop of each episode:

            F3 #1 -> start driving the arm  (episode recording starts)
            F3 #2 -> pause + re-home        (episode recording stops)
            F3 #3 -> start driving again    (next episode starts)

        The arm begins paused at the reset pose, so the very first F3 both
        starts control and starts recording.
        """
        try:
            from pynput import keyboard
        except Exception as e:  # noqa: BLE001
            print(f"[Franka] pynput unavailable ({e}); F3 control toggle disabled.")
            return None

        # Virtual-key codes for the left arrow across backends: X11/XKB (65361),
        # Windows (37), macOS (123). On some setups the arrow keys are not
        # delivered as the keyboard.Key.left enum but as a raw KeyCode carrying
        # only a vk, so the plain `key == keyboard.Key.left` comparison fails.
        LEFT_ARROW_VKS = {65361, 37, 123}

        def matches(key, target, vks=()):
            if key == target:
                return True
            if getattr(key, "name", None) == getattr(target, "name", None) and getattr(target, "name", None):
                return True
            vk = getattr(key, "vk", None)
            return vk is not None and vk in vks

        def _pause_and_rehome(reason):
            # Force the arm into the paused state (pause + re-home). Idempotent:
            # if already paused this is a no-op, so it stays in phase with the
            # recording pipeline (next F3 resumes from the reset pose).
            if not self._paused_state:
                self._paused_state = True
                for sh in self._shared.values():
                    sh.paused.value = True
                print(f"[Franka] {reason} -> PAUSE control + re-home.")

        def on_press(key):
            try:
                if matches(key, keyboard.Key.f3):
                    self._paused_state = not self._paused_state
                    for sh in self._shared.values():
                        sh.paused.value = self._paused_state
                    if self._paused_state:
                        print("[Franka] F3 -> PAUSE recording + re-home. Press F3 again to resume.")
                    else:
                        print("[Franka] F3 -> START/RESUME control + recording.")
                elif matches(key, keyboard.Key.left, LEFT_ARROW_VKS):
                    # Left arrow re-records the current episode in the pipeline;
                    # stop driving the arm so it doesn't keep moving, and stay in
                    # phase (the re-record waits for F3 again).
                    _pause_and_rehome("Left arrow (re-record)")
                elif matches(key, keyboard.Key.esc):
                    # Esc stops recording; stop driving the arm as well.
                    _pause_and_rehome("Esc (stop)")
            except Exception as e:  # noqa: BLE001
                print(f"[Franka] control-toggle listener error: {e}")

        listener = keyboard.Listener(on_press=on_press)
        listener.start()
        return listener

    def disconnect(self):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "FrankaRobot is not connected. Run `robot.connect()` before disconnecting."
            )

        if self._pause_listener is not None:
            self._pause_listener.stop()
            self._pause_listener = None
        self._paused_state = True

        for shared in self._shared.values():
            shared.is_running.value = False
        for p in self._procs:
            p.join(timeout=5.0)
            if p.is_alive():
                p.terminate()
        self._procs.clear()
        self._shared.clear()

        for name in self.cameras:
            self.cameras[name].disconnect()

        self.is_connected = False

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()

    # ------------------------------------------------------------------
    # State / action assembly
    # ------------------------------------------------------------------
    def _commanded_gripper_width(self, shared: _ArmShared) -> float:
        """Target width implied by the open/close toggle — the gripper *action*."""
        return self.config.gripper_open_width if shared.open_command.value else self.config.gripper_close_width

    def gripper_open_state(self, arm: str) -> bool:
        """Side ``arm``'s hand has actually reached (True = open).

        Unlike the width in ``observation.state`` this is a *settled* flag: it
        only flips once ``_gripper_proc`` finished the blocking move/grasp for
        the last ``send_action`` toggle, so a caller can wait on it to know the
        hand is done moving.
        """
        return bool(self._shared[arm].gripper_open_state.value)


    def _measured_gripper_width(self, shared: _ArmShared) -> float:
        """Finger width read back from the hardware by ``_gripper_proc``.

        Falls back to the commanded width when no live reading is available
        (mock mode, or the state reader could not talk to the gripper).
        """
        width = shared.measured_gripper_width.value
        if not np.isfinite(width):
            return self._commanded_gripper_width(shared)
        return float(width)

    def _read_joint_vector(self, kind: str) -> torch.Tensor:
        """Concatenate [7 joints + 1 gripper] per arm. kind in {'commanded', 'measured'}.

        The gripper channel follows `kind`: the commanded width is the recorded
        action, the measured finger width the honest `observation.state` reading.
        """
        parts: list[np.ndarray] = []
        for arm in self.arm_names:
            shared = self._shared[arm]
            src = shared.commanded_q if kind == "commanded" else shared.measured_q
            if kind == "commanded":
                gripper_width = self._commanded_gripper_width(shared)
            else:
                gripper_width = self._measured_gripper_width(shared)
            with src.get_lock():
                q = np.frombuffer(src.get_obj(), dtype=np.float32).copy()
            parts.append(np.concatenate([q, [gripper_width]], dtype=np.float32))
        return torch.from_numpy(np.concatenate(parts, dtype=np.float32))

    def _read_cameras(self) -> dict[str, torch.Tensor]:
        images = {}
        for name in self.cameras:
            before = time.perf_counter()
            img = self.cameras[name].async_read()
            # `async_read` returns a numpy view over a RealSense frame buffer that
            # librealsense recycles from a small pool. Without copying, every
            # recorded tensor would alias those few buffers and end up showing the
            # same (latest) frame — i.e. the images look "frozen". Copy to own it.
            images[name] = torch.from_numpy(np.ascontiguousarray(img).copy())
            self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
            self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before
        return images

    # ------------------------------------------------------------------
    # Core control interface
    # ------------------------------------------------------------------
    def teleop_step(
        self, record_data: bool = False
    ) -> None | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if not self.is_connected:
            raise RobotDeviceNotConnectedError("FrankaRobot is not connected. Run `robot.connect()`.")

        # The SpaceMouse -> Franka loop runs autonomously in the background; here
        # we only sample the latest published joints.
        if not record_data:
            return None

        before = time.perf_counter()
        # Action = the joint signal commanded to the executor (q_d + gripper).
        action = self._read_joint_vector("commanded")
        # Observation state = measured joints (+ gripper).
        state = self._read_joint_vector("measured")
        self.logs["read_pos_dt_s"] = time.perf_counter() - before

        images = self._read_cameras()

        obs_dict: dict[str, torch.Tensor] = {"observation.state": state}
        for name in images:
            obs_dict[f"observation.images.{name}"] = images[name]
        action_dict: dict[str, torch.Tensor] = {"action": action}
        return obs_dict, action_dict

    def capture_observation(self) -> dict[str, torch.Tensor]:
        if not self.is_connected:
            raise RobotDeviceNotConnectedError("FrankaRobot is not connected. Run `robot.connect()`.")

        state = self._read_joint_vector("measured")
        images = self._read_cameras()
        obs_dict: dict[str, torch.Tensor] = {"observation.state": state}
        for name in images:
            obs_dict[f"observation.images.{name}"] = images[name]
        return obs_dict

    def send_action(self, action: torch.Tensor) -> torch.Tensor:
        """Push a joint-position action (replay mode). Layout matches `action` features:
        per arm [7 joints + 1 gripper]."""
        if not self.is_connected:
            raise RobotDeviceNotConnectedError("FrankaRobot is not connected. Run `robot.connect()`.")
        if self.config.control_mode != "replay":
            raise RuntimeError("send_action requires control_mode='replay'.")

        goal = action.detach().cpu().to(torch.float32)
        if self.config.max_relative_target is not None:
            present = self._read_joint_vector("measured")
            goal = ensure_safe_goal_position(goal, present, self.config.max_relative_target)

        goal_np = goal.numpy()
        for i, arm in enumerate(self.arm_names):
            base = i * ARM_TOTAL_DOF
            arm_q = goal_np[base : base + ARM_DOF]
            gripper = goal_np[base + ARM_DOF]
            shared = self._shared[arm]
            with shared.target_q.get_lock():
                np.frombuffer(shared.target_q.get_obj(), dtype=np.float32)[:] = arm_q.astype(np.float32)
            # Threshold gripper width to an open/close toggle.
            mid = 0.5 * (self.config.gripper_open_width + self.config.gripper_close_width)
            shared.open_command.value = bool(gripper >= mid)
        return goal

    def print_logs(self):
        pass