#!/usr/bin/env python3
"""Run a remote FastWAM policy on the Franka FR3 arm.

This is the robot-side half of a two-process deployment:

    Franka arm (this repo)        <--TCP-->   FastWAM model server
    franka_fastwam_client.py                    (fwam repo)

The client captures observations from the arm (``robot.capture_observation()``),
streams them to the FastWAM policy server, receives a joint-position action and
executes it with ``robot.send_action(...)`` (replay mode's 1 kHz tracker).

Key behaviours:

* The server only re-infers when its internal action queue is drained (a
  "replan" boundary). To avoid shipping ~5 MB of camera frames every control
  step, the client first asks ``should_request_observation``: the full
  image+state payload is sent only on replan steps, an empty poke otherwise.
* TOPP handshake: on startup the client reports its control period and the
  arm's joint velocity/acceleration limits (``configure``), so the server can
  re-time each predicted chunk under those limits before handing out actions.
  ``--no-topp`` keeps the raw-chunk behaviour.
* 1 kHz interpolation: the main loop runs at ``--fps`` (e.g. 15 Hz), but every
  pair of consecutive actions is linearly interpolated into a 1 kHz sequence so
  the arm's 1 kHz control cycle always sees a smooth target instead of 15 Hz
  jumps.
* Before the rollout, F3 records a high-view human demo (cameras only, the arm
  parks at the reset pose) and ships it to the server as the ``video_ctx``
  condition; F2 keeps the server's startup default. In a headless environment
  the server default is used and Ctrl-C stops the loop.

Wire protocol: length-prefixed pickle (must match the FastWAM server).

Example (server on the same machine, mock arm + cameras for a bench run):
    PYTHONPATH="$(pwd)" python lerobot/scripts/franka_fastwam_client.py \
        --host 127.0.0.1 --port 5555 \
        --task "Pick up the block" --mock --control-time-s 10
"""

import argparse
import io
import logging
import pickle
import signal
import socket
import struct
import time
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from lerobot.common.robot_devices.robots.configs import FrankaRobotConfig
from lerobot.common.robot_devices.robots.frankarobot import (
    ARM_DOF,
    ARM_TOTAL_DOF,
    REPLAY_MAX_JOINT_VEL,
)
from lerobot.common.robot_devices.robots.utils import make_robot_from_config
from lerobot.common.robot_devices.utils import busy_wait

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("franka_client")

OBS_STATE_KEY = "observation.state"
OBS_IMAGE_PREFIX = "observation.images."
# Seconds to idle before the video_ctx prompt / after it, so the hardware settles.
WARMUP_S = 5.0


# ---------------------------------------------------------------------------
# TCP helpers (length-prefixed pickle) -- kept in sync with the FastWAM server.
# ---------------------------------------------------------------------------
def _send_msg(sock: socket.socket, obj: Any) -> None:
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(struct.pack(">I", len(data)) + data)


def _recv_exact(sock: socket.socket, n: int) -> Optional[bytes]:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


class _NumpyCompatUnpickler(pickle.Unpickler):
    """Load pickles across the numpy 1.x / 2.x boundary.

    numpy 2 moved its internals from ``numpy.core`` to ``numpy._core``, and the
    array reducers it pickles name the *new* path -- so a numpy-1 process fails
    to load a numpy-2 array with ``No module named 'numpy._core.numeric'``.
    Client and server are separate machines with separate envs here, so retry
    the other spelling instead of forcing both to pin the same numpy major.
    """

    _ALIASES = (("numpy._core", "numpy.core"), ("numpy.core", "numpy._core"))

    def find_class(self, module: str, name: str) -> Any:
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError):
            for old, new in self._ALIASES:
                if module == old or module.startswith(old + "."):
                    return super().find_class(new + module[len(old):], name)
            raise


def _loads(payload: bytes) -> Any:
    return _NumpyCompatUnpickler(io.BytesIO(payload)).load()


def _recv_msg(sock: socket.socket) -> Any:
    header = _recv_exact(sock, 4)
    if header is None:
        return None
    length = struct.unpack(">I", header)[0]
    payload = _recv_exact(sock, length)
    if payload is None:
        return None
    return _loads(payload)


class FastWAMPolicyClient:
    """Thin TCP client mirroring the FastWAM policy server's command interface."""

    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((host, port))
        logger.info("Connected to FastWAM policy server at %s:%d", host, port)

    def _request(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        _send_msg(self.sock, msg)
        response = _recv_msg(self.sock)
        if response is None:
            raise ConnectionError("Server closed the connection unexpectedly.")
        if response.get("status") == "error":
            raise RuntimeError(
                f"Server error: {response.get('message')}\n{response.get('traceback', '')}"
            )
        return response

    def should_request_observation(self) -> bool:
        return bool(self._request({"cmd": "should_request_observation"})["result"])

    def configure(
        self,
        fps: float,
        max_vel: np.ndarray,
        max_acc: np.ndarray,
        gripper_dims: Sequence[int],
        enable: bool = True,
    ) -> Tuple[bool, str]:
        """Report our control period + the robot's joint limits to the server.

        This is what turns on the server's TOPP re-timing. The re-timing lives
        there rather than here because the server predicts a whole chunk per
        replan and only hands out ``replan_steps`` of them: it is the only side
        that holds a *path* to parameterize, so it can bound velocity and
        acceleration without the lookahead delay a client-side filter would
        need. What it cannot know is the robot's limits or our control rate,
        hence this handshake.

        Returns ``(topp_active, detail)``. A server that predates the command --
        or one started with ``--no-topp`` -- is not an error: the rollout then
        runs on raw chunks exactly as before, so report it and carry on.
        """
        try:
            resp = self._request({
                "cmd": "configure",
                "fps": float(fps),
                "max_vel": np.asarray(max_vel, dtype=np.float32),
                "max_acc": np.asarray(max_acc, dtype=np.float32),
                "gripper_dims": [int(i) for i in gripper_dims],
                "topp": bool(enable),
            })
        except RuntimeError as exc:  # reported by the server, connection is fine
            return False, f"server rejected `configure` ({exc})"
        return bool(resp.get("topp", False)), str(resp.get("detail", ""))

    def step(
        self,
        instruction: str,
        state: Optional[np.ndarray],
        images: Optional[Dict[str, np.ndarray]],
    ) -> Optional[np.ndarray]:
        msg: Dict[str, Any] = {"cmd": "step", "instruction": instruction}
        if state is not None and images is not None:
            msg["state"] = state
            msg["images"] = images
        return self._request(msg).get("action")

    def reset(self) -> None:
        self._request({"cmd": "reset"})

    def set_video_ctx(self, frames: Optional[np.ndarray]) -> bool:
        """Set the server's video_ctx global condition.

        ``frames`` None -> the server restores the default demo it loaded at
        startup (--video-ctx-path). ``frames`` a ``[T, H, W, 3]`` uint8 RGB
        array -> the server builds a fresh condition from this demo. Returns
        whether the model actually consumes a video_ctx condition.
        """
        resp = self._request({"cmd": "set_video_ctx", "frames": frames})
        return bool(resp.get("has_video_ctx", False))

    def get_video_ctx_info(self) -> Dict[str, Any]:
        """Query the server's video_ctx geometry (num_frames it downsamples to)."""
        return self._request({"cmd": "get_video_ctx_info"})

    def close(self) -> None:
        try:
            self._request({"cmd": "shutdown"})
        except Exception:  # noqa: BLE001 - best-effort teardown
            pass
        try:
            self.sock.close()
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# TOPP limit payload
# ---------------------------------------------------------------------------
def build_topp_limits(
    n_arms: int,
    arm_dof: int,
    max_vel,
    max_acc,
) -> Tuple[np.ndarray, np.ndarray, list]:
    """Per-dimension TOPP limits for an ``[arm_dof joints + 1 gripper] x n_arms`` action.

    ``max_vel`` / ``max_acc`` are either a scalar (one cap for every joint) or
    one value per arm joint. Returns ``(max_vel, max_acc, gripper_dims)`` over
    the full action width. The gripper slots are filled in only to keep the
    arrays rectangular -- the server replaces them, since a gripper width
    shares no units with a joint; what it needs is ``gripper_dims`` telling it
    which entries those are.
    """
    def per_joint(values, name: str) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        if arr.size == 1:
            arr = np.full(arm_dof, float(arr[0]))
        if arr.size != arm_dof:
            raise ValueError(
                f"{name} must be a scalar or {arm_dof} per-joint values, got {arr.size}"
            )
        if not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
            raise ValueError(f"{name} entries must all be finite and > 0: {arr.tolist()}")
        return arr

    vel_arm = per_joint(max_vel, "max_vel")
    acc_arm = per_joint(max_acc, "max_acc")
    total_dof = arm_dof + 1
    vel = np.tile(np.append(vel_arm, vel_arm.max()), n_arms)
    acc = np.tile(np.append(acc_arm, acc_arm.max()), n_arms)
    gripper_dims = [i * total_dof + arm_dof for i in range(n_arms)]
    return vel.astype(np.float32), acc.astype(np.float32), gripper_dims


def configure_topp(
    client: "FastWAMPolicyClient",
    fps: float,
    n_arms: int,
    arm_dof: int,
    max_vel,
    max_acc,
    enable: bool = True,
) -> bool:
    """Build the limit payload, hand it to the server and log what came back."""
    vel, acc, gripper_dims = build_topp_limits(n_arms, arm_dof, max_vel, max_acc)
    active, detail = client.configure(fps, vel, acc, gripper_dims, enable=enable)
    if active:
        logger.info("Server TOPP re-timing ACTIVE | %s", detail)
    else:
        logger.warning(
            "Server TOPP re-timing INACTIVE (%s): the raw predicted chunk will be "
            "executed one waypoint per tick, unsmoothed.", detail,
        )
    return active


def observation_to_payload(observation: Dict[str, torch.Tensor]):
    """Split an observation into (state ndarray, {cam: image ndarray}).

    ``capture_observation`` returns ``observation.state`` (torch float32) and
    ``observation.images.<cam>`` (torch uint8, HWC RGB). We hand the raw values
    to the server, which owns all model-specific preprocessing.
    """
    state = observation[OBS_STATE_KEY].cpu().numpy().astype(np.float32)
    images: Dict[str, np.ndarray] = {}
    for key, value in observation.items():
        if key.startswith(OBS_IMAGE_PREFIX):
            cam = key[len(OBS_IMAGE_PREFIX):]
            images[cam] = value.cpu().numpy()
    return state, images


def downsample_frames(frames: np.ndarray, target: int) -> np.ndarray:
    """Uniformly subsample ``frames`` ([T, ...]) down to ``target`` frames.

    Matches the server's ``np.linspace`` uniform downsample so we ship only the
    frames the model will actually keep. A no-op when ``target`` <= 0 or the clip
    is already short enough (the server pads short clips).
    """
    if target <= 0 or frames.shape[0] <= target:
        return frames
    idx = np.linspace(0, frames.shape[0] - 1, target, dtype=int)
    return frames[idx]


# ---------------------------------------------------------------------------
# Keyboard (single listener for both the video_ctx choice and the rollout).
# ---------------------------------------------------------------------------
def init_keyboard(stop: Dict[str, bool]):
    """Start a pynput listener for F2 / F3 / Esc.

    Returns ``(listener, events)``. ``events`` carries boolean ``"f2"`` /
    ``"f3"`` flags that the callers consume (reset to False after handling);
    Esc flips ``stop["flag"]`` directly. ``listener`` is None in a headless
    environment without pynput -- the flags then stay False, so the caller must
    fall back to the server default condition and rely on Ctrl-C to stop.
    """
    events = {"f2": False, "f3": False}
    try:
        from pynput import keyboard
    except Exception as e:  # noqa: BLE001 - headless / no display
        logger.warning("pynput unavailable (%s): F2/F3/Esc disabled; use Ctrl-C to stop.", e)
        return None, events

    def on_press(key):
        if key == keyboard.Key.f2:
            print("F2 pressed.")
            events["f2"] = True
        elif key == keyboard.Key.f3:
            print("F3 pressed.")
            events["f3"] = True
        elif key == keyboard.Key.esc:
            print("Esc pressed -> stopping.")
            stop["flag"] = True

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    return listener, events


def auto_press_f3(events: Dict[str, bool], stop: Dict[str, bool], record_seconds: float,
                  initial_delay: float = 2.0) -> None:
    """Testing hook: emulate two F3 presses from a background thread.

    The first press (after ``initial_delay``) releases `configure_video_ctx`'s
    wait and starts the demo recording; the second, ``record_seconds`` later,
    stops it -- so the recorded video_ctx demo is ``record_seconds`` long. It
    goes through the same ``events`` flags as the pynput listener, and the
    second press is consumed by the recording loop itself.
    """
    import threading

    def _run():
        time.sleep(initial_delay)
        if stop["flag"]:
            return
        print("F3 pressed (simulated) -> start video_ctx recording.")
        events["f3"] = True
        time.sleep(record_seconds)
        print("F3 pressed (simulated) -> stop video_ctx recording.")
        events["f3"] = True

    threading.Thread(target=_run, daemon=True).start()


# ---------------------------------------------------------------------------
# video_ctx demo capture
# ---------------------------------------------------------------------------
def record_high_view_demo(robot, events, stop, high_cam_key: str, fps: int) -> Optional[np.ndarray]:
    """Record the ``high_cam_key`` view until F3 is pressed again.

    Only the high camera is read (not a full ``capture_observation``) to keep the
    loop cheap. The arm is *not* driven: in replay mode it parks at the reset
    pose, so this captures a human performing the task in front of it. Returns a
    ``[T, H, W, 3]`` uint8 RGB array, or None if nothing was captured.
    """
    if high_cam_key not in robot.cameras:
        raise KeyError(
            f"high camera '{high_cam_key}' not configured; available: {list(robot.cameras)}"
        )
    camera = robot.cameras[high_cam_key]
    logger.info("F3: demo recording started ('%s') -- press F3 again to stop.", high_cam_key)
    events["f3"] = False
    frames = []
    period = (1.0 / fps) if fps else 0.0
    while not events["f3"] and not stop["flag"]:
        loop_start = time.perf_counter()
        # async_read hands back a view over a recycled RealSense buffer; copy it
        # so every stored frame owns its pixels instead of aliasing the latest.
        frames.append(np.ascontiguousarray(camera.async_read()).astype(np.uint8).copy())
        if fps:
            busy_wait(period - (time.perf_counter() - loop_start))
    events["f3"] = False
    logger.info("F3: recording stopped (%d frames).", len(frames))
    return np.stack(frames, axis=0) if frames else None


def configure_video_ctx(client, robot, args, events, stop) -> None:
    """Block for F2/F3 to choose the server's video_ctx condition, then send it."""
    if events is None:
        client.set_video_ctx(None)  # keep server default
        return
    logger.info(
        "Press F3 to record a video_ctx demo (%s view), or F2 to use the server's "
        "startup-default condition.",
        args.high_cam_key,
    )
    while not (events["f2"] or events["f3"] or stop["flag"]):
        time.sleep(0.05)
    if stop["flag"]:
        return
    if events["f3"]:
        frames = record_high_view_demo(robot, events, stop, args.high_cam_key, args.fps)
        # Downsample to exactly what the server keeps before shipping it.
        target = int(client.get_video_ctx_info().get("num_frames", 0) or 0)
        if frames is not None:
            n0 = frames.shape[0]
            frames = downsample_frames(frames, target)
            logger.info(
                "video_ctx demo: %d frames -> %d (server keeps %d).", n0, frames.shape[0], target
            )
        logger.info(
            "Sending video_ctx demo to server: %s",
            None if frames is None else tuple(frames.shape),
        )
        logger.info("Server video_ctx active: %s", client.set_video_ctx(frames))
    else:  # F2
        events["f2"] = False
        logger.info("F2: using server default video_ctx condition.")
        client.set_video_ctx(None)


# ---------------------------------------------------------------------------
# Robot construction / action checking
# ---------------------------------------------------------------------------
def build_robot(args) -> Any:
    """Instantiate a FrankaRobot in replay mode (the only mode send_action allows)."""
    kwargs: Dict[str, Any] = {"control_mode": "replay", "mock": args.mock}
    # Lift the speed limit: the default 0.08 (8% speed) would throttle the
    # replay tracker far below REPLAY_MAX_JOINT_VEL (~2.0-2.5 rad/s). At 1.0 it
    # tracks the interpolated targets at full speed.
    kwargs["reset_speed_factor"] = 1.0
    if args.arm_names:
        n = len(args.arm_names)
        if not args.robot_ips or len(args.robot_ips) != n:
            raise ValueError("--robot-ips must list one ip per --arm-names entry.")
        kwargs["arm_names"] = list(args.arm_names)
        kwargs["robot_ips"] = list(args.robot_ips)
        # SpaceMouse paths are unused in replay mode, but the config validates
        # that the three per-arm lists have the same length.
        kwargs["mouse_paths"] = ["" for _ in range(n)]
    if args.max_relative_target is not None:
        kwargs["max_relative_target"] = args.max_relative_target
    if args.skip_gripper_homing:
        kwargs["skip_gripper_homing"] = True
    return make_robot_from_config(FrankaRobotConfig(**kwargs))


def check_action(action: np.ndarray, expected_dof: int, arm_names) -> np.ndarray:
    """Validate the server's action against the robot's action layout."""
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.shape[0] != expected_dof:
        raise ValueError(
            f"Server returned a {action.shape[0]}-dim action but this robot expects "
            f"{expected_dof}: {len(arm_names)} arm(s) {list(arm_names)} x "
            f"[{ARM_DOF} joints + 1 gripper width]. Check that the server's "
            "checkpoint matches the --arm-names configuration."
        )
    return action


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a remote FastWAM policy on the Franka FR3 arm")
    p.add_argument("--host", default="127.0.0.1", help="FastWAM server address (default: loopback).")
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--task", required=True, help="Task / instruction text sent to the policy.")
    p.add_argument("--fps", type=int, default=15, help="Control loop frequency.")
    p.add_argument("--control-time-s", type=float, default=None,
                   help="Stop after this many seconds (default: run until Esc / Ctrl-C).")
    p.add_argument("--arm-names", nargs="+", default=None,
                   help="Arms to drive, e.g. `--arm-names left` for a single arm "
                        "(default: the config's arms). Requires a matching --robot-ips.")
    p.add_argument("--robot-ips", nargs="+", default=None,
                   help="One Franka ip per --arm-names entry.")
    p.add_argument("--max-relative-target", type=float, default=None,
                   help="Per-step joint goal clamp (radians) applied in send_action. NOTE: it "
                        "clamps the gripper-width entry too, so a small value slows opening.")
    p.add_argument("--skip-gripper-homing", action="store_true",
                   help="Skip the gripper homing() sweep at startup (hand must be calibrated).")
    p.add_argument("--high-cam-key", default="cam_high",
                   help="Camera whose view is recorded for the video_ctx demo (default: cam_high).")
    p.add_argument("--simulate-video-ctx", type=float, default=0.0, metavar="SECONDS",
                   help="Testing hook: emulate pressing F3 twice (start, then stop after "
                        "SECONDS) so a video_ctx demo is recorded without a keyboard. "
                        "0 disables (default).")
    p.add_argument("--mock", action="store_true", help="Use a mock arm + cameras (no hardware).")
    # TOPP: the limits reported to the server default to the ones the 1 kHz replay
    # tracker itself enforces (from FrankaRobotConfig), so the two agree.
    p.add_argument("--no-topp", action="store_true",
                   help="Do not ask the server to TOPP-re-time its chunks; send the raw "
                        "predicted actions and let the replay tracker chase them.")
    p.add_argument("--topp-max-vel", type=float, nargs="+", default=None,
                   help="Override the per-joint velocity limit (rad/s) reported for the "
                        f"re-timing: one value or {ARM_DOF} (default: the config's "
                        "REPLAY_MAX_JOINT_VEL * reset_speed_factor).")
    p.add_argument("--topp-max-acc", type=float, nargs="+", default=None,
                   help="Override the per-joint acceleration limit (rad/s^2) reported for "
                        f"the re-timing: one value or {ARM_DOF} (default: the config's "
                        "replay_joint_acceleration).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    robot = build_robot(args)
    if not robot.is_connected:
        robot.connect()  # homes every arm, then starts the 1 kHz replay loop

    expected_dof = len(robot.arm_names) * ARM_TOTAL_DOF
    # No reset() here: the server already starts a fresh episode when it accepts
    # the connection, so an extra one only bumps its episode counter twice.
    client = FastWAMPolicyClient(args.host, args.port)

    # Tell the server our control period and the arm's joint limits so it can
    # TOPP-re-time each predicted chunk before handing out actions. Default to the
    # very caps `_franka_replay_loop` enforces, so the server plans against the
    # limits the tracker will actually apply -- a target the tracker has to ramp
    # into is exactly the jump we are trying to remove.
    configure_topp(
        client,
        fps=args.fps,
        n_arms=len(robot.arm_names),
        arm_dof=ARM_DOF,
        max_vel=(
            args.topp_max_vel if args.topp_max_vel is not None
            else REPLAY_MAX_JOINT_VEL * robot.config.reset_speed_factor
        ),
        max_acc=(
            args.topp_max_acc if args.topp_max_acc is not None
            else robot.config.replay_joint_acceleration
        ),
        enable=not args.no_topp,
    )

    stop = {"flag": False}

    def _handle_sigint(signum, frame):  # noqa: ANN001, ARG001
        logger.info("Interrupt received, stopping control loop...")
        stop["flag"] = True

    signal.signal(signal.SIGINT, _handle_sigint)

    listener, events = init_keyboard(stop)
    if args.simulate_video_ctx > 0:
        auto_press_f3(events, stop, args.simulate_video_ctx)

    # WARM-UP: idle for a couple of seconds so the camera streams settle and the
    # first observation is not a stale frame.
    warmup_start = time.perf_counter()
    while not stop["flag"] and (time.perf_counter() - warmup_start) < WARMUP_S:
        loop_start = time.perf_counter()
        robot.capture_observation()
        if args.fps:
            busy_wait(1 / args.fps - (time.perf_counter() - loop_start))

    use_events = listener is not None or args.simulate_video_ctx > 0
    configure_video_ctx(client, robot, args, events if use_events else None, stop)

    # Give the scene a moment to settle after the video_ctx prompt before the
    # first rollout observation is captured.
    time.sleep(WARMUP_S)

    control_time_s = float("inf") if args.control_time_s is None else args.control_time_s
    start_t = time.perf_counter()
    step_idx = 0
    paused = False
    prev_action = None  # previous action, for interpolation between consecutive actions
    try:
        while not stop["flag"] and (time.perf_counter() - start_t) < control_time_s:
            loop_start = time.perf_counter()

            if events["f3"]:
                events["f3"] = False
                paused = not paused
                if paused:
                    logger.info("F3 -> PAUSED (arm holds its last target). Press F3 to resume.")
                else:
                    # Drop the server's queued actions so we resume on a fresh
                    # plan instead of replaying a chunk from before the pause.
                    client.reset()
                    # Drop the previous action so we do not interpolate from a stale target
                    prev_action = None
                    logger.info("F3 -> RESUMED (server replanning).")
            if paused:
                time.sleep(0.05)
                continue

            # Keep the camera buffers fresh every step; only serialize + ship the
            # heavy image payload on replan steps (when the server needs it).
            observation = robot.capture_observation()
            if client.should_request_observation():
                state, images = observation_to_payload(observation)
                action = client.step(args.task, state, images)
            else:
                action = client.step(args.task, None, None)

            if action is None:
                logger.warning("Server returned no action at step %d; skipping.", step_idx)
                # Keep the --fps pacing even when the server returns no action
                if args.fps:
                    busy_wait(1 / args.fps - (time.perf_counter() - loop_start))
            else:
                action = check_action(action, expected_dof, robot.arm_names)

                if prev_action is None:
                    # First action (or the first after a pause): nothing to interpolate from
                    robot.send_action(torch.from_numpy(action))
                else:
                    # Linearly interpolate the pair [prev_action, action] into a
                    # 1 kHz sequence so the arm's 1 kHz control cycle always has a
                    # smooth target instead of 15 Hz jumps. A pair spans ONE
                    # interval, so the substep count is 1000/fps + 1 (~68 @15 Hz),
                    # not the whole-trajectory old*1000/fps used in replay mode.
                    n_sub = round(1000 / args.fps) + 1                 # substeps between the pair
                    t = np.linspace(0.0, 1.0, n_sub)[:, None]          # [n_sub, 1]
                    sub_actions = prev_action[None, :] + (action - prev_action)[None, :] * t
                    for a in sub_actions:
                        t0 = time.perf_counter()
                        robot.send_action(torch.from_numpy(a.astype(np.float32)))
                        busy_wait(1 / 1000 - (time.perf_counter() - t0))  # enforce 1 kHz

                prev_action = action

            step_idx += 1
    finally:
        if listener is not None:
            listener.stop()
        client.close()
        if robot.is_connected:
            robot.disconnect()
        logger.info("Stopped after %d steps.", step_idx)


if __name__ == "__main__":
    main()
