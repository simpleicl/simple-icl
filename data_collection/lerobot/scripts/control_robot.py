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
"""
Utilities to teleoperate a robot and record a teleoperation dataset.

Examples of usage:

- Unlimited teleoperation at a limited frequency of 30 Hz, with the 3D SpaceMouse driving the arm:
```bash
PYTHONPATH="$(pwd)" python lerobot/scripts/control_robot.py \
    --robot.type=franka \
    --control.type=teleoperate \
    --control.fps=30
```

- Record a paired teleoperation dataset following the plan.csv collection plan,
with 2 seconds of warmup and 60 seconds of recording per episode:
```bash
PYTHONPATH="$(pwd)" python lerobot/scripts/control_robot.py \
    --robot.type=franka \
    --control.type=record \
    --control.fps=30 \
    --control.single_task="Pick up bananas and put into the plate." \
    --control.repo_id=brave/bananas \
    --control.root=outputs/1015 \
    --control.warmup_time_s=2 \
    --control.episode_time_s=60 \
    --control.reset_time_s=2 \
    --control.num_episodes=10 \
    --control.push_to_hub=false \
    --control.video=false
```

**NOTE**: You can use your keyboard to control data recording flow.
- Tap F3 key to early exit while recording an episode and go to resetting the environment.
- Tap left arrow key '<-' to early exit and re-record the current episode. The rejected episode is not
  deleted: it is moved into a separate "failure" LeRobot dataset (see `--control.failure_root`, which
  defaults to '<root>_failure'). Pass `--control.save_failure_episodes=false` to delete it instead.
- Tap escape key 'esc' to stop the data recording.
This might require a sudo permission to allow your terminal to monitor keyboard events.
"""

import logging
import os, shutil
import traceback
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from pprint import pformat

import rerun as rr

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.robot_devices.control_configs import (
    ControlConfig,
    ControlPipelineConfig,
    RecordControlConfig,
    TeleoperateControlConfig,
)
from lerobot.common.robot_devices.control_utils import (
    control_loop,
    init_keyboard_listener,
    is_headless,
    read_plan_row,
    record_episode,
    reset_environment,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
    stop_recording,
    warmup_record,
    write_plan_data,
)
from lerobot.common.robot_devices.robots.utils import Robot, make_robot_from_config
from lerobot.common.robot_devices.utils import safe_disconnect
from lerobot.common.utils.utils import has_method, init_logging, log_say
from lerobot.configs import parser

########################################################################################
# Duplicate first-frame guard
########################################################################################


def _load_episode_first_frame(dataset: LeRobotDataset, ep_idx: int, cam_key: str):
    """Return the first frame (HxWx3 uint8 RGB ndarray) of `ep_idx` for `cam_key`.

    Handles both storage layouts: PNG frames under images/, or the encoded mp4
    under videos/ (first frame decoded via PyAV).
    """
    import numpy as np

    if cam_key in dataset.meta.video_keys:
        import av

        vid_path = dataset.root / dataset.meta.get_video_file_path(ep_idx, cam_key)
        if not vid_path.exists():
            return None
        with av.open(str(vid_path)) as container:
            for frame in container.decode(video=0):
                return frame.to_ndarray(format="rgb24")
        return None
    else:
        from PIL import Image

        img_path = dataset._get_image_file_path(ep_idx, cam_key, 0)
        if not img_path.exists():
            return None
        return np.asarray(Image.open(img_path).convert("RGB"))


def _load_current_episode_first_frame(dataset: LeRobotDataset, cam_key: str):
    """Return the first frame (HxWx3 uint8 RGB ndarray) of the current, not-yet-saved
    episode held in `dataset.episode_buffer` for `cam_key`.

    Even in video mode the raw frames are written as PNGs to a temporary images
    directory during `add_frame`; the buffer stores their paths, so we can read the
    first one before `save_episode` encodes/moves anything.
    """
    import numpy as np
    from PIL import Image

    buffer = dataset.episode_buffer
    if not buffer or cam_key not in buffer or len(buffer[cam_key]) == 0:
        return None

    first = buffer[cam_key][0]
    if not isinstance(first, str):
        return None
    if not os.path.exists(first):
        return None
    return np.asarray(Image.open(first).convert("RGB"))


def check_current_episode_duplicate(dataset: LeRobotDataset, threshold: float) -> list[str]:
    """Compare the current (in-buffer, not-yet-saved) episode's first frame against the
    previous episode's first frame for every camera key.

    A mean absolute pixel difference (0-255 scale) below `threshold` means the camera
    feed was likely frozen/duplicated. Returns the list of human-readable descriptions
    of duplicate cameras (empty if none / when disabled via threshold <= 0).
    """
    import numpy as np

    if threshold is None or threshold <= 0:
        return []

    cams = dataset.meta.camera_keys
    if not cams:
        return []

    cur_ep = dataset.num_episodes  # index the current buffer will be saved under
    prev_ep = cur_ep - 1
    if prev_ep < 0:
        return []  # nothing recorded before this to compare against

    duplicates = []
    for cam in cams:
        cur = _load_current_episode_first_frame(dataset, cam)
        prev = _load_episode_first_frame(dataset, prev_ep, cam)
        if cur is None or prev is None:
            logging.warning(
                f"[dup-check] could not load first frames for episodes "
                f"{prev_ep}/{cur_ep} on '{cam}'; skipping this pair."
            )
            continue
        if cur.shape != prev.shape:
            continue  # different resolution -> not a duplicate
        diff = float(np.abs(cur.astype(np.float32) - prev.astype(np.float32)).mean())
        logging.info(f"[dup-check] episodes {prev_ep}/{cur_ep} '{cam}' mean abs diff={diff:.4f}")
        if diff < threshold:
            duplicates.append(
                f"episode {cur_ep:06d} vs {prev_ep:06d} on '{cam}' "
                f"(mean abs diff={diff:.4f} < {threshold})"
            )
    return duplicates


########################################################################################
# Failed (re-recorded) episodes -> separate "failure" dataset
########################################################################################


def _resolve_failure_paths(dataset: LeRobotDataset, cfg: RecordControlConfig) -> tuple[str, str]:
    """Root and repo_id of the failure dataset, defaulting to '<root>_failure' / '<repo_id>_failure'."""
    if cfg.failure_root is not None:
        failure_root = str(cfg.failure_root)
    else:
        failure_root = str(dataset.root.parent / f"{dataset.root.name}_failure")
    failure_repo_id = cfg.failure_repo_id if cfg.failure_repo_id is not None else f"{cfg.repo_id}_failure"
    return failure_root, failure_repo_id


def make_failure_dataset(dataset: LeRobotDataset, robot: Robot, cfg: RecordControlConfig) -> LeRobotDataset:
    """Open the dataset collecting the rejected episodes, creating it on first use.

    It is a standalone LeRobotDataset with the same features/fps as the main one, so it can be
    postprocessed (convert_to_video.py, split_dataset.py, ...) exactly like a normal recording.
    """
    failure_root, failure_repo_id = _resolve_failure_paths(dataset, cfg)
    root_path = Path(failure_root)

    if (root_path / "meta" / "info.json").is_file():
        failure_dataset = LeRobotDataset(failure_repo_id, root=failure_root)
        sanity_check_dataset_robot_compatibility(failure_dataset, robot, cfg.fps, cfg.video)
        logging.info(
            f"[failure] appending to existing failure dataset '{failure_root}' "
            f"({failure_dataset.num_episodes} episodes already stored)."
        )
        return failure_dataset

    if root_path.exists():
        if any(root_path.iterdir()):
            raise RuntimeError(
                f"Failure dataset directory '{failure_root}' already exists but holds no "
                f"meta/info.json. Please remove it or point --control.failure_root elsewhere."
            )
        root_path.rmdir()  # LeRobotDatasetMetadata.create insists on creating the root itself

    failure_dataset = LeRobotDataset.create(
        failure_repo_id,
        cfg.fps,
        root=failure_root,
        robot=robot,
        use_videos=cfg.video,
    )
    logging.info(f"[failure] created failure dataset at '{failure_root}'.")
    return failure_dataset


def _log_failure_episode(
    failure_dataset: LeRobotDataset,
    dataset: LeRobotDataset,
    src_ep_idx: int,
    dst_ep_idx: int,
    num_frames: int,
    plan_row: dict | None,
) -> None:
    """Append one line per stashed episode under <failure root>/logs, to keep the provenance."""
    logs_dir = failure_dataset.root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    plan = ""
    if plan_row is not None:
        plan = f" plan={plan_row.get('scene', '')}/{plan_row.get('clutter', '')}/{plan_row.get('action', '')}"
    with open(logs_dir / "failure_episodes.log", "a") as f:
        f.write(
            f"[{timestamp}] rejected episode {src_ep_idx:06d} of '{dataset.root}' "
            f"stored as episode {dst_ep_idx:06d} ({num_frames} frames).{plan}\n"
        )


def discard_episode_frames(dataset: LeRobotDataset, ep_idx: int) -> None:
    """Delete the recorded frames of episode `ep_idx` and reset the buffer.

    Same intent as `dataset.clear_episode_buffer()`, but the episode index is passed explicitly
    instead of read from the buffer, so it also works after a failed/partial transplant (which
    leaves the buffer in an intermediate state). Leftover frame directories must not survive: the
    next episode reuses that index and video encoding would pick up the stale frames.
    """
    dataset._wait_image_writer()
    for cam_key in dataset.meta.camera_keys:
        img_dir = dataset._get_image_file_path(
            episode_index=ep_idx, image_key=cam_key, frame_index=0
        ).parent
        if img_dir.is_dir():
            shutil.rmtree(img_dir)
    dataset.episode_buffer = dataset.create_episode_buffer()


def stash_failed_episode(
    dataset: LeRobotDataset,
    failure_dataset: LeRobotDataset | None,
    robot: Robot,
    cfg: RecordControlConfig,
    plan_row: dict | None = None,
) -> LeRobotDataset | None:
    """Move the current (in-buffer, not-yet-saved) episode into the failure dataset.

    The frames of the episode being recorded already live on disk as PNGs under
    '<main root>/images/<cam key>/episode_<main idx>/'; the buffer only holds their paths. So the
    episode is transplanted by moving those directories under the failure dataset's own episode
    index, repointing the buffer at the new paths, and letting the failure dataset save it as its
    next episode. Returns the failure dataset (created on first call).
    """
    buffer = dataset.episode_buffer
    if not buffer or buffer.get("size", 0) == 0:
        logging.info("[failure] no frame recorded for this episode, nothing to store.")
        dataset.clear_episode_buffer()
        return failure_dataset

    # The PNGs are written asynchronously: make sure they all landed before moving them.
    dataset._wait_image_writer()

    if failure_dataset is None:
        failure_dataset = make_failure_dataset(dataset, robot, cfg)

    src_ep_idx = buffer["episode_index"]
    dst_ep_idx = failure_dataset.num_episodes
    num_frames = buffer["size"]

    for cam_key in dataset.meta.camera_keys:
        src_dir = dataset._get_image_file_path(
            episode_index=src_ep_idx, image_key=cam_key, frame_index=0
        ).parent
        dst_dir = failure_dataset._get_image_file_path(
            episode_index=dst_ep_idx, image_key=cam_key, frame_index=0
        ).parent
        if not src_dir.is_dir():
            raise RuntimeError(f"[failure] frames directory '{src_dir}' of the rejected episode is missing.")
        dst_dir.parent.mkdir(parents=True, exist_ok=True)
        if dst_dir.exists():
            shutil.rmtree(dst_dir)
        shutil.move(str(src_dir), str(dst_dir))
        # Frame file names are unchanged by the move, only the episode directory differs.
        buffer[cam_key] = [
            str(
                failure_dataset._get_image_file_path(
                    episode_index=dst_ep_idx, image_key=cam_key, frame_index=i
                )
            )
            for i in range(num_frames)
        ]

    # Hand the buffer over: `save_episode` requires its episode_index to match the target dataset.
    buffer["episode_index"] = dst_ep_idx
    failure_dataset.episode_buffer = buffer
    failure_dataset.save_episode()

    _log_failure_episode(failure_dataset, dataset, src_ep_idx, dst_ep_idx, num_frames, plan_row)
    logging.info(
        f"[failure] rejected episode {src_ep_idx:06d} stored as episode {dst_ep_idx:06d} "
        f"of '{failure_dataset.root}' ({num_frames} frames)."
    )

    # `save_episode` consumed the buffer object we handed over; give the main dataset a fresh one.
    dataset.episode_buffer = dataset.create_episode_buffer()
    return failure_dataset


########################################################################################
# Control modes
########################################################################################


@safe_disconnect
def teleoperate(robot: Robot, cfg: TeleoperateControlConfig):
    control_loop(
        robot,
        control_time_s=cfg.teleop_time_s,
        fps=cfg.fps,
        display_data=cfg.display_data,
    )


@safe_disconnect
def record(
    robot: Robot,
    cfg: RecordControlConfig,
) -> LeRobotDataset:
    if cfg.resume:
        dataset = LeRobotDataset(
            cfg.repo_id,
            root=cfg.root,
        )
        if len(robot.cameras) > 0:
            dataset.start_image_writer(
                num_processes=cfg.num_image_writer_processes,
                num_threads=cfg.num_image_writer_threads_per_camera * len(robot.cameras),
            )
        sanity_check_dataset_robot_compatibility(dataset, robot, cfg.fps, cfg.video)
    else:
        # Create empty dataset or load existing saved episodes
        sanity_check_dataset_name(cfg.repo_id, None)
        if os.path.exists(cfg.root):
            assert False, f"Directory {cfg.root} already exists. Please remove it or use --resume to continue recording."
        dataset = LeRobotDataset.create(
            cfg.repo_id,
            cfg.fps,
            root=cfg.root,
            robot=robot,
            use_videos=cfg.video,
            image_writer_processes=cfg.num_image_writer_processes,
            image_writer_threads=cfg.num_image_writer_threads_per_camera * len(robot.cameras),
        )

    if dataset.num_episodes % cfg.num_episodes != 0:
        warn_message = (
            f"Warning: the requested episode count {cfg.num_episodes} does not "
            f"divide the existing episode count {dataset.num_episodes}. "
            f"The count will be adjusted to "
            f"{cfg.num_episodes - dataset.num_episodes % cfg.num_episodes}. "
            f"Continue? (y/n) "
        )
        logging.warning(warn_message)

        # Persist the warning under <dataset root>/logs.
        logs_dir = dataset.root / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(logs_dir / "num_episodes_warning.log", "a") as f:
            f.write(f"[{timestamp}] {warn_message}\n")

        _continue = input()
        if _continue != "y":
            return
        else:
            cfg.num_episodes -= dataset.num_episodes % cfg.num_episodes

    if not robot.is_connected:
        robot.connect()

    listener, events = init_keyboard_listener()

    # Execute a few seconds without recording to:
    # 1. teleoperate the robot to move it in starting position,
    # 2. give times to the robot devices to connect and start synchronizing,
    # 3. place the cameras windows on screen
    log_say("Warmup record", cfg.play_sounds)
    warmup_record(robot, events, cfg.warmup_time_s, cfg.display_data, cfg.fps)

    if has_method(robot, "teleop_safety_stop"):
        robot.teleop_safety_stop()

    # Resolve the plan CSV: explicit config wins, else fall back to "plan.csv" in the cwd.
    plan_csv_path = cfg.plan_csv
    if plan_csv_path is None and os.path.exists("plan.csv"):
        plan_csv_path = "plan.csv"
    if plan_csv_path is not None and not os.path.exists(plan_csv_path):
        logging.warning(f"plan_csv path '{plan_csv_path}' does not exist; no plan will be shown.")
        plan_csv_path = None

    recorded_episodes = 0
    # Opened lazily, on the first episode rejected with the left arrow key.
    failure_dataset = None
    while True:
        if recorded_episodes >= cfg.num_episodes:
            break

        # Read the first plan row that still lacks a recorded-data mark.
        plan_row = read_plan_row(plan_csv_path) if plan_csv_path is not None else None

        ep_idx = dataset.num_episodes
        log_say(f"Recording episode {ep_idx}", cfg.play_sounds)
        record_episode(
            robot=robot,
            dataset=dataset,
            events=events,
            episode_time_s=cfg.episode_time_s,
            display_data=cfg.display_data,
            fps=cfg.fps,
            single_task=cfg.single_task,
            plan_row=plan_row,
        )

        # Execute a few seconds without recording to give time to manually reset the environment.
        # Skip reset for the last episode to be recorded.
        if not events["stop_recording"] and (
            (recorded_episodes < cfg.num_episodes - 1) or events["rerecord_episode"]
        ):
            log_say("Reset the environment", cfg.play_sounds)
            reset_environment(robot, events, cfg.reset_time_s, cfg.fps)

        if events["rerecord_episode"]:
            log_say("Re-record episode", cfg.play_sounds)
            events["rerecord_episode"] = False
            events["exit_early"] = False
            if cfg.save_failure_episodes:
                # Keep the rejected data as an episode of a separate failure dataset instead of
                # dropping it. A problem there must not abort an ongoing collection session: fall
                # back to the old behaviour (delete) and keep going, loudly.
                try:
                    failure_dataset = stash_failed_episode(
                        dataset, failure_dataset, robot, cfg, plan_row
                    )
                except Exception:
                    logging.error(
                        "[failure] could not store the rejected episode in the failure dataset; "
                        "discarding it instead.\n" + traceback.format_exc()
                    )
                    discard_episode_frames(dataset, ep_idx)
            else:
                dataset.clear_episode_buffer()
            continue

        # Guard against a frozen/duplicated camera feed *before* committing the episode:
        # compare the just-recorded (still in-buffer) episode's first frame with the previous
        # episode's across all camera keys. If a duplicate is found, drop the current episode's
        # data, log the details under <dataset root>/logs, and abort the collection program.
        if cfg.dup_check_every > 0 and recorded_episodes % cfg.dup_check_every == 0:
            duplicates = check_current_episode_duplicate(dataset, cfg.dup_frame_threshold)
            if duplicates:
                ep_idx = dataset.num_episodes
                message = (
                    "Duplicate first frame(s) detected — the camera feed may be frozen. "
                    f"Discarding episode {ep_idx:06d} and aborting recording:\n  "
                    + "\n  ".join(duplicates)
                )

                # Persist the check details before tearing anything down.
                logs_dir = dataset.root / "logs"
                logs_dir.mkdir(parents=True, exist_ok=True)
                log_path = logs_dir / f"duplicate_check_ep{ep_idx:06d}.log"
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(log_path, "w") as f:
                    f.write(f"[{timestamp}] {message}\n")
                logging.error(f"{message}\n[dup-check] details written to {log_path}")

                # Drop the current (duplicated) episode's buffered data, then exit.
                dataset.clear_episode_buffer()
                raise RuntimeError(message)

        dataset.save_episode()
        recorded_episodes += 1

        # Write the recorded episode's data file path into the plan's "data_path" column.
        if plan_row is not None:
            ep_data_path = dataset.root / dataset.meta.get_data_file_path(ep_index=ep_idx)
            write_plan_data(plan_csv_path, plan_row["row_index"], ep_data_path)

            # Copy the (now-updated) plan file into the collected-data output directory.
            assert plan_csv_path is not None and os.path.exists(plan_csv_path)
            os.makedirs(cfg.root, exist_ok=True)
            shutil.copy(plan_csv_path, os.path.join(cfg.root, os.path.basename(plan_csv_path)))

        if events["stop_recording"]:
            break

    log_say("Stop recording", cfg.play_sounds, blocking=True)
    stop_recording(robot, listener, cfg.display_data)

    # Copy the (now-updated) plan file into the collected-data output directory.
    if plan_csv_path is not None and os.path.exists(plan_csv_path):
        try:
            os.makedirs(cfg.root, exist_ok=True)
            shutil.copy(plan_csv_path, os.path.join(cfg.root, os.path.basename(plan_csv_path)))
        except Exception as e:
            logging.warning(f"Failed to copy plan file '{plan_csv_path}' to '{cfg.root}': {e}")

    if cfg.push_to_hub:
        dataset.push_to_hub(tags=cfg.tags, private=cfg.private)

    log_say("Exiting", cfg.play_sounds)
    return dataset


def _init_rerun(control_config: ControlConfig, session_name: str = "lerobot_control_loop") -> None:
    """Initializes the Rerun SDK for visualizing the control loop."""
    if control_config.display_data and not is_headless():
        # Configure Rerun flush batch size default to 8KB if not set
        batch_size = os.getenv("RERUN_FLUSH_NUM_BYTES", "8000")
        os.environ["RERUN_FLUSH_NUM_BYTES"] = batch_size

        rr.init(session_name)
        # Get memory limit for rerun viewer parameters
        memory_limit = os.getenv("LEROBOT_RERUN_MEMORY_LIMIT", "10%")
        rr.spawn(memory_limit=memory_limit)


@parser.wrap()
def control_robot(cfg: ControlPipelineConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    robot = make_robot_from_config(cfg.robot)

    if isinstance(cfg.control, TeleoperateControlConfig):
        _init_rerun(control_config=cfg.control, session_name="lerobot_control_loop_teleop")
        teleoperate(robot, cfg.control)
    elif isinstance(cfg.control, RecordControlConfig):
        _init_rerun(control_config=cfg.control, session_name="lerobot_control_loop_record")
        record(robot, cfg.control)

    if robot.is_connected:
        # Disconnect manually to avoid a "Core dump" during process
        # termination due to camera threads not properly exiting.
        robot.disconnect()


if __name__ == "__main__":
    control_robot()
