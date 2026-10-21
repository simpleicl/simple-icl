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

########################################################################################
# Utilities
########################################################################################


import csv
import logging
import os
import time
import traceback
import unicodedata
from functools import cache

import rerun as rr
from deepdiff import DeepDiff
from termcolor import colored

from lerobot.common.datasets.image_writer import safe_stop_image_writer
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import get_features_from_robot
from lerobot.common.robot_devices.robots.utils import Robot
from lerobot.common.robot_devices.utils import busy_wait
from lerobot.common.utils.utils import has_method


@cache
def is_headless():
    """Detects if python is running without a monitor."""
    try:
        import pynput  # noqa

        return False
    except Exception:
        print(
            "Error trying to import pynput. Switching to headless mode. "
            "As a result, the video stream from the cameras won't be shown, "
            "and you won't be able to change the control flow with keyboards. "
            "For more info, see traceback below.\n"
        )
        traceback.print_exc()
        print()
        return True


def init_keyboard_listener():
    # Keyboard controls:
    # - F3: Start recording (when waiting) / End current episode (when recording)
    # - Left arrow: Rerecord the last episode
    # - Escape: Stop all data recording
    events = {}
    events["exit_early"] = False
    events["rerecord_episode"] = False
    events["stop_recording"] = False
    events["start_recording"] = False

    if is_headless():
        logging.warning(
            "Headless environment detected. On-screen cameras display and keyboard inputs will not be available."
        )
        listener = None
        return listener, events

    # Only import pynput if not in a headless environment
    from pynput import keyboard

    def on_press(key):
        try:
            if key == keyboard.Key.f3:
                print("F3 pressed.")
                events["start_recording"] = True
                events["exit_early"] = True
            elif key == keyboard.Key.left:
                print("Left arrow key pressed. Exiting loop and rerecord the last episode...")
                events["rerecord_episode"] = True
                events["exit_early"] = True
            elif key == keyboard.Key.esc:
                print("Escape key pressed. Stopping data recording...")
                events["stop_recording"] = True
                events["exit_early"] = True
        except Exception as e:
            print(f"Error handling key press: {e}")

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    return listener, events


def warmup_record(robot, events, warmup_time_s, display_data, fps):
    """Teleoperate without recording for a while, to move the robot to its
    starting position and give the devices time to connect and synchronize."""
    control_loop(
        robot=robot,
        control_time_s=warmup_time_s,
        display_data=display_data,
        events=events,
        fps=fps,
    )


# Column headers expected in the plan CSV.
PLAN_COL_SCENE = "scene"
PLAN_COL_CLUTTER = "objects"
PLAN_COL_ACTION = "action"
PLAN_COL_DATA = "data_path"


def _plan_header_index(header, name):
    stripped = [c.strip() for c in header]
    return stripped.index(name) if name in stripped else None


def read_plan_row(plan_csv_path):
    """Return the first plan row that still lacks a 'data_path' mark.

    The returned dict has:
      - ``row_index``: 0-based index into the data rows (header excluded), used to
        write the recorded data path back into the 'data_path' column afterwards.
      - ``scene`` / ``clutter`` / ``action``: the 'scene' / 'objects' / 'action'
        cells.

    Returns None if the file is missing, malformed, or every row is already filled.
    """
    if not plan_csv_path or not os.path.exists(plan_csv_path):
        return None

    with open(plan_csv_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    if len(rows) < 2:
        return None

    header = rows[0]
    scene_i = _plan_header_index(header, PLAN_COL_SCENE)
    clutter_i = _plan_header_index(header, PLAN_COL_CLUTTER)
    action_i = _plan_header_index(header, PLAN_COL_ACTION)
    data_i = _plan_header_index(header, PLAN_COL_DATA)
    if action_i is None or data_i is None:
        logging.warning(
            f"plan.csv ({plan_csv_path}) is missing required columns "
            f"'{PLAN_COL_ACTION}' and/or '{PLAN_COL_DATA}'."
        )
        return None

    def cell(row, i):
        return row[i].strip() if i is not None and i < len(row) else ""

    for idx, row in enumerate(rows[1:]):
        if cell(row, data_i):
            continue
        return {
            "row_index": idx,
            "scene": cell(row, scene_i),
            "clutter": cell(row, clutter_i),
            "action": cell(row, action_i),
        }
    logging.info(f"plan.csv ({plan_csv_path}) has no unfilled '{PLAN_COL_DATA}' rows left.")
    return None


def write_plan_data(plan_csv_path, row_index, data_path):
    """Write `data_path` into the 'data_path' cell of data row `row_index` (0-based)."""
    if not plan_csv_path or not os.path.exists(plan_csv_path):
        return

    with open(plan_csv_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    if len(rows) < 2:
        return

    data_i = _plan_header_index(rows[0], PLAN_COL_DATA)
    if data_i is None:
        return

    target = row_index + 1  # skip header
    if target >= len(rows):
        logging.warning(f"plan.csv ({plan_csv_path}) has no data row {row_index} to update.")
        return

    row = rows[target]
    while len(row) <= data_i:
        row.append("")
    row[data_i] = str(data_path)
    rows[target] = row

    with open(plan_csv_path, "w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerows(rows)


def print_plan_console(scene, clutter, action):
    """Print the plan for this episode to the terminal.

    When a scene is present, the scene and clutter are shown inside an asterisk box
    (one per line) and the action on a single dash-bordered line below it. When the
    scene is empty, only the action line is printed.
    """

    def cell_width(s):
        # East-Asian wide/fullwidth glyphs occupy two terminal columns.
        return sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in s)

    print()
    if scene:
        scene_lines = [f"{PLAN_COL_SCENE}: {scene}", f"{PLAN_COL_CLUTTER}: {clutter}"]
        inner = max(max(cell_width(line) for line in scene_lines), 30)
        border = "*" * (inner - 8)

        def boxed(line):
            gap = inner - cell_width(line)
            return "***" + " " + line + " " * (gap + 1) + "***"

        box = [border]
        for line in scene_lines:
            box.append(boxed(line))
        box += [border]
        for line in box:
            print(colored(line, "yellow", attrs=["bold"]))

    action_line = f"{PLAN_COL_ACTION}: {action}"
    dash = "-" * (cell_width(action_line) + 4)
    print(colored(dash, "cyan", attrs=["bold"]))
    print(colored(f"- {action_line} -", "cyan", attrs=["bold"]))
    print("---Reset the scene first, then execute!!!---")
    print(colored(dash, "cyan", attrs=["bold"]))
    print(colored(">>> Press F3 to start recording <<<", "green", attrs=["bold"]))
    print()


def wait_for_f3(events):
    """Block until F3 is pressed (events['start_recording'] becomes True)."""
    if is_headless():
        # Without a keyboard listener there is no way to press F3, so waiting
        # here would hang forever. Skip the wait instead: each episode starts
        # right away. This is what makes (mock) recording work headless.
        logging.warning("Headless environment: F3 not available, starting the episode immediately.")
        return
    while not events.get("start_recording", False):
        time.sleep(0.05)
    events["start_recording"] = False
    assert events["exit_early"]
    events["exit_early"] = False


def record_episode(
    robot,
    dataset,
    events,
    episode_time_s,
    display_data,
    fps,
    single_task,
    plan_row=None,
):
    # Show the plan for this episode (if any) while waiting for F3.
    if plan_row is not None:
        print_plan_console(plan_row["scene"], plan_row["clutter"], plan_row["action"])

    # Wait for F3 before starting to record this episode
    wait_for_f3(events)

    if episode_time_s <= 0:
        episode_time_s = None
    control_loop(
        robot=robot,
        control_time_s=episode_time_s,
        display_data=display_data,
        dataset=dataset,
        events=events,
        fps=fps,
        single_task=single_task,
    )


@safe_stop_image_writer
def control_loop(
    robot,
    control_time_s=None,
    display_data=False,
    dataset: LeRobotDataset | None = None,
    events=None,
    fps: int | None = None,
    single_task: str | None = None,
):
    if not robot.is_connected:
        robot.connect()

    if events is None:
        events = {"exit_early": False}

    unbounded = control_time_s is None
    if unbounded:
        control_time_s = float("inf")

    if dataset is not None and single_task is None:
        raise ValueError("You need to provide a task as argument in `single_task`.")

    if dataset is not None and fps is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset['fps']} != {fps}).")

    # Count frames instead of wall-clock time when both a duration and a frame rate
    # are given. A purely time-based loop drops frames whenever an iteration takes
    # longer than 1/fps (notably the first one, which pays lazy initialization
    # costs), so episodes end up shorter than `control_time_s * fps`.
    if unbounded or fps is None:
        num_frames = None
    elif control_time_s > 0:
        num_frames = max(1, round(control_time_s * fps))
    else:
        num_frames = 0
    frame_count = 0

    start_episode_t = time.perf_counter()

    while True:
        if num_frames is not None:
            if frame_count >= num_frames:
                break
        elif time.perf_counter() - start_episode_t >= control_time_s:
            break

        start_loop_t = time.perf_counter()

        observation, action = robot.teleop_step(record_data=True)

        if dataset is not None:
            frame = {**observation, **action, "task": single_task}
            dataset.add_frame(frame)

        if display_data and not is_headless():
            for k, v in action.items():
                for i, vv in enumerate(v):
                    rr.log(f"sent_{k}_{i}", rr.Scalar(vv.numpy()))

            image_keys = [key for key in observation if "image" in key]
            for key in image_keys:
                rr.log(key, rr.Image(observation[key].numpy()), static=True)

        if fps is not None:
            dt_s = time.perf_counter() - start_loop_t
            busy_wait(1 / fps - dt_s)

        frame_count += 1

        if events["exit_early"]:
            events["exit_early"] = False
            if events["start_recording"]:
                events["start_recording"] = False
            break


def reset_environment(robot, events, reset_time_s, fps):
    if has_method(robot, "teleop_safety_stop"):
        robot.teleop_safety_stop()

    print("Please reset the scene!")


def stop_recording(robot, listener, display_data):
    robot.disconnect()

    if not is_headless() and listener is not None:
        listener.stop()


def sanity_check_dataset_name(repo_id, policy_cfg):
    _, dataset_name = repo_id.split("/")
    # either repo_id doesnt start with "eval_" and there is no policy
    # or repo_id starts with "eval_" and there is a policy

    # Check if dataset_name starts with "eval_" but policy is missing
    if dataset_name.startswith("eval_") and policy_cfg is None:
        raise ValueError(
            f"Your dataset name begins with 'eval_' ({dataset_name}), but no policy is provided ({policy_cfg.type})."
        )

    # Check if dataset_name does not start with "eval_" but policy is provided
    if not dataset_name.startswith("eval_") and policy_cfg is not None:
        raise ValueError(
            f"Your dataset name does not begin with 'eval_' ({dataset_name}), but a policy is provided ({policy_cfg.type})."
        )


def sanity_check_dataset_robot_compatibility(
    dataset: LeRobotDataset, robot: Robot, fps: int, use_videos: bool
) -> None:
    fields = [
        ("robot_type", dataset.meta.robot_type, robot.robot_type),
        ("fps", dataset.fps, fps),
        ("features", dataset.features, get_features_from_robot(robot, use_videos)),
    ]

    mismatches = []
    for field, dataset_value, present_value in fields:
        diff = DeepDiff(dataset_value, present_value, exclude_regex_paths=[r".*\['info'\]$"])
        if diff:
            mismatches.append(f"{field}: expected {present_value}, got {dataset_value}")

    if mismatches:
        raise ValueError(
            "Dataset metadata compatibility check failed with mismatches:\n" + "\n".join(mismatches)
        )
