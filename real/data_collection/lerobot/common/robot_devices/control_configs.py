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

from dataclasses import dataclass
from pathlib import Path

import draccus

from lerobot.common.robot_devices.robots.configs import RobotConfig


@dataclass
class ControlConfig(draccus.ChoiceRegistry):
    pass


@ControlConfig.register_subclass("teleoperate")
@dataclass
class TeleoperateControlConfig(ControlConfig):
    # Limit the maximum frames per second. By default, no limit.
    fps: int | None = None
    teleop_time_s: float | None = None
    # Display all cameras on screen
    display_data: bool = False


@ControlConfig.register_subclass("record")
@dataclass
class RecordControlConfig(ControlConfig):
    # Dataset identifier. By convention it should match '{hf_username}/{dataset_name}' (e.g. `lerobot/test`).
    repo_id: str
    # A short but accurate description of the task performed during the recording (e.g. "Pick the Lego block and drop it in the box on the right.")
    single_task: str
    # Root directory where the dataset will be stored (e.g. 'dataset/path').
    root: str | Path | None = None
    # Limit the frames per second.
    fps: int | None = None
    # Number of seconds before starting data collection. It allows the robot devices to warmup and synchronize.
    warmup_time_s: int | float = 10
    # Number of seconds for data recording for each episode.
    episode_time_s: int | float = 60
    # Number of seconds for resetting the environment after each episode.
    reset_time_s: int | float = 60
    # Number of episodes to record.
    num_episodes: int = 50
    # Encode frames in the dataset into video
    video: bool = True
    # Upload dataset to Hugging Face hub.
    push_to_hub: bool = True
    # Upload on private repository on the Hugging Face hub.
    private: bool = False
    # Add tags to your dataset on the hub.
    tags: list[str] | None = None
    # Number of subprocesses handling the saving of frames as PNG. Set to 0 to use threads only;
    # set to ≥1 to use subprocesses, each using threads to write images. The best number of processes
    # and threads depends on your system. We recommend 4 threads per camera with 0 processes.
    # If fps is unstable, adjust the thread count. If still unstable, try using 1 or more subprocesses.
    num_image_writer_processes: int = 0
    # Number of threads writing the frames as png images on disk, per camera.
    # Too many threads might cause unstable teleoperation fps due to main thread being blocked.
    # Not enough threads might cause low camera fps.
    num_image_writer_threads_per_camera: int = 4
    # Display all cameras on screen
    display_data: bool = False
    # Use vocal synthesis to read events.
    play_sounds: bool = False
    # Resume recording on an existing dataset.
    resume: bool = False
    # Path to a CSV file whose i-th row is shown on screen as the plan before recording episode i.
    # If None, falls back to "plan.csv" in the current working directory when present.
    plan_csv: str | None = None
    # Every `dup_check_every` saved episodes, scan the just-recorded window and compare each
    # episode's first frame with the previous episode's, for every camera key. If the mean
    # absolute pixel difference (0-255 scale) is below this threshold, the two first frames are
    # considered identical and recording aborts with an error. Set threshold to 0 to disable.
    dup_frame_threshold: float = 0.001
    # Run the duplicate first-frame scan once per this many recorded episodes.
    dup_check_every: int = 1
    # When the left arrow key rejects an episode, keep its data by moving it into a separate
    # ("failure") LeRobot dataset instead of deleting it. Set to false for the old delete behaviour.
    save_failure_episodes: bool = True
    # Root directory of that failure dataset. Defaults to '<root>_failure' when left unset.
    failure_root: str | Path | None = None
    # Repo id of the failure dataset. Defaults to '<repo_id>_failure' when left unset.
    failure_repo_id: str | None = None
    # dup_check_every: int = 10
    # dup_check_every: int = 0


@dataclass
class ControlPipelineConfig:
    robot: RobotConfig
    control: ControlConfig
