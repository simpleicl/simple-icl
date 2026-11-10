#!/usr/bin/env python3
"""Replay a recorded Franka episode.

Drives the Franka arm with the recorded ``action`` of one episode: every frame
is sent through ``robot.send_action`` (executed by replay mode's 1 kHz joint
tracker), the arm's current state is read back with
``robot.capture_observation`` and compared against the recorded state,
printing per-joint / L2 / max-abs errors plus an overall summary at the end.
``--display`` optionally shows a 2x2 camera window (off by default so headless
machines never load cv2's Qt plugin); with ``--live-image`` the window shows
the live camera frames instead of the recorded PNGs.

The data was recorded with::

    PYTHONPATH="$(pwd)" python lerobot/scripts/control_robot.py \
        --robot.type=franka --control.type=record \
        --control.root=outputs/1015 ...

Mock replay (the default; runs without hardware)::

    PYTHONPATH="$(pwd)" python lerobot/scripts/replay_franka.py \
        --root outputs/1015 --episode 0

Real-arm replay (drives the arm / reads real states)::

    PYTHONPATH="$(pwd)" python lerobot/scripts/replay_franka.py \
        --root outputs/1015 --episode 0 --no-mock --live-image

Or use ``replay.sh`` in the repo root (real arm by default, MOCK=1 for mock).
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lerobot.common.robot_devices.robots.configs import FrankaRobotConfig
from lerobot.common.robot_devices.robots.utils import make_robot_from_config
from lerobot.common.robot_devices.utils import busy_wait
from lerobot.common.utils.utils import init_logging

# cv2 is imported lazily inside replay(), after the arm control processes are
# forked: importing it at module load initializes its Qt plugin, which crashes
# the forked children on headless machines.
cv2 = None


# ----------------------------------------------------------------------------
# Dataset loading helpers
# ----------------------------------------------------------------------------
def load_info(root: Path) -> dict:
    with open(root / "meta" / "info.json") as f:
        return json.load(f)


def load_episode_frames(root: Path, info: dict, episode: int) -> pd.DataFrame:
    """Load the recorded parquet for one episode, ordered by frame_index."""
    chunk = episode // info.get("chunks_size", 1000)
    data_path = info["data_path"].format(episode_chunk=chunk, episode_index=episode)
    parquet_path = root / data_path
    if not parquet_path.exists():
        raise FileNotFoundError(f"Episode parquet not found: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    return df.sort_values("frame_index").reset_index(drop=True)


def recorded_image_path(root: Path, camera_key: str, episode: int, frame_index: int) -> Path:
    return (
        root
        / "images"
        / f"observation.images.{camera_key}"
        / f"episode_{episode:06d}"
        / f"frame_{frame_index:06d}.png"
    )


def image_camera_keys(info: dict) -> list[str]:
    image_keys = [
        key.removeprefix("observation.images.")
        for key, feature in info.get("features", {}).items()
        if feature.get("dtype") == "image"
    ]
    if not image_keys:
        raise ValueError("No image camera keys found in dataset metadata.")
    return image_keys


# ----------------------------------------------------------------------------
# Display helpers
# ----------------------------------------------------------------------------
def to_bgr_uint8(img: np.ndarray) -> np.ndarray:
    """Convert an RGB tensor/array (from the robot) to a BGR uint8 image for cv2."""
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    # Robot cameras deliver RGB; cv2 windows expect BGR.
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def annotate(img: np.ndarray, lines: list[str]) -> np.ndarray:
    img = img.copy()
    y = 22
    for line in lines:
        cv2.putText(img, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
        y += 26
    return img


def label_tile(img: np.ndarray, label: str) -> np.ndarray:
    img = img.copy()
    cv2.putText(img, label, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, label, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    return img


def build_camera_mosaic(
    frames: dict[str, np.ndarray],
    camera_keys: list[str],
) -> np.ndarray | None:
    if not camera_keys:
        return None

    ordered_keys = list(camera_keys[:4])
    tiles: list[np.ndarray] = []
    tile_height = tile_width = None

    for camera_key in ordered_keys:
        frame = frames.get(camera_key)
        if frame is None:
            continue
        if tile_height is None:
            tile_height, tile_width = frame.shape[:2]
        elif frame.shape[:2] != (tile_height, tile_width):
            frame = cv2.resize(frame, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
        tiles.append(label_tile(frame, camera_key))

    if not tiles:
        return None

    blank = np.zeros_like(tiles[0])
    while len(tiles) < 4:
        tiles.append(blank.copy())

    top_row = np.hstack([tiles[0], tiles[1]])
    bottom_row = np.hstack([tiles[2], tiles[3]])
    return np.vstack([top_row, bottom_row])


# ----------------------------------------------------------------------------
# Main replay loop
# ----------------------------------------------------------------------------
def replay(args):
    init_logging()
    root = Path(args.root)
    info = load_info(root)
    fps = args.fps or info["fps"]

    df = load_episode_frames(root, info, args.episode)
    n = len(df)
    logging.info(f"Loaded episode {args.episode}: {n} frames @ {fps} fps from {root}")
    camera_keys = image_camera_keys(info)
    if args.camera in camera_keys:
        camera_keys = [args.camera] + [key for key in camera_keys if key != args.camera]
    logging.info(f"Visualizing cameras: {', '.join(camera_keys)}")

    # Joint names for pretty per-joint error printing.
    joint_names = info["features"]["action"].get("names") or [
        f"j{i}" for i in range(len(df["action"].iloc[0]))
    ]

    # Replay mode: send_action drives the arm via joint-position control.
    # mock=True lets it run without hardware.
    cfg = FrankaRobotConfig(control_mode="replay", mock=args.mock)
    robot = make_robot_from_config(cfg)
    robot.connect()

    window = "replay - 2x2 cameras"
    display = args.display
    if display:
        try:
            global cv2
            import cv2
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        except Exception as e:
            logging.warning(f"Cannot open display / import cv2 ({e}); running without visualization.")
            display = False

    state_key = "observation.state"

    errors = []  # per-frame (current_state - recorded_state)
    try:
        for i in range(n):
            t0 = time.perf_counter()

            rec_action = np.asarray(df["action"].iloc[i], dtype=np.float32)
            rec_state = np.asarray(df[state_key].iloc[i], dtype=np.float32)
            frame_index = int(df["frame_index"].iloc[i])

            # 1) Drive the arm with the recorded action.
            robot.send_action(torch.from_numpy(rec_action))

            # 2) Read the current state back from the robot.
            obs = robot.capture_observation()
            cur_state = obs[state_key].numpy().astype(np.float32)

            # 3) Compare current vs recorded state.
            err = cur_state - rec_state
            errors.append(err)
            l2 = float(np.linalg.norm(err))
            max_abs = float(np.max(np.abs(err)))

            # 4) Show the current main-view image.
            if display:
                images = {}
                for camera_key in camera_keys:
                    obs_key = f"observation.images.{camera_key}"
                    img = None
                    if args.live_image and obs_key in obs:
                        img = to_bgr_uint8(obs[obs_key].numpy())
                    else:
                        p = recorded_image_path(root, camera_key, args.episode, frame_index)
                        if p.exists():
                            img = cv2.imread(str(p))
                    if img is not None:
                        images[camera_key] = img

                img = build_camera_mosaic(images, camera_keys)
                if img is not None:
                    img = annotate(
                        img,
                        [
                            f"ep {args.episode}  frame {frame_index}/{n - 1}",
                            f"L2 err {l2:.4f}   max|e| {max_abs:.4f}",
                        ],
                    )
                    cv2.imshow(window, img)
                    # 'q' or ESC quits early.
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        logging.info("Quit requested from display window.")
                        break

            # Keep the requested replay rate.
            dt = time.perf_counter() - t0
            busy_wait(1 / fps - dt)
    finally:
        if display:
            cv2.destroyAllWindows()
        if robot.is_connected:
            robot.disconnect()

    # Summary statistics over all replayed frames.
    if errors:
        errs = np.stack(errors)  # (frames, dof)
        abs_errs = np.abs(errs)
        logging.info("=" * 70)
        logging.info(f"Replay finished: {len(errs)} frames.")
        logging.info(f"Mean L2 error : {np.mean(np.linalg.norm(errs, axis=1)):.5f}")
        logging.info(f"Max  L2 error : {np.max(np.linalg.norm(errs, axis=1)):.5f}")
        logging.info(f"Overall max |e|: {np.max(abs_errs):.5f}")
        per_joint_mean = "  ".join(
            f"{name}:{m:.4f}" for name, m in zip(joint_names, abs_errs.mean(axis=0))
        )
        logging.info(f"Per-joint mean |e|: {per_joint_mean}")


def parse_args():
    p = argparse.ArgumentParser(description="Replay a recorded Franka episode.")
    p.add_argument("--root", required=True, help="Dataset root directory (e.g. outputs/1015).")
    p.add_argument("--episode", type=int, default=0, help="Episode index to replay.")
    p.add_argument("--fps", type=int, default=None, help="Replay rate (default: dataset fps).")
    p.add_argument("--camera", default="cam_high", help="Main-view camera key to display.")
    p.add_argument(
        "--mock",
        dest="mock",
        action="store_true",
        default=True,
        help="Use mock robot/cameras (default; matches how the demo data was recorded).",
    )
    p.add_argument(
        "--no-mock",
        dest="mock",
        action="store_false",
        help="Drive the REAL Franka arm and read real states.",
    )
    p.add_argument(
        "--live-image",
        action="store_true",
        help="Show the live camera frame instead of the recorded PNG (use on real hardware).",
    )
    p.add_argument(
        "--display",
        dest="display",
        action="store_true",
        default=False,
        help="Show a 2x2 camera window while replaying (needs a display; off by default "
        "so headless machines never load cv2's Qt plugin).",
    )
    return p.parse_args()


if __name__ == "__main__":
    replay(parse_args())
