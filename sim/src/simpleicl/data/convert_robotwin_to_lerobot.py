#!/usr/bin/env python3
"""Convert paired RoboTwin scenes or tar shards into LeRobot episodes.

Each episode contains synchronized camera videos, joint states, next-step
joint actions, language instructions, and paired ALOHA demonstration videos.
States and actions contain the 14 ALOHA joint and gripper coordinates.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Iterator

import av
import h5py
import numpy as np
import pandas as pd
import pyarrow.parquet as parquet
import yaml
from PIL import Image


CHUNKS_SIZE = 1000
ROBOT_DIR_NAME = "aloha-agilex"
CAMERA_NAMES = {
    "head_camera": "cam_high",
    "left_camera": "cam_left_wrist",
    "right_camera": "cam_right_wrist",
}
IMAGE_SIZE = (640, 480)  # width, height; matches pi0 process_data_dual.py
ARCHIVE_SUFFIXES = (".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz")
JOINT_COMPONENT_KEYS = (
    "/joint_action/left_arm",
    "/joint_action/left_gripper",
    "/joint_action/right_arm",
    "/joint_action/right_gripper",
)


@dataclass(frozen=True)
class ArchiveMember:
    archive_path: Path
    member_name: str
    logical_path: PurePosixPath
    size: int

    def __str__(self) -> str:
        return f"{self.archive_path}::{self.member_name}"


SourceFile = Path | ArchiveMember


@dataclass(frozen=True)
class Episode:
    path: SourceFile
    condition_path: SourceFile | None
    instruction_path: SourceFile | None
    episode_id: str
    condition_paths: tuple[SourceFile, ...] = ()
    condition_complete: bool = True


def _episode_condition_paths(episode: Episode) -> tuple[SourceFile, ...]:
    """Return all condition videos, including legacy single-path episodes."""
    if episode.condition_paths:
        return episode.condition_paths
    return (episode.condition_path,) if episode.condition_path is not None else ()


class ImageStats:
    def __init__(self) -> None:
        self.minimum = np.full(3, 255.0, dtype=np.float64)
        self.maximum = np.full(3, 0.0, dtype=np.float64)
        self.total = np.zeros(3, dtype=np.float64)
        self.total_sq = np.zeros(3, dtype=np.float64)
        self.pixel_count = 0

    def update(self, image: np.ndarray) -> None:
        values = image.astype(np.float64)
        self.minimum = np.minimum(self.minimum, values.min(axis=(0, 1)))
        self.maximum = np.maximum(self.maximum, values.max(axis=(0, 1)))
        self.total += values.sum(axis=(0, 1))
        self.total_sq += np.square(values).sum(axis=(0, 1))
        self.pixel_count += image.shape[0] * image.shape[1]

    def as_json(self, frame_count: int) -> dict:
        mean = self.total / self.pixel_count / 255.0
        variance = np.maximum(
            self.total_sq / self.pixel_count - np.square(self.total / self.pixel_count),
            0.0,
        )
        std = np.sqrt(variance) / 255.0

        def channel_values(values: np.ndarray) -> list[list[list[float]]]:
            return [[[float(value)]] for value in values]

        return {
            "min": channel_values(self.minimum / 255.0),
            "max": channel_values(self.maximum / 255.0),
            "mean": channel_values(mean),
            "std": channel_values(std),
            "count": [int(frame_count)],
        }


def _is_archive(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(suffix) for suffix in ARCHIVE_SUFFIXES)


def _checked_member_path(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"tar contains an unsafe member: {name!r}")
    return path


def _is_conversion_source(path: PurePosixPath) -> bool:
    suffix = path.suffix.lower()
    return suffix in {".hdf5", ".h5", ".mp4"} or (
        suffix == ".json" and path.parent.name == "instructions"
    )


def _input_logical_prefix(input_path: Path) -> PurePosixPath:
    anchor = input_path if input_path.is_dir() else input_path.parent
    return (
        PurePosixPath(anchor.name)
        if anchor.name.startswith("split_")
        else PurePosixPath()
    )


def _logical_path(input_path: Path, path: Path) -> PurePosixPath:
    root = input_path if input_path.is_dir() else input_path.parent
    relative = path.relative_to(root)
    return _input_logical_prefix(input_path) / PurePosixPath(relative.as_posix())


def _source_logical_path(source: SourceFile) -> PurePosixPath:
    if isinstance(source, ArchiveMember):
        return source.logical_path
    return PurePosixPath(source.as_posix())


def _source_size(source: SourceFile) -> int:
    return source.size if isinstance(source, ArchiveMember) else source.stat().st_size


@contextmanager
def _open_source(source: SourceFile) -> Iterator[BinaryIO]:
    if isinstance(source, Path):
        with source.open("rb") as handle:
            yield handle
        return
    with tarfile.open(source.archive_path, "r:*") as archive:
        member = archive.getmember(source.member_name)
        if not member.isfile():
            raise ValueError(f"tar member is not a regular file: {source}")
        handle = archive.extractfile(member)
        if handle is None:
            raise FileNotFoundError(f"Cannot read tar member: {source}")
        with handle:
            yield handle


def _read_source_text(source: SourceFile | None) -> str:
    if source is None:
        raise FileNotFoundError("Source file does not exist")
    with _open_source(source) as handle:
        return handle.read().decode("utf-8")


def _load_archive_manifests(input_root: Path) -> list[dict]:
    records: list[dict] = []
    for manifest_path in sorted(input_root.rglob("archive_manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid shard manifest: {manifest_path}: {error}") from error
        if manifest.get("format") != "robotwin-tar-shards":
            continue
        entries = manifest.get("archives")
        if not isinstance(entries, list):
            raise ValueError(f"Shard manifest is missing the archives list: {manifest_path}")
        prefix = _logical_path(input_root, manifest_path.parent / "placeholder").parent
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
                raise ValueError(f"Shard manifest contains an invalid archive entry: {manifest_path}")
            archive_name = _checked_member_path(entry["name"])
            if len(archive_name.parts) != 1:
                raise ValueError(
                    f"Invalid archive name in shard manifest: {manifest_path}: {entry['name']!r}"
                )
            members = entry.get("members")
            if members is not None and not isinstance(members, list):
                raise ValueError(
                    f"Shard manifest contains invalid members: {manifest_path}: {entry['name']}"
                )
            records.append(
                {
                    "manifest_path": manifest_path,
                    "archive_path": manifest_path.parent / entry["name"],
                    "logical_prefix": prefix,
                    "expected_size": entry.get("size"),
                    "members": members,
                }
            )
    return records


def _manifest_member_sources(record: dict) -> list[ArchiveMember]:
    members = record["members"]
    if members is None:
        return []
    sources = []
    for item in members:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValueError(f"Invalid member in shard manifest: {record['manifest_path']}")
        member_path = _checked_member_path(item["path"])
        if not _is_conversion_source(member_path):
            continue
        size = item.get("size", 0)
        if not isinstance(size, int) or size < 0:
            raise ValueError(
                f"Invalid member size in shard manifest: {record['manifest_path']}: {item['path']}"
            )
        sources.append(
            ArchiveMember(
                archive_path=record["archive_path"],
                member_name=item["path"],
                logical_path=record["logical_prefix"] / member_path,
                size=size,
            )
        )
    return sources


def _validate_manifest_records(
    records: list[dict], local_paths: set[PurePosixPath]
) -> None:
    problems: list[str] = []
    for record in records:
        archive_path = record["archive_path"]
        expected_size = record["expected_size"]
        actual_size = archive_path.stat().st_size if archive_path.is_file() else None
        complete = actual_size is not None and (
            not isinstance(expected_size, int) or actual_size == expected_size
        )
        if complete:
            continue
        members = record["members"]
        if members is not None:
            required = {
                source.logical_path for source in _manifest_member_sources(record)
            }
            if required.issubset(local_paths):
                continue
        if actual_size is None:
            problems.append(f"missing: {archive_path}")
        else:
            problems.append(
                f"incomplete: {archive_path} ({actual_size}/{expected_size} bytes)"
            )
    if problems:
        preview = "; ".join(problems[:5])
        remaining = f"; {len(problems) - 5} more issues" if len(problems) > 5 else ""
        raise ValueError(f"Incomplete tar shards; stopping conversion: {preview}{remaining}")


def _validate_archive_manifests(input_root: Path) -> None:
    """Reject manifest shards that are neither complete nor locally extracted."""
    local_paths = {
        _logical_path(input_root, path)
        for path in input_root.rglob("*")
        if path.is_file() and _is_conversion_source(_logical_path(input_root, path))
    }
    _validate_manifest_records(_load_archive_manifests(input_root), local_paths)


def _scan_archive(archive_path: Path, prefix: PurePosixPath) -> list[ArchiveMember]:
    sources = []
    with tarfile.open(archive_path, "r:*") as archive:
        for member in archive:
            member_path = _checked_member_path(member.name)
            if not member.isfile() or not _is_conversion_source(member_path):
                continue
            sources.append(
                ArchiveMember(
                    archive_path, member.name, prefix / member_path, int(member.size)
                )
            )
    return sources


def _catalog_input(input_path: Path) -> dict[PurePosixPath, SourceFile]:
    """Index local files and tar members without extracting archive contents."""
    if input_path.is_file():
        if not _is_archive(input_path):
            raise ValueError(f"--input is not a directory or supported tar archive: {input_path}")
        sources = {
            source.logical_path: source
            for source in _scan_archive(input_path, _input_logical_prefix(input_path))
        }
        if not any(path.suffix.lower() in {".hdf5", ".h5"} for path in sources):
            raise FileNotFoundError(f"No HDF5 files found in tar archive: {input_path}")
        return sources
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    sources: dict[PurePosixPath, SourceFile] = {}
    for path in input_path.rglob("*"):
        if not path.is_file():
            continue
        logical = _logical_path(input_path, path)
        if _is_conversion_source(logical):
            sources[logical] = path

    records = _load_archive_manifests(input_path)
    _validate_manifest_records(records, set(sources))
    recorded_archives = {record["archive_path"].resolve() for record in records}
    for record in records:
        archive_path = record["archive_path"]
        expected_size = record["expected_size"]
        if not archive_path.is_file():
            continue
        if (
            isinstance(expected_size, int)
            and archive_path.stat().st_size != expected_size
        ):
            continue
        member_sources = _manifest_member_sources(record)
        if not member_sources:
            member_sources = _scan_archive(archive_path, record["logical_prefix"])
        for source in member_sources:
            sources.setdefault(source.logical_path, source)

    for archive_path in sorted(
        path for path in input_path.rglob("*") if path.is_file() and _is_archive(path)
    ):
        if archive_path.resolve() in recorded_archives:
            continue
        prefix = _logical_path(input_path, archive_path).parent
        for source in _scan_archive(archive_path, prefix):
            sources.setdefault(source.logical_path, source)

    if not any(path.suffix.lower() in {".hdf5", ".h5"} for path in sources):
        raise FileNotFoundError(f"No HDF5 files found in directory or tar archive: {input_path}")
    return sources


def _episode_number(path: SourceFile) -> int:
    name = _source_logical_path(path).name
    match = re.fullmatch(r"episode(\d+)\.(?:hdf5|h5)", name, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"Invalid episode filename: {path}")
    return int(match.group(1))


def _episode_id(
    path: SourceFile,
    logical_path: PurePosixPath | None = None,
) -> str:
    """Return a stable identity rooted at the indexed input layout."""
    parts = (logical_path or _source_logical_path(path)).parts
    split_positions = [
        index for index, part in enumerate(parts) if part.startswith("split_")
    ]
    if split_positions:
        parts = parts[split_positions[-1] :]
    elif logical_path is None:
        robot_index = parts.index(ROBOT_DIR_NAME)
        parts = parts[max(robot_index - 1, 0) :]
    return PurePosixPath(*parts).as_posix()


def _episode_task_name(episode: Episode) -> str:
    """Return the RoboTwin task directory containing an episode."""
    return _source_logical_path(episode.path).parent.parent.parent.name


def _episode_group_key(episode: Episode) -> PurePosixPath:
    """Return the placement/task directory shared by perturbations."""
    return _source_logical_path(episode.path).parent.parent.parent


def _perturbation_number(episode: Episode) -> int:
    match = re.search(
        r"__pert(\d+)$", _source_logical_path(episode.path).parent.parent.name
    )
    return int(match.group(1)) if match else -1


def _normalize_task_names(values: list[str] | None) -> set[str]:
    """Normalize repeated/space- or comma-separated task names."""
    names: set[str] = set()
    for value in values or []:
        for name in str(value).split(","):
            name = name.strip()
            if name:
                names.add(name.casefold())
    return names


def _filter_episodes_by_task(
    episodes: list[Episode],
    excluded_tasks: set[str],
) -> tuple[list[Episode], dict[str, int]]:
    """Drop episodes whose immediate RoboTwin task directory is excluded."""
    if not excluded_tasks:
        return episodes, {}
    kept: list[Episode] = []
    excluded: dict[str, int] = {}
    for episode in episodes:
        task_name = _episode_task_name(episode)
        if task_name.casefold() in excluded_tasks:
            excluded[task_name] = excluded.get(task_name, 0) + 1
        else:
            kept.append(episode)
    return kept, excluded


def _limit_aloha_episodes_round_robin(
    episodes: list[Episode],
    max_episodes: int | None,
    seed: int,
    task_limits: dict[str, int] | None = None,
) -> tuple[list[Episode], dict[str, dict[str, int]]]:
    """Select ALOHA perturbations by placement round-robin within each task."""
    if max_episodes is None and not task_limits:
        return sorted(episodes, key=lambda item: item.episode_id), {}
    if max_episodes is not None and max_episodes < 1:
        raise ValueError("--max-episodes must be greater than 0")

    grouped_tasks: dict[str, dict[PurePosixPath, list[Episode]]] = {}
    for episode in episodes:
        grouped_tasks.setdefault(_episode_task_name(episode), {}).setdefault(
            _episode_group_key(episode), []
        ).append(episode)

    rng = np.random.default_rng(seed)
    selected: list[Episode] = []
    details: dict[str, dict[str, int]] = {}
    for task_name in sorted(grouped_tasks, key=str.casefold):
        groups = grouped_tasks[task_name]
        group_items = []
        for group_key, values in groups.items():
            group_items.append(
                (str(group_key), sorted(values, key=_perturbation_number))
            )
        rng.shuffle(group_items)
        for _group_key, values in group_items:
            rng.shuffle(values)
        candidates = sum(len(values) for _key, values in group_items)
        limit = (task_limits or {}).get(task_name.casefold(), max_episodes)
        target = candidates if limit is None else min(limit, candidates)
        task_selected: list[Episode] = []
        round_index = 0
        while len(task_selected) < target:
            progressed = False
            for _group_key, values in group_items:
                if round_index < len(values):
                    task_selected.append(values[round_index])
                    progressed = True
                    if len(task_selected) >= target:
                        break
            if not progressed:
                break
            round_index += 1
        selected.extend(task_selected)
        details[task_name] = {
            "discovered": candidates,
            "placements": len(group_items),
            "selected": len(task_selected),
            "discarded": candidates - len(task_selected),
        }
    return selected, details


def _load_task_filter(path: str | None) -> tuple[set[str], dict[str, int]]:
    if not path:
        return set(), {}
    config = yaml.safe_load(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Task filter YAML must map task names to ex or positive integers")
    excluded, limits = set(), {}
    for name, value in config.items():
        task = str(name).strip().casefold()
        if not task:
            continue
        if isinstance(value, str) and value.strip().casefold() in {"ex", "exclude"}:
            excluded.add(task)
        elif isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"Invalid task filter value: {name}: {value!r} (expected ex or a positive integer)")
        else:
            limits[task] = value
    return excluded, limits


def _discover_episodes(
    root: Path | dict[PurePosixPath, SourceFile],
) -> list[Episode]:
    """Use perturbed ALOHA subjects where available, otherwise baseline episodes.

    Unperturbed ALOHA videos provide the demonstration. Archives containing only
    subject variants use all matching variant videos as their condition set.
    """
    sources = _catalog_input(root) if isinstance(root, Path) else root
    condition_index = _index_condition_sources(sources, ROBOT_DIR_NAME)
    subjects: dict[PurePosixPath, list[tuple[PurePosixPath, SourceFile]]] = {}
    pattern = re.compile(rf"{re.escape(ROBOT_DIR_NAME)}__pert\d+")
    for logical_path, source in sources.items():
        if (
            logical_path.suffix.lower() not in {".hdf5", ".h5"}
            or logical_path.parent.name != "data"
        ):
            continue
        robot_dir = logical_path.parent.parent
        if robot_dir.name == ROBOT_DIR_NAME or pattern.fullmatch(robot_dir.name):
            subjects.setdefault(robot_dir.parent, []).append((logical_path, source))
    episodes = []
    for task_dir, candidates in subjects.items():
        variants = [
            (path, source)
            for path, source in candidates
            if pattern.fullmatch(path.parent.parent.name)
        ]
        selected = variants or candidates
        expected: dict[int, set[str]] = {}
        for path, source in selected:
            expected.setdefault(_episode_number(source), set()).add(
                path.parent.parent.name
            )
        for logical_path, source in selected:
            robot_dir = logical_path.parent.parent
            ep_num = _episode_number(source)
            conditions = condition_index.get((task_dir, ep_num), [])
            available = {name for name, _ in conditions}
            complete = available == {ROBOT_DIR_NAME} or available == expected[ep_num]
            condition_paths = tuple(video for _, video in conditions)
            episodes.append(
                Episode(
                    path=source,
                    condition_path=condition_paths[0] if condition_paths else None,
                    instruction_path=sources.get(
                        robot_dir / "instructions" / f"episode{ep_num}.json"
                    ),
                    episode_id=_episode_id(source, logical_path),
                    condition_paths=condition_paths,
                    condition_complete=complete,
                )
            )
    if not episodes:
        raise FileNotFoundError(f"No ALOHA episodes found under {root}")
    return sorted(episodes, key=lambda item: item.episode_id)


def _index_condition_sources(
    sources: dict[PurePosixPath, SourceFile],
    condition_robot: str,
) -> dict[tuple[PurePosixPath, int], list[tuple[str, SourceFile]]]:
    """Index exact or numbered condition videos once for all episodes."""
    pattern = re.compile(rf"^{re.escape(condition_robot)}__pert(\d+)$")
    exact: dict[tuple[PurePosixPath, int], SourceFile] = {}
    variants: dict[tuple[PurePosixPath, int], list[tuple[int, str, SourceFile]]] = {}
    for logical_path, source in sources.items():
        if logical_path.suffix.lower() != ".mp4" or logical_path.parent.name != "video":
            continue
        episode_match = re.fullmatch(
            r"episode(\d+)\.mp4", logical_path.name, flags=re.IGNORECASE
        )
        if episode_match is None:
            continue
        episode_number = int(episode_match.group(1))
        condition_dir = logical_path.parent.parent
        key = (condition_dir.parent, episode_number)
        if condition_dir.name == condition_robot:
            exact[key] = source
            continue
        match = pattern.fullmatch(condition_dir.name)
        if match is not None:
            variants.setdefault(key, []).append(
                (int(match.group(1)), condition_dir.name, source)
            )
    result = {key: [(condition_robot, source)] for key, source in exact.items()}
    for key, items in variants.items():
        if key not in result:
            result[key] = [(item[1], item[2]) for item in sorted(items)]
    return result


def _select_episode_variants(episodes: list[Episode], seed: int) -> list[Episode]:
    """Select one visual episode for each ALOHA subject directory."""
    grouped: dict[PurePosixPath, list[Episode]] = {}
    for episode in episodes:
        robot_dir = _source_logical_path(episode.path).parent.parent
        grouped.setdefault(robot_dir, []).append(episode)

    rng = np.random.default_rng(seed)
    selected = []
    for variants in grouped.values():
        raw_paths = sorted({_source_logical_path(episode.path) for episode in variants})
        selected_path = (
            raw_paths[int(rng.integers(len(raw_paths)))]
            if len(raw_paths) > 1
            else raw_paths[0]
        )
        selected.extend(
            episode
            for episode in variants
            if _source_logical_path(episode.path) == selected_path
        )
    return sorted(selected, key=lambda item: item.episode_id)


def _instruction(
    episode: Episode,
    description_type: str,
    index: int,
    rng: np.random.Generator | None,
) -> str | None:
    try:
        data = json.loads(_read_source_text(episode.instruction_path))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    values = data.get(description_type)
    if not isinstance(values, list) or not values:
        return None
    if rng is not None:
        selected = values[int(rng.integers(0, len(values)))]
    else:
        selected = values[index % len(values)]
    if not isinstance(selected, str) or not selected.strip():
        return None
    return selected.strip()


def _decode_image(value: bytes | np.bytes_) -> np.ndarray:
    raw = bytes(value).rstrip(b"\x00")
    if not raw:
        raise ValueError("Encountered an empty JPEG frame")
    with Image.open(BytesIO(raw)) as image:
        image = image.convert("RGB")
        if image.size != IMAGE_SIZE:
            image = image.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.uint8).copy()


def _scalar_stats(values: np.ndarray, *, include_quantiles: bool = False) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    result = {
        "min": array.min(axis=0).tolist(),
        "max": array.max(axis=0).tolist(),
        "mean": array.mean(axis=0).tolist(),
        "std": array.std(axis=0).tolist(),
        "count": [int(array.shape[0])],
    }
    if include_quantiles:
        result["q01"] = np.quantile(array, 0.01, axis=0).tolist()
        result["q99"] = np.quantile(array, 0.99, axis=0).tolist()
    return result


def _open_video(path: Path, fps: int, codec: str, width: int, height: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(path), mode="w")
    stream = container.add_stream(codec, rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    return container, stream


def _flush_video(container, stream) -> None:
    try:
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()


def _load_state(ep: h5py.File, expected_dim: int = 14) -> np.ndarray:
    if expected_dim != 14:
        raise ValueError(f"ALOHA state dimension must be 14; got {expected_dim}")
    arm_dim = (expected_dim - 2) // 2
    if all(key in ep for key in JOINT_COMPONENT_KEYS):
        left_arm = np.asarray(ep["/joint_action/left_arm"][()])
        left_gripper = _as_joint_column(
            ep["/joint_action/left_gripper"][()], "left_gripper"
        )
        right_arm = np.asarray(ep["/joint_action/right_arm"][()])
        right_gripper = _as_joint_column(
            ep["/joint_action/right_gripper"][()], "right_gripper"
        )
        if (
            left_arm.ndim != 2
            or right_arm.ndim != 2
            or left_arm.shape[1] != arm_dim
            or right_arm.shape[1] != arm_dim
        ):
            raise ValueError(
                f"Dual-arm states must have shape [N, {arm_dim}]; got {left_arm.shape} and {right_arm.shape}"
            )
        lengths = [len(left_arm), len(left_gripper), len(right_arm), len(right_gripper)]
        if len(set(lengths)) != 1:
            raise ValueError(f"Dual-arm state lengths do not match: {lengths}")
        state = np.concatenate(
            [left_arm, left_gripper, right_arm, right_gripper], axis=1
        ).astype(np.float32)
    elif "/joint_action/vector" in ep:
        state = np.asarray(ep["/joint_action/vector"][()]).astype(np.float32)
    else:
        raise KeyError("Missing joint_action dual-arm state fields")
    if state.ndim != 2 or state.shape[1] != expected_dim:
        raise ValueError(f"Expected {expected_dim}-dimensional dual-arm joint states; got {state.shape}")
    return state


def _as_joint_column(value: np.ndarray, name: str) -> np.ndarray:
    """Normalize scalar gripper datasets stored as [N] or [N, 1]."""
    array = np.asarray(value)
    if array.ndim == 1:
        return array[:, None]
    if array.ndim == 2 and array.shape[1] == 1:
        return array
    raise ValueError(f"{name} state must have shape [N] or [N, 1]; got {array.shape}")


def _state_length(ep: h5py.File, expected_dim: int = 14) -> int:
    """Read the state length without loading all joint values."""
    if expected_dim != 14:
        raise ValueError(f"ALOHA state dimension must be 14; got {expected_dim}")
    if all(key in ep for key in JOINT_COMPONENT_KEYS):
        lengths = [int(ep[key].shape[0]) for key in JOINT_COMPONENT_KEYS]
        if len(set(lengths)) != 1:
            raise ValueError(f"Dual-arm state lengths do not match: {lengths}")
        left_shape = ep[JOINT_COMPONENT_KEYS[0]].shape
        right_shape = ep[JOINT_COMPONENT_KEYS[2]].shape
        arm_dim = (expected_dim - 2) // 2
        if (
            len(left_shape) != 2
            or len(right_shape) != 2
            or left_shape[1] != arm_dim
            or right_shape[1] != arm_dim
        ):
            raise ValueError(
                f"Dual-arm states must have shape [N, {arm_dim}]; got {left_shape} and {right_shape}"
            )
        for key in JOINT_COMPONENT_KEYS[1::2]:
            shape = ep[key].shape
            if len(shape) not in (1, 2) or (len(shape) == 2 and shape[1] != 1):
                raise ValueError(f"{key} state must have shape [N] or [N, 1]; got {shape}")
        return lengths[0]
    if "/joint_action/vector" in ep:
        vector = ep["/joint_action/vector"]
        if vector.ndim != 2 or vector.shape[1] != expected_dim:
            raise ValueError(
                f"Expected {expected_dim}-dimensional dual-arm joint states; got {vector.shape}"
            )
        return int(vector.shape[0])
    raise KeyError("Missing joint_action dual-arm state fields")


def _build_dataset_stats_from_episode_stats(episode_stats: list[dict]) -> dict:
    """Aggregate scalar episode stats without retaining all transitions in RAM."""

    def field(category: str) -> dict:
        values = [item[category] for item in episode_stats]
        counts = np.asarray([item["count"][0] for item in values], dtype=np.float64)
        total = counts.sum()
        means = np.asarray([item["mean"] for item in values], dtype=np.float64)
        stds = np.asarray([item["std"] for item in values], dtype=np.float64)
        mean = (means * counts[:, None]).sum(axis=0) / total
        variance = ((stds**2 + (means - mean) ** 2) * counts[:, None]).sum(
            axis=0
        ) / total
        minimum = np.min(
            np.asarray([item["min"] for item in values], dtype=np.float64), axis=0
        )
        maximum = np.max(
            np.asarray([item["max"] for item in values], dtype=np.float64), axis=0
        )
        q01 = np.min(
            np.asarray(
                [item.get("q01", item["min"]) for item in values], dtype=np.float64
            ),
            axis=0,
        )
        q99 = np.max(
            np.asarray(
                [item.get("q99", item["max"]) for item in values], dtype=np.float64
            ),
            axis=0,
        )
        result = {
            "min": minimum.tolist(),
            "max": maximum.tolist(),
            "q01": q01.tolist(),
            "q99": q99.tolist(),
            "mean": mean.tolist(),
            "std": np.sqrt(np.maximum(variance, 0)).tolist(),
        }
        # Current SimpleICL configs use global_* with z-score normalization.
        return {
            f"{prefix}_{name}": value
            for prefix in ("stepwise", "global")
            for name, value in result.items()
        }

    return {
        "state": {"default": field("observation.state")},
        "action": {"default": field("action")},
        "num_episodes": len(episode_stats),
        "num_transition": int(
            sum(item["observation.state"]["count"][0] for item in episode_stats)
        ),
    }


def _copy_condition(
    condition: SourceFile,
    output_dir: Path,
    episode_index: int,
    condition_index: int = 0,
    condition_count: int = 1,
) -> Path:
    if _source_size(condition) <= 0:
        raise ValueError(f"Condition video is empty: {condition}")
    if condition_count == 1:
        target = output_dir / "conditions" / f"episode_{episode_index:06d}.mp4"
    else:
        target = (
            output_dir
            / "conditions"
            / f"episode_{episode_index:06d}"
            / f"condition_{condition_index:02d}.mp4"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    try:
        with _open_source(condition) as source, temporary.open("wb") as destination:
            shutil.copyfileobj(source, destination, length=1024 * 1024)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    if not _video_is_readable(target, decode_frames=True):
        raise ValueError(f"Copied condition video is invalid or cannot be fully decoded: {target}")
    return target.resolve()


def _copy_conditions(
    conditions: tuple[SourceFile, ...],
    output_dir: Path,
    episode_index: int,
) -> list[Path]:
    if not conditions:
        raise ValueError("Missing condition video")
    return [
        _copy_condition(condition, output_dir, episode_index, index, len(conditions))
        for index, condition in enumerate(conditions)
    ]


def _condition_values(value: object) -> list[str]:
    """Normalize legacy condition metadata and the new list representation."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str) and item]
    return []


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    items = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                # A killed final metadata write may leave one truncated line.
                continue
    return items


def _resolve_output_path(value: str, output_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (output_dir / path).resolve()


def _video_is_readable(
    path: SourceFile,
    *,
    decode_frames: bool = False,
    expected_frames: int | None = None,
) -> bool:
    """Check that a video has a stream and, when requested, decodes frames."""
    try:
        if _source_size(path) <= 0:
            return False
        with _open_source(path) as source:
            with av.open(source) as container:
                streams = container.streams.video
                if not streams:
                    return False
                if decode_frames:
                    stream = streams[0]
                    frame_count = sum(1 for _frame in container.decode(video=0))
                    declared_count = int(stream.frames or 0)
                    if frame_count <= 0 or (
                        declared_count > 0 and frame_count != declared_count
                    ):
                        return False
                    return expected_frames is None or frame_count == expected_frames
                return True
    except Exception:
        return False


def _video_info(path: Path, fps: int) -> dict:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        return {
            "video.height": stream.height,
            "video.width": stream.width,
            "video.codec": stream.codec_context.name,
            "video.pix_fmt": stream.codec_context.pix_fmt,
            "video.is_depth_map": False,
            "video.fps": int(stream.average_rate or fps),
            "video.channels": 3,
            "has_audio": False,
        }


def _parquet_row_count(path: Path) -> int:
    """Read Parquet metadata without materializing episode columns."""
    return int(parquet.ParquetFile(path).metadata.num_rows)


def _load_resume_metadata(output_dir: Path) -> dict:
    """Load prior metadata used to decide which episodes can be reused."""
    meta_dir = output_dir / "meta"
    episodes = {
        int(item["episode_index"]): item
        for item in _read_jsonl(meta_dir / "episodes.jsonl")
        if "episode_index" in item and "raw_episode_id" in item
    }
    stats = {
        int(item["episode_index"]): item["stats"]
        for item in _read_jsonl(meta_dir / "episodes_stats.jsonl")
        if "episode_index" in item and "stats" in item
    }
    conditions_path = meta_dir / "conditions.json"
    conditions = (
        json.loads(conditions_path.read_text(encoding="utf-8"))
        if conditions_path.is_file()
        else {}
    )
    info_path = meta_dir / "info.json"
    info = (
        json.loads(info_path.read_text(encoding="utf-8")) if info_path.is_file() else {}
    )
    tasks = {
        int(item["task_index"]): str(item["task"])
        for item in _read_jsonl(meta_dir / "tasks.jsonl")
        if "task_index" in item and "task" in item
    }
    progress_dir = meta_dir / ".conversion_progress"
    for progress_path in (
        sorted(progress_dir.glob("episode_*.json")) if progress_dir.is_dir() else ()
    ):
        try:
            result = json.loads(progress_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        episode_index = result.get("episode_index")
        if episode_index is None or episode_index in episodes or result.get("skipped"):
            continue
        episodes[int(episode_index)] = {
            "episode_index": int(episode_index),
            "tasks": [result["instruction"]],
            "length": int(result["length"]),
            "raw_file_name": result["raw_file_name"],
            "raw_episode_id": result["raw_episode_id"],
        }
        stats[int(episode_index)] = result["stats"]
        condition_value = result.get("condition_paths", result.get("condition_path"))
        if condition_value is not None:
            conditions[str(episode_index)] = condition_value
        if "instruction" in result and "task_index" in result:
            tasks.setdefault(int(result["task_index"]), str(result["instruction"]))
    return {
        "episodes": episodes,
        "episodes_by_id": {
            item["raw_episode_id"]: (int(index), item)
            for index, item in episodes.items()
        },
        "stats": stats,
        "conditions": conditions,
        "info": info,
        "tasks": tasks,
    }


def _write_progress_result(output_dir: Path, result: dict) -> None:
    if result.get("skipped"):
        return
    progress_dir = output_dir / "meta" / ".conversion_progress"
    progress_dir.mkdir(parents=True, exist_ok=True)
    path = progress_dir / f"episode_{int(result['episode_index']):06d}.json"
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _reusable_episode(
    task: dict,
    metadata: dict,
    output_dir: Path,
) -> dict | None:
    """Return an existing result only when all episode artifacts are readable."""
    episode_index = int(task["episode_index"])
    episode = task["episode"]
    stable_id = episode.episode_id
    stored = metadata.get("episodes_by_id", {}).get(stable_id)
    if stored is None:
        return None
    episode_index, item = stored
    if metadata.get("info", {}).get("fps") not in (None, int(task["fps"])):
        return None
    stats = metadata["stats"].get(episode_index)
    condition_value = metadata["conditions"].get(str(episode_index))
    if not item or stats is None or condition_value is None:
        return None
    if item["raw_episode_id"] != stable_id:
        return None
    if not item.get("tasks") or item["tasks"][0] != task["instruction"]:
        return None
    frame_count = int(item.get("length", 0))
    if frame_count != int(task["frame_count"]):
        return None

    parquet_path = (
        output_dir
        / f"data/chunk-{episode_index // CHUNKS_SIZE:03d}/episode_{episode_index:06d}.parquet"
    )
    if not parquet_path.is_file():
        return None
    try:
        row_count = _parquet_row_count(parquet_path)
    except Exception:
        return None
    if row_count != frame_count:
        return None

    video_info = {}
    configured_features = metadata["info"].get("features", {})
    for camera in CAMERA_NAMES.values():
        key = f"observation.images.{camera}"
        video_path = (
            output_dir
            / f"videos/chunk-{episode_index // CHUNKS_SIZE:03d}/{key}/episode_{episode_index:06d}.mp4"
        )
        if not video_path.is_file():
            return None
        try:
            video_info[key] = configured_features.get(key, {}).get(
                "info"
            ) or _video_info(video_path, int(task["fps"]))
            if not _video_is_readable(
                video_path, decode_frames=True, expected_frames=frame_count
            ):
                return None
        except Exception:
            return None

    stored_conditions = _condition_values(condition_value)
    source_conditions = _episode_condition_paths(episode)
    if not stored_conditions or len(stored_conditions) != len(source_conditions):
        return None
    expected_conditions = [
        output_dir / f"conditions/episode_{episode_index:06d}.mp4"
        if len(source_conditions) == 1
        else output_dir
        / f"conditions/episode_{episode_index:06d}"
        / f"condition_{index:02d}.mp4"
        for index in range(len(source_conditions))
    ]
    for stored_path, expected_path, source_condition in zip(
        stored_conditions, expected_conditions, source_conditions
    ):
        condition_path = _resolve_output_path(stored_path, output_dir)
        if condition_path != expected_path:
            return None
        # Condition files are copied separately and can be truncated by an
        # interrupted conversion. Decode them fully before reusing the episode.
        if not _video_is_readable(condition_path, decode_frames=True):
            return None
        if not _video_is_readable(source_condition, decode_frames=True):
            return None
        try:
            if _source_size(source_condition) != condition_path.stat().st_size:
                return None
        except OSError:
            return None
    task_index = int(stats.get("task_index", {}).get("min", [task["task_index"]])[0])
    return {
        "skipped": False,
        "reused": True,
        "episode_index": episode_index,
        "instruction": item["tasks"][0],
        "task_index": task_index,
        "length": frame_count,
        "raw_file_name": str(episode.path),
        "raw_episode_id": stable_id,
        "stats": stats,
        "condition_paths": stored_conditions,
        "condition_path": stored_conditions[0],
        "parquet_path": str(parquet_path),
        "video_info": video_info,
    }


def _convert_episode_impl(task: dict) -> dict:
    """Convert one episode; this function is process-pool safe."""
    episode: Episode = task["episode"]
    output_dir = Path(task["output_dir"])
    episode_index = int(task["episode_index"])
    ep_chunk = episode_index // CHUNKS_SIZE
    fps = int(task["fps"])
    video_codec = str(task["video_codec"])
    with _open_source(episode.path) as source, h5py.File(source, "r") as ep:
        state_all = _load_state(ep, int(task.get("state_dim", 14)))
        frame_count = state_all.shape[0] - 1
        if frame_count < 1:
            return {
                "skipped": True,
                "episode": str(episode.path),
                "episode_index": episode_index,
                "reason": "Episode is too short",
            }
        image_sets = {
            output_name: ep[f"/observation/{raw_name}/rgb"]
            for raw_name, output_name in CAMERA_NAMES.items()
        }
        if any(len(images) != state_all.shape[0] for images in image_sets.values()):
            raise ValueError(f"State and camera frame counts do not match: {episode.path}")
        states = state_all[:-1]
        actions = state_all[1:]
        if not np.isfinite(states).all() or not np.isfinite(actions).all():
            raise ValueError(f"state/action contains non-finite values: {episode.path}")

        video_handles = {}
        image_stats = {}
        video_paths = {}
        try:
            for camera in CAMERA_NAMES.values():
                key = f"observation.images.{camera}"
                path = (
                    output_dir
                    / f"videos/chunk-{ep_chunk:03d}/{key}/episode_{episode_index:06d}.mp4"
                )
                video_handles[camera] = _open_video(path, fps, video_codec, *IMAGE_SIZE)
                video_paths[camera] = path
                image_stats[camera] = ImageStats()
            for frame_index in range(frame_count):
                for raw_name, camera in CAMERA_NAMES.items():
                    image = _decode_image(image_sets[camera][frame_index])
                    image_stats[camera].update(image)
                    container, stream = video_handles[camera]
                    video_frame = av.VideoFrame.from_ndarray(image, format="rgb24")
                    for packet in stream.encode(video_frame):
                        container.mux(packet)
        finally:
            for container, stream in video_handles.values():
                _flush_video(container, stream)

    frame_indices = np.arange(frame_count, dtype=np.int64)
    timestamps = (frame_indices / fps).astype(np.float32)
    global_start = int(task["global_start"])
    global_indices = np.arange(global_start, global_start + frame_count, dtype=np.int64)
    data = pd.DataFrame(
        {
            "observation.state": list(states),
            "action": list(actions),
            "timestamp": timestamps,
            "frame_index": frame_indices,
            "episode_index": np.full(frame_count, episode_index, dtype=np.int64),
            "index": global_indices,
            "task_index": np.full(frame_count, int(task["task_index"]), dtype=np.int64),
        }
    )
    parquet_path = (
        output_dir / f"data/chunk-{ep_chunk:03d}/episode_{episode_index:06d}.parquet"
    )
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(parquet_path, engine="pyarrow", index=False)
    try:
        if _parquet_row_count(parquet_path) != frame_count:
            raise ValueError(f"Parquet frame count mismatch: {parquet_path}")
    except Exception as error:
        raise ValueError(f"Invalid Parquet file: {parquet_path}: {error}") from error

    ep_stats = {
        f"observation.images.{camera}": image_stats[camera].as_json(frame_count)
        for camera in CAMERA_NAMES.values()
    }
    ep_stats.update(
        {
            "observation.state": _scalar_stats(states, include_quantiles=True),
            "action": _scalar_stats(actions, include_quantiles=True),
            "timestamp": _scalar_stats(timestamps),
            "frame_index": _scalar_stats(frame_indices),
            "index": _scalar_stats(global_indices),
            "episode_index": {
                "min": [episode_index],
                "max": [episode_index],
                "mean": [episode_index],
                "std": [0.0],
                "count": [frame_count],
            },
            "task_index": {
                "min": [int(task["task_index"])],
                "max": [int(task["task_index"])],
                "mean": [float(task["task_index"])],
                "std": [0.0],
                "count": [frame_count],
            },
        }
    )

    condition_paths = _copy_conditions(
        _episode_condition_paths(episode), output_dir, episode_index
    )
    video_info = {}
    for camera, path in video_paths.items():
        if not _video_is_readable(
            path, decode_frames=True, expected_frames=frame_count
        ):
            raise ValueError(f"Output video is invalid or cannot be fully decoded: {path}")
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            video_info[f"observation.images.{camera}"] = {
                "video.height": stream.height,
                "video.width": stream.width,
                "video.codec": stream.codec_context.name,
                "video.pix_fmt": stream.codec_context.pix_fmt,
                "video.is_depth_map": False,
                "video.fps": int(stream.average_rate or fps),
                "video.channels": 3,
                "has_audio": False,
            }
    return {
        "skipped": False,
        "episode_index": episode_index,
        "instruction": task["instruction"],
        "task_index": int(task["task_index"]),
        "length": frame_count,
        "raw_file_name": str(episode.path),
        "raw_episode_id": episode.episode_id,
        "stats": ep_stats,
        "condition_paths": [str(path) for path in condition_paths],
        "condition_path": str(condition_paths[0]),
        "parquet_path": str(parquet_path),
        "video_info": video_info,
    }


def _cleanup_episode_artifacts(task: dict) -> None:
    """Remove partial outputs so skipped episodes are never referenced later."""
    output_dir = Path(task["output_dir"])
    episode_index = int(task["episode_index"])
    paths = _episode_artifact_paths(
        output_dir,
        episode_index,
        condition_count=len(_episode_condition_paths(task["episode"])),
    )
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def _episode_artifact_paths(
    output_dir: Path,
    episode_index: int,
    condition_count: int = 1,
) -> list[Path]:
    chunk = episode_index // CHUNKS_SIZE
    paths = [
        output_dir / f"data/chunk-{chunk:03d}/episode_{episode_index:06d}.parquet",
    ]
    if condition_count <= 1:
        paths.append(output_dir / f"conditions/episode_{episode_index:06d}.mp4")
    else:
        paths.extend(
            output_dir
            / f"conditions/episode_{episode_index:06d}/condition_{index:02d}.mp4"
            for index in range(condition_count)
        )
    paths.extend(
        output_dir
        / f"videos/chunk-{chunk:03d}/observation.images.{camera}/episode_{episode_index:06d}.mp4"
        for camera in CAMERA_NAMES.values()
    )
    return paths


def _compact_successful_results(
    conversion_tasks: list[dict],
    results: dict[int, dict],
    output_dir: Path,
) -> None:
    """Remove episode-index gaps created when a worker skips a bad episode."""
    successful = [
        results[task["episode_index"]]
        for task in conversion_tasks
        if not results[task["episode_index"]].get("skipped")
    ]
    if all(
        int(result["episode_index"]) == index for index, result in enumerate(successful)
    ):
        return
    next_index = 0
    global_start = 0
    moves: list[tuple[int, dict, list[Path], list[Path]]] = []
    for result in successful:
        old_index = int(result["episode_index"])
        new_index = next_index
        if old_index != new_index:
            condition_count = len(
                _condition_values(
                    result.get("condition_paths", result.get("condition_path"))
                )
            )
            old_paths = _episode_artifact_paths(output_dir, old_index, condition_count)
            new_paths = _episode_artifact_paths(output_dir, new_index, condition_count)
            temporary_paths = []
            for source in old_paths:
                temporary = source.with_name(
                    f".{source.name}.compact-{os.getpid()}-{old_index}"
                )
                os.replace(source, temporary)
                temporary_paths.append(temporary)
            moves.append((new_index, result, temporary_paths, new_paths))
        else:
            moves.append((new_index, result, [], []))
        next_index += 1

    for new_index, result, temporary_paths, new_paths in moves:
        for temporary, target in zip(temporary_paths, new_paths):
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(temporary, target)
        frame_count = int(result["length"])
        parquet_path = (
            output_dir
            / f"data/chunk-{new_index // CHUNKS_SIZE:03d}/episode_{new_index:06d}.parquet"
        )
        data = pd.read_parquet(parquet_path)
        data["episode_index"] = np.full(frame_count, new_index, dtype=np.int64)
        data["index"] = np.arange(
            global_start, global_start + frame_count, dtype=np.int64
        )
        data.to_parquet(parquet_path, engine="pyarrow", index=False)
        result["episode_index"] = new_index
        result["parquet_path"] = str(parquet_path)
        condition_count = len(
            _condition_values(
                result.get("condition_paths", result.get("condition_path"))
            )
        )
        if condition_count <= 1:
            condition_paths = [output_dir / f"conditions/episode_{new_index:06d}.mp4"]
        else:
            condition_paths = [
                output_dir
                / f"conditions/episode_{new_index:06d}/condition_{index:02d}.mp4"
                for index in range(condition_count)
            ]
        result["condition_paths"] = [str(path) for path in condition_paths]
        result["condition_path"] = str(condition_paths[0])
        result["stats"]["episode_index"] = {
            "min": [new_index],
            "max": [new_index],
            "mean": [new_index],
            "std": [0.0],
            "count": [frame_count],
        }
        result["stats"]["index"] = _scalar_stats(
            np.arange(global_start, global_start + frame_count, dtype=np.int64)
        )
        global_start += frame_count


def _compact_task_indices(
    conversion_tasks: list[dict],
    results: dict[int, dict],
    output_dir: Path,
) -> dict[str, int]:
    """Drop tasks belonging only to skipped episodes and close task-index gaps."""
    successful = [
        results[task["episode_index"]]
        for task in conversion_tasks
        if not results[task["episode_index"]].get("skipped")
    ]
    used = sorted({int(result["task_index"]) for result in successful})
    remap = {old: new for new, old in enumerate(used)}
    for result in successful:
        old = int(result["task_index"])
        new = remap[old]
        if old != new:
            data = pd.read_parquet(result["parquet_path"])
            data["task_index"] = np.full(len(data), new, dtype=np.int64)
            data.to_parquet(result["parquet_path"], engine="pyarrow", index=False)
            count = int(result["length"])
            result["stats"]["task_index"] = {
                "min": [new],
                "max": [new],
                "mean": [float(new)],
                "std": [0.0],
                "count": [count],
            }
        result["task_index"] = new
    return {result["instruction"]: int(result["task_index"]) for result in successful}


def _convert_episode(task: dict) -> dict:
    """Convert one episode and turn any bad input/output into a skip result."""
    try:
        result = _convert_episode_impl(task)
        if result.get("skipped"):
            _cleanup_episode_artifacts(task)
        return result
    except Exception as error:
        _cleanup_episode_artifacts(task)
        return {
            "skipped": True,
            "episode": str(task["episode"].path),
            "episode_index": int(task["episode_index"]),
            "reason": _error_category(error),
            "detail": str(error),
        }


def _error_category(error: BaseException) -> str:
    text = str(error).lower()
    if "instruction" in text:
        return "instruction"
    if "condition" in text:
        return "condition"
    if "jpeg" in text or "image" in text or "rgb" in text:
        return "camera image"
    if "parquet" in text or "pyarrow" in text:
        return "parquet"
    if "video" in text or "codec" in text or isinstance(error, av.FFmpegError):
        return "video"
    if (
        "hdf5" in text
        or "joint" in text
        or "state" in text
        or isinstance(error, (OSError, KeyError))
    ):
        return "hdf5/state"
    return type(error).__name__


def _write_conversion_summary(
    output_dir: Path,
    *,
    discovered: int,
    candidates: int,
    success: int,
    reused: int,
    skipped: list[dict],
    excluded_tasks: dict[str, int] | None = None,
    task_episode_limits: dict[str, dict[str, int]] | None = None,
) -> None:
    summary = {}
    for item in skipped:
        reason = item.get("reason", "unknown")
        summary[reason] = summary.get(reason, 0) + 1
    payload = {
        "discovered": discovered,
        "candidates": candidates,
        "success": success,
        "reused": reused,
        "skipped": len(skipped),
        "excluded_tasks": dict(sorted((excluded_tasks or {}).items())),
        "task_episode_limits": dict(sorted((task_episode_limits or {}).items())),
        "skip_reasons": dict(sorted(summary.items())),
        "skipped_episodes": [
            {
                key: item[key]
                for key in ("episode", "episode_index", "reason", "detail")
                if key in item
            }
            for item in skipped
        ],
    }
    summary_path = output_dir / "meta" / "conversion_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[summary] discovered={discovered} candidates={candidates} success={success} "
        f"reused={reused} skipped={len(skipped)}",
        flush=True,
    )
    if summary:
        details = ", ".join(
            f"{reason}={count}" for reason, count in sorted(summary.items())
        )
        print(f"[summary] skip reasons: {details}", flush=True)
    print(f"[summary] details: {summary_path}", flush=True)


def convert(args: argparse.Namespace) -> None:
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    if args.fps < 1:
        raise ValueError("--fps must be greater than 0")
    if args.max_episodes is not None and args.max_episodes < 1:
        raise ValueError("--max-episodes must be greater than 0")
    if args.workers < 0:
        raise ValueError("--workers must not be less than 0")
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite cannot be used together")
    if (
        input_path == output_dir
        or (input_path.exists() and input_path.is_relative_to(output_dir))
        or (input_path.is_dir() and output_dir.is_relative_to(input_path))
    ):
        raise ValueError(
            "--output must not equal or contain --input, to prevent --overwrite from deleting source data"
        )
    if output_dir.exists():
        if args.resume:
            if not output_dir.is_dir():
                raise NotADirectoryError(f"--resume requires the output path to be a directory: {output_dir}")
        elif not args.overwrite:
            raise FileExistsError(f"Output directory already exists; pass --overwrite: {output_dir}")
        else:
            shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.extract_dir:
        print("[stream] --extract-dir is no longer needed; tar contents are not extracted to disk", flush=True)

    def _run_conversion() -> None:
        source_catalog = _catalog_input(input_path)
        local_count = sum(
            isinstance(source, Path) for source in source_catalog.values()
        )
        archive_count = len(source_catalog) - local_count
        print(
            f"[stream] indexed local={local_count} archive_members={archive_count}",
            flush=True,
        )
        state_dim = 14
        discovered_episodes = _discover_episodes(source_catalog)
        selected_episodes = _select_episode_variants(discovered_episodes, args.seed)
        discovered_raw = len({episode.path for episode in discovered_episodes})
        selected_raw = len({episode.path for episode in selected_episodes})
        discarded_raw = discovered_raw - selected_raw
        discarded_episodes = len(discovered_episodes) - len(selected_episodes)
        if discarded_raw:
            print(
                f"[discover] randomly selected one episode per task placement and robot arm "
                f"with seed={args.seed}; raw={discovered_raw}->{selected_raw} "
                f"discarded={discarded_raw}; subject episodes={len(discovered_episodes)}"
                f"->{len(selected_episodes)} discarded={discarded_episodes}",
                flush=True,
            )
        yaml_excluded, yaml_limits = _load_task_filter(
            getattr(args, "task_filter_yaml", None)
        )
        excluded_tasks = (
            _normalize_task_names(getattr(args, "exclude_tasks", None)) | yaml_excluded
        )
        episodes, excluded_counts = _filter_episodes_by_task(
            selected_episodes, excluded_tasks
        )
        if excluded_counts:
            excluded_detail = ", ".join(
                f"{name}={count}" for name, count in sorted(excluded_counts.items())
            )
            print(f"[discover] excluded tasks: {excluded_detail}", flush=True)
        episodes, task_episode_limits = _limit_aloha_episodes_round_robin(
            episodes,
            args.max_episodes,
            args.seed,
            yaml_limits,
        )
        limited_tasks = {
            name: detail
            for name, detail in task_episode_limits.items()
            if detail["discarded"]
        }
        if limited_tasks:
            detail = ", ".join(
                f"{name}={item['discovered']}->{item['selected']}"
                for name, item in sorted(limited_tasks.items())
            )
            mode = "round-robin perturbation limit"
            print(f"[discover] {mode}={args.max_episodes}: {detail}", flush=True)
        if not episodes:
            if excluded_tasks:
                raise ValueError(
                    "No episodes available for conversion after applying --exclude-tasks or --max-episodes"
                )
            raise ValueError("No episodes available for conversion after applying --max-episodes")
        print(
            f"[discover] {len(episodes)} episodes under {input_path} "
            f"(discovered={len(discovered_episodes)})",
            flush=True,
        )

        rng = np.random.default_rng(args.seed) if args.random_instruction else None
        resume_metadata = (
            _load_resume_metadata(output_dir)
            if args.resume
            else {
                "episodes": {},
                "episodes_by_id": {},
                "stats": {},
                "conditions": {},
                "info": {},
                "tasks": {},
            }
        )
        if args.resume and resume_metadata["info"]:
            expected_mode = "round_robin_perturbation"
            stored_dim = int(resume_metadata["info"].get("state_dim", 14))
            stored_mode = resume_metadata["info"].get("selection_mode", "episode_limit")
            if stored_dim != state_dim or stored_mode != expected_mode:
                raise ValueError(
                    "Output data schema mismatch for --resume: "
                    f"stored state_dim={stored_dim}, selection_mode={stored_mode}; "
                    f"requested state_dim={state_dim}, selection_mode={expected_mode}"
                )
        tasks: dict[str, int] = {
            value: key for key, value in resume_metadata["tasks"].items()
        }
        episodes_meta: list[dict] = []
        episodes_stats: list[dict] = []
        condition_map: dict[str, str | list[str]] = {}
        total_frames = 0
        video_info: dict[str, dict] | None = None

        conversion_tasks = []
        skipped_results: list[dict] = []
        global_start = 0
        reserved_indices = set(resume_metadata["episodes"])
        next_episode_index = max(reserved_indices, default=-1) + 1
        for episode in episodes:
            instruction = _instruction(
                episode,
                args.description_type,
                args.instruction_index,
                rng,
            )
            if instruction is None:
                skipped_results.append(
                    {
                        "skipped": True,
                        "episode": str(episode.path),
                        "reason": "instruction",
                        "detail": str(episode.instruction_path),
                    }
                )
                continue
            try:
                with _open_source(episode.path) as source, h5py.File(source, "r") as ep:
                    frame_count = _state_length(ep, state_dim) - 1
                    missing_cameras = [
                        raw_name
                        for raw_name in CAMERA_NAMES
                        if f"/observation/{raw_name}/rgb" not in ep
                    ]
            except Exception as error:
                reason = _error_category(error)
                skipped_results.append(
                    {
                        "skipped": True,
                        "episode": str(episode.path),
                        "reason": reason,
                        "detail": str(error),
                    }
                )
                continue
            if missing_cameras:
                skipped_results.append(
                    {
                        "skipped": True,
                        "episode": str(episode.path),
                        "reason": "video",
                        "detail": f"Missing camera data: {', '.join(missing_cameras)}",
                    }
                )
                continue
            if frame_count < 1:
                skipped_results.append(
                    {
                        "skipped": True,
                        "episode": str(episode.path),
                        "reason": "Episode is too short",
                    }
                )
                continue
            if not episode.condition_complete:
                skipped_results.append(
                    {
                        "skipped": True,
                        "episode": str(episode.path),
                        "reason": "condition",
                        "detail": "Incomplete condition perturbation set for ALOHA placement",
                    }
                )
                continue
            condition_paths = _episode_condition_paths(episode)
            if not condition_paths or any(
                not _video_is_readable(condition, decode_frames=True)
                for condition in condition_paths
            ):
                skipped_results.append(
                    {
                        "skipped": True,
                        "episode": str(episode.path),
                        "reason": "condition",
                        "detail": "; ".join(
                            str(condition) for condition in condition_paths
                        )
                        or "missing condition video",
                    }
                )
                continue
            stored_episode = resume_metadata["episodes_by_id"].get(episode.episode_id)
            if stored_episode is None:
                episode_index = next_episode_index
                next_episode_index += 1
            else:
                episode_index = int(stored_episode[0])
            task_index = tasks.setdefault(instruction, len(tasks))
            conversion_tasks.append(
                {
                    "episode": episode,
                    "output_dir": str(output_dir),
                    "episode_index": episode_index,
                    "global_start": global_start,
                    "task_index": task_index,
                    "instruction": instruction,
                    "frame_count": frame_count,
                    "fps": args.fps,
                    "video_codec": args.video_codec,
                    "state_dim": state_dim,
                }
            )
            global_start += frame_count

        if not conversion_tasks:
            _write_conversion_summary(
                output_dir,
                discovered=len(discovered_episodes),
                candidates=0,
                success=0,
                reused=0,
                skipped=skipped_results,
                excluded_tasks=excluded_counts,
                task_episode_limits=task_episode_limits,
            )
            return
        requested_workers = (os.cpu_count() or 1) if args.workers == 0 else args.workers
        pending_tasks = []
        results: dict[int, dict] = {}
        reused_count = 0
        for task in conversion_tasks:
            if args.resume:
                reused = _reusable_episode(task, resume_metadata, output_dir)
                if reused is not None:
                    results[task["episode_index"]] = reused
                    tasks[reused["instruction"]] = reused["task_index"]
                    reused_count += 1
                    continue
            pending_tasks.append(task)
        worker_count = (
            min(requested_workers, len(pending_tasks)) if pending_tasks else 0
        )
        print(
            f"[convert] {len(conversion_tasks)} episodes pending={len(pending_tasks)} "
            f"reused={reused_count} workers={worker_count}",
            flush=True,
        )
        if worker_count == 1:
            for index, task in enumerate(pending_tasks, 1):
                print(
                    f"[convert] {index}/{len(conversion_tasks)} {task['episode'].path}",
                    flush=True,
                )
                result = _convert_episode(task)
                results[task["episode_index"]] = result
                _write_progress_result(output_dir, result)
                if result.get("skipped"):
                    skipped_results.append(result)
        elif worker_count > 1:
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                futures = {
                    executor.submit(_convert_episode, task): task
                    for task in pending_tasks
                }
                for completed, future in enumerate(as_completed(futures), 1):
                    task = futures[future]
                    try:
                        result = future.result()
                    except Exception as error:
                        _cleanup_episode_artifacts(task)
                        result = {
                            "skipped": True,
                            "episode": str(task["episode"].path),
                            "episode_index": int(task["episode_index"]),
                            "reason": _error_category(error),
                            "detail": str(error),
                        }
                    results[task["episode_index"]] = result
                    _write_progress_result(output_dir, result)
                    if result.get("skipped"):
                        skipped_results.append(result)
                    print(
                        f"[convert] completed {completed}/{len(futures)} episode={task['episode_index']}",
                        flush=True,
                    )

        # Validate worker outputs once more before compacting indices.  This
        # keeps a late filesystem/read failure from creating metadata gaps.
        for task in conversion_tasks:
            result = results[task["episode_index"]]
            if result.get("skipped"):
                continue
            try:
                if _parquet_row_count(Path(result["parquet_path"])) != int(
                    result["length"]
                ):
                    raise ValueError(f"Parquet frame count mismatch: {result['parquet_path']}")
            except Exception as error:
                result["skipped"] = True
                result["reason"] = _error_category(error)
                result["detail"] = str(error)
                skipped_results.append(result)
                _cleanup_episode_artifacts(task)

        _compact_successful_results(conversion_tasks, results, output_dir)
        tasks = _compact_task_indices(conversion_tasks, results, output_dir)
        for task in conversion_tasks:
            result = results[task["episode_index"]]
            if result["skipped"]:
                continue
            episode_index = result["episode_index"]
            frame_count = result["length"]
            instruction = result["instruction"]
            episodes_stats.append(
                {"episode_index": episode_index, "stats": result["stats"]}
            )
            episodes_meta.append(
                {
                    "episode_index": episode_index,
                    "tasks": [instruction],
                    "length": frame_count,
                    "raw_file_name": result["raw_file_name"],
                    "raw_episode_id": result["raw_episode_id"],
                }
            )
            condition_values = result.get("condition_paths", result["condition_path"])
            normalized_values = _condition_values(condition_values)
            condition_map[str(result["episode_index"])] = (
                normalized_values[0]
                if len(normalized_values) == 1
                else normalized_values
            )
            total_frames += frame_count
            if video_info is None:
                video_info = result["video_info"]

        for item in skipped_results:
            item.setdefault("episode_index", None)
        _write_conversion_summary(
            output_dir,
            discovered=len(discovered_episodes),
            candidates=len(conversion_tasks),
            success=len(episodes_meta),
            reused=reused_count,
            skipped=skipped_results,
            excluded_tasks=excluded_counts,
            task_episode_limits=task_episode_limits,
        )
        if not episodes_meta:
            return
        assert video_info is not None
        features = {
            key: {
                "dtype": "video",
                "shape": [IMAGE_SIZE[1], IMAGE_SIZE[0], 3],
                "names": ["height", "width", "rgb"],
                "info": video_info[key],
            }
            for key in video_info
        }
        for key, dim in (("observation.state", state_dim), ("action", state_dim)):
            features[key] = {
                "dtype": "float32",
                "shape": [dim],
                "names": {"motors": [f"motor_{i}" for i in range(dim)]},
            }
        features.update(
            {
                "timestamp": {"dtype": "float32", "shape": [1], "names": None},
                "frame_index": {"dtype": "int64", "shape": [1], "names": None},
                "episode_index": {"dtype": "int64", "shape": [1], "names": None},
                "index": {"dtype": "int64", "shape": [1], "names": None},
                "task_index": {"dtype": "int64", "shape": [1], "names": None},
            }
        )

        meta_dir = output_dir / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        info = {
            "codebase_version": "v2.1",
            "robot_type": ROBOT_DIR_NAME,
            "state_dim": state_dim,
            "selection_mode": "round_robin_perturbation",
            "total_episodes": len(episodes_meta),
            "total_frames": total_frames,
            "total_tasks": len(tasks),
            "total_videos": len(episodes_meta) * len(CAMERA_NAMES),
            "total_chunks": (len(episodes_meta) - 1) // CHUNKS_SIZE + 1,
            "chunks_size": CHUNKS_SIZE,
            "fps": args.fps,
            "splits": {"train": f"0:{len(episodes_meta)}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": features,
        }
        (meta_dir / "info.json").write_text(
            json.dumps(info, indent=2), encoding="utf-8"
        )
        with (meta_dir / "tasks.jsonl").open("w", encoding="utf-8") as handle:
            for task, task_index in sorted(tasks.items(), key=lambda item: item[1]):
                handle.write(
                    json.dumps(
                        {"task_index": task_index, "task": task}, ensure_ascii=False
                    )
                    + "\n"
                )
        with (meta_dir / "episodes.jsonl").open("w", encoding="utf-8") as handle:
            for item in episodes_meta:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        with (meta_dir / "episodes_stats.jsonl").open("w", encoding="utf-8") as handle:
            for item in episodes_stats:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        (meta_dir / "conditions.json").write_text(
            json.dumps(condition_map, indent=2), encoding="utf-8"
        )
        (output_dir / "dataset_stats.json").write_text(
            json.dumps(
                _build_dataset_stats_from_episode_stats(
                    [item["stats"] for item in episodes_stats]
                ),
                indent=2,
            ),
            encoding="utf-8",
        )
        shutil.rmtree(meta_dir / ".conversion_progress", ignore_errors=True)
        print(
            f"[done] {output_dir} episodes={len(episodes_meta)} frames={total_frames} tasks={len(tasks)}",
            flush=True,
        )

    _run_conversion()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Raw RoboTwin directory (may be a parent containing multiple split_* directories), tar file, or directory containing tar files",
    )
    parser.add_argument("--output", required=True, help="LeRobot output directory")
    parser.add_argument(
        "--extract-dir",
        default=None,
        help="Legacy compatibility option; streaming mode no longer extracts or saves tar contents",
    )
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument(
        "--video-codec",
        default="libsvtav1",
        help="PyAV codec, e.g. libsvtav1 or libx264",
    )
    parser.add_argument(
        "--description-type", choices=("seen", "unseen"), default="seen"
    )
    parser.add_argument(
        "--instruction-index",
        type=int,
        default=0,
        help="Instruction index to select for each episode",
    )
    parser.add_argument(
        "--random-instruction",
        action="store_true",
        help="Randomly select instructions using --seed",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Control episode variant selection, per-task subsampling, and random instruction selection",
    )
    parser.add_argument(
        "--exclude-tasks",
        "--exclude_tasks",
        nargs="+",
        action="extend",
        default=[],
        metavar="TASK",
        help="Exclude specified RoboTwin tasks from conversion; accepts space- or comma-separated names and repeated options",
    )
    parser.add_argument(
        "--max-episodes",
        "--max_episodes",
        type=int,
        default=None,
        help="Convert at most N ALOHA subject episodes per task; select round-robin across placements and perturbations",
    )
    parser.add_argument(
        "--task-filter-yaml",
        help="Task filter YAML: map task names to ex (exclude) or positive integers (episodes to keep per task)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of CPU processes for episode processing; 0 uses the CPU core count",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Keep the output directory and skip completed episodes"
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


if __name__ == "__main__":
    convert(build_parser().parse_args())
