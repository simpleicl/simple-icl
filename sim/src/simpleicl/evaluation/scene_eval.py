"""Fixed-scene evaluation adapter for RoboTwin scene-mode datasets.

The adapter deliberately keeps RoboTwin imports lazy.  This lets the metadata
validation helpers run in a lightweight process and avoids binding the
production RoboTwin checkout until a scene rollout is requested.
"""

from __future__ import annotations

import importlib
import hashlib
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence
import re
import random

import numpy as np
import yaml

from .task_names import (
    public_scene_metadata,
    public_task_name,
    sampling_scene_name,
    sampling_task_name,
)


class SceneEvalError(ValueError):
    """Raised when a scene-evaluation input is missing or inconsistent."""


@dataclass(frozen=True)
class SceneEpisode:
    scene_dir: Path
    manifest_path: Path
    placement_dir: Path
    placement_meta_path: Path
    scene_spec: dict[str, Any]
    placement_meta: dict[str, Any]
    task_name: str
    embodiment: str
    episode_id: int
    episode_data_path: Path
    instruction_path: Path
    instruction: str
    placement_seed: int
    task_binding: dict[str, list[str]]
    task_snapshot: dict[str, Any] | None


@dataclass(frozen=True)
class SceneEvalCase:
    """One persisted evaluation episode and any matching condition videos."""

    episode: SceneEpisode
    condition_videos: tuple[Path, ...]


def sample_scene_cases_per_task(
    cases: Sequence[SceneEvalCase],
    episodes_per_task: int | None,
    seed: int | None,
) -> list[SceneEvalCase]:
    """Select at most ``episodes_per_task`` cases for each task name."""
    if episodes_per_task is None:
        return list(cases)
    if (
        isinstance(episodes_per_task, bool)
        or not isinstance(episodes_per_task, int)
        or episodes_per_task <= 0
    ):
        raise SceneEvalError(
            f"episodes_per_task must be a positive integer or null, got {episodes_per_task!r}"
        )
    if seed is None:
        raise SceneEvalError(
            "visual_episode_seed is required when sampling task episodes"
        )

    grouped_indices: dict[str, list[int]] = {}
    for index, case in enumerate(cases):
        grouped_indices.setdefault(case.episode.task_name, []).append(index)

    selected_indices: set[int] = set()
    for task_name, indices in sorted(grouped_indices.items()):
        if len(indices) <= episodes_per_task:
            selected_indices.update(indices)
            continue
        sample_key = sampling_task_name(task_name)
        digest = hashlib.sha256(f"{int(seed)}:{sample_key}".encode("utf-8")).digest()
        task_seed = int.from_bytes(digest[:8], "big")
        selected_indices.update(
            random.Random(task_seed).sample(indices, episodes_per_task)
        )
    return [case for index, case in enumerate(cases) if index in selected_indices]


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise SceneEvalError(f"{label} not found: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SceneEvalError(f"Invalid JSON in {label}: {path}") from exc
    if not isinstance(value, dict):
        raise SceneEvalError(f"Expected an object in {label}: {path}")
    return value


def _parse_episode_id(value: Any) -> int:
    try:
        episode_id = int(value)
    except (TypeError, ValueError) as exc:
        raise SceneEvalError(f"episode_id must be an integer, got {value!r}") from exc
    if episode_id < 0:
        raise SceneEvalError(f"episode_id must be >= 0, got {episode_id}")
    return episode_id


def _placement_dir(scene_dir: Path, placement_id: Any) -> Path:
    if placement_id is None or str(placement_id).strip() == "":
        raise SceneEvalError("placement_id is required in scene evaluation mode")
    text = str(placement_id)
    if text.startswith("placement"):
        suffix = text[len("placement") :]
    else:
        suffix = text
    try:
        pid = int(suffix)
    except ValueError as exc:
        raise SceneEvalError(
            f"placement_id must be an integer or placementN, got {placement_id!r}"
        ) from exc
    if pid < 0:
        raise SceneEvalError(f"placement_id must be >= 0, got {pid}")
    return scene_dir / f"placement{pid}"


def _resolve_episode_embodiment(task_dir: Path, requested: str) -> str | None:
    """Resolve a logical embodiment to its persisted data directory.

    Scene exports may register ``aloha-agilex`` in metadata while storing
    episodes under perturbation directories such as ``aloha-agilex__pert0``.
    For the logical ALOHA name, use the numerically smallest perturbation.
    """
    requested = str(requested).strip()
    exact = task_dir / requested
    pert_pattern = re.compile(rf"^{re.escape(requested)}__pert(\d+)$")
    pert_candidates: list[tuple[int, str]] = []
    if task_dir.is_dir():
        for path in task_dir.iterdir():
            if not path.is_dir() or not (path / "data").is_dir():
                continue
            match = pert_pattern.fullmatch(path.name)
            if match is not None:
                pert_candidates.append((int(match.group(1)), path.name))
    if pert_candidates:
        return min(pert_candidates, key=lambda item: (item[0], item[1]))[1]
    if exact.is_dir() and (exact / "data").is_dir():
        return requested
    return None


def load_scene_episode(
    scene_dir: str | os.PathLike[str],
    placement_id: Any,
    task_name: str,
    embodiment: str,
    episode_id: Any,
    instruction_type: str = "unseen",
) -> SceneEpisode:
    """Validate and load one production scene/placement/task/episode."""
    root = Path(scene_dir).expanduser().resolve()
    if not root.is_dir():
        raise SceneEvalError(f"scene_dir not found: {root}")
    if not task_name or not str(task_name).strip():
        raise SceneEvalError("task_name is required in scene evaluation mode")
    if not embodiment or not str(embodiment).strip():
        raise SceneEvalError("embodiment is required in scene evaluation mode")
    task_name = public_task_name(str(task_name))

    manifest_path = root / "manifest.json"
    spec = public_scene_metadata(_read_json(manifest_path, "scene manifest"))
    placement_dir = _placement_dir(root, placement_id)
    if not placement_dir.is_dir():
        raise SceneEvalError(f"placement not found: {placement_dir}")
    placement_meta_path = placement_dir / "placement_meta.json"
    stored_meta = _read_json(placement_meta_path, "placement metadata")
    meta = public_scene_metadata(stored_meta)

    supported = meta.get("supported_tasks")
    if not isinstance(supported, list) or task_name not in supported:
        raise SceneEvalError(
            f"Task {task_name!r} is not supported by {placement_meta_path}; "
            f"available: {supported or []}"
        )
    embodiments = meta.get("embodiments")
    if isinstance(embodiments, list) and str(embodiment) not in {
        str(item) for item in embodiments
    }:
        raise SceneEvalError(
            f"Embodiment {embodiment!r} is not recorded by {placement_meta_path}; "
            f"available: {embodiments}"
        )
    manifest_task = next(
        (
            entry
            for entry in spec.get("supported_tasks", [])
            if isinstance(entry, dict) and entry.get("task_name") == task_name
        ),
        None,
    )
    if not isinstance(manifest_task, dict) or not isinstance(
        manifest_task.get("binding"), dict
    ):
        raise SceneEvalError(
            f"No persisted binding for task {task_name!r} in {manifest_path}"
        )
    task_binding = manifest_task["binding"]

    episode = _parse_episode_id(episode_id)
    task_root = placement_dir / task_name
    if not task_root.is_dir():
        stored_task = next(
            name
            for name in stored_meta["supported_tasks"]
            if public_task_name(name) == task_name
        )
        task_root = placement_dir / stored_task
    data_embodiment = _resolve_episode_embodiment(task_root, str(embodiment))
    if data_embodiment is None:
        raise SceneEvalError(
            f"Embodiment data directory not found for {embodiment!r} under {task_root}"
        )
    task_dir = task_root / data_embodiment
    data_path = task_dir / "data" / f"episode{episode}.hdf5"
    instruction_path = task_dir / "instructions" / f"episode{episode}.json"
    if not data_path.is_file():
        raise SceneEvalError(f"Episode data not found: {data_path}")
    instructions = _read_json(instruction_path, "episode instructions")
    values = instructions.get(instruction_type)
    if not isinstance(values, list) or not values:
        raise SceneEvalError(
            f"Instruction type {instruction_type!r} is empty or missing in {instruction_path}"
        )
    instruction = str(values[0]).strip()
    if not instruction:
        raise SceneEvalError(f"Instruction is empty in {instruction_path}")

    try:
        placement_seed = int(meta["placement_seed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SceneEvalError(
            f"Invalid placement_seed in {placement_meta_path}"
        ) from exc
    placement = meta.get("placement", {})
    task_snapshot = None
    if isinstance(placement, dict):
        task_by_emb = placement.get(task_name, {})
        if isinstance(task_by_emb, dict):
            task_snapshot = task_by_emb.get(data_embodiment)

    return SceneEpisode(
        scene_dir=root,
        manifest_path=manifest_path,
        placement_dir=placement_dir,
        placement_meta_path=placement_meta_path,
        scene_spec=spec,
        placement_meta=meta,
        task_name=str(task_name),
        embodiment=str(embodiment),
        episode_id=episode,
        episode_data_path=data_path,
        instruction_path=instruction_path,
        instruction=instruction,
        placement_seed=placement_seed,
        task_binding=task_binding,
        task_snapshot=task_snapshot,
    )


def discover_scene_cases(
    scene_root: str | os.PathLike[str],
    evaluation_embodiment: str,
    condition_embodiment: str = "aloha-agilex",
    instruction_type: str = "unseen",
    require_condition: bool = True,
    sample_visual_episode: bool = False,
    visual_episode_seed: int | None = None,
    episodes_per_task: int | None = None,
    task_names: Iterable[str] | None = None,
) -> list[SceneEvalCase]:
    """Discover valid episodes under a scene directory or scene root.

    require_condition also requires matching ALOHA condition videos. Ignore empty
    scenes; reject partial data with a path-specific error. sample_visual_episode
    selects one episode per task using visual_episode_seed. episodes_per_task
    limits cases globally by task before worker sharding.
    """
    root = Path(scene_root).expanduser().resolve()
    allowed_tasks = (
        None
        if task_names is None
        else {public_task_name(str(name)) for name in task_names}
    )
    if not root.is_dir():
        raise SceneEvalError(f"scene_root not found: {root}")
    if (root / "manifest.json").is_file():
        scene_dirs = [root]
    else:
        scene_dirs = sorted(
            path
            for path in root.iterdir()
            if path.is_dir() and (path / "manifest.json").is_file()
        )
    if not scene_dirs:
        raise SceneEvalError(f"No scene manifest found under: {root}")
    if require_condition and not str(condition_embodiment).strip():
        raise SceneEvalError("condition_embodiment must not be empty")

    episode_pattern = re.compile(r"^episode(\d+)\.hdf5$")
    cases: list[SceneEvalCase] = []
    for scene_dir in scene_dirs:
        for placement_dir in sorted(scene_dir.glob("placement*")):
            if not placement_dir.is_dir():
                continue
            placement_id = placement_dir.name.removeprefix("placement")
            for task_dir in sorted(
                path for path in placement_dir.iterdir() if path.is_dir()
            ):
                if (
                    allowed_tasks is not None
                    and public_task_name(task_dir.name) not in allowed_tasks
                ):
                    continue
                data_embodiment = _resolve_episode_embodiment(
                    task_dir, evaluation_embodiment
                )
                if data_embodiment is None:
                    continue
                data_dir = task_dir / data_embodiment / "data"
                if not data_dir.is_dir():
                    continue
                episode_ids = sorted(
                    int(match.group(1))
                    for path in data_dir.iterdir()
                    if (match := episode_pattern.match(path.name)) is not None
                )
                if not episode_ids:
                    continue
                if sample_visual_episode and len(episode_ids) > 1:
                    if visual_episode_seed is None:
                        raise SceneEvalError(
                            "visual_episode_seed is required when sampling a visual episode"
                        )
                    identity = "|".join(
                        (
                            sampling_scene_name(scene_dir.name),
                            placement_dir.name,
                            sampling_task_name(task_dir.name),
                        )
                    )
                    digest = hashlib.sha256(
                        f"{int(visual_episode_seed)}:{identity}".encode("utf-8")
                    ).digest()
                    choice_seed = int.from_bytes(digest[:8], "big")
                    episode_ids = [random.Random(choice_seed).choice(episode_ids)]
                condition_dirs: list[Path] = []
                if require_condition:
                    condition_dirs = sorted(
                        path
                        for path in task_dir.iterdir()
                        if path.is_dir()
                        and (path / "video").is_dir()
                        and (
                            path.name == condition_embodiment
                            or path.name.startswith(f"{condition_embodiment}__")
                        )
                    )
                    if not condition_dirs:
                        raise SceneEvalError(
                            f"No condition embodiment directories matching "
                            f"{condition_embodiment!r} under {task_dir}"
                        )
                for episode_id in episode_ids:
                    episode = load_scene_episode(
                        scene_dir=scene_dir,
                        placement_id=placement_id,
                        task_name=task_dir.name,
                        embodiment=evaluation_embodiment,
                        episode_id=episode_id,
                        instruction_type=instruction_type,
                    )
                    videos: list[Path] = []
                    for condition_dir in condition_dirs:
                        video_path = (
                            condition_dir / "video" / f"episode{episode_id}.mp4"
                        )
                        if not video_path.is_file():
                            raise SceneEvalError(
                                f"Condition video missing for episode {episode_id}: "
                                f"{video_path}"
                            )
                        videos.append(video_path)
                    cases.append(
                        SceneEvalCase(episode=episode, condition_videos=tuple(videos))
                    )
    return sample_scene_cases_per_task(
        cases,
        episodes_per_task=episodes_per_task,
        seed=visual_episode_seed,
    )


def infer_robotwin_root(scene_dir: Path, configured_root: Path) -> Path:
    """Use the production checkout that owns a scene manifest when available."""
    return configured_root.resolve()


def validate_robotwin_root(root: Path) -> Path:
    """Validate and normalize a production RoboTwin checkout path."""
    resolved = root.expanduser().resolve()
    required = (resolved / "envs" / "scene", resolved / "task_config")
    missing = [str(path) for path in required if not path.is_dir()]
    if missing:
        raise SceneEvalError(
            f"Invalid RoboTwin root: {resolved}. Missing required directories: "
            + ", ".join(missing)
        )
    return resolved


def _module_is_from_root(module: Any, root: Path) -> bool:
    locations: list[str] = []
    module_file = getattr(module, "__file__", None)
    if module_file:
        locations.append(str(module_file))
    module_path = getattr(module, "__path__", None)
    if module_path:
        locations.extend(str(path) for path in module_path)
    for location in locations:
        try:
            Path(location).resolve().relative_to(root)
            return True
        except ValueError:
            continue
    return False


@contextmanager
def _robotwin_context(root: Path) -> Iterator[None]:
    old_cwd = Path.cwd()
    resolved_root = root.resolve()
    root_text = str(resolved_root)
    old_sys_path = list(sys.path)
    sys.path[:] = [root_text] + [
        entry
        for entry in old_sys_path
        if Path(entry or old_cwd).resolve() != resolved_root
    ]

    env_module_names = [
        name for name in sys.modules if name == "envs" or name.startswith("envs.")
    ]
    root_module = sys.modules.get("envs")
    has_foreign_envs = bool(env_module_names) and (
        root_module is None or not _module_is_from_root(root_module, resolved_root)
    )
    saved_env_modules = {}
    if has_foreign_envs:
        saved_env_modules = {name: sys.modules.pop(name) for name in env_module_names}

    os.chdir(resolved_root)
    try:
        yield
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_sys_path
        if saved_env_modules:
            for name in list(sys.modules):
                if name == "envs" or name.startswith("envs."):
                    del sys.modules[name]
            sys.modules.update(saved_env_modules)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SceneEvalError(f"RoboTwin task config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise SceneEvalError(f"Expected a mapping in task config: {path}")
    return value


def build_runtime_args(
    robotwin_root: Path,
    task_config: str,
    task_name: str,
    embodiment: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Build the RoboTwin env kwargs used by the normal evaluator."""
    args = _load_yaml(robotwin_root / "task_config" / f"{task_config}.yml")
    args["task_name"] = task_name
    args["task_config"] = task_config
    args["policy_name"] = "simpleicl_policy"
    args["eval_mode"] = True
    args["render_freq"] = 0
    args["save_data"] = False
    # Match recorded scene replay: settle fully without rejecting the saved seed.
    args["ignore_stability_failure"] = True
    args["save_path"] = str(output_dir)
    args["eval_video_save_dir"] = (
        str(output_dir) if bool(args.get("eval_video_log", False)) else None
    )
    args["domain_randomization"] = {
        "random_background": False,
        "cluttered_table": False,
        "clean_background_rate": 1,
        "random_head_camera_dis": 0,
        "random_table_height": 0,
        "random_light": False,
        "crazy_random_light_rate": 0,
    }
    camera_type = args.get("camera", {}).get("head_camera_type")
    camera_cfg = _load_yaml(robotwin_root / "task_config" / "_camera_config.yml")
    if camera_type not in camera_cfg:
        raise SceneEvalError(f"Unknown head camera type {camera_type!r}")
    args["head_camera_h"] = camera_cfg[camera_type]["h"]
    args["head_camera_w"] = camera_cfg[camera_type]["w"]

    embodiment_cfg = _load_yaml(
        robotwin_root / "task_config" / "_embodiment_config.yml"
    )
    if embodiment != "aloha-agilex":
        raise SceneEvalError(
            f"SimpleICL supports only aloha-agilex, got {embodiment!r}"
        )

    def robot_path(name: str) -> Path:
        spec = embodiment_cfg.get(name)
        path = spec.get("file_path") if isinstance(spec, dict) else None
        if not path:
            raise SceneEvalError(f"Embodiment {name!r} has no file_path")
        from simpleicl.paths import resource_path

        return resource_path(str(path))

    left_path = right_path = robot_path("aloha-agilex")
    args["embodiment"] = ["aloha-agilex"]
    args["left_robot_file"] = str(left_path)
    args["right_robot_file"] = str(right_path)
    args["dual_arm_embodied"] = True
    args["left_embodiment_config"] = _load_yaml(left_path / "config.yml")
    args["right_embodiment_config"] = _load_yaml(right_path / "config.yml")
    return args


def _start_eval_video(
    env: Any,
    args: dict[str, Any],
    output_dir: Path,
) -> tuple[subprocess.Popen | None, Path | None]:
    if env.eval_video_path is None:
        return None, None

    video_size = f"{int(args['head_camera_w'])}x{int(args['head_camera_h'])}"
    video_path = output_dir / f"episode{int(env.test_num)}.mp4"
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            video_size,
            "-framerate",
            "10",
            "-i",
            "-",
            "-pix_fmt",
            "yuv420p",
            "-vcodec",
            "libx264",
            "-crf",
            "23",
            str(video_path),
        ],
        stdin=subprocess.PIPE,
    )
    env._set_eval_video_ffmpeg(process)
    return process, video_path


def prepare_scene_runtime(
    episode: SceneEpisode,
    robotwin_root: Path,
    task_config: str,
    output_dir: Path,
    *,
    legacy_articulated_placement: bool = True,
) -> tuple[Any, dict[str, Any], Any]:
    """Import RoboTwin lazily and instantiate a scene-bound task environment."""
    with _robotwin_context(robotwin_root):
        scene_mod = importlib.import_module("envs.scene")
        task_mod = importlib.import_module(f"envs.{episode.task_name}")
        env_class = getattr(task_mod, episode.task_name)
        requirement = scene_mod.get_requirement(episode.task_name)
        if requirement is None:
            raise SceneEvalError(
                f"Task requirement is unavailable: {episode.task_name}"
            )
        scene_spec_obj = scene_mod.SceneSpec.from_dict(episode.scene_spec)
        scene_set = scene_spec_obj.build_object_set()
        match = scene_mod.match_from_binding(
            requirement, scene_set, episode.task_binding
        )
        if match is None:
            raise SceneEvalError(
                f"Persisted binding is invalid for task {episode.task_name!r}"
            )
        args = build_runtime_args(
            robotwin_root,
            task_config,
            episode.task_name,
            episode.embodiment,
            output_dir,
        )
        env = env_class()
        env.scene_legacy_articulated_placement = legacy_articulated_placement
        env.scene_spec = scene_set
        env.scene_binding = match
        env.scene_environment = scene_spec_obj.environment
        if isinstance(episode.task_snapshot, dict):
            env.scene_task_rng_state = episode.task_snapshot.get("task_rng_state")
        env.scene_visual_rng = np.random.RandomState(
            episode.placement_seed * 1000 + episode.episode_id
        )
        return env, args, match


def run_scene_episode(
    episode: SceneEpisode,
    robotwin_root: Path,
    task_config: str,
    output_dir: Path,
    model: Any,
    policy_module: Any,
    *,
    legacy_articulated_placement: bool = True,
) -> dict[str, Any]:
    """Run exactly one policy rollout in the persisted scene placement."""
    with _robotwin_context(robotwin_root):
        env, args, _ = prepare_scene_runtime(
            episode,
            robotwin_root,
            task_config,
            output_dir,
            legacy_articulated_placement=legacy_articulated_placement,
        )
        args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
        args["right_arm_dim"] = len(
            args["right_embodiment_config"]["arm_joints_name"][1]
        )
        video_process: subprocess.Popen | None = None
        video_path: Path | None = None
        video_closed = False
        try:
            env.suc = 0
            env.test_num = 0
            env.setup_demo(
                now_ep_num=episode.episode_id,
                seed=episode.placement_seed,
                is_test=True,
                **args,
            )
            env.set_instruction(episode.instruction)
            video_process, video_path = _start_eval_video(env, args, output_dir)
            policy_module.reset_model(model)
            while env.take_action_cnt < env.step_lim and not env.eval_success:
                observation = env.get_obs()
                policy_module.eval(env, model, observation)
            success = bool(env.eval_success)
            if video_process is not None:
                env._del_eval_video_ffmpeg()
                video_closed = True
                if video_path is None or not video_path.is_file():
                    raise FileNotFoundError(
                        f"Expected eval video file not found: {video_path}"
                    )

            env.suc = int(success)
            env.test_num = 1
            if success:
                print("\033[92mSuccess!\033[0m")
            else:
                print("\033[91mFail!\033[0m")
            print(
                f"{episode.task_name} | simpleicl_policy | {task_config}\n"
                f"Success rate: {env.suc}/{env.test_num} => {env.suc / env.test_num * 100:.1f}%, "
                f"current seed: {episode.placement_seed}"
            )
            return {
                "success": success,
                "success_rate": float(env.suc / env.test_num),
                "steps": int(env.take_action_cnt),
                "step_limit": int(env.step_lim),
                "task_name": episode.task_name,
                "embodiment": episode.embodiment,
                "episode_id": episode.episode_id,
                "placement_seed": episode.placement_seed,
                "legacy_articulated_placement": legacy_articulated_placement,
                "instruction": episode.instruction,
                "video_path": str(video_path) if video_path is not None else None,
            }
        finally:
            try:
                if video_process is not None and not video_closed:
                    try:
                        env._del_eval_video_ffmpeg()
                    except Exception:
                        pass
                try:
                    env.close_env(clear_cache=True)
                except Exception:
                    pass
            finally:
                finalize_episode = getattr(policy_module, "finalize_episode", None)
                if callable(finalize_episode):
                    finalize_episode(model)
