"""Evaluate persisted RoboTwin scenes with reproducible sampling and GPU sharding."""

import os
import json
import re
import secrets
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from simpleicl.paths import ROOT as PROJECT_ROOT, CONFIGS, OUTPUTS
from simpleicl.utils.config_resolvers import register_default_resolvers
from simpleicl.evaluation.settings import resolve_video_sampling

register_default_resolvers()
SRC_ROOT = PROJECT_ROOT / "src"
POLICY_NAME = "policy"
HARD_CODED_EXCLUDED_TASKS = set(
    "beat_block_hammer,blocks_ranking_rgb,click_bell,dump_bin_bigbin,lift_pot,move_can_pot,move_playingcard_away,open_laptop,place_a2b_left,place_a2b_right,place_bread_basket,place_cans_plasticbox,place_dual_shoes,place_object_scale,place_phone_stand,rotate_qrcode,stack_blocks_three,stack_bowls_three,stack_bowls_two,turn_switch,blocks_ranking_size,click_alarmclock,grab_roller,handover_mic,move_stapler_pad,open_microwave,place_bread_skillet,place_burger_fries,place_can_basket,place_container_plate,place_empty_cup,place_fan,place_mouse_pad,place_object_basket,place_object_stand,place_shoe,press_stapler,stack_blocks_two,stamp_seal".split(
        ","
    )
)


def _filter_hardcoded_tasks(cfg, cases):
    if not bool(cfg.EVALUATION.get("exclude_hardcoded_tasks", False)):
        return cases
    return [c for c in cases if c.episode.task_name not in HARD_CODED_EXCLUDED_TASKS]


def _configured_task_names(cfg) -> tuple[str, ...] | None:
    """Return the optional fixed-scene task allowlist from Hydra config."""
    value = cfg.EVALUATION.get("task_names")
    if value is None:
        return None
    if isinstance(value, str):
        raise ValueError("EVALUATION.task_names must be a non-empty list of task names")
    try:
        names = [str(item).strip() for item in value]
    except TypeError as exc:
        raise ValueError(
            "EVALUATION.task_names must be a non-empty list of task names"
        ) from exc
    names = list(dict.fromkeys(names))
    if not names or any(
        re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name) is None for name in names
    ):
        raise ValueError(
            "EVALUATION.task_names must be a non-empty list of valid task names"
        )
    return tuple(names)


def _filter_scene_cases(cfg, cases):
    """Apply the configured hardcoded-task and explicit task filters."""
    cases = _filter_hardcoded_tasks(cfg, cases)
    task_names = _configured_task_names(cfg)
    if task_names is None:
        return cases
    allowed = set(task_names)
    return [case for case in cases if case.episode.task_name in allowed]


if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _resolve_path(path_str: str, *, base: Path) -> Path:
    path = Path(os.path.expanduser(os.path.expandvars(str(path_str))))
    if not path.is_absolute():
        path = (base / path).resolve()
    return path.resolve()


def _resolve_optional_path(path_value: Any, *, base: Path) -> Path | None:
    if path_value is None:
        return None
    text = str(path_value).strip()
    if text == "" or text.lower() in {"none", "null"}:
        return None
    return _resolve_path(text, base=base)


def _resolve_result_subdir(cfg: DictConfig) -> str | None:
    value = cfg.EVALUATION.get("result_subdir")
    if value is None:
        return None
    name = str(value).strip()
    if not name or name in {".", ".."} or Path(name).name != name:
        raise ValueError(
            f"EVALUATION.result_subdir must be a single directory name, got: {value!r}"
        )
    return name


def _link_condition_video(condition_video: Path, output_dir: Path) -> Path:
    """Expose the source condition clip beside each rollout result."""
    link_path = output_dir / "condition_video.mp4"
    target = condition_video.resolve()
    if link_path.is_symlink():
        if link_path.resolve() == target:
            return link_path
        link_path.unlink()
    elif link_path.exists():
        raise FileExistsError(
            f"Cannot create condition video symlink; path is not a symlink: {link_path}"
        )
    link_path.symlink_to(target)
    return link_path


def _resolve_dataset_stats_path(
    cfg: DictConfig,
    ckpt_path: Path,
    output_dir: Path | None = None,
) -> Path:
    explicit = _resolve_optional_path(
        cfg.EVALUATION.dataset_stats_path, base=PROJECT_ROOT
    )
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)

    for parent in list(ckpt_path.parents)[:4]:
        candidates.append((parent / "dataset_stats.json").resolve())
        candidates.append((parent / "dataset_stats").resolve())

    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file():
            return resolved
        if resolved.is_dir():
            stats_paths = sorted(
                path
                for path in resolved.glob("*.json")
                if path.name.endswith("dataset_stats.json")
            )
            if not stats_paths:
                raise FileNotFoundError(
                    f"No *dataset_stats.json files found in stats directory: {resolved}"
                )
            if len(stats_paths) == 1:
                return stats_paths[0]
            if output_dir is None:
                raise ValueError(
                    "Merging multiple dataset stats files requires an output directory."
                )
            from simpleicl.datasets.lerobot.utils.normalizer import (
                load_dataset_stats_from_json,
                merge_dataset_stats,
                save_dataset_stats_to_json,
            )

            output_dir.mkdir(parents=True, exist_ok=True)
            merged_path = output_dir / "merged_dataset_stats.json"
            merged = merge_dataset_stats(
                [load_dataset_stats_from_json(str(path)) for path in stats_paths]
            )
            save_dataset_stats_to_json(merged, str(merged_path))
            return merged_path

    raise FileNotFoundError(
        "Failed to locate dataset_stats.json. Tried explicit "
        "EVALUATION.dataset_stats_path and checkpoint parent directories. "
        "Please pass EVALUATION.dataset_stats_path=/path/to/dataset_stats.json."
    )


def _discover_condition_videos(
    task_dir: Path,
    episode_id: int,
    condition_embodiment: str,
) -> list[Path]:
    """Find all condition videos for one fixed scene episode.

    A scene may contain multiple directories such as ``aloha-agilex__pert0``.
    The ALOHA subject episode remains the evaluation environment; these videos are
    only used as ``video_all`` conditioning inputs.
    """
    if not task_dir.is_dir():
        raise FileNotFoundError(f"Task directory not found: {task_dir}")

    prefix = str(condition_embodiment).strip()
    if not prefix:
        raise ValueError("EVALUATION.condition_embodiment must not be empty")
    condition_dirs = sorted(
        (
            path
            for path in task_dir.iterdir()
            if path.is_dir()
            and (path.name == prefix or path.name.startswith(f"{prefix}__"))
        ),
        key=lambda path: path.name,
    )
    if not condition_dirs:
        raise FileNotFoundError(
            f"No condition embodiment directories matching {prefix!r} under {task_dir}"
        )

    episode_name = f"episode{int(episode_id)}.mp4"
    videos: list[Path] = []
    missing: list[Path] = []
    for condition_dir in condition_dirs:
        video_path = condition_dir / "video" / episode_name
        if video_path.is_file():
            videos.append(video_path)
        else:
            missing.append(video_path)
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"Condition video missing for episode {episode_id}: {missing_text}"
        )
    return videos


def _prepare_condition_latents(
    policy: Any,
    video_path: Path,
    video_all_size: list[int],
    video_all_num_frames: int | None,
    sampled_video_path: Path,
    video_all_frame_percentage: float | None = None,
    video_all_max_num_frames: int | None = None,
) -> Any:
    """Encode one ALOHA video and save its pre-encoding model input."""
    from simpleicl.evaluation.policy import deploy_policy

    model = getattr(policy, "model", None)
    if model is None:
        raise TypeError("Fixed-scene evaluation requires an in-process policy model")
    video_expert = getattr(model, "video_expert", None)
    is_action_only = model.__class__.__name__ == "SimpleICLActionOnly"
    if not is_action_only and not getattr(
        video_expert, "has_video_all_condition", False
    ):
        raise RuntimeError(
            "The loaded model does not enable video_all conditioning. "
            "Use a checkpoint/config with model.video_dit_config.has_video_all_condition=true."
        )
    return deploy_policy._prepare_video_all_latents(
        model,
        str(video_path),
        video_all_size,
        str(model.device),
        model.torch_dtype,
        num_frames=video_all_num_frames,
        frame_percentage=video_all_frame_percentage,
        max_num_frames=video_all_max_num_frames,
        sampled_video_path=str(sampled_video_path),
    )


def _policy_has_video_all_condition(policy: Any) -> bool:
    """Return whether the loaded policy model accepts a ``video_all`` input."""
    model = getattr(policy, "model", None)
    video_expert = getattr(model, "video_expert", None)
    return bool(
        getattr(video_expert, "has_video_all_condition", False)
        or model.__class__.__name__ == "SimpleICLActionOnly"
    )


def _attention_visualization_config(cfg: DictConfig) -> dict[str, Any]:
    value = cfg.EVALUATION.get("attention_visualization", {})
    if isinstance(value, DictConfig):
        value = OmegaConf.to_container(value, resolve=True)
    if not isinstance(value, dict):
        raise ValueError("EVALUATION.attention_visualization must be a mapping")
    return dict(value)


def _attention_visualization_enabled(cfg: DictConfig) -> bool:
    return bool(_attention_visualization_config(cfg).get("enabled", False))


def _collect_sim_extra_overrides() -> list[str]:
    """Forward model/data overrides into the nested simulation config."""
    hydra_config = HydraConfig.get()
    override_section = getattr(hydra_config, "overrides", None)
    task_overrides = getattr(override_section, "task", None)
    if task_overrides:
        raw_overrides = [str(value) for value in task_overrides]
    else:
        # Compatibility fallback for older Hydra versions without the parsed
        # override list.  The dirname form is lossy for values containing `,`.
        hydra_overrides = hydra_config.job.override_dirname
        raw_overrides = hydra_overrides.split(",") if hydra_overrides else []
    return [
        part.strip()
        for part in raw_overrides
        if part.strip()
        and (
            part.split("=", 1)[0].strip() in {"model", "data"}
            or part.split("=", 1)[0].strip().startswith(("model.", "data."))
        )
    ]


def _run_fixed_scene(
    cfg: DictConfig,
    *,
    ckpt_path: Path,
    robotwin_root: Path,
    run_output_dir: Path,
    scene_dir: Path,
) -> None:
    """Run one persisted scene episode with or without video conditioning."""
    from simpleicl.evaluation import scene_eval
    from simpleicl.evaluation.policy import deploy_policy

    task_name = str(cfg.EVALUATION.task_name)
    embodiment = str(cfg.EVALUATION.embodiment)
    episode = scene_eval.load_scene_episode(
        scene_dir=scene_dir,
        placement_id=cfg.EVALUATION.placement_id,
        task_name=task_name,
        embodiment=embodiment,
        episode_id=cfg.EVALUATION.episode_id,
        instruction_type=str(cfg.EVALUATION.instruction_type),
    )
    # Use the same policy factory as the normal RoboTwin deployment wrapper.
    policy_args: dict[str, Any] = {
        "ckpt_setting": str(ckpt_path),
        "sim_cfg_path": str((CONFIGS / "sim_robotwin.yaml").resolve()),
        "sim_task": str(
            getattr(cfg.EVALUATION, "model_task", None)
            or HydraConfig.get().runtime.choices.get("task")
        ),
        "mixed_precision": cfg.mixed_precision,
        "device": cfg.EVALUATION.device,
        "dataset_stats_path": str(
            _resolve_dataset_stats_path(cfg, ckpt_path, run_output_dir)
        ),
        "pretrained_model_path": cfg.EVALUATION.pretrained_model_path,
        "action_horizon": cfg.EVALUATION.action_horizon,
        "replan_steps": cfg.EVALUATION.replan_steps,
        "num_inference_steps": cfg.EVALUATION.num_inference_steps,
        "sigma_shift": cfg.EVALUATION.sigma_shift,
        "seed": cfg.seed,
        "text_cfg_scale": cfg.EVALUATION.text_cfg_scale,
        "negative_prompt": cfg.EVALUATION.negative_prompt,
        "override_instruction": cfg.EVALUATION.override_instruction,
        "rand_device": cfg.EVALUATION.rand_device,
        "tiled": cfg.EVALUATION.tiled,
        "timing_enabled": cfg.EVALUATION.timing_enabled,
        "attention_visualization": _attention_visualization_config(cfg),
    }
    sim_extra_overrides = _collect_sim_extra_overrides()
    if sim_extra_overrides:
        policy_args["sim_extra_overrides"] = json.dumps(sim_extra_overrides)
    model = deploy_policy.get_model(policy_args)
    has_condition = _policy_has_video_all_condition(model)
    video_all_size: list[int] | None = None
    video_all_num_frames: int | None = None
    video_all_frame_percentage: float | None = None
    video_all_max_num_frames: int | None = None
    if has_condition:
        condition_embodiment = str(
            getattr(cfg.EVALUATION, "condition_embodiment", "aloha-agilex")
        )
        condition_videos: list[Path | None] = [
            path
            for path in _discover_condition_videos(
                # Reuse the loader's resolution of public and legacy task names.
                task_dir=episode.episode_data_path.parents[2],
                episode_id=episode.episode_id,
                condition_embodiment=condition_embodiment,
            )
        ]
        video_all_size = [int(value) for value in cfg.data.train.video_all_size]
        (
            video_all_num_frames,
            video_all_frame_percentage,
            video_all_max_num_frames,
        ) = resolve_video_sampling(cfg)
    else:
        condition_videos = [None]

    for condition_video in condition_videos:
        condition_name = (
            condition_video.parent.parent.name
            if condition_video is not None
            else "condition_free"
        )
        fixed_output = run_output_dir / (
            f"scene_{episode.scene_dir.name}_{episode.placement_dir.name}"
            f"_{task_name}_{embodiment}_episode{episode.episode_id}_{condition_name}"
        )
        fixed_output.mkdir(parents=True, exist_ok=True)
        condition_video_link = (
            _link_condition_video(condition_video, fixed_output)
            if condition_video is not None
            else None
        )
        sampled_condition_video = (
            fixed_output / "condition_video_sampled.mp4"
            if condition_video is not None
            else None
        )
        if condition_video is not None:
            assert video_all_size is not None
            assert sampled_condition_video is not None
            model.video_all_latents = _prepare_condition_latents(
                model,
                condition_video,
                video_all_size,
                video_all_num_frames,
                sampled_condition_video,
                video_all_frame_percentage=video_all_frame_percentage,
                video_all_max_num_frames=video_all_max_num_frames,
            )
        else:
            model.video_all_latents = None
        if _attention_visualization_enabled(cfg):
            if sampled_condition_video is None:
                raise RuntimeError("Attention visualization requires a condition video")
            model.start_attention_episode(fixed_output, sampled_condition_video)
        print(f"[eval_robotwin] condition={condition_name} video={condition_video}")
        result = scene_eval.run_scene_episode(
            episode=episode,
            robotwin_root=robotwin_root,
            task_config=str(cfg.EVALUATION.task_config),
            output_dir=fixed_output,
            model=model,
            policy_module=deploy_policy,
            legacy_articulated_placement=bool(
                cfg.EVALUATION.get("legacy_articulated_placement", True)
            ),
        )
        result.update(
            {
                "scene_dir": str(episode.scene_dir),
                "manifest_path": str(episode.manifest_path),
                "placement_meta_path": str(episode.placement_meta_path),
                "episode_data_path": str(episode.episode_data_path),
                "instruction_path": str(episode.instruction_path),
                "robotwin_root": str(robotwin_root),
                "checkpoint": str(ckpt_path),
                "override_instruction": model.override_instruction,
                "effective_instruction": model.override_instruction
                or episode.instruction,
                "evaluation_embodiment": embodiment,
                "condition_embodiment": condition_name,
                "condition_video_path": str(condition_video)
                if condition_video is not None
                else None,
                "condition_video_link": str(condition_video_link)
                if condition_video_link is not None
                else None,
                "condition_video_sampled_path": (
                    str(sampled_condition_video)
                    if sampled_condition_video is not None
                    else None
                ),
                "condition_mode": "video_all"
                if condition_video is not None
                else "none",
                "output_dir": str(fixed_output),
            }
        )
        attention_result = model.get_attention_result()
        if attention_result is not None:
            result["attention_visualization"] = attention_result
        result_txt_path = fixed_output / "_result.txt"
        result_txt_path.write_text(
            f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            f"Instruction Type: {str(cfg.EVALUATION.instruction_type)}\n"
            f"Condition: {condition_name}\n\n"
            f"{result['success_rate']}",
            encoding="utf-8",
        )
        result["result_txt_path"] = str(result_txt_path)
        result_path = fixed_output / "result.json"
        result_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"Data has been saved to {result_txt_path}")
        print(f"Fixed scene evaluation result saved to: {result_path}")


def _fixed_policy_args(
    cfg: DictConfig,
    ckpt_path: Path,
    run_output_dir: Path,
) -> dict[str, Any]:
    """Build the in-process policy arguments shared by batch evaluation."""
    model_task = str(getattr(cfg.EVALUATION, "model_task", "") or "")
    if not model_task:
        model_task = str(HydraConfig.get().runtime.choices.get("task"))
    args: dict[str, Any] = {
        "ckpt_setting": str(ckpt_path),
        "sim_cfg_path": str((CONFIGS / "sim_robotwin.yaml").resolve()),
        "sim_task": model_task,
        "mixed_precision": cfg.mixed_precision,
        "device": cfg.EVALUATION.device,
        "dataset_stats_path": str(
            _resolve_dataset_stats_path(cfg, ckpt_path, run_output_dir)
        ),
        "pretrained_model_path": cfg.EVALUATION.pretrained_model_path,
        "action_horizon": cfg.EVALUATION.action_horizon,
        "replan_steps": cfg.EVALUATION.replan_steps,
        "num_inference_steps": cfg.EVALUATION.num_inference_steps,
        "sigma_shift": cfg.EVALUATION.sigma_shift,
        "seed": cfg.seed,
        "text_cfg_scale": cfg.EVALUATION.text_cfg_scale,
        "negative_prompt": cfg.EVALUATION.negative_prompt,
        "override_instruction": cfg.EVALUATION.override_instruction,
        "rand_device": cfg.EVALUATION.rand_device,
        "tiled": cfg.EVALUATION.tiled,
        "timing_enabled": cfg.EVALUATION.timing_enabled,
        "attention_visualization": _attention_visualization_config(cfg),
    }
    sim_extra_overrides = _collect_sim_extra_overrides()
    if sim_extra_overrides:
        args["sim_extra_overrides"] = json.dumps(sim_extra_overrides)
    return args


def _episodes_per_task(cfg: DictConfig) -> int | None:
    value = cfg.EVALUATION.get("episodes_per_task")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(
            f"EVALUATION.episodes_per_task must be a positive integer or null, got {value!r}"
        )
    return value


def _fixed_scene_output_dir(
    run_output_dir: Path,
    episode: Any,
    condition_name: str,
) -> Path:
    """Return the deterministic output directory for one scene rollout."""
    return run_output_dir / (
        f"scene_{episode.scene_dir.name}_{episode.placement_dir.name}"
        f"_{episode.task_name}_{episode.embodiment}_episode{episode.episode_id}_{condition_name}"
    )


def _valid_result_payload(
    result: Any,
    *,
    episode: Any,
    condition_name: str,
    condition_mode: str,
    output_dir: Path,
    checkpoint: Path | None = None,
    evaluation_embodiment: str | None = None,
    require_attention: bool = False,
    legacy_articulated_placement: bool | None = None,
) -> bool:
    """Check that a persisted result belongs to the expected rollout."""
    if not isinstance(result, dict):
        return False
    required = {"success", "success_rate", "task_name", "episode_id", "output_dir"}
    if not required.issubset(result):
        return False
    if (
        legacy_articulated_placement is not None
        and result.get("legacy_articulated_placement", legacy_articulated_placement)
        != legacy_articulated_placement
    ):
        return False
    if str(result["task_name"]) != str(episode.task_name):
        return False
    try:
        if int(result["episode_id"]) != int(episode.episode_id):
            return False
        float(result["success_rate"])
    except (TypeError, ValueError):
        return False
    if str(result.get("condition_embodiment")) != condition_name:
        return False
    if str(result.get("condition_mode")) != condition_mode:
        return False
    if checkpoint is not None and str(result.get("checkpoint")) != str(checkpoint):
        return False
    if (
        evaluation_embodiment is not None
        and str(result.get("evaluation_embodiment")) != evaluation_embodiment
    ):
        return False
    try:
        result_output_dir = Path(str(result["output_dir"])).resolve()
    except (TypeError, ValueError):
        return False
    if result_output_dir != output_dir.resolve():
        return False
    if require_attention and not _valid_attention_visualization(result, output_dir):
        return False
    return True


def _valid_attention_visualization(result: dict[str, Any], output_dir: Path) -> bool:
    payload = result.get("attention_visualization")
    if not isinstance(payload, dict) or payload.get("enabled") is not True:
        return False
    expected_dir = (output_dir / "attention").resolve()
    expected_paths = {
        "manifest_path": expected_dir / "manifest.json",
        "action_video_path": expected_dir / "action_attention.mp4",
        "video_video_path": expected_dir / "video_attention.mp4",
    }
    for key, expected in expected_paths.items():
        value = payload.get(key)
        if (
            not value
            or Path(str(value)).resolve() != expected
            or not expected.is_file()
            or expected.stat().st_size <= 0
        ):
            return False
    try:
        manifest = json.loads(
            expected_paths["manifest_path"].read_text(encoding="utf-8")
        )
        replan_count = int(manifest["replan_count"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    if replan_count <= 0 or int(payload.get("replan_count", -1)) != replan_count:
        return False
    replans = manifest.get("replans")
    if not isinstance(replans, list) or len(replans) != replan_count:
        return False
    for replan in replans:
        if not isinstance(replan, dict):
            return False
        relative = Path(str(replan.get("directory", "")))
        if relative.is_absolute() or ".." in relative.parts:
            return False
        replan_dir = (output_dir / relative).resolve()
        if expected_dir not in replan_dir.parents:
            return False
        for source in ("action", "video"):
            spatial = replan_dir / f"{source}_spatial.png"
            temporal = replan_dir / f"{source}_temporal.png"
            if not spatial.is_file() or spatial.stat().st_size <= 0:
                return False
            if not temporal.is_file() or temporal.stat().st_size <= 0:
                return False
        if bool(manifest.get("save_stats", True)):
            stats_path = replan_dir / "attention_stats.npz"
            if not stats_path.is_file() or stats_path.stat().st_size <= 0:
                return False
    return True


def _load_completed_result(
    output_dir: Path,
    *,
    episode: Any,
    condition_name: str,
    condition_mode: str,
    checkpoint: Path | None = None,
    evaluation_embodiment: str | None = None,
    require_attention: bool = False,
    legacy_articulated_placement: bool | None = None,
) -> dict[str, Any] | None:
    """Load a result suitable for resume, returning None for partial output."""
    result_path = output_dir / "result.json"
    if not result_path.is_file():
        return None
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not _valid_result_payload(
        result,
        episode=episode,
        condition_name=condition_name,
        condition_mode=condition_mode,
        output_dir=output_dir,
        checkpoint=checkpoint,
        evaluation_embodiment=evaluation_embodiment,
        require_attention=require_attention,
        legacy_articulated_placement=legacy_articulated_placement,
    ):
        return None
    return result


def _expected_resume_results(
    cases: list[Any],
    *,
    condition_mode: str,
    run_output_dir: Path,
) -> dict[Path, tuple[Any, str]]:
    """Build expected result paths and metadata for one worker shard."""
    expected: dict[Path, tuple[Any, str]] = {}
    for case in cases:
        episode = case.episode
        if condition_mode == "none":
            condition_name = "condition_free"
            expected[
                _fixed_scene_output_dir(run_output_dir, episode, condition_name)
            ] = (
                episode,
                condition_name,
            )
            continue
        for condition_video in case.condition_videos:
            condition_name = condition_video.parent.parent.name
            expected[
                _fixed_scene_output_dir(run_output_dir, episode, condition_name)
            ] = (
                episode,
                condition_name,
            )
    return expected


def _load_complete_worker_summary(
    summary_path: Path,
    *,
    worker_output_dir: Path,
    cases: list[Any],
    condition_cases: list[Any],
    scene_root: Path,
    ckpt_path: Path,
    evaluation_embodiment: str,
    condition_embodiment: str,
    visual_episode_seed: int,
    episodes_per_task: int | None,
    require_attention: bool = False,
    legacy_articulated_placement: bool | None = None,
) -> dict[str, Any] | None:
    """Return a reusable worker summary when it exactly matches this run."""
    if not summary_path.is_file():
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(summary, dict):
        return None
    metadata = {
        "scene_root": str(scene_root),
        "checkpoint": str(ckpt_path),
        "evaluation_embodiment": evaluation_embodiment,
        "condition_embodiment": condition_embodiment,
        "visual_episode_seed": visual_episode_seed,
        "episodes_per_task": episodes_per_task,
    }
    if any(summary.get(key) != value for key, value in metadata.items()):
        return None
    results = summary.get("results")
    if not isinstance(results, list) or not results:
        return None
    modes = {
        str(result.get("condition_mode"))
        for result in results
        if isinstance(result, dict)
    }
    if modes == {"none"}:
        expected = _expected_resume_results(
            cases, condition_mode="none", run_output_dir=worker_output_dir
        )
    elif modes == {"video_all"}:
        expected = _expected_resume_results(
            condition_cases,
            condition_mode="video_all",
            run_output_dir=worker_output_dir,
        )
    else:
        return None
    actual_paths = [
        Path(str(result.get("output_dir"))).resolve()
        for result in results
        if isinstance(result, dict) and result.get("output_dir")
    ]
    if len(actual_paths) != len(results) or len(actual_paths) != len(set(actual_paths)):
        return None
    if set(expected) != set(actual_paths):
        return None
    for result in results:
        output_dir_value = (
            result.get("output_dir") if isinstance(result, dict) else None
        )
        if not output_dir_value:
            return None
        output_dir = Path(str(output_dir_value)).resolve()
        episode, condition_name = expected.get(output_dir, (None, None))
        if episode is None:
            return None
        if not _valid_result_payload(
            result,
            episode=episode,
            condition_name=condition_name,
            condition_mode=str(result.get("condition_mode")),
            output_dir=output_dir,
            checkpoint=ckpt_path,
            evaluation_embodiment=evaluation_embodiment,
            require_attention=require_attention,
            legacy_articulated_placement=legacy_articulated_placement,
        ):
            return None
        persisted = _load_completed_result(
            output_dir,
            episode=episode,
            condition_name=condition_name,
            condition_mode=str(result.get("condition_mode")),
            checkpoint=ckpt_path,
            evaluation_embodiment=evaluation_embodiment,
            require_attention=require_attention,
            legacy_articulated_placement=legacy_articulated_placement,
        )
        if persisted is None or persisted != result:
            return None
    return summary


def _build_batch_summary(
    results: list[dict[str, Any]],
    *,
    scene_root: Path,
    ckpt_path: Path,
    evaluation_embodiment: str,
    condition_embodiment: str,
    visual_episode_seed: int,
    episodes_per_task: int | None,
    workers: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    successes = sum(int(bool(result.get("success"))) for result in results)
    grouped: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    tasks: dict[str, dict[str, Any]] = {}

    for result in results:
        task_name = str(result["task_name"])
        success = int(bool(result.get("success")))
        task = tasks.setdefault(
            task_name,
            {"task_name": task_name, "total": 0, "successes": 0},
        )
        task["total"] += 1
        task["successes"] += success

        group_key = (
            Path(str(result["scene_dir"])).name,
            Path(str(result["placement_meta_path"])).parent.name,
            task_name,
            str(result["condition_embodiment"]),
        )
        group = grouped.setdefault(
            group_key,
            {
                "scene": group_key[0],
                "placement": group_key[1],
                "task_name": task_name,
                "condition_embodiment": group_key[3],
                "total": 0,
                "successes": 0,
            },
        )
        group["total"] += 1
        group["successes"] += success

    task_rows = sorted(tasks.values(), key=lambda item: item["task_name"])
    for task in task_rows:
        task["failures"] = task["total"] - task["successes"]
        task["success_rate"] = task["successes"] / task["total"]
    group_rows = sorted(
        grouped.values(),
        key=lambda item: (
            item["scene"],
            item["placement"],
            item["task_name"],
            item["condition_embodiment"],
        ),
    )
    for group in group_rows:
        group["failures"] = group["total"] - group["successes"]
        group["success_rate"] = group["successes"] / group["total"]

    total = len(results)
    summary: dict[str, Any] = {
        "scene_root": str(scene_root),
        "checkpoint": str(ckpt_path),
        "evaluation_embodiment": evaluation_embodiment,
        "condition_embodiment": condition_embodiment,
        "visual_episode_seed": visual_episode_seed,
        "episodes_per_task": episodes_per_task,
        "task_count": len(task_rows),
        "total": total,
        "successes": successes,
        "failures": total - successes,
        "success_rate": successes / total if total else 0.0,
        "macro_success_rate": (
            sum(task["success_rate"] for task in task_rows) / len(task_rows)
            if task_rows
            else 0.0
        ),
        "tasks": task_rows,
        "groups": group_rows,
        "results": results,
    }
    if workers is not None:
        summary["workers"] = workers
    return summary


def _markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _render_batch_summary_markdown(summary: dict[str, Any]) -> str:
    episode_limit = summary["episodes_per_task"]
    metadata = (
        ("Checkpoint", summary["checkpoint"]),
        ("Scene root", summary["scene_root"]),
        ("Evaluation embodiment", summary["evaluation_embodiment"]),
        ("Condition embodiment", summary["condition_embodiment"]),
        ("Episodes per task", episode_limit if episode_limit is not None else "all"),
        ("Sampling seed", summary["visual_episode_seed"]),
    )
    lines = [
        "# RoboTwin Evaluation Summary",
        "",
        "## Run",
        "",
        "| Field | Value |",
        "| --- | --- |",
    ]
    lines.extend(f"| {field} | {_markdown_cell(value)} |" for field, value in metadata)
    lines.extend(
        [
            "",
            "## Overall",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| Tasks | {summary['task_count']} |",
            f"| Evaluated | {summary['total']} |",
            f"| Successes | {summary['successes']} |",
            f"| Failures | {summary['failures']} |",
            f"| Overall success rate | {summary['success_rate']:.2%} |",
            f"| Macro task success rate | {summary['macro_success_rate']:.2%} |",
            "",
            "## Tasks",
            "",
            "| Task | Evaluated | Successes | Failures | Success Rate |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    lines.extend(
        "| {task} | {total} | {successes} | {failures} | {rate:.2%} |".format(
            task=_markdown_cell(task["task_name"]),
            total=task["total"],
            successes=task["successes"],
            failures=task["failures"],
            rate=task["success_rate"],
        )
        for task in summary["tasks"]
    )
    return "\n".join(lines) + "\n"


def _write_batch_summary(
    summary: dict[str, Any], output_dir: Path
) -> tuple[Path, Path]:
    summary_json = output_dir / "summary.json"
    summary_md = output_dir / "summary.md"
    summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary_md.write_text(_render_batch_summary_markdown(summary), encoding="utf-8")
    return summary_json, summary_md


def _run_fixed_scene_batch(
    cfg: DictConfig,
    *,
    ckpt_path: Path,
    robotwin_root: Path,
    run_output_dir: Path,
    scene_root: Path,
    shard_index: int = 0,
    shard_count: int = 1,
) -> None:
    """Evaluate the selected persisted episodes under ``scene_root``."""
    from simpleicl.evaluation import scene_eval
    from simpleicl.evaluation.policy import deploy_policy

    evaluation_embodiment = str(cfg.EVALUATION.embodiment)
    condition_embodiment = str(cfg.EVALUATION.condition_embodiment)
    visual_episode_seed = int(cfg.EVALUATION.visual_episode_seed)
    episode_limit = _episodes_per_task(cfg)
    task_names = _configured_task_names(cfg)
    resume_enabled = bool(cfg.EVALUATION.get("resume", False))
    cases = scene_eval.discover_scene_cases(
        scene_root=scene_root,
        evaluation_embodiment=evaluation_embodiment,
        condition_embodiment=condition_embodiment,
        instruction_type=str(cfg.EVALUATION.instruction_type),
        require_condition=False,
        sample_visual_episode=True,
        visual_episode_seed=visual_episode_seed,
        episodes_per_task=episode_limit,
        task_names=task_names,
    )
    cases = _filter_scene_cases(cfg, cases)
    if not cases:
        raise scene_eval.SceneEvalError(
            f"No valid fixed-scene episodes found under: {scene_root}"
        )
    if shard_count <= 0 or shard_index < 0 or shard_index >= shard_count:
        raise ValueError(f"Invalid batch shard {shard_index}/{shard_count}")
    if shard_count > 1:
        cases = cases[shard_index::shard_count]
        if not cases:
            raise scene_eval.SceneEvalError(
                f"Batch shard {shard_index} received no scene cases out of {shard_count}"
            )

    model = deploy_policy.get_model(_fixed_policy_args(cfg, ckpt_path, run_output_dir))
    has_condition = _policy_has_video_all_condition(model)
    video_all_size: list[int] | None = None
    video_all_num_frames: int | None = None
    video_all_frame_percentage: float | None = None
    video_all_max_num_frames: int | None = None
    if has_condition:
        cases = scene_eval.discover_scene_cases(
            scene_root=scene_root,
            evaluation_embodiment=evaluation_embodiment,
            condition_embodiment=condition_embodiment,
            instruction_type=str(cfg.EVALUATION.instruction_type),
            require_condition=True,
            sample_visual_episode=True,
            visual_episode_seed=visual_episode_seed,
            episodes_per_task=episode_limit,
            task_names=task_names,
        )
        cases = _filter_scene_cases(cfg, cases)
        if not cases:
            raise scene_eval.SceneEvalError(
                f"No valid fixed-scene episodes with condition videos found under: {scene_root}"
            )
        if shard_count > 1:
            cases = cases[shard_index::shard_count]
            if not cases:
                raise scene_eval.SceneEvalError(
                    f"Batch shard {shard_index} received no scene cases out of {shard_count}"
                )
        video_all_size = [int(value) for value in cfg.data.train.video_all_size]
        (
            video_all_num_frames,
            video_all_frame_percentage,
            video_all_max_num_frames,
        ) = resolve_video_sampling(cfg)

    results: list[dict[str, Any]] = []
    for case in cases:
        episode = case.episode
        rollout_conditions: tuple[Path | None, ...]
        if has_condition:
            rollout_conditions = case.condition_videos
        else:
            rollout_conditions = (None,)
        for condition_video in rollout_conditions:
            condition_name = (
                condition_video.parent.parent.name
                if condition_video is not None
                else "condition_free"
            )
            fixed_output = _fixed_scene_output_dir(
                run_output_dir, episode, condition_name
            )
            if resume_enabled:
                resumed_result = _load_completed_result(
                    fixed_output,
                    episode=episode,
                    condition_name=condition_name,
                    condition_mode="video_all"
                    if condition_video is not None
                    else "none",
                    checkpoint=ckpt_path,
                    evaluation_embodiment=evaluation_embodiment,
                    require_attention=_attention_visualization_enabled(cfg),
                    legacy_articulated_placement=bool(
                        cfg.EVALUATION.get("legacy_articulated_placement", True)
                    ),
                )
                if resumed_result is not None:
                    print(f"[eval_robotwin] resume skip output={fixed_output}")
                    results.append(resumed_result)
                    continue
            fixed_output.mkdir(parents=True, exist_ok=True)
            condition_video_link = (
                _link_condition_video(condition_video, fixed_output)
                if condition_video is not None
                else None
            )
            sampled_condition_video = (
                fixed_output / "condition_video_sampled.mp4"
                if condition_video is not None
                else None
            )
            if condition_video is not None:
                assert video_all_size is not None
                assert sampled_condition_video is not None
                model.video_all_latents = _prepare_condition_latents(
                    model,
                    condition_video,
                    video_all_size,
                    video_all_num_frames,
                    sampled_condition_video,
                    video_all_frame_percentage=video_all_frame_percentage,
                    video_all_max_num_frames=video_all_max_num_frames,
                )
            else:
                model.video_all_latents = None
            if _attention_visualization_enabled(cfg):
                if sampled_condition_video is None:
                    raise RuntimeError(
                        "Attention visualization requires a condition video"
                    )
                model.start_attention_episode(fixed_output, sampled_condition_video)
            print(f"[eval_robotwin] condition={condition_name} video={condition_video}")
            result = scene_eval.run_scene_episode(
                episode=episode,
                robotwin_root=robotwin_root,
                task_config=str(cfg.EVALUATION.task_config),
                output_dir=fixed_output,
                model=model,
                policy_module=deploy_policy,
                legacy_articulated_placement=bool(
                    cfg.EVALUATION.get("legacy_articulated_placement", True)
                ),
            )
            result.update(
                {
                    "scene_dir": str(episode.scene_dir),
                    "manifest_path": str(episode.manifest_path),
                    "placement_meta_path": str(episode.placement_meta_path),
                    "episode_data_path": str(episode.episode_data_path),
                    "instruction_path": str(episode.instruction_path),
                    "robotwin_root": str(robotwin_root),
                    "checkpoint": str(ckpt_path),
                    "override_instruction": model.override_instruction,
                    "effective_instruction": model.override_instruction
                    or episode.instruction,
                    "evaluation_embodiment": evaluation_embodiment,
                    "condition_embodiment": condition_name,
                    "condition_video_path": str(condition_video)
                    if condition_video is not None
                    else None,
                    "condition_video_link": str(condition_video_link)
                    if condition_video_link is not None
                    else None,
                    "condition_video_sampled_path": (
                        str(sampled_condition_video)
                        if sampled_condition_video is not None
                        else None
                    ),
                    "condition_mode": "video_all"
                    if condition_video is not None
                    else "none",
                    "output_dir": str(fixed_output),
                }
            )
            attention_result = model.get_attention_result()
            if attention_result is not None:
                result["attention_visualization"] = attention_result
            (fixed_output / "result.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            (fixed_output / "_result.txt").write_text(
                f"Instruction Type: {str(cfg.EVALUATION.instruction_type)}\n"
                f"Condition: {condition_name}\n\n"
                f"{result['success_rate']}",
                encoding="utf-8",
            )
            results.append(result)

    summary = _build_batch_summary(
        results,
        scene_root=scene_root,
        ckpt_path=ckpt_path,
        evaluation_embodiment=evaluation_embodiment,
        condition_embodiment=condition_embodiment,
        visual_episode_seed=visual_episode_seed,
        episodes_per_task=episode_limit,
    )
    summary_json, summary_md = _write_batch_summary(summary, run_output_dir)
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "scene_root",
                    "checkpoint",
                    "task_count",
                    "total",
                    "successes",
                    "success_rate",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(
        f"Batch fixed-scene evaluation summaries saved to: {summary_json}, {summary_md}"
    )


def _resolve_multirun_gpu_ids(cfg: DictConfig) -> list[int]:
    raw_gpu_ids = cfg.MULTIRUN.get("gpu_ids")
    if raw_gpu_ids is None or str(raw_gpu_ids).strip().lower() in {"", "none", "null"}:
        ids = list(range(int(cfg.MULTIRUN.num_gpus)))
    elif isinstance(raw_gpu_ids, str):
        ids = [int(value.strip()) for value in raw_gpu_ids.split(",") if value.strip()]
    else:
        ids = [int(value) for value in raw_gpu_ids]
    if not ids or any(value < 0 for value in ids):
        raise ValueError(f"MULTIRUN.gpu_ids must contain non-negative ids, got {ids}")
    return ids


def _collect_batch_worker_overrides() -> list[str]:
    """Forward user overrides while reserving scene batch worker controls."""
    blocked = {
        "ckpt",
        "gpu_id",
        "EVALUATION.task_name",
        "EVALUATION.scene_dir",
        "EVALUATION.output_dir",
        "EVALUATION.batch_worker",
        "EVALUATION.batch_shard",
        "EVALUATION.batch_shards",
        "EVALUATION.batch_output_subdir",
        "EVALUATION.visual_episode_seed",
        "EVALUATION.episodes_per_task",
        "EVALUATION.resume",
        "EVALUATION.resume_dir",
    }
    overrides = []
    for raw in HydraConfig.get().overrides.task:
        key = str(raw).split("=", 1)[0].lstrip("+~").strip()
        if key in blocked or key.startswith("MULTIRUN.") or key.startswith("hydra."):
            continue
        overrides.append(str(raw))
    return overrides


def _run_fixed_scene_distributed(
    cfg: DictConfig,
    *,
    ckpt_path: Path,
    robotwin_root: Path,
    run_output_dir: Path,
    scene_root: Path,
) -> None:
    """Shard fixed-scene cases across independent CUDA worker processes."""
    from simpleicl.evaluation import scene_eval

    episode_limit = _episodes_per_task(cfg)
    task_names = _configured_task_names(cfg)
    resume_enabled = bool(cfg.EVALUATION.get("resume", False))
    cases = scene_eval.discover_scene_cases(
        scene_root=scene_root,
        evaluation_embodiment=str(cfg.EVALUATION.embodiment),
        condition_embodiment=str(cfg.EVALUATION.condition_embodiment),
        instruction_type=str(cfg.EVALUATION.instruction_type),
        require_condition=False,
        sample_visual_episode=True,
        visual_episode_seed=int(cfg.EVALUATION.visual_episode_seed),
        episodes_per_task=episode_limit,
        task_names=task_names,
    )
    cases = _filter_scene_cases(cfg, cases)
    condition_cases: list[Any] = []
    if resume_enabled:
        try:
            condition_cases = scene_eval.discover_scene_cases(
                scene_root=scene_root,
                evaluation_embodiment=str(cfg.EVALUATION.embodiment),
                condition_embodiment=str(cfg.EVALUATION.condition_embodiment),
                instruction_type=str(cfg.EVALUATION.instruction_type),
                require_condition=True,
                sample_visual_episode=True,
                visual_episode_seed=int(cfg.EVALUATION.visual_episode_seed),
                episodes_per_task=episode_limit,
                task_names=task_names,
            )
            condition_cases = _filter_scene_cases(cfg, condition_cases)
        except scene_eval.SceneEvalError:
            # A checkpoint without video_all does not need condition discovery.
            condition_cases = []
    gpu_ids = _resolve_multirun_gpu_ids(cfg)
    slots_per_gpu = int(cfg.MULTIRUN.max_tasks_per_gpu)
    if slots_per_gpu <= 0:
        raise ValueError("MULTIRUN.max_tasks_per_gpu must be > 0")
    worker_gpus = [gpu for gpu in gpu_ids for _ in range(slots_per_gpu)]
    worker_count = min(len(worker_gpus), len(cases))
    worker_gpus = worker_gpus[:worker_count]
    if worker_count <= 1:
        _run_fixed_scene_batch(
            cfg,
            ckpt_path=ckpt_path,
            robotwin_root=robotwin_root,
            run_output_dir=run_output_dir,
            scene_root=scene_root,
        )
        return

    worker_root = run_output_dir / "workers"
    worker_root.mkdir(parents=True, exist_ok=True)
    overrides = _collect_batch_worker_overrides()
    reused_summaries: dict[int, dict[str, Any]] = {}
    processes: list[tuple[int, int, subprocess.Popen[bytes], Any]] = []
    for index, gpu_id in enumerate(worker_gpus):
        worker_dir_name = f"worker_{index:03d}"
        worker_output_dir = worker_root / worker_dir_name
        summary_path = worker_output_dir / "summary.json"
        if resume_enabled:
            reused = _load_complete_worker_summary(
                summary_path,
                worker_output_dir=worker_output_dir,
                cases=cases[index::worker_count],
                condition_cases=condition_cases[index::worker_count],
                scene_root=scene_root,
                ckpt_path=ckpt_path,
                evaluation_embodiment=str(cfg.EVALUATION.embodiment),
                condition_embodiment=str(cfg.EVALUATION.condition_embodiment),
                visual_episode_seed=int(cfg.EVALUATION.visual_episode_seed),
                episodes_per_task=episode_limit,
                require_attention=_attention_visualization_enabled(cfg),
                legacy_articulated_placement=bool(
                    cfg.EVALUATION.get("legacy_articulated_placement", True)
                ),
            )
            if reused is not None:
                reused_summaries[index] = reused
                print(f"[eval_robotwin] resume reuse worker={worker_dir_name}")
                continue
        worker_log_path = worker_root / f"{worker_dir_name}.log"
        command = [
            sys.executable,
            "-m",
            "simpleicl.evaluation.evaluate",
            f"ckpt={ckpt_path}",
            f"gpu_id={gpu_id}",
            "EVALUATION.task_name=null",
            f"EVALUATION.scene_dir={scene_root}",
            f"EVALUATION.output_dir={run_output_dir}",
            "EVALUATION.batch_worker=true",
            f"EVALUATION.batch_shard={index}",
            f"EVALUATION.batch_shards={worker_count}",
            f"EVALUATION.batch_output_subdir=workers/{worker_dir_name}",
            f"EVALUATION.visual_episode_seed={int(cfg.EVALUATION.visual_episode_seed)}",
            f"EVALUATION.episodes_per_task={episode_limit if episode_limit is not None else 'null'}",
        ]
        if resume_enabled:
            command.extend(
                [
                    "EVALUATION.resume=true",
                    f"EVALUATION.resume_dir={run_output_dir}",
                ]
            )
        command.extend(overrides)
        log_handle = worker_log_path.open("w", encoding="utf-8")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["PYTHONUNBUFFERED"] = "1"
        print(f"[eval_robotwin] launch worker={worker_dir_name} gpu={gpu_id}")
        process = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        processes.append((index, gpu_id, process, log_handle))

    failed: list[str] = []
    for index, gpu_id, process, log_handle in processes:
        return_code = process.wait()
        log_handle.close()
        if return_code != 0:
            failed.append(f"worker_{index:03d}(gpu={gpu_id}, exit={return_code})")
    if failed:
        raise RuntimeError("Fixed-scene worker failure: " + ", ".join(failed))

    shard_summaries = []
    workers = []
    launched_indices = {index for index, _, _, _ in processes}
    for index, gpu_id in enumerate(worker_gpus):
        if index in reused_summaries:
            summary = reused_summaries[index]
            status = "reused"
            summary_path = worker_root / f"worker_{index:03d}" / "summary.json"
        else:
            if index not in launched_indices:
                raise RuntimeError(f"Missing process state for worker_{index:03d}")
            summary_path = worker_root / f"worker_{index:03d}" / "summary.json"
            if not summary_path.is_file():
                raise FileNotFoundError(f"Worker summary not found: {summary_path}")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            status = "completed"
        shard_summaries.append(summary)
        workers.append(
            {
                "worker": index,
                "gpu_id": gpu_id,
                "summary": str(summary_path),
                "status": status,
            }
        )
    results = [
        result for summary in shard_summaries for result in summary.get("results", [])
    ]
    merged = _build_batch_summary(
        results,
        scene_root=scene_root,
        ckpt_path=ckpt_path,
        evaluation_embodiment=str(cfg.EVALUATION.embodiment),
        condition_embodiment=str(cfg.EVALUATION.condition_embodiment),
        visual_episode_seed=int(cfg.EVALUATION.visual_episode_seed),
        episodes_per_task=episode_limit,
        workers=workers,
    )
    summary_json, summary_md = _write_batch_summary(merged, run_output_dir)
    print(
        f"Batch fixed-scene evaluation summaries saved to: {summary_json}, {summary_md}"
    )


@hydra.main(
    version_base="1.3", config_path="../configs", config_name="sim_robotwin.yaml"
)
def main(cfg: DictConfig):
    if cfg.ckpt is None:
        raise ValueError("`ckpt` must not be None.")

    ckpt_path = _resolve_path(str(cfg.ckpt), base=PROJECT_ROOT)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    sim_task = HydraConfig.get().runtime.choices.get("task")
    result_alias = cfg.EVALUATION.get("result_name")
    result_name = str(sim_task if result_alias is None else result_alias)
    if result_alias is not None and (
        not result_name.strip()
        or result_name in {".", ".."}
        or Path(result_name).name != result_name
    ):
        raise ValueError(
            "EVALUATION.result_name must be a single directory name, "
            f"got: {result_name!r}"
        )

    robotwin_root = _resolve_path(str(cfg.EVALUATION.robotwin_root), base=PROJECT_ROOT)
    if not robotwin_root.exists():
        raise FileNotFoundError(f"RoboTwin root not found: {robotwin_root}")

    output_dir = _resolve_path(str(cfg.EVALUATION.output_dir), base=PROJECT_ROOT)
    result_subdir = _resolve_result_subdir(cfg)
    scene_dir = _resolve_optional_path(cfg.EVALUATION.scene_dir, base=PROJECT_ROOT)
    if scene_dir is None:
        raise ValueError("EVALUATION.scene_dir is required for fixed-scene evaluation")
    # Use the public directory name even when data is linked to a legacy archive.
    scene_root_name = Path(
        os.path.expanduser(os.path.expandvars(str(cfg.EVALUATION.scene_dir)))
    ).name
    is_scene_root = (
        scene_dir is not None and not (scene_dir / "manifest.json").is_file()
    )
    resume_enabled = bool(cfg.EVALUATION.get("resume", False))
    resume_dir = _resolve_optional_path(
        cfg.EVALUATION.get("resume_dir"), base=PROJECT_ROOT
    )
    if resume_enabled:
        if resume_dir is None:
            raise ValueError(
                "EVALUATION.resume=true requires EVALUATION.resume_dir to point "
                "to an existing fixed-scene run directory."
            )
        if not resume_dir.is_dir():
            raise FileNotFoundError(f"Resume run directory not found: {resume_dir}")
        run_output_dir = resume_dir
        run_ts = run_output_dir.name
    else:
        run_ts = output_dir.name
        if run_ts == "":
            raise ValueError(
                f"Invalid EVALUATION.output_dir (missing run_ts): {output_dir}"
            )
        if is_scene_root and scene_dir is not None:
            run_output_dir = (
                OUTPUTS
                / "evaluation"
                / "robotwin"
                / result_name
                / scene_root_name
                / run_ts
            )
        else:
            run_output_dir = OUTPUTS / "evaluation" / "robotwin" / result_name
            if result_subdir is not None:
                run_output_dir = run_output_dir / result_subdir
            run_output_dir = run_output_dir / run_ts
        run_output_dir.mkdir(parents=True, exist_ok=True)

    if resume_enabled and (scene_dir is None or not is_scene_root):
        raise ValueError(
            "EVALUATION.resume is supported only for fixed scene-root evaluation "
            "with EVALUATION.scene_dir."
        )
    if cfg.EVALUATION.task_name is None and (scene_dir is None or not is_scene_root):
        raise ValueError(
            "`EVALUATION.task_name` is required for random or single-scene evaluation; "
            "omit it only when EVALUATION.scene_dir points to a scene root."
        )
    effective_robotwin_root = robotwin_root
    if scene_dir is not None:
        from simpleicl.evaluation.scene_eval import (
            infer_robotwin_root,
            validate_robotwin_root,
        )

        effective_robotwin_root = infer_robotwin_root(scene_dir, robotwin_root)
        effective_robotwin_root = validate_robotwin_root(effective_robotwin_root)
        print(f"[eval_robotwin] using RoboTwin checkout: {effective_robotwin_root}")

    policy_source_dir = (Path(__file__).parent / POLICY_NAME).resolve()
    if not policy_source_dir.is_dir():
        raise FileNotFoundError(
            f"Policy source directory not found: {policy_source_dir}"
        )

    if scene_dir is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)
        if is_scene_root:
            if cfg.EVALUATION.visual_episode_seed is None:
                cfg.EVALUATION.visual_episode_seed = secrets.randbits(63)
            if bool(cfg.EVALUATION.batch_worker):
                worker_subdir = cfg.EVALUATION.batch_output_subdir
                if worker_subdir is not None and str(worker_subdir).strip():
                    worker_path = Path(str(worker_subdir))
                    if worker_path.is_absolute() or ".." in worker_path.parts:
                        raise ValueError(
                            "EVALUATION.batch_output_subdir must be a relative safe path"
                        )
                    run_output_dir = run_output_dir / worker_path
                    run_output_dir.mkdir(parents=True, exist_ok=True)
                _run_fixed_scene_batch(
                    cfg,
                    ckpt_path=ckpt_path,
                    robotwin_root=effective_robotwin_root,
                    run_output_dir=run_output_dir,
                    scene_root=scene_dir,
                    shard_index=int(cfg.EVALUATION.batch_shard),
                    shard_count=int(cfg.EVALUATION.batch_shards),
                )
            elif bool(cfg.MULTIRUN.enabled):
                _run_fixed_scene_distributed(
                    cfg,
                    ckpt_path=ckpt_path,
                    robotwin_root=effective_robotwin_root,
                    run_output_dir=run_output_dir,
                    scene_root=scene_dir,
                )
            else:
                _run_fixed_scene_batch(
                    cfg,
                    ckpt_path=ckpt_path,
                    robotwin_root=effective_robotwin_root,
                    run_output_dir=run_output_dir,
                    scene_root=scene_dir,
                )
        else:
            _run_fixed_scene(
                cfg,
                ckpt_path=ckpt_path,
                robotwin_root=effective_robotwin_root,
                run_output_dir=run_output_dir,
                scene_dir=scene_dir,
            )
        OmegaConf.save(
            config=cfg,
            f=str(run_output_dir / "eval_config_scene.yaml"),
        )
        return


if __name__ == "__main__":
    main()
