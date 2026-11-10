import logging
import os
import sys
import time
import inspect
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

import av
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from simpleicl.paths import ROOT as PROJECT_ROOT, CONFIGS
from simpleicl.evaluation.settings import resolve_video_sampling

SRC_ROOT = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from simpleicl.datasets.lerobot.processors.simpleicl_processor import SimpleICLProcessor
from simpleicl.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from simpleicl.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from simpleicl.utils.video_io import save_mp4, save_video_to_mp4
from simpleicl.utils.attention_visualization import AttentionVisualizationCollector
from simpleicl.utils.video_sampling import (
    align_frame_count_to_vae,
    resolve_video_frame_count,
)

logger = logging.getLogger(__name__)


def _is_none_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "none", "null"}
    return False


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    raise ValueError(f"Cannot parse bool value: {value}")


def _parse_optional_int(value: Any) -> Optional[int]:
    if _is_none_like(value):
        return None
    return int(value)


def _parse_optional_float(value: Any) -> Optional[float]:
    if _is_none_like(value):
        return None
    return float(value)


def _build_instruction_prompt(
    instruction: str,
    override_instruction: Optional[str] = None,
) -> str:
    task = (
        instruction
        if _is_none_like(override_instruction)
        else str(override_instruction)
    )
    return DEFAULT_PROMPT.format(task=task)


def _normalize_mixed_precision(mixed_precision: str) -> str:
    key = str(mixed_precision).strip().lower()
    if key not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported mixed_precision: {mixed_precision}. "
            "Expected one of: ['no', 'fp16', 'bf16']."
        )
    return key


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    precision = _normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def _resolve_sim_cfg_name(
    sim_cfg_path: Optional[str], sim_cfg_name: Optional[str]
) -> str:
    configs_root = (CONFIGS).resolve()
    if not _is_none_like(sim_cfg_path):
        cfg_path = Path(str(sim_cfg_path)).expanduser().resolve()
        try:
            relative = cfg_path.relative_to(configs_root)
        except ValueError as exc:
            raise ValueError(
                f"`sim_cfg_path` must be under {configs_root}, got: {cfg_path}"
            ) from exc
        return relative.as_posix()

    if _is_none_like(sim_cfg_name):
        return "sim_robotwin.yaml"
    return str(sim_cfg_name)


def _compose_sim_cfg(
    sim_cfg_path: Optional[str],
    sim_cfg_name: Optional[str],
    sim_task: Optional[str],
    extra_overrides: Optional[list[str]] = None,
) -> DictConfig:
    config_name = _resolve_sim_cfg_name(
        sim_cfg_path=sim_cfg_path, sim_cfg_name=sim_cfg_name
    )
    configs_root = (CONFIGS).resolve()
    overrides = []
    if not _is_none_like(sim_task):
        overrides.append(f"task={str(sim_task)}")
    if extra_overrides:
        overrides.extend(extra_overrides)

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_dir(version_base="1.3", config_dir=str(configs_root)):
        cfg = compose(config_name=config_name, overrides=overrides)
    return cfg


def _resolve_dataset_stats_path(dataset_stats_path: Optional[str]) -> Path:
    if _is_none_like(dataset_stats_path):
        raise FileNotFoundError(
            "`dataset_stats_path` is required. "
            "Please pass it from eval entrypoint overrides."
        )
    resolved = Path(str(dataset_stats_path)).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Dataset stats path not found: {resolved}")
    return resolved


def _use_local_pretrained_model(cfg: DictConfig, pretrained_model_path: Any) -> None:
    if _is_none_like(pretrained_model_path):
        return

    model_path = (
        Path(os.path.expandvars(str(pretrained_model_path))).expanduser().resolve()
    )
    required_files = [
        model_path / "Wan2.2_VAE.pth",
        model_path / "models_t5_umt5-xxl-enc-bf16.pth",
        model_path / "google" / "umt5-xxl" / "tokenizer_config.json",
    ]
    missing = [str(path) for path in required_files if not path.is_file()]
    if not any(model_path.glob("diffusion_pytorch_model*.safetensors")):
        missing.append(str(model_path / "diffusion_pytorch_model*.safetensors"))
    if missing:
        raise FileNotFoundError(
            "Incomplete local Wan2.2 model snapshot. Missing: " + ", ".join(missing)
        )

    cfg.model.model_id = str(model_path)
    cfg.model.tokenizer_model_id = str(model_path)
    cfg.model.redirect_common_files = False
    logger.info("Using local Wan2.2 model snapshot: %s", model_path)


def _resize_rgb(image: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    pil_image = Image.fromarray(image.astype(np.uint8), mode="RGB")
    resized = pil_image.resize(size_wh, resample=Image.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


@torch.no_grad()
def _prepare_video_all_latents(
    model: torch.nn.Module,
    video_path: str,
    video_all_size: list[int],
    device: str,
    dtype: torch.dtype,
    num_frames: Optional[int] = 16,
    sampled_video_path: Optional[str] = None,
    frame_percentage: Optional[float] = None,
    max_num_frames: Optional[int] = None,
) -> torch.Tensor:
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        total = stream.frames if stream.frames and stream.frames > 0 else None
        target_frames = resolve_video_frame_count(
            total,
            frame_percentage=frame_percentage,
            num_frames=num_frames,
            max_num_frames=max_num_frames,
        )
        if target_frames is None:
            target_frames = 16
        if total is not None:
            available_frames = int(total)
            if max_num_frames is not None:
                available_frames = min(available_frames, int(max_num_frames))
            target_frames = align_frame_count_to_vae(
                target_frames,
                max_num_frames=available_frames,
            )
        if total is not None and total > target_frames:
            needed = set(np.linspace(0, total - 1, target_frames, dtype=int).tolist())
        else:
            needed = None
        frames = []
        for i, frame in enumerate(container.decode(video=0)):
            if needed is None or i in needed:
                frames.append(
                    torch.from_numpy(frame.to_ndarray(format="rgb24")).permute(2, 0, 1)
                )
            if needed is not None and len(frames) >= target_frames:
                break

    if not frames:
        raise ValueError(f"Video contains no decodable frames: {video_path}")
    video = torch.stack(frames).float() / 255.0
    if total is None:
        if frame_percentage is not None:
            target_frames = resolve_video_frame_count(
                int(video.shape[0]),
                frame_percentage=frame_percentage,
                max_num_frames=max_num_frames,
            )
        available_frames = int(video.shape[0])
        if max_num_frames is not None:
            available_frames = min(available_frames, int(max_num_frames))
        target_frames = align_frame_count_to_vae(
            target_frames,
            max_num_frames=available_frames,
        )
    if needed is None and video.shape[0] > target_frames:
        indices = np.linspace(0, video.shape[0] - 1, target_frames, dtype=int)
        video = video[indices]
    if video.shape[0] < target_frames:
        raise ValueError(
            f"Decoded only {video.shape[0]} frames, but {target_frames} "
            f"were required for VAE-aligned sampling: {video_path}"
        )

    target_h, target_w = video_all_size
    _, C, H, W = video.shape
    scale = max(target_w / W, target_h / H)
    new_h, new_w = round(H * scale), round(W * scale)
    video = torch.nn.functional.interpolate(
        video,
        size=(new_h, new_w),
        mode="bilinear",
        align_corners=False,
    )
    crop_top = (new_h - target_h) // 2
    crop_left = (new_w - target_w) // 2
    video = video[
        :, :, crop_top : crop_top + target_h, crop_left : crop_left + target_w
    ]

    video = video * 2.0 - 1.0
    video = video.permute(1, 0, 2, 3).unsqueeze(0)  # [1, C, T, H, W]

    T_va = video.shape[2]
    if T_va % 4 != 1:
        raise ValueError(
            f"VAE-aligned video_all sampling produced invalid T={T_va}; "
            "expected T % 4 == 1."
        )

    video = video.to(device=device, dtype=dtype)
    if sampled_video_path is not None:
        save_video_to_mp4(
            video,
            sampled_video_path,
            fps=8,
            value_range="[-1,1]",
        )
        logger.info("Saved sampled video_all model input to %s", sampled_video_path)

    return model._encode_cond_latents(video)


class WorldActionRobotWinPolicy:
    def __init__(
        self,
        model_cfg: DictConfig,
        processor_cfg: DictConfig,
        checkpoint_path: str,
        dataset_stats_path: Path,
        device: str,
        model_dtype: torch.dtype,
        action_horizon: int,
        replan_steps: int,
        num_inference_steps: int,
        sigma_shift: Optional[float],
        seed: Optional[int],
        text_cfg_scale: float,
        negative_prompt: str,
        override_instruction: Optional[str],
        rand_device: str,
        tiled: bool,
        timing_enabled: bool,
        num_video_frames: int,
        vis_video: bool = False,
        vis_output_dir: Optional[str] = None,
        attention_visualization: Optional[dict[str, Any]] = None,
    ) -> None:
        model_cfg_copy = OmegaConf.create(
            OmegaConf.to_container(model_cfg, resolve=True)
        )
        model_cfg_copy.load_text_encoder = True

        self.model = instantiate(model_cfg_copy, model_dtype=model_dtype, device=device)
        self.model.load_checkpoint(checkpoint_path)
        self.model = self.model.to(device).eval()

        self.processor: SimpleICLProcessor = instantiate(processor_cfg).eval()
        dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
        self.processor.set_normalizer_from_stats(dataset_stats)

        self.action_horizon = int(action_horizon)
        self.replan_steps = int(max(1, min(replan_steps, action_horizon)))
        self.num_inference_steps = int(num_inference_steps)
        self.sigma_shift = sigma_shift
        self.seed = seed
        self.text_cfg_scale = float(text_cfg_scale)
        self.negative_prompt = str(negative_prompt)
        self.override_instruction = (
            None if _is_none_like(override_instruction) else str(override_instruction)
        )
        self.rand_device = str(rand_device)
        self.tiled = bool(tiled)
        self.timing_enabled = bool(timing_enabled)
        self._num_video_frames = int(num_video_frames)
        self.vis_video = bool(vis_video)
        self.vis_output_dir = (
            Path(vis_output_dir) if not _is_none_like(vis_output_dir) else None
        )
        self._vis_chunk_count = 0
        attention_cfg = dict(attention_visualization or {})
        self.attention_visualization_enabled = _parse_bool(
            attention_cfg.get("enabled", False)
        )
        self._attention_result: Optional[dict[str, Any]] = None
        self._attention_collector = (
            AttentionVisualizationCollector(
                self.model,
                save_stats=_parse_bool(attention_cfg.get("save_stats", True)),
                fps=int(attention_cfg.get("fps", 8)),
            )
            if self.attention_visualization_enabled
            else None
        )
        self.video_all_path = None
        self.video_all_size = None
        self.video_all_num_frames: Optional[int] = 16
        self.video_all_frame_percentage: Optional[float] = None
        self.video_all_max_num_frames: Optional[int] = None
        self.video_all_latents = None
        self._video_all_seeds: Optional[list[int]] = None

        self.pending_actions: deque[np.ndarray] = deque()
        self.episode_count = 0
        self.step_count = 0
        self._timing_rollout = {"infer_s": 0.0, "sim_s": 0.0}

        logger.info(
            "Initialized WorldActionRobotWinPolicy | ckpt=%s | stats=%s | horizon=%d | replan=%d",
            checkpoint_path,
            dataset_stats_path,
            self.action_horizon,
            self.replan_steps,
        )

    def _load_video_all_seeds(self) -> list[int]:
        seed_file = Path(self.video_all_path) / "seed.txt"
        if not seed_file.exists():
            raise FileNotFoundError(f"video_all seed file not found: {seed_file}")
        seeds = []
        for line in seed_file.read_text().split():
            line = line.strip()
            if line:
                seeds.append(int(line))
        self._video_all_seeds = seeds
        return seeds

    def update_video_all_for_seed(self, now_seed: int) -> None:
        if self.video_all_path is None:
            return
        seeds = self._load_video_all_seeds()
        if now_seed not in seeds:
            logger.warning(
                "seed %d not found in %s/seed.txt, clearing video_all_latents",
                now_seed,
                self.video_all_path,
            )
            self.video_all_latents = None
            return
        episode_idx = seeds.index(now_seed)
        video_file = Path(self.video_all_path) / "video" / f"episode{episode_idx}.mp4"
        if not video_file.exists():
            raise FileNotFoundError(
                f"video_all episode file not found: {video_file} "
                f"(seed={now_seed}, index={episode_idx})"
            )
        device = str(next(self.model.parameters()).device)
        dtype = next(self.model.parameters()).dtype
        self.video_all_latents = _prepare_video_all_latents(
            self.model,
            str(video_file),
            self.video_all_size,
            device,
            dtype,
            num_frames=self.video_all_num_frames,
            frame_percentage=self.video_all_frame_percentage,
            max_num_frames=self.video_all_max_num_frames,
        )
        logger.info(
            "Loaded video_all for seed=%d from %s",
            now_seed,
            video_file,
        )

    def _normalize_state(self, state: np.ndarray) -> torch.Tensor:
        state_meta = self.processor.shape_meta["state"]
        if len(state_meta) != 1:
            raise ValueError(
                "Expected exactly one merged state key in shape_meta['state']."
            )
        state_key = state_meta[0]["key"]

        state_batch = {
            "state": {
                state_key: torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            }
        }
        state_batch = self.processor.action_state_transform(state_batch)
        state_batch = self.processor.normalizer.forward(state_batch)
        return state_batch["state"][state_key]

    def _denormalize_action(self, action: torch.Tensor) -> np.ndarray:
        if action.ndim == 2:
            action = action.unsqueeze(0)
        if action.ndim != 3:
            raise ValueError(
                f"Expected action tensor [B,T,D], got {tuple(action.shape)}"
            )

        action_meta = self.processor.shape_meta["action"]
        if len(action_meta) != 1:
            raise ValueError(
                "Expected exactly one merged action key in shape_meta['action']."
            )

        action_key = action_meta[0]["key"]
        normalizer = self.processor.normalizer.normalizers["action"][action_key]
        denorm = normalizer.backward(action.to(dtype=torch.float32, device="cpu"))
        return denorm.numpy()

    def _build_robotwin_image_tensor(self, observation: Dict[str, Any]) -> torch.Tensor:
        image = self._build_robotwin_image(observation)
        image_tensor = (
            torch.from_numpy(image)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(
                device=self.model.device,
                dtype=self.model.torch_dtype,
            )
        )
        image_tensor = image_tensor * (2.0 / 255.0) - 1.0
        return image_tensor

    def _build_robotwin_image(self, observation: Dict[str, Any]) -> np.ndarray:
        obs_data = observation["observation"]
        head = _resize_rgb(obs_data["head_camera"]["rgb"], (320, 256))
        left = _resize_rgb(obs_data["left_camera"]["rgb"], (160, 128))
        right = _resize_rgb(obs_data["right_camera"]["rgb"], (160, 128))
        bottom = np.concatenate([left, right], axis=1)
        return np.concatenate([head, bottom], axis=0)  # [384, 320, 3]

    def _infer_action_chunk(
        self, observation: Dict[str, Any], instruction: str
    ) -> np.ndarray:
        image = self._build_robotwin_image(observation)
        image_tensor = (
            torch.from_numpy(image)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(
                device=self.model.device,
                dtype=self.model.torch_dtype,
            )
        )
        image_tensor = image_tensor * (2.0 / 255.0) - 1.0
        state_vector = np.asarray(
            observation["joint_action"]["vector"], dtype=np.float32
        )
        proprio = self._normalize_state(state_vector)

        prompt = _build_instruction_prompt(instruction, self.override_instruction)
        infer_kwargs = {
            "prompt": prompt,
            "input_image": image_tensor,
            "action_horizon": self.action_horizon,
            "proprio": proprio,
            "negative_prompt": self.negative_prompt,
            "text_cfg_scale": self.text_cfg_scale,
            "num_inference_steps": self.num_inference_steps,
            "sigma_shift": self.sigma_shift,
            "seed": self.seed,
            "rand_device": self.rand_device,
            "tiled": self.tiled,
        }
        infer_kwargs["input_image"] = infer_kwargs["input_image"][:, [2, 1, 0], :, :]
        if self.video_all_latents is not None:
            infer_kwargs["video_all_latents"] = self.video_all_latents
        if self._attention_collector is not None:
            if self.video_all_latents is None:
                raise RuntimeError("Attention visualization requires video_all_latents")
            self._attention_collector.start_replan(
                observation=image,
                sim_step=self.step_count,
                video_all_latents=self.video_all_latents,
            )
        infer_t0 = time.perf_counter() if self.timing_enabled else 0.0
        try:
            with torch.no_grad():
                if self.vis_video:
                    # Jointly infer the action chunk and its accompanying video so the
                    # predicted rollout video can be visualized/saved for this replan.
                    infer_kwargs["num_video_frames"] = int(self._num_video_frames)
                    infer_kwargs["test_action_with_infer_action"] = False
                    pred = self.model.infer_joint(**infer_kwargs)
                    self._save_vis_video(pred.get("video"))
                else:
                    if (
                        "num_video_frames"
                        in inspect.signature(self.model.infer_action).parameters
                    ):
                        infer_kwargs["num_video_frames"] = int(self._num_video_frames)
                    pred = self.model.infer_action(**infer_kwargs)
        except Exception:
            if self._attention_collector is not None:
                self._attention_collector.abort_replan()
            raise
        else:
            if self._attention_collector is not None:
                self._attention_collector.finish_replan()
        if self.timing_enabled:
            self._timing_rollout["infer_s"] += time.perf_counter() - infer_t0
        action_tensor = pred["action"]  # [T, D]
        action_chunk = self._denormalize_action(action_tensor)[0]  # [T, D]
        return action_chunk

    def _save_vis_video(self, frames: Optional[list]) -> None:
        if frames is None:
            logger.warning(
                "vis_video enabled but infer_joint returned no video frames."
            )
            return
        if self.vis_output_dir is None:
            logger.warning(
                "vis_video enabled but no vis_output_dir provided; skip saving."
            )
            return
        out_dir = self.vis_output_dir / "vis_video" / f"episode{self.episode_count}"
        out_dir.mkdir(parents=True, exist_ok=True)
        video_path = out_dir / f"chunk_{self._vis_chunk_count:04d}.mp4"
        try:
            save_mp4(frames, str(video_path), fps=8)
            logger.info("Saved predicted video for chunk to %s", video_path)
        except Exception as e:
            logger.warning("Failed to save vis_video to %s: %s", video_path, e)
        self._vis_chunk_count += 1

    def _fill_action_queue(self, observation: Dict[str, Any], instruction: str) -> None:
        action_chunk = self._infer_action_chunk(
            observation=observation, instruction=instruction
        )
        n_exec = min(self.replan_steps, action_chunk.shape[0])
        for i in range(n_exec):
            self.pending_actions.append(np.asarray(action_chunk[i], dtype=np.float32))

    def should_request_observation(self) -> bool:
        return not self.pending_actions

    def step(self, task_env, observation: Optional[Dict[str, Any]]) -> None:
        if not self.pending_actions:
            if observation is None:
                raise ValueError(
                    "Observation is required when action queue is empty "
                    "(replan step for simpleicl)."
                )
            instruction = task_env.get_instruction()
            self._fill_action_queue(observation=observation, instruction=instruction)
        if not self.pending_actions:
            logger.warning("No action generated; skip current eval step.")
            return

        action = self.pending_actions.popleft()
        sim_t0 = time.perf_counter() if self.timing_enabled else 0.0
        task_env.take_action(action, action_type="qpos")
        if self.timing_enabled:
            self._timing_rollout["sim_s"] += time.perf_counter() - sim_t0
        self.step_count += 1

    def reset_timing_rollout(self) -> None:
        self._timing_rollout["infer_s"] = 0.0
        self._timing_rollout["sim_s"] = 0.0

    def start_attention_episode(
        self, output_dir: str | Path, condition_video_path: str | Path
    ) -> None:
        if self._attention_collector is None:
            return
        self._attention_result = None
        self._attention_collector.start_episode(output_dir, condition_video_path)

    def finish_attention_episode(self) -> dict[str, Any]:
        if self._attention_collector is None:
            return {}
        self._attention_result = self._attention_collector.finish_episode()
        return dict(self._attention_result)

    def get_attention_result(self) -> Optional[dict[str, Any]]:
        return (
            dict(self._attention_result) if self._attention_result is not None else None
        )

    def get_timing_rollout(self) -> Dict[str, float]:
        return {
            "infer_s": float(self._timing_rollout["infer_s"]),
            "sim_s": float(self._timing_rollout["sim_s"]),
        }

    def reset(self) -> None:
        self.pending_actions.clear()
        self.episode_count += 1
        self.step_count = 0
        self._vis_chunk_count = 0
        self.reset_timing_rollout()


def encode_obs(observation: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    return observation


def get_model(usr_args: Dict[str, Any]):
    sim_cfg_path = usr_args.get("sim_cfg_path")
    sim_cfg_name = usr_args.get("sim_cfg_name")
    sim_task = usr_args.get("sim_task")
    # Forward any extra Hydra overrides (model.*, data.*) so that the
    # composed config matches the training config used for the checkpoint.
    extra_overrides_raw = usr_args.get("sim_extra_overrides")
    extra_overrides: Optional[list[str]] = None
    if not _is_none_like(extra_overrides_raw):
        if isinstance(extra_overrides_raw, str):
            # Serialized as a single comma-separated string or JSON list
            import json as _json

            try:
                extra_overrides = _json.loads(extra_overrides_raw)
            except (ValueError, TypeError):
                extra_overrides = [
                    s.strip() for s in extra_overrides_raw.split(",") if s.strip()
                ]
        elif isinstance(extra_overrides_raw, (list, tuple)):
            extra_overrides = list(extra_overrides_raw)
    cfg = _compose_sim_cfg(
        sim_cfg_path=sim_cfg_path,
        sim_cfg_name=sim_cfg_name,
        sim_task=sim_task,
        extra_overrides=extra_overrides,
    )
    _use_local_pretrained_model(cfg, usr_args.get("pretrained_model_path"))

    checkpoint_path = usr_args.get("ckpt_setting")
    if _is_none_like(checkpoint_path):
        raise ValueError(
            "`ckpt_setting` is required and must be a valid checkpoint path."
        )

    device = str(usr_args.get("device") or cfg.EVALUATION.get("device") or "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA is unavailable; fallback device to cpu.")
        device = "cpu"

    mixed_precision = str(
        usr_args.get("mixed_precision") or cfg.get("mixed_precision", "bf16")
    )
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)

    dataset_stats_path = _resolve_dataset_stats_path(
        dataset_stats_path=usr_args.get("dataset_stats_path"),
    )

    action_horizon = _parse_optional_int(usr_args.get("action_horizon"))
    if action_horizon is None:
        eval_horizon = _parse_optional_int(cfg.EVALUATION.get("action_horizon"))
        action_horizon = (
            eval_horizon
            if eval_horizon is not None
            else int(cfg.data.train.num_frames) - 1
        )
    if action_horizon <= 0:
        raise ValueError(f"`action_horizon` must be positive, got {action_horizon}")

    replan_steps = _parse_optional_int(usr_args.get("replan_steps"))
    if replan_steps is None:
        replan_steps = int(cfg.EVALUATION.get("replan_steps", 8))

    num_inference_steps = _parse_optional_int(usr_args.get("num_inference_steps"))
    if num_inference_steps is None:
        num_inference_steps = int(
            cfg.EVALUATION.get("num_inference_steps", cfg.eval_num_inference_steps)
        )

    sigma_shift = _parse_optional_float(usr_args.get("sigma_shift"))
    if sigma_shift is None:
        sigma_shift = _parse_optional_float(cfg.EVALUATION.get("sigma_shift"))

    seed = _parse_optional_int(usr_args.get("seed"))
    text_cfg_scale = float(
        usr_args.get("text_cfg_scale", cfg.EVALUATION.get("text_cfg_scale", 1.0))
    )
    negative_prompt = str(
        usr_args.get("negative_prompt", cfg.EVALUATION.get("negative_prompt", ""))
    )
    override_instruction_raw = usr_args.get(
        "override_instruction", cfg.EVALUATION.get("override_instruction")
    )
    override_instruction = (
        None
        if _is_none_like(override_instruction_raw)
        else str(override_instruction_raw)
    )
    rand_device = str(
        usr_args.get("rand_device", cfg.EVALUATION.get("rand_device", "cpu"))
    )
    tiled = _parse_bool(usr_args.get("tiled", cfg.EVALUATION.get("tiled", False)))
    timing_enabled = _parse_bool(
        usr_args.get("timing_enabled", cfg.EVALUATION.get("timing_enabled", False))
    )
    vis_video = _parse_bool(
        usr_args.get("vis_video", cfg.EVALUATION.get("vis_video", False))
    )
    vis_output_dir = usr_args.get("eval_output_dir")
    attention_visualization = usr_args.get(
        "attention_visualization", cfg.EVALUATION.get("attention_visualization", {})
    )
    if isinstance(attention_visualization, DictConfig):
        attention_visualization = OmegaConf.to_container(
            attention_visualization, resolve=True
        )
    if not isinstance(attention_visualization, dict):
        raise ValueError("EVALUATION.attention_visualization must be a mapping")

    policy = WorldActionRobotWinPolicy(
        model_cfg=cfg.model,
        processor_cfg=cfg.data.train.processor,
        checkpoint_path=str(checkpoint_path),
        dataset_stats_path=dataset_stats_path,
        device=device,
        model_dtype=model_dtype,
        action_horizon=action_horizon,
        replan_steps=replan_steps,
        num_inference_steps=num_inference_steps,
        sigma_shift=sigma_shift,
        seed=seed,
        text_cfg_scale=text_cfg_scale,
        negative_prompt=negative_prompt,
        override_instruction=override_instruction,
        rand_device=rand_device,
        tiled=tiled,
        timing_enabled=timing_enabled,
        num_video_frames=(int(cfg.data.train.num_frames) - 1)
        // int(cfg.data.train.action_video_freq_ratio)
        + 1,
        vis_video=vis_video,
        vis_output_dir=vis_output_dir,
        attention_visualization=attention_visualization,
    )

    has_video_all = getattr(policy.model, "video_expert", None) is not None and getattr(
        policy.model.video_expert, "has_video_all_condition", False
    )
    if has_video_all:
        video_all_path = usr_args.get("video_all_path")
        if _is_none_like(video_all_path):
            video_all_path = cfg.EVALUATION.get("video_all_path")
        if not _is_none_like(video_all_path):
            video_all_size = list(cfg.data.train.get("video_all_size"))
            (
                video_all_num_frames,
                video_all_frame_percentage,
                video_all_max_num_frames,
            ) = resolve_video_sampling(cfg, usr_args)
            policy.video_all_path = str(video_all_path)
            policy.video_all_size = video_all_size
            policy.video_all_num_frames = (
                int(video_all_num_frames) if video_all_num_frames is not None else None
            )
            policy.video_all_frame_percentage = video_all_frame_percentage
            policy.video_all_max_num_frames = video_all_max_num_frames
            logger.info(
                "video_all per-seed mode enabled, path=%s, num_frames=%s, percentage=%s, max=%s",
                video_all_path,
                policy.video_all_num_frames,
                policy.video_all_frame_percentage,
                policy.video_all_max_num_frames,
            )
        else:
            logger.warning(
                "has_video_all_condition=True but no video_all_path provided"
            )

    return policy


def eval(TASK_ENV, model, observation: Optional[Dict[str, Any]]):
    obs = encode_obs(observation)
    model.step(TASK_ENV, obs)


def reset_model(model):
    model.reset()


def finalize_episode(model):
    return model.finish_attention_episode()
