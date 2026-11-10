import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional
import numpy as np
import traceback
import av
import torch
import torch.nn.functional as torch_F
import torchvision.transforms.functional as transforms_F
from torch.utils.data import default_collate

from omegaconf import DictConfig, OmegaConf

from hydra.utils import instantiate
from .base_lerobot_dataset import (
    BaseLerobotDataset,
    normalize_condition_paths,
    select_condition_path_no_repeat,
)
from .utils.normalizer import (
    load_dataset_stats_from_json,
    merge_dataset_stats,
    save_dataset_stats_to_json,
)
from ..dataset_utils import ResizeSmallestSideAspectPreserving, CenterCrop, Normalize
from simpleicl.utils.logging_config import get_logger
from simpleicl.utils import misc
from simpleicl.utils.video_sampling import (
    align_frame_count_to_vae,
    resolve_video_frame_count,
)
from accelerate import PartialState

logger = get_logger(__name__)


DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"


def _video_stream_frame_count(stream) -> Optional[int]:
    total = stream.frames if stream.frames and stream.frames > 0 else None
    if total is None and stream.duration and stream.average_rate and stream.time_base:
        total = int(
            float(stream.duration * stream.time_base) * float(stream.average_rate)
        )
        if total <= 0:
            total = None
    return total


def robot_video_collate_fn(batch):
    """Collate robot samples while padding variable-length ``video_all`` clips."""
    if not batch or not any("video_all" in sample for sample in batch):
        return default_collate(batch)
    if not all("video_all" in sample for sample in batch):
        raise ValueError("All samples in a batch must contain video_all together.")

    collated = default_collate(
        [
            {key: value for key, value in sample.items() if key != "video_all"}
            for sample in batch
        ]
    )
    max_frames = max(int(sample["video_all"].shape[1]) for sample in batch)
    padded = []
    for sample in batch:
        video_all = sample["video_all"]
        pad_frames = max_frames - int(video_all.shape[1])
        if pad_frames:
            video_all = torch_F.pad(
                video_all, (0, 0, 0, 0, 0, pad_frames), mode="replicate"
            )
        padded.append(video_all)
    collated["video_all"] = torch.stack(padded, dim=0)
    return collated


class RobotVideoDataset(torch.utils.data.Dataset):
    collate_fn = staticmethod(robot_video_collate_fn)

    def __init__(
        self,
        dataset_dirs,
        shape_meta,
        num_frames=33,
        video_size=[384, 640],
        camera_key=None,
        processor=None,
        text_embedding_cache_dir=None,
        context_len=128,
        pretrained_norm_stats=None,
        val_set_proportion=0.05,
        is_training_set=False,
        global_sample_stride=1,
        action_video_freq_ratio: int = 1,
        skip_padding_as_possible: bool = False,
        max_padding_retry: int = 3,
        concat_multi_camera: str = "horizontal",  # "horizontal", "vertical", "robotwin", or None
        override_instruction: Optional[
            str
        ] = None,  # whether to hardcode a specific instruction for all samples, for debugging
        skip_episodes: Optional[dict] = None,
        replay_dir: Optional[str] = None,
        mask_cam: Optional[str] = None,
        video_all_size=None,
        video_all_num_frames: Optional[int] = 16,
        video_all_frame_percentage: Optional[float] = None,
        video_all_max_num_frames: Optional[int] = None,
        video_all_bucket_batching: bool = False,
    ):
        # Hydra/OmegaConf passes dict-like configs; convert to a plain dict.
        if skip_episodes is not None and not isinstance(skip_episodes, dict):
            skip_episodes = OmegaConf.to_container(skip_episodes, resolve=True)
        self.lerobot_dataset = BaseLerobotDataset(
            dataset_dirs=dataset_dirs,
            shape_meta=OmegaConf.to_container(shape_meta, resolve=True),
            obs_size=num_frames,
            action_size=num_frames - 1,
            val_set_proportion=val_set_proportion,
            is_training_set=is_training_set,
            global_sample_stride=global_sample_stride,
            skip_episodes=skip_episodes,
        )

        self.num_frames = num_frames
        self.action_video_freq_ratio = action_video_freq_ratio

        assert (num_frames - 1) % self.action_video_freq_ratio == 0, (
            f"num_frames-1 must be divisible by action_video_freq_ratio, got {num_frames - 1} and {self.action_video_freq_ratio}"
        )
        assert ((num_frames - 1) // self.action_video_freq_ratio) % 4 == 0, (
            f"video frames must be divisible by 4 for tokenization, got {(num_frames - 1) // self.action_video_freq_ratio}"
        )
        self.video_sample_indices = list(
            range(0, num_frames, self.action_video_freq_ratio)
        )

        self.camera_key = camera_key
        self.lerobot_dataset._set_return_images(True)

        self.video_size = video_size
        self.text_embedding_cache_dir = text_embedding_cache_dir
        self.context_len = context_len
        self.skip_padding_as_possible = skip_padding_as_possible
        self.max_padding_retry = max_padding_retry
        self.concat_multi_camera = concat_multi_camera
        self.override_instruction = override_instruction

        # Blank replay frames (optional).  Key: (dataset_name, episode_index).
        self.replay_dir = Path(replay_dir) if replay_dir is not None else None
        self._blank_mmap_cache: dict = {}  # populated lazily per worker
        self._task_episodes_cache: dict = {}  # (dataset_index, task_tuple) → [ep_indices]

        self.mask_cam = mask_cam

        self.resize_transform = ResizeSmallestSideAspectPreserving(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.crop_transform = CenterCrop(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )

        self.video_all_size = video_all_size
        self.video_all_num_frames = video_all_num_frames
        self.video_all_frame_percentage = video_all_frame_percentage
        self.video_all_max_num_frames = video_all_max_num_frames
        self.video_all_bucket_batching = bool(video_all_bucket_batching)
        self.batch_bucket_ids: Optional[torch.Tensor] = None
        self._bucket_condition_paths: dict[tuple[int, int], list[tuple[str, int]]] = {}
        self._condition_selection_state: dict = {}
        if self.video_all_frame_percentage is not None:
            resolve_video_frame_count(
                1,
                frame_percentage=self.video_all_frame_percentage,
                max_num_frames=self.video_all_max_num_frames,
            )
        if self.video_all_size:
            self.resize_transform_all = ResizeSmallestSideAspectPreserving(
                args={"img_w": video_all_size[1], "img_h": video_all_size[0]},
            )
            self.crop_transform_all = CenterCrop(
                args={"img_w": video_all_size[1], "img_h": video_all_size[0]},
            )
            if self.video_all_bucket_batching:
                self._build_video_all_bucket_index()

        self.normalize_transform = Normalize(
            args={"mean": 0.5, "std": 0.5},
        )

        if processor is not None:
            if isinstance(processor, DictConfig):
                processor = instantiate(processor)
            if not pretrained_norm_stats:
                if not is_training_set:
                    raise ValueError(
                        "pretrained_norm_stats must be provided for validation/test sets since we don't want to calculate stats on them."
                    )
                if PartialState().is_main_process:
                    logger.info("Calculating dataset stats for normalization...")
                    dataset_stats = self.lerobot_dataset.get_dataset_stats(processor)
                    work_dir = misc.get_work_dir()
                    if work_dir is not None:
                        save_dataset_stats_to_json(
                            dataset_stats, os.path.join(work_dir, "dataset_stats.json")
                        )
                else:
                    dataset_stats = None
                if (
                    torch.distributed.is_available()
                    and torch.distributed.is_initialized()
                ):
                    obj_list = [dataset_stats]
                    torch.distributed.broadcast_object_list(obj_list, src=0)
                    dataset_stats = obj_list[0]
            else:
                if OmegaConf.is_list(pretrained_norm_stats) or isinstance(
                    pretrained_norm_stats, (list, tuple)
                ):
                    stats_paths = list(pretrained_norm_stats)
                    dataset_stats = merge_dataset_stats(
                        [load_dataset_stats_from_json(path) for path in stats_paths]
                    )
                    logger.info(f"Using merged dataset stats: {stats_paths}")
                else:
                    dataset_stats = load_dataset_stats_from_json(pretrained_norm_stats)
                    logger.info(f"Using dataset stats: {pretrained_norm_stats}")
                if PartialState().is_main_process:
                    work_dir = misc.get_work_dir()
                    if work_dir is not None:
                        save_dataset_stats_to_json(
                            dataset_stats, os.path.join(work_dir, "dataset_stats.json")
                        )

            processor.set_normalizer_from_stats(dataset_stats)
            self.lerobot_dataset.set_processor(processor)

    def __len__(self):
        return len(self.lerobot_dataset)

    # ------------------------------------------------------------------ #
    # Pickle support: don't serialise mmap handles across process forks.  #
    # Each DataLoader worker reopens files lazily into its own cache.     #
    # ------------------------------------------------------------------ #
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_blank_mmap_cache"] = {}
        state["_task_episodes_cache"] = {}
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _get(self, idx):
        sample_idx = idx
        sample = None
        for attempt in range(self.max_padding_retry + 1):
            sample = self.lerobot_dataset[sample_idx]

            if not self.skip_padding_as_possible:
                break

            action_is_pad = sample["action_is_pad"]
            image_is_pad = sample["image_is_pad"]
            proprio_is_pad = sample["proprio_is_pad"]
            has_pad = False
            if bool(action_is_pad.any().item()):
                has_pad = True
            if bool(image_is_pad.any().item()):
                has_pad = True
            if bool(proprio_is_pad.any().item()):
                has_pad = True

            if not has_pad or attempt >= self.max_padding_retry:
                break

            sample_idx = np.random.randint(len(self.lerobot_dataset))

        image_is_pad = sample["image_is_pad"]

        video = sample["pixel_values"]  # [T, C, H, W] or [num_cameras, T, C, H, W]
        num_cameras = 1
        if video.ndim == 5:
            video = video[
                :, self.video_sample_indices, :, :, :
            ]  # [num_cameras, T_video, C, H, W]
            num_cameras, T_video, C, H, W = video.shape
        else:
            assert video.ndim == 4, (
                f"Expected video to have shape [T, C, H, W], but got {video.shape}"
            )
            video = video[self.video_sample_indices, :, :, :]  # [T_video, C, H, W]
            T_video, C, H, W = video.shape
        image_is_pad = image_is_pad[self.video_sample_indices]

        video = video.view(
            num_cameras, T_video, C, H, W
        )  # [num_cameras, T_video, C, H, W]

        if self.concat_multi_camera == "robotwin":
            if num_cameras != 3:
                raise ValueError(
                    f"`concat_multi_camera='robotwin'` requires exactly 3 cameras, got {num_cameras}"
                )
            cam_top = transforms_F.resize(
                video[0],
                size=[256, 320],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )  # [T_video, C, 256, 320]
            cam_left = transforms_F.resize(
                video[1],
                size=[128, 160],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )  # [T_video, C, 128, 160]
            cam_right = transforms_F.resize(
                video[2],
                size=[128, 160],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )  # [T_video, C, 128, 160]
            bottom = torch.cat([cam_left, cam_right], dim=-1)  # [T_video, C, 128, 320]
            video = torch.cat([cam_top, bottom], dim=-2)  # [T_video, C, 384, 320]

        elif num_cameras > 1:
            if self.concat_multi_camera == "horizontal":
                video = torch.cat(
                    [video[i] for i in range(num_cameras)], dim=-1
                )  # [T_video, C, H, num_cameras*W]
            elif self.concat_multi_camera == "vertical":
                video = torch.cat(
                    [video[i] for i in range(num_cameras)], dim=-2
                )  # [T_video, C, num_cameras*H, W]
            else:
                raise ValueError(
                    f"Invalid concat_multi_camera: {self.concat_multi_camera}. "
                    "Expected one of: horizontal, vertical, robotwin."
                )
        else:
            video = video.squeeze(0)  # [T_video, C, H, W]

        # Inject blank future replay frames before resizing, cropping, and normalization.
        if self.replay_dir is not None:
            video = self._inject_blank_frames(video, sample)

        if self.mask_cam is not None:
            assert self.concat_multi_camera != "robotwin" and num_cameras > 1
            _mask = torch.ones_like(video)
            if self.mask_cam == "wrist":
                # For horizontal concat, the wrist camera is the right half of the image
                if self.concat_multi_camera == "horizontal":
                    _mask[1:, :, :, W:] = 0.0
                # For vertical concat, the wrist camera is the bottom half of the image
                elif self.concat_multi_camera == "vertical":
                    _mask[1:, :, H:, :] = 0.0
            elif self.mask_cam == "agent":
                # For horizontal concat, the agent camera is the left half of the image
                if self.concat_multi_camera == "horizontal":
                    _mask[1:, :, :, :W] = 0.0
                # For vertical concat, the agent camera is the top half of the image
                elif self.concat_multi_camera == "vertical":
                    _mask[1:, :, :H, :] = 0.0
            else:
                raise ValueError(
                    f"Invalid mask_cam: {self.mask_cam}. Expected one of: wrist, agent."
                )
            video = video * _mask

        video = self.resize_transform(video)
        video = self.crop_transform(video)
        video = self.normalize_transform(video)  # [T_video, C, H, W]

        video = video.permute(1, 0, 2, 3)  # [C, T_video, H, W], range [-1, 1]

        # Proxy (from lerobot):
        #   action: [num_frames-1, action_dim] # start from t0, except the last frame
        #   proprio: [num_frames, proprio_dim] # start from t0 to the last frame, aligned with video frames
        action = sample["action"]  # [T-1, action_dim]
        proprio = sample["proprio"][:-1, :]  # [T-1, state_dim], to align with action
        if video.shape[1] <= 1:
            raise ValueError(
                f"`video` must have at least 2 frames, got shape {tuple(video.shape)}"
            )
        if action.shape[0] % (video.shape[1] - 1) != 0:
            raise ValueError(
                f"`action` horizon must be divisible by `video` transitions, got {action.shape[0]} and {video.shape[1] - 1}"
            )

        task = sample["instruction"]

        # FIXME
        if self.override_instruction is not None:
            task = self.override_instruction
        instruction = DEFAULT_PROMPT.format(task=task)

        context, context_mask = self._get_cached_text_context(instruction)
        # NOTE: to keep consistent with wan2.2's behavior
        context[~context_mask] = 0.0
        context_mask = torch.ones_like(context_mask)

        data = {
            "video": video,
            "action": action,
            "proprio": proprio,
            "prompt": instruction,
            "context": context,
            "context_mask": context_mask,
            "image_is_pad": image_is_pad,
            "action_is_pad": sample["action_is_pad"],
            "proprio_is_pad": sample["proprio_is_pad"],
        }
        if self.video_all_size:
            # Pick a random episode with the same task from the same dataset
            dataset_index = int(sample.get("dataset_index", 0))
            episode_index = int(sample["episode_index"])
            ds = self.lerobot_dataset.multi_dataset._datasets[dataset_index]
            ds_dir = self.lerobot_dataset.dataset_dirs[dataset_index]
            conditions = self.lerobot_dataset.conditions.get(ds_dir)
            if conditions is not None and episode_index in conditions:
                try:
                    if self.video_all_bucket_batching:
                        candidates = self._bucket_condition_paths[
                            (dataset_index, episode_index)
                        ]
                        sample_bucket = int(self.batch_bucket_ids[int(sample["idx"])])
                        matching = [
                            path
                            for path, bucket in candidates
                            if bucket == sample_bucket
                        ]
                        if not matching:
                            raise ValueError(
                                "No condition video matches the assigned video_all "
                                f"bucket {sample_bucket} for episode {episode_index}."
                            )
                        source = select_condition_path_no_repeat(
                            matching,
                            self._condition_selection_state,
                            (dataset_index, episode_index, "bucket", sample_bucket),
                        )
                    else:
                        source = select_condition_path_no_repeat(
                            conditions[episode_index],
                            self._condition_selection_state,
                            (dataset_index, episode_index),
                        )
                except ValueError as error:
                    raise ValueError(
                        f"No valid condition video for episode {episode_index}"
                    ) from error
            else:
                raise ValueError(
                    f"Missing paired condition video for episode {episode_index} in {ds_dir}"
                )
            assert os.path.exists(source), f"source not found: {source}"
            data["source"] = source

            # TODO: randomly add static video_all
            video_all = self._read_video_all(
                source,
                self.video_all_num_frames,
                self.video_all_frame_percentage,
                self.video_all_max_num_frames,
            )  # [num_frames, C, H, W], [0,1]
            video_all = self.resize_transform_all(video_all)
            video_all = self.crop_transform_all(video_all)
            video_all = self.normalize_transform(video_all)  # [-1, 1]
            video_all = video_all.permute(1, 0, 2, 3)  # [C, T, H, W]
            data["video_all"] = video_all
        return data

    # ------------------------------------------------------------------ #
    # Blank-replay frame injection                                        #
    # ------------------------------------------------------------------ #

    def _load_blank_frames(self, dataset_name: str, episode_index: int):
        """Return a memory-mapped numpy array for one episode, or None.

        Shape: [T_ep, H, W, 3] uint8  (two cameras concatenated horizontally).
        The array is kept open in ``_blank_mmap_cache`` for fast repeated access.
        """
        cache_key = (dataset_name, episode_index)
        if cache_key not in self._blank_mmap_cache:
            path = self.replay_dir / dataset_name / f"episode_{episode_index:06d}.npy"
            if not path.exists():
                self._blank_mmap_cache[cache_key] = None
                return None
            self._blank_mmap_cache[cache_key] = np.load(str(path), mmap_mode="r")
        return self._blank_mmap_cache[cache_key]

    def _inject_blank_frames(
        self,
        video: torch.Tensor,
        sample: dict,
    ) -> torch.Tensor:
        """Replace video frames at positions 1..T_video-1 with blank-replay frames.

        ``video`` is shaped [T_video, C, H, W] (float, [0, 1]) at this point in
        ``_get`` – i.e. after camera concatenation but before resize / crop /
        normalise.  The first frame (real observation) is left untouched.

        Returns the same tensor with future frames replaced in-place.
        """
        episode_index = sample.get("episode_index")
        frame_index = sample.get("frame_index")
        dataset_index = sample.get("dataset_index", 0)

        if episode_index is None or frame_index is None:
            return video

        episode_index = int(episode_index)
        frame_index = int(frame_index)
        dataset_index = int(dataset_index)

        dataset_dirs = self.lerobot_dataset.dataset_dirs
        ds_dir_idx = min(dataset_index, len(dataset_dirs) - 1)
        dataset_name = Path(dataset_dirs[ds_dir_idx]).name

        arr = self._load_blank_frames(dataset_name, episode_index)
        if arr is None:
            return video

        T_ep = len(arr)
        gss = self.lerobot_dataset.global_sample_stride

        ep_frame_indices = [
            min(frame_index + self.video_sample_indices[i] * gss, T_ep - 1)
            for i in range(1, len(self.video_sample_indices))
        ]

        # Load: [N, H_arr, W_arr, 3] uint8 → [N, C, H_arr, W_arr] float [0,1]
        blank_np = arr[ep_frame_indices].copy()  # copy needed for fancy mmap index
        blank = (
            torch.from_numpy(blank_np)  # [N, H, W, 3]
            .permute(0, 3, 1, 2)  # [N, 3, H, W]
            .to(dtype=video.dtype)
            / 255.0
        )

        # Resize to match video spatial dimensions if they differ
        _, C, H, W = video.shape
        if blank.shape[-2:] != (H, W):
            blank = torch_F.interpolate(
                blank, size=(H, W), mode="bilinear", align_corners=False
            )

        video[1:] = blank
        return video

    def _aligned_video_all_frame_count(self, total_frames: int) -> int:
        target_frames = resolve_video_frame_count(
            total_frames,
            frame_percentage=self.video_all_frame_percentage,
            num_frames=self.video_all_num_frames,
            max_num_frames=self.video_all_max_num_frames,
        )
        if target_frames is None:
            target_frames = 16
        available_frames = int(total_frames)
        if self.video_all_max_num_frames is not None:
            available_frames = min(
                available_frames,
                int(self.video_all_max_num_frames),
            )
        return align_frame_count_to_vae(
            target_frames,
            max_num_frames=available_frames,
        )

    def _condition_frame_bucket(self, video_path: str) -> int:
        with av.open(video_path) as container:
            stream = container.streams.video[0]
            total_frames = _video_stream_frame_count(stream)
        if total_frames is None:
            raise ValueError(
                "video_all bucket batching requires MP4 frame-count metadata: "
                f"{video_path}"
            )
        return self._aligned_video_all_frame_count(int(total_frames))

    def _build_video_all_bucket_index(self) -> None:
        condition_paths = []
        for conditions in self.lerobot_dataset.conditions.values():
            for value in conditions.values():
                condition_paths.extend(normalize_condition_paths(value))
        condition_paths = list(dict.fromkeys(condition_paths))
        if not condition_paths:
            raise ValueError(
                "video_all_bucket_batching=true requires condition paths in "
                "meta/conditions.json."
            )

        state = PartialState()
        path_buckets = None
        if state.is_main_process:
            logger.info(
                "Indexing %d video_all condition clips for length-bucket batching...",
                len(condition_paths),
            )
            worker_count = min(16, max(1, os.cpu_count() or 1))
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                buckets = executor.map(self._condition_frame_bucket, condition_paths)
                path_buckets = dict(zip(condition_paths, buckets, strict=True))

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            payload = [path_buckets]
            torch.distributed.broadcast_object_list(payload, src=0)
            path_buckets = payload[0]
        if path_buckets is None:
            raise RuntimeError(
                "Failed to share video_all frame buckets across processes."
            )

        bucket_ids = torch.empty(len(self), dtype=torch.int16)
        global_offset = 0
        for dataset_index, dataset in enumerate(
            self.lerobot_dataset.multi_dataset._datasets
        ):
            ds_dir = self.lerobot_dataset.dataset_dirs[dataset_index]
            conditions = self.lerobot_dataset.conditions.get(ds_dir, {})
            episode_ids = (
                dataset.episodes
                if dataset.episodes is not None
                else list(range(len(dataset.episode_data_index["from"])))
            )
            for local_episode_index, episode_index in enumerate(episode_ids):
                episode_index = int(episode_index)
                paths = normalize_condition_paths(conditions.get(episode_index))
                if not paths:
                    raise ValueError(
                        "Missing video_all condition paths for "
                        f"dataset={ds_dir}, episode={episode_index}."
                    )
                self._bucket_condition_paths[(dataset_index, episode_index)] = [
                    (path, int(path_buckets[path])) for path in paths
                ]

                start = global_offset + int(
                    dataset.episode_data_index["from"][local_episode_index]
                )
                end = global_offset + int(
                    dataset.episode_data_index["to"][local_episode_index]
                )
                candidate_buckets = torch.tensor(
                    [path_buckets[path] for path in paths],
                    dtype=torch.int16,
                )
                # Cycle variants across transitions so every condition remains represented.
                choices = torch.arange(start, end) % len(paths)
                bucket_ids[start:end] = candidate_buckets[choices]
            global_offset += int(dataset.num_frames)

        if global_offset != len(self):
            raise ValueError(
                "video_all bucket index length mismatch: "
                f"indexed={global_offset}, dataset={len(self)}."
            )
        self.batch_bucket_ids = bucket_ids
        bucket_values, bucket_counts = torch.unique(bucket_ids, return_counts=True)
        logger.info(
            "video_all length buckets ready: %s",
            ", ".join(
                f"{int(value)}:{int(count)}"
                for value, count in zip(bucket_values, bucket_counts, strict=True)
            ),
        )

    def _read_video_all(
        self,
        video_path: str,
        num_frames: Optional[int] = 16,
        frame_percentage: Optional[float] = None,
        max_num_frames: Optional[int] = None,
    ) -> torch.Tensor:
        """Read an entire video and uniformly sample a fixed count or percentage.

        Percentage sampling uses the stream's actual frame count and is capped
        by ``max_num_frames`` when configured. Returns ``[T, C, H, W]`` float32
        in ``[0, 1]``.
        Only calls ``to_ndarray`` on the sampled frames to minimize overhead.
        """
        with av.open(video_path) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"

            total = _video_stream_frame_count(stream)

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
                needed = set(
                    np.linspace(0, total - 1, target_frames, dtype=int).tolist()
                )
            else:
                needed = None

            frames = []
            for i, frame in enumerate(container.decode(video=0)):
                if needed is None or i in needed:
                    frames.append(
                        torch.from_numpy(frame.to_ndarray(format="rgb24")).permute(
                            2, 0, 1
                        )
                    )
                if needed is not None and len(frames) >= target_frames:
                    break

        if not frames:
            raise ValueError(f"Video contains no decodable frames: {video_path}")
        video = torch.stack(frames, dim=0).float() / 255.0

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

        return video

    def _get_cached_text_context(self, prompt: str):
        if self.text_embedding_cache_dir is None:
            raise ValueError("text_embedding_cache_dir is not set.")
        cache_dir = self.text_embedding_cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_path = os.path.join(
            cache_dir, f"{hashed}.t5_len{self.context_len}.wan22ti2v5b.pt"
        )
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Missing text embedding cache: {cache_path}. "
                "Run scripts/precompute_text_embeds.py first."
            )
        payload = torch.load(cache_path, map_location="cpu")
        context = payload["context"]
        context_mask = payload["mask"].bool()
        if context.ndim != 2:
            raise ValueError(
                f"Cached `context` must be 2D [L, D], got shape {tuple(context.shape)} in {cache_path}"
            )
        if context_mask.ndim != 1:
            raise ValueError(
                f"Cached `mask` must be 1D [L], got shape {tuple(context_mask.shape)} in {cache_path}"
            )
        if context.shape[0] != self.context_len:
            raise ValueError(
                f"Cached context_len mismatch: expected {self.context_len}, got {context.shape[0]} in {cache_path}"
            )
        if context_mask.shape[0] != self.context_len:
            raise ValueError(
                f"Cached mask_len mismatch: expected {self.context_len}, got {context_mask.shape[0]} in {cache_path}"
            )

        return context, context_mask

    def __getitem__(self, idx):
        try:
            data = self._get(idx)
        except Exception as e:
            print(
                f"Error processing sample idx {idx}: {e}. Returning a random sample instead."
            )
            # trace back
            print(traceback.format_exc())
            random_idx = np.random.randint(len(self))
            data = self._get(random_idx)
        return data
