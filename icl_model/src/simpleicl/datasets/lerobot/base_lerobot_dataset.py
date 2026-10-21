import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Literal, Dict, Optional, Any, DefaultDict
from tqdm import tqdm
import os

from .lerobot.lerobot_dataset import LeRobotDatasetMetadata, MultiLeRobotDataset

from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback
from simpleicl.utils.logging_config import get_logger
from .processors.base_processor import BaseProcessor

logger = get_logger(__name__)

MAX_GETITEM_ATTEMPT = 5


def normalize_condition_paths(value: object) -> list[str]:
    """Normalize legacy condition strings and multipertur path lists."""
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str) and item]
    return []


def select_condition_path(value: object, rng: np.random.Generator | None = None) -> str:
    """Select one condition path, preserving random choice for multipertur data."""
    paths = normalize_condition_paths(value)
    if not paths:
        raise ValueError("No condition video path is configured")
    if rng is None:
        index = int(np.random.randint(len(paths)))
    else:
        index = int(rng.integers(len(paths)))
    return paths[index]


class BaseLerobotDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dirs: List[str],

        # shapes
        shape_meta: Dict[str, Any],
        action_size: int = 1,
        past_action_size: int = 0, # Excludes the current frame
        obs_size: int = 1,
        past_obs_size: int = 0,

        # train vs val
        val_set_proportion: float = 0.05,
        is_training_set: bool = False,
        seed: int = 42,

        # sampling
        global_sample_stride: int = 1,

        # per-dataset episode exclusion. Maps `dataset_dir` (same string as in
        # `dataset_dirs`) to a list of episode indices to drop before train/val
        # split. Indices not present in a given dataset are silently ignored.
        skip_episodes: Optional[Dict[str, List[int]]] = None,
        cond_flags: Optional[dict] = None,
    ):
        assert len(dataset_dirs) > 0, "At least one dataset directory is required"
        assert past_action_size == 0
        assert past_obs_size == 0
        assert action_size == obs_size - 1, "In this dataset, action_size should be obs_size - 1"
        
        self.dataset_dirs = dataset_dirs
        self.shape_meta = shape_meta
        self.action_size = action_size
        self.past_action_size = past_action_size
        self.obs_size = obs_size
        self.processor = None  # Will be set externally
        metas = []
        for ds_dir in dataset_dirs:
            ds_root = Path(ds_dir)
            repo_id = ds_dir
            meta = LeRobotDatasetMetadata(repo_id=repo_id, root=ds_root)
            metas.append(meta)

        fps_list = [m.fps for m in metas]
        assert len(set(fps_list)) == 1, f"All dataset_dirs must have the same fps, got {fps_list}"
        fps = fps_list[0]
        if fps != 30:
            logger.info("Dataset fps values: %s", fps_list)
        
        self.global_sample_stride = global_sample_stride

        self.val_set_proportion = val_set_proportion
        self.is_training_set = is_training_set

        self.image_meta = shape_meta["images"]
        self.state_meta = shape_meta["state"]
        self.action_meta = shape_meta["action"]

        # Assign lerobot_key once; it is independent of the sampling stride.
        for meta in self.image_meta:
            key = meta["key"]
            meta["lerobot_key"] = f"observation.images.{key}" if key != "default" else "observation.images"
        for meta in self.state_meta:
            key = meta["key"]
            meta["lerobot_key"] = f"observation.state.{key}" if key != "default" else "observation.state"
        for meta in self.action_meta:
            key = meta["key"]
            meta["lerobot_key"] = f"action.{key}" if key != "default" else "action"

        def _build_delta_timestamps(stride: int) -> Dict[str, List[float]]:
            dt: Dict[str, List[float]] = {}
            for meta in self.image_meta:
                dt[meta["lerobot_key"]] = [
                    (t * stride) / fps for t in range(-past_obs_size, -past_obs_size + obs_size)
                ]
            for meta in self.state_meta:
                dt[meta["lerobot_key"]] = [
                    (t * stride) / fps for t in range(-past_obs_size, -past_obs_size + obs_size)
                ]
            for meta in self.action_meta:
                dt[meta["lerobot_key"]] = [
                    (t * stride) / fps for t in range(-past_action_size, -past_action_size + action_size)
                ]
            return dt

        delta_timestamps = _build_delta_timestamps(global_sample_stride)

        episodes = {}
        skip_episodes = skip_episodes or {}

        # Auto-populate skip_episodes and per-episode conditions from
        # meta/conditions.json when present.  Keys are episode indices (as
        # strings), values are either null (→ skip) or a source path string.
        self.conditions: Dict[str, Dict[int, Any]] = {}
        list_cond = bool(cond_flags.get("list_cond", False)) if cond_flags else False

        def _is_valid_source(v: Optional[str]) -> bool:
            return v is not None and os.path.exists(v)

        def _remap_condition_path(v: Any, ds_dir: str) -> Any:
            # Condition paths stored in conditions.json carry the (possibly stale)
            # absolute prefix of the machine that generated them, e.g.
            #   /path/to/lerobot_collect/outputs/processed/0726/human/videos/...
            # while the actual data lives alongside ds_dir, e.g.
            #   <ds_dir>=.../0726_cut/realworld  ->  .../0726_cut/human/videos/...
            # Re-root each path onto the parent of ds_dir, keeping the trailing
            # portion from the sibling-dataset dir (the component right before the
            # lerobot "videos"/"video" folder) onward.  Idempotent for paths that
            # already point to the correct location.
            if not isinstance(v, str):
                return v
            parts = Path(v).parts
            anchor = next(
                (i for i, p in enumerate(parts) if p in ("videos", "video")), None
            )
            if anchor is None or anchor == 0:
                logger.warning(f"[conditions.json] {ds_dir}: invalid condition path: {v}")
                return v
            target_root = os.path.dirname(os.path.normpath(ds_dir))
            return os.path.join(target_root, *parts[anchor - 1:])

        for ds_dir in dataset_dirs:
            # When list_cond is enabled, prefer meta/conditions_list.json (values
            # are lists of candidate source paths); fall back to conditions.json.
            conditions_path = Path(ds_dir) / "meta" / "conditions.json"
            if list_cond:
                list_path = Path(ds_dir) / "meta" / "conditions_list.json"
                if list_path.exists():
                    conditions_path = list_path
                    logger.info("[conditions.json] %s: list_cond is enabled, using conditions_list.json", ds_dir)
            if not conditions_path.exists():
                continue
            with open(conditions_path) as f:
                raw: dict = json.load(f)
            null_eps: list[int] = []
            cond_map: Dict[int, Any] = {}
            for k, v in raw.items():
                ep_idx = int(k)
                if isinstance(v, (list, tuple)):
                    v = [_remap_condition_path(s, ds_dir) for s in v]
                    valid = [s for s in v if _is_valid_source(s)]
                    if valid:
                        cond_map[ep_idx] = valid
                    else:
                        null_eps.append(ep_idx)
                else:
                    v = _remap_condition_path(v, ds_dir)
                    if _is_valid_source(v):
                        cond_map[ep_idx] = v
                    else:
                        null_eps.append(ep_idx)
            if null_eps:
                prev = skip_episodes.get(ds_dir, [])
                existing = set(prev)
                skip_episodes[ds_dir] = prev + [e for e in null_eps if e not in existing]
                logger.info(
                    f"[conditions.json] {ds_dir}: auto-skipping {len(null_eps)} null episodes."
                )
            self.conditions[ds_dir] = cond_map

        # Normalize skip lists to sets for O(1) membership tests.
        skip_sets: Dict[str, set] = {k: set(int(i) for i in v) for k, v in skip_episodes.items()}
        if val_set_proportion < 1e-6:
            for meta in metas:
                all_eps = list(range(meta.total_episodes))
                skip = skip_sets.get(meta.repo_id, set())
                if skip:
                    kept = [i for i in all_eps if i not in skip]
                    logger.info(
                        f"[skip_episodes] {meta.repo_id}: dropping {len(all_eps) - len(kept)} / "
                        f"{len(all_eps)} episodes (e.g. {sorted(skip)[:5]}...)."
                    )
                else:
                    kept = all_eps
                episodes.update({meta.repo_id: kept})
        else:
            for meta in metas:
                # Build the candidate pool *after* removing skipped episodes so that
                # the val split is taken from the remaining episodes only.
                all_eps = list(range(meta.total_episodes))
                skip = skip_sets.get(meta.repo_id, set())
                pool = [i for i in all_eps if i not in skip]
                if skip:
                    logger.info(
                        f"[skip_episodes] {meta.repo_id}: dropping {len(all_eps) - len(pool)} / "
                        f"{len(all_eps)} episodes before train/val split."
                    )
                split_idx = int(len(pool) * (1 - val_set_proportion))
                # random shuffle episode indices before splitting
                rng = np.random.default_rng(seed)
                rng.shuffle(pool)
                if self.is_training_set:
                    episodes.update({meta.repo_id: [pool[i] for i in range(split_idx)]})
                else:
                    episodes.update({meta.repo_id: [pool[i] for i in range(split_idx, len(pool))]})

        self.multi_dataset = MultiLeRobotDataset(
            dataset_dirs=self.dataset_dirs,
            episodes=episodes,
            delta_timestamps=delta_timestamps,
            # torchcodec (the auto-selected default) fails on this env with
            # "Could not push packet to decoder: Function not implemented" and
            # silently falls back to pyav for every frame.
        )

        # LeRobot's multi-dataset does not expose a merged episode_data_index;
        # build one manually by chaining the per-dataset indices.
        episode_data_index = []
        end_index = 0
        for dataset in self.multi_dataset._datasets:
            multi_episode_data_index = {
                "from": dataset.episode_data_index["from"] + end_index,
                "to": dataset.episode_data_index["to"] + end_index,
            }
            episode_data_index.append(multi_episode_data_index)
            end_index = multi_episode_data_index["to"][-1]

        self.episode_data_index = {
            "from": torch.cat([dataset["from"] for dataset in episode_data_index]),
            "to": torch.cat([dataset["to"] for dataset in episode_data_index]),
        }

    def _get_action(self, meta, lerobot_sample) -> torch.Tensor:
        key, lerobot_key, raw_shape = meta["key"], meta["lerobot_key"], meta["raw_shape"]
        action: torch.Tensor = lerobot_sample[lerobot_key] # [T, action_dim]
        if action.ndim == 1: # for shape of 1, like gripper
            action = action.unsqueeze(-1)
        assert action.shape[-1] == raw_shape, f"Action '{key}' shape {action.shape[-1]} mismatch with meta {raw_shape}."
        return action

    def _get_state(self, meta, lerobot_sample) -> torch.Tensor:
        key, lerobot_key, raw_shape = meta["key"], meta["lerobot_key"], meta["raw_shape"]
        state: torch.Tensor = lerobot_sample[lerobot_key]
        if state.ndim == 1: # for shape of 1, like gripper
            state = state.unsqueeze(-1)
        assert state.shape[-1] == raw_shape, f"State '{key}' shape {state.shape[-1]} mismatch with meta {raw_shape}."
        return state
    
    def _get_image(self, meta, lerobot_sample) -> torch.Tensor:
        key, lerobot_key, raw_shape = meta["key"], meta["lerobot_key"], meta["raw_shape"]
        image: torch.Tensor = lerobot_sample[lerobot_key]
        if image.ndim == 3: # time dim will lost when obs_size is 1
            image = image.unsqueeze(0)        
        image = (image * 255).to(torch.uint8) # (1, 3, H, W)
        return image
    
    def _split_lerobot_sample(self, lerobot_sample) -> Dict[str, Any]:
        return lerobot_sample
    
    def _get_episode_data(self, episode_idx):
        lerobot_sample = self.multi_dataset.get_episode_data(episode_idx)
        lerobot_sample = self._split_lerobot_sample(lerobot_sample)
        state, action = {}, {}
        for meta in self.state_meta:
            s = self._get_state(meta, lerobot_sample)
            state[meta["key"]] = s.unsqueeze(1).float()
        for meta in self.action_meta:
            a = self._get_action(meta, lerobot_sample)
            a = sliding_window_with_replication(a, self.action_size)
            action[meta["key"]] = a.float()
        return {"action": action, "state": state}

    def _set_return_images(self, flag: bool):
        self.return_images = flag
        self.multi_dataset.set_during_training(flag)

    def __len__(self):
        return self.multi_dataset.num_frames

    def _get_additional_data(self, sample, lerobot_sample):
        return sample

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds {len(self)}.")

        # Retry with random indices until we successfully load a frame.
        sample_idx = idx
        attempt = 0
        last_exception: Optional[Exception] = None
        while attempt < MAX_GETITEM_ATTEMPT:
            try:
                lerobot_sample = self.multi_dataset[sample_idx]
                lerobot_sample = self._split_lerobot_sample(lerobot_sample)
                break
            except Exception as err:
                attempt += 1
                last_exception = err
                logger.warning(
                    f"Error loading sample {sample_idx} (attempt {attempt}). "
                    "Retrying with a random index. "
                    f"Error: {err}"
                )
                sample_idx = np.random.randint(len(self))
                logger.warning(traceback.format_exc())
        else:
            raise RuntimeError(
                f"Failed to load a valid sample after {MAX_GETITEM_ATTEMPT} attempts "
                f"for index {idx}."
            ) from last_exception

        # Get data from lerobot, organized in nested dict
        sample = {
            "idx": sample_idx,
            "task": lerobot_sample["task"],
            "action": {},
            "state": {},
            "images": {},
        }
        for meta in self.state_meta:
            sample["state"][meta["key"]] = self._get_state(meta, lerobot_sample)

        for meta in self.action_meta:
            sample["action"][meta["key"]] = self._get_action(meta, lerobot_sample)

        for meta in self.image_meta:
            sample["images"][meta["key"]] = self._get_image(meta, lerobot_sample)

        sample["action_is_pad"] = lerobot_sample[f"{self.action_meta[0]['lerobot_key']}_is_pad"]
        sample["state_is_pad"] = lerobot_sample[f"{self.state_meta[0]['lerobot_key']}_is_pad"]
        sample["image_is_pad"] = lerobot_sample[f"{self.image_meta[0]['lerobot_key']}_is_pad"]

        sample = self._get_additional_data(sample, lerobot_sample)

        for key in lerobot_sample:
            if key not in sample and "observation" not in key and "action" not in key:
                sample[key] = lerobot_sample[key]

        if self.processor is not None:
            sample = self.processor.preprocess(sample)

        return sample

    def set_processor(self, processor: BaseProcessor):
        """Set processor instance from external initialization."""
        self.processor = processor
        if self.is_training_set:
            self.processor.train()
        else:
            self.processor.eval()
        return self

    def get_dataset_stats(self, preprocessor: BaseProcessor):
        state_min = DefaultDict(list)
        state_max = DefaultDict(list)
        state_mean = DefaultDict(list)
        state_var = DefaultDict(list)
        state_q01 = DefaultDict(list)
        state_q99 = DefaultDict(list)

        action_min = DefaultDict(list)
        action_max = DefaultDict(list)
        action_mean = DefaultDict(list)
        action_var = DefaultDict(list)
        action_q01 = DefaultDict(list)
        action_q99 = DefaultDict(list)

        episodes_num = self.multi_dataset.num_episodes
        
        def process_episode(episode_idx):
            batch = self._get_episode_data(episode_idx) 
            batch = preprocessor.action_state_transform(batch)
            return batch
        
        multi_thread = True
        if not multi_thread:
            for episode_idx in tqdm(range(episodes_num), desc="Iterating dataset to get normalization"):
                batch = process_episode(episode_idx)
                for meta in self.state_meta:
                    key = meta["key"]
                    cur_state: torch.Tensor = batch["state"][key] # (B, T, dim)
                    state_min[key].append(cur_state.amin(0))
                    state_max[key].append(cur_state.amax(0))
                    state_mean[key].append(cur_state.mean(0))
                    state_var[key].append(cur_state.var(0))
                    state_q01[key].append(torch.quantile(cur_state, 0.01, dim=0, keepdim=False))
                    state_q99[key].append(torch.quantile(cur_state, 0.99, dim=0, keepdim=False))
                for meta in self.action_meta:
                    key = meta["key"]
                    cur_action: torch.Tensor = batch["action"][key] # (B, T, dim)
                    action_min[key].append(cur_action.amin(0))
                    action_max[key].append(cur_action.amax(0))
                    action_mean[key].append(cur_action.mean(0))
                    action_var[key].append(cur_action.var(0))
                    action_q01[key].append(torch.quantile(cur_action, 0.01, dim=0, keepdim=False))
                    action_q99[key].append(torch.quantile(cur_action, 0.99, dim=0, keepdim=False))
        
        else:
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(process_episode, num) for num in range(episodes_num)]
                
                for future in tqdm(as_completed(futures), total=episodes_num, desc="Iterating dataset to get normalization"):
                    try:
                        batch = future.result()
                        for meta in self.state_meta:
                            key = meta["key"]
                            cur_state: torch.Tensor = batch["state"][key] # (B, T, dim)
                            state_min[key].append(cur_state.amin(0))
                            state_max[key].append(cur_state.amax(0))
                            state_mean[key].append(cur_state.mean(0))
                            state_var[key].append(cur_state.var(0))
                            state_q01[key].append(torch.quantile(cur_state, 0.01, dim=0, keepdim=False))
                            state_q99[key].append(torch.quantile(cur_state, 0.99, dim=0, keepdim=False))

                        for meta in self.action_meta:
                            key = meta["key"]
                            cur_action: torch.Tensor = batch["action"][key] # (B, T, dim)
                            action_min[key].append(cur_action.amin(0))
                            action_max[key].append(cur_action.amax(0))
                            action_mean[key].append(cur_action.mean(0))
                            action_var[key].append(cur_action.var(0))
                            action_q01[key].append(torch.quantile(cur_action, 0.01, dim=0, keepdim=False))
                            action_q99[key].append(torch.quantile(cur_action, 0.99, dim=0, keepdim=False))

                    except Exception as e:
                        logger.error(f"Error processing episode: {e}")
                        logger.error(traceback.format_exc())
                        raise

        # assume that each minibatch has equal number of samples
        def get_mean_std(means, vars):
            means = torch.stack(means)
            vars = torch.stack(vars)
            stepwise_mean = means.mean(0)
            stepwise_std = (vars + (means - stepwise_mean) ** 2).mean(0).sqrt()
            global_mean = means.mean((0, 1))
            global_std = (vars + (means - global_mean) ** 2).mean((0, 1)).sqrt()
            return stepwise_mean, stepwise_std, global_mean, global_std

        stats = {"state": DefaultDict(dict), "action": DefaultDict(dict), "num_episodes": episodes_num, "num_transition": self.multi_dataset.num_frames}
        for meta in self.state_meta:
            key = meta["key"]
            stats["state"][key]["stepwise_min"] = torch.stack(state_min[key]).amin(0)
            stats["state"][key]["stepwise_max"] = torch.stack(state_max[key]).amax(0)
            stats["state"][key]["global_min"] = stats["state"][key]["stepwise_min"].amin(0)
            stats["state"][key]["global_max"] = stats["state"][key]["stepwise_max"].amax(0)
            stats["state"][key]["stepwise_q01"] = torch.stack(state_q01[key]).amin(0)
            stats["state"][key]["stepwise_q99"] = torch.stack(state_q99[key]).amax(0)
            stats["state"][key]["global_q01"] = stats["state"][key]["stepwise_q01"].amin(0)
            stats["state"][key]["global_q99"] = stats["state"][key]["stepwise_q99"].amax(0)
            (
                stats["state"][key]["stepwise_mean"],
                stats["state"][key]["stepwise_std"],
                stats["state"][key]["global_mean"],
                stats["state"][key]["global_std"],
            ) = get_mean_std(state_mean[key], state_var[key])

        for meta in self.action_meta:
            key = meta["key"]
            stats["action"][key]["stepwise_min"] = torch.stack(action_min[key]).amin(0)
            stats["action"][key]["stepwise_max"] = torch.stack(action_max[key]).amax(0)
            stats["action"][key]["global_min"] = stats["action"][key]["stepwise_min"].amin(0)
            stats["action"][key]["global_max"] = stats["action"][key]["stepwise_max"].amax(0)
            stats["action"][key]["stepwise_q01"] = torch.stack(action_q01[key]).amin(0)
            stats["action"][key]["stepwise_q99"] = torch.stack(action_q99[key]).amax(0)
            stats["action"][key]["global_q01"] = stats["action"][key]["stepwise_q01"].amin(0)
            stats["action"][key]["global_q99"] = stats["action"][key]["stepwise_q99"].amax(0)
            (
                stats["action"][key]["stepwise_mean"], 
                stats["action"][key]["stepwise_std"], 
                stats["action"][key]["global_mean"], 
                stats["action"][key]["global_std"],
            ) = get_mean_std(action_mean[key], action_var[key])

        return stats


def sliding_window_with_replication(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Construct a sliding-window tensor from the input tensor x (shape: [N, D]).
    The output shape is [N, window_size, D].
    
    For each starting index i:
        out[i, j, :] =
            x[i + j, :]      if i + j < N
            x[-1, :]         otherwise (replicate the last row when out of bounds)
    
    Args:
        x (torch.Tensor): Input tensor of shape [N, D]
        window_size (int): Size of the sliding window
    
    Returns:
        torch.Tensor: Tensor of shape [N, window_size, D]
    """
    assert x.dim() == 2
    assert window_size > 0
    
    N, D = x.shape
    
    # shape [N, window_size]
    # indices[i, j] = i + j
    i_indices = torch.arange(N).unsqueeze(1)            # [N, 1]
    j_indices = torch.arange(window_size).unsqueeze(0)  # [1, window_size]
    indices = i_indices + j_indices                     # [N, window_size]

    # Clamp so out-of-range windows repeat the first/last sample:
    # out[i, j, :] = x[clamped_indices[i, j], :]
    clamped_indices = torch.clamp(indices, min=0, max=N - 1)  # [N, window_size]

    out = x[clamped_indices]  # [N, window_size, D]

    return out
