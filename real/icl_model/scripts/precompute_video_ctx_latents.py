import logging
import os
import sys
from pathlib import Path
from typing import Any

import hydra
import torch
import torch.distributed as dist
import torch.nn.functional as torch_F
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from simpleicl.datasets.lerobot.base_lerobot_dataset import normalize_condition_paths
from simpleicl.datasets.lerobot.video_ctx_latent_cache import (
    CACHE_SCHEMA_VERSION,
    atomic_json_save,
    atomic_torch_save,
    build_payload,
    cache_entry_path,
    canonical_source_path,
    load_latent,
    load_manifest,
    manifest_path,
    preprocessing_settings,
    source_fingerprint,
    validate_manifest,
)
from simpleicl.models.wan22.helpers.loader import _load_registered_model, _resolve_configs
from simpleicl.utils.config_resolvers import register_default_resolvers
from simpleicl.utils.logging_config import get_logger, setup_logging
from simpleicl.utils.video_sampling import align_frame_count_to_vae


register_default_resolvers()
logger = get_logger(__name__)


def _init_distributed() -> tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 1, 0

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")
    return True, dist.get_rank(), dist.get_world_size(), local_rank


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        value = value.strip().lower()
        if value in {"1", "true", "yes", "y"}:
            return True
        if value in {"0", "false", "no", "n"}:
            return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def _condition_sources(dataset) -> list[str]:
    sources = set()
    for conditions in dataset.lerobot_dataset.conditions.values():
        for value in conditions.values():
            for source in normalize_condition_paths(value):
                sources.add(canonical_source_path(source))
    return sorted(sources)


def _load_condition_vae(cfg: DictConfig, *, device: str):
    model_cfg = cfg.model
    model_id = str(model_cfg.model_id)
    tokenizer_model_id = str(model_cfg.tokenizer_model_id)
    redirect_common_files = bool(model_cfg.get("redirect_common_files", True))
    _, _, vae_config, _ = _resolve_configs(
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        redirect_common_files=redirect_common_files,
    )
    vae_config.download_if_necessary()
    vae = _load_registered_model(
        vae_config.path,
        "wan_video_vae",
        torch_dtype=torch.bfloat16,
        device=device,
    ).eval()

    vae_cond_path = model_cfg.get("vae_cond_pretrain_path")
    if vae_cond_path:
        payload = torch.load(str(vae_cond_path), map_location="cpu", weights_only=True)
        if isinstance(payload, dict) and "vae_cond" in payload:
            state_dict = payload["vae_cond"]
        elif isinstance(payload, dict) and all(
            isinstance(value, torch.Tensor) for value in payload.values()
        ):
            state_dict = payload
        else:
            raise ValueError(
                f"Checkpoint {vae_cond_path} has no `vae_cond` weights."
            )
        missing, unexpected = vae.load_state_dict(state_dict, strict=False)
        logger.info(
            "Loaded condition VAE checkpoint %s (missing=%d, unexpected=%d).",
            vae_cond_path,
            len(missing),
            len(unexpected),
        )
    return vae, str(vae_config.path)


def _manifest_payload(
    *,
    preprocessing: dict[str, Any],
    cfg: DictConfig,
    vae_path: str | None,
    entry_count: int,
    complete: bool,
) -> dict[str, Any]:
    vae_cond_path = cfg.model.get("vae_cond_pretrain_path")
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "complete": complete,
        "entry_count": entry_count,
        "preprocessing": preprocessing,
        "vae": {
            "model_id": str(cfg.model.model_id),
            "redirect_common_files": bool(cfg.model.get("redirect_common_files", True)),
            "vae_path": vae_path,
            "vae_cond_pretrain_path": str(vae_cond_path) if vae_cond_path else None,
            "vae_cond_pretrain_fingerprint": (
                source_fingerprint(str(vae_cond_path)) if vae_cond_path else None
            ),
        },
    }


def _encode_source(dataset, vae, source: str, device: str) -> torch.Tensor:
    video_ctx = dataset._read_video_ctx(
        source,
        dataset.video_ctx_num_frames,
        dataset.video_ctx_frame_percentage,
        dataset.video_ctx_max_num_frames,
    )
    video_ctx = dataset.resize_transform_all(video_ctx)
    video_ctx = dataset.crop_transform_all(video_ctx)
    video_ctx = dataset.normalize_transform(video_ctx)

    # Pad short clips to the canonical VAE-aligned frame count so every cached
    # latent has the same temporal dimension. Without this, source videos shorter
    # than video_ctx_num_frames get aligned to a smaller valid length by
    # align_frame_count_to_vae, producing latents of varying T in the cache.
    if dataset.video_ctx_num_frames is not None:
        target_frames = align_frame_count_to_vae(dataset.video_ctx_num_frames)
        pad_frames = target_frames - int(video_ctx.shape[0])
        if pad_frames > 0:
            # video_ctx is [T, C, H, W]; pad temporal dim (dim 0) at the end.
            video_ctx = torch_F.pad(
                video_ctx, (0, 0, 0, 0, 0, 0, 0, pad_frames), mode="replicate"
            )
        elif pad_frames < 0:
            raise ValueError(
                f"video_ctx for {source} has {video_ctx.shape[0]} frames, "
                f"but the canonical aligned target is {target_frames}."
            )

    video_ctx = video_ctx.permute(1, 0, 2, 3).unsqueeze(0).to(
        device=device,
        dtype=torch.bfloat16,
        non_blocking=True,
    )
    with torch.no_grad():
        latent = vae.encode(video_ctx, device=torch.device(device), tiled=False)
    if isinstance(latent, list):
        if len(latent) != 1:
            raise ValueError(f"Expected one VAE latent for {source}, got {len(latent)}.")
        latent = latent[0].unsqueeze(0)
    if latent.ndim != 5 or latent.shape[0] != 1:
        raise ValueError(
            f"VAE latent for {source} must be [1,C,T,H,W], got {tuple(latent.shape)}."
        )
    return latent[0]


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    setup_logging(log_level=logging.INFO)
    is_distributed, rank, world_size, local_rank = _init_distributed()
    overwrite = _to_bool(cfg.get("overwrite", False))
    verify_only = _to_bool(cfg.get("verify_only", False))
    cache_dir_value = cfg.data.train.get("video_ctx_latent_cache_dir")
    if cache_dir_value is None or not str(cache_dir_value).strip():
        raise ValueError("data.train.video_ctx_latent_cache_dir must be set.")
    cache_dir = Path(str(cache_dir_value)).expanduser()
    vae_cond_path = cfg.model.get("vae_cond_pretrain_path")
    if vae_cond_path:
        source_fingerprint(str(vae_cond_path))

    dataset = instantiate(
        cfg.data.train,
        processor=None,
        video_ctx_latent_cache_mode="off",
    )
    if not dataset.video_ctx_size:
        raise ValueError("video_ctx_size must be configured for latent precomputation.")
    preprocessing = preprocessing_settings(
        video_ctx_size=dataset.video_ctx_size,
        video_ctx_num_frames=dataset.video_ctx_num_frames,
        video_ctx_frame_percentage=dataset.video_ctx_frame_percentage,
        video_ctx_max_num_frames=dataset.video_ctx_max_num_frames,
    )
    sources = _condition_sources(dataset)
    if not sources:
        raise ValueError("No valid condition videos found in dataset metadata.")
    local_sources = sources[rank::world_size]

    if verify_only:
        manifest = load_manifest(cache_dir)
        validate_manifest(manifest, expected_preprocessing=preprocessing)
        failures = 0
        for source in tqdm(
            local_sources,
            desc=f"Verifying video latents (rank {rank}/{world_size})",
            unit="video",
            disable=is_distributed and rank != 0,
        ):
            try:
                load_latent(
                    cache_dir,
                    source,
                    expected_preprocessing=preprocessing,
                )
            except Exception as error:
                failures += 1
                logger.error("Invalid cache entry for %s: %s", source, error)
        counts = torch.tensor([failures], device=device_for_collective(device=None, local_rank=local_rank), dtype=torch.long)
        if is_distributed:
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        if int(counts.item()) != 0:
            raise RuntimeError(f"Video latent cache verification failed for {int(counts.item())} entries.")
        if rank == 0:
            logger.info("Verified %d video_ctx latent cache entries in %s.", len(sources), cache_dir)
        return

    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    if rank == 0:
        atomic_json_save(
            _manifest_payload(
                preprocessing=preprocessing,
                cfg=cfg,
                vae_path=None,
                entry_count=len(sources),
                complete=False,
            ),
            manifest_path(cache_dir),
        )
    if is_distributed:
        dist.barrier()

    vae, vae_path = _load_condition_vae(cfg, device=device)
    encoded = 0
    skipped = 0
    for source in tqdm(
        local_sources,
        desc=f"Encoding video latents (rank {rank}/{world_size})",
        unit="video",
        disable=is_distributed and rank != 0,
    ):
        path = cache_entry_path(cache_dir, source)
        if path.exists() and not overwrite:
            try:
                load_latent(
                    cache_dir,
                    source,
                    expected_preprocessing=preprocessing,
                )
                skipped += 1
                continue
            except Exception:
                pass
        latent = _encode_source(dataset, vae, source, device)
        payload = build_payload(
            source_path=source,
            preprocessing=preprocessing,
            latent=latent,
        )
        atomic_torch_save(payload, path)
        encoded += 1

    reduce_device = device_for_collective(device=device, local_rank=local_rank)
    counts = torch.tensor([encoded, skipped], device=reduce_device, dtype=torch.long)
    if is_distributed:
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        dist.barrier()
    if rank == 0:
        atomic_json_save(
            _manifest_payload(
                preprocessing=preprocessing,
                cfg=cfg,
                vae_path=vae_path,
                entry_count=len(sources),
                complete=True,
            ),
            manifest_path(cache_dir),
        )
        logger.info(
            "Finished video_ctx latent precomputation in %s: encoded=%d skipped=%d total=%d.",
            cache_dir,
            int(counts[0].item()),
            int(counts[1].item()),
            len(sources),
        )


def device_for_collective(*, device: str | None, local_rank: int) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


if __name__ == "__main__":
    main()
