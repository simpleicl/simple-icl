import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any

import torch


CACHE_SCHEMA_VERSION = 1
MANIFEST_FILENAME = "manifest.json"


class VideoCtxLatentCacheError(RuntimeError):
    pass


def canonical_source_path(source_path: str) -> str:
    return str(Path(source_path).expanduser().resolve())


def source_fingerprint(source_path: str) -> dict[str, Any]:
    canonical_path = canonical_source_path(source_path)
    stat = os.stat(canonical_path)
    return {
        "path": canonical_path,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def preprocessing_settings(
    *,
    video_ctx_size,
    video_ctx_num_frames,
    video_ctx_frame_percentage,
    video_ctx_max_num_frames,
) -> dict[str, Any]:
    return {
        "video_ctx_size": list(video_ctx_size) if video_ctx_size is not None else None,
        "video_ctx_num_frames": video_ctx_num_frames,
        "video_ctx_frame_percentage": video_ctx_frame_percentage,
        "video_ctx_max_num_frames": video_ctx_max_num_frames,
        "normalize_mean": 0.5,
        "normalize_std": 0.5,
    }


def cache_entry_path(cache_dir: str | Path, source_path: str) -> Path:
    source_key = canonical_source_path(source_path).encode("utf-8")
    filename = f"{hashlib.sha256(source_key).hexdigest()}.pt"
    return Path(cache_dir) / "entries" / filename


def manifest_path(cache_dir: str | Path) -> Path:
    return Path(cache_dir) / MANIFEST_FILENAME


def load_manifest(cache_dir: str | Path) -> dict[str, Any]:
    path = manifest_path(cache_dir)
    if not path.exists():
        raise VideoCtxLatentCacheError(
            f"Missing video_ctx latent manifest: {path}. "
            "Run scripts/precompute_video_ctx_latents.py first."
        )
    try:
        with path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
    except (OSError, json.JSONDecodeError) as error:
        raise VideoCtxLatentCacheError(f"Invalid video_ctx latent manifest: {path}") from error
    if not isinstance(manifest, dict):
        raise VideoCtxLatentCacheError(f"Video_ctx latent manifest must be an object: {path}")
    return manifest


def validate_manifest(
    manifest: dict[str, Any],
    *,
    expected_preprocessing: dict[str, Any],
) -> None:
    if manifest.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise VideoCtxLatentCacheError(
            "Video_ctx latent cache schema mismatch: "
            f"expected {CACHE_SCHEMA_VERSION}, got {manifest.get('schema_version')}"
        )
    if manifest.get("complete") is not True:
        raise VideoCtxLatentCacheError("Video_ctx latent cache is incomplete.")
    if manifest.get("preprocessing") != expected_preprocessing:
        raise VideoCtxLatentCacheError(
            "Video_ctx latent cache preprocessing does not match the dataset configuration."
        )


def build_payload(
    *,
    source_path: str,
    preprocessing: dict[str, Any],
    latent: torch.Tensor,
) -> dict[str, Any]:
    if latent.ndim != 4:
        raise ValueError(f"Cached video_ctx latent must be [C,T,H,W], got {tuple(latent.shape)}")
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "source": source_fingerprint(source_path),
        "preprocessing": preprocessing,
        "latent": latent.detach().to(device="cpu", dtype=torch.bfloat16).contiguous(),
    }


def validate_payload(
    payload: dict[str, Any],
    *,
    source_path: str,
    expected_preprocessing: dict[str, Any],
) -> torch.Tensor:
    if not isinstance(payload, dict):
        raise VideoCtxLatentCacheError("Video_ctx latent cache entry must be a dictionary.")
    if payload.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise VideoCtxLatentCacheError("Video_ctx latent cache entry schema mismatch.")
    try:
        current_source = source_fingerprint(source_path)
    except OSError as error:
        raise VideoCtxLatentCacheError(
            f"Unable to fingerprint video_ctx source: {source_path}"
        ) from error
    if payload.get("source") != current_source:
        raise VideoCtxLatentCacheError(
            f"Video_ctx latent cache is stale for source: {canonical_source_path(source_path)}"
        )
    if payload.get("preprocessing") != expected_preprocessing:
        raise VideoCtxLatentCacheError(
            f"Video_ctx latent cache preprocessing mismatch for source: {canonical_source_path(source_path)}"
        )
    latent = payload.get("latent")
    if not isinstance(latent, torch.Tensor) or latent.ndim != 4:
        raise VideoCtxLatentCacheError("Cached video_ctx latent must be a 4D tensor [C,T,H,W].")
    if latent.shape[0] != 48:
        raise VideoCtxLatentCacheError(
            f"Cached video_ctx latent must have 48 channels, got {latent.shape[0]}."
        )
    # Clone to detach from any mmap-backed or otherwise non-resizable storage
    # (PyTorch 2.6 defaults torch.load to mmap=True). default_collate calls
    # resize_ on the tensor's storage when stacking a batch, which fails for
    # file-backed storage, especially under multi-node DataLoader workers.
    return latent.contiguous().clone()


def load_latent(
    cache_dir: str | Path,
    source_path: str,
    *,
    expected_preprocessing: dict[str, Any],
) -> torch.Tensor:
    path = cache_entry_path(cache_dir, source_path)
    if not path.exists():
        raise VideoCtxLatentCacheError(
            f"Missing video_ctx latent cache entry: {path}. "
            "Run scripts/precompute_video_ctx_latents.py first."
        )
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=False)
    except Exception as error:
        raise VideoCtxLatentCacheError(f"Unable to read video_ctx latent cache entry: {path}") from error
    return validate_payload(
        payload,
        source_path=source_path,
        expected_preprocessing=expected_preprocessing,
    )


def atomic_torch_save(payload: dict[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{path.name}.tmp.{uuid.uuid4().hex}"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def atomic_json_save(payload: dict[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{path.name}.tmp.{uuid.uuid4().hex}"
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, sort_keys=True, indent=2)
        f.write("\n")
    os.replace(tmp_path, path)
