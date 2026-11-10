#!/usr/bin/env python3
"""Training preflight: create whatever the config needs but is missing.

Invoked automatically by scripts/train_zero1.sh before launching training
(set PREFLIGHT=0 to skip). Also runnable standalone with the same hydra
overrides as train.py.

Prepares, in order:

  1. the T5 text-embedding cache for the configured datasets
     (scripts/precompute_text_embeds.py, incremental per prompt);
  2. the video_ctx latent cache when `video_ctx_latent_cache_mode: required`
     (scripts/precompute_video_ctx_latents.py, incremental; automatically
     rebuilt when the datasets or the preprocessing settings change, tracked
     via a signature stored in the cache directory);
  3. the ActionDiT backbone (`model.action_dit_pretrained_path`), linearly
     interpolated from the Wan2.2 DiT.

Everything is idempotent and incremental; dataset paths that do not exist
yet are left to training itself to report.
"""
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO_ROOT / "scripts"
for _p in (str(_SCRIPTS), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import hydra  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

import precompute_text_embeds as pte  # noqa: E402  (registers resolvers on import)

# CLI overrides, captured before hydra.compose consumes them.
OVERRIDES = sys.argv[1:]

SIGNATURE_FILENAME = "preflight_dataset_signature.json"


def _log(msg: str):
    print(f"[preflight] {msg}", flush=True)


def _repo_path(value) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (_REPO_ROOT / path)


def _run(script: str, *args: str):
    cmd = [sys.executable, str(_SCRIPTS / script), *args]
    _log(f"running {' '.join([script, *args])}")
    subprocess.run(cmd, cwd=_REPO_ROOT, check=True)


def _iter_cache_nodes(cfg: DictConfig):
    yield from pte._iter_dataset_nodes(cfg, path="data")


def _datasets_available(dataset_dirs: list[str]) -> bool:
    return bool(dataset_dirs) and all(_repo_path(d).exists() for d in dataset_dirs)


def _dataset_signature(dataset_dirs: list[str]) -> str:
    """Hash of the conditions metadata; changes when datasets are swapped
    or their condition references change."""
    parts = [str(_repo_path(d).resolve()) for d in dataset_dirs]
    for d in dataset_dirs:
        for name in ("conditions_list.json", "conditions.json"):
            meta = _repo_path(d) / "meta" / name
            if meta.exists():
                digest = hashlib.sha256(meta.read_bytes()).hexdigest()
                parts.append(f"{meta.name}:{digest}")
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def _check_text_cache(cfg: DictConfig, dataset_dirs: list[str]) -> bool:
    cache_dirs, context_lens = [], set()
    for _path, node in _iter_cache_nodes(cfg):
        cache = node.get("text_embedding_cache_dir")
        if cache is None:
            continue
        cache_path = _repo_path(cache)
        if cache_path not in cache_dirs:
            cache_dirs.append(cache_path)
        if node.get("context_len") is not None:
            context_lens.add(int(node.get("context_len")))
    if not cache_dirs:
        return False

    try:
        prompts = pte._read_unique_prompts([str(_repo_path(d)) for d in dataset_dirs])
    except (FileNotFoundError, KeyError):
        return False  # dataset metadata incomplete; training reports it
    context_len = pte._resolve_context_len(context_lens) if context_lens else pte.DEFAULT_CONTEXT_LEN
    enc_id = pte._model_id_to_enc_id(str(cfg.model.get("model_id", pte.DEFAULT_MODEL_ID)))

    missing = 0
    for prompt in prompts:
        filename = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        filename = f"{filename}.t5_len{context_len}.{enc_id}.pt"
        if any(not (d / filename).exists() for d in cache_dirs):
            missing += 1
    if missing:
        _log(f"text embeds: {len(prompts) - missing}/{len(prompts)} prompts cached")
    return missing > 0


def _video_ctx_settings(cfg: DictConfig, dataset_dirs: list[str]) -> dict:
    """Everything that determines the video_ctx latent cache contents."""
    train = cfg.data.train

    def plain(value):
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return OmegaConf.to_container(value)

    return {
        "datasets": _dataset_signature(dataset_dirs),
        "video_ctx_size": plain(train.get("video_ctx_size")),
        "video_ctx_num_frames": train.get("video_ctx_num_frames"),
        "video_ctx_frame_percentage": train.get("video_ctx_frame_percentage"),
        "video_ctx_max_num_frames": train.get("video_ctx_max_num_frames"),
        "model_id": str(cfg.model.model_id),
        "vae_cond_pretrain_path": str(cfg.model.get("vae_cond_pretrain_path") or ""),
    }


def _video_ctx_cache_needs_rebuild(cfg: DictConfig, dataset_dirs: list[str]) -> bool:
    train = cfg.data.train
    mode = str(train.get("video_ctx_latent_cache_mode") or "off").lower()
    cache = train.get("video_ctx_latent_cache_dir")
    if mode != "required" or cache is None:
        return False
    cache_dir = _repo_path(cache)
    if not (cache_dir / "manifest.json").exists():
        _log(f"video_ctx latents: no manifest under {cache}")
        return True
    sig_file = cache_dir / SIGNATURE_FILENAME
    if not sig_file.exists():
        _log("video_ctx latents: dataset signature missing")
        return True
    try:
        recorded = json.loads(sig_file.read_text())
    except json.JSONDecodeError:
        _log("video_ctx latents: dataset signature unreadable")
        return True
    if recorded != _video_ctx_settings(cfg, dataset_dirs):
        _log("video_ctx latents: data or preprocessing settings changed")
        return True
    return False


def _write_signature(cfg: DictConfig, dataset_dirs: list[str]):
    train = cfg.data.train
    cache = train.get("video_ctx_latent_cache_dir")
    if cache is None or str(train.get("video_ctx_latent_cache_mode") or "off").lower() != "required":
        return
    cache_dir = _repo_path(cache)
    payload = _video_ctx_settings(cfg, dataset_dirs)
    (cache_dir / SIGNATURE_FILENAME).write_text(json.dumps(payload, sort_keys=True))
    return


def _check_backbone(cfg: DictConfig) -> bool:
    backbone = cfg.model.get("action_dit_pretrained_path")
    if not backbone or bool(cfg.model.get("skip_dit_load_from_pretrain", False)):
        return False
    if _repo_path(backbone).exists():
        return False
    _log(f"ActionDiT backbone missing: {backbone}")
    return True


def _write_model_config(cfg: DictConfig) -> Path:
    # Resolve against the original root so `${data.train...}` interpolations
    # inside the model node can be resolved (create() would re-root and break).
    resolved = OmegaConf.to_yaml(cfg.model, resolve=True)
    tmp = tempfile.NamedTemporaryFile(
        "w", suffix=".yaml", prefix="preflight_model_", delete=False
    )
    tmp.write(resolved)
    tmp.close()
    return Path(tmp.name)


def main():
    with hydra.initialize(config_path="../configs", version_base="1.3"):
        cfg = hydra.compose(config_name="train", overrides=OVERRIDES)

    dataset_dirs, _cache_dirs, _ctx = pte._collect_dataset_settings(cfg)

    if _datasets_available(dataset_dirs):
        if _check_text_cache(cfg, dataset_dirs):
            _run("precompute_text_embeds.py", *OVERRIDES)
        if _video_ctx_cache_needs_rebuild(cfg, dataset_dirs):
            _run("precompute_video_ctx_latents.py", *OVERRIDES)
            _write_signature(cfg, dataset_dirs)
    # Datasets not in place yet: training itself reports the missing path.

    if _check_backbone(cfg):
        model_cfg = _write_model_config(cfg)
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            _run(
                "preprocess_action_dit_backbone.py",
                "--model-config", str(model_cfg),
                "--output", str(_repo_path(cfg.model.action_dit_pretrained_path)),
                "--device", device,
                "--dtype", "bfloat16",
            )
        finally:
            model_cfg.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
