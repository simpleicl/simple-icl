from __future__ import annotations

import json
import math
import os
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import imageio
import numpy as np
import torch
from PIL import Image


def _expand_attention_mask(mask: torch.Tensor, query_len: int) -> torch.Tensor:
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(1)
    elif mask.ndim != 4:
        raise ValueError(f"Unsupported attention mask shape: {tuple(mask.shape)}")
    if mask.shape[-2] not in (1, query_len):
        raise ValueError(
            f"Attention mask query length mismatch: {mask.shape[-2]} vs {query_len}"
        )
    return mask


@torch.no_grad()
def summarize_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    num_heads: int,
    attn_mask: Optional[torch.Tensor] = None,
    key_limit: Optional[int] = None,
    keep_matrix: bool = False,
    query_chunk_size: int = 16,
) -> dict[str, np.ndarray]:
    """Recompute softmax attention and retain compact CPU summaries."""
    if q.ndim != 3 or k.ndim != 3 or q.shape[0] != k.shape[0]:
        raise ValueError(
            f"Expected q/k as [B,L,D], got {tuple(q.shape)} and {tuple(k.shape)}"
        )
    if q.shape[-1] % num_heads or k.shape[-1] != q.shape[-1]:
        raise ValueError(
            "q/k hidden dimensions must match and be divisible by num_heads"
        )

    batch_size, query_len, hidden_dim = q.shape
    key_len = k.shape[1]
    head_dim = hidden_dim // num_heads
    selected_keys = key_len if key_limit is None else min(int(key_limit), key_len)
    if selected_keys <= 0:
        raise ValueError(f"key_limit must select at least one key, got {key_limit}")

    qh = q.reshape(batch_size, query_len, num_heads, head_dim).permute(0, 2, 1, 3)
    kh = k.reshape(batch_size, key_len, num_heads, head_dim).permute(0, 2, 1, 3)
    expanded_mask = (
        _expand_attention_mask(attn_mask, query_len).to(device=q.device)
        if attn_mask is not None
        else None
    )
    key_totals = torch.zeros(
        (num_heads, selected_keys), dtype=torch.float64, device="cpu"
    )
    entropy_totals = torch.zeros(num_heads, dtype=torch.float64, device="cpu")
    peak_totals = torch.zeros(num_heads, dtype=torch.float64, device="cpu")
    topk_totals = torch.zeros(num_heads, dtype=torch.float64, device="cpu")
    selected_mass_totals = torch.zeros(num_heads, dtype=torch.float64, device="cpu")
    matrix_chunks: list[np.ndarray] = []

    chunk_size = max(1, int(query_chunk_size))
    for start in range(0, query_len, chunk_size):
        end = min(query_len, start + chunk_size)
        logits = torch.matmul(qh[:, :, start:end], kh.transpose(-2, -1))
        logits = logits.float() * (head_dim**-0.5)
        if expanded_mask is not None:
            chunk_mask = (
                expanded_mask
                if expanded_mask.shape[-2] == 1
                else expanded_mask[..., start:end, :]
            )
            if chunk_mask.dtype == torch.bool:
                logits = logits.masked_fill(~chunk_mask, float("-inf"))
            else:
                logits = logits + chunk_mask.to(dtype=logits.dtype)
        probs = torch.softmax(logits, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)

        selected = probs[..., :selected_keys]
        key_totals += selected.sum(dim=(0, 2)).double().cpu()
        selected_mass_totals += selected.sum(dim=-1).sum(dim=(0, 2)).double().cpu()
        entropy = -(probs.clamp_min(1e-12).log() * probs).sum(dim=-1)
        entropy_totals += entropy.sum(dim=(0, 2)).double().cpu()
        peak_totals += probs.max(dim=-1).values.sum(dim=(0, 2)).double().cpu()
        topk_totals += (
            probs.topk(min(5, key_len), dim=-1)
            .values.sum(dim=-1)
            .sum(dim=(0, 2))
            .double()
            .cpu()
        )
        if keep_matrix:
            matrix_chunks.append(probs.mean(dim=(0, 1)).float().cpu().numpy())

    row_count = float(batch_size * query_len)
    result = {
        "key_attention": (key_totals / row_count).float().numpy(),
        "head_entropy": (entropy_totals / row_count).float().numpy(),
        "head_peak": (peak_totals / row_count).float().numpy(),
        "head_top5_mass": (topk_totals / row_count).float().numpy(),
        "head_selected_mass": (selected_mass_totals / row_count).float().numpy(),
    }
    if keep_matrix:
        result["matrix"] = np.concatenate(matrix_chunks, axis=0)
    return result


def attention_rollout(
    query_relevance: np.ndarray,
    self_attention: dict[int, np.ndarray],
    cross_attention: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trace final Q-Former query relevance back to visual patch tokens."""
    relevance = np.asarray(query_relevance, dtype=np.float64)
    relevance = relevance / max(float(relevance.sum()), 1e-12)
    visual = None
    contributions: list[np.ndarray] = []
    contribution_layers: list[int] = []
    all_layers = sorted(set(self_attention) | set(cross_attention), reverse=True)
    for layer_idx in all_layers:
        if layer_idx in cross_attention:
            contribution = relevance @ np.asarray(
                cross_attention[layer_idx], dtype=np.float64
            )
            contributions.append(contribution)
            contribution_layers.append(layer_idx)
            visual = contribution.copy() if visual is None else visual + contribution
        if layer_idx in self_attention:
            matrix = np.asarray(self_attention[layer_idx], dtype=np.float64)
            augmented = matrix + np.eye(matrix.shape[0], dtype=np.float64)
            augmented /= np.maximum(augmented.sum(axis=-1, keepdims=True), 1e-12)
            relevance = relevance @ augmented
    if visual is None:
        raise ValueError("Q-Former rollout requires at least one cross-attention layer")
    visual /= max(float(visual.sum()), 1e-12)
    stacked = np.stack(contributions, axis=0)
    stacked /= np.maximum(stacked.sum(axis=-1, keepdims=True), 1e-12)
    return (
        visual.astype(np.float32),
        stacked.astype(np.float32),
        np.asarray(contribution_layers),
    )


@dataclass
class _AttentionRecord:
    key_attention: np.ndarray
    head_entropy: np.ndarray
    head_peak: np.ndarray
    head_top5_mass: np.ndarray
    head_selected_mass: np.ndarray


@dataclass
class _ReplanCapture:
    index: int
    sim_step: int
    observation: np.ndarray
    patch_grid_size: tuple[int, int, int]
    encoder_grid_size: tuple[int, int, int]
    qformer_self: dict[int, np.ndarray] = field(default_factory=dict)
    qformer_cross: dict[int, np.ndarray] = field(default_factory=dict)
    qformer_stats: dict[str, dict[int, _AttentionRecord]] = field(
        default_factory=lambda: {"self": {}, "cross": {}}
    )
    dit: dict[str, dict[int, list[_AttentionRecord]]] = field(
        default_factory=lambda: {
            "action": defaultdict(list),
            "video": defaultdict(list),
        }
    )


class AttentionVisualizationCollector:
    SCHEMA_VERSION = 1

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        save_stats: bool = True,
        fps: int = 8,
        query_chunk_size: int = 16,
    ) -> None:
        self.model = model
        self.save_stats = bool(save_stats)
        self.fps = max(1, int(fps))
        self.query_chunk_size = max(1, int(query_chunk_size))
        self.qformer = self._validate_model(model)
        self.num_query_tokens = int(self.qformer.num_query_tokens)
        self._observer_modules: list[tuple[Any, Any]] = []
        self._active: Optional[_ReplanCapture] = None
        self._replans: list[dict[str, Any]] = []
        self.output_dir: Optional[Path] = None
        self.condition_video_path: Optional[Path] = None
        self.condition_frames: list[np.ndarray] = []
        self._attach_observers()

    @staticmethod
    def _validate_model(model: torch.nn.Module) -> Any:
        video_expert = getattr(model, "video_expert", None)
        if video_expert is None or not getattr(
            video_expert, "has_video_all_condition", False
        ):
            raise ValueError(
                "Attention visualization requires a video_all-conditioned VideoDiT"
            )
        if getattr(video_expert, "video_all_injection_mode", None) != "qformer":
            raise ValueError(
                "Attention visualization requires video_all_injection_mode='qformer'"
            )
        qformer = getattr(video_expert, "video_all_qformer", None)
        if qformer is None:
            raise ValueError(
                "Attention visualization requires video_expert.video_all_qformer"
            )
        if getattr(model, "action_expert", None) is None:
            raise ValueError("Attention visualization requires an ActionDiT expert")
        return qformer

    def _install(self, module: Any, callback: Callable[..., None]) -> None:
        previous = getattr(module, "_attention_observer", None)
        if previous is not None:
            raise RuntimeError("An attention observer is already installed")
        module._attention_observer = callback
        self._observer_modules.append((module, previous))

    def _attach_observers(self) -> None:
        for layer_idx, block in enumerate(self.qformer.blocks):
            self._install(
                block.self_attn,
                lambda layer_idx=layer_idx, **kwargs: self._capture_qformer(
                    "self", layer_idx, **kwargs
                ),
            )
            if getattr(block, "has_cross_attn", False):
                self._install(
                    block.cross_attn,
                    lambda layer_idx=layer_idx, **kwargs: self._capture_qformer(
                        "cross", layer_idx, **kwargs
                    ),
                )
        for source, expert in (
            ("video", self.model.video_expert),
            ("action", self.model.action_expert),
        ):
            for layer_idx, block in enumerate(expert.blocks):
                self._install(
                    block.cross_attn,
                    lambda source=source, layer_idx=layer_idx, **kwargs: (
                        self._capture_dit(source, layer_idx, **kwargs)
                    ),
                )

    def detach(self) -> None:
        for module, previous in self._observer_modules:
            if previous is None:
                try:
                    delattr(module, "_attention_observer")
                except AttributeError:
                    pass
            else:
                module._attention_observer = previous
        self._observer_modules.clear()

    def start_episode(
        self, output_dir: Path | str, condition_video_path: Path | str
    ) -> None:
        if self._active is not None:
            raise RuntimeError(
                "Cannot start an attention episode during an active replan"
            )
        self.output_dir = Path(output_dir).resolve()
        self.condition_video_path = Path(condition_video_path).resolve()
        if not self.condition_video_path.is_file():
            raise FileNotFoundError(
                f"Sampled condition video not found: {self.condition_video_path}"
            )
        self.condition_frames = self._read_video(self.condition_video_path)
        self._replans = []

    def start_replan(
        self,
        observation: np.ndarray,
        sim_step: int,
        video_all_latents: torch.Tensor,
    ) -> None:
        if self.output_dir is None:
            raise RuntimeError("start_episode must be called before start_replan")
        if self._active is not None:
            raise RuntimeError("Previous attention replan is still active")
        patch_grid_size = self._condition_grid_size(video_all_latents)
        self._active = _ReplanCapture(
            index=len(self._replans),
            sim_step=int(sim_step),
            observation=np.asarray(observation, dtype=np.uint8).copy(),
            patch_grid_size=patch_grid_size,
            encoder_grid_size=tuple(
                int(value) for value in video_all_latents.shape[-3:]
            ),
        )

    def abort_replan(self) -> None:
        self._active = None

    def finish_replan(self) -> dict[str, Any]:
        capture = self._active
        if capture is None:
            raise RuntimeError("No active attention replan")
        try:
            result = self._build_replan_result(capture)
            self._write_replan(capture, result)
            self._replans.append(
                {
                    "index": result["index"],
                    "sim_step": result["sim_step"],
                    "patch_grid_size": result["patch_grid_size"],
                    "encoder_grid_size": result["encoder_grid_size"],
                    "directory": result["directory"],
                    "qformer_self_layers": sorted(result["qformer_self"]),
                    "qformer_cross_layers": sorted(result["qformer_cross"]),
                    "dit_layers": {
                        source: values["layers"].tolist()
                        for source, values in result["sources"].items()
                    },
                    "dit_call_count": {
                        source: int(values["qformer_attention"].shape[0])
                        for source, values in result["sources"].items()
                    },
                    "observation": result["observation"],
                    "sources": {
                        source: {
                            key: values[key]
                            for key in (
                                "condition_attention",
                                "spatial_attention",
                                "temporal_attention",
                            )
                        }
                        for source, values in result["sources"].items()
                    },
                }
            )
            return result
        finally:
            self._active = None

    def finish_episode(self) -> dict[str, Any]:
        if self.output_dir is None:
            return {}
        if self._active is not None:
            self.abort_replan()
        attention_dir = self.output_dir / "attention"
        attention_dir.mkdir(parents=True, exist_ok=True)
        if not self._replans:
            self.output_dir = None
            self.condition_video_path = None
            self.condition_frames = []
            return {}
        video_paths: dict[str, str] = {}
        for source in ("action", "video"):
            path = attention_dir / f"{source}_attention.mp4"
            self._write_source_video(source, path)
            video_paths[source] = str(path)
        manifest_path = attention_dir / "manifest.json"
        manifest = {
            "schema_version": self.SCHEMA_VERSION,
            "method": "attention_rollout",
            "normalization": "head mean; DiT query/layer/step mean; Q-Former residual rollout",
            "interpolation": "latent attention is bilinearly/linearly interpolated only for display",
            "num_query_tokens": self.num_query_tokens,
            "save_stats": self.save_stats,
            "condition_video_path": str(self.condition_video_path),
            "replan_count": len(self._replans),
            "sources": ["action", "video"],
            "videos": video_paths,
            "replans": [
                {
                    "index": item["index"],
                    "sim_step": item["sim_step"],
                    "patch_grid_size": list(item["patch_grid_size"]),
                    "encoder_grid_size": list(item["encoder_grid_size"]),
                    "qformer_self_layers": item["qformer_self_layers"],
                    "qformer_cross_layers": item["qformer_cross_layers"],
                    "dit_layers": item["dit_layers"],
                    "dit_call_count": item["dit_call_count"],
                    "directory": item["directory"],
                }
                for item in self._replans
            ],
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        result = {
            "enabled": True,
            "manifest_path": str(manifest_path),
            "action_video_path": video_paths.get("action"),
            "video_video_path": video_paths.get("video"),
            "replan_count": len(self._replans),
        }
        self.output_dir = None
        self.condition_video_path = None
        self.condition_frames = []
        self._replans = []
        return result

    def _condition_grid_size(self, latents: torch.Tensor) -> tuple[int, int, int]:
        if latents.ndim != 5:
            raise ValueError(
                f"video_all_latents must be [B,C,T,H,W], got {tuple(latents.shape)}"
            )
        conv = self.qformer.video_all_embedding
        dims = []
        for size, kernel, stride, padding, dilation in zip(
            latents.shape[-3:],
            conv.kernel_size,
            conv.stride,
            conv.padding,
            conv.dilation,
        ):
            dims.append(
                (int(size) + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1
            )
        return tuple(dims)  # type: ignore[return-value]

    def _capture_qformer(self, kind: str, layer_idx: int, **kwargs: Any) -> None:
        capture = self._active
        if capture is None:
            return
        summary = summarize_attention(
            kwargs["q"],
            kwargs["k"],
            num_heads=kwargs["num_heads"],
            attn_mask=kwargs.get("attn_mask"),
            keep_matrix=True,
            query_chunk_size=self.query_chunk_size,
        )
        matrix = summary.pop("matrix")
        if kind == "cross" and matrix.shape[1] != math.prod(capture.patch_grid_size):
            raise ValueError(
                "Q-Former condition token count "
                f"{matrix.shape[1]} does not match grid {capture.patch_grid_size}"
            )
        getattr(capture, f"qformer_{kind}")[layer_idx] = matrix
        capture.qformer_stats[kind][layer_idx] = _AttentionRecord(**summary)

    def _capture_dit(self, source: str, layer_idx: int, **kwargs: Any) -> None:
        capture = self._active
        if capture is None:
            return
        summary = summarize_attention(
            kwargs["q"],
            kwargs["k"],
            num_heads=kwargs["num_heads"],
            attn_mask=kwargs.get("attn_mask"),
            key_limit=self.num_query_tokens,
            query_chunk_size=self.query_chunk_size,
        )
        capture.dit[source][layer_idx].append(_AttentionRecord(**summary))

    def _stack_dit(self, capture: _ReplanCapture, source: str) -> dict[str, np.ndarray]:
        by_layer = capture.dit[source]
        layers = sorted(by_layer)
        if not layers:
            raise ValueError(f"No {source} DiT cross-attention was captured")
        call_count = min(len(by_layer[layer]) for layer in layers)
        if call_count <= 0 or any(
            len(by_layer[layer]) != call_count for layer in layers
        ):
            counts = {layer: len(by_layer[layer]) for layer in layers}
            raise ValueError(f"Inconsistent {source} DiT attention calls: {counts}")

        def stack_field(name: str) -> np.ndarray:
            return np.stack(
                [
                    np.stack(
                        [getattr(by_layer[layer][call], name) for layer in layers],
                        axis=0,
                    )
                    for call in range(call_count)
                ],
                axis=0,
            )

        per_head = stack_field("key_attention")
        return {
            "layers": np.asarray(layers, dtype=np.int32),
            "qformer_attention": per_head.mean(axis=2),
            "head_entropy": stack_field("head_entropy"),
            "head_peak": stack_field("head_peak"),
            "head_top5_mass": stack_field("head_top5_mass"),
            "head_qformer_mass": stack_field("head_selected_mass"),
        }

    def _build_replan_result(self, capture: _ReplanCapture) -> dict[str, Any]:
        if not capture.qformer_self or not capture.qformer_cross:
            raise ValueError("Incomplete Q-Former attention capture")
        sources: dict[str, dict[str, np.ndarray]] = {}
        for source in ("action", "video"):
            dit = self._stack_dit(capture, source)
            query_relevance = dit["qformer_attention"].mean(axis=(0, 1))
            visual, contributions, contribution_layers = attention_rollout(
                query_relevance, capture.qformer_self, capture.qformer_cross
            )
            patch_grid = visual.reshape(capture.patch_grid_size)
            encoder_grid = self._backproject_patch_attention(
                patch_grid, capture.encoder_grid_size
            )
            sources[source] = {
                **dit,
                "query_relevance": query_relevance
                / max(float(query_relevance.sum()), 1e-12),
                "patch_attention": patch_grid,
                "condition_attention": encoder_grid,
                "spatial_attention": encoder_grid.sum(axis=0),
                "temporal_attention": encoder_grid.sum(axis=(1, 2)),
                "qformer_cross_contributions": contributions.reshape(
                    (contributions.shape[0],) + capture.patch_grid_size
                ),
                "qformer_cross_contribution_layers": contribution_layers,
            }
        return {
            "index": capture.index,
            "sim_step": capture.sim_step,
            "patch_grid_size": capture.patch_grid_size,
            "encoder_grid_size": capture.encoder_grid_size,
            "observation": capture.observation,
            "qformer_self": capture.qformer_self,
            "qformer_cross": capture.qformer_cross,
            "qformer_stats": capture.qformer_stats,
            "sources": sources,
        }

    def _write_replan(self, capture: _ReplanCapture, result: dict[str, Any]) -> None:
        assert self.output_dir is not None
        relative_dir = Path("attention") / f"replan_{capture.index:04d}"
        output_dir = self.output_dir / relative_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        result["directory"] = str(relative_dir)
        for source in ("action", "video"):
            values = result["sources"][source]
            self._save_spatial_plot(
                values["spatial_attention"],
                output_dir / f"{source}_spatial.png",
                source,
            )
            self._save_temporal_plot(
                values["temporal_attention"],
                output_dir / f"{source}_temporal.png",
                source,
            )
        if self.save_stats:
            arrays: dict[str, np.ndarray] = {
                "patch_grid_size": np.asarray(capture.patch_grid_size, dtype=np.int32),
                "encoder_grid_size": np.asarray(
                    capture.encoder_grid_size, dtype=np.int32
                ),
                "qformer_self_layers": np.asarray(
                    sorted(capture.qformer_self), dtype=np.int32
                ),
                "qformer_self_attention": np.stack(
                    [capture.qformer_self[i] for i in sorted(capture.qformer_self)]
                ).astype(np.float16),
                "qformer_cross_layers": np.asarray(
                    sorted(capture.qformer_cross), dtype=np.int32
                ),
                "qformer_cross_attention": np.stack(
                    [capture.qformer_cross[i] for i in sorted(capture.qformer_cross)]
                ).astype(np.float16),
            }
            for kind in ("self", "cross"):
                stats = capture.qformer_stats[kind]
                layers = sorted(stats)
                for field_name in (
                    "head_entropy",
                    "head_peak",
                    "head_top5_mass",
                    "head_selected_mass",
                ):
                    arrays[f"qformer_{kind}_{field_name}"] = np.stack(
                        [getattr(stats[i], field_name) for i in layers]
                    ).astype(np.float16)
            for source, values in result["sources"].items():
                for key, value in values.items():
                    array = np.asarray(value)
                    arrays[f"{source}_{key}"] = (
                        array.astype(np.float16)
                        if np.issubdtype(array.dtype, np.floating)
                        else array
                    )
            np.savez_compressed(output_dir / "attention_stats.npz", **arrays)

    def _backproject_patch_attention(
        self,
        patch_attention: np.ndarray,
        encoder_grid_size: tuple[int, int, int],
    ) -> np.ndarray:
        conv = self.qformer.video_all_embedding
        output = np.zeros(encoder_grid_size, dtype=np.float64)
        for patch_index in np.ndindex(patch_attention.shape):
            covered: list[tuple[int, int, int]] = []
            for kernel_index in np.ndindex(conv.kernel_size):
                latent_index = tuple(
                    patch_index[axis] * conv.stride[axis]
                    - conv.padding[axis]
                    + kernel_index[axis] * conv.dilation[axis]
                    for axis in range(3)
                )
                if all(
                    0 <= latent_index[axis] < encoder_grid_size[axis]
                    for axis in range(3)
                ):
                    covered.append(latent_index)
            if not covered:
                continue
            share = float(patch_attention[patch_index]) / len(covered)
            for latent_index in covered:
                output[latent_index] += share
        output /= max(float(output.sum()), 1e-12)
        return output.astype(np.float32)

    @staticmethod
    def _pyplot():
        mpl_cache = Path(tempfile.gettempdir()) / f"simpleicl-matplotlib-{os.getpid()}"
        mpl_cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        return plt

    def _save_spatial_plot(self, values: np.ndarray, path: Path, source: str) -> None:
        plt = self._pyplot()
        fig, ax = plt.subplots(figsize=(6.4, 5.0), dpi=120)
        image = ax.imshow(values, cmap="magma", interpolation="nearest")
        ax.set(
            title=f"{source.capitalize()}DiT spatial attention",
            xlabel="latent W",
            ylabel="latent H",
        )
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)

    def _save_temporal_plot(self, values: np.ndarray, path: Path, source: str) -> None:
        plt = self._pyplot()
        fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=120)
        indices = np.arange(len(values))
        ax.plot(indices, values, color="#1976a3", marker="o", linewidth=2)
        ax.fill_between(indices, values, color="#67b7c7", alpha=0.3)
        ax.set(
            title=f"{source.capitalize()}DiT temporal attention",
            xlabel="condition latent time",
            ylabel="normalized weight",
            xlim=(-0.25, max(len(values) - 0.75, 0.25)),
            ylim=(0, max(float(values.max()) * 1.15, 1e-3)),
        )
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)

    @staticmethod
    def _read_video(path: Path) -> list[np.ndarray]:
        reader = imageio.get_reader(str(path))
        try:
            frames = [np.asarray(frame, dtype=np.uint8) for frame in reader]
        finally:
            reader.close()
        if not frames:
            raise ValueError(f"Condition video contains no frames: {path}")
        return frames

    @staticmethod
    def _resize_map(values: np.ndarray, height: int, width: int) -> np.ndarray:
        image = Image.fromarray(np.asarray(values, dtype=np.float32), mode="F")
        return np.asarray(
            image.resize((width, height), resample=Image.BILINEAR), dtype=np.float32
        )

    def _interpolated_attention_frames(self, grid: np.ndarray) -> list[np.ndarray]:
        target_count = len(self.condition_frames)
        if grid.shape[0] == 1:
            return [grid[0]] * target_count
        positions = np.linspace(0, grid.shape[0] - 1, target_count)
        frames = []
        for position in positions:
            low = int(math.floor(position))
            high = min(low + 1, grid.shape[0] - 1)
            alpha = position - low
            frames.append(grid[low] * (1.0 - alpha) + grid[high] * alpha)
        return frames

    def _write_source_video(self, source: str, path: Path) -> None:
        plt = self._pyplot()
        writer = imageio.get_writer(
            str(path),
            fps=self.fps,
            codec="libx264",
            format="FFMPEG",
            pixelformat="yuv420p",
        )
        try:
            for replan in self._replans:
                values = replan["sources"][source]
                grid_frames = self._interpolated_attention_frames(
                    values["condition_attention"]
                )
                latent_temporal = values["temporal_attention"]
                temporal = np.interp(
                    np.linspace(
                        0, len(latent_temporal) - 1, len(self.condition_frames)
                    ),
                    np.arange(len(latent_temporal)),
                    latent_temporal,
                )
                global_max = max(float(values["condition_attention"].max()), 1e-12)
                for frame_idx, (condition, attention) in enumerate(
                    zip(self.condition_frames, grid_frames)
                ):
                    height, width = condition.shape[:2]
                    heat = self._resize_map(attention, height, width)
                    heat_rgb = plt.get_cmap("magma")(np.clip(heat / global_max, 0, 1))[
                        ..., :3
                    ]
                    overlay = np.clip(
                        condition.astype(np.float32) / 255.0 * 0.55 + heat_rgb * 0.45,
                        0,
                        1,
                    )
                    fig, axes = plt.subplots(2, 2, figsize=(12, 7.04), dpi=100)
                    axes[0, 0].imshow(replan["observation"])
                    axes[0, 0].set_title("Current robot observation")
                    axes[0, 1].imshow(overlay)
                    axes[0, 1].set_title(
                        f"Condition attention | frame {frame_idx + 1}/{len(grid_frames)}"
                    )
                    spatial_image = axes[1, 0].imshow(
                        values["spatial_attention"],
                        cmap="magma",
                        interpolation="nearest",
                    )
                    axes[1, 0].set_title("Spatial attention sum")
                    fig.colorbar(spatial_image, ax=axes[1, 0], fraction=0.046, pad=0.04)
                    axes[1, 1].plot(
                        np.arange(len(temporal)), temporal, color="#1976a3", linewidth=2
                    )
                    axes[1, 1].scatter(
                        [frame_idx], [temporal[frame_idx]], color="#d62728", s=45
                    )
                    axes[1, 1].set(
                        title="Temporal attention",
                        xlabel="sampled condition frame",
                        ylabel="interpolated weight",
                        ylim=(0, max(float(temporal.max()) * 1.15, 1e-3)),
                    )
                    axes[1, 1].grid(alpha=0.25)
                    for ax in axes.flat[:3]:
                        ax.set_xticks([])
                        ax.set_yticks([])
                    fig.suptitle(
                        f"{source.capitalize()}DiT attention | replan {replan['index']} | sim step {replan['sim_step']}"
                    )
                    fig.tight_layout()
                    fig.canvas.draw()
                    canvas = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[
                        ..., :3
                    ].copy()
                    writer.append_data(canvas)
                    plt.close(fig)
        finally:
            writer.close()
