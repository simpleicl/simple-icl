"""Minimal, self-contained LoRA utilities.

Kept dependency-free (no `peft`) so LoRA fine-tuning works in this environment.
The adapters wrap `nn.Linear` layers of the video-expert DiT: the original
weight is frozen and a low-rank residual `scaling * (x @ A^T @ B^T)` is added.
"""

import math
from typing import Iterable, Sequence

import torch
import torch.nn as nn

from simpleicl.utils.logging_config import get_logger

logger = get_logger(__name__)


class LoRALinear(nn.Module):
    """Wraps a frozen `nn.Linear` with an additive low-rank update.

    `forward(x) = base(x) + scaling * dropout(x) @ A^T @ B^T`, where
    `A` has shape `[rank, in_features]` and `B` has shape `[out_features, rank]`.
    `B` is zero-initialized so the adapter is an identity at the start of training.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRALinear expects an nn.Linear, got {type(base)}")
        if rank <= 0:
            raise ValueError(f"LoRA rank must be > 0, got {rank}")

        self.base = base
        self.base.requires_grad_(False)

        self.in_features = base.in_features
        self.out_features = base.out_features
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank

        device = base.weight.device
        dtype = base.weight.dtype
        self.lora_A = nn.Parameter(
            torch.empty(self.rank, self.in_features, device=device, dtype=dtype)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(self.out_features, self.rank, device=device, dtype=dtype)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.lora_dropout = (
            nn.Dropout(dropout) if dropout and dropout > 0.0 else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        lora = self.lora_dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        return out + lora * self.scaling


def _matches_suffix(dotted_name: str, suffixes: Sequence[str]) -> bool:
    for suffix in suffixes:
        if dotted_name == suffix or dotted_name.endswith("." + suffix):
            return True
    return False


def _under_prefix(dotted_name: str, prefixes: Sequence[str]) -> bool:
    for prefix in prefixes:
        if dotted_name == prefix or dotted_name.startswith(prefix + "."):
            return True
    return False


def inject_lora(
    root: nn.Module,
    target_suffixes: Sequence[str],
    rank: int,
    alpha: float,
    dropout: float = 0.0,
    exclude_prefixes: Sequence[str] = (),
) -> list[str]:
    """Replace every `nn.Linear` under `root` whose dotted path ends with one of
    `target_suffixes` (e.g. `self_attn.q`, `ffn.0`) with a `LoRALinear`.

    Any module whose dotted path falls under one of `exclude_prefixes` (e.g.
    `video_all_qformer`) is skipped, so fully-trainable submodules are never
    LoRA-wrapped and stay full-parameter.

    Returns the list of replaced module paths.
    """
    target_suffixes = list(target_suffixes)
    exclude_prefixes = list(exclude_prefixes)
    replaced: list[str] = []
    for module_name, module in list(root.named_modules()):
        for child_name, child in list(module.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            full_name = f"{module_name}.{child_name}" if module_name else child_name
            if exclude_prefixes and _under_prefix(full_name, exclude_prefixes):
                continue
            if _matches_suffix(full_name, target_suffixes):
                setattr(
                    module,
                    child_name,
                    LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout),
                )
                replaced.append(full_name)
    return replaced


# video_all projection names depend on injection mode; action_embedding is optional.
_VIDEO_ALL_ALIASES = {"video_all_qformer", "video_all_embedding"}
_OPTIONAL_MODULES = {"action_embedding"}


def resolve_train_modules(root: nn.Module, train_prefixes: Iterable[str]) -> list:
    """Resolve train_modules against the built model's parameters.

    Swap video_all_qformer/video_all_embedding to the available alias, or drop
    both if neither exists. Drop absent action_embedding with a warning; reject
    other unmatched prefixes with ValueError. Every returned prefix matches
    parameters, ensuring trained modules are included in LoRA checkpoints.
    """
    train_prefixes = list(train_prefixes)
    existing = {name.split(".", 1)[0] for name, _ in root.named_parameters()}

    resolved: list = []
    missing: list = []
    for prefix in train_prefixes:
        if prefix in existing:
            if prefix not in resolved:
                resolved.append(prefix)
            continue
        if prefix in _VIDEO_ALL_ALIASES:
            other = (_VIDEO_ALL_ALIASES - {prefix}).pop()
            if other in existing:
                logger.warning(
                    "train_modules: '%s' not present in the built model; using '%s' "
                    "instead (matches `video_all_injection_mode`).",
                    prefix,
                    other,
                )
                if other not in resolved:
                    resolved.append(other)
            else:
                logger.warning(
                    "train_modules: '%s' dropped; model has no video_all projection "
                    "(has_video_all_condition=False).",
                    prefix,
                )
            continue
        if prefix in _OPTIONAL_MODULES:
            logger.warning(
                "train_modules: optional module '%s' not present in the built model "
                "(e.g. action_conditioned=False); dropping it.",
                prefix,
            )
            continue
        missing.append(prefix)

    if missing:
        raise ValueError(
            "`lora.train_modules` names submodules that do not exist in the built "
            f"video-expert: {missing}. They would be trained on nothing and omitted "
            f"from the checkpoint. Existing top-level submodules: {sorted(existing)}."
        )
    return resolved


def is_lora_trainable_param(name: str, train_prefixes: Iterable[str]) -> bool:
    """True for LoRA adapter params or params of top-level submodules listed in
    `train_prefixes` (e.g. `video_all_qformer`) that stay fully trainable.
    """
    if (
        ".lora_A" in name
        or ".lora_B" in name
        or name.startswith("lora_A")
        or name.startswith("lora_B")
    ):
        return True
    for prefix in train_prefixes:
        if name == prefix or name.startswith(prefix + "."):
            return True
    return False


def set_lora_trainable(video_expert: nn.Module, train_prefixes: Iterable[str]) -> None:
    """Freeze all base params of `video_expert`, unfreeze only LoRA adapters and
    the fully-trainable `train_prefixes` submodules.
    """
    train_prefixes = list(train_prefixes)
    for name, param in video_expert.named_parameters():
        param.requires_grad = is_lora_trainable_param(name, train_prefixes)


def lora_trainable_state_dict(
    video_expert: nn.Module, train_prefixes: Iterable[str]
) -> dict:
    """State-dict subset holding LoRA adapters + `train_prefixes` submodules."""
    train_prefixes = list(train_prefixes)
    return {
        name: value
        for name, value in video_expert.state_dict().items()
        if is_lora_trainable_param(name, train_prefixes)
    }


def load_state_dict_lora_aware(module: nn.Module, state_dict: dict):
    """Load `state_dict` into `module` (strict=False), remapping legacy linear
    keys `...q.weight` -> `...q.base.weight` where the target linear is now a
    wrapped `LoRALinear`. This lets a full (non-LoRA) DiT checkpoint initialize a
    LoRA-wrapped backbone; LoRA/`train_modules` keys already match and load directly.
    """
    model_keys = set(module.state_dict().keys())
    remapped = {}
    for key, value in state_dict.items():
        if key in model_keys:
            remapped[key] = value
            continue
        head, sep, tail = key.rpartition(".")
        if sep:
            candidate = f"{head}.base.{tail}"
            if candidate in model_keys:
                remapped[candidate] = value
                continue
        # Older checkpoints also saved disconnected representation-learning heads.
        # They are never read by the query-only inference used in this release.
        legacy_prefixes = (
            "video_all_qformer.cls_token",
            "video_all_qformer.temp",
            "video_all_qformer.video_text_embedding.",
            "video_all_qformer.vision_proj.",
            "video_all_qformer.text_proj.",
            "video_all_qformer.itm_head.",
            "video_all_first_frame_embedding.",
        )
        if key.startswith(legacy_prefixes):
            logger.warning(
                "Excluding legacy auxiliary tensor unused by query-only inference: %s",
                key,
            )
            continue
        raise ValueError(f"Checkpoint contains an unsupported tensor: {key}")
    return module.load_state_dict(remapped, strict=False)


@torch.no_grad()
def merge_lora_state_dict_into_module(
    module: nn.Module,
    lora_state_dict: dict,
    lora_config: dict | None,
) -> dict:
    """Merge a LoRA checkpoint into a plain module in place.

Load pretrained backbone weights first. Merge adapters as
W = W_base + (alpha / rank) * (B @ A), then copy fully trained parameters.
For legacy wrapped submodules, use checkpoint base.weight/base.bias instead
of pretrained weights. The target module must contain plain nn.Linear layers.
Return {"merged": adapter_count, "copied": passthrough_parameter_count}.
"""
    if lora_config is None:
        raise ValueError(
            "Merging a LoRA checkpoint into a non-LoRA model requires the "
            "checkpoint's `lora_config` (for the alpha/rank scaling factor)."
        )
    rank = int(lora_config["rank"])
    alpha = float(lora_config["alpha"])
    if rank <= 0:
        raise ValueError(f"LoRA rank must be > 0, got {rank}")
    scaling = alpha / rank

    # Separate adapters, legacy trained base weights, and plain trained parameters.
    adapters: dict[str, dict[str, torch.Tensor]] = {}
    wrapped_base: dict[str, dict[str, torch.Tensor]] = {}
    passthrough: dict[str, torch.Tensor] = {}
    for key, value in lora_state_dict.items():
        if key.endswith(".lora_A"):
            adapters.setdefault(key[: -len(".lora_A")], {})["A"] = value
        elif key.endswith(".lora_B"):
            adapters.setdefault(key[: -len(".lora_B")], {})["B"] = value
        elif key.endswith(".base.weight"):
            wrapped_base.setdefault(key[: -len(".base.weight")], {})["weight"] = value
        elif key.endswith(".base.bias"):
            wrapped_base.setdefault(key[: -len(".base.bias")], {})["bias"] = value
        else:
            passthrough[key] = value

    model_state = module.state_dict()
    merged: dict[str, torch.Tensor] = {}
    for prefix, ab in adapters.items():
        if "A" not in ab or "B" not in ab:
            raise ValueError(f"LoRA adapter '{prefix}' is missing its A or B tensor.")
        weight_key = f"{prefix}.weight"
        if weight_key not in model_state:
            raise ValueError(
                f"LoRA adapter '{prefix}' has no matching base linear weight "
                f"'{weight_key}' in the target module; cannot merge."
            )
        ref = model_state[weight_key]
        # A wrapped train_module linear carries its trained base in the checkpoint;
        # a frozen backbone linear does not, so reconstruct from the target's
        # already-loaded pretrained weight.
        base_w = wrapped_base.get(prefix, {}).get("weight", ref)
        base_w = base_w.to(device=ref.device, dtype=torch.float32)
        lora_a = ab["A"].to(device=ref.device, dtype=torch.float32)
        lora_b = ab["B"].to(device=ref.device, dtype=torch.float32)
        delta = scaling * (lora_b @ lora_a)
        if tuple(delta.shape) != tuple(ref.shape):
            raise ValueError(
                f"LoRA delta shape {tuple(delta.shape)} does not match base weight "
                f"{tuple(ref.shape)} for '{weight_key}'."
            )
        merged[weight_key] = (base_w + delta).to(dtype=ref.dtype)

    # Fold the remaining wrapped-base tensors (biases, plus any base.weight whose
    # adapter was absent) directly onto the corresponding plain linear params.
    for prefix, wb in wrapped_base.items():
        if "bias" in wb:
            bias_key = f"{prefix}.bias"
            ref = model_state.get(bias_key)
            merged[bias_key] = (
                wb["bias"]
                if ref is None
                else wb["bias"].to(device=ref.device, dtype=ref.dtype)
            )
        if "weight" in wb and prefix not in adapters:
            weight_key = f"{prefix}.weight"
            ref = model_state.get(weight_key)
            merged[weight_key] = (
                wb["weight"]
                if ref is None
                else wb["weight"].to(device=ref.device, dtype=ref.dtype)
            )
    stray = [k for k in merged if k not in model_state]
    if stray:
        raise ValueError(
            "LoRA checkpoint has wrapped-linear base params with no counterpart in "
            f"the target module: {stray[:8]}{'...' if len(stray) > 8 else ''}."
        )

    new_state = dict(model_state)
    unexpected = [k for k in passthrough if k not in model_state]
    if unexpected:
        raise ValueError(
            "LoRA checkpoint contains fully-trainable params with no counterpart in "
            f"the target module: {unexpected[:8]}{'...' if len(unexpected) > 8 else ''}."
        )
    new_state.update(merged)
    for key, value in passthrough.items():
        ref = model_state[key]
        new_state[key] = value.to(device=ref.device, dtype=ref.dtype)
    module.load_state_dict(new_state, strict=True)

    logger.info(
        "Merged LoRA checkpoint into base weights: %d adapters folded (scaling=%s), "
        "%d fully-trainable params copied.",
        len(merged),
        scaling,
        len(passthrough),
    )
    return {"merged": len(merged), "copied": len(passthrough)}
