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
        self.lora_A = nn.Parameter(torch.empty(self.rank, self.in_features, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.rank, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.lora_dropout = nn.Dropout(dropout) if dropout and dropout > 0.0 else nn.Identity()

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
    `video_ctx_qformer`) is skipped, so fully-trainable submodules are never
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
                setattr(module, child_name, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
                replaced.append(full_name)
    return replaced


# Conditioning submodules whose presence depends on the model config:
# `video_ctx_qformer` exists only when has_video_ctx_condition=True,
# `action_embedding` only when action_conditioned=True.
_OPTIONAL_MODULES = {"action_embedding", "video_ctx_qformer"}


def resolve_train_modules(root: nn.Module, train_prefixes: Iterable[str]) -> list:
    """Reconcile configured `train_modules` with the actually-built model, failing
    fast (at construction time) on names that would otherwise be silently dropped.

    A `train_modules` entry matching no parameter is trained on nothing AND excluded
    from the checkpoint by `lora_trainable_state_dict`, so a config/mode mismatch
    silently produces a checkpoint missing that freshly-initialized module. We catch
    it here instead of at save/eval time:

    * an absent optional module (`action_embedding`, `video_ctx_qformer`) is
      dropped with a warning.
    * any other name matching no parameter raises `ValueError`.

    Returns the resolved list; every entry is guaranteed to match real parameters.
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
        if prefix in _OPTIONAL_MODULES:
            logger.warning(
                "train_modules: optional module '%s' not present in the built model "
                "(disabled by the model config); dropping it.", prefix,
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
    """True for LoRA adapter params, or params of top-level submodules listed in
    `train_prefixes` (e.g. `video_ctx_qformer`) that stay fully trainable.
    """
    if ".lora_A" in name or ".lora_B" in name or name.startswith("lora_A") or name.startswith("lora_B"):
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


def lora_trainable_state_dict(video_expert: nn.Module, train_prefixes: Iterable[str]) -> dict:
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
        remapped[key] = value  # leave as-is; strict=False will drop unknowns
    return module.load_state_dict(remapped, strict=False)


@torch.no_grad()
def merge_lora_state_dict_into_module(
    module: nn.Module,
    lora_state_dict: dict,
    lora_config: dict | None,
) -> dict:
    """Fold a LoRA checkpoint into a plain (non-LoRA) `module`, in place.

    A LoRA payload (see `lora_trainable_state_dict`) stores only the low-rank
    adapters (`<path>.lora_A`, `<path>.lora_B`) plus the params of any
    fully-trainable submodules (`train_modules`, e.g. `video_ctx_qformer.*`). The
    frozen backbone is *not* stored -- it is reconstructed from the pretrained
    weights already loaded into `module`.

    This merges each adapter into its base linear weight,
        W_merged = W_base + (alpha / rank) * (B @ A),
    matching `LoRALinear.forward`'s `base(x) + scaling * x @ A^T @ B^T`, then
    copies the fully-trainable submodule params directly. It lets a checkpoint
    trained with `use_lora=True` load into a model built with `use_lora=False`.

    For a frozen backbone adapter `W_base` is the target module's already-loaded
    pretrained weight. Legacy checkpoints (saved before `inject_lora` gained
    `exclude_prefixes`) also LoRA-wrapped their fully-trainable submodules, so the
    payload carries `<path>.base.weight`/`.base.bias` for those wrapped linears --
    there `W_base` comes from the checkpoint and is folded onto the plain
    `<path>.weight`/`<path>.bias` of the target.

    `module` must NOT itself be LoRA-wrapped (its target linears are plain
    `nn.Linear` with a `<path>.weight`, not `<path>.base.weight`).

    Returns a summary dict `{"merged": <#adapters>, "copied": <#passthrough params>}`.
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

    # Split the payload into three groups, keyed by the base-linear dotted path:
    #   * adapters      -- `<path>.lora_A` / `<path>.lora_B` low-rank tensors.
    #   * wrapped_base  -- `<path>.base.weight` / `<path>.base.bias`, present when a
    #     fully-trainable submodule (a `train_modules` prefix) had its internal
    #     linears LoRA-wrapped AND its base was trained (legacy checkpoints saved
    #     before `inject_lora(exclude_prefixes=...)`). Here the trained base lives in
    #     the checkpoint, not the target module.
    #   * passthrough   -- plain fully-trainable params copied verbatim.
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
            merged[bias_key] = wb["bias"] if ref is None else wb["bias"].to(device=ref.device, dtype=ref.dtype)
        if "weight" in wb and prefix not in adapters:
            weight_key = f"{prefix}.weight"
            ref = model_state.get(weight_key)
            merged[weight_key] = wb["weight"] if ref is None else wb["weight"].to(device=ref.device, dtype=ref.dtype)
    stray = [k for k in merged if k not in model_state]
    if stray:
        raise ValueError(
            "LoRA checkpoint has wrapped-linear base params with no counterpart in "
            f"the target module: {stray[:8]}{'...' if len(stray) > 8 else ''}."
        )

    # Overlay merged base weights + passthrough params onto the current state and
    # load strictly, so any stray key (e.g. a `train_modules` submodule absent
    # from the built model) surfaces instead of being silently dropped.
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
        len(merged), scaling, len(passthrough),
    )
    return {"merged": len(merged), "copied": len(passthrough)}
