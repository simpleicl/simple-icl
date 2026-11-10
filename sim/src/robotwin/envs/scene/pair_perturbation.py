"""Deterministic placement perturbations for ALOHA demonstration/subject pairs.

The demonstration stays unchanged; numbered subject variants use a seed derived
from the placement seed, embodiment name, and variant number. This module is independent of SAPIEN;
_base_task._apply_pair_perturbation applies the sampled poses.
"""

from __future__ import annotations

import math
import zlib

import numpy as np
import transforms3d as t3d

PAIR_PERTURBATION_DEFAULTS = {
    "translation": 0.02,  # Uniform xy offset range per axis (meters).
    "yaw_deg": 15.0,  # Uniform yaw range about world z (degrees).
    "max_retries": 10,  # Maximum collision-avoidance retries per object.
    "clearance": 0.0,  # Extra separation in dist > r_i + r_j + clearance (meters).
}

# Each subject is recorded separately, including when K=1.
PAIR_PERTURBATION_MULTI_DEFAULTS = {
    "num_variants": 1,  # K: subject variants per ALOHA placement.
    "min_variants": 1,  # m: minimum successful variants required to retain a pair.
    "fallback_zero_translation": True,  # After retries, use zero translation if True; otherwise fail the variant.
}


class PerturbationInfeasible(Exception):
    """Collision retries exhausted with zero-translation fallback disabled."""


def normalize_pair_perturbation(
    raw: dict | None, baseline_embodiment: str
) -> dict | None:
    """Fill defaults and attach the baseline name; return None when disabled."""
    if not raw or not raw.get("enabled"):
        return None
    cfg = dict(PAIR_PERTURBATION_DEFAULTS)
    for key in PAIR_PERTURBATION_DEFAULTS:
        if raw.get(key) is not None:
            cfg[key] = int(raw[key]) if key == "max_retries" else float(raw[key])
    for key, default in PAIR_PERTURBATION_MULTI_DEFAULTS.items():
        val = raw.get(key)
        cfg[key] = (
            default
            if val is None
            else (bool(val) if isinstance(default, bool) else int(val))
        )
    if cfg["num_variants"] < 1 or not (1 <= cfg["min_variants"] <= cfg["num_variants"]):
        raise ValueError(
            f"Invalid pair_perturbation: num_variants={cfg['num_variants']} "
            f"min_variants={cfg['min_variants']} (requires 1 ≤ m ≤ K)"
        )
    cfg["enabled"] = True
    cfg["baseline_embodiment"] = baseline_embodiment
    cfg["variant_directories"] = True
    return cfg


def merge_pair_perturbation_override(
    raw: dict | None, override: dict | None
) -> dict | None:
    """Apply plan-level K/m overrides while retaining the scene perturbation."""
    if not override:
        return raw
    allowed = {"num_variants", "min_variants"}
    unknown = sorted(set(override) - allowed)
    if unknown:
        raise ValueError(f"Unsupported fields in pair_perturbation override: {unknown}")
    if not raw or not raw.get("enabled"):
        raise ValueError("pair_perturbation override requires perturbation to be enabled in the scene config")
    merged = dict(raw)
    for key in allowed:
        if key in override:
            merged[key] = int(override[key])
    return merged


def perturbation_seed(
    placement_seed: int, embodiment_name: str, variant: int | None = None
) -> int:
    """Derive a stable perturbation seed across processes.

    A salt separates this seed from env_seed. variant=None preserves legacy
    seeds; multi-variant mode adds a |v<k> suffix.
    """
    key = f"pair_perturbation|{placement_seed}|{embodiment_name}"
    if variant is not None:
        key += f"|v{int(variant)}"
    return zlib.crc32(key.encode()) & 0xFFFFFFFF


def yaw_premultiply(q: list, dyaw: float) -> list:
    """Pre-multiply a wxyz quaternion by world-z yaw, preserving its tilt."""
    dq = t3d.quaternions.axangle2quat([0, 0, 1], dyaw)
    return [float(x) for x in t3d.quaternions.qmult(dq, q)]


def sample_pair_offsets(
    placement_seed: int,
    embodiment_name: str,
    cfg: dict | None,
    items: list,
    variant: int | None = None,
) -> list:
    """Return (dx, dy, dyaw) offsets for (x, y, radius) items sorted by oid.

    Retry translations up to max_retries, requiring separation greater than
    r_i + r_j + clearance. Earlier objects use perturbed positions; later ones
    use baseline positions. On exhaustion, use zero translation if enabled
    (assuming valid baseline separation), or raise PerturbationInfeasible.
    """
    n = len(items)
    if not cfg or variant is None:
        return [(0.0, 0.0, 0.0)] * n

    rng = np.random.RandomState(
        perturbation_seed(placement_seed, embodiment_name, variant)
    )
    t = float(cfg["translation"])
    yaw = math.radians(float(cfg["yaw_deg"]))
    clearance = float(cfg["clearance"])
    max_retries = int(cfg["max_retries"])
    fallback_zero = bool(cfg.get("fallback_zero_translation", True))

    xy = [[x, y] for x, y, _r in items]
    out = []
    for i in range(n):
        dyaw = float(rng.uniform(-yaw, yaw)) if yaw > 0 else 0.0
        dx = dy = None
        for _ in range(max_retries):
            cand = rng.uniform(-t, t, size=2) if t > 0 else np.zeros(2)
            nx, ny = items[i][0] + cand[0], items[i][1] + cand[1]
            if all(
                np.hypot(nx - xy[j][0], ny - xy[j][1])
                > items[i][2] + items[j][2] + clearance
                for j in range(n)
                if j != i
            ):
                dx, dy = float(cand[0]), float(cand[1])
                break
        if dx is None:
            if not fallback_zero:
                raise PerturbationInfeasible(
                    f"pair perturbation infeasible: seed={placement_seed} "
                    f"variant={variant} object_index={i} after retries"
                )
            dx = dy = 0.0
        xy[i] = [items[i][0] + dx, items[i][1] + dy]
        out.append((dx, dy, dyaw))
    return out
