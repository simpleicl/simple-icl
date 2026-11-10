"""Validate fixed Aloha offsets used by exported evaluation scenes."""

import math


def recorded_offsets(environment, *, task_name, embodiment, seed, oids):
    replay = (environment or {}).get("recorded_replay")
    if replay is None:
        return None
    expected = (replay["task"], replay["embodiment"], replay["placement_seed"])
    if (task_name, embodiment, seed) != expected:
        raise ValueError(f"Recorded scene identity mismatch: {expected}")
    offsets = replay["offsets"]
    if set(offsets) != set(oids):
        raise ValueError("Recorded scene object IDs do not match the manifest")
    result = []
    for oid in oids:
        row = offsets[oid]
        if len(row) != 3 or not all(math.isfinite(float(v)) for v in row):
            raise ValueError(f"Invalid recorded offset for {oid}")
        result.append(tuple(float(v) for v in row))
    return result
