"""ALOHA placement sampling and quota selection."""

import sys

sys.path.append("./")

from script.scene_driver import (
    class_decorator, build_args, is_scene_ready, embodiment_name_of,
)
from envs.scene import get_requirement, match_task, match_from_binding


def parse_quota(spec: str) -> dict:
    """Parse k:n quotas: at least n placements supporting at least k tasks."""
    quota = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        k, n = part.split(":")
        quota[int(k)] = int(n)
    return quota


def prepare_embodiment(scene_set, task_names, scene_config, embodiment,
                       bindings_by_task=None):
    """Build task environments, arguments, and static bindings for one embodiment."""
    emb_name = embodiment_name_of(embodiment)
    tasks, envs, args_by_task, matches_by_task = [], {}, {}, {}
    for tn in task_names:
        if not is_scene_ready(tn):
            print(f"[sample] Skipping {tn}: bind_actors is not implemented (recording unavailable in scene mode)")
            continue
        req = get_requirement(tn)
        persisted = (bindings_by_task or {}).get(tn)
        if persisted is not None:
            match = match_from_binding(req, scene_set, persisted)
            if match is None:
                print(f"[sample] Skipping {tn}: invalid scene binding")
                continue
        else:
            match = match_task(req, scene_set)
        if match is None:
            print(f"[sample] Skipping {tn}: static object set mismatch")
            continue
        tasks.append(tn)
        envs[tn] = class_decorator(tn)
        args_by_task[tn] = build_args(tn, scene_config, embodiment_override=embodiment)
        matches_by_task[tn] = match
    return tasks, emb_name, envs, args_by_task, matches_by_task


def quota_helps(remaining: dict, m: int) -> list:
    """Return quota buckets that a placement supporting m tasks can fill."""
    return [k for k in remaining if k <= m and remaining[k] > 0]


def select_by_quota(accepted: dict, quota: dict, required_tasks=None) -> list:
    """Fill quotas from successful paired placements in ascending seed order."""
    required = set(required_tasks or ())
    remaining = dict(quota)
    selected = []
    for seed in sorted(accepted.keys()):
        tasks = accepted[seed]
        if not required.issubset(tasks):
            continue
        m = len(tasks)
        if m < 1:
            continue
        helps = quota_helps(remaining, m)
        if helps:
            selected.append((seed, sorted(tasks)))
            for k in helps:
                remaining[k] -= 1
        if not any(v > 0 for v in remaining.values()):
            break
    unmet = {k: v for k, v in remaining.items() if v > 0}
    if unmet:
        print(f"[sample] Quotas unmet after pairing: {unmet} (task success rates after perturbation affect the number of pairs; "
              f"increase --max_search or relax quota)")
    return selected


class OnlineQuotaSelector:
    """Apply placement quotas to asynchronous worker results in seed order."""

    def __init__(self, n_arms: int, quota: dict, seed_start: int = 0,
                 required_tasks=None):
        self.n_arms = n_arms
        self.remaining = dict(quota)
        self.required_tasks = set(required_tasks or ())
        self.selected = []
        self._pending = {}
        self._next = seed_start

    @property
    def met(self) -> bool:
        """Return whether every quota has been filled."""
        return not any(v > 0 for v in self.remaining.values())

    def report(self, idx: int, seed: int, success) -> list:
        """Accept a worker result and return newly selected placements."""
        self._pending.setdefault(seed, {})[idx] = set(success)
        newly = []
        while self._next in self._pending and len(self._pending[self._next]) == self.n_arms:
            succ = self._pending.pop(self._next)
            common = set.intersection(*succ.values())
            if not self.required_tasks.issubset(common):
                self._next += 1
                continue
            helps = quota_helps(self.remaining, len(common))
            if helps:
                item = (self._next, sorted(common))
                self.selected.append(item)
                newly.append(item)
                for k in helps:
                    self.remaining[k] -= 1
            self._next += 1
            if self.met:
                break
        return newly
