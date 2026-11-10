"""Task-driven scene collection: freeze a manifest, find placements, and record pairs.

Output layout:
  data/<scene_config>/<scene_id>/placement<pid>/<task>/<embodiment>/{data,video,instructions,...}

Use --tasks to select tasks, --quota to set placement counts, and --visual
with --visual_variants to record appearance variants of each trajectory.
"""

import sys

sys.path.append("./")

import os
import json
import shutil
import argparse
import hashlib
import re
import zlib
from copy import deepcopy
from datetime import datetime
from collections import defaultdict

import numpy as np
import yaml

from script.scene_driver import (
    build_args, DR_ALL_OFF, list_embodiments, embodiment_name_of,
    run_one, teardown_env, binding_to_dict, resolve_pair_perturbation,
)
from script.placement_sampler import (
    parse_quota, prepare_embodiment, select_by_quota, OnlineQuotaSelector,
    quota_helps,
)
from envs.scene import (
    SceneSpec, candidate_tasks, get_requirement, match_task,
    mixed_scene_for_tasks, sample_environment, task_table_z_baseline,
)

# Instruction helpers resolve paths independently of the working directory.
sys.path.append("./description/utils")
from generate_episode_instructions import (  # noqa: E402
    generate_episode_descriptions,
    extract_episodes_from_scene_info,
)


# Configuration
EMPTY_RESULT_FILE = "empty_result.json"
_PERT_DIR_RE = re.compile(r"^(.+)__pert(\d+)$")


def base_emb_name(dir_name: str) -> str:
    """Strip the __pert<k> suffix to recover the embodiment name."""
    m = _PERT_DIR_RE.match(dir_name)
    return m.group(1) if m else dir_name


def perturb_variant_plan(pair_perturbation: dict | None) -> tuple[int, int]:
    """Return variant count and minimum successes (K, m); default to (1, 1)."""
    if not pair_perturbation:
        return 1, 1
    return (int(pair_perturbation.get("num_variants", 1)),
            int(pair_perturbation.get("min_variants", 1)))


def _note_search_attempt(stats: dict, task_name: str, env, ok: bool) -> None:
    task_stats = stats.setdefault(task_name, defaultdict(int))
    status = getattr(env, "_last_run_status", "success" if ok else "unknown_failure")
    task_stats["attempts"] += 1
    task_stats[status] += 1


def _note_search_task_result(stats: dict, task_name: str, success: bool,
                             multi_variant: bool) -> None:
    task_stats = stats.setdefault(task_name, defaultdict(int))
    if success:
        task_stats["task_success"] += 1
    elif multi_variant:
        task_stats["variant_threshold_failure"] += 1


def _search_stats_payload(stats: dict) -> dict:
    return {
        task_name: dict(sorted(counts.items()))
        for task_name, counts in sorted(stats.items())
    }


def _validate_ready_tasks(ready: dict, expected_tasks: list[str], names: dict) -> None:
    issues = []
    for idx, response in sorted(ready.items()):
        prepared = set(response.get("tasks") or [])
        missing = [task for task in expected_tasks if task not in prepared]
        if missing:
            issues.append(f"worker#{idx} {names[idx]} missing tasks {missing}")
    if issues:
        raise SystemExit("[parallel] Worker task preparation failed; refusing to mark the search empty: " + "; ".join(issues))


def _print_search_stats(stats_by_worker: dict, names: dict) -> None:
    for idx in sorted(stats_by_worker):
        for task_name, counts in sorted(stats_by_worker[idx].items()):
            preferred = ("attempts", "task_success", "success",
                         "variant_threshold_failure")
            keys = [key for key in preferred if key in counts]
            keys += sorted(key for key in counts if key not in preferred)
            detail = " ".join(f"{key}={counts[key]}" for key in keys)
            print(f"[search] worker#{idx} {names[idx]} {task_name}: {detail}")


def _task_table_z_baseline(task_names: list[str]) -> float:
    """Reject task groups that require incompatible fixed table-height baselines."""
    baselines = {task_table_z_baseline(task_name) for task_name in task_names}
    if len(baselines) != 1:
        raise ValueError(
            "Tasks require conflicting table-height baselines; split them into separate scene groups: "
            + ", ".join(f"{task_name}={task_table_z_baseline(task_name):.3f}"
                        for task_name in task_names))
    return baselines.pop()


def _load_generation_cfg(scene_config: str) -> dict:
    """Load collection-time manifest environment sampling options."""
    path = os.path.join("scene_config", f"{scene_config}.yml")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader) or {}
    return cfg.get("generation", {}) or {}


def load_pair_perturbation_cfg(scene_config: str, override=None) -> dict | None:
    """Load paired placement perturbations; return None when disabled."""
    from envs.scene.pair_perturbation import merge_pair_perturbation_override

    path = os.path.join("scene_config", f"{scene_config}.yml")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader) or {}
    cfg["pair_perturbation"] = merge_pair_perturbation_override(
        cfg.get("pair_perturbation"), override)
    return resolve_pair_perturbation(cfg)


def load_collection_cfg(scene_config: str) -> dict:
    """Load visual randomization settings, disabling clutter that breaks replay."""
    path = os.path.join("scene_config", f"{scene_config}.yml")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader) or {}
    dr = cfg.get("collection", {}) or {}
    dr = dict(dr)
    dr["cluttered_table"] = False
    return dr


def teardown_all_envs(envs: dict) -> None:
    """Release all robot GPU resources before switching embodiments to avoid CUDA errors."""
    for env in envs.values():
        teardown_env(env)


def _file_sha256(path: str) -> str:
    """Return a content hash used to bind a completion record to its inputs."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalise_exclude_tasks(exclude_tasks) -> list[str]:
    """Normalize task exclusions to a deterministic, duplicate-free list."""
    if exclude_tasks is None:
        return []
    if isinstance(exclude_tasks, str):
        values = exclude_tasks.split(",")
    else:
        values = exclude_tasks
    return sorted({str(task).strip() for task in values if str(task).strip()})


def _normalise_required_tasks(required_tasks) -> list[str]:
    """Normalize tasks that must succeed in every selected placement."""
    if required_tasks is None:
        return []
    if isinstance(required_tasks, str):
        values = required_tasks.split(",")
    else:
        values = required_tasks
    return list(dict.fromkeys(str(task).strip() for task in values if str(task).strip()))


def collection_request(task_names: list[str], scene_config: str, group_seed: int,
                       emb_names: list[str], exclude_tasks=None,
                       required_tasks=None, pair_perturbation=None) -> dict:
    """Return the immutable task-driven manifest request fingerprint."""
    config_path = os.path.join("scene_config", f"{scene_config}.yml")
    request = {
        "source": "scene_mode",
        "tasks": list(task_names),
        "task_selection": "all_compatible",
        "scene_config": scene_config,
        "scene_config_sha256": _file_sha256(config_path),
        "group_seed": group_seed,
        "embodiments": list(emb_names),
    }
    excluded = _normalise_exclude_tasks(exclude_tasks)
    if excluded:
        request["exclude_tasks"] = excluded
    required = _normalise_required_tasks(required_tasks)
    if required:
        request["required_tasks"] = required
    if pair_perturbation is not None:
        request["pair_perturbation"] = pair_perturbation
    return request


def derive_task_manifest(task_spec: str, scene_id: str | None, group_seed: int,
                         scene_config: str, embodiments: list,
                         exclude_tasks=None, required_tasks=None,
                         pair_perturbation_override=None) -> tuple[SceneSpec, list[str]]:
    """Derive one immutable SceneSpec from requested tasks without simulation."""
    raw_tasks = [name.strip() for name in task_spec.split(",") if name.strip()]
    if not raw_tasks:
        raise SystemExit("[collect] --tasks must not be empty")

    valid_tasks = []
    seen = set()
    unknown = []
    for task_name in raw_tasks:
        if get_requirement(task_name) is None:
            unknown.append(task_name)
        elif task_name not in seen:
            seen.add(task_name)
            valid_tasks.append(task_name)
    if unknown:
        raise SystemExit("[collect] Unknown tasks: " + ", ".join(sorted(set(unknown))))

    required = _normalise_required_tasks(required_tasks)
    missing_required = sorted(set(required) - set(valid_tasks))
    if missing_required:
        raise SystemExit(
            "[collect] required_tasks must be drawn from --tasks: " + ", ".join(missing_required))

    table_baseline = _task_table_z_baseline(valid_tasks)
    scene_set, sid, skipped = mixed_scene_for_tasks(
        valid_tasks, scene_id=scene_id, seed=group_seed)
    if skipped:
        raise SystemExit("[collect] Cannot infer object requirements for tasks: " + ", ".join(t for t, _ in skipped))
    if not scene_set.objects:
        raise SystemExit("[collect] The inferred object set is empty")

    for task_name in valid_tasks:
        match = match_task(get_requirement(task_name), scene_set)
        if match is None:
            raise SystemExit(f"[collect] Static matching failed: {task_name}")

    excluded = set(_normalise_exclude_tasks(exclude_tasks))
    compatible = [match for match in candidate_tasks(scene_set)
                  if match.task_name not in excluded]

    env_seed = group_seed + (zlib.crc32(sid.encode()) & 0xFFFF)
    environment = sample_environment(
        _load_generation_cfg(scene_config), seed=env_seed, table_z_bias=table_baseline)
    spec = SceneSpec.from_object_set(sid, scene_set, environment=environment)
    emb_names = [embodiment_name_of(e) for e in embodiments]
    pair_perturbation = (load_pair_perturbation_cfg(
        scene_config, pair_perturbation_override)
        if pair_perturbation_override else None)
    spec.request = collection_request(
        valid_tasks, scene_config, group_seed, emb_names, excluded, required,
        pair_perturbation)
    spec.request["environment_seed"] = env_seed
    spec.supported_tasks = [
        {"task_name": match.task_name, "binding": binding_to_dict(match),
         "status": "static_ready"}
        for match in compatible
    ]
    return spec, valid_tasks


def write_manifest(scene_dir: str, spec: SceneSpec) -> str:
    """Atomically persist the exact collection input before dynamic search begins."""
    path = os.path.join(scene_dir, "manifest.json")
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(spec.to_dict(), f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)
    return path


def _remove_episode_artifacts(save_dir: str, episode: int) -> None:
    """Remove all persisted files for one replay that did not reproduce."""
    for subdir, suffix in (("data", ".hdf5"), ("video", ".mp4"),
                           ("instructions", ".json")):
        path = os.path.join(save_dir, subdir, f"episode{episode}{suffix}")
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


def _discard_failed_replay(env, save_dir: str, episode: int, clear_cache: bool) -> None:
    """Clean up a failed replay without terminating the worker."""
    try:
        env.close_env(clear_cache=clear_cache)
    except Exception:
        pass
    try:
        env.remove_data_cache()
    except Exception:
        pass
    _remove_episode_artifacts(save_dir, episode)


def prepare_collection_dir(scene_dir: str, reset: bool = False) -> None:
    """Clear an interrupted collection while protecting a completed dataset.

    Collection is restart-from-scratch rather than resume-in-place: the selected
    placements depend on the full paired search, so retaining an incomplete run
    would mix old and new quota decisions.
    """
    completion_path = os.path.join(scene_dir, "completion.json")
    if os.path.exists(completion_path) and not reset:
        raise SystemExit(
            f"[collect] A completed dataset already exists: {completion_path}; pass --reset to recollect")

    os.makedirs(scene_dir, exist_ok=True)
    for name in os.listdir(scene_dir):
        path = os.path.join(scene_dir, name)
        if (name.startswith("_seed") or re.fullmatch(r"placement\d+", name)
                or name in {"manifest.json", "manifest.json.tmp", "summary.md",
                            "completion.json", "completion.json.tmp",
                            EMPTY_RESULT_FILE, f"{EMPTY_RESULT_FILE}.tmp"}):
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            else:
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass


def visual_seed_for(placement_seed, variant):
    """Preserve legacy visual seeds while supporting large placement IDs."""
    return (int(placement_seed) * 1000 + int(variant)) % (2**32)


# Record one (placement, task, embodiment)
def record_task(env, save_dir, task_name, placement_seed, traj, snapshot, scene_set, match,
                environment, scene_config, embodiment, domain_randomization, visual_variants,
                use_visual_rng, clear_cache_freq=5, perturb_variant=None):
    """Replay a sampled trajectory with visual variants; return recorded episode count.

    Warm up this same env with run_one immediately before recording: a fresh
    recording setup can crash when mplib scans scaled objects.
    embodiment is [emb] or [left, right, separation], passed to build_args.
    Restore only the pre-bind RNG state from snapshot; rebuild and fully settle
    the scene with the original seed. Each variant uses the same placement and
    trajectory, with an independent visual RNG when use_visual_rng is enabled.
    """
    args = build_args(task_name, scene_config, domain_randomization=domain_randomization,
                      embodiment_override=embodiment, perturb_variant=perturb_variant)
    args["save_path"] = save_dir
    args["need_plan"] = False
    args["save_data"] = True
    args["render_freq"] = 0
    # Settle fully, but skip the repeated stability rejection.
    args["ignore_stability_failure"] = True
    os.makedirs(save_dir, exist_ok=True)

    info_path = os.path.join(save_dir, "scene_info.json")
    info_db = {}

    for v in range(visual_variants):
        visual_seed = visual_seed_for(placement_seed, v)
        env.scene_spec = scene_set
        env.scene_binding = match
        env.scene_environment = environment
        env.scene_task_rng_state = snapshot.get("task_rng_state") if isinstance(snapshot, dict) else None
        # A separate visual RNG preserves placement; None uses the global RNG.
        env.scene_visual_rng = np.random.RandomState(visual_seed) if use_visual_rng else None

        # Inject cached trajectories; _init_task_env_ resets the counters.
        args["left_joint_path"] = deepcopy(traj["left_joint_path"])
        args["right_joint_path"] = deepcopy(traj["right_joint_path"])
        clear_cache = ((v + 1) % clear_cache_freq == 0)
        try:
            env.setup_demo(now_ep_num=v, seed=placement_seed, **args)
            env.set_path_lst(args)

            from script.scene_driver import snapshot_placement
            initial_scene = snapshot_placement(env, scene_set) if scene_set is not None else None
            info = env.play_once()
            if initial_scene is not None:
                info["initial_scene"] = initial_scene
            info["visual_seed"] = visual_seed
            info_db[f"episode_{v}"] = info

            replay_ok = bool(getattr(env, "plan_success", True) and env.check_success())
            env.close_env(clear_cache=clear_cache)
            if not replay_ok:
                print(f"    [record] {task_name} placement_seed={placement_seed} "
                      f"variant={v}: check_success=False (replay failed to reproduce success; skipping)")
                info_db.pop(f"episode_{v}", None)
                env.remove_data_cache()
                _remove_episode_artifacts(save_dir, v)
                continue
            env.merge_pkl_to_hdf5_video()
            env.remove_data_cache()
        except IndexError as e:
            print(f"    [record] {task_name} placement_seed={placement_seed} "
                  f"variant={v}: {type(e).__name__}: {e} (skipping)")
            info_db.pop(f"episode_{v}", None)
            _discard_failed_replay(env, save_dir, v, clear_cache)

    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info_db, f, ensure_ascii=False, indent=2)

    _write_instructions(save_dir, task_name, info_db, args.get("language_num", 100))
    return len(info_db)


def _write_instructions(save_dir, task_name, info_db, language_num):
    """Write scene-based instructions to save_dir/instructions/episode{v}.json."""
    episodes = extract_episodes_from_scene_info(info_db)
    results = generate_episode_descriptions(task_name, episodes, language_num)
    out_dir = os.path.join(save_dir, "instructions")
    os.makedirs(out_dir, exist_ok=True)
    # episode_index follows info_db order and matches the episode variant index.
    keys = list(info_db.keys())
    for desc in results:
        i = desc["episode_index"]
        v = keys[i].split("_")[-1]
        with open(os.path.join(out_dir, f"episode{v}.json"), "w") as f:
            json.dump({"seen": desc.get("seen", []),
                       "unseen": desc.get("unseen", [])}, f, indent=2)


# Summary report
def _human_bytes(n: int) -> str:
    """Format a byte count in B, KB, MB, GB, or TB."""
    x = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if x < 1024:
            return f"{x:.0f} {unit}" if unit == "B" else f"{x:.1f} {unit}"
        x /= 1024
    return f"{x:.1f} TB"


def write_summary_report(scene_dir, spec, scene_set, task_names, quota, a,
                         placement_records, total_eps, visual_variants, emb_names,
                         pair_perturbation=None):
    """Write collection statistics to <scene_dir>/summary.md.

    placement_records contains pid, seed, and (task, {embodiment: episode_count})
    entries. emb_names sets embodiment order. Statistics use the frozen manifest,
    runtime counters, and HDF5 file sizes.
    """
    # Manifest only proves static role bindings; dynamic support is represented by recorded pairs.
    static_by_task = {t["task_name"]: t for t in spec.supported_tasks}

    # Aggregate task counts, merging perturbation variants under their base embodiment.
    task_eps = defaultdict(int)
    task_placements = defaultdict(int)
    task_eps_by_emb = defaultdict(lambda: defaultdict(int))  # {task: {emb: n}}
    emb_eps = defaultdict(int)                               # {emb: total}
    for rec in placement_records:
        for task_name, eps_by_emb in rec["tasks"]:
            n_task = sum(eps_by_emb.values())
            task_eps[task_name] += n_task
            task_placements[task_name] += 1
            for name, n in eps_by_emb.items():
                base = base_emb_name(name)
                task_eps_by_emb[task_name][base] += n
                emb_eps[base] += n

    # List candidate tasks first, then any extra recorded tasks.
    cols = list(task_names)
    for rec in placement_records:
        for tn, _ in rec["tasks"]:
            if tn not in cols:
                cols.append(tn)

    # Sum HDF5 sizes under scene_dir.
    n_hdf5, bytes_hdf5 = 0, 0
    for root, _, files in os.walk(scene_dir):
        for fn in files:
            if fn.endswith(".hdf5"):
                n_hdf5 += 1
                bytes_hdf5 += os.path.getsize(os.path.join(root, fn))

    # Each retained recording directory requests visual_variants episodes.
    n_pairs = sum(len(rec["tasks"]) for rec in placement_records)
    n_dirs = sum(len(eps_by_emb) for rec in placement_records
                 for _t, eps_by_emb in rec["tasks"])
    requested = n_dirs * visual_variants
    repro = (total_eps / requested) if requested else 0.0

    L = []  # Markdown lines
    L.append(f"# Collection Summary Report — {spec.scene_id}")
    L.append("")
    L.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    L.append("")

    # Scene overview
    L.append("## Scene Overview")
    L.append("")
    z = spec.environment.get("table_z_bias")
    L.append(f"- scene_id: `{spec.scene_id}`")
    L.append(f"- Object count: {len(scene_set.objects)}")
    L.append(f"- Robot: {', '.join(emb_names)}")
    L.append(f"- Table-height offset table_z_bias: {z:.4f}" if z is not None else "- Table-height offset table_z_bias: —")
    L.append("")
    L.append("| oid | modelname | kind | model_id |")
    L.append("| --- | --- | --- | --- |")
    for o in scene_set.objects:
        mid = getattr(o, "model_id", None)
        L.append(f"| `{o.oid}` | {o.modelname} | {o.kind} | {mid if mid is not None else 'random'} |")
    L.append("")

    # Collection settings
    L.append("## Collection Settings")
    L.append("")
    quota_str = ", ".join(f"{k}:{v}" for k, v in sorted(quota.items()))
    L.append(f"- scene_config: `{a.scene_config}`")
    L.append(f"- Candidate tasks: {', '.join(task_names)}")
    required = (spec.request or {}).get("required_tasks", [])
    if required:
        L.append(f"- Required tasks: {', '.join(required)}")
    L.append(f"- quota (minimum placement counts): {quota_str}")
    L.append(f"- max_search: {a.max_search}    seed_start: {a.seed_start}")
    L.append(f"- Visual randomization: {'enabled' if a.visual else 'disabled'}    Visual variants per (placement, task, embodiment): {visual_variants}")
    if pair_perturbation:
        K, m = perturb_variant_plan(pair_perturbation)
        extra = f", {K} perturbation variants per placement, at least {m} successful" if K > 1 else ""
        L.append(f"- ALOHA subject placement perturbation: ±{pair_perturbation['translation']:.3f} m / "
                 f"±{pair_perturbation['yaw_deg']:.0f}° (baseline "
                 f"{pair_perturbation['baseline_embodiment']} with no perturbation{extra})")
    L.append("")

    # Collection results
    L.append("## Collection Results")
    L.append("")
    L.append(f"- Sampled placements: {len(placement_records)}")
    L.append(f"- (placement, task) pairs: {n_pairs}")
    L.append(f"- Total episodes: **{total_eps}** (requested {requested}, reproduction rate {repro:.0%})")
    emb_eps_str = ", ".join(f"{name}: {emb_eps.get(name, 0)}" for name in emb_names)
    L.append(f"- ALOHA episodes: {emb_eps_str}")
    L.append(f"- Dataset size: {n_hdf5} HDF5 files, total {_human_bytes(bytes_hdf5)}")
    L.append("")

    # Per-task summary
    L.append("### Per-task Summary")
    L.append("")
    emb_hdr = " | ".join(f"{name} eps" for name in emb_names)
    L.append(f"| Task | Placements | {emb_hdr} | Total episodes | Static binding |")
    L.append("| --- | ---: | " + " | ".join("---:" for _ in emb_names) + " | ---: | --- |")
    for tn in cols:
        entry = static_by_task.get(tn, {})
        binding_state = "derived" if entry.get("status") == "static_ready" else "scene-compatible"
        emb_cells = " | ".join(str(task_eps_by_emb[tn].get(name, 0)) for name in emb_names)
        L.append(f"| {tn} | {task_placements.get(tn, 0)} | {emb_cells} | {task_eps.get(tn, 0)} | {binding_state} |")
    L.append("")

    # Placement-by-task episode counts
    L.append("### Placement × Task Matrix (episode counts)")
    L.append("")
    L.append("| placement | seed | " + " | ".join(cols) + " | Subtotal |")
    L.append("| --- | ---: | " + " | ".join("---:" for _ in cols) + " | ---: |")
    for rec in placement_records:
        by_task = {tn: sum(eps.values()) for tn, eps in rec["tasks"]}
        cells = [str(by_task.get(tn, 0)) if tn in by_task else "·" for tn in cols]
        subtotal = sum(by_task.values())
        L.append(f"| placement{rec['pid']} | {rec['seed']} | " + " | ".join(cells) + f" | {subtotal} |")
    # Totals
    total_cells = [str(task_eps.get(tn, 0)) for tn in cols]
    L.append("| **Total** | | " + " | ".join(f"**{c}**" for c in total_cells) + f" | **{total_eps}** |")
    L.append("")

    L.append("> Note: `·` indicates that the placement does not support the task; `0` indicates support but no successful replays.")
    L.append("> Strict pairing: for each accepted placement, all listed tasks were verified successfully on every embodiment.")
    L.append("")

    report_path = os.path.join(scene_dir, "summary.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    return report_path


# Embodiment order for progressive intersection
def finalize_layout(scene_dir, selected, emb_names, recorded, snapshots, environment,
                    visual_variants, pair_perturbation=None, required_tasks=None):
    """Finalize staged _seed<seed> recordings as placement<pid>/<task>/<emb>/.

    Remove unselected seeds, unpaired tasks, and incomplete variants. Multi-variant
    pairs require a complete baseline and at least m complete variants.
    Rename selected seeds in order and write shared placement_meta.json snapshots.
    Return (placement_records, total_eps) for the summary report.
    """
    num_variants, min_variants = perturb_variant_plan(pair_perturbation)
    required = set(required_tasks or ())
    baseline = (pair_perturbation or {}).get("baseline_embodiment")
    multi = bool((pair_perturbation or {}).get(
        "variant_directories", num_variants > 1))

    def pair_dirs(seed, task_name):
        """Return retained recording directories in order, or None for an incomplete pair."""
        names = []
        for emb_name in emb_names:
            if recorded .get ((seed ,task_name ,emb_name ),0 )!=visual_variants :
                return None
            names .append (emb_name )
            if multi:
                good = [f"{emb_name}__pert{k}" for k in range(num_variants)
                        if recorded.get((seed, task_name, f"{emb_name}__pert{k}"), 0)
                        == visual_variants]
                if len(good) < min_variants:
                    return None
                names.extend(good )
        return names

    # A selected task is valid only when every embodiment produced every visual
    # variant (multi mode: baseline complete + >= m complete perturbation dirs).
    # Delete incomplete baseline and subject pairs.
    # Retain only complete recordings.
    complete, dirs_by_pair = [], {}
    for seed, tasks in selected:
        keep = []
        for task_name in tasks:
            names = pair_dirs(seed, task_name)
            if names is None:
                print(f"[collect] Discarding incomplete pair seed={seed} task={task_name}")
            else:
                keep.append(task_name)
                dirs_by_pair[(seed, task_name)] = names
        if required.issubset(keep) and keep:
            complete.append((seed, keep))
        elif keep:
            missing = sorted(required - set(keep))
            print(f"[collect] Discarding placement with missing required tasks seed={seed} missing={missing}")
    selected = complete

    selected_seeds = {seed for seed, _ in selected}
    # Remove staging directories for unselected placements.
    if os.path.isdir(scene_dir):
        for name in os.listdir(scene_dir):
            if not name.startswith("_seed"):
                continue
            try:
                seed = int(name[len("_seed"):])
            except ValueError:
                continue
            if seed not in selected_seeds:
                shutil.rmtree(os.path.join(scene_dir, name), ignore_errors=True)

    placement_records, total_eps = [], 0
    for pid, (seed, tasks) in enumerate(selected):
        stage_dir = os.path.join(scene_dir, f"_seed{seed}")
        placement_dir = os.path.join(scene_dir, f"placement{pid}")
        if os.path.isdir(stage_dir):
            # Remove unpaired or incomplete tasks to avoid exposing partial pairs.
            for tname in list(os.listdir(stage_dir)):
                if tname not in tasks:
                    shutil.rmtree(os.path.join(stage_dir, tname), ignore_errors=True)
                    continue
                # Keep only complete recording directories, including perturbation variants.
                keep_dirs = set(dirs_by_pair[(seed, tname)])
                task_dir = os.path.join(stage_dir, tname)
                for dname in list(os.listdir(task_dir)):
                    if dname not in keep_dirs:
                        shutil.rmtree(os.path.join(task_dir, dname), ignore_errors=True)
            if os.path.isdir(placement_dir):
                shutil.rmtree(placement_dir, ignore_errors=True)
            os.rename(stage_dir, placement_dir)
        else:
            os.makedirs(placement_dir, exist_ok=True)

        with open(os.path.join(placement_dir, "placement_meta.json"), "w") as f:
            json.dump({
                "placement_seed": seed,
                "table_z_bias": environment.get("table_z_bias"),
                "distractors": False,
                "embodiments": emb_names,
                "pair_perturbation": pair_perturbation,
                "supported_tasks": list(tasks),
                # Save per-task, per-directory world poses and model IDs for diagnostics.
                # Different model IDs across embodiments indicate placement RNG divergence;
                # matching IDs with different poses can indicate robot-induced settling shifts.
                # <emb>__pert<k> snapshots also include per-object perturbation offsets.
                "placement": {
                    t: {name: snapshots.get((seed, t, name))
                        for name in dirs_by_pair[(seed, t)]}
                    for t in tasks
                },
            }, f, ensure_ascii=False, indent=2)

        rec_tasks = []
        for t in tasks:
            eps_by_emb = {name: recorded.get((seed, t, name), 0)
                          for name in dirs_by_pair[(seed, t)]}
            total_eps += sum(eps_by_emb.values())
            rec_tasks.append((t, eps_by_emb))
        placement_records.append({"pid": pid, "seed": seed, "tasks": rec_tasks})
    return placement_records, total_eps


def quota_is_met(placement_records, quota: dict) -> bool:
    """Check quotas against the pairs that survived strict completeness filtering."""
    remaining = dict(quota)
    for rec in placement_records:
        for k in quota_helps(remaining, len(rec["tasks"])):
            remaining[k] -= 1
    return not any(n > 0 for n in remaining.values())


def write_completion(scene_dir, manifest_path, spec, task_names, quota, a,
                     placement_records, total_eps, visual_variants, emb_names,
                     pair_perturbation=None, required_tasks=None) -> str | None:
    """Atomically mark a run complete when it produced at least one usable pair."""
    if not placement_records:
        print("[collect] No complete pairs available; completion.json will not be written")
        return None

    required = _normalise_required_tasks(required_tasks)
    for rec in placement_records:
        recorded_tasks = {task_name for task_name, _eps in rec["tasks"]}
        if not set(required).issubset(recorded_tasks):
            print(f"[collect] Placement is missing required tasks; completion.json will not be written: seed={rec['seed']}")
            return None

    expected_pairs = sum(len(rec["tasks"]) for rec in placement_records)
    if expected_pairs < 1:
        print("[collect] No complete pairs available; completion.json will not be written")
        return None
    # Each retained directory contributes visual_variants episodes; directory counts may vary.
    n_dirs = sum(len(eps_by_emb) for rec in placement_records
                 for _t, eps_by_emb in rec["tasks"])
    expected_eps = n_dirs * visual_variants
    if total_eps != expected_eps:
        print("[collect] Episode counts are incomplete; completion.json will not be written")
        return None

    completion = {
        "version": 3,
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "source": "scene_mode",
        "scene_id": spec.scene_id,
        "manifest_sha256": _file_sha256(manifest_path),
        "scene_config": a.scene_config,
        "tasks": list(task_names),
        "quota": {str(k): v for k, v in sorted(quota.items())},
        "seed_start": a.seed_start,
        "max_search": a.max_search,
        "visual": bool(a.visual),
        "visual_variants": visual_variants,
        "embodiments": list(emb_names),
        "placements": len(placement_records),
        "pairs": expected_pairs,
        "episodes": total_eps,
        "quota_met": quota_is_met(placement_records, quota),
        "pair_perturbation": pair_perturbation,
    }
    if required:
        completion["required_tasks"] = required
    path = os.path.join(scene_dir, "completion.json")
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(completion, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)
    return path


def write_empty_result(scene_dir, manifest_path, spec, task_names, quota, a,
                       emb_names, required_tasks=None) -> str:
    """Mark a completed search with no usable pairs so production can clean it up."""
    result = {
        "version": 1,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "source": "scene_mode",
        "status": "no_usable_pairs",
        "reason": "search_exhausted_without_usable_pairs",
        "scene_id": spec.scene_id,
        "manifest_sha256": _file_sha256(manifest_path),
        "scene_config": a.scene_config,
        "tasks": list(task_names),
        "quota": {str(k): v for k, v in sorted(quota.items())},
        "seed_start": a.seed_start,
        "max_search": a.max_search,
        "embodiments": list(emb_names),
    }
    required = _normalise_required_tasks(required_tasks)
    if required:
        result["required_tasks"] = required
    path = os.path.join(scene_dir, EMPTY_RESULT_FILE)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)
    return path


def search_pairs(scene_set, task_names, scene_config, order, environment,
                 seed_start, max_search, bindings_by_task=None, pair_perturbation=None,
                 debug=False, debug_dir=None, debug_max_videos=20,
                 required_tasks=None):
    """Search embodiments sequentially; return paired {seed: tasks} and successful variants.

    ALOHA subjects try K variants per (seed, task), requiring a demonstration and m
    successes. variants_by_pair stores successful indices; single-variant mode
    leaves it empty. This fallback searches the full seed window.
    """
    num_variants, min_variants = perturb_variant_plan(pair_perturbation)
    required = set(required_tasks or ())
    baseline = (pair_perturbation or {}).get("baseline_embodiment")
    variants_by_pair = {}
    accepted = None
    debug_state = {"max_videos": debug_max_videos, "reserved": 0}
    for idx, (emb, emb_name) in enumerate(order):
        tasks, _name, envs, args_by_task, matches = prepare_embodiment(
            scene_set, task_names, scene_config, emb, bindings_by_task)
        if not tasks:
            raise SystemExit(f"[collect] No tasks available for sampling on embodiment {emb_name}")
        missing_required = sorted(required - set(tasks))
        if missing_required:
            raise SystemExit(f"[collect] Cannot prepare required tasks for embodiment {emb_name}: {missing_required}")
        multi_other = bool((pair_perturbation or {}).get(
            "variant_directories", num_variants > 1))

        if idx == 0:
            seeds_iter = range(seed_start, seed_start + max_search)
        else:
            seeds_iter = sorted(accepted.keys())

        emb_success = {}
        for seed in seeds_iter:
            cand = tasks if idx == 0 else [t for t in tasks if t in accepted.get(seed, ())]
            for t in cand:
                if multi_other:
                    baseline_ok ,*_ =run_one (
                    envs [t ],t ,seed ,args_by_task [t ],scene_set ,matches [t ],environment ,
                    debug =debug ,debug_dir =debug_dir ,debug_state =debug_state )
                    if not baseline_ok :
                        continue
                    good = []
                    for k in range(num_variants):
                        ok, _info, _traj, _snap = run_one(
                            envs[t], t, seed, args_by_task[t], scene_set, matches[t],
                            environment, perturb_variant=k, debug=debug,
                            debug_dir=debug_dir, debug_state=debug_state)
                        if ok:
                            good.append(k)
                    if len(good) >= min_variants:
                        emb_success.setdefault(seed, set()).add(t)
                        variants_by_pair[(seed, t)] = good
                else:
                    ok, _info, _traj, _snap = run_one(
                        envs[t], t, seed, args_by_task[t], scene_set, matches[t], environment,
                        debug=debug, debug_dir=debug_dir, debug_state=debug_state)
                    if ok:
                        emb_success.setdefault(seed, set()).add(t)
        teardown_all_envs(envs)

        if idx == 0:
            accepted = {s: set(ts) for s, ts in emb_success.items()}
        else:
            accepted = {s: (emb_success.get(s, set()) & accepted[s]) for s in accepted}
        accepted = {s: ts for s, ts in accepted.items()
                    if ts and required.issubset(ts)}
        print(f"[search] Embodiment {emb_name} completed; successful ALOHA placements={len(accepted)}")
    return accepted or {}, variants_by_pair


def record_selected(scene_dir, selected, scene_set, scene_config, order, environment,
                    domain_randomization, visual_variants, use_visual_rng,
                    bindings_by_task=None, pair_perturbation=None, variants_by_pair=None):
    """Record quota-selected pairs sequentially by embodiment.

    Replay the demonstration and successful subject variants, using
    __pert<k> directory suffixes.
    """
    num_variants, _min_variants = perturb_variant_plan(pair_perturbation)
    baseline = (pair_perturbation or {}).get("baseline_embodiment")
    variants_by_pair = variants_by_pair or {}
    recorded, snapshots = {}, {}
    sel_tasks = sorted({t for _seed, tasks in selected for t in tasks})
    for emb, emb_name in order:
        _tasks, _name, envs, args_by_task, matches = prepare_embodiment(
            scene_set, sel_tasks, scene_config, emb, bindings_by_task)
        multi_other = bool((pair_perturbation or {}).get(
            "variant_directories", num_variants > 1))
        for seed, tasks_sel in selected:
            for t in tasks_sel:
                if t not in envs:
                    continue
                variant_list = ([None ,*variants_by_pair.get((seed, t), [])]
                                if multi_other else [None])
                for k in variant_list:
                    dir_name = emb_name if k is None else f"{emb_name}__pert{k}"
                    ok, _info, traj, snap = run_one(
                        envs[t], t, seed, args_by_task[t], scene_set, matches[t],
                        environment, perturb_variant=k)
                    if not ok:
                        print(f"    [record] {t} seed={seed} {dir_name}: "
                              f"Succeeded during search but failed to reproduce success during recording; skipping")
                        recorded[(seed, t, dir_name)] = 0
                        continue
                    save_dir = os.path.join(scene_dir, f"_seed{seed}", t, dir_name)
                    n = record_task(
                        envs[t], save_dir, t, seed, traj, snap, scene_set, matches[t],
                        environment, scene_config, emb, domain_randomization,
                        visual_variants, use_visual_rng, perturb_variant=k)
                    recorded[(seed, t, dir_name)] = n
                    snapshots[(seed, t, dir_name)] = snap
        teardown_all_envs(envs)
    return recorded, snapshots


def _prep_config(a):
    """Build collection settings shared by the launcher and workers."""
    if a.manifest:
        spec = SceneSpec.load(a.manifest)
    else:
        spec, _ = derive_task_manifest(
            a.tasks, a.scene_id, a.group_seed, a.scene_config,
            list_embodiments(a.scene_config), getattr(a, "exclude_tasks", None),
            getattr(a, "required_tasks", None),
            getattr(a, "pair_perturbation", None))
    scene_set = spec.build_object_set()
    embodiments = list_embodiments(a.scene_config)
    emb_names_cfg = [embodiment_name_of(e) for e in embodiments]
    task_names = [t["task_name"] for t in spec.supported_tasks]
    required_tasks = _normalise_required_tasks((spec.request or {}).get("required_tasks"))
    missing_required = sorted(set(required_tasks) - set(task_names))
    if missing_required:
        raise SystemExit(f"[collect] Scene does not support required tasks: {missing_required}")
    entries = {t.get("task_name"): t for t in spec.supported_tasks}
    bindings_by_task = {
        task_name: entries[task_name].get("binding")
        for task_name in task_names if task_name in entries
    }
    quota = parse_quota(a.quota)
    if a.visual:
        domain_randomization = load_collection_cfg(a.scene_config)
        use_visual_rng = True
        visual_variants = max(1, a.visual_variants)
    else:
        domain_randomization = dict(DR_ALL_OFF)
        use_visual_rng = False
        visual_variants = 1
    pair_perturbation = load_pair_perturbation_cfg(
        a.scene_config, getattr(a, "pair_perturbation", None))
    requested_perturbation = (spec.request or {}).get("pair_perturbation")
    if requested_perturbation is not None and requested_perturbation != pair_perturbation:
        raise SystemExit("[collect] pair_perturbation differs between the manifest and runtime arguments")
    return {
        "spec": spec, "scene_set": scene_set, "environment": spec.environment,
        "scene_id": spec.scene_id, "embodiments": embodiments, "emb_names_cfg": emb_names_cfg,
        "task_names": task_names, "quota": quota, "bindings_by_task": bindings_by_task,
        "required_tasks": required_tasks,
        "domain_randomization": domain_randomization,
        "use_visual_rng": use_visual_rng, "visual_variants": visual_variants,
        "pair_perturbation": pair_perturbation,
    }


def _gather(conns, phase=""):
    """Gather one response per worker; abort if a worker disconnects or reports an error."""
    out = {}
    for idx, c in conns.items():
        try:
            r = c.recv()
        except EOFError:
            raise SystemExit(f"[parallel] worker#{idx} disconnected{' @'+phase if phase else ''}; see worker logs")
        if isinstance(r, dict) and "error" in r:
            raise SystemExit(f"[parallel] worker#{idx} failed{' @'+phase if phase else ''}:{r['error']}")
        out[idx] = r
    return out


def _recv_until(conn, key, idx, phase=""):
    """Skip pending search messages until the requested response arrives."""
    while True:
        try:
            m = conn.recv()
        except EOFError:
            raise SystemExit(f"[parallel] worker#{idx} disconnected"
                             f"{' @'+phase if phase else ''}; see worker logs")
        if isinstance(m, dict):
            if "error" in m:
                raise SystemExit(f"[parallel] worker#{idx} failed"
                                 f"{' @'+phase if phase else ''}:{m['error']}")
            if key in m:
                return m


def run_worker(a):
    """Search and cache ALOHA trajectories, then record selected baseline and subject pairs."""
    from multiprocessing.connection import Client

    cfg = _prep_config(a)
    scene_set = cfg["scene_set"]
    environment = cfg["environment"]
    embodiments = cfg["embodiments"]
    emb = embodiments[a.emb_idx]
    emb_name = embodiment_name_of(emb)
    scene_dir = os.path.join(a.out_dir, a.scene_config, cfg["scene_id"])

    authkey = os.environ.get("COLLECT_AUTHKEY", "").encode()
    conn = Client((a.conn_host, a.conn_port), authkey=authkey)
    conn.send({"emb_idx": a.emb_idx, "emb_name": emb_name})

    tasks, _name, envs, args_by_task, matches = prepare_embodiment(
        scene_set, cfg["task_names"], a.scene_config, emb, cfg["bindings_by_task"])
    conn.send({"ready": True, "tasks": tasks})

    # Main workflow
    pp = cfg["pair_perturbation"]
    num_variants, min_variants = perturb_variant_plan(pp)
    multi_other =bool((pp or {}).get("variant_directories", num_variants > 1 ))

    traj_cache = {}
    search_stats = {}
    debug_state = {"max_videos": a.debug_max_videos, "reserved": 0}
    try:
        go = conn.recv()
        if go.get("cmd") == "search":
            end = a.seed_start + a.max_search
            s = a.seed_start
            stopped = False
            while s < end:
                if conn.poll(0):
                    m = conn.recv()
                    if m.get("cmd") in ("stop_search", "stop"):
                        stopped = True
                        break
                success = []
                for t in tasks:
                    if multi_other:
                        ok ,_info ,traj ,snap =run_one (
                        envs [t ],t ,s ,args_by_task [t ],scene_set ,matches [t ],environment ,
                        debug =a .debug ,debug_dir =a .debug_dir ,debug_state =debug_state )
                        _note_search_attempt (search_stats ,t ,envs [t ],ok )
                        if not ok :
                            _note_search_task_result (search_stats ,t ,False ,True )
                            continue
                        traj_cache [(s ,t )]=(traj ,snap )
                        good = []
                        for k in range(num_variants):
                            ok, _info, traj, snap = run_one(
                                envs[t], t, s, args_by_task[t], scene_set, matches[t],
                                environment, perturb_variant=k, debug=a.debug,
                                debug_dir=a.debug_dir, debug_state=debug_state)
                            _note_search_attempt(search_stats, t, envs[t], ok)
                            if ok:
                                good.append(k)
                                traj_cache[(s, t, k)] = (traj, snap)
                        task_ok = len(good) >= min_variants
                        _note_search_task_result(search_stats, t, task_ok, True)
                        if task_ok:
                            success.append(t)
                        else:
                            traj_cache .pop ((s ,t ),None )
                            for k in good:
                                traj_cache.pop((s, t, k), None)
                    else:
                        ok, _info, traj, snap = run_one(
                            envs[t], t, s, args_by_task[t], scene_set, matches[t], environment,
                            debug=a.debug, debug_dir=a.debug_dir, debug_state=debug_state)
                        _note_search_attempt(search_stats, t, envs[t], ok)
                        _note_search_task_result(search_stats, t, ok, False)
                        if ok:
                            success.append(t)
                            traj_cache[(s, t)] = (traj, snap)
                conn.send({"seed": s, "success": success})
                s += 1
            conn.send({"search_done": True, "stopped": stopped,
                       "stats": _search_stats_payload(search_stats)})

        while True:
            msg = conn.recv()
            cmd = msg.get("cmd")
            if cmd == "record":
                recorded, snaps = {}, {}
                for pr in msg.get("pairs", []):
                    s, t = int(pr[0]), pr[1]
                    if multi_other:
                        entries = [(None ,traj_cache .get ((s ,t )))]+[
                        (k, traj_cache.get((s, t, k)))for k in range(num_variants)]
                    else:
                        entries = [(None, traj_cache.get((s, t)))]
                    for k, cached in entries:
                        if cached is None:
                            continue
                        traj, snap = cached
                        dir_name = emb_name if k is None else f"{emb_name}__pert{k}"
                        key = f"{s}|{t}" if k is None else f"{s}|{t}|{k}"
                        save_dir = os.path.join(scene_dir, f"_seed{s}", t, dir_name)
                        n = record_task(
                            envs[t], save_dir, t, s, traj, snap, scene_set, matches[t],
                            environment, a.scene_config, emb, cfg["domain_randomization"],
                            cfg["visual_variants"], cfg["use_visual_rng"],
                            perturb_variant=k)
                        recorded[key] = n
                        snaps[key] = snap
                conn.send({"recorded": recorded, "snapshots": snaps})
            elif cmd == "stop":
                break
            elif cmd == "stop_search":
                continue
        teardown_all_envs(envs)
        conn.send({"stopped": True})
    except Exception as e:
        import traceback
        try:
            conn.send({"error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}"})
        except Exception:
            pass
        try:
            teardown_all_envs(envs)
        except Exception:
            pass
        raise
    finally:
        try:
            conn.close()
        except Exception:
            pass


def worker_cmd(a, emb_idx: int, host: str, port: int) -> list[str]:
    cmd = [sys.executable, "-u", "script/collect_scene_data.py",
           "--manifest", a.manifest, "--scene_config", a.scene_config,
           "--quota", a.quota, "--max_search", str(a.max_search),
           "--seed_start", str(a.seed_start), "--out_dir", a.out_dir,
           "--visual_variants", str(a.visual_variants),
           "--worker", "--emb_idx", str(emb_idx),
           "--conn_host", host, "--conn_port", str(port)]
    if getattr(a, "pair_perturbation", None):
        cmd += ["--pair-perturbation", json.dumps(
            a.pair_perturbation, sort_keys=True, separators=(",", ":"))]
    if a.visual:
        cmd += ["--visual"]
    if getattr(a, "debug", False):
        # Keep parallel workers in separate folders; each worker owns its env.
        base_debug_dir = getattr(a, "debug_dir", None) or os.path.join(a.out_dir, ".debug")
        debug_dir = os.path.join(base_debug_dir, f"worker{emb_idx}")
        total_workers = max(1, len(str(getattr(a, "gpus", "")).split(",")))
        max_videos = max(0, int(getattr(a, "debug_max_videos", 20)))
        per_worker, remainder = divmod(max_videos, total_workers)
        worker_limit = per_worker + (1 if emb_idx < remainder else 0)
        cmd += ["--debug", "--debug_dir", debug_dir,
                "--debug_max_videos", str(worker_limit)]
    return cmd


def run_parallel(a, cfg):
    """Launch the ALOHA worker, select placements online, and record cached trajectories."""
    import socket
    import secrets
    import subprocess
    from multiprocessing.connection import Listener, wait as mp_wait

    embodiments = cfg["embodiments"]
    quota = cfg["quota"]
    gpus = [g.strip() for g in a.gpus.split(",") if g.strip()]
    if len(gpus) < len(embodiments):
        raise SystemExit(f"[parallel] --gpus supplied {len(gpus)} GPUs for {len(embodiments)} workers; "
                         f"a GPU is required for each worker")

    authkey = secrets.token_hex(16)
    listener = Listener(("127.0.0.1", 0), authkey=authkey.encode())
    host, port = listener.address
    idx2name = {i: embodiment_name_of(embodiments[i]) for i in range(len(embodiments))}

    procs = []
    for i in range(len(embodiments)):
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = gpus[i]
        env["COLLECT_AUTHKEY"] = authkey
        cmd = worker_cmd(a, i, host, port)
        print(f"[parallel] Starting worker#{i} {idx2name[i]} @ GPU {gpus[i]}")
        procs.append(subprocess.Popen(cmd, env=env))

    def _terminate_all():
        for p in procs:
            if p.poll() is None:
                try:
                    p.terminate()
                except Exception:
                    pass

    try:
        conns = {}
        try:
            listener._listener._socket.settimeout(3.0)
        except Exception:
            pass
        while len(conns) < len(embodiments):
            try:
                c = listener.accept()
            except socket.timeout:
                for p in procs:
                    if p.poll() is not None:
                        raise SystemExit(f"[parallel] Worker exited during startup(code={p.returncode}); see worker logs")
                continue
            hello = c.recv()
            conns[hello["emb_idx"]] = c
            print(f"[parallel] worker#{hello['emb_idx']} {hello.get('emb_name')} connected")
        ready_responses = _gather(conns, "ready")
        _validate_ready_tasks(ready_responses, cfg["task_names"], idx2name)

        def broadcast(msg):
            for c in conns.values():
                c.send(msg)

        broadcast({"cmd": "search"})

        print(f"\n===== Search (online quota selection and early stopping) =====")
        conn_of = {c: i for i, c in conns.items()}
        sel = OnlineQuotaSelector(len(conns), quota, a.seed_start,
                                  cfg["required_tasks"])
        searching = set(conns.values())
        search_stats = {}
        while searching and not sel.met:
            ready = mp_wait(list(searching), timeout=2.0)
            if not ready:
                for i, p in enumerate(procs):
                    if p.poll() is not None:
                        raise SystemExit(f"[parallel] worker#{i} {idx2name[i]} exited during search"
                                         f"(code={p.returncode}); see worker logs")
                continue
            for c in ready:
                i = conn_of[c]
                while True:
                    try:
                        msg = c.recv()
                    except EOFError:
                        raise SystemExit(f"[parallel] worker#{i} disconnected@search; see worker logs")
                    if isinstance(msg, dict) and "error" in msg:
                        raise SystemExit(f"[parallel] worker#{i} failed @search:{msg['error']}")
                    if msg.get("search_done"):
                        searching.discard(c)
                        search_stats[i] = msg.get("stats", {})
                        print(f"[parallel] worker#{i} {idx2name[i]} exhausted the search window")
                        break
                    for s_sel, ts_sel in sel.report(i, msg["seed"], msg["success"]):
                        unmet = {k: v for k, v in sel.remaining.items() if v > 0}
                        print(f"[parallel] seed={s_sel} successful_tasks={ts_sel} -> selected"
                              f"(selected={len(sel.selected)}, remaining_quota={unmet})")
                    if sel.met or not c.poll(0):
                        break
                if sel.met:
                    break

        if sel.met:
            print(f"[parallel] Quota met; stopping search")
        for c in searching:
            try:
                c.send({"cmd": "stop_search"})
            except Exception:
                pass
        for c in list(searching):
            i = conn_of[c]
            done = _recv_until(c, "search_done", i, "stop_search")
            search_stats[i] = done.get("stats", {})
            searching.discard(c)
        _print_search_stats(search_stats, idx2name)

        selected = sel.selected
        if not selected:
            print("[parallel] No usable placements selected; increase --max_search or relax --quota")
        unmet = {k: v for k, v in sel.remaining.items() if v > 0}
        if unmet:
            print(f"[parallel] Search ended with unmet quotas: {unmet}")

        record_pairs = [[s, t] for s, ts in selected for t in ts]
        print(f"\n===== Recording ({len(selected)} placements / {len(record_pairs)} task pairs) =====")
        broadcast({"cmd": "record", "pairs": record_pairs})
        recorded, snapshots = {}, {}
        for i, c in conns.items():
            rec = _recv_until(c, "recorded", i, "record")
            name = idx2name[i]
            # Keys identify the baseline or a numbered subject variant.
            def _dir_key(key):
                parts = key.split("|")
                if len(parts) == 3:
                    return int(parts[0]), parts[1], f"{name}__pert{parts[2]}"
                s_str, t = parts
                return int(s_str), t, name
            for key, n in rec["recorded"].items():
                s_val, t, dir_name = _dir_key(key)
                recorded[(s_val, t, dir_name)] = n
            for key, snap in rec.get("snapshots", {}).items():
                s_val, t, dir_name = _dir_key(key)
                snapshots[(s_val, t, dir_name)] = snap

        broadcast({"cmd": "stop"})
        for p in procs:
            try:
                p.wait(timeout=60)
            except Exception:
                p.terminate()
        return selected, recorded, snapshots
    except BaseException:
        _terminate_all()
        raise
    finally:
        listener.close()


# Main workflow
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_config", default="default")
    parser.add_argument("--tasks", default=None,
                        help="Comma-separated task names")
    parser.add_argument("--exclude-tasks", default=None,
                        help="Comma-separated task names to exclude from compatible tasks inferred from the scene")
    parser.add_argument("--required-tasks", default=None,
                        help="Comma-separated task names that must succeed during search for every selected placement")
    parser.add_argument("--pair-perturbation", type=json.loads, default=None,
                        help="Plan-level perturbation JSON override; currently supports num_variants/min_variants")
    parser.add_argument("--scene_id", default=None,
                        help="Task-driven scene name (defaults to a name based on the inferred object set)")
    parser.add_argument("--group_seed", type=int, default=0,
                        help="Seed for task-driven object selection and environment sampling")
    parser.add_argument("--quota", default="1:1",
                        help="k:n,... (at least n placements supporting ≥k tasks; minimum counts, "
                             "placements supporting more tasks are preferred and retained; e.g. 1:4,2:2)")
    parser.add_argument("--max_search", type=int, default=200, help="Maximum number of placement_seed values to try")
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--visual", action="store_true",
                        help="Enable visual randomization during collection (using scene_config.collection and an independent visual RNG)")
    parser.add_argument("--visual_variants", type=int, default=1,
                        help="Visual variants to record per (placement, task) (L4); values >1 only apply with --visual")
    parser.add_argument("--debug", action="store_true",
                        help="Keep head-camera rendered videos and failure.json for a limited number of failed seeds")
    parser.add_argument("--debug_dir", default=None,
                        help="Failure debugging artifact directory; defaults to data/.debug/<scene_config>/<scene_id>/<run>")
    parser.add_argument("--debug_max_videos", type=int, default=20,
                        help="Maximum number of failure videos to keep (counted per worker/task/seed)")
    parser.add_argument("--out_dir", default="data")
    parser.add_argument("--reset", action="store_true",
                        help="Delete existing scene collection results and recollect (completed data is protected by default)")
    parser.add_argument("--dry-run", action="store_true")
    # The launcher uses --gpus; workers use the internal connection arguments.
    parser.add_argument("--gpus", default=None,
                        help="GPU for the ALOHA worker (the pipeline assigns one GPU per scene group). "
                             "Omit to use the serial fallback.")
    parser.add_argument("--worker", action="store_true", help="(internal) Worker subprocess mode")
    parser.add_argument("--emb_idx", type=int, default=0, help="(internal) Worker index")
    parser.add_argument("--conn_host", default="127.0.0.1", help="(internal) Launcher address")
    parser.add_argument("--conn_port", type=int, default=0, help="(internal) Launcher port")
    parser.add_argument("--manifest", default=None, help="(internal) Path to the frozen manifest.json")
    a = parser.parse_args()

    if not a.tasks and not a.manifest:
        parser.error("--tasks is required")

    # Workers keep their environments alive between search and recording.
    if a.worker:
        run_worker(a)
        return

    cfg = _prep_config(a)
    spec, scene_set = cfg["spec"], cfg["scene_set"]
    environment, scene_id = cfg["environment"], cfg["scene_id"]
    if a.debug and not a.debug_dir:
        a.debug_dir = os.path.join(
            a.out_dir, ".debug", a.scene_config, scene_id,
            datetime.now().strftime("%Y%m%d_%H%M%S"),
        )
        os.makedirs(a.debug_dir, exist_ok=True)
        print(f"[debug] Failure video directory={a.debug_dir} max={a.debug_max_videos}")
    embodiments, emb_names_cfg = cfg["embodiments"], cfg["emb_names_cfg"]
    task_names, quota = cfg["task_names"], cfg["quota"]
    required_tasks = cfg["required_tasks"]
    bindings_by_task = cfg["bindings_by_task"]
    domain_randomization = cfg["domain_randomization"]
    use_visual_rng, visual_variants = cfg["use_visual_rng"], cfg["visual_variants"]
    if not a.visual and a.visual_variants > 1:
        print("[collect] --visual is disabled; visual variants do not apply → visual_variants=1")

    print(f"[collect] scene={scene_id} objects={[o.oid for o in scene_set.objects]}")
    print(f"[collect] Robot={emb_names_cfg}")
    print(f"[collect] Candidate tasks={task_names} quota={quota} "
          f"visual={a.visual} variants={visual_variants} gpus={a.gpus} env={environment}")
    if required_tasks:
        print(f"[collect] Required tasks={required_tasks}")
    if cfg["pair_perturbation"]:
        pp = cfg["pair_perturbation"]
        K, m = perturb_variant_plan(pp)
        extra = f", {K} perturbation variants per placement, at least {m} successful" if K > 1 else ""
        print(f"[collect] ALOHA subject placement perturbation: ±{pp['translation']:.3f} m / ±{pp['yaw_deg']:.0f}°"
              f" (baseline {pp['baseline_embodiment']} with no perturbation{extra})")

    if not task_names:
        raise SystemExit("[collect] No candidate tasks")

    if a.dry_run:
        print("[dry-run] Parse tasks, static bindings, and environment only; skip manifest writing and simulation.")
        return

    scene_dir = os.path.join(a.out_dir, a.scene_config, scene_id)
    prepare_collection_dir(scene_dir, reset=a.reset)
    # Workers read only the frozen manifest to isolate collection from plan changes.
    manifest_path = write_manifest(scene_dir, spec)
    a.manifest = manifest_path
    cfg = _prep_config(a)
    spec, scene_set = cfg["spec"], cfg["scene_set"]
    environment, scene_id = cfg["environment"], cfg["scene_id"]
    embodiments, emb_names_cfg = cfg["embodiments"], cfg["emb_names_cfg"]
    task_names, quota = cfg["task_names"], cfg["quota"]
    bindings_by_task = cfg["bindings_by_task"]
    domain_randomization = cfg["domain_randomization"]
    use_visual_rng, visual_variants = cfg["use_visual_rng"], cfg["visual_variants"]

    if a.gpus:
        # Search ALOHA demonstrations and subjects, then record quota-selected pairs.
        selected, recorded, snapshots = run_parallel(a, cfg)
    else:
        # Serial fallback: search the full window and record quota-selected placements.
        order = [(emb, embodiment_name_of(emb)) for emb in embodiments]
        print(f"\n===== Phase A: Search ALOHA baseline and subject pairs (no recording) =====")
        accepted, variants_by_pair = search_pairs(
            scene_set, task_names, a.scene_config, order, environment,
            a.seed_start, a.max_search, bindings_by_task, cfg["pair_perturbation"],
            required_tasks=required_tasks,
            debug=a.debug, debug_dir=a.debug_dir,
            debug_max_videos=a.debug_max_videos)
        print(f"[collect] Search complete: placements successfully paired between demonstration and subject={len(accepted)}")
        selected = select_by_quota(accepted, quota, required_tasks)
        if not selected:
            print("[collect] ⚠️ No usable placements selected; "
                  "increase --max_search or relax --quota")
        recorded, snapshots = {}, {}
        if selected:
            print(f"\n===== Phase B: Record {len(selected)} confirmed paired placements =====")
            recorded, snapshots = record_selected(
                scene_dir, selected, scene_set, a.scene_config, order, environment,
                domain_randomization, visual_variants, use_visual_rng,
                bindings_by_task, cfg["pair_perturbation"], variants_by_pair)

    # Finalize complete demonstration/subject pairs.
    placement_records, total_eps = finalize_layout(
        scene_dir, selected, emb_names_cfg, recorded, snapshots, environment,
        visual_variants, cfg["pair_perturbation"], required_tasks)

    print(f"\n[collect] Completed: {len(placement_records)} fully paired placements, "
          f"{total_eps} episodes recorded → {scene_dir}")

    report_path = write_summary_report(
        scene_dir, spec, scene_set, task_names, quota, a,
        placement_records, total_eps, visual_variants, emb_names_cfg,
        cfg["pair_perturbation"])
    print(f"[collect] Summary report → {report_path}")
    completion_path = write_completion(
        scene_dir, manifest_path, spec, task_names, quota, a,
        placement_records, total_eps, visual_variants, emb_names_cfg,
        cfg["pair_perturbation"], required_tasks)
    if completion_path:
        print(f"[collect] Completion marker → {completion_path}")
    elif not selected:
        empty_path = write_empty_result(
            scene_dir, manifest_path, spec, task_names, quota, a, emb_names_cfg,
            required_tasks)
        print(f"[collect] Zero-pair marker → {empty_path}")


if __name__ == "__main__":
    # Only workers and single-GPU runs need the rendering test; the launcher creates no env.
    _is_launcher = (("--gpus" in sys.argv) and ("--worker" not in sys.argv)
                    or "--dry-run" in sys.argv)
    if not _is_launcher:
        import torch.multiprocessing as mp
        mp.set_start_method("spawn", force=True)

    main()
