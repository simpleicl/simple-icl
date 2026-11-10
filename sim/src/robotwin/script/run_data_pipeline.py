#!/usr/bin/env python
"""Run task-driven collection for each scene group in a plan.

Collection freezes tasks, group seeds, and scene_config in
data/<config>/<group>/manifest.json. Completion checks use the manifest hash
so changed plans cannot reuse incompatible data.
"""

import sys

sys.path.append("./")

import os
import json
import hashlib
import argparse
import subprocess
import shlex
import time
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

import yaml


EMPTY_RESULT_FILE = "empty_result.json"


def embodiment_name_of(emb) -> str:
    """Return the stable directory key for one embodiment specification."""
    if len(emb) == 1:
        return str(emb[0])
    if len(emb) == 3:
        return f"{emb[0]}+{emb[1]}"
    raise ValueError(f"bad embodiment spec {emb!r}")


def list_embodiment_names(scene_config: str) -> list:
    path = os.path.join("scene_config", f"{scene_config}.yml")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader) or {}
    embs = cfg.get("embodiments") or [cfg["embodiment"]]
    return [embodiment_name_of(list(e)) for e in embs]


GLOBAL_KEYS = ("scene_config", "visual", "visual_variants", "quota",
               "max_search", "seed_start", "group_seed", "exclude_tasks",
               "required_tasks", "pair_perturbation")
DEFAULTS = {"scene_config": "dual_arm_multi_perturb", "visual": True, "visual_variants": 2,
            "quota": "1:5", "max_search": 100, "seed_start": 0, "group_seed": 0,
            "exclude_tasks": [], "required_tasks": [], "pair_perturbation": None}


def normalise_exclude_tasks(exclude_tasks) -> list[str]:
    """Normalize task exclusions for command construction and fingerprints."""
    if exclude_tasks is None:
        return []
    if isinstance(exclude_tasks, str):
        values = exclude_tasks.split(",")
    else:
        values = exclude_tasks
    return sorted({str(task).strip() for task in values if str(task).strip()})


def normalise_required_tasks(required_tasks) -> list[str]:
    """Normalize tasks that must succeed in every selected placement."""
    if required_tasks is None:
        return []
    if isinstance(required_tasks, str):
        values = required_tasks.split(",")
    else:
        values = required_tasks
    return list(dict.fromkeys(str(task).strip() for task in values if str(task).strip()))


def load_plan(path: str) -> tuple[dict, list]:
    """Read plan defaults and normalize every named task group."""
    with open(path, "r", encoding="utf-8") as f:
        plan = yaml.load(f.read(), Loader=yaml.FullLoader) or {}
    defaults = dict(DEFAULTS)
    for key in GLOBAL_KEYS:
        if key in plan:
            defaults[key] = plan[key]
    groups = []
    for raw in plan.get("groups", []):
        if "name" not in raw or "tasks" not in raw:
            raise SystemExit(f"[plan] Group is missing name/tasks: {raw!r}")
        group = dict(defaults)
        group.update(raw)
        if not isinstance(raw["tasks"], list) or not raw["tasks"]:
            raise SystemExit(f"[plan] Group tasks must be a non-empty list: {raw!r}")
        group["tasks"] = [str(task).strip() for task in raw["tasks"] if str(task).strip()]
        if not group["tasks"]:
            raise SystemExit(f"[plan] Group tasks must contain valid task names: {raw!r}")
        group["required_tasks"] = normalise_required_tasks(group.get("required_tasks"))
        pair_override = group.get("pair_perturbation")
        if pair_override is not None and not isinstance(pair_override, dict):
            raise SystemExit(
                f"[plan] pair_perturbation for group {group['name']} must be a mapping")
        missing = sorted(set(group["required_tasks"]) - set(group["tasks"]))
        if missing:
            raise SystemExit(
                f"[plan] required_tasks for group {group['name']} must be drawn from tasks: {missing}")
        groups.append(group)
    names = [group["name"] for group in groups]
    duplicate = {name for name in names if names.count(name) > 1}
    if duplicate:
        raise SystemExit(f"[plan] Duplicate scene group names: {sorted(duplicate)}")
    return defaults, groups


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collection_request(grp: dict, emb_names: list) -> dict:
    """Mirror the task-derived manifest fingerprint produced by collect."""
    request = {
        "source": "scene_mode",
        "tasks": list(dict.fromkeys(grp["tasks"])),
        "task_selection": "all_compatible",
        "scene_config": grp["scene_config"],
        "scene_config_sha256": _file_sha256(
            os.path.join("scene_config", f"{grp['scene_config']}.yml")),
        "group_seed": grp.get("group_seed", 0),
        "embodiments": list(emb_names),
    }
    excluded = normalise_exclude_tasks(grp.get("exclude_tasks"))
    if excluded:
        request["exclude_tasks"] = excluded
    required = normalise_required_tasks(grp.get("required_tasks"))
    if required:
        request["required_tasks"] = required
    pair_override = grp.get("pair_perturbation")
    if pair_override:
        from envs.scene.pair_perturbation import (
            merge_pair_perturbation_override, normalize_pair_perturbation,
        )
        config_path = os.path.join("scene_config", f"{grp['scene_config']}.yml")
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.load(f.read(), Loader=yaml.FullLoader) or {}
        raw = merge_pair_perturbation_override(
            config.get("pair_perturbation"), pair_override)
        embs = config.get("embodiments") or [config["embodiment"]]
        request["pair_perturbation"] = normalize_pair_perturbation(
            raw, embodiment_name_of(list(embs[0])))
    return request


def quota_record(spec: str) -> dict:
    out = {}
    for part in spec.split(","):
        if part.strip():
            key, value = part.split(":", 1)
            out[str(int(key.strip()))] = int(value.strip())
    return dict(sorted(out.items(), key=lambda item: int(item[0])))


def collection_complete(data_dir: str, grp: dict, emb_names: list) -> bool:
    """Accept only a complete output tied to this task-derived manifest."""
    manifest_path = os.path.join(data_dir, "manifest.json")
    completion_path = os.path.join(data_dir, "completion.json")
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        with open(completion_path, "r", encoding="utf-8") as f:
            completion = json.load(f)
    except Exception:
        return False
    expected = collection_request(grp, emb_names)
    request = manifest.get("request") or {}
    if any(request.get(key) != value for key, value in expected.items()):
        return False
    entries = {entry.get("task_name"): entry for entry in manifest.get("supported_tasks", [])}
    supported_tasks = [entry.get("task_name") for entry in manifest.get("supported_tasks", [])
                       if entry.get("status") == "static_ready" and entry.get("task_name")]
    required_tasks = [task for task in expected["tasks"]
                      if task not in set(expected.get("exclude_tasks", []))]
    if (manifest.get("scene_id") != grp["name"]
            or any(entries.get(task, {}).get("status") != "static_ready"
                   for task in required_tasks)):
        return False
    expected_required = expected.get("required_tasks", [])
    if expected_required and completion.get("required_tasks") != expected_required:
        return False
    return (
        completion.get("version") == 3
        and completion.get("source") == "scene_mode"
        and completion.get("scene_id") == grp["name"]
        and completion.get("manifest_sha256") == _file_sha256(manifest_path)
        and completion.get("scene_config") == grp["scene_config"]
        and completion.get("tasks") == supported_tasks
        and completion.get("quota") == quota_record(str(grp["quota"]))
        and completion.get("seed_start") == grp["seed_start"]
        and completion.get("max_search") == grp["max_search"]
        and completion.get("visual") == bool(grp.get("visual"))
        and completion.get("visual_variants") == (
            grp["visual_variants"] if grp.get("visual") else 1)
        and completion.get("embodiments") == list(emb_names)
        and ("pair_perturbation" not in expected
             or completion.get("pair_perturbation") == expected["pair_perturbation"])
        and count_episodes(data_dir) == completion.get("episodes")
    )


def collection_empty(data_dir: str, grp: dict, emb_names: list) -> bool:
    """Accept a full-window search that normally ended without a usable pair."""
    manifest_path = os.path.join(data_dir, "manifest.json")
    result_path = os.path.join(data_dir, EMPTY_RESULT_FILE)
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        with open(result_path, "r", encoding="utf-8") as f:
            result = json.load(f)
    except Exception:
        return False
    expected = collection_request(grp, emb_names)
    request = manifest.get("request") or {}
    if any(request.get(key) != value for key, value in expected.items()):
        return False
    entries = {entry.get("task_name"): entry for entry in manifest.get("supported_tasks", [])}
    supported_tasks = [entry.get("task_name") for entry in manifest.get("supported_tasks", [])
                       if entry.get("status") == "static_ready" and entry.get("task_name")]
    required_tasks = [task for task in expected["tasks"]
                      if task not in set(expected.get("exclude_tasks", []))]
    if (manifest.get("scene_id") != grp["name"]
            or any(entries.get(task, {}).get("status") != "static_ready"
                   for task in required_tasks)):
        return False
    expected_required = expected.get("required_tasks", [])
    if expected_required and result.get("required_tasks") != expected_required:
        return False
    return (
        result.get("version") == 1
        and result.get("source") == "scene_mode"
        and result.get("status") == "no_usable_pairs"
        and result.get("reason") == "search_exhausted_without_usable_pairs"
        and result.get("scene_id") == grp["name"]
        and result.get("manifest_sha256") == _file_sha256(manifest_path)
        and result.get("scene_config") == grp["scene_config"]
        and result.get("tasks") == supported_tasks
        and result.get("quota") == quota_record(str(grp["quota"]))
        and result.get("seed_start") == grp["seed_start"]
        and result.get("max_search") == grp["max_search"]
        and result.get("embodiments") == list(emb_names)
    )


def count_episodes(data_dir: str) -> int:
    return sum(sum(fn.endswith(".hdf5") for fn in files)
               for _root, _dirs, files in os.walk(data_dir))


def format_duration(seconds: float) -> str:
    """Format elapsed seconds compactly for terminal status lines."""
    seconds = max(0, round(seconds))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def is_progress_line(line: str) -> bool:
    """Keep collect milestones visible without mirroring simulator noise."""
    if is_expected_search_failure(line):
        return False
    text = line.strip()
    prefixes = ("[collect]", "[parallel]", "[search]", "[sample]", "[record]", "[dry-run]")
    if text.startswith(prefixes):
        return True
    lower = text.lower()
    return any(token in lower for token in (
        "traceback", "error", "exception", "failed", "segmentation fault", "cuda illegal",
    ))


def is_expected_search_failure(line: str) -> bool:
    """Recognize normal seed-search rejections."""
    return expected_search_failure_kind(line) is not None


def expected_search_failure_kind(line: str) -> str | None:
    if "UnStableError:" in line:
        return "UnStableError"
    if "PlacementError: failed to place " in line and " after retries" in line:
        return "PlacementError"
    return None


def _write_log_header(log, title: str, command: str) -> None:
    log.write(f"# {title}\n")
    log.write(f"# started_at: {datetime.now().isoformat(timespec='seconds')}\n")
    log.write(f"# command: {command}\n\n")


def default_log_dir(plan_path: str) -> Path:
    """Create a unique log directory for one pipeline run."""
    plan_name = Path(plan_path).stem
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + f"_{os.getpid()}"
    return Path("logs") / "pipeline" / plan_name / run_id


def group_log_paths(log_dir: str | os.PathLike, group_name: str) -> tuple[Path, Path]:
    """Return progress and raw log paths for one scene group."""
    group_dir = Path(log_dir) / "groups" / group_name
    return group_dir / "progress.log", group_dir / "raw.log"


def run_cmd(cmd: list, env: dict, event_log_path: str, raw_log_path: str,
            quiet: bool, verbose: bool, dry: bool,
            cwd: str | os.PathLike | None = None,
            progress_prefix: list[str] | None = None) -> tuple[int, float]:
    """Log collect output, mirroring it to the terminal only in verbose mode."""
    printable = shlex.join(cmd)
    if verbose or dry:
        print(f"  Command   {printable}", flush=True)
    if dry:
        return 0, 0.0
    os.makedirs(os.path.dirname(event_log_path), exist_ok=True)
    os.makedirs(os.path.dirname(raw_log_path), exist_ok=True)
    started = time.monotonic()
    rejection_counts: dict[str, int] = {}
    with (open(event_log_path, "w", encoding="utf-8") as event_log,
          open(raw_log_path, "w", encoding="utf-8") as raw_log):
        _write_log_header(event_log, "RoboTwin collect progress", printable)
        _write_log_header(raw_log, "RoboTwin collect raw output", printable)
        for line in progress_prefix or []:
            event_log.write(line.rstrip("\n") + "\n")
            raw_log.write(line.rstrip("\n") + "\n")
        event_log.flush()
        raw_log.flush()
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1,
                                cwd=cwd)
        for line in proc.stdout:
            raw_log.write(line)
            raw_log.flush()
            kind = expected_search_failure_kind(line)
            if kind is not None:
                rejection_counts[kind] = rejection_counts.get(kind, 0) + 1
            if verbose or is_progress_line(line):
                event_log.write(line)
                event_log.flush()
                if verbose and not quiet:
                    print(f"  {format_duration(time.monotonic() - started)}  {line.rstrip()}")
        proc.stdout.close()
        proc.wait()
        elapsed = time.monotonic() - started
        if rejection_counts and not verbose:
            summary = ", ".join(f"{name}: {count} entries" for name, count in rejection_counts.items())
            notice = (f"[search] Expected rejections summarized: {summary} "
                      f"(see {raw_log_path} for individual entries)\n")
            event_log.write(notice)
            event_log.flush()
            if verbose and not quiet:
                print(f"  {format_duration(elapsed)}  {notice.rstrip()}")
        footer = (f"\n# finished_at: {datetime.now().isoformat(timespec='seconds')}\n"
                  f"# exit_code: {proc.returncode}\n"
                  f"# elapsed: {format_duration(elapsed)}\n")
        event_log.write(footer)
        raw_log.write(footer)
    return proc.returncode, elapsed


def collect_cmd(grp: dict, gpus: str | None, out_dir: str,
                debug: bool = False, debug_dir: str | None = None,
                debug_max_videos: int = 20) -> list:
    """Construct the sole pipeline command; it never relies on a scene file."""
    cmd = [sys.executable, "-u", "script/collect_scene_data.py",
           "--tasks", ",".join(grp["tasks"]),
           "--scene_id", grp["name"],
           "--group_seed", str(grp.get("group_seed", 0)),
           "--scene_config", grp["scene_config"],
           "--quota", str(grp["quota"]),
           "--max_search", str(grp["max_search"]),
           "--seed_start", str(grp["seed_start"]),
           "--out_dir", out_dir]
    excluded = normalise_exclude_tasks(grp.get("exclude_tasks"))
    if excluded:
        cmd += ["--exclude-tasks", ",".join(excluded)]
    required = normalise_required_tasks(grp.get("required_tasks"))
    if required:
        cmd += ["--required-tasks", ",".join(required)]
    if grp.get("pair_perturbation"):
        cmd += ["--pair-perturbation", json.dumps(
            grp["pair_perturbation"], sort_keys=True, separators=(",", ":"))]
    if grp.get("visual"):
        cmd += ["--visual", "--visual_variants", str(grp["visual_variants"])]
    if gpus:
        cmd += ["--gpus", gpus]
    if debug:
        cmd.append("--debug")
        if debug_dir:
            # A group gets an isolated directory; worker processes add their
            # own suffix below this path when parallel collection is enabled.
            group_debug_dir = os.path.join(debug_dir, grp["scene_config"], grp["name"])
            cmd += ["--debug_dir", group_debug_dir]
        cmd += ["--debug_max_videos", str(debug_max_videos)]
    return cmd


def main():
    parser = argparse.ArgumentParser(description="Run task-driven collection for each plan group")
    parser.add_argument("--plan", default="scene_plan.yml")
    parser.add_argument("--scene_config", default=None, help="Override scene_config from the plan")
    parser.add_argument("--only", default=None, help="Comma-separated group names to run")
    parser.add_argument("--skip", default=None, help="Comma-separated group names to skip")
    parser.add_argument("--exclude-tasks", default=None,
                        help="Comma-separated task names; override the compatible-task exclusion set in the plan")
    parser.add_argument("--gpu", default="0", help="GPU to use for serial collection on a single GPU")
    parser.add_argument("--gpus", default=None, help="Comma-separated GPU IDs; each GPU independently collects one scene group")
    parser.add_argument("--out_dir", default="data")
    parser.add_argument("--debug", action="store_true",
                        help="Keep rendered videos and failure.json for seeds that fail collection")
    parser.add_argument("--debug_dir", default=None,
                        help="Root directory for failure diagnostics; defaults to a directory per scene group")
    parser.add_argument("--debug_max_videos", type=int, default=20,
                        help="Maximum failure videos per scene group")
    parser.add_argument(
        "--log_dir", default=None,
        help="Directory for per-group progress.log and raw.log files")
    parser.add_argument("--recollect", action="store_true", help="Recollect completed groups")
    parser.add_argument("--continue-on-error", action="store_true")
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument("--quiet", action="store_true", help="Only show group status (the default)")
    output_mode.add_argument("--verbose", action="store_true", help="Show all raw collection output in the terminal")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.log_dir = str(Path(args.log_dir) if args.log_dir else default_log_dir(args.plan))

    _defaults, groups = load_plan(args.plan)
    if args.exclude_tasks is not None:
        excluded = normalise_exclude_tasks(args.exclude_tasks)
        for group in groups:
            group["exclude_tasks"] = excluded
    if args.scene_config:
        for group in groups:
            group["scene_config"] = args.scene_config
    only = {item.strip() for item in args.only.split(",")} if args.only else None
    skip = {item.strip() for item in args.skip.split(",")} if args.skip else set()
    selected = [group for group in groups if (only is None or group["name"] in only)
                and group["name"] not in skip]
    if only:
        unknown = only - {group["name"] for group in groups}
        if unknown:
            print(f"[plan] Unknown group names: {sorted(unknown)}")

    mode = f"GPUs {args.gpus}" if args.gpus else f"GPU {args.gpu}"
    print(f"[plan] Starting {'dry run' if args.dry_run else 'collection'}: "
          f"{len(selected)}/{len(groups)} groups | {mode}", flush=True)
    gpu_ids =(args .gpus or args .gpu ).split (",")
    if any (not gpu .strip ().isdigit ()for gpu in gpu_ids ):
        parser .error ("GPU indices must be comma-separated nonnegative integers")
    gpu_ids = [str (int (gpu ))for gpu in gpu_ids ]
    if len (set (gpu_ids ))!=len (gpu_ids ):
        parser .error ("GPU indices must be unique")
    available =Queue ()
    for gpu in gpu_ids :
        available .put (gpu )

    def collect_group (index, group ,gpu ):
        name, cfg = group["name"], group["scene_config"]
        data_dir = os.path.join(args.out_dir, cfg, name)
        emb_names = list_embodiment_names(cfg)
        prefix = f"[{index:02d}/{len(selected):02d}] {name}"
        if not args.recollect and collection_complete(data_dir, group, emb_names):
            print(f"{prefix} SKIP (matching completed dataset)", flush=True)
            status, elapsed = "skip", 0.0
        elif not args.recollect and collection_empty(data_dir, group, emb_names):
            print(f"{prefix} SKIP (search exhausted without usable pairs)", flush=True)
            status, elapsed = "empty", 0.0
        else:
            event_log, raw_log = group_log_paths(Path(args.log_dir).resolve(), name)
            event_log_path, raw_log_path = str(event_log), str(raw_log)
            metadata = [
                f"{prefix} START | GPU {gpu}",
                f"  Tasks     {len(group['tasks'])}: {', '.join(group['tasks'])}",
                f"  Config    {cfg} | robots: {', '.join(emb_names)}",
                f"  Search    quota {group['quota']} | max {group['max_search']}"
                f" | seed_start {group['seed_start']} | group_seed {group.get('group_seed', 0)}",
                f"  Visual    {'on' if group.get('visual') else 'off'}"
                f" | variants {group.get('visual_variants', 2) if group.get('visual') else 1}",
            ]
            if group.get("required_tasks"):
                metadata.append(f"  Required  {', '.join(group['required_tasks'])}")
            metadata.append(f"  Output    {data_dir}")
            metadata.append(f"  Raw log   {raw_log_path}")
            metadata.append(f"  Progress  {event_log_path}")
            # Emit one block so concurrent groups keep their metadata together.
            print("\n".join(metadata), flush=True)
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] =gpu
            cmd = collect_cmd(group,gpu , args.out_dir,
                              debug=args.debug, debug_dir=args.debug_dir,
                              debug_max_videos=args.debug_max_videos)
            if args.recollect or os.path.exists(os.path.join(data_dir, "completion.json")):
                cmd.append("--reset")
            details = [f"[plan] Group: {name} | GPU {gpu}",
                       f"[plan] Settings: {json.dumps(group, ensure_ascii=False, default=str)}",
                       f"[plan] Output: {data_dir}"]
            rc, elapsed = run_cmd(cmd, env, event_log_path, raw_log_path,
                                  args.quiet, args.verbose, args.dry_run,
                                  progress_prefix=details)
            status = ("empty" if rc == 0 and collection_empty(data_dir, group, emb_names)
                      else "ok" if rc == 0 else f"FAIL({rc})")
            print(f"{prefix} STOP {status.upper()} | {count_episodes(data_dir)} episodes"
                  f" | {format_duration(elapsed)}"
                  + (f" | log: {raw_log_path}" if rc else ""), flush=True)
            if rc != 0 and not args.continue_on_error and not args.dry_run:
                raise SystemExit(f"[plan] Collection failed for {name}; stopping")
        return (name, status, 0 if args.dry_run else count_episodes(data_dir), elapsed )

    def collect_on_available_gpu (item ):
        gpu =available .get ()
        try :
            return collect_group (*item ,gpu )
        finally :
            available .put (gpu )

    with ThreadPoolExecutor (max_workers =len (gpu_ids ))as executor :
        report =list (executor .map (collect_on_available_gpu ,enumerate (selected ,1 )))

    print(f"[plan] Finished: {len(report)} groups | "
          f"{sum(item[2] for item in report)} episodes", flush=True)


if __name__ == "__main__":
    main()
