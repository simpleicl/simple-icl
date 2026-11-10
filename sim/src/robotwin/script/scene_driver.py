"""Scene environment setup, single-seed execution, and resource cleanup."""

import os
from simpleicl.paths import resource_path
import importlib
import ast
import traceback
import json
import shutil
from copy import deepcopy

import numpy as np
import yaml

from envs._GLOBAL_CONFIGS import CONFIGS_PATH


_TRACED_FAILURES = set()
_DEBUG_ATTEMPT = 0


def _json_default(value):
    """Serialize NumPy values used by task checks and planner poses."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _debug_prepare(args, task_name, seed, perturb_variant, debug_dir, debug_state):
    """Reserve a temporary cache directory for one failed-attempt recording."""
    if not debug_dir or debug_state is None:
        return None
    limit = int(debug_state.get("max_videos", 0))
    if int(debug_state.get("reserved", 0)) >= limit:
        return None
    global _DEBUG_ATTEMPT
    _DEBUG_ATTEMPT += 1
    emb_name = str(args.get("embodiment_name", "embodiment"))
    variant = "base" if perturb_variant is None else f"pert{perturb_variant}"
    path = os.path.join(
        debug_dir,
        emb_name,
        task_name,
        f"seed{int(seed)}_{variant}_attempt{_DEBUG_ATTEMPT:04d}",
    )
    os.makedirs(path, exist_ok=True)
    debug_state["reserved"] = int(debug_state.get("reserved", 0)) + 1
    return path


def _debug_cache_path(env):
    folder = getattr(env, "folder_path", None) or {}
    return folder.get("cache")


def _debug_write_result(
    path,
    task_name,
    seed,
    perturb_variant,
    status,
    stage,
    checks=None,
    error=None,
    video_path=None,
    planner_failure=None,
):
    if not path:
        return
    payload = {
        "task": task_name,
        "seed": int(seed),
        "variant": perturb_variant,
        "status": status,
        "stage": stage,
        "success_checks": checks or {},
        "planner_failure": planner_failure,
        "error": error,
        "video": video_path,
    }
    result_path = os.path.join(path, "failure.json")
    temp_path = f"{result_path}.tmp"
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=_json_default)
        os.replace(temp_path, result_path)
    except Exception as exc:
        # Diagnostics must never terminate a search worker. Keep a readable
        # fallback even if a task returns an unexpected custom value.
        fallback = {
            "task": str(task_name),
            "seed": int(seed),
            "variant": None if perturb_variant is None else int(perturb_variant),
            "status": str(status),
            "stage": None if stage is None else str(stage),
            "serialization_error": f"{type(exc).__name__}: {exc}",
            "raw": repr(payload),
        }
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(fallback, f, ensure_ascii=False, indent=2)
        os.replace(temp_path, result_path)


def _debug_finalize(env, path, failed):
    """Convert cached RGB frames to MP4 only for failed runs, then clean cache."""
    cache_path = _debug_cache_path(env)
    if not path:
        return None
    video_path = None
    try:
        if failed and cache_path and os.path.isdir(cache_path):
            frames = [
                name
                for name in os.listdir(cache_path)
                if name.endswith(".pkl") and name[:-4].isdigit()
            ]
            if frames:
                from envs.utils.pkl2hdf5 import process_folder_to_hdf5_video

                hdf5_path = os.path.join(path, "failure.hdf5")
                candidate_video_path = os.path.join(path, "failure.mp4")
                process_folder_to_hdf5_video(
                    cache_path, hdf5_path, candidate_video_path
                )
                video_path = candidate_video_path
                try:
                    os.remove(hdf5_path)
                except FileNotFoundError:
                    pass
    except Exception as exc:
        try:
            os.remove(os.path.join(path, "failure.mp4"))
        except FileNotFoundError:
            pass
        try:
            os.remove(os.path.join(path, "failure.hdf5"))
        except FileNotFoundError:
            pass
        try:
            with open(
                os.path.join(path, "video_error.txt"), "w", encoding="utf-8"
            ) as f:
                f.write(f"{type(exc).__name__}: {exc}\n")
        except OSError:
            pass
    finally:
        if cache_path:
            shutil.rmtree(cache_path, ignore_errors=True)
    return video_path


# Environment setup
def env_class(task_name):
    """Return the task environment class."""
    envs_module = importlib.import_module(f"envs.{task_name}")
    return getattr(envs_module, task_name)


def class_decorator(task_name):
    """Instantiate a task environment."""
    return env_class(task_name)()


def get_embodiment_config(robot_file):
    with open(os.path.join(robot_file, "config.yml"), "r", encoding="utf-8") as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


def embodiment_name_of(emb_type) -> str:
    """Convert an embodiment specification to a stable directory name."""
    if list(emb_type) != ["aloha-agilex"]:
        raise ValueError("SimpleICL supports only the aloha-agilex embodiment")
    return "aloha-agilex"


def list_embodiments(scene_config="default") -> list:
    """Read the embodiment list from scene_config."""
    with open(f"./scene_config/{scene_config}.yml", "r", encoding="utf-8") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader) or {}
    embs = cfg.get("embodiments")
    if embs:
        return [list(e) for e in embs]
    return [list(cfg["embodiment"])]


def resolve_pair_perturbation(cfg: dict) -> dict | None:
    """Parse perturbation settings with embodiments[0] as the baseline."""
    from envs.scene.pair_perturbation import normalize_pair_perturbation

    embs = cfg.get("embodiments")
    baseline = list(embs[0]) if embs else list(cfg["embodiment"])
    return normalize_pair_perturbation(
        cfg.get("pair_perturbation"), embodiment_name_of(baseline)
    )


DR_ALL_OFF = {
    "random_background": False,
    "cluttered_table": False,
    "clean_background_rate": 1,
    "random_head_camera_dis": 0,
    "random_table_height": 0,
    "random_light": False,
    "crazy_random_light_rate": 0,
}


def build_args(
    task_name,
    scene_config="default",
    domain_randomization=None,
    embodiment_override=None,
    perturb_variant=None,
):
    """Build environment arguments for one task and embodiment from scene_config."""
    with open(f"./scene_config/{scene_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    pair_perturbation = resolve_pair_perturbation(args)
    args.pop("generation", None)
    args.pop("collection", None)
    args.pop("embodiments", None)
    args.pop("pair_perturbation", None)
    args["pair_perturbation"] = pair_perturbation
    args["perturb_variant"] = perturb_variant
    args["domain_randomization"] = (
        dict(DR_ALL_OFF) if domain_randomization is None else dict(domain_randomization)
    )
    args["task_name"] = task_name
    args["task_config"] = scene_config

    with open(
        os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), "r", encoding="utf-8"
    ) as f:
        _emb = yaml.load(f.read(), Loader=yaml.FullLoader)

    def emb_file(t):
        rf = _emb[t]["file_path"]
        if rf is None:
            raise ValueError("missing embodiment files")
        return str(resource_path(rf))

    emb_type = (
        list(embodiment_override)
        if embodiment_override is not None
        else args["embodiment"]
    )
    args["embodiment"] = emb_type
    embodiment_name_of(emb_type)
    args["left_robot_file"] = emb_file("aloha-agilex")
    args["right_robot_file"] = emb_file("aloha-agilex")
    args["dual_arm_embodied"] = True
    embodiment_name = embodiment_name_of(emb_type)

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])
    args["embodiment_name"] = embodiment_name
    args["render_freq"] = 0
    args["save_data"] = False
    args["save_path"] = os.path.join("./data", task_name, scene_config)
    return args


def _class_defines_scene_binding(module_name, class_name, seen=None) -> bool:
    """Find bind_actors in the inheritance chain before Base_Task."""
    key = (module_name, class_name)
    seen = set() if seen is None else seen
    if key in seen or key == ("envs._base_task", "Base_Task"):
        return False
    seen.add(key)

    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None:
        return False
    with open(spec.origin, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=spec.origin)

    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    cls = classes.get(class_name)
    if cls is None:
        return False
    if any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "bind_actors"
        for node in cls.body
    ):
        return True

    imported = {}
    package = module_name.rpartition(".")[0]
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        target = node.module or ""
        if node.level:
            target = importlib.util.resolve_name("." * node.level + target, package)
        for alias in node.names:
            imported[alias.asname or alias.name] = (target, alias.name)

    for base in cls.bases:
        if not isinstance(base, ast.Name):
            continue
        parent = imported.get(base.id, (module_name, base.id))
        if _class_defines_scene_binding(*parent, seen):
            return True
    return False


def is_scene_ready(task_name) -> bool:
    """Check whether the task or a shared base class implements bind_actors."""
    try:
        return _class_defines_scene_binding(f"envs.{task_name}", task_name)
    except (ImportError, OSError, SyntaxError, ValueError):
        return False


def binding_to_dict(match) -> dict:
    """Convert TaskMatch to a JSON-serializable {role_name: [oid, ...]} mapping."""
    return {b.role.name: [o.oid for o in b.objects] for b in match.bindings}


def snapshot_placement(env, scene_set):
    """Save settled poses and pre-bind RNG state for diagnostics and local target reconstruction."""
    oid2name = {o.oid: o.modelname for o in scene_set.objects}
    objects = []
    for oid, actor in getattr(env, "_scene_actors_by_oid", {}).items():
        pose = actor.get_pose()
        objects.append(
            {
                "oid": oid,
                "modelname": oid2name.get(oid, oid.split("#")[0]),
                "model_id": getattr(actor, "model_id", None),
                "p": [float(x) for x in pose.p],
                "q": [float(x) for x in pose.q],
            }
        )
        if next(o for o in scene_set.objects if o.oid == oid).kind == "articulated":
            objects[-1]["joint_qpos"] = [float(v) for v in actor.get_qpos()]
    snapshot = {"objects": objects}
    offsets = getattr(env, "_pair_perturbation_offsets", None)
    if offsets is not None:
        snapshot["pair_perturbation_offsets"] = deepcopy(offsets)
    task_rng_state = getattr(env, "_scene_task_rng_state", None)
    if task_rng_state is not None:
        snapshot["task_rng_state"] = deepcopy(task_rng_state)
    return snapshot


def teardown_env(env):
    """Release planner GPU resources in child and main processes."""
    import gc

    robot = getattr(env, "robot", None)
    if robot is not None:
        if getattr(robot, "communication_flag", False):
            for conn_attr, proc_attr in (
                ("left_conn", "left_proc"),
                ("right_conn", "right_proc"),
            ):
                conn = getattr(robot, conn_attr, None)
                proc = getattr(robot, proc_attr, None)
                try:
                    if conn is not None:
                        conn.send({"cmd": "exit"})
                except Exception:
                    pass
                try:
                    if proc is not None:
                        proc.join(timeout=5)
                        if proc.is_alive():
                            proc.terminate()
                except Exception:
                    pass
        for attr in (
            "left_planner",
            "right_planner",
            "left_mplib_planner",
            "right_mplib_planner",
        ):
            try:
                if hasattr(robot, attr):
                    delattr(robot, attr)
            except Exception:
                pass
    try:
        if hasattr(env, "robot"):
            del env.robot
    except Exception:
        pass
    try:
        env.close_env()
    except Exception:
        pass
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass


def run_one(
    env,
    task_name,
    seed,
    args,
    scene_set,
    match,
    environment=None,
    perturb_variant=None,
    debug=False,
    debug_dir=None,
    debug_state=None,
):
    """Run one seed; return success, trajectories, and the pre-execution placement snapshot.

    perturb_variant selects a multi-variant index; None uses single-variant behavior.
    """
    ok, info = False, None
    run_status = "unknown_failure"
    try:
        env._planner_failure = None
    except Exception:
        pass
    traj = {"left_joint_path": [], "right_joint_path": []}
    snapshot = []
    debug_path = (
        _debug_prepare(args, task_name, seed, perturb_variant, debug_dir, debug_state)
        if debug
        else None
    )
    run_args = dict(args) if debug_path else args
    if debug_path:
        # Keep the normal planner path, but persist sparse RGB observations in
        # a private cache so failed attempts can be rendered after the run.
        run_args.update(
            {
                "save_data": True,
                "save_path": debug_path,
                "render_freq": 0,
                "save_freq": int(run_args.get("debug_save_freq", 10)),
            }
        )
    debug_error = None
    if perturb_variant is not None:
        run_args = dict(run_args)
        run_args["perturb_variant"] = perturb_variant
    try:
        env.scene_spec = scene_set
        env.scene_binding = match
        env.scene_environment = environment
        env.scene_task_rng_state = None
        env.setup_demo(now_ep_num=0, seed=seed, **run_args)
        snapshot = snapshot_placement(env, scene_set)
        info = env.play_once()
        traj = {
            "left_joint_path": deepcopy(env.left_joint_path),
            "right_joint_path": deepcopy(env.right_joint_path),
        }
        if not env.plan_success:
            stage = getattr(env, "_expert_stage", None)
            run_status = f"plan_failure@{stage}" if stage else "plan_failure"
        else:
            ok = bool(env.check_success())
            if ok:
                run_status = "success"
            else:
                checks = getattr(env, "_success_checks", None) or {}
                failed = sorted(name for name, passed in checks.items() if not passed)
                run_status = (
                    f"check_failure@{'+'.join(failed)}" if failed else "check_failure"
                )
        env.close_env()
    except Exception as e:
        ok = False
        run_status = type(e).__name__
        debug_error = f"{type(e).__name__}: {e}"
        print(f"    [{task_name}] seed={seed}: {type(e).__name__}: {e}")
        signature = (task_name, type(e).__name__, str(e))
        expected = (
            type(e).__name__ == "UnStableError"
            or type(e).__name__ == "PerturbationInfeasible"
            or (
                type(e).__name__ == "PlacementError"
                and str(e).startswith("failed to place ")
                and str(e).endswith(" after retries")
            )
        )
        if not expected and signature not in _TRACED_FAILURES:
            _TRACED_FAILURES.add(signature)
            traceback.print_exc()
        try:
            env.close_env()
        except Exception:
            pass
    if debug_path:
        checks = getattr(env, "_success_checks", None) or {}
        video_path = _debug_finalize(env, debug_path, not ok)
        if not ok:
            try:
                _debug_write_result(
                    debug_path,
                    task_name,
                    seed,
                    perturb_variant,
                    run_status,
                    getattr(env, "_expert_stage", None),
                    checks,
                    debug_error,
                    video_path,
                    getattr(env, "_planner_failure", None),
                )
            except Exception as exc:
                # A diagnostic filesystem/serialization error must not abort
                # the placement search or its worker process.
                try:
                    with open(
                        os.path.join(debug_path, "failure.json"), "w", encoding="utf-8"
                    ) as f:
                        json.dump(
                            {
                                "task": str(task_name),
                                "seed": int(seed),
                                "status": str(run_status),
                                "diagnostic_error": f"{type(exc).__name__}: {exc}",
                            },
                            f,
                            ensure_ascii=False,
                            indent=2,
                        )
                except Exception:
                    pass
        else:
            # A reserved slot is only consumed by a failed attempt.
            if debug_state is not None:
                debug_state["reserved"] = max(
                    0, int(debug_state.get("reserved", 1)) - 1
                )
            shutil.rmtree(debug_path, ignore_errors=True)
    env._last_run_status = run_status
    return ok, info, traj, snapshot
