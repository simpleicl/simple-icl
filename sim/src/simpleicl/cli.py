"""Public collection, preprocessing, training, and evaluation commands."""

import argparse
import shlex
import subprocess
import sys
import warnings
from pathlib import Path

from simpleicl.paths import (
    ASSETS,
    CHECKPOINTS,
    CONFIGS,
    DATA,
    OUTPUTS,
    ROBOTWIN,
    ROOT,
    require,
    subprocess_env,
)


def _run(
    command: list[str],
    *,
    cwd: Path = ROOT,
    dry_run: bool = False,
    env: dict[str, str] | None = None,
    show_command: bool = True,
):
    if show_command or dry_run:
        print(shlex.join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=cwd, env=env or subprocess_env(), check=True)


def _arguments(parser):
    argv = sys.argv[1:]
    if "--" in argv:
        separator = argv.index("--")
        return parser.parse_args(argv[:separator]), argv[separator + 1 :]
    return parser.parse_args(argv), []


def _gpus(value: str) -> list[int]:
    try:
        ids = [int(item) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Use comma-separated GPU indices, e.g. 0,1"
        ) from exc
    if not ids or min(ids) < 0 or len(set(ids)) != len(ids):
        raise argparse.ArgumentTypeError("GPU indices must be nonnegative and unique")
    return ids


def _pretrained(path: Path) -> Path:
    path = path.expanduser().absolute()
    for name in (
        "Wan2.2_VAE.pth",
        "models_t5_umt5-xxl-enc-bf16.pth",
        "google/umt5-xxl/tokenizer_config.json",
    ):
        require(path / name, "Wan pretrained component")
    if not list(path.glob("diffusion_pytorch_model*.safetensors")):
        raise FileNotFoundError(f"Missing Wan DiT shards: {path}")
    return path


def collect():
    parser = argparse.ArgumentParser(
        description="Generate a collection plan or collect paired ALOHA scenes."
    )
    parser.add_argument(
        "--plan", type=Path, help="Existing YAML plan; omit to generate one"
    )
    parser.add_argument("--generate-only", action="store_true")
    task_source = parser.add_mutually_exclusive_group()
    task_source.add_argument("--tasks", help="Comma-separated tasks for plan generation")
    task_source.add_argument(
        "--tasks-file", type=Path,
        help="Task list for plan generation; default: RoboTwin task_config/plan_gen_list.yaml",
    )
    parser.add_argument("--groups", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpus", type=_gpus, default=[0])
    parser.add_argument("--output", type=Path, default=OUTPUTS / "collection")
    parser.add_argument("--dry-run", action="store_true")
    args, extra = _arguments(parser)
    if args.groups < 1:
        parser.error("--groups must be positive")
    output = args.output.expanduser().absolute()
    plan = (
        args.plan.expanduser().absolute()
        if args.plan
        else output / "plan.yaml"
    )
    env = subprocess_env()
    config = "dual_arm_multi_perturb"
    if not args.plan:
        task_file = (
            args.tasks_file.expanduser().absolute()
            if args.tasks_file
            else ROBOTWIN / "task_config" / "plan_gen_list.yaml"
        )
        command = [
            sys.executable,
            "-m",
            "robotwin.script.generate_scene_plans",
            "--output",
            str(plan),
            "--scene-config",
            config,
            "--seed",
            str(args.seed),
            "--count",
            str(args.groups),
        ]
        if args.tasks:
            command += ["--tasks", args.tasks]
        else:
            command += ["--tasks-file", str(task_file)]
        _run(command, cwd=ROBOTWIN, dry_run=args.dry_run, env=env)
    if args.generate_only:
        return
    if not args.dry_run:
        require(ASSETS / "embodiments", "simulation assets")
        require(plan, "collection plan")
    command = [
        sys.executable,
        "-m",
        "robotwin.script.run_data_pipeline",
        "--plan",
        str(plan),
        "--gpus",
        ",".join(map(str, args.gpus)),
        "--out_dir",
        str(output / "scenes"),
        "--log_dir",
        str(output / "logs"),
        *extra,
    ]
    _run(command, cwd=ROBOTWIN, dry_run=args.dry_run, env=env,
         show_command=False)


def prepare_data():
    parser = argparse.ArgumentParser(
        description="Convert RoboTwin data or prepare training caches/initialization."
    )
    parser.add_argument("stage", choices=("convert", "text", "action-init"))
    parser.add_argument(
        "--model",
        choices=("action_only", "shared_ctx", "text_only"),
        default="action_only",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATA / "train_lerobot",
        help="Dataset to scan for text prompts",
    )
    parser.add_argument("--text-cache", type=Path, default=OUTPUTS / "cache/text")
    parser.add_argument(
        "--pretrained", type=Path, default=CHECKPOINTS / "Wan2.2-TI2V-5B"
    )
    parser.add_argument("--dry-run", action="store_true")
    args, extra = _arguments(parser)
    modules = {
        "convert": "convert_robotwin_to_lerobot",
        "text": "precompute_text_embeds",
        "action-init": "preprocess_action_dit_backbone",
    }
    command = [sys.executable, "-m", f"simpleicl.data.{modules[args.stage]}"]
    if args.stage == "text":
        from simpleicl.artifacts import hydra_value

        pretrained = args.pretrained.expanduser().absolute()
        command += [
            f"task={args.model}",
            f"model.model_id={hydra_value(str(pretrained))}",
            f"model.tokenizer_model_id={hydra_value(str(pretrained))}",
            "model.redirect_common_files=false",
            f"data.train.dataset_dirs={hydra_value([str(args.dataset.expanduser().absolute())])}",
            f"data.train.text_embedding_cache_dir={hydra_value(str(args.text_cache.expanduser().absolute()))}",
        ]
    elif args.stage == "action-init":
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
        from simpleicl.utils.config_resolvers import register_default_resolvers

        register_default_resolvers()
        if args.dry_run:
            config_path = OUTPUTS / "action_init_config.yaml"
        else:
            pretrained = _pretrained(args.pretrained)
            with initialize_config_dir(version_base="1.3", config_dir=str(CONFIGS)):
                cfg = compose(config_name="train", overrides=["task=shared_ctx"])
            cfg.model.model_id = str(pretrained)
            cfg.model.tokenizer_model_id = str(pretrained)
            cfg.model.redirect_common_files = False
            config_path = OUTPUTS / "action_init_config.yaml"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            OmegaConf.save(OmegaConf.to_container(cfg.model, resolve=True), config_path)
        command += [
            "--model-config",
            str(config_path),
            "--output",
            str(CHECKPOINTS / "action_init.pt"),
        ]
    command += extra
    _run(command, dry_run=args.dry_run)


def train():
    parser = argparse.ArgumentParser(
        description="Train a public SimpleICL model, optionally with DeepSpeed ZeRO-1."
    )
    parser.add_argument(
        "--model", choices=("action_only", "shared_ctx", "text_only"), required=True
    )
    parser.add_argument("--gpus", type=_gpus, default=[0])
    parser.add_argument("--dataset", type=Path, default=DATA / "train_lerobot")
    parser.add_argument(
        "--pretrained", type=Path, default=CHECKPOINTS / "Wan2.2-TI2V-5B"
    )
    parser.add_argument(
        "--action-init", type=Path, default=CHECKPOINTS / "action_init.pt"
    )
    parser.add_argument("--text-cache", type=Path, default=OUTPUTS / "cache" / "text")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args, extra = _arguments(parser)
    from simpleicl.artifacts import hydra_value

    dataset = args.dataset.expanduser().absolute()
    pretrained = args.pretrained.expanduser().absolute()
    init = args.action_init.expanduser().absolute()
    if not args.dry_run:
        require(dataset / "meta/info.json", "LeRobot dataset metadata")
        require(dataset / "dataset_stats.json", "dataset normalization statistics")
        require(init, "ActionDiT initialization (run prepare_data.py action-init)")
        require(
            args.text_cache.expanduser().absolute(),
            "text cache (run prepare_data.py text)",
        )
        _pretrained(pretrained)
    command = [sys.executable, "-m", "simpleicl.training"]
    if len(args.gpus) > 1:
        command = [
            sys.executable,
            "-m",
            "accelerate.commands.launch",
            "--use_deepspeed",
            "--num_processes",
            str(len(args.gpus)),
            "--num_machines",
            "1",
            "--zero_stage",
            "1",
            "--offload_optimizer_device",
            "none",
            "--offload_param_device",
            "none",
            "--mixed_precision",
            "bf16",
            "--module",
            "simpleicl.training",
        ]
    command += [
        f"task={args.model}",
        f"data.train.dataset_dirs={hydra_value([str(dataset)])}",
        f"data.train.pretrained_norm_stats={hydra_value([str(dataset / 'dataset_stats.json')])}",
        f"data.train.text_embedding_cache_dir={hydra_value(str(args.text_cache.expanduser().absolute()))}",
        f"model.model_id={hydra_value(str(pretrained))}",
        f"model.tokenizer_model_id={hydra_value(str(pretrained))}",
        "model.redirect_common_files=false",
        f"model.action_dit_pretrained_path={hydra_value(str(init))}",
    ]
    if args.output:
        command.append(
            f"output_dir={hydra_value(str(args.output.expanduser().absolute()))}"
        )
    if args.resume:
        command.append(
            f"resume={hydra_value(str(args.resume.expanduser().absolute()))}"
        )
    command += extra
    env = subprocess_env()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.gpus))
    env.setdefault("HF_HOME", str(OUTPUTS / "cache/huggingface"))
    # Preserve the source launcher's workaround for partial NUMA mappings.
    env.setdefault("NCCL_CUMEM_HOST_ENABLE", "0")
    _run(command, dry_run=args.dry_run, env=env)


def evaluate():
    parser = argparse.ArgumentParser(
        description="Evaluate a SimpleICL task recipe on the Native or OOD fixed-scene benchmark."
    )
    parser.add_argument(
        "--model", choices=("action_only", "shared_ctx", "text_only"), required=True
    )
    parser.add_argument("--benchmark", choices=("native", "ood"), required=False)
    parser.add_argument(
        "--stats",
        type=Path,
        help="Normalization statistics; default: checkpoint or training run directory",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--pretrained", type=Path, default=CHECKPOINTS / "Wan2.2-TI2V-5B"
    )
    parser.add_argument("--scene-dir", type=Path)
    parser.add_argument(
        "--task-list", type=Path,
        help="Optional task subset; default: discover all tasks in --scene-dir",
    )
    parser.add_argument("--gpus", type=_gpus, default=[0])
    parser.add_argument("--episodes-per-task", type=int)
    parser.add_argument("--visual-episode-seed", type=int, default=42)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args, extra = _arguments(parser)
    if args.benchmark is None and args.scene_dir is None:
        parser.error("provide --benchmark or --scene-dir")
    from simpleicl.artifacts import (
        hydra_value,
        read_task_list,
        resolve_model_artifacts,
    )
    from simpleicl.evaluation.scene_eval import discover_scene_cases

    artifacts = resolve_model_artifacts(args.model, args.checkpoint, args.stats)
    scene_dir = (
        (args.scene_dir or DATA / f"eval_{args.benchmark}").expanduser().absolute()
    )
    tasks = read_task_list(args.task_list) if args.task_list else None
    episodes = (
        args.episodes_per_task
        if args.episodes_per_task is not None
        else (20 if args.benchmark == "native" else 30)
    )
    if episodes < 1:
        parser.error("--episodes-per-task must be positive")
    if not args.dry_run:
        require(ASSETS / "embodiments", "simulation assets")
        _pretrained(args.pretrained)
        require(scene_dir, "benchmark scenes")
        found = discover_scene_cases(
            scene_dir,
            evaluation_embodiment="aloha-agilex",
            require_condition=args.model != "text_only",
            sample_visual_episode=True,
            visual_episode_seed=args.visual_episode_seed,
            task_names=tasks,
        )
        if tasks is None:
            tasks = sorted({case.episode.task_name for case in found})
            if not tasks:
                raise ValueError(f"No evaluation tasks found in benchmark scenes: {scene_dir}")
        counts = {
            task: sum(case.episode.task_name == task for case in found)
            for task in tasks
        }
        missing = {task: count for task, count in counts.items() if count < episodes}
        if any(count == 0 for count in counts.values()):
            raise ValueError(f"Benchmark tasks have no available cases: {missing}")
        if missing:
            warnings.warn(
                f"Episode limit is {episodes}; these tasks will use all available cases: {missing}",
                stacklevel=2,
            )
    command = [
        sys.executable,
        "-m",
        "simpleicl.evaluation.evaluate",
        f"task={args.model}",
        f"ckpt={hydra_value(str(artifacts.checkpoint))}",
        f"EVALUATION.result_name={args.model}",
        f"EVALUATION.robotwin_root={hydra_value(str(ROBOTWIN))}",
        f"EVALUATION.scene_dir={hydra_value(str(scene_dir))}",
        f"EVALUATION.task_names={hydra_value(tasks)}",
        f"EVALUATION.episodes_per_task={episodes}",
        f"EVALUATION.visual_episode_seed={args.visual_episode_seed}",
        "EVALUATION.embodiment=aloha-agilex",
        "EVALUATION.condition_embodiment=aloha-agilex",
        "EVALUATION.legacy_articulated_placement=false",
        "EVALUATION.attention_visualization.enabled=false",
        f"EVALUATION.dataset_stats_path={hydra_value(str(artifacts.stats))}",
        f"EVALUATION.pretrained_model_path={hydra_value(str(args.pretrained.expanduser().absolute()))}",
        f"gpu_id={args.gpus[0]}",
        f"MULTIRUN.enabled={str(len(args.gpus) > 1).lower()}",
        f"MULTIRUN.num_gpus={len(args.gpus)}",
        f"MULTIRUN.gpu_ids={hydra_value(args.gpus)}",
    ]
    if args.resume:
        command += [
            "EVALUATION.resume=true",
            f"EVALUATION.resume_dir={hydra_value(str(args.resume.expanduser().absolute()))}",
        ]
    _run(command + extra, dry_run=args.dry_run)
