"""Model weights, normalization statistics, and command-line configuration values."""

from dataclasses import dataclass
from pathlib import Path

from simpleicl.paths import CHECKPOINTS, require

MODELS = ("action_only", "shared_ctx", "text_only")


@dataclass(frozen=True)
class ModelArtifacts:
    checkpoint: Path
    stats: Path


def resolve_model_artifacts(
    model: str, checkpoint: Path | None = None, stats: Path | None = None
) -> ModelArtifacts:
    """Resolve weights and statistics; model settings come from configs/task."""
    if model not in MODELS:
        raise ValueError(f"Unknown model: {model}")
    checkpoint = (
        (checkpoint or CHECKPOINTS / model / "model.pt").expanduser().absolute()
    )
    if stats is None:
        # Training keeps statistics at the run root, above checkpoints/weights.
        candidates = [
            parent / "dataset_stats.json" for parent in list(checkpoint.parents)[:4]
        ]
        stats = next((path for path in candidates if path.is_file()), candidates[0])
    stats = stats.expanduser().absolute()
    return ModelArtifacts(
        require(checkpoint, "checkpoint"),
        require(stats, "normalization statistics (use --stats to set a path)"),
    )


def hydra_value(value) -> str:
    import json

    if isinstance(value, dict):
        return (
            "{"
            + ",".join(f"{key}:{hydra_value(item)}" for key, item in value.items())
            + "}"
        )
    if isinstance(value, list):
        return "[" + ",".join(hydra_value(item) for item in value) + "]"
    return json.dumps(value, ensure_ascii=False)


def read_task_list(path: Path) -> list[str]:
    import re

    text = require(path, "task list").read_text(encoding="utf-8-sig")
    if path.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        settings = yaml.safe_load(text)
        if isinstance(settings, dict):
            available = settings.get("tasks")
            values = settings.get("required_tasks", available)
            if not isinstance(available, list) or not isinstance(values, list):
                raise ValueError(f"Task settings must contain task lists: {path}")
            if any(task not in available for task in values):
                raise ValueError(f"required_tasks must be drawn from tasks: {path}")
        else:
            values = settings
        if not isinstance(values, list):
            raise ValueError(f"Expected a task list: {path}")
    else:
        values = [line.partition("#")[0].strip() for line in text.splitlines()]

    tasks = []
    for number, task in enumerate(values, 1):
        if not isinstance(task, str):
            raise ValueError(f"Invalid task at {path}:{number}: {task!r}")
        task = task.strip()
        if not task:
            continue
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", task):
            raise ValueError(f"Invalid task at {path}:{number}: {task!r}")
        if task not in tasks:
            tasks.append(task)
    if not tasks:
        raise ValueError(f"Empty task list: {path}")
    return tasks
