"""Locations of packaged code and user-provided resources."""

import os
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
CONFIGS = PACKAGE / "configs"
ROBOTWIN = PACKAGE.parent / "robotwin"
_source_root = PACKAGE.parents[1]
ROOT = (
    Path(
        os.environ.get(
            "SIMPLEICL_ROOT",
            str(
                _source_root
                if (_source_root / "pyproject.toml").is_file()
                else Path.cwd()
            ),
        )
    )
    .expanduser()
    .absolute()
)
ASSETS = (
    Path(os.environ.get("SIMPLEICL_ASSETS", ROOT / "assets")).expanduser().absolute()
)
CHECKPOINTS = (
    Path(os.environ.get("SIMPLEICL_CHECKPOINTS", ROOT / "checkpoints"))
    .expanduser()
    .absolute()
)
DATA = Path(os.environ.get("SIMPLEICL_DATA", ROOT / "data")).expanduser().absolute()
OUTPUTS = (
    Path(os.environ.get("SIMPLEICL_OUTPUTS", ROOT / "outputs")).expanduser().absolute()
)


def resource_path(path: str) -> Path:
    """Resolve legacy RoboTwin asset references without depending on cwd."""
    value = Path(path).expanduser()
    if value.is_absolute():
        return value
    if value.parts and value.parts[0] == "assets":
        return ASSETS.joinpath(*value.parts[1:])
    return ROOT / value


def subprocess_env() -> dict[str, str]:
    """Keep subprocess imports within this installation, including RoboTwin."""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(PACKAGE.parent), str(ROBOTWIN)))
    env["SIMPLEICL_ROOT"] = str(ROOT)
    env["PYTHONNOUSERSITE"] = "1"
    return env


def require(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {label}: {path} (check the path or symlink target)"
        )
    return path
