#!/usr/bin/env python3
"""
Usage: conda activate simpleicl-sim; python scripts/install.py --stage all
       python scripts/install.py --stage all --dry-run
"""

import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sys
from urllib.parse import unquote, urlparse

from inspect_environment import TARGETS, patch_state

ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = ROOT / "requirements"


def require_environment():
    prefix = Path(sys.prefix).resolve()
    venv = sys.prefix != sys.base_prefix
    conda = (prefix / "conda-meta").is_dir() and (
        os.environ.get("CONDA_PREFIX") == sys.prefix
        and os.environ.get("CONDA_DEFAULT_ENV") != "base"
    )
    if not (venv or conda):
        raise RuntimeError("Activate a dedicated simpleicl-sim environment first.")
    venv_config = prefix / "pyvenv.cfg"
    if venv_config.is_file() and re.search(
        r"include-system-site-packages\s*=\s*true", venv_config.read_text(), re.IGNORECASE
    ):
        raise RuntimeError("Create a simulation environment without system site packages.")
    if sys.version_info[:2] != (3, 10) or sys.platform != "linux" or platform.machine() != "x86_64":
        raise RuntimeError("This baseline targets Linux x86_64 and Python 3.10 only.")
    try:
        dist = metadata.distribution("simpleicl")
    except metadata.PackageNotFoundError:
        return
    source = json.loads(dist.read_text("direct_url.json") or "{}")
    parsed = urlparse(source.get("url", ""))
    if parsed.scheme != "file" or Path(unquote(parsed.path)).resolve() != ROOT:
        raise RuntimeError("Another simpleicl is installed. Create a fresh simulation environment.")


def pins(stage):
    for line in (REQUIREMENTS / f"{stage}.txt").read_text().splitlines():
        if line and not line.startswith(("#", "--")):
            yield line


def verify_pins(stages):
    for stage in stages:
        for pin in pins(stage):
            if " @ " in pin:
                name, source = pin.split(" @ ", 1)
                dist = metadata.distribution(name)
                direct = json.loads(dist.read_text("direct_url.json") or "{}")
                repository, commit = source.removeprefix("git+").rsplit("@", 1)
                if direct.get("url") != repository or direct.get("vcs_info", {}).get("commit_id") != commit:
                    raise RuntimeError(f"Unexpected source revision for {name}: {direct}")
            else:
                name, expected = pin.split("==", 1)
                dist = metadata.distribution(name)
                if dist.version != expected:
                    raise RuntimeError(f"{name}: expected {expected}, found {dist.version}")
            if not Path(dist.locate_file("")).resolve().is_relative_to(Path(sys.prefix).resolve()):
                raise RuntimeError(f"{name} is loaded from outside the active environment")


def load_patch_baseline(baseline):
    patches = baseline.get("patches")
    if not isinstance(patches, dict) or set(patches) != set(TARGETS):
        raise RuntimeError(
            "SAPIEN/MPLib baseline patches are not confirmed. Run inspect_environment.py "
            "in the working environment and record its patches in requirements/baseline.json. "
            "The base and native stages can be prepared independently."
        )
    for package, (version, _) in TARGETS.items():
        record = patches[package]
        if record.get("version") != version or record.get("state") not in {"original", "robotwin"}:
            raise RuntimeError(f"Unsupported patch baseline for {package}: {record}")
        if not re.fullmatch(r"[0-9a-f]{64}", record.get("sha256", "")):
            raise RuntimeError(f"Missing reference file hash for {package}")
    return patches


def verify_patches(patches):
    # The working environment uses unmodified SAPIEN and MPLib. Never patch them.
    for package, (version, relative_path) in TARGETS.items():
        dist = metadata.distribution(package)
        if dist.version != version:
            raise RuntimeError(f"Unsupported {package} {dist.version}")
        path = Path(dist.locate_file(relative_path)).resolve()
        if not path.is_relative_to(Path(sys.prefix).resolve()):
            raise RuntimeError(f"Source file is outside this environment: {path}")
        content = path.read_bytes()
        expected = patches[package]
        if (hashlib.sha256(content).hexdigest() != expected["sha256"]
                or patch_state(package, content.decode("utf-8")) != expected["state"]):
            raise RuntimeError(
                f"{package} source differs from the working baseline. "
                "Reinstall its pinned wheel in a fresh environment; no patch was applied."
            )


def run(command, dry_run):
    print(shlex.join(map(str, command)), flush=True)
    if not dry_run:
        env = os.environ.copy()
        env["PYTHONNOUSERSITE"] = "1"
        env.pop("PYTHONPATH", None)
        env.setdefault("MAX_JOBS", "2")
        subprocess.run(command, cwd=ROOT, env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "base", "native", "project", "check"), default="all")
    parser.add_argument("--dry-run", action="store_true", help="Print steps without installing, importing GPU code, or editing packages")
    args = parser.parse_args()
    baseline = json.loads((REQUIREMENTS / "baseline.json").read_text())
    stages = ("base", "native", "project", "check") if args.stage == "all" else (args.stage,)
    patches = None
    if not args.dry_run:
        require_environment()
        if any(stage in stages for stage in ("project", "check")):
            patches = load_patch_baseline(baseline)
        if "native" in stages:
            missing = [tool for tool in ("git", "nvcc", "c++", "ffmpeg", "vulkaninfo", "nvidia-smi") if not shutil.which(tool)]
            if missing:
                raise RuntimeError(f"Install system prerequisites first: {', '.join(missing)}")
            cuda_home = os.environ.get("CUDA_HOME")
            if cuda_home and (Path(cuda_home) / "bin/nvcc").resolve() != Path(shutil.which("nvcc")).resolve():
                raise RuntimeError("CUDA_HOME and nvcc on PATH must select the same toolkit.")
    elif baseline.get("patches") is None:
        print("NOTE: patch baseline pending; an actual all/project/check run will stop.")
    pip = [sys.executable, "-m", "pip", "--isolated", "install", "--no-user", "--no-deps"]
    for stage in stages:
        print(f"\nStage: {stage}", flush=True)
        if stage == "base":
            for group in ("build", "torch", "python"):
                run([*pip, "-r", str(REQUIREMENTS / f"{group}.txt")], args.dry_run)
        elif stage == "native":
            if not args.dry_run:
                verify_pins(("build", "torch", "python"))
            run([*pip, "--no-build-isolation", "--no-cache-dir", "--force-reinstall",
                 "--no-binary=deepspeed", "-r", str(REQUIREMENTS / "native.txt")], args.dry_run)
        elif stage == "project":
            if not args.dry_run:
                verify_pins(("build", "torch", "python", "native"))
                verify_patches(patches)
            else:
                print("Verify original SAPIEN/MPLib source hashes; do not apply upstream patches.")
            run([*pip, "--no-build-isolation", "-e", str(ROOT)], args.dry_run)
        elif stage == "check":
            if not args.dry_run:
                verify_pins(("build", "torch", "python", "native"))
                verify_patches(patches)
                # Requires the local package to exist, not just its dependencies.
                metadata.distribution("simpleicl")
            run([sys.executable, "-m", "pip", "check"], args.dry_run)
    if not args.dry_run:
        print("Requested stages completed.")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, metadata.PackageNotFoundError, OSError, subprocess.CalledProcessError) as exc:
        sys.exit(str(exc))
