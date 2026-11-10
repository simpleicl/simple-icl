#!/usr/bin/env python3
"""
Usage: /path/to/.venv/bin/python inspect_environment.py > environment.json
"""

import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


TARGETS = {
    "mplib": ("0.2.1", "mplib/planner.py"),
    "sapien": ("3.0.0b1", "sapien/wrapper/urdf_loader.py"),
}
MPLIB_ORIGINAL = "if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:"
MPLIB_PATCHED = "if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:"
SAPIEN_ORIGINAL = ['open(urdf_file, "r")', 'open(srdf_file, "r")']
SAPIEN_PATCHED = [text[:-1] + ', encoding="utf-8")' for text in SAPIEN_ORIGINAL]


def patch_state(package, text):
    if package == "mplib":
        if text.count(MPLIB_ORIGINAL) == 1 and MPLIB_PATCHED not in text:
            return "original"
        if text.count(MPLIB_PATCHED) == 1 and MPLIB_ORIGINAL not in text:
            return "robotwin"
    if package == "sapien":
        if all(text.count(s) == 1 for s in SAPIEN_ORIGINAL) and not any(
            s in text for s in SAPIEN_PATCHED
        ):
            return "original"
        if all(text.count(s) == 1 for s in SAPIEN_PATCHED) and not any(
            s in text for s in SAPIEN_ORIGINAL
        ):
            return "robotwin"
    return "unknown"


def inspect_patches():
    result = {}
    for package, (expected_version, relative_path) in TARGETS.items():
        try:
            dist = metadata.distribution(package)
            path = Path(dist.locate_file(relative_path))
            content = path.read_bytes()
            text = content.decode("utf-8")
            result[package] = {
                "version": dist.version,
                "state": patch_state(package, text),
                "sha256": hashlib.sha256(content).hexdigest(),
                "supported_version": dist.version == expected_version,
                "excerpts": [
                    line.strip() for line in text.splitlines()
                    if "delta_twist) <" in line or "open(urdf_file" in line
                    or "open(srdf_file" in line or "srdf_file = urdf_file" in line
                ],
            }
        except (metadata.PackageNotFoundError, OSError, UnicodeError) as exc:
            result[package] = {"state": "unknown", "error": str(exc)}
    return result


def command_output(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        return {"returncode": result.returncode, "stdout": result.stdout,
                "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}


def main():
    packages = []
    for dist in metadata.distributions():
        item = {"name": dist.metadata["Name"], "version": dist.version}
        direct_url = dist.read_text("direct_url.json")
        if direct_url:
            item["direct_url"] = json.loads(direct_url)
        packages.append(item)
    report = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "os_release": platform.freedesktop_os_release() if sys.platform == "linux" else {},
        "patches": inspect_patches(),
        "packages": sorted(packages, key=lambda item: item["name"].lower()),
        "build_environment": {key: os.environ.get(key) for key in (
            "CUDA_HOME", "CC", "CXX", "TORCH_CUDA_ARCH_LIST", "DS_BUILD_OPS", "MAX_JOBS"
        )},
        "nvcc": command_output(["nvcc", "--version"]),
        "gpu": command_output(["nvidia-smi", "--query-gpu=name,driver_version",
                               "--format=csv,noheader"]),
        "ffmpeg": command_output(["ffmpeg", "-version"]),
        "vulkan": command_output(["vulkaninfo", "--summary"]),
        "pip_check": command_output([sys.executable, "-m", "pip", "check"]),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
