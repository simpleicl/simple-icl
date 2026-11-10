#!/usr/bin/env python3
"""
Usage: conda activate simpleicl-sim; python scripts/verify_environment.py
Prerequisites: staged installation complete; NVIDIA GPU, Vulkan and FFmpeg available.
"""

import importlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

from install import REQUIREMENTS, ROOT, load_patch_baseline, require_environment, verify_patches, verify_pins


def main():
    require_environment()
    verify_pins(("build", "torch", "python", "native"))
    verify_patches(load_patch_baseline(json.loads((REQUIREMENTS / "baseline.json").read_text())))
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
    for package in ("simpleicl", "robotwin"):
        spec = importlib.util.find_spec(package)
        expected = ROOT / "src" / package / "__init__.py"
        if spec is None or not spec.origin or Path(spec.origin).resolve() != expected:
            raise RuntimeError(f"{package} is not imported from this simulation checkout")
    print("PASS dependency versions, provenance and original third-party source", flush=True)

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot use the CUDA device")
    x = torch.ones((16, 16), device="cuda")
    if not torch.all((x @ x) == 16).item():
        raise RuntimeError("CUDA matrix multiplication failed")
    torch.cuda.synchronize()
    for extension in ("lbfgs_step_cu", "kinematics_fused_cu", "line_search_cu", "tensor_step_cu", "geom_cu"):
        importlib.import_module(f"curobo.curobolib.{extension}")
    importlib.import_module("deepspeed")
    importlib.import_module("mplib")
    importlib.import_module("open3d")
    print("PASS CUDA operation and native extension imports (planning/training checked separately)", flush=True)

    import numpy as np
    import sapien.core as sapien

    engine = sapien.Engine()
    renderer = sapien.SapienRenderer()
    engine.set_renderer(renderer)
    sapien.render.set_camera_shader_dir("rt")
    sapien.render.set_ray_tracing_samples_per_pixel(4)
    scene = engine.create_scene(sapien.SceneConfig())
    scene.add_ground(0)
    scene.set_ambient_light([0.5, 0.5, 0.5])
    builder = scene.create_actor_builder()
    builder.add_box_visual(half_size=[0.2, 0.2, 0.2], material=[1, 0, 0])
    actor = builder.build_static(name="installation-check")
    actor.set_pose(sapien.Pose([0, 0, 0.5]))
    camera = scene.add_camera("installation-check", 64, 64, math.pi / 3, 0.1, 10)
    camera.entity.set_pose(sapien.Pose([-2, 0, 0.5]))
    scene.update_render()
    camera.take_picture()
    frame = camera.get_picture("Color")
    if frame.shape != (64, 64, 4) or not np.isfinite(frame).all() or frame[..., :3].std() <= 0:
        raise RuntimeError("SAPIEN did not render a valid nonuniform image")
    print("PASS SAPIEN headless ray-traced image", flush=True)

    import av
    from simpleicl.datasets.lerobot.lerobot.datasets.video_utils import decode_video_frames

    with tempfile.TemporaryDirectory(prefix="simpleicl-video-") as directory:
        for codec, suffix in (("libx264", "h264"), ("libaom-av1", "av1")):
            path = Path(directory) / f"{suffix}.mp4"
            subprocess.run([
                "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=32x32:rate=4",
                "-frames:v", "4", "-c:v", codec, "-pix_fmt", "yuv420p", str(path),
            ], check=True, timeout=60)
            with av.open(str(path)) as video:
                if len(list(video.decode(video=0))) != 4:
                    raise RuntimeError(f"PyAV failed to decode {suffix}")
            for backend in ("pyav", None):
                frames = decode_video_frames(path, [0.0, 0.25, 0.5, 0.75], 0.01, backend=backend)
                if tuple(frames.shape) != (4, 3, 32, 32) or not torch.isfinite(frames).all().item():
                    raise RuntimeError(f"Project video decoder failed for {suffix}, backend={backend}")
    print("PASS FFmpeg H.264/AV1 encoding and project decoding (including existing PyAV fallback)", flush=True)
    print("Environment smoke checks passed. Run the workflow acceptance steps in INSTALL.md.")


if __name__ == "__main__":
    main()
