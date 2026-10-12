"""Optional two-pass derivatives. Originals and existing artifacts are never overwritten."""

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import subprocess
import numpy as np
from .media import probe_video, metadata_dict, sha256_file


@dataclass(frozen=True)
class StabilizationConfig:
    enabled: bool = False
    shakiness: int = 5
    accuracy: int = 15
    mincontrast: float = 0.2
    smoothing: int = 10
    optzoom: int = 0
    crop: str = "black"
    interpol: str = "bicubic"

    def validate(self):
        if (
            self.smoothing not in (10, 15)
            or self.optzoom != 0
            or self.crop != "black"
            or self.interpol != "bicubic"
        ):
            raise ValueError(
                "Phase 0 allows only declared gentle smoothing=10/15 recipes"
            )
        if (self.shakiness, self.accuracy, self.mincontrast) != (5, 15, 0.2):
            raise ValueError("Undeclared detection recipe")


def stabilization_commands(source, output, config):
    config.validate()
    detect = f"vidstabdetect=shakiness={config.shakiness}:accuracy={config.accuracy}:mincontrast={config.mincontrast}:result=transforms.trf"
    transform = f"vidstabtransform=input=transforms.trf:smoothing={config.smoothing}:optzoom=0:crop=black:interpol=bicubic"
    common = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-n",
        "-i",
        str(Path(source).resolve()),
    ]
    return [
        common + ["-vf", detect, "-an", "-f", "null", "-"],
        common
        + [
            "-vf",
            transform,
            "-fps_mode",
            "passthrough",
            "-enc_time_base",
            "-1",
            "-c:v",
            "libx264",
            "-crf",
            "17",
            "-preset",
            "medium",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(Path(output).resolve()),
        ],
    ]


def stabilize(source, work_dir, recording, config=StabilizationConfig()):
    config.validate()
    source = Path(source).resolve(strict=True)
    original = probe_video(source)
    if original.sha256 != recording.source_sha256 or recording.variant_id != "raw":
        raise ValueError("Stabilization requires the declared immutable original")
    if original.rotation % 90:
        raise ValueError("Only right-angle display rotations are supported")
    work = Path(work_dir).resolve()
    # Dedicated new work directory prevents sidecar collision and overwrite.
    work.mkdir(parents=True, exist_ok=False)
    output = work / "stabilized.mp4"
    commands = stabilization_commands(source, output, config) if config.enabled else []
    manifest = {
        "schema_version": "video-variant-v1",
        "recording": asdict(recording),
        "variant_id": f"vidstab-s{config.smoothing}" if config.enabled else "raw",
        "config": asdict(config),
        "input": metadata_dict(original),
        "output": None,
        "commands": commands,
        "status": "STARTED",
        "ffmpeg_version": subprocess.check_output(
            ["ffmpeg", "-version"], text=True
        ).splitlines()[0],
        "timing_transform": None,
        "orientation_transform": f"FFmpeg display autorotation {original.rotation} degrees in BOTH passes",
        "quality_review": "NOT_ASSESSED: review black borders and feet",
        "transformation_sha256": None,
    }
    try:
        if config.enabled:
            for i, command in enumerate(commands):
                with open(work / f"pass{i + 1}.log", "xb") as log:
                    subprocess.run(
                        command, cwd=work, stdout=log, stderr=log, check=True
                    )
            derivative = probe_video(output)
            manifest["transformation_sha256"] = sha256_file(work / "transforms.trf")
            manifest["output"] = metadata_dict(derivative)
            a, b = np.asarray(original.timestamps), np.asarray(derivative.timestamps)
            # MP4 may shift origin and quantize timestamps. Validate every frame.
            if len(a) != len(b) or not np.allclose(
                a - a[0], b - b[0], atol=0.0001, rtol=0
            ):
                raise ValueError("Derivative does not preserve frame timing")
            display_size = (
                (original.height, original.width)
                if original.rotation % 180
                else (original.width, original.height)
            )
            if (
                display_size != (derivative.width, derivative.height)
                or derivative.rotation % 360
            ):
                raise ValueError("Unexpected derivative dimensions")
            manifest["output"] = metadata_dict(derivative)
            manifest["timing_transform"] = {
                "pts_offset_sec": float(b[0] - a[0]),
                "max_relative_pts_error_sec": float(
                    np.max(np.abs((a - a[0]) - (b - b[0])))
                ),
            }
        else:
            manifest["output"] = metadata_dict(original)
            manifest["timing_transform"] = {
                "pts_offset_sec": 0,
                "max_relative_pts_error_sec": 0,
            }
        if sha256_file(source) != original.sha256:
            raise ValueError("Source changed during processing")
        manifest["status"] = "OK"
    except Exception as exc:
        manifest["status"] = "FAILED"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        with open(work / "manifest.json", "x") as f:
            json.dump(manifest, f, indent=2)
    return manifest
