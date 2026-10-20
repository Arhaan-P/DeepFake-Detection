"""Optional two-pass derivatives. Originals and existing artifacts are never overwritten."""

from dataclasses import dataclass, asdict
from pathlib import Path
import os
import subprocess
import numpy as np
from .media import probe_video, metadata_dict, sha256_file
from .capture_registry import canonical_sha256, write_new_json


@dataclass(frozen=True)
class AugmentationConfig:
    """One conservative spatial/encoding perturbation; never temporal remapping.

    FFmpeg eq brightness/contrast, gblur sigma, or libx264 CRF. Units and
    invocation are recorded verbatim. Bounds are engineering limits, not evidence
    that any perturbation improves pose or verification. No composition is allowed.
    """

    kind: str
    value: float
    recipe_version: str = "controlled-ffmpeg-v1"
    enabled: bool = True

    def validate(self):
        bounds = {
            "brightness": (-0.1, 0.1),
            "contrast": (0.85, 1.15),
            "blur": (0.1, 1.0),
            "compression": (23, 32),
        }
        if (
            self.kind not in bounds
            or self.recipe_version != "controlled-ffmpeg-v1"
            or self.enabled is not True
        ):
            raise ValueError("Unsupported controlled augmentation")
        lo, hi = bounds[self.kind]
        if (
            type(self.value) not in (int, float)
            or not np.isfinite(self.value)
            or not lo <= self.value <= hi
        ):
            raise ValueError("Augmentation outside conservative engineering bounds")
        if self.kind == "compression" and int(self.value) != self.value:
            raise ValueError("Compression requires integer CRF")

    @property
    def variant_id(self):
        self.validate()
        return f"{self.kind}-{canonical_sha256(asdict(self))[:16]}"


def augmentation_commands(source, output, config):
    config.validate()
    value = format(config.value, ".12g")
    filters = {
        "brightness": f"eq=brightness={value}",
        "contrast": f"eq=contrast={value}",
        "blur": f"gblur=sigma={value}",
        "compression": "null",
    }
    return [
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-n",
            "-i",
            str(Path(source).resolve()),
            "-map",
            "0:v:0",
            "-vf",
            filters[config.kind],
            "-fps_mode",
            "passthrough",
            "-enc_time_base",
            "-1",
            "-c:v",
            "libx264",
            "-threads",
            "1",
            "-crf",
            value if config.kind == "compression" else "17",
            "-preset",
            "medium",
            "-pix_fmt",
            "yuv420p",
            "-map_metadata",
            "-1",
            "-an",
            str(Path(output).resolve()),
        ]
    ]


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
    work.mkdir(parents=True, exist_ok=False, mode=0o700)
    output = work / "stabilized.mp4"
    controlled = isinstance(config, AugmentationConfig)
    commands = (
        (
            augmentation_commands(source, output, config)
            if controlled
            else stabilization_commands(source, output, config)
        )
        if config.enabled
        else []
    )
    manifest = {
        "schema_version": "video-variant-v1",
        "recording": asdict(recording),
        "variant_id": config.variant_id
        if controlled
        else (f"vidstab-s{config.smoothing}" if config.enabled else "raw"),
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
                fd = os.open(
                    work / f"pass{i + 1}.log",
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                )
                with os.fdopen(fd, "wb") as log:
                    subprocess.run(
                        command, cwd=work, stdout=log, stderr=log, check=True
                    )
            derivative = probe_video(output)
            if not controlled:
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
        for artifact in (output, work / "transforms.trf"):
            if artifact.is_file():
                os.chmod(artifact, 0o600)
        write_new_json(work / "manifest.json", manifest)
    return manifest
