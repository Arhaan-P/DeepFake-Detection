"""Audited official GREW variants; no upstream training entry point is invoked."""

from pathlib import Path
import subprocess

REVISION = "d3ce631971a01d806eeb3a154774d482c0951813"
CONFIGS = {
    "gpgait": "configs/gpgait/gpgait_GREW.yaml",
    "gpgaitpp": "configs/gpgait++/gpgait++_GREW.yaml",
    "gpgaitpp_casia": "configs/gpgait++/gpgait++.yaml",
}


def validate_source(source):
    source = Path(source).resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain"], text=True
    ).strip()
    if revision != REVISION or dirty:
        raise ValueError("Requires clean audited FastPoseGait revision")
    return source


def official_config(source, variant):
    import yaml

    if variant not in CONFIGS:
        raise ValueError("Unsupported FastPoseGait variant")
    config = yaml.safe_load((Path(source) / CONFIGS[variant]).read_text())
    dataset = "CASIA-B" if variant == "gpgaitpp_casia" else "GREW"
    if config["data_cfg"]["dataset_name"] != dataset:
        raise ValueError("Wrong pretrained dataset config")
    return config["model_cfg"]
