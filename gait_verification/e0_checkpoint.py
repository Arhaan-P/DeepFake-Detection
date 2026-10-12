"""Restricted extraction of the individually audited shipped historical E0 head.

Only this exact artifact is supported. No original training pickle is executed,
and no safe globals are added. The restricted loader remains unchanged.
"""

import hashlib
import io
from pathlib import Path
import pickletools
import zipfile

from .backends.checkpoint_extract import MODEL_GLOBALS
from .media import sha256_file
from .pipeline import _publish_exclusive

SHIPPED_SHA256 = "6c37f3b9b9172b42e6329a310f3e51d0b9882a5a9d59806851687731d67b1589"
PICKLE_SHA256 = "f031326216030ec2164d7600960ced35314be09958605634f645d0d4a5794200"
CONFIG_SHA256 = "bc21187735bd19aba051ca78c791ee538bc20a81f87d995c6d95899bd3604d1a"
SCHEMA = "frozen-e0-head-v1"
TRAINING_ORIGIN = "historical-local-13-subject-shipped"


def restricted_e0_pickle(data):
    """Audited static slice: epoch + full tensor state + tensor feature statistics.

    Optimizer/scheduler/NumPy metrics are omitted. Two memo references in statistics
    pointed to the optimizer's literal 'cpu' string; replace ONLY those exact opcodes
    with that same literal. Static disassembly verifies stack/memo consistency.
    """
    if hashlib.sha256(data).hexdigest() != PICKLE_SHA256:
        raise ValueError("Unaudited E0 pickle bytes")
    ops = {p: (o.name, a) for o, a, p in pickletools.genops(data)}
    expected = {
        22185: ("BUILD", None),
        22186: ("BINUNICODE", "optimizer_state_dict"),
        22289: ("BINUNICODE", "cpu"),
        22297: ("LONG_BINPUT", 1352),
        27785: ("BINUNICODE", "feature_stats"),
        27850: ("LONG_BINGET", 1352),
        27938: ("LONG_BINGET", 1352),
    }
    if any(ops.get(p) != v for p, v in expected.items()):
        raise ValueError("Unexpected E0 pickle structure")
    literal_cpu = data[22289:22297]
    safe = data[:22186] + data[27785:27850] + literal_cpu
    safe += data[27855:27938] + literal_cpu + data[27943:]
    globals_found = {a for o, a, _ in pickletools.genops(safe) if o.name == "GLOBAL"}
    if not globals_found <= MODEL_GLOBALS:
        raise ValueError("Non-tensor global in E0 inference section")
    pickletools.dis(safe, out=io.StringIO())
    return safe


def extract_e0(source, destination, expected_sha256):
    """Create a new tensor-only research artifact; never overwrite the source."""
    import torch

    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    if expected_sha256 != SHIPPED_SHA256 or sha256_file(source) != SHIPPED_SHA256:
        raise ValueError("E0 checkpoint is not an exact audited artifact")
    if source.stat().st_size != 4784940:
        raise ValueError("Audited E0 size mismatch")
    with zipfile.ZipFile(source) as archive:
        names = archive.namelist()
        pickles = [n for n in names if n.endswith("/data.pkl")]
        if len(pickles) != 1 or len(names) != len(set(names)) or archive.testzip():
            raise ValueError("Unexpected or corrupt E0 archive")
        name = pickles[0]
        safe = restricted_e0_pickle(archive.read(name))
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as clean:
            for info in archive.infolist():
                clean.writestr(
                    info, safe if info.filename == name else archive.read(info)
                )
        buffer.seek(0)
        obj = torch.load(buffer, map_location="cpu", weights_only=True)
    if type(obj) is not dict or set(obj) != {
        "epoch",
        "model_state_dict",
        "feature_stats",
    }:
        raise ValueError("Unexpected restricted E0 envelope")
    state, stats = obj["model_state_dict"], obj["feature_stats"]
    if len(state) != 127 or set(stats) != {"mean", "std"}:
        raise ValueError("Unexpected E0 state/statistics")
    for mapping in (state, stats):
        if not all(
            type(k) is str and type(v) is torch.Tensor and torch.isfinite(v).all()
            for k, v in mapping.items()
        ):
            raise ValueError("E0 inference tensors are invalid")
    if any(stats[k].shape != (78,) for k in stats) or not (stats["std"] > 0).all():
        raise ValueError("Invalid E0 normalization statistics")
    head = {}
    for k, v in state.items():
        if k.startswith("diff_conv."):
            head["head.conv." + k.removeprefix("diff_conv.")] = v.clone().contiguous()
        elif k.startswith("diff_classifier."):
            head["head.classifier." + k.removeprefix("diff_classifier.")] = (
                v.clone().contiguous()
            )
    if len(head) != 25:
        raise ValueError("Unexpected E0 decisive head")
    bundle = {
        "schema_version": SCHEMA,
        "model": head,
        "feature_stats": {k: v.clone().contiguous() for k, v in stats.items()},
        "config": {
            "input_dim": 78,
            "hidden": 64,
            "dropout": 0.1,
            "sequence_length": 60,
        },
        "training_origin": TRAINING_ORIGIN,
        "source_sha256": SHIPPED_SHA256,
        "source_config_sha256": CONFIG_SHA256,
    }
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    _publish_exclusive(lambda f: torch.save(bundle, f), destination)
    if sha256_file(source) != SHIPPED_SHA256:
        raise ValueError("Original E0 changed during extraction")
    return {
        "original_sha256": SHIPPED_SHA256,
        "pickle_sha256": PICKLE_SHA256,
        "model_only_sha256": sha256_file(destination),
        "head_parameters": 133058,
        "head_state_entries": len(head),
        "training_origin": TRAINING_ORIGIN,
        "safety": "audited-static-slice; unchanged weights_only=True; no allowlisting",
    }
