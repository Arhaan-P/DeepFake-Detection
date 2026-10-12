"""Bounded model-section extraction for individually audited official checkpoints.

Never deserialize the original training pickle. Only exact hash-pinned artifacts
are accepted. Static slicing closes the already complete first `model` value;
PyTorch's unchanged restricted loader is still the only deserializer. No custom
unpickler, safe-global additions, scheduler reconstruction or safety patching.
"""

from collections import OrderedDict
import hashlib
import io
from pathlib import Path
import pickletools
import shutil
import tempfile
import zipfile

from ..media import sha256_file

AUDITED = {
    "87eb74fa0b2ce14dcff025d6f41fd1d4637afafd3af2aa0c6a06a690ff52a20d": {
        "dataset": "CASIA-B",
        "bytes": 63712570,
        "model_end": 173246,
        "pickle_sha256": "677def1465de52ea29f9b405d568ff1dcb8879ed504024ae4b7c4228f1f11a23",
        "state_entries": 1010,
    },
    # GPGait++_GREW.pt, Drive ID 1PV3MnXIyrehx8TFHFxo7_yY4y1ItT7-Y; all 8041 ZIP
    # CRC-32s pass; static audit: tensor-only globals before `optimizer`.
    "eb37d274640d59665e500f8881811b3effd285302f5e0b9164ae3e22cb74e8d7": {
        "dataset": "GREW",
        "bytes": 1308128306,
        "model_end": 423713,
        "pickle_sha256": "7b1f91ab30af7c4e2241beeea73f34d97f68cd722bfa46817a06635090a08c52",
        "state_entries": 2570,
    },
}
MODEL_GLOBALS = {
    "collections OrderedDict",
    "torch._utils _rebuild_tensor_v2",
    "torch FloatStorage",
    "torch LongStorage",
}


def static_pickle_audit(path):
    """Describe a torch ZIP checkpoint's pickle without executing any opcode.

    Produces the facts an AUDITED entry pins (pickle hash, candidate model
    boundary, globals). A human must review the output before adding an entry.
    """
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        pickles = [n for n in names if n.endswith("/data.pkl")]
        if len(pickles) != 1 or len(names) != len(set(names)):
            raise ValueError("Unexpected checkpoint archive")
        data = archive.read(pickles[0])
    ops = list(pickletools.genops(data))
    strings = [
        (p, a)
        for o, a, p in ops
        if o.name in ("BINUNICODE", "SHORT_BINUNICODE")
        and a in ("model", "optimizer", "scheduler", "iteration")
    ]
    boundary = next(
        (
            i
            for i, (o, a, _) in enumerate(ops)
            if o.name == "BINUNICODE"
            and a == "optimizer"
            and ops[i - 1][0].name == "BUILD"
        ),
        None,
    )
    model_end = None if boundary is None else ops[boundary][2]
    model_globals = (
        None
        if model_end is None
        else sorted({a for o, a, p in ops if o.name == "GLOBAL" and p < model_end})
    )
    return dict(
        bytes=path.stat().st_size,
        zip_members=len(names),
        pickle_member=pickles[0],
        pickle_bytes=len(data),
        pickle_sha256=hashlib.sha256(data).hexdigest(),
        head=[(o.name, a) for o, a, _ in ops[:5]],
        key_strings=strings[:16],
        all_globals=sorted({a for o, a, _ in ops if o.name == "GLOBAL"}),
        candidate_model_end=model_end,
        model_section_globals=model_globals,
        model_section_globals_allowed=(
            None if model_globals is None else set(model_globals) <= MODEL_GLOBALS
        ),
    )


def restricted_model_pickle(data, spec):
    if hashlib.sha256(data).hexdigest() != spec["pickle_sha256"]:
        raise ValueError("Unaudited pickle bytes")
    end = spec["model_end"]
    ops = list(
        pickletools.genops(data)
    )  # Static parsing; never evaluates GLOBAL/REDUCE.
    boundary = next((i for i, (_, _, p) in enumerate(ops) if p == end), None)
    if (
        boundary is None
        or ops[boundary][0].name != "BINUNICODE"
        or ops[boundary][1] != "optimizer"
    ):
        raise ValueError("Unexpected model-section boundary")
    if [(o.name, a) for o, a, _ in ops[:5]] != [
        ("PROTO", 2),
        ("EMPTY_DICT", None),
        ("BINPUT", 0),
        ("MARK", None),
        ("BINUNICODE", "model"),
    ] or ops[boundary - 1][0].name != "BUILD":
        raise ValueError("Unexpected checkpoint structure")
    prefix = data[:end] + b"u."  # SETITEMS closes root {model: state}; STOP.
    globals_found = {a for o, a, _ in pickletools.genops(prefix) if o.name == "GLOBAL"}
    if not globals_found <= MODEL_GLOBALS:
        raise ValueError("Non-tensor global in model section")
    # Checks stack balance/memo references; does not execute any pickle instructions.
    pickletools.dis(prefix, out=io.StringIO())
    return prefix


def extract_model_only(source, destination, expected_sha256):
    import torch

    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    checksum = sha256_file(source)
    if checksum != expected_sha256 or checksum not in AUDITED:
        raise ValueError("Checkpoint is not an exact audited artifact")
    spec = AUDITED[checksum]
    if source.stat().st_size != spec["bytes"]:
        raise ValueError("Checkpoint size mismatch")
    with zipfile.ZipFile(source) as archive, tempfile.TemporaryDirectory() as work:
        names = archive.namelist()
        pickles = [n for n in names if n.endswith("/data.pkl")]
        if len(pickles) != 1 or len(names) != len(set(names)):
            raise ValueError("Unexpected checkpoint archive")
        name = pickles[0]
        prefix = restricted_model_pickle(archive.read(name), spec)
        # Preserve raw storage bytes; the temporary archive contains no training
        # object pickle. Unreferenced storage bytes are inert and never executed.
        temporary = Path(work) / "restricted.pt"
        with zipfile.ZipFile(temporary, "w") as clean:
            for info in archive.infolist():
                if info.filename == name:
                    clean.writestr(info, prefix)
                else:
                    with (
                        archive.open(info) as src,
                        clean.open(info, "w", force_zip64=True) as dst,
                    ):
                        shutil.copyfileobj(src, dst, length=1024 * 1024)
        obj = torch.load(temporary, map_location="cpu", weights_only=True)
        if type(obj) is not dict or set(obj) != {"model"}:
            raise ValueError("Unexpected model envelope")
        state = obj["model"]
        if (
            not isinstance(state, (dict, OrderedDict))
            or len(state) != spec["state_entries"]
        ):
            raise ValueError("Unexpected model state count")
        if not all(
            type(k) is str and type(v) is torch.Tensor for k, v in state.items()
        ):
            raise ValueError("Model contains non-tensors")
        # Clone to remove storage aliases and strip non-tensor _metadata attributes.
        state = {k: v.clone().contiguous() for k, v in state.items()}
        with open(destination, "xb") as f:
            torch.save({"model": state}, f)
    return dict(
        original_sha256=checksum,
        original_bytes=source.stat().st_size,
        dataset=spec["dataset"],
        model_end=spec["model_end"],
        pickle_sha256=spec["pickle_sha256"],
        state_entries=len(state),
        model_only_sha256=sha256_file(destination),
        model_only_bytes=destination.stat().st_size,
        extraction="audited-static-prefix-v1; unchanged weights_only=True; tensor-only resave",
    )
