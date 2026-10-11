"""Per-take templates. The store API deliberately exposes only direct claim lookup."""

from dataclasses import dataclass
from typing import Protocol
import numpy as np
from .contracts import GaitEmbedding, Recording, valid_claim


@dataclass(frozen=True)
class EnrollmentTemplate:
    claimed_subject_id: str
    embedding: GaitEmbedding
    recording: Recording
    quality_weight: float = 1.0


def validate_templates(claim, templates):
    seen = set()
    for t in templates:
        if t.claimed_subject_id != claim or t.recording.subject_id != claim:
            raise ValueError("Enrollment belongs to a different claim")
        if t.embedding.original_take_id != t.recording.original_take_id:
            raise ValueError("Enrollment lineage mismatch")
        if not t.embedding.source_sha256 or not t.recording.session_id:
            raise ValueError("Missing enrollment provenance")
        if t.recording.original_take_id in seen:
            raise ValueError(
                "Repeated windows/derivatives are not independent enrollments"
            )
        seen.add(t.recording.original_take_id)
        if not np.isfinite(t.quality_weight) or t.quality_weight <= 0:
            raise ValueError("Invalid quality weight")
        t.embedding.normalized()
    if len({t.embedding.model_id for t in templates}) > 1:
        raise ValueError("Mixed enrollment recipes")


class EnrollmentStore(Protocol):
    def get_claim(self, claimed_subject_id: str) -> tuple[EnrollmentTemplate, ...]: ...


class MemoryEnrollmentStore:
    def __init__(self):
        self._records = {}

    def enroll(self, claim, templates):
        if not valid_claim(claim):
            raise ValueError("Invalid claim syntax")
        templates = tuple(templates)
        validate_templates(claim, templates)
        self._records[claim] = templates

    def get_claim(self, claimed_subject_id):
        return self._records.get(claimed_subject_id, ())


def prototype(templates):
    z = np.stack([t.embedding.normalized() for t in templates])
    out = np.average(z, axis=0, weights=[t.quality_weight for t in templates])
    norm = np.linalg.norm(out)
    if norm < 1e-12:
        raise ValueError("Degenerate enrollment prototype")
    return out / norm


class DirectoryEnrollmentStore:
    """Private JSON artifact per hashed claim; never enumerate/load other claims."""

    def __init__(self, root):
        from pathlib import Path

        self.root = Path(root)

    def _path(self, claim):
        import hashlib

        if not valid_claim(claim):
            raise ValueError("Invalid claim syntax")
        return self.root / (hashlib.sha256(claim.encode()).hexdigest() + ".json")

    def enroll(self, claim, templates):
        from dataclasses import asdict
        import json
        import os

        templates = tuple(templates)
        validate_templates(claim, templates)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        rows = []
        for template in templates:
            row = asdict(template)
            row["embedding"]["values"] = template.embedding.normalized().tolist()
            rows.append(row)
        fd = os.open(self._path(claim), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump(
                {"schema_version": "enrollment-v1", "templates": rows}, f, indent=2
            )

    def get_claim(self, claimed_subject_id):
        import json

        path = self._path(claimed_subject_id)
        if not path.exists():
            return ()
        with open(path) as f:
            data = json.load(f)
        if data["schema_version"] != "enrollment-v1":
            raise ValueError("Unsupported enrollment format")
        rows = []
        for row in data["templates"]:
            emb = row["embedding"]
            emb["values"] = np.asarray(emb["values"], dtype=np.float32)
            rows.append(
                EnrollmentTemplate(
                    row["claimed_subject_id"],
                    GaitEmbedding(**emb),
                    Recording(**row["recording"]),
                    row["quality_weight"],
                )
            )
        validate_templates(claimed_subject_id, rows)
        return tuple(rows)
