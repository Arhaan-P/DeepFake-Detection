"""Per-original reference embeddings. The store API deliberately exposes only direct claim lookup.

Every reference is one independent ORIGINAL recording embedded with one frozen recipe.
A prototype is derived on demand (`scoring.prototype`) and never replaces the individual
vectors. Windows and RAW/S10/S15 derivatives of one take are never separate references.
Files are private-by-permission only (0600/0700); they are NOT encrypted.
"""

from dataclasses import dataclass, field, asdict
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Protocol
import numpy as np
from .contracts import GaitEmbedding, Recording, valid_claim
from .scoring import prototype  # noqa: F401  (historical import location)

SCHEMA = "enrollment-v2"
READABLE_SCHEMAS = ("enrollment-v1", SCHEMA)
_ID = re.compile(r"[A-Za-z0-9_.:-]{1,128}")


class CorruptEnrollment(ValueError):
    """Unreadable or inconsistent store content; never silently repaired or overwritten."""


@dataclass(frozen=True)
class EnrollmentTemplate:
    claimed_subject_id: str
    embedding: GaitEmbedding
    recording: Recording
    quality_weight: float = 1.0
    enrollment_id: str = ""
    # Pose/gait fingerprints, window recipe, quality summary, consent reference.
    provenance: dict = field(default_factory=dict)


def validate_templates(claim, templates):
    """Validate ONE recipe's references for ONE claim."""
    seen_takes, seen_hashes, seen_ids = set(), set(), set()
    for t in templates:
        r = t.recording
        if t.claimed_subject_id != claim or r.subject_id != claim:
            raise ValueError("Enrollment belongs to a different claim")
        if t.embedding.original_take_id != r.original_take_id:
            raise ValueError("Enrollment lineage mismatch")
        if not t.embedding.source_sha256 or not r.session_id:
            raise ValueError("Missing enrollment provenance")
        if t.embedding.source_sha256 != r.source_sha256:
            raise ValueError("Conflicting source checksums")
        if r.variant_id != "raw" or r.parent_asset_id is not None:
            raise ValueError("Enrollment must be an original, not a derivative")
        if r.original_take_id in seen_takes:
            raise ValueError(
                "Repeated windows/derivatives are not independent enrollments"
            )
        if r.source_sha256 in seen_hashes:
            raise ValueError("Same source bytes enrolled under two takes")
        seen_takes.add(r.original_take_id)
        seen_hashes.add(r.source_sha256)
        if t.enrollment_id:
            if t.enrollment_id in seen_ids:
                raise ValueError("Duplicate enrollment ID")
            seen_ids.add(t.enrollment_id)
        if not np.isfinite(t.quality_weight) or t.quality_weight <= 0:
            raise ValueError("Invalid quality weight")
        t.embedding.normalized()
    if len({t.embedding.model_id for t in templates}) > 1:
        raise ValueError("Mixed enrollment recipes")


def validate_claim_references(claim, templates):
    """One claim may hold several independent recipes; each is validated alone."""
    groups = {}
    for t in templates:
        groups.setdefault(t.embedding.model_id, []).append(t)
    for group in groups.values():
        validate_templates(claim, group)
    hashes = {}
    for t in templates:
        r = t.recording
        if hashes.setdefault(r.original_take_id, r.source_sha256) != r.source_sha256:
            raise ValueError("Conflicting source checksums")
        if (
            t.enrollment_id
            and sum(u.enrollment_id == t.enrollment_id for u in templates) > 1
        ):
            raise ValueError("Duplicate enrollment ID")


def _check_sources(index, subject_key, templates):
    """Reject source bytes or take IDs owned by another subject. Updates `index`."""
    for t in templates:
        sha, take = t.recording.source_sha256, t.recording.original_take_id
        owner = index["sha"].get(sha)
        if owner is not None and owner != [subject_key, take]:
            raise ValueError(
                "Source recording already enrolled for another subject/take"
            )
        if index["take"].get(take, sha) != sha:
            raise ValueError("Conflicting source checksums")
        index["sha"][sha] = [subject_key, take]
        index["take"][take] = sha


class EnrollmentStore(Protocol):
    def get_claim(self, claimed_subject_id: str) -> tuple[EnrollmentTemplate, ...]: ...


class MemoryEnrollmentStore:
    def __init__(self):
        self._records = {}
        self._index = {"sha": {}, "take": {}}

    def enroll(self, claim, templates):
        if not valid_claim(claim):
            raise ValueError("Invalid claim syntax")
        templates = tuple(templates)
        validate_claim_references(claim, templates)
        index = json.loads(json.dumps(self._index))
        _check_sources(index, claim, templates)
        self._index, self._records[claim] = index, templates

    def add_reference(self, claim, template):
        self.add_references(claim, (template,))

    def add_references(self, claim, templates):
        old = self._records.get(claim, ())
        self.enroll(claim, old + tuple(templates))

    def get_claim(self, claimed_subject_id):
        return self._records.get(claimed_subject_id, ())


def _row(template):
    row = asdict(template)
    row["embedding"]["values"] = template.embedding.normalized().tolist()
    return row


def _template(row):
    emb = dict(row["embedding"])
    emb["values"] = np.asarray(emb["values"], dtype=np.float32)
    return EnrollmentTemplate(
        row["claimed_subject_id"],
        GaitEmbedding(**emb),
        Recording(**row["recording"]),
        row["quality_weight"],
        row.get("enrollment_id", ""),
        row.get("provenance", {}),
    )


class DirectoryEnrollmentStore:
    """Private JSON artifact per hashed claim; never enumerate/load other claims.

    `source-index.json` holds only SHA-256 -> (hashed subject, take) so cross-subject
    reuse of one recording can be refused at ENROLLMENT time. It holds no embeddings and
    is never read during verification.
    """

    def __init__(self, root):
        self.root = Path(root)

    def _path(self, claim):
        if not valid_claim(claim):
            raise ValueError("Invalid claim syntax")
        return self.root / (hashlib.sha256(claim.encode()).hexdigest() + ".json")

    @staticmethod
    def _read_json(path):
        try:
            with open(path) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise CorruptEnrollment("Unreadable enrollment store file") from exc

    def _load(self, claim):
        path = self._path(claim)
        if not path.exists():
            return [], {}
        data = self._read_json(path)
        try:
            if data["schema_version"] not in READABLE_SCHEMAS:
                raise CorruptEnrollment("Unsupported enrollment format")
            if data.get("claimed_subject_id", claim) != claim:
                raise CorruptEnrollment("Enrollment file belongs to another claim")
            rows = list(data["templates" if "templates" in data else "references"])
            [_template(r) for r in rows]
            return rows, dict(data.get("revocations", {}))
        except CorruptEnrollment:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise CorruptEnrollment("Malformed enrollment record") from exc

    def _atomic_write(self, path, payload):
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        data = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, "w") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

    def _locked(self):
        import fcntl
        from contextlib import contextmanager

        @contextmanager
        def lock():
            self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
            fd = os.open(self.root / ".lock", os.O_WRONLY | os.O_CREAT, 0o600)
            with os.fdopen(fd, "w") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                yield

        return lock()

    def _add(self, claim, new, create_only):
        path = self._path(claim)
        for t in new:
            if not _ID.fullmatch(t.enrollment_id or ""):
                raise ValueError(
                    "A persistent enrollment requires a valid enrollment_id"
                )
        with self._locked():
            if create_only and path.exists():
                raise FileExistsError("Claim already enrolled; use add_reference")
            rows, revoked = self._load(claim)
            active = [
                _template(r) for r in rows if r.get("enrollment_id", "") not in revoked
            ]
            keys = {
                (t.recording.original_take_id, t.embedding.model_id) for t in active
            }
            for t in new:
                if (t.recording.original_take_id, t.embedding.model_id) in keys:
                    raise ValueError(
                        "Take already enrolled for this recipe; revoke first"
                    )
            if {t.enrollment_id for t in new} & {
                r.get("enrollment_id", "") for r in rows
            }:
                raise ValueError("Duplicate enrollment ID")
            validate_claim_references(claim, active + list(new))
            index_path = self.root / "source-index.json"
            index = (
                self._read_json(index_path)
                if index_path.exists()
                else {"sha": {}, "take": {}}
            )
            _check_sources(index, hashlib.sha256(claim.encode()).hexdigest(), new)
            new_rows = [_row(t) for t in new]
            payload = {
                "schema_version": SCHEMA,
                "claimed_subject_id": claim,
                "references": rows + new_rows,
                "revocations": revoked,
            }
            json.dumps(payload, allow_nan=False)  # fail before touching any file
            self._atomic_write(index_path, index)
            self._atomic_write(path, payload)

    def enroll(self, claim, templates):
        """Create a claim's first enrollment. Never overwrites an existing claim."""
        self._add(claim, tuple(templates), create_only=True)

    def add_reference(self, claim, template):
        self._add(claim, (template,), create_only=False)

    def add_references(self, claim, templates):
        """All recipes of one original in ONE atomic write: all stored or none."""
        self._add(claim, tuple(templates), create_only=False)

    def revoke_reference(self, claim, enrollment_id, reason):
        """Exclude one reference from lookups; its vector and record stay on disk."""
        if not reason:
            raise ValueError("A revocation reason is required")
        with self._locked():
            rows, revoked = self._load(claim)
            if enrollment_id not in {r.get("enrollment_id") for r in rows}:
                raise ValueError("Unknown enrollment ID for this claim")
            if enrollment_id in revoked:
                raise ValueError("Reference already revoked")
            revoked[enrollment_id] = {"reason": reason}
            self._atomic_write(
                self._path(claim),
                {
                    "schema_version": SCHEMA,
                    "claimed_subject_id": claim,
                    "references": rows,
                    "revocations": revoked,
                },
            )

    def get_claim(self, claimed_subject_id):
        rows, revoked = self._load(claimed_subject_id)
        templates = tuple(
            _template(r) for r in rows if r.get("enrollment_id", "") not in revoked
        )
        try:
            validate_claim_references(claimed_subject_id, templates)
        except ValueError as exc:
            raise CorruptEnrollment("Stored references fail validation") from exc
        return templates
