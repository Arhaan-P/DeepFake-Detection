"""Capture registry: originals and REGISTERED derivatives, resolved by actual SHA-256.

The registry is the authoritative lineage source for managed-study workflows.

* Operator-SUPPLIED metadata (subject, session, time, view, consent reference, camera
  notes) is kept apart from program-VERIFIED facts (file bytes, ffprobe timing).
* A consent reference points at a record kept outside the repository; it is not proof
  that consent was obtained.
* A file whose bytes are not registered is UNRESOLVED. It is never treated as a new
  original, and filenames are never used to infer a take, subject or lineage.
* Limitation: SHA-256 identifies exact bytes only. A re-encoded or edited copy that was
  never registered cannot be linked to its source by this registry; managed workflows
  therefore refuse it instead of guessing. This is not a perceptual-copy detector.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Optional
from .contracts import Recording, valid_claim
from .media import probe_video, sha256_file

SCHEMA = "capture-registry-v1"
ROLES = ("development", "calibration", "locked_final")
_ID = re.compile(r"[A-Za-z0-9_.-]{1,128}")
_SHA = re.compile(r"[a-f0-9]{64}")
_PLACEHOLDER = {"unknown", "historical", "none", "null", "na", "default", "tbd"}
_PLACEHOLDER |= {"placeholder", "unset", "missing", "todo", "tmp", "dummy"}
# Registry-only keys; a raw row is otherwise a capture-manifest-v1 row.
_REGISTRY_KEYS = {"asset_id", "variant_id", "parent_asset_id", "split_role"}
_REGISTRY_KEYS |= {"transformation_manifest_ref", "transformation_manifest_sha256"}
_DERIVED_REQUIRED = {"asset_id", "variant_id", "parent_asset_id", "original_take_id"}
_DERIVED_REQUIRED |= {"subject_id", "sha256", "source_file_uri"}
_DERIVED_REQUIRED |= {"transformation_manifest_ref"}
# Optional on a derivative; when present they must equal the original's values.
_INHERITED = {"session_id": "session_id", "date": "recorded_at", "view_label": "view"}
_DERIVED_KEYS = _DERIVED_REQUIRED | set(_INHERITED)
_DERIVED_KEYS |= {"split_role", "transformation_manifest_sha256", "recording_notes"}
OPERATOR_SUPPLIED = (
    "subject_id",
    "session_id",
    "recorded_at",
    "view",
    "consent_ref",
    "capture",
)


class ProvenanceError(ValueError):
    """Registry content that cannot establish trustworthy lineage. Never repaired."""


def validate_capture_rows(rows, schema):
    """capture-manifest-v1 originals; unknown acquisition settings stay null."""
    seen = set()
    sources = set()
    for row in rows:
        missing = set(schema["required_fields"]) - row.keys()
        if missing:
            raise ValueError(f"Missing fields: {sorted(missing)}")
        for name in (
            "original_take_id",
            "subject_id",
            "session_id",
            "date",
            "sha256",
            "consent_ref",
            "source_file_uri",
            "actual_frame_timestamps_ref",
        ):
            if not row[name]:
                raise ValueError(f"Missing required provenance: {name}")
        datetime.fromisoformat(row["date"])
        if not re.fullmatch(r"[a-f0-9]{64}", row["sha256"]):
            raise ValueError("Invalid SHA-256")
        if row["original_take_id"] in seen or row["sha256"] in sources:
            raise ValueError("Duplicate original take/content")
        seen.add(row["original_take_id"])
        sources.add(row["sha256"])
        if row["view_label"] not in schema["planned_views"]:
            raise ValueError("Unknown capture view")
        if row["duration_sec"] is not None and row["duration_sec"] <= 0:
            raise ValueError("Invalid duration")
    return len(rows)


def load_capture_schema():
    from .registry import load_json

    return load_json("configs/gait_verification/capture_schema.json")


def canonical_sha256(payload):
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()


def write_new_json(path, payload):
    """Atomic create of a private file; an existing file is never overwritten."""
    path = Path(path)
    data = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.link(tmp, path)  # FileExistsError instead of replacing
    finally:
        tmp.unlink(missing_ok=True)
    return path


def _placeholder(value):
    return any(t.lower() in _PLACEHOLDER for t in re.split(r"[-_.]", value))


def _timestamp(value, now):
    try:
        moment = datetime.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ProvenanceError("Invalid recording date/time") from exc
    if moment.tzinfo is None:
        raise ProvenanceError("Recording time needs an explicit UTC offset")
    if moment > now:
        raise ProvenanceError("Recording time is in the future")
    return moment


@dataclass(frozen=True)
class Asset:
    asset_id: str
    variant_id: str
    parent_asset_id: Optional[str]
    original_take_id: str
    subject_id: str  # operator-supplied; offline labelling only, never a verifier input
    session_id: str
    recorded_at: str
    view: str
    source_uri: str
    sha256: str
    declared_split_role: Optional[str]
    consent_ref: str  # a reference, not proof of consent
    transformation_manifest_ref: Optional[str] = None
    transformation_manifest_sha256: Optional[str] = None
    capture: dict = field(default_factory=dict)

    @property
    def is_original(self):
        return self.variant_id == "raw"


class CaptureRegistry:
    """Validated, immutable view of one registry document."""

    def __init__(self, document, schema=None, now=None):
        if not isinstance(document, dict) or document.get("schema_version") != SCHEMA:
            raise ProvenanceError("Unsupported capture registry schema")
        rows = document.get("assets")
        if not isinstance(rows, list) or not rows:
            raise ProvenanceError("Registry needs a non-empty asset list")
        self.document = {"schema_version": SCHEMA, "assets": rows}
        self.fingerprint = canonical_sha256(
            {"schema_version": SCHEMA, "assets": sorted(rows, key=_asset_key)}
        )
        self._assets = _validate(
            rows, schema or load_capture_schema(), now or datetime.now(timezone.utc)
        )
        self._by_sha = {a.sha256: a for a in self._assets.values()}

    def asset(self, asset_id):
        try:
            return self._assets[asset_id]
        except KeyError:
            raise ProvenanceError("Unknown asset ID") from None

    def assets(self):
        return tuple(self._assets[k] for k in sorted(self._assets))

    def ancestry(self, asset_id):
        """Asset first, original last. Validated acyclic at construction."""
        chain = [self.asset(asset_id)]
        while chain[-1].parent_asset_id is not None:
            chain.append(self._assets[chain[-1].parent_asset_id])
        return tuple(chain)

    def original(self, asset_id):
        return self.ancestry(asset_id)[-1]

    def originals(self, subject_id=None):
        return tuple(
            a
            for a in self.assets()
            if a.is_original and subject_id in (None, a.subject_id)
        )

    def subjects(self):
        return tuple(sorted({a.subject_id for a in self._assets.values()}))

    def variant_of(self, original_take_id, variant_id):
        """The one registered asset of this variant descending from the original."""
        found = [
            a
            for a in self.assets()
            if a.variant_id == variant_id
            and self.original(a.asset_id).original_take_id == original_take_id
        ]
        if len(found) > 1:
            raise ProvenanceError("Ambiguous variant: several registered assets")
        return found[0] if found else None

    def resolve_sha(self, sha256):
        return self._by_sha.get(sha256)

    def resolve_file(self, path):
        """Resolve a file by its ACTUAL bytes; None when unregistered."""
        return self.resolve_sha(sha256_file(path))

    def lineage_ids(self, asset_id):
        """Original take plus every ancestor asset ID and checksum."""
        chain = self.ancestry(asset_id)
        ids = {chain[-1].original_take_id}
        for a in chain:
            ids |= {a.asset_id, a.sha256}
        return frozenset(ids)

    def path_of(self, asset, root):
        return _contained(root, asset.source_uri)

    def recording(self, asset_id, split_role):
        a = self.asset(asset_id)
        if a.declared_split_role not in (None, split_role):
            raise ProvenanceError("Registry split role disagrees with split manifest")
        return Recording(
            a.asset_id,
            a.original_take_id,
            a.subject_id,
            a.session_id,
            a.view,
            a.recorded_at,
            a.sha256,
            split_role,
            a.variant_id,
            a.parent_asset_id,
        )

    def summary(self):
        assets = self.assets()
        variants = {}
        for a in assets:
            variants[a.variant_id] = variants.get(a.variant_id, 0) + 1
        sessions = {(a.subject_id, a.session_id) for a in assets if a.is_original}
        return {
            "schema_version": SCHEMA,
            "registry_fingerprint": self.fingerprint,
            "subjects": len(self.subjects()),
            "originals": len(self.originals()),
            "assets_by_variant": dict(sorted(variants.items())),
            "subject_sessions": len(sessions),
            "views": sorted({a.view for a in assets}),
            "consent_refs_present": sum(bool(a.consent_ref) for a in assets),
            "consent_verified_by_program": False,
            "operator_supplied_fields": list(OPERATOR_SUPPLIED),
        }

    def verify_files(self, root, measure=False):
        """Program-verified facts: bytes, optional ffprobe timing, transform manifests.

        Returns a report; `ok` is False when any check fails. Nothing is modified.
        """
        errors, files = [], {}
        outputs = {}
        for a in self.assets():
            entry = files[a.asset_id] = {"exists": False, "sha256_verified": False}
            try:
                path = self.path_of(a, root)
            except ProvenanceError as exc:
                errors.append(f"{a.asset_id}: {exc}")
                continue
            if not path.is_file():
                errors.append(f"{a.asset_id}: source file missing")
                continue
            entry["exists"] = True
            entry["sha256_verified"] = sha256_file(path) == a.sha256
            if not entry["sha256_verified"]:
                errors.append(f"{a.asset_id}: checksum mismatch")
            if measure and entry["sha256_verified"]:
                try:
                    m = probe_video(path)
                    entry["measured"] = dict(
                        width=m.width,
                        height=m.height,
                        rotation=m.rotation,
                        codec=m.codec,
                        nominal_fps=m.nominal_fps,
                        average_fps=m.average_fps,
                        time_base=m.time_base,
                        frame_count=len(m.timestamps),
                        pts_span_sec=m.timestamps[-1] - m.timestamps[0],
                        duration_sec=m.duration_sec,
                    )
                except ValueError as exc:
                    errors.append(f"{a.asset_id}: unreadable media ({exc})")
            if a.is_original:
                continue
            try:
                manifest = self._transformation(a, root)
                source, output = (
                    manifest["input"]["sha256"],
                    manifest["output"]["sha256"],
                )
            except (OSError, ValueError, KeyError, TypeError) as exc:
                errors.append(f"{a.asset_id}: transformation manifest ({exc})")
                continue
            if source != self._assets[a.parent_asset_id].sha256:
                errors.append(f"{a.asset_id}: transformation input is not the parent")
            if output != a.sha256:
                errors.append(f"{a.asset_id}: transformation output is not this file")
            outputs[output] = a.asset_id
        for a in self.originals():
            if a.sha256 in outputs:
                errors.append(
                    f"{a.asset_id}: registered as an original but is a transformation output"
                )
        return {
            "registry_fingerprint": self.fingerprint,
            "ok": not errors,
            "errors": errors,
            "files": files,
            "verified_by_program": ["sha256", "measured", "transformation hashes"],
            "operator_supplied_not_verified": list(OPERATOR_SUPPLIED),
        }

    def _transformation(self, asset, root):
        path = _contained(root, asset.transformation_manifest_ref)
        if (
            asset.transformation_manifest_sha256
            and sha256_file(path) != asset.transformation_manifest_sha256
        ):
            raise ValueError("manifest checksum mismatch")
        manifest = json.loads(path.read_text())
        if manifest.get("status") != "OK":
            raise ValueError("transformation did not complete")
        return manifest


def load_registry(path, schema=None, now=None):
    try:
        document = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ProvenanceError("Unreadable capture registry") from exc
    return CaptureRegistry(document, schema, now)


def _asset_key(row):
    return str(row.get("asset_id")) if isinstance(row, dict) else ""


def _contained(root, uri):
    if not isinstance(uri, str) or not uri or Path(uri).is_absolute():
        raise ProvenanceError("Registry paths must be relative to the data root")
    base = Path(root).resolve()
    path = (base / uri).resolve()
    if not path.is_relative_to(base):
        raise ProvenanceError("Registry path escapes the data root")
    return path


def _check_id(value, what):
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ProvenanceError(f"Invalid {what}")


def _role(value):
    if value is not None and value not in ROLES:
        raise ProvenanceError("Invalid split role")
    return value


def _original(row, schema, now):
    if row.get("variant_id") != "raw":
        raise ProvenanceError("Originals must have variant_id 'raw'")
    if row.get("parent_asset_id") is not None or row.get("transformation_manifest_ref"):
        raise ProvenanceError("A derivative cannot be registered as an original")
    unknown = row.keys() - set(schema["required_fields"]) - _REGISTRY_KEYS
    if unknown:
        raise ProvenanceError(f"Ambiguous unknown fields: {sorted(unknown)}")
    try:
        validate_capture_rows([row], schema)
    except (KeyError, TypeError, ValueError) as exc:
        raise ProvenanceError(f"Invalid capture row: {exc}") from exc
    _check_id(row["original_take_id"], "original take ID")
    if not valid_claim(row["subject_id"]):
        raise ProvenanceError("Invalid pseudonymous subject ID")
    _check_id(row["session_id"], "session ID")
    if _placeholder(row["session_id"]):
        raise ProvenanceError("Placeholder session IDs are not real sessions")
    if not isinstance(row["consent_ref"], str) or _placeholder(row["consent_ref"]):
        raise ProvenanceError("Missing consent reference")
    _timestamp(row["date"], now)
    capture = {
        k: v
        for k, v in row.items()
        if k not in _REGISTRY_KEYS and k not in ("sha256", "source_file_uri")
    }
    return Asset(
        row["asset_id"],
        "raw",
        None,
        row["original_take_id"],
        row["subject_id"],
        row["session_id"],
        row["date"],
        row["view_label"],
        row["source_file_uri"],
        row["sha256"],
        _role(row.get("split_role")),
        row["consent_ref"],
        capture=capture,
    )


def _validate(rows, schema, now):
    raw, derived, shas = {}, {}, set()
    for row in rows:
        if not isinstance(row, dict):
            raise ProvenanceError("Registry rows must be objects")
        _check_id(row.get("asset_id"), "asset ID")
        if row["asset_id"] in raw or row["asset_id"] in derived:
            raise ProvenanceError("Duplicate asset ID")
        sha = row.get("sha256")
        if not isinstance(sha, str) or not _SHA.fullmatch(sha):
            raise ProvenanceError("Missing or invalid SHA-256")
        if sha in shas:
            raise ProvenanceError("Same bytes registered under two assets")
        shas.add(sha)
        if row.get("variant_id") == "raw" and row.get("parent_asset_id") is None:
            raw[row["asset_id"]] = _original(row, schema, now)
        else:
            missing = _DERIVED_REQUIRED - {k for k, v in row.items() if v}
            if missing:
                raise ProvenanceError(
                    f"Missing derivative provenance: {sorted(missing)}"
                )
            if (
                row["variant_id"] == "raw"
                or row.get("transformation_manifest_ref") is None
            ):
                raise ProvenanceError(
                    "A derivative cannot be registered as an original"
                )
            unknown = row.keys() - _DERIVED_KEYS
            if unknown:
                raise ProvenanceError(f"Ambiguous unknown fields: {sorted(unknown)}")
            for name in ("variant_id", "parent_asset_id", "original_take_id"):
                _check_id(row[name], name)
            derived[row["asset_id"]] = row
    takes, sessions = {}, {}
    for a in raw.values():
        if takes.setdefault(a.original_take_id, a.asset_id) != a.asset_id:
            raise ProvenanceError("Conflicting original take IDs")
        day = datetime.fromisoformat(a.recorded_at).date()
        if sessions.setdefault((a.subject_id, a.session_id), day) != day:
            raise ProvenanceError("One session spans several recording days")
    assets = dict(raw)
    for asset_id, row in derived.items():
        seen, cursor = {asset_id}, row["parent_asset_id"]
        while cursor in derived:
            if cursor in seen:
                raise ProvenanceError("Cyclic derivative ancestry")
            seen.add(cursor)
            cursor = derived[cursor]["parent_asset_id"]
        if cursor not in raw:
            raise ProvenanceError("Parent reference cannot be resolved to an original")
        root = raw[cursor]
        if row["original_take_id"] != root.original_take_id:
            raise ProvenanceError("Derivative names a different original take")
        if row["subject_id"] != root.subject_id:
            raise ProvenanceError("Derivative subject differs from its original")
        if row.get("split_role") not in (None, root.declared_split_role):
            raise ProvenanceError("Derivative split role differs from its original")
        for key, attr in _INHERITED.items():
            if row.get(key) is not None and row[key] != getattr(root, attr):
                raise ProvenanceError(f"Derivative {key} differs from its original")
        assets[asset_id] = Asset(
            asset_id,
            row["variant_id"],
            row["parent_asset_id"],
            root.original_take_id,
            root.subject_id,
            root.session_id,
            root.recorded_at,
            root.view,
            row["source_file_uri"],
            row["sha256"],
            root.declared_split_role,
            root.consent_ref,
            row["transformation_manifest_ref"],
            row.get("transformation_manifest_sha256"),
            {"recording_notes": row.get("recording_notes")},
        )
    return assets
