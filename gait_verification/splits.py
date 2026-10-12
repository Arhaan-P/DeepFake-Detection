"""Subject-level split manifests and the locked-final-test access guard.

The participant is the independent unit: original takes, derivatives, windows,
sessions and views inherit their subject's role. Role counts and the seed are
explicit owner inputs; nothing here chooses an allocation (24/8/8 in the PRD is an
illustration, not an approved split). A DRAFT may be regenerated; a LOCKED split can
only change through an explicitly recorded protocol revision.
"""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import re
from .capture_registry import ROLES, canonical_sha256
from .contracts import valid_claim

SCHEMA = "subject-split-v1"
_SHA = re.compile(r"[a-f0-9]{64}")


class LockedTestAccessDenied(PermissionError):
    """Locked-final subjects requested without a valid, explicit authorization."""


@dataclass(frozen=True)
class SplitPolicy:
    version: str
    seed: int
    counts: dict  # role -> number of subjects; must cover every subject exactly
    # Pilot participants whose recordings influenced settings stay in development.
    development_pinned: tuple = ()

    def validate(self, subjects):
        if not self.version or type(self.seed) is not int:
            raise ValueError("Split policy needs a version and an integer seed")
        if set(self.counts) != set(ROLES) or any(
            type(n) is not int or n < 0 for n in self.counts.values()
        ):
            raise ValueError("Counts must give a nonnegative integer for every role")
        if sum(self.counts.values()) != len(subjects):
            raise ValueError("Role counts must sum to the number of subjects")
        if not set(self.development_pinned) <= set(subjects):
            raise ValueError("Pinned subject is not in the registry")
        if len(self.development_pinned) > self.counts["development"]:
            raise ValueError("More pinned subjects than development places")


@dataclass(frozen=True)
class LockedTestAuthorization:
    """Explicit, recorded permission to open the locked final group ONCE.

    Not used during Phase 1D. `frozen_bundle_sha256` must identify the frozen
    model/preprocessing/threshold/trial-policy bundle chosen without locked data.
    """

    split_fingerprint: str
    frozen_bundle_sha256: str
    authorized_by: str
    reason: str


def subject_set_sha256(subjects):
    return hashlib.sha256("\n".join(sorted(subjects)).encode()).hexdigest()


def _content(manifest):
    keys = ("schema_version", "policy", "subject_set_sha256", "assignments")
    return {k: manifest[k] for k in keys} | {"revisions": manifest["revisions"]}


def assign_split(subjects, policy):
    """Deterministic DRAFT split: seeded hash order, independent of input order."""
    subjects = sorted(set(subjects))
    if not subjects or not all(valid_claim(s) for s in subjects):
        raise ValueError("Invalid subject IDs")
    policy.validate(subjects)
    order = sorted(
        (s for s in subjects if s not in policy.development_pinned),
        key=lambda s: hashlib.sha256(f"{policy.seed}:{s}".encode()).hexdigest(),
    )
    order = list(policy.development_pinned) + order
    assignments, start = {}, 0
    for role in ROLES:
        for s in order[start : start + policy.counts[role]]:
            assignments[s] = role
        start += policy.counts[role]
    manifest = {
        "schema_version": SCHEMA,
        "status": "DRAFT",
        "policy": asdict(policy)
        | {"development_pinned": list(policy.development_pinned)},
        "subject_set_sha256": subject_set_sha256(subjects),
        "assignments": dict(sorted(assignments.items())),
        "revisions": [],
    }
    manifest["fingerprint"] = canonical_sha256(_content(manifest))
    return manifest


def validate_split(manifest, subjects=None):
    """Each subject exactly one role; fingerprint matches; pinned never locked."""
    if manifest.get("schema_version") != SCHEMA:
        raise ValueError("Unsupported split schema")
    if manifest.get("status") not in ("DRAFT", "LOCKED"):
        raise ValueError("Unknown split status")
    assignments = manifest["assignments"]
    if any(role not in ROLES for role in assignments.values()):
        raise ValueError("Invalid split role")
    if manifest["fingerprint"] != canonical_sha256(_content(manifest)):
        raise ValueError("Split manifest was modified after fingerprinting")
    if subject_set_sha256(assignments) != manifest["subject_set_sha256"]:
        raise ValueError("Split subject set changed")
    if subjects is not None and set(subjects) != set(assignments):
        raise ValueError("Split does not cover exactly the registry subjects")
    for s in manifest["policy"]["development_pinned"]:
        if assignments.get(s) != "development":
            raise ValueError("Pinned development subject moved out of development")
    if manifest["status"] == "LOCKED" and not manifest.get("lock", {}).get("locked_by"):
        raise ValueError("Locked split lacks lock record")
    return manifest


def lock_split(manifest, locked_by, locked_at=None):
    validate_split(manifest)
    if manifest["status"] != "DRAFT":
        raise ValueError("Only a DRAFT split can be locked")
    if not locked_by:
        raise ValueError("Lock needs a responsible person")
    locked_at = locked_at or datetime.now(timezone.utc).isoformat()
    return manifest | {
        "status": "LOCKED",
        "lock": {"locked_by": locked_by, "locked_at": locked_at},
    }


def revise_split(locked, moves, revision_id, rationale, authorized_by):
    """The only way a LOCKED split changes: an explicit, recorded protocol revision."""
    validate_split(locked)
    if locked["status"] != "LOCKED":
        raise ValueError("Only a LOCKED split needs a protocol revision")
    if not (moves and revision_id and rationale and authorized_by):
        raise ValueError("Revision needs moves, ID, rationale and an authorizer")
    assignments = dict(locked["assignments"])
    for subject, role in sorted(moves.items()):
        if subject not in assignments or role not in ROLES:
            raise ValueError("Revision names an unknown subject or role")
        if assignments[subject] == "locked_final" or role == "locked_final":
            # Moving data into or out of the protected group would expose it.
            raise ValueError("Locked-final membership cannot be revised")
        assignments[subject] = role
    revision = {
        "revision_id": revision_id,
        "previous_fingerprint": locked["fingerprint"],
        "moves": dict(sorted(moves.items())),
        "rationale": rationale,
        "authorized_by": authorized_by,
    }
    revised = locked | {
        "assignments": assignments,
        "revisions": locked["revisions"] + [revision],
    }
    revised["fingerprint"] = canonical_sha256(_content(revised))
    return validate_split(revised)


def check_transition(old, new):
    """Reject any role change of a LOCKED split not explained by one new revision."""
    if old["status"] != "LOCKED" or _content(new) == _content(old):
        return
    validate_split(new)
    revisions = new["revisions"]
    if (
        new["status"] != "LOCKED"
        or revisions[: len(old["revisions"])] != old["revisions"]
        or len(revisions) != len(old["revisions"]) + 1
        or revisions[-1]["previous_fingerprint"] != old["fingerprint"]
    ):
        raise ValueError("Locked split changed without a recorded protocol revision")
    expected = old["assignments"] | revisions[-1]["moves"]
    if expected != new["assignments"]:
        raise ValueError("Split changes are not explained by the recorded revision")


def require_roles(split, roles, authorization=None):
    """Gate for every workflow that reads subjects of the given roles."""
    roles = tuple(roles)
    if not roles or any(r not in ROLES for r in roles):
        raise ValueError("Unknown split role")
    if "locked_final" not in roles:
        return roles
    a = authorization
    if (
        split.get("status") != "LOCKED"
        or not isinstance(a, LockedTestAuthorization)
        or a.split_fingerprint != split["fingerprint"]
        or not _SHA.fullmatch(a.frozen_bundle_sha256 or "")
        or not a.authorized_by
        or not a.reason
    ):
        raise LockedTestAccessDenied(
            "Locked final test requires explicit authorization"
        )
    return roles


def subjects_for(split, roles, authorization=None):
    roles = require_roles(split, roles, authorization)
    return tuple(s for s, r in split["assignments"].items() if r in roles)
