"""Narrow two-set adapter. Scientific verification-trials-v3 stays unchanged.

Uses the existing split validator and unchanged claim request/result contract.
Pilot access is development-only, DRAFT, never a scientific partition or lock.
"""

from dataclasses import asdict

from .capture_registry import canonical_sha256
from .evaluation import VerificationTrial
from .offline_eval import ClaimRequest
from .pilot_intake import EVIDENCE, NOTICE, VIEWS
from .splits import SplitPolicy, assign_split, validate_split


def pilot_split(registry):
    if len(registry.subjects()) != 2 or any(
        a.declared_split_role != "development" for a in registry.assets()
    ):
        raise ValueError(
            "Pilot requires exactly two explicitly development-only subjects"
        )
    return assign_split(
        registry.subjects(),
        SplitPolicy(
            EVIDENCE,
            0,
            {"development": 2, "calibration": 0, "locked_final": 0},
            registry.subjects(),
        ),
    )


def pilot_trials(registry, split, sheet, direction="set1-to-set2"):
    validate_split(split, registry.subjects())
    if split != pilot_split(registry):
        raise ValueError("Pilot cannot use a scientific or locked split")
    if direction not in ("set1-to-set2", "set2-to-set1"):
        raise ValueError("Unknown pilot direction")
    reference_set, query_set = (1, 2) if direction == "set1-to-set2" else (2, 1)
    matrix = {}
    for row in sheet["rows"]:
        c = row["capture"]
        asset = registry.asset(c["asset_id"])
        if not asset.is_original or any(
            c.get(k) != getattr(asset, attr)
            for k, attr in (
                ("sha256", "sha256"),
                ("original_take_id", "original_take_id"),
                ("subject_id", "subject_id"),
                ("view_label", "view"),
                ("session_id", "session_id"),
                ("date", "recorded_at"),
                ("consent_ref", "consent_ref"),
                ("source_file_uri", "source_uri"),
            )
        ):
            raise ValueError("Intake/registry ownership or provenance mismatch")
        if any(
            row.get(k) is not True
            for k in (
                "operator_confirmed",
                "consent_confirmed",
                "independent_original_confirmed",
            )
        ):
            raise ValueError("Pilot intake assertions unconfirmed")
        key = (asset.subject_id, asset.view, row["set_number"])
        if key in matrix:
            raise ValueError("Ambiguous pilot take matrix")
        matrix[key] = asset
    if set(matrix) != {
        (s, v, n) for s in registry.subjects() for v in VIEWS for n in (1, 2)
    }:
        raise ValueError("Incomplete pilot matrix")
    trials = []
    for (subject, view, number), query in sorted(matrix.items()):
        if number != query_set:
            continue
        for claim in registry.subjects():
            refs = sorted(
                (matrix[(claim, v, reference_set)] for v in VIEWS if v != view),
                key=lambda a: a.original_take_id,
            )
            if any(
                registry.lineage_ids(query.asset_id) & registry.lineage_ids(r.asset_id)
                for r in refs
            ):
                raise ValueError("SHARED_ORIGINAL")
            takes = tuple(r.original_take_id for r in refs)
            trial_id = canonical_sha256(
                [EVIDENCE, direction, query.asset_id, claim, takes]
            )[:24]
            trial = VerificationTrial(
                trial_id,
                query.original_take_id,
                "raw",
                subject,
                claim,
                takes,
                claim == subject,
                view,
                tuple(r.view for r in refs),
                query.session_id,
                direction,
                "development",
                EVIDENCE,
                query.asset_id,
                registry.fingerprint,
                split["fingerprint"],
            )
            trials.append(asdict(trial))
    # Canonical JSON conversion keeps tuples/lists identical before/after persistence.
    import json

    body = json.loads(
        json.dumps(
            dict(
                schema_version="two-subject-pilot-trials-v1",
                evidence_class=EVIDENCE,
                notice=NOTICE,
                direction=direction,
                reference_rule="exclude_probe_view",
                registry_fingerprint=registry.fingerprint,
                split_fingerprint=split["fingerprint"],
                intake_sha256=canonical_sha256(sheet),
                independent_originals=16,
                participants=2,
                trials=trials,
            )
        )
    )
    return body | {"contract_sha256": canonical_sha256(body)}


def validate_pilot_trials(manifest, registry, split, sheet):
    expected = pilot_trials(registry, split, sheet, manifest["direction"])
    if manifest != expected:
        raise ValueError(
            "Pilot contract/policy/rows differ from deterministic regeneration"
        )
    return manifest


def claim_request(trial):
    return ClaimRequest(
        trial["trial_id"],
        trial["query_asset_id"],
        trial["query_original_take_id"],
        trial["query_variant_id"],
        trial["claimed_subject_id"],
        tuple(trial["reference_original_take_ids"]),
    )
