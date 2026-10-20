"""Pilot adapter must not weaken scientific v3 or locked-final contracts."""

from copy import deepcopy
from dataclasses import asdict

import pytest

from gait_verification.pilot_protocol import (
    pilot_split,
    pilot_trials,
    validate_pilot_trials,
    claim_request,
)
from gait_verification.pilot_intake import validate_intake
import test_pilot_intake as intake_fixtures

sheet = intake_fixtures.sheet


def test_two_set_claim_only_cross_view_protocol(sheet):
    root, doc = sheet
    _, registry, measured = validate_intake(doc, root)
    split = pilot_split(registry)
    manifest = pilot_trials(registry, split, measured)
    assert split["status"] == "DRAFT"  # no hidden lock, no scientific allocation
    assert len(manifest["trials"]) == 16
    assert sum(t["label_same_identity"] for t in manifest["trials"]) == 8
    assert manifest == pilot_trials(registry, split, measured)
    validate_pilot_trials(manifest, registry, split, measured)
    for t in manifest["trials"]:
        assert "set2" in t["query_original_take_id"]
        refs = [
            a
            for a in registry.originals(t["claimed_subject_id"])
            if a.original_take_id in t["reference_original_take_ids"]
        ]
        assert len(refs) == 3
        assert all(
            "set1" in a.original_take_id and a.view != t["probe_view"] for a in refs
        )
        request = asdict(claim_request(t))
        assert (
            "query_subject_id" not in request and "label_same_identity" not in request
        )
    swap = pilot_trials(registry, split, measured, direction="set2-to-set1")
    assert {t["trial_id"] for t in manifest["trials"]}.isdisjoint(
        t["trial_id"] for t in swap["trials"]
    )
    assert swap["direction"] != manifest["direction"]


@pytest.mark.parametrize("mutation", ["reference", "policy", "label", "count"])
def test_pilot_trials_detect_contract_tampering(sheet, mutation):
    root, doc = sheet
    _, registry, measured = validate_intake(doc, root)
    split = pilot_split(registry)
    manifest = deepcopy(pilot_trials(registry, split, measured))
    if mutation == "reference":
        manifest["trials"][0]["reference_original_take_ids"] = [
            manifest["trials"][0]["query_original_take_id"]
        ]
    elif mutation == "policy":
        manifest["reference_rule"] = "same_view"
    elif mutation == "label":
        manifest["trials"][0]["label_same_identity"] = False
    else:
        manifest["trials"].pop()
    with pytest.raises(ValueError):
        validate_pilot_trials(manifest, registry, split, measured)


def test_pilot_cannot_use_scientific_or_final_split(sheet):
    from gait_verification.splits import SplitPolicy, assign_split

    root, doc = sheet
    _, registry, measured = validate_intake(doc, root)
    final = assign_split(
        registry.subjects(),
        SplitPolicy(
            "scientific", 1, {"development": 1, "calibration": 0, "locked_final": 1}
        ),
    )
    with pytest.raises(ValueError):
        pilot_trials(registry, final, measured)


def test_set_labels_use_explicit_sheet_not_filename(sheet):
    root, doc = sheet
    _, registry, measured = validate_intake(doc, root)
    for r in measured["rows"]:
        r["set_number"] = 3 - r["set_number"]
    trials = pilot_trials(registry, pilot_split(registry), measured)
    assert all("set1" in t["query_original_take_id"] for t in trials["trials"])
