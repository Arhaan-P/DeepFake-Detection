"""Invented operator metadata and generated test media, never human gait evidence."""

from copy import deepcopy

import pytest

from gait_verification.pilot_intake import intake_template, validate_intake
from test_phase1c import make_video


@pytest.fixture(scope="module")
def sheet(tmp_path_factory):
    root = tmp_path_factory.mktemp("pilot-media")
    doc = intake_template(("SYN001", "SYN002"))
    for i, row in enumerate(doc["rows"]):
        path = root / row["capture"]["source_file_uri"]
        path.parent.mkdir(parents=True, exist_ok=True)
        make_video(path, 2.1 + i * 0.04)
        row.update(
            operator_confirmed=True,
            consent_confirmed=True,
            independent_original_confirmed=True,
        )
        row["capture"].update(
            session_id="SYNTHETIC-occasion1",
            date="2026-01-01T10:00:00+05:30",
            consent_ref="SYNTHETIC-consent-only",
            camera_motion_mode="stationary",
            viewing_angle_definition="SYNTHETIC-test-geometry",
            walking_direction="forward",
            quality_review_status="accepted",
        )
    return root, doc


def test_empty_matrix_does_not_invent_metadata(tmp_path):
    doc = intake_template(("S001", "S002"))
    assert len(doc["rows"]) == 16
    assert (
        len(
            {
                (
                    r["capture"]["subject_id"],
                    r["capture"]["view_label"],
                    r["set_number"],
                )
                for r in doc["rows"]
            }
        )
        == 16
    )
    assert all(r["capture"]["date"] is None for r in doc["rows"])
    result, registry, _ = validate_intake(doc, tmp_path)
    assert registry is None and len(result["missing_originals"]) == 16
    assert not result["ok"]


def test_valid_same_session_and_measured_facts(sheet):
    root, doc = sheet
    report, registry, measured = validate_intake(doc, root)
    assert report["ok"] and len(registry.originals()) == 16
    assert registry.summary()["subject_sessions"] == 2
    assert all(r["capture"]["sha256"] for r in measured["rows"])
    assert all(r["measured"]["timestamps"] for r in report["recordings"])
    assert all(r["decoder_health"] == "OK" for r in report["recordings"])


@pytest.mark.parametrize(
    "field,value",
    [
        ("consent_ref", None),
        ("session_id", "unknown"),
        ("date", "2026-01-01T10:00:00"),
        ("date", None),
        ("sha256", "f" * 64),
        ("quality_review_status", "pending"),
        ("camera_motion_mode", None),
        ("viewing_angle_definition", None),
    ],
)
def test_bad_metadata_refused(sheet, field, value):
    root, original = sheet
    doc = deepcopy(original)
    doc["rows"][0]["capture"][field] = value
    report, registry, _ = validate_intake(doc, root)
    assert not report["ok"] and registry is None


@pytest.mark.parametrize(
    "confirmation",
    ["operator_confirmed", "consent_confirmed", "independent_original_confirmed"],
)
def test_confirmation_is_required(sheet, confirmation):
    root, original = sheet
    doc = deepcopy(original)
    doc["rows"][0][confirmation] = False
    assert validate_intake(doc, root)[1] is None


def test_missing_matrix_and_duplicate_bytes(sheet):
    root, original = sheet
    doc = deepcopy(original)
    doc["rows"].pop()
    assert not validate_intake(doc, root)[0]["ok"]
    doc = deepcopy(original)
    doc["rows"][1]["capture"]["source_file_uri"] = doc["rows"][0]["capture"][
        "source_file_uri"
    ]
    report, registry, _ = validate_intake(doc, root)
    assert registry is None and report["duplicate_sources"]


def test_explicit_real_distinct_sessions_preserved(sheet):
    root, original = sheet
    doc = deepcopy(original)
    for r in doc["rows"]:
        if r["set_number"] == 2:
            r["capture"].update(
                session_id="SYNTHETIC-occasion2", date="2026-01-02T10:00:00+05:30"
            )
    assert validate_intake(doc, root)[1].summary()["subject_sessions"] == 4


def test_protected_row_refused_before_source_access(sheet, monkeypatch):
    root, original = sheet
    doc = deepcopy(original)
    for r in doc["rows"]:
        r["capture"]["split_role"] = "locked_final"
    monkeypatch.setattr(
        "gait_verification.pilot_intake.probe_video",
        lambda *a: pytest.fail("protected media accessed"),
    )
    assert validate_intake(doc, root)[1] is None
