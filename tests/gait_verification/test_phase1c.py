"""Phase 1C: enrollment store, claimed-subject scoring and video-to-claim orchestration.

Sections are separated by evidence type:
  * mocked unit tests        - synthetic embeddings and stub extractors/encoders
  * media integration tests  - real ffmpeg media + real native pose extractor/cache/quality
                               with an injected (fake) detector and encoder
  * real CUDA (opt-in)       - GAIT_V2_CUDA_TESTS=1, frozen GaitPT on a historical pose cache
Nothing here measures recognition accuracy; synthetic vectors only test contracts.
"""

from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
import numpy as np
import pytest
from gait_verification import enrollments
from gait_verification.contracts import (
    GaitEmbedding,
    ModelMetadata,
    QualityAssessment,
    Status,
)
from gait_verification.enrollments import (
    CorruptEnrollment,
    DirectoryEnrollmentStore,
    EnrollmentTemplate,
    MemoryEnrollmentStore,
    _row,
)
from gait_verification.media import InvalidMedia, probe_video
from gait_verification.scoring import BASELINE, STRATEGIES, prototype, score_claim
from gait_verification.skeleton_contracts import InsufficientGait
from gait_verification.verifier import ClaimVerifier, VerificationPolicy
from test_foundation import recording

REPO = Path(__file__).resolve().parents[2]
MODEL = "gaitpt:ck:recipe:cfg:pipeline=p:windows=w"


def emb(values=(1.0, 0.0), take="probe", model=MODEL):
    return GaitEmbedding(np.array(values, float), model, take, take + "-hash")


def tmpl(subject="S1", take="r1", values=(1.0, 0.0), model=MODEL, **changes):
    t = EnrollmentTemplate(
        subject,
        emb(values, take, model),
        recording(subject, take),
        1.0,
        f"enr-{subject}-{take}-{model.split(':')[0]}",
        {"consent_ref": "test"},
    )
    return replace(t, **changes)


def good(quality=True):
    return QualityAssessment(quality)


def make_policy(model=MODEL, threshold=0.8):
    return VerificationPolicy(model, threshold, "TEST_ONLY", test_only=True)


# ------------------------------------------------------------------ enrollment store


def test_multiple_independent_originals_kept_individually(tmp_path):
    s = DirectoryEnrollmentStore(tmp_path)
    s.enroll("S1", [tmpl(take="r1", values=(3, 4))])
    s.add_reference("S1", tmpl(take="r2", values=(0, 2)))
    s.add_reference("S1", tmpl(take="r3", values=(1, 1)))
    refs = s.get_claim("S1")
    assert [t.recording.original_take_id for t in refs] == ["r1", "r2", "r3"]
    np.testing.assert_allclose(refs[0].embedding.values, [0.6, 0.8], atol=1e-7)
    # Stored vectors are unit-norm individually and survive prototype construction.
    assert all(np.isclose(np.linalg.norm(t.embedding.values), 1) for t in refs)
    p = prototype(refs)
    expected = np.mean([t.embedding.values for t in refs], axis=0)
    np.testing.assert_allclose(p, expected / np.linalg.norm(expected), atol=1e-6)
    assert np.isclose(np.linalg.norm(p), 1)
    assert len(s.get_claim("S1")) == 3  # nothing was collapsed on disk


def test_duplicate_derivative_and_conflicting_hash_rejected(tmp_path):
    s = DirectoryEnrollmentStore(tmp_path)
    s.add_reference("S1", tmpl())
    with pytest.raises(ValueError, match="already enrolled"):
        s.add_reference("S1", replace(tmpl(), enrollment_id="enr-other"))
    r = recording(take="r9")
    for bad in (
        replace(r, asset_id="r9-s10", variant_id="vidstab-s10", parent_asset_id="r9"),
        replace(r, variant_id="vidstab-s10"),
        replace(r, parent_asset_id="r9"),
    ):
        with pytest.raises(ValueError, match="original"):
            s.add_reference("S1", replace(tmpl(take="r9"), recording=bad))
    with pytest.raises(ValueError, match="Conflicting source checksums"):
        s.add_reference(
            "S1",
            replace(
                tmpl(take="r4"),
                recording=replace(recording(take="r4"), source_sha256="other"),
            ),
        )
    # Same take, different recipe, different bytes: conflicting checksum.
    other = tmpl(take="r1", model="gpgait:ck:recipe:cfg")
    forged = replace(
        other,
        embedding=replace(other.embedding, source_sha256="forged"),
        recording=replace(other.recording, source_sha256="forged"),
    )
    with pytest.raises(ValueError, match="Conflicting source checksums"):
        s.add_reference("S1", forged)
    # Same bytes under two take IDs.
    aliased = tmpl(take="r5")
    aliased = replace(
        aliased,
        embedding=replace(aliased.embedding, source_sha256="r1-hash"),
        recording=replace(aliased.recording, source_sha256="r1-hash"),
    )
    with pytest.raises(ValueError):
        s.add_reference("S1", aliased)
    assert len(s.get_claim("S1")) == 1


def test_cross_subject_source_reuse_rejected_without_reading_other_templates(tmp_path):
    s = DirectoryEnrollmentStore(tmp_path)
    s.add_reference("S1", tmpl("S1", "r1"))
    stolen = tmpl("S2", "r1x")
    stolen = replace(
        stolen,
        embedding=replace(stolen.embedding, source_sha256="r1-hash"),
        recording=replace(stolen.recording, source_sha256="r1-hash"),
    )
    with pytest.raises(ValueError, match="another subject"):
        s.add_reference("S2", stolen)
    assert s.get_claim("S2") == ()
    # The index stores hashes only, never embeddings or subject names.
    index = json.loads((tmp_path / "source-index.json").read_text())
    assert "S1" not in json.dumps(index) and "values" not in json.dumps(index)


def test_invalid_subjects_path_traversal_and_missing_ids(tmp_path):
    s = DirectoryEnrollmentStore(tmp_path / "store")
    for claim in ("../x", "a/b", "", "x" * 81, "S 1", "S1\n", "..", "%2e%2e"):
        with pytest.raises(ValueError):
            s.add_reference(claim, tmpl())
        with pytest.raises(ValueError):
            s.get_claim(claim)
    with pytest.raises(ValueError, match="enrollment_id"):
        s.add_reference("S1", tmpl(enrollment_id=""))
    with pytest.raises(ValueError, match="enrollment_id"):
        s.add_reference("S1", tmpl(enrollment_id="../../etc"))
    # Nothing escaped the store root.
    assert all(p.is_relative_to(tmp_path / "store") for p in tmp_path.rglob("*"))
    assert (
        sorted(p.name for p in tmp_path.iterdir()) == ["store"]
        or not (tmp_path / "x").exists()
    )


@pytest.mark.parametrize(
    "values", [(0.0, 0.0), (float("nan"), 1.0), (float("inf"), 0.0), ()]
)
def test_zero_and_nonfinite_embeddings_not_enrollable(tmp_path, values):
    s = DirectoryEnrollmentStore(tmp_path)
    with pytest.raises(ValueError):
        s.add_reference("S1", tmpl(values=values))
    assert s.get_claim("S1") == ()


def test_corrupt_store_never_overwritten_and_fails_closed(tmp_path):
    s = DirectoryEnrollmentStore(tmp_path)
    s.add_reference("S1", tmpl())
    path = s._path("S1")
    for content in (b"{truncated", b"\xff\xfe", b'{"schema_version": "enrollment-v2"}'):
        path.write_bytes(content)
        with pytest.raises(CorruptEnrollment):
            s.get_claim("S1")
        with pytest.raises(CorruptEnrollment):
            s.add_reference("S1", tmpl(take="r2"))
        assert path.read_bytes() == content
        r = ClaimVerifier(s).verify_embedding(emb(), "S1", good())
        assert r.status == Status.PROCESSING_ERROR and r.verdict is None
    # A claim file copied under another claim's hashed name is refused.
    s2 = DirectoryEnrollmentStore(tmp_path / "b")
    s2.add_reference("S1", tmpl())
    shutil.copy(s2._path("S1"), s2._path("S2"))
    with pytest.raises(CorruptEnrollment):
        s2.get_claim("S2")


def test_atomic_updates_survive_replace_failure(tmp_path, monkeypatch):
    s = DirectoryEnrollmentStore(tmp_path)
    s.add_reference("S1", tmpl(take="r1"))
    before = s._path("S1").read_bytes()
    real, calls = os.replace, []

    def flaky(src, dst):
        calls.append(dst)
        if len(calls) == 2:  # index replaced, claim replace fails
            raise OSError("disk full")
        return real(src, dst)

    monkeypatch.setattr(enrollments.os, "replace", flaky)
    with pytest.raises(OSError):
        s.add_reference("S1", tmpl(take="r2"))
    monkeypatch.setattr(enrollments.os, "replace", real)
    assert s._path("S1").read_bytes() == before  # prior approved file untouched
    assert not list(tmp_path.glob(".*.tmp"))
    s.add_reference("S1", tmpl(take="r2"))  # orphan index entry does not block a retry
    assert len(s.get_claim("S1")) == 2


def test_serialization_reproducible_permissions_and_schema(tmp_path):
    a, b = (
        DirectoryEnrollmentStore(tmp_path / "a"),
        DirectoryEnrollmentStore(tmp_path / "b"),
    )
    for s in (a, b):
        s.enroll("S1", [tmpl(take="r1"), tmpl(take="r2", values=(0, 1))])
    assert a._path("S1").read_bytes() == b._path("S1").read_bytes()
    assert oct(a._path("S1").stat().st_mode & 0o777) == "0o600"
    assert oct((tmp_path / "a").stat().st_mode & 0o777) == "0o700"
    data = json.loads(a._path("S1").read_text())
    assert data["schema_version"] == "enrollment-v2" and "values" in json.dumps(data)
    with pytest.raises(FileExistsError):
        a.enroll("S1", [tmpl(take="r9")])
    (tmp_path / "a" / "source-index.json").unlink()  # verification never needs it
    assert len(a.get_claim("S1")) == 2
    data["schema_version"] = "enrollment-v9"
    a._path("S1").write_text(json.dumps(data))
    with pytest.raises(CorruptEnrollment):
        a.get_claim("S1")


def test_v1_files_remain_readable_and_extendable(tmp_path):
    s = DirectoryEnrollmentStore(tmp_path)
    row = _row(tmpl())
    for k in ("enrollment_id", "provenance"):
        row.pop(k)
    tmp_path.mkdir(exist_ok=True)
    s._path("S1").write_text(
        json.dumps({"schema_version": "enrollment-v1", "templates": [row]})
    )
    (old,) = s.get_claim("S1")
    assert old.enrollment_id == "" and old.provenance == {}
    s.add_reference("S1", tmpl(take="r2"))
    assert len(s.get_claim("S1")) == 2


def test_revocation_keeps_record_and_allows_explicit_replacement(tmp_path):
    s = DirectoryEnrollmentStore(tmp_path)
    s.add_reference("S1", tmpl(take="r1", values=(1, 0)))
    s.add_reference("S1", tmpl(take="r2", values=(0, 1)))
    with pytest.raises(ValueError):
        s.revoke_reference("S1", "missing", "why")
    with pytest.raises(ValueError, match="reason"):
        s.revoke_reference("S1", "enr-S1-r1-gaitpt", "")
    s.revoke_reference("S1", "enr-S1-r1-gaitpt", "operator error")
    assert [t.recording.original_take_id for t in s.get_claim("S1")] == ["r2"]
    on_disk = json.loads(s._path("S1").read_text())
    assert len(on_disk["references"]) == 2 and on_disk["revocations"]
    with pytest.raises(ValueError, match="already revoked"):
        s.revoke_reference("S1", "enr-S1-r1-gaitpt", "again")
    # Replacement re-uses the take only under a new enrollment ID, never silently.
    s.add_reference("S1", replace(tmpl(take="r1"), enrollment_id="enr-S1-r1-v2"))
    assert len(s.get_claim("S1")) == 2


def test_one_claim_holds_independent_recipes_without_pooling(tmp_path):
    s = DirectoryEnrollmentStore(tmp_path)
    other = "gpgait:ck:recipe:cfg:pipeline=p:windows=w"
    s.add_reference("S1", tmpl(take="r1"))
    s.add_reference(
        "S1", tmpl(take="r1", model=other, values=(0, 1, 0))
    )  # same original
    s.add_reference("S1", tmpl(take="r2", model=other, values=(0, 0, 1)))
    refs = s.get_claim("S1")
    assert len(refs) == 3
    v = ClaimVerifier(s, policy=make_policy(other), allow_test_policy=True)
    probe = emb((0, 1, 0), "q", other)
    result, diag = v.verify_embedding_detailed(probe, "S1", good())
    assert (
        diag["reference_count"] == 2 and diag["model_id"] == other
    )  # gaitpt untouched
    assert result.status == Status.OK


# ------------------------------------------------------------------ scoring


def test_cosine_scores_and_dev_only_strategies():
    refs = [tmpl(take="a", values=(1, 0)), tmpl(take="b", values=(0, 1))]
    q = emb((1, 0))
    assert score_claim(q, refs).score == pytest.approx(np.sqrt(0.5), abs=1e-6)
    assert score_claim(q, refs, "mean_cosine").score == pytest.approx(0.5)
    assert score_claim(q, refs, "max_cosine").score == 1.0
    assert score_claim(q, refs, "top2_mean_cosine").score == pytest.approx(0.5)
    assert score_claim(emb((0.6, 0.8)), refs[:1]).score == pytest.approx(0.6)
    with pytest.raises(ValueError):
        score_claim(q, refs, "nearest_other_subject")
    # Alternatives can never back an operational policy.
    assert BASELINE == STRATEGIES[0]
    for name in STRATEGIES[1:]:
        policy = VerificationPolicy(MODEL, 0.5, "x", "a" * 64, aggregation=name)
        assert not policy.ready()


@pytest.mark.parametrize(
    "probe,refs",
    [
        (emb((1, 0, 0)), [tmpl(values=(1, 0))]),  # dimension mismatch
        (emb((float("nan"), 0)), [tmpl(values=(1, 0))]),
        (emb((0, 0)), [tmpl(values=(1, 0))]),
        (emb((1, 0)), [tmpl(take="a", values=(1, 0)), tmpl(take="b", values=(-1, 0))]),
    ],
)
def test_degenerate_scoring_inputs_fail_closed(probe, refs):
    with pytest.raises(ValueError):
        score_claim(probe, refs)
    store = MemoryEnrollmentStore()
    store.enroll("S1", refs)
    r = ClaimVerifier(
        store, policy=make_policy(), allow_test_policy=True
    ).verify_embedding(probe, "S1", good())
    assert r.status == Status.PROCESSING_ERROR and r.verdict is None


# ------------------------------------------------------------------ verification contract


class Spy:
    """Fails loudly if anything other than the claimed subject is requested."""

    def __init__(self, allowed, records):
        self.allowed, self.records, self.calls = allowed, records, []

    def get_claim(self, claim):
        self.calls.append(claim)
        assert claim == self.allowed, f"looked up {claim}"
        return self.records.get(claim, ())


def test_claim_only_lookup_no_identity_disclosure_and_no_probe_leak():
    records = {
        "S1": (tmpl("S1", "r1", (1, 0)),),
        "S2": (tmpl("S2", "r2", (0, 1)),),
    }
    spy = Spy("S1", records)
    v = ClaimVerifier(spy, policy=make_policy(), allow_test_policy=True)
    result, diag = v.verify_embedding_detailed(emb((0, 1)), "S1", good())
    assert (
        spy.calls == ["S1"] and result.verdict is False
    )  # impostor-like vector, no search
    public = json.dumps(result.public_dict())
    assert set(result.public_dict()) == {
        "schema_version",
        "claimed_subject_id",
        "status",
        "verdict",
    }
    assert "S2" not in public and str(diag["score"])[:6] not in public
    assert "S2" not in json.dumps(diag)
    # An unrelated claim is a single lookup too (and cannot reach S1's records).
    spy2 = Spy("S9", records)
    assert (
        ClaimVerifier(spy2).verify_embedding(emb(), "S9", good()).status
        == Status.CLAIM_NOT_ENROLLED
    )
    assert spy2.calls == ["S9"]


@pytest.mark.parametrize(
    "probe_model",
    [
        "gpgait:ck:recipe:cfg:pipeline=p:windows=w",  # GaitPT vs GPGait
        "gpgaitpp:ck:recipe:cfg:pipeline=p:windows=w",  # GPGait vs GPGait++ differ
        "gaitpt:OTHERck:recipe:cfg:pipeline=p:windows=w",  # checkpoint
        "gaitpt:ck:recipe2:cfg:pipeline=p:windows=w",  # preprocessing
        "gaitpt:ck:recipe:cfg:pipeline=q:windows=w",  # pose / RAW vs stabilized
        "gaitpt:ck:recipe:cfg:pipeline=p:windows=other",  # window aggregation
        "gaitpt:ck:recipe:cfg:pipeline=p",  # whole sequence vs windowed
    ],
)
def test_incompatible_recipes_rejected_even_with_equal_dimensions(probe_model):
    store = MemoryEnrollmentStore()
    store.enroll(
        "S1", [tmpl(take="r1", values=(1, 0)), tmpl(take="r2", values=(0.9, 0.1))]
    )
    v = ClaimVerifier(store, policy=make_policy(probe_model), allow_test_policy=True)
    result, diag = v.verify_embedding_detailed(
        emb((1, 0), model=probe_model), "S1", good()
    )
    assert result.status == Status.CLAIM_NOT_ENROLLED and result.verdict is None
    assert "score" not in diag and diag["reason"] == "RECIPE_MISMATCH"


@pytest.mark.parametrize(
    "probe,lineage",
    [
        (emb(take="r1"), ()),  # same take
        (replace(emb(), source_sha256="r1-hash"), ()),  # same bytes, new take id
        (emb(take="s10-of-r1"), ("r1",)),  # derivative declares its original
        (emb(take="re-encoded"), ("r1-hash",)),  # copy declares original bytes
    ],
)
def test_same_original_leakage_rejected(probe, lineage):
    store = MemoryEnrollmentStore()
    store.enroll("S1", [tmpl(take="r1")])
    v = ClaimVerifier(store, policy=make_policy(), allow_test_policy=True)
    result, diag = v.verify_embedding_detailed(probe, "S1", good(), lineage)
    assert result.status == Status.INVALID_INPUT and result.verdict is None
    assert diag["reason"] == "SHARED_ORIGINAL" and "score" not in diag


def test_missing_policy_scores_internally_but_never_decides():
    store = MemoryEnrollmentStore()
    store.enroll("S1", [tmpl()])
    for v in (
        ClaimVerifier(store),
        ClaimVerifier(store, policy=make_policy()),  # test policy without allow flag
        ClaimVerifier(
            store, policy=VerificationPolicy(MODEL, 0.5), allow_test_policy=True
        ),
        ClaimVerifier(
            store, policy=make_policy("another-model"), allow_test_policy=True
        ),
    ):
        result, diag = v.verify_embedding_detailed(emb((1, 0)), "S1", good())
        assert result.status == Status.POLICY_NOT_READY and result.verdict is None
        assert diag["score"] == pytest.approx(1.0)  # restricted diagnostic only
        assert "score" not in json.dumps(result.public_dict())


def test_test_only_policy_true_false_and_blocked_models_stay_blocked():
    store = MemoryEnrollmentStore()
    store.enroll("S1", [tmpl(values=(1, 0))])
    v = ClaimVerifier(store, policy=make_policy(threshold=0.8), allow_test_policy=True)
    assert v.verify_embedding(emb((1, 0)), "S1", good()).verdict is True
    assert v.verify_embedding(emb((0, 1)), "S1", good()).verdict is False
    assert v.verify_embedding(emb((0.8, 0.6)), "S1", good()).verdict is True  # boundary
    # A "calibrated-looking" policy cannot decide for a BLOCKED encoder.
    real = VerificationPolicy(MODEL, 0.5, "x", "a" * 64)
    blocked = SimpleNamespace(metadata=SimpleNamespace(readiness="BLOCKED"))
    r = ClaimVerifier(store, blocked, policy=real).verify_embedding(emb(), "S1", good())
    assert r.status == Status.POLICY_NOT_READY


def test_configuration_and_cli_cannot_enable_a_decision():
    models = json.loads((REPO / "configs/gait_verification/models.json").read_text())
    assert all(m["production_ready"] is False for m in models.values())
    assert all(str(m["readiness"]).startswith("BLOCKED") for m in models.values())
    for name in ("phase0.json", "phase1a.json"):
        assert (
            json.loads((REPO / "configs/gait_verification" / name).read_text())[
                "threshold"
            ]
            is None
        )
    for script in ("verify_claim.py", "enroll_subject.py"):
        text = (REPO / "scripts/gait_verification" / script).read_text()
        for forbidden in ("allow_test_policy", "VerificationPolicy", "test_only="):
            assert forbidden not in text


# ------------------------------------------------------------------ status precedence


class Extractor:
    def __init__(self, exc=None):
        self.exc, self.calls = exc, 0

    def extract(self, path):
        self.calls += 1
        if self.exc:
            raise self.exc
        return SimpleNamespace(original_take_id="q", source_sha256="q-hash")


class Encoder:
    metadata = SimpleNamespace(readiness="BLOCKED")

    def __init__(self, exc=None, probe=None):
        self.exc, self.probe = exc, probe

    def encode(self, sequence):
        if self.exc:
            raise self.exc
        return self.probe or emb((1, 0))


def pipeline(store, extractor, encoder, usable=True, **kw):
    return ClaimVerifier(
        store, encoder, extractor, quality_assessor=lambda s: good(usable), **kw
    )


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "walk.mp4"
    path.write_bytes(b"not decoded: extractor is injected")
    return path


def enrolled_store():
    s = MemoryEnrollmentStore()
    s.enroll("S1", [tmpl()])
    return s


@pytest.mark.parametrize(
    "case,expected",
    [
        ("invalid_claim", Status.INVALID_INPUT),
        ("missing_video", Status.INVALID_INPUT),
        ("missing_video_and_unenrolled", Status.INVALID_INPUT),
        ("unenrolled_hides_bad_gait", Status.CLAIM_NOT_ENROLLED),
        ("corrupt_media", Status.INVALID_INPUT),
        ("abstaining_extractor", Status.INSUFFICIENT_GAIT),
        ("quality_gate", Status.INSUFFICIENT_GAIT),
        ("encoder_abstains", Status.INSUFFICIENT_GAIT),
        ("bad_gait_beats_missing_policy", Status.INSUFFICIENT_GAIT),
        ("extractor_crash", Status.PROCESSING_ERROR),
        ("encoder_crash", Status.PROCESSING_ERROR),
        ("encoder_value_error_is_not_invalid_input", Status.PROCESSING_ERROR),
        ("store_failure", Status.PROCESSING_ERROR),
        ("shared_original", Status.INVALID_INPUT),
        ("usable_no_policy", Status.POLICY_NOT_READY),
        ("pipeline_not_installed", Status.POLICY_NOT_READY),
    ],
)
def test_status_precedence_and_null_verdicts(case, expected, video, tmp_path):
    store, ext, enc, claim, path, usable = (
        enrolled_store(),
        Extractor(),
        Encoder(),
        "S1",
        video,
        True,
    )
    if case == "invalid_claim":
        claim = "../S1"
    elif case in ("missing_video", "missing_video_and_unenrolled"):
        path = tmp_path / "absent.mp4"
        claim = "S1" if case == "missing_video" else "nobody"
    elif case == "unenrolled_hides_bad_gait":
        claim, ext = "nobody", Extractor(InsufficientGait("x"))
    elif case == "corrupt_media":
        ext = Extractor(InvalidMedia("truncated"))
    elif case == "abstaining_extractor":
        ext = Extractor(InsufficientGait("Multiple people"))
    elif case == "quality_gate" or case == "bad_gait_beats_missing_policy":
        usable = False
    elif case == "encoder_abstains":
        enc = Encoder(InsufficientGait("no accepted windows"))
    elif case == "extractor_crash":
        ext = Extractor(RuntimeError("/private/path"))
    elif case == "encoder_crash":
        enc = Encoder(RuntimeError("CUDA OOM at /private"))
    elif case == "encoder_value_error_is_not_invalid_input":
        enc = Encoder(ValueError("Degenerate skeleton height"))
    elif case == "store_failure":
        store = SimpleNamespace(
            get_claim=lambda c: (_ for _ in ()).throw(OSError("/x"))
        )
    elif case == "shared_original":
        enc = Encoder(probe=emb(take="r1"))
    v = pipeline(store, ext, enc, usable)
    if case == "pipeline_not_installed":
        v = ClaimVerifier(store)
    result, diag = v.verify_detailed(path, claim)
    assert result.status == expected, (result, diag)
    assert result.verdict is None  # never False for an error/abstention
    assert "private" not in json.dumps(result.public_dict())
    if case == "unenrolled_hides_bad_gait":
        assert ext.calls == 0  # no expensive extraction for an unenrolled claim


# ------------------------------------------------------------------ real media (no GPU)

needs_ffmpeg = pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg required"
)

POINTS = np.array(
    [[100, 30], [95, 25], [105, 25], [90, 30], [110, 30], [80, 60], [120, 60],
     [75, 85], [125, 85], [75, 110], [125, 110], [90, 120], [110, 120],
     [90, 160], [110, 160], [90, 200], [110, 200]],
    float,
)  # fmt: skip


class FakeDetector:
    """Deterministic COCO17 'walker' for real decoded frames (not a pose model)."""

    layout, backend_id = "coco17", "fake-rtmpose:v1"

    def __init__(self, amplitude=8.0, people=1):
        self.amplitude, self.people, self.i, self.last_bboxes = (
            amplitude,
            people,
            0,
            None,
        )

    def __call__(self, frame):
        i, self.i = self.i, self.i + 1
        if self.people == 0:
            return np.empty((0, 17, 2)), np.empty((0, 17))
        p = POINTS.copy()
        p[15, 0] += self.amplitude * np.sin(i / 5)
        p[16, 0] -= self.amplitude * np.sin(i / 5)
        return np.stack([p] * self.people), np.ones((self.people, 17))


class StatEncoder:
    """Stand-in frozen encoder: unit vector of ankle-trajectory statistics."""

    min_frames = 31
    embedding_dim = 3
    metadata = ModelMetadata(
        "stat:recipe", "none", "none", "n/a", 0, "synthetic", "stat", "none"
    )

    def encode(self, s):
        a = s.xy[:, 15, 0]
        v = np.array([a.mean(), a.std(), 1.0])
        return GaitEmbedding(
            v / np.linalg.norm(v), "stat:recipe", s.original_take_id, s.source_sha256
        )


def make_video(path, duration):
    subprocess.run(
        ["ffmpeg", "-v", "error", "-n", "-f", "lavfi", "-i",
         f"testsrc2=size=240x320:rate=30:duration={duration}",
         "-c:v", "libx264", str(path)],
        check=True,
    )  # fmt: skip
    return path


@pytest.fixture
def media_stack(tmp_path):
    from gait_verification.pose_quality import (
        WindowQualityPolicy,
        WindowedEncoder,
    )
    from gait_verification.pipeline import CachedPoseExtractor

    policy = WindowQualityPolicy()
    return SimpleNamespace(
        tmp=tmp_path,
        policy=policy,
        encoder=WindowedEncoder(StatEncoder(), policy),
        extractor=lambda det, take=None, cache=None: CachedPoseExtractor(
            det, cache, take
        ),
    )


@needs_ffmpeg
def test_decode_pose_cache_windows_and_timestamps(media_stack):
    from gait_verification.pose import load_pose
    from gait_verification.pose_quality import analyze_pose

    video = make_video(media_stack.tmp / "a.mp4", 4.2)
    cache = media_stack.tmp / "cache"
    seq = media_stack.extractor(FakeDetector(), "take-a", cache).extract(video)
    meta = probe_video(video)
    np.testing.assert_array_equal(
        seq.timestamps, meta.timestamps
    )  # real PTS, no 1/fps clock
    assert seq.source_sha256 == meta.sha256 and seq.original_take_id == "take-a"
    assert seq.detection_counts is not None  # diagnostics collected
    (cached,) = cache.glob("*.npz")
    # Second extraction must come from the immutable cache, not the detector.

    class Forbidden(FakeDetector):
        def __call__(self, frame):
            raise AssertionError("detector re-run despite cache")

    again = media_stack.extractor(Forbidden(), "take-a", cache).extract(video)
    np.testing.assert_array_equal(again.xy, seq.xy)
    assert load_pose(cached).source_sha256 == meta.sha256
    report = analyze_pose(seq, media_stack.policy)
    assert report["summary"]["usable"] and report["accepted_segments"]
    # A cache entry whose provenance disagrees with the source is refused, never trusted.
    from gait_verification.pose import save_pose

    cached.unlink()
    save_pose(replace(seq, source_sha256="f" * 64), cached)
    with pytest.raises(ValueError, match="Pose cache"):
        media_stack.extractor(FakeDetector(), "take-a", cache).extract(video)


@needs_ffmpeg
def test_ambiguous_missing_and_corrupt_media(media_stack):
    from gait_verification.pose_quality import assess_window_quality

    video = make_video(media_stack.tmp / "a.mp4", 3.0)
    crowded = media_stack.extractor(FakeDetector(people=2), "t").extract(video)
    assert not assess_window_quality(crowded, media_stack.policy).usable
    empty = media_stack.extractor(FakeDetector(people=0), "t").extract(video)
    assert not assess_window_quality(empty, media_stack.policy).usable
    junk = media_stack.tmp / "junk.mp4"
    junk.write_bytes(b"\x00" * 4096)
    truncated = media_stack.tmp / "cut.mp4"
    truncated.write_bytes(video.read_bytes()[:3000])
    for bad in (junk, truncated):
        with pytest.raises(InvalidMedia):
            probe_video(bad)


@needs_ffmpeg
def test_enroll_then_claim_only_video_scoring_end_to_end(media_stack):
    from gait_verification.pipeline import (
        EnrollmentRefused,
        enroll_original,
        query_verifier,
    )

    tmp, policy, encoder = media_stack.tmp, media_stack.policy, media_stack.encoder
    store = DirectoryEnrollmentStore(tmp / "enroll")
    videos = {
        n: make_video(tmp / f"{n}.mp4", d)
        for n, d in (("a", 3.0), ("b", 3.2), ("q", 3.4))
    }

    def enroll(subject, name, amplitude=8.0, take=None, detector=None):
        take = take or f"{subject}-{name}"
        return enroll_original(
            videos[name],
            media_stack.extractor(detector or FakeDetector(amplitude), take),
            [encoder],
            store,
            policy,
            subject_id=subject,
            take_id=take,
            session_id="session1",
            view="left45",
            recorded_at="2026-10-12T10:00:00+05:30",
            consent_ref="TEST-ONLY",
        )

    (t1,) = enroll("S1", "a", 8.0)
    enroll("S1", "b", 6.0)
    refs = store.get_claim("S1")
    assert {r.recording.original_take_id for r in refs} == {"S1-a", "S1-b"}
    assert refs[0].provenance["window_aggregation"].startswith("equal-weight")
    assert refs[0].provenance["consent_ref"] == "TEST-ONLY"
    assert (
        ":windows=" in t1.embedding.model_id
        and t1.provenance["gait_model"]["readiness"] == "BLOCKED"
    )
    # Refusals leave the store untouched.
    with pytest.raises(ValueError):
        enroll("S1", "a", take="S1-again")  # same bytes under another take
    with pytest.raises(ValueError, match="another subject"):
        enroll("S2", "a")  # cross-subject reuse
    with pytest.raises(EnrollmentRefused):
        enroll("S2", "q", detector=FakeDetector(people=2))  # ambiguous tracks
    with pytest.raises(EnrollmentRefused):
        enroll("S2", "q", detector=FakeDetector(people=0))
    with pytest.raises(EnrollmentRefused, match="invalid_subject_id"):
        enroll("../S2", "q")
    assert store.get_claim("S2") == () and len(store.get_claim("S1")) == 2

    class CountingStore:
        calls = []

        def get_claim(self, claim):
            self.calls.append(claim)
            return store.get_claim(claim)

    counting = CountingStore()
    verifier = query_verifier(
        counting, encoder, media_stack.extractor(FakeDetector(7.0)), policy
    )
    result, diag = verifier.verify_detailed(videos["q"], "S1")
    assert counting.calls == ["S1"]
    assert result.status == Status.POLICY_NOT_READY and result.verdict is None
    assert diag["reference_count"] == 2 and -1 <= diag["score"] <= 1
    assert diag["strategy"] == BASELINE
    # Re-submitting an enrolled original is a leakage error, not positive evidence.
    leak = query_verifier(
        counting, encoder, media_stack.extractor(FakeDetector(8.0)), policy
    ).verify_detailed(videos["a"], "S1")[0]
    assert leak.status == Status.INVALID_INPUT and leak.verdict is None
    # Derivative of an enrolled original declares its lineage.
    derivative = query_verifier(
        counting, encoder, media_stack.extractor(FakeDetector(8.0)), policy
    ).verify(videos["q"], "S1", lineage_ids=("S1-a",))
    assert derivative.status == Status.INVALID_INPUT
    # Unenrolled claim, corrupt file, and crowded scene never reach a verdict.
    junk = tmp / "junk.mp4"
    junk.write_bytes(b"\x00" * 2048)
    for claim, path, det, status in (
        ("S2", videos["q"], FakeDetector(), Status.CLAIM_NOT_ENROLLED),
        ("S1", junk, FakeDetector(), Status.INVALID_INPUT),
        ("S1", videos["q"], FakeDetector(people=2), Status.INSUFFICIENT_GAIT),
    ):
        r = query_verifier(store, encoder, media_stack.extractor(det), policy).verify(
            path, claim
        )
        assert r.status == status and r.verdict is None


# ------------------------------------------------------------------ real CUDA (opt-in)

CUDA = os.environ.get("GAIT_V2_CUDA_TESTS") == "1"
POSE_CACHE = REPO / "outputs/gait_verification/historical-native-pose.npz"


@pytest.mark.skipif(
    not CUDA or not POSE_CACHE.exists(),
    reason="set GAIT_V2_CUDA_TESTS=1 (needs GPU + caches)",
)
@pytest.mark.parametrize("model", ["gaitpt", "gpgait", "gpgaitpp", "gpgaitpp_casia"])
def test_real_cuda_query_embedding_scored_against_claim_only(model):
    import torch
    from gait_verification.pose import load_pose
    from gait_verification.registry import load_encoder

    assert torch.cuda.is_available()
    encoder = load_encoder(
        model,
        REPO / ".cache/gait_verification/sources",
        REPO / ".cache/gait_verification/checkpoints",
    )
    sequence = load_pose(POSE_CACHE)
    probe = encoder.encode(sequence)
    assert ":windows=" in probe.model_id and np.isclose(np.linalg.norm(probe.values), 1)
    assert len(probe.values) == (256 if model == "gaitpt" else 4864)
    rng = np.random.default_rng(0)
    reference = replace(
        probe,
        values=rng.normal(size=len(probe.values)),
        original_take_id="synthetic-ref",
        source_sha256="synthetic-ref-hash",
    )
    store = MemoryEnrollmentStore()
    store.enroll(
        "S1",
        [EnrollmentTemplate("S1", reference, recording("S1", "synthetic-ref"))],
    )
    v = ClaimVerifier(store, encoder, quality_assessor=lambda s: good())
    result, diag = v.verify_embedding_detailed(probe, "S1", good())
    assert result.status == Status.POLICY_NOT_READY and result.verdict is None
    assert np.isfinite(diag["score"])
