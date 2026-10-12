"""Synthetic fixtures test engineering contracts, never recognition accuracy."""

from dataclasses import replace
import json
import numpy as np
import pytest
from gait_verification.contracts import (
    PoseSequence,
    GaitEmbedding,
    Recording,
    QualityAssessment,
    Status,
    VerificationResult,
)
from gait_verification.skeleton_contracts import (
    gaitpt_grew_inputs,
    coco_arrays,
    GAITPT_ORDER,
    MEDIAPIPE_TO_COCO,
    InsufficientGait,
    assess_quality,
)
from gait_verification.enrollments import MemoryEnrollmentStore, EnrollmentTemplate
from gait_verification.verifier import ClaimVerifier, VerificationPolicy
from gait_verification.evaluation import (
    generate_trials,
    verification_metrics,
    validate_recordings,
    manifest_hash,
    evaluate_trials,
)
from gait_verification.pose import save_pose, load_pose
from gait_verification.stabilization import StabilizationConfig, stabilization_commands


def pose(n=90, layout="coco17"):
    joints = {"coco17": 17, "halpe26": 26, "mediapipe33": 33}[layout]
    xy = (
        np.broadcast_to(
            np.stack([100 + np.arange(joints) * 3, 50 + np.arange(joints) * 5], -1),
            (n, joints, 2),
        )
        .copy()
        .astype(float)
    )
    # Marker at each joint and time makes mapping and crop errors visible.
    xy[..., 0] += np.arange(n)[:, None] * 0.1
    return PoseSequence(
        xy,
        np.ones((n, joints)),
        np.ones((n, joints), bool),
        np.arange(n) / 60,
        layout,
        640,
        480,
        "probe",
        "probe-hash",
        "synthetic",
        np.arange(n),
        np.zeros(n, int),
        np.tile([100, 50, 220, 200], (n, 1)),
    )


def recording(subject="S1", take="r1", split="development", **kw):
    return Recording(
        take,
        take,
        subject,
        "session1",
        "left45",
        "2026-10-12T10:00:00+05:30",
        take + "-hash",
        split,
        **kw,
    )


def embedding(values=(1.0, 0.0), take="probe"):
    return GaitEmbedding(np.array(values), "test-recipe", take, take + "-hash")


def store():
    s = MemoryEnrollmentStore()
    s.enroll("S1", [EnrollmentTemplate("S1", embedding(take="r1"), recording())])
    s.enroll(
        "S2",
        [EnrollmentTemplate("S2", embedding((0.0, 1.0), "r2"), recording("S2", "r2"))],
    )
    return s


def verifier(s=None, policy=True):
    p = (
        VerificationPolicy("test-recipe", 0.8, "test-only", test_only=True)
        if policy
        else None
    )
    return ClaimVerifier(s or store(), policy=p, allow_test_policy=True)


def test_anatomical_mapping_and_height_scaling():
    seq = pose()
    batch = gaitpt_grew_inputs(seq)
    assert batch.shape == (2, 30, 18, 2)
    np.testing.assert_allclose(batch[0], seq.xy[7:37, GAITPT_ORDER] / 480, atol=1e-7)
    # Nose is duplicated; never substitute a neck/hip or an unrelated joint.
    np.testing.assert_array_equal(batch[:, :, 0], batch[:, :, 3])
    mp = pose(layout="mediapipe33")
    np.testing.assert_array_equal(coco_arrays(mp)[0], mp.xy[:, MEDIAPIPE_TO_COCO])
    halpe = pose(layout="halpe26")
    np.testing.assert_array_equal(coco_arrays(halpe)[0], halpe.xy[:, :17])


@pytest.mark.parametrize(
    "field,value",
    [
        ("xy", np.zeros((90, 16, 2))),
        ("xy", np.zeros((90, 17, 3))),
        ("xy", np.full((90, 17, 2), np.nan)),
        ("confidence", np.full((90, 17), np.inf)),
        ("timestamps", np.zeros(90)),
        ("timestamps", np.full(90, np.nan)),
        ("layout", "legacy78"),
        ("observed", np.ones((90, 17))),
    ],
)
def test_malformed_pose_rejected(field, value):
    with pytest.raises(ValueError):
        gaitpt_grew_inputs(replace(pose(), **{field: value}))


def test_missing_essential_and_short_sequences_abstain():
    seq = pose()
    seq.observed[:, 15] = False
    assert not assess_quality(seq).usable
    with pytest.raises(InsufficientGait):
        gaitpt_grew_inputs(seq)
    with pytest.raises(InsufficientGait):
        gaitpt_grew_inputs(pose(30))


def test_pose_cache_keeps_pts_provenance_and_refuses_overwrite(tmp_path):
    seq = pose()
    path = tmp_path / "pose.npz"
    save_pose(seq, path)
    restored = load_pose(path)
    np.testing.assert_array_equal(seq.timestamps, restored.timestamps)
    np.testing.assert_array_equal(seq.xy, restored.xy)
    assert restored.source_sha256 == seq.source_sha256
    with pytest.raises(FileExistsError):
        save_pose(seq, path)


def test_native_simcc_scores_above_one_preserved(tmp_path):
    seq = pose()
    seq.confidence[0, 15] = 1.12
    save_pose(seq, tmp_path / "native.npz")
    restored = load_pose(tmp_path / "native.npz")
    np.testing.assert_array_equal(restored.confidence, seq.confidence)
    assert assess_quality(restored).usable
    seq.confidence[0, 15] = -0.1
    with pytest.raises(ValueError, match="Negative confidence"):
        seq.validate()


def test_true_false_and_abstention_paths():
    v = verifier()
    quality = QualityAssessment(True)
    assert v.verify_embedding(embedding(), "S1", quality).verdict is True
    assert v.verify_embedding(embedding((0, 1)), "S1", quality).verdict is False
    r = v.verify_embedding(embedding(), "S1", QualityAssessment(False))
    assert r.status == Status.INSUFFICIENT_GAIT and r.verdict is None
    assert (
        v.verify_embedding(embedding(), "absent", quality).status
        == Status.CLAIM_NOT_ENROLLED
    )
    assert (
        verifier(policy=False).verify_embedding(embedding(), "S1", quality).status
        == Status.POLICY_NOT_READY
    )


def test_only_claimed_templates_loaded_no_alternative_disclosure():
    class Spy:
        calls = []

        def get_claim(self, claim):
            self.calls.append(claim)
            assert claim == "S1"
            return store().get_claim(claim)

    spy = Spy()
    r = verifier(spy).verify_embedding(embedding((0, 1)), "S1", QualityAssessment(True))
    assert spy.calls == ["S1"]
    assert set(r.public_dict()) == {
        "schema_version",
        "claimed_subject_id",
        "status",
        "verdict",
    }
    assert "S2" not in json.dumps(r.public_dict())


def test_result_invariants_and_no_test_policy_in_production():
    with pytest.raises(ValueError):
        VerificationResult("S1", Status.INSUFFICIENT_GAIT, False)
    with pytest.raises(ValueError):
        VerificationResult("S1", Status.OK)
    policy = VerificationPolicy("test-recipe", 0.8, test_only=True)
    v = ClaimVerifier(store(), policy=policy)
    assert (
        v.verify_embedding(embedding(), "S1", QualityAssessment(True)).status
        == Status.POLICY_NOT_READY
    )
    assert (
        verifier()
        .verify_embedding(embedding(), "../S1", QualityAssessment(True))
        .status
        == Status.INVALID_INPUT
    )


def test_invalid_embeddings_recipe_and_probe_reference_overlap():
    v = verifier()
    assert (
        v.verify_embedding(embedding(take="r1"), "S1", QualityAssessment(True)).status
        == Status.INVALID_INPUT
    )
    bad = replace(embedding(), model_id="another-checkpoint")
    # Phase 1C: no reference with the probe's recipe means no valid enrollment.
    assert (
        v.verify_embedding(bad, "S1", QualityAssessment(True)).status
        == Status.CLAIM_NOT_ENROLLED
    )
    for values in [(0, 0), (float("nan"), 0), (1, 2, 3)]:
        r = v.verify_embedding(embedding(values), "S1", QualityAssessment(True))
        assert r.status == Status.PROCESSING_ERROR and r.verdict is None


def test_enrollment_rejects_duplicate_derivatives_and_foreign_claims():
    s = MemoryEnrollmentStore()
    t = store().get_claim("S1")[0]
    with pytest.raises(ValueError):
        s.enroll("S1", [t, t])
    with pytest.raises(ValueError):
        s.enroll("S2", [t])


def test_trials_deterministic_grouped_and_no_source_overlap():
    records = [recording(s, s + str(i)) for s in ("S1", "S2") for i in range(2)]
    a, b = generate_trials(records), generate_trials(list(reversed(records)))
    assert a == b and len(a) == 8 and manifest_hash(a) == manifest_hash(b)
    assert sum(t.label_same_identity for t in a) == 4
    assert all(t.query_original_take_id not in t.reference_original_take_ids for t in a)
    report = evaluate_trials(
        a, lambda t: (Status.OK, 0.9 if t.label_same_identity else 0.1), threshold=0.8
    )
    assert report["overall"]["roc_auc"] == 1 and len(report["by_subject_session"]) == 2


@pytest.mark.parametrize("change", ["split", "session", "hash_alias", "orphan"])
def test_leakage_rejected(change):
    r = recording()
    if change == "split":
        other = replace(recording(take="r2"), split_role="locked_final")
    elif change == "session":
        other = replace(
            r,
            asset_id="child",
            variant_id="stabilized",
            parent_asset_id="r1",
            session_id="different",
        )
    elif change == "hash_alias":
        other = replace(recording(take="r2"), source_sha256=r.source_sha256)
    else:
        other = replace(
            r, asset_id="child", variant_id="stabilized", parent_asset_id="missing"
        )
    with pytest.raises(ValueError):
        validate_recordings([r, other])


def test_metrics_known_values_ties_and_coverage():
    m = verification_metrics([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], threshold=0.5)
    assert m["roc_auc"] == 0.75 and m["eer"] == 0.5
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (1, 0, 2, 1)
    assert m["tar_at_far"] == 0.5
    tie = verification_metrics([0, 1], [0.5, 0.5])
    assert tie["roc_auc"] == 0.5 and tie["eer"] == 0.5 and tie["tar_at_far"] == 0
    abstain = verification_metrics(
        [0, 1], [None, 0.7], [Status.INSUFFICIENT_GAIT, Status.OK]
    )
    assert (
        abstain["coverage"] == 0.5
        and abstain["roc_auc"] is None
        and "undefined_reason" in abstain
    )
    with pytest.raises(ValueError):
        verification_metrics([0, 1], [0.1, 0.7], [Status.INSUFFICIENT_GAIT, Status.OK])


def test_stabilization_is_opt_in_and_never_overwrites():
    assert not StabilizationConfig().enabled
    for smoothing in (10, 15):
        commands = stabilization_commands(
            "/tmp/source.mp4",
            "/tmp/derivative.mp4",
            StabilizationConfig(True, smoothing=smoothing),
        )
        assert all("-n" in c and "-y" not in c for c in commands)
        assert (
            f"smoothing={smoothing}:optzoom=0:crop=black:interpol=bicubic"
            in commands[1][commands[1].index("-vf") + 1]
        )
        assert (
            "-r" not in commands[1]
            and "mincontrast=0.2" in commands[0][commands[0].index("-vf") + 1]
        )


def test_private_directory_store_claim_lookup_and_no_overwrite(tmp_path):
    from gait_verification.enrollments import DirectoryEnrollmentStore

    s = DirectoryEnrollmentStore(tmp_path)
    templates = tuple(
        replace(t, enrollment_id="enr-1") for t in store().get_claim("S1")
    )
    s.enroll("S1", templates)
    assert s.get_claim("S2") == ()
    assert s.get_claim("S1")[0].recording == templates[0].recording
    with pytest.raises(FileExistsError):
        s.enroll("S1", templates)


def test_video_orchestration_quality_and_errors(tmp_path):
    video = tmp_path / "fixture.mp4"
    video.write_bytes(b"x")  # extractor is injected; only existence is checked

    class Extractor:
        def extract(self, path):
            return pose(12)

    class Encoder:
        def encode(self, seq):
            raise AssertionError("Bad quality must not reach encoder")

    v = ClaimVerifier(store(), Encoder(), Extractor())
    assert v.verify(video, "S1").status == Status.INSUFFICIENT_GAIT

    class Broken:
        def extract(self, path):
            raise RuntimeError("internal path should not appear in public response")

    result = ClaimVerifier(store(), Encoder(), Broken()).verify(video, "S1")
    assert result.status == Status.PROCESSING_ERROR and result.verdict is None
    assert "internal" not in json.dumps(result.public_dict())


def test_bootstrap_clusters_subjects_and_handles_degenerate_groups():
    from gait_verification.evaluation import subject_bootstrap

    args = (
        [0, 1, 0, 1],
        [0.1, 0.9, 0.2, 0.8],
        [Status.OK] * 4,
        ["S1", "S1", "S2", "S2"],
    )
    result = subject_bootstrap(*args, repetitions=20, seed=5)
    assert result["roc_auc_ci95"] == [1, 1]
    assert result == subject_bootstrap(*args, repetitions=20, seed=5)
    assert (
        subject_bootstrap([1], [0.9], [Status.OK], ["S1"])["status"]
        == "UNDEFINED_METRIC"
    )
