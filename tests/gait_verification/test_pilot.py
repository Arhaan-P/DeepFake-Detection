"""Targeted temporal/paired engineering regressions; no training or calibrators."""

from dataclasses import replace
import numpy as np
import pytest
from gait_verification.contracts import (
    PoseSequence,
    Status,
    GaitEmbedding,
    ModelMetadata,
)
from gait_verification.pose import (
    NativePoseExtractor,
    save_pose,
    load_pose,
    RTMPoseDetector,
)
from gait_verification.pose_quality import (
    analyze_pose,
    WindowQualityPolicy,
    slice_pose,
    WindowedEncoder,
    assess_window_quality,
)
from gait_verification.pilot import compare_reports
from gait_verification.skeleton_contracts import InsufficientGait


def moving_pose(n=240):
    # Anatomical markers + alternating ankle excursions, not real-video proof.
    points = np.array(
        [
            [100, 30],
            [95, 25],
            [105, 25],
            [90, 30],
            [110, 30],
            [80, 60],
            [120, 60],
            [75, 85],
            [125, 85],
            [75, 110],
            [125, 110],
            [90, 120],
            [110, 120],
            [90, 160],
            [110, 160],
            [90, 200],
            [110, 200],
        ],
        float,
    )
    xy = np.broadcast_to(points, (n, 17, 2)).copy()
    xy[:, 15, 0] += 8 * np.sin(np.arange(n) / 5)
    xy[:, 16, 0] -= 8 * np.sin(np.arange(n) / 5)
    return PoseSequence(
        xy,
        np.ones((n, 17)),
        np.ones((n, 17), bool),
        np.arange(n) / 60,
        "coco17",
        320,
        240,
        "take",
        "hash",
        "detector:display-rotation=0",
        np.arange(n),
        np.zeros(n, int),
        np.tile([70, 20, 130, 210], (n, 1)),
    )


def test_long_video_occasional_missing_ankle_keeps_unchanged_segments():
    s = moving_pose(1800)
    s.observed[180, 15] = False
    s.confidence[180, 15] = 0
    before = s.xy.copy()
    times = s.timestamps.copy()
    r = analyze_pose(s)
    assert r["summary"]["usable"] and len(r["windows"]) == 15
    assert r["summary"]["accepted_window_count"] == 15
    assert all(not (a <= 180 < b) for a, b in r["accepted_segments"])
    np.testing.assert_array_equal(s.xy, before)
    np.testing.assert_array_equal(s.timestamps, times)
    for a, b in r["accepted_segments"]:
        fragment = slice_pose(s, a, b)
        assert fragment.observed.all()
        np.testing.assert_array_equal(fragment.frame_indices, s.frame_indices[a:b])


@pytest.mark.parametrize(
    "fault,reason",
    [
        ("ankles", "missing_or_cropped_ankles"),
        ("gap", "timestamp_gap"),
        ("static", "insufficient_motion_evidence"),
        ("short", "too_short"),
    ],
)
def test_failure_categories(fault, reason):
    s = moving_pose()
    if fault == "ankles":
        s.xy[:, 15:17, 1] = 239
    elif fault == "gap":
        s.timestamps[70:] += 1
    elif fault == "static":
        s.xy[:] = s.xy[0]
    elif fault == "short":
        s = slice_pose(s, 0, 20)
    r = analyze_pose(s)
    assert any(reason in w["rejection_reasons"] for w in r["windows"])
    if fault != "gap":
        assert not r["summary"]["usable"]


def test_missing_and_low_confidence_bone_values_do_not_bias_metrics():
    s = moving_pose()
    s.observed[:, 15] = False
    s.xy[:, 15] = 0
    s.confidence[:, 15] = 0
    r = analyze_pose(s)
    assert r["summary"]["bone_length_cv"]["13-15"] is None
    assert not r["summary"]["usable"]


def test_tracking_break_splits_segments_and_keeps_potential_camera_category():
    s = moving_pose()
    s.track_ids[60:] = 1
    s.xy[60:] += 100
    r = analyze_pose(s)
    assert r["summary"]["tracking_break_count"] == 1
    assert not r["summary"]["usable"]
    assert r["summary"]["video_rejection_reasons"] == ["ambiguous_track_identity"]
    assert "tracking_failure" in r["frames"][60]["categories"]
    assert all(not (a < 60 < b) for a, b in r["accepted_segments"])


def test_windowed_encoder_only_gets_valid_segments_and_binds_recipe():
    class Encoder:
        metadata = ModelMetadata(
            "model", "repo", "rev", "sha", 1, "GREW", "native", "academic"
        )
        embedding_dim = None

        def encode(self, s):
            assert s.observed.all() and len(s.xy) >= 31
            return GaitEmbedding(
                np.array([1.0, 2.0]), "native", s.original_take_id, s.source_sha256
            )

    s = moving_pose()
    s.observed[70, 15] = False
    e = WindowedEncoder(Encoder())
    a = e.encode(s)
    assert np.isclose(np.linalg.norm(a.values), 1) and ":windows=" in a.model_id
    assert a.original_take_id == s.original_take_id and e.embedding_dim == 2
    with pytest.raises(InsufficientGait):
        e.encode(moving_pose(20))


def test_diagnostic_extractor_preserves_scores_and_ambiguous_frames(
    tmp_path, monkeypatch
):
    import gait_verification.pose as mod
    from gait_verification.contracts import VideoMetadata

    s = moving_pose(6)
    meta = VideoMetadata(
        "fixture", "hash", 320, 240, "60/1", "60/1", "1/60", tuple(s.timestamps), "test"
    )
    monkeypatch.setattr(mod, "probe_video", lambda path: meta)
    monkeypatch.setattr(
        mod,
        "decode_frames",
        lambda m: (
            (i, t, np.zeros((240, 320, 3), np.uint8))
            for i, t in enumerate(m.timestamps)
        ),
    )

    class Detector:
        layout = "coco17"
        backend_id = "fixture"
        i = 0

        def __call__(self, frame):
            self.i += 1
            if self.i == 2:
                return np.empty((0, 17, 2)), np.empty((0, 17))
            if self.i == 4:
                return np.stack([s.xy[0], s.xy[0]]), np.ones((2, 17))
            scores = np.ones((1, 17)) * 1.1
            scores[0, 0] = 0
            return s.xy[:1], scores

    seq = NativePoseExtractor(Detector(), "take", collect_diagnostics=True).extract(
        "fixture"
    )
    assert seq.detection_counts.tolist() == [1, 0, 1, 2, 1, 1]
    assert seq.tracking_events[3] == 2 and seq.track_ids[3] == -1
    assert seq.xy[0, 0, 0] == s.xy[0, 0, 0] and not seq.observed[0, 0]
    assert seq.confidence[0, 15] == 1.1
    assert seq.candidate_confidence[3, 1, 15] == 1
    assert seq.candidate_observed[3, 1, 15]
    save_pose(seq, tmp_path / "pose.npz")
    restored = load_pose(tmp_path / "pose.npz")
    np.testing.assert_array_equal(restored.tracking_events, seq.tracking_events)
    save_pose(s, tmp_path / "old.npz")
    assert load_pose(tmp_path / "old.npz").detection_counts is None


def triple():
    reports = []
    for v in ("raw", "vidstab-s10", "vidstab-s15"):
        r = analyze_pose(replace(moving_pose(), variant_id=v))
        r.update(original_source_sha256="original", timing_consistent=True)
        reports.append(r)
    return reports


def test_paired_comparison_equal_original_weights():
    reports = triple()
    reports += [dict(r, original_take_id="second") for r in triple()]
    r = compare_reports(reports)
    assert r["independent_original_count"] == 2
    assert (
        r["paired_deltas_from_raw"]["take"]["vidstab-s10"]["valid_ankle_fraction"] == 0
    )


@pytest.mark.parametrize(
    "fault", ["missing", "duplicate", "policy", "source", "time", "model"]
)
def test_paired_comparison_rejects_invalid_pairs(fault):
    r = triple()
    if fault == "missing":
        r.pop()
    elif fault == "duplicate":
        r.append(r[0])
    elif fault == "policy":
        r[1]["policy"] = dict(r[1]["policy"], min_confidence=0.8)
    elif fault == "source":
        r[1]["original_source_sha256"] = "another"
    elif fault == "time":
        r[1]["timing_consistent"] = False
    elif fault == "model":
        r[1]["backend_id"] = "another-model"
    with pytest.raises(ValueError):
        compare_reports(r)


def test_pose_artifact_checksum_fails_before_runtime_import(tmp_path):
    path = tmp_path / "fake.onnx"
    path.write_bytes(b"fake")
    with pytest.raises(ValueError, match="checksum"):
        RTMPoseDetector(path, path, expected_detector_sha256="0" * 64)


def test_temporal_quality_abstains_before_encoder_or_boolean():
    from gait_verification.verifier import ClaimVerifier

    class Store:
        def get_claim(self, claim):
            return (object(),)

    class Extractor:
        def extract(self, path):
            return moving_pose(20)

    v = ClaimVerifier(
        Store(), object(), Extractor(), quality_assessor=assess_window_quality
    )
    result = v.verify(__file__, "S1")
    assert result.status == Status.INSUFFICIENT_GAIT and result.verdict is None


def test_invalid_window_policy_rejected():
    with pytest.raises(ValueError):
        analyze_pose(moving_pose(), WindowQualityPolicy(window_seconds=0))


def test_no_person_detection_never_uses_full_image_pose_fallback():
    from types import SimpleNamespace

    detector = RTMPoseDetector.__new__(RTMPoseDetector)

    def forbidden(*args, **kwargs):
        raise AssertionError("Missing person cannot reach full-image pose fallback")

    detector.runtime = SimpleNamespace(
        det_model=lambda image: np.empty((0, 4)), pose_model=forbidden
    )
    points, scores = detector(np.zeros((32, 32, 3), np.uint8))
    assert points.shape == (0, 17, 2) and scores.shape == (0, 17)
