"""
Tests for the future-work modules (no dataset, GPU or MediaPipe needed).
=======================================================================
Uses a synthetic side-view walker with KNOWN stride frequency, so the
gait-cycle maths is checked against ground truth, plus equivalence checks
that tie the new code to the frozen baseline:

  * the vectorised 78-D descriptor equals GaitFeatureExtractor's loops
    (skipped when MediaPipe is not importable)
  * TCNVerifier with copied weights gives bit-identical logits to the
    deployed diff_conv / diff_classifier path, and has 133,058 parameters

Runs under pytest, or directly:
    python tests/test_future_work.py
"""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from utils.gait_cycle import analyse_gait, cycle_normalise, dtw, dtw_warp
from utils.gait_descriptors import Clip, build_descriptor
from utils.gait_features import (
    FAMILY_DIMS,
    FEATURE_SETS,
    baseline_descriptor,
    descriptor_dim,
    resample_sequence,
)
from utils.keypoint_augment import (
    ROBUSTNESS_SUITE,
    augment_clip,
    mirror_pose,
    perturb_clip,
)
from utils.verification_metrics import (
    apply_platt,
    calibration_summary,
    eer,
    fit_platt,
    roc_auc,
    wilcoxon,
)


def synthetic_walker(
    freq=0.9, amp=0.45, thigh=0.12, shank=0.12, seconds=6.0, fps=30.0, seed=0
):
    """Side-view walker in canonical 33-slot layout. Returns (pose, t)."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * fps)) / fps
    n = len(t)
    pose = np.zeros((n, 33, 3))
    hx = 0.2 + 0.1 * t
    hy = 0.5 + 0.004 * np.sin(4 * np.pi * freq * t)
    phase = 2 * np.pi * freq * t
    for side, off in (("L", 0.0), ("R", np.pi)):
        th = amp * np.sin(phase + off)
        flex = 0.6 * np.clip(np.sin(phase + off - np.pi / 2), 0, None)
        hip = np.stack([hx, hy, np.full(n, 0.01 if side == "L" else -0.01)], 1)
        knee = hip + np.stack([thigh * np.sin(th), thigh * np.cos(th), 0 * t], 1)
        ank = knee + np.stack(
            [shank * np.sin(th - flex), shank * np.cos(th - flex), 0 * t], 1
        )
        heel = ank + np.array([-0.015, 0.01, 0.0])
        toe = ank + np.array([0.035, 0.015, 0.0])
        idx = {"L": (23, 25, 27, 29, 31, 11), "R": (24, 26, 28, 30, 32, 12)}[side]
        for j, p in zip(idx[:5], (hip, knee, ank, heel, toe)):
            pose[:, j] = p
        pose[:, idx[5]] = hip + np.array([0.0, -0.25, 0.0])
    for j in range(33):  # remaining landmarks: park on the trunk
        if not pose[:, j].any():
            pose[:, j] = pose[:, 11]
    pose += rng.normal(0, 0.0005, pose.shape)
    return pose, t


def make_clip(name="Synth_S1", **kw):
    pose, t = synthetic_walker(**kw)
    ident, view = name.split("_")[0], name.split("_")[1][0]
    return Clip(name, ident, view, "1", pose, t, 30.0, 1.0)


# ------------------------------------------------------------------ features


def test_baseline_descriptor_matches_frozen_extractor():
    try:
        from utils.pose_extraction import GaitFeatureExtractor
    except Exception as e:  # mediapipe missing / broken in this environment
        print(f"  skipped (pose_extraction not importable: {type(e).__name__})")
        return
    pose, _ = synthetic_walker()
    ext = object.__new__(GaitFeatureExtractor)
    ext.sequence_length = 60
    p60 = ext._normalize_sequence_length(pose)
    assert np.allclose(p60, resample_sequence(pose, 60))
    gf = ext.compute_gait_features(p60)
    ref = np.concatenate(
        [
            gf["normalized_coords"].reshape(60, -1),
            gf["joint_angles"],
            gf["velocities"].reshape(60, -1),
        ],
        axis=1,
    )
    assert np.allclose(baseline_descriptor(p60), ref, atol=1e-9)


def test_every_feature_set_builds_with_declared_dims():
    clip = make_clip()
    for name, fams in FEATURE_SETS.items():
        for mode in ("clip", "cycle"):
            d = build_descriptor(clip, fams, mode)
            assert d.shape == (60, descriptor_dim(fams)), (name, mode, d.shape)
            assert np.isfinite(d).all(), (name, mode)
    assert descriptor_dim(FEATURE_SETS["baseline78"]) == 78
    assert set(FAMILY_DIMS) >= {f for fs in FEATURE_SETS.values() for f in fs}


# ------------------------------------------------------------------ gait cycle


def test_gait_cycle_recovers_known_rhythm():
    for freq in (0.8, 0.95, 1.1):
        pose, t = synthetic_walker(freq=freq)
        ga = analyse_gait(pose, t, 30.0)
        assert abs(ga.stride_period - 1 / freq) / (1 / freq) < 0.06, (
            freq,
            ga.stride_period,
        )
        r = ga.rhythm_dict()
        assert abs(r["cadence_spm"] - 120 * freq) < 8, (freq, r["cadence_spm"])
        assert r["phase_locking"] > 0.9  # clean anti-phase legs
        assert r["phase_offset"] < 0.1
        assert len(ga.hs_l) >= 3 and len(ga.hs_r) >= 3
        cycles = cycle_normalise(pose.reshape(len(t), -1), t, ga.hs_l, 50)
        assert cycles.shape[1:] == (50, 99)


def test_dtw_is_zero_on_identity_and_undoes_time_warp():
    a = np.sin(np.linspace(0, 4 * np.pi, 60))[:, None]
    d0, _ = dtw(a, a)
    assert d0 < 1e-12
    warped = np.sin(np.linspace(0, 4 * np.pi, 60) ** 1.1 / (4 * np.pi) ** 0.1)[:, None]
    d_raw = np.abs(warped - a).mean()
    d_dtw, _ = dtw(warped, a, band=15)
    assert d_dtw < d_raw
    assert np.abs(dtw_warp(warped, a, band=15) - a).mean() < d_raw


# ------------------------------------------------------------------ augmentation


def test_augmentation_and_perturbations_keep_shapes():
    clip = make_clip()
    rng = np.random.default_rng(0)
    for preset in ("rhythm_safe", "paper_like"):
        aug = augment_clip(clip, rng, preset)
        assert aug.pose.shape[1:] == (33, 3) and len(aug.pose) == len(aug.t)
        assert np.all(np.diff(aug.t) > 0)
    for spec in ROBUSTNESS_SUITE:
        p = perturb_clip(clip, spec)
        assert build_descriptor(p, FEATURE_SETS["baseline78"]).shape == (60, 78), spec
    assert np.allclose(mirror_pose(mirror_pose(clip.pose)), clip.pose)


# ------------------------------------------------------------------ metrics


def test_metrics_against_known_values():
    y = np.array([0, 0, 1, 1])
    assert roc_auc(y, np.array([0.1, 0.4, 0.35, 0.8])) == 0.75
    assert roc_auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert eer(y, np.array([0.1, 0.2, 0.8, 0.9]))[0] == 0.0
    rng = np.random.default_rng(0)
    yy = rng.integers(0, 2, 4000)
    p = np.clip(yy * 0.6 + rng.normal(0.2, 0.15, 4000), 0.01, 0.99)
    ab = fit_platt(yy, p)
    cal = calibration_summary(yy, apply_platt(p, ab))
    assert cal["ece"] < calibration_summary(yy, p)["ece"] + 1e-9
    assert np.all(np.diff(apply_platt(np.linspace(0.05, 0.95, 20), ab)) > 0)
    assert wilcoxon(np.ones(20) + 0.1, np.ones(20))["p"] < 0.01


# ------------------------------------------------------------------ models


def test_tcn_verifier_is_the_deployed_decision_path():
    from models.full_pipeline import create_model
    from models.verifier_variants import (
        VERIFIERS,
        TCNVerifier,
        build_verifier,
        count_parameters,
    )

    deployed = create_model({"verification_hidden": 64, "dropout": 0.1}).eval()
    tcn = TCNVerifier(78, hidden=64, dropout=0.1).eval()
    assert count_parameters(tcn) == 133058
    tcn.head.conv.load_state_dict(deployed.diff_conv.state_dict())
    tcn.head.classifier.load_state_dict(deployed.diff_classifier.state_dict())
    v, c = torch.randn(3, 60, 78), torch.randn(3, 60, 78)
    with torch.no_grad():
        ref = deployed(v, c, mode="verification")["verification"]["logits"]
        assert torch.allclose(tcn(v, c), ref, atol=1e-6)

    from utils.gait_features import family_slices

    sl = family_slices(FEATURE_SETS["baseline78"])
    for name in VERIFIERS:
        m = build_verifier(name, 78, sl)
        assert m(v, c).shape == (3, 2), name


# ------------------------------------------------------------------ harness


def test_harness_end_to_end_on_synthetic_subjects():
    from utils.pose_backends import PoseTrack
    from utils.verification_harness import ExperimentConfig, run_experiment

    with tempfile.TemporaryDirectory() as tmp:
        cache = os.path.join(tmp, "synthetic")
        os.makedirs(cache)
        people = {"Ann": (0.8, 0.40), "Bob": (0.95, 0.50), "Cid": (1.1, 0.35)}
        for k, (who, (f, a)) in enumerate(people.items()):
            for take in range(1, 4):
                pose, t = synthetic_walker(freq=f, amp=a, seed=10 * k + take)
                view = "S" if take < 3 else "F"
                PoseTrack(
                    pose, np.ones((len(t), 33)), 30.0, 100, 100, "synthetic"
                ).save(os.path.join(cache, f"{who}_{view}{take}.npz"))
        cfg = ExperimentConfig(
            name="synthetic",
            backend="synthetic",
            cache_root=tmp,
            n_aug=2,
            epochs=2,
            seeds=[0],
            perturbations=["noise:0.01"],
            feature_set="b+rhythm",
        )
        res = run_experiment(
            cfg, descriptor_cache=os.path.join(tmp, "desc"), log=lambda m: None
        )
        assert res["n_trials"] == 9 * 3
        assert 0.0 <= res["aggregate"]["pooled_roc_auc_mean"] <= 1.0
        assert "noise:0.01" in res["aggregate"]["robustness_auc"]


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for n, f in tests:
        print(f"{n} ...", flush=True)
        f()
    print(f"\nALL {len(tests)} FUTURE-WORK TESTS PASSED")
