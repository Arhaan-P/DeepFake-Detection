# Future-work experiment results

All experiments share the protocol in `utils/verification_harness.py` (13-fold leave-one-subject-out, training-subject normalisation, strict enrolment, exhaustive claims). AUC/EER in %. dAUC = paired mean difference of per-fold AUC vs `E0_baseline` (positive = better); CI = subject-bootstrap 95% interval of the pooled-AUC difference; p = Wilcoxon signed-rank over folds x seeds.


## P0 - frozen baseline and protocol checks

| Experiment | Params | Pooled AUC | Fold AUC | EER | TPR@5%FPR | AUC side | AUC frontal | Rank-1 ID | ECE→Platt | dAUC | 95% CI | p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E0_baseline | 133,058 | 85.24 ± 0.30 | 85.28 ± 8.96 | 22.35 | 36.36 | 88.01 | 82.10 | 40.4 | 11.8→1.3 |  |  |  |
| E0_legacy_enrol | 133,058 | 82.82 ± 0.60 | 83.62 ± 10.14 | 23.88 | 28.28 | 83.33 | 82.72 | 39.4 | 10.6→1.4 | -1.66 | [-3.65, +0.54] | 0.0304 * |
| E0_paper_aug | 133,058 | 83.58 ± 0.68 | 83.70 ± 10.88 | 24.83 | 32.83 | 85.80 | 81.22 | 39.4 | 16.3→1.3 | -1.57 | [-3.61, +0.34] | 0.0926 |

- `E0_baseline`: Frozen baseline: MediaPipe lite, 78-D, whole-clip resampled to 60, deployed temporal-CNN verifier, strict protocol. Also scores every robustness condition (E8).
- `E0_legacy_enrol`: E0 with the original enrolment rule (query clip included in its own claimed signature) -- measures that leakage.
- `E0_paper_aug`: E0 with mirror + 0.8/1.2x speed augmentation, closer to the paper's video augmentations.

## E1 - pose backend (P1)

| Experiment | Params | Pooled AUC | Fold AUC | EER | TPR@5%FPR | AUC side | AUC frontal | Rank-1 ID | ECE→Platt | dAUC | 95% CI | p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E1_mediapipe_full | 133,058 | 83.29 ± 0.17 | 84.70 ± 10.82 | 24.35 | 35.86 | 83.36 | 83.44 | 44.9 | 11.9→1.6 | -0.58 | [-6.69, +2.66] | 0.728 |
| E1_mediapipe_heavy | 133,058 | 82.52 ± 1.57 | 82.71 ± 11.28 | 25.25 | 39.90 | 86.29 | 77.77 | 41.4 | 12.0→1.8 | -2.56 | [-6.43, +1.17] | 0.145 |
| E1_mediapipe_heavy_video | 133,058 | 74.88 ± 2.49 | 76.94 ± 20.60 | 29.71 | 26.77 | 80.06 | 68.79 | 30.8 | 12.2→1.2 | -8.33 | [-24.37, -0.09] | 0.0201 * |
| E1_mediapipe_lite_2d | 100,802 | 84.16 ± 0.66 | 86.24 ± 7.80 | 25.25 | 36.87 | 85.98 | 82.23 | 46.5 | 13.0→1.9 | +0.96 | [-3.87, +2.13] | 0.635 |
| E1_mediapipe_lite_video | 133,058 | 83.25 ± 1.20 | 85.13 ± 9.09 | 24.37 | 25.25 | 84.54 | 82.07 | 38.4 | 11.9→1.2 | -0.14 | [-4.40, +1.75] | 0.851 |
| E1_rtmpose_2d | 100,802 | 85.90 ± 0.98 | 87.27 ± 11.79 | 22.18 | 40.40 | 87.19 | 84.69 | 46.0 | 11.5→1.4 | +1.99 | [-2.22, +5.35] | 0.0939 |
| E1_rtmw3d | 133,058 | 79.98 ± 0.66 | 81.24 ± 10.39 | 27.27 | 27.27 | 81.37 | 78.69 | 34.3 | 11.0→1.4 | -4.03 | [-7.91, +2.32] | 0.0416 * |
| E1_vitpose_2d | 100,802 | 84.77 ± 2.33 | 86.10 ± 11.93 | 23.95 | 40.91 | 86.58 | 83.05 | 47.0 | 12.1→1.6 | +0.82 | [-4.77, +4.59] | 0.477 |

- `E1_mediapipe_full`: Pose backend: MediaPipe full model.
- `E1_mediapipe_heavy`: Pose backend: MediaPipe heavy model.
- `E1_mediapipe_heavy_video`: Pose backend: MediaPipe heavy with VIDEO-mode temporal tracking.
- `E1_mediapipe_lite_2d`: Matched 2D control: baseline with z dropped, for comparison with 2D-only detectors.
- `E1_mediapipe_lite_video`: Pose backend: MediaPipe lite with VIDEO-mode temporal tracking.
- `E1_rtmpose_2d`: Pose backend: RTMPose-m Halpe26 (2D), matched 2D descriptor.
- `E1_rtmw3d`: Pose backend: RTMW3D keypoints (image x,y + model depth) in the unchanged 78-D descriptor.
- `E1_vitpose_2d`: Pose backend: ViTPose-B coco_25 (2D), matched 2D descriptor.

## E2 - RGB-only 3D pose

| Experiment | Params | Pooled AUC | Fold AUC | EER | TPR@5%FPR | AUC side | AUC frontal | Rank-1 ID | ECE→Platt | dAUC | 95% CI | p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E2_mediapipe_heavy_world | 133,058 | 81.89 ± 0.61 | 82.17 ± 11.42 | 26.70 | 32.83 | 85.01 | 77.88 | 37.4 | 13.2→1.1 | -3.10 | [-8.28, +1.81] | 0.167 |
| E2_mediapipe_lite_world | 133,058 | 84.82 ± 1.22 | 85.08 ± 8.38 | 22.66 | 31.31 | 86.53 | 83.19 | 38.4 | 11.4→0.8 | -0.20 | [-4.23, +4.75] | 0.895 |
| E2_rtmw3d_world | 133,058 | 85.97 ± 1.76 | 87.14 ± 10.01 | 21.25 | 34.85 | 89.48 | 81.52 | 42.4 | 9.3→1.4 | +1.86 | [-2.40, +6.82] | 0.361 |

- `E2_mediapipe_heavy_world`: RGB-only 3D: MediaPipe heavy world landmarks.
- `E2_mediapipe_lite_world`: RGB-only 3D: MediaPipe lite metric world landmarks instead of image landmarks.
- `E2_rtmw3d_world`: RGB-only 3D: RTMW3D 3D skeleton.

## E3 - gait cycle and rhythm (P2)

| Experiment | Params | Pooled AUC | Fold AUC | EER | TPR@5%FPR | AUC side | AUC frontal | Rank-1 ID | ECE→Platt | dAUC | 95% CI | p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E3_cycle | 133,058 | 81.28 ± 2.32 | 83.32 ± 11.62 | 25.53 | 34.85 | 82.57 | 79.95 | 39.9 | 12.7→1.0 | -1.96 | [-5.91, -0.40] | 0.107 |
| E3_cycle_rhythm_all | 174,722 | 81.38 ± 0.66 | 83.67 ± 9.61 | 25.76 | 32.32 | 80.64 | 82.51 | 34.8 | 9.7→1.7 | -1.60 | [-7.69, +2.07] | 0.622 |
| E3_dtw | 0 | 46.84 ± 0.00 | 46.97 ± 6.00 | 50.00 | 1.52 | 42.42 | 46.91 | 1.5 | 41.5→0.3 | -38.30 | [-44.51, -34.84] | 0.000244 * |
| E3_frequency | 154,562 | 85.11 ± 0.74 | 85.83 ± 10.94 | 23.63 | 41.92 | 85.76 | 84.90 | 43.9 | 12.3→1.0 | +0.56 | [-3.00, +2.41] | 0.557 |
| E3_phase | 138,434 | 84.45 ± 0.97 | 84.16 ± 9.69 | 24.20 | 41.41 | 84.82 | 84.36 | 38.9 | 11.1→1.4 | -1.12 | [-2.66, +1.24] | 0.308 |
| E3_rhythm | 147,842 | 85.19 ± 0.56 | 86.14 ± 8.71 | 22.73 | 37.88 | 87.83 | 81.89 | 41.4 | 11.3→1.3 | +0.86 | [-2.28, +2.36] | 0.267 |
| E3_rhythm_all | 174,722 | 84.82 ± 0.72 | 86.07 ± 10.18 | 22.66 | 40.91 | 86.01 | 83.51 | 42.4 | 10.8→1.3 | +0.79 | [-4.15, +2.52] | 0.387 |

- `E3_cycle`: RQ4: baseline families, gait-cycle-normalised (mean heel-strike-to-heel-strike cycle) instead of whole-clip resampling.
- `E3_cycle_rhythm_all`: Cycle normalisation + all rhythm families.
- `E3_dtw`: Non-learned DTW template matching on the baseline descriptor (speed-invariant alignment).
- `E3_frequency`: Baseline + harmonic spectra of knee angles, pelvis vertical, heel separation.
- `E3_phase`: Baseline + per-frame left/right gait phase (sin, cos).
- `E3_rhythm`: Baseline + clip-level rhythm (cadence, stride time/CV, regularity, harmonic ratio, phase synchrony).
- `E3_rhythm_all`: Baseline + rhythm + frequency + phase.

## E4 - biomechanical feature families (P3)

| Experiment | Params | Pooled AUC | Fold AUC | EER | TPR@5%FPR | AUC side | AUC frontal | Rank-1 ID | ECE→Platt | dAUC | 95% CI | p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E4_acceleration | 181,442 | 83.97 ± 0.25 | 84.83 ± 8.62 | 24.64 | 32.32 | 86.26 | 81.40 | 40.4 | 10.8→1.3 | -0.45 | [-2.40, +0.88] | 0.572 |
| E4_angles_3d | 141,122 | 83.34 ± 0.62 | 83.37 ± 11.07 | 23.40 | 34.85 | 84.15 | 82.60 | 35.9 | 10.7→1.1 | -1.91 | [-4.54, +1.02] | 0.124 |
| E4_angles_ext | 143,810 | 86.31 ± 0.43 | 86.36 ± 9.39 | 21.72 | 40.91 | 88.85 | 83.13 | 42.9 | 11.3→1.3 | +1.09 | [-0.06, +1.97] | 0.114 |
| E4_angular_velocity | 141,122 | 85.10 ± 1.33 | 85.02 ± 9.59 | 23.30 | 36.87 | 87.53 | 82.29 | 42.4 | 11.7→1.1 | -0.25 | [-1.21, +1.15] | 0.944 |
| E4_biomech | 225,794 | 83.53 ± 0.92 | 83.93 ± 12.26 | 22.77 | 36.87 | 85.83 | 80.68 | 42.4 | 9.4→1.1 | -1.34 | [-4.08, +1.48] | 0.415 |
| E4_com | 138,434 | 83.12 ± 1.64 | 84.19 ± 12.18 | 23.76 | 40.40 | 85.49 | 80.20 | 42.9 | 11.4→1.0 | -1.09 | [-4.43, +0.29] | 0.251 |
| E4_coords_angles | 84,674 | 85.63 ± 0.33 | 85.59 ± 8.85 | 23.70 | 32.83 | 87.09 | 84.63 | 41.9 | 13.4→1.6 | +0.31 | [-1.86, +2.99] | 0.922 |
| E4_coords_only | 76,610 | 85.54 ± 1.05 | 85.86 ± 8.55 | 22.64 | 39.90 | 87.82 | 82.62 | 43.4 | 16.1→1.3 | +0.59 | [-2.07, +2.25] | 0.562 |
| E4_foot | 141,122 | 85.16 ± 0.19 | 84.95 ± 10.62 | 24.24 | 34.34 | 87.46 | 82.62 | 41.9 | 11.4→1.8 | -0.33 | [-1.26, +1.18] | 0.816 |
| E4_full | 267,458 | 81.20 ± 3.22 | 82.91 ± 13.50 | 25.23 | 36.87 | 82.55 | 79.98 | 42.9 | 8.9→1.3 | -2.37 | [-7.59, +1.34] | 0.217 |
| E4_interpretable | 116,930 | 75.34 ± 1.72 | 77.04 ± 11.64 | 31.71 | 23.74 | 76.69 | 73.54 | 27.3 | 12.0→0.7 | -8.24 | [-16.64, -4.04] | 5.67e-05 * |
| E4_jerk | 181,442 | 86.78 ± 1.05 | 87.48 ± 8.10 | 22.12 | 37.88 | 88.92 | 84.19 | 43.9 | 9.2→1.3 | +2.20 | [-0.40, +4.18] | 0.0245 * |
| E4_scaled | 133,058 | 82.39 ± 1.46 | 85.04 ± 6.89 | 25.76 | 30.30 | 82.53 | 82.03 | 38.9 | 11.9→1.4 | -0.23 | [-7.02, +2.26] | 0.421 |
| E4_stride | 146,498 | 85.74 ± 2.05 | 85.91 ± 10.06 | 21.32 | 43.43 | 87.22 | 83.88 | 42.9 | 11.9→1.4 | +0.63 | [-1.90, +4.86] | 0.805 |
| E4_symmetry | 139,778 | 85.74 ± 0.00 | 86.19 ± 9.45 | 22.73 | 41.41 | 87.76 | 83.57 | 46.0 | 10.8→1.2 | +0.91 | [-1.26, +3.65] | 0.412 |

- `E4_acceleration`: Baseline + acceleration (torso lengths / s^2).
- `E4_angles_3d`: Baseline + the six angles computed in 3D.
- `E4_angles_ext`: Baseline + trunk lean, pelvic obliquity, shoulder tilt, foot pitch, inter-thigh, shank angles.
- `E4_angular_velocity`: Baseline + joint angular velocities.
- `E4_biomech`: Baseline + all biomechanical families.
- `E4_com`: Baseline + centre-of-mass proxy (pelvis sway, trunk vector).
- `E4_coords_angles`: Feature-group decomposition: coordinates + angles (no velocity).
- `E4_coords_only`: Feature-group decomposition: coordinates only.
- `E4_foot`: Baseline + heel/toe heights and toe vertical velocity.
- `E4_full`: Baseline + all biomechanical and rhythm families.
- `E4_interpretable`: Only interpretable biomechanics/rhythm, no raw coordinates or velocities.
- `E4_jerk`: Baseline + jerk.
- `E4_scaled`: Baseline with coordinates in torso lengths (distance-invariant).
- `E4_stride`: Baseline + step/stride length, width, stance and double-support fractions, foot clearance.
- `E4_symmetry`: Baseline + per-frame left-right symmetry signals.

## E7 - verifier architecture

| Experiment | Params | Pooled AUC | Fold AUC | EER | TPR@5%FPR | AUC side | AUC frontal | Rank-1 ID | ECE→Platt | dAUC | 95% CI | p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E7_raw_bilstm | 379,074 | 82.37 ± 1.77 | 82.85 ± 9.05 | 23.76 | 23.23 | 84.05 | 80.63 | 22.7 | 10.2→2.6 | -2.42 | [-3.34, +2.95] | 0.159 |
| E7_raw_freq | 257,156 | 85.85 ± 0.67 | 85.75 ± 8.92 | 22.73 | 39.90 | 87.68 | 83.72 | 40.4 | 11.6→1.3 | +0.48 | [-1.05, +3.34] | 0.879 |
| E7_raw_stgcn | 323,626 | 83.34 ± 1.83 | 83.66 ± 9.25 | 24.20 | 30.81 | 84.41 | 82.12 | 33.3 | 11.0→1.6 | -1.61 | [-3.69, +3.40] | 0.24 |
| E7_raw_transformer | 712,002 | 81.86 ± 1.29 | 82.58 ± 11.26 | 24.79 | 24.24 | 83.09 | 80.52 | 26.8 | 14.9→1.5 | -2.70 | [-4.17, +0.84] | 0.029 * |
| E7_siamese | 59,970 | 77.21 ± 2.38 | 76.96 ± 12.61 | 28.26 | 17.68 | 75.75 | 79.27 | 15.2 | 15.3→1.2 | -8.31 | [-13.07, +0.28] | 0.00206 * |
| E7_stgcn_only | 218,794 | 81.07 ± 1.16 | 81.23 ± 9.89 | 26.39 | 25.25 | 81.27 | 80.86 | 25.8 | 13.6→1.1 | -4.05 | [-7.84, +0.24] | 0.00707 * |

- `E7_raw_bilstm`: Deployed path + BiLSTM encoder comparison.
- `E7_raw_freq`: RQ5: deployed path + spectral comparison branch.
- `E7_raw_stgcn`: RQ9: deployed path + skeleton-graph (ST-GCN) encoder comparison.
- `E7_raw_transformer`: Deployed path + Transformer encoder comparison.
- `E7_siamese`: Metric learning: shared embedding, cosine-similarity verification.
- `E7_stgcn_only`: RQ9: ST-GCN encoder comparison replacing raw differencing.

## E8 - cross-view and robustness (P5)

| Experiment | Params | Pooled AUC | Fold AUC | EER | TPR@5%FPR | AUC side | AUC frontal | Rank-1 ID | ECE→Platt | dAUC | 95% CI | p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E8_cross_view_F2S | 133,058 | 75.66 ± 1.93 | 76.72 ± 15.79 | 30.74 | 16.22 | 75.66 | – | 28.8 | 12.5→1.2 |  |  |  |
| E8_cross_view_S2F | 133,058 | 77.08 ± 2.63 | 81.26 ± 15.05 | 29.89 | 24.14 | – | 77.08 | 29.9 | 15.6→1.9 |  |  |  |
| E8_full_robustness | 267,458 | 81.20 ± 3.22 | 82.91 ± 13.50 | 25.23 | 36.87 | 82.55 | 79.98 | 42.9 | 8.9→1.3 | -2.37 | [-7.59, +1.34] | 0.217 |
| E8_same_view | 133,058 | 76.38 ± 0.79 | 77.13 ± 14.80 | 30.16 | 25.76 | 77.70 | 74.95 | 26.3 | 12.1→0.6 | -8.14 | [-15.53, -4.09] | 2.66e-05 * |
| E8_scaled_robustness | 133,058 | 82.39 ± 1.46 | 85.04 ± 6.89 | 25.76 | 30.30 | 82.53 | 82.03 | 38.9 | 11.9→1.4 | -0.23 | [-7.02, +2.26] | 0.421 |

- `E8_cross_view_F2S`: RQ6: enrol on frontal walks, verify side walks.
- `E8_cross_view_S2F`: RQ6: enrol on side walks, verify frontal walks.
- `E8_full_robustness`: Robustness of the full engineered descriptor under the same suite.
- `E8_same_view`: Signatures built from the query's own view only.
- `E8_scaled_robustness`: Does distance-invariant scaling buy robustness? Same perturbation suite as E0.
