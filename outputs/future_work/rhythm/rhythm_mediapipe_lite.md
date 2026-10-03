# Gait parameter reliability (mediapipe_lite, 66 clips)

| Parameter | ICC all | ICC side | ICC frontal | Fisher all | Fisher side | mean | within SD | between SD |
|---|---|---|---|---|---|---|---|---|
| step_width | 0.25 | 0.36 | 0.52 | 0.68 | 1.42 | 0.122 | 0.047 | 0.039 |
| double_support_frac | 0.23 | 0.37 | 0.59 | 0.56 | 1.47 | 0.287 | 0.069 | 0.051 |
| cadence_spm | 0.17 | 0.40 | -0.05 | 0.50 | 1.74 | 112.715 | 9.896 | 7.024 |
| step_length_r | 0.12 | 0.07 | 0.41 | 0.42 | 0.62 | 0.535 | 0.178 | 0.115 |
| stride_length | 0.11 | 0.16 | 0.47 | 0.40 | 0.82 | 1.082 | 0.328 | 0.208 |
| stride_time_s | 0.09 | 0.41 | -0.06 | 0.36 | 1.75 | 1.078 | 0.116 | 0.069 |
| stance_frac_r | 0.06 | 0.24 | 0.16 | 0.33 | 1.00 | 0.544 | 0.125 | 0.072 |
| step_length_l | 0.05 | 0.08 | 0.17 | 0.30 | 0.68 | 0.547 | 0.203 | 0.112 |
| stride_regularity | 0.02 | 0.15 | 0.84 | 0.26 | 0.83 | 0.785 | 0.167 | 0.084 |
| step_length_asym | 0.02 | -0.00 | 0.06 | 0.26 | 0.52 | 0.445 | 0.471 | 0.240 |
| step_time_asym | -0.01 | -0.02 | 0.42 | 0.23 | 0.57 | 0.307 | 0.333 | 0.161 |
| step_regularity | -0.02 | -0.05 | 0.11 | 0.19 | 0.50 | 0.792 | 0.196 | 0.086 |
| regularity_symmetry | -0.05 | 0.16 | -0.78 | 0.19 | 0.79 | 1.024 | 0.200 | 0.087 |
| stance_frac_l | -0.05 | -0.06 | -0.16 | 0.19 | 0.50 | 0.545 | 0.124 | 0.054 |
| stride_time_cv | -0.05 | -0.07 | -0.06 | 0.18 | 0.54 | 0.165 | 0.071 | 0.030 |
| clearance_r | -0.07 | -0.12 | 0.39 | 0.17 | 0.39 | 0.341 | 0.184 | 0.075 |
| spectral_entropy | -0.08 | -0.03 | 0.66 | 0.15 | 0.56 | 0.483 | 0.103 | 0.040 |
| harmonic_ratio_v | -0.10 | -0.25 | -0.02 | 0.13 | 0.22 | 1.525 | 0.706 | 0.257 |
| phase_offset | -0.12 | 0.06 | -0.02 | 0.11 | 0.61 | 0.223 | 0.256 | 0.087 |
| clearance_l | -0.12 | -0.23 | 0.35 | 0.11 | 0.26 | 0.340 | 0.183 | 0.060 |
| phase_locking | -0.22 | 0.05 | 0.27 | 0.02 | 0.59 | 0.637 | 0.323 | 0.050 |

## Identity from parameters alone (no learned model)

| Subset | Rank-1 ID | Chance | Verification AUC | EER |
|---|---|---|---|---|
| all | 0.061 | 0.077 | 0.473 | 0.530 |
| F | 0.138 | 0.077 | 0.576 | 0.480 |
| S | 0.081 | 0.077 | 0.499 | 0.515 |

## Gait-event detectability by view

- **F**: median stride-period confidence 0.51, 1.35 heel strikes / s
- **S**: median stride-period confidence 0.59, 1.87 heel strikes / s

## Temporal consistency and alignment (RQ4, model-free, joint angles)

- median correlation between gait cycles within a clip: 0.22
- same-view clip pairs: 1072 (54 genuine)

| Comparison | Genuine-vs-impostor AUC |
|---|---|
| whole clip resampled to 60, L2 | 0.526 |
| mean gait cycle (phase-aligned), L2 | 0.576 |
| whole clip, DTW-aligned | 0.598 |
