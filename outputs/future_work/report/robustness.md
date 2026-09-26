# Robustness (E8): clean enrolment, degraded query

AUC / EER in %, mean over seeds. Keypoint perturbations are applied to the extracted pose stream; `video:` rows re-ran pose estimation on degraded copies of the videos.


## E0_baseline

| Condition | AUC | ΔAUC vs clean | EER |
|---|---|---|---|
| clean | 85.24 | – | 22.35 |
| noise:0.005 | 84.73 | -0.51 | 22.14 |
| noise:0.01 | 81.43 | -3.81 | 24.37 |
| noise:0.02 | 70.68 | -14.56 | 36.36 |
| occlude_feet:0.2 | 83.04 | -2.20 | 24.20 |
| occlude_feet:0.4 | 82.28 | -2.95 | 24.24 |
| fps:15 | 85.93 | +0.69 | 22.77 |
| fps:10 | 86.19 | +0.95 | 21.65 |
| truncate:0.66 | 80.55 | -4.69 | 26.79 |
| truncate:0.4 | 76.69 | -8.55 | 29.82 |
| drop:0.3 | 85.33 | +0.09 | 23.65 |
| scale:0.6 | 74.33 | -10.91 | 32.11 |
| scale:1.3 | 79.92 | -5.32 | 27.31 |
| rotate:8 | 85.17 | -0.07 | 23.32 |
| speed:0.85 | 85.24 | +0.00 | 22.35 |
| speed:1.15 | 85.24 | +0.00 | 22.35 |
| mirror | 78.35 | -6.89 | 29.34 |
| video:mediapipe_lite__jpeg15 | 85.38 | +0.14 | 22.85 |
| video:mediapipe_lite__scale0.25 | 85.83 | +0.59 | 23.23 |
| video:mediapipe_lite__gamma2.5 | 77.47 | -7.77 | 29.78 |
| video:mediapipe_lite__blur15 | 85.53 | +0.29 | 22.77 |

## E8_full_robustness

| Condition | AUC | ΔAUC vs clean | EER |
|---|---|---|---|
| clean | 81.20 | – | 25.23 |
| noise:0.01 | 73.40 | -7.81 | 32.22 |
| occlude_feet:0.4 | 67.31 | -13.90 | 37.42 |
| fps:10 | 68.90 | -12.30 | 32.39 |
| truncate:0.4 | 67.90 | -13.31 | 36.68 |
| scale:0.6 | 64.34 | -16.87 | 40.21 |
| scale:1.3 | 75.88 | -5.33 | 29.86 |
| speed:1.15 | 79.30 | -1.90 | 26.26 |
| video:mediapipe_lite__jpeg15 | 81.52 | +0.31 | 25.32 |
| video:mediapipe_lite__scale0.25 | 82.14 | +0.94 | 23.74 |
| video:mediapipe_lite__gamma2.5 | 74.42 | -6.78 | 32.01 |
| video:mediapipe_lite__blur15 | 82.43 | +1.22 | 24.24 |

## E8_scaled_robustness

| Condition | AUC | ΔAUC vs clean | EER |
|---|---|---|---|
| clean | 82.39 | – | 25.76 |
| noise:0.01 | 79.48 | -2.90 | 28.81 |
| occlude_feet:0.4 | 80.50 | -1.89 | 27.71 |
| fps:10 | 84.00 | +1.62 | 23.32 |
| truncate:0.4 | 72.24 | -10.15 | 33.00 |
| scale:0.6 | 83.28 | +0.89 | 24.52 |
| scale:1.3 | 80.44 | -1.94 | 27.29 |
| speed:1.15 | 82.39 | +0.00 | 25.76 |
| video:mediapipe_lite__jpeg15 | 82.76 | +0.37 | 26.22 |
| video:mediapipe_lite__scale0.25 | 83.31 | +0.93 | 24.75 |
| video:mediapipe_lite__gamma2.5 | 80.24 | -2.14 | 27.15 |
| video:mediapipe_lite__blur15 | 84.04 | +1.65 | 22.22 |
