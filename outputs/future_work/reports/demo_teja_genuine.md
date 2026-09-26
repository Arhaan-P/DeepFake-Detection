# Gait verification report: `data/pose_cache/mediapipe_lite/Teja_S1.npz`

**Claimed identity:** Teja  
**Verdict:** **AUTHENTIC**  
**P(authentic), even prior:** 0.813 (raw score 0.736, decision threshold 0.314)  
**P(authentic), evaluation prior:** 0.266 (calibrated where 1 in 13 claims is genuine)  
**Best-matching enrolled gait:** Teja (0.736)  
**Clip:** 304 frames, 6.5 s, 10 strides detected

## Interpretable gait agreement with the claimed identity

Agreement = exp(-z²/2), z = (query - claimed mean) / within-person SD. ICC is the parameter's test-retest reliability on the enrolment data; treat parameters with ICC < 0.3 as weak evidence.

| Parameter | Query | Claimed mean | z | Agreement | ICC |
|---|---|---|---|---|---|
| Step width | 0.119 | 0.090 | +0.63 | 0.82 | 0.28 (weak) |
| Double-support fraction | 0.340 | 0.307 | +0.48 | 0.89 | 0.23 (weak) |
| Cadence (steps/min) | 105.924 | 102.848 | +0.31 | 0.95 | 0.17 (weak) |
| Right step length | 0.743 | 0.551 | +1.07 | 0.56 | 0.12 (weak) |
| Stride length (leg lengths) | 1.466 | 1.178 | +0.88 | 0.68 | 0.10 (weak) |
| Stride time (s) | 1.133 | 1.168 | -0.30 | 0.96 | 0.08 (weak) |
| Right stance fraction | 0.643 | 0.569 | +0.59 | 0.84 | 0.06 (weak) |
| Left step length | 0.723 | 0.627 | +0.47 | 0.90 | 0.03 (weak) |
| Stride regularity | 0.543 | 0.807 | -1.59 | 0.28 | 0.02 (weak) |
| Step-length asymmetry | 0.028 | 0.211 | -0.38 | 0.93 | -0.00 (weak) |
| Regularity symmetry | 1.378 | 0.994 | +1.97 | 0.14 | -0.01 (weak) |
| Step regularity | 0.748 | 0.804 | -0.28 | 0.96 | -0.03 (weak) |
| Step-time asymmetry | 0.018 | 0.106 | -0.26 | 0.97 | -0.04 (weak) |
| Stride-time variability (CV) | 0.146 | 0.158 | -0.17 | 0.99 | -0.06 (weak) |
| Right foot clearance | 0.123 | 0.355 | -1.26 | 0.45 | -0.06 (weak) |
| Left stance fraction | 0.592 | 0.568 | +0.19 | 0.98 | -0.07 (weak) |
| Harmonic ratio (vertical) | 1.929 | 1.655 | +0.40 | 0.92 | -0.08 (weak) |
| Spectral entropy | 0.408 | 0.498 | -0.88 | 0.68 | -0.10 (weak) |
| Left foot clearance | 0.243 | 0.337 | -0.51 | 0.88 | -0.14 (weak) |
| L/R phase offset | 0.040 | 0.166 | -0.49 | 0.89 | -0.14 (weak) |
| L/R phase locking | 0.986 | 0.634 | +1.09 | 0.55 | -0.22 (weak) |

## Joints driving the decision (gradient x input, max = 1)

- R_Ankle: 1.00
- L_Ankle: 0.95
- R_Heel: 0.94
- R_Foot: 0.93
- L_Heel: 0.91
- R_Shoulder: 0.89
- R_Knee: 0.86
- L_Shoulder: 0.85
- L_Knee: 0.79
- L_Foot: 0.78
- R_Hip: 0.69
- L_Hip: 0.64

## Feature-family share of attribution

- coords: 59.6%
- velocity: 32.2%
- angles: 8.2%

## Where in the walk the mismatch lies

Score change if that window of the query is replaced by the claimed identity's enrolled gait (largest = most responsible):

- 3.32-4.32 s: +0.043
- 2.77-3.77 s: +0.021
- 2.22-3.21 s: -0.016

## Counterfactual: replace one feature family with the claim's

- velocity: score -0.670
- coords: score -0.518
- angles: score +0.134

---
Scope: this verifies consistency of the walking pattern with an enrolled identity. It is not a general detector of AI-generated video, and it cannot detect manipulations that also re-synthesise body motion.
