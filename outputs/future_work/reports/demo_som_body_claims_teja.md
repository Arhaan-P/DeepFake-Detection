# Gait verification report: `data/pose_cache/mediapipe_lite/Som_S1.npz`

**Claimed identity:** Teja  
**Verdict:** **IDENTITY_MISMATCH**  
**P(authentic), even prior:** 0.047 (raw score 0.001, decision threshold 0.314)  
**P(authentic), evaluation prior:** 0.004 (calibrated where 1 in 13 claims is genuine)  
**Best-matching enrolled gait:** Vedant (0.498)  
**Likely body source:** Vedant (weak evidence: the top match was the true person in 38% of held-out tests)  
**Clip:** 410 frames, 7.7 s, 11 strides detected

## Interpretable gait agreement with the claimed identity

Agreement = exp(-z²/2), z = (query - claimed mean) / within-person SD. ICC is the parameter's test-retest reliability on the enrolment data; treat parameters with ICC < 0.3 as weak evidence.

| Parameter | Query | Claimed mean | z | Agreement | ICC |
|---|---|---|---|---|---|
| Step width | 0.138 | 0.090 | +1.05 | 0.58 | 0.28 (weak) |
| Double-support fraction | 0.322 | 0.307 | +0.22 | 0.98 | 0.23 (weak) |
| Cadence (steps/min) | 108.200 | 102.848 | +0.53 | 0.87 | 0.17 (weak) |
| Right step length | 0.626 | 0.551 | +0.42 | 0.92 | 0.12 (weak) |
| Stride length (leg lengths) | 1.380 | 1.178 | +0.61 | 0.83 | 0.10 (weak) |
| Stride time (s) | 1.109 | 1.168 | -0.50 | 0.88 | 0.08 (weak) |
| Right stance fraction | 0.657 | 0.569 | +0.70 | 0.78 | 0.06 (weak) |
| Left step length | 0.754 | 0.627 | +0.62 | 0.82 | 0.03 (weak) |
| Stride regularity | 0.777 | 0.807 | -0.18 | 0.98 | 0.02 (weak) |
| Step-length asymmetry | 0.186 | 0.211 | -0.05 | 1.00 | -0.00 (weak) |
| Regularity symmetry | 1.010 | 0.994 | +0.08 | 1.00 | -0.01 (weak) |
| Step regularity | 0.784 | 0.804 | -0.10 | 1.00 | -0.03 (weak) |
| Step-time asymmetry | 0.102 | 0.106 | -0.01 | 1.00 | -0.04 (weak) |
| Stride-time variability (CV) | 0.151 | 0.158 | -0.10 | 1.00 | -0.06 (weak) |
| Right foot clearance | 0.305 | 0.355 | -0.27 | 0.96 | -0.06 (weak) |
| Left stance fraction | 0.679 | 0.568 | +0.88 | 0.68 | -0.07 (weak) |
| Harmonic ratio (vertical) | 2.675 | 1.655 | +1.47 | 0.34 | -0.08 (weak) |
| Spectral entropy | 0.335 | 0.498 | -1.58 | 0.29 | -0.10 (weak) |
| Left foot clearance | 0.092 | 0.337 | -1.34 | 0.41 | -0.14 (weak) |
| L/R phase offset | 0.002 | 0.166 | -0.63 | 0.82 | -0.14 (weak) |
| L/R phase locking | 0.925 | 0.634 | +0.90 | 0.66 | -0.22 (weak) |

## Joints driving the decision (gradient x input, max = 1)

- R_Ankle: 1.00
- L_Ankle: 0.77
- R_Foot: 0.76
- R_Knee: 0.73
- R_Heel: 0.73
- L_Heel: 0.68
- R_Shoulder: 0.66
- L_Knee: 0.62
- L_Shoulder: 0.62
- L_Foot: 0.50
- R_Hip: 0.48
- L_Hip: 0.41

## Feature-family share of attribution

- coords: 57.7%
- velocity: 32.8%
- angles: 9.5%

## Where in the walk the mismatch lies

Score change if that window of the query is replaced by the claimed identity's enrolled gait (largest = most responsible):

- 6.52-7.69 s: +0.018
- 5.86-7.04 s: +0.007
- 5.21-6.39 s: +0.006

## Counterfactual: replace one feature family with the claim's

- coords: score +0.012
- velocity: score -0.001
- angles: score +0.000

---
Scope: this verifies consistency of the walking pattern with an enrolled identity. It is not a general detector of AI-generated video, and it cannot detect manipulations that also re-synthesise body motion.
