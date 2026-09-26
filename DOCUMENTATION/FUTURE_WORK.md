# Future Work: Implementation and Results

For the project team and supervisor. This document maps each section of *Future Plan for Gait Analysis* to code on the `research/gait-roadmap` branch, says how to run it, and reports what the experiments found on GaitDeepfake-13. The depth-camera sections (plan §4, experiments E5/E6) are out of scope and were not implemented.

**Guiding rule, as in the plan:** the RGB/MediaPipe system stays the frozen baseline (E0). Every experiment changes one variable and runs through the same protocol, on the same folds and the same trials.

## Findings at a glance

All 47 experiments ran with 13 subject-disjoint folds × 3 seeds on 66 clips (858 verification trials per run: 66 genuine, 792 impostor). Full tables: `outputs/future_work/report/summary.md` and `robustness.md`. Every number below is the pooled ROC-AUC (%), with Δ measured against E0 on identical trials. A difference is called *significant* only when the paired Wilcoxon test over 39 fold × seed pairs gives p < 0.05. With 13 subjects, subject-bootstrap 95% intervals are about ±3–5 points wide, so most single-variable changes cannot be separated from E0.

1. **The baseline holds up under a stricter protocol, but below the paper's number.** E0 reaches **85.2 AUC / 22.4% EER** (per-fold 85.3 ± 9.0), against 94.95 in the paper. The protocol here removes three optimistic choices in the original evaluation: the query clip averaged into its own enrolled signature, test folds normalised with their own statistics, and random rather than exhaustive impostor claims. Treat 85 as the honest reference point.
2. **The model identifies people mostly by body geometry, not motion dynamics.** Coordinates alone reach 85.5, the same as the full 78-D descriptor. Adding angles (85.6) or velocities (85.2) changes nothing measurable. A model-free check agrees: joint-angle trajectories alone separate identities barely above chance (AUC 0.53–0.60). This shapes how the paper should describe "gait": the signal is largely the shape and posture of the walking body.
3. **A better pose estimator gives a cleaner signal, but only a small, non-significant verification gain.** RTMPose, ViTPose and RTMW3D remove 76–91% of left/right leg swaps and cut joint jitter by 35–52% compared with MediaPipe lite. On matched 2D descriptors, RTMPose scores 85.9 against MediaPipe's 84.2 (Δ +2.0 vs E0, p = 0.09). The best single result in the study is RTMW3D's RGB-only 3D skeleton: 86.0 AUC and the lowest EER, 21.3%. MediaPipe's heavy model and its video-tracking mode do not help. Heavy + tracking fails badly on one subject (Aarav, AUC 0.17), because one clip's tracker locked onto a wrong pose (jitter 10–30× that subject's other clips). A pose-quality gate should reject such clips at enrolment.
4. **Rhythm features do not help on clips this short.** Cadence, regularity, harmonics, phase and cycle normalisation each change AUC by −2.0 to +0.9. Cycle normalisation is significantly *worse* on the bootstrap interval. The model-free reliability analysis explains why: with 2–4 strides per clip, even the best clip-level rhythm parameters (side-view cadence and stride time) reach a test-retest ICC of only about 0.4. Rhythm needs longer, multi-cycle recordings. The data-collection protocol now asks for walks of at least 8 s.
5. **Among the new biomechanical families, only jerk gives a significant gain.** Jerk: +2.2, p = 0.025, AUC 86.8, the best in E4. Extended angles (86.3) and stride parameters (85.7, EER 21.3) trend positive. Stacking every family (81.2) or keeping only interpretable ones (75.3) hurts. On 13 subjects, a larger descriptor overfits.
6. **Bigger verifiers are worse, as in the paper's ablation.** BiLSTM 82.4, Transformer 81.9, ST-GCN 83.3, ST-GCN-only 81.1, Siamese metric learning 77.2. The Transformer, ST-GCN-only and Siamese results are significantly below E0. The frequency branch (85.9) is the only neutral addition. Explicit skeleton-graph modelling (RQ9) does not beat treating the descriptor as a time series at this cohort size. The non-learned DTW template (46.8) carries no identity on the raw descriptor.
7. **Robustness: the practical weak points are camera distance, low light, short clips and mirroring.** On E0, a camera about 1.7× farther away costs −10.9, low light −7.8, keeping only 40% of a clip −8.5, a mirrored walk −6.9, and heavy keypoint noise −14.6. JPEG compression, blur, 1/4 resolution, lower frame rates, frame drops and speed changes cost nothing. The baseline is speed-invariant by construction, because it resamples every clip to 60 steps. **Torso-scaled coordinates trade 2.8 clean AUC points for large robustness gains:** a farther camera goes from −10.9 to +0.9, and low light from −7.8 to −2.1. The full engineered descriptor is fragile: derivatives amplify noise, so foot occlusion costs −13.9 and 10 fps capture −12.3.
8. **View matters, but this corpus cannot isolate it.** Side walks verify better than frontal walks in almost every experiment (E0: 88.0 vs 82.1). Frontal gait events are also harder to detect (median stride-period confidence 0.51 vs 0.59). Same-view and cross-view protocols drop to 75.7–77.1, but each view has only 2–3 clips per person. Those signatures average 1–2 clips instead of E0's 4, so part of the drop is thinner enrolment rather than view. More takes per view are needed to separate the two effects.
9. **Scores are well ranked but poorly calibrated, and body-source identification is weak.** Raw P(authentic) has an ECE of about 12%. Platt scaling, cross-fitted so that no subject calibrates itself, brings this to 1–2% in every experiment. Picking the true person out of 13 (rank-1 identification) succeeds only 40% of the time for E0, against 7.7% by chance. Reports should present the "likely body source" of a face-swap as weak evidence. The paper's 3/3 body-source matches relied on the query being part of its own signature.

**Protocol lessons worth keeping.** Two bugs found during this work would each have inflated results, and both now have regression tests. (a) Leave-one-out genuine signatures averaged fewer clips than impostor signatures, so a verifier could "detect" genuine claims by counting clips. All signatures now average the same number of clips. (b) Platt fitting diverged on saturated scores. It now uses a damped, line-searched fit.

**Recommended next steps, in order:** (1) record longer walks (≥ 8 s) and a second session per person, following `DATA_COLLECTION_PROTOCOL.md`, because rhythm and view questions cannot be answered on the current clips; (2) adopt RTMPose or RTMW3D-world as the pose backend, with a pose-quality gate at enrolment; (3) add jerk and torso-scaled coordinates as the next descriptor candidates, re-tested on the larger cohort; (4) keep the small difference-based verifier; (5) evaluate real face-swaps from several generators with `generator_robustness.py`.

### Pose-signal quality (same 66 videos, `outputs/future_work/pose_benchmark/`)

| Backend | Detection | Jitter ↓ | L/R swaps ↓ | Stride recoverable ↑ | Verification AUC |
|---|---|---|---|---|---|
| MediaPipe lite (E0) | 98.5% | 0.043 | 1.48% | 0.55 | 85.2 (78-D) / 84.2 (2D) |
| MediaPipe lite, video tracking | 99.8% | 0.024 | 0.94% | 0.57 | 83.3 |
| MediaPipe heavy, video tracking | 99.8% | 0.020 | 0.54% | 0.63 | 74.9 (one tracking failure) |
| RTMPose-m (Halpe-26) | 100% | 0.024 | 0.13% | 0.64 | 85.9 (2D) |
| ViTPose-B (coco-25) | 100% | 0.021 | 0.36% | 0.64 | 84.8 (2D) |
| RTMW3D (3D) | 100% | 0.028 | 0.21% | 0.64 | 80.0 (image) / **86.0 (3D)** |

Jitter is the RMS of the >6 Hz residual, in torso lengths. "Stride recoverable" is the autocorrelation height of the detected stride period. Per-frame latency was measured under very different machine loads, so it is recorded in the JSON but is not a fair speed comparison.

### Explainability and deployment demos

`train_final.py` built a verifier with two clips held out of training *and* enrolment. On those unseen clips, the forensic report (`outputs/future_work/reports/`) and the streaming verifier gave:

- **Simulated face-swap** (Som's walk, claiming Teja): **IDENTITY_MISMATCH**. P(authentic) is 0.047 at an even prior. The streaming score falls from 0.03 to 0.00. The report names the wrong body source (Vedant) and flags it as weak evidence (38% rank-1 accuracy).
- **Genuine walk** (Teja claiming Teja): **AUTHENTIC**. P(authentic) is 0.81 at an even prior. The streaming score stays at 0.91–0.94.
- **Unknown claim** ("Mallory"): refused with UNKNOWN_IDENTITY.
- The streaming pose stage runs at about 30 ms/frame (about 33 fps) on this laptop's CPU. Only skeletons are kept (privacy mode).

The cross-generator script (E9) runs end to end, but no face-swap clips were available on this machine, so E9 has no results yet.

### Compute notes

Most experiments ran on CPU. The four heaviest verifiers (E7 BiLSTM, Transformer, ST-GCN, ST-GCN-only) ran on an RTX 4050 GPU after the CPU estimate reached ~10 h per run. Every result file records its device under `environment`. GPU and CPU arithmetic are not bit-identical, but the difference is far below the seed-to-seed spread.

## Where each part of the plan lives

| Plan section | What was built | Code |
|---|---|---|
| §2 Build, do not restart | Frozen E0 config; one controlled harness for every experiment | `configs/future_work/experiment_matrix.json`, `utils/verification_harness.py` |
| §3 Pose estimation | Pluggable backends in one 33-landmark layout: MediaPipe lite/full/heavy (image and video-tracking modes, plus metric world-3D), RTMPose-m Halpe26, ViTPose-B coco_25, RTMW3D (3D), OpenPose BODY_25. Full-frame-rate pose cache with per-frame latency. | `utils/pose_backends.py`, `scripts/future_work/extract_pose_cache.py` |
| §3 "Compare pose quality first" | Detection rate, confidence, >6 Hz jitter, acceleration RMS, bone-length stability, L/R swap rate, planted-heel stillness, gait-period recoverability, agreement with a reference backend | `utils/pose_quality.py`, `scripts/future_work/pose_benchmark.py` |
| §5 Feature engineering | Registry of feature families: frozen baseline (coords, angles, velocity), 2D and torso-scaled coordinates, acceleration, jerk, angular velocity, extended and 3D angles, L/R symmetry, foot dynamics, centre-of-mass sway, stride parameters, rhythm, harmonic spectra, gait phase. 19 named feature sets form the ablation hierarchy. | `utils/gait_features.py`, `utils/gait_descriptors.py` |
| §6 Rhythm and temporal analysis | Walking-axis PCA, stride period by autocorrelation, Zeni heel-strike / toe-off events, cadence, stride CV, step asymmetry, step/stride regularity, harmonic ratio, spectral entropy, Hilbert L/R phase synchrony, cycle normalisation, DTW | `utils/gait_cycle.py`, `scripts/future_work/rhythm_analysis.py` |
| §7 Models | Deployed temporal CNN (bit-identical to `models/full_pipeline.py`), raw+BiLSTM/Transformer/CNN/hybrid, ST-GCN skeleton-graph encoder, frequency branch, Siamese metric learning, non-learned DTW template | `models/verifier_variants.py`, `models/graph_encoder.py` |
| §8 Data expansion | Recording protocol, manifest schema and validator; `cross:<field>:<A>:<B>` protocols for any recorded condition | `DOCUMENTATION/DATA_COLLECTION_PROTOCOL.md`, `scripts/future_work/validate_recordings.py` |
| §8 Robustness | Keypoint perturbations (jitter, foot occlusion, frame rate, truncation, frame drop, distance, rotation, speed, mirror); video degradation re-run through pose estimation (JPEG q15, 1/4 resolution, low light, blur); cross-view protocols | `utils/keypoint_augment.py`, `extract_pose_cache.py --degrade` |
| §8 / E9 Generators | Per-generator and per-codec rejection rate and body-source identification from a clip manifest | `scripts/future_work/generator_robustness.py` |
| §9 Evaluation | Subject-disjoint LOSO, 3 seeds, per-fold mean ± sd and pooled scores, EER / Youden / TPR@FPR, ECE / Brier with cross-fitted Platt calibration, subject-bootstrap CIs, paired Wilcoxon / t-tests, failure analysis | `utils/verification_metrics.py`, `scripts/future_work/compare_results.py` |
| §10 Explainability | Interpretable sub-scores with reliability (ICC), joint and family attribution, temporal localisation of the mismatch, counterfactual family swap | `utils/gait_verifier.py`, `scripts/future_work/forensic_report.py` |
| §11 Deployment | Deployable bundle (threshold and calibration from held-out scores), rolling real-time verifier, skeleton-only privacy mode, unknown-identity refusal, quality-weighted multi-camera fusion | `scripts/future_work/train_final.py`, `scripts/future_work/realtime_verifier.py` |

## The controlled protocol, and how it differs from the paper's LOOCV

`utils/verification_harness.py` applies these rules to every experiment:

1. **Leave-one-subject-out folds (13).** Every clip of the held-out person is excluded from training.
2. **Normalisation from training subjects only.** The committed `loocv_results.json` normalised each test fold with its own statistics; `scripts/evaluation/ablation_loocv.py` already flags this.
3. **Strict enrolment.** A query clip is never part of the signature it is compared with. In the original pipeline, the held-out subject's signature averages all of their videos, *including the query*. `enrol="legacy"` reproduces that, so E0_legacy_enrol measures the effect.
4. **Equal-size signatures.** Every evaluation signature, genuine or impostor, averages the same number of clips (k = 4 here), and training signatures average a random 1–k clips, so the number of averaged clips carries no label information.
5. **Impostor claims during training come from training identities only.** The original data loader could also draw the held-out subject's signature as a training negative.
6. **Exhaustive, deterministic test trials.** Every original clip of the held-out subject is scored against all 13 enrolled identities: 66 genuine and 792 impostor trials. The original drew one random impostor per sample. Because every experiment scores identical trials, differences between experiments can be tested pairwise.
7. **Test clips are originals only.** Augmented copies are used for training only.
8. **Keypoint-level augmentation** (`rhythm_safe` preset) replaces the 16× video augmentation, so one augmentation policy applies equally to every pose backend. The preset leaves out speed changes and mirroring, because training for invariance to cadence would erase the rhythm features being tested. `E0_paper_aug` measures what the paper-style preset changes.

These rules are stricter than the paper's evaluation, so **E0's numbers are expected to be lower than the paper's 94.95% AUC**. Every experiment below is compared against this E0, never against the paper's figure.

## How to reproduce

PowerShell, from the repository root, in a fresh environment (see `requirements-future-work.txt` for the MediaPipe/protobuf note):

```powershell
python -m venv .venv; .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt -r requirements-future-work.txt

# 0. unpack the 66 walking videos to data\videos (e.g. from 13_People_Dataset.zip)

# 1. full-frame-rate pose caches (one per backend / degradation)
python scripts/future_work/extract_pose_cache.py --backend mediapipe_lite
python scripts/future_work/extract_pose_cache.py --backend mediapipe_heavy_video
python scripts/future_work/extract_pose_cache.py --backend rtmpose_balanced --workers 6
python scripts/future_work/extract_pose_cache.py --backend mediapipe_lite --degrade jpeg:15
#    (all backends and degradations used: see the experiment matrix)

# 2. pose-signal quality benchmark and model-free rhythm analysis
python scripts/future_work/pose_benchmark.py
python scripts/future_work/rhythm_analysis.py

# 3. the experiment matrix (about 45 min per experiment per core pair on CPU)
python scripts/future_work/run_experiment.py --all --parallel 4 --skip_done
#    with an NVIDIA GPU (CUDA build of torch), add:  --set device=cuda
#    (10-40x faster for the Transformer / ST-GCN verifiers)
python scripts/future_work/compare_results.py

# 4. a deployable verifier, a forensic report, a streaming run
python scripts/future_work/train_final.py --experiment E0_baseline
python scripts/future_work/forensic_report.py --checkpoint outputs/future_work/checkpoints/E0_baseline.pt --video path\to\clip.mp4 --claimed Teja
python scripts/future_work/realtime_verifier.py --checkpoint outputs/future_work/checkpoints/E0_baseline.pt --claimed Teja --source 0

# 5. face-swap clips from several generators (E9)
python scripts/future_work/generator_robustness.py --checkpoint outputs/future_work/checkpoints/E0_baseline.pt --manifest data\deepfake\manifest.csv --compress 50 20

# tests (numpy + torch only)
python tests/test_future_work.py
```

Ad-hoc experiments reuse the same protocol:

```powershell
python scripts/future_work/run_experiment.py --name my_idea --set feature_set=b+foot model=raw+stgcn
python scripts/future_work/run_experiment.py --name cross_session --set metadata_csv=recordings.csv protocol=cross:session:1:2
```
