# Project Context — Portfolio Summary

## 1. Project Name
**Gait-Based Deepfake Detection** (repo name: `DeepFake-Detection`)

## 2. One-Liner
A deepfake video detector that verifies identity by analyzing how a person *walks* (skeletal gait patterns) instead of their face, catching face-swap deepfakes that fool facial-recognition-based detectors.

## 3. Problem / Motivation
Most deepfake detectors analyze facial artifacts, which face-swap tools are increasingly good at faking convincingly. Gait — the biomechanical pattern of how someone walks — is much harder to forge because face-swap tools only replace the face, not body motion. This project builds a detector that flags a video as authentic, an identity mismatch, or a suspected deepfake by comparing extracted gait signatures against an enrolled "true" walking pattern for a claimed identity. It was developed as a research project (with an accompanying IEEE-style paper, abstract, methodology write-up, and a literature review of 70+ references) alongside being a full engineering pipeline.

## 4. My Role
**Solo project.** All 13 commits are from a single author (Arhaan Penwala, under two git name variants, same email). Designed, implemented, and evaluated the entire pipeline — data collection/augmentation, model architecture, training, evaluation (LOOCV + ablation study), and documentation/paper drafting.

## 5. Tech Stack
- **Language:** Python 3.9+ (100% Python; no JS/web layer)
- **Deep Learning:** PyTorch (`torch`, `torch.nn`, `torch.optim`, TensorBoard logging), CUDA-accelerated (developed on RTX 3050)
- **Computer Vision / Pose Estimation:** MediaPipe (Pose Landmarker Lite), OpenCV, Albumentations (video augmentation)
- **ML / Evaluation:** scikit-learn (ROC-AUC, EER, confusion matrix, LOOCV)
- **Data / Visualization:** NumPy, pandas, matplotlib, seaborn, tqdm
- **Database:** None
- **Cloud/Infra:** None — no Docker, no CI/CD, no cloud deployment. Purely a local research pipeline with PowerShell/bash setup scripts for environment provisioning.
- **Packaging:** `pyproject.toml` (setuptools, editable install of `models`/`utils` as a package)

## 6. Key Features
- Custom **gait feature extraction pipeline**: MediaPipe pose → 12 gait keypoints → 78-dim feature vector per frame (normalized joint coordinates + joint angles + frame-to-frame velocities), sequences normalized to 60 frames.
- **Hybrid neural architecture**: 1D CNN encoder with residual blocks (`GaitEncoder`) feeding a dual-path BiLSTM + Transformer temporal model, plus a difference-based CNN classifier (video vs. claimed-identity gait comparison via diff/abs-diff/product features) that resolved an embedding-collapse issue from an earlier Siamese-network approach.
- **Three-way inference verdict system**: AUTHENTIC / IDENTITY MISMATCH / SUSPECTED DEEPFAKE, with a decision threshold (0.7737) derived empirically from LOOCV Youden's J statistic rather than hardcoded arbitrarily.
- **Rigorous evaluation methodology**: Leave-one-out cross-validation across 13 subjects, plus a 4-way ablation study (CNN-only, LSTM-only, Transformer-only, full hybrid) with full metrics and parameter counts for each variant.
- **Explainability**: Grad-CAM-style analysis (`GaitGradCAM`, `JointImportanceAnalyzer`) identifying which joints/timesteps drive classification decisions (e.g., shoulder and heel movement contribute most).
- **Reproducible dataset release tooling**: cross-platform setup scripts (PowerShell + bash) and a quickstart verification script, built for distributing the curated "GaitDeepfake-13" dataset via IEEE DataPort.

## 7. Architecture Notes
CLI-driven offline pipeline: video file → MediaPipe pose extraction → 78-dim gait feature sequence → CNN encoder → dual-path BiLSTM/Transformer temporal model → difference-based CNN comparator against an enrolled identity's stored gait signature → softmax classification → terminal-based verdict report (with optional saved visualization PNG). No client-server architecture, no persistent database — enrolled identities and extracted features are cached as pickle files (`gait_features.pkl`, `enrolled_identities.pkl`).

## 8. Technical Challenges Solved
- **Embedding collapse**: an initial Siamese/triplet-loss identity-verification approach collapsed embeddings during training; resolved by switching the primary classifier to a difference-feature CNN (diff, abs-diff, product of video vs. claimed gait sequences).
- **Threshold selection**: avoided a hardcoded/guessed classification threshold by deriving it from Youden's J statistic over LOOCV results.
- **Small-dataset evaluation**: used leave-one-out cross-validation (13-fold, one per subject) rather than a simple train/test split, appropriate for a small, self-collected identity dataset.
- **Data scarcity mitigation**: built a video augmentation pipeline generating 16x augmented variants (blur, flip, rotation, speed changes, noise, zoom, grayscale, combined transforms) from 66 original recordings, yielding 1,056 training videos.
- **Model interpretability**: implemented Grad-CAM adapted for temporal skeletal/gait input to explain which joints and frames drive predictions, rather than treating the model as a black box.
- **Architecture/documentation discrepancy** (self-flagged): the deployed checkpoints use smaller dimensions (`encoder_output_dim=128`, etc.) than the larger architecture described in the README — worth reconciling docs before presenting.

## 9. Metrics / Impact
Backed by real JSON output files, not just narrative claims:
- **AUC-ROC:** 94.95% ± 2.81% (LOOCV, pooled), 95.10% (aggregate)
- **Accuracy:** 87.27% ± 3.76% (per-fold mean), 87.04% (pooled)
- **F1 Score:** 86.56% ± 4.56% (per-fold), 87.12% (pooled)
- **Equal Error Rate (EER):** 13.19% ± 4.21% (per-fold), 12.77% (pooled)
- **Decision threshold:** 0.7737 (Youden's J-optimal)
- **Ablation study** (accuracy): CNN-Only 88.93%, LSTM-Only 89.33%, Transformer-Only 90.51%, Full Hybrid 90.32% (best recall: 90.91%)
- **Confusion matrix (pooled, 2,240 pairs):** TP 987, TN 962, FP 158, FN 133
- **Evaluation runtime:** full 13-fold LOOCV completed in ~20 minutes (1,228s)
- Dataset: 13 enrolled identities, 66 original videos → 1,056 augmented training videos, plus 4 real face-swapped deepfake test videos generated via FaceFusion/InsightFace.

## 10. Live Demo / Repo Link
[ADD LINK]

## 11. Screenshots Needed
**No** — there is no UI/web interface (CLI-only tool with terminal output). Use `screenshots/hero.png` as the single card/thumbnail image — an abstract joint-importance skeleton diagram (gradient-attribution heatmap over body joints) that conveys the core "detect deepfakes via body/gait analysis, not faces" concept at a glance, without showing an identifiable person.

## 12. Suggested Portfolio Tier
**Featured.** This goes well beyond a typical ML side project — it includes a novel problem framing (gait over face), a custom hybrid architecture with an ablation study proving each component's contribution, rigorous LOOCV evaluation with real quantified results, explainability analysis, and research-paper-grade documentation (abstract, methodology, literature review, dataset release prep for IEEE DataPort). Ideal as a detailed case study.
