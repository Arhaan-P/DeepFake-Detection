# Gait-Based Deepfake Detection

Detects face-swap deepfakes by verifying identity from **skeletal gait** instead of facial features. Face-swap generators composite a synthesised face through a mask confined to the facial region — the body underneath still walks the way the source person walks. Given a video and a claimed identity, the system extracts a 78-dimensional per-frame gait descriptor via MediaPipe and classifies the discrepancy between the observed sequence and the claimed identity's enrolled signature, using a difference-based temporal convolutional verification network.

Full method, evaluation, and limitations are in the accompanying paper (`paper.tex`); this README covers running the code. `NOTES.md` and `AUDIT_FINDINGS.md` document known issues and design decisions in detail.

## Key Idea

Behavioural biometrics have been applied to deepfake detection since 2019, and whole-body pose/motion cues since 2024–2025 (see `references.bib` for the related-work citations). What's under-explored is using **locomotion gait-cycle skeletal dynamics as the primary discriminative feature** for verifying a claimed identity in a face-swapped video, under a subject-disjoint evaluation protocol. This is a niche rather than a vacant space — see the paper's Related Work section for how it's positioned against gait-recognition and face-forgery-detection literature.

## Results

Numbers below are pooled/per-fold statistics from `outputs/evaluation/loocv/loocv_results.json`, reproducible via `python scripts/generate_figures/generate_all.py`. Full derivation in the paper (Section V) and `NOTES.md §1.1`.

### LOOCV Evaluation (13-fold, subject-disjoint, 2,240 verification pairs)

| Metric    | Per-fold (mean ± std) | Pooled     |
| --------- | ---------------------- | ---------- |
| ROC-AUC   | 95.10% ± 3.08%          | **94.95%** |
| Accuracy  | 87.04% ± 3.65%          | 87.01%     |
| F1 Score  | 87.12% ± 3.77%          | 87.15%     |
| Precision | 86.51% ± 4.72%          | 86.20%     |
| Recall    | 88.23% ± 6.82%          | 88.12%     |
| EER       | 12.27% ± 3.80%          | 12.77%     |

Youden-optimal threshold (pooled ROC): **τ\* = 0.7737** (TPR 83.57%, FPR 8.48%, J = 0.7509). Per-fold accuracy/F1/precision/recall above are computed at the network's native τ = 0.5 argmax — the two operating points shouldn't be quoted interchangeably (paper Section VI-C).

### Architectural Ablation

The model also contains an auxiliary CNN+BiLSTM+Transformer embedding branch (see [Architecture](#architecture)) that is *not* on the decision path. A paired 13-fold × 3-seed LOOCV ablation (n=39 observations per variant) tests whether it should be — every configuration that wires it onto the decision path is significantly worse than the deployed raw-difference network:

| Variant             | Params  | ROC-AUC (%)      | Δ vs. deployed | p (paired) |
| -------------------- | ------- | ----------------- | -------------- | ---------- |
| **Raw (deployed)**   | 133,058 | **94.25 ± 3.09**  | —              | —          |
| Raw + CNN            | 434,690 | 91.72 ± 4.13       | −2.53          | 2.9e-04    |
| Raw + Transformer    | 712,002 | 90.37 ± 5.13       | −3.88          | 3.5e-05    |
| Raw + Hybrid         | 980,994 | 90.03 ± 6.11       | −4.22          | 3.7e-04    |
| Raw + BiLSTM         | 379,074 | 88.32 ± 6.08       | −5.93          | 1.7e-07    |
| Hybrid only (no raw) | 876,162 | 50.70 ± 20.81      | −43.55         | 1.1e-15    |

Reproduce with `python scripts/evaluation/ablation_loocv.py`. Full protocol and interpretation in the paper (Section VI-F) and `NOTES.md §2.2`. `scripts/evaluation/ablation_study.py` is an earlier, superseded version of this experiment, kept for provenance — its numbers should not be used.

### Face-Swap Video Validation

3 FaceFusion/InSwapper face-swap clips (one enrolled subject's face composited onto another's walking footage), scored against the claimed (face) identity at τ\* = 0.7737:

| Clip | Similarity to claimed identity | Verdict            | Matched true body source | Similarity to match |
| ---- | ------------------------------- | ------------------- | -------------------------- | --------------------- |
| 1    | 0.0057                          | IDENTITY_MISMATCH   | ✓                           | 0.9999                |
| 2    | 0.0003                          | IDENTITY_MISMATCH   | ✓                           | 0.9999                |
| 3    | <0.0001                         | IDENTITY_MISMATCH   | ✓                           | 0.99999               |

3/3 correctly rejected the claimed identity and matched the true body source. n=3 from a single generator under one masking configuration — an existence check consistent with the core hypothesis, not a statistically powered claim (paper Section VI-G, `NOTES.md §4`).

### Explainability

Gradient-times-input attribution over the 12 gait keypoints (26 samples, 2 per identity). Top joints: left shoulder (1.000), right heel (0.940), left foot index (0.931), left knee (0.897). By feature group, against dimensional share: coordinates 47.68% (1.03× their 46.15% dimensional share), velocities 37.40% (0.81×), joint angles 14.92% (**1.94×** their 7.69% share — the most information-dense block per dimension, despite being the smallest). Full ranking in the paper (Section VII) and figures 10–11.

## Architecture

The verification decision and the auxiliary embedding branch are two separate lanes that do not join — see `AUDIT_FINDINGS.md #1` and paper Section III-F for why this is a reviewed design decision, not an oversight (the ablation above shows connecting them costs accuracy).

```
                Video features V, claimed signature C
                       both: (batch, T=60, 78)
                              │
        ┌─────────────────────┴─────────────────────┐
        │                                             │
  VERIFICATION PATH (trained,                AUXILIARY BRANCH (not on the
  produces the AUTHENTIC/DEEPFAKE verdict)   decision path — enrolment/GradCAM only)
  ─────────────────────────────────          ──────────────────────────────────
  combined = [V-C ‖ |V-C| ‖ V⊙C]             GaitEncoder: 1D CNN + residual
    (batch, T, 234)                          blocks, 78 → 64 → 128, on V alone
        │                                             │
  Temporal CNN (k=7,5,3; 234→64→64→32)       DualPathTemporalModel:
        │                                     BiLSTM(h=64) ‖ Transformer(d=128,
  AdaptiveAvgPool1d → MLP(32→32→2)            h=4, L=2) → fused 128-dim embedding
        │                                             │
  softmax → P(AUTHENTIC)                     (IdentityVerifier + standalone
        │                                     classifier also live here)
  AUTHENTIC / DEEPFAKE
  ─────────────────────                       ──────────────────────────────
  133,058 params (15.7%)                       715,556 params (84.3%)
  on the decision path                         not on the decision path
```

- **Training**: PyTorch on CUDA (RTX 3050)
- **Feature extraction**: MediaPipe Pose Landmarker (CPU), `pose_landmarker_lite`. `utils/pose_extraction_gpu.py` (TensorFlow MoveNet) exists for reference only and is not used for any reported result.

## Project Structure

```
├── scripts/
│   ├── run_pipeline.py              # End-to-end pipeline orchestrator
│   ├── preprocessing/
│   │   ├── augment_videos.py        # 16x data augmentation
│   │   ├── preprocess_videos.py     # MediaPipe feature extraction
│   │   └── extract_faces.py         # Source-face extraction for FaceFusion
│   ├── training/
│   │   └── train.py                 # Model training with balanced sampling
│   ├── evaluation/
│   │   ├── evaluate.py              # Single-checkpoint + LOOCV evaluation
│   │   ├── ablation_loocv.py        # Definitive paired LOOCV ablation
│   │   ├── ablation_study.py        # Superseded ablation (kept for provenance)
│   │   ├── run_gradcam.py           # Explainability (gradient-times-input + Grad-CAM)
│   │   ├── visualize_keypoints.py   # Publication figure generation
│   │   └── verify_gait_preservation.py  # Gait-preservation check on face-swap clips
│   ├── generate_figures/            # Paper figures 5-11, generated from outputs/*.json
│   ├── inference/
│   │   └── inference.py             # Single video prediction
│   └── enrollment/
│       └── enroll_identities.py     # Build identity gait signatures
├── models/
│   ├── gait_encoder.py              # 1D CNN spatial encoder (auxiliary branch)
│   ├── temporal_model.py            # BiLSTM + Transformer dual path (auxiliary branch)
│   ├── identity_verifier.py         # Siamese comparator (auxiliary branch)
│   └── full_pipeline.py             # Model assembly; the trained verification head
├── utils/
│   ├── pose_extraction.py           # MediaPipe 78-dim feature extractor (used)
│   ├── pose_extraction_gpu.py       # MoveNet extractor (reference only, unused)
│   ├── data_loader.py               # Dataset with balanced pair sampling
│   ├── gradcam.py                   # Grad-CAM + gradient-times-input attribution
│   ├── visualization.py             # Plotting utilities
│   └── logger.py                    # Logging utility
├── diagrams/                        # Conceptual figures 1-4 (Gemini-generated)
├── figures/                         # Figures 5-11 (generated from evaluation output)
├── tests/
│   └── smoke_test.py                # End-to-end pipeline smoke test (no data/GPU needed)
├── ieee_scripts/                    # Setup/verification scripts for the IEEE DataPort release
├── data/
│   ├── videos/                      # Original walking videos
│   ├── augmented_videos/            # Augmented training videos
│   └── gait_features/               # Extracted features (.pkl)
├── outputs/
│   ├── checkpoints/                 # Model checkpoints
│   ├── evaluation/                  # LOOCV results
│   ├── ablation/                    # Ablation study results
│   └── gradcam/                     # GradCAM / attribution visualizations
├── NOTES.md                         # Paper-build notes, discrepancy log, decisions
├── AUDIT_FINDINGS.md                # Known issues logged, not silently fixed
├── LICENSE
├── CONTRIBUTING.md
├── pyproject.toml                   # Package config (pip install -e .)
└── requirements.txt                 # Python dependencies
```

## Setup

### Requirements

- Python 3.9+
- NVIDIA GPU with CUDA (for training)
- ~6 GB GPU memory (RTX 3050 or equivalent)

### Installation

```bash
git clone https://github.com/Arhaan-P/DeepFake-Detection.git
cd DeepFake-Detection
pip install -e .
pip install -r requirements.txt
```

The editable install (`pip install -e .`) makes the `models` and `utils` packages importable from any script location. See `CONTRIBUTING.md` for the full development setup, including the smoke test and lint tooling.

### Data Preparation

1. Place walking videos in `data/videos/` named as `{Name}_{View}{Number}.mp4` (e.g., `SubjectA_F1.mp4`, `SubjectB_S2.mp4`)
2. Place deepfake videos in `data/deepfake/`

## Usage

### Full Pipeline

```bash
python scripts/run_pipeline.py --videos_dir data/videos --augmented_dir data/augmented_videos
```

### Step-by-Step

```bash
# 1. Augment videos (16x)
python scripts/preprocessing/augment_videos.py --input_dir data/videos --output_dir data/augmented_videos

# 2. Extract gait features (MediaPipe, 78-dim)
python scripts/preprocessing/preprocess_videos.py --videos_dir data/videos --augmented_dir data/augmented_videos --output data/gait_features/gait_features.pkl

# 3. Enroll identities (--from_features required, otherwise this defaults to re-processing raw videos)
python scripts/enrollment/enroll_identities.py --from_features --features_file data/gait_features/gait_features.pkl

# 4. Train model
python scripts/training/train.py --features_file data/gait_features/gait_features.pkl --epochs 50

# 5. Evaluate (single checkpoint)
python scripts/evaluation/evaluate.py --checkpoint outputs/checkpoints/checkpoint_epoch_best.pth --save_plots --save_results

# 6. Run LOOCV (13-fold cross-validation)
python scripts/evaluation/evaluate.py --loocv --loocv_epochs 30

# 7. Run the definitive ablation (paired LOOCV, see Results above)
python scripts/evaluation/ablation_loocv.py

# 8. Run explainability analysis
python scripts/evaluation/run_gradcam.py

# 9. Inference on a single video
python scripts/inference/inference.py --video path/to/video.mp4 --claimed_identity "PersonName"

# 10. Generate deepfakes using FaceFusion
# First, activate FaceFusion environment and run GUI (manual process)
cd ../facefusion
conda activate facefusion
python facefusion.py run --ui-layouts default
# Then configure: inswapper_128_fp16 model, strict memory, face-only mask
# Generate face-swaps: save as data/deepfake/{BodyPerson}_body_{FacePerson}_face.mp4

# 11. Verify gait preservation in face-swapped videos -- takes no arguments,
# auto-discovers all data/deepfake/*_body_*_face.mp4 pairs
python scripts/evaluation/verify_gait_preservation.py

# 12. Run inference on deepfake video to detect it
python scripts/inference/inference.py --video data/deepfake/{BodyPerson}_body_{FacePerson}_face.mp4 --claimed_identity "{FacePerson}"
```

### Inference Output

```
AUTHENTIC — Verified as SubjectA    (similarity: 0.87, confidence: 0.92)
DEEPFAKE OF SubjectA                (similarity: 0.23, confidence: 0.95)
```

## Evaluation Metrics

| Metric           | Description                           |
| ---------------- | -------------------------------------- |
| AUC-ROC          | Area under ROC curve                  |
| EER              | Equal Error Rate (FAR = FRR)          |
| F1 Score         | Harmonic mean of precision and recall |
| Precision        | True positives / predicted positives  |
| Recall           | True positives / actual positives     |
| Confusion Matrix | TP, FP, TN, FN breakdown              |

### Cross-Validation

13-fold leave-one-subject-out cross-validation: every recording of the held-out subject (original and all augmentations) is withheld from training, and normalization statistics are recomputed from the remaining 12 subjects. Reports both per-fold (mean ± std) and pooled statistics — see [Results](#results) for why the two differ.

## Feature Extraction

MediaPipe Pose extracts 33 3D landmarks per frame; 12 gait-relevant landmarks are retained. The gait feature vector (78-dim) comprises:

| Component               | Dimensions | Description                              |
| ------------------------ | ---------- | ------------------------------------------ |
| Normalized coordinates   | 36 (12×3)  | Hip-centered x,y,z of 12 gait keypoints    |
| Joint angles             | 6          | Knee, hip, ankle flexion angles             |
| Velocities               | 36 (12×3)  | Frame-to-frame coordinate deltas            |

## Citation

If you use this work, please cite the paper and the dataset:

```bibtex
@misc{penwala2026gaitdeepfake,
  title  = {Detecting Face-Swap Deepfakes by Verifying Skeletal Gait Dynamics},
  author = {Penwala, Arhaan and Arun Raja K, Abhishek and Bharti, Aditya and Sharma, Vibhav},
  year   = {2026},
  note   = {Under review}
}

@misc{gaitdeepfake13,
  author    = {Penwala, Arhaan and Kodeeswaran, Abhishek Arun Raja and
               Bharti, Aditya and Johnson, Deepika Roselind},
  title     = {Deepfake Detection Using Gait Analysis ({GaitDeepfake-13})},
  year      = {2026},
  doi       = {10.21227/ngh5-b637},
  note      = {IEEE DataPort}
}
```

## License

MIT — see [LICENSE](LICENSE).
