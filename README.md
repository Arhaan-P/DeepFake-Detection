# Gait-Based Deepfake Detection

A novel deepfake detection system that analyzes **walking patterns (gait)** instead of facial features. Given a video and a claimed identity, the model extracts skeletal pose keypoints via MediaPipe, compares gait embeddings using a CNN+BiLSTM+Transformer hybrid, and determines whether the person is authentic or a deepfake.

## Key Innovation

No existing published work directly uses gait analysis for deepfake detection (see [LITERATURE_REVIEW.md](LITERATURE_REVIEW.md)). Current deepfake generators focus on face/voice synthesis and do not model individualized biomechanics — gait patterns remain a reliable identity signal that deepfakes cannot replicate.

## Results

### LOOCV Evaluation (13-fold, subject-level splits)

| Metric    | Value               |
| --------- | ------------------- |
| AUC-ROC   | **94.95% ± 2.81%**  |
| Accuracy  | **87.27% ± 3.76%**  |
| F1 Score  | **86.56% ± 4.56%**  |
| EER       | **13.19% ± 4.21%**  |
| Threshold | 0.7737 (Youden's J) |

### Ablation Study

| Variant          | Accuracy   |
| ---------------- | ---------- |
| CNN-Only         | 88.93%     |
| LSTM-Only        | 89.33%     |
| Transformer-Only | 90.51%     |
| **Full Hybrid**  | **90.32%** |

### GradCAM Explainability

Top discriminative joints: L_Shoulder (1.0), R_Heel (0.94), L_Foot (0.93), L_Knee (0.90).
Feature group contributions: Coordinates 47.7%, Velocities 37.4%, Joint Angles 14.9%.

## Architecture

```
Video Input + Claimed Identity
         │
    MediaPipe Pose Estimation (CPU)
    33 landmarks → 12 gait keypoints + 6 joint angles + velocities → 78-dim features
         │
    ┌────┴────┐
    │ GaitEncoder (1D CNN + Residual Blocks)
    │ 78 → 64 → 128
    └────┬────┘
         │
    ┌────┴────────────────────────────┐
    │ DualPathTemporalModel            │
    │ ├─ BiLSTM (1-layer, h=64)        │
    │ └─ Transformer (d=128, h=4, L=2) │
    │    Fusion MLP → 128-dim          │
    └────┬────────────────────────────┘
         │
    ┌────┴────┐
    │ IdentityVerifier (Siamese)
    │ Compare video vs enrolled gait → AUTHENTIC / DEEPFAKE
    └─────────┘
```

- **Training**: PyTorch on CUDA (RTX 3050)
- **Feature extraction**: MediaPipe (CPU) — chosen over MoveNet for superior angular accuracy in gait analysis

## Project Structure

```
├── scripts/
│   ├── run_pipeline.py              # End-to-end pipeline orchestrator
│   ├── preprocessing/
│   │   ├── preprocess_videos.py     # MediaPipe feature extraction
│   │   └── augment_videos.py        # 16x data augmentation
│   ├── training/
│   │   └── train.py                 # Model training with balanced sampling
│   ├── evaluation/
│   │   ├── evaluate.py              # LOOCV, EER, ROC-AUC evaluation
│   │   ├── ablation_study.py        # Component ablation experiments
│   │   └── run_gradcam.py           # GradCAM explainability
│   ├── inference/
│   │   └── inference.py             # Single video prediction
│   └── enrollment/
│       └── enroll_identities.py     # Build identity gait signatures
├── models/
│   ├── gait_encoder.py              # 1D CNN spatial encoder
│   ├── temporal_model.py            # BiLSTM + Transformer dual path
│   ├── identity_verifier.py         # Siamese comparison + classifier
│   └── full_pipeline.py             # End-to-end model assembly
├── utils/
│   ├── pose_extraction.py           # MediaPipe 78-dim feature extractor
│   ├── data_loader.py               # Dataset with balanced pair sampling
│   ├── gradcam.py                   # GradCAM for gait encoder
│   ├── visualization.py             # Plotting utilities
│   └── logger.py                    # Logging utility
├── data/
│   ├── videos/                      # Original walking videos
│   ├── augmented_videos/            # Augmented training videos
│   └── gait_features/               # Extracted features (.pkl)
├── outputs/
│   ├── checkpoints/                 # Model checkpoints
│   ├── evaluation/                  # LOOCV results
│   ├── ablation/                    # Ablation study results
│   └── gradcam/                     # GradCAM visualizations
├── PLAN.md                          # Implementation roadmap
├── LITERATURE_REVIEW.md             # IEEE-format literature review
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
git clone https://github.com/your-username/DeepFake-Detection.git
cd DeepFake-Detection
pip install -e .
pip install -r requirements.txt
```

The editable install (`pip install -e .`) makes the `models` and `utils` packages importable from any script location.

### Data Preparation

1. Place walking videos in `data/videos/` named as `{Name}_{View}{Number}.mp4` (e.g., `Arhaan_F1.mp4`, `John_S2.mp4`)
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
python scripts/preprocessing/preprocess_videos.py --input_dir data/augmented_videos --output_file data/gait_features/gait_features.pkl

# 3. Enroll identities
python scripts/enrollment/enroll_identities.py --features_file data/gait_features/gait_features.pkl

# 4. Train model
python scripts/training/train.py --features_file data/gait_features/gait_features.pkl --epochs 50

# 5. Evaluate (single checkpoint)
python scripts/evaluation/evaluate.py --checkpoint outputs/checkpoints/checkpoint_epoch_best.pth --save_plots --save_results

# 6. Run LOOCV (13-fold cross-validation)
python scripts/evaluation/evaluate.py --loocv --loocv_epochs 30

# 7. Run ablation study
python scripts/evaluation/ablation_study.py

# 8. Run GradCAM explainability
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

# 11. Verify gait preservation in face-swapped videos
python scripts/evaluation/verify_gait_preservation.py --original_video path/to/original.mp4 --deepfake_video path/to/deepfake.mp4

# 12. Run inference on deepfake video to detect it
python scripts/inference/inference.py --video data/deepfake/{BodyPerson}_body_{FacePerson}_face.mp4 --claimed_identity "{FacePerson}"
```

### Inference Output

```
AUTHENTIC — Verified as Arhaan    (similarity: 0.87, confidence: 0.92)
DEEPFAKE OF Arhaan                (similarity: 0.23, confidence: 0.95)
```

## Evaluation Metrics

| Metric           | Description                           |
| ---------------- | ------------------------------------- |
| AUC-ROC          | Area under ROC curve                  |
| EER              | Equal Error Rate (FAR = FRR)          |
| F1 Score         | Harmonic mean of precision and recall |
| Precision        | True positives / predicted positives  |
| Recall           | True positives / actual positives     |
| Confusion Matrix | TP, FP, TN, FN breakdown              |

### Cross-Validation

LOOCV with 13 folds (1 subject held out per fold) ensures no identity leakage between training and testing. Reports mean ± std across all folds.

## Feature Extraction

MediaPipe Pose extracts 33 3D landmarks per frame. The gait feature vector (78-dim) comprises:

| Component              | Dimensions | Description                             |
| ---------------------- | ---------- | --------------------------------------- |
| Normalized coordinates | 36 (12×3)  | Hip-centered x,y,z of 12 gait keypoints |
| Joint angles           | 6          | Knee, hip, ankle flexion angles         |
| Velocities             | 36 (12×3)  | Frame-to-frame coordinate deltas        |

## Citation

If you use this work, please cite:

```
@misc{gait-deepfake-2026,
  title={Gait-Based Deepfake Detection Using Skeletal Pose Analysis},
  year={2026}
}
```

## License

This project is for academic/research purposes.
