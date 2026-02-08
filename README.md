# Gait-Based Deepfake Detection

A novel deepfake detection system that analyzes **walking patterns (gait)** instead of facial features. Given a video and a claimed identity, the model extracts skeletal pose keypoints, compares gait embeddings, and determines whether the person is authentic or a deepfake.

## Key Innovation

No existing published work directly uses gait analysis for deepfake detection. This project proposes that since current deepfake generators focus on face/voice synthesis and do not model individualized biomechanics, gait patterns remain a reliable identity signal that deepfakes cannot replicate.

## Architecture

```
Video Input + Claimed Identity
         │
    MediaPipe Pose Estimation (CPU)
    33 landmarks → 12 gait keypoints + 6 joint angles + velocities → 78-dim features
         │
    ┌────┴────┐
    │ GaitEncoder (1D CNN + Residual Blocks)
    │ 78 → 64 → 128 → 256
    └────┬────┘
         │
    ┌────┴────────────────────────┐
    │ DualPathTemporalModel       │
    │ ├─ BiLSTM (2-layer, h=128)  │
    │ └─ Transformer (d=256, h=8) │
    │    Fusion MLP → 256-dim     │
    └────┬────────────────────────┘
         │
    ┌────┴────┐
    │ IdentityVerifier (Siamese)
    │ Compare video vs enrolled gait → AUTHENTIC / DEEPFAKE
    └─────────┘
```

- **Parameters**: ~3.8M
- **Training**: PyTorch on CUDA (GPU)
- **Feature extraction**: MediaPipe (CPU) — chosen over MoveNet for superior angular accuracy in gait analysis

## Project Structure

```
├── train.py                 # Model training with balanced sampling
├── evaluate.py              # Evaluation with LOOCV, EER, ROC-AUC
├── inference.py             # Single video prediction
├── preprocess_videos.py     # MediaPipe feature extraction
├── enroll_identities.py     # Build identity gait signatures
├── augment_videos.py        # 16x data augmentation
├── run_pipeline.py          # End-to-end pipeline orchestrator
├── models/
│   ├── gait_encoder.py      # 1D CNN spatial encoder
│   ├── temporal_model.py    # BiLSTM + Transformer dual path
│   ├── identity_verifier.py # Siamese comparison + classifier
│   └── full_pipeline.py     # End-to-end model assembly
├── utils/
│   ├── pose_extraction.py   # MediaPipe feature extractor (primary)
│   ├── data_loader.py       # Dataset with balanced pair sampling
│   ├── gradcam.py           # GradCAM explainability for gait
│   ├── visualization.py     # Plotting utilities
│   └── logger.py            # Logging utility
├── data/
│   ├── videos/              # Original walking videos
│   ├── augmented_videos/    # Augmented training videos
│   └── gait_features/       # Extracted features (.pkl)
├── outputs/
│   ├── checkpoints/         # Model checkpoints
│   └── tensorboard/         # Training logs
├── PLAN.md                  # Full implementation roadmap
└── requirements.txt         # Python dependencies
```

## Setup

### Requirements

- Python 3.9+
- NVIDIA GPU with CUDA (for training)
- ~6GB GPU memory (RTX 3050 or equivalent)

### Installation

```bash
git clone https://github.com/your-username/DeepFake-Detection.git
cd DeepFake-Detection
pip install -r requirements.txt
```

### Data Preparation

1. Place walking videos in `data/videos/` named as `{Name}_{View}{Number}.mp4` (e.g., `Arhaan_F1.mp4`, `John_S2.mp4`)
2. Place deepfake videos in `data/deepfake/`

## Usage

### Full Pipeline

```bash
python run_pipeline.py --videos_dir data/videos --augmented_dir data/augmented_videos
```

### Step-by-Step

```bash
# 1. Augment videos (16x)
python augment_videos.py --input_dir data/videos --output_dir data/augmented_videos

# 2. Extract gait features (MediaPipe)
python preprocess_videos.py --input_dir data/augmented_videos --output_file data/gait_features/gait_features.pkl

# 3. Enroll identities
python enroll_identities.py --features_file data/gait_features/gait_features.pkl

# 4. Train model
python train.py --features_file data/gait_features/gait_features.pkl --epochs 50

# 5. Evaluate
python evaluate.py --checkpoint outputs/checkpoints/checkpoint_epoch_best.pth --save_plots --save_results

# 6. Run LOOCV (13-fold cross-validation)
python evaluate.py --loocv --loocv_epochs 30

# 7. Inference on a single video
python inference.py --video path/to/video.mp4 --claimed_identity "PersonName"
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

## Explainability

GradCAM visualization shows which temporal frames and body joints the model attends to:

- **Temporal heatmap**: Which frames in the walking cycle matter most
- **Joint importance**: Which body parts (knees, ankles, hips) drive the decision
- **Feature group contribution**: Relative importance of coordinates vs joint angles vs velocities

## Feature Extraction

MediaPipe Pose extracts 33 3D landmarks per frame. The gait feature vector (78-dim) comprises:

| Component              | Dimensions | Description                             |
| ---------------------- | ---------- | --------------------------------------- |
| Normalized coordinates | 36 (12×3)  | Hip-centered x,y,z of 12 gait keypoints |
| Joint angles           | 6          | Knee, hip, ankle flexion angles         |
| Velocities             | 36 (12×3)  | Frame-to-frame coordinate deltas        |

## License

This project is for academic/research purposes.
