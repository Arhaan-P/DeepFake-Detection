# 🎯 Gait-Based Deepfake Detection - Implementation Guide

## Project Overview

**Title:** Deepfake Detection using Gait Analysis  
**Core Concept:** Detect whether a video is REAL or FAKE by analyzing the person's gait (walking pattern) and verifying against a CLAIMED IDENTITY.

---

## 🔑 Key Principle

```
INPUT: Video + Claimed Identity (e.g., "Arhaan")
OUTPUT: AUTHENTIC or DEEPFAKE
```

The system:

1. Extracts gait features from the video
2. Compares against the enrolled gait signature of the claimed identity
3. Determines if the gait matches → AUTHENTIC, or mismatches → DEEPFAKE

---

## 📁 Project Structure

```
DeepFake-Detection/
├── data/
│   ├── videos/                    # Original videos (PersonName_View.mp4)
│   ├── augmented_videos/          # Augmented dataset
│   ├── deepfake/                  # Deepfake test videos
│   ├── gait_features/             # Extracted pose sequences
│   └── enrolled_identities/       # Enrolled gait signatures per person
├── models/
│   ├── gait_encoder.py            # CNN for spatial features
│   ├── temporal_model.py          # BiLSTM + Transformer
│   ├── identity_verifier.py       # Identity matching network
│   └── full_pipeline.py           # Complete model
├── utils/
│   ├── pose_extraction.py         # MediaPipe pose extraction
│   ├── data_loader.py             # Dataset and DataLoader
│   └── visualization.py           # Grad-CAM, plots
├── train.py                       # Training script
├── inference.py                   # Prediction script
├── augment_videos.py              # Data augmentation
└── requirements.txt
```

---

## 🧠 Model Architecture

### Recommended: Hybrid CNN + BiLSTM + Transformer

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT VIDEO                               │
│                    (T frames × H × W × 3)                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    POSE EXTRACTION (MediaPipe)                   │
│              Extract 33 body landmarks per frame                 │
│                  Output: (T × 33 × 3) keypoints                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SPATIAL ENCODER (CNN)                         │
│         Process each frame's pose independently                  │
│              Extract spatial gait features                       │
│                  Output: (T × feature_dim)                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│               SHORT-TERM TEMPORAL (BiLSTM)                       │
│          Capture local motion patterns (stride, cadence)         │
│                  Output: (T × hidden_dim)                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│               LONG-TERM TEMPORAL (Transformer)                   │
│         Self-attention for global temporal dependencies          │
│                  Output: (sequence_embedding)                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DUAL OUTPUT HEADS                             │
│  ┌─────────────────────┐    ┌─────────────────────────────────┐ │
│  │  Gait Embedding     │    │  Classification Head            │ │
│  │  (for identity      │    │  (Real vs Fake)                 │ │
│  │   matching)         │    │                                 │ │
│  └─────────────────────┘    └─────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    IDENTITY VERIFICATION                         │
│       Compare extracted embedding with enrolled identity         │
│            Cosine Similarity > threshold → AUTHENTIC             │
│            Cosine Similarity < threshold → DEEPFAKE              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 Research-Based Model Comparison

| Architecture                    | Accuracy | AUC   | F1    | Notes                  |
| ------------------------------- | -------- | ----- | ----- | ---------------------- |
| **CNN + BiLSTM + Transformer**  | 98%      | 98.5% | 90.6% | ⭐ RECOMMENDED         |
| ST-GCN (Graph CNN for skeleton) | 98.34%   | -     | -     | Excellent for pose     |
| EfficientNetB0 + LSTM           | 95-98%   | 94%+  | 92.5% | Lightweight            |
| ResNeXt50 + LSTM                | 91.88%   | -     | 88%   | Robust features        |
| Vision Transformer              | 99.13%   | 98.5% | 99%   | Frame-level only       |
| 3D CNN Ensemble                 | 90.73%   | 99.3% | -     | Computationally heavy  |
| Motion-based (Pose + LSTM)      | 93%      | -     | -     | Gait-specific baseline |

---

## 🔧 Key Implementation Details

### 1. Pose Extraction (MediaPipe)

```python
# 33 body landmarks with (x, y, z, visibility)
# Key joints for gait: hips, knees, ankles, shoulders
GAIT_KEYPOINTS = [
    11, 12,  # Shoulders
    23, 24,  # Hips
    25, 26,  # Knees
    27, 28,  # Ankles
    29, 30,  # Heels
    31, 32   # Foot tips
]
```

### 2. Gait Features to Extract

- **Spatial:** Joint angles, limb lengths, body proportions
- **Temporal:** Stride length, cadence, step time, symmetry
- **Kinematic:** Joint velocities, accelerations

### 3. Training Strategy

- **Loss Functions:**
  - Cross-Entropy for classification
  - Triplet Loss / Contrastive Loss for identity embedding
  - Combined: `L = L_classification + λ * L_embedding`

- **Metrics:**
  - Accuracy, Precision, Recall, F1-Score
  - AUC-ROC curve
  - EER (Equal Error Rate) for identity verification

### 4. Data Split

- **By Person** (critical to prevent data leakage)
- Train: 80% of people
- Validation: 20% of people
- All augmentations of same person go to same split

---

## 🖥️ GPU/CUDA Requirements

```python
import torch

# ALWAYS check device at start
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Move model to GPU
model = model.to(device)

# Move data to GPU in training loop
inputs = inputs.to(device)
labels = labels.to(device)
```

---

## 📈 Progress Monitoring Requirements

```python
# Training loop must include:
for epoch in range(num_epochs):
    print(f"\nEpoch [{epoch+1}/{num_epochs}]")
    print("-" * 50)

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        # ... training code ...

        if batch_idx % 10 == 0:
            print(f"  Batch [{batch_idx}/{len(train_loader)}] "
                  f"Loss: {loss.item():.4f} "
                  f"Acc: {accuracy:.2f}%")

    # Epoch summary
    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
    print(f"  Time: {epoch_time:.2f}s | ETA: {eta:.2f}min")
```

---

## 🎨 Explainability (Grad-CAM)

Visualize what the model focuses on:

- For CNN: Highlight important body regions
- For temporal: Show important frames
- Use pytorch-grad-cam library

---

## 📋 Evaluation Protocol

### For Deepfake Detection:

1. Input: Video + Claimed Identity
2. Extract gait signature from video
3. Compare with enrolled signature of claimed identity
4. If similarity > threshold → AUTHENTIC
5. If similarity < threshold → DEEPFAKE

### For Testing:

- Use videos of enrolled people → should be AUTHENTIC
- Use videos of person A claiming to be person B → should be DEEPFAKE
- Use synthetic/manipulated videos → should be DEEPFAKE

---

## 🚀 Quick Start Commands

```bash
# 1. Run augmentation (already running)
python augment_videos.py

# 2. Extract gait features
python extract_gait_features.py

# 3. Train model
python train.py --epochs 100 --batch_size 16

# 4. Evaluate
python evaluate.py --checkpoint models/best_model.pth

# 5. Inference
python inference.py --video path/to/video.mp4 --claimed_identity "Arhaan"
```

---

## 📚 References

1. Motion-based Deepfake Detection (Pose + LSTM): 93% accuracy
2. CNN-BiLSTM-Transformer for video: 98% accuracy, 98.5% AUC
3. ST-GCN for skeleton-based recognition: 98.34% accuracy
4. Gait recognition with CNN-LSTM: 97.11% accuracy on CASIA-B
5. DeepGaitV2: State-of-the-art gait recognition

---

## ⚠️ Important Rules

1. **ALWAYS use GPU/CUDA** - Check device at start
2. **ALWAYS show progress** - Print epoch, batch, loss, accuracy
3. **Split by PERSON** - Not by video (prevents leakage)
4. **Claimed Identity** - Model must verify identity, not just detect fake
5. **Temporal Consistency** - Augmentations must be frame-consistent

---

_Last Updated: January 26, 2026_
