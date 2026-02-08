# Gait-Based Deepfake Detection — Session Progress Report

**Date:** February 8, 2026  
**Status:** Phase 1 Complete ✅ | Phase 2 In Progress (partial)  
**Session Context:** Multiple bug fixes, code cleanup, new utilities, and feature extraction setup

---

## 🎯 Project Overview

A **novel gait-based deepfake detection system** that analyzes walking patterns instead of faces. Given a video + claimed identity, the model determines if it's AUTHENTIC or DEEPFAKE.

- **Architecture:** CNN + BiLSTM + Transformer hybrid (3.8M params)
- **Features:** MediaPipe Pose → 78-dim gait features (12 keypoints + 6 angles + velocities)
- **Dataset:** 13 identities, 1,122 augmented videos
- **Training:** PyTorch on GPU (NVIDIA RTX 3050, CUDA)
- **Evaluation:** LOOCV (13-fold), comprehensive metrics (AUC, EER, F1, etc.)

---

## ✅ COMPLETED IN THIS SESSION

### Phase 1: Code Fixes & Cleanup

#### 1. **Critical Bug Fixes** ✅

- **data_loader.py**: Fixed 90% class imbalance → balanced 50/50 pair sampling
- **train.py**: Added class-weighted `CrossEntropyLoss`, removed dead imports, reduced epochs 100→50
- **inference.py**: Fixed feature pipeline (use pre-computed enrolled features), corrected output messages
- **evaluate.py**: Fixed `create_data_loaders()` API call (4 returns, correct params), fixed confusion matrix
- **run_pipeline.py**: Fixed augmented_dir default path
- **augment_videos.py**: Fixed seed contamination (RandomState instead of global seed)
- **pose_extraction_gpu.py**: Implemented lazy TensorFlow loading (no GPU init at import)

#### 2. **Dead Code Removal** ✅

- Removed unused `TemporalAttentionPool` from models/full_pipeline.py
- Removed 8 unused imports across 7 files (DataLoader, nn, F, Path, sys, torch, Tuple, train_test_split)
- Removed 5 redundant `sys.path.insert(0, ...)` statements
- Cleaned up resulting unused `import sys` from 5 files

#### 3. **New Utilities Created** ✅

- **utils/gradcam.py** (275 lines): GradCAM visualization for gait analysis
  - `GaitGradCAM`: Temporal frame importance heatmaps
  - `JointImportanceAnalyzer`: Body joint + angle importance via input gradients
  - `generate_explainability_report()`: Full explainability pipeline with plots
  - Visualization functions for temporal, joint, and feature group importance

#### 4. **Enhanced evaluate.py** ✅

- Added `compute_eer()`: Equal Error Rate calculation (biometric standard)
- Added `run_loocv()`: 13-fold Leave-One-Out Cross-Validation with per-subject training
- Added `--loocv` CLI argument: Separate LOOCV workflow (trains N new models)
- Added EER to all output/save paths
- Added DET curve support

#### 5. **Documentation** ✅

- **README.md**: Complete project overview, setup, usage, metrics explanation
- **PLAN.md (updated)**: 20-step implementation roadmap with detailed phases
- **.github/instructions/rules.instructions.md (updated)**: 10 core development rules

#### 6. **Dependencies & Config** ✅

- Updated **requirements.txt** to pin mediapipe==0.10.32
- Fixed MediaPipe API compatibility (updated from old mp.solutions.pose to new Vision API)

#### 7. **Environment Setup** ✅

- Updated **.gitignore** to exclude data/, outputs/, logs/, checkpoints/

### Phase 2: Feature Extraction

#### **Step 1: Extract Gait Features** ✅ COMPLETE

```
✓ Command: python preprocess_videos.py --augmented_dir data/augmented_videos --output data/gait_features/gait_features.pkl --sequence_length 60
✓ Status: COMPLETED (Feb 8, 17:49 UTC)
✓ Output: data/gait_features/gait_features.pkl (1,122 videos, 78-dim features)
✓ Details: 13 identities with 85-102 videos each
```

---

## ⏳ REMAINING TASKS (Not Yet Started)

### Phase 2: Feature Preparation & Model Training

#### **Step 2: Enroll Identities** (2-3 min) — NOT YET RUN

```powershell
python enroll_identities.py --features_file data/gait_features/gait_features.pkl --output_file data/gait_features/enrolled_identities.pkl
```

- Creates gait signatures for all 13 identities
- Output: `data/gait_features/enrolled_identities.pkl`

#### **Step 3: Train Fresh Model** (45-60 min) — NOT YET RUN

```powershell
python train.py --epochs 50 --batch_size 16 --learning_rate 1e-4 --features_file data/gait_features/gait_features.pkl --enrolled_file data/gait_features/enrolled_identities.pkl --output_dir outputs
```

- Fresh training on GPU with balanced sampling + 78-dim MediaPipe features
- Should see train accuracy increase (NOT stay at 90%)
- Should see val authentic F1 > 0%
- Output: `outputs/checkpoints/checkpoint_epoch_best.pth`

#### **Step 5: Test Inference** (5 min) — NOT YET RUN

```powershell
python inference.py --video data/videos/Arhaan_F.mp4 --claimed_identity Arhaan --checkpoint outputs/checkpoints/checkpoint_epoch_best.pth
```

- Test 3 scenarios:
  1. Real video (X) + claim X → `AUTHENTIC — Verified as <name>`
  2. Real video (X) + claim Y → `DEEPFAKE OF <Y>`
  3. Deepfake video + claim X → `DEEPFAKE OF <X>`

#### **Step 4: LOOCV (Leave-One-Out CV)** (2-3 hours) — NOT YET RUN (OVERNIGHT)

```powershell
python evaluate.py --loocv --loocv_epochs 20 --batch_size 16 --features_file data/gait_features/gait_features.pkl --enrolled_file data/gait_features/enrolled_identities.pkl --output_dir outputs/loocv
```

- Trains 13 separate models (each leaving out 1 subject)
- Computes mean ± std of all metrics across folds
- Output: `outputs/loocv/loocv_results.json`
- **NOTE:** Run this overnight — takes 2-3 hours

### Phase 3: Ablation & Comparison (Not Started)

- Train alternative architectures (CNN-only, LSTM-only, Transformer-only)
- Generate ablation study comparison table
- Compare against published methods in literature

### Phase 4: Hyperparameter Tuning (Not Started)

- Systematic search over learning rate, dropout, embedding dim, etc.
- Use LOOCV results to select best config

### Phase 5: GitHub Cleanup (Not Started)

- Final code review & polish
- Add comprehensive docstrings
- Prepare for publication

---

## 📊 Current Project State

### Files Modified This Session

| File                    | Changes                         | Status      |
| ----------------------- | ------------------------------- | ----------- |
| data_loader.py          | 50/50 balanced sampling         | ✅ Fixed    |
| train.py                | Class weights, dead imports     | ✅ Fixed    |
| inference.py            | Feature pipeline, output format | ✅ Fixed    |
| evaluate.py             | API alignment, LOOCV, EER       | ✅ Enhanced |
| run_pipeline.py         | Path defaults                   | ✅ Fixed    |
| augment_videos.py       | Seed contamination              | ✅ Fixed    |
| pose_extraction_gpu.py  | Lazy TF loading                 | ✅ Fixed    |
| pose_extraction.py      | MediaPipe 0.10.32 API           | ✅ Fixed    |
| models/full_pipeline.py | Dead code removal               | ✅ Cleaned  |
| Add: utils/gradcam.py   | NEW: Explainability             | ✅ Created  |
| requirements.txt        | MediaPipe pinned                | ✅ Updated  |
| .gitignore              | Project paths                   | ✅ Updated  |
| README.md               | NEW: Full documentation         | ✅ Created  |
| rules.instructions.md   | Core development rules          | ✅ Updated  |

### Data Status

```
✓ Videos: 1,122 augmented (13 identities, 85-102 each)
✓ Features: Extracted (78-dim MediaPipe, data/gait_features/gait_features.pkl)
✓ Enrolled IDs: Pre-made (data/gait_features/enrolled_identities.pkl)
⏳ Model training: Ready to start (old checkpoints exist but will be overwritten)
```

### Known Issues Resolved

- ❌ MediaPipe 0.10.32 API incompatibility → ✅ FIXED (updated pose_extraction.py)
- ❌ Class imbalance (90% deepfake) → ✅ FIXED (balanced pair sampling)
- ❌ Feature dimension mismatch → ✅ FIXED (MediaPipe 78-dim features)
- ❌ evaluate.py API mismatches → ✅ FIXED (corrected function calls)

---

## 🚀 NEXT STEPS FOR NEW CHAT

**Run these commands in order:**

```powershell
# Step 2: Enroll Identities (2-3 min)
python enroll_identities.py --features_file data/gait_features/gait_features.pkl --output_file data/gait_features/enrolled_identities.pkl

# Step 3: Train Fresh Model (45-60 min)
python train.py --epochs 50 --batch_size 16 --learning_rate 1e-4 --features_file data/gait_features/gait_features.pkl --enrolled_file data/gait_features/enrolled_identities.pkl --output_dir outputs

# Step 5: Test Inference (5 min) — after Step 3 completes
python inference.py --video data/videos/Arhaan_F.mp4 --claimed_identity Arhaan --checkpoint outputs/checkpoints/checkpoint_epoch_best.pth

# Step 4: LOOCV [RUN OVERNIGHT] (2-3 hours)
python evaluate.py --loocv --loocv_epochs 20 --batch_size 16 --features_file data/gait_features/gait_features.pkl --enrolled_file data/gait_features/enrolled_identities.pkl --output_dir outputs/loocv
```

---

## 📝 Key Metrics to Watch

**During Training (Step 3):**

- Train accuracy should **increase** (not stay at 90%)
- Val accuracy should improve with epochs
- Val authentic F1 should be **> 0%** (proves learning)
- Val deepfake F1 should be high

**During LOOCV (Step 4):**

- Accuracy: mean ± std across 13 folds
- F1 Score: mean ± std
- ROC-AUC: mean ± std
- EER: mean ± std

**Expected Results:**

- Accuracy: 75-95% (depends on gait distinctiveness)
- AUC-ROC: 0.85-0.99
- EER: 5-15%
- F1: 0.80-0.95

---

## 🔧 Tech Stack

| Component          | Technology              | Version          |
| ------------------ | ----------------------- | ---------------- |
| Pose Backend       | MediaPipe               | 0.10.32          |
| Feature Extraction | MediaPipe Pose          | 78-dim           |
| Model Framework    | PyTorch                 | Latest           |
| GPU                | NVIDIA CUDA             | 12.4             |
| Training           | Adam + CrossEntropyLoss | Balanced weights |
| Evaluation         | LOOCV + scikit-learn    | 13 folds         |
| Visualization      | Grad-CAM + Matplotlib   | Custom           |

---

## 📚 Architecture Overview

```
Input: Video + Claimed Identity
        │
        ▼
MediaPipe Pose Extraction (CPU, 78-dim)
        │
        ├─ 12 keypoints × 3 coords (normalized)
        ├─ 6 joint angles
        └─ 12 keypoints × 3 velocities
        │
        ▼
GaitEncoder (1D CNN + Residual Blocks)
78 → 64 → 128 → 256 features
        │
        ▼
DualPathTemporalModel (BiLSTM + Transformer)
├─ 2-layer BiLSTM (h=128)
├─ 4-layer Transformer (d=256, h=8)
└─ Fusion MLP → 256-dim embedding
        │
        ▼
IdentityVerifier (Siamese)
Compare video vs enrolled gait
        │
        ▼
Output: AUTHENTIC / DEEPFAKE
```

---

## 💾 Important Paths

```
Project Root: C:\Arhaan\PROJECTS\DeepFake-Detection
├── data/
│   ├── videos/                    # Original videos
│   ├── augmented_videos/          # 1,122 augmented videos
│   └── gait_features/
│       ├── gait_features.pkl      # ✅ CREATED (Step 1)
│       └── enrolled_identities.pkl # ⏳ TO CREATE (Step 2)
├── outputs/
│   ├── checkpoints/
│   │   └── checkpoint_epoch_best.pth  # ⏳ TO CREATE (Step 3)
│   └── loocv/                    # ⏳ TO CREATE (Step 4)
├── models/                        # ✅ All ready
├── utils/
│   ├── pose_extraction.py        # ✅ Fixed (MediaPipe 0.10.32)
│   └── gradcam.py               # ✅ NEW
├── logs/
│   └── preprocess_20260208_*.txt # Latest: 17:49
└── PLAN.md, README.md, rules.instructions.md  # ✅ Documentation
```

---

## 📋 Verification Checklist

- ✅ All Python files have valid syntax (17 files checked)
- ✅ All imports resolve correctly
- ✅ MediaPipe 0.10.32 API compatibility verified
- ✅ Feature extraction completed (1,122 videos processed)
- ✅ No uncommitted data leakage issues
- ✅ GPU/CUDA available for training
- ✅ Class balancing logic implemented
- ✅ Metrics computation ready (EER, AUC, F1, etc.)

---

## 🎓 Context for Next Chat

When you open a new chat, mention:

> "I'm continuing a gait-based deepfake detection project. **Step 1 (feature extraction) is COMPLETE.** I need to run Steps 2-5:
>
> - Step 2: Enroll identities
> - Step 3: Train model
> - Step 5: Test inference
> - Step 4: LOOCV overnight
>
> All critical bugs are fixed, new utilities are in place (GradCAM, LOOCV, EER metrics). The codebase is clean and ready. Let me know when ready to proceed."

Include this file link: `SESSION_PROGRESS.md`

---

**Last Updated:** February 8, 2026 @ 17:49 UTC  
**Session Time:** ~4 hours (bug fixes + cleanup + feature extraction)  
**Next Estimated Time:** ~2-3 hours (Steps 2-5)
