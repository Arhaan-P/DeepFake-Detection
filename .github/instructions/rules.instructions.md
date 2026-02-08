---
applyTo: "**"
---

# Copilot Rules — Gait-Based Deepfake Detection

## Project Context

This is a **gait-analysis-based deepfake detection** system. Given a video and a claimed identity, the model extracts skeletal pose keypoints (MediaPipe, 78-dim), analyzes walking patterns via a CNN+BiLSTM+Transformer hybrid, and outputs AUTHENTIC or DEEPFAKE.

**Read `PLAN.md` at project root for the full implementation roadmap before starting any major work.**

---

## Core Rules

### #1 — Gait Analysis is the Foundation
The ML model detects deepfakes by analyzing **gait (walking patterns)**, NOT facial features. The system:
1. Takes a video + claimed identity as input
2. Extracts gait keypoints using MediaPipe (78-dim features)
3. Compares gait embedding against enrolled identity
4. Outputs: `AUTHENTIC — Verified as <name>` or `DEEPFAKE OF <claimed_identity>`

### #2 — Never Hardcode or Cheat
- No hardcoded thresholds — use ROC-derived optimal threshold
- No accuracy inflation — report only on held-out subjects
- No training data leakage — LOOCV with subject-level splits
- If accuracy is low, report honestly and investigate why

### #3 — Pose Backend: MediaPipe (CPU)
- Feature extraction uses `utils/pose_extraction.py` (MediaPipe)
- 33 landmarks, 3D coords, 12 gait points, 6 joint angles → **78-dim features**
- Do NOT use MoveNet (`utils/pose_extraction_gpu.py`) for features — worse gait accuracy
- MoveNet file exists for reference only

### #4 — Model Training: PyTorch on GPU
- Training runs on CUDA (RTX 3050)
- Always check and print device at start:

```python
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.2f} GB")
```

### #5 — Progress Monitoring
- Display epoch numbers and batch progress
- Show training and validation metrics (loss, accuracy, F1)
- Include time estimates and completion percentages
- Log checkpoint saves and model updates

### #6 — Class Balance
- Data loader MUST use 50/50 balanced pair sampling
- Loss function MUST use class weights
- If train accuracy = 90% exactly, the model has collapsed — investigate immediately

### #7 — Feature Consistency
- Training features, enrolled identity features, and inference features must ALL use the same extractor (MediaPipe, 78-dim)
- `input_dim` in model config must match feature extractor output (78)
- Any mismatch = runtime crash or garbage predictions

### #8 — Research Paper Quality
- Use LOOCV (13 folds) for evaluation
- Report: AUC-ROC, EER, F1, Precision, Recall, Confusion Matrix
- Do NOT use L1/L2/MSE/RMSE — those are regression metrics
- Include ablation study (CNN-only, LSTM-only, Transformer-only, Hybrid)

### #9 — No Summary Files
Never create summary markdown files unless explicitly requested by the user.

### #10 — Research and Verification
Always research before making claims. Use Perplexity search when stuck or unsure.

---

## File Structure Reference

| File | Purpose |
|------|---------|
| `train.py` | Model training with balanced sampling |
| `evaluate.py` | Comprehensive metrics + LOOCV |
| `inference.py` | Single video prediction |
| `preprocess_videos.py` | MediaPipe feature extraction |
| `enroll_identities.py` | Build identity gait signatures |
| `augment_videos.py` | 16x data augmentation |
| `models/gait_encoder.py` | 1D CNN spatial encoder |
| `models/temporal_model.py` | BiLSTM + Transformer dual path |
| `models/identity_verifier.py` | Siamese comparison + classifier |
| `models/full_pipeline.py` | End-to-end assembly |
| `utils/pose_extraction.py` | MediaPipe feature extractor (USE THIS) |
| `utils/pose_extraction_gpu.py` | MoveNet extractor (DO NOT USE for features) |
| `utils/data_loader.py` | Dataset with balanced pair sampling |
| `utils/gradcam.py` | GradCAM explainability |
| `utils/visualization.py` | Plotting utilities |
| `utils/logger.py` | Logging utility |
| `PLAN.md` | Full implementation roadmap |
