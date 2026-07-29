# Deepfake Detection using Gait Analysis — Technical Overview

> Panel reference notes. Plain, technical, "what we used and why."
> Authors: Arhaan Penwala, Aditya Bharti, Abhishek Arun Raja Kodeeswaran · Faculty: Dr. Deepika Roselind Johnson (VIT Chennai)
> Dataset (IEEE DataPort, DOI 10.21227/ngh5-b637): https://ieee-dataport.org/documents/deepfake-detection-using-gait-analysis

---

## 1. The core idea (one line)

Detect deepfakes by **how a person walks (gait)**, not by their face. A face can be swapped perfectly; a person's natural walking motion is much harder to fake and stays consistent frame-to-frame.

**Why gait works when face-based detectors fail:**
- Face detectors look at facial artifacts, blinking, lighting, compression → new generators remove these.
- Gait is **behavioral + biomechanical**, not just visual pixels. Current deepfake tools swap the *face* and don't model the target's body motion.
- So: swap A's face onto B's body → the face looks like A, but the **walk is still B's**. We catch that mismatch.

---

## 2. Problem framing

- Task is **verification, not classification.** We don't ask "is this fake in general?" We ask: *"Does the walk in this video match the person it claims to be?"*
- Output: **AUTHENTIC** (walk matches claimed identity) or **DEEPFAKE** (walk doesn't match), plus a similarity score and confidence.
- Why verification: it generalizes to **new people not seen in training** — you just enroll their gait signature, no retraining. This is the standard biometric setup (like face-ID / fingerprint verification).

---

## 3. Dataset

| Item | Value |
|---|---|
| Subjects | 13 (male + female, ~19–25 yrs) |
| Raw videos | ~65 (each subject, multiple passes, **2 views**: Frontal + Side) |
| Augmentation | **16×** → **1,056** video samples |
| Format | MP4, `SubjectName_ViewN.mp4`; features shipped as `.pkl` |
| Per-frame feature | **78-dim** gait vector |
| Sequence length | **60 frames** normalized (~2 gait cycles @ 30 fps) |

**Why augmentation (16×):** only 13 subjects → tiny dataset → deep model would overfit. Augmentation multiplies samples *and* simulates real-world capture variation the model must survive: lighting, camera quality, angle, sensor noise, mirrored walking direction, speed changes.
Augmentations used (`augment_videos.py`): horizontal flip, brightness/contrast, slight rotation (±10°), Gaussian blur, color jitter, grayscale, Gaussian noise, zoom, plus temporal speed (0.8×/1.2×), reverse, jitter. **All spatial augmentations are applied identically across every frame of a clip** — this is deliberate so the gait motion stays temporally consistent and isn't corrupted.

**Synthetic/deepfake samples:** generated with **FaceFusion** (`inswapper_128_fp16`, face-only mask) — swap person A's face onto person B's walking body. We then verify the swap preserved the body's gait, so the deepfake is a valid "wrong walk" test case.

---

## 4. Feature extraction — MediaPipe Pose (78-dim vector)

Pipeline: video → MediaPipe Pose → 33 body landmarks/frame → keep 12 gait keypoints → build 78-dim vector/frame → normalize to 60 frames.

**12 keypoints used:** shoulders, hips, knees, ankles, heels, foot-tips (lower-body + upper-body sway).

**The 78 dimensions:**
| Component | Dims | What it captures |
|---|---|---|
| Hip-centered 3D coordinates | 36 (12×3) | body pose/shape each frame |
| Joint flexion angles | 6 | knee/hip/ankle bend — biomechanics |
| Frame-to-frame velocities | 36 (12×3) | how joints move — the actual "motion" |

**Why these design choices:**
- **MediaPipe over MoveNet** → better angular accuracy for gait, gives 3D (x,y,z) landmarks, runs **on CPU**, no GPU needed for extraction, and needs no face → works even when the face is swapped.
- **Hip-centering** (subtract mid-hip from all points) → removes where the person is in the frame / camera position. Model sees the *walk*, not the location. Translation-invariant.
- **Angles + velocities added to raw coords** → coords alone = a pose; adding joint angles = biomechanics, adding velocities = dynamics. GradCAM later confirmed all three matter (coords 47.7%, velocities 37.4%, angles 14.9%).
- **60-frame normalization** (interpolation) → fixed length so we can batch; ~2 full gait cycles is enough to capture the repeating walking rhythm.

Enrollment (`enroll_identities.py`): a person's reference "gait signature" = the **average** of their original (non-augmented) clips' features. Verification compares a query video against this signature.

---

## 5. Model architecture

Designed hybrid: **1D CNN → (BiLSTM ‖ Transformer) → verification head.**

```
78-dim/frame gait features (60 frames)
        │
   GaitEncoder        1D CNN + residual blocks   78 → 64 → 128
   (spatial)          learns per-frame spatial features
        │
   DualPathTemporalModel  (two parallel paths, then fused)
     ├─ BiLSTM   1 layer, hidden 64   → local motion (stride timing, step rhythm)
     └─ Transformer  d=128, 4 heads, 2 layers → global long-range motion across the whole gait cycle
     └─ Fusion MLP → 128-dim gait embedding
        │
   Verification head (Siamese-style compare vs enrolled signature)
        │
   AUTHENTIC / DEEPFAKE  + similarity + confidence
```

**Why three model types stacked (this is the key "why"):**
- **CNN** — good at local **spatial** patterns within a frame (relationship between joints). Cheap, fast, extracts features before temporal modeling. Residual blocks = stable gradients.
- **BiLSTM** — reads the sequence in order, both directions. Captures **short-term** temporal patterns: step timing, stride rhythm, left-right cadence.
- **Transformer (self-attention)** — captures **long-range** dependencies: relates frame 5 directly to frame 55 without passing through every step. A gait cycle spans the whole clip; attention sees the whole thing at once.
- **Fusion of BiLSTM + Transformer** — short-term rhythm + long-term structure together. Ablation confirmed the combined model is competitive/best.

**Verification head — honest technical detail:** in the code's verification mode, the AUTHENTIC/DEEPFAKE decision is made by a **difference-based 1D CNN**. It takes the query video's per-frame features and the enrolled signature, computes their per-timestep **difference, |difference|, and product**, and a small CNN classifies that. We tried the simpler route (encode both → compare embeddings) but it **collapsed** (model ignored the input). Feeding raw per-frame differences gives the classifier direct access to *where* the two walks disagree, which trained stably. The CNN+BiLSTM+Transformer embedding is still produced (used for the similarity score, standalone classification, and the ablation study).

Model is **small** (~a few M params, ~fits 6 GB GPU) — this matters for the embedded/Jetson plan (Section 8).

---

## 6. Training & evaluation

**Training (`train.py`):** PyTorch on CUDA (RTX 3050). AdamW optimizer, class-weighted cross-entropy, gradient clipping (max-norm 1.0), `ReduceLROnPlateau` scheduler, early stopping. Balanced pair sampling: every sample yields one **positive** pair (claim true identity) + one **negative** pair (claim someone else) → 50/50, so the model can't cheat by always guessing one class.

**Data split — by subject, never by clip.** All of one person's videos go entirely to train *or* test, never both.
**Why:** if the same person appears in train and test, the model memorizes *them*, not gait in general → fake high scores (data leakage). Splitting by person forces true generalization.

**LOOCV (Leave-One-Out Cross-Validation), 13 folds:** train on 12 subjects, test on the 1 held-out subject, repeat 13×, average. Standard for small datasets and the honest way to report "how well does this work on a person we've never seen."

**Metrics:** we report biometric-standard metrics, not just accuracy:
- **AUC-ROC** — threshold-independent separability (main headline number).
- **EER** (Equal Error Rate) — where false-accept = false-reject; standard for verification systems.
- **F1 / Precision / Recall**, confusion matrix, and **Youden's-J optimal threshold**.

### Results (LOOCV, 13-fold, subject-level)

| Metric | Value |
|---|---|
| AUC-ROC | **94.95% ± 2.81%** |
| Accuracy | 87.27% ± 3.76% |
| F1 | 86.56% ± 4.56% |
| EER | 13.19% ± 4.21% |
| Operating threshold | 0.7737 (Youden's J) |

### Ablation (which component matters)

| Variant | Accuracy |
|---|---|
| CNN-only | 88.93% |
| LSTM-only | 89.33% |
| Transformer-only | 90.51% |
| **Full hybrid** | **90.32%** |

→ Every component alone is decent; the transformer path is strongest; the hybrid is on par with the best single path while being more robust across folds.

### Explainability — GradCAM

- Most discriminative joints: **L-Shoulder, R-Heel, L-Foot, L-Knee** → matches biomechanics (upper-body sway + foot strike carry identity).
- Feature-group contribution: **coordinates 47.7%, velocities 37.4%, joint angles 14.9%** → confirms motion (velocities) is nearly as important as pose, so the model really is using *how* they move, not just static shape.

---

## 7. Why this is novel

No prior **published dataset or method** does gait-based *deepfake* detection specifically (gait recognition exists; using it against deepfakes doesn't). We published the dataset on IEEE DataPort to fill that gap. Deepfake generators optimize faces/voices and don't model individual biomechanics → gait remains a signal they can't currently reproduce.

---

## 8. Where it's going (AI & Robotics + Cyber-Physical integration)

The project must show **both departments' aspects**. Plan:

**(a) Embedded / edge deployment — NVIDIA Jetson.**
- Move MediaPipe extraction + model inference onto a **Jetson** (borrowed) → real-time, **on-device** gait verification with no cloud.
- Feasible because: MediaPipe runs on CPU/edge, and our model is small enough to quantize (INT8 / TensorRT) for the Jetson's GPU.
- Payoff: privacy (video never leaves the device), low latency, a real **cyber-physical** artifact (camera + edge compute) instead of just a notebook.

**(b) Robotic Process Automation (RPA)** — as advised by faculty coordinator Dr. Roselind Johnson.
- Automate the **detection workflow** as an unattended pipeline: ingest video → extract gait → verify against enrolled identity → flag + log/report.
- RPA orchestrates the stages (currently manual CLI steps) into one hands-off process — useful for **batch screening** of media feeds.

**(c) Domain expansion.** Reframe from "deepfake detection" to a **surveillance / fake-video verifier** for high-stakes domains (e.g., political / public-figure video authenticity).

**(d) Data expansion.** Collect more subjects across **demographics, ages, and countries** (collaborators at other universities) → improves generalization and strengthens the IEEE dataset.

---

## 9. Likely panel questions — quick answers

**Q: Why not just use existing face-based deepfake detectors?**
They fail on modern generators that remove facial artifacts. Gait is a complementary, behavioral signal that current fakes don't reproduce.

**Q: Only 13 subjects — isn't that too few?**
Yes, it's a pilot dataset (the first of its kind). We handle it with 16× augmentation, subject-level LOOCV (no leakage), and report AUC/EER with std across folds. Section 8(d) is the scale-up plan.

**Q: Isn't augmentation just duplicating data / inflating scores?**
No — augmentations simulate real capture variation (lighting, angle, noise, speed, mirror) and are held **temporally consistent per clip**. And crucially, LOOCV tests on a **completely unseen person**, so augmented copies of a test subject never appear in training.

**Q: Could a future deepfake also fake the gait?**
Possibly, which is why we treat gait as a **complementary** signal and keep explainability (GradCAM) to show what drives decisions. Today's face-swap tools don't model body biomechanics.

**Q: What's the actual decision mechanism?**
A Siamese-style verification: compare the query video's gait against the claimed identity's enrolled signature. Implemented as a difference-based CNN over per-frame feature differences (chose this over embedding-comparison because the latter collapsed during training).

**Q: Why MediaPipe and not OpenPose/MoveNet?**
Better joint-angle accuracy for gait, gives 3D landmarks, CPU-friendly (important for Jetson), and face-independent.

**Q: How is it "AI and Robotics" + "cyber-physical"?**
AI = the detection model. Robotics/CPS = Jetson edge deployment (camera + on-device inference) and RPA automation of the pipeline (Section 8).

---

## 10. One-paragraph summary (for the pitch)

We built the first gait-based deepfake detector and released its dataset on IEEE DataPort. Instead of inspecting faces, we extract 78-dimensional skeletal gait features per frame with MediaPipe Pose (coordinates, joint angles, velocities of 12 keypoints), normalize each clip to 60 frames, and feed them to a CNN + BiLSTM + Transformer model that learns a person's walking signature. Verification compares a query video's gait against an enrolled signature and flags a mismatch as a deepfake. Under strict subject-level Leave-One-Out cross-validation it reaches **94.95% AUC-ROC**. Next, we move inference to an **NVIDIA Jetson** for real-time on-device (cyber-physical) detection and use **RPA** to automate the screening pipeline, targeting fake-video verification in high-stakes domains like political media.
