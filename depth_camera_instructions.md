# Depth-camera (Xbox 360 Kinect) study: instructions and findings

**Written:** 2026-09-19, in a read-only audit session at commit `6c2e23f` (branch `main`, clean tree).
**Audience:** a fresh Claude Code session, and the repo owner, with no memory of that session.
**Status:** nothing described here has been run on Kinect data. No existing file, checkpoint, cache or result was changed. This file is the only thing the session created.

---

## 0. How to use this document

1. Read §1 (goals and bottom line) and §5 (bugs). Then go straight to §14 (checklist) to see where things stand.
2. **Before running anything, ask the user the questions in §13.2.** Several steps need installs or hardware only the user can provide.
3. Never run a script with its default paths. Every default points at the existing phone dataset or model and will overwrite or append to it (§9.3).
4. Every number here traces to one of four things:
   - a repo file (path given),
   - a command output from the audit session (the scratch scripts are described inline),
   - a cited source (§15),
   - a stated estimate, which is labelled **estimate**.

   If you can't reproduce a number, trust the file, not this document.

---

## 1. Purpose, goals, and bottom line

The project detects face-swap deepfakes by checking whether the **gait** (walking pattern) in a video matches the claimed identity's enrolled gait. The original dataset was recorded on a phone.

- **Goal A (research question).** Would recording with a Microsoft Kinect for Xbox 360 depth camera change the performance metrics this project reports?
- **Goal B (practical).** Record a new dataset with the Kinect, run the whole existing pipeline on it, and train a **new model under a new name**, without touching the existing model or results.

### 1.1 Bottom line (blunt)

- **The published numbers (pooled ROC-AUC 94.95%, and the rest) will not "change".** They are properties of the phone dataset and its evaluation code. A Kinect study produces *new numbers on a new dataset*. Attributing any difference to "the Kinect" needs the controls in §11. With 13 subjects it can only detect effects of about 1.2–2.6 AUC points or more (§11.5).
- **Deepfake detection gets no benefit from depth, by construction.** A face-swapped video is an RGB file with no depth channel. Depth can only enter if the threat model becomes *live capture through a trusted depth sensor*. That is a different product, and there depth-based liveness checks (established for faces) are the right tool, not gait.
- **Identity verification (every LOOCV metric) could move either way.** The direction can't be predicted without data:
  - **MediaPipe run on the Kinect's RGB stream** is the only option compatible with the deepfake use case. It is plausibly similar to, or somewhat worse than, the phone. Resolution is probably not the main factor, because MediaPipe crops the person to 256×256 anyway. More likely factors are Kinect v1 image quality, the fixed 30 fps, and the loss of a subject-correlated frame-rate confound in the current data (§5, B10).
  - **Depth-derived metric 3D** could raise identity-verification scores, but mostly through **body size** (anthropometrics), not walking dynamics. Andersson & Araújo (AAAI 2015), using an Xbox 360 Kinect on 140 people, reported:

    | Features | Identification accuracy |
    |---|---|
    | Anthropometric only | 84.7–85.4% |
    | Gait only | 59.2–62.9% |

    Those gains are not available for RGB-only deepfake videos.
- **The Kinect v1 is a poor fit for this project's deepfake goal.** It is a reasonable, cheap *research instrument* for one question: how much the pipeline depends on MediaPipe's monocular z and on the recording setup. Its practical costs are high:
  - Windows 11 driver failures have been reported (§6.3).
  - The skeleton range is 0.8–4.0 m.
  - There are no heel or toe joints in its skeleton.
  - Depth is poor at the feet, which are the highest-attribution joints.
- **The existing evaluation has leakage and inflation problems** (§5). They must be fixed *in the comparison protocol*, or they can be as large as any sensor effect. The biggest:
  - enrollment templates contain the test sequences,
  - test data is normalized with its own statistics,
  - 66 duplicate sequences,
  - frame-inconsistent augmentation,
  - a threshold tuned on test scores,
  - face-swap clips that are copies of enrolled recordings.

### 1.2 Key decisions made in the audit session (reasons in §8)

| # | Decision |
|---|---|
| D1 | **Primary new model:** `gait-kinect-rgb-v1` = the **unchanged** MediaPipe 78-dim pipeline and unchanged architecture, run on the Kinect RGB stream (640×480, 30 fps). This is the Goal B deliverable. It is the only variant usable on RGB deepfakes. |
| D2 | **Record depth anyway** (plus the Kinect skeleton if the SDK works), so the depth arms (D5) can run later without re-recording. |
| D3 | **Two evaluation tiers.** (i) The *legacy* pipeline exactly as it exists (Goal B continuity). (ii) A new leak-free protocol **P2** (§10.3) for every scientific comparison. Never compare a P2 number to a legacy number. |
| D4 | **Record a phone at the same time as the Kinect** (same walks), plus a *phone-downscaled-to-Kinect* control. This separates sensor, resolution and depth effects. |
| D5 | **Depth arms on identical Kinect frames:** A1 MediaPipe xyz, A2 no-z, A3 MediaPipe xy + Kinect depth z, A4 fully metric 3D. The Kinect SDK skeleton (A5) is optional and last, because it has no heels or foot-index joints and needs SDK 1.8. |
| D6 | **Code changes:** small safety fixes to existing scripts, plus three new files. No rewrite of the model. The model code stays untouched (§8.4, §10). |
| D7 | **Separate trees** `data_kinect/` and `outputs_kinect/`. Returning subjects keep the exact same names as the phone dataset (§9). |

### 1.3 Environment facts established in the audit (read-only)

| Item | Finding (how checked) |
|---|---|
| Machine | Windows 11 Home 10.0.26200 (64-bit); i5-13420H; 15.7 GB RAM; NVIDIA RTX 3050 6 GB Laptop GPU; 137.5 GB free on C: (`Get-CimInstance`, `Get-PSDrive`) |
| Kinect | **No Kinect, adapter, SDK, OpenNI or libfreenect present or ever enumerated.** No `VID_045E` Kinect PIDs in `Get-PnpDevice`; no `C:\Program Files\Microsoft SDKs\Kinect`, `OpenNI*`, `libfreenect`; no matching uninstall entries. |
| Python | `venv/Scripts/python.exe`: Python 3.11.0, numpy 2.2.6, torch 2.10.0+cu130 (CUDA available), scikit-learn 1.8.0, opencv 4.12.0, mediapipe 0.10.32, albumentations 2.0.8, **tensorflow 2.20.0 (CPU-only build, no GPU visible)**, scipy 1.16.3 |
| **Package not installed** | `pip show gait-deepfake-detection` finds nothing in the venv. **`venv/Scripts/python.exe scripts/training/train.py --help` fails with `ModuleNotFoundError: No module named 'models'`.** Every script under `scripts/` needs `PYTHONPATH` set to the repo root (see §10.1). |
| ffmpeg | `C:\ffmpeg\bin\ffmpeg.exe` (2026-02-09 build), with `libx264` and `ffv1` encoders |
| MediaPipe model | `~/.mediapipe/models/pose_landmarker_lite.task` is cached. A MoveNet model is also cached (`~/.keras/models/movenet_thunder`), so MoveNet was run at some point. The shipped features are MediaPipe, as confirmed by the pickle shape `(60, 33, 3)`. |
| FaceFusion | **Not** at the README's `../facefusion`. Location unknown. |
| MediaPipe speed | Lite model, IMAGE mode, this CPU: **22.3 ms/frame at 1080×1920** and **19.6 ms/frame at a 640×480 letterbox**, 90/90 frames detected in both (`A2_S1.mp4`, 90 frames, scratch timing script). Decode time not included. |
| "Latest re-run" | The user said they recently re-ran the model. The newest model files anywhere under `C:\Arhaan`, Downloads, Documents, Desktop and OneDrive are `outputs/ablation/*_best.pth` from 2026-08-28 (the superseded `ablation_study.py`). The newest result is `outputs/ablation/ablation_loocv_results.json` (2026-08-28 19:48). **The main deployment checkpoints are all from 2026-02-08; the best is `outputs/checkpoints/checkpoint_epoch_46_best.pth`.** If a newer model exists elsewhere, ask for its path (§13.2). |

---

## 2. The existing pipeline, stage by stage (verified against the code)

**In plain terms, the pipeline does six things:**
1. Records walking videos.
2. Makes 15 altered copies of each.
3. Runs MediaPipe on every frame to get body landmarks.
4. Turns 12 of those landmarks into 78 numbers per frame, resampled to 60 time steps.
5. Averages each person's sequences into a "template".
6. Trains a network that takes a video sequence plus the claimed person's template and outputs P(same person). It is evaluated by leaving one person out at a time.

All commands assume the repo root as the working directory, PowerShell, and `$env:PYTHONPATH` set (§10.1). Line references are to commit `6c2e23f`.

### 2.0 Recording (no script)

- **Videos** (checked in `data/videos/` with OpenCV metadata, scratch script):
  - **66 originals, all 1080×1920 portrait H.264.**
  - Frame rates are mixed:
    - 30 fps: A2, Aarav, Ananya, Bharti, Devika, Prakhar, Vibhav.
    - ~59.9 fps: Arhaan, Prayag, Vedant, Vedant2.
    - Variable 43.7–59.8 fps: Som, Teja.
  - Clip length is 77–583 frames, **2.56–11.90 s** (median 5.85 s).
- **Naming:** `{Name}_{View}{Take}.mp4`.
  - View is `F` (frontal, walking toward or away from the camera) or `S` (side).
  - There are 5 takes per subject, 6 for Bharti.
  - The identity is **the text before the first underscore**, so names must not contain `_`.
- **Subject IDs:** `data/subject_map.json` maps the 13 names to "Subject 1..13" for the paper. Vedant and Vedant2 are **different people** (confirmed by the user).
- **Not recorded anywhere:** camera height, distance, lighting and trimming rules. **Unknown.**

### 2.1 Augmentation: `scripts/preprocessing/augment_videos.py`

- **Command:** `python scripts/preprocessing/augment_videos.py --input_dir data/videos --output_dir data/augmented_videos`
- **Input:** `*.mp4` and `*.avi`.
- **Output:**
  - A byte copy of every original (`shutil.copy2`).
  - 15 variants named `{stem}_{aug}.mp4`, written with OpenCV `mp4v` at the source fps and size:
    - 12 spatial: `hflip, bright_up, bright_down, contrast_up, rotate_left, rotate_right, blur, color_jitter, grayscale, combined_1, noise, zoom_in`.
    - 3 temporal: `slow` (0.8×, nearest-frame resampling), `fast` (1.2×), `reverse`.
  - That is 66 × 16 = 1,056 files, 25.8 GB on disk.
- **Skips outputs that already exist**, by filename.
- **Hard-coded crop:** `zoom_in` = `RandomScale(0.1–0.2)` → `CenterCrop(height=480, width=640)` → resize back to the source size. On 1080×1920 portrait that is an extreme, aspect-distorting zoom. On 640×480 Kinect frames it is a mild 1.1–1.2× zoom. **The same augmentation means different things on the two cameras.**
- **Defects** (measured, §5 B7–B9): random-parameter transforms are **not** consistent from frame to frame, and the copied originals become duplicate samples.

### 2.2 Feature extraction: `scripts/preprocessing/preprocess_videos.py` + `utils/pose_extraction.py`

- **Command:** `python scripts/preprocessing/preprocess_videos.py --videos_dir data/videos --augmented_dir data/augmented_videos --output data/gait_features/gait_features.pkl`
- **Discovery:** `os.walk` over both directories picks up **every** `.mp4/.avi/.mov/.mkv/.webm`, recursively.
- **Identity** = the first `_` token of the filename.
- **Backend (lines 110–139):** if `import tensorflow` succeeds and TensorFlow lists a GPU, the script **silently switches to MoveNet**. MoveNet gives 8 joints in 2D and 4 angles, so a 36-dim feature. On this machine TensorFlow sees no GPU, so MediaPipe is used.
- **Per frame:**
  - Converts BGR to RGB.
  - Runs MediaPipe Pose Landmarker **lite** in **IMAGE mode** (no temporal tracking), confidence 0.5.
  - Keeps the first person's 33 `pose_landmarks` (normalized) as (33, 3). **World landmarks (metres) are ignored.**
  - **Drops** frames with no detection and concatenates the rest, so time gaps collapse.
  - Rejects the clip if fewer than 10 valid frames remain.
- **What MediaPipe's coordinates mean** (MediaPipe docs, §15):
  - x = pixel x / image **width**; y = pixel y / image **height**.
  - z = "landmark depth, with the depth at the midpoint of the hips as the origin".
  - So one x unit ≠ one y unit unless the image is square.
- **Resampling:** the valid-frame sequence (12 to 583 frames here) is **linearly interpolated to exactly 60 steps**, with no padding and no anti-aliasing. Those 60 steps span 2.6–11.9 s depending on the clip.
- **Features**, computed *after* resampling on the (60, 33, 3) array. Order follows `utils/data_loader.py::_prepare_features`.

| Block | Dims | Index | Exact definition |
|---|---|---|---|
| Hip-centred coordinates | 36 | 0–35, joint-major [j0x, j0y, j0z, j1x, …] | 12 landmarks minus the mean of landmarks 23 and 24. Order: 11 L-shoulder, 12 R-shoulder, 23 L-hip, 24 R-hip, 25 L-knee, 26 R-knee, 27 L-ankle, 28 R-ankle, **29 L-heel, 30 R-heel, 31 L-foot-index, 32 R-foot-index** |
| Joint angles (°) | 6 | 36–41 | L-knee(23,25,27), R-knee(24,26,28), L-hip(11,23,25), R-hip(12,24,26), L-ankle(25,27,31), R-ankle(26,28,32). **Computed from x, y only**, in the anisotropic units above |
| Velocities | 36 | 42–77 | `np.diff` of the coordinates along the 60 steps; the **last** row is duplicated |

  - Unused extras are also stored: `accelerations`, `step_width`, `symmetry`.
  - Hip-centring removes translation only. **Scale is not normalized**, so apparent body size (camera distance, aspect) is inside the features.
  - The hip-midpoint z is ≈0 by MediaPipe's definition (measured mean 7.7e-5, SD 2.8e-4).
- **Output:** `dict[path -> {video_path, fps, total_frames, valid_frames, pose_sequence (60,33,3), gait_features, frame_indices, identity}]`, keyed by Windows-style paths.
  - The shipped file has 1,120 entries: 66 originals and 1,054 augmented.
  - Two augmented clips failed: `A2_F1_rotate_right` and `Arhaan_S3_zoom_in`.
  - **66 augmented-folder entries are bit-identical copies of the originals.**
  - Per identity: 84–85 entries, Bharti 102.
- **Resume:** if `--output` exists, it is loaded and new videos are **appended** to it.
- **Auto-enrolment at the end:** `from enroll_identities import …`.
  - This import fails unless `scripts/enrollment` is on `PYTHONPATH`. The features are already saved when it fails, but the process exits non-zero.
  - If the import works, the enrolled path is `args.output.replace("gait_features.pkl","enrolled_identities.pkl")`. **If `--output` is not named `gait_features.pkl`, the enrolment dict overwrites the features file.**

### 2.3 Enrolment (templates): `scripts/enrollment/enroll_identities.py`

- **Command that matches the shipped file:** `python scripts/enrollment/enroll_identities.py --from_features --features_file data/gait_features/gait_features.pkl --output data/gait_features/enrolled_identities.pkl`
- **Without `--from_features`** it defaults to re-extracting `data/videos` only, which gives different templates.
- **What `enroll_from_features` averages:** **every** entry of each identity (originals, augmentations and duplicates) into `avg_normalized_coords`, `avg_joint_angles` and `avg_velocities`.
  - Verified: `num_videos` is 84–102.
  - Verified: the template equals the mean of all sequences (A2 and Vedant2 checked).
  - **So every template contains the held-out subject's own test sequences** (§5 B1).

### 2.4 Deployment model training: `scripts/training/train.py` (+ `utils/data_loader.py`, `models/full_pipeline.py`)

- **Command:** `python scripts/training/train.py --features_file … --enrolled_file … --output_dir <DIR> --epochs 50`. **The default `--output_dir outputs` overwrites the existing checkpoints.**
- **Split:** 10 train / 3 val subjects.
  - The subject list comes from `list(set(...))` *before* the seeded shuffle, and set order depends on `PYTHONHASHSEED`. **So the split cannot be reproduced and was never logged.**
  - The only train log, `logs/train_20260209_013910.txt`, is an argparse usage dump.
- **Pairs:** every sequence gives one positive pair (its own template, label 1 = AUTHENTIC) and one negative pair (a `random.choice` of another identity's template, label 0). The validation set uses training statistics.
- **Architecture:** only the difference head decides. The config is saved to `<DIR>/model_config.json`; the shipped one is `outputs/model_config.json`.
  - The head: [V−C, |V−C|, V⊙C] (234 channels) → Conv1d (kernels 7, 5, 3; channels 64, 64, 32) → average-pool → MLP 32→32→2. That is 133,058 of the 848,614 parameters.
  - The CNN+BiLSTM+Transformer branch is computed but receives no gradient.
- **Training settings:**
  - AdamW, lr 1e-3, weight decay 1e-4, batch 16, cross-entropy with weights [1,1], gradient clipping at 1.0.
  - ReduceLROnPlateau and early stopping both watch **validation accuracy**.
  - A checkpoint is saved every epoch; `_best` when val accuracy improves. `feature_stats` is stored inside each checkpoint.
- **Shipped checkpoint:** `checkpoint_epoch_46_best.pth` (`epoch` field 45), train 98.38%, val 93.92%.
  - Epoch 50 reaches only 90.20% val.
  - Earlier bests: epochs 1, 2, 4, 8, 15, 26, 27, 30, 40.
  - Dated 2026-02-08.

### 2.5 LOOCV evaluation, source of the headline numbers: `scripts/evaluation/evaluate.py --loocv`

- **Command:** `python scripts/evaluation/evaluate.py --loocv --loocv_epochs N --lr 1e-4 --features_file … --enrolled_file … --output_dir <DIR>`. It writes `<DIR>/loocv/loocv_results.json`. **The default `<DIR>` is `outputs/evaluation`, which overwrites the headline file.**
- **Each of the 13 folds:**
  - Trains a fresh full model on the 12 other subjects (all of their sequences, augmentations included).
  - Keeps the state with the **lowest training loss**.
  - Tests on the held-out subject: 1 positive and 1 random negative per sequence, 168–204 pairs.
  - **Normalizes the test set with the held-out subject's own statistics** (no `feature_stats` passed, lines 339–344).
  - Uses templates from the global `enrolled_identities.pkl`, which includes the held-out subject.
  - Reads the config from the hard-coded path `outputs/model_config.json` (line 779).
- **The headline run used 50 epochs per fold, not 30.** `logs/evaluate_20260209_001725.txt` line 22 says "Epochs per fold: 50", and its per-fold table and "1228.3s total" match `loocv_results.json` exactly.
  - It was produced by an older, root-level `evaluate.py`: the log cites `DeepFake-Detection\evaluate.py:120`.
  - The learning rate was not logged.
- **Metrics:**
  - Accuracy/F1/precision/recall at argmax (≡ 0.5), with positive = AUTHENTIC.
  - ROC-AUC on P(authentic).
  - EER = (FPR+FNR)/2 at the ROC point with the smallest |FPR−FNR|.
  - "Mean ± std" uses the population std (ddof=0) over the 13 folds.
  - "Pooled" is one ROC over 2,240 scores from 13 *different* models.

### 2.6 Definitive ablation: `scripts/evaluation/ablation_loocv.py`

- **Command:** `python scripts/evaluation/ablation_loocv.py --features_file … --enrolled_file … --output <FILE>`. It renames any existing `<FILE>` to `*.bak.json` before writing.
- **Protocol:**
  - Same folds as §2.5, but test data **is normalized with training statistics**.
  - lr 1e-3, 30 epochs, seeds 0/1/2. torch and numpy are seeded; Python `random`, which draws the negatives, is **not**.
- **Arms:** Raw (= the deployed head), Raw+CNN, Raw+BiLSTM, Raw+Transformer, Raw+Hybrid, Hybrid-only, and one Raw arm with legacy normalization.
- **Hard-coded 78:** `RawEncoder.out_dim`, `raw_dim`, and the encoders' `input_dim`.
- **Cost:** 247 runs took 4 h 25 min 57 s (`logs/ablation_loocv_run.log`, UTF-16). Raw runs took about 27–59 s each.
- **Superseded, do not use:** `ablation_study.py` (v1/v2) and `_verify_ablation_fix.py`.

### 2.7 Explainability: `scripts/evaluation/run_gradcam.py` + `utils/gradcam.py`

- **Command:** `python scripts/evaluation/run_gradcam.py --checkpoint <ckpt> --enrolled_file … --features_file … --output_dir <DIR> --aggregate --n_samples 30`. Without `--checkpoint` it auto-picks from `outputs/checkpoints`, i.e. the phone model.
- **Aggregate mode:**
  - Matches keys with `stem.rsplit('_',1)[0]`, so only un-augmented clips match.
  - Uses 2 per person, which is where the "26 samples" comes from.
  - Pairs each clip with **its own** template.
  - Normalizes with the checkpoint's `feature_stats`.
- **Attribution method:** gradient × input, |x·∂logit/∂x|.
  - Joint score = coordinate + velocity attribution.
  - Groups: coords 0–35, angles 36–41, velocities 42–77.
  - Hard-coded to 12 joints, 6 angles, 78 dims.
- **Single-video mode is broken:** it calls `extract_from_video`, which does not exist.

### 2.8 Inference: `scripts/inference/inference.py`

- **Command:** `python scripts/inference/inference.py --video V --claimed_identity NAME --checkpoint C --enrolled_file E --threshold T`.
- **Phone defaults:** the latest `*_best.pth` in `outputs/checkpoints`, the phone templates, and **T = 0.7737**.
- **Verdict:**
  - `similarity > T` → AUTHENTIC.
  - Otherwise every identity is scored. If the best is above T → **IDENTITY_MISMATCH** (the printed message says "No deepfake detected").
  - Otherwise → SUSPECTED_DEEPFAKE.

### 2.9 Face-swap clips and the gait-preservation check

- **How the clips were made:** FaceFusion GUI, `inswapper_128_fp16`, face-only mask, saved as `data/deepfake/{Body}_body_{Face}_face.mp4`. FaceFusion is not on this machine.
- **The three clips are face-swaps of enrolled recordings.** Frame count, fps and valid-frame count all match:

  | Clip | Frames @ fps | Source recording |
  |---|---|---|
  | `Devika_body_Ananya_face` | 135 @ 30.01 | Devika_F1 |
  | `Prakhar_body_Vedant_face` | 149 @ 30.06 | Prakhar_F2 |
  | `Prayag_body_Bharti_face` | 327 @ 59.93 | Prayag_F1 |

- **An unused fourth file:** `WhatsApp Video 2026-02-13 at 09.09.43.mp4` (720×1280, 24 fps).
- **`verify_gait_preservation.py`** takes no arguments and always compares against `{Body}_F1`. It has two bugs (§5 B20).
- **`outputs/evaluation/deepfake_test/faceswap_validation_results.json`** (checkpoint epoch 46, τ = 0.7737) **has no generating script in the repo.**

### 2.10 Figures

- `scripts/generate_figures/*.py` reads phone paths from `figstyle.py` and **writes into `figures/`**, overwriting the paper figures.
- `generate_paper_figures.py` writes to `outputs/paper_figures/`.
- `visualize_keypoints.py` writes to `outputs/keypoint_visualizations/`.
- **Do not run any of these for Kinect data** without redirecting the output (§9.3).

### 2.11 Orchestrator: `scripts/run_pipeline.py`

Its evaluate and demo stages look for `checkpoint_epoch_best.pth`, **a name `train.py` never writes**, so those stages always fail. Run the stages one by one instead.

### 2.12 What is phone- or RGB-specific, or breaks or degrades silently with Kinect input

| Item | Effect with Kinect input |
|---|---|
| Normalized x/y depend on the aspect ratio (phone 9:16 vs Kinect 4:3) | Coordinate, velocity and angle values are not comparable across cameras. **Enrolling on the phone and probing on the Kinect fails silently.** Within one camera the values are consistent. |
| No scale normalization | Apparent body size is part of the features. The Kinect needs ≥ ~2.7 m for full-body framing (§6.2), so people appear smaller than in the phone videos. |
| Whole clip resampled to 60 steps | The time per step depends on clip duration. Kinect walk durations are set by the range limits (§9.4), which changes velocities. |
| Phone data has mixed fps; the Kinect is fixed at 30 | Removes a subject-correlated nuisance (§5 B10), which alone could lower scores. |
| `zoom_in` crop fixed at 640×480 | Mild on the Kinect, extreme on the phone. |
| MediaPipe's landmarker takes a 256×256 crop (MediaPipe docs) | Resolution above a ~256 px person height is mostly discarded, so raw resolution is a weaker confound than it looks. People far away in frontal walks can drop below 256 px (**estimate:** ~180 px at 6 m). |
| Colour order | Pipeline input is OpenCV BGR from a video file. When writing Kinect frames yourself, an RGB/BGR swap silently changes MediaPipe's input. |
| Partial bodies at the frame edges | MediaPipe hallucinates landmarks instead of failing, so those frames are not dropped. **Trim every take** (§9.4). |
| Low light and sensor noise | The phone `noise` augmentation cut MediaPipe detection to 51% of frames on average (minimum 17%). **Measure the detection rate on Kinect footage.** |
| TensorFlow-with-GPU switches to MoveNet | Would silently give 36-dim features on a Linux/CUDA machine. |
| Pickle keys contain `\\` | Name parsing via `Path(...).stem` breaks if a Windows-made pickle is read on Linux. |

---

## 3. Every reported number, and exactly how it was computed

All values below were recomputed in the audit from the JSON files with scikit-learn 1.8.0, and they match the files.

### 3.1 What is a "sample", a "pair", a "deepfake"

- **Sample:** one 60×78 sequence from one video file (an original, an augmentation, or a duplicate copy). There are 1,120 in total.
- **Pair:** (sample, claimed identity's template).
  - **Positive:** the sample's own identity.
  - **Negative:** one uniformly random *other* enrolled identity. It is re-drawn on every pass, unseeded.
- **"DEEPFAKE" label in LOOCV = identity mismatch between a real video and another real person's template.** No synthesized video enters LOOCV. The only synthetic videos are the 3 face-swap clips (§3.5).

### 3.2 Headline LOOCV: `outputs/evaluation/loocv/loocv_results.json`

These numbers come from `evaluate.py --loocv`: 50 epochs per fold, legacy normalization, templates that include the test data, and unseeded negatives.

| Metric | Per-fold mean ± std (ddof=0) | Pooled (2,240 scores) |
|---|---|---|
| ROC-AUC | 95.10 ± 3.08 (ddof=1: ± 3.21) | **94.95** |
| Accuracy @0.5 | 87.04 ± 3.65 | 87.01 |
| F1 @0.5 | 87.12 ± 3.77 | 87.15 |
| Precision @0.5 | 86.51 ± 4.72 | 86.20 |
| Recall @0.5 | 88.23 ± 6.82 | 88.12 |
| EER | 12.27 ± 3.80 | 12.77 |

- **Confusion matrix @0.5, pooled:** TN 962, FP 158, FN 133, TP 987.
- **Youden threshold on the pooled test ROC:** τ\* = 0.77369, with TPR 83.57%, FPR 8.48%, J 0.7509.
- **Confusion matrix @τ\*:** TN 1025, FP 95, FN 185, TP 935; accuracy 87.50%.
- **Total runtime:** 1,228.3 s.
- **Per-fold results** (AUC / accuracy / n pairs):

  | Subject | AUC | Accuracy | n |
  |---|---|---|---|
  | A2 | .982 | .893 | 168 |
  | Aarav | .950 | .877 | 170 |
  | Ananya | .955 | .882 | 170 |
  | Arhaan | .909 | .821 | 168 |
  | Bharti | .926 | .848 | 204 |
  | Devika | .967 | .865 | 170 |
  | Prakhar | .957 | .882 | 170 |
  | Prayag | .966 | .882 | 170 |
  | Som | .957 | .865 | 170 |
  | Teja | .972 | .900 | 170 |
  | Vedant | .966 | .900 | 170 |
  | **Vedant2** | **.870** | **.777** | 170 |
  | Vibhav | .985 | .924 | 170 |

- **Why per-fold and pooled differ:** each fold is a *different model*. Pooling forces one global ranking across models whose score scales differ, and folds have unequal n.
- **Why accuracy/F1 and τ\* shouldn't be mixed:** they are different operating points (0.5 vs 0.7737).
- **τ\* is optimistic:** it was chosen on the same test scores it is reported on.

### 3.3 Definitive ablation: `outputs/ablation/ablation_loocv_results.json` (2026-08-28)

Protocol: 13 folds × seeds {0, 1, 2}, normalized with training statistics, lr 1e-3, 30 epochs.

| Variant | Params | Per-fold AUC, seed 0 / 1 / 2 | Pooled AUC, seed 0 / 1 / 2 | Seed-averaged per-fold mean |
|---|---|---|---|---|
| Raw (deployed) | 133,058 | 94.67 / 94.00 / 94.09 | 94.30 / 93.50 / 93.99 | **94.25** |
| Raw + CNN | 434,690 | 91.86 / 90.71 / 92.59 | 91.32 / 90.31 / 91.59 | 91.72 |
| Raw + BiLSTM | 379,074 | 87.71 / 88.14 / 89.12 | 87.60 / 88.23 / 88.82 | 88.32 |
| Raw + Transformer | 712,002 | 89.30 / 91.80 / 90.00 | 88.57 / 90.93 / 89.20 | 90.37 |
| Raw + Hybrid | 980,994 | 91.15 / 86.52 / 92.41 | 90.22 / 86.97 / 91.27 | 90.03 |
| Hybrid only | 876,162 | 59.58 / 47.34 / 45.18 | 58.61 / 47.36 / 45.21 | 50.70 |
| Raw, **legacy** normalization (seed 0) | 133,058 | 94.97 ± 2.90 | 94.83 | — |

- **Legacy minus train normalization, seed 0:**
  - +0.30 per-fold AUC (range −4.98 to +5.26 by subject; paired t, n = 13: p = 0.73).
  - **+0.53 pooled** (94.83 vs 94.30).
- **Significance re-tested with the subject as the unit** (seed-averaged, n = 13). The paper used n = 39:

  | Variant vs Raw | Δ AUC | t-test, n=39 (paper) | t-test, n=13 | Wilcoxon, n=13 | Seed-runs where the variant beat Raw (of 39) |
  |---|---|---|---|---|---|
  | +CNN | −2.53 | 2.9e-4 | 2.6e-3 | 2.4e-3 | 9 |
  | +Transformer | −3.88 | 3.5e-5 | 5.8e-5 | 4.9e-4 | 7 |
  | +Hybrid | −4.22 | 3.7e-4 | 4.8e-3 | 8.1e-3 | 9 |
  | +BiLSTM | −5.93 | 1.7e-7 | 6.6e-4 | 2.4e-4 | 5 |
  | Hybrid only | −43.55 | 1.1e-15 | 2.0e-8 | 2.4e-4 | 0 |

  **The conclusions hold, but the paper's p-values are overstated by pseudo-replication.** Comparing the n=39 t-test with the n=13 t-test, the overstatement is about 1.7× for Transformer, 9× for CNN, 13× for Hybrid, ~3,900× for BiLSTM, and orders of magnitude more for Hybrid-only.
- **Seed noise (used for power in §11.5):** for the Raw model, the per-subject AUC difference between two seeds has SD 3.09 points; the largest single-subject spread across seeds is 7.82 points.
- **Why the ablation's Raw number (94.25) differs from the headline (95.10):** the two runs differ in learning rate (1e-3 vs an unlogged value; the current `evaluate.py` default is 1e-4), epochs (30 vs 50), normalization (train vs legacy), and in the full model vs head only. **They are not the same experiment.**

### 3.4 Superseded ablations

- **`outputs/ablation/ablation_results.json`** (v2, embedding comparison, a single split): AUC CNN 65.84, LSTM 40.40, Transformer 56.53, Hybrid 78.80.
- **`ablation_results_v1_shared_head_DEPRECATED.json`:** accuracy CNN 88.93, LSTM 89.33, Transformer 90.51, Hybrid 90.32; AUC 96.40 / 96.10 / 96.89 / 96.11.
  - In v1 all four "variants" were the same diff head (the named branch never reached the logits), so the spread is training noise.
  - **`DOCUMENTATION/context.md`, `PROJECT_TECHNICAL_OVERVIEW.md` and `outputs/paper_figures/all_tables.txt` Table 4 still quote v1.**

### 3.5 Face-swap validation: `outputs/evaluation/deepfake_test/faceswap_validation_results.json`

Model: checkpoint epoch 46, τ = 0.7737.

| Clip | P(auth \| claimed face identity) | Best match | Its score | Other scores above τ |
|---|---|---|---|---|
| Devika body / Ananya face | 0.00569 | Devika | 0.99986 | none |
| Prakhar body / Vedant face | 0.00031 | Prakhar | 0.99997 | none |
| Prayag body / Bharti face | 3.9e-7 | Prayag | 0.999992 | **Teja 0.99986** |

- All three got the verdict **IDENTITY_MISMATCH**, not "SUSPECTED DEEPFAKE".
- The paper discloses the Teja near-tie; the README does not.
- **Had the clip claimed "Teja", the system would have said AUTHENTIC.**
- All three bodies are enrolled recordings (§2.9), so the high match to the body source is expected (§5 B6).

### 3.6 Explainability: `outputs/gradcam/aggregate/gradcam_results.json` (26 samples, checkpoint epoch 46)

- **Joints (normalized to max = 1):**
  - L-shoulder 1.000, R-heel 0.940, L-foot 0.931, L-knee 0.897, R-shoulder 0.873, R-ankle 0.810.
  - R-foot 0.766, R-knee 0.726, L-heel 0.709, L-ankle 0.673, L-hip 0.665, R-hip 0.566.
- **Angles:** L-ankle 1.000, L-knee 0.896, R-hip 0.597, L-hip 0.571, R-ankle 0.486, R-knee 0.451.
- **Groups:** coords 47.68%, velocities 37.40%, angles 14.92%. That is 1.03×, 0.81× and 1.94× of their share of the 78 dimensions.
- **Caveats:**
  - Only 2 clips per person, each paired with a template that contains that clip.
  - The model's training subjects are unknown.
  - **There is no per-axis (x/y/z) split, so the audit can't say how much the model uses z.**
- **Heels and feet rank near the top, and these are exactly the joints where Kinect depth is weakest (§6.5).**

### 3.7 Other numbers

- **Parameters:** 848,614 total, of which 133,058 are on the decision path (131,936 conv + 1,122 MLP). The state dict also holds a 64,000-value positional-encoding buffer that is not a parameter.
- **Deployment training:** val accuracy 93.92% / train 98.38% at epoch 46.
- **`outputs/paper_figures/all_tables.txt` errors:**
  - Table 1 is titled "Pooled" but lists per-fold means.
  - Table 4 is the deprecated v1 ablation.

---

## 4. Documentation claims vs code and results

Every markdown and text file was treated as a set of claims to check. ✓ = verified. ✗ = contradicted, with the evidence.

### 4.1 `README.md`

- ✓ Every pooled and per-fold number, τ\*, the ablation table and the face-swap table.
- ✗ "normalization statistics are recomputed from the remaining 12 subjects". False for the headline numbers (§2.5).
- ✗ Step 5 checkpoint `checkpoint_epoch_best.pth` does not exist.
- ✗ Step 6 `--loocv_epochs 30`. The headline run used 50.
- ✗ Ablation p-values use n = 39 (§3.3).
- ✗ The Teja near-tie is omitted from the "3/3".
- ✗ "pip install -e ." is presented as done. It is not installed in the venv.

### 4.2 `paper/deepfake_paper.tex`

- ✓ Tables for LOOCV, per-fold, ablation, configuration and attribution.
- ✓ Deployment checkpoint 93.92 / 98.38 at epoch 46.
- ✓ The legacy-vs-train disclosure (§IV-C) and the Teja caveat.
- ✗ "66 original recordings at 30 fps and 720p–1080p". All are 1080×1920, and 36 of 66 run at 44–60 fps.
- ✗ "3–5 per subject". The actual count is 5–6.
- ✗ "T = 60 frames (≈2 s)" and "shorter ones zero-padded". The 60 steps span 2.6–11.9 s, and the code interpolates rather than pads.
- ✗ "Photometric operations change pixels but not the pose MediaPipe recovers". Detection drops to 51% of frames (noise) and 91% (grayscale), and rotations jitter from frame to frame (§5 B7–B8).
- ✗ Hip-centring "removes … much of the scale variation". It removes translation only.
- ✗ Angles are "scale-free", with a 3D formula. In the code they are 2D, on anisotropic image units.
- ✗ "v₁ = 0". The code duplicates the last velocity row instead.
- ✗ Algorithm 2 and §V-A: statistics from the "training split only", and "re-enrol the 12 training identities". Neither is in the code: templates are global and include test data.
- ✗ §IV-D: enrolment from "original (non-augmented) recordings". It averages all 84–102.
- ✗ LOOCV "30 epochs". The headline run used 50.
- ✗ "CUDA 12.4". The log says CUDA 13.0.
- ✗ Paired t-tests with n = 39 (pseudo-replication).
- ✗ **Missing disclosure:** the 3 face-swap bodies are enrolled recordings (§5 B6).
- ✗ **Missing disclosure:** 24 of the 78 dimensions are MediaPipe's *monocular* z-estimate.
- ✗ **Missing disclosure:** a subject-correlated frame-rate and duration confound (§5 B10).
- ✗ "The threshold is derived from data, not assigned". It was derived from pooled *test* scores of different models and then applied to the deployment model.

### 4.3 `DOCUMENTATION/DATASET.txt` (IEEE DataPort abstract)

- ✗ "AUC-ROC 94.95% ± 2.81%" and "F1 86.56%". Neither value appears in any results file (the file has ± 3.08; F1 is 87.12 / 87.15).
- ✗ Augmentations "temporal jitter, Gaussian noise on keypoints, occlusion simulation". None is used; the noise is pixel noise.
- ✗ "~65 raw videos". There are 66.
- ✗ The pickle holds "(sequence, label, subject_id) tuples". It is a dict of dicts keyed by path.
- ✗ "60 frames ≈ 2 gait cycles at 30 fps". Frame rates are mixed and clips are 2.6–11.9 s.
- ✗ "CNN+BiLSTM+Transformer hybrid … achieving". The decision comes from the 133k raw-difference head.

### 4.4 `DOCUMENTATION/context.md`

- ✗ Accuracy 87.27 ± 3.76, F1 86.56 ± 4.56, EER 13.19 ± 4.21, AUC ± 2.81. No results file contains these.
- ✗ "Accuracy 87.04% (pooled)". 87.04 is the per-fold mean; the pooled value is 87.01.
- ✗ Ablation CNN 88.93 and the rest come from the **deprecated v1**.
- ✗ "Ablation proving each component's contribution". The definitive result shows the opposite.
- ✗ "4 real face-swap test videos". 3 were evaluated.
- ✗ "Solo project". The paper and dataset list 4 authors.
- ✗ "1,056 training videos". 1,054 were extracted.
- ✓ CM 987/962/158/133 (at τ = 0.5).
- ✓ Runtime 1,228 s.

### 4.5 `DOCUMENTATION/PROJECT_TECHNICAL_OVERVIEW.md`

- ✗ The signature is "the average of original (non-augmented) clips". It is all 84–102 sequences.
- ✗ Augmentations "applied identically across every frame". Measured false (§5 B7).
- ✗ The results and ablation tables are the same wrong or deprecated numbers as `context.md`.
- ✗ "The embedding is used for the similarity score". The similarity is P(authentic) from the difference head.
- ✗ "~a few M params". There are 848,614.

### 4.6 Other documentation files

- **`.github/instructions/rules.instructions.md`:**
  - It points to `PLAN.md`, which does not exist.
  - It calls `ablation_study.py` the ablation; that script is superseded.
- **`CONTRIBUTING.md` and `tests/smoke_test.py`:** both point to `AUDIT_FINDINGS.md`, and `NOTES.md` is cited in code. Neither file exists.
- **`ieee_scripts/quickstart.py`:** points to `DATASET_INSTRUCTIONS.md`, which does not exist.
- **`figures/README.md`:** says fig 9 reads `ablation_results.json`. The script actually reads `ablation_loocv_results.json`.
- **`research/claude_research_verification.md`:** a prior fact-check of the paper's citations. It was not re-verified here, except that it quotes the DataPort "± 2.81" as matching the draft; that number is itself unsupported (§4.3).

---

## 5. Bugs, leakage and inflated or unsupported results (reported prominently)

Ordered by how much each should change how the results are read. "Measured" means verified in this audit.

### Leakage and inflation

| ID | Problem | Evidence | Effect |
|---|---|---|---|
| **B1** | **Enrolment templates include the test data.** Every template is the mean of all 84–102 sequences of that person, *including the held-out subject's test sequences*. Every positive LOOCV pair compares a sequence against an average that contains that sequence and its 15 siblings. | `enroll_identities.py::enroll_from_features`; template equals the mean of all sequences (measured) | Inflates positive-pair similarity. **Size unmeasured**; protocol P2 (§10.3) measures it. |
| **B2** | Test data normalized with the held-out subject's own statistics | `evaluate.py` lines 339–344; ablation legacy arm | Measured: +0.30 per-fold AUC, +0.53 pooled. Small. |
| **B3** | 66 bit-identical duplicate sequences (5.9%): each original is extracted from both `data/videos` and its copy in `data/augmented_videos` | Measured with `np.array_equal` | Double-weighted in templates and double-counted in test pairs |
| **B4** | The headline LOOCV used **50** epochs per fold, not the documented 30. It ran on an older script, and its learning rate was not logged. | `logs/evaluate_20260209_001725.txt` line 22 | The headline config is not exactly reproducible from current code |
| **B5** | τ\* = 0.7737 was tuned on the pooled *test* scores of the 13 LOOCV models, then used as the deployment threshold for a *different* model (train.py, lr 1e-3, unknown split) and for the face-swap test | §3.2, §2.8 | Optimistic operating point that doesn't transfer. **Must be re-derived for any new model.** |
| **B6** | **The face-swap test clips are face-swaps of enrolled recordings** (Devika_F1, Prakhar_F2, Prayag_F1). Those exact walks, plus 15 augmentations each, are in the templates, and possibly in the deployment model's training split (unrecoverable). | Frame count, fps and valid frames match exactly (§2.9); not verified pose-by-pose | "Matched true body source at 0.9999" is close to guaranteed. It shows face-swapping doesn't disturb pose extraction; **it does not show gait generalizes to unseen footage.** The paper doesn't disclose this. |
| **B12** | Pseudo-replication: n = 39 seed×fold "paired" tests | §3.3 | p-values overstated (from ~2× to ~3,900× across the four Raw+X arms); conclusions hold |
| **B13** | Negative pairs use unseeded Python `random` in `evaluate.py` and `ablation_loocv.py` | Code | Results aren't bit-reproducible; "paired" arms get different negatives |
| **B26** | **Training negatives are drawn from *all* enrolled identities**, including the held-out subject (`utils/data_loader.py` line 221, comment: "Use all enrolled identities"). During LOOCV training the held-out subject's template, which is built from its test sequences (B1), is shown as a "wrong identity" for training videos. | Code | A further leak of test-derived information into training; size unmeasured. P2 draws training negatives from training subjects only. |
| **B14** | The deployment model's train/val split can't be recovered (set order + `PYTHONHASHSEED`, never logged). The 3 validation subjects drive LR scheduling, early stopping *and* checkpoint selection. | `utils/data_loader.py` lines 301–303 | Its 93.92% val accuracy is selection-biased. Nobody knows which subjects the deployed model saw. |

### Data and feature defects

| ID | Problem | Evidence | Effect |
|---|---|---|---|
| **B7** | **Augmentation is not frame-consistent.** With albumentations 2.0.8, `random.setstate` doesn't control albumentations' own RNG, so rotation, brightness, blur, colour and noise are re-drawn every frame. | Measured on `A2_F1_rotate_left.mp4`: frame-to-frame rotation up to **±3.4°** (original ±0.03°). `A2_F1_bright_up` brightness swings 179.5–199.8 over 12 frames. Synthetic check confirms. | Injects fake joint motion into velocities (rotations) and flicker. The docs claim the opposite. |
| **B8** | Photometric augmentations break pose detection | Share of frames detected, by augmentation (measured): noise mean 51% (min 17%), zoom_in 36% (min 3.5%), grayscale 91%, bright_down 92%, original 98.5% | Undetected frames are dropped and time is collapsed, so these sequences are temporally distorted |
| **B9** | `zoom_in` crops 480×640 from 1080×1920 and stretches it back | Code; the file is 1080×1920 | Extreme anisotropic zoom; the person is mostly out of frame |
| **B10** | **Frame rate and clip duration are confounded with subject.** 30 fps for 7 subjects, ~60 fps for 4, variable 44–60 for 2. Durations run 2.6–11.9 s. Resampling to 60 steps makes velocity scale and time per step depend on recording conditions. | §2.0 | A possible shortcut the model can exploit. A fixed-fps sensor (Kinect) removes it, and **that alone could lower scores.** |
| **B11** | Joint angles are 2D on anisotropic normalized coordinates (x/W, y/H) | `pose_extraction.py` line 185 | Not true angles; they depend on aspect ratio, so they aren't comparable across cameras |
| **B23** | `hflip` swaps anatomical left and right. `reverse` makes a backwards walk. Both are claimed to be gait-preserving. | Code | Contested as identity-preserving augmentations; left unchanged in §8 for comparability |

### Tooling defects

| ID | Problem | Evidence | Effect |
|---|---|---|---|
| **B15** | The package isn't installed. Scripts fail with `ModuleNotFoundError` unless `PYTHONPATH` is set. | Measured | Nothing runs as documented |
| **B16** | `run_pipeline.py` evaluate/demo stages and README step 5 look for `checkpoint_epoch_best.pth` | Code | Those stages always fail |
| **B17** | The auto-enrol import in `preprocess_videos.py` fails, or, if it works with a non-standard `--output` name, **overwrites the features file** | Code (§2.2) | Data loss risk |
| **B18** | Silent MoveNet switch when TensorFlow sees a GPU | Code | Silent 36-dim features |
| **B19** | `run_gradcam.py` single-video mode calls a method that doesn't exist | Code | Crash |
| **B20** | `verify_gait_preservation.py`: (a) "torso height" uses gait indices 0,1 and 6,7, which are **shoulders and ankles**, not hips and shoulders, so the PCK tolerance is ~2–3× too loose (estimate); (b) it always compares against `{Body}_F1`, but `Prakhar_body_Vedant_face` came from Prakhar_**F2**, a different walk | Code | The preservation "check" is unreliable. No saved output exists. |
| **B21** | `faceswap_validation_results.json` has no generating script | grep | Not reproducible |
| **B24** | `evaluate.py` (non-LOOCV mode) rebuilds its own 80/20 split instead of the checkpoint's | Code | Test subjects may have been training subjects |

### Semantics

| ID | Problem | Evidence | Effect |
|---|---|---|---|
| **B25** | The face-swap verdict was "IDENTITY_MISMATCH: No deepfake detected" | `inference.py` lines 352–368 | By the system's own 3-way semantics it cannot tell a face-swap of an enrolled body from a real video with a wrong claim. It "catches" swaps only as mismatches. This is inherent to the threat model; the wording is misleading. |

---

## 6. The Kinect: what the device is and what it can do

Tags: **[P]** = checked against a primary source in this session. **[S]** = secondary source only. **[E]** = my own estimate or derivation. Sources are in §15.

### 6.1 Which device you have

- **"Xbox 360 Kinect" is the original Kinect (v1), a structured-light sensor.**
  - Andersson & Araújo used "the 2010 model for the X-Box 360 video-game console, connected through an adapter cable to a PC running the SDK version 1.0" [P, S10].
  - Model numbers:
    - Xbox 360 Kinect = **1414** (2010) and a later revision **1473**.
    - Kinect for Windows v1 = **1517** [S, S5].
  - Some software reportedly handles 1473 and 1517 less reliably than 1414 [S, S5].
- **Not the same device:** the Xbox One Kinect (v2) is time-of-flight, with different specs and a different driver (`libfreenect2`) [P, S12].
- **How to confirm your unit:**
  - The label under the base gives the model number.
  - An Xbox 360 unit has a motorised tilt base and a **proprietary USB-like plug** (built for the Xbox 360 S AUX port).
- **Connecting to a PC needs the "Kinect power supply / USB adapter".** It supplies 12 V (~1.08 A output on common units) and has a USB-A data plug. Microsoft part numbers 1429 and 1432 are cited [S, S6]. **Without it the sensor won't work on a PC.**

### 6.2 Specifications

| Property | Value | Tag |
|---|---|---|
| Field of view | 57° horizontal × 43° vertical | [P, S1] |
| Tilt motor | ±27° | [P, S1] |
| Colour and depth stream rate | 30 fps | [P, S1] |
| Colour sensor | 1280×960; streams at 640×480 @ 30 fps; 1280×960 @ up to 12 fps | 1280×960 sensor [P, S1]; 640×480 RGB used by libfreenect's recorder [P, S8]; 12 fps mode [S] |
| Depth stream | 640×480, 320×240 or 80×60, all @ 30 fps; 11-bit disparity | [P, S2, S7] |
| Depth range (SDK, default mode) | 0.8–4.0 m; skeleton "practical range" 1.2–3.5 m | [P, S3] |
| Near mode | 0.4/0.5–3.0 m. **"If you are using an Xbox Kinect with the Kinect for Windows SDK then Near Mode is not supported"** | [P, S4] |
| Depth random error | "a few millimeters up to about 4 cm at the maximum range", growing with distance². Depth point spacing: ~2 mm at 1 m, ~2.5 cm at 3 m, ~7 cm at 5 m. Recommended 1–3 m for mapping. | [P, S7] |
| Skeleton (SDK) | 20 joints: HIP_CENTER, SPINE, SHOULDER_CENTER, HEAD, and SHOULDER, ELBOW, WRIST, HAND, HIP, KNEE, ANKLE, **FOOT** on each side. **No heel and no separate toe/foot-index.** 2 people tracked, up to 6 detected; coordinates in metres. | [P, S9, S3, S4] |
| Skeleton caveat | "optimized to recognize users … facing the Kinect; **sideways poses provide some challenges**". Half of this dataset is side view. | [P, S3] |

**Framing geometry** [E, from the 43°/57° FOV]:
- To fit a 1.8 m person plus margin (~2.1 m vertical), the subject must be at least 2.1 / (2·tan 21.5°) ≈ **2.7 m** away.
- At 3.5 m the frame covers ~2.8 m vertically and ~3.8 m horizontally. That is about 174 px/m, so a 1.75 m person is ~305 px tall.
- By comparison, phone subjects are about 0.28 of a 1920 px frame from shoulder to ankle (median, measured), roughly 680 px for the whole body [E].
- **Frontal walks:** a full body with valid depth exists only between ~2.7 m and 4.0 m, which is about one stride.
- **Side walks at 3.5 m:** about 3.8 m of path, roughly 2.5 gait cycles.

### 6.3 Getting data off it today

- **Kinect for Windows SDK v1.8** (`KinectSDK-v1.8-Setup.exe`, 222.4 MB, v1.8.0.595, still on Microsoft's Download Center) [P, S11]:
  - Supported OS: Windows 7, 8, 8.1 and Embedded Standard 7. **Windows 10 and 11 are not listed.**
  - Needs a dedicated USB 2.0 bus, VS 2010/2012 for development, and .NET 4/4.5.
  - The Developer Toolkit v1.8 is a separate download (id 40276) [P, S11].
  - **Windows 11:** a Microsoft Q&A thread (2022–2025, unresolved) reports a Kinect v1 recognised only as an audio device. Replies say it is "100% Windows 11-related", and **downgrading to Windows 10 fixed it** [P, S13]. **This machine is Windows 11 build 26200, so expect this failure.**
  - Using the SDK with an Xbox 360 unit works for development (Andersson did it with SDK 1.0), but Near Mode is unavailable [P, S4, S10]. Licensing for commercial use with an Xbox unit was not verified.
- **libfreenect (OpenKinect)** [P, S8]:
  - Runs on Linux, macOS and Windows. Model 1414 is supported; "newer Kinect models may require audio firmware for motor and LED support" (relevant if yours is a 1473).
  - Windows needs libusb ≥ 1.0.22, plus Zadig to install the **libusbK** driver for each device, plus a CMake build.
  - The Python wrappers are "not guaranteed to be API stable".
  - The **`fakenect` `record` tool** writes 640×480 RGB as `r-<time>-<devtimestamp>.ppm`, 11-bit depth as `d-….pgm` (16-bit container), accelerometer dumps and an `index.txt`, with the device timestamp for every frame [P, S8]. That is a complete recorder with no custom capture code.
  - `FREENECT_DEPTH_REGISTERED` = depth in mm aligned to the 640×480 RGB image [S, S8]. **Whether the `record` tool saves registration parameters, and whether the Python wrapper exposes `DEPTH_REGISTERED`, is unverified. Check in the pilot.**
- **OpenNI2:** the Kinect v1 driver is a legacy bridge (on Windows it depends on the Kinect SDK). The 2.2.0.33 builds are hosted by structure.io [S]. **Not recommended**, because it adds a layer without adding capability.
- **Recommended formats:** lossless masters, meaning RGB as PPM/PNG or an FFV1 `.mkv`, and depth as 16-bit PNG/PGM, plus a per-frame timestamp CSV. Pipeline input is an H.264 MP4 made from the masters (§9.4). ffmpeg with libx264 and ffv1 is already installed at `C:\ffmpeg\bin\ffmpeg.exe`.

### 6.4 Skeleton comparison: Kinect SDK vs MediaPipe, for this project's 12 landmarks

| This project needs | MediaPipe (33) | Kinect v1 SDK (20) |
|---|---|---|
| Shoulders L/R | 11/12 ✓ | SHOULDER_L/R ✓ |
| Hips L/R | 23/24 ✓ | HIP_L/R ✓ (plus HIP_CENTER) |
| Knees L/R | 25/26 ✓ | KNEE_L/R ✓ |
| Ankles L/R | 27/28 ✓ | ANKLE_L/R ✓ |
| **Heels L/R** | 29/30 ✓ | **✗ none** |
| **Foot index L/R** | 31/32 ✓ | FOOT_L/R (one "foot" point; not the same definition) |

Consequences of using the Kinect skeleton:
- The 12-joint set becomes 10 joints. The ankle angle would use knee–ankle–foot.
- The feature becomes 30 + 6 + 30 = **66 dims**, which breaks every hard-coded 78 (§2.6, §2.7).
- The model is no longer comparable with the phone model.
- The skeleton comes from depth, so **it can't be computed for an RGB deepfake video**.

### 6.5 Known failure modes that matter here

- **Sunlight and wide-spectrum light:** "the Kinect will not work in direct sunlight, e.g. outdoors" [P, S4]. Kinect v1 performs poorly under high-intensity wide-spectrum ambient light such as sunlight or halogen lamps [P, S14]. **Record indoors, away from windows, and don't use halogen lighting.**
- **Materials:** depth fails on "transparent, reflective, or IR light-absorbing materials such as water bottles, mirrors or leather fabrics" [P, S14, citing Mallick et al. 2014]. **Leather or dark shoes sit exactly at the heels and toes, the joints with the highest attribution (§3.6).**
- **Glass:** IR "can see through glass" [P, S4].
- **Other IR sources:** other Kinects or IR emitters reduce skeletal accuracy; use no more than one IR source per field of view [P, S3].
- **Distance:** depth error grows with distance² [P, S7]. At the ≥2.7 m needed for full-body framing, expect roughly 1.5–4 cm random error, which is comparable to heel and toe motion [E].
- **Structured-light shadow:** the projector–camera baseline leaves invalid-depth bands beside the silhouette. Feet next to the floor also mix foot and floor depth. [E, general knowledge; confirm in the pilot.]
- **Temporal alignment:** treat RGB and depth frames as not frame-locked; pair them by device timestamp [E; verify in the pilot].

---

## 7. Literature: established, contested, unknown

Every source was located. Where only the title/venue could be verified, that is stated.

### Established

- MediaPipe's normalized z is a relative depth with the hip midpoint as origin. World landmarks are metres, hip-origin, and still inferred from a single image [P, S15].
- MediaPipe's landmark model takes a 256×256 input (the detector 224×224) [P, S15].
- Kinect v1 depth noise and quantization grow roughly with distance² [P, S7].
- **Kinect-based person identification leans heavily on body size** [P, S10]:

  | Features (140 subjects, Xbox 360 Kinect, SDK 1.0) | Accuracy |
  |---|---|
  | Anthropometric only | 84.7% (SVM), 85.4% (KNN) |
  | Gait only | 62.9% (SVM), 59.5% (KNN) |
  | All | 86.3% (SVM), 87.7% (KNN) |

  Gait adds "only an average of 3.3 percentage points" over anthropometrics, and "12 of the 14 selected attributes are anthropometric".
- Multimodal gait datasets recorded with depth exist. TUM-GAID has RGB + depth + audio, 305 subjects, and a 32-subject second session. Its authors report "multimodal fusion being beneficial" [P, S16; the abstract verified; that the sensor was a Kinect is from secondary sources only].
- Structured-light depth fails in sunlight and on IR-absorbing or reflective materials [P, S4, S14].
- Depth helps face presentation-attack detection *at capture time* [title/venue verified: Atoum et al., IJCB 2017, S17]. This is a live-capture setting, not a recorded video.

### Contested or limited

- **Kinect v1 skeleton joint angles vs Vicon.** Reported: low hip-angle correlation (r < 0.30), knee better but not clinical-grade, stride timing well correlated [S18: Pfister et al., J Med Eng Technol 2014; title, authors and venue verified via Semantic Scholar; the numbers come from a search-engine summary of the abstract, full text not read].
- **Whether depth improves *gait dynamics* recognition beyond body shape.** TUM-GAID says fusion helps. Andersson shows gait dynamics add little once anthropometrics are present. No controlled depth-vs-monocular comparison on the same frames was found.
- **MediaPipe 3D accuracy depends strongly on viewing angle and task** [Dill et al., Sensors 2024, 24:7772; title/venue located; S19's numeric claims (e.g. 30.1 mm RMSE) **not re-verified**, because the page was blocked].

### Unknown (no source found; the search was not exhaustive)

- Any Kinect-depth vs MediaPipe-monocular comparison for gait **identity verification**.
- Any depth-based gait method for **face-swap deepfake** detection.
- **How much this project's model relies on MediaPipe z.** No per-axis attribution exists. A crude in-repo proxy follows.
  - Measure: second differences of the 60-step sequences, which mixes real motion with noise.
  - In absolute units, z jitters 2.5× more than x and 6.8× more than y.
  - Relative to each axis's own within-sequence spread, the ratio is x 1.30, z 1.01, y 0.55.
  - **So these data do not show z as uniquely unreliable.** A prior session's "z is 2.6–2.9× noisier" claim should not be relied on. The Kinect pilot's MediaPipe-z vs depth comparison (§11) answers the question directly.

### Not used

These were deliberately left out because they couldn't be verified: a "twelve studies" Kinect-validity summary (the page was captcha-blocked), HackerNoon-style blog claims about dark surfaces, and the numbers in the repo's own `research/claude_research_verification.md`.

---

## 8. Goal A analysis and the decisions (with reasons)

### 8.1 What could change, in which direction, and why

"Kinect RGB" (arm **A1**) means MediaPipe run on the Kinect colour stream. "Depth arms" means A3 (MediaPipe x,y plus Kinect depth as z) and A4 (fully metric 3D).

| Quantity | Kinect RGB (A1) | Depth arms (A3, A4) | Reasoning |
|---|---|---|---|
| **Identity verification** (LOOCV AUC, EER, acc/F1/P/R) | **Unknown direction; likely similar or slightly lower** [E] | **A3:** ~unchanged if the model barely uses z (untested). **A4:** plausibly **higher**, via body size | **Could go down:** Kinect v1 colour noise may drop detections, as the `noise` augmentation did; people appear smaller (~305 px vs ~680 px, still above MediaPipe's 256 px crop); the subject-correlated fps/duration confound (B10) disappears. **Could go up:** fixed 30 fps and fixed framing reduce nuisance variance. **A4:** metric limb lengths and height are strong identity cues [S10], **but** foot and heel depth is the least reliable (§6.5) and side view is weak for the Kinect. |
| **Face-swap deepfake detection** | Same mechanism as the phone model; rejection depends on identity-verification quality | **Not applicable.** A face-swapped video has no depth channel. | Depth only matters in a live-capture threat model (trusted sensor at capture time), and there depth-based liveness is the established tool, not gait [S17]. |
| Operating threshold τ\* | **Must be re-derived** | Must be re-derived | Score distributions differ per model; 0.7737 was already optimistic (B5). |
| Per-fold spread | Unknown | Unknown | Driven by the 13-subject cohort (between-subject AUC SD ≈ 2.7–3.1 points). |
| Parameter counts | Unchanged (78 dims → 133,058 on the decision path) | Unchanged for A3/A4 (78 dims) | Only a 66-dim Kinect-skeleton arm (A5) or the 54-dim no-z arm (A2) changes the first conv layer. |
| Ablation conclusion ("raw differencing is best") | Unchanged unless re-run | Unchanged unless re-run | It is a statement about this cohort size. |
| Attribution ranking | Likely changes | Likely changes (heels/feet) | New sensor, new noise profile. |

**Separating the sensor from depth.** A Kinect v1 differs from a phone in many ways:
- colour-camera quality and noise,
- resolution and aspect (640×480 landscape vs 1080×1920 portrait),
- field of view and focal length,
- fixed frame rate,
- the distance forced by framing,
- and the fact that every recording is a *new session*.

**None of these is depth.** Any phone-vs-Kinect difference measured with A1 is a *camera* effect. A *depth* effect can only be measured by holding the frames fixed and changing only the z source: A1 vs A3 vs A4 on the same Kinect recordings. That is the design in §11.

### 8.2 Feature strategy (item 8 of the brief)

| Option | What "the Kinect effect" then means | Pros | Cons | Decision |
|---|---|---|---|---|
| **(a) MediaPipe on Kinect RGB** | Effect of the *camera* (RGB only) | Zero pipeline change. The same 78-dim model. **Usable on RGB deepfake videos.** | Measures nothing about depth | **Primary (A1). The Goal B model `gait-kinect-rgb-v1`.** |
| (b) Kinect SDK skeleton | Effect of a different pose estimator driven by depth | Metric 3D, depth-based | No heels or foot index → 66 dims, not comparable. Weak for side views. Needs SDK 1.8, which fails on Windows 11. Can't be computed for RGB deepfakes. | **Optional last arm (A5)**, only if the SDK works |
| (c) MediaPipe 2D + depth lookup | Effect of *measured* vs *monocular* z (A3), and of metric scale (A4) | Keeps the same 12 joints and 78 dims. A controlled, same-frame contrast. | Depth at the feet is unreliable. Needs registered depth. Unusable on RGB deepfakes. | **Secondary research arms A3/A4** |
| A2: drop z | Contribution of the current MediaPipe z | Free (a feature mask) | 54 dims | **Control arm** |

- **Reasoning.** Goal B asks for "the entire existing pipeline" on Kinect data. Only (a) does that, and only (a) keeps the deepfake use case intact. Goal A's depth question needs (c) on the same frames.
- **Depth is recorded from day one,** so (c) and (b) need no re-recording.
- **Consequence of dimension changes:**
  - A1, A3 and A4 keep 78 dims, so the head architecture is identical and directly comparable.
  - A2 (54) and A5 (66) change only `raw_dim`, i.e. the first Conv1d's input channels.
  - The new P2 script (§10.3) takes the dimension from the data, so no model code changes are needed.

### 8.3 Phone data: keep separate, never mix

- **The existing phone dataset stays frozen.** It serves as the historical baseline and is never re-augmented or rewritten in place.
- **No model is trained on pooled phone and Kinect sequences.**
  - The features are not comparable across cameras (aspect-dependent coordinates and angles, §2.12).
  - If the subject sets differ by sensor, sensor becomes a subject shortcut.
- **The fair phone comparison uses *new* phone video recorded at the same time as the Kinect.** Same walks, same session, same subjects, processed with the same code (§11.2).
- **Enrol-on-phone / probe-on-Kinect is run only as a stress test.** It is expected to fail with the v1 features because of B11. That is a finding, not a bug to hide.

### 8.4 Code-change decision

**Small tweaks plus three new files. No model rewrite, and no refactor of `models/` or `utils/data_loader.py`.**

- **Why not nothing:**
  - Scripts don't run without `PYTHONPATH` (B15).
  - Defaults overwrite phone artefacts.
  - There is one data-loss trap (B17) and one silent-backend trap (B18).
  - Augmentation is frame-inconsistent (B7, B9).
  - There is no Kinect I/O.
  - The legacy protocol leaks (B1–B3, B13), so it can't support a scientific comparison.
- **Why not a rewrite:** the deployed head is 133k parameters and the ablation shows it is the right size for this cohort. Every experiment here is about **features and protocol**, not architecture.

The ordered change list is in §10.2, with specs in §10.3.

---

## 9. Kinect setup, recording protocol, naming and export

### 9.1 Hardware and driver setup (staged; **ask the user before any install**)

**Hardware**
- Kinect (note the model number on the base).
- The **Kinect power/USB adapter** (§6.1).
- A tripod with a flat plate.
- A second tripod or bracket for the phone.
- A USB 2.0 port on its own controller: the SDK requires a "dedicated USB 2.0 bus". Avoid hubs.

**Stage 1: SDK 1.8 on the recording PC (≈1 h, needs install approval)**
1. Install the Kinect for Windows SDK v1.8 and Developer Toolkit v1.8 (§6.3) with the sensor **unplugged**.
2. Then plug in the adapter, power it, and connect USB.
3. Run the Toolkit's "Kinect Explorer" / "Depth Basics" sample.
4. **Pass:** colour and depth images appear.
5. **Fail:** the device shows up only as an audio device (the known Windows 11 symptom, §6.3). Go to Stage 2.

**Stage 2: libfreenect**
- **(a) Linux, preferred if available.** A Ubuntu machine or live USB, then `apt-cache show freenect` / `sudo apt install freenect`. Whether the distro package ships the `record` tool is **unverified**. If it doesn't, build libfreenect from source (CMake).
- **(b) Windows.** Build libfreenect with CMake + MSVC Build Tools + libusb ≥ 1.0.22, and use Zadig to put **libusbK** on each Kinect interface (camera and motor, plus audio on a 1473) [S8].
- **Do not use WSL.** USB isochronous pass-through is unreliable [E].
- **Pass:** `freenect-glview` shows RGB and depth.

**Stage 3: a Windows 10 PC with SDK 1.8.** Reported to work (§6.3).

**Record which stage succeeded** in `data_kinect/SETUP.md`, with the OS build, driver and library versions, and the Kinect model.

### 9.2 Recording protocol

- **Room:** indoors, ≥ 9 m × 3 m clear.
  - No direct sunlight: blinds closed, no windows in view.
  - **No halogen lamps**; use LED or fluorescent light.
  - No mirrors or glass in view.
  - **No other IR emitters**: other Kinects, RealSense, IR cameras.
  - Keep the lighting the same across sessions.
- **Kinect placement:**
  - Tripod, lens ≈ **1.0 m** above the floor (≈ hip height, which reduces limb foreshortening) [E].
  - Tilt set to **0°** by command, not by hand, and checked with a spirit level.
  - Log height, tilt, and the positions of the tape lines in `meta.json`. **Never move the camera within a session.**
- **Phone (simultaneous):**
  - Rigidly mounted 10–15 cm directly above the Kinect lens.
  - **Portrait** 1080×1920, the same orientation as the old dataset.
  - **Locked 30 fps.** Stabilisation, HDR and beauty filters off. Exposure locked if the app allows.
- **Floor tape:**
  - **Side (S):** a walking line parallel to the image plane at **3.5 m** from the Kinect. Start ≥ 1.5 m before the left edge of the field of view and stop ≥ 1.5 m after the right edge, so the gait is steady inside the frame. About 3.8 m of path is visible [E].
    - **Alternate direction** L→R and R→L between takes, and log it.
  - **Frontal (F):** walk along the optical axis, starting at **8 m** and walking toward the camera, with a stop line at **2.5 m**.
    - The RGB is usable along the whole path.
    - A full body with valid depth exists only at ~2.7–4.0 m (§6.2). **This is a hard limitation of the Kinect v1 for frontal gait. Accept it and report it.**
- **Takes:** per subject per session, **4 frontal + 4 side** (F1–F4, S1–S4).
  - Enrolment takes: **F1, S1**. Probe takes: the rest.
  - The old dataset had 2–3 takes per view, which makes a leak-free split too thin.
- **Sessions:** **2 sessions on different days** (d1, d2) if at all possible. The realistic deepfake scenario has enrolment footage and suspect footage from different days.
- **Subjects:**
  - Ideally the **same 13 people** under the same names, plus as many extra people as possible. Power is limited by subject count (§11.5); ≥ 30 subjects would help most.
  - Natural pace, arms free, everyday clothes.
  - **Log shoe type** (leather or dark shoes degrade depth at the feet, §6.5) and any long garments. Don't change what people wear to suit the sensor.
- **Per take:**
  - One **clap** at the start, visible to both cameras (sync marker).
  - Walk, then stop recording after the person leaves the frame or reaches the stop line.
- **Per session:**
  - **20 checkerboard images** with the Kinect RGB (for intrinsics).
  - A flat board at 2, 3 and 4 m measured with a tape (depth sanity check).
  - One photo of the setup.
- **Consent** must cover RGB video, depth data, phone video, publication, and use of face images to make face-swaps.

### 9.3 Naming conventions, directory layout, and what must never be overwritten

- **Subject names:** reuse the *exact* phone names (`A2, Aarav, Ananya, Arhaan, Bharti, Devika, Prakhar, Prayag, Som, Teja, Vedant, Vedant2, Vibhav`) for returning subjects. New subjects get new unique names made only of letters and digits: **no `_`, no spaces**.
- **Take ID:** `{Name}_{V}{T}d{S}`, for example `Arhaan_F1d1` or `Teja_S3d2`.
  - V ∈ {F, S}, T ∈ 1..4, S = session number.
  - This fits the pipeline's identity parsing (the first `_` token), `run_gradcam.py`'s `rsplit('_',1)`, and the augmentation naming `{stem}_{aug}`.
- **Layout (new, git-ignored trees):**

```
data_kinect/
  SETUP.md                      driver stage used, versions, Kinect model
  takes.csv                     take_id,subject,session,view,take,direction,trim_start,trim_end,shoes,notes
  calib/d1/, calib/d2/          checkerboard images, intrinsics.json, depth-check photos
  raw/<TakeID>/                 LOSSLESS MASTER, never edited: rgb frames, depth frames, index/timestamps.csv, meta.json
  raw_phone/<TakeID>.mp4        untouched phone originals
  kinect_rgb/videos/<TakeID>.mp4             trimmed, 640x480, 30 fps, H.264  <- pipeline input (A1)
  kinect_rgb/augmented_videos/<TakeID>_<aug>.mp4
  kinect_depth/<TakeID>/        trimmed, registered 16-bit depth + timestamps (A3/A4)
  phone/videos/<TakeID>.mp4     trimmed simultaneous phone, 1080x1920, 30 fps
  phone/augmented_videos/...
  phone_kres/videos/<TakeID>.mp4   phone degraded to Kinect geometry (control, §9.4)
  phone_kres/augmented_videos/...
  gait_features/<dataset>/gait_features.pkl        dataset in {kinect_rgb_v1, kinect_rgbd_v1, phone_v1, phone_kres_v1}
  gait_features/<dataset>/enrolled_identities.pkl
outputs_kinect/
  gait-kinect-rgb-v1/           train.py --output_dir: checkpoints/, model_config.json, training_history.json, tensorboard/
  gait-kinect-rgb-v1/evaluation/loocv/loocv_results.json     (legacy protocol, continuity only)
  gait-kinect-rgb-v1/gradcam/aggregate/
  p2/<arm>/results.json, pairs.csv                            (protocol P2, one dir per arm)
  p2/compare/<armX>_vs_<armY>.json
  REPORT.md
```

- **The features file must be named exactly `gait_features.pkl`** inside its own directory. Any other name, plus the auto-enrol step, overwrites it (B17).
- **New model name:** `gait-kinect-rgb-v1`. Research arms are named by protocol and arm, e.g. `p2/A3-kinect-depthz`, never as deployable models.
- **Add to `.gitignore` (proposed; not done):** `data_kinect/`, `outputs_kinect/`.

**Never overwrite, move or delete:**
- `data/videos/`, `data/augmented_videos/`, `data/deepfake/` (including `faces/`)
- `data/gait_features/gait_features.pkl`, `data/gait_features/enrolled_identities.pkl`, `data/subject_map.json`
- `outputs/checkpoints/*`, `outputs/model_config.json`, `outputs/evaluation/**`, `outputs/ablation/**` (JSON and `.pth`), `outputs/gradcam/**`, `outputs/paper_figures/**`, `outputs/keypoint_visualizations/**`, `outputs/graphs/**`
- `figures/*.png`, `paper/*`, existing `logs/*`, `DOCUMENTATION/*`, `README.md`

**Dangerous defaults.** Always pass the override.

| Script | Default that hits phone artefacts | Required override |
|---|---|---|
| `augment_videos.py` | `--input_dir data/videos --output_dir data/augmented_videos` | both |
| `preprocess_videos.py` | `--output data/gait_features/gait_features.pkl` (**appends** if it exists) | `--output data_kinect/gait_features/<ds>/gait_features.pkl` |
| `enroll_identities.py` | `--from_videos` mode, `--output data/gait_features/enrolled_identities.pkl` | `--from_features --features_file … --output …` |
| `train.py` | `--output_dir outputs`, phone features and enrolled file | all three |
| `evaluate.py --loocv` | `--output_dir outputs/evaluation`, phone features | all |
| `ablation_loocv.py` | `--output outputs/ablation/ablation_loocv_results.json` (renames the existing file) | `--output`, plus `--features_file`, `--enrolled_file` |
| `run_gradcam.py` | checkpoint auto-picked from `outputs/checkpoints`, `--output_dir outputs/gradcam` | `--checkpoint`, `--output_dir`, `--features_file`, `--enrolled_file` |
| `inference.py` | phone checkpoint, phone templates, `--threshold 0.7737` | all |
| `scripts/generate_figures/*`, `generate_paper_figures.py`, `visualize_keypoints.py` | write into `figures/` or `outputs/…` | **don't run** for Kinect (no override exists in `figstyle.py`) |
| `run_pipeline.py` | everything | **don't use** |

### 9.4 Export, trimming and conversion

1. **Keep masters untouched** in `data_kinect/raw/<TakeID>/`.
2. **Check timestamps.** Consecutive device-timestamp gaps should be ≈ 1/30 s. Any gap > 1.5 intervals is a dropped frame: fill it by repeating the previous frame so the time base stays constant, and set `dropped_frames` in `meta.json`. Reject a take with > 3% dropped frames.
3. **Trim.** Keep the longest contiguous run of frames where MediaPipe landmarks 11, 12 and 23–32 all lie inside [0.02, 0.98]² with visibility ≥ 0.5 (the whole body is in frame). Write `trim_start` and `trim_end` to `takes.csv`, and eyeball the first, middle and last frames. Use the **same time window** for the phone take, aligned on the clap.
4. **Encode the pipeline input** (ffmpeg is present):
   ```
   C:\ffmpeg\bin\ffmpeg.exe -framerate 30 -start_number <trim_start> -i data_kinect\raw\<TakeID>\rgb\%06d.png -frames:v <trim_end-trim_start+1> -c:v libx264 -crf 17 -pix_fmt yuv420p data_kinect\kinect_rgb\videos\<TakeID>.mp4
   ```
   If the masters are libfreenect `r-*.ppm` files, the converter first renames them to `%06d.png` in index order (§10.3, `convert_kinect_take.py`). **Check colour order.** OpenCV must read a red shirt as red; compare one frame against the master.
5. **Phone:** `ffmpeg -ss <t0> -to <t1> -i raw_phone\<TakeID>.mp4 -r 30 -c:v libx264 -crf 17 -pix_fmt yuv420p phone\videos\<TakeID>.mp4`.
6. **Phone-to-Kinect-geometry control (`phone_kres`):**
   - Scale the phone frames by s = (median Kinect person height in px) / (median phone person height in px), with both heights measured by MediaPipe on the same takes.
   - Crop a 480-row band centred on the person's median vertical position, then pad to 640×480:
     `-vf "scale=iw*S:-2:flags=area,crop=iw:480:0:Y,pad=640:480:(640-iw)/2:0"`
   - The result has the Kinect's frame size, aspect and person size, but the phone's optics, sensor and colour. Check that `iw*S` ≤ 640.
7. **Depth (A3/A4):** save registered 16-bit depth in mm, trimmed to the same frame indices, in `kinect_depth/<TakeID>/%06d.png`. Store the colour intrinsics (from the checkerboard) in `meta.json`.

---

## 10. Code changes and the step-by-step procedure

### 10.1 Environment (every PowerShell session)

```powershell
cd C:\Arhaan\PROJECTS\DeepFake-Detection
$env:PYTHONPATH = (Get-Location).Path      # B15: the package is not installed; this replaces `pip install -e .`
$env:PYTHONHASHSEED = "0"                  # makes train.py's subject split reproducible (B14)
$py = ".\venv\Scripts\python.exe"
& $py scripts\training\train.py --help     # EXPECT: argparse usage. If ModuleNotFoundError, PYTHONPATH is wrong.
& $py tests\smoke_test.py                  # EXPECT: "SMOKE TEST PASSED"
```

- `pip install -e .` is the documented alternative, but it's an install: **ask first**.
- If `scripts\enrollment` is not on `PYTHONPATH`, `preprocess_videos.py` ends with an ImportError from its auto-enrol step *after* saving the features. That is harmless, but it exits non-zero. Change C2 fixes it.

### 10.2 Ordered code changes (none made yet; make them on a new branch, e.g. `kinect-study`, and ask before committing)

| # | File | Change | Why | Size | Check to leave behind |
|---|---|---|---|---|---|
| C1 | `scripts/preprocessing/preprocess_videos.py` | Delete the TensorFlow GPU auto-detect (the `try: import tensorflow …` block, lines 110–119). MoveNet is used only with an explicit `--use_gpu`. | B18: silent 36-dim features | −10 lines | Run on 1 video; the log prints "Using MediaPipe (CPU)" |
| C2 | same file | (a) `enrolled_file = str(Path(args.output).with_name("enrolled_identities.pkl"))`; (b) before the import, `sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "enrollment"))` | B17: data-loss trap and failing import | 3 lines | Output ≠ enrolled path (assert) |
| C3 | `scripts/preprocessing/augment_videos.py` | (a) Make every `A.Compose` an `A.ReplayCompose`. In `apply_spatial_augmentation`, call it on frame 0 and apply `A.ReplayCompose.replay(first["replay"], image=f)` to the other frames. (b) Change `zoom_in` to `A.Affine(scale=(1.1, 1.2), p=1.0)` (resolution-independent). (c) Stop copying the originals into `--output_dir`. The originals come from `--videos_dir`, and the copies were the B3 duplicates. | B7, B9, B3 | ~15 lines | A `__main__` self-check: 5 identical frames in → 5 identical frames out, for every spatial config. The audit verified that `ReplayCompose` gives identical frames for rotate, brightness, Affine and HSV. |
| C4 | **new** `scripts/kinect/convert_kinect_take.py` | Raw take → timestamp check and dropped-frame fill → trim proposal → MP4 and depth export → `phone_kres` mode (§9.4) | Kinect I/O | ~150 lines | After export: frame count, 30 fps and 640×480 re-read with OpenCV (assert) |
| C5 | **new** `scripts/evaluation/loocv_p2.py` | Protocol P2 (`run` / `compare`), spec in §10.3 | A leak-free, seeded, paired comparison | ~250 lines | Built-in leakage asserts, plus `--quick` (2 folds × 2 epochs) |
| C6 | **new** `utils/pose_extraction_rgbd.py` | Arms A3/A4: MediaPipe 2D + registered depth → z (A3) or metric 3D with 3D angles (A4). Same output schema as `process_video`. | Depth arms | ~150 lines | Synthetic check: a flat plane at a known depth maps to the known Z; the hip-centred z mean ≈ 0 |
| C7 | *optional* `scripts/kinect/record_freenect.py` | Fallback recorder if libfreenect's `record` tool doesn't store registration. Loops `freenect.sync_get_video()` and `freenect.sync_get_depth(format=<REGISTERED>)`, writes PNGs and timestamps. **First check the Python wrapper exposes the registered-depth format.** | Capture | ~60 lines | 100 frames recorded, no gaps |
| C8 | *optional, A5 only* Kinect SDK recorder (C#, from the Toolkit's Color/Depth/SkeletonBasics samples) | Writes colour, depth (`MapDepthFrameToColorFrame`) and 20 joints plus tracking state per frame | Kinect skeleton arm | Visual Studio needed | — |
| C9 | *optional* `scripts/evaluation/verify_gait_preservation.py` | Fix the torso indices (hips = gait indices 2,3; shoulders = 0,1). Add `--original <path>` so the exact source take is compared. | B20 | ~5 lines | Comparing a video with itself gives PCK = 1 |
| C10 | `.gitignore` | Add `data_kinect/` and `outputs_kinect/` | Keep large data out of git | 2 lines | `git status` stays clean after runs |

**Deliberately unchanged:**
- `models/*`, `utils/data_loader.py`, `utils/pose_extraction.py` (the phone features stay reproducible), `train.py`, `evaluate.py`, `ablation_loocv.py`, `inference.py`, `run_gradcam.py`.
- The legacy tier uses them *as they are* (Goal B continuity). Their flaws are neutralised by P2, not patched, so the phone numbers keep their meaning.

### 10.3 Specs for the new files

**`scripts/evaluation/loocv_p2.py` (protocol P2)**

```
run     --features_file F --out DIR --enroll_takes F1,S1 [--enroll_session d1 --probe_session d2]
        [--drop_z] [--train_aug] [--seeds 0 1 2 3 4] [--epochs 30 --lr 1e-3 --batch_size 16 --dropout 0.1] [--quick]
compare DIR_A DIR_B --out FILE.json
```

**Parsing.** Parse each pickle key as `stem = Path(key.replace('\\','/')).stem`, `parts = stem.split('_')`:
- name = `parts[0]`
- take = `parts[1]`
- aug = `'_'.join(parts[2:])`, or `orig` if empty
- session = the `dN` suffix of the take, or `d1` if absent. The old phone keys have no suffix.

Drop an augmented-folder `orig` if the same stem exists from the videos folder (B3).

**Features.** Reuse `GaitDataset._prepare_features(SimpleNamespace(include_angles=True, include_velocities=True), entry)` so the 78-dim order is identical. `--drop_z` removes columns `2,5,…,35` and `44,47,…,77` (54 dims left).

**Per fold** (held-out subject s):
- **Templates:** for every subject, the mean of the **original** sequences of its enrolment takes in the enrolment session. No augmentations, no probes.
- **Training probes:** sequences of the other 12 subjects' **probe takes**. Originals only, plus their augmentations with `--train_aug`. **Enrolment takes and their augmentations are excluded.**
- **Test probes:** s's probe takes in the probe session, **originals only**.
- **Normalization:** mean and std over the training probes (std < 1e-6 → 1), applied to the training probes, the test probes and all templates.
- **Training pairs:** each epoch, every training probe gets 1 positive (its own template) and 1 negative (a random *training-subject* template, `random.Random(seed*1000+epoch)`). This fixes B26.
- **Test pairs, deterministic:** each test probe against its own template (positive) and against **all 12 other templates** (negatives). No sampling noise, and identical pairs across arms and seeds.
- **Model:** `EncodedDiffVerifier(RawEncoder(), raw_dim=D)`, loaded from `scripts/evaluation/ablation_loocv.py` via importlib. With a raw-only encoder only `raw_dim` is used, so no edit is needed. AdamW with weight decay 1e-4, CE, gradient clipping 1.0, 30 epochs, state chosen by lowest training loss. Seed `random`, `numpy`, `torch` and `torch.cuda`.

**Outputs:**
- `pairs.csv`: fold, seed, probe_key, true_id, claimed_id, label, score.
- `results.json`:
  - Per-fold AUC and EER (per seed and seed-averaged); pooled AUC and EER.
  - TAR at FAR 1% and 5% (pooled).
  - **Rank-1 identification:** the argmax over the 13 templates, per probe.
  - Accuracy/F1 at 0.5.
  - **LOO-threshold accuracy:** τ\_k = Youden on the pooled scores of all *other* folds, applied to fold k. This is an honest data-derived operating point.
  - Deploy threshold = Youden on all folds, labelled "for deployment only; not an evaluation number".
  - Counts, config, git hash, package versions, and the sha256 of the features file.

**Asserts (the check), printed per fold:**
- No test-probe take, and no augmentation of one, is among the training keys.
- s's template uses only enrolment takes.
- The enrolment and probe take sets are disjoint.
- The test-pair hash is identical across seeds.

**`compare`:**
- Matches subjects.
- Per-subject ΔAUC, seed-averaged.
- 10,000-sample **subject-level** bootstrap 95% CI.
- Wilcoxon signed-rank test with n = number of subjects.
- The MDE computed from the seed SD of both arms.
- Writes JSON.
- **Never pool seeds as independent samples.**

**`utils/pose_extraction_rgbd.py` (arms A3/A4)**

`RGBDGaitExtractor(GaitFeatureExtractor)`, with `process_take(video_path, depth_dir, intrinsics, mode)`:
- For each frame: MediaPipe landmarks → pixel (u, v) → **median of valid (>0) depth in a 5×5 window**, in mm.
- **`depthz` (A3):** keep MediaPipe x, y (normalized). z = (Z_joint − Z_hipmid) in metres.
- **`metric` (A4):** X = (u−cx)·Z/fx, Y = (v−cy)·Z/fy, Z in metres. Override `compute_joint_angles` to use all 3 coordinates.
- **Missing depth:**
  - Frames with missing hip depth are dropped (like non-detections).
  - Other missing joints are linearly interpolated over time.
  - Store the per-joint depth-valid fraction, and flag (never silently drop) takes with < 70% valid at any of the 12 joints.
- Then run the usual `_normalize_sequence_length` and `compute_gait_features`, so the output schema is identical and `loocv_p2.py` works unchanged.
- Also save the per-frame (MediaPipe z, depth z) pairs to `outputs_kinect/p2/zagreement/<TakeID>.npz` for the Bland–Altman analysis (§11.3).
- CLI: `--videos_dir --depth_dir --intrinsics --mode {depthz,metric} --output <dir>/gait_features.pkl`.

**Augmentation for the depth arms:** geometric augmentations applied to RGB would desynchronise it from depth. So **all depth-contrast arms (A1-noaug, A2, A3, A4) run without augmentation.** A1-with-augmentation is reported separately as the Goal B model.

**`scripts/kinect/convert_kinect_take.py`**

Modes:
- `check`: timestamps and dropped frames.
- `trim`: the MediaPipe full-body test (§9.4); writes proposals to `takes_trim_proposed.csv` and never edits `takes.csv`.
- `export`: MP4 via ffmpeg and trimmed depth PNGs, plus `meta.json`.
- `phone_kres`.

The input reader handles libfreenect `record` dumps (`index.txt`, `r-*.ppm`, `d-*.pgm`) and the C7/C8 PNG dumps. Always pass colour to OpenCV as BGR.

### 10.4 Step-by-step: from raw Kinect recordings to a trained, evaluated model

Run every command after §10.1. Where a step says `<ds>`, use the dataset folder name, e.g. `kinect_rgb_v1`.

**Step 0: ask the user (§13.2), make changes C1–C6 and C10 on a branch, and run their checks.**
- Also run `& $py scripts\evaluation\loocv_p2.py run --features_file data\gait_features\gait_features.pkl --out outputs_kinect\p2\phone-legacydata-quick --enroll_takes F1,S1 --quick`.
- EXPECT: the leakage asserts print OK and a `results.json` is written. This reads the phone pickle; it writes only under `outputs_kinect`.

**Step 1: driver (§9.1) and pilot recording.** Record 2 subjects × (2F + 2S), then convert, trim and extract.

Go/no-go criteria:

| Check | Pass |
|---|---|
| Dropped frames | < 3% per take |
| MediaPipe detection on trimmed frames | ≥ 95% (phone originals had 98.5%) |
| Colour | Correct (red is red) |
| Person height at the side line | ≥ 256 px |
| Depth-valid fraction at heels and feet | Recorded, whatever the value. If < 70%, **A3/A4 results must be flagged** |
| Checkerboard reprojection error | < 1 px |
| Flat-board depth error at 3 m | Within ~2–3 cm (Khoshelham) |

**Step 2: record everyone** (§9.2). Fill `takes.csv` as you go.

**Step 3: convert.** For each take:
```
& $py scripts\kinect\convert_kinect_take.py check --raw data_kinect\raw\<TakeID>
& $py scripts\kinect\convert_kinect_take.py trim --raw … --out data_kinect\takes_trim_proposed.csv
# review the proposals and copy the accepted trims into takes.csv, then:
& $py scripts\kinect\convert_kinect_take.py export --raw … --takes data_kinect\takes.csv --rgb_out data_kinect\kinect_rgb\videos --depth_out data_kinect\kinect_depth
```
Also export the phone takes, then `phone_kres` (§9.4).

SANITY:
- Every subject has 8 takes per session in each of `kinect_rgb/videos`, `phone/videos` and `phone_kres/videos`, with identical stems.
- Kinect and `phone_kres` files are 640×480; phone files are 1080×1920; all are 30 fps.

**Step 4: augment (Goal B path; uses C3).**
```
& $py scripts\preprocessing\augment_videos.py --input_dir data_kinect\kinect_rgb\videos --output_dir data_kinect\kinect_rgb\augmented_videos
```
EXPECT: 15 files per take (no copied originals, after C3). Repeat for `phone` and `phone_kres` if they are to be compared with augmentation.

**Step 5: extract features.**
```
& $py scripts\preprocessing\preprocess_videos.py --videos_dir data_kinect\kinect_rgb\videos --augmented_dir data_kinect\kinect_rgb\augmented_videos --output data_kinect\gait_features\kinect_rgb_v1\gait_features.pkl
```
EXPECT:
- "Using MediaPipe (CPU)".
- Entries ≈ takes × 16.
- Failures are listed.
- Time ≈ frames × ~20 ms (§1.3) plus decoding.

SANITY:
- `& $py ieee_scripts\quickstart.py --features data_kinect\gait_features\kinect_rgb_v1\gait_features.pkl --enrolled data_kinect\gait_features\kinect_rgb_v1\enrolled_identities.pkl` passes the 60×78 layout check.
- `pose_sequence.shape == (60, 33, 3)`.
- The valid/total frame ratio per augmentation type is tabulated, as in §5 B8.
- **No original has a valid ratio < 0.9** (otherwise re-trim).

Repeat for `phone_v1`, `phone_kres_v1`, and a no-augmentation `kinect_rgb_noaug_v1` (omit `--augmented_dir`). Build the depth arms with C6 (`--mode depthz` → `kinect_rgbd_depthz_v1`, `--mode metric` → `kinect_rgbd_metric_v1`).

**Step 6: enrol (legacy tier only).** With C2, enrolment is written automatically at the end of Step 5. Otherwise:
```
& $py scripts\enrollment\enroll_identities.py --from_features --features_file data_kinect\gait_features\kinect_rgb_v1\gait_features.pkl --output data_kinect\gait_features\kinect_rgb_v1\enrolled_identities.pkl
```
EXPECT: one entry per subject, with `num_videos` = that subject's entry count. These templates contain test data (B1). P2 builds its own.

**Step 7: train the new deployment model `gait-kinect-rgb-v1` (the Goal B deliverable).**
```
& $py scripts\training\train.py --features_file data_kinect\gait_features\kinect_rgb_v1\gait_features.pkl --enrolled_file data_kinect\gait_features\kinect_rgb_v1\enrolled_identities.pkl --output_dir outputs_kinect\gait-kinect-rgb-v1 --epochs 50 --seed 42
```
EXPECT:
- `outputs_kinect\gait-kinect-rgb-v1\checkpoints\checkpoint_epoch_{1..50}.pth`, plus `*_best.pth` files.
- `model_config.json` with `input_dim` 78.
- About 2 minutes of GPU time. This is an **estimate** from 70–115 s per 50-epoch LOOCV fold in `logs/evaluate_20260209_001725.txt`.
- The printed train/val subject lists are in `logs\train_<timestamp>.txt`. **Copy them into `outputs_kinect\gait-kinect-rgb-v1\SPLIT.txt`** (B14).

SANITY:
- Validation accuracy moves off 50%.
- The `[Diag] Pred: auth=…/…` counts are not all one class.
- `feature_stats` has dimension 78.
- **`outputs\checkpoints` modification times are unchanged.**

**Step 8: legacy LOOCV (continuity; same flaws as the phone headline).**
```
& $py scripts\evaluation\evaluate.py --loocv --loocv_epochs 50 --lr 1e-4 --features_file data_kinect\gait_features\kinect_rgb_v1\gait_features.pkl --enrolled_file data_kinect\gait_features\kinect_rgb_v1\enrolled_identities.pkl --output_dir outputs_kinect\gait-kinect-rgb-v1\evaluation
```
EXPECT: `outputs_kinect\gait-kinect-rgb-v1\evaluation\loocv\loocv_results.json`, in ~20 min (the phone run took 1,228 s).
- **Run the identical command on `phone_v1`** into `outputs_kinect\phone-v1-legacy\evaluation`. Only that phone number may be set beside the Kinect legacy number. The 94.95 was produced by an older script with an unlogged learning rate (B4).
- Note: this command reads `outputs/model_config.json`, read-only, which is fine.

**Step 9: protocol P2, all arms, 5 seeds each.**
```
& $py scripts\evaluation\loocv_p2.py run --features_file data_kinect\gait_features\kinect_rgb_v1\gait_features.pkl --train_aug --enroll_takes F1,S1 --out outputs_kinect\p2\A1-kinect-rgb-aug
& $py scripts\evaluation\loocv_p2.py run --features_file data_kinect\gait_features\kinect_rgb_noaug_v1\gait_features.pkl --enroll_takes F1,S1 --out outputs_kinect\p2\A1-kinect-rgb
& $py scripts\evaluation\loocv_p2.py run --features_file data_kinect\gait_features\kinect_rgb_noaug_v1\gait_features.pkl --drop_z --enroll_takes F1,S1 --out outputs_kinect\p2\A2-kinect-noz
& $py scripts\evaluation\loocv_p2.py run --features_file data_kinect\gait_features\kinect_rgbd_depthz_v1\gait_features.pkl --enroll_takes F1,S1 --out outputs_kinect\p2\A3-kinect-depthz
& $py scripts\evaluation\loocv_p2.py run --features_file data_kinect\gait_features\kinect_rgbd_metric_v1\gait_features.pkl --enroll_takes F1,S1 --out outputs_kinect\p2\A4-kinect-metric
& $py scripts\evaluation\loocv_p2.py run --features_file data_kinect\gait_features\phone_v1\gait_features.pkl --enroll_takes F1,S1 --out outputs_kinect\p2\P-phone
& $py scripts\evaluation\loocv_p2.py run --features_file data_kinect\gait_features\phone_kres_v1\gait_features.pkl --enroll_takes F1,S1 --out outputs_kinect\p2\P-phone-kres
```
With two sessions, repeat everything with `--enroll_session d1 --probe_session d2`, output `…-xsession`. That cross-session result is the **primary** endpoint.

SANITY:
- The leakage asserts pass.
- The test-pair hash is identical across all arms that use the same takes.
- Per-fold n = probes × 13.

**Step 10: compare** (the contrasts are in §11.2):
```
& $py scripts\evaluation\loocv_p2.py compare outputs_kinect\p2\P-phone outputs_kinect\p2\A1-kinect-rgb --out outputs_kinect\p2\compare\phone_vs_kinectrgb.json
```
…and so on for each contrast.

**Step 11 (optional): explainability for the new model.**
```
& $py scripts\evaluation\run_gradcam.py --checkpoint outputs_kinect\gait-kinect-rgb-v1\checkpoints\<best>.pth --features_file data_kinect\gait_features\kinect_rgb_v1\gait_features.pkl --enrolled_file data_kinect\gait_features\kinect_rgb_v1\enrolled_identities.pkl --output_dir outputs_kinect\gait-kinect-rgb-v1\gradcam --aggregate --n_samples 30
```
Aggregate mode only (B19). It uses enrolled templates that contain the samples, so treat it as descriptive.

**Step 12: face-swap test** (§11.4).

**Step 13: inference with the new model.**
```
& $py scripts\inference\inference.py --video <mp4> --claimed_identity <Name> --checkpoint outputs_kinect\gait-kinect-rgb-v1\checkpoints\<best>.pth --enrolled_file data_kinect\gait_features\kinect_rgb_v1\enrolled_identities.pkl --threshold 0.5
```
**Never use the phone threshold 0.7737.** Use 0.5 until a threshold is derived without test data. P2's "deploy threshold" comes from the Raw-head retrains, not from this exact checkpoint, so treat it as approximate.

**Step 14: write `outputs_kinect/REPORT.md`.** Every number must cite its JSON. Label proxies as proxies.

---

## 11. Evaluation and comparison design

### 11.1 Hypotheses (write them down before looking at results)

- **Primary endpoint:** per-subject ROC-AUC under P2, seed-averaged. Use cross-session if 2 sessions were recorded; otherwise within-session.
- **Secondary endpoints:** EER, TAR at FAR 5%, rank-1 identification, and LOO-threshold accuracy.

| H | Contrast (same subjects, same takes, same P2 pairs, same seeds) | What it isolates |
|---|---|---|
| H1 | **P-phone vs A1-kinect-rgb** | Total *camera* effect, RGB only. This is Goal A as asked. |
| H2 | P-phone vs **P-phone-kres** | Resolution + aspect + person-size effect, with the phone sensor held fixed |
| H3 | **P-phone-kres vs A1** | What remains after geometry is matched: optics, noise, colour, 10–15 cm viewpoint offset |
| H4 | A1 vs **A2-noz** | How much the current MediaPipe z contributes |
| H5 | A1 vs **A3-depthz** | **Depth effect:** measured vs monocular z, everything else fixed |
| H6 | A3 vs **A4-metric** | Effect of metric scale and 3D angles (the anthropometric cue) |
| H7 | A1-aug vs A1 | Effect of the fixed augmentation |
| H8 | Within-session vs cross-session (same arm) | Session effect, i.e. the realistic enrol/suspect gap |

**Deepfake:** there is no depth hypothesis. Depth can't be computed for an RGB suspect video (§8.1).

### 11.2 Fair-comparison rules

1. **Compare only within one protocol.** P2 against P2, legacy against legacy (Step 8). **Never set a Kinect number beside 94.95.**
2. **Keep everything fixed except what is under test:** subjects, takes, trims, pairs, seeds, epochs, learning rate, augmentation code, feature code.
3. **The subject is the unit.** Average seeds within each subject first, then compute the paired ΔAUC. Report a subject-bootstrap 95% CI and a Wilcoxon signed-rank p-value. **No n = folds × seeds tests** (B12).
4. **No threshold tuned on test data.** Use threshold-free metrics, operating points at 0.5, and LOO-thresholds (§10.3). Label any oracle τ "optimistic".
5. **Correct for multiple comparisons:** Holm across H1–H6.
6. **Report the minimum detectable effect** (§11.5) next to every Δ. A result below it is "inconclusive", **not "no effect"**.
7. **Report feature-quality numbers alongside the metrics:**
   - MediaPipe detection rate.
   - Per-joint depth validity.
   - Dropped frames.
   - Person pixel height.
   - Clip duration and fps. A fixed 30 fps should remove the B10 confound; confirm it.

### 11.3 Direct measurement of MediaPipe z quality (the question the phone data can't answer)

On every Kinect take, compare MediaPipe's per-frame, hip-relative z (converted with the MediaPipe scale convention) against Kinect depth z (hip-relative, metres), per joint:
- Bland–Altman bias and limits of agreement, after fitting a per-take linear scale, because MediaPipe z has no metric unit.
- Pearson r of the time series.
- Test-retest ICC across takes.

**This measures how wrong the current z is on your subjects.** It needs no model training.

### 11.4 Face-swap test (deepfake side)

- **Make ≥ 20 clips per sensor** from **probe takes only** (never enrolment takes; this fixes B6), with the same FaceFusion settings.
  - At 3.5 m a Kinect face is ~35 px tall [E]. FaceFusion or InsightFace may fail to detect or swap it. If so, record that; it is a finding about the Kinect.
- **Score each clip with its own sensor's model and templates:**
  - Rejection rate of the claimed identity, with a Clopper–Pearson 95% CI.
  - Rank of the true body source.
  - Number of *other* identities scoring above τ, which is the Teja problem in §3.5.
- **Run `verify_gait_preservation.py` only after fix C9,** against the exact source take.
- **Report face-swap verdicts as "claimed identity rejected", not as "deepfake detected"** (B25).

### 11.5 Sample size and power (13 subjects)

- **Run-noise floor, measured.** Across seeds of the same Raw model, the per-subject AUC difference has SD 3.09 points. With a paired test, n = 13, α = 0.05 and power 0.8, the minimum detectable ΔAUC is:

  | Seeds per arm | MDE |
  |---|---|
  | 1 | **2.62 points** |
  | 3 | **1.51 points** |
  | 5 | **1.17 points** |

  This counts run noise only, so the real MDE is larger.
- **Between-subject spread:** the SD of per-subject AUC is 2.66 points (Raw, seed-averaged).
- **P2 test set:** with 4+4 takes, F1/S1 as enrolment, and originals only, there are 6 probes × 13 subjects = 78 genuine and 936 impostor pairs per session. That is small. **Per-subject AUCs will be coarse**: each rests on 6 genuine scores.
- **Plausible effect sizes** (e.g. H1) are of the same order as the MDE. **An inconclusive outcome is a real possibility.**
- **The biggest lever is more subjects (≥ 30),** not more augmentation. Augmentation multiplies sequences, not gaits.

### 11.6 What the data can and cannot support

**Can support:**
- For these subjects, in this room, with this pipeline: whether MediaPipe-on-Kinect-RGB verifies identity better, worse, or indistinguishably from a phone.
- Which of geometry, sensor and depth drives any difference.
- How accurate MediaPipe's z is against Kinect depth.

**Cannot support:**
- Population-level claims.
- Claims about other depth sensors (Kinect v2, Azure Kinect, RealSense).
- Claims about outdoor use (structured light fails in sunlight).
- Any claim that depth improves *deepfake* detection of recorded RGB videos.
- Claims about other face-swap generators.

---

## 12. What can be tested now, without the camera

**Every result from this section is a PROXY on the existing phone data. It is NOT a Kinect result, and must be labelled that way in any report.**

Outputs go to `outputs_kinect/proxy/<name>/`. Inputs are read-only.

| ID | Experiment | Answers | Needs | Est. cost |
|---|---|---|---|---|
| X1 | `loocv_p2.py run` on `data/gait_features/gait_features.pkl`, enrolment F1,S1, `--train_aug`, 5 seeds | An honest phone baseline. Its gap to the ablation's Raw number (94.25, legacy templates) bounds the combined effect of B1+B3+B26+random negatives. | C5 | ~30–65 min GPU [E: 13 folds × 5 seeds × 27–59 s per Raw run] |
| X2 | X1 with `--drop_z` | H4 proxy: does the current model need MediaPipe z at all? If not, depth replacing z (A3) is unlikely to matter, and only metric scale (A4) could. | C5 | same as X1 |
| X3 | Re-extract the 66 originals at Kinect geometry (§9.4 `phone_kres` transform) and at 30 fps (drop every 2nd frame of the ~60 fps clips); P2 without augmentation, native vs kres | H2 proxy: does Kinect-like geometry alone change the results? | C4, C5; ~17k frames × ~20 ms ≈ 6 min CPU per version [E] | < 1 h |
| X4 | Re-extract the originals with a fixed time base (30 fps, middle 3 s of each walk) before the 60-step resampling | Size of the fps/duration confound (B10) | as X3 | < 1 h |
| X5 | Re-measure B7 on all 132 rotate videos (ECC frame-to-frame rotation), and count detections per augmentation | Confirms the augmentation defect across the dataset | Scratch script | minutes |

**Caveats:**
- The old phone data has only 5–6 takes per subject. With F1/S1 as enrolment that leaves 3–4 probes per subject, about **39–52 genuine test pairs in total**, so X1–X4 have very wide CIs.
- Without augmentation, X3/X4 train on about 36–48 sequences per fold.

---

## 13. Costs, risks, open questions

### 13.1 Cost and time ([E] = estimate)

| Step | Cost |
|---|---|
| Adapter | Required for the Xbox 360 unit. Widely sold, Microsoft part 1429 or third-party. Price not verified. |
| Driver setup | 1–3 h [E]. High risk of Stage 1 failing on Windows 11 (§6.3). |
| Code changes C1–C6 + checks | ~1 day of work [E] |
| Pilot (2 subjects) + go/no-go | ½ day [E] |
| Recording | ~20–30 min per subject per session [E] → ~5–7 h per session for 13 subjects |
| Conversion + trimming review | ~2–3 min per take [E] → 208 takes (13 × 2 × 8) ≈ 8 h, mostly review |
| Augmentation | Not measured for the phone data. Kinect frames have 6.75× fewer pixels than phone frames [computed]. Hours, not days [E]. |
| MediaPipe extraction | Measured 19.6–22.3 ms/frame plus decode. 208 takes × ~150 frames × 16 versions ≈ 500k frames ≈ **2.7–3 h CPU per dataset** [E]. There are three datasets (Kinect, phone, phone_kres), plus the depth arms (no augmentation, ~1/16 of that). |
| Training / evaluation | `train.py` ≈ 2 min [E, from log timings]. Legacy LOOCV ≈ 20 min (measured 1,228 s on the phone data). P2 ≈ 7–13 min per arm per seed [E] → 7 arms × 5 seeds × 1–2 designs ≈ **4–15 h GPU**. |
| Face-swap clips | Manual GUI; not measured |
| Disk | Phone augmentation was 25.8 GB for 66 clips. Scaling by take count, **~80 GB for 208 new phone takes plus augmentation** [E], plus tens of GB of Kinect masters. **Only 137.5 GB is free on C:**. Plan external storage, or drop phone augmentation. |

### 13.2 Ask the user BEFORE running

1. The **Kinect model number** (1414 or 1473) on the base. Is the **power/USB adapter** on hand?
2. **Which machine and OS will record?** This laptop is Windows 11, where SDK 1.8 has reported failures. Is a Windows 10 PC or a Linux machine available?
3. **Subjects:** the same 13 people, under the same names? Any additional people? Are **two sessions on different days** possible? Does the consent cover depth data and face-swaps?
4. **Permission to install:** SDK 1.8 + Toolkit, *or* the libfreenect toolchain (CMake, MSVC Build Tools, libusb, Zadig), *or* a Linux setup. And `pip install -e .` vs using `PYTHONPATH`.
5. **Permission for the code changes C1–C10** on a branch, and whether to commit.
6. **Simultaneous phone recording:** which phone? Can it lock 30 fps?
7. **The "latest re-run" model.** Only the 2026-08-28 ablation artefacts and the 2026-02-08 checkpoints were found. Where is the newer model, if one exists?
8. **FaceFusion location,** and willingness to generate ≥ 20 new clips per sensor.
9. **Disk budget** (137.5 GB free), and where masters should live.
10. **Whether to address the published-claims issues separately** (B1, B4, B5, B6, B10, §4). They are out of scope here, but they affect the paper and the DataPort page.

### 13.3 Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| SDK 1.8 won't detect the sensor on Windows 11 | High (§6.3) | Stage 2/3 fallback (§9.1) |
| A 1473 needs audio firmware for motor/LED under libfreenect | Medium [S8] | If motor control is unavailable, don't force the motor. Leave the tilt, check it with a spirit level, and record it. The camera streams may still work (verify in the pilot). |
| Frontal full-body **depth** window is only ~2.7–4 m | Certain | Report frontal depth arms separately; RGB arms use the whole path |
| Invalid depth at heels and feet (dark/leather shoes, floor contact, shadow) | Likely | Log per-joint validity; flag A3/A4 |
| Kinect skeleton weak on side views | Documented [S3] | A5 is optional and last |
| Kinect RGB quality lowers MediaPipe detection | Unknown | The pilot go/no-go measures it |
| Study underpowered at 13 subjects | Likely | Pre-register; report MDE; recruit more |
| Registration offset between depth and RGB | Medium | Overlay check in the pilot; 5×5 median lookup |
| Accidental overwrite of phone artefacts | Low if §9.3 is followed | Explicit paths; check `outputs\` mtimes after each step |
| Silent colour swap or dropped frames | Medium | Converter asserts; colour check |
| Face-swap tools fail on ~35 px Kinect faces | Medium [E] | Record the failure as a result |
| Disk exhaustion | Medium | External storage; delete only *regenerable* derived files, and only with the user's approval |

---

## 14. Checklist

**Before any hardware work**
- [ ] Questions §13.2 1–10 answered and recorded in `data_kinect/SETUP.md`
- [ ] §10.1 environment works: `train.py --help` prints usage; smoke test passes
- [ ] Branch created; C1, C2, C3, C10 applied, each with its check passing
- [ ] C5 `loocv_p2.py` written; `--quick` run on the phone pickle passes the leakage asserts
- [ ] (Optional, no camera) proxies X1–X5 run and labelled PROXY

**Hardware and pilot**
- [ ] Kinect model number recorded; adapter connected
- [ ] Driver stage that worked recorded (§9.1)
- [ ] C4 converter (and C7 if needed) written and checked
- [ ] Room set up per §9.2; tape lines measured; Kinect at ~1.0 m, tilt 0°; phone mounted above it, portrait, 30 fps
- [ ] Checkerboard intrinsics (< 1 px reprojection error) and flat-board depth check saved
- [ ] Pilot of 2 subjects passes the go/no-go table (§10.4 Step 1)
- [ ] C6 RGB-D extractor written and checked on the pilot; depth-valid fraction per joint recorded

**Recording and conversion**
- [ ] All subjects recorded, 4F + 4S per session, clap at every take, `takes.csv` complete (direction, shoes, notes)
- [ ] Second session recorded (if possible)
- [ ] All takes converted, trimmed and exported; counts and resolutions checked; masters untouched

**Goal B (new model)**
- [ ] Kinect RGB augmented (fixed C3 code)
- [ ] Features extracted to `data_kinect/gait_features/kinect_rgb_v1/gait_features.pkl`; MediaPipe confirmed; layout check passes
- [ ] Enrolled file created next to the features
- [ ] `gait-kinect-rgb-v1` trained into `outputs_kinect/gait-kinect-rgb-v1/`; `SPLIT.txt` saved; `outputs/checkpoints` untouched (mtime check)
- [ ] Legacy LOOCV run for Kinect **and** for the new phone data with identical arguments

**Goal A (science)**
- [ ] Datasets extracted: `phone_v1`, `phone_kres_v1`, `kinect_rgb_noaug_v1`, `kinect_rgbd_depthz_v1`, `kinect_rgbd_metric_v1`
- [ ] P2 run for every arm × 5 seeds (within-session, plus cross-session if available)
- [ ] Comparisons H1–H8 with subject-level bootstrap CIs, Wilcoxon, Holm, and MDE
- [ ] MediaPipe-z vs depth agreement (§11.3)
- [ ] Face-swap test (§11.4) with ≥ 20 clips per sensor from probe takes only
- [ ] `outputs_kinect/REPORT.md`: every number cites a JSON; proxies labelled; "inconclusive" used where Δ < MDE

**Finally**
- [ ] Phone artefacts verified unchanged (§9.3 list: mtimes and sizes)

---

## 15. Sources

Tags as in §6: [P] = primary, checked in this session; [S] = secondary.

- **S1** [P] Microsoft, *Kinect for Windows Sensor Components and Specifications* (SDK 1.5–1.8): FOV 43°×57°, tilt ±27°, 30 fps, RGB 1280×960. https://learn.microsoft.com/en-us/previous-versions/windows/kinect-1.8/jj131033(v=ieb.10)
- **S2** [P] Microsoft, *DepthImageFormat Enumeration* (640×480, 320×240, 80×60 @ 30 fps). https://learn.microsoft.com/en-us/previous-versions/windows/kinect-1.8/hh855229(v=ieb.10)
  - Note: the SDK-1.8 archive pages for `JointType` (hh855342) and `ColorImageFormat` (hh855203) show **v2** content and should not be trusted for v1.
- **S3** [P] Microsoft, *Skeletal Tracking* (default 0.8–4.0 m, practical 1.2–3.5 m; sideways poses; IR interference). https://learn.microsoft.com/en-us/previous-versions/windows/kinect-1.8/hh973074(v=ieb.10)
- **S4** [P] Microsoft Robotics, *Kinect Sensor* (range; "Near Mode is not supported" with an Xbox Kinect; no direct sunlight; sees through glass; metres; tilt −27..+27). https://learn.microsoft.com/en-us/previous-versions/microsoft-robotics/hh438998(v=msdn.10)
- **S5** [S] NI mate forum, *Kinect for Windows 1 / Kinect for Xbox 360 (models 1414 & 1473)*. https://forum.ni-mate.com/t/documentation-kinect-for-windows-1-kinect-for-xbox-360-models-1414-1473/288 ; Cycling '74 forum. https://cycling74.com/forums/alle-the-different-kinect-models-which-to-get
- **S6** [S] Adapter listings (12 V proprietary connector plus USB; parts 1429/1432), e.g. https://www.endlesscables.com/product-page/xbox-360-kinect-ac-adapter-12v-12w-proprietary-connector
- **S7** [P] Khoshelham K., Oude Elberink S. (2012), *Accuracy and Resolution of Kinect Depth Data for Indoor Mapping Applications*, Sensors 12(2):1437–1454. https://pmc.ncbi.nlm.nih.gov/articles/PMC3304120/
- **S8** [P] OpenKinect libfreenect README and `fakenect/record.c`. https://github.com/OpenKinect/libfreenect/blob/master/README.md , https://github.com/OpenKinect/libfreenect/blob/master/fakenect/record.c ; registration header [S]: https://github.com/OpenKinect/libfreenect/blob/master/include/libfreenect_registration.h
- **S9** [P] `NuiSkeleton.h` (Microsoft Kinect SDK v1 header, copy): `NUI_SKELETON_POSITION_INDEX`, 20 joints. https://github.com/neilmendoza/ofxKinectSdk/blob/master/libs/kinect/include/NuiSkeleton.h
- **S10** [P] Andersson V.O., Araújo R.M. (2015), *Person Identification Using Anthropometric and Gait Data from Kinect Sensor*, AAAI-15, pp. 425–431 (PDF read). https://ojs.aaai.org/index.php/AAAI/article/download/9212/9071 ; dataset: https://github.com/Vortander/KinectGait
- **S11** [P] Microsoft Download Center: Kinect for Windows SDK v1.8 (id 40278), Developer Toolkit v1.8 (40276), Runtime v1.8 (40277). https://www.microsoft.com/en-us/download/details.aspx?id=40278
- **S12** [S] OpenKinect libfreenect2 (Kinect v2 only). https://github.com/OpenKinect/libfreenect2
- **S13** [P] Microsoft Q&A, *Windows 11 is not working with kinect* (2022–2025, unresolved). https://learn.microsoft.com/en-us/answers/questions/1129421/windows-11-is-not-working-with-kinect
- **S14** [P] Naeemabadi M. et al. (2018), *Investigating the impact of a motion capture system on Microsoft Kinect v2 recordings*, PLOS One 13(9):e0204052 (quotes its literature review on v1 sunlight and material failures). https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0204052
- **S15** [P] Google, MediaPipe Pose Landmarker (landmarker input 256×256; detector 224×224; landmarks 23–32) and the Python guide (x/y normalized by width/height; z hip-midpoint origin; world landmarks in metres; VIDEO mode uses tracking). https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker , https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python
- **S16** [P, abstract] Hofmann M. et al. (2014), *The TUM Gait from Audio, Image and Depth (GAID) database*, J. Visual Communication and Image Representation (via Semantic Scholar API); https://www.ce.cit.tum.de/en/mmk/misc/tum-gaid-database/
- **S17** [title/venue verified] Atoum Y., Liu Y., Jourabloo A., Liu X. (2017), *Face anti-spoofing using patch and depth-based CNNs*, IJCB. https://cvlab.cse.msu.edu/face-anti-spoofing-using-patch-and-depth-based-cnns.html
- **S18** [title/venue verified; numbers from a search summary] Pfister A., West A., Bronner S., Noah J. (2014), *Comparative abilities of Microsoft Kinect and Vicon 3D motion capture for gait analysis*, J. Med. Eng. Technol. 38(5), doi:10.3109/03091902.2014.909540
- **S19** [title/venue located; numbers not verified] Dill S. et al. (2024), *Accuracy Evaluation of 3D Pose Reconstruction Algorithms Through Stereo Camera Information Fusion for Physical Exercises with MediaPipe Pose*, Sensors 24(23):7772. https://pubmed.ncbi.nlm.nih.gov/39686309/

---

## Appendix: audit provenance

**Read in full**
- Code: `utils/pose_extraction.py`, `utils/pose_extraction_gpu.py`, `utils/data_loader.py`, `utils/gradcam.py`, `utils/logger.py`, `models/full_pipeline.py`, `models/gait_encoder.py`, `models/temporal_model.py`
- Scripts: `augment_videos.py`, `extract_faces.py`, `preprocess_videos.py`, `enroll_identities.py`, `train.py`, `evaluate.py`, `ablation_loocv.py`, `run_gradcam.py`, `inference.py`, `verify_gait_preservation.py`, `run_pipeline.py`
- Tests, setup and config: `tests/smoke_test.py`, `ieee_scripts/*`, `pyproject.toml`, `requirements.txt`, CI
- Docs: README, CONTRIBUTING, all of `DOCUMENTATION/`, `.github/instructions/rules.instructions.md`, `research/claude_research_verification.md`, `figures/README.md`
- Paper: `paper/deepfake_paper.tex` lines 280–1490 (method, experiments, results, discussion, limitations)
- Results and logs: every results JSON, `all_tables.txt`, and the evaluate, train and ablation logs

**Skimmed or grepped only** (paths, constants, dimensions)
- `models/identity_verifier.py` and `utils/visualization.py` (auxiliary; not on any decision path)
- `ablation_study.py`, `_verify_ablation_fix.py`, `generate_paper_figures.py`, `visualize_keypoints.py`, `scripts/generate_figures/*`
- `figures/GEMINI_PROMPTS.md`
- Paper intro, related work and references (lines 1–279, 1490–1524)
- Figure images were not inspected.

**Read-only computations, run from scratchpad scripts with `venv\Scripts\python.exe`**
- Recomputed every metric from the JSONs.
- Seed-averaged significance tests and MDE.
- Pickle and enrolment inspection (duplicates, template composition, fps, frame counts, detection ratios by augmentation, per-axis jitter).
- Video metadata.
- Frame-to-frame rotation of augmented videos (ECC).
- albumentations consistency, with and without `ReplayCompose`.
- MediaPipe timing at two resolutions.
- Environment and driver inventory.

**No repository file was modified** apart from creating this document.

**Prior-session note.** A 2026-09-18/19 session planned a similar study for an Intel RealSense D455. Its list of evaluation issues (called L1–L9 there) was used only as leads. Each item kept here was re-verified; the numbering here is new (B1–B26). One of its claims ("MediaPipe z 2.6–2.9× noisier") was **not** reproduced (§7).
