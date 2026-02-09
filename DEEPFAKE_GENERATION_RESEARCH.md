# Deepfake Generation Tools Research (2025–2026)
## Comprehensive Guide for Gait-Based Deepfake Detection Testing

**Research Date:** February 9, 2026
**Project:** Gait-Based Deepfake Detection (Skeletal Pose Analysis)
**Goal:** Generate 10–15 face-swap walking videos to test hypothesis that deepfakes preserve body/gait while changing face

---

## 0. High-Level Recommendation

For generating 10–15 face-swap walking videos (10–30 s, 720p–1080p) with *unchanged body/gait* on an RTX 3050 (6 GB), the best practical stack in 2025–26 is:

1. **FaceFusion using InsightFace `inswapper_128_fp16`** as the primary model, optionally trying `ghost_1_256` or `simswap_256` on a few clips.
2. **A minimal Python pipeline with the InsightFace `inswapper_128.onnx` model directly** (via `insightface` or `haofanwang/inswapper`) for maximum control and reproducibility.
3. **SimSwap** (original repo or via FaceFusion's SimSwap model) to diversify the generation methods (and align with datasets like Celeb‑DF++ that explicitly use SimSwap and InSwapper/GHOST).
4. **DeepFaceLab** only if willing to invest time in per‑pair training (6 GB VRAM is on the low side).

All of these, when configured with *face‑only masks*, preserve the original body and hence gait essentially perfectly by design: only a tight face region is synthesized and composited back frame‑wise; the rest of the frame is untouched.

---

## 1. What Matters for Your Use Case

Key constraints for your gait‑based detection study:

- **Full‑body walking**: face is relatively small in the frame; tools must handle low‑resolution faces robustly.
- **Gait must be untouched**: the body's 2D pose trajectories must remain pixel‑identical (modulo codec interpolation).
- **6 GB VRAM**: excludes heavy high‑res GANs or very large StyleGAN‑based models at 4K.
- **Academic/non‑commercial**: acceptable licenses include GPL, MIT, Apache, research‑only weights.
- **Offline, not live**: you can trade speed for quality, but not the other way around.

Modern one‑shot face‑swappers (InsightFace `inswapper_128`, SimSwap, Ghost, UniFace) all use the same basic pipeline: detect and align faces, encode source identity, generate a swapped face crop, then paste it back into each original frame. This inherently keeps the *body motion* the same unless you deliberately use wider "head/whole‑face" masks.

---

## 2. Ranked Tool Recommendations (Top 3–4)

### 2.1 FaceFusion (with `inswapper_128_fp16` as default) ✅ RECOMMENDED PRIMARY

**What it is**
FaceFusion is an actively maintained, Windows‑friendly face manipulation platform (CLI+GUI) with a Windows installer and batch mode. It wraps multiple ONNX models, including InsightFace `inswapper_128(_fp16)`, several Ghost models, SimSwap, UniFace, etc., and provides robust video handling and VRAM‑aware processing options.

**GitHub:** https://github.com/facefusion/facefusion

**Why it fits your project**

- **Full‑body/walking quality**
  - FaceFusion detects faces per frame and crops them to the resolution the model expects (e.g., 128×128 for InSwapper, 256×256 for Ghost/SimSwap) before swapping.
  - Because InSwapper was trained at *128×128* resolution explicitly, it handles *small faces* especially well; any upscaling from ~60–100 px to 128 px doesn't hurt it as much as 256‑based models.
  - Community experience (including a 2025 comparative deep‑dive) still finds `inswapper_128(_fp16)` the most robust model for video swaps in FaceFusion, with Ghost and SimSwap generally trailing in stability and identity fidelity.

- **Body/gait preservation**
  - FaceFusion swaps only the detected facial region and composites it back; the rest of the frame, including torso, limbs and background, is unchanged.
  - You can explicitly pick face‑only masks in the UI; avoid "whole head" or extended masks to guarantee that shoulders and neck remain untouched.

- **VRAM and speed on 6 GB (RTX 3050)**
  - `inswapper_128_fp16` is optimized for low VRAM and real‑time use; the model is ~130M parameters but compressed to FP16, and operates at 128×128, so VRAM is dominated by video buffers rather than the network.
  - FaceFusion exposes `--video-memory-strategy {strict,moderate,tolerant}` and other flags to keep VRAM within 4–5 GB.
  - For 720p–1080p 25/30 fps walking clips, offline processing at several fps is realistic on your GPU.

- **Ease of use**
  - Windows installer, plus CLI (`python facefusion.py run ...`) for reproducible experiments.
  - Built‑in job system with headless/batch modes useful for running multiple subject pairs.

- **License / academic use / maintenance**
  - FaceFusion itself is open‑source and long used under MIT, though newer builds transition to a more restrictive OpenRAIL‑style license to discourage harmful uses. Academic research falls under allowed use.
  - It is actively maintained; latest release line is 3.2.0–3.5.x with 2025 commits.
  - The bundled InsightFace weights (`inswapper_128`) are licensed for **non‑commercial research use only**, which matches your course project.

**Minimal pipeline for you with FaceFusion**

```bash
# Install
pip install facefusion

# GUI (easier for interactive work)
facefusion

# CLI (for batch processing)
python facefusion.py run \
  --target-path data/videos/PersonA_walking.mp4 \
  --source-path data/deepfake/faces/PersonB.jpg \
  --output-path data/deepfake/PersonA_body_PersonB_face.mp4 \
  --frame-processor face_swapper face_enhancer \
  --video-memory-strategy strict
```

You can optionally re‑run a few sequences with `ghost_1_256` and `simswap_256` to introduce method diversity while still preserving gait.

---

### 2.2 Direct InsightFace / `inswapper_128` Python Pipeline ✅ RESEARCH-GRADE

**What it is**
The official InsightFace project provides the `inswapper_128.onnx` model with a minimal example, and haofanwang/inswapper wraps it as a one‑click script. This is the underlying model behind Roop and many other tools.

**GitHub:** https://github.com/haofanwang/inswapper
**Official InsightFace:** https://github.com/deepinsight/insightface

**Advantages for your study**

- **Maximum control / reproducibility**
  - You can write a Python script that:
    - Loads each frame.
    - Runs face detection + alignment via `insightface`.
    - Runs the `INSwapper` on the face crop only.
    - Pastes the swapped face back into the full‑resolution frame.
  - That guarantees you never touch non‑face pixels, which is ideal for asserting that gait remains identical.

- **Small‑face robustness / video quality**
  - As above, InSwapper is designed for 128×128 crops and is widely regarded as the best "general" one‑shot face swapper, especially at video resolutions, often outperforming SimSwap and Ghost in identity fidelity and stability.
  - You can add optional post‑processing (e.g., CodeFormer or GFPGAN) only on the face region if you need more detail.

- **VRAM / speed**
  - On your RTX 3050, processing frame‑wise with a single face per frame and FP16 inference (via `onnxruntime-gpu`) will comfortably fit into 6 GB.
  - Because you control batching, you can set batch size = 1 to minimize peak memory.

- **License / use**
  - InsightFace code is MIT‑licensed, with models for non‑commercial research only.
  - You can ship scripts and mention model version in your paper for reproducibility.

**Sample Python pseudocode:**

```python
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
import cv2
import numpy as np

# Initialize
analyzer = FaceAnalysis(providers=['CUDAExecutionProvider'])
swapper = get_model('inswapper_128', providers=['CUDAExecutionProvider'])

# Load source face (extract from image or video frame)
source_img = cv2.imread('source_face.jpg')
source_faces = analyzer.get(source_img)
source_face = source_faces[0]

# Process video
cap = cv2.VideoCapture('target_walking_video.mp4')
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect face in frame
    faces = analyzer.get(frame)
    for face in faces:
        # Swap
        frame = swapper.get(frame, face, source_face, paste_back=True)

    # Write output frame
    ...
```

Given your background, this is probably the cleanest route if you want to publish a methodologically tight gait‑deepfake dataset.

---

### 2.3 SimSwap (standalone or via FaceFusion `simswap_256`) ✅ FOR DIVERSITY

**What it is**
SimSwap is an academic arbitrary face‑swapping model (ACM MM 2020) that injects source identity features into a UNet‑like generator and is trained to preserve background and non‑face features via a weak feature‑matching loss.

**GitHub:** https://github.com/neuralchen/SimSwap

**Why it is worth including**

- **Relevance to current datasets**
  - Celeb‑DF++ explicitly includes SimSwap and InSwapper, GHOST, HifiFace, UniFace, MobileFaceSwap, BlendFace, DaGAN, HyperReenact, LivePortrait, SadTalker, etc. in its face‑swap scenario, to reflect "realistic" modern deepfakes.
  - Using SimSwap gives you better comparability with literature that evaluates detectors on Celeb‑DF++.

- **Visual properties**
  - SimSwap tends to produce very seamless composites: lighting, skin tone and background are strongly preserved from the target, which makes the swap plausible to casual viewers.
  - Identity transfer is sometimes slightly weaker than InSwapper (results can look like "target with heavy makeup"); but for your purpose—visual plausibility plus body mismatch—that is acceptable.

- **Full‑body / walking**
  - Architecturally it is still a face‑crop model; it does not modify the body region.
  - As with FaceFusion/InSwapper, the body and gait are untouched; only the face patch is synthesized and pasted back.

- **VRAM / usage**
  - The standard 256×256 model is moderate size (~50–60M parameters); it should run on 6 GB VRAM for single‑face videos.
  - Integration options:
    - Use SimSwap via FaceFusion's `simswap_256` model for convenience.
    - Or run the original PyTorch repo (`neuralchen/SimSwap`) on Windows with CUDA.

For your dataset you might, for example, generate half the fakes with InSwapper and half with SimSwap, documenting the method label per clip.

---

### 2.4 DeepFaceLab (for "classic", per-pair training) ⚠️ OPTIONAL / LOW PRIORITY

**What it is**
DeepFaceLab is the traditional high‑quality deepfake framework used in many early detection papers; it gives you full control over training per‑pair autoencoders, masks (face/head/whole), and manual curation.

**GitHub:** https://github.com/iperov/DeepFaceLab

**Pros for research**

- **Historical compatibility**
  - FaceForensics++ used Deepfakes/FaceSwap/NeuralTextures and related methods to build its dataset. DeepFaceLab is in the same lineage and is often used to reproduce that style of manipulation.
- **Quality**
  - When properly trained, DeepFaceLab can produce extremely natural results, including profile views and occlusions, especially with XSeg masks.

**Limitations for your scenario**

- **VRAM & complexity**
  - 6 GB VRAM is just about usable if you choose small models, low resolution, and tiny batch sizes; but expect long training times.
  - You must train a separate model per source–target pair, which is heavy for 10–15 video pairs.
- **Maintenance status**
  - The original repo is now archived and read‑only as of Nov 2024. Community forks continue, but it is no longer a "living" project.

Given your constraints, DeepFaceLab is best used if you specifically want one or two "gold‑standard" deepfakes to visually benchmark against. For the bulk of your dataset, the one‑shot approaches above are far more efficient.

---

## 3. Other Tools You Mentioned (and Why They Are Secondary)

### 3.1 Roop
- Roop is a one‑click face‑swap tool using InSwapper under the hood; original project discontinued but still works on Windows.
- Pros: dead simple GUI; good for ad‑hoc experiments.
- Cons: less configurable than FaceFusion or a custom pipeline (e.g., no fine control over masks, VRAM strategies, or model choice), and not actively developed.
- For a serious academic dataset, it is better to use InsightFace directly or via FaceFusion rather than rely on Roop.

### 3.2 DeepFaceLive
- Real‑time face swap application oriented to streaming/video calls.
- It *can* run offline on prerecorded videos, and it preserves body motion (only the face area is rendered).
- However, quality and stability tend to be lower than offline batch‑processed frameworks; tuning is more complex.
- Useful if you want to explore real‑time attacks, but not necessary for your core gait dataset.

### 3.3 Ghost / Sber-swap standalone, Ghost 2.0
- The original Sber‑swap (GHOST) repository is available and can swap faces in images and videos via CLI.
- Ghost 2.0 (2025) targets *head swapping*, using a reenactment‐then‑blending pipeline that handles whole head shape, hair, and skin color.
- These are powerful, but installation is Linux‑oriented and more complex (multi‑repo dependencies, 3D components). For your Windows 11 laptop, the **FaceFusion‑packaged Ghost models (`ghost_1_256`, etc.)** are easier to use.

---

## 4. Face Reenactment and Full‑Body Manipulation (Secondary for You)

### 4.1 Face Reenactment (expression transfer while keeping body)

Most modern reenactment tools focus on talking heads, not small faces in full‑body walking:

- **First Order Motion Model (FOMM)**: self‑supervised keypoint‑based animation; can transfer motion from a source driving video to a target image, including full bodies (originally demonstrated on fashion datasets). Works in principle for walking motion transfer.
- **UniFace**: unified face‑swap/reenactment model in FaceFusion that disentangles identity and attributes. Still face‑crop‑based; body in the original frame is preserved when used for swapping.
- **HyperReenact, LivePortrait, SadTalker, etc.**: widely used for talking‑head lip‑sync and expression transfer. They assume relatively large facial regions; performance on full‑body walking scenes with tiny faces is poor.

For your *walking* data, these are best treated as:
- Tools for **upper‑body / mid‑shot reenactment** (if you ever want another modality), not for long‑shot full‑body gait videos.
- Potentially, you could crop the upper body, perform reenactment, and then composite back into the full frame. That would still preserve gait if the crop is rigidly re‑inserted, but it requires custom compositing logic.

### 4.2 Full‑Body Deepfake / Motion Transfer (Synthetic Gait)

There is a separate line of research on full‑body reenactment:

- **First Order Motion Model** already trains on fashion/full‑body datasets and handles large pose changes.
- **Reenact Anything (2025)** proposes motion‑textual inversion on top of image‑to‑video models to transfer motion semantics from a reference video to arbitrary still images, including full‑body and inanimate objects.
- **OmniHuman‑1** (2025 demos) shows full‑body animation driven by input video, including synced hand and body motion.

However:
- These frameworks *replace* both face and body motion with a generated sequence; they no longer preserve the original subject's gait.
- They are great for *attacking* your system with "entirely synthetic human motion", but **not** appropriate for validating your "face‑swap preserves body" hypothesis.

For your core study, keep them as a **separate threat model**: test whether your gait model flags these fully synthetic motions as "no enrolled identity".

---

## 5. AI Video Generators (Kling, Runway, Pika, etc.) and Gait Realism

Text/image‑to‑video systems (Kling, Runway Gen‑2, Pika Labs) have advanced a lot:

- **Kling 2.6 "Motion Control"** markets precise, physics‑aware full‑body motion and detailed limb trajectories; it can follow drawn motion paths and maintain coherent full‑body motion.
- **Runway Gen‑2 and Pika** support text‑to‑video, image‑to‑video and video‑to‑video remixing, generating plausible walking, running, dancing, etc.

From a **gait biometrics** angle:

- Their motion is **synthetic**, sampled from the model's learned prior; you cannot assume it corresponds to any enrolled identity's gait.
- Even when using video‑to‑video "motion control" (Kling Motion Control, etc.), the generated body can deviate from exact biomechanical trajectories; camera and body may be co‑optimized for aesthetics rather than physical correctness.
- These tools are also **closed and cloud‑hosted**, not ideal for a reproducible academic corpus (and not guaranteed free for research).

For your work, they are most useful as a **separate class of negatives**:
- "Purely AI‑generated walking sequences of a nominal 'person X'."
- Your gait model should ideally reject them as not matching any enrolled identity—exactly the hypothesis you mention.

But they are *not* the right tools for generating *controlled* face‑swap deepfakes where ground‑truth gait is known.

---

## 6. How Existing Deepfake Detection Datasets Generate Fakes

Understanding dataset practices can guide your own test set.

### 6.1 FaceForensics++

- 1,000 original YouTube videos (mostly frontal faces) are manipulated with **four automated methods**: classic Deepfakes, Face2Face, FaceSwap, and NeuralTextures.
- Each video exists at multiple **compression levels**:
  - `c0`: original codec;
  - `c23` and `c40`: H.264 with CRF 23 (moderate) and CRF 40 (strong), corresponding to medium and heavy compression.
- The focus is **upper‑body / talking‑head**, not full‑body walking, but the important pattern is:
  - **Multiple independent manipulation methods** per real video.
  - **Multiple compression settings** to test robustness.

**Reference:** https://github.com/ondyari/FaceForensics

### 6.2 Celeb-DF / Celeb-DF++

- Celeb‑DF originally used an improved DeepFake generation pipeline to produce 5,639 high‑quality celebrity interview deepfakes, avoiding obvious artifacts in early datasets.
- Celeb‑DF++ (2024) systematically expands this with **three scenarios**—Face‑swap (FS), Face‑reenactment (FR), and Talking‑face (TF)—and **22 DeepFake methods** across them, including SimSwap, InSwapper, GHOST, UniFace, MobileFaceSwap, BlendFace, DaGAN, HyperReenact, LivePortrait, SadTalker, etc.
- For FS and FR, they generate >13k deepfakes from 2,000 random identity pairs; for TF, >20k lip‑synced videos.

Key takeaways:
- It is now *standard* to mix **many distinct generation algorithms** in one dataset.
- Each method is applied across **many identity pairs** with varied scenes, but typically in talking‑head settings.

**Reference:** Celeb-DF++ paper (arxiv preprint available)

### 6.3 DFDC (Facebook / Meta)

- DFDC uses 3,426 paid actors to record diverse real videos, then applies **several DeepFake, GAN‑based, and non‑learned** face‑swap methods to produce over 100,000 manipulated clips.
- They deliberately include variation in:
  - Lighting, backgrounds, occlusions.
  - Compression artifacts, re‑encodings, and social‑media–like degradation.

### 6.4 WildDeepfake

- WildDeepfake does **not** construct deepfakes; it *collects* 707 real‑world deepfake videos from the internet to expose detection models to in‑the‑wild content.
- It underlines that models trained only on single‑method, clean lab datasets often fail in the wild.

**Reference:** https://github.com/OpenTAI/wild-deepfake

---

## 7. Best Practices for Your Gait-Deepfake Test Dataset

Given the above, a practical, publishable protocol for your project could be:

### 7.1 Subjects and Capture

- Recruit, say, 6–10 participants (university peers), record **2–3 walking clips per person**, indoor and outdoor:
  - Different speeds (slow/normal/fast).
  - Straight path, slight curves, maybe stairs for a subset (if safe).
- Keep resolution 1080p, 25–30 fps; use a tripod to avoid camera shake confounding gait.

### 7.2 Real vs Fake Splits

From this, construct:

- **Real set**: original walking clips (A on A, B on B, etc.).
- **Face‑swap fakes**: Person A's face on Person B's body for a subset of pairs, using:
  - **Method 1**: FaceFusion + `inswapper_128_fp16`.
  - **Method 2**: FaceFusion + `simswap_256` or Ghost.
  - **Method 3** (optional): DeepFaceLab for 1–2 exemplar pairs.
- Ensure each *body identity* has:
  - At least one *real* video.
  - At least one *deepfake with a different face*.

### 7.3 Processing Settings

- Use face‑only masks; verify visually that shoulders/neck boundaries are not being altered.
- Export **two compression levels**:
  - High quality (CRF ≈ 18–23, similar to FF++ `c23`).
  - Strong compression (CRF ≈ 35–40, like `c40`), to test robustness to artifacts.

### 7.4 Verification of Gait Preservation

For a subset, run your MediaPipe skeleton extractor on:
- The original video.
- The face‑swapped version.

Compute per‑joint L2 distances across all frames in pixel coordinates; they should be near zero (within codec‑induced sub‑pixel noise). This empirically confirms that the body motion is unchanged.

**Python pseudocode:**

```python
import pickle
import numpy as np

# Load original and swapped feature sequences (N_frames × 78)
with open('original_gait_features.pkl', 'rb') as f:
    original = pickle.load(f)

with open('swapped_gait_features.pkl', 'rb') as f:
    swapped = pickle.load(f)

# Compute L2 distance per frame
distances = np.linalg.norm(original - swapped, axis=1)
print(f"Mean L2 distance: {distances.mean():.4f} pixels")
print(f"Std L2 distance: {distances.std():.4f} pixels")
print(f"Max L2 distance: {distances.max():.4f} pixels")

# Should be near-zero (< 1 pixel) if gait is truly preserved
if distances.mean() < 1.0:
    print("✅ Gait preservation verified")
else:
    print("⚠️ Warning: Gait may have been altered")
```

### 7.5 Optional: Synthetic-Motion Negatives

Generate separate clips with Kling / Runway / Pika where a stylized "version of a subject" walks through a scene.

Treat them as "AI‑person with no enrollment"; your model should reject them as unknown identity based on gait.

---

## 8. Proposed Implementation Timeline

| Step | Task | Est. Time | Tool(s) |
|------|------|-----------|---------|
| 1 | Get permission from teacher + verbal consent from 6–10 subjects | 10 min | (Email/Conversation) |
| 2 | Install FaceFusion + verify GPU setup | 15 min | `pip install facefusion` |
| 3 | Prepare 2–3 walking clips per subject (already recording or use existing data) | 5–10 min | Existing videos |
| 4 | Extract target faces (one clear frontal face per subject) | 10 min | MediaPipe or frame extraction |
| 5 | Generate 10–15 face-swapped videos (15–30 min depending on GPU) | 30–45 min | FaceFusion + InSwapper |
| 6 | Optional: Re-generate a few with SimSwap for method diversity | 15–20 min | FaceFusion + SimSwap |
| 7 | Verify gait preservation (run MediaPipe on originals + swaps, compute L2 distances) | 15 min | Python script + MediaPipe |
| 8 | Test inference on deepfakes (expect IDENTITY MISMATCH or SUSPECTED DEEPFAKE) | 10 min | `scripts/inference/inference.py` |
| 9 | Document results in evaluation report | 15 min | Markdown + results table |

**Total estimated time:** ~2–3 hours (mostly GPU processing time)

---

## 9. Summary Recommendation (Concrete Choice List)

Putting it all together, and weighting your constraints:

### 9.1 Primary Tool
**FaceFusion + `inswapper_128_fp16`**

Rationale:
- Best robustness on small faces
- Low VRAM footprint
- High identity fidelity
- Preserves body motion by design
- Actively maintained
- Easy to use on Windows

### 9.2 Research-Grade Pipeline
**Custom InsightFace `INSwapper` Python script**

Rationale:
- Maximum control and auditability over which pixels are changed
- Ideal if you want to release code with the paper

### 9.3 Secondary Method for Diversity
**SimSwap (`simswap_256` via FaceFusion, or original repo)**

Rationale:
- Widely used in modern benchmarks (Celeb‑DF++)
- Good blending results
- Gives you method diversity to test generalization

### 9.4 Optional High-Effort Baseline
**DeepFaceLab**

Rationale:
- Classic, very high‑quality per‑pair deepfakes
- Heavy training and less relevant for a 6 GB laptop
- Useful if you want to compare against "cinema‑grade" swaps

---

## 10. Next Steps

1. **Choose your primary tool:** FaceFusion is recommended for speed and ease; InsightFace pipeline for reproducibility.
2. **Prepare your video data:**
   - Ensure you have 6–10 different walking videos (different people, different speeds)
   - Extract one clear frontal face image per person
3. **Generate deepfakes using your chosen tool**
4. **Verify gait preservation** by comparing MediaPipe skeleton features before/after
5. **Run inference tests** on your gait model to confirm it detects the mismatch
6. **Document and report results** in your evaluation report

---

**Document Created:** February 9, 2026
**Status:** Ready for implementation
**Next Milestone:** Execute Steps 1–3 before deepfake generation
