# Gait Verification V2 — Phase 1A capture-pilot engineering report

**Date:** 12 October 2026 (Asia/Kolkata). **Branch:** `feat/gait-verification-v2`.
**Starting commit:** `02b0312b5d16c84032a59b3ca31f75a9b77ad8cd`.
**PHASE 0 ENGINEERING: COMPLETE after a verified native-score cache fix.**
**PHASE 1A ENGINEERING: COMPLETE for the explicitly permitted available-fixture scope.**
**Representative moving-camera pilot validation: PENDING.**
**Phase 1B: ready to begin an empirical four-view pilot; not ready for a frozen SOP,
production biometric decisions or final scientific evaluation.**

## 1. Executive summary

Independently audited current code, configuration, commits, test bodies and executable
behavior before Phase 1A edits. Reused the official RTMPose weights, frozen GaitPT and
GPGait implementations and FFmpeg stabilization. Added opt-in temporal quality analysis,
unchanged contiguous encoder segments, versioned walk aggregation, paired comparisons,
local visual QA, checksum-pinned pilot configuration, a repeatable real CUDA command,
a CUDA environment snapshot and a practical student SOP draft.

Fresh real-video CUDA inference works; it does not establish biometric accuracy.
No training, fine-tuning, calibration fitting or operational threshold installation
was performed in this audit/Phase 1A session. The two prohibited legacy tests were
inspected and excluded before any pytest execution. No commit, push, merge, original
video edit, checkpoint overwrite, entire-dataset download or gallery search occurred.

RAW/S10/S15 were measured on **two independent historical original files**, apparently
from the same historical participant based on filenames; they are not independent
subjects or sessions. The side clip is usable, the frontal clip has background people
and is intentionally unassigned. No result supports choosing stabilization for the
new moving-camera protocol. The planned 40 × 4 × ~30-second cohort is not collected.

## 2. Phase 0 independent verdict and repository audit

The initial tree was clean on the required branch, with the latest Phase 0 CUDA
commit `02b0312`. Expected runtime, scripts, tests, configs, requirements and reports
are tracked. Checkpoints, native caches, cloned sources and generated pilot media are
ignored. `git diff 1bce211 HEAD -- models utils tests/test_future_work.py outputs
scripts/training scripts/inference` was empty. Historical E0 source/artifacts/reports
remain untouched; equivalence and 133,058-parameter regression tests pass. Historical
accuracy numbers were not replayed or claimed for the new cohort.

Read AGENTS.md, both complete project documents, the complete Phase 0 report including
its CUDA amendment, both mandatory configs and requirements before code edits. The
report's earlier unavailable-CUDA/RTMPose/GPGait statements are superseded by its
amendment and fresh execution. Source-dataset access, scientific thresholds and the
future cohort are not classified as Phase 0 engineering failures under the owner's
newer explicit gate. Production model metadata stays BLOCKED, appropriately.

Fresh profiled extraction on `data/videos/Aarav_S1.mp4` yielded 140 frames, display size
1080×1920, intact source PTS and a usable provisional gate. Strict official pretrained
forwards yield finite normalized GaitPT 256-D and GPGait 4,864-D outputs, repeat delta
zero. Source revisions and original/model-only checkpoint hashes were checked.
No restricted-deserialization guard was bypassed.

## 3. Genuine Phase 0 gap fixed

Initial real extraction failed with `ValueError: Confidence outside [0,1]`. RTMPose
SimCC scores can exceed one. The cache validator now accepts finite **nonnegative
native scores**, preserves values without clipping, and still rejects negative/NaN/Inf.
A roundtrip regression covers 1.12 confidence. Final Phase 1A source review also found RTMLib pose-only inference defaults to a full-image crop when boxes are empty. The wrapper now returns an explicit missing detection instead; an actual CUDA YOLOX run on a blank image and a targeted regression passed. Neither measured historical clip has zero-box frames, so their paired results are unchanged. RTMPose sessions disable runtime fallback
after checking CUDA initialization; normal CPU shape nodes are separately logged.

The `/tmp` model source clones referenced by old instructions had disappeared. Restored
the exact clean upstream commits into ignored `.cache/gait_verification/sources/`;
no checkpoint downloads were necessary. This was local reproducibility restoration,
not a new architecture. The old 90-frame cache checksum matches none of the 66 current
originals, so it is preserved as historical evidence and not used as proof of new
source lineage. Fresh audit caches come from verified originals.

## 4. RTMPose source, artifact and native contract

Official [RTMLib Body configuration](https://github.com/Tau-J/rtmlib/blob/main/rtmlib/tools/solution/body.py)
and the installed RTMLib 0.0.16 configuration list the selected OpenMMLab archives.
The keypoint order was checked against the official
[MMPose COCO definition](https://github.com/open-mmlab/mmpose/blob/759b39c13fea6ba094afc1fa932f51dc1b11cbf9/configs/_base_/datasets/coco.py).
The local cached artifacts were reused. No new weights, duplicate archives or training
datasets were downloaded. Checksums are locally recorded provenance, not publisher signatures.

| Artifact | Exact source archive | ONNX size (bytes) | SHA-256 | Native input |
|---|---|---:|---|---|
| yolox_m_8xb8-300e_humanart-c2c7a14a.onnx | [OpenMMLab archive](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip) | 101400344 | `3dea6513388889f0fff4b77bf7a26013600321b9eb9ceb0e9a400a82572f5f23` | `[1, 3, 640, 640]` |
| rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.onnx | [OpenMMLab archive](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip) | 54330655 | `5c0a4bf67953e6d2ac43ce15e77dc9d5d354ae18430a47d2c5963a7bc5683e3c` | `['batch', 3, 256, 192]` |

Detector: YOLOX-M HumanArt, 640×640. Pose: RTMPose-M body7 SimCC, 192×256
(width×height), dynamic person batch, output 17 joints. ONNX coordinates are mapped
back to **display-oriented source image pixels**, x right/y down, with native scores.
COCO17 order: nose, left/right eye, left/right ear, left/right shoulder, left/right
elbow, left/right wrist, left/right hip, left/right knee, left/right ankle. No heels
or toes exist in this model's outputs; shoe/foot completeness requires visual QA.

CUDAExecutionProvider is used for model computation. CPU input-shape inspection is
metadata-only, not inference fallback. Initialization without CUDA or runtime CUDA
failure fails explicitly; ORT CPU administrative/shape nodes are not represented as
CPU-only model inference. Native cache order, scores, masks, source index/PTS, displayed
size, bounding boxes, take/variant, estimator hashes and source checksum are preserved.
Detector boxes replace skeleton-extrema boxes when available. Existing caches still load.

## 5. CUDA runtime and gait checkpoints

Python `3.11.17 (main, Oct  9 2026, 15:21:28) [Clang 22.1.3 ]`; GPU `NVIDIA GeForce RTX 3050 6GB Laptop GPU`; NVIDIA driver `595.99.02`.

| Dependency | Measured version |
|---|---|
| torch | 2.10.0+cu128 |
| torchvision | 0.25.0+cu128 |
| onnxruntime-gpu | 1.23.2 |
| rtmlib | 0.0.16 |
| numpy | 2.4.6 |
| opencv-python | 5.0.0.93 |

NVIDIA-SMI advertises driver CUDA 13.2 support; PyTorch was built with CUDA 12.8.
A real CUDA tensor sum produced 4,950 on cuda:0. ORT profiles count executed kernel
nodes, not just available providers:

| Runtime | CUDA node events | CPU node events |
|---|---:|---:|
| detector | 43820 | 560 |
| pose | 25900 | 280 |

| Gait artifact | Source revision | Checkpoint bytes | SHA-256 |
|---|---|---:|---|
| GaitPT GREW | `4977618a36ef77ee9a2863c9f8e27ab862894d70` | 18,061,251 | `e6b287e2c2ae27a9344b8009077c9d4f70811d2d6f3ff20198b99bfa4f824637` |
| Original GPGait GREW | `d3ce631971a01d806eeb3a154774d482c0951813` | 1,264,347,406 | `929fc52bd74eed9d7f3bbf535a03ea5f531938ae300c67a7b219a909cd41d2a2` |
| GPGait model-only | same FastPoseGait revision | 1,263,627,531 | `12c15082cbd9c464b6f250630ba2147728d6f6c205a2c7f87026625bc68a1302` |

Official artifact sources: [GaitPT GREW](https://drive.google.com/uc?id=1itpe5zwg-K4OtSKii7-83C7XH8tO9uAR)
and [GPGait GREW original](https://drive.google.com/uc?id=1_cCKtVR0v1s3fhLVECSIm9QjveAHRSLv).
The model-only artifact is a preexisting safely extracted derivative; original and
extracted hashes both match. Restricted loading finds 1,310 model state entries.
The worker requires exactly a model-only envelope and uses strict model loading,
CUDA, eval, requires_grad(False) and inference_mode. No unrestricted pickle load.
GaitPT weights were also compared before/after repeated inference and stayed unchanged.

GaitPT preserves official GREW height scaling, anatomical reorder/repeated nose and
two centered 30-frame crops. GPGait preserves official GREW Affine→RescaleCenter→HOD→
SkeletonInput, with finite `(1,6,T,17,1)` input and `(1,256,19)` output. No legacy
78-channel features are substituted, and no missing joints are synthesized.

| Encoder path | Dimension | Norm | Repeat max difference |
|---|---:|---:|---:|
| gaitpt, Phase 0 native | 256 | 1.00000000 | 0.0 |
| gaitpt, Phase 1A window aggregation | 256 | 1.00000000 | 0.0 |
| gpgait, Phase 0 native | 4864 | 0.99999994 | 0.0 |
| gpgait, Phase 1A window aggregation | 4864 | 1.00000000 | 0.0 |

Actual bad-checksum, short-sequence, missing-joint and GPGait CPU rejection checks
passed; these are separate from mocked unit tests. GPGait++ was not integrated or
loaded in this phase. Its earlier restricted-load blocker remains historical and unresolved.

## 6. Reproducibility and setup

Use the working `.venv-gait-v2-cuda` when present. Do not install into the global
Python or overwrite a working environment. `uv pip check --python
.venv-gait-v2-cuda/bin/python` reported all 83 installed packages compatible.
A newly supplied CUDA snapshot reflects the observed environment, not the older
CPU requirements. RTMLib depends on the CPU ORT distribution; both CPU/GPU package
metadata coexist in the working environment. For a **new** environment, install GPU
ORT last to restore its shared Python namespace and then require real CUDA profiling.
The existing environment was not reinstalled. A completely fresh CUDA installation
from this snapshot was not executed here; runtime reproducibility was verified by
restoring sources and rerunning the existing environment.

```bash
uv venv --python 3.11.17 .venv-gait-v2-cuda-repro
uv pip install --python .venv-gait-v2-cuda-repro/bin/python   --index-strategy unsafe-best-match -r requirements-gait-v2-cuda.txt
uv pip install --python .venv-gait-v2-cuda-repro/bin/python   --reinstall --no-deps onnxruntime-gpu==1.23.2
uv pip check --python .venv-gait-v2-cuda-repro/bin/python
```

The two OpenCV distributions also share cv2; both are pinned to 5.0.0.93 as in the
working environment. Driver and CUDA/cuDNN shared-library availability must be verified
by the smoke command rather than assumed from installation success. Downloads should
retain TLS verification. No model download failures or partial files occurred here.

The executed source-restoration commands were:

```bash
mkdir -p .cache/gait_verification/sources
git clone https://github.com/UbiquitousCognitiveSystems/GaitPT .cache/gait_verification/sources/GaitPT
git -C .cache/gait_verification/sources/GaitPT checkout 4977618a36ef77ee9a2863c9f8e27ab862894d70
git clone https://github.com/BNU-IVC/FastPoseGait .cache/gait_verification/sources/FastPoseGait
git -C .cache/gait_verification/sources/FastPoseGait checkout d3ce631971a01d806eeb3a154774d482c0951813
```

Skip cloning if those exact clean sources already exist. Cached weights are explicitly
passed, hash-checked and reused; this report does not provide checkpoint redistribution.

## 7. Real-video extraction and temporal quality behavior

Source-00: `data/videos/Aarav_S1.mp4`, 140 frames, probed duration 4.655933 s,
PTS span 4.622611 s. Source-01: `data/videos/Aarav_F3.mp4`, 78 frames, duration
2.589589 s, PTS span 2.556267 s. Both displayed 1080×1920. The original dates,
sessions, precise view geometry and camera movement were not recorded; explicit
historical-unknown labels were used, not invented new-cohort metadata. Filename
side/frontal labels are historical hints, not standardized left45/toward geometry.

Native caches retain unassociated candidate xy/confidence/observed arrays as well as
single-track arrays. In ambiguous frames the selected track is absent; candidate
observations remain intact. Missing or low-confidence measurements are never filled.
Brief missed detections can return to the same provisional continuity ID if the
center is consistent. Multiple people or a center discontinuity force uncertain
continuity, and aggregation across distinct assigned tracks is refused. This is a
conservative single-walker association heuristic, not a full multi-person tracker.

Diagnostics separate insufficient pose information, low-confidence/missing ankles,
observed points outside the border margin, tracking failures, timestamp gaps,
potential camera-motion/scale artifacts and large relative skeleton steps. Ordinary
observed motion is not automatically labeled verified walking. All outputs include
per-frame data, per-window metrics, accepted/rejected IDs and explicit reasons.

`phase1a.json` declares provisional constants: 2-second nonoverlapping source-PTS
bins; ≥31 unchanged contiguous frames; score ≥0.3; 2-pixel border margin; ≥80%
complete/ankle frames in an accepted bin; interval gap >3×median; relative joint step
>0.15 body heights; center step >0.25 body heights; scale step >20%; reject bins with
>10% step flags; ankle excursion ≥0.005 body heights; full-frame Laplacian variance
<20 flags possible blur. No values are biometric acceptance thresholds or empirically
validated quality cutoffs. Full-frame blur/dark edges and body-scale changes are
context-dependent proxies. Bone CV includes perspective/gait foreshortening, not just
pose error. Natural motion is not smoothed away to improve statistics.

Only fully observed, confident, in-frame, same-track, no-gap/no-step contiguous runs
reach the unchanged official preprocessors. Equal averaging of normalized run
embeddings, followed by normalization, is an opt-in engineering recipe hashed into
model IDs. Windows are never independent enrollments or split units. ClaimVerifier
accepts an injected window assessor; its default Phase 0 gate remains available.
The public CLI still fails closed without an approved enrollment/model/policy.

## 8. Moving-camera findings and scope limits

No suitable labeled 30-second moving-camera pilot exists in the repository. A search
found 66 originals, historical/generated stabilization artifacts and augmented copies;
no claimed new pilot paths were invented. Only the explicit two-file selection was
processed. Some background pans may exist in these short clips, but intentional
front/rear operator translation was not established. Therefore the intended moving-camera
comparison, four-view ankle occlusion, true 60-fps behavior and long-video real gait
coverage remain **pending**. A 1,800-frame synthetic regression demonstrates 30-second
window mechanics only; it is not empirical capture evidence.

Visual QA of the side clip shows an anatomically placed leg/ankle skeleton and
visible shoes. S15 exposes a black strip at the left edge. Visual QA of the frontal
clip confirms background bystanders, consistent with 2–3 detections in every frame.
No candidate is selected by prominence. Stabilization does not resolve that ambiguity.
No inference that camera translation was removed, or that lack of proxy flags proves
absence of camera artifacts, is made.

## 9. RAW/S10/S15 controlled experiment and provenance

Used the existing two-pass runner with shakiness 5, accuracy 15, mincontrast 0.2,
smoothing 10 or 15, optzoom 0, crop black, interpolation bicubic, CRF17/libx264.
Both passes use the same original; frame rate is not converted. RAW remains the
unmodified original. FFmpeg 8.0.1-3ubuntu2 includes libvidstab. The official
[vid.stab source](https://github.com/georgmartius/vid.stab) was checked.

Each source/variant manifest records source SHA, derivative SHA, original Recording
metadata, exact argument arrays, FFmpeg version, display orientation, time base,
full PTS list, source-relative origin offset/error, and a separate transformation
sidecar. The runner now also hashes the transform file and probes stream duration.
For this experiment the final pose-cache update reused existing generated derivatives;
it separately hashes the `.trf` file and links its original transformation manifest.
Neither original changed. All four stabilized derivatives have zero PTS offset and
zero relative PTS error, equal frame count and equal probed duration.

| Source | Variant | SHA-256 | Frames | Duration s |
|---|---|---|---:|---:|
| source-00 | raw | `3980dd513f5818755e680d04a469693d1e4eada0e9a17b70a5cf8c0259206a08` | 140 | 4.655933 |
| source-00 | vidstab-s10 | `dd82655872018fdbca6f7517d9e6a086fadb6838647a24dd7c763f905dc408aa` | 140 | 4.655933 |
| source-00 | vidstab-s15 | `9563fff6edf2da4680ec1862b55fc7c2f09f4c219954eef3cf33d347ffa96cfe` | 140 | 4.655933 |
| source-01 | raw | `33b19483a72e549d710855879a6a848344e24b15699501f75d11d46dff115c5c` | 78 | 2.589589 |
| source-01 | vidstab-s10 | `99fb83bf14a6750145f0d750903ae2044b93af572922c6b3abccab383df00da3` | 78 | 2.589589 |
| source-01 | vidstab-s15 | `56c355ed0b73adbc587aeb161d1e60de2aab3b854580999fb2a5f4a87d8f6368` | 78 | 2.589589 |

Exact command arrays and logs are local in
`outputs/gait_verification/phase1a-historical-pilot/source-XX/VARIANT/manifest.json`.
No original/derivative changed split or gained an independent take identity.

## 10. Per-original measurements and equal-original aggregates

Full-body/ankle/missingness metrics refer to the **assigned single-person track**.
The frontal clip has native candidate skeletons but no assigned person; its 100%
selected missingness means abstention, not that RTMPose failed to detect humans.
Confidence/bone metrics are undefined there and remain null, not zero. Aggregate
confidence/bone values therefore have support of only **one original**; other fractions
are equal means over both originals. The JSON records these denominators explicitly.

| Original / variant | Detection | Ambiguous | Full body | Ankles | Mean confidence | Missing assigned joints | Track continuity | Step events | Mean bone CV | Accepted/rejected windows | Dark edge fraction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| historical-pilot-0 / raw | 1.000000 | 0.000000 | 1.000000 | 1.000000 | 0.844329 | 0.000000 | 1.000000 | 0.000000 | 0.106516 | 2/1 | 0.009233 |
| historical-pilot-0 / vidstab-s10 | 1.000000 | 0.000000 | 1.000000 | 1.000000 | 0.837588 | 0.000000 | 1.000000 | 2.000000 | 0.107456 | 2/1 | 0.011972 |
| historical-pilot-0 / vidstab-s15 | 1.000000 | 0.000000 | 1.000000 | 1.000000 | 0.836547 | 0.000000 | 1.000000 | 0.000000 | 0.107829 | 2/1 | 0.026659 |
| historical-pilot-1 / raw | 1.000000 | 1.000000 | 0.000000 | 0.000000 | undefined | 1.000000 | 0.000000 | 0.000000 | undefined | 0/2 | 0.007536 |
| historical-pilot-1 / vidstab-s10 | 1.000000 | 1.000000 | 0.000000 | 0.000000 | undefined | 1.000000 | 0.000000 | 0.000000 | undefined | 0/2 | 0.007418 |
| historical-pilot-1 / vidstab-s15 | 1.000000 | 1.000000 | 0.000000 | 0.000000 | undefined | 1.000000 | 0.000000 | 0.000000 | undefined | 0/2 | 0.009765 |

| Equal-original mean | RAW | S10 | S15 |
|---|---:|---:|---:|
| person_detection_fraction | 1.000000 | 1.000000 | 1.000000 |
| ambiguous_detection_fraction | 0.500000 | 0.500000 | 0.500000 |
| full_body_visibility_fraction | 0.500000 | 0.500000 | 0.500000 |
| valid_ankle_fraction | 0.500000 | 0.500000 | 0.500000 |
| confidence_mean | 0.844329 | 0.837588 | 0.836547 |
| confidence_p10 | 0.707754 | 0.702524 | 0.701937 |
| missing_joint_fraction | 0.500000 | 0.500000 | 0.500000 |
| low_confidence_fraction | 0.000000 | 0.000000 | 0.000000 |
| tracking_continuity_fraction | 0.500000 | 0.500000 | 0.500000 |
| tracking_break_count | 0.000000 | 0.000000 | 0.000000 |
| jitter_events | 0.000000 | 1.000000 | 0.000000 |
| mean_bone_length_cv | 0.106516 | 0.107456 | 0.107829 |
| accepted_window_count | 1.000000 | 1.000000 | 1.000000 |
| rejected_window_count | 1.500000 | 1.500000 | 1.500000 |
| border_proximity_fraction | 0.000000 | 0.000000 | 0.000000 |
| mean_border_dark_fraction | 0.008385 | 0.009695 | 0.018212 |
| potential_blur_fraction | 0.000000 | 0.000000 | 0.000000 |
| potential_camera_artifact_events | 0.000000 | 0.000000 | 0.000000 |

Both originals have regular measured intervals around 0.033322 s and no >3×median
gaps. Side-clip rejection: short final bin (<31 frames, no contiguous encoder segment).
Frontal rejection: insufficient assigned pose, missing assigned ankles, no contiguous
segment, insufficient motion evidence in both bins, plus a short tail. That absence
of assigned motion is not proof that the visible walker is stationary. Zero track
changes in the ambiguous frontal clip means **no track was ever assigned**, not good
continuity. Per-frame tracking failure is 100% there.

Paired side-clip changes from RAW: mean confidence −0.006741 (S10), −0.007782 (S15);
bone CV +0.000940/+0.001312; dark edge +0.002739/+0.017426. S10 adds two skeleton
step flags; S15 adds none. Neither gains accepted windows or feet coverage. The
frontal clip remains ambiguous across all three. These are pose diagnostics with
strong within-clip correlation, not human-subject statistical evidence or recognition
scores. No frame-level significance test or accuracy claim is made.

**Conditional recommendation:** retain RAW as the control for this evidence. There is
no observed reason to promote S10/S15 on these fixtures. Re-evaluate on actual moving
front/rear pilot recordings, requiring unchanged anatomy/feet and usable windows;
lower jitter alone would still not establish improved biometric verification.

## 11. Implemented files and use

- `gait_verification/contracts.py`: native-score fix, additive diagnostics/candidates,
  displayed-coordinate clarification and probed duration.
- `pose.py`: pinned optional ONNX hashes, CUDA fallback rejection, actual detector
  boxes, optional per-frame diagnostic collection and conservative association.
- `media.py`, `stabilization.py`: probe duration and hash transformation sidecars.
- `pose_quality.py`: versioned temporal reports, unchanged segment slicing,
  abstention and opt-in WindowedEncoder aggregation.
- `pilot.py`: explicit development-only selection, immutable paired processing,
  per-original/delta/equal-original tables with missing-metric support and local QA.
- `verifier.py`: injected quality assessor while preserving default/claim-only API.
- `backends/gaitpt.py`: CUDA default with explicit failure, retaining explicit CPU
  experiments only when requested via device argument. GPGait implementation reused.
- `scripts/gait_verification/{compare_capture_pilot,validate_cuda_runtime}.py`:
  reproducible pilot and real GPU/model execution commands.
- `prepare_pose_cache.py`: opt-in `--window-quality` with fixed artifact/config checks.
- `configs/gait_verification/phase1a.json`, `requirements-gait-v2-cuda.txt`:
  declared provisional policy and measured isolated CUDA dependency snapshot.
- `tests/gait_verification/{test_pilot,test_pilot_media}.py`, one new foundation
  regression: targeted engineering coverage, no training/fitting.
- Both mandatory project docs and Phase 0 report: append dated amendments, preserving
  historical observations. This report and `CAPTURE_SOP_DRAFT.md` are the deliverables.

Executed pilot command (with OMP_NUM_THREADS=2):

```bash
.venv-gait-v2-cuda/bin/python scripts/gait_verification/compare_capture_pilot.py   --selection-json outputs/gait_verification/phase1a-selection.json   --detector /home/arhaan/.cache/rtmlib/hub/checkpoints/yolox_m_8xb8-300e_humanart-c2c7a14a.onnx   --pose-model /home/arhaan/.cache/rtmlib/hub/checkpoints/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.onnx   --new-output-dir outputs/gait_verification/phase1a-historical-pilot
```

Selection JSON is an explicit list of `{video, recording}` rows. Recording fields
are `asset_id, original_take_id, subject_id, session_id, view, recorded_at,
source_sha256, split_role, variant_id, parent_asset_id`; pilot originals must be
`split_role=development`, `variant_id=raw`, null parent. For new pilot files use
actual metadata; the historical unknown sentinels here are not a new-cohort protocol.
Output directories must be new: reruns refuse overwriting, and generation never
traverses all existing videos. To inspect one new cache use `prepare_pose_cache.py
--window-quality --video ... --take-id ... --detector ... --pose-model ... --output ...`.

A local extraction-only update (`/tmp/gait_phase1a_reextract.py`) reused the completed
variant manifests, checked original and derivative hashes, extracted revised candidate
caches and recomputed the **final** diagnostics in `outputs/gait_verification/phase1a-final`.
It preserved the earlier immutable outputs. Full final per-frame/per-window reports
and 18 local overlays are there, along with `comparison.json`. Figures were viewed
locally; no recognizable footage or biometric artifacts were uploaded/published.

Executed repeatable real CUDA validation:

```bash
.venv-gait-v2-cuda/bin/python scripts/gait_verification/validate_cuda_runtime.py   --video data/videos/Aarav_S1.mp4   --detector /home/arhaan/.cache/rtmlib/hub/checkpoints/yolox_m_8xb8-300e_humanart-c2c7a14a.onnx   --pose-model /home/arhaan/.cache/rtmlib/hub/checkpoints/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.onnx   --new-output-dir outputs/gait_verification/phase1a-cuda-runtime
```

Evidence: `runtime.json`, native pose NPZ and detector/pose ORT profile JSON. The
earlier independent Phase 0 audit is separately retained at
`outputs/gait_verification/independent-phase0-fixed/audit.json`. Checkpoint tests and
scientific source-data reproduction must not be confused with these local forwards.

## 12. Tests, commands, failures and exclusions

Inspected all selected test bodies first. Before changes: **38 passed, 2 deselected**.
After Phase 0 score fix: **39 passed, 2 deselected**. Final suite: **63 passed,
2 deselected, one non-failing upstream Transformer warning; no skips/final failures**.

```bash
.venv-gait-v2-cuda/bin/python -m pytest -q tests/gait_verification tests/test_future_work.py   -k 'not test_harness_end_to_end_on_synthetic_subjects and not test_metrics_against_known_values'
.venv-gait-v2-cuda/bin/ruff check gait_verification scripts/gait_verification tests/gait_verification
.venv-gait-v2-cuda/bin/python -m compileall -q gait_verification scripts/gait_verification
git diff --check
uv pip check --python .venv-gait-v2-cuda/bin/python
```

Targeted new tests: **24 passed** (20 temporal/paired tests and four actual FFmpeg
RAW/S10/S15/orientation tests); the extra native-score regression brings the suite
from 38 to 63. They cover 30-second window mechanics, occasional missing ankles,
static/short/cropped/time-gap reasons, track discontinuities, no pooling of uncertain
walkers, cache/candidate roundtrip, checksum rejection, temporal abstention,
paired recipe/source/time rejection, source immutability and portrait display rotation.
The existing irregular-PTS FFmpeg regression and historical E0 feature/model parity
also pass. Synthetic fixtures and mocked interfaces are labeled engineering evidence;
real GPU inference is the separately executed profiled command above.

Transient failures discovered and resolved: real RTMPose >1 score cache rejection;
window-boundary timestamp gaps omitted from rejection reasons; generated rotation
fixture's obsolete metadata flag (replaced by FFmpeg display_rotation); initial new
code formatting/lint violations (formatter corrected). `python -m pip list` failed
because this uv environment has no pip module; `uv pip freeze/check` succeeded.
`graphify query/update` both fail with command not found; the graph is unchanged.
The latter is a tooling limitation, not a gait runtime failure.

Excluded throughout: `test_harness_end_to_end_on_synthetic_subjects` (neural training)
and `test_metrics_against_known_values` (calibrator fitting). The Phase 0 report's
prior accidental legacy training is historical; it was not repeated in this session.

## 13. Capture-pilot recommendations

Use the [student SOP draft](CAPTURE_SOP_DRAFT.md) to record 2–3 consenting volunteers
in all four candidate views with the actual Android phone, consistent main lens and
1080p60 where available. Explicitly measure route/angle/distance/height and log stock
stabilization, actual timestamps, sessions and operator. Keep bystanders out and
feet visible. Use a spotter and safe route for backward camera movement. Compare
stationary versus moving oblique procedures before settling the 45° definition.

Provisional capture review: at least ten accepted 2-second windows spread through a
~30-second take, one unambiguous walker, healthy media and intact shoe framing.
This is a pilot target to evaluate, not a proven biometric minimum. Review flags
visually and count every failure/recapture. Do not drop hard recordings or turn a
quality failure into FALSE. All pilot-driven subjects/derivatives stay development.

## 14. Remaining scientific, license and runtime limitations

- No representative long moving-camera pilot; actual blur/occlusion/border robustness
  and all four view geometry remain unvalidated. Tracking uses a conservative
  heuristic; subtle swaps after short occlusions can remain undetected.
- COCO17 does not measure heels/toes; native detector confidence and finite outputs
  do not certify correct anatomical tracking or continuous walking.
- No new cohort, validated operating threshold, calibration, verification accuracy,
  cross-day robustness, comparative winner or biometric security claim.
- GREW/source-sample benchmark reproduction remains access-restricted. Native input
  preprocessing parity and real forwards do not establish detector-domain equivalence.
- GaitPT source CC BY-NC-ND 4.0 and FastPoseGait academic/noncommercial restrictions
  remain relevant; checkpoint redistribution/deployment permissions are not resolved.
  Local wrappers do not grant rights. GPGait metadata remains license-unverified for
  deployment, and both encoder readiness values remain BLOCKED.
- GPGait++ restricted checkpoint issue remains; not a Phase 1A dependency or attempted
  security bypass. Consent lifecycle/service deployment are later work.
- CUDA dependency snapshot reflects an existing modern runtime; a brand-new install,
  upstream old dependency stack and multi-machine reproduction have not been tested.
- Proxy thresholds and selected two-second/equal-segment recipe need pilot calibration;
  no smoothing/interpolation has been used to improve apparent pose statistics.

## 15. Phase 1B readiness and exact next step

Phase 0 engineering is complete, including the fixed cache contract. Phase 1A's
implementation, testing and explicitly allowed available-fixture comparison are
complete. **The scientific capture-pilot validation itself is not complete.** The
absence of the future cohort is not an engineering failure, and successful inference
is not proof of recognition accuracy.

Phase 1B can begin as the real four-view acquisition/pilot review milestone (the
project docs do not otherwise define the Phase 1B label). **Next:** collect 2–3
volunteers' independent ~30-second left45/right45/toward/away originals with measured
candidate geometry and real session metadata; run the explicit paired command with
one frozen config; visually adjudicate ambiguous tracks/feet/borders and compare
accepted windows by view; revise provisional quality rules and SOP based on those
measurements. Freeze the acquisition SOP only after that evidence, before mass
collection. Keep all pilot participants outside locked final testing. Training,
calibration, mass automatic processing and production deployment are not authorized
by this readiness declaration.
