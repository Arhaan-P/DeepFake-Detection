# Gait Verification V2 — Phase 0 implementation report

**Audit date:** 12 October 2026 (Asia/Kolkata). **Branch:** `feat/gait-verification-v2`.
**Starting revision:** `1bce2110571fe957edd079b5bcb7390bdba2c795`.
**Disposition:** substantial Phase 0 engineering implemented; **not all PRD readiness gates passed**.

## CUDA engineering validation amendment — 12 October 2026

This amendment supersedes earlier statements that CUDA execution,
RTMPose extraction, and pretrained GPGait inference were unverified.
The original audit is retained as historical evidence.

### Verified CUDA implementation

- Environment: Python 3.11, PyTorch 2.10.0+cu128,
  NVIDIA RTX 3050 6GB.
- RTMPose: detector and pose ONNX models ran using CUDAExecutionProvider.
  A real 90-frame COCO17 pose cache passed provisional quality checks.
- GaitPT-GREW: official pretrained weights loaded strictly.
  Frozen CUDA inference produced normalized 256-dimensional embeddings.
- GPGait-GREW: official checkpoint was safely reduced to model-only.
  Restricted checkpoint loading passed with 1,310 state entries,
  zero missing keys, and zero unexpected keys.
- GPGait preprocessing: official transformations generated a finite
  input tensor of shape (1, 6, 90, 17, 1).
- GPGait CUDA inference: produced an embedding of shape (1, 256, 19),
  flattened and normalized to 4,864 dimensions.
- GPGait repeatability: maximum difference between two runs was 0.0.
- GPGait safety checks: insufficient frames, missing joints,
  incorrect checkpoint checksum, and CPU execution were rejected.
- The isolated inference worker and GPGaitEncoder are implemented.
- Four new automated GPGait interface tests passed.

### Checkpoint provenance

Original GPGait GREW SHA-256:
929fc52bd74eed9d7f3bbf535a03ea5f531938ae300c67a7b219a909cd41d2a2

Extracted model-only GPGait SHA-256:
12c15082cbd9c464b6f250630ba2147728d6f6c205a2c7f87026625bc68a1302

GaitPT GREW SHA-256:
e6b287e2c2ae27a9344b8009077c9d4f70811d2d6f3ff20198b99bfa4f824637

FastPoseGait source revision:
d3ce631971a01d806eeb3a154774d482c0951813

### Regression evidence

Final selected regression suite:
38 passed, 2 deselected, 1 warning.

The two excluded legacy tests perform synthetic training or
calibration fitting. They remain excluded under the no-training
and no-fitting constraint.

### Remaining limitations

- The planned 40-person dataset has not yet been collected.
- Recognition accuracy and error rates have not been established.
- No scientifically validated decision threshold exists.
- The production verification policy remains blocked.
- Official GREW dataset reproduction remains access-blocked.
- GPGait++ pretrained inference is not validated.
- Model licensing requires review before deployment.
- Provisional quality rules and stabilization settings require
  validation using the new recordings.

These results demonstrate successful engineering functionality,
not scientifically validated biometric performance.

---

## A. Executive summary

Implemented an isolated `gait_verification` package: timestamp-aware media ingestion,
native pose caches, anatomical skeleton conversion, frozen official GaitPT loading,
per-original enrollment, one-claim scoring, explicit non-Boolean failures, offline
verification trials/metrics, capture-manifest validation, and optional two-pass
stabilization with immutable source lineage. The public result contains only the
claim, status, Boolean-or-null verdict and schema version. No alternative identity,
identity ranking, similarity probability, or deepfake verdict is provided.

**VERIFIED BY RUNNING CODE:** the official GREW GaitPT checkpoint loaded strictly
and produced deterministic finite normalized **256-dimensional** embeddings on both
synthetic markers and a real historical clip's newly extracted MediaPipe poses.
There are **4,484,384 parameters**, measured from the loaded model. Two repeated
forwards had maximum absolute difference **0**, all parameters had gradients disabled,
and the complete state dictionary was unchanged. This is engineering evidence,
not recognition-accuracy evidence or source-dataset reproduction.

A historical 78-frame portrait clip produced a native `78×33×2` pose cache with
exact source PTS. Its raw and smoothing-10 derivatives were processed locally.
The corrected derivative preserves all 78 frames and relative PTS exactly. The
original checksum is unchanged. No full-dataset stabilization was performed.

**Important execution deviation:** two initial full legacy pytest runs included
`test_harness_end_to_end_on_synthetic_subjects`, which internally trains temporary
models for two epochs on three synthetic subjects. This violated the owner's
no-training constraint. It was discovered when inspecting the test body and was
excluded from subsequent commands. The legacy metrics fixture also fits a Platt
calibrator on synthetic scores; it is excluded from the final no-fitting command.
No operational calibration artifact was installed. No real-data training, pretrained-backbone
optimization, or modification of historical checkpoints occurred. Do not describe
this session as having performed zero training of any kind.

No final architecture, stabilization policy, or acceptance threshold was selected.
The planned 40-person cohort does not exist in this checkout and was not benchmarked.

## B. Repository observations

| Area | Actual observation / reuse boundary |
|---|---|
| Git | Correct feature branch already existed. Initial working tree had only untracked `AGENTS.md`; it remains unchanged. No commit, push, merge or history rewrite. |
| Layout | Existing root packages `models/`, `utils/`; scripts grouped under training, inference, evaluation, preprocessing, enrollment and future_work. V2 follows root-package convention rather than restructuring E0. |
| Data | 66 files under `data/videos`, 1,056 augmented video files, 1,120 entries in `data/gait_features/gait_features.pkl`. Current `enrolled_identities.pkl` has **one entry**, not 13 enrolled templates. Historical corpus reports 13 subjects; no new cohort inferred. |
| Original extractor | `utils/pose_extraction.py`: MediaPipe Pose Lite; selected joints `[11,12,23,24,25,26,27,28,29,30,31,32]`; detected poses resampled by index to 60 steps. |
| 78-D features | Hip-relative 12×3 coordinates + six joint angles + 12×3 adjacent differences. Differences are not physical velocity. `utils/gait_features.py` offers vectorized parity implementation. |
| E0 | `models/verifier_variants.py::TCNVerifier`: concatenate difference, absolute difference, product; Conv1D kernels 7/5/3, hidden 64/64/32, pooling + MLP. 133,058 parameters, confirmed by existing equivalence regression. |
| Legacy wrapper | `models/full_pipeline.py` computes an auxiliary embedding, but decisive verification logits use raw pair differences. `outputs/model_config.json` supplies the smaller historical architecture. Default `create_model()` smoke constructs 4,085,510 parameters; do not confuse defaults with the historical configured wrapper. |
| Entry points | `scripts/training/train.py`, `scripts/inference/inference.py`, `scripts/evaluation/evaluate.py`, `ablation_loocv.py`, and `scripts/future_work/run_experiment.py`. No training entry point was intentionally invoked. |
| Checkpoints | Existing `.pth` files in `outputs/checkpoints` and `outputs/ablation`. Legacy loader uses `model_state_dict`, model config, feature statistics, and `torch.load(weights_only=False)`. V2 does not replace it and uses restricted loading for external checkpoints. |
| Results | `outputs/future_work/report/summary.md` preserves E0 AUC 85.24%, EER 22.35%, TAR@5% FAR 36.36%. These were read, **not reproduced** or applied to the new cohort. |
| Metadata/loaders | `utils/data_loader.py` loads feature/enrollment pickles; `utils/gait_descriptors.py` parses names into identity/view/take, fabricates time from index/FPS, and fills missing coordinates. It does not establish independent sessions. V2 does not import this path. |
| Pose backends | `utils/pose_backends.py`: MediaPipe, RTMPose Halpe26, ViTPose, RTMW3D, OpenPose mapped into a sparse MediaPipe33 layout. Cached native joints/actual PTS are absent; `_pick_person` chooses prominent bounding box without durable association. `data/pose_cache` was absent. |
| Augmentation | `scripts/preprocessing/augment_videos.py` makes RGB derivatives; `utils/keypoint_augment.py` provides synthetic skeleton perturbations. Both remain historical; derivatives are not independent originals. |
| Dependencies | Root pins Python >=3.9 and torch2.10/NumPy2.2.6/MediaPipe0.10.32. Default host Python3.14 had no ML stack and lacked ensurepip. Separate uv-managed Python3.11.17 environments were created; root requirements were not changed. |
| Device | Legacy `check_device` and harness `resolve_device` use CUDA availability. Physical NVIDIA RTX3050 6GB is present; V2 tested with torch2.10.0+cpu, which reports CUDA unavailable. CUDA runtime validation remains outstanding. |
| Ignore rules | `docs/` was entirely ignored; no broad `.pth` catch-all existed (some historical checkpoints deliberately tracked). Added scoped exceptions for gait Markdown and ignored only V2 external sources/checkpoints/private outputs. |
| Tests | `tests/smoke_test.py` checks default wrapper output shapes; `tests/test_future_work.py` includes feature parity, TCN equivalence, augmentation, metrics, signatures, and a synthetic **training** harness. The latter is excluded from the final no-training test command. |
| Reuse | Reuse scientific observations, anatomical constants after audit, FFmpeg, existing E0 parity tests and frozen model code. Do not reuse legacy identity-search/deepfake labels, nominal-time caches, or interpolation semantics in V2. |

`CLAUDE.md` requests graphify queries/updates. `graphify-out/graph.json` exists,
but `graphify` is unavailable and `graphify-out/wiki/index.md` is absent. Both the
query and final update attempts are reported as tooling blockers; source inspection
was used instead. Historical graph contents are unchanged.

## C. Official pretrained model audit

### GaitPT

**VERIFIED FROM OFFICIAL SOURCE:**
[repository](https://github.com/UbiquitousCognitiveSystems/GaitPT), revision
`4977618a36ef77ee9a2863c9f8e27ab862894d70`;
[paper DOI](https://doi.org/10.1109/FG59268.2024.10581947).
[License](https://github.com/UbiquitousCognitiveSystems/GaitPT/blob/4977618a36ef77ee9a2863c9f8e27ab862894d70/LICENSE):
CC BY-NC-ND 4.0. External source is imported unmodified, not relicensed or vendored
under this repository's MIT license. Commercial use/redistribution of modifications
requires separate review; this report does not grant rights to checkpoints.

Official [checkpoint folder](https://drive.google.com/drive/folders/1gHouvtwEI7h0gylb5JjEwWZyu9iVMuNR):

| Artifact | Availability / action |
|---|---|
| GREW | Downloaded from file ID `1itpe5zwg-K4OtSKii7-83C7XH8tO9uAR`; 18,061,251 bytes; SHA256 `e6b287e2c2ae27a9344b8009077c9d4f70811d2d6f3ff20198b99bfa4f824637`; strict model load and forward **passed**. |
| CASIA-B | Folder lists `casia-checkpoint.pth`, ID `1HNfMJ413DkGwzWAIhD9ibiohxVxcT61w`, 18,118,851 bytes. Listed, not downloaded/loaded. |
| Gait3D | Official evaluation script exists; no Gait3D checkpoint found in the inspected folder. Availability **NOT YET VERIFIED**. |
| Source sample | GREW/Gait3D poses require author contact per README. CASIA archive listed at 295,510,456 bytes; full dataset not downloaded. Source-dataset reproduction **BLOCKED**. |

No matching pretrained checkpoint was present before download. TLS verification
was retained. SHA256 is a locally measured provenance checksum, not an independently
publisher-signed authenticity guarantee.

Official environment pins torch1.10.0/torchvision0.11.1, NumPy1.22.4,
SciPy1.7.3, pandas1.4.0, einops0.5.0, PyYAML6.0, vit-pytorch0.35.8,
pytorch-metric-learning1.6.3, easydict1.10. Installing those old pins into the
host Python3.14 stack was avoided. The isolated wrapper needs only the official
`gaitpt.py`, YAML config, torch, einops and NumPy and ran on torch2.10 CPU.
Official `main.py` constructs training loaders/optimizers even for evaluation;
V2 bypasses that entry point. Standalone embedding inference needs no training data;
benchmark reproduction still needs source evaluation data.

### GPGait and GPGait++

**VERIFIED FROM OFFICIAL SOURCE:**
[FastPoseGait](https://github.com/BNU-IVC/FastPoseGait), revision
`d3ce631971a01d806eeb3a154774d482c0951813`;
[GPGait DOI](https://doi.org/10.1109/ICCV51070.2023.01795),
[GPGait++ DOI](https://doi.org/10.1109/TPAMI.2025.3577594).
README restricts use to academic/noncommercial purposes; no standalone LICENSE
was found. Do not assume MIT inheritance from the local project.

The current checkout contains `modeling/models/gpgait.py`,
`modeling/models/gpgait++.py`, training configs and a checked training-code release
item. This **supersedes the documents' unreleased-code assumption**, not the need
to validate weights. Both project documents have explicit dated amendments.

[Standard model folder](https://drive.google.com/drive/folders/1jsQ5cZcZ5YXVvzSvyEETlN6l1B7T6paD)
was enumerated without downloading datasets. Vanilla GPGait links include GREW
`1_cCKtVR0v1s3fhLVECSIm9QjveAHRSLv`, CASIA-B
`1DxouTfJeuy9Ep0RuA-MAvV_WYH7MwGzf`, Gait3D and OUMVLP. **GPGait weights were not downloaded or loaded.**

[GPGait++ folder](https://drive.google.com/drive/folders/1NjY0gK89aXO4lFM_YyDiE336njNctHOI)
lists CASIA-B, CCPG, Gait3D, GREW and OUMVLP. GREW is listed at 1,308,128,306
bytes; it was not downloaded. Smaller CASIA-B file
`1FIIXgG7Dxwr8TObo_nN3pwsUhWVC-XjG` was downloaded: **63,712,570 bytes**,
SHA256 `87eb74fa0b2ce14dcff025d6f41fd1d4637afafd3af2aa0c6a06a690ff52a20d`.
`torch.load(weights_only=True)` fails on **unsupported GLOBAL getattr**.
Static unsafe-global inspection also fails (`Unsupported operand 48`). Restricted
loading was not disabled; no GPGait++ model weight-load success is claimed.

Official dependencies: Python>=3.9, torch>=1.8, torchvision, PyYAML, tqdm,
tensorboard, pytorch-metric-learning. Native `BaseModel` constructs dataset/DDP/CUDA
infrastructure. An isolated CPU test of the unmodified PAGCN component fails at
`unit_pagcn.py:43`, `A.cuda(x.get_device())`, with **Device index must not be negative**.
This component test used random initialization solely to reproduce a device error;
it is not pretrained inference. A CUDA environment or reviewed device-portability
adaptation plus safe checkpoint conversion is still needed.

**VERIFIED BY RUNNING CODE:** official GREW Affine→RescaleCenter→HOD→SkeletonInput
preprocessing produced finite `(1,6,90,17,1)` output on anatomical markers.
**NOT YET VERIFIED:** either challenger's real embedding shape, determinism,
checkpoint compatibility, source-sample performance, and standalone pretrained forward.
Both remain **BLOCKED**. The adapter fails explicitly instead of substituting random weights.

## D. Input contract findings

The following are source contracts, not assertions of Android-domain equivalence.

- COCO17 order: nose; left/right eye; left/right ear; left/right shoulder;
  left/right elbow; left/right wrist; left/right hip; left/right knee; left/right ankle.
- MMPose [COCO](https://github.com/open-mmlab/mmpose/blob/759b39c13fea6ba094afc1fa932f51dc1b11cbf9/configs/_base_/datasets/coco.py)
  and [Halpe26](https://github.com/open-mmlab/mmpose/blob/759b39c13fea6ba094afc1fa932f51dc1b11cbf9/configs/_base_/datasets/halpe26.py)
  definitions were checked by running name-order comparisons. Halpe's first17 match COCO;
  extra head/neck/hip/toes/heels are not used by these COCO contracts.
- MediaPipe33→COCO indices: `[0,2,5,7,8,11,12,13,14,15,16,23,24,25,26,27,28]`.
  This is an anatomical compatibility mapping, **not proof that detector distributions
  match GREW**. Sparse legacy remapped tracks cannot restore discarded eyes/ears.
- GaitPT head-upper-lower input order in COCO indices:
  `[0,1,3,0,2,4,5,7,9,6,8,10,11,13,15,12,14,16]`.
  Index3 repeats the nose exactly as official `CustomRepresentation`; no invented neck.
- GREW/Gait3D GaitPT use image pixels divided by **image height on both axes**,
  without hip centering. Confidence is dropped after V2 validation. CASIA divides
  by 320 instead and uses a different sequence length; the GREW adapter must not
  be used with CASIA weights. GREW default evaluation uses two centered 30-frame
  crops, with flip/mirror disabled. This implementation matches even the source's
  end-exclusive `n-1` second segment. Crops are not independent source recordings.
- Forward: GaitPT takes `(B,30,18,2)` and returns `(B,d)` normalized projections;
  **d=256 measured**, not hardcoded into core contracts. V2 normalizes the averaged
  crop embedding. Source evaluation averages normalized outputs without a final
  normalization; the final normalization is the PRD cosine-scoring convention.
- GPGait/GPGait++ GREW: input COCO17 pixel `(T,17,3)` (xy/confidence); affine
  spine correction; per-frame skeleton vertical extent scaled to225; shoulder
  midpoint centering; HOD joint/bone/angle channels; `(B,6,T,17,1)`.
  Confidence is dropped by HOD, not interpreted as depth. Missing values are
  rejected before upstream `nan_to_num`. GREW evaluation is `all_ordered`; training
  frame counts/sampling differ by model and are **not** copied into V2 inference.
  Native forward takes `(ipts,labs,types,views,seqL)`; returns
  `inference_feat['embeddings']`. Source heads suggest `(B,256,19)`, **not runtime verified**.
  CASIA++ evaluation omits Affine and has different channels/config; downloaded
  CASIA++ weights are not represented as GREW-compatible.
- Pose caches preserve actual frame PTS/index, native xy/confidence/observed mask,
  bbox, single-track ID, estimator hash, take/variant and source checksum. Display
  rotation is explicit and lossless in decoded frames; original bytes are unchanged.
- Quality is conservative and **provisional**: every mapped joint observed and
  confidence>=0.3, border margin2px, >=31frames, single continuous track, no timestamp
  gap exceeding3×median interval. No missing-joint interpolation. This all-frame
  rule may abstain too often. It is not validated for blur, actual walking motion,
  toe visibility, subtle track swaps, or 30-second moving-camera recordings.
- Native RTMPose ONNX interface inspects static input sizes and accepts explicit
  local detector/pose weights. Actual RTMPose extraction is **NOT YET VERIFIED**;
  those weight files were absent. Native MediaPipe IMAGE-mode baseline did run.
- Deterministic marker tests and a viewed skeleton-only plot confirmed anatomical
  index correspondence. Plot: `outputs/gait_verification/joint-mapping.png`.
  This does not establish source-detector semantic equivalence.

## E. Created and modified files

Existing files changed:

- `.gitignore`: expose only gait Markdown docs; ignore V2 weights/sources/caches/output.
- `pyproject.toml`: include new root package in packaging; no dependency version changes.
- Both required gait documents: append explicit stabilization override, current
  GPGait++ source-status correction and response naming clarification. They existed
  locally but were previously ignored; now visible as untracked Git content.

New runtime files:

| File | Purpose |
|---|---|
| `gait_verification/__init__.py` | Small public API |
| `contracts.py` | Media/pose/recording/model/embedding/quality/result types |
| `media.py` | Hashing, ffprobe actual PTS, orientation-aware decoding |
| `pose.py` | Native RTMPose/MediaPipe adapters, conservative tracking, immutable NPZ cache |
| `skeleton_contracts.py` | Anatomical mappings, provisional validation, GREW crop/normalization |
| `enrollments.py` | Per-take templates, weighted prototype, direct-claim memory/private JSON stores |
| `verifier.py` | One lookup, status policy, test-only thresholds, cosine decision |
| `evaluation.py` | Deterministic trial manifest, split/lineage checks, tie-aware metrics, grouping, probe-subject bootstrap |
| `stabilization.py` | Opt-in new-directory FFmpeg runner, hashes/PTS/commands/failure manifests |
| `backends/__init__.py` | Backend namespace |
| `backends/fastpose_worker.py` | Packaged subprocess worker for upstream imports |
| `backends/gaitpt.py` | Revision/hash-pinned strict frozen official GREW wrapper |
| `backends/gpgait.py` | Isolated native preprocessing and explicit unavailable encoder |

Other new files:

- `configs/gait_verification/{phase0,models,capture_schema}.json`: null threshold,
  candidate status/hashes, capture fields (including session/operator/consent/time).
- `requirements-gait-v2.txt`: separate pinned environment; no legacy environment changes.
- `scripts/gait_verification/audit_checkpoints.py`: safe read-only inventory.
- `scripts/gait_verification/fastpose_worker.py`: isolate upstream `data`/`utils` namespace collisions.
- `scripts/gait_verification/inspect_pose_contract.py`: skeleton-only mapping image.
- `scripts/gait_verification/prepare_pose_cache.py`: single-video native cache CLI.
- `scripts/gait_verification/smoke_encoder.py`: real-weight frozen/determinism smoke.
- `scripts/gait_verification/stabilize_pilot.py`: explicit single-source raw/s10/s15 runner.
- `scripts/gait_verification/validate_capture_manifest.py`: new-cohort field validation.
- `scripts/gait_verification/verify_claim.py`: `(video, claimed-id)` public status demo;
  fails closed pending validated model/policy installation, **not a production verifier**.
- `tests/gait_verification/test_foundation.py`: synthetic contract/claim/leakage/metric tests.
- `tests/gait_verification/test_media.py`: actual FFmpeg VFR timing regression.
- This report.

Ignored local artifacts include downloaded models, private pose caches, mapping image,
raw/s10 pose comparison, failed and successful stabilization directories. Existing
RGBs, E0 code, old checkpoints, cached historical features and reports are untouched.

## F. Commands and test evidence

Reproducible primary commands (use isolated Python; source checkout paths are explicit):

```bash
uv venv --python 3.11 .venv-gait-v2-clean
uv pip install --python .venv-gait-v2-clean/bin/python --index-strategy unsafe-best-match -r requirements-gait-v2.txt
.venv-gait-v2-clean/bin/python -m pytest -q tests/gait_verification tests/test_future_work.py -k 'not harness_end_to_end_on_synthetic_subjects and not metrics_against_known_values'
.venv-gait-v2-clean/bin/python tests/smoke_test.py
.venv-gait-v2-clean/bin/ruff check gait_verification scripts/gait_verification tests/gait_verification
```

Model audit source clones were created at `/tmp/gait-v2-GaitPT`,
`/tmp/gait-v2-FastPoseGait`, `/tmp/gait-v2-mmpose`, each pinned to revisions above.
For later use, clone into ignored `.cache/gait_verification/` and checkout those
exact revisions. No external source edits were made.

```bash
.venv-gait-v2-runtime/bin/python scripts/gait_verification/smoke_encoder.py \
  --source /tmp/gait-v2-GaitPT \
  --checkpoint .cache/gait_verification/checkpoints/grew-checkpoint.pth \
  --sha256 e6b287e2c2ae27a9344b8009077c9d4f70811d2d6f3ff20198b99bfa4f824637
# Repeat with actual cached historical poses:
.venv-gait-v2-runtime/bin/python scripts/gait_verification/smoke_encoder.py \
  --source /tmp/gait-v2-GaitPT \
  --checkpoint .cache/gait_verification/checkpoints/grew-checkpoint.pth \
  --sha256 e6b287e2c2ae27a9344b8009077c9d4f70811d2d6f3ff20198b99bfa4f824637 \
  --pose-cache outputs/gait_verification/historical-native-pose.npz
```

Evidence:

- Initial V2 fixtures:23 passed; after additions, full combined legacy/V2 run:
  36 passed (includes the unauthorized synthetic-training test described above).
  Final no-training selection and clean environment outcome recorded below.
- E0 smoke: passed; TCN parity/133,058 count passed; no historical metric replay claimed.
- Official GaitPT crop/reorder code vs V2: numerical parity passed (`atol=1e-7`).
- MMPose COCO/Halpe joint-name parity passed.
- GaitPT synthetic and historical real-weight inference: passed, normalized d256,
  exact repeat consistency, frozen state unchanged. Source sample remains unavailable.
- Original historical media probe: `A2_F1.mp4`,1920×1080 encoded, rotation-90,
  nominal30fps,190frames, actual PTS span6.296211s. No uniform FPS clock substituted.
- Historical pilot `Aarav_F3.mp4`:78frames; displayed1080×1920; native MediaPipe cache
  roundtrip exact PTS. MediaPipe Lite hash
  `59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a`.
- Stabilization first attempt correctly failed max relative timestamp error0.002066s.
  `-enc_time_base -1` fixed it; successful pilot logged0s error/offset and78/78frames.
  Failed artifact retained. Synthetic VFR test initially assumed absolute PTS identical;
  corrected to test explicitly logged origin offset, since a constant offset is permitted.
- Raw/s10 provisional pose gate: both usable, valid mapped-joint fraction1.0.
  Mean ankle confidence0.96687 vs0.96482; left-shank length CV0.30149 vs0.30355.
  These single-clip pose diagnostics **do not show a stabilization benefit** and are
  not calibrated quality limits or recognition metrics.
- `verify_claim.py --video ... --claimed-id S999`: `CLAIM_NOT_ENROLLED`, null verdict,
  exit2 (intentional non-decision), no other identities/scores.
- GPGait++ restricted load: failed as recorded above. GPGait CUDA-component CPU
  probe: failed as recorded above. GREW HOD preprocessing: passed.
- Initial legacy test collection failed missing cv2; after isolated installation it passed.
  Initial lint issues were unused import/semicolon formatting and were corrected.
- `git diff --check` passed; diff of historical `models`, `utils`, training/inference,
  old outputs and existing tests is empty. `graphify` commands unavailable.

## G. Remaining blockers and unvalidated requirements

1. **Source-sample gate:** GREW checkpoint-native source pose reproduction not done;
   authors' coordinate extraction/crop conventions cannot be established from tensor
   shape alone. Android-domain compatibility remains an empirical question.
2. **Challengers:** GPGait weights not loaded; GPGait++ restricted pickle load blocked;
   CUDA runtime or reviewed CPU portability needed. Do not claim measured output d.
3. **Pose:** RTMPose detector/pose ONNX weights absent; no RTMPose video smoke.
   Provisional single-person continuity heuristic is not a complete multi-person tracker.
   Full 30s motion/blur/feet/border quality, window rejection, and confidence gates need pilot.
4. **Calibration:** no threshold/model-selection/calibration data or approved operating FAR;
   policy remains null, public CLI cannot issue real TRUE/FALSE yet. Template authorization,
   consent lifecycle/encryption and full service deployment are future work.
5. **Evaluation:** trial/metric primitives exist, but full E0/GaitPT/GPGait orchestration,
   locked trial persistence, fair cohort comparison, two-way subject uncertainty and
   cross-session evaluation are unvalidated. Probe-subject bootstrap is explicitly
   limited when claimed subjects are reused across trials.
6. **Stabilization:** only one existing short frontal clip tested; s15 config tested,
   no s15 biometric comparison. Border/feet visual QA over four views and 30-second
   moving-camera acquisition is not complete. No stabilization winner established.
7. **Licenses:** noncommercial/academic restrictions and GaitPT no-derivatives condition
   remain relevant for any sharing/deployment; checkpoint-specific distribution grants
   were not separately supplied.
8. **Environment:** original official dependency stacks not reproduced. Modern CPU
   GaitPT compatibility is demonstrated; GPU and full official evaluators are not.
   graphify absent. The initial V2 env contains overlapping cv2 distributions; a clean
   pinned environment is separately checked below, not assumed equivalent by install alone.

## H. Exact Phase 1 readiness

| PRD gate | Status |
|---|---|
| P0.1 preservation/audit | Implemented; E0 source/artifacts unchanged, with explicit synthetic-test scope deviation |
| P0.2 checkpoint contract | GaitPT engineering inference works; production READY gate **BLOCKED**; both challengers **BLOCKED** |
| P0.3 media/pose/cache | Native MediaPipe/PTS pipeline demonstrated on historical clip; RTMPose + robust quality/tracking unfinished |
| P0.4 verification | Claim-only contract and mock decisions tested; no real calibrated policy |
| P0.5 evaluator | Deterministic primitives/leak tests/metrics ready; cohort runner and final uncertainty protocol unfinished |
| P0.6 stabilization | Raw/s10 historical pilot and timestamp regression passed; four-view capture pilot outstanding |
| P0.7 packaging/demo | Isolated package and engineering scripts usable; real calibrated end-to-end verification is not ready |

**Ready to begin Phase 1 capture-pilot engineering**, with the available media/cache/
stabilization tools. **Not ready to declare Phase 0 scientifically complete, enroll
production users, freeze a recording SOP, or start comparative recognition claims.**
Next: acquire/source an authorized native source-pose sample; validate RTMPose weights
and full-body tracking on 2–3 pilot volunteers; resolve challenger loading/runtime;
review four-view raw/s10/s15 quality; then freeze the capture SOP and development split.

## I. Research integrity

Historical observations, current official-source facts, executed engineering tests,
implementation assumptions, and unperformed empirical experiments are separated above.
Cosine scores are not probabilities. A FALSE verdict does not prove altered pixels.
No 1:N functionality was imported into the V2 public API. No data from the planned
cohort, new-day sessions, model superiority, or final recognition accuracy was invented.
No public upload of videos/poses/embeddings occurred. The accidental synthetic legacy
training tests are disclosed; pretrained inference itself remained frozen.


### Final verification addendum

- Fresh `.venv-gait-v2-clean` installed the pinned requirements on Python3.11.17.
  RTMLib transitively requires `opencv-python` while MediaPipe requires contrib;
  both are now explicitly pinned to **5.0.0.93** in this separate environment,
  avoiding the initial mismatched4.12/5.0 versions. Root legacy pins remain untouched.
  `uv pip check` reports all64 installed packages compatible.
- Final selected regression command: **34 passed, 2 deselected**, one upstream
  Transformer nested-tensor warning. The excluded tests are the synthetic training
  harness and synthetic Platt-fitting metrics fixture. An earlier35-pass run had
  excluded only the neural harness; the final command performs neither fitting task. E0 standalone smoke passed. Ruff and compileall passed.
- Fresh-environment real GaitPT forward on the historical cache passed again:
  d256, norm0.99999994, repeat delta0, frozen state unchanged. Encoder recipe IDs
  also bind pose-estimator and raw/stabilized variant provenance.
- Packaged isolated FastPoseGait HOD worker returned `(1,6,90,17,1)` successfully.
  New capture-schema synthetic valid-row and duplicate-original checks passed.
- `git diff --check` passed. Existing tracked modifications are only `.gitignore`
  and package discovery in `pyproject.toml`; V2 additions and previously ignored
  gait docs are untracked pending user review. `AGENTS.md` remains the original
  user-provided untracked file. No commit or staging was performed.
- Confirmed downloaded checkpoints and private generated pose/media artifacts
  are ignored. `graphify update .` failed: command not found.
