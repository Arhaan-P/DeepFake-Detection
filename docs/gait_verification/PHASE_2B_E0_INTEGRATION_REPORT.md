# Phase 2B — Frozen E0 integration and common-model evaluation

**Date:** 12 October 2026 (Asia/Kolkata). **Author:** Codex.
**Branch:** `feat/gait-verification-v2`. **HEAD:** `433cdd0e4bb3c075ab39731254e1e108fbbfe096`.

**PHASE 2B ENGINEERING: COMPLETE.** The frozen shipped-head and common-trial scope
passes final regression; see §16.
Scientific and deployment readiness remain BLOCKED.

## 1. Executive summary

Implemented an additive, frozen historical temporal-CNN scorer on the existing
`ClaimRequest` interface. E0 uses its own MediaPipe33 normalized-image xyz extraction
and historical 60×78 representation. The four pretrained encoders retain their native
RTMPose/skeleton/embedding paths. All use the same validated verification manifest.

The shipped historical checkpoint safely supplies the decisive 133,058-parameter
head and its original z-score tensors. Feature and decisive-logit parity are tested
against historical source. Real MediaPipe extraction and frozen CUDA inference work.
No training, fitting, collection, threshold, winner, final-test access or deployment
occurred. Successful E0 scoring returns `POLICY_NOT_READY`, `verdict=null`.

**Important distinction:** the controlled LOSO experiment's `E0_baseline.pt` is absent.
This integration is labeled **historical-local-13-subject-shipped**, not a replay of
the controlled experiment's AUC/EER or fold checkpoints. Historical metrics remain
historical. The frozen shipped head is an engineering baseline, not evidence of
unseen-subject generalization or new-cohort accuracy.

## 2. Repository and Git state

The required branch was already active. Phase 2 was uncommitted: nine tracked files
modified and nine new files present. Those changes were preserved. `CLAUDE.md` is
absent. All mandatory documents and relevant source/configuration/tests were opened
before implementation. The deferred register received OPEN E0-01…E0-10 before code
edits. No staging, commit, push, merge, reset, clean or history change occurred.

Initial status/diff and SHA-256 snapshots are in `/tmp/gait-phase2b-preservation/`.
Final preservation checked **168** historical source/checkpoint/original-video files:
zero changes. Historical Git paths also have no diff. The new tensor-only checkpoint,
private evidence and other V2 artifacts remain ignored and untracked.

## 3. Phase 2 handoff verification

Both permitted baselines were reproduced before source edits:

| Baseline | Passed | Skipped | Deselected | Failed |
|---|---:|---:|---:|---:|
| Default | 303 | 9 | 2 | 0 |
| `GAIT_V2_CUDA_TESTS=1` | 312 | 0 | 2 | 0 |

Reused the managed capture registry, lineage/source checks, protected split guard,
trial generation/validation, claim-only request, offline runner, metrics/coverage,
paired comparisons, canonical JSON seals, private exclusive writes, quarantine and
append-only ledger. The existing four model integrations were preserved.

## 4. Historical architecture audit

- `models/verifier_variants.py::TCNVerifier` uses `[q-r, abs(q-r), q*r]`:
  60×234 pair features, Conv1D kernels 7/5/3, channels 64/64/32, BatchNorm/ReLU,
  adaptive average pool, Linear 32→32→2; dropout 0.1. **133,058 parameters**.
- `models/full_pipeline.py` has the identical `diff_conv` / `diff_classifier`
  decisive path. CNN/BiLSTM/Transformer embeddings and identity-verifier diagnostics
  do not contribute to those logits. The adapter loads only the decisive weights.
- `utils/verification_harness.py` and `utils/gait_verifier.py` historically average
  normalized reference features into signatures. Historical class 1 denotes a
  genuine pair; `softmax(logits)[:,1]` increases with claim support.
- The legacy `GaitVerifier.verify()` loads/scores all identities and can expose a
  best match. Its API, identity enumeration, thresholds, Platt parameters and
  attribution behavior are deliberately excluded from V2.
- `scripts/future_work/train_final.py` describes a controlled bundle, but its output
  directory/checkpoint is absent. It was read, never executed.
- The existing shipped inference loader uses `weights_only=False`; it was read,
  never invoked. Historical source and checkpoints were not modified.

## 5. Actual checkpoint and preprocessing contracts

| Artifact/fact | Verified value |
|---|---|
| Shipped source | `outputs/checkpoints/checkpoint_epoch_46_best.pth` |
| Original bytes | 4,784,940 |
| Original SHA-256 | `6c37f3b9b9172b42e6329a310f3e51d0b9882a5a9d59806851687731d67b1589` |
| Original state entries | 127 |
| Decisive state entries | 25, including BatchNorm buffers |
| Native input | query and references each 60×78 float32 |
| Normalization | original checkpoint `feature_stats.mean/std`, each length 78; strictly positive std |
| Extracted SHA-256 | `f65c44cde5af178b154efbcf47b9aa18dc67c37db19b63e25d27b9a781be65fd` |
| Original model config SHA-256 | `bc21187735bd19aba051ca78c791ee538bc20a81f87d995c6d95899bd3604d1a` |

The twelve joints, in order, are left/right shoulder, hip, knee, ankle, heel and
foot-index: MediaPipe indices **11,12,23,24,25,26,27,28,29,30,31,32**.
The feature order is 36 hip-relative xyz coordinates; six image-plane angles in
degrees (left/right knee, hip, ankle); 36 adjacent-coordinate differences.
Differences repeat the **last** difference, not a zero/first row. They are not
physical velocities. Detected poses are linearly resampled by index to 60 steps;
the historical minimum is ten detections. No timestamp-aware replacement,
body-scale normalization, gap filling or fabricated heel/toe coordinates is added.

MediaPipe Pose Lite IMAGE mode uses detection/presence confidence 0.5 and native
normalized-image xyz, including the estimator's z coordinate (not metric depth).
The worker reuses `GaitFeatureExtractor.process_video` with a local model-path
override and the verified vectorized baseline math. Decoder/orientation behavior
remains historical OpenCV behavior. Actual source timestamps remain the media's
timestamps; E0's internal resampled steps are never represented as physical time.

## 6. Frozen loading and deserialization security

The complete source fails `weights_only=True` on NumPy scalar/dtype globals in
training metrics. No unrestricted load, custom unpickler, safe-global allowlist or
deserializer patch was used.

`e0_checkpoint.py` accepts only the exact original hash, size and pickle hash
`f031326216030ec2164d7600960ced35314be09958605634f645d0d4a5794200`.
Static opcode inspection retains the complete model-state prefix before offset
22186 and tensor statistics from offset 27785. Two statistics references to the
omitted optimizer's memo entry 1352 are replaced with the exact original literal
`cpu`. Stack/memo disassembly and the existing tensor-global restriction validate
the reconstructed pickle. PyTorch's **unchanged `weights_only=True`** loads it.
Optimizer, scheduler and NumPy metric objects are never deserialized.

Only decisive weights and statistics are resaved to a new tensor-only artifact.
Original ZIP CRCs pass; original bytes remain unchanged. The CLI pins the extracted
hash via `configs/gait_verification/e0.json`. The isolated worker rechecks the hash,
envelope, source/config identity, tensor types, statistics, dimensions and strict
weight compatibility, then uses eval, disabled gradients and inference mode.
It verifies repeated logits and unchanged state. TF32 is explicitly disabled for
float32 parity. An unsafe-checkpoint fixture proves its payload is never executed.

## 7. Native extraction and cache integration

`e0_processing.py` stores **un-z-scored** native features under a separate feature
root using Phase 2's seal/exclusive-write/quarantine/ledger primitives. No historical
pickle feature or enrollment cache is loaded or promoted to managed study data.

Keys and sealed records bind asset ID, original take, original source SHA, raw variant,
parent lineage, pose model SHA, IMAGE/detection settings, decoder convention,
joint/block/angle ordering, 60-step resampling, normalization semantics, historical
source/worker hashes and NumPy/MediaPipe/OpenCV versions. The model's checkpoint and
z-score tensors are separately bound at scoring; they do not contaminate reusable
un-normalized feature fingerprints.

Wrong source/recipe/lineage, stripped seals, malformed shape, missing values and
NaN/Inf are refused. Processing preserves corrupt records by quarantine before
recomputation. Valid reruns leave artifacts unchanged and load no extractor.
Sources are checked before and after extraction, and at scoring even on cache hits.
Registered originals cannot sit inside the output root. Interrupted work resumes.
Unknown/denied selections cannot falsely report COMPLETE.

## 8. E0 scorer adapter architecture

`E0TrialScorer` accepts the unchanged `ClaimRequest` and returns the existing
`(status, reason, score, verdict, quality_evaluable)` tuple. It resolves query lineage
through the capture registry, resolves only the claimed person's explicitly listed
originals, checks roles and independent sources, validates feature artifacts, then
calls `FrozenE0`. Missing checkpoints/features, source changes, corruption and
insufficient native poses produce explicit non-Boolean outcomes.

Legacy model/math imports stay in `scripts/gait_verification/e0_worker.py`, outside
the V2 package. The Phase 1D import-isolation regression remains intact. Worker
cold starts intentionally favor isolation and simplicity; no persistent inference
service was introduced. Model loading happens after usable artifacts exist.

## 9. Claim-only enforcement

No actual query subject or genuine/impostor label is an input to the scorer or worker.
The registry's subject metadata is used for split authorization and reference
ownership, never as an identity hint to the model. Reference loads are checked
against the requested claim and exact take set. No alternative identities,
rankings or gallery scores are returned. Public results retain only the claim,
status, null verdict and schema. Per-trial raw scores remain restricted artifacts.

For offline E0, the validated trial's explicit reference originals define its
reference set; preparation does not create production enrollments. Missing reference
artifacts return `CLAIM_NOT_ENROLLED`. An empty/unregistered claim has no references.

## 10. Deterministic reference aggregation

Version **`mean-zscored-reference-feature-signature-v1`**: z-score each independent
reference with the frozen checkpoint statistics, average its feature tensors, then
score the query against this single signature. This matches the historical
signature approach, preserves every requested original and uses no optimized
pair-score aggregation. Tensor ordering is canonical. Tests verify permutation
invariance and parity with the historical normalized-mean decisive path.

## 11. Common-trial orchestration

`run_common_evaluation` sends exactly the same requests to each selected scorer.
`run-common` supplies explicit per-system artifact locations, writes separate run
reports/outcomes, and produces a comparison. Trial IDs, labels, sources, views,
reference sets and roles remain bound by the original manifest and its validation.
E0 runs on originals only; derivative/shared-original requests are refused rather
than silently matching a raw recipe. Pretrained processing continues through the
existing Phase 2 processing/enrollment path.

The report includes both pair-specific scoreable subsets and the intersection
scoreable by **all primary systems**. CASIA-B appears as a standalone labeled
ablation with full coverage/metrics, outside the primary intersection/comparisons.

## 12. Pretraining-regime comparison policy

Default **`matched-pretraining-v1`** retains the existing refusal of different
pretraining datasets. Explicit **`historical-e0-versus-grew-architecture-v1`** permits
only the audited shipped E0 origin versus GREW, plus GREW-matched pairs. It checks
E0 family, source hash, training origin and score semantics; evidence classes and
trial inputs must match. CASIA-B/unknown origins/policies and final-test rows are
refused. Reports state **training populations differ** and disallow causal claims
of architecture superiority. No guard was globally relaxed.

## 13. Native scores and preprocessing differences

E0 reports **logit(class 1) − logit(class 0)**; higher supports the claim. Its sigmoid
equals historical class-1 softmax mathematically, but neither is called calibrated
probability. Keeping the unsaturated margin avoids unnecessary softmax rounding.

GaitPT-GREW uses height-scaled native skeleton crops and 256-D embeddings. GPGait
and GPGait++ GREW use native Affine/RescaleCenter/HOD/SkeletonInput and 4,864-D
embeddings. CASIA-B omits GREW's Affine and is separate. Those systems retain
their own cosine/prototype semantics, pose/window recipes and individual reports.
No E0/cosine fusion, shared numeric threshold or cross-model embedding pooling exists.

## 14. Coverage and abstentions

Full-manifest score, query-quality and operational coverage remain separate.
`POLICY_NOT_READY` with a finite score counts as scored and quality-evaluable, with
zero operational decisions; it is not a quality failure. Every original trial and
its genuine/impostor status/reason count remains in the output. All-abstention and
one-class metrics remain explicitly undefined. No operational FAR/FRR/confusion
counts are reported without a threshold. CLI FAR arguments request descriptive
mathematics only; synthetic values do not choose an owner's operating point.

## 15. CLI commands and examples

Use `.venv-gait-v2-cuda/bin/python scripts/gait_verification/evaluate_claims.py` (`CL`
below). Paths below are examples. Outputs are exclusive; choose fresh report/run
paths on rerun. Processing reuses valid caches; scoring reads them without fitting.

```bash
CL=".venv-gait-v2-cuda/bin/python scripts/gait_verification/evaluate_claims.py"
$CL e0-audit --extract-to .cache/gait_verification/checkpoints/e0-shipped-head-v1.pt \
  --out outputs/gait_verification/e0-extraction.json
$CL e0-validate-features --native-pose fixture-mediapipe33.npz \
  --out outputs/gait_verification/e0-native-validation.json
$CL e0-features --registry REGISTRY.json --split SPLIT.json --data-root DATA \
  --assets-file originals.txt --e0-feature-store WORK/e0-features \
  --out outputs/gait_verification/e0-processing.json
$CL run --model e0 --registry REGISTRY.json --split SPLIT.json --trials TRIALS.json \
  --data-root DATA --e0-feature-store WORK/e0-features \
  --e0-checkpoint .cache/gait_verification/checkpoints/e0-shipped-head-v1.pt \
  --e0-sha256 f65c44cde5af178b154efbcf47b9aa18dc67c37db19b63e25d27b9a781be65fd \
  --evidence-class SYNTHETIC_FIXTURE --target-far 0.05 --out RUNS/e0
$CL run --model gaitpt --registry REGISTRY.json --split SPLIT.json --trials TRIALS.json \
  --enrollments ENROLLMENTS --query-store PREPARED_GAITPT_QUERIES \
  --evidence-class SYNTHETIC_FIXTURE --target-far 0.05 --out RUNS/gaitpt
$CL run-common --registry REGISTRY.json --split SPLIT.json --trials TRIALS.json \
  --data-root DATA --systems systems.json --evidence-class SYNTHETIC_FIXTURE \
  --target-far 0.05 --comparison-policy historical-e0-versus-grew-architecture-v1 \
  --out RUNS/common
$CL compare --run RUNS/e0 --run RUNS/gaitpt --target-far 0.05 \
  --comparison-policy historical-e0-versus-grew-architecture-v1 --out comparison.json
```

Native fixtures must contain `pose[N,33,3]` and Boolean `observed[N,33]`; anatomical
observations cannot be guessed from COCO17. This validates features, not RGB quality.
`systems.json` contains `systems` entries: E0 needs `model`, `feature_store`,
`checkpoint`, `sha256`; pretrained entries need `model`, `enrollments`, `query_store`.
Use distinct entries for `gaitpt`, `gpgait`, `gpgaitpp`; optionally add
`gpgaitpp_casia`. Existing `process --model` commands prepare pretrained queries;
E0 does not replace them. `compare` accepts two runs or an all-primary multi-run
comparison with three or more. No CLI offers final-test authorization or a threshold.

## 16. Tests and evidence

| Check | Actual outcome |
|---|---|
| Phase 2 baselines | 303 passed / 9 skipped; CUDA 312 passed; both 2 deselected |
| New Phase 2B tests, default | 61 passed, 1 opt-in CUDA skipped |
| Final full default | **364 passed, 10 skipped, 2 deselected, 0 failed** |
| Final full CUDA | **374 passed, 0 skipped, 2 deselected, 0 failed** |
| E0 CUDA focused after correction | 1 passed, 18 deselected |
| Ruff / compileall / Git whitespace | Passed |
| Historical preservation / private artifacts | 168 files unchanged; zero private V2 tracked files |

Full commands:

```bash
.venv-gait-v2-cuda/bin/python -m pytest -q tests/gait_verification tests/test_future_work.py \
  -k "not test_harness_end_to_end_on_synthetic_subjects and not test_metrics_against_known_values"
GAIT_V2_CUDA_TESTS=1 .venv-gait-v2-cuda/bin/python -m pytest -q \
  tests/gait_verification tests/test_future_work.py \
  -k "not test_harness_end_to_end_on_synthetic_subjects and not test_metrics_against_known_values"
.venv-gait-v2-cuda/bin/ruff check gait_verification scripts/gait_verification tests/gait_verification
.venv-gait-v2-cuda/bin/python -m compileall -q gait_verification scripts/gait_verification tests/gait_verification
git diff --check
```

Both prohibited legacy training/calibrator-fitting tests remained deselected throughout.
Evidence separates pure math/security unit cases, invented managed cohort/scorer/CLI
fixtures, real checkpoint/source parity, real MediaPipe/CUDA smoke, and unperformed
scientific evaluation. No mathematical fixture is biometric accuracy evidence.

RED tests first failed on missing modules. Intermediate fixes addressed a normalized
mean versus raw mean rounding assumption, role refusal before any path resolution,
the existing metric schema's absence of thresholded FAR, and fixture-import lint.
CUDA initially differed from CPU by 0.003128 with TF32; disabling TF32 reduced the
fixture delta to 0.000015259. Cross-device parity uses float32 `rtol=1e-6, atol=1e-5`;
same-device repeat/state checks remain exact. Final tests do not bypass failures.

## 17. Measured inference and parity

Python 3.11.17, torch 2.10.0+cu128, NVIDIA RTX 3050 Laptop GPU 6GB.
Fresh native extraction on preserved `Arhaan_S1.mp4` / `Arhaan_S2.mp4`, followed by
the frozen shipped head on CUDA, returned a finite score; 133,058 parameters,
eval mode, gradients disabled, repeat delta 0 and identical pre/post state digest
`c82fc47b6260f0b7550e8367a83cf349e4a6066cf2bc88cb68b93db4d2aff2e0`.
The first smoke's **31.947 s** wall time includes both extractions and worker
startup/inference; it is not forward latency or a 30-second-cohort benchmark.

Private evidence: `outputs/gait_verification/phase2b-e0-extraction.json` and
`outputs/gait_verification/phase2b-smoke/historical-inference.json`. That first
smoke preceded the numerical TF32 correction; corrected behavior is independently
covered by the final CUDA test. No raw score is presented as accuracy/confidence.

Tests compare native features against `GaitFeatureExtractor` and single/multiple
reference logits against the historical decisive modules on identical tensors,
using the real safely extracted weights. The source wrapper's unused branches
cannot affect those logits. **Controlled LOSO numerical replay is not established**:
its trained fold/bundle weights are absent. New-cohort empirical parity is pending.

## 18. Phase 2B changed files

New: `configs/gait_verification/e0.json`;
`gait_verification/{e0,e0_checkpoint,e0_processing,e0_scorer,common_eval}.py`;
`scripts/gait_verification/e0_worker.py`;
`tests/gait_verification/test_phase2b_{e0,integration,common,cli}.py`; this report.

Extended: `gait_verification/offline_eval.py` (explicit comparison contract and
full coverage); `scripts/gait_verification/evaluate_claims.py` (existing CLI);
the deferred register and both main document amendments. The original uncommitted
Phase 2 files remain. No historical code/checkpoint/video/report was rewritten.

## 19. Engineering limits and blockers

No blocker remains for the frozen shipped-head adapter/common-run engineering
scope after final checks. Missing external checkpoints/features are explicit runtime
outcomes. The absent controlled LOSO bundle prevents replay of that separate
scientific artifact; no weights were invented or retrained to replace it.

The historical MediaPipe minimum/finite-landmark gate is provisional and less rich
than V2's RTMPose temporal quality. It is not a validated walking/liveness, crop,
occlusion or multi-person safety gate. MediaPipe's historical single output cannot
certify absence of bystanders. This remains a scientific validation requirement,
not a claim that native landmark existence establishes useful gait.

E0's isolated worker currently reloads weights per trial and checks repeated outputs
each time. Cold-start overhead and future 30-second throughput are not optimized or
benchmarked. Concurrent writers/power-loss durability inherit Phase 2's limits;
use one active writer per cache root. New E0 processing has no final authorization
path and refuses derivatives, rather than defining a new stabilization experiment.

## 20. Scientific limitations and deferred requirements

All EXT/OWN/SCI/SEC items remain OPEN. Pilot, consent-backed cohort, final SOP,
split counts/seeds, impostor policy, FAR target, retraining/adaptation permission,
new-cohort E0 parity, calibration, winners, locked evaluation and deployment are
unresolved. Historical training membership/subject overlap must be assessed under
a future approved protocol. Single-day data cannot establish cross-day reliability.
Scores, fixture metrics and CUDA success do not establish identity accuracy, fraud,
synthetic pixels, scientific model selection or deployment fitness.

## 21. Security and licensing limits

Seals/checksums detect corruption; they are not signatures/authenticated storage.
Stores rely on filesystem permissions, without encryption or access auditing.
Legacy pickle caches are not trusted imports. Historical unknown recording dates
and consent were never fabricated or registered as study data; the historical smoke
remains a separate lower-trust engineering observation.

E0 remains research-only. Repository source licensing does not establish historical
biometric consent/weight redistribution rights. GaitPT CC BY-NC-ND and FastPoseGait
academic/noncommercial restrictions, pose/checkpoint permissions, privacy/retention
and deployment approval remain unresolved. All four pretrained candidates remain
BLOCKED, `production_ready=false`, with null thresholds. No dependencies or datasets
were installed/downloaded in Phase 2B.

## 22. Readiness for the next architecture-only phase

The validated scorer/native feature/cache/common-manifest/reporting interfaces can
be used for the next architecture-only review without a pilot or new cohort.
No scientific setting is selected on the owner's behalf. Production Boolean
verification and scientific comparison remain blocked by the deferred evidence.

## 23. Exact recommended next task

Perform an independent pre-collection architecture acceptance audit of Phases 0–2B:
exercise the operator workflow with synthetic fixtures, review the shipped-versus-
controlled E0 distinction, cache/lineage/security contracts and provisional quality
limitations, and reconcile any remaining agreed software requirements. Continue
architecture work as needed before requesting collection. Do not train, calibrate,
choose a winner or open final-test data under this handoff.
