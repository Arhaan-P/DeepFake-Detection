# Two-subject pilot — implementation report

**Date:** 20 October 2026. **Branch:** `feat/gait-verification-v2`, HEAD `eac8b70`.
**Author:** Claude Code, continuing an uncommitted start (intake/protocol/augmentation
modules and tests) left in the worktree by an earlier session; that work was audited,
kept and completed. Nothing was committed, staged, pushed, reset or cleaned.

**TWO-SUBJECT PILOT ENGINEERING: COMPLETE.** Real pilot evidence: **none** — no
recording has been provided. Scientific and deployment readiness remain **BLOCKED**.

**TWO-PERSON EXPLORATORY PILOT — NOT A VALIDATED BIOMETRIC PERFORMANCE RESULT.**

## 1. Scope readback

Only 1:1 claimed-identity verification. A query is compared only with the claimed
subject's enrolled independent originals; no identification, ranking, gallery search
or deepfake attribution. Without an authorized policy every successful score returns
`POLICY_NOT_READY`, `verdict=null`. E0 is the historical shipped head
(`historical-local-13-subject-shipped`), not the absent controlled LOSO model. GaitPT,
GPGait, GPGait++ GREW and CASIA-B stay frozen, separate and BLOCKED. RAW originals are
immutable; every derivative is a lineage-bound analysis copy. No training, fitting,
threshold, winner, locked-final access or deployment occurred.

**Documents read in full:** `AGENTS.md`, the PRD and handoff, the Phase 2, 2B and 2C
reports, the Phase 2 operator guide, the deferred register and the capture SOP draft.
`CLAUDE.md` is absent. Relevant source and tests were inspected before editing.

**Contradictions found:** none with repository code. Two tensions are resolved
conservatively. (1) The owner's matched-variant request conflicts with the Phase 1C
rule that only RAW originals can be references, so matched-variant scoring is
**restricted** (`RECIPE_MISMATCH`) rather than relaxed. (2) Batch enrollment through
the scientific CLI requires a LOCKED split, while the pilot must never be locked. The
pilot therefore calls the same `enroll_registered` library function under a DRAFT
development-only split. That function's own role guards are unchanged.

## 2. What already existed and was reused

| Need | Reused component |
|---|---|
| Registry, lineage, duplicate bytes, source checks | `capture_registry`, `dataset_processing.source_issue/protect_sources` |
| Media facts and full decode | `media.probe_video/decode_frames` |
| S10/S15 and derivative registration | `stabilization.stabilize`, `dataset_processing.derive_variants` |
| Pose, quality, embeddings, resume, GPU slots | `dataset_processing.build_plan/run_processing/coverage_report` |
| E0 native features and scorer | `e0_processing.process_e0/E0FeatureStore`, `e0_scorer.E0TrialScorer`, `e0.FrozenE0` |
| Enrollment | `pipeline.enroll_registered`, `DirectoryEnrollmentStore` |
| Claim-only scoring | `offline_eval.EmbeddingTrialScorer`, `ClaimRequest`, `write_run` |
| Split guard | `splits.assign_split/validate_split` (DRAFT, development only) |
| Overlays | `pilot.write_overlay` |

## 3. What was added

- `gait_verification/pilot_intake.py`: `intake_template` (16 rows, 2×4×2) and
  `validate_intake`. Explicit paths only; full decode; measured SHA-256, ffprobe and
  frame timestamps; duplicate-byte refusal. Operator/consent/independent-take
  confirmations and session/date/consent/geometry/motion/review metadata are
  mandatory. Protected roles are refused before any media is opened. The output is an
  all-or-nothing `capture-registry-v1`.
- `gait_verification/pilot_protocol.py`: DRAFT two-subject development split;
  deterministic `two-subject-pilot-trials-v1`. Set 2 queries use the independent Set 1
  originals of the claimed subject, excluding the probe view (3 per claim). The Set 2→1
  swap has disjoint trial IDs, and regeneration detects tampering. The actual query
  subject is never part of `ClaimRequest`.
- `gait_verification/stabilization.py`: `AugmentationConfig` (`controlled-ffmpeg-v1`).
  Brightness `eq` in [-0.1, 0.1], contrast `eq` in [0.85, 1.15], `gblur` sigma in
  [0.1, 1.0] and libx264 CRF in [23, 32]. Variant IDs are derived from the exact
  parameters, with single-threaded encoding, passthrough timestamps and the existing
  frame-timing/dimension validation. Flips, speed, reversal, crops, warps and
  composition are not expressible.
- `gait_verification/dataset_processing.py`: `derive_variants(recipes=…)` and recipe-aware
  manifest verification, including for already-registered derivatives.
- `gait_verification/pilot.py`: `write_overlay(selected_indices=…)` with private 0600 writes.
- `gait_verification/two_subject_pilot.py`: `PilotWorkflow` stages `validate`, `register`,
  `plan`, `augment`, `process`, `enroll`, `trials`, `score`, `previews`, `report`, `all`,
  plus `summary_markdown`.
- `scripts/gait_verification/two_subject_pilot.py`: CLI with the same commands plus
  `init`. Real dependencies load lazily, and the E0 checkpoint hash stays pinned to
  `configs/gait_verification/e0.json`.
- `configs/gait_verification/two_subject_pilot.example.json` and
  `two_subject_pilot_intake.template.json`, both generated by `init` and free of personal data.

## 4. Workflow semantics

| Stage | Behaviour |
|---|---|
| validate/register | Configured asset IDs and subjects must equal the sheet. Real-media configs refuse `SYNTHETIC` metadata. The registry and measured sheet are written once. Later intake edits stop every stage. |
| Every later stage | Reloads the registry, checks that the artifact root contains no source, and rechecks every original's SHA-256 before access. |
| plan | Read-only. A deterministic `plan_id` binds the registry, the Phase 2 processing plan, variants/recipes, models and directions. Before registration it validates in memory. |
| augment | Configured S10/S15 + opt-in recipes for all 16 originals. The report covers RAW preserved/rechecked, every derivative's commands, FFmpeg version, hashes, frames, timing, storage and inherited subject/session/view/role. |
| process | Originals: pose, quality, embeddings for each pretrained model, plus native E0 features. Derivatives: pose and quality only. Derivatives are verified but never created here. |
| enroll | Per direction: the four reference-set RAW originals per subject and model go into `enrollments/<direction>/`. Pose comes only from sealed processing caches, and stored references are skipped without loading a model. |
| trials/score | Manifests are persisted and must regenerate identically. Each model scores all 16 claims through the existing scorer. Results are checked against the public contract: an OK status or a non-null verdict is refused, and scores are allowed only with `POLICY_NOT_READY` plus quality-evaluable input. |
| previews | Explicit assets only. Start/middle/end frames plus up to N tracking-flagged frames, written private (0600) and never regenerated over existing files. |
| report | `report.json` + `summary.md` in a new private directory. Covers intake, augmentation, per-asset quality with `recommend_manual_review` and `delta_from_raw` (IMPROVED/WORSENED/UNCHANGED for seven measured pose quantities), enrollment, per-model claim counts by status/reason × claim class/view/set/subject, the separately labelled set swap and the matched-variant restriction. It contains no ROC/EER/FAR/TAR/threshold. |

Raw scores exist only in `work/pilot/runs/<direction>/<model>-<id>/outcomes.jsonl`
(0600, hash-bound to its report).

## 5. Tests and commands

```bash
.venv-gait-v2-cuda/bin/python -m pytest -q tests/gait_verification tests/test_future_work.py \
  -k "not test_harness_end_to_end_on_synthetic_subjects and not test_metrics_against_known_values"
GAIT_V2_CUDA_TESTS=1 .venv-gait-v2-cuda/bin/python -m pytest -q tests/gait_verification tests/test_future_work.py \
  -k "not test_harness_end_to_end_on_synthetic_subjects and not test_metrics_against_known_values"
.venv-gait-v2-cuda/bin/ruff check gait_verification scripts/gait_verification tests/gait_verification
.venv-gait-v2-cuda/bin/ruff format --check <11 new/changed Python files>
.venv-gait-v2-cuda/bin/python -m compileall -q gait_verification scripts/gait_verification tests/gait_verification
git diff --check
```

| Execution | Result |
|---|---|
| HEAD baseline (isolated `git archive`), default | 403 passed, 11 skipped, 2 deselected, 1 failed. The failure is `test_biometric_and_runtime_paths_are_git_ignored_and_untracked`, an archive artifact: it calls `git check-ignore` outside a repository. That matches the accepted 404/11. |
| HEAD baseline, CUDA | 414 passed, 2 deselected, 1 failed (the same archive-only case), matching the accepted 415 |
| New pilot cases | 41 collected: 40 default + 1 opt-in real-CUDA case |
| Final default suite | **444 passed, 12 skipped, 2 deselected, 0 failed** (162.9 s) = 404 + 40 new; skips = 11 existing CUDA + 1 new opt-in |
| Final CUDA suite | **456 passed, 0 skipped, 2 deselected, 0 failed** (637.5 s) = 415 + 41 new |
| Opt-in real CUDA pilot (`test_cli_all_with_real_frozen_cuda_models`) | 1 passed, 405 s: CLI `all` with real GaitPT/GPGait/GPGait++ GREW/CASIA-B and the real E0 head; 16/16 restricted scores per model, all `POLICY_NOT_READY`; a `process` rerun reused 16/16 artifacts per model |
| Ruff lint / format (11 files) / compileall / `git diff --check` / untracked whitespace scan | Passed |
| Preservation | Shipped checkpoint `6c37f3b9…` and E0 head `f65c44cd…` hashes unchanged. No file under `data/`, `outputs/`, checkpoint/source caches, `models/` or `utils/` modified during the session. No diff to historical code, E0 tests or phase reports. |

Test coverage: 2×4×2 matrix, missing rows/files, invalid metadata, missing consent,
supplied-hash mismatch, duplicate bytes, same-session versus explicitly distinct
sessions, protected-role refusal before media access. Five recipes with provenance,
timing, inheritance, reuse and corruption detection; unsafe recipe and relabel
refusal. Set 1/Set 2 claims, cross-view references, swap separation, claim-only
requests, scientific/final split refusal, sheet-not-filename set labels, contract
tampering. A complete 16-original workflow: 48 derivatives, 64 quality rows, five
models, 16 claims each, `POLICY_NOT_READY`, four references per subject and model,
private overlays, unchanged originals, resume without rewriting NPZs, and the set
swap. Waiting-for-inputs without registration; work-dir aliasing and changed-original
refusal. Failed-recording isolation with exact count reconciliation; CLI
init/validate/plan; no CPU fallback; the real CUDA path.

Failures during development: a no-metrics assertion matched vid.stab's `accuracy=15`
parameter (fixed to check report keys, excluding augmentation provenance); a
no-CUDA assertion missed the correct downstream `POSE_ARTIFACT_NOT_READY` block;
ruff F811 on shared fixtures (repo pattern adopted). No product defect was hidden.

## 6. Real pilot status and exact missing inputs

`data/gait_verification/pilot_2subjects/` does not exist. Missing: all 16 originals
`originals/{S001,S002}/{left45,right45,toward,away}/{set1,set2}.mp4`, the filled
`intake.json`, and for every row: real session ID, timezone datetime, consent reference,
angle/direction/motion description, capture review and the three confirmations.
Commands to run after copying them are in the
[operator guide](TWO_SUBJECT_PILOT_OPERATOR_GUIDE.md) §3–5.

## 7. Engineering limitations

- **Matched-variant scoring is restricted.** Derivatives cannot be enrolled or scored
  against RAW references. Their robustness evidence is pose/quality deltas only.
- E0 and the pretrained models use unequal native quality gates (MediaPipe minimum
  versus RTMPose temporal windows), as reported in Phase 2C. In the failed-recording
  test, E0 scores claims that the RTMPose path abstains on. That is expected and
  disclosed, not reconciled.
- Enrollment re-encodes reference originals once per model, reusing cached pose. It
  does not reuse processing's query embedding vectors, because templates need encoder
  provenance.
- `process`, `previews` and `report` verify derivatives through `derive_variants`. A
  derivative directory that fails verification is renamed aside (kept, never deleted),
  matching Phase 2 quarantine behaviour.
- Scoring rechecks source bytes per trial (the Phase 2C rule). Real 30-second throughput,
  64-video pose runtime and peak worker memory are unmeasured.
- Single writer per work directory and one active GPU workflow remain operating assumptions.
- Overlays are full frames and can show faces. They are private files only, with no
  redaction.

## 8. Scientific limitations

Two participants support no accuracy, FAR/FRR/EER/TAR, ROC, threshold or model,
stabilization or augmentation choice. Both swap directions reuse the same 16 takes and
are not independent estimates. Same-occasion sets are cross-view/different-take
evidence, not cross-day evidence. Pilot participants stay development-only and must
never join a locked-final group. Quality thresholds are provisional. EXT/OWN/SCI/SEC
items, including PIL-07 (the real run), remain OPEN.

## 9. Changed files

New: `gait_verification/{pilot_intake,pilot_protocol,two_subject_pilot}.py`,
`scripts/gait_verification/two_subject_pilot.py`, `tests/gait_verification/test_{pilot_intake,
pilot_protocol,pilot_augmentation,two_subject_pilot}.py`, the two `configs/gait_verification/two_subject_pilot*`
files, and `docs/gait_verification/TWO_SUBJECT_PILOT_{TODO,OPERATOR_GUIDE,IMPLEMENTATION_REPORT}.md`.
Modified: `gait_verification/{stabilization,dataset_processing,pilot}.py` (additive), the
PRD (§25), the handoff (dated amendment) and the deferred register (PIL-01…PIL-07).
All changes are left unstaged for owner review.
