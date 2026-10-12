# Gait Verification V2 — Phase 2 dataset-integration report

**Date:** 12 October 2026. **Branch:** `feat/gait-verification-v2`.
**HEAD:** `433cdd0e4bb3c075ab39731254e1e108fbbfe096` (Phase 1D).
**Authors:** Claude Code (initial implementation); Codex (independent continuation and fixes).

**PHASE 2 ENGINEERING: COMPLETE.** Scientific and deployment readiness remain **BLOCKED**. This report contains engineering fixture evidence only.

## 1. Executive summary

Claude had implemented the Phase 2 registry-driven ingestion report, deterministic planner,
resumable processing runner, failure ledger, cache validation, CUDA slot controls, coverage
reporting, optional stabilization, decision guards, 60 test cases and the operator guide.
Codex recovered that exact uncommitted worktree, inspected every changed/new source and test
file, reproduced Phase 1D independently, and added 36 focused regression cases. The existing
implementation was extended surgically; it was not rebuilt.

Processing is an explicit-asset workflow, separate from enrollment and evaluation. It can
receive future recordings without choosing a split, scientific policy, model or threshold.
The owner intentionally deferred recordings until the software architecture is complete.
The planned ~40-person, four-view, ~30-second cohort and consenting-volunteer pilot do not exist.

## 2. Owner decision and scope

The existing Phase 2 amendments in PRD §21 and the companion handoff authorize architecture
work using clearly labeled generated/synthetic fixtures before physical collection. They
supersede the earlier sequence requiring collection before dataset-ingestion engineering;
they do not waive the scientific pilot gate or authorize any protocol decision.

Only **1:1 claimed-identity verification** exists. Lookup and scoring use only the claimed
person's references. Usable enrolled claims require an authorized validated policy before
TRUE/FALSE; without one they remain `POLICY_NOT_READY` with null verdict. Bad input, absent
enrollment, insufficient gait and runtime failures have explicit non-Boolean statuses.
No identity search, other-person prediction, deepfake assertion, neural training, calibrator
fitting, threshold installation, model winner or locked-final evaluation occurred.

## 3. Phase 1D baseline verification

A read-only `git archive 433cdd0` was extracted into an isolated temporary directory. Its
external model/source and historical pose fixtures were shared by symlink for read-only
inference. The current feature worktree was never checked out, reset or replaced.

| Permitted suite on the committed baseline | Passed | Failed | Skipped | Deselected |
|---|---:|---:|---:|---:|
| Default | 212 | 0 | 4 | 2 |
| `GAIT_V2_CUDA_TESTS=1` | 216 | 0 | 0 | 2 |

Both reproduce the handoff. The recovered Phase 2 worktree, before Codex edits, independently
returned **267 passed, 0 failed, 9 skipped, 2 deselected** (one more passing case than the
older reported 266). Exact command and final results are in §19.

## 4. Exact repository state and preservation

At recovery: branch as above; HEAD `433cdd0`; nine modified tracked files and six untracked
Phase 2 files. Nothing was staged, committed, pushed, merged, reset, cleaned or deleted.
The initial changed files and complete diff were also preserved in a temporary recovery
archive. `AGENTS.md` and both main documents were read in full, along with the Phase 1D
report, TODO register and operator guide. Repository-root `CLAUDE.md` is absent and Git-ignored;
no substitute from another project was used.

Historical E0 paths have no diff from HEAD: `models/`, `utils/`,
`scripts/{training,inference,evaluation}/`, `tests/test_future_work.py` and
`outputs/checkpoints/`. The V2 package does not import E0's `models` or `utils` packages.
Private V2 recordings, processing artifacts, sources and checkpoints remain ignored and
untracked. Existing tracked historical E0 checkpoints are intentionally preserved.

## 5. Existing architecture reused

| Component | Phase 2 use |
|---|---|
| `CaptureRegistry` | Authoritative identity/session/view/consent-reference and lineage metadata |
| Media probe and SHA-256 helpers | Real frame timing, media facts and immutable-source checks |
| `NativePoseExtractor`, pose NPZ functions | Native COCO-17 tracks with diagnostics; safe `allow_pickle=False` reads |
| `CachedPoseExtractor` | Immutable NPZ artifacts plus checksum/recipe sidecars; legacy compatibility |
| Phase 1A quality policy and `WindowedEncoder` | Provisional temporal quality and unchanged native model segments |
| Registry encoder/runtime loaders | Existing four frozen CUDA candidates and pinned RTMPose |
| `QueryEmbeddingStore`, `embed_query` | Query-only artifacts, including explicit abstentions |
| `ClaimVerifier`, enrollment store, offline scorer | Existing claim-only downstream scoring |
| `write_new_json`, stabilization runner | Exclusive private JSON writes and existing FFmpeg recipe |
| Split access guard | Refusal of protected locked-final processing; no CLI authorization |

## 6. Actual files changed

New Phase 2 files authored initially by Claude:

- `gait_verification/{dataset_ingest,dataset_processing,decisions}.py`
- `tests/gait_verification/test_phase2.py`
- `docs/gait_verification/{DEFERRED_EVIDENCE_AND_DECISIONS,PHASE_2_OPERATOR_GUIDE}.md`

Tracked files modified initially by Claude and retained:

- `configs/gait_verification/phase1d_evaluation.json`
- `gait_verification/{capture_registry,offline_eval,pipeline,pose_quality}.py`
- `scripts/gait_verification/evaluate_claims.py`
- `tests/gait_verification/test_phase1d.py`
- Both main gait documents.

Codex additionally created this report and
`tests/gait_verification/test_phase2_{integrity,recovery}_regressions.py`, corrected the
processing/integrity/CLI defects described below, reconciled the register and guide, and
added implementation amendments to both main documents. Ruff formatting was limited to
new and intentionally changed Python files; untouched historical files were not reformatted.

## 7. Ingestion architecture

`dataset_ingest.ingest_report` consumes a validated capture registry and explicit data root.
It never scans a directory or infers identity/consent from filenames. The registry refuses
duplicate IDs, duplicate bytes, conflicting originals, invalid SHA-256, missing required
identity/session/date/consent metadata, and broken derivative relationships.

Program-verified media facts, operator-supplied metadata, provisional pose quality and
scientific capture suitability remain four separate layers. `--measure` uses real ffprobe
facts: dimensions, orientation, FPS, frame counts, source PTS, duration and frame intervals.
Declared-versus-measured differences are reported using explicitly provisional tolerances.
Optional missing metadata is listed rather than guessed. Consent references are never proof
of consent. Scientific suitability is always `NOT_ASSESSED`.

## 8. Explicit asset selection

`--asset` is repeatable; `--assets-file` supplies IDs. Selection is sorted and deduplicated.
There is no implicit “all” or filesystem discovery. Unknown IDs and invalid sources remain
visible in blocked-input reporting and prevent a successful COMPLETE run. A denied
locked-final asset is listed without reading its bytes or creating processing tasks.

An artifact root containing any registered source is refused before writes, so source
videos cannot become processing output/ledger files. Derivative paths must stay inside the
data root and cannot contain registered source files either.

## 9. Read-only dry-run and plan identity

`build_plan` creates no directories, ledger, enrollment, split, policy or model. Tests
compare bytes, timestamps and directory trees before/after planning. Optional `--out`
explicitly publishes a new plan file; that output does not mutate the source or cache.

IDs bind registry fingerprint, requested assets, their hashes, supplied split fingerprint,
pose backend/extractor settings, quality policy, selected model keys and stages. Reordered
selection yields the same ID. Volatile free-disk measurements and current cache states are
outside the plan-ID content, allowing the reviewed scope to resume as tasks finish.
`--expect-plan-id` rejects a different plan before creating the work directory.

Plans report asset/original/derivative/subject/session counts, task states, relative
embedding-store locations, GPU work counts and actual free space. Pose storage is a labeled
uncompressed array estimate from measured frame counts, not a measured artifact size.
Unmeasured storage/latency/throughput is not invented.

## 10. Processing dependencies

```text
registered asset + verified ancestry
          |
          v
native pose NPZ + checksum/recipe sidecar
          |                         |
          v                         v
quality JSON                per-model query embedding JSON
(provisional diagnostics)   (same quality gate recomputed before inference)
```

Execution orders pose → quality → one embedding model at a time. Each downstream artifact
pins the pose NPZ hash. `--stages` can select a subset; missing pose artifacts block
quality/embedding when pose processing was omitted. A persisted quality JSON is a report,
not a required inference input: embedding recomputes the same configured quality gate.
No processing task enrolls a subject or makes an identity decision.

## 11. Cache fingerprints and integrity trust boundary

Pose keys bind source hash, backend, variant and original take. Sidecars must have the
correct schema, extractor settings, lineage, frame count and valid NPZ SHA-256. Quality
artifacts additionally bind policy and pose key/hash. Embedding directories bind model key
(which contains checkpoint/preprocessing identity), policy, pose backend, extractor settings
and aggregation. Embedding records bind source, original take, variant and pose hash.
Changed recipes produce separate directories or invalidate the existing pose sidecar;
valid old artifacts are preserved.

Quality and Phase 2 query records carry SHA-256 seals over canonical JSON, with nonfinite
values rejected. Missing, invalid and incompatible records are not accepted as cache hits.
Malformed sidecars, missing fields, stale lineage and nonfinite JSON are classified INVALID,
rather than crashing the planner. Invalid files are renamed aside before recomputation.

`QueryEmbeddingStore(require_integrity=True)` is used by processing. The default reader
preserves genuine Phase 1D unsealed records. Any record carrying a seal or Phase 2 processing
fingerprints (`recipe`/`pose_npz_sha256`) must pass the seal, even if its seal key was removed.
The offline scorer converts unreadable/compromised records to
`PROCESSING_ERROR / QUERY_EMBEDDING_INTEGRITY`, with no score or verdict. Legacy compatibility
is explicitly tested. Old unsealed records carry no retroactive integrity guarantee.

These are **checksums, not signatures or authenticated storage**. Someone who can rewrite
both payload and checksum can recompute a seal. Encryption, signatures and access auditing
remain deployment/security work, not a claim of this phase.

## 12. Resumability, failures and corrected defects

The append-only private ledger records per-task state, attempts and timing. Completed
artifacts are revalidated and skipped without loading a model. Missing/invalid/incomplete
work resumes; retryable failures retry; permanent input failures need `--retry-permanent`.
A torn ledger event is preserved, and the next append starts on a separate line so recovery
cannot hide a later complete event.

The continuation corrected these verified defects:

1. Stripped Phase 2 seals could enter the legacy read path and be scored.
2. Malformed cache objects/nonfinite JSON/missing fields could crash planning/reporting.
   Final review also reproduced incomplete scoring-record crashes; required fields, statuses
   and embedded vectors are now validated before scoring, preserving every trial's error status.
3. Re-sealed stale lineage fields were not all checked against selected assets.
4. Detector/encoder load exceptions escaped before a run report; stage resource failures
   now persist as retryable, including a temporarily missing checkpoint.
5. An all-unknown/all-denied selection could falsely report COMPLETE.
6. A source deleted during processing could be omitted from the final integrity check;
   source-change reporting now prevents COMPLETE.
7. CLI `plan --split` failed to pass its split to the planner; plan identity now also binds
   split and requested scope/extractor settings.
8. Derivative generation did not enforce the locked-final guard.
9. A torn ledger line swallowed the next event.
10. Derivative processing verified child bytes without verifying ancestry/transform manifests.
11. A work directory could contain registered sources and risk changing them.

Derivative reuse also validates the declared S10/S15 recipe and output bytes, including
already-registered assets. No unrelated components were refactored.

Decoder-invalid inputs are permanent failures; CUDA absence/busy slots/OOM and model setup
failures are retryable. Failed originals remain in selected-asset/coverage reports. Quality
abstention is a successfully recorded outcome, not an identity mismatch or runtime failure.
An interruption saves completed work and leaves unfinished tasks `NOT_RUN`.

## 13. Atomic artifact behavior

JSON uses private temporary files, fsync and exclusive hard-link publication. Pose NPZ is
written privately, fsynced and published exclusively before its sidecar. The pair is not a
single filesystem transaction: a crash between the two leaves an invalid partial entry
that the next run quarantines and rebuilds. A valid artifact is never overwritten.
Quarantined files and historical experiments are preserved for inspection.

This is process-interruption recovery evidence. Power-loss durability of every parent
directory entry has not been established. Concurrent runs writing the same work directory
are not a fully transactional multi-writer service.

## 14. CUDA resource management

Real runtime: Python 3.11.17, PyTorch 2.10.0+cu128, ONNX Runtime 1.23.2;
NVIDIA RTX 3050 Laptop GPU, 6,144 MiB, driver 595.99.02. Inference is frozen and CUDA-only;
no CPU substitution exists. Checkpoints/source revisions are validated by existing loaders.

GPU stages hold cooperative flock slots (default one) under a shared per-Unix-user lock
root. Within a run all tasks/models are sequential, loaded lazily once per stage and
released afterwards. Tests cover busy slots, absent CUDA, load/inference OOM and resume
without model loading. Processes must use the same configured slot bound/lock root;
this is not a cross-user GPU scheduler or hard VRAM quota.

Per-task seconds and run wall time are measured. Peak CUDA allocation applies only to
in-process encoders; GPGait worker VRAM/host peak RSS are not measured. No dataset-throughput
or future 30-second-video performance estimate is claimed.

## 15. Pose quality and coverage

Coverage preserves every requested recording's pose state, provisional outcome, reasons,
frames/PTS span, accepted/rejected windows, ankle/full-body visibility, missingness,
detection/ambiguity fractions and track breaks. Decoder/runtime failures are separate.
Windows and derivatives retain original ancestry; they are not independent recordings.
Scientific capture suitability remains `NOT_ASSESSED`, including successful CUDA runs.
Metadata/media checks come from `validate`; pose diagnostics come from `report`.

## 16. RAW/S10/S15 handling

RAW is the baseline and is never derived or replaced. `derive` acts only on explicitly
selected originals and versions S10/S15 separately. It uses the existing two-pass FFmpeg
settings (shakiness 5, accuracy 15, mincontrast 0.2; smoothing 10/15; optzoom 0, black crop,
bicubic interpolation). The existing runner validates every relative frame timestamp and
display dimensions, preserves transformation logs and hashes, and checks source bytes.

A new registry records parent/original/checksum/transform ancestry; the old registry stays
unchanged. Registered derivatives inherit subject/session/view/consent/split roles. An
explicit split or declared locked-final role blocks derivation. A stabilized embedding has
`:variant=...` in its model recipe and cannot match RAW enrollment references. OWN-08 is
still unset; no stabilization winner or mixed RAW/stabilized policy was installed.

## 17. CLI usage and tested examples

See [operator guide](PHASE_2_OPERATOR_GUIDE.md) for validation, planning, selective processing,
resume/retry, failure inspection, cache reuse, reports and derivative generation.
All listed flags were checked against actual parser/help and subprocess/integration tests.

Continuation smoke used:

```bash
CL=".venv-gait-v2-cuda/bin/python scripts/gait_verification/evaluate_claims.py"
R=outputs/gait_verification/phase2-smoke/registry.json
D=outputs/gait_verification/phase2-smoke/data
W=outputs/gait_verification/phase2-smoke/work
A=outputs/gait_verification/phase2-smoke/assets.txt
$CL validate --registry "$R" --data-root "$D" --measure \
  --out outputs/gait_verification/phase2-smoke/continuation-validate.json
$CL plan --registry "$R" --data-root "$D" --work-dir "$W" --assets-file "$A" \
  --model gaitpt --measure --out outputs/gait_verification/phase2-smoke/continuation-plan.json
$CL process --registry "$R" --data-root "$D" --work-dir "$W" --assets-file "$A" \
  --model gaitpt --expect-plan-id e80e362697a7a6fee5e7536997bca552
$CL report --registry "$R" --data-root "$D" --work-dir "$W" --assets-file "$A" \
  --model gaitpt --out outputs/gait_verification/phase2-smoke/continuation-coverage.json
```

All exited 0. Validation matched three originals, two synthetic subjects and two sessions;
planning found nine VALID tasks; processing returned `SKIPPED_VALID: 9`, with all three
source hashes unchanged. Resume run: `run-20261012T140325-f11bf715`. Outputs are new,
Git-ignored files. Commands publishing a named file are exclusive; choose a new output
filename when repeating them. Plan IDs change when bound configuration changes.

## 18. Real CUDA smoke evidence

Existing evidence under `outputs/gait_verification/phase2-smoke/` was inspected and retained:

- `run-20261012T134135-4d4f459d`: 9 DONE tasks; real RTMPose CUDA and GaitPT loaded;
  recorded wall time 13.185 seconds on three generated testsrc2 clips.
- `run-20261012T134148-807e4940`: 9 SKIPPED_VALID tasks, no model load.
- All three persisted outcomes were `INSUFFICIENT_GAIT`, with no embeddings or decisions.
  GaitPT stage execution on these clips is not evidence of a successful gait forward pass.
- Their sealed artifacts remain valid under the continuation checks.

The final opt-in pytest run separately tests real RTMPose on a generated clip and actual
frozen CUDA forward passes for **all four** encoders using synthetic, anatomically ordered
pose fixtures through the processing runner: GaitPT 256-D; GPGait, GPGait++ GREW and
GPGait++ CASIA-B 4,864-D. These are model/runtime engineering checks, not genuine recordings,
consent evidence, accuracy, or model ranking. CASIA-B remains a separate ablation.

## 19. Unit, integration and regression results

The two prohibited legacy training/calibration-fitting tests remained deselected:
`test_harness_end_to_end_on_synthetic_subjects` and `test_metrics_against_known_values`.

```bash
.venv-gait-v2-cuda/bin/python -m pytest -q \
  tests/gait_verification tests/test_future_work.py \
  -k "not test_harness_end_to_end_on_synthetic_subjects and not test_metrics_against_known_values"
GAIT_V2_CUDA_TESTS=1 .venv-gait-v2-cuda/bin/python -m pytest -q \
  tests/gait_verification tests/test_future_work.py \
  -k "not test_harness_end_to_end_on_synthetic_subjects and not test_metrics_against_known_values"
```

| Continuation check | Result |
|---|---|
| Full permitted default suite | 303 passed, 0 failed, 9 skipped, 2 deselected |
| Full permitted CUDA suite | 312 passed, 0 failed, 0 skipped, 2 deselected |
| Added regression cases | 36 total; corruption/lineage/recovery controls plus reproductions |
| Ruff lint (`gait_verification`, `scripts/gait_verification`, `tests/gait_verification`) | Passed |
| Ruff format check (all 12 new/intentionally changed Python files) | Passed |
| `compileall` on those package/script/test directories | Passed |
| `git diff --check` | Passed |
| Historical E0 diff and private V2 ignore/untracked checks | Passed |

During RED verification 32 new cases exposed failures; four new compatibility/control cases
already passed. One intermediate implementation regression changed interruption polling;
it was corrected and the original interruption/recovery test passed again. No tests were
weakened or disabled. PyTorch emits the existing nested-tensor optimization warning; it
is not an inference/test failure. The permitted suites are also rerun after documentation reconciliation before the final handoff.

## 20. Engineering limitations and deployment blockers

- The managed registry cannot identify the source of an unregistered re-encode; it refuses
  it instead. A future arbitrary-upload service needs a separate provenance design.
- Unsealed Phase 1D compatibility is an explicit lower-trust path. Seals/sidecars detect
  corruption but do not authenticate edits by someone with write access.
- Work artifacts/enrollments are protected by filesystem permissions, not encryption or
  a signed audit log. Derivative data-root access/permissions remain the operator's duty.
- GPU locks are cooperative per user/configuration; no hard host/worker-memory quota.
- Concurrent same-work-directory writers and abrupt machine power loss are not a fully
  transactional storage guarantee. Use one active writer per work directory.
- Provisional quality is not a validated walking/liveness or biometric suitability test.
  No representative 30-second moving-camera throughput/quality study exists.
- A downstream enrollment/evaluation job still requires its own approved split/trials and
  references. Processing never supplies scientific choices or enrolls automatically.
- The real E0 comparative scorer adapter remains future engineering; empirical E0 parity
  and any new-cohort retraining remain unvalidated/unauthorized.
- Official source-dataset reproduction, Android-domain validation and license clearance
  remain outstanding. No inference checkpoint is missing in this environment, but **all
  four model entries remain BLOCKED**, `production_ready=false`, with null thresholds.

## 21. Deferred external/scientific requirements

[Deferred register](DEFERRED_EVIDENCE_AND_DECISIONS.md) remains authoritative. Physical pilot,
consent/metadata, four real ~30-second views, ~40-person cohort, capture-SOP validation,
actual recording days, split counts/seeds, impostor sampling, target FAR, E0/adaptation
permission, quality/stabilization/model selection, threshold calibration, locked-final
study, licensing, retention, encryption/access auditing and deployment approval stay OPEN.
Engineering completion satisfies none of these. A single-day cohort can support
cross-view/different-take evidence, not cross-day reliability.

## 22. Phase 2B architecture readiness

Phase 2's software interfaces are suitable for further architecture work without collecting
recordings. “Phase 2B” has no separate detailed scope in the current PRD; it must not be
interpreted as authorization to train, calibrate or open locked-final data.

A useful next engineering slice is the additive **frozen E0 scorer adapter and common-run
orchestration**: consume the existing `ClaimRequest`, use only its query and claimed-person
reference originals, preserve E0's 78-channel preprocessing, and emit explicit score/status
outcomes on the same manifest as the embedding scorers. Synthetic/engineering fixtures can
verify lineage, failures and equal trial IDs before real data. No retraining is implied.

## 23. Exact recommended next task

Implement and test that E0 adapter behind the V2 scorer interface, without modifying the
historical E0 implementation/checkpoints, fitting a calibrator or installing a threshold.
Keep scientific comparison blocked and use the existing synthetic trial framework for
contract tests. Finish the remaining agreed architecture before requesting collection;
the owner explicitly deferred physical recordings. The capture pilot remains a later
scientific dependency, not the next required action in this continuation.
