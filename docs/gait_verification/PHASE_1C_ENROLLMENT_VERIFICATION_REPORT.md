# Gait Verification V2 — Phase 1C enrollment and claimed-identity verification report

**Date:** 12 October 2026. **Branch:** `feat/gait-verification-v2`. **HEAD at start:** `704bf09`
(Phase 0, 1A and 1B are committed; the tree was clean). Phase 1C changes are **uncommitted**.
Author: Claude Code.

**PHASE 1C ENGINEERING: COMPLETE.** Scientific validation: **not started and not claimed.**

## 1. Executive summary

A `(video, claimed_subject_id)` request now runs the full real pipeline on CUDA:
decode with source PTS → RTMPose → immutable native pose cache → Phase 1A window
quality → frozen encoder → window-aggregated walk embedding → lookup of **only the
claimed subject's** references → recipe and lineage checks → cosine score → policy gate.
Enrollment stores one reference per independent original, per model recipe.

Because no scientifically derived threshold exists, every real request for a usable,
enrolled claim returns `POLICY_NOT_READY`, `verdict=null`. The raw score is computed and
available only through a restricted diagnostics file. No threshold, calibration,
training, identification or model selection was done. Passing tests and finite scores
are **engineering** evidence only; nothing here measures recognition accuracy.

## 2. Phase 1B handoff verification

Baseline reproduced before any edit: **95 passed, 2 deselected**, as reported. Registry
entries for all four candidates remain `BLOCKED` / `production_ready=false` with null
thresholds. Phase 1B encoders were reused unchanged (`backends/` was not modified).
Real CUDA re-verification of all four candidates is in §12.

## 3. Git branch and working-tree status

Branch `feat/gait-verification-v2`, start tree clean at `704bf09`. Nothing was
reverted, reset, cleaned, committed, pushed or merged. E0 paths (`models/`, `utils/`,
`scripts/training`, `scripts/inference`, `tests/test_future_work.py`) have **no diff**.
Real enrollments, embeddings, pose caches and diagnostics live under git-ignored paths
(`data/gait_verification/`, `outputs/gait_verification/`, `.cache/`).

## 4. Existing components reused

`contracts.py` (Recording, GaitEmbedding, VerificationResult, Status), `media.py`,
`pose.py` (RTMPose CUDA, `NativePoseExtractor`, `save_pose/load_pose`),
`pose_quality.py` (`analyze_pose`, `assess_window_quality`, `WindowedEncoder`),
`skeleton_contracts.py`, `stabilization.py` manifests, all four Phase 1B encoders,
`models.json`, `phase1a.json`, `DirectoryEnrollmentStore` (extended, same class and
hashed-claim file layout), the existing `verify_claim.py` entry point, and the existing
claim-only `ClaimVerifier`.

## 5. Gaps found and fixes implemented

| # | Gap in the audited code | Fix |
|---|---|---|
| 1 | `verify()` returned before encoding for any `BLOCKED` encoder, and the policy check ran before scoring, so no real score was ever produced; the CLI never processed video | Score first, policy last; restricted diagnostics; real CLI |
| 2 | Store was create-once (`O_EXCL`), non-atomic, one recipe per claim, no revocation, no corrupt-file distinction, no cross-subject check | Atomic append store with lock, per-recipe groups, revocation, `CorruptEnrollment`, hash-only source index |
| 3 | Templates did not require raw originals, matching recording/embedding hashes, or unique source bytes | Added to `validate_templates` |
| 4 | Recipe mismatch reported `POLICY_NOT_READY`; broad `ValueError → INVALID_INPUT` mislabelled model errors; corrupt media (`CalledProcessError`) became `PROCESSING_ERROR` | `InvalidMedia` class, explicit precedence (§11), mismatch → `CLAIM_NOT_ENROLLED` |
| 5 | Leakage check covered only the embedding's take/hash | Checks recording take/hash/asset/parent plus declared derivative lineage |
| 6 | No enrollment CLI; no cached-pose/window-quality query pipeline | `pipeline.py`, `registry.py`, `enroll_subject.py`, extended `verify_claim.py` |

Alternative scoring strategies (mean / max / top-2 mean, all within the claimed subject)
were small enough to add as **development-only**. View-aware prototypes were not
implemented (no view-labelled data exists). No alternative is selected.

## 6. Enrollment workflow

```text
check  → pose + window quality only: eligible original?
add    → validate metadata → extract once → quality gate → encode per --models
         → one reference per recipe → atomic append
show   → one subject's non-biometric summary; revoke → exclude a reference, keep record
```

Run `add` once per independent original take. Refusals (before anything is written):
invalid subject/take ID, missing session or consent reference, unparseable
`--recorded-at`, ambiguous tracks, no usable gait segment. Store refusals: re-adding a
take for a recipe, derivative/non-raw recordings, conflicting or aliased source hashes,
cross-subject source reuse, zero/non-finite embeddings, invalid enrollment IDs.
Nothing is repaired silently; metadata is never defaulted except `--split-role`
(`development`).

## 7. Enrollment data contracts

Existing `EnrollmentTemplate`/`Recording`/`GaitEmbedding` are reused with two additive
fields: `enrollment_id` and a JSON `provenance` dict. Schema `enrollment-v2`
(`enrollment-v1` files still load). Per reference:

- Identity/lineage: subject, `enrollment_id`, `original_take_id`, session, view,
  `recorded_at`, `source_sha256`, `split_role`, raw variant, no parent.
- Recipe: `embedding.model_id` already binds architecture, checkpoint SHA, preprocessing,
  config fingerprint, pose recipe + variant, and window-policy hash.
- Provenance dict: `pose_backend_id` (RTMPose detector+pose SHA), `gait_model` (full
  `ModelMetadata`: revision, checkpoint SHA, dataset, licence, readiness), `window_policy`,
  `window_aggregation`, quality summary, `consent_ref`, independence note.
- Normalized final walk embedding. Quality weight is 1.0 (equal weights).

Capture-manifest fields not carried by `Recording` (camera, distance, operator, …) are not
duplicated here; they remain in the capture manifest (`capture_schema.json`).

## 8. Reference template storage

`DirectoryEnrollmentStore`: one JSON file per SHA-256-hashed claim, mode 0600 in a 0700
directory; writes use a temp file + `fsync` + `os.replace` under an `flock`; keys are
sorted so identical input is byte-identical. Individual normalized vectors are kept; the
prototype is derived on demand. A corrupt, wrong-claim or unsupported-schema file raises
`CorruptEnrollment` and is **never overwritten**. `revoke_reference` records a reason and
excludes the reference; replacement requires a new `enrollment_id`. `enroll()` still
refuses to touch an existing claim (`FileExistsError`); `add_reference()` appends.
`source-index.json` holds only SHA-256 → (hashed subject, take); it is used at
enrollment, never at verification. **Not encrypted** (§20).

## 9. Claim-only lookup enforcement

`ClaimVerifier` calls `store.get_claim(claim)` exactly once; the store API has no
enumeration. Tests use a spy store that fails on any other identity, assert the one call,
and confirm the public JSON and diagnostics never contain another subject. The real CLI
smoke ran with other subjects present in none of the lookups.

## 10. Cosine / prototype scoring

`scoring.py`: unit-normalize each reference and the query → baseline
`quality_weighted_centroid` (normalized weighted mean, then dot product, clipped to
[-1, 1]). Invalid dimension, non-finite or zero vectors, or a cancelling (zero-norm)
prototype raise → `PROCESSING_ERROR`. Only the baseline can back a policy
(`VerificationPolicy.ready()` requires it). The score is **not** a probability and is
only exposed in restricted diagnostics.

## 11. Full video-to-claim execution and status precedence

Implemented in `pipeline.py` (`CachedPoseExtractor`, `query_verifier`) with the same
extractor, quality policy and `WindowedEncoder` for enrollment and query. Precedence
(first match wins; every non-OK status has `verdict=null`):

1. `INVALID_INPUT` — claim syntax; missing/unreadable/corrupt video; probe shares an original with a reference.
2. `PROCESSING_ERROR` — store unreadable/corrupt.
3. `CLAIM_NOT_ENROLLED` — no active reference, **or none with the probe's recipe**. An unenrolled claim returns before any extraction.
4. `INSUFFICIENT_GAIT` — extractor/quality/encoder abstains.
5. `PROCESSING_ERROR` — any other runtime/model/scoring failure (an encoder `ValueError` is not "invalid input").
6. `POLICY_NOT_READY` — scoring succeeded, no authorized policy.
7. `OK` — only with an authorized policy.

Tested as a 16-case table. **Owner decision needed/recorded:** incompatible-recipe
enrollments map to `CLAIM_NOT_ENROLLED` (no valid enrollment for this model), since the
existing contract has no dedicated status; the reason code `RECIPE_MISMATCH` is in
diagnostics. This replaced the Phase 0 behaviour of `POLICY_NOT_READY` (two old tests
were updated for this and for the new store/video-existence rules; nothing else changed).

## 12. CUDA execution evidence

Environment: torch 2.10.0+cu128, RTX 3050 6GB, onnxruntime CUDA RTMPose.

- Opt-in test (`GAIT_V2_CUDA_TESTS=1`): for each of GaitPT (256-D), GPGait, GPGait++ GREW,
  GPGait++ CASIA-B (4,864-D) the historical RTMPose cache is encoded through the windowed
  recipe on CUDA, scored against a **synthetic** reference, and returns `POLICY_NOT_READY`
  with a finite diagnostic score: 4 passed.
- Real CLI smoke (scratch store under `outputs/gait_verification/phase1c-smoke/`,
  consent marker `ENGINEERING_SMOKE_ONLY_NOT_STUDY_CONSENT`, pseudonym `SMOKE_A`; **not**
  a study enrollment): two historical originals (`Arhaan_S1`, `Arhaan_S2`) enrolled under
  all four recipes (8 references, 4 per take). Queries on `Arhaan_S3` (all four models) and
  `Ananya_S1` (GaitPT) produced `POLICY_NOT_READY` with finite restricted scores and
  `reference_count=2`; the enrolled `Arhaan_S1` itself → `INVALID_INPUT`
  (`SHARED_ORIGINAL`); a vid.stab S10 derivative with its manifest → `INVALID_INPUT`;
  unenrolled claim → `CLAIM_NOT_ENROLLED`; missing file / `../x` claim → `INVALID_INPUT`;
  `Aarav_S2` (bystanders) → `INSUFFICIENT_GAIT` for query and refused for enrollment.
  Score values are deliberately not reported: they have no validated meaning.

## 13. Model / preprocessing compatibility

Compatibility is exact equality of the full `model_id` (architecture, checkpoint SHA,
preprocessing recipe, config fingerprint, pose backend, variant, window recipe).
Tested rejections with equal dimensions: GaitPT vs GPGait, GPGait vs GPGait++, changed
checkpoint, changed preprocessing, changed pose/RAW-vs-stabilized pipeline, changed window
recipe, native vs windowed. One claim may hold several recipes; the verifier uses only
the probe's recipe group and never pools or fuses models.

## 14. Original-take leakage prevention

Enrollment refuses non-raw/derivative recordings and duplicate take, hash or recipe.
Verification refuses a probe whose take ID, source SHA, or **declared lineage**
(`--query-take-id`, `--derivative-manifest`) overlaps any reference's take, hash, asset or
parent IDs. Windows of one original share its take ID, so they never count as separate
references. **Limitation:** a re-encoded or stabilized copy of an enrolled original that
is submitted *without* declared lineage has different bytes and cannot be detected from
bytes alone (demonstrated in the smoke run). Phase 1D must resolve query lineage from the
capture manifest. Recording session, view, time and day are stored per reference, so a
single-session enrollment is visible and cannot support a cross-day claim.

## 15. Public response behaviour

`public_dict()` still returns only `schema_version`, `claimed_subject_id`, `status`,
`verdict`. No other subjects, rankings, scores, embeddings, paths, probabilities. An
invalid claim echoes `""`. Exception text never reaches the response (tested with
private-path exceptions). Restricted diagnostics (`verify_detailed`, CLI
`--diagnostics-out`, new 0600 file) hold reason, score, strategy, reference count.

## 16. Fail-closed missing-threshold handling

No policy is constructed anywhere in the CLIs or configs (tests assert no
`VerificationPolicy`/`allow_test_policy`/`test_only=` in scripts, all `threshold` fields
null, all registry entries `BLOCKED`). `TEST_ONLY` policies decide only when a test passes
`allow_test_policy=True`; a calibration-looking policy still cannot decide for a
`BLOCKED` encoder. Boolean logic (true, false, boundary `>=`) is covered by isolated
tests only.

## 17. CLI commands and usage

```bash
PY=.venv-gait-v2-cuda/bin/python; E=data/gait_verification/enrollments   # git-ignored
$PY scripts/gait_verification/enroll_subject.py --enrollments $E check \
    --video V.mp4 --take-id TAKE --pose-cache-dir CACHE
$PY scripts/gait_verification/enroll_subject.py --enrollments $E add \
    --video V.mp4 --take-id TAKE --pose-cache-dir CACHE --subject-id S001 \
    --models gaitpt gpgait gpgaitpp gpgaitpp_casia --session-id SESSION --view left45 \
    --recorded-at 2026-10-20T10:15:00+05:30 --consent-ref CONSENT_RECORD_ID
$PY scripts/gait_verification/enroll_subject.py --enrollments $E show --subject-id S001
$PY scripts/gait_verification/verify_claim.py --enrollments $E --video Q.mp4 \
    --claimed-id S001 --model gaitpt --pose-cache-dir CACHE \
    [--query-take-id ID] [--derivative-manifest manifest.json] \
    [--diagnostics-out NEW_RESTRICTED.json] [--scoring-strategy mean_cosine]  # dev only
```

Exit status 0 only for a Boolean verdict (impossible from this CLI). Defaults:
`~/.cache/rtmlib/hub/checkpoints` (RTMPose), `.cache/gait_verification/{sources,checkpoints}`.
Isolated test-only Boolean cases: `pytest tests/gait_verification/test_phase1c.py -k
"test_only or true_false"`. Real CUDA tests: `GAIT_V2_CUDA_TESTS=1 pytest ... -k real_cuda`.

## 18. Test results

Required command (inspected test bodies first; the two legacy training/calibration tests
stay deselected):

| Run | Result |
|---|---|
| Baseline before edits | 95 passed, 2 deselected |
| Final, default | **148 passed, 4 skipped (opt-in CUDA), 0 failed, 2 deselected** |
| Final, `GAIT_V2_CUDA_TESTS=1` | **152 passed, 0 skipped, 0 failed, 2 deselected** |
| `test_phase1c.py` | 57 tests: unit/mocked, ffmpeg media integration, 4 opt-in CUDA |
| Ruff check | passed (all V2 paths) |
| Ruff format | all new/changed files clean; `pose_quality.py` and `test_gpgait.py` were already unformatted at HEAD and left untouched |
| `compileall`, `git diff --check`, trailing-whitespace scan of new files | passed |
| E0 diff | none |

Evidence types are separate: mocked unit tests; media integration (real ffmpeg decode,
real native extractor, cache and window analysis with an **injected fake detector and
stand-in encoder**); real CUDA (§12); scientific evaluation — **not performed**.

## 19. Changed files

Modified: `gait_verification/{enrollments,verifier,media}.py`,
`scripts/gait_verification/verify_claim.py`,
`tests/gait_verification/{test_foundation,test_phase1b,test_pilot}.py` (three tests
adjusted for the deliberate contract changes in §11), PRD and handoff amendments.
New: `gait_verification/{scoring,pipeline,registry}.py`,
`scripts/gait_verification/enroll_subject.py`,
`tests/gait_verification/test_phase1c.py`, this report.
Ignored evidence: `outputs/gait_verification/phase1c-smoke/`.

## 20. Security / privacy limitations

- The store is protected by file permissions only; **no encryption, access auditing or
  key management**. Embeddings are biometric templates; keep the directory private.
- `flock` locking is POSIX/single-host. A crash can leave an orphan source-index entry
  (harmless, retry-safe) but never a partial claim file.
- `add` with several models writes sequentially; a failure mid-way can leave some
  recipes stored (re-run after revoking or fixing).
- Restricted diagnostics contain raw scores; the CLI writes them only to an explicit new file.
- Subject-to-name and consent mappings must stay outside the repository.

## 21. Scientific limitations

No biometric accuracy, threshold, calibration, FAR/FRR/EER or model winner exists. The
40-person cohort and the four-view moving-camera pilot are not collected. Single-clip
and 2–3 historical clips only demonstrate plumbing; same-participant historical takes are
not a verified identity benchmark. Window-level mean aggregation, equal quality weights
and the provisional Phase 1A quality constants remain untested engineering choices.
Quality gating can silently cost coverage; abstentions must be reported in Phase 1D.

## 22. Phase 1D readiness

Ready for **engineering** work that does not need new data: trial generation and scoring
over stored references, coverage reporting, capture-manifest ingestion (take-ID and
lineage resolution), and a batch enrollment driver. Not ready for any accuracy,
threshold or model-selection claim.

## 23. Exact next engineering step

Add a capture-manifest–driven ingestion step that (a) resolves every query/enrollment
file's `original_take_id`, subject, session and view from the validated manifest by SHA,
removing the manual flags and closing the undeclared-derivative gap, and (b) feeds the
existing deterministic trial generator with per-recipe stored references. Still pending
externally: the representative four-view pilot (2–3 consenting volunteers, ~30 s,
`left45/right45/toward/away`, real metadata), then the 40-person cohort.
