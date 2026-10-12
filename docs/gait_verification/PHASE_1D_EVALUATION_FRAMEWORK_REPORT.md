# Gait Verification V2 — Phase 1D evaluation-framework report

**Date:** 12 October 2026. **Branch:** `feat/gait-verification-v2`. **HEAD at start:** `d265d53`
(Phase 1C committed; tree clean). Phase 1D changes are **uncommitted**. Author: Claude Code.

**PHASE 1D ENGINEERING: COMPLETE.** Real new-cohort scientific evaluation: **not possible
and not attempted.** No cohort exists, so no biometric performance is reported.

## 1. Executive summary

Phase 1D adds four things:

- a capture registry that is the authoritative lineage source for managed-study workflows;
- subject-level split manifests with a lock and a locked-final access guard;
- deterministic, versioned genuine/impostor trial manifests with invariant checks;
- a restricted offline evaluator built on the existing Phase 1C `ClaimVerifier`, with
  metrics, subject-cluster uncertainty and paired comparisons.

Two existing defects were fixed:

1. The metric engine raised an error on any `POLICY_NOT_READY` trial that carried a score.
   Every real trial today is in that state, so no real trial could have been evaluated.
2. Multi-recipe enrollment wrote recipes one at a time, so an interruption could leave a
   partial group. It is now a single atomic write.

The Phase 1C gap is closed for registered files. A stabilized derivative of an enrolled
original, submitted **with no lineage flags**, is resolved by checksum and refused
(`SHARED_ORIGINAL`). This was demonstrated with a real vid.stab derivative. An unregistered
re-encode fails closed (`UNREGISTERED_PROVENANCE`) instead of being treated as a new original.

No threshold, calibration, model selection, training, or locked-test access occurred.
Every metric in this report's tests comes from synthetic score arrays.

## 2. Phase 1C handoff verification

- Before any edit, the permitted suite reproduced the Phase 1C result exactly:
  **148 passed, 4 skipped, 2 deselected**.
- The Phase 1C report says its changes were uncommitted. Git shows they were since committed
  as `d265d53`, and the tree was clean at the start of this phase.
- All four registry entries remain `BLOCKED`, with `production_ready=false` and null
  thresholds.
- The Phase 1C limitation (§14 of its report) was confirmed in the code. The leakage check
  used only the probe's own take/hash plus manually declared lineage.

## 3. Git branch and working-tree status

The branch is `feat/gait-verification-v2`. Nothing was reset, cleaned, reverted, committed,
pushed or merged. The changed and new files are listed in §20.

One accidental formatter rewrite of the untouched `pose_quality.py` was noticed. It was
inspected (a formatting-only diff) and restored to HEAD.

The E0 paths have **no diff**: `models`, `utils`, `scripts/{training,inference,evaluation}`,
`tests/test_future_work.py` and `outputs/checkpoints`. Smoke artifacts live in git-ignored
`outputs/gait_verification/phase1d-smoke/`.

## 4. Existing code reused

| Reused | How |
|---|---|
| `ClaimVerifier` | Unchanged claim-only scoring and status precedence for every offline trial, through a one-claim store view |
| `scoring.py` | Baseline prototype cosine |
| `DirectoryEnrollmentStore` | Same class and file layout; adds `add_references` |
| `pipeline.enroll_original`, `build_template`, `CachedPoseExtractor`, `query_verifier` | Reused by the registry-driven paths |
| `evaluation.VerificationTrial` | Extended additively |
| `verification_metrics`, `subject_bootstrap` | Strengthened, same names |
| Capture-row validator | Moved into the package; the script re-exports it |
| `capture_schema.json` | Raw registry rows *are* capture-manifest-v1 rows |
| `stabilization.stabilize` | Its manifest `input/output.sha256` is the derivative proof |
| `models.json` | Model IDs and training datasets |

## 5. Capture-manifest implementation

The new module is `gait_verification/capture_registry.py`, schema `capture-registry-v1`:
`{"schema_version", "assets": [...]}`.

**Raw original row:** every `capture-manifest-v1` field, plus `asset_id`,
`variant_id="raw"`, `parent_asset_id=null` and an optional `split_role`. `date` must be an
actual ISO datetime with a UTC offset.

**Derivative row:** `asset_id`, `variant_id`, `parent_asset_id`, `original_take_id`,
`subject_id`, `sha256`, `source_file_uri` and `transformation_manifest_ref`. Optional fields
are `transformation_manifest_sha256`, `split_role`, `recording_notes`, and
`session_id`/`date`/`view_label` (these must equal the original's values when present).
Session, time, view and consent reference are inherited from the original.

**Operator-supplied vs program-verified.** `summary()` lists the operator-supplied fields
(subject, session, time, view, consent reference, capture notes). `verify_files(data_root,
measure=True)` checks, without modifying anything:

- actual file SHA-256;
- ffprobe facts (size, rotation, codec, nominal/average FPS, frame count, PTS span, duration);
- that each transformation manifest is `status=OK`, its input SHA is the parent and its
  output SHA is this file, plus its own SHA when one is declared.

It reports `consent_verified_by_program: false`: a consent reference is not proof of consent.

**Paths and writes.** Paths must be relative to an explicit data root; traversal outside it
is refused. Generated manifests, splits, trials and reports go through `write_new_json`
(temp file + fsync + hard link, 0600), which never overwrites an existing file.

## 6. Original/derivative lineage validation

Each item below raises `ProvenanceError` and is covered by a parametrized test:

1. Duplicate asset IDs.
2. Conflicting original take IDs.
3. One original with conflicting subjects.
4. The same bytes under two assets.
5. Missing provenance (consent reference, transformation manifest, derivative fields).
6. An unresolvable parent.
7. Cyclic ancestry.
8. A derivative subject, session, view, time or original take that differs from its original.
9. A derivative split role that differs from its original.
10. A derivative represented as an original, which is caught two ways:
    - **structurally**: a raw row with a parent or a transformation reference;
    - **by bytes**: a "raw" asset whose SHA is any registered transformation's output
      (`verify_files`).
11. A derivative with its parent's bytes.
12. Unsupported schema, unknown/misspelled fields, invalid IDs.
13. Naive or future datetimes, placeholder sessions (`unknown`, `historical`, `tbd`, …), and
    one session spanning several days.

Multi-level ancestry (raw → stabilized → re-encode) resolves to the original. Windows and
derivatives are never independent samples: trials and enrollment use only originals, plus
one explicit query variant.

## 7. Resolution of the undeclared-derivative limitation

`pipeline.verify_registered` and `verify_claim.py --capture-registry` resolve a query by its
**actual bytes**. The original take ID, variant and every ancestor's asset ID and SHA come
from the registry and go to the verifier as lineage. They do not come from flags or
filenames.

Lineage overlap is now refused **before** pose extraction, which matches precedence rank 1.
An unregistered or unreadable file returns `INVALID_INPUT` with restricted reason
`UNREGISTERED_PROVENANCE` or `VIDEO_NOT_FOUND`.

`verify_claim.py` now requires an explicit mode:

- `--capture-registry`: the managed study path.
- `--development-unmanaged`: the old Phase 1C manual flags, labelled development-only in
  help text and diagnostics.

Manual flags are refused in managed mode.

**What remains impossible:** a file that was edited, re-encoded or stabilized and **never
registered** has new bytes. SHA-256 cannot link it to its source. The registry refuses it
rather than accepting it as a new recording, but it cannot say what it is. This is not a
perceptual-copy detector. A future public-facing service, which must accept unregistered
uploads, needs a different provenance design. The managed contract here covers controlled
evaluation ingestion only.

## 8. Subject split management

`gait_verification/splits.py`, schema `subject-split-v1`.

- `SplitPolicy(version, seed, counts, development_pinned)` has **no default counts**. The
  config file holds nulls, and the CLI requires all three counts and the seed. 24/8/8 is not
  chosen anywhere.
- Assignment is a seeded SHA-256 ordering, independent of input order. Pinned pilot
  participants go to development first.
- The manifest records the policy, the subject-set hash and sorted assignments. Its
  fingerprint covers the content, not the lock metadata.
- `validate_split` checks:
  - exactly one valid role per subject;
  - the split covers exactly the registry's subjects;
  - the fingerprint matches (tampering is detected);
  - pinned subjects are still in development.
- Takes, derivatives, windows, sessions and views inherit their subject's role. A registry
  role that disagrees with the split is refused.

## 9. Locked-test protections

- `lock_split` moves DRAFT → LOCKED (records who and when). Trial generation requires a
  LOCKED split.
- `revise_split` is the only way to change a locked split. It records the revision ID,
  rationale, authorizer, previous fingerprint and moves. **Locked-final membership cannot be
  revised in either direction.**
- `check_transition` rejects any changed assignment not explained by exactly one new
  revision. A test caught a real bug here: the first version trusted a stale fingerprint.
- `require_roles` / `subjects_for` refuse `locked_final` unless all of these hold: the split
  is LOCKED, and a `LockedTestAuthorization` carries the split fingerprint, a frozen-bundle
  SHA-256, an authorizer and a reason. Trial generation, trial validation, batch enrollment
  and the runner all pass through it.
- `paired_comparison` refuses locked-final rows, so locked scores can never feed model
  comparison.
- The CLIs offer only `development`/`calibration` and expose no authorization option. The
  locked path was tested only for refusal, plus one synthetic check that a well-formed
  authorization object is recognized. It was never used to evaluate anything.
- A future locked-final subject can still be enrolled from separate reference originals,
  under authorization.

## 10. Genuine and impostor trial generation

`evaluation.generate_verification_trials(registry, split, TrialPolicy, role)` works as
follows:

- **Queries:** each original in the role, or its single registered `query_variant_id`
  derivative.
- **Reference rule `exclude_probe_view`:** references are the claimed subject's raw originals
  whose view differs from the probe's view. They are never the probe's own original or its
  descendants. Genuine and impostor claims share the same view rotation and the same
  reference count, which avoids the count leakage the historical harness documented.
- **Genuine claim:** the walker's own subject.
- **Impostors:** `impostor_claims_per_query` other subjects in the same role, sampled without
  replacement by `sha256(seed|query take|candidate)`.
- Shortfalls and skipped queries (no variant, too few references) are counted, never hidden.
- Wrong claims only substitute `claimed_subject_id`. The walker's subject labels the trial
  and is never part of a scorer input.

`validate_trial_manifest` re-checks:

- schema, registry and split fingerprints, LOCKED status, role authorization, content hash;
- unique trial IDs and unique (query, claim) pairs;
- provenance against the registry;
- label correctness;
- role membership of both subjects;
- non-empty, unique references that are originals of the claimed subject;
- no shared original or ancestry between query and references;
- enrollment views;
- that each trial ID matches its content.

Tests cover nine tampering cases.

## 11. Trial manifest schema and fingerprints

The schema is `verification-trials-v2`. Each trial row has:

- `trial_id` (a SHA-256 of policy version, query asset, claim and references);
- the query's `query_asset_id`, `query_original_take_id`, `query_variant_id` and
  `query_session_id`;
- `query_subject_id` (offline label only);
- `claimed_subject_id`, `reference_original_take_ids` and `label_same_identity`;
- `probe_view` and `enroll_views`;
- `fold_id` (`probe-view:<view>`), `split_role` and `trial_policy_version`;
- `dataset_fingerprint` and `split_fingerprint`.

The manifest also stores the full policy and reference rule, the counts and
`manifest_sha256`. It is regenerated byte-identically from the same inputs (tested with
reversed row order).

## 12. Enrollment batch ingestion

`pipeline.enroll_registered` and `enroll_subject.py batch` take explicitly listed asset IDs.
For each asset they:

1. resolve it in the registry;
2. refuse derivatives and subjects outside the selected roles (development by default);
3. re-check the file's SHA;
4. take all metadata (take, session, view, time, consent reference, split role) from the
   registry;
5. enroll one reference per recipe.

Every asset gets a result row (`ENROLLED`, `ALREADY_ENROLLED`, `REFUSED` with a reason, or
`ERROR`). Partial failures never stop the others or disappear.

**Atomic multi-recipe writes.** `DirectoryEnrollmentStore.add_references` writes all recipes
of one original in one atomic update. `enroll_original` now uses it, so the Phase 1C
`add --models` partial-group risk is removed. A simulated crash leaves no references, and a
retry succeeds; the orphan source-index entry is harmless, as before.

**Idempotent reruns.** `skip_enrolled=True` skips recipes already stored for that take.

There is no automatic traversal. No historical recording was enrolled into a study store.

## 13. Offline scoring architecture

`gait_verification/offline_eval.py`:

- **`QueryEmbeddingStore`:** one private JSON per (query asset, encoder). It holds no subject
  ID.
- **`embed_registered_queries`:** embeds the trial manifest's assets once, using the same
  extractor, quality gate and windowed encoder as enrollment.
  - Persisted: embeddings, abstentions and invalid media.
  - Reported but not persisted: transient errors and missing or changed files, so a rerun
    retries them.
  - Records runtime, and in-process CUDA peak memory where it can be measured.
- **`EmbeddingTrialScorer`:** receives only a `ClaimRequest` (trial ID, query lineage,
  claimed ID, reference takes). For each trial it:
  1. looks up the claimed subject once (cached; a spy test confirms only claimed subjects
     are read);
  2. selects exactly the trial's reference takes for the run's recipe;
  3. checks them against the registry's checksum and the split role;
  4. calls the unchanged `ClaimVerifier.verify_embedding_detailed` with registry lineage.

  Missing pieces become explicit statuses: `NO_REFERENCES`, `ENROLLMENT_INCOMPLETE`,
  `RECIPE_MISMATCH`, `ENROLLMENT_PROVENANCE_MISMATCH`, `QUERY_EMBEDDING_MISSING` and
  `QUERY_PROVENANCE_MISMATCH`.
- **`run_evaluation`:** validates the manifest and keeps every trial in order. It refuses
  mixed recipes, a scorer that returns a score with an abstention status, and an evidence
  class that doesn't match the role.
- **Outputs:** `write_run` creates a new private directory containing `report.json` (no
  embeddings, consent references or paths) and `outcomes.jsonl`. The outcomes file is
  restricted: it holds per-trial raw scores, bound to the report by SHA-256.
- **Policy:** the scorer refuses any non-ready policy, and refuses a calibrated-looking
  policy for a BLOCKED model. A `TEST_ONLY` policy is accepted only with
  `allow_test_policy`, inside an isolated test. Public `VerificationResult` is unchanged.

Per-trial fields: `status`, `reason`, `score`, `verdict`, `score_available`,
`quality_evaluable` and `operational_decision_available`. A valid score with no threshold is
`POLICY_NOT_READY`: scored, quality-evaluable, not decided, and **not** a quality failure.

## 14. ROC / EER / TAR / FAR / FRR

`verification_metrics` (orientation: a higher cosine supports the claim):

- **ROC:** tie-aware (tied scores move together); AUC by trapezoid.
- **EER:** linear interpolation of the FAR = FRR crossing. It is descriptive only.
- **TAR@FAR:** for every declared FAR it reports `{tar, empirical_far}` at the best achievable
  FAR ≤ target, with no interpolation. It warns when the target is below 1/impostors.
- **FAR, FRR and confusion counts:** produced only when a threshold is passed in. The
  boundary rule is that score = threshold accepts. Neither the CLI nor the runner can supply
  a threshold, because none is authorized.
- **Undefined metrics** (one class, all abstained, no trials, no impostors for FAR) stay
  `None` with a reason in `undefined`. Nonfinite scores, scores on abstentions, invalid
  targets and invalid thresholds raise errors.

Known-value tests, all on synthetic arrays:

| Case | AUC | EER | TAR |
|---|---|---|---|
| 3 genuine / 3 impostor | 8/9 | 1/3 | 2/3 at 5% FAR, 1.0 at 1/3 FAR |
| Tied pair | 0.875 | 0.25 (interpolated) | — |
| Flat-FAR crossing | — | 0.5 | — |
| Reversed scores | 1/9 | — | — |

The tied-pair case also checks threshold boundary counts.

## 15. Coverage and abstention accounting

The report keeps three coverages separate:

- **score coverage:** scored / submitted;
- **quality coverage:** queries that passed quality / submitted;
- **operational coverage:** the PRD definition, `status=OK` / submitted. It is **0** until an
  authorized policy exists.

`status_reason_counts` breaks every `status/reason` down by genuine and impostor. Metrics are
also grouped by probe view, query subject and query session. A warning is added when no
subject has more than one session ("cross-view / different-take, not cross-session").

## 16. Subject-aware uncertainty

`subject_bootstrap` resamples **probe subjects** with all of their trials, so sessions stay
nested. It never resamples pair scores, frames or windows. Seeds are fixed and results are
deterministic. It gives percentile 95% intervals for AUC, EER and TAR@FAR.

It returns no interval in these cases:

- `UNDEFINED_METRIC`: fewer than two subjects, or no valid replicate.
- `UNSTABLE`: fewer than 90% of replicates are defined.

It warns `FEW_CLUSTERS` below 10 subjects.

**Limitation:** claimed subjects are crossed across trials, so this is not a full two-way
cluster model. The synthetic fixtures say nothing about the uncertainty of the future cohort.

## 17. Paired model-comparison design

`paired_comparison` and `compare_runs`:

- **Refuse** when:
  - the manifests, registries or splits differ;
  - the trial IDs, labels, subjects or roles differ;
  - the same system appears twice;
  - the pretraining datasets differ (GPGait++ CASIA-B is reported separately from the three
    GREW candidates);
  - any locked-final rows are present.
- **Report:**
  - each system's full-manifest coverage;
  - metrics on the explicitly identified common-scoreable subset, with a hash of its trial
    IDs;
  - A − B deltas;
  - a paired probe-subject bootstrap of those deltas.
- **Never name a winner.** A test asserts that no "winner" or "better" appears in the output.

## 18. E0 comparison readiness

The runner's scorer contract is `ClaimRequest → (status, reason, score, verdict,
quality_evaluable)`. A test runs an E0-shaped stand-in scorer through the same runner and
compares it with the embedding scorer on identical trial IDs.

A real E0 adapter must map each request's query original and reference originals to its
historical 78-channel features. It must **never** feed those features to the pretrained
encoders. It also needs a development-split E0 retrained under the PRD's §8.3 rules, which
is not authorized now.

E0 code, checkpoints and results are untouched. A test asserts that the V2 package never
imports `models`/`utils`. **Empirical E0 parity has not been demonstrated.**

## 19. Tests and results

The permitted command (legacy training and calibration tests stay deselected; their bodies
were inspected in earlier phases and again here):

```bash
.venv-gait-v2-cuda/bin/python -m pytest -q tests/gait_verification tests/test_future_work.py \
  -k "not test_harness_end_to_end_on_synthetic_subjects and not test_metrics_against_known_values"
```

| Run | Result |
|---|---|
| Baseline before edits | 148 passed, 4 skipped, 2 deselected |
| Final, default | **212 passed, 4 skipped (opt-in CUDA), 0 failed, 2 deselected** |
| Final, `GAIT_V2_CUDA_TESTS=1` | **216 passed, 0 skipped, 0 failed, 2 deselected** (all four models) |
| `test_phase1d.py` | 64 passed |
| Ruff check, Ruff format (all new/changed files), `compileall`, `git diff --check`, whitespace scan | passed |

The 64 Phase 1D tests by evidence type:

1. **Pure unit (45):** registry rules and resolution; splits, lock and access; metric known
   values and edge cases; bootstrap; paired comparison; atomic store; public contract; E0
   import isolation.
2. **Synthetic cohort (17):** trial generation and tampering; offline runs; enrollment
   statuses; policy refusal; run comparison; the full CLI workflow via subprocess.
3. **Real media integration (2):** ffmpeg plus **real vid.stab** derivative and manifest,
   real decode/pose cache/window quality with a fake detector and stand-in encoder; managed
   resolution; batch enrollment; query ingestion.
4. **Real CUDA engineering smoke (CLI, not pytest)** in `outputs/gait_verification/phase1d-smoke/`.
   It used three newly generated synthetic `testsrc2` clips registered with their actual
   generation time; historical clips have no real recording dates, so they were not
   registered. With real RTMPose and GaitPT on an RTX 3050:
   - `validate --measure`: ok;
   - split, lock, trials: 2 genuine, 2 impostor, 1 shortfall, 1 skipped (matches a hand
     count);
   - `batch`: both refused, `insufficient_pose_information,…`;
   - `embed`: 3 `INSUFFICIENT_GAIT` persisted;
   - `run`: 4 trials preserved as `CLAIM_NOT_ENROLLED`;
   - `verify_claim --capture-registry`: a byte-identical copy under another filename
     resolved to the registered take; a re-encoded copy gave `UNREGISTERED_PROVENANCE`
     without loading models.

   The registry also refused the first attempt, because deterministic `testsrc2` output made
   three "originals" byte-identical. This is plumbing evidence only.
5. **Real new-cohort evaluation:** unavailable.

## 20. Changed files

**New:**

- `gait_verification/{capture_registry,splits,offline_eval}.py`
- `scripts/gait_verification/evaluate_claims.py`
- `configs/gait_verification/phase1d_evaluation.json`
- `tests/gait_verification/test_phase1d.py`
- this report

**Modified:**

- `gait_verification/evaluation.py`: trials v2, metrics, bootstrap, paired comparison.
- `gait_verification/pipeline.py`: `enroll_registered`, `verify_registered`, atomic and
  idempotent `enroll_original`, `asset_id` passthrough.
- `gait_verification/enrollments.py`: `add_references`.
- `gait_verification/verifier.py`: lineage refused before extraction.
- `gait_verification/registry.py`: `REGISTRY_KEYS`, `model_entry`.
- `scripts/gait_verification/verify_claim.py`: required managed/unmanaged mode.
- `scripts/gait_verification/enroll_subject.py`: `batch`.
- `scripts/gait_verification/validate_capture_manifest.py`: reuses the moved validator.
- The PRD (§20) and handoff (Phase 1D amendment).

No existing test was changed.

## 21. Remaining engineering limitations

- Unregistered modified copies cannot be linked to their source (§7).
- Take IDs following the SOP pattern embed the subject pseudonym. The verifier uses take IDs
  only for equality lineage checks, never for lookup or scoring.
- The registry is a JSON file. Integrity rests on checksums, private permissions and
  never-overwrite writes; there is no signature or access audit. The enrollment and query
  stores are not encrypted.
- A split freezes the registry's subject set. A participant added after locking needs a new
  split version, not a revision.
- `query_variant_id` other than `raw` yields `RECIPE_MISMATCH` against raw references,
  because the pose variant is part of the recipe. The raw-vs-stabilized evaluation design
  (matched-variant references) is a Phase 4 decision.
- Only one reference rule exists (cross-view). A same-view multi-take or multi-session rule
  needs those recordings to exist first.
- CUDA peak memory is measured only for in-process models (GaitPT). GPGait-family workers
  report `None`.
- The bootstrap is probe-subject only (§16).
- `graphify` is not installed, so the graph was neither queried nor updated.

## 22. Scientific and data-collection blockers

- The four-view pilot (2–3 consenting volunteers) and the ~40-person cohort are not
  collected.
- No subject split counts, seed, impostor sampling count or target operating FAR is
  approved.
- There is no E0 retraining authorization, threshold, calibration or model readiness.
- If all four views are recorded on one day, results are cross-view only.
- Licensing for any deployment is unresolved.

## 23. Phase 2 dataset-integration readiness

The engineering side is ready. Received originals can be:

1. registered (raw rows are capture-manifest rows plus four registry keys);
2. verified (`validate --data-root --measure`);
3. split once the owner approves counts and seed (`split` → `lock`);
4. turned into frozen trials;
5. batch-enrolled (development);
6. embedded per model;
7. scored into restricted reports;
8. compared on identical trials.

Nothing in Phase 2 requires the historical 13-person data.

## 24. Exact next recommended step

Record the four-view capture pilot following `CAPTURE_SOP_DRAFT.md`. Register it with
`capture-registry-v1` rows carrying real dates, sessions and consent references. Run
`evaluate_claims.py validate --data-root … --measure`.

Then make **owner decisions**, recorded in the PRD before any split is locked:

- split counts and seed, with pilot subjects pinned to development;
- `impostor_claims_per_query` and the trial seed;
- the declared target FAR(s).

Only then run `split`/`lock`/`trials` on development subjects, plus `batch`/`embed`/`run`
for the three GREW candidates (CASIA-B separately). Do not select a model or threshold from
pilot data.
