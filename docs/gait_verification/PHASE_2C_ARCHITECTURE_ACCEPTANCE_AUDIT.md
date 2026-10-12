# Phase 2C — Independent pre-collection architecture acceptance audit

**Audit:** 12–13 October 2026 (Asia/Kolkata). **Reviewer:** Codex, with independent
read-only model/E0 and evaluation/security review perspectives.
**Baseline:** `feat/gait-verification-v2`, `f1213c7`.
**Final verification:** complete on 13 October 2026. Both permitted suites, static
checks and preservation reconciliation passed. No reproduced critical/high blocker remains
in the audited managed-intake scope.

## 1. Executive summary

Passing Phase 2B tests did not establish acceptance. Both reported baselines reproduced,
but adversarial review found protected-data entry-point bypasses, a direct wrong-person
reference path, split-transition loopholes, mutable trial policy, writes through linked
ledgers and source/cache/report contract defects. Bounded corrections preserve the existing
encoders, E0 weights/features and scientific sampling procedure. Regression tests reproduce
the defects rather than weakening prior checks.

The complete generated-media workflow now reaches scored, undecided E0/GaitPT comparison
on one manifest. Pose and E0 feature observations are explicitly synthetic injections;
the opt-in variant uses actual frozen CUDA weights. Neither variant is biometric evidence.
All successful real-model scores remain `POLICY_NOT_READY`, `verdict=null`.

## 2. Engineering acceptance verdict

**ARCHITECTURE ACCEPTED WITH DOCUMENTED ENGINEERING LIMITATIONS**

Acceptance concerns ingestion of properly collected research recordings only. It does not authorize a study
protocol, TRUE/FALSE operational decisions, accuracy claims, model selection, collection,
production deployment or security certification.

## 3. Current Git baseline and preservation

The actual initial tree was clean, with no staged, unstaged or untracked changes.
Phase 2/2B had been committed; the pasted expectation of uncommitted work was stale.
Recent commits: `f1213c7` (Phase 2B tests), `433cdd0` (Phase 1D), `d265d53` (Phase 1C),
`704bf09` (Phase 1B), `02b0312` (Phase 0 CUDA), `d041f07` (Phase 0 foundation).
No commit, staging, push, merge, reset, clean, revert or checkpoint replacement occurred.

A pre-edit SHA snapshot covered 1,228 files, including original and augmented historical
media, E0 source/checkpoints and mandatory documents. The protected non-document set is
1,216 files. Historical phase reports are checked separately. New biometric/runtime
artifacts remain ignored. The final worktree contains only this audit's bounded edits.

## 4. Scope, contract and limitations

Only `(walking_video, claimed_subject_id)` 1:1 verification is supported. References
belong solely to the supplied claim. No identity search, alternate-person disclosure,
deepfake attribution, score fusion, local training, calibrator fitting, threshold
installation, model winner, physical collection or actual locked-final opening occurred.
The planned ~40 × 4 × ~30-second cohort remains future work.

The PRD and its dated owner amendments govern. In particular, collection is deferred,
frozen shipped E0 replaces no controlled LOSO artifact, CASIA-B is a separate ablation,
and engineering success does not satisfy scientific/deployment gates.

## 5. Documents and source reviewed

Read in full: `AGENTS.md`, both gait handoff/PRD documents, Phase 0, 1A, 1B, 1C, 1D,
2 and 2B reports, the Phase 2 operator guide and deferred register. Root `CLAUDE.md`
is absent. Historical phase reports were preserved.

Reviewed the capture schema and all gait configs; `capture_registry`, `dataset_ingest`,
`dataset_processing`, `media`, `pose`, `pose_quality`, `skeleton_contracts`, `pipeline`,
`registry`, `enrollments`, `scoring`, `verifier`, `splits`, `evaluation`, `offline_eval`,
`common_eval`, E0 modules and worker; pretrained loaders/contracts/workers/extraction;
operator CLIs and gait tests. Independently compared historical `models`/`utils` feature,
normalization, signature and decisive-head paths with E0. Reviewed pinned upstream
GaitPT/FastPoseGait source/configuration and local dependency snapshots/ignore rules.

## 6. Phase-by-phase verification matrix

| Phase | Independently verified engineering | Still unvalidated |
|---|---|---|
| 0 | Native mapping, timing, frozen safe loaders, public claim-only contract, E0 preservation | Source-dataset reproduction and scientific validity |
| 1A | Temporal windows, ambiguity/abstention, actual decoding and RAW/S10/S15 tests | Representative moving-camera quality and final SOP |
| 1B | All four strict checkpoint/CUDA paths, dimensions, normalization, repeat/frozen tests | Target-domain validity; CASIA-B remains separate |
| 1C | Independent references, claim-only spies, recipe/source checks, lazy status precedence | Operational policy, enrollment lifecycle/security deployment |
| 1D | Splits, deterministic trials, all-trial coverage, metrics/paired comparisons; corrected guards | Real study, two-way uncertainty, approved sampling/FAR |
| 2 | Source/ancestry checks, read-only plans, sealed caches, quarantine/resume, bounded processing | Multi-writer transactions, power-loss guarantee, real 160-video throughput |
| 2B | Actual shipped E0 safe extraction, native features, frozen head parity, common-run integration | Controlled LOSO replay, quality equivalence, unseen-subject performance |

## 7. End-to-end execution flow

```text
registry + explicit asset selection + validated split
  -> source/ancestor/transform verification -> read-only plan
  -> native pose NPZ + sidecar -> provisional temporal quality
  -> checkpoint-native conversion -> frozen encoder -> walk embedding
  -> independent claimed-person enrollment references
  -> v3 common trial manifest -> one-claim scorer -> private outcomes/report
  -> full coverage + paired/common-scoreable comparison (no winner)

separate E0 path:
registered original RGB -> native MediaPipe33 IMAGE observations
  -> historical 60 x 78 features -> frozen checkpoint z-score tensors
  -> mean z-scored claimed-reference signature -> decisive TCN logit margin
  -> same trial IDs/labels/reference sets -> same coverage/metric framework
```

Labels and actual query identities stay in offline trial construction/reporting.
`ClaimRequest` carries query lineage, claimed ID and exact references, never the label
or actual walker ID. The production result contains only claim, schema, status and verdict.

## 8. Severity-ranked findings register

Every correction below is FIXED unless explicitly marked DOCUMENTED. Commands in §18
reproduce the regression evidence. No scientific policy or model semantics were changed.

| ID | Severity | Affected files | Reproduction/evidence and impact | Correction / regression | Remaining limit |
|---|---|---|---|---|---|
| C-01 | CRITICAL | `pipeline.py`, `verify_claim.py`, `offline_eval.py`, `evaluate_claims.py` | Managed verification lacked split/ancestry guards; forged development `embed` header reached protected assets; direct scorer read final artifacts | Required managed context and validated full manifest; role guards before model/artifact access. `test_phase2c_boundaries`/`evaluation` | Caller-supplied byte aliases require hashing to resolve; unknown provenance fails closed |
| C-02 | CRITICAL | `offline_eval.py` | Relabel another subject's reference while retaining take/hash: direct scorer produced a score | Check registry ownership of every requested reference before store reads; owner spy regression | Registry subject labels are operator assertions, not physical identity proof |
| C-03 | CRITICAL | `splits.py` | Rehashed revision moved final subject into development; LOCKED→DRAFT passed | Validate transitions, forbid unlocking and any final membership change; two regressions | Unsigned standalone JSON cannot defeat an authorized filesystem writer |
| C-04 | CRITICAL | `dataset_processing.py` | Symlink/hard link from ledger to an original received appended bytes | No-follow descriptor open and single-link check before writes; two original-immutability regressions | Trusted hierarchy/single writer; no hostile concurrent ancestor-swap guarantee |
| C-05 | HIGH | `evaluation.py` | Policy seed/minimum/rule changed without row-hash change; rehashed forbidden references passed | v3 full-contract hash plus exact deterministic regeneration; policy/reference tests | v2 archives preserved; regenerate new files from unchanged explicit inputs |
| C-06 | HIGH | `offline_eval.py`, `pipeline.py`, scoring CLI | Changed source returned EXISTS; sealed S10 cache scored RAW; source substitution could publish enrollment; derivative pose relabeled RAW | Source checks before hits/publication/scoring, variant/recipe checks, source-root requirement and raw-pose refusal; cache/enrollment regressions | Genuine unsealed Phase 1D offline compatibility remains lower trust |
| C-07 | HIGH | `offline_eval.py` | Non-OK Boolean, OK/null, scored quality=false, calibration mislabeled development, mixed evidence comparisons accepted | Enforce result/quality/evidence-role contracts; synthetic outcome/evidence regressions | Synthetic/STUDY declarations are not authenticated attestations |
| C-08 | MEDIUM | `capture_registry.py`, `dataset_processing.py`, `offline_eval.py` | List transform, null/list ledger events or string timing crashed recovery/reporting | Object/event/numeric checks; malformed entries refused or ignored, all trials retained | Checksums are not signatures; writer can recompute them |
| C-09 | MEDIUM | `pose.py`, `stabilization.py` | New NPZ/derivative manifests had 0644/0755 defaults under umask 022 | Private exclusive NPZ, 0700 new derivative directory, 0600 artifacts/logs and atomic JSON; permission tests | Existing artifacts not silently chmodded; operator must restrict data root |
| C-10 | MEDIUM | `e0.py` | Stub E0 CUDA worker dispatched while sole cooperative slot was held | Hold default shared slot before CUDA dispatch; occupied-slot regression | Direct enrollment/verification/legacy embed scheduling remains separate |
| C-11 | MEDIUM | `common_eval.py`, scoring CLI | Comparisons omitted MediaPipe-vs-RTMPose gate asymmetry; runtime omitted E0 scoring | Gate descriptors/asymmetry and explicit extraction-only runtime scope; five model/report tests | No gates scientifically equated or validated; scoring throughput unmeasured |
| C-12 | MEDIUM | `verify_claim.py` | Unenrolled claim tried loading CUDA weights before lookup | Lazy extraction/encoder after lookup; loader-spy regression | Direct library `make_verifier` failures still return explicit processing error |
| C-13 | LOW | `common_eval.py` | Equal gate dictionaries with reordered keys reported inequality | Semantic dictionary equality; order regression | UNKNOWN descriptors remain explicitly unknown |
| C-14 | MEDIUM | Processing/storage/workflow | Cooperative slots, separate direct CLI loaders, cold E0 per-trial load, one-writer and power-loss limits | DOCUMENTED in guide and §§16/21; existing OOM/busy/recovery tests pass | Run one writer and one active GPU workflow; no hard quota or 160-video SLA |
| C-15 | MEDIUM | Requirements/setup | Existing CUDA env verified; clean machine and overlapping ORT/OpenCV namespaces unproven | DOCUMENTED; `uv pip check`, pinned revisions/hashes and 13 CLI helps verified | No fresh install, dependency vulnerability certification or license clearance claimed |

## 9. Defects reproduced

The initial boundary file produced ten RED failures; evaluation review reproduced
locked/owner/split/policy/cache/outcome/evidence/timing defects independently; model
review produced five RED failures. Later regressions cover enrollment source/variant,
lazy unenrolled lookup, private stabilization and gate ordering. The corrected source
substitution fixture was also run against the original HEAD enrollment function in an
isolated module: it failed with DID NOT RAISE, then passed against the correction.

All malicious-checkpoint cases retain restricted loading and execute no payload.
All protected-data repros use invented identities/bytes, never real final subjects.

## 10. Fixes and compatibility

Changes strengthen existing boundaries rather than replace working modules. Native
feature and encoder math, weights, comparison guard policies, reference rule, split
counts/seeds and FAR defaults are preserved. The independent reviewer compared old and
new generation across sixteen synthetic combinations: identical trials, counts and
row hashes. v3 adds the whole-contract seal and refuses inconsistent regeneration.

Managed CLI changes are intentional: `verify_claim` requires data root/split; `embed`
requires split; all scorers require a data root. Archived v2 files remain unchanged.
Regeneration uses their same explicit policy/registry/split into new files, not silent
in-place migration. Existing Phase 1D tests keep their assertions and supply the new
context; no tests were disabled to obtain pass counts.

## 11. Security, privacy and licenses

New managed stores use exclusive/private writes, safely loaded NPZs and tensor-only
checkpoint loads. Linked append targets are refused. Biometric outputs remain private
and ignored; public results disclose no scores, embeddings, alternate IDs or paths.
There is no encryption, protected signing key, access audit, key management or service
authentication certification. A writer can rewrite data and SHA seals together.

Pinned local GaitPT source carries CC BY-NC-ND 4.0; FastPoseGait declares academic/
noncommercial restrictions. Repository MIT metadata grants no external checkpoint,
dataset, historical biometric consent or deployment rights. Model/checkpoint/pose
licenses, retention/deletion, consent custody and deployment approval stay OPEN.
No new restricted dataset/checkpoint or external service was accessed/downloaded.
Private logs, hash/CRC/static-check evidence and both full operator fixture directories
are retained in ignored, 0700/0600 `outputs/gait_verification/phase2c-evidence/`.

## 12. Provenance and leakage

Tests cover duplicate IDs/bytes/takes, changed sources/parents/transforms, registered
derivatives, unknown copies, crossed roles, same-original probe/reference overlap,
incorrect labels/ownership, stale recipes and protected subjects. Actual PTS are decoded;
windows never become independent takes or references. Stabilization keeps original
bytes, timing and raw-vs-derivative lineage. No raw/stabilized winner is selected.

SHA-256 can establish exact-byte equality and corruption, not physical take independence,
participant identity, valid consent, true capture time/session, or ancestry of an
unregistered re-encode. A dishonest new raw registration with distinct bytes cannot be
scientifically disproved by these hashes. Managed inputs therefore require accountable
capture registration and an explicit independent-original protocol.

## 13. E0 checkpoint and native feature review

Actual shipped source: `outputs/checkpoints/checkpoint_epoch_46_best.pth`,
SHA `6c37f3b9b9172b42e6329a310f3e51d0b9882a5a9d59806851687731d67b1589`.
Extracted head SHA `f65c44cde5af178b154efbcf47b9aa18dc67c37db19b63e25d27b9a781be65fd`;
133,058 parameters. Hash-pinned static slicing plus unchanged `weights_only=True`
preserves tensor statistics/decisive weights and rejects unsafe/incompatible inputs.

Native MediaPipe33 xyz joints 11,12,23,24,25,26,27,28,29,30,31,32 supply 36 hip-relative
coordinates, six image-plane angles and 36 detected-index differences with last-row
padding. Sixty-step detected-index resampling and checkpoint z-score statistics remain
historical. Differences are not physical velocities. Frozen head/signature numerical
parity, state/repeat checks and real CUDA tests passed. No COCO17 heel/toe fabrication.

Origin remains `historical-local-13-subject-shipped`; the absent controlled LOSO bundle
is a separate research/reproduction limitation. No controlled-metric replay, historical
training-population independence or new-cohort generalization is asserted. Its minimal
MediaPipe gate is less extensive than RTMPose temporal quality, now visible in reports.

## 14. Pretrained model compatibility

| Candidate | Checkpoint SHA (full digest in private artifact audit) | Native contract | Measured dimension |
|---|---|---|---:|
| GaitPT GREW | `e6b287e2…f824637` | Height scaling; official repeated-nose reorder; two center30 crops | 256 |
| GPGait GREW | `12c15082…68a1302` | Affine→RescaleCenter→HOD→SkeletonInput | 4,864 |
| GPGait++ GREW | `be1bcbf9…930a45b` | GREW native YAML/transform contract | 4,864 |
| GPGait++ CASIA-B | `0b13777d…bb94ca3` | CASIA-B YAML, no GREW Affine | 4,864 |

The pinned, clean upstream revisions are GaitPT `4977618a36ef77ee9a2863c9f8e27ab862894d70`
and FastPoseGait `d3ce631971a01d806eeb3a154774d482c0951813`. Actual inference files match
registry hashes. All four original checkpoint hashes and every ZIP CRC also pass
(289 / 4,621 / 8,041 / 2,438 members respectively). Native preprocessing parity, frozen CUDA, normalization, determinism,
wrong checkpoint/model/CPU/worker rejection are tested. Independent GaitPT transform
checks matched official geometry across eight synthetic lengths 31–900.
All model readiness remains BLOCKED. CASIA-B stays an ablation, never a primary GREW
comparison or pooled embedding. Source-dataset and Android-domain validity remain OPEN.

## 15. Statistical/evaluation correctness

Higher score supports a claim, but E0 margin and cosine scales remain separate.
Every trial/abstention remains in reports; score, quality and operational coverage
stay distinct. Scored `POLICY_NOT_READY` is descriptive evidence with zero operational
coverage. Scoreless `POLICY_NOT_READY` also remains supported, never a Boolean.

Tie-aware AUC, interpolated descriptive EER, discrete TAR at attainable FAR, `>=` boundary,
undefined metrics, genuine/impostor counts and common-subset guards passed. An independent
reviewer checked 939 tied two-class examples against pairwise AUC and threshold enumeration.
Probe-subject clustering retains correlated windows/trials but is not full two-way
claim/probe uncertainty; FEW_CLUSTERS/undefined replicates remain disclosed.

Matched-pretraining guards stay strict. The explicit E0-vs-GREW exception checks shipped
origin/semantics; CASIA-B stays separate. Paired comparisons require the same evidence
class/manifest and refuse final rows. No numeric fusion, threshold fitting or winner.

## 16. CUDA and recovery

Observed Python 3.11.17, torch 2.10.0+cu128 / CUDA 12.8, ORT 1.23.2;
RTX 3050 Laptop 6,144 MiB, driver 595.99.02. `uv pip check`: 83 compatible packages.
Actual CUDA tests cover RTMPose on generated media and all four frozen encoders, plus
E0 parity and the full real-weight operator workflow. No silent CPU fallback is used
in those paths; the default operator fixture explicitly chooses CPU E0 for its test.

Existing busy/OOM/load/interruption/torn-ledger/quarantine/retry tests pass. E0 reloads
its small head per trial and now holds a cooperative slot for CUDA dispatch. This is a
throughput limitation, with no demonstrated correctness failure in bounded work. Direct
enrollment/verification/legacy embedding are not a universal shared scheduler. One active
GPU workflow and one writer per artifact root remain the supported operating discipline.
Synthetic timing cannot establish real 30-second or 160-video performance.

## 17. Complete operator-workflow evidence

`test_phase2c_operator.py` invokes the actual CLI parser/main paths: validate/measure,
explicit synthetic split/lock, deterministic v3 trials, read-only plan, processing,
resume/cache reuse, coverage report, selected-reference batch enrollment, E0 native
feature preparation/resume, and `run-common`. It inspects private reports/provenance
and asserts unchanged original bytes and artifact reuse. Existing subprocess CLI tests
separately exercise packaging/real script execution, unavailable resources and status paths.

Four generated clips for two invented subjects × two views produce eight claim trials.
RGB is actual FFmpeg/PTS decoding evidence only; a named fake detector supplies explicit
anatomical observations. E0 receives explicit synthetic 60×78 features. The default
GaitPT-shaped encoder is a stand-in; the opt-in case uses actual frozen GaitPT and E0
CUDA weights. All eight trials are scoreable and undecided in each system; their identical
IDs/references flow into the common comparison. No fixture value becomes a project default.
Real RTMPose on generated patterns separately abstains; no successful gait detection is
fabricated from `testsrc2`. Native feature/head math parity has separate real-weight tests. A fresh, separate
historical-only smoke extracted real MediaPipe features from two unchanged originals
and ran the frozen head on CUDA: both 60×78, 133,058 parameters, frozen/eval, repeat
delta zero and identical state digest. Its 31.63 s includes extraction/startup; it is
not a latency/accuracy benchmark. No historical consent/date was invented or entered
into a study registry. Evidence: `outputs/gait_verification/phase2c-evidence/native-e0-smoke.json`.

## 18. Actual commands, tests and failures

```bash
.venv-gait-v2-cuda/bin/python -m pytest -q tests/gait_verification tests/test_future_work.py \
  -k 'not test_harness_end_to_end_on_synthetic_subjects and not test_metrics_against_known_values'
GAIT_V2_CUDA_TESTS=1 .venv-gait-v2-cuda/bin/python -m pytest -q \
  tests/gait_verification tests/test_future_work.py \
  -k 'not test_harness_end_to_end_on_synthetic_subjects and not test_metrics_against_known_values'
.venv-gait-v2-cuda/bin/ruff check gait_verification scripts/gait_verification tests/gait_verification
.venv-gait-v2-cuda/bin/python -m compileall -q gait_verification scripts/gait_verification tests/gait_verification
git diff --check
uv pip check --python .venv-gait-v2-cuda/bin/python
```

| Execution | Actual result |
|---|---|
| Unmodified default baseline | 364 passed, 10 skipped, 2 deselected; 68.92 s |
| Unmodified CUDA baseline | 374 passed, 0 skipped, 2 deselected; 112.59 s |
| Final default | 404 passed, 11 skipped, 2 deselected; 81.72 s |
| Final CUDA | 415 passed, 0 skipped, 2 deselected; 140.92 s |
| Full operator default / real CUDA focused | 1 pass + 1 CUDA skip / 1 pass; 15.68 / 20.87 s |
| Last privacy/gate/media slice | 24 passed |
| Final deferred-register guard after document reconciliation | 1 passed, 59 deselected |
| Independent model review slices | 64 passed/1 CUDA skip; final 33 passed/1 CUDA skip |
| Lint, changed-file formatting, compile, whitespace | Passed; formatting checked all 17 changed Python files |
| Preservation | 1,216 protected files unchanged, no historical phase-report changes, no private V2 tracked artifacts |

The two legacy training/calibrator-fitting tests were excluded throughout. Default
skips are the ten existing CUDA cases plus the new real-weight operator CUDA case;
no required case was silently disabled. PyTorch nested-tensor optimization warnings
remain non-failing. Initial RED failures are expected evidence. Intermediate failures
included an insufficient-motion test fixture and an E0 fixture tuple/key unpacking error;
both were corrected and transparently separated from runtime defects. A preservation
script initially classified the intentionally updated operator guide as a historical
phase report; the final check distinguishes them and passes. A malformed patch
attempt applied no source changes; subsequent exact patches and checks succeeded.

There are 41 added collected cases: 40 default passes and one additional opt-in CUDA
operator case. Full final suites have zero failures. The branch/HEAD remain unchanged,
and all 22 changed/new files are unstaged for owner review.

## 19. Reproducibility

Thirteen operator scripts returned valid `--help`. Current-machine dependencies,
source revisions and checkpoint fingerprints are pinned/checked; no install/download
was performed. Root legacy dependency metadata differs from the isolated CUDA stack;
use the documented V2 environment. CPU/GPU ORT and OpenCV packages share namespaces,
so an install success alone is not proof of CUDA execution. Exact source repositories
and ignored weights must be acquired under their existing permissions; they are not
bundled into the public tree. A clean-machine install and full official evaluators
were not attempted, and current runtime tests establish neither.

## 20. Deferred external/scientific requirements

All EXT/OWN/SCI/SEC entries remain OPEN. This includes valid consent/capture metadata,
pilot, true independent views/days, SOP approval, cohort, approved split/trial/FAR
inputs, source-sample reproduction, quality/stabilization/model evidence, calibration,
controlled E0 reproduction/retraining permission, locked-final authorization, licenses,
retention, consent mapping and deployment approval. Same-day data cannot establish
cross-day robustness. The audit creates no recording request or collection date.

## 21. Remaining engineering risks

Supported intake is a local managed research workflow with trustworthy registration,
restricted filesystem access, one writer per artifact directory and one active GPU
workflow. Checksums establish integrity, not authenticity. Genuine old unsealed records
remain a disclosed compatibility limit; new processing and managed pose paths are stricter.
Direct library injections can deliberately omit filesystem checks for isolated synthetic
tests; managed CLIs require the necessary root/split context.

No hard VRAM/host-memory scheduler, concurrent service transaction, fully fsynced directory
power-loss guarantee, hostile filesystem-race defense, automatic rollback service,
arbitrary-upload provenance or participant erasure service is established. E0 timing
excludes scoring; native quality gates are unequal. Real long-video throughput and
clean-install reproducibility remain unmeasured. These are documented limits, not claims
that passing fixtures establish production safety.

## 22. Changed files

Runtime: `gait_verification/{capture_registry,common_eval,dataset_processing,e0,evaluation,
offline_eval,pipeline,pose,splits,stabilization}.py`; CLI:
`scripts/gait_verification/{evaluate_claims,verify_claim}.py`.
Tests: updated `test_phase1d.py` to supply strengthened context without weakening its
assertions; new `test_phase2c_{boundaries,evaluation,models,operator}.py`.
Docs: this report, deferred register, operator guide and substantive factual/contract
amendments appended to both main gait documents. Historical reports, E0 source,
original/augmented media, checkpoints, upstream sources and dependencies are unchanged.

## 23. Acceptance gates

| Gate | Evidence / disposition |
|---|---|
| Actual repository baseline preserved | PASS: complete checksum/Git reconciliation |
| High-confidence critical/high defects corrected | Reproductions, bounded fixes, independent re-review |
| Claimed-only access, native model/E0 contracts, no operational policy | Spies, parity, recipe/ownership and null-verdict tests |
| Provenance, protected split, policy/metric correctness | Adversarial suites and independent arithmetic/generation checks |
| Full synthetic operator integration | Passed default and real-weight CUDA focused workflow |
| Permitted default/CUDA regression and static checks | PASS: 404 default / 415 CUDA; lint, formatting, compile and whitespace |
| Limits/deferred requirements kept visible | This report, guide and register; all scientific/deployment gates remain OPEN |

## 24. Exact next step

Review the bounded Phase 2C diff and intake instructions, then freeze the accepted
engineering revision under the owner's normal review process. No commit is made by this audit. No further architecture rewrite is justified by this audit.
Under the owner's existing deferral, the later scientific step is an explicitly approved,
consent-backed pilot/SOP validation followed by proper registry intake and owner-defined
study settings. This report does not initiate collection, training, calibration, model
selection or locked-final evaluation. Production Boolean decisions remain blocked.
