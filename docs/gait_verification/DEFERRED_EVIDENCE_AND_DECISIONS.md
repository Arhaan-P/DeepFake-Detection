# Gait Verification V2 — Deferred evidence and decisions register

**Created:** 12 October 2026 (start of Phase 2). **Branch:** `feat/gait-verification-v2`.

**Owner decision (12 October 2026).** Physical data collection and all empirical scientific
validation are **intentionally deferred by the project owner** until the complete software
architecture has been built. Engineering proceeds using clearly labeled synthetic or generated
test fixtures. Scientific and deployment readiness remain **blocked**.

This file is the authoritative checklist of everything that is *not* done and *cannot* be
claimed. Rules:

1. **No item is complete because a software interface exists.** An item is `DONE` only when the
   listed evidence exists, was produced under an approved protocol, and is recorded here with a
   pointer to it.
2. Statuses: `OPEN` (nothing done), `ENGINEERING_READY` (software can receive the evidence, but
   the evidence does not exist), `DONE` (evidence exists and is linked).
3. "Blocks" says what the open item prevents: **ENG** (engineering), **SCI** (scientific
   validation), **DEP** (deployment).
4. Nothing here authorizes a choice. Values are deliberately unset in code and config.
5. The project is **not** biometrically validated. No accuracy, threshold or model ranking exists.

Categories: `TODO_EXTERNAL` (physical collection, third parties, permissions, real-world evidence),
`TODO_OWNER_DECISION` (a later explicit owner decision), `TODO_SCIENTIFIC` (needs valid empirical
data and an approved evaluation), `TODO_SECURITY` (security, privacy, licensing before deployment),
`IMPLEMENTABLE_NOW` (engineering that proceeds immediately).

---

## TODO_EXTERNAL

| ID | Description | Status | Why deferred | Evidence required | Prerequisite | Blocks |
|---|---|---|---|---|---|---|
| EXT-01 | Record a 2–3 volunteer capture pilot | OPEN | Owner deferred collection until software architecture is complete | Registered originals from 2–3 consenting volunteers, with real metadata | Phase 2 engineering complete; EXT-03 | SCI |
| EXT-02 | Capture ~30 s independent originals in left45, right45, toward and away views | OPEN | Same as EXT-01 | Original files per view, checksummed, probed (real duration, FPS, size) | EXT-01 | SCI |
| EXT-03 | Obtain valid consent and real capture metadata | OPEN | No participants exist; consent is never inferred from files | Consent references held in restricted storage, real ISO dates with UTC offset, real session IDs, operator codes | Supervisor/ethics arrangement (SEC-02) | SCI, DEP |
| EXT-04 | Verify actual phone FPS, operator movement and camera geometry | OPEN | Needs real footage; menu settings are not proof of constant FPS | Measured timestamps (ffprobe), measured offsets/heights/distances, start/middle/end geometry notes | EXT-01 | SCI |
| EXT-05 | Evaluate the draft recording SOP on real moving-camera footage | OPEN | No real footage | Pilot review result per `CAPTURE_SOP_DRAFT.md`, recorded deviations, revised SOP | EXT-02 | SCI |
| EXT-06 | Collect the planned ~40-participant, 160-original cohort | OPEN | Depends on the pilot and the SOP approval | 160 registered originals, 4 independent per participant | EXT-05, OWN-10 | SCI |
| EXT-07 | Record whether separate filming days or sessions are available | OPEN | Unknown; session independence is never inferred | Recorded plan stating real session IDs/dates per participant, or an explicit statement that all takes are single-session | EXT-01 | SCI (cross-session claims only) |
| EXT-08 | Obtain access to official evaluation samples where permitted and necessary | OPEN | Third-party access and licensing | Written permission/licence terms and a dated, checksummed copy | SEC-01 | SCI |

## TODO_OWNER_DECISION

| ID | Description | Status | Why deferred | Evidence required | Prerequisite | Blocks |
|---|---|---|---|---|---|---|
| OWN-01 | Final subject split counts | OPEN (unset in code/config) | Cohort does not exist; no default may be assumed (24/8/8 is **not** chosen) | Owner decision recorded in the PRD with date | Cohort size known (EXT-06) | SCI |
| OWN-02 | Split random seed | OPEN | Must not be chosen by an engineer or from results | Owner decision recorded in the PRD | OWN-01 | SCI |
| OWN-03 | Number of impostor claims per query | OPEN | Affects statistical power and FAR resolution; owner policy | Owner decision recorded in the PRD | OWN-01 | SCI |
| OWN-04 | Trial-generation random seed | OPEN | Same as OWN-02 | Owner decision recorded in the PRD | OWN-03 | SCI |
| OWN-05 | Target FAR operating point(s) | OPEN | Application risk decision | Owner-declared FAR target(s) | OWN-01 | SCI, DEP |
| OWN-06 | Whether `exclude_probe_view` remains the primary reference rule | OPEN | Needs pilot evidence about cross-view behaviour | Owner decision recorded in the PRD | EXT-02 | SCI |
| OWN-07 | Same-view versus cross-view reference comparisons | OPEN | Requires extra independent same-view takes to exist | Owner decision and, if same-view, capture plan | EXT-07 | SCI |
| OWN-08 | RAW versus stabilized model-recipe policy | OPEN | Must be evidence-based (SCI-03); RAW stays the control | Owner decision after SCI-03 | SCI-03 | SCI |
| OWN-09 | Final oblique-camera capture geometry (left45/right45 definition, static vs moving) | OPEN | Needs pilot comparison | Owner decision recorded in the SOP | EXT-05 | SCI |
| OWN-10 | Final capture SOP approval | OPEN | SOP is a draft | Signed-off SOP version | EXT-05, OWN-09 | SCI |
| OWN-11 | E0 retraining authorization | OPEN | Not authorized; historical E0 stays untouched | Explicit owner authorization naming split and protocol | OWN-01, OWN-02 | SCI |
| OWN-12 | Optional adaptation/training authorization | OPEN | Not authorized; models are frozen | Explicit owner authorization naming data, budget, protocol | OWN-11 | SCI |

## TODO_SCIENTIFIC

| ID | Description | Status | Why deferred | Evidence required | Prerequisite | Blocks |
|---|---|---|---|---|---|---|
| SCI-01 | Validate pose quality on the actual pilot | OPEN | No pilot footage; Phase 1A thresholds are provisional | Overlay review plus measured quality statistics from the pilot | EXT-02 | SCI |
| SCI-02 | Validate provisional temporal-window quality settings | OPEN | Same | Documented acceptance of or revision to the window gate with data | SCI-01 | SCI |
| SCI-03 | Select stabilization (RAW/S10/S15) based on evidence | OPEN | RAW is the control; no variant is superior by assumption | Development-split comparison under an approved protocol | SCI-01, OWN-08 | SCI |
| SCI-04 | Measure real genuine/impostor score distributions | OPEN | No real trials | Restricted offline run on a locked development split | OWN-01..04, EXT-06 | SCI |
| SCI-05 | Compare models on a common, leakage-safe trial manifest | OPEN | Same | `compare_runs` output on identical trial IDs; CASIA-B reported separately | SCI-04 | SCI |
| SCI-06 | Establish development-derived thresholds and calibration (if authorized) | OPEN | Not authorized; no thresholds exist | Authorized calibration run on the calibration split only | SCI-04, OWN-05 | SCI, DEP |
| SCI-07 | Estimate ROC-AUC, EER, FAR, FRR, TAR and coverage on real data | OPEN | No real data | Reports with subject-cluster uncertainty | SCI-04 | SCI |
| SCI-08 | Select a model and scoring recipe from development evidence | OPEN | No winner is chosen; synthetic fixtures say nothing | Written selection rationale from development evidence | SCI-05, SCI-07 | SCI, DEP |
| SCI-09 | Perform a properly authorized locked-final evaluation | OPEN | Locked-final is protected and untouched | `LockedTestAuthorization` bound to the frozen bundle SHA-256 | SCI-08, SCI-06 | SCI, DEP |
| SCI-10 | Cross-session/cross-day robustness (only if the protocol provides it) | OPEN | Depends on EXT-07 | Real multi-session recordings with true dates | EXT-07 | SCI (cross-session claims only) |

## TODO_SECURITY

| ID | Description | Status | Why deferred | Evidence required | Prerequisite | Blocks |
|---|---|---|---|---|---|---|
| SEC-01 | Review model and checkpoint licenses (GaitPT, GPGait, GPGait++, RTMPose, GREW/CASIA-B terms) | OPEN | Needs a legal/licence decision beyond engineering | Written licence review per checkpoint and dataset | None | DEP |
| SEC-02 | Define biometric-media and enrollment retention/deletion policy | OPEN | Needs owner/supervisor/ethics input | Approved retention schedule and deletion procedure | None | SCI, DEP |
| SEC-03 | Evaluate encrypted storage and access auditing (stores are filesystem-permission-only today) | OPEN | Key management needs an owner decision | Design decision, implementation and test of encryption-at-rest and an access log | SEC-02 | DEP |
| SEC-04 | Protect the participant-to-pseudonym mapping | OPEN | Mapping must live outside the repo; custodian not decided | Named custodian, storage location, access rule | EXT-03 | SCI, DEP |
| SEC-05 | Obtain deployment authorization if deployment is later intended | OPEN | No deployment intended now | Written authorization | SEC-01..04, SCI-09 | DEP |

## IMPLEMENTABLE_NOW (Phase 2 engineering; proceeds immediately)

These are engineering tasks that need no new recordings. `DONE` here means the *software* exists
and is tested with synthetic fixtures. It does not satisfy any item above.

| ID | Description | Status | Engineering evidence |
|---|---|---|---|
| IMP-01 | Phase 2 gap assessment of the Phase 1D dataset path | DONE | [Phase 2 report](PHASE_2_DATASET_INTEGRATION_REPORT.md) §1–6, §12 |
| IMP-02 | Registry-driven dataset ingestion/validation report (SHA-256, ffprobe, metadata, lineage, duplicates) | DONE | [Phase 2 report](PHASE_2_DATASET_INTEGRATION_REPORT.md) §7–8; ingestion/CLI integration tests |
| IMP-03 | Deterministic dry-run processing plan (no mutation, no locking) | DONE | [Phase 2 report](PHASE_2_DATASET_INTEGRATION_REPORT.md) §9–10; dry-run and split-guard regressions |
| IMP-04 | Resumable, integrity-checked pose/embedding processing with per-asset states | DONE | [Phase 2 report](PHASE_2_DATASET_INTEGRATION_REPORT.md) §11–13; interruption, corruption, loader and ledger regressions |
| IMP-05 | Cache fingerprinting and invalidation (checkpoint, pose recipe, window policy, variant) | DONE | [Phase 2 report](PHASE_2_DATASET_INTEGRATION_REPORT.md) §11; stale-lineage and recipe tests |
| IMP-06 | Bounded-CUDA resource management without silent CPU fallback | DONE | [Phase 2 report](PHASE_2_DATASET_INTEGRATION_REPORT.md) §14, §18–19; opt-in tests for all four encoders |
| IMP-07 | Per-original capture-quality report keeping media / metadata / pose-gate / suitability separate | DONE | [Phase 2 report](PHASE_2_DATASET_INTEGRATION_REPORT.md) §7, §15; quality/coverage tests |
| IMP-08 | Explicit-selection derivative (RAW/S10/S15) processing with provenance | DONE | [Phase 2 report](PHASE_2_DATASET_INTEGRATION_REPORT.md) §16; real vid.stab, ancestry and protected-role tests |
| IMP-09 | Synthetic fixture dataset and Phase 2 regression tests | DONE | [Phase 2 report](PHASE_2_DATASET_INTEGRATION_REPORT.md) §19; 60 original Phase 2 cases plus 36 continuation cases |
| IMP-10 | Operator/developer usage instructions and Phase 2 report | DONE | [Phase 2 report](PHASE_2_DATASET_INTEGRATION_REPORT.md) §17–23; reconciled operator guide and CLI smoke |

---

## Change log

- 2026-10-12: Register created at the start of Phase 2. All deferred items `OPEN`.

- 2026-10-12: Codex recovered Claude's uncommitted Phase 2 work, reproduced Phase 1D, corrected integrity/recovery/provenance defects and completed verification/documentation. IMP-01…IMP-10 are DONE with engineering evidence above. All EXT/OWN/SCI/SEC items remain OPEN; no scientific or deployment gate is satisfied.

## IMPLEMENTABLE_NOW — Phase 2B engineering (12 October 2026)

Added before implementation. These tasks do not satisfy scientific or deployment gates.

| ID | Description | Status | Engineering evidence |
|---|---|---|---|
| E0-01 | Historical inference and checkpoint audit | DONE | [Phase 2B report](PHASE_2B_E0_INTEGRATION_REPORT.md) §4–6; actual source/ZIP/tensor audit; controlled bundle absence documented |
| E0-02 | Native 60×78 feature-contract validation | DONE | Report §5, §16–17; historical feature parity, joint ordering, heel/toe and finite-input tests |
| E0-03 | Frozen E0 scorer adapter | DONE | Report §8, §17; real checkpoint CPU/CUDA and managed synthetic scorer tests |
| E0-04 | Safe hash-pinned strict checkpoint loading | DONE | Report §6; restricted extraction, unsafe payload refusal, missing/incompatible state/statistics/config tests |
| E0-05 | Claim-only scoring and reference enforcement | DONE | Report §9–10; exact reference-access spy, no actual-identity request, source/lineage/split/public-output tests |
| E0-06 | Native preprocessing and integrity-checked feature cache | DONE | Report §7; sealed source/recipe/shape/tampering tests, quarantine and resume tests; real MediaPipe engineering smoke |
| E0-07 | Common-trial evaluation orchestration | DONE | Report §11, §15–16; identical request/ID tests and run-common subprocess workflow |
| E0-08 | Cross-system reporting and comparison guards | DONE | Report §12–14; default/narrow-policy refusal tests, CASIA-B separation, distinct score scales and all-primary coverage |
| E0-09 | End-to-end synthetic integration and regression | DONE | Report §16; 61 new default passes + 1 opt-in E0 CUDA; full default 364 passed/10 skipped, CUDA 374 passed; both 2 deselected |
| E0-10 | Phase 2B documentation and handoff | DONE | [Phase 2B report](PHASE_2B_E0_INTEGRATION_REPORT.md), reconciled register, dated PRD/handoff amendments and tested CLI examples |

- 2026-10-12: Phase 2B started; E0-01…E0-10 OPEN. All EXT/OWN/SCI/SEC items remain OPEN.
- 2026-10-12: Phase 2B engineering verified and documented; E0-01…E0-10 DONE with the evidence above. All EXT/OWN/SCI/SEC items remain OPEN. Native/source and frozen inference parity do not establish empirical new-cohort parity or recreate the absent controlled LOSO weights.

## IMPLEMENTABLE_NOW — Phase 2C independent engineering acceptance audit

Opened before source corrections on 12 October 2026 at `f1213c7`, with a clean
`feat/gait-verification-v2` worktree. DONE means inspected and verified engineering
evidence, including documented limitations; it does not close scientific or deployment gates.

| ID | Audit task | Status | Evidence / outcome |
|---|---|---|---|
| AUD-01 | End-to-end contract audit | DONE | [Phase 2C audit](PHASE_2C_ARCHITECTURE_ACCEPTANCE_AUDIT.md) §7, §10, §17: both native paths and complete common-trial workflow |
| AUD-02 | Input and capture-registry audit | DONE | [Phase 2C audit](PHASE_2C_ARCHITECTURE_ACCEPTANCE_AUDIT.md) §6, §12: schema, bytes, duplicate and metadata checks |
| AUD-03 | Source/derivative lineage audit | DONE | [Phase 2C audit](PHASE_2C_ARCHITECTURE_ACCEPTANCE_AUDIT.md) C-01/C-06, §12: ancestry, source substitution and same-original refusal |
| AUD-04 | Split and locked-final protection audit | DONE | [Phase 2C audit](PHASE_2C_ARCHITECTURE_ACCEPTANCE_AUDIT.md) C-01/C-03, §8: protected entry points and locked transitions corrected |
| AUD-05 | Pose, quality and model compatibility audit | DONE | [Phase 2C audit](PHASE_2C_ARCHITECTURE_ACCEPTANCE_AUDIT.md) §14/§16: four pinned frozen CUDA candidates; scientific gates still open |
| AUD-06 | Enrollment and claim-only access audit | DONE | [Phase 2C audit](PHASE_2C_ARCHITECTURE_ACCEPTANCE_AUDIT.md) C-02/C-06/C-12: claim/owner spies, independent references and lazy status precedence |
| AUD-07 | E0 baseline parity and limitation audit | DONE | [Phase 2C audit](PHASE_2C_ARCHITECTURE_ACCEPTANCE_AUDIT.md) §13/§17: historical feature/head parity and fresh native MediaPipe/CUDA smoke; LOSO absent |
| AUD-08 | Cache, checkpoint and artifact integrity audit | DONE | [Phase 2C audit](PHASE_2C_ARCHITECTURE_ACCEPTANCE_AUDIT.md) C-04/C-06/C-08/C-09: seals, restricted loads, CRCs, link refusal, private writes/recovery |
| AUD-09 | Trial generation and statistical evaluation audit | DONE | [Phase 2C audit](PHASE_2C_ARCHITECTURE_ACCEPTANCE_AUDIT.md) C-05/C-07, §15: deterministic v3 policy, metric and paired/evidence validation |
| AUD-10 | CUDA resource/recovery audit | DONE | [Phase 2C audit](PHASE_2C_ARCHITECTURE_ACCEPTANCE_AUDIT.md) C-10/C-14, §16: actual CUDA/busy/OOM/recovery; single-workflow throughput limits |
| AUD-11 | Privacy/security/license readiness audit | DONE | [Phase 2C audit](PHASE_2C_ARCHITECTURE_ACCEPTANCE_AUDIT.md) §11/§21: privacy/permission/ignore review; all SEC licensing/deployment items still OPEN |
| AUD-12 | Complete operator-workflow acceptance tests | DONE | [Phase 2C audit](PHASE_2C_ARCHITECTURE_ACCEPTANCE_AUDIT.md) §17/§18: full default and real-weight CUDA operator workflows with explicit injected observations |
| AUD-13 | Reproducibility and final architecture acceptance report | DONE | [Phase 2C audit](PHASE_2C_ARCHITECTURE_ACCEPTANCE_AUDIT.md) §18–24: 404 default/415 CUDA passes, static/preservation checks and acceptance with limitations |

- 2026-10-12: Phase 2C AUD-01…AUD-13 opened. All deferred EXT/OWN/SCI/SEC items preserved OPEN. No training, calibration, collection or final evaluation authorized.

- 2026-10-13: Phase 2C AUD-01…AUD-13 DONE after independent inspection, adversarial regressions, bounded corrections, both permitted suites and preservation checks. Verdict: ARCHITECTURE ACCEPTED WITH DOCUMENTED ENGINEERING LIMITATIONS. All EXT/OWN/SCI/SEC remain OPEN. No collection, training, fitting, threshold, winner, locked-final study or deployment occurred.
