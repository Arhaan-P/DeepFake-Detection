# Two-subject pilot task register

Opened 20 October 2026 before implementation; reconciled the same day after
engineering completion. Evidence class: `TWO_SUBJECT_PILOT_DIAGNOSTIC`. Only 1:1
claims; successful scores remain `POLICY_NOT_READY`, null verdict. Two people
cannot support scientific validation.

Statuses: ENGINEERING_READY, WAITING_FOR_RECORDINGS, WAITING_FOR_METADATA,
WAITING_FOR_CONSENT_CONFIRMATION, REQUIRES_MANUAL_REVIEW,
COMPLETED_WITH_EVIDENCE, SCIENTIFICALLY_UNVALIDATED.

`COMPLETED_WITH_EVIDENCE` below means **software** evidence on generated media
with explicitly injected observations. No real pilot recording exists:
`data/gait_verification/pilot_2subjects/` is absent.

| Task | Status at opening | Current status | Evidence / acceptance |
|---|---|---|---|
| 16 independent originals (2 subjects × 4 views × 2 takes) | WAITING_FOR_RECORDINGS | WAITING_FOR_RECORDINGS | All 16 files missing; exact paths in the [operator guide](TWO_SUBJECT_PILOT_OPERATOR_GUIDE.md) §2 |
| Real metadata and independent-take confirmation | WAITING_FOR_METADATA | WAITING_FOR_METADATA | Intake refuses missing session/date/geometry/motion/review fields and unconfirmed rows (`test_pilot_intake.py`) |
| Consent confirmation | WAITING_FOR_CONSENT_CONFIRMATION | WAITING_FOR_CONSENT_CONFIRMATION | Missing `consent_ref` or `consent_confirmed` blocks registration; never a warning |
| SHA-256 registry and source validation | REQUIRES_MANUAL_REVIEW | COMPLETED_WITH_EVIDENCE | Measured SHA/ffprobe/full decode, duplicate-byte refusal, capture-registry-v1, source rechecks before every stage |
| Controlled augmentation | REQUIRES_MANUAL_REVIEW | COMPLETED_WITH_EVIDENCE | Existing S10/S15 + opt-in brightness/contrast/blur/CRF; real FFmpeg, lineage, timing, reuse, corruption tests |
| Pose extraction and quality diagnostics | ENGINEERING_READY | COMPLETED_WITH_EVIDENCE | Existing RTMPose/window pipeline over 16 originals + derivatives; per-view/set/variant quality report |
| E0 native features | ENGINEERING_READY | COMPLETED_WITH_EVIDENCE | Existing MediaPipe33/60×78 path; shipped baseline only; real head in opt-in CUDA test |
| Four frozen pretrained embeddings | ENGINEERING_READY | COMPLETED_WITH_EVIDENCE | Existing pinned CUDA paths; real-weight opt-in pilot test; no CPU fallback |
| Set 1 enrollment / Set 2 queries | REQUIRES_MANUAL_REVIEW | COMPLETED_WITH_EVIDENCE | RAW independent originals, one store per direction, cache-only pose reuse |
| Genuine / wrong-claim trials and diagnostic scoring | REQUIRES_MANUAL_REVIEW | COMPLETED_WITH_EVIDENCE | 16 claims per direction, claim-only `ClaimRequest`, deterministic manifest, tamper refusal |
| Cross-view analysis | REQUIRES_MANUAL_REVIEW | COMPLETED_WITH_EVIDENCE | `exclude_probe_view`, three references per claim; per-view status breakdown |
| Per-model coverage reports | REQUIRES_MANUAL_REVIEW | COMPLETED_WITH_EVIDENCE | Every trial counted once; status/reason by claim class, view, set and subject |
| Matched-variant scoring | — | SCIENTIFICALLY_UNVALIDATED | Explicitly RESTRICTED (`RECIPE_MISMATCH`); derivatives get quality analysis only |
| Visual review aids | REQUIRES_MANUAL_REVIEW | COMPLETED_WITH_EVIDENCE | Private 0600 overlays (start/middle/end + flagged frames) for explicit assets |
| Operator guide and reproducible commands | REQUIRES_MANUAL_REVIEW | COMPLETED_WITH_EVIDENCE | CLI tests; [operator guide](TWO_SUBJECT_PILOT_OPERATOR_GUIDE.md) |
| Real pilot run and manual review of real overlays | — | WAITING_FOR_RECORDINGS | Run the guide's commands after copying the 16 originals |
| Scientific validation / SOP / main study | SCIENTIFICALLY_UNVALIDATED | SCIENTIFICALLY_UNVALIDATED | All deferred scientific and deployment tasks remain open |

Remaining scientific restrictions: no accuracy/error rates/ROC/threshold from two
people; no model, stabilization or augmentation choice; pilot people stay
development-only and out of any locked-final group; single-occasion takes are not
cross-day evidence. Details: [implementation report](TWO_SUBJECT_PILOT_IMPLEMENTATION_REPORT.md)
and the central [deferred register](DEFERRED_EVIDENCE_AND_DECISIONS.md) (PIL-01…PIL-07).
