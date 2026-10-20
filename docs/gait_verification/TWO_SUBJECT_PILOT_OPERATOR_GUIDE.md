# Two-subject pilot — operator guide

20 October 2026. Evidence class `TWO_SUBJECT_PILOT_DIAGNOSTIC`.

**TWO-PERSON EXPLORATORY PILOT — NOT A VALIDATED BIOMETRIC PERFORMANCE RESULT.**
This workflow checks whether the software can ingest, process, enroll and score real
recordings. Two people cannot establish recognition accuracy, error rates, a threshold
or a winning model. Every successful score is `POLICY_NOT_READY` with a null verdict.
Raw scores exist only in restricted `outcomes.jsonl` files under the work directory.

All commands run from the repository root:

```bash
PILOT=".venv-gait-v2-cuda/bin/python scripts/gait_verification/two_subject_pilot.py"
P=data/gait_verification/pilot_2subjects          # Git-ignored private data root
W=outputs/gait_verification/pilot_2subjects/work  # Git-ignored private artifacts
```

## 1. What you record

Two consenting participants × four views × two independently recorded takes, ~30 s each:
16 original videos. Views: `left45` (camera at the walker's front-left), `right45`
(front-right), `toward` (walker approaches the camera), `away` (walker moves away).
Set 1 and Set 2 are separate recordings, not clips cut from one file. Follow
[`CAPTURE_SOP_DRAFT.md`](CAPTURE_SOP_DRAFT.md) for framing, safety and metadata.

## 2. Folder layout (convenience only)

```text
data/gait_verification/pilot_2subjects/
    intake.json                 # your filled intake sheet (private)
    originals/S001/left45/set1.mp4   set2.mp4
    originals/S001/right45/set1.mp4  set2.mp4
    originals/S001/toward/set1.mp4   set2.mp4
    originals/S001/away/set1.mp4     set2.mp4
    originals/S002/...               (same eight files)
```

The folder names are **not evidence** of subject, view, take or session. Only the
intake sheet's explicit, operator-confirmed rows are used. Copy the phone files
unchanged (no trimming, re-encoding or cloud "enhancement"). If your files are `.mov`
or named differently, edit `source_file_uri` in the sheet instead of renaming evidence.

## 3. Create the intake sheet and config

```bash
$PILOT init --subjects S001 S002 --media-kind REAL_PARTICIPANT_RECORDINGS \
  --data-root $P --work-dir $W --registry $P/registry.json \
  --intake-out $P/intake.json --config-out $P/pilot.json
```

`init` writes two new files and never overwrites. Tracked references with the same
content: [`configs/gait_verification/two_subject_pilot.example.json`](../../configs/gait_verification/two_subject_pilot.example.json)
and [`two_subject_pilot_intake.template.json`](../../configs/gait_verification/two_subject_pilot_intake.template.json).

Fill every row in `intake.json` from real records. Never guess.

| Field | Rule |
|---|---|
| `session_id` | A real recording occasion. If Set 1 and Set 2 were filmed on the same occasion, use the **same** session ID. Different file IDs do not mean different days. |
| `date` | Actual recording time with UTC offset, e.g. `2026-11-03T10:15:00+05:30`. If the phone time is unreliable, use your written log. File modification times are not proof. |
| `consent_ref` | A pointer to the consent record kept outside Git. Missing consent blocks registration. Keep names/consent documents out of the repository. |
| `viewing_angle_definition`, `walking_direction`, `camera_motion_mode` | Required. Describe the actual geometry (e.g. "front-left, ~45° at the route midpoint, stationary"). |
| `phone_model`, `camera_lens`, `camera_settings`, `capture_stabilization_mode`, `approximate_subject_distance_m`, `operator_id_pseudonym`, `recording_notes` | Fill where known; write unknown values as `"unknown"` in notes rather than inventing them. |
| `quality_review_status` | `accepted` only after you watched the take (full body, both shoes visible). Anything else blocks registration. |
| `sha256`, `duration_sec`, `resolution`, `nominal_fps` | Leave `null`: they are measured. A value you supply must match the measurement. |
| `operator_confirmed`, `consent_confirmed`, `independent_original_confirmed` | Set `true` only when personally verified for that row. |
| `split_role` | Must stay `development`. Pilot people never enter a locked-final group. |

Optional settings in `pilot.json`: `variants` (`vidstab-s10`, `vidstab-s15`), `augmentations`,
`set_swap`, `preview_assets`, `preview_flagged_frames`, `models`, `runtime` checkpoint paths.

## 4. Validate, register, plan

```bash
$PILOT validate --config $P/pilot.json
```

Every command also accepts `--out NEW_FILE.json` (parent directory must exist; files are
never overwritten) to save the complete private result.

Exit 0 = all 16 rows valid. Exit 2 lists `missing_originals`, `missing_matrix_rows`,
`duplicate_sources`, `errors` and per-row `row_issues` (e.g. `MISSING_METADATA:date`,
`REQUIRES_EXPLICIT_consent_confirmed`, `SOURCE_SHA256_MISMATCH`). Each video is fully
decoded; FPS, frame timestamps, dimensions, rotation and duration come from ffprobe.

```bash
$PILOT register --config $P/pilot.json      # writes registry.json once (capture-registry-v1)
$PILOT plan --config $P/pilot.json          # read-only: creates and loads nothing
```

Registration is all-or-nothing. If you later edit the intake sheet, every stage refuses
to continue; register into a **new** registry path (edit `registry` in the config).

## 5. Run everything

```bash
$PILOT all --config $P/pilot.json            # primary Set 1 -> Set 2 experiment
$PILOT all --config $P/pilot.json --set-swap # also the separate Set 2 -> Set 1 experiment
```

`all` runs exactly the stages below with the same checks. Run one GPU workflow at a time.

| Stage | Command | What it does |
|---|---|---|
| Augment | `$PILOT augment --config $P/pilot.json` | Creates the configured derivatives under `$P/derived/<variant>/<asset>/`, with FFmpeg commands, version, input/output SHA-256 and frame-timing checks. RAW is never modified. |
| Process | `$PILOT process --config $P/pilot.json` | RTMPose pose + provisional quality for originals **and** derivatives; frozen GaitPT/GPGait/GPGait++ GREW/CASIA-B embeddings and native E0 MediaPipe 60×78 features for **originals only**. CUDA only, no CPU fallback. |
| Enroll | `$PILOT enroll --config $P/pilot.json [--direction set2-to-set1]` | Enrolls the reference set's four RAW originals per subject per model into `$W/enrollments/<direction>/`. |
| Trials | `$PILOT trials --config $P/pilot.json [--direction …]` | 16 claims: each query take claims its own subject (genuine) and the other subject (wrong claim). References exclude the probe's view (`exclude_probe_view`): 3 independent originals per claim. |
| Score | `$PILOT score --config $P/pilot.json [--direction …]` | Claim-only scoring for every model. The scorer receives only the query take and the claimed subject's reference takes. |
| Previews | `$PILOT previews --config $P/pilot.json` | Private overlay JPEGs for `preview_assets` (start/middle/end + flagged frames). |
| Report | `$PILOT report --config $P/pilot.json` | New `$W/pilot/reports/<id>/report.json` and `summary.md`. |

### Augmentations (disabled unless listed)

```json
"augmentations": [
  {"kind": "brightness", "value": 0.05},
  {"kind": "contrast", "value": 1.10},
  {"kind": "blur", "value": 0.6},
  {"kind": "compression", "value": 28}
]
```

FFmpeg `eq=brightness` in [-0.1, 0.1], `eq=contrast` in [0.85, 1.15], `gblur=sigma`
in [0.1, 1.0], libx264 CRF in [23, 32]. Each is a single transformation of the RAW
original with an ID derived from its exact parameters; composition, flips, speed
change, reversal, cropping and perspective warps are refused. These are robustness
conditions, not improvements: they can make pose quality worse.

### Before/after previews

List an original and its derivative to compare the same frame indices, e.g.
`"preview_assets": ["S001_toward_set2", "S001_toward_set2.vidstab-s10"]`. Previews show
real frames (faces included): keep them private, never upload.

## 6. Reading the outputs

- `$W/pilot/reports/<id>/summary.md` — plain-language summary: accepted originals,
  recordings needing review/recapture, flagged views, transformation effects on
  measured pose diagnostics, enrollment availability, claim-scoring availability.
- `report.json` sections: `intake`, `augmentation`, `quality` (per participant/view/set/variant:
  pose gate, windows, missing joints, ankle visibility, ambiguity, tracking breaks,
  `recommend_manual_review`, `delta_from_raw`), `enrollment`, `verification` (counts,
  status/reason breakdown by claim class, by view, by set, by subject; no scores),
  `verification_set_swap`, `matched_variant_scoring`, `previews`.
- `$W/pilot/runs/<direction>/<model>-<id>/outcomes.jsonl` — **RESTRICTED** raw scores.

A quality failure is `INSUFFICIENT_GAIT` (recapture), not evidence against a claim.
`CLAIM_NOT_ENROLLED/ENROLLMENT_INCOMPLETE` means a reference take failed processing.
Derivatives cannot be scored against RAW references (`RECIPE_MISMATCH`); they get
pose/quality analysis only.

## 7. Retry and resume

Rerun the same command. Valid pose/quality/embedding/E0 artifacts are reused
(`cache_reused` counts them), enrollment skips stored references without loading a
model, existing trial manifests must regenerate identically, and every report or run
is a new file. Transient GPU failures (`GPU_BUSY`, `CUDA_UNAVAILABLE_NO_CPU_FALLBACK`,
out-of-memory) are retried on rerun. A changed original stops every stage with a
source-check error: restore the original bytes. Ctrl-C stops processing cleanly.
For failure details see `$W/runs/<run_id>.json` and `$W/ledger.jsonl` (Phase 2 guide §5).

## 8. Limits

Single-occasion takes support cross-view/different-take engineering checks only, not
cross-day robustness. Quality thresholds are provisional. E0 is the historical shipped
head (not the absent controlled LOSO model) with a weaker MediaPipe gate than RTMPose.
Pilot results must never be reported as study performance or used to choose a model,
threshold or stabilization policy. See the
[implementation report](TWO_SUBJECT_PILOT_IMPLEMENTATION_REPORT.md) and
[pilot task register](TWO_SUBJECT_PILOT_TODO.md).
