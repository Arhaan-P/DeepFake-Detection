# Audit Findings

Findings from a repo-cleanup pass (2026-08-28) that were **deliberately not
fixed** because fixing them would change model behavior, a reported metric, a
checkpoint, or a documented claim. Each is logged here with file/line and a
one-line description so they can be triaged and fixed in a separate,
dedicated pass. Grouped by severity. None of the changes made alongside this
document alter architecture, training, or evaluation logic — see the commit
history for the full diff of what *was* changed.

Most of these were originally diagnosed in `NOTES.md`, which has the full
derivation, evidence, and resolution rationale for each. This file exists to
give them a stable, scannable checklist home; `NOTES.md` is the primary
source of truth for the "why."

---

## Correctness bugs

### 1. Verification-mode logits never consume the trained encoder/temporal stack
**File:** `models/full_pipeline.py`, `GaitDeepfakeDetector.forward()`, lines ~209–247
(see `encode_gait()` lines 168–191 for the encoder call site).

`encode_gait()` runs `GaitEncoder` → `DualPathTemporalModel` and produces
`video_embedding`, which is stored in the output dict but **never reaches the
logits computed in `mode='verification'`**. Those logits are instead computed
from a separate branch: raw per-timestep `diff`/`abs_diff`/`product` of the
*un-encoded* 78-dim `video_features` vs. `claimed_features`, through
`self.diff_conv` → `self.diff_classifier`. `self.identity_verifier`
(`IdentityVerifier`, instantiated at line 107) is constructed but has zero
call sites in `forward()`. `scripts/training/train.py` backpropagates
cross-entropy on the verification-mode logits only, with no triplet/
contrastive loss enabled — so `gait_encoder` (129,600 params),
`temporal_model` (546,304 params), and `identity_verifier` (31,266 params) —
706,170 of the model's 848,614 total params (~83%) — receive no gradient and
affect no prediction in the mode used for every reported result. Only
`diff_conv` + `diff_classifier` (133,058 params, ~15.7%) are on the live
decision path. Full derivation, parameter-count verification, and the
paired-LOOCV ablation confirming this ordering is *correct given the cohort
size* (not a bug to blindly "fix" by wiring the embedding in) are in
`NOTES.md §2.1–2.2`.

### 2. LOOCV feature normalization is transductive
**File:** `scripts/evaluation/evaluate.py` — never passes `feature_stats` into
`GaitDataset`, so z-score statistics are computed from the held-out subject's
own data rather than training-only statistics.

Measured impact: +0.30 AUC points (94.97±2.90 held-out-norm vs. 94.67±3.06
training-norm), p=0.73 paired across 13 folds — not significant, but it is a
real protocol deviation from what the normalization prose implies. See
`NOTES.md §1.6`.

### 3. `JointImportanceAnalyzer` computes gradient×input, not Grad-CAM
**File:** `utils/gradcam.py:199-218` — the method is gradient×input
attribution, not Grad-CAM, despite the class/module naming. The gradient path
does run through `diff_conv` (the actual decision function), so the
attribution results themselves are sound — this is a naming/terminology
issue, not a validity issue. See `NOTES.md §2.3`.

---

## Doc-vs-code mismatches

### 4. README architecture diagram contradicts the actual decision path
**File:** `README.md`, "Architecture" section (~lines 35–62). Shows
`GaitEncoder → DualPathTemporalModel → IdentityVerifier (Siamese) → decision`
as the verification path. Per finding #1, the actual trained decision
function is the raw-difference CNN (`diff_conv`/`diff_classifier`); the
encoder/temporal/verifier stack is an auxiliary branch not on the decision
path for any reported result.

### 5. README headline LOOCV metrics don't match the committed results JSON
**File:** `README.md`, "Results" section (~lines 13–19) vs.
`outputs/evaluation/loocv/loocv_results.json`.

| Metric | README | JSON |
|---|---|---|
| ROC-AUC | 94.95 ± 2.81 | 95.10 ± 3.08 per-fold (94.95 pooled) |
| Accuracy | 87.27 ± 3.76 | 87.04 ± 3.65 |
| F1 | 86.56 ± 4.56 | 87.12 ± 3.77 |
| EER | 13.19 ± 4.21 | 12.27 ± 3.80 per-fold (12.77 pooled) |

The pooled AUC (94.95) is correct; its paired std (±2.81) matches nothing
reproducible. Full numeric derivation in `NOTES.md §1.1`, which also flags
that the **published IEEE DataPort abstract** (DOI 10.21227/ngh5-b637) carries
the same stale numbers.

### 6. README ablation table reports the superseded, methodologically-flawed experiment
**File:** `README.md`, "Ablation Study" section (~lines 21–28). Reports the
original 4-variant % accuracy table. Per `NOTES.md §2.2`, that experiment (v1)
had the same architecture bug as finding #1 — every variant shared identical
133,058 trainable parameters — so the reported 88.93–90.51% spread was
run-to-run variance, not component contribution. A corrected, definitive
paired-LOOCV ablation exists (`scripts/evaluation/ablation_loocv.py`,
results consumed by `figures/fig9_ablation.png`) with the opposite
conclusion: adding encoder capacity *hurts* at this cohort size. README was
not updated to avoid pre-empting how you want to reconcile it with the paper.

### 7. README links two files that don't exist in the repo
**File:** `README.md` line 9 (`[LITERATURE_REVIEW.md](LITERATURE_REVIEW.md)`)
and the "Project Structure" section (lists `PLAN.md` and
`LITERATURE_REVIEW.md`). Neither file is tracked in git or present in the
working tree. `.github/instructions/rules.instructions.md` also references
`PLAN.md` as required reading. Unclear whether these were removed
deliberately (content folded into `paper.tex`/`references.bib`) or lost —
worth confirming before deciding whether to restore, stub, or deregister them.

### 8. IEEE DataPort abstract has a second documented mismatch: augmentation list
Per `NOTES.md §1.5`, the DataPort abstract describes "temporal jitter,
Gaussian noise injection on keypoints, speed perturbation, horizontal
flipping, and occlusion simulation." The actual 1,056 files in
`data/augmented_videos/` (not tracked in git, but this is a claims issue
regardless) carry 15 different suffixes, none of which include occlusion
simulation or keypoint-space noise; `apply_temporal_jitter` exists in
`augment_videos.py` but isn't among the 16 shipped outputs.

---

## Stale references

### 9. README Usage step 7 points at the superseded ablation script
**File:** `README.md` line 162 (`python scripts/evaluation/ablation_study.py`).
Still runs, still produces output — but per finding #6, its methodology is
superseded by `scripts/evaluation/ablation_loocv.py` (untracked as of this
audit; user has since finalized it). README's numbered Usage list has no
mention of `ablation_loocv.py` anywhere.

### 10. `run_pipeline.py`'s augmentation stage passes flags `augment_videos.py` doesn't have
**File:** `scripts/run_pipeline.py`, `stage_augment()` (~lines 73–85). Calls
`augment_videos.py --input_dir <videos_dir> --output_dir <output_dir>`, but
`augment_videos.py` has no `argparse` at all — it hardcodes
`INPUT_DIR = "data/videos"` and `OUTPUT_DIR = "data/augmented_videos"` in its
`__main__` block (confirmed: no `add_argument` calls anywhere in the file).
The passed flags are silently ignored; the stage always reads/writes the
hardcoded paths regardless of `--videos_dir`/`--augmented_dir` given to
`run_pipeline.py`. Not fixed here (touches a script's actual subprocess call,
slightly outside the narrower "fix stale README commands" scope this pass
was authorized for), but it's the identical category of bug as finding #9 —
worth fixing in the same pass as adding real CLI args to `augment_videos.py`
and correcting this call site together.

### 11. `extract_faces.py` isn't documented in README's Usage steps
**File:** `scripts/preprocessing/extract_faces.py` exists to produce
FaceFusion source-face images, which README's step 10 (manual FaceFusion GUI
process) implicitly depends on but never names as a prerequisite script.

### 12. `figures/README.md` says figures 1–4 are "Missing"
**File:** `figures/README.md` lines 26–36 state the four conceptual diagrams
(`fig1_pipeline.png` … `fig4_architecture.png`) still need to be generated.
They are in fact already tracked in git under `diagrams/` (not `figures/`) —
either they need to be copied/symlinked into `figures/` (if that's what
`paper.tex` actually reads from) or this doc note is simply stale. Not
changed here since it borders on describing the paper's figure pipeline.

---

## Structural drift

### 13. README's "Project Structure" tree omits several real top-level directories
`ieee_scripts/`, `diagrams/`, `figures/`, `scripts/generate_figures/`,
`AUDIT_FINDINGS.md`, `NOTES.md`, and `context.md` all exist and are tracked,
but aren't in README's tree. Conversely it lists `PLAN.md` and
`LITERATURE_REVIEW.md`, which don't exist (finding #7).

---

## Unused code left in place (not deleted — see rationale)

Per the hard constraint on not altering anything under `models/`, the
following confirmed-unused classes were **left in place** rather than
removed, even though grep confirms zero call sites outside their own defining
file:

- `models/identity_verifier.py`: `GaitComparisonNetwork`, `ContrastiveLossNetwork`
- `models/full_pipeline.py`: `GaitDeepfakeDetectorWithTriplet` (and, transitively, `TripletLossNetwork`, which only that class uses)
- `models/gait_encoder.py`: `MultiScaleGaitEncoder` (reachable via `train.py --multi_scale`, but not the default and not used for any reported result)
- `models/temporal_model.py`: `TemporalAttentionPool` (defined, exported, zero call sites)

`utils/pose_extraction_gpu.py` (`MoveNetExtractor`) is likewise left in place
— reachable via `preprocess_videos.py --use_gpu`, explicitly discouraged in
`.github/instructions/rules.instructions.md` ("worse gait accuracy... exists
for reference only"), and not used for any reported result. Its docstring now
carries a stronger disclaimer, but the code itself is untouched.

`scripts/evaluation/_verify_ablation_fix.py` — a standalone dev-time
assertion script proving the v1→v2 ablation architecture fix — has no
external callers and isn't referenced in README or the rules file. Left in
place as provenance for the fix described in `NOTES.md §2.2`; consider moving
to a `dev/` or `tools/` directory if you want it out of the main script tree.

`Required for Research Paper/` (local working tree only, not tracked in git —
already excluded via `.gitignore`) contains an older, explicitly superseded
figure/abstract set per `NOTES.md`. Not deleted since it isn't a git-tracked
concern and deleting local files wasn't something to do unprompted.

---

## Lint findings left unfixed

`ruff check` on `models/`, `utils/`, `scripts/`, `ieee_scripts/`, `tests/`
found 75 issues; 44 (extraneous f-string prefixes, unused `except ... as e`
bindings) were fixed as pure formatting. 31 remain, deliberately unfixed
because each would require a real code edit, not just an import/whitespace
change:

- **`E402` (import not at top of file)** in `_verify_ablation_fix.py`,
  `verify_gait_preservation.py`, `run_pipeline.py` — in each case the late
  import is deliberate (stderr/stdout suppression around a noisy C++ import,
  `os.chdir()` before importing something path-relative, or a `sys.modules`
  stub injected before importing the module that needs it). Reordering would
  change behavior, not just style.
- **`E722` (bare `except:`)** in `ablation_study.py:417` and
  `preprocess_videos.py:118` — narrowing to `except Exception:` changes what
  gets caught (e.g. `KeyboardInterrupt`), which is a real (if usually
  harmless) behavior change to error-handling scope in an evaluation script.
- **`F841` (assigned but unused local)** — 15 instances across
  `generate_paper_figures.py`, `visualize_keypoints.py`,
  `augment_videos.py`, `pose_extraction.py`, `pose_extraction_gpu.py`. Some
  (`bars1 = ax.bar(...)`, `cases = plot_four_cases(...)`, `tables =
  generate_tables(...)`) have side-effecting right-hand sides (plotting,
  file writes) where only the variable binding is unused — safe to fix but
  requires editing each call site individually rather than a blanket rule.
  Others (`T = len(...)`, `rng = np.random.RandomState(42)`) look like
  leftover locals from a prior version of the function and may be worth a
  closer read before removing, in case they're evidence of an incomplete
  refactor rather than truly inert.

## Test coverage gap

No unit tests exist anywhere in the repo (confirmed: no `tests/` directory,
no `test_*.py` files, prior to this pass). A smoke test was added
(`tests/smoke_test.py`) that verifies the model builds and runs a forward
pass in both `classification` and `verification` modes without crashing, on
synthetic data — this is a coverage floor, not a correctness test. It
intentionally does **not** assert anything about finding #1 either way, so it
will keep passing whether or not that bug is ever fixed. Real unit tests
(feature extraction shape/range checks, data loader balance invariants, loss
computation, checkpoint save/load round-trips) are a gap worth a dedicated
pass.

---

## Privacy note (informational, not a defect)

`outputs/evaluation/deepfake_test/*.json` is excluded from git via
`.gitignore` (`outputs/evaluation/`). Per `NOTES.md`'s resolution note it was
independently anonymized to "Subject N" labels via the same mapping used for
publication figures. Since it's outside git's scope, this audit didn't touch
it — flagging only so that if this file is ever distributed outside the git
repo (e.g. as supplementary material), its anonymization gets verified
separately.
