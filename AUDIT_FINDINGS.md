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

**Status (2026-08-29): reviewed, confirmed as-is.** Every configuration that
connects the encoder/temporal stack to the decision path was measured (paired
13-fold × 3-seed LOOCV, `NOTES.md §2.2`) to perform *worse* than the deployed
raw-difference classifier — Raw+CNN −2.53 AUC (p=2.9e-04), Raw+Transformer
−3.88 (p=3.5e-05), Raw+Hybrid −4.22 (p=3.7e-04), Raw+BiLSTM −5.93 (p=1.7e-07),
Hybrid-only (no raw comparison) −43.55 (p=1.1e-15). Wiring the embedding in
would not be a bug fix; it would degrade the reported result. Decision:
ship as-is. The alternative (removing the now-confirmed-unhelpful
encoder/temporal/verifier stack from the model entirely) was considered and
explicitly deferred — that is an architecture change requiring its own
re-training, re-evaluation, and updated paper/checkpoint artifacts, out of
scope for a cleanup pass.

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
the same stale numbers (tracked separately in finding #8; DataPort is an
external platform this repo can't push to).

**Status (2026-08-29): resolved.** README's Results table now reports both
per-fold and pooled statistics matching `loocv_results.json` exactly,
reconciled directly against `paper.tex` Table VII.

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

**Status (2026-08-29): resolved.** README's Ablation section now reports the
paired LOOCV ablation table from `paper.tex` Table IX, with the old 4-variant
accuracy table removed. `ablation_study.py` is explicitly labelled superseded
in both README and its own module docstring.

### 7. README links two files that don't exist in the repo
**File:** `README.md` line 9 (`[LITERATURE_REVIEW.md](LITERATURE_REVIEW.md)`)
and the "Project Structure" section (lists `PLAN.md` and
`LITERATURE_REVIEW.md`). Neither file is tracked in git or present in the
working tree. `.github/instructions/rules.instructions.md` also references
`PLAN.md` as required reading. Unclear whether these were removed
deliberately (content folded into `paper.tex`/`references.bib`) or lost —
worth confirming before deciding whether to restore, stub, or deregister them.

**Status (2026-08-29): resolved for README.** The dead link and both
nonexistent files were removed from README's "Key Idea" section and Project
Structure tree; the novelty claim there now uses the paper's own "under-explored
niche" positioning (Section II-D) instead of the removed literature-review
doc. `rules.instructions.md`'s reference to `PLAN.md` is a separate,
lower-traffic file (Copilot/agent instructions, not user-facing docs) and was
left as-is — flag if you want it cleaned up too.

### 8. IEEE DataPort abstract has stale metrics and a wrong augmentation list
Per `NOTES.md §1.1/§1.5` and `DOCUMENTATION/DATASET.txt` (a local copy of the
live listing text), the published abstract at DOI 10.21227/ngh5-b637 states:
"achieving an AUC-ROC of 94.95% ± 2.81% and an F1 score of 86.56% under
LOOCV" (stale, see finding #5) and describes the augmentation pipeline as
"temporal jitter, Gaussian noise injection on keypoints, speed perturbation,
horizontal flipping, and occlusion simulation" — none of which matches
`augment_videos.py`'s actual 15 named operations (Table II in `paper.tex`);
there is no occlusion simulation and no keypoint-space noise, and
`apply_temporal_jitter` exists in code but isn't among the 16 shipped video
variants (variant #1 is the unaugmented original).

**Status (2026-08-29): text drafted, not pushed** — this repo has no access
to the external IEEE DataPort platform. Corrected replacement text for the
two stale paragraphs (matching `DOCUMENTATION/DATASET.txt`'s current
structure), ready to paste into the DataPort abstract editor:

> Through a 16× data augmentation pipeline — horizontal flip, Gaussian blur,
> brightness adjustment (up and down), contrast increase, colour jitter,
> combined multi-augmentation, grayscale conversion, rotation (left and
> right, capped at 10°), speed perturbation (0.8× and 1.2×), temporal
> reversal, zoom, and Gaussian noise injection on pixels — the dataset
> expands to 1,056 augmented video samples.
>
> This dataset was used to train and evaluate a difference-based temporal
> convolutional verification network (with an auxiliary CNN+BiLSTM+Transformer
> embedding branch used for enrolment diagnostics, not the verification
> decision), achieving a pooled ROC-AUC of 94.95% (per-fold 95.10% ± 3.08%),
> accuracy of 87.04% ± 3.65%, and an F1 score of 87.12% ± 3.77% under 13-fold
> leave-one-subject-out cross-validation.

The "filling a confirmed research gap" phrase in the same paragraph should
also be softened to match the paper's "under-explored niche" framing (see
finding #7) if the abstract is revised at all.

---

## Stale references

### 9. README Usage step 7 points at the superseded ablation script
**File:** `README.md` line 162 (`python scripts/evaluation/ablation_study.py`).
Still runs, still produces output — but per finding #6, its methodology is
superseded by `scripts/evaluation/ablation_loocv.py` (untracked as of this
audit; user has since finalized it). README's numbered Usage list has no
mention of `ablation_loocv.py` anywhere.

**Status (2026-08-29): resolved.** README Usage step 7 now runs
`ablation_loocv.py`; `ablation_study.py` is no longer in the numbered steps.

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

**Status (2026-08-29): resolved.** `augment_videos.py` now has real
`argparse` with `--input_dir`/`--output_dir` (defaults unchanged:
`data/videos` / `data/augmented_videos`, so bare invocation behaves exactly
as before). `run_pipeline.py`'s existing call site needed no changes — its
flags now do what they always looked like they did.

### 11. `extract_faces.py` isn't documented in README's Usage steps
**File:** `scripts/preprocessing/extract_faces.py` exists to produce
FaceFusion source-face images, which README's step 10 (manual FaceFusion GUI
process) implicitly depends on but never names as a prerequisite script.

**Status (2026-08-29): partially resolved.** `extract_faces.py` is now listed
in README's Project Structure tree with a description. Not added as a
numbered Usage step, since it's a manual-process helper (prints a
recommended-pairs table for the FaceFusion GUI) rather than a pipeline stage
with a fixed invocation — adding a prescriptive command risked overstating
how automated step 10 actually is.

### 12. `figures/README.md` says figures 1–4 are "Missing" — and paper.tex can't see the real ones either
**File:** `figures/README.md` lines 26–36 state the four conceptual diagrams
(`fig1_pipeline.png` … `fig4_architecture.png`) still need to be generated.
They are in fact already tracked in git — **but under `diagrams/`, not
`figures/`.**

**This is more than a stale doc note.** `paper.tex` line 28 sets
`\graphicspath{{figures/}{./}}` and lines 280/302/429/565 call
`\concept{figures/fig1_pipeline.png}{...}` etc. — the `\concept` macro
(line 34) does `\IfFileExists{figures/fig1_pipeline.png}{include it}{render a
placeholder box}`. Verified directly: `figures/fig1_pipeline.png` does **not**
exist (only `diagrams/fig1_pipeline.png` does, ~5MB, a real generated image).
So as currently wired, `paper.tex` compiles clean but **silently falls back to
placeholder boxes for all four conceptual figures**, even though the finished
diagrams already exist in the repo. Not fixed here since it's a `paper.tex`
edit, outside what was authorized this pass — but flagging prominently since
it's a one-line-per-figure fix (either move/copy the 4 PNGs into `figures/`,
or change the 4 `\concept{...}` paths to `diagrams/...`, or add `diagrams/`
to `\graphicspath` — that last option alone won't fix it, since `\IfFileExists`
checks the literal path given, not the graphics search path) and it
determines whether the submitted PDF has real figures or grey boxes.

---

## Structural drift

### 13. README's "Project Structure" tree omits several real top-level directories
`ieee_scripts/`, `diagrams/`, `figures/`, `scripts/generate_figures/`,
`AUDIT_FINDINGS.md`, `NOTES.md`, and `context.md` all exist and are tracked,
but aren't in README's tree. Conversely it lists `PLAN.md` and
`LITERATURE_REVIEW.md`, which don't exist (finding #7).

**Status (2026-08-29): resolved.** README's Project Structure tree now
includes `ieee_scripts/`, `diagrams/`, `figures/`, `tests/`,
`scripts/generate_figures/`, `NOTES.md`, `AUDIT_FINDINGS.md`, `LICENSE`, and
`CONTRIBUTING.md`; `PLAN.md`/`LITERATURE_REVIEW.md` are gone. `context.md`
(a portfolio/resume-style summary, not a paper-facing doc) was left out
deliberately, since the Project Structure section documents the pipeline, not
every root-level file.

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
