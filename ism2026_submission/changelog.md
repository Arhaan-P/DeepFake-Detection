# Changelog — ISM 2026 retarget

Source: `paper/deepfake_paper.tex` (IEEEtran `[journal]`, 15 pp)
Result: `ism2026_submission/paper/ism_paper.tex` (IEEEtran `[conference]`, **8 pp, $0 in extra-page fees**)

> **SUPERSEDED — see §10 for the final state (8 pages, 4 numbered figures / 5 illustrations).** Earlier state: 9 pages, 3 figures (1 regenerated plot + 2 new vector diagrams).
> Sections 2–8 below describe the 8-page, 1-figure state reached first; **§9 records the
> two TikZ figures added afterwards and the one extra page they cost.** Where §2 and §9
> disagree, §9 is current.

**The original files were not modified.** Verified: `git status` shows no changes
under `paper/` or `figures/`.

Requirement IDs (R1–R18) refer to the compliance table in
[fit_report.md](fit_report.md) §2. Every requirement URL is in
[requirements.md](requirements.md).

---

## 1. Document class and front matter

| # | Change | Why | Satisfies |
|---|---|---|---|
| 1.1 | `\documentclass[journal]{IEEEtran}` → `\documentclass[conference]{IEEEtran}` | ISM requires "double-column IEEE format" for conference proceedings | R2 |
| 1.2 | Deleted `\markboth{IEEE Transactions on Biometrics, Behavior, and Identity Science}{...}` | The running head named **a different venue**. Left in place it reads as a recycled T-BIOM submission — the worst possible first impression on an ISM reviewer. | R3 |
| 1.3 | Journal-style `\author{...\thanks{...}}` → `\IEEEauthorblockN/A` conference block | Conference class convention | R2 |
| 1.4 | Student registration numbers (23BRS1155, 23BRS1152, 23BRS1199, 23BPS1158) removed from the author block | Not conventional in IEEE conference papers, and uniquely identifying if ISM turns out to be double-blind | R2, R7 |
| 1.5 | Added `\newif\ifismanon` / `\ismanonfalse` toggle wrapping the author block and acknowledgment | ISM publishes **no review-mode policy**. Built single-blind (the IEEE conference default); flipping one line to `\ismanontrue` anonymizes the whole document. | R7 |
| 1.6 | Dataset DOI + GitHub URL moved from a title-page `\thanks` into the Acknowledgment | So `\ismanontrue` removes both identifying links in the same stroke | R7 |
| 1.7 | Keywords retargeted: added *media forensics*, *multimedia security*; dropped *skeletal keypoints* | ISM indexes under Multimedia Security and Forensics; the new terms match the CFP's own vocabulary | R9, topic fit |
| 1.8 | Abstract 341 → 243 words | Space; no ISM limit is stated. **Every numeric result retained.** | R1 |
| 1.9 | Added float-packing parameters (`\topfraction` etc.) and `\arraystretch{0.96}` | Placement/typography only — changes no content | R1 |

## 2. Figures: 11 → 1

Original figure payload was **18.5 MB of PNG** producing a **12.9 MB** PDF. The
ISM PDF is **0.47 MB** — a 96% reduction.

| Original | Decision | Why |
|---|---|---|
| Fig. 1 pipeline (AI-generated, 4.95 MB) | **Dropped** | Pipeline fully described in §III prose; figure added no information |
| Fig. 2 threat model (AI-generated, 4.45 MB) | **Dropped** (last cut made) | Its whole content — "the generator rewrites only the face, so the body's gait survives" — is stated in the abstract, twice in §I, and in §III-A ¶1. Was the highest-value diagram, but also the most redundant. **Reversible: see §6.** |
| Fig. 3 keypoints (AI-generated, 4.28 MB) | **Dropped** | The 12 landmarks are now named in one prose sentence |
| Fig. 4 architecture (AI-generated, 4.80 MB) | **Dropped** | The inlined forward-pass description is more precise in less space |
| Fig. 5 ROC | **KEPT**, regenerated, cropped to panel (a), moved to single column | Curve shape is the one thing in the paper that a table cannot express |
| Fig. 6 confusion | **Dropped** | Both matrices stated numerically in §V-B |
| Fig. 7 score distribution | **Dropped** | Reduces to one sentence (medians 0.999 / 0.0007; 86 pairs = 3.8%) |
| Fig. 8 LOOCV spread | **Dropped** | Per-fold table carries the same data more densely |
| Fig. 9 ablation | **Dropped** | Fully duplicated by Table III (AUC, accuracy, params, Δ, significance) and the prose |
| Fig. 10 joint importance | **Dropped** | All 12 attribution values now given in §VI-A prose |
| Fig. 11 feature groups | **Dropped** | The three ratios are stated in §VI-B prose |

**Figure quality work (R15):**
- Regenerated Fig. 5 **from its source data** (`outputs/evaluation/loocv/loocv_results.json`)
  at **400 DPI** instead of 300, via a runtime patch of `figstyle.DPI` and `figstyle.FIGDIR`.
  **Nothing in `scripts/` or `figures/` was modified.** The regeneration re-printed every
  metric from the stored artifacts, independently confirming the numbers used in the text.
- Final placed resolution: **386 DPI** at column width (was 291 DPI at `\textwidth` — under
  the 300 convention).
- Flattened RGBA → RGB on white so the figure prints correctly without alpha.
- Caption cut from 7 lines to 4; the analysis moved into body text.

## 3. Tables: 14 → 4

| Original table | Decision | Why |
|---|---|---|
| I Dataset composition | **Merged into prose** | 9 rows → 2 sentences, all values kept |
| II Augmentation (16 rows) | **→ prose** | Now one sentence grouping the 16 ops as photometric / geometric / temporal, each named |
| III Landmarks (12 rows) | **→ prose** | One sentence naming all 12 |
| IV Descriptor composition | **→ dropped** | Duplicated the equation directly above it |
| V Model configuration | **→ prose** | Decision-path parameter counts (133,058 / 848,614 / 15.7%) kept in §III-E |
| VI Training configuration | **→ prose** | All 12 hyperparameters retained in one §III-F paragraph |
| VII LOOCV aggregate | **Merged into Table I** | Per-fold table already had Mean/Std rows; added a `Pooled` row + Youden footnote. One table now does two jobs. |
| VIII Per-fold results | **KEPT** → Table I | The core subject-disjoint evidence. All 13 folds retained. |
| IX Ablation | **KEPT** → Table III | The paper's strongest result |
| X Face-swap validation | **KEPT** → Table IV | Only direct deepfake-detection evidence; n=3 deserves full per-clip disclosure |
| XI Joint ranking (12 rows) | **→ prose** | All 12 values listed inline in §VI-A |
| XII Feature groups | **→ dropped** | The paragraph above stated all 6 percentages and all 3 ratios verbatim |
| XIII Positioning/comparison | **→ prose** | See §5 below — this was also the top reviewer risk |

## 4. Algorithms: 2 → 1

- **Algorithm 1** (verification forward pass): **inlined into §III-E prose.** All kernel
  sizes, channel counts, normalisation, dropout placement and MLP dimensions preserved
  verbatim (`234→64, k=7; 64→64, k=5; 64→32, k=3`; GAP; `32→32→2`).
- **Algorithm 2** (LOOCV procedure): already prose in the original §IV-A; the float was
  redundant and is gone.

## 5. Substantive editorial changes

These change emphasis or placement, never claims. Each addresses a risk from
[fit_report.md](fit_report.md) §7.

| # | Change | Why | Risk addressed |
|---|---|---|---|
| 5.1 | **Positioning table → prose argument.** Replaced Table XIII with a paragraph that quotes the same competitor figures (FTCN 86–89%, AltFreezing 89.5%, GaitFormer 92.5%, GaitPT 82.6%) and states explicitly that tabulating them "would invite exactly the ranking they cannot support." | A table *looks* like a leaderboard no matter what the caption says. Prose cannot be misread as a ranking. | 7.1 — highest |
| 5.2 | **Transductive-normalisation disclosure promoted** from §III-B-3 (buried in feature preprocessing) to §IV-A-1, immediately after the LOOCV protocol, under its own heading "A protocol detail, stated up front". Measurement unchanged (94.67%±3.06 vs 94.97%±2.90, p=0.73, n=13). | A reviewer hunting for leakage now finds the disclosure *before* the result. Converts a "gotcha" into "handled." | 7.4 |
| 5.3 | **Dead-branch disclosure now forward-references the ablation immediately** ("Section V-C asks whether that stack *should* be connected, and finds that it should not"). | 84% of parameters receiving no gradient reads as sloppiness until the reader knows it was tested. | 7.5 |
| 5.4 | **Added an explicit limitation**: "*No standard-benchmark comparison.*" — states that FF++/Celeb-DF/DFDC lack full-body walking footage, so the method is not evaluable on them without new data. | The original left this implicit; a reviewer would otherwise raise it as an omission rather than a scoping decision. | 7.6 |
| 5.5 | **Limitations reordered** to put the n=3 face-swap validation first (was third). | It is the paper's weakest point; leading with it is more credible than burying it. | 7.2 |
| 5.6 | **Removed the Grad-CAM sentence.** The original said Grad-CAM was computed "for temporal localisation" but **no temporal-localisation result appears anywhere in the paper.** | Describing an analysis whose output is never reported invites "where is it?" Removing it also freed its reference. | — |
| 5.7 | Confusion-matrix note reworded from a correction of *earlier external reports* ("the frequently quoted…has sometimes been reported alongside") to an internal note about this paper's two matrices. | ISM reviewers have not seen the earlier reports; the self-referential version was addressed to nobody. The τ=0.5 vs τ* distinction is still made. | — |
| 5.8 | ~2,400 words of prose tightened across every section | 16 pp → 8 pp. **No claim, number, hedge or limitation removed.** | R1 |

## 6. Reversing cuts if you buy extra pages

ISM allows **2 extra pages at $150 each** (regular papers, max 10 pp). If you take them,
restore in this order — highest value first. All assets are still in
`ism2026_submission/paper/figures/`.

1. **`fig_threat.png`** — re-add as a `figure` at `width=0.82\columnwidth`. The clearest
   single explanation of the paper's idea. (~0.3 pp)
2. **`fig_ablation.png`** — re-add as a `figure*`. Panel (b)'s paired-CI forest plot is
   more persuasive visually than Table III's asterisks. (~0.4 pp)
3. **`fig_roc.png`** — swap the cropped `fig_roc_pooled.png` back for the full two-panel
   version as a `figure*`, restoring the per-fold envelope. (~0.2 pp)

Each is a one-line `\includegraphics` change.

## 7. References

| Change | Detail |
|---|---|
| **Corrected `fsbi` author list** | Was `{Le, Ahmed Abul Hasanaath and Woo, Deok-Kyeong}` — a corrupted parse fusing one author's given names with two surnames that are not on the paper. Now `{Hasanaath, Ahmed Abul and Luqman, Hamzah and Katib, Raed and Anwar, Saeed}`, **verified against https://arxiv.org/abs/2406.08625**. Title also corrected: "Deepfake**s** Detection". |
| Cited references 57 → 30 | Removed method-boilerplate citations (Adam, dropout, early stopping, batch size, Xavier/He init, GELU, pre-LN, ResNet, LSTM, Transformer, FaceNet, augmentation survey, and duplicate biomechanics/metric cites). **No citation supporting a factual or comparative claim was removed.** |
| `IEEEtran.bst`, numeric style | Unchanged — already compliant (R8) |
| **Not done: preprint venue checks** | 6 arXiv-only entries remain. The bib's own comments say "confirm venue before submission." I could not confirm these from primary sources today. **Flagged in fit_report.md §6.** |
| **Not done: multimedia-venue citations** | The paper still cites almost nothing from ISM/ICME/ACM MM/TMM. Adding 3–5 would materially help at this venue. **Not done because inventing citations was out of bounds.** **Flagged in fit_report.md §6.** |

## 8. What was verified, not assumed

- **Page count under the ISM template was measured, not estimated.** The original was
  recompiled under `[conference]` before any edits: it came out at **16 pages**, i.e. the
  class switch made it *longer*. That measurement drove the scale of the cuts.
- **Every number was diffed** between the original and final PDFs (`numdiff.py`, excluding
  reference lists). **0 values appear in the ISM version that are not in the original.**
  20 values are absent from the ISM version; every one is accounted for — student
  registration numbers, years from the dropped positioning table, the dropped
  model-configuration table's per-component parameter counts, and two rounded EER bounds
  from a dropped figure caption. See Phase 4 report in the final summary.
- **Fonts, metadata and page size verified on the final PDF** — see final summary.

---

## 9. Figures added (post-review request)

Two **vector TikZ** figures were built to replace what the raster originals did badly.
Source files: `paper/fig_architecture.tex`, `paper/fig_threat.tex` — both `\input` by the
main document, so they are editable text, not images.

| | Fig. 1 — Architecture | Fig. 2 — Threat model |
|---|---|---|
| Placement | `figure*`, full width | `figure`, column width |
| Size | 515.5 × 193.5 pt (= exactly `\textwidth`) | 206 × 181 pt |
| Replaces | `fig1_pipeline.png` + `fig4_architecture.png` (9.75 MB raster) | `fig2_threat_model.png` (4.45 MB raster) |
| Cost in PDF | ~31 KB of vector drawing commands | ~20 KB |

**Why vector, not the original PNGs.** The four conceptual figures in the original were
Gemini-generated rasters at 4–5 MB each — they were 96% of the old 12.9 MB PDF, they
capped out at a fixed resolution, and `figures/GEMINI_PROMPTS.md` is deleted so they could
never be regenerated or corrected. TikZ output is resolution-independent (the ≥300 DPI
requirement stops applying at all), diffable in git, and editable without regenerating an
image.

**Content decisions:**
- Fig. 1 makes the paper's most confusing structural point visible: the ⊗ stop-gradient
  marker and the dashed lane show at a glance that 715,556 of 848,614 parameters never
  reach the decision path. That took three paragraphs of §III-E to explain in the original.
- Fig. 2 is grayscale-safe by construction: the rewritten band is marked by a heavy
  **dashed red border** *and* an explicit "(synthesised)" label, so no information is
  carried by colour alone.
- Both use the legacy `arrows` TikZ library, not `arrows.meta` — the installed pgf is a
  2014 build (`\pgfversion` 0.0, dated 2014-07-01) and does not ship `arrows.meta`.

**One number was caught and removed.** The first draft of Fig. 1 labelled the auxiliary
branch "$84.3\%$ of the model". That is correct arithmetic (715,556 / 848,614) but it is
**not a figure the original paper states**, and the numeric diff flagged it. It was
replaced with "715,556 of 848,614 par.", both of which the original does state, so the
"nothing invented" guarantee stays absolute. Re-verified: **0 numbers in the ISM version
that are not in the original.**

### Page-count consequence — measured

| Build | Pages |
|---|---|
| No new figures (previous state) | 8 |
| **+ Fig. 1 only** (architecture) | **9** |
| **+ Fig. 1 and Fig. 2** | **9** |

The reference list was already sitting exactly at the page-8 boundary, so **any**
meaningfully sized figure spills it onto page 9. Adding the second figure is therefore
**free** once the first is in — both configurations cost the same single extra page.

**Implication:** the real choice is *both figures for $150*, or *no figures for $0*. There
is no cheaper middle option. Reaching 8 pages with the figures would require cutting
roughly 600 further words (~9% of the body) — realistically the Explainability section's
discussion, "What This Approach Buys", and Future Work.


---

## 10. Final state (8 pages, no extra fees, 5 illustrations)

**Page count: 8, including all references — inside ISM's free limit.** Verified by build.

| Fig. | Content | Source | Placement |
|---|---|---|---|
| 1 | End-to-end architecture, stop-gradient lane | new TikZ vector | full width |
| 2 | Threat model | new TikZ vector | column |
| 3 (left) | Pooled ROC, EER + Youden points | your `fig5_roc` data, regenerated at 400 DPI | full-width pair |
| 3 (right) | Paired-CI forest plot of the ablation | panel (b) of your `fig9_ablation`, regenerated at 400 DPI | full-width pair |
| 4 | Attribution share vs. dimensional share (the 1.94x finding) | panel (a) of your `fig11_feature_groups` | column |

Chosen because each shows something a table cannot: the curve shape, the CI overlap with zero,
the per-dimension ratio. Dropped as redundant with tables/prose: confusion matrix, score
distribution, per-fold spread, joint-ranking bars.

**Author added:** Dr. Deepika Roselind J appended as last author, same affiliation line. This is a
**guess at format**: I have no designation or department for her, so none was invented. If she
should be an acknowledgment rather than an author, or needs a different title/affiliation,
edit lines 63-66 of `ism_paper.tex`. She already appears as a dataset co-creator in the
`gaitdeepfake13` bib entry.

**Space recovered (~1 page) without removing any number or caveat:** Contributions, Conclusion,
Comparative Positioning, Limitations (now run-in), Operating Points, Face-Swap results, Metrics and
Future Work were rewritten more tightly; three simple display equations set inline; float spacing
tightened; the Ablation "two consequences" paragraph folded into the preceding one. Removed the
Dataset paragraph that restated the effective-sample-size limitation already in Sec. VII-C.

**Verification:** 0 numbers in the ISM PDF that are absent from the original; 0 LaTeX warnings, 0
overfull boxes; all fonts embedded, no Type 3; US Letter; no author metadata.
