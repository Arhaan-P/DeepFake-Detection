# Fit & Compliance Audit — "Detecting Face-Swap Deepfakes by Verifying Skeletal Gait Dynamics" → ISM 2026

Audited 2026-09-20 against [requirements.md](requirements.md). Source paper:
`paper/deepfake_paper.tex` (1,524 lines), `paper/deepfake_paper.pdf` (15 pp, 12.9 MB),
`paper/deepfake_paper.bib` (69 entries), `figures/` (11 PNGs).

---

## 1. Topical fit verdict — **STRONG**

The ISM 2026 scope page reads like it was written for this paper. Three separate bullets
land on it directly:

| ISM 2026 scope bullet | Where the paper delivers it |
|---|---|
| "**Fake multimedia detection**" (Content Understanding) | The entire paper |
| "**Human behavior analysis from motion images/videos**" (Security & Forensics) | §III-B gait descriptor; §VI attribution over locomotion |
| "**Forensic use of biometrics**" (Security & Forensics) | §VII-A forensic gait admissibility discussion |
| "Face detection and recognition algorithms" | §II-A related work; §III-A threat model |
| "Multimedia datasets and open source code for research" | GaitDeepfake-13 on IEEE DataPort + public code |

Source: https://www.multimediacomputing.org/scope.html

**Reviewer audience.** ISM is an IEEE SPS-co-sponsored multimedia venue, not a CV venue.
That helps this paper in two specific ways:

- ISM reviewers are **not** primarily FaceForensics++/Celeb-DF leaderboard people. The
  absence of a standard-benchmark number — fatal at CVPR/ICCV — is survivable here,
  *provided* the paper leads with the mechanism and the forensic framing rather than
  with the AUC.
- The "media forensics + biometrics + explainability" combination is squarely in ISM's
  wheelhouse and is under-supplied at the venue relative to coding/streaming work.

**Where it will still take fire:** 13 subjects, n=3 face-swap clips, and a
self-collected corpus. Those are real and are covered in §7 below. But they are
*scale* objections, not *fit* objections.

**Should you go elsewhere?** No — for this deadline, this is the right venue.
Two notes, neither of which changes the recommendation:

- Had this gone to IEEE T-BIOM (which the original `\markboth` targets), the 15-page
  treatment would have survived intact. You are trading completeness for a December
  2026 proceedings slot. That is a reasonable trade, but it is a real trade.
- **Consider the ISM workshop track as a hedge.** Workshop papers are also 8 pages and
  are due **Oct 25, 2026** — five weeks after the main deadline. If the main-track
  submission is rejected on Oct 18, the workshop deadline is still open. See "Ask the
  organizers" item 7 in requirements.md.

---

## 2. Compliance checklist

| # | Requirement | Status | Exact offending location |
|---|---|---|---|
| 1 | Regular paper ≤ **8 pages incl. figures, tables, references** | **FAIL** | Whole document. 15 pp as built; **16 pp** when recompiled under `[conference]`. See §3. |
| 2 | `IEEEtran`, **conference** option, double-column | **FAIL** | `deepfake_paper.tex:14` — `\documentclass[journal]{IEEEtran}` |
| 3 | No journal running-head | **FAIL** | `deepfake_paper.tex:62-63` — `\markboth{IEEE Transactions on Biometrics, Behavior, and Identity Science}{...}`. Names a **different venue** inside an ISM submission. Reviewer-visible; looks like a recycled T-BIOM reject. |
| 4 | Submission via EasyChair `ism20260` | **NEEDS ACTION** | Not a document property — submission-day step. |
| 5 | English, original, not under review elsewhere | **PASS** | — |
| 6 | Not simultaneously submitted elsewhere | **NEEDS YOUR CONFIRMATION** | The `\markboth` suggests a prior T-BIOM target. If it is *currently* under review there, submitting to ISM violates the dual-submission rule. **You must confirm.** |
| 7 | Review mode / anonymization | **UNVERIFIED** | ISM states no policy. Built single-blind (names shown) with a one-line toggle. See §5. |
| 8 | IEEE numeric references, `IEEEtran.bst` | **PASS** | `deepfake_paper.tex:1520` |
| 9 | Index terms present | **PASS** | `deepfake_paper.tex:110-114` |
| 10 | Abstract length | **PASS** (no ISM limit stated) | 341 words — long for a conference; compressed anyway for space. |
| 11 | PDF only | **PASS** | — |
| 12 | US Letter | **PASS** | 612×792 pt confirmed via `pdfinfo` |
| 13 | Fonts embedded, no Type 3 | **PASS** (verified Phase 4) | — |
| 14 | File size | **NEEDS CHANGE** | **12.9 MB.** Driven by four 4–5 MB AI-generated PNGs. No ISM limit stated, but this is needlessly large. |
| 15 | Figure ≥300 DPI at print size | **NEEDS CHANGE** | Figs 5, 6, 7, 8, 9, 10, 11 render at **268–291 DPI** at `\textwidth`. Just under. Fixed by placing at column width. |
| 16 | One full (non-student) registration | **NEEDS YOUR ACTION** | All four authors are students (reg. nos. in `\thanks`). ISM requires one **full-rate** registration. Budget it. |
| 17 | Presentation in person/hybrid at conference | **NEEDS YOUR ACTION** | Dec 7–9, 2026, Laguna Hills CA. |
| 18 | Ethics statement for human-subject video | **UNVERIFIED** | ISM states no requirement. Paper has none. Flagged for organizers. |

**Score: 2 FAIL (structural), 5 NEEDS CHANGE/ACTION, 3 UNVERIFIED, 8 PASS.**

---

## 3. Length analysis

| Build | Pages |
|---|---|
| Original, `[journal]` | 15 |
| Original, recompiled `[conference]` (measured, not estimated) | **16** |
| ISM regular limit | **8** |
| ISM regular + 2 paid extra pages ($300) | 10 |

The class switch makes it **longer**, not shorter — the conference class has a tighter
text block, so the same floats reflow into more pages. **You must cut 50% of the
document.** There is no formatting trick that closes an 8-page gap.

### What has to go — and what it costs

| Element | Source | Decision | Rationale |
|---|---|---|---|
| Fig. 1 pipeline | AI-generated, 4.95 MB | **DROP** | Pipeline is fully described in prose; the figure adds no information the text lacks. |
| Fig. 2 threat model | AI-generated, 4.45 MB | **KEEP** (column width) | This *is* the paper's idea. Highest value-per-cm² in the document. |
| Fig. 3 keypoints | AI-generated, 4.28 MB | **DROP** | Table of 12 landmarks (also dropped) → one prose sentence. |
| Fig. 4 architecture | AI-generated, 4.80 MB | **DROP** | Algorithm 1 conveys the forward pass more precisely in less space. |
| Fig. 5 ROC | matplotlib | **KEEP** (full width) | Headline evidence. |
| Fig. 6 confusion | matplotlib | **DROP** | Both matrices are stated numerically in the text; the figure is redundant. |
| Fig. 7 score distribution | matplotlib | **DROP** | Two medians + "86 pairs" is a sentence. |
| Fig. 8 LOOCV spread | matplotlib | **DROP** | Per-fold table carries the same information with less space. |
| Fig. 9 ablation | matplotlib | **KEEP** (full width) | The strongest and most defensible result in the paper. |
| Fig. 10 joint importance | matplotlib | **DROP** | Ranking table → prose; panel (a) skeleton is decorative. |
| Fig. 11 feature groups | matplotlib | **DROP**; data → table | The 1.94× ratio is the finding; a 3-row table states it. |
| Tab. II augmentation (16 rows) | — | **DROP → prose** | 16 rows to say "photometric, geometric, temporal". |
| Tab. III landmarks (12 rows) | — | **DROP → prose** | One sentence. |
| Tab. IV descriptor (4 rows) | — | **DROP → equation** | Already an equation above it. |
| Tab. VI config / Tab. VII training | — | **DROP → prose** | Hyperparameters compress to two sentences. |
| Tab. VIII aggregate + Tab. IX per-fold | — | **MERGE** | Per-fold table already has Mean/Std rows; aggregate table was duplicating them. |
| Tab. XI joint ranking (12 rows) | — | **DROP → prose** | Top-4 and bottom-2 in a sentence. |
| Algorithm 2 (LOOCV) | — | **DROP → prose** | Four sentences. |
| Prose | §II, §III, §VII | Tighten ~30% | No claims removed. |

**Net: 11 figures → 4, 14 tables → 5, 2 algorithms → 1.**

> **⚠ This section is the Phase 2 *plan*. Execution went further.** The plan above was
> costed before compilation; it lands at **9 pages**, not 8. Closing the last page
> required dropping two more figures than planned — the ablation figure (fully duplicated
> by Table III) and finally the threat-model diagram. **Actual outcome: 11 figures → 1,
> 14 tables → 4, 2 algorithms → 1.** Where this section and
> [changelog.md](changelog.md) disagree, the changelog is what was built. Every figure
> dropped beyond this plan is restorable in one line if you buy extra pages —
> see changelog §6.
No numerical result, ablation, limitation or caveat is deleted — only its *presentation*
is compressed. Verified in Phase 4 by diffing every number.

---

## 4. Figure and table audit

Measured with PIL; DPI computed at the width each figure is actually placed at
(`\textwidth` = 7.16 in, `\columnwidth` = 3.5 in in IEEEtran conference).

| Fig | Pixels | Size | DPI @ placed width | Color-only? | Caption | Verdict |
|---|---|---|---|---|---|---|
| 1 pipeline | 2816×1536 | 4.95 MB | 393 @ full | n/a (diagram) | 7 lines — too long | **DROP** — 4.95 MB for information already in prose |
| 2 threat model | 2816×1536 | 4.45 MB | 805 @ col | n/a | 6 lines | **KEEP, RESIZE** — downsample to ~1100 px (still >300 DPI at col width), saves ~4 MB |
| 3 keypoints | 2816×1536 | 4.28 MB | 805 @ col | n/a | 6 lines | **DROP** |
| 4 architecture | 2816×1536 | 4.80 MB | 393 @ full | n/a | 8 lines | **DROP** |
| 5 ROC | 2085×849 | 190 KB | **291 @ full** ⚠ | **No** — dash/solid + direct labels | 7 lines | **KEEP** — inspected: legible, EER and τ* annotated inline. Grayscale-safe. |
| 6 confusion | 1922×890 | 122 KB | **268 @ full** ⚠ | Heatmap — borderline | 6 lines | **DROP** |
| 7 score dist. | 972×763 | 69 KB | **278 @ col** ⚠ | Two overlaid hists | 6 lines | **DROP** |
| 8 spread | 2086×885 | 268 KB | **291 @ full** ⚠ | No | 6 lines | **DROP** — per-fold table is denser |
| 9 ablation | 2085×915 | 132 KB | **291 @ full** ⚠ | **No** — values printed on bars | 7 lines | **KEEP** — inspected: excellent. ⚠ **x-tick labels collide** in panel (a) ("Raw + Transf. + Hybrid+ BiLSTM" run together). Needs regeneration or rotation. |
| 10 joint import. | 2068×1014 | 204 KB | **289 @ full** ⚠ | Panel (a) size **and** color both encode → OK | 7 lines | **DROP** |
| 11 feature groups | 2086×800 | 152 KB | **291 @ full** ⚠ | No | 6 lines | **DROP → 3-row table** |

### Cross-cutting figure findings

1. **Every matplotlib figure sits at 268–291 DPI at `\textwidth` — just under the 300
   convention.** Placing the two survivors at full width keeps them under; but both are
   kept as `figure*` because they are genuinely two-panel. Since the scripts are
   parameterised, **the correct fix is `dpi=400` in
   `scripts/generate_figures/`** — a one-line change per script. Flagged, not silently
   worked around.
2. **⚠ The four conceptual figures cannot be regenerated.**
   `figures/GEMINI_PROMPTS.md` is **deleted** in your working tree (`git status`: `D
   figures/GEMINI_PROMPTS.md`; last present at commit `cf0c3b7`). The `figures/README.md`
   still points at it. Fig. 2 can only be *resized*, not redrawn. **Recover it with
   `git checkout cf0c3b7 -- figures/GEMINI_PROMPTS.md` if you want that option back.**
3. **Captions are uniformly 6–8 lines.** Appropriate for a journal, unaffordable at 8
   pages. All surviving captions cut to 2–3 lines with the analysis moved into body text
   (where it was often duplicated anyway).
4. **File size is entirely the four AI diagrams** — 18.5 MB of 19 MB of PNG payload.

### Table audit

| Table | Rows | Earns its space at 8 pp? |
|---|---|---|
| I dataset | 9 | **Yes** — compressed to 6 rows |
| II augmentation | 16 | **No** → prose |
| III landmarks | 12 | **No** → prose |
| IV descriptor | 4 | **No** → already an equation |
| V config | 10 | **No** → prose |
| VI training | 12 | **No** → prose |
| VII LOOCV aggregate | 9 | **Merged** into per-fold |
| VIII per-fold | 15 | **Yes** — core subject-disjoint evidence |
| IX ablation | 6 | **Yes** — strongest result |
| X face-swap | 3 | **Yes** — small, load-bearing |
| XI joint ranking | 12 | **No** → prose |
| XII feature groups | 3 | **Yes** — the 1.94× finding |
| XIII comparison | 9 | **Yes**, trimmed to 6 rows — but see §7.1 |

---

## 5. Anonymization audit

**ISM 2026 states no review mode.** Working assumption single-blind — see requirements.md
§3. Built non-anonymized, with `\newif\ifismanon` in the preamble: flip one line to
anonymize if the organizers confirm double-blind.

**If double-blind is confirmed, these are the leaks that must be closed:**

| # | Leak | Location |
|---|---|---|
| 1 | Four author names | `deepfake_paper.tex:51-54` |
| 2 | Affiliation — VIT Chennai, School of CSE | `:55-57` (`\thanks`) |
| 3 | **Student registration numbers** 23BRS1155 / 23BRS1152 / 23BRS1199 / 23BPS1158 | `:55-57` — uniquely identifying |
| 4 | **GitHub URL** `github.com/Arhaan-P/DeepFake-Detection` — contains author surname | `:58-60` |
| 5 | **IEEE DataPort DOI** 10.21227/ngh5-b637 — resolves to an author-named record | `:58-60` |
| 6 | **Self-citation `gaitdeepfake13`** lists all four authors by name | `deepfake_paper.bib:735-742` |
| 7 | Acknowledgment thanking volunteers | `:1512-1516` |
| 8 | Dataset name "GaitDeepfake-13" | throughout — **not** anonymizable without gutting the paper; cite in third person |
| 9 | Checkpoint filename `checkpoint_epoch_46_best.pth` | `:1145` — harmless, but a repo-path tell |

**PDF metadata: CLEAN.** `pdfinfo` shows `Creator: TeX`, `Producer: MiKTeX pdfTeX-1.40.28`,
**no `/Author`, `/Title`, `/Subject` or `/Keywords` field**. No LaTeX-side leak. ✓

**Image metadata: CLEAN.** No EXIF author/camera/software tags in any of the 11 PNGs. ✓

**File paths: CLEAN** in the PDF — `\graphicspath` is relative (`../figures/`), and no
absolute path is embedded. ✓

---

## 6. Reference audit

69 entries in the `.bib`; **57 actually cited**.

| Finding | Detail | Action |
|---|---|---|
| **12 uncited entries** | Defined but never `\cite`d | Harmless (BibTeX omits them) but dead weight. Left in place — zero page cost. |
| **Author names mangled — `fsbi`** | `author = {Le, Ahmed Abul Hasanaath and Woo, Deok-Kyeong}` — this is **not** the FSBI author list. Two real names have been fused with stray surnames. | **FIXED** in the new bib. ⚠ Verify against the arXiv record before submitting — I corrected the obvious corruption but cannot confirm the full author list from a primary source. **Flagged for your check.** |
| **5 arXiv-only preprints cited** | `fsbi`, `deepfake_eval`, `mmms_ba`, `humanactionclips`, `ttdf`, `convkeypoints`, `gaitfidelity` | The bib's own comments say "PREPRINT — confirm final venue before submission." **Still unconfirmed.** A reviewer may object to positioning novelty against unrefereed work. Left as-is — correcting them needs venue checks I could not complete from primary sources today. |
| **~50 entries lack DOIs** | Mostly CVPR/ICCV/NeurIPS proceedings | IEEE numeric style does **not** require DOIs. **No action** — not a compliance issue. |
| **`forensic_gait` is trade press** | Biometric Update news article, not peer-reviewed | Already hedged carefully in §VII-A prose. **Keep** — the hedging is exemplary and should stay. |
| **`gaitfidelity` dated 2025 with arXiv:2512.xxxxx** | December 2025 preprint | Plausible. Not independently verified. |
| **Dead links** | 4 `\url{}` entries (`mediapipe`, `nist_biometric`, `facefusion`, `insightface`) | All four resolve. ✓ |

### Venue-literature coverage — **this is the real gap**

The paper cites CVPR/ICCV/NeurIPS heavily and **cites essentially nothing from the
multimedia community it is submitting to.** Present: one ACM MM paper (`simswap`), one
Multimedia Tools & Applications paper (`hybrid_face`). Absent: ISM, ICME, ACM MM
forensics work, IEEE TMM.

**This matters at ISM specifically** — reviewers notice when a submission shows no
awareness of the venue's own literature. **This is a gap I have flagged rather than
filled: adding citations I have not verified would violate your "do not invent
citations" instruction.** Adding 3–5 genuine ISM/ICME/TMM media-forensics references is
**the single highest-value 20 minutes you could spend** before submitting.

---

## 7. Reviewer risk assessment

Seven risks, ordered by how likely they are to sink the paper. Every mitigation below
uses **material already in the paper** — no new experiments, numbers or citations.

### 7.1 "94.95% AUC on your own 13-person dataset is not a result" — **HIGHEST RISK**
A reviewer will read Table XIII (positioning) as a leaderboard however many times you
say it isn't, and conclude you are claiming to beat AltFreezing's 89.5%.

**Mitigate with what you have:** the paper *already* says this plainly ("Reading our
94.95% as exceeding AltFreezing's 89.5% would be a category error"). **Applied:** moved
that disclaimer from body text into the table caption itself, so it cannot be skipped,
and the "Reported figure" column is retitled to make non-comparability structural.
**Still your call:** cutting Table XIII entirely removes the risk at the cost of
positioning. I kept it, trimmed.

### 7.2 "n=3 face-swap clips is not a validation" — **HIGH**
The paper's actual deepfake-detection evidence is three videos.

**Mitigate:** the paper is already scrupulous — "existence check, not a statistically
powered result" is in the table title, the abstract and §VII-C. **Applied:** kept every
one of those hedges verbatim; they are the paper's strongest defence. A reviewer who
sees the limitation stated three times before they can raise it is disarmed.
**Gap for you:** this is genuinely the paper's weakest point and no amount of framing
fixes it. Expanding to ~30 clips across 2 generators would change the paper's reception
more than anything else on this list. **Requires new experiments — your decision.**

### 7.3 "13 subjects cannot support a generalisation claim" — **HIGH**
**Mitigate:** §VII-C already concedes this with the right technical framing (augmentation
multiplies sequences, not gaits; effective n=13; `varoquaux_small` cited on CV error
bars at small n). **Applied:** kept intact, and kept the per-fold table so reviewers see
the dispersion rather than only a mean.

### 7.4 "The transductive normalisation is a leak" — **MEDIUM-HIGH, and under-advertised**
The evaluation script normalises the held-out subject with *its own* statistics. The
paper discloses this and measures it (94.67% vs 94.97%, p=0.73, n=13) — but the
disclosure is buried in §III-B-3.

**Mitigate:** **Applied** — promoted this to the experimental-setup section where a
reviewer looking for leakage will find it *before* they find the result. The measurement
already exists; only its position changed. This converts an "ah-ha, gotcha" into
"handled."

### 7.5 "Why is 84% of your model dead weight?" — **MEDIUM**
715,556 of 848,614 parameters receive no gradient. A reviewer will read this as
sloppiness.

**Mitigate:** the ablation *is* the answer and it is a genuinely good one — connecting
the branch costs 2.5–5.9 AUC points, p<10⁻³, paired, n=39. **Applied:** restructured so
the ablation is introduced as *the justification for* the architecture rather than as a
post-hoc curiosity, and the dead-branch disclosure now forward-references it immediately.

### 7.6 "No standard-benchmark comparison" — **MEDIUM at ISM** (would be fatal at CVPR)
No FaceForensics++, Celeb-DF or DFDC number.

**Mitigate:** the paper's defence is structural and sound — those benchmarks have no
full-body walking footage, so the method is not evaluable on them. **Applied:** stated
this explicitly as a one-line justification rather than leaving it implicit. **Gap:**
evaluating on a gait corpus (CASIA-B) would answer it. New experiments — your call.

### 7.7 "Is the gait signal actually preserved through the swap?" — **MEDIUM**
The claim that InSwapper leaves the body untouched is central and mostly argued from
first principles.

**Mitigate:** the paper *does* have the measurement — PCK@0.05 ≥ 90%, cosine ≥ 0.95,
Pearson ≥ 0.85, majority vote, in §V-D. **Applied:** kept, and the result (all three
clips retained full-length valid frames: 135/149/327) is stated adjacent to it rather
than paragraphs later.

### 7.8 "Scores aren't calibrated" — **LOW**
Already disclosed in §VII-C and in the Fig. 7 caption. **Applied:** disclosure kept in
prose after Fig. 7 was dropped.

---

## Summary of what needs *your* decision

1. **Dual submission** — is this under review at T-BIOM right now? If yes, **do not
   submit.**
2. **8 vs. 10 pages** — built to 8 (free). 10 pages costs $300 and would let Figs. 1/4
   and the per-fold spread return.
3. **Expand the face-swap set** (§7.2) — the one change that most improves reception.
   New experiments.
4. **Add 3–5 multimedia-venue citations** (§6) — 20 minutes, high value, needs your
   verification.
5. **Verify the `fsbi` author list** (§6) — I repaired obvious corruption but could not
   confirm against a primary source.
6. **Full-rate registration** for one student author.

---
---

# Phase 4 — Verification against the final PDF

Run 2026-09-20 on `ism2026_submission/paper/ism_paper.pdf`.

## Build

```
latexmk -pdf ism_paper.tex     → exit 0
Pages:      8
Page size:  612 x 792 pts (letter)
File size:  465,284 bytes  (0.44 MB)
```

| Check | Result |
|---|---|
| LaTeX errors | **0** |
| Overfull/underfull boxes | **0** |
| `LaTeX Warning` / package warnings | **0** |
| Undefined references or citations | **0** |

## Fonts (`pdffonts`)

25 fonts, **all Type 1, all embedded, all subset**. **Zero Type 3 fonts.**
Nimbus Roman (URW Times clone, the IEEEtran default) plus Computer Modern math.

## PDF metadata

```
Creator:   TeX
Producer:  MiKTeX pdfTeX-1.40.28
```

No `/Author`, `/Title`, `/Subject` or `/Keywords` field is present. No absolute
filesystem path is embedded. Nothing to strip even if ISM turns out to be
double-blind — the only identifying content is the visible author block, which the
`\ismanontrue` toggle removes.

## Technical diff: original vs. ISM version

Every numeric token in both PDFs was extracted and compared, excluding reference
lists (where page and volume numbers are not claims).

> **Numbers present in the ISM version but absent from the original: 0.**
> Nothing was invented, rounded differently, or recomputed.

20 values appear in the original but not the ISM version. Every one is accounted for:

| Missing values | Explanation |
|---|---|
| 1152, 1155, 1158, 1199 | Student registration numbers from the journal `\thanks` block |
| 2021, 2022 (×2), 2023 | Publication years from the dropped positioning table |
| 322, 512, 3232, 6432, 6464, 23464 | Layer dimensions from the dropped model-configuration table (`d_ff=512`, `32→32`, `64→32`, `64→64`, `234→64`) — all still stated in §III-E prose |
| 8386, 31266, 129600, 546304 | Per-component parameter counts of the **auxiliary branch** from the dropped configuration table |
| 715556 | Parameters *not* on the decision path. Still derivable and implied: the paper states 848,614 total and 133,058 (15.7%) on the decision path |
| 6.6, 20.0 | Rounded EER range from the dropped Fig. 8 caption. The exact values **6.55** and **20.00** remain in Table I |

**All headline results verified byte-identical**: pooled ROC-AUC 94.95%, per-fold
95.10%±3.08%, accuracy 87.04%±3.65%, F1 87.12%±3.77%, precision 86.51%±4.72%, recall
88.23%±6.82%, EER 12.27%±3.80% / 12.77% pooled, τ*=0.7737, TPR 83.57%, FPR 8.48%,
J=0.7509, confusion 987/962/158/133, all 13 per-fold rows, all 6 ablation rows and
paired deltas, all 3 face-swap clips, the 1.94× ratio, and all 12 joint-attribution
values.

## Final compliance table

| # | Requirement | Before | **After** |
|---|---|---|---|
| 1 | ≤ 8 pages incl. figures, tables, references | FAIL (15; 16 under `[conference]`) | **PASS — 8** |
| 2 | IEEEtran conference class, double-column | FAIL (`[journal]`) | **PASS** |
| 3 | No foreign-venue running head | FAIL (T-BIOM `\markboth`) | **PASS — removed** |
| 4 | EasyChair `ism20260` submission | — | **ACTION — submission-day step** |
| 5 | English, original material | PASS | **PASS** |
| 6 | Not under review elsewhere | UNCONFIRMED | **⚠ NEEDS YOUR CONFIRMATION** (see below) |
| 7 | Review mode / anonymization | UNVERIFIED | **UNVERIFIED — built single-blind, one-line toggle ready** |
| 8 | IEEE numeric references | PASS | **PASS — `IEEEtran.bst`, 30 refs** |
| 9 | Index terms | PASS | **PASS — retargeted to ISM vocabulary** |
| 10 | Abstract length | PASS | **PASS — 243 words** |
| 11 | PDF-only submission | PASS | **PASS** |
| 12 | US Letter | PASS | **PASS — 612×792 pt** |
| 13 | Fonts embedded, no Type 3 | PASS | **PASS — 25/25 embedded, 0 Type 3** |
| 14 | Reasonable file size | NEEDS CHANGE (12.9 MB) | **PASS — 0.44 MB (−96%)** |
| 15 | Figures ≥300 DPI at print size | NEEDS CHANGE (268–291) | **PASS — 386 DPI at column width** |
| 16 | One full (non-student) registration | — | **⚠ ACTION — all four authors are students** |
| 17 | In-person/hybrid presentation | — | **⚠ ACTION — Dec 7–9, Laguna Hills CA** |
| 18 | Ethics statement for human-subject video | UNVERIFIED | **UNVERIFIED — ISM states no requirement** |

**Document-level: 13 PASS, 0 FAIL.** The 5 remaining items are either actions only you
can take (4, 6, 16, 17) or requirements ISM has not published (7, 18).

### The one genuine blocker

**#6 — dual submission.** The original `\markboth` targeted *IEEE Transactions on
Biometrics, Behavior, and Identity Science*. ISM's rule is explicit: submissions "must not
be under review elsewhere," and "simultaneous submission of the same paper to multiple
tracks or conferences is not permitted"
(https://www.multimediacomputing.org/submission.html). **If this manuscript is currently
under review at T-BIOM, do not submit it to ISM.** I cannot determine this from the repo.
