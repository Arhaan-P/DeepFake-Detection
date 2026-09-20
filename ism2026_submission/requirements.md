# ISM 2026 — Submission Requirements

Compiled 2026-09-20 from primary sources. Every line carries its source URL.
Anything not confirmed on an official ISM 2026 page is marked **UNVERIFIED**.
No item on this page is filled in from memory of previous years.

---

## Phase 0 — Which "ISM 2026"?

### Candidates found

| # | Conference | Dates | Venue | Publisher / sponsor | URL |
|---|---|---|---|---|---|
| 1 | **IEEE International Symposium on Multimedia (ISM 2026)** — 28th edition | **Dec 7–9, 2026** | Laguna Hills, California, USA (hybrid) | IEEE; technically co-sponsored by **IEEE Signal Processing Society**; co-located with AIxSoftware 2026 | https://www.multimediacomputing.org/ |
| 2 | International Conference on Industry of the Future and Smart Manufacturing (ISM 2026) | Nov 10–14, 2026 | Padua & Venice, Italy | MSC-LES / Univ. della Calabria; Elsevier Procedia-style | https://www.msc-les.org/ism2026/ |
| 3 | International Symposium on Multidisciplinary Studies and Innovative Technologies (ISMSIT 2026) | 2026 | Türkiye | IEEE | https://www.ismsitconf.org/ |
| 4 | Int. Society for Music Information Retrieval (ISMIR) | — | — | ISMIR | Different acronym; listed only to rule out |

### Verdict — **Candidate 1: IEEE International Symposium on Multimedia**

Your suspicion was right. Reasons:

1. **The CFP scope names this paper's exact topic.** The ISM 2026 scope page lists a
   *Multimedia Security and Forensics* track containing "Face detection and recognition
   algorithms", "**Human behavior analysis from motion images/videos**", "**Forensic use
   of biometrics**", and "Multimedia-based computer forensics"; the *Content
   Understanding* track separately lists "**Fake multimedia detection**" and "Multimedia
   datasets and open source code for research."
   — https://www.multimediacomputing.org/scope.html
2. Candidate 2 is a **manufacturing / Industry 4.0** conference — no overlap with video
   forensics. Its own description is "smart culture … the 4th industrial revolution."
   — https://www.msc-les.org/ism2026/
3. Candidates 3 and 4 are different acronyms entirely.

**Not ambiguous. Proceeding with IEEE ISM 2026 without stopping.**

---

## 1. Important dates

Source for every row: https://www.multimediacomputing.org/about.html (and the identical
block on https://www.multimediacomputing.org/)

| Milestone | Date |
|---|---|
| Workshop proposal | Aug 1, 2026 PT |
| **Paper submission** | Sep 1, 2026 → **Sep 20, 2026, 11:59 PM PT — EXTENDED** |
| Workshop proposal acceptance | Sep 1, 2026, 11:59 PM PT |
| Notification of paper acceptance | Oct 1, 2026 → **Oct 18, 2026 PT — EXTENDED** |
| Workshop paper submission | Oct 15, 2026 → **Oct 25, 2026 PT — EXTENDED** |
| Poster abstract submission | Oct 15, 2026 → **Oct 25, 2026 PT — EXTENDED** |
| Camera-ready **and registration** | Nov 1, 2026, 11:59 PM PT |
| Conference | Dec 7–9, 2026 |

> ⚠️ **The full-paper deadline is TODAY (Sep 20, 2026, 11:59 PM PT).**
> Site announcement banner: "Paper submission is open through September 20, 2026,
> 11:59 PM PT." — https://www.multimediacomputing.org/

**There is no separate abstract deadline** for regular papers. (The "Poster Abstract"
row is a distinct poster-only track, Oct 25.)

Conflicting third-party listings — **do not rely on these**, they are stale:
- IEEE SPS event page still shows submission "1 September 2026" —
  https://signalprocessingsociety.org/events/2026-28th-international-symposium-multimedia-ism
- WikiCFP shows submission Jul 1, notification Sep 1 —
  http://wikicfp.com/cfp/servlet/event.showcfp?eventid=192756

### Submission system
**EasyChair**, conference id `ism20260`:
https://easychair.org/conferences/?conf=ism20260
— linked from https://www.multimediacomputing.org/ and
https://www.multimediacomputing.org/submission.html

"Technical papers must be submitted through EasyChair." / "Only electronic PDF
submission is accepted." — https://www.multimediacomputing.org/submission.html

---

## 2. Paper types and page limits

Source: https://www.multimediacomputing.org/submission.html

| Type | Limit | Extra pages allowed |
|---|---|---|
| **Regular paper** | **8 pages** | up to 2 |
| Short paper | 4 pages | yes, under fee policy |
| Position paper | 2 pages | yes, under fee policy |
| Workshop paper | 8 pages | up to 2 |
| Poster abstract | <2000 characters, one paragraph | n/a |

- **References COUNT toward the limit.** Verbatim: "Regular, short, workshop and position
  papers are limited to 8, 4, 8 and 2 pages respectively, **including figures, tables and
  references**."
- **Extra pages cost $150 per page.**
- All papers reviewed by **at least three experts**.

**Target for this paper: regular, 8 pages, no extra-page fee.**

### Poster abstract rules (only if you fall back to that track)
In English, one paragraph, <2000 characters, **no symbols, special characters,
abbreviations, footnotes, references, equations or tables**. IEEE letter-size conference
template; PDF contains only title, authors, abstract, keywords. In-person only; separate
registration; does not count toward multi-paper discount. Max poster 48in W × 36in H.

---

## 3. Review mode and anonymization

**UNVERIFIED — this is the single most important open item.**

- The official submission page **makes no mention of double-blind review, anonymization,
  or removing author names.** It says only: "All papers are reviewed by at least three
  experts." — https://www.multimediacomputing.org/submission.html
- No anonymization instruction appears anywhere on the ISM 2026 site
  (checked: `/`, `/about.html`, `/submission.html`, `/scope.html`).
- Targeted web searches for an ISM 2026 blinding policy returned nothing from a primary
  source.

**Working assumption: SINGLE-BLIND (author names and affiliations shown).** This is the
IEEE-conference default, and the absence of any anonymization instruction on a page that
otherwise spells out formatting, page limits and fees in detail is meaningful.

**→ Listed under "Ask the organizers" (ism@uci.edu). The delivered paper is built so
anonymization is a one-line switch** (`\ismanon` toggle in the preamble) if they confirm
double-blind.

---

## 4. Template

Source: https://www.multimediacomputing.org/submission.html

- "Manuscripts must be written in English and use the **IEEE manuscript formatting and
  templates**."
- "**double-column IEEE format**"
- Official template download link given by the conference:
  **https://www.ieee.org/conferences/publishing/templates.html**
- Camera-ready: "Final papers must follow **IEEE Computer Society Proceedings** manuscript
  formatting."
- "The official camera-ready upload link is currently **TBA**."

**Concrete class used here:** `\documentclass[conference]{IEEEtran}` — the standard IEEE
conference class (10pt, two-column, US Letter), which is what the IEEE templates page
serves for conference proceedings.

> ⚠️ **AMBIGUITY — for the organizers.** The submission page asks for *IEEE* format at
> submission but *IEEE Computer Society Proceedings* format at camera-ready. These are
> historically two different templates (`IEEEtran` vs. the CS `latex8`/compsoc style).
> `IEEEtran` with the `conference` option is the safe choice for review; confirm before
> camera-ready. Logged under "Ask the organizers."

**UNVERIFIED (not stated anywhere on the ISM 2026 site):** explicit font size, margin
measurements, or column widths beyond "double-column IEEE format". These are inherited
from whatever the IEEE template enforces.

---

## 5. Figures, tables, equations, references

**UNVERIFIED — ISM 2026 publishes no figure or table rules of its own.** Nothing on
`/submission.html` addresses DPI, color vs. grayscale, font embedding in figures, caption
style, or maximum figure width. The only instruction is the generic "Authors should
proofread and check the layout before submission."

Applied in this submission (IEEE conference convention, **not** an ISM-stated rule):
- Figures ≥300 DPI at final print size; vector where available.
- Captions below figures, above tables; sentence case; IEEE numbering (`Fig. 1`, `TABLE I`).
- Legible when printed grayscale — no information carried by color alone.
- References: **IEEE numeric style**, `\bibliographystyle{IEEEtran}`.
- Index terms via `\begin{IEEEkeywords}`.

**UNVERIFIED:** abstract word/character limit for regular papers. None is stated. (The
<2000-character limit is explicitly scoped to *poster abstracts* only.)

---

## 6. PDF requirements

- "**Only electronic PDF submission is accepted.**"
  — https://www.multimediacomputing.org/submission.html
- Page size: **UNVERIFIED**. Not stated. US Letter used (IEEE US default; the original
  paper is already 612×792 pt Letter).
- **PDF eXpress: UNVERIFIED.** No mention of IEEE PDF eXpress anywhere on the ISM 2026
  site. IEEE-published proceedings normally require it at *camera-ready*, not at
  submission. The camera-ready upload link is marked TBA, so this will likely appear
  later. Logged under "Ask the organizers."
- Embedded fonts / no Type 3 fonts: **UNVERIFIED** as an ISM rule, but it is a standard
  IEEE Xplore requirement and is verified on the delivered PDF in Phase 4 regardless.
- Max file size: **UNVERIFIED**. Not stated by ISM; EasyChair's general limit is commonly
  ~50 MB but that is not an ISM source. The delivered PDF is kept small as a precaution.

---

## 7. Policies

Source: https://www.multimediacomputing.org/submission.html unless noted.

| Policy | What it says |
|---|---|
| **Originality** | "Submissions must contain original material owned by the declared authors and must not be under review elsewhere." |
| **Dual submission** | "Simultaneous submission of the same paper to multiple tracks or conferences is not permitted." |
| **Presentation (mandatory)** | "Every paper accepted for publication in the proceedings must be presented during the conference." |
| **Registration (mandatory)** | "Every accepted paper must have at least one full member/nonmember registration attached to it. **If all authors are students, one student author must register at the full registration rate.**" |
| **Failure to comply** | "Failure to meet these requirements results in **removal from the conference proceedings**." |
| **IEEE ethics** | "Submissions are subject to IEEE publication, originality and copyright policies." → https://journals.ieeeauthorcenter.ieee.org/become-an-ieee-journal-author/publishing-ethics/ |

> ⚠️ **Applies directly to you.** All four authors on the original paper are students
> (registration numbers 23BRS1155 / 23BRS1152 / 23BRS1199 / 23BPS1158). Per the rule
> above, **one of you must register at the full, non-student rate.** Budget for this.

**UNVERIFIED — no ISM 2026 policy found on any of these:**
- arXiv / preprint policy
- AI-tool (LLM) use disclosure
- Ethics or human-subjects data statement requirements
- Reproducibility or artifact-evaluation track
- Copyright form mechanics (IEEE eCF presumably at camera-ready; not stated)

The ethics link above points to IEEE's general author policies, which do cover AI-tool
disclosure and human-subjects research at the IEEE level, but ISM 2026 states no
track-specific requirement.

> ⚠️ **Relevant to you:** GaitDeepfake-13 is human-subject video data collected by the
> authors and published on IEEE DataPort (DOI 10.21227/ngh5-b637). ISM states no ethics
> statement requirement, but IEEE's general policies apply. Logged under "Ask the
> organizers" — worth confirming whether a consent/IRB statement is expected.

---

## 8. Topics of interest

Source: https://www.multimediacomputing.org/scope.html — eight groups:

1. Systems and Architectures
2. Communications and Streaming
3. Multimedia Interfaces
4. Media Coding, Processing, and Quality Measurement
5. **Multimedia Security and Forensics** ← primary fit
6. **Content Understanding, Modeling, Management, and Retrieval** ← secondary fit
7. Mobile Media
8. Applications

**Bullets this paper lands on directly:**

Under *Multimedia Security and Forensics*:
- Face detection and recognition algorithms
- **Human behavior analysis from motion images/videos**
- Multimedia-based computer forensics (e.g., crime scene investigation, user profiling)
- **Forensic use of biometrics**
- Surveillance and monitoring methods

Under *Content Understanding, Modeling, Management, and Retrieval*:
- **Fake multimedia detection**
- Multimedia datasets and open source code for research
- Object, event, emotion, text detection and recognition

Under *Applications*:
- Deep learning of multimedia data

**No special sessions or named workshop tracks are listed on the ISM 2026 site as of
2026-09-20.** Workshop proposals closed Aug 1; workshop *papers* are due Oct 25. The
accepted workshop list is not published. **UNVERIFIED** whether a forensics-specific
workshop exists.

---

## Ask the organizers (ism@uci.edu)

Contact source: https://www.multimediacomputing.org/contact.html

1. **Is review single-blind or double-blind?** No anonymization instruction appears
   anywhere on the site. (Highest priority — changes the title block.)
2. **Which template governs camera-ready** — `IEEEtran` conference, or IEEE Computer
   Society Proceedings format? The submission page names both.
3. **Is IEEE PDF eXpress compliance required**, and at submission or camera-ready only?
4. **Is there a maximum PDF file size** on the EasyChair track?
5. **Is an ethics / informed-consent statement expected** for author-collected
   human-subject video data?
6. **Is there an arXiv/preprint policy**, and does prior IEEE DataPort publication of the
   dataset affect originality?
7. **Is a forensics/security workshop running**, given workshop papers are due Oct 25 —
   a better-fitting venue may exist within ISM.

---

## Fallback usage

**None required.** Every item above marked verified comes from an ISM **2026** page. ISM
2025 pages were not used as a substitute for any requirement.
