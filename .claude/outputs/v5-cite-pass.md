# v5-cite-pass — paper-writer report
Task + acceptance criterion: replace V5 PALM's hand-built `\item` bibliography and manual author-year prose strings with real BibTeX citations against the verified `refs.bib`; 0 `OTHER` hunks, 0 residual manual strings, clean build.
Status: **done** — with 4 reported items requiring an Advisor/Head ruling (nothing was "fixed").

## ⚠ FIRST TEN LINES — the things that must not be missed
1. **`pj_sub.tex` DID NOT MOVE.** Boot and end both `md5 2ee08554373bc90a53de1b60c91e8eba`, mtime `2026-08-26T03:21:38` unchanged. **The Advisor's copy-back is safe.** (At boot `paper.tex` and `pj_sub.tex` were byte-identical.)
2. **⛔ TASK PREMISE CORRECTION (§2b.4): the Jude et al. (2023) mention is NOT in the prose.** It existed **only as a `\item` in the hand-built list**. So "leave that one prose mention as a manual string" had no prose site to apply to, and "delete the list in full" would have deleted it — the two ⛔s collide. **I preserved it** (specific carve-out beats general instruction; deleting is the one irreversible act) as a one-item `itemize` after `\bibliography{refs}`. **This needs ratification** — it renders as a lone bullet after the reference list. Six-line deletion reverses it.
3. **Two references silently dropped from the printed bibliography:** `rusch_long_2022`, `wang_agentic_2026` are in `refs.bib` but cited nowhere in prose ⇒ BibTeX omits them. Printed refs went **31 hand-built → 28 generated + 1 retained Jude**. Per §2.4 I reported and did not fix. One-line restore: `\nocite{rusch_long_2022,wang_agentic_2026}`.
4. **`Bourtoule et al. 2021 → 2019`** on render: prose cited the 2021 IEEE S&P venue, the `.bib` carries the 2019 arXiv year. Also `Packer 2023 → 2024`. Self-correction working as designed, but Bourtoule's is a venue-vs-arXiv question owned by `refs.bib`, which this pass left byte-untouched.
5. Reconciliation list owner: items 2–4 above are for the **Shorts Advisor**, and item 4 may need a `refs.bib` edit pass.

## DIAL DECLARATION (echoed)
**Dials touched: NONE.** No experiment, config, registry or charter. One `.tex` file's citation apparatus + one build note. No new numbers, no new claims. Laundering control / falsifier: N/A (mechanical transformation, not a claim).

## Precondition check (§7.1)
`~/Desktop/V5_PALM_Submission/refs.bib` **EXISTS**, 30 `@` entries, `md5 5dd221b1f93642d15001a2ff6e69e4ff`, mtime 2026-08-26T02:28:48. Byte-untouched at end. Proceeded.

## What I did
- Converted **32 citation sites → 39 citation commands**: 19 `\citep`, 6 `\citet`, 3 `\citealp`, 5 `\citeauthor`, 6 `\citeyearpar`.
- Replaced the 31-item `\item` list with `\bibliographystyle{plainnat}` + `\bibliography{refs}`; **removed the manual `\section*{References}`** because natbib's generated bibliography emits its own heading (verified exactly 1 "References" heading in the PDF, per §2.2).
- Retained the Jude item verbatim (see item 2 above) with a two-line source comment explaining why.
- Wrote `BUILD-NOTE-CITE.md` into the submission folder **before** shipping the PDF (deliverable #1).
- Pre-pass file preserved at `.claude/outputs/v5-cite-pass/paper.tex.PREPASS`; conversion script at `.claude/outputs/v5-cite-pass/convert.py` (asserts an exact occurrence count per site and aborts on any mismatch).

## How I verified
**Diff contract (§4/§7.3)** — `difflib` opcode classification of every changed region:
```
replace 35    CITATION      replace 165->138  CITATION
replace 40    CITATION      replace 168->141  CITATION
replace 42    CITATION      replace 171->144  CITATION
replace 52    CITATION      replace 359->332  CITATION
replace 109   CITATION      replace 396->369  CITATION
replace 126   BIBLIOGRAPHY  delete 128-140    BIBLIOGRAPHY
delete 142-158 BIBLIOGRAPHY
-> CITATION 10 · BIBLIOGRAPHY 3 · OTHER 0
```
**Word-level byte-identity (§4, mandatory)** — for each site the citation portion was replaced by a sentinel in *both* old and new; remainders compared. **32/32 byte-identical, 0 violations.** Examples:
```
C01  'canonical §CITE§ placement'                 C17  'removal is §CITE§ Sec.~2'
C05  'introduced as CHLU by §CITE§'               C27  'is §CITE§, whose table'
C06  'Utilizing §CITE§ stable-matching'           C28  '§CITE§ had already'
C07  '(e.g., MemGPT, §CITE§; Mem0, §CITE§)'       C30b 'in a geometric setting is also taken §CITE§.'
```
**Residual manual strings (§7.2)** — author-year regex over `paper.tex`, excluding the retained Jude line: **0 matches**. **Positive control:** the identical regex against `paper.tex.PREPASS` returns **7** — the regex fires. Bare-arXiv-id sweep (`arXiv:` outside the Jude line): **0**.
**Build (§7.4)** — `/Library/TeX/texbin/pdflatex → bibtex → pdflatex → pdflatex`, in the Desktop folder only:
```
LaTeX errors 0 · undefined citations 0 · undefined references 0 · multiply-defined 0
BibTeX warnings 0 · bibitems 28 · Output written on paper.pdf (20 pages, 1790059 bytes)
```
**Page split (§5.5):** main text pp. 1–5 · References pp. 5–7 · Appendices pp. 7–20. (Measured, not acted on — page limits out of scope.)
**Integrity (§5.6):** `refs.bib` `5dd221b1f93642d15001a2ff6e69e4ff`, `neurips_2026.sty` `f447d3302c8719cb27619a074c876b44`, `figs/` tree digest `60da2c3140d5c07cf513e51dc109ff96` — all byte-untouched. `.claude/NIPSsubmission/v5-palm/**` untouched. Final `paper.tex` **`md5 f4c0ba8f9c1c45ee6f2c5887a3f921c0`**, mtime 2026-08-26T03:30:49 (boot mtime 03:21:45; no foreign edit observed mid-pass).

## Citation map (for one-word strike-out by the Head)
| # | line | command | key(s) | attaches to |
|---|---|---|---|---|
| 1 | 35 | `\citet` | blelloch_strongly_2007 | abstract: "under canonical … placement, store-level deletion is exact" |
| 2 | 40 | `\citep` | yang_control-plane_2026, uddin_recall_2026 | intro: "benchmarks often fail to adequately measure" |
| 3 | 42 | `\citep` | rasmussen_zep_2025, chhikara_mem0_2025 | intro: "superficial bookkeeping, such as a timestamp or a dropped row" |
| 4 | 42 | `\citep` | chakraborttii_ghost_2026, wang_memleak_2026 | intro: "leaves measurable residue … after the fact" |
| 5 | 52 | `\citet` | jawahar_chlu_2026 | **CLU continuity sentence** (charter-mandated, see below) |
| 6 | 109 | `\citeauthor`+`\citeyearpar` | blelloch_strongly_2007 | §deletion: "Utilizing … stable-matching table rules" |
| 7 | 138 | `\citealp` ×2 | packer_memgpt_2024, chhikara_mem0_2025 | app: "(e.g., MemGPT, …; Mem0, …)" |
| 8 | 138 | `\citealp` | munkhdalai_leave_2024 | app: "(e.g., Infini-attention, …)" |
| 9 | 138 | `\citep` | park_generative_2023 | app: "0.995 recency decay factor in ranking heuristics" |
| 10 | 138 | `\citep` | zhong_memorybank_2023 | app: "Ebbinghaus decay functions applied to memory strength" |
| 11 | 138 | `\citep` | sukhbaatar_not_2021 | app: "learned per-memory expiration spans" |
| 12 | 138 | `\citep` ×2 | behrouz_titans_2024 | app: "dynamic forget gates" and "the learned gate in Titans" |
| 13 | 141 | `\citep` | rasmussen_zep_2025 | app: "Zep invalidates contradicted edges with timestamps" |
| 14 | 141 | `\citep` | chakraborttii_ghost_2026 | app: "soft-deleted vectors … remain reconstructible" |
| 15 | 141 | `\citep` | yang_control-plane_2026 | app: "failures are predominantly forgetting failures" |
| 16 | 141 | `\citep` | uddin_recall_2026 | app: "outdated retention driving … recommendation errors" |
| 17 | 141 | `\citeauthor`+`\citeyearpar` | guo_certified_2023 | app: "certified removal is … Sec.~2, Eq.~(1)" |
| 18 | 141 | `\citep` | bourtoule_machine_2019, ginart_making_2019, sekhari_remember_2021 | app: "exact methods delete by isolation" |
| 19 | 141 | `\citep` | blelloch_strongly_2007 | app: "canonical geometric placement" |
| 20 | 144 | `\citep` | hochreiter_long_1997 | app: "learned recurrent models like LSTMs" |
| 21 | 144 | `\citep` | aitken_geometry_2022 | app: "representational drift … over time" |
| 22 | 144 | `\citep` | minami_spontaneous_2018 | app: "type-A Nambu–Goldstone modes … diffusive" |
| 23 | 144 | `\citep` | mo_symmetry-protected_2026 | app: "Lyapunov neutral modes" (was the bare string `(arXiv:2605.03338)`) |
| 24 | 332 | `\citet` ×2 | snyder_uniquely_1977, andersson_new_1995 | app: "canonical data structures date to … and …" |
| 25 | 332 | `\citeauthor`+`\citeyearpar` | micciancio_oblivious_1997 | app: "obliviousness … is Micciancio's" |
| 26 | 332 | `\citeauthor`+`\citeyearpar` | naor_anti-persistence_2001 | app: "weak and strong history-independence notions" |
| 27 | 332 | `\citeauthor`+`\citeyearpar` | blelloch_strongly_2007 | app: "the open-addressed realization is …, whose table" |
| 28 | 332 | `\citet` | hutchison_uniquely_2008 | app: "… had already taken unique representation into computational geometry" |
| 29 | 332 | `\citep` | goos_lower_2003 | app: "exponential slowdown relative to the weak notion" |
| 30 | 369 | `\citet` | blelloch_strongly_2007 | negatives table: "own it outright" |
| 31 | 369 | `\citeyearpar` | hutchison_uniquely_2008 | negatives table: "is also taken (2008)" |

⚠ **Two keys are named after LNCS series editors, wired by author not by key** (task §2b.1, confirmed against the `author` fields): `goos_lower_2003` = **Buchbinder & Petrank** (CRYPTO 2003) · `hutchison_uniquely_2008` = **Blelloch, Golovin & Vassilevska** (SWAT 2008). Not renamed.

## Uncited entries / unmatched mentions (§2.4 — reported, not fixed)
- **Uncited `.bib` entries (2):** `rusch_long_2022`, `wang_agentic_2026`. Computed as `comm -23` of all 30 keys against the 28 keys in `paper.bbl`.
- **Prose mention with no `.bib` key (1):** Jude, Perich, Miller & Hennig (2023) — the single known exception of §2b.4, except that it is a **bibliography-line** mention, not a prose one (see first-ten-lines item 2).

## Part B — the three decided items, honoured
1. **No `Anonymous (2026)` theory-note entry**, no supplementary reference, no "provided in the supplementary material" note. `refs.bib` contains no such entry; none added.
2. **No `\TODO` tag re-created.** ⚠ Note for the Hub: `tasks/v5-derivation-appendix.md`'s output **has already landed** in the build copy — `\section{Derivations of the Closed Forms}\label{app:derivation}` (pp. 18–20) with `\label`s `app:deriv:1…8`. I neither added nor altered any reference to it; the pre-existing `\ref{app:derivation}` cross-references were already in the text and resolve (0 undefined references).
3. **CLU continuity sentence kept, parenthetical converted only:** "Our reference architecture is the Causal Learning Unit (CLU), introduced as CHLU by `\citet{jawahar_chlu_2026}`." Renders as *"introduced as CHLU by Jawahar and Pierini [2026]"* — third-person self-citation preserved, not anonymized. C-1(d) satisfied: J&P 2026 is cited for the primitive's introduction only.

⚠ Rendering consequence pre-authorized by §3, confirmed present and **not stripped**: `\citep{mo_symmetry-protected_2026}` renders **"[Mo, 2026]"** in body text. "Morse"/"Moser" do not occur in the file; no sweep collision.

## Rendering deltas (style-driven, from the verified records — none is an `OTHER` edit)
| class | before (hand-typed) | after (plainnat) |
|---|---|---|
| delimiters | `(Park et al., 2023)` | `[Park et al., 2023]` |
| multi-cite separator | `;` | `,` |
| ampersand | `Blelloch \& Golovin` | `Blelloch and Golovin` |
| 3-author `\citet` | `Blelloch, Golovin \& Vassilevska (2008)` | `Blelloch et al. [2008]` |
| year self-correction | `Packer et al., 2023` / `Bourtoule et al., 2021` | `Packer et al., 2024` / `Bourtoule et al., 2019` |

Cause of the brackets: `neurips_2026.sty` loads `natbib` **with no options**, and natbib 8.31b's no-option default punctuation is `[ ]`+`,` even under an author-year `.bst`. Internally consistent and ordinary NeurIPS appearance, so I did **not** add `\setcitestyle` (that would be an unauthorized preamble edit). One line restores the old look if the Head wants it: `\setcitestyle{round,semicolon}`.

## Deviation from the letter of the task (one, declared)
§2.3 names `\citet`/`\citep`/`\citealp` only. The **5 possessive sites** cannot be rendered by any of those three without reordering words (`\citet` yields "Blelloch and Golovin [2007]'s", not "Blelloch and Golovin's [2007]"), which the §4 diff contract forbids. I used `\citeauthor{k}'s \citeyearpar{k}`, which is still natbib and still sources author+year from `refs.bib` — preserving the ⭐ self-correction property. Verified in the PDF: `Micciancio's [1997]`, `Naor and Teague's [2001]`, `Blelloch and Golovin's [2007]`, `Guo et al.'s [2020]`.

## Git footprint
**None.** All edits are to `~/Desktop/V5_PALM_Submission/paper.tex` (untracked build copy) and `.claude/**` (gitignored). No branch, no commit. `.claude/NIPSsubmission/v5-palm/**` byte-untouched.

## Open questions / follow-ups / risks
1. **Ratify or reverse the Jude retention** (first-ten-lines item 2). It is the only cosmetically odd thing in the PDF.
2. **Rule on the 2 dropped references** — `\nocite` them, cite them in prose, or accept the drop.
3. **`refs.bib` may need a year pass** for `bourtoule_machine_2019` (2019 arXiv vs 2021 S&P venue) and `packer_memgpt_2024` (2024 vs the 2023 arXiv posting the prose used). Out of scope here — `refs.bib` was left byte-untouched.
4. **Bracket vs parenthesis citation style** — a one-line Head decision.
5. Copy-back is safe as of this report's timestamp; `pj_sub.tex` had not moved. Re-check its md5 immediately before copying.

## Proposed handover updates (for the Hub)
- `v5-cite-pass` **done** 2026-08-26. `~/Desktop/V5_PALM_Submission/paper.tex` now carries a real BibTeX apparatus (39 citation commands, 32 sites) against the 30-entry `refs.bib`; builds `0 errors · 0 undefined citations · 0 undefined references · 0 BibTeX warnings`, 20 pages (main 1–5 / refs 5–7 / appendices 7–20). Final `md5 f4c0ba8f9c1c45ee6f2c5887a3f921c0`. `OTHER` hunks = 0, word-level byte-identity 32/32.
- **`pj_sub.tex` unmoved through the pass** (`2ee08554373bc90a53de1b60c91e8eba`) — Advisor copy-back is safe pending a fresh md5 check.
- **Four open rulings** carried in `BUILD-NOTE-CITE.md` §"Four things": Jude retention, 2 uncited-and-dropped refs, Bourtoule/Packer year provenance in `refs.bib`, bracket citation style.
- **Task-premise correction for the record:** the Jude reference was never a prose mention — §2b.4's factual premise was wrong, and the "single known exception to zero-manual-strings" is a bibliography line, not prose. Future task scoping for this file should not assume it.
- The derivation appendix (`tasks/v5-derivation-appendix.md`) has **already landed** in the build copy as App. `app:derivation`, pp. 18–20.
