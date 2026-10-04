# FINAL-INDEX — the closing audit of PW-2 (both drafts, as installed after pw2-draft-leftovers)

Session `pw2-draft-final`, branch `pw2/draft-final`, batmac. Read-only audit of `~/clu/pw2/live/v2/DRAFT-v2.tex` and `~/clu/pw2/live/style/DRAFT-v2-style.tex`. **No content edit, no mechanical fix needed, nothing installed.** Working folder: `final-20260925T180521Z`.

## Summary: the eleven points of the brief

| # | point | DRAFT-v2 | DRAFT-v2-style |
|---|---|---|---|
| 1 | full compile from a clean copy | exit 0 on every pass, 0 undefined citations, 0 undefined references, 0 errors, **0 overfull**, 16 underfull, 56 bibitems, no Rerun, **42 pages** | same, 20 underfull, **43 pages** |
| | bibtex warnings (verbatim) | `Warning--empty journal in ha_world_2018` (1) | the same (1) |
| | mechanical defects | none: no `??`, no machine path, 2-line title, no doubled word in the source, no forced break in prose | none |
| 2 | style diff-check | number multiset **3,362 = 3,362** (1,758 distinct), lost ∅, added ∅; placeholders, artefact paths (607), section order (100 headings, 67 labels): **ALL EMPTY** | |
| 3 | provenance vs PW-1 final | **IDENTICAL**: 3 unsourced lines, 9 not-found tokens, same findings (line numbers masked) | **IDENTICAL**: 3 and 10 |
| 4 | figure/table index | 13 figures, 20 tables (34 labels: Tab. 19 carries `tab:cmazet` and `tab:cmaze-del`); every label agrees with its lof/lot entry | same numbers; 31 floats sit one page later or earlier |
| 5 | pages | main text (p. 1 → AI use statement) **17.95 pp**, **+8.95 over 9** | **18.22 pp**, **+9.22 over 9** |
| 6 | numbers left in the main text | 383 tokens (198 digit, 185 word); 91 Head-held; **0 UNCLASSIFIED**; 32 rounded phrases re-checked, **0 FAIL**; abstract 5 quantities | 381 tokens; 0 UNCLASSIFIED; 0 FAIL |
| | against pw2-draft-numbers | one number its list missed: "more than **eighteen** tuples" (a rounded floor, record holds); every other difference is tokeniser or taxonomy | the same |
| 7 | tells now (T1…T7 totals) | 5 · 19 · 5 · 37 · 0 · 7 · 52 (tells' after: 5 · 20 · 5 · 48 · 0 · 7 · 51) | 5 · 18 · 4 · 37 · 0 · 7 · 54 (after: 5 · 19 · 4 · 48 · 0 · 7 · 53) |
| 8 | Head-held paragraphs | 7 of 7 byte-equal (sha256 and raw lines) | 7 of 7 |
| 9 | `\TODO` / `\PLACEHOLDER` | 3 / 0: maze sentence (abstract, Head-held), AI use statement, code archive | 3 / 0, identical arguments |
| 10 | `jawahar2026chlu` scope | 3 citations (Methods ×2: the unit; Discussion 5.3: the unit's zero-dynamics limit); **0 VIOLATION** | the same |
| 11 | bib sha256 | `b2cf715f…` unchanged, 56 entries, at start and end | the same |

No content edit and no mechanical fix: nothing needed fixing, so nothing was written to the live folders. Every scan has a positive control that fires (11 of 11, end of this file).


## 1. Full compile of both live folders, from a clean copy

Built from a copy of `~/clu/pw2/live/{v2,style}` without any build products (`.aux .log .bbl .blg .out .pdf` excluded), FROM the folder: pdflatex, bibtex, pdflatex until no "Rerun" (`pw2_draft_sections/compile_folder.sh`). The copied `.tex` files are cmp-equal to the live ones. The extracted text of the fresh builds equals, page by page, the text of the PDFs installed in the live folders (`checks/LIVE-PDF-VS-BUILD.txt`).

| draft | exit codes | undefined citations | undefined references | errors | multiply defined | overfull | underfull | bibitems | Rerun left | pages |
|---|---|---|---|---|---|---|---|---|---|---|
| DRAFT-v2 | pdflatex1=0 bibtex=0 later_pdflatex=[ 0 0 0 ] passes=4 | 0 | 0 | 0 | 0 | 0 | 16 | 56 | 0 | 42 |
| DRAFT-v2-style | pdflatex1=0 bibtex=0 later_pdflatex=[ 0 0 0 ] passes=4 | 0 | 0 | 0 | 0 | 0 | 20 | 56 | 0 | 43 |

**Overfull boxes: none in either draft.** Underfull boxes, every one (the log line and the source lines it names; "lines 66--73" are in the `.bbl` (the Efron 1979 entry), every other line range is in the `.tex`; lines 295--304 = Methods "The read", 428/426--438/436 = Experiments "Baselines", the rest are appendix table cells):

- DRAFT-v2 (16):
  - `Underfull \hbox in paragraph at lines 66--73` ×4
  - `Underfull \vbox has occurred while \output is active []` ×3
  - `Underfull \hbox in paragraph at lines 295--304` ×3
  - `Underfull \hbox in paragraph at lines 428--438`
  - `Underfull \hbox in paragraph at lines 1655--1655`
  - `Underfull \hbox in paragraph at lines 2358--2358`
  - `Underfull \hbox in paragraph at lines 2361--2361`
  - `Underfull \hbox in paragraph at lines 2369--2369`
  - `Underfull \hbox in paragraph at lines 2374--2374`
- DRAFT-v2-style (20):
  - `Underfull \vbox has occurred while \output is active []` ×7
  - `Underfull \hbox in paragraph at lines 66--73` ×4
  - `Underfull \hbox in paragraph at lines 295--304` ×3
  - `Underfull \hbox in paragraph at lines 426--436`
  - `Underfull \hbox in paragraph at lines 1643--1643`
  - `Underfull \hbox in paragraph at lines 2346--2346`
  - `Underfull \hbox in paragraph at lines 2349--2349`
  - `Underfull \hbox in paragraph at lines 2357--2357`
  - `Underfull \hbox in paragraph at lines 2362--2362`

BibTeX warnings, verbatim (`.blg`), both drafts identical:

- DRAFT-v2: `Warning--empty journal in ha_world_2018` · `(There was 1 warning)`
- DRAFT-v2-style: `Warning--empty journal in ha_world_2018` · `(There was 1 warning)`

Other LaTeX / package warnings (counts by kind, both drafts):

- DRAFT-v2: LaTeX Warning: `…' float specifier changed to `…'. ×12; Package hyperref Warning: Token not allowed in a PDF string (Unicode): ×10; LaTeX Warning: You have requested package `…', ×1
- DRAFT-v2-style: LaTeX Warning: `…' float specifier changed to `…'. ×11; Package hyperref Warning: Token not allowed in a PDF string (Unicode): ×10; LaTeX Warning: You have requested package `…', ×1

**Mechanical defects** (`scans.py defects`): 0 rendered `??`, 0 undefined-citation marks, 0 absolute machine paths in either `.tex` (code and comments), the title renders on 2 lines in both drafts, 0 doubled words in the source, 0 forced `\\` breaks in prose. The rendered-text scan does list doubled words, all of them inside figure panels or adjacent table cells (text-extraction joins, not source text; `checks/DEFECTS-*.md`). **No defect to fix, so no edit was made and nothing was installed.**

## 2. Style diff-check between the drafts (`pw1_draft_sweeps/style_diff_check.py`, unchanged; exit 0)

| check | DRAFT-v2 | DRAFT-v2-style | diff |
|---|---|---|---|
| number multiset (every digit run, whole file, code only) | 3362 tokens (1758 distinct) | 3362 tokens (1758 distinct) | lost (empty), added (empty): **EMPTY** |
| `\PLACEHOLDER` arguments, in order | 0 | 0 | **EMPTY** |
| artefact paths in `% SRC` comments | 607 (80 distinct) | 607 (80 distinct) | **EMPTY** |
| heading sequence (section … paragraph) | 100 | 100 | identical: **True** |
| `\label` sequence | 67 | 67 | identical: **True** |

All four diffs EMPTY (`checks/STYLE-DIFF-CHECK.md`). The main-text digit tokens (this audit's scanner, §6) are also equal between the drafts, except one `3` inside the name "Shapes3D" in the style twin's own Head-held Experiments sentence ("This section measures the three properties on Shapes3D.").

## 3. Provenance findings against the PW-1 final state

`pw1_draft_sweeps/check_provenance.py` (unchanged) run now on the live drafts and on the PW-1 final drafts (`outputs/pw1-draft-citations/stage-145204/post-install/`, sha256 = `SHA256-installed.txt` there = `outputs/w17-paper/`). Findings compared as multisets with line numbers masked (`prov_compare.py`).

#### DRAFT-v2
| | PW-1 final | now |
|---|---|---|
| checked number tokens | 2597 | 2342 |
| unsourced lines | 3 | 3 |
| not-found tokens | 9 | 9 |

- findings in both (line numbers masked): 12
- only now (regressions): none
- only in PW-1 (resolved since): none
- verdict: **IDENTICAL**

Findings now (line numbers masked):
- `['027']`
- `['027', '027', '027']`
- `['027', '027']`
- `8,10,12,16,20,24,28,32 not in w13-seeds-and-tables/TABLES.md`
- `12,310,672 not in w15-table-fixes/TABLE-FIXES.md`
- `11,941,360 not in w15-table-fixes/TABLE-FIXES.md`
- `11,850,912 not in w15-table-fixes/TABLE-FIXES.md`
- `16,424 not in w15-table-fixes/TABLE-FIXES.md`
- `11,488 not in w15-table-fixes/TABLE-FIXES.md`
- `5,288 not in w15-table-fixes/TABLE-FIXES.md`
- `1,322 not in w15-table-fixes/TABLE-FIXES.md`
- `106 not in w23-seed5-diagnosis/FINDINGS.md; w23-seed5-diagnosis/Q1-CONTENT.json`
#### DRAFT-v2-style
| | PW-1 final | now |
|---|---|---|
| checked number tokens | 2597 | 2342 |
| unsourced lines | 3 | 3 |
| not-found tokens | 10 | 10 |

- findings in both (line numbers masked): 13
- only now (regressions): none
- only in PW-1 (resolved since): none
- verdict: **IDENTICAL**

Findings now (line numbers masked):
- `['027']`
- `['027', '027', '027']`
- `['027', '027']`
- `8,10,12,16,20,24,28,32 not in w13-seeds-and-tables/TABLES.md`
- `12,310,672 not in w15-table-fixes/TABLE-FIXES.md`
- `11,941,360 not in w15-table-fixes/TABLE-FIXES.md`
- `11,850,912 not in w15-table-fixes/TABLE-FIXES.md`
- `16,424 not in w15-table-fixes/TABLE-FIXES.md`
- `11,488 not in w15-table-fixes/TABLE-FIXES.md`
- `5,288 not in w15-table-fixes/TABLE-FIXES.md`
- `1,322 not in w15-table-fixes/TABLE-FIXES.md`
- `16,384 not in w15-table-fixes/TABLE-FIXES.md`
- `106 not in w23-seed5-diagnosis/FINDINGS.md; w23-seed5-diagnosis/Q1-CONTENT.json`

No change in either draft: the same 12 (v2) / 13 (style) findings, all present at PW-1 final. The three "unsourced" lines are the `2027` of `iclr2027` in the preamble (`\input@path`, the `\RequirePackage` line) and the `\bibliographystyle` line; the not-found tokens are strings their cited file does not print verbatim (the K-grid list, the App. E byte totals, and the seed-5 count 106, which moved with item 5 into Tab. cmazet-sym). Checked number tokens fell 2,597 → 2,342 because numbers left the prose in item 5.

## 4. Figure / table index (number → label → caption lead → page → section)

From the compiled `.aux` (`pw1_draft_final/fig_table_index.py`, unchanged): every label cross-checked against its list-of-figures/tables entry (caption start and page agree), and the label set equals the lof/lot set.

### DRAFT-v2: 13 figures, 20 tables (34 labels; shared numbers: table 19 carries `tab:cmazet` and `tab:cmaze-del`); lof/lot disagreements 0; lof/lot without label none; label without lof/lot none

| kind | number | label | caption lead | page | section |
|---|---|---|---|---|---|
| figure | 1 | `fig:task-shapes` | The Shapes3D task. | 5 | 3 Experiments |
| figure | 2 | `fig:task-cmazet` | The CMAZE-T task. | 6 | 3 Experiments |
| figure | 3 | `fig:shapes-per-model` | Each model succeeding and failing on held-out combinations (Shapes3D, seed 0, K* =32). | 9 | 4.1 Composition on Shapes3D |
| figure | 4 | `fig:shapes-baseline-wins` | Held-out requests on which a baseline succeeds and the n-CLU fails (seed 0). | 10 | 4.1 Composition on Shapes3D |
| figure | 5 | `fig:ladder` | The ladder at K* against data-matched baselines, 3 seeds. | 12 | 4.2 The n-way ladder |
| figure | 6 | `fig:crossover` | The K-curve, validation, 3 seeds at every K. | 27 | D The K-curve: composition and the memory  |
| figure | 7 | `fig:deletion` | Deletion on the single design at K* , rows 3-4, test, fit pool 131,072 rows, seeds 0, 1 and 2. | 29 | F The earlier composition cell: byte and p |
| figure | 8 | `fig:adapt` | Encoder adaptation margins. | 31 | H Light encoder adaptation in full |
| figure | 9 | `fig:maze-wm` | CMAZE-T: the n-CLU against the best matched baseline under four input choices, and six seeds without and with  | 36 | N Maze world-model rows |
| figure | 10 | `fig:cmazet-families` | Held-out CMAZE-T transitions, one per rule family, with every model's next state (words plus observation, test | 38 | N Maze world-model rows |
| figure | 11 | `fig:cmazet-per-arm` | Each CMAZE-T model succeeding and failing (words plus observation, test, seed 0). | 39 | N Maze world-model rows |
| figure | 12 | `fig:cmazet-rel-per-arm` | Each CMAZE-T model succeeding and failing with relational coding (words plus observation, test). | 40 | N Maze world-model rows |
| figure | 13 | `fig:cmazet-door-tint` | Door-then-tint transitions: the move opens a door with the key in the pocket and enters a tint of the same col | 41 | N Maze world-model rows |
| table | 1 | `tab:t7` | Composition at K* =32 against data-matched baselines, Shapes3D value hold-out, TEST, n=3 seeds (0, 1, 2). | 8 | 4.1 Composition on Shapes3D |
| table | 2 | `tab:ledger-kstar` | The footprint breakdown of the K* =32 object, per seed, never netted. | 10 | 4.1 Composition on Shapes3D |
| table | 3 | `tab:ladder` | The ladder at K* =32, n=3 seeds, distinct valid novel tuples per combination, per seed (0 / 1 / 2). | 11 | 4.2 The n-way ladder |
| table | 4 | `tab:del` | Deletion on the single design, the four measurements, K* =32, rows 3-4, TEST, fit pool 131,072 rows, one row p | 13 | 4.3 Deletion on Shapes3D |
| table | 5 | `tab:hole` | Refusal DISPLAY rows. A display is never a result. | 14 | 4.5 Refusal through the hole |
| table | 6 | `tab:gentable-kstar` | The generation table, K* single design, one declared evaluation run, generation-2 cache. | 25 | B Tab. [tab:t7]: what it does and does not |
| table | 7 | `tab:lace-poe` | LACE and PoE at their own settings, Shapes3D 3dshapes18,kim_disentangling_2019 value hold-out, TEST, n=3 seeds | 25 | B Tab. [tab:t7]: what it does and does not |
| table | 8 | `tab:k5` | Ladder at K=5, the sweep's endpoint, rebuilt n-CLU, 3 seeds. | 26 | C The K=5 endpoint of the sweep |
| table | 9 | `tab:kcurve` | The K-curve, validation, symmetric-triangle convention. | 26 | D The K-curve: composition and the memory  |
| table | 10 | `tab:kcurve3` | The K-curve at n=3 seeds on every row, validation (a smoke, not a claim). | 27 | D The K-curve: composition and the memory  |
| table | 11 | `tab:impl` | Deletion on Shapes3D, matched on the n-CLU, implementations 1 and 2, bands not shared. | 28 | E Deletion on Shapes3D: two implementation |
| table | 12 | `tab:ablations` | The two ablations, earlier composition cell, TEST, n=3 seeds, one pin (524,288-byte n-CLU), K=5. | 29 | F The earlier composition cell: byte and p |
| table | 13 | `tab:rivals-del` | Deletion cost for the data-matched baselines: retrain without the rows, on the n-CLU's 131,072-row pool, at th | 30 | G Deletion comparators in full (fit pool,  |
| table | 14 | `tab:rivals-del-seeds` | The n-CLU after deletion against each baseline retrained without the same rows, all 109 combinations, seeds 0  | 30 | G Deletion comparators in full (fit pool,  |
| table | 15 | `tab:adapt` | Encoder adaptation at K* , TEST, 3 seeds x 3 configurations (nine arm-runs, three seeds). | 30 | H Light encoder adaptation in full |
| table | 16 | `tab:encloop` | Encoder loop, shortened decoder (400 joint steps, 8,192 images), 3 seeds, one declared test run. | 31 | I The encoder loop in full (shortened-deco |
| table | 17 | `tab:bard-gpu` | The unfreezing decomposition on the longer leg only (4,000 joint steps, 32,768 training images, 1,800 decoder- | 32 | I The encoder loop in full (shortened-deco |
| table | 18 | `tab:crr` | Declared criteria of the retention law on the field-mode n-CLU, 49 held-out combinations x 3 seeds, 192,975 ro | 33 | K The retention law on the field-mode n-CL |
| table | 19 | `tab:cmazet` | Maze world-model rows: CMAZE-T (transition generation), TEST, base n-CLU, 3 seeds (one declared test run per s | 37 | N Maze world-model rows |
| table | 19 | `tab:cmaze-del` | Maze world-model rows: CMAZE-T (transition generation), TEST, base n-CLU, 3 seeds (one declared test run per s | 37 | N Maze world-model rows |
| table | 20 | `tab:cmazet-sym` | CMAZE-T with symmetric inputs: every condition, unseen n-way, the engine's judge. | 42 | N Maze world-model rows |

### DRAFT-v2-style: 13 figures, 20 tables (34 labels; shared numbers: table 19 carries `tab:cmazet` and `tab:cmaze-del`); lof/lot disagreements 0; lof/lot without label none; label without lof/lot none

| kind | number | label | caption lead | page | section |
|---|---|---|---|---|---|
| figure | 1 | `fig:task-shapes` | The Shapes3D task. | 5 | 3 Experiments |
| figure | 2 | `fig:task-cmazet` | The CMAZE-T task. | 6 | 3 Experiments |
| figure | 3 | `fig:shapes-per-model` | Each model succeeding and failing on held-out combinations (Shapes3D, seed 0, K* =32). | 10 | 4.1 Composition on Shapes3D |
| figure | 4 | `fig:shapes-baseline-wins` | Held-out requests on which a baseline succeeds and the n-CLU fails (seed 0). | 11 | 4.1 Composition on Shapes3D |
| figure | 5 | `fig:ladder` | The ladder at K* against data-matched baselines, 3 seeds. | 13 | 4.2 The n-way ladder |
| figure | 6 | `fig:crossover` | The K-curve, validation, 3 seeds at every K. | 28 | D The K-curve: composition and the memory  |
| figure | 7 | `fig:deletion` | Deletion on the single design at K* , rows 3-4, test, fit pool 131,072 rows, seeds 0, 1 and 2. | 30 | F The earlier composition cell: byte and p |
| figure | 8 | `fig:adapt` | Encoder adaptation margins. | 32 | H Light encoder adaptation in full |
| figure | 9 | `fig:maze-wm` | CMAZE-T: the n-CLU against the best matched baseline under four input choices, and six seeds without and with  | 37 | N Maze world-model rows |
| figure | 10 | `fig:cmazet-families` | Held-out CMAZE-T transitions, one per rule family, with every model's next state (words plus observation, test | 39 | N Maze world-model rows |
| figure | 11 | `fig:cmazet-per-arm` | Each CMAZE-T model succeeding and failing (words plus observation, test, seed 0). | 40 | N Maze world-model rows |
| figure | 12 | `fig:cmazet-rel-per-arm` | Each CMAZE-T model succeeding and failing with relational coding (words plus observation, test). | 41 | N Maze world-model rows |
| figure | 13 | `fig:cmazet-door-tint` | Door-then-tint transitions: the move opens a door with the key in the pocket and enters a tint of the same col | 42 | N Maze world-model rows |
| table | 1 | `tab:t7` | Composition at K* =32 against data-matched baselines, Shapes3D value hold-out, TEST, n=3 seeds (0, 1, 2). | 8 | 4.1 Composition on Shapes3D |
| table | 2 | `tab:ledger-kstar` | The footprint breakdown of the K* =32 object, per seed, never netted. | 9 | 4.1 Composition on Shapes3D |
| table | 3 | `tab:ladder` | The ladder at K* =32, n=3 seeds, distinct valid novel tuples per combination, per seed (0 / 1 / 2). | 12 | 4.2 The n-way ladder |
| table | 4 | `tab:del` | Deletion on the single design, the four measurements, K* =32, rows 3-4, TEST, fit pool 131,072 rows, one row p | 14 | 4.3 Deletion on Shapes3D |
| table | 5 | `tab:hole` | Refusal DISPLAY rows. A display is never a result. | 15 | 4.5 Refusal through the hole |
| table | 6 | `tab:gentable-kstar` | The generation table, K* single design, one declared evaluation run, generation-2 cache. | 26 | B Tab. [tab:t7]: what it does and does not |
| table | 7 | `tab:lace-poe` | LACE and PoE at their own settings, Shapes3D 3dshapes18,kim_disentangling_2019 value hold-out, TEST, n=3 seeds | 26 | B Tab. [tab:t7]: what it does and does not |
| table | 8 | `tab:k5` | Ladder at K=5, the sweep's endpoint, rebuilt n-CLU, 3 seeds. | 27 | C The K=5 endpoint of the sweep |
| table | 9 | `tab:kcurve` | The K-curve, validation, symmetric-triangle convention. | 27 | D The K-curve: composition and the memory  |
| table | 10 | `tab:kcurve3` | The K-curve at n=3 seeds on every row, validation (a smoke, not a claim). | 28 | D The K-curve: composition and the memory  |
| table | 11 | `tab:impl` | Deletion on Shapes3D, matched on the n-CLU, implementations 1 and 2, bands not shared. | 29 | E Deletion on Shapes3D: two implementation |
| table | 12 | `tab:ablations` | The two ablations, earlier composition cell, TEST, n=3 seeds, one pin (524,288-byte n-CLU), K=5. | 30 | F The earlier composition cell: byte and p |
| table | 13 | `tab:rivals-del` | Deletion cost for the data-matched baselines: retrain without the rows, on the n-CLU's 131,072-row pool, at th | 31 | G Deletion comparators in full (fit pool,  |
| table | 14 | `tab:rivals-del-seeds` | The n-CLU after deletion against each baseline retrained without the same rows, all 109 combinations, seeds 0  | 31 | G Deletion comparators in full (fit pool,  |
| table | 15 | `tab:adapt` | Encoder adaptation at K* , TEST, 3 seeds x 3 configurations (nine arm-runs, three seeds). | 31 | H Light encoder adaptation in full |
| table | 16 | `tab:encloop` | Encoder loop, shortened decoder (400 joint steps, 8,192 images), 3 seeds, one declared test run. | 32 | I The encoder loop in full (shortened-deco |
| table | 17 | `tab:bard-gpu` | The unfreezing decomposition on the longer leg only (4,000 joint steps, 32,768 training images, 1,800 decoder- | 33 | I The encoder loop in full (shortened-deco |
| table | 18 | `tab:crr` | Declared criteria of the retention law on the field-mode n-CLU, 49 held-out combinations x 3 seeds, 192,975 ro | 34 | K The retention law on the field-mode n-CL |
| table | 19 | `tab:cmazet` | Maze world-model rows: CMAZE-T (transition generation), TEST, base n-CLU, 3 seeds (one declared test run per s | 38 | N Maze world-model rows |
| table | 19 | `tab:cmaze-del` | Maze world-model rows: CMAZE-T (transition generation), TEST, base n-CLU, 3 seeds (one declared test run per s | 38 | N Maze world-model rows |
| table | 20 | `tab:cmazet-sym` | CMAZE-T with symmetric inputs: every condition, unseen n-way, the engine's judge. | 43 | N Maze world-model rows |

Numbering identical between the drafts: **True**. Pages differ for 31 floats (the style twin is one page longer): `fig:shapes-per-model` 9/10, `fig:shapes-baseline-wins` 10/11, `tab:ledger-kstar` 10/9, `tab:ladder` 11/12, `fig:ladder` 12/13, `tab:del` 13/14, `tab:hole` 14/15, `tab:gentable-kstar` 25/26, `tab:lace-poe` 25/26, `tab:k5` 26/27, `tab:kcurve` 26/27, `fig:crossover` 27/28, `tab:kcurve3` 27/28, `tab:impl` 28/29, `tab:ablations` 29/30, `fig:deletion` 29/30, `tab:rivals-del` 30/31, `tab:rivals-del-seeds` 30/31, `tab:adapt` 30/31, `fig:adapt` 31/32, `tab:encloop` 31/32, `tab:bard-gpu` 32/33, `tab:crr` 33/34, `fig:maze-wm` 36/37, `tab:cmazet` 37/38, `tab:cmaze-del` 37/38, `fig:cmazet-families` 38/39, `fig:cmazet-per-arm` 39/40, `fig:cmazet-rel-per-arm` 40/41, `fig:cmazet-door-tint` 41/42, `tab:cmazet-sym` 42/43.

## 5. Page counts per section (marker method C, `pw2_draft_sections/page_markers_c.py` + `pw1_draft_final/pages.py`)

Exact heading positions at shipout (a `\pdfsavepos` marker inside each heading). The marker builds reproduce every `\newlabel` of the unmarked build and 0 toc/marker page mismatches, in both drafts. "len" = text-flow length in pages from the heading to the next heading of the same or a higher level (floats count where they land). **Main text** = from the top of page 1 (title, abstract) to the "AI use statement" heading, as in PW-1 (statements, references and appendices do not count toward ICLR's limit).

| | DRAFT-v2 | DRAFT-v2-style |
|---|---|---|
| total pages | 42 | 43 |
| **main text (p. 1 top → AI use statement)** | **17.95 pp** (ends on p. 18) | **18.22 pp** (ends on p. 19) |
| main text against the 9-page limit | **+8.95 pp over** | **+9.22 pp over** |
| of which title + abstract (p. 1 top → Introduction) | 0.86 pp | 0.86 pp |
| Introduction → AI use statement | 17.10 pp | 17.36 pp |

| # | heading | DRAFT-v2: page, len (pp) | DRAFT-v2-style: page, len (pp) |
|---|---|---|---|
| 1 | **Introduction** | 1, 1.93 | 1, 1.97 |
| 1.1 | &nbsp;&nbsp;Positioning among naive units | 1, 0.98 | 1, 0.97 |
| 1.2 | &nbsp;&nbsp;The three properties | 2, 0.28 | 2, 0.36 |
| 1.3 | &nbsp;&nbsp;Contributions | 3, 0.62 | 3, 0.60 |
| 2 | **Methods** | 3, 2.03 | 3, 2.00 |
| 3 | **Experiments** | 5, 2.58 | 5, 2.64 |
| 3.1 | &nbsp;&nbsp;The Shapes3D value hold-out | 5, 1.08 | 5, 1.15 |
| 3.2 | &nbsp;&nbsp;Baselines and the matching rule | 6, 0.36 | 7, 0.35 |
| 3.3 | &nbsp;&nbsp;The $n$-way ladder protocol | 7, 0.20 | 7, 0.19 |
| 3.4 | &nbsp;&nbsp;The deletion protocol | 7, 0.27 | 7, 0.26 |
| 3.5 | &nbsp;&nbsp;The CMAZE-T transition task | 7, 0.64 | 7, 0.66 |
| 4 | **Results** | 8, 8.57 | 8, 8.69 |
| 4.1 | &nbsp;&nbsp;Composition on Shapes3D | 8, 2.34 | 8, 0.84 |
| 4.2 | &nbsp;&nbsp;The $n$-way ladder | 10, 2.08 | 9, 2.26 |
| 4.3 | &nbsp;&nbsp;Deletion on Shapes3D | 12, 1.55 | 11, 2.02 |
| 4.4 | &nbsp;&nbsp;Deletion against retrained baselines | 14, 0.29 | 13, 0.29 |
| 4.5 | &nbsp;&nbsp;Refusal through the hole | 14, 0.38 | 13, 1.34 |
| 4.6 | &nbsp;&nbsp;Two ablations | 15, 0.37 | 15, 0.39 |
| 4.7 | &nbsp;&nbsp;Encoder adaptation and generative baselines | 15, 0.44 | 15, 0.43 |
| 4.8 | &nbsp;&nbsp;CMAZE-T conditions and relational coding | 15, 1.08 | 16, 1.08 |
| 5 | **Discussion** | 16, 1.99 | 17, 2.05 |
| 5.1 | &nbsp;&nbsp;Limitations | 17, 0.75 | 17, 0.77 |
| 5.2 | &nbsp;&nbsp;A planned CMAZE suite | 17, 0.12 | 18, 0.13 |
| 5.3 | &nbsp;&nbsp;The zero-dynamics clue | 17, 0.27 | 18, 0.28 |
| 5.4 | &nbsp;&nbsp;The Latent Dynamics Model class | 18, 0.37 | 18, 0.39 |
| 5.5 | &nbsp;&nbsp;Summary | 18, 0.22 | 18, 0.20 |
| 5.6 | &nbsp;&nbsp;Physics follow-up | 18, 0.20 | 19, 0.21 |
|  | Statement: AI use statement | 18, 0.12 | 19, 0.12 |
|  | Statement: Ethics statement | 19, 0.15 | 19, 0.15 |
|  | Statement: Reproducibility statement | 19, 0.19 | 19, 0.21 |
|  | References | 19, 4.60 | 19, 5.32 |
|  | Appendices (preface) | 24, 0.11 | 25, 0.11 |
| A | Reporting conventions | 24, 0.42 | 25, 0.42 |
| B | Tab.~\ref{tab:t7}: what it does and does not claim | 24, 1.13 | 25, 1.13 |
| C | The $K=5$ endpoint of the sweep | 25, 0.58 | 26, 0.58 |
| D | The $K$-curve: composition and the memory footprint against $K$ | 26, 1.62 | 27, 1.62 |
| E | Deletion on Shapes3D: two implementations, and the failure they agree on | 27, 0.95 | 28, 0.97 |
| F | The earlier composition cell: byte and parameter match, and the ceiling ladder | 28, 1.01 | 29, 1.05 |
| G | Deletion comparators in full (fit pool, 0 test runs) | 29, 0.63 | 30, 0.69 |
| H | Light encoder adaptation in full | 30, 0.44 | 31, 0.96 |
| I | The encoder loop in full (shortened-decoder instrument, and the longer leg) | 30, 1.00 | 32, 0.68 |
| J | The zero-dynamics identity: conditions and provenance | 31, 0.92 | 33, 0.73 |
| K | The retention law on the field-mode n-CLU in full | 32, 0.89 | 33, 0.91 |
| L | Prior derivations | 33, 1.30 | 34, 1.42 |
| M | What an evaluation must contain: the ten requirements | 35, 0.41 | 36, 0.42 |
| N | Maze world-model rows | 35, 1.42 | 36, 2.17 |
| O | The encoder: adaptation and the unfreezing decomposition | 36, 2.07 | 38, 1.87 |
|  | &nbsp;&nbsp;Light encoder adaptation (secondary) | 36, 0.97 | 38, 0.86 |
|  | &nbsp;&nbsp;Unfreezing with the n-CLU on the gradient path: the decomposition | 37, 1.06 | 39, 0.98 |
| P | The two laws on this study's n-CLU, stated in full | 38, 1.84 | 40, 1.78 |
| Q | Further limitations | 40, 0.76 | 42, 0.24 |
| R | Placeholder index | 41, 0.18 | 42, 0.19 |
| S | Declared not-runs | 41, 0.96 | 42, 0.96 |

Five main sections (len, pp): Introduction 1.93 / 1.97; Methods 2.03 / 2.00; Experiments 2.58 / 2.64; Results 8.57 / 8.69; Discussion 1.99 / 2.05 (v2 / style). Statements (AI use + ethics + reproducibility): 0.46 / 0.48. References: 4.60 / 5.32. Appendices (preface to end): 18.64 / 18.89.

## 6. Every number left in the main text, with its justification (re-derived independently)

Scanner and classifier written fresh for this audit (`pw2_draft_final/numscan.py`, `numclass.py`); they import nothing from `pw2_draft_numbers`. Main text = `\title` to the "AI use statement" heading, abstract included; comments, floats (captions and tables), `\figslot` calls and `\ref/\label/\cite` keys excluded. Tokens: every digit run, and every spelled cardinal (zero … twenty, the tens, hundred, thousand, million, billion, half, twice, dozen). Each token gets a class and a justification; **UNCLASSIFIED: 0 in both drafts.** Every rounded value is re-checked against its record, verbatim (`rounded_check.py`, 32 phrases, **0 FAIL** in both drafts; planted control FAILs). Full per-token lists: `NUMBERS-NOW-DRAFT-v2.md`, `NUMBERS-NOW-DRAFT-v2-style.md`.

### DRAFT-v2: 383 tokens (198 digit, 185 word)

| section | digit | word | HEAD-HELD | IDENTIFIER | PROTOCOL/DEFINITION | ROUNDED | COUNT | NOT-A-NUMBER | UNCLASSIFIED |
|---|---|---|---|---|---|---|---|---|---|
| Abstract | 1 | 8 | 2 | 3 | 0 | 2 | 1 | 1 | 0 |
| Introduction | 8 | 30 | 2 | 9 | 0 | 4 | 22 | 1 | 0 |
| Methods | 13 | 5 | 0 | 9 | 4 | 0 | 4 | 1 | 0 |
| Experiments | 37 | 19 | 36 | 5 | 0 | 0 | 15 | 0 | 0 |
| Results | 121 | 85 | 50 | 77 | 10 | 18 | 50 | 1 | 0 |
| Discussion | 18 | 38 | 1 | 22 | 4 | 7 | 20 | 2 | 0 |
| **total** | 198 | 185 | 91 | 125 | 18 | 31 | 112 | 6 | 0 |

Digit tokens outside Head-held text: 128; inside Head-held text: 70 (Experiments, first paragraph 32, Results, first paragraph 38).

### DRAFT-v2-style: 381 tokens (199 digit, 182 word)

| section | digit | word | HEAD-HELD | IDENTIFIER | PROTOCOL/DEFINITION | ROUNDED | COUNT | NOT-A-NUMBER | UNCLASSIFIED |
|---|---|---|---|---|---|---|---|---|---|
| Abstract | 1 | 7 | 2 | 3 | 0 | 2 | 1 | 0 | 0 |
| Introduction | 8 | 29 | 2 | 9 | 0 | 5 | 20 | 1 | 0 |
| Methods | 13 | 4 | 0 | 9 | 4 | 0 | 4 | 0 | 0 |
| Experiments | 38 | 20 | 38 | 5 | 0 | 0 | 15 | 0 | 0 |
| Results | 121 | 84 | 50 | 77 | 10 | 18 | 49 | 1 | 0 |
| Discussion | 18 | 38 | 1 | 22 | 4 | 7 | 20 | 2 | 0 |
| **total** | 199 | 182 | 93 | 125 | 18 | 32 | 109 | 4 | 0 |

Digit tokens outside Head-held text: 128; inside Head-held text: 71 (Experiments, first paragraph 33, Results, first paragraph 38).

### Abstract, DRAFT-v2

- l.88 `three` — HEAD-HELD: abstract, opening sentence
- l.91 `one` — NOT-A-NUMBER: pronoun "one"
- l.95 `three` — COUNT: the three properties (the paper's frame)
- l.100 `3` — IDENTIFIER: part of a name (`Shapes3D`: dataset, dtype or criterion name)
- l.110 `seven` — ROUNDED: rounded; record re-checked, rounded_check row 1 (mean paired margin over the MLP +7.034 rounds to 7)
- l.120 `six` — ROUNDED: rounded; record re-checked, rounded_check row 4 (abstract: per-seed survivor losses 6.13 / 6.43 / 6.30 each round to 6)
- l.126 `six` — HEAD-HELD: abstract, maze `\TODO` (whole lines)
- l.130 `zero` — IDENTIFIER: name of the limit / arm (zero-dynamics, zero-step)
- l.131 `zero` — IDENTIFIER: name of the limit / arm (zero-dynamics, zero-step)

**Quantities in the abstract: 5** ("three", "three", "seven", "six", "six"); the plan's limit is five. Names ("Shapes3D", "zero-dynamics", "zero-step") and the pronoun "one" are not quantities.

### Abstract, DRAFT-v2-style

- l.94 `three` — HEAD-HELD: abstract, opening sentence
- l.100 `three` — COUNT: the three properties (the paper's frame)
- l.104 `3` — IDENTIFIER: part of a name (`Shapes3D`: dataset, dtype or criterion name)
- l.114 `seven` — ROUNDED: rounded; record re-checked, rounded_check row 1 (mean paired margin over the MLP +7.034 rounds to 7)
- l.124 `six` — ROUNDED: rounded; record re-checked, rounded_check row 4 (abstract: per-seed survivor losses 6.13 / 6.43 / 6.30 each round to 6)
- l.130 `six` — HEAD-HELD: abstract, maze `\TODO` (whole lines)
- l.134 `zero` — IDENTIFIER: name of the limit / arm (zero-dynamics, zero-step)
- l.134 `zero` — IDENTIFIER: name of the limit / arm (zero-dynamics, zero-step)

**Quantities in the abstract: 5** ("three", "three", "seven", "six", "six"); the plan's limit is five. Names ("Shapes3D", "zero-dynamics", "zero-step") and the pronoun "one" are not quantities.

### Every number outside Head-held text (DRAFT-v2), with its justification

| line | section | number | class | justification | context |
|---|---|---|---|---|---|
| 91 | Abstract | `one` | NOT-A-NUMBER | pronoun "one" | `chine unlearning is an established one. We work at the level of a single \emph{n` |
| 95 | Abstract | `three` | COUNT | the three properties (the paper's frame) | `y the simplest object that has all three properties, the \emph{naive-CLU} (n-CLU` |
| 100 | Abstract | `3` | IDENTIFIER | part of a name (`Shapes3D`: dataset, dtype or criterion name) | `without those rows. On a Shapes3D value hold-out, at the finest partition on` |
| 110 | Abstract | `seven` | ROUNDED | rounded; record re-checked, rounded_check row 1 (mean paired margin over the MLP +7.034 rounds to 7) | `er steps, the n-CLU produces about seven more distinct` |
| 120 | Abstract | `six` | ROUNDED | rounded; record re-checked, rounded_check row 4 (abstract: per-seed survivor losses 6.13 / 6.43 / 6.30 each round to 6) | `red value: survivors fall by about six tuples per` |
| 130 | Abstract | `zero` | IDENTIFIER | name of the limit / arm (zero-dynamics, zero-step) | `uccessor: the n-CLU is exactly the zero-dynamics limit of a settling unit, with` |
| 131 | Abstract | `zero` | IDENTIFIER | name of the limit / arm (zero-dynamics, zero-step) | `the zero-step arm bit-identical to the static rea` |
| 154 | Introduction | `2024` | IDENTIFIER | year in a citation phrase | `2024 journal version~` |
| 155 | Introduction | `half` | NOT-A-NUMBER | "either half" (of a two-part claim) | `dress book. We do not claim either half. In particular we do not claim` |
| 157 | Introduction | `one` | COUNT | determiner/count "one" (a single thing) | `wndate primitive, which follows in one line from the summation` |
| 170 | Introduction | `four` | ROUNDED | rounded; record re-checked, rounded_check row 11 (literature screen 4,050 rounds to 4,000) | `about four thousand unique works: those citing the` |
| 170 | Introduction | `thousand` | ROUNDED | rounded; record re-checked, rounded_check row 11 (literature screen 4,050 rounds to 4,000) | `about four thousand unique works: those citing the gener` |
| 173 | Introduction | `ten` | COUNT | count word of discrete things (ten of the) | `Deletion language appears in ten of the generation-side works, most of the` |
| 175 | Introduction | `six` | COUNT | count word of discrete things (six claim) | `anguage; the highest number of the six claim` |
| 176 | Introduction | `four` | COUNT | count word of discrete things (four; and of) | `toms reached by any single work is four; and of the deletion-side works none com` |
| 180 | Introduction | `zero` | COUNT | count word of discrete things (zero works on) | `\wedge$ a deletion phrase, returns zero works on the screen.` |
| 184 | Introduction | `three` | ROUNDED | rounded; record re-checked, rounded_check row 30 (the record itself states the approximate window "< ~3 months") | `three months is incomplete (Sec.~` |
| 185 | Introduction | `Three` | COUNT | count word of discrete things (Three families of) | `Three families of systems built on top of uni` |
| 200 | Introduction | `One` | COUNT | determiner/count "one" (a single thing) | `One distinction is ours. The decremental-PCA` |
| 208 | Introduction | `three` | COUNT | the three properties (the paper's frame) | `\subsection{The three properties}` |
| 212 | Introduction | `three` | COUNT | the three properties (the paper's frame) | `d trusted with the answer. We name three properties and use them throughout.` |
| 218 | Introduction | `three` | COUNT | the three properties (the paper's frame) | `on those rows. The three pull against each other: a lookup table` |
| 222 | Introduction | `three` | COUNT | the three properties (the paper's frame) | `The object we study makes the three properties of one n-CLU, and we measure` |
| 222 | Introduction | `one` | COUNT | determiner/count "one" (a single thing) | `tudy makes the three properties of one n-CLU, and we measure, row by row, where` |
| 230 | Introduction | `3` | IDENTIFIER | part of a name (`Shapes3D`: dataset, dtype or criterion name) | `Shapes3D~ va` |
| 230 | Introduction | `32` | IDENTIFIER | partition size $K$ (design name / sweep grid) | `value hold-out at $\Kstar=32$ wells, three scored seeds, with every` |
| 230 | Introduction | `three` | COUNT | count word of discrete things (three scored seeds,) | `e hold-out at $\Kstar=32$ wells, three scored seeds, with every` |
| 232 | Introduction | `seven` | ROUNDED | rounded; record re-checked, rounded_check row 2 (Contribution 1: +7.034 rounds to 7) | `fitted on its own rows: by about seven tuples` |
| 236 | Introduction | `3` | IDENTIFIER | rung name ($n$-way ladder) | `Sec.~ ). For $n\ge3$ the rungs are a training-data axis.` |
| 250 | Introduction | `six` | COUNT | count word of discrete things (six seeds, the) | `symmetric inputs, six seeds, the n-CLU is ahead of` |
| 251 | Introduction | `three` | COUNT | seed-outcome tally (a count of seeds; the per-seed values are in the cited table) | `the best matched baseline on three seeds, level with it on two and behind on` |
| 251 | Introduction | `two` | COUNT | seed-outcome tally (a count of seeds; the per-seed values are in the cited table) | `on three seeds, level with it on two and behind on one` |
| 251 | Introduction | `one` | COUNT | seed-outcome tally (a count of seeds; the per-seed values are in the cited table) | `vel with it on two and behind on one` |
| 256 | Introduction | `Two` | COUNT | count word of discrete things (Two ablations.} A) | `\item \textbf{Two ablations.} A composition-only n-CLU comp` |
| 258 | Introduction | `zero` | IDENTIFIER | name of the limit / arm (zero-dynamics, zero-step) | `limitations, its measured zero-dynamics identity and the theory we carry` |
| 263 | Introduction | `one` | COUNT | determiner/count "one" (a single thing) | `on one environment (Shapes3D) at three scored se` |
| 263 | Introduction | `3` | IDENTIFIER | part of a name (`Shapes3D`: dataset, dtype or criterion name) | `on one environment (Shapes3D) at three scored seeds; no Shapes3D row ca` |
| 263 | Introduction | `three` | COUNT | count word of discrete things (three scored seeds;) | `on one environment (Shapes3D) at three scored seeds; no Shapes3D row can excee` |
| 263 | Introduction | `3` | IDENTIFIER | part of a name (`Shapes3D`: dataset, dtype or criterion name) | `D) at three scored seeds; no Shapes3D row can exceed three seeds,` |
| 263 | Introduction | `three` | COUNT | count word of discrete things (three seeds,) | `seeds; no Shapes3D row can exceed three seeds,` |
| 264 | Introduction | `3` | IDENTIFIER | seed index | `because seeds 3 and 4 have verified caches but no frozen de` |
| 264 | Introduction | `4` | IDENTIFIER | seed index | `because seeds 3 and 4 have verified caches but no frozen decoder` |
| 266 | Introduction | `six` | COUNT | count word of discrete things (six seeds on) | `, under the engine's judge, prints six seeds on its words-plus-observation rows` |
| 286 | Methods | `three` | COUNT | the three properties (the paper's frame) | `A claim about any of the three properties is only as precise as the de` |
| 299 | Methods | `64` | PROTOCOL/DEFINITION | assertion size of a code check (64 checks) | `le under feature perturbation over 64 checks at $\Kstar$).` |
| 305 | Methods | `four` | COUNT | count word of discrete things (four byte counts) | `footprint.} Every n-CLU row prints four byte counts` |
| 306 | Methods | `1` | IDENTIFIER | list enumerator (1)-(4) | `separately: (1) the atoms, (2) the covariance and count st` |
| 306 | Methods | `2` | IDENTIFIER | list enumerator (1)-(4) | `separately: (1) the atoms, (2) the covariance and count statistic, (3) th` |
| 306 | Methods | `3` | IDENTIFIER | list enumerator (1)-(4) | `e covariance and count statistic, (3) the address book, and` |
| 307 | Methods | `4` | IDENTIFIER | list enumerator (1)-(4) | `(4) the coupling and hole table used for refus` |
| 308 | Methods | `64` | IDENTIFIER | part of a name (`float64`: dataset, dtype or criterion name) | `onvention (symmetric-triangle float64 for the statistic, unless marked` |
| 309 | Methods | `524,288` | PROTOCOL/DEFINITION | the one definition of the byte reference (524,288 B / 131,072 float32) | `otherwise). The 524,288-byte figure that recurs below is a \e` |
| 311 | Methods | `131,072` | PROTOCOL/DEFINITION | the one definition of the byte reference (524,288 B / 131,072 float32) | `about 131,072 float32 parameters at it.` |
| 311 | Methods | `32` | IDENTIFIER | part of a name (`float32`: dataset, dtype or criterion name) | `about 131,072 float32 parameters at it.` |
| 327 | Methods | `3` | IDENTIFIER | part of a name (`Shapes3D`: dataset, dtype or criterion name) | `judge and the estimand.} On Shapes3D an external classifier, hash-pinned per` |
| 329 | Methods | `six` | COUNT | count word of discrete things (six-factor) | `h a fixed ceiling for pair and for six-factor` |
| 330 | Methods | `1.0` | PROTOCOL/DEFINITION | the declared 1.0-tuple margin (defined once, here) | `over combinations with a margin of 1.0 tuple per held-out combination declared` |
| 335 | Methods | `1` | IDENTIFIER | part of a name (`S1`: dataset, dtype or criterion name) | `claration (for example, criterion S1 or criterion S2).` |
| 335 | Methods | `2` | IDENTIFIER | part of a name (`S2`: dataset, dtype or criterion name) | `xample, criterion S1 or criterion S2).` |
| 343 | Methods | `one` | COUNT | determiner/count "one" (a single thing) | `Sec.~ ). We use one definition of deletion throughout, and st` |
| 360 | Methods | `one` | NOT-A-NUMBER | pronoun "one" | `e scoping sentence we carry is the one our deletion theory states for its own n-` |
| 406 | Experiments | `3` | IDENTIFIER | part of a name (`Shapes3D`: dataset, dtype or criterion name) | `\subsection{The Shapes3D value hold-out}` |
| 434 | Experiments | `Four` | COUNT | count word of discrete things (Four controls, which) | `ed \emph{floor}, never a baseline. Four controls, which we call the \emph{pre-de` |
| 441 | Experiments | `one` | COUNT | determiner/count "one" (a single thing) | `with one schedule, the same for` |
| 453 | Experiments | `2` | IDENTIFIER | rung name ($n$-way ladder) | `ntly pinned to held-out values; $n=2$ and` |
| 454 | Experiments | `6` | IDENTIFIER | rung name ($n$-way ladder) | `$n=6$ are the two ends. Tab.~ gi` |
| 454 | Experiments | `two` | COUNT | count word of discrete things (two ends. Tab.~\ref{tab:ladder}) | `$n=6$ are the two ends. Tab.~ gives the n-C` |
| 455 | Experiments | `two` | COUNT | count word of discrete things (two read-path-matched baseline) | `seed beside two read-path-matched baseline sets on the sa` |
| 455 | Experiments | `four` | COUNT | count word of discrete things (four) | `nations: the data-matched set, all four` |
| 458 | Experiments | `3` | IDENTIFIER | rung name ($n$-way ladder) | `rungs $n\ge3$ are exactly constant by construction.` |
| 460 | Experiments | `two` | COUNT | count word of discrete things (two legs keep) | `two legs keep a single surviving novel tuple` |
| 466 | Experiments | `Three` | COUNT | count word of discrete things (Three deletions are) | `Three deletions are run on the $\Kstar$ objec` |
| 466 | Experiments | `three` | COUNT | count word of discrete things (three % SRC:) | `fit pool (Tab.~ ), at three` |
| 471 | Experiments | `1` | IDENTIFIER | the item-deletion scope name "random 1%" | `tion result. \emph{Item}: a random 1\% of fit rows, which is` |
| 472 | Experiments | `three` | COUNT | count word of discrete things (three operations on) | `tability. \emph{Forgetting $c$} is three operations on one object:` |
| 472 | Experiments | `one` | COUNT | determiner/count "one" (a single thing) | `etting $c$} is three operations on one object:` |
| 474 | Experiments | `four` | COUNT | count word of discrete things (four measurements) | `very deletion row therefore prints four measurements` |
| 492 | Experiments | `four` | COUNT | count word of discrete things (four facts) | `ates the next state; the answer is four facts` |
| 495 | Experiments | `five` | COUNT | count word of discrete things (five words. The) | `floors are printed. The request is five words. The` |
| 496 | Experiments | `two` | COUNT | count word of discrete things (two-way word pairs) | `are a never-trained value, unseen two-way word pairs and unseen $n$-way word` |
| 498 | Experiments | `one` | COUNT | determiner/count "one" (a single thing) | `dely from seed to seed, and it has one engine; those are` |
| 506 | Results | `3` | IDENTIFIER | part of a name (`Shapes3D`: dataset, dtype or criterion name) | `\subsection{Composition on Shapes3D}` |
| 586 | Results | `0` | IDENTIFIER | seed index | `b.~ as images, at seed 0 and $\Kstar=32$, against the` |
| 586 | Results | `32` | IDENTIFIER | partition size $K$ (design name / sweep grid) | `as images, at seed 0 and $\Kstar=32$, against the` |
| 587 | Results | `four` | COUNT | count word of discrete things (four data-matched baselines) | `four data-matched baselines of that table (ML` |
| 587 | Results | `10` | IDENTIFIER | schedule name "$10\times$ training steps" | `table (MLP and continuous MLP at $10\times$ training steps, LSTM and` |
| 590 | Results | `0` | IDENTIFIER | seed index | `Over all held-out requests at seed 0 the judge marks` |
| 591 | Results | `nine` | ROUNDED | rounded; record re-checked, rounded_check row 12 (n-CLU valid 5,177 of 5,688 at seed 0 = 0.910 -> 0.9) | `the n-CLU valid on about nine in ten, more often than any baseline, an` |
| 591 | Results | `ten` | ROUNDED | rounded; record re-checked, rounded_check row 12 (n-CLU valid 5,177 of 5,688 at seed 0 = 0.910 -> 0.9) | `the n-CLU valid on about nine in ten, more often than any baseline, and the lo` |
| 593 | Results | `one` | COUNT | determiner/count "one" (a single thing) | `he image is almost always wrong in one factor only` |
| 747 | Results | `2` | IDENTIFIER | rung name ($n$-way ladder) | `ery $n$; the n-CLU's own rise from 2-way to 6-way is not to be inferred, because` |
| 747 | Results | `6` | IDENTIFIER | rung name ($n$-way ladder) | `the n-CLU's own rise from 2-way to 6-way is not to be inferred, because rungs 3-` |
| 747 | Results | `3` | IDENTIFIER | rung name ($n$-way ladder) | `not to be inferred, because rungs 3--6` |
| 747 | Results | `6` | IDENTIFIER | rung name ($n$-way ladder) | `t to be inferred, because rungs 3--6` |
| 749 | Results | `3,4,5,6` | IDENTIFIER | rung name ($n$-way ladder) | `rungs $n=3,4,5,6$ probe the identical combinations and` |
| 753 | Results | `2` | IDENTIFIER | rung name ($n$-way ladder) | `Only $n=2$ against $n=6$ is a genuine change of probe` |
| 753 | Results | `6` | IDENTIFIER | rung name ($n$-way ladder) | `Only $n=2$ against $n=6$ is a genuine change of probe, on disjoint` |
| 757 | Results | `95` | PROTOCOL/DEFINITION | interval level (95%) | `data-matched baseline, the n-CLU's 95\%` |
| 758 | Results | `zero` | PROTOCOL/DEFINITION | zero as the null reference of an interval ("above / excludes / crosses zero") | `lies above zero on every rung and seed` |
| 760 | Results | `ten` | IDENTIFIER | schedule name ("ten times" / "thirty times" its selected steps) | `not proven converged everywhere. A ten-times spot check` |
| 761 | Results | `one` | ROUNDED | rounded; record re-checked, rounded_check row 13 (LSTM ten-times spot check 20.55 -> 21.85, +1.30 -> 1) | `moves the LSTM by about one tuple at one rung and seed (Tab.~` |
| 761 | Results | `one` | COUNT | determiner/count "one" (a single thing) | `ves the LSTM by about one tuple at one rung and seed (Tab.~ ). Th` |
| 767 | Results | `four` | ROUNDED | rounded; record re-checked, rounded_check row 31 (an exact ratio: data-matched pool 131,072 rows / n-CLU content rows 32,768 = 4) | `ether the data-matched set's rows, four times the n-CLU's` |
| 771 | Results | `95` | PROTOCOL/DEFINITION | interval level (95%) | `and every 95\% interval of the n-CLU-content` |
| 772 | Results | `zero` | PROTOCOL/DEFINITION | zero as the null reference of an interval ("above / excludes / crosses zero") | `margins lies above zero.` |
| 773 | Results | `ten` | IDENTIFIER | schedule name ("ten times" / "thirty times" its selected steps) | `rung and seed and is flat from its ten-times fit to its` |
| 775 | Results | `One` | COUNT | determiner/count "one" (a single thing) | `baseline. One more caveat bears on every single-fit bas` |
| 775 | Results | `two` | COUNT | count word of discrete things (two MLP fits) | `fit baseline number in this paper: two MLP fits with the` |
| 777 | Results | `four` | ROUNDED | rounded; record re-checked, rounded_check row 14 (seed-to-seed MLP difference up to 4.13 -> 4) | `up to about four novel tuples per held-out combination, a` |
| 777 | Results | `zero` | PROTOCOL/DEFINITION | zero as the null reference of an interval ("above / excludes / crosses zero") | `per-combination interval excludes zero on most` |
| 781 | Results | `6` | IDENTIFIER | rung name ($n$-way ladder) | `st of the inference note refits $n=6$ with the` |
| 782 | Results | `3` | IDENTIFIER | rung name ($n$-way ladder) | `fit pool restricted to $n=3$'s train index, holding depth at 6 and cove` |
| 782 | Results | `6` | IDENTIFIER | rung name ($n$-way ladder) | `3$'s train index, holding depth at 6 and coverage at $n=3$'s.` |
| 782 | Results | `3` | IDENTIFIER | rung name ($n$-way ladder) | `ding depth at 6 and coverage at $n=3$'s.` |
| 783 | Results | `3` | IDENTIFIER | rung name ($n$-way ladder) | `\textbf{Result.} For $n\ge3$ the ladder's rise is fit-pool` |
| 784 | Results | `3` | IDENTIFIER | rung name ($n$-way ladder) | `ion it can only be coverage: rungs 3--6 probe the identical combinations (Tab.~` |
| 784 | Results | `6` | IDENTIFIER | rung name ($n$-way ladder) | `it can only be coverage: rungs 3--6 probe the identical combinations (Tab.~` |
| 786 | Results | `6` | IDENTIFIER | rung name ($n$-way ladder) | `so $n=6$ refitted on $n=3$'s fit pool \emph{is} run` |
| 786 | Results | `3` | IDENTIFIER | rung name ($n$-way ladder) | `so $n=6$ refitted on $n=3$'s fit pool \emph{is} rung $n=3$. Its TEST` |
| 786 | Results | `3` | IDENTIFIER | rung name ($n$-way ladder) | `$n=3$'s fit pool \emph{is} rung $n=3$. Its TEST cell is therefore` |
| 787 | Results | `3` | IDENTIFIER | rung name ($n$-way ladder) | `\emph{derived}: it is the $n=3$ n-CLU cell of Tab.~ , and n` |
| 789 | Results | `6` | IDENTIFIER | rung name ($n$-way ladder) | `On TEST, $n=6$ at $n=3$'s coverage is therefore the $n=3$` |
| 789 | Results | `3` | IDENTIFIER | rung name ($n$-way ladder) | `On TEST, $n=6$ at $n=3$'s coverage is therefore the $n=3$ cell of` |
| 789 | Results | `3` | IDENTIFIER | rung name ($n$-way ladder) | `=3$'s coverage is therefore the $n=3$ cell of Tab.~ :` |
| 790 | Results | `3` | IDENTIFIER | rung name ($n$-way ladder) | `the whole $n=3\to6$ rise, nearly five tuples per held-out` |
| 790 | Results | `6` | IDENTIFIER | rung name ($n$-way ladder) | `the whole $n=3\to6$ rise, nearly five tuples per held-out comb` |
| 790 | Results | `five` | ROUNDED | rounded; record re-checked, rounded_check row 15 (TEST rise n=3 -> 6 +4.989: below five and within 10%) | `the whole $n=3\to6$ rise, nearly five tuples per held-out combination, is fit-` |
| 791 | Results | `1` | PROTOCOL/DEFINITION | coverage share 1 by construction (a definitional identity) | `it is depth; the coverage share is 1 by construction. That TEST cell is \emph{de` |
| 795 | Results | `one` | NOT-A-NUMBER | determiner "the one measured ..." (coinage, Head-held elsewhere) | `combination rows. Validation, the one measured leg, measures it` |
| 796 | Results | `6` | IDENTIFIER | rung name ($n$-way ladder) | `directly: $n=6$ at $n=3$'s coverage equals $n=3$ to every` |
| 796 | Results | `3` | IDENTIFIER | rung name ($n$-way ladder) | `directly: $n=6$ at $n=3$'s coverage equals $n=3$ to every printed d` |
| 796 | Results | `3` | IDENTIFIER | rung name ($n$-way ladder) | `n=6$ at $n=3$'s coverage equals $n=3$ to every printed decimal on every seed` |
| 799 | Results | `two` | COUNT | seed-outcome tally (a count of seeds; the per-seed values are in the cited table) | `s below the pre-declared margin on two of three seeds, so the` |
| 799 | Results | `three` | COUNT | count word of discrete things (three seeds, so) | `the pre-declared margin on two of three seeds, so the` |
| 820 | Results | `3` | IDENTIFIER | part of a name (`Shapes3D`: dataset, dtype or criterion name) | `\subsection{Deletion on Shapes3D}` |
| 889 | Results | `3` | IDENTIFIER | part of a name (`Shapes3D`: dataset, dtype or criterion name) | `r-tuple aggregates and every Shapes3D tuple is one row, so its statistic is the` |
| 889 | Results | `one` | COUNT | determiner/count "one" (a single thing) | `egates and every Shapes3D tuple is one row, so its statistic is the` |
| 897 | Results | `six` | ROUNDED | rounded; record re-checked, rounded_check row 5 (Sec. 4.3: 6.13 / 6.43 / 6.30 each round to 6) | `by about six tuples per held-out combination, on all t` |
| 897 | Results | `three` | COUNT | seed-outcome tally (a count of seeds; the per-seed values are in the cited table) | `s per held-out combination, on all three seeds (Tab.~ , Fig.~` |
| 897 | Results | `0` | IDENTIFIER | seed index | `Fig.~ ). At seed 0 the loss is concentrated where` |
| 899 | Results | `thirty` | ROUNDED | rounded; record re-checked, rounded_check row 8 (Sec. 4.3: -32.1 at seed 0 rounds to 30 (nearest ten)) | `about thirty each, those that request a sphere in a` |
| 899 | Results | `one` | ROUNDED | rounded; record re-checked, rounded_check row 9 (Sec. 4.3: -1.47 rounds to 1) | `sphere in another colour by about one, and the others barely move` |
| 900 | Results | `1` | IDENTIFIER | the item-deletion scope name "random 1%" | `he same pipeline deleting a random 1\% of items moves the survivors by at most a` |
| 907 | Results | `one` | COUNT | "one-tuple" (a compound: the declared one-tuple margin / one-token lookup / one-parameter form) | `by less than the declared one-tuple margin (Tab.~ ).` |
| 912 | Results | `0.6` | ROUNDED | rounded; record re-checked, rounded_check row 16 (relative RMS 0.610083 (red) / 0.614277 (others) -> 0.6) | `ook carry, the read moves by about 0.6 relative RMS on` |
| 920 | Results | `5` | IDENTIFIER | partition size $K$ (design name / sweep grid) | `rather than vanished: at $K=5$ the same red deletion was concentrated,` |
| 932 | Results | `three` | COUNT | count word of discrete things (three seeds (Tabs.~\ref{tab:rivals-del}) | `the baseline's selected schedule, three seeds (Tabs.~` |
| 935 | Results | `95` | PROTOCOL/DEFINITION | interval level (95%) | `baseline on every seed, with every 95\% interval above` |
| 936 | Results | `zero` | PROTOCOL/DEFINITION | zero as the null reference of an interval ("above / excludes / crosses zero") | `zero. After the red-sphere deletion it is ahe` |
| 937 | Results | `zero` | PROTOCOL/DEFINITION | zero as the null reference of an interval ("above / excludes / crosses zero") | `zero against the MLP at seeds 1 and 2 and aga` |
| 937 | Results | `1` | IDENTIFIER | seed index | `zero against the MLP at seeds 1 and 2 and against the LSTM at seed 2 (Fig.~` |
| 937 | Results | `2` | IDENTIFIER | seed index | `ero against the MLP at seeds 1 and 2 and against the LSTM at seed 2 (Fig.~` |
| 937 | Results | `2` | IDENTIFIER | seed index | `and 2 and against the LSTM at seed 2 (Fig.~ ):` |
| 938 | Results | `six` | ROUNDED | rounded; record re-checked, rounded_check row 6 (Sec. 4.4: 6.13 / 6.43 / 6.30 each round to 6) | `s own loss on that deletion, about six tuples per held-out combination, is large` |
| 939 | Results | `2` | IDENTIFIER | seed index | `except the continuous MLP at seed 2, so on` |
| 945 | Results | `0` | IDENTIFIER | seed index | `lected schedule, so the MLP (seeds 0 and 1) and continuous-MLP convergence flags` |
| 945 | Results | `1` | IDENTIFIER | seed index | `schedule, so the MLP (seeds 0 and 1) and continuous-MLP convergence flags` |
| 957 | Results | `two` | COUNT | count word of discrete things (two holes cost) | `$\Kstar$ object two holes cost a byte each, read no data rows` |
| 961 | Results | `One` | COUNT | determiner/count "one" (a single thing) | `One hole (red $\wedge$ cube) is refusable alt` |
| 966 | Results | `1` | IDENTIFIER | the item-deletion scope name "random 1%" | `1\% of items clears no value bit and punches` |
| 999 | Results | `one` | COUNT | "one-token" (a compound: the declared one-tuple margin / one-token lookup / one-parameter form) | `deletion the task is a one-token lookup, and a free value-presence b` |
| 1002 | Results | `one` | COUNT | determiner/count "one" (a single thing) | `, where the coupling hole costs one` |
| 1006 | Results | `three` | COUNT | count word of discrete things (three) | `d refuses every red request on all three` |
| 1007 | Results | `one` | COUNT | "one-tuple" (a compound: the declared one-tuple margin / one-token lookup / one-parameter form) | `nges composition beyond a declared one-tuple bar on two of them,` |
| 1007 | Results | `two` | COUNT | seed-outcome tally (a count of seeds; the per-seed values are in the cited table) | `beyond a declared one-tuple bar on two of them,` |
| 1012 | Results | `Two` | COUNT | count word of discrete things (Two ablations}) | `\subsection{Two ablations}` |
| 1018 | Results | `5` | IDENTIFIER | partition size $K$ (design name / sweep grid) | `$K=5$ endpoint of the ladder it is bit-identical` |
| 1019 | Results | `1.4` | ROUNDED | rounded; record re-checked, rounded_check row 17 (composition-only bytes 712,072..719,352 / 524,288 = 1.358..1.372 < 1.4) | `seed, at just under 1.4 times the reference in bytes (Tab.~` |
| 1021 | Results | `two` | COUNT | count word of discrete things (two reads are) | `generation time, which is why the two reads are equal.` |
| 1025 | Results | `three` | COUNT | count word of discrete things (three seeds, the) | `On its own scored TEST legs, three seeds, the composition-only n-CLU beats` |
| 1027 | Results | `half` | ROUNDED | rounded; record re-checked, rounded_check row 18 (byte/parameter match -0.304% .. +0.495%, all \|.\| < 0.5%) | `within half a percent (App.~ ), the arm c` |
| 1027 | Results | `two` | COUNT | seed-outcome tally (a count of seeds; the per-seed values are in the cited table) | `he arm carrying more parameters on two of three seeds.` |
| 1027 | Results | `three` | COUNT | count word of discrete things (three seeds.) | `carrying more parameters on two of three seeds.` |
| 1033 | Results | `4` | IDENTIFIER | row index of Tab. t7 (row 3 / row 4) | `composes poorly. On row 4 it sits below the learned` |
| 1034 | Results | `eighteen` | ROUNDED | rounded; record re-checked, rounded_check row 19 (item memory row 4 minus learned reference, 3-seed mean of 21.08 / 16.53 / 18.08 = 18.56 > 18) | `reference by more than eighteen tuples per held-out combination on t` |
| 1034 | Results | `3` | COUNT | seed count in "3-seed mean" | `es per held-out combination on the 3-seed mean (Tab.~ ), a low` |
| 1039 | Results | `one` | COUNT | determiner/count "one" (a single thing) | `the join the single design is for: one ablation composes and cannot` |
| 1040 | Results | `one` | COUNT | determiner/count "one" (a single thing) | `and the single design does both on one object at` |
| 1047 | Results | `three` | COUNT | seed-outcome tally (a count of seeds; the per-seed values are in the cited table) | `r the best learned baseline on all three seeds under an` |
| 1048 | Results | `two` | COUNT | count word of discrete things (two) | `anchored full unfreeze and on two` |
| 1049 | Results | `three` | COUNT | seed-outcome tally (a count of seeds; the per-seed values are in the cited table) | `of three under a low-rank adapter (Tab.~` |
| 1057 | Results | `one` | ROUNDED | rounded; record re-checked, rounded_check row 28 (LACE rows 3-4 per seed 0.117 / 0.033 / 0.267 and 0.224 / 0.122 / 0.184, PoE at most 0.020: all below a third of a tuple) | `his feature space, score far below one distinct valid novel tuple per held-out c` |
| 1059 | Results | `3` | IDENTIFIER | row index of Tab. t7 (row 3 / row 4) | `on rows 3--4, render no exact-match request on the ge` |
| 1059 | Results | `4` | IDENTIFIER | row index of Tab. t7 (row 3 / row 4) | `on rows 3--4, render no exact-match request on the gener` |
| 1066 | Results | `3` | IDENTIFIER | part of a name (`Shapes3D`: dataset, dtype or criterion name) | `generation table is a second Shapes3D instrument: a fixed set of` |
| 1068 | Results | `one` | COUNT | determiner/count "one" (a single thing) | `to its arms and scored once, in one of that table's three declared evaluation` |
| 1068 | Results | `three` | COUNT | count word of discrete things (three declared evaluation) | `cored once, in one of that table's three declared evaluation runs (3 seeds,` |
| 1068 | Results | `3` | COUNT | seed count "(3 seeds" | `'s three declared evaluation runs (3 seeds,` |
| 1070 | Results | `four` | COUNT | count word of discrete things (four landscape controls) | `App.~ ). All four landscape controls collapse on every see` |
| 1070 | Results | `1` | IDENTIFIER | part of a name (`S1`: dataset, dtype or criterion name) | `on every seed (declared criterion S1,` |
| 1071 | Results | `2` | IDENTIFIER | part of a name (`S2`: dataset, dtype or criterion name) | `on this table. Declared criterion S2, against the earlier` |
| 1072 | Results | `0` | IDENTIFIER | seed index | `ed margin), \textsc{fails} on seed 0 though its mean is positive. The` |
| 1075 | Results | `3` | IDENTIFIER | part of a name (`S3`: dataset, dtype or criterion name) | `(S3 \textsc{lose}) at about four times the refe` |
| 1075 | Results | `four` | ROUNDED | rounded; record re-checked, rounded_check row 20 (nearest-prototype ceiling at 4.09 .. 4.11 x the reference -> 4) | `(S3 \textsc{lose}) at about four times the reference bytes, and beats LAC` |
| 1076 | Results | `4` | IDENTIFIER | part of a name (`S4`: dataset, dtype or criterion name) | `and PoE on the mean (S4; Tab.~ ).` |
| 1086 | Results | `three` | COUNT | seed-outcome tally (a count of seeds; the per-seed values are in the cited table) | `d the best matched baseline on all three TEST seeds; on main-effect words alone` |
| 1087 | Results | `six` | COUNT | count word of discrete things (six seeds, against) | `plus observation, TEST, six seeds, against the` |
| 1088 | Results | `three` | COUNT | count word of discrete things (three) | `eline over both step counts it has three` |
| 1089 | Results | `0` | IDENTIFIER | seed index | `clear wins (seeds 0, 2 and 3), two ties (seeds 1 and 4) and` |
| 1089 | Results | `2` | IDENTIFIER | seed index | `clear wins (seeds 0, 2 and 3), two ties (seeds 1 and 4) and` |
| 1089 | Results | `3` | IDENTIFIER | seed index | `clear wins (seeds 0, 2 and 3), two ties (seeds 1 and 4) and` |
| 1089 | Results | `two` | COUNT | seed-outcome tally (a count of seeds; the per-seed values are in the cited table) | `clear wins (seeds 0, 2 and 3), two ties (seeds 1 and 4) and` |
| 1089 | Results | `1` | IDENTIFIER | seed index | `seeds 0, 2 and 3), two ties (seeds 1 and 4) and` |
| 1089 | Results | `4` | IDENTIFIER | seed index | `0, 2 and 3), two ties (seeds 1 and 4) and` |
| 1090 | Results | `one` | COUNT | seed-outcome tally (a count of seeds; the per-seed values are in the cited table) | `one loss (seed 5; Tab.~ ).` |
| 1090 | Results | `5` | IDENTIFIER | seed index | `one loss (seed 5; Tab.~ ). Seed 3's lead` |
| 1090 | Results | `3` | IDENTIFIER | seed index | `; Tab.~ ). Seed 3's lead is over a set of baselines weakened` |
| 1091 | Results | `5` | IDENTIFIER | seed index | `at either step count, and on seed 5 the n-CLU scores nothing on two` |
| 1091 | Results | `two` | COUNT | count word of discrete things (two) | `seed 5 the n-CLU scores nothing on two` |
| 1092 | Results | `eight` | COUNT | count word of discrete things (eight other arms) | `lly supported in training and that eight other arms solve. This miss is` |
| 1094 | Results | `one` | COUNT | determiner/count "one" (a single thing) | `e hold-out draw: both tuples share one outcome code (legal, move south, pocket 2` |
| 1094 | Results | `2` | IDENTIFIER | outcome code "(legal, move south, pocket 2)" | `me code (legal, move south, pocket 2),` |
| 1095 | Results | `6` | IDENTIFIER | design name (6-atom / 10-atom) | `a content-capacity failure of the 6-atom design on that code: the code decodes` |
| 1099 | Results | `10` | IDENTIFIER | design name (6-atom / 10-atom) | `the 10-atom relational` |
| 1113 | Results | `one` | COUNT | determiner/count "one" (a single thing) | `small or absent on every seed but one, and on seeds 3 and 4 the` |
| 1113 | Results | `3` | IDENTIFIER | seed index | `n every seed but one, and on seeds 3 and 4 the` |
| 1113 | Results | `4` | IDENTIFIER | seed index | `y seed but one, and on seeds 3 and 4 the` |
| 1114 | Results | `4` | IDENTIFIER | seed index | `he baselines without them. On seed 4 a control` |
| 1116 | Results | `5` | IDENTIFIER | seed index | `there; on seed 5 a matched baseline ties it (a transformer a` |
| 1116 | Results | `10` | IDENTIFIER | schedule name "$10\times$ training steps" | `aseline ties it (a transformer at $10\times$ training steps). The n-CLU's lead o` |
| 1117 | Results | `2` | IDENTIFIER | seed index | `baselines is sizeable only on seed 2. Relational coding has one limit: relations` |
| 1117 | Results | `one` | COUNT | determiner/count "one" (a single thing) | `y on seed 2. Relational coding has one limit: relations` |
| 1120 | Results | `zero` | PROTOCOL/DEFINITION | structural zero (the colour-naming columns are zero by construction) | `ry column that names the colour is zero, so the value row stays a` |
| 1123 | Results | `six` | COUNT | seed-outcome tally (a count of seeds; the per-seed values are in the cited table) | `behind on all six seeds. We report the configuration as evi` |
| 1129 | Results | `six` | COUNT | count word of discrete things (six seeds at) | `on any of the six seeds at a per-step accuracy of about nin` |
| 1129 | Results | `nine` | ROUNDED | rounded; record re-checked, rounded_check row 21 (per-step accuracy 0.900 .. 0.930 -> 0.9) | `ds at a per-step accuracy of about nine in ten, and n-CLU-REL a minority on ever` |
| 1129 | Results | `ten` | ROUNDED | rounded; record re-checked, rounded_check row 21 (per-step accuracy 0.900 .. 0.930 -> 0.9) | `per-step accuracy of about nine in ten, and n-CLU-REL a minority on every seed` |
| 1130 | Results | `two` | COUNT | count word of discrete things (two-way pairs a) | `limitation of every arm. On unseen two-way pairs a learned` |
| 1151 | Results | `5` | IDENTIFIER | panel crop size notation ($5\times5$) | `is a $5\times5$ crop of the maze centred on the age` |
| 1151 | Results | `5` | IDENTIFIER | panel crop size notation ($5\times5$) | `is a $5\times5$ crop of the maze centred on the agent's ce` |
| 1155 | Results | `1` | IDENTIFIER | math notation ($s_{t+1}$, $(\cdot)^{-1}$, $\lambda=0$, $[0.5,2]$ bucket) | `up), the second the engine's $s_{t+1}$, and each further column the $s_{t+1}$ th` |
| 1155 | Results | `1` | IDENTIFIER | math notation ($s_{t+1}$, $(\cdot)^{-1}$, $\lambda=0$, $[0.5,2]$ bucket) | `and each further column the $s_{t+1}$ that a` |
| 1156 | Results | `four` | COUNT | count word of discrete things (four) | `s not score the picture: it scores four` |
| 1158 | Results | `four` | COUNT | count word of discrete things (four. So a) | `not show all four. So a strip under every $s_{t+1}$ prints` |
| 1158 | Results | `1` | IDENTIFIER | math notation ($s_{t+1}$, $(\cdot)^{-1}$, $\lambda=0$, $[0.5,2]$ bucket) | `four. So a strip under every $s_{t+1}$ prints the emitted facts (under the engin` |
| 1164 | Results | `four` | COUNT | count word of discrete things (four facts, so) | `on all four facts, so two panels that show the same` |
| 1164 | Results | `two` | COUNT | count word of discrete things (two panels that) | `on all four facts, so two panels that show the same picture and the` |
| 1180 | Discussion | `three` | COUNT | seed-outcome tally (a count of seeds; the per-seed values are in the cited table) | `of the best matched baseline on three of six seeds,` |
| 1180 | Discussion | `six` | COUNT | count word of discrete things (six seeds,) | `est matched baseline on three of six seeds,` |
| 1181 | Discussion | `two` | COUNT | seed-outcome tally (a count of seeds; the per-seed values are in the cited table) | `level on two and behind on one, and behind on every seed` |
| 1181 | Discussion | `one` | COUNT | seed-outcome tally (a count of seeds; the per-seed values are in the cited table) | `level on two and behind on one, and behind on every seed of the full word` |
| 1186 | Discussion | `six` | ROUNDED | rounded; record re-checked, rounded_check row 7 (Limitation 2: 6.13 / 6.43 / 6.30 -> 6; 32.1 -> 30 (nearest ten)) | `surviving combinations by about six tuples per held-out combination, about thir` |
| 1186 | Discussion | `thirty` | ROUNDED | rounded; record re-checked, rounded_check row 7 (Limitation 2: 6.13 / 6.43 / 6.30 -> 6; 32.1 -> 30 (nearest ten)) | `per held-out combination, about thirty on the combinations that request red` |
| 1187 | Discussion | `0` | IDENTIFIER | seed index | `with another shape at seed 0, while a random 1\% deletion moves them by at` |
| 1187 | Discussion | `1` | IDENTIFIER | the item-deletion scope name "random 1%" | `shape at seed 0, while a random 1\% deletion moves them by at most about a tent` |
| 1195 | Discussion | `four` | ROUNDED | rounded; record re-checked, rounded_check row 22 (deletable object 4.0897 / 4.0768 / 4.0847 x the reference -> 4) | `etable object at $\Kstar$ is about four` |
| 1196 | Discussion | `nine` | COUNT | count word of discrete things (nine swept partitions) | `~ ); across nine swept partitions the honest` |
| 1197 | Discussion | `four` | COUNT | count word of discrete things (four-part total is) | `four-part total is U-shaped in $K$, with its mi` |
| 1197 | Discussion | `24` | IDENTIFIER | partition size $K$ (design name / sweep grid) | `d in $K$, with its minimum at $K=24$ (Tab.~ ).` |
| 1198 | Discussion | `one` | COUNT | determiner/count "one" (a single thing) | `nt of smallest footprint. On the one configuration at $K\ge36$ whose composition` |
| 1198 | Discussion | `36` | IDENTIFIER | partition size $K$ (design name / sweep grid) | `On the one configuration at $K\ge36$ whose composition` |
| 1200 | Discussion | `64` | IDENTIFIER | part of a name (`float64`: dataset, dtype or criterion name) | `No object with a float64 sufficient statistic composes and forgets in` |
| 1201 | Discussion | `64` | IDENTIFIER | rank 64 (design parameter) | `64 on any swept partition (the validation sweep` |
| 1205 | Discussion | `0` | IDENTIFIER | seed index | `f its grids, and the MLP at seed 0 still gains more than a tuple` |
| 1206 | Discussion | `ten` | IDENTIFIER | schedule name ("from ten to thirty times its steps") | `per held-out combination from ten to thirty times its steps (Sec.~` |
| 1206 | Discussion | `thirty` | IDENTIFIER | schedule name ("ten times" / "thirty times" its selected steps) | `held-out combination from ten to thirty times its steps (Sec.~` |
| 1209 | Discussion | `3` | IDENTIFIER | row index of Tab. t7 (row 3 / row 4) | `he pre-declared falsifier on row 3 of the shortened leg, a falsifier that involv` |
| 1216 | Discussion | `three` | COUNT | count word of discrete things (three seeds on) | `\item \textbf{At most three seeds on Shapes3D.} Seeds 3 and 4 canno` |
| 1216 | Discussion | `3` | IDENTIFIER | part of a name (`Shapes3D`: dataset, dtype or criterion name) | `extbf{At most three seeds on Shapes3D.} Seeds 3 and 4 cannot be judged there (no` |
| 1216 | Discussion | `3` | IDENTIFIER | seed index | `st three seeds on Shapes3D.} Seeds 3 and 4 cannot be judged there (no decoder, n` |
| 1216 | Discussion | `4` | IDENTIFIER | seed index | `ee seeds on Shapes3D.} Seeds 3 and 4 cannot be judged there (no decoder, no` |
| 1217 | Discussion | `3` | IDENTIFIER | part of a name (`Shapes3D`: dataset, dtype or criterion name) | `xternal classifier), so no Shapes3D row can exceed three seeds; CMAZE-T prints s` |
| 1217 | Discussion | `three` | COUNT | count word of discrete things (three seeds; CMAZE-T) | `), so no Shapes3D row can exceed three seeds; CMAZE-T prints six.` |
| 1217 | Discussion | `six` | COUNT | count word of discrete things (six.) | `ceed three seeds; CMAZE-T prints six.` |
| 1233 | Discussion | `zero` | IDENTIFIER | name of the limit / arm (zero-dynamics, zero-step) | `\subsection{The zero-dynamics clue}` |
| 1236 | Discussion | `zero` | IDENTIFIER | name of the limit / arm (zero-dynamics, zero-step) | `The n-CLU we evaluate is the zero-dynamics limit of the CLU~` |
| 1236 | Discussion | `zero` | PROTOCOL/DEFINITION | definition of the limit (settling step count set to zero) | `settling loop's step count set to zero. Inside a well this is an` |
| 1239 | Discussion | `zero` | IDENTIFIER | name of the limit / arm (zero-dynamics, zero-step) | `). We asserted this in code. The zero-step arm was required to be bit-identica` |
| 1240 | Discussion | `256` | PROTOCOL/DEFINITION | assertion size of a code check (256 requests,) | `d by \texttt{np.array\_equal} over 256 requests, and it was, on all three scored` |
| 1240 | Discussion | `three` | COUNT | count word of discrete things (three scored) | `r 256 requests, and it was, on all three scored` |
| 1244 | Discussion | `1` | IDENTIFIER | math notation ($s_{t+1}$, $(\cdot)^{-1}$, $\lambda=0$, $[0.5,2]$ bucket) | `is $z^\star=(\beta I-\lambda W)^{-1}\beta z_{\text{req}}$, which returns` |
| 1245 | Discussion | `0` | IDENTIFIER | math notation ($s_{t+1}$, $(\cdot)^{-1}$, $\lambda=0$, $[0.5,2]$ bucket) | `_{\text{req}}$ exactly at $\lambda=0$. Switching the dynamics on lowers composit` |
| 1246 | Discussion | `one` | ROUNDED | rounded; record re-checked, rounded_check row 24 (gentlest dynamics setting lowers composition by 0.531 / 0.510 / 0.388 < 1) | `they move the state: by less than one six-factor tuple per` |
| 1246 | Discussion | `six` | COUNT | count word of discrete things (six-factor tuple per) | `y move the state: by less than one six-factor tuple per` |
| 1247 | Discussion | `forty` | ROUNDED | rounded; record re-checked, rounded_check row 25 (full strength -39.959 / -46.918 / -39.061, seed mean 41.98 -> 40 (nearest ten)) | `etting that moves it, and by about forty on the seed mean at full strength, whil` |
| 1250 | Discussion | `four` | COUNT | count word of discrete things (four named conditions) | `The identity holds under four named conditions (App.~ ).` |
| 1261 | Discussion | `Two` | COUNT | count word of discrete things (Two laws measured) | `\paragraph{Two laws measured on this study's own n-CLU.}` |
| 1263 | Discussion | `half` | NOT-A-NUMBER | the term "half-life" | `ing fitted, it predicts the settle half-life` |
| 1264 | Discussion | `three` | COUNT | count word of discrete things (three seeds (Tab.~\ref{tab:crr}),) | `r on the held-out combinations and three seeds (Tab.~ ), and beats a` |
| 1264 | Discussion | `one` | COUNT | "one-parameter" (a compound: the declared one-tuple margin / one-token lookup / one-parameter form) | `(Tab.~ ), and beats a one-parameter form fitted with sight of the h` |
| 1265 | Discussion | `one` | COUNT | determiner/count "one" (a single thing) | `n every combination on every seed; one of six declared criteria (the V-curve arg` |
| 1265 | Discussion | `six` | COUNT | count word of discrete things (six declared criteria) | `combination on every seed; one of six declared criteria (the V-curve argmin) mi` |
| 1266 | Discussion | `three` | COUNT | seed-outcome tally (a count of seeds; the per-seed values are in the cited table) | `all three seeds, and the law predicts where its s` |
| 1269 | Discussion | `one` | NOT-A-NUMBER | pronoun "one" | `e memory equals the never-inserted one is` |
| 1270 | Discussion | `two` | ROUNDED | rounded; record re-checked, rounded_check row 26 (image differs 65.17% .. 65.39%: below 2/3 (66.67%) and within 10%) | `\textsc{refuted} (nearly two thirds of the image differs, most of it b` |
| 1281 | Discussion | `3` | IDENTIFIER | part of a name (`Shapes3D`: dataset, dtype or criterion name) | `Shapes3D that scores above read-path-matched baseli` |
| 1281 | Discussion | `seven` | ROUNDED | rounded; record re-checked, rounded_check row 3 (Summary: +7.034 rounds to 7) | `s fitted on its own rows, by about seven tuples per` |
| 1289 | Discussion | `0.5,2` | IDENTIFIER | math notation ($s_{t+1}$, $(\cdot)^{-1}$, $\lambda=0$, $[0.5,2]$ bucket) | `$h/h^\star\in[0.5,2]$ bucket, is named and not run.` |
| 1294 | Discussion | `three` | COUNT | the three properties (the paper's frame) | `the settling (physics) CLU on the three properties of` |
| 1295 | Discussion | `zero` | IDENTIFIER | name of the limit / arm (zero-dynamics, zero-step) | `ng, with this paper's n-CLU as its zero-dynamics` |
| 1300 | Discussion | `95` | PROTOCOL/DEFINITION | interval level (95%) | `LSTM on the red sphere, where its 95\%` |
| 1301 | Discussion | `zero` | PROTOCOL/DEFINITION | zero as the null reference of an interval ("above / excludes / crosses zero") | `interval against them includes zero on some seeds (Sec.~` |

The Head-held tokens (Experiments first paragraph: the per-K seed SDs, per-seed footprints and gaps; Results first paragraph: the per-seed margins, the mean +7.034, +3.73, wins/losses, parameter and step counts, baseline ranges) are listed with their lines in `NUMBERS-NOW-DRAFT-v2.md`. They are per-seed and table-readable values that item 5 would otherwise remove; PW-2 may not touch them.

### Rounded values, re-checked against their records (DRAFT-v2; style identical, 32 rows, 0 FAIL)

| # | anchor (main text) | in text | record file(s) | values verbatim | rule | result |
|---|---|---|---|---|---|---|
| 1 | `produces about seven more distinct` | 1 | `w20-shapes-data-matched/SHAPES-MATCHED.md` | 7.034 ✓ | mean paired margin over the MLP +7.034 rounds to 7 | **PASS** |
| 2 | `by about seven tuples per held-out factor combination over the strongest` | 1 | `w20-shapes-data-matched/SHAPES-MATCHED.md` | 7.034 ✓ | Contribution 1: +7.034 rounds to 7 | **PASS** |
| 3 | `by about seven tuples per held-out combination over the strongest` | 1 | `w20-shapes-data-matched/SHAPES-MATCHED.md` | 7.034 ✓ | Summary: +7.034 rounds to 7 | **PASS** |
| 4 | `survivors fall by about six tuples` | 1 | `w20-shapes-data-matched/FINDINGS.md` | 6.13 ✓, 6.43 ✓, 6.30 ✓ | abstract: per-seed survivor losses 6.13 / 6.43 / 6.30 each round to 6 | **PASS** |
| 5 | `surviving combinations by about six tuples per held-out combination, on all three seeds` | 1 | `w20-shapes-data-matched/FINDINGS.md` | 6.13 ✓, 6.43 ✓, 6.30 ✓ | Sec. 4.3: 6.13 / 6.43 / 6.30 each round to 6 | **PASS** |
| 6 | `its own loss on that deletion, about six tuples` | 1 | `w20-shapes-data-matched/FINDINGS.md` | 6.13 ✓, 6.43 ✓, 6.30 ✓ | Sec. 4.4: 6.13 / 6.43 / 6.30 each round to 6 | **PASS** |
| 7 | `surviving combinations by about six tuples per held-out combination, about thirty` | 1 | `w20-shapes-data-matched/FINDINGS.md` | 6.13 ✓, 6.43 ✓, 6.30 ✓, 32.1 ✓ | Limitation 2: 6.13 / 6.43 / 6.30 -> 6; 32.1 -> 30 (nearest ten) | **PASS** |
| 8 | `by about thirty each` | 1 | `w20-shapes-data-matched/FINDINGS.md` | 32.1 ✓ | Sec. 4.3: -32.1 at seed 0 rounds to 30 (nearest ten) | **PASS** |
| 9 | `in another colour by about one` | 1 | `w20-shapes-data-matched/FINDINGS.md` | 1.47 ✓ | Sec. 4.3: -1.47 rounds to 1 | **PASS** |
| 10 | `by at most about a tenth of a tuple` | 2 | `w20-shapes-data-matched/FINDINGS.md` | 0.11 ✓ | largest item-deletion move 0.11 rounds to 0.1 | **PASS** |
| 11 | `screened about four thousand unique works` | 1 | `_thinkpad/w8/w8-novelty/NOVELTY.md` | 4,050 ✓ | literature screen 4,050 rounds to 4,000 | **PASS** |
| 12 | `valid on about nine in ten` | 1 | `../../outputs/pw1-render-shapes/REQUESTS-pw1.json` | 5177 ✓, 5688 ✓ | n-CLU valid 5,177 of 5,688 at seed 0 = 0.910 -> 0.9 | **PASS** |
| 13 | `moves the LSTM by about one tuple` | 1 | `w20-shapes-data-matched/FINDINGS.md` | 20.55 ✓, 21.85 ✓ | LSTM ten-times spot check 20.55 -> 21.85, +1.30 -> 1 | **PASS** |
| 14 | `up to about four novel tuples` | 1 | `w21-ladder-rivals-32k/FINDINGS.md` | 4.13 ✓ | seed-to-seed MLP difference up to 4.13 -> 4 | **PASS** |
| 15 | `rise, nearly five tuples` | 1 | `w15-coverage-control/FINDINGS.md` | 4.989 ✓ | TEST rise n=3 -> 6 +4.989: below five and within 10% | **PASS** |
| 16 | `moves by about 0.6 relative RMS` | 1 | `w13-seeds-and-tables/TABLES.md` | 0.610083 ✓, 0.614277 ✓ | relative RMS 0.610083 (red) / 0.614277 (others) -> 0.6 | **PASS** |
| 17 | `just under 1.4 times the reference` | 1 | `w13-seeds-and-tables/TABLES.md` | 712,072 ✓, 719,352 ✓, 524,288 ✓ | composition-only bytes 712,072..719,352 / 524,288 = 1.358..1.372 < 1.4 | **PASS** |
| 18 | `within half a percent` | 1 | `w13-seeds-and-tables/TABLES.md` | -0.304% ✓, +0.495% ✓ | byte/parameter match -0.304% .. +0.495%, all |.| < 0.5% | **PASS** |
| 19 | `more than eighteen tuples` | 1 | `w13-seeds-and-tables/TABLES.md` | -21.0816 ✓, -16.5306 ✓, -18.0816 ✓ | item memory row 4 minus learned reference, 3-seed mean of 21.08 / 16.53 / 18.08 = 18.56 > 18 | **PASS** |
| 20 | `at about four times the reference bytes` | 1 | `w15-lace-poe-rows/FINDINGS.md` | 4.09 ✓, 4.11 ✓ | nearest-prototype ceiling at 4.09 .. 4.11 x the reference -> 4 | **PASS** |
| 21 | `accuracy of about nine in ten` | 1 | `w21-maze-t-seed4/FINDINGS.md`; `w21-rel-newseeds/FINDINGS.md` | 0.900 ✓, 0.930 ✓ | per-step accuracy 0.900 .. 0.930 -> 0.9 | **PASS** |
| 22 | `at $Kstar$ is about four times the reference` | 1 | `w13-referee-2/REFEREE-v1.md` | 4.0897 ✓, 4.0768 ✓, 4.0847 ✓ | deletable object 4.0897 / 4.0768 / 4.0847 x the reference -> 4 | **PASS** |
| 23 | `still gains more than a tuple` | 1 | `w20-shapes-data-matched/FINDINGS.md` | 1.47 ✓ | MLP seed 0, ten -> thirty times its steps +1.47 > 1 | **PASS** |
| 24 | `by less than one six-factor tuple` | 1 | `w12-clue-theorist/CLUE.md` | 0.531 ✓, 0.510 ✓, 0.388 ✓ | gentlest dynamics setting lowers composition by 0.531 / 0.510 / 0.388 < 1 | **PASS** |
| 25 | `by about forty on the seed mean` | 1 | `w12-clue-theorist/CLUE.md` | 39.959 ✓, 46.918 ✓, 39.061 ✓ | full strength -39.959 / -46.918 / -39.061, seed mean 41.98 -> 40 (nearest ten) | **PASS** |
| 26 | `nearly two thirds of the image differs` | 1 | `w10-physics-section-evidence/EVIDENCE.md` | 65.17 ✓, 65.39 ✓ | image differs 65.17% .. 65.39%: below 2/3 (66.67%) and within 10% | **PASS** |
| 27 | `refit takes longer than a baseline` | 1 | `w20-shapes-data-matched/FINDINGS.md`; `w20-shapes-data-matched/SHAPES-MATCHED.md` | 2,367 ✓, 2064.5 ✓ | fastest book refit 2,367 s > slowest baseline retrain 2,064.5 s | **PASS** |
| 28 | `score far below one distinct valid novel tuple` | 1 | `w15-lace-poe-rows/FINDINGS.md` | 0.117 ✓, 0.033 ✓, 0.267 ✓, 0.224 ✓, 0.122 ✓, 0.184 ✓, 0.020 ✓ | LACE rows 3-4 per seed 0.117 / 0.033 / 0.267 and 0.224 / 0.122 / 0.184, PoE at most 0.020: all below a third of a tuple | **PASS** |
| 29 | `(more than four of the six claim atoms|reached by any single work is four)` | 1 | `_thinkpad/w8/w8-novelty/NOVELTY.md` | 4 of 6 ✓ | an exact count: highest claim-atom count of any single work "4 of 6" (the record) | **PASS** |
| 30 | `younger than about three months` | 1 | `_thinkpad/w8/w8-novelty/NOVELTY.md` | < ~3 months ✓ | the record itself states the approximate window "< ~3 months" | **PASS** |
| 31 | `rows, four times the n-CLU` | 1 | `w21-ladder-rivals-32k/CAPTION.md` | 131,072 ✓, 32,768 ✓ | an exact ratio: data-matched pool 131,072 rows / n-CLU content rows 32,768 = 4 | **PASS** |
| 32 | `retained statistic is nearly a gigabyte` | 2 | `w20-shapes-data-matched/SHAPES-MATCHED.json` | 967304800 ✓ | 967,304,800 B = 0.967 GB (decimal): below 1 GB, within 10% | **PASS** |

32 rows, 0 FAIL


### Disagreements with pw2-draft-numbers / pw2-draft-leftovers (`NUMBERS-LEFT.md`, the latest list)

#### DRAFT-v2.tex: token match

- this audit: 383 tokens (198 digit, 185 word); pw2-draft-leftovers NUMBERS-LEFT: 358 tokens (174 digit, 184 word)
- matched on (line, number): 350; only in this audit: 33; only in NUMBERS-LEFT: 8

| line | number | only in | this audit's class / their justification | code |
|---|---|---|---|---|
| 100 | `3` ×1 | this audit | IDENTIFIER: part of a name (`Shapes3D`: dataset, dtype or criterion name) — **tokeniser: digits glued to letters or a macro (their regex skips a digit preceded by a letter)** | `ithout those rows. On a Shapes3D value hold-out, at the finest partiti` |
| 230 | `3` ×1 | this audit | IDENTIFIER: part of a name (`Shapes3D`: dataset, dtype or criterion name) — **tokeniser: digits glued to letters or a macro (their regex skips a digit preceded by a letter)** | `Shapes3D~` |
| 236 | `3` ×1 | this audit | IDENTIFIER: rung name ($n$-way ladder) — **tokeniser: digits glued to letters or a macro (their regex skips a digit preceded by a letter)** | `). For $n\ge3$ the rungs are a training-data axis.` |
| 263 | `3` ×2 | this audit | IDENTIFIER: part of a name (`Shapes3D`: dataset, dtype or criterion name) — **tokeniser: digits glued to letters or a macro (their regex skips a digit preceded by a letter)** | `on one environment (Shapes3D) at three scored seeds; no Shapes3D r` |
| 308 | `64` ×1 | this audit | IDENTIFIER: part of a name (`float64`: dataset, dtype or criterion name) — **tokeniser: digits glued to letters or a macro (their regex skips a digit preceded by a letter)** | `tion (symmetric-triangle float64 for the statistic, unless marked` |
| 311 | `32` ×1 | this audit | IDENTIFIER: part of a name (`float32`: dataset, dtype or criterion name) — **tokeniser: digits glued to letters or a macro (their regex skips a digit preceded by a letter)** | `about 131,072 float32 parameters at it.` |
| 327 | `3` ×1 | this audit | IDENTIFIER: part of a name (`Shapes3D`: dataset, dtype or criterion name) — **tokeniser: digits glued to letters or a macro (their regex skips a digit preceded by a letter)** | `e and the estimand.} On Shapes3D an external classifier, hash-pinned p` |
| 335 | `1` ×1 | this audit | IDENTIFIER: part of a name (`S1`: dataset, dtype or criterion name) — **tokeniser: digits glued to letters or a macro (their regex skips a digit preceded by a letter)** | `tion (for example, criterion S1 or criterion S2).` |
| 335 | `2` ×1 | this audit | IDENTIFIER: part of a name (`S2`: dataset, dtype or criterion name) — **tokeniser: digits glued to letters or a macro (their regex skips a digit preceded by a letter)** | `e, criterion S1 or criterion S2).` |
| 406 | `3` ×1 | this audit | IDENTIFIER: part of a name (`Shapes3D`: dataset, dtype or criterion name) — **tokeniser: digits glued to letters or a macro (their regex skips a digit preceded by a letter)** | `\subsection{The Shapes3D value hold-out}` |
| 410 | `3` ×1 | this audit | HEAD-HELD: Experiments, first paragraph (preceded by the float chunk at lines 370–404) — **tokeniser: digits glued to letters or a macro (their regex skips a digit preceded by a letter)** | `on the Shapes3D value hold-out~` |
| 412 | `5,8,10,12,16,20,24,28,32` ×1 | this audit | HEAD-HELD: Experiments, first paragraph (preceded by the float chunk at lines 370–404) — **tokeniser: a comma list or a repeated digit that the two tokenisers split or count differently** | `0.02\}$, over $K\in\{5,8,10,12,16,20,24,28,32\}$. Every swept` |
| 416 | `32` ×1 | this audit | HEAD-HELD: Experiments, first paragraph (preceded by the float chunk at lines 370–404) — **tokeniser: digits glued to letters or a macro (their regex skips a digit preceded by a letter)** | `$K=5\ldots32$: the seed standard deviation of vali` |
| 458 | `3` ×1 | this audit | IDENTIFIER: rung name ($n$-way ladder) — **tokeniser: digits glued to letters or a macro (their regex skips a digit preceded by a letter)** | `rungs $n\ge3$ are exactly constant by construction.` |
| 506 | `3` ×1 | this audit | IDENTIFIER: part of a name (`Shapes3D`: dataset, dtype or criterion name) — **tokeniser: digits glued to letters or a macro (their regex skips a digit preceded by a letter)** | `bsection{Composition on Shapes3D}` |
| 783 | `3` ×1 | this audit | IDENTIFIER: rung name ($n$-way ladder) — **tokeniser: digits glued to letters or a macro (their regex skips a digit preceded by a letter)** | `\textbf{Result.} For $n\ge3$ the ladder's rise is fit-pool` |
| 790 | `6` ×1 | this audit | IDENTIFIER: rung name ($n$-way ladder) — **tokeniser: digits glued to letters or a macro (their regex skips a digit preceded by a letter)** | `the whole $n=3\to6$ rise, nearly five tuples per held-out` |
| 820 | `3` ×1 | this audit | IDENTIFIER: part of a name (`Shapes3D`: dataset, dtype or criterion name) — **tokeniser: digits glued to letters or a macro (their regex skips a digit preceded by a letter)** | `\subsection{Deletion on Shapes3D}` |
| 889 | `3` ×1 | this audit | IDENTIFIER: part of a name (`Shapes3D`: dataset, dtype or criterion name) — **tokeniser: digits glued to letters or a macro (their regex skips a digit preceded by a letter)** | `le aggregates and every Shapes3D tuple is one row, so its statistic is` |
| 1034 | `eighteen` ×1 | this audit | ROUNDED: rounded; record re-checked, rounded_check row 19 (item memory row 4 minus learned reference, 3-seed mean of 21.08 / 16.53 / 18.08 = 18.56 > 18) — **word not in their word list** | `reference by more than eighteen tuples per held-out combination` |
| 1066 | `3` ×1 | this audit | IDENTIFIER: part of a name (`Shapes3D`: dataset, dtype or criterion name) — **tokeniser: digits glued to letters or a macro (their regex skips a digit preceded by a letter)** | `ation table is a second Shapes3D instrument: a fixed set of` |
| 1070 | `1` ×1 | this audit | IDENTIFIER: part of a name (`S1`: dataset, dtype or criterion name) — **tokeniser: digits glued to letters or a macro (their regex skips a digit preceded by a letter)** | `ery seed (declared criterion S1,` |
| 1071 | `2` ×1 | this audit | IDENTIFIER: part of a name (`S2`: dataset, dtype or criterion name) — **tokeniser: digits glued to letters or a macro (their regex skips a digit preceded by a letter)** | `is table. Declared criterion S2, against the earlier` |
| 1075 | `3` ×1 | this audit | IDENTIFIER: part of a name (`S3`: dataset, dtype or criterion name) — **tokeniser: digits glued to letters or a macro (their regex skips a digit preceded by a letter)** | `(S3 \textsc{lose}) at about four times the` |
| 1076 | `4` ×1 | this audit | IDENTIFIER: part of a name (`S4`: dataset, dtype or criterion name) — **tokeniser: digits glued to letters or a macro (their regex skips a digit preceded by a letter)** | `and PoE on the mean (S4; Tab.~ ).` |
| 1151 | `5` ×1 | this audit | IDENTIFIER: panel crop size notation ($5\times5$) — **tokeniser: a comma list or a repeated digit that the two tokenisers split or count differently** | `is a $5\times5$ crop of the maze centred on th` |
| 1198 | `36` ×1 | this audit | IDENTIFIER: partition size $K$ (design name / sweep grid) — **tokeniser: digits glued to letters or a macro (their regex skips a digit preceded by a letter)** | `e one configuration at $K\ge36$ whose composition` |
| 1200 | `64` ×1 | this audit | IDENTIFIER: part of a name (`float64`: dataset, dtype or criterion name) — **tokeniser: digits glued to letters or a macro (their regex skips a digit preceded by a letter)** | `No object with a float64 sufficient statistic composes and forge` |
| 1216 | `3` ×1 | this audit | IDENTIFIER: part of a name (`Shapes3D`: dataset, dtype or criterion name) — **tokeniser: digits glued to letters or a macro (their regex skips a digit preceded by a letter)** | `{At most three seeds on Shapes3D.} Seeds 3 and 4 cannot be judged ther` |
| 1217 | `3` ×1 | this audit | IDENTIFIER: part of a name (`Shapes3D`: dataset, dtype or criterion name) — **tokeniser: digits glued to letters or a macro (their regex skips a digit preceded by a letter)** | `al classifier), so no Shapes3D row can exceed three seeds; CMAZE-T pri` |
| 1281 | `3` ×1 | this audit | IDENTIFIER: part of a name (`Shapes3D`: dataset, dtype or criterion name) — **tokeniser: digits glued to letters or a macro (their regex skips a digit preceded by a letter)** | `Shapes3D that scores above read-path-matched b` |
| 1289 | `0.5,2` ×1 | this audit | IDENTIFIER: math notation ($s_{t+1}$, $(\cdot)^{-1}$, $\lambda=0$, $[0.5,2]$ bucket) — **tokeniser: a comma list or a repeated digit that the two tokenisers split or count differently** | `$h/h^\star\in[0.5,2]$ bucket, is named and not run.` |
| 308 | `4` ×1 | NUMBERS-LEFT | identifier: tokeniser artefact inside the type name float64 / float32 — **tokeniser: a piece of a token this audit lists whole** | |
| 311 | `2` ×1 | NUMBERS-LEFT | definition: parameters at the byte reference (524,288 / 4), part of the same definition — **tokeniser: a piece of a token this audit lists whole** | |
| 412 | `8,10,12,16,20,24,28,32` ×1 | NUMBERS-LEFT | Head-held (Experiments, first paragraph (preceded by the float chunk at lines 370–404)) — **tokeniser: a piece of a token this audit lists whole** | |
| 416 | `2` ×1 | NUMBERS-LEFT | Head-held (Experiments, first paragraph (preceded by the float chunk at lines 370–404)) — **tokeniser: a piece of a token this audit lists whole** | |
| 1198 | `6` ×1 | NUMBERS-LEFT | identifier: the partition range $K\ge36$ (the partitions that fit the byte reference; the measured one is in Tab. kcurve caption) — **tokeniser: a piece of a token this audit lists whole** | |
| 1200 | `4` ×1 | NUMBERS-LEFT | identifier: tokeniser artefact inside the type name float64 / float32 — **tokeniser: a piece of a token this audit lists whole** | |
| 1289 | `0.5` ×1 | NUMBERS-LEFT | identifier: named bucket $h/h^\star\in[0.5,2]$ — **tokeniser: a piece of a token this audit lists whole** | |
| 1289 | `2` ×1 | NUMBERS-LEFT | identifier: named bucket $h/h^\star\in[0.5,2]$ — **tokeniser: a piece of a token this audit lists whole** | |

#### DRAFT-v2.tex: class agreement on matched tokens (coarse scale)

- agree: 333; disagree: 17
  - this audit ID/PROTOCOL / NUMBERS-LEFT COUNT: 12
  - this audit ROUNDED / NUMBERS-LEFT COUNT: 3
  - this audit NOT-A-NUMBER / NUMBERS-LEFT COUNT: 2

| line | number | this audit | NUMBERS-LEFT | code |
|---|---|---|---|---|
| 155 | `half` | NOT-A-NUMBER — "either half" (of a two-part claim) | count word, high-level (not table-readable) | `book. We do not claim either half. In particular we do not claim` |
| 170 | `thousand` | ROUNDED — rounded; record re-checked, rounded_check row 11 (literature screen 4,050 rounds to 4,000) | count word, high-level (not table-readable) | `about four thousand unique works: those citing the` |
| 184 | `three` | ROUNDED — rounded; record re-checked, rounded_check row 30 (the record itself states the approximate window "< ~3 months") | count word, high-level (not table-readable) | `three months is incomplete (Sec.~` |
| 758 | `zero` | PROTOCOL/DEFINITION — zero as the null reference of an interval ("above / excludes / crosses zero") | count word, high-level (not table-readable) | `lies above zero on every rung and seed` |
| 760 | `ten` | IDENTIFIER — schedule name ("ten times" / "thirty times" its selected steps) | count word, high-level (not table-readable) | `roven converged everywhere. A ten-times spot check` |
| 767 | `four` | ROUNDED — rounded; record re-checked, rounded_check row 31 (an exact ratio: data-matched pool 131,072 rows / n-CLU content rows 32,768 = 4) | count word, high-level (not table-readable) | `the data-matched set's rows, four times the n-CLU's` |
| 772 | `zero` | PROTOCOL/DEFINITION — zero as the null reference of an interval ("above / excludes / crosses zero") | count word, high-level (not table-readable) | `margins lies above zero.` |
| 773 | `ten` | IDENTIFIER — schedule name ("ten times" / "thirty times" its selected steps) | count word, high-level (not table-readable) | `and seed and is flat from its ten-times fit to its` |
| 777 | `zero` | PROTOCOL/DEFINITION — zero as the null reference of an interval ("above / excludes / crosses zero") | count word, high-level (not table-readable) | `combination interval excludes zero on most` |
| 936 | `zero` | PROTOCOL/DEFINITION — zero as the null reference of an interval ("above / excludes / crosses zero") | count word, high-level (not table-readable) | `zero. After the red-sphere deletion it i` |
| 937 | `zero` | PROTOCOL/DEFINITION — zero as the null reference of an interval ("above / excludes / crosses zero") | count word, high-level (not table-readable) | `zero against the MLP at seeds 1 and 2 an` |
| 1120 | `zero` | PROTOCOL/DEFINITION — structural zero (the colour-naming columns are zero by construction) | count word, high-level (not table-readable) | `lumn that names the colour is zero, so the value row stays a` |
| 1206 | `ten` | IDENTIFIER — schedule name ("from ten to thirty times its steps") | count word, high-level (not table-readable) | `r held-out combination from ten to thirty times its steps (Sec.~` |
| 1206 | `thirty` | IDENTIFIER — schedule name ("ten times" / "thirty times" its selected steps) | count word, high-level (not table-readable) | `out combination from ten to thirty times its steps (Sec.~` |
| 1236 | `zero` | PROTOCOL/DEFINITION — definition of the limit (settling step count set to zero) | count word, high-level (not table-readable) | `ling loop's step count set to zero. Inside a well this is an` |
| 1263 | `half` | NOT-A-NUMBER — the term "half-life" | count word, high-level (not table-readable) | `itted, it predicts the settle half-life` |
| 1301 | `zero` | PROTOCOL/DEFINITION — zero as the null reference of an interval ("above / excludes / crosses zero") | count word, high-level (not table-readable) | `nterval against them includes zero on some seeds (Sec.~` |


#### DRAFT-v2-style.tex: token match

- this audit: 381 tokens (199 digit, 182 word); pw2-draft-leftovers NUMBERS-LEFT: 355 tokens (174 digit, 181 word)
- matched on (line, number): 347; only in this audit: 34; only in NUMBERS-LEFT: 8

| line | number | only in | this audit's class / their justification | code |
|---|---|---|---|---|
| 104 | `3` ×1 | this audit | IDENTIFIER: part of a name (`Shapes3D`: dataset, dtype or criterion name) — **tokeniser: digits glued to letters or a macro (their regex skips a digit preceded by a letter)** | `On a Shapes3D value hold-out, at the finest partiti` |
| 231 | `3` ×1 | this audit | IDENTIFIER: part of a name (`Shapes3D`: dataset, dtype or criterion name) — **tokeniser: digits glued to letters or a macro (their regex skips a digit preceded by a letter)** | `Shapes3D~` |
| 237 | `3` ×1 | this audit | IDENTIFIER: rung name ($n$-way ladder) — **tokeniser: digits glued to letters or a macro (their regex skips a digit preceded by a letter)** | `). For $n\ge3$ the rungs are a training-data axis.` |
| 264 | `3` ×2 | this audit | IDENTIFIER: part of a name (`Shapes3D`: dataset, dtype or criterion name) — **tokeniser: digits glued to letters or a macro (their regex skips a digit preceded by a letter)** | `on one environment (Shapes3D) at three scored seeds; no Shapes3D r` |
| 309 | `64` ×1 | this audit | IDENTIFIER: part of a name (`float64`: dataset, dtype or criterion name) — **tokeniser: digits glued to letters or a macro (their regex skips a digit preceded by a letter)** | `(symmetric-triangle float64 for the statistic, unless marked othe` |
| 311 | `32` ×1 | this audit | IDENTIFIER: part of a name (`float32`: dataset, dtype or criterion name) — **tokeniser: digits glued to letters or a macro (their regex skips a digit preceded by a letter)** | `ines carry about 131,072 float32 parameters at it.` |
| 327 | `3` ×1 | this audit | IDENTIFIER: part of a name (`Shapes3D`: dataset, dtype or criterion name) — **tokeniser: digits glued to letters or a macro (their regex skips a digit preceded by a letter)** | `e and the estimand.} On Shapes3D an external classifier, hash-pinned p` |
| 334 | `1` ×1 | this audit | IDENTIFIER: part of a name (`S1`: dataset, dtype or criterion name) — **tokeniser: digits glued to letters or a macro (their regex skips a digit preceded by a letter)** | `tion (for example, criterion S1 or criterion S2).` |
| 334 | `2` ×1 | this audit | IDENTIFIER: part of a name (`S2`: dataset, dtype or criterion name) — **tokeniser: digits glued to letters or a macro (their regex skips a digit preceded by a letter)** | `e, criterion S1 or criterion S2).` |
| 404 | `3` ×1 | this audit | IDENTIFIER: part of a name (`Shapes3D`: dataset, dtype or criterion name) — **tokeniser: digits glued to letters or a macro (their regex skips a digit preceded by a letter)** | `\subsection{The Shapes3D value hold-out}` |
| 407 | `3` ×1 | this audit | HEAD-HELD: Experiments, first paragraph (preceded by the float chunk at lines 368–402) — **tokeniser: digits glued to letters or a macro (their regex skips a digit preceded by a letter)** | `the three properties on Shapes3D. The deletable object is the \emph{si` |
| 408 | `3` ×1 | this audit | HEAD-HELD: Experiments, first paragraph (preceded by the float chunk at lines 368–402) — **tokeniser: digits glued to letters or a macro (their regex skips a digit preceded by a letter)** | `on the Shapes3D value hold-out~` |
| 410 | `5,8,10,12,16,20,24,28,32` ×1 | this audit | HEAD-HELD: Experiments, first paragraph (preceded by the float chunk at lines 368–402) — **tokeniser: a comma list or a repeated digit that the two tokenisers split or count differently** | `0.02\}$, over $K\in\{5,8,10,12,16,20,24,28,32\}$. Every swept` |
| 414 | `32` ×1 | this audit | HEAD-HELD: Experiments, first paragraph (preceded by the float chunk at lines 368–402) — **tokeniser: digits glued to letters or a macro (their regex skips a digit preceded by a letter)** | `$K=5\ldots32$. The seed standard deviation of vali` |
| 456 | `3` ×1 | this audit | IDENTIFIER: rung name ($n$-way ladder) — **tokeniser: digits glued to letters or a macro (their regex skips a digit preceded by a letter)** | `rungs $n\ge3$ are exactly constant by construction.` |
| 504 | `3` ×1 | this audit | IDENTIFIER: part of a name (`Shapes3D`: dataset, dtype or criterion name) — **tokeniser: digits glued to letters or a macro (their regex skips a digit preceded by a letter)** | `bsection{Composition on Shapes3D}` |
| 781 | `3` ×1 | this audit | IDENTIFIER: rung name ($n$-way ladder) — **tokeniser: digits glued to letters or a macro (their regex skips a digit preceded by a letter)** | `\textbf{Result.} For $n\ge3$ the ladder's rise is fit-pool` |
| 788 | `6` ×1 | this audit | IDENTIFIER: rung name ($n$-way ladder) — **tokeniser: digits glued to letters or a macro (their regex skips a digit preceded by a letter)** | `the whole $n=3\to6$ rise, nearly five tuples per held-out` |
| 818 | `3` ×1 | this audit | IDENTIFIER: part of a name (`Shapes3D`: dataset, dtype or criterion name) — **tokeniser: digits glued to letters or a macro (their regex skips a digit preceded by a letter)** | `\subsection{Deletion on Shapes3D}` |
| 887 | `3` ×1 | this audit | IDENTIFIER: part of a name (`Shapes3D`: dataset, dtype or criterion name) — **tokeniser: digits glued to letters or a macro (their regex skips a digit preceded by a letter)** | `le aggregates and every Shapes3D tuple is one row, so its statistic is` |
| 1028 | `eighteen` ×1 | this audit | ROUNDED: rounded; record re-checked, rounded_check row 19 (item memory row 4 minus learned reference, 3-seed mean of 21.08 / 16.53 / 18.08 = 18.56 > 18) — **word not in their word list** | `reference by more than eighteen tuples per held-out combination` |
| 1060 | `3` ×1 | this audit | IDENTIFIER: part of a name (`Shapes3D`: dataset, dtype or criterion name) — **tokeniser: digits glued to letters or a macro (their regex skips a digit preceded by a letter)** | `ation table is a second Shapes3D instrument: a fixed set of` |
| 1064 | `1` ×1 | this audit | IDENTIFIER: part of a name (`S1`: dataset, dtype or criterion name) — **tokeniser: digits glued to letters or a macro (their regex skips a digit preceded by a letter)** | `ery seed (declared criterion S1,` |
| 1065 | `2` ×1 | this audit | IDENTIFIER: part of a name (`S2`: dataset, dtype or criterion name) — **tokeniser: digits glued to letters or a macro (their regex skips a digit preceded by a letter)** | `is table. Declared criterion S2, against the earlier` |
| 1069 | `3` ×1 | this audit | IDENTIFIER: part of a name (`S3`: dataset, dtype or criterion name) — **tokeniser: digits glued to letters or a macro (their regex skips a digit preceded by a letter)** | `(S3 \textsc{lose}) at about four times the` |
| 1070 | `4` ×1 | this audit | IDENTIFIER: part of a name (`S4`: dataset, dtype or criterion name) — **tokeniser: digits glued to letters or a macro (their regex skips a digit preceded by a letter)** | `and PoE on the mean (S4; Tab.~ ).` |
| 1145 | `5` ×1 | this audit | IDENTIFIER: panel crop size notation ($5\times5$) — **tokeniser: a comma list or a repeated digit that the two tokenisers split or count differently** | `is a $5\times5$ crop of the maze centred on th` |
| 1192 | `36` ×1 | this audit | IDENTIFIER: partition size $K$ (design name / sweep grid) — **tokeniser: digits glued to letters or a macro (their regex skips a digit preceded by a letter)** | `e one configuration at $K\ge36$ whose composition` |
| 1194 | `64` ×1 | this audit | IDENTIFIER: part of a name (`float64`: dataset, dtype or criterion name) — **tokeniser: digits glued to letters or a macro (their regex skips a digit preceded by a letter)** | `No object with a float64 sufficient statistic composes and forge` |
| 1210 | `3` ×1 | this audit | IDENTIFIER: part of a name (`Shapes3D`: dataset, dtype or criterion name) — **tokeniser: digits glued to letters or a macro (their regex skips a digit preceded by a letter)** | `{At most three seeds on Shapes3D.} Seeds 3 and 4 cannot be judged ther` |
| 1211 | `3` ×1 | this audit | IDENTIFIER: part of a name (`Shapes3D`: dataset, dtype or criterion name) — **tokeniser: digits glued to letters or a macro (their regex skips a digit preceded by a letter)** | `al classifier), so no Shapes3D row can exceed three seeds; CMAZE-T pri` |
| 1271 | `3` ×1 | this audit | IDENTIFIER: part of a name (`Shapes3D`: dataset, dtype or criterion name) — **tokeniser: digits glued to letters or a macro (their regex skips a digit preceded by a letter)** | `Shapes3D that scores above read-path-matched b` |
| 1279 | `0.5,2` ×1 | this audit | IDENTIFIER: math notation ($s_{t+1}$, $(\cdot)^{-1}$, $\lambda=0$, $[0.5,2]$ bucket) — **tokeniser: a comma list or a repeated digit that the two tokenisers split or count differently** | `$h/h^\star\in[0.5,2]$ bucket, is named and not run.` |
| 309 | `4` ×1 | NUMBERS-LEFT | identifier: tokeniser artefact inside the type name float64 / float32 — **tokeniser: a piece of a token this audit lists whole** | |
| 311 | `2` ×1 | NUMBERS-LEFT | definition: parameters at the byte reference (524,288 / 4), part of the same definition — **tokeniser: a piece of a token this audit lists whole** | |
| 410 | `8,10,12,16,20,24,28,32` ×1 | NUMBERS-LEFT | Head-held (Experiments, first paragraph (preceded by the float chunk at lines 368–402)) — **tokeniser: a piece of a token this audit lists whole** | |
| 414 | `2` ×1 | NUMBERS-LEFT | Head-held (Experiments, first paragraph (preceded by the float chunk at lines 368–402)) — **tokeniser: a piece of a token this audit lists whole** | |
| 1192 | `6` ×1 | NUMBERS-LEFT | identifier: the partition range $K\ge36$ (the partitions that fit the byte reference; the measured one is in Tab. kcurve caption) — **tokeniser: a piece of a token this audit lists whole** | |
| 1194 | `4` ×1 | NUMBERS-LEFT | identifier: tokeniser artefact inside the type name float64 / float32 — **tokeniser: a piece of a token this audit lists whole** | |
| 1279 | `0.5` ×1 | NUMBERS-LEFT | identifier: named bucket $h/h^\star\in[0.5,2]$ — **tokeniser: a piece of a token this audit lists whole** | |
| 1279 | `2` ×1 | NUMBERS-LEFT | identifier: named bucket $h/h^\star\in[0.5,2]$ — **tokeniser: a piece of a token this audit lists whole** | |

#### DRAFT-v2-style.tex: class agreement on matched tokens (coarse scale)

- agree: 329; disagree: 18
  - this audit ID/PROTOCOL / NUMBERS-LEFT COUNT: 12
  - this audit ROUNDED / NUMBERS-LEFT COUNT: 4
  - this audit NOT-A-NUMBER / NUMBERS-LEFT COUNT: 2

| line | number | this audit | NUMBERS-LEFT | code |
|---|---|---|---|---|
| 157 | `half` | NOT-A-NUMBER — "either half" (of a two-part claim) | count word, high-level (not table-readable) | `ddress book. We claim neither half.` |
| 172 | `thousand` | ROUNDED — rounded; record re-checked, rounded_check row 11 (literature screen 4,050 rounds to 4,000) | count word, high-level (not table-readable) | `screened about four thousand unique works: those citing the` |
| 177 | `four` | ROUNDED — rounded; record re-checked, rounded_check row 29 (an exact count: highest claim-atom count of any single work "4 of 6" (the record)) | count word, high-level (not table-readable) | `single work reaches more than four of the six claim atoms;` |
| 186 | `three` | ROUNDED — rounded; record re-checked, rounded_check row 30 (the record itself states the approximate window "< ~3 months") | count word, high-level (not table-readable) | `than about three months is incomplete (Sec.~` |
| 756 | `zero` | PROTOCOL/DEFINITION — zero as the null reference of an interval ("above / excludes / crosses zero") | count word, high-level (not table-readable) | `lies above zero on every rung and seed` |
| 758 | `ten` | IDENTIFIER — schedule name ("ten times" / "thirty times" its selected steps) | count word, high-level (not table-readable) | `roven converged everywhere. A ten-times spot check` |
| 765 | `four` | ROUNDED — rounded; record re-checked, rounded_check row 31 (an exact ratio: data-matched pool 131,072 rows / n-CLU content rows 32,768 = 4) | count word, high-level (not table-readable) | `the data-matched set's rows, four times the n-CLU's` |
| 770 | `zero` | PROTOCOL/DEFINITION — zero as the null reference of an interval ("above / excludes / crosses zero") | count word, high-level (not table-readable) | `margins lies above zero.` |
| 771 | `ten` | IDENTIFIER — schedule name ("ten times" / "thirty times" its selected steps) | count word, high-level (not table-readable) | `and seed and is flat from its ten-times fit to its` |
| 775 | `zero` | PROTOCOL/DEFINITION — zero as the null reference of an interval ("above / excludes / crosses zero") | count word, high-level (not table-readable) | `combination interval excludes zero on most` |
| 934 | `zero` | PROTOCOL/DEFINITION — zero as the null reference of an interval ("above / excludes / crosses zero") | count word, high-level (not table-readable) | `zero. After the red-sphere deletion it i` |
| 935 | `zero` | PROTOCOL/DEFINITION — zero as the null reference of an interval ("above / excludes / crosses zero") | count word, high-level (not table-readable) | `zero against the MLP at seeds 1 and 2 an` |
| 1114 | `zero` | PROTOCOL/DEFINITION — structural zero (the colour-naming columns are zero by construction) | count word, high-level (not table-readable) | `lumn that names the colour is zero, so the value row stays a` |
| 1200 | `ten` | IDENTIFIER — schedule name ("from ten to thirty times its steps") | count word, high-level (not table-readable) | `r held-out combination from ten to thirty times its steps (Sec.~` |
| 1200 | `thirty` | IDENTIFIER — schedule name ("ten times" / "thirty times" its selected steps) | count word, high-level (not table-readable) | `out combination from ten to thirty times its steps (Sec.~` |
| 1231 | `zero` | PROTOCOL/DEFINITION — definition of the limit (settling step count set to zero) | count word, high-level (not table-readable) | `ling loop's step count set to zero. Inside a well this is an algebraic` |
| 1256 | `half` | NOT-A-NUMBER — the term "half-life" | count word, high-level (not table-readable) | `itted, it predicts the settle half-life` |
| 1289 | `zero` | PROTOCOL/DEFINITION — zero as the null reference of an interval ("above / excludes / crosses zero") | count word, high-level (not table-readable) | `nterval against them includes zero on some seeds (Sec.~` |


**Substantive disagreement: one.** "more than **eighteen** tuples per held-out combination" (Results, Item memory; v2 l.1034, style l.1028) is a number in the main text that `NUMBERS-LEFT.md` does not list (its word list stops before "eighteen"). It is a rounded floor, and its record holds (3-seed mean 18.56 > 18; `w13-seeds-and-tables/TABLES.md`, rounded_check row 19). Every other token-level difference is the tokeniser: `NUMBERS-LEFT` skips a digit glued to letters (Shapes3D, S1–S4, float64, `$n\ge3$`, `$K\ge36$`, `$n=3\to6$`) and splits comma lists differently. The class differences are taxonomy, not substance: this audit classes "zero" in "the interval lies above zero" as a reference point (not a count), "ten times / thirty times its steps" as the schedule name, "either half" and "half-life" as not numbers, and "about four thousand", "about three months", "four times the n-CLU's content rows" (and, in the style twin, "more than four of the six claim atoms") as rounded or exact values with a record (all re-checked) where `NUMBERS-LEFT` calls them count words. No number was found that item 5 should have moved and did not, outside Head-held text.

## 7. Per-tell counts per section, now (pw2-draft-tells' detectors, unchanged)

Columns: pw2-draft-tells' before → after (its `TELLS-COUNT.md`) → **now**. `[nh]` = hits in Head-held text. ◆ = now differs from after (changed by pw2-draft-leftovers). Main-text sections below; every appendix row is in `TELLS-NOW.md`.

#### T1 Triads
| section | DRAFT-v2 before → after → **now** | DRAFT-v2-style before → after → **now** |
|---|---|---|
| Abstract | 1 → 1 [1h] → **1** [1h] | 1 → 1 [1h] → **1** [1h] |
| Introduction | 2 → 2 → **2** | 2 → 2 → **2** |
| Methods | 0 → 0 → **0** | 0 → 0 → **0** |
| Experiments | 1 → 1 → **1** | 1 → 1 → **1** |
| Results | 0 → 0 → **0** | 0 → 0 → **0** |
| Discussion | 1 → 0 → **0** | 0 → 0 → **0** |
| Statements | 0 → 0 → **0** | 0 → 0 → **0** |
| **total** | 6 → 5 → **5** [1h] | 5 → 5 → **5** [1h] |

#### T2 Contrast constructions
| section | DRAFT-v2 before → after → **now** | DRAFT-v2-style before → after → **now** |
|---|---|---|
| Abstract | 1 → 1 → **1** | 1 → 1 → **1** |
| Introduction | 5 → 2 [1h] → **2** [1h] | 5 → 2 [1h] → **2** [1h] |
| Methods | 0 → 0 → **0** | 0 → 0 → **0** |
| Experiments | 4 → 1 → **1** | 4 → 1 → **1** |
| Results | 14 → 2 [1h] → **2** [1h] | 14 → 2 [1h] → **2** [1h] |
| Discussion | 5 → 1 → **1** | 5 → 1 → **1** |
| Statements | 1 → 1 → **1** | 1 → 1 → **1** |
| **total** | 49 → 20 → **19** [2h] | 49 → 19 → **18** [2h] |

#### T3 Self-narration of discipline
| section | DRAFT-v2 before → after → **now** | DRAFT-v2-style before → after → **now** |
|---|---|---|
| Abstract | 2 → 0 → **0** | 2 → 0 → **0** |
| Introduction | 7 → 2 [2h] → **2** [2h] | 6 → 1 [1h] → **1** [1h] |
| Methods | 1 → 0 → **0** | 1 → 0 → **0** |
| Experiments | 6 → 1 [1h] → **1** [1h] | 6 → 1 [1h] → **1** [1h] |
| Results | 12 → 2 [2h] → **2** [2h] | 12 → 2 [2h] → **2** [2h] |
| Discussion | 0 → 0 → **0** | 0 → 0 → **0** |
| Statements | 0 → 0 → **0** | 0 → 0 → **0** |
| **total** | 38 → 5 → **5** [5h] | 37 → 4 → **4** [4h] |

#### T4 Numbers in prose
| section | DRAFT-v2 before → after → **now** | DRAFT-v2-style before → after → **now** |
|---|---|---|
| Abstract | 0 → 0 → **0** | 0 → 0 → **0** |
| Introduction | 1 → 1 → **1** | 1 → 1 → **1** |
| Methods | 2 → 2 → **2** | 2 → 2 → **2** |
| Experiments | 12 → 12 [12h] → **12** [12h] | 12 → 12 [12h] → **12** [12h] |
| Results | 22 → 21 [18h] → **21** [18h] | 22 → 21 [18h] → **21** [18h] |
| Discussion | 12 → 12 → **1** ◆ | 12 → 12 → **1** ◆ |
| Statements | 0 → 0 → **0** | 0 → 0 → **0** |
| **total** | 49 → 48 → **37** [30h] | 49 → 48 → **37** [30h] |

#### T5 Headings
| section | DRAFT-v2 before → after → **now** | DRAFT-v2-style before → after → **now** |
|---|---|---|
| Abstract | 0 → 0 → **0** | 0 → 0 → **0** |
| Introduction | 0 → 0 → **0** | 0 → 0 → **0** |
| Methods | 2 → 0 → **0** | 2 → 0 → **0** |
| Experiments | 2 → 0 → **0** | 2 → 0 → **0** |
| Results | 11 → 0 → **0** | 11 → 0 → **0** |
| Discussion | 2 → 0 → **0** | 2 → 0 → **0** |
| Statements | 0 → 0 → **0** | 0 → 0 → **0** |
| **total** | 17 → 0 → **0** [0h] | 17 → 0 → **0** [0h] |

#### T6 Coinages
| section | DRAFT-v2 before → after → **now** | DRAFT-v2-style before → after → **now** |
|---|---|---|
| Abstract | 2 → 1 → **1** | 2 → 1 → **1** |
| Introduction | 4 → 1 [1h] → **1** [1h] | 4 → 1 [1h] → **1** [1h] |
| Methods | 0 → 0 → **0** | 0 → 0 → **0** |
| Experiments | 0 → 0 → **0** | 0 → 0 → **0** |
| Results | 5 → 3 → **3** | 5 → 3 → **3** |
| Discussion | 1 → 2 [2h] → **2** [2h] | 5 → 2 [2h] → **2** [2h] |
| Statements | 0 → 0 → **0** | 0 → 0 → **0** |
| **total** | 12 → 7 → **7** [3h] | 16 → 7 → **7** [3h] |

#### T7 Rhythm
| section | DRAFT-v2 before → after → **now** | DRAFT-v2-style before → after → **now** |
|---|---|---|
| Abstract | 1 → 0 → **0** | 1 → 0 → **0** |
| Introduction | 9 → 1 [1h] → **1** [1h] | 10 → 2 [1h] → **2** [1h] |
| Methods | 6 → 3 [1h] → **3** [1h] | 7 → 3 [1h] → **3** [1h] |
| Experiments | 6 → 2 → **2** | 7 → 3 [1h] → **3** [1h] |
| Results | 19 → 9 [1h] → **9** [1h] | 17 → 10 [1h] → **10** [1h] |
| Discussion | 12 → 11 [2h] → **11** [2h] | 13 → 11 [2h] → **11** [2h] |
| Statements | 3 → 3 → **3** | 3 → 3 → **3** |
| **total** | 97 → 51 → **52** [5h] | 98 → 53 → **54** [6h] |


#### Sections where "now" differs from pw2-draft-tells' "after"

| tell | section | draft | after | now |
|---|---|---|---|---|
| T2 | App R: Placeholder index | v2 | 1 | 0 |
| T2 | App R: Placeholder index | style | 1 | 0 |
| T4 | Discussion | v2 | 12 | 1 |
| T4 | Discussion | style | 12 | 1 |
| T7 | App R: Placeholder index | v2 | 0 | 1 |
| T7 | App R: Placeholder index | style | 0 | 1 |


Reading the ◆ rows: Discussion T4 12 → 1 is item 5 on the limitations list (pw2-draft-leftovers part 7; the one hit left is "256 requests", an assertion size). App R T2 1 → 0 and T7 0 → 1: pw2-draft-leftovers cut two sentences from App R (Placeholder index) after the placeholders were removed; the contrast went with them, and the paragraph lost its only short sentence. Positive controls: `pw2_draft_tells/controls.py` on the live v2 draft, ALL CONTROLS PASS (every detector CAUGHT, the brief's seven examples MATCHED).

## 8. Head-held paragraphs: byte-equal to the latest HEAD-HELD.md

Latest list: `outputs/pw2-draft-leftovers/HEAD-HELD.md` (= pw2-draft-tells' list, unchanged by item 4). Two independent checks per draft: `pw2_draft_sections/headheld.py` (sha256 per paragraph) and a raw-line byte comparison of the lines HEAD-HELD.md names in the draft it was written from (pw2-draft-tells' install = pw2-draft-leftovers' `snap-start/`) against the lines headheld.py locates now (`scans.py headheld`).

### DRAFT-v2

| paragraph | HEAD-HELD.md lines (in the draft it was written from) | lines now | raw lines byte-equal | sha256 (HEAD-HELD.md) = sha256 (now) |
|---|---|---|---|---|
| abstract, opening sentence | 88–90 | 88–90 | YES | YES |
| abstract, maze `\TODO` (whole lines) | 126–128 | 126–128 | YES | YES |
| Introduction, first paragraph | 142–148 | 142–148 | YES | YES |
| Methods, first paragraph | 274–283 | 275–284 | YES | YES |
| Experiments, first paragraph (preceded by the float chunk at lines 369–403) | 408–423 | 409–424 | YES | YES |
| Results, first paragraph | 508–529 | 509–530 | YES | YES |
| Discussion, first paragraph | 1172–1172 | 1172–1172 | YES | YES |

HEAD-HELD RAW CHECK: PASS (7 of 7)


headheld.py: HEAD-HELD CHECK: PASS (all 7 equal)

### DRAFT-v2-style

| paragraph | HEAD-HELD.md lines (in the draft it was written from) | lines now | raw lines byte-equal | sha256 (HEAD-HELD.md) = sha256 (now) |
|---|---|---|---|---|
| abstract, opening sentence | 94–96 | 94–96 | YES | YES |
| abstract, maze `\TODO` (whole lines) | 130–132 | 130–132 | YES | YES |
| Introduction, first paragraph | 145–151 | 145–151 | YES | YES |
| Methods, first paragraph | 275–284 | 276–285 | YES | YES |
| Experiments, first paragraph (preceded by the float chunk at lines 367–401) | 406–421 | 407–422 | YES | YES |
| Results, first paragraph | 506–527 | 507–528 | YES | YES |
| Discussion, first paragraph | 1166–1166 | 1166–1166 | YES | YES |

HEAD-HELD RAW CHECK: PASS (7 of 7)


headheld.py: HEAD-HELD CHECK: PASS (all 7 equal)

## 9. Every remaining `\TODO` and `\PLACEHOLDER`, by argument

### DRAFT-v2

| macro | line | section | argument (whole, whitespace collapsed) |
|---|---|---|---|
| `\TODO` | 126 | Abstract | CMAZE-T sentence reserved for the Head: the six-seed symmetric-input condition table (\S\ref{sec:maze-wm}, Table~\ref{tab:cmazet-sym}) replaces the earlier asymmetric-input sentence; no number is written here. |
| `\TODO` | 1307 | Statements: AI use statement | AI use statement --- the authors to confirm which of the venue's required disclosure categories apply. This draft, the evidence pipeline and most of the code were produced with substantial LLM-agent assistance (implementing methods, designing and critiquing experiments, interpreting results, drafting the paper); the statement must say so accurately. |
| `\TODO` | 1329 | Statements: Reproducibility statement | anonymised code archive not yet built. |

\TODO uses: 3; \PLACEHOLDER uses: 0; comment lines mentioning TODO/PLACEHOLDER (not uses): 6
rendered: "[TODO:" 3; "[PLACEHOLDER:" 0


### DRAFT-v2-style

| macro | line | section | argument (whole, whitespace collapsed) |
|---|---|---|---|
| `\TODO` | 130 | Abstract | CMAZE-T sentence reserved for the Head: the six-seed symmetric-input condition table (\S\ref{sec:maze-wm}, Table~\ref{tab:cmazet-sym}) replaces the earlier asymmetric-input sentence; no number is written here. |
| `\TODO` | 1295 | Statements: AI use statement | AI use statement --- the authors to confirm which of the venue's required disclosure categories apply. This draft, the evidence pipeline and most of the code were produced with substantial LLM-agent assistance (implementing methods, designing and critiquing experiments, interpreting results, drafting the paper); the statement must say so accurately. |
| `\TODO` | 1317 | Statements: Reproducibility statement | anonymised code archive not yet built. |

\TODO uses: 3; \PLACEHOLDER uses: 0; comment lines mentioning TODO/PLACEHOLDER (not uses): 7
rendered: "[TODO:" 3; "[PLACEHOLDER:" 0


`\TODO` 3 and `\PLACEHOLDER` 0 in each draft, with byte-identical arguments between the drafts: the abstract's maze sentence (Head-held), the AI use statement (the Head's) and the anonymised code archive. The preamble keeps both macro definitions and their convention comments (pw2-draft-leftovers' Head note 3); App. R "Placeholder index" now indexes no placeholder (its Head note 2).

## 10. Citation-scope audit of `jawahar2026chlu`

Every sentence that carries the key (code, comments stripped), its topics and verdict (`scans.py cite`). Allowed: the unit, its dynamics (Hamiltonian, symplectic, settling), the relativistic governor. Disallowed in a citing sentence: memory, deletion/forgetting/unlearning, test-time compute, retention, GMOR, dissipation. Exit 1 on VIOLATION; planted control (the key cited for "the retention law") → VIOLATION.

### DRAFT-v2

| line | section | citing sentence | topics (allowed) | disallowed terms in the sentence | verdict |
|---|---|---|---|---|---|
| 276 | Methods | \citet{jawahar2026chlu} introduced the CHLU as a latent dynamics model (defined here as an algorithm that either enforces or extracts the dynamics of the latent space itself to either design or approximately understand semantic rules created by the encoder that produced this abstract space such that downstream tasks can be appropriately dealt with or downstream design choices can be better informed). | unit, dynamics | — | **OK: cited for the unit / dynamics** |
| 280 | Methods | \citet{jawahar2026chlu} envision their unit (as the name suggests) as a step that provides a clue towards developing better latent dynamics models. | unit, dynamics | — | **OK: cited for the unit / dynamics** |
| 1236 | Discussion | The n-CLU we evaluate is the zero-dynamics limit of the CLU~\citep{jawahar2026chlu}: the same content, wells and read, with the settling loop's step count set to zero. | unit, dynamics | — | **OK: cited for the unit / dynamics** |
| 281 | Methods | We build on top of this idea and drop the complex physics design to construct our naive-CLU as a special case of the CHLU and test it as a factorized memory unit for compositional generation with additional properties such as deletion as a virtue of the design itself. | — | delet, memory | **no citation in this sentence; memory unit / deletion attributed to "we" (this paper), not to the cited work; the citation sits in the two preceding sentences (the unit), this sentence carries none** |

citations of `jawahar2026chlu` in code: 3 (in 3 citing sentences); VIOLATION: 0


### DRAFT-v2-style

| line | section | citing sentence | topics (allowed) | disallowed terms in the sentence | verdict |
|---|---|---|---|---|---|
| 277 | Methods | \citet{jawahar2026chlu} introduced the CHLU as a latent dynamics model (defined here as an algorithm that either enforces or extracts the dynamics of the latent space itself to either design or approximately understand semantic rules created by the encoder that produced this abstract space such that downstream tasks can be appropriately dealt with or downstream design choices can be better informed). | unit, dynamics | — | **OK: cited for the unit / dynamics** |
| 281 | Methods | \citet{jawahar2026chlu} envision their unit (as the name suggests) as a step that provides a clue towards developing better latent dynamics models. | unit, dynamics | — | **OK: cited for the unit / dynamics** |
| 1230 | Discussion | The n-CLU we evaluate is the zero-dynamics limit of the CLU~\citep{jawahar2026chlu}: the same content, wells and read, with the settling loop's step count set to zero. | unit, dynamics | — | **OK: cited for the unit / dynamics** |
| 282 | Methods | We build on top of this idea and drop the complex physics design to construct our naive-CLU as a special case of the CHLU and test it as a factorized memory unit for compositional generation with additional properties such as deletion as a virtue of the design itself. | — | delet, memory | **no citation in this sentence; memory unit / deletion attributed to "we" (this paper), not to the cited work; the citation sits in the two preceding sentences (the unit), this sentence carries none** |

citations of `jawahar2026chlu` in code: 3 (in 3 citing sentences); VIOLATION: 0


Three citations per draft, all for the unit and its dynamics; **0 VIOLATION**. The Methods sentence that names "a factorized memory unit" and "deletion" carries no citation and attributes both to "we". The bibliography entry renders as "Chlu: The causal hamiltonian learning unit …" (lower-cased by the .bst, reported by pw2-draft-methods; the bib is the Head's).

## 11. Bibliography

| file | sha256 | 56 entries |
|---|---|---|
| `live/v2/refs_clu_iclr.bib` | `b2cf715f25e020acf02038d5df4a2fd0ce816fb59e0aa3cfb101335684a71bd3` = b2cf715f… **unchanged** | 56 |
| `live/style/refs_clu_iclr.bib` | `b2cf715f25e020acf02038d5df4a2fd0ce816fb59e0aa3cfb101335684a71bd3` = b2cf715f… **unchanged** | 56 |


## Positive controls (every scan; planted copies under `controls/`, never the live drafts)

```
DRAFT-v2 | ucit=1 uref=1 errors=0 multiply_defined=0 overfull=0 underfull=16 rerun=0 bibitems=56 pages=42 | EXITS pdflatex1=0 bibtex=0 later_pdflatex=[ 0 0 0 ] passes=4
  - rendered `??` (undefined reference): 1; `(?)` or `[?]` (undefined citation): 1
CONTROL compile: planted undefined ref + cite -> ucit/uref > 0 and ?? rendered: CAUGHT
CONTROL style diff-check: one number changed in the style twin -> exit 1: CAUGHT
CONTROL provenance: planted unsourced 123.456 -> REGRESSED, exit 1: CAUGHT
CONTROL numbers: planted 7.486 in editable prose -> UNCLASSIFIED, exit 1: CAUGHT
CONTROL rounded: 7.034 claimed to round to 8 -> FAIL, exit 1: CAUGHT
CONTROL Head-held: one added space in the Results first paragraph -> headheld.py FAIL and raw check FAIL: CAUGHT
CONTROL TODO/PLACEHOLDER: planted \PLACEHOLDER -> listed, count 1: CAUGHT
CONTROL citation scope: jawahar2026chlu cited for the retention law -> VIOLATION, exit 1: CAUGHT
CONTROL tells: planted tells raise all seven detectors; the brief's seven examples match: CAUGHT
CONTROL defects: planted absolute path + doubled word in the source -> both listed: CAUGHT
CONTROL bib: one appended byte -> sha256 differs from b2cf715f...: CAUGHT
```
