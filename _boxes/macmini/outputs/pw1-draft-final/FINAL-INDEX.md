# FINAL-INDEX — PW-1 closing audit (pw1-draft-final)

Session `pw1-draft-final` · batmac · branch `pw1/draft-final` · the drafts as installed after this pass (the title line-break fix, §1.3). Everything below is read from the compiled `.aux`/`.log` of the live tree and from `final-125012/`.
Title (recorded for the report): **naive-CLU: Deletable Compositional Generation on a Frozen Encoder**.

## 1. Final compile (both drafts, scratch copy and in place)

| build | draft | exits (pdflatex 1, bibtex, later pdflatex) | pdflatex passes | undefined citations | undefined references | errors | overfull boxes | underfull (info) | bib items | pages |
|---|---|---|---|---|---|---|---|---|---|---|
| scratch (clean copy of the w17-paper tree) | DRAFT-v2 | 0, 0, 0 0 0 | 4 | 0 | 0 | 0 | 0 | 15 | 55 | 42 |
| scratch (clean copy of the w17-paper tree) | DRAFT-v2-style | 0, 0, 0 0 0 | 4 | 0 | 0 | 0 | 0 | 12 | 55 | 42 |
| in place (`~/clu/outputs/w17-paper/`) | DRAFT-v2 | 0, 0, 0 | 2 | 0 | 0 | 0 | 0 | 15 | 55 | 42 |
| in place (`~/clu/outputs/w17-paper/`) | DRAFT-v2-style | 0, 0, 0 | 2 | 0 | 0 | 0 | 0 | 12 | 55 | 42 |

- Scratch vs in place: DRAFT-v2.aux in-place == scratch; DRAFT-v2.bbl in-place == scratch; DRAFT-v2.out in-place == scratch; DRAFT-v2 PDF text in-place == scratch; DRAFT-v2 overfull boxes: 0; DRAFT-v2-style.aux in-place == scratch; DRAFT-v2-style.bbl in-place == scratch; DRAFT-v2-style.out in-place == scratch; DRAFT-v2-style PDF text in-place == scratch; DRAFT-v2-style overfull boxes: 0. **Page counts equal (42 = 42 in both drafts).** The PDF bytes differ only through the embedded date/ID (text, `.aux`, `.bbl`, `.out` byte-equal).
- Overfull boxes: **none** in either draft (both builds), so there is nothing to list.
- Style diff-check (`pw1_draft_sweeps/style_diff_check.py`, DRAFT-v2 vs DRAFT-v2-style): **ALL DIFFS EMPTY** (numbers, placeholders, TODOs, artefact paths, provenance comments in order, headings, labels, floats, preamble).
- Provenance DRAFT-v2: checked number tokens: 2597; unsourced lines: 7 ; not-found tokens: 9; raw not-found tokens: 9 ; derived-verified: 1 ; not-found after derivation check: 8. DERIVED-VERIFIED 1009 106 = 317 - 211 (operands found in w23-seed5-diagnosis/FINDINGS.md; w23-seed5-diagnosis/Q1-CONTENT.json) (identical to the snapshot before this pass).
- Provenance DRAFT-v2-style: checked number tokens: 2597; unsourced lines: 7 ; not-found tokens: 10; raw not-found tokens: 10 ; derived-verified: 1 ; not-found after derivation check: 9. DERIVED-VERIFIED 1004 106 = 317 - 211 (operands found in w23-seed5-diagnosis/FINDINGS.md; w23-seed5-diagnosis/Q1-CONTENT.json) (identical to the snapshot before this pass).
- The one derived number (Hub note): **106 = 317 − 211**, raw provenance "not found" +1, verified by `pw1_draft_wording2/check_derived.py` (both operands found in the cited W23 files). Reported, not changed.

### 1.3 The one draft edit of this pass (mechanical defect fix)
The title rendered on three lines, `NAIVE-CLU: DELETABLE COMPOSITIONAL GENERA-` / `TION` / `ON A FROZEN ENCODER`: item 13 dropped "Exactly" and kept the forced break after "Generation"; the small-caps first line then overfills and LaTeX hyphenates. Fix (no word changed, identical in both drafts): `\title{naive-CLU: Deletable Compositional Generation\\ on a Frozen Encoder}` → `\title{naive-CLU: Deletable Compositional\\ Generation on a Frozen Encoder}` (DRAFT-v2 l.75, DRAFT-v2-style l.81). Staged, built from the w17-paper layout, checked (0/0 undefined, 0 overfull, 42 pp, style diff ALL EMPTY, provenance identical, `\PLACEHOLDER`/`\TODO` 2/4 unchanged, **0 labels moved**; the PDF text differs only in the two title lines), installed byte-identically into both drafts. Images: `title-DRAFT-v2.png` (before), `stage-title-*/title-after-*.png` (after).

## 2. Figure and table index (from the compiled `.aux`; every row cross-checked against the list of figures/tables: 0 disagreements)

### Figures

| DRAFT-v2 no. | label | caption, first line (bold lead) | page | section where the float is in the source | style no. | style page |
|---|---|---|---|---|---|---|
| 1 | `fig:task-shapes` | The Shapes3D task. | 4 | 2 The object, the judge and the memory footprint | 1 | 4 |
| 2 | `fig:task-cmazet` | The CMAZE-T task. | 5 | 2 The object, the judge and the memory footprint | 2 | 5 |
| 3 | `fig:shapes-per-model` | Each model succeeding and failing on held-out combinations (Shapes3D, seed 0, K* =32). | 8 | 3.1 Composition at K*: rows 3 and 4, three scored se | 3 | 9 |
| 4 | `fig:shapes-baseline-wins` | Held-out requests on which a baseline succeeds and the n-CLU fails (seed 0). | 9 | 3.1 Composition at K*: rows 3 and 4, three scored se | 4 | 10 |
| 5 | `fig:ladder` | The ladder at K* against data-matched baselines, 3 seeds. | 11 | 3.2 The n-way ladder at K*, with its inference note | 5 | 12 |
| 6 | `fig:crossover` | The K-curve, validation, 3 seeds at every K. | 27 | D The K-curve: composition and the memory footprin | 6 | 27 |
| 7 | `fig:deletion` | Deletion on the single design at K* , rows 3-4, test, fit pool 131,072 rows, seeds 0, 1 and 2. | 29 | F The earlier composition cell: byte and parameter | 7 | 29 |
| 8 | `fig:adapt` | Encoder adaptation margins. | 31 | H Light encoder adaptation in full | 8 | 31 |
| 9 | `fig:maze-wm` | CMAZE-T: the n-CLU against the best matched baseline under four input choices, and six seeds without and with relational coding. | 36 | N Maze world-model rows | 9 | 36 |
| 10 | `fig:cmazet-families` | Held-out CMAZE-T transitions, one per rule family, with every model's next state (words plus observation, test). | 38 | N Maze world-model rows | 10 | 38 |
| 11 | `fig:cmazet-per-arm` | Each CMAZE-T model succeeding and failing (words plus observation, test, seed 0). | 39 | N Maze world-model rows | 11 | 39 |
| 12 | `fig:cmazet-rel-per-arm` | Each CMAZE-T model succeeding and failing with relational coding (words plus observation, test). | 40 | N Maze world-model rows | 12 | 40 |
| 13 | `fig:cmazet-door-tint` | Door-then-tint transitions: the move opens a door with the key in the pocket and enters a tint of the same colour (words plus observation, test). | 41 | N Maze world-model rows | 13 | 41 |

### Tables

| DRAFT-v2 no. | label | caption, first line (bold lead) | page | section where the float is in the source | style no. | style page |
|---|---|---|---|---|---|---|
| 1 | `tab:t7` | Composition at K* =32 against data-matched baselines, Shapes3D value hold-out, TEST, n=3 seeds (0, 1, 2). | 7 | 3.1 Composition at K*: rows 3 and 4, three scored se | 1 | 7 |
| 2 | `tab:ledger-kstar` | The footprint breakdown of the K* =32 object, per seed, never netted. | 9 | 3.1 Composition at K*: rows 3 and 4, three scored se | 2 | 8 |
| 3 | `tab:ladder` | The ladder at K* =32, n=3 seeds, distinct valid novel tuples per combination, per seed (0 / 1 / 2). | 10 | 3.2 The n-way ladder at K*, with its inference note | 3 | 11 |
| 4 | `tab:del` | Deletion on the single design, the four measurements, K* =32, rows 3-4, TEST, fit pool 131,072 rows, one row per seed (n=1 each). | 12 | 3.3 Deletion on the single design: value, conjunctio | 4 | 13 |
| 5 | `tab:hole` | Refusal DISPLAY rows. A display is never a result. | 14 | 3.5 Refusal after deletion: the hole, a separate mec | 5 | 14 |
| 6 | `tab:gentable-kstar` | The generation table, K* single design, one declared evaluation run, generation-2 cache. | 25 | B Tab. [tab:t7]: what it does and does not claim | 6 | 25 |
| 7 | `tab:lace-poe` | LACE and PoE at their own settings, Shapes3D value hold-out, TEST, n=3 seeds (0, 1, 2). | 25 | B Tab. [tab:t7]: what it does and does not claim | 7 | 25 |
| 8 | `tab:k5` | Ladder at K=5, the sweep's endpoint, rebuilt n-CLU, 3 seeds. | 26 | C The K=5 endpoint of the sweep | 8 | 26 |
| 9 | `tab:kcurve` | The K-curve, validation, symmetric-triangle convention. | 26 | D The K-curve: composition and the memory footprin | 9 | 26 |
| 10 | `tab:kcurve3` | The K-curve at n=3 seeds on every row, validation (a smoke, not a claim). | 27 | D The K-curve: composition and the memory footprin | 10 | 27 |
| 11 | `tab:impl` | Deletion on Shapes3D, matched on the n-CLU, implementations 1 and 2, bands not shared. | 28 | E Deletion on Shapes3D: two implementations, and t | 11 | 28 |
| 12 | `tab:ablations` | The two ablations, earlier composition cell, TEST, n=3 seeds, one pin (524,288-byte n-CLU), K=5. | 28 | F The earlier composition cell: byte and parameter | 12 | 29 |
| 13 | `tab:rivals-del` | Deletion cost for the data-matched baselines: retrain without the rows, on the n-CLU's 131,072-row pool, at the selected schedule, seeds 0 / 1 / 2. | 29 | G Deletion comparators in full (fit pool, 0 test r | 13 | 30 |
| 14 | `tab:rivals-del-seeds` | The n-CLU after deletion against each baseline retrained without the same rows, all 109 combinations, seeds 0 / 1 / 2. | 30 | G Deletion comparators in full (fit pool, 0 test r | 14 | 30 |
| 15 | `tab:adapt` | Encoder adaptation at K* , TEST, 3 seeds x 3 configurations (nine arm-runs, three seeds). | 30 | H Light encoder adaptation in full | 15 | 31 |
| 16 | `tab:encloop` | Encoder loop, shortened decoder (400 joint steps, 8,192 images), 3 seeds, one declared test run. | 31 | I The encoder loop in full (shortened-decoder inst | 16 | 32 |
| 17 | `tab:bard-gpu` | The unfreezing decomposition on the longer leg only (4,000 joint steps, 32,768 training images, 1,800 decoder-refit steps, 4,000 warm steps), TEST, 3 seeds, one declared test run. | 31 | I The encoder loop in full (shortened-decoder inst | 17 | 32 |
| 18 | `tab:crr` | Declared criteria of the retention law on the field-mode n-CLU, 49 held-out combinations x 3 seeds, 192,975 rows, n-CLU pinned at 524,288 B. | 32 | K The retention law on the field-mode n-CLU in ful | 18 | 33 |
| 19 | `tab:cmaze-del` | Maze world-model rows: CMAZE-T (transition generation), TEST, base n-CLU, 3 seeds (one declared test run per seed). | 37 | N Maze world-model rows | 19 | 37 |
| 19 | `tab:cmazet` | Maze world-model rows: CMAZE-T (transition generation), TEST, base n-CLU, 3 seeds (one declared test run per seed). | 37 | N Maze world-model rows | 19 | 37 |
| 20 | `tab:cmazet-sym` | CMAZE-T with symmetric inputs: every condition, unseen n-way, the engine's judge. | 42 | N Maze world-model rows | 20 | 42 |

`tab:cmaze-del` is a second `\label` on Tab. 19 (`tab:cmazet`, DRAFT-v2 l.2162); no `\ref` uses it. Figure and table numbers are identical in both drafts; pages differ by at most one (the style draft's Sec. 3 sets differently).

## 3. Page counts per section (markers on a scratch copy)

Method: before every heading `\typeout{PW1MARK A|id|\thepage}` (A, as briefed) and, inside the heading text, `\pdfsavepos` + a delayed `\write` of the SHIPPED page and the y position (C). The marker builds move no label (every `\newlabel` equal to the unmarked build) and C equals the `.aux` toc page at every numbered heading (0 mismatches). A lags C at 17 of 47 markers (DRAFT-v2) and 14 of 47 (style), because `\thepage` is read before the page builder ships the page; **C is the page count used**. Length = difference of text-block coordinates, (page − 1) + (y_top − y)/\textheight, to the next heading of the same or higher level, in pages; it includes any floats that TeX placed in between.

| section | title | DRAFT-v2 start page (C) | `\typeout` page (A) | length, pages | style start (C) | style A | style length |
|---|---|---|---|---|---|---|---|
| 1 | Introduction: a naive unit, and why its three properties matter | 2 | 1 | 2.74 | 2 | 1 | 2.81 |
| &nbsp;&nbsp;1.1 | Positioning: a naive unit, compared with naive units | 2 | 1 | 1.02 | 2 | 1 | 1.02 |
| &nbsp;&nbsp;1.2 | Why the three properties matter: world models need all three at once | 3 | 2 | 0.29 | 3 | 2 | 0.35 |
| &nbsp;&nbsp;1.3 | What we show, and at what thickness | 3 | 3 | 1.39 | 3 | 3 | 1.41 |
| 2 | The object, the judge and the memory footprint | 4 | 3 | 1.90 | 4 | 3 | 1.95 |
| 3 | The n-CLU on Shapes3D: the single design at $\Kstar$ | 6 | 6 | 8.69 | 6 | 6 | 8.72 |
| &nbsp;&nbsp;3.1 | Composition at $\Kstar$: rows 3 and 4, three scored seeds | 6 | 6 | 2.93 | 7 | 7 | 1.06 |
| &nbsp;&nbsp;3.2 | The $n$-way ladder at $\Kstar$, with its inference note | 9 | 9 | 2.69 | 8 | 8 | 3.20 |
| &nbsp;&nbsp;3.3 | Deletion on the single design: value, conjunction, item | 12 | 11 | 0.91 | 11 | 11 | 1.89 |
| &nbsp;&nbsp;3.4 | What the baselines can do instead: retrain without, or approximate | 13 | 13 | 0.33 | 13 | 12 | 0.32 |
| &nbsp;&nbsp;3.5 | Refusal after deletion: the hole, a separate mechanism | 13 | 13 | 0.62 | 13 | 13 | 0.67 |
| &nbsp;&nbsp;3.6 | Two ablations, each keeping one property | 14 | 14 | 0.45 | 14 | 14 | 0.43 |
| &nbsp;&nbsp;3.7 | The encoder: frozen by design, lightly adapted; the encoder loop licen | 14 | 14 | 0.47 | 15 | 14 | 0.49 |
| 4 | The same properties on a maze | 15 | 15 | 1.67 | 15 | 15 | 1.66 |
| 5 | Limitations of the naive unit, and a leaving note | 17 | 16 | 1.67 | 17 | 17 | 1.69 |
| &nbsp;&nbsp;5.1 | What the naive unit does not do (measured, not hedged) | 17 | 16 | 0.70 | 17 | 17 | 0.69 |
| &nbsp;&nbsp;5.2 | The clue: the n-CLU is the zero-dynamics limit of a settling unit | 17 | 17 | 0.27 | 17 | 17 | 0.29 |
| &nbsp;&nbsp;5.3 | The theory we carry: the Latent Dynamics Model class | 18 | 17 | 0.43 | 18 | 18 | 0.41 |
| &nbsp;&nbsp;5.4 | Leaving note | 18 | 18 | 0.24 | 18 | 18 | 0.23 |
| 6 | Follow-up: a full evaluation of the physics CLU | 18 | 18 | 0.23 | 18 | 18 | 0.23 |
| (stmt) | AI use statement | 18 | 18 | 0.10 | 19 | 19 | 0.12 |
| (stmt) | Ethics statement | 19 | 18 | 0.15 | 19 | 19 | 0.15 |
| (stmt) | Reproducibility statement | 19 | 19 | 0.19 | 19 | 19 | 0.21 |
| (refs) | References | 19 | 19 | 4.66 | 19 | 19 | 4.44 |
|  | Appendices | 24 | 24 | 0.13 | 24 | 24 | 0.13 |
| A | Reporting conventions | 24 | 24 | 0.42 | 24 | 24 | 0.42 |
| B | Tab.~\ref{tab:t7}: what it does and does not claim | 24 | 24 | 1.12 | 24 | 24 | 1.12 |
| C | The $K=5$ endpoint of the sweep | 25 | 24 | 0.58 | 25 | 24 | 0.58 |
| D | The $K$-curve: composition and the memory footprint against $K$ | 26 | 25 | 1.59 | 26 | 25 | 1.60 |
| E | Deletion on Shapes3D: two implementations, and the failure they agree  | 27 | 27 | 0.96 | 27 | 27 | 0.97 |
| F | The earlier composition cell: byte and parameter match, and the ceilin | 28 | 28 | 0.95 | 28 | 28 | 1.00 |
| G | Deletion comparators in full (fit pool, 0 test runs) | 29 | 29 | 0.66 | 29 | 29 | 0.64 |
| H | Light encoder adaptation in full | 30 | 29 | 0.44 | 30 | 29 | 0.20 |
| I | The encoder loop in full (shortened-decoder instrument, and the longer | 30 | 30 | 1.16 | 30 | 30 | 0.26 |
| J | The zero-dynamics identity: conditions and provenance | 32 | 31 | 0.73 | 30 | 30 | 1.92 |
| K | The retention law on the field-mode n-CLU in full | 32 | 32 | 0.85 | 32 | 32 | 0.89 |
| L | Prior derivations | 33 | 33 | 1.31 | 33 | 33 | 1.41 |
| M | What an evaluation must contain: the ten requirements | 34 | 34 | 0.46 | 35 | 35 | 0.41 |
| N | Maze world-model rows | 35 | 35 | 1.41 | 35 | 35 | 2.21 |
| O | The encoder: adaptation and the unfreezing decomposition | 36 | 35 | 2.11 | 37 | 36 | 1.96 |
| &nbsp;&nbsp; | Light encoder adaptation (secondary) | 36 | 35 | 0.94 | 37 | 36 | 0.92 |
| &nbsp;&nbsp; | Unfreezing with the n-CLU on the gradient path: the decomposition | 37 | 36 | 1.14 | 38 | 37 | 1.00 |
| P | The two laws on this study's n-CLU, stated in full | 38 | 38 | 1.85 | 39 | 39 | 1.86 |
| Q | Further limitations | 40 | 39 | 0.21 | 41 | 40 | 0.21 |
| R | Placeholder index | 40 | 40 | 0.86 | 41 | 41 | 0.78 |
| S | Declared not-runs | 41 | 41 | 0.88 | 42 | 41 | 0.37 |
| (end) | end of document | 42 | 42 |  | 42 | 42 |  |

- **DRAFT-v2:** main text (title to the first statement heading, "AI use statement") = **17.92 pages**, ending on p. 18; ICLR 2027 initial submission limit 9 pages (VENUE.md §1) → **8.92 pages over** (the main text is 199% of the limit). Statements (AI use, ethics, reproducibility; not counted by ICLR): 0.44 pp. References: pp. 19–23. Appendices: pp. 24–42 (19 pages). **Total pages: 42.**
- **DRAFT-v2-style:** main text (title to the first statement heading, "AI use statement") = **18.09 pages**, ending on p. 19; ICLR 2027 initial submission limit 9 pages (VENUE.md §1) → **9.09 pages over** (the main text is 201% of the limit). Statements (AI use, ethics, reproducibility; not counted by ICLR): 0.49 pp. References: pp. 19–23. Appendices: pp. 24–42 (19 pages). **Total pages: 42.**

## 4. Every remaining `\TODO` and `\PLACEHOLDER` (counted by argument, brace-balanced, comments stripped)

Positive control: two `\TODO{}`s planted on one line and one `\PLACEHOLDER{}` → arguments TODO 4 → 6, PLACEHOLDER 2 → 3, while line counts go 4 → 5 (`grep -c` would miss one).

| # | kind | DRAFT-v2 line | style line | section | in float | argument (full) |
|---|---|---|---|---|---|---|
| 1 | TODO | 128 | 131 | abstract | — | CMAZE-T sentence reserved for the Head: the six-seed symmetric-input condition table (\S\ref{sec:maze-wm}, Table~\ref{tab:cmazet-sym}) replaces the earlier asymmetric-input sentence; no number is written here. |
| 2 | TODO | 921 | 918 | 3.6 Two ablations, each keeping one property | — | no record prints the byte ratio of the composition-only n-CLU to the single design; the records give the two totals printed here and in Tab.~\ref{tab:ledger-kstar}, and they are not divided here. |
| 3 | TODO | 1223 | 1212 | (unnumbered) AI use statement | — | AI use statement --- the authors to confirm which of the venue's required disclosure categories apply. This draft, the evidence pipeline and most of the code were produced with substantial LLM-agent assistance (implementing methods, designing and critiquing experiments, interpreting results, drafting the paper); the statement must say so accurately. |
| 4 | TODO | 1245 | 1234 | (unnumbered) Reproducibility statement | — | anonymised code archive not yet built. |
| 5 | PLACEHOLDER | 2208 | 2199 | N Maze world-model rows | fig:cmazet-families | per-model $s_{t+1}$ for the observation-only input of Tab.~\ref{tab:cmazet} (n-CLU, Transformer, MLP, LSTM) with the engine's judge's marks, one success and one failure per model: not drawable, those models were not persisted; may remain undrawn, and if drawn may show the n-CLU failing where a baseline succeeds |
| 6 | PLACEHOLDER | 2245 | 2236 | N Maze world-model rows | fig:cmazet-door-tint | per-model $s_{t+1}$ on the door-then-tint transitions for the observation-only input of Tab.~\ref{tab:cmazet}, with the engine's judge's marks: not drawable, those models were not persisted; may remain undrawn, and if drawn may show every model, the n-CLU included, failing |

Counts: DRAFT-v2 {'TODO': 4, 'PLACEHOLDER': 2}; DRAFT-v2-style {'TODO': 4, 'PLACEHOLDER': 2} (arguments byte-identical and in the same order). Line counts: {'TODO': 4, 'PLACEHOLDER': 2} / {'TODO': 4, 'PLACEHOLDER': 2}.

## 5. Whole-wave residual audit of rendered text (items 2, 3, 5, 8, 13, 14, 15, 17 and the pendings)

Channels: **src** = the `.tex` without comments (`\label`/`\ref` keys blanked, except the removed-keys rule, which reads them); **pdf** = the in-place PDF text (pypdf, vector-figure text included). Positive control: one paragraph planting every token, compiled into scratch copies of both drafts: **every rule caught its token in src; in pdf every rule but two** — `\_` names (pypdf extracts the drawn underscore as a space, so `landscape_removed` reads as the allowed plain name) and a word hyphenated across a page break with a float between the halves ("expen-/sive", DRAFT-v2 only). The src channel covers both.

| rule | item | DRAFT-v2 src | DRAFT-v2 pdf | style src | style pdf | residuals: every hit, with its reason |
|---|---|---|---|---|---|---|
| rival | 2 | 0 | 0 | 0 | 0 | 0 |
| store | 2 | 0 | 0 | 0 | 0 | 0 |
| gate | 2 | 0 | 0 | 0 | 0 | 0 |
| artifact/artefact | 2 | 0 | 0 | 0 | 0 | 0 |
| pre-registered | 2 | 0 | 0 | 0 | 0 | 0 |
| A1 | 2 | 0 | 0 | 0 | 0 | 0 |
| A10 | 2 | 0 | 0 | 0 | 0 | 0 |
| C-512 | 2 | 0 | 0 | 0 | 0 | 0 |
| improved | 2 | 0 | 0 | 0 | 0 | 0 |
| config (abbrev.) | 2 | 0 | 1 | 0 | 1 | PDF only: pypdf splits the English word at a line-break hyphen ("config- urations"); the source has 0 abbreviations. |
| configuration (English word) | 2 | 16 | 16 | 16 | 16 | The English word ("configurations", table header "configurations"); the plan bans the abbreviation "config", which is 0. Kept as in pw1-draft-sweeps. |
| spoke | 2 | 0 | 0 | 0 | 0 | 0 |
| hub | 2 | 0 | 0 | 0 | 0 | 0 |
| wave | 2 | 0 | 0 | 0 | 0 | 0 |
| charter | 2 | 0 | 0 | 0 | 0 | 0 |
| ledger | 3 | 0 | 0 | 0 | 0 | 0 |
| bar k | 2 | 0 | 0 | 0 | 0 | 0 |
| scored run | 2 | 0 | 0 | 0 | 0 | 0 |
| budget | 2 | 0 | 0 | 0 | 0 | 0 |
| spend/spent | 2 | 0 | 2 | 0 | 2 | PDF only, inside the Fig. 2 image (fig_task_cmazet): the engine's event names "key spent at door" and "event spend". Item 6 rules Fig. 2 final with no changes. |
| price/priced/pricing | 2 | 0 | 0 | 0 | 0 | 0 |
| buy/buys/bought | 2 | 0 | 0 | 0 | 0 | 0 |
| pay/paid | 2 | 0 | 0 | 0 | 0 | 0 |
| cost (technical only) | 2 | 7 | 9 | 7 | 9 | Technical sense in all 7 source hits: seconds per seed (Tab. del "cost" row block, Tab. 13 "Deletion cost ... retrain"), bytes ("two holes cost 2 bytes", "the coupling hole costs one byte", "cost line"), "memory cost", "memory and time cost". The 2 extra PDF hits are the author name "Costa" in the bibliography. |
| cheap | 2 | 1 | 1 | 1 | 1 | "Deletion is not cheap in memory or time" (limitation 3): "not cheap" is allowed by item 2. |
| expensive | 2 | 0 | 0 | 0 | 0 | 0 |
| t_pay / price law / priced channel | 2 | 0 | 0 | 0 | 0 | 0 |
| corner | 3 | 0 | 3 | 0 | 3 | PDF only, inside the Fig. 1 image (fig_task_shapes): "shape in the corner", "pair corner (2-way)", "six-factor corner (n-way)". Item 3: "Fig. 1 itself: no changes." The source has 0. |
| held-out composition | 3 | 0 | 0 | 0 | 0 | 0 |
| oblong | 3 | 0 | 0 | 0 | 0 | 0 |
| exact judge | pending | 0 | 0 | 0 | 0 | 0 |
| exact / exactly / exactness (all senses) | 13 | 34 | 35 | 34 | 35 | No deletion sense (34 = 9 + 4 + 5 + 16): numeric equality (9: "exactly 0.0000" / "exactly 0.000" x3, "exactly constant" x2, "exactly 16,384 B", "exactly 0.0" x2, "exactly zero off the coupling graph"), the same rows (4: "exact 131,072-row pool", "exactly the n-CLU's content rows" x3), exact-match requests (5), mathematics/physics (16: zero-dynamics limit, returns z_req exactly at lambda=0, exactly delta/(1+beta), sqrt(1-gamma), exact log-derivative, exact inversion, exact root, bit-exactly (replication), ties exactly on two refusal rows, exact one-step Jacobian, exact only at zero damping, exact value fidelity (trilemma), 39% exactly optimal, exact-key lookup, exact quadratic, "were then exact"); the 34 of pw1-draft-wording's list. PDF: 33 of the 34 (pypdf drops the space in "exactly delta") + 2 cited titles in the bibliography ("Exact federated continual unlearning ...", "... Efficient and exact machine unlearning ..."). |
| 10x / 10x steps | 2 | 0 | 1 | 0 | 1 | PDF only, fig_ladder legend "seed at 10x steps": item 10.6 prescribes this entry verbatim ("seed at $10\times$ steps"). Item 1.8 says never "10x steps" for the maze figure. FOR THE HEAD: the two rulings differ; the figure follows item 10.6. |
| internal control names | 2 | 0 | 0 | 0 | 0 | 0 |
| Naive CLU / naive CLU | 2 | 0 | 0 | 0 | 0 | 0 |
| bare CLU | 2 | 3 | 8 | 3 | 7 | Source (3): the CLU class / the settling physics CLU (Sec. 5.2 "limit of the CLU", Sec. 6 title and first sentence), not the naive unit. PDF extra hits are pypdf splitting "n-CLU" at a line-break hyphen ("n- CLU"). |
| hard (any sense) | 8 | 1 | 1 | 1 | 1 | "Hard read isolation (designed memory only)", App. L (physics trilemma), a memory property; no seed is called hard. |
| easy (any sense) | 8 | 0 | 0 | 0 | 0 | 0 |
| difficult* | 8 | 0 | 0 | 0 | 0 | 0 |
| ceiling | 1.5/8 | 16 | 16 | 16 | 16 | No per-seed / best-of-arms maze ceiling (item 1.5): Shapes3D count ceilings 36 / 72 (Sec. 2, Tab. 1 caption, App. F header x2, App. I caption), "its 24-tuple ceiling" (Sec. 3.2), "a ceiling and not an arm" (item memory, Sec. 3.6), nearest-prototype ceiling (Sec. 3.7, App. B caption and Tab. lace-poe x2), "the ceiling ladder" (App. F title), "oracle read (ceiling)", "absolute content ceiling at the pin", "Two ceiling rungs" (App. F), and Tab. 19 (w16 asymmetric row) "Ceiling set by the request design, not the n-CLU" (a design plateau, flagged by pw1-draft-wording for PW-2). |
| best fitted arm | 1.5 | 0 | 0 | 0 | 0 | 0 |
| WIN / LOSE / COMPARABLE / LOSS labels | 1.3 | 4 | 3 | 4 | 3 | FOR THE HEAD: declared-criterion outcome words in small caps, not item 1.3's figure label column: Sec. 3.7 "(-0.1150, S3 LOSE)"; App. B, Tab. lace-poe caption "S3 ... LOSE; S4 ... WIN" and its last row "LOSE" (with PASS/FAIL beside them). Outside every PW-1 item; not a mechanical defect, so not edited. |
| CMAZE-1 / 2 / 2b / O | 14 | 0 | 0 | 0 | 0 | 0 |
| route | 14 | 7 | 7 | 7 | 7 | Six in the English sense of a deletion path (Sec. 1.1 x2, Sec. 3.4 x4: "only route to that match is the retrain", "Our route matches the refit", ...); one in the mandated CMAZE-suite paragraph ("route planning", item 14 wording). No route-task result remains. |
| verb | 14 | 0 | 0 | 0 | 0 | 0 |
| request layer | 14 | 0 | 0 | 0 | 0 | 0 |
| red ∧ sphere | 12.5 | 0 | 0 | 0 | 0 | 0 |
| Exactly Deletable (title) | 13 | 0 | 0 | 0 | 0 | 0 |
| § / \S | 5 | 1 | 1 | 1 | 1 | Inside the abstract's CMAZE-T \TODO (Head-held, byte-identical by ruling): "(\S\ref{sec:maze-wm}, Table~\ref{tab:cmazet-sym})". |
| Section~ / Section N | 5 | 0 | 0 | 0 | 0 | 0 |
| Figure~ / Figure N (not a caption label) | 5 | 0 | 0 | 0 | 0 | 0 |
| Table~ / Table N (not a caption label) | 5 | 1 | 1 | 1 | 1 | The same abstract \TODO as above ("Table 20" in the PDF). Every other "Table N" in the PDF is LaTeX's own caption label "Table N:". |
| Appendix~ / Appendix X | 5 | 0 | 0 | 0 | 0 | 0 |
| ref without ~ after Sec./Fig./Tab./App. (breaking space) | 5 | 0 | 0 | 0 | 0 | 0 |
| removed keys/files (items 14, 15, 17) | 14/15/17 | 0 | 0 | 0 | 0 | 0 |

Presence checks: title (source) = "naive-CLU: Deletable Compositional Generation on a Frozen Encoder" (both drafts: True / True); first mention "naive-CLU (n-CLU)" present: True / True; "the engine's judge" 12 / 12 times; "$10\times$ training steps" 13 / 13 times.
Every hit with 70 characters of context: `AUDIT-DRAFT-v2.json`, `AUDIT-DRAFT-v2-style.json`. Unclassified residuals: **none**.

## 6. Figure-folder audit (`w17-paper/figures/` and `paper-visuals/`)

| placed file | sha256 (16 hex), w17-paper/figures | paper-visuals | equal |
|---|---|---|---|
| `fig_task_shapes.pdf` | c83b24d551ec128e | c83b24d551ec128e | yes |
| `fig_task_shapes.png` | 78a8d7ed2995dd1b | 78a8d7ed2995dd1b | yes |
| `fig_task_cmazet.pdf` | f701f98fa9a9f0ab | f701f98fa9a9f0ab | yes |
| `fig_task_cmazet.png` | 10c9aa59c18611fd | 10c9aa59c18611fd | yes |
| `fig_ladder.pdf` | 949d5b6d1ff2fac3 | 949d5b6d1ff2fac3 | yes |
| `fig_ladder.png` | a5d05a35d53e5272 | a5d05a35d53e5272 | yes |
| `fig_crossover.pdf` | 0dfda2510acbce7f | 0dfda2510acbce7f | yes |
| `fig_crossover.png` | 22262422cea96dd8 | 22262422cea96dd8 | yes |
| `fig_deletion.pdf` | 8faeaf9fa28f05d8 | 8faeaf9fa28f05d8 | yes |
| `fig_deletion.png` | b444cfe683d2669a | b444cfe683d2669a | yes |
| `fig_encoder_adapt.pdf` | 3c0a924da28bf664 | 3c0a924da28bf664 | yes |
| `fig_encoder_adapt.png` | fe565909d28da32a | fe565909d28da32a | yes |
| `fig_maze_conditions.pdf` | 3896e8128efb912b | 3896e8128efb912b | yes |
| `fig_maze_conditions.png` | f4815379a04d9ade | f4815379a04d9ade | yes |
| `fig_cmazet_families.pdf` | b774ac729782480d | b774ac729782480d | yes |
| `fig_cmazet_families.png` | bc029153e19370f9 | bc029153e19370f9 | yes |
| `fig_cmazet_per_arm.pdf` | 744e643df043307e | 744e643df043307e | yes |
| `fig_cmazet_per_arm.png` | 000ed77447ccecbf | 000ed77447ccecbf | yes |
| `fig_cmazet_rel_per_arm.pdf` | 22454719c5f85e3a | 22454719c5f85e3a | yes |
| `fig_cmazet_rel_per_arm.png` | 6e33878c83139496 | 6e33878c83139496 | yes |
| `fig_cmazet_door_then_tint.pdf` | ea36760fe82cfc2d | ea36760fe82cfc2d | yes |
| `fig_cmazet_door_then_tint.png` | 8900a6dfff9bd4f4 | 8900a6dfff9bd4f4 | yes |
| `rendered/rc_pw1_shapes_per_model.pdf` | f9dfc17eacf0e6ad | f9dfc17eacf0e6ad | yes |
| `rendered/rc_pw1_shapes_per_model.png` | 3d1d7c3707ebec57 | 3d1d7c3707ebec57 | yes |
| `rendered/rc_pw1_shapes_baseline_wins.pdf` | 18661bc6d18ee5f5 | 18661bc6d18ee5f5 | yes |
| `rendered/rc_pw1_shapes_baseline_wins.png` | 11d82bc2ef1de8ac | 11d82bc2ef1de8ac | yes |

- placed stems per draft: 13 (identical lists in both drafts); files 26; present+equal in both folders: 26
- w17-paper/figures SHA256-w22-build1.txt: 22 entries, 0 problems
- w17-paper/figures SHA256-w22-build2.txt: 22 entries, 0 problems
- w17-paper/figures rendered/SHA256-pw1-render-build1.txt: 6 entries, 0 problems
- w17-paper/figures rendered/SHA256-pw1-render-build2.txt: 6 entries, 0 problems
- paper-visuals SHA256-w22-build1.txt: 22 entries, 0 problems
- paper-visuals SHA256-w22-build2.txt: 22 entries, 0 problems
- paper-visuals rendered/SHA256-pw1-render-build1.txt: 6 entries, 0 problems
- paper-visuals rendered/SHA256-pw1-render-build2.txt: 6 entries, 0 problems
- paper-visuals README: make_index exit 0; installed README == regenerated: True
- w17-paper-figures README: make_index exit 0; installed README == regenerated: True
- Positive control: in a copy of paper-visuals, one byte appended to `fig_ladder.png` and a line appended to README.md → the file is flagged unequal, both manifests flag it, and the README check reads False.
- Step 0 (Hub note): `bash build.sh` in place, default destinations: exit 0, both builds byte-identical; **26 / 26 placed files byte-unchanged in each folder**; the new `PROVENANCE.json` (`ec5a6122…`) is cmp-equal to the one pw1-visuals-tooling predicted, and its JSON diff against the old one is exactly the three edits; only `PROVENANCE.json`, `SHA256-w22-build{1,2}.txt` and (after make_index) `README.md` changed in each folder.
