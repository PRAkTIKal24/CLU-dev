# FINAL-INDEX-template — the drafts after the ICLR 2027 template pass (pw1-draft-template)

Before = the drafts as installed by pw1-draft-final (its `final-125012/`; sources cmp-equal to this pass's snapshot, labels equal). After = this pass's installed drafts (`final-130917/`). Same tooling (`pw1_draft_final/run_final.sh --inplace`). A page number that moved is printed **before → after**.

## 1. Compile (after)

```
# COMPILE-scratch.txt
DRAFT-v2 | ucit=0 uref=0 errors=0 overfull=0 underfull=14 rerun=0 bibitems=55 bibtexwarn=0 pages=42 | EXITS pdflatex1=0 bibtex=0 later_pdflatex=[ 0 0 0 ] passes=4
DRAFT-v2-style | ucit=0 uref=0 errors=0 overfull=0 underfull=13 rerun=0 bibitems=55 bibtexwarn=0 pages=43 | EXITS pdflatex1=0 bibtex=0 later_pdflatex=[ 0 0 0 ] passes=4
# COMPILE-inplace.txt
DRAFT-v2 | ucit=0 uref=0 errors=0 overfull=0 underfull=14 rerun=0 bibitems=55 bibtexwarn=0 pages=42 | EXITS pdflatex1=0 bibtex=0 later_pdflatex=[ 0 ] passes=2
DRAFT-v2-style | ucit=0 uref=0 errors=0 overfull=0 underfull=13 rerun=0 bibitems=55 bibtexwarn=0 pages=43 | EXITS pdflatex1=0 bibtex=0 later_pdflatex=[ 0 ] passes=2
# JOB1-compare.txt
DRAFT-v2.aux in-place == scratch
DRAFT-v2.bbl in-place == scratch
DRAFT-v2.out in-place == scratch
DRAFT-v2 PDF text in-place == scratch
DRAFT-v2 overfull boxes: 0
DRAFT-v2-style.aux in-place == scratch
DRAFT-v2-style.bbl in-place == scratch
DRAFT-v2-style.out in-place == scratch
DRAFT-v2-style PDF text in-place == scratch
DRAFT-v2-style overfull boxes: 0
```

## 2a. Figures (number, label, caption lead, page; both drafts)

| DRAFT-v2 no. | label | caption, first line (bold lead) | DRAFT-v2 page | style no. | style page |
|---|---|---|---|---|---|
| 1 | `fig:task-shapes` | The Shapes3D task. | 4 | 1 | 4 |
| 2 | `fig:task-cmazet` | The CMAZE-T task. | 5 | 2 | 5 |
| 3 | `fig:shapes-per-model` | Each model succeeding and failing on held-out combinations (Shapes3D, seed 0, K* =32). | 8 | 3 | 9 |
| 4 | `fig:shapes-baseline-wins` | Held-out requests on which a baseline succeeds and the n-CLU fails (seed 0). | 9 | 4 | 10 |
| 5 | `fig:ladder` | The ladder at K* against data-matched baselines, 3 seeds. | 11 | 5 | 12 |
| 6 | `fig:crossover` | The K-curve, validation, 3 seeds at every K. | 27 | 6 | 27 |
| 7 | `fig:deletion` | Deletion on the single design at K* , rows 3-4, test, fit pool 131,072 rows, seeds 0, 1 and 2. | 29 | 7 | 29 |
| 8 | `fig:adapt` | Encoder adaptation margins. | 31 | 8 | 31 |
| 9 | `fig:maze-wm` | CMAZE-T: the n-CLU against the best matched baseline under four input choices, and six seeds without and with  | 36 | 9 | 36 |
| 10 | `fig:cmazet-families` | Held-out CMAZE-T transitions, one per rule family, with every model's next state (words plus observation, test | 38 | 10 | 38 |
| 11 | `fig:cmazet-per-arm` | Each CMAZE-T model succeeding and failing (words plus observation, test, seed 0). | 39 | 11 | 39 |
| 12 | `fig:cmazet-rel-per-arm` | Each CMAZE-T model succeeding and failing with relational coding (words plus observation, test). | 40 | 12 | 40 |
| 13 | `fig:cmazet-door-tint` | Door-then-tint transitions: the move opens a door with the key in the pocket and enters a tint of the same col | 41 | 13 | 41 |

## 2b. Tables (number, label, caption lead, page; both drafts)

| DRAFT-v2 no. | label | caption, first line (bold lead) | DRAFT-v2 page | style no. | style page |
|---|---|---|---|---|---|
| 1 | `tab:t7` | Composition at K* =32 against data-matched baselines, Shapes3D value hold-out, TEST, n=3 seeds (0, 1, 2). | 7 | 1 | 7 |
| 2 | `tab:ledger-kstar` | The footprint breakdown of the K* =32 object, per seed, never netted. | 9 | 2 | 8 |
| 3 | `tab:ladder` | The ladder at K* =32, n=3 seeds, distinct valid novel tuples per combination, per seed (0 / 1 / 2). | 10 | 3 | 11 |
| 4 | `tab:del` | Deletion on the single design, the four measurements, K* =32, rows 3-4, TEST, fit pool 131,072 rows, one row p | 12 | 4 | 13 |
| 5 | `tab:hole` | Refusal DISPLAY rows. A display is never a result. | 14 | 5 | 14 |
| 6 | `tab:gentable-kstar` | The generation table, K* single design, one declared evaluation run, generation-2 cache. | 25 | 6 | 25 |
| 7 | `tab:lace-poe` | LACE and PoE at their own settings, Shapes3D value hold-out, TEST, n=3 seeds (0, 1, 2). | 25 | 7 | 25 |
| 8 | `tab:k5` | Ladder at K=5, the sweep's endpoint, rebuilt n-CLU, 3 seeds. | 26 | 8 | 26 |
| 9 | `tab:kcurve` | The K-curve, validation, symmetric-triangle convention. | 26 | 9 | 26 |
| 10 | `tab:kcurve3` | The K-curve at n=3 seeds on every row, validation (a smoke, not a claim). | 27 | 10 | 27 |
| 11 | `tab:impl` | Deletion on Shapes3D, matched on the n-CLU, implementations 1 and 2, bands not shared. | 28 | 11 | 28 |
| 12 | `tab:ablations` | The two ablations, earlier composition cell, TEST, n=3 seeds, one pin (524,288-byte n-CLU), K=5. | 28 | 12 | 29 |
| 13 | `tab:rivals-del` | Deletion cost for the data-matched baselines: retrain without the rows, on the n-CLU's 131,072-row pool, at th | 29 | 13 | 30 |
| 14 | `tab:rivals-del-seeds` | The n-CLU after deletion against each baseline retrained without the same rows, all 109 combinations, seeds 0  | 30 | 14 | 30 |
| 15 | `tab:adapt` | Encoder adaptation at K* , TEST, 3 seeds x 3 configurations (nine arm-runs, three seeds). | 30 | 15 | 31 |
| 16 | `tab:encloop` | Encoder loop, shortened decoder (400 joint steps, 8,192 images), 3 seeds, one declared test run. | 31 | 16 | 32 |
| 17 | `tab:bard-gpu` | The unfreezing decomposition on the longer leg only (4,000 joint steps, 32,768 training images, 1,800 decoder- | 31 | 17 | 32 |
| 18 | `tab:crr` | Declared criteria of the retention law on the field-mode n-CLU, 49 held-out combinations x 3 seeds, 192,975 ro | 32 | 18 | 33 |
| 19 | `tab:cmazet` | Maze world-model rows: CMAZE-T (transition generation), TEST, base n-CLU, 3 seeds (one declared test run per s | 37 | 19 | 37 |
| 19 | `tab:cmaze-del` | Maze world-model rows: CMAZE-T (transition generation), TEST, base n-CLU, 3 seeds (one declared test run per s | 37 | 19 | 37 |
| 20 | `tab:cmazet-sym` | CMAZE-T with symmetric inputs: every condition, unseen n-way, the engine's judge. | 42 | 20 | 42 |

Float numbers: identical before and after in both drafts (asserted). Float pages moved: 0. lof/lot cross-check after: DRAFT-v2 lo_without_label [], label_without_lo []; style [], [].

## 3. Pages per section (marker method C: the shipped page, inside the heading; lengths in pages)

| section | title | DRAFT-v2 start | DRAFT-v2 length | style start | style length |
|---|---|---|---|---|---|
| 1 | Introduction: a naive unit, and why its three properties matter | 2 | 2.74 | 2 | **2.81 → 2.87** |
| &nbsp;&nbsp;1.1 | Positioning: a naive unit, compared with naive units | 2 | **1.02 → 1.01** | 2 | 1.02 |
| &nbsp;&nbsp;1.2 | Why the three properties matter: world models need all three at once | 3 | **0.29 → 0.28** | 3 | **0.35 → 0.38** |
| &nbsp;&nbsp;1.3 | What we show, and at what thickness | 3 | **1.39 → 1.40** | 3 | **1.41 → 1.44** |
| 2 | The object, the judge and the memory footprint | 4 | 1.90 | 4 | **1.95 → 1.96** |
| 3 | The n-CLU on Shapes3D: the single design at $\Kstar$ | 6 | 8.69 | 6 | **8.72 → 8.74** |
| &nbsp;&nbsp;3.1 | Composition at $\Kstar$: rows 3 and 4, three scored seeds | 6 | 2.93 | 7 | 1.06 |
| &nbsp;&nbsp;3.2 | The $n$-way ladder at $\Kstar$, with its inference note | 9 | 2.69 | 8 | **3.20 → 3.21** |
| &nbsp;&nbsp;3.3 | Deletion on the single design: value, conjunction, item | 12 | 0.91 | 11 | **1.89 → 1.92** |
| &nbsp;&nbsp;3.4 | What the baselines can do instead: retrain without, or approximate | 13 | 0.33 | 13 | **0.32 → 0.55** |
| &nbsp;&nbsp;3.5 | Refusal after deletion: the hole, a separate mechanism | 13 | 0.62 | **13 → 14** | **0.67 → 0.42** |
| &nbsp;&nbsp;3.6 | Two ablations, each keeping one property | 14 | 0.45 | 14 | 0.43 |
| &nbsp;&nbsp;3.7 | The encoder: frozen by design, lightly adapted; the encoder loop licen | 14 | 0.47 | 15 | 0.49 |
| 4 | The same properties on a maze | 15 | 1.67 | 15 | **1.66 → 1.67** |
| 5 | Limitations of the naive unit, and a leaving note | 17 | **1.67 → 1.72** | 17 | **1.69 → 1.76** |
| &nbsp;&nbsp;5.1 | What the naive unit does not do (measured, not hedged) | 17 | **0.70 → 0.75** | 17 | **0.69 → 0.76** |
| &nbsp;&nbsp;5.2 | The clue: the n-CLU is the zero-dynamics limit of a settling unit | 17 | **0.27 → 0.29** | **17 → 18** | **0.29 → 0.30** |
| &nbsp;&nbsp;5.3 | The theory we carry: the Latent Dynamics Model class | 18 | **0.43 → 0.42** | 18 | **0.41 → 0.42** |
| &nbsp;&nbsp;5.4 | Leaving note | 18 | **0.24 → 0.23** | 18 | **0.23 → 0.20** |
| 6 | Follow-up: a full evaluation of the physics CLU | 18 | **0.23 → 0.22** | **18 → 19** | 0.23 |
| (stmt) | AI use statement | 18 | **0.10 → 0.12** | 19 | 0.12 |
| (stmt) | Ethics statement | 19 | 0.15 | 19 | 0.15 |
| (stmt) | Reproducibility statement | 19 | 0.19 | 19 | 0.21 |
| (refs) | References | 19 | **4.66 → 4.60** | 19 | **4.44 → 4.29** |
|  | Appendices | 24 | 0.13 | 24 | 0.13 |
| A | Reporting conventions | 24 | 0.42 | 24 | 0.42 |
| B | Tab.~\ref{tab:t7}: what it does and does not claim | 24 | 1.12 | 24 | 1.12 |
| C | The $K=5$ endpoint of the sweep | 25 | 0.58 | 25 | 0.58 |
| D | The $K$-curve: composition and the memory footprint against $K$ | 26 | 1.59 | 26 | 1.60 |
| E | Deletion on Shapes3D: two implementations, and the failure they agree  | 27 | 0.96 | 27 | 0.97 |
| F | The earlier composition cell: byte and parameter match, and the ceilin | 28 | 0.95 | 28 | 1.00 |
| G | Deletion comparators in full (fit pool, 0 test runs) | 29 | 0.66 | 29 | 0.64 |
| H | Light encoder adaptation in full | 30 | 0.44 | 30 | 0.20 |
| I | The encoder loop in full (shortened-decoder instrument, and the longer | 30 | 1.16 | 30 | 0.26 |
| J | The zero-dynamics identity: conditions and provenance | 32 | 0.73 | 30 | 1.92 |
| K | The retention law on the field-mode n-CLU in full | 32 | 0.85 | 32 | 0.89 |
| L | Prior derivations | 33 | 1.31 | 33 | **1.41 → 1.50** |
| M | What an evaluation must contain: the ten requirements | 34 | **0.46 → 0.48** | 35 | **0.41 → 0.43** |
| N | Maze world-model rows | 35 | **1.41 → 1.40** | 35 | **2.21 → 2.14** |
| O | The encoder: adaptation and the unfreezing decomposition | 36 | 2.11 | 37 | 1.96 |
| &nbsp;&nbsp; | Light encoder adaptation (secondary) | 36 | 0.94 | 37 | **0.92 → 0.95** |
| &nbsp;&nbsp; | Unfreezing with the n-CLU on the gradient path: the decomposition | 37 | 1.14 | 38 | **1.00 → 0.98** |
| P | The two laws on this study's n-CLU, stated in full | 38 | **1.85 → 1.86** | 39 | **1.86 → 1.87** |
| Q | Further limitations | 40 | **0.21 → 0.82** | 41 | **0.21 → 0.24** |
| R | Placeholder index | **40 → 41** | **0.86 → 0.29** | 41 | 0.78 |
| S | Declared not-runs | 41 | **0.88 → 0.91** | 42 | **0.37 → 0.41** |
| (end) | end of document | 42 |  | **42 → 43** |  |

## 4. Main text against the 9-page limit

| draft | main text, pages (title → "AI use statement") | ends on page | over 9 pp | statements, pages | references | appendices | total pages |
|---|---|---|---|---|---|---|---|
| DRAFT-v2 | **17.92 → 17.95** | 18 | **8.92 → 8.95** | **0.44 → 0.46** | pp. 19–23 | pp. 24–42 | 42 |
| DRAFT-v2-style | **18.09 → 18.25** | 19 | **9.09 → 9.25** | **0.49 → 0.48** | pp. 19–23 | **pp. 24–42 → pp. 24–43** | **42 → 43** |

## 5. `\TODO` / `\PLACEHOLDER` (by argument)

- DRAFT-v2: {'TODO': 4, 'PLACEHOLDER': 2} → {'TODO': 4, 'PLACEHOLDER': 2}; arguments, in order, BYTE-IDENTICAL; lines 128, 921, 1223, 1245, 2208, 2245
- DRAFT-v2-style: {'TODO': 4, 'PLACEHOLDER': 2} → {'TODO': 4, 'PLACEHOLDER': 2}; arguments, in order, BYTE-IDENTICAL; lines 131, 918, 1212, 1234, 2199, 2236

