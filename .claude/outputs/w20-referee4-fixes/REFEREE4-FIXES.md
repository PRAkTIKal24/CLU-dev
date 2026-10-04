# REFEREE4-FIXES: w20 `referee4-fixes`, answering REFEREE-v3 (w18 `referee-4`)

## Files

| file | before (snapshot `snap-191242/`) | after (installed in `w17-paper/`) |
|---|---|---|
| `DRAFT-v2.tex` | `e41968894a71f9f1…`, 2,804 lines | `e49a6910c6740207…`, 2,926 lines |
| `DRAFT-v2-style.tex` | `98e41695c9aea095…`, 2,793 lines | `fd0ff46d13475245…`, 2,915 lines |
| `references.bib` | `226ddd54790f3035…` | unchanged |

- Before the edits, both drafts were byte-identical to the referee's snapshot, so REFEREE-v3's L-numbers apply to the "before" column.
- Just before install, `cmp` confirmed that nobody had changed the drafts since my snapshot.
- Unified diffs: `DRAFT-v2.referee4.diff` and `DRAFT-v2-style.referee4.diff`.

**Method.** All edits were made by one script, `tools/apply.py`. It holds 56 edits, and each one:
- lists the old text for each file and aborts unless that text occurs exactly the expected number of times;
- writes identical new text into both drafts, so numbers, provenance comments, placeholders, floats and headings stay in lock-step.

Where the style draft uses `\emph` item heads and v2 uses `\textbf`, the head markup is kept per file.

**Line numbers below are in the AFTER files, given as v2 / style.**

---

## Must-fix items applied

### MF-1: CMAZE-T input asymmetry, disclosed

**Before**
- §4.4 said only "Rivals are matched on read floats".
- The abstract said "the store does compose held-out n-way transitions".
- Contribution 3 and the thickness note presented a clean store row.
- The planner columns and the family mix were in the appendix table only.

**After**

1. **New §4.4 paragraph, "CMAZE-T: the inputs are not symmetric"** (L1098 / L1094). It states:
   - the store reads five transition words through a `[bias | main | pair]` design whose pair block is hand-built interaction structure;
   - the learned blind arms read reduced forms of that request;
   - the matched rivals regress from a 636-dim observation through the frozen state encoder and get no pair block.

   Sources: `PREREG-w16-maze-wm-T1.md` (Config A and l.55), `w15-cmaze-t-harness/FINDINGS.md` §5.4, and w16 FINDINGS §2.6 ("the head fixes which held-out 4-tuples are readable").

   The paragraph also:
   - says the row shows the store is **load-bearing under this request**, and does **not** show that the store rather than the request design carries the margin;
   - cites Config B (the interface lifted the rival +0.40 and the store +0.20);
   - points to `tab:cmazet-sym` as **pending**.
2. **The planner columns are now in the main text:** 0.9071 / 0.5509 / 0.6047.
   - I did **not** argue them away.
   - The text says the planner reads here as it did on CMAZE-2 (0.9077): the rule order it ignores matters only on tinted-door rows, a minority of rows.
   - So the row tests held-out word combinations mostly on rows that do not need the door-then-tint interaction.
   - The learned blind arms, which must generalise, fail (0.3418 / 0.0716).
3. **Family mix printed, with "no per-family score exists":**

   | family | seed 0 | seed 1 | seed 2 |
   |---|---|---|---|
   | free | 345 | 328 | 309 |
   | rule_b | 316 | 307 | 288 |
   | interaction | 70 | 79 | 55 |
   | open_door_tint | 30 | 28 | 33 |
   | rule_a | 0 | 0 | 1 |
   | **total** | **761** | **742** | **686** |

   The referee's list left out `open_door_tint`; it is from `REQUESTS-w19.json`.
4. **Scoping added in every other place the row appears:**
   - §4.4 heading → "…under a pair-design request" (L1081);
   - the `tab:cmazet` caption states the asymmetry and points to `tab:cmazet-sym`;
   - §4.4 closing summary;
   - abstract (L123 / L124): "…but the store reads a word request with a hand-built pair block that the rivals, which read the observation, do not receive, so the row is scoped to that request until a symmetric-input test is run";
   - contribution 3;
   - thickness note (MF-5b);
   - limitation 1 (MF-5a).
5. **No w19 result is anticipated.** The text says only "pending" / "until a symmetric-input test is run".

### MF-2: the w20 placeholder's fit pools, corrected

**Before** (L503–511): "the store's own fit pool is 131,072 rows". This contradicted App D ("the same 32,768 fit rows as the store's content").

**After** (L510, placeholder text only; it is not filled and not deleted). Pools are given per table:

| table | store | rivals |
|---|---|---|
| Table 1 (`tab:t7`) | 131,072 rows of the value hold-out pool | 16,384 rows of that pool |
| ladder (`tab:ladder`) | content fitted on 32,768 rows of each rung's pool | data-matched MLP: the same 32,768 rows; other rivals: 32,256 rows of the combination-split pool, which carries the held-out values |

- **Pending list:** it describes w20 as briefed, stating plainly that its 131,072-row refit is **four times the ladder store's 32,768 content rows**. It does not predict the result. The "may land as a NEGATIVE" clause is kept verbatim.
- **Sources:**
  - `kstar_ladder/2_seed0_K32/LEG.json`: `n_fit_rows` 32768; `partitions.json`: `n_store_train` 32768, `n_pool` 131072;
  - the same cache and 288,000-row train split as the K = 5 ladder's `runs/2_seed0/LEG.json`.
- ⚠ **Not mine to edit, reported for the Head:** `tasks/w20/shapes-data-matched.md` l.9 still says "The store's pool is 131,072 rows". REFEREE-v3 flags that this reverses the asymmetry on the ladder.

### MF-3: the data-matched, read-path-matched rival, printed

**Before:** `grep ownpool` over the draft returned 0 hits; the 1× margin appeared in App D only.

**After**

1. **New §3.1 paragraph, "A rival matched on both the read path and the data"** (L523). It gives:
   - the MLP at the read-path pin, **130,514** parameters;
   - fitted on **the same first 32,768 rows** from which the store's content is fitted, on every rung and seed;
   - n = 2: **5.5667 / 7.5833 / 9.3000** against the store's 31.800 / 32.500 / 33.633;
   - n = 6: **10.6667 / 15.1167 / 21.0333** against 65.908 / 66.250 / 67.650;
   - on the 109 corners: **9.1835 / 11.1193 / 15.1284** against 47.4954 / 47.4037 / 49.1193.

   The paragraph states that **on the 109-corner leg the MLP is fitted on 32,768 rows of the value hold-out pool, NOT on the 131,072 rows** from which that store is fitted, so there it is "matched on the population but not on the row count".
2. **New row in `tab:ladder`:** "MLP, data-matched (store's own fit rows)", all five rungs × 3 seeds, from `w13 TABLES.md` l.703–710. The caption names its 130,514 parameters and 32,768 rows.
3. **The 1× margin is now in the main text** as the conservative comparison:
   - 536,270 to 536,972 parameters, the same 32,768 rows;
   - beaten on every rung and seed, smallest margin **+3.117**;
   - labelled a replication, not a scored row, with one family only.
4. **Abstract and contribution 1** name the fit pools (see MF-8).

- ⚠ **Printed honestly, against the paper's interest.** This MLP is **not** the strongest rival. At n = 2 it is about 13 tuples per corner below the best combination-pool rival. `AMENDMENT-3-CORRECTION.md` withdrew the opposite claim; its plausible reason is that the value hold-out pool lacks the held-out values. The paragraph says so.
- It therefore answers "naive vs naive is parameter-only" **for the MLP family** on data. It is not a data-matched LSTM or transformer; that is still w20's.
- `ownpool` now appears only in `% SRC` paths (3 hits per draft). The rendered text uses "data-matched MLP", keeping pipeline identifiers out of the PDF (SF-8).

### MF-4: N359, shortened-leg margins no longer argued from

**Before:** "success bar fails on both rows"; "the store's margin over a matched learner does not grow, so the store does not shape the representation" (§3.7, App T, limitation 4).

**After**
- **`tab:encloop` last row** relabelled "margin over the frozen-feature rival (not a measurement of the refitted learner)" (L1971). The numbers are kept, labelled.
- **App J (L1977):**
  - the store-only falsifier (net ≤ 0 on ≥ 2 seeds) stands;
  - the success bar is "not assessed on this leg… criterion D on this leg is unmeasured";
  - the 0.006 is attributed to the frozen-feature rival.
- **App T (L2784):** "not measured… margins over the frozen-feature rival… criterion D here is unmeasured, and on the longer leg… mixed with no claim either way".
- **§3.7 (L868):** "licenses no store claim… mixed on the longer leg… unmeasured on the shortened leg". It also drops "unfreezing hurts", which the longer leg does not reproduce.
- **§3.7 heading** → "…lightly adapted; the encoder loop licenses no store claim" (L862).
- **Limitation 4** → "The encoder loop licenses no store claim", keeping the store-only falsifier (L1190).

### MF-5: stale text contradicting landed rows

| item | before | after |
|---|---|---|
| (a) limitation 1 | "No store row on the maze" | "No store row on the route tasks (CMAZE-2, CMAZE-2b)", plus a CMAZE-T scope sentence (L1171) |
| (b) thickness | "The one environment with an exact route judge and three scored seeds returns a negative for the store" | Routes: negative for the store, positive for the interface. CMAZE-T, same engine, exact judge, 3 seeds: one store row, scoped to its request (L262 / L259) |
| (c) App J | "Its scoring stages were not run." | deleted; the SRC comment is kept |
| (d) §3.7 | "with the pending scored row of criterion D" | the MIXED verdict (see MF-4) |
| (e) §4.1 | "It is the one deletion leg in the study that sits inside scored runs" | "It was the first… (the deletion rows of the two world-model instruments of §4.4 also do)" (L920) |

### MF-6: bannered placeholder figures, counted

The figures were **not redrawn**: that is zero-compute but the figures owner's job, and it needs the plotting pipeline. They are now counted:

1. **Placeholder index (App W, L2885)**, new text: "Three figures are placeholder figures in addition to the eleven rows, and each carries an in-image placeholder banner". It names:
   - `fig:ladder`: 3 seeds, but without the data-matched MLP;
   - `fig:crossover`: the single-seed curve, "which three seeds do not support as a selection";
   - `fig:deletion`: seed 0 only.

   It adds that their data has landed and that each is to be redrawn, with the banner removed, before submission.
2. **Each of the three captions** now points to the index.
3. **The `fig_crossover` caption** now says its flat seed-0 line "is not the three-seed result: three seeds do not resolve the validation plateau".

- **`fig_maze_readverb`: confirmed NOT a placeholder** (PNG viewed).
  - It has no banner, and it draws the complete w14 3-seed read-verb data (`tab:verb`).
  - `figures/FIGURES.md` l.106's "PLACEHOLDER / awaiting w16" entry is stale. That file is the figures owner's, so I did not edit it.
- ⚠ The in-image text "wave 18 regenerates" (M7(e)) stays until the three PNG/PDFs are redrawn.

### MF-7: LACE and PoE, moved out of the headline tables

**Before:** LACE/PoE rows sat inside `tab:t7` and `tab:ladder`.

**After**
- **Both blocks moved, byte-for-byte**, into a new appendix table, `tab:lace-poe` (App D, L1406). Its caption puts their generation-table aggregates (0.3032 / 0.1968) beside:
  - the landscape controls, 0.1088–0.1860;
  - the store-removed arm, 0.1204;
  - "neither renders an exact-match request on any seed".
- **The full §3.7 LACE/PoE paragraph** moved with the table (renamed "LACE and PoE in full", L1433).
- **Main text (L874)** keeps one paragraph:
  - "at most 0.267 and 0.020… render no exact-match request… at the level of our landscape controls; we report them for completeness, not as evidence";
  - the standing scope, verbatim in substance: *as instantiated in this feature space at their own declared budgets… not a refutation of LACE in general*.
- **New limitation**, "No functioning generative baseline" (L1194).
- ⚠ A literal "render no valid full tuple" (the referee's wording) would be **false** on rows 3–4: LACE reads up to 0.267 valid novel tuples per corner. So the text uses "at most 0.267 / 0.020" and "no exact-match request" (0 / 0 / 0 in `w15-lace-poe-rows/FINDINGS.md`).

### MF-8: the headline comparison, restated

**Before:** abstract, contribution 1 and leaving note: "beating every read-path-matched learned rival on 109 of 109 corners" (against the weaker, 16,384-row set).

**After**
- **Abstract (L100 / L103):**
  - the store, fitted on 131,072 rows, reads 47.4954 / 47.4037 / 49.1193;
  - the **strongest read-path-matched rival of either set** reads 30.0275 / 28.9174 / 37.5138, and its 32,256 fit rows carry held-out values the store never saw;
  - the store scores above the data-matched MLP on the store's own 32,768 rows on every rung and seed;
  - it beats the 1× MLP on those rows by ≥ +3.117.
- **Contribution 1** (L227): the same comparison.
- **Leaving note** (L1268): "composition on Shapes3D that scores above the strongest read-path-matched learner of either of our rival sets and above a read-path-matched MLP fitted on the store's own rows".
- **No new number.** Every figure was already in §3.1, TABLES.md or App D.

### MF-10: K\* in the abstract

**Before:** "with the partition selected on validation (K\* = 32 wells)".

**After** (L100): "at K\* = 32 wells, the finest partition on a validation plateau (K = 5…32) that three seeds do not resolve".

### SF-7: contribution 3's "survives rivals trained ten times longer"

**After** (L246 / L243):
- "on validation it stays ahead of rivals trained ten times longer";
- "its scored replication against such rivals keeps the margin while one landscape control misses its collapse bar on one seed" (C-512 seed 1: random atoms 0.464 against 0.567).

### M7(e) regression: pipeline phrasing, stripped again

| before | after |
|---|---|
| "scored run 3 of 3", "`generation-table` run 3 of 3" | "one of that table's three pre-registered evaluation runs" / "one pre-registered evaluation run" |
| "runs 1--3", "4--6", "7--9", "10--12", "13--18", "9 of 18" | "one pre-registered scored run per seed", "three further scored runs", "of the eighteen pre-registered scored runs, nine were spent…", "six pre-registered runs" |
| "the shipped read" / `clu (shipped read)` | "the earlier store's read" |
| `knn_proto` | "nearest-prototype ceiling" |
| the three literal harness warning identifiers (`charter_C_…`) | plain-English equivalents |
| "this programme" (×2) | "this study" |
| "1 of 3" / "2 of 3" (encoder loop) | "one pre-registered scored run" |

- The generation table is now introduced in one sentence (SF-8's request).
- **Residual grep:** 0 hits in rendered text for `run N of N`, `runs N--N`, `shipped read`, `knn_proto`, `charter_C`, `programme`.
- The one `charter-C` left is **inside a w19 `\PLACEHOLDER{}`**, so it is untouched by rule.

---

## Not mine: left exactly as they were

| item | status |
|---|---|
| **MF-9 / M3** (deletion wording) | **Untouched. The Head rewrites it.** The abstract's "the store after deletion equals the store fitted without those rows", §1.2, the scope box and the ethics statement are byte-identical to before. |
| **MF-11** (page trim) | **Untouched, reported only.** Main text measured with a `\typeout{\thepage}` marker before the AI use statement: **v2 p. 19, style p. 19** (baseline measured the same way: **18 / 19**). Total pages **47 / 48** (baseline 45 / 46). Against the 9 pp ICLR limit, the main text is **≈ 2×**. My additions cost about one page (MF-1's paragraph, MF-3's paragraph and row, and the new limitation). |
| **MF-12** (AI use statement, code archive, deadline) | **Untouched.** The `\TODO`s are byte-identical. |
| **The 11 `\PLACEHOLDER{}`s** | **11 = 11 in both drafts.** The 10 w19 placeholders are byte-identical to before. The w20 placeholder's *text* was corrected as MF-2 requires; it is not filled and not deleted, and its "may land as a NEGATIVE" clause is verbatim. |

---

## Invariants, checked (not assumed)

- **Four style diffs: ALL EMPTY** (`STYLE-DIFF-CHECK-w20-referee4.md`, unmodified copy of `style_diff_check.py`):

  | check | result |
  |---|---|
  | numbers | empty |
  | placeholders | ordered and byte-identical |
  | artefact-path comments | 438 = 438, same order |
  | headings / labels | 111 = 111 and 82 = 82, same sequence |

  Also: floats and figslots 45 = 45 byte-identical; preamble identical; **0 non-ASCII** characters in both files.
- **Every number carries an SRC comment naming its artefact.** The provenance checker (`check_provenance.py`, unmodified copy) gives findings **identical to the baseline on both files**:
  - the same 7 "unsourced" lines, all template years and wave labels;
  - the same 1 tokenizer artefact (the K-set on the K\* rule line);
  - no new unsourced number and no number missing from its cited file.

  Every new number was first checked by grep against its artefact. For 32,768 I cite the leg JSONs, because `w13 TABLES.md`'s 32,768 is a different context (NR2).
- **Compile** (`build-final-20260923T192449Z/`, laid out like `w17-paper/`; `BIBINPUTS` / `BSTINPUTS` = `w17-paper/`). Sequence: pdflatex → bibtex → pdflatex ×3.

  | draft | undefined citations | undefined refs | `!` errors | overfull | `\bibitem`s | bibtex exit | pdflatex exit | pages |
  |---|---|---|---|---|---|---|---|---|
  | `DRAFT-v2` | **0** | 0 | 0 | 0 | 55 | 0 | 0 | 47 |
  | `DRAFT-v2-style` | **0** | 0 | 0 | 0 | 55 | 0 | 0 | 48 |

  Logs: `compile-DRAFT-v2.log` and `compile-DRAFT-v2-style.log` (and the per-file `.log`s in the build dir). The baseline build of the untouched snapshot (`build-baseline-192405/`) also had 0 overfull boxes; underfull boxes went 16 → 14 (v2) and 20 → 12 (style).

---

## "What would prove this wrong": the MF-1 answer, not softened

With the asymmetry disclosed, **the CMAZE-T row does not support a claim that the *store* composes better than the rivals** before w19 reports:
- the store's input carries a hand-built pair block the rivals never see;
- the same study showed a task-shaped interface lifting rivals more than the store (Config B);
- a rule-knowing composition-blind planner scores above the store;
- most held-out n-way rows do not need the door-then-tint interaction.

**What the row does support now:**
- the store gate: all four controls collapse, so the store is load-bearing under its own request;
- a margin over rivals that read the observation;
- exact deletion on the winning configuration.

I kept the sentence in the abstract, **scoped explicitly to "that request"**, rather than drop it. ⚠ **The Head may prefer the referee's alternative:** drop the CMAZE-T sentence from the abstract until w19 lands. I did not make that call, because it is a content decision.

The disclosure is **not** softened to protect the claim.

---

## Declared NOT done

- **No PDF page was visually inspected.** This box has no `pdftoppm`, `pdftotext`, `mutool` or PyMuPDF, and the Read tool cannot render PDFs without poppler. Layout was checked only through the log: 0 overfull boxes, and the page counts above.
- **No figure was redrawn** (MF-6); the banners remain in the images.
- `figures/FIGURES.md` and `tasks/w20/shapes-data-matched.md` were **not edited** (not mine). The stale entries are reported above.
- **No scored run, and no new number.** Every number printed is read from a w ≤ 16 artefact already cited in the draft's evidence tree.
- **Should-fixes not in my scope, not attempted:** SF-1–SF-6, SF-8's residual items beyond the M7(e) list, and SF-9–SF-14. Nice-to-haves N1–N7 were not attempted either.
  - Of these, SF-2 is covered by MF-7's new limitation, SF-3 is resolved by MF-7's move, and SF-8 is covered by the M7(e) list above.
- **I did not predict any w19 or w20 result.** No text says how either will land.
