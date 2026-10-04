# REFEREE-v3: final referee pass on `DRAFT-v2.tex` / `DRAFT-v2-style.tex` (w18 `referee-4`)

**Refereed files** (snapshots kept beside this report):

| file | SHA-256 | lines |
|---|---|---|
| `DRAFT-v2.tex` | `e41968894a71f9f1…` | 2,804 |
| `DRAFT-v2-style.tex` | `98e41695c9aea095…` | 2,793 |
| `references.bib` | `226ddd54790f3035…` | 55 entries |

Copies: `~/clu/outputs/w18-referee-4/DRAFT-v2{,-style}.snapshot.tex`. **Line numbers (L…) refer to the `DRAFT-v2.tex` snapshot.**

**Checked against:**
- the evidence root `~/clu/evidence/outputs/` (w ≤ 16 primary records);
- `w18-registry-fold/negative_results.md` (N351, N359, N360);
- `VENUE.md`;
- the w19 figure artefacts;
- the w19 and w20 task files.

**Reference point:** `REFEREE-v2.md` (3/10, M1–M7).

**Reviewer model:** unchanged from REFEREE-v2. A hostile but fair ICLR area reviewer who knows unlearning and compositional generation, reads the main text and the main tables closely, and skims the appendix.

---

## 0. Compile, length, and whether the two drafts agree

| check | result |
|---|---|
| Compile | `pdflatex → bibtex → pdflatex ×3` on scratch copies laid out like `w17-paper/`: **exit 0 on both, 0 undefined citations, 0 undefined references**, 55 `\bibitem`s. ⚠ `\bibliographystyle{iclr2027/iclr2027_conference}` hard-codes the sub-folder. A build from any other layout silently produces an empty bibliography, and my first flat-layout build did exactly that. Not a defect of the draft, but the Head should build from `w17-paper/`. |
| Total pages | `DRAFT-v2`: **45**. `DRAFT-v2-style`: **46**. |
| **Main text** (p.1 to the end of §6; the AI use statement starts on the next line) | **v2: ≈ 18 pp** (the body ends on p. 18, where the AI use statement begins). **style: ≈ 18 pp** (§6 on p. 18, AI use statement on p. 19). Measured with `\typeout{\thepage}` markers in scratch copies. |
| Against the hub bound (≤ 12 pp) | **over by ≈ 6 pp** |
| Against the ICLR 2027 limit (9 pp, strict, desk-reject) | **over by ≈ 9 pp: the main text is twice the limit.** REFEREE-v2 measured 12 pp. The growth comes from the task figures, the per-arm panels, and the w15/w16 fill-in. ⛔ Reported only; trimming is the Head's. |
| The two drafts agree | Same 54 section headings in the same order. Same label order. 11 / 11 `\PLACEHOLDER`. 5 / 5 `\TODO`. Identical citation-key sets. The multiset of result numbers is identical: an order-aware diff finds only digits inside labels and titles ("Shapes3D", `tab:t7`) and line shifts. ✔ |
| Anonymity (rendered text and PDF) | No name, institution, URL, e-mail or machine path in any non-comment line. PDF `/Author()` and `/Title()` are empty. ⚠ The `.tex` **comments** carry `_thinkpad/…`, `C:\chlu…` and similar paths. Harmless in the PDF, but **the source must never be uploaded as-is** (arXiv, or supplementary material). |

---

## 1. What changed since REFEREE-v2, checked item by item

| REFEREE-v2 item | status now | evidence |
|---|---|---|
| **M1** (the headline rivals are fitted on 16,384 rows; the store's pool is 131,072) | **OPEN, and worse than a placeholder in one respect.** The placeholder (L503–511) is honest in form: both sets, both pools, and "may land as a NEGATIVE". But **the zero-run parts of M1 were not done.** The abstract (L103–104), contribution 1 (L211–214) and the leaving note (L1209–1210) still lead with "beating every read-path-matched learned rival". The existing data-matched comparisons are still appendix-only or absent. **And the placeholder states the wrong store pool for the ladder** (MF-2). | §2 below |
| **M2** (two rival sets an order of magnitude apart) | **Disclosed** (L468–497, "Two rival sets, and which is stronger"). The stronger set is named correctly. The store is compared with the strongest rival of either set on every seed. **Adequate as disclosure.** The `\TODO` at L498 remains. | L468–502 ✔ |
| **M3** (exact deletion overclaims; the address book is carried, not refit) | **OPEN, deliberately left to the Head.** The abstract (L98–99): "the store after deletion equals the store fitted without those rows". §1.2 (L201–202): "the deletable object equals…". The scope box (L321–327) is still missing the "NOT a byte-level claim about the address book" bullet. The paper's own probe ("the object still answers red", L710–720) and its companion law (L2723–2726: **83.5%** of the post-delete vs never-inserted difference comes from the address book and coupling table fitted on the pool) are the rebuttal a reviewer will quote. | carried as MF-9 so the Head's list is complete |
| **M4** (positioning) | **Fixed.** "The intersection is empty" has 0 hits in both drafts. The naive-unit framing is stated at L132–139 and in the abstract (L90–93). SISA, compartmentalized diffusion, PINN and HNN each get one sentence (L166–181). "Can delete exactly only by retraining (the whole model, or a shard of it)" (L203–204). ✔ One residual inconsistency: see SF-3 (LACE / PoE). | ✔ |
| **M5** (two environments in one table) | **Fixed.** `tab:req` / `tab:req2b`, `tab:cmazet` / `tab:cmazeo` / `tab:cmazet-sym` and `tab:impl-maze` / `tab:impl` are each split. The two-environment figure was removed (L2033). ✔ But see SF-4: the draft's own **stricter** rule on budgets is broken by `tab:t7` and `tab:ladder`. | ✔ |
| **M6** (seed-count column) | **Fixed.** All 30 tables have a `seeds` column (checked mechanically). | ✔ |
| **M7**(a) AI use statement | **OPEN** (`\TODO`, L1233) | MF-12 |
| **M7**(b) pages | **Worse:** 12 pp → ≈ 18 pp against a 9 pp limit | §0 |
| **M7**(c) bibliography | **Fixed.** 55 / 55 entries cited, 0 undefined. ⚠ Nine works are cited as hand-typed text plus `\nocite` (SF-10). | ✔ |
| **M7**(d) abstract deadline passed | **Unchanged.** Head decision. | MF-12 |
| **M7**(e) pipeline language | **Partly regressed.** "Corrections absorbed" was moved out ✔. "BAR" has 0 hits ✔. But the w15/w16 fill-in brought back "scored run 3 of 3", "runs 1–3 / 4–6 / 7–9", "the shipped read", `knn_proto` and `charter_C` into rendered text (SF-8). Three figures carry an in-image banner "PLACEHOLDER · w≤14 evidence · wave 18 regenerates" (MF-6). | SF-8, MF-6 |
| **M7**(f) zero placeholders and TODOs at submission | 11 `\PLACEHOLDER` + 5 `\TODO` + **3 un-indexed bannered placeholder figures** (and a 4th listed as a placeholder in `FIGURES.md`) | MF-6 |
| S1 (the deletion seed count) | **Fixed:** `tab:del` is 3 seeds, and the "not capability erasure" check is at 3 seeds. ⚠ The RMS probe (L712–716) is still seed 0 with no seed label (SF-12). | |
| S3 (floor called a "comparator") | **Open** (L397–399) | SF-12 |
| S4 (bytes without dtype) | **Partly open** (SF-9) | |
| S5 (crossover placeholder) | **Resolved:** the rows landed, and the plateau is stated (see §5) | |
| S6 (the LACE / PoE-win reading) | Moot: they did not win. But see SF-2 (they barely function). | |
| S7 (`fig_ladder` panel b) | **Open**, and the figure is still a placeholder (MF-6) | |
| S8 (no generative baseline) | **Nominally answered by LACE / PoE. Substantively not** (SF-2) | |
| S9 (metric motivation), S10 ("CLU" never expanded; glossary) | **Open.** "CLU" is still never expanded (L95, L132, L1167, L1223). | SF-13 |
| S11 (rank/width warning appendix-only), S12 (maze verb disclosures appendix-only) | **Open** | SF-12 |

---

## 2. MUST-FIX (each one alone is enough for a reviewer to score the paper down, or it is a factual error or self-contradiction)

### MF-1. The CMAZE-T "store row" gives the store a different input from the rivals. The abstract leads with it, and the asymmetry is not disclosed in the main text.
- **Where:** abstract L114–118; contribution 3 L224–231; §4.4 L1054–1076; `tab:cmazet` L2485–2518.
- **Defect.** In w16 the CLU reads a `[bias | main | pair]` word request, while the matched rivals regress from the **observation**: *"rivals in their observation, lookup floors in their key, CLU/blind-sequential in the request"* (`w16-maze-wm/PREREG-w16-maze-wm-T1.md` l.55).
  - w16's own FINDINGS §2.6: *"The `[bias|main|pair]` head fixes which held-out 4-tuples are readable; the store decides only whether they are read cleanly"*, and n-way validity is the **same triple across ~15 store configurations**.
  - The pair block is hand-built interaction structure that no rival receives. w19 `maze-t-symmetric` exists because of this; its WORDS-FULL condition *"decides whether the pair block or the store carries the margin"*.
  - w16 FINDINGS also records that on this task family *"a task-shaped interface lifts the rivals more than it lifts the store"* for the **third** time: w14 verb, CMAZE-T Config B (+0.40 vs +0.20), CMAZE-O C-field.
  - §4.4 says only "Rivals are matched on read floats". The w19 condition table (`tab:cmazet-sym`, App S) is **not referenced from §4.4 or the abstract**.
  - The paper also applies a double standard to its own tripwire. On CMAZE-2, BLIND-SEQUENTIAL at 0.9077 showed that the request layer was broken (§4.2). On CMAZE-T, the composition-blind BLIND-SEQUENTIAL **planner** reads **0.9071**, above the store's 0.7246 on unseen n-way (`tab:cmazet` L2508; w16 TABLES). The w15 harness shipped the warning `composition_blind_PLANNER_scores_near_the_clu_on_unseen_nway` (L2461). §4.4 prints only the **learned** blind arms (0.3418 / 0.0716).
  - The n-way TEST rows are mostly not the interaction the task figure advertises:
    - `free` 345 / 328 / 309;
    - `rule_b` 316 / 307 / 288;
    - `interaction` 70 / 79 / 55;
    - `rule_a` 0 / 0 / 1;
    - out of 761 / 742 / 686 (`figures/rendered/REQUESTS-w19.json → cmazet/family_counts_per_seed`).

    No per-family score exists.
- **Why it sinks the paper if w19 lands negative:** if WORDS-FULL rivals match the store, the abstract sentence, contribution 3 and §4.4's "STORE row" all become an interface row. At present the maze section's only store-composition claim **depends on w19 winning**.
- **Minimal fix (zero runs):**
  1. In §4.4, add one sentence stating the input asymmetry and pointing to `tab:cmazet-sym`.
  2. Print the planner columns (0.9071 / 0.5509 / 0.6047) in the main text beside the learned blind arms, with the reason they do not void the row. They are rule-knowing oracles, not learned composition-blind arms. The paper must say why that distinction holds here when it did not on CMAZE-2.
  3. State the n-way family mix, and that no per-family score exists.
  4. In the abstract and contribution 3, qualify "the store does compose held-out n-way transitions" with "given a pair-indicator request the rivals do not receive; a symmetric-input test is pending". Otherwise drop the sentence from the abstract until w19 lands.

### MF-2. The M1/M2 placeholder states the wrong store fit pool for the ladder, and the draft contradicts itself.
- **Where:** L507 ("the store's own fit pool is 131,072 rows") against App D L1410 ("fitted from the same 32,768 fit rows as the store's content").
- **Defect.** Table 1's store (`w12-single-design-crossover`, value hold-out) is fitted on a **131,072**-row pool ✔. The **ladder** store at K\* is fitted on the generation-2 cache's first **`n_store_train = 32,768`** rows:
  - `w12b-nr6-and-kstar/kstar_ladder/*/rows_learned_1x.json → n_fit_rows 32768`;
  - `w12-composition-ladder/AMENDMENT-1.md` l.29;
  - `w13-referee-2/REFEREE-v1.md` SF-d.

  So the ladder comparison is **already size-matched** (store 32,768 vs rivals 32,256), and the rivals are **advantaged** because they saw the held-out values (L482–486).
- **The same error sits in the w20 task file** (`tasks/w20/shapes-data-matched.md` l.9, "The store's pool is 131,072 rows"). As written, w20 will refit the ladder rivals on **4×** the ladder store's data, which reverses the asymmetry. ⚠ The Head should correct the w20 brief before it scores.
- **Minimal fix:** state the pool per table: Table 1 store 131,072 vs rivals 16,384; ladder store 32,768 vs rivals 32,256 (combination-split population). Rewrite the placeholder's pending list to match.

### MF-3. The paper already holds a data-matched, read-path-matched rival, never prints it, and leaves its headline without insurance.
- **Where:** abstract L103–104; §3.1 L392–399, L468–511; §3.2; App D.
- **Evidence (w ≤ 14, zero runs):**
  - `learned_ref_ownpool` (`w12-composition-ladder/AMENDMENT-1.md` l.29, `AMENDMENT-3.md`) is an MLP at the **read-path pin** (130,514 parameters) fitted on **the same 32,768 rows as the store's content**.
  - `w13-seeds-and-tables/TABLES.md` l.703–710 prints it at **3 seeds on every rung**:
    - n = 2: 5.5667 / 7.5833 / 9.3000;
    - n = 3: 8.2167 / 10.9083 / 16.2583;
    - … n = 6: 10.6667 / 15.1167 / 21.0333;
    - on the **109-corner rows-3–4 leg**: 9.1835 / 11.1193 / 15.1284.

    The K\* store reads 32–67 on the same corner sets.
  - The 1× MLP (536k parameters, same 32,768 rows) is beaten at every rung and seed, with the smallest margin **+3.117** (`tab:1x`). It still sits in App D only, labelled "not the paper's matching rule".
  - `grep ownpool` over the draft: **0 hits**.
- **Why it matters:** these two rows are the paper's answer to "naive vs naive is parameter-only". Printed now, the Shapes headline survives a w20 negative on LSTM / transformer, at least for the MLP family and on the ladder. Left out, the paper survives only if w20 wins.
- **Minimal fix:**
  1. In §3.1 / §3.2, add one sentence plus one row (ladder) giving `learned_ref_ownpool` as the data-matched, read-path-matched MLP, with its source.
  2. Put the 1× margin (+3.117 minimum) in the main text as the conservative comparison.
  3. In the abstract, replace "beating every read-path-matched learned rival" with wording that names the fit pools. For example: "…every read-path-matched rival, including a data-matched MLP fitted on the store's own 32,768 rows".

  ⚠ `learned_ref_ownpool` on the 109-corner leg is fitted on 32,768 rows of the value hold-out pool, not on Table 1's 131,072. Say so.

### MF-4. N359 exposure: numbers the paper itself calls unmeasured are still printed and argued from.
- **Where:** `tab:encloop` last row (L1860: "margin over matched learner, alone → arm +12.078 → +10.311 | +24.687 → +26.048"); L1862–1864; App T L2668–2672; §3.7 L842–845; limitations item 4 (L1142–1143).
- **Defect.**
  - Registry N359: the w8 scorer loaded spoke A's **frozen-feature** rival, so the shortened leg's matched-learner rows are **not a measurement of the refit baseline**, and criterion D on that leg is **UNMEASURED**.
  - The draft states exactly this at L1893–1899 ("unmeasured, not void"). Yet the same appendix prints the margin row. It says "The pre-declared success bar fails on both rows" (L1862), and the success bar is a margin-growth criterion against that rival.
  - App T argues "The store's margin over the matched-parameter learner does not grow (+12.078 → +10.311 …; +24.687 → +26.048), so… 'the store shapes the representation' is not supported". Note also that row 4 *rises* by 1.36.
  - §3.7 generalises this to "the store's margin over a matched learner does not grow, so the store does not shape the representation".
  - On the one leg where it **was** measured, criterion D is **MIXED on both rows, with no claim in either direction** (L1882–1884; `tab:bard-gpu`). "Does not grow" is therefore a directional claim that neither leg licenses.
- **Minimal fix:**
  1. Delete the margin row of `tab:encloop`, or label it "frozen-feature rival (N359): not a measurement".
  2. Strike "the success bar fails on both rows" for the shortened leg, or restrict it to the store-only falsifier (A net ≤ 0 on two or more seeds; that falsifier does not involve the rival and stands).
  3. In App T and §3.7, replace "does not grow / does not shape" with "criterion D is mixed on the longer leg (no claim either way) and unmeasured on the shortened leg".
  4. Retitle limitations item 4 to "The encoder loop licenses no store claim" and keep the store-only falsifier.

### MF-5. Text still written for rows that have since landed, now contradicting the filled results.
- **(a) Limitations item 1, "No store row on the maze"** (L1125) contradicts §4.4 ("CMAZE-T: a store row", L1060), the abstract (L114–118) and contribution 3. Fix: "No store row on the route tasks (CMAZE-2, CMAZE-2b)", plus one clause on CMAZE-T's scope (MF-1).
- **(b) Intro "Thickness"** (L248–250): "The one environment with an exact route judge and three scored seeds returns a negative for the store". Reconcile it with CMAZE-T, which is on the same engine, uses an exact judge, has 3 seeds and gives a positive store row.
- **(c) App J L1869–1875**, *"… Its scoring stages were not run."* (SRC w8 fold) sits directly before "Criterion D on the longer leg, scored" (L1876). Delete the stale sentence.
- **(d) §3.7 L845–846**, "with the pending scored row of pre-registered criterion D on the longer leg": it is no longer pending. Replace it with the MIXED verdict.
- **(e) §4.1 L899–900**, "It is the one deletion leg in the study that sits inside scored runs": the CMAZE-T deletion rows were produced in scored runs 1–3 (`w16-maze-wm/t1/A1_test_seed*.json → DELETION`; `tab:cmazet`). Correct the claim.

### MF-6. Three bannered placeholder figures render in the PDF and are not counted among the 11 placeholders.
- **Where:**
  - `fig_ladder` (main text §3.2, L646, caption "(placeholder figure)");
  - `fig_crossover` (App E, L1552, "Placeholder figure, drawn from the single-seed middle rows");
  - `fig_deletion` (App G, L1688, "Placeholder figure, seed 0 only");
  - the Placeholder index (App W, L2756–2769), which says "Eleven rows remain pending".
- **Defect.**
  - All three PNG/PDFs carry an **in-image red banner, "PLACEHOLDER · w≤14 evidence · wave 18 regenerates"** (viewed). `fig_maze_readverb`, listed as a placeholder in `figures/FIGURES.md` l.106, renders without a banner. Its status should be confirmed.
  - The data these figures await **has landed**: 3-seed deletion (`w15-single-design-seeds`) and the 3-seed crossover (`w15-crossover-seeds`).
  - ⭐ `fig_crossover` draws the **seed-0 curve** as a clean flat line, with seed bands only at K = 5 and 32. That is exactly the picture the K\* plateau paragraph now says three seeds do not support. **The figure contradicts the paper's most careful paragraph.**
  - "wave 18 regenerates" in a figure is pipeline language (M7(e)).
- **Minimal fix (zero scored runs; the figures owner re-draws from existing artefacts):**
  1. Regenerate `fig_crossover` from `tab:kcurve3`, mean ± seed sd at every K.
  2. Regenerate `fig_deletion` at 3 seeds.
  3. Regenerate `fig_ladder` with panel (b) re-captioned as the data-matched comparison (MF-3).
  4. Remove every banner.
  5. Until then, add the three (and `fig_maze_readverb`, if it is still a placeholder) to the placeholder index.

### MF-7. LACE and PoE sit inside the headline table while performing at control level.
- **Where:** `tab:t7` L436–441; `tab:ladder` L577–580; §3.7 L848–863.
- **Defect.**
  - LACE reads 0.03–0.27 and PoE 0–0.02 tuples per corner at **27–31×** the reference budget.
  - On the generation table their per-factor aggregate is 0.303 and 0.197. The store's own **landscape controls** read 0.109–0.186 there, and `store_removed` reads 0.120 (`tab:gentable-kstar`). **Both have 0 exact matches on every seed.**
  - A reviewer will read two compositional-generation baselines scoring at control level as a failed re-implementation. Printed with "Δ −47" in Table 1, they cost credibility rather than add it.
  - The draft's own scoping sentence (L860–863) is correct. But a table row speaks louder than a caveat in §3.7, and §3.7 sits under the heading "The encoder" (SF-5).
- **Minimal fix:** move LACE / PoE out of `tab:t7` and `tab:ladder` into their own appendix table, with the generation-table per-factor numbers beside the controls. In the main text keep one sentence: "LACE and PoE, re-implemented at their own budgets in this feature space, render no valid full tuple (0 exact matches) and score at the level of our landscape controls; we report them for completeness, not as evidence." Then list "no working generative baseline" as a limitation (S8 stays open).

### MF-8. The abstract, contribution 1 and the leaving note overstate the headline relative to the paper's own disclosures.
- **Where:** L100–104; L211–215; L1209–1210.
- **Defect:** see MF-3. The abstract still says "beating every read-path-matched learned rival on 109 of 109 corners on every seed". §3.1 now concedes these are **the weaker of the two rival sets**, fitted at the deletion study's build/smoke scale (L392–394, L468–476), and that the paper "does not lead with the margins of Table 1 over its own rivals" (L495–497). **The abstract does exactly what §3.1 says the paper does not do.**
- **Minimal fix:** the abstract states the comparison §3.1 endorses: against the strongest rival of either set, plus the data-matched rows of MF-3. No new number is needed; 47.50 / 47.40 / 49.12 against 30.03 / 28.92 / 37.51 is already in §3.1.

### MF-9. M3 is still open: "exact deletion" in the abstract and §1.2 is defined on an object whose address book is not refit. (Held for the Head by ruling; listed so the must-fix list is complete.)
- **Where:** abstract L98–99; §1.2 L201–202; scope box L321–327; ethics statement L1240–1243.
- **Defect:** unchanged from REFEREE-v2 M3. The paper now proves the point at **three seeds**: red-requesting corners are unchanged after deleting every red row (L700–703), and the value identity "lives in the address book" (L717). The CMAZE-2b exact-head downdate (App M, L2138–2141: every atom and ≈ 99.98% of the address book change) shows a head that *is* downdated exactly. Contrasting that with the Shapes carry would strengthen the fix.
- **Minimal fix:** as REFEREE-v2 M3 items 1–4. The Head's wording.

### MF-10. K\* in the abstract reads as a clean selection. The main text is honest; the abstract is not.
- **Where:** abstract L101–102 ("with the partition selected on validation (K\* = 32 wells)"); §3 heading and Table 1 caption ("Composition at K\* = 32").
- **Defect:** §3 L370–383 and App E L1504–1520 state the plateau correctly (see §5). The abstract, the first thing a reviewer reads, says "selected".
- **Minimal fix:** "…at K\* = 32 wells, the finest partition on a validation plateau (K = 5…32) that three seeds do not resolve".

### MF-11. The main text is ≈ 18 pp against a 9 pp desk-reject limit.
Reported, not a drafting item. ⛔ Not trimmed by this referee. It is listed as a must-fix only because the paper cannot be submitted without the cut. The cut will force decisions (MF-7, SF-6, SF-7) that this report recommends anyway.

### MF-12. Venue items that remain blocking (Head-only)
- **AI use statement:** `\TODO` at L1233. **Required** by VENUE.md §7, and a Code-of-Ethics desk-reject risk if it is absent or inaccurate.
- **Anonymised code archive:** `\TODO` at L1251.
- **ICLR 2027 abstract deadline** (18 Sep 2026): passed. Whether an abstract was registered decides whether the venue is ICLR 2027 at all.
- **At submission:** 0 `\PLACEHOLDER`, 0 `\TODO`, 0 figure banners.

---

## 3. SHOULD-FIX

- **SF-1. The K\* control gate uses the weak `random_atoms` form, and the main text does not say so.**
  - App D L1449–1451: at K\* the control is the post-fit form (atoms randomised, address book not refit). The strong, refit form is a declared not-run at K\* and reads 0.25, not 0, at K = 5.
  - The abstract's "all four landscape controls collapsing to exactly 0.0000" rests on it.
  - Fix: one clause in §3.1, noting that the strong form still collapses by the margin at K = 5.
- **SF-2. S8 is only nominally closed.** See MF-7. Add "no functioning generative baseline in this feature space" to §5.1.
- **SF-3. The naive-unit framing is inconsistent over LACE / PoE.** §1.1 puts "systems built on units" out of scope by definition. LACE (an energy-based controllable generator over a pretrained generator's latent) and PoE (a six-expert product) are, by that definition, systems rather than naive units. Yet they appear in the naive-vs-naive table. Fix: one sentence classifying them, or (better) MF-7's move.
- **SF-4. Mixed budgets in one table, against the draft's own rule.** L1263–1264: "Tables in different environments, budgets or recipes never share a table". Broken by:
  - `tab:t7`: matched rivals, own-budget LACE / PoE, and the as-trained vs refit-post-fix recipes;
  - `tab:ladder`;
  - `tab:gentable-kstar`: K\* at 4.1×, `knn_proto` at a retired budget, LACE not byte-matched.

  Fix: narrow the stated rule to environments, or split the tables.
- **SF-5. Misplaced subsections.** The LACE / PoE paragraph and the generation-table paragraph sit under §3.7 "The encoder…" (L848–875, label `sec:bard`). Table 1's scope pointer ("Scope: §adapt") sends the reader to the encoder. Fix: move both paragraphs to §3.1 or their own subsection.
- **SF-6. The abstract is selective on the maze.** It reports the CMAZE-T win but not that CMAZE-O's rival margin was **withdrawn**, nor the two CMAZE-T declared negatives (2-way, value hold-out). Contribution 3 does mention the withdrawal. Fix: one clause in the abstract, or drop the maze world-model sentence (see MF-1).
- **SF-7. Contribution 3's "it survives rivals trained ten times longer" drops two qualifiers.** The ten-times check is on **validation** (A1). The scored ten-times replication (C-512) has `random_atoms` at 0.464 against 0.567 on seed 1, below the 0.15 bar (L1066–1072). Say both in the contribution line. §4.4 already does. This is also an N351-type paraphrase risk: the w16 Hub report made exactly this conflation (N351 item 4). ✔ §4.4 itself is correct on all eight N351 items (checked: 0.988 / 0.990 / 0.990; gate holds; three links; 80k = A1 VAL + C-512 TEST; 7.7e-16…1.2e-15 on the winner vs 0.0 on C-field; diagnostic transformer 0.022; 0.464 vs 0.567; C-field deletion ran).
- **SF-8. Pipeline language in rendered text, reintroduced by the w15/w16 fill-in.**
  - "scored run 3 of 3" and "`generation-table` run 3 of 3" (L865–867, L1311, L1906–1909);
  - "runs 1–3 / 4–6 / 7–9 / 10–12 / 13–18" (L1062, L1070, L1094, L1099, L2583, L2599–2615);
  - "the shipped read" and `clu (shipped read)` (L869–871, L1315);
  - `knn_proto` (L872, L1316);
  - `charter_C_control_did_not_collapse` (L2462);
  - "this programme" (L2379, L2426);
  - the "generation table", never introduced to an outside reader.

  Fix: "pre-registered run k of n" becomes "one of the three pre-registered evaluation runs"; define the generation table in one sentence; rename `knn_proto` to "a nearest-prototype ceiling".
- **SF-9. Byte figures whose parts are not all named.**
  - `tab:ledger-kstar` names float64 for the covariance only; atoms and address book have no dtype.
  - `tab:maze-del`: atoms and address book have no dtype (while `tab:maze-del-2b` names float32).
  - `tab:t7`'s LACE / PoE "31.08× / 27.13×" is "a parameter count against store-mode bytes" (float32 × 4 is implied, not stated).
  - `tab:hole` and `tab:rivals-del`: "bytes rewritten" has no dtype.
  - `tab:crr`: "store pinned at 524,288 B".

  Fix: name the dtype per part where the artefact names it (`CROSSOVER-3SEED.md` names atoms float32). Where the artefact does not, write "dtype as serialised; not stated in the artefact".
- **SF-10. Nine works are cited as hand-typed author–year text plus `\nocite`** (Tenenbaum & Freeman; Vasilescu & Terzopoulos; Vlasic; Cao & Yang; Ginart; Golatkar et al.; Raissi et al.; Greydanus et al.; Hall et al.; L141–145, 149, 170, 175–181, 186). VENUE.md §8 requires `\citet` / `\citep`. **Golatkar et al. is printed with no year at all** (L170). Fix: convert to `\citet`.
- **SF-11. `tab:del` formatting.** The seed-1 and seed-2 block prints `cos_min` to 17 significant digits (e.g. `0.04069122680502917`, L676–682) against 6 digits in the seed-0 block, and "0.0000" against "+0.0000". The anchor arm's L2 distances in App I have 18 digits (L1827–1828). Fix: print at the precision of the upper block; the artefact values are unchanged.
- **SF-12. Carried from REFEREE-v2, still open, all small.**
  - S3: "smallest paired delta against any comparator … +39.266 against the table floor" (L397–399). The floor is not a rival; give the smallest learned-rival delta, +39.716 (LSTM, seed 2).
  - S11: rank/width 0.9688–0.9844 above the 0.938 decorative threshold. One clause in §3.1.
  - S12: the maze verb's seen row is a lookup verdict, and the budget condition departs from the primary. One sentence in §4.3.
  - The probe at L708–716 (0.610083 / 0.614277 RMS) is seed 0; label it.
- **SF-13. Readability.** "CLU" is never expanded. "Well", "corner", "rung", "address book", "carry", "hole" and "landscape" are used without a glossary. No schematic of the object. The metric is under-motivated (S9, S10).
- **SF-14. Appendix Q (the prior-derivation ledger: GMOR, the friction-hole vault, the trilemma, reversible O(1) training…) and §5.3's "Latent Dynamics Model" theory are unrelated to a naive-unit evaluation.** They read as an off-topic physics dump and invite the question "is this one paper or two?". If any of these laws appear in the authors' own public preprints, VENUE.md §4 requires a third-person citation, and none is given. Fix: cut App Q to one paragraph with citations, or remove it. The Head decides.

---

## 4. NICE-TO-HAVE

- **N1.** The inference note is called "verbatim" twice in two different wordings (L585–587 and L633–635). Keep one.
- **N2.** `tab:ladder`'s row "store, n = 6 on n = 3's fit pool" is a derived, not measured, cell in a results table. Move it to the text; the paragraph already carries it.
- **N3.** `fig_ladder` panel (a) labels the rivals "MLP (524,288 B)" while the text matches on parameters, and names the arm "CLU" while the text says "store". Use one name.
- **N4.** `fig_maze_wm` labels the arm "CLU" as well. Same fix.
- **N5.** App intro L1262: "Appendix-maximalism is in force" is internal jargon.
- **N6.** The price-of-exactness line from REFEREE-v2 N1 is still not in the main text.
- **N7.** The TeX comment at L1253 ("this draft carries no \cite yet") is stale.

---

## 5. The four hunts the task named

### 5a. Does the paper stand if w19 and w20 return NEGATIVES? (all 11 placeholders)

| placeholder | if it lands negative | does the paper stand? |
|---|---|---|
| **w20 `shapes-data-matched`** (L503) | The Table 1 margins shrink or reverse against 131,072-row rivals | **Only if MF-3 and MF-8 are applied first.** As written, the abstract's first result sentence and contribution 1 rest on the weaker, 16,384-row set. With `learned_ref_ownpool`, the 1× MLP and the strongest-rival comparison printed, the Shapes claim becomes "beats a data-matched MLP and the strongest rival of either set". That claim survives a w20 negative on other families as a scoped claim. ⚠ MF-2 must be fixed first, or w20 will mis-scope the ladder. |
| **w19 WORDS-MAIN / WORDS-FULL / OBS / BOTH / rival budget** (L2567–2571) | The CMAZE-T margin belongs to the pair-indicator interface | **No, as written.** The abstract and contribution 3 present CMAZE-T as a store composition row. A WORDS-FULL negative turns it into the fourth instance of the paper's own "interface lifts rivals more" pattern. With MF-1 applied, the maze section still stands on deletion transfer (CMAZE-2 / 2b / T / O), the verb as an interface result, and a CMAZE-T row explicitly scoped "under a pair-indicator request". A negative then *confirms* the paper's interface thesis instead of refuting a headline. **That is the version to write now.** |
| w19 deletion row (L2572) | A declared not-run | Stands. Deletion transfer is already shown on four maze instruments. |
| w19 rollout validity (L2573) | Zero again | Stands. Already a declared negative. |
| 3 × CMAZE-T per-arm figure panels (L2529, L2536, L2544) | Stay undrawn (N363: not buildable without a persistence re-run) | Stands, but a reviewer will note that Shapes has qualitative per-arm samples and CMAZE-T has none. Say once in §4.4 that the w16 models were not persisted, so no per-arm sample can be drawn. |

**Every placeholder states its possible negative, and none presumes a win ✔.** The placeholder *text* is honest. The *paper around* the w19 placeholders is not yet written for the negative case (MF-1), and the paper around the w20 placeholder has not yet used the zero-run evidence that would let it survive one (MF-3).

### 5b. Is the NAIVE-UNIT framing carried through?
**Mostly yes.**
- The abstract (L90–93) and §1.1 (L132–139) state it.
- No "best system", "state of the art" or "the only" survives. I grepped the abstract, intro, leaving note and title and found none.
- The title ("Exactly Deletable Compositional Generation on a Frozen Encoder") makes no system claim.

**Residual problems:**
- LACE / PoE are systems inside the naive-vs-naive table (SF-3).
- The leaving note (L1209–1210), "composition that beats read-path-matched learners", is naive-unit-compatible, but it inherits MF-8's data asymmetry.

### 5c. K\*: is the plateau stated honestly?
**In the main text, yes.**
- L370–383: "on three seeds, however, the rule cannot separate the plateau K = 5…32: the seed standard deviation … exceeds the rule's 0.02 tolerance at K = 10, 16, 20, 24 and 28 … and at K\* = 32 (0.0259), so three seeds do not distinguish K\* from its neighbours". Every sd was checked against `w15-crossover-seeds/CROSSOVER-3SEED.md` ✔.
- App E (L1504–1520) adds that the ledger minimum at K = 24 *is* resolvable (7,060 B band vs 12,203 / 30,507 B gaps ✔), and that K = 20 fails the strict gate ✔.

**Not honest in two places:**
- the abstract (MF-10);
- `fig_crossover`, which draws the single-seed curve with no middle-K spread (MF-6).

### 5d. N351 and N359
- **N351** (a paraphrase is not a source): **the main text is not exposed.** Every one of the eight disputed CMAZE-T / O facts is stated as the primary artefact states it. No SRC comment cites `w16-advisor-report.md`. Contribution 3's summary line is the one paraphrase at risk (SF-7).
- **N359** (the frozen-feature rival, so CPU BAR D is UNMEASURED): **the paper is exposed** in App J, App T, §3.7 and limitations item 4 (MF-4). It states the N359 conclusion correctly in one paragraph (L1893–1899) and contradicts it in four others.

### 5e. Every claim cross-checked against its artefact
Checked at cell level against their artefacts; each was found:
- **Numbers new since REFEREE-v2:**
  - `CROSSOVER-3SEED.md`: every sd, the ledger minima, the band, the gaps, 3.9675, the K = 20 gate;
  - `T7-3SEEDS.md` and `w15-single-design-seeds/FINDINGS.md`: every seed-1 and seed-2 deletion cell, 2,074,368 / 2,078,464, the rival retrain-without table, the 33.500 / 33.600 / 33.550 capability rows;
  - `w13 TABLES.md` l.710: 30.0275 / 28.9174 / 37.5138 with their family identities ✔;
  - spoke A `mlp.json` / `REPORT.md`: 117,878–119,858 parameters, `input_kind: onehot`;
  - `w15-coverage-control`: all 17 figures;
  - `w15-bar-d-score`: every cell of `tab:bard-gpu`, including the Unicode-minus cells;
  - `w15-lace-poe-rows`: all rows 3–4, ladder and generation-table cells;
  - `w16-maze-wm` TABLES / FINDINGS: 32 CMAZE-T / O figures, including rival parameter counts 4,781 / 4,290 / 5,129;
  - `REQUESTS-w19.json`: every count in the Figure 4 caption (26 / 11 / 3, 13 / 13 / 25, 2 / 3 / 6, 0 / 1 / 0, table 67 / 26 / 46, CLU 5,177 / 5,167 / 5,354).
- **No transcription error was found.** The defects above concern framing, staleness and omission, not wrong numbers. The **one factual error** is the ladder pool in MF-2.
- **Located after a second pass:** "5,040 steps, 40 epochs, batch 256, lr 0.003" for spoke A are in the nested training block of `_thinkpad/w4/spoke-a-rivals/rivals/seed_0/mlp.json` ✔. ⚠ The same file also carries an `"epochs": 20` block (a second schedule, probably the partial-request variant). The SRC owner should confirm that L478–480 quotes the block for the `full_only` artefact that was actually scored. **Every number I set out to check was located.**

---

## 6. SIMULATED VERDICT

**Score: 4 / 10 (reject) as submitted. Confidence 4 / 5.**
With MF-1 to MF-10 applied and MF-11 / MF-12 resolved by the Head, I expect **5** (marginally below threshold). It could reach **6** if w20 lands a data-matched win, or if MF-3's rows carry the Shapes claim on their own.

**How it moved from 3, and why.**

*Up (+1):*
- The positioning problem (M4) is gone. The paper now makes a narrow, defensible "naive unit vs naive units" claim with SISA and CDM distinguished.
- The rival-set inconsistency (M2) is disclosed honestly, including the uncomfortable fact that Table 1's rivals are the weaker set.
- Shapes deletion is now at three seeds, with the capability-erasure non-result at three seeds.
- The K\* selection is re-stated as a plateau.
- Criterion D is reported as MIXED with no claim.
- Environments are split, every table has a seeds column, and the bibliography is real.
- The paper's self-refutation culture (the lookup floor, withdrawn margins, declared negatives with failing links) is intact and now extends to CMAZE-O. Reviewers reward this.

*Not up more, because:*
1. The headline composition claim still leads with the weakest rivals, although the paper holds better evidence (MF-3, MF-8).
2. The maze world-model row a reviewer will read in the abstract compares the store under a hand-built pair-indicator request with rivals reading raw observations, and a composition-blind planner beats it (MF-1).
3. "Exact deletion" is still defined on an object that still answers the deleted value (MF-9).
4. A new class of internal inconsistency arrived with the fill-in: stale "pending" / "not run" / "no store row on the maze" sentences, and N359-contaminated margins argued in four places (MF-4, MF-5).
5. The paper doubled in length to twice the venue limit.

**Rationale, paragraph 1 (what I believe).** The mechanism is correct and simple. Downdating the untruncated statistic and re-eigendecomposing is exact to roundoff: float64 ~1e-16 on four maze instruments; float32 ~1e-8 on Shapes at three seeds. It needs no data and is priced honestly in a four-part ledger. The Shapes store composes held-out tuples far better than every naive unit the paper fits, including one fitted on the store's own rows. That comparison is buried in the evidence, and the paper should lead with it. Its negatives are real and well argued: the verb as an interface property, withdrawn margins, rollouts that never validate, and the address book that remembers.

**Rationale, paragraph 2 (why I still reject).** The reader meets these results through the wrong door:
- The abstract's first result is a margin over rivals the paper itself calls the weaker, smoke-scale set.
- Its maze result is a margin obtained with an input advantage the paper does not disclose outside a pending appendix table.
- Its deletion guarantee is phrased on an object that the paper's own probe shows still answers the deleted value.
- A reviewer who finds any one of these in the appendix will discount the rest. That is unfair to a paper this careful, but it is how reviews work.

Every must-fix above is an edit that needs **no new scored run**. That is why the ceiling is higher than the score.

---

## 7. Judgement: **READY for the Head's editing pass, NOT ready for submission.**

It is ready for the editing pass because:
- every must-fix is an edit or a zero-run figure regeneration, with its location and minimal fix stated;
- the two drafts agree number-for-number;
- no transcription error was found;
- the placeholders are honestly worded.

It is not ready for submission because of MF-1 to MF-12:
- 11 placeholders, 5 `\TODO`s and 3 bannered figures remain;
- the AI use statement is missing;
- the main text is ≈ 18 pp against 9.

**Order suggested to the Head:**
1. Correct the w20 brief's ladder pool now, before w20 scores (MF-2).
2. Apply MF-3 / MF-8 and MF-1, which make the paper survive w19 and w20 negatives.
3. Apply MF-4 / MF-5, pure consistency fixes.
4. Write the M3 wording (MF-9).
5. Then trim (MF-11).

### The three things most likely to sink it at review
1. **"Naive vs naive" is not data-matched, and the abstract leads with the weakest rivals** (MF-3 / MF-8, M1). The cheapest reviewer attack on the headline, and the one the paper can already answer from w ≤ 14 evidence.
2. **The CMAZE-T store row gives the store a pair-indicator input the rivals lack, and a composition-blind planner outscores it** (MF-1). The paper's own CMAZE-2 standard voids rows for less.
3. **"Exact deletion" where the model still answers the deleted value** (MF-9 / M3). A privacy-minded reviewer will call the headline property misleading, and the paper supplies the evidence.

*(A desk-level gate precedes all three: the page count, the AI use statement and the passed abstract deadline, MF-11 / MF-12.)*

---

## 8. What this referee did NOT do (DECLARED NOT-DONE)
- **No scored run, no compute beyond `pdflatex` / `bibtex`** on scratch copies. No number was re-derived. Every number above is read from an artefact, and each is named where it is used.
- **I did not edit either draft**, `references.bib`, the figures, the registry or any task file. The w20 brief error (MF-2) is reported, not corrected.
- **I viewed 6 of the 14 figures** (`fig_ladder`, `fig_crossover`, `fig_deletion`, `fig_maze_wm`, `fig_maze_readverb`, `rc_w19_t7_rival_succeeds_clu_fails`), as PNG. I did not open the other eight or any PDF version, and did not confirm that each PDF matches its PNG.
- **External literature was not re-checked** (no web access used). Bib-entry accuracy, e.g. `quan2026exact` and `sun2025generative`, is **not verified** by me.
- **The PDF anonymity check** grepped uncompressed bytes only. Metadata inside compressed figure streams was not decoded.
- **I did not read `DRAFT-v2-style.tex` line by line.** I checked it mechanically against v2 (sections, labels, placeholders, citation set, result-number sequence) and measured its page count.
- **The per-family CMAZE-T scores do not exist.** I report the family mix only.
