# FINDINGS: w18 `placeholder-fill`, SECOND PASS (batmac). COMPLETE

**Branch** `w18/placeholder-fill`. **Scored runs: 0. Compute: pdflatex/bibtex only.** No figure was modified.
**Input:** the post-w19 drafts (sha `33d1b530…` / `9864922f…`, = w19-draft-figure-slots "after"), snapshotted in `pre2/`.
**Output:** `w17-paper/DRAFT-v2.tex` `de7b5fdb…`, `DRAFT-v2-style.tex` `2359dfe1…` (`SHA256-post2.txt`).
Rebuild from `pre2/` with `w18_placeholder_fill/apply_fill2.py` (text in `fill_text2.py`; every anchor asserted exactly once; identical bytes into both drafts).

## 0. Acceptance

| criterion | result | proof |
|---|---|---|
| the four w15 placeholders gone, **filled, not deleted** | ✅ 0 `\PLACEHOLDER{w15` left in either draft; each replaced by its filled row/paragraph/table (§1) | `grep -c` = 0 / 0; `apply_fill2.py` log |
| coverage-control TODO closed | ✅ the `SRC-PENDING` comment and the orange TODO are replaced by the artefact's numbers with real `% SRC`. TODO count 6 → **5** (the other 5 are pre-existing, not mine to close) | STYLE-DIFF-CHECK-v2-pass2.md §2 |
| exactly the 10 w19 placeholders remain | ✅ **10 = 10**, and `apply_fill2.py` asserts they are **byte-identical and in the same order** as in the w19 "after" drafts (3 × w16 CMAZE-T per-arm figure slots + 7 × w19 `maze-t-symmetric`) | STYLE-DIFF-CHECK-v2-pass2.md §2 lists them |
| both drafts compile, **0 undefined citations (checked)** | ✅ pdflatex → bibtex (`BIBINPUTS`/`BSTINPUTS` = `w17-paper/`) → pdflatex ×3, all 5 steps exit 0. bibtex: style `iclr2027/iclr2027_conference.bst` found, 63 entries, 0 errors. Final pass: **0 errors, 0 undefined citations, 0 undefined references, 0 "There were undefined", 0 "labels may have changed", 0 overfull. 46 pp each** (was 44) | `compile-v2-pass2.log`, `compile-v2-style-pass2.log`, `DRAFT-v2*-pass2.blg`, `final-compile2-20260923T172846Z/`; script `w18_placeholder_fill/compile2.sh` |
| four diffs EMPTY | ✅ numbers **EMPTY**, placeholders **EMPTY**, artefact comments **EMPTY** (paths and whole strings, same order), section order **identical** (109 headings, 82 labels). Extras: floats and preamble byte-identical. **ALL DIFFS EMPTY** | `STYLE-DIFF-CHECK-v2-pass2.md` (run on the installed drafts; also copied to `w17-paper/`) |
| every number names its artefact | ✅ provenance checker (2,786 tokens): UNSOURCED / NOT-FOUND findings **identical to the pre2 baseline** (6 pre-existing lines, 1 pre-existing not-found). Every inserted number found verbatim in the file its `% SRC` names (`w15-bar-d-score/FINDINGS.md`, `w15-lace-poe-rows/FINDINGS.md`, `w15-coverage-control/FINDINGS.md`) | `final-compile2-20260923T172846Z/prov-{before,after}-*.txt` |
| pure insertion/replacement | ✅ `diff pre2 → installed`: 32 lines removed, **all** of them placeholder arguments, the coverage TODO + SRC-PENDING, the pass-1 coverage table row, the 3-line header comment and the Placeholder-index paragraph (both pass-1 bookkeeping). No paper prose outside those was touched (see §3 for the one pass-1 sentence) | `diff pre2/DRAFT-v2.tex ../w17-paper/DRAFT-v2.tex` |

⚠ **Correction to the launch note about pass 1:** pass 1's *final* compile did resolve bibtex (it copied `iclr2027/` into its build dir, so the .bst was found): its final `.log` has **0** undefined citations. The 226 were in w17's own `compile-v2*.log`. Pass 1's claim stands; this pass uses the prescribed BSTINPUTS route anyway.

## 1. What was filled, and how it reads

### 1a. `bar-d-score` (2 placeholders) → **MIXED, NO CLAIM EITHER WAY**
- **App. encloop** (was placeholder 3): paragraph "Criterion D on the longer leg, scored" + new **`tab:bard-gpu` (GPU pins only: 4,000 / 32,768 / dec 1,800 / warm 4,000; 3 seeds a column)**. It prints M(alone), M(loop_swap), growth with CI95 on both rows, verdict **MIXED: no claim** on row 3 and row 4, 0 VOID seeds, absolute store/learner means, and the three controls collapsing (≤ 0.490, bar ≤ 1.0), with **"the margin is ~0 ⇒ no store claim in either direction"** said in words.
  - The saturation of `loop_swap` (store 1.000/0.995/0.998, learner 1.000/0.991/0.997 of the ceiling) is labelled **"descriptive, not a bar"**, and it is why growth could not show.
  - The wiring defect is included: the CPU leg's `rival_mlp_180k` rows at 0.000 are the frozen-feature rival, so **CPU Criterion D is unmeasured, not void**. Re-scoring it is declared not-run.
- **App. encoder-main** (was placeholder 4): the decomposition restated on the longer leg. **ORDERING NOT REPRODUCED**: A′ < 0 on 2/6, A″ > 0 on 6/6.
  - The swap-loss half survives. "Unfreezing hurts" is **not a general finding**: it is a property of one seed's well-trained `alone` control.
  - The features improved for every arm, so this is a result about the encoder, not the store.
  - ⛔ **GPU and CPU are two separate blocks.** The GPU A′/A″ are only in `tab:bard-gpu`, and the CPU A′/A″ stay only in `tab:encloop`. The prose names each leg separately. Nothing is pooled.

### 1b. `lace-poe-rows` (2 placeholders) → **LACE/PoE do NOT beat the CLU, scoped as the spoke scoped it**
- **`tab:t7`**: the placeholder row is replaced by LACE and PoE rows (row 3 and row 4 per seed). Their Δ is printed as **rival − store, the artefact's sign, and labelled so**: −47.33/−47.33/−48.89 and −47.49/−47.40/−49.12. A full-width note row says they are **added, the rivals are not replaced**, they run at their **own declared budgets** (31.08× / 27.13× the 524,288 B reference, labelled as a parameter count against store-mode bytes), they are **not** read-path matched, and the store wins 109/109 on every seed.
- **`tab:ladder`**: LACE and PoE rows are added on every rung, 3 seeds, labelled "own budget, not matched". The existing rival rows are untouched.
- **§3.7 (`sec:adapt`)** (was placeholder 2): paragraph "LACE and PoE, at their own budgets", with store 47.50/47.40/49.12 vs LACE 0.165/0.073/0.229 vs PoE 0.009/0.000/0.000, 109/109 on every seed, and every rung won on every corner. Budgets are spelled out.
  - ⛔ **Scoping carried verbatim.** "as instantiated in this feature space at their own declared budgets… **not a refutation of LACE in general**". The spoke's own sentence is quoted: "LACE/PoE, as re-implemented at these budgets, do not produce valid full tuples here", not "the CLU out-composes LACE in general". The zero-exact-match caveat is included.
- **Generation table run 3/3**: a paragraph in §3.7 plus a new **`tab:gentable-kstar`** in App. B (its own instrument, never merged with `tab:t7`; seeds a column; every byte figure names its convention). It covers K\* 0.7500 ± 0.1807 against the shipped read 0.5949 ± 0.0082. **S1 PASS ⇒ the store claim stands. S2 FAIL on seed 0 (−0.0185) ⇒ "beats the shipped read" is NOT made.** S3 LOSE vs `knn_proto` (−0.1150). S4 WIN vs LACE/PoE. 3 seeds, nothing NOT-RUN.

### 1c. `coverage-control` (pass-1 TODO) → closed from the artefact
- The Hub's statement is **checked against the artefact and confirmed**. The val verdict is **"NO RISE TO DECOMPOSE ON VAL"**: rise +1.200 / +0.567 / +0.367 against the 1.0 margin, so it is below the margin on 2 of 3 seeds.
- **Coverage, by construction**:
  - n = 6 at n = 3's coverage **is** rung n = 3.
  - TEST 61.614 vs 66.603. The whole +4.989 (+5.758/+3.467/+5.742) is coverage.
  - Coverage share 1.00 on every seed; mask channel 0.000.
  - The identity checks are printed (acceptance 1.000000, 3,600/3,600, 8,640 rows).
- ⛔ **The TEST cell is labelled "derived, not measured"** twice: in the text, and in the `tab:ladder` row, which now prints 60.150/62.783/61.908 under n = 6 with the label "derived".
- The val per-seed scores are printed: 22.767 (21.767/23.050/23.483) vs 23.478 (22.967/23.617/23.850).
- **The Head's inference note travels with the ladder table verbatim**, quoted from the artefact: *"We beat the naive fair rivals at every n; the CLU's own 2-way → 6-way rise is not to be inferred, because rungs 3–6 share corners and differ in coverage."*

## 2. Tables and CMAZE rule
New tables: `tab:bard-gpu` (App. encloop) and `tab:gentable-kstar` (App. B). No CMAZE table was touched, and none of the new tables contains a maze row. Seed count is a column in every new table and row. Byte figures: store-mode bytes (524,288 B), four-part never-netted (K\*), atom block float32 (`pca_tensor`), parameter counts labelled as such.

## 3. ⚠ Flagged for the Head (NOT changed: it is prose outside the placeholders)
The new artefacts make these existing sentences stale or unsafe:
1. **§3.7 L~810–813**: "a wash of two real effects, **unfreezing hurts and a swap loss repairs**, and the store's margin over a matched learner does not grow … **with the pending scored row** of pre-registered criterion D". Problems: (a) "pending" is now false; (b) "unfreezing hurts" does not reproduce at GPU pins (bar-d §3); (c) see item 3 for the CPU margin.
2. **App. encloop L~1843**: "Its scoring stages were not run." This is now false; they ran in w15 `bar-d-score`. The filled paragraph follows it directly.
3. ⚠⚠ **The CPU (shortened-decoder) matched-learner numbers**:
   - `tab:encloop`, row "margin over matched learner, alone → arm" (+12.078 → +10.311; +24.687 → +26.048).
   - The sentence "the matched learner sits at or within 0.02 of zero (0.006 on row 3)".
   - The same numbers in App. encoder-main (L~2638), and contribution/limitation item L1110 ("The store does not shape the representation").
   - bar-d-score §4 says the w8 CPU scorer loaded **spoke A's frozen-feature rival, not the refit baseline**, so those rows "are not a measurement of the refit baseline" and CPU BAR D is **unmeasured**. I have **not** verified that the w13 TABLES.md margin rows come from that exact scorer run. The pins and the A′/A″ values match the w8 CPU leg exactly, so it is very likely. **The Head should check this before the paper leans on "+12.078 → +10.311".**
4. **One pass-1 sentence, changed on purpose.** Pass 1 wrote "…and no new number is **printed** for it. On validation … the restricted n = 6 fit's rise over n = 3 is below the pre-declared margin…". The first clause became "**measured**" because a derived number is now printed. The second clause was **wrong as transcribed** (the rise is n = 6 over n = 3, B − A; the restricted fit's rise is C − A = 0), so it was replaced by the artefact's statement. Both sentences were my own pass-1 text, written under SRC-PENDING.
5. Placeholder index (App. W) and the file-header comment were rewritten to the true state (10 w19 placeholders). w19-draft-figure-slots asked for this.
6. The §3.2 paragraph "The inference note, **verbatim in substance**" is a paraphrase. The artefact's verbatim note is now printed right after it, and the Head may want one of the two.
7. Length: 44 → 46 pp. Main text: §3.7 grows by about half a page. The ≤ 12 pp bound was already exceeded.

## 4. DID NOT RUN / NOT DONE
- **`fig_ladder`** was not regenerated with LACE/PoE, and its caption still says "(placeholder figure)". Figure regeneration is out of this pass's scope. `fig_crossover` and `fig_deletion` are the same as in pass 1.
- No re-scoring, no evidence edit, no registry or matrix edit. The CPU-leg re-score with the repaired rival is the bar-d spoke's declared not-run.
- The w19 placeholders, the w19 figure slots and all figure files are untouched.
- The pass-1 items MF2 and MF14(i) and the §8 caption-convention patch are still left for the Head, as in pass 1.
- Visual check: rendered pages 6, 9, 13, 14, 24, 25 and 33 (Ghostscript, 80–200 dpi). `tab:ladder` was already very small in the w19 build and is now slightly smaller; it is legible at 200 dpi. There was no full proof-read of the 46 pages.
- One logging slip, corrected in LOG.md: a sha guard tripped on sort order and a trailing echo logged a false "installed" line. Nothing was installed at that moment. The correction line follows it.

## 5. Files (this pass)
`pre2/` · `build2-*/` (every build kept) · `final-compile2-20260923T172846Z/` (final compile of the installed drafts, with prov-before/after) · `compile-v2-pass2.log`, `compile-v2-style-pass2.log`, `DRAFT-v2*-pass2.blg`, `DRAFT-v2*.pdf` · `STYLE-DIFF-CHECK-v2-pass2.md` · `SHA256-post2.txt` · `FINDINGS-pass1.md` (pass 1 as written) · scripts on the branch: `w18_placeholder_fill/{fill_text2.py, apply_fill2.py, compile2.sh}`.

---
# (FIRST PASS, as written; superseded where the second pass above says so)
# FINDINGS: w18 `placeholder-fill` (batmac). A PARTIAL pass by design

**Branch** `w18/placeholder-fill` (base `148f3b7`). **Scored runs: 0. Compute: figures and pdflatex only.**
**Drafts edited:** `~/clu/outputs/w17-paper/DRAFT-v2.tex` and `DRAFT-v2-style.tex`. The pre-edit snapshots and their sha256 values are in `pre/`; the post-edit sha256 values are in `SHA256-post.txt`.

## 0. Acceptance for this pass

| criterion | result | proof |
|---|---|---|
| exactly 4 `\PLACEHOLDER{}` remain, all named, none deleted | ✅ **4 in each draft**, and they are **byte-identical** to their pre-edit arguments | `STYLE-DIFF-CHECK-v2.md` §2 lists them. A script comparison of pre and post `macro_args` returns `post == awaiting(pre)` → **True** for both drafts |
| both drafts compile | ✅ pdflatex → bibtex → pdflatex ×2. **0 errors, 0 undefined references, 0 overfull boxes, 39 pages** each | `compile-v2.log`, `compile-v2-style.log`; the PDFs are copied here |
| four diffs empty (numbers, placeholders, artefact comments, section order) | ✅ **ALL DIFFS EMPTY**. The extra checks also pass: floats and figslots byte-identical, preamble identical, 0 non-ASCII | `STYLE-DIFF-CHECK-v2.md`, also copied to `w17-paper/` |
| every inserted number names its artefact | ✅ The provenance checker read **2,484** number tokens. Its UNSOURCED and NOT-FOUND findings are **identical to the pre-edit baseline** (6 pre-existing unsourced lines, all citation years or `iclr2027`, plus 1 pre-existing not-found list). **Every new number was found verbatim in the file its `% SRC` names** | `build-*/prov_orig.txt` against `pre/prov_orig.txt` |

⚠ **Count correction.** The launch prompt says there are **30** placeholder uses and **26** to fill. The checker's `macro_args` count finds **29** uses in each draft, and so **25** to fill. The w17 STYLE-DIFF-CHECK also records 29. Nothing is missing: the "19 w16 mentions" are 18 by my count (P13–P30).

## 1. AWAITING, second pass. Left exactly as they were

| placeholder | draft line (orig / style) | spoke |
|---|---|---|
| `w15 \texttt{lace-poe-rows}: LACE and PoE at their own budgets; may beat the store` (`tab:t7`) | L378 / L378 | **lace-poe-rows**, still running on the other box |
| `w15 \texttt{lace-poe-rows}: LACE… and PoE… rows (at their own declared budgets)…` (§ encoder) | L737 / L734 | **lace-poe-rows** |
| `w15 \texttt{bar-d-score}: the scored BAR D row of the longer leg…` (app: encoder loop) | L1710 / L1696 | **bar-d-score** |
| `w15 \texttt{bar-d-score} (one \texttt{encoder-loop} scored run, new pre-registration)…` | L2392 / L2381 | **bar-d-score** |

None of these was deleted, softened, or given a number. The Placeholder-index appendix now names exactly these four and the coverage TODO (§2).

## 2. ⚠⚠ coverage-control. Filled, but its artefact is NOT on this machine

The launch lists `coverage-control` as landed. **It is not staged on batmac.** There is no `w15-coverage-control` under `evidence/outputs/`, and a search of `evidence/`, `outputs/`, `stage/`, the worktrees and `/` (depth 5) found no artefact. TABLES.md §6.1 and the registry fold both still list it as AWAITING. What I did, and why it is honest:
- **Filled with no new number.** The construction argument is sourced: rungs 3–6 probe the identical 120 corners (`w12-composition-ladder/FINDINGS.md`), so n = 6 on n = 3's fit pool **is** rung n = 3. The TEST cell is printed as **"derived, not measured"** and points at the existing n = 3 cell. A labelled row was added to `tab:ladder`.
- The **validation verdict "NO RISE"** (below the pre-declared margin on 2 of 3 seeds) is transcribed from the Hub's launch instruction. It carries a `% SRC-PENDING: w15-coverage-control` comment and a visible orange **`\TODO{no artefact for w15 coverage-control on the drafting machine…}`**. The TODO count goes from 5 to 6, identical in both drafts. **The per-seed validation scores are NOT printed.**
- ⛔ **Second pass:** stage `w15-coverage-control/`, replace the TODO with its per-seed scores and a real `% SRC`, and check that the artefact says "NO RISE, 2 of 3 seeds" as the Hub stated.

## 3. What was filled (25 placeholders), and from where

| P | row | filled with | source (under `evidence/outputs/`) |
|---|---|---|---|
| P4 | coverage-control | §2 above | construction: `w12-composition-ladder/FINDINGS.md`; verdict: Hub (TODO) |
| P5 | `tab:del` seeds 1, 2 | 6 rows: value, red sphere and item at seeds 1 and 2, with red ∧ cube vacuous. Exactness 1.074e-08 … 6.253e-08; cos_min at full JSON precision (not rounded). The caption title now says seeds 0, 1 and 2 | `w15-single-design-seeds/T7-3SEEDS.md`, `…/deletion/test_seed{1,2}_K32/deletion.json` |
| P6 | deletion prose | seeds 1–2 reproduce seed 0. The bytes rewritten are per seed (2,074,368 and 2,078,464; the 2,081,024 in the text is seed 0's). ⭐ New paragraph **"Deletion is not capability erasure, at three seeds"**: the 20 deleted corners still compose red on all 3 seeds (seed 2: 33.600 → 33.550) | `w15-single-design-seeds/FINDINGS.md` |
| P8 | CMAZE-2b deletion | a short main-text statement, plus the new **`tab:maze-del-2b` (CMAZE-2b ONLY)** with its detail paragraph in the read-verb appendix. Statistic residual 7.212e-16 … 9.093e-16 and projector 0.000, quoted together; clean checkout; the exact-head downdate changes ~99.98 % of the book; **deletion does not induce refusal**; the carry path destroys survivors; the transformer retrain costs 88.6–90.8 s over 256,000 reads; two legs are declared not-run | `w15-maze-deletion-2b/TABLE.md`, `FINDINGS.md`. The carry seconds come from FINDINGS, because TABLE prints "—" there (TABLES.md §6.2c) |
| P9 | crossover-seeds | new `tab:kcurve3` (all 9 K at n = 3, convention named). K\* = 32 with no crossover; **the rule is not resolvable at n = 3** (the sd exceeds 0.02 at 6 of 9 K); the U-shaped minimum at K = 24 is resolvable; **K = 20 fails the strict gate** (across-wells 0.0139, seed 1). "+0.012731 at K = 10" is scoped as a seed-0 value. The w12 non-deranged control is declared | `w15-crossover-seeds/CROSSOVER-3SEED.md`, `FINDINGS.md` |
| P10 | rival retrain cost, seeds 1–2 | new `tab:rivals-del-seeds`: seconds per seed, 15,032 rows, rival after retrain, and paired store − rival. Caveats: no 1× row; 16,384-row rival pool | `w15-single-design-seeds/T7-3SEEDS.md`, `FINDINGS.md` |
| P12 | harness smokes | CMAZE-T: replay of 164,767 transitions with 0 disagreements; 0 impossible; VAL smoke 0.6911 n-way; the 2-way across-wells miss (0.2535 vs 0.2432) makes that row interface-only; **all three shipped warnings named**. CMAZE-O: 105,408 states replayed with 0 disagreements; 30,588 requests with 0 impossible; the controls do not collapse (interface; 99.24 % of the floats). **The N350 control-defect paragraph is here** | `w15-cmaze-t-harness/FINDINGS.md`, `HARNESS-READY.json`; `w15-cmaze-o-harness/FINDINGS.md`; `w16-maze-wm/FINDINGS.md` |
| P13–P20 | `tab:cmazet` (CMAZE-T ONLY) | seen (sanity, no claim) · 2-way (declared negative, **interface**) · **n-way STORE row** · C-512 replication with its seed-1 caveat · the 80k-step budget diagnostic · rollout (**declared negative, read**, with the 10× diagnostic qualifier) · value (declared negative, **content**) · controls, floors and planner columns · deletion in both scopes · ledger with its convention. Rivals' match residuals (+4.3 / −6.4 / +11.9 %) are printed | `w16-maze-wm/TABLES.md`, `FINDINGS.md` |
| P21–P27 | `tab:cmazeo` (CMAZE-O ONLY) | caption: **"The rival margin is WITHDRAWN BY THE HUB; the store gate holds."** The n-way row prints the withdrawal, not the margin. The store-gate row (4 controls at 0.0000) is separate. The atom-row identity is labelled "printed, never evidence". The value row is judge integrity. Deletion is in both scopes and on both sides. Ledger. The caption is corrected: **no autoencoder in the read path**, and rivals regress the identical field | `w16-maze-wm/TABLES.md`, `FINDINGS.md`, `MAZE-WM.json` |
| P28 | configuration ladder and budget | CMAZE-T sweep 25/53 → A1; C-512; Config B failed VB3 (+0.20 vs +0.40); CMAZE-O A (VB0), C-field, C2 (VB1). Budget: 9 of 18 spent, 9 declared not-run, each with its reason | `w16-maze-wm/FINDINGS.md`, `TABLES.md` |
| P29 | fallback | **declared not-run** (runs 13–18): Task 1 won, and the gate opened only on "no win" | `w16-maze-wm/FINDINGS.md` |
| P30 | declared negatives | 7 negatives, each with its link. **CMAZE-O has three links, not one** (§4b, disagreement 3) | `w16-maze-wm/FINDINGS.md` |

TABLES.md §4b was followed throughout, and the artefact wins every disagreement: the C2 triples are **not** attached to C-field; the CMAZE-T deletion prints the S_k residual (not "0.0"); the 80k diagnostic is labelled **A1, VAL**; the CMAZE-O deletion **ran**.

## 4. The five things that had to go in, and where they are

1. **CMAZE-T A1 STORE row** (0.7246 vs 0.4035; four controls collapse; survives rivals trained 10× longer): main text §4.4, paragraph 1; `tab:cmazet` n-way, replication and budget-diagnostic rows; the abstract; contribution 3; `fig_maze_wm` (a).
2. **CMAZE-O withdrawn margin vs the holding store gate, printed as two statements**: §4.4 paragraph 3 ("Two statements must be kept apart…"); the `tab:cmazeo` caption, n-way row and controls row; the P30 negatives.
3. **Rollout validity 0.000 for the CLU and every matched rival** (per-step 0.925 vs 0.818, over 43 steps). The 10×-float diagnostic transformer is non-zero on seed 2 (0.022), and that qualifier travels with the row. Link **READ**: §4.4 paragraph 2; the `tab:cmazet` rollout row; `fig_maze_wm` (b), which labels the diagnostic arm "(diagnostic, not matched)".
4. **coverage-control**: coverage, by construction; the TEST cell DERIVED; VAL "NO RISE". This is in §3.2 and in a `tab:ladder` row, with the artefact caveat of §2.
5. **Deletion is not capability erasure** (20 deleted corners still compose red on 3 seeds): the new paragraph in §3.3, worded as "store-scoped exactness plus a data-free adaptation cost".

**N350 correction.** It is in the harness-smokes paragraph (app: maze world-model rows): `rng.permutation(K)` is the identity half the time at K = 2; it **was** the identity on w16 Config-A VAL seeds 0 and 1 (0.8019, 0.5689); the derangement fix (0.2504, was 0.489); every scored across-wells cell uses the fix. **The w15 smoke numbers were NOT re-measured, and the 0.2535 carries the caveat by name.** The P28 ladder paragraph also points to it. The crossover sweep's own non-deranged across-wells control (w12's identity-rejected permutation) is declared separately in `app:kcurve`.

## 5. Edits beyond the placeholders (flagged for the Head; every one is identical in both drafts)

- **The abstract, contribution 3 and the "Thickness" paragraph** said composition does not transfer through the store on the maze. With the CMAZE-T store row that sentence is false, so I scoped it: "composition **on routes** does not transfer…". I added one sentence to the abstract (0.7246 vs 0.4035; no valid chained rollout), one to contribution 3 (the CMAZE-O gate held, the margin withdrawn), and one to Thickness. ⚠ **This is a framing change. The Head should read it.**
- §4.4 is rewritten as results, and "(pending)" is dropped from its heading and from the appendix heading. The §4 environment list no longer says "pending". The Placeholder-index appendix is rewritten.
- Figure captions: `fig:verb` (the "Placeholder figure." marker is removed) and `fig:maze-wm` (new). `fig:crossover` and `fig:deletion` now say that they are placeholder figures and point to the n = 3 tables.
- The provenance header comment was edited in place with the same line count, so the preamble offset of the checker holds.
- **Main text length:** §6 (Follow-up) moves from p13 to p14. The main text grows by about one page (the CMAZE-2b table went to the appendix to limit this). The ≤12 pp bound was already exceeded before this pass. Trimming belongs to the Head.

## 6. Figures

- **`fig_maze_wm` is NEW** (CMAZE-T only), built from `w16-maze-wm/t1/A1_test_seed{0,1,2}.json`: (a) n-way per arm, (b) rollout validity beside per-step exactness, (c) post-deletion composition. The script asserts that downdate equals fitted-without.
- **`fig_maze_readverb` and `fig_maze_requestlayer`** were rebuilt by the unchanged w17 code with the red PLACEHOLDER tag switched off. Their data is final w14 data; the w16 rows live in their own figure and tables.
- Build: `w18_placeholder_fill/build_maze_figures.sh` builds twice, **byte-identical** (`figures/SHA256-maze-w18-build{1,2}.txt`), and installs **only** the `fig_maze_*` files and `PROVENANCE-maze-w18.json`. The replaced PNG and PDF files are kept in `figures-backup-w17/`. `figures/rendered/` and the non-maze figures were not touched. The w17 `PROVENANCE.json` and `FIGURES.md` were **not** edited, so their maze entries are stale. The w18 provenance is `PROVENANCE-maze-w18.json`.

## 7. DECLARED NOT-DONE (and why)

- **The 4 awaiting placeholders**: their spokes are running on the other box.
- **The coverage-control per-seed numbers**: the artefact is not on batmac (§2).
- **Regenerating `fig_crossover`, `fig_deletion` and `fig_ladder`** from the w15 data: these are not maze figures, which is outside this pass's figure scope. They stay marked placeholders, and their captions now point to the n = 3 tables. `fig_ladder` also awaits LACE/PoE.
- **MF2** (23.5117 → 23.5121, L1317/L2262) and **MF14(i)** (BLIND columns into `tab:cmaze2`): these are not placeholders, and TABLES.md §6.2e assigns them to the Head or to this spoke. Left for the Head, untouched.
- **The §8 caption-convention patch** (`caption_conventions/`): not applied. The new tables name their own conventions.
- **The label collision** `tab:cmaze-del` on the CMAZE-T table was kept as it was, because it is referenced. The CMAZE-2b deletion is under `tab:maze-del-2b` only.
- **Visual review**: I checked rendered PNGs of pages 7, 12, 36 and 37 only (Ghostscript at 80–90 dpi). There was no full proof-read of the 39 pages.
- The rebuttal kit is on HARD HOLD and was not touched. I made no evidence edits and no registry or matrix edits.

## 8. Files

`LOG.md` · `START_UTC.txt` / `END_UTC.txt` · `pre/` (snapshots, sha256, pre-edit checks) · `build*/` and `final-compile-*/` (builds; nothing deleted) · `STYLE-DIFF-CHECK-v2.md` · `compile-v2.log`, `compile-v2-style.log`, PDFs · `SHA256-post.txt` · `figures-backup-w17/` · `w18-placeholder-fill.bundle`. The scripts are in the branch: `w18_placeholder_fill/{fill_text.py, apply_fill.py, run_all.sh, make_maze_figures.py, w17_make_figures.py (vendored unchanged), build_maze_figures.sh}`. `run_all.sh <fresh dir>` rebuilds both drafts from `pre/` and re-runs every check.
