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
