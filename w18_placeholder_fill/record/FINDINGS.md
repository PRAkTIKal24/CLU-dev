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
