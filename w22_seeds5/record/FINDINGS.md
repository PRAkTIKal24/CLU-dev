# w22 spoke/seeds-and-tables: FINDINGS (FIFTH PASS)

**Branch:** `w22/seeds5`, cut from `w18/seeds-and-tables`. **Machine:** batmac. **Scored runs: 0. Compute: 0**, apart from LaTeX compiles. No ledger write. No PREREG.
**Both drafts were edited and installed**: `DRAFT-v2.tex` e27a0009 → **6cecd36a**, and `DRAFT-v2-style.tex` 848f91d1 → **72327303**.
Every number in this file is either quoted from `TABLES.md` (the section is named) or carries its own `path:line`. `TABLES.md` is in `outputs/w18-seeds-and-tables/`, which is the register this pass extends.

## 0. The one that mattered: the w15 fold is CLOSED (`TABLES.md` §13)

Passes 3 and 4 both declared the `bar-d-score` / `coverage-control` / `lace-poe-rows` fold out of brief. **This pass closes it.**

**The check.** Each draft prints **199 number tokens** under a `% SRC` or `% SRCBLOCK` that names one of those three spokes. The scanner is `w22_seeds5/scan_draft.py`: decimals, signed decimals, comma-grouped integers, 4+-digit integers, and `a of b` counts. Each token is now three registered cells, re-checked on every build:
1. It is **verbatim on its line** in the installed draft, both drafts, through pinned copies in `draft_pass5/`.
2. It is **verbatim in the `FINDINGS.md` the draft cites**: 197/199. The other 2 are the corrected value, below.
3. It **is a named JSON leaf** at the printed precision: exact, or half-up / half-even from the float's repr. **199/199** after the correction. The sources are `COVERAGE-CONTROL.json`, `GENTABLE-KSTAR.json`, `BAR-D.json`, `CROSS-ARM-GPU-test.json`, `READ-TABLE.json`, and `IDENTITY-CHECK.txt` for three text-only facts. The 4 counts (`109/109`, `60 of 60`, `120 of 120`) are checked against every `corners_control_wins` leaf.

There are also **8 extra checks** for numbers the scanner cannot see: the PoE row-3 zeros, `0.020 / 0 / 0`, "5 of 6" (the six A″ CI lower bounds) and "+32".

**⛔ The one disagreement. The artefact wins.**
- The drafts printed PoE's budget as **27.13×**, twice per draft, copying `w15-lace-poe-rows/FINDINGS.md`.
- `GENTABLE-KSTAR.json` `/PART_B_rows_3_4/arms/PoE/budget_per_seed/0/x_store_reference` = **27.135**, and 14,226,660 / 524,288 = 27.1350. To two decimals that is 27.14. FINDINGS truncated where it should have rounded.
- **Both drafts now print 27.135×**, and each line's SRC also names the JSON (edit E7).
- A negative test proves that reverting to 27.13 now fails the build.

**Also checked.** The `alone` probe margins (−0.0505 / +0.0242 / −0.0862) in the bar-d paragraph are w8's numbers, quoted by w15. They are verified against the JSON that w15's FINDINGS names: `w8-encoder-loop/cloud/manifests/seed_{s}/alone/probe.json` `/MARGIN_OVER_RANDOM_INIT/aggregate`.

**Nothing in the three spokes is left unfolded.**

## 1. w20 + w21 folded (`TABLES.md` §14, §15)

- **w20 `shapes-data-matched`** (§14):
  - The headline is a JSON cell: **+7.03** (+7.49 / +8.60 / +5.02), and +6.54 against the best of every fit.
  - P1 convergence, the M-131k arms, the four-measurement deletion rows (all 3 scopes × 3 seeds), rival retrain-without, and the ladder margins are copied row for row from `SHAPES-MATCHED.md`.
  - ⛔ **w20's own `FINDINGS.md` disagrees with its artefact.** It says "the rivals lose < 1 novel/corner when retrained without red∧sphere". `SHAPES-MATCHED.md` gives MLP seed 2 at 43.174 → 40.220 and continuous MLP seed 2 at 26.844 → 17.275 (§14.4). The drafts already print each rival's before → after (w22 draft-rulings found the same).
  - A labelling nuance, not a number: for seed 2, `SHAPES-MATCHED.md` names `mlp__tenx` as the strongest rival (the max of DECLARED and TENX, 44.101). The JSON's `per_family.mlp.2.carried_rival` is `mlp__declared` (the P1-carried fit). The margin, +5.018, is the same in both.
- **w21 `words-full-test`** (§15.1): TEST **LOSS**, Δ̄ **−0.0175**, 0/3 seeds ahead. The label is a JSON cell from `BARS-WORDS-FULL-test.json`.
- **w21 `maze-t-seed4`** (§15.2):
  - The six-seed table.
  - The **per-seed ceiling row**: 0.830 / 0.567 / 0.808 / 0.714 / 0.535 / 0.939.
  - The cut, **0.6407**, gives easy {0,2,3,5} and hard {1,4}, with the full cut-sensitivity table.
  - CLU − best matched rival: +0.173 / −0.001 / +0.023 / +0.262 / +0.001 / −0.185, mean +0.046.
  - The per-seed A1 CLU values are JSON cells from all six run JSONs.
- **w21 `maze-t-clu-improve` + `rel-newseeds`** (§15.3):
  - CLU-REL-A10 six-seed mean **0.925** against A1 **0.677**. The best matched rival is 0.724, and the `+rel` diagnostic is **0.894**, printed beside them, with the per-seed lift and share.
  - The per-seed REL CLU values are JSON cells from all six run JSONs.
- **⛔⛔ The two ties are CELLS** (§15.4):
  - **Seed 4.** `control: random_atoms` == CLU == 0.9004149377593361 on n-way. This is a JSON equality on `BOTH_test_seed4.json`. The per-tuple counts are **identical on all 16 held-out tuples**: a JSON equality on the replay's `PER_TUPLE_NWAY`, asserted to hold 16 items, with 24 objects hash-checked.
  - **Seed 5.** `rival@10x-steps: transformer` == CLU == 0.9393939393939394. This is a JSON equality on `BOTH_test_seed5.json`.
  - The TABLES rows and the lines where both drafts print the ties are cells as well.
  - Breaking either equality fails the build (negative tests).
- **w21 `ladder-rivals-32k`** (§15.5):
  - All three rival sets are printed, with pool AND steps. The per-rung margins are JSON cells.
  - Mean over rungs: **A +17.53 · B +9.97 · C +10.28 · C-sm +10.75**. These means exist only in the unit's FINDINGS prose. I re-derived each from the five JSON rung means (LOG), and they match. The lowest CI lower bound is **+1.98**.
  - The X1 single-fit variance table is printed.

## 2. The drafts (`w22_seeds5/apply_pass5_edits.py`; guarded, idempotent, the same text in both drafts)

| edit | what |
|---|---|
| E1 | The `tab:ladder` caption names every set's **pool and steps**. The "a third set … is pending" sentence becomes the description of the store-content set |
| E2 | Block headers: **A** 32,256 rows / 5,040 steps · **B** 131,072 rows / 20,480, and 204,800 for MLP and continuous MLP · **C** 32,768 rows / 5,120 / 51,200 (+ step-matched) |
| E3 | ⭐ **The 32,768-row `\PLACEHOLDER` is FILLED.** Eight family rows (5,120 and 51,200 steps, per seed, per rung), then store − strongest (paired) with its mean, then store − strongest step-matched with its mean. The rows are copied from `w21-ladder-rivals-32k/TABLES.md` (P1, L3) by the script, not retyped. They print at 2 dp, which is the precision of that record |
| E4 | The ladder paragraph gets the set-C reading, the four over-rung means, the lowest bound (+1.98), MLP flat from 51,200 to 204,800 steps, and continuous MLP still gaining. It also gets **the X1 caveat**: two identically configured 2,560-step MLP fits differ per seed by −2.15 … +4.13, and the CI excludes 0 on 12 of 15 legs |
| E5 | The placeholder index: **three remain, and they are PERMANENTLY BLOCKED**, because w16 persisted no arm object and no per-transition prediction. The set-C rows are listed among the filled rows |
| E6 | Declared not-runs: the third set is no longer "pending". LACE/PoE at 32,768 / 131,072 on the rungs, and continuous MLP past 204,800 steps, are declared |
| E7 | PoE 27.13× → **27.135×** (§0) |

**Checks.** The edits were staged, compiled and checked, then installed.
- **All four style diffs are EMPTY**: numbers, placeholders, artefact paths, section order (`STYLE-DIFF-CHECK-installed.md`).
- The **provenance checker's findings are identical to the baseline** on both drafts. It checked 3,037 → 3,236 tokens (`prov-*`).
- **Compile**, built in place from `w17-paper/` with `BIBINPUTS`: both drafts give exit 0, **0 undefined citations**, **0 undefined references**, 0 errors, 0 overfull boxes, 0 "Figure pending", 55 bibitems, **50 / 50 pages** (was 49 / 50) (`installed-compile-*.log`).
- **Install.** The live drafts were `cmp`-equal to `snapshot/` right before the install. The lock was re-checked at 06:37:15Z: `tmux` showed authtest, hubhost and this session only (registry6 had finished), and `ps` showed no draft / figure / placeholder / citation / referee session. After the install, the drafts are `cmp`-equal to `staged-1/` and to the pinned `draft_pass5/` copies.

**Final counts, both drafts.**
- **`\PLACEHOLDER` 3** (was 4). All three are the w16 CMAZE-T per-arm panels (`fig:cmazet-families`, `fig:cmazet-door-tint`, `fig:cmazet-deletion`), and all three are **permanently blocked**. w16 persisted no arm objects: `outputs/w19-success-failure-panels/LOG.md:2`, "DECLARED NOT BUILDABLE". ⚠ The brief says two. The drafts carry three of this kind. All three share one cause, and all three are reported.
- **`\TODO` 4** (unchanged):
  - the Head's abstract CMAZE-T sentence;
  - the `pca_tensor` byte ratio (no artefact);
  - the AI-use statement;
  - the code archive.

  None of them is a table cell.

## 3. The register (`w18_seeds_and_tables/`)

- **Pass-4 draft cells were broken by other spokes.** w22 draft-rulings and figures3 rewrote the live drafts after pass 4. The 15 pass-4 cells that pointed at the live files failed. The pass-4 text was reconstructed **exactly** (the pass-3 copy + `draft_pass4/*.pass4.diff` → md5 6caf80cd / 5a58d5d9, as pass 4 recorded), and those cells are pinned to it. The new pass-5 draft cells point to pinned copies of the installed bytes, so the next edit cannot silently break them.
- **Two new cell kinds** in `build_tables.py`:
  - **JSON cells**: the printed value must BE the leaf at the printed precision; a string must be equal.
  - **JSON equalities**: two leaves must be equal, optionally holding n items.

  The register is self-contained (`jnum.py` is copied in).
- **Build:** **2,282 cells** verbatim-verified (1,512 new), **299 JSON cells**, **3 JSON equalities**, 2,483 used, 29 historical, **0 unsourced numbers**. **Two builds are byte-identical**, and so is a build from the `outputs/` copy. The sha256 values are in `SHA256-TABLES.txt`.
- **Negative tests, 12/12 exit 1** (`NEGATIVE-TESTS.txt`, `w22_seeds5/negative_tests.sh`): a mutated T7 cell, a pass-4 draft cell, a w15 draft cell, a w15 FINDINGS cell, **PoE reverted to 27.13**, the w20 headline, **either tie broken**, an orphan value-only token, a value-only JSON token, a bare number, and a tampered historical cell.
  - ⚠ On its first run the historical-tamper test "passed" by crashing: the output dir was missing, and the tampered cell was not one the build reads from the manifest. Both were fixed, and a Traceback no longer counts as a catch.

## 4. What did NOT run / not done

- **No scored run, no compute, no ledger write, no PREREG.** No w15/w20/w21 number was re-derived. The one piece of arithmetic is the four over-rung means, checked against the JSON (LOG).
- **The three w16 CMAZE-T per-arm placeholders are not filled, and cannot be**: the objects never existed. `NEEDS-ARTEFACTS.json` records that no fetchable artefact would change this.
- **Not edited:**
  - the other placeholders and TODOs;
  - the draft header comment block;
  - `CORRECTIONS-ABSORBED.tex`;
  - any figure. ⚠ `fig_ladder` panel (b) still draws the 2,560-step 32,768-row MLP, which is the figures owner's job (w22 draft-rulings flagged it).
- ⚠ **Pre-existing layout, not changed:** `tab:ladder` renders at a small adjustbox scale, because its data cells set its width. It was the same before this pass (`pages/p8-live.png` against `pages/p8-ladder.png`). It is 8 rows taller now and still fits on page 8.
- ⚠ The main text is still over ICLR length (MF-11, the Head's).
- **Background jobs: none. Sub-agents: none.** Nothing was killed, and `tmux kill-server` was not used.
