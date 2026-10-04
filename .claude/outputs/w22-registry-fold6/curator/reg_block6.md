
## ◈.18 WHAT CHANGED SINCE THE FIFTH PASS, AND WHAT THIS PASS READ

**Commissioned by** the w22 `registry6` brief (`~/clu/tasks/w22/registry6.md`, 2026-09-25; pre-authorised in full). **Base:** this file's own pass 5
(`N372`–`N390`, `◈.15`–`◈.17`; 8,042 lines). ⛔ Pass-1 to pass-5 entries are **not edited**. The only changes to them are **inserted** `⟲ PASS 6` lines.
Nothing was replaced, deleted or renumbered. This block adds **`N391`–`N397`**.

**Why a sixth pass.** Pass 5 ended `2026-09-24T15:32:32Z`. Three units closed after it and were not folded:
- `w21/rel-newseeds` (START `2026-09-24T15:24:11Z`, END `2026-09-24T16:16:03Z`);
- `w21/ladder-rivals-32k` (START `2026-09-24T13:39:40Z`, END `2026-09-24T22:03:55Z`);
- `w22/figures3` (START `2026-09-24T21:44:48Z`, END `2026-09-24T22:13:46Z`).

⛔⛔ **Standing rule (Head, 2026-09-24): pre-registrations are banned.** Report everything, omit nothing. A pre-declared bar found in this evidence (the
ladder's X1 bar, LOG 13:58Z) is quoted **as the record of what the spoke declared**, never as authority over a measured number. The curator writes no bar,
label or prediction of its own.

⭐ **The rule this pass applies to its own brief: a paraphrase is not a source** (`N351`, `N371`, `N381`). Every number below is read from a JSON or a
generated TABLES file. Where the brief or a report's prose disagrees with the artefact, **the artefact wins**, and the disagreement is itself folded (`N396`).

**What this pass read** (unprefixed paths are relative to `~/clu/evidence/outputs/`):
- **`w21-rel-newseeds/`**: `FINDINGS.md` and `LOG.md` in full; `TABLES.md` §0–§4, §8, §9; `REL-NEWSEEDS.json` (`nway_summary`,
  `rival_nonconvergence_flags`, `class_from_maze_t_seed4`); `replay/BOTH_test_seed{0..5}_PERTUPLE.json` (`PER_TUPLE_NWAY`, per-tuple control equality
  **counted by the curator**).
- **`w21-ladder-rivals-32k/`** (876 files): `FINDINGS.md`, `LOG.md`, `TABLES.md` (L3, F, P1, X1, X2), `CAPTION.md`, `checks/pool_identity.json`;
  **numbers from `LADDER-RIVALS-32K.json`** → `rungs.<n>.{crosscheck.X1_same_rows_same_steps, crosscheck.X1_diagnostic_ref_recipe_replay,
  margins_K32_store_minus_rival.*}`.
- **`w22/figures3`**: the staged `w22-figures3-FINDINGS.md` (identical to the branch copy). The JSONs are **not** staged in `evidence/`. They were read
  from the branch **`w22/figures3` @ `cb5e0ac`** (`w22_figures3/{AMBIGUITY.json, AMBIGUITY-before.json, GLYPH-IDENTITY.json, PALETTE-MAZE_RGB-before.txt,
  PALETTE-MAZE_RGB.txt, LOG.md, gate2/cmazet_items_rel.REPRO.json}`), copied read-only to `~/clu/outputs/w22-registry-fold6/curator/src_w22_figures3/`,
  and md5-checked against `~/clu/outputs/w22-figures3/`. The page crop `~/clu/outputs/w22-figures3/pages/p46-rule_a-crop2.png` was viewed.
  `FIGURE-DEFECT-cmazet-legality.md` (the Hub's defect note) was read.
- **Re-read for comparison:** `w20-shapes-data-matched/SHAPES-MATCHED.json` → `headline` (the +7.03 the X1 result bears on).
- **Curator arithmetic** (the only derived numbers in this block) comes from a script: `~/clu/outputs/w22-registry-fold6/curator/curator_arithmetic_p6.py` →
  `curator_arithmetic_p6.json` (cited as **CA6**). It reads evidence only and computes means, differences and counts.

⛔ **Rules this pass kept:** ZERO scored runs, ZERO compute beyond the CA6 script. No ledger written. No draft or `.tex` read for editing or edited. The
`.tex` lock was not taken. `~/clu/evidence/` was not written. ⚠ Every `◈` tier is **CURATOR-PROPOSED**.

## ◈.19 ⭐ THE ENTRIES — `N391`–`N397`

**Reading key (CMAZE-T).** Unseen n-way, `exact`, condition BOTH, **TEST**, seeds 0–5 (seeds 0–2 = the peer's `maze-t-clu-improve` runs, seeds 3–5 =
`rel-newseeds` runs 1–3/3). **CLU-REL-A10** = the peer's frozen config (K 2, 10 atoms, ridge 1, 9,200 read floats); **A1** = the w19 config (4,584
read floats). **Ceiling** = the best trained arm (`w21_tables.CEIL_PREFIX`: `+rel` diagnostics and planners excluded). **`+rel`** = a rival handed the
CLU's colour-relational columns: a diagnostic, never a rival row.

### N391 — ⭐⭐ **CLU-REL-A10 ON ALL SIX SEEDS, BOTH TEST: 0.925 AGAINST A1's 0.677. It reaches the trained-arm ceiling on every seed and lifts A1's two lowest seeds, 4 (0.533 → 0.900) and 1 (0.566 → 0.894). Its margin over the best matched frozen-input rival is +0.202 on average, and it does not rest on a rival that failed to train: no WORSE@80k on seeds 3–5, and the best rival there is unflagged. ⛔ Read with `N392`: most of that margin is the input coding, not the store.** · tier A (CURATOR-PROPOSED ◈) · memory-architecture / benchmark design (w21 spoke `rel-newseeds`, scored runs `maze-t-rel-newseeds` 1–3/3, seeds 3 / 4 / 5; seeds 0–2 re-read from `maze-t-clu-improve`; no PREREG, no label)

**Source:** `w21-rel-newseeds/REL-NEWSEEDS.json` → `nway_summary`, `rival_nonconvergence_flags`; `TABLES.md` §0, §4, §8; `FINDINGS.md`; CA6 `rel_newseeds`.

| | s0 | s1 | s2 | s3 | s4 | s5 | mean (6) |
|---|---|---|---|---|---|---|---|
| A1 CLU | 0.800 | 0.566 | 0.808 | 0.674 | 0.533 | 0.682 | 0.677 |
| **CLU-REL-A10** | 0.908 | 0.894 | 0.984 | **0.927** | **0.900** | **0.939** | **0.925** |
| REL best matched rival (frozen input) | 0.803 | 0.567 | 0.685 | 0.723 | 0.624 | **0.939** | 0.724 |
| REL CLU − best matched rival | +0.105 | +0.326 | +0.299 | +0.204 | +0.277 | **+0.000** | +0.202 |
| REL CLU − A1 CLU | +0.108 | +0.327 | +0.176 | +0.253 | **+0.368** | +0.258 | — |

- **Exact values** (`nway_summary`): REL CLU mean **0.925454677527981**, A1 **0.6770460727620738** (CA6). Seed 4 = 0.9004149377593361 (651 of 723
  rows, CA6 `pertuple_controls`).
- ⭐ **Ceiling:** the REL CLU equals the trained-arm ceiling on all six seeds (CA6 `CLU_at_ceiling_all6` = true). On seed 4 it **shares** it with
  `control: random_atoms`; on seed 5 with `rival@10x-steps: transformer` and the 10×-read-floats transformer (`N392`). ⚠ **On seed 0 the ceiling that
  includes `+rel` is 0.915, above the CLU's 0.908** (`diagnostic+rel@80k: mlp +rel`).
- ⭐ **Where A1 was weak** (A1 hard seeds by measured ceiling, maze-t-seed4: {1, 4}): seed 1 0.566 → 0.894, **seed 4 0.533 → 0.900**. A1's losing seed,
  seed 5 (A1 −0.185 against the matched mlp-80k, `N384`), goes 0.682 → 0.939, and there the REL CLU **ties** its best rival.
- ⭐ **Convergence** (`TABLES.md` §4; HIGH = final loss > 1e-2, WORSE@80k = the 80k run ends at a loss ≥ its 8k run): **no WORSE@80k on seeds 3–5**.
  The best matched rival on seeds 3, 4 and 5 is the transformer, with none of its losses flagged. The "best UNFLAGGED matched rival" margins equal the
  margins above. ⚠ **HIGH flags printed, not netted:** s4 lstm-8k (0.027), s4 mlp-8k (0.0126), s4 mlp+rel-8k (0.0128), s5 lstm-8k (0.053),
  s5 lstm+rel-8k (0.052). ⚠ Seed 2 (peer run): matched mlp HIGH + WORSE@80k (0.0128 → 0.0670), the same failure as A1's seed-3 mlp (`N387`).
- ⭐ **Integrity:** the config sha256 `0d4ec5b9…` is asserted in every run JSON. The zero-fit replay hash-matches **24 objects per seed on all six
  seeds**, and all 32 n-way arm numbers reproduce bit-exact (`TABLES.md` §8).
- **Other rows** (`FINDINGS.md`; CLU / best matched rival, s3/s4/s5): seen 1.000/1.000, 0.999/0.999, 1.000/1.000. Pair 0.937/0.620, 0.872/0.515,
  0.837/0.244. ⛔ **Value 0.525/0.822, 0.692/0.748, 0.604/0.831: the CLU is BELOW the best matched rival on the value row on all six seeds**
  (s0–2: 0.489/0.682, 0.501/0.861, 0.512/0.839). Deletion is store-scoped exact on seeds 3–5 (max rel ΔS ≤ 1.94e-15). The CLU adapts in 0.06–0.17 s;
  the rival's retrain-without takes 23–26 s. ⛔ **Rollout validity 0.267 / 0.156 / 0.000** (per-step 0.970 / 0.958 / 0.962).
- **Where the REL CLU misses** (`TABLES.md` §7, §9): on seeds 3–5 the misses are exactly the all-equal-colour tuples (pocket = door = tint). On seeds 1
  and 5 the CLU equals the relation-unreachable bound exactly (seed 5: 1 − 40/660 = 0.9394). ⚠ **Seed 3:** on (2,2,2,0) / (2,2,2,3) / (3,3,3,0) the
  frozen-input matched transformer (8k) scores 0.889 / 1.000 / 1.000 and the CLU scores 0.000.
- **Matrix row:** `CM-W21-4`. ⟲ at `N383`, `N384`, `N388`.

### N392 — ⛔⛔ **THREE MEASUREMENTS SAY THE STORE IS NOT WHAT CARRIES CLU-REL-A10's MARGIN. (1) Handing the rivals the relational columns leaves the CLU +0.031 on average (−0.007 / +0.024 / +0.165 / +0.006 / +0.000 / +0.000). (2) On seed 4 TEST, `control: random_atoms` = CLU = 0.9004, and the two match on all 16 held-out tuples. (3) On seed 5, CLU 0.939 = best matched rival 0.939. ⭐ The claim that survives is colour-relational request coding: an INPUT REPRESENTATION. ⛔ It is NOT a store/memory result. ⚠ Seed 2 is the exception: there the CLU keeps +0.165 beyond the coding.** · tier A (CURATOR-PROPOSED ◈) · memory-architecture / claims hygiene / control integrity (w21 `rel-newseeds`; ⭐ **the question `N383` left open on three seeds, answered on six**)

**Source:** `w21-rel-newseeds/TABLES.md` §2 (row `nway`), §3; `REL-NEWSEEDS.json` → `nway_summary`; `replay/BOTH_test_seed{0..5}_PERTUPLE.json`;
`FINDINGS.md` items 1–3; CA6 `rel_newseeds.{CLU_minus_best_rel, rel_lift, share_recovered, pertuple_controls}`.

**(1) The `+rel` diagnostic, n-way, per seed** (`TABLES.md` §2):

| | s0 | s1 | s2 | s3 | s4 | s5 | mean (6) |
|---|---|---|---|---|---|---|---|
| best `+rel` diagnostic | 0.915 | 0.869 | 0.819 | 0.921 | 0.900 | 0.939 | 0.894 |
| `+rel` lift = best `+rel` − best matched rival | +0.112 | +0.302 | +0.134 | +0.199 | +0.277 | +0.000 | +0.170 |
| **CLU − best `+rel`** | **−0.007** | +0.024 | **+0.165** | +0.006 | +0.000 | +0.000 | **+0.031** |
| share of the CLU's margin the lift recovers | 1.06 | 0.93 | **0.45** | 0.97 | 1.00 | n/a (margin 0) | — |

- Exact six-seed mean of CLU − best `+rel`: **0.031389569311635133** (CA6). ⭐ On seeds 0, 3, 4 and 5 the best `+rel` rival comes within 0.007 of the CLU,
  and on seed 0 it is above it. The single fixed arm the peer quoted (transformer `+rel` @80k) reads 0.908 / 0.869 / 0.809 / 0.905 / 0.900 / 0.939, a
  six-seed mean of **0.889** (the peer's three-seed mean was 0.862).
- ⚠ **The exception, printed:** on **seed 2** the CLU keeps **+0.165** over the best `+rel` rival (share recovered 0.45). On seed 1 it keeps +0.024.
  ⇒ "Most of the margin is the coding" holds at the six-seed mean and on four of six seeds. It is **not** zero everywhere.

**(2) The control tie on seed 4 TEST repeats the VAL tie** (`TABLES.md` §3; CA6 `pertuple_controls`):
- n-way: CLU **0.9004**, `random_atoms` **0.9004** (**TIE/ABOVE**); VAL: 0.8994 = 0.8994 (`N386`). Per tuple, `random_atoms` equals the CLU on **16
  of 16** held-out tuples (651 exact rows each of 723; CA6). ⇒ On seed 4 a store built from random atoms does exactly what the learned store does.
- ⚠ `landscape_removed` reads **0.2656** on seed 4 TEST (0.2414 on VAL; `N386`). The A1 run reads the same 0.2656 on seed 4 TEST.
- ⚠ **Near-ties elsewhere, per tuple:** seed 3, `random_atoms` 0.698 and across-wells 0.695, each equal to the CLU on **15 of 18** tuples. Seed 5,
  across-wells 0.827, equal on **15 of 16**. Seed 0, across-wells 0.778. On the n-way score no control ties or beats the CLU on any seed other than 4.
  The across-wells derangement is [1, 0] on every seed.
- ⭐ **What the controls do show:** on seeds 0–3 and 5 the strongest control sits 0.112–0.507 below the CLU (CA6 `CLU_minus_max_control`). The CLU's
  **own** read uses its landscape. What `+rel` shows is that a **rival** given the same columns needs no such store to match it.

**(3) Seed 5 has no margin** (`TABLES.md` §0): CLU **0.939** = `rival@10x-steps: transformer` 0.939 = the 10×-floats transformer = all four `+rel`
arms. The CLU sits exactly at the relation-unreachable bound 1 − 40/660 (§9).

- ⭐ **The claim, as the artefacts support it:** *colour-relational request coding, an input representation, carried through the CLU's closed-form
  head, lifts CMAZE-T BOTH TEST from 0.677 to 0.925 on six seeds.* ⛔ **Not** *"the store composes where rivals cannot"*, and not a memory result.
  The rivals reach the same numbers once they are given the same columns (+0.031 residual), a random-atom store ties on seed 4, and seed 5 has no
  margin at all.
- ⛔ The curator computes no label (the spoke printed none).
- **Matrix row:** `CM-W21-4`. ⟲ at `N383`, `N386`.

**Reading key (ladder).** Shapes composition ladder, rungs n = 2–6 (n = 2: 60 corners; n = 3–6: the same 120 corners), 3 seeds, "novel/corner" as in
`N377`. **Store** = the w12b K\* = 32 ladder rows, content fitted on the rung's first **32,768** rows of a 131,072-row pool. **Set A** = spoke-A `@ref`
rivals (32,256 rows, 5,040 steps). **Set B** = w20 (131,072 rows; DECLARED 20,480 / TENX 204,800 steps). **Set C** = w21 (the store's own 32,768 rows;
DECLARED 5,120 / TENX 51,200 steps). **C-sm** = set C step-matched to B (mlp / cont_mlp 204,800; lstm / transformer 20,480). Margin = store − the strongest
rival per seed (max over families of max(DECLARED, TENX); w20's convention), paired per corner, bootstrap CI95. FREE replication: no ledger row.
(Source: `w21-ladder-rivals-32k/CAPTION.md`, `TABLES.md` L3.)

### N393 — ⭐⭐ **MF-2 IS REAL ON PAPER AND IMMATERIAL WHEN MEASURED. Rivals fitted on exactly the store's own 32,768 content rows leave the ladder margin where the 131,072-row rivals left it: mean over rungs A +17.53, B +9.97, C +10.28, C step-matched +10.75. Set C lands within −1.21 … +1.45 of set B per rung, in both directions, and every CI95 lower bound in B / C / C-sm is above 0 (lowest +1.98). ⭐⭐ This independently confirms w20: at matched steps the POOL barely moves these rivals; the STEP BUDGET does.** · tier A (CURATOR-PROPOSED ◈) · benchmark design / baselines (w21 spoke `ladder-rivals-32k`, FREE replication, 0 scored runs; ⭐ **MF-2 answered by measurement**)

**Source:** `w21-ladder-rivals-32k/LADDER-RIVALS-32K.json` → `rungs.<n>.margins_K32_store_minus_rival.{setA_vs_strongest,
setB_vs_strongest_declared_or_tenx_RECOMPUTED, setC_vs_strongest_declared_or_tenx, setC_vs_strongest_stepmatched, setC_vs_best_of_every_fit,
setB_integrity_vs_w20_printed}`; `TABLES.md` L3, F, P1; `FINDINGS.md` §0–§3, §5; `checks/pool_identity.json`; CA6 `ladder_margins`.

| rung | **A** (32,256 · 5,040) | **B** (131,072 · 20,480 / 204,800) | **C** (32,768 · 5,120 / 51,200) s0/s1/s2 [mean] | **C-sm** [mean] | C best-of-every-fit [mean] |
|---|---|---|---|---|---|
| n=2 | +10.76 | +5.13 | +6.65 / +7.37 / +4.22 [**+6.08**] | +5.87 | +5.57 |
| n=3 | +15.62 | +11.18 | +13.01 / +15.37 / +6.07 [**+11.48**] | +13.40 | +11.48 |
| n=4 | +20.09 | +12.55 | +16.14 / +16.73 / +9.13 [**+14.00**] | +14.54 | +13.20 |
| n=5 | +20.58 | +10.69 | +11.12 / +10.60 / +6.69 [**+9.47**] | +10.62 | +9.01 |
| n=6 | +20.61 | +10.30 | +11.30 / +12.49 / +7.29 [**+10.36**] | +9.30 | +9.30 |
| mean over rungs | **+17.53** | **+9.97** | **+10.28** | **+10.75** | 9.71111111111111 (CA6) |

- **C − B per rung** (CA6 `C_minus_B_per_rung`): +0.9444444444444446 / +0.29722222222221895 / +1.450000000000001 / **−1.2138888888888868** /
  +0.06111111111111356. **C-sm − B**: +0.7388888888888889 / +2.2194444444444414 / +1.9861111111111125 / −0.06666666666666643 / −1.0. ⇒ The brief's
  "±1.5" holds for set C; step-matched C sits up to +2.2194444444444414 above B at n = 3.
- ⭐ **Lowest CI lower bounds** (CA6): A 7.266666666666667, B 2.4581250000000003, C 2.85, **C-sm 1.9833333333333334** (n = 2 seed 2, against the
  step-matched mlp, Δ +3.25). Smallest per-seed margins: B +3.85, C 4.216666666666667, C-sm +3.25.
- ⭐ **Pool identity, from the artefacts, 15/15 legs:** `n_fit_rows = 32768` (w12b `LEG.json`) and `n_store_train = 32768` (`partitions.json`); the rung
  pool is identical to the w12 composition-ladder pool, and seed-independent (`checks/pool_identity.json`). Every set-C fit read `pool[:32768]`, which
  holds the store's content rows **and** `learned_ref_ownpool`'s rows. Set B recomputed from w20's score files **equals w20's printed ladder P3 on 15/15
  legs** (`setB_integrity_vs_w20_printed.identical`).
- ⭐⭐ **Data at matched steps** (`TABLES.md` F; `FINDINGS.md` §3): four times the rows at the same steps moves every family by **−3.5 … +3.9, with no
  consistent sign**. mlp @204,800: 26.77 / 27.51 (n = 2) … 57.30 / 55.47 (n = 6), 32k / 131k. Doubling `learned_ref_ownpool`'s steps (2,560 → 5,120)
  more than triples its score (×3.2 – ×3.5). This is the ladder version of w20's rows 3–4 finding (`N373`).
- ⚠ **Convergence, printed not netted** (`TABLES.md` P1; bar TENX − DECLARED > 1.0): P1 FAILS on mlp 14/15, lstm 10/15, transformer 5/15, cont_mlp
  15/15 legs. The mlp is **flat from 51,200 to 204,800 steps** (−4.76 … +2.02). ⛔ **cont_mlp is not converged at 204,800** (step-matched − TENX up to
  **+19.33**), but it never becomes the strongest rival (≤ 48.81 mean at n = 5, below the mlp on every rung). **All 60 selections chose lr = 3e-3 and 40
  epochs, the edge of both grids**, as at 131k in w20. The grids were not widened.
- ⛔ **Scope:** ladder rungs only. **No store was fitted or scored** (the K\* = 32 rows are w12b's; their controls are w20's). No rows 3–4 TEST. No
  LACE / PoE at 32,768 rows. Set B's lstm / transformer TENX exists only for rungs 2 and 6, seed 0 (1 seed). ⛔ The ladder shrink from set A to B / C is
  **real** and survives the data match.
- **Matrix row:** `CM-W21-5`. ⟲ at `N373`, `N377`, `N381` item 7.

### N394 — ⚠⚠⚠ **(applies to every single-fit margin in the paper) A SINGLE 2,560-STEP RIVAL FIT MOVES BY UP TO 4.13 NOVEL TUPLES PER CORNER ON THE PRNG DRAW ALONE. The ladder's X1 check held rows, steps, width, parameters, trainer and schedule identical and left only the init and shuffle keys different. The per-seed paired Δ runs from −2.15 to +4.13, and its per-corner CI95 excludes 0 on 12 of 15 legs. The headline Shapes margin is +7.03 (per seed +7.49 / +8.60 / +5.02). ⛔ ⇒ A per-corner bootstrap CI does not contain refit variance: a per-seed CI above 0 does not show that a refit would stay above 0 once the margin is within ≈ 4 of zero.** · tier A (CURATOR-PROPOSED ◈) · statistics / benchmark design / claims hygiene (w21 `ladder-rivals-32k`, `crosscheck.X1_*`; ⭐ **its own entry, by the brief**)

**Source:** `w21-ladder-rivals-32k/LADDER-RIVALS-32K.json` → `rungs.<n>.crosscheck.{X1_same_rows_same_steps.per_seed.<s>.paired_mine_minus_ref,
X1_diagnostic_ref_recipe_replay}`; `TABLES.md` X1, P1; `FINDINGS.md` §2–§3; `LOG.md` 13:58Z, 18:10Z, the CORRECTION line, 23:45Z;
`checks/x1_encoding_identity.json`; `w20-shapes-data-matched/SHAPES-MATCHED.json` → `headline`; CA6 `ladder_X1`.

**The two arms.** Both are mlp, width 61, 130,514 params, the rung's first 32,768 rows, 20 epochs, lr 1e-3, wd 0, batch 256, **2,560 steps**, so
`mine_steps = ref_steps = 2560` and `n_fit_rows = 32768` on every leg (CA6). The input encoding is identical element for element (a 32,768 × 57 one-hot,
checked at rungs 2 and 6), and the trainer is the same AdamW + warmup-cosine with the same per-epoch jax-key permutation. **What differs by
construction** (LOG 18:10Z): the init key (ref `PRNGKey(seed)`, mine `PRNGKey(seed*7717+SEL_RNG+crc(fam))`) and the shuffle key. ⭐ **The diagnostic
that pins the cause:** `learned_ref_ownpool`'s own recipe, replayed with **its own** keys, reproduces the existing rows **identically per corner on 15/15
legs** (0 of 60 / 0 of 120 corners differing; `X1_diagnostic_ref_recipe_replay`). ⇒ The only thing left between the arms is the PRNG draw.

| rung | s0 Δ [CI95] | s1 Δ [CI95] | s2 Δ [CI95] | Δ of 3-seed means |
|---|---|---|---|---|
| n=2 | +1.80 [+0.67, +2.92] | **−2.15** [−3.80, −0.58] | +1.53 [+0.17, +2.95] | +0.39 |
| n=3 | +0.41 [−0.93, +1.73] | −1.38 [−2.64, −0.16] | −1.54 [−3.06, +0.03] | −0.84 |
| n=4 | +3.34 [+1.95, +4.73] | −0.65 [−2.26, +0.97] | +3.52 [+1.58, +5.41] | +2.07 |
| n=5 | +3.98 [+2.48, +5.49] | −1.93 [−3.70, −0.15] | **+4.13** [+2.18, +6.04] | +2.06 |
| n=6 | +3.86 [+2.22, +5.47] | −2.08 [−3.85, −0.28] | +2.84 [+0.84, +4.78] | +1.54 |

- **Exact, CA6 `ladder_X1`:** 15 legs; **`n_ci_excl0` = 12**; min −2.15, **max 4.133333333333334**; 9 of 15 positive. RMS Δ **2.615173848764028**.
  If the two draws were independent and equally variable, one fit's draw SD would be RMS/√2 = **1.8492071624427666**. ⚠ That assumption does not hold
  cleanly: the sign follows the **seed index** (mine above ref at seed 0 on 5/5 rungs, below at seed 1 on 5/5, above at seed 2 on 4/5). So the 15 legs
  are **3 key pairs × 5 rungs**, not 15 independent draws.
- ⛔ **The spoke's verdict, as record only.** "REPRODUCES within seed spread" at 5/5 rungs is judged against a bar the spoke **pre-declared** at LOG 13:58Z
  (|Δ of 3-seed means| ≤ the larger across-seed SD). By the Head's ban it carries no authority here. The measured per-seed numbers above are the finding.
  The spoke itself states: *"Family differences smaller than about 4 on a single seed should therefore not be read as real."*
- ⭐⭐ **Why this bears on every single-fit margin.** Every per-seed margin in the programme pairs one store with **one** fit of each rival per seed, and
  its CI95 is a **per-corner bootstrap** that holds that fit fixed. X1 shows two fits that differ only in the PRNG draw, each with a per-corner CI that
  excludes 0 on 12/15 legs. ⇒ The printed CIs measure corner sampling, **not refit variance**.
  - **Shapes rows 3–4 headline** (`SHAPES-MATCHED.json` `headline.data_matched_margin_rows34_K32`): per seed 7.486238532110092 / 8.596330275229358 /
    5.018348623853211, mean 7.033639143730887. The largest X1 swing (4.133333333333334) is 0.5876521739130436 of the mean, and seed 2's margin exceeds it by
    only 0.8850152905198776 (CA6).
  - **Ladder:** one per-seed margin in set B (+3.85) and one in C-sm (+3.25), both n = 2 seed 2, are **below** the largest X1 swing (CA6
    `n_per_seed_margins_below_X1_max_abs_*`). No set-A or set-C per-seed margin is.
  - ⭐ **What still holds:** the 3-seed **mean** margins (+5.13 … +14.54 on the ladder in sets B / C / C-sm, +7.03 on Shapes) are several single-fit draw SDs above 0, and every
    per-seed margin is positive. The **sign** is not threatened by X1. What is threatened is the precision of per-seed digits and of any
    between-family or between-set difference under ≈ 4 (e.g. `N393`'s C − B ±1.5, w20's "≈ 0.5 for 8× data", `N373`).
- ⚠ **Scope limits, stated:**
  1. X1 is an **mlp at 2,560 steps**, an under-trained regime (5.43 … 23.88 novel/corner, against ≈ 25–60 at the carried budgets). No same-recipe PRNG
     replicate exists at the **carried** budgets. The nearest evidence is indirect: the mlp's step-matched − TENX differences (204,800 vs 51,200 steps,
     a regime where the mlp is flat) run −4.76 … +2.02. The spoke reads that as the same magnitude.
  2. Shapes / ladder only. No PRNG-replicate measurement exists for CMAZE-T's exact-match metric.
  3. The margin convention takes the rival as a **max over families × stages**. A max over several noisy fits is biased **upward**, which on average
     works against the store. It does not bound a single seed.
- ⛔ **What would settle it (NOT RUN; zero compute at this pass):** k PRNG replicates of the carrying rival per seed at the carried budget, giving a refit-
  variance interval beside every per-corner CI. Flagged for the Head (`§I8.21`).
- **Matrix row:** `CM-W21-6`. ⟲ at `N372`, `N373`, `N377`.

**Reading key (figures).** A CMAZE-T **fact vector** is `[legal, displacement, event, pocket-next]`; `judge_transition` marks an emission ✓ / ✗ on all four
facts (only pocket-next for an illegal emission). A **panel** draws the resulting **state**. An **opposite-mark pair** = two arms on the same held-out
transition with different marks. **"Same picture"** is measured at two levels in `AMBIGUITY.json`: `state` (the drawn state tuple) and `pixels` (the
rendered `draw_state` pixels).

### N395 — ⚠ **(figures) THE CMAZE-T PANELS DREW THE STATE WHILE THE JUDGE SCORES THE FACT VECTOR, SO FACTS THAT DO NOT CHANGE THE STATE WERE INVISIBLE. 33,412 of 92,628 opposite-mark arm-pairs (A1, seeds 0–2) rendered identically. 3,740 of them differ only in the legality bit (the Head's class); 16,283 have pocket-next as the reason. ⭐ A fact strip brings every count to 0. ⛔ No number was ever affected: scoring always used all four facts. ⚠ The task figure drew `c1` blue where the panels drew it red; now unified and verified pixel-identical (7/7 discs).** · tier B (CURATOR-PROPOSED ◈) · figures / claims hygiene (w22 spoke `figures-redraw` pass 3; the Hub's defect note)

**Source:** branch `w22/figures3` @ `cb5e0ac` → `w22_figures3/AMBIGUITY.json`, `AMBIGUITY-before.json`, `GLYPH-IDENTITY.json`, `PALETTE-MAZE_RGB-before.txt`,
`PALETTE-MAZE_RGB.txt`, `LOG.md`, `gate2/cmazet_items_rel.REPRO.json` (copies: `~/clu/outputs/w22-registry-fold6/curator/src_w22_figures3/`);
`w22-figures3-FINDINGS.md`; `FIGURE-DEFECT-cmazet-legality.md`; CA6 `figures3`.

| items · arms | opposite-mark pairs | same picture (`pixels`) | same state (`state`) | legality shown only (`state+legal`) | Head's class (legal bit only) | **after the strip** |
|---|---|---|---|---|---|---|
| A1 (w19) s0–2 · all 16 | 92,628 | **33,412** | 30,930 | 18,035 | **3,740** | **0** |
| A1 (w19) s0–2 · 13 drawn | 63,624 | 22,340 | 20545 | 12,624 | 1,261 | **0** |
| CLU-REL-A10 s0–5 · all 22 | 373,583 | 154,671 | 150440 | 107,148 | 19,915 | **0** |
| CLU-REL-A10 s0–5 · 19 drawn | 295,294 | 124,450 | 120991 | 88,384 | 13,115 | **0** |

- **By reason, A1 all arms** (`by_reason_pair`, pixel level): legality 11175, pocket **16283**, incoherent-key-event 5954; sum 33412 (CA6). ⚠ The
  pre-fix file (`AMBIGUITY-before.json`) gives incoherent-key-event 3472 (state level; 11175 + 16283 + 3472 = the state-level 30930). Pixel-level
  opposite-mark pairs that **are** visibly different: 92,628 − 33,412 = **59216** (CA6).
- **The worked example** (seed 2, item 482; `rule_a`, pocket 3, door 1, tint none, east; truth `[0,0,0,3]`, illegal). The CLU emitted `[1,0,0,3]` ✗
  `legality`; the matched mlp emitted `[0,0,0,3]` ✓. Both panels showed the agent in place with pocket 3. ⭐ After the fix the strip reads grey
  **`legal`** under the CLU and bold **`illegal`** under the mlp. The curator viewed the compiled page crop at print scale (`pages/p46-rule_a-crop2.png`):
  the `legal` / `illegal` words are visible under every panel of that row.
- ⭐ **The fix:** under every panel, line 1 prints legal / illegal; line 2 prints move, key event and pocket after the move (only the pocket for an illegal
  emission). The drawn state and disc are unchanged. Applied to every figure sharing `draw_state`: `fig_cmazet_families`, `_per_arm`,
  `_door_then_tint`, `_deletion`, and the new `fig_cmazet_rel_per_arm` (19 arms, one success and one failure each, seed 0). The captions now say the
  tick is the judge's verdict on all four facts.
- ⛔ **No number affected:** the reported scores are the judge's verdicts over all four facts. The defect note records **0** cases where the same emitted
  vector gets different verdicts. The REL extractor re-read the staged objects and matched **132/132** arm scores float-`==` to the run JSONs, with 144
  objects hash-checked; CLU per seed 0.9080 / 0.8935 / 0.9840 / 0.9274 / 0.9004 / 0.9394.
- ⚠ **Glyph and palette** (`GLYPH-IDENTITY.json`; `PALETTE-*.txt`):
  - `fig_task_cmazet` is now drawn by `draw_state` itself. Its 52 disc pixels equal a result-panel raster on **7 of 7** blocks (`identical` = true), and
    ring `0c0c0c` and centres `cc302b` / `e0a800` / `ffffff` appear as the same bytes in both PDFs.
  - The task figure's c1–c4 had been blue / orange / aqua / violet, against the panels' red / blue / amber / purple. They now use the panels' `MAZE_RGB`.
  - ⚠ **The panels' palette had FAILED the validator since pass 1:** blue↔purple normal-vision dE **13.8** (floor 15) and CVD dE **7.2**
    (`PALETTE-MAZE_RGB-before.txt`). c4 was re-stepped to `#9b3fa0`: dE 17.8, CVD 8.2, all checks pass. Amber stays below 3:1 on white (WARN, relieved
    by digits and row labels).
- ⛔ **Not done** (`w22-figures3-FINDINGS.md`): no REL deletion panel (the REL deletion objects were not re-read); `fig_task_shapes` not rebuilt.
  ⚠ **The 32,768-row ladder slot was left `not_landed`**, although `ladder-rivals-32k` ended (22:03:55Z) inside figures3's own run (21:44:48Z – 22:13:46Z).
  The slot is still unfilled (`§I8.21`).
- **Matrix row:** `CM-W22-1`.

### N396 — ⚠ **(provenance) NINE PLACES WHERE THE PASS-6 BRIEF, A SPOKE'S PROSE OR THE HUB'S DEFECT NOTE DISAGREES WITH THE PRIMARY RECORD. The artefact wins each one. This is `N351` / `N371` / `N381`'s rule applied a fourth time, twice to this pass's own brief.** · tier C (CURATOR-PROPOSED ◈) · process / provenance (the pass-6 brief; `ladder-rivals-32k` LOG; `figures3` FINDINGS and LOG; `FIGURE-DEFECT-cmazet-legality.md`)

**Source:** as cited per row; CA6.

| # | the prose | where | the artefact |
|---|---|---|---|
| 1 | X1 *"CIs excluding 0 on 10/15 legs"* | the brief; `ladder-rivals-32k` `LOG.md` 18:10Z | **12 of 15** (CA6 `n_ci_excl0`; `FINDINGS.md` §2). The ladder LOG's own later line reads *"CORRECTION to 18:01Z: per-seed X1 paired CIs exclude 0 on 12 of 15 legs (final JSON), not 10"*. The brief copied the superseded line |
| 2 | X1 *"±4.1"* | the brief; ladder `LOG.md` 18:10Z | **−2.15 … +4.13**. The negative side reaches 2.15, not 4.1; 4.133333333333334 is one leg (n = 5, seed 2) |
| 3 | CLU-REL-A10 *"fixes A1's two weak seeds (seed 1 0.566 → 0.894, seed 5 0.682 → 0.939)"* | the brief | A1's hard seeds are **1 and 4** (`class_from_maze_t_seed4`), and its lowest seed is **4** (0.533), where the REL gain is the largest (+0.368) **and** where REL ties `random_atoms`. Seed 5 is A1's **losing** seed (−0.185, `N384`), where REL ties its best rival. Both brief numbers are correct; the selection of seeds omits the one that complicates the story |
| 4 | *"⛔ NOT a store/memory result"* as a blanket statement | the brief | Holds at the six-seed mean (+0.031) and on seeds 0 / 3 / 4 / 5. ⚠ On **seed 2** the CLU keeps **+0.165** beyond the best `+rel` rival (share recovered 0.45), and on seeds 0–3 / 5 the CLU's own read drops by 0.112–0.507 under the strongest control (`N392`) |
| 5 | palette *"normal-vision ΔE 13.7 … CVD 7.1"* | `w22-figures3-FINDINGS.md`; figures3 `LOG.md` | **13.8** and **7.2** (`PALETTE-MAZE_RGB-before.txt`, the validator's own output) |
| 6 | *"The Head's defect file `evidence/outputs/FIGURE-DEFECT-cmazet-legality.md` does not exist"* | figures3 FINDINGS; LOG 21:44Z | The file **exists** on batmac at that path (read at this pass), mtime `2026-09-24T22:59:24+0200` (20:59:24Z), **before** figures3's START (21:44:48Z). Why the spoke's search missed it is not determinable from the record. Every number in the note was re-derived by figures3 anyway |
| 7 | *"…visibly different predictions (figure is fine): 88,888"*; *"⚠ The pocket is not the problem … Re-rendering the pocket does not help"* | `FIGURE-DEFECT-cmazet-legality.md` | 88,888 = 92,628 − 3,740 counts only the Head's class. Pixel-level, **59216** pairs are visibly different and 33,412 are not (CA6). The pocket statement holds **for item 482**. On **16,283** same-picture pairs the reason is pocket-next, which the strip now prints. The note's 92,628 and 3,740 reproduce exactly |
| 8 | the ambiguity cascade *"33,412 … would still leave 18,035 … 16,283 … 1,752"* | `w22-figures3-FINDINGS.md` | Mixed levels: 33,412 is the **pixel** count; 18,035 and 1,752 are **state**-level (state total **30,930**, figures3 `LOG.md`). 16,283 happens to be the pocket count at both levels. Each number is right at its own level |
| 9 | LOG timestamps labelled `Z` | ladder `LOG.md` (*"23:45Z X1 DIAGNOSTIC RESULT"* after END `22:03:55Z` and after the 22:03Z bundle line; 21:50Z listed before 21:42Z); figures3 `LOG.md` (*"23:52Z GATE 2 PASSED"*, *"00:1xZ FINAL INSTALL"* before END `22:13:46Z`) | Not monotone, and some postdate END. They are consistent with local time (UTC+2) written as `Z`, but the record does not say so. ⚠ **No number depends on these timestamps**: the X1 replay result and gate 2 are in their JSONs |

- ⛔ **FORBIDDEN in reports:** items 1, 2 and 5 as worded. Item 3 without seed 4. Item 4 without the seed-2 exception. Item 7's *"88,888 … figure is fine"*.
- The curator edits **no** evidence file and no branch. The fixes are flagged for the Hub (`§I8.21`).
- **Matrix row:** `CM-W21-4` (items 3, 4), `CM-W21-6` (items 1, 2), `CM-W22-1` (items 5–8).

### N397 — ⚠ **(process, omit nothing) THE PASS-6 SWEEP FOR WITHHELD ROWS AND NOT-RUNS in the three units. None of them wrote a PREREG. One pre-declared bar exists (the ladder's X1 bar, LOG 13:58Z); it is quoted as record, never as authority (`N394`). Every row a unit declared withheld or not run is listed with its status.** · tier C (CURATOR-PROPOSED ◈) · process / claims hygiene (w21 `rel-newseeds`, w21 `ladder-rivals-32k`, w22 `figures3`; the Head's ban)

**Source:** each unit's `FINDINGS.md` ("What did NOT run" / "Did NOT run") and `LOG.md`; `w21-ladder-rivals-32k/FINDINGS.md` §5; CA6.

| unit | item | status | where folded |
|---|---|---|---|
| rel-newseeds | smoke run | NOT RUN (D2: the same runner and flags ran to EXIT 0 three times on VAL) | here |
| rel-newseeds | OBS / WORDS-FULL / WORDS-MAIN with CLU-REL-A10 on seeds 3–5 | NOT RUN (not in the brief) | here; ⚠ the WORDS-FULL VAL zero-store result stays three-seed (`N383`) |
| rel-newseeds | replay of seen / pair / value / deletion / rollout | NOT RE-SCORED (hash-checked only); the n-way row is replayed bit-exact | `N391` |
| rel-newseeds | A1 `+rel` | DOES NOT EXIST (A1 runs never had one) | `N392` |
| rel-newseeds | value row, seed-0 `+rel` ceiling, shares on seeds 0–2, HIGH flags, per-tuple control near-ties, seed-3 tuples a rival reaches and the CLU does not, rollout | ⭐ **printed by the spoke, not named by the brief: FOLDED** | `N391`, `N392` |
| ladder-rivals-32k | anything on w20's outputs (set B not refitted or re-scored) | NOT RUN by rule; set B recomputed from w20's score files = w20's printed P3 on 15/15 | `N393` |
| ladder-rivals-32k | store fit or re-score; the store's controls | NOT RUN (w12b rows, w20 controls: all four collapse on every K\* = 32 ladder store) | `N393` |
| ladder-rivals-32k | cont_mlp beyond 204,800 steps | NOT RUN; not proven converged (up to +19.33), never the strongest rival | `N393` |
| ladder-rivals-32k | wider spoke-A grids | NOT RUN; all 60 selections at the grid edge | `N393` |
| ladder-rivals-32k | LACE / PoE at 32,768 rows; rows 3–4 TEST | NOT RUN (not in the brief; the scorer refuses non-rung corners) | `N393` |
| ladder-rivals-32k | PRNG replicates at the carried budgets | NOT RUN (not in the brief); X1 exists only at 2,560 steps | `N394`; `§I8.21` |
| ladder-rivals-32k | the X1 verdict *"REPRODUCES"* | a **pre-declared** bar (LOG 13:58Z); record only | `N394` |
| figures3 | REL deletion panel; `fig_task_shapes`; the 32,768-row ladder slot (`not_landed`) | NOT DONE | `N395`; `§I8.21` |
| figures3 | the draft no longer places `fig_headline_steps` or the three `rendered/rc_w19_t7_*` figures (another session's change); `\PLACEHOLDER` count 4 (pass 2 reported 15) | reported by the spoke; not touched by it or by the curator | here; `§I8.21` |
| figures3 | the dataviz JS validator | NOT RUN (no `node` on batmac); the repo's Python port was used | `N395` |

- **Left unreported, with reason:** nothing. ⛔ The curator re-ran nothing and verified no persisted object: only manifests, JSONs and one page crop are
  on batmac for these units.
- **Matrix row:** `CM-W17-9`.

## ◈.20 ⭐ THE FOLD-IN LEDGER, PASS 6 — every brief item, and where it now lives

| brief item | status | where |
|---|---|---|
| ⭐ CLU-REL-A10 six-seed mean 0.925 vs A1 0.677; ceiling on all six; no WORSE@80k on seeds 3–5; best rival unflagged | ⭐ **FOLDED**, ⚠ with seed 4 named (A1's lowest seed; `N396` item 3) and the seed-0 `+rel` ceiling | `N391`; `CM-W21-4`; ⟲ at `N383`, `N384`, `N388` |
| ⛔⛔ `+rel` erases the margin (−0.007 / +0.024 / +0.165 / +0.006 / +0.000 / +0.000; mean +0.031; share 0.97 / 1.00 / n.a.) | ⭐ **FOLDED**, with shares on all six seeds (1.06 / 0.93 / 0.45 / 0.97 / 1.00 / n/a) and the seed-2 exception | `N392`; `CM-W21-4` |
| ⛔ seed 4 TEST `random_atoms` = CLU = 0.9004, 16/16 tuples | ⭐ **FOLDED**, **re-counted by the curator from the replay files** (16/16, 651 of 723 rows) | `N392`; ⟲ at `N386` |
| ⛔ seed 5 CLU 0.939 = best matched rival 0.939 | ⭐ FOLDED | `N392` |
| ⭐ the claim: an input representation, not a store / memory result | ⭐ **FOLDED as scoped by the artefacts** | `N392`; `CM-W21-4` |
| ⭐ MF-2 immaterial: A +17.53 · B +9.97 · C +10.28 · C-sm +10.75; ±1.5; CI lows > 0 (lowest +1.98); pool identity 15/15 | ⭐ **FOLDED**, ⚠ "±1.5" scoped to set C (C-sm − B up to +2.2194444444444414, CA6) | `N393`; `CM-W21-5`; ⟲ at `N377`, `N381` item 7 |
| ⭐⭐ at matched steps the pool barely moves the rivals; the step budget does | ⭐ FOLDED | `N393`; ⟲ at `N373` |
| ⚠⚠ X1: single-fit RNG-draw variance, its own entry | ⭐ **FOLDED AS ITS OWN ENTRY**, ⚠ with "10/15" → **12/15** and "±4.1" → **−2.15 … +4.13** | `N394`; `CM-W21-6`; ⟲ at `N372` |
| ⛔ figure defect: 33,412 of 92,628; 3,740 legal-bit; 16,283 pocket-next; fixed to 0; no number affected; c1 unified; 7/7 discs | ⭐ **FOLDED**, with the REL counts, the pre-existing palette failure (13.8 / 7.2) and the unfilled ladder slot | `N395`; `CM-W22-1` |
| *(curator-added)* disagreements between the prose and the artefacts | ⭐ FOLDED | `N396` |
| *(curator-added)* the omit-nothing sweep for the three units | ⭐ FOLDED | `N397` |

**⛔ WHAT DID NOT RUN AT THIS PASS**

| not run | reason |
|---|---|
| ⛔ any scored run, compute beyond reading JSONs, GPU | ZERO. The CA6 script only reads evidence and takes means, differences and counts |
| ⛔ PRNG-replicate fits at the carried budgets (the measurement `N394` says would settle refit variance) | zero compute; the Head's (`§I8.21`) |
| ⛔ verifying persisted rel-newseeds / ladder objects | only manifests and JSONs are on batmac; the spokes' own replays are quoted |
| ⛔ re-rendering or pixel-checking any figure | only one page crop was viewed; the counts are the spoke's `AMBIGUITY.json`, which the curator did not recompute |
| ⛔ computing any label, or ruling the paper wording for `N392` / `N394` | the ban; the Head's and the Advisor's |
| ⛔ editing any draft, `.tex`, ledger, evidence file or other branch | standing rules; the `.tex` lock was not taken |

⭐ **No seventh pass is required for these three units.** Anything that lands after `2026-09-25T06:16:47Z` (this pass's START) is a new pass.
