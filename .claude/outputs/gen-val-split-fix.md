# gen-val-split-fix — experiment-engineer report

**Task + acceptance criterion:** repair the `fit`/`val` split (stratified re-select), then re-run
**every** configuration selection — ours, `knn_proto`/kNN, LACE incl. its whitening-exactness
fallback — on the repaired split and re-score, with a MOVED/UNCHANGED table and **no rival weaker**.
**Status: done, with ONE stop-and-report item (§6).** 12 production re-runs + a 3-run laundering
control; nothing NOT RUN.

> ## ⛔ DOWNSTREAM RECONCILIATION LIST — 5 items, needs a Hub owner (protocol §5 corollary, first-10-lines rule)
> 1. ⭐⭐ **THE CLU ARM DID NOT MOVE AT ALL — BIT-IDENTICAL, 3/3 SEEDS — AND EVERY RIVAL DID.**
>    `clu` in-corner **0.6568 ± 0.0226**, composition-B **0.5949 ± 0.0082**, 9.0 distinct tuples:
>    Δ = 0.000000 exactly. The CLU read is addressed through the store fitted on `store_train`,
>    which this repair does not touch. ⛔ So the repair cannot have manufactured a win for us; it
>    re-priced the *comparators* only. **Owner: Hub — this is the sentence the re-score is for.** (§4)
> 2. ⛔ **STOP-AND-REPORT (acceptance item 5): `knn_proto`'s composition-B aggregate FALLS and the fall
>    is REPRODUCIBLE, not noise.** 0.9005 ± 0.0182 → **0.8650 ± 0.0122** (cell harness, external
>    instrument) and 0.8410 ± 0.0330 → **0.8110 ± 0.0257** (baselines). A 3-way laundering control
>    (three *further* stratified splits, different salts) puts the cross-split spread at **± 0.0010**
>    ⇒ the published 0.7963 (seed 0) was an artefact of the head/tail `fit`, and ~0.776 is what any
>    balanced `fit` gives. ⭐ `knn_proto` **has no tuning axis**, so this is the `fit`-row swap, ⛔ not
>    a selection rule. **I am not clearing this myself. Hub ruling needed.** (§6)
> 3. ⭐⭐ **THE REPAIR REMOVES A BIAS BUT DOES NOT BUY IDENTIFICATION.** On the *exchangeable*
>    (interleaved) halving the two halves of `val` still pick different winners in **5 of 6** cells —
>    the same 5-of-6 that motivated this task. Best-to-second gaps are **0.0007–0.0130** against a
>    trial spread of **0.068–0.111**. ⛔ A tuned `gen` configuration still must not be quoted as
>    *identified*; it is now merely *unbiased*. **Owner: Hub — `tune_n`/`n_val` is the real fix.** (§5)
> 4. ⚠ **FIVE published reference constants are now stale on disk.** `GenAtomMeanConfig
>    .knn_proto_probe_reference_per_seed` is re-based in this branch (commit `ca14a34`);
>    `GenCellConfig.{clu,knn_proto,lace_ode}_probe_reference_per_seed` are **not** (they use a 0.05
>    tolerance, so they warn rather than abort). `GenCellConfig.lace_selection_source` still points at
>    `gen-arms-a7-wiring/runs/historical_688128`, which is the **broken-split** selection. (§7)
> 5. ⚠ **`gen-arms-a7-wiring/refusal_scores.json` is superseded for refusal claims.** The repaired-split
>    replacement is `gen-val-split-fix/refusal_scores.json`, `"status": "complete"`, `not_run=[]`,
>    2.9 MB, both byte legs × 3 seeds. Refusal moved **a lot** (kNN AUROC(C vs B) **0.5976 ± 0.1159 →
>    0.8457 ± 0.0238**). (§4)

---

## ⭐ DIAL DECLARATION (echoed, protocol §7)
- **Dial:** ⛔ **none — a REPAIR and a RE-SCORE.** No new arm, no mechanism, no verdict on any claim.
- **Laundering control:** ⭐ **every rival was re-selected by the same procedure, from the same config
  block, on the same `(seed, 32256, 512)`** — one code path (`chlu.data.gen_rowsplit`) serves ours,
  kNN/`knn_proto` and LACE, so a repair cannot reach our split without reaching theirs. A **second**
  control was added for the one number that fell (§6): three further independent stratified splits.
- **Falsifies:** nothing of ours. It moved published numbers, including refusal's. That was the point.
- **Does NOT falsify:** a rival number going *down* is not automatically a hobbling — `knn_proto` has
  no tuning axis, so anything it does is the `fit`-row swap, which the task's own constraint
  (inherit 32,256/512, ⛔ never shrink `fit`) makes unavoidable.

## ⭐ FLAG PROVENANCE (protocol §5, mandatory)

| item | value |
|---|---|
| branch / base | `agent/experiment-engineer/gen-val-split-fix`, base **`main @ 5a3c32c`** (unmoved; `git rebase main` = "up to date") |
| commits | `e8cc1a2` the split + module + tests · `ca14a34` re-based `knn_proto` constants · `6c693aa` retire two now-false notes |
| runs executed at | ⚠ **leg D and leg H seed 0 at working tree `e8cc1a2`; leg H seeds 1–2, legs A, C and the null at `6c693aa`** (the two intervening commits touch only `GenAtomMeanConfig`'s reference constants — unused by `exp_gen_baselines` — and two comment/log strings, so ⛔ no leg-H number is affected). HEADs echoed into `runs/progress.txt` by the runner |
| worktree / env | `/Users/user/Desktop/CHLU-gen-valfix`; ⭐ **main venv reused, ⛔ no `uv sync`** (`PYTHONPATH=<worktree> /Users/user/Desktop/CHLU/.venv/bin/python`). `JAX_PLATFORMS=cpu`. `CHLU_CLAUDE_ROOT` / `CHLU_GEN_DATA_ROOT` / `CHLU_GEN_BRIDGE_ROOT` / `CHLU_GEN_DECLIFT_DIR` / `CHLU_GEN_STORE_RUNS` all set (a worktree has no `.claude/`) |
| ⛔ shared checkout | `/Users/user/Desktop/CHLU` was on another agent's branch `agent/experiment-engineer/pilot-ttt-nan-and-d5-wiring` and was **never touched** (protocol §3.2) |
| dataset / pretext / condition | `shapes3d` / `recon` / `recomb2element` (Montero) |
| seeds | **0, 1, 2** on every leg; ⛔ nothing NOT RUN |
| rows | fit **32,256** · val **512** · A **1,024** (cell/atom 512) · B **72** · MS **198/199/199** — ⭐ `GenBaselinesConfig`'s own, ⛔ never a spoke's budget |
| **NEW flag** | ⭐ `GenRowSplitConfig.mode` — **default CHANGED**, `"head_tail"` → **`"stratified"`**. Also new: `tiebreak_jitter=1.0`, `seed_mult=6637`, `seed_offset=20260914`, `assert_balanced=True`, `require_all_values=True`, `max_count_ratio=1.25` |
| ⛔ changed constants | `GenAtomMeanConfig.knn_proto_probe_reference_per_seed` re-based (see §7) |
| everything else | shipped defaults, unchanged: 36-trial kNN grid, 36-trial LACE grid, `lace_n_epochs=50`, `lace_batch=128`, `lace_lr=1e-4`, `lace_ld_steps=100`, `lace_whitening_exact_tol=0.10`, `lace_whitening_exactness_fallback=True`, `refusal_valid_quantile=0.95`, `knn_kmeans_iters=25`, `decoder_source="decoder_lift_v1"` / `"winner"` with **`v1_capacity_schedule` asserted by name AND hash** (`01ef91f7…`/`001011bc…`/`2d7eea75…`, ⛔ never the `"baseline"` default), `assert_decoder_hash=True`, external classifier gate **PASS** (`classifier_gate.json`, floor 0.95, 3 seeds, asserted in-run) |
| legs | **D** deployed 524,288 B `--arm knn` ×3 · **H** historical 688,128 B `--arm all` ×3 (LACE's published leg + the R27.3 fallback) · **C** cell harness ×3 · **A** atom-mean ×3 (2 budgets each) · **null** 3× leg-D seed 0 at alternative split salts |
| ± convention | **population sd (ddof=0)** everywhere below |
| wall clock | D 5+8+7 min · H 28+65+44 min · A 4+8+8 min · C 17+16+16 min · null 3×≈3 min ⇒ **≈ 4 h 0 m, 15 runs**, all `rc=0` |
| artefacts | `.claude/outputs/gen-val-split-fix/` — `PREREG.md` (filed **before** the split was built), `tables.md`, `balance.json`, **`refusal_scores.json` (complete, 2.9 MB)**, `runs/{deployed_524288,historical_688128,cell,atom_mean,split_resample_null}/`, `runs/*.log`, `runs/progress.txt`, `run_rescore.sh`, `balance_probe.py`, `compare.py`, `selection.py`, `headlines*.py`, `weakness_audit.py`, `split_resample_null.py`, `assemble.py` |

---

## 1. ⭐ THE INHERITED NEGATIVE, LED WITH — and it is worse than it was reported

`gen-atom-mean-control` §7 reported that the broken `val` realised **1** `floor_hue` value and **2**
`wall_hue` values. ⭐ **That understated it.** The broken `val` also realised its *other four* factors
at wildly unequal counts, and two of those four are `scale` and `orientation` — the factors this
phase turns on:

| factor | card | ⛔ BROKEN `val` `n_values_realised` | ⛔ BROKEN `val` per-value **count ratio** | ⭐ FIXED `val` nvr | ⭐ FIXED count ratio |
|---|---|---|---|---|---|
| floor_hue | 10 | **1** | 1.000 (only one bin to be flat in) | **10** | 1.040–1.060 |
| wall_hue | 10 | **2** | 1.498–1.573 | **10** | 1.040–1.060 |
| **object_hue** | 10 | 10 | ⛔ **2.630–3.524** | **10** | 1.040–1.060 |
| **scale** | 8 | 8 | ⛔ **1.333–1.463** | **8** | 1.032–1.082 |
| shape | 4 | 4 | 1.151–1.621 | **4** | 1.016–1.024 |
| **orientation** | 15 | 15 | ⛔ **1.692–1.880** | **15** | 1.061–1.091 |

(3 seeds, ranges across them; full per-seed table in `tables.md` / `balance.json`.)
⇒ **"balanced on the hues but not on `scale`/`orientation`" was the named falsifier and it is
answered on every factor**, and the *new* finding is that the broken split was never only a hue
problem: `object_hue` was realised 3.5× more often at one value than another.

**Feature variance — measured, not argued:** `val` **0.0924 → 0.2232** against `fit`'s **0.2227**
(seed 0; 0.0934→0.2267 vs 0.2268 and 0.1050→0.2402 vs 0.2399 on seeds 1–2). `PREREG` §4 predicted
"within ±10 % of `fit`'s 0.2229"; realised **+0.2 %**.

**And the consequence the defect predicted is gone.** The atom-mean arm's relative read error on
`val` vs on `A` (the unfitted primary):

| | read error on `A` | read error on `val` | gap |
|---|---|---|---|
| ⛔ broken split, budget A | 0.3567 ± 0.0319 | **0.8189 ± 0.0986** | **0.46** |
| ⭐ repaired, budget A | 0.3451 ± 0.0382 | **0.3204 ± 0.0468** | **0.025** |
| ⛔ broken split, budget B | 0.1851 ± 0.0324 | **0.3993 ± 0.0649** | **0.21** |
| ⭐ repaired, budget B | 0.2062 ± 0.0347 | **0.1820 ± 0.0304** | **0.024** |

`PREREG` P4 predicted the two columns converge to within ±0.10; realised **0.025 / 0.024**.

## 2. What I built (minimal diff, 3 commits, 744 insertions / 43 deletions)

- **`chlu/data/gen_rowsplit.py` (new, 305 lines).** `stratified_fit_val` = *deficit-priority greedy*
  against the **pool's own** per-factor marginals (⛔ not a uniform target, which would impose a
  distribution the pool does not have), seeded tie-break; `head_tail_fit_val` reproduces the broken
  split bit-for-bit; `row_split_balance` + `assert_row_split_balanced` (**capable of failing**: handed
  the shipped split it raises on `floor_hue`).
  ⚠ My first draft was a *cap-and-relax* filter and it was **not good enough** — a cap is one-sided,
  so a row rejected late because one of six buckets was full left the other five underfilled:
  measured `floor_hue` counts **42..53** (ratio 1.262) on seed 1, no tighter than plain random
  sampling. That version is documented in the module docstring rather than quietly dropped.
- **One appended config block** at EOF of `chlu/config.py`, `BEGIN`/`END`-delimited,
  **zero house-wiring sites** (accessor `gen_row_split_config`, the `gen_navigate_config` pattern).
  ⭐ **Default changed to `"stratified"`** — preserving the broken split as the default would preserve
  a coin flip at the centre of every tuned configuration in the phase.
- **Three call sites** (`exp_gen_baselines.select_baseline_rows`, `exp_gen_cell`, `exp_gen_atom_mean`)
  now share that one path; each **asserts** the balance and **records** it in its manifest
  (`row_split.balance`).
- **`exp_gen_atom_mean`'s half-split diagnostic gained an INTERLEAVED halving** beside the contiguous
  one. On a *sorted* `val` the contiguous halves are non-exchangeable by construction; the interleaved
  halves inherit `val`'s marginals and are the halving that actually asks "is this win reproducible?".
  ⛔ The contiguous leg is kept so the published 5-of-6 stays comparable.
- **`tests/test_gen_rowsplit.py` (9 tests)** — incl. the positive control (the gate refuses the shipped
  split, on the toy grid *and* on the real 480,000-row Shapes3D grid) and the "head_tail reproduces it
  bit-for-bit" test.

**Structural guarantees (`PREREG` P7): all UNCHANGED.** `pool` (`store_train`) untouched ⇒ `no_leak`
0 leaked of 72 on 3/3; row counts 32256/512/1024/72/198–199 identical; store plan `K=5`, `n_modes=64`,
`D=2048`, deployed bytes 524,288, channel 30 bits, prototype table 475,136 B, byte target 688,128 B,
and all decoder/encoder/probe/classifier hashes identical. `fit` changes by exactly the swapped rows:
**500 / 503 / 503 of 32,256 = 1.55 %**.

## 3. ⭐ ALL THREE SELECTION FAMILIES RE-RUN — and the selection was a coin flip

| family | (arm, seed, rule) cells | selection MOVED |
|---|---|---|
| kNN, deployed 524,288 B | 6 | **5 / 6** |
| kNN + LACE, historical 688,128 B | 12 | **8 / 12** (⭐ seed 2 unchanged for *both* arms, both rules) |
| atom-mean (ours), 2 budgets × 3 seeds | 6 | ⛔ **6 / 6** |
| cell `ebm_compose` (ours) | 3 | 2 / 3 |
| cell LACE (inherited, ⛔ never re-tuned here) | 3 | 2 / 3, inherited from the re-run leg H |

⭐ **A coherence signal.** Under the repaired split the kNN selection lands on `int8` (248 entries —
the *most* entries the budget buys) on **3/3** seeds at both budgets, where the broken split picked
`int8`/`fp16`/`int8`. A balanced `val` makes "more entries is better" visible; the broken one did not.
⛔ `PREREG` P3 predicted the argmax would move in 2–5 of 6 and **not in all 6**; for the atom-mean
family it moved in **all 6**. **P3 is falsified on our own arm** — our selection was *entirely*
unidentified, more so than the rivals'.

**LACE's R27.3 whitening-exactness fallback re-ran, once per firing seed.** The gate is decided purely
by *which shrinkage is selected* — `PREREG` P6 predicted the whitening statistics themselves would move
< 0.01 and the exactness booleans would not change; measured `mean_abs_diag_minus_one` moves of
**−0.0007 / +0.0046 / +0.0038** (`none` / `ledoit_wolf` / `1e-3`, seed 0) with the 0.10 classification
unchanged. **P6 confirmed.** The gate fired on **seeds 0 and 2** before and on **seeds 1 and 2** after;
the fallback leg ran on each firing seed exactly once and ⛔ never twice on a configuration already
being run.

## 4. ⭐ THE MOVED / UNCHANGED TABLE (full version in `tables.md`)

### ⛔ UNCHANGED — bit-identical, Δ = 0.000000, 3/3 seeds
| row | value |
|---|---|
| ⭐ **`clu` in-corner shape balacc (external)** | **0.6568 ± 0.0226** |
| ⭐ **`clu` in-corner AUROC** | **0.7653 ± 0.0123** |
| ⭐ **`clu` composition-B aggregate** | **0.5949 ± 0.0082** |
| ⭐ **`clu` distinct factor tuples on B** | **9.00 ± 0.82** |
| `store_removed` (the null) — every statistic | balacc 0.5000, comp-B 0.1204 ± 0.0131, 1 tuple |
| atom-mean read error k=1 on `A`, budget B | 0.2306 ± 0.0038 |
| atom-mean centroid control k=1, budget B | 0.1695 ± 0.0021 |
| atom-mean medoid-over-centroid penalty, budget B | 1.3604 ± 0.0108 |
| every byte ledger, row count, hash, `no_leak` | see §2 |

### ⭐ MOVED — the rivals (re-selected on the repaired split)
| row | ⛔ broken split (published) | ⭐ repaired split | Δmean |
|---|---|---|---|
| kNN refusal **AUROC(C vs B)**, deployed | 0.6306 ± 0.1471 | **0.8357 ± 0.0218** | **+0.2051** |
| kNN refusal **AUROC(C vs B)**, historical | 0.5976 ± 0.1159 | **0.8457 ± 0.0238** | **+0.2481** |
| kNN A7-controlled `out_of_range` AUROC, deployed | 0.8281 ± 0.1299 | **0.9583 ± 0.0048** | **+0.1303** |
| kNN composition-B aggregate, deployed | 0.4028 ± 0.0774 | 0.4599 ± 0.0406 | +0.0571 |
| kNN composition-A aggregate, deployed | 0.4731 ± 0.0193 | 0.4954 ± 0.0044 | +0.0222 |
| kNN in-corner balacc, deployed | 0.4993 ± 0.0220 | 0.5176 ± 0.0230 | +0.0183 |
| **LACE `ode` in-corner balacc** (the published row) | 0.6797 ± 0.0207 | **0.6873 ± 0.0259** | +0.0076 |
| LACE `ode` composition-B aggregate | 0.3102 ± 0.0444 | 0.3241 ± 0.0165 | +0.0139 |
| LACE `ode` refusal AUROC(C vs B) | 0.6981 ± 0.0444 | 0.7220 ± 0.0627 | +0.0239 |
| LACE `ld_published` refusal AUROC(C vs B) | 0.7154 ± 0.0510 | 0.8097 ± 0.0210 | +0.0943 |
| LACE `ld_published` A7 `out_of_range` AUROC | 0.4383 ± 0.1063 | 0.6267 ± 0.0441 | +0.1884 |
| `knn_proto` in-corner balacc (baselines) | 0.8884 ± 0.0088 | 0.8817 ± 0.0023 | −0.0067 |
| `knn_proto` in-corner balacc (cell, external) | 0.9119 ± 0.0222 | 0.9165 ± 0.0250 | +0.0046 |
| ⛔ **`knn_proto` composition-B (cell, external)** | 0.9005 ± 0.0182 | **0.8650 ± 0.0122** | **−0.0355** ← §6 |
| `lace` in-corner balacc (cell, external) | 0.6347 ± 0.0177 | 0.6278 ± 0.0289 | −0.0069 |
| `ebm_compose` in-corner balacc (cell) | 0.5408 ± 0.0087 | 0.5477 ± 0.0216 | +0.0069 |
| `ebm_compose` composition-B (cell) | 0.2130 ± 0.0132 | 0.1937 ± 0.0182 | −0.0193 |

⭐ **The seed spread collapses almost everywhere it moved** (kNN refusal sd 0.147 → 0.022; A7
`out_of_range` sd 0.130 → 0.005; `knn_proto` in-corner sd 0.0088 → 0.0023). ⇒ the broken `val` was not
only biased, it was **noisy**, and a balanced one buys a 3–6× tighter across-seed estimate.

### ⭐ MOVED — ours (the atom-mean control, i.e. the BAR the dynamical read must clear)
| row | ⛔ broken (published) | ⭐ repaired | Δ |
|---|---|---|---|
| read error on `A`, k=1, budget A → B | 0.4067 ± 0.0094 → 0.2306 ± 0.0038 | **0.3986 ± 0.0023 → 0.2306 ± 0.0038** | −0.0081 / **bit-identical** |
| read error on `A`, best k, budget A | **0.3567 ± 0.0319** | **0.3451 ± 0.0382** | −0.0116 (bar HARDER) |
| read error on `A`, best k, budget B | **0.1851 ± 0.0324** | **0.2062 ± 0.0347** | +0.0211 (bar EASIER) |
| **fitted exponent (k=1)** / implied `PR` | **−0.2727 ± 0.0141** / 7.33 | **−0.2632 ± 0.0095** / **7.60** | +0.0095 |
| distinct atom configurations on B, budget A | 24.33 ± 8.81 | 22.00 ± 5.72 | −2.33 |
| distinct atom configurations on B, budget B | 37.33 ± 11.61 | 28.67 ± 6.24 | −8.67 |
| in-corner shape balacc (external), budget A | 0.6849 ± 0.0202 | 0.6364 ± 0.0645 | −0.0484 |
| in-corner shape balacc (external), budget B | 0.6685 ± 0.0927 | 0.6997 ± 0.0400 | +0.0312 |

⛔ **`gen-atom-mean-control`'s three "handover updates" numbers all move and must be restated:** the bar
is **0.3451 ± 0.0382 (A) / 0.2062 ± 0.0347 (B)**, the exponent **−0.2632 ± 0.0095**, and the distinct
count **22.0 ± 5.7 → 28.7 ± 6.2 of 72**.

## 5. ⭐ THE REPAIR REMOVES A BIAS; IT DOES NOT BUY IDENTIFICATION

| halving | ⛔ broken split | ⭐ repaired split |
|---|---|---|
| contiguous halves of `val` disagree about the winner | **5 / 6** cells | 4 / 6 |
| ⭐ **interleaved (exchangeable) halves disagree** | — (not measurable: both halves degenerate) | ⛔ **5 / 6** |

Best-to-second gaps **0.0007–0.0130** against a trial spread of **0.068–0.111**. The grid is a plateau
and 256 rendered validation rows cannot resolve it. ⛔ **A tuned `gen` configuration is now *unbiased*
but still not *identified*** — the honest fix is `tune_n` / `n_val`, not the split, and that is a
budget decision for the Hub, ⛔ not one I made.

## 6. ⛔ STOP-AND-REPORT (acceptance item 5) — the full weakness audit, then the one real item

Rule applied is the one pre-registered (`PREREG` P5): a rival's **3-seed mean falling by more than one
OLD seed-sd**. ⛔ Run over **every** rival statistic where higher = stronger, not a chosen few
(`weakness_audit.py`): **156 statistics compared, 13 flagged.**

- **6 of 13** are LACE `ode`'s refusal AUROCs against the **A / A7** controls (e.g. `auroc_c_vs_a`
  0.7391 ± 0.0108 → 0.7067 ± 0.0123). Its **declared** refusal number, AUROC(C vs B), went **up**
  (+0.0239). ⚠ Worth a Hub look; ⛔ not the headline.
- **2 of 13** are the same `PER_CONTROL.A7.out_of_range` companion for kNN (0.8090 → 0.7559), while
  the **declared** control (`base`) for that kind went **up +0.1303**. Companion, not headline.
- **1 of 13** is `lace.distinct.MS.factor_tuples` 195.33 → 194.33 of 198. Immaterial.
- **3 of 13** are `ebm_compose` / `lace/ld_corrected` composition-B, −0.017 to −0.019 — inside the
  sampler-noise band the cell harness already carries.
- ⭐ **1 of 13 is real: `knn_proto`'s composition-B aggregate, −0.0355 (cell) / −0.0301 (baselines).**

**And the laundering control says it is NOT noise.** `knn_proto` has **no tuning axis**, so it can only
move via the `fit` rows. I ran **three further, equally valid stratified splits** (same rule, different
salt) at deployed budget, seed 0:

| split | in-corner balacc | composition-B | composition-A |
|---|---|---|---|
| ⛔ published (BROKEN head/tail) | 0.8763 | **0.7963** | 0.7842 |
| ⭐ the shipped repair (salt 20260914) | 0.8788 | 0.7755 | 0.7832 |
| alternative salt 20260915 | 0.8763 | 0.7778 | 0.7821 |
| alternative salt 20260916 | 0.8763 | 0.7755 | 0.7826 |
| alternative salt 20260917 | 0.8788 | 0.7755 | 0.7845 |
| **spread over the four stratified splits** | ± 0.0013 | **± 0.0010** | ± 0.0009 |

⇒ **the published 0.7963 is the outlier and ~0.776 is what *any* balanced `fit` gives.** The broken
`fit` withheld the highest-`floor_hue` rows — exactly the corner `B` lives in — and that flattered
`knn_proto`'s composition on `B`. So my reading is **a correction, not a hobbling**; and the
constraint that forces it is the task's own (`n_fit + n_val == 32,768` exactly, and ⛔ never shrink
`fit`, since at 1,024 rows `knn_proto` falls to 0.8308). **⛔ I am not clearing it on my own authority
— per the Head's ruling this is where I stop and hand the ruling to the Hub.**

⚠ **One flag that is NOT mine:** the cell harness's own shipped guard reports `lace` `regressed: true`
on seed 0 (Δ −0.0676 vs its published probe constant). ⭐ **It fired in the published run too**, at
Δ **−0.0840** — the repair made the pre-existing regression **less** severe, and the 3-seed mean is
0.6389 → 0.6399 (**+0.0011**, i.e. unchanged). ⛔ Inherited, not introduced.

## 7. How I verified (commands + real output)

- `uv`-free, main venv: `PYTHONPATH=<worktree> /Users/user/Desktop/CHLU/.venv/bin/python -m pytest …`
  **`tests/test_gen_rowsplit.py tests/test_gen_baselines.py tests/test_gen_cell.py
  tests/test_gen_atom_mean.py tests/test_gen_data_splits.py tests/test_gen_store.py
  tests/test_gen_whiten.py` → `206 passed, 0 failed, 0 skipped in 158.91s`** with
  `CHLU_GEN_DATA_ROOT=/Users/user/Desktop/CHLU/.claude/data/gen` (⛔ no full suite — 16 GB machine).
  Without the data root the real-data test *skips*, which is why the root is in the command.
- `ruff check chlu/ tests/test_gen_rowsplit.py` → **All checks passed!**
- The gate's positive control, on the **real 480,000-row grid**:
  `RowSplitError: ⛔ the fit/val split is NOT balanced … - floor_hue: val realises 1 of 10 values
  (fit realises 10)` — fires on 3/3 seeds when handed the shipped split.
- `knn_proto` bit-identity, atom-mean leg, against the re-based constants:
  `probe in-corner 0.8787878787878788 vs published 0.8787878787878788 -> Δ = +0, bit_identical=True`
  (3/3 seeds).
- All 15 runs `rc=0` (`runs/progress.txt`); `refusal_scores.json` `status=complete`, `not_run=[]`.

## 8. ⭐ PREREG SCORECARD (filed before the split was built; `gen-val-split-fix/PREREG.md`)

| # | pre-registered prediction | outcome | evidence |
|---|---|---|---|
| **P1** | nothing is bit-identical; the `knn_proto` bit-identity guard **FIRES** | ⚠ **SPLIT** — the guard fired as predicted (constants re-based, `ca14a34`), but **"nothing is bit-identical" is FALSE**: the whole `clu` arm, `store_removed`, and three atom-mean budget-B rows are Δ = 0.000000. ⭐ **That is the most load-bearing result in the report and I predicted against it.** | §4 |
| **P2** | mean \|move\| of any headline < 0.03, and < 1 seed-sd for ≥ 2/3 | ⚠ **FAILS on the upside** — kNN refusal AUROC(C vs B) moved **+0.2481**, ~8× my bound. I under-predicted how much a broken `val` distorts refusal. | §4 |
| **P3** | argmax moves in **2–5 of 6** cells; ⛔ not 0, ⛔ not all 6 | ⚠ **SPLIT** — rivals 5/6 (leg D) and 8/12 (leg H), inside the band; **our own atom-mean arm moved 6/6**, outside it. **P3 is falsified on our arm** and the finding is that our selection was *less* identified than the rivals'. | §3 |
| **P4** | `val` and `A` read-error columns converge to within ±0.10 | ✅ **HELD** — realised **0.025** (budget A) and **0.024** (budget B), from gaps of 0.46 / 0.21. | §1 |
| **P5** | `knn_proto` in-corner stays within ±0.02 per seed; 3-seed mean falls ≤ 0.01 | ✅ **HELD** on in-corner (per-seed +0.0025/−0.0075/−0.0151; mean −0.0067). ⛔ **But I pre-registered the wrong statistic** — composition-B, which I did not name, fell 0.0301–0.0355. §6. | §4, §6 |
| **P6** | whitening statistics move < 0.01; exactness booleans unchanged; the gate is decided by *which shrinkage is selected* | ✅ **HELD** — moves −0.0007 / +0.0046 / +0.0038; 0.10 classification unchanged for all three shrinkages; the gate flipped seeds purely because the selected shrinkage changed. | §3 |
| **P7** | structural guarantees untouched | ✅ **HELD** — `no_leak` 0/72, every row count, byte ledger and artefact hash bit-identical, 3/3. | §2 |
| **§4 balance** | `val` realises 10/10/10/8/4/15 at count-ratio ≤ 1.15 | ✅ **HELD** — realised ratio **≤ 1.091**, all six factors, 3/3 seeds. | §1 |
| **§4 variance** | `val` variance within ±10 % of `fit`'s | ✅ **HELD** — **+0.2 %** (0.2232 vs 0.2227). | §1 |

---

## Git footprint

| | |
|---|---|
| branch | `agent/experiment-engineer/gen-val-split-fix` (off `main @ 5a3c32c`; ⛔ not pushed, left for review) |
| commits | `e8cc1a2`, `ca14a34`, `6c693aa` — **verified from the MAIN repo**: `git log --oneline main..agent/experiment-engineer/gen-val-split-fix` lists all three |
| files | `chlu/data/gen_rowsplit.py` (**new**, +305) · `tests/test_gen_rowsplit.py` (**new**, +198) · `chlu/config.py` (+80/−2) · `chlu/experiments/exp_gen_atom_mean.py` (+~80/−~20) · `exp_gen_baselines.py` (+~52/−~12) · `exp_gen_cell.py` (+~33/−~5). **744 insertions / 43 deletions.** |
| ⭐ `chlu/config.py` discipline | the `BEGIN gen-val-split-fix` / `END` block is **one contiguous region appended at EOF with 0 deletions**. The **2** deletions in that file are a deliberate in-block edit of `GenAtomMeanConfig.knn_proto_probe_reference_per_seed` (§4/§6), with the superseded constants preserved in the comment. |
| rebase | `git rebase main` → "up to date" (base never moved) |
| conflicts | none |
| ⚠ worktree | **left in place** at `/Users/user/Desktop/CHLU-gen-valfix` (the branch ref is verified from the main repo above, so it *can* be removed — but `runs/run_rescore.sh` hard-codes that path, so removing it makes the re-score non-reproducible without re-creating it first) |

## Open questions / follow-ups / risks

1. ⛔ **The `knn_proto` composition-B ruling (§6) is the Hub's, not mine.**
2. ⭐ **Identification, not bias, is now the binding constraint (§5).** `tune_n=256` / `n_val=512`
   cannot resolve a 24–36-trial grid whose best-to-second gap is 0.0007. Costed guess: `tune_n=512`
   roughly doubles the atom-mean tuning cost (≈ +4 min/seed/budget).
3. ⚠ **`GenCellConfig` still points at the broken-split selection** (`lace_selection_source =
   "gen-arms-a7-wiring/runs/historical_688128"`) and still carries broken-split rival reference
   constants. I overrode the source **per-run** via `cell_rescore.yaml` rather than flipping a shared
   default another spoke may depend on. One-line change when the Hub blesses the new leg H.
4. ⚠ **Only Shapes3D / `recon` / Montero / `full` requests.** dSprites, the wide condition and the
   partial-request cell are ⛔ **NOT RUN** and inherit the repaired split the moment they are.
5. ⚠ Two cosmetic log strings changed between leg D and legs H/A/C (commit `6c693aa`); no number is
   affected, but seed-0 leg-H logs read slightly differently from seeds 1–2.
6. ⚠ The atom-mean/cell experiments append their own `seed_N` under `--out`, so their artefacts live
   at `runs/<leg>/seed_N/seed_N/`. My runner's skip-if-exists probe looked one level up, so those two
   legs were **not** resumable-by-skip (they were run once each, cleanly; noted so the next spoke
   does not trip on it).

---

## Proposed handover updates (for the Hub)

- **§7 RESOLVED (row construction — the standing HIGH item from `gen-atom-mean-control`).**
  `select_baseline_rows` and its two copies no longer cut `fit`/`val` from a sorted array.
  `chlu.data.gen_rowsplit` stratifies against the pool's own marginals; `val` realises **10/10/10/8/4/15**
  values at a max per-value count ratio of **≤ 1.091** on 3/3 seeds, feature variance **0.0924 → 0.2232**
  against `fit`'s 0.2227. ⛔ **Newly measured and worse than reported:** the broken split was also
  imbalanced on `object_hue` (count ratio **up to 3.52**), `orientation` (**1.88**) and `scale` (**1.46**).
- **§7 NEW (standing, HIGH — replaces the one above as the live worry).** Fixing the split removed the
  *bias*; it did **not** buy *identification*. On the exchangeable halving the two halves of `val` still
  pick different winners in **5 of 6** cells, with best-to-second gaps of 0.0007–0.0130 against a trial
  spread of 0.068–0.111. ⛔ No `gen` spoke may quote a tuned configuration as identified. The fix is
  `tune_n` / `n_val`, and it is unbudgeted.
- **§3 config default CHANGED.** `GenRowSplitConfig.mode` defaults to **`"stratified"`**
  (`gen_row_split_config()`, appended block at the tail of `chlu/config.py`, zero wiring sites).
  `"head_tail"` reproduces the broken split bit-for-bit, must be named, and shouts in the log.
- **§3 config constant RE-BASED.** `GenAtomMeanConfig.knn_proto_probe_reference_per_seed`
  → `[0.8787878787878788, 0.8844221105527639, 0.8819095477386935]` (old values kept in the comment).
  ⚠ **Not yet re-based:** `GenCellConfig.{clu,knn_proto,lace_ode}_probe_reference_per_seed` and
  `GenCellConfig.lace_selection_source`.
- **§3 (⛔ still a trap, unchanged).** `GenBridgeConfig.dec_recipe` **still defaults to `"baseline"`** —
  the recipe that FAILED the gate. This is the **sixth** module to carry a private guard against it.
- **Results to record (Shapes3D `recon`, Montero, seeds {0,1,2}, `v1_capacity_schedule` decoder,
  `gen-external-classifier`'s gated instrument, repaired stratified split):**
  - ⭐ **`clu` is BIT-IDENTICAL** — in-corner **0.6568 ± 0.0226**, composition-B **0.5949 ± 0.0082**,
    9.0 ± 0.8 distinct tuples, Δ = 0.000000 on 3/3. The repair re-priced the comparators only.
  - **kNN (bottlenecked)** refusal **AUROC(C vs B) 0.5976 ± 0.1159 → 0.8457 ± 0.0238**;
    A7-controlled `out_of_range` **0.8281 ± 0.1299 → 0.9583 ± 0.0048**.
  - **LACE `ode`** (the published composition row) in-corner **0.6797 ± 0.0207 → 0.6873 ± 0.0259**;
    composition-B 0.3102 ± 0.0444 → 0.3241 ± 0.0165. **R27.4 survives and tightens:** LACE
    `duplicate_factor` **0.9936 ± 0.0057 → 0.9906 ± 0.0033** vs kNN **0.3338 ± 0.0692 → 0.3103 ± 0.0013**.
  - ⛔ **`knn_proto` composition-B falls: 0.9005 ± 0.0182 → 0.8650 ± 0.0122** (cell) — reproducible
    across four independent stratified splits (spread ± 0.0010), so the **published value was the
    artefact**. **Hub ruling required before refusal/composition numbers are re-quoted.**
  - **The atom-mean BAR the dynamical read must clear is restated:** relative read error
    **0.3451 ± 0.0382 (budget A) / 0.2062 ± 0.0347 (budget B)**, exponent **−0.2632 ± 0.0095**
    (`PR` 7.60), distinct atom configurations **22.0 ± 5.7 → 28.7 ± 6.2** of 72.
  - **Refusal's quotable status:** the repaired-split per-request file is
    `gen-val-split-fix/refusal_scores.json` (`status: complete`, `not_run: []`, both byte legs × 3
    seeds). `gen-arms-a7-wiring/refusal_scores.json` is **superseded** for refusal claims.
  - **`gen-dynamical-read`'s ceiling row is now re-basable**; its scored run was blocked on this.
