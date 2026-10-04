# gen-atom-mean-resweep — experiment-engineer report

**Task + acceptance criterion:** sweep `n_val` and report the smallest value at which the
interleaved-half winner disagreement falls to **≤ 1 of 6** cells — or state explicitly that
none does — pricing the `n_fit` traded away at every row; plus one re-measurement of the
stale deployable exponent. **Status: done.** 18/18 production units `rc=0`, ⛔ nothing NOT RUN.

> ## ⛔ DOWNSTREAM RECONCILIATION LIST — 3 items, needs a Hub owner (protocol §5 corollary)
> 1. ⭐⭐ **THE GATE IN `gen-dynamical-read` PREREG §5b.0 IS NECESSARY BUT NOT SUFFICIENT,
>    AND I CAN SHOW IT.** `n_val = 4,096` **does** meet the pre-registered criterion
>    (**1 of 6**). But `n_val = 4,096` and `n_val = 8,192` — **two draws that BOTH pass the
>    gate** — pick a **different winner in 3 of 6 cells**. The interleaved-half test compares
>    two halves of the *same* draw (same rows, same `fit`, same atoms, same prototypes), so it
>    measures **within-draw row noise only** and is **blind to across-draw variation**.
>    ⛔ A rung may therefore pass §5b.0 and still be unidentified. **Owner: Hub — §5b.0 needs
>    a second clause.** (§4, §5)
> 2. ⭐⭐ **THE LADDER IS PARTIALLY IDENTIFIED, AND THE DECOMPOSITION SAYS WHICH HALF.**
>    Over all **36** cell-points, `scope = per_well` and `normaliser = none` are selected
>    **36/36 — unanimous at every `n_val`, including 512.** `k` and `metric` are **not**
>    identified at **any** `n_val` in range (4/6 cells stable even at `n_val ≥ 1024`).
>    ⇒ ⭐ a rung whose lever behaves like `scope`/`normaliser` is quotable **today**; a rung
>    whose lever behaves like `k`/`metric` is not quotable at any affordable `n_val`.
>    **Owner: Hub — this is the usable form of the answer.** (§4)
> 3. ⚠ **`stratified_fit_val` CLAMPS `n_val` TO `pool.size − n_fit`** — 512 at the shipped
>    `n_fit`. ⛔ Any spoke passing a larger `n_val` gets **512 rows and a completely normal
>    log**. Closed inside `exp_gen_atom_mean` (§2); ⛔ **`exp_gen_baselines` and
>    `exp_gen_cell` call the same function and are NOT fixed.** **Owner: Hub.** (§2)

---

## ⭐ DIAL DECLARATION (echoed, protocol §7)
- **Dial:** ⛔ **none — a CALIBRATION and one re-measurement.** No arm, no mechanism,
  ⛔ no verdict on any claim.
- **Laundering control:** the sweep is over **`n_val` only**; every other knob sits at the
  `gen-val-split-fix` leg-A value (shipped defaults, `--stage atoms`, no config file).
  ⭐ **`n_fit` is not a flag** — it is derived as `32,768 − n_val`, so ⛔ a starved `fit`
  could not be asked for even deliberately.
- **Falsifies:** potentially the whole six-rung tuning ladder. ⭐ It did not falsify it
  outright; it **bisected** it (item 2 above).
- **Does NOT falsify:** ⛔ `knn_proto` moving under a deliberately reduced `n_fit` is the
  **price of the validation rows**, ⛔ never a regression — it has no tuning axis.

## ⭐ FLAG PROVENANCE (protocol §5, mandatory)

| item | value |
|---|---|
| branch / base | `agent/experiment-engineer/gen-atom-mean-resweep`, base **`main @ 3eb9eb0`** (unmoved) |
| commit | **`7a47b0a`** — the only commit; working tree clean at report time |
| ⚠ runs executed at | **anchor leg at the uncommitted working tree; all five sweep points at `7a47b0a`.** ⛔ The tree and `7a47b0a` are byte-identical (the commit was taken between the two legs with no edit in between); HEAD echoed into `runs/progress.txt` by the runner |
| worktree / env | `/Users/user/Desktop/CHLU-resweep`; ⭐ **main venv reused, ⛔ no `uv sync`** (`PYTHONPATH=<worktree> /Users/user/Desktop/CHLU/.venv/bin/python`). `JAX_PLATFORMS=cpu`. `CHLU_CLAUDE_ROOT` / `CHLU_GEN_DATA_ROOT` / `CHLU_GEN_BRIDGE_ROOT` / `CHLU_GEN_DECLIFT_DIR` / `CHLU_GEN_STORE_RUNS` all set |
| ⛔ shared checkout | `/Users/user/Desktop/CHLU` was on another agent's branch `agent/experiment-engineer/pilot-ttt-nan-and-d5-wiring` and was **never touched** (protocol §3.2) |
| dataset / pretext / condition | `shapes3d` / `recon` / `recomb2element` |
| seeds | **0, 1, 2** at **every** point; ⛔ nothing NOT RUN |
| budgets | **A = 524,288 B** (64 atoms/well) · **B = 4,194,304 B** (512 atoms/well); byte equality asserted in-run, `matched_exactly=True`, `D = 2048` |
| ⭐ swept | `n_val ∈ {512, 1024, 2048, 4096, 8192}` with **`tune_n = n_val`**, plus the **anchor** `n_val = 512, tune_n = 256` (the shipped pair). Realised `n_tune` = 256/512/1024/2048/4096/8192, halves = 128/256/512/1024/2048/4096 |
| ⭐ derived, not asked for | realised `n_fit` = **32,256 / 32,256 / 31,744 / 30,720 / 28,672 / 24,576** = `32,768 − n_val` |
| NEW flags | `--n-val`, `--tune-n`, `--allow-knn-proto-move` (all **off** by default) |
| decoder | `decoder_source="winner"`, recipe **`v1_capacity_schedule` asserted by NAME and by param HASH per seed**: `01ef91f75eeaae7b…` / `001011bc603a7cb9…` / `2d7eea75add77044…`. ⛔ never the `"baseline"` default |
| instrument | external classifier per seed `c8741ac417d88ebf…` / `aa69b436989b2de5…` / `ad16c4e532731958…`; gate **PASS** (floor 0.95, 3 seeds); `refusal_scores.json` status=complete |
| split | `GenRowSplitConfig.mode="stratified"` (shipped), `tiebreak_jitter=1.0`, `seed_mult=6637`, `seed_offset=20260914`, `assert_balanced=True`, `require_all_values=True`, `max_count_ratio=1.25`; realised max per-value count ratio **1.072–1.083** at `n_val=8192` |
| everything else | shipped defaults: 24-trial grid (`k ∈ {1,3,5}` × `{euclidean,cosine}` × `{none,zscore}` × `{per_well,global}`), `atom_method="kmedoid"`, `atom_kmeans_iters=25`, `eval_batch=256`, `n_eval_a=512`, `run_subsample_companion=True` |
| ⛔ NOT used | **`--quick` (the CLI now REFUSES it beside `--n-val`/`--tune-n`)**, `--config`, any `chlu/config.py` edit, any full-suite run |
| ± convention | **population sd (ddof=0)** throughout |
| cost | **16 GB macOS box, CPU only, 1 process.** 18 units, **4 h 09 m wall** (15:53→20:05); per point 14.0 / 16.5 / 23.3 / 36.5 / 52.8 / **105.5** min |
| artefacts | `.claude/outputs/gen-atom-mean-resweep/` — `PREREG.md` (+ addendum A1), `tables.md`, `anchor_check.md`, `run_sweep.sh`, `assemble.py`, `check_anchor.py`, `cross_point.py`, `runs/{nval*_tune*}/seed_{0,1,2}/atom_manifest.json`, `runs/*.log`, `runs/progress.txt` |

---

## 1. ⭐⭐ THE ANSWER

> **Recommended `n_val` = 4,096** — the smallest value in range at which the pre-registered
> criterion (interleaved-half winner disagreement **≤ 1 of 6**) is met. It costs **11.1 %**
> of the fit rows (32,256 → 28,672) and ⛔ **costs the rivals nothing measurable** (§3).
>
> ⛔⛔ **AND THE RECOMMENDATION MUST NOT TRAVEL WITHOUT §4:** at `n_val = 4,096` the
> selection is **still not stable across independent draws**. The criterion is
> **necessary, not sufficient.**

### T1 — disagreement vs `n_val` (⛔ every cell, 3 seeds, 2 budgets)

| `n_val` | `tune_n` | rows/half | ⭐ **INTERLEAVED disagree** | contiguous | gap best→2nd | median gap | trial spread | realised `n_fit` | fit traded | σ_c(half) ± se | ⭐ **median gap / σ_c** | wall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 512 | 256 ⭐anchor | 128 | **5 / 6** | 4 / 6 | 0.0007–0.0130 | 0.0042 | 0.0678–0.1111 | 32,256 | 0 | 0.0061 ± 0.0010 | **0.69** | 14.0 m |
| 512 | 512 | 256 | **3 / 6** | 5 / 6 | 0.0020–0.0094 | 0.0028 | 0.0678–0.1061 | 32,256 | 0 | 0.0046 ± 0.0014 | **0.61** | 16.5 m |
| 1024 | 1024 | 512 | **3 / 6** | 5 / 6 | 0.0003–0.0042 | 0.0013 | 0.0686–0.1042 | 31,744 | 512 | 0.0023 ± 0.0006 | **0.56** | 23.3 m |
| 2048 | 2048 | 1024 | **4 / 6** | 4 / 6 | 0.0008–0.0044 | 0.0019 | 0.0669–0.1082 | 30,720 | 1,536 | 0.0032 ± 0.0004 | **0.60** | 36.5 m |
| **4096** | **4096** | 2048 | ⭐ **1 / 6** | 3 / 6 | 0.0002–0.0061 | 0.0020 | 0.0699–0.1043 | **28,672** | 3,584 | 0.0014 ± 0.0006 | **1.42** | 52.8 m |
| 8192 | 8192 | 4096 | **2 / 6** | 4 / 6 | 0.0002–0.0041 | 0.0015 | 0.0674–0.1063 | 24,576 | 7,680 | 0.0009 ± 0.0003 | **1.72** | 105.5 m |

⚠ **The count is a 6-cell statistic and 1-vs-2 is not a distinction it can support.** The
**resolution ratio** (median gap / σ_c) is the powered version of the same question, and it
is flat at **0.56–0.69** across 512→2048, then rises to **1.42 / 1.72**. ⭐ **The turn is
real and it happens between 2,048 and 4,096.** ⛔ But identification wants this ratio above
~2 and it never gets there: **at the top of an affordable grid the winner's margin is still
under two half-noise units.**

**The 3 disagreeing cells at the top two points, with their margins:**
`4096 s1/A` gap **0.0002** · `8192 s0/B` gap **0.0002** · `8192 s2/B` gap **0.0008**.
⭐ Every one is a near-exact tie — ⛔ these are not "wrong" selections, they are **ties being
broken by noise**, which is the honest description of the whole phenomenon.

---

## 2. ⛔⛔ THE TRAP THAT WOULD HAVE VOIDED THIS SWEEP (found by a test, before any run)

`chlu/data/gen_rowsplit.py::stratified_fit_val` clamps in this order:

```
n_fit = min(n_fit, pool.size)
n_val = min(n_val, max(pool.size - n_fit, 0))     # <- 32,768 - 32,256 = 512
```

⇒ ⛔ **`--n-val 8192` at the shipped `n_fit` cuts 512 rows, selects on 512 rows, and logs
entirely normally.** The whole disagreement curve would have been **flat for a plumbing
reason** and I would have reported a plateau that was my own bug — the `--quick` failure
mode the task file warns about, in a different file.

Caught by `test_raising_n_val_COSTS_fit_rows_arithmetically`, written to pin PREREG §3 P5's
arithmetic, **before the first sweep run**. Closed by:
- `fit_rows_for_val()` — derives `n_fit = n_pool − n_val` so the asked-for cut survives.
  ⭐ `n_fit` stays **un-flaggable**: the caller names val rows, the fit cost is arithmetic.
- **`MIN_FIT_ROWS = 2,048`** — a hard refusal, 2× the 1,024 rows where `knn_proto` fell.
- `run_seed` **raises** if the realised `val` size ≠ the asked-for `n_val`.
- ⚠ **PREREG §1 said "`stratified_fit_val` cuts `val` first"; that was wrong about the
  order and is corrected in ADDENDUM A1**, filed with zero sweep runs on disk.

⛔ **`exp_gen_baselines` and `exp_gen_cell` call the same clamped function** and are not in
my scope. Reconciliation item 3.

---

## 3. ⭐ THE PRICE — and ⛔ it is NOT a hobbled rival

`knn_proto` is the instrument: it has **no tuning axis**, so anything it does under a reduced
`n_fit` is the `fit`-row swap and nothing else.

| `n_val` | realised `n_fit` | % of 32,256 | s0 | s1 | s2 | mean ± sd | Δ vs published |
|---|---|---|---|---|---|---|---|
| 512 / 256 | 32,256 | 100.0 % | 0.8788 | 0.8844 | 0.8819 | **0.8817 ± 0.0023** | **+0.0000** ✅ bit-identical |
| 512 / 512 | 32,256 | 100.0 % | 0.8788 | 0.8844 | 0.8819 | **0.8817 ± 0.0023** | **+0.0000** ✅ bit-identical |
| 1024 | 31,744 | 98.4 % | 0.8763 | 0.8869 | 0.8844 | 0.8825 ± 0.0046 | +0.0008 |
| 2048 | 30,720 | 95.2 % | 0.8763 | 0.8894 | 0.8794 | 0.8817 ± 0.0056 | −0.0000 |
| **4096** | **28,672** | **88.9 %** | 0.8687 | 0.8794 | 0.8844 | **0.8775 ± 0.0066** | **−0.0042** |
| 8192 | 24,576 | 76.2 % | 0.8838 | 0.8920 | 0.8719 | 0.8826 ± 0.0083 | +0.0008 |

⭐ **Every point stays at 0.877–0.883** against a reference of 0.8817 — and against the
**0.8308** that a genuinely starved `fit` (1,024 rows) produces. The largest move,
**−0.0042 at `n_val = 4,096`**, is **smaller than that point's own across-seed sd (0.0066)**
and one tenth of the starving signature. ⭐ The lowest `n_fit` reached, **24,576**, is **24×
above** the row count at which the rival degrades. ⇒ ⛔ **The identification was not bought
by hobbling the comparators.** (PREREG **P5 CONFIRMED**.)

⚠ Reported honestly: the across-seed **sd grows monotonically** with the trade
(0.0023 → 0.0083). Fewer fit rows buy a noisier rival even where they do not buy a weaker
one. ⛔ Not a degradation; ⭐ a cost, and it is on the record.

---

## 4. ⭐⭐ THE FINDING THAT MATTERS MORE THAN THE NUMBER

### (a) Two draws that both pass the gate disagree with each other

| cell | `n_val` 4,096 | `n_val` 8,192 | |
|---|---|---|---|
| s0/A | k3/euc/none/per_well | k3/**cos**/none/per_well | ⛔ **DIFFER** |
| s0/B | k3/cos/none/per_well | k3/**euc**/none/per_well | ⛔ **DIFFER** |
| s1/A | k5/cos/none/per_well | k5/cos/none/per_well | ✅ match |
| s1/B | k1/cos/none/per_well | k1/cos/none/per_well | ✅ match |
| s2/A | k3/cos/none/per_well | k3/cos/none/per_well | ✅ match |
| s2/B | **k1**/cos/none/per_well | **k3**/cos/none/per_well | ⛔ **DIFFER** |

⇒ ⛔⛔ **3 of 6.** The interleaved-half test splits **one** draw — same rows, same `fit`,
same atoms, same prototypes — so it prices **row-sampling noise within a draw** and is
**structurally blind** to the variation between draws. ⭐ **A rung can pass §5b.0's gate and
still be unidentified**, which is exactly the failure the gate was written to prevent.

### (b) But the ladder is not dead — it is **bisected**, and cleanly

**Per-axis identification, over all 36 cell-points:**

| axis | stable over **all 6** points | stable over the 4 points `n_val ≥ 1024` | verdict |
|---|---|---|---|
| `normaliser` | **6 / 6** | **6 / 6** | ✅ **IDENTIFIED — `none` chosen 36/36, unanimous** |
| `scope` | **6 / 6** | **6 / 6** | ✅ **IDENTIFIED — `per_well` chosen 36/36, unanimous** |
| `k` | 2 / 6 | 4 / 6 | ⛔ **NOT identified at any `n_val` in range** |
| `metric` | 3 / 6 | 4 / 6 | ⛔ **NOT identified at any `n_val` in range** |

Within-draw, decomposed — ⭐ the whole-tuple disagreement is **entirely** `k` and `metric`:

| `n_val` | `k` | `metric` | `normaliser` | `scope` | whole tuple |
|---|---|---|---|---|---|
| 512/256 | 3/6 | 3/6 | **0/6** | **0/6** | 5/6 |
| 512/512 | 3/6 | 1/6 | **0/6** | **0/6** | 3/6 |
| 1024 | 3/6 | 3/6 | **0/6** | **0/6** | 3/6 |
| 2048 | 1/6 | 4/6 | **0/6** | **0/6** | 4/6 |
| 4096 | 1/6 | 1/6 | **0/6** | **0/6** | **1/6** |
| 8192 | 1/6 | 1/6 | **0/6** | **0/6** | 2/6 |

⇒ ⭐⭐ **THE USABLE FORM OF THE ANSWER.** `scope` and `normaliser` are identified at the
**shipped** `n_val = 512` and never wobble once — ⛔ the calibration was never needed for
them. `k` and `metric` are coin flips at every `n_val` this box can afford. ⇒ **a ladder rung
is quotable if its lever separates the grid like `scope` does (`per_well` 0.60 vs `global`
0.38 — a ~0.22 gulf), and not quotable if it separates it like `metric` does (0.0002).**
⛔ The question "is six rungs a measurement?" has no single answer; **it depends on the rung**,
and §5b.0's per-rung quotability clause is the right instrument — ⭐ it just needs the
cross-draw check of (a) added to it.

---

## 5. ⭐ PREREG SCORECARD (filed before the sweep; `gen-atom-mean-resweep/PREREG.md`)

| # | prediction | measured | verdict |
|---|---|---|---|
| **P1** | the anchor reproduces `gen-val-split-fix` leg A bit-identically | **78 / 78 comparisons matched, 0 moved** — all 6 cells' selected tuples, both argmaxes, both agreement booleans, both gap/spread values, and **all 24 trials × 5 score columns** (max\|Δ\| = 0.00e+00) | ✅ **CONFIRMED** — ⭐ the one-pass `_trial_scores` moved nothing, so the sweep's numbers are the module's |
| **P2** | σ_c = 0.0061·√(256/`n_val`), within ±30 % | 0.0046 (+7 %) · 0.0023 (−26 %) · **0.0032 (+45 %)** · 0.0014 (−7 %) · 0.0009 (−18 %) | ◐ **4 of 5 in band; `n_val`=2048 misses at +45 %** (≈2.5 se). ⛔ Reported as a miss, not rounded into the band |
| **P3** ⭐primary | plateau at ≈ 4 of 6; ⛔ **no `n_val` reaches ≤ 1 of 6**; confidence 60 % | **4,096 reached 1 of 6**; the resolution ratio turned 0.60 → 1.42 | ⛔ **FALSIFIED.** ⭐ My primary hypothesis was wrong and the grid's top half is where it broke |
| **P4** rival | ≤ 1 of 6 at `n_val` = **2048**; recommend 2048 | 2048 gave **4 of 6** | ⛔ **also wrong** — right that a resolvable point exists, wrong by one grid step |
| — | pre-registered **NEITHER** branch: "not ≤1 at 2048 **and** not ≥3 at 8192 ⇒ report as neither, ⛔ do not spin" | 2048 → 4 of 6 · 8192 → 2 of 6 | ⭐ **THIS BRANCH FIRED.** Reported as written: **neither hypothesis survives**, and the honest statement is §1 + §4, ⛔ not a claimed win for either |
| **P5** | `knn_proto` > 0.86 everywhere; \|Δ\| < 0.01 at `n_val` ≤ 2048, < 0.025 at 8192 | 0.8775–0.8826 everywhere; Δ = +0.0000 / +0.0000 / +0.0008 / −0.0000 / −0.0042 / +0.0008 | ✅ **CONFIRMED** — ⛔ no starving |
| **P6** | anchor reproduces split-fix's repaired exponent **−0.153105 / −0.166621 / −0.153665** | **exactly those three**, bit-identical | ✅ **CONFIRMED** |

⭐ **A pre-registered prediction that fails is a finding.** P3 was my primary and it is dead:
I predicted a plateau and the curve turned. ⛔ What replaced it is *not* P4 either — the
pre-registered "neither" branch fired, and §4 is what the data actually supports.

---

## 6. ⭐ THE ONE STALE NUMBER — ⛔ SUSPENDED, NON-ADJUDICATING

The **deployable (request-addressed, k = 1) exponent**:

| | s0 | s1 | s2 | **mean ± sd** |
|---|---|---|---|---|
| ⛔ **OLD — published, BROKEN split** (`gen-atom-mean-control` §4(d)) | −0.165481 | −0.165334 | −0.154152 | **−0.1617 ± 0.0053** |
| ⭐ **NEW — repaired stratified split** | −0.153105 | −0.166621 | −0.153665 | **−0.1578 ± 0.0062** |

**Δ = +0.0039**, inside one seed-sd. ⭐ Stable across the whole sweep (−0.1578 / −0.1588 /
−0.1580 / −0.1578 / −0.1601 / −0.1592 at the six points) — ⛔ this exponent does **not**
depend on `n_val`, which is expected: it is fitted on `A`, and `A` never enters selection.

⛔⛔ **EXPONENT STUDIES ARE SUSPENDED (Head, 2026-09-15). This number buys no claim,
adjudicates nothing, and I draw no conclusion from it.** It is here so the record is clean
into the rebuttal. Companion legs, same status: `request_addressed/k_selected`
**−0.1334 ± 0.0438** (⚠ sd 0.044 — ⛔ not quotable), `oracle_feature_addressed/k1`
**−0.2632 ± 0.0095** (unchanged from split-fix).

---

## 7. How I verified

```
# tests (targeted; ⛔ no full suite — 16 GB box)
PYTHONPATH=… CHLU_GEN_DATA_ROOT=…/.claude/data/gen JAX_PLATFORMS=cpu \
  .venv/bin/python -m pytest tests/test_gen_atom_mean.py tests/test_gen_rowsplit.py -q
  -> 47 passed, 0 failed, 0 skipped (15.75 s)
     (test_gen_atom_mean.py alone: 38 passed — 30 pre-existing + 8 new)
# the sweep
.claude/outputs/gen-atom-mean-resweep/run_sweep.sh all
  -> 18/18 units rc=0, 15:53:37 -> 20:04:54  (runs/progress.txt)
# P1/P6 control
.venv/bin/python …/check_anchor.py   -> 78 comparisons, 78 matched
```

**New tests (8):** `test_trial_scores_match_score_images_BIT_IDENTICALLY` (zero tolerance,
odd row count) · `test_the_two_halvings_are_DIFFERENT_slicings` (⛔ a slice typo would
fabricate agreement) · `test_stratified_split_CLAMPS_n_val_to_the_fit_remainder` (§2's trap)
· `test_raising_n_val_COSTS_fit_rows_arithmetically` (+ the `MIN_FIT_ROWS` refusal) ·
`test_the_shipped_production_pair_is_a_NO_OP_for_the_trade` ·
`test_quick_REFUSES_the_sweep_flags_it_would_clobber` ·
`test_the_knn_proto_WAIVER_prices_the_move_instead_of_aborting`.

## Git footprint
- **Branch** `agent/experiment-engineer/gen-atom-mean-resweep`, base `main @ 3eb9eb0`
  (unmoved; nothing to rebase). Worktree `/Users/user/Desktop/CHLU-resweep`, **left in place
  for Hub review**; branch ref verified from the main repo (protocol §3.2).
- **Commit `7a47b0a`** — files touched, ⛔ exactly the two I own:
  `chlu/experiments/exp_gen_atom_mean.py` (+3 CLI flags, `fit_rows_for_val`, `MIN_FIT_ROWS`,
  `_trial_scores`, realised-cut assertion, `n_tune_realised` in the manifest, two hard-coded
  "32256/512" log strings parameterised) · `tests/test_gen_atom_mean.py` (+8 tests).
- ⛔ **`chlu/config.py` NOT touched** (peer spoke owns it). ⛔ `chlu/data/gen_rowsplit.py`
  NOT touched. ⛔ Nothing pushed, no PR. Working tree clean.

## Open questions / follow-ups / risks
1. ⭐⭐ **The §5b.0 gate needs a cross-draw clause** (reconciliation 1). The cheap form: run
   each rung's selection at **two independent `n_val` draws** and require the winner to
   match — my 4,096 / 8,192 pair is exactly that test and it fails 3 of 6.
   ⚠ **Caveat on my own evidence:** those two points differ in `n_fit` as well as in val
   rows (28,672 vs 24,576), so "different draw" and "different fit set" are **confounded**
   here. ⛔ The clean version is two `n_val = 4,096` draws at **different split salts**, same
   `n_fit` — ⭐ ~53 min, and I did not run it because it is outside this task's grid.
2. ⚠ **The `n_val` = 2048 σ_c miss (+45 %, P2)** is unexplained. Most likely the estimator's
   own 6-cell noise plus the fact that the top-2 *pair* changes between points, so σ_c is not
   a fixed quantity. ⛔ Not investigated; flagged rather than explained away.
3. ⚠ **`n_val = 8,192` costs 105 min for 3 seeds and gives a worse count than 4,096.**
   ⛔ I do not recommend the grid be extended upward: the resolution ratio is rising ~√`n_val`
   and reaching 2.0 would need `n_val ≈ 11,000`, i.e. `n_fit ≈ 21,700` and ~2.5 h/point —
   ⭐ and §4(a) says the criterion it would satisfy is the wrong criterion anyway.
4. ⚠ **`--allow-knn-proto-move` is a waiver and waivers rot.** It is off by default, refused
   in combination with nothing, and asserted off by a test — but ⛔ a future spoke that
   passes it without moving `n_fit` gets a silently unguarded run.

---

## Proposed handover updates (for the Hub)

- ⭐⭐ **`n_val` CALIBRATION ANSWERED. Recommended `n_val` = 4,096** (smallest in range
  meeting ≤ 1 of 6; **1 of 6**), at 88.9 % of the fit rows and ⛔ **no measurable cost to any
  rival** (`knn_proto` 0.8775 ± 0.0066 vs 0.8817 reference; the starving signature is 0.8308).
  ⛔ **Never quote it without the clause in the next bullet.**
- ⭐⭐ **THE GATE IS NECESSARY, NOT SUFFICIENT.** `n_val` 4,096 and 8,192 — **both passing
  the gate** — select **different winners in 3 of 6 cells**. The interleaved-half test splits
  one draw and is blind to across-draw variation. **§5b.0 needs a cross-draw clause.**
- ⭐⭐ **THE LADDER IS BISECTED, NOT DEAD.** `scope` (`per_well`) and `normaliser` (`none`)
  are **unanimous 36/36 at every `n_val` including 512** — identified, and the calibration was
  never needed for them. `k` and `metric` are **not identified at any affordable `n_val`**;
  the whole-tuple disagreement is **entirely** those two axes (`normaliser`/`scope` contribute
  **0/6 at every single point**). ⇒ **rung quotability is a per-rung question**, and §5b.0's
  own per-rung clause is the right instrument once (2) is added.
- ⚠ **`stratified_fit_val` silently clamps `n_val` to `pool.size − n_fit`.** Fixed inside
  `exp_gen_atom_mean`; ⛔ **`exp_gen_baselines` and `exp_gen_cell` share the defect.**
- ⭐ **The last stale number is cleared.** Deployable (request-addressed, k=1) exponent
  **−0.162 ± 0.005 (broken) → −0.1578 ± 0.0062 (repaired)**, Δ +0.0039, inside one seed-sd.
  ⛔ **SUSPENDED, non-adjudicating.**
- ⛔ **PREREG P3 (mine, primary) was FALSIFIED and P4 (the rival) was also wrong**; the
  pre-registered "neither" branch fired. ⭐ Recorded as a finding, ⛔ not reshaped into a win.
