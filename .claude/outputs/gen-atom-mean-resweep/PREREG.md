# PREREG — `gen-atom-mean-resweep` (the `n_val` calibration)

**Filed:** 2026-09-15, **before** any sweep run was launched.
**Agent:** `experiment-engineer` · **Branch:** `agent/experiment-engineer/gen-atom-mean-resweep`
**Base:** `main @ 3eb9eb0` · **Worktree:** `/Users/user/Desktop/CHLU-resweep`
**Filed at working tree:** code changes to `chlu/experiments/exp_gen_atom_mean.py` +
`tests/test_gen_atom_mean.py` complete and linted; ⛔ **zero sweep runs executed.**

---

## ⭐ DIAL DECLARATION (echoed, protocol §7)
- **Dial:** ⛔ **none — a CALIBRATION and one re-measurement.** No arm, no mechanism,
  ⛔ no verdict on any claim.
- **Laundering control:** the sweep is over **`n_val` only**. Every other knob stays at
  the `gen-val-split-fix` leg-A value (shipped `GenAtomMeanConfig` defaults, `--stage
  atoms`, no config file), so a change in disagreement cannot be attributed to anything
  else. ⭐ **`n_fit` is not even a flag** — `stratified_fit_val` cuts `val` first and
  gives `fit` the remainder, so the realised `n_fit` falls out **arithmetically** as
  `32,768 − n_val` and ⛔ nobody gets to ask for a starved `fit` by name.
- **Falsifies:** ⭐ potentially this programme's whole tuning ladder (`gen-dynamical-read`
  PREREG §5b.0). If no affordable `n_val` reaches ≤ 1-of-6, the six-rung ladder is not a
  measurement.
- **Does NOT falsify:** ⛔ the disagreement failing to reach ≤ 1-of-6 is **a finding, not
  a failed task** — the task file says so in its own words. ⛔ `knn_proto` moving under
  a deliberately reduced `n_fit` is the **price of the validation rows**, ⛔ never a
  regression: it has no tuning axis, so anything it does is the `fit`-row swap.

---

## 1. What is being measured, in one line

**The smallest `n_val` at which the interleaved-half winner disagreement over the six
(seed, budget) cells of `exp_gen_atom_mean` falls to ≤ 1 of 6** — or an explicit
statement that no `n_val` in `{512, 1024, 2048, 4096, 8192}` achieves it.

**Grid (⛔ may not grow after it runs):** `n_val ∈ {512, 1024, 2048, 4096, 8192}` with
`tune_n = n_val` (⭐ selection renders **every** cut val row), 3 seeds `{0,1,2}`,
2 budgets `{A = 524,288 B, B = 4,194,304 B}` ⇒ 6 cells per `n_val` point.
**Plus one ANCHOR:** `n_val = 512, tune_n = 256` — ⭐ the shipped pair, i.e. exactly the
`gen-val-split-fix` leg-A configuration, run again through my harness.

⚠ **`tune_n` must track `n_val` or the sweep measures nothing.** `run_budget` uses
`n_tune = min(tune_n, n_val)`; at the shipped `tune_n = 256` the selection would still see
256 rows at every point and the curve would be flat **for a plumbing reason**.
⇒ ⭐ the anchor and the `n_val = 512` sweep point are **different cells** (256 vs 512
rendered rows) and are reported as such.

---

## 2. ⭐ WHERE THE PREDICTIONS COME FROM — the existing n_val = 512 evidence

⛔ Derived from `gen-val-split-fix/runs/atom_mean/seed_{0,1,2}/seed_{0,1,2}/
atom_manifest.json` — **already on disk**, read before this file was written. Nothing
below is a guess dressed as a derivation.

At the shipped configuration (`tune_n = 256` ⇒ **128 rows per interleaved half**):

| cell | best | 2nd | `c_full` = gap(1,2) | `c_ihalf1` | `c_ihalf2` | trial spread | ihalves agree |
|---|---|---|---|---|---|---|---|
| s0/A | 0.5964 | 0.5833 | **0.0130** | +0.0156 | +0.0104 | 0.0970 | ✅ **True** |
| s0/B | 0.6491 | 0.6419 | 0.0072 | 0.0000 | +0.0143 | 0.0678 | ❌ |
| s1/A | 0.5814 | 0.5768 | 0.0046 | −0.0026 | +0.0117 | 0.0942 | ❌ |
| s1/B | 0.6628 | 0.6621 | **0.0007** | +0.0052 | −0.0039 | 0.0721 | ❌ |
| s2/A | 0.6250 | 0.6211 | 0.0039 | +0.0013 | +0.0065 | 0.1111 | ❌ |
| s2/B | 0.6875 | 0.6836 | 0.0039 | −0.0013 | +0.0091 | 0.0842 | ❌ |

⇒ **5 of 6 disagree**, gaps **0.0007–0.0130**, spread **0.068–0.111**. ⭐ Reproduces the
numbers `gen-val-split-fix` §5 and `gen-dynamical-read` PREREG §5b.0 quote.

⭐ **The quantity that actually decides an argmax**, and the one I model, is **not** the
per-trial noise but the noise on the **top-2 contrast** `c = score(best) − score(2nd)`
evaluated on the *same* rows (the trials are strongly correlated — same renders, same
rows — so the contrast noise is far smaller than `√2 σ_trial`). Its two independent
128-row estimates are `c_ihalf1`, `c_ihalf2` above, so with
`E|X−Y| = 2σ/√π` for iid normals:

```
mean |c_ihalf1 − c_ihalf2| = 0.0098
  =>  sigma_c(m = 128 rows/half) = 0.0098 * sqrt(pi) / 2 / sqrt(2) = 0.0061
  =>  sigma_c(m)  =  0.0061 * sqrt(128 / m),   m = n_val / 2
```

For reference the *per-trial* half-score noise is `sigma_trial(128) = 0.0089`
(mean over cells of `sd(ihalf1 − ihalf2)/√2` across the 24 trials) — ⛔ the larger,
wrong ruler, recorded so the two are never confused.

**Agreement criterion used below:** a cell's two halves pick the same winner when the
*true* contrast clears the half-noise, `c_true > 2·sigma_c(m)` (each half then places the
true best on top with ≈ 0.98, both with ≈ 0.95).

**Per-cell `c_true`, de-biased** by `c_true = sqrt(max(c_full² − sigma_c(256)², 0))`
with `sigma_c(256) = 0.0043` — ⚠ still an **upper** bound, because `c_full` is a max over
24 trials and therefore selection-inflated:

| cell | `c_full` | de-biased `c_true` |
|---|---|---|
| s0/A | 0.0130 | **0.0123** |
| s0/B | 0.0072 | **0.0058** |
| s1/A | 0.0046 | 0.0016 |
| s2/A | 0.0039 | ~0 |
| s2/B | 0.0039 | ~0 |
| s1/B | 0.0007 | ~0 |

⇒ **two cells have a real winner; three or four are statistically tied at this
resolution.** That split is what the two competing hypotheses below disagree about.

---

## 3. ⛔ THE PREDICTIONS — committed before the harness ran

### P1 — the ANCHOR reproduces `gen-val-split-fix` leg A, exactly
At `n_val = 512, tune_n = 256`: **5 of 6** cells disagree; the six selected tuples,
the six `gap_best_to_second` values (0.0130 / 0.0072 / 0.0046 / 0.0007 / 0.0039 / 0.0039)
and the six trial spreads (0.0970 / 0.0678 / 0.0942 / 0.0721 / 0.1111 / 0.0842) come back
**bit-identically**, and `n_fit = 32,256`.
⛔ **Any mismatch means my one-pass scorer (`_trial_scores`) or my harness moved
something, and every number downstream of it is void.** This is the control on me.

### P2 — the noise law: `sigma_c ∝ n_val^(−1/2)`
Measured as the half-to-half spread of the top-2 contrast at each point.

| `n_val` | rows/half | **predicted `sigma_c`** |
|---|---|---|
| 512 | 256 | 0.0043 |
| 1024 | 512 | 0.0031 |
| 2048 | 1024 | 0.0022 |
| 4096 | 2048 | 0.0015 |
| 8192 | 4096 | 0.0011 |

**Predicted within ±30 % at every point.** ⛔ If `sigma_c` falls *slower* than
`n_val^(−1/2)`, a non-sampling term dominates — the first candidate being the `fit` rows
the val rows cost (P5), which would mean the sweep is partly measuring its own price.

### P3 — ⭐⭐ THE HEADLINE, and it is the pessimistic branch
Applying `c_true > 2·sigma_c(n_val)` to the de-biased column of §2:

| `n_val` | `2·sigma_c` | cells clearing it | **predicted cells DISAGREEING (of 6)** |
|---|---|---|---|
| 512 (anchor, `tune_n` 256) | 0.0086 | s0/A | **5** |
| 512 (`tune_n` 512) | 0.0086 | s0/A | **4–5** |
| 1024 | 0.0061 | s0/A | **4–5** |
| 2048 | 0.0043 | s0/A, s0/B | **4** |
| 4096 | 0.0031 | s0/A, s0/B | **4** |
| 8192 | 0.0022 | s0/A, s0/B | **4** |

⭐ **PRIMARY PREDICTION: the curve turns once — between 512 and 2048, recovering the one
cell (s0/B) whose winner is real but marginal — and then PLATEAUS at ≈ 4 of 6.**
⇒ ⛔⛔ **NO `n_val` in `{512 … 8192}` reaches ≤ 1 of 6.** Confidence ≈ 60 %.
⇒ ⭐ **the answer the task calls legitimate and important is the one I expect to give**,
and the mechanism is named in advance: **three or four of the six cells are genuine
plateaus — several grid configurations are truly tied, and no number of validation rows
resolves a tie.**

### P4 — the RIVAL hypothesis, pre-registered so it can win
**H-noise-pure:** the grid *does* have a unique best in every cell and every small
observed gap is sampling noise, so `c_true ≈ 0.0052` (the de-biased RMS over cells)
everywhere. Then `2·sigma_c(n_val) < 0.0052` at `n_val ≥ 1,534` and:
⇒ **≤ 1 of 6 is reached at `n_val = 2048`**, and the recommended `n_val` is **2048**.

**The discriminating measurement, stated now:** whether the disagreement count is
**≤ 1 of 6 at `n_val = 2048`** (⇒ P4, a recommendable `n_val` exists) or **≥ 3 of 6 at
`n_val = 8192`** (⇒ P3, the ladder is unidentifiable as designed). ⛔ An outcome between
the two (2 of 6 at 8192, still falling) is **neither** and is reported as "the curve is
still falling at the top of the grid; the grid was too small" — ⛔ not spun as either.

### P5 — ⭐ THE PRICE, and the falsifier for "you manufactured identification"
Realised `n_fit = 32,768 − n_val` (**arithmetic, not a choice**):

| `n_val` | **realised `n_fit`** | fit rows traded away | % of shipped |
|---|---|---|---|
| 512 | 32,256 | 0 | 0.0 % |
| 1024 | 31,744 | 512 | 1.6 % |
| 2048 | 30,720 | 1,536 | 4.8 % |
| 4096 | 28,672 | 3,584 | 11.1 % |
| 8192 | 24,576 | **7,680** | **23.8 %** |

`knn_proto` is the instrument that prices this: it has **no tuning axis**, so anything it
does under a reduced `n_fit` is the `fit`-row swap and nothing else. Reference: the split
repair swapped **1.55 %** of `fit` and moved it **0.8884 → 0.8817 (−0.0067)**; at
**1,024** fit rows it fell **0.8763 → 0.8308 (−0.0455)**.

**Predicted:** `knn_proto` probe in-corner balacc stays **> 0.86 at every point**, with
|Δ| < 0.01 at `n_val ≤ 2048` and |Δ| < 0.025 at `n_val = 8192` (log-interpolating the two
reference points: 24,576 rows is 24× more than the 1,024 that cost 0.0455).
⛔ **FALSIFIED — and the whole sweep is then reported as having bought identification by
hobbling the comparators — if `knn_proto` drops below 0.86 at any point that I would
otherwise recommend.** ⛔ The lowest `n_fit` reached is **24,576**, which is **24× above**
the 1,024 rows where the rival degraded; ⛔ the sweep stops at 8,192 and does not go on.

### P6 — the one stale number (⛔ a TRANSCRIPTION CHECK, not a forward prediction)
The **deployable (request-addressed) `k = 1` exponent**. Published **−0.162 ± 0.005** is a
**broken-split** value (`gen-atom-mean-control` §4(d); per-seed **−0.165481 / −0.165334 /
−0.154152** ⇒ **−0.1617 ± 0.0053**, ddof = 0, read from that spoke's manifests).
`gen-val-split-fix` leg A **already re-ran this on the repaired split** but never
extracted it, so I am **not** predicting a new number — I am predicting that my anchor
**reproduces theirs bit-identically**:
> **−0.153105 / −0.166621 / −0.153665 ⇒ −0.1578 ± 0.0062.**

⛔ Any deviation means my harness moved something (same control as P1).
⛔⛔ **EXPONENT STUDIES ARE SUSPENDED (Head, 2026-09-15). This number buys no claim,
adjudicates nothing, and ⛔ no conclusion is drawn from it. It exists so the record is
clean into the rebuttal.**

---

## 4. ⛔ What would prove this spoke wrong (task file's own list, answered in advance)

| the charge | the measurement that answers it | where it lands |
|---|---|---|
| "the disagreement never falls" | P3's curve. ⛔ If flat at 5-of-6 from 512 to 8192, the grid is a plateau, ⭐ **report it plainly — it kills the ladder as designed and is a finding** | §3 P3 |
| "you bought val rows by starving `fit`" | P5: realised `n_fit` at every row **and** `knn_proto` re-measured at every point. ⭐ `n_fit` is not a flag, so I could not have starved it deliberately | §3 P5 |
| "you used `--quick`" | ⛔ `--quick` now **raises SystemExit** if combined with `--n-val`/`--tune-n` (`test_quick_REFUSES_the_sweep_flags_it_would_clobber`). ⛔ No sweep run uses it | tests |
| "you let `dec_recipe` default" | `decoder_source = "winner"` + `expected_decoder_recipe = "v1_capacity_schedule"` asserted **by name and by param hash** per seed in `resolve_decoder`; the three hashes are echoed in the report's provenance table | §5 |
| "the one-pass scorer changed a number" | P1 + `test_trial_scores_match_score_images_BIT_IDENTICALLY` (tolerance exactly zero, odd row count) | §3 P1 |
| "the two halvings are secretly the same slicing" | `test_the_two_halvings_are_DIFFERENT_slicings` — ⛔ a typo there would fabricate agreement | tests |

---

## 5. Provenance committed in advance

- **Env:** main venv (`/Users/user/Desktop/CHLU/.venv/bin/python`), ⛔ **no `uv sync`**;
  `PYTHONPATH=/Users/user/Desktop/CHLU-resweep`; `JAX_PLATFORMS=cpu`; `CHLU_CLAUDE_ROOT`
  / `CHLU_GEN_DATA_ROOT` / `CHLU_GEN_BRIDGE_ROOT` / `CHLU_GEN_DECLIFT_DIR` /
  `CHLU_GEN_STORE_RUNS` all set (a worktree has no `.claude/`).
- **Invocation, every point:** `python -m chlu.experiments.exp_gen_atom_mean --dataset
  shapes3d --pretext recon --stage atoms --seed S --out DIR --n-val V --tune-n V
  [--allow-knn-proto-move]`. ⛔ No `--quick`, ⛔ no `--config`, ⛔ no other override.
- **Seeds:** 0, 1, 2 at **every** point; a point that does not complete is reported
  **NOT RUN**, ⛔ never a gap and never a zero.
- **±:** population sd (**ddof = 0**) throughout, matching `gen-val-split-fix`.
- **Hardware:** this 16 GB macOS box, CPU only. ⛔ No full test suite.

**SIGN-OFF.** Predictions above are fixed. `gen-atom-mean-resweep` — experiment-engineer,
2026-09-15, before the first sweep run.

---

## ⭐ ADDENDUM A1 — 2026-09-15, still BEFORE the first sweep run

**What happened.** `test_raising_n_val_COSTS_fit_rows_arithmetically` — written to pin
§3 P5's arithmetic — **failed**, and the failure was the code's, not the test's:

```
n_fit = min(n_fit, pool.size)
n_val = min(n_val, max(pool.size - n_fit, 0))      # <- gen_rowsplit.stratified_fit_val
```

⛔⛔ **`n_val` is clamped by `pool.size − n_fit`, which is 32,768 − 32,256 = 512.**
So `--n-val 8192` at the shipped `n_fit` would have **cut 512 rows, selected on 512
rows, and logged entirely normally.** The whole disagreement curve would have been
**flat for a plumbing reason** and I would have reported a plateau that was my own bug —
the exact shape of the `--quick` trap the task file warns about, in a different file.

⚠ §1 of this PREREG asserted "`stratified_fit_val` cuts `val` first and gives `fit` the
remainder". **That was wrong about the order** and is corrected here.

**The code response** (in `exp_gen_atom_mean.py`, a file this spoke owns; ⛔
`gen_rowsplit.py` untouched):
- `fit_rows_for_val(n_fit_asked, n_val, n_pool)` derives `n_fit = n_pool − n_val` so the
  asked-for cut survives the clamp. ⭐ `n_fit` is still **not** a CLI flag — the caller
  names val rows, the fit cost is arithmetic, logged, and in `manifest.row_split
  .fit_val_trade`.
- **`MIN_FIT_ROWS = 2,048`**, a hard refusal — 2× the 1,024 rows where `knn_proto` fell.
- `run_seed` now **raises** if the realised `val` size ≠ the asked-for `n_val`. ⛔ A cut
  that is silently clamped measures nothing while logging normally.
- Three tests pin all of it, including that the shipped (32,256, 512) pair is a **no-op**.

**⛔ NOTHING IN §3 CHANGES.** P5's realised-`n_fit` table (32,256 / 31,744 / 30,720 /
28,672 / 24,576 = `32,768 − n_val`) was already exactly this arithmetic; the discovery is
that the code did not previously *deliver* it. P1–P6 stand as filed.
**Zero sweep runs had been executed when this addendum was written**
(`.claude/outputs/gen-atom-mean-resweep/runs/` held only `pytest_atom_mean.log`).

**Tests at this point:** `tests/test_gen_atom_mean.py` **38 passed, 0 failed, 0 skipped**
(4.50 s, main venv, `JAX_PLATFORMS=cpu`).
