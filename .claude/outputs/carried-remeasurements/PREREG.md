# PREREG — carried-remeasurements (w26)

**Written before any harness was run.** Base commit `ff85573f0bc6dc2240297dcfbf8deaecae51ca45`
(local `main`, post-w25), working tree clean. Author: results-analyst.

Protocol §5 pre-registration rule applies because both items 1 and 2 have acceptance
criteria that are *measured ratios / curves*: item 1 registers an **identity** (four τ
curves equal), item 2 registers a **shape** (AUC vs load factor).

Registered *before* looking at any new number. Code was read (to know what the harness
does) but no new run was executed prior to this file being written.

---

## Item 1 — does "τ is not binding in φ-space" replicate at ≥3 seeds?

### Mechanism I am committing to (derived from `exp_retry_compute._retry_ladder`)

The gate is `eligible = (~locked) & (cos < τ)`; the round then takes the
`step_n = round(retry_step_frac · N) = round(0.1·N)` lowest-cosine eligible reads and
**locks** them. Therefore:

> **The four τ values in `retry_tau_grid = {0.99, 0.999, 0.9999, 1.0}` produce
> bit-identical ladders iff, at every round `j = 1…8`, the number of *unlocked* reads
> with `cos < 0.99` is ≥ `step_n`.**

Because 8 rounds lock at most `8 · 0.1 · N = 0.8 N` reads, a sufficient condition is that
**≥ 80 % of the population sits below cosine 0.99 after the first-pass settle**. If that
holds, τ cannot change which reads are picked, so accuracy *and* the compute multiplier
must agree exactly — the identity is then **structural, not stochastic**, and should
replicate at every seed.

### Registered predictions

| # | prediction | falsified if |
|---|---|---|
| **T1** | At the **hardest** corruption level (`retry_mask_p = 0.8`, the last entry of `retry_mask_levels`), all four τ ladders are **identical at every k ∈ {0,1,2,4,8}, to the last printed digit, at all 3 seeds** (accuracy AND compute multiplier). | any (seed, k) where two τ curves differ by >0 |
| **T2** | The mechanism: fraction of first-pass reads with `cos₀ < 0.99` is **≥ 0.85** at every seed at p = 0.8 (I predict 0.95–1.00; `mean_confidence` ≈ 0.90–0.97). | fraction < 0.80 while T1 still holds (⇒ my mechanism is wrong even if the identity is right) |
| **T3** | Compute multiplier at k = 8 equals **1.800 ± 0.001** for all four τ and all seeds (i.e. the eligible pool never empties, so the auto-stop never fires). | any τ/seed with compute@8 < 1.79 (that would mean the pool *did* empty, i.e. τ IS binding for some τ) |
| **T4** | Because of T1, the **gated k = 8 accuracy** at p = 0.8 is the same number for τ = 0.99 and τ = 1.0 at every seed; across seeds it varies by ≤ 0.10 absolute. | between-seed spread > 0.10 (then the cell is too noisy to adjudicate anything) |
| **T5** | **Regime statement (the thing I actually expect to have to write):** the identity is a property of the **φ-space** cell (crowded 200-item store, corruption applied in pixel space, cosine-to-nearest-well is low for nearly every read), *not* a contradiction of w24's pixel-space `exp_retry_compute` finding that τ = 1.0 over-retries. In pixel space the settled reads sit in normalised pattern wells with cos ≈ 1, so τ = 1.0 makes already-correct high-confidence reads eligible and boosting them corrupts them. **Prediction: `mean_confidence` in the φ cell is materially below the pixel-space regime's, and no read in the φ cell exceeds cos 0.999.** | max first-pass cosine ≥ 0.999 in the φ cell (then the two regimes are not separated by the confidence distribution and I need another explanation) |

**Verdict rule, registered now:** if T1 holds at 3/3 seeds ⇒ the w25 line stands and the
w24 threshold clause must be **scoped to pixel space**, not retracted. If T1 fails at any
seed ⇒ the w25 single-seed line is **retracted** and w24's clause stands globally.

**Known limitation registered in advance:** the shipped code runs the τ-sweep **only on
the last corruption level** (`if li == len(cfg.retry_mask_levels) - 1`). I therefore
**cannot** test τ at p = 0.5 without editing tracked code, which this task forbids. I will
report the p = 0.5 confidence distribution as the proxy and say so.

---

## Item 2 — occupancy sweep on the allocator trace (`z_hole`)

### Setup I am committing to

Identical to `mia-decay-measurement` §2 except the number of **background offers**
`N_BG ∈ {1, 3, 5, 7}` ⇒ nominal load **2/8, 4/8, 6/8, 8/8** items in a capacity-8 store.
Everything else held fixed: `dim = 3`, `α = 0.02`, `s = 0.35`, `d_safe = 1.54`,
proposal disk `R = radius_for_capacity(8, 1.54) = 2.2869`, packing bound 8.00, seeds
0/1/2, 8 targets, 128 worlds, 16 queries, shipped `Controller` / `two_phase` read.
`N_BG = 7` is the exact w25 configuration and is therefore a **replication cell**.

### Analytic model I am committing to (this is the derivation, not a guess)

Writing the target first reserves a `d_safe = 1.54` exclusion disk around `c_i` that
survives its eviction. Let `p` = probability that a *uniformly proposed* background site
lands within 1.54 of `c_i`:

`p = E_c[ area(B(c, 1.54) ∩ D(0, 2.2869)) / (π · 2.2869²) ]`, with `|c| ≤ 0.8 · 2.2869
= 1.829` (the target-draw rule in the harness). Lens-area evaluation: `p = 0.453` at
`|c| = 0` and `p = 0.277` at `|c| = 1.829`; area-weighted mean ⇒ **`p ≈ 0.36`**.

If **any** OUT-history background lands inside the exclusion radius, the IN world's
`z_hole` is definitively larger ⇒ that world pair is separated. Otherwise both draws come
from the same truncated law ⇒ chance. Hence

> **`AUC(z_hole) ≈ 1 − ½ (1 − p)^n`** with `n = N_BG`.

giving **0.680 / 0.869 / 0.946 / 0.978** at `n = 1 / 3 / 5 / 7`. This model **ignores
mutual background exclusion**, which at high load compresses the OUT world's sites into
the free area and makes separation *better* than the model — the w25 measurement at
`n = 7` was **0.99985**, above the model's 0.978. So I register the model as a **lower
bound that becomes loose as load rises**.

### Registered predictions

| # | prediction | band |
|---|---|---|
| **O1** | `AUC(z_hole, history)` is **strictly monotonically increasing** in load factor. | no band — a non-monotone curve falsifies |
| **O2** | Point values: **2/8 → 0.68**, **4/8 → 0.87**, **6/8 → 0.95**, **8/8 → 0.9998**. | ±0.10 / ±0.08 / ±0.05 / [0.995, 1.000] |
| **O3** | **Replication:** the 8/8 cell reproduces w25's `0.99985 ± 0.00070` to within **±0.002**. | a miss here means my copy of the harness is not the w25 harness ⇒ everything else is void |
| **O4** | **The leak does NOT disappear at low load:** `AUC(z_hole)` at 2/8 is **≥ 0.60** (materially above chance). | AUC(2/8) < 0.55 ⇒ the leak *is* essentially an occupancy artefact and the headline 0.99985 must be quoted as "at near-full occupancy" only |
| **O5** | **The dial-declaration falsifier:** `AUC(2/8) ≥ 0.99` would mean the leak is **not** occupancy-driven at all — worse than the w25 framing, and it raises the stakes on `placement-landing`. I predict this does **not** happen. | — |
| **O6** | `TPR@FPR 1 %` for `z_hole`: **1.000 at 8/8**, and **< 0.50 at 2/8**. | ±0.15 at the low end |
| **O7** | **Paired-placement column = `0.5000 ± 0.0000` exactly**, on all of `s1, s2, s4, s5, z_hole`, at **every** load. Any deviation is a harness bug, not a finding (registered as such in advance). | exact |
| **O8** | `AUC(n_live)`: **0.500 ± 0.02 at 2/8** (no refusals: `n_live_in − 1 == n_live_oh` identically, fully tied), rising to **≈ 0.81 at 8/8** (w25 replication, ±0.05). | as stated |
| **O9** | Post-evict **retention = 0.0000** at every load (the target is gone from `V`). | exact |
| **O10** | `evict` still leaves `centers`/`payloads` verbatim at every load (D1, unchanged code): centre max err ≤ 1e−6, payload max err = 0.0. | exact |

**Decision I am pre-committing to:** `placement-landing`'s acceptance test target
(post-evict `AUC(z_hole) → 0.5`) is **load-dependent**. If O2 holds, the honest statement
of the acceptance test is "at 8/8", and a placement rule that reaches 0.5 only at low load
has not fixed anything. I will report the target at all four loads so the acceptance test
cannot be gamed by choosing a load.

---

## Item 3 — high-load sharding probe (P4: d = 8, K = 256, 8×32)

**Not pre-registered here beyond what `lattice-sharded-store` already registered**
(expected FAIL, union separation 0.55–0.85, RG route accuracy 1.000, union separation
0.714). Per the task file this item is **droppable**. I register now, before knowing the
compute situation, that if it does not run I will report it as **NOT RUN**, never as a
null, and that if it does run at the parameter-matched 8-way split I will report the cell
as **budget-confounded** (N107: the w23 atom floor `512·√2^d` is per-store, not per-item).

---

## What would make me retract each item

- **Item 1:** any τ pair differing at any k at any seed ⇒ the w25 "τ not binding" line is
  retracted as a single-seed artefact and w24's clause stands.
- **Item 2:** O7 failing (paired ≠ 0.5000) ⇒ the whole measurement is void (harness bug);
  O3 failing ⇒ my copy diverges from the w25 harness and nothing here is comparable.

---

# ADDENDUM 1 (written 2026-07-28, **after** the seed-0 `[0.5, 0.8]` run, **before** any p=0.5 τ-sweep run)

## Why an addendum
The main PREREG registered, as a *known limitation*, that the shipped code sweeps τ only
on the **last** entry of `retry_mask_levels`, so p = 0.5 is untestable without editing
tracked code. **It is testable without editing code**: a project `config.yaml` with
`retry_mask_levels: [0.5]` makes p = 0.5 the last level. That is a **configuration**
change, not a code change, so it is in scope. I am registering the predictions now,
before running it.

## What seed 0 (already run, `[0.5, 0.8]`) tells me, and what it does not
Observed at seed 0 (from `projects/w26tau0/results/exp_cl_entry_mnist_metrics.json`):
τ-identity **holds exactly** at p = 0.8 (max|Δacc| = 0, max|Δcompute| = 0), and the
retry count at k = 8 is **160/200 = the unstarved maximum `8 · step_n`** — i.e. the
eligible pool never emptied. But at **p = 0.5** the *gated* line (which runs at
`cfg.retry_tau = 0.99`) **auto-stops**: compute at k = 8 is **1.121×** (end-of-stream)
and **1.235×** (mid-stream) instead of 1.80×.

Because `_retry_ladder` only ever updates the reads it *retries* (and then locks them),
an unlocked read's cosine is frozen at its first-pass value. Therefore
`total retried by k = #{cos₀ < τ}` capped at `k · step_n`, exactly. So the p = 0.5
auto-stop means `#{cos₀ < 0.99} ≈ 24` of 199 (12 %) end-of-stream and ≈ 47 of 200 (24 %)
mid-stream — far below the 160-read budget. **τ therefore MUST bind at p = 0.5.**

## Registered predictions for the p = 0.5 τ-sweep (`retry_mask_levels: [0.5]`, seeds 0,1,2)

| # | prediction | falsified if |
|---|---|---|
| **A1** | The four τ ladders at p = 0.5 are **NOT identical** — at least one k where two τ differ. | identity holds at p = 0.5 too |
| **A2** | Compute at k = 8: **τ = 0.99 → ≈1.121× (end-of-stream, seed 0), τ = 1.0 → 1.804×**, with 0.999 and 0.9999 in between and monotonically non-decreasing in τ. | non-monotone compute in τ |
| **A3** | Accuracy at k = 8 is **monotonically non-increasing in τ**: `acc(0.99) ≥ acc(0.999) ≥ acc(0.9999) ≥ acc(1.0)`, and **`acc(1.0) < 0.95`** at end-of-stream (vs ≈0.995 at τ = 0.99) — i.e. w24's "τ = 1.0 over-retries and costs accuracy" **reproduces inside φ-space**. | acc(1.0) ≥ acc(0.99) (⇒ over-retry is costless here and the w24 clause really is pixel-space-only) |
| **A4** | The p = 0.5 cell in this run is **bit-identical** to the p = 0.5 cell of the `[0.5, 0.8]` run at the same seed (same first-pass accuracy, same gated ladder), because it is the first level and consumes the same PRNG split. | any difference ⇒ the two runs are not comparable and A1–A3 are void |

## The regime statement I am now committing to (to be confirmed or refuted)
> The variable that decides whether τ binds is **NOT `φ`-space vs pixel-space**. It is the
> **first-pass confidence distribution** — specifically `#{cos₀ < τ}` relative to the
> retry budget `k · step_n`. τ is non-binding exactly when the low-confidence pool is
> larger than the budget (hard queries / crowded store), and binding — and *costly* at
> τ = 1.0 — when it is smaller (easy queries / small store). w24 and w25 measured the two
> ends of the *same* axis, and neither is wrong.

**Falsified if:** the p = 0.5 φ-space sweep shows the identity (A1 fails), which would
mean something other than the confidence distribution is responsible.
