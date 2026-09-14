# PREREG — gen-r0-wide-refit (written BEFORE the harness existed)

**Written:** 2026-09-11, before `wider0.py` was written and before any wide store was fitted.
**Rule invoked:** protocol §5 pre-registration (the acceptance criterion contains measured margins,
an agreement index and a gate verdict).

## What is being measured
R0 of the factorisation ladder — grouping **and** store — re-fitted on the **wide** feature cache
(`condition = recomb2element_wide`, 13 held-out cells, 3 seeds), with:
1. the adjusted-Rand agreement between Montero's shipped grouping and the wide grouping (R24 rider 1);
2. ⛔ **the gate**: the wide read margin (correlation − random) against Montero's shipped
   `+0.1504 ± 0.0094` (aggregate) / `+0.2008 ± 0.0165` (`shape`) (R24 rider 3);
3. R0's wide ladder coordinates: corrected leakage (R23.2), MIG, DCI-C, and the read metric.

## ⛔ A fact that must be stated before any prediction is read
The two feature caches come from **two different encoder fits** — that is the *point* of the wide
configuration (a wide-condition encoder must never have seen the wide cells). Their parameter hashes
therefore **differ by construction** and a hash mismatch is **not** the plumbing fault the gate hunts.
What makes the two groupings comparable at all is that the two encoders share:
* the same architecture and the same PRNG key chain (`seed*10007 + 20260909`) ⇒ **the same init**, and
* training rows that differ only by the held-out cells (479,064 vs 479,928 eligible rows).
⭐ Whether that is enough to preserve **channel identity** is itself an empirical question, and it is
pre-registered below (P1) as the *precondition* for interpreting P2 at all.

## Predictions (point, and the interval/threshold that decides them)

| # | prediction | derivation | threshold |
|---|---|---|---|
| **P1** | ⭐ **Channel identity survives the re-fit**: mean over channels of `corr(Montero ch. i, wide ch. i)` on the SAME rows ≥ **0.80**, point **0.95**; and the diagonal is the argmax for ≥ 90 % of live channels | identical init + 99.8 % identical rows ⇒ SGD lands in the same basin, channel-for-channel | ≥ 0.80 mean diagonal |
| **P1-alt** | ⛔ **Counter-hypothesis, registered so it can win:** 1500 SGD steps decorrelate channel identity even from a shared init ⇒ mean diagonal ≈ 0 and `ARI` is then **not a grouping-stability statement**, it is an encoder-reproducibility statement, and it must be reported as such | — | mean diagonal < 0.30 |
| **P2** | **ARI(Montero grouping, wide grouping) ≥ 0.30**, point **0.60**, per seed, **conditional on P1** | a near-copy encoder ⇒ a near-copy correlation matrix ⇒ ward returns a near-copy partition; the residual is the ward cut's sensitivity to small perturbations, which `gen-store-grouping` §2 showed is a low-contrast matrix (mean off-diagonal \|corr\| 0.143) | ≥ 0.30 |
| **P3** | ⛔ **THE GATE: CONSISTENT.** \|wide aggregate read margin − 0.1504\| ≤ 3 × pooled seed spread. Point wide margin **+0.15**, 80 % interval **[+0.10, +0.20]**; `shape` point **+0.20**, interval **[+0.13, +0.27]** | the read metric is measured on `eval_a` (seen rows) in both configurations, with the same store config, the same `K` and the same 30-bit channel; only the encoder fit and the held-out rows move | ≤ 3 pooled spreads |
| **P4** | ⭐ **The ruling-2 fallback TRIGGERS**: the `shape`-leakage margin (random − correlation) separates the arms by **< 3 × seed spread** ⇒ `shape`-leakage declared **underpowered**. Point margin **+0.05**, interval **[−0.05, +0.15]**. Stated confidence **60 %** | Montero's corrected value was `+0.0147 ± 0.0771` (0.19×). The wide split makes both sides genuinely 4-class, which fixes the *definition*; it does not obviously add *contrast*, because the leakage probes read a well's sub-vector, not the store | < 3× vs ≥ 3× |
| **P5** | pooled leakage margin (random − correlation) **> 0 on ≥ 2/3 seeds**, point **+0.13** | Montero corrected `+0.1337 ± 0.0592` (2.26×) | ≥ 2/3 positive |
| **P6** | MIG margin (correlation − random) ≥ **+0.10** (point +0.18) and DCI-C margin ≥ **+0.10** (point +0.175) | Montero's shipped `+0.1823` / `+0.175`; these are code-space and decoder-independent | ≥ +0.10 |
| **P7** | **non-degeneracy**: 0 NaN cells over all (well × factor) cells; `max = +1.0000` exactly attainable and never exceeded (structural: balanced accuracy ≥ 0 ⇒ leakage ≤ 1); `min ≥ −1.5` | the corrected metric's bound; Montero measured `[−0.5534, +1.0000]`, 0 NaN | 0 NaN, max ≤ 1 |
| **P8** | the absolute read stays **low for both arms** — correlation aggregate read **0.61**, interval **[0.50, 0.72]** — against a wide no-store ceiling ≈ **0.99** | 30 bits (`K·log₂M = 5·6`) reconstructing 2048 dims; a declared design constant, named in advance as a **non-falsifier** | — |
| **P9** | `assert_plan_matched` passes: `K = 5`, `n_modes = 64`, `D = 2048`, `store_bytes = 524288` unchanged | ⛔ `K` is a ladder constant, loaded verbatim | exact |

## What would make me report a PLUMBING FAULT
P3 failing — i.e. the wide and Montero read margins differing by more than 3 pooled seed spreads.
⛔ In that event the report says **STOP** and does not rationalise the divergence as a property of the
split, and R1 does not start.

## What I will NOT do
- ⛔ re-derive `K` (ladder constant, `store_plan.json`, `assert_plan_matched`);
- ⛔ re-run, re-fit or modify Montero R0 (read-only, shipped numbers only);
- ⛔ decode through the `"baseline"` recipe (the superseded 0.7191 decoder): the wide bridge's decoder
  is the `v1_capacity_schedule` winner and its recipe name **and** parameter hash are asserted against
  `gen-bridge-wide`'s published per-seed hashes before any number is recorded;
- ⛔ pool the shared-corner cells (`montero`, `wide-000`) into the headline (R23.3).
