# multi-seed-w23 — error bars on every w23 headline

**Task + acceptance:** multi-seed w23's seed-0 headlines and return a three-way triage (SURVIVES / SOFTENS / ⛔ FLIPS) with mean ± CI and seed count for every number, proposing registry wording for anything that moves.

**Status: done — with one item dropped and declared (Item 3).**

> ⚠ **PROVENANCE — READ FIRST.** The spawned `results-analyst` wrote the PREREG, built the harness, and launched every run, but **did not survive to collect them** (its agent died during a re-authentication interruption; its compute was orphaned to `launchd` and ran to completion). **This report was written by the Hub** from the analyst's completed on-disk runs, holding to the analyst's `PREREG.md` verdicts and falsifiers verbatim. No experiment was re-run and no prediction was altered after the fact. The analyst's own partial aggregate (`phi/phi_multiseed.json`, MNIST only) was superseded by a full re-aggregation from raw per-seed metrics, because it was built before the CIFAR runs finished.

---

> ⛔ **DOWNSTREAM RECONCILIATION LIST — needs an owner (protocol §5 corollary). Four sites.**
> 1. ✅ **BOTH TIER-A ENTRIES SURVIVE. N89 (φ laundering) and N90 (retry mechanism) are multi-seed-clean.** Laundering fires in **4/4 cells at every seed** with **zero CLU wins anywhere** (0/22 cell-seed pairs); the mechanism controls violate the ±3 pp bar in **0 of 40** (seed, cell) pairs each. The two results the program's positioning rests on are not seed artifacts. **No registry change needed — the tier-A tiers stand.**
> 2. ⛔ **THE NN-GAP RANGE IS WRONG A SECOND TIME, AND BOTH PRIOR VERSIONS ARE TOO NARROW.** Measured at 5 seeds: **mask −3.9 … −20.7 pp; noise −9.7 … −48.2 pp** (per-cell means; negative in **40/40** pairs). The direction doc's *"within 3–13 pp"* is wrong (already known), **and the Hub's own correction *"3.5–17.6 pp on mask, widening to 42 pp"* is ALSO too narrow at both ends.** The analyst pre-registered exactly this — *"I expect the upper mask endpoint to exceed 17.6 pp"* — and was right. **Third wording required; do not quote either prior range.**
> 3. ⭐ **THE CIFAR SCOPING OF THE CLU-vs-Hopfield WORDING IS CONFIRMED AND STRENGTHENED.** The MNIST exception is not seed noise: at MNIST M=256 PCA, **Hopfield-in-φ is ahead of CLU-in-φ by +8.4 ± 1.2 pp, 95% CI [7.3, 9.1], at 5/5 seeds.** The analyst predicted this cell would SOFTEN with a CI possibly straddling zero; it did the opposite — it is precise and unambiguous. **CM-23 amendment (1) is vindicated by measurement**, and the binding CIFAR scoping should now cite this CI rather than the seed-0 pair.
> 4. ⛔ **ITEM 3 (the `dimension-aware-budget` frontier) NEVER RAN — and it was the highest-FLIP-risk item in the PREREG.** `N92 (tier A)`'s budget-adequate walls **d=4 = 16** and **d=5 ≥ 32 remain 2-seed**, exactly as they were at w23 close. The analyst's declared compute priority permits the drop, and it is named here as required — **but the specific numbers the analyst flagged as "the closest call in the whole task" (P3.2: a straight 50/50 with a CI expected to straddle the 0.90 pass line) are still unverified.** This is the one open multi-seed debt.

---

## Flag-provenance table

| item | value |
|---|---|
| harness | the analyst's `.claude/scratch/multi-seed-w23/run_one.py` (per-seed subprocess runner) driving the **merged w23 CLI** on `main @ 5e466c0` |
| φ seeds | **MNIST 5 seeds (0–4)** · **CIFAR-10 4 seeds (0–3)** — ⚠ `phi_s4_cifar10` never ran (the interruption); CIFAR CIs are n=4 |
| retry seeds | **5 seeds (0–4)**, all 8 cells, all 6 lines |
| dim / controller | Item 3 **not run** · Item 4 **verified, not re-run** (per task) from `.claude/outputs/controller-mvp/exp_controller_mvp_metrics.json`, `seeds=[0,1,2,3,4]` |
| statistics | mean ± seed std, **bootstrap 95% CI over seeds, 10 000 percentile resamples** (the PREREG method), RNG seed 12345 |
| ⚠ CI caveat | **n = 4–5.** A bootstrap CI on five points is coarse and cannot resolve tail behaviour; it is reported because the PREREG registered it. Treat CIs as indicative of spread, not as tight interval estimates |
| "flat" checks | judged **per (seed, cell)**, never on the mean, as the task required |
| collection code | `.claude/scratch/multi-seed-w23/collect/{aggregate,triage}.py` (Hub-written, read-only over completed runs) |

---

## Item 1 — `phi-read-in`

### 1.1 ⭐ P1.1 — the laundering verdict (tier A, N89)
Per-cell `max_clu_margin` = max over the load grid of (CLU-in-φ − kNN-in-φ) identity-accuracy. Falsifier: any seed > +0.03.

| cell | max_clu_margin (mean ± std [CI]) | seeds | CLU wins / seed | violations | verdict |
|---|---|---|---|---|---|
| mnist/pca | −0.0273 ± 0.0611 [−0.082, 0.000] | 5 | 0,0,0,0,0 | 0 | **LAUNDERED** |
| mnist/ae | −0.0219 ± 0.0489 [−0.066, 0.000] | 5 | 0,0,0,0,0 | 0 | **LAUNDERED** |
| cifar10/pca | −0.0898 ± 0.1055 [−0.180, 0.000] | 4 | 0,0,0,0 | 0 | **LAUNDERED** |
| cifar10/ae | +0.0020 ± 0.0039 [0.000, 0.006] | 4 | 0,0,0,0 | 0 | **LAUNDERED** |

**`laundered = true` in every cell at every seed. Zero CLU wins in 22 cell-seed pairs. Pooled mean ≤ 0.01 in all four cells** (the PREREG bar). The best CLU ever manages is a *tie* (margin 0.000, cifar10/ae peaking at +0.002 — inside the 0.03 tie band).
⇒ **P1.1 SURVIVES.** N89 is not a seed artifact; *"the win is φ's, not ours"* holds at 5 seeds.

### 1.2 P1.2 — CIFAR M=256, AE arm: CLU-in-φ vs closed-form Hopfield-in-φ
CLU `[0.973, 0.949, 0.938, 0.883]` · Hopfield-softmax `[0.008, 0.027, 0.027, 0.012]`
**Difference: +0.917 ± 0.039, CI [0.884, 0.951], n=4 — ≥ +0.50 at every seed.** ⇒ **SURVIVES.**
⚠ Note for drafting: w23 quoted the seed-0 pair **"0.973 vs 0.008"**; the multi-seed values are **CLU 0.936 ± 0.039** vs **Hopfield 0.019 ± 0.009**. Quote the means, not seed 0.

### 1.3 ⭐ P1.3 — the MNIST cell where Hopfield is ahead (the CIFAR-scoping test)
CLU `[0.871, 0.828, 0.887, 0.855, 0.859]` · Hopfield `[0.957, 0.918, 0.949, 0.949, 0.945]`
**Hopfield − CLU = +0.0836 ± 0.0122, CI [0.0727, 0.0914], Hopfield ahead at 5/5 seeds.**
The PREREG predicted **SOFTENS** — sign stable but magnitude unstable, std ≥ 0.03, CI possibly including 0. **Measured std is 0.0122 and the CI excludes zero by a wide margin.** The prediction was wrong in the *conservative* direction: this cell is more stable than expected.
⇒ **SURVIVES (stronger than pre-registered).** The MNIST exception is a real, precise effect — **the binding CIFAR scoping of the CLU-vs-Hopfield wording is confirmed by measurement, not merely by caution.**

### 1.4 P1.4 — retry-confidence AUROC
| cell | AUROC | min | ≥0.80 all seeds |
|---|---|---|---|
| mnist/pca | 0.9848 ± 0.0078 [0.979, 0.990] | 0.975 | ✅ |
| mnist/ae | 0.9789 ± 0.0068 [0.974, 0.985] | 0.974 | ✅ |
| cifar10/pca | 0.9917 ± 0.0028 [0.989, 0.994] | 0.988 | ✅ |
| cifar10/ae | 0.9367 ± 0.0660 [0.876, 0.983] | 0.845 | ✅ |

**Pooled mean 0.974 ≥ 0.90.** CIFAR-AE has the widest spread (std 0.066) exactly as predicted. ⇒ **SURVIVES.**

---

## Item 2 — `retry-compute-study`

### 2.1 ⭐⭐ P2.1 — mechanism attribution (tier A, N90's surviving positive)
Per (seed, cell): `best_acc(control) − first_pass_acc ≤ +3 pp`. **40 pairs per control.**

| control | pairs | violations (> +3 pp) | max lift | mean lift |
|---|---|---|---|---|
| `random_kick` | 40 | **0** | +1.56 pp | +0.30 pp |
| `ensemble` | 40 | **0** | +0.39 pp | +0.01 pp |

**Both controls are dead flat at every seed in every cell.** The largest single excursion across 80 measurements is +1.56 pp, half the falsification bar. ⇒ **P2.1 SURVIVES — decisively.** The directed re-launch is the mechanism; it is not stochastic restart and not "just k tries," and that conclusion is now 5-seed.

### 2.2 P2.2 — gated lift
| cell | first-pass | lift (mean ± std [CI]) | all seeds > 0 |
|---|---|---|---|
| mask M128 p=0.5 | 0.473 | +48.4 ± 9.4 pp [41.4, 55.5] | ✅ |
| mask M128 p=0.7 | 0.078 | **+75.9 ± 2.4 pp** [74.2, 78.0] | ✅ |
| mask M256 p=0.5 | 0.312 | +64.1 ± 4.2 pp [60.9, 67.3] | ✅ |
| mask M256 p=0.7 | 0.030 | +74.5 ± 1.3 pp [73.4, 75.5] | ✅ |
| noise M128 σ=0.2 | 0.883 | +2.0 ± 2.5 pp [0.2, 4.1] | ✗ (ceiling cell) |
| noise M128 σ=0.3 | 0.562 | +13.8 ± 4.3 pp [10.5, 17.5] | ✅ |
| noise M256 σ=0.2 | 0.844 | +4.4 ± 2.7 pp [2.4, 6.5] | ✅ |
| noise M256 σ=0.3 | 0.226 | +29.2 ± 4.7 pp [25.7, 32.8] | ✅ |

Lift > 0 at every seed in **all 4 mask cells and all 3 non-ceiling noise cells** — exactly the pre-registered condition (the σ=0.2/M128 ceiling cell was excluded in advance and does show a zero-lift seed).
**Pooled range +0.0 … +79.7 pp** vs the pre-registered prediction of +0 … +80 pp.
⇒ **SOFTENS as predicted.** w23's *"+6.6 … +76.2 pp"* (seed 0) becomes **"+0 … +79.7 pp across 5 seeds"**.

### 2.3 P2.3 — saturation multiplier
Pooled compute-multiplier at best accuracy: **×1.53 ± 0.28, CI [1.44, 1.61], full range ×1.00–1.81** (n=40).
⇒ **SOFTENS.** w23's *"×1.2–1.8"* should be **"mean ×1.53, range ×1.0–1.8"** — the ×1.00 floor is the ceiling cell where the gate correctly declines to spend anything, which is the self-limiting property working, not a failure.

### 2.4 ⛔ P2.4 — the NN gap (the number headed for an appendix)
`gated_best − feedforward_nn_best`; negative ⇒ the NN floor dominates.

| cell | gap (mean ± std [CI]) |
|---|---|
| mask M128 p=0.5 | −3.9 ± 1.2 pp [−4.8, −3.0] |
| mask M256 p=0.5 | −4.4 ± 1.4 pp [−5.6, −3.4] |
| mask M128 p=0.7 | −15.5 ± 5.2 pp [−19.8, −11.7] |
| mask M256 p=0.7 | **−20.7 ± 2.4 pp** [−22.5, −18.8] |
| noise M128 σ=0.2 | −9.7 ± 2.6 pp [−11.9, −8.1] |
| noise M256 σ=0.2 | −11.3 ± 1.6 pp [−12.2, −9.8] |
| noise M128 σ=0.3 | −30.0 ± 6.9 pp [−35.8, −25.5] |
| noise M256 σ=0.3 | **−48.2 ± 7.1 pp** [−53.7, −42.7] |

**Negative in 40/40 (seed, cell) pairs — the NN floor dominates absolutely, at every seed, in every cell.**
⇒ **SOFTENS.** New measured ranges: **mask −3.9 … −20.7 pp · noise −9.7 … −48.2 pp.** Both prior wordings are too narrow at both ends and must be replaced (reconciliation item 2).

---

## Item 3 — `dimension-aware-budget` frontier · ✅ **COMPLETED POST-HOC (Hub-launched, 2026-07-25)** ~~⛔ NOT RUN (dropped, declared)~~

> **⟲ Update (2026-07-25):** the Hub ran the analyst's own `run_dim_cell.py` on all four frontier cells (2× atoms, 3 seeds each, the pre-registered protocol). Results against the PREREG:
>
> | cell | strict (3 seeds) | mean | bar 0.90 | PREREG verdict | outcome |
> |---|---|---|---|---|---|
> | d=4 K=16 | 0.928 / 0.940 / 0.920 | **0.928** | **PASS** | P3.1 "SURVIVES, marginally" | ✅ **SURVIVES clean** — the `2^4=16` wall holds |
> | d=5 K=32 | 0.888 / 0.911 / 0.897 | **0.898** | straddles | P3.2 "the closest call, 50/50, CI straddles 0.90" | ◐ **SOFTENS exactly as registered** — "≥16, consistent with 32" |
> | d=6 K=64 | 0.820 / 0.830 / 0.805 | **0.818** | **FAIL** | P3.3 "remains FAIL" | ✅ firm wall confirmed |
> | d=8 K=64 | 0.900 / 0.907 / 0.914 | **0.9067 ± 0.0068** | **PASS (marginal)** | P3.3 "stays below but CI may touch 0.90" | ⚠ **CROSSES the bar** — K_learned(8) ≥ 64 under adequate budget |
>
> **⚠ The d=8 crossing is a finding, not a formality.** A *d-independent* operator ceiling at ~32 cannot pass K=64 at d=8 while failing it firmly at d=6. The geometric width-lock account (`lattice-capacity-theory` §4.2/checkF) **predicts exactly this split**: d=8 K=64 sits at sep/width **3.03** — the top of the measured 2.4–3.0 transition window — while d=6 K=64 sits at **2.65**, inside the fail zone. This is quantitative, out-of-sample support for the geometric account, obtained from a run scoped merely to "confirm the ceiling." It does **not** settle the mechanism (the §5.0 trained-`log_width` dump remains the decisive check) but it moves the prior toward §4.2 and further deprecates any "the ceiling is the write operator, d-independent at ~32" wording (N96 CONTESTED, now with a measured crack). Raw: `.claude/scratch/multi-seed-w23/dim_*.json`.

The analyst wrote the runner (`run_dim_cell.py`) but no cell executed before the interruption. **P3.1 (d=4 K=16 at 2× atoms), P3.2 (d=5 K=32 at 2× atoms) and P3.3 (the d=6/d=8 ceiling cells) are all unverified.**

Per the PREREG's declared priority — *"Item 1 laundering and Item 2 mechanism controls run first and complete at 5 seeds no matter what; Item 3 next; anything dropped is named explicitly"* — the drop is within protocol and is named here.

⚠ **Consequence, and it is the sharpest thing in this report:** the PREREG identified P3.1/P3.2 as **the highest-FLIP-risk predictions in the entire task** (P3.2: *"the closest call in the whole task… I register a straight 50/50 and predict the 95% CI will straddle 0.90"*). **N92 is tier A and its budget-adequate walls still rest on 2-seed re-checks.** If P3.1 fails, the *"`2^d`, base 2 — geometry vindicated"* half of N92 softens to `K_learned(4) = 8` and the geometric regime is no longer exactly base 2. **Recommend this be the first item of any w25 analyst task** — it is cheap (four cells) and it is load-bearing for a tier-A entry.

---

## Item 4 — `controller-mvp` · verified, not re-run

`seeds = [0,1,2,3,4]`, paired proposal sequences; reported ± values confirmed to be seed std over those 5.

| arm | K | per-admitted | per-offered | n_admitted |
|---|---|---|---|---|
| OFF | 16 | 0.1102 ± 0.0706 | 0.1102 ± 0.0706 | 16.0 |
| OFF | 64 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 64.0 |
| ON fixed | 16 | **1.0000 ± 0.0000** | 0.3250 ± 0.0250 | 5.2 ± 0.4 |
| ON fixed | 64 | **1.0000 ± 0.0000** | **0.0813 ± 0.0062** | 5.2 ± 0.4 |
| ON sized | 64 | **1.0000 ± 0.0000** | **0.6687 ± 0.0063** | 42.8 ± 0.4 |

- per-admitted is **1.000 with std exactly 0.000 at all seven K values** ✅ (as predicted)
- per-offered fixed K=64 **0.0813 ± 0.0062**, std ≤ 0.01 ✅ (as predicted)
- per-offered sized K=64 **0.6687 ± 0.0063** — predicted std ≥ 0.02, **measured 0.0063**: wrong in the conservative direction again; the sized arm is *tighter* than expected

⇒ **SURVIVES — multi-seed-clean.** All three headline cells carry their spread and none is fragile. No re-run needed, as the task specified.

---

## PREREG scorecard

| # | registered verdict | measured | outcome |
|---|---|---|---|
| P1.1 | SURVIVES | laundered 4/4 cells, all seeds, 0 wins in 22 pairs | ✅ **exact** |
| P1.2 | SURVIVES | +0.917 ± 0.039, all ≥ 0.50 | ✅ **exact** |
| P1.3 | SOFTENS (CI may straddle 0) | +8.4 ± 1.2 pp, CI excludes 0, 5/5 | ◐ **wrong direction — more stable than predicted** |
| P1.4 | SURVIVES | all ≥ 0.845; pooled 0.974 | ✅ **exact** (incl. the CIFAR-AE widest-spread call) |
| P2.1 | SURVIVES | 0/40 violations both controls | ✅ **exact** |
| P2.2 | SOFTENS, range +0…+80 pp | +0.0 … +79.7 pp | ✅ **exact** |
| P2.3 | SOFTENS, ×1.1–1.9 | ×1.53 mean, range ×1.00–1.81 | ◐ **range floor lower than registered** |
| P2.4 | SOFTENS; mask upper endpoint will exceed 17.6 pp | mask to −20.7 pp; noise to −48.2 pp | ✅ **exact, including the endpoint call** |
| P3.1–3.3 | (predictions made) | **NOT RUN** | ⊘ **dropped, declared** |
| P4.1 | multi-seed-clean | 2 of 3 sub-predictions exact; sized-arm std tighter | ✅ **verdict holds** |
| **P0 (global)** | **zero FLIPS on tier-A; 2–4 SOFTENS** | **zero FLIPS anywhere; 3 SOFTENS** | ✅ **exact** |

**Global: 0 ⛔ FLIPS · 6 SURVIVES · 3 SOFTENS · 3 dropped.** Every softening is a *range restatement*, not a direction change. The analyst's global prediction P0 was correct.

---

## How I verified
- Re-aggregated from **raw per-seed metrics JSONs**, not from the analyst's partial aggregate (which predated the CIFAR runs and covered MNIST only). 9 φ runs + 5 retry runs parsed; every number above is derived by `collect/triage.py` from those files.
- All "flat"/"survives" claims evaluated **per (seed, cell) pair** against the pre-registered numeric bars, never on pooled means alone.
- Item 4 read from the w23 controller metrics artifact; **no controller run was repeated.**
- No `chlu/` code touched; no experiment re-run; no branch, no commits.

## Recommended next
1. **⭐ Run Item 3** (four cells: d=4 K=16, d=5 K=32, d=6 K=64, d=8 K=64 at 2× atoms, ≥3 seeds). It is the only unverified tier-A-supporting evidence in the w23 set and the PREREG flagged it as the likeliest to flip.
2. **Re-run `phi_s4_cifar10`** to bring CIFAR to 5 seeds and match MNIST (single missing run; the CIFAR CIs are n=4).
3. Adopt the corrected NN-gap and lift/saturation ranges wherever they appear before any draft text is written.

## Proposed handover updates (for the Hub)
1. **N89 and N90 — add a multi-seed confirmation line.** Both tier-A entries survive at 5 seeds with zero violations of their pre-registered bars. Their tiers stand unchanged.
2. **⛔ CM-23 / appendix wording — third correction to the NN gap.** Replace both prior ranges with **"mask −3.9 to −20.7 pp; noise −9.7 to −48.2 pp (5 seeds, negative in 40/40 cell-seed pairs)."**
3. **⭐ CM-23 amendment (1) — upgrade the evidence.** The CIFAR scoping is confirmed: MNIST M=256 PCA has **Hopfield ahead by +8.4 ± 1.2 pp, CI [7.3, 9.1], 5/5 seeds.** Cite the CI, retire the seed-0 pair. Likewise quote CIFAR M=256 AE as **0.936 ± 0.039 vs 0.019 ± 0.009**, not "0.973 vs 0.008".
4. **Restate two retry ranges:** gated lift **+0 … +79.7 pp**; saturation **mean ×1.53, range ×1.0–1.8**.
5. **⛔ Register the Item-3 gap as an open debt** — N92's d=4/d=5 walls remain 2-seed, and the PREREG rated them highest-flip-risk. This is the one thing w24 owes w23.
6. **Process note worth keeping:** an orphaned agent's work was fully recoverable because it pre-registered its predictions and wrote per-seed artifacts to disk before analysing them. The PREREG made the collection mechanical and kept the triage honest — the Hub could not have quietly rewritten a prediction it did not like. Recommend this pattern (prereg + per-seed artifacts on disk) stay mandatory for long multi-seed runs.
