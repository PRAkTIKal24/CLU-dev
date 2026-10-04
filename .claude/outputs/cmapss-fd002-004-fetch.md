# cmapss-fd002-004-fetch — experiment-engineer report

Task + acceptance criterion: fetch/convert FD002/003/004 onto the CAFE path, settle the cycles-remaining-vs-true-RUL labelling question with CLU measured both ways, run all four cells with the raw-stats reference, PREREG the predicted multi-regime failure, re-measure the FD002 relaxation-budget curve, and define+measure the bounded-vs-informative rollout diagnostic.
Status: **done** — all six items measured. PREREG landed on the headline and **missed on two sub-predictions, both reported**.

> ⚠ **DOWNSTREAM RECONCILIATION LIST — needs an owner (protocol §5, first 10 lines).**
> 1. **RETRACT the "single-basin collapse, q* spread exactly 0.000" finding.** It is **numerical overflow masked by `nan_to_num`**, not physics. It is currently in the handover as ⭐ matrix-worthy, in the WAVE-17 review, and was baked into `chlu/eval/config.py`'s docstring. Measured: at budget 64, γ=0.5 → **0% non-finite, spread 0.819**; γ≥5 → **100% non-finite, spread exactly 0.000**. Code + docstring fixed on my branch; the **handover text still needs correcting**.
> 2. **RETRACT "the relaxation BUDGET γ·steps·dt is the knob, not either factor alone."** Falsified by a 2-D grid: **iso-γ across a 400× budget range gives h-AUROC identical to 4 decimals**; iso-budget across γ gives a 0.11 spread. **γ alone controls it.**
> 3. **The +0.061 "budget lever" (0.6554→0.7168) was a γ lever (0.1→0.5)**, and it is monotone toward *doing less dynamics*. Any doc calling budget a top-3 lever needs rewording.
> 4. **CLU's best arm is +0.0027 over `q_last`** (0.7230 vs **0.7203**), with corr(q*, q_last) = **0.924**. The raw-stats reference is not the only thing that beats CLU — a single unmodelled sensor reading nearly matches it.
> 5. **CAFE's label bug is worth ~+0.20 h-AUROC, not a rounding error**, and under corrected labels a 56-d raw-statistics baseline scores **0.9545** — above CAFE's README HEPA number (0.918). See `CAFE_BUG_REPORT.md` (forwardable verbatim).

---

## Answer first

**Data:** all four FD00x were already on disk at `~/.hepa/data/CMAPSS/` (from the HEPA-SP checkout) — **no download needed**. All four verified canonical, and HEPA's `FD001` is **numerically identical** to the HF-derived FD001 used last wave, which cross-validates the source.

**PREREG P1 landed:** CLU loses to the raw-stats reference on **all four** cells, and loses **by more** on the multi-regime sets. The deficit tracks **operating conditions, not fault modes** — the cleanest form of the multi-basin claim available.

**But the mechanism story underneath it was wrong**, and that is the wave's real result. Both load-bearing mechanism findings from `clu-cafe-integration` fail to reproduce: the "single-basin collapse" is an overflow artifact, and the "budget" lever is a γ lever. Corrected, the picture is starker: **CLU's h-AUROC is monotone increasing in how much the Hamiltonian rollout is switched off**, and its best configuration is 0.92-correlated with the last raw sensor reading.

**And the planned rollout experiment has a problem worse than collapse:** CLU's iterated rollout **never beats a persistence baseline at any horizon, at any γ — including at n=1** (MSE 0.825 vs 0.600).

---

## What I did

- Located all four FD00x already present at `~/.hepa/data/CMAPSS/`; verified row/unit counts; staged into `~/cafe-data/cmapss/` (kept **outside** the repo). No HF download required.
- Wrote **`PREREG.md`** (4 predictions, both hypotheses pre-registered for the budget discriminator) **before** running any harness.
- Ran the four cells × 3 seeds × 2 arms + raw-stats reference through CAFE's own loader → CoxPH probe → `evaluate_event`.
- Quantified the label bug analytically and numerically on all four sets; ran the **true-RUL ablation on FD001** (evaluation labels swapped, probe held identical).
- Built the **bounded-vs-informative rollout diagnostic** (`chlu/eval/rollout_diag.py`, 16 tests) and measured it on FD001+FD002 to n=4096.
- Ran the FD002 budget curve — and, when it came out flat, ran a **2-D (γ, steps) grid** that falsified the budget claim, plus **overflow forensics** that falsified the collapse claim.
- Fixed the silent `nan_to_num` masking in `encode`; corrected the config docstring in place.

## How I verified

```
full suite (baseline 354)                    -> 370 passed, 0 failed  (354 + 16 mine)
ruff check <all touched files>               -> All checks passed
FD001 reproduction vs clu-cafe-integration   -> EXACT, seed-for-seed:
    raw_stats 0.7486 | ALL .7158/.7189/.7156 | basin .7230 | spread .7544
label-window alignment guard                 -> assert t_cafe == B.t_test  PASSED
--quick FD002 pipeline smoke                 -> h_auroc 0.4889 (pipeline check, not a result)
```
FD001 reproducing bit-for-bit against the prior wave is what licenses treating the FD002/003/004 numbers and the two retractions as comparable to it.

---

## Findings / results

### 1. Data — all four fetched and verified (no download needed)

| set | train rows | train units | test rows | test units | `RUL` lines | canonical? |
|---|---|---|---|---|---|---|
| FD001 | 20631 | 100 | 13096 | 100 | 100 | ✅ |
| FD002 | 53759 | 260 | 33991 | 259 | 259 | ✅ |
| FD003 | 24720 | 100 | 16596 | 100 | 100 | ✅ |
| FD004 | 61249 | **249** | 41214 | 248 | 248 | ✅ |

FD001 matches last wave's counts exactly. **CAFE's `_CMAPSS_INFO["FD004"]` claims 248 train units; it is 249** (bug 2 in the report). Windowed (window=30): FD002 → 46219/26505, FD003 → 21820/13696, FD004 → 54028/34081.

### 2. ⭐ The labelling question — settled, and it is worth more than the numbers

**Exact location:** `cafe_bench/datasets/event/cmapss.py`, `_load_split`, **lines 69 and 72**: `rul = max_cycle - i` and `e_list.append(1)`. `RUL_FD00x.txt` is never opened, and `_load_split` is called identically for train and test (lines 82–83).

**Derivation (pre-registered, then confirmed):** train recordings run to failure so `max_cycle − i` *is* true RUL; test recordings are truncated `RUL_unit` cycles early, so `t_true = RUL_unit + (max_cycle − i)`. ⇒ **every test label is under-estimated by exactly `RUL_unit`, a per-unit constant.** Confirmed: train label error is exactly **0.00** in all four sets.

| set | test windows | mean err | max | mislabelled | flips ≥1 horizon | P(event by h=125) CAFE vs true |
|---|---|---|---|---|---|---|
| FD001 | 10196 | **+62.5** | 145 | **100%** | 89.1% | 0.891 vs **0.497** |
| FD002 | 26505 | **+58.9** | 194 | **100%** | 85.0% | 0.850 vs **0.498** |
| FD003 | 13696 | **+62.7** | 145 | **100%** | 72.0% | 0.720 vs **0.376** |
| FD004 | 34081 | **+71.7** | 195 | **100%** | 69.9% | 0.699 vs **0.343** |

**(b) Does the ranking change? NO — but the levels move enormously.** FD001, evaluation labels swapped, probe identical:

| encoder | CAFE labels | **true RUL** | Δ |
|---|---|---|---|
| raw_stats 56-d | 0.7486 | **0.9545** | +0.206 |
| CLU `basin_coords` (3 seeds) | 0.7230 | **0.9093** | +0.186 |
| CLU ALL (3 seeds) | 0.7168 | **0.9054** | +0.189 |

**Ranking preserved** (raw_stats > basin > ALL in both), so the *relative* CLU-vs-baseline conclusions in this program survive. But **no CAFE C-MAPSS number is externally comparable**, and under correct labels a 56-d summary-statistics baseline scores **0.9545 — above the CAFE README's HEPA 0.918.** That makes it very hard to read the README cells as a meaningful bar.

Minimal repro + forwardable write-up: `.claude/outputs/cmapss-fd002-004-fetch/CAFE_BUG_REPORT.md`.

### 3. ⭐ CLU on all four cells — the deficit tracks OPERATING CONDITIONS, not fault modes

3 seeds each, `relax_budget=1.6`, CAFE loader → default CoxPH probe → `evaluate_event`.

| set | op cond | fault modes | **raw_stats 56-d** | CLU ALL | CLU `basin_coords` | **best CLU − raw** |
|---|---|---|---|---|---|---|
| FD001 | 1 | 1 | **0.7486** | 0.7168 ± 0.0015 | 0.7230 ± 0.0000 | **−0.0256** |
| FD003 | 1 | 2 | **0.7971** | 0.7706 ± 0.0006 | 0.7710 ± 0.0007 | **−0.0261** |
| FD002 | **6** | 1 | **0.6908** | 0.6421 ± 0.0004 | 0.6541 ± 0.0007 | **−0.0367** |
| FD004 | **6** | 2 | **0.7361** | 0.6600 ± 0.0004 | 0.6734 ± 0.0006 | **−0.0627** |

**CLU loses on 4/4.** Sorting by structure rather than by name is the finding:
- **1 op condition:** gap = −0.0256 (FD001), −0.0261 (FD003) — *adding a second fault mode costs essentially nothing (−0.0005)*.
- **6 op conditions:** gap = −0.0367 (FD002), −0.0627 (FD004) — **1.4× and 2.4× worse**.

⇒ **The deficit is driven by operating-condition multiplicity (regimes), not fault-mode multiplicity**, and the two compound. That is the multi-regime = multi-basin prediction in its cleanest measurable form.

**PREREG P1 scorecard (honest):**
| prediction | outcome |
|---|---|
| sign of (CLU − raw_stats) negative on all four | ✅ **4/4** |
| FD002/FD004 gaps *more* negative than FD001's −0.026 | ✅ (−0.037, −0.063) |
| FD002 CLU ∈ [0.60, 0.70] | ✅ 0.6541 |
| FD004 CLU ∈ [0.58, 0.68] | ✅ 0.6734 |
| FD003 CLU ∈ [0.68, 0.76] | ❌ **missed high** — 0.7710 |
| ordering FD003 ≳ FD001 > FD002 > FD004 | ❌ **FD002 and FD004 swapped** — measured FD003 > FD001 > FD004 > FD002 |

I predicted FD004 would be CLU's worst *cell*; it is FD002. FD004 is the worst *gap*. The op-condition/fault-mode decomposition above is a better description than my ordering, and I did not pre-register it — flagging it as **post-hoc structure, to be confirmed, not quoted as pre-registered**.

### 4. ⛔ RETRACTION 1 — the "single-basin collapse" is numerical overflow

The handover records, as ⭐ matrix-worthy: *"at γT=64 every window settles onto the same point — cross-sample std exactly 0.000, embedding rank-deficient, probe fails."*

`encode` ends with `np.nan_to_num(Z, nan=0, posinf=0, neginf=0)`. The dissipative step is `p ← (1−γ)p`, so **|1−γ| > 1 amplifies momentum every step** and the rollout overflows. Zero-filling then makes every coordinate exactly 0.0 ⇒ std exactly 0.000 ⇒ singular probe — **visually identical to a basin collapse.** FD001, budget 64 held fixed, only the (γ, steps) split varied:

| γ | steps | budget | non-finite rows | spread after `nan_to_num` |
|---|---|---|---|---|
| 0.5 | 2560 | 64 | **0%** | **0.819** |
| 2.0 | 640 | 64 | **0%** | **0.856** |
| 5.0 | 256 | 64 | **100%** | **0.000000** |
| 10.0 | 128 | 64 | **100%** | **0.000000** |
| 40.0 | 32 | 64 | **100%** | **0.000000** |
| 40.0 | 8 | 16 | 0% | 2.7e10 *(blow-up in progress)* |

Three independent measurements agree there is **no collapse**: forensics 0.819 at γT=64; FD002 budget curve 0.564 at γT=64 (probe fine, h=0.6534, **no ConvergenceError**); long rollout S_rel = 0.755 at n=4096 (budget 102). **The learned potential is not demonstrably single-basin. The evidence for that claim is withdrawn.**

*Fixed on branch:* `encode` now records `last_nonfinite_fraction` and warns. Note the new warning **fires on a pre-existing test** (`test_relax_override_changes_the_settled_point`, `relax_gamma=5.0`) — that test was already exercising a divergent config.

### 5. ⛔ RETRACTION 2 — γ is the knob, not the budget (FD002 curve + 2-D grid)

The FD002 budget curve at fixed γ=0.5 came out **flat**: h-AUROC 0.6537 / 0.6537 / 0.6537 / 0.6537 / 0.6537 / 0.6537 / 0.6536 / **0.6534** across budgets 0.16 → 64. The cross-check at the *same* budget 0.16 via (γ=0.1, steps=32) gave **0.5491**. So I ran the 2-D grid:

**ISO-BUDGET (budget = 1.6, γ varied) — must agree if budget is the knob:**
| γ | steps | FD001 h | FD002 h |
|---|---|---|---|
| 0.05 | 640 | **0.6109** | **0.5169** |
| 0.10 | 320 | 0.6744 | 0.5459 |
| 0.20 | 160 | 0.7132 | 0.6269 |
| 0.50 | 64 | **0.7230** | **0.6537** |

**ISO-γ (γ = 0.5, budget varied 400×) — must differ if budget is the knob:**
| steps | budget | FD001 h | FD002 h |
|---|---|---|---|
| 6 | 0.15 | 0.7230 | 0.6537 |
| 64 | 1.6 | 0.7230 | 0.6537 |
| 512 | 12.8 | 0.7230 | 0.6536 |
| 2560 | 64.0 | 0.7230 | 0.6534 |

**Exactly backwards from the claim.** γ alone determines h-AUROC; steps is inert to 4 decimals over a 400× range. (γ=0.1 likewise: 0.6744 at steps=320 *and* at steps=2560.)

**PREREG P2:** I pre-registered H_single ("optimum stays in [0.8, 3.2]") vs H_multi ("optimum moves ≥12.8"). **Both are void** — there is no optimum in budget, because budget does not control the outcome. This is the pre-registration doing its job: it forced the discriminating measurement that showed the axis itself was wrong.

### 6. ⭐ Why higher γ wins: it wins by not doing physics

h-AUROC rises monotonically with γ, and so does the similarity of `q*` to the window's last raw observation (FD001, seed 42):

| γ | steps | h-AUROC | median rel. displacement | **corr(q\*, q_last)** |
|---|---|---|---|---|
| 0.05 | 64 | 0.6147 | 3.52 | 0.559 |
| 0.10 | 32 | 0.6773 | 1.39 | 0.786 |
| 0.20 | 16 | 0.7140 | 0.33 | **0.972** |
| 0.50 | 6 | **0.7230** | 0.42 | 0.920 |
| 0.50 | 2560 | **0.7230** | 0.47 | 0.924 |
| — | — | **`q_last`, no CLU at all: 0.7203** | 0 | 1.000 |

**CLU's best arm beats "just take the last sensor reading" by +0.0027.** Larger γ overdamps the state so it barely moves; the best configuration is the one where the Hamiltonian rollout does the least. Combined with the raw-stats reference (0.7486) and the physics-scalars-only result from last wave (0.5887, near chance), the honest summary is: **on C-MAPSS the CLU encoder's measurable contribution over a trivial feature is ~0.003 h-AUROC.**

### 7. ⭐ The bounded-vs-informative rollout diagnostic — defined, measured, and it kills the planned framing

**Definition (reusable; shipped as `chlu/eval/rollout_diag.py`, adopt verbatim).**
For anchors *i*, channels *c*, rollout step *n*:
```
S(n)     = mean_c std_i q_i(n)[c]          (cross-sample spread)
S_rel(n) = S(n) / S(0)
COLLAPSE LENGTH  n* = min { n : S_rel(n) < tau },   tau = 0.01
```
Report `n*` with the damping budget `γ·n*·dt`. `n*` is **the honest ceiling on any long-horizon claim**: past it the model emits the same state regardless of input.

**Stated failure modes** (in the module docstring): (1) spread is **necessary, not sufficient** — it stays high for divergent/chaotic rollouts, so always read it beside the error curve and a persistence baseline; (2) it is **blind to rank collapse** (variance preserved on a shrinking manifold) — use a covariance participation ratio if that is the concern; (3) `S(0)` is anchor-dependent; (4) `tau` is a convention — report the curve.

**Measured (FD001 / FD002, 512 anchors, to n=4096):**

| γ | S_rel(64) | S_rel(256) | S_rel(4096) | **n\*** | max abs q | bounded |
|---|---|---|---|---|---|---|
| **FD001** 0.0 | 21.1 | 35.4 | 16.0 | **none** | 346 | yes |
| 0.1 | 3.78 | 3.32 | 0.140 | **none** | 13.4 | yes |
| 0.5 | 1.56 | 1.51 | 0.755 | **none** | 7.4 | yes |
| **FD002** 0.0 | — | 36.0 | 17.0 | **none** | 191 | yes |
| 0.1 | — | 3.11 | 0.360 | **none** | 11.0 | yes |
| 0.5 | — | 1.48 | 1.06 | **none** | 3.9 | yes |

**Collapse length: `n* > 4096` for every γ on both datasets** (budget > 102 at γ=0.5). So the pre-registered collapse budget γT* ∈ [3, 30] is **refuted** — a direct consequence of retraction 1, since that band was inferred from the bogus 0.000.

**PREREG P3 scorecard:** monotone-exponential decay ❌ (**non-monotone**: γ=0.5 contracts to 1.51 by n=8 then decays only slowly; γ=0 *expands* 35×); collapse budget ∈ [3,30] ❌ (>102); FD002 does not delay collapse ✅ (curves nearly identical — 1.48 vs 1.51 at n=256); **boundedness does not discriminate ✅ — every single configuration is "bounded", including the γ=0 rollout whose error reaches 484.**

**⛔ The load-bearing negative result — worse than collapse.** CLU's rollout is compared against **persistence** (predict `q(n) = q(0)`), FD001:

| n | 1 | 8 | 16 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|
| **persistence** | **0.600** | **0.580** | **0.589** | **0.591** | **0.777** | 6.17 |
| CLU γ=0.5 | 0.825 | 1.096 | 1.104 | 1.054 | 1.204 | **6.50** |
| CLU γ=0.1 | 0.825 | 2.752 | 4.132 | 5.093 | 4.787 | 8.89 |
| CLU γ=0.0 | 0.825 | 4.256 | 12.08 | 166.7 | 507.0 | 483.7 |

**CLU never beats persistence at any horizon, at any γ — including a single step** (0.825 vs 0.600). The failure is not that the rollout collapses; it is that **the rollout is not predictive at all**, while remaining perfectly "bounded" at γ=0.5. A rollout-vs-HEPA experiment scored on error-and-stability would therefore have reported a stable, bounded CLU rollout and buried the fact that "predict no change" is uniformly better. **This reshapes the planned rollout experiment: the prerequisite is a rollout that beats persistence at n=1, which does not currently exist.** Much cheaper to know now.

---

## Git footprint
- Branch **`agent/experiment-engineer/cmapss-fd002-004-fetch`**, off local `main` @ **`d805cd4`** (task said `8068d4f`; `main` had advanced 3 Hub commits — I based on current `main` and note it). Rebased onto `main` (no-op, base unmoved). **Not pushed.** Working tree clean.
- Commits: **`1448726`** (add rollout diagnostic + 14 tests), **`1e7ace5`** (loud non-finite guard; correct the falsified budget docstring; +2 tests).
- Files: `chlu/eval/rollout_diag.py` (new), `tests/test_eval_rollout_diag.py` (new, 16 tests), `chlu/eval/cafe_model.py` (**+1 import, `encode` guard, 1 attr init**), `chlu/eval/config.py` (**docstring only — no field, default or behavior changed**).
- Suite **370 passed** (354 baseline + 16). `ruff check` clean on all touched files. `clu_scorer.py` untouched. **No changes to the horizon/head path** (separate task, per scope note).
- Scratch/artifacts under `.claude/scratch/cmapss-fd002-004-fetch/` and `.claude/outputs/cmapss-fd002-004-fetch/`. CAFE checkout and data stayed outside the repo.

### Flag provenance
| item | value |
|---|---|
| commit | `1e7ace5` (cell numbers produced at `1448726`; unaffected — no behavior change) |
| seeds | 42, 43, 44 (cells, label ablation); 42 (budget grid, rollout, forensics) |
| datasets | `cmapss_fd001/2/3/4`, CAFE loader, window 30, C=14, horizons 1…125 |
| data source | `~/.hepa/data/CMAPSS/` → `~/cafe-data/cmapss/`; FD001 numerically identical to last wave's HF conversion |
| model | `clu`, `encode()`-only, **default CoxPH probe** (`penalizer=0.1`) |
| CLU config | `kinetic_mode=newtonian_learned`, `potential_type=mlp`, `hidden=64`, `dt=0.05`, `gamma=0.1`, `epochs=150`, `lr=1e-3`, `batch_size=64`, `max_fit_windows=4000`, `predict_horizon=16`, `relax_steps=32`, `neg_noise_scale=0.5`, `energy_reg=0.005`, `momentum_init=finite_diff`, no lattice |
| encode config | cells: `feature_groups`=ALL(7) or `("basin_coords",)`, `standardize=True`, `batch_size=512`, `relax_gamma=0.5`, `relax_steps=64` (budget 1.60) |
| grid/forensics | γ ∈ {0.05,0.1,0.2,0.5,2,5,10,40} × steps ∈ {6,8,16,32,64,128,160,256,320,512,640,2560} |
| rollout diag | `horizon` 256 (with truth) / 4096 (spread only), `n_anchors` 222–1500 (truth) / 512 (long), `gammas` (0.0,0.1,0.5), `tau`=0.01, train-split anchors |
| env | JAX **0.9.0**, equinox 0.13.4, main `.venv` (**no worktree sync**), lifelines 0.30.3, pandas 2.3.3, CPU |
| CAFE | `~/cafe-bench` @ `dc3dbd0` |

**PREREG** written before any harness run: `.claude/outputs/cmapss-fd002-004-fetch/PREREG.md`. Outcome reported above — **P1 landed on the headline (4/4 on the sign, 2/3 bands), P2 void (the axis was wrong), P3 mostly refuted, P4 landed on ranking and direction.** The op-condition/fault-mode decomposition (§3) is **post-hoc**.

---

## Open questions / follow-ups / risks
1. **The two retractions need an owner in the handover** (reconciliation items 1–3). The collapse finding is currently cited as measured evidence for `anti-collapse-characterization`; that support is withdrawn. The *theory* may still be right — but this benchmark no longer evidences it.
2. **Is there any CLU configuration that beats `q_last` by more than noise on C-MAPSS?** Currently +0.0027. If not, the encoder-swap claim needs a different dataset or a horizon-conditioned objective before it is worth running.
3. **The rollout experiment's prerequisite is unmet** — beat persistence at n=1 first. Recommend the Hub re-scope it as "can CLU's rollout be made predictive at all", with the diagnostic and the persistence baseline as the gate.
4. **FD003 is CLU's best cell (0.7710) and I under-predicted it.** 1 op condition + 2 fault modes. Worth understanding — it is the cell where HEPA also does best (0.82 App. G).
5. **Not done:** FD002/003/004 under true-RUL labels (only FD001 ablated); `q_last` reference on FD002/003/004; anomaly track still unmeasured.
6. **`max_probe_train` was left at `None`** (full CoxPH) for comparability with last wave's FD001 numbers; FD004 runs are correspondingly slow.

## Proposed handover updates (for the Hub)
1. **⛔ Strike the "single-basin collapse / spread exactly 0.000 / probe singular" entry** (§WAVE-17 review, the ⭐ matrix-worthy block, and the claims matrix). Replace with: *numerical overflow of the dissipative step for `relax_gamma > 2`, masked by `nan_to_num`; no collapse is observed at any tested γ ≤ 2 out to budget 102.* Guard + tests now on branch.
2. **⛔ Strike "the relaxation budget γ·steps·dt is the knob that matters."** Replace with: *γ alone controls h-AUROC; steps is inert across a 400× range; performance is monotone in γ because larger γ freezes the state nearer `q_last`.* Reclassify the +0.061 "budget lever" as a **γ lever**.
3. **Add: CLU's best C-MAPSS arm is +0.0027 over `q_last`** (0.7230 vs 0.7203, corr 0.924). This belongs beside the raw-stats reference in every downstream doc — it is the stronger version of the same honesty point.
4. **New result for the matrix: the CLU−baseline deficit scales with OPERATING CONDITIONS, not fault modes** (−0.026/−0.026 at 1 op condition; −0.037/−0.063 at 6). Post-hoc; pre-register before quoting.
5. **CAFE label bug quantified and forwardable** — 100% of test windows mislabelled in all four sets, mean error 59–72 cycles, P(event by h=125) nearly doubled (0.891 vs 0.497 on FD001); worth **+0.19–0.21 h-AUROC**; under correct labels **raw stats = 0.9545 > README HEPA 0.918**. Send `CAFE_BUG_REPORT.md` verbatim (also contains the FD004 unit-count bug and the dead download path).
6. **Ranking is label-robust** — CLU-vs-baseline orderings survive the relabelling, so this program's *relative* conclusions stand even though no absolute CAFE number is externally comparable.
7. **New reusable asset:** `chlu/eval/rollout_diag.py` + `RolloutDiagConfig` (collapse length `n*`, tau=0.01, documented failure modes). The `clu-horizon-encoder` / rollout tasks should adopt it verbatim.
8. **Data location note:** all four C-MAPSS variants + RUL files live at `~/.hepa/data/CMAPSS/`; no HF download needed. FD001 there is numerically identical to last wave's HF conversion.
9. **Env unchanged this wave** — no new deps; `lifelines`/pandas decision from last wave still owed.
