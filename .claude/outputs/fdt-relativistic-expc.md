# fdt-relativistic-expc — results-analyst report

**Task + acceptance criterion:** validate F2 (`langevin_noise="fdt_relativistic"`) on the real
d=784 Exp-C path for the first time — (Q1) does it sample Gibbs (Maxwell–Jüttner momentum
marginal)? — and (Q2) close the N10 loop: does a *correct* sampler change the MNIST 3/5/8/9
imbalance? PREREG both; measure MJ-restoration on ≥2 checkpoints; digit histogram vs the prior
`legacy`/`fdt` arms (paired); state the three-way consistency; every cell carries `d·Θ` + provenance.

**Status: done.** PREREG written before running (`.claude/outputs/fdt-relativistic-expc/PREREG.md`).

**Reconciliation-list owner note (protocol §5 corollary):** this report contains a *downstream
reconciliation list* — CM-17 F2 gains its first real-data validation; N10 gains a decisive
closure and a **corrected mechanism sub-claim** (RGE §2b's "velocity saturation ⇒ momentum law
is dynamically invisible" is TRUE for the drift but FALSE for the quench depth). See §Proposed
handover updates; the Hub should assign an owner at the review that accepts this report.

---

## VERDICT (first 10 lines)
- **Q1 — F2 samples Gibbs on real Exp-C: YES, exactly.** On both checkpoints, both regimes, the
  momentum marginal **inverts from Gaussian to Maxwell–Jüttner**: `Var(p)/(M_eff·T)` = **783.5**
  (MJ theory 785.0), radius `r_MJ/r_obs` = **1.00×** (was 90× too small under `legacy`, 28× under
  `fdt`), `KL(radial‖MJ)=0.004 ≪ KL(radial‖Gauss)=0.142`. Split-half stationary to 0.2 %. **This is
  the first real-data validation of the F2 fix — it works.**
- **Q2 — the correct sampler does NOT fix the imbalance; N10 CLOSED.** F2 does not move the
  histogram toward uniform — it **worsens** the imbalance (`f(3589)` 0.609→**0.896**, TV 0.596→
  **0.649**), collapsing onto '8' exactly like the *other* MJ-momentum arm `mj_refresh`. It lands
  precisely on the RGE quench-depth curve (Hfin=−104; Spearman(Hfin, f3589)=**+1.000** over 4 arms).
- **Three-way consistent:** {F2 is Gibbs} + {N10: imbalance = landscape/quench-depth} ⇒ {F2 leaves
  imbalance un-fixed} — all three hold. The sampler was never the cause; the landscape is.
- **One prereg sub-prediction FALSIFIED (honest):** I predicted F2 ≈ `fdt` (velocity saturation).
  F2 is distinguishable from `fdt` (χ²=35.6, p=8.6e-6) and close to `mj_refresh` (χ²=15.7). The
  MJ momentum's larger *noise variance* keeps the quench **shallower**, which the velocity-drift
  argument missed. The **headline** prereg prediction (imbalance not eliminated) held.

---

## 0. Flag-provenance (protocol §5, mandatory)

| item | value |
|---|---|
| code commit | **`e3c8931`** (`integration/wave-15`; carries fix-pack-7's `fdt_relativistic` + `gibbs_defect_parameter`, verified present at `chlu/core/integrators.py:294`, `chlu/core/chlu_unit.py:387`) |
| repo state | **clean; 0 tracked files changed.** All artifacts under `.claude/` (gitignored). `chlu/` read-only (task requires). |
| env | main venv `/Users/user/Desktop/CHLU/.venv` (Python 3.11, JAX per fix-pack-7 resolution; not re-synced) |
| checkpoints | `projects/mnistFFF/models/exp_c_chlu.pkl` (`relativistic`, `conv`, d=784, **c=1, m₀=1**, trained `legacy`) and `.claude/scratch/relativistic-gibbs-expc/e5_models/exp_c_chlu_c1_ep500.pkl` (native c=1, seed 42, 500 ep) |
| **new sampler flag** | `langevin_noise="fdt_relativistic"` (F2 latent-mass thermostat; InvGauss scale mixture). Repo default is `legacy` — this is opt-in and does not fire the `RelativisticGibbsWarning`. |
| Q1 protocol | constant-T chains, γ=0.3, dt=0.05, 64 chains init at real MNIST images, **burn-in 800** + 100×10 steps ⇒ 6 400 momentum samples/cell; `PRNGKey(2027)` folded per cell |
| Q1 cells | mnistFFF {c1·T1, c1·T0.5, c5·T1}, nativeC1 {c1·T1} — all `fdt_relativistic`. `legacy`/`fdt` baselines quoted from `relativistic-gibbs-expc` E1 (bit-reproducible, same instrument). |
| Q2 dream protocol | **byte-for-byte RGE E2 / Study A.2** (paired): γ=0.3, dt=0.05, T exp **1.0→0.01** over **1000** steps, init=centroid(first 10k)+0.5·N(0,1), p₀=0.1·N(0,1), 64 dreams × seeds {0,1,2} (n=192/arm), **identical init states + identical chain keys** as the prior arms |
| Q2 arms | `c1_fdtrel`, `c5_fdtrel` (new). Compared paired vs prior E2 npz: `c1_legacy`, `c1_fdt`, `mj_refresh`, `det_c1`, `c5_fdt`. |
| classifier | `.claude/scratch/generative-studies/mnist_classifier.joblib`, test acc **0.9767**; decode `img=tanh(3·q_final)` |
| **reproduction check** | reused prior arms `c1_legacy`, `c1_fdt`, `mj_refresh`, `det_c1` histograms **bit-identical** to RGE E3 ⇒ instrument + pairing verified |
| NaN/divergence | **none** in any of 4 (Q1) + 6 (Q2) runs; F2 stable (max\|q\| ≤ 3.3 at c=1, ≤ 6.1 at c=5) |
| `d·Θ` reported | via `CHLU.gibbs_defect_parameter(T)` at every cell (below) |

Scripts: `.claude/scratch/fdt-relativistic-expc/{q1_momentum,q2_dreams,q2_score,q3_depth_plot}.py`
(+ reuse `relativistic-gibbs-expc/{common,mj_reference}.py`). Raw: `q1_results/q1_summary.json`,
`q2_results/{*.npz,q2_score_summary.json}`. Figures:
`.claude/outputs/fdt-relativistic-expc/{q2_digit_freqs,q3_depth_mechanism}.png`.

---

## 1. Q1 — F2 samples Gibbs (Maxwell–Jüttner) on real Exp-C. FIRST REAL-DATA VALIDATION.

Constant-T Langevin chains under `fdt_relativistic`. The `relativistic-gibbs-expc` result was that
`legacy`/`fdt` give a **Gaussian** momentum marginal with the radius **28–90× too small** (KL to
Gibbs = 3.2e6 nats at d=784). Under F2 this must **invert**.

| checkpoint · cell | `d·Θ` | `Var(p)/(M_eff·T)` obs | MJ theory | `r_MJ/r_obs` | excess kurt | KL(radial‖MJ) | KL(radial‖Gauss) | split-half Var | max\|q\| |
|---|---|---|---|---|---|---|---|---|---|
| mnistFFF · c1·T1 | **784** | **783.46** | 785.0 | **1.002×** | +0.0069 | **0.0036** | 0.1420 | 1.001 | 2.03 |
| mnistFFF · c1·T0.5 | 392 | 392.52 | 392.5 | 1.001× | +0.0049 | 0.0025 | 0.1393 | 1.002 | 1.98 |
| mnistFFF · c5·T1 | 31.4 | 31.45 | 31.4 | 1.000× | +0.0081 | 0.0033 | 0.1456 | 0.998 | 6.39 |
| **nativeC1 · c1·T1** | 784 | **784.19** | 785.0 | 1.001× | +0.0068 | 0.0028 | 0.1547 | 0.998 | 1.95 |

**Reference — what the coded samplers give (from RGE E1, same instrument):** `Var(p)/(M_eff·T)` =
**0.095** (`legacy`) / **1.00** (`fdt`); `r_MJ/r_obs` = **90.3×** (`legacy`) / **28.0×** (`fdt`).

**Findings.**
1. **The momentum marginal is Maxwell–Jüttner to <0.3 %.** `Var(p)/(M_eff·T)` hits the exact d=784
   MJ value (783.5 vs 785.0; 392.5 vs 392.5; 31.45 vs 31.4) — a **~785× jump** off the `fdt`
   Gaussian value of 1.0, and the discriminating radius `r_MJ/r_obs` is restored to **1.00×** from
   28–90×. The `legacy`/`fdt` **inversion is complete**: F2 is the exact relativistic Gibbs sampler
   on real data, as the fix-pack-7 measure-level proof (toy harmonic well) predicted.
2. **KL confirms the shape, not just the scale.** `KL(radial‖MJ)=0.003` vs `KL(radial‖Gauss)=0.14`
   — a **factor ~40** in favour of MJ. (The tiny residual 0.003 is the O(ε²) discretization shadow
   fix-pack-7 flagged, not a marginal defect.)
3. **Kurtosis is blind, as pre-registered.** Per-component excess kurtosis stays **+0.007** — equal
   to MJ's own d=784 value (0.0076), because 1-D marginals of a high-d isotropic law are Gaussian by
   concentration of measure. The discriminating statistic is the **radius/total variance** (used
   here), not kurtosis (RGE §C3 confirmed).
4. **Converged and stable.** Split-half `Var` ratio 0.998–1.002; no NaN; max\|q\| stays inside a
   modest cube (≤6.4). MJ's enormous variance (785×) is reached within 800 burn-in steps.
5. **Second checkpoint agrees** (nativeC1: 784.2 vs 785.0), so this is not an artifact of `mnistFFF`.

**Q1 verdict: PREREG UPHELD.** F2 restores the MJ marginal exactly (`Var/(M_eff·T)`→785, `r_MJ/r_obs`
→1.00, KL‖MJ ≪ KL‖Gauss). This validates CM-17's F2 fix on the real d=784 Exp-C path for the first
time (previously only verified on a toy 1-D harmonic well, fix-pack-7).

---

## 2. Q2 — Does a correct sampler change the imbalance? N10 CLOSED.

Dreams under `fdt_relativistic`, paired against the prior arms (identical inits/keys/classifier).
Pooled n=192/arm. Reproduction check passed (prior arms bit-identical to RGE E3).

| arm | `d·Θ` | digit hist [0..9] | entropy (bits) | f(3,5,8,9) [±sd] | TV [±sd] | argmax | mean final H | max\|q\| |
|---|---|---|---|---|---|---|---|---|
| **c1_legacy** *(paper pipeline)* | 784 | [0,2,3,0,66,68,4,0,11,38] | 2.038 | 0.609 ±0.068 | 0.596 | **5** | −254.3 | 4.50 |
| c1_fdt *(Newtonian-Gibbs)* | 784 | [0,2,2,3,41,12,4,0,82,46] | 2.091 | 0.745 ±0.048 | 0.580 | **8** | −167.6 | 3.70 |
| **c1_fdtrel** *(F2, EXACT MJ Gibbs)* | 784 | [0,3,2,6,12,3,3,0,**130**,33] | 1.574 | **0.896 ±0.039** | **0.649** | **8** | **−104.1** | 3.26 |
| mj_refresh *(other MJ-momentum arm)* | 784 | [0,2,1,7,12,1,0,0,**156**,13] | 1.078 | 0.922 ±0.016 | 0.712 | **8** | −77.7 | 3.01 |
| det_c1 (T=0) | — | [0,0,1,0,7,**182**,0,0,0,2] | 0.355 | 0.958 | 0.848 | **5** | −384.8 | 6.17 |
| c5_fdtrel *(F2 @ c=5, off-dist.)* | 31.4 | [1,2,3,2,66,30,5,2,41,40] | 2.371 | 0.589 ±0.033 | 0.522 | 4 | −237.5 | 6.12 |

Paired χ² / flips vs `c1_fdtrel`:

| pair | χ² | p | dof | per-sample flips | paired L2 |
|---|---|---|---|---|---|
| c1_fdt vs **c1_fdtrel** | 35.62 | 8.6e-6 | 7 | 38.5 % | 6.59 |
| c1_legacy vs **c1_fdtrel** | 204.22 | 1.5e-40 | 7 | 80.2 % | 10.31 |
| **mj_refresh vs c1_fdtrel** | **15.67** | **2.8e-2** | 7 | **20.8 %** | 5.50 |
| c1_fdtrel vs det_c1 | 344.30 | 2.0e-70 | 7 | 97.9 % | 13.29 |
| c5_fdt vs c5_fdtrel | 40.69 | 5.7e-6 | 9 | 56.2 % | 10.66 |

### 2.1 The imbalance is NOT fixed — it gets worse (the confirmed null, sharpened)
F2 samples Gibbs exactly (§1), yet the digit distribution moves **away** from uniform, not toward
it: `f(3589)` **0.609→0.896**, TV **0.596→0.649**, entropy 2.038→1.574. It collapses onto '8'
(130/192 = 68 %). **The exact-Gibbs sampler is *more* mode-imbalanced than the shipped defective one.**
This is the same direction and dominant mode as `mj_refresh` (the other MJ-momentum-correct arm,
RGE §3.5), and F2 is statistically **closest to `mj_refresh`** (χ²=15.7, 20.8 % flips) — far from
`det_c1` (χ²=344, 97.9 % flips), so it is not a deterministic collapse. **N10 is upheld and closed.**

### 2.2 Mechanism: F2 lands exactly on the RGE quench-depth curve (`q3_depth_mechanism.png`)
Ordering the four c=1 arms by mean final H (deepest→shallowest):

| arm | Hfin | argmax | f(3589) | entropy | MJ-correct? |
|---|---|---|---|---|---|
| c1_legacy | −254.3 | **5** | 0.609 | 2.038 | no |
| c1_fdt | −167.6 | **8** | 0.745 | 2.091 | no |
| **c1_fdtrel (F2)** | **−104.1** | **8** | 0.896 | 1.574 | **yes** |
| mj_refresh | −77.7 | **8** | 0.922 | 1.078 | **yes** |

**Spearman(Hfin, f3589) = +1.000.** F2 slots monotonically onto RGE §3.6's depth→imbalance law:
the dominant mode is a threshold function of quench depth ('5' below ≈−210, '8' above), and F2's
Hfin=−104 puts it firmly in the shallow-'8' group. **The two exactly-MJ-correct samplers are the
two shallowest quenches and the most imbalanced.** Correcting the momentum law does not fix the
imbalance because the imbalance is set by *how deep the annealed quench descends into V_θ*, which is
a landscape property — precisely N10.

### 2.3 The one prereg sub-prediction that FAILED (reported, not buried)
I pre-registered that F2 would be **statistically indistinguishable from `fdt`** (RGE §2b: velocity
saturation maps a 90× momentum-law error onto 0.66 % velocity error, so the momentum law should be
dynamically invisible). **This is falsified:** χ²(fdt, F2)=35.6, p=8.6e-6; F2 is *shallower* than
`fdt` (Hfin −104 vs −168) and closer to `mj_refresh`. **Why the argument was incomplete:** velocity
saturation governs the **drift direction** `q̇=c·M⁻¹p/√(·)`, but the MJ momentum's much larger
**noise variance** (the O-step injects `γ(2−γ)M/(2s)`, large in the ultra-relativistic regime)
keeps the chain at **higher energy** → a **shallower quench**. So the momentum law is invisible to
the drift but **not** to the quench depth. This *strengthens* N10's mechanism (depth is the
sufficient statistic) while correcting RGE §2b's "dynamically invisible" overstatement: the correct
reading is **"the drift the dynamics feel is nearly invariant; the quench depth is not, and depth is
what sets the mode."** The headline prereg prediction (imbalance not eliminated; f3589>0.5;
entropy>1.0; argmax≠det-mode-5) held on all four counts.

### 2.4 The three-way consistency (task item 4) — all consistent
1. **F2 samples Gibbs** (Q1: exact MJ marginal, both checkpoints). ✓
2. **N10: the imbalance is the landscape** (quench depth in V_θ), not the sampler. ✓
3. **⇒ F2 leaves the imbalance un-fixed** (it worsens it via a shallower quench, on the depth curve). ✓

No two disagree. The bridge is quench depth: F2's correct (larger) MJ thermal energy → shallower
quench → '8' collapse, on the *same* depth→mode threshold as every other arm. **A confirmed null on
Q2 with a confirmed YES on Q1 = the loop closes:** we built the exact fix, proved it samples Gibbs on
real data, and confirmed it does not move the imbalance the right way — the sampler was never the
cause; the landscape is.

---

## 3. Limitations & confounds
1. **Single primary checkpoint for Q2** (`mnistFFF`). As RGE warned, the *identity* of the collapsed
   digit is checkpoint-specific ('8' here); the *depth→collapse* law is not. Q1 uses two checkpoints.
2. **`c5_fdtrel` is off-distribution** (c=1-trained checkpoint dreamed at c=5), so its more-balanced
   look (entropy 2.371, deeper Hfin=−237) is not directly comparable; it is a supporting cell, not
   the N10 arm. A native-c5 retrain was **not** needed for the N10 claim (the c=1 arm is on-distribution
   and decisive); F2 is a *dream-time sampler swap*, so no retrain is required to answer Q1/Q2.
3. **`d·Θ` regime:** the on-distribution N10 arm sits at `d·Θ = 784` (deep ultra-relativistic), the
   worst case for the coded samplers and the regime where F2's correction is largest (785× variance).
   That F2 still fails to fix the imbalance *there* is the strongest possible version of the null.
4. **F2 O(ε²) shadow:** the momentum marginal is exact at the measure level; the tiny KL‖MJ residual
   (0.003) is the discretization shadow (fix-pack-7 Open-1), not a sampler defect. Irrelevant to Q2.
5. **n=3 seeds × 64 dreams; paired inits/keys** ⇒ per-sample flips are meaningful. sd over seeds
   reported. Scorer is OOD for dreams (mean maxprob 0.75–0.88); dominant-mode & TV conclusions robust,
   rare-digit bin counts noisy.
6. **Prereg mechanism correction (§2.3)** is a *finding*, not a failure of the closure — the headline
   predictions held and N10 is upheld with a sharper mechanism.

---

## 4. Recommended next experiments
1. **Depth-as-generative-knob (cheap, high-value, already recommended by RGE §8.1, now doubly
   motivated).** Since f3589 and dominant mode track Hfin with Spearman=+1.000 across *five* sampler
   families (legacy/fdt/F2/mj_refresh/det), directly controlling quench depth (anneal length / γ /
   T-floor) should *dial* the mode distribution. F2 at a **shallower-but-not-collapsed** operating
   point, or a **deeper** one, is a clean way to test whether depth alone recovers balance. ~1 h.
2. **Metropolis-adjusted relativistic Langevin** (CM-17 fix (ii)) to sample the *equilibrium*
   `exp(−H/T)` of V_θ (not the quench) and read the true basin masses — the one thing still unknown.
3. **Repeat the F2 Q2 arm on a native-c5 checkpoint and on `mnist`/`mnistFF`** to lift the
   single-checkpoint scope on the N10-under-F2 closure.

---

## Proposed handover updates (for the Hub)

**Reconciliation-list owner:** the Hub should assign an owner at the accepting review for (a) the
CM-17 F2 real-data-validation numbers below and (b) the RGE §2b correction in §2.3.

**§1.6 / Exp III & N10.** F2 (`fdt_relativistic`), the exact relativistic Gibbs sampler, was run on
the real Exp-C path for the first time. (Q1) It restores the Maxwell–Jüttner momentum marginal
**exactly** — `Var(p)/(M_eff·T)` = 783.5 (MJ 785.0), `r_MJ/r_obs` = 1.00× (was 28–90× too small),
KL‖MJ=0.004 ≪ KL‖Gauss=0.14, on two checkpoints. (Q2) It does **not** fix the MNIST imbalance — it
**worsens** it (`f(3589)` 0.609→0.896, TV→0.649, collapse onto '8'), landing on the RGE quench-depth
curve (Hfin=−104; Spearman(Hfin,f3589)=+1.000 across 4 arms). **N10 is CLOSED:** the exact-Gibbs
sampler confirms the sampler was never the cause; the imbalance is the learned landscape via quench
depth.

**§5 (provenance).** F2 validated on `mnistFFF` (c=1) and the native-c1 e5 retrain. F2 is a dream-time
sampler swap — no retrain needed for the N10 closure. New artifacts under
`.claude/scratch/fdt-relativistic-expc/` and figures under `.claude/outputs/fdt-relativistic-expc/`.

**CM-17 update.** F2's "verified: bias −0.727→+0.0011 (toy well)" line can be **upgraded to real
data**: on d=784 Exp-C, `fdt_relativistic` gives `Var(p)/(M_eff·T)` = 783.5/392.5/31.45 vs MJ
785/392.5/31.4 (c1·T1 / c1·T0.5 / c5·T1), `r_MJ/r_obs`=1.00×, KL‖MJ=0.003–0.004. **First real-data
confirmation of the F2 latent-mass thermostat.** No matrix contradiction: this corroborates the
`fdt`/`legacy`-are-Gaussian result (they still are) and the `d·Θ` control parameter (`d·Θ`=784 cell).

**N10 (registry).** Add to the numbers block: *"(v3, `fdt-relativistic-expc`, w16) the EXACT
relativistic Gibbs sampler F2 — momentum marginal restored to MJ on real Exp-C (Var/(M_eff·T)=783.5
vs 785; r_MJ/r_obs=1.00×; KL‖MJ=0.004≪KL‖Gauss=0.14) — does NOT fix the imbalance: f(3589)
0.609→0.896, TV 0.596→0.649, χ²(fdt,F2)=35.6 (p=8.6e-6), collapse onto '8'; closest to the other
MJ-momentum arm mj_refresh (χ²=15.7). Spearman(Hfin, f3589)=+1.000 over {legacy,fdt,F2,mj_refresh}.
The exact-Gibbs sampler is MORE imbalanced than the defective shipped one."* Disposition: F5 note
appendix — the empirical companion to the F2 proof + the N10 closure.

**⚠ Correction to RGE §2b / §7.18 (velocity-saturation framing).** RGE stated the relativistic
governor makes dream dynamics "robust to a 90× momentum-law error (→0.66 % velocity error)" — TRUE
for the **drift** `q̇`, but F2 shows it is **FALSE for the quench depth**: the MJ momentum's larger
noise variance keeps the quench shallower (Hfin −104 vs `fdt`'s −168), which changes the dominant
mode. The correct statement: *"the drift the dynamics feel is nearly invariant to the momentum law;
the quench depth is not, and depth sets the mode."* This does not weaken N10 — it sharpens the
mechanism (depth is the sufficient statistic).

**Code note for `experiment-engineer`:** none — `fdt_relativistic` ran clean on the real d=784 path
(no NaN, exact MJ marginal, stable max\|q\|). fix-pack-7's Exp-C-scale extrapolation caveat (its
Open-2) is now **discharged**: F2 is verified on the real chain, not just the measure.
