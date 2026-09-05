# relativistic-gibbs-expc — results-analyst report

**Task + acceptance criterion:** F-9 — does the missing Maxwell–Jüttner tail drive the MNIST 3/5/8/9 imbalance? Measure the momentum-marginal shape on the real Exp-C path; run the 2×2 (`c` × `langevin_noise`); handle the dynamics-vs-sampler confound; deliver an explicit verdict on **N10** with registry-ready text.

**Status: done.**

**Verdict: (a) N10 UPHELD, and materially strengthened.** The imbalance survives (i) `c=5`, (ii) `m₀=25`, and (iii) an **exactly-Gibbs Maxwell–Jüttner momentum law**. Two configurations with *numerically identical* MJ defect produce maximally different digit histograms (χ²=216.0, p=2.7e-42, 80.7 % flips) — so the digit law is **not a function of the defect**. Along the way, three of R8/F-9's own quantitative premises are corrected.

**This is not the free paper fix.** R8's proposed one-line remedy ("raise `c`") does **not** restore Gibbs at `d=784`, makes the dynamically-felt error **20× worse**, and — under the shipped `legacy` noise — empirically **destroys** Exp-C's generative behaviour (collapses it onto the deterministic mode, χ² vs `det_c1` = 1.27, p=0.74).

---

## 0. Flag-provenance (protocol §5, mandatory)

| item | value |
|---|---|
| commit (E0–E5) | **`d6f8bac`** ("[experiment-engineer] fix FDT noise inertia: effective_mass() -> effective_inertia()") = post-`fix-pack-5`, as task item 6 requires |
| commit (E6) | **`df5e44d`** (`integration/wave-14` fast-forwarded onto `main` at 12:36 mid-session, while my E5 process was already running). **Verified numerically inert:** `T`, `H`, `mass_vector`, `effective_inertia`, `effective_mass`, `step`, `stochastic_step` are **line-for-line identical** between the two commits (modulo docstrings and one `warnings.warn`). Both E5 models were trained by a process that imported `d6f8bac` (c=1 saved 12:20 < 12:36; the c=5 model was trained in the same already-imported process). E6 dreams imported `df5e44d`. **All results are therefore mutually comparable.** See §6b — the merge's *docstrings* are not inert as scientific claims. |
| repo state | **clean; 0 tracked files changed by me.** All artifacts under `.claude/` (gitignored). No code in `chlu/` touched. |
| env | Python 3.11.13 · **jax 0.9.0 (CpuDevice)** · equinox 0.13.4 · sklearn 1.8.0 · numpy 2.4.1 · scipy 1.17.0 |
| checkpoint (E1–E4) | `projects/mnistFFF/models/exp_c_chlu.pkl` — `kinetic_mode=relativistic`, `potential_type=conv`, `dim=784`, **`c=1.0`**, `m₀=1.0`, `M_eff` mean 0.6204, max/min **1.584** |
| checkpoints (E5–E6) | freshly trained, `.claude/scratch/relativistic-gibbs-expc/e5_models/exp_c_chlu_c{1,5}_ep500.pkl` |
| dream protocol | γ=0.3, dt=0.05, `T` exponential **1.0→0.01** over **1000** steps, init = centroid(first 10 000 imgs)+0.5·N(0,1), p₀=0.1·N(0,1), 64 dreams × seeds **{0,1,2}** (n=192/arm), **identical init states + identical chain keys across all arms (paired)** |
| E1 probe | constant-T chains, γ=0.3, dt=0.05, T∈{1.0, 0.5}, 64 chains, burn-in 500 + 100×10 steps ⇒ 6 400 momentum samples/arm |
| non-default flags in effect | `langevin_noise ∈ {legacy, fdt}` (varied; repo default `legacy`), `kinetic_mode=relativistic`, `potential_type=conv`, `c ∈ {1,5}`, `rest_mass ∈ {1,25}`, `tie_channel_mass=False`, `friction_field=None`, `persistent_sleep_buffer` n/a (dream-only), `sleep_temperature=0.5`, `experiment_c.friction=0.3` |
| E5 training flags | epochs=500, lr default, batch=64, dt=0.05, buffer=10000, reinit_prob=0.25, k_steps=100, clamp_outputs=True, energy_weight=1.0, **sleep γ = `experiment_c.friction` = 0.3** (see §6 bug), `sleep_temperature=0.5`, `input_noise_σ=0.05`, `langevin_noise=legacy`, seed=42 |
| scorer | `.claude/scratch/generative-studies/mnist_classifier.joblib` — the **same recorded artifact as `generative-studies` Study A**, test acc **0.9767**; decode `img = tanh(3·q_final)` (exp_c pca-None path) |
| reproduction check | `c1_legacy` pooled hist **bit-identical** to Study A.2 `legacy`; `c1_fdt` **bit-identical** to `fdt_nominal` ⇒ instrument + pipeline verified against the prior result |
| NaN/divergence | **none** in any of 27 (E2) + 10 (E1) + 12 (E6) runs |

Scripts: `.claude/scratch/relativistic-gibbs-expc/{mj_reference,common,smoke,e0_kl,e1_momentum_marginal,e1b_velocity,e1_plot,e2_dreams,e3_score,e4_supp,e5_retrain,e6_native_dreams}.py`
Figures: `.claude/outputs/relativistic-gibbs-expc/{f9_momentum_marginal,f9_digit_freqs,f9_sample_grids,f9_depth_vs_gibbsdefect}.png`
Raw: `.../e1_results/`, `.../e2_results/{e3_summary,e4_supp}.json`, `.../e6_results/e6_summary.json`

---

## 1. Three corrections to F-9's own premises (before any experiment)

F-9 and R8 index the defect with a **d = 1** table. **Exp-C runs at d = 784** (`pca_dim=784`, conv potential). The Gibbs momentum marginal is `π(p) ∝ exp(−T(p)/T)` with `T(p)=c√(pᵀM⁻¹p+m₀²c²)`; in scaled coordinates `u=M^{-1/2}p` its radial density is `r^{d−1}e^{−β√(r²+a²)}` (β=c/T, a=m₀c), which concentrates at `r ≈ (d−1)T/c`, **not** `√(dT)`.

My quadrature reproduces R8's d=1 table to 4 digits (validation, `mj_reference.py`):

| `T/(m₀c²)` | `Var_MJ/(M_eff T)` mine | R8 | excess kurt mine | R8 | `KL(MJ‖Gauss)` mine | R8 |
|---|---|---|---|---|---|---|
| 0.01 | 1.0150 | 1.0150 | 0.030 | 0.030 | 7.0e-5 | 7.4e-5 |
| 0.10 | 1.1533 | 1.1534 | 0.296 | 0.295 | 6.77e-3 | 6.8e-3 |
| **1.00** | **2.6988** | 2.6995 | **1.858** | 1.857 | **0.3837** | 0.384 |
| 8.00 | 16.2756 | 16.282 | 2.910 | 2.907 | 6.313 | 6.31 |

Applying the *same* instrument at the *actual* dimension:

**C1 — the variance defect is ~290× larger than stated.** In the ultra-relativistic regime `Var_MJ/(M_eff·T) → (d+1)·T/(m₀c²)` (verified: d=1→2.699, d=10→11.11, d=100→101.0, d=784→**785.00**, asymptote 785.0).

| c | T | `T/(m₀c²)` | `Var_MJ/(M_eff T)` @ d=784 | R8's d=1 number |
|---|---|---|---|---|
| 1 | 1.0 | 1.00 | **785.00** | 2.70 |
| 1 | 0.5 | 0.50 | 392.50 | ~1.9 |
| 5 | 1.0 | 0.04 | **31.43** | ~1.06 |
| 5 | 0.01 | 0.0004 | 1.17 | ~1.00 |

**C2 — `c=5` is *not* benign.** At `T/(m₀c²)=0.04` the residual variance error is **31.4×**, and `KL(Gibbs ‖ coded) = 10 578 nats` (`e0_kl.py`). R8's inference "`finalA` used c=5 ⇒ benign — which may be why it behaved better" does not survive d=784. Full KL at the real operating point:

| arm | `T/(m₀c²)` | KL(Gibbs‖coded), d=784 | KL/dim | R8's d=1 KL |
|---|---|---|---|---|
| c1_legacy, T=1 | 1.00 | **3 241 788 nats** | 4134.9 | — |
| c1_fdt, T=1 | 1.00 | **304 716 nats** | 388.7 | 0.384 |
| c5_legacy, T=1 | 0.04 | 127 295 nats | 162.4 | — |
| c5_fdt, T=1 | 0.04 | 10 577.9 nats | 13.49 | 0.0011 |
| **m25_fdt, T=1** | **0.04** | **10 577.9 nats** | 13.49 | 0.0011 |

(`c5_fdt` and `m25_fdt` agree to all printed digits — the defect is a function of `T/(m₀c²)` **alone**, exactly as R8 claims. I use this below as the confound-splitter.)

**C3 — the instrument F-9 specifies is blind.** Task item 1 predicts "excess kurtosis ≈ 1.86". At d=784 the **MJ law's own** per-component excess kurtosis is **0.0076** (concentration of measure: 1-D marginals of any high-d isotropic law are near-Gaussian). Gaussian gives 0. So the statistic F-9 names cannot distinguish the coded sampler from Gibbs — the gap is 0.008, not 1.86. **The discriminating observable is the radial norm `r=‖M^{-1/2}p‖`.** I measured both.

---

## 2. Item 1 — momentum-marginal shape on the real path

`e1_momentum_marginal.py`, 6 400 samples/arm, no NaNs. Split-half stationarity of `Var(pᵢ)`: median ratio 0.997–1.003; of `r̄`: 0.999–1.001 in all 10 arms.

| arm | `T/(m₀c²)` | `Var(p)/(M_eff·T)` | coded pred. | excess kurt | MJ kurt (d=784) | `r̄` obs | `r` coded pred. | `r_MJ` | **`r_MJ/r_obs`** | max\|q\| |
|---|---|---|---|---|---|---|---|---|---|---|
| c1_legacy T=1 | 1.00 | 0.0953 | 0.0954 | **−0.0008** | 0.0076 | 8.69 | 8.65 | 784.5 | **90.3×** | 2.76 |
| c1_fdt T=1 | 1.00 | 1.0007 | 1.0000 | −0.0035 | 0.0076 | 28.02 | 28.01 | 784.5 | **28.0×** | 2.02 |
| c5_legacy T=1 | 0.04 | 0.0953 | 0.0954 | −0.0010 | 0.0076 | 8.68 | 8.64 | 157.0 | 18.1× | 8.03 |
| c5_fdt T=1 | 0.04 | 0.9998 | 1.0000 | −0.0002 | 0.0076 | 28.01 | 28.00 | 157.0 | 5.6× | 5.92 |
| c1_m25_fdt T=1 | 0.04 | 1.0006 | 1.0000 | −0.0065 | 0.0076 | 140.02 | 140.04 | 784.9 | 5.6× | 1.92 |

(T=0.5 block in `e1_results/e1_summary.json`; same pattern.)

**Findings.**
1. **The coded chain's momentum marginal is Gaussian to 4 decimals in every arm** (|excess kurt| ≤ 0.0065). R8's proof (linear OU O-step ⇒ Gaussian stationary law) holds exactly on the real trained model, with a conv potential and relativistic coupling live.
2. **Variance matches the closed forms exactly**: legacy `2dtT/((2−γ)M_eff T)` = 0.0954 (obs 0.0953); fdt `= 1` (obs 1.0007). Post-`fix-pack-5`, `fdt` delivers exact Maxwell–**Boltzmann**.
3. **`fdt` is *not* the Gibbs sampler in relativistic mode.** It is Gibbs for the *Newtonian* `½pᵀM⁻¹p`. Gibbs for the coded `T(p)` demands `r̄ = 784.5`; `fdt` delivers 28.0. Confirms R8/X6, and means **the phrase "the FDT fix" must carry a kinetic-mode qualifier.**
4. **The defect is live and enormous on the real path**: at the anneal start the momentum radius is **90.3× too small** (legacy) / **28.0× too small** (fdt). Yes/no answer to item 1: **yes, live.**

### 2b. Why it nevertheless cannot matter — the velocity map saturates

`q̇ = c·M⁻¹p/√(pᵀM⁻¹p+m₀²c²)`, so the dynamics see only `v/c = r/√(r²+(m₀c)²)` — a bounded, concave function of `r` (`e1b_velocity.py`):

| arm | `r_MJ/r_obs` (momentum error) | `v/c` obs | `v/c` MJ | **relative velocity error** |
|---|---|---|---|---|
| **c1_legacy** (the paper pipeline) | **90.3×** | 0.99343 | 1.00000 | **0.66 %** |
| c1_fdt | 28.0× | 0.99936 | 1.00000 | 0.06 % |
| **c5_legacy** | 18.1× | 0.86644 | 0.99949 | **13.31 %** |
| c5_fdt | 5.6× | 0.98441 | 0.99949 | 1.51 % |
| m25_fdt | 5.6× | 0.98440 | 0.99949 | 1.51 % |
| c5_legacy @T=0.5 | 12.7× | 0.77668 | 0.99797 | **22.17 %** |

**A 90× error in the momentum distribution produces a 0.66 % error in the velocity the dynamics feel.** The causal speed cap absorbs essentially the entire Gibbs defect.

**And the inversion that kills R8's fix (iii):** the regime where the defect is *largest* (`c=1`, ultra-relativistic, `r≫m₀c`) is the regime where it matters *least*, because `∂v/∂r→0`. Raising `c` shrinks the defect ratio (90.3→18.1) while moving the chain off the saturation plateau, so the dynamically-felt error **grows 20×** (0.66 %→13.31 %). This is a physics result in CHLU's favour: **the relativistic kinetic governor makes the dream dynamics robust to a two-order-of-magnitude error in the sampler's momentum law.**

---

## 3. Items 2 & 3 — the 2×2 and the confound-split

`e2_dreams.py` + `e3_score.py`; 9 arms × 3 seeds × 64 dreams, paired. Pooled n=192.

| arm | `T/(m₀c²)` | digit hist [0..9] | entropy (bits) | TV vs unif [95 % CI] | f(3,5,8,9) | f(4,5,9) | argmax | mean final H | max\|q\| |
|---|---|---|---|---|---|---|---|---|---|
| **c1_legacy** *(= paper pipeline)* | 1.00 | [0,2,3,0,66,68,4,0,11,38] | 2.038 | 0.596 [0.554,0.637] | 0.609 | 0.896 | 5 | −254.3 | 4.50 |
| c1_fdt | 1.00 | [0,2,2,3,41,12,4,0,82,46] | 2.091 | 0.580 [0.533,0.622] | 0.745 | 0.516 | 8 | −167.6 | 3.70 |
| **c5_legacy** | 0.04 | [0,0,0,0,8,**181**,0,0,0,3] | **0.365** | **0.843** [0.806,0.874] | 0.958 | 1.000 | 5 | −420.8 | 7.68 |
| c5_fdt | 0.04 | [0,1,4,0,65,74,1,0,14,33] | 1.967 | 0.596 [0.558,0.638] | 0.630 | 0.896 | 5 | −295.3 | 6.47 |
| m25_legacy | 0.04 | [0,2,1,3,23,5,2,0,**131**,25] | 1.533 | 0.632 [0.596,0.670] | 0.854 | 0.276 | 8 | −97.9 | 3.09 |
| m25_fdt | 0.04 | [0,4,1,7,15,2,0,1,**139**,23] | 1.430 | 0.644 [0.601,0.696] | 0.891 | 0.208 | 8 | −77.7 | 3.17 |
| **mj_refresh** *(exact Gibbs momentum)* | 1.00 | [0,2,1,7,12,1,0,0,**156**,13] | **1.078** | **0.712** [0.660,0.765] | 0.922 | 0.135 | 8 | −77.7 | 3.01 |
| det_c1 (T=0) | — | [0,0,1,0,7,**182**,0,0,0,2] | 0.355 | 0.848 [0.817,0.874] | 0.958 | 0.995 | 5 | −384.8 | 6.17 |
| det_c5 (T=0) | — | [0,1,0,0,0,**191**,0,0,0,0] | 0.047 | 0.895 [0.884,0.900] | 0.995 | 0.995 | 5 | −484.2 | 8.03 |

### 3.1 The F-9 prediction fails, in both cells

> F-9: *"if the missing MJ tails drive the imbalance, the 3/5/8/9 over-representation should measurably **shrink** at `c=5`."*

- **legacy:** f(3589) **0.609 → 0.958**; entropy 2.038 → **0.365** bits; TV 0.596 → **0.843**. The imbalance **explodes**. `c5_legacy` is statistically **indistinguishable from the deterministic T=0 collapse**: χ²(c5_legacy, det_c1) = **1.27, p = 0.74**, only **5.7 % flips**.
- **fdt:** f(3589) 0.609 → 0.630, TV 0.596 → 0.596. `c5_fdt` is distributionally **indistinguishable from the paper pipeline**: χ²(c1_legacy, c5_fdt) = **3.25, p = 0.78**.

Neither cell shrinks the imbalance. Under `legacy` — the shipped default and the setting the paper's Exp III actually ran — **raising `c` converts the stochastic dreamer into the deterministic mode-collapse**, because the drift speeds up 4.4× (v: 0.993→4.33) at fixed noise `σ=√(2γTdt)` (which is `c`-independent).

### 3.2 The 2×2 is dominated by interaction ⇒ `c` is not acting through a sampler channel

Per-seed factorial (mean ± sd over 3 seeds):

| metric | main effect of `c` | main effect of noise | **interaction** |
|---|---|---|---|
| entropy (bits) | −0.872 ± 0.217 | +0.788 ± 0.105 | **+1.504 ± 0.505** |
| TV | +0.134 ± 0.043 | −0.129 ± 0.016 | **−0.220 ± 0.099** |
| f(3,5,8,9) | +0.117 ± 0.064 | −0.096 ± 0.055 | **−0.464 ± 0.140** |

The interaction is larger than either main effect: **the sign of the `c` effect flips with noise mode.** A defect that lived in the momentum marginal's *shape* would not behave this way (the shape is Gaussian in all four cells, §2).

### 3.3 The structural reason F-9 cannot test what it wants to test

Measured in `smoke.py`, exactly:
- `σ_legacy = √(2γTdt)` — **independent of `c`** (0.173205 at both c=1 and c=5).
- `σ_fdt = √(M_eff·T·γ(2−γ))`, `M_eff = m₀(M+1e-6)` — **independent of `c`** (mean 0.562111 at both).
- `∇_q H` — **bit-identical** across `c` (`jnp.allclose` = True; V doesn't depend on p).
- `∇_p T` — the **only** thing `c` changes (‖·‖ = 1.260 → 5.175).

⇒ **`c` is a drift knob, not a sampler knob.** At fixed `noise_mode` the coded momentum recursion's damping and noise are unchanged; the stationary law stays Gaussian. Raising `c` does not move the sampler toward Gibbs — it moves the *Gibbs target* toward the (unchanged) Gaussian, and simultaneously rescales the dynamics. **The `c ∈ {1,5}` sweep therefore confounds "distance to Gibbs" with "drift speed" irreparably**, which is exactly the failure mode F-9 was written to avoid in N10.

### 3.4 The confound-split (task item 5, done properly)

`c↑` and `m₀↑` reduce `T/(m₀c²)` **identically** but move drift in **opposite** directions (`‖∇_pT‖→c` vs `→p/(m₀M)`). So I built a **matched-defect, mismatched-dynamics** pair:

| | `T/(m₀c²)` | KL(Gibbs‖coded) | `r_MJ/r_obs` | rel. velocity error | drift speed `v` |
|---|---|---|---|---|---|
| `c5_fdt` | 0.04 | 10 577.9 nats | 5.6× | 1.509 % | **4.92** |
| `m25_fdt` | 0.04 | 10 577.9 nats | 5.6× | 1.509 % | **0.984** |

Every quantity by which R8/F-9 index the defect is **identical to all printed digits**. Only the dynamics differ (5×).

> **Result: χ² = 215.97, p = 2.7e-42, dof = 8, 80.7 % per-sample flips, paired L2 = 11.68.**
> `c5_fdt` → [0,1,4,0,65,74,1,0,14,33] (argmax **5**) vs `m25_fdt` → [0,4,1,7,15,2,0,1,139,23] (argmax **8**).
> Same under legacy: χ²(c5_legacy, m25_legacy) = **330.1, p = 2.3e-67, 95.8 % flips.**

**Two configurations with a numerically identical Maxwell–Jüttner defect produce maximally different digit distributions.** Therefore the digit distribution **is not a function of the MJ defect**. This falsifies F-9's mechanism *without needing the `c=5` arm to be interpretable at all.*

### 3.5 The Gibbs-momentum arm (item 3: "closest thing to a Gibbs sampler")

`mj_refresh`: momentum resampled **exactly** from the d=784 MJ law at `T_t` at every step (inverse-CDF on the radial density × uniform direction, then `p = M^{1/2}u`), followed by one Verlet step at γ=0. Its momentum marginal is Gibbs **by construction at every step** — the defect R8 identified is *removed*, not mitigated. (Validated: sampled `Var/(M_eff T)` = 784.26 vs theory 785.00 at c=1; 31.40 vs 31.40 at c=5.)

> **The imbalance does not shrink — it grows.** entropy 2.038 → **1.078** bits; TV 0.596 → **0.712** (bootstrap CIs [0.554,0.637] vs [0.660,0.765], **non-overlapping**); f(3,5,8,9) 0.609 → **0.922**. χ²(c1_legacy, mj_refresh) = **252.6, p = 7.8e-51**, 85.4 % flips. It collapses onto '8' (81 %).

**This is the closest thing to a Gibbs sampler this codebase can run** — and it is *more* mode-imbalanced than the defective shipped sampler. (Caveat §5.2: full refresh also destroys momentum correlation, so it is not a clean sampler-only swap. It is nonetheless a direct test of "would the correct MJ momentum law remove the imbalance?" Answer: no.)

### 3.6 The positive mechanism: quench depth, not sampler

Ordering all 9 arms by mean final `H`:

| arm | mean final H | argmax digit | entropy |
|---|---|---|---|
| det_c5 | −484.3 | **5** | 0.047 |
| c5_legacy | −420.8 | **5** | 0.365 |
| det_c1 | −384.8 | **5** | 0.355 |
| c5_fdt | −295.3 | **5** | 1.967 |
| c1_legacy | −254.3 | **5** | 2.038 |
| — *threshold ≈ −210* — | | | |
| c1_fdt | −167.6 | **8** | 2.091 |
| m25_legacy | −97.9 | **8** | 1.533 |
| mj_refresh | −77.7 | **8** | 1.078 |
| m25_fdt | −77.7 | **8** | 1.430 |

**Perfect separation of the dominant mode by quench depth** (Mann–Whitney U=0, one-sided **p=0.0097**, n₁=5 deep→'5', n₂=4 shallow→'8'; gap 86.7 energy units). '5' is this checkpoint's deepest accessible basin (it is what the T=0 control collapses to); '8' is where a quench stops if it does not descend far.

Entropy vs depth is an **inverted-U**, not monotone — hence `Spearman(H, entropy) = +0.483, p=0.19 (n.s.)`, reported honestly. Diversity is maximal at intermediate depth; both deep and shallow quenches collapse, onto different digits. And `Spearman(Gibbs defect, entropy) = +0.580, p=0.23 (n.s.)` — with the **wrong sign**: the arms *furthest* from Gibbs are the *most* diverse.

**Everything that changes the digit law does so by changing how deep the 1000-step quench descends into `V_θ`.** That is N10's "learned landscape" conclusion, now with a mechanism and a sufficient statistic.

### 3.7 The deeper methodological point

Exp-C's dream is a **1000-step annealed non-equilibrium quench** (`T: 1.0→0.01`), not equilibrium sampling. Study A already established there is **no positional stationary state** (mean H falls monotonically; chains exit the trained cube). My `max|q|` values (up to 8.03 vs a data cube of [−1,1]) confirm it. **The stationary law of the sampler is therefore not the object that produces the digits**, and both R8's conjecture *and* N10's original framing attack a mechanism the generation process never invokes. That is why a 3.2-million-nat error in the momentum law is invisible in the output, while a 4.4× change in drift speed rewrites it entirely.

---

## 4. Item 5 — the dynamics-vs-sampler confound, and the native-`c` control

**Stated plainly, as the task requires: every `mnist*` checkpoint was trained at `speed_of_causality = 1.0`** (verified in all four `projects/mnist*/config/config.yaml`). So in E2 the `c=5` and `m₀=25` arms **dream an off-distribution potential**. Direct evidence: `max|q|` rises 4.50 → 7.68 (c5_legacy) and the chains leave the `[−1,1]` cube the conv potential was trained on and the training loop clamps to.

To close this, `e5_retrain.py` trains **two Exp-C models from the same seed (42), same 10 k data, same init key, differing only in `c`** (500 epochs each, `langevin_noise=legacy` as `mnistFFF` was trained), and `e6_native_dreams.py` dreams each **natively** (same protocol/seeds/paired keys/scorer). Both converged comparably and neither NaN'd:

| model | wall | total loss (first-50 → last-50) | wake | sleep | target_E |
|---|---|---|---|---|---|
| native c=1 | 22.1 min | −7.87 → **−198.12** | −98.95 | −99.17 | −122.094 |
| native c=5 | 24.0 min | −16.35 → **−196.14** | −98.01 | −98.13 | −120.443 |

**Result — F-9's prediction fails again, on matched dynamics:**

| arm | digit hist [0..9] | entropy | TV | f(3,5,8,9) | argmax | mean final H | max\|q\| |
|---|---|---|---|---|---|---|---|
| nat_c1_legacy | [0,16,31,33,27,32,14,11,2,26] | **2.960** | 0.276 | 0.484 | 3 | −450.1 | 4.81 |
| nat_c1_fdt | [0,13,16,45,21,27,11,16,21,22] | **3.042** | 0.208 | 0.599 | 3 | −253.6 | 3.52 |
| nat_c5_legacy | [0,6,6,24,13,4,0,19,0,**120**] | **1.821** | 0.550 | 0.771 | 9 | −1111.5 | 11.14 |
| nat_c5_fdt | [0,0,3,24,17,10,1,20,6,**111**] | **1.993** | 0.507 | 0.786 | 9 | −777.6 | 8.60 |

- **legacy:** f(3589) 0.484 → **0.771** (Δ **+0.286**); TV 0.276 → 0.550; entropy 2.960 → 1.821 bits. χ² = **128.19, p = 6.7e-24**, 78.6 % flips.
- **fdt:** f(3589) 0.599 → **0.786** (Δ **+0.188**); TV 0.208 → 0.507; entropy 3.042 → 1.993 bits. χ² = **113.19, p = 8.4e-21**, 70.8 % flips.

**In both noise modes the 3/5/8/9 over-representation *grows* at `c=5`, on a potential trained at `c=5`.** The off-distribution confound is therefore **not** what produced the E2 result — `c=5` genuinely makes Exp-C's mode imbalance worse, natively.

**The depth mechanism reproduces on this second, independent landscape.** Ordering the four native arms by mean final H: `nat_c1_fdt` (−253.6, 3.042 bits) → `nat_c1_legacy` (−450.1, 2.960) → `nat_c5_fdt` (−777.6, 1.993) → `nat_c5_legacy` (−1111.5, 1.821). **Spearman(H, entropy) = +1.000, Spearman(H, TV) = −1.000** (all four sit on the deep branch of §3.6's inverted-U). Dominant mode flips 3 → 9 at depth, exactly as it flipped 8 → 5 on `mnistFFF`. The *identity* of the collapsed digit is checkpoint-specific (as Study A warned); the *depth → collapse* law is not.

Two further notes:
- The natively-trained `c=1` model is **far more balanced** than `mnistFFF` (entropy 2.96–3.04 bits vs 2.04; TV 0.21–0.28 vs 0.60). Expected: `mnistFFF`'s 10 k subsample is unseeded (§7.11) and its init differs. This *strengthens* the landscape reading — the imbalance is a property of the particular learned `V_θ`, and it moves a lot when `V_θ` moves, while being inert to the sampler's momentum law.
- `max|q|` reaches **11.14** for `nat_c5_legacy`: even trained natively at `c=5`, the dream leaves the `[−1,1]` cube (conv potential is architecturally non-coercive, §7.7). `c=5` dreaming is intrinsically a longer, deeper excursion — which is precisely why it collapses.

---

## 5. Limitations & confounds

1. **Single checkpoint for E1–E4** (`mnistFFF`). Study A showed the over-represented set is checkpoint-dependent ({4,5,9}+8 here vs the paper's {3,5,8,9}); conclusions are about *imbalance*, not about which digits.
2. **`mj_refresh` is unadjusted and over-mixes.** No Metropolis correction (O(dt²) bias in q), and full momentum refresh decorrelates momentum every step, so it changes the dream dynamics as well as the momentum law. It is the closest available Gibbs-momentum sampler, not a clean sampler-only swap. **The confound-split (§3.4) is the load-bearing evidence; `mj_refresh` corroborates it.**
3. **`m25` changes `M_eff`** (=25·M), hence `σ_fdt` scales by √25. It is *not* "the same sampler". What is matched — and all that F-9 predicates on — is the **defect**: `T/(m₀c²)`, `r_MJ/r_obs`, KL, and relative velocity error, all identical to printed precision.
4. **n = 3 noise seeds × 64 dreams = 192/arm.** Init states and chain keys are paired across arms, so per-sample flip counts are meaningful. The 95 % TV CIs are bootstrap over dreams, not over seeds; seed-level sd is reported in the factorial table.
5. **The scorer is OOD for dreams** (trained on real MNIST; mean maxprob 0.75–0.93). Bin-level counts for rare digits are noisy; the dominant-mode and TV conclusions are not.
6. **`Spearman(H, entropy)` is not significant** (p=0.19, n=9). Only the *dominant-mode* separation by depth is (p=0.0097). Stated as such; I do not claim a monotone depth→entropy law.
7. **E5/E6's `c=1` model is not `mnistFFF`** — `load_mnist_pca`'s subsample is unseeded (§7.11), and I train on `X[:10000]` directly. E5/E6 is an internally-matched pair, not a reproduction of `mnistFFF`.

---

## 6. Code bug found (for `experiment-engineer`)

**`training.sleep_friction` is a silent no-op on the Exp-C / generative path.** `chlu/training/train_generative.py:102-103`:

```python
if sleep_friction is None:
    sleep_friction = config.experiment_c.friction     # <-- NOT config.training.sleep_friction
```

`sleep_temperature` *is* read from `config.training` (line 104-105). So on `train_generative` the sleep phase runs at γ = `experiment_c.friction` (0.3 for `mnistFFF`), while `training.sleep_friction` (=0.0) is ignored. Consequences:
- Exp-C training **is** stochastic in the sleep phase (γ=0.3, T=0.5, legacy noise) — `langevin_noise` is **live during Exp-C training**, not just at dream time. (I had to correct my own retrain design for this.)
- **N19 ("`sleep_temperature` is a no-op whenever `sleep_friction=0`") does not apply to Exp-C**, because the 0.0 never reaches the sampler. N19's scope line should say "dynamics path (`train_chlu`)".
- This is a **fourth silent knob** (after N17/N18/N19-N20 class). Not a physics bug; a config-plumbing bug. Either honour `training.sleep_friction` or delete it from the schema. Low risk, but it silently changes what a reader thinks Exp-C trained under.

---

## 6b. ⚠ URGENT — `df5e44d` has just written the refuted claims into tracked code

While this task ran, `integration/wave-14` merged to `main` (`df5e44d`) and landed a CM-17 guard-rail: `chlu/core/chlu_unit.py` gains `thermal_causal_ratio()` + a `RelativisticGibbsWarning`, and `chlu/core/integrators.py` / `train_generative.py` gain matching docstrings. **The mechanism (linear OU ⇒ Gaussian; no σ works; the defect is a function of `T/(m₀c²)` alone) is exactly right — my data confirms all three.** But the *quantitative* and *prescriptive* content is now wrong in tracked code:

1. **`integrators.py` and `CHLU.thermal_causal_ratio` quote the d=1 table** — "`Var_MJ/(M_eff*T) = 1.015 / 1.153 / 2.70 / 16.28`, `KL = 7.4e-5 / 6.8e-3 / 0.384 / 6.31 nats`" — with **no dimension qualifier**, in a docstring whose only named consumer is **Exp-C at `d=784`**, where the true values are **785** and **3.24e6 nats** (§1). Off by 290× and 8.4e6× respectively.
2. **Three sites now recommend the refuted fix**: *"Free mitigation: raise `c` or `rest_mass` until T ≪ m₀c²"* (`integrators.py`), *"the free mitigation is to raise `model.speed_of_causality` or `model.rest_mass`"* (`train_generative.py`), *"`finalA` used `c=5` ⇒ ratio 0.04, **benign**"* (`thermal_causal_ratio`). At `d=784`, `c=5` leaves a **31.4×** variance error and **10 578 nats**; it makes the dynamically-felt velocity error **20× worse** (0.66 %→13.31 %); and empirically it **destroys** Exp-C generation under the shipped `legacy` noise (χ² vs `det_c1` = 1.27, p=0.74) and **worsens** the imbalance on natively-trained `c=5` models (f(3589) +0.286 legacy / +0.188 fdt).
3. The runtime `RelativisticGibbsWarning` **fired during my E6 run** and told the user, verbatim: *"the defect vanishes as this ratio → 0. Free mitigation: raise c or rest_mass."* Users will act on this and silently degrade their generative runs.

**Recommended for `experiment-engineer` (small, surgical, docstring-only + one method docstring):**
- Qualify every number as **d=1**, and add the general law `Var_MJ/(M_eff·T) → (d+1)·T/(m₀c²)` (ultra-relativistic), with the d=784 values for Exp-C.
- Replace "free mitigation: raise c" with: *"raising `c` shrinks the ratio but does **not** restore Gibbs at large `d`, and moves the chain off the velocity-saturation plateau where the residual error becomes dynamically visible; empirically it collapses Exp-C generation. The only correct fixes are Metropolis adjustment or an exact Maxwell–Jüttner momentum refresh — and the latter does not improve Exp-C's mode balance (N10)."*
- Delete "`finalA` used c=5 ⇒ benign" or restate it as "`finalA`'s ratio is 0.04; the residual d=784 defect there is still ≈31×, so its better behaviour is **not** explained by Gibbs-ness."
- Keep the warning; fix its remediation sentence.

This is **arXiv-bound text in tracked code**, and it is the same failure mode as the F5 kinetic-isotropy clause (§7.19): a correct theorem shipped with a wrong corollary.

---

## 7. Verdict on N10 (task item 4)

**(a) UPHELD — and on materially stronger ground.** The task's worry was legitimate: both N10 arms *were* non-Gibbs samplers, and its instrument (variance-level, per-mode `T_eff`) *was* blind to a tail-shape error. I re-opened it with an instrument that can see the second defect, and with a sampler that does not have it.

The imbalance survives:
- `c=5` (both noise modes) — it **worsens** under `legacy`;
- `c=5` **on models natively trained at `c=5`** (off-distribution confound removed): f(3589) +0.286 (legacy) / +0.188 (fdt), entropy 2.960→1.821 / 3.042→1.993 bits;
- `m₀=25`;
- an **exactly-Gibbs Maxwell–Jüttner momentum law** (`mj_refresh`) — it **worsens**, TV 0.596→0.712, non-overlapping CIs;
- and, decisively, **two arms with numerically identical MJ defect give maximally different histograms** (χ²=216.0, p=2.7e-42) ⇒ the digit law is not a function of the defect.

N10's *scope* should be **widened**, and its *mechanism sharpened*: the imbalance is set by the depth the annealed quench reaches in `V_θ` (dominant mode separates perfectly by final H, p=0.0097), not by any property of the sampler's stationary law — and, in relativistic mode, it *cannot* be, because velocity saturation converts a 90× momentum-law error into a 0.66 % dynamical error.

**It is not the free paper fix (b).** The opposite: R8's recommended remedy would have degraded Exp III.

### Registry-ready text (C-9: negatives are never dropped)

Proposed **replacement** for `negative_results.md` N10, and a new **N10b**:

```markdown
### N10 — the sampler's momentum law does NOT drive the MNIST digit-mode imbalance · tier A · generative/F-layer
- **Tried:** (v1, `generative-studies A`) the F5 Prop-9 conjecture — per-mode temperature violation explains
  digit over-/under-representation. (v2, `relativistic-gibbs-expc`, w14) **re-opened** because both v1 arms ran
  `kinetic_mode=relativistic`, where R8 proves *no* σ gives the coded Langevin a Gibbs invariant: v1 tested one
  sampler defect while a second (missing Maxwell–Jüttner tails) was present in both arms and invisible to a
  variance-level instrument.
- **Numbers (v1):** scale-matched fdt χ²=**0.08**, p=**1.00**, 2/192 flips. Mechanism exact: legacy slope
  log Var(p) vs log M = 0.002, ρ(T_eff,1/M)=0.994; fdt restores slope 1.000.
- **Numbers (v2, the stronger closure):** on the real Exp-C path the momentum marginal is **Gaussian to 4 dp**
  (|excess kurt| ≤ 0.0065) where Gibbs demands MJ; the radius is **90.3× too small** (legacy, anneal start),
  KL(Gibbs‖coded) = **3.24e6 nats** at d=784. Correcting it *exactly* (`mj_refresh`, MJ momentum resampled every
  step) does **not** shrink the imbalance: TV **0.596 → 0.712** (bootstrap CIs non-overlapping), entropy
  2.038 → 1.078 bits, χ²=**252.6**, p=7.8e-51. Decisive control: `c5_fdt` and `m25_fdt` have **identical**
  T/(m₀c²)=0.04, identical KL (10 577.9 nats), identical r_MJ/r_obs (5.6×) and identical relative velocity error
  (1.509 %) — yet χ²=**216.0**, p=2.7e-42, **80.7 % flips**, different dominant mode. The digit law is **not a
  function of the defect**.
- **Mechanism:** the dominant mode is a threshold function of **quench depth** (mean final H): all arms with
  H < −250 collapse to '5' (the deepest basin, = the T=0 control), all with H > −168 to '8'; perfect separation,
  Mann–Whitney one-sided p=**0.0097** (n=9 arms). Reproduced on a second, independently trained landscape
  (native c∈{1,5}, §4): Spearman(H, entropy) = **+1.000**, dominant mode flips 3→9 with depth. Exp-C's dream is an
  annealed **non-equilibrium quench**, so the sampler's stationary law is not the object that generates digits. In
  relativistic mode it *cannot* be: velocity saturation maps a 90.3× momentum-law error onto a **0.66 %** error in q̇ (§2b).
- **Scope:** widened from "the FDT per-mode-T_eff violation" to "**any property of the sampler's momentum law**",
  for `kinetic_mode=relativistic` at d=784. Two independently trained landscapes (mnistFFF + a matched native-c
  retrain pair, seed 42). Conditional promotion (narrow M, N7) is unchanged.
- **Disposition:** generative/F-layer appendix. A negative that was re-opened with a better instrument and a
  nearly-correct sampler, and re-closed harder.

### N10b — R8's "free fix" (raise c) is refuted: it does not restore Gibbs and it destroys Exp-C generation · tier A · generative
- **Tried:** R8's fix (iii) "raise c (or m₀) so T ≪ m₀c² — free, one config line, already validated by finalA."
- **Numbers:** R8's benign-ness table is **d=1**. At Exp-C's **d=784**, Var_MJ/(M_eff·T) → (d+1)·T/(m₀c²):
  **785** at c=1 (not 2.70) and **31.4** at c=5 (not ~1.06); KL at c=5 is still **10 578 nats**. So c=5 does not
  restore Gibbs. Worse, it moves the chain off the velocity-saturation plateau: relative velocity error
  **0.66 % → 13.31 %** (T=1) / 1.29 % → **22.17 %** (T=0.5). Empirically, under the shipped `legacy` noise
  `c=5` collapses the dreamer onto the deterministic mode: χ²(c5_legacy, det_c1)=**1.27, p=0.74**, 5.7 % flips;
  entropy 2.038 → **0.365** bits. **On models trained natively at c=5** (no off-distribution confound) the
  imbalance still *grows*: f(3589) 0.484→**0.771** (legacy, χ²=128.2, p=6.7e-24) and 0.599→**0.786**
  (fdt, χ²=113.2, p=8.4e-21); entropy 2.960→1.821 / 3.042→1.993 bits.
- **Mechanism:** σ_legacy=√(2γTε) and σ_fdt=√(M_eff Tγ(2−γ)) are both **c-independent**, and ∇_qH is bit-identical
  in c. **c is a drift knob, not a sampler knob**: it cannot move the coded (Gaussian) momentum law toward MJ, it
  only rescales q̇ (‖∇_pT‖ 1.260 → 5.175) and moves the Gibbs *target*.
- **Scope:** kills fix (iii) for relativistic mode. Fixes (i) exact MJ refresh and (ii) Metropolis remain the only
  correct routes — and (i) was run: it makes generation *worse* (N10). `finalA`'s good behaviour must be explained
  by something other than "benign T/(m₀c²)".
- **Disposition:** F5/appendix; corrigendum to deep-dive §7bis R8's fix ranking and its d=1 table.
```

---

## 8. Recommended next experiments

1. **(cheap, high value) Depth-controlled dreaming.** If the dominant mode is a threshold function of final H, then *directly controlling quench depth* (anneal length / `dream_steps` / γ) should let one **dial the mode distribution** — and a depth chosen at the entropy-maximising middle (H ≈ −250) should give the most diverse samples. This converts N10's negative into a **positive, usable generative knob** and is ~1 h of compute on existing checkpoints.
2. **Metropolis-adjusted relativistic Langevin** (R8's fix (ii)) as the one genuinely-Gibbs equilibrium sampler; measure the *equilibrium* digit law (not the quench). This is the only way to ask "what does `exp(−H/T)` of the learned `V_θ` actually look like?" — the answer is currently unknown, and would tell us whether the landscape's basin masses match the imbalance.
3. **Sweep the confound-splitter properly:** `(c, m₀)` on the hyperbola `T/(m₀c²)=const`, 4–5 points, 5 seeds. Predicts (from §3.6) that the digit law tracks `v = c·s` monotonically, with `T/(m₀c²)` inert. A clean, publishable dissociation figure.
4. **Repeat the confound-split on `mnist`/`mnistFF`** to lift the single-checkpoint scope on N10 v2.
5. **Fix the `sleep_friction` plumbing** (§6), then re-check whether any archived Exp-C run intended γ_sleep=0.

## Proposed handover updates (for the Hub)

**§1.6 / Exp III.** Add: the MNIST mode imbalance is governed by **quench depth in `V_θ`**, not by the sampler. Dominant mode separates perfectly by mean final H (threshold ≈ −210; MW p=0.0097). The paper's deterministic-collapse observation and the stochastic imbalance are **the same phenomenon at two depths**.

**§5 (provenance).** All four `mnist*` checkpoints trained at `c=1.0`; `finalA` at `c=5.0`. Any `c=5` dreaming on `mnist*` is **off-distribution** (`max|q|` 4.50→7.68 vs data cube [−1,1]). New checkpoints: `.claude/scratch/relativistic-gibbs-expc/e5_models/exp_c_chlu_c{1,5}_ep500.pkl` (seed 42, 500 ep, matched init).

**§7 (known issues) — new items.**
- **7.23 [OPEN, low risk] `training.sleep_friction` is a silent no-op for `train_generative`** (`train_generative.py:102-103` reads `experiment_c.friction`). Exp-C training is therefore stochastic (γ=0.3) and `langevin_noise` is live *during training*. Fix or delete the field. Also amend **N19**'s scope to "dynamics path only".
- **7.24 [OPEN, HIGH — arXiv-bound] `df5e44d` ships R8's d=1 table and the refuted "raise c" mitigation in tracked code** (`chlu/core/integrators.py`, `chlu/core/chlu_unit.py::thermal_causal_ratio`, `chlu/training/train_generative.py`, plus a user-facing `RelativisticGibbsWarning`). Numbers off by 290× (variance) / 8.4e6× (KL) at Exp-C's d=784; the recommended remedy degrades Exp III. **Docstring-only fix specced in §6b of this report.** The CM-17 *mechanism* is correct and confirmed — only the corollary is wrong (same failure mode as §7.19's F5 clause).

**§7.18 / R8 corrigendum.** The deep-dive's R8 table is **d=1** and is quoted against a **d=784** experiment. Correct law: `Var_MJ/(M_eff·T) → (d+1)·T/(m₀c²)` (ultra-relativistic). At Exp-C: **785×** (c=1), **31.4×** (c=5). `KL(Gibbs‖coded)` = **3.24e6 / 3.05e5 nats** (legacy/fdt) at the anneal start, not 0.384. **R8's fix (iii) "raise c" is refuted** (see N10b), including on natively-`c=5`-trained models. R8's *qualitative* claims (linear OU ⇒ Gaussian; no σ works; defect is a function of `T/(m₀c²)` alone) are all **confirmed exactly** on the real path — the last one to all printed digits (`c5_fdt` ≡ `m25_fdt` in KL, ratio, and velocity error). R8's own honesty note (X5, the under-converged MJ-refresh arm) is now **superseded by a converged one**: `mj_refresh` here is a valid annealed-quench arm, and it makes generation *worse*.

**§8 (open directions).** "Generative: characterize the mode imbalance — energy-landscape asymmetry or sampler bias?" → **RESOLVED: landscape (quench depth).** Replace with the depth-as-generative-knob experiment (§8.1 above) and the Metropolis-adjusted equilibrium sampler (§8.2).

**New negatives:** N10 replacement text + **N10b** (both in §7 above), tier A.

**Positive result worth a paper line:** *the relativistic kinetic governor buys sampler-robustness* — velocity saturation maps a 90.3× error in the momentum distribution onto a 0.66 % error in q̇. The regime where the Gibbs defect is largest (`T ≳ m₀c²`) is precisely the regime where it is dynamically invisible. This is a memory/generation-side payoff of the causal cap, adjacent to R7's causal-retention floor, and it is measured, not conjectured.
