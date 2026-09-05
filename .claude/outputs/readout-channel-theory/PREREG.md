# PREREG — readout-channel-theory (physics-theorist, w26)

Written **before** any of the measurement scripts (`fit_reach.py`, `dyn_toy.py`,
`q2_designed.py`, `anneal.py`) existed or ran. What preceded it: (i) a JAX warm-up /
timing script (`warmup.py`, times only — d=2 K=8 write 6.2 s, read 2.0 s, import 8.4 s),
(ii) `geom.py` (deterministic site geometry from `designed_sites`, config not measurement),
(iii) `kappa_static.py` (a *derivation* helper — it evaluates a closed form, it measures
nothing), (iv) re-reading `r2-geometry-revival`'s saved per-item JSON
(`m_d4_K16_w{0.15,0.30}.json`), which is **prior measured data, not mine**; any claim
scored against it is flagged POST-HOC-ON-EXISTING-DATA and never counted as a
pre-registered confirmation.

Base: local `main` @ `ff85573`. JAX 0.9.0, main venv. No tracked code touched.

---

## 0. The model being registered (derivation, stated so it can be falsified)

Single composite well of depth `D`, width `s`, at `z_i = (c_i, a_i)`; confinement
`α|q|²` (`α = learned_confine = 0.05`); launch on the payload-zero manifold at
`(c_i + ξ, 0)`.

**(R) address-hold / reach.** With the payload channel still at 0, the address-plane
landscape is a Gaussian well of *gated* depth `D_eff = D·exp(−a_i²/2s²)` at `c_i` plus the
confinement, whose inward pull at the site is `2α|c_i|`. A local minimum near `c_i` exists
iff the well's maximum restoring force (attained at offset `δ = s`) beats it:

```
D·exp(−a_i²/2s²) / (s·√e)  ≥  2α|c_i|
⟺  |a_i| ≤ Ψ(s) ≡ s·sqrt( 2·ln( D / (2α|c_i|·s) ) − 1 )                      (R)
```

**(S) payload stagnation.** Roots of `h(v) = v(1+β e^{−v²/2}) = a/s`,
`β = D/(2αs²)`; a spurious minimum on the payload ray exists iff
`a/s > κ_stat(β) = h(v_b)`, `v_b` = larger root of `h'=0`. Exists only for
`β > e^{1.5}/2 = 2.2408`.

**(T) payload-descent time budget.** `t_pay ≈ (η s⁴/(D a²))·e^{a²/2s²} < T_phase`.

**(M) merge / discrimination.** Two equal wells at separation `sep` have distinct minima
iff `sep > 2s`; the watershed sits at `sep/2` shifted by `(s²/sep)·ln(D_i/D_j)`; query
jitter `σ_q` must fit inside it.

`κ` in w25's `s ≳ |a|max/κ` is therefore **`κ_R = Ψ(s)/s = sqrt(2 ln(D/(2α|c|s)) − 1)`**,
a **square root of a logarithm** — which is why it is stubbornly O(3).
Evaluated (`kappa_static.py`, derivation): `κ_stat` = 4.08 (β=100), 4.37 (β=295),
4.47 (β=444); raising `κ_stat` from 4 to 5 needs `β` ×55 (at s=0.3: `D` 0.69 → 37.9);
to 6, `β` ×9200.

---

## 1. Q1 predictions

| # | prediction | band | falsifier |
|---|---|---|---|
| **P1.1** | **`D_fit`** (3-param radial fit `V0 + D(1−e^{−r²/2s²})` of the trained learned `V` at each site, d=4 K=16 8192 atoms, seed 0) at `atom_init_width=0.30` | **`D_fit ∈ [1.5, 6.0]`**, median over 16 sites | `D_fit < 1.0` or `> 12` ⇒ (R)'s single-Gaussian idealisation cannot be calibrated; report as such |
| **P1.2** | (R) is the **binding** constraint, (S) and (T) are slack, at BOTH widths: `Ψ(s) < κ_stat·s` and `Ψ(s) < a_T` (the (T) threshold) | strict inequality with ≥20 % margin | either (S) or (T) tighter than (R) at either width |
| **P1.3** ⭐ | Using the measured `D_fit`, `s_fit`: **cell strict ≈ Ψ(s)/|a|max** (fraction of a uniform codebook inside the reach radius) | width 0.30: predict **0.88 ± 0.10** (measured elsewhere 0.937–0.947); width 0.15: predict **0.56 ± 0.10** (measured elsewhere 0.586–0.594). **Δ(0.30→0.15) predicted 0.31 ± 0.10 vs 0.34 measured** | predicted strict off by > 0.15 at either width, or Δ off by > 0.15 |
| **P1.4** ⭐ | **Exact (R) on the trained shipped `V`** (no idealisation): local gradient descent on the payload-zero slice from `(c_i, 0)` either stays within `sep/2` of `c_i` (HOLD) or escapes (LOSE). HOLD/LOSE classifies per-item `strict > 0.5` | **≥ 14/16 items correct at each width** (28/32 overall) | ≤ 11/16 at either width ⇒ (R) is not the mechanism |
| **P1.5** | **Exact (S) on the trained `V`**: an interior stationary point of `y ↦ V(c_i, y)` on `(0, a_i)`. Predict it is **rarer** than (R)-escape: (S) fires on **≤ 4/16 items at width 0.15 and ≤ 1/16 at width 0.30** | as stated | (S) fires on ≥ 8/16 at width 0.30 ⇒ (S) binds, not (R) |
| **P1.6** | **Toy dynamical threshold** (hand-built single Gaussian + confinement, dim 5, **shipped** `_two_phase` schedule γ .05×400 → .02×800, dt .05): the measured capture threshold `a*/s` at `D=1, α=0.05, |c|=0.9, s=0.30` | **`a*/s ∈ [2.2, 3.2]`** (i.e. `κ_R = 2.45`, not `κ_stat = 4.11`) | `a*/s > 3.8` (⇒ κ_stat governs) or `< 1.6` (⇒ neither) |
| **P1.7** | Toy: `a*` scales as `Ψ` in **all four** parameters. Predicted ratios: `a*(α=0.025)/a*(α=0.05) = 1.13 ± 0.06`; `a*(D=4)/a*(D=1) = 1.24 ± 0.08`; `a*(|c|=0.3)/a*(|c|=0.9) = 1.20 ± 0.08`; `a*(s=0.45)/a*(s=0.30) = 1.66 ± 0.15` | as stated | any ratio outside band, or wrong sign |
| **P1.8** | Toy: the failure is an **address** failure (basin miss ⇒ `basin ≈ strict`), and failed queries end nearer the ball centre than `c_i` | `|x_end| < |c_i|` for ≥ 80 % of failures | failures keep the address and only miss the value |

**Q1 second half (the sharp-but-not-causal correlate).** Registered *explanation*, scored
against the r2 §1 table (**POST-HOC-ON-EXISTING-DATA**, declared): the criterion is
`sep ≥ 2|a|max/κ_R(s)`, so with `κ_R` varying only ~7 % over the whole cell set the
boundary is `sep* ≈ 0.82·|a|max` — a threshold on **`sep` alone**. Registered claims:
(i) `sep` alone separates all **11/11** Stage-0 cells (boundary in (0.795, 0.849)),
including d=2/3 where `sep/w` fails; (ii) with **one** fitted `D`, `2|a|max/κ_R(s_cell)`
separates 11/11; (iii) w25's conjecture ("both numerator and denominator drift with d") is
**replaced**: the numerator carries the signal, the denominator is the write's chosen `s`,
and `sep/w` fails at d ≤ 3 because `w` drops there while `sep*` does not.

---

## 2. Q2 predictions (designed `AtomStorePotential`, shipped store + shipped read)

Store: `dim=3, capacity=8, α=0.02, s=s_pay=0.35, κ=1.0`, `designed_payloads(8, seed=0)`,
`d_safe=4.4s`, shipped `make_queries_at`/`two_phase`/`evaluate_items`, `payload_tol=0.1`.

**Mechanism (derivation).** At launch the site carries a payload hill `0.5κ a_i²`
(stiffness `κa_i²/s²`) against a well of depth `A_i` (stiffness `A_i/s²`). Net address
curvature `2α + (A − κa²)/s²`; the payload spring relaxes with `τ_y ≈ η/κ`
(η_addr = −ln(1−γ)/dt = 1.0259) while the address is ejected at rate
`λ = ½(−η + sqrt(η² + 4[(κa² − A)/s² − 2α]))`. Retention fails when the address escapes to
`w = d_safe/2` from `δ₀ ≈ σ_q` before the payload settles:
`ln(w/δ₀)/λ < τ_y`.

| # | prediction | band | falsifier |
|---|---|---|---|
| **P2.1** | Shipped-store retention vs `A` reproduces `mia-decay` §5 to within sampling error at `a=±1, ±0.714, ±0.429, ±0.143` | `A=0.06`: 0.50/0.66 (a=∓1), ≥0.96 (0.714), ≥0.99 (≤0.429), each ±0.15 | mismatch > 0.25 ⇒ my harness ≠ theirs, stop |
| **P2.2** ⭐ | The **race criterion** `ln(w/δ₀)/λ(a,A) = τ_y` predicts `A_crit(a)`: predicted `A_crit` = **0.55 ± 0.20** (a=1), **0.10 ± 0.10** (a=0.714), **< 0.05** (a≤0.429) — i.e. retention-halving before the floor only for `|a| = 1` | as stated; ordering must be strictly monotone in `a²` | `A_crit` ordering not monotone in `a²`, or `A_crit(0.429) > 0.2` |
| **P2.3** ⭐ | **Option (a) (`payloads*amps`) is fatal**: read returns `A·a_i`, so the value criterion dies at `A < 1 − tol/|a_i|`. Predicted retention **≈0 for `A ≤ 0.85` at `a=±1`** and **≤0.3 at `A ≤ 0.25` for `a=±0.143`**; the payload dependence **inverts** (small-`\|a\|` items now outlive large ones by a factor ≈ 3 in `τ`) | retention at `A=0.5`: **≤ 0.15** for all `\|a\| ≥ 0.3` | retention at `A=0.5` ≥ 0.6 for `a=1` ⇒ my reading of the option is wrong |
| **P2.4** ⭐ | **Option (b) breaks the anti-decoration guard, measurably**: the *trivial substitute* "return `S(q_addr(0))` at the launch point, run no dynamics at all" already passes the value criterion for | **≥ 0.75** of queries (mean over the 8 items, `A=1`) | < 0.45 ⇒ the guard is not broken by (b) at this jitter, and (b) must be judged on other grounds |
| **P2.5** ⭐ | **Option (c) is infeasible for most of the codebook**: `∃ leak` hitting a target half-life requires `A*` with `R(A*)=0.5` and `A* > amp_floor = 0.05`. Predict solvable for **2 of 8** codewords (`a = ±1`), unsolvable for **6 of 8** | 2/8 (±1 only); at most 3/8 | ≥ 5/8 solvable ⇒ (c) is broadly feasible and my ruling weakens |
| **P2.6** | Option (c) cannot fix the **shape**: the `τ`-width from `R=0.9` to `R=0.5` differs by ≥ 5× across the codebook (predicted `≈0.3` for `a=1`, **undefined/∞** for `\|a\| ≤ 0.43` because `R` never reaches 0.5) | as stated | widths within 2× across the codebook |
| **P2.7** | **Option (d) — gated spring** `0.5κ·G(x)(y − ā(x))²`, `G = Σ m_iA_i e_i + g₀`, `ā` = amplitude-weighted normalised payload (**NOT shipped code — a theorist's toy**): hill/well stiffness ratio becomes `κa²`, A-independent. Predict retention curves for different `a_i` **collapse** under `A → A(1 − κa_i²)`, and the value is exact at every `A` | collapse residual ≤ 0.10 in retention over `A ∈ [0.05, 1]`; `\|read − a_i\| < 0.02` at `A = 0.06` for all items | curves do not collapse, or the value degrades ⇒ (d) is not the fix either |

---

## 3. Q3 predictions (for `r2-excursion-reach` to score)

Definitions: `r` = payload-space Euclidean radius of the code (`r = |a|max` at m=1);
`σ_a` = **absolute** read-out noise on a payload coordinate; `μ(d)` = empirical packing
exponent of `designed_sites` (`sep ∝ K^{−μ}`; measured from config geometry:
μ = 0.590/0.407/0.350/0.283/0.243/0.185 at d = 2/3/4/5/6/8, i.e. `d_eff = 1/μ`);
`Ψ(s)` as in §0. Feasibility of a cell: `r ≤ Ψ(s)` and `2s + c_jσ_q ≤ sep`.

| # | prediction | band | falsifier |
|---|---|---|---|
| **P3.1** ⭐ | **Arm (a) is NOT a cancelling free lunch under absolute read noise.** `σ_a` is a property of the read-out, not of the code, so per-axis levels `n = 1 + r_axis/(zσ_a)`. Halving `r` at m=1 costs **one** dimension of value resolution (`K_value` ×½) and buys `2^{1/μ}` of address packing (`K_addr` ×7.3 at d=4) ⇒ **net gain `2^{d_eff−1}` = 3.6× per halving at d=4** | net item-capacity gain per halving of `r`: **3.6× (band 2×–6×) at d=4** — measurable as `K_wall(r/2)/K_wall(r)` with `payload_tol` **held absolutely fixed** | `K_wall` does not move when `r` halves at fixed absolute tolerance ⇒ the lunch does cancel |
| **P3.2** ⭐ | **m-channel excursion law**: at fixed `K` and fixed absolute per-axis precision `Δ = 2z σ_a`, `r(m) = z σ_a (K^{1/m} − 1)√m`, monotone decreasing for `m ≤ log₂K`, with **`m* = ⌈log₂K⌉`** (binary code) and `r(m*) = zσ_a√(log₂K)`. Beyond `m*` extra channels **increase** `r` as `√m` | at K=16, matched to the shipped codebook (`Δ = 2/15`): `r` = **1.000 / 0.283 / 0.133** at m = 1 / 2 / 4 (a 3.5× / 7.5× reduction). `m>4` at K=16 must NOT help | `r` measured/derived from the implemented code differs > 20 % from the formula, or m=8 at K=16 helps |
| **P3.3** ⭐ | **Wall movement, arm (a).** `K_wall(d, r)` solves `sep(d,K) = max(2r/κ_R, sep_jit)`. Calibrating `κ_R` from the r=1 boundary (`sep* ≈ 0.82`) ⇒ `κ_R ≈ 2.44`; `sep_jit ≤ 0.549` (because d=4 K=64 passes at r=0.5). Predicted at d=4: **`K_wall(1.0) ≈ 21` (obs: 16 PASS, 32 FAIL ✓)**, **`K_wall(0.5) ≈ 78` (band 60–150; obs K=64 → 0.9922 ✓)**, **`K_wall(0.25) ≈ 78–110` — i.e. the arm SATURATES below r ≈ 0.4** because the query-jitter floor takes over | at d=4, m=2 (r=0.283): **strict ≥ 0.9 at K=64**, and **K=256 FAILS**; the r=0.25 and r=0.125 walls within 1 K-rung of each other | K=256 passes at d=4 (wall ≫ prediction), or the r=0.25 wall is ≥ 2 rungs above the r=0.5 wall (no saturation) |
| **P3.4** ⭐ | **Past `r ≈ 0.4` at d=4 the only remaining lever is `σ_q`** — and lowering the query jitter is **laundering** (it makes the task easier). Flagged in advance so no one reports it as capacity | `K_wall` at r=0.25 and r=0.125 differ by ≤ 1 rung unless `σ_q` is also reduced | walls keep moving at fixed `σ_q` ⇒ my jitter floor is wrong |
| **P3.5** ⭐ | **Arm (b) (anneal) has a hard ceiling `r ≤ Ψ(sep/2)`** — the anneal's whole benefit is to let the *read* run at the merge-limited width `sep/2` while the write keeps its own `s`. Gain factor in excursion `Ψ(sep/2)/Ψ(s_write)`. At d=4 with `D=3.2`: `Ψ(0.4515) = 1.24` vs `Ψ(0.302) = 0.88` ⇒ **1.42×** | **the annealed read moves `K_wall(d=4, r=1)` from 16 to 32 and NO further** (K=64 must still FAIL: `Ψ(sep/2=0.2745) = 0.80 < 1`). Annealed strict at d=4 K=32 r=1: **0.90–1.00** (from 0.824) | K=64 passes under the anneal at r=1 (ceiling wrong), or K=32 does not improve (anneal does nothing) |
| **P3.6** | **Anneal schedule.** Widen during the **address** phase only, return to native width before the payload settles. Terminal-width constraint from payload cross-talk: bias `≈ a_rms√N_nb·e^{−sep²/2s²} < tol` ⇒ at d=4 K=16 **`s_read ≤ 0.38`**; native 0.30 satisfies it. Adiabaticity: `|ds/dt| ≤ (ω²/η)·s`, `ω² = D/s²` ⇒ at D=3.2, s=0.3, η=1.026 ⇒ `|ds/dt| ≤ 10` — the 20-time-unit address phase is **~200× longer than adiabaticity requires**, so the schedule shape is irrelevant; only the endpoints matter | annealing *through* the read phase must LOSE value accuracy: payload abs-err at `s_read = 0.5` **≥ 0.15** (vs ~1e−4) | annealing through the read phase is harmless ⇒ cross-talk model wrong |
| **P3.7** | Combined `(a)+(b)`: the two are **not** additive — both act on the same inequality `r ≤ Ψ(s_read)`. Predicted joint wall = the arm-(a) wall (jitter-floor-limited), i.e. **the anneal adds ≤ 1 rung on top of arm (a)** | ≤ 1 rung | anneal adds ≥ 2 rungs on top of arm (a) |
| **P3.8** | Read noise `σ_a` on the payload channel also enters **(R)** through nothing at all (it is a *measurement* noise, added after settling) but enters the **value criterion** as `P(|ε| < tol)`. Predicted: turning on `σ_a = 0.05` with `tol = 0.1` costs **≤ 5 %** strict at every `m` (`2Φ(2)−1 = 0.954`), and **the cost is m-independent per axis but grows as `1 − (2Φ(tol/σ_a)−1)^m` in total**: 4.6 % (m=1), 9.0 % (m=2), 17 % (m=4) | as stated ±3 % | the m-channel arm loses more than `1−(2Φ(z))^m` ⇒ noise is not per-axis independent |

**Registered impossibility check (an acceptance if it fires).** If P3.1's net gain measures
≤ 1.2× at fixed absolute tolerance, arm (a) is a free lunch that cancels and the wave should
stop spending compute on it. I am predicting it does **not** cancel (3.6× at d=4), and that
the *reason* it does not is `d_eff > 1`.

---

## 4. What would make me withdraw the whole (R) account

- `D_fit` outside [1, 12] (P1.1), **and** the exact test P1.4 at ≤ 11/16 — then the
  composite well is not Gaussian enough for any single-well theory and the account is a
  cartoon.
- The toy threshold P1.6 landing at `κ_stat` (≥ 3.8) — then the confinement-driven
  *address* loss is not the mechanism and the payload-ray stagnation is.
- P1.3 off by > 0.15 at both widths — then the "strict = reach fraction" arithmetic that
  makes this quantitative is wrong even if the sign is right.

---

## 5. AMENDMENT A1 (declared before the test it describes was written or run)

Written after `dyn_toy.py`, `q2_designed.py` and `cells.py` returned, and **before**
`surrogate.py` existed. Two changes, both declared:

**A1.1 — the criterion is upgraded (and the upgrade is POST-HOC).** `dyn_toy` showed the
registered closed form `Ψ` is *systematically* 1.36× conservative (ratio
`a*/Ψ = 1.357 ± 0.064` over 7 variants) while scaling correctly in all four parameters.
The exact fixed-point structure of the same model gives the **saddle criterion (U)**:
all stationary points lie on the origin→target ray at distance `R` from the target with
`(D R/s²)e^{−R²/2s²} = 2α(L−R)`, `L = |z_i| = sqrt(|c_i|² + a_i²)`; with three roots
`R1<R2<R3`, the launch point (at distance `|a_i|` from the target) is inside the well's
basin iff **`|a_i| < R2`**. (U) matches the 7 toy thresholds at **1.013 ± 0.025**.
(U) was derived *after* seeing the offset, so its agreement with the toy is a **fit**, not
a pre-registered confirmation. Its **out-of-sample test** is the trained shipped `V`
(P1.4/P1.5, unrun at the time of writing) and A1.2 below.

**A1.2 — new registered test: the designed-Gaussian SURROGATE.** Build
`V = α|q|² − Σ_i D·exp(−|q−z_i|²/2s²)` on the **real** `designed_sites`/`designed_payloads`
geometry of each of the 11 Stage-0 cells, with `s` = the **measured trained width** of that
cell and a single global `D`; read it with the **shipped** two-phase read and score with the
shipped criterion. No training anywhere.
- **Registered prediction A1.2a:** with `D = 1`, the surrogate reproduces r2's 11-cell
  PASS/FAIL pattern at **≥ 9/11** (band 9–11), i.e. the ceiling is a property of
  *geometry + read*, not of the learned write.
- **Registered prediction A1.2b:** the surrogate's `strict` falls with K at fixed `d` and
  fixed `s` (the three high-K FAIL cells: d4K32, d5K64, d6K64 predicted **< 0.9**), which is
  the multi-well interference the single-well criterion (U) does **not** contain — (U)
  scores all three as PASS (measured: `cells.py` gives 8/11 with D=1, misses exactly those
  three).
- **Falsifier:** ≤ 7/11 ⇒ the learned write's structure (not the geometry) sets the wall,
  and the whole single-well account is a poor model of the shipped `V`.
