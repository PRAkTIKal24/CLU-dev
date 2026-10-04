# doctrine-repairs — physics-theorist report

> ## ⚠ DATED ERRATUM BANNER (Hub, 2026-08-01, C2W4 second review — appended under the ratified C-3 precedent; the body below is NOT edited)
> **§2.3's "2 recovered false negatives" is a RESOLUTION-BAND statement, not a shipped-band prediction — never quote the count without its band.** `harness-debt` (C2W4) landed the `+eps_acq` half and re-scored the recorded readings offline: at the **shipped round-off band** (`eps_acq_rel = 1e-9`, mirroring the loss half) the two recoveries **do not fire** and the post-repair count is **27 → 27, zero cells changed** (✅ **CONFIRMED, Head ruling 2026-08-01 — band left alone**). The two cells §2.3 names (`overload/base@s0`, `overload/reach_free@s0`) **do** recover exactly as written at the **resolution floor** `1/(n_probed·W) ≈ 4.2e-2` — **~6 orders above** the shipped band — where the count is 29. **§2.3's mechanism and its two named cells are right; reading it as "the shipped monitor will add 2 trips" is wrong.** Registry: **N203** (never-quote: the count without its band); the resolution-band reading stays filed as the documented alternative.

Task + acceptance: adjudicate charter §A2.5 (P1 / merge certificate, pre-registered), dispose of the nine owed reconciliations, diff the 13-row table, deliver the soft-certificate spec. Status: **done.**
⭐ **BLOCKING ITEMS DELIVERED (this wave, §1 and §2 below): the monitor-#6 `eps` and the monitor-#3 validity-leg replacement are both final and landable now.**
⭐ **HEADLINE: §A2.5 is CONFIRMED under the shipped hard gate (7/7 on the published grid, 0 witnesses in a 72-config extension) and REFUTED the moment the admission radius is decoupled from the certificate (11/12 soft witnesses).** The certificate and monitor #3's fire-rate band are **the same object in the shipped code** (`d_safe := 2s_max + κ′σ_q`), so violating the certificate drives `f → 1.000` and #3 trips. Decoupling (`d_safe = 0.6·sep`) opens a non-separable feasible region with `ρ_ex` up to **6.3×** the C2W1 witness at a `λ_min` price of **2.2–6.0×**. That region is C2W3's target and it did not previously exist.

> ## ⚠ SITE-BY-SITE RECONCILIATION LIST — self-contained, no curator this wave (protocol §5 corollary)
> | # | site (file:line where known) | change | owner |
> |---|---|---|---|
> | S1 | `chlu/core/monitors.py:670` `ObjectiveDivergenceMonitor.observe` | replace predicate with the two-sided dead-band of **§2.3** (`eps_loss`, `eps_acq` formulas given; report both in `detail`) | **`phi-particle-head`, this wave** |
> | S2 | `chlu/core/monitors.py:527-541` `VacuousGateMonitor` leg (ii) | **retire the correlation**; land the C3-calibration leg of **§1.3**; return INAPPLICABLE when <3 qualifying pairs | **`phi-particle-head` this wave (predicate) / C2W3 (plumbing)** |
> | S3 | `chlu/core/clu_system.py:906` `lam = max(self._lambda_min_at(...), 1e-9)` | silent clamp → **declared `λ_floor`**; return INAPPLICABLE, never a floored ratio (a floored λ makes the C3 ratio collapse to ≈0 and reads as "certificate perfect") | C2W3 engineer |
> | S4 | `chlu/core/clu_system.py:510` + `chlu/core/clu_controller.py:265` | `gate_margin` logs `d_min_proposed` = the **pre-relocation** distance; on a relocated write it describes a site that was never written. Log `d_min_written` and pair the drift with **that** | C2W3 engineer |
> | S5 | `chlu/core/monitors.py:~470` `SettleArgminMonitor` | `U` uses `sep/2` as if certified. Use §3-R1's corrected radius; **`r_i := 0` and monitor INAPPLICABLE wherever `λ_min,i ≤ 0`** (this alone dissolves the gym's "D1 violated 7.44×") | C2W3 engineer |
> | S6 | `.claude/outputs/controller-doctrine.md` §2 row 1 (band) | **my own error:** `γ∈[0.05,0.5]@N=400` is scored at `tol=1e-3`, `γ∈[0.02,0.9]@N=1500` at `tol≤1e-4`; the row claims `1e-6`. Replace both with the **convergence budget** `C = N ln(1/ρ)` (§3-R2d) | curator (this file supersedes) |
> | S7 | `.claude/outputs/controller-doctrine.md` §2b row 8 ("`sep/2` over-claims by up to 4.8 %") | restate with the **validity domain** `s/sep ∈ [0.15, 0.30]` and the corrected proxy; the 4.8 % figure is only that store's number | curator (this file supersedes) |
> | S8 | `handover_context.md` §7 (`full-clu-harness` R1/R2/R3, `memory-gym-v0` R1/R2/R3/R5) | replace with §3's dispositions verbatim; **retire "Prop D1 is violated"** — it is an uncertified-store artefact | Hub/curator |
> | S9 | `.claude/outputs/trainability-spike-theory.md` §Q4.2, §7 requests 3/7/8 and every `k*=269` quoting site | add **"of `∂q_N/∂θ`, for parameters whose fixed-point sensitivity dominates their transient"** (§3-R-1) and the **head/tail direction** (§3-R-2) | theorist(me)+curator |
> | S10 | `handover_context.md` §7 monitor-#10 wording | "the knob is inert" → **"no shipped ψ consumes the buffer"** (§3-R-3) | Hub/curator |

---

## ⭐ DIAL DECLARATION (echoed, protocol §7)
**Dial:** none — doctrine/theory. No performance claim, no benchmark, no dividend. **Laundering control:** n/a, except that **mode #2 is the laundering control promoted to a runtime monitor** and remains the table's most important row. **Falsifies:** §A2.5 refuted / a mode with no runtime-computable invariant / two productive bands provably disjoint after the soft relaxation. **Does NOT falsify:** a narrow band; a band that is only measurable; a monitor that costs a diagnostic pass; a repair that makes a monitor trip **more** often (§2.3 does exactly that, twice, on purpose).

## 0. Flag provenance (every number in this report)

| item | value |
|---|---|
| repo | read-only, **`main @ 233fd9e`**, clean tree, no worktree, **zero tracked-code edits** |
| code | pure numpy 2.4.1, main venv `/Users/user/Desktop/CHLU/.venv` (python 3.11.13); **no repo module imported anywhere**; integrator = the shipped damped velocity-Verlet copied line-for-line (`common.py`, from `controller-doctrine`) |
| re-scored artifacts (read-only) | `.claude/scratch/controller-doctrine/{s5b,s2b}_results.json` · `.claude/outputs/memory-gym-v0/exp_memory_gym_metrics.json` (28 cells) |
| kinetic mode | **Newtonian, M = I** everywhere. `T = 0`, `p₀ = 0`, no Langevin, no wake–sleep |
| ε (dt) | 0.05 everywhere |
| γ | A2.5 grid **0.20** (as published) · R1 **0.05** · R3 toy **0.20** · R-1/R-2 toy **0.05** |
| steps | A2.5 700 · R1 1500 · R3 900 · R-1/R-2 400 |
| potentials | `V = α‖q‖² − Σ A_i exp(−½(q−c_i)ᵀS_i⁻¹(q−c_i))`, `α = 0.05` throughout |
| seeds | A2.5 `rng(1000+0)` (the published single-seed convention, reproduced bit-for-bit) · R3 seeds 0–11 · R1 `rng(11)` · R-1/R-2 deterministic · R5 `rng(7)`. **Every number here is a THEORY check; none is a paper number.** |
| scripts | `.claude/scratch/doctrine-repairs/`: `common.py · reach.py · r2_eps.py · r3_validity.py · r3b_validity_domain.py · r3c_guard.py · a25_merge.py · a25b_soft.py · r1_inradius.py · rm1_truncation.py · r2gamma.py · r5_probe.py` + `{r2,r3,r3b,r3c,a25,a25b,r1,rm1,r2gamma,r5}_results.json` |
| PREREG | `.claude/outputs/doctrine-repairs/PREREG.md`, written **before any script existed** |
| wall | ≈45 min total (the 72-config A2.5 extension is 155 s; the batched inradius bisection 6 min) |

---

# 1. ⛔ BLOCKING #1 — R3(gym): monitor #3's validity leg

**Verdict: the leg is RETIRED as a correlation and REPLACED by a first-order calibration test. The shipped statistic is not "wrong on a learned `V_θ`" — it was measured on sites that are not minima, and it is only rank-faithful in a domain the gym is outside of.** Both halves are measured.

### 1.1 Why `corr(gate_margin, post_write_drift)` is sign-unstable — four named causes

The harm the gate must bound is the first-order crowding drift of an incumbent minimum,
```
Δq_i ≈ ‖∇δV_j(q_i*)‖ / λ_min,i ,     ‖∇δV_j(q_i*)‖ = (A_j/s_j²)·d_ij·exp(−d_ij²/2s_j²)
```
and `gate_margin = d_min_proposed` is a pure distance. Then:
1. **`λ_min ≤ 0` at most gym cells.** Re-scoring the published artifact: **18 of 28 cells (64.3 %) have `N3_lambda_min ≤ 0`.** Where the recorded site is not a minimum, "drift of the fixed point" has no first-order theory and the correlation is measuring nothing. Grouped: `λ_min > 1` (n=3, the converged shipped-anchor cells) → validity **+0.470 mean, min +0.330, all three positive**; `0 < λ ≤ 1` (n=7) → mean **−0.155**; `λ ≤ 0` (n=18) → mean **−0.013**, range **[−0.994, +0.559]**, sign-unstable. `corr(λ_min, validity) = +0.245` over all cells. ⇒ **every cell in which the write actually converged passes the 0.30 bar; the sign instability lives entirely at `λ_min ≤ 0`.**
2. **The `(A_j, s_j)` heterogeneity a learned write produces.** At a *fixed* distance `d = 0.9`, sweeping `A ∈ {0.5,1.0,1.6} × s ∈ {0.20,0.30,0.42}` moves `‖∇δV‖` by **1823×** (`r3_results.json:AS_confound`). None of it enters the margin. (N120 measured the shipped write does exactly this: `corr(s_fit,|a_i|) = +0.821`.)
3. **Non-monotonicity.** `d·exp(−d²/2s²)` peaks at **`d = s`** (measured argmax 0.298 for s = 0.30) — closer is not monotonically worse.
4. **A pairing bug (S4).** `gate_margin` is logged as `d_min_proposed`, the distance **before** relocation; on a `relocate` decision the drift is caused by `d_min_written`, a different site.

**The domain where the shipped leg *is* valid** (`r3b_results.json`, 10 seeds per width, K=9): the margin's rank power decays monotonically with the spacing-to-width ratio.

| median `d/s` | 10.40 | 7.11 | 5.20 | 3.91 | 3.08 | 2.32 |
|---|---|---|---|---|---|---|
| mean `corr(margin, drift)` | +0.494 | +0.595 | +0.564 | +0.521 | +0.341 | **+0.289** |
| seeds below the 0.30 bar | 0/10 | 0/10 | 0/10 | 1/10 | 5/10 | **5/10** |
| seeds with the **sign flipped** | 0 | 0 | 0 | 0 | 1 | **2** |

⇒ **validity domain: `d/s ≳ 4`.** The gym runs at `atom_init_width = 0.30` with `d_safe_override = 0.58` ⇒ `d/s ≈ 1.9–2` at the closest pairs: outside the domain by 2×. *(Honest note: my first toy — `r3_validity.py`, narrow wells — did **not** reproduce the instability, corr +0.56…+0.93 with 0/24 sign flips. "A distance margin cannot predict drift" is FALSE as a general statement; the width sweep is what located the failure.)*

### 1.2 The replacement predicts drift — head-to-head, gym-like regime (12 seeds)

| leg | mean | min | sign flips | below the 0.30 bar |
|---|---|---|---|---|
| shipped: `−corr(margin, drift)` | **+0.412** | **−0.111** | 1/12 | 4/12 |
| **replacement: `spearman(log B, log Δ)`** | **+0.914** | **+0.782** | **0/12** | **0/12** |

with per-pair soundness `P[Δ > 2B] = 0.056` and tightness `median Δ/B = 0.863`. Pooled over 1800 pairs and five widths: `spearman = +0.916`, `median Δ/B = 0.974`, `q95 Δ/B = 4.20`, `P[Δ>2B] = 0.089`, `P[Δ>3B] = 0.062`.

### 1.3 ⭐ THE SPEC `phi-particle-head` LANDS (monitor #3, leg ii)

> **Leg (ii′) — the C3 first-order calibration leg.** At each admitted write `j`, for each of the `k` nearest incumbents `i`:
> ```
> B_ij = ‖∇δV_j(q_i*)‖ / λ_min,i           # predicted drift, first order
> Δ_ij = ‖q_i*(after) − q_i*(before)‖      # measured AT THE RELAXED FIXED POINT (N74)
> ```
> **Qualifying pair** iff `λ_min,i > λ_floor` **and** `Δ_ij > δ_num` (`δ_num` = the settle's numerical floor, 1e-6 at 900 steps).
> **Statistics:** `ρ_C3 = median(Δ/B)` (calibration) and `v = P[Δ > κB]` (soundness), `κ = 3` declared.
> **Trip iff** `ρ_C3 ∉ [1/3, 3]` **or** `v > 0.10`. **INAPPLICABLE** (never "pass") if fewer than 3 qualifying pairs.
> **Cost: zero.** `chlu/core/clu_system.py::_c3_check` already computes exactly this ratio at every write (it feeds monitor #12(b)); the leg is a predicate on an existing quantity. **Fix S3 first** — the `max(λ, 1e-9)` clamp makes `B → ∞` and `ρ_C3 → 0` at any non-minimum, i.e. it reports a perfect certificate precisely where there is none.
> **The old leg survives as a REPORTED diagnostic only**, annotated with `d/s`; it may not trip.

**Constants, derived not tuned.** `κ = 3` from the pooled `q95(Δ/B) = 4.20` rounded inward to the nearest integer and cross-checked against N74's shipped gated band `[0.73, 1.63]` (which is *inside* `[1/3, 3]`, so a healthy store passes with room). `η = 0.10` from the measured ungated `P[Δ>3B] = 0.062` plus a 1.6× margin. **The `B ≤ c_q·s` linearisation guard is deliberately NOT used for the trip decision:** it improves soundness (`c_q = 0.1` → `P[Δ>2B] = 0.036`) but discards **57 % of the total drift mass** — i.e. it silences exactly the pairs the monitor exists to catch. Report it; do not gate on it.
**Expected effect:** in the gym, 18/28 cells become **INAPPLICABLE** on leg (ii) (their sites are not minima, which is #8-N3's trip and must not be double-counted), and the three shipped-anchor cells pass with `ρ_C3` computable rather than a correlation.

---

# 2. ⛔ BLOCKING #2 — R2(gym): the monitor-#6 dead-band `eps`

### 2.1 The derivation (two floors; the larger binds)
`slope = polyfit(arange(n), y, 1)[0]`, `n = window+1 = 4`, span `W = n−1 = 3`.
1. **Roundoff floor.** For an exactly-constant window the true slope is 0 and the computed value is `O(c·u·Y)`, `u = 2⁻⁵³ = 1.110e-16`, `Y = max|y|` in the window. Measured envelope over 16 000 constant and constant-to-1-ulp windows spanning 10 decades: **`c_max = 3.099`** (p99.9 = 2.824). Take `c = 8` (2.6× headroom).
2. **Resolution floor.** A slope whose extrapolated change over the window is below the quantity's own resolution is not a measurement. `acq` is a proportion over `n_probed` self-probes ⇒ quantum `1/n_probed`; the write loss has no quantum, so use a relative resolution `f_rel = 1e-3` of the window's own scale.

### 2.2 The numbers (re-scoring the 28 published gym cells, `r2_results.json`)
- artefact population (`|slope| < 1e-10`): **n = 12, max = 5.193e-17**
- genuine population: **n = 16, min = 3.739e-04**
- **separation = 12.86 orders of magnitude**; the trip table is **invariant for any `eps ∈ [5.6e-17, 3.2e-4]` (12.8 orders)** — the choice is not a tuning decision.
- the derived `eps_loss` spans **9.49e-10 … 1.54e-4** across cells (it is scale-relative, as it must be: the shipped-anchor cells run at write loss 3e-4, the others at 0.2).

### 2.3 ⭐ THE SPEC `phi-particle-head` LANDS (`chlu/core/monitors.py:670`)
```python
U   = 2.0**-53                       # 1.110e-16
W   = self.window                    # = n - 1 = 3 by default
Yl  = float(np.max(np.abs(h[:, 0]))) # window scale of the write loss
Ya  = float(max(np.max(np.abs(h[:, 1])), 1e-12))
npb = int(ctx.get("self_probe", "n_probed", 0) or 8)
eps_loss = max(8.0 * U * Yl, 1e-3 * Yl / W)
eps_acq  = max(8.0 * U * Ya, 1.0 / (npb * W))
tripped  = bool(slope_loss < -eps_loss and slope_acq <= +eps_acq)   # NOTE the +
# report eps_loss, eps_acq in detail{} for auditability
```
**The `+eps_acq` is load-bearing and is the half nobody asked for.** The shipped `slope_acq <= 0.0` means a `+1e-17` roundoff slope counts as "acquisition is rising" and *suppresses* a genuine trip: the same missing dead-band produces **false negatives** as well as false positives.

**Measured effect on the 28 published cells: 13 trips → 9.**
- **6 removed** (all artefacts): `overload/load1x_ref8@s0`, `overload/load1x_ref3@s0`, `overload/load1x_shipped@s0`, `overload/load1x_shipped@s2`, `manifold/base@s0`, `manifold/base@s2`.
- **2 added** (recovered false negatives, `slope_loss = −4.0e-2` with `slope_acq = +7.8e-4 ≪ eps_acq`): `overload/base@s0`, `overload/reach_free@s0`.
- **7 kept, 0 genuine trips lost.**
- ⭐ **Consequence for the C2W1 record:** the gym's "clean" configuration `overload/load1x_shipped` trips **only #3** after this repair, not #3 and #6.

*(PREREG P-B predicted this cell-by-cell before the re-score: 6 lost, 2 gained, the same names, 12.9 orders of separation. Confirmed exactly. `c_max` predicted ≈2.3, measured 3.099 — inside the registered `c ≤ 8` bound, point estimate 1.35× low.)*

---

# 3. The nine owed reconciliations — dispositions

### R1 — `Prop D1`'s `sep/2` inradius proxy · **fix = theory + code (S5, S7)**
**Corrected form** (two-well saddle balance, equal widths, linearised in `ln(A_i/A_j)`):
```
r_i ≈ min_j [ d_ij/2 + δ_ij ],   δ_ij = ln(A_i/A_j) / ( d_ij/s² − 4/d_ij )      (deeper well ⇒ larger basin)
                                 → d_ij/2 + (s²/d_ij)·ln(A_i/A_j)   for d ≫ 2s
```
**Validity domain (all four required; outside it the proxy is FORBIDDEN):**
`(V1) λ_min,i > 0` · `(V2) s/sep ∈ [0.15, 0.30]` · `(V3) |δ_ij| ≤ 0.25 d_ij` · `(V4) the basin is nonempty under the *shipped γ*, not merely `λ_min > 0`` (see the new blocker below).

**Verification** (`r1_results.json`; the s2b store reproduced: K=6 ring, sep = 2.0, s = 0.35, depth ratio 2.155):

| | measured `r_i` (32-direction bisection) | max abs error |
|---|---|---|
| `sep/2` | — | **0.0469 (4.7 %)** ← the published "up to 4.8 %" |
| **corrected** | `[0.953, 0.994, 1.005, 0.961, 0.953, 1.045]` | **0.0032 → 14.55× better** |

Width sweep (2 wells, depth ratio 2, spacing 2):

| `s/sep` | 0.10 | 0.15 | 0.20 | 0.25 | 0.30 | 0.375 | 0.45 |
|---|---|---|---|---|---|---|---|
| `sep/2` rel. error | 30.8 % | 3.9 % | 6.9 % | 12.1 % | 20.6 % | **100 %** | **100 %** |
| corrected rel. error | 29.3 % | **0.5 %** | **0.3 %** | **0.5 %** | **1.1 %** | 55 % | 58 % |
| `λ_min` (shallow well) | +17.60 | +7.88 | +4.47 | +2.87 | +1.89 | **+0.910** | **+0.388** |

⭐ **Two new blockers found, both load-bearing:**
- **Below `s/sep ≈ 0.15` the basin boundary is INERTIAL, not a static watershed.** At `s/sep = 0.10` the deeper well's measured inradius is **1.306** against a midpoint of 1.000 (and the shallow well's 0.692) while the static correction predicts a shift of only 0.014. Under the shipped underdamped read (`γ = 0.05`) a particle released between narrow wells carries enough momentum to be captured by the deeper one from far past the watershed. **No static proxy is valid there** — including the corrected one.
- **`λ_min > 0` does NOT certify a nonempty basin.** At `s/sep ≥ 0.375` the shallower well is still a genuine minimum (`λ_min = +0.910`) and its measured basin is **0.000** — every trajectory escapes over the low barrier. The certified inradius is a **dynamical** quantity depending on `γ` and the barrier height `h_i`, not on the Hessian alone. (This is the same physics as the ladder rule C2-N3 `E < h_i`, now on the *read* side.)

**The consequence for monitor #2, and the disposition of "D1 is violated".** `U` computed with an over-claimed radius is biased **low** — measured: `U(sep/2)` vs `U(measured r)` = 0.1893/0.2003, 0.1857/0.2330, 0.1785/0.3847 (**up to 2.16× under-count**). In the gym, **all 7 cells with `D/U > 4` have `λ_min < 0`** (−0.372 … −1.199): their certified radius is **0**, so `U` was computed against a certificate that does not exist.
> ⛔ **Prop D1 was never violated. `D ≤ U` is a theorem under a *certified* ball; the gym measured `D` against an *uncertified* one.** The honest statement is: **`sep/2` may not be quoted as a certified inradius anywhere; where `λ_min,i ≤ 0`, `r_i := 0` and monitor #2 must return INAPPLICABLE.** (My toy did not itself produce a D1 violation — `D/U` = 0.04/0.19/0.52 with either radius — so the 2.16× under-count is *sufficient in direction* but not, by itself, in magnitude for 7.44×; the rest is the `r=0` effect above.)

### R-1 — `k* = 269` governs an endpoint loss · **fix = theory + wording (S9)**
**The corrected statement, in two parts** (`rm1_results.json`; exact forward-mode sensitivities through the shipped map, 1-D two-well, `γ=0.05`, `dt=0.05`, `N=400`, measured `ρ = 0.97468`, measured `∂q_n/∂q₀` decay `0.97451` = 0.02 % off):

**(a) Read-out axis.** For a ψ that pools with weights `w_n`, the retained depth obeys
> ### `k*(ε) ≤ W_ψ(ε) + ln(1/ε)/ln(1/ρ)`, `W_ψ(ε)` = smallest tail window carrying `1−ε` of the pooling mass.

| ψ | `W_ψ(1e-3)` | `k*` predicted | `k*` measured |
|---|---|---|---|
| endpoint | 1 | **270.3** | **269** ✅ (0.5 %) |
| exp tail τ=10 | 70 | 339.3 | 279 |
| exp tail τ=40 | 275 | 544.3 | 349 |
| **uniform (whole window)** | **401** | 670.3 | **400 = N ⇒ no truncation exists** |

Measured tail-truncation error of the **θ**-gradient under a uniform ψ: **0.9999 / 0.9999 / 0.9993 / 0.9946** at k = 50/100/180/270 — flat and O(1), the same qualitative result as the engineer's 0.680 (theirs is milder because their phase 2 was always fully retained). The bound is **tight for the endpoint read-out and conservative by ≤22 % for exponential tails**; use it as a design rule, not an equality.

**(b) ⭐ Parameter axis — new, and it bites a K-item store.** Even for an *endpoint* loss, `ρ^k` governs only those θ whose **fixed-point** sensitivity dominates their **transient** sensitivity:

| θ | `max_n|∂q_n/∂θ| / |∂q_N/∂θ|` | trunc. err at k=270 | `ρ²⁷⁰` |
|---|---|---|---|---|
| depth of the **settled** well | **64×** | **2.478e-05** | 9.83e-04 ✅ Q4.2 holds |
| depth of a **far** well (`‖c−q*‖ ≈ 2`) | **27 396×** | **0.448** | 9.83e-04 ⛔ wrong by 456× |

In a K-item store the "far well" case is **every item that is not the one being read** — i.e. exactly the crowding/interference gradients that teach the store to separate items. **Truncated BPTT preserves the on-well gradient and destroys the interference gradient.** The discriminator is cheap and runtime-computable: `max_n|∂q_n/∂θ| / |∂q_N/∂θ|`.
**Every quoting site gains:** *"…of `∂q_N/∂θ`, and only for parameters whose transient/fixed-point sensitivity ratio is O(10) or less."*

### R-2 — the §7 truncation recipe makes φ untrainable · **fix = theory (S9)**
**Named: the direction is TAIL, and φ lives at the HEAD.** `∂L/∂φ = Σ_n w_n (∂q_n/∂q₀)(∂q₀/∂φ)` with `∂q_n/∂q₀ = ρ^n` — the sum is dominated by the **earliest** points and is complete after `O(1/(1−ρ))` steps. A tail-retention seam makes `q₀` a constant, so the gradient is **exactly** zero.
**Reproduced exactly and generalised:** `‖∂L/∂φ‖ = 0.0` (bit-exact) at every `k < N` for **all five** pooling profiles, endpoint included.
> ### The corrected recipe: a **two-sided retained window** — head `h* = ln(1/ε)/ln(1/ρ)` for φ, tail `k*` (above) for θ, `stop_gradient` on the middle.
Head truncation measured (uniform ψ): rel. error **0.2116 / 0.0472 / 0.0040 / 1.18e-5** at h = 50/100/180/270 vs `ρ^h` = 0.2774 / 0.0769 / 0.0099 / 9.83e-4 ⇒ **the same `k*` number, at the other end, and slightly conservative** (`h*(1e-3) ∈ (180, 270]` vs the bound 269).
⛔ **And a refuted convenience.** My own PREREG P-E predicted that under a whole-window ψ the pooled θ-gradient is a scalar multiple of the implicit gradient (`0.967` expected). **REFUTED, badly:** measured pooled/endpoint ratio **1093×** (uniform), **250×** (τ=150), **−0.465×** (τ=10 — *opposite sign*). The linear-model factorisation `∂q_n/∂θ = (1−ρⁿ)∂q*/∂θ` fails because the trajectory's *transit-time* sensitivity is not a fixed-point sensitivity. ⇒ **The implicit/DEQ gradient is not a drop-in substitute for a trajectory-read θ-gradient; they are different objects and can differ in sign.** This is a direct constraint on any C2 trainer that hoped to use implicit-for-θ + head-window-for-φ: it must be *measured* per store, not assumed.

### R-3 — monitor #10's "dead axis" wording · **fix = wording (S10)**
**Confirmed as the engineer states it; nothing to derive.** `settled_point_psi` moves **exactly 0.000** noise units at every stride × 3 seeds because that ψ never touches the buffer; `tail_mean_psi` moves 4.5e-4–2.5e-2 (still ≪ the 3σ bar). The correct wording is **"no shipped ψ consumes the buffer"**, not "the knob is inert". Consequence for the table: **monitor #10 tier (b) must skip any knob for which no live consumer is registered, and report `no_consumer`, not `dead`** — otherwise every trajectory knob trips forever while the point read ships. `traj_stride`/`gamma_read` become genuinely testable only once a buffer-consuming ψ is the read of record. Sites: handover §7 (from `full-clu-harness` R3), `monitors.py` docstring for #10.

### R2(gym) → §2. R3(gym) → §1.

### R2(doctrine) — the γ band's constants do not transfer · **fix = theory + my own erratum (S6)**
**The harness-invariant object is not γ. It is the convergence budget**
> ### `C ≡ Σ_phases N_p·ln(1/ρ_p)`, in band iff `C ≥ ln(1/tol)`; `ρ = ` spectral radius of the linearised damped-Verlet propagator.
Edges (single phase): `γ_min = 1 − exp(−2 ln(1/tol)/N)`; `γ_max = 2κ/(2ln(1/tol) + κ)`, `κ = N dt² λ̄`.

**Verification against the published `s2b` γ×N table** (`λ̄ = 2α + Ā/s² = 7.539`, `dt = 0.05`, `tol = 1e-6`): **4/4 N-values match exactly at grid resolution** — N=100 → ∅ (predicted `[0.241, 0.128]`, empty); N=400 → `{0.1, 0.2}` (predicted `[0.0667, 0.4287]`); N=1500 → `{0.02…0.9}` (predicted `[0.0183, 1.011]`); N=4000 → `{0.01…0.9}` (predicted `[0.0069, 1.464]`). Over the whole table, `ρ^N` reproduces the measured `ρ_conv` with **median ratio 0.94×** and 20/24 cells within 3 orders; the 3 residual outliers are all cells whose *measured* value sits at the table's numerical floor (2.4e-14 vs a floor of ~2.1e-15), i.e. the measurement is saturated, not the law.

⭐ **Sharpening of Q4.2's `ρ`.** The published form `ρ = max(√(1−γ), |1−(2−γ)dt²λ/(2γm)|)` **exceeds 1 in the weakly-damped regime and then predicts divergence**: at `γ=0.005, N=4000` it gives `1.000` where the truth is `4.452e-05` (4.3 orders). The **exact** `ρ` — the spectral radius of
`M = [[1−a, dt], [−(1−γ)(dtλ/2)(2−a), (1−γ)(1−a)]]`, `a = dt²λ/2` — gives **4.428e-05 (0.5 % error)**. Use `M`; keep the closed form only as an interpretive bracket.

⛔ **My own erratum (S6).** The published row-1 band pair is scored at **two different tolerances**: `γ ∈ [0.05, 0.5] at N=400` is a `tol = 1e-3` band (the table reads `3.62e-05` at `γ=0.05, N=400`, which fails the row's own `1e-6`), while `γ ∈ [0.02, 0.9] at N=1500` is a `tol ≤ 1e-4` band. At `1e-6` the N=400 band is `[0.1, 0.2]`. **Retire both quoted intervals; quote `C` and the two edge formulas.**
**Applied to `full-clu-harness`:** `C_addr(γ=.05, N=400) = 10.26` + `C_read(γ=.02, N=800) = 8.08` = **18.34 ≥ 13.82** ⇒ in band, consistent with its measured `4.3e-7` (the budget is conservative by ~40×). The annealed read's trip is then not a band violation but a **budget** statement: any schedule change must be re-scored as `Σ N_p ln(1/ρ_p)`, which is exactly why R3's "+2× read steps" fixed it (`C → 26.42`).

### R5(gym) — the reach probe that also clears `d_safe` · **fix = construction (spec below)**
**Diagnosis:** the two constraints act on **different coordinates** and are separable — admission is evaluated on **address coordinates only** (`chlu/core/admission.py`: *"q_new: proposed site (address coordinates only)"*), while reach is a single-site condition on `L = √(‖c‖² + ‖a‖²)`. The gym's probe was refused because it let the family's placement choose the address, not because the constraints conflict.
> **The construction.** (1) `c_probe = argmax_{‖c‖ = R_ball} min_j ‖c − c_j‖` — the farthest-point site **on the ball boundary** (the boundary simultaneously maximises separation from an interior-packed store **and minimises `a_U`**, which is monotone decreasing in `‖c‖`: measured **1.1548 → 0.8905** over `‖c‖ = 0.2 → 1.2`). (2) assert `min_j‖c_probe − c_j‖ ≥ d_safe`; if it fails, **no probe exists and that is #12/`expand`, not #11**. (3) `‖a_probe‖ = 1.15·a_U(‖c_probe‖, s, D, α)`.
> **Feasibility condition:** `max_c min_j ‖c−c_j‖ ≥ d_safe` ⇔ the ball still admits one free site ⇔ `N_pack(R_ball, d) > K`.

**Verified at gym-like parameters** (`d=4, R_ball=1.0, d_safe=0.58, s=0.30, D=0.50, α=0.05`): at K = 6/8/12/17 the probe has `d_min` = 1.273/1.027/0.948/**0.893** (all ≥ 0.58, margin **+54 %** at the gym's own K=17) and reach margin **−0.1390 (−15 %)** ⇒ **#3 admits, #11 trips — exactly the intended probe.** Frontier: feasible up to **K = 38** at these parameters. *(PREREG P-G asked for ≥20 % gate margin and ≥10 % reach violation; measured 54 % and 15 %.)*

### I-14 / #9 — what C2 may claim about lifetimes until gated stiffness is on
**Disposition: a scope clause, not a fix.** Monitor #9 trips everywhere except R1/R3 and is pre-declared uncleanable; C1W27's option-(d) gated stiffness measured the band **payload-independent at every amplitude (N119)** but **ships OFF**, and C2 must not build it.
> **Admissible for C2, verbatim:** *"Per-item lifetimes are settable and the decay law is exact (`A ← A·e^{−leak}`, commutes with delete, PGCP Thm 4); with the shipped potential the **realised** retention additionally depends on the payload excursion (`Δ_ret` 0.047 → 0.467 as the excursion goes 0.15 → 0.70·R in the C2W1 toy), so a lifetime is a **dial on the amplitude**, not yet a guarantee on the recall. A payload-independent lifetime has been measured (N119) but is not in the shipped configuration."*
> **Inadmissible until gated stiffness ships:** "lifetime is a dial you set" without the excursion clause; any half-life claim quoted at a *fixed* recall threshold across items of different `‖a‖`; any #9 trip counted as harness failure (I-14 stands).
> **Runtime consequence:** #9 stays a *reported, pre-declared-uncleanable* trip. It may **never** gate a wave's acceptance criterion, and — new — it may not be used to *rank* configurations either, because its magnitude is dominated by the excursion distribution, which is a property of the task, not of the store.

---

# 4. ⭐ THE ADJUDICATION — charter §A2.5 (P1 / merge certificate)

### 4.1 PREREG scorecard

| # | pre-registered | measured | verdict |
|---|---|---|---|
| **P-A1** | all 7 zero-trip configs satisfy `2s_max + 2.576σ_q ≤ sep`; margins **+0.1271** (R=1.3) and **+0.3271** (R=1.5) | **7/7**, margins **+0.1271** ×4 and **+0.3271** ×3 | ✅ **exact** |
| **P-A2** | N2 ⇒ certificate iff `s_max ≤ 1.287 σ_q`; grid ratio 1.156 ⇒ implication holds at all 54 points | threshold **1.2870**, grid max ratio **1.1556**, implication **36/36** of the N2-satisfying configs; grid-wide 48/54 | ✅ **exact** |
| **P-A3** | ≥1 witness (13 bands clear + certificate violated) in an extended grid; 3–15 % of it | **0 witnesses / 72 configs** under the shipped gate | ⛔ **REFUTED — and the refutation is the finding (§4.3)** |
| **P-B** | #6 `eps` spec; 6 trips lost / 2 gained, named; 12.9-order separation | **6 lost / 2 gained, the same names**; 12.86 orders; `c_max` 3.099 (predicted 2.3, bound ≤8) | ✅ (point estimate 1.35× low) |
| **P-C** | λ_min>1 group all positive, ≥+0.35 mean; λ≤0 group ≤+0.05 and sign-unstable; replacement `corr ≥ 0.90` | **+0.470 / −0.013**; replacement spearman **+0.914 mean, +0.782 min** | ✅ |
| **P-D** | corrected inradius max error < 0.015, ≥3.2× better; all 7 gym `D/U>4` cells have λ_min<0 | **0.0032, 14.55×**; **7/7** have λ_min ∈ [−1.199, −0.372] | ✅✅ (better than registered) |
| **P-E** | `k* ≤ W_ψ + ln(1/ε)/ln(1/ρ)`; tail⇒`∂L/∂φ = 0` exactly; head err `= ρ^h`; **pooled θ-grad = 0.967 × implicit** | law holds (endpoint 270.3 vs 269); φ **exactly 0.0** ×5 profiles; head 0.2116 vs ρ^h 0.2774; **pooled/endpoint = 1093×, −0.465× (sign flip)** | ◐ **3 confirmed, the scalar-multiple claim REFUTED** |
| **P-F** | 4/4 band edges predicted; the published band mixes tolerances | **4/4**; `[0.05,0.5]@400` is a `1e-3` band, `[0.02,0.9]@1500` is `≤1e-4` | ✅ |
| **P-G** | probe: gate margin ≥20 %, reach violated ≥10 % | **+54 %**, **−15 %**, feasible to K=38 | ✅ |

**Score: 6 confirmed (2 sharper than registered) · 1 partial · 2 refuted (P-A3, P-E's scalar claim).** Both refutations are findings.

### 4.2 §A2.5 under the shipped rule: CONFIRMED, and it is *not* a theorem
- **Scored per config:** all **7/7** zero-trip configs of the published 54-point grid satisfy the merge certificate, with margins **+0.1271** (m=1 R=1.3 ×3, m=2 R=1.3 ×1) and **+0.3271** (m=1 R=1.5 ×3). `s_max = 0.27735 = 0.20 × 1.38675` (the grid's largest anisotropy factor).
- **Is it forced?** Partly, and by an identifiable inequality: **N2 (`sep ≥ 5.15σ_q`) implies the certificate iff `s_max ≤ (5.15 − κ′)/2 · σ_q = 1.287 σ_q`.** The grid sits at `s_max/σ_q = 1.1556` — inside by **11 %** — so on *this* grid the implication holds at every point (36/36 of the N2-satisfying configs) and the 7/7 is **forced by N2, not independent evidence**. ⇒ **§A2.5 is confirmed as a property of the searched region, not as a theorem.** The general statement requires `s_max ≤ 1.287σ_q`, which no band enforces.
- **Extension (72 configs, well width added as an axis, `s_max/σ_q ∈ [0.99, 3.47]`, `a25_results.json`):** 6 zero-trip configs, **0 of them violating the certificate**. §A2.5 survives an honest attempt to break it.

### 4.3 ⭐ WHY — and the refutation the campaign should act on
Every certificate-violating config in the extension trips **monitor #3's fire-rate leg**, and the mechanism is structural, not statistical:
> **The shipped harness sets `d_safe := 2s_max + κ′σ_q` (doctrine I-13/R5). The admission radius IS the certificate radius. Driving the certificate into violation therefore drives the gate's fire rate to `f = 1.000` (it refuses everything), and `f ∈ {0,1}` is #3's trip. The two bands are one object.** This is the harness's "mutually exclusive by construction" statement, now with its cause.

**Decouple them and the region opens** (`a25b_results.json`; identical stores, admission radius set to the harness's own S4 convention `d_safe = 0.6·sep`, everything else unchanged):

| config (m=1) | cert. margin | deficit (% of `sep`) | `f` hard → soft | `λ_min` | `D` | `ρ_ex` | acq | dividend | 13 bands |
|---|---|---|---|---|---|---|---|---|---|
| **C2W1 witness** R=1.3 σ=.24 w=0.20 | **+0.127** | — | 0.983 | 12.45 | 0.0053 | 0.127 | 0.992 | +0.0033 | clear (hard) |
| R=1.3 σ=.24 w=0.30 | −0.150 | 11.6 % | 1.000 → **0.963** | 5.59 | 0.0077 | **0.183** | 0.991 | +0.0043 | **clear** |
| R=1.3 σ=.24 w=0.40 | −0.428 | 32.9 % | 1.000 → **0.963** | 3.19 | 0.0123 | **0.294** | 0.985 | −0.0017 | **clear** |
| R=1.6 σ=.24 w=0.40 | −0.128 | 8.0 % | 1.000 → **0.963** | 3.19 | 0.0033 | **0.270** | 0.998 | −0.0007 | **clear** |
| R=1.6 σ=.20 w=0.50 | −0.302 | 18.9 % | 1.000 → **0.963** | 2.08 | 0.0027 | **0.800** | 0.998 | −0.0020 | **clear** |

**11 of 12 tested soft configs clear all 13 bands while violating the merge certificate** (the 12th fails #8-N2, unrelated). ⇒ **§A2.5 is refuted the moment the certificate stops being the gate.** The intersection *does* reach into the non-separable regime; it was the `d_safe ≡ certificate` identification that hid it.

**The price, quantified (this is what Prop D1's guarantee costs):**
- `λ_min` falls **2.23×** (w=0.30) to **6.00×** (w=0.50) — the implicit-gradient conditioning and the truncation depth both scale as `1/λ_min` (trainability-spike Q3.5/Q4.2), so this is a **2–6× trainer cost**, in the same currency as Q3.3's `d_safe` price.
- `ρ_ex` (exploitation head-room) **rises 1.44× → 6.30×**, `D` rises 1.4–2.5×: the *room* for a dividend grows by up to 6×.
- acquisition is essentially unchanged (**−0.0067 … +0.0073**); the measured dividend stays **≈0** (+0.0043 … −0.0067).
> ⛔ **Read this honestly: the soft certificate removes a structural blocker; it does not create a dividend.** With a settled-point read the dividend stays ≈0 exactly as A2.1's point-estimator diagnosis predicts. The relaxation is a *precondition* for C2W3, not a result.

**The violation budget is measurable, and it has an outer edge.** At R=1.3, σ_q=0.24 the sequence w = 0.20/0.30/0.40/**0.50** gives deficits 0/11.6/32.9/**54.2 %** and `ρ_ex` 0.127/0.183/0.294/**7.94** — the last one is `D > U`, i.e. the R1 estimator breaking down (wide wells ⇒ `sep/2` over-claims ⇒ `U` under-counted). ⇒ **budget ≤ 33 % of `sep` on this grid**, with the caveat that the edge is currently located by the *broken proxy*; with R1's corrected radius the true edge is likely further out. **R1's repair is therefore a prerequisite for setting the budget, not an independent cleanup.**

### 4.4 ⭐ THE SOFT-CERTIFICATE SPEC (C2W3's factored store, charter §A4.5)

> **SC-1 — break the identification.** `d_safe` becomes an **independent, declared** admission radius `d_safe = ζ·sep_expected` (`ζ = 0.6`, the harness's existing S4 convention), **not** `2s_max + κ′σ_q`. The certificate radius `R_cert = 2s_max + κ′σ_q` is still computed and reported; it is no longer the gate. *(This retires the `d_safe_override` hack: the gym needed it twice and the harness once, always "deliberately out of band" — the override was the soft certificate, undeclared.)*
> **SC-2 — the reported margin.** Every admitted write logs `cert_margin = sep_after − R_cert` and `deficit_rel = max(0, −cert_margin)/sep_after`.
> **SC-3 — the violation budget (declared, per run, in PREREG).** `deficit_rel ≤ B` with **`B = 0.33`** as the C2W3 default (measured edge, §4.3), and `mean(deficit_rel over live items) ≤ B/2`. Exceeding the budget is a **trip of #3**, not a refusal.
> **SC-4 — what replaces the guarantee (three runtime legs, all already computed):** (i) `λ_min,i > λ_floor` at every live site (#8-N3) — this is the *exact* non-merger condition the certificate was approximating; (ii) the **C3 calibration leg of §1.3** — the certificate's job was to bound crowding drift, so bound it *measured*; (iii) monitor #2 reported with the **corrected** radius (§3-R1), INAPPLICABLE where (i) fails.
> **SC-5 — what is given up, stated in the artifact.** *"Prop D1's guarantee (`settle = argmin` inside every certified ball, hence `D ≤ U`) no longer holds a priori. It is replaced by a measured pair (`λ_min`, `ρ_C3`) and a declared budget. The measured price on this grid: `λ_min` ÷2.2–6.0, implicit-gradient conditioning and truncation depth ∝ 1/λ_min, `ρ_ex` ×1.4–6.3."*
> **SC-6 — the hard floor that does NOT relax.** `λ_min,i > 0` at every live site. Below it the site is not a minimum, the read is not a settle, and **R1's new blocker applies: `λ_min > 0` is necessary but not sufficient for a nonempty basin under the shipped `γ`** — so C2W3 must additionally measure a **capture radius** at least once per consolidation (32-direction bisection at one site; ~6 s in the toy) and refuse to certify any site whose measured basin is below `σ_q`.
> **SC-7 — falsifier for C2W3.** If the factored store's shared wells cannot hold `λ_min > λ_floor` at *any* admissible `B`, then basin interaction and non-degeneracy are genuinely disjoint and **that is the Head-ruling escalation**. On this grid they are not: at `B = 0.33` we measure `λ_min = +3.19` with `ρ_ex = 0.294`.

---

# 5. The 13-row diff against my own C2W1 table

Provenance: `[gym]` = `memory-gym-v0` measurement, `[harness]` = `full-clu-harness`, `[spike]` = `trainability-spike(-theory)`, `[here]` = this report.

| # | verdict | what changes |
|---|---|---|
| 1 | ◐ **SHARPENED (constants retired)** | the γ **band** is retired as a portable object; the invariant is the **convergence budget** `C = Σ N_p ln(1/ρ_p) ≥ ln(1/tol)` with the exact 2×2 propagator `ρ` `[here]`. Both published intervals were scored at the wrong tolerance (S6). Predicate itself (`ρ_conv`, `δ`) **unchanged and confirmed** — it is the *threshold* that must be re-derived per harness. |
| 2 | ◐ **SHARPENED (twice)** | (a) `U` must use the **corrected** inradius and the monitor is **INAPPLICABLE where `λ_min ≤ 0`** `[here]` — this retires the "D1 violated 1.5–7.44×" claim `[harness][gym]`; (b) I-6's `U < 0.01` inapplicability **confirmed as landed**; (c) `D` is **not a progress signal** — the `D = 0.931` cell has dividend −0.875 `[gym]`. Still **no verb**. |
| 3 | ⛔ **REPLACE (leg ii)** | the correlation leg is retired → the **C3 calibration leg** (§1.3); fire-rate and utilisation legs unchanged; `d_safe` is **decoupled from the certificate** (§4.4 SC-1) so the fire-rate band and the certificate stop being one object `[here]`. |
| 4 | ✅ **CONFIRM** | 0/56 trips in the gym, 0 in the harness; the empirical-marginal chance rule works. The trajectory-launder rider (I-2) is **still owed** and becomes live the moment a buffer-consuming ψ ships. |
| 5 | ✅ **CONFIRM** | `acq` self-probe unchanged; the annealed read remains the measured fix (0.828 → 1.000 `[harness]`). |
| 6 | ⛔ **REPLACE (predicate)** | two-sided dead-band, §2.3 `[here]`. Kills 6 artefact trips and **recovers 2 false negatives** `[gym]`. |
| 7 | ◐ **SHARPENED (scope)** | trajectory-wise + `kinetic_mode` confirmed landed (<1e-5, Newtonian `[harness]`). **Add:** the gauge test must state which θ it varies — R-1(b) shows endpoint-only quantities have parameter-dependent sensitivity ratios up to 27 396× `[here]`. |
| 8 | ◐ **SHARPENED** | N2 unchanged (`erf(2.576/√2) = 0.9900` confirmed in code `[harness]`). **N3 is weakened by measurement: `λ_min > 0` does NOT certify a nonempty basin** (measured basin 0.000 at `λ_min = +0.910`) ⇒ N3 gains a **capture-radius** leg `[here]`. The `sep/2` cheap proxy is **replaced** by the corrected form with a stated domain `s/sep ∈ [0.15, 0.30]`; below 0.15 the boundary is **inertial** and no static proxy applies. |
| 9 | ✅ **CONFIRM + scope clause** | effect-size predicate unchanged; add the I-14 admissibility text (§3) and the new rule that #9 may not **rank** configurations. |
| 10 | ◐ **SHARPENED (wording + a new state)** | "dead axis" → **`no_consumer`** vs `inert` `[spike]`; tier (a) access counter still owed (`knob_tier_a_implemented: false` `[harness]`). |
| 11 | ✅ **CONFIRM — still the strongest row** | reproduces 7/7 published `a_U` anchors ≤0.1 % `[harness]`. **Add** the R5 probe construction and its feasibility condition `N_pack(R,d) > K` `[here]`. |
| 12 | ✅ **CONFIRM + a promotion** | the C3 ratio is now **load-bearing twice** — it is also monitor #3's validity leg (§1.3). Fix the silent `λ ≥ 1e-9` clamp (S3) before either consumer trusts it. |
| 13 | ✅ **CONFIRM** | unchanged. |
| M14 | ⚠ **STILL UNEXERCISED** | neither the gym nor the harness ran a canary `[gym §3.6]`. It is the only check that catches a *learned* policy making a guard arithmetically vacuous — and §4.3 just showed the shipped code contains exactly that pathology (`f = 1.000` by construction) with no learning required. **Priority raised.** |

**Score: 5 confirm · 6 sharpen · 2 replace · 1 raised-priority.** Both replacements are, again, cases where a **correlation** was used where a **calibrated bound** was needed — the same failure mode as C2W1's rows 1 and 9.

---

## 6. Falsifiers (task §4), adjudicated
- ⛔ **§A2.5 refuted?** **Not under the shipped rule** (0/72). **Yes under the soft certificate** (11/12). Reported as the headline; it re-prices the campaign **upward** and hands C2W3 a target region with a measured price. **No Head ruling required.**
- ⛔ **A mode with no runtime-computable invariant after the repairs?** **Did not fire.** Every one of the 13 still has an invariant computable from quantities CLU has; #3's new leg is computed by shipped code today.
- ⛔ **Two productive bands provably disjoint after the soft relaxation?** **Did not fire — the opposite fired.** The pair that looked disjoint (#3 fire-rate × the certificate) was **one object**, and separating them makes both satisfiable simultaneously. **No escalation.**

## 7. Git footprint
**None.** No tracked code touched; repo read-only at `main @ 233fd9e`, clean. All artifacts under `.claude/scratch/doctrine-repairs/` and `.claude/outputs/doctrine-repairs/`.

## 8. Open questions / risks
- **OQ-A (the budget's edge is measured with a broken ruler).** `B = 0.33` is located by the `sep/2` proxy's own breakdown. Re-locate it with the corrected radius + measured capture radius before C2W3 freezes `B`. **Cheap (≈10 min) and it is the single highest-value follow-up here.**
- **OQ-B (the soft witnesses' N4 leg is vacuous).** In **all 11** soft witnesses `δ_read` collapses to the settle's numerical floor, so `N4 = pay_sep/(2δ_read)` returns **2.3e4 … 1.3e11** (9 of the 11 at ~1e11). The band's *intent* (payload separation exceeds read noise) is satisfied maximally — every in-basin query lands on the same fixed point — but the **number is meaningless**; report `N4: satisfied (δ_read below floor)`, never the ratio. The 13-band clearance of the soft witnesses therefore rests on 12 informative legs and one that passes vacuously; that caveat travels with §4.3.
- **OQ-C (composition, carried and unchanged).** Everything here is designed wells, `d = 2`, K = 6, one seed. `s_max/σ_q`, `d/s` and `s/sep` are the three ratios that turned out to control R1, R3 and §4 — **all three are unmeasured on the shipped learned `V_θ`**, where an "item" is a group of atoms with no single width. **Naming `s` for a learned multi-atom well is an open modelling question and it gates the transfer of every domain statement in this report.**
- **OQ-D (implicit ≠ trajectory θ-gradient).** R-2's refutation (1093×, sign flips) means C2's trainer cannot assume the implicit path substitutes for the trajectory path. The cheap decisive experiment: measure `pooled/endpoint` θ-gradient ratio on the real store for (i) the read item's atoms and (ii) a neighbour's atoms. If (ii) is ≫1, truncated BPTT is silently deleting the interference gradient — which would retro-explain w20's "free learning erases design" from the *optimiser* side rather than the objective side. **Conjecture, not evidenced.**
- **Risk.** All checks Newtonian, `M = I`, `p₀ = 0`, `T = 0`, single-seed. Constants are kinetic-mode-specific even though the structure is not (C2W1 R2 stands).

---

## Proposed handover updates (for the Hub)

**§1 (physics addendum).**
- **The merge certificate and the admission gate are the same object in the shipped code** (`d_safe := 2s_max + κ′σ_q`). Consequence, measured: any store that violates the certificate has fire rate `f = 1.000` and trips monitor #3 — so the 13-band intersection is confined to the separable regime **by the gate, not by the physics**. Decoupling `d_safe = 0.6·sep` yields **11/12** configs clearing all 13 bands with the certificate violated by 1.5–32.9 % of `sep`, `ρ_ex` up to **6.3×** the C2W1 witness, at a `λ_min` price of **2.2–6.0×** and **no** improvement in the measured dividend (still ≈0 under a settled-point read).
- **`λ_min > 0` does not certify a nonempty basin.** Measured: a genuine minimum (`λ_min = +0.910`) with a measured capture radius of **0.000** under the shipped underdamped read. The certified inradius is **dynamical** (`γ`, barrier height), not spectral.
- **Truncated BPTT is parameter-selective.** `k* = ln(1/ε)/ln(1/ρ)` governs `∂q_N/∂θ` only where the fixed-point sensitivity dominates the transient (ratio 64× ⇒ holds; 27 396× ⇒ error 0.448, flat in k). In a K-item store the far-well parameters are the **interference** gradients.
- **Whole-window read-out:** `k*(ε) ≤ W_ψ(ε) + ln(1/ε)/ln(1/ρ)`; `∂L/∂φ` flows only through the **head** (`h* = ln(1/ε)/ln(1/ρ)`), and tail truncation zeroes it **exactly**. The pooled θ-gradient is **not** a scalar multiple of the implicit gradient (measured 1093×, and sign-flipped for a short-tail ψ).

**§7 (known issues / live) — replace the four inherited entries.**
- ⛔ **Retire "Prop D1 is violated (1.5–7.44×)".** `D ≤ U` is a theorem under a **certified** ball; the harness/gym computed `U` from `sep/2` on stores whose sites were **not minima** (all 7 cells with `D/U > 4` have `λ_min ∈ [−1.199, −0.372]`). Corrected proxy: `r_i ≈ min_j[d_ij/2 + ln(A_i/A_j)/(d_ij/s² − 4/d_ij)]`, **valid only for `s/sep ∈ [0.15, 0.30]` and `λ_min > 0`** (14.55× more accurate than `sep/2` there; both fail ≥29 % below `s/sep = 0.15`, where capture is inertial).
- ⚠ **Monitor #6's dead-band is derived and landable:** `eps_loss = max(8u·Y, 1e-3·Y/W)`, `eps_acq = max(8u·Y, 1/(n_probed·W))`, predicate `slope_loss < −eps_loss AND slope_acq ≤ +eps_acq`. On the published gym: 13 trips → 9 (6 artefacts removed, **2 false negatives recovered**), decision invariant over 12.8 orders of `eps`.
- ⚠ **Monitor #3's validity leg is replaced, not repaired:** the C3 first-order calibration test (`ρ_C3 = median(Δ/B) ∈ [1/3,3]`, `P[Δ>3B] ≤ 0.10`), computed from `_c3_check`'s existing output at zero extra cost; INAPPLICABLE below 3 qualifying pairs. The old correlation is rank-faithful only for `d/s ≳ 4`; the gym runs at `d/s ≈ 2`.
- ⚠ **The γ band does not transfer because a γ band is not a portable object.** Use `C = Σ N_p ln(1/ρ_p) ≥ ln(1/tol)`; edges `γ_min = 1−exp(−2ln(1/tol)/N)`, `γ_max = 2κ/(2ln(1/tol)+κ)`. Predicts 4/4 measured band edges. **Erratum:** the C2W1 row-1 intervals mix `tol = 1e-3` (N=400) and `tol ≤ 1e-4` (N=1500).
- ⚠ **`gate_margin` logs the pre-relocation distance** (`d_min_proposed`), so on relocated writes it describes a site that was never written.

**§8/§10 (record).**
- ⭐ **§A2.5 ADJUDICATED: confirmed 7/7 under the shipped rule (and 0 witnesses in a 72-config extension), refuted 11/12 under the soft certificate.** The confirmation is a **property of the grid** (`s_max/σ_q = 1.156` against the threshold `1.287` at which N2 forces the certificate — an 11 % margin), not a theorem.
- ⭐ **The soft-certificate spec is written** (§4.4, SC-1…SC-7) with a declared violation budget `B = 0.33` of `sep`, the price quantified (`λ_min` ÷2.2–6.0, `ρ_ex` ×1.4–6.3), and three runtime legs replacing Prop D1's a-priori guarantee. C2W3's factored store can now be scoped against it.
- ⭐ **Two of the nine reconciliations (R1, R3-gym) have one root cause:** the **well-width-to-spacing ratio**. `sep/2` fails outside `s/sep ∈ [0.15, 0.30]`; the gate-margin↔drift correlation fails below `d/s ≈ 4`. The shipped gym sits outside both. **Any future "the physics is broken on a learned store" claim should first check these two ratios.**
- ⛔ **M14 (guard liveness) is still unexercised and its priority is raised:** §4.3 shows the shipped code already contains the exact pathology M14 exists to detect (a guard driven arithmetically vacuous, `f = 1.000`) — **without any learning**. It should not reach C2W4 untested.
