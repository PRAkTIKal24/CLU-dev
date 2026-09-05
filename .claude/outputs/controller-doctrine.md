# controller-doctrine — physics-theorist report

Task + acceptance criterion: turn the 13 measured collapse modes into a controller spec — per mode a runtime-computable monitor invariant, a productive band with provenance, and a restoring verb — plus P1 (band compatibility), P2 (verb completeness), P3 (trigger ordering), P4 (designed/learned split). Falsified by (a) a mode with no runtime-computable invariant or (b) two provably-disjoint bands.
Status: **done.** Falsifier (a) did NOT fire (all 13 modes have a computable invariant); falsifier (b) did NOT fire (**7 of 54 grid configurations satisfy all 13 bands simultaneously while remaining non-degenerate**) — so **staged activation is feasible as specified and no Head ruling is needed**. 9 pre-registered predictions scored: **7 confirmed · 1 refuted-as-registered-but-confirmed-as-physics (P2) · 1 refuted (P7)**. Plus one **self-retraction** of a mid-task claim that the declared grid search killed (§3.2). 11 scripts, 11 result JSONs, all numbers reproducible from `.claude/scratch/controller-doctrine/`.

> ## ⚠ RECONCILIATION LIST — needs a Hub-assigned owner (protocol §5 corollary, first-10-lines rule)
> **R1 — my own C1 spec is mis-stated and it is quoted downstream.** `clu-controller-spec` §2-C2-N2 says *"margin_i ≥ κσ, κ = 5 gives ≈ 99 %."* **`κ = 5` indexes `spacing/σ`, not `margin/σ`.** Measured: `erf(margin/√2σ)` with `margin = spacing/2` reproduces the three C1 data points to **≤ 0.0036**; `margin = 5σ` gives **0.999999**, not 0.99. Correct form: **99 % needs `margin ≥ 2.576 σ` ⇔ `spacing ≥ 5.15 σ`.** Sites: `clu-controller-spec` §2/§4-defaults, `full-clu-harness` provisional row 8, any handover §8 wording. *(Owner: curator + this file supersedes.)*
> **R2 — "mass stores nothing" (N76 / Prop F1, tier A) is a NEWTONIAN-kinetic statement.** Verified exact under Newtonian `T` (rel. dev **0.0** at λ=2, **2.8e-16** at λ=7). Under the relativistic `T` the same gauge move breaks as **O(1/c²)** (log-log slope **−2.030**), giving **9.1e-2** relative trajectory deviation at the paper's `c = 5.0`. Every quoting site needs a kinetic-mode scope clause, and the charter's *"mass as selector"* (intervention §4-Read) is **not** excluded by N76 in relativistic mode. *(Owner: curator; theory follow-up flagged OQ-3.)*
> **R3 — Hub provisional monitor #1 must be REPLACED:** `corr(q*, q_launch) > 0.90` **false-trips on a healthy store** (measured **0.973–0.978** healthy vs **0.993–1.000** unconverged — everything is above the threshold and the gap is **0.015**). *(Owner: engineer, `full-clu-harness`.)*
> **R4 — Hub provisional monitor #9 must be REPLACED:** `|corr(retention,|a|)| > 0.30` **trips at every payload excursion tested** (|corr| **0.55 → 0.99** while the retention *effect size* is **0.047 → 0.467**). A monitor that is always on is a monitor that gets disabled. *(Owner: engineer, `full-clu-harness`.)*
> **R5 — `d_safe = 4.4 s` is not self-consistent as an admission gate.** At the well widths a working store needs, `4.4·s` exceeds the store's own site spacing (measured **1.625 vs 1.050**) ⇒ the store fails its own gate (this is N74's vacuity with the sign flipped). Use the derived merge+margin form **`d_safe = 2·s_max + κ′σ_q`**. *(Owner: engineer.)*

---

## 0. Flag provenance (all results in this report)

| item | value |
|---|---|
| repo | read-only, **`main @ 082d095`**, clean tree, no worktree, **zero tracked-code edits** |
| code | pure numpy 2.x, main venv `/Users/user/Desktop/CHLU/.venv`; **no repo module imported anywhere** (the reach solver is re-implemented from `readout-channel-theory` §1.0's published closed form and independently validated against its 1.06) |
| integrator | line-for-line the shipped damped velocity-Verlet (`p_½ = p − ε/2 ∇V; q ← q + ε ∇_p T; p* = p_½ − ε/2 ∇V; p ← (1−γ)p*`), `common.py` |
| kinetic mode | **Newtonian, M = I** everywhere except S1/S1b, which is the relativistic-vs-Newtonian comparison itself (`c = 5.0`, `m₀ = 1.0`) |
| langevin_noise / temperature | **N/A** — deterministic, `T = 0`, `p₀ = 0` at every read (`p₀ ≠ 0` only in S1/S1b's gauge probes) |
| ε (dt) | 0.05 everywhere |
| γ | S1 0.02 · S2 0.05 (+ the 0.005–0.9 band sweep) · S2b 0.05 · S3/S3b/S3c/S5/S5b **0.20** (chosen from S2b's measured residual table, not tuned to an outcome) · S4B 0.1/0.2 |
| steps | S1 800/400 · S2 4000 · S2b 4000 · S3 800 · S5 900 · S5b 700 |
| potentials | `V = α‖q‖² − Σ A_i exp(−½ (q−c_i)ᵀ S_i⁻¹ (q−c_i))`, `α = 0.05` throughout; `S_i` isotropic except S3/S3b/S3c/S5/S5b where per-item anisotropy is the object under test |
| seeds | S1/S1b `default_rng(0)` · S2/S2b `rng(11)` · S3 `rng(7)` · S3b `rng(21)` · S3c `rng(5)` · S4B `rng(3)`/`rng(31)` · S5 `rng(101)` · S5b `rng(1000+cfg)` — **single seed per cell; every number here is a THEORY check, none is a paper number** |
| N94 | no trained model touched; the only trained-model quantities used are **cited** (`D_fit = 0.910`, `s_fit = 0.320`, `|c| = 0.98`, `sep = 0.849` — `readout-channel-theory`, 600 write steps, shipped default) |
| scripts | `common.py · s1_gauge.py · s1b_gauge_scaling.py · s2_settle.py · s2b_margin.py · s3_dividend.py · s3b_peritem.py · s3c_rotated.py · reach.py · s4_bands.py · s4b_confound.py · s5_witness.py · s5b_search.py` + `s{1,1b,2,2b,3,3b,3c,4,4b,5,5b}_results.json` |
| PREREG | `.claude/outputs/controller-doctrine/PREREG.md`, written **before** any script existed |

## ⭐ DIAL DECLARATION (echoed, protocol §7)
**Dial:** none — doctrine/theory; no performance claim, no benchmark. **Laundering control:** n/a, except that mode #2 *is* the laundering control promoted to a runtime monitor and is treated as the table's most important row. **Falsifies:** a mode with no runtime-computable invariant, or two provably-disjoint bands. **Does NOT falsify:** a narrow band; a band that is only measurable; a monitor that costs a diagnostic pass.

---

## 1. Pre-registered failure list, and the scorecard

### 1.1 Modes I registered as expected to fail criterion (a) — outcome

| mode | registered | outcome |
|---|---|---|
| **#7 mass** | FAIL (correctly) — test-time gauge, not runtime; **+ sub-prediction: gauge is kinetic-mode-dependent** | ✅ **as registered.** Not a runtime monitor. Sub-prediction ✅ **confirmed and quantified** (R2 above): Newtonian exact, relativistic breaks at O(1/c²). |
| **#13 maturity** | FAIL (correctly) — provenance field | ✅ **as registered.** I confirm the Hub's row. A weak runtime *gate* exists (§2, row 13). |
| **#2 settle→arg-min** | PARTIAL FAIL — dominance needs labels; a label-free **necessary** condition exists | ✅ **as registered, and sharpened into Prop D1/D2 (§3.1).** The label-free necessary statistic is the **settle↔arg-min disagreement mass `D`**, and Prop D1 makes it *exactly* necessary: `D = 0 ⇒ dividend ≤ 0`, with equality only if the settle's payload read is exact. |
| **#10 silent knobs** | PARTIAL FAIL — semantic form is a startup sweep; cheap plumbing form catches N19/N58 | ✅ **as registered.** N19 and N58 are both *mis-wired fields never read*; an access-counting config proxy catches both at O(1). |
| all other 9 | PASS | ✅ **all 9 pass**, four of them (#5, #6, #9, #12) via the registered **self-probe diagnostic pass** (label-free: the store knows what it wrote). |

**⇒ Falsifier (a) did not fire.** All 13 modes have an invariant computable from quantities CLU has; 2 of them (#7, #13) are correctly *not* runtime trips, and 2 (#2, #10) are correctly split into a cheap runtime part and an offline part.

### 1.2 Numerical prediction scorecard

| # | registered | measured | verdict |
|---|---|---|---|
| P1 | Newtonian gauge ≤ 1e-12 rel. | **0.0** (λ=2), **2.809e-16** (λ=7); control (0.1 % depth change) 2.03e-5 | ✅ |
| P2 | relativistic gauge breaks > 1e-2 at λ=2 | **3.617e-03** at λ=2/N=800 — *below* the registered bar ⇒ **MISS as registered**; but **9.08e-2** at the same λ with N=400, **0.9497** at λ=7, and the c-scan gives log-log slope **−2.030** ⇒ the mechanism (O(1/c²)) is confirmed decisively | ◐ **refuted as registered, confirmed as physics.** The registered threshold was the wrong instrument: **the endpoint deviation is masked once both trajectories settle into the same minimum** (3.6e-3 at N=800 vs 9.1e-2 at N=400, identical seed/params) → implementation request I-7. |
| P3 | Prop D1 bound, zero violations | bound `A ≤ 1 − coverage` holds **6/6 σ**; with the **measured** per-item certificate, in-ball landing-on-own-item = **1.000000 at every σ**, **0** certificate violations | ✅ |
| P4 | Hub row-1 predicate false-trips (healthy corr > 0.90) | healthy **0.9777 / 0.9730**, unconverged **0.9929 / 0.9996**; separation **0.0152** | ✅ (and worse than registered — the threshold is in the wrong decade) |
| P5 | residual invariant separates by ≥ 2 orders | **8.34 orders** (healthy 9.5e-16 / 2.0e-9 vs unconverged 4.4e-1 / 9.6e-1) | ✅ |
| P6 | erf law with `margin = spacing/2` within 0.02; `κ=5` is a spacing criterion | max deviation **0.0036** (0.984↔0.9876, 0.953↔0.9545, 0.848↔0.8469); `margin = 5σ → 0.999999` | ✅ → **R1** |
| P7 | neighbour distance confounds `corr(retention,|a|)` by ≥ 0.20 | bare `r = −0.8764`, `r(ret, d_nn) = +0.0438`, **partial `r = −0.8794`** ⇒ **no material confound**; standardised OLS: `|a₀|` **−0.4267** vs `d_nn` **+0.0384** | ❌ **REFUTED as registered.** The Hub's row-9 *statistic* is sound; its *predicate* is not (R4). Adversarially co-varying `d_nn` with `|a|` shifts the bare correlation by **+0.1393** — a caveat, not a confound. |
| P8 | a simultaneous-band witness exists with `D > 0` | **7 of 54** grid configurations clear **all 13** invariants at once, all with `D > 0` (`ρ_ex` 0.103–0.127, dividend +0.0000…+0.0033) | ✅ **P1's answer is: the intersection is NON-EMPTY** |
| P9 | `a_U(K)` monotone decreasing; `a_U(64)/a_U(4) < 0.60` | **1.803 / 1.484 / 1.220 / 1.004 / 0.826 / 0.680** at K = 4/8/16/32/64/128; ratio **0.4582** | ✅ (and the solver reproduces N120's shipped `a_U = 1.06` as **1.0040**, 5.3 % low, zero free parameters) |

**6 confirmed · 1 confirmed-as-physics-refuted-as-instrument (P2) · 2 refuted-as-registered (P2, P7).** Both refutations are findings: P2 produced implementation request I-7, P7 *exonerated* a Hub row I expected to break.

---

## 2. THE 13-ROW TABLE (this supersedes the Hub's provisional table)

⚠ **Word usage, stated once.** "**certificate / certified**" in this document means **only** the geometric margin condition of `clu-controller-spec` §2 (`ball(c_i, r_i) ⊂ B_i`). It never means the machine-unlearning sense (CM-22(m): no (ε,δ) is supplied anywhere here). Likewise "deletion" statements below carry their standing scope — **below capacity or under set-function eviction** (N112).

Notation: `q₀` launch (= φ(x)); `q*` settled point; `R_γ` the relaxation map; `𝒜` the codebook of derived addresses `c_i`; `a_i` payload; `s_i, D_i` fitted well width/depth; `σ_q` query-noise scale at the address plane; `sep` min pairwise address spacing; `λ_min,i = λ_min(∇²V(c_i))`; `r_i` = certified basin inradius; **self-probe** = the store re-reading its own written items (label-free).

| # | collapse mode | monitor invariant + **trip predicate** | productive band (provenance) | restoring verb (direction) |
|---|---|---|---|---|
| 1 | overdamping → "the last observation" | **(i)** `ρ_conv = med‖∇V(q*)‖ / ‖∇V(q₀)‖`; **(ii)** `δ = med‖q*−q₀‖ / sep`. **Trip if `ρ_conv > 1e-6` OR `δ < 0.02`.** `corr(q*,q₀)` is REPORTED, never a trip | `γ·N` such that `ρ_conv < 1e-6`: **measured band `γ ∈ [0.05, 0.5]` at `N = 400, ε = 0.05`; `γ ∈ [0.02, 0.9]` at `N = 1500`** (S2b table). Derived bracket: `γ ∈ [2ε√λ_min, 2ε√λ_max]` (= [0.03, 0.29] here; the measured optimum is γ ≈ 0.2) | **`anneal`** ↑ steps or ↓ γ toward the bracket; **`retry`** if per-query; **`stop`** if the budget cannot reach `ρ_conv` |
| 2 | **settle → arg-min** (the dividend) | **Label-free (runtime): `D = P_q[argmin_i‖q*−c_i‖ ≠ argmin_i‖q₀−c_i‖]`, `U = P_q[q₀ ∉ ∪ball(c_i, r_i)]`, exploitation `ρ_ex = D/U`. Trip if `ρ_ex < 0.10`.** Labelled (eval-time only): the dividend itself | `D ≤ U` is a **theorem** (Prop D1). Non-degeneracy needs geometric heterogeneity: **`D = 0.000000` exactly for equal-depth symmetric stores at every σ ∈ {0.3,0.6,1.0,1.5}** (derived + verified); `D` rises monotonically with depth heterogeneity (0 → 0.0269 at h = 0.6, σ = 0.6). Witness band `ρ_ex ∈ [0.10, 0.13]` (S5b) | **NO VERB RESTORES IT** — see P2. It is an **escalation**: the fix is a configuration change (heterogeneous/anisotropic wells, non-metric-native query law, trajectory read), not an action on an item |
| 3 | vacuous gate | **(i)** gate fire-rate `f` over the stream; **(ii)** gate *validity* `corr(gate margin, measured post-write drift)`; **(iii)** packing utilisation `n_live / N_pack`. **Trip if `f ∈ {0, 1}` OR (ii) `< 0.3` OR `n_live/N_pack > 0.95`** | `f ∈ (0,1)` strictly. Measured violations: **N74** `spacing 1.4142 vs d_safe 1.10` ⇒ `f = 0`; **N91** `f: 0.20 → 0.97` over K with per-offered capped at `N_pack/K`. `d_safe` band: **`2 s_max + κ′σ_q ≤ d_safe ≤ sep`** (derived; the C1 `4.4 s` form violates the upper bound — R5) | **`admit`** ↓ strictness if `f = 1`; **`expand`** (missing verb, P2) if `n_live/N_pack → 1`; **`stop`** if (ii) fails — a certificate that does not predict drift is not a certificate (N74) |
| 4 | blank controls passing | blank/empty-store read on **the strongest read in use**, scored by **decode** (never `tol`, N110). **Trip if `acc_blank ≥ 1/K + 3·se`.** ⚠ With a trajectory read the blank must be run on the *same* ψ, because the trajectory **contains `q₀ = φ(x)`** | `acc_blank ≈ 1/K`. Measured: N68 blanks **0.992–1.000** on a classification read (violation); N110 decode-blank **0.0312 = 1/32** exactly (in band). Witness: **0.1687 vs chance+3sd 0.1871** | **`stop`** — a blank-passing instrument invalidates every other reading; nothing may be repaired until it is fixed |
| 5 | learned addressing dies | **self-probe basin rate** `acq = P[basin(R_γ(φ(x̂_i))) = i]` over live items. **Trip if `acq < 0.90` (chance = 1/K).** Report `basin` and `strict` separately | `acq > 0.90`. Measured violation w19 **4.2 % = chance**; w26 `readonly` control shows the whole annealed-read gain **is** address acquisition (baseline reproduced to 4 dp). Witness **0.9894** | **`anneal`** (the annealed read is the measured fix, N109) → **`route`**/wormhole → **`place`** (re-derive the address by relaxation + `λ_min > 0`, **never** a critical-point solver — C1 §C4.3) |
| 6 | objective/goal divergence | rolling window: `slope(write-loss)` and `slope(acq)` from #5. **Trip if `slope(loss) < 0` AND `slope(acq) ≤ 0` for W consecutive windows.** Leading indicator: `min_i (sep_i − 2s_i)` trending down while loss falls | sign agreement. Measured violation w25/w26 (write loss → 0, retrieval fails) | **`stop`** (halt the write objective) → **`consolidate`** (re-derive + re-certify) → resume. **No monitor quantity may be added to the objective** (that is how #6 and #8 are *caused*) |
| 7 | mass stores nothing | **`pytest` gauge assertion, not a runtime monitor** — apply `(M,V,p₀) → (λM, λV, λp₀)` and assert **trajectory** (not endpoint) invariance | **Newtonian: exact** (0.0 / 2.8e-16 measured; Prop F1's 6.2e-16). **Relativistic: NOT a gauge** — breaks as O(1/c²), **9.1e-2 at c = 5, λ = 2** (derived + measured, this task) | n/a. Consequence for the controller: **mass may not be used as an information channel in Newtonian mode** (it is exactly gauge); **in relativistic mode it may be, and N76 does not forbid it** |
| 8 | learning erases design | **N1** injectivity of item→site · **N2** `sep/σ_q ≥ 5.15` (⇔ 99 % by the erf law) · **N3** `λ_min,i > 0` at every site · **N4** `min_{i≠j}‖a_i−a_j‖ ≥ 2 δ_read`, **`δ_read` measured CONDITIONED ON CORRECT BASIN**. **Trip on any one.** | N2: `acc ≈ erf(margin/√2σ)`, verified to **0.0036** (R1). N3 `λ_min > 0` — measured 12.45 in the witness. N4 ratio **5.27** in the witness; `δ_read` **0.0997–0.1394** measured. Cheap proxy for the inradius: `r_i ≈ sep/2` **over-claims by up to 4.8 %** at depth ratio 2.155 (measured) | **`place`** (relocate/re-derive) ↑ margin · **`admit`** ↓ (refuse) · **`evict`** last resort. ⚠ On a **global-support learned write** the spacing certificate certifies nothing (N74): then the only honest verb is **`stop`**+report, and the fix is the write operator (masked 70×, anchored 103×) — a design choice, not a verb |
| 9 | payload-dependent lifetimes | **effect size**, not correlation: `Δ_ret = max_i ret_i − min_i ret_i` grouped by `‖a_i‖`, from the #5 self-probe pass. **Trip if `Δ_ret > 0.10`.** `corr(ret, ‖a‖)` reported as a direction indicator only | Measured curve (this task, excursion scale → spread/corr): **0.15→0.047/−0.55 · 0.25→0.053/−0.60 · 0.35→0.057/−0.67 · 0.45→0.070/−0.72 · 0.55→0.123/−0.99 · 0.70→0.467/−0.92.** ⇒ band **excursion ≲ 0.45·R in this toy** (R=1.5, s=0.20–0.27, K=6) under the `q₂(0)=0` launch guard — **the exponent/knee is toy-specific; the shape (flat then knee) is the transferable part.** Under C1W27's option-(d) gated stiffness the band is measured **payload-independent at every amplitude** (N119) | **`decay`** — recalibrate per-item `leak_i` so the *half-life* is the dial (Advisor-2 option (c)); ⚠ **no verb restores it at large excursion**; that needs C1W27's potential change, which C2 must **not** build (task constraint) |
| 10 | degenerate axes / silent knobs | **two tiers. (a) plumbing (O(1)):** every config field wrapped in an access-counting proxy; **trip at startup if a declared knob is never read.** **(b) semantic (startup sweep):** each knob perturbed ±; **trip if no declared observable moves by > 3·noise** | (a) catches **N19, N58, N20** exactly — all three are *mis-wired fields that are never read*. (b) catches the read-but-inert class (`sleep_temperature` at `γ=0`: `σ ∝ √(γT)` ⇒ bit-identical output; read-mode axis dead at `clu_steps=1`) | **`stop`** at startup. A dead axis makes every band statement about that axis vacuous |
| 11 | reach failure | **the saddle criterion on `L = √(‖c_i‖² + a_i²)`.** Per item: `β_i = D_i/(2αs_i²)`; capture iff `L_i/s_i < κ_stat(β_i)` **or** `‖a_i‖ < R₂` (middle root of `h(v) ≡ v(1+βe^{−v²/2}) = L/s`). **Trip at write time on any item with margin `a_U − ‖a_i‖ ≤ 0`** | Zero free parameters. Independent re-implementation reproduces N120's shipped `a_U = 1.06` as **1.0040**. Band closes with K: **`a_U` = 1.803/1.484/1.220/1.004/0.826/0.680 at K = 4…128 (d=4)**, and reach is **logarithmically un-buyable** — restoring the K=32 ceiling at K=64 costs **×9.2 depth** (derived here; consistent with N120's "κ 4→5 needs β ×55") | **`place`** ↓ `‖c_i‖` (move the item inward — measured lever: |c| 0.9→0.3 moves the threshold +9.5 %) · **`anneal`** (the annealed read, N109's measured fix) · **`route`**/wormhole · **`admit`** refuse an unreachable item at write time |
| 12 | starve-and-overwrite | **(a) allocation fairness** `min_i D_i / max_i D_i`; **(b) the C3 ratio at each write:** `‖∇δV(c_i)‖/λ_min,i` on the **k nearest** stored items; **(c) self-probe retention of the OLDEST items after each write.** **Trip if (a) < 0.5, or (b) exceeds the first-order bound by > 2×, or (c) drops > ε per write** | (b) is the transferable law (N74: meas/pred ∈ [0.73, 1.63] over 6 classes and 4.6 decades; C1 gated: median ratio **1.0002**, 100 % within 2×; ungated max drift **8.39**). Witness: (a) **0.615**, (b) **6.4e-12/sep** | **`admit`** refuse · **`place`** (canonical/PGCP, order-independent) · **`expand`** (missing verb) · **`evict`** last. ⚠ Evaluate at the **relaxed fixed point**, not the launch point (N74) |
| 13 | under-trained artefacts | **a provenance field, NOT a trip** (I confirm the Hub). Every reading carries `{epochs, write_steps, wall_clock}`; a **maturity gate** refuses to *promote* any reading below threshold to a band statement | `epochs ≥ 40` (N94: `<40`-epoch diagnostics are not properties of the shipped model). Order parameters anneal monotonically: `q*` spread 9.15e-08 (10 ep) → 0.31–0.67 (150 ep) | n/a — a reporting discipline. The only "verb" is **`stop`** on *promotion*, never on the run |

### 2b. Cost and false-trip mode (the column that keeps a monitor alive)

| # | cost | **false-trip mode** (the benign situation that fires it) |
|---|---|---|
| 1 | 1 extra `∇V` per query (free — already computed) | a store whose items sit at very flat minima has small `‖∇V(q₀)‖` ⇒ `ρ_conv` noisy. Guard: floor the denominator; require `‖∇V(q₀)‖ > 10·machine-ε·scale` |
| 2 | `O(Kd)` arg-min per query (negligible). Labelled form = a **second full read pass** | **the query distribution is genuinely tight.** If φ is excellent, all mass lands inside the certified balls, `U → 0`, `ρ_ex` is 0/0. Guard: report `U`; declare the monitor **inapplicable** (not tripped) when `U < 0.01` |
| 3 | free (admission log) | **a stream of genuinely well-separated proposals** gives `f = 0` legitimately. Guard: require the *validity* leg (ii) before acting on the fire-rate leg |
| 4 | 1 settle per probe batch (run at ~1 % sampling ⇒ 0.01× read cost) | a task whose **marginal label distribution is skewed** makes "chance" ≠ 1/K. Guard: compute chance from the empirical marginal, not 1/K |
| 5 | 1 diagnostic pass: `K × n_probe` settles per consolidation | **an item admitted-then-decayed on purpose** self-probes badly and is correct behaviour. Guard: exclude items whose `A_i` is below the decay floor |
| 6 | 2 scalars/step + #5's pass | **a curriculum change** (harder items introduced) legitimately flattens retrieval while loss falls. Guard: the W-window + the leading indicator `min(sep−2s)`, which is curriculum-blind |
| 7 | `pytest` only, ~seconds | **endpoint-only comparison passes vacuously** whenever both trajectories settle into the same minimum (measured: 9.1e-2 → 3.6e-3 purely by doubling N). Guard: compare the **whole trajectory** (I-7) |
| 8 | `K` Hessians (`K d²` V-evals) + `K` inradius bisections (~64 dirs × 12 settles each — **the expensive one**) | **a deliberate register/coset** shares a site by design ⇒ N1 fires. Guard: N1 exempts declared register members with coset separation ≥ max(4σ_write, ℓ_θ). Also: use the cheap proxy `sep/2` per step and the true inradius only at consolidation |
| 9 | free given #5 | **a store holding items of genuinely different importance** (deliberately different `leak_i`) shows retention spread that is intended. Guard: group by `‖a‖` **within** a single declared `leak` cohort |
| 10 | (a) O(1) counters; (b) 1 run per knob | a knob that is **live but currently at a no-op value** (e.g. `γ_φ` with no trash region declared) is not dead. Guard: (a) trips only on *never read*; (b) trips only for knobs declared "active this run" |
| 11 | closed form: fit `(D_i, s_i)` = 64 dirs × 40 radii V-evals per item; or the flow test = 4000 `∇V` per item | **the criterion is single-well** (N120's own scope): it has no neighbour term and mis-classifies crowded high-K cells as reachable. Guard: pair it with #12(b); do not use (U) alone above 80 % packing utilisation |
| 12 | (b) `k` nearest items × (1 `∇V` + cached `λ_min`) per write | **an intentional overwrite/update of an existing item** looks like starvation. Guard: exempt writes whose target id is already live |
| 13 | free | n/a (not a trip) |

---

## 3. P1 — the compatibility question

### 3.1 The load-bearing proposition (the one the whole campaign turns on)

**Prop D1 (agreement inside the certificate).** Let `c_i` be the recorded addresses, `B_i` the `R_γ`-basin of item `i`, `Vor_i` the Euclidean Voronoi cell of `c_i` in `𝒜`, and let the store carry monitor #8's certificate `ball(c_i, r_i) ⊂ B_i` with `r_i ≤ ½ min_{j≠i}‖c_i − c_j‖`. Then for every query `q ∈ ball(c_i, r_i)`: `settle(q) = i = argmin(q)`. Consequently the settle↔arg-min disagreement mass obeys
`D ≤ U ≡ P_q[q ∉ ∪_i ball(c_i, r_i)]`.
*Proof.* `ball(c_i,r_i) ⊂ B_i` gives `settle(q)=i`; `r_i ≤ ½·min-spacing` gives `ball(c_i,r_i) ⊂ Vor_i`, so `argmin(q)=i`. ∎
**Status: proven** (elementary), **verified**: 6/6 σ values, in-ball landing-on-own-item **1.000000** at every σ, zero certificate violations (`s2b_margin.py`).

**Prop D2 (the dividend ceiling, and why w26's 6/6 was structural).** On the agreement set `{settle = argmin}` the same-keys launder returns the **exactly stored** payload while the CLU read returns a value with error `≥ 0`. Hence
`dividend ≤ D · (max per-query gain) − (1−D) · (read-error cost)`,
and in particular **`D = 0 ⇒ dividend ≤ 0`**, with equality only if the settle's payload read is exact.
**Corollary D2a (symmetry).** If the wells are identical (`A_i ≡ A`, `S_i ≡ S`) and the site configuration has a symmetry group whose reflections are the perpendicular bisectors (e.g. a regular ring), then those bisectors are invariant sets that separate the basins, so `B = Vor` and **`D = 0` exactly**.
*Verified:* equal-depth regular ring, `D = **0.000000**` at σ = 0.3, 0.6, 1.0, 1.5 (`s3_dividend.py`). *Verified converse:* `D` rises monotonically with depth heterogeneity `h`: **0 → 0.00125** (σ=0.3) and **0 → 0.02687** (σ=0.6) at h = 0…0.6.
**⇒ N114's "the same-keys launder beats CLU 6/6" is predicted, not accidental**: an engineered-separable, homogeneous, settled-point-only store has `D = 0` and therefore a structurally non-positive dividend. **Label: proven (D1, D2, D2a) + verified.**

**Prop D3 (existence — the dividend is not structurally zero).** If the query law is anisotropic and the wells' iso-potential contours match that anisotropy, the basin partition approximates the Bayes partition better than the Euclidean Voronoi partition does, and the settle strictly beats the same-keys arg-min launder.
*Verified, three cells, each with an exact null control:*

| cell | settle | same-keys (Euclidean) launder | best **shared**-metric launder | **per-item**-metric launder | isotropic-well **control** |
|---|---|---|---|---|---|
| global 4:1 anisotropy (`s3`) | **0.6776** | 0.5828 (**+0.0948**) | 0.7339 (−0.0562) | — | **+0.0000** (0.5828 = 0.5828) |
| per-item ratios (`s3b`) | **0.9686** | 0.9458 (**+0.0229**) | 0.9697 (−0.0010) | 0.9783 (−0.0097, +16 floats) | **+0.0000** |
| per-item **orientation** (`s3c`) | **0.9782** | 0.9527 (**+0.0255**) | 0.9531 (**+0.0251**, +3 floats) | 0.9905 (−0.0124, **+24 floats**) | **+0.0000** (0.952675 = 0.952675) |

**Reading (honest, both halves).** ⭐ In the per-item-orientation cell the settle beats **both** the same-keys launder **and** the strongest launder that costs only a shared `O(d²)` metric, by **+0.0255 / +0.0251** at **zero extra stored bytes**, with the isotropic-well control at **exactly +0.0000** — so the gain is attributable to the well *shape* and nothing else. ⛔ The launder that wins is the one **given per-item covariances** (+24 floats here) — i.e. **not matched-bytes**. ⇒ *"a settled-point read can strictly beat arg-min over the same keys when the landscape encodes an item-dependent metric the codebook does not store."* **Label: constructed existence proof (evidenced, toy scale, single seed). It does NOT show a learned `V_θ` will find such a landscape** — but N120 measured that the shipped learned write already does something of exactly this type (`corr(s_fit, |a_i|) = +0.821`; far-payload wells 1.31× wider; a constant-width surrogate reproduces only 5/11 cells).
⚠ **Fairness note for the Hub/engineer:** because D3's gain is metric-shaped, the harness should report the dividend against **two** launders — the charter's Euclidean same-keys null **and** a best-shared-metric arg-min — so a future positive dividend cannot be dismissed as "you beat a weaker metric".

### 3.2 Pairwise tensions — every candidate, adjudicated

| pair | verdict | evidence |
|---|---|---|
| **#2 non-separability × #8 design-preservation** (registered as the hard one) | **NON-EMPTY, but quantitatively coupled — this is the campaign's central trade** | Prop D1: the certificate caps `D` at the uncovered mass `U`. Both hold simultaneously iff `sep/σ_q ≥ 5.15` (99 % grade) **and** the query law puts mass outside the balls. Witness: `sep/σ_q = 5.42`, `U = 0.042`, `D = 0.0053`, `ρ_ex = 0.127`. **Joint band exists and is narrow.** |
| **#1 damping × #11 reach** (registered as an expected tension) | **NOT a tension — the registered expectation was wrong.** Reach is a property of `∇V` (the saddle at `R₂`); γ does not enter (★). The real γ trade is #1 vs *convergence*, and it is one interval | S2b γ×N table: `ρ_conv` in band over `γ ∈ [0.05, 0.5]` at N=400 and `[0.02, 0.9]` at N=1500; optimum γ ≈ 0.2 vs derived critical-damping bracket [0.03, 0.29] |
| **#3 admission × #12 starvation** | **EMPTY at fixed address-space volume once `K > N_pack(R,d)`; non-empty iff the address space can grow or shard** | N91: fixed R=2 disk, `N_pack = 6.12`, per-offered capped at `N_pack/K` = 0.081 at K=64; sized `R = 0.808√K` → per-offered **0.669 > gru 0.57**. ⇒ **produces a missing verb (`expand`), P2** |
| **#9 lifetimes × #2 basin interaction** | **NON-EMPTY; the registered confound does not exist** (P7 refuted). But **#9's *predicate* has an empty band** — see R4 | partial `r = −0.8794` ≈ bare `−0.8764`; OLS `d_nn` coefficient +0.038 vs `|a₀|` −0.427. Predicate: `|corr| > 0.30` fires at **every** excursion (0.55…0.99) |
| **#11 reach × #3 admission (merge)** | **NON-EMPTY but the tightest pair; it closes with K** | two-sided window `‖a‖max ≤ a_U(s,D,α,‖c‖)` and `2s + c_jσ_q ≤ sep`. `a_U(K)` falls 1.803 → 0.680 over K = 4 → 128; shipped d=4 margin is **6 %** (N120). Restoring reach at K=64 by depth alone costs **×9.2** |
| **#8-N4 decodability × #9 lifetimes** | ✅ **NOT a tension — ALIGNED.** ⚠ *I claimed the opposite from a hand-tuning path and the grid refuted it; the retraction is the finding.* Both improve monotonically as the payload excursion shrinks | m=1, R=1.3, σ_q=0.24, excursion 0.24/0.33/0.45: N4 ratio **5.266 / 2.397 / 1.843** and `Δ_ret` **0.024 / 0.024 / 0.032** — both best at the smallest excursion. `δ_read` grows **superlinearly** with the excursion (0.0091 → 0.0997 for a 1.9× excursion increase), so N4 is *read-precision*-bound, not codebook-spacing-bound. ⛔ I therefore **withdraw** the inference that the monitor bands re-derive the multi-channel code: m=2 is better than m=1 in only some cells (e.g. R=1.8, pay 0.24: 2.090 vs 1.524) and worse in others (R=1.3: 3.304 vs 5.266) |
| **#2 exploitation × {#8-N4, #9} on the QUERY-NOISE axis `σ_q`** (⭐ **NOT on the task's candidate list — found here, and it is the real one**) | **NON-EMPTY but a narrow two-sided window in `σ_q`** | `σ_q` = 0.24/0.28/0.32 at m=1, R=1.3, excursion 0.24: `D` **0.0053 / 0.0117 / 0.0167** (rises), N4 ratio **5.266 / 2.382 / 1.461** (falls), `Δ_ret` **0.024 / 0.048 / 0.072** (rises). ⇒ #2 wants **more** query noise, #8-N4 and #9 want **less**. Bounded below by Prop D1 (`σ_q → 0 ⇒ U → 0 ⇒ D → 0`) and above by decodability. **All 7 clear grid cells sit at the grid's lowest `σ_q` = 0.24** with `ρ_ex` only 0.103–0.127 against a 0.10 bar ⇒ the window is real but tight, and the grid does not bracket its lower edge |
| #4 blank × #2 trajectory read (⭐ found here) | **NON-EMPTY but #4 becomes load-bearing the moment the trajectory read lands** | `read()` returns the trajectory, which **contains `q₀ = φ(x)`** ⇒ a learned ψ over the raw trajectory has direct access to the query embedding and a blank-store ψ-read is exactly "a classifier on φ(x)" — N68's 1e-4 leak at 100 % strength. ⇒ **implementation request I-2 (the trajectory launder)** |

### 3.3 The simultaneous intersection (P1's actual question)

**Answer: NON-EMPTY, and exhibited.** A declared 54-point grid (`m ∈ {1,2}` payload channels × ring radius ∈ {1.3, 1.5, 1.8} × `σ_q` ∈ {0.24, 0.28, 0.32} × excursion ∈ {0.24, 0.33, 0.45}, K = 6) was evaluated against all 13 invariants. **7 / 54 configurations trip nothing**, and every one of them is **non-degenerate** (`D > 0`, `ρ_ex` 0.103–0.127, dividend +0.0000…+0.0033).

Witness (the best cell): `m = 1, R = 1.3, σ_q = 0.24, excursion 0.24, K = 6` —
`ρ_conv = 1.97e-15 · δ = 0.236 · U = 0.0420 · D = 0.00533 · ρ_ex = 0.1270 · dividend +0.0033 · f_gate = 0.983 · blank 0.1687 (chance+3sd 0.1871) · acq 0.9920 · λ_min 12.45 · sep/σ_q 5.417 · N4 ratio 5.266 · Δ_ret 0.0240 · reach margin +0.485 · fairness 0.615.`

**Trip frequency across the grid — the binding constraint is #2:**
`#2 exploitation 38/54 · #3 gate fire-rate 18/54 · #8-N2 18/54 · #8-N4 15/54 · #9 spread 14/54.`
⇒ **Staged activation is feasible as specified (charter §3.1), and the stage that will be hardest to reach is the dividend itself.** This is exactly the right shape of difficulty: the monitors do not fight each other, they fight the *degeneracy*.

⚠ **One caveat that must travel with the "7/54":** **all 7 clear cells sit at the grid's lowest `σ_q` (0.24) and at the two smallest ring radii**, with `ρ_ex` only 0.103–0.127 against a 0.10 bar. The grid **does not bracket the lower edge** of the `σ_q` window (Prop D1 says `D → 0` as `σ_q → 0`, so #2 must trip somewhere below 0.24). The honest statement is *"non-empty, and located at a corner of the searched region"* — not *"comfortably non-empty"*.

**Scope, stated plainly.** This is a `d = 2` address space, K = 6, designed (not learned) wells, one seed per cell, Newtonian, `p₀ = 0`, no φ, no ψ, no training. It proves the 13 bands are **not pairwise or jointly contradictory as specified**. It does **not** prove a learned `V_θ` at CLU dimension can be held in that intersection — that is `full-clu-harness`'s question, and this table is the instrument for answering it.

---

## 4. P2 — verb completeness

**Verdict: {admit, place, evict, decay, route, retry, stop} is INCOMPLETE. Two minimal additions, and one mode that correctly has no verb.**

| mode | restoring verb in the designed set? |
|---|---|
| 1 | partly — `retry` covers "more steps"; **changing the read schedule (γ, widths) is not `retry`** → `anneal` |
| 2 | ⛔ **NONE.** Its restoration is a *configuration* change (heterogeneity, non-metric-native queries, trajectory read), not an action on an item or a query. **Correct handling: escalate, never act.** ⭐ This is load-bearing: a controller *able* to act on #2 would learn to act on it by suppressing the settle — which is w20 at the controller level, and it is precisely how N114 happened |
| 3 | `admit` (recalibrate) + **`expand`** when the binding constraint is the address space (N91), + `stop` when the certificate is invalid (N74) |
| 4 | `stop` ✅ |
| 5 | `route`, `place`, and **`anneal`** (the measured fix, N109) |
| 6 | `stop` + consolidate ✅ |
| 7 | n/a (test) ✅ |
| 8 | `place`, `admit`, `evict` ✅ — **except** under a global-support write, where no verb helps and the fix is the write operator (a design choice) |
| 9 | `decay` (per-item `leak_i` recalibration) ✅ at small excursion; **no verb** at large excursion — needs C1W27's gated stiffness |
| 10 | `stop` ✅ |
| 11 | `place`, `route`, `admit`, **`anneal`** |
| 12 | `admit`, `place`, `evict`, **`expand`** |
| 13 | n/a ✅ |

**The two additions (minimal, and each is measured-load-bearing):**
- **`expand(Δ)` — grow or shard the address space.** Store-level structural op (raise the ball radius, add a shard, or open a fresh dimension). *Why it must be a verb and not a hyperparameter:* N91 measured that with a fixed address space the whole controller is capped at `N_pack/K` (per-offered **0.081**, last of seven), and that sizing the space to the load takes the same controller to **0.669**, beating all four primitives. **The binding constraint was the address space, and there is no verb for it.** Designed guard: `expand` may never *reduce* the space while items are live, and must preserve the placement rule's set-function property (N99/PGCP).
- **`anneal(schedule)` — set the read schedule** (payload/address widths, γ, step budget, boost energies) for a query or a query class. `retry` is the degenerate case (same schedule, more steps). *Why:* the annealed read is the measured mechanism that unclamped `K_learned(4)` 16 → 32 at zero extra bytes/dims/steps, and its `readonly` control localises the entire gain to address acquisition (N109) — i.e. `anneal` is the verb that restores modes #5 and #11. Designed guard: the schedule must **return to the stored landscape** before the value is read (the `static` control reaches basin 0.9993 and reads the **wrong value**, N109).

---

## 5. P3 — trigger ordering

**Principle (not a heuristic): order by TRIP-IMPLICATION.** If monitor `X`'s trip is *implied* by monitor `Y`'s trip, then `Y`'s verb fires first, because acting on `X` treats a symptom whose cause is `Y`. Where no implication exists, the monitors are incomparable and their verbs may fire concurrently (they act on disjoint objects). Two additional axioms are needed because implication alone does not order everything:

- **A1 (instrument before object).** A monitor whose trip invalidates the *semantics of other readings* precedes all of them. This is not a preference: a reading computed on an instrument that passes on a blank store carries **zero** information about the store, so acting on it is acting on noise.
- **A2 (irreversible last).** Deletion is the only irreversible verb; it is **maximal** in the order and additionally gated by a persistence window `W` + hysteresis (C1 §3.C), and may not fire while any class-I monitor is tripped.

**The implication DAG (each edge is measured or proven, not assumed):**
```
 #13 maturity ──▶ {#1,#5,#6,#8,#9,#11,#12}      (N94: sub-40-epoch diagnostics are not properties of the model)
 #10 dead knob ──▶ any band statement about that knob   (N19/N58: the knob was never read)
 #4  blank ─────▶ every performance reading              (N68: blanks 0.992–1.000)
 #7  gauge ─────▶ any claim that mass carries information (Prop F1; scope R2)
 #3  vacuous gate ──▶ #12 starvation ──▶ #8 certificates ──▶ {#5 addressing, #9 lifetimes}
 #11 reach ─────▶ #5 addressing         (N120: "the reach failure is an ADDRESS failure — basin ≡ strict")
 #1  unconverged settle ──▶ #2 dividend  (q* ≈ q₀ ⇒ settle-assign = argmin-assign ⇒ D → 0; verified: A = 0.0000 in both unconverged arms of S2)
```
**The resulting partial order (classes; within a class, incomparable ⇒ concurrent):**

| class | monitors | verbs permitted | why it precedes the next |
|---|---|---|---|
| **I — instrument validity** | #4, #10, #13, #7, and #3's *validity* leg | **`stop` / report only.** No memory-mutating verb may fire | A1 |
| **II — structural integrity** | #3 (fire-rate), #11, #12, #8 | `admit`, `place`, `expand` | II's failures *imply* III's (DAG edges #11→#5, #8→#5/#9, #3→#12→#8) |
| **III — dynamics regime** | #1, #5, #6 | `anneal`, `retry`, `route`, `stop`(training) | III's failure (#1) implies IV's (#2) |
| **IV — policy / economics** | #9, #2 | `decay`; **#2 is report-and-escalate only** | acting on IV before I–III optimises a metric on an uncertified store |
| **V — irreversible** | eviction/deletion | `evict`, `delete` | A2 |

**Two consequences the engineer must implement literally.**
1. **`evict` may not fire in the same step as any class-I trip.** Otherwise the store irreversibly deletes items on the basis of readings known to be invalid.
2. **#2 never fires a verb.** It writes a report and escalates. (If it fired a verb, the only actions available to it would reduce the physics' influence — which is the measured failure it exists to detect.)

**Re-derivation stays constrained (carried from C1, unchanged, and it is an ordering constraint too):** any address re-derivation must be by `γ > 0` **relaxation** with a `λ_min(H) > 0` check, **never** a critical-point solver — a Newton re-derivation wrote a **saddle** into the codebook and the deadband then faithfully preserved it for 150 epochs (`clu-controller-spec` §C4.3/P12). The `λ_min` check therefore precedes the deadband, which precedes the commit.

---

## 6. P4 — the designed / learned boundary

**The formal split.** Let `Θ` be the controller policy's parameters and `𝒢` the set of designed guard predicates. Then:
> **Guards are CONSTRAINTS (a projection on the action), never PENALTIES (a term in an objective).**
> Formally: the policy proposes `u = π_Θ(obs)`; the controller executes `Π_𝒢(u)`, the projection of `u` onto the designed feasible set. `𝒢` does not appear in any loss. The learned part sets **when** (the timing/threshold) and **how hard** (the magnitude), each **box-constrained to a band whose endpoints are designed**.

*Why this exact form.* If a guard were a penalty, the policy could trade it off against reward; with enough reward pressure it will, and the result is a controller that has learned never to fire its guards — w20 ("free learning erases design") reproduced one level up. Under projection this is not merely discouraged, it is **unreachable**: no `Θ` maps to an infeasible action.

| verb | **designed invariant** (hard; the learned policy can never violate it) | **free parameter** (learned: when / how hard) |
|---|---|---|
| `admit` | never admit a site violating the merge certificate `2s + κ′σ_q ≤ sep` **or** the reach certificate `‖a_i‖ < a_U` (#11) | the *utility/priority threshold* at which an item is offered; the refusal rate under budget pressure ∈ [f_min, f_max] |
| `place` | placement must be a **set function of the retained set** (canonical/PGCP), so `delete = set-minus` survives **below capacity / under set-function eviction** (N112 scope); and the committed candidate must satisfy `λ_min(H) > 0` | the priority function — **provided it depends only on the key, never on arrival order** |
| `evict` | irreversible ⇒ requires `W` consecutive trips + hysteresis + dependency-count 0 + no class-I trip; **eviction must itself be a set-function policy** (priority/attribute), **never LRU** (N99) | the eviction score's weights; `W ∈ [W_min, W_max]` |
| `decay` | the amplitude law is exactly `A ← A·e^{−leak}` and decay **commutes with delete** (PGCP Thm 4) | per-item `leak_i` (equivalently: the user sets a half-life, the store solves for `leak_i`) |
| `route` | route on **address geometry** only; **post-settle energy is NOT a routing/confidence signal** (N97) and may not be wired to one | shard/wormhole selection policy; hop budget |
| `retry` | may not exceed the declared compute budget; ladder energies must stay **sub-barrier** (`E < h_i`, C2-N3); may not consume ground truth | confidence threshold τ; ladder depth R |
| `anneal` *(new)* | the schedule **must return to the stored landscape before the value is read** (the `static` control reads the wrong value at basin 0.9993, N109) | the schedule shape (widths, γ, steps) within designed bounds |
| `expand` *(new)* | may not shrink the space while items are live; must preserve the placement rule's set-function property | when to expand; the growth factor ∈ [1, g_max] |
| `stop` | **always available; fires unconditionally on any class-I monitor.** Not a learnable decision | none |

**⭐ The mechanism that prevents relearning w20 at the controller level (two parts, both required).**
1. **No monitor quantity may enter any objective.** (Already the Hub's rule; here it is derived: #6 and #8 are *caused* by optimising against a proxy the objective can satisfy without the goal.)
2. **A guard-liveness meta-monitor, `M14`.** Mode #3 (vacuous gate) applied to the controller itself: on a **canary stream constructed to require intervention**, every guard must fire at least once. **Trip if any guard's firing rate on the canary is 0.** This is the only check that catches "the policy has learned a parameter setting that makes a guard arithmetically unable to fire" — which is exactly N74's failure (`spacing 1.4142` vs `d_safe 1.10` ⇒ gated ≡ ungated to all reported digits) transplanted to a learned policy. **Cost: one canary stream per consolidation.** Without M14, part (1) alone is insufficient: the policy cannot weaken a guard through the loss, but it *can* drive the store into a regime where the guard is vacuous.

---

## 7. Implementation requests (for `experiment-engineer` — I edit no code)

| # | request | why |
|---|---|---|
| **I-1** | **The codebook must retain the derived address `c_i` per live item even when payloads live in `V_θ`.** Without it the same-keys launder cannot be constructed and **monitor #2 has no runtime form at all** | Prop D1/D2; charter §2.1 defines the dividend against "the store's own admitted wells" |
| **I-2** | **Expose the trajectory to ψ in a store-relative form**: alongside the raw strided buffer, provide `traj − q₀` and a flag to mask `q₀`. Add a **trajectory launder** to `eval/dividend.py`: `ψ(traj)` vs `ψ(q₀)` vs `ψ(q₀, q*)` | `read()` returns the trajectory, which **contains `φ(x)`** ⇒ a blank-store ψ-read is a classifier on `φ(x)` (N68 at 100 % strength). Without this the trajectory pillar's first datum is uninterpretable |
| **I-3** | Monitor #1: implement `ρ_conv = med‖∇V(q*)‖/‖∇V(q₀)‖` and `δ = med‖q*−q₀‖/sep`. **Demote `corr(q*,q₀)` to a reported diagnostic** | R3; measured 8.34 orders of separation vs 0.015 |
| **I-4** | Monitor #9: trip on the **effect size** `Δ_ret = max−min retention across `‖a‖` groups`, band ≤ 0.10; report `corr` as direction only | R4; `|corr|` exceeds 0.30 at every excursion tested |
| **I-5** | Monitor #8-N4: compute `δ_read` **conditioned on correct basin assignment** | otherwise N4 double-counts monitor #5's address errors and becomes uncleanable by any read-side action (measured: `δ_read` 0.139 unconditioned vs 0.100 conditioned in the same store) |
| **I-6** | Monitor #2: report `U` (uncovered mass) and `ρ_ex = D/U`, and mark the monitor **INAPPLICABLE (not tripped)** when `U < 0.01` | 0/0; and `U` is the ceiling the dividend must live under |
| **I-7** | Monitor #7 (`pytest`): compare the **whole trajectory**, not the settled point, and **parameterise by `kinetic_mode`**; record the relativistic cell as a *scope*, not a pass | endpoint-only comparison passes vacuously once both runs settle (9.1e-2 → 3.6e-3 by doubling N alone); and the gauge is Newtonian-only (R2) |
| **I-8** | Monitor #10: wrap config in an **access-counting proxy** and fail at startup on any declared-but-never-read field, *before* the expensive semantic sweep | N19, N20, N58 are all "field never read"; the O(1) check catches all three |
| **I-9** | Monitor #3: add the **validity leg** — log `(gate margin, measured post-write C3 drift)` per write and trip if their correlation < 0.3; and log `n_live/N_pack` | N74: on a learned landscape a gate can fire at a healthy rate and certify nothing |
| **I-10** | Add the two verbs: **`expand(Δ)`** (grow/shard the address space, set-function-preserving) and **`anneal(schedule)`** (read schedule; must return to the stored landscape before the value read) | P2; N91 (address space was binding) and N109 (annealed read is the measured fix) |
| **I-11** | Implement **`M14` guard-liveness**: a canary stream per consolidation on which every guard must fire ≥ once | P4; without it the policy can drive the store into a regime where a guard is arithmetically vacuous |
| **I-12** | The dividend report should carry **two launders** — Euclidean same-keys (charter) **and** best-shared-metric arg-min | Prop D3: the dividend mechanism is metric-shaped, so a future positive number must be pre-immunised against "you beat a weaker metric" |
| **I-13** | Replace `d_safe = 4.4 s` with `d_safe = 2 s_max + κ′σ_q`, and assert `d_safe ≤ sep` at construction | R5: at working widths, `4.4 s` exceeds the store's own spacing (1.625 vs 1.050) — the store fails its own gate |
| **I-14** | Declare monitors #9 (large excursion) and #2 as **known-uncleanable-by-any-verb** in the harness's own PREREG | so a trip there is not scored as harness failure. #9's fix is C1W27's gated stiffness; #2's "fix" is a configuration change, and the acceptance criterion is *"does not collapse"*, not *"dividend positive"* |

---

## 8. Diff against the Hub's provisional table

| # | verdict | what changes |
|---|---|---|
| 1 | ⛔ **REPLACE** | `corr(q*,q_launch) > 0.90` false-trips on healthy stores (0.973–0.978). Replace with `ρ_conv` + `δ`; keep corr as a diagnostic. (R3, I-3) |
| 2 | ◐ **SHARPEN (substantially)** | The Hub's inline-launder form needs labels ⇒ not runtime. Add the label-free `D`, `U`, `ρ_ex`; add Prop D1's ceiling; add the **inapplicable-when-`U`-small** state; add the second launder. And **remove any verb**: #2 escalates, never acts. (I-1, I-6, I-12) |
| 3 | ◐ **SHARPEN** | fire-rate alone is necessary-not-sufficient (N74: a gate can fire and certify nothing). Add the validity leg and the packing-utilisation leg; fix `d_safe`. (I-9, I-13) |
| 4 | ✅ **CONFIRM + one addition** | predicate is right; chance must come from the empirical marginal, and the blank must be run through the **same ψ on the same trajectory representation** once the trajectory read lands. (I-2) |
| 5 | ✅ **CONFIRM** | `basin`-vs-`strict` split is correct; specify the source as the **self-probe** pass (label-free) and add `anneal` to the verb list |
| 6 | ◐ **SHARPEN** | make the retrieval leg the self-probe `acq` (label-free), add the W-window, and add the curriculum-blind leading indicator `min(sep − 2s)` |
| 7 | ◐ **SHARPEN (scope)** | Correct that it is a `pytest` gauge. **Add:** compare trajectories not endpoints, and parameterise by `kinetic_mode` — **the gauge is Newtonian-only** (R2, I-7) |
| 8 | ◐ **SHARPEN** | κ = 5 is **spacing/σ**, not margin/σ (R1). N4's `δ_read` must be basin-conditioned (I-5). Add the register/coset exemption to N1. Add the cheap-proxy/expensive-truth split for the inradius (`sep/2` over-claims by ≤ 4.8 % at depth ratio 2.16) |
| 9 | ⛔ **REPLACE** | `|corr| > 0.30` trips at every excursion. Replace with the effect-size predicate; keep corr as direction. Also: **P7 refuted** — no neighbour confound exists, so the Hub's *statistic* was fine and only the *threshold* was wrong. (R4, I-4) |
| 10 | ◐ **SHARPEN** | split into the O(1) plumbing tier (catches N19/N20/N58 exactly) and the expensive semantic tier. (I-8) |
| 11 | ✅ **CONFIRM — strongest row in the table** | Independent re-implementation of (★) reproduces the shipped `a_U = 1.06` as **1.0040** with zero free parameters. **Add:** the `a_U(K)` band table, the ×9.2 depth price at K=64, and N120's own scope limit (single-well ⇒ pair with #12(b) above 80 % utilisation) |
| 12 | ◐ **SHARPEN** | keep the C3 bound; add allocation fairness and oldest-item self-probe retention; restrict the C3 evaluation to the `k` nearest stored items (the write decays as `e^{−d²/2s²}`) and evaluate at the **relaxed fixed point** (N74) |
| 13 | ✅ **CONFIRM** | provenance field, not a trip. Add the maturity **gate on promotion** (a sub-threshold reading may be logged, never promoted to a band statement) |

**Score: 4 confirm · 7 sharpen · 2 replace.** The provisional table is sound in structure; the two replacements are both cases where a *correlation* was used where an *effect size* or a *residual* was needed.

---

## 9. Git footprint
**None.** No tracked code touched; repo read-only at `main @ 082d095`, clean. All artifacts under `.claude/scratch/controller-doctrine/` and `.claude/outputs/controller-doctrine/`.

## 10. Open questions / follow-ups / risks
- **OQ-1 (the composition gap, carried from C1 and unchanged).** Every band here is verified on **designed** wells at `d = 2–3`, K ≤ 8, one seed. Joint sufficiency on a learned `V_θ` at CLU dimension is untested; `full-clu-harness` is the vehicle.
- **OQ-2 (the honest limit of monitor #2).** `ρ_ex > 0` is **necessary** but not sufficient for a positive dividend, and **no label-free statistic can be made sufficient in a metric-native query space**, because there arg-min is Bayes-optimal. Sufficiency requires either labels or a non-metric-native query law — which is exactly intervention §6 criterion 4. A monitor cannot substitute for an admissible benchmark.
- **OQ-3 (⭐ new, and it is an opportunity).** R2 shows the mass gauge is Newtonian-only. **Does the relativistic kinetic term give mass a genuine information channel?** The O(1/c²) breaking is the leading correction; whether it is *usable* (a read whose output depends on `m_i` beyond the gauge) is unmeasured. This is the first mechanism the program has found that could make "mass as selector" (intervention §4) non-vacuous. Cheap first check: does `read(q; m_i)` separate two items at equal well depth under relativistic `T`? ⚠ Note N97 (post-settle energy is not a routing signal) is the nearest negative and it was measured Newtonian.
- **OQ-4.** Prop D3 is a *constructed* existence proof. The open question is **learnability**: does a trained `V_θ` acquire item-dependent well anisotropy aligned with the query law? N120's `corr(s_fit,|a_i|) = +0.821` says the write adapts widths to *payload*; nothing yet says it adapts them to the *query covariance*. This is the sharpest single experiment I can name for Phase B.
- **OQ-5.** The `a_U(K)` band table uses one measured anchor (`sep = 0.849` at d=4 K=32) and N120's `D ∝ s^{1.46}`. It is a **derived-with-one-anchor** curve, not a measured curve; if the anchor moves, the whole column moves proportionally.
- **Risk.** All checks Newtonian, `M = I`, `p₀ = 0`, `T = 0`. The relativistic `T` kinetically couples modes at finite `p` (F5 Prop-2), and R2 shows it also breaks the mass gauge — so the *constants* in every band above are kinetic-mode-specific even though the *structure* is not.

---

## Proposed handover updates (for the Hub)

**§1 (physics addendum).**
- **Prop D1/D2 (the dividend ceiling).** Under monitor #8's margin certificate, the settle and the same-keys arg-min launder **agree on every query inside a certified ball**, so `D ≤ U` and `D = 0 ⇒ dividend ≤ 0`. **Corollary: for equal-depth, equal-width, symmetrically-placed wells `D = 0` exactly (verified 0.000000 at four σ) ⇒ w26's same-keys 6/6 loss is structurally predicted, not accidental.** The dividend lives in `B_i Δ Vor_i`, and that set is created by **geometric heterogeneity**.
- **Prop D3 (existence).** A settled-point-only read can strictly beat arg-min over the same keys when the landscape encodes an **item-dependent metric the codebook does not store**: measured **+0.0255** vs the Euclidean same-keys launder and **+0.0251** vs the best shared-metric launder at **zero extra bytes**, with an isotropic-well control at **exactly +0.0000**. The only launder that wins needs **per-item covariances (+24 floats)** — not matched-bytes. *(Constructed existence proof, toy scale, single seed. Learnability is OQ-4.)*

**§7 (new/current issues).**
- ⚠ **R2 — "mass stores nothing" is Newtonian-only.** Exact under Newtonian `T` (0.0 / 2.8e-16); breaks as **O(1/c²)** under relativistic `T` (log-log slope −2.030), **9.1e-2** relative at the shipped `c = 5.0`, λ = 2. Every N76/Prop-F1 quoting site needs the scope clause; monitor #7 must be parameterised by `kinetic_mode`.
- ⚠ **Endpoint-only invariance tests pass vacuously.** Doubling the step budget alone took the measured gauge violation from **9.1e-2 to 3.6e-3** with identical seed and parameters, purely because both trajectories settled into the same minimum. Any invariance/equivalence assertion in this program must compare trajectories.
- ⚠ **R5 — `d_safe = 4.4 s` is not self-consistent** (measured 1.625 vs a store spacing of 1.050). Use `d_safe = 2 s_max + κ′σ_q` and assert `d_safe ≤ sep`.

**§8 (open directions / record).**
- **R1 correction (mine):** the margin law is `acc ≈ erf(margin/√2σ)`; **`κ = 5` indexes SPACING/σ, not margin/σ.** 99 % needs `margin ≥ 2.576 σ ⇔ spacing ≥ 5.15 σ`. Retire "margin ≥ 5σ ⇒ 99 %" wherever it appears.
- **P1 answered: the 13 productive bands have a non-empty simultaneous intersection** — 7/54 grid configurations trip nothing while remaining non-degenerate (`D > 0`). **Staged activation is feasible as specified.** The binding constraint is **#2 (38/54)**, i.e. the dividend itself, which is the correct shape of difficulty.
- **P2: the verb set is incomplete.** Add **`expand`** (grow/shard the address space — N91 measured this as the binding constraint, 0.081 → 0.669) and **`anneal`** (read schedule — N109's measured unclamping mechanism). **#2 correctly has no verb and must escalate, never act** — a controller able to act on the dividend would learn to suppress the settle.
- **P4: guards are constraints (projection), never penalties**, plus a new meta-monitor **`M14` guard-liveness** on a canary stream. This is the formal mechanism that stops w20 recurring at the controller level; part (1) alone is insufficient because a policy can drive the store into a regime where a guard is *arithmetically vacuous* (N74's failure, learned).
- **New pairwise tension found (not on the task's list): #2 exploitation × {#8-N4 decodability, #9 lifetimes}, on the QUERY-NOISE axis.** `D` rises with `σ_q` while decodability and retention-uniformity fall ⇒ a **narrow two-sided window in `σ_q`**, bounded below by Prop D1 and above by read precision. All 7 clear grid cells sit at the grid's lowest `σ_q`. ⚠ **Self-retraction to record with it:** I first reported #8-N4 × #9 as an opposed pair (from a hand-tuning path) and the declared grid search **refuted it — they are aligned**; the `δ_read`-vs-excursion scaling is superlinear, so N4 is read-precision-bound. The inference that the bands "re-derive the multi-channel code" is **withdrawn**.
- **Claims-matrix candidates:** Prop D2a (`D = 0` exactly for symmetric stores — the structural explanation of N114) and Prop D3 (the matched-bytes existence proof with a `+0.0000` null control) are the two strongest items here. Both are theory-with-numerical-witness at toy scale and must carry that label.
