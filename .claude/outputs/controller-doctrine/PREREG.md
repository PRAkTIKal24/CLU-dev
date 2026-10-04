# PREREG — controller-doctrine (physics-theorist, C2W1)

**Written before working through the 13 modes and before any script in
`.claude/scratch/controller-doctrine/` was written or run.**
Repo state at registration: `main @ 082d095`, clean tree, nothing checked out, no worktree.
All checks below are pure numpy (main venv), no repo code imported.

The task's falsifiers are (a) *a mode with no runtime-computable invariant* and (b) *two productive
bands provably disjoint*. Task requires: **pre-register which modes I expect to fail (a).**

---

## R1 — Modes I predict will FAIL criterion (a) (no invariant computable at runtime from quantities CLU has)

| mode | prediction | reason registered in advance |
|---|---|---|
| **#7 mass stores nothing** | **FAIL (correctly) — test-time, not runtime** | it is a gauge identity, not a state-dependent quantity; there is nothing to observe on a stream. I concur with the Hub's row. **Additional registered sub-prediction: the gauge is KINETIC-MODE-DEPENDENT — exact under Newtonian `T`, BROKEN under relativistic `T`** (the `m₀²c⁴` term is not homogeneous under `M→λM`). If true, the Hub's row needs a scope clause. |
| **#13 under-trained artefacts** | **FAIL (correctly) — provenance field** | maturity is a property of the *fit*, not of the stream. I concur with the Hub. Registered sub-prediction: a *weak* runtime form exists (a maturity gate on write-step/epoch counters) and it is a gate, not a trip. |
| **#2 settle → arg-min** | **PARTIAL FAIL** | the *dominance* form (dividend > 0) needs a task metric ⇒ labels ⇒ not available at runtime. I predict a **label-free NECESSARY condition exists** (a settle-induced reassignment rate) and that the sufficient condition remains offline/eval-only. |
| **#10 degenerate axes / silent knobs** | **PARTIAL FAIL** | the semantic form ("knob moves a reported metric") costs a full sweep per knob and is not a stream monitor. I predict a **cheap plumbing form** (config-field read counter) catches N19/N58 exactly, because both were *mis-wired fields that are never read*, and that the semantic form stays a startup/CI sweep. |
| all other 9 modes (#1,#3,#4,#5,#6,#8,#9,#11,#12) | **PASS** | each is expected to reduce to a function of {launch point, trajectory, settled point, ∇V, Hess V, codebook, admission log, write log, self-probe read-back}. #5, #6, #9, #12 are expected to need a **self-probe diagnostic pass** (the store re-reading its own written items — label-free because the store knows what it wrote), not labels. |

**Registered cost prediction:** exactly four monitors (#5, #6, #9, #12) require a diagnostic pass over
live items; the other runtime monitors are O(1) per query on quantities already computed.

## R2 — Pairwise band tensions: predictions before analysis

| pair | registered prediction |
|---|---|
| **#2 non-separability × #8 design-preservation (margin certificate)** | **the hard pair.** Registered prediction: **NON-EMPTY but quantitatively coupled** — I predict a provable ceiling of the form *"the settle and the same-keys launder agree on every query landing inside the certified margin ball, hence the achievable dividend is bounded by the query mass outside those balls."* If instead the certificate provably covers all admissible query mass, the pair is **EMPTY** and that is the wave's headline. |
| **#1 damping × #11 reach** | registered prediction: **NOT a tension — aligned.** Reach is a property of `∇V` geometry (saddle at `R₂`), and inertia (lower γ) can only help barrier crossing. The real γ tension is #1 (settle must not be a no-op) vs *convergence* (settle must finish in budget), which is a γ×N band, non-empty. |
| **#3 admission strictness × #12 starvation** | registered prediction: **EMPTY at fixed address-space volume once `K > N_pack(R,d)`; non-empty iff the address space can grow or shard** (N91: the address space, not the controller, was binding; sized geometry `R = 0.808√K` beat all four primitives). ⇒ I expect this to produce a **missing verb**. |
| **#9 lifetimes × #2 basin interaction** | registered prediction: **non-empty band but a MONITOR CONFOUND** — basin interaction supplies a second, neighbour-distance channel into retention, so the Hub's `\|corr(retention,\|a_i\|)\| > 0.30` trip will **false-trip** under exactly the configuration C2 wants. Predicted fix: partial correlation controlling for nearest-neighbour distance. |
| **#11 reach × #3 admission (merge)** | registered prediction: **non-empty but the TIGHTEST pair in the table** — the two-sided window `\|a\|max ≤ a_U(s,D,α,\|c\|)` and `2s + cσ_q ≤ sep` closes as K grows because `sep` falls; at the shipped d=4 point the margin is 6% (`a_U = 1.06` vs `\|a\|max = 1.0`, N120). |

## R3 — Numbered quantitative predictions (each scored in the report)

| # | prediction | falsified if |
|---|---|---|
| **P1** | Newtonian gauge `(M,V,p₀) → (λM, λV, λp₀)` leaves the damped-Verlet trajectory invariant to **≤ 1e-12 relative** (Prop F1 measured 6.2e-16) | relative deviation > 1e-10 |
| **P2** | The **same transformation under relativistic `T`** changes the trajectory by a **relative amount > 1e-2** at λ = 2 (i.e. the gauge is Newtonian-only) | relativistic deviation ≤ 1e-6 ⇒ my sub-prediction in R1 is wrong and the Hub's row needs no scope clause |
| **P3** | **Prop D1 (dividend ceiling).** With pairwise spacing ≥ 2·margin and every stored address certified at radius `r = κσ`, the settle-vs-argmin **reassignment rate `A` satisfies `A ≤ 1 − (query mass inside ∪ball(c_i, r))`**, with **zero violations** over a sweep of σ | any query inside a certified ball where settle and argmin disagree (i.e. one counterexample kills the proposition as stated) |
| **P4** | **The Hub's row-1 trip predicate false-trips.** In a *healthy, working* store (settle strictly denoises, reassignment rate `A > 0.1`) `corr(q*, q_launch) > 0.90` | corr stays ≤ 0.90 in the healthy store ⇒ the Hub's predicate is safe and I confirm rather than replace row 1 |
| **P5** | A **convergence-residual** invariant `‖∇V(q*)‖ / ‖∇V(q_launch)‖` separates *overdamped/unconverged* from *healthy* by **≥ 2 orders of magnitude**, where `corr(q*,q_launch)` separates them by < 0.15 in absolute value | residual separation < 10× |
| **P6** | The **erf margin law** `acc ≈ erf(margin/(√2σ))` reproduces the four C1-measured points (σ/spacing = 0.10/0.20/0.25/0.35 → 0.984/0.953/0.848 at the last three) to **≤ 0.02** *when margin is read as spacing/2* — and therefore that **C1's "κ = 5 ⇒ 99%" is a SPACING criterion, not a margin criterion** (margin = 5σ gives 0.999999, not 0.99) | the law with margin = spacing/2 misses the measured points by > 0.02, or margin = κσ = 5σ reproduces 0.99 |
| **P7** | **Neighbour confound for #9:** at *fixed* payload `\|a\|`, retention of a decayed item varies with nearest-neighbour distance by **≥ 0.2 in absolute retention** across an admissible spacing range ⇒ a bare `corr(retention, \|a\|)` is confounded | variation < 0.05 ⇒ no confound, Hub's row-9 predicate confirmed |
| **P8** | **The simultaneous-intersection witness exists:** a single toy configuration in which **all 13 invariants read inside band at once** AND the store is **non-degenerate** (`A > 0`, i.e. basins interact and the settle is not arg-min) | no witness found after the registered search ⇒ report as *not exhibited* (⚠ this would be evidence for, not proof of, an empty intersection) |
| **P9** | **Reach×merge window closes with K:** the admissible payload excursion `a_U` scales like `sep ∝ K^{−1/d}` through `s`, so `a_U(K)` is **monotonically decreasing in K** over the registered sweep, with `a_U(K=64)/a_U(K=4) < 0.6` at d=4 | non-monotone, or ratio ≥ 0.6 |

## R4 — What I will NOT do
- No repo code imported except where a check is *about* shipped behaviour (none is planned; the reach
  criterion is re-implemented from `readout-channel-theory` §1.0's published closed form).
- No re-verification of what `clu-controller-spec` already verified (IFT/`∂q*/∂θ`, the contraction
  `2.2e-12`, the gauge permutation spread 0.0, the deadband, the admission gate). Those are cited.
- No performance claim, no benchmark, no band promoted from "measured at wave n" to "derived".
