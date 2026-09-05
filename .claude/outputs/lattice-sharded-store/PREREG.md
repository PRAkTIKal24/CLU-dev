# PREREG — lattice-sharded-store (w25)

**Written 2026-07-25, BEFORE any cell of the discriminator was run.** Base: local
`main @ 63c668d`, branch `agent/experiment-engineer/lattice-sharded-store`,
worktree `../CHLU-shard`, main venv (JAX 0.9.0).

Protocol §5 pre-registration rule: the acceptance criterion is a set of measured
**pass/fail ratios**, so predictions and their derivations are committed here first.

## ⭐ DIAL DECLARATION (echoed from the task file)
- **Dials:** capacity (R2 revival route ii) + **isolation** (W2 = a *provable* per-unit
  non-interference guarantee).
- **Laundering control:** the **monolithic store at the same `K_total`**, matched total
  atoms and identical geometry/rollout/write operator; and the routing half is declared
  **classical NN indexing** up front (N89 discipline) — never a dynamical result.
- **Falsifies:** neither the d=6 nor the d=8 sharded cell fixes its wall ⇒ additivity
  REFUTED. Retrieval degrading with N *under the global allocator* ⇒ Prop L4 wrong.
- **Does NOT falsify:** the d=4 K=32 control still failing (it is supposed to);
  per-offered abstention costs under undersized geometry (N91).

---

## 0. What the numbers mean (definitions fixed before measurement)

* **strict** — as in w22/w23/w24, unchanged: predicted item = (routed shard, nearest
  center *within that shard*); `strict_ok = (predicted item == label) ∧ |read − payload| < 0.1`.
  A routing miss is therefore counted as a **failure**, exactly like a basin miss in the
  monolithic arm. Comparable to the monolithic number by construction.
* **PASS** = mean strict ≥ **0.90** over seeds ∧ value-blank ≤ trivial ceiling (w23 criterion).
* **Headline abstention deadband = 0** (never abstain), so the sharded number is
  per-offered and directly comparable to the monolithic one. The deadband sweep is a
  separate, secondary item (N91 discipline: no abstention credit in the headline).
* **Two atom budgets, both reported** (this is a fork I am registering, not choosing
  post-hoc):
  * `per_shard` — **PRIMARY.** Each shard gets exactly the atom budget a *monolithic*
    store of `K/N` items at that `d` would get (`_atoms_for(dm, K/N, d)`). This is the
    literal instantiation of "N independent stores", and total parameters are ≈`N×`.
    Its laundering line is the **monolithic arm at the SAME total atom count**
    (`monolithic_2x` etc.), because w23 measured that more atoms make the monolithic
    d=6 K=64 cell *worse* (0.855 → 0.809).
  * `total_matched` — the **parameter-matched** control: total atoms equal the
    monolithic budget, split evenly across shards. A win here is strictly stronger
    (a free lunch); a win only under `per_shard` still supports Theorem L1 but must be
    stated as *"additivity costs N× parameters"*.
* Geometry, rollout, write operator, query noise and pass criterion are **inherited
  verbatim** from `experiment_designed_mechanism` (w23). Only the sharding knobs are new.
* **Per-shard query noise:** each shard is its own `d`-dim CLU unit, so the `fixed_norm`
  jitter is `σ_q/√d` with `d` = the SHARD address dimension, never `σ_q/√(N·d)`
  (§2.3 #10 fairness trap — avoided by construction, asserted in a test).

---

## 1. Registered predictions — the §5.1 2×2 discriminator (theorist's, adopted verbatim + my bands)

| # | cell | w23/w24 baseline | **PREDICTION** | band | derivation |
|---|---|---|---|---|---|
| **P1** | d=6, K=64, **2 shards × 32** | 0.855 (w23) / 0.858±0.001 (w24, 3 seeds) FAIL | **PASS, strict ≥ 0.90** | 0.90–0.97 | Theorem L1 `K_total = min(K_addr(6), N·K_ceiling)`. Each dig now carves 32 valleys, at/below the measured per-dig ceiling; w24 measured monolithic d=6 **K=32 = 0.951±0.006**, so a shard *is* a K=32 store and the only new failure channel is cross-shard routing. Band top = the K=32 line (0.951) minus routing loss. |
| **P2** | d=4, K=32, 2 × 16 ⭐ **the control** | ~0.83 (w23), flat over a 16× atom sweep; w24 `baseline_global` d=4 K=32 = **0.801** | **still FAILS, ≤ 0.87** | 0.70–0.87 | d=4's wall (K=16, 0.928 at 3 seeds) is *below* the ~32 per-dig ceiling ⇒ it is `K_addr(4)`-bound, and `K_addr` is **conserved by sharding** (Prop L4: the union must still be discriminated). If this cell passes, geometry never bound anything and the law is `K = 32N` unclamped — a **different and bigger** result, to be reported as such, not as confirmation. |
| **P3a** | d=8, K=64, 2 × 32 | 0.883 (w22/w23); ⚠ **0.9067 ± 0.0068 marginal-PASS at 2× atoms, 3 seeds** (w24 post-hoc dim run) | **PASS ≥ 0.90** | 0.90–0.98 | same as P1; d=8 has far more room (`K_addr(8) ≫ 64`). ⚠ Baseline honesty: the monolithic cell is *already marginal* at 2× atoms, so P3a alone is weak evidence; the informative quantity is **Δ(sharded − monolithic) at matched atoms**, predicted **≥ +0.03**. |
| **P3b** | d=8, K=64, **4 × 16** | — | **PASS, and ≥ 2×32** | 0.92–1.00 | additivity lives in `N·K_ceiling`, not in shard size: 4 digs of 16 items each are *further* below the per-dig ceiling. Falsifier: 4×16 < 2×32 by >0.03 ⇒ shard size matters ⇒ the ceiling is not purely per-dig. |
| **P4** | d=8, K=256, **8 × 32** | untested | **PASS iff `K_addr(8) ≥ 256`**; I predict **FAIL, 0.55–0.85** | — | `d_eff < d` shell concentration (`address-space-dimension-scaling`) pulls `K_addr(8)` below the ball estimate, and the union separation at K=256, R=1, d=8 is ≈0.72 ⇒ ~2.4× the 0.30 atom width, i.e. *inside* the measured 2.4–3.0 transition window. Registered as an **expected-fail probe of where geometry takes over**, per the theorist's own §7.6 warning. |

**The 2×2 is the result, not any single cell.**
* **P1 ∧ P3 pass, P2 fails** ⇒ Theorem L1's `min` law, `K_total = min(K_addr(d), N·32)`. *(my prior: most likely)*
* **P1 ∧ P3 ∧ P2 all pass** ⇒ ceiling entirely per-dig, `K_total = 32N` unclamped (bigger result, must be reported as a surprise, not a confirmation).
* **Neither P1 nor P3 passes** ⇒ **additivity REFUTED**, R2-route-ii dead, Theorem L1's conditions re-audited starting at W3/§1.4. Decision-grade; I will report it plainly.

## 2. Registered prediction — the read-side check (§5.3, Prop L4)

| # | quantity | **PREDICTION** | falsifier |
|---|---|---|---|
| **P5** | designed store, **GLOBAL** allocation, N ∈ {2,4,8} at fixed `K_total`, router R2/R3, deadband 0 | **parity with the monolithic store: \|Δ strict\| ≤ 0.02**, and route accuracy ≥ 0.98 | any monotone degradation with N ⇒ **Prop L4 is wrong**, an unmodelled cross-shard channel exists (suspects: §2.3 audit #6 relativistic non-separability, #10 query-noise scaling) ⇒ ESCALATE, do not paper over |
| **P6** | designed store, **LOCAL** (per-shard) allocation, same grid | **degrades monotonically with N**, reproducing the theorist's 0.92 / 0.73 / 0.62 / 0.55 shape (I predict the *shape*, not the values — his store was an analytic Gaussian one, mine is `BallRegisterPotential` under the real two-phase Verlet read) | no degradation ⇒ the global allocator is not load-bearing and build item 3 is unnecessary |
| **P7** | ⛔ **R1 is not shipped** (post-settle energy, N97). Registered as a *code* prediction: `ROUTERS == ("R2", "R3")` and a test asserts R1 is absent. | — | — |

## 3. Registered prediction — the cost claim (the O(1)-in-depth half)

| # | quantity | **PREDICTION** |
|---|---|---|
| **P8** | wall-clock of an **R2 routing decision** (`argmin_r V_r(q)`, no dynamics) vs **one two-phase read** (1200 Verlet steps) | routing ≤ **5%** of one read at N ≤ 8; theorist's estimate `N/2400` ⇒ ≈0.3% at N=8. I widen to ≤5% for Python/dispatch overhead. |
| **P9** | one **joint lattice rollout** settles all N shards ⇒ read wall-clock is **O(1) in rollout depth** and grows only with total atom count, i.e. sharded-at-matched-total-atoms read time ≈ monolithic read time (**ratio ≤ 1.5×**) | ratio > 2× ⇒ the "no N-fold read cost" claim needs qualifying |

## 4. Registered prediction — the N98 initialisation fix (build item 2)

| # | quantity | **PREDICTION** |
|---|---|---|
| **P10** | localized atom init (group *j*'s atoms in a ball of radius `2s` around site *j*, **address axes only** — the payload axis keeps the w23 scatter so the writer is not handed the answer) at the d=6 K=64 cell, applied identically to BOTH arms | **it helps both arms** (Δ strict ≥ +0.02 monolithic, ≥ +0.01 sharded) because foreign atoms in a well are removed at step 0. ⚠ It is an **initialisation**, therefore a potential N46 fairness hazard, therefore it is **OFF in the headline 2×2** (default `atom_init_local_radius = 0.0`) and reported only as a separate ablation. If it alone lifts the monolithic cell over 0.90, that is a **finding about w23's init, not about sharding**, and it will be reported as such — it would also partially explain N92. |

## 5. Things that will NOT be claimed whatever the numbers say

1. *"Capacity multiplies by sharding"* as a **dynamical** result. Approved wording only:
   **"the write is additive at zero optimizer cost, and the read stays O(1) in depth,
   because a classical O(N) score suffices to route."** The router is nearest-neighbour
   indexing over stored addresses (N89).
2. Nothing about **heterogeneous per-unit γ** (§2.3 #11) — uniform γ per rollout here.
3. No claim that the sharded store beats any external baseline; this is a law about the
   primitive (the w24 Head boundary ruling: R2's capacity law is exempt from the
   demotion **but its figure must never be framed as beating anything**).
4. If `per_shard` wins and `total_matched` does not, the claim is **"additivity costs
   N× parameters"**, not "free capacity".

## 6. Deviations policy
Any deviation from the above (seed counts cut for compute, a cell dropped, a budget
changed) is declared explicitly in the report with the reason, per protocol §6.
