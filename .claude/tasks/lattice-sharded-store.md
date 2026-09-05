# Task: lattice-sharded-store — the first real N-unit sharded store (Prop L2 made flesh) (w25)

- **Agent:** `experiment-engineer` · **Output:** `.claude/outputs/lattice-sharded-store.md` · **Branch:** `agent/experiment-engineer/lattice-sharded-store`
- **Read first:** `.claude/AGENT_PROTOCOL.md` (§3.2 worktree · **§7 dial declaration**) · `.claude/outputs/lattice-capacity-theory.md` — **§5.1 is your experiment spec and §"What the engineer must build" is your build list; §2.3 is your trap list. Build to them.** · `negative_results.md` **N97/N98** · `controller-mvp.md` (the allocator you extend)

## ⭐ DIAL DECLARATION (protocol §7)
- **Dials:** capacity (R2 revival route ii) + **isolation** (W2 gives a *provable* per-unit non-interference guarantee, not an empirical one).
- **Laundering control:** the monolithic store at the same `K_total` and matched total atoms/geometry — sharding must beat *it*, not a strawman; and the routing half is declared as classical NN indexing up front (never presented as a dynamical result — N89 discipline).
- **Falsifies:** §5.1's 2×2 discriminator — if NEITHER the d=6 nor the d=8 sharded cells fix their walls, additivity is REFUTED and Theorem L1's conditions get re-audited (start at W3/§1.4). Retrieval degrading with N under a *global* allocator falsifies Prop L4 (a second unmodelled cross-shard channel — suspects: audit items #6/#10).
- **Does NOT falsify:** the d=4 K=32 control cell still failing (it is *supposed* to — its wall is geometry-bound below the ceiling; if sharding fixes it too, that is the `K=32N`-unclamped branch, a different and bigger result); per-offered abstention costs under undersized geometry (N91, known).

## Why
Prop L2 is proven on paper and in small numerics: masked writes on disjoint atom groups ARE N independent optimizers — nothing to synchronize. Nobody has built the actual N-unit store. The Head's w24 assessment names this route explicitly: the levers that failed in `write-ceiling-break` all operated **inside one shared atom pool**; a sharded store with **per-unit budgets** is architecturally immune to both measured failure modes (atoms/K starvation, later-item overwrite). This is R2's second revival route, and it runs in parallel with `r2-geometry-revival` (route i) — **coordinate nothing; the routes are independent by design.**

## Item 1 — the build (the theorist's five-point list, verbatim)
1. **`CLULattice(units, edges=(), couplings=())`** as the shard container — **never one wide relativistic CHLU** (§2.3 #6: one global Lorentz factor couples every block, off-block ∂²T = 8.4e-2, block speed falls 11× with hot neighbours; `CLULattice` per-unit T is exactly separable; #8: `*_zeromean` mass centring couples coordinates). Assert `edges == ()` in code.
2. **Localized atom initialisation** per group (~5 lines + a flag) — fixes **N98** (init_scale=1.0 scatter is the measured 1.4% W3 violation). Init group *j*'s atoms in a ball of radius ~`2s` around site *j*.
3. **Global address allocator** in `Controller` — extend the existing spacing test to run **across shards** (`stored_addresses` = the union). The ONLY global object; a registry, not an optimizer.
4. **Routers R2** (`argmin_r V_r(q)`, pre-settle) **and R3** (`argmin_r ‖x_final − q‖`, displacement) + a **top-2 abstention deadband**. ⛔ **Do NOT ship R1** (post-settle energy — measured at-or-below chance, N97).
5. **Per-shard query noise** (§2.3 #10) — `σ_q` fixed per shard, never `σ/√(N·d)`; the fairness trap is pre-registered, do not fall into it.

## Item 2 — the §5.1 2×2 discriminator (the experiment; predictions are the theorist's, adopt as PREREG)
| cell | baseline (w23/w24) | prediction | discriminates |
|---|---|---|---|
| d=6, K=64, **2 shards × 32** | 0.855/0.818 FAIL | **≥0.90 PASS** | additivity `min(K_addr, N·32)` |
| d=4, K=32, 2 × 16 | ~0.83 FAIL (flat over 16× atoms) | **still FAILS ≤0.87** | ⭐ the control — geometry-bound below the ceiling |
| d=8, K=64, 2×32 AND 4×16 | 0.883 FAIL (⚠ now 0.9067 marginal-PASS at 2× atoms, 3 seeds — state this baseline honestly) | both PASS; 4×16 ≥ 2×32 | additivity is in `N·K_ceiling`, not shard size |
| d=8, K=256, 8 × 32 | untested | PASS iff `K_addr(8) ≥ 256` | where geometry takes over (`d_eff<d` may bite — expected, report it) |
**The 2×2 is the result, not any single cell:** both main cells fixed ⇒ ceiling entirely per-dig, `K_total = 32N`; only d=6 fixed ⇒ Theorem L1's `min` law; neither ⇒ **additivity REFUTED** (report plainly; that kills R2-route-ii and is decision-grade). ≥3 seeds; N92 budget adequacy per cell; monolithic laundering line at every `K_total`.

## Item 3 — the read-side check (§5.3, cheap, protects the claim)
Global-allocator sharding at N ≤ 8 retrieves at **parity with the monolithic store** (theorist measured 1.000 vs 1.000); with per-shard (local) allocation it degrades 0.92→0.55. Reproduce both lines in the real build; if parity fails **with** the global allocator, Prop L4 is wrong and there is an unmodelled cross-shard channel — escalate, do not paper over.

## Acceptance
PREREG (adopt §5.1's table as registered predictions + your bands). The build (5 items), the 2×2 with interpretations, the read-parity check, wall-clock of routing vs read (the O(1)-in-depth claim: routing must be evaluable WITHOUT running the dynamics). Tests green (incl. a test asserting the relativistic wide-unit path is refused/warned for shards); config at all sites + `save_config`; `ruff` clean; worktree; echo the DIAL DECLARATION.

## ⚠ Standing traps
- Routing claims: *"the write is additive at zero optimizer cost and the read stays O(1) in depth because a classical O(N) score suffices to route"* — approved wording; "capacity multiplies by sharding" as a dynamical claim is forbidden.
- Heterogeneous per-unit γ breaks conformal symplecticity (§2.3 #11) — uniform γ per rollout in this task.
- `git -C <worktree>` always.
