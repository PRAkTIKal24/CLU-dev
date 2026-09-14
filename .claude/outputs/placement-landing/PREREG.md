# PREREG — placement-landing (experiment-engineer, w26)

Written **before** any measurement harness was run (code not yet written at the time of
writing; only the *geometry* derivations below were computed, from the theorist's
reference `pgcp.hex_cells`, in order to derive the predictions).

Base: local `main` @ `ff85573`. Branch `agent/experiment-engineer/placement-landing`
(worktree `../CHLU-placement-landing`). JAX 0.9.0, main venv (protocol §4).

## 0. Derivation inputs (computed before prediction, stated for audit)

`hex_cells(R, d_safe=1.54)` cell counts (pure geometry, no measurement):

| R | 2.2869 (mia geometry, `radius_for_capacity(8)`) | 2.4012 (=×1.05) | 2.667 | 2.668+ | 6.4685 (`radius_for_capacity(64)`) | 6.7919 (=×1.05) |
|---|---|---|---|---|---|---|
| N_cells | **7** | 7 | 7 | **13** | **61** | **73** |

⇒ the mia store (**8 offers**: 1 target + 7 background) is **at overflow** on its own
geometry (7 cells < 8 items). The theorist's exactness claim is scoped to
**below-capacity or set-function eviction** (§4a/b); P2 (waitlist) is *not* being built.
So the acceptance test is pre-registered in **two cells**, and both are reported.

## 1. Predictions — acceptance test (re-run of `mia-decay-measurement` §2 history column)

Statistic definitions unchanged from `mia_harness.py`; the only change is
`placement="canonical"` + `Controller.delete(target)` replacing the raw `store.evict(slot)`
for the IN-after-removal arm.

**A1 — below capacity (lattice sized so N_cells ≥ 8; R = 2.2869 × 1.35 = 3.087 → 13 cells).**
Derivation: Theorem 2 ⇒ `delete(Store(S)) = Store(S∖{i})` bit-identically, and the
harness's OUT-history world *is* `Store(S∖{target})` (same background keys, same anchors,
same priorities). Bit-identical arrays ⇒ every adversary statistic is tied pairwise.

| quantity | prediction |
|---|---|
| `AUC(z_hole)` history | **0.5000 ± 0.0000** (exact tie) |
| `AUC(n_live)` history | **0.5000 ± 0.0000** |
| `AUC(s1/s2/s4/s5)` history | **0.5000 ± 0.0000** each |
| `TPR@FPR 1 %` (all history stats) | **0.000** exactly (the LiRA llr is identically 0 on tied IN/OUT fits, so nothing exceeds the 99th-percentile threshold). ⚠ The task text says "→ ~0.01"; I register **0.000** and will report both readings. |
| paired-placement column (sanity) | **0.5000 ± 0.0000** on all statistics |
| max abs array difference IN-after-delete vs OUT-history | **0.0** (`tobytes()` equality) |

**A2 — the un-inflated mia geometry (R = 2.2869, 7 cells, 8 offers ⇒ overflow, no waitlist).**
Derivation: with 8 offers into 7 cells the lowest-priority key is refused *and discarded*
(P1). In the IN world the target occupies a cell, so one background item is refused; in the
OUT-history world all 7 background items fit. Deleting the target therefore leaves a
**hole** the counterfactually-refused background item does not return to fill — the exact
§4(b) counterfactual gap.

| quantity | prediction |
|---|---|
| `AUC(z_hole)` history | **≥ 0.85** (leak survives; point estimate 0.95) |
| `AUC(n_live)` history | **≥ 0.95** (6 live vs 7 live is near-deterministic; point 1.00) |
| paired column | still **0.5000** |

⇒ **Registered claim boundary**: exactness holds *below capacity*; at overflow without a
waitlist the deletion claim is NOT defensible, and A2 is the pre-registered demonstration
of that, not a failure of the build.

## 2. Predictions — the rematch cell (theorist §6.4, real two-phase Verlet read)

`controller_line(K=64, arm="on_sized", placement="canonical")`, 3 seeds, `d_safe = 1.54`,
read = shipped `two_phase` (`dt 0.05`, `γ_addr 0.05 × 400` → `γ_read 0.0 × 800`, tail 0.25,
8 subsamples), 16 queries/item, strict criterion (own basin among live sites AND
`|v − a| < payload_tol`).

| quantity | prediction | derivation |
|---|---|---|
| `n_admitted` (mult 1.0) | **61 / 64 exactly, zero variance** | deterministic: admitted = min(K, N_cells) = 61 |
| per-admitted retention | **1.000** (registered band **≥ 0.98**) | theorist H5 = 1.0000 (976/976) under gradient-flow relaxation at exactly this spacing; the open scope gap is that the shipped read is two-phase Verlet |
| per-offered retention | **0.953** (band **[0.935, 0.960]**) | 0.953 × per-admitted |
| `n_admitted` (mult 1.05 → 73 cells) | **64 / 64** | N_cells 73 ≥ K |
| per-offered (mult 1.05) | **1.000** (band ≥ 0.98) | — |
| `min_spacing_live` | **= 1.54** (± 1e-6, float32 store) | lattice invariant |

**Falsifier for the H5 scope gap:** per-admitted < 0.98 under the real read ⇒ the lattice
constant must inflate (cost ≈ m⁻² in cells) and the packing win is partly repaid.

## 3. Predictions — the ported exactness/packing/cascade tests

| # | prediction |
|---|---|
| T1 | n=4, all 24 write orders → **24/24 bit-identical** (`tobytes()`) |
| T2 | n ∈ {8,16,40,64}, random orders → all bit-identical; `min_spacing = d_safe` exactly |
| T3 | write/delete interleavings → bit-identical to a fresh build of the final set |
| T4 | mid-decay delete → survivors bit-identical to the never-written history |
| T5 | write-then-delete = never-written |
| packing | `N_cells(R(K)·1.05) ≥ K` for K ∈ {16,32,64,128} → **19/37/73/139** |
| cascade | full-load (61/61) mean moves/delete **2.84** (band [2.0, 4.0]), max ≤ 15 |
| scrub (D1) | after zeroing `centers`/`payloads` on evict, **max |ΔV| = 0.0 exactly** over random queries (`active` multiplies both terms) — **no physics number moves** |
| LRU guard | `placement="canonical"` + `evict_policy="staleness"` **raises** |

## 4. What would falsify the task's headline claim
- A1 `AUC(z_hole)` does **not** fall to 0.5 ⇒ no deletion-flavoured sentence is defensible
  and the report says so regardless of the number.
- Rematch per-admitted < 0.98 under the real read ⇒ H5 scope gap is real; lattice must inflate.

## 5. Declared *non*-falsifiers (protocol §7)
- delete-time churn ~2.84 moves/delete at full load (pre-measured price);
- LRU/staleness staying outside the claim;
- per-offered 0.953 < 1.000 at mult 1.0 (that is the un-inflated sizing, by design);
- A2's leak at overflow (P2 waitlist explicitly out of scope).
