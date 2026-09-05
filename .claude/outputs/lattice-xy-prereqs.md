# lattice-xy-prereqs — experiment-engineer report

**Task + acceptance criterion:** land the two code prerequisites for `xy-1d-control` — (1) `GatedCoupling` returns the annealed **free energy** (default) with the legacy mean-energy behind a flag + a monotone-force regression test; (2) a U(1)-preserving coupling (`coupling_type="channel_spring"`, conformal init, `so2_invariant` default, warning) — plus `torus_edges(L)`, a scope docstring, and a written blast-radius confirmation; **defaults for non-`so2_invariant` lattices unchanged; full suite green (≥217).**

**Status: done.** All four items landed, 4 atomic commits, full suite **227 passed / 1 skipped** (baseline `d6f8bac`: 217 passed / 1 skipped ⇒ **+10 tests, 0 regressions**). Every quantitative claim below was executed, not asserted.

---

## 1. Flag provenance (mandatory, §5)

| item | value |
|---|---|
| base commit | **`d6f8bac`** (local `main`; unchanged throughout — verified after rebase) |
| branch | `agent/experiment-engineer/lattice-xy-prereqs` @ **`8c675e9`** |
| commits | `7e217bf`, `04e2b86`, `f1a430f`, `8c675e9` |
| env | **main venv reused** (`/Users/user/Desktop/CHLU/.venv`, per §4 w6 lesson — no worktree `uv sync`). **jax 0.9.0**, equinox 0.13.4, numpy 2.4.1, chlu 0.2.4, CPU |
| precision | **float64** (`jax_enable_x64=True`) for all numerics + `tests/test_lattice.py` (set at module import, pre-existing) |
| seeds | tests: `PRNGKey(1,4,5,6,9,11)` as written; init-statistics sweep: seeds `0..199`; `exp-lattice` smoke: `--seed 42` |
| non-default flags in the smoke run | `--quick --skip-training`; `gate_energy_mode="free_energy"` (**new default**), `coupling_type="auto"` (**new default** → `"spring"` for the mlp-potential lattices exp-lattice builds), `proj_init_mode="random"` (legacy default) |
| **not exercised** | no training, no Langevin/FDT, no governor. `langevin_noise` untouched (still `"legacy"` default). None of this task's numbers are training-config-sensitive. |
| worktree | `../CHLU-lattice-xy-prereqs` (isolated from concurrent `fix-pack-6`); removed after verifying the branch ref from the main repo |

---

## 2. What I did

### Item 1 — `GatedCoupling`: mean energy → **free energy** (`7e217bf`)
- `energy_mode: str = eqx.field(static=True, default="free_energy")`.
  - `"free_energy"` → `-width * jax.nn.softplus((threshold - v)/width)`.
  - `"mean_energy"` → the legacy `sigmoid((t-v)/w) * v`, **kept bit-for-bit** (C-9), pinned by a test so it cannot be silently re-defaulted.
- Added `GatedCoupling.occupancy(q_i, q_j)` = `⟨σ⟩ = sigmoid((t-v)/w)` — in free-energy mode this is exactly `dV_wh/dv`, i.e. the transmitted force fraction.
- `build_lattice(gate_energy_mode=...)` + validation; `ExperimentLatticeConfig.gate_energy_mode = "free_energy"`.
- `exp_lattice.wormhole_smoke` now reports the **occupancy** next to the energy (and writes it to the `.npz`). *This is the direct fix for why the bug was invisible: `v3-lattice-build` asserted energy suppression only.*

### Item 2 — U(1)-preserving coupling (`04e2b86`)
- `coupling_type="channel_spring"` — first-class `build_lattice` option wrapping the existing `channel_spring_coupling`, channel = `range(coupling_dim)`.
- `spring_coupling(init_mode="conformal")` — `W_i = W_j = eye(k, d)` at init, **still trainable leaves**.
- `coupling_type="auto"` (new `build_lattice` **and** config default): `channel_spring` iff `potential_type=="so2_invariant"`, else `spring`.
- `UserWarning` when an `so2_invariant` lattice is built with a random-`W` spring, naming the `h₂/|J|` risk.
- `ExperimentLatticeConfig`: `coupling_type "spring" → "auto"`, `+ proj_init_mode="random"`.

### Item 3 — `torus_edges(L)` (`f1a430f`)
- `2L²` bonds, degree 4, index `i = x + L·y`. **No other new lattice code** (`build_lattice` already takes arbitrary edge lists).

### Item 4 — scope docstring (`8c675e9`)
- `chlu/data/two_timescale_orbits.py` module docstring records: independent per-unit phases ⇒ `V_c` unidentifiable; circular-orbit data ⇒ trained `so2_invariant` minimum at the **origin** (`r* < 1e-3`, 3/3 seeds) ⇒ no vacuum ring/coset. **Generator unchanged** (CM-5/CM-10).

---

## 3. How I verified (real output)

### 3.1 The gate — every theorist number reproduced on the real class
`GatedCoupling` with `base = channel_spring_coupling(2,2,κ=1.0)`, `t=1.0, w=0.25`, float64, `jax.grad` through the class:

| quantity | theorist (§6.2) | **measured here** |
|---|---|---|
| legacy force min | `−0.718` | **`−0.718078`** |
| legacy force sign change | `v = 0.8020` | **`v = 0.8020`** (brentq) |
| free-energy `dF/dv` vs `⟨σ⟩` | `2.3e-10` | **`3.33e-16`** (max abs) |
| free-energy force range | `[0,1]` | **`[0.000000, 0.982014]`** |
| free-energy `V` lower bound `−w·ln(1+e^{t/w})` | — | **`−1.0045375`** (attained at `v=0`, to 1e-12); `V` monotone ↑ |

XY-bond harmonics, `v(Δθ)=J(1−cosΔθ)`, `(J,t,w)=(0.1,0.05,0.02)`, `J_n = -(2/N)Σ V cos(nΔθ)`, N=2048/4096:

| bond potential | `J₁` | `J₂` | `J₃` |
|---|---|---|---|
| ungated | `+0.100000` | `−3.5e-18` | `+5.2e-18` |
| **shipped/legacy `mean_energy`** | **`−0.007224`** | `+0.004841` | `+0.007843` |
| **new default `free_energy`** | **`+0.022307`** | `+0.012344` | `+0.004001` |

Matches theorist's `−0.0072 / +0.0048 / +0.0078` and `+0.0223 / +0.0123 / +0.0040` to the quoted digits. **The legacy gate is antiferromagnetic; the fix is ferromagnetic.**

### 3.2 The coupling — U(1) breaking measured on the real `spring_coupling`
2-D FFT of `V_c` on the vacuum torus (`r*=1`, 64×64 grid, `κ=0.05` ⇒ `J_true = 2κr*² = 0.1`):

```
coupling            J           h2/|J|     U(1)break/|J|
conformal     0.100000   0.000e+00     7.729e-18
random (200 seeds): P(J<0) = 0.545, median h2/|J| = 1.007, median U(1)break/|J| = 1.663
```
Theorist (400 inits, `k=2`): `P(J<0)=0.517`, median `h₂/|J|=1.00`, median U(1)-break`/|J|=1.48`. My `P(J<0)` and `h₂/|J|` agree within sampling scatter (200 seeds); U(1)-break is 12% higher — **note my estimator uses `2|F[1,1]|` (the `cos(θ_i+θ_j)` mode amplitude) which need not be the theorist's exact normalisation**, so treat the *level* of that one column as mine, not as a reproduction. The **sign** results (`P(J<0)≈0.5`, `h₂/|J|≈1`) are what P5 rests on and they reproduce.

Conformal init gives `J = 0.100000` **exactly** and `h₂ = 0` **exactly** — i.e. exactly `channel_spring` at step 0 (asserted array-equal in the tests).

### 3.3 Tests
```
tests/test_lattice.py ......................  22 passed   (12 → 22; +10 new)
full suite (branch):   227 passed, 1 skipped in 216.82s
full suite (base d6f8bac, main checkout): 217 collected  ⇒ +10, zero regressions
ruff check chlu/ tests/  → All checks passed!
ruff format --check      → all touched files formatted
```
Each intermediate commit was independently linted **and** `tests/test_lattice.py + tests/test_wormhole.py` run green before committing (28 → 32 → 22-file-local passes).

New tests (all in `tests/test_lattice.py`, sections `(6)/(7)/(8)`):
1. `test_gate_free_energy_force_monotone_and_bounded` — **the force-direction assertion the task demands**: force ≥ 0, ≤ 1, `== ⟨σ⟩` to 1e-12, `V` strictly monotone, bounded below by `−w·ln(1+e^{t/w})`.
2. `test_gate_mean_energy_legacy_force_reverses_sign` — pins the defect (`min −0.718078`, first sign change `v∈(0.79,0.81)`, strictly inside `v < t`) **and** the legacy expression bit-for-bit.
3. `test_gate_on_xy_bond_stays_ferromagnetic` — `J₁ > 0` under the fix, `J₁ < 0` under legacy.
4. `test_build_lattice_gate_energy_mode_flag` — default is `free_energy`; legacy constructible; unknown mode raises.
5. `test_conformal_init_equals_channel_spring_and_is_trainable` — `W` array-equal to `channel_spring`, energies equal to 1e-15, **gradients flow to `W_i`/`W_j`**, `eye(2,4)` on spectators, bad `coupling_dim`/mode raise.
6. `test_coupling_type_auto_defaults_by_potential_type` — **the bit-compat guard** (see §4).
7. `test_channel_spring_lattice_is_exactly_xy_on_the_vacuum_torus` — `|J − 2κr*²| < 1e-12`; `h₂` and the `cos(θ_i+θ_j)` mode `< 1e-14`.
8. `test_so2_with_random_spring_warns_about_u1_breaking` — warns for so2+random-W; **silent** for conformal / channel_spring / auto / any non-so2.
9. `test_torus_edges_degree_four_no_duplicates` — L=3,4,8: `2L²` edges, degree exactly 4, no dups/self-loops; L=2 raises; `allow_double_bonds=True` → 8 bonds / 4 pairs; L=1 raises.
10. `test_torus_lattice_builds_and_steps` — L=3 (N=9, D=18) SO(2) torus builds via `torus_edges`, all couplings are `eye(2,2)`, `H` finite, 5-step rollout finite.

### 3.4 End-to-end experiment smoke (the affected code path)
`python -m chlu.experiments.exp_lattice --quick --skip-training --seed 42`:
```
kappa=0.01  mu_rel^2=0.040000 (pred 0.040000)  sync=159 (pred 157)  n_1/2=1544 (pred 1537)  latch freeze=0.00e+00  Noether err=1.15e-07
kappa=0.1   mu_rel^2=0.400000 (pred 0.400000)  sync=51  (pred 50)   n_1/2=158  (pred 151)   latch freeze=0.00e+00  Noether err=6.22e-08
kappa=0.3   mu_rel^2=1.200000 (pred 1.200000)  sync=29  (pred 29)   n_1/2=55   (pred 48)    latch freeze=0.00e+00  Noether err=4.89e-08
Fitted slopes: sync ∝ kappa^-0.499 (predicted -0.5), n_1/2 ∝ kappa^-0.982 (predicted -1)
N=2 (D=4): ||J^T O J - O|| = 5.40e-08, energy drift = 4.82e-04
N=4 (D=8): ||J^T O J - O|| = 4.87e-08, energy drift = 4.82e-05
[wormhole] aligned: dF on unit 3 = 3.712e-03 (V_wh = -1.004e+00, <sigma> = 0.982);
           far:     V_wh = -4.707e-11, <sigma> = 1.883e-10 (gate closed)
```
Pricing law (`κ^{-0.499}`, `κ^{-0.982}`) is **unchanged** from `v3-lattice-build` — as required, since that lattice uses `channel_spring_coupling` directly and never touches the gate. The wormhole smoke now shows the gate is open (`⟨σ⟩=0.982`) with `V_wh = −1.004` (the free-energy floor, since `v≈0`) and closed at distance (`⟨σ⟩=1.9e-10`).

---

## 4. Blast-radius confirmation (the Hub's audit — **CONFIRMED by grep, not re-derived**)

1. `GatedCoupling` is **constructed in exactly one production site**: `chlu/core/lattice.py:781`, inside `build_lattice`, only for `wormhole_edges`. (Other hits: 4 in `tests/test_lattice.py`, 1 in `tests/test_wormhole.py`.)
2. The **only non-test caller** of `build_lattice(..., wormhole_edges=…)` is `chlu/experiments/exp_lattice.py:322` (`wormhole_smoke`) — a smoke that printed the gate's *energy* and never its *force*.
3. `chlu/experiments/exp_v1_wormhole.py` **does not use this class.** It imports only `channel_spring_coupling` (line 61) and applies its own gate at line 640: `g_smooth = smooth_gate(_zscore(R0), …)`, computed in **numpy** from the relax residual `R0` and then held constant while scaling the coupling. Its force is therefore `g·∇v` with `g` constant w.r.t. the traced `q` ⇒ monotone, attractive, **no pathology**.

> ⇒ **No shipped claim is contaminated. This was a latent trap**, reachable only through `exp-lattice`'s wormhole smoke. V1's wormhole results stand.

**Bit-compat guard (`coupling_type "spring" → "auto"`).** Verified two ways:
- **grep:** every `build_lattice` call in `chlu/` and `tests/` passes `potential_type="mlp"` (exp_lattice ×3, test_lattice ×6). **No caller anywhere builds an `so2_invariant` lattice via `build_lattice` today** — so the new so2 default changes nothing that has ever been run.
- **assertion:** `test_coupling_type_auto_defaults_by_potential_type` builds `auto` and explicit-`"spring"` lattices from the same `PRNGKey(4)` and asserts `W_i`, `W_j` array-equal and `float(H(q,p))` **exactly equal**.
- **Files changed: none of `integrators.py` / `transforms.py` / `chlu_unit.py` / `train*.py`** ⇒ zero overlap with the concurrent `fix-pack-6`.

---

## 5. Deviation from the task spec (one, deliberate)

**`torus_edges(2)` raises instead of returning 8 edges.** The acceptance text asks for "edge count `2L²`, every degree exactly 4, no duplicates/self-loops" at `L = 2, 3, 8`. **At `L=2` those three conditions are mutually inconsistent**: on a 2×2 torus the `+x` and `−x` neighbours of a site are the *same* site, so the periodic lattice is a **multigraph** — `2L² = 8` bonds over only **4 distinct pairs**, and the simple graph has degree **2**, not 4. There is no simple degree-4 2-torus at `L=2`.

Resolution (smallest reasonable assumption, stated): `torus_edges(2)` raises a `ValueError` explaining the degeneracy; `torus_edges(2, allow_double_bonds=True)` returns the physically-honest 8-bond multigraph (each pair coupled twice ⇒ effective `2κ` on that pair). Both branches are tested; `L = 3, 4, 8` satisfy all three conditions exactly. The KT experiment uses `L ∈ {8,16,32}`, so nothing downstream is affected.

---

## 6. Git footprint

- **Branch:** `agent/experiment-engineer/lattice-xy-prereqs` (off local `main` `d6f8bac`). **Not pushed, not merged.**
- **Commits** (rebased onto local `main` — no-op, `main` never moved; `origin/main` never touched, per §7.21):
  - `7e217bf` fix GatedCoupling: return the annealed FREE energy, not the mean energy
  - `04e2b86` add U(1)-preserving lattice couplings (channel_spring + conformal init)
  - `f1a430f` add torus_edges(L): periodic L×L nearest-neighbour edge list
  - `8c675e9` document two_timescale_orbits scope limit
- **Files touched (5):** `chlu/core/lattice.py` (+237/−…), `chlu/config.py` (+17), `chlu/experiments/exp_lattice.py` (+22), `chlu/data/two_timescale_orbits.py` (+20), `tests/test_lattice.py` (+360). Total `636 insertions(+), 20 deletions(-)`.
- **Worktree:** `../CHLU-lattice-xy-prereqs`, created off `d6f8bac`; **branch ref verified from the main repo (`git -C … log main..agent/…` showed all 4 commits) before removal** (§3.2). Worktree removed; `main` checkout clean and still at `d6f8bac`. Sibling `../CHLU-fix-pack-6` untouched.
- **Unresolved conflicts:** none.
- **Process note:** commits were reconstructed stage-by-stage from the base (the three features interleave inside `build_lattice`'s signature/docstring, so hunk-level splitting was unsafe). Each stage was linted + tested before commit, and the final tree was `diff`-verified **byte-identical** to the fully-tested snapshot.

---

## 7. Open questions / follow-ups / risks

1. **`GatedCoupling` free energy carries a `v`-dependent offset of order `−t`.** At `v→0`, `V_wh → −w·ln(1+e^{t/w}) ≈ −1.0045` (defaults), vs `≈0` for the legacy form. `H` is therefore shifted by ≈`−1` per open wormhole edge. This is physically correct (an open bond *lowers* the free energy) and gradients are unaffected, but **any absolute-energy comparison across the gate fix is meaningless** — e.g. `exp_lattice`'s `wormhole_energy_aligned` flipped from `≈+1e-2` to `−1.004`. Flag for anyone diffing old `.npz` files. If the Hub prefers a zero-referenced gate, `F(v) + w·softplus(t/w)` is a one-line change (constant, zero force).
2. **`proj_init_mode` default stays `"random"`.** I did *not* flip the learnable spring's init to conformal, because `coupling_type="auto"` already routes `so2_invariant` lattices to `channel_spring` (untrainable, exactly XY) and flipping the init would change every non-so2 lattice's parameters. If the Hub wants a *trainable* U(1)-symmetric coupling for `so2_invariant` (rather than the frozen `channel_spring`), the switch is `coupling_type="spring", proj_init_mode="conformal"` — that combination is silent (no warning) and tested.
3. **`channel_spring` has no coupling parameters.** So an `so2_invariant` lattice at the new default has *no learnable coupling* — `test_mlp_coupling_and_gradient_flow` only covers `spring`/`mlp`, and rightly so. If any future experiment expects `grads.couplings` to be non-empty on an so2 lattice, it must opt into `spring + conformal`.
4. **P6 (`κ_c ≤ k_r/40`) is not enforced anywhere.** The theorist flags that trained designed-SO(2) checkpoints sit at `κ/k_r ∈ [0.054, 0.109]`, within **1.15–2.3×** of ring collapse (`κ = k_r/8`). `build_lattice` cannot check this (it doesn't know `k_r`). Candidate follow-up: a `κ/k_r` assertion in the XY experiment's setup, not in `lattice.py`.
5. **Not addressed here (owned by `fix-pack-6`):** P1, the blocking `sqrt(0)` FDT NaN gradient in `integrators.py`. `xy-1d-control` still needs it.
6. My U(1)-break *magnitude* (1.663 vs theorist's 1.48) may reflect a different normalisation of the `cos(θ_i+θ_j)` amplitude — see §3.2. Not load-bearing; flagging for the record.

---

## Proposed handover updates (for the Hub)

**§7 — new entries (both defects from `xy-lattice-theory`, now RESOLVED on this branch, not merged):**

> **7.23 [RESOLVED on `agent/experiment-engineer/lattice-xy-prereqs` @ `7e217bf`, unmerged] `GatedCoupling` returned the mean energy, not the free energy.** `sigmoid((t−v)/w)·v` has force `⟨σ⟩ − (v/w)⟨σ⟩(1−⟨σ⟩)`, which **reverses sign at `v = 0.8020`** (defaults `t=1.0, w=0.25`), min **`−0.718078`** — *inside* the nominally-open region. `V_wh` was non-monotone: **the wormhole repelled its own endpoints.** On an XY bond `(J,t,w)=(0.1,0.05,0.02)` it flipped the exchange antiferromagnetic (`J₁ = −0.007224` vs the correct `+0.022307`). **Blast radius confirmed nil:** the class is constructed only in `build_lattice` for `wormhole_edges`; the only non-test caller is `exp_lattice.wormhole_smoke`; **`exp_v1_wormhole` does not use it** (own gate, held constant ⇒ force `g·∇v`, no pathology). **No shipped claim contaminated — latent trap.** Fixed: default `energy_mode="free_energy"` (`−w·softplus((t−v)/w)`, force `= ⟨σ⟩ ∈ [0,1]`, monotone, bounded in `[−w·ln(1+e^{t/w}), 0)`); legacy retained as `"mean_energy"` and **pinned by a test**. `exp_lattice` now reports the gate *occupancy* alongside the energy — v3-lattice-build's energy-only check is exactly why this hid.

> **7.24 [RESOLVED on the same branch @ `04e2b86`, unmerged] the default lattice coupling broke the lattice's global U(1)** (prerequisite **P5**). Random `W ~ N(0, 0.1²)`: `P(J<0) = 0.545`, median `h₂/|J| = 1.007` (200 inits, reproduced on the real class). The `p=2` anisotropy is a **relevant** perturbation at the KT fixed point (`x₂ = 1/2`) ⇒ it would destroy any 2-D memory phase. Fixed: `coupling_type="channel_spring"` (exact XY, `J = 2κr*²`, verified `|J − 2κr*²| < 1e-12`, `h₂ < 1e-14`), `spring_coupling(init_mode="conformal")` (`W = 𝟙₂`, still trainable), and `coupling_type="auto"` = channel_spring for `so2_invariant` / spring otherwise. **Bit-compat asserted:** no `build_lattice` caller in the repo uses `so2_invariant`, and mlp lattices are bit-identical under `auto` (same `W`, exactly equal `H`).

**§3 — config-default changes (both behavior-preserving for every lattice ever run):**
| group | field | old | new |
|---|---|---|---|
| `experiment_lattice` | `coupling_type` | `"spring"` | **`"auto"`** (→ `"spring"` for all current callers) |
| `experiment_lattice` | `proj_init_mode` | — | **`"random"`** (new; legacy behavior) |
| `experiment_lattice` | `gate_energy_mode` | — | **`"free_energy"`** (new; **changes** wormhole-gate physics) |

**§2 — architecture note:** `chlu/core/lattice.py` gains `torus_edges(L)` (periodic `L×L`, degree 4, `2L²` bonds; raises at `L=2` unless `allow_double_bonds=True` — the 2×2 torus is a genuine multigraph) and `GatedCoupling.occupancy()`. `build_lattice` gains `coupling_type ∈ {auto, spring, channel_spring, mlp}`, `proj_init_mode ∈ {random, conformal}`, `gate_energy_mode ∈ {free_energy, mean_energy}`.

**§7 — data-design note (recorded in code):** `chlu/data/two_timescale_orbits.py` now carries the scope limit in its module docstring: independent per-unit phases ⇒ `V_c` unidentifiable; circular-orbit data ⇒ trained `so2_invariant` minimum at the origin (`r* < 1e-3`) ⇒ no vacuum ring. **`exp-lattice`'s banded-vs-uniform smoke says nothing about coupling physics or Goldstone structure.** Generator deliberately unchanged.

**Program status:** **P5 is done.** `xy-1d-control` is now blocked only on **P1** (the `sqrt(0)` FDT NaN gradient in `integrators.py`, owned by `fix-pack-6`) — P2/P3/P4/P6 are config settings, P7 is the analyst's.
