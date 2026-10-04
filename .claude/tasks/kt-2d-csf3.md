# Task: kt-2d-csf3 — the 2-D Kosterlitz–Thouless memory phase (w16, analyst; the NMI physics flagship)

- **Agent:** `results-analyst` · **Output:** `.claude/outputs/kt-2d-csf3.md`
- **Read first:** protocol (**§5 pre-registration MANDATORY — this is a measured universal-jump / exponent claim**) · **`.claude/outputs/xy-1d-control.md`** (the GO verdict + the *exact* prerequisites it validated on the real path + the winding-τ∝1/N 1-D null it recommends running first) · **`.claude/outputs/xy-lattice-theory.md` §7** (the full experiment spec: topology, sizes, observables, nulls, kill criterion) · `.claude/claims_matrix.md` (CM-16b, CM-17 — Newtonian only) · `chlu/core/lattice.py` (`torus_edges`, `channel_spring`) · `scripts/csf3/README.md`.
- **This is unblocked and needs NO scorer** — it runs the lattice + Langevin directly. It is the nearest-ready flagship science and should be the **first real CSF3 physics run.**

## The claim (pre-register it, parameter-free where possible)
On an `L×L` torus of designed SO(2) CLU registers with `channel_spring(κ)`, `fdt` noise, no governor, `newtonian_learned`: the equilibrium has a **Kosterlitz–Thouless transition at `T_KT = 1.786 κ r*²`**, and **the KT transition *is* the memory transition** — the only thermodynamically stable stored quantity is the winding number, with lifetime `τ ∝ L^{πρ_s/T − 2}`, the exponent vanishing exactly at the Nelson–Kosterlitz universal jump `ρ_s/T → 2/π`.

## Prerequisites (all validated by xy-1d-control on the real path — carry them)
- `langevin_noise="fdt"` (legacy is 47.5× off), `use_governor=False` (a governed array has no temperature), `kinetic_mode="newtonian_learned"` (**not relativistic** — CM-17: no Gibbs invariant), `coupling_type="channel_spring"` (random-`W` breaks U(1), a *relevant* perturbation that destroys KT), `κ` in the Born-Oppenheimer regime (`κ ≤ k_r/40`, and `κ < k_r/8` or the ring collapses).
- **Sampler:** prefer HMC / MALA(σ*) at γ=0 for equilibrium quantities (exact Gibbs); γ>0 Langevin only when a timescale is the observable. Extend xy-1d-control's `s4b` equilibrium-start stationarity check to `N>2` before trusting any number.

## Items
1. **Run the 1-D winding null FIRST (laptop, cheap, xy-1d-control's recommended follow-up):** write `w=1` on an `N` ring, confirm `τ_winding ∝ 1/N` (1-D memory *degrades* with size) — the foil against which the 2-D "memory *improves* with size" claim is judged. If this doesn't hold on the real path, stop.
2. **2-D torus, `L ∈ {8, 16, 32}`** (`N = 64/256/1024`; `L=4/N=16` is useless — the naive `T_KT` crossing is 13.5% off). `torus_edges(L)` exists.
3. **Observables (xy-lattice-theory §7.4):** helicity modulus/spin stiffness `ρ_s(T)` (the headline — crosses `2/π`, drifts as `T_KT + a/(ln L + b)²`), correlation function `C(r) ~ r^{−η}` (`η(T_KT)=1/4`), vortex density `n_v(T)`, and **the memory observable: written-winding survival `τ ∝ L^{πρ_s/T−2}`** (this is what makes it an ML result, per P1).
4. **Measure `ρ_s` two ways** — from the reduced angles AND from the full-Hamiltonian twisted-boundary free-energy second derivative (the latter does NOT assume the reduction, so it's a genuine test that the CLU array *is* the XY model).
5. **Mandatory nulls (xy-lattice-theory §7.5):** the 1-D control (item 1), and the **broken-symmetry control** (random-`W` `spring_coupling` → no `2/π` jump, orders into Ising) — the cheap devastating one that doubles as the P5 justification.
6. **Kill criterion (§7.7):** if CLU-Langevin `ρ_s(T,L)` at `L=8,16` doesn't converge to the reduced-XY `ρ_s` at `J=2κr*²`, the reduction failed on the real path at that κ — report as a negative, loudly.
7. **Compute (§7.6):** `L=32` (2048 dims, ~10⁶–10⁷ steps/T) is the program's first genuinely A100-scale run. Consider the cheap+honest route: reduced-model Wolff MC for equilibrium quantities + verify (at `L=8`) that the CLU Langevin's stationary law equals the reduced model's. Near `T_KT` critical slowing (`z≈2`) is real — budget for it. Watch the `--extra eval` env note is NOT needed here (no loaders), but `--extra cuda` is.

## Acceptance
1-D winding null confirmed first; the `2/π` universal jump measured across `L∈{8,16,32}` with the Weber–Minnhagen log-correction fit; `T_KT` located; both `ρ_s` routes agree; winding-survival `τ ∝ L^{πρ_s/T−2}` with exponent sign-change at `T_KT`; both nulls run; PREREG written before measurement; every number carries its flag-provenance. **A clean kill is a real result** — if the reduction fails at scale, that ends the NMI phase-diagram thesis and we say so. This is the experiment that decides whether "memory as a thermodynamic phase" is a paper.
