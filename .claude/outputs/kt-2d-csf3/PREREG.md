# PREREG — kt-2d-csf3 (2-D Kosterlitz–Thouless memory phase)

**Written before any harness that measures the quantities below was run.**
Commit `e3c8931`. Analyst: results-analyst, w16.

The task's acceptance criteria are *measured laws/exponents* (universal jump `2/π`,
`T_KT`, winding exponent `πρ_s/T − 2`, `τ∝1/N`). Per protocol §5, I commit to values
and their derivation here, then measure.

## Constants (from the verified dictionary, xy-lattice-theory)
- Designed lattice: `λ=f=1` ⇒ `k_r = 8λf² = 8`, `r* = 1`, `κ = 0.05`.
- Exchange `J = 2κr*² = 0.10`. Born–Oppenheimer: `J₂/J₁ = (κ/k_r)/(1−4κ/k_r) = 0.634%` (safely XY, P6).
- Work in `T/J` units for the reduced model; CLU temperatures `T = (T/J)·0.10`.

## Predictions I commit to

### P-A. 1-D winding null (item 1, the gate) — REAL CLU PATH
- **Claim:** written winding `w=1` on an `N`-unit CLU ring decays with lifetime `τ_winding ∝ 1/N`
  (1-D memory *degrades* with size). Log-log slope `d ln τ / d ln N = −1.0`.
- **Derivation:** a phase slip in 1-D is a *local* saddle of cost `O(J)`; the slip rate is extensive
  in N (∑ of independent local escape rates), so `rate ∝ N`, `τ ∝ 1/N`. Per-site slip rate constant.
- **Pre-registered numbers:** log-log slope of `τ(N)` vs `N` = **−1.0 ± 0.20** over `N ∈ {8,16,32,64}`;
  per-site rate `N·rate(N)/rate(N=16)` constant to within CV ≤ 25% (Langevin dynamics, not MC sweeps,
  so the *absolute* rate will differ from the theorist's reduced-MC `0.00656`; only the **scaling** is pre-registered).
- **Kill for the whole program:** if the CLU ring shows `τ` *increasing* with N (memory improving in 1-D),
  the winding-memory picture is wrong on the real path — stop.

### P-B. 2/π universal jump & T_KT (items 2–3) — reduced-XY MC, L∈{8,16,32}
- **Claim:** `ρ_s(T)/T` crosses the line `2/π` (Nelson–Kosterlitz universal jump); the naive crossing of
  `ρ_s(T)` with `2T/π` drifts as `T_KT + a/(ln L + b)²` (Weber–Minnhagen log correction).
- **Pre-registered numbers** (literature + theorist's single-seed MC, now to be reproduced multi-seed):
  - Naive crossing `T_×/J`: **≈ 0.96 (L=8), 0.94 (L=16), 0.92 (L=32)**, monotone decreasing.
  - Weber–Minnhagen fit extrapolation: **`T_KT/J = 0.893 ± 0.010`** (Hasenbusch 2005: `0.8929`).
    In CLU units **`T_KT = 1.786κr*² = 0.1786`**.
  - Universal jump value: at the crossing, `ρ_s/T = 2/π = 0.6366` by construction; the *content* is that
    the crossing sits at `T_KT`, and `ρ_s` drops discontinuously (finite-size rounded) there in the L→∞ limit.
  - `η(T_KT) = 1/4`: `C(r) ∼ r^{−η}` at the transition with `η = 0.25 ± 0.03` (power law below T_KT,
    exponential above).
  - Vortex density `n_v(T)`: rising, ≈ `3e-3 (T=0.8J) → 2e-2 (1.0J) → 7e-2 (1.2J)` (theorist L=32).

### P-C. Both ρ_s routes agree (item 4)
- **Claim:** ρ_s from the reduced-angle fluctuation formula = ρ_s from the twisted-boundary free-energy
  second derivative (finite-difference), to within MC error.
- **Pre-registered:** `|ρ_s^twist − ρ_s^fluct| / ρ_s ≤ 5%` at each (T,L) away from the immediate T_KT
  rounding region.

### P-D. Winding survival exponent (item 3, memory observable) — reduced-XY MC
- **Claim:** `τ_winding(L) ∝ L^{πρ_s/T − 2}`; exponent > 0 for `T < T_KT`, < 0 for `T > T_KT`, sign change at T_KT.
- **Pre-registered:** at `T = 0.7J` (below T_KT, `ρ_s/T ≈ 1.1` ⇒ exponent `π·1.1−2 ≈ +1.5`): `τ` **increases**
  with L, log-log slope **> 0** (memory improves). At `T = 1.1J` (above T_KT): `τ` **decreases** with L,
  slope **< 0**. Sign change bracketing `T_KT`. (Absolute exponent value is approximate — ρ_s is T,L-dependent;
  the pre-registered, falsifiable content is the **sign** and the **contrast with the 1-D −1 slope**.)

### P-E. Kill criterion (item 6) — REAL CLU PATH at L=8
- **Claim:** CLU-Langevin stationary `ρ_s(T)` and `⟨cosΔθ⟩` at `L=8` equal the reduced-XY values at `J=2κr*²`.
- **Pre-registered:** `ρ_s^CLU / ρ_s^reduced ∈ [0.90, 1.10]` and `⟨cosΔθ⟩^CLU/⟨cosΔθ⟩^reduced ∈ [0.92, 1.06]`
  at 3 temperatures spanning T_KT (as in 1-D xy-1d-control: 1.5–6.8% BO+thermal dressing deficit, growing with T,
  ρ_s slightly *below* reduced). **If ρ_s^CLU does not track ρ_s^reduced at L=8 → kill (reduction failed at 2-D).**

### P-F. Broken-symmetry null (item 5) — random-W spring
- **Claim:** the same 2-D lattice with default random-`W` `spring_coupling` shows **no 2/π jump** —
  the p=2 anisotropy (relevant, `x₂=1/2`) drives Ising ordering.
- **Pre-registered:** reduced XY-with-`h₂cos2θ` at `h₂/J = 1` shows `ρ_s(T)/T` NOT crossing `2/π` in the
  KT manner (either Ising-like jump at higher T, or QLRO destroyed); and/or the real random-W CLU chain
  reproduces xy-1d-control's `C(1) ≈ 0` (vs XY 0.446). Devastating, as in 1-D.

## What is honestly OUT of scope on this laptop (deferred to CSF3/A100)
- Full CLU-Langevin at L=16, L=32 (2048 dims, 10⁶–10⁷ steps/T, z≈2 critical slowing) — A100-scale.
  The laptop delivers: reduced-XY MC for the phase diagram (all L) + the L=8 CLU-vs-reduced bridge (P-E)
  that licenses the reduced model as a proxy on the real path. This is the theorist's recommended route (a)+(b).
