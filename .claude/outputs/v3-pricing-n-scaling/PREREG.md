# PREREG — v3-pricing-n-scaling (pricing law at N > 2)

Written **before** the seeded trained-grid harness is run. A calibration/timing smoke
(`smoke.py`) has been run and confirmed only the *extractor identity* and per-run
timing; the deliverable — the **seeded, trained** exponent grid with CIs and the
control comparison — has **not** been measured at prereg time.

## Object under test (V3 §3.3 priced-channel law)
On a trained CLU lattice with SO(2) (MexicanHat) units, channel inertial mass M=1,
coupling strength κ, extractor `κ_eff = ¼·M·λ_max(mass-weighted coupling Hessian)`:
1. `μ_rel² = 4κ_eff/M` (stiffest phase mode).
2. `sync ∝ κ_eff^(−1/2)` (quarter period of that mode, γ=0).
3. `n₁/₂ ∝ κ_eff^(−1)` (half-life at γ=0.2).

At N=2 these are measured (v3-lattice-build: −0.499 / −0.986; seed-sweeps item2: μ_rel²
to ≤1.2%). App C concedes the exponents are **inconclusive on trained lattices**. This
prereg commits to what N∈{4,8,16} will show, and how the three candidate explanations
of App C (a artifact / b extractor loses power / c law degrades) will be separated.

## Predicted values (committed)
**Primary arm — `channel_spring` (U(1)-preserving, frozen coupling), MexicanHat units, wake-only trained, ≥5 seeds, chain + ring:**
- `κ_eff = 0.5·κ·λ_max(L)` **exactly** (ring N≥4: κ_eff=2κ; chain: ramps 0.5·κ·[2, 3.414, 3.848, 3.962]). Derivation: phase-sector Hessian of Σ_edges κ‖q_i−q_j‖² is (2κ/M)(L⊗I); extractor reads λ_max. *(Confirmed to 5 dp in the calibration smoke — recorded here as a pre-measurement identity, not a finding.)*
- **sync exponent vs κ_eff = −0.50 ± 0.03** at every N∈{2,4,8,16}, both topologies. Derivation: quarter period ∝ 1/μ ∝ κ_eff^(−1/2), and the phase sector is pure coupling for SO(2) units so it is N-independent.
- **n₁/₂ exponent vs κ_eff = −1.00**, expected to read **−0.95 ± 0.10** because of the known first-crossing / kick-phase ripple (F5 App-N) that grows toward high κ (h→h*(γ)); this ripple is N-independent.
- **μ_rel² = 4κ_eff/M pointwise to < 2%** at every N (≈ exact by construction; deviation only from unit-training drift of the vacuum).
- Seed scatter of the trained primary arm is predicted **small** (< a few %), because the coupling is frozen and MexicanHat units start at the analytic vacuum (smoke: κ_eff moved 0.1981→0.1983 over 60 epochs).

**Control arm — `spring_random` (legacy random-W, U(1)-breaking, trainable W, static knob KAPPA_S=0.1), trained to reproduce data at each κ_target:**
- Read `κ_eff` **clusters**: over κ_target spanning 100× (0.01→1.0), trained κ_eff range-factor **< 1.5×** (item2 at N=2: 0.044–0.054, 1.22×). ⇒ the exponent fit spans too small a κ_eff range to read a −1/2 / −1 law → **inconclusive**, at every N.
- U(1) breaking is **measurable**: grad_norm at the SO(2) vacuum shifts off 0, and measured μ_rel² departs from 4κ_eff/M by ≫ the primary arm's < 2%.

## Falsification / decision rule (committed before measurement)
Let `s_N` = sync exponent, `h_N` = n₁/₂ exponent at unit-count N (primary arm).
- **"Law HOLDS at N>2"** iff for all N∈{4,8,16} and both topologies: `s_N ∈ [−0.55,−0.45]` and `h_N ∈ [−1.15,−0.80]`, with per-N 95% CI **excluding** the N=2 value by less than its own width (i.e. no significant drift), and μ_rel² pointwise < 2%.
- **"Law DEGRADES with N"** iff `|s_N|` or `|h_N|` drifts **monotonically toward 0** as N grows with N=2 vs N=16 CIs **separated** (e.g. s_16 shrinks to −0.35 with CI not covering −0.50).
- **"Extractor loses power at N>2"** (candidate b) iff κ_eff departs from the analytic `0.5·κ·λ_max(L)` by > 5% at N≥8, **or** the per-seed log-log fit R² collapses (< 0.95) at N=16 while the raw μ_rel² still tracks κ. (Smoke already shows κ_eff exact — b is disfavored a priori for the primary arm, but the fit-quality half is measured on the trained seeded grid.)
- **App C's "inconclusive" = ARTIFACT (candidate a), now fixable** iff: control κ_eff range-factor < 1.5× across a 100× κ_target sweep (exponent unfittable) **AND** primary channel_spring κ_eff spans the full decade with `s_N`, `h_N` passing the HOLDS rule at the same N. If instead the **primary** arm also fails the HOLDS rule at N>2, App C is candidate (c) — a genuine physics degradation — and V3 must scope its title accordingly (a clean negative, and a publishable one).

## Grid, seeds, commands
- Primary: N∈{2,4,8,16} × {chain,ring} × κ∈{0.01,0.03,0.1,0.3,1.0} × seeds{0..4}. Wake-only train (sleep off, per v2 Finding 0), 60 epochs, window 128, dt=0.05, γ_probe=0.2, x64.
- Control: N∈{2,4,8,16} × chain × κ_target∈{0.01,0.03,0.1,0.3,1.0} × seeds{0..4}, trainable-W spring, static KAPPA_S=0.1.
- Commit: `df5e44d` (main; channel_spring + gate-free-energy prereqs merged). JAX 0.9.0, CPU, x64.
- `python runner.py --arm channel_spring` ; `python runner.py --arm spring_random` ; `python analyze.py`.
