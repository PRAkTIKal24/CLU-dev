# PREREG — designed-mechanism-learned-content (w22)

Written BEFORE running the production harness (protocol §5 pre-registration rule).
The acceptance criterion is a measured law (`K_learned(d)` growth), so numbers and
their derivation are committed here first.

## The two hypotheses (the decider)

Fixed **designed mechanism** = `AtomDictionaryPotential` (learned amplitudes/centers/
widths, group-masked writes). **Learned content** trained by the static write
objective (`train_memory.py`) on a `d`-dimensional address ball (sites =
`designed_sites(d,K)`, payload channel at index `d`). Parameter budget scaled with
K (`n_atoms = atoms_per_item · K`, groups = K) so a plateau is a LEARNING failure,
not a parameterization-capacity failure (theorist §4.3, `B_total ≤ P·b_θ`).

- **H-GEOMETRY:** the wall is the ring running out of room. `K_learned` grows
  strongly with `d` (designed capacity is `4·2^d`). Primitive claim ALIVE.
- **H-LEARNING:** gradient descent cannot fill a landscape past ~8 items
  regardless of room. `K_learned` stays near ~8 at every `d` while `K_designed`
  climbs `4·2^d`. Primitive claim in serious trouble.

## Primary prediction (committed)

**H-GEOMETRY-WEAK** (my primary, ~55% weight): `K_learned` **grows with d** (H-LEARNING
strict-plateau rejected), but **sub-exponentially / below the designed base 2.0**,
because the static write pays a dimension-dependent tax (more atoms to place, more
barrier constraints, worse Adam conditioning as K·d grows).

Derivation of point predictions. At fixed K, ball site separation is
`Δ(d,K) ≈ 2R·K^{-1/d}` (measured `d_eff≈0.72–0.83·d` in address-space-dimension-scaling).
A local atom write succeeds while `Δ` stays above ~`3·w` (the `Δ_req≈3.1·max(w,σ)`
resolution floor). With `w≈0.3` (atom init width) and `R≈1`, the geometry admits
far more than 8 wells at every d≥2, so if learning tracked geometry we would see
`4·2^d`. I predict learning realizes a **fraction** of that:

| d | K_designed = 4·2^d | **K_learned (predicted)** | log2 |
|---|---|---|---|
| 2 | 16 | **4** | 2.0 |
| 3 | 32 | **8** | 3.0 |
| 4 | 64 | **16** | 4.0 |
| 6 | 256 | **32** | 5.0 |
| 8 | 1024 | **64** | 6.0 |

i.e. `K_learned ≈ 4·2^{0.8(d-2)}` → **fitted exponential base A_learned ∈ [1.55, 1.85]**
(vs designed 2.0), R² ≥ 0.9, and `K_learned/K_designed` **falling** with d (16/16=1.0
at d=2 → 64/1024≈0.06 at d=8). Strict-plateau (all K_learned within ±1 ladder rung
of 8) is REJECTED under this prediction.

## Competing prediction (committed, ~35%)

**H-LEARNING**: `K_learned ∈ {4,8,8,8,8}` (flat near 8), base A_learned ≈ 1.0,
while designed climbs 2.0. If observed, this is the most consequential result the
program could produce — report loudly, do not tune around it.

## Residual (~10%): **H-GEOMETRY-STRONG** — `K_learned` tracks `4·2^d` within a
factor ~2 (base ≥ 1.9). If observed, the primitive claim is strongly alive.

## Decision rule (stated before measuring)
- Fit `log2 K_learned` vs `d` (exponential base A_learned) over the measured
  (non-censored) cells, ≥5 seeds per discriminator cell.
- **A_learned ≥ 1.9** ⇒ H-GEOMETRY-STRONG. **1.3 ≤ A_learned < 1.9 with R²≥0.8 and
  growth ≥ +2 ladder rungs over d=2→8** ⇒ H-GEOMETRY-WEAK (primary). **A_learned <
  1.3 / no growth beyond ±1 rung of 8** ⇒ H-LEARNING.
- The designed arm is re-measured on the SAME harness (not assumed `4·2^d`); if the
  designed arm does not itself reproduce ≈`4·2^d`, the harness is suspect and the
  learned numbers are not reportable (a gate, like potential-function-class's
  replication gate).

## Item 2 (mass) prediction
Per Prop F1 (relaxation-fiber-capacity), mass is address-side and worth ~0 bits in
a SEPARABLE well. The atom wells `−A exp(−|q−c|²/2s²)` have a Hessian at their own
center `= (A/s²)·I` on the address block (isotropic, **diagonal**), so off-diagonal
coupling `∂_i∂_j V ≈ 0`. **Prediction: per-item mass gives ~0 gain** on
`K_learned`/fidelity (Δstrict within seed noise), and this is a *confirmed
prediction*, not a failure. Measured coupling ratio `mean|off-diag| / mean|diag|`
at stored sites predicted **< 0.1**.

## Item 3 (interference across d) prediction
`potential-function-class` found the WRITE OPERATOR governs interference (masked
70× better than global at d=2). **Prediction: the masked-write advantage SURVIVES
at every d** (masked corruption ≪ global), and the advantage is roughly d-INDEPENDENT
because masking is exact in parameter space (frozen atoms bit-identical) at any d.
Global-write corruption should if anything *fall* with d (sites separate → Gaussian
tails smaller), but masked stays ~0.

Seeds: ≥5 on the discriminator (item 1). Interference ≥3 seeds/d. Mass ≥3 seeds.
