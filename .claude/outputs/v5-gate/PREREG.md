# v5-gate — PRE-REGISTERED PREDICTIONS (written BEFORE any measurement)

Written 2026-07-09, before R1/R3 harnesses were executed. Numbers derive from CM-16's
verified law + the on-disk geometry `.claude/outputs/t-lever-forgetting/s0_geometry.json`
(designed150 seeds 42–46) and `.claude/outputs/v2-full-runs/emergent_summary.json`.

Fixed throughout: `langevin_noise="fdt"`, `common.retie`, ε = dt = 0.05, Δ = 0.5 rad,
float64, γ = 0.05 unless stated, T = 1e-3 unless stated.

---

## R3 — the friction hole (T5/T6)

### R3.0 — A CORRECTION TO THE `t-lever-forgetting` §7 "13.9×" NUMBER (registered as a prediction)

`t-lever-forgetting` §7/§8.3-T6 predicts the enclosed latch decays **13.9× slower**, from
`n₁/₂ ∝ γ_eff/(2−γ_eff)`. **That formula assumes the hole is a *locally thermalized bath*
(noise scale rebuilt from γ_eff).** The **shipped** `FrictionField` is deliberately
**absorb-only**: `integrators.langevin_step` applies `p ← (1−γ_φ(q))(1−γ)p` and then adds
noise whose scale still uses the **scalar γ only**
(`chlu/core/integrators.py`, "the field friction is deliberately NOT coupled to the noise
scale — a pure sink (absorb-only)").

Re-deriving the coset diffusion for the shipped path, with `a = 1−γ_eff`,
`γ_eff = 1−(1−γ)(1−γ_φ)`, `σ² = M T γ(2−γ)`:

```
Var(p)_stat = σ² / (γ_eff(2−γ_eff))
D_θ         = (ε / 2M²r*²) · Var(p) · (1+a)/(1−a)
            = ε T γ(2−γ) / (2 F² γ_eff²)              ← absorb-only (SHIPPED)
D_θ         = ε T (2−γ_eff) / (2 F² γ_eff)            ← coupled local bath (t-lever §7 assumption)
```

Hence **`n₁/₂ ∝ γ_eff²`, not `γ_eff`**, and a friction hole is simultaneously a **brake and a
refrigerator**: the local kinetic temperature drops to

```
T_local_eff = T · γ(2−γ) / (γ_eff(2−γ_eff))
```

**Registered predictions (γ = 0.05, γ_φ = 0.5 ⇒ γ_eff = 0.525):**

| quantity | absorb-only (shipped) | coupled bath (t-lever §7) |
|---|---|---|
| vault factor `n₁/₂(in)/n₁/₂(out)` | **110.25×** | 13.88× |
| `D_θ(in)/D_θ(out)` | **0.009070** | 0.07205 |
| `Var(p)(in)/Var(p)(out)` | **0.12591** | 1.0 |
| `T_local_eff` | **1.2591e-4** | 1e-3 |

These two hypotheses differ by a factor **7.942** in `n₁/₂` and by **7.942** in `Var(p)`.
The measurement is therefore decisive, not a fit.

**Secondary registered predictions**, γ = 0.05, T = 1e-3, vault factor `= (γ_eff/γ)²`:

| γ_φ | γ_eff | `n₁/₂` vault (absorb) | `n₁/₂` vault (coupled) | `Var(p)` ratio (absorb) | `T_local_eff` |
|---|---|---|---|---|---|
| 0.0 | 0.050 | 1.0000× | 1.0000× | 1.00000 | 1.000e-3 |
| 0.1 | 0.145 | 8.4100× | 3.0485× | 0.36249 | 3.625e-4 |
| 0.2 | 0.240 | 23.0400× | 5.3182× | 0.23082 | 2.308e-4 |
| 0.3 | 0.335 | 44.8900× | 7.8468× | 0.17480 | 1.748e-4 |
| 0.5 | 0.525 | 110.2500× | 13.8814× | 0.12591 | 1.259e-4 |

vault(absorb) `= (γ_eff/γ)²` · vault(coupled) `= [γ_eff/(2−γ_eff)]/[γ/(2−γ)]` ·
`Var(p)` ratio `= γ(2−γ)/(γ_eff(2−γ_eff))` (`= 1` exactly under the coupled hypothesis).

**Absolute `n₁/₂` (median FPT, Δ=0.5), designed150, γ=0.05, T=1e-3, absorb-only, γ_φ=0.5:**

| seed | F² | `n₁/₂` no field | `n₁/₂` γ_φ=0.5 (absorb) | `n₁/₂` γ_φ=0.5 (coupled) |
|---|---|---|---|---|
| 42 | 0.639324 | 1241.8 | **136904** | 17237 |
| 43 | 0.639011 | 1241.1 | **136837** | 17229 |
| 44 | 0.638001 | 1239.2 | **136620** | 17202 |
| 45 | 0.607511 | 1180.0 | **130091** | 16380 |
| 46 | 0.674749 | 1310.6 | **144490** | 18192 |

### R3.1 — T5 (registered)
`T = 0`, compact γ_φ hole of ANY strength, latch written **inside** and **outside**:
drift `< 1e-11` rad over 200k steps in **both**. Reason: with `p = 0` at the latched vacuum,
`p ← (1−γ_φ(q))(1−γ)p = 0`; the coset tangent eigenvalue of the one-step Jacobian is
`|λ_flat| = 1` exactly, **even where `∇γ_φ ≠ 0`** (the `∇γ_φ` term multiplies `p* = 0`).
⇒ **0% erasure. A friction hole cannot delete latched coset content.**

### R3.2 — T6 (registered)
`T = 1e-3` global, compact hole covering `|θ| ≤ 0.62` on the ring (so the whole `|Δθ| ≤ 0.5`
exit region has `γ_φ = 0.5` exactly, and the outside arm at `θ₀ = π` has `γ_φ = 0` exactly).
Latch inside decays **110.25×** slower than the latch outside. **A friction hole is a memory
vault (and a cryostat), not a shredder.**

### R3.3 — the scalar control (registered, the discriminator)
Raising the **scalar** γ from 0.05 to 0.525 (no field) at T = 1e-3 gives a vault of only
**13.88×**, and `Var(p) = M_i·T` **exactly** (unchanged). Same γ_eff, 7.94× different
half-life. If both arms come out at 13.88× the field is a coupled bath and my §R3.0
re-derivation is wrong.

---

## R1 — the emergent (MLP) generalization gate

`emergent150_s{42,43,44}` (`potential_type="mlp"`). Prior on-disk measurement
(`.claude/outputs/v2-full-runs/emergent_summary.json`): softest `μ² = 0.0545 / 0.0?? / 0.0??`,
softest-mode angular overlap **0.697** (s42), ring ripple (washboard) **5.67e-2**,
`n_half` of the softest mode at γ=0.05: pred 246.6, obs 268–277.

**Registered predictions** (`μ²_ring` = true coset curvature `V_ring''(θ_min)/F²`, measured in E0):

1. **The T=0 latch DECAYS** (pseudo-Goldstone): `n₁/₂ = ln2 / gap(μ²_ring, ε, γ)`, i.e.
   `|λ_flat| < 1` strictly — the *opposite* of CM-16(a)'s `||λ|−1| ≤ 1.7e-14`.
2. **`n₁/₂(γ)` at T=0 is the V-curve of CM-16(c)'s unification**, minimum at
   `γ_crit = 2εμ_ring`. With `μ²_ring ≈ 5e-2 ⇒ μ_ring ≈ 0.22 ⇒ γ_crit ≈ 0.022`.
   ⇒ `∂n₁/₂/∂γ < 0` for `γ ≲ 0.011`, **`> 0` for `γ ≳ 0.045`**.
   **So the γ-slope at the CM-16 grid (γ ∈ {0.05,0.1,0.2}) is predicted POSITIVE on
   emergent too — but for the *massive/overdamped* reason, not the Goldstone-diffusive one.**
   This is a confound the task's framing does not anticipate, and it is the crux of the gate:
   *a positive γ-slope on emergent would NOT confirm CM-16(c).* The discriminator is the
   **T-dependence** (CM-16: `∂n₁/₂/∂T = −1`; massive: `∂n₁/₂/∂T ≈ 0`).
3. **Crossover temperature** `T*(γ)` where relaxation and diffusion give equal `n₁/₂`:
   `T* = 0.378748 Δ² F² γ · gap(μ²_ring,ε,γ) / (ε²(2−γ) ln2)`.
   For the overdamped branch `gap ≈ ε²μ²/γ`, so `T* ≈ 0.378748 Δ² F² μ²_ring / ((2−γ) ln2)`,
   nearly γ-independent. With `F² ≈ 0.98`, `μ²_ring ≈ 0.05`, γ=0.05: **`T* ≈ 3.4e-3`.**
   Below `T*`: `∂n₁/₂/∂T ≈ 0` (massive). Above `T*`: `∂n₁/₂/∂T → −1` (diffusive).
4. **BUT** the coset is *pinned*, not merely soft: the ring has a washboard with a
   **finite barrier** (ripple 5.7e-2). Thermal angular spread
   `σ_θ = √(T/(F²μ²_ring)) ≈ 0.14 rad at T=1e-3` — well inside Δ=0.5. So at `T < T*` the
   register does **not** diffuse out at all; it relaxes to the nearest washboard minimum.
   **Registered risk:** `T*` may sit at or above the anharmonic/barrier-hopping temperature
   (`T ≳ barrier`), in which case the diffusive branch is **never** cleanly reachable on
   emergent checkpoints, and CM-16's sign flip does **not** generalize. That is the honest
   kill-shot for the V5 short.

**Decision rule (fixed in advance).**
- **GO** iff (R3 lands: vault ≥ 10×, T5 drift < 1e-11) **AND** (R1 exhibits a *measurable*
  `T*` with a diffusive branch above it, i.e. `∂n₁/₂/∂T` reaching ≤ −0.5 at some
  `T < barrier`, with `∂n₁/₂/∂γ > 0` there).
- **V2-APPENDIX** if R3 lands but R1's diffusive branch is unreachable/muddy on emergent.
- **KILL** if R3 fails (no vault).
