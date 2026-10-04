# PREREG — lattice-capacity-theory (physics-theorist, w24)

Protocol §5 pre-registration rule. This task's acceptance criterion is a *theory
note*, not a measured law, but four of my checks measure ratios/laws, so each is
registered here with its prediction **and the order in which it was written vs
run**. Anything discovered *after* running is labelled EXPLORATORY and carries no
pre-registration credit.

Written before any script was executed except where stated.

| # | check | prediction registered BEFORE running | status |
|---|---|---|---|
| P1 | `checkA` A1 superposition | cross-item corruption **superposes linearly** in the number of foreign items (V is an exact sum of atoms ⇒ forces add) ⇒ additivity budget `K_max ≈ tol/eps` | ◐ **REFINED**: accumulation is **incoherent (√m)**, not linear — random-direction perturbations. Budget is `(tol/eps)²`, i.e. far *more* permissive than registered. Direction of the error is against my own conservatism. |
| P2 | `checkA` A3 random placement | `K` reachable by *unmanaged* random placement scales as `N_pack^{1/2}` (expected-violating-pairs = C(K,2)·(d_safe/2R)^d = 1 ⇒ exponent d/2) | ✅ exponent confirmed; measured K_random(50%) sits at or slightly below `√N_pack` |
| P3 | `checkB` B1 factorization | masked-sequential write on a shared atom pool is **bit-identical** to N independent shard writes, *up to the Gaussian site tail* `exp(−sep²/2s²) ≈ 1.5e-5` | ❌ **FAILED**: deviation is **1.4e-2 relative on amps — 3 orders ABOVE the tail.** Cause found post-hoc: `init_scale=1.0` scatters every group's atoms over the whole ball, so groups overlap functionally at init. Registered mechanism (site tail) is not the binding one. |
| P4 | `checkC` barrier dilution | `write_loss`'s all-pairs `mean` normalisation makes the crowding term's gradient share fall like **1/K** ⇒ the Head's scale-invariance ablation should move the ceiling | ❌ **REFUTED**: ratio ‖∇l_bar‖/‖∇(l_grad+l_min)‖ = 0.12 → 0.55 → 0.29 over K = 4 → 16 → 128, i.e. **roughly flat**, because the violating-pair *fraction* stays O(1) as sites crowd. My dilution mechanism is not present. |
| P5 | `checkB` B2 separability | a single relativistic CHLU of dimension N·d is **not** shard-separable (the √ couples all momenta); a `CLULattice` with per-unit T **is** | ✅ confirmed: off-block ∂²T/∂p∂p = 8.4e-2 vs diagonal 0.81 for the single unit; **exactly 0.0** for the lattice |
| P6 | `checkD` routing | broadcast + **argmin post-settle energy** is a valid zero-learning router | ❌ **REFUTED**: post-settle energy routes at or *below* chance (0.031–0.55 vs 1/N), because every block's well has the same depth. Pre-settle energy / settling displacement route at **1.000**. |

**EXPLORATORY (not pre-registered — discovered while running `checkE`/`checkF`,
must be treated as a hypothesis, not a result):** the critical site separation at
every measured capacity wall — designed *and* learned, d = 2…8 — sits at
**≈2.4–3.0 × the well width**, which would make N92's "d-independent write
ceiling ≈32" a fixed-width criterion in a fixed-radius ball rather than an
optimizer limit. This was found, not predicted. It needs its own pre-registered
test (§5 of the report names it).
