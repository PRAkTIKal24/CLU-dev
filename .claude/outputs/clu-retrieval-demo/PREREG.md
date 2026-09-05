# PREREG — clu-retrieval-demo

Written **before** running the harness. Base `main` @ `1e7ace5`.
Author: experiment-engineer. Task: `.claude/tasks/clu-retrieval-demo.md`.

## The design I am about to build (stated so the predictions are falsifiable)

**One hand-designed potential family, `RingRegisterPotential`, dim = 3.**

```
V(q) = lam*(r^2 - f^2)^2                  # ring vacuum, r^2 = q0^2+q1^2  (radial confinement)
     + b*(1 - cos(K*theta))               # K angular wells = K item sites, theta = atan2(q1,q0)
     + 0.5*kappa*(q2 - s(theta))^2        # PAYLOAD coordinate, s(theta) = sum_k a_k * bump_k(theta)
```
- **Item k** = the pair (well at `theta_k = 2*pi*k/K`, payload value `a_k`).
- `a_k` are designed, **non-monotone in k** (fixed permutation of a grid), so payload is not a smooth function of address angle.
- **Address** = `(m, q0, p0)`; the payload coordinate is **always launched at `q2(0)=0`**, so the retrieved payload is information that exists *only in `V`*, never in the address. This is the anti-decoration guard.
- **Read** = linear probe on the **tail** (last 25%) of the rollout. Two reads reported:
  (i) full-state tail, (ii) **payload-only tail (`q2` alone)** — the strict read, blind to the address plane.
- **Blank control**: identical addresses in `V_blank` (same hat, `b=0`, `kappa=0`, payload decoupled). A working loop must show payload-only accuracy at **chance** in blank.

Item 3 uses a separate designed dim-4 landscape (`ThreeModePotential`): SO(2) Mexican-hat channel (0,1) — permanent angle (mu^2 == 0) + decaying radius (mu_rad^2 = 8*lam*f^2) — plus an uncorrelated double-well site on (2,3).

## Predictions (committed)

**P1 — Selectivity at 2 items.** Payload-only linear read recovers item identity at **>= 95%** (2-way, chance 50%) over noisy query replicates. Blank control **<= 60%** (i.e. chance within noise). Retrieval survives the full rollout because friction parks the particle in the well: accuracy should be **flat in tail position** once settled, so "trajectory length over which it survives" = unbounded at gamma>0, and *degrading* at gamma=0 (particle never settles, keeps circulating).

**P2 — Mass as an address key. I predict the STRONG FORM FAILS.**
Reasoning stated up front, so a pass/fail is meaningful:
- At **fixed `p0`**, mass is not a time reparameterization (that only holds at fixed initial *velocity*): initial KE = `p0^T M^-1 p0 / 2`, so a **scalar** mass `m` is a pure **energy dial** — monotone in `1/m`. It can index items only along an *ordered* 1-D ladder (how far around the ring the particle coasts before friction parks it). Monotone => cannot realize an arbitrary item<->mass assignment (a permutation).
- A **mass vector** `(m_0,m_1)` steers the launch *direction*: `qdot(0) = M^-1 p0`, direction angle `atan2(p0_1/m_1, p0_0/m_0)`. Since `m_i > 0`, the reachable directions are confined to the **open sign-orthant of `p0`** — a 90-degree cone in the 2-D address plane, not the full circle.
- **Quantitative commitment:** at fixed `(q0,p0)` on a K=8 ring, sweeping mass over 3 decades retrieves **<= 3 distinct items** (<= 1.6 bits), and items outside the `sign(p0)` cone are **unreachable at any mass**. Verdict I expect to report: *mass works as a coarse ordinal/energy key, not as a general address key.*
- Falsified if >= 5 distinct items are retrieved by mass alone, or if items outside the sign cone are reached.

**P3 — Three write modes.** All three realizable side by side (designed, additive).
- Permanent (channel angle): retained `|Delta theta| <= 1e-6` over the full rollout at gamma>0 — protected by *exact* designed SO(2) symmetry (torque-free), not by learning.
- Decaying (radial excursion, same locus): finite half-life; underdamped envelope => `t_1/2 ~ ln2 / (gamma_eff/2)` in step units, i.e. **`n_1/2 ~ 2*ln2/gamma`**. At gamma=0.01 I predict `n_1/2 = 139 +/- 40` steps.
- Corruption of the permanent neighbour by the decaying write: **<= 1e-6** (machine-precision), because a purely radial write carries zero angular momentum and `grad V` is radial.

**P4 — Interference vs item count.** Payload-only accuracy: **>= 95% at K=2 and K=4**, **< 90% at K=8**, **< 70% at K=16**. **Stated practical ceiling: 4-8 items** in a 2-D address plane at fixed well width. This is *consistent with* (not independent of) v5-gate's measured 1-1.6 bit register capacity, and I expect to report the ceiling **bluntly** as a handful of items.

**P5 — The learnability crux (WEAK form; the primary number).**
- **(a) Restructuring test.** Plain GD on `(log m, q0, p0)` against a retrieval loss `(mean tail q2 - a_target)^2`, from deliberately bad initial addresses. I predict **partial failure**: the loss inherits the washboard barriers, so GD is trapped in whichever well it starts near. Committed numbers: success rate **<= 30% at K=8**, **>= 50% at K=4** (wider, overlapping payload bumps give longer-range gradient), and **~0% for an antipodal init**. Steps to succeed when it does: **O(100)**.
- **(b) Smoothness.** Retrieval quality vs `q0` is **piecewise-smooth with sharp cliffs at basin separatrices** (not fractal, but not globally convex). Gradient norm vs rollout length **grows** at small gamma (no contraction, conservative-ish) and **saturates/decays** at larger gamma. I explicitly note in advance that (b) looking ugly does **not** override (a).

## Scoring rule
Each prediction is marked ✅ (within stated band) / ❌ (outside) / ~ (partial), with the measured number next to it. No post-hoc band widening.
