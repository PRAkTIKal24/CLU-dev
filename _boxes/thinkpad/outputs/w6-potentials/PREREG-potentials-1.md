# PREREG — w6 hub/potentials — scored run 1 of 3

**Written 2026-09-20, BEFORE any held-out number.** Charter §A.1/§A.3: this file is not edited
once a result has been seen against it; amendments are appended and dated.

- **Hub:** `w6/potentials` · **Base:** `base/iter6 = df025685` · **Worktree:** `/c/chlu/worktrees/CHLU-w6-potentials`
- **Dial:** the kernel and the dynamics — width, separation, friction, mass.
- **Scored budget:** `potentials` 3. **This is run 1.**
- **Cache generation 2 only** (`cache_gen2_root` from `RIVALS-READY.json`). Byte pin **524,288 B, tolerance 0**.

---

## 0. What this hub is NOT allowed to claim, and why the plan changed

1. ⛔ **No refusal claim.** Rows 5/6 are closed; the 8-byte presence+bracket rule ties the field store at
   (1.0000, 1.0000) on both and the refusal target is 0/3 — SPENT. No refusal row is scored here.
2. ⛔ **No interpolation-floor row.** Row 2 belongs to a size-matched continuous MLP on every seed under
   two implementations. The floor is measured here as a **property of the kernel** and reported as
   mechanism only.
3. ⛔ **My task file's law 1 — "Mexican-hat width/separation sets the interpolation floor and the refusal
   gap" — DOES NOT EXIST IN THE SHORTS, and is therefore not pre-registered against.** Established twice,
   independently: by my own exhaustive search of all three PDFs (LOG 06:36Z) and by the inventory spoke
   (`/c/chlu/outputs/w6-shorts-inventory/shorts-experiments-inventory.md` §1.1, *"ABSENT. Do not
   pre-register against it."*). "interpolation" occurs **zero** times in the corpus; "Mexican hat" occurs
   **once**, as a float64 round-off testbed in V2 App. M.4, with no width and no separation parameter.
   ⛔ The inventory's §3 additionally forbids the two nearest substitutes: **A2** (`mu^2 ∝ A/l^2`, an
   undefined V1 table cell — *"the closest thing in the corpus to a 'width' law and it must not be treated
   as one"*) and **A5** (the measured minimum separation 1.540000, a constant, not a law).
4. ⭐ Therefore **all three scored runs go on my task file's law 2** — *"Friction and mass: do the derived
   settle-time laws match measurement?"* — which the inventory's §1.3 confirms **is** derived in closed
   form, and on the **§2.3 friction dial** the Head redirected me to.

---

## 1. The object

The **field store** at the 524,288 B pin (`gen_field.build_field`, `exp_w5_compose_field.build_field_for_leg`),
gen-2 cache, built on the same fit pool and the same inherited `factor_aligned_balanced` partition as every
other store arm. 5 wells, 64 atoms × 2048 f32. Labels enter at fit time only (address book); the landscape,
the read and the write are label-free. **No content is added and no byte is changed by this hub.**

Its landscape is the energy this repository already uses (`gen_landscape.store_energy`), evaluated on the
field store's own atoms, its own per-well width `well_sigma[k]` and its own lifetimes:

```
E_k(q) = -sigma_k^2 * logsumexp_j [ log life_j - ||q - a_kj||^2 / (2 sigma_k^2) ]
```

Its dynamics are `chlu.core.integrators.velocity_verlet_step`: kick–drift–kick of `H = p^T p/(2M) + E(q)`
followed by `p <- (1-gamma) p`. ⭐ **That map IS the shorts' map** (inventory D3: `tr J = (2-gamma)(1-h^2/2)`,
`det J = 1-gamma`), and this is asserted mechanically, not assumed:
`exp_w6_potentials.assert_map_matches_repo_integrator` reproduces the repo integrator to `1.9e-08` and the
leg **refuses to run** if it does not.

---

## 2. THE PREDICTED CURVES — parameter-free, with the papers' own numbers

Symbols as the shorts define them: `eps` step, `gamma` per-step damping, `mu` spectral mass,
`M` inertial mass, `h = eps*mu`, `n_half` half-life in map applications.

**`eps = 0.05` is inherited verbatim** — every quantitative result in all three shorts is at `eps = 0.05`
(inventory "Shared rider"). It is not a knob I chose.

### P0 — the measurement instrument (D15, V2 App. A / V5 §1)
`mu_k^2 = lambda_k(M^-1/2 grad^2 V M^-1/2)` at a critical point of `V`. Computed exactly and in closed
form: `grad E = q - <a>_w`, `Hess E = I - Cov_w(a)/sigma^2` (verified against autodiff to `3.1e-15`), so the
critical point is the mean-shift fixed point and the spectrum comes off a 64×64 Gram matrix rather than a
2048×2048 Hessian.

### P1 — the exact one-step half-life (D3, D4, D5, D7)
```
n_half_pred(mu^2, eps, gamma) = ln2 / (-ln |lambda_slow|)
lambda_slow = larger-modulus root of  l^2 - (2-gamma)(1 - h^2/2) l + (1-gamma) = 0,  h^2 = mu^2 eps^2
```
**Parameter-free: nothing is fitted.** Validated against the shorts' own reported numbers before any
held-out data was touched:

| check | source | paper's number | this formula |
|---|---|---|---|
| underdamped floor at `gamma = 0.05` | V2 §4.1 | **27.03** | **27.02681** |
| `lambda_ret` at `gamma = 0.002` | V5 App. (line 684) | **0.998999499** | **0.9989994995** |
| `n_half` at `gamma = 0.002` | V5 App. (line 684) | **692.5** | **692.4538** |

### P2 — the bands and their edges (D8, D9)
```
h*  = sqrt(2/(2-gamma)) (1 - sqrt(1-gamma)) = gamma/2 + O(gamma^2)      [crossover / exceptional point]
h_f = sqrt(2/(2-gamma)) (1 + sqrt(1-gamma)) = 2 - O(gamma^2)            [stability limit]
```
At `gamma = 0.05`: **`h* = 0.0256431`**, **`h_f = 1.9998356`**. At `gamma = 0.2`: `h* = 0.1112835`.

### P3 — the overdamped branch (D5, D6)
```
n_half ~= 2 gamma ln2 / [(2-gamma)(eps mu)^2]          slope  d log n_half / d log mu^2 = -1
```
⚠ The shorts write `≈`, not `=` (inventory §1.2): it is exact only to leading order in `h^2/gamma^2`, with
stated error `O(h^4/gamma^3)`. **The scored comparison is against P1 (exact), not against P3.**

### P4 — the curvature-independent floor (D7)
```
n_half = 2 ln2 / (-ln(1-gamma))     independent of mu, because |lambda|^2 = det A = 1 - gamma
```
**27.0268 at `gamma = 0.05`; 6.2126 at `gamma = 0.2`; 692.454 at `gamma = 0.002`.**

### P5 — the V-curve critical point (D16, V5 App. G.3)
```
gamma_crit = 2 eps mu          the retention dial's computable minimum
```

### P6 — the two branch log-slopes (D17, D18, V5 App. G.3) — ⭐ **the sharp test**
```
overdamped :  d ln n_half / d ln gamma = 1 + gamma/(2-gamma)      NOT +1
underdamped:  d ln n_half / d ln gamma = -gamma/[(1-gamma)(-ln(1-gamma))] = -1 - gamma/2 + O(gamma^2)
```
At `gamma = 0.05`: **+1.0256** and **−1.0256**. At `gamma = 0.2`: **+1.1111** and **−1.1157**.
At `gamma = 0.5`: **+1.3333** and **−1.4427**.
The shorts are explicit that the overdamped slope *"exceeds +1 at every finite gamma"*.

### P7 — confinement (D14, V5 App. G.4)
Adding `alpha ||q - q*||^2` lifts **every** Hessian eigenvalue by **exactly `2 alpha`**.

---

## 3. THE ESTIMAND, and the corners

- **Corners:** the **49 Montero-shaped n-way corners** (`exp_gen_factor_wells.nway_corners`), the row-4
  corner set, ≥ 49 as w6-common §2 requires. One request per corner (the corner's own factor values);
  the address is `gen_field.field_address`, request-only by signature.
- **Cells per corner:** 5 wells × {alpha} × {mass} × {gamma} × {amplitude} × 6 curvature modes, log-spaced
  across the well's spectrum. `mass` moves `mu^2 = hess_eig/M`, which is how `h` is swept across ~4.5 orders
  of magnitude — the range V2 Fig. 1 verifies over. The field store's raw Hessian spectrum spans only
  `[0.16, 1.0]` and would otherwise probe one band only.
- **Censoring:** V5's own protocol. A cell whose **predicted** half-life exceeds the declared cap is
  censored and not rolled; a rollout that does not halve is a censored lower bound; no half-life is quoted
  from an `n < 3` cell (V5's rule).
- ⭐ **THE INSTRUMENT, named before the run, because two exist and they are not the same
  number.** V2 Fig. 1 and V5 §D both report an **envelope half-life** and a **raw first-crossing
  time**, and state that the two *coincide on the overdamped branch and diverge above the exceptional
  point*. `n_half = ln2/(-ln|lambda|)` predicts the **ENVELOPE**: underdamped, the mode oscillates inside
  a decaying envelope, so `|x|` first touches `|x_0|/2` because of the PHASE — at a time set by the
  oscillation period — while `|lambda| = sqrt(1-gamma)` sets the envelope. ⛔ Reading the first crossing
  there and comparing it to the CRR would compare two different quantities and **manufacture a failure of
  the law**. The scored instrument is therefore the **envelope**, taken as the reverse cumulative maximum
  of `|x|` (monotone by construction; no fit, no window length to choose). The first-crossing time is
  recorded alongside on every row and reported, and **their divergence at the crossover is itself one of
  the shorts' predictions**, tested here.
- **Corner-level quantity (the scored unit):** for corner `c`,
  `e_CRR(c) = median over that corner's live cells of |log10( n_meas_envelope / n_pred_P1 )|`.

### The pre-registered falsifier the task file demands
> *"The derived law predicts the measurement no better than a fitted one-parameter curve ⇒ the physics is
> decoration on this object and you say so in those words."*

Two one-parameter rivals, each given **every advantage including sight of the held-out measurements**:

- **FIT1** — `n = C / (eps mu)^2`, `C` fitted to minimise mean-squared `log10` residual over all scored rows.
- **NAIVE±1** — the V-curve branch slopes are `+1` and `-1` (the textbook reading P6 contradicts).

Paired per corner: `delta(c) = e_CRR(c) - e_FIT1(c)`.

---

## 4. THE MARGIN AND THE BAR — declared here, before the first held-out number

**Primary bar (P1, the exact half-life).** The CRR **HOLDS** on the field store iff BOTH:

- **B1 (absolute):** `median over the 49 corners of e_CRR(c) <= 0.10` — i.e. the parameter-free prediction
  is within a factor **1.26** of measurement at the median corner. *Rationale for 0.10: the shorts verify
  on designed/trained checkpoints with an analytic tilt and report `mu^2_meas/mu^2_pred = 1.000000 ± 5e-12`
  and a fitted slope of −0.985 against −1. This object is a 2048-dimensional learned field store rolled
  nonlinearly from a finite displacement, not a single mass-whitened analytic mode; one decimal order of
  magnitude is the loosest margin at which the claim still says something, and the tightest at which a
  real object could plausibly clear it.*
- **B2 (paired, vs the fitted rival):** `mean over corners of delta(c) < 0` **and** `delta(c) < 0` on at
  least **40 of 49** corners.

⭐⭐ **THE INSTRUMENT FLOOR, measured on `val` and declared here, BEFORE the held-out run.** How well can
this measurement chain recover the CRR on a landscape where the CRR is **true by construction**? One atom
at the origin makes `E = ||q||^2/2` exactly, so `hess_eig = 1`, `mu^2 = 1/mass`, and the one-step map *is*
the shorts' Eq. (3) matrix. Any disagreement is then the **instrument**, not the physics. Measured over 17
cells spanning both bands (`exp_w6_potentials.instrument_floor_on_exact_quadratic`, and the leg recomputes
and records it on every run):

| statistic | value |
|---|---|
| median `\|log10(n_env / n_pred)\|` | **0.0349** |
| p90 | 0.0749 |
| max | 0.1601 (at `h/h* = 1.42`, i.e. **at the exceptional point**) |
| median over cells with `0.3 <= h/h* <= 3` | 0.0540 |

⭐ The worst cell sits exactly where V2 App. C says it must: at `h = h*` the matrix is **defective** (one
Jordan block, D10) with secular transients `~ n lambda^n`, so the envelope is not a pure exponential there
and `ln2/(-ln|lambda|)` under-predicts. **That is the theory explaining my instrument's worst case, not an
excuse added afterwards.**

⇒ **B1's bar of 0.10 is ~3x this floor.** ⛔ A bar tighter than ~0.035 would be measuring my rollout rather
than the law. This is why 0.10 is the right number and it is fixed here.

⚠ **A resolution limit, declared in advance so it cannot be invoked afterwards.** The half-life is read off
an integer step index, so a cell whose true half-life is ~5 steps carries ~1 step of quantisation = 0.08 in
`log10` — the width of B1's whole bar — before any physics enters. The shorts hit the same limit and censor
`n < 3`. **B1 is therefore reported twice: over all live cells (the bar), and restricted to cells with
`n_pred >= 10` where quantisation is below 0.04 in `log10` (B1-resolved, reported beside it).** ⛔ B1 over
all live cells is the bar; B1-resolved is context and never replaces it.

**Secondary bars, reported whether or not B1/B2 clear** (they are mechanism, not the headline):

- **B3 (P4, the floor):** among cells the law places in the underdamped band, the measured half-life is
  independent of curvature — `|d log10 n_meas / d log10 mu^2| <= 0.10` — and sits within 0.10 in `log10` of
  `2 ln2/(-ln(1-gamma))`.
- **B4 (P5, the critical point):** the median measured V-curve argmin lies within a factor **2** of
  `gamma_crit = 2 eps mu`, in units of `gamma_crit`.
- **B5 (P6, the branch slopes):** the derived slopes beat NAIVE±1 — `mean |s_meas - s_pred| <
  mean |s_meas - (+-1)|` on both branches.
- **B6 (P7, confinement):** `max |hess_eig(alpha) - [hess_eig(0) + 2 alpha]| <= 1e-9` (this is arithmetic
  on the measured spectrum and should sit at the float64 floor; a miss means the Hessian is wrong).

**Expected direction (charter §A.1).** I expect **B1 to be the one at risk** and B3–B6 to hold. The
prediction is a linearisation and the measurement is a finite-amplitude nonlinear rollout; the shorts
themselves censor and report where their estimator fails. ⭐ **A miss on B1 with B3–B6 holding is a
result, not a failure** — it locates the law's domain rather than refuting it, and I will report it in
those words.

---

## 5. Controls on every row (charter §C.2, w6-common §1)

| control | what it does | what it must do |
|---|---|---|
| `landscape_removed` | atoms zeroed | `E = ||q||^2/2` exactly, every `mu^2 = 1/M`, all basin structure gone |
| `landscape_shuffled` | atom channels permuted within the well | same bytes, same width, destroyed geometry |
| `pca_tensor` | the w5 incumbent content in place of the items | the law is tested on the object that actually won w5 |

⚠ **The `knn_density` trap, checked explicitly and in advance.** `knn_density` failed at 0.0203 because its
length scale came from the **spacing of stored items**. `sigma_provenance` computes, on the same atoms, the
field store's `well_sigma` beside `gen_landscape.atom_bandwidths`' two spacing rules. **Measured on the val
store before this pre-registration: `well_sigma/nn_median` = 0.871, 0.853, 0.856, 2.097, 1.210 across the
five wells** — not 1.0 on any well, and on both sides of 1. ⛔ The field store's width is a within-value
POOL spread, not an item spacing; the failure is not inherited. This is stated here so it cannot be claimed
afterwards.

---

## 6. What would prove this hub wrong

- B1 and B2 both miss ⇒ **the physics is decoration on this object**, and I say so in those words.
- The law holds only at a width or a mass nobody would deploy ⇒ report the range it holds over.
- A law I pre-register turns out not to be in the shorts ⇒ already handled: §0.3 above.
- The kernel width turns out to be an item spacing ⇒ already measured against: §5 above.

## 7. Replication (charter §A.5)

The winning configuration is re-run from this file by a separate spoke on a clean checkout of
`w6/potentials`. Unreplicated = not a result.

---

## AMENDMENT 1 — 2026-09-20T07:02:02Z, appended before the first held-out number

Charter §A.3: appended, dated, and the text above is not edited. Everything here was decided from the
**selection set (`val`)** only. ⛔ No held-out number has been computed at the time of writing.

### A1.1 B3's statistic, disambiguated
§4's B3 wrote `|d log10 n_meas / d log10 mu^2| <= 0.10` without saying over what. The slope must be fitted
**per gamma** (the floor itself depends on gamma), which gives one slope per gamma and needs a summary.
**The bar is read on the MEDIAN of `|slope|` over gammas; the MAX is printed beside it on every run**, so
the stricter reading is always visible and a reader may apply either.
⚠ And the fit uses the **global gamma grid only** — see A1.2; a refined gamma exists for a single
`(well, mass, mode)`, so a slope against `mu^2` at such a gamma is fitted on one curvature and is noise.
On `val` this mattered: including refined gammas turned a median `|slope|` of **0.0050** into **0.6177**.

### A1.2 A local gamma bracket, so B4 tests D16 and not my grid spacing
On `val` the 9-point global gamma grid steps by ~2x, and because the underdamped floor
`2 ln2/(-ln(1-gamma))` **falls** with gamma, a coarse grid's argmin is biased onto the last underdamped
point below `gamma*`. **Each mode now also gets a 5-point geometric bracket `[gamma*/2, 2 gamma*]` around
its own predicted critical damping.** ⛔ This does not put the answer in: the measured argmin remains free
to land anywhere on the union of both grids. Its sufficiency is *demonstrated on every run* — the report
prints **the PREDICTED curve's own argmin over `gamma*_exact`**, which must be ≈ 1.00 for the grid to be
adequate (on `val` it is exactly 1.00).

### A1.3 B4 is also reported against the EXACT critical damping
D16 gives `h*(gamma) = (1 - sqrt(1-gamma)) sqrt(2/(2-gamma))` and then inverts it *"at small h"* to
`gamma_crit = 2 eps mu`. Inverting exactly: `gamma* = 1 - r^2`, `r = (2 - sqrt(4 - c^2))/c`, `c = 2 - h^2`.
The two agree as `h -> 0` (0.001998 vs 0.002 at `h = 0.001`) and differ by 30 % at `h = 0.4` (0.563 vs 0.8).
**B4's declared bar (factor 2 of `2 eps mu`) is UNCHANGED**; the exact-inversion ratio is reported beside it.

### A1.4 ⭐ A REVISED EXPECTATION, declared against myself before the run
§4 said I expected **B1** to be the bar at risk and **B3–B6** to hold. **On `val`, B1 passed at 0.0293 and
B4 missed at 0.349**, and I have diagnosed why: the measured V-curve minimum cannot sit on the derived
critical point, because the **defective (Jordan-block) map at the exceptional point** makes the true
envelope decay *slower* than `lambda^n` exactly there, filling in the predicted dip. The shorts' own I-J
instrument (*"the exact one-step Jacobian"*) cannot see this — on I-J the measurement **is** the
eigenvalue, so its argmin lies on the crossover by construction — while mine is a rollout, and V5 itself
says *"the instrument gap is a level, not a rate"*.
⇒ **I now expect B4 to MISS on the held-out set, for that reason, and I say so here rather than after.**
⛔ The bar is not moved. If B4 misses as predicted, the reportable statement is: *the derived critical
point is where the eigenvalues collide and the prediction's own V-curve minimum sits on it exactly, but a
rollout's minimum does not, because the degeneracy that defines the critical point is what invalidates the
pure-exponential envelope there.* If B4 instead **passes**, that prediction of mine was wrong and I will
say so.

### A1.5 Controls
`landscape_removed`, `landscape_shuffled` and `pca_tensor` were tied to the aux-corner count and could
silently run on zero corners; they now have their own `--control-corners` (4 on the scored run).
⚠ **Restated from §5 and from the LOG: a control is NOT expected to break the CRR.** The CRR is a law about
the *dynamics* and holds of any landscape with a well-defined Hessian at a critical point. What a control
collapses is the **curvature spectrum**, and the spectrum column is where charter §C.2 is satisfied —
`landscape_removed` has `Cov_w = 0` and therefore NO non-trivial spectrum at all.
