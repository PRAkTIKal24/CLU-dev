# PREREG — `ssb-shell-atoms` (Route 2: designed degeneracy — shell atoms + the pseudo-Goldstone tilt dial)

**Written 2026-07-30, BEFORE any measured run of the shell store.** Protocol §5 pre-registration rule
(acceptance criteria here are measured ratios/slopes/laws ⇒ mandatory). Every number below is derived
from either (a) closed-form analysis of the shipped potential/integrator, or (b) an already-banked
measurement quoted with its source. Nothing below is derived from a shell-store run.

Derivation inputs (all quoted from banked outputs, not re-measured):
- `memory-gym-v0.md` §0 flag table: `d=4, m=1, n_spectator=1` on the manifold family ⇒ `dim=6`;
  `confine α = 0.05`; `atom_init_width s = 0.3`; `atom_depth_init = 1e-4`; `atom_init_scale = 1.0`;
  write = 300 steps, Adam(W) `lr 3e-3, wd 1e-4`, `n_perturb 32`, `σ_addr 0.25`, `σ_pay 0.6`,
  **`margin 0.15`**, `barrier 0.2`, `barrier_pairs "nn"`; read `dt 0.05`, `γ_address 0.05`,
  `γ_read 0.02`, 400+800 steps.
- `memory-gym-v0.md` §3.2: `manifold/base r2 = −0.1802 ± 0.1708`, `overload/base decode = 0.2593 ±
  0.0668`, `aggregate/base neg_mae = −0.5261 ± 0.0863`; echo substitute **1.0000**; manifold byte
  ratio **52.0×**, overload/base **17.1×**, aggregate **54.6×**.
- `memory-gym-v0.md` §3.5: no flat direction — `λ_min = 0.0846–0.1000 ≈ 2α = 0.10` at 14/18 sites;
  ridge write ⇒ **saddle**, `λ_min = −0.5946`, spectator participation 1.000, `r2 = −1.205`.
- `memory-gym-v0.md` §3.1: below the shipped atom budget the write does not converge
  (`final loss 0.20–0.24`, `λ_min −0.21…−1.20`); at the shipped budget `final loss 0.0002`,
  `λ_min +3.24`.
- `trainability-spike-theory.md` §Q2.4: `cond(H)` 7.69 → 9.6e7 as the pseudo-flat band closes;
  recommendation "exactly flat or comfortably massive".

---

## 0. The construction being registered (fixed here; not tunable after the fact)

**Shell atom** (charter §A4.1, verbatim form):

    V = α‖q‖²  −  Σ_j A_j · exp( −(‖q − c_j‖ − r_j)² / (2 s_j² + 1e−9) ),   A_j = amp_j², s_j = e^{log_width_j}

with `r_j = radius_scale · softplus(radius_raw_j)`, `radius_scale` a **static** float.
Implementation detail that makes the r=0 gate exact: the shell displacement is evaluated as
`u² = d2 − 2·ρ·r + r²` (with `d2 = ‖q−c‖²`, `ρ = sqrt(d2 + 1e−12)`), **not** as `(ρ − r)²`, so at
`r ≡ 0` every arithmetic op reduces to the shipped `exp(−d2/(2s²+1e−9))` bit-for-bit.

**Tilt** (charter §A4.2), rank-1, per group, envelope-weighted and **group-normalised**:

    V_tilt(q) = (ε/2) · Σ_j w_j(q) · ( û_{o(j)} · (q − c_j) )²,
    w_j = g_j / (Σ_{k ∈ group(j)} g_k + 1e−6),   g_j = exp(−u_j²/(2s_j²+1e−9)),   û = u/‖u‖

`ε` is the scalar dial (static); `u_g ∈ R^dim` is **learned** (one direction per item group).
`ε = 0` ⇒ the term is not evaluated at all (bit-identical to no tilt).

**Arms** (declared now; nothing added after seeing a result):
`gauss` (control, today's store) · `shell_r0` (`radius_scale = 0`, the regression gate) ·
`shell` (learned `r_j`, `radius_scale = 1`, init `r ≈ 0.5`) · `shell_fixed` (`r` **designed** at 0.5,
radius updates frozen by the write mask — the w20-doctrine arm: designed degeneracy, learned
placement) · `shell+tilt(ε)` on `shell_fixed` over the ε grid below.

**ε grid (declared, ≥3 values spanning ≥2 decades, with a destructive anchor):**
`ε ∈ {0 (mandatory zero), 1e−3, 1e−2, 1e−1, 1.0, 10.0}` — 5 non-zero values spanning **4 decades**.
`ε = 10.0` is the **liveness anchor**: it is `≈ λ_massive` (predicted 3–11, see P3) and is expected to
be visibly destructive.

---

## 1. P1 — the r=0 regression gate (BLOCKING, §6 falsifier)

| id | prediction | tolerance |
|---|---|---|
| **P1a** | `V_shell(q; r≡0) − V_gauss(q)` = **0.0 exactly** (bit-identical), all q in a 512-point random batch, float32 **and** float64 | **exact 0**, no tolerance |
| **P1b** | `∇V` bit-identical | exact 0 |
| **P1c** | `Hess V` bit-identical | exact 0 |
| **P1d** | the **written store** after a 300-step masked write is bit-identical (all leaves) | exact 0 |
| P1e | with `radius_scale = 1` and `radius_raw → −∞` the shell converges to the Gaussian | rel. err < 1e−5 at `softplus⁻¹(1e−4)` |

**Derivation.** `u² = d2 − 2ρr + r²` with `r` an exactly-zero array gives `d2 − 0.0 + 0.0 = d2`
identically in IEEE754 (subtracting/adding +0.0 is exact); `ρ` is finite because of the `+1e−12` under
the sqrt, so the reverse-mode contribution `−2r·(diff/ρ)` is an exact `0.0` rather than `0·NaN`.
**Risk I am registering in advance:** if XLA fuses/reassociates the added ops, one ULP could appear.
If it does, P1 is reported **FAILED at bit level** with the observed max-|Δ| and the gate is treated as
failed (§6 of the task) — I will not silently relax it to a tolerance.

## 2. P2 — the tilt dial: `λ_min` tracks ε (the conditioning claim)

At a written site `z` sitting on the shell with `û·(z−c_j) ≈ 0` and `Σ_j w_j = 1`, the tilt's Hessian
contribution is exactly `ε ûûᵀ` (all other product-rule terms carry a factor `û·(q−c_j) = 0`). Hence:

| id | prediction | point | range |
|---|---|---|---|
| **P2a** | `λ_soft(ε) = λ_soft(0) + κ·ε` with **κ = 1.00** (the normalisation is designed to make it exactly 1) | κ = **1.00** | [0.5, 2.0] |
| **P2b** | log–log slope of `(λ_soft(ε) − λ_soft(0))` vs `ε` over `ε ∈ [1e−3, 1e−1]` | **1.00** | [0.85, 1.15] |
| **P2c** | `λ_min > 0` at every registered `ε ≥ 1e−3` on the `shell+tilt` arm | — | binary |
| **P2d** | `λ_soft(0)` on the `shell_fixed` arm — the *residual* breaking from the shipped confinement — equals `2α‖c‖/ρ` (derived below) | **0.10** | [0.02, 0.40] |

**Derivation of P2d (new, and it is the honest counterweight to the whole route).** A shell atom is
**not** exactly degenerate inside the shipped confinement. For `V = −A e^{−u²/2s²} + α‖q‖²` the
stationarity condition on the shell gives, for the tangential curvature at the selected vacuum
(`n̂ = −ĉ`):

    λ_tan = 2α − A f'(u)/ρ = 2α·‖c‖/ρ ,   ρ = ‖q−c‖ ≈ r.

So the confinement **already** tilts the shell by `2α‖c‖/r`, i.e. the designed flat direction is
pseudo-flat before any `ε` is added. With `α = 0.05`, `r = 0.5` and atom centres landing at
`‖c‖ ≈ 0.5` (the write pulls a group's atoms to within ~`r` of its own site, whose address norm is
≈1 on the unit `d`-ball but whose atom-centre norm after a 300-step write is not) the point estimate
is `2(0.05)(0.5)/0.5 = 0.10` — **numerically the same 2α floor the Gaussian store already has.**
⇒ Registered consequence: **`ε` only becomes the dominant breaking above ≈0.1**, and any claim that
the tilt "restores conditioning" below that is spurious. This is a falsifiable prediction about the
*route*, not about the code.

## 3. P3 — the curvature hierarchy and the soft-mode participation ratio

| id | prediction | point | range |
|---|---|---|---|
| **P3a** | `λ_massive` (radial mode) `≈ A/s² + 2α` at a written site | **3.3** | [0.5, 12] |
| **P3b** | hierarchy `λ_massive / λ_soft ≥ 10` at `ε ≤ 0.1` on `shell_fixed(+tilt)` | ≥10 | ≥3 |
| **P3c** | participation of the softest eigenvector on the **designed shell coordinate** (the learned tilt direction `û`, and separately the spectator axis `e_spec`): `⟨v_min, û⟩² ≥ 0.7` at `ε ∈ [1e−3, 1e−1]` | 0.85 | ≥0.5 |
| **P3d** | `λ_min ≥ 0` at the written site on `shell_fixed` — i.e. a **valley, not a saddle**, in explicit contrast with the ridge write's **−0.5946** | **+0.02** | [−0.02, +0.15] |
| **P3e** | at `ε = 10` the hierarchy **collapses** (`λ_soft ≳ λ_massive`) and decode/`r2` degrade — the declared destructive liveness anchor | — | binary |

`A/s²`: the shipped write reaches `λ_min = +3.24` at the shipped atom budget (gym §3.1), which is
`A_eff/s² + 2α` for the Gaussian ⇒ `A_eff/s² ≈ 3.14`; I carry that across as the point estimate.

## 4. P4 — ⭐ the lifetime dial (`ε` = manifold-payload drift timescale)

The soft mode under the shipped read is a damped harmonic oscillator `q̈ + Γq̇ + λq = 0` with
`Γ = γ/dt`. In phase 2, `γ_read = 0.02`, `dt = 0.05` ⇒ **`Γ = 0.4`**. The damping regimes give the
**exponent, and a knee the charter did not predict**:

    overdamped (λ < Γ²/4):   τ = Γ/λ            ⇒ d log τ / d log ε = −1
    underdamped (λ > Γ²/4):  τ = 2/Γ = 5.0      ⇒ d log τ / d log ε =  0

| id | prediction | point | range |
|---|---|---|---|
| **P4a** | log–log slope of `τ(ε)` (steps for an on-shell offset to decay to `1/e`) over `ε ∈ [1e−3, 1e−2]` | **−1.00** | [−1.15, −0.85] |
| **P4b** | ⭐ a **knee** at `ε* = Γ²/4 = 0.04` (charter §A4.2 predicts a pure `1/ε` with no knee — I am registering *against* the charter here) | **ε\* = 0.04** | [0.01, 0.15] |
| **P4c** | slope over `ε ∈ [0.1, 10]` | **0.00** | [−0.25, +0.25] |
| **P4d** | `τ` floor `= 2/Γ = 5.0` time units `= 100` steps | **100 steps** | [60, 180] |
| **P4e** | the read budget is 800 phase-2 steps ⇒ the on-shell payload **survives the read** for `ε ≤ 1e−2` (`τ ≥ 40` time `= 800` steps) and is **erased** for `ε ≥ 0.1` | — | binary |

## 5. P5 — write convergence under a designed flat direction (the structural prediction)

`write_loss`'s `L_min` term charges `relu(V(z) − V(z+δ) + margin)`. Along an **exactly flat** direction
`V(z+δ) = V(z)` ⇒ the violation is exactly `margin = 0.15` for every perturbation that lies in the flat
subspace. With `dim = 6` and one designed flat direction the flat fraction of an isotropic perturbation
set is `1/6` of the variance, but the relu is on the *scalar* value, so the expected `L_min` floor is
`≈ margin · P(δ mostly tangential)`.

| id | prediction | point | range |
|---|---|---|---|
| **P5a** | ⭐ `final_write_loss(shell_fixed) − final_write_loss(gauss) > 0` — **the shipped write objective structurally penalises designed degeneracy** | **+0.05** | [+0.005, +0.15] |
| **P5b** | `final_write_loss(shell_fixed)` in absolute terms | **0.10** | [0.02, 0.30] |
| **P5c** | ⭐ **the `shell` arm's learned `r_j` collapses toward 0** under the endpoint write objective (w20: free learning erases design) — group-mean `r` after 300 steps | **< 0.15** | [0.0, 0.35] |
| **P5d** | consequently `r2(shell) ≈ r2(gauss)` within noise | Δ = 0.00 | [−0.25, +0.25] |

**Registered admissibility convention (declared BEFORE the run, per task §1(i)).** A flat direction
keeps `L_min` at the `margin` floor by construction, so **loss-plateau is not a valid non-convergence
test on the shell arms.** `gate_admissible` is therefore defined **spectrally**: `λ_min ≥ −1e−3` at
every recorded site **and** `‖∇V(z)‖ < 0.1`. `final_write_loss` is reported for every cell either way,
and the Gaussian arms are additionally reported against the gym's own `≤0.05` loss convention so the
two conventions can be compared.

## 6. P6 — per-family dividends per arm (the race card)

`dividend = full − settle_deleted_launder`. The launder is arm-independent (same codebook), so these
are predictions about `full`. All at 3 seeds, `SE = sd/√3`.

### manifold (metric `r2`; launder ≡ 0.0000; **+0 B echo substitute ≡ 1.0000**)

| arm | predicted `r2` (point) | range | predicted dividend |
|---|---|---|---|
| `gauss` | **−0.180** | [−0.40, +0.05] | −0.180 (reproduce gym) |
| `shell_r0` | **−0.180**, `|Δ| < 1e−6` vs `gauss` | exact | −0.180 |
| `shell` (learned r) | **−0.180** | [−0.45, +0.10] | ≈ gauss (P5c/P5d) |
| **`shell_fixed`** | **+0.10** | [−0.60, +0.60] | **+0.10** |
| `shell+tilt(1e−3)` | +0.10 | [−0.60, +0.60] | +0.10 |
| `shell+tilt(1e−2)` | +0.08 | [−0.60, +0.55] | +0.08 |
| `shell+tilt(1e−1)` | −0.05 | [−0.70, +0.35] | −0.05 |
| `shell+tilt(1.0)` | −0.15 | [−0.9, +0.1] | −0.15 |
| `shell+tilt(10.0)` | **−1.0** (destructive anchor) | [−4, −0.2] | −1.0 |

**Derivation of the `shell_fixed` point estimate.** The manifold query launches at
`q0 = (c_addr, 0, spec)` and the target is `spec`. Under a shell of radius `r` centred at
`c = (c_addr, a, 0)` the overdamped settle lands at `c + r·(q0−c)/‖q0−c‖`, so the settled spectator is
`r·spec/√(a² + spec²)` — **monotone in `spec`** (the Gaussian's is ≡0). With `|a| ~ U(0,1)` and
`spec ∈ [−0.6, 0.6]` a least-squares slope of ~1 is attainable at `r ≈ 0.6`, but the map is
**saturating**, so `r2` cannot reach 1. Point estimate `+0.10`, and the *directional* claim I am
registering is **`r2(shell_fixed) > r2(gauss)`**, i.e. `Δ = +0.28`, range `[−0.4, +0.8]`.
⚠ **A ≤0 dividend here does not falsify the route** (task §6) — the echo substitute is 1.0000 by
construction, so the substitute margin will be **≈ −0.9 on every arm** and every manifold "proceed"
is at best a **weak proceed** (charter §A6). I register that in advance.

### overload (metric `decode`; the store change must not destroy addressing — task §6 falsifier)

| arm | predicted `decode` | range | note |
|---|---|---|---|
| `gauss` | 0.259 | [0.15, 0.40] | gym reproduction |
| `shell_r0` | 0.259, exact match | exact | gate |
| `shell` | 0.259 | [0.10, 0.40] | r collapses ⇒ ≈ gauss |
| **`shell_fixed`** | **0.10** | **[0.00, 0.28]** | the settle lands anywhere on a sphere ⇒ the payload coordinate is no longer pinned |
| `shell+tilt(1e−2)` | 0.11 | [0.00, 0.30] | |
| `shell+tilt(1.0)` | 0.16 | [0.00, 0.35] | tilt re-pins the payload axis if `û` learns it |
| `shell+tilt(10.0)` | 0.05 | [0.00, 0.26] | destructive anchor |

⛔ **Registered falsifier range:** the §6 falsifier ("shell atoms destroy addressing") fires if
`decode(shell_fixed)` on `overload` falls **below 0.00**, which is impossible, so I restate it
operationally: it fires if `decode(shell_fixed) < 0.10` **and** `acq < 0.5`, i.e. the item is not even
acquired. Predicted: **it fires** with probability ~0.5 — the basis change is expected to cost
addressing, and I am saying so in advance.

### aggregate (metric `neg_mae`)

| arm | predicted `neg_mae` | range |
|---|---|---|
| `gauss` | −0.526 | [−0.70, −0.40] |
| `shell_r0` | −0.526 exact | exact |
| `shell_fixed` | **−0.65** | [−0.95, −0.45] |
| `shell+tilt(1.0)` | −0.60 | [−0.95, −0.42] |

## 7. P7 — the byte ledger (declared before the run)

Per-atom float count: Gaussian `dim + 2` (centres, `log_width`, `amp`); shell `dim + 3` (+`radius_raw`).
Tilt adds `dim` floats **per group**, not per atom (rank-1, group-shared).

| id | prediction |
|---|---|
| **P7a** | shell byte overhead = `1/(dim+2)` of the atom bytes = **+12.5 %** at `dim=6` (manifold), **+14.3 %** at `dim=5` (overload/aggregate) |
| **P7b** | manifold byte ratio: `52.0× → 58.4×` (`ratio = (A(dim+3) + d)/(d+m)`, `A = 32`) |
| **P7c** | tilt overhead over `shell`: `+dim·G` floats total ⇒ **< +0.5 %** of the ratio (manifold: `58.4× → 59.6×`, i.e. +2.1 %) |
| **P7d** | `matched = False` on every cell (architectural, ≥2.20× — gym PREREG-B1); the shell **raises** the floor to `≥2.40×` at `dim=5` |

## 8. P8 — compute cost (task §3: report if `∇V`/`Hess V` cost > 2× the Gaussian)

| id | prediction | point | range |
|---|---|---|---|
| **P8a** | shell `V`-eval wall-clock / Gaussian `V`-eval | **1.15×** | [1.0, 1.6] |
| **P8b** | shell+tilt `V`-eval / Gaussian | **1.5×** | [1.0, 2.5] |
| **P8c** | full race-card run (≈81 cells) wall-clock | **45 min** | [20 min, 3 h] |

## 9. What would make me report a NULL rather than a result

- P1 fails ⇒ **stop**, no science cell runs (task §6, blocking).
- The tilt produces no measurable `λ_min` response at **any** registered `ε` (P2b slope < 0.1 and no
  observable moves by >3× its noise) ⇒ **§A4.2's mechanism is refuted on a real store**; that is the
  headline, per task §6, and it is *not* buried.
- Every family has zero `gate_admissible` cells after one declared budget escalation ⇒ **ABSTAIN**
  (neither blocks B′ nor supports "proceed"), reported with admissible-cell coverage at the top.

## 10. Declared NOT-RUN unless time allows (never reported as a null)

- The **2×2 combined cell** (Route 1 × Route 2) unless `traj-write-objective`'s write-objective commit
  has landed on `main` before my final race run (task §D4).
- `recency` family (Route 1 owns its D4 diagnostic; it is out of the gate until then).
- Rank > 1 tilts; per-atom (rather than per-group) tilt directions; learned `ε`.
- Shell atoms at the **shipped** atom budget (478×) if the wall-clock budget does not allow it; if it
  does run, it is declared as an **extra** arm and is not substituted for the gym-comparable budget.
