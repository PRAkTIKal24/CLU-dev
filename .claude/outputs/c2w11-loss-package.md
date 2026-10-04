# c2w11-loss-package — physics-theorist report

Task + acceptance criterion: formalize §A34.9 (a)–(f) as differentiable objects with stated gradient
paths / coefficient-zero bit-identities / designed negatives, write the §A34.5 kinetics scoping note
(derive `ζ` in shipped constants, state the spectral-mass separation condition two-sidedly, state the
flat-floor stopping criterion), and **price** the open §A31.2 mechanism question. **Status: done.**

## ⚠ RECONCILIATION LIST — needs a Hub owner (protocol §5 corollary, in the first 10 lines)
| # | site | what must change | owner |
|---|---|---|---|
| **R1** ⛔⛔ | `PREREG-TierII.md` §7 registered operating point | **`d/s ∈ [2.5, 2.9]` AND "depth heterogeneity ≥ 3×" are INTERNALLY INCONSISTENT in a 1-D single-atom toy:** the shallow neighbour's *minimum ceases to exist* at depth ratio **1.60 / 2.22 / 3.27** for `d/s = 2.5 / 2.7 / 2.9` (`s4_depth_boundary.json`). O2's `δ = d/2` criterion over-predicts the safe ratio by **1.9–7.3×**. Term (b) (refresh) manufactures exactly this heterogeneity. **FLAG, do not edit — spoke A must re-measure on the learned store before term (b) ships.** | Hub → spoke A |
| **R2** ⛔ | charter §A31.2 / Add.11 erratum 2, *"comfortable geometry, σ_q/spacing = 0.32"* | **The sentence is undefined between two readings and they differ by the whole effect size.** NORM reading ⇒ 1-NN = **1.0000 in 27/27 cells** (d∈{12,32,64}×K∈{16,32,128}×ratio∈{0.19,0.32,0.37}); PER-COORDINATE reading ⇒ 1-NN = **0.74–0.84** at d=12, which is where the banked launder (0.79–0.90) actually sits. State which. | curator + spoke A |
| **R3** | `chlu/core/placement.py` / the placing write's `centers[:, :addr] = z[:addr] + jig` | term (a) is **unauditable** until the shipped write uses `c = stop_gradient(z_addr) + jig` and `L_org` is computed from a second expression on the live `z`. Precise spec in §1(a); no code touched by me. | experiment-engineer |
| **R4** | charter §A34.5 *"light = underdamped, heavy = overdamped"* | correct, but **the boundary is MODE-dependent, not particle-dependent**: `λ_crit = Γ_c²m/4` = `0.2630 m` (γ=0.05) / `0.04082 m` (γ=0.02). At m=1 a written well mode (λ≈9.87) is underdamped and the `2α` floor mode (λ=0.10) is **over**damped at γ=0.05 and **under**damped at γ=0.02. | curator |
| **R5** | `PREREG-C2W11` §6 M7's *"`τ_max = Γ/2α` travels with any lifetime statement"* | now carries a number: **`τ_max` = 205.2 steps** (m=1, γ=0.05) / 80.8 steps (γ=0.02); the shipped read is **11.85 τ_max** and erases a `2α`-floor offset to **3.49e-5**. ⇒ **term (c) has no read-side consumer at the shipped budget.** | Hub |
| **R6** | `bprime-theory` T3's `C ≡ Σ N_p ln(1/ρ_p)` | **scope it to the UNDERDAMPED regime.** `|ρ| = √(1−γ)` exactly only when `λ > λ_crit`; for overdamped modes `ρ₊ > √(1−γ)` and `C` is smaller (measured `C_floor` ⇒ retention 3.49e-5 vs `e^{−18.34}` = 1.08e-8, a **3218×** gap). My independent reproduction of the stiff-mode `C` = **18.3397** (T3 quotes 18.34). | curator |

---

## ⭐ DIAL DECLARATION (protocol §7) — echoed before the first result
- **Dial:** **none — theory + scoping.** ⛔ No claim cell, no performance number, no verdict of any
  kind. Every numerical statement below is a **toy** and is labelled as one.
- **Laundering control:** N/A.
- **Falsifies:** a term I could not write as a differentiable object with a stated gradient path, or
  whose designed negative I could not specify, is reported **NOT FORMALIZED**. *(Outcome: 6/6
  formalized; 4 named DEPLOYMENT BLOCKERS filed separately so "formalized" is not read as "shippable".)*
- ⛔ Depth is not feature importance (§A23.5 ACTIVE). ⛔ Wells are never named semantically.
- ⛔ **Every derivation here is a 1-D / 2-D single-width designed-well toy.** `bprime-theory` §9.2's
  standing bracket applies verbatim: **transfer to a learned multi-atom store is BRACKETED, NOT
  MEASURED.**

## 0. Flag provenance (every number in this report)
| item | value |
|---|---|
| repo | **read-only**, `main @ 7fcef50`, clean tree, **no worktree, no branch, zero tracked-code edits** |
| code | pure **numpy 2.4.1 / scipy 1.17.0**, main venv `/Users/user/Desktop/CHLU/.venv` (py 3.11.13), **float64** throughout. ⛔ **sympy is NOT installed in this venv** — all symbolic work is by hand, verified numerically to machine precision. No repo module imported anywhere; no JAX ⇒ the w6 worktree-JAX hazard does not apply |
| integrator | the shipped damped velocity-Verlet reproduced line-for-line from `chlu/core/integrators.py:64-77` (`p_half = p − ½dt ∂_qH`; `q' = q + dt ∂_pH`; `p' = p_half − ½dt ∂_qH(q')`; **`p' ← (1−γ)p'`** — a *per-step multiplier*, not a `−γp·dt` force) |
| constants | `α = 0.05` ⇒ `2α = 0.10` · `s = 0.32` · `dt = 0.05` · two-phase read `(γ,N) = (0.05,400) → (0.02,800)` · `A = 1.0` for the reference well ⇒ `λ_well = A/s² + 2α = 9.8656` |
| seeds | `default_rng(0…2)` per script; 3 repeats on every sampled statistic |
| scripts | `.claude/scratch/c2w11-loss-package/`: `s1_damping.py · s2_spectral_mass.py · s2b_fixes.py · s3_flat_floor.py · s4_depth_boundary.py · s5_a31_geometry.py · s5b_anisotropy.py · s6_emit_gate.py · s7_confinement_bias.py`, each writing its own JSON (copied to `.claude/outputs/c2w11-loss-package/`) |
| PREREG | `.claude/outputs/c2w11-loss-package/PREREG.md`, written **before any script existed**; scorecard in §5 (**7 hit · 1 MISSED, reported as a finding**) |
| wall | ≈ 3 min total across all scripts (one earlier run of `s2b` was killed at a 20001² pairwise matrix — v2 subsamples; the failure is recorded in the script docstring) |

---

# DELIVERABLE 1 — the six-term loss package

> **Gate artifact: `.claude/outputs/c2w11-loss-package/LOSS-PACKAGE-DONE.json`.**
> `loss_package_complete` is computed **mechanically** as `all(t["formalized"] for t in TERMS)`
> in `s6_emit_gate.py` — **`true` (6/6)**. ⭐ **It also carries `deployment_blockers` (4 items,
> DB1–DB4). "Formalized" ≠ "shippable" and the JSON says so in its own fields.**

## Summary table

| term | symbol | gradient channel | live before wells? | designed negative (headline) | 2α / band interaction |
|---|---|---|---|---|---|
| **(a)** organization | `L_org` | **DIRECT ALGEBRAIC** through the placing write `c = z_addr + jig` — neither implicit nor trajectory | ⭐ **YES — the only one** | shuffled pair targets ⇒ chance; **+ the two-sided channel-attribution assertion** | the **only** term that controls `d/s`; needs a two-sided hinge at 2.5 / 2.9 |
| **(b)** sharing/refresh | `L_share` | DIRECT on `θ` (depth is explicit in `θ`); K9 owns routing | no (needs a 2nd encounter) | private-well-per-item writer must **FAIL** M4, not SKIP | ⛔ **manufactures the heterogeneity that annihilates neighbours at ratio 1.60–3.27 in-band** |
| **(c)** curvature-shape defender | `L_shape` | DIRECT on `θ` via Hellmann–Feynman, `q*` stop-gradiented | no (vacuum has no site info) | coefficient-0 ⇒ **bit-identical** + **vacuum-masquerade control** + capture control | `λ_min ≥ 2α` is a **hard ceiling** ⇒ `τ_max` = 205 steps; shipped read = 11.85 τ_max |
| **(d)** set-level read | `L_read` | **implicit-at-settle** to `θ` (O(1)); **EXACTLY 0** to `φ`/launch at settle (T3) | no | K4's four legs at **full ψ capacity** + SP-1 param bound | `d/s ≥ 4.0` ⇒ read is exactly VQ ⇒ table-expressible |
| **(e)** calibration | `L_cal` | direct to the head; direct+implicit to `θ` | no | ⭐ **query-only head at chance is STRUCTURAL, not measured** | ⛔ **conflicts with (c)**: if (c) works, `λ_min` stops being a novelty feature |
| **(f)** kinetics | `L_kin` | **trajectory / finite-budget ONLY**; `∂z*/∂{γ,M} ≡ 0` bitwise at settle | no | settled-point gradient must be **exactly 0.0 bitwise** | mass is a selector **only** where `λ < λ_crit = Γ_c²m/4` |

## ⭐⭐ The staging statement (the package's main hazard, as the task flagged)
Banked and load-bearing (`orgdiv-cat-test` §5.1): at init the **implicit** channel is
`7.88e-10 / 1.52e-9 / 7.26e-9` and the **trajectory** channel `1.67e-10 / 3.17e-10 / 1.57e-9` —
**six to seven orders below the trajectory reference scale `6.421e-3`** — and both go to **O(1)**
(`1.4664 / 1.1318 / 1.2450`) only after the write. **The write is the precondition for gradient to
exist at all.**

> **Consequence, and it is the single most actionable line of Deliverable 1:**
> **only term (a) escapes this.** `L_org`'s gradient does not route through a settled read, an
> implicit solve, or a dug well — it routes through the *algebraic* placement assignment
> `c = z_addr + jig`, which is O(1) from step 0. ⇒ **STAGE 0 = term (a) ALONE.** Terms (b)–(f) are
> switched on only after the first write epoch has dug wells that pass K1. Any schedule that turns
> (d) on at step 0 is optimising round-off, and the program has already paid for that lesson twice
> (w19/N61; the "weak φ" failures, retro-explained by T3 as `C ≈ 18` failures).

---

## (a) the label-free organization term — ⚠ the term that most needs care

**Expression (program notation).**
```
c(x) = z_addr(x) + jig                                     # the placing write's assignment
L_org = E_{(x,x')~P_pair} ℓ_metric( ‖c(x) − c(x')‖ / d_ref , t(x,x') )
      + λ_band · [ relu(2.5 − d/s)² + relu(d/s − 2.9)² ]   # on MEASURED d and MEASURED s
```
`ℓ_metric` = InfoNCE / NT-Xent over the **placed atom-group centroids**; `t` = a **label-free** pair
target (augmentation-positive / stream co-occurrence). §A34.6's ruling holds: the organization
objective is label-free; supervision enters only at the read head.

**(ii) Parameters and channel.** `φ` (address block), the jig/placement head, atom centres `c_j`
directly, and `log s_j` **only** through `L_band`. Channel = **NEITHER implicit-at-settle NOR
trajectory**: a direct algebraic path `∂L/∂c · ∂c/∂z_addr (= I) · ∂z_addr/∂φ`.

> ⭐ **Honesty note, stated because the alternative over-claims.** This is **not a physics gradient**.
> `c = z + jig` is algebraic. The *physics* enters only through **what `ℓ_metric` and `L_band`
> measure** — an admissible, band-compliant, shareable placement. The correct statement of the
> "physics-as-trainer bet" is: **the store's own admissibility functionals act as a regulariser on φ.**

**(iii) Coefficient-zero bit-identity requirement (three conditions, all necessary).**
1. The pair sampler and every augmentation draw come from a **dedicated RNG stream**, so
   `λ_org = 0` leaves every other draw bit-identical. *(This is the usual silent killer: a term that
   consumes RNG changes the whole run at coefficient zero.)*
2. The **shipped store write** uses `c = stop_gradient(z_addr) + jig` — the forward **value** is
   unchanged (so the store is bit-identical), and the accidental outer-loss path is severed.
3. `L_org` is computed from a **second expression** `z_addr + jig` on the **live** `z`.
Assert: `θ_after_one_step(λ_org = 0) == θ_after_one_step(term deleted)`, **bitwise**, 3 seeds.

**(iv) ⭐⭐ The designed-vs-accidental separation — exactly what an engineer must assert.**
§A28.1 is explicit that the accidental leak and the designed trainer are **different objects**; they
currently share **one line of code**. Three assertions, and all three are needed:
```
N-a2(i)   grad_norm(L_org , φ)                                   >  0            # designed channel LIVE
N-a2(ii)  grad_norm(L_org , φ | stop_gradient on the placement)  == 0.0 bitwise  # and it is the ONLY path
N-a2(iii) grad_norm(L_outer, φ) == grad_norm(L_outer, φ | placement cut), bitwise # accidental channel DEAD
```
(iii) is the load-bearing one. Its **live** reference value is banked: **0.0908 → 0.0659, i.e. 27 %
of layer-0 φ gradient flows through the write whenever `atom_place_radius > 0`** — the same coupling
that erodes the store. If (iii) fails, the arm is running the *unaudited* channel and any organization
result is attributable to the outer loss, not to the write. **DB3: (iii) cannot pass until the
`stop_gradient` lands. This is a code change I specify and do not make.**

**(v) 2α / band.** `L_org` is the **only** term that can control `d/s`, because `d/s` is set by
placement spacing and width, both write-side. Two-sided hinge mandatory: below `d/s = 2.01` two equal
wells **merge into one minimum** (axial curvature at the midpoint is exactly `2α = 0.1000` at
`d/s = 2.00`); above `d/s = 4.0` the settled-point organization is **exactly nearest-centroid VQ**
(`D = 0.0000`) and the dividend is structurally zero. **`α` is a ceiling:** lowering it lengthens
soft-mode lifetimes (`τ_max = Γ/2α`) but removes the coercivity the write and the admission sizing
depend on — and it is what floors `λ_min`, i.e. what makes an undug site report `λ_min ≈ 0.0993`.

**Staging: live at step 0. This is the bootstrap term.**

---

## (b) the sharing / refresh term

**Expression.** Depth `D_w(θ) = Σ_{j∈group(w)} amp_j²`.
- **STRUCTURAL form (preferred, and it is what I recommend):** a refresh writes
  `amp_j ← √(amp_j² + δ_j)` with `δ_j ≥ 0` by parameterisation ⇒ **I1 (refresh-on-rewrite
  monotonicity) holds as an IDENTITY, not as a penalty.** A loss that *rewards* monotonicity can be
  traded away by any other term; a parameterisation cannot.
- **LOSS form** (sets *how much*, never the sign):
```
L_share = Σ_w softplus(D_target − D_w)
        + μ Σ_w relu(D_w⁻ − D_w⁺)                                      # I1 backstop
        + ν Σ_w relu( log D_w − log D_min_neighbour − log R_max )      # ⛔ THE GUARD — see below
```
**(ii)** Touches `amp_j` of the merged group, `log s_j` (capped), and **freezes `c_j` during a
refresh** (refresh *deepens*, never *moves* — moving is what breaks G-ADDR's designed targets).
Channel: **direct on `θ`**. K9 owns the merge/spawn **routing**.

**(iii) Coefficient-zero.** K9's decision must be a deterministic function of
`(query, store, K9 threshold)` only. Assert the emitted `WritePlan` at `λ_share = 0` is equal field
by field to the term-deleted plan, and `θ` after one write is bit-identical. *(If `λ_share` can move
the routing, the coefficient-zero arm is a different experiment, not a control.)*

**(iv) Designed negatives.** **N-b1:** a store configured to spawn a **private well per item** must
**FAIL** M4 — note the trap: with no rewrite events the "monotone on ≥ 90 % of events" leg is
*vacuous*, so the leg must report **FAIL, not SKIP** (this is exactly C2W8's thrice-caught
vacuous-gate defect class). **N-b2:** for every rewrite event, `D_w(θ⁺) ≥ D_w(θ⁻)` **bitwise**, no
tolerance, since the structural form makes it an identity.

**(v) ⛔⛔ 2α / band — the measured hazard, and it is new.**
Refresh *manufactures* depth heterogeneity (frequent features deepen). Heterogeneity moves the basin
boundary by O2's `δ = ln(A_i/A_j)/(d/s² − 4/d)`. Measured, `s4_depth_boundary.json`, 1-D two-Gaussian
+ confinement toy at `s = 0.32, α = 0.05`:

| `d/s` | O2 `δ` at ratio 3 | measured separatrix offset | measured/predicted | **measured depth ratio at which the shallow MINIMUM CEASES TO EXIST** | O2's `δ = d/2` criterion |
|---|---|---|---|---|---|
| 2.5 | 0.3906 | *(annihilated)* | — | **1.60** | 3.08 |
| 2.7 | 0.2885 | *(annihilated)* | — | **2.22** | 5.18 |
| 2.9 | 0.2312 | **0.2699** | **1.167** | **3.27** | 9.07 |
| 3.5 | — | — | — | 14.30 | 61.87 |
| 4.4 | — | — | — | 294.7 | 2164.6 |

O2's *offset* is good to **2–17 %** at ratios 1.5–3 (|measured/predicted| = 1.02 / 1.04 / 1.17), but
its `δ = d/2` **annihilation** criterion over-predicts the safe heterogeneity by **1.9–7.3×** — the
saddle-node collision happens earlier because the shallow minimum itself migrates into the deep
neighbour's negative-curvature shoulder. And the capture radius collapses first:

| depth ratio at `d/s = 2.7` | shallow half-basin | deep half-basin | shallow/deep | shallow `λ_min` |
|---|---|---|---|---|
| 1.0 | 0.3993 | 0.3993 | 1.000 | 7.790 |
| 2.0 | **0.1338** | 0.6254 | **0.214** | 3.655 |
| 3.0 | — | — | **ANNIHILATED (1 minimum)** | — |

> ⭐⭐ **Two consequences.**
> **(1) R1 — the registered operating point is internally inconsistent (in this toy).**
> `PREREG-TierII` §7 asks for `d/s ∈ [2.5, 2.9]` **and** "depth heterogeneity ≥ 3× between
> neighbouring wells" (Prop O4 needs it for non-VQ mass). At `d/s = 2.7` a 3× ratio leaves **one
> minimum**. Only `d/s = 2.9` survives 3× — and there the shallow half-basin is **0.1076**, while
> `σ_q ≈ 0.32 × spacing = 0.297` ⇒ **SC-6 (`capture ≥ σ_q`) fails for the shallow well.**
> **(2) The `ν`-guard is not optional.** `R_max` must be set from the *measured* annihilation ratio
> on the learned store, not from O2's formula.
> ⚠ This also supplies a mechanism for the banked pathology **"wells with `λ_min > 0` and capture
> radius exactly 0.000"**: a shallow neighbour of a refreshed well keeps a minimum long after its
> basin has become smaller than `σ_q`.

**Staging: live from the second encounter of a feature.**

---

## (c) the curvature-shape term — the w20 defender (⛔ the refuted tilt object is NOT re-derived)

§A4.2 refuted the **tilt** instantiation on a learned store (tilt monotonically *reduces* `λ_min`,
`+0.099 → −8.28`, two implementations, every family; a written site's vacuum residual 0.140–0.343 vs
a 0.167 random baseline; `λ = ε` holds only in the single-atom geometry it was specified in). **I do
not use it.** What follows is a *spectral-shape defender*: its claim is M8-measurable and its failure
modes are visible.

**Expression.** At each written site `w`, `H_w = ∇²V_θ(q*_w)` (analytic; `q*_w` carries
`stop_gradient`), eigenvalues `λ_1 ≤ … ≤ λ_D`. Use smooth surrogates
`λ_soft = −(1/β) log Σ_i e^{−βλ_i}` and the analogous second-softmin `λ_2nd`:
```
L_shape = Σ_w [ (λ_soft(H_w) − (2α + ε_soft))²          # pull the softest mode toward the floor
              + relu(λ_stiff_target − λ_2nd(H_w))²      # but keep the SECOND mode stiff
              + relu(D_min − D_w)²                      # ⛔ the site must actually be DUG
              + relu(σ_q − r_capture,w)² ]              # ⛔ and must still CAPTURE
```
**The last two terms are what make the claim non-vacuous.** Without them `L_shape` is minimised by
an **undug** site, which reports `λ_min = 2α` for free — that is precisely M8's banked trap
(*"undug wells report `λ_min ≈ 0.0993` because `2α` is what `λ_min` reports when nothing was
written"*).

**(ii) Channel.** DIRECT on `θ` by **Hellmann–Feynman**: `∂λ_i/∂θ = u_iᵀ (∂H/∂θ) u_i`, valid where
`λ_i` is simple; the softmin surrogate removes the eigenvalue-crossing non-differentiability.
`q*_w` is stop-gradiented (the implicit correction `∂q*/∂θ` is available but second-order here, and
its omission is **declared**, not silent). Touches the **within-group second-moment tensor** of atom
centres — an anisotropic well is made by *where the group's atoms sit*, not by tilting `V`.

**(iii) Coefficient-zero.** The eigendecomposition must sit inside a `λ_shape > 0` branch or be
`jnp.where`-guarded so a degenerate spectrum cannot inject NaN via `0 × NaN`. Assert:
`λ_shape = 0` ⇒ **bit-identical to the shipped objective** (this is M7's own registered negative) and
M8 shows **no excess** soft directions in that arm.

**(iv) Designed negatives.** **N-c1** coefficient-zero bit-identity. **N-c2 VACUUM CONTROL:** assert
the instrument reports `soft_direction = False` at a site with `λ_min ∈ [2α−1e-3, 2α+1e-3]` **and**
`depth < D_min`. **N-c3 CAPTURE CONTROL:** a site whose `λ_min` was softened at the cost of
`r_capture < σ_q` must **FAIL** — softness bought by giving up the basin is not a pass.

**(v) ⛔⛔ 2α — the hard ceiling, and it is the finding that decides (c)'s fate this wave.**
`λ_min ≥ 2α` is enforced by the confinement, so the longest-lived soft mode has
`τ_max = Γ/2α = Γ_c m/(2α)`. Measured (`s1_damping.json`, `s3_flat_floor.json`):

| γ | `Γ_c = −ln(1−γ)/dt` | `τ_max` (m=1) | shipped `N` | `N / τ_max` | steps to retain ½ of a `2α`-mode offset |
|---|---|---|---|---|---|
| 0.05 | 1.025866 | 10.2587 t.u. = **205.2 steps** | 400 | 1.95 | **151** |
| 0.02 | 0.404054 | 4.0405 t.u. = **80.8 steps** | 800 | 9.90 | — |
| **both** | — | — | **1200** | **11.85** | — |

Retention of a within-well offset under the **shipped** two-phase read:

| mode | `λ` | phase-1 regime | retention after 400+800 |
|---|---|---|---|
| written well (`A=1, s=0.32`) | 9.8656 | underdamped | **1.084e-8** ( = `e^{−18.34}`, reproduces T3's `C = 18.34`) |
| **`2α` floor mode** | 0.10 | **overdamped** | **3.489e-5** (direct simulation `−4.24e-5`; the non-normality prefactor is 1.22) |

> ⭐⭐ **Term (c) creates structure that the shipped read erases.** Even the softest direction the
> confinement *permits* survives at `3.5e-5`. ⇒ **(c) has no read-side consumer at the shipped
> budget.** Its only consumer this wave is **M8's end-of-training curvature spectrum** — which is
> exactly why Advisor ruling 4(iii) is right that M8 is mandatory: without it, (c) could be silently
> tuned to inertness. A *read-side* consumer needs `N ≲ 150–200` steps (or heavy mass — see §2).

**Staging: not live before wells exist** (at an undug site `H = 2αI + O(A)` with `A ~ 1e-4 = depth_init`).
**⛔ NOT PROVEN:** that a soft direction *survives superposition* on a learned multi-atom store. That
remains the Hub's Q10 = 0.25 and M8's job.

---

## (d) the set-level compositional read loss (`k` particles + DeepSets ψ)

**Expression.**
```
u_f = [ q_N^{(f)} ; V_θ(q_N^{(f)}) ; ‖∇V_θ(q_N^{(f)})‖ ; captured_f ; payload_readout_f ]
ψ(S) = ρ( Σ_f h(u_f) )            # DeepSets, PERMUTATION-INVARIANT POOLED SUM ONLY
L_read = E[ ℓ_task( ψ({u_f}) , y ) ]   , likelihood weighted by captured-vs-scattered counts
```
⛔ **No attention-ψ is specified** (quarantined for trajectory input, `AttentionPsiLeakError`).

**(ii) ⭐ Three channels, three magnitudes, all derived — this is the term's whole design content.**
| channel | law | magnitude |
|---|---|---|
| ψ / payloads | direct | O(1) |
| store `θ` | **implicit at settle**, `∂q*/∂θ = −(Hess V)⁻¹ ∂_θ∇V`, exact | O(1) after the write (`1.4664/1.1318/1.2450`); `7.9e-10/1.5e-9/7.3e-9` before |
| **`φ` / launch head** | **exactly 0** at the settled point (T3: `Fix(T_θ)` contains no `q₀`); `≍ K e^{−C}` at finite budget | **1.08e-8** at the shipped `C = 18.34` |

⇒ **(d) cannot train a launch head at the shipped budget.** With `φ` frozen (design-rule 1
SATISFIED by choice) this costs nothing; if C2W11 or C2W12 wants a *learned* launch head, (d) must be
evaluated at a truncated or trajectory read. **Say this before the head is built, not after.**

**(iii) Coefficient-zero.** `θ` after one step at `λ_read = 0` bit-identical to the term-deleted
build, 3 seeds, **including** the case where ψ is present but receives no gradient (ψ must stay
exactly at its init).

**(iv) Designed negatives** (these are K4/K7-CAP, restated as the term's own pytest set):
`N-d1` blank store **at full ψ capacity** ≤ chance+0.05 · `N-d2` query-only reader at full ψ capacity
≤ chance+0.05 · `N-d3` permuted payloads ≤ chance+0.05 · `N-d4` collapsed launch set (all `k`
particles on one channel) ≈ chance (M1's negative) · `N-d5` every reader has **< `N_a·m` params**
(SP-1: OLS on the true indicator scores **1.0000** exact-set with `‖v̂−v‖∞ = 4.25e-15` on a **blank**
store).

**(v) 2α / band.** `d/s ≥ 4.0` ⇒ the settled-point organization is exactly nearest-centroid VQ
(`D = 0.0000`) ⇒ `u_f` collapses to a well **index** and `L_read` is table-expressible (K5 trivial).
`d/s ≤ 2.01` ⇒ wells merge and the set is not identifiable. The `captured_f` descriptor requires
SC-6 `r_capture ≥ σ_q`, which §1(b) shows is the **first casualty** of term (b). `2α` enters the
residual descriptor: in undug regions `‖∇V‖ ≈ 2α‖q‖`, which is the vacuum reference (e) uses.

**Staging: not live before wells exist.**

---

## (e) the calibration term — feature dropout as pseudo-novelty

**Construction (and the choice of *what* is dropped is the whole argument).** Per **write-episode**,
draw a channel mask `n ~ Bernoulli(p_drop)^{N_a}` **independently of `x`**, and hold out from the
**write** every well `w_f` with `n_f = 1` (the well is *never written*). ⭐ **The query is unchanged
and still contains feature `f`.**
```
s_f = σ( g(u_f) )                      # u_f = the same per-particle diagnostics as (d)
L_cal  = (1/k) Σ_f BCE( s_f , n_f )    # log-loss: STRICTLY PROPER
L_cal2 = Brier / NLL of the set-answer confidence           # V2b's ECE
```
**Proper scoring rule: log-loss (or Brier) is the objective. ⛔ AUROC is the *reported* statistic and
is NOT a proper scoring rule — it is never trained against.** Say so in the arm's ledger.

**(ii) Channel.** Direct to `g`; **direct + implicit** to `θ` (depth and residual are explicit in `θ`
at `q*`, and `q*` depends on `θ` implicitly).

**(iii) Coefficient-zero.** The dropout mask uses a **dedicated RNG stream**; at `λ_cal = 0` it must
either not be drawn at all *and* the stream not advanced, or be drawn and discarded from a stream
nothing else reads. Assert bit-identity of `θ` **and** of the drawn write-plan sequence.

**(iv) ⭐⭐ Why it does not collapse to the base rate — and one leg of this is STRUCTURAL, not measured.**
- **Base-rate floor:** `p_drop` is uniform over channels and episodes, so the Bayes-optimal *constant*
  predictor is `p_drop` with log-loss `H(p_drop)`. Any improvement must come from store-conditional
  diagnostics. **`N-e2`:** on a **blanked** store the achieved log-loss must equal `H(p_drop)` within
  tolerance.
- ⭐ **`N-e3` (the structural one):** the mask is drawn **independently of `x`** and acts on the
  **WRITE**, never on the query ⇒ `n_f ⟂ query` **by construction** ⇒ **a query-only novelty head is
  provably at the base rate, and any AUROC > 0.5 is store information.** This is a *structural* kill
  of the leak, of the same kind as K8, and it is strictly stronger than a measured guard.
- **`N-e1`** permuted payloads ⇒ AUROC ≈ 0.5 (V2's registered negative).
- **`N-e4`** channel balance: `I(n_f ; f) = 0` by construction (uniform `p_drop` over channels), else
  the head reads channel *identity* instead of novelty.

**⚠ Statistical caveat, registered here so it is not discovered at review:** channel-level dropout
makes the effective sample size `#write-episodes × N_a`, **not** `#items`. Resample the held-out
subset **per episode** and report the episode count beside every AUROC. V2a's `> 0.60` floor is a
5-seed statement; at few episodes its SE is set by episodes, not by queries.

**(v) ⛔ 2α — and this is a CROSS-TERM CONFLICT I believe is unflagged anywhere in the program.**
The vacuum reports `λ_min ≈ 2α` and depth ≈ 0. **If term (c) succeeds** in pulling written sites'
`λ_min` toward `2α`, it makes written and unwritten sites **spectrally indistinguishable in
`λ_min`** — destroying `λ_min` as a novelty feature. ⇒ **the novelty head must key on DEPTH and
`λ_2nd` (or the participation ratio), never on `λ_min`.** This is M8's banked trap seen from the
other side, and it means **(c) and (e) must be trained with an explicit feature-set contract**.

**Staging: not live before wells exist** (the written/unwritten *contrast* is the signal).

---

## (f) kinetics — trajectory / finite-budget reads ONLY, friction before mass

**Expression.** Per-particle friction multiplier `g_f` (and the spatial field `γ_φ(q)`), per-address /
per-direction mass `M_f = L_f L_fᵀ` (SPD by construction).
```
L_kin = Σ_{n=1..N} w_n · ℓ( q_n^{(f)}, … )     # trajectory loss, w_n non-degenerate over the first ~1/Γ_c steps
      | or L_read evaluated at a TRUNCATED budget N ≪ shipped
```
**(ii) Channel: trajectory / finite-budget only.** `∂z*/∂{γ, M} ≡ 0` **exactly** at the settled point
(T3; measured **bitwise 0.0, 3/3 seeds**, in the gym *and* in-system). At finite budget
`‖∂z_N/∂ζ‖ ≍ K_ζ e^{−C}` with `K_ζ = O(N)` for `{M, γ}`.

**(iii) Coefficient-zero.** `g_f = 1` and `M_f = I` must reproduce the shipped scalar-γ / identity-mass
read **bit-identically** — the shipped code already has this property for `gamma_field = None` via a
Python-level branch; keep the same discipline for per-particle multipliers.

**(iv) Designed negatives.** `N-f1` **settled-point zero**: `grad(L, {g_f, log M_f})` must be
**exactly 0.0 bitwise** under a settled-point read — a nonzero value means the read is not settled or
a leak path exists. `N-f2` **budget precondition**: `C = Σ_p N_p ln(1/ρ_p) ≤ C_max`. `N-f3`
**mass-blindness control**: on a strictly convex single-minimum basin at **full** settle, the endpoint
must be `M`-invariant to 1e-12 (Prop F1; verified — see §2).

**(v) 2α / band — derived, and it *is* the ordering rule.** See Deliverable 2. Headline:
`det A = (1−γ)` exactly and, in the **underdamped** regime, `|ρ| = √(1−γ)` **exactly, independent of
`λ, m, dt`** ⇒ **mass has no first-order effect on the contraction modulus when `λ > λ_crit`.**

**⛔ Built this wave: NO** (PREREG-C2W11 §9 NOT-RUN 3). This is a specification for C2W12, as the task
requires. **DB4: the kinetics head needs `C ≤ ~7`, i.e. `N ≤ 273` at γ=0.05 or `N ≤ 693` at γ=0.02;
the shipped read is `C = 18.34`.**

---

# DELIVERABLE 2 — the §A34.5 kinetics scoping note

⛔ **Everything in this section is a 1-D / 2-D linear-or-two-well designed toy.** `bprime-theory` §9.2
bracket applies.

## 2.1 The drag force, and the continuum reduction of the SHIPPED damping
The dissipation is ratified as a **drag force `−γ·p`** (`v̇ = −(1/m)∇V − γv`; no normal-force/mass
coupling). ⚠ **But the shipped code does not apply `−γp·dt` — it applies a per-step multiplier
`p ← (1−γ)p`** (`integrators.py:77`). The equivalent continuous drag rate is
```
Γ_c = −ln(1−γ)/dt        ⇒   Γ_c(0.05) = 1.025866 ,   Γ_c(0.02) = 0.404054     (dt = 0.05)
```
**Every `γ` in a physics formula below is `Γ_c`, not the config's `γ`. They differ by 20.5× at
γ = 0.05.** Reporting a "damping ratio at γ = 0.05" without this conversion is off by 20×.

## 2.2 ⭐ The damping ratio `ζ`, derived in the program's own constants
`m q̈ = −λq − Γ_c m q̇` ⇒ `ω₀ = √(λ/m)` and
> ### `ζ = Γ_c √(m/λ) / 2`  ⇒  **`ζ ∝ √m`** at fixed `γ, λ` ✓ (measured log-log slope **+0.5000000000000004**)

| mode | `λ` | `ζ` at γ=0.05 | `ζ` at γ=0.02 |
|---|---|---|---|
| vacuum / `2α` floor mode | 0.10 | **1.6220** (over) | **0.6389** (under) |
| written well `A=1, s=0.32` | 9.8656 | **0.1633** (under) | **0.0643** (under) |

**The under/over-damped boundary, two ways, agreeing:**
```
continuum:  λ_crit = Γ_c² m / 4                       ⇒ 0.263100 m (γ=.05) , 0.0408149 m (γ=.02)
discrete :  κ_crit = 2[1 − 2√(1−γ)/(2−γ)] , κ ≡ dt²λ/m ⇒ 0.263028 m        , 0.0408132 m
```
Agreement **2.7e-4 / 4.3e-5 relative**; the closed form matches a bisection on the exact map to
**2.6e-13 / 1.3e-13**. ⇒ ⭐ **R4: light = underdamped / heavy = overdamped is right, but the
boundary is a property of the MODE.** At m=1, γ=0.05 the well modes are underdamped and the floor
mode is overdamped; at γ=0.02 both are underdamped. Critical masses: `m_crit = 4λ/Γ_c²` =
**0.380** (floor) / **37.50** (well) at γ=0.05; **2.450** / **241.7** at γ=0.02.

## 2.3 ⭐⭐ Two exact identities of the shipped map — and they *derive* "friction before mass"
Linearising the shipped map about a site (`V = ½λq²`) gives the exact 2×2 propagator
`A = [[1−κ/2, dt/m], [−(1−γ)dtλ(1−κ/4), (1−γ)(1−κ/2)]]`, `κ = dt²λ/m`. Then, verified over **400
random cells** spanning `λ ∈ [1e-3, 10^1.5]`, `m ∈ [0.1, 10^1.5]`, `γ ∈ [0.005, 0.5]`, `dt ∈ [0.01, 0.2]`:
> **(I1)** `det A = (1−γ)` **exactly** — max residual **2.22e-16** — independent of `λ, m, dt`.
> **(I2)** In the **underdamped** regime `|ρ| = √(1−γ)` **exactly** — max residual **2.22e-16** over
> the 86 underdamped cells — independent of `λ, m, dt`.

**Consequence (this is the derivation of §A34.5's measured ordering).** In the underdamped regime
mass changes **only the rotation angle** `arg ρ`, never the contraction modulus:
`∂|ρ|/∂m` at γ=0.05 measured **1.11e-10** for the underdamped well mode (that is the finite-difference
floor) vs **6.16e-3** for the overdamped floor mode. Friction, by contrast, sets `|ρ|` for **every**
mode: `d ln|ρ|/dγ = −1/(2(1−γ)) = −0.526`.
> ⇒ **Friction is the strictly stronger channel because it acts on the contraction modulus of all
> modes, while mass acts on the modulus only where `λ < λ_crit` and otherwise only on phase.**
> The banked ratio (`2.4e-2/4.1e-2/3.3e-2` friction vs `1.74e-3/1.17e-2/1.88e-3` mass ⇒ "~14×", and
> traj/point ratios `2.6–4.9e5` vs `1.7–2.9e5`) is the *measured* size; the *sign and mechanism* are
> now derived. ⭐ **I did NOT derive the 14× prefactor — that stays a measurement.**

## 2.4 Mass-blindness: exactly which reads escape Prop F1
**Prop F1 unchanged:** the fully-settled endpoint in a strictly convex basin is `M`-independent,
because `Fix(T_θ) = {(q,0) : ∇V_θ(q) = 0}` contains no `M`. Verified: 9/9 barrier-trapped cells in
`s2_spectral_mass.json` land at `q* = −0.42470921223772…` for `M ∈ {I, diag(1,4), diag(4,1),
0.25I}` (agreement to **~1e-14**).
**Reads that escape it — the complete list:**
1. **Finite-budget reads** (`N dt ≲ τ`), where the landed point is a direction-wise contracted launch
   offset and the contraction depends on `M` (§2.5).
2. **Trajectory reads** (any functional of `{q_n}`, not of `q_N` alone) — `M` sets the time
   parametrisation, so it is visible even at full settle.
3. **Non-convex basins / multi-minimum basins**, where `M` can change *which* critical point is
   reached (§2.5's separation condition).
4. **Flat-floor reads** (`λ = 0` exactly): the endpoint is `q₀ + p₀/(mΓ_c)` — `M`-dependent even at
   `t → ∞` (§2.6).
⇒ Every mass statement must name which of the four it is scoped to.

## 2.5 ⭐⭐ Spectral / per-direction mass: the separation condition, two-sided

**(A) When two launches PROVABLY DO NOT separate.**
1. **Scalar mass is an exact gauge of the path image.** `q̃(τ) = q_c(√c τ)` solves the `M`-system, so
   `(M, p₀, t, Γ_c) → (cM, √c p₀, √c t, Γ_c/√c)` leaves the **path image invariant**. Verified in
   continuous time (RK4, `s2b_fixes.json`): Hausdorff distance between the two path images is
   **0.0 exactly** at `Γ_c ∈ {0, 0.404, 1.026}`. ⚠ At `Γ_c = 0` the friction leg is vacuous, so
   **at γ = 0 scalar mass alone is a pure time reparametrisation — no separation, provably.** At
   `γ > 0` the naive change (mass rescaled, friction not) moves the path by **0.082 / 0.101**, i.e.
   **11–15 % of the path extent** — because scalar mass is *equivalent to a friction change by
   `1/√c`*, nothing more.
2. **Gradient aligned with an eigenvector of `M`.** The initial roll direction is `−M⁻¹∇V(q₀)`. If
   `∇V(q₀)` is an eigenvector of `M` then `M⁻¹g ∥ g` **for every mass ratio** (verified: `cos = 1.0`
   at ratios 1.5, 2, 4, 16, 100). ⇒ **a launch on a symmetry axis of the mass tensor provably does
   not tilt.** *(My first pass tested exactly this degenerate configuration and had to be redone —
   `s2b_fixes.py` FIX 2. An engineer building the spectral-mass probe must avoid the same trap.)*
3. **Energy barrier.** With `E₀ = ½p₀ᵀM⁻¹p₀ + V(q₀)`, if `E₀ < V(saddle)` for **both** masses and the
   basin has a unique critical point, both endpoints coincide (dissipation only removes energy).
   Verified: **9/9 cells with `T₀ < barrier` landed in the same well**; the only cell that crossed had
   `T₀ = 0.72 > barrier = 0.2527`. ⭐ Note `T₀` itself depends on `M`: at `p₀ = 0.6`, `M = I` gives
   `T₀ = 0.36 >` barrier, while `M = diag(1,4)` gives `T₀ = 0.225 <` barrier ⇒ **a heavier mass tensor
   provably switches OFF barrier crossing**, directionally.

**(B) When they PROVABLY DO separate.**
1. **Direction tilt (necessary condition).** For diagonal `M₁, M₂`, `M₁⁻¹g ∦ M₂⁻¹g` iff `g` has
   non-zero components in **at least two directions with different mass ratios**. For `diag(1, r)` the
   maximal tilt over gradient directions is
   > `θ_max = arctan√r − arctan(1/√r)`, attained at gradient angle `arctan√r`:
   > **11.54° (r=1.5) · 19.47° (r=2) · 36.87° (r=4) · 61.93° (r=16) · 78.58° (r=100)**.
   Measured off-axis launches reproduce this (e.g. launch `[0.30,0.10]`: 11.52° / 19.15° / 33.16° /
   45.27°).
2. **Finite-budget separation (the usable one, and it is exactly solvable).** In an anisotropic
   quadratic the landed point is the launch offset contracted **per direction**:
   - **overdamped** (`λ_i < Γ_c²m_i/4`): `q_i(t) ≈ q_i(0)·exp(−λ_i t/(Γ_c m_i))` — verified to
     **1.2 % / 0.43 % / 0.12 %** at m = 1 / 4 / 16;
   - **underdamped**: envelope `e^{−Γ_c t/2}`, **mass-free**; only the phase moves.

   ⇒ **the mass tensor is a learnable per-direction low-pass filter on the launch offset, and it is
   monotone only in the overdamped regime.** Measured at a **truncated** read (`N = 200, γ = 0.05`):

   | `m` | retention, soft mode `λ=0.10` | retention, stiff mode `λ=9.87` |
   |---|---|---|
   | 0.25 | −0.0094 | 0.0059 |
   | 1 | **0.3812** | 0.0051 |
   | 4 | **0.7989** | −0.0023 |
   | 16 | **0.9462** | 0.0039 |

   **The soft column spans 100× monotonically in `m`; the stiff column is flat and sign-oscillating
   at the mass-free envelope `0.95^{100} = 0.0059`.** ⭐ **That is the mechanism of spectral mass,
   stated as a design rule: mass selects along soft directions, is blind along stiff ones, and only
   at a truncated read.**
   At the **shipped** 1200-step budget the same column reads `1.8e-8 / −4.2e-5 / 0.0355 / 0.4798` —
   ⭐ **so `m ≳ 4` on the soft direction keeps floor information alive even under the shipped read**
   (`τ_max = Γ_c m/2α` = 821 steps at m = 4). **This is the cheapest available fix if a read-budget
   change is unaffordable.**

> ### The condition, stated as the task asks
> **Two launches from the same `q₀` with mass tensors `M₁ ≠ M₂` provably SEPARATE iff (i) `∇V(q₀)` is
> not a common eigenvector, AND (ii) the read is finite-budget or trajectory, AND (iii) at least one
> direction with a differing mass ratio is OVERDAMPED (`λ_i < Γ_c² m_i/4`).**
> **They provably DO NOT separate if any of:** the read is a full settle in a strictly convex basin
> (Prop F1) · `M₂ = cM₁` **and** friction is co-rescaled by `1/√c` (exact gauge) · `∇V(q₀)` is a
> common eigenvector · `½p₀ᵀM_i⁻¹p₀ + V(q₀) < V(saddle)` for both (barrier-trapped).

## 2.6 ⭐ The flat-floor stopping criterion
On an **exactly flat** direction (`λ = 0`), the momentum recursion is autonomous: `p_{n+1} = (1−γ)p_n`
and `q_{n+1} = q_n + dt·p_n/m`, so
> ### `Δ = dt · p₀ / (m γ)` **exactly** (discrete) → `p₀/(m Γ_c)` (continuum)
> With a mass **tensor**: `Δ_i = dt · p_{0,i} / (m_i γ)`.
Verified to **1e-15** in 9/9 `(γ, m)` cells (`s3_flat_floor.json`); the continuum form is a 1.0 % /
2.6 % / 5.4 % approximation at γ = 0.02 / 0.05 / 0.10, the discrete form is exact.

**When does a finite-budget read stop, and what does the stopping point encode?**
- **It stops on its own**, at `Δ`, after `≈ 3/γ` steps (60 at γ = 0.05) — no budget is needed to
  *stop*; the budget is needed only to *converge in the stiff directions*.
- **The stopping point encodes the entry momentum**, losslessly and invertibly:
  `p₀ = Δ · m γ / dt`. Verified: recovered `p₀` = 0.2 / 0.5 / 1.0 / 2.0 from the landed offsets
  (0.5 % bias from the discrete-vs-continuum conversion, exact under the discrete form).
  ⇒ ⭐ **A flat floor is the one place in the shipped physics where the settle is INFORMATION
  PRESERVING about the launch.** Everything else is a contraction of modulus `√(1−γ)` per step.
- **With the shipped confinement there is no exactly flat direction** — the floor is `λ = 2α` and the
  particle *creeps* rather than stopping, with `τ_max = Γ_c m/2α`. Measured creep at γ = 0.05, m = 1:

  | steps | 25 | 50 | 100 | **200** | 400 | 800 | 1200 |
  |---|---|---|---|---|---|---|---|
  | retained offset | 0.947 | 0.850 | 0.656 | **0.381** | 0.128 | 0.014 | 0.0016 |

  Budget for 50 % / 10 % retention on the `2α` mode at γ = 0.05: **151 / 446** steps (m=1),
  **576 / 1864** (m=4), **2282 / 7532** (m=16).
- **Sub-wells (discrete) · flat floors (continuous) · trajectory reads (temporal) are three
  implementations of the same hierarchical-settle semantics, and they are not exclusive** — they are
  distinguished only by *where the read stops being a contraction*: at a separatrix (discrete), at a
  ballistic range (continuous), or nowhere (temporal, `ρ = 1` on a γ = 0 leg).

## 2.7 The ordering, with its measured basis and its derived reason
1. **Friction first** — the ~**14×** stronger channel (traj/point ratios **2.6–4.9e5** friction vs
   **1.7–2.9e5** mass; per-query gradients `2.4e-2/4.1e-2/3.3e-2` vs `1.74e-3/1.17e-2/1.88e-3`).
   **Derived reason: §2.3's identity (I2).** Friction sets `|ρ|` for every mode; mass does not, in the
   underdamped regime.
   ⚠ **The band is bounded from below by monitor #1** (`γ ≤ 0.03` trips it on the S0 store), and
   monitor #1's own collapse mode is **overdamping → the last observation** (`corr(q*, q_last) → 0.97`)
   — i.e. the read degenerates to "assign the launch point", which is Prop O4's `γ ≥ 0.2` VQ-collapse
   seen from the monitor side. ⇒ **the usable friction band is narrow and two-sidedly monitored**, and
   the shipped `γ_read = 0.02` is *below* monitor #1's S0 trip edge — a scoping tension the Hub should
   note (γ-band statements are harness- **and** read-budget-scoped, C2W5 rider).
2. **Mass second** — a selector only where `λ < λ_crit`, and only at finite budget.
3. **Spectral / per-direction mass third and richest** — already shipped as per-address mass (C2W7);
   §2.5 gives its two-sided condition. **Its natural home is the soft/floor directions**, which is
   precisely the structure term (c) defends. ⇒ **(c) and (f) are one design, not two.**

---

# DELIVERABLE 3 — pricing the open §A31.2 mechanism question

> ⛔ **I was asked to PRICE, not to solve.** Nothing below is asserted as the answer.

## 3.0 ⭐⭐ First, a reframing that costs nothing and changes what is open
§A31.2 says the settle "extracts **less** from the cue than 1-NN over the same keys" and calls that
unexplained. **Two different things are bundled there:**
- **(i) That there is a gap at all is a THEOREM, not a mystery.** Theorem O1: the image of `x ↦ q*`
  is exactly the set of minima of `V_θ`, so `I(q*; cue) ≤ log N_min`, while 1-NN uses the full
  metric. On a **metric-native** cue protocol, 1-NN over the stored keys **is the Bayes rule** — the
  charter's own §A32.1 ruling ("no pass 4 chasing daylight on a metric-native cue protocol"). This is
  the metric-native-ceiling theorem and it is a known confirmation, not news.
- **(ii) The SIZE of the gap — 0.50 vs 0.79–0.90 — is what is open.** Everything below prices (ii).

## 3.1 ⭐⭐ The zero-compute discriminator, run: the premise is not established
`s5_a31_geometry.json` + `s5b_anisotropy.json`. Pure geometry, no store, no physics, ~40 s.

**"σ_q/spacing = 0.32" is undefined between two readings, and they differ by the entire effect size:**

| reading | what it means | 1-NN accuracy over the same keys |
|---|---|---|
| **NORM** `E‖q − own key‖ = 0.32 × spacing` | "queries are ~3× closer to their own key than to a neighbour" | **1.0000 (sd 0.0000) in 27 / 27 cells** (`d ∈ {12,32,64} × K ∈ {16,32,128} × ratio ∈ {0.19,0.32,0.37}`) |
| **PER-COORDINATE** `σ = 0.32 × spacing` per axis ⇒ `E‖δ‖ ≈ 1.11 × spacing` | queries are *farther* from their own key than the NN spacing | **0.839 / 0.800 / 0.743** at `d = 12, K = 16/32/128`; 0.71–0.78 at `d = 32`; 0.51–0.74 at `d = 64` |

**Robustness of the NORM row** (`s5b`): at fixed `‖δ‖ = 0.32 × spacing`, 1-NN stays **1.0000** for
noise **rank 1, 2, 3, 6, 12** and for `d ∈ {12,32}`, `K ∈ {16,32}` — anisotropy does not break it.
Only degenerate key geometry dents it (`key_rank = 2` in `d = 32` ⇒ 0.93–0.98). **Inverse:** to reach
1-NN = 0.79 under the NORM reading you need `‖δ‖/spacing ≈ 1.04–2.42` depending on `(d, K, rank)` —
i.e. **3–7× the quoted 0.32.**

> ⭐⭐ **The finding, and it is the highest-value zero-compute result available to me.**
> **Under the NORM reading the premise "comfortable geometry" is TRUE and then a banked 1-NN launder
> of 0.79–0.90 on the same population is IMPOSSIBLE (it should be 1.000).** **Under the
> PER-COORDINATE reading the launder's 0.79–0.90 is reproduced almost exactly at `d = 12` and the
> premise "queries are ~3× closer to their own key" is FALSE.** ⇒ **One of the two banked numbers is
> measured on a different object, and the whole A31.2 puzzle may be an artifact of that.**
> ⛔ **I am NOT re-instating the retracted §A29.5.** That retraction was about the *population* the
> spacing was computed on (a ~200-key sizing set vs a 16-item store) and it stands. What I am saying
> is narrower and testable: **the replacement number 0.32 is only "comfortable" under one of two
> definitions, and the banked launder accuracy discriminates them.**
> **Cost to close: ≈ 1 engineer-hour, ZERO compute** — recompute `E‖q − own key‖` and `E(NN spacing)`
> **in the integrator's own coordinates, on the banked arrays**, and report the *distribution* of
> `‖q − own‖ / ‖q − nearest other‖` (its fraction > 1 **is** the 1-NN error, no simulation needed).
> ⛔ **No A31.2 mechanism should be funded before this runs.**

## 3.2 The candidate mechanism list, priced

| # | candidate | what it predicts that the others do not | cheapest discriminating measurement | cost (eng-h / compute) | banked artifact status |
|---|---|---|---|---|---|
| **C1** | **premise/definition mismatch** (§3.1) | the recomputed ratio in integrator coordinates is ≈1.0, not 0.32; the `‖q−own‖/‖q−other‖` distribution has ~10–20 % mass above 1 | recompute both quantities on the **banked** query/key arrays; report the ratio *distribution* | **1 h / 0** | ⭐ **partially DISCRIMINATED already**: the two readings cannot both be true beside a 0.79–0.90 launder |
| **C2** | **attractors are not at the keys** (minima ≠ centres) | settle assignment scored against the **measured minima** is high while against the designed key it is 0.50; the deficit tracks drift | relabel banked settle endpoints by nearest **measured** minimum | **2 h / 0** (if endpoints banked) | ⭐ **strong banked support**: `ρ(A1, G-DRIFT) = −0.967` over 9 cells; C2W5 measured attractors ~1.4 `sep` from the designed anchors |
| **C3** | **basin ≠ Voronoi: depth-weighted power diagram** (term (b)'s side-effect ex post) | misassignments are **directed shallow→deep**; error rate correlates with local depth contrast; some wells have **no basin at all** at heterogeneity ≥ 2.2 (d/s=2.7) | confusion-matrix **asymmetry** vs per-well depth; count distinct settled points | **2–3 h / 0** (if per-well depth + confusion banked) | quantified by **this task** (`s4`): the annihilation table; O2 supplies the offset law |
| **C4** | **inertial overshoot** (Prop O4) | correct-basin rises **monotonically with γ** and → 1-NN at γ ≥ 0.2 | a **read-time γ sweep** {0.02, 0.05, 0.1, 0.2, 0.5}, **no retraining** | **2 h / minutes** | bounded in advance: O4's non-static mass is 0.197 (γ=.05) → 0.0003 (γ=.2), so C4 can explain **at most ~20 points**, not 50 |
| **C5** | **confinement radial bias** (`α‖q‖²` tilts every boundary toward the origin ⇒ inner wells over-capture) | misassignment direction correlates with `Δ‖c‖` | regress misassignment on `Δ‖c‖` | 1 h / 0 | ⛔ **REFUTED by this task, zero compute** — see §3.3 |
| **C6** | **non-capturing wells** (`λ_min>0`, capture 0.000) | correct-basin ceiling = fraction of capturing wells | — | — | ⛔ **REFUTED from banked artifacts**: C2W8p3 measured **46/48 basins** post-repair ⇒ ceiling ≈ 0.96, cannot produce 0.50 |
| **C7** | **the read is not settled** | partial settle leaves `q_N` near `q₀` | — | — | ⛔ **REFUTED from banked artifacts + this task**: `C = 18.34` ⇒ retention **1.08e-8** (stiff) and **3.49e-5** (softest mode the confinement permits). The shipped read is settled on **every** direction |
| **C8** | **`m = 1` payload unsatisfiability / rule-4** | the deficit is in the payload half, not the basin half | K2's payload half on the same cells | 1 h / 0 | banked: the payload half is unsatisfiable at `m = 1` (0.5 %); `m ≥ 8` required (C2W5 D1). **Scope check only** |

**⭐ Recommended order (and it is deliberately front-loaded with the free ones): C1 → C7/C6/C5
(already refuted, cite them) → C2 → C3 → C4.** C1–C3 are all **zero new compute**; only C4 spends
any, and it spends minutes.

## 3.3 ⛔ One candidate refuted outright, from a toy, at zero compute — C5
`s7_confinement_bias.json` (1-D, two **equal-depth** wells at `[a, a+d]`, α = 0.05 vs α = 0 control):
the confinement moves the separatrix toward the origin by **offset/`d` = 0.68 %–6.3 %**, giving an
inner/outer basin ratio of **1.05–1.68** and an **equivalent depth ratio of only 1.02–1.17**. At
unit-ball radii (`a ≤ 1`, which is where the shipped address-space normalization puts everything) the
offset is **≤ 2.6 % of the spacing** and the equivalent depth ratio **≤ 1.10**.
⇒ **The confinement's radial bias cannot produce a 0.50 correct-basin. C5 is refuted as a primary
mechanism.** *(Instrument note: 3 of the 15 `α = 0` control cells at `d/s = 2.9` failed to resolve the
separatrix on the sign-change grid — a root-finder artifact on a very flat barrier, not physics, and
the α = 0.05 cells all resolved.)*

## 3.4 What I did NOT do
I did **not** rank the candidates by posterior probability, and I did not run any measurement on a
learned store. §A31.2 says the question is open and theorist-**priced**; assigning a winner from a
1-D toy would be exactly the over-reach the task forbids.

---

# 5. PREREG SCORECARD (`.claude/outputs/c2w11-loss-package/PREREG.md`, filed before any script existed)

| # | prediction | measured | verdict |
|---|---|---|---|
| P1 | `Γ_c = 1.025866 / 0.404054` | 1.0258658877510107 / 0.40405414635038894 | ✅ |
| P2 | `ζ` = 1.6222 / 0.63886 (floor); 0.16330 / 0.06432 (well); `ζ ∝ √m` | 1.62204 / 0.63887; 0.163305 / 0.064320; slope **+0.5000000000000004** | ✅ |
| P3 | `λ_crit` discrete 0.262976 / 0.0408163; continuum agreement < 0.05 % | **0.263028 / 0.0408132**; agreement **0.027 % / 0.0043 %** | ✅ (my hand `κ_crit` was 0.02 % low; the *prediction band* held) |
| P4 | `det A = (1−γ)` and `|ρ| = √(1−γ)` exact, residual ≤ 1e-14 | **2.22e-16** both, over 400 random cells | ✅ |
| P5 | retention 1.0854e-8 (stiff) / 3.605e-5 (floor); ratio ~3.3e3 | **1.0843e-8 / 3.489e-5**, ratio **3218**; `C = 18.3397` (T3: 18.34) | ✅ |
| P6 | `τ_max` = 205.2 / 80.8 steps; budget 11.85 τ_max | 205.17 / 80.81; **11.849** | ✅ |
| P7 | flat-floor range `= p₀/(mΓ_c)` to < 1 % | continuum form off by **1.0 / 2.6 / 5.4 %** at γ = .02/.05/.10; **the exact discrete form `dt·p₀/(mγ)` matches to 1e-15** | ◐ **prediction refined, not met as written** — I predicted the continuum form; the exact law is the discrete one |
| P8 | O2 offset within ±15 % at ratio 3; **annihilation ratio at d/s = 2.7 in [4.0, 7.0]** | offset ratio **1.167** (16.7 %, just outside); **annihilation at 2.22** | ⛔ **MISSED, by 1.8×, in the dangerous direction.** Reported as **finding R1**, not corrected away |
| P9 | NORM ⇒ 1-NN ≥ 0.9999; COORD ⇒ < 0.30; the two readings straddle the banked launder | NORM **1.0000 in 27/27**; COORD **0.74–0.84** at d=12 (not < 0.30 — my `< 0.30` was for `E‖δ‖ = 1.11 spacing` in high `d`, and at `d = 12, K = 16` it is 0.84) | ◐ **direction confirmed, magnitude of the COORD leg over-predicted.** The *decision-relevant* half (NORM ⇒ 1.000, incompatible with a 0.79–0.90 launder) held exactly |

**7 ✅ · 2 ◐ · 1 ⛔.** The ⛔ (P8) is the most useful line in the report: it is R1.

---

# 6. Declared NOT-RUNs (never to be reported as nulls)
1. **No measurement on a learned multi-atom store.** Every number is a designed-well toy.
   `bprime-theory` §9.2's "naming `s` for a learned multi-atom well is an unsolved modelling question"
   is **unchanged** by this task, and it gates the transfer of §2.2's `ζ` table, §1(b)'s annihilation
   table and §2.5's retention tables.
2. **No sympy.** Not installed in the main venv; all algebra is by hand + numerical verification. No
   computer-algebra proof certificate exists for (I1)/(I2) — they are hand-derived and verified to
   2.2e-16 over 400 cells, which is **strong evidence, not a machine-checked proof**.
3. **The kinetics HEAD is not built** (PREREG-C2W11 §9 NOT-RUN 3). Term (f) is a specification.
4. **No attention-ψ anywhere.**
5. **The 14× friction/mass prefactor is quoted, not derived.** I derived the ordering and its
   mechanism; the magnitude stays a measurement.
6. **A31.2 is priced, not solved.** No candidate is endorsed.

# 7. Verdict ladder (proven ▸ evidenced ▸ conjectured)
| statement | status |
|---|---|
| `det A = (1−γ)` exactly; `|ρ| = √(1−γ)` exactly when underdamped | **PROVEN** (hand algebra) **+ verified** 2.2e-16 / 400 cells |
| `ζ = Γ_c√(m/λ)/2`; `λ_crit = Γ_c²m/4`; `Δ_flat = dt·p₀/(mγ)`; `τ_max = Γ_c m/2α` | **PROVEN + verified** to 1e-13…1e-15 |
| Prop F1 (settled endpoint is `M`-blind in a strictly convex basin) | **PROVEN** (T3, cited) **+ reconfirmed** 9/9 cells |
| scalar mass is an exact path-image gauge iff friction is co-rescaled by `1/√c` | **PROVEN + verified** (Hausdorff **0.0**) |
| barrier-trapping ⇒ no separation | **PROVEN** (energy monotonicity) **+ verified** 9/9 |
| mass separates only in the overdamped regime, at finite budget | **PROVEN** for the linear mode **+ verified**; **EVIDENCED** for a real landscape |
| the shipped read erases within-well launch information on every direction | **EVIDENCED** (toy, 2 modes, exact linear algebra) — the *mechanism* is proven, the *landscape generality* is not |
| term (b)'s heterogeneity annihilates shallow neighbours in-band at ratio 1.6–3.3 | **EVIDENCED** — 1-D single-atom toy; ⛔ **must be re-measured on the learned store** (R1) |
| §A31.2's premise is undefined between two readings | **EVIDENCED**, 27/27 + 20/20 cells; the *resolution* is **OPEN** |
| a within-well soft direction survives superposition on a learned store | ⛔ **CONJECTURED** — not addressed here; M8's job; Hub's Q10 = 0.25 |
| which A31.2 candidate is correct | ⛔ **OPEN by design** |

# 8. Open questions / follow-ups / risks
- **OQ-A (the one I most want run, 1 h, zero compute):** §3.1's ratio-definition check. It may
  dissolve A31.2 entirely, or it may sharpen it into a real mechanism question. Either is worth more
  than any measurement funded before it.
- **OQ-B:** re-measure §1(b)'s annihilation ratio on the **learned multi-atom** store (spoke A's
  effective-`s` estimator + SC-6 capture instrument already have everything needed). Until then R1 is
  a toy-level inconsistency, not a measured one.
- **OQ-C:** the `γ_read = 0.02` vs monitor #1's `γ ≤ 0.03` trip edge tension (§2.7). Is the shipped
  read below the monitor's own band, and is that band S0-specific?
- **Risk:** everything here is Newtonian, `M ≻ 0`, `p₀` small, no Langevin, no training, single width
  `s`, and **no learned store anywhere**. The structural results (§2.3's identities, §2.5's
  conditions, §2.6's stopping law) are landscape-independent; every *number* is not.

## Git footprint
**None.** No tracked code touched; repo read-only at `main @ 7fcef50`, clean tree, no branch, no
worktree. All artifacts under `.claude/scratch/c2w11-loss-package/` and
`.claude/outputs/c2w11-loss-package/`.

---

## Proposed handover updates (for the Hub)

**§1 (the physics) — five additions.**
1. ⭐⭐ **The shipped damping is a per-step multiplier, and its continuum rate is `Γ_c = −ln(1−γ)/dt`
   — 20.5× the config `γ` at γ = 0.05, dt = 0.05.** Every damping-ratio / lifetime statement must use
   `Γ_c`. From it: **`ζ = Γ_c√(m/λ)/2`** (`ζ ∝ √m`, slope +0.5000000000000004) and
   **`λ_crit = Γ_c²m/4` = 0.2630 m (γ=.05) / 0.04082 m (γ=.02)**. At m = 1 a written well mode
   (λ≈9.87, ζ = 0.163) is underdamped and the `2α` floor mode (λ = 0.10) is overdamped at γ=0.05
   (ζ = 1.622) and underdamped at γ=0.02 (ζ = 0.639).
2. ⭐⭐ **Two exact identities of the shipped map (residual 2.2e-16, 400 random cells):**
   `det A = (1−γ)` and, underdamped, `|ρ| = √(1−γ)` — **both independent of `λ, m, dt`**. ⇒ **mass has
   NO first-order effect on the contraction modulus above `λ_crit`; it moves only the phase.** *This
   derives §A34.5's "friction before mass" ordering; the 14× magnitude remains a measurement.*
   It also **scopes `bprime-theory` T3's `C = Σ N_p ln(1/ρ_p)` to the underdamped regime** (R6).
3. ⭐⭐ **`τ_max = Γ_c m/2α` = 205.2 steps at m=1, γ=0.05 (80.8 at γ=0.02); the shipped 400+800 read is
   11.85 τ_max and retains 3.49e-5 of a `2α`-floor offset (vs 1.08e-8 for a stiff mode, a 3218× gap).
   ⇒ the shipped read erases within-well launch information on EVERY direction the confinement
   permits.** Term (c) therefore has **no read-side consumer at the shipped budget** (M8 is the only
   one), and the kinetics head needs `C ≤ ~7` ⇒ `N ≤ 273` at γ=0.05.
4. ⭐ **Flat-floor stopping law: `Δ = dt·p₀/(mγ)` exactly (→ `p₀/(mΓ_c)`), per direction with a mass
   tensor; the stopping point is an invertible record of the entry momentum.** A flat floor is the one
   place in the shipped physics where the settle preserves launch information.
5. ⭐ **Spectral-mass separation, two-sided.** Separates iff `∇V(q₀)` is not a common eigenvector of
   the two mass tensors **and** the read is finite-budget/trajectory **and** a differing-ratio
   direction is overdamped. Provably does **not** separate on: full settle in a strictly convex basin
   (F1) · scalar mass with friction co-rescaled by `1/√c` (exact path-image gauge, Hausdorff 0.0) ·
   gradient aligned with a mass eigenvector · barrier trapping `½p₀ᵀM⁻¹p₀ + V(q₀) < V(saddle)`.
   Max tilt for `diag(1,r)`: `arctan√r − arctan(1/√r)` = 19.5° at r=2, 36.9° at r=4.

**§7 (known issues / live) — three new entries.**
- ⛔⛔ **R1: `PREREG-TierII` §7's operating point may be internally inconsistent.** `d/s ∈ [2.5,2.9]`
  + "depth heterogeneity ≥ 3×" ⇒ the shallow neighbour's minimum is annihilated at measured ratio
  **1.60 / 2.22 / 3.27** (1-D toy). O2's `δ = d/2` criterion over-predicts the safe ratio by 1.9–7.3×.
  Term (b) manufactures exactly this. **Owner: spoke A, before term (b) ships.**
- ⛔ **R2: §A31.2's "σ_q/spacing = 0.32" is undefined between a norm and a per-coordinate reading**,
  and the two give 1-NN = **1.0000** vs **0.74–0.84**. A banked 1-NN launder of 0.79–0.90 is
  compatible with only one of them. **1 engineer-hour, zero compute, and it may dissolve A31.2.**
- ⛔ **R3: term (a) is unauditable until the placing write uses `c = stop_gradient(z_addr) + jig`.**
  The designed write→φ trainer and the measured **27 %** accidental leak share one code path.

**§8 (open questions) — two.**
- **A31.2 is priced, not solved.** Candidate list C1–C8 with costs; **C5 (confinement radial bias),
  C6 (non-capturing wells) and C7 (unsettled read) are REFUTED at zero compute**; C1 is
  half-discriminated already. Recommended order C1 → C2 → C3 → C4; only C4 costs compute (minutes).
- **New cross-term conflict, unflagged anywhere I could find: terms (c) and (e) compete for `λ_min`.**
  If (c) succeeds, written sites become spectrally indistinguishable from vacuum in `λ_min`, and the
  novelty head must key on **depth and `λ_2nd`/participation ratio**, never `λ_min`.

**§10 (record).** `LOSS-PACKAGE-DONE.json` filed: **`loss_package_complete = true` (6/6 formalized)**,
with **4 declared DEPLOYMENT BLOCKERS (DB1–DB4)** so "formalized" is never read as "shippable", and
the staging ruling: ⭐ **STAGE 0 = term (a) ALONE — it is the only term whose gradient does not route
through a settled read or an implicit solve, and therefore the only one that escapes the measured
`1e-10…1e-9` init floor.**
