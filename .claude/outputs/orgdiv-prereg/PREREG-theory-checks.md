# PREREG — `orgdiv-prereg`'s own numerical theory checks

**Written 2026-07-31 BEFORE any script in `.claude/scratch/orgdiv-prereg/` existed** (protocol §5
pre-registration rule). These are **theory sanity checks**, not results, and none of them is a
paper number. The *deliverable* pre-registration for C2W5 is `PREREG-TierII.md` beside this file;
this file only binds me, so that the derivations I lean on in that document were committed to
before they were measured.

Constants throughout (the shipped rig, `memory-gym-v0` §0): `V(q) = α‖q‖² − Σ_j A_j exp(−‖q−c_j‖²/2s²)`,
`α = 0.05`, `s ∈ {0.20, 0.30, 0.35}`, `A ∈ {0.7 … 6}`, two-phase read `dt = 0.05`,
`(γ₁,N₁) = (0.05,400)` then `(γ₂,N₂) = (0.02,800)`, Newtonian `M = I`, `p₀ = 0`, `T = 0`, float64.

---

## P1 — the merger threshold ("the composition window's lower edge")

**Derivation registered:** for two equal wells at `±h` on a line, `V''(0) = 2α + (2A/s²)e^{−u/2}(1−u)`,
`u ≡ h²/s²`. The midpoint stops being a maximum (the two wells merge into one minimum) when
`V''(0) > 0`, i.e. when `(A/s²) e^{−u/2}(u−1) < α`.

**Predictions.**
- P1a: at `A = 1, s = 0.30, α = 0.05` the root is `u* = 1.005 ± 0.010` ⇒ **merger threshold
  `d/s = 2h/s = 2√u* = 2.005 ± 0.010`** — i.e. **`d/s = 2` to within 1 %**, and the confinement
  contributes < 1 % of the threshold.
- P1b: the threshold is **weakly** dependent on `A`: over `A ∈ [0.7, 6]` at `s = 0.30` the
  predicted `d/s` threshold stays inside **[1.99, 2.03]**.
- P1c: a numerical minima-count along the axis (dense grid + sign changes of `V'`) transitions from
  1 minimum to 2 minima at the predicted `d/s` within **2 %**.

## P2 — the partition law: the CLU's basin boundary is a POWER DIAGRAM to leading order

**Derivation registered:** the axial separatrix between two Gaussian wells of amplitudes `A_i, A_j`
at separation `d` sits at offset `δ = ln(A_i/A_j)/(d/s² − 4/d)` from the midpoint
(`bprime-theory` R1/T2.1). Expanding, `δ = (s²/d)·ln(A_i/A_j)·[1 − 4s²/d²]⁻¹`. An **additively
weighted Voronoi (power) diagram** with weights `w_i = 2s² ln A_i` puts the boundary at
`(w_i − w_j)/(2d) = (s²/d) ln(A_i/A_j)`. Hence

> **the CLU's settled-point partition equals a power diagram up to a relative correction
> `[1 − 4(s/d)²]⁻¹ − 1 = 4(s/d)² + O((s/d)⁴)`.**

**Predictions.**
- P2a: measured settle boundary (bisection under the shipped two-phase read) vs the **power-diagram**
  prediction: relative error ≈ `4(s/d)² = 4/(d/s)²` ⇒ **0.21 at `d/s = 4.4`, 0.48 at `d/s = 2.9`,
  1.0 at `d/s = 2`.** Registered: the power-diagram boundary's relative error tracks `4/(d/s)²`
  **within a factor 2** over `d/s ∈ [2.9, 6]`, while the **δ-law** (which contains the correction) is
  accurate to **≤ 10 %** over the same range.
- P2b: at **equal depths** the measured boundary is the midpoint (= both Voronoi and power diagram)
  to `< 1e-3·d` ⇒ **`D = 0` exactly** (Prop D2a re-verified in the tier-ii control's language).
- P2c: the query mass `D` (fraction of a `σ_q = 0.15` isotropic query law whose settle disagrees with
  the assignment rule) at `A_i/A_j = 1.5`, `d/s = 2.9`: **`D_voronoi > 0.02`** and
  **`D_power / D_voronoi < 0.20`** — i.e. a power-diagram VQ captures most of the cell-shape term.

## P3 — the 3-body term (the tier-ii analogue of T5.4, and the only structural non-VQ term)

**Derivation registered:** a Voronoi/power partition is **pairwise** — cell boundaries depend only on
the two sites they separate (plus their weights). The CLU's boundary is a level set of a **sum over all
wells**, so a third well `k` shifts the `i–j` separatrix by an amount proportional to its gradient there,
`∝ exp(−½(d_k/s)²)` (the T5.5 suppression law).

**Predictions.**
- P3a: the measured `i–j` boundary shift caused by adding a third, equidistant well is **nonzero**
  (`> 1e-3·d`) at `d_k/s ≤ 3` and decays with `log₁₀`-slope matching `−(d_k/s)²/(2 ln 10)` to within
  **30 %** over `d_k/s ∈ [2.2, 4.0]`.
- P3b: the shift is **not** reproducible by any re-weighting of the power diagram *that is fitted
  pairwise* — operationally, the shift's **sign/magnitude depends on the third well's position**, so a
  single weight per site cannot absorb it in a 3-site configuration where the same pair is separated in
  two different third-well contexts. Registered: **two contexts differing by ≥ 2× in shift** at
  `d_k/s = 2.5`.

## P4 — sharing arithmetic for a factored store (exact rational)

**Derivation registered** (`bprime-theory` T1.3, corrected byte law
`ratio = [A_tot(D+2)/S + d]/(d+m)` with `A_tot` = an item's atom budget, `S` = items per shared atom).

**Predictions** (`d = 4, m = 1, n_spec = 0 ⇒ D = 5`):
- P4a: `S* = A_tot(D+2)/m = 7·A_tot`. For a factored item with `F = 4` attribute wells at `a = 1`
  atom/well (`A_tot = 4`) ⇒ **`S* = 28` items per well for matched bytes**; at `a = 8` ⇒ **`S* = 224`**.
- P4b: the private-fraction cap `p ≤ [(d+m)r − d]/[(D+2)A_tot]` gives, at `r = 2.0` and `A_tot = 4`,
  **`p ≤ 0.214`** (i.e. ≤ 0.857 of one atom private per item) — registered to 3 s.f.
- P4c: a factored store with `K` items, `N_a` wells, `F` wells/item has `S = KF/N_a`; the cat-test
  scale I will register (`K = 128`, `N_a = 16`, `F = 4`) gives **`S = 32 > S* = 28` at `a = 1`** ⇒
  matched bytes is **reachable at `a = 1` and unreachable for `a ≥ 2`** (`ratio(a=2) = 1.15×`,
  `ratio(a=8) = 2.20×` — registered to 3 s.f.).

## P5 — the quantizer bound (Theorem O1)

**Registered as a proof, with a numerical existence check:** under a settled-point read the query→latent
map's image is exactly the set of minima of `V_θ`, so a table with `N_min` rows reproduces the read for
**every** reader. Numerical check: on a 3-well store with 4000 queries, the number of distinct settled
points equals the number of minima (**≤ 1e-6 clustering tolerance**), and a `N_min`-row nearest-center
table reproduces the settled read on **≥ 1 − D** of queries. Registered: **exact agreement on
`1 − D_voronoi` of queries and no more.**

---

**Scoring rule.** Each prediction is ✅ (inside the registered tolerance), ◐ (right mechanism, wrong
constant — the corrected constant is then reported as the finding) or ⛔ (refuted). A ⛔ is a finding and
is reported as such; it is not quietly dropped. Nothing here is a paper number and none of it is a
tier-ii result — the tier-ii result does not exist and cannot exist until C2W5 runs.
