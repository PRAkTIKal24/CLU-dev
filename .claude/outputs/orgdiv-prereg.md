# orgdiv-prereg — physics-theorist report

**Task + acceptance criterion:** pre-register **tier ii** (§A13) — the organization-dividend metric, the
cat-test design, the matched-capacity non-physics organizer arms, and **falsifiers that can actually
fire** — so C2W5 is startable on day one. **Status: done.** Deliverable:
**`.claude/outputs/orgdiv-prereg/PREREG-TierII.md`** (+ `PREREG-theory-checks.md`, my own predictions,
written before any script existed). ⛔ **No runs, no code, no worktree, no repo edits. Git footprint:
NONE** (repo read-only at `main @ d4f56c8`, clean tree, zero branches).

> ## ⚠ RECONCILIATION LIST — needs an owner (protocol §5 corollary, first-10-lines rule). 2 items, both small.
> | # | site | correction | owner |
> |---|---|---|---|
> | **R-MERGE** | any site describing the gym rig's `d_safe_override = 0.58` geometry as having neighbouring **wells** (`memory-gym-v0` §2; `bprime-theory` T5.5 column *"where the gym actually runs"*; charter §A12 quoting `d/s ≈ 1.9–2.0`) | ⭐ add: **`d/s ≈ 1.93` is BELOW the merger edge.** Two equal-depth Gaussian wells merge into a **single** minimum below `d/s = 2.0005…2.0145` across the whole shipped `(A,s)` box (derived + verified to 8.7e-11). So at that override the pair is **at or past merger**, which is the mechanism behind the measured `λ_min` collapse — ⚠ **as a single-atom-width bracket, not a measurement of the learned store** (`s` for a learned multi-atom well is still unsolved, `bprime-theory` §9.2). No published number changes. | curator |
> | **R-COUPLE** | `bprime-theory` T5.5 / charter §A12+§A14.1, wherever the `exp(−½(d/s)²)` coupling is quoted | ⚠ add a scope word: that law is the **third well's gradient ratio at an item's own well**. The **boundary-position** shift it causes is a *different observable* with a *different argument*: `shift ≈ 1.19·exp(−½(D_k/s)²)` with `D_k` = the third well's distance **to the boundary** (measured log-slope **−0.962** vs −1). At `d/s = 2.9` the gradient ratio is `0.111` but the boundary shift is `1.5e-4·d_ij`. **Do not read T5.5's table as a partition-geometry effect size.** | curator + theorist (this file) |

---

## ⭐ DIAL DECLARATION (protocol §7, C2 form — echoed before the first result)
- **Dial / pillar:** **TIER ii — the organization dividend.** ⛔ **This task registers; it does not run
  and it does not claim.** Nothing below is a tier-ii result.
- **Control (registered, not mine to relax):** the **ORGANIZER SWAP** — physics-organized store vs a
  matched-capacity **non-physics** organizer at same φ, same bytes, same capacity, same readers.
  ⛔ The settle-deleted / matched-bytes launder is **tier i's** control and is the **wrong** control
  here; it is reported as an inherited diagnostic only.
- **Falsifies:** the falsifiers are the deliverable (`PREREG-TierII.md` §3). Each has a sign, a
  threshold, a tolerance and a seed count, and §7 below shows two of them **already fire** at operating
  points C2W5 might otherwise have chosen by default.
- **Does NOT falsify:** a derivation that narrows the claim · a tie · losing to a classical method on a
  metric-native protocol (standing theorem, not news).

## 0. Flag provenance (every number in this report)

| item | value |
|---|---|
| repo | **read-only**, `main @ d4f56c8`, clean tree, **no worktree, no branch, zero tracked-code edits** |
| code | pure **numpy 2.4.1 / scipy 1.17.0**, main venv `/Users/user/Desktop/CHLU/.venv` (py 3.11.13), **float64**, **no repo module imported anywhere**. Integrator = the shipped damped velocity-Verlet, `common.py` copied verbatim from `bprime-theory` (itself line-for-line the shipped form: half-kick, drift, half-kick, `p ← (1−γ)p`) |
| scripts | `.claude/scratch/orgdiv-prereg/`: `s1_merger.py · s2_partition.py · s3_quantizer.py · s3b_gmm_null.py · s3c_gamma.py · s3d_coherence.py · s4_sharing_bytes.py · s5_design.py · s6_frontier.py` (+ `common.py`), each writing its own `*.json` |
| PREREG | `.claude/outputs/orgdiv-prereg/PREREG-theory-checks.md`, written **before any script existed**; scorecard §8 |
| constants | `V = α‖q‖² − Σ_j A_j exp(−‖q−c_j‖²/2s²)`, `α = 0.05`; `s ∈ {0.20, 0.30, 0.35}` (0.30 default); `A ∈ {0.7…6}`; read `dt = 0.05`, two-phase `(γ,N) = (0.05,400) → (0.02,800)` unless a γ-sweep is named; `σ_q = 0.15` isotropic; **Newtonian `M = I`, `p₀ = 0`, `T = 0`**, no Langevin, no wake–sleep, **no training anywhere** |
| geometry | 1-D and 2-D toys, **single-atom wells**; byte arithmetic at `d = 4, m = 1, n_spec = 0 ⇒ D = 5` |
| seeds | `default_rng(0)`; every number is a **theory check**, none is a paper number |
| wall | ≈ 3 min total (s2 45 s · s3 20 s · s3b 28 s · s3c 33 s · s3d 18 s · s5 26 s · rest instant) |

---

## 1. What I did / how I verified

**How I verified (summary; per-claim detail is inline in §2–§7 and scored in §8):** every load-bearing
claim is (i) derived symbolically, then (ii) checked numerically on a toy that reproduces the shipped
damped velocity-Verlet read line-for-line, with the prediction **registered first** in
`PREREG-theory-checks.md`. Commands: `cd .claude/scratch/orgdiv-prereg && /Users/user/Desktop/CHLU/.venv/bin/python s{1,2,3,3b,3c,3d,4,5,6}*.py`
(≈3 min total, each writes its own JSON). No repo module is imported anywhere; no experiment was run.

1. Read §A13 in full plus §A14.1/§A14.7/§A14.8, §A4.5, §A9.8/§A9.9, §2.1, §A2, the intervention §5/§8,
   `bprime-theory` T1.3/T1.4/T3/T5.1–T5.3, `route3-stage1-plus-2x2`, `bprime-fb4-gate` §A3.7,
   `controller-doctrine`, and the C2W3/C2W4 §10 entries.
2. **Derived** what the tier-ii metric can and cannot measure, from the shipped dynamics: the
   quantizer bound (O1), the power-diagram equivalence (O2), the 3-body term (O3), the inertial term
   (O4), and the sharing/affordability arithmetic (O5).
3. **Verified each numerically** on toys reproducing the shipped read, and **pre-registered my
   predictions first**.
4. **Wrote `PREREG-TierII.md`**: the metric (with the organizer-swap control and the explicit
   permission for table-like inference reads), the cat-test construction with the rule-4 proof
   obligation, five falsifiers + five blocking pre-conditions, the five organizer arms with tuning
   budgets and the honest `null*`, the sharing-affordability check against `S*`, the deletion **curve**
   definition, the §5 design-rule compliance table, the operating point, the named open questions, and
   a declared NOT-DERIVED list.

## 2. ⭐ THEOREM O1 — the quantizer bound (why the cat test cannot use a settled-point read)

**Claim.** Under a settled-point read, the query→latent map `x ↦ q*(φ(x))` has image **exactly the set
of minima of `V_θ`**. Hence for **any** reader `R`, the composed read takes at most `N_min` distinct
values, and a table of `N_min` rows reproduces it **exactly**.

**Assumptions.** `Fix(T_θ) = {(q,0) : ∇V_θ(q) = 0}` (Prop Q1.1 / theorem T3) — the fixed-point set
contains **no** query variable; the settle converges (budget `C = Σ N_p ln(1/ρ_p)` large; the shipped
read has `C = 18.34`); ties on separatrices are measure-zero.

**Derivation.** `q*` is a fixed point, so `∇V_θ(q*) = 0`; the query enters only through `q₀`, which
selects *which* fixed point. Therefore `q*` is a **piecewise-constant** function of `x`, constant on
basins, with jumps on codimension-1 separatrices (T3's sharp form). The image is a subset of the
minima set, so `|image| ≤ N_min`. ∎

**Numerical verification** (`s3_quantizer.py`): 4000 queries (`σ_q = 0.15`) on a 2-well store give
**2 distinct settled points** at 1e-6 clustering tolerance in every non-merged cell (**1** in the
merged cell). A 41-point sweep of the *entire* segment between the wells gives **3** distinct settled
points at every `d/s ∈ {2.2, 2.5, 2.9, 4.4}` — the third being the separatrix itself (measure zero).

**Verdict: PROVEN** (the bound), **EVIDENCED** (the numeric image counts).

**Implication for CHLU — decision-grade.** ⛔ **A compositional answer cannot live in a settled point.**
The settle returns "which basin", nothing more; the "composition between basins" of §A4.5 is *not* a
settled position. ⇒ **the cat test's read must be a multi-particle occupancy read** (`P` designed
launches → the *set* of wells visited; image up to `N_min^P`), which is exactly §A4.5's multi-particle
read, and this is registered as **mandatory**, not optional. *(Corollary the engineer needs: with `P`
launches the per-item content is still bounded — `bprime-theory` T5.4(4)'s `3d+1` launch-manifold rank
— so, as in Route 3, any claim must be **cross-item/set-valued**, never per-item richness.)*

## 3. ⭐⭐ THEOREM O2 — the VQ ceiling (the tier-ii analogue of Prop D2a, and it is the load-bearing result)

**Claim.** To leading order the CLU's settled-point partition **is a power diagram** (additively
weighted Voronoi / Gaussian-mixture MAP) with weights `w_j = 2s² ln A_j`. Exactly, the axial boundary
between wells `i,j` at separation `d` sits at
### `δ = ln(A_i/A_j)/(d/s² − 4/d) = (s²/d)·ln(A_i/A_j)·[1 − 4(s/d)²]⁻¹`
i.e. **the power-diagram offset amplified by `[1 − 4(s/d)²]⁻¹`** — and at (approximately) uniform site
spacing, **a single fitted weight scale absorbs the amplification entirely.**

**Assumptions.** Equal widths `s`; two dominant wells; `|δ| ≤ 0.25 d`; overdamped-or-settled read
(inertia not dominant — see O4 for where this fails); confinement negligible at the boundary.

**Derivation.** The separatrix is where the **force** balances, not the potential:
`A_i e^{−a²/2s²}·a/s² = A_j e^{−b²/2s²}·b/s²` with `a = d/2 − δ`, `b = d/2 + δ`. Taking logs,
`ln(A_i/A_j) = (a²−b²)/2s² − ln(a/b) = −dδ/s² + 4δ/d + O(δ³)`, giving the law above. A power diagram
balances **values** (`A_i e^{−a²/2s²} = A_j e^{−b²/2s²}`), which drops the `ln(a/b)` term — the entire
difference between the two partitions is that one term. ∎

**Numerical verification** (`s2_partition.py`, `s3_quantizer.py`, `s3b_gmm_null.py`; bisection of the
launch coordinate under the shipped two-phase read, 45 iterations):

| `d/s` | power-diagram relative error (measured) | `4(s/d)²` (predicted) | δ-law relative error |
|---|---|---|---|
| 6.0 | 0.138–0.151 | 0.111 | 0.030–0.045 |
| 4.4 | 0.214–0.216 | 0.207 | 0.010–0.012 |
| 3.5 | 0.333–0.339 | 0.327 | 0.010–0.019 |
| 2.9 | 0.483–0.549 | 0.476 | 0.014–0.141 |
| 2.5 | 0.655–0.688 | 0.640 | 0.043–0.132 |

*(depth ratios 1.25–3.0; the equal-depth rows are excluded as `0/0`. Ratio of measured error to
`4(s/d)²`: **1.015–1.363**, the tail at `d/s = 6` being the confinement term, which is no longer
negligible against the Gaussian tails at that separation.)*

The **fitted** amplification at `d/s = 2.9` is **`c = 1.940` against `1.907` predicted (1.7 %)**. The
dynamic boundary equals the static separatrix to all printed digits at every cell.

**⇒ The consequence that decides the wave** (query mass `D` on which the settle disagrees with the
assignment rule; 4000–6000 queries/cell, `σ_q = 0.15`):

| `d/s` | depth ratio | `D` vs plain Voronoi | vs power (theory `w`) | vs **fitted** power | vs **fitted GMM-MAP** (spherical cells) |
|---|---|---|---|---|---|
| **4.4** | 1.5 / 3.0 / 6.0 | **0.0000** | 0.0000 | 0.0000 | 0.0000 |
| 2.9 | 1.5 | 0.0073–0.0080 | 0.0057–0.0060 | **0.0010** | 0.0010 |
| 2.9 | 3.0 | 0.2472 | 0.2388 | 0.1933 | **0.1933** |
| 2.5 | 1.5 | 0.2340 | 0.2267 | 0.2002 | **0.2002** |

**Verdict: PROVEN** (the leading-order equivalence and the correction term), **EVIDENCED** (the error
law and the fitted amplification).

**Implications for CHLU.**
- ⛔ **At the designed admission gate (`d_safe = 4.4 s`) the physics-organized store's settled-point
  read is EXACTLY a nearest-centroid VQ** — `D = 0.0000` at every depth ratio up to 6. **A tier-ii
  experiment run at the designed gate has a structurally zero organization dividend against a VQ with
  the same codebook, for every reader.** This is Prop D2a re-derived one level up, in the tier-ii
  control's own language, and it must be in the prereg so nobody spends a wave discovering it.
- ⇒ **the only surviving tier-ii claims are (i) a PLACEMENT claim (where the codebook goes — which
  `null*` attacks head-on) and (ii) whatever survives at `d/s ≤ 2.9` (O4).**
- ⭐ **It also tells the engineer how to build the strongest null** (the N78 "rescue the baseline"
  lesson, done with algebra instead of hindsight): the honest static null is not k-means — it is a
  **fitted power/GMM-MAP diagram**, which is *provably* the physics arm's own decision rule to leading
  order.

## 4. Prop O3 — the 3-body term is real, is provably outside the VQ class, and is unusable

**Claim.** A power/Voronoi partition is **pairwise**: the `i–j` boundary depends only on `(c_i,c_j,w_i,w_j)`.
The CLU's boundary is a level set of a sum over **all** wells, so a third well shifts it — and the
shift **flips sign** with the third well's side, so no assignment of per-site weights can absorb it.

**Numerical verification** (`s2_partition.py` §b, `s3_quantizer.py` §C; `d_ij/s = 2.9`, equal depths,
third well collinear): shifts `±6.317e-4` (`d_k/s = 2.5`, exactly antisymmetric between the two
sides), `±1.331e-4` (2.9), `9.32e-6` (3.5), `7.62e-7` (4.0), `8.53e-8` (4.4), `2.35e-9` (5.0). Fitting
`ln|shift|` against `½(D_k/s)²` with **`D_k` = the third well's distance to the boundary** gives slope
**−0.962** (predicted −1.0) and prefactor 1.19.

**Verdict: PROVEN** (outside the power-diagram class), **EVIDENCED** (the `exp(−½(D_k/s)²)` law).
⛔ **But the magnitude is `≤ 7e-4 · d_ij` at every admissible spacing** ⇒ **tier ii may not be built on
it.** *(This is not a contradiction of `bprime-theory` T5.5's `0.111` at `d/s = 2.9`: that is a
**gradient ratio at an item's own well**, a different observable — see R-COUPLE.)*

## 5. ⭐⭐ Prop O4 — the inertial term: the one non-VQ channel with usable magnitude, and it is γ-gated

**Claim.** At small `γ` and interacting spacing, a substantial fraction of query mass is assigned to a
well that **no fitted static-geometric rule reproduces** — and that fraction is **momentum-carried**,
collapsing as `γ` grows, with the large-`γ` limit being literally "assign the launch point" (monitor #1).

**Numerical verification** (`s3c_gamma.py`, `s3d_coherence.py`; irreducible disagreement = the minimum
over a fitted GMM-MAP family, i.e. over all planar **and** spherical cells):

| store | γ = 0.01 | 0.02 | **0.05 (shipped)** | 0.1 | 0.2 | 0.5 | 0.9 |
|---|---|---|---|---|---|---|---|
| `d/s = 2.9`, ratio 3.0 | 0.173 | 0.189 | **0.197** | 0.040 | 0.0003 | 0.0003 | 0.0005 |
| `d/s = 2.5`, ratio 1.5 | 0.174 | 0.185 | **0.203** | 0.032 | 0.0000 | 0.0000 | 0.0000 |
| `d/s = 2.9`, ratio 1.5 | 0.051 | 0.031 | **0.0013** | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

⚠ The `γ ≤ 0.02` rows are partly *unconverged* settles (3960–3971 distinct "settled" points at
γ = 0.01 — the read has not finished; `C = 1200·½ln(1/(1−γ))` is only ≈ 6 there), and the `γ = 0.9`
rows are partly **monitor #1** (the particle barely moves, so `q* ≈ q₀` and the assignment is Voronoi
by construction). **The honest window is `γ ∈ [0.05, 0.1]`.**

**Is it information or chaos?** (`s3d_coherence.py`, shipped read): assignment flip rate under a launch
perturbation of `0.1 σ_q` is **0.0 %–2.0 %**; leave-one-out **1-NN** prediction of the assignment from
the launch coordinate is **0.9893–1.0000** (15-NN 0.988–1.000). ⇒ **coherent, not chaotic.**

**Verdict: EVIDENCED** (measured on toys; the mechanism — momentum carrying the particle past the
static watershed — is T4.1's inertial boundary, here quantified as *query mass* rather than as a
capture radius).

**Implications for CHLU — two, both binding.**
1. ⭐⭐ **The tier-ii dividend can only be measured OFF the training support.** In-distribution, a kNN
   null recovers ~99 % of the only channel that distinguishes the physics arm from a fitted geometric
   organizer. The unseen-combination design is therefore **derived, not preferred** — and the metric's
   held-out split is load-bearing, not decoration.
2. ⭐ **γ is the tier-ii operating dial.** At `γ ≥ 0.2` the physics organizer *is* a VQ (a ×600 collapse
   of the non-VQ mass); at `γ ≤ 0.02` the read has not settled. This is monitor #1 seen from the
   organization side, and it gives C2W5 a free **internal control**: run `γ = 0.2` as a
   *physics-disabled* arm of the same code path.

## 6. Prop O5 — sharing affordability: the arithmetic, in advance (task §3 "sharing is unaffordable")

**Claim.** With the corrected byte law, `ratio = [A(D+2)+d]/(d+m)` with `A = N_at/K`, i.e. at
`d=4, m=1, n_spec=0`: ### `ratio = 1.4·(N_at/K) + 0.8` — **the byte ratio depends only on effective
atoms per live item.** Sharing is *only* a device for lowering `N_at/K`; matched bytes ⇔
`N_at/K ≤ m/(D+2) = 1/7 = 0.1429`, equivalently `S ≥ S* = A_tot(D+2)/m = 7 A_tot`.

**Verification** (`s4_sharing_bytes.py`, exact `Fraction` arithmetic): reproduces `bprime-theory` T1.3
exactly — `S* = 7` at `A_tot = 1`, **2387** at 341, `p ≤ 4.19e-4` at `r = 1, A_tot = 341`,
`dp/dr = 2.095e-3`, floor `2.20×`.

**The cat test's own numbers** (`N_a` wells, `F = 4` wells/item, `a` atoms/well, `K` items,
`S = KF/N_a`):

| design | `S` | `S*` | `ratio` | note |
|---|---|---|---|---|
| `N_a=32, K=128, a=1` | 16 | 28 | **1.15×** | bytes nearly matched, **1 atom per well** |
| `N_a=32, K=128, a=12` | 16 | 336 | **5.00×** | ⭐ registered default (write has a chance) |
| `N_a=16, K=128, a=1` | 32 | 28 | **0.975×** | ⛔ **byte-matched but COMBINATORIALLY VOID** — see §7 |
| matched-bytes frontier, `a = 11.65` | 326 | 326 | **1.000×** | needs `N_a = 44`, **`K ≈ 3 589` items** |
| matched-bytes frontier, `a = 341` | 9548 | 9548 | 1.000× | needs `K ≈ 372 372` items |

> ⭐⭐ **The registered affordability answer, stated now rather than after a wave:** the shipped write
> **converges** at 341 atoms/item (loss 2e-4, `λ_min +3.24`) and **fails** at ≤ 11.65 (loss 0.20–0.24,
> `λ_min ∈ [−1.20, −0.21]`). Matched bytes needs `N_at/K ≤ 0.143`. ⇒ **a byte-matched cat test at the
> write's known failure edge needs ≈ 3.6 k items; at its converged budget, ≈ 372 k.** ⛔ **C2W5 must not
> promise a byte-matched tier-ii result.** Tier ii's control is the organizer swap, which is
> byte-matched **across arms by construction**; the table-relative ratio is *reported*, never claimed.
> ⚠ **One genuine unknown that could move this a lot:** the shipped failure evidence is per **item**-well;
> a factored write digs `N_a` wells, not `K`. Whether the write converges at `a ≈ 12` atoms **per well**
> is **open**, and it is pre-condition **K1**.

**Deletion curve (§A9.9), registered:** x-axis `p = a_priv/(a_priv+Fa)` (equivalently `r`), two series —
byte-exactness on the private fraction (1.0 by construction; `AUC(z_hole)`) and **measured degradation**
on the shared fraction (read error + MIA-AUROC after leaving shared atoms; wall-clock after re-fitting
them, which is a *write*, i.e. the retraining baseline). Anchors: `A_tot = 4 ⇒ p ≤ 0.036 (r=1) /
0.214 (r=2) / 0.286 (r=2.4) / 0.750 (r=5)`; `A_tot = 32 ⇒ 0.0045 / 0.027 / 0.036 / 0.094`.

## 7. ⭐ The cat test is constructible — and the obvious design point is void

Rule 4 (*the answer is provably not in the table*) is discharged **combinatorially**: every held-out
well-set must satisfy `|A ∩ B| ≤ F−2` for every stored `B`, so any single stored row is wrong in ≥ 2 of
`F` wells. One stored item blocks `F(N_a−F)+1` combinations, so the held-out count is
`≈ C(N_a,F)·exp(−K·B/C(N_a,F))`. Verified against explicit greedy constructions (`s5_design.py`, F=4):

| `N_a` | `K` | total | **rule-4-valid held-out** |
|---|---|---|---|
| 16 | 64 | 1 820 | 148 |
| **16** | **128** | 1 820 | ⛔ **0** |
| 24 | 128 | 10 626 | 3 358 |
| **32** | **128** | **35 960** | ✅ **23 193** |
| 32 | 256 | 35 960 | 14 261 |

⇒ ⛔ **`N_a = 16` at `K ≥ 128` is registered as FORBIDDEN: the held-out split does not exist.** Note
this is exactly the configuration that looked best on bytes (0.975×) — **the byte-optimal design is the
combinatorially void one**, which is the sort of thing that is cheap to find now and expensive to find
in week three. Registered design point: **`N_a = 32, F = 4, K = 128, a = 12`**.

## 8. PREREG scorecard (`PREREG-theory-checks.md`, written before any script existed)

| # | registered | measured | verdict |
|---|---|---|---|
| P1a | merger edge `d/s = 2.005 ± 0.010` at `A=1, s=0.3`; confinement < 1 % | **2.0074** | ✅ |
| P1b | threshold ∈ [1.99, 2.03] over `A ∈ [0.7,6] × s ∈ [0.2,0.35]` | **[2.0005, 2.0145]** | ✅ |
| P1c | numeric minima-count transition within 2 % of analytic | **8.5e-11 … 8.7e-11** | ✅✅ |
| P2a | power-diagram relative error tracks `4(s/d)²` within ×2 over `d/s ∈ [2.9,6]`; δ-law ≤ 10 % | error/`4(s/d)²` = **1.015–1.363** over 18 depth-ratio cells (the 1.24–1.36 tail is at `d/s = 6`, where the **confinement** `2αq` is no longer negligible against the Gaussian tails); δ-law **0.96–4.53 %** in **16 of 18** cells, and the **2** exceptions (13.2 % at `d/s=2.5,ratio 1.5`; 14.1 % at `2.9, ratio 3`) are exactly the cells outside its own declared domain (`s/sep = 0.40 > 0.30`; `|δ| = 0.252 > 0.25d`) | ✅ (the δ-law's domain statement predicts its own two failures) |
| P2b | equal depths ⇒ boundary at the midpoint to `< 1e-3·d`; `D = 0` | boundary **−1.24e-14**; `D = 0.0000` at `d/s = 2.9, 4.4` (**0.0120** at 2.5 — the *inertial* width, not asymmetry) | ✅ |
| P2c | `D_voronoi > 0.02` and `D_power/D_voronoi < 0.20` at `d/s=2.9, ratio 1.5` | **`D = 0.0073–0.0080`** ⛔ (2.5× smaller than registered); theory-weight ratio **0.71** ⛔; ⭐ **fitted-weight ratio 0.125** ✅ | ⛔→◐ **refuted as registered; the corrected object (a FITTED power diagram) is what the honest null must be — and that changed §4 of the deliverable** |
| P3a | 3-body shift decays with log-slope matching `−(d_k/s)²/2` within 30 % | slope **−0.962** vs −1 **only after correcting the variable to `D_k` = distance to the BOUNDARY** | ◐ **mechanism confirmed, variable corrected** |
| P3b | two third-well contexts differ by ≥ 2× at `d_k/s = 2.5` | shifts are **exactly antisymmetric (±6.317e-4)** — the sign flips, which is strictly stronger than a 2× magnitude gap | ◐ **confirmed in substance, sharper than registered** |
| P4a | `S* = 7A_tot`; 28 at `F=4,a=1`; 224 at `a=8` | **exact** | ✅ |
| P4b | `p ≤ 0.214` at `r=2, A_tot=4` | **0.2143** | ✅ |
| P4c | `N_a=16,K=128` ⇒ `S=32 > S*=28`, ratios 0.975 / 1.15 / 2.20 at `a = 1/2/8` | **exact**, ⛔ **but that design point has ZERO rule-4 held-out queries** — byte-feasible, combinatorially void | ✅ arithmetic / ⛔ **as a design point** (found by `s5`, changed the registered design) |
| P5 | image cardinality = number of minima; table reproduces on `1 − D` | **2 distinct settled points from 4000 queries** (1 when merged); 3 on the full segment (the third = the separatrix) | ✅ |

**Score: 8 ✅ · 3 ◐ · 1 ⛔-then-corrected.** Both corrections changed the deliverable: P2c made the
**fitted** geometric diagram the registered honest null (arm N3), and P4c moved the registered design
point from `N_a = 16` to `N_a = 32`.

## 9. Open questions / follow-ups / risks

- **OQ-A (blocking, first day of C2W5).** The **effective `s` of a learned multi-atom well** is
  unmeasured; every `d/s` statement here is a single-atom bracket. Cheap instrument: fit
  `A e^{−r²/2s²}` to each written well's radial profile. Until then the operating point is a spec, not
  a measurement.
- **OQ-B (decides F4).** Does the write converge at `a ≈ 12` atoms **per well** in a *factored* store?
  The shipped failure evidence is per **item**-well and does not transfer automatically.
- **OQ-C.** Does the inertial term (O4) **extrapolate**? It is coherent and non-VQ in-distribution;
  its off-support behaviour is the tier-ii hypothesis and is untested. If it does not, F1 fires.
- **OQ-D.** `bprime-c6`'s re-located `B` — dependency declared in the prereg; if it lands, §7's band
  should be reconciled with it (they should agree: `B = 0.33` ⇔ `d/s ≈ 2.9`, which is inside my band).
- **OQ-E (a real risk to the whole vehicle).** The **query-only leak**. A compositional target computed
  from the query's own attribute set is trivially predictable unless the per-well payloads `v_j` are
  store-only. The prereg fixes this by construction (`y = Σ_{j∈A(x)} v_j`, `v_j` drawn at write time),
  and control §2.5(2) is the check — **this is the most likely way the family dies, and it dies
  cheaply.**
- **Risk.** Everything here is Newtonian, `p₀ = 0`, `T = 0`, untrained, 1-/2-D, single-atom wells. The
  *structural* statements (O1, O2's leading order, O3's pairwise argument, O5's arithmetic) are
  geometry-independent; the *constants* are not.

## Git footprint
**NONE.** No tracked code touched, no branch, no worktree; repo read-only at `main @ d4f56c8`, clean
tree. All artifacts under `.claude/outputs/orgdiv-prereg/` and `.claude/scratch/orgdiv-prereg/`.

---

## Proposed handover updates (for the Hub)

**§1 (physics addendum) — three additions, all new and all decision-grade.**
- ⭐⭐ **Theorem O2 (the VQ ceiling).** The CLU's settled-point partition **is a power diagram** with
  `w_j = 2s² ln A_j`, amplified by `[1 − 4(s/d)²]⁻¹` (fitted amplification **1.940** vs **1.907**
  predicted). ⇒ **at the designed gate `d_safe = 4.4 s` the settled-point read is EXACTLY a
  nearest-centroid VQ** — disagreement `D = 0.0000` at depth ratios 1.5 / 3.0 / 6.0. **Any organization
  claim measured at the designed gate is structurally zero**, and the honest static null is a *fitted*
  power/GMM-MAP diagram, not k-means. *(Prop D2a, one level up, in tier ii's own control language.)*
- ⭐⭐ **Theorem O1 + Prop O4.** A settled-point read's image is the **minima set** ⇒ an `N_min`-row
  table reproduces it for **any** reader ⇒ **composition cannot live in a settled point; multi-particle
  occupancy reads are mandatory** for the factored store. The only non-VQ channel with usable magnitude
  is **inertial**: 19–20 % of query mass at `γ = 0.05, d/s ∈ [2.5,2.9]`, depth ratio ≥ 3 — collapsing
  **×600 by `γ = 0.2`** (monitor #1 from the organization side) and **99 % recoverable by 1-NN
  in-distribution**, which is *why* tier ii must be scored on unseen combinations.
- ⭐ **The merger edge.** Two equal wells merge into ONE minimum below **`d/s = 2.0005…2.0145`** across
  the shipped `(A,s)` box (axial curvature `2α + (2A/s²)e^{−u/2}(1−u)`, `u = (d/2s)²`), and **at
  `d/s = 2.00` the axial curvature is exactly `2α = 0.1000`** — the **2α coercivity floor and the
  merger edge are the same object**. The gym's `d_safe_override = 0.58` sits at `d/s ≈ 1.93`, i.e. **at
  or past merger** (single-atom bracket).

**§7 (known issues / live) — two entries.**
- ⚠ **R-MERGE:** sites describing the `d_safe_override = 0.58` rig as having neighbouring *wells* need
  the merger-edge qualifier (`d/s ≈ 1.93 < 2.007`), flagged as a single-atom bracket.
- ⚠ **R-COUPLE:** `bprime-theory` T5.5's `exp(−½(d/s)²)` is a **gradient ratio at an item's own well**;
  the **boundary-shift** observable obeys `≈1.19·exp(−½(D_k/s)²)` with `D_k` = distance **to the
  boundary** and is `≤ 7e-4·d_ij` at every admissible spacing. Do not read one as the other.

**§8 / §10 (record).**
- ⭐ **Tier ii is pre-registered**: `.claude/outputs/orgdiv-prereg/PREREG-TierII.md` — metric
  (organizer swap; table-like reads permitted on both arms, stated in the metric), cat test
  (`y = Σ_{j∈A(x)} v_j` with `v_j` store-only; rule 4 discharged as `|A∩B| ≤ F−2`), **5 falsifiers +
  5 blocking pre-conditions run before the arms are compared**, 5 organizer arms with a **computed**
  `null* = max over arms × tuning grid`, the affordability check, the deletion curve, the compliance
  table, and the operating point.
- ⭐ **Registered design point `N_a = 32, F = 4, K = 128, a = 12`** (23 193 rule-4-valid held-out
  combinations; ratio 5.00×, reported not claimed). ⛔ **`N_a = 16` at `K ≥ 128` is FORBIDDEN — the
  held-out split is empty**, and it is precisely the byte-optimal design (0.975×).
- ⭐ **Affordability, costed in advance:** `ratio = 1.4·(N_at/K) + 0.8` depends **only** on effective
  atoms per live item, so matched bytes ⇔ `N_at/K ≤ 1/7`. **A byte-matched cat test needs ≈ 3.6 k items
  at the write's known failure edge (11.65 atoms/well) and ≈ 372 k at its converged budget (341).**
  ⇒ **C2W5 promises no byte-matched tier-ii result**; the organizer swap is byte-matched across arms by
  construction.
- ⭐ **Pre-registered prior on the tier-ii outcome** (so it is not post-hoc): `P(clear) ≈ 0.20`,
  `P(tie) ≈ 0.55`, `P(physics loses) ≈ 0.25` — because at the designed gate the arms are provably the
  same function, and off it they differ only by an inertial term that is 99 % kNN-recoverable
  in-distribution. **A tie is a finding** (tier ii becomes "a competitive organizer with byte-exact
  deletion", an "and also" claim), not a null.
