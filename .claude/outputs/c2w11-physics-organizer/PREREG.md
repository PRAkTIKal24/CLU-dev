# PREREG — c2w11-physics-organizer (spoke B)

**Filed 2026-08-11, BEFORE the first cell of any registered quantity.** Base `main @ 168a892`,
worktree `/Users/user/Desktop/CHLU-c2w11b`, branch `c2w11-physics-organizer`.
Only thing run before this file existed: an **instrument timing probe** (`t0_probe.py` — JAX import
18 s, write 6.6 s, read 128+512 queries 8.2 s, `fam.tol = 0.286960063782279` reproducing the frozen
post-repair value). ⛔ No score, no reader, no registered statistic was computed by it.

## 0. Frozen inputs taken from the JSONs, never re-derived
`family.tol = 0.286960063782279` · `family.chance_per_seed = [0.0, 0.001953125, 0.0]`
(`chance = 0.00065104`) · `N_a=32, F=4, K=128, m=8, a=12, d_addr=4` · `payload_radius = 0.60`,
`atom_payload_init_radius = 0.60` · `atom_width_frac_spacing = 0.37` (selected; the harness refuses
any other) · launch = feature-factored `k=4`, `sigma_q=0.15`, shell `R=2.0`, payload block pinned 0 ·
φ = `build_phi(cfg, phi_seed=20260801)`, `phi_bytes=576`, hash `a2713a0f…` ·
reader class = `{sum_linear, well_table, knn, mlp, zero_parameter_identity}` ·
`v3_budget_grid = [50,100,200,400,800,1200]` total Verlet steps × 4 particles evolved,
split `address = round(b/3)`, `read = b − round(b/3)`, `dt=0.05`, `γ_addr=0.05`, `γ_read=0.02` ·
`k8_structural_split = (N_a=32, F=4, K=24, m=8)`.

## 1. Registered protocol decisions (fixed here, never a judgement call later)

**P1 — the ψ budget is SET BY THE MEASURED LEAK, not chosen.** ψ capacity grid
`hidden ∈ {8, 16, 32}`, `depth = 2`, pooled-sum DeepSets only. Run K4 leg 1 (blank store) and leg 2
(query-only) at each capacity; the deployed ψ is the **largest** capacity whose every K4 leg is
`≤ chance + 0.05`. If even `hidden = 8` leaks, the family is VOID and I report that, not a score.

**P2 — ψ is an ADDED reader, never a substitution.** V1 is scored on the frozen 5-member class
**plus** `deepsets_psi` labelled as an addition, params ledgered. The `≥ 3 of 4 readers` structure of
V1 is evaluated on the frozen class; ψ's score is reported beside it.

**P3 — K5's abstain rule** (registered by the gate, echoed): K5 abstains iff BOTH the per-item table
AND the arm's best read score are `≤ chance + δ`, `δ = 0.01`. Otherwise K5 scores; bar = the read
beats the table by `> 0.10` on ≥ 1 reader.

**P4 — the organizer's trainable object.** φ is frozen (byte-identity is a wave invariant), so the
organizer trains **the placement** — a per-well jig `J ∈ R^{N_a×d}` on top of φ's frozen code
geometry (`u_j = R·e_j + J_j`, then the shipped separation relaxation) — plus the store parameters θ
through terms (b) and (c). ⭐ This is exactly what the swap contests: *who decides where the wells
go*. The jig is byte-ledgered (`N_a·d·4 = 512 B`) and is the arm's only organizer parameter.
**Bound:** `‖J_j‖ ≤ jig_max = 0.5 × σ_q/…` — registered as `jig_max_frac_capture = 0.75` of the
measured SC-6 capture radius, so the organizer can never move a well out of its own cue's reach
(the payload-reach trap in mirror image).

**P5 — staging (w20, mandatory).** Stage 0 = term (a) ALONE on the jig (algebraic channel, O(1) from
step 0). Then the placing write digs the wells. Stage 1 = terms (b), (c) on θ. Stage 2 = terms (d),
(e) on ψ / the novelty head. ⛔ Grad norms at init are emitted for every term to show the staging is
a measurement and not a preference.

**P6 — traversal trigger threshold, registered before the run.** At the end of the address phase
(step 400) a particle's **causal diamond** is the ball of radius `reach = 2.0 × s_measured` about its
current position (spoke A's registered coverage reach). For each query, `frac_unreachable` = fraction
of needed wells `j ∈ A(x)` that are (i) not occupied by any particle and (ii) outside **every**
particle's diamond. **TRAVERSAL FAILURE FIRES iff `mean_x frac_unreachable > 0.20`** (spoke A's
coverage threshold, deliberately identical). If it fires I append a dated section to
`TRAVERSAL-FAILURE-SIGNATURE.md`; if not, I create nothing and say so.

**P7 — M3's declared target.** For query `x` and true feature `f ∈ A(x)`: if some launch channel `c`
selected code `f`, M3 asks whether particle `c`'s settled point resolves to well `f` **and** sits
inside well `f`'s **measured** SC-6 capture radius. Score = mean over `(x, f∈A(x))` pairs with a
launch channel asserting `f`; `any_basin` and `margin_in_SE` reported beside every boolean.
Bar = `max(4·chance_M3, chance_M3 + 2 SE)` with `chance_M3 = 1/N_a = 0.03125` ⇒ **0.125**.

## 2. ⭐ NUMERIC PREDICTIONS (scored in §Scorecard at the end)

| # | quantity | point | band | P(the stated event) |
|---|---|---|---|---|
| **B1** ⛔ | **K5 on the ORGANIZED arm — does it SCORE (not abstain)?** i.e. best read **or** table `> chance + 0.01` | best read **0.05** | [0.002, 0.25] | **P(scores) = 0.60** |
| **B2** | K5 PASSES (read beats table by > 0.10 on ≥ 1 reader) | margin **+0.02** | [−0.05, +0.20] | **0.20** |
| **B3** | K4-at-full-ψ: every leg ≤ chance+0.05 (family sound) | blank-store leg **0.02** | [0.00, 0.10] | **0.75** |
| **B4** | ψ budget selected by P1 | `hidden = 16` | {8,16,32} | P(32 survives) = 0.35 |
| **B5** | **V1** physics arm best-reader exact-set acc, unseen | **0.06** | [0.00, 0.30] | P(> chance+0.05) = 0.30 |
| **B6** | V1 at K8 (`K=24 < N_a`) agrees in sign with headline | — | — | **0.70** |
| **B7** | **V2a** per-feature novelty AUROC (physics) | **0.82** | [0.55, 0.99] | P(> 0.60 floor) = **0.80** |
| **B8** | **V2b** set-level answer ECE (physics) | **0.12** | [0.03, 0.35] | — |
| **B9** | V2a's designed negative (permuted payloads) | **0.50** | [0.45, 0.55] | P(asserted ≈0.5) = 0.90 |
| **B10** | **V3-MECHANICS**: the curve is monotone and non-flat (spread max−min over the 6 grid points) | spread **0.04** | [0.00, 0.20] | P(non-flat, i.e. spread > 2 grains = 0.0039) = **0.55** |
| **B11** | V3-REPORTED read-compute ratio vs a matched static read | **> 1000×** | — | P(< 100×) = 0.10 |
| **B12** | **M3** per-feature G-ADDR | **0.22** | [0.05, 0.55] | P(≥ 0.125 bar) = 0.65 |
| **B13** | M3 `any_basin` (reported, NOT the leg) | **0.95** | [0.70, 1.00] | — |
| **B14** | **M7**: fraction of written sites carrying a soft direction (λ_min within 0.02 of the 2α floor **and** depth ≥ D_min **and** capture ≥ σ_q) under (c) live | **0.35** | [0.00, 0.80] | P(M7 two-sided positive) = **0.30** |
| **B15** | **M8**: participation-ratio excess (physics − coefficient-zero) at written sites | **+0.6** | [−0.5, +3.0] | P(coef-0 shows NO excess soft dirs) = 0.85 |
| **B16** | ⭐ **the weak-supervision arm** (the A31.4 inversion at the ORGANIZER level): V1(weak) − V1(label-free) | **−0.01** | [−0.10, +0.05] | P(helpful beyond 2 SE) = **0.25** · P(harmful beyond 2 SE) = 0.30 · P(tie) = 0.45 |
| **B17** | the C2W9 **traversal** trigger fires at P6's threshold | `mean frac_unreachable` **0.12** | [0.00, 0.60] | **0.40** |
| **B18** | term-(a) designed/accidental separation: `grad_norm(L_outer, φ)` is bitwise unchanged when the placement path is cut (accidental channel DEAD on this arm) | — | — | **0.85** (φ is frozen here, so the 27 % leak should be structurally absent — this is a prediction that it is, not an assumption) |

**Reasoning for the two predictions that carry the wave.**
- **B1/B5 (low).** Spoke A measured the un-organized arm at the metric floor (top physics score
  0.001953125 = ONE grain vs chance 0.0), with launch-head precision 0.2303 and correct-and-distinct
  0.92 of 4 — a read that visits ~1 of the 4 needed wells cannot sum 4 payloads. The organizer moves
  wells by at most 0.75 capture radii; it cannot manufacture the missing 3 visits. My honest prior is
  that ψ, not the store, is where any lift would come from — which is exactly why B3/K8 carry the load.
- **B7 (high).** A *dropped* channel's well is **never written**: the region is vacuum, where
  `‖∇V‖ ≈ 2α‖q‖` and depth ≈ 0, versus a written site with depth ≈ 0.30. That is a large, mechanical,
  store-conditional contrast, and the mask is drawn independently of `x` and acts on the **write**
  (§1(e) `N-e3`), so a query-only head is provably at base rate. I therefore predict a strong AUROC
  and register it now so that a strong AUROC is *not* readable as a leak discovered post hoc.

## 3. What I will NOT do (declared NOT-RUN, never nulls)
`OD`, `OD_min`, any swap verdict, any tier-ii/full-CLU verdict, any paper number · attention-ψ ·
loss term **(f) kinetics** · the hierarchy, lifecycle verbs, wormholes, learned `p₀` · any null arm.

*Filed before the first registered cell by the C2W11 spoke-B engineer.*

---

# ⭐ AMENDMENT §A1 (2026-08-11, same day) — filed BEFORE any claim cell, AFTER three diagnostics

Three cheap diagnostics were run between the filing above and the first claim cell. They are
**instrument/diagnostic** cells (no VALUE number, no K5 verdict, no score entered any selection):

| # | measurement | value |
|---|---|---|
| D-1 | cue→well displacement `‖u_j − R·e_j‖` on the coefficient-zero arm, seed 0 | median **0.0073**, p90 0.1618, max 0.3324; **every** cue's nearest well is its own index (32/32) |
| D-2 | fraction of matched-filter channel picks lying inside `A(x)` (unseen, seed 0) | **0.2266** (launch occupancy precision 0.2378, reproducing spoke A's banked 0.2303) |
| D-3 | ⭐ **the ORACLE-ADDRESSING read ceiling** — launch the `k=4` particles at the *needed* wells' address anchors (+σ_q), shipped 400+800 settle, **zero-parameter raw sum** of the settled payload blocks | **exact-set 0.8613** @ `tol`; 0.9961 @ 4×tol; per-particle `‖pay(q*) − v_occupied‖` median **0.0002** |

**What they establish, and why an amendment is warranted rather than silent tuning.** D-3 shows the
store + settle + payload composition **work**: the read is not inert, it reaches 0.86 exact-set the
moment addressing is correct. D-1 shows the placement is *already* cue-aligned, so the InfoNCE-on-
centroids instantiation of term (a) has nothing to fix. D-2 localises the entire deficit in the
**launch head**. ⛔ None of these three is a score of the physics arm on the family's metric under the
shipped protocol; none was used to select a coefficient.

**A1.1 — term (a) gains a second, registered instantiation, and BOTH are raced.**
`org_mode = "reach"` (new default for the `phys` arm): `L_reach = E_x Σ_{j∈A(x)} softmin_c ‖u_j −
q_c(x)‖` + a two-sided separation hinge — *place the wells where the store's own launches can reach
them*, which is §A20.3(c)'s guard written as a differentiable objective. It is **label-free** (it
consumes the frozen φ's launch geometry and the item's own composition; `y` is never read).
`org_mode = "nt_xent"` (the theorist's spec) is raced as the ablation on the same cells.

**A1.2 — the jig bound is now the MEASURED capture radius, as §P4 said it should be.** `jig_max =
0.75 × 0.896484375` (spoke A's measured SC-6 median at the selected payload radius) `= 0.6724`,
replacing the placeholder `0.75 × σ_q = 0.1125` that the smoke cell ran at.

**A1.3 — ⭐ NEW PREDICTION, registered before the run (this is the amendment's own falsifiable claim).**
The organizer moves the wells; it cannot move φ. So its ceiling on launch precision is the best
**bijective re-assignment** of wells to cue sites under the co-occurrence matrix
`P[j,c] = E[#times code c is picked | feature j present]`. Measured on the seen split, seed 0:
identity **0.2591** → **best bijective assignment 0.3378** → row-max (non-bijective) bound 0.3897.

| # | quantity | point | band | P |
|---|---|---|---|---|
| **B19** | launch occupancy precision achieved by the `reach` organizer | **0.30** | [0.25, 0.34] | P(> 0.3378 assignment ceiling) = **0.05** |
| **B20** | the ⭐ **consequent** prediction: exact-set needs ~all 4 of 4 wells (one missing payload contributes `‖v_j‖ = 0.60 > tol = 0.287`), and 0.3378×4 = **1.35 of 4** ⇒ **K5 abstains again** | — | — | **P(K5 abstains) = 0.85** (was 0.40 at B1) |

⛔ **B20 is registered as a prediction, not used as a reason to skip the cell.** K5 is run at full
budget on 5 seeds and on both organizer instantiations, and the abstain verdict is *measured*.
