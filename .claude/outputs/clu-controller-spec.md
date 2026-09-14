# clu-controller-spec — physics-theorist report

Task + acceptance criterion: derive the conditions a CLU controller must satisfy (formal, labelled), mechanise the three decisions (write-near / open-new / trash), deliver a minimum-viable controller spec with its non-differentiability debt itemized, and state what no controller can fix — with numerical sanity checks where they settle something.
Status: **done** (14 pre-registered predictions + 2 instrument amendments, all run and scored; PREREG written before any harness ran).

> **Reconciliation note (first-10-lines rule):** no shipped number is retracted. Two of my own pre-registered instruments failed *as registered* and were amended with the failure recorded (§6): **P3** (the registered θ-probe had no signal — itself a verification of exponential write locality) and **P12** (the registered Newton-based re-derivation **wrote a saddle into the codebook** — an instrument artifact that is also a genuine controller design lesson, now condition **C4.3**). One cross-report consequence needs an owner: **the re-derivation operator in ANY derived-address implementation must be the γ>0 relaxation with a minimum-check, never a critical-point solver** — this constrains `relaxation-addressing-theory`, `learned-landscape-write-read`, and any engineer build of derived addresses.

**What I did:** formalized the controller's contract as five conditions C1–C5 (§2), each with a proof-level statement, a status label, and a measured violation counterexample; mechanised the three decisions as trigger/action/failure-mode rules over quantities CLU actually has (§3); wrote the MVC-0 spec (§4) with the non-differentiability debt itemized — including the discovery that the largest apparent debt item (differentiating the derived address) is **already discharged by the implicit function theorem, verified to 1e-7**; stated the impossibility results as propositions (§5); and answered the task's invitation on the Hub's two-phase addressing proposal: **I endorse it, with one sharpening it needs to survive review** (§7).

**How I verified:** 5 scripts + 2 diagnostics in `.claude/scratch/clu-controller-spec/` (`common.py`, `s1_consistency.py`, `s2_nondegeneracy.py`, `s3_admission.py`, `s4_deadband.py`, `s4_diag.py`, `s4b_deadband_relax.py`), results JSONs alongside (`s{1,2,3,4,4b}_results.json`). Pure numpy 2.4.1 (main venv), no repo code imported; integrator is line-for-line the shipped damped `velocity_verlet_step` form (Newtonian T, M=1). PREREG: `.claude/outputs/clu-controller-spec/PREREG.md`, written before any run.

**Flag-provenance (all checks).** Self-contained numpy; repo untouched, read-only at `main @ 6dd43bd`, clean tree; zero config/flag dependence on the repo. Landscapes `V = α‖q‖² − Σ A_i exp(−‖q−c_i‖²/2s²)`, s=0.35 throughout. **S1:** K=4 ring R=2, α=0.05, A=1; relax γ=0.05, ε=0.05, N=3000; seeds `default_rng(0)`; FD Hessian δ=1e-5, Newton tol 1e-13. **S2:** K=8 ring R=2, α=0.05, A_i∼U[0.7,1.3] (`rng(1)`); read γ=0.02, N=1500; 64 queries/item, σ_r=0.05. **S3:** box proposals U[−2,2]², α=0.02, A=1, d_safe=4.4s≈1.54, 20 admissions, relocation from 400 candidates U[−2.6,2.6]²; selectivity: 20 queries/item σ=0.1, γ=0.05, N=1500; seed 2 (+1000+k per eval). **S4/S4b:** K=6 ring R=2, α=0.05; 300 epochs, center jitter σ_dr=0.02, drift 0.004/epoch tangential on item 0 (total 1.2), δ_dead=0.3; re-derivation Newton (S4, registered) vs relax γ=0.1 N=800 (S4b, corrected); eval every 10 epochs, σ_q=0.1; seed 3.

---

## 0. Headline verdicts

1. **The controller's whole contract compresses to five conditions (C1–C5), all of which are checkable at write time from physical observables** — no learned component is required to *enforce* any of them. Every condition has both a verified positive instance and a measured violation counterexample (this program's or this task's). The controller is a **certifier + allocator**, not an optimizer.
2. **The deepest structural fact (verified to 11.3 orders of magnitude): the relaxation map is contractive in state space but smooth in parameter space.** ∂(endpoint)/∂q₀ = 2.2e-12 (Prop 5's death of address search, reproduced) while ∂q\*/∂θ = −H⁻¹∂_θ∇V is O(1/λ_min) and matches finite differences to 1e-5…1e-7 (IFT). **Derived addresses are differentiable in θ for free, without unrolling** — the controller lives entirely on the smooth parameter side. This simultaneously (i) confirms the Hub's two-phase addressing structurally, (ii) discharges the largest non-differentiability debt item, and (iii) gives C1 its maintenance law: fixed points drift as −H⁻¹∇δV (verified to 0.017% relative at perturbation 1e-3, error scaling exactly linear, ratio 9.99).
3. **"Non-degenerate enough to be functional" is now a theorem-shaped criterion, not a vibe** (§2 C2): injectivity + margin ≥ κσ + sub-barrier energy + payload separation. Assignment is *exactly* gauge under these conditions — 10 random item→well permutations give retrieval accuracy identical to all digits printed (spread **0.0**) — and the margin condition is *quantitatively* the Gaussian-tail law acc ≈ erf(margin/√2σ) (max deviation 0.021 across σ/spacing ∈ [0.1, 0.35]), which turns the engineer's empirical "break at σ/spacing ≈ 0.2" into a predictable admission criterion (κ ≈ 5 for 99%-grade).
4. **The spacing-gated admission rule keeps the system out of regime 2, with corruption at the first-order bound:** gated — selectivity **1.000 at every K** (13 admitted, 7 refused), max fixed-point drift per write **8.0e-5**, prediction ‖∇δV(q\*)‖/λ_min matching at median ratio **1.0002**, 100% within 2×; ungated — selectivity collapses 1.0 → **0.35** exactly as pairs enter the partial-merger band. w19's "locality is designed" now has its admission-rule form: **locality is a certificate the controller checks, not a hope.**
5. **The deadband is verified free and necessary, and it exposed a new failure mode.** Corrected instrument: accuracy gap deadband-vs-always **0.0 exactly** at **600× fewer updates** (3 vs 1800); never-update fails under real drift (drifting item → 0.15). Address error is *exactly* wake-null until the separatrix (retrieval flat at 1.00 up to 0.8× half-margin, cliff to 0.006 past it) — T2 observed in read space. **New failure mode (registered instrument): Newton-based address re-derivation captured a saddle and the deadband then faithfully preserved the corrupted address** — re-derivation must be by relaxation + λ_min > 0 check (C4.3).
6. **Trash is re-specified as a 4-step policy — demote → overwrite → heat → remove — with friction holes forbidden** (vault theorem 107.77×; N12 composition Pareto-dominated; N13 λ-tuning cannot rescue placement). Deletion decisions get the same deadband treatment in *time* (persistence window + hysteresis), because deletion is the one irreversible verb.
7. **Item 4 lands as a proposition: designed structure is necessary; the controller is also necessary; neither substitutes for the other** (§5). A fully unconstrained V_θ cannot be made a usable memory by any controller in the stated action space — permanence needs exact flatness (measure-zero, 12-order measured gap), and even finite-τ memory fails C3/C5 by default under global-support writes. Conversely T3 + the dead gradient-search make the write-side controller the *only* mechanism left that can perform assignment. **The division of labor is forced, and every leg of it has a measured counterexample when violated.**

---

## 1. Objects and notation

Phase space Γ = ℝ^{2d}; landscape V_θ; per-launch mass m (Prop 6 — per-address, pending the API change); relaxation map **R_γ**: q ↦ lim of the damped (γ>0) Verlet flow from (q, 0); flow map Φ^N_{m,γ}. Memory 𝕄 = (V_θ, 𝒜), codebook 𝒜 = {a_i = (m_i, q₀ᵢ, p₀ᵢ)}, items (S_i, ρ_i, τ_i) as in `clu-memory-architecture` §1. B_i = basin of attraction of item i's structure under R_γ. margin_i = dist(q₀ᵢ, ∂B_i). h_i = lowest saddle out of B_i above V(q\*_i). E(a) = T_m(p₀) + V(q₀). M\*_i = p₀²/2h_i (Prop 2). κσ notation: σ = query-embedding noise scale at the address plane.

**The controller** 𝒞 is a policy over: structural write ops (add/move/ramp atom; register ops; spurion μ² ops), codebook ops (record/update/delete entries), launch scheduling (γ phases, boost ladder), and maintenance (re-derivation, audits). Its inputs are the physical observables listed in §3.0. Nothing in 𝒞 is required to be learned.

---

## 2. Item 1 — the conditions (C1–C5)

### C1 — Consistency (read lands where write wrote). **Status: proven (first-order laws) + verified; the genericity clause is standard Morse theory.**

**Write rule (derived addresses):** at write time, a_i := (m_i, R_γ(φ(x_i)), p₀ᵢ) — the writer records where the write relaxed to. **Consistency condition:** for any query x intended for item i, R_γ(φ(x)) ∈ B_i under the *current* V_θ. Decomposition into the four ways it can fail, each with its guard:

1. **The landscape moved since write.** Fixed points are implicit functions of θ: to first order **δq\*_i = −H_i⁻¹ ∇(δV)(q\*_i)** (H_i the Hessian at q\*_i). *Verified:* under a smooth global perturbation a·sin(k·q+φ), relative mismatch of this law is **1.7e-4 at a=1e-3, 1.7e-3 at a=1e-2, 8.6e-3 at a=5e-2** — error exactly first-order (ratio 9.99 per decade). ⇒ **Maintenance law:** between two maintenance passes the cumulative drift must satisfy ‖H_i⁻¹∇δV(q\*_i)‖ < margin_i − κσ, else re-derive. Since the controller re-derives (C4), the binding form is per-pass: drift-per-epoch ≪ margin.
2. **Query noise.** Covered by C2's margin condition (below).
3. **R_γ ill-defined.** Requires V_θ **coercive** (the α‖q‖² term; ⚠ Deep/Conv potentials omit it — §7.7 defect, architecture fix) and **Morse in the operating region** so that a.e. query converges to a hyperbolic minimum. Note the asymmetry with D3: exact *symmetry* is measure-zero (that is why latches must be designed), but *Morse-ness is open-dense* — consistency does not fight D3; it is the cheap genericity.
4. **Saddle capture in re-derivation (new, measured this task).** Any root-finding re-derivation (Newton on ∇V=0) can converge to a saddle; in S4 the registered instrument did exactly this and the codebook then held a saddle (queries split ~50/50 between the flanking wells; the deadband faithfully preserved the corruption for 150 epochs). **Guard: re-derive by R_γ only (its stable set excludes saddles up to measure zero) and check λ_min(H(cand)) > 0 before commit.** With this guard the identical protocol gives gap 0.0 (S4b).

**The dual smoothness fact (load-bearing for the whole architecture):** the same contraction that makes retrieval robust — ∂R_γ/∂q₀ measured **2.2e-12** after 3000 damped steps — coexists with parameter-side smoothness ∂q\*/∂θ = −H⁻¹∂_θ∇V, verified against finite differences to **1.0e-5 (own atom amplitude), 1.0e-7 (confinement α), 3.1e-7 (a mid-range atom at write distance 1.2)**. Contrast: **11.3 orders of magnitude.** Consistency is therefore maintainable *by construction* (addresses track θ smoothly) even though it is unlearnable *by search* (q₀-gradients are dead). This is C1's formal content and the two-phase proposal's justification (§7).

### C2 — Non-degeneracy vs correctness. **Status: proven (each condition necessary — measured counterexample when violated) + verified (sufficiency at toy scale); the gauge statement verified exactly.**

**Definition.** (V_θ, 𝒜, π) with assignment π: items → structures is **functionally non-degenerate at (σ, δ_read)** iff:

- **(N1) Injectivity.** π injective on live items; or deliberate sharing of one *register* with coset separations ≥ max(4σ_write, ℓ_θ).
- **(N2) Margin.** margin_i ≥ κσ, with the *quantitative law* acc_i ≈ erf(margin_i/(√2 σ)); κ = 5 gives ≈ 99%. *Verified:* measured accuracy tracks the Gaussian-tail law within **0.021** across σ/spacing ∈ {0.10…0.35} (e.g. 0.984 vs 0.988 at 0.20; 0.953 vs 0.954 at 0.25; 0.848 vs 0.847 at 0.35). The engineer's "break at σ/spacing ≈ 0.2" is this law's κ≈5 point — **an admission criterion, not an empirical accident.**
- **(N3) Confinement.** E(a_i) < h_i, equivalently m_i > M\*_i = p₀ᵢ²/2h_i. *Verified in read form:* sub-barrier (E = 0.5h) accuracy **0.969**; supra-barrier (E = 1.5h) collapses to **0.268**.
- **(N4) Decodability.** Payload separation ≥ 2δ_read under the committed read functional (time-averaged linear; P4b lesson). The engineer's blank-landscape 0.469 is the measured violation instance.

**Gauge theorem (T2/T3 in read space), verified exactly:** among assignments satisfying N1–N4, retrieval accuracy is invariant under arbitrary permutation of which item sits where — 10 random permutations on heterogeneous wells (A_i ∈ [0.7,1.3]): accuracy 0.9921875 *identically*, spread **0.0**. Violating N1 (two items → one well): the merged pair decodes at **0.508 ≈ chance** while all others hold at 0.982. **This is the precise sense in which the Head's "allow arbitrary choices" is licensed: an assignment needs a margin certificate, never a semantic justification.** "Correct" appears nowhere in N1–N4.

### C3 — Stability under rewrite. **Status: bound proven + verified at the 1.0002-ratio level; the MLP-failure half is program-measured (CM-6/N46 lineage), not re-tested here.**

**Condition.** A write for item K+1 with landscape change δV is **admissible w.r.t. stored items** iff for every stored i: (i) ‖H_i⁻¹∇δV(q\*_i)‖ ≤ δ_budget (< δ_dead so maintenance absorbs it silently); (ii) |δh_i| < h_i − E(a_i) − safety (N3 preserved); (iii) payload perturbation < δ_read.

**Designed realization (atom writes + spacing gate d ≥ d_safe = 4.4s):** ‖∇δV(q\*_i)‖ = A(d/s²)e^{−d²/2s²} decays super-exponentially. *Verified (S3, gated):* max drift over 13 sequential admissions **8.0e-5**; the first-order prediction matches at **median ratio 1.0002, 100% of writes within 2×**. This is the admission-rule form of w19's measured 4.17e-7: **locality is certified, not hoped for.** Ungated: max drift **8.39** (wells merged), and beyond first-order the bound degrades honestly (68% within 2×) — the bound is a *small-perturbation* law, which is exactly what the gate enforces.

**Does it survive a learned landscape?** Not by default. An MLP V_θ has global support: a free-form θ-gradient write moves every stored item (CM-6 erosion is the measured instance; the fit-gap N13 finding — register damage came through V-perturbation — is another). It survives **iff the write operator is structured**: (a) dictionary/atom writes (what S3 verified — exactly the `AtomDictionaryPotential` of §4), (b) anchors R-2/R-3 on stored items during writes (measured envelope λ=100), or (c) NTK-orthogonalized write currents (unproven, expensive). Formally the condition on a learned write is **|⟨∂_θ V(q\*_i), δθ⟩| ≤ λ_min,i·δ_budget for all stored i** — the controller can *check* this (it is a dot product) even when it cannot guarantee it.

### C4 — Anti-thrashing (the deadband, specified). **Status: verified (zero cost, 600× suppression, necessity of updates under drift); the width condition is derived + consistent with all runs.**

**What the deadband is on** (three distinct deadbands):
1. **Codebook coordinates:** commit a re-derived address only if ‖cand − q₀ᵢ‖ > δ_dead. **Width: 10×(re-derivation noise floor) ≤ δ_dead ≤ ½(margin_i − κσ).** Lower bound = don't chase noise (S4b: σ_dr = 0.02 jitter produced **zero** false commits at δ_dead = 0.3); upper bound = stay inside the wake-null plateau. *The plateau is measured and it is exactly flat:* offsetting every stored address tangentially, retrieval = **1.000** at 0.1–0.5× half-margin, 0.997 at 0.8×, then a cliff (0.503 at 1.0×, 0.006 at 1.2×). Address error below the margin is *exactly free* — the T2 wake-null observed in read space — so pinning costs nothing, and only genuine drift (accumulated > δ_dead) triggers an update.
2. **Structural moves** (relocate/merge/split/mass-band changes): hinge on predicted loss gain > ε_hinge, with hysteresis (undo bar > do bar).
3. **Deletion (time-deadband):** see §3-C — persistence window W + hysteresis, because deletion is irreversible.

*Verified end-to-end (S4b, corrected instrument):* deadband vs always-commit accuracy gap **0.0 at every checkpoint**, **3 vs 1800 commits (600×)**; never-update fails under real drift (drifting item's retrieval → **0.15** once cumulative drift 1.2 > margin). ⚠ Registered-instrument failure recorded honestly: with *Newton* re-derivation the gap was 0.067 — the saddle-capture artifact of C1.4, not a deadband defect; the deadband preserved whatever it was given, which is why **C4 must include C1.4's minimum-check on every candidate before the deadband ever sees it.**

### C5 — Capacity admissibility (stay out of regime 2). **Status: verified at toy scale; the regime taxonomy is w19-measured; the no-incremental-densification clause is a derived rule (conjectured at CLU scale, flagged OQ-D).**

Obligations, in decision order:
- **(A1) Spacing gate:** admit a new discrete site iff min_j d(q_new, q_j) ≥ d_safe (toy: 4.4s; general form: the induced pairwise saddle must satisfy h_ij ≥ max E_read + ΔE_safety). *Verified:* gated selectivity **1.000 at every K** reached; ungated collapses to **0.35–0.40** with degradation onset exactly when pairs enter the partial-merger band (min spacing 0.15 ≪ 2s = 0.7 at the end).
- **(A2) Refusal is admissible:** 7/20 proposals refused with zero accuracy cost — refuse-and-relocate (or refuse-and-report) is a *correct* controller output, not a failure.
- **(A3) Never densify through regime 2:** if a region's required item density exceeds the regime-1 packing bound, convert the cluster to a **designed regime-3 register in one structural operation** (capacity then resolution-limited: range/(4·max(σ_write, ℓ_θ)), tens of bits). Incremental density growth passes through the washboard death zone (selectivity 0.49 at the w19 K=16 point) — the curriculum hazard OQ-D. The controller makes the crossing *discontinuous* on purpose.
- **(A4) Prefer fresh territory:** the packing bound (1+2R/w)^d is exponential in d — count is never the binding constraint; local density is. (The d-scaling task will price this properly; do not quote the ring ceiling.)

---

## 3. Item 2 — the three decisions, mechanised

### 3.0 The controller's observables (all computable, none learned)
For a candidate item x with write embedding q_w = φ(x): relaxation endpoint **r = R_γ(q_w)** and residual ‖∇V(r)‖ (convergence check + λ_min(H(r)) > 0 saddle check); **energy drop** V(q_w) − V(r); nearest stored entry **i\* = argmin_i ‖r − q₀ᵢ‖** and d\*; **basin membership** (r within d_capture of q\*_{i\*}); **barrier estimates** h (cheap proxy: max V on the segment q\*_i → q\*_j minus V(q\*_i); the S2 arc-max is this in ring form); **payload correlation** c(x, i\*) (similarity in payload space — content, not geometry); **utility** u_i (read counts; loss-attribution by masking item i); spacing statistics and σ estimate from recent queries.

⚠ One honest circularity, named: *basin membership is correlation under the current landscape's metric* — the relaxation IS the content-addressable correlation detector (Hopfield-exact), but a query can land in a foreign basin by geometric accident precisely because assignment is gauge (C2). **The payload check c(x, i\*) exists to break that circularity.** Both must pass to call two items "correlated."

### 3.A Write near an existing item (correlated, allowed to decay → mode-b)
- **Trigger:** r ∈ B_{i\*} **and** c(x, i\*) ≥ c_min **and** not flagged permanent.
- **Action:** write on i\*'s **register** (same structure, offset along the coset orbit), *not* a satellite well — a satellite inside d_safe would violate C5(A1) by construction, which is why "near" in the vision must mean *same structure, different coset coordinate*. Offset Δ ≥ max(4σ_write, ℓ_θ). Set the half-life by the spurion: μ² per GMOR (μ²F² = δΣ), retention n_½ ∝ 1/μ² (T=0) or place γ on the V-curve (min at γ_crit = 2εμ) if fast decay is wanted. Record a_new = (m_{i\*}, register slot, p₀).
- **Failure modes:** (i) **spurion leakage** — a global spurion tilts the parent latch; require block separation (CM-9) or ⟨∇V_spurion, ∂_orbit V_a⟩ = 0; (ii) **offset below resolution** → merges with parent (N1 violation; the merged-pair 0.508 is what that looks like); (iii) **register full** — slot count = range/(4·max(σ_write, ℓ_θ)); overflow reroutes to Decision B.
- If i\* has no register and the item still qualifies as correlated-decaying: attach a designed register block to i\* (a structural op, priced once), or accept storage as a *trajectory-stretch payload* within B_{i\*} read by a finer time-resolved functional — flagged **conjectured**, untested.

### 3.B Open a new basin (essential, uncorrelated → mode-c or mode-a)
- **Trigger:** d\* > d_capture (landed unclaimed) **or** (r ∈ B_{i\*} but c(x, i\*) < c_min and the item is flagged essential — the "geometric accident" case; relocate).
- **Action:** location = r if the spacing gate passes, else the nearest admissible location (arbitrary — C2 says this is exactly free; deadband pins it thereafter). Structural op: **add atom(s)** (mode-c: massive, finite τ chosen on the V-curve, band per required timescale) or **add a designed register/SSB block** (mode-a: τ=∞ — designed-only, N46/CM-16a; there is no emergent path, §5). Assign mass: m ≥ 2M\* = p₀²/h for isolation with safety margin, in the band matching the item's timescale; register (m, q₀, p₀); renormalize the log-mass common mode (softplus trap, N64-candidate).
- **Failure modes:** (i) **no admissible space** → capacity alarm: either extend territory (raise R / use fresh dimensions, A4) or trigger the A3 regime-3 conversion — never squeeze; (ii) **m < M\*** → the item is readable but not isolated (reads foreign items — supra-barrier 0.268); (iii) **permanent requested without a designed block** → must be refused: no soft mechanism can grant it (Prop C-N, §5); (iv) mass-band collapse under training — *not controller-fixable*: requires per-address masses (Prop 6 API change) and the E_reg common-mode fix; the controller can only renormalize and alarm.

### 3.C Trash (the least developed — specified with the most conservatism)
Mechanism ≠ policy (task's warning). N12 (governor+field compose Pareto-dominated), N13 (λ-tuning cannot rescue placement), and the v5-gate vault theorem (a friction hole is a **107.77×** *protector*, erasing nothing at T=0) jointly rule out the vision's original "activate the trash region" as an *eraser*. Policy:

- **Trigger (time-deadband + hysteresis):** u_i < u_min for **W consecutive epochs** and not permanent-flagged; re-admission bar u_re > u_min (hysteresis kills delete/reopen thrash). W is the C4 deadband transplanted to the one irreversible verb.
- **Action ladder (ranked, stop at the first that applies):**
  1. **DEMOTE (default, reversible):** raise μ² via spurion — the item becomes mode-(b) with n_½ ∝ 1/μ² and dies on its own; during the grace period a returning utility signal re-flattens it. This converts deletion from an event into a *decay-rate decision*, which is the physically native form.
  2. **OVERWRITE (registers):** equivariant write current moves the register value to null — deletion as a write; zero thermodynamics; exactly local for block-separated registers.
  3. **HEAT (latched content needing fast death):** local T_φ hot spot, rate ∝ T (t-lever §8 spec; unbuilt). The only fast eraser for a latch.
  4. **REMOVE (point items):** ramp the atom amplitude A_i → 0 over epochs (a discontinuous removal changes the landscape under in-flight rollouts), applying C3 to the *negative* write — removal changes neighbors' barriers too (δh check).
  - **NEVER:** γ_φ friction hole as an eraser. Where γ_φ survives: implementing γ → γ_crit = 2εμ *locally* for massive content (V-curve minimum = fastest eraser for mode-b), and as a *vault* when protection is wanted.
- **Failure modes:** (i) **mis-attribution under redundancy** — two redundant items each look useless when masked alone (Shapley problem); demote-first + grace makes this recoverable rather than fatal; (ii) **dependency violation** — trashing a register block that carries mode-(b) satellites; the codebook must keep a dependency count and refuse until dependents die; (iii) thrash — cured by W + hysteresis by construction.

---

## 4. Item 3 — MVC-0, the minimum viable controller (engineer-implementable spec)

Hand-coded, dumb, and sufficient to make every verb exist. **Head's scope ruling honored: nothing below is designed for differentiability; the debt is itemized separately.**

**State.**
- `AtomDictionaryPotential`: α‖q‖² + Σ_i (−A_i exp(−‖q−c_i‖²/2s_i²)) + optional designed register blocks (SO(2) block-untied per CM-9, spurion δ_i per block). Extends `chlu/core/memory_potentials.py`'s pattern; writes are atom/block ops — C3-local by construction (verified: drift 8.0e-5, prediction ratio 1.0002).
- **Codebook** (plain table): id → {mode a/b/c, m, q₀, p₀, μ², write-embedding x̂ (kept for re-derivation), payload decoder row, u-stats, dependency count, timestamps}.
- Globals: d_safe, κ=5, running σ estimate, δ_dead, W, u_min < u_re, mass bands, s_min (read-score bar), R (retry budget).

**WRITE(x, payload, flags):** q_w = φ(x) → r = R_γ(q_w) (γ=0.05–0.1, N_relax with residual check + λ_min(H(r)) > 0) → decision tree §3 (A/B/C-adjacent) → structural op → record derived address. *The writer knows where it wrote; nothing is searched.*

**READ(x):** r = R_γ(φ(x)) **forward-only** → nearest codebook entry within d_capture → launch (m_i, q₀ᵢ, p₀ᵢ) at **γ=0**, N steps → **time-averaged linear read** (tail 25%; never pointwise — P4b) → payload. If score < s_min: **boost-retry ladder** k = 1…R with E_k set to quantiles of the stored barrier heights (energy-shell filtration semantics, w19 §8) — the *sole* cross-basin mechanism; gradient capture is measured-dead and must not be reintroduced.

**MAINT (per epoch):** for each i: cand = R_γ(x̂_i); **verify λ_min(H(cand)) > 0** (the S4 saddle lesson); commit iff ‖cand − q₀ᵢ‖ > δ_dead. Renormalize log-mass common mode to zero mean. Audit: pairwise saddle margins (C5), margins vs κσ (C2), N3 confinement, dependency counts; alarm on violation. *Cost: K relaxations + K Hessian λ_min's per epoch — trivial next to training.*

**Explicitly out of scope for MVC-0 (recorded, not hidden):** learned triage (permanence/essential flags are supplied by hand or task heuristic); learned φ/ψ; the amortized selector (train later by regression/distillation on write-time (content, address) pairs — no rollout backprop); wormholes/multi-particle consolidation.

**Defaults from measurement:** γ_relax 0.05–0.1 · γ_read 0 (payload) / 0.02–0.05 (capture) · d_safe = 4.4s · κ = 5 (σ/spacing ≤ 0.2) · δ_dead ∈ [10× re-derivation noise, ½(margin − κσ)] · barrier scale 0.2-class · m ≥ 2M\* · read = tail-25% time average.

### The non-differentiability debt, itemized

| # | non-diff. piece | where | eventual differentiable relaxation | debt class |
|---|---|---|---|---|
| 1 | nearest-codebook lookup | READ | softmax over −β·d(r, q₀ᵢ) — the Hopfield/attention form; also the distillation target for the amortized selector | inference-path, standard |
| 2 | A/B/C decision tree thresholds | WRITE | sigmoid gates / straight-through. ⚠ DNC precedent: soft allocation gates are trainable but unstable; 2 of 3 diagnosed DNC failure modes bite our scheme (scout) — record, don't promise | training-time only |
| 3 | atom add/delete | WRITE/TRASH | fixed max-pool of atoms, amplitude gates A_i ∈ [0, A_max], deletion = ramp to 0, hinge sparsity | training-time only |
| 4 | deadband / hysteresis / W | MAINT/TRASH | **none needed — zero debt by design.** T2: the parameter is wake-null inside the band; there is no gradient information to preserve (verified: plateau exactly flat) | — |
| 5 | boost-retry argmax | READ | test-time search (transformers do test-time search too); a retry-policy head can later train by distillation/REINFORCE off the selection scores | inference-path, deferrable |
| 6 | masking-based utility | TRASH | sampled Shapley or a learned utility head | training-time only |
| 7 | ~~derived address through relaxation~~ | WRITE/MAINT | **already discharged: ∂q\*/∂θ = −H⁻¹∂_θ∇V (IFT), verified to 1e-5…1e-7; one linear solve, no unrolling.** Prop 5 makes unrolled backprop pointless anyway; use IFT or stop-grad | **paid** |

Items 2, 3, 6 never sit on the inference gradient path; only 1 and 5 do, and both have standard relaxations. **The debt is smaller than it looks, and the largest apparent item was payable in closed form.**

---

## 5. Item 4 — what the controller CANNOT fix

**Proposition C-N (designed structure is necessary).** *No controller whose actions are (i) launching particles, (ii) moving θ of a fully unconstrained smooth V_θ via gradient-based training signals, and (iii) codebook bookkeeping can produce a memory containing any τ=∞ item.*
*Argument.* τ=∞ ⇔ μ² = 0 exactly on a continuous coset (Cor-13/CM-16a, operator identity, verified ≤1.7e-14). Exact flatness is measure-zero in θ-space (D3); generic gradient noise destroys it; no finite-λ soft regularizer produces it (R-4: μ²(λ) > 0 ∀λ < ∞; measured designed-vs-emergent gap **12 orders**, N46 9/9 cells). Action (iii) does not touch θ; action (i) does not touch θ; action (ii) reaches only generic (μ² > 0) configurations. ∎
**Label: proven modulo the action-space classification** (the classification is definitional — a controller that *installs a hard-constrained parameterization* is precisely what "designed-in structure" means, so the escape route is the conclusion). **This is the program-level result the task asked for: permanent memory is a constraint, not a training outcome.**

**Corollary C-N′ (finite-τ does not escape either).** An unconstrained V_θ *can* generically host regime-1 basins (Morse-ness is open-dense — C1.3), so finite-τ memory is not blocked by C-N. But it is measured to fail C5 (emergent capacity 1–1.6 bits, stuck in regime 2 — v5-gate) and fails C3 by default (global-support writes; CM-6). So even sub-permanent memory needs structure or anchors; the controller's decision rules *presuppose* a C2/C3/C5-satisfiable substrate — they certify structure, they cannot create it.

**Proposition C-D (the division of labor is forced — the positive complement).** Assignment cannot come from any permutation-symmetric regularizer (T3, proven), cannot come from gradient search across basins (measured dead: 4 protocols + engineer's 2, best ≈ chance), and cannot come from the physics (assignment is gauge — C2, spread 0.0). **The write-side controller is therefore the only mechanism remaining that can perform assignment — it is necessary, not a convenience.** Structure: designed (C-N). Assignment: controller. Retention: physics knobs (μ², γ). Restructuring within the scaffold: the loss (Toy D basin-level 9/18; Toy E ratios to 2.2e-14; IFT address-tracking here). Each leg has a measured counterexample when its assignee is swapped.

**The itemized cannot-fix list** (each with its non-controller fix): shared-global-mass hierarchy (E_reg common-mode + EBM mass-blindness → **per-address mass API**, Prop 6) · dead cross-basin address gradients (**route around**: derived addresses + boost-retry; do not re-fund) · regime-3 write-precision floor K_max ≈ range/(4·max(σ_write, ℓ_θ)) (physics bound) · non-coercive Deep/Conv potentials (architecture) · γ=0 reads don't settle (engineer's 0.813; γ-phase scheduling is the fix and is already in the spec).

---

## 6. PREREG scorecard (registered file: `outputs/clu-controller-spec/PREREG.md`, written first)

| pred | registered | outcome |
|---|---|---|
| P1 landing acc ≥ 0.99 | | ✅ 1.000; endpoint err 2.2e-16 |
| P2 drift law ≤20% @1e-2, ~linear scaling | | ✅ 0.17% @1e-2; ratio 9.99/decade (exactly first-order) |
| P3 IFT vs FD ≤1% | neighbor-atom probe | ❌ **as registered: instrument had no signal** (neighbor derivative 2.7e-14 — itself verifying exponential write locality). Amended probes (own amp / α / mid-range atom d=1.2): rel err **1.0e-5 / 1.0e-7 / 3.1e-7** ✅. P4b-class lesson: registered instrument wrong, derivation right |
| P4 contraction ≤1e-6; contrast ≥5 orders | | ✅ 2.2e-12; contrast **11.3 orders** |
| P5 permutation gauge spread ≤1% | | ✅ spread **0.0** (identical to all digits) |
| P6 merged pair ≤0.6, others ≥0.98 | | ✅ 0.508 / 0.982 |
| P7 Gaussian-tail law ±0.03 | | ✅ max dev 0.021 |
| P8 supra ≤0.6, sub ≥0.95 | | ✅ 0.268 / 0.969 (h=0.822) |
| P9 sub-margin address error free; cliff at separatrix | | ✅ 1.000 flat to 0.8×; 0.503 at 1.0×; 0.006 at 1.2× |
| P10 corruption ≤1e-3 gated; pred within 2× | | ✅ 8.0e-5; median ratio 1.0002, 100% within 2× (ungated: 68% within 2× — first-order law honestly breaking at large drift) |
| P11 gated ≥0.95 ∀K; ungated <0.8 past merger | | ✅ 1.000 ∀K (7 refusals, free); ungated → 0.35 |
| P12 deadband gap ≤0.02 | Newton re-derivation | ❌ **as registered: 0.067** — diagnosed (s4_diag): **Newton wrote a saddle into the codebook** (lag 0.6 ≈ saddle distance; errors 100% (0→1); queries from a saddle split). Corrected instrument (relaxation re-derivation = what the controller actually specifies): gap **0.0** ✅. The failure is promoted to spec condition C4.3/C1.4 |
| P13 updates ≤10, ratio ≥100× | | ✅ 3 updates; **600×** |
| P14 never-update drifter <0.5; others ≥0.95 | | ✅ 0.15; others 1.00 |

**Verdict labels.** *Proven:* the first-order drift/IFT laws (C1), the gauge statement and necessity of N1–N4 (C2), the C3 bound, C-N/C-D (modulo the stated classification), T2-basis of the deadband. *Verified-numerically (toy scale, 2-D):* all of the above plus the admission rule, the deadband's zero cost, the saddle-capture failure mode, the erf margin law. *Conjectured:* sufficiency of C1–C5 at CLU dimension with a learned V_θ (the composition is untested — same status as w19's derived-address gate); the A3 one-shot regime conversion; the trajectory-stretch variant of Decision A; the c_min payload-correlation threshold's placement.

---

## 7. On the Hub's two-phase addressing (the task's standing invitation to disagree)

**I endorse it — it is structurally correct, and this task's numbers are independent evidence for it** (obtained on different landscapes/seeds than w19). One sharpening it *needs*, which `relaxation-addressing-theory` should reach independently (convergence check): the worry "does the θ-dependence of the derived address re-import Prop 5 through the back door" conflates two different derivatives. ∂(endpoint)/∂q₀ → 0 (contraction; 2.2e-12) is Prop 5 and is *desirable* (robustness). ∂q\*/∂θ = −H⁻¹∂_θ∇V is the implicit-function derivative of the relaxation's *image*, is O(1/λ_min), well-conditioned inside any basin, and verified to 1e-7. **The dissipative phase is not a gradient barrier in θ; it is a projection whose target moves smoothly with θ.** Implementation rule: compute address-gradients by IFT (one linear solve) or stop-grad them; never backprop through the unrolled relaxation (wasteful, and its q₀-components are (1−γ)^N-suppressed noise). One warning transplanted from this task's own instrument failure: **any implementation of "derive the address" must use the relaxation, not a critical-point solver — Newton-type derivation can and did return a saddle.**

---

## Git footprint
None — no tracked code touched (repo read-only at `main @ 6dd43bd`, clean). All artifacts under `.claude/scratch/clu-controller-spec/` and `.claude/outputs/clu-controller-spec/`.

## Open questions / follow-ups / risks
- **OQ-1 (the composition gap, same as w19's):** C1–C5 are individually verified in 2-D designed landscapes; their joint sufficiency on a learned V_θ at CLU dimension is the untested claim. The named falsifier remains the **derived-address retrieval gate** (w19 §8) — MVC-0 is exactly the vehicle to run it.
- **OQ-2:** the c_min payload-correlation threshold (Decision A vs B) has no measured value; it needs a small calibration study once payload space is fixed.
- **OQ-3:** barrier estimation at d ≫ 2 (segment-max proxy vs true min-energy path) — the proxy overestimates h; overestimation is *conservative* for N3 but *anti-conservative* for the boost ladder's E_k quantiles. Cheap check worth one script when d-scaling runs.
- **OQ-4:** A3's one-shot regime conversion (cluster → designed register) is specified but never executed anywhere; it is the controller op with the largest blast radius. Should be exercised in isolation before MVC-0 relies on it.
- **Risk:** all checks Newtonian, M=1, p₀=0 except where stated; the relativistic kinetic couples modes at finite p (F5 Prop-2) and could shift M\* and the erf-law constants. The *structure* of C1–C5 is kinetic-agnostic; the constants are not.

## Proposed handover updates (for the Hub)
- **§1 (memory formalism addendum):** the controller contract C1–C5, and the load-bearing dual: **contraction in state space (∂/∂q₀ ≈ 1e-12) + smoothness in parameter space (∂q\*/∂θ = −H⁻¹∂_θ∇V, verified 1e-7) — derived addresses are differentiable in θ by IFT without unrolling** (discharges the main differentiability worry of the two-phase scheme; independent convergence expected from `relaxation-addressing-theory`).
- **§7 (new, small but general):** ⚠ **address re-derivation by critical-point solvers can return saddles** and a deadband will then faithfully preserve the corrupted address (measured: 50/50 query splitting, 150 epochs). Rule: re-derive by γ>0 relaxation + λ_min(H) > 0 check. Applies to every derived-address implementation, engineer builds included.
- **§8:** (i) the margin law is quantitative — acc ≈ erf(margin/√2σ), κ=5 ⇒ the "0.2 break" is an admission criterion (retire "empirical ceiling" phrasing); (ii) record **Prop C-N** ("permanent memory is a constraint, not a training outcome") and **Prop C-D** (forced division of labor: design→structure, controller→assignment, physics→retention, loss→restructuring) as program-level statements; (iii) trash is policy-complete: demote→overwrite→heat→remove, friction forbidden, time-deadband W + hysteresis; (iv) MVC-0 is specified and implementable now — its only API prerequisites are the per-launch mass override (w19 spec) and `AtomDictionaryPotential`.
- **Claims-matrix candidates:** the C2 gauge measurement (spread exactly 0.0) is the cleanest T2/T3 empirical instance the program owns; the saddle-capture failure mode is an N-candidate (tier B, instrument/design class).
