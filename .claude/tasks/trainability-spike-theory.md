# trainability-spike-theory — implicit/DEQ gradients at a settled point: existence, conditioning, and the flat-direction tension

**Campaign 2, wave C2W1. Agent:** physics-theorist. **No worktree, no production-code edits**
(numerical checks in `.claude/scratch/trainability-spike-theory/`). Charter §6.4, theorist half.
**⭐ This half GATES the engineer half** (`trainability-spike`) — it can and should start immediately,
in parallel with `rival-recon` and `controller-doctrine`, at zero worktree cost.

Read first: `.claude/AGENT_PROTOCOL.md`, `.claude/advisor-head-c2-charter.md` (**§2.4 placeholder
policy — implicit/DEQ gradients at settled points are Head-approved machinery; §4 pillar 1**),
`.claude/advisor-head-intervention.md` (§3.2 trajectory-as-latent, §3.4 the V2 physics),
and **your own** `.claude/outputs/clu-controller-spec.md` §C1 (you have already verified the core
identity — you are extending your own result, not restarting).

## ⭐ DIAL DECLARATION (protocol §7, C2 form)
- **Dial (C2 form):** none — theory. Enables the **trainability** of the whole C2 programme; makes no
  performance claim and measures no benchmark.
- **Laundering control:** n/a. ⚠ The engineer half's gradcheck against **truncated unroll** is the
  substitute-control analogue and you specify its tolerance (§Deliverable 4).
- **Falsifies:** the implicit-function hypotheses **fail generically** on the operating set — i.e.
  `λ_min` of the relevant Jacobian is **not** bounded away from zero over the configurations the
  harness actually visits — so implicit gradients are not usable without a modification you must then
  name. **Pre-register your expected answer before deriving it.**
- **Does NOT falsify:** needing a **ridge/damping term** (`(H + λI)⁻¹`) — that is standard DEQ practice,
  not a defect · ill-conditioning at *deliberately* degenerate points (that is §Q2, the interesting
  case, not a failure) · a conditioning bound that is loose but correct.

## Q1 — Existence: the fixed point of the DISSIPATIVE VERLET MAP, not of a gradient flow
The settle is a fixed point of the **shipped discrete damped velocity-Verlet map** `T_θ: (q,p) ↦ (q',p')`,
not of `q̇ = −∇V`. State and prove the implicit-function conditions **for the map we actually ship**:
1. Characterise the fixed-point set of `T_θ`. Confirm (or refute) that `T_θ(z*) = z*` reduces to
   `p* = 0, ∇V_θ(q*) = 0` at the shipped γ>0, `dt` as shipped, and say what happens at γ=0
   (where there is no settle at all — this is why the two-phase read exists).
2. The invertibility condition is on **`I − ∂T_θ/∂z`** at `z*`, **not** on `Hess V` directly. Give the
   relation between the two (the damping and `dt` enter the spectrum), and state the condition in terms
   of `Hess V` **with the `(γ, dt, M)` prefactors made explicit** — the engineer needs the version with
   the constants in it.
3. **Smoothness in θ.** Your C1 result stands: `∂q*/∂θ = −H⁻¹ ∂_θ∇V`, verified to 1e-5…1e-7, against
   `∂R_γ/∂q₀ = 2.2e-12` — **11.3 orders of magnitude**, the fact that the whole architecture rests on.
   Restate it as a proposition **for the discrete map** and say whether the constant changes.
4. **Genericity.** Morse-ness is open-dense (your C1 §C1.3). Does that survive a **learned `V_θ` with
   permitted basin interaction** (§8.2 forbids engineering the wells apart)? This is the case C1 never
   covered, because C1 always had `d_safe` separation by design.

## ⭐ Q2 — THE CENTRAL TENSION (the reason this task is worth a theorist)
Charter §4, pillar 1: **flat directions store a manifold** of settled states — "which no lookup table
can express." A flat direction means `λ_min(Hess V) → 0` along it. But the implicit gradient is
`−H⁻¹ ∂_θ∇V`, whose conditioning is `O(1/λ_min)`.
> **So: does the program's highest-novelty pillar structurally break its own trainer?**

Answer it properly. The candidate resolutions to test, at minimum:
- **Quotient the flat direction out.** The gradient is only ill-posed *along* the flat direction, and the
  loss may be **invariant** there (that is the point of a Goldstone/flat mode). If so, the correct object
  is the implicit function on the **normal bundle** / the quotient by the symmetry orbit, and the
  conditioning is set by `λ_min` of the *transverse* Hessian. **Prove or refute that the loss is
  invariant along the designed flat direction.** ⚠ Note N46: the *emergent* arm has no coset register —
  its "flat" direction was a mid-spectrum massive mode. Distinguish designed from emergent throughout.
- **Ridge / Tikhonov.** `(H + λI)⁻¹`: give the bias-vs-conditioning trade-off and a principled λ.
- **Truncated unroll on the flat direction only** (a hybrid).
State which resolution you recommend, and **what it costs**. If the tension is real and unresolvable at
this weight class, say so — that is a Head-level ruling on pillar 1, and it is far cheaper to learn now.

## Q3 — Conditioning on the operating set
Bound (or measure) `λ_min` over the configurations the harness visits: near-capacity stores, interacting
basins, items at the **reach boundary**. ⭐ Cross-check against the reach theory you already own — the
**saddle criterion on `L=√(|c|²+a²)`, verified 31/32 on the trained shipped `V` with zero free
parameters**. An item at the reach boundary is near a saddle, and near a saddle `λ_min → 0` **and changes
sign**. So: **is "reach failure" (collapse mode #11) the same object as "implicit-gradient
ill-conditioning"?** If yes, monitor #11 doubles as the trainer's health check — a genuinely useful
unification, and worth stating as a proposition. (Also relevant: C4.3 — a Newton-based re-derivation
captured a **saddle** and wrote it into a codebook. Re-derivation by relaxation with a `λ_min > 0` check
is already law; say whether the same guard is needed on the gradient path.)

## Q4 — Truncated backprop over trajectory reads
Charter §2.4 also approves **truncated backprop over trajectory reads**. The trajectory read is *not* at
a fixed point, so the implicit theorem does not apply. Give the theory: what does truncation at depth `k`
cost, and what is a defensible `k`? ⭐ You have the contraction constant — `∂R_γ/∂q₀ = 2.2e-12` after
3000 damped steps — so the *gradient* through the unroll dies geometrically. **State the truncation
depth beyond which additional unroll steps are numerically worthless**, with the γ-dependence. This is
directly actionable and the engineer will use the number.

## Q5 — Consolidation (repositioned wake–sleep)
Charter §2.4 repositions wake–sleep as the **consolidation phase** — offline maintenance: re-packing,
decay enforcement, gate re-calibration. Two questions, both cheap: (i) does consolidation need gradients
at all, or is it a controller/allocator pass (your C1 verdict was *certifier + allocator, not
optimizer*)? (ii) N5 is on record — **the sleep phase inverted a designed degenerate vacuum**. Does
consolidation-as-maintenance re-expose N5, and what guard prevents it?

## Numerical sanity checks (only where they settle something)
Pure numpy/jax/sympy in `.claude/scratch/trainability-spike-theory/`. The shipped integrator form should
be reproduced line-for-line as in your C1 spec. **Do not re-verify what C1 already verified** — cite it.
Highest-value checks: the Q2 loss-invariance test on a designed flat direction; the Q3 `λ_min` sweep
across the reach boundary; the Q4 truncation-depth curve.

## File ownership
**You own:** `.claude/outputs/trainability-spike-theory.md` + `.claude/scratch/trainability-spike-theory/`.
⛔ **You edit no tracked code.** Requirements for the engineer go in an **implementation requests**
section, one line each. ⛔ Do not edit `.claude/outputs/clu-controller-spec.md` or
`.claude/outputs/readout-channel-theory.md` (C1 artifacts — supersede by reference, never in place).

## Deliverable
`.claude/outputs/trainability-spike-theory.md`, protocol §5 format, containing: (0) your pre-registered
answer to the falsifier, written before the derivation; (1) Q1 as labelled propositions with the
`(γ, dt, M)` constants explicit; (2) **Q2 with a recommendation and its cost — the headline**;
(3) Q3 incl. the reach ↔ conditioning unification, confirmed or refuted; (4) Q4 with **a number** for the
defensible truncation depth; (5) Q5; (6) **implementation requests** for the engineer, incl. the
gradcheck tolerance the engineer should register against; (7) reconciliation list in the **first 10
lines**.

⛔ **Do-not-quote, carried:** "certified" in any technical sense · the √2 / `d^1.62` exponent ·
"the write operator is the ceiling" · width-lock-as-cause · N46's coset register as anything but
**designed-only**. **Standing:** quote the curve, not the endpoint; a pre-registration that fails is a
finding, not an embarrassment.
</content>
