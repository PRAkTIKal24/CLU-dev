# c2w11-loss-package — formalize the §A34.9 loss package, scope the kinetics, price A31.2

**Campaign 2, wave 11 (THE COMPOSITIONAL WAVE). Agent:** physics-theorist.
⛔ **NO WORKTREE. NO BRANCH. NO tracked-code edit.** Scratch scripts only, under
`.claude/scratch/c2w11-loss-package/` (pure numpy/scipy/sympy, main venv
`/Users/user/Desktop/CHLU/.venv`, float64, each writing its own JSON).
Writes `.claude/outputs/c2w11-loss-package.md` + artifacts to
`.claude/outputs/c2w11-loss-package/`.
**Budget:** ≈ 1 day. **You run FIRST and you are UNGATED** — the two engineer spokes downstream of you
are gated on your deliverable file.

**Binding documents, read first, in this order:**
1. `.claude/outputs/c2w11/PREREG-C2W11.md` **IN FULL** (this wave's prereg — your work is its §A34.9 row).
2. charter **ADDENDUM 12 (§A33–§A34) IN FULL** — you are formalizing **§A34.9 (a)–(f)** and writing the
   **§A34.5 kinetics scoping note**.
3. charter **ADDENDUM 11 §A31.2** — the **open mechanism question** you are pricing.
4. `.claude/outputs/orgdiv-prereg/PREREG-TierII.md` **§6 and §7** (the design-rule compliance table and
   the derived operating point — your terms must not break either).
5. `.claude/outputs/bprime-theory.md` §9.2 (the effective-`s` modelling question) ·
   `.claude/outputs/orgdiv-cat-test.md` §5.1 (the measured grad-norm table).

## ⭐ DIAL DECLARATION (protocol §7) — echo before your first result
- **Dial:** none — **theory + scoping.** ⛔ No claim cell, no performance number, no verdict of any
  kind. Numerical sanity checks are toys and are labelled as toys.
- **Laundering control:** N/A.
- **Falsifies:** a term you cannot write down as a differentiable object with a stated gradient path,
  or whose designed negative you cannot specify, is reported as **NOT FORMALIZED** — never as done.
- ⛔ Depth is not feature importance (§A23.5 ACTIVE). ⛔ Wells are never named semantically.

---

## DELIVERABLE 1 (the mechanical gate the organizer spoke waits on)

> **`.claude/outputs/c2w11-loss-package/LOSS-PACKAGE-DONE.json`**

Top-level `loss_package_complete` computed **mechanically** as the AND over the six terms. Per term
(a)–(f): `formalized: true|false`, the symbol, the **gradient path** it trains through, the
**designed negative** that must fail it, and a one-line statement. ⛔ Anything you could not close is
`false` with its reason — never omitted, never quietly true.

## THE SIX TERMS (§A34.9) — formalize each one

For **each** term give: (i) the expression, in the program's notation; (ii) **which parameters it
touches and through which channel** (implicit-at-settle / trajectory / neither); (iii) its
**coefficient-zero bit-identity** requirement; (iv) the **designed negative** an engineer can
pytest-assert; (v) its interaction with the `2α` coercivity floor and the `d/s ∈ [2.5, 2.9]` band.

- **(a) the label-free organization term** — trains φ/placement toward semantic metric structure
  **through the write** (§A28.1's *designed* write→φ organization gradient — the physics-as-trainer
  bet). ⚠ **This is the term that most needs your care.** §A28.1 is explicit that the accidental leak
  and a designed trainer are **different objects**: the leak is an unaudited entanglement of placement
  with the **outer** loss (the same coupling that erodes the store, and it is measured at **27 %** of
  layer-0 φ gradient whenever `atom_place_radius > 0`). Your job is the **designed** version: routed
  through the **write objective**, byte-ledgered, auditable, declared. **State exactly what an
  engineer must assert to prove the designed channel is live and the accidental one is not.**
- **(b) the sharing / refresh term** — a re-encountered feature **deepens the existing well**; K9
  gates merges. Must satisfy **I1 (refresh-on-rewrite monotonicity)**: a write into an existing well
  must never REDUCE its depth.
- **(c) the curvature-shape term** — **DEFENDS** within-well soft directions (the w20 guard).
  ⚠ **§A4.2 REFUTED the tilt instantiation on a learned store** (tilt monotonically *reduces* `λ_min`,
  +0.099 → −8.28, two independent implementations, every family; a written site's vacuum residual
  0.140–0.343 vs a 0.167 random baseline; `λ = ε` holds only in the single-atom geometry it was
  specified in). **Do not re-derive the refuted object.** Specify a defender whose claim is
  *measurable* and whose failure mode is *visible*, and state the `2α` floor's consequence
  (`τ_max = Γ/2α`; α is the ceiling and lowering it breaks the write).
- **(d) the set-level compositional read loss** through `k` particles + a **DeepSets** ψ
  (⛔ attention-ψ is quarantined for trajectory input — do not specify one).
- **(e) the calibration loss** — **feature dropout as pseudo-novelty**. Specify the target, the
  proper scoring rule, and why it does not collapse to predicting the base rate.
- **(f) kinetics trained through trajectory / finite-budget reads ONLY** — **friction before mass**
  (T3: settled-point kinetic gradients are **bitwise zero**, 3/3 seeds, in-system as well as in the
  gym). ⭐ This wave does **NOT** build the kinetics head; you specify it so C2W12 can, and so that
  term (c) defends the right structure.

⭐ **Cross-cutting, and it is the package's main hazard:** state the **staging** (staged
store-then-launch co-training, w20) and, for each term, whether it can be live before wells exist.
Banked and load-bearing: **the organizer is untrainable until the wells exist** — at init both the
implicit and trajectory channels sit at **1e-10 – 1e-9** (six to seven orders below the trajectory
reference scale) and go to **O(1)** only after the write. **The write is the precondition for gradient
to exist at all.**

## DELIVERABLE 2 — the §A34.5 kinetics scoping note

Formalize, with toy numerical confirmation where cheap:
- the dissipation is a **drag force `−γ·p`** (ratified). With drag, `v̇ = −(1/m)∇V − γv`, so mass sets
  the **damping ratio and inertia**: **`ζ ∝ √m` at fixed `γ, λ`** — light = underdamped (overshoot,
  longer floor travel), heavy = overdamped (local settle). **Derive `ζ` explicitly and state the
  underdamped/overdamped boundary in the program's shipped constants** (`α = 0.05`, `s ≈ 0.32`,
  `dt = 0.05`, two-phase `(γ,N) = (0.05,400) → (0.02,800)`).
- **mass-blindness holds ONLY for the fully-settled endpoint in a strictly convex well** (Prop F1
  unchanged) — state precisely which reads escape it.
- ⭐ **spectral / per-direction mass** (already shipped as per-address mass, C2W7): with a mass
  **tensor**, `M⁻¹∇V` **tilts the roll direction**, so two particles from the same launch traverse
  different paths to different sub-wells / floor positions. **Give the condition on `M` and the
  landscape under which two launches provably separate**, and the condition under which they
  provably do not.
- **the flat-floor stopping question:** the mechanism REQUIRES any of — within-well flat/soft floors ·
  finite-budget reads · trajectory reads. **State the stopping criterion on a flat floor** (when does
  a finite-budget read stop, and what does the stopping point encode?), and note that sub-wells
  (discrete), flat floors (continuous) and trajectory reads (temporal) are **three implementations of
  the same hierarchical-settle semantics; not exclusive.**
- **the ordering, with its measured basis:** friction first (the ~**14×** stronger channel:
  trajectory/point ratios 2.6–4.9e5 friction vs 1.7–2.9e5 mass), mass second, spectral mass as the
  richer selector. Monitor #1 bounds the band from below (`γ ≤ 0.03` trips on S0), and monitor #1's
  own collapse mode is "overdamping → the last observation" (`corr(q*, q_last) → 0.97`).

## DELIVERABLE 3 — price the open A31.2 mechanism question

> **§A31.2 (Advisor erratum 2, registered):** with **comfortable geometry** (σ_q/spacing = **0.32**
> MNIST, **0.19–0.37** CIFAR — queries are ~3× *closer* to their own key than to a neighbour), **real
> basins**, and correct-basin **≈ 0.50**, the settle still extracts **less** from the cue than **1-NN
> over the same keys**. Consistent with the settle-destroys-launch-information family (N213; C2W7's
> scattering 4.83 → 5.67 distinct wells; C2W5's occupancy 0.406 → 0.297), **unexplained here.**

⛔ **You are asked to PRICE it, not to solve it.** Deliver: the **candidate mechanism list** with, for
each, (i) what it predicts that the others do not, (ii) the **cheapest discriminating measurement**
and its cost in engineer-hours and compute, (iii) whether any existing banked artifact already
discriminates it. ⚠ Do **not** assume the answer; §A31.2 says it is open and theorist-priced.
⭐ **If any candidate is refutable from banked artifacts alone, say so and refute it** — that is the
highest-value outcome available to you and it costs no compute.

---

## Acceptance (mechanical)
1. `LOSS-PACKAGE-DONE.json` exists with the six-term table and a mechanically computed
   `loss_package_complete`.
2. Every term has a **stated gradient path**, a **coefficient-zero bit-identity requirement**, and a
   **designed negative an engineer can pytest-assert**. A term without all three is `false`.
3. The kinetics note derives `ζ` in the program's own constants and states the spectral-mass
   separation condition two-sidedly (when it separates, when it provably does not).
4. A31.2 is priced with a candidate list, discriminating measurements and costs — **not solved by
   assertion**.
5. Reconciliation list in the **first 10 lines**. NOT-RUNs declared as NOT-RUNs, never nulls.
6. Every derivation that is a **1-/2-D single-atom toy** carries that label explicitly (the standing
   `bprime-theory` §9.2 bracket: transfer to a learned multi-atom store is **bracketed, not
   measured**).

## FILE OWNERSHIP (declared)
**You own:** `.claude/outputs/c2w11-loss-package*` and `.claude/scratch/c2w11-loss-package/`.
⛔ **You touch NO tracked code and NO other agent's outputs.** You may **read** anything.
⛔ You do not edit `PREREG-C2W11.md`, `PREREG-TierII.md` or the charter — flag, never edit.
