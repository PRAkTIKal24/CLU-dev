# Task: v2-revision-5 — the CM-16 split and the CM-17 qualifier (w15, writer, SMALL)

- **Agent:** `paper-writer` · **Output:** `.claude/outputs/v2-revision-5.md`
- **Read first:** `.claude/claims_matrix.md` **v1.9** — specifically the **CM-16a / CM-16b split** and the **rewritten CM-17** · `.claude/outputs/v5-gate.md` §3 (the evidence for the split) · `.claude/outputs/f5-corrigendum-2.md` §"Downstream" · your own `.claude/outputs/v2-revision-4.md`.
- **Draft:** `.claude/papers/v2-short/draft.md` (v0.5, 24 pp, 6 figures). Rebuild the PDF.
- **Scope:** this is a **correctness pass, not a widening.** G7 is a longs mandate; add no new results. Two claims the draft currently makes are now out of scope, and one piece of fine print is now wrong. Fix exactly those.

## Item 1 — CM-16 has been SPLIT; the draft may no longer cite it as one claim
`v5-gate` measured the emergent (MLP) arm and the claim does not survive intact:

- **CM-16a (the latch face) is DESIGNED-ONLY.** On emergent checkpoints the coset direction is a **middle-of-the-spectrum massive mode**, only `1.7–4.9×` softer than the *stiffest* mode in the whole spectrum (`1−|λ_coset| ≈ 1e-3` vs designed `≤1.1e-15` — **~12 orders**). Any written coset value `δ ∈ {0.1, 0.3, 0.5}` rad **relaxes completely** (`|retained| ≤ 2.1e-3`). **An emergent CLU stores nothing on its coset**: capacity ≈ `log₂(2–3 washboard minima) ≈ 1–1.6 bits`, not a continuum.
- **CM-16b (the unification + the `T>0` law) DOES generalize.** The emergent `n₁/₂(γ)` is the predicted **V-curve**, minimum at `0.902 ± 0.003 × γ_crit`, log-slopes `−1.0020` below and `+1.116` above, **3/3 seeds**; the designed coset is the same curve's `μ→0` corner. The `T>0` sign flip replicates (**10/10** conditions) above a measured crossover `T* ≈ 3e-3`.

**Wherever §3.4 / §4 / App J lean on CM-16, scope (a) to the designed arm and keep (b) as the general result.** This is not a retreat — *one damping-optimum curve across eleven orders of magnitude in `μ²`* is a **stronger** statement than a designed-only latch, and it is the paper's best answer to "does any of this survive outside your designed testbed?" **Lead with (b); state (a)'s scope plainly.** Add the negative to the appendix per C-9 (registry entry **N46**).

⚠ **Instrument warning that must travel with any exponent you quote:** raw `d log n₁/₂/d log T` and `d log n₁/₂/d log γ` exponents are **not discriminating** designed-vs-emergent — a matched designed control gives the same shallow slopes, because both carry the `ℓ_θ/Δ` boundary-layer bias. **Never quote `n₁/₂` without `Δ` and `ℓ_θ/Δ`.**

## Item 2 — the `fdt` fine print in App F is now wrong
`v2-revision-4` landed the mandatory `langevin_noise="fdt"` fine print inline (correctly). But **App F states Prop-9's `σ*` fix as a class-level "neutral theorem."** Per **CM-17** it is exact **only in the Newtonian kinetic modes**: the coded O-step is an additive Gaussian kick, so its invariant momentum marginal is Gaussian-smoothed, while the relativistic Gibbs marginal is Maxwell–Jüttner ⇒ **no σ works.**

Add the kinetic-mode qualifier. Two things to get right:
- **The failure is in the SAMPLER, not the thermodynamics.** `π_q ∝ e^{−V_θ/T}` is relativity-insensitive (the momentum integral factorizes), so the unit's equilibrium is fine — the chain just doesn't sample it. **Never write "relativistic CLUs have no equilibrium";** a reviewer will catch it and it is false.
- **V2's units are `newtonian_learned`**, so every §3 result is unaffected. Write this as a **scope clause, not a correction to our own results.**
- The F5 note now carries this as **Prop-9′**. Cite it as you cite the rest of the note (`Anonymous, 2026`, third person).

## Item 3 — carry-overs
- The **`13.9×` vault number is retracted** program-wide (the shipped `FrictionField` is absorb-only; measured `107.77 ± 4.78×`). If any V2 appendix or forward-reference mentions it, fix it.
- MF-1 (theory note is still a placeholder) is a standing Head dependency. Mention once, do not itemize.

**Acceptance:** CM-16 cited as (a)/(b) with (a) scoped to the designed arm and (b) led with; the `ℓ_θ/Δ` instrument warning attached to every quoted exponent; App F's σ* carries the kinetic-mode qualifier with the sampler-vs-thermodynamics distinction stated correctly; N46 in the appendix; no new results; PDF clean. **If any of the three items turns out already correct in the draft, say so and change nothing.**
