# Task: rb-bound-trained — F-12: turn "M is the budget allocator" into a conservation law (w14, analyst)

- **Agent:** `results-analyst` · **Output:** `.claude/outputs/rb-bound-trained.md`
- **Read first:** protocol (§5 flag-provenance mandatory) · **`.claude/outputs/v2-symmetry-deepdive.md` §7bis R5 (+ R1/R2) and falsifiable F-12** · `.claude/claims_matrix.md` **CM-16, CM-17, CM-5/CM-10** (banding) · `.claude/outputs/v3-band-selection.md` + `.claude/outputs/seed-sweeps.md` (the banded checkpoints and the `κ_eff` extractor).
- **Repo:** read-only.
- **Scope discipline (BINDING):** **G7 is a LONGS mandate — this does not widen any short.** Feeds the ICLR long / Nature-MI. Do not propose draft edits.

## The claim to test
F5 §5 says *"the inertial mass `M` is the primitive's budget allocator."* That has been a slogan, corroborated only by "banded beats uniform." R5 makes it a **conservation law**:

```
n₁/₂ · θ̇_max²  =  2γ ln2 · m₀c² / ( (2−γ) ε² · δΣ )        — independent of M and r*
```

Retention scales as `F²`; bandwidth `θ̇_max = c√m₀/F` scales as `1/F`. **The decay constant cancels from the product.** `M` moves retention and write-bandwidth in *opposite* directions and the product is fixed by the causal budget `m₀c²` over the forgetting drive `δΣ`. Only raising `c` (or `m₀`), or lowering the breaking, buys both.

**Crucially: this trade-off exists only in relativistic mode.** In Newtonian mode `θ̇` is unbounded, so there is no bandwidth constraint and no bound. The relativistic governor *creates* the conservation law.

The theorist verified it on the analytic damped map (`M ∈ {0.5,1,2,4}`, products `681.7/696.0/702.9/706.4` vs predicted `709.7`; residual **exactly** the deep-overdamped correction `(εμ/γ)²` — deficits `3.95/1.94/0.96/0.48%` vs `(εμ/γ)² = 4.00/2.00/1.00/0.50%`). **It has never been tested on a trained checkpoint.**

## Items
1. **Use V3's mass banding as the `M`-lever.** Vary `M` (hence `F² = M_ch r*²`) across bands at **fixed `δΣ`**, on trained designed-SO(2) checkpoints. Measure `n₁/₂` and `θ̇_max = c/(√M_ch r*)` per band; test that the product is invariant. **Prediction: invariant to `(εμ/γ)²`.** Report the residual against `(εμ/γ)²` per band — matching the *correction term*, not just the leading constant, is what makes this a law rather than a fit.
2. **Register predictions before measuring** (`PREREG.md` in your output dir, as `v5-gate` did — that practice caught a real correction last wave and is now expected). Predict the product and each band's `(εμ/γ)²` residual from `s0`-style geometry first.
3. **The Newtonian null is mandatory and is half the result.** Run the identical sweep in `newtonian_learned`. There `θ̇_max = ∞`, so no bound exists and the product is undefined/divergent. **Show that.** A conservation law that only holds where the theory says it should is worth far more than one that "holds."
4. **Confounds to handle explicitly.**
   - `δΣ` must be held fixed while `M` varies. Use the **linear spurion** (`spurion_delta`, `LinearSpurionPotential`) — the shipped *angular* tilt normalizes the condensate away and cannot resolve `Σ` (CM-15). Measure `Σ = r*(δ)` independently per band, do not assume it.
   - `n₁/₂` needs its `Δ` **and** its `ℓ_θ/Δ` reported. `v5-gate` §3.5 showed the raw exponents are contaminated by CM-16(d)'s boundary-layer bias `1 + 3.099·ℓ_θ/Δ`, **which is present in designed and emergent alike** and would have produced a spurious "effect." Stay in the deep-diffusive regime (`ℓ_θ/Δ < 0.06`) or bias-correct and say so. **Never quote `n₁/₂` without `Δ` and `ℓ_θ/Δ`.**
   - **R5 is derived in the deep-overdamped band `εμ ≪ γ`.** Its underdamped counterpart is *not derived* (theorist's open O6). Stay inside the derived band or flag the excursion.
5. ⚠ **CM-17 rides along.** The relativistic arm's Langevin **does not sample Gibbs** (no σ does). Report `T/(m₀c²)` for every relativistic cell. If any part of your `n₁/₂` instrument depends on an equilibrium temperature, it is **invalid in the relativistic arm** — prefer a `T=0`/deterministic-relaxation or pathwise instrument, or run the retention measurement Newtonian and the `θ̇_max` measurement analytically, and say exactly which you did. **This is the trap in this task; the deep-dive's own R5 table used the analytic map, not a thermal sampler.**
6. If the law holds on trained models, state it in one ML sentence: *the mass spectrum cannot buy retention and write-bandwidth at the same time; only the causal budget can.*

**Acceptance:** pre-registered predictions; product invariance measured across ≥3 bands with the `(εμ/γ)²` residual matched; the Newtonian null exhibited; `δΣ` held fixed with an independently measured `Σ`; every `n₁/₂` carries `Δ` and `ℓ_θ/Δ`; CM-17 handled explicitly. A clean negative is a fine outcome — say so plainly if the product moves.
