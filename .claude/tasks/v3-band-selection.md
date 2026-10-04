# Task: v3-band-selection — from "banding works" to "banding is a method" (critique P10/V3.2; frame per CM-5)

- **Agent:** `experiment-engineer` · **Output:** `.claude/outputs/v3-band-selection.md`
- **Read first:** protocol · `.claude/critique_register.md` (V3.2: "we told the model the answer") · `.claude/claims_matrix.md` **CM-5 (the binding frame: ordering inducible, magnitude designed)** · `.claude/outputs/mass-lr-doctrine-test.md` (esp. follow-ups 1+3) · `seed-sweeps.md` item 1.
- **Git:** branch `agent/experiment-engineer/v3-band-selection` — **worktree MANDATORY** (`v1-router-baseline` runs concurrently, both touch `chlu/`).

## Items
1. **Mis-banded control (the confound-killer):** train lattices with bands ANTI-matched and orthogonally-matched to the data timescales (same spread as the matched-banded arm), 5 seeds × both budgets from seed-sweeps. Deliverable: the degradation curve matched > uniform > mis-banded (or whatever is true) — "a correct prior helps, a wrong prior costs X, and here is the price of guessing."
2. **A band-selection recipe (the method):** implement + test the cheapest defensible selector — spectral estimate of data timescales (FFT/autocorr per component) → band assignment — vs (a) matched-designed, (b) **mass-lr-as-initializer** (train short with mass_lr_mult=10 to induce the ordering per CM-5, then freeze/snap masses to bands and retrain). 5 seeds. Deliverable: selector-vs-oracle gap; the V3 short needs one honest sentence: "bands can be chosen from data by [recipe] at [cost] vs the oracle prior."
3. **Sweet-spot generalization (mass-lr follow-up 1):** mult ∈ {3,10,30} × N ∈ {2,4} × data-mass-ratio ∈ {4×,16×}, 3 seeds, 1500 ep — does "≈10× induces ordering, 100× inverts" hold beyond the original cell? (Confirms/denies the CM-5 sweet-spot as a usable default.)
4. Negative results (mis-banded costs, selector failures) → written up per charter C-9 (they are appendix material, not embarrassments).

**Acceptance:** V3.2's attack answered end-to-end: confound killed (item 1) + selection story exists (item 2) + inducer characterized (item 3). Flag-provenance per §5.
