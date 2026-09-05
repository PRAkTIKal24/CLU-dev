# Task: v3-pricing-n-scaling — the one experiment that moves V3 to accept (w15, analyst)

- **Agent:** `results-analyst` · **Output:** `.claude/outputs/v3-pricing-n-scaling.md`
- **Read first:** protocol (**§5 — the new pre-registration rule is MANDATORY here: this task's acceptance criterion is a set of measured exponents**) · **`.claude/outputs/v3-referee-2.md`** (MF-2, and the sentence *"One new experiment (pricing at N > 2) would move this to accept"*) · `.claude/outputs/v3-lattice-build.md` (the pricing law + the `κ_eff` extractor) · `.claude/outputs/seed-sweeps.md` (extractor validation) · `.claude/outputs/v3-interference-extra.md` (the N∈{2..16} apparatus — reuse it) · `.claude/claims_matrix.md` **v1.9** (CM-9, CM-16a/b, CM-17).
- **Repo:** read-only. Runs in parallel with `v3-revision-4` (writer), which is holding a marked slot for your result.

## The gap, stated exactly
V3's surviving physics-specific claim is the **priced channel**: `μ_rel² = 4κ/M` and `sync ∝ κ^{−1/2}`, `n₁/₂ ∝ κ^{−1}`. It is measured on **2-unit** trained lattices, 3 seeds. `App C` concedes that on trained lattices the `κ_eff` exponents are **inconclusive** — *"the exponent authority remains the designed lattice."*

So the paper's scaling result belongs to a mechanism it disclaims (the firewall is parameter separation, not physics), and its physics result lives at `N = 2`. **Close the second half.**

## Items
1. **Pre-register.** Before running anything, write `PREREG.md`: the predicted exponents (`sync ∝ κ^{−1/2}`, `n₁/₂ ∝ κ^{−1}`, `μ_rel² = 4κ/M`), the `N` values, the seeds, and — critically — **what result would falsify the pricing law at `N > 2`**. State in advance how you will distinguish "the law holds" from "the law degrades with `N`" from "the extractor loses power with `N`."
2. **Pricing at `N ∈ {2, 4, 8, 16}`**, ≥5 seeds, on **trained** lattices, chain and ring topology. Reuse `v3-interference-extra`'s apparatus and the validated `κ_eff` extractor. Report per-`N` exponents with CIs.
3. **⚠ Two prerequisites that did not exist when the original pricing was measured — use them.**
   - **`coupling_type="channel_spring"`** now exists (`lattice-xy-prereqs`, on `main` at `df5e44d`). The shipped random-`W` `spring_coupling` **breaks the global U(1)** and a trained one learns `J/J_true ≈ 0.02` with `h₂/|J| = 0.6–2.1`. **A pricing law measured through a symmetry-broken coupling is measuring the wrong object.** Run the designed channel spring as the primary arm; if you also run the legacy random-`W` arm, that is a **control**, and its disagreement is a *result* (it explains App C's "inconclusive").
   - **`GatedCoupling` now returns the free energy, not the mean energy** (same branch). If any pricing arm ever routed through a gated edge, the old force reversed sign at `v = 0.8020`. Confirm your arms do not, or re-run.
4. **Say whether App C's "inconclusive" was a physics fact or an artifact.** This is the highest-value sentence you can write. Three candidate explanations, and the data can separate them: (a) the trained coupling was U(1)-broken (item 3) ⇒ artifact, now fixable; (b) the extractor loses power at `N > 2`; (c) the law genuinely degrades. **Do not assume (a) because it is the convenient answer** — test it.
5. **Scope honesty.** `two_timescale_orbits.py` **cannot identify a coupling** (independent per-unit phases; trained `so2_invariant` units collapse to `r* < 1e-3` — no ring, no coset). If your training data is that generator, the coupling is unidentified and the whole measurement is void. **Check your data generator before you train anything**, and say what you used.
6. Report a one-paragraph, drop-in verdict for `v3-revision-4`'s marked slot: does the pricing law hold at `N > 2`, and at what scope?

**Acceptance:** PREREG written before measurement; exponents at `N ∈ {2,4,8,16}` with CIs, ≥5 seeds, on a U(1)-preserving coupling; an explicit verdict on App C's "inconclusive"; the data generator justified. **A clean negative is a publishable outcome and a fine result** — if the pricing law degrades with `N`, V3 says so and scopes its title accordingly. Do not manufacture an exponent.
