# Task: v3-revision-5 — fold the pricing-at-N≤16 result into the V3 short (w16, paper-writer, SMALL)

- **Agent:** `paper-writer` · **Output:** `.claude/outputs/v3-revision-5.md` + updated `.claude/papers/v3-short/{draft.md,draft.tex}` + PDF.
- **Read first:** **`.claude/outputs/v3-pricing-n-scaling.md` §5** (the drop-in canonical wording — use it) + its §2/§3 tables + the two figures (`.claude/outputs/v3-pricing-n-scaling/fig_{pricing_law,exponent_vs_N}.png`) · your own `.claude/outputs/v3-revision-4.md` (it left a **marked slot** for exactly this) · `.claude/claims_matrix.md` **CM-10** (now carries the N≤16 extension) · v3-referee-2 MF-2 (the composition gap this closes).
- **Note:** `v3-pricing-n-scaling` was Hub-recovered from a torn-down analyst thread (training complete, analysis re-run from the committed grids + PREREG). The result is clean and pre-registered; treat it as a normal analyst deliverable, but if you want, the numbers re-derive in seconds via `analyze.py`.

## The one thing that changed
The priced-channel law is **no longer N=2-only**. It holds at **N∈{2,4,8,16}, both chain+ring, 5 seeds, pre-registered** (sync −0.49±0.02, n₁/₂ −0.91±0.03, μ_rel² ≤0.45%, R²≥0.998, flat in N), and **App C's "inconclusive on trained lattices" is now attributed to the random-`W` U(1)-breaking coupling** (control κ_eff clusters to ≤1.4×), resolved by `channel_spring`. **This closes MF-2** — the composition gap where "the scaling result belongs to a mechanism it disclaims, and the physics result lives at N=2" is gone: the physics result now spans N≤16.

## Items
1. **Fill v3-revision-4's marked slot** with the §5 canonical wording. Do not re-open the rest of the draft.
2. **Update the abstract** so "scaling" may now legitimately attach to the priced channel (it holds to N≤16) — but keep it honest: N≤16, laptop, designed SO(2) units, the coupling must be U(1)-preserving.
3. **Fold the App-C resolution into §3.3 / App C**: replace "inconclusive on trained lattices" with the attributed explanation (artifact of U(1)-breaking, fixed by `channel_spring`), citing the control's clustered κ_eff. This is the sentence that most strengthens the paper — the honest concession became a resolved result.
4. **Figures:** `fig_pricing_law.png` is the natural §3.3 headline (log-log per-N); `fig_exponent_vs_N.png` the flat-in-N corroboration. Embed at least the first.
5. **Scope discipline:** N≤16, both topologies, designed units, `channel_spring` only. State it. The random-`W` control's failure is a *result* (it explains App C), not a weakness — present it that way.
6. Keep CM-13 scope (reversible: γ=0 only, not in trainer) and MF-1 (theory note = Anonymous) as-is; do not re-litigate.

**Acceptance:** the slot filled with the canonical wording; abstract honestly upgraded; App C resolved in-text; the headline figure embedded; scope stated; PDF clean. **This should move V3 from weak-accept toward accept** — the referee said "one new experiment (pricing at N>2) would move this to accept," and it landed positive. Report whether, in your judgement, the composition objection is now fully answered.
