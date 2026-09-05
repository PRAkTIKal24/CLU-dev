# Task: v3-referee — adversarial review of the V3 short (the lattice paper) (w10)

- **Agent:** `paper-referee` · **Output:** `.claude/outputs/v3-referee.md`
- **Read first:** protocol · Charter · `.claude/claims_matrix.md` v1.4 (V3 rows: CM-1, CM-5, CM-9, CM-10, CM-11, CM-13) · `.claude/critique_register.md` (esp. V3.1/V3.2/V3.3 — the original V3 attacks; verify the draft answers them) · then `.claude/papers/v3-short/draft.md` (v0.3, 4 figs embedded).
- **Simulated venue:** ML4PS composite (4–5 pp).

## KNOWN-PENDING items — do NOT spend findings on these (asset/bib finish is in-flight, w10/w11):
- Fig 1 is currently the interference *bars* figure with a swap note; the headline O(N)-vs-O(1) *scaling-curve* PNG now exists (`v3-interference-ntk/fig_scaling_curve.png`) and will be swapped in at the next revision. Review Fig 1's CLAIM, not the placeholder asset.
- Fig 2 (banding) has no PNG yet (`v3-banding-figure` running); §3.4 pointer is pending.
- Three bib strings are marked pending: Mo + Di Bernardo/Keller (from their scout reports) and RevNet/checkpointing/momentum-net (Jul-11 scout). Review the related-work CLAIMS, not the placeholder strings.
Flag these once as "known-pending, confirmed in-flight" and move on.

## Specific attack surface
1. **CM-9 firewall = the headline.** Is "guarantees survive scaling ONLY because of modularity" earned, with the monolith as the measured foil? Check the metric-discipline (report basin-displacement R, never NTK cosine — the draft must not let a reader use the 0.99 cosine). Stress the N≤8 scope: does any sentence generalize "the lattice scales" beyond N=8?
2. **CM-10 pricing-predictive:** "registered before measured" is a strong methodological claim — verify the draft states it honestly and that the ranking-Spearman-1.0 / sync-≤8% split (ranking exact, continuous quantity ≤8%) is not overstated as pointwise-exact.
3. **CM-11 banding-as-method:** the FFT selector's "gap 0.000" carries a MANDATORY qualifier (only when timescales are spectrally separable). Verify it travels with the claim (C-5). The mis-banding price curve is the anti-"we told it the answer" defense (V3.2) — is it foregrounded?
4. **CM-13 reversible:** γ=0-only exactness + "not in shipped trainer" + CPU/small-D wall-time caveat must all sit next to the claim. It is labeled structural-measurement-on-untrained-models — verify that grade label is honest (not smuggled as "evidence").
5. **CM-1 price-of-physics (shared with V2/V1):** the ≈700-step loan crossing + the "NOT lowest plateau" disclaimer + reach-secondary must all be present and scoped.
6. Contribution clarity p.1; C-2 verification/evidence/structural labels on every figure.

**Report:** verdict + MUST/SHOULD/NICE + three hostile sentences + missing-experiment list. The V3.1 interference gap the register flagged is now MEASURED — confirm the draft closes it convincingly, as that was the biggest standing V3 credibility attack.
