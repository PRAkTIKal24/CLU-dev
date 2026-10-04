# v2-revision-2 — paper-writer report

**Task + acceptance criterion:** fold SF-1/2/3 (from `v2-referee-experiments.md`) into V2 short; remove all three `[pending: v2-referee-experiments]` tokens; PDFs build; diff-summary per item. Edit `.claude/papers/v2-short/` in place (CHANGELOG v0.3).
**Status:** **done** — all three tokens closed (md=0, tex=0), tectonic build clean (draft.pdf 596 KiB, 5 figures), CM-4/CM-6 amended wordings + canonical numbers used verbatim from the source report.

## What I did (diff-summary per task item)

### Item 1 — SF-1 (§3.2 / Fig 2b): predictor-substitution closed + 44% correction
- **New paragraph "Predictor-substitution closed (measured, Figure 2b)"** after the Lyapunov-blindness sentence in §3.2. Wires Mo's OWN λ̂(T=128) as the lifetime predictor: overdamped `corr(log pred_λ̂, log meas)=0.9995` (tighter than exact-gap 0.9987), meas/pred 0.86–1.03; past EP fails in the same ballistic direction, **0.30 @ δ=4** (mirroring exact-gap 0.31). Claim: "Mo's law is the overdamped face" now rests on Mo's own estimator.
- **Corrected the 44% attribution** (was: "deviates … by up to 44% when gap·T≲0.1"). Now: finite-horizon λ̂(T=128) bias is **−15.6% max deep-overdamped (gap·T<0.1)**, rising to a per-row max **−44.5% at the near-EP row δ=0.17 (gap·T≈3.1)** — explicitly "not at gap·T≲0.1 as an earlier draft stated." Kept the robustness note (bias lengthens predicted lifetime → meas/pred still tracks; Fig 2 prediction curve still uses exact Jacobian gap).
- **Added Figure 2b** (`figs/sf1_mo_estimator_overlay.png`, copied from source). Caption = three lifetime curves + meas/pred ratio panel, 5 seeds, shaded past-EP band. Headline remains Figure 2.

### Item 2 — SF-2 (§3.3): retire "≈4×" as a compute claim
- **Rewrote the "Honest gap" paragraph** (removed the pending token). Leads with the qualitative triad (latch / μ⁻²-budget / bounded drift) as architecture-/compute-independent; explicitly **retires the 263-vs-69 map-step ratio as a compute claim**. Honest per-step line: CLU Verlet (h64, 2 ∇V backprops) ≈**6.2× LSTM / 3.1× LEM wall (≈14–15× FLOPs)**; retention-per-compute **inverts** — 23.5× wall (54.8× FLOPs) vs LSTM, 14.6× wall (70.7× FLOPs) vs LEM. **Width-mismatch confound stated in-sentence** (h64 vs h16; sign robust; retention numbers from these exact configs).
- Inline `(≈4×)` in §3.3 prose → `(≈4× longer in map-steps; per-step compute caveat below)`.
- Contribution bullet 3 → `≈4× longer in map-steps … (the map-step ratio inverts under per-step compute normalization; §3.3)`.
- **New Appendix H** (full SF-2 table: params/FLOPs/wall + the compute-normalization inversion table), cross-ref'd to CM-4 amendment.

### Item 3 — SF-3 (§3.5 + App B.7): laws survive the cure
- **Rewrote the §3.5 closing clause** (removed pending token). Framing upgraded from "cure holds the vacuum" to **"the paper's headline laws hold under it at ≈20× the erosion horizon."** Reports (anchored λ=100, 3000 ep, 3 seeds): vacuum intact (ring depth +0.10–+0.12, r*=0.917±0.007, flat μ²≈1e-15), **GMOR exact 1.00000±1.5e-12 over 4.6 decades**, retention slope **−0.956** + floor **27.03**, **EP φ=0 below / 0.5165 above (bit-identical to 150 ep)**.
- **New Appendix B.7** with full per-seed numbers + **Figure B1** (`figs/sf3_anchored3000_laws.png`, copied from source).
- Abstract + contribution bullet 5 synced (added the "laws hold at ≈20× the erosion horizon" clause).

### Item 4 — provenance + rebuild
- **New Appendix A.5** — flag-provenance rows for SF-1 (extraction, dbeb2c2/150-ep battery), SF-2 (37dc664, per-step FLOP/timing config), SF-3 (37dc664, anchored λ=100/3000-ep/seeds 42–44 + full non-default flag set + cross-check).
- **Rebuilt draft.pdf** via tectonic (only cosmetic under/overfull-hbox warnings; no errors; 5 `\includegraphics` all resolved → figures present). Closing figure-list note updated (added Fig 2b + Fig B1; fixed stale `figA_retention_overlay`→`fig3_retention_overlay`).

## How I verified
- `grep -c "pending: v2-referee-experiments"` → **0 in draft.md, 0 in draft.tex**.
- `tectonic draft.tex` → `Writing draft.pdf (582→596 KiB)`, no errors; warnings are underfull/overfull hbox only (one pre-existing overfull in A.3).
- Key numbers present in both md+tex: 0.9995, 44.5, 6.2, 23.5, 0.917, 0.956 (grep-confirmed).
- Numbers cross-checked verbatim against `v2-referee-experiments.md` SF-1 table (lines 27–47), SF-2 tables (lines 59–77), SF-3 (lines 98–110). No number adjusted/rounded.

## Compliance notes (charter + claims matrix)
- **CM-4 amendment (SF-2):** used the approved wording — "≈4× RETIRED as a compute claim; lead qualitative triad; per-step 6.2×/3.1× wall, 14–15× FLOPs, not width-matched; retention-per-compute inverts 23.5×/14.6×." SF-1 half: "Mo's OWN λ̂(128) tracks overdamped (corr 0.9995), fails ballistic past EP (0.30); 44% is a near-EP transient (gap·T≈3.1), not gap·T≲0.1." ✓
- **CM-6 amendment (SF-3):** "GMOR exact to 1e-12 over 4.6 decades, slope −0.956, floor 27.03, latch machine-flat, EP 0.5165 bit-identical to 150 ep." ✓
- **C-2:** SF-1/SF-2 kept as evidence-grade (learned/trained-model + real timing); SF-3 labeled as designed-testbed *under the shipped cure* (verification of exactness carried through erosion) — the survival-through-3000-epochs framed as evidence for the recipe. Labels retained in captions.
- **C-5:** scale/confound qualifiers in-sentence (3 seeds; dim 4; not width-matched h64 vs h16; single-core CPU/batch-1).
- **C-6:** the corrected-44% fine print sits immediately next to the predictor-substitution claim.
- **C-8/M1:** no cross-short citations introduced; internal CM/SF labels used only as appendix provenance markers (consistent with existing draft style, e.g. §3.4 "CM-1 scope"); to be stripped at pruning.
- **C-10:** all corollary/robustness material went to appendices (H new; B.7 new; A.5 new), main text carries only the headline sentences.

## Figure-label consistency (intentional, both trace to source)
- SF-3 retention slope: prose **−0.956** (report headline, SF-3 §), Fig B1 caption **−0.961** (the figure image's own fit label). GMOR ratio: prose **±1.5e-12** (report), caption **±1e-12** (figure title). EP: prose **0.5165** (report), caption **0.516** (figure). Each number matches its own source artifact; not a contradiction.

## Open questions / follow-ups / risks
1. **"Rebuild both PDFs"** — the v2-short dir contains a single `draft.pdf` (no separate anonymized build present). I rebuilt that one; md is the canonical sync target. If the Hub expects a second (anon) build variant, it does not currently exist in the tree — flag for clarification.
2. **Pre-existing "up to 5× underdamped"** (abstract + §3.2) vs the now-explicit 0.30/0.31 ratios (≈3.2× misprediction at δ=4, the deepest of the 10 tabulated rows; the full sweep has 14 δ, so 5× is presumably a more-extreme censored row). This "5×" predates this task and is out of the three SF slots, so I left it untouched — but a referee reading §3.2 could see 0.30↔3.3× next to "5×". Recommend the Hub either (a) confirm 5× is the extreme-δ value from the full 14-point sweep, or (b) soften to "≈3×" for the tabulated range. Not changed unilaterally (out of scope + would need the source number).
3. SF-3 is 3 seeds (task-specified) vs §3.1's 5 — noted in the report's own follow-ups; if the Hub wants matched error bars, seeds 45/46 at 3000 ep are the cheap add (analyst's recommendation).

## Proposed handover updates (for the Hub)
- **V2 short at v0.3:** all three `[pending: v2-referee-experiments]` slots closed; SF-1 (Fig 2b + 44%-correction), SF-2 (§3.3 4×-retirement + Appendix H), SF-3 (§3.5 upgrade + Appendix B.7 + Fig B1) folded; A.5 provenance added; draft.pdf rebuilt (596 KiB, 5 figs). Ready for **v2-referee-2 (w10) clean-pass verdict**.
- **CM matrix:** V2 now consumes the CM-4 and CM-6 amendments in full; no forbidden claims (CM-3 energy-signal-superiority) touched. If the "5×" (item 2 above) is resolved, note it in the matrix scope line for CM-4.
- **Charter open slot (CM-6 erosion placement, Head after Jul-11 novelty confirm):** SF-3 now lives in Appendix B.7 (with the phenomenon novelty still marked pending scout); placement decision unaffected by this fold-in.
