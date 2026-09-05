# v3-referee — paper-referee report (adversarial review of V3 short, draft v0.3)
Task + acceptance criterion: simulate an ML4PS composite reviewer against `.claude/papers/v3-short/draft.md`; verdict + MUST/SHOULD/NICE + missing-experiment list + 3 hostile sentences. Report only, no draft edits.
Status: done

---

## VERDICT: **weak-accept**, conditional on MF-1 (the headline framing). **Borderline → reject if MF-1 is left as written.**

**Meta-review.** This is a genuinely strong, unusually honest workshop short. It answers the exact credibility attack (V3.1: "interference NTK unmeasured — V3's own named firewall") that the register flagged as the biggest standing V3 gap, and it answers it with a *measured foil* (the parameter-matched shared-potential monolith) rather than an assertion — the single most important thing this paper had to do, and it does it. The metric discipline is exemplary: the draft pre-empts the "0.99 NTK cosine" cheat by name and reports basin-displacement R instead (§3.2, App D). Scope qualifiers, negative results (App B/E/G), and flag-provenance (App A) are all present and honest; the "banding is a method not an oracle gift" and "loan called at ≈700 steps" framings are the right defensive posture. **The one thing standing between this and a clean accept is that the *headline claim itself* — "O(1)-in-N vs O(N)" interference — is not supported by the two-point measurement it rests on.** The modular aggregate interference grew 2.6× across the single N-doubling measured (N=4→8), the monolith grew 2.2×; the two curves have *near-identical empirical slopes* and are separated only by absolute magnitude (4 decades) and by the structural fact that non-neighbors are exactly zero. The "flat in N / O(1) vs O(N)" *growth-rate* story is the most quotable sentence in the paper and it is the one a sharp reviewer will falsify from the paper's own Appendix numbers. Reframe it around the structural argument (coordination-number boundedness + R_far≡0 exact) and the absolute separation, or add N points — either fixes it, but as written it is the paper's throat.

---

## Itemized findings

### MUST-FIX

**MF-1 — The headline "O(1)-in-N vs O(N)" is a two-point claim whose two points have the same slope. (Abstract; §1 contribution 2; §3.2 "The scaling result"; §4; headline Figure 1.)**
- *The attack (G3 toy-scale + scaling-law-from-2-points).* The O(N)/O(1) separation is asserted from exactly two lattice sizes, N=4 and N=8. From the source (`v3-interference-ntk/interference_init.json`, `v3-scaling-figure.md`): modular aggregate S grows **6.79×10⁻⁵ → 1.74×10⁻⁴ (×2.56)**; monolith grows **0.635 → 1.384 (×2.18)**. The modular curve grew *faster in relative terms than the monolith*. On a log-y plot (which is exactly what Fig 1 will be once the scaling PNG is swapped in) a reviewer sees **two roughly-parallel positive-slope lines 4 decades apart** — not "one flat, one rising." The neighbour leak R_nn itself doubles (4.53×10⁻⁵ → 9.92×10⁻⁵) and the error bars do not overlap, so the modular growth is real, not seed noise. "Flat in N" is therefore false as an *empirical growth-rate* statement.
- *The evidence that the result is nonetheless real.* The O(1) claim is legitimately supported by two things the draft already has, just not the two-point slope: (a) **R_far ≡ 0.0 exactly** — non-adjacent units contribute nothing, so per-unit S is bounded by the chain coordination number (≤2), independent of N; and (b) the **4-orders-of-magnitude absolute separation**. Both are structural/topological, not empirical-scaling, arguments.
- *Triage: MUST-FIX (reframe, not a missing result).* Recast the headline as **"structurally O(1): interference is confined to graph neighbours (R_far ≡ 0 exactly), so per-unit received interference is bounded by coordination number, not width — 4 decades below a monolith that has no such confinement"** and drop or heavily qualify the "grows O(N)" / "flat in N" growth-rate language. If the growth-rate framing is to be *kept*, it needs ≥3–4 N values (N∈{2,4,6,8}) showing divergent slopes — currently unavailable (see missing-experiment list). The claim-matrix wording CM-9 ("O(1) in N" / "O(N)") inherits the same weakness and should be re-scoped in lockstep. This is the difference between a reviewer quoting your figure to *support* you vs to *bury* you.

### SHOULD-FIX

**SF-1 — "monolith interferes with itself more than it stores (S=1.38>1)" leans on the S=1 reference line, which is an artifact of the R normalization, not a storage capacity.** (§3.2; §4; Fig 1 caption.)
- *The attack.* S is a sum of *normalized* basin displacements (each R normalized by the intended change at the updated unit). "S>1 ⇒ interferes with itself more than it stores" reifies a normalized force-field ratio into a storage claim. A reviewer will ask: S=1 is not a measured storage threshold; it is the point where summed cross-unit ΔF equals one unit's own ΔF. That is suggestive but the "more than it stores" gloss overstates it.
- *Evidence.* The metric R is explicitly ΔF-based, not a re-settle of attractors (draft §5 concedes this; App C lists "dynamical interference half-life" as unmeasured).
- *Triage: SHOULD-FIX.* Soften to "aggregate cross-unit force perturbation exceeds a unit's own update magnitude by N=8" and keep the storage interpretation as suggestive, cross-referenced to the App-C unmeasured dynamical half-life.

**SF-2 — sync "≤8%" is 7.5% by the source's own denominator but 8.2% against the measured value on the first row.** (§3.3 table + abstract "pointwise to ≤8%".)
- *The attack (certificate fine print / G5).* The source `sync_rel_err` = {0.0754, 0.0091, 0.0207, 0.0448, 0.0322}, i.e. max **7.5% relative to the prediction**. The first row is sync_pred 210.9 vs sync_meas 195 — **8.2% if a reviewer normalizes by the measured value** (16/195). The claim is honest under the natural "error relative to prediction" reading, but "≤8%" sits right on the boundary and a hostile reviewer will pick the 211-vs-195 row.
- *Triage: SHOULD-FIX.* State "≤7.5% relative to the registered prediction (max residual on the weakest-coupling lattice)" — the precise, unimpeachable form. Do not round 7.5→8 upward into a rounder-looking bound; it invites the exact recomputation that flips it.

**SF-3 — Figures carry no per-figure verification/evidence/structural grade label (task item 6, C-2).** The *sections* are graded (§3.1 verification, §3.2 evidence, §3.5 structural, etc.) but the figure captions in the asset map (Figs 1–5) do not repeat the grade. A reviewer skimming figures should not have to infer whether Fig 4 (designed-lattice price law) is verification while Fig 3 (pricing parity) is evidence. *Triage: SHOULD-FIX* — add "[verification]"/"[evidence]"/"[structural]" to each caption.

**SF-4 — Six contributions + six result subsections is over-budget for a 4–5pp ML4PS short.** Contributions 5 (reversible O(1), structural, untrained, not-in-trainer) and 6 (price-of-physics) are load-bearing for the *program* but secondary to the paper's own threefold thesis (firewall / price list / banding). *Triage: SHOULD-FIX* — the pruning pass (C-10) should demote §3.5–§3.6 toward appendix-forward summaries so the three headline results breathe on p.1–3. Flag for the Hub's deadline pruning run, not a blocker.

### NICE

**N-1 — "guarantees survive scaling only because of modularity" (§3.2, §1) is stated cleanly with the monolith foil, but the scope qualifier (N≤8, chain, 2-dim, MLP) is section-header-level, not in-sentence (C-5 prefers in-sentence).** The contribution-2 bullet says "O(1) vs O(N) in width" with no inline scope; scope arrives in the block-closing paragraph (line 33). Consider inlining "(measured N≤8, chain, 2-dim units)".

**N-2 — App A.6 memory metric caveat is exemplary** (XLA `temp_size_in_bytes` = "compiler scratch estimate … a proxy for tape size, not runtime peak-RSS"). Keep it; it pre-empts the "946× is a compiler artifact not real HBM" attack. No action.

**N-3 — "block-structured monolith … Unmeasured" is honestly in App C and §5,** but a reviewer's *first* question ("is it modularity or is it just separate nets?") deserves a one-sentence pointer in §3.2 main text, not only the appendix. The separate-nets extreme trivially gives zero interference; the block-diagonal monolith is the discriminating control. See missing-experiment list.

### Known-pending (confirmed in-flight; per task, flagged once, not spending findings)
- Fig 1 is the interference *bars* placeholder with a swap note; the O(N)-vs-O(1) scaling-curve PNG exists (`v3-interference-ntk/fig_scaling_curve.png`, 61 KB, dated Jul 9) and swaps in next revision. **Note:** when it swaps in, it will *visually expose MF-1* (two parallel-slope curves) — MF-1 must land before or with the swap.
- Fig 2 (banding degradation) has no PNG; §3.4 "Figures 2–3" pointer is pending.
- Three bib strings pending (Mo; Di Bernardo/Keller; RevNet/checkpointing/momentum-net). Related-work *claims* reviewed; strings not.

---

## Charter / matrix compliance (spot audit)
- **CM-9 firewall:** ✅ measured, monolith foil present, metric discipline (R not cosine) enforced §3.2 + App D. ⚠ the "O(1)/O(N)" *wording* overreaches the 2-point data — MF-1. Numbers match source (2e-5 modular, 0.20 monolith, 1:9,000, slope 1.99, R_far≡0).
- **CM-10 pricing-predictive:** ✅ "registered before measurement" stated; ranking Spearman 1.0 exact; continuous sync ≤7.5% (draft says ≤8% — SF-2). n₁/₂ correctly restricted to ranking-only (source n12_rel_err 47–56% — draft does not overclaim it pointwise; honest). ✅
- **CM-11 banding-as-method:** ✅ "when timescales are spectrally separable" qualifier travels with the gap-0.000 claim (abstract + §3.4(ii)); mis-banding price curve foregrounded §3.4(i) as the anti-"told it the answer" defense. ✅
- **CM-13 reversible:** ✅ γ=0-only, "not wired into shipped `train_chlu`", CPU/small-D wall-time caveat all sit next to the claim (§3.5 + §5 + App A.6); graded "structural measurement on untrained models" honestly, not smuggled as evidence. ✅
- **CM-1 price-of-physics:** ✅ ≈700-step crossing, "NOT lowest plateau" disclaimer (broken-vol 0.14 / LSTM 0.13 < CLU 0.22), reach secondary/aggregate-only all present §3.6 + App F. ✅
- **C-1 (no audit confession):** ✅ absent, correctly noted in status blurb.
- **C-8 hermetic:** ✅ cites only J&P 2026 + Anonymous theory note; no cross-short cites.
- **Forbidden claims (CM-3 energy-as-better-confidence):** none present. ✅

No internal number contradictions found between §3, appendices, and source JSON (the M4/cross-section-drift class the matrix exists to prevent — clean this time).

---

## Missing-experiment list (for the Hub)
1. **[genuinely missing] Interference at N∈{2,6} (≥ two more sizes).** The whole O(1)-vs-O(N) headline currently rests on N∈{4,8}. Four points showing *divergent slopes* would convert MF-1 from a reframe into a defensible empirical scaling claim. Cheap (same harness). **Highest leverage.**
2. **[genuinely missing] Block-structured monolith control** (App C cat. iii). A single V with explicit block-diagonal structure — the "is it modularity or just separate nets" discriminator. A reviewer's first question; currently only asserted as future work. Would either strengthen ("block structure alone recovers the firewall") or bound the claim.
3. **[genuinely missing] Trajectory-mediated interference channel** (App C). §3.2 measures the potential NTK / force field; banding changes *which* loci drive updates — the indirect channel is unmeasured. Repeat §3.2 with loci driven from actual banded-vs-uniform rollouts.
4. **[genuinely missing] Dynamical interference half-life** (App C) — needed to license the "interferes with itself more than it stores" (SF-1) gloss as a *storage* rather than force-field statement.
5. **[in-flight, already in handover] Reversible-BPTT wired into `train_chlu` + accelerator (GPU/HBM) memory & honest wall-time at T≳4k;** gradient-checkpointing O(√T) Pareto baseline. Not blocking for the workshop short (§3.5 is graded structural), blocking for the ICLR systems claim.
6. **[in-flight, already in handover] Interference/pricing beyond N=8 and beyond MLP potentials.**

---

## The three sentences a hostile reviewer would quote
1. *"The paper's headline is an O(1)-vs-O(N) interference separation, but it is measured at exactly two lattice sizes — and across that single doubling the 'O(1)' modular curve grew by 2.6× while the 'O(N)' monolith grew by 2.2×, so the two curves have the same empirical slope and the claimed scaling distinction is invisible in the authors' own Appendix A/scaling data."*
2. *"The authors never run the obvious control — a block-structured monolith — so the paper cannot distinguish 'modularity buys the firewall' from 'they simply trained N nearly-separate networks,' and they relegate this to the appendix's unmeasured list."*
3. *"'Predicted pointwise to ≤8%' is stated at the boundary: the weakest-coupling lattice mispredicts synchronization by 8.2% of the measured value, and the recall-horizon n₁/₂ itself is off by 47–56%, so the 'predictive price list' delivers exact *ranking* and one continuous quantity near its own error bar — a narrower result than the abstract's framing implies."*

## Proposed handover updates (for the Hub)
- **V3 blocker before submission:** MF-1 (reframe O(1)/O(N) around R_far≡0 + coordination-number + absolute separation, OR add N points). CM-9 wording inherits the same fix — re-scope the matrix row in lockstep so the cross-short program stays consistent.
- **Two new Hub task candidates (genuinely missing, cheap, same harness):** (a) interference at N∈{2,6}; (b) block-structured-monolith control. Both directly retire the two most-quotable hostile sentences above.
- **Editorial (pruning pass):** demote §3.5/§3.6 to appendix-forward summaries (SF-4, page budget); add per-figure grade labels (SF-3); restate sync bound as "≤7.5% vs prediction" (SF-2); inline N≤8 scope on contribution-2 (N-1).
- Draft is otherwise charter/matrix-compliant and closes the V3.1 standing attack convincingly. No internal-contradiction (M4) issues found.
