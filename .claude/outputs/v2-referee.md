# v2-referee — paper-referee report (adversarial review of V2 short + F5 note, w7)

**Task + acceptance criterion:** simulate a NeurReps/ML4PS composite reviewer against `.claude/papers/v2-short/draft.md` and `.claude/papers/f5-note/f5-note.tex`; run all five passes; deliver verdict + itemized MUST/SHOULD/NICE + the three sentences a hostile reviewer quotes + a missing-experiment list (should be ≈empty; flag anything not already in `.claude/outputs/`). Report-only; drafts never edited.
**Status:** done. Both drafts read in full; cross-checked against charter C-1…C-10, CM-1…CM-12, critique register G1–G6/V*/M*, and the two w6 source reports (`fit-gap-anatomy`, `anchor-robustness`).

---

## VERDICT — **borderline (leaning weak-accept after MUST-FIX)**

**Meta-review.** This is a strong, unusually honest short with a genuine contribution: an exactly-solvable retention budget *verified on trained models*, a clean containment of a published ML lifetime law (Mo's single-exponential predictor as the "overdamped face"), a qualitative-absence result against well-trained LSTM/LEM baselines, and a measured price for the physics prior. The positioning discipline (verification-vs-evidence labels, scope qualifiers, flag-provenance appendix, honest-gap paragraphs) is well above workshop median and closes most of the register's standing attacks by construction. **However, the draft as it stands cannot be submitted:** (1) it still carries the §1 audit-confession paragraph that Head policy C-1 now forbids and that is a de-anonymization vector; (2) its headline evidence figure (Fig 2, Mo head-to-head) and two other load-bearing figures are not embedded — a text-only Mo table is not a headline; (3) §3.4's emergent-lifetime numbers are internally inconsistent with their own Appendix D, a contradiction a cross-reading reviewer will find in five minutes; and (4) it under-claims a result the program now owns — the fit-vs-horizon crossing is *measured* (≈700 steps, `fit-gap-anatomy`) but §3.4 still says "unmeasured." The Mo head-to-head also has a soft spot a Mo-literate reviewer will exploit: the "prediction" uses the exact Jacobian gap, not Mo's own estimator, so "Mo's law is the overdamped face" is asserted on a substituted predictor. Fix the four MUST-FIX items and wire the two w6 folds and this is a clean weak-accept; leave them and a hostile reviewer has a reject-grade thread on internal consistency alone.

---

## MUST-FIX (blocks submission)

### MF-1 — Remove the §1 audit-confession paragraph (C-1 REVERSED). `draft.md` line 23.
The paragraph "**A brief audit disclosure (own the falsifications first).** …" states three legacy-mechanism defects with their numbers (dead Lyapunov regularizer ≡½ln(1−γ) "inert for the program's entire history"; Langevin "≈11× colder"; identity-mass cap). This is exactly the defensive audit confession C-1 now forbids ("(a) no defensive audit paragraph in any paper"), and it asserts legacy-paper mechanism-numbers as content (C-1(b) forbids). It is also a de-anonymization vector (M2): a reviewer reads "an audit methodology we apply to our own primitive" + "J&P 2026" and identifies authorship.
- **Removal is safe (verified):** no main-text claim leans on those three numbers. The mechanisms are used in current fixed form via the provenance flags (`lyapunov_penalty="max"` = the fix; `langevin_noise="legacy"` is inert here — the paper does no Gibbs sampling). The class-level theory of the corrected mechanisms already lives in the F5 note as neutral theorems (Cor. 2, §11 limitations), which is the correct register per C-1(c).
- **Consequential edits required for a clean removal (do not orphan the cross-refs):**
  - App F line 206 "**Audit negatives (physics-audit paragraph, §1):** N17 … N18 … N22 …" — the `(physics-audit paragraph, §1)` cross-ref dies with the paragraph. Reframe N17/N18 out (they are legacy-mechanism confessions, not V2 negatives) or demote to a single neutral "class-level design caveats — see the theory note" line with no legacy numbers.
  - §5 limitations line 137 references "the proven saddle-blindness caveat … Appendix F, N22." N22 (saddle-blindness / isoenergetic-escape) is a *genuine scope limitation* and may stay — but reframe it as a neutral limitation of the class, not an "audit negative," and drop the §1 back-reference.
- **Also verify (task item 1):** §2 line 44 "trained at 150 epochs (see §3.5 / Appendix B for why 150 and not the default 1000)" is not audit-confession framing (it points forward to the erosion recipe) — acceptable, keep.

### MF-2 — Wire the measured fit-vs-horizon crossing; the draft under-claims. `draft.md` §3.4 line 109; source `fit-gap-anatomy.md` + CM-1 (updated).
Line 109 currently reads: "*we do **not** claim it persists at long horizon — the fit-vs-horizon crossing is unmeasured. The twin's 15× edge is a loan, not a verdict.*" This was written before w6. `fit-gap-anatomy` **measured the crossing** (3 seeds, matched params ±0.2%): the twin leads 1–2 orders to ~500 steps, **crosses CLU at ~700 steps, and diverges to 196 @5000** vs CLU's bounded ≈0.20–0.23 plateau. CM-1 now says "DO claim the crossing + boundedness-by-construction." Evidence exists — this is a wiring note, not a new experiment.
- Add: the crossing (~700 steps), the twin's catastrophic divergence (196 @5000), CLU boundedness-*by-construction* as the asset.
- Add the recovery ladder (feeds the Pareto frame all three shorts share): **+γ global recovers 92% of the twin fit gap with BIBO/latch/μ² all preserved** (contraction is licensed; det J=(1−γ)^d); **+γ_φ learned field does NOT recover fit (−24%) — wrong tool, targeted forgetting** (paid mechanisms are not interchangeable).
- **Do NOT overclaim (CM-1 guard):** do not claim CLU has the lowest long-horizon plateau — broken-volume (0.14) and LSTM (0.13) plateau *below* CLU (0.22). The asset is boundedness-by-construction, not lowest steady MSE. State this explicitly or a reviewer who reads the loan curve will catch the omission.

### MF-3 — §3.4 emergent-lifetime numbers contradict Appendix D. `draft.md` line 95 vs lines 182–188.
§3.4 line 95: "measured $n_{1/2}=277/257/303$ vs exact-map prediction $247/227/263$ — a **consistent +12–28%** positive bias whose decomposition … is Appendix D." Computed ratios: **1.121 / 1.132 / 1.152** (i.e. +12 to +15%). Appendix D's per-seed kick→0 ratios are **1.125 / 1.288 / 1.052** (+12.5 / +28.8 / **+5.2%**). Two defects a cross-reading reviewer constructs instantly:
1. The displayed triple (277/257/303 ÷ 247/227/263) spans **+12–15%**, not "+12–28%"; the "28" is imported from a *different* set of numbers (App D seed-44).
2. The claim "**consistent**" is false against its own appendix: App D shows one seed at **+5.2%** (below the stated +12% floor) and one at **+28.8%** — the bias is *not* consistent, and +5.2% falls outside the advertised "+12–28%" band.
   Additionally §3.4 states the softest emergent μ² as "$5.1/5.9/5.4\times10^{-2}$ (3 seeds)" while App D lists $\mu^2_{\rm ang}=5.45\times10^{-2}/1.07\times10^{-1}/2.03\times10^{-2}$ — seed-44's $1.07\times10^{-1}$ is ~2× the §3.4 max. If §3.4 (source `v2-full-runs` item 4/5) and App D (`v2-prefreeze-baselines` item 4) are different probes on the same emergent checkpoints, the draft must say so and stop presenting App D as "the decomposition" of §3.4's exact triple. Pick one coherent bias range that *includes* the +5% anharmonic seed, and make the displayed numbers reproduce the range endpoints. This is precisely the cross-section contradiction M4/CM-7 exist to prevent.

### MF-4 — Load-bearing figures are not embedded. `draft.md` line 211 (figure list, no floats in text).
Text-only fails review for the three figures the argument leans on:
- **Fig 2 (Mo head-to-head)** — declared the *headline*. A workshop reviewer expects the headline figure on p.1–2; a headline table is not a headline figure. Non-negotiable.
- **Fig 1 (GMOR law + C2 metric bifurcation)** — the −0.985 log-log slope, the 4.5-decade collapse, and the floor are figure claims; a reader cannot assess "slope −0.985 (predicted −1)" from prose.
- **Fig 3 (baseline retention overlay)** — the CM-4 qualitative-absence headline (263 bounded vs 5.6/56/69 randomizing) is a *visual* claim; the overlay is where "qualitatively absent" is won or lost.
Fig 4/5 (emergent/isotropy/erosion) are appendix-grade and can ship as text+table if space-constrained. Embed Fig 1–3 before submission.

### MF-5 (F5 note) — Strip `[TODO-HEAD]` margin notes; neutralize the Cor-3 footnote. `f5-note.tex` lines 38, 234, 242, 273, 285 (margins); 284–285 (footnote).
- The five `\todohead{…}` margin notes render **in the compiled PDF** as red "[TODO-HEAD]" boxes (Open Q1 type-B Goldstone; Q3 anharmonic-% unverified; Q5 check-(c) unused; title pick; Cor-3). If the arXiv build ships with these visible, a reviewer reads them as an unfinished manuscript and as a map of the authors' own doubts (Q3: "the few-to-fifteen-percent figure is anticipated from companion work, **not verified in this note**" — directly exploitable, since the V2 short's C-2 framing leans on "laws holding to 2–15% with deviations predicted"). Strip all margin notes on the submission build; resolve Q3 by either dropping the parenthetical percentage from §11 or citing the short's own §3.4 evidence (within anonymity limits).
- **Cor-3 footnote (lines 284–285):** "*A regularizer of the degenerate form appears in **at least one public instantiation, where it is inert by construction**…*" — the margin note itself flags this as "the most de-anonymizing item." Combined with the J&P 2026 reference-instantiation citation, this is an implied audit confession + a de-anon vector (M2), and it is the F5 analog of the §1 paragraph C-1 just deleted. Task item 1 requires the F5 note to read as neutral class-level theory, "never 'our previous paper was wrong'." **Recommend cut, or reduce to the pure class fact** ("a mean-spectrum penalty is degenerate for the whole class" — already in Cor. 2 body) with the "public instantiation" clause removed. NOTE: this collides with the standing Head decision (charter C-10 appendix-maximalism: "retain … incl. the Cor-3 footnote"). **Triage: MUST-FIX-before-arXiv, escalate as a DEC** — the Head's "retain until pruning" and the C-1/M2 anonymity constraint are in direct tension on this one footnote and only the Head resolves it.

---

## SHOULD-FIX

### SF-1 — §3.2 Mo head-to-head: the "prediction" is the exact Jacobian gap, not Mo's estimator. `draft.md` line 71.
"*we used the exact Jacobian gap for the prediction and report both if his protocol is adopted verbatim.*" A Mo-literate reviewer (task item 2): *"You claim 'Mo's law is the overdamped face,' but your predictor is your own exact gap, not Mo's finite-horizon $\hat\lambda(T{=}128)$ estimator — which you yourself say deviates up to 44% when gap·T≲0.1. So the overdamped agreement is between YOUR prediction and measurement; you never showed Mo's actual estimator tracks the budget."* The containment argument needs Mo's *own* estimator run on the trained models across all regimes, plotted beside the exact-gap prediction — not a conditional "report both if adopted verbatim." The 44% number implies the data exists in `v2-full-runs`; make it unconditional and put Mo's estimator curve in Fig 2. Otherwise the headline is one substituted-predictor question away from unraveling. **Does the "overdamped face" claim survive Mo's framing?** Mostly yes — the containment logic (saturation floor + EP ringing are structurally invisible to a first-order flow) is sound and mo-deep-read §4's positioning is preserved, not diluted. But close this predictor-substitution gap or the win is contestable.

### SF-2 — §3.3 "≈4× longer" is in map-steps with no FLOP/wall-time conversion. `draft.md` line 89.
The honest-gap paragraph (task item 3) *is* honestly stated, not buried — it names map-applications-vs-wall-time and the input-driven-RMSE gap (N6) in the main text. Good. But the residual attack: one CLU dissipative-Verlet step (hidden-64 MLP potential, KDK) is not obviously ≤¼ the cost of one LSTM cell, so "263 vs 69 map-steps = 4× longer" can evaporate in wall-clock/FLOPs. The draft concedes "the 4× ratio is not [unit-free]" but gives no conversion. Add a one-line per-step FLOP ratio (cheap to compute) so the 4× survives a compute-normalized reading; otherwise a reviewer discounts the quantitative retention edge and you keep only the *qualitative* triad (which is the stronger claim anyway — consider leading with qualitative absence and demoting the 4×).

### SF-3 — Retention laws verified at 150 ep; anchored 3000-ep model's laws not re-verified.
§3.1/§3.2 (GMOR, latch, Mo) run on 150-ep checkpoints — an epoch count chosen (§2/§3.5) to *precede* vacuum erosion. §3.5 then shows the anchor holds the vacuum to 3000 ep, but the retention *law sweep* (tilt-δ GMOR, EP onset) is not re-run on an anchored long-horizon checkpoint. Reviewer: "Do your headline laws still hold once you train past the erosion horizon with your own cure?" `anchor-robustness` shows μ²_min≈1e-6 (flat mode preserved) and r*≈0.91 at 3000 ep λ=100, so the latch almost certainly survives — but the GMOR sweep on the anchored model is a genuine, cheap confirmation that is currently absent. Small Hub candidate (see missing-experiment list).

### SF-4 — Results ordering mildly violates C-3/C-4 (ML-first). §3.1 leads with designed-testbed machine-precision verification.
Charter C-3: "The Mo head-to-head … leads the results"; C-4: "Lead with measured." Structurally §3.1 (designed-testbed, 4.5-decade, machine-precision, labeled *verification*) physically precedes §3.2 (Mo, the declared headline) and §3.3 (baselines, *evidence*). Opening a workshop short with entry-by-entry machine-precision verification of a 2×2 theory risks exactly the "damped-oscillator problem set" reading C-3 warns against. Consider reordering: lead §3 with Mo + baseline collapse (evidence), fold the GMOR-exactness verification behind them (it is *labeled* verification, so demotion is on-message). At minimum, a one-paragraph "what a trained network obeys" evidence-first framing before §3.1.

### SF-5 — Reach-pricing is scoped out but is now measured. `draft.md` §3.4 line 109.
"*the second, separately-proven restriction (reach-pricing) is out of this paper's scope.*" `fit-gap-anatomy` item 1 now *measures* it (relativistic cap active: +77% MSE at c=0.5, collapses once v_max≥req, aggregate-only, no per-step far-region concentration). Scoping it out is defensible for a 4–5pp short, but "separately-proven" undersells "separately-measured." Either update the parenthetical to "separately measured (secondary, aggregate-only at single-unit scale)" or leave fully to V1/V3 — but don't imply it is only theoretical when evidence now exists.

### SF-6 — "solved constitutive problem" is a scope-widened phrase. `draft.md` §5 line 135.
"*We have treated retention in a trained physics-structured recurrent unit as a **solved** constitutive problem.*" At dim 4, ≤5 seeds, one S¹ toy, laptop-CPU, quadratic core with local linearization, this is the sentence a hostile reviewer quotes as overreach (C-5). The limitations paragraph that follows walks it back, but the topic sentence should not hand the reviewer the quote. Soften to "as a *constitutive* problem with an exact solvable core and quantified anharmonic deviations."

---

## NICE

- **N-1** Abstract "The budget **subsumes** a recently published ML lifetime law" (line 13) — "subsumes" is more aggressive than §3.2's "containment, not conflict." Align to "contains … as its overdamped limit" to match the (correct, softer) body framing and not antagonize a Mo-aligned reviewer at the abstract.
- **N-2** F5 abstract "All **twelve** results are verified" vs 13 checks (a–m) in the table, one of which (c, squeeze) is "claimed nowhere" per Open Q5. Reconcile the count or prune row (c).
- **N-3** Erosion novelty hedging (task item 4) is adequate: §3.5 header, App B header, and §4 all carry "pending scout confirmation." One residual — §4 line 129 "our §3.5 **contributes** a sharp instance … a degeneracy-specificity demarcation, and a cheap value-anchor cure (novelty pending, §4/Appendix B)" — "contributes" reads as a novelty assertion; the "(novelty pending)" tag saves it, but consider "our §3.5 *reports*" until Jul-11 confirms. No unhedged novelty sentence found elsewhere.
- **N-4** Page budget: main text §1–5 is dense and, with three embedded figures, will likely exceed 5pp. The pruning pass (C-10) should target §3.1's machine-precision enumerations (→ App E is already partly there) and the double-reporting of emergent lifetimes across §3.3/§3.4/App D.
- **N-5** Related-work (§4) coverage is good vs the scout ledgers (Mo, Di Bernardo, Keller, Iqbal-Goldstone, LSTM/LEM/coRNN, geometric-integration, Hinton/Tieleman/Fischer-Igel/Nijkamp). No missing neighbor of concern for a memory-framed short.

---

## Reviewer-hat attack pass (register G1–G6 / V* / M* against THIS draft)

- **G1 (unit test on a theory-built testbed):** *Largely defended.* The verification/evidence labeling is explicit and the constitutive-vs-kinematic contrast is foregrounded (C-2). Residual: SF-4 ordering; and the emergent-lifetime inconsistency (MF-3) hands G1 a foothold ("even on your own testbed the numbers don't reconcile").
- **G2 (which component buys what):** *Well answered* by the twin/broken-volume table (§3.4) and — once MF-2 lands — the recovery ladder (+γ 92%, γ_φ −24%). This is the draft's strongest anti-G2 asset; wiring the ladder makes it airtight.
- **G3 (toy scale):** Scope qualifiers are attached (dim 4, ≤5 seeds, laptop, S¹). §5 "solved" (SF-6) is the one scale-free slip. Otherwise compliant.
- **G5 (certificate fine print):** BIBO is stated within "coercive-potential / compact-sublevel-set scope" with the saddle-blindness caveat in §5 main text (C-6 compliant). Good — but keep N22 as a neutral limitation post-MF-1, not an "audit negative."
- **G6 (foundational-paper falsifications):** This is the crux of the C-1 reversal. MF-1 (draft §1) and MF-5 (F5 Cor-3 footnote) are the two live G6/audit-confession surfaces; both must be closed. Post-fix, G6 risk is carried by the F5 note as neutral class theorems, as intended.
- **M2/M3 (de-anon / salami optics):** MF-1 and MF-5 are also the de-anon vectors. Citation architecture is otherwise C-8-clean (only J&P 2026 + the theory note, third person; no cross-short citations). The three-shorts program reads as one coherent matrix.
- **Fresh attack the register missed — predictor substitution (SF-1):** the Mo head-to-head predicts with the exact Jacobian gap, not Mo's estimator. A venue reviewer who has read Mo will press this; it is not in the register.
- **Fresh attack — cross-section number drift (MF-3):** the emergent lifetimes appear in three places (§3.3 table, §3.4 prose, App D) with three different number sets. This is the M4-class contradiction and it is currently constructible.

---

## Missing-experiment list for the Hub (should be ≈empty — flagged items are wiring, not new science)

1. **[WIRING, not missing]** Fit-vs-horizon crossing + recovery ladder — evidence exists in `fit-gap-anatomy.md` (crossing ≈700; +γ 92%; γ_φ −24%). MF-2. No new run.
2. **[WIRING, not missing]** Anchor 3000-ep envelope + tilt-immunity-as-theory + anchor⊥volume — evidence in `anchor-robustness.md`. **Already wired** in §3.5 and App B.4 (verified: 3000 ep, λ∈{1,10,100}, λ=100 5/5 r*=0.911±0.016, tilt δ≥0.05, 35× wake MSE, broken-vol non-rescue). Contrary to the task's prior characterization, §3.5/App B are **not** on the older 2-seed cure — this fold is done. No action beyond confirming numbers match (they do).
3. **[WIRING]** Mo's own $\hat\lambda(T{=}128)$ estimator curve across all regimes on the trained models (SF-1). The 44% deviation figure implies the data is in `v2-full-runs`; if so, this is a plotting/wording fix. If the full-regime estimator sweep was *not* run, it is a small genuine experiment — **flag to confirm with the analyst.**
4. **[SMALL GENUINE CANDIDATE]** GMOR tilt-sweep + EP onset re-verified on an **anchored 3000-ep** checkpoint (SF-3) — confirms the headline laws survive past the erosion horizon under the shipped recipe. Cheap, laptop-scale, not currently in any output. Hub task candidate.
5. **[SMALL GENUINE CANDIDATE]** Per-step FLOP/wall-time conversion for CLU-step vs LSTM/LEM-cell (SF-2) — to defend or retire the "4× longer" retention claim in compute-normalized terms. Trivial to measure; not in outputs.

Everything else the draft claims traces to an existing report. No large missing experiment.

---

## The three sentences a hostile reviewer would quote

1. **(§3.4, line 95)** *"the emergent lifetimes are priced by the budget: measured $n_{1/2}=277/257/303$ vs exact-map prediction $247/227/263$ — a consistent +12–28% positive bias …"* — "The displayed numbers give +12–15%, the appendix that supposedly decomposes them lists +5.2% and +28.8% with different μ², and the authors call this 'consistent.' If the paper's own two sections disagree on its central quantitative fit, why should I trust the machine-precision claims?"
2. **(§3.2, line 71)** *"we used the exact Jacobian gap for the prediction and report both if his protocol is adopted verbatim."* — "So the 'Mo is the overdamped face' headline compares the authors' *own* exact prediction to measurement; Mo's actual estimator — which they admit is off by up to 44% — is never shown to track the budget. The containment claim rests on a substituted predictor."
3. **(§5, line 135)** *"We have treated retention in a trained physics-structured recurrent unit as a solved constitutive problem …"* — "'Solved,' at dim 4, five seeds, one circle task, on a laptop, with a quadratic core linearized at a critical point. This is a promising verification of a 2×2 model, not a solved problem."

(Runner-up, if MF-1 is not fixed: the §1 audit paragraph — *"that methodology first turned up three defects in the reference instantiation"* — which both violates C-1 and de-anonymizes the submission.)

---

## Craft summary
- **p.1 contribution clarity:** strong — five numbered contributions, headline explicitly tagged (§3.2, Fig 2), verification/evidence discipline stated. Keep.
- **Headline figure:** declared (Fig 2) but not embedded — MF-4.
- **Related work:** adequate vs scout ledgers (N-5).
- **Appendix completeness (C-10):** good — flag-provenance A.1–A.4, erosion (B), isotropization (C), bias decomposition (D), EP (E), negatives (F). App F needs the MF-1 reframe.
- **Page budget:** likely over 5pp with figures; pruning-pass targets in N-4.

---

## Proposed handover updates (for the Hub)
- **V2 draft, MUST-FIX before submission:** (MF-1) delete §1 audit paragraph + reframe App F "audit negatives" + §5 N22; (MF-2) wire the ≈700-step crossing + recovery ladder into §3.4 per CM-1; (MF-3) reconcile §3.4↔App D emergent lifetimes (numbers do not currently match — verified: 1.121/1.132/1.152 vs App D 1.125/1.288/1.052, and +5.2% falls outside the claimed "+12–28%"); (MF-4) embed Fig 1/2/3.
- **F5 note, MUST-FIX before arXiv:** strip all `\todohead` margin notes; resolve Cor-3 footnote de-anon (escalate as DEC — collides with C-10 "retain until pruning").
- **SHOULD-FIX:** SF-1 Mo-estimator curve (confirm data exists with analyst); SF-2 FLOP conversion; SF-4 ML-first reordering.
- **Anchor-robustness fold is already wired** (§3.5/App B.4) — the task's "older 2-seed cure" note is stale; numbers match `anchor-robustness.md`. No action.
- **Two small experiment candidates** (not blocking): GMOR sweep on anchored 3000-ep checkpoint (SF-3); per-step FLOP ratio CLU vs LSTM (SF-2).
