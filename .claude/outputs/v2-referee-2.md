# v2-referee-2 — paper-referee clean-pass re-review (V2 short v0.3 + F5 note v0.2, w10)

**Task + acceptance criterion:** clean-pass re-review of the revised `.claude/papers/v2-short/draft.md` (v0.3, 5 figs) + `.claude/papers/f5-note/f5-note.tex` (v0.2). Verify the prior MF-1…5 / SF-1…6 punch-list is CLOSED; hunt any NEW inconsistency the edits introduced; return the "5×" determination + submission-readiness verdict + the three hostile-reviewer sentences. Report-only; drafts never edited.
**Status:** done. Both drafts read in full; cross-checked against Charter C-1…C-10, claims-matrix v1.4 (CM-1/CM-4/CM-6 amended), critique register G1–G6/V*/M*, my prior `v2-referee.md`, and the source reports `v2-full-runs`, `v2-referee-experiments`, `minus-the-physics`, `fit-gap-anatomy`.

---

## VERDICT — **weak-accept (submission-ready modulo TWO cheap number-hygiene MUST-FIX items + pruning + the F5 arXiv id).**

**Meta-review.** The revision is a large, clean step up. **Every item on the prior punch-list is closed:** the §1 audit-confession paragraph is gone (MF-1); the ≈700-step crossing + recovery ladder are wired with the correct non-overclaim ("boundedness by construction, not lowest plateau"; MF-2); the §3.4↔App D emergent-lifetime contradiction is resolved by explicit attribution ("distinct probes, not one decomposing the other"; the word "consistent" is deleted; the +5% seed now sits inside a stated +5–29% band; MF-3); Fig 1/2/2b/3 are embedded and the PNGs exist on disk (MF-4); the F5 note renders zero `\todohead`, the Cor-3 footnote is reduced to the pure class fact with the "public instantiation" de-anon clause cut, and the count reads "All results" not "twelve" (MF-5). The SHOULD-FIXes also landed: Mo's own estimator now runs across all regimes (Fig 2b, corr 0.9995; SF-1 predictor-substitution closed), the "≈4× longer" is explicitly retired as a compute claim with the honest inversion in the main text (SF-2), §3 opens with an evidence-first reading-order note (SF-4), reach-pricing is now "separately **measured**" (SF-5), and "solved" is softened to "constitutive problem" (SF-6). The positioning discipline is now well above venue median and closes the standing register attacks by construction.

**But it cannot ship as-is** because the edits left/created two constructible cross-section number contradictions — the exact M4 class this program's matrix exists to prevent, and the same class as the MF-3 the last pass caught:
1. **The "5×" is not traceable to the trained-model data** (writer-flagged, unresolved). The abstract and §3.2 both say the misprediction reaches "up to 5× underdamped," but the §3.2 table and Fig 2 caption bottom at 0.309 / 0.31 (**≈3.2×**), which is also the deepest point of the full 14-δ sweep in both source reports. A reviewer reads 0.31 next to "5×" in one glance.
2. **A NEW contradiction the recovery-ladder wiring (MF-2) introduced:** §3.4's headline table shows the "CLU (symplectic)" fitting **15× worse** than the twin (0.190 vs 0.0128), while the abstract + §3.4 + App G say licensed damping "**recovers 92% of** [that gap]" — with the App-G +γ=0.05 unit at MSE **0.0216** against a **different** twin (0.0047). The §3.4 CLU is unlabeled as γ=0, and the two experiments' absolute MSE scales are cross-quoted without a comparability caveat. A hostile reviewer asks: is the shipped unit at γ=0 or γ=0.05, and why does "the same" γ=0.05 CLU fit at both 0.190 and 0.0216?

Both are wording/label fixes — **no new experiments** — but both are main-text-visible and belong to the contradiction class the paper is otherwise scrupulous about. Fix these two and this is a clean weak-accept; leave them and a cross-reading reviewer has an internal-consistency thread on the paper's two central quantitative claims (the Mo containment and the price-of-physics).

---

## The "5×" determination (task item 2) — **STALE / mis-sourced. MUST-FIX.**

**Not traceable to the V2 short's own trained-model data.** Traced to source:
- **V2 §3.2 table** (draft line 71): deepest underdamped meas/pred = **0.309 at δ=4** → 1/0.309 = **3.24×**. Fig 2 caption (line 77): "a decline to **0.31×** deep underdamped." Fig 2b caption (line 79): "**0.30** at δ=4."
- **`v2-referee-experiments` SF-1 table** (14-δ sweep, 10 shown rows): bottoms at δ=4 → 0.309 (exact-gap) / 0.304 (Mo λ̂). **The censored rows are δ≤3e-4 — the OVERDAMPED side.** There is no deeper *underdamped* row than δ=4. The trained-model sweep's max misprediction is **3.2–3.3×**, full stop.
- **`v2-full-runs` line 79** is the contamination source: it annotates "0.309 ± 0.012 **(5× failure**, calculable direction)" — i.e. the source report itself mislabelled a 3.2× ratio as "5×." The draft inherited the mislabel.
- **The genuine "5×" lives only in the F5 note**, on the *exact analytic map* at γ=0.1, ε=0.1: Prop. metric-bifurcation ("fails by up to ∼5×") and check (k), whose deepest underdamped ratio is **0.19 = 5.3×**. That is a different γ and a deeper δ than any trained-model run. The F5 note is internally consistent on 5×; the V2 short is not, because it reports trained-model data that bottoms at 3.2×.

**Fix (pick one):** (a) change "up to 5×" → "**≈3.2× (meas/pred 0.31 at δ=4, the deepest breaking tested)**" in the abstract and §3.2, matching the table/figure; or (b) if a "5×" is wanted for rhetorical reach, attribute it explicitly — "**≈3.2× at the deepest trained-model breaking; the exact map continues to ≈5× further underdamped (theory note, check k)**." Note "5×" is **not** a claims-matrix-canonical constant (CM-4 and the canonical-constants table state no numeric misprediction factor), so changing it costs nothing cross-document. **Recommend (b)** for cross-doc coherence with the F5 note.

---

## MUST-FIX (blocks submission)

### MF2-A — "up to 5×" contradicts the §3.2 table/Fig 2 (≈3.2×). Abstract line 13; §3.2 line 73.
See the determination above. This is the writer-flagged item; it is a real number-hygiene defect (0.31 in the table/caption vs "5×" in prose/abstract), on the paper's **headline** result. Cheap wording fix.

### MF2-B — NEW: the "15× fit gap" (§3.4 table) vs "+γ recovers 92% of it" (abstract/App G) is a constructible cross-section contradiction. §3.4 lines 113/123; abstract line 13; App G.2 lines 254–261.
The recovery-ladder wiring (MF-2 from last pass) spliced two experiments with **different absolute MSE scales and different twins** into one narrative without a comparability note:
- **§3.4 table (`minus-the-physics`, seeds 42–44 @ `b41410f`):** CLU (symplectic) **0.190**, twin **0.0128** → 14.8× ("≈15× better").
- **App G.2 ladder (`fit-gap-anatomy`, seeds 0–2 @ `9a13455`, wake-only):** twin **0.0047**, CLU-conservative(γ=0) **0.2066**, +γ(0.05) **0.0216** → "**92%** of the 0.202 gap recovered."

Three reader traps, all constructible:
1. **γ label missing.** §3.4's CLU (0.190) ≈ App-G's *conservative γ=0* CLU (0.2066) — so §3.4's "CLU (symplectic)" is effectively the **γ=0** unit, but it is not labelled so. A reviewer reads "the symplectic unit fits 15× worse" and "adding γ=0.05 recovers 92%" and cannot tell whether the shipped/headline unit is γ=0 (0.190) or γ=0.05 (0.0216) — **the same "CLU at γ=0.05" appears to fit at both 0.190 and 0.0216.**
2. **Two different twins** (0.0128 vs 0.0047) and two different gaps (§3.4's 15× vs App-G's implicit 44× = 0.2066/0.0047) are cross-quoted; CM-1 blends the 15× (minus-the-physics) with the 92% (fit-gap-anatomy) but the draft never says these are different runs.
3. **"92% of the gap" is absolute-MSE, not ratio.** The +γ unit (0.0216) still trails the twin (0.0047) by **4.6× in ratio** even after recovering 92% of the absolute gap. The abstract's "recovers 92% of it [the ≈15× fit gap]" invites the false reading "fits within ~8% of the twin."

**Fix (labels + one caveat, no new runs):** (i) label the §3.4 table CLU "**CLU (γ=0 conservative)**"; (ii) add one sentence to §3.4/App G: "the App-G ladder is a separate wake-only experiment (`fit-gap-anatomy`, seeds 0–2, `9a13455`); its absolute MSE scale and twin (0.0047) are not cross-comparable with the §3.4 `minus-the-physics` table (twin 0.0128, seeds 42–44) — only within-table gaps are meaningful"; (iii) reword the abstract to "recovers 92% of the **absolute** contraction-forbidden fit gap (the unit still trails the twin ~5× by ratio but is bounded by construction)." This is exactly the M4 class the matrix guards, now spanning abstract + §3.4 + App G.

---

## SHOULD-FIX

### SF2-1 — F5 provenance appendix says "all **14** checks" but the verification table has **13** rows (a–m). `f5-note.tex` line 283.
`a,b,c,d,e,f,g,h,i,j,k,l,m` = 13. The appendix is marked "strip on arXiv," so it will not render in the submission build — hence SHOULD not MUST — but if the strip is ever missed, "14 checks / 13 rows" is a free reviewer snag. Reconcile the count or delete the number.

### SF2-2 — Abstract "≈57–69 recurrent-map applications" rounds LEM's 56.4 up to 57. Abstract line 13.
Minor: 56.4 → 56, not 57 (§3.3 table gives 56.4/68.7). coRNN's 5.6 is deliberately excluded (body: "we do not lean on coRNN") — that is fine. Just correct 57→56, or write "≈56–69."

### SF2-3 — Page budget still likely > 5 pp with 4 embedded floats. (C-10 pruning, carried from N-4.)
Main text §1–5 is dense; with Fig 1/2/2b/3 inline it will exceed a 5-pp short. Pruning targets unchanged: §3.1's machine-precision enumerations (App E already partly holds them) and the triple-reporting of emergent lifetimes across §3.3/§3.4/App D. Not a blocker; a pruning-pass note.

---

## NICE

- **N2-1** Abstract "the budget **contains** a recently published ML lifetime law as its overdamped limit" — good, matches the (softer, correct) §3.2 "containment, not conflict." The prior N-1 "subsumes" is gone. Keep.
- **N2-2** §4 line 143 "our §3.5 **reports** a sharp instance … (novelty pending, §4/Appendix B)" — the prior N-3 "contributes" was softened to "reports." Good; no unhedged novelty sentence remains.
- **N2-3** Related-work coverage vs the scout ledgers is complete (Mo, Di Bernardo, Keller, Iqbal-Goldstone, LSTM/LEM/coRNN, Hairer geometric-integration, Hinton/Tieleman/Fischer-Igel/Nijkamp). No missing neighbour.
- **N2-4** The end-note figure list (line 293) references `fig4_emergent`/`fig5_isotropy`/`fig1_erosion_curves`/`fig4_cures` for appendix "Fig 4/5" that are **not** in `figs/` and **not** embedded (no `![]` float). No broken embed results, but if those appendix figures are promised they must be produced before a camera-ready; for submission they can stay as text+table (appendix-grade, per prior N-4).

---

## Reviewer-hat attack pass (register G1–G6 / V* / M* against THIS revision)

- **G1 (unit test on a theory-built testbed):** *Defended.* Verification/evidence labels explicit; §3 opens with the evidence-first reading-order note (SF-4 addressed). The only residual G1 foothold is MF2-B — if the reviewer finds the CLU-MSE scale confusion, "even your own price-of-physics numbers don't reconcile across sections" is the G1-flavoured jab. Close MF2-B and G1 is fully answered.
- **G2 (which component buys what):** *Strongest asset, now airtight* — the twin/broken-volume table + the wired recovery ladder (+γ 92% / γ_φ −24%) is a clean component-attribution story. Only caveat: the ladder's absolute scale must be disclaimed (MF2-B) so the "which component buys what" table isn't read as inconsistent with §3.4.
- **G3 (toy scale):** Scope qualifiers attached throughout (dim 4, ≤5 seeds, laptop, S¹); §5 "solved"→"constitutive problem" closed the one scale-free slip. Compliant.
- **G5 (certificate fine print):** BIBO stated "within the coercive-potential / compact-sublevel-set scope" inline (line 119) with the saddle-blindness caveat in §5 main text (N22) reframed as a neutral class limitation, not an "audit negative." C-6 compliant.
- **G6 (foundational-paper falsifications):** The two live audit surfaces from last pass are both closed — §1 paragraph deleted (MF-1), F5 Cor-3 footnote de-anon clause cut (MF-5). G6 risk now carried by the F5 note as neutral class theorems (Cor. 2, §11), as intended. Clean.
- **M2/M3 (de-anon / salami optics):** Citation architecture C-8-clean (only J&P 2026 + the theory note, third person; no cross-short citations). The audit/de-anon vectors are removed. Program reads as one coherent matrix.
- **Fresh attack — compute-inversion self-harm (SF-2, now in main text):** the honest App-H numbers ("23.5× more wall … 54.8× FLOPs per unit of retention") are quotable *against* the retention contribution. This is the right call (honesty > a fragile 4×), but the writer should ensure §3.3 leads unambiguously with the **qualitative** triad so the surviving claim is the compute-independent one. It does (line 95). Acceptable.
- **Fresh attack — the two MF2 items above** are the new de-facto register entries: headline-number-vs-table drift (5×) and cross-experiment MSE-scale splice (15×/92%). Both are the M4 class.

---

## Missing-experiment list for the Hub (≈empty — both MUST-FIX items are wording/label, not science)

1. **[WORDING]** MF2-A: change "up to 5×" → "≈3.2× (deepest trained δ)" (+optional exact-map/theory-note pointer to ≈5×). No run.
2. **[LABEL + CAVEAT]** MF2-B: label §3.4 CLU as γ=0; add the `minus-the-physics`↔`fit-gap-anatomy` non-comparability caveat; reword abstract "92% of the absolute gap (still ~5× by ratio, bounded)." No run.
3. **[OPTIONAL, from prior SF-3, now largely discharged]** The §3.1 laws on an anchored 3000-ep checkpoint were the prior open candidate — **now DONE** and wired (App B.7 / SF-3: GMOR exact to 1e-12 over 4.6 decades, slope −0.956, floor 27.03, EP slope 0.5165 bit-identical). No action.
4. **[OPTIONAL]** Width-matched CLU (h16 potential) per-step ratio to isolate the "2 grad-evals" cost from the width cost (flagged in `v2-referee-experiments` follow-ups) — would let §3.3 state a width-controlled per-step factor. Not blocking; a fortification candidate only.

No large missing experiment. Everything the draft claims traces to an existing report.

---

## The three sentences a hostile reviewer would quote

1. **(Abstract line 13 / §3.2 line 73, next to Fig 2 caption line 77)** *"…mispredicts by up to 5× past the crossover…"* — "Your headline table and figure bottom at 0.31 (3.2×) at your deepest breaking, and the full 14-δ sweep never goes deeper. On trained models — the entire point of §3.2 — the misprediction never exceeds 3.3×. Where does '5×' come from, and why is it in the abstract?"
2. **(§3.4 table line 113 vs abstract line 13 / App G line 258)** *"…the physics prior costs raw fit — the unconstrained twin fits ≈15× better…"* / *"…a licensed global damping recovers 92% of it…"* — "So the symplectic unit fits 15× worse, but damping recovers 92% of that gap — yet your headline table's CLU sits at 0.190 with no γ label while the recovery ladder's γ=0.05 unit sits at 0.0216 against a different twin (0.0047 vs 0.0128). Which unit, and which twin, is the paper's actual price of physics?"
3. **(§3.3 line 97 / Appendix H)** *"…per unit of retention the unit spends ≈23.5× the wall time of the LSTM it out-holds (≈54.8× FLOPs)…"* — "By the authors' own appendix the retention 'advantage' is a 23–71× compute deficit, so the only surviving claim is a qualitative one — which is fine, but the paper should not have led §3.3 with '≈4× longer' at all."

(The first two are the blocking pair; the third is honest-but-quotable and is best neutralized by keeping §3.3 led firmly by the qualitative triad, which it now is.)

---

## Craft summary
- **p.1 contribution clarity:** strong — five numbered contributions, headline explicitly tagged (§3.2/Fig 2), verification/evidence discipline stated in the contributions block and the §3 reading-order note. Keep.
- **Headline figure:** Fig 2 embedded, PNG present; Fig 2b (predictor-substitution) is a genuine strengthening over the last pass. Good.
- **Related work:** complete vs scout ledgers.
- **Appendix completeness (C-10):** excellent — flag-provenance A.1–A.5 (now includes the referee-experiment additions), erosion (B, with B.7 laws-survive-the-cure), isotropy (C), bias decomposition (D, now with the explicit "distinct probe" disambiguation), EP (E), negatives (F, reframed neutral post-MF-1), loan+ladder (G), per-step compute (H).
- **Page budget:** likely > 5 pp with 4 floats; pruning targets in SF2-3.
- **F5 note:** arXiv-clean — zero todohead, neutral Cor-3 footnote, "All results" count, provenance block flagged strip-on-arXiv. Only residual is the 14-vs-13 count in that strip block (SF2-1).

**Bottom line:** near-final. Land MF2-A and MF2-B (both wording/label, hours not days), run the pruning pass, insert the F5 arXiv id once live — then submit.

---

## Proposed handover updates (for the Hub)
- **Prior punch-list CLOSED:** MF-1 (audit ¶ deleted; App F reframed; N22 neutral), MF-2 (crossing + ladder wired, non-overclaim intact), MF-3 (§3.4↔App D reconciled by attribution; "consistent" removed; +5% seed inside band), MF-4 (Fig 1/2/2b/3 embedded, PNGs on disk), MF-5 (F5 todohead stripped, Cor-3 footnote neutralized, count fixed). SF-1…6 all folded. Verified against sources.
- **TWO new MUST-FIX before submission (both wording/label, no experiments):**
  - **MF2-A ("5×"):** abstract + §3.2 say "up to 5×"; §3.2 table + Fig 2/2b caption + the full 14-δ source sweep bottom at 0.309/0.31 (≈3.2×). "5×" is a mislabel inherited from `v2-full-runs` line 79 and/or a borrow from the F5 exact-map regime (γ=0.1, check k → 0.19). Change to ≈3.2× (optionally + exact-map/theory-note pointer to ≈5×). Not a matrix-canonical number.
  - **MF2-B (15×/92% splice, NEW from the ladder wiring):** §3.4 table CLU (0.190, unlabeled γ; 15× vs twin 0.0128 from `minus-the-physics`) vs abstract/App-G "+γ recovers 92%" (App-G twin 0.0047, +γ unit 0.0216 from `fit-gap-anatomy`). Fix = label §3.4 CLU as γ=0 + one non-comparability caveat between the two experiments + clarify "92% of the **absolute** gap (still ~5× by ratio, bounded)."
- **F5 note:** arXiv-clean; only residual is "14 checks" vs 13 rows (a–m) in the strip-on-arXiv provenance block (SF2-1).
- **Claims-matrix note:** neither "5×" nor a numeric misprediction factor is canonical in CM-4 or the constants table — MF2-A can be fixed freely. Consider adding a canonical form ("misprediction ≈3.2× on trained models to δ=4; exact map ≈5× deeper underdamped") to CM-4 to prevent recurrence.
