# pj-restore-v2 — paper-writer report

Task + acceptance criterion: **bounded, additive restoration of `NIPSsubmission/v2-neurreps/pj_sub.tex`** — every Part-A worklist item inserted at the site the fidelity report names, every Part-B correction (a)–(j) applied exactly as enumerated, zero edits elsewhere, clean rebuild, two-way numeric check, positive-controlled sweeps.
Status: **done.** ⭐ **Nothing on the worklist was blocked. Every Part-A item and every Part-B correction was executed.**
**DIAL DECLARATION (echoed): none — editorial restoration. Zero new measurements, zero new numbers without an ancestor. No performance claim, so no laundering control applies.**

⚠ **This report contains a short list of items the Head must rule on (§5) and one item with no ancestor, flagged rather than naturalised (§4, dev. 5).** It contains **no** downstream reconciliation list of its own; the one reconciliation still outstanding (the `−0.956`/`−0.961` statistic) is *labelled* in the paper by this pass and still needs an owner.

**Deliverable #1 is `NIPSsubmission/v2-neurreps/BUILD-NOTE-R3.md`** — the full edit map (every edit → worklist item → base/output ancestor line), the Part-B before→after table, the two-way numeric check, the sweep table, and the deviation list. This report is the summary.

---

## 1. Headline, in six lines

1. **`pj_sub.tex` md5 `d15de78712d90eb94d2495d4bd9ad948` → `396a00d0c41285a471fddc8567ac256d`.** 395 → 438 lines. **Build clean: 0 errors, 0 undefined references, 14 pp → 17 pp.** The one overfull hbox is *pre-existing* — proven by rebuilding the pre-pass file in scratch (identical 11.27979 pt box, same table). **No new bad boxes.**
2. **True page split (same instrument both prior BUILD-NOTEs used): main text 7.50 pp · references 1.93 pp · appendices A–G 7.57 pp = 17.00 pp.** (Was ≈6.6 / ≈2.2 / ≈5.2 = 14.)
3. **All ten Part-B corrections applied verbatim as enumerated**, each printed before→after in the build note. The two highest-consequence: the abstract's necessity claim is gone (**R2-1**) and the base's own *"the theorem states sufficiency; equivariance is not necessary"* fence is back at the claim site; and §G.1 is **un-fused** into the base's two separate facts with the quality-factor condition and the dynamically-silent clause restored (**S-1**).
4. **The two mandatory sentences that were absent across both fidelity rounds are in:** the **§A20.5 substrate-scope sentence** (P-1, verbatim, new Discussion bullet) and the **score sentence in its measured form** (P-2: *"No external benchmark is won on its own headline metric anywhere in this paper"* + the honest-gap clause + the task-RMSE-axis clause). The Head's own two sentences at that site were **kept**, not replaced.
5. **Numbers: two-way check passes.** 79 numeric tokens increased in count; **77 have a `submission.tex` ancestor and the other 2 are exactly the two the task authorises from named output files** (`0.973` pooled corr → `v2-full-runs.md:81`; `0.82` coRNN RMSE → `philosophy-synthesis.md:628`). **Exactly one token decreased — `2025`, the deleted authorless poster entry.** No digit, precision, unit, exponent or ± was altered anywhere.
6. **Sweeps clean and positive-controlled** (never-quote: 1 hit = the same inherited `2.6` false positive, control 372; author-token: 2 hits = the permitted bibliography entry + the *Morse* survival trap, 0 in body/captions/labels/filenames; `pseudo-gap` = 0; semantic hermeticity = 0, control 132). **Figures: 5/5 PNGs used, 0 added, 0 unused.** **`submission.tex` and every other folder file byte-identical**; only `pj_sub.{tex,pdf,aux,log}` changed.

## 2. What is now in the paper that was not (grouped by what a reviewer would have hit)

- **Claim-strength repairs at the claim site.** Sufficiency-not-necessity (base 86) + the measured non-equivariant battery (base 103(b)); "guarantees **at least**"; "**most** infinitesimal perturbations"; the Mo boundary in the referee's stricter form at **both** sites (abstract: *"no closed-form constitutive law linking the transverse curvature of a trained potential to lifetime has been measured"*; §2: *"these estimator-based (kinematic) lifetime predictions and existence proofs"*) — which removes the paper's single most quotable inconsistency (referee hostile quote #2).
- **The fences that were carrying nothing.** CM-17's sampler-not-thermodynamics paragraph restored **in full**, including *"We therefore never assert that a relativistic unit 'has no equilibrium'"* and the `newtonian_learned` no-touch scope. The `legacy`-default warning + both flag names, now that the file carries a live finite-$T$ number (`3.77±0.23×`). The single-representative-checkpoint disclosure under the zero-drift claim. The N46 designed-only rider **with** the Vafidis counterexample, at the negative's own site. The anchor's non-novelty clause at §4.4. Appendix G's demotion label, precision fine print (*"do not quote 2.2×10⁻¹⁶"*), the δ=0.3 no-NLO fence, and G.6 honest scope. `x` is no longer an orphan symbol.
- **The six missing negatives rows** (task-RMSE axis, N51 exponents + its reading rule, friction×governor composition, friction-cannot-stabilize-a-saddle, mean-spectrum chaos, `sleep_temperature` silent knob) plus **all six within-row dropped numbers** and **both reading rules** (never a "drift rate"; the breaking coefficient is not $\varepsilon$).
- **Referee wiring.** Both slopes labelled with their fit spec at **both** sites (MF-1 — neither picked); `0.9987` scoped overdamped-only with `0.973` pooled (MF-4 — which *strengthens* the paper's own regime-structure point); the 4.5-vs-4.6-decade grids distinguished; the 5-seed/3-seed experiment attribution; the sleep flags (f5/s500/CD); `≈35×` wake-MSE restored; the floor ripple (±8 steps at δ=4); the coRNN honest-weak footnote; "70" unified.
- **Reproducibility.** The width-match confound and the scan-amortized timing protocol in App D; the head-to-head's finite-horizon robustness check in §4.2 **and** its numbers in App F (`corr=0.9995`, `0.86–1.03`, `0.30` vs exact-gap `0.31`) — the answer to the referee's third hostile quote now exists in writing.
- **Citations.** Every one of the six previously-decorative entries now has an in-text use; the unverified authorless poster is softened to uncited prose and its entry deleted (**References 51 → 50, still 0 new records, 0 dangling**); `(preprint; single author)` restored on arXiv:2605.03338.

## 3. Referee-finding disposition (r2 report)

| grade | disposition |
|---|---|
| **MUST-FIX** | MF-1 ✅ (both labelled) · MF-2 ✅ · MF-3 ✅ (softened + entry dropped) · MF-4 ✅ · MF-5 ✅ (both sites, referee's exact forms) — **5/5 closed** |
| **SHOULD-FIX** | SF-1 ✅ · SF-2 ✅ · SF-3 ✅ · SF-4 ✅ · SF-5 ✅ (⚠ ancestor-less prose, §4 dev. 5) · SF-6 ✅ · **SF-7 ⛔ not on the worklist — left as the Head wrote it (§5)** · SF-8 ✅ · SF-9 — not on the worklist; ⚠ **partially mitigated for free**: the base's own "≈5× is exact-map only" separation is *not* in `pj_sub`, and line 108 still reads "On exact analytic maps, this divergence extends to ≈5×", which is the correct scope-carrying form (CM-4's canonical rule). No action taken. |
| **NICE** | N-1 not on the worklist (Mo still cited by arXiv number in running text — deliberate, per the author-token rule) · N-2 ✅ · N-3 ✅ · N-4 not on the worklist (figure work excluded, A7) · N-5 not on the worklist (style-file string) · N-6 ✅ · N-7 ✅ (subsumed by P-2) |

## 4. Deviations — the honest list (full text in the build note §6)

1. **B(f)** is a *form*-restoration, not a verbatim base copy: `pj_sub`'s nouns kept, the base's assertion structure restored (measured, not "proven mathematically") + the causal-cap clause added.
2. **R2-7 demotion label** drops the base's two words "of this abstract" (register, consistent with B(c) "chapter"→"paper").
3. **G.5 restores the fence, not the claim.** Base F.5 carries both the definition of $x$ (needed to de-orphan the caption) and a resonance-saturation claim; only the definition + the δ=0.3 breakdown fence were imported. Consequently base F.6's one clause fencing *that* claim is also omitted — it would fence something not in the paper. Everything else in F.6 is in.
4. **MF-1 labels both slopes** per instruction; picking a canonical statistic is still the Hub's.
5. ⚠ **Appendix A prose has no ancestor.** Base's App A is figure-only; SF-5 asks for pointing prose. Written for this pass in `pj_sub`'s register; asserts **no number and no claim**, only cross-references and the already-stated λ=100 / 3000-epoch / 3-seed configuration. **Flagged, not naturalised** — if the Advisor prefers zero ancestor-less prose, delete lines 253 and the item reverts to an honest stub.
6. Punctuation-only comma at the retention caption so the verbatim rider attaches; em-dashes follow `pj_sub`'s unspaced `---` convention (base uses ` --- `), text otherwise diff-identical.
7. **Not restored per the task's own instruction:** DOIs and the other stripped status annotations. Only the claim-bearing one was restored.

## 5. Open editorial questions for the Advisor / Head (each is one word or one line)

1. **SF-7 / claims-table #24 — the one surviving claim widening.** §1 line 35 still reads *"Our results hold **generally** for the class of damped symplectic recurrences."* The base has no "generally"; fidelity graded it *WIDER (trivial)*, the referee graded it SHOULD-FIX (a C-5 scope-free generalization, and the only one in the file). **It was not on my worklist so I did not touch it.** One word deletes it, or the referee's reword — *"The laws are derived for the class of damped symplectic recurrences; we verify them on one trained instance"* — replaces it. **Rule requested.**
2. **The bibliography annotations and 28 DOIs.** Deferred by the task to a one-word Head option. Say the word; it is one pass.
3. **App C (`app:retention`) still has no lead-in prose.** Base 263's one-liner (*"every number in it is stated there and none is added here"*) is a clean verbatim restoration if wanted; it was not on the worklist.
4. **Per-figure `\emph{Verification.}` / `\emph{Evidence.}` caption tags** remain unrestored (not on the worklist). The two-layer scheme survives in its §1 form at line 50 and is applied in prose at lines 94 and 415.
5. **The `−0.956` / `−0.961` reconciliation still needs an owner.** This pass makes the paper *self-consistent* — both numbers now carry their fit spec at both sites, so no reviewer can find a contradiction — but it does not pin a canonical statistic in the matrix. Until one is pinned, every future artifact quoting "the anchored overdamped slope" reopens it.
6. **Page discipline.** 17 pp against the EA track's 4. Deferred by Head ruling; recorded here so the number is on the table when the pruning pass is scoped. The natural migrations are unchanged from the referee's read (the §2 delineation paragraphs, the §3 bullet blocks, §4.3's second half) — and note that **appendix maximalism (C-10) is now fully satisfied**, so the pruning pass has a complete bank to cut from.

## Git footprint
**None.** No tracked file touched; all writes are under gitignored `.claude/` — `NIPSsubmission/v2-neurreps/{pj_sub.tex, pj_sub.pdf, pj_sub.aux, pj_sub.log, BUILD-NOTE-R3.md}`, this report, and scratch under `.claude/scratch/pj-restore-v2/` (pre-pass backup `pj_sub.tex.BEFORE`, `md5-BEFORE.txt`, the five idempotent edit scripts `edit1..5.py`, and `beforebuild/` — the pre-pass rebuild used to prove the overfull box is inherited).

## Proposed handover updates (for the Advisor / Hub)

1. **`pj-restore-v2` = DONE.** All Part-A items restored, all Part-B corrections applied, build clean at **17 pp (7.50 main / 1.93 refs / 7.57 appendix)**, two-way numeric check passes with exactly two output-file-ancestored additions (`0.973`, `0.82`), sweeps clean and positive-controlled, `submission.tex` and every other folder file byte-untouched. **`pj_sub.tex` md5 = `396a00d0c41285a471fddc8567ac256d`** — the Advisor's post-pass verification is one command.
2. **The fidelity r2 reconciliation list is discharged.** R2-1…R2-12 and P-1…P-9 all executed; **S-1…S-7 all executed.** The two items that survived both fidelity rounds unfixed (P-1 substrate scope, P-2 score sentence) are in, verbatim.
3. **The referee r2 slate: 5/5 MUST closed, 8/9 SHOULD closed** (SF-7 deliberately untouched, §5 item 1; SF-9 not on the worklist and already scope-carrying), **3/7 NICE closed** (the other four were figure/style items excluded by A7 or the author-token rule).
4. **One Head ruling is owed before this artifact is final: SF-7** (§5 item 1). It is the only claim-strength item in the file that a fidelity/referee pass has flagged and no pass has been authorised to touch.
5. **Two registry actions remain surfaced and unowned** (both predate `pj_sub.tex`, both re-raised by fidelity r1 and r2): the **N46 per-seed discrepancy** (`negative_results.md` vs `submission.tex:132`) and the **anchored-slope statistic**. This pass labels the second inside the paper; neither is resolved.
6. **For any future referee pass:** the four true positives the last round predicted (abstract equivariance overstatement, no-benchmark statement, designed-curve seed status, instrument substitution) are **all closed**. A round-3 referee reporting any of them would be reading a stale PDF.
