# v2-style-pass — paper-writer report

Task + acceptance criterion: execute the Head's six 2026-08-19 instructions on `papers/v2-short/submission/submission.tex` (restructure → related work as §2 · appendix cut to plot/table sections · retention plot promoted to main text · Add.30 brevity pass · de-bold main text · total 8–9 pp, ≤12 hard), zero content/number changes.
Status: **done, with two acceptance criteria measurably MISSED and costed** (see "Findings" §A and the menu in §F).

**DIAL DECLARATION (echoed): none — editorial pass; zero content, number or claim changes.** Laundering control: n/a. Falsifies: n/a. Does not falsify: n/a.

⚠ **DOWNSTREAM RECONCILIATION LIST IN THIS REPORT (needs a Hub owner):** (1) the residual page gap — main 5.72 pp vs ≤4, total 11 pp vs 8–9 — is closable only by items that contradict this pass's own instructions or Charter C-6; the costed menu is §F and BUILD-NOTE §9.7, and it is a **Head decision**. (2) The colleague's $SO(2)$ primer is now **out of the submission artifact** (canonical retains it) — **the camera-ready acknowledgment obligation is unchanged and still owed**, and the colleague's S1 sign-off gate is now a canonical-only gate for this artifact.

## What I did

**Instruction 1 — restructure + retirements dissolved.** New order §1 Introduction → **§2 Related work** → §3 Setup and the two axes (fine print intact) → §4 Results → §5 Discussion; every label/`\ref` rewired (0 undefined refs). The retirements appendix (r8 App. K) **no longer exists**: K.1–K.4 are a compressed `\footnotesize` run-in paragraph in §2 ("Four claims we retire, so that a reviewer's first objections are answered in writing"), K.5's bounding prior art folded into §2 ¶2 (Ramsauer/modern-Hopfield sentence joins the HiPPO / Kong / NTM-DNC / EDEN / UnICORNN / Titans list). The CM-21 approved narrow-claim wording carries **verbatim** in §2 ¶2.

**Instruction 2 — appendix cut, plot/table criterion.** Survivors merged into three headings: **A Supplementary results** (A.1 loan curve + recovery ladder, A.2 per-step compute) · **B Prominent negatives** · **C The mode-mass budget verified, and GMOR proper**. Dropped to canonical-only: $SO(2)$ primer, erosion study, isotropization, exceptional-point signatures, $T>0$ coset diffusion, loan-appendix G.3 (reach rungs, prose-only) and — under instruction 6's overflow rule — the kick-amplitude probe. Full drop map + every submission-absent/canonical-present number: **BUILD-NOTE §9.4**.

**Riders relocated, never dropped** (full table in BUILD-NOTE §9.5): FDT/kinetic-mode flag box → §3 beside fine print (a) (verbatim bar three repositioning words, listed); chain-length scope clause → §4.3 verbatim; charge-oscillation reporting caution → Appendix B under the isotropization row; damping-corollary consequences (i)/(ii) → Appendix C.2; kick-probe anharmonicity result inlined in the §5 scope box so its pointer does not dangle. Fine print (a)–(c), negatives reading rules, sampler kinetic-mode scope, non-comparability caveat, GMOR precision fine print, width-matching confound, narrow-claim wording, continuity sentence: **verbatim, unmoved**.

**Instruction 3 — figure promotion.** `fig2_anchor_cure_laws.png` is main-text **Figure 2**, placed with §4.3's anchor-cure result; caption verbatim from the r8 appendix with one cross-reference edit. Main text now carries Figure 1 (head-to-head, C-3 headline, **content untouched**) and Figure 2, both at `0.9\linewidth`.

**Instructions 4+5 — brevity and de-bold.** ABT openings on the abstract, §1 and each results subsection; contributions enumerated on page 1; macro-to-micro flow kept; `\textbf` in main text **= 0** (nine `\paragraph{}` structural run-in headers remain, which the instruction exempts; definitions now carry italics or plain prose). Appendix table headers keep structural bold. **The whole measured menu fired**: item 1 (§budget verification demoted to the GMOR appendix, −261 w, now Appendix C.1/C.2 — a 6-number pointer sentence keeps the headline figures in main text), item 2 (related-work ¶1 compressed), item 3 (limitations → scope box).

**Instruction 6 — the total.** 20 pp → **11 pp**.

## How I verified (commands + observed output)

- Build: `pdflatex -interaction=nonstopmode submission.tex` ×3, TeX Live 2026 → `Output written on submission.pdf (11 pages, 860384 bytes)`. **0 errors · 0 undefined references · 0 overfull boxes · 8 underfull hboxes**, all inside the `\scriptsize` narrow `p{}` cells of the negatives table. Ragged-right column specs clear all 8 — **measured: at the cost of +1 page (11→12)** — so justification was kept and the 8 are reported, not silenced.
- Page split measured from **PDF word bounding boxes** (`pdftotext -bbox`), text block top 72 pt / bottom 720 pt, page 792 pt — not from character counts.
- De-bold grep: `\textbf` in `\maketitle`…`\section*{References}` = **0**; `\paragraph{` = 9; appendix `\textbf` = 49 (permitted).
- Numeric two-way check, printed below.
- Final sweep (§8 pattern set, per-file, positive-controlled), printed below.
- `pdfinfo`: Title/Subject/Keywords/Author **all empty**, Creator `LaTeX with hyperref`, Producer `pdfTeX-1.40.29`.
- Canonical untouched: `../draft.md` and `../draft.tex` mtimes unchanged (Aug 18 23:39); `supplementary-theory-note.{tex,pdf}` unchanged (Aug 19 01:37).

## Findings / results

### A. The measured page split (acceptance criterion 1)

| block | r8 | **r9** | verdict |
|---|---|---|---|
| main text | 5.42 pp (1 figure) | **5.72 pp** (2 figures) | ❌ vs ≤4 pp |
| references | 1.06 pp, 33 entries | **1.02 pp**, 28 entries | — |
| appendices | ≈12 pp | **4.23 pp** | ⚠ vs the ~3–3.5 pp budget |
| **TOTAL** | **20 pp** | **11 pp** | ✅ vs the 12-pp hard ceiling; ❌ vs the 8–9 pp target |

Related work **is** §2 (✅ criterion 1, third clause). Page split printed (✅ criterion 1, fourth clause).

### B. Why the main text did not shrink — the arithmetic, not an excuse

The main text is **0.30 pp longer** than r8 despite a full brevity pass, because the pass was ordered to *add* to it:
`+0.27` promoted Figure 2 + caption · `+0.36` retirements paragraph (instruction 1) · `+0.13` FDT flag box (rider relocation) · `+0.09` chain-length scope clause (rider relocation) · `+0.04` inlined kick-probe result and the Ramsauer sentence — against `−0.39` menu item 1 · `−0.16` from setting the four protected boxes and the retirements paragraph in `\footnotesize` · `−0.04` of prose compression.

**The prose compression is genuinely small because the main text is not fat.** Of its 3,549 words: ≈1,300 are protected-verbatim blocks (fine print (a)–(c), FDT box, scope box, narrow claim, scope clause, the untouchable head-to-head), ≈520 are riders this pass relocated *into* it, ≈175 are figure captions; the remainder is number- or citation-bearing results prose. r8 had already condensed twice by relocation. **I did not fabricate compression that the text does not contain.**

### C. De-bold (acceptance criterion 2) — printed

```
\textbf in MAIN TEXT = 0        \paragraph{} structural headers in main = 9
\textbf in appendix  = 49       (appendix table headers may keep structural bold)
```

### D. Numeric two-way check (acceptance criterion 3) — printed

```
(i)  numeric tokens in submission.tex NOT in the canonical:  2  -> ['1.52','2.06']
     both are p{} column widths in the negatives table (typographic, not content)
(ii) numeric tokens of canonical MAIN TEXT absent from submission MAIN TEXT:  8
     ['3.3','3.4','3.5']  canonical section numbers, now rendered by \ref
     ['22','46','149']    registry tokens, stripped per BUILD-NOTE §5
     ['6','2.7']          present ELSEWHERE in the submission (they travelled with the
                          demoted verification block into Appendix C)
     -> NO content number left the submission via the main text.
total distinct numeric tokens in submission.tex: 330
```

### E. Final sweep (acceptance criterion 3) — printed

```
ZERO-HIT SET: ALL CLEAR  (commit · agent/ · chlu/ · .claude · tectonic · draft.md · draft.tex ·
Registry/registry · provenance · Appendix M · CM- · SF- · MF- · [WORKING TITLE ·
[AUTHORS PLACEHOLDER] · <!-- · CLU-former · certified · unlearning · exact deletion ·
"the item is gone" · "exact discrete FDT" · "samples Gibbs" · 0.384 · 16.28 · CAFE · C-MAPSS ·
N-CMAPSS · HEPA · CAMELS · bpc · S_eff · z_hole · 0.99985 · 54.56 · 306.76 · 300.09 · deltanet ·
ttt_mlp · MUNKEY · 0.4545 · 13.9 · memory vault · 107.77 · compositional · unaskable · Guo ·
Ginart · Sekhari · Track A · waitlist · paid-access · companion · sibling · "our other" ·
"this program" · "the program" · experiment-engineer · "per the Head" · wormhole)
N<digits> registry tokens: 0

HITS, context-checked, compliant:
  CHLU x2            main x1 (the sanctioned continuity sentence) + the reference entry
  "energy units" x1  the FDT flag box's refusal statement, now in §3
  2.6 x4             all grid/probe values, ALL in the appendix, never the retracted number
  critical-damping x1, V-shape x1   BOTH after \appendix; ZERO in the abstract, §1, §2 or §3
  V-curve x0

POSITIVE CONTROLS FIRED: GMOR 13 · "introduced as CHLU" 1 · Rusch 6 · verification 6 ·
evidence 10 · Anonymous 2 · 28 reference entries
```

### F. The residual gap — costed menu (nothing on it taken unilaterally)

Every item contradicts one of this pass's own binding instructions or Charter C-6, which is why I stopped:

| item | measured/estimated saving | what it costs |
|---|---|---|
| Drop Figure 2 from main text | ≈0.27 pp | contradicts instruction 3 |
| Retirements back to an appendix | ≈0.36 pp | contradicts instruction 1; CM-21 currently discharged in main text |
| §5 scope box → appendix | ≈0.35 pp | contradicts C-6 |
| Fine print (a)–(c) + FDT box out of §3 | ≈0.48 pp | contradicts C-6 and the rider rule |
| Drop the per-step-compute appendix (A.2) | ≈0.60 pp | contradicts instruction 2's explicit KEEP (the ratios survive in §4.3) |
| Drop the recovery-ladder table | ≈0.20 pp | contradicts instruction 2's KEEP |
| Demote §4.2's price paragraph | ≈0.30 pp | removes evidence-grade result from main text (C-2/C-3) |

### G. Charter compliance notes

- **C-1 (as reversed):** no audit-confession paragraph anywhere; the continuity sentence cites J&P 2026 for the primitive's introduction only. ✅
- **C-2:** verification/evidence labelling survives the restructure — the §4 opener labels the demoted block "verification, reported in Appendix C"; §4.1 "Evidence: …"; §4.3 "Evidence."; Appendix C's preamble "therefore verification of the theory's exactness". ✅
- **C-3:** ML-first order preserved and strengthened — main-text Results are now *all* trained/learned-system evidence, with designed-testbed verification in Appendix C. ✅
- **C-5:** scale qualifiers in-sentence throughout (abstract "dim 4 … ≤5 seeds … laptop CPU"; scope box (i)). ✅
- **C-6:** fine print (a)–(c), the FDT flag box and the GMOR precision fine print all sit next to the claims they qualify. ✅
- **C-9:** the negatives table is complete (11 rows) with all riders. ✅
- **C-10 (appendix maximalism):** deliberately overridden for this **derived** artifact by the Head's 2026-08-19 ruling; the canonical `../draft.{md,tex}` remains the maximal archive and is byte-untouched. Flagged here because it is a Charter-level deviation with a dated authority.
- **M1 hermeticity:** no other short referenced; theory note cited as "Anonymous (2026), provided in the supplementary material". ✅

## Git footprint

None — all edits are inside gitignored `.claude/papers/v2-short/submission/`. No tracked file touched, no branch created.

Files changed: `submission/submission.tex` (rewritten), `submission/submission.pdf` (rebuilt, 11 pp), `submission/BUILD-NOTE.md` (§1 page row + §9 added; §3 and §4 marked HISTORICAL and pointed at §9), `../CHANGELOG.md` (one new top line).
Files verified untouched: `../draft.md`, `../draft.tex`, `submission/supplementary-theory-note.{tex,pdf}`, `submission/figs/*`, `submission/neurips_2025_ml4ps.sty`.

## Open editorial questions for the Hub/Head

1. **The page verdict.** Main 5.72 pp / total 11 pp. Which items of §F fire? My reading of the instruction set is that the *total* was the Head's primary target and the ≤4 pp main was the instrument; if so, the cheapest total-page item is dropping the per-step-compute appendix (≈0.60 pp), but that is the honest-gap receipt and I would not do it without an explicit ruling.
2. **The primer drop.** The colleague's $SO(2)$ primer is out of the submission (canonical retains it, camera-ready may restore). The **camera-ready acknowledgment obligation stands** and the colleague's S1 sign-off gate now applies to the canonical only. Confirm the Head accepts this as the default disposition (the Advisor flagged drop-as-default; I executed it).
3. **The 8 underfull hboxes** in the negatives table: keep justified at 11 pp (current), or ragged-right at 12 pp? I chose 11 pp.
4. **Appendix C's title** now reads "The mode-mass budget verified, and GMOR proper on trained checkpoints" because menu item 1 moved the budget verification there. If the Head restores §budget to the main text, that title reverts.
5. **Reference pruning.** 5 entries were dropped as orphaned by the appendix cut (Agoritsas 2023 · Bhatt 2016 · Decelle 2021 · Fischer & Igel **2011** · Toledo-Marín 2025). If any appendix is restored, its citations must come back with it — the mapping is in BUILD-NOTE §9.4.
6. **Template caveat unchanged:** the page count is measured in the NeurIPS-2025-ML4PS geometry, the only NeurIPS-family style file obtainable on this machine. It must be re-measured in the real venue template before submission.

## Proposed handover updates (for the Hub)

- V2 submission artifact is at **r9**: 11 pp total (main 5.72 · refs 1.02 · appendices 4.23), related work is §2, retirements folded into §2, six appendices dropped to canonical-only, two figures in main text, zero `\textbf` in main text, all checks re-run and printed in BUILD-NOTE §9.8.
- **V2's remaining gates:** (a) the Head's ruling on BUILD-NOTE §9.7 (the residual page gap); (b) the colleague's S1 sign-off (now canonical-only) + the camera-ready acknowledgment obligation; (c) the true-template page re-measure.
- Sources used for positioning prose: none new — §2's related-work prose descends from the r8 submission §4 and the dissolved Appendix K (whose own source was `scout-dynamical-memory-priorart` §§1–7 via claims-matrix **CM-21**), reused verbatim/compressed, not rewritten from a scout report.
