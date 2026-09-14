# v2-style-pass — paper-writer (V2's FINAL pass: restructure + appendix cut + figure promotion + brevity + de-bold; ≤12 pp TOTAL)

**Commissioned by the Shorts Advisor (charter: `.claude/advisor-head-shorts-charter.md`, Addendum 32; Head directives 2026-08-19 — supersedes the earlier v2-style-pass scope).** Read `.claude/AGENT_PROTOCOL.md`, then this file. You edit `papers/v2-short/submission/submission.tex` (+ rebuild the PDF, update `BUILD-NOTE.md` §9). ⛔ Canonical untouched; supplementary untouched.

**DIAL DECLARATION: none — editorial pass; zero content, number or claim changes.**

## The Head's six instructions (binding, in this order of execution)

**1 — Restructure: related work moves directly after the introduction.** New order: §1 Introduction → §2 Related work → §3 Setup/two-axes (fine print intact) → §4 Results → §5 Discussion. Renumber and re-`\ref` everything. Fold the retirements appendix (currently K) INTO the new §2 as compressed sentences — the four retirements are related-work content; the appendix dissolves. ⛔ The approved narrow-claim wording carries verbatim.

**2 — Appendix cut: ONLY sections presenting plots or results tables survive; drop the rest from the submission.** Applying the criterion to the current set:
- **KEEP (has figure/table):** erosion study (Figure 2 — but see instruction 3) · kick-amplitude probe (table) · negatives (the 11-row table + its two mandatory riders) · loan curve/recovery ladder (tables) · per-step compute (tables) · GMOR (table + Figure 3) · the T>0 diffusion section IF its content is table-borne (writer judges by the criterion; the mandatory FDT flag box travels with whatever survives that cites it).
- **DROP from the submission (no visuals):** the SO(2) primer (⚠ the Add.11 exception is OVERRIDDEN by the Head's tightened rule, flagged in the Advisor's report — it remains in the canonical; the camera-ready may restore it) · isotropization/charge oscillation · exceptional-point signatures (if no table; its damping-corollary sentence may fold into the results prose if a rider requires a home) · anything else prose-only.
- Every drop mapped in BUILD-NOTE §9 to its canonical home. ⛔ If a dropped appendix carried a MANDATORY rider that qualifies a main-text claim (e.g. the sampler kinetic-mode scope), the rider relocates to the claim it qualifies — riders never drop with their appendix.

**3 — Figure promotion: move the retention plot (Figure 2, the anchored-laws/retention figure) into the main text.** Main text then carries two figures (the head-to-head headline + the retention plot). Place it with the result it illustrates.

**4 — Brevity pass** (the Add.30 style directive, full force): rewrite the main text to `.claude/PJ_Writing_Style_Context.md` — ABT openings (abstract, §1, each results subsection) · macro-to-micro flow · zero weasel words · succinct, syntactically varied prose · signposting where the reader needs it · "we" for actions, passive for established facts · **simple basic technical terms; never complicate for no reason**. The main-text page budget stays ≤4 pp WITH both figures — brevity pays for the promoted figure. If short after full compression, the measured menu fires in order (§budget-verification demotes to the GMOR appendix → related-work ¶1 compresses → limitations becomes a scope box), stopping at budget.

**5 — De-bold:** remove ALL bold in the main text except natural structural formatting (section/paragraph headers the class produces). The style file's "bold for new-concept definitions" yields to the Head's explicit instruction — definitions carry italics or plain prose instead. Appendix table headers may keep structural bold.

**6 — The total: TARGET 8–9 pages, everything included** (main + references + appendices); 12 is a hard ceiling, not the target. Budget arithmetic: main ≤4 + references ~1.5 ⇒ **the appendix block gets ~3–3.5 pp**. Consequence for instruction 2: survivors are cut to the bone — each surviving appendix is its figure/table + the result sentence + mandatory fine print, single-paragraph prose maximum; merge related survivors under one heading where natural (e.g. one "Supplementary results" appendix with subsections); if the plots/tables set still overflows ~3.5 pp, the least load-bearing tables move to canonical-only and are listed in the drop map. Report the final split (main/refs/appendix pages) in BUILD-NOTE §9.

## Boundaries (absolute, unchanged)
1. ⛔ Approved wordings, mandatory riders, scope qualifiers, fine print: VERBATIM — repositionable, never paraphrased. Protected sets per BUILD-NOTE: the fine print (a)–(c) · the FDT flag box · the negatives-table riders · the narrow-claim wording · the non-comparability caveats · the continuity sentence.
2. ⛔ Zero number changes; zero dropped findings (the negatives table is the findings' home — it stays complete). The two-way numeric-token check re-run and printed. ⚠ Numbers whose only home was a DROPPED appendix: list them in BUILD-NOTE §9 as submission-absent-canonical-present (that is allowed — the canonical is the archive) — but any number CITED from the main text must still resolve inside the submission.
3. ⛔ §results head-to-head + Figure 1 untouchable in content; the final sweep (BUILD-NOTE §8 pattern set) re-run and printed.

## Acceptance criteria
1. Total in the 8–9 pp target band (≤12 hard); main ≤ 4 pp with both figures; related work is §2; page split printed.
2. Zero `\textbf` in main text outside structural headers (grep printed).
3. ABT spot-checkable on abstract/§1/results subsections; the numeric two-way check and final sweep pass and are printed.
4. BUILD-NOTE §9: per-section word deltas · the appendix drop map · the rider-relocation list · menu items fired.
