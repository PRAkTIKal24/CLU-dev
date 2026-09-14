# pj-minimal-v5 — paper-writer — the HEAD-APPROVED minimal pass on V5 `pj_sub.tex` (figures + exactly 12 approved text items + §-symbol rule + appendix-only bank restorations)

**Commissioned by the Shorts Advisor (charter Addendum 61, 2026-08-22). ⛔ THE PREVIOUS RESTORATION (R3) WAS REJECTED BY THE HEAD AS OVER-SCOPED AND REVERTED — this pass executes EXACTLY the Head-approved line-item list. Anything not enumerated here is FORBIDDEN, however beneficial it looks.** Read `.claude/AGENT_PROTOCOL.md`, then this file. Object: `.claude/NIPSsubmission/v5-palm/pj_sub.tex` (md5 at issue: `6c1902f74ee9611d718cc65b9fd1a031`). Sources: `submission.tex` (same folder, primary ancestor) and `.claude/scratch/pj-restore-R3-preserved/pj_sub_v5_R3.tex` (the R3 bank — already-fitted blocks incl. the transposed non-truncating tables; cite which source each edit used). Report → `.claude/outputs/pj-minimal-v5.md`; `BUILD-NOTE-R4.md` in the folder = deliverable #1.

**DIAL DECLARATION: none — editorial; zero new numbers without ancestors.**

## Head style rule (global, this file)
⛔ **The `§` symbol is never used for section references.** Write `Sec.~\ref{...}` / `Appendix~\ref{...}` instead. Convert the file's existing 4 occurrences (add a `\label` where needed); every added sentence uses the same form. Report the conversion table.

## Part 1 — FIGURES (the Head's instruction: actual figures where needed; banked material to appendices)
- **Main text:** replace the Fig-1 placeholder with `figs/fig1_damping_optimum.png` (caption already in place) and add `figs/fig2_vault.png` at the §3.2 vault result (both referee MUST-restores — "where needed" = the two headline results).
- **Appendices:** replace the two remaining placeholders with their figures (`figB_dlaw`, `figC_lambda_coset`); wire the other 7 (`fig2_two_instruments` · `figB_signflip` · `figB_massive_vs_flat` · `figC2_vault_emergent` · `figC_register_capacity` · `figA1_damping_optimum_full` · `figC_Tstar`) to their base appendix homes, captions verbatim from base. A figure whose claim is not in the paper still ships to its appendix with its base caption (the caption states its result — that is banked information in an appendix, exactly the Head's rule).
- **Fix the two truncated tables** (the base tables at l.192–200 / 226–235): use the R3 bank's transposed versions (proven 0 overfull, all 39 values + zero content change). ⛔ No `\small`.

## Part 2 — the 12 approved text items (word/clause/sentence level; NOTHING else in main text)
1. l.80: correct the Guo sentence to the base's verbatim form (§2 Eq. (1), an ε condition with an unnumbered (ε,δ) relaxation immediately after; Sekhari restored to the cite) — written WITHOUT the § symbol per the style rule (spell it: *"Guo et al.'s (2020) Sec.~2, Eq.~(1)"*).
2. l.83: `(Mo, 2026)` → `(arXiv:2605.03338)`.
3. l.188: *"strictly aligning with"* → *"against"*; l.93: *"strictly minimized at"* → *"minimized at"*.
4. l.291: *"Emprical"* → *"Empirical"*.
5. References: the J&P entry loses *"[Reference redacted for double-blind review.]"* → the full third-person entry (ancestor: `v2-neurreps/submission.tex` l.179, Advisor-ratified Add.60).
6. §3.3: append the clause *"and we claim no certified $(\varepsilon,\delta)$ unlearning"* (base L79).
7. §3.3: add the encoder-exclusion sentence (base L79: *"This is a store-level guarantee only --- the frozen encoder and any residue of past writes in a learned landscape are separate channels"*), verbatim.
8. §3.3 conditions clause: *"attribute-based eviction"* → *"priority/attribute-based eviction"* + append the recency-exclusion clause (base form); abstract: add the scale clause (*"on a designed, non-learned 3-dimensional datastore at capacities 8--64"*, base ancestor).
9. Add the score sentence (base L79 verbatim) in the Limitations/scope block.
10. l.116 area: add *"($\ell_\theta/\Delta<0.05$)"* at the 3.77× site (base L73).
11. Add the 86.97× estimator sentence at the first 107.77× site (base verbatim: the first-passage counterpart + travels-with-its-estimator's-name).
12. Add the probe-floor rider at the 11-orders sentence (base verbatim: *"…eleven orders is one curve on one instrument"*).
⛔ Item struck by consequence of Part 3(a): the "every negative result" softening is NOT applied if the rows restore (the sentence becomes true); if the Head later strikes the rows, soften instead. State which branch executed.

## Part 3 — banked information → APPENDICES ONLY
(a) the 15 missing negatives rows → Appendix E (base verbatim; 20/20 total; ⚠ flag the lifecycle row in the build note — its host claim is not in the paper, Head to keep/cut);
(b) the 2 missing App-C tables (emergent refrigerator; confinement/hop fractions — these carry the confinement control numbers) + the App-B instrument-gap table + the surviving App-B table's mean±sd row → their base homes (R3 bank versions are pre-fitted);
(c) the App-D prior-art paragraph + no-priority clause + fourth Blelloch–Golovin attribution (base L304 block, verbatim) → Appendix D (its base home IS App D — the section is titled "Prior Art");
(d) the R₅₀ differentiator numbers + TTL comparator (1.146→0.752 · 0.75–0.77 · 1.52×) → their base appendix-D home ⛔ NOT main text.
⛔ Banked items whose base home is MAIN TEXT stay out entirely (seam sentence · Titans one-liner · abstract TTL/conditions restorations beyond item 8 · trilemma dial · status headers · anonymization note · fdt-beside-§3.2 · T* claim text · designed-symmetry §3.2 block · all A2 vocabulary fixes not enumerated · all intensifier/garble edits). List them as not-restored in the build note.

## After editing (mandatory)
Rebuild ×2 (`/Library/TeX/texbin/pdflatex`, in-folder; 0 errors / 0 undefined refs / 0 overfull table boxes) · two-way numcheck (every added token → ancestor line; expect an empty orphan list) · sweeps positive-controlled (the fidelity §A.4 zero list incl. `13.9` · "certified" per-occurrence table: denial present, zero affirmative · author-token bibliography-only · honest-scope sentence ×1) · **⭐ THE HEAD'S REPORT ITEM: a table of which appendices were ADDED or CHANGED relative to the Head's rewrite** (expected: A–E changed by figures/tables/rows; none newly created — say so explicitly either way) · final md5 printed · every other folder file byte-untouched · page split reported, not optimized.

## Acceptance criteria
Exactly the enumerated edits + Part-1 figures + Part-3 appendix restorations + § conversion — zero other diffs (classified edit list in the build note); riders verbatim; orphan list empty; 11/11 figures shipped; `submission.tex` untouched; the appendices-changed table present.
