# pj-minimal-v2 — paper-writer report

Task + acceptance criterion: **execute EXACTLY the Head-approved minimal list on `NIPSsubmission/v2-neurreps/pj_sub.tex`** — the 12 main-text items (+ item 13), the `§`→`Sec.~\ref` style rule, the Part-2 appendix-ONLY bank restorations, zero other diffs, clean ×2 rebuild, two-way numcheck, positive-controlled sweeps, `BUILD-NOTE-R4.md` as deliverable #1.
Status: **done.** ⭐ **Every enumerated item executed; nothing blocked; zero unenumerated diffs (30/30 changed line-blocks classified); orphan list EMPTY.**
**DIAL DECLARATION (echoed): none — editorial. Zero new numbers without an ancestor; no performance claim, so no laundering control applies.**

⚠ **No downstream reconciliation list of my own.** Two pre-existing ones remain unowned and are *labelled but not resolved* by this pass: the `−0.956`/`−0.961` anchored-slope statistic, and the N46 per-seed discrepancy. **§6 lists 4 items needing a Head/Hub ruling.**

**Deliverable #1 is `NIPSsubmission/v2-neurreps/BUILD-NOTE-R4.md`** — §-conversion table, the per-item edit map with per-edit source attribution, the appendices ADDED/CHANGED table (the Head's report item, §4 there), the 30-row classified edit list, the two-way numcheck, the sweep table, the page split, and the NOT-RESTORED list. This report is the summary.

---

## 1. Headline, in seven lines

1. **`pj_sub.tex` md5 `d15de78712d90eb94d2495d4bd9ad948` → `a5758ad3eafcaf8971c73e7685d21450`.** 395 → 431 lines. **Build ×2: 0 errors, 0 undefined refs, 0 multiply-defined labels. 14 pp → 15 pp.**
2. **Zero unenumerated diffs.** A `difflib` opcode diff gives **30 changed line-blocks; every one maps to an enumerated item** (table in build note §5). No sentence of the Head's rewrite was reworded, reordered or deleted outside them.
3. **Numbers: 59 distinct numeric tokens increased in count; 59/59 have a `submission.tex` ancestor line. ORPHAN LIST EMPTY — zero output-file ancestors needed** (R3 needed two, `0.973` and `0.82`; the clauses that carried them are not on the approved list, so they are absent here). Exactly one token decreased: `2025` (−3), the deleted authorless poster.
4. **`§` retired.** 11 pre-existing `\S\ref` occurrences (on 9 lines — lines 46 and 58 carry two each, which explains the task's "9") + 2 arriving with restored blocks = **13 conversions**; post-pass the file has **0** `\S\ref`, **13** `Sec.~\ref`. **0 `\label` commands added** — every referenced section already had one; appendix refs were already `Appendix~\ref{}`.
5. **Part 2 landed in appendices only.** ⛔ The main-text §4.2 finite-horizon sentence stayed OUT, as instructed; the robustness numbers live in Appendix F alone.
6. **Sweeps clean and positive-controlled** (never-quote: 1 hit = the same inherited `2.6\b` false positive at the base-ancestored `h-h^*=2.6×10⁻⁶`, control 350; author-token: 2 hits = the permitted bibliography entry + the *Morse* survival trap, 0 in body/captions/labels/filenames; `pseudo-gap` = 0; semantic hermeticity = 0, control 112).
7. **Figures 5/5** PNGs used, 0 added, 0 unused (Part 3 confirmed). **`submission.tex` and every other folder file byte-identical**; only `pj_sub.{tex,pdf,aux,log,out}` + the new `BUILD-NOTE-R4.md` changed.

## 2. ⭐ The Head's report item — appendices ADDED or CHANGED

**ADDED: none.** No `\section` was created; the file has the same seven appendices A–G, same titles, same labels, same order, before and after.

| appendix | changed? | by what |
|---|---|---|
| **A** `app:anchor` | CHANGED | item 12b (the −0.961 fit-spec clause in the `fig:sf3` caption) |
| **B** `app:loan` | CHANGED | item 7 ("except the $\gamma_\phi$ rung") |
| **C** `app:retention` | CHANGED | item 11 (single-representative-checkpoint clause) |
| **D** `app:compute` | CHANGED | Part 2(c): scan-amortized timing protocol + width-match confound |
| **E** `app:neg` | CHANGED | Part 2(a): 6 rows + 6 within-row numbers; Part 2(e): the CM-17 sampler-scope paragraph |
| **F** `app:pos` | CHANGED | Part 2(b): the finite-horizon head-to-head paragraph |
| **G** `app:gmor` | CHANGED | item 8 (un-fused G.1) + Part 2(d): demotion label, precision fine print, G.5, G.6 |

⚠ **The task expected "E, F, D, G changed". Measured: all seven.** A, B, C change **only** because three of the Head's own *Part-1* line items (12b, 7, 11) sit at sites physically inside appendices. **No Part-2 bank content entered A, B or C.**

## 3. Evidence backing each section of the edit (source attribution, per the task)

- **Base `submission.tex` (primary ancestor)**: items 1, 2, 3, 4, 5a/5b, 7 (L225), 9 (L145(iii)), 10 (L145(ii) verbatim), 11 (L267), 8 (L382); Part 2(a) rows/numbers (L312–L355), (b) (L376), (c) (L273, L299), (d) (L380 + App F.5/F.6), (e) (L361).
- **R3 bank `scratch/pj-restore-R3-preserved/pj_sub_v2_R3.tex` (already-fitted blocks, used for register/format fitting)**: item 6 (L108, pooled clause excluded), item 8's fitted form (L417), item 9's bullet label (L139), item 11's caption fit (L307), items 12a/12b (L130, L257, cross-reference halves excluded), item 13a (L58), Part 2 (c1) L313, (c2) L339, (a) L352/355/358/361 + L365–394, (e) L398, (b) L411, (d) L415/431/433/435.
- **Related-work positioning prose**: none added — item 13a's replacement clause is a *removal* of an unverified citation, taken from the R3 bank's wording; no scout-report prose entered this pass.

## 4. Compliance notes (Charter / claims matrix)

- **C-1 (as REVERSED 2026-07-07):** no defensive audit-confession paragraph exists or was added. J&P 2026 remains cited for the primitive's introduction only, and the CLU continuity sentence ("the Causal Learning Unit (CLU), introduced as CHLU in Jawahar & Pierini (2026)") is untouched at line 35.
- **C-2:** the designed/learned labelling in the abstract, §1 line 50, §4.1 and the restored G.6 honest-scope block is unchanged/reinforced; the restored blocks are all verification-side fences, never upgraded claims.
- **C-5:** the pass adds three scope qualifiers in-sentence (item 6 "overdamped-only", item 2 "at least", item 4 "most") and adds **no** generalizing claim. ⚠ The one surviving widening ("hold **generally** for the class", line 35) is **not on the approved list and was not touched** — Head's ruling still owed (SF-7).
- **C-6:** certificate fine print sits next to its claim — Part 2(d) puts the "do not quote 2.2×10⁻¹⁶ relative" precision block immediately under the GMOR figure it fences, and G.5's δ=0.3 no-NLO clause next to the expansion variable it bounds.
- **CM-17:** restored at its Appendix-E FAQ-row site in the approved wording — sampler-not-thermodynamics, the explicit "we never assert a relativistic unit 'has no equilibrium'", and the `newtonian_learned` no-touch scope. No `d·Θ` claim was imported (not enumerated, and its home is the theory note).
- **M1 hermeticity:** 0 hits, control 112. The file's only non-public reference is the pre-existing anonymized "the theory note" supplementary entry — untouched by this pass.

## 5. NOT RESTORED (verbatim from the build note §9, abridged)

- **Main-text-home bank items, excluded by the task:** N46 rider · anchor non-novelty clause · legacy-default warning · sleep flags · orphan-citation sentences · 4.5/4.6-decade clauses · seed attributions · ≈35× · floor ripple · coRNN footnote · "generally" · all intensifier/garble edits.
- **Appendix-home items excluded because not enumerated** (flagged for the Hub, ⚠ each is a one-line restoration if the Head wants them): (i) Appendix E's two *prose* reading rules — "never a 'drift rate'" (base L357) and "the breaking coefficient is **not** the integrator step ε" (base L359); the third reading rule is in, inside its own restored row; (ii) base L323's "A designed degeneracy does not survive superposition" (non-numeric, existing row); (iii) base L324's "tangential curvature predicted 0.100, measured 0.0994" (absent from the R3 bank's fitted row); (iv) base L382's far-field-bending clause (a fourth fact where item 8 names three); (v) R3's cross-referencing halves of the two slope labels and R3's `0.973` pooled clause; (vi) App C's lead-in one-liner and App A pointer prose; (vii) DOIs/status annotations.

## 6. Open editorial questions for the Hub / Head

1. **SF-7 (unruled since Addendum 60).** Line 35 still reads "Our results hold **generally** for the class of damped symplectic recurrences." Not on the approved list, so untouched. One word deletes it.
2. **The `−0.956` / `−0.961` statistic still has no owner.** Both sites now carry their fit spec, so no reviewer finds a contradiction — but the matrix pins nothing, so any future artifact quoting "the anchored overdamped slope" reopens it.
3. **The two Appendix-E prose reading rules** (§5(i)) are appendix-home banked fences that the approved list does not name. If the Head's amendment ("banked information goes to appendices only") is meant to cover them, they are a two-sentence pass. **Rule requested.**
4. **Page discipline: 15 pp** (main 6.91 / refs 1.90 / appendices 6.19), against the EA track's 4. Deferred by Head ruling; recorded, not optimized. This pass spent **+0.98 pp in appendices and +0.08 pp in main text**.

## 7. How I verified

- `python3 edit.py` — every replacement assertion-guarded at exactly one occurrence; input md5 asserted before the first edit; §-conversion a counted regex. Output: `edits applied: 28 · section-symbol conversions: 13 · new md5 a5758ad3eafcaf8971c73e7685d21450`.
- `/Library/TeX/texbin/pdflatex -interaction=nonstopmode pj_sub.tex` ×2, both exit 0; `grep -c "^! " = 0`; no `undefined`/`multiply defined` in `pj_sub.log`; `Output written on pj_sub.pdf (15 pages, 955879 bytes)`.
- **Bad boxes:** 1 Overfull `\hbox` (11.27979 pt, lines 288–298). **Proven inherited** — a rebuild of the pre-pass file in `scratch/pj-minimal-v2/beforebuild/` yields the identical single box. Underfull `\hbox` 7 → 22 (all inside Appendix E's narrow `p{1.52in}` columns, i.e. the restored rows; cosmetic). **No new overfull box.**
- Two-way numcheck + 6 positive-controlled sweeps (numbers in build note §6–7); PDF text-layer spot check confirming every restored block renders (task-RMSE row p. 12, chaos/`sleep_temperature` rows p. 13, sampler fence p. 13, finite-horizon paragraph p. 14, fine print p. 14, G.6 p. 15, substrate-scope + score sentence p. 7).
- `md5 *.tex *.sty figs/*` before/after: only `pj_sub.tex` differs.

## Git footprint
**None.** No tracked file touched. All writes are under gitignored `.claude/`: `NIPSsubmission/v2-neurreps/{pj_sub.tex, pj_sub.pdf, pj_sub.aux, pj_sub.log, pj_sub.out, BUILD-NOTE-R4.md}`, this report, and scratch under `.claude/scratch/pj-minimal-v2/` (`pj_sub.tex.BEFORE`, `md5-BEFORE.txt`, `edit.py`, `editlog.json`, `pass.diff`, `build1.log`, `build2.log`, `beforebuild/`). `BUILD-NOTE-R3.md` and `BUILD-NOTE.md` were left in place, untouched.

## Proposed handover updates (for the Hub)
- **V2 artifact state:** `pj_sub.tex` = **`a5758ad3eafcaf8971c73e7685d21450`**, **15 pp** (main 6.91 / refs 1.90 / appendix 6.19), 5/5 figures, References 51 → 50, build clean, **BUILD-NOTE-R4.md** is the current note (R3's note is superseded but retained for provenance).
- **Verification hook for the Advisor's zero-unenumerated-diffs check:** build note §5 is the 30-row classified list; `scratch/pj-minimal-v2/pass.diff` + `editlog.json` reproduce it mechanically.
- **Standing lesson to relay:** the task's "9 `§` occurrences" was a *line* count; the file had **11 occurrences on 9 lines**. Occurrence-vs-line counts should be stated explicitly in future style-rule worklists.
- **Still unowned (unchanged by this pass):** the anchored-slope statistic; the N46 per-seed discrepancy; SF-7.
