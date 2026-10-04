# pw1-draft-sweeps — plan items 2, 3, 5 (+ "the engine's judge") in BOTH drafts
Session `pw1-draft-sweeps` · worktree `~/clu/worktrees/CHLU-pw1-draft-sweeps` · branch `pw1/draft-sweeps` · out `~/clu/outputs/pw1-draft-sweeps/`
**You hold the `.tex` lock for your whole run.** Read `COMMON.md` first, then PLAN.md §1 items 2, 3, 5, 13 and "Pendings resolved".

## The one job
Sweep the two `.tex` files (prose, captions, table headers, table cells, appendices, the abstract except its `\TODO` maze sentence) so that every later edit starts in the new vocabulary and cross-reference style.
- **Item 2:** the nomenclature rule, the banned-word list, the economic-metaphor replacements, "scored run(s)"→runs, "spend/spent (runs)"→used, the physics renames (price law→retention law, priced channel→dissipative channel, $t_{\mathrm{pay}}$→$t_{1/2}$, applied consistently), "store"→n-CLU (or "the n-CLU's content" / "well contents" for the per-well content), and "naive-CLU (n-CLU)" at first mention in the text, with naive-CLU in the title. Also give the internal control names (`landscape_removed`, `landscape_shuffled`, `landscape_shuffled_across_wells`, `random_atoms`) plain names, each defined once where the controls are introduced, and use those names everywhere.
- **Item 3:** held-out factor combination at first use, then held-out combination / combination; "per held-out combination" in headers; capsule never oblong; ledger→memory footprint / "the footprint breakdown"; "held-out composition" never.
- **Item 5:** every cross-reference becomes `Sec.~\ref`, `Fig.~\ref`, `Tab.~\ref`, `App.~\ref`, `Eq.~\ref`, `Alg.~\ref` (plural forms `Figs.~\ref{a} and \ref{b}`). No `\S` anywhere. Baseline counts in DRAFT-v2: `\S\ref` 43, `Figure~\ref` 7, `Table~\ref` 80, `Appendix~\ref` 29.
- **Pending resolved:** "exact judge" → "the engine's judge" everywhere. State once, where the judge is defined, that it uses integer arithmetic with no tolerance.

## Not yours
- The figure images and figure scripts: other spokes regenerate them. Sweep only the `.tex` side, captions included.
- Deleting maze material (items 14, 15, 17): sweep it like everything else, and do not delete it.
- Wording items 7, 8, 13: not yours. Do not introduce "exact" in any new text, though.
- The abstract's maze `\TODO` sentence: leave it byte-identical.

## Acceptance (measured, printed in FINDINGS)
1. Before→after replacement table per rule, with counts per draft (the two drafts must show identical counts, or explain each difference).
2. A residual scan of RENDERED text (exclude `%` comments) for every banned token, using word boundaries. Every residual is either 0 or listed with its line and the reason it stays: a proper name such as "Gated DeltaNet", a bibliography title, or a technical sense the plan allows, such as "cost" or "not cheap". Print the scan command.
3. 0 occurrences of `\S`, `Section~`, `Figure~`, `Table~`, `Appendix~` before a `\ref` in either draft. Replacement counts per kind.
4. Both drafts: exit 0, 0 undefined citations/references, style diff-check EMPTY, provenance findings identical or improved, `\PLACEHOLDER` count and order unchanged. Page counts before and after.
5. Installed byte-identically into both drafts and both PDFs.
**What would prove this wrong:** any banned token surviving in rendered text without a listed reason, or a meaning changed by a mechanical replace. Check the economic-metaphor replacements in context, one by one, and do not run a blind regex over them. Print every metaphor replacement with its sentence.
