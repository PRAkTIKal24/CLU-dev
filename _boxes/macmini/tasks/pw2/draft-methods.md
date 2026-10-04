# pw2-draft-methods — PW-2 item 3: the Methods opening, BOTH drafts
Session `pw2-draft-methods` · branch `pw2/draft-methods` (from `pw2/draft-sections`) · out `~/clu/outputs/pw2-draft-methods/`
**You hold the `.tex` lock.** Read COMMON.md, PLAN.md §2 item 3 IN FULL, and `~/clu/outputs/pw2-draft-sections/{FINDINGS,HEAD-HELD}.md`.

## The one job
1. At `% PW2-ITEM3-SLOT` (the top of Methods), insert the Head's text VERBATIM, as the plan gives it between the quote marks, in both drafts. You set only:
   - the citation macro: each "Jawahar \& Pierini (2026)" becomes `\citet{jawahar2026chlu}`, the author-as-noun form, which prints "Jawahar & Pierini (2026)"; keep `~` spacing where natbib needs it;
   - LaTeX typography: `\&`, and quotes and dashes set properly.
   Change no word, no punctuation other than LaTeX escapes, and no spelling (keep "factorized", "virtue"). Print the rendered sentence beside the plan's text, and a word diff that must be EMPTY apart from the citation rendering.
2. **The citation-scope rule, everywhere the CHLU is mentioned** in both drafts (text, captions, appendices):
   - `jawahar2026chlu` may be cited ONLY for the unit, its symplectic Hamiltonian dynamics, and the relativistic governor;
   - NEVER for memory, deletion, test-time compute, retention or GMOR;
   - the physics section's laws (retention law, dissipative channel, GMOR) stay this paper's own theory, and no sentence may imply the arXiv paper contains them;
   - the shorts are never cited.
   List every CHLU mention (line, sentence), with the verdict: cite / do not cite / already fine / VIOLATION. Add `~\citep{jawahar2026chlu}` at the first mention of the CHLU unit or its dynamics outside Methods only where the rule allows it. Where a sentence would imply the arXiv paper holds a law, report it; do not rewrite it (item 1 and the Head own prose).
3. Update `HEAD-HELD.md` in your out dir, adding the new Methods first paragraph and its hashes.
## Acceptance
Both drafts: compile per COMMON, `jawahar2026chlu` cited (so 56/56 entries cited), 0 undefined; style diff EMPTY; bib sha unchanged; installed into both live folders.
