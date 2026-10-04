# pw2-draft-sections — PW-2 item 2: plain headings and re-sectioning, BOTH drafts
Session `pw2-draft-sections` · branch `pw2/draft-sections` (from `pw1/draft-citations`) · out `~/clu/outputs/pw2-draft-sections/`
**You hold the `.tex` lock.** Read COMMON.md, then PLAN.md §2 item 2 in full, and the item-1 tell 5 rule on headings.

## The one job
Restructure the main text of both drafts into, in order: **Introduction → Methods → Experiments → Results → Discussion**, then References, then the Appendices.
- **MOVE paragraphs; do not rewrite them.** The mapping (verbatim from the plan):
  - Introduction = positioning, why the three properties matter, contributions.
  - Methods = the object (n-CLU, read, address book, footprint), the relational design (n-CLU-REL), the judges, deletion by downdate.
  - Experiments = the tasks and protocols: Shapes3D hold-outs, the ladder, the deletion protocol, CMAZE-T and its request/hold-out families, the baselines and the matching rule, seeds.
  - Results = the numbers: Shapes composition and ladder, deletion, maze conditions and relational coding, with tables and figures placed with their results.
  - Discussion = limitations, the seed-5 and relational limits, the CMAZE suite paragraph, the zero-dynamics clue, and the physics follow-up.
- Where one paragraph mixes method and result, keep it whole. Place it where its main content belongs and log it. Do NOT split sentences between sections.
- The only new text allowed: the five section headings; subsection headings as plain noun phrases (tell 5: no colons-as-sentences, no parentheticals); and, where a moved paragraph's opening connective would now dangle ("As above, …"), deletion of that connective only. Log every one.
- **Methods:** leave an empty slot at its very top (a `% PW2-ITEM3-SLOT` comment line). The next spoke inserts the Head's opening there.
- The AI use, ethics and reproducibility statements stay where the template puts them, at the end of the main text. **Appendix headings are NOT changed.** Appendix order is unchanged.
- Figures and tables are placed with their results. A float's position may move; its content does not.
- Fix every `\label`/`\ref` so there are 0 undefined references. Keep existing label keys wherever possible; if a section label must change, update every `\ref` to it.
- The abstract is untouched except for `Sec.~\ref` targets that now resolve differently. Its maze `\TODO` stays byte-identical.

## Deliverables
- `MOVE-MAP.md`: every paragraph (first 8 words + sha256), from old section → new section, in both drafts. Paragraph count before = after. The multiset of paragraph hashes before = after, excluding only the logged connective deletions; print that proof.
- `HEAD-HELD.md`: the abstract's opening sentence and the first paragraph of each of the five main sections after your restructure, per draft, with sha256. Methods' first paragraph = the item-3 slot, to be filled next.
- Section index and page counts per section, before → after; the figure/table index (number → label → caption lead → page).
- Both drafts: compile checks per COMMON, the style diff-check EMPTY (section order identical between the drafts), and installation into both live folders.
**What would prove this wrong:** a paragraph lost or duplicated; a changed sentence other than a logged dangling connective; a dangling `\ref`.
