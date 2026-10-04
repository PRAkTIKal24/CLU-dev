# pw2-draft-numbers — PW-2 item 5: numbers out of the main text, BOTH drafts
Session `pw2-draft-numbers` · branch `pw2/draft-numbers` (from `pw2/draft-methods`) · out `~/clu/outputs/pw2-draft-numbers/`
**You hold the `.tex` lock.** Read COMMON.md, PLAN.md §2 item 5 and item 1 (tell 4) IN FULL, and `~/clu/outputs/pw2-draft-{sections,methods}/{FINDINGS,HEAD-HELD}.md`.

## The one job
In the MAIN TEXT (title through the end of Discussion; the abstract included; captions and tables excluded), leave NO per-seed number and NO number that can be read directly from a table or a figure caption. What remains is high-level and rounded, stated plainly and without exaggeration. The Head's example, corrected by the plan: "the n-CLU produces about 7 more distinct valid novel tuples per held-out combination than the strongest matched baseline, an MLP".
- **Every number you remove from the prose** must already be in, or be moved into, a table or caption that the sentence now cites by `Tab.~\ref` / `Fig.~\ref`, carrying its `% SRC`. Never delete a number from the paper altogether without a pointer comment. If no table can take it, add a row or a note to the nearest existing table (log it), or keep a `%` pointer.
- **Every rounded number you write** must follow from a recorded value. Print: rounded text → source value → file → path. "About 7" means the recorded value rounds to 7. No exaggeration: never round up past the record, and never write "roughly double" unless the ratio is 1.8–2.2.
- **Abstract: at most five numbers.** Print the before and after lists. The abstract's opening (Head-held) and the maze `\TODO` stay byte-identical: count the numbers inside them, and cut elsewhere.
- **Head-held paragraphs:** leave them byte-identical. Count their numbers in the report as "Head-held".
- Numbers that are identifiers, not results, may stay: K*=32 as a design name, "six seeds" as protocol, section numbers, years in citations. List each one with its justification.
- Change only the numeric phrase and the minimal words around it (e.g. "reads 0.554 against 0.797" → "stays well behind the best matched baseline (Tab.~\ref{…})"). Do not rewrite sentences for style. Item 1 comes after you.

## Deliverables
- `NUMBERS-LEFT.md`: every number remaining in the main text of each draft (line, sentence, number, justification: rounded high-level / identifier / Head-held), and the count per section.
- `NUMBERS-MOVED.md`: every number removed from the prose: before sentence → after sentence → where the number now lives (table/caption label) → `% SRC`.
- Compile checks per COMMON, reporting the number multiset between the two drafts. Installed into both live folders.
**What would prove this wrong:** a remaining per-seed or table-readable number in the main text; a rounded number that does not follow from its record; a number that vanished from the paper without a pointer.
