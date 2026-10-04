# w17 spoke/cite-tagger — MAKE BOTH DRAFTS COMPILE WITH REAL CITATIONS
**Read `COMMON.md` first.** Slug `cite-tagger`. Agent type: paper-writer.
⭐ **Gate: start when ALL THREE exist —** `~/clu/outputs/w17-paper/CITATIONS.md`, `DRAFT-v2.tex`, `DRAFT-v2-style.tex`, and the END_UTC of their owners.

## The one job, three parts
1. **Insert `\cite{key}`** at the right locations in **BOTH** `.tex` files, per the citation style `VENUE.md` names (`\citep`/`\citet`/numeric — use what the venue says).
2. **Generate `references.bib` AUTOMATICALLY from arXiv/DOI metadata** so **both drafts COMPILE now**, without waiting for the Head. ⭐ This bib is the one the drafts compile against; ⛔ nothing in waves 17–18 depends on it being replaced later.
3. **Write `CITATIONS-FOR-ZOTERO.md`** containing **exactly two copyable blocks**, for a later Zotero pass outside these waves:
   - **Block 1:** the arXiv IDs, **one ID per line**, nothing else — no bullets, no numbering, no commentary.
   - **Block 2:** the DOIs, **one DOI per line**, nothing else.
   ⭐ Each block must be a single fenced code block that a human can select and paste in one go. ⛔ Entries with no arXiv ID are omitted from block 1; entries with no DOI are omitted from block 2 — and you state the omitted counts beneath the blocks.

## Hard requirements
- ⛔ **Do not fabricate a bib entry.** If citation-scout marked an entry `no DOI`/`no arXiv`, build the entry from what exists and mark it.
- ⛔ **Do not change one word of prose** in either draft. You insert `\cite{}` and nothing else.
- ⭐ **Prove both drafts compile.** Run the compiler; save the logs as `compile-v2.log` and `compile-v2-style.log`. If LaTeX is unavailable, say so in `LOG.md` and state exactly which command should be run — ⛔ do not claim a compile you did not perform.

## Acceptance criterion
`references.bib` + `CITATIONS-FOR-ZOTERO.md` (two clean copyable blocks + omitted counts) + both compile logs, and both `.tex` files carrying `\cite{}` with no prose change (show a word-count diff proving prose is untouched).

## Files you own
`references.bib`, `CITATIONS-FOR-ZOTERO.md`, the compile logs, and `\cite{}` insertions in both drafts. ⛔ No other edit to either draft.
