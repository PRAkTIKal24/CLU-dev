# pw1-draft-citations — cite against the Head's Zotero bib `refs_clu_iclr.bib`, in BOTH drafts
Session `pw1-draft-citations` · worktree `~/clu/worktrees/CHLU-pw1-draft-citations` · branch `pw1/draft-citations` (from `pw1/draft-template` a4de290; the draft checkers are in the tree) · out `~/clu/outputs/pw1-draft-citations/`
**You hold the `.tex` lock.** Read `COMMON.md`, then `~/clu/outputs/pw1-draft-template/FINDINGS.md` (the drafts' current state).

## ⛔ THE BIB IS THE HEAD'S. Never write, edit, reformat, sort or regenerate ANY `.bib` file.
The Head exported `refs_clu_iclr.bib` from Zotero. The Hub copied it byte-identical to `~/clu/tasks/pw1/refs_clu_iclr.bib` (sha256 `26cc298f0ffabdda3be64b71faeaa3cbb3c04d978b05be59caacdc7c30cd4520`, 55 entries). The Hub places the same bytes in `w17-paper/`. Check its sha256 before and after your run; it must not change. If an entry makes bibtex warn or fail, REPORT it (the key, the field, the message) and change nothing in the bib.

## The one job
1. **Keys.** Replace every old key in every `\cite…{}` of both drafts with its new key. The Hub's DOI/arXiv/title mapping is at `~/clu/tasks/pw1/keymap.json` (55 → 55, one-to-one). Verify it independently: for each pair, the DOI, arXiv ID or normalised title must agree. Print the table, old key → new key → matching field.
2. **`\citet` vs `\citep`, at every citation.** Use `\citet` where the sentence uses the authors as a noun, for example "X et al. show …", "as in X", "following X", or "X's method". Use `\citep` where the citation is a parenthetical pointer. Write each one in the Head's form, `text~\citet{…}` / `text~\citep{…}`, with a non-breaking space before it and none dangling at a line start. For a `\citet` the author names come from the command, so remove any hand-typed author name it would duplicate. Print every citation, before → after, with its sentence.
3. **Right locations.** Scan the rendered text for every named work, method, dataset or model the paper relies on that has an entry in the bib but no citation at that point: for example Shapes3D, the Transformer, the LSTM, Mamba-2, Gated DeltaNet, LACE, PoE, SISA, the named unlearning methods, and the physics references. Add the citation at FIRST mention in the main text, and at first mention in the appendices when the appendix is read on its own. Do not cite the same work twice in one sentence, and do not add citations inside the abstract's Head-held `\TODO`. Only keys that exist in `refs_clu_iclr.bib` may be cited. If the text names a work that has no entry, print it in a "missing from the bib" list and add nothing.
4. **Point the drafts at the new bib:** `\bibliography{references}` → `\bibliography{refs_clu_iclr}`, and the `\IfFileExists{references.bib}` guard to match. The `\bibliographystyle` line is unchanged.
5. **Every one of the 55 entries is cited at least once** (that is the current state). If an entry ends up uncited, print it. Print, for each entry, the number of citations before and after.

## Nothing else changes
No sentence rewritten beyond inserting or re-forming a citation (and deleting a hand-typed author name a `\citet` now prints). No numbers, no vocabulary edits. The abstract's `\TODO` stays byte-identical. The drafts stay in lockstep: identical citation keys and forms at the same locations in both, so the style diff-check stays EMPTY.

## Acceptance (printed in FINDINGS)
- Both drafts: exit 0, **0 undefined citations**, 0 undefined references, 0 errors, 0 overfull boxes; bibtex warnings listed verbatim; the number of bibitems; pages before → after.
- Style diff-check EMPTY; provenance identical; `\PLACEHOLDER` / `\TODO` byte-identical.
- The rendered-text word diff before → after shows ONLY citation text (author-year strings) and removed hand-typed names. Print the non-citation residue, which must be empty.
- `refs_clu_iclr.bib` sha256 unchanged in `~/clu/tasks/pw1/` and in `w17-paper/`.
- Installed byte-identically into both drafts and both PDFs.
**What would prove this wrong:** any undefined citation, any changed bib byte, a `\citet` whose author names now print twice, or a citation in the abstract `\TODO`.
