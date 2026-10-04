# pw2-draft-final — the closing audit of PW-2
Session `pw2-draft-final` · branch `pw2/draft-final` (from `pw2/draft-leftovers`) · out `~/clu/outputs/pw2-draft-final/`
**You hold the `.tex` lock.** Read COMMON.md, PLAN.md §2 IN FULL, and every `~/clu/outputs/pw2-*/FINDINGS.md`. **No content edits.** Only a mechanical defect fix is allowed (a broken ref, or a line-break defect), printed and installed by the standard method.

## The one job: produce `FINAL-INDEX.md`
1. A full compile of both live folders from a clean copy: 0 undefined citations/references, 0 errors; overfull boxes listed; bibtex warnings verbatim; pages.
2. The style diff-check between the drafts: placeholders, artefact paths, section order EMPTY; the number multiset EQUAL between the two drafts (print both multisets' sizes and any difference).
3. Provenance findings compared with the PW-1 final state (`pw1-draft-citations`): identical or improved; list any change.
4. The figure/table index, number → label → caption lead → page, for both drafts.
5. Page counts per section (marker method C from `pw1_draft_final/`): the five main sections, the statements, the references, each appendix; main text against 9 pp.
6. **Every number left in the main text**, with its justification. Re-derive it independently of pw2-draft-numbers, and flag any disagreement.
7. **Per-tell counts per section, now**, re-measured with pw2-draft-tells' detectors, beside that spoke's before counts.
8. **Head-held paragraphs:** each is byte-equal to the latest HEAD-HELD.md.
9. Every remaining `\TODO` and `\PLACEHOLDER`, by argument.
10. A citation-scope audit of `jawahar2026chlu`: every citing sentence, with its topic (unit / dynamics / governor only).
11. The bib sha256, unchanged.
Positive-control each scan.
