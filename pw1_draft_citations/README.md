# pw1_draft_citations — cite against the Head's Zotero bib `refs_clu_iclr.bib`, both drafts

Never writes a `.bib` (the bib is the Head's; `bibparse.py` only reads).

| file | job |
|---|---|
| `bibparse.py` | brace-balanced BibTeX reader; DOI / arXiv / normalised-title extractors |
| `keymap_verify.py OLD.bib NEW.bib keymap.json` | independent check of the Hub's 55 -> 55 key map: per pair, which of DOI, arXiv ID, normalised title agree |
| `scan_named.py DRAFT.tex` | first mention of each named work / method / dataset / model in the main text and the appendices, with the citations in that sentence (NOT IN BIB marks names with no entry) |
| `rules.py` | the pass: H1–H14 (hand-typed author-year -> natbib), M1–M2 (added at first main-text mention), P1–P6 (added at first appendix mention), B1–B3 (bibliography lines; url hyphen breaks) |
| `cite_edit.py IN OUT keymap.json REPORT.md` | applies `rules.py` (each rule exactly once, else exit 2), renames every key of every `\cite*`/`\nocite` on code, keeps the line count, never leaves a citation at a line start |
| `neutralise.py IN OUT` + `neutral_builds.sh STAGE` | control builds in which every citation command and the bibliography print nothing; their rendered word diff can show only hand-typed citation text |
| `residue.py WORDDIFF-neutral.txt` | removes the 14 hand-typed strings (H1–H14) from the BEFORE leftovers; the rest must equal the AFTER leftovers (non-citation residue EMPTY) |
| `cite_table.py BEFORE AFTER AFTER.bbl keymap OLD.bib NEW.bib OUT.md` | every citation before -> after with its sentence and rendered string; per-entry counts before -> after; uncited entries; abstract citations; doubled `\citet` names |

Order used: `keymap_verify.py`; `scan_named.py` (before); `cite_edit.py` V0 -> V2 per draft; `pw1_draft_final/compile_tree.sh` V0, V2;
`pw1_draft_cuts/pdftext.py`; `pw1_draft_template/check_stage.sh V0 V2`; `neutral_builds.sh`; `residue.py` (+ `worddiff.py --plant` controls);
`cite_table.py`; install; fresh rebuild of the installed tree; `scan_named.py` (after).
