2026-09-22T23:01:22Z start: read AMENDMENT-1, COMMON, cite-tagger task
- gate: CITATIONS.md, DRAFT-v2.tex, DRAFT-v2-style.tex present; END_UTC exists for citation-scout, draft-v2, style-paraphrase.
- VENUE.md §8: natbib author-year round (\citet/\citep), bst iclr2027_conference.
- DECISION (insertion-only rule): where the draft already prints a hand-written author-year string (e.g. "Tenenbaum and Freeman, 2000", "Cao and Yang (2015)"), a visible \citep would print author-year twice and replacing the string would be a prose change. There I insert \nocite{key}: the reference enters the bibliography and the rendered sentence is unchanged. Everywhere else (method/concept named without author-year) I insert \citep{key}. Converting the hand-typed strings to \citet/\citep is a prose edit left to the Head's editing pass.
- DECISION: citations placed only where the draft's own sentence makes the claim CITATIONS.md maps to that key; no cite in the abstract (ICLR convention). Keys with no slot stay in references.bib (not \nocite'd, so they do not appear in the reference list).
2026-09-22T23:06:03Z running build_bib.py (Crossref/arXiv/DBLP fetch)
2026-09-22T23:17:50Z installed cited DRAFT-v2.tex, DRAFT-v2-style.tex, references.bib into w17-paper (originals kept in pre-cite/ with SHA256)
- 2026-09-22T23:19:33Z build_bib.py: 96 entries (Crossref/arXiv/S2; DBLP bot-walled -> Semantic Scholar). lecun2022path excluded.
- insert_cites.py: 38 insertions per draft (32 citep, 6 nocite); identical key order; each anchor matched exactly once.
- PROSE-CHECK: byte-identical after stripping inserted cites; word counts equal; word diff 0.
- STYLE-DIFF-CHECK (style spoke's own script) on cited pair: 4x EMPTY DIFF.
- provenance: 3 false positives from \nocite key digits (checker ignores cite[pt]? only); with nocite ignored, findings identical to pre-cite.
- compile: pdflatex->bibtex->pdflatex x3 on both, all exit 0, 35 pp, 62 refs, no undefined cites/refs. Logs: w17-paper/compile-v2.log, compile-v2-style.log.
- CITATIONS-FOR-ZOTERO.md: 76 arXiv / 48 DOI; omitted 20 / 48; 2 in neither.
- See FINDINGS.md for uncited keys (34) and Head items.
