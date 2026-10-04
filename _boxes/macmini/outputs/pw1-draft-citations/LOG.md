# LOG — pw1-draft-citations
- 14:46Z start; read COMMON.md, brief, PLAN.md, pw1-draft-template FINDINGS.
- 14:47Z snapshots in before/ (SHA256-before.txt). bib sha 26cc298f… in tasks/ and w17-paper/. keymap verified independently: 55/55 match on DOI/arXiv/title (KEYMAP-VERIFY.md). Year differences old->new bib: golatkar 2023->2024, guo 2020->2023, kim 2018->2019, quan 2026->2027 (ceci_exact_2027), vaswani 2017->2023 (AUTHOR-YEAR-COMPARE.md); reported, bib untouched.
- CHOICES (logged, closest to an ICLR main-track paper):
  * abstract: no citation added (ICLR abstracts carry none; the maze sentence is the Head's \TODO). "Main text" = after \end{abstract}.
  * section headings: no citation (they go to PDF bookmarks); the next sentence carries it (world models, Sec. 1.2).
  * appendices: "first mention when read on its own" = first mention across the appendix block (one citation per work there), captions included.
  * Sec. 1.1 opening sentence ("sharded retraining, compositional diffusion and physics-informed variants ... each is cited in one sentence below"): no citation added — the sentence itself defers them to the next paragraph, where they are cited; adding them would contradict the sentence and change nothing but repetition.
  * \citealp (natbib: \citep without brackets) used ONLY where a bare hand-typed "Author, Year" is an item inside an existing parenthesis (Tenenbaum & Freeman; Vlasic et al.; Hall et al.): \citep there would print "((…))". Named methods inside the lists get Method~\citep{…} and lose the hand-typed year the command prints.
  * Blelloch--Golovin stable-matching table: the authors are used as a noun modifier ("X's table") -> the~\citet{…}; hand-typed names removed.
  * Every edit keeps the line count: when a replaced span held a line break, the break moves to the nearest space before the span.
- 14:52Z stage stage-145204: V0 = live copy, V1 = cite_edit.py output (24 rule edits, 72 keys renamed, line counts equal, both drafts same rules).
- 14:57Z V2 = cite_edit (rules H1-H14, M1-M2, P1-P6, B1-B3). Build: both exit 0, 0/0 undefined, 0 errors, 0 overfull (B3 url hyphens fixes the lee_fast_2021 URL, 133pt overfull in V1), bibtex 1 warning 'empty journal in ha_world_2018'. Style diff ALL EMPTY; PH/TODO byte-identical; provenance improved 7->3 unsourced lines (hand-typed years gone), not-found equal; neutral-build residue EMPTY both drafts, positive controls NOT EMPTY. Pages: DRAFT-v2 42->42 (0 labels moved), style 43->44 (refs one page longer).
- 14:58Z lock guard clear; live==snapshot; installed V2 .tex/.pdf/.aux/.bbl/.blg/.log/.out into w17-paper (previous build files kept in before/w17-build-artefacts/).
- 15:00Z fresh rebuild of installed tree == stage; after-scan written; FINDINGS written; committed 2318c1b; bundle verified (exit 0); replay == installed; bib sha unchanged at end (26cc298f… both). DONE.
