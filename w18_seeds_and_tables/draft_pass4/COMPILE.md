# Compile results, pass 4

Sequence: pdflatex -> bibtex (BIBINPUTS / BSTINPUTS = ~/clu/outputs/w17-paper/) -> pdflatex x3, in scratch build dirs laid out like w17-paper/ (`compile.sh`; counts grepped from doc.log).

| file | state | pdflatex exit | bibtex exit | pages | undefined citations | undefined refs | overfull | warnings |
|---|---|---|---|---|---|---|---|---|
| DRAFT-v2.tex | before (pass 3, md5 bb489831b1d0b14a3f1af2b68e61b8a3) | 0 | 0 | 47 | 0 | 0 | 0 | 40 |
| DRAFT-v2-style.tex | before (pass 3, md5 70b7d89cdd51c522a2c361d82cb211c2) | 0 | 0 | 48 | 0 | 0 | 0 | 39 |
| DRAFT-v2.tex | after, LIVE (md5 6caf80cd5278b9dea2d0d2971b5c729f) | 0 | 0 | 47 | 0 | 0 | 0 | 40 |
| DRAFT-v2-style.tex | after, LIVE (md5 5a58d5d9bed2965d4a4a1f63b9726ea4) | 0 | 0 | 47 | 0 | 0 | 0 | 40 |

The one extra warning in the style draft is one more "`h' float specifier changed to `ht'" (the longer tab:del). \cite lines: 32 (v2) and 30 (style); every citation resolves.
Style diffs (w20-referee4-fixes/tools/style_diff_check.py, live files): numbers EMPTY, PLACEHOLDERs EMPTY (byte-identical ordered sequence, the four new ones included), artefact paths EMPTY, section order identical (109 headings). Exit 0. See STYLE-DIFF-CHECK-pass4.md (baseline: STYLE-DIFF-CHECK-pass4-baseline.md, exit 0).
Provenance checker (w20-referee4-fixes/tools/check_provenance.py): findings identical to the pass-3 baseline on both files (7 unsourced lines; 8 / 9 not-found tokens, all pre-existing); checked tokens 2863 -> 2763 (rows removed).
Figure: fig_deletion rebuilt from figures/make_figures.py (pass-4 patch: make_figures.pass4.diff). Two builds byte-identical; every other make_figures.py output byte-identical to the previously installed SHA256-build1.txt. Installed: fig_deletion.{pdf,png}, PROVENANCE.json (fig_deletion entry only differs), make_figures.py, SHA256-build{1,2}.txt. Pre-install copies: figures_before/. Note: the installed fig_maze_readverb / fig_maze_requestlayer are the maze-w18 builder's (SHA256-maze-w18-build*.txt), not make_figures.py's; that was already so before pass 4 and was not changed.
