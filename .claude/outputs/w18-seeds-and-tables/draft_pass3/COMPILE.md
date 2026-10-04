# Compile results, pass 3

Sequence: pdflatex -> bibtex (BIBINPUTS / BSTINPUTS = ~/clu/outputs/w17-paper/) -> pdflatex x3, in scratch build dirs laid out like w17-paper/ (script: compile.sh logic, counts grepped from doc.log).

| file | state | pdflatex exit | bibtex exit | pages | undefined citations | undefined refs | overfull |
|---|---|---|---|---|---|---|---|
| DRAFT-v2.tex | before (md5 fab8ddcb8830661ba7dad2fdf6206f7a) | 0 | 0 | 47 | 0 | 0 | 0 |
| DRAFT-v2-style.tex | before (md5 a77cdae5e27f1f62d83fffb592c9dc59) | 0 | 0 | 48 | 0 | 0 | 0 |
| DRAFT-v2.tex | after, LIVE (md5 bb489831b1d0b14a3f1af2b68e61b8a3) | 0 | 0 | 47 | 0 | 0 | 0 |
| DRAFT-v2-style.tex | after, LIVE (md5 70b7d89cdd51c522a2c361d82cb211c2) | 0 | 0 | 48 | 0 | 0 | 0 |

32 \cite commands in each draft; all resolve. Style diffs (w20-referee4-fixes/tools/style_diff_check.py, live files): numbers EMPTY, PLACEHOLDERs EMPTY (10 = 10, byte-identical order), artefact paths EMPTY (+ provenance comments identical order), section order identical (111 headings, 82 labels). Exit 0. See STYLE-DIFF-CHECK-pass3.md.
