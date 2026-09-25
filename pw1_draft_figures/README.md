# pw1_draft_figures — PW-1 figure install, text side (items 1, 1.5/8, 9, 10, 11, 12/12.7, 16.1)

    bash ../w21_figures/build.sh DEST1 DEST2                  # (unchanged) the figure build, run twice into scratch
    python3 edit.py IN.tex OUT.tex EDIT.json                  # captions, slots and paragraphs; 30 anchored ops per draft
    bash ../pw1_draft_sweeps/compile_check.sh STAGE BUILD [FIGDIR]
    bash check_stage.sh SNAPDIR STAGEDIR                      # style diff, provenance, captions scan, PDF text, index
    python3 captions.py DRAFT.tex OUT.json [OUT.tex]          # figure captions by label (banned-word scan input)
    python3 make_index.py FOLDER PAPER_DIR AUX_DIR OUT_DIR COMMIT   # README.md of a figure folder (port of w22 make_index)
    python3 figures_md.py FOLDER INDEX.json COMMIT            # the PW-1 section at the top of FIGURES.md

blocks/*.tex hold the new captions and paragraphs (figure spokes' wording, SRC comments naming real files).
make_index_w22_orig.py is `git show w22/paper-visuals:w22_paper_visuals/make_index.py`, kept for the diff.
Report: ~/clu/outputs/pw1-draft-figures/FINDINGS.md.
