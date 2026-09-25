# pw2_draft_sections — PW-2 item 2: plain headings and re-sectioning (both drafts)

    python3 blocks.py DRAFT.tex                      # list the main-text chunks (paragraphs, floats) and heading runs
    python3 restructure.py IN.tex OUT.tex {v2|style} # emit the item-2 layout (layout.py); asserts every chunk used once
    bash compile_folder.sh FOLDER STEM               # pdflatex, bibtex, pdflatex until no "Rerun", built FROM the folder
    bash measure.sh STAGE {before|after}             # fig/table index (pw1 fig_table_index.py) + exact heading page markers
    python3 report.py STAGE OUTDIR                   # MOVE-MAP.md, HEAD-HELD.md, PROOF.txt (paragraph multiset proof)
    python3 index_md.py STAGE OUTDIR                 # SECTIONS.md, FIGTAB.md, REFS.md
    python3 headheld.py DRAFT.tex HEAD-HELD.md       # later spokes: re-check the Head-held paragraphs (exit 1 on change)

`page_markers_c.py` is pw1_draft_final's, with the bibliography hook pointed at `refs_clu_iclr.bib`.
Reused unchanged: `pw1_draft_sweeps/style_diff_check.py`, `pw1_draft_sweeps/check_provenance.py`,
`pw1_draft_final/{fig_table_index.py,pages.py}`, `pw1_draft_cuts/pdftext.py`.
Outputs and the report: ~/clu/outputs/pw2-draft-sections/ (FINDINGS.md).
