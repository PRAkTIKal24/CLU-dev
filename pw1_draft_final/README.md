# pw1_draft_final — the PW-1 closing audit (jobs 1-6), zero content change

    bash hash_placed.sh OUTDIR TAG              # sha256 of the 26 placed figure files in both folders
    bash compile_tree.sh TREE                   # pdflatex, bibtex, pdflatex until no "Rerun" (max 5); metrics per draft
    python fix_title.py IN.tex OUT.tex          # the one mechanical fix: the title's forced line break
    bash run_final.sh OUTDIR [--inplace]        # jobs 1-6 on the live tree (scratch + optional in-place compile)
    python make_report.py OUTDIR FINAL-INDEX.md # the tables for the Head
Pieces: fig_table_index.py (job 2), page_markers_c.py + pages.py (job 3; page_markers.py = the A/B variants that were
measured and rejected), ph_list.py (job 4), residual_audit.py (job 5), folder_audit.py (job 6).
Report: ~/clu/outputs/pw1-draft-final/{FINAL-INDEX.md,FINDINGS.md}.
