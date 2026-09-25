# pw1_draft_sweeps — PW-1 items 2, 3, 5 and "the engine's judge" (vocabulary and cross-reference sweep)

    python3 sweep.py IN.tex OUT.tex COUNTS.json     # one draft; comments, \label/\ref/\cite keys and the abstract's Head-held \TODO are masked
    python3 scan.py DRAFT.tex [--lines] [token ...] # residual scan of rendered source (comments and ref keys excluded)
    python3 make_tables.py SNAPDIR STAGEDIR OUTDIR  # REPLACEMENTS.md and RESIDUALS.md
    python3 style_diff_check.py A.tex B.tex         # (reused, w17) numbers / placeholders / provenance / headings / floats
    python3 check_provenance.py DRAFT.tex           # (reused, w17) SRC-comment provenance checker
    bash compile_check.sh SRCDIR BUILDDIR           # (reused) builds both drafts in the w17-paper layout

Outputs and the report: ~/clu/outputs/pw1-draft-sweeps/ (FINDINGS.md).
