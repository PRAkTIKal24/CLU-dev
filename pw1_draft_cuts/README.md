# pw1_draft_cuts — PW-1 items 14, 15, 17 (maze section CMAZE-T only; Figs. maze-rel-tuples and cmazet-deletion out)

    python3 cuts.py IN.tex OUT.tex CUTS.json {v2|style}      # the cut; every op asserts a unique literal anchor
    python3 scan_cuts.py DRAFT.tex [--lines]                  # residual scan: CMAZE-1/2/2b/O, verb, route, keys, file stems
    python3 report.py SNAP STAGE BUILD_BEFORE BUILD_AFTER OUT.md  # removal table, refs, placeholders, index, pages
    uv run --with pypdf python pdftext.py IN.pdf OUT.txt      # PDF text for the rendered residual scan

Builds and checks reuse ../pw1_draft_sweeps (compile_check.sh, style_diff_check.py, check_provenance.py, scan.py).
Report: ~/clu/outputs/pw1-draft-cuts/FINDINGS.md.
