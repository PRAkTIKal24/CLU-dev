# pw2_draft_final — the closing audit of PW-2 (both drafts), read-only on the live folders

    bash run_final.sh OUTDIR                  # everything below on a clean copy of ~/clu/pw2/live/{v2,style}; stage in OUTDIR/final-<UTC>/
    python3 make_final_index.py STAGE OUTDIR  # FINAL-INDEX.md (+ NUMBERS-NOW-*.md) from the stage (needs STAGE/checks/SUMMARY.md)

Pieces (all written fresh for this audit unless marked reused):
    numscan.py DRAFT.tex                      # every digit run and spelled cardinal in the main text (title .. AI use statement)
    numclass.py DRAFT.tex [--json OUT]        # class + justification per token; exit 1 on UNCLASSIFIED (independent of pw2_draft_numbers)
    rounded_check.py DRAFT.tex [--plant]      # every rounded phrase: record values verbatim in the evidence file + the rounding rule
    compare_numbers.py MINE.json NUMBERS-LEFT.md DRAFT.tex   # token and class agreement with pw2-draft-leftovers' list
    tells_now.py V2 STYLE TELLS-COUNT.md      # pw2_draft_tells/detect.py (reused, unchanged) now, beside that spoke's before -> after
    prov_compare.py PW1.txt NOW.txt           # check_provenance.py findings vs the PW-1 final drafts, line numbers masked
    scans.py todo|cite|headheld|defects ...   # \TODO/\PLACEHOLDER by argument; jawahar2026chlu scope; Head-held raw bytes; defects
    controls.sh STAGE                         # positive controls for every scan, on planted copies (one planted-defect compile)
Reused unchanged: pw2_draft_sections/{compile_folder.sh,measure.sh,page_markers_c.py,headheld.py},
pw1_draft_final/{fig_table_index.py,pages.py}, pw1_draft_sweeps/{style_diff_check.py,check_provenance.py},
pw1_draft_cuts/pdftext.py, pw2_draft_tells/{detect.py,controls.py}.
Report: ~/clu/outputs/pw2-draft-final/{FINAL-INDEX.md,FINDINGS.md}.
