# pw2_draft_methods — PW-2 item 3: the Methods opening and the CHLU citation-scope rule (both drafts)

    python3 edit.py IN.tex OUT.tex              # the Head's text at % PW2-ITEM3-SLOT (\citet{jawahar2026chlu}), + the one
                                                #   ~\citep{jawahar2026chlu} at "zero-dynamics limit of the CLU"; anchors asserted once
    python3 rendered.py PDF.txt [--plant]       # rendered Methods opening vs PLAN.md's quoted text, word diff + exact string
    python3 charproof.py BEFORE.txt AFTER.txt [--plant]  # whole-PDF character multiset: after = before + exactly 3 insertions
    python3 chlu_scan.py DRAFT.tex              # every CHLU / CLU / physics-law mention, sentence-level (comments stripped)
    python3 verdicts.py DRAFT.tex               # the scan with the citation-scope verdict per row; exit 1 on VIOLATION/UNCLASSIFIED

PDF text via `pw1_draft_cuts/pdftext.py`; builds via `pw2_draft_sections/compile_folder.sh`; Head-held via
`pw2_draft_sections/headheld.py`; `pw1_draft_sweeps/{style_diff_check,check_provenance}.py` unchanged.
Outputs and the report: ~/clu/outputs/pw2-draft-methods/ (FINDINGS.md).
