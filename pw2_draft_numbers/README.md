# pw2_draft_numbers — PW-2 item 5: numbers out of the main text (both drafts)

    python3 edits.py                          # (data) every edit: exact before/after strings per draft + moved-number records
    python3 apply.py V2 STYLE V2_OUT STYLE_OUT  # applies edits.E; each `old` must occur once; refuses an edit whose trailing
                                              #   % comment would swallow the rest of its source line
    python3 scan.py DRAFT.tex [--json OUT]    # every number (digit runs + spelled cardinals) in the main text, floats/captions/comments excluded
    python3 numbers_left.py V2 STYLE          # NUMBERS-LEFT.md: every remaining number with its class; exit 1 on UNCLASSIFIED
    python3 moved_md.py                       # NUMBERS-MOVED.md from edits.py
    python3 rounded.py [--plant]              # every rounded phrase checked against its record (verbatim value + rule); exit 1 on FAIL
    python3 worddiff.py BEFORE.txt AFTER.txt  # rendered word diff (PDF text via pw1_draft_cuts/pdftext.py)

Builds: `pw2_draft_sections/compile_folder.sh`; Head-held: `pw2_draft_sections/headheld.py`;
`pw1_draft_sweeps/{style_diff_check,check_provenance}.py` unchanged. Outputs and the report: ~/clu/outputs/pw2-draft-numbers/ (FINDINGS.md).
