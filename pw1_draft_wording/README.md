# pw1_draft_wording — PW-1 wording items 7, 8, 13 in both drafts

    python3 edit.py IN.tex OUT.tex EDIT.json          # 56 anchored operations per draft (per-draft alternatives)
    bash ../pw1_draft_sweeps/compile_check.sh STAGE BUILD
    bash check_stage.sh SNAPDIR STAGEDIR              # style diff, provenance, captions scan, PDF text, index, PH/TODO
    python3 report.py V2.json STYLE.json OUT.md       # before -> after table

check_stage.sh is pw1_draft_figures/check_stage.sh with the hard-coded worktree path replaced by this repo's.
Report: ~/clu/outputs/pw1-draft-wording/FINDINGS.md.
