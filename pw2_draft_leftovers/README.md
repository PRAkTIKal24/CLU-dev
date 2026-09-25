# pw2_draft_leftovers — PW-2 item 4 (the leftovers) + the item-5 follow-up on the Discussion limitations list, both drafts

    # the ladder legend (part 1): w21_figures/make_figures_w22.py carries the one-string edit
    #   "seed at $10{\times}$ steps" -> "seed at $10{\times}$ training steps"; build it from a copy of paper-visuals:
    cp -Rp ~/clu/outputs/paper-visuals SRC && cp w21_figures/make_figures_w22.py SRC/ && (cd SRC && bash build.sh D1 D2)
    #   (two builds, byte-identical, into D1/D2 only; install fig_ladder.pdf alone into the live folders' figures/)
    python3 edits.py                          # (data) every .tex edit: exact old/new per draft (parts 2-7), moved numbers (part 7)
    python3 apply.py V2 STYLE V2_OUT STYLE_OUT  # each old must occur once; refuses a comment-swallowing edit
    python3 check.py BASE STAGE > CHECKS.md    # \TODO 4->3, \PLACEHOLDER 2->0 (named), 0 absolute paths, Figs. 5-8 cited,
                                              #   quote without \small, labels (exit 1 on a failed line)
    python3 rounded.py [--plant]              # part 7: every rounded phrase in the limitations list against its record
    python3 numbers_left.py V2 STYLE          # pw2_draft_numbers' classifier with the limitations "lock-step" class removed
    python3 report.py BASE STAGE OUT          # EDITS.md, LIMITATIONS.md, NUMBERS-LEFT-DISCUSSION.md
    uv run --with pypdf python rendered.py BEFORE.pdf AFTER.pdf   # rendered spot check of the edited sentences
    uv run --with pypdf python quotesize.py DRAFT.pdf             # the deletion-scope quote's font size vs the body

Builds: `pw2_draft_sections/compile_folder.sh`; Head-held: `pw2_draft_sections/headheld.py`; style diff-check and
provenance: `pw1_draft_sweeps/{style_diff_check,check_provenance}.py`. Outputs and the report:
~/clu/outputs/pw2-draft-leftovers/ (FINDINGS.md).
