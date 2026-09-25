# pw1_draft_wording2 — PW-1 item 7 completion (the seed-5 0.04 margin and 106 of 317)

    python3 verify_record.py                  # value -> file -> path for 0.04, 317, 211, 63, 43 (and 106 = 317 - 211); exit 1 = STOP
    python3 edit.py IN.tex OUT.tex            # one anchored operation, same for DRAFT-v2 and DRAFT-v2-style
    bash ../pw1_draft_sweeps/compile_check.sh STAGE STAGE/build
    bash ../pw1_draft_wording/check_stage.sh SNAPDIR STAGE
    python3 check_derived.py DRAFT.tex PROV.txt   # re-reads provenance not-found tokens; an arithmetic token counts only if
                                                  # the comment line below states '<tok> is A - B' with A, B in the cited sources
Report: ~/clu/outputs/pw1-draft-wording2/FINDINGS.md.
