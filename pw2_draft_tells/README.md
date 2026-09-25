# pw2_draft_tells — PW-2 item 1: the seven machine-prose tells, subtractive, both drafts (+ the item-4 borrowings)

    python3 segment.py DRAFT.tex              # units: paragraphs (cut at \paragraph and headings; Head-held lines split off),
                                              #   list items, the quote, captions, table rows, headings; section + Head-held flag
    python3 detect.py DRAFT.tex [--list T3]   # the seven detectors (rules in the docstring); per-section counts or a hit list
    python3 propose.py DRAFT.tex              # for each T7-failing paragraph, split points that would give a <=8-word sentence
                                              #   directly after a >=18-word one (review aid only; edits are written by hand)
    python3 edits.py                          # (data) every edit: exact old/new per draft, tells, note
    python3 apply.py V2 STYLE V2_OUT STYLE_OUT   # applies edits.E; each old must occur once; refuses a comment-swallowing edit
    python3 audit.py V2B STYLEB V2A STYLEA OUT.json  # words added/removed per edit; whole-document net change == edits' net change;
                                              #   words per section before -> after
    python3 controls.py DRAFT.tex OUTDIR      # positive controls: planted tells raise every detector; the brief's 7 examples match
    python3 report.py V2B STYLEB V2A STYLEA AUDIT.json OUTDIR   # TELLS-COUNT.md, EDITS.md, WORDCOUNT.md

Builds: `pw2_draft_sections/compile_folder.sh`; Head-held: `pw2_draft_sections/headheld.py`; style diff-check and provenance:
`pw1_draft_sweeps/{style_diff_check,check_provenance}.py`. Outputs and the report: ~/clu/outputs/pw2-draft-tells/ (FINDINGS.md).
