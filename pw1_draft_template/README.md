# pw1_draft_template — the ICLR 2027 template pass (zero content change)

| file | job |
|---|---|
| `apply_template.py IN OUT [--only header,lists,LINE,...]` | the two mechanical edits: load the template's own `fancyhdr`/`natbib` on the class-style line (running header), and drop enumitem keys that override the style file's list parameters |
| `variants.sh OUTDIR build NAME ARG_V2 ARG_STYLE` | builds one variant of both drafts in a fresh copy of the w17-paper tree (V0 unchanged, Vh, Vl, Vhl, one per list line) |
| `effect.py VARDIR` | page effect of each variant against V0: total pages, AI-use statement page, labels whose page moved |
| `worddiff.py BEFORE.txt AFTER.txt [--plant] [--plant-move]` | the rendered word diff (page furniture removed, hyphen join, word diff, character-level move resolution, character multiset); positive controls both ways |
| `blind_pdf.py FILE.pdf` | formatting-level double-blind scan: /Info, XMP, every embedded figure's PTEX.FileName and PTEX.InfoDict, all decompressed streams |
| `check_stage.sh BEFORE_TREE AFTER_TREE OUTDIR` | every acceptance check on two built trees |
| `make_index_template.py BEFORE_FINAL AFTER_FINAL OUT.md` | FINAL-INDEX-template.md from two `pw1_draft_final/run_final.sh` directories, with before → after page columns |

Order used: `variants.sh` (V0, Vh, Vl, Vhl, L*), `effect.py`, `check_stage.sh V0 Vhl`, install Vhl, `pw1_draft_final/run_final.sh OUT --inplace`, `worddiff.py` on the installed PDFs, `make_index_template.py`.
