# pw1-plot-shapes LOG
- 11:20Z start; read COMMON, brief, PLAN §1 (items 2,3,10,11,12,13). dataviz skill loaded (existing validated palette kept: no colour change is in scope).
- Final stems come from: fig_crossover = make_figures_w21.fig_crossover; fig_ladder, fig_deletion, fig_headline_steps = make_figures_w22; fig_encoder_adapt = make_figures.py. The make_figures.py versions of crossover/ladder/deletion are overwritten by step 2 (never installed).
- (~11:25Z) code done (ladder 1 panel, crossover labels, deletion 2 panels, headline/encoder vocab, readverb/requestlayer dispatch removed, check_vocab_pw1). full build.sh to scratch: exit 0, BYTE-IDENTICAL, maze figs cmp-equal to pw1-plot-maze/build/a, no retired stem. build/a,b byte-identical.
- (~11:27Z) CAPTIONS.tex, DELETION-FACTS.tex written; banned-word grep on non-comment text clean (the one hit is \sci under -i, a false positive). GAP: '1e-8' in the brief does not hold (content 1.07e-08..5.38e-07).
- (~11:28Z) committed e7e418e; bundle created + verified
- (~11:29Z) FINDINGS.md written; finishing
- note: the times in parentheses were corrected; the first-written values were guesses that ran ahead of the clock. File mtimes are authoritative.
