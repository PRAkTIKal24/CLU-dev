#!/usr/bin/env python3
"""w22 figures-redraw pass 2, draft edits: the captions of the three redrawn figures and two new
figure slots (fig_headline_steps after fig_ladder; fig_maze_rel_tuples after fig_maze_conditions).

Identical edits to DRAFT-v2.tex and DRAFT-v2-style.tex. Each OLD string must occur exactly once in
each draft, or nothing is written. No \\PLACEHOLDER{} argument is touched (asserted: the ordered
argument list is unchanged). Every caption number carries a % SRC: naming an artefact that
contains it (tools/check_provenance.py).
usage: apply_pass3.py <in dir> <out dir>
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from texnum import macro_args  # noqa: E402

LADDER_OLD = """\\figslot{fig_ladder}{\\textbf{The ladder at $\\Kstar$, one panel per rival set, 3 seeds.} Distinct
valid novel tuples per corner against $n$: the store beside (a)~the read-path-matched rivals fitted % SRC: _thinkpad/w4/spoke-a-rivals/rivals/seed_0/mlp.rival.json
on the 32,256-row combination-split pool and (b)~the data-matched MLP fitted on the same 32,768 % SRC: _thinkpad/w4/spoke-a-rivals/rivals/seed_0/mlp.rival.json; w12-composition-ladder/runs/shipped_seed0/rows_learned_ref_ownpool.json
rows of each rung's pool as the store's content, with the table floor in both panels
(Table~\\ref{tab:ladder}). Open markers are the seeds; the line is their mean. For $n\\ge3$ the rungs
probe the identical 120 corners, so the horizontal axis is a fit-pool coverage axis, not a % SRC: w12-composition-ladder/FINDINGS.md
depth axis.}{fig:ladder}"""

LADDER_NEW = """\\figslot{fig_ladder}{\\textbf{The ladder at $\\Kstar$, one panel per rival set, 3 seeds; every panel
title prints the rivals' fit pool and optimiser steps.} Distinct valid novel tuples per corner
against $n$: the store beside (a)~the read-path-matched rivals fitted on the 32,256-row % SRC: _thinkpad/w4/spoke-a-rivals/rivals/seed_0/mlp.rival.json
combination-split pool for 5,040 steps, (b)~the data-matched MLP fitted for 2,560 steps on the same 32,768 % SRC: _thinkpad/w4/spoke-a-rivals/rivals/seed_0/mlp.json; w12-composition-ladder/runs/shipped_seed0/rows_learned_ref_ownpool.json
rows of each rung's pool as the store's content, and (c)~the four data-matched rivals fitted on each
rung's own 131,072-row pool, the store's exact rows, at 20,480 steps, or at 204,800 where the % SRC: w20-shapes-data-matched/SHAPES-MATCHED.json
20,480-step fit had not converged (filled dots), with the table floor in every panel % SRC: w20-shapes-data-matched/SHAPES-MATCHED.json
(Table~\\ref{tab:ladder}). Open markers are the seeds; the line is their mean. The store's margin
over the strongest rival is smallest against (c) on every rung, and the store stays above every
drawn rival on every rung and seed. For $n\\ge3$ the rungs probe the identical 120 corners, so the % SRC: w12-composition-ladder/FINDINGS.md
horizontal axis is a fit-pool coverage axis, not a depth axis.}{fig:ladder}

\\figslot{fig_headline_steps}{\\textbf{The rivals were short of optimiser steps, not of data (rows
3--4, TEST, 109 corners, 3 seeds).} (a)~The MLP rival at every fit pool and step budget that was % SRC: w20-shapes-data-matched/SHAPES-MATCHED.json
run. At 20,480 steps it scores within one tuple per corner of itself on 16,384, 32,768 and 131,072 % SRC: w20-shapes-data-matched/SHAPES-MATCHED.json
rows, while the draft's comparator (16,384 rows, 1,280 steps) and the data-matched MLP (32,768 % SRC: w20-shapes-data-matched/LOG.md
rows, 2,560 steps) sit near the table floor; ten and thirty times the steps on 131,072 rows add % SRC: w20-shapes-data-matched/SHAPES-MATCHED.json
under two tuples per corner. (b)~All four families: 1,280 steps on 16,384 rows (dotted, because % SRC: w20-shapes-data-matched/LOG.md
it is another pool), then 20,480, 204,800 and, for the MLP and the continuous MLP, 614,400 steps % SRC: w20-shapes-data-matched/SHAPES-MATCHED.json; w20-shapes-data-matched/rivals/test30x/seed0/mlp/declared.json
on the store's 131,072 rows. The blue line is the store at $\\Kstar=32$ on 131,072 rows; the band % SRC: w20-shapes-data-matched/SHAPES-MATCHED.json
spans its fits on 16,384, 32,768 and 131,072 rows (47.28, 47.83 and 48.01). Large markers are seed % SRC: w20-shapes-data-matched/FINDINGS.md
means; small open markers are the seeds.}{fig:headline-steps}"""

DELETION_OLD = """\\figslot{fig_deletion}{\\textbf{Deletion on the single design at $\\Kstar$, TEST, 3 seeds.}
(a)~Composition on the corners that survive each deletion (measurement (d) of
Table~\\ref{tab:del}), before (open) and after (filled): the store before and after its downdate,
and each rival as trained and retrained without the deleted rows; the red $\\wedge$ cube scope
deletes no row and is not drawn. (b)~Adaptation seconds and (c)~data rows read, value scope: the
store's downdate reads no row; its refit-without reference and each rival's retrain-without read
every surviving row of their own fit pool (131,072 rows for the store, 16,384 for each rival). % SRC: w15-single-design-seeds/T7-3SEEDS.json
Bars are the mean of seeds 0, 1 and 2 and dots are the seeds (Tables~\\ref{tab:del}
and~\\ref{tab:rivals-del-seeds}).}{fig:deletion}"""

DELETION_NEW = """\\figslot{fig_deletion}{\\textbf{Deletion on the single design at $\\Kstar$, the four measurements,
rows 3--4, TEST, fit pool 131,072 rows, seeds 0, 1 and 2.} Forgetting a value or a conjunction % SRC: w20-shapes-data-matched/SHAPES-MATCHED.json
downdates the store, downdates the address book's statistic and refits the book from it, and
punches the hole; unlike the (d) rows of Table~\\ref{tab:del}, which carry the book across the basis
change, the book here is refit. (a)~The store against the store refit without the rows (largest
mode deviation) and (b)~the refit book against the book refit without the rows (relative deviation
of the rendered features; the book's count and first-moment statistic deviate by exactly 0), per
scope and seed. (c)~Refusal through the hole: a display, never a result. (d)~Composition on the
surviving corners before and after, and (e)~the same as a paired change with its 95\\% bootstrap % SRC: w20-shapes-data-matched/SHAPES-MATCHED.json
interval. The value deletion moves the store by under one tuple per corner; the red $\\wedge$ sphere
deletion lowers it by 6.13, 6.43 and 6.30 on seeds 0, 1 and 2; deleting a random 1\\% of items, the % SRC: w20-shapes-data-matched/FINDINGS.md
control for refit instability, moves it by 0.02, 0.11 and 0.01. The conjunction loss is the deleted % SRC: w20-shapes-data-matched/FINDINGS.md
content, not the refit. (f)~The store after deletion minus each rival retrained without the same
rows of the same pool, with 95\\% intervals; on red $\\wedge$ sphere the interval crosses zero against % SRC: w20-shapes-data-matched/SHAPES-MATCHED.json
the MLP at seeds 1 and 2 and against the LSTM at seed 2.}{fig:deletion}"""

MAZE_OLD_SRC = """% SRCBLOCK: w16-maze-wm/t1/BARS-A1-test.json; w19-maze-t-symmetric/val/BARS-WORDS-MAIN-val.json; w21-words-full-test/test/BARS-WORDS-FULL-test.json; w19-maze-t-symmetric/val/BARS-OBS-val.json; w19-maze-t-symmetric/test/BARS-BOTH-test.json; w21-words-full-test/PREREG-w21-words-full-test.md; w15-cmaze-t-harness/FINDINGS.md; w21-words-full-test/TABLES.md; w19-maze-t-symmetric/FINDINGS.md
\\figslot{fig_maze_conditions}{\\textbf{CMAZE-T, one row per input condition, unseen $n$-way, 3 seeds
per row.} Validity under the exact judge at each seed for the store (circles) and for the"""

MAZE_NEW_SRC = """% SRCBLOCK: w16-maze-wm/t1/BARS-A1-test.json; w19-maze-t-symmetric/val/BARS-WORDS-MAIN-val.json; w21-words-full-test/test/BARS-WORDS-FULL-test.json; w19-maze-t-symmetric/val/BARS-OBS-val.json; w19-maze-t-symmetric/test/BARS-BOTH-test.json; w21-words-full-test/PREREG-w21-words-full-test.md; w15-cmaze-t-harness/FINDINGS.md; w21-words-full-test/TABLES.md; w19-maze-t-symmetric/FINDINGS.md; w21-maze-t-clu-improve/test/LABEL-BOTH-test.json; w21-rel-newseeds/REL-NEWSEEDS.json; w21-rel-newseeds/TABLES.md; w21-maze-t-seed4/MAZE-T-SEED4.json
\\figslot{fig_maze_conditions}{\\textbf{CMAZE-T, one row per input condition, unseen $n$-way; (a)~3 seeds
per row, (b), (c)~six seeds and two configurations.} (a)~Validity under the exact judge at each seed for the store (circles) and for the"""

MAZE_OLD_END = """rows are validation; the others are TEST. On words plus observation the rule reads \\textsc{win},
although that condition's pre-registered margin bar was not met. No row is drawn as a margin
bar.}{fig:maze-wm}
% ENDSRC"""

MAZE_NEW_END = """rows are validation; the others are TEST. On words plus observation (A1) the rule reads \\textsc{win},
although that condition's pre-registered margin bar was not met. The last row is the CLU-REL-A10
configuration (colour-relational request coding) on seeds 0--2. No row is drawn as a margin bar.
(b), (c)~Words plus observation, TEST, seeds 0--5, for A1 and for CLU-REL-A10: at each seed the
store, the best of the six matched rivals on that seed, for CLU-REL-A10 the best rival handed the
same relational request columns, the highest control, and the ceiling (the best arm fitted to the
training data; in (c) including the relational rivals). Seeds 1 and 4 are hard: A1's ceiling on
them falls below 0.6407, the largest gap in A1's per-seed ceilings, a split made by the ceiling and
never by the store's own score. The relational coding raises the store on every seed. Handed the
same columns, the best rival reaches 0.915, 0.869, 0.819, 0.921, 0.900 and 0.939 on seeds 0 to 5
against the store's 0.908, 0.894, 0.984, 0.927, 0.900 and 0.939: the gain comes from the request
coding, not from the store. On seed 4 the random-atoms control also equals the store exactly
(0.9004, on every held-out tuple, Figure~\\ref{fig:maze-rel-tuples}); on seed 5 the best matched
rival equals it. Every seed where a rival or a control reaches the store is marked. The six-seed
panels carry no claim label.}{fig:maze-wm}
% ENDSRC

\\figslot{fig_maze_rel_tuples}{\\textbf{CMAZE-T, CLU-REL-A10, words plus observation, TEST: every
held-out $n$-way tuple, seeds 0--5.} Each cell is the fraction of the tuple's rows judged valid by
the exact judge, for the store, the seed's best matched rival, the best rival handed the relational
request columns and the highest control; each row names its arm, and the count on the right is the
number of tuples on which that arm's valid and total rows equal the store's. On seed 4 the % SRC: w21-rel-newseeds/FINDINGS.md
random-atoms control and the best relational rival equal the store on 16 of 16 tuples; on seed 5 % SRC: w21-rel-newseeds/FINDINGS.md
the best matched rival (a transformer at 80,000 steps) and the best relational rival do. Counts % SRC: w21-rel-newseeds/test/BOTH_test_seed5.json
come from the zero-fit replay of the persisted objects, which are hash-identical to the objects
that were scored.}{fig:maze-rel-tuples}"""

EDITS = [(LADDER_OLD, LADDER_NEW), (DELETION_OLD, DELETION_NEW), (MAZE_OLD_SRC, MAZE_NEW_SRC),
         (MAZE_OLD_END, MAZE_NEW_END)]


def main():
    src, dst = sys.argv[1], sys.argv[2]
    os.makedirs(dst, exist_ok=True)
    for name in ("DRAFT-v2.tex", "DRAFT-v2-style.tex"):
        t = open(os.path.join(src, name)).read()
        ph0 = macro_args(t, "PLACEHOLDER")
        for old, new in EDITS:
            n = t.count(old)
            assert n == 1, (name, n, old[:80])
            t = t.replace(old, new)
        assert macro_args(t, "PLACEHOLDER") == ph0, "PLACEHOLDER arguments changed"
        open(os.path.join(dst, name), "w").write(t)
        print(name, "ok;", len(ph0), "PLACEHOLDERs unchanged")


if __name__ == "__main__":
    main()
