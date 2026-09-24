#!/usr/bin/env python3
"""w21 figures-redraw, draft pass 1 (Parts A, B, D): captions of the redrawn figures, the maze
figure slot, the placeholder index's figure paragraph and the stale header comment.

Identical edits to DRAFT-v2.tex and DRAFT-v2-style.tex (figure slots and captions are verbatim in
both). Each OLD string must occur exactly once in each draft, or nothing is written. No
\\PLACEHOLDER{} argument is touched (asserted: the ordered argument list is unchanged).
usage: apply_pass1.py <in dir> <out dir>
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from texnum import macro_args  # noqa: E402

EDITS = [
    # ---- stale header comment (the in-text index is the count)
    ("%     coverage-control TODO. Eleven PLACEHOLDERs remain: ten w19 (w19-draft-figure-slots/FIGURE-SLOTS.md), one w20.\n",
     "%     coverage-control TODO. The PLACEHOLDER count is the in-text placeholder index's (app:placeholders);\n"
     "%     w21 figures-redraw replaced a stale count here.\n"),
    # ---- fig_ladder
    ("""\\figslot{fig_ladder}{\\textbf{The ladder at $\\Kstar$ (placeholder figure, Appendix~\\ref{app:placeholders}).} (a) Score
against $n$ per arm, 3 seeds, with the read-path rivals and the table floor. (b) Margin of the store
over a learner matched to the object's own four-part total, i.e.\\ the stress comparison of
Appendix~\\ref{app:1x}, \\emph{not} the paper's read-path matching rule. For $n\\ge3$ the
horizontal axis is a training-data axis, not a depth axis.}{fig:ladder}""",
     """\\figslot{fig_ladder}{\\textbf{The ladder at $\\Kstar$, one panel per rival set, 3 seeds.} Distinct
valid novel tuples per corner against $n$: the store beside (a)~the read-path-matched rivals fitted % SRC: _thinkpad/w4/spoke-a-rivals/rivals/seed_0/mlp.rival.json
on the 32,256-row combination-split pool and (b)~the data-matched MLP fitted on the same 32,768 % SRC: _thinkpad/w4/spoke-a-rivals/rivals/seed_0/mlp.rival.json; w12-composition-ladder/runs/shipped_seed0/rows_learned_ref_ownpool.json
rows of each rung's pool as the store's content, with the table floor in both panels
(Table~\\ref{tab:ladder}). Open markers are the seeds; the line is their mean. For $n\\ge3$ the rungs
probe the identical 120 corners, so the horizontal axis is a fit-pool coverage axis, not a % SRC: w12-composition-ladder/FINDINGS.md
depth axis.}{fig:ladder}"""),
    # ---- fig_crossover
    ("""\\figslot{fig_crossover}{\\textbf{The $K$-curve.} Validation composition and the honest four-part
ledger (symmetric-triangle convention) against $K$; the number of seeds is printed per point. Placeholder
figure (Appendix~\\ref{app:placeholders}), drawn from the single-seed middle rows, so its flat
seed-0 line is not the three-seed result: three seeds do not resolve the validation plateau
(\\S\\ref{sec:shapes}); the three-seed curve is Table~\\ref{tab:kcurve3}.}{fig:crossover}""",
     """\\figslot{fig_crossover}{\\textbf{The $K$-curve, validation, 3 seeds at every $K$.} (a)~Composition
against $K$: the mean over seeds 0, 1 and 2 with a band of one seed standard deviation, each seed
marked, and the highest of the four charter-C controls. (b)~The honest four-part ledger
(symmetric-triangle convention), the seed mean stacked by part and never netted, with the seed
standard deviation of the total; 524,288~B is a reference line, not a budget. Three seeds do not % SRC: w15-crossover-seeds/CROSSOVER-3SEED.md
resolve the validation plateau (\\S\\ref{sec:shapes}); the values are those of
Table~\\ref{tab:kcurve3}.}{fig:crossover}"""),
    # ---- fig_deletion
    ("""\\figslot{fig_deletion}{\\textbf{Deletion on the single design at $\\Kstar$.} Survivor composition before and after deletion (measurement (d) of Table~\\ref{tab:del}) and adaptation cost per arm (store: downdate and carry, 0 rows read; rivals: retrain without). Placeholder figure (Appendix~\\ref{app:placeholders}), seed 0 only; the seed-1--2 rows are in Table~\\ref{tab:del} and Table~\\ref{tab:rivals-del-seeds}.}{fig:deletion}""",
     """\\figslot{fig_deletion}{\\textbf{Deletion on the single design at $\\Kstar$, TEST, 3 seeds.}
(a)~Composition on the corners that survive each deletion (measurement (d) of
Table~\\ref{tab:del}), before (open) and after (filled): the store before and after its downdate,
and each rival as trained and retrained without the deleted rows; the red $\\wedge$ cube scope
deletes no row and is not drawn. (b)~Adaptation seconds and (c)~data rows read, value scope: the
store's downdate reads no row; its refit-without reference and each rival's retrain-without read
every surviving row of their own fit pool (131,072 rows for the store, 16,384 for each rival). % SRC: w15-single-design-seeds/T7-3SEEDS.json
Bars are the mean of seeds 0, 1 and 2 and dots are the seeds (Tables~\\ref{tab:del}
and~\\ref{tab:rivals-del-seeds}).}{fig:deletion}"""),
    # ---- the maze results figure becomes the condition table (Part B); label kept
    ("""\\figslot{fig_maze_wm}{\\textbf{CMAZE-T, Config A1, TEST, 3 seeds (Table~\\ref{tab:cmazet}).} (a) Unseen
$n$-way: the store against the matched rivals, the four pre-declared landscape controls and the cross-maze
lookup floor, per seed. (b) Rollout validity is zero for the store and every matched rival; the
per-step exactness that compounds over the chained steps is drawn beside it. (c) Post-deletion
composition on the surviving colours: the downdated store (equal to the store fitted without)
against the rival retrained without, value and conjunction scopes. CMAZE-T only.}{fig:maze-wm}""",
     """% SRCBLOCK: w16-maze-wm/t1/BARS-A1-test.json; w19-maze-t-symmetric/val/BARS-WORDS-MAIN-val.json; w21-words-full-test/test/BARS-WORDS-FULL-test.json; w19-maze-t-symmetric/val/BARS-OBS-val.json; w19-maze-t-symmetric/test/BARS-BOTH-test.json; w21-words-full-test/PREREG-w21-words-full-test.md; w15-cmaze-t-harness/FINDINGS.md; w21-words-full-test/TABLES.md; w19-maze-t-symmetric/FINDINGS.md
\\figslot{fig_maze_conditions}{\\textbf{CMAZE-T, one row per input condition, unseen $n$-way, 3 seeds
per row.} Validity under the exact judge at each seed for the store (circles) and for the
condition's best matched rival, the matched rival with the highest three-seed mean over every
family and both training budgets; a tick marks each three-seed mean. One rule labels every row
from its artefact: \\textsc{win} if the store is ahead of that rival on the mean and on at least two
of the three seeds and every charter-C control's mean sits at least 0.15 below the store's (the
store gate); otherwise \\textsc{comparable} if the mean paired difference is within one seed's
spread (the sample standard deviation of the three per-seed differences); otherwise \\textsc{loss}
if the store is behind. The first row is not like-for-like and carries no label: its rivals read a
636-dimensional observation, the store the 318-wide word design. The main-effects and observation
rows are validation; the others are TEST. On words plus observation the rule reads \\textsc{win},
although that condition's pre-registered margin bar was not met. No row is drawn as a margin
bar.}{fig:maze-wm}
% ENDSRC"""),
    # ---- placeholder index: the figure paragraph
    ("""Three figures are placeholder figures in addition to the eleven rows, and each carries an in-image
placeholder banner: Figure~\\ref{fig:ladder} (three seeds, but without the data-matched MLP of
Table~\\ref{tab:ladder}), Figure~\\ref{fig:crossover} (the single-seed curve, which three seeds do not
support as a selection) and Figure~\\ref{fig:deletion} (seed 0 only). The data each awaits has
landed (Tables~\\ref{tab:ladder}, \\ref{tab:kcurve3} and~\\ref{tab:del}); each is to be redrawn from
it, with the banner removed, before submission.""",
     """No figure is a placeholder figure and none carries an in-image banner: Figures~\\ref{fig:ladder},
\\ref{fig:crossover} and~\\ref{fig:deletion} are drawn from the three-seed artefacts of
Tables~\\ref{tab:ladder}, \\ref{tab:kcurve3} and~\\ref{tab:del}, and Figure~\\ref{fig:ladder} takes one
panel per data-matched rival set as each set lands."""),
]


def main():
    src, dst = sys.argv[1], sys.argv[2]
    os.makedirs(dst, exist_ok=True)
    for name in ("DRAFT-v2.tex", "DRAFT-v2-style.tex"):
        t = open(os.path.join(src, name), encoding="utf-8").read()
        before = macro_args(t, "PLACEHOLDER")
        for old, new in EDITS:
            n = t.count(old)
            assert n == 1, f"{name}: expected 1 occurrence, found {n}: {old[:80]!r}"
            t = t.replace(old, new)
        assert macro_args(t, "PLACEHOLDER") == before, "a PLACEHOLDER argument changed"
        orig = open(os.path.join(src, name), encoding="utf-8").read()
        assert sum(ord(c) > 127 for c in t) == sum(ord(c) > 127 for c in orig), "new non-ASCII"
        open(os.path.join(dst, name), "w", encoding="utf-8").write(t)
        print(name, "OK", len(EDITS), "edits; placeholders", len(before))


if __name__ == "__main__":
    main()
