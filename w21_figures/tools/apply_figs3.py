#!/usr/bin/env python3
"""w22 figures-redraw pass 3, draft edits.

1. fig:task-cmazet: the agent is now the result panels' glyph (black disc, centre = pocket colour,
   white = empty); keys are diamonds, the goal a cross; the step budget is no longer drawn.
2. fig:cmazet-families / -per-arm / -door-tint / -deletion: what the tick means (the judge's verdict
   on all four emitted facts) and the fact strip under every s_{t+1} that makes legality visible.
3. New slot fig_cmazet_rel_per_arm (CLU-REL-A10) after fig_cmazet_per_arm (gate 2 passed).

Identical edits to DRAFT-v2.tex and DRAFT-v2-style.tex. Each OLD string must occur exactly once in
each draft, or nothing is written. No \\PLACEHOLDER{} argument is touched (asserted). Every new
caption number has an artefact on its % SRC / SRCBLOCK line (tools/check_provenance.py).
usage: apply_figs3.py <in dir> <out dir>
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from texnum import macro_args  # noqa: E402

FIGD = "../../outputs/w17-paper/figures/data/"

TASK_OLD = """walls, floor tinted in one of four colours (each mark carries its colour's digit), keys, closed and
opened doors, the agent with the key in its pocket (at most one), the goal, and the step budget."""
TASK_NEW = """walls, floor tinted in one of four colours (each mark carries its colour's digit), keys (diamonds),
closed and opened doors, the goal (a cross), and the agent: a black disc whose centre shows the key in
its pocket, white when the pocket is empty (it holds at most one key). Every maze cell is drawn by the
renderer of the result panels (Figures~\\ref{fig:cmazet-families}--\\ref{fig:cmazet-deletion}), so
the symbols here are the symbols there."""

TASK_C_OLD = """another tint. (c) An illegal move under rule B: the agent does not move, so after equals before and
the verdict is carried by the mark."""
TASK_C_NEW = """another tint. (c) An illegal move under rule B: the agent does not move, so after equals before and
the verdict is carried by the mark and by \\texttt{legal}~$=0$."""

SRCBLOCK_OLD = """% SRCBLOCK: w19-maze-t-symmetric/test/BOTH_test_seed0.json; w19-maze-t-symmetric/test/BOTH_test_seed1.json; w19-maze-t-symmetric/test/BOTH_test_seed2.json; ../../outputs/w17-paper/figures/data/cmazet_items.REPRO.json; ../../outputs/w17-paper/figures/data/cmazet_panels.json; w19-maze-t-symmetric/FINDINGS.md"""
SRCBLOCK_NEW = SRCBLOCK_OLD + ("; " + FIGD + "CMAZET-AMBIGUITY.json; " + FIGD
                               + "cmazet_items_rel.REPRO.json; "
                               "w21-maze-t-clu-improve/test/BOTH_test_seed0.json")

FAM_OLD = """learned blind arms. Every panel is re-read from the condition's persisted objects, and the re-read
reproduces every arm's scored TEST score on every seed."""
FAM_NEW = """learned blind arms. The picture is the state an arm's emission produces, but the judge scores the four
emitted facts (\\texttt{legal}, \\texttt{disp}, \\texttt{event}, \\texttt{pocket}), and the state
need not show the emitted legality or the emitted pocket. So the strip under every $s_{t+1}$ prints the
emitted facts (the engine column prints the truth), and the tick or cross is the judge's verdict on
all four. In the \\texttt{rule\\_a} row the store emits a legal move that goes nowhere and is marked
wrong; the MLP emits an illegal move and is marked right; only the strip tells the two panels apart.
Over every held-out transition of the three seeds, 22,340 pairs of drawn arms produce the same picture
with opposite marks; with the strip, none do. Every panel is re-read from the condition's persisted objects, and the re-read
reproduces every arm's scored TEST score on every seed."""

PERARM_OLD = """only a failure. Columns: $s_t$ with the action, the engine's $s_{t+1}$, and the arm's generated
$s_{t+1}$ with the exact judge's mark.}{fig:cmazet-per-arm}"""
PERARM_NEW = """only a failure. Columns: $s_t$ with the action, the engine's $s_{t+1}$, and the arm's generated
$s_{t+1}$ with the exact judge's mark; under each $s_{t+1}$, its emitted facts, as in
Figure~\\ref{fig:cmazet-families}.}{fig:cmazet-per-arm}
% SOURCE ARTEFACT: figures/fig_cmazet_rel_per_arm.pdf (cmazet_extract_rel.py -> data/cmazet_items_rel.json)
\\figslot{fig_cmazet_rel_per_arm}{\\textbf{Each CMAZE-T arm succeeding and failing under the
colour-relational request coding (CLU-REL-A10; words plus observation, TEST, seed 0).} As
Figure~\\ref{fig:cmazet-per-arm}, for the CLU-REL-A10 store and its arms, and for the six rivals
handed the store's relational request columns (MLP, LSTM and transformer at 8,000 and 80,000
steps), a diagnostic and not a matched rival. Every panel is re-read from this configuration's
persisted objects; the re-read reproduces all 132 scored TEST arm scores exactly, over seeds 0--5.
Under each $s_{t+1}$, its emitted facts, as in Figure~\\ref{fig:cmazet-families}.}{fig:cmazet-rel-per-arm}"""

DOOR_OLD = """Columns as in Figure~\\ref{fig:cmazet-families}. For the arms of Table~\\ref{tab:cmazet}:"""
DOOR_NEW = """Columns, and the emitted facts under each $s_{t+1}$, as in Figure~\\ref{fig:cmazet-families}. For
the arms of Table~\\ref{tab:cmazet}:"""

DEL_OLD = """same outcome on every surviving row, every seed), and the matched MLP as trained and retrained
without. For the arms of Table~\\ref{tab:cmazet}:"""
DEL_NEW = """same outcome on every surviving row, every seed), and the matched MLP as trained and retrained
without; under each $s_{t+1}$, its emitted facts, as in Figure~\\ref{fig:cmazet-families}. For the
arms of Table~\\ref{tab:cmazet}:"""

EDITS = [(TASK_OLD, TASK_NEW), (TASK_C_OLD, TASK_C_NEW), (SRCBLOCK_OLD, SRCBLOCK_NEW),
         (FAM_OLD, FAM_NEW), (PERARM_OLD, PERARM_NEW), (DOOR_OLD, DOOR_NEW), (DEL_OLD, DEL_NEW)]


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
