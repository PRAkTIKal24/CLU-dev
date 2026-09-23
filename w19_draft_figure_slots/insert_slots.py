#!/usr/bin/env python3
"""w19 draft-figure-slots: insert float environments (figures + one placeholder table) into
DRAFT-v2.tex and DRAFT-v2-style.tex at label anchors. Pure insertion: no existing line is
changed or removed (asserted). Identical bytes go into both files, so the four style diffs
stay empty.  usage: insert_slots.py FILE [FILE ...]  (edits in place)."""
import sys

W19SF = '../../outputs/w17-paper/figures/rendered'   # relative to EV (~/clu/evidence/outputs/)
W19TF = '../../outputs/w19-task-figures'
W19SFR = '../../outputs/w19-success-failure-panels'

TASK = r"""
% ---- w19 draft-figure-slots: the two TASK figures, placed before the first results section ----
% SRCBLOCK: %(TF)s/TASK-FIGURES.md; %(TF)s/proof/PROVENANCE-task-figures.json
% SOURCE ARTEFACT: figures/fig_task_shapes.pdf (w19 task-figures; 3dshapes.h5 at the runbook pin;
%   value_holdout/{factor_index.npz,value_cells.json}; rendered/REQUESTS.json requests[0];
%   frozen decoder seed 0, param_hash 01ef91f7...; external classifier, param_hash c8741ac4...). No result in it.
\figslot{fig_task_shapes}{\textbf{The Shapes3D task.} (a) The six factors (floor hue, wall hue,
object hue, scale, shape, orientation), each row varying one factor of real dataset images with the
other five held at the request's values. (b) A request is a tuple of factor values; the answer is an
image rendered by the one frozen decoder, and the exact judge is the hash-pinned external classifier,
which must read back all six factors. The image shown is the frozen decoder on the encoder's own
feature of that row: it shows the answer format, not any arm's output. (c) The two-way hold-out on
object hue $\times$ shape: grey cells have rows in the scored fit pool; the outlined cells are pair
corners, whose values are each seen but never together; the hatched value (the held-out object hue
and the cube) is absent from the fit pool altogether. (d) A six-factor corner is absent from the fit
pool although every two-factor projection of it is present (top, left); only the joint slice (bottom)
shows the hole. No score of any arm appears in this figure.}{fig:task-shapes}
% SOURCE ARTEFACT: figures/fig_task_cmazet.pdf (w19 task-figures; CMAZE-T engine: w8 maze_colour +
%   w9 dynamics.step + chlu/exp_w15_cmaze_t.py at 9f80f59; one generated TRAIN-bank maze, seed 0). No result in it.
\figslot{fig_task_cmazet}{\textbf{The CMAZE-T task.} (a) One generated maze, drawn by the engine:
walls, floor tinted in one of four colours (each mark carries its colour's digit), keys, closed and
opened doors, the agent with the key in its pocket (at most one), the goal, and the step budget.
Rule A: a door of colour $c$ needs key $c$ and consumes it. Rule B: floor tinted $c$ is forbidden
while holding key $c$. Rule A is applied before rule B. (b)--(d) One transition each: the request is
five words (\texttt{w\_agent}, \texttt{w\_key}, \texttt{w\_door}, \texttt{w\_tint}, \texttt{w\_dir})
and the answer is four facts (\texttt{legal}, \texttt{disp}, \texttt{event}, \texttt{pocket}),
computed by the engine and checked against an independent stepper. (b) A legal move onto floor of
another tint. (c) An illegal move under rule B: the agent does not move, so after equals before and
the verdict is carried by the mark. (d) The same pocket and tint as (c) with a closed door of the
pocket's colour: the door consumes the key first, so the move is legal. Panels (c) and (d) differ
only in the door, which is the pocket $\times$ door $\times$ tint interaction held out on this
instrument. No score of any arm appears in this figure.}{fig:task-cmazet}
% ENDSRC
"""

T7 = r"""
% ---- w19 draft-figure-slots: T7 success / failure panels, beside Table tab:t7 (w<=14 evidence) ----
% SRCBLOCK: %(SF)s/REQUESTS-w19.json; %(SF)s/SAME-DECODER-w19.json; %(SF)s/README-w19.md; %(SFR)s/REPORT.md
% SOURCE ARTEFACT: figures/rendered/rc_w19_t7_per_arm_success_failure.pdf (w19 success-failure-panels;
%   re-render of the w12 / w12b T7 legs, every per-corner n_valid equal to their scored.json).
\figslot{rendered/rc_w19_t7_per_arm_success_failure}{\textbf{Each arm succeeding and failing,
Table~\ref{tab:t7}, seed 0, $\Kstar=32$.} For each arm (bracket and coloured frame at left) one
held-out request on which that arm renders validly and one on which it does not, each drawn
uniformly with a fixed seed among the qualifying requests. Every arm, and the lookup floor, is
rendered on every request through one frozen decoder, asserted identical across panels; the mark is
the external judge's verdict. The store's own failure is shown in its second row.}{fig:t7-per-arm}
% SOURCE ARTEFACT: figures/rendered/rc_w19_t7_rival_succeeds_clu_fails.pdf (seed 0)
\figslot{rendered/rc_w19_t7_rival_succeeds_clu_fails}{\textbf{Requests on which a rival succeeds
and the store fails exist on every seed (Table~\ref{tab:t7}).} Counted over all 5,688 held-out
requests per seed, a learned rival renders validly where the store does not on 30 / 24 / 30 requests
at seeds 0 / 1 / 2 (MLP 26 / 11 / 3, LSTM 13 / 13 / 25, transformer 2 / 3 / 6, continuous MLP
0 / 1 / 0; the store is valid on 5,177 / 5,167 / 5,354); the lookup floor, which is not a rival, does
so on 67 / 26 / 46. One such request per rival is shown at seed 0 (MLP, LSTM, transformer); the store's
panel carries the blue frame and a failing mark on every row.}{fig:t7-rival-wins}
% SOURCE ARTEFACT: figures/rendered/rc_w19_t7_rival_succeeds_clu_fails_seed1.pdf (seed 1, seed 1's own frozen decoder)
\figslot{rendered/rc_w19_t7_rival_succeeds_clu_fails_seed1}{\textbf{The continuous MLP's only
request, on any seed, on which it renders validly and the store does not} (seed 1, rendered through
seed 1's frozen decoder; Figure~\ref{fig:t7-rival-wins}).}{fig:t7-rival-wins-s1}
% ENDSRC
"""

MAZE = r"""
% ---- w19 draft-figure-slots: CMAZE-T engine-only panels, beside Table tab:cmazet ----
% These figures carry NO arm output: w16 persisted no codec, store, control or rival object and no
% per-transition prediction, so the arm columns are hatched and each caption carries a PLACEHOLDER.
% They share no panel with a w16 row and no w19 maze-t-symmetric condition.
% SRCBLOCK: %(SF)s/REQUESTS-w19.json; %(SF)s/README-w19.md; %(SFR)s/REPORT.md; %(SFR)s/NEEDS-ARTEFACTS.json
% SOURCE ARTEFACT: figures/rendered/rc_w19_cmazet_families.pdf (w19 success-failure-panels; engine re-derivation
%   of the scored TEST leg's own rows, bank meta and hold-outs equal to w16-maze-wm/t1/A1_test_seed{0,1,2}.json)
\figslot{rendered/rc_w19_cmazet_families}{\textbf{Held-out CMAZE-T transitions, one per rule family
(engine only).} For each family ($n$-way TEST hold-out, seed 0 unless marked): $s_t$ with the action,
and the engine's true $s_{t+1}$ with its legal or illegal verdict. The \texttt{rule\_a} family has
almost no TEST $n$-way rows and is shown from seed 2. The arm columns are hatched: no arm's output
is drawn. \PLACEHOLDER{w16 CMAZE-T per-arm generated $s_{t+1}$ (store, transformer, MLP, LSTM) with
exact-judge marks, one success and one failure per arm: not buildable from persisted artefacts (a
persistence re-run would be a new scored run); may remain undrawn, and if drawn may show the store
failing where a rival succeeds}}{fig:cmazet-families}
% SOURCE ARTEFACT: figures/rendered/rc_w19_cmazet_door_then_tint.pdf
\figslot{rendered/rc_w19_cmazet_door_then_tint}{\textbf{Door-then-tint (\texttt{interaction})
transitions, one per move direction present in the seed-0 hold-out (engine only).} $s_t$ with the
action and the engine's $s_{t+1}$; the arm columns are hatched. \PLACEHOLDER{w16 CMAZE-T per-arm
generated $s_{t+1}$ on the \texttt{interaction} family, with exact-judge marks: not buildable from
persisted artefacts; may remain undrawn, and if drawn may show every arm, the store included,
failing the interaction}}{fig:cmazet-door-tint}
% SOURCE ARTEFACT: figures/rendered/rc_w19_cmazet_deletion.pdf
\figslot{rendered/rc_w19_cmazet_deletion}{\textbf{A transition on the deleted colour (colour 1, engine
only).} $s_t$ with the action and the engine's $s_{t+1}$; the columns for the store before and after
the downdate, and for each rival as trained and retrained without, are hatched.
\PLACEHOLDER{w16 CMAZE-T post-deletion per-arm generated $s_{t+1}$ (store before / after downdate,
each rival as trained / retrained without): not buildable from persisted artefacts; may remain
undrawn, and if drawn may show the downdated store failing where a retrained rival
succeeds}}{fig:cmazet-deletion}
% ENDSRC

% ---- w19 draft-figure-slots: w19 maze-t-symmetric rows, NOT LANDED. Own table, never merged with
% Table tab:cmazet (w16). Every row a PLACEHOLDER that names it and says it may land as a negative.
\begin{table}[h]
\centering\scriptsize
\caption{\textbf{CMAZE-T with symmetric inputs (wave 19, \texttt{maze-t-symmetric}): pending.} The % SRC: ../../tasks/w19/maze-t-symmetric.md
instrument, banks, hold-outs, bars, controls, floors and the store configuration are those of
Table~\ref{tab:cmazet}; only the input changes, and within a condition every arm receives the
identical input. Every row is pending and may land as a negative; the table is printed in full
whatever the outcome. This table contains no row of Table~\ref{tab:cmazet} and no CMAZE-O,
CMAZE-1, CMAZE-2 or CMAZE-2b row.}
\label{tab:cmazet-sym}
\setlength{\tabcolsep}{2pt}
\begin{adjustbox}{max width=\linewidth}\begin{tabular}{p{1.9cm}p{1.2cm}p{10.0cm}}
\toprule
condition & seeds & CMAZE-T, symmetric inputs \\
\midrule
% SRCBLOCK: ../../tasks/w19/maze-t-symmetric.md
WORDS-MAIN & 3 & \PLACEHOLDER{w19 \texttt{maze-t-symmetric} WORDS-MAIN: the main-effect word one-hots for the store and every rival, which is also the learned blind-sequential reading; unseen $n$-way (primary), seen, unseen 2-way, value hold-out and per-step exactness, with the charter-C controls (across-wells control a derangement); may land as a negative: the store may tie or lose} \\
WORDS-FULL & 3 & \PLACEHOLDER{w19 \texttt{maze-t-symmetric} WORDS-FULL: the full pair design given to the store and to every rival at matched read floats, deciding whether the pair block or the store carries the margin; same rows and controls; may land as a negative: if the rivals match the store, the margin belongs to the interface} \\
OBS & 3 & \PLACEHOLDER{w19 \texttt{maze-t-symmetric} OBS: observation features and dynamics scalars for every arm, the store reading through a declared word extractor whose accuracy is printed on every row, and a second rival row with the extractor's output appended; may land as a negative} \\
BOTH & 3 & \PLACEHOLDER{w19 \texttt{maze-t-symmetric} BOTH (words and observation): validation for all arms, scored only if runs remain; may be a declared not-run or a negative} \\
rival budget & 3 & \PLACEHOLDER{w19 \texttt{maze-t-symmetric}: every condition's rivals at the declared training budget with the ten-times diagnostic column; may withdraw a margin, as on CMAZE-O} \\
deletion & 3 & \PLACEHOLDER{w19 \texttt{maze-t-symmetric}: deletion rows re-run on the winning condition; may be a declared not-run if no condition wins} \\
rollout validity & 3 & \PLACEHOLDER{w19 \texttt{maze-t-symmetric}: rollout validity, printed once as a limitation; may again be zero for every arm} \\
% ENDSRC
\bottomrule
\end{tabular}\end{adjustbox}
\end{table}
"""

TASK, T7, MAZE = (b.replace('%(TF)s', W19TF).replace('%(SF)s', W19SF).replace('%(SFR)s', W19SFR)
                  for b in (TASK, T7, MAZE))

def after(lines, i, block):
    return lines[:i + 1] + block.strip('\n').split('\n') + [''] + lines[i + 1:]

for path in sys.argv[1:]:
    src = open(path).read()
    assert 'w19 draft-figure-slots' not in src, 'already inserted: ' + path
    L = src.split('\n')
    orig = list(L)
    # anchor 3 first (bottom-up keeps earlier indices valid)
    k = L.index(r'\label{tab:cmazet}\label{tab:cmaze-del}')
    e3 = next(j for j in range(k, len(L)) if L[j] == r'\end{table}')
    k = L.index(r'\label{tab:t7}')
    e2 = next(j for j in range(k, len(L)) if L[j] == r'\end{table}')
    k = L.index(r'\label{sec:object}')
    assert L[k + 1].startswith('% ====='), L[k + 1]
    e1 = k + 1
    L = after(L, e3, '\n' + MAZE)
    L = after(L, e2, '\n' + T7)
    L = after(L, e1, '\n' + TASK)
    # pure insertion: the original lines are a subsequence, in order, of the new file
    it = iter(L)
    assert all(any(x == y for y in it) for x in orig), 'not a pure insertion'
    open(path, 'w').write('\n'.join(L))
    print(path, len(orig), '->', len(L), 'lines; anchors', e1, e2, e3)
