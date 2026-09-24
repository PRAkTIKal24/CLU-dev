# -*- coding: utf-8 -*-
"""New text for apply_rulings.py. Every number carries a % SRC pointer to the artefact it was read
from (checked by check_provenance.py). Paths are relative to ~/clu/evidence/outputs/."""

W20M = 'w20-shapes-data-matched/SHAPES-MATCHED.md'
W20J = 'w20-shapes-data-matched/SHAPES-MATCHED.json'
W20F = 'w20-shapes-data-matched/FINDINGS.md'

ABSTRACT_SHAPES = r'''
% w22 draft-rulings (Ruling 2): the earlier headline margins, against comparators fitted on 16,384 rows for
% 1,280 optimiser steps, are retired from the rendered text; they remain in w12b-nr6-and-kstar/NR6-T7.md and in
% w20-shapes-data-matched/SHAPES-MATCHED.md ("The draft's margin, recomputed"). learned_1x is not in the paper (Ruling 6).
% SRCBLOCK: w20-shapes-data-matched/FINDINGS.md; w20-shapes-data-matched/SHAPES-MATCHED.md
Against four read-path-matched rivals fitted on the store's own 131,072 rows, each also trained for
ten times its selected optimiser steps, the store leads the strongest, an MLP, by +7.03 distinct
valid novel tuples per corner on its 109 held-out corners (+7.49 / +8.60 / +5.02 over three seeds,
every per-seed 95\% interval above +3.73), by +6.54 against that rival's best fit of any length, and
by +10.468 to +21.936 against the transformer; on every rung of an $n$-way ladder its mean margin
over the strongest rival fitted on the rung's own rows is +5.133 to +12.550.
% ENDSRC
Deletion is exact on the store and on its address book: after a value, a conjunction or a random
1\% of items is deleted, both equal the objects refit without those rows (store mode deviation at
most \sci{5.38}{-07}, book statistic deviation 0). It is not free: the book's exact statistic is % SRC: w20-shapes-data-matched/SHAPES-MATCHED.md
0.97 GB and its refit takes 2,367 to 5,913 s, and the deletable object is 4.0768 to 4.0897 times % SRC: w20-shapes-data-matched/FINDINGS.md; w13-seeds-and-tables/TABLES.md; w13-referee-2/REFEREE-v1.md
the 524,288-byte reference in its four-part ledger. Value and item deletions leave composition on % SRC: w13-referee-2/REFEREE-v1.md; w13-seeds-and-tables/TABLES.md
the surviving corners within one tuple per corner, but deleting the conjunction red $\wedge$ sphere
changes what the book knows about the shared value: survivors fall by 6.13 / 6.43 / 6.30 tuples per % SRC: w20-shapes-data-matched/FINDINGS.md
corner, against at most 0.11 for a random 1\% deletion. % SRC: w20-shapes-data-matched/FINDINGS.md
'''

ABSTRACT_MAZE = r'''
% w22 draft-rulings (Ruling 12): the CMAZE-T sentence of the abstract is left to the Head. The asymmetric-input
% sentence it replaces is in the w16/w19 artefacts (w16-maze-wm/FINDINGS.md; w19-maze-t-symmetric/FINDINGS.md).
\TODO{CMAZE-T sentence reserved for the Head: the six-seed symmetric-input condition table
(\S\ref{sec:maze-wm}, Table~\ref{tab:cmazet-sym}) replaces the earlier asymmetric-input sentence; no
number is written here.}
'''

CONTRIB = r'''
\item \textbf{Composition at a stated partition, against data-matched, read-path-matched rivals.} On a
  Shapes3D~\citep{burgess2018shapes,kim2018disentangling} value hold-out at $\Kstar=32$ wells, three scored seeds, with every % SRC: w12b-nr6-and-kstar/NR6-T7.md
  pre-declared landscape control collapsing, the store scores above every read-path-matched rival
  fitted on its own 131,072 rows, with each rival's convergence printed beside it: by +7.03 tuples % SRC: w20-shapes-data-matched/FINDINGS.md
  per corner over the strongest (+6.54 against that rival's best fit of any length) and by +10.468 % SRC: w20-shapes-data-matched/FINDINGS.md; w20-shapes-data-matched/SHAPES-MATCHED.md
  to +21.936 over the transformer (\S\ref{sec:shapes-t7}). On every rung of an $n$-way ladder it % SRC: w20-shapes-data-matched/SHAPES-MATCHED.md
  scores above the strongest rival fitted on the rung's own rows, by +5.133 to +12.550 on the seed % SRC: w20-shapes-data-matched/SHAPES-MATCHED.md
  mean (\S\ref{sec:ladder}); for $n\ge3$ the rungs are a training-data axis and not a depth axis.
\item \textbf{Exact deletion on the store and on its address book, with its cost and its limitation
  printed.} Value, conjunction and item deletion leave the store and the refit address book equal to
  those fitted without the rows. Value and item deletion move composition on the surviving corners by
  at most 0.61 tuples per corner, but deleting a conjunction measurably changes what the book knows % SRC: w20-shapes-data-matched/FINDINGS.md
  about the value it shares with the survivors (\S\ref{sec:shapes-del}). The deletion reads no
  training row, but the book's exact statistic is 0.97 GB, the size of its fit data, and its refit % SRC: w20-shapes-data-matched/FINDINGS.md
  takes 2,367 to 5,913 s, so this is neither a size nor a speed claim. Refusal after deletion is a % SRC: w20-shapes-data-matched/FINDINGS.md
  separate, structural mechanism, a hole in a coupling table (\S\ref{sec:hole}).
\item \textbf{The same properties on a maze.} Deletion transfers exactly; composition on routes is
  recovered by a plan-space verb, which we report as an \emph{interface} result because it lifts a
  matched-bytes lookup floor more than it lifts the store (\S\ref{sec:maze}). On a transition
  instrument with symmetric inputs, six seeds and a measured per-seed ceiling, the store is ahead of
  the best matched rival on three seeds, level with it at the ceiling on two and behind on one (mean
  +0.046). A colour-relational request coding lifts the store on every seed, but rivals handed the % SRC: w21-maze-t-seed4/FINDINGS.md
  same columns close most of that gap, so it is an input-representation result, not a store result
  (\S\ref{sec:maze-wm}). We offer the transition instrument itself as a minor contribution.
'''

THICKNESS = r'''
\noindent\textbf{Thickness, stated in the introduction.} The positive composition claim rests
on one environment (Shapes3D) at three scored seeds; no Shapes3D row can exceed three seeds,
because seeds 3 and 4 have verified caches but no frozen decoder and no external classifier, so no
corner of theirs can be judged. % SRC: w12b-nr6-and-kstar/FINDINGS.md
On routes, where the maze's exact judge scores whole plans at three scored seeds, the result is
a negative for the store and a positive for the interface. The transition instrument on the same
engine, also with an exact judge, prints six seeds on its words-plus-observation rows
(\S\ref{sec:maze-wm}). The rows still pending a run are marked placeholders whose text names the row.
'''

OBJECT_DELETION = r'''
\paragraph{The retained statistic and exact deletion.} Beside the content the store keeps, per
well, the running covariance, mean and count of the fit rows. Deleting rows subtracts their
contribution from the statistic, and the atoms are re-eigendecomposed from the downdated
statistic. The address book keeps its own exact statistic, per request tuple the count and first
moment of its fit rows; deleting rows downdates it, and the book is refit from it. Neither step
reads a training row, but the book's statistic is as large as its fit data
(\S\ref{sec:shapes-del}). We use one definition of exactness throughout, and state its scope
before any table:

\begin{quote}\small
\textsc{Exact is scoped to the store and its address book.} After deleting a set of rows, the store
content and the address book equal, to float roundoff, those fitted from the retained rows alone,
holding the frozen encoder and decoder fixed. Neither frozen network is refit at any point. It is
not retrain-equivalence of the generative model, it is not a claim that the renderer forgets, and it
is not a claim that the answers on the surviving requests are unchanged.
% SRC: w20-shapes-data-matched/FINDINGS.md
\end{quote}
'''

T7_INTRO = r'''
Table~\ref{tab:t7} is the scored result. Row 3 holds out 60 pair corners and row 4 holds out
49 six-factor corners; the arm is compared with each rival on all 109 corners with the corner
as the unit. % SRC: w12b-nr6-and-kstar/NR6-T7.md
The control gate is checked before any margin, and was measured, not assumed, on every seed:
all four landscape controls read exactly 0.000 on both rows on all three seeds, with the
across-wells control a derangement. % SRC: w20-shapes-data-matched/SHAPES-MATCHED.md
% SRCBLOCK: w20-shapes-data-matched/SHAPES-MATCHED.md; w20-shapes-data-matched/FINDINGS.md; w20-shapes-data-matched/SHAPES-MATCHED.json
Every rival is read-path matched (130,433 to 131,891 parameters) and fitted on the store's exact
131,072-row pool by one protocol: six short learning-rate and weight-decay trials and two epoch
counts, selected on a fixed 10\% held-out split, then refit on all rows. Every family selected the
largest learning rate and the longer epoch count of the grids, 20,480 optimiser steps, so every
family is also fitted at ten times those steps, and the ten-times fit carries the family wherever
it gained more than one tuple per corner (the MLP at seeds 0 and 1, the continuous MLP at every
seed). Against the strongest rival, the MLP at ten times its steps, the store leads by +7.486 /
+8.596 / +5.018 (mean +7.034), every 95\% interval above +3.73, winning 76 / 84 / 69 of the 109
corners and losing 22 / 16 / 13. The MLP had not stopped improving at seed 0: thirty times its steps
add +1.47, and against that rival's best fit of any length the margin is +6.018 / +8.596 / +5.018
(mean +6.544). The store leads the LSTM by +8.422 to +15.046, the continuous MLP by +6.523 to +18.963
and the transformer by +10.468 to +21.936. Because every family selected the edge of both search
grids, the rival search space may still be truncated; the ten-times column is the instrument we
have for that, and the grids were not widened after the fact.
% ENDSRC
'''

T7_TABLE = r'''
\begin{table}[t]
\centering\small
% w22 draft-rulings (Ruling 2): the comparators this table carried before (16,384 rows, 1,280 steps) are retired
% from the paper; their rows remain in w12b-nr6-and-kstar/NR6-T7.md and w20-shapes-data-matched/SHAPES-MATCHED.md.
\caption{\textbf{Composition at $\Kstar=32$ against data-matched rivals, Shapes3D value hold-out, TEST, $n=3$
seeds (0, 1, 2).} Distinct valid novel tuples per corner (ceiling 36 on row 3, 72 on row 4). The store % SRC: w13-seeds-and-tables/TABLES.md
and every rival are fitted on the same 131,072 rows; rivals are matched on read-path parameters % SRC: w20-shapes-data-matched/SHAPES-MATCHED.md
(130,433 to 131,891) and each family is printed at its selected schedule and at ten times it % SRC: w20-shapes-data-matched/SHAPES-MATCHED.md
(optimiser steps in brackets); $\ast$ marks the fit each paired $\Delta$ is taken against: the
selected schedule where ten times the steps moved the family by at most one tuple per corner,
otherwise, and for the MLP on every seed, ten times the steps. ``Paired $\Delta$'' is the store minus
that fit over all 109 corners. The table is a floor, never a rival. LACE and PoE are in % SRC: w12b-nr6-and-kstar/NR6-T7.md
Table~\ref{tab:lace-poe}.}
\label{tab:t7}
\setlength{\tabcolsep}{3pt}
\begin{adjustbox}{max width=\linewidth}\begin{tabular}{lcccccc}
\toprule
arm (optimiser steps) & seeds & all 109 corners & row 3 (60 pair) & row 4 (49 six-factor) & paired $\Delta$, store $-$ arm & corners won / tied / lost \\ % SRC: w12b-nr6-and-kstar/NR6-T7.md
 & & seed 0 / 1 / 2 & 3-seed mean & 3-seed mean & seed 0 / 1 / 2 & seed 0 $\cdot$ 1 $\cdot$ 2 \\
\midrule
% SRCBLOCK: w20-shapes-data-matched/SHAPES-MATCHED.md; w20-shapes-data-matched/SHAPES-MATCHED.json; w20-shapes-data-matched/rivals/test30x/seed0/mlp/declared.json
\textbf{store, $\Kstar=32$ (closed form)} & 3 & \textbf{47.495 / 47.404 / 49.119} & \textbf{32.350} & \textbf{67.177} & --- & --- \\
MLP (20,480) & 3 & 36.706 / 37.569 / 43.174 & 25.394 & 55.993 & & \\
MLP (204,800)$^\ast$ & 3 & 40.009 / 38.807 / 44.101 & 26.417 & 58.796 & +7.486 / +8.596 / +5.018 & 76/11/22 $\cdot$ 84/9/16 $\cdot$ 69/27/13 \\
MLP (614,400; diagnostic) & 3 & 41.477 / 36.541 / 43.495 & 25.006 & 59.483 & +6.018 (seed 0) & 67/14/28 (seed 0) \\
LSTM (20,480)$^\ast$ & 3 & 33.972 / 32.358 / 40.697 & 23.467 & 50.626 & +13.523 / +15.046 / +8.422 & \\
LSTM (204,800) & 3 & 33.688 / 32.541 / 39.853 & 22.994 & 50.503 & & \\
transformer (20,480)$^\ast$ & 3 & 25.560 / 27.789 / 38.651 & 20.050 & 43.667 & +21.936 / +19.615 / +10.468 & \\
transformer (204,800) & 3 & 25.303 / 27.239 / 37.844 & 18.783 & 44.020 & & \\
continuous MLP (20,480) & 3 & 17.917 / 16.541 / 26.844 & 11.722 & 31.102 & & \\
continuous MLP (204,800)$^\ast$ & 3 & 28.532 / 32.936 / 42.596 & 22.589 & 49.503 & +18.963 / +14.468 / +6.523 & \\
continuous MLP (614,400; diagnostic) & 3 & 27.284 / 34.009 / 38.706 & 21.894 & 47.340 & & \\
table (floor, never a rival) & 3 & 8.229 / 5.073 / 9.752 & 5.167 & 10.769 & & \\
\midrule
four landscape controls & 3 & 0.000 (all) & 0.000 (all) & 0.000 (all) & & \\
% ENDSRC
\bottomrule
\end{tabular}\end{adjustbox}
\end{table}
'''

T7_PANELS_RETIRED = r'''
% ---- w22 draft-rulings (Ruling 2): the three T7 rendered success/failure panel slots are removed. Their arms are the
% retired 16,384-row comparators (figures/rendered/rc_w19_t7_per_arm_success_failure.pdf,
% rc_w19_t7_rival_succeeds_clu_fails.pdf, rc_w19_t7_rival_succeeds_clu_fails_seed1.pdf; w19-success-failure-panels/REPORT.md).
% A re-render against the data-matched rivals of Table tab:t7 is a figures-owner item (declared in app:notrun).
'''

EARLIER_SET = r'''
\paragraph{A second, earlier rival set.} The ladder (\S\ref{sec:ladder}) also prints an earlier set
of read-path-matched rivals, fitted on 32,256 rows of a combination-split pool, at 117,878 to % SRC: _thinkpad/w4/spoke-a-rivals/rivals/seed_0/mlp.rival.json; _thinkpad/w4/spoke-a-rivals/REPORT.md
119,858 parameters, 40 epochs at batch 256 and learning rate 0.003 (5,040 steps), the same for % SRC: _thinkpad/w4/spoke-a-rivals/REPORT.md; _thinkpad/w4/spoke-a-rivals/rivals/seed_0/mlp.json
every family and seed. % SRC: _thinkpad/w4/spoke-a-rivals/rivals/seed_0/mlp.json
The combination-split pool holds out combinations but no factor value, so that set was fitted on
rows carrying the held-out \texttt{object\_hue} and shape values that the value hold-out store never
saw; its own record states that such a comparison is not like-for-like and flatters whichever arm
saw more, here the rival. It is printed beside the data-matched set, never in its place.
% SRC: _thinkpad/w4/spoke-a-rivals/rivals/seed_0/mlp.rival.json; w12-composition-ladder/PREREG.md
'''

LADDER_INTRO = r'''
The ladder asks whether the store keeps composing as more factors are pinned jointly. A rung
$n$ holds out corners in which $n$ factors are jointly pinned to held-out values; $n=2$ and
$n=6$ are the two ends. Table~\ref{tab:ladder} gives the store at $\Kstar$ on every rung and
seed beside two read-path-matched rival sets on the same corners: the data-matched set, all four
families refit on each rung's own 131,072-row pool, and the earlier 32,256-row set, whose features % SRC: w20-shapes-data-matched/FINDINGS.md
are a function of the request alone, so that its scores do not depend on the rung's fit pool and on
rungs $n\ge3$ are exactly constant by construction. % SRC: w12-composition-ladder/FINDINGS.md
The store's control gate at $\Kstar$ collapses on all 15 legs by the pre-declared 1.0 margin;
13 of 15 read exactly 0.0000 and two ($n=3$, seeds 0 and 1) read 0.0083 on
\texttt{landscape\_shuffled}, one surviving novel tuple over 120 corners, which we print
rather than round away. % SRC: w12b-nr6-and-kstar/FINDINGS.md
'''

LADDER_TABLE = r'''
\begin{table}[t]
\centering\small
\caption{\textbf{The ladder at $\Kstar=32$, $n=3$ seeds, distinct valid novel tuples per % SRC: w12b-nr6-and-kstar/KSTAR-LADDER.md
corner, per seed (0 / 1 / 2).} Store rows from the $\Kstar$ replication (a free replication,
no scored budget; every row held out). \emph{Data-matched set}: the four read-path-matched families % SRC: w20-shapes-data-matched/SHAPES-MATCHED.md
refit on each rung's own 131,072-row pool at the selected schedule (20,480 optimiser steps) and, for % SRC: w20-shapes-data-matched/SHAPES-MATCHED.json
the MLP and the continuous MLP, at ten times it (204,800); the margin row is the store minus the % SRC: w20-shapes-data-matched/SHAPES-MATCHED.json
strongest of them, paired per corner. \emph{Earlier set}: fitted on 32,256 rows of a % SRC: _thinkpad/w4/spoke-a-rivals/rivals/seed_0/mlp.rival.json
combination-split pool that carries the held-out values (\S\ref{sec:shapes-t7}); a rival of that set
does not depend on the rung. A third set, fitted on 32,768 rows of each rung's pool, is pending. LACE % SRC: w20-shapes-data-matched/FINDINGS.md
and PoE are in Table~\ref{tab:lace-poe}. $n=2$: 60 corners; $n\ge3$: the identical % SRC: w12-composition-ladder/FINDINGS.md
120 corners (see the inference note). \emph{Inference note:} for $n\ge3$ the rungs differ in % SRC: w12-composition-ladder/FINDINGS.md
fit-pool coverage, not in composition depth.}
\label{tab:ladder}
\setlength{\tabcolsep}{2.6pt}
\begin{adjustbox}{max width=\linewidth}\begin{tabular}{lcccccc}
\toprule
arm & seeds & $n=2$ & $n=3$ & $n=4$ & $n=5$ & $n=6$ \\
\midrule
% SRCBLOCK: w12b-nr6-and-kstar/KSTAR-LADDER.md
\textbf{store, $\Kstar$} & 3 & \textbf{31.800 / 32.500 / 33.633} & \textbf{60.150 / 62.783 / 61.908} & \textbf{65.242 / 65.817 / 67.200} & \textbf{65.642 / 66.817 / 67.283} & \textbf{65.908 / 66.250 / 67.650} \\
\quad 3-seed mean & 3 & 32.644 & 61.614 & 66.086 & 66.581 & 66.603 \\
% ENDSRC
store, $n=6$ on $n=3$'s fit pool & 3 & \multicolumn{4}{l}{TEST cell \emph{derived, not measured}: the $n=3$ store cell, by construction (text)} & \emph{60.150 / 62.783 / 61.908, derived} \\ % SRC: w15-coverage-control/FINDINGS.md
\midrule
\multicolumn{7}{l}{\emph{data-matched set: each rung's own 131,072 rows}} \\ % SRC: w20-shapes-data-matched/FINDINGS.md
% SRCBLOCK: w20-shapes-data-matched/SHAPES-MATCHED.md; w20-shapes-data-matched/SHAPES-MATCHED.json
MLP (20,480) & 3 & 22.450 / 25.200 / 29.267 & 43.558 / 46.742 / 51.592 & 45.408 / 48.642 / 57.300 & 51.192 / 52.825 / 58.842 & 49.525 / 54.692 / 58.867 \\
MLP (204,800) & 3 & 27.167 / 25.583 / 29.783 & 47.258 / 46.833 / 57.200 & 50.517 / 50.017 / 60.075 & 53.642 / 53.275 / 60.767 & 54.183 / 52.200 / 60.033 \\
LSTM (20,480) & 3 & 20.550 / 22.383 / 27.150 & 36.400 / 37.233 / 40.908 & 42.492 / 39.958 / 53.517 & 46.492 / 47.442 / 55.692 & 47.758 / 46.275 / 55.417 \\
transformer (20,480) & 3 & 16.417 / 17.183 / 25.750 & 26.083 / 25.192 / 30.550 & 31.792 / 33.717 / 48.800 & 37.142 / 37.458 / 52.475 & 36.592 / 37.300 / 52.942 \\
continuous MLP (20,480) & 3 & 10.383 / 11.517 / 11.383 & 16.008 / 21.225 / 29.167 & 21.000 / 24.642 / 36.183 & 21.817 / 32.583 / 37.550 & 19.817 / 29.517 / 39.875 \\
continuous MLP (204,800) & 3 & 21.333 / 21.950 / 24.867 & 24.125 / 41.300 / 39.808 & 29.883 / 45.158 / 44.883 & 40.192 / 49.858 / 54.258 & 48.633 / 49.400 / 49.525 \\
\textbf{store $-$ strongest, paired} & 3 & +4.633 / +6.917 / +3.850 & +12.892 / +15.950 / +4.708 & +14.725 / +15.800 / +7.125 & +12.000 / +13.542 / +6.517 & +11.725 / +11.558 / +7.617 \\
\quad 3-seed mean & 3 & +5.133 & +11.183 & +12.550 & +10.686 & +10.300 \\
% ENDSRC
\midrule
\multicolumn{7}{l}{\emph{earlier set: 32,256 rows of the combination-split pool}} \\ % SRC: _thinkpad/w4/spoke-a-rivals/rivals/seed_0/mlp.rival.json
% SRCBLOCK: w13-seeds-and-tables/TABLES.md
MLP & 3 & 17.0167 / 22.0000 / 23.3333 & \multicolumn{4}{c}{35.2083 / 41.5667 / 49.7500 (identical on $n=3$--$6$)} \\
LSTM & 3 & 19.4167 / 19.3333 / 23.7500 & \multicolumn{4}{c}{44.4333 / 39.4000 / 46.6750 (identical)} \\
transformer & 3 & 12.7500 / 15.7833 / 24.2500 & \multicolumn{4}{c}{28.8000 / 33.9000 / 51.9917 (identical)} \\
continuous MLP & 3 & 3.8833 / 4.5667 / 6.5333 & \multicolumn{4}{c}{6.3333 / 9.1167 / 12.6667 (identical)} \\
table (floor) & 3 & 5.8333 / 3.4833 / 6.1833 & \multicolumn{4}{c}{8.4333 / 7.9167 / 12.2333 (identical)} \\
% ENDSRC
\midrule
32,768-row set & 3 & \multicolumn{5}{l}{\PLACEHOLDER{ladder rival set fitted on 32,768 rows of each rung's pool, four families, 3 seeds: the column slot, pending; may land as a negative}} \\ % SRC: w20-shapes-data-matched/FINDINGS.md
\bottomrule
\end{tabular}\end{adjustbox}
\end{table}
'''

LADDER_PAIRED = r'''
On every rung and every seed the store's score in Table~\ref{tab:ladder} exceeds every rival's
printed score. Paired per corner against the strongest data-matched rival, the store's 95\% % SRC: w20-shapes-data-matched/FINDINGS.md
bootstrap interval~\citep{efron1979bootstrap} lies above zero on every rung and seed (lowest lower
bound +2.46, at $n=3$, seed 2), and the smallest seed-level margin is +3.85. Two caveats travel with % SRC: w20-shapes-data-matched/FINDINGS.md
the rival rows. The data-matched set is not proven converged everywhere: a ten-times spot check % SRC: w20-shapes-data-matched/FINDINGS.md
moves the LSTM from 20.55 to 21.85 at $n=2$, seed 0, printed and not carried. The earlier set's % SRC: w20-shapes-data-matched/FINDINGS.md
\texttt{rival\_transformer} and \texttt{rival\_lstm} were fitted before a later dtype repair of the
token families and were not re-fitted on the ladder's rungs. Nothing in this table is a
composition-depth result.
% SRC: w12b-nr6-and-kstar/FINDINGS.md
'''

HEADLINE_FIG_RETIRED = r'''
% w22 draft-rulings (Ruling 2): the figure slot fig_headline_steps (the step-matched diagnostic, with the retired
% 16,384-row / 1,280-step comparator) is removed from the paper. The figure stays in figures/fig_headline_steps.pdf
% and its data in w20-shapes-data-matched/SHAPES-MATCHED.json (rows34.arms).
'''

DEL_INTRO = r'''
Three deletions are run on the $\Kstar$ object on TEST, on its 131,072-row fit pool, at three % SRC: w20-shapes-data-matched/FINDINGS.md
seeds. \emph{Value}: every fit row carrying red. \emph{Conjunction}: every fit row carrying red
\emph{and} sphere; we call it the red sphere everywhere. The conjunction first named in the design,
red $\wedge$ cube, deletes zero rows, because cube is the shape held out of the split entirely; it is
printed as vacuous and never as a deletion result. \emph{Item}: a random 1\% of fit rows, which is
also the control for refit instability. \emph{Forgetting $c$} is three operations on one object:
downdate the store; downdate the address book's statistic and refit the book from it; and punch the
hole for $c$ (\S\ref{sec:hole}). Every deletion row therefore prints four measurements
(Table~\ref{tab:del}): (a) the store equals the store refit without the rows, a result; (b) the book
equals the book refit without the rows, a result; (c) requests for $c$ are refused through the hole,
a display row only, because the hole is a flag and symbolic refusal is not claimed; and (d)
composition on the surviving corners before and after, a result, with the check that the downdated
object and the object refit without the rows compose identically.
'''

DEL_TABLE = r'''
\begin{table}[t]
\centering\small
\caption{\textbf{Deletion on the single design, the four measurements, $\Kstar=32$, rows 3--4, TEST, fit % SRC: w20-shapes-data-matched/FINDINGS.md
pool 131,072 rows, one row per seed ($n=1$ each).} Forgetting $c$ is: downdate the store, downdate % SRC: w20-shapes-data-matched/FINDINGS.md
the address book's statistic and refit the book from it, punch the hole. \textbf{(a) A result}: the
store's largest absolute mode deviation from the store refit without the rows. \textbf{(b) A
result}: the book's statistic (per-tuple count and first moment) against that of the rows kept, and
the relative deviation of the rendered features of the refit book from the book refit without the
rows. \textbf{(c) A display, never a result}: refusal through the hole. \textbf{(d) A result}:
composition on the surviving corners, before $\to$ after, with the paired change, and the downdated
object minus the object refit without the rows.}
\label{tab:del}
\setlength{\tabcolsep}{2.6pt}
\begin{adjustbox}{max width=\linewidth}\begin{tabular}{lccccccc}
\toprule
scope & seed & rows del. & (a) store & (b) book & (b) rendered & (d) survivors, before $\to$ after & (d) downdate \\
 & & & max $|$mode dev$|$ & statistic dev & rel.\ dev & (paired $\Delta$) & $-$ refit-without \\
\midrule
% SRCBLOCK: w20-shapes-data-matched/SHAPES-MATCHED.md
value: red & 0 & 11,044 & 5.96e-08 & 0 & 1.85e-09 & 50.921 $\to$ 50.865 ($-0.056$) & $+0.000$ \\
 & 1 & 11,044 & 4.82e-08 & 0 & 1.42e-09 & 50.528 $\to$ 50.854 ($+0.326$) & $+0.000$ \\
 & 2 & 11,044 & 2.01e-08 & 0 & 1.22e-09 & 52.607 $\to$ 52.000 ($-0.607$) & $+0.000$ \\
conj.: red $\wedge$ cube & 0--2 & 0 & \multicolumn{5}{l}{vacuous: cube is held out of the split; printed, never a result} \\
conj.: red sphere & 0 & 5,511 & 5.38e-07 & 0 & 3.29e-08 & \textbf{47.495 $\to$ 41.367 ($-6.128$)} & $+0.000$ \\
 & 1 & 5,511 & 1.07e-08 & 0 & 7.81e-08 & \textbf{47.404 $\to$ 40.972 ($-6.431$)} & $+0.000$ \\
 & 2 & 5,511 & 6.25e-08 & 0 & 2.03e-09 & \textbf{49.119 $\to$ 42.817 ($-6.303$)} & $+0.000$ \\
item: random 1\% & 0 & 1,311 & 2.34e-07 & 0 & 5.83e-10 & 47.495 $\to$ 47.477 ($-0.018$) & $+0.000$ \\
 & 1 & 1,311 & 1.49e-08 & 0 & 9.74e-10 & 47.404 $\to$ 47.294 ($-0.110$) & $+0.000$ \\
 & 2 & 1,311 & 3.16e-08 & 0 & 6.87e-10 & 49.119 $\to$ 49.110 ($-0.009$) & $+0.000$ \\
% ENDSRC
\midrule
\multicolumn{8}{l}{\textbf{(c) requests for $c$ refused through the hole}: a \textsc{display} row, never a result; the hole is a flag} \\
% SRCBLOCK: w20-shapes-data-matched/SHAPES-MATCHED.json; w20-shapes-data-matched/SHAPES-MATCHED.md
value: red & 0--2 & \multicolumn{6}{p{10.6cm}}{value bit cleared (1 bit, 0 rows read): red requests refused 1.000, false refusal on the other requests 0.000, every seed} \\
conj.: red sphere & 0--2 & \multicolumn{6}{p{10.6cm}}{coupling hole (1 B of the 1,322 B table): refusal \emph{undefined}, no rows-3--4 corner requests the red sphere; false refusal 0.000} \\
item: random 1\% & 0--2 & \multicolumn{6}{p{10.6cm}}{N/A: an item deletion clears no value bit and punches no hole} \\
% ENDSRC
\midrule
\multicolumn{8}{l}{\textbf{cost}, seconds per seed (0 / 1 / 2)} \\
% SRCBLOCK: w20-shapes-data-matched/SHAPES-MATCHED.json; w20-shapes-data-matched/SHAPES-MATCHED.md
store downdate & 0--2 & \multicolumn{6}{p{10.6cm}}{value 12.176 / 24.758 / 7.094; red sphere 2.389 / 6.574 / 19.39; item 2.73 / 1.196 / 2.452; 0 rows read} \\
book refit from its statistic & 0--2 & \multicolumn{6}{p{10.6cm}}{value 2,590.7 / 4,390.9 / 2,601.8; red sphere 3,671.2 / 4,463.2 / 4,066.6; item 5,426.7 / 2,367.3 / 5,912.9; reads the 967,304,800 B statistic, 0 pool rows} \\
refit without the rows & 0--2 & \multicolumn{6}{p{10.6cm}}{reads 120,028 (value) / 125,561 (red sphere) / 129,761 (item) rows} \\
% ENDSRC
\bottomrule
\end{tabular}\end{adjustbox}
\end{table}

\noindent Measurements (a) and (b) are exact on every scope and seed, and (d)'s check is exact too:
the downdated object and the object refit without the rows compose identically on every corner of
all nine legs. The deletion reads no training row: the store's downdate reads 0 data rows, and the % SRC: w20-shapes-data-matched/SHAPES-MATCHED.json
book's refit reads no pool row but reads its exact statistic, 967,304,800 B (0.97 GB). The book sees % SRC: w20-shapes-data-matched/SHAPES-MATCHED.json; w20-shapes-data-matched/FINDINGS.md
data only through per-tuple aggregates and every Shapes3D tuple is one row, so its statistic is the
per-row first moment of its 117,964 fit rows, the fit data itself; and its refit takes 2,367 to % SRC: w20-shapes-data-matched/FINDINGS.md
5,913 s, about as long as refitting the book without the rows. ``Reads no training row'' is % SRC: w20-shapes-data-matched/FINDINGS.md
therefore true of this object and is not a claim about size or speed.

\paragraph{Deleting a conjunction changes what the book knows about the shared value.} This is a
measured limitation of the unit. Deleting the red sphere lowers composition on the surviving corners
by 6.13 / 6.43 / 6.30 tuples per corner, on all three seeds. At seed 0 the loss is concentrated where % SRC: w20-shapes-data-matched/FINDINGS.md
the deleted rows tied red to a shape: the 20 corners that request red with another shape fall by % SRC: w20-shapes-data-matched/FINDINGS.md
32.1 each, the 36 corners that request a sphere in another colour by 1.47, and the other 53 move % SRC: w20-shapes-data-matched/FINDINGS.md
by +0.51. The same pipeline deleting a random 1\% of items moves the survivors by $-0.02$ / % SRC: w20-shapes-data-matched/FINDINGS.md
$-0.11$ / $-0.01$, so the loss is the deleted content, not instability of the refit; and since the % SRC: w20-shapes-data-matched/FINDINGS.md
downdated object equals the object refit without the rows on every corner, the loss is what a book
fitted without the red sphere's rows no longer knows about red. An earlier protocol that carried the
book across the basis change instead of refitting it left the survivors unchanged, because it did
not delete from the book at all. A value deletion moves the survivors by $-0.06$ / $+0.33$ / % SRC: w20-shapes-data-matched/FINDINGS.md
$-0.61$, within the declared one-tuple margin. % SRC: w20-shapes-data-matched/FINDINGS.md
'''

DEL_COMPARATORS = r'''
A learned rival's only exact deletion route is to retrain from scratch without the rows. We ran
that route for every data-matched rival family on the same deleted rows of the same 131,072-row % SRC: w20-shapes-data-matched/FINDINGS.md
pool, at the rival's selected schedule, three seeds (Tables~\ref{tab:rivals-del}
and~\ref{tab:rivals-del-seeds}, Appendix~\ref{app:delbase}). Each retrain reads 120,028 to 129,761 % SRC: w20-shapes-data-matched/SHAPES-MATCHED.md
rows and takes 547.7 to 2,064.5 s. After the value and item deletions the store stays ahead of every % SRC: w20-shapes-data-matched/SHAPES-MATCHED.md
retrained rival on every seed, by +6.954 to +32.607 tuples per corner with every 95\% interval above % SRC: w20-shapes-data-matched/SHAPES-MATCHED.md
zero. After the red-sphere deletion it is ahead on the mean at every seed, but its interval crosses
zero against the MLP at seeds 1 and 2 (+2.486 and +2.596) and against the LSTM at seed 2 % SRC: w20-shapes-data-matched/SHAPES-MATCHED.md
(+2.477): the store's own loss on that deletion, about six tuples per corner, is larger than the % SRC: w20-shapes-data-matched/SHAPES-MATCHED.md
change any retrained rival shows except the continuous MLP at seed 2 (26.844 to 17.275), so on % SRC: w20-shapes-data-matched/SHAPES-MATCHED.md
collateral damage the rivals' crude retrain is cleaner than the store's exact deletion. Nor is the
store's route faster end to end: its downdate takes seconds, but the book's refit takes 2,367 to % SRC: w20-shapes-data-matched/FINDINGS.md
5,913 s against the rivals' 547.7 to 2,064.5 s. What the store's route has that the retrain does not % SRC: w20-shapes-data-matched/FINDINGS.md; w20-shapes-data-matched/SHAPES-MATCHED.md
is exactness by construction and no pool row read. Two caveats travel with these rows: the rivals
retrain at their selected schedule, so the MLP (seeds 0 and 1) and continuous-MLP convergence flags
of Table~\ref{tab:t7} travel with them; and approximate unlearning of the data-matched rivals was not
run (Appendix~\ref{app:notrun}).
% w22 draft-rulings (Ruling 2): the earlier deletion comparators (16,384-row rivals, their retrain-without and the
% NegGrad+ / fine-tune approximate unlearners) are retired from the paper; see w12-deletion-baselines/FINDINGS.md and
% w15-single-design-seeds/T7-3SEEDS.md.
'''

MAZE_WM = r'''
The two instruments below are the maze counterpart of \S\ref{sec:shapes}: they ask the store to
\emph{generate} a world model's transitions (CMAZE-T) and states (CMAZE-O) under the exact judge,
on random mazes never seen at test, so that the within-maze lookup floor cannot transfer and the
cross-maze fragment lookup is the floor. Rivals are matched on read floats; the pre-declared
landscape controls and blind columns are printed; every row prints its seed count; and the two
instruments never share a table (Tables~\ref{tab:cmazet-sym}, \ref{tab:cmazet}
and~\ref{tab:cmazeo}, Appendix~\ref{app:mazewm}).

\paragraph{CMAZE-T, a minor contribution: a small world-model task for composition.} We offer the
transition instrument to researchers working on composition, as a small task and not as a benchmark.
Given a colour-maze state and a move, an arm generates the next state; the answer is four facts
(legality, displacement, event, pocket) and the engine judges it exactly, with no tolerance
(Figure~\ref{fig:task-cmazet}). Test mazes are generated at random, so a lookup of training mazes
cannot transfer, and the cross-maze lookup floors are printed. The request is five words; the
held-out families are a never-trained value, unseen two-way word pairs and unseen $n$-way word
combinations; the judge accepts or refuses validity; learned blind arms that drop the interaction
terms or the colour identity are printed as falsifiers; and every row prints its measured per-seed
ceiling. The task is small, its seeds differ sharply in difficulty, and it has one engine; those are
reasons to report it per seed, not to rank systems on it.

% SRCBLOCK: w21-maze-t-seed4/FINDINGS.md; w21-maze-t-seed4/TABLES.md; w21-words-full-test/FINDINGS.md; w19-maze-t-symmetric/FINDINGS.md
\paragraph{CMAZE-T with symmetric inputs: six seeds, with the ceiling beside every seed.}
Table~\ref{tab:cmazet-sym} gives every input condition, and within a condition every arm receives the
identical input. The store is Config A1 (latent 128, $K=2$, 6 atoms). On the full word design given
to every arm the store is behind the best matched rival on all three TEST seeds, by 0.0250 / 0.0013 /
0.0262; on main-effect words alone and on the observation alone it is behind on every seed. On words
plus observation, TEST, six seeds, it reads 0.800 / 0.566 / 0.808 / 0.674 / 0.533 / 0.682 against the
best matched rival over both training budgets, 0.627 / 0.567 / 0.784 / 0.412 / 0.531 / 0.867: three
clear wins (seeds 0, 2 and 3), two ties at the ceiling (seeds 1 and 4, $-0.001$ and $+0.001$) and
one loss (seed 5, $-0.185$), a six-seed mean of +0.046. Seed 3's lead is over a rival field weakened
by a matched MLP that did not converge at either budget, and on seed 5 the store scores 0.000 on two
held-out tuples that are fully supported in training and that eight other arms solve. Every landscape
control stays below the store on every seed (highest 0.495, seed 5), and deletion stays exact on all
six seeds.
% ENDSRC

% SRCBLOCK: w21-maze-t-seed4/FINDINGS.md; w21-maze-t-seed4/TABLES.md; w21-rel-newseeds/FINDINGS.md
\paragraph{The measured per-seed ceiling, a ceiling for identity-coded readers.} The ceiling of a
seed is the best score that any arm fitted to its training data reaches on it: 0.830 / 0.567 / 0.808
/ 0.714 / 0.535 / 0.939 on seeds 0--5. Two seeds sit far below the others, and the reason is in the
hold-out, not in the arms. On seeds 1 and 4 the random two-way hold-out removes every training row in
which the pocket and the tint share a colour, for some colour, leaving 403 of 742 (0.543) and 305 of
723 (0.422) held-out $n$-way rows with a colour pair that has no training support. Every reader that
codes colour by identity caps there, and the store is within 0.003 of the cap on both seeds. A code of
colour equality passes straight through it: all but 79 of seed 1's unsupported rows, and all of seed
4's, are reachable through the relation ``pocket equals tint'', and only the interaction rows in which
pocket, door and tint share one colour are unreachable by any arm. The ceiling is therefore a ceiling
for identity-coded readers on that seed, not a constant of the data. We report it per seed, and we
recommend per-seed reporting of the ceiling to anyone who uses the task.
% ENDSRC

% SRCBLOCK: w21-rel-newseeds/FINDINGS.md; w21-maze-t-clu-improve/FINDINGS.md; w21-maze-t-seed4/FINDINGS.md
\paragraph{An improved store configuration is an input-representation result.} Changing only the
store's request coding, adding colour-equality indicators crossed with the door and the direction
(Config CLU-REL-A10), lifts the store on every seed, to 0.908 / 0.894 / 0.984 / 0.927 / 0.900 / 0.939
against the best matched rival's 0.803 / 0.567 / 0.685 / 0.723 / 0.624 / 0.939. The mechanism is the
relational request coding carried through the store's closed-form head, not the store. Rivals handed
the same relational columns (the \texttt{+rel} diagnostic, printed beside every seed in
Table~\ref{tab:cmazet-sym}) reach 0.915 / 0.869 / 0.819 / 0.921 / 0.900 / 0.939, so the store's lead
over them is $-0.007$ / $+0.024$ / $+0.165$ / $+0.006$ / $+0.000$ / $+0.000$, and on seeds 3 and 4 the
columns recover 0.97 and 1.00 of the store's margin over the rivals without them. On seed 4 a control
ties the store (random atoms, 0.9004, equal on every held-out tuple), so the store is not load-bearing
there; on seed 5 a matched rival ties it (a transformer at 80,000 steps). The store's lead over the
relational rivals is sizeable only on seed 2 (+0.165). We report the configuration as evidence about
request coding, which any learner can be given.
% ENDSRC

% SRCBLOCK: w21-maze-t-seed4/FINDINGS.md; w21-rel-newseeds/FINDINGS.md; w21-maze-t-clu-improve/FINDINGS.md; w16-maze-wm/FINDINGS.md; w19-maze-t-symmetric/FINDINGS.md
\paragraph{CMAZE-T: what is not claimed.} Chained rollouts are not valid: Config A1's rollout validity
is 0.000 on all six seeds at a per-step exactness of 0.900 to 0.930, and CLU-REL-A10 reaches 0.067 /
0.000 / 0.289 / 0.267 / 0.156 / 0.000, a limitation of every arm. On unseen two-way pairs a learned
arm that reads main effects only beats the store on several seeds, and on the never-trained value the
store is below the best matched rival, so nothing is claimed on either row. The earlier asymmetric
store row, in which the store read the word design and the rivals read only the observation, is
printed in Table~\ref{tab:cmazet} and is not a comparison of like with like. Exact deletion holds on
words plus observation on every seed: the downdated statistic matches the statistic fitted without
the rows to a relative \sci{1.94}{-15} at worst, the deployed atoms do not change, and the downdated
store reads exactly as the store fitted without the rows, in 0.09 to 0.88 s with no gradient step,
against 71 to 140 s and 8,000 gradient steps for a rival's retrain-without.
% ENDSRC
'''

MAZE_CLOSE = r'''
The maze section therefore stands on deletion transfer (CMAZE-2, CMAZE-2b and CMAZE-T), the verb as
an interface result (CMAZE-2b), the CMAZE-T symmetric-input table with its per-seed ceiling, where the
store is ahead on three of six seeds and its improved configuration is a request-coding result, and
the CMAZE-O store gate with its withdrawn margin. The budget, the configuration ladder, the fallback
and every negative with its link are in Appendix~\ref{app:mazewm}.
'''

LIMITS = r'''
\begin{enumerate}[leftmargin=*,itemsep=1pt]
\item \textbf{No store row on the route tasks (CMAZE-2, CMAZE-2b).} On CMAZE-2 a matched-parameter transformer beats the
  store on every composition row ($-0.0560$ / $-0.0209$ / $-0.0434$), and the landscape controls do
  not collapse (\texttt{landscape\_shuffled} 0.9509 and \texttt{random\_atoms} 0.9601 against
  1.0000 on the seen row), so no row is claimed for the store (Appendix~\ref{app:cmaze2}). On
  CMAZE-2b the controls again fail to collapse (\S\ref{sec:verb}). % SRC: w13-seeds-and-tables/TABLES.md
  On CMAZE-T with symmetric inputs the store is ahead of the best matched rival on three of six seeds,
  level at the ceiling on two and behind on one, and behind on every seed of the full word design % SRC: w21-maze-t-seed4/FINDINGS.md
  (\S\ref{sec:maze-wm}, Table~\ref{tab:cmazet-sym}).
\item \textbf{Deleting a conjunction changes what the book knows about the shared value.} Deletion
  is exact on the store and on the address book, but deleting the red sphere lowers composition on
  the surviving corners by 6.13 / 6.43 / 6.30 tuples per corner, 32.1 on the corners that request red % SRC: w20-shapes-data-matched/FINDINGS.md
  with another shape at seed 0, while a random 1\% deletion moves them by at most 0.11; retraining a % SRC: w20-shapes-data-matched/FINDINGS.md
  rival without the same rows is cleaner on collateral damage (\S\ref{sec:shapes-del},
  \S\ref{sec:del-comparators}).
\item \textbf{Exact deletion is not cheap in memory or time.} The book's exact statistic is 0.97 GB, % SRC: w20-shapes-data-matched/FINDINGS.md
  the size of its fit data, and its refit takes 2,367 to 5,913 s, longer than a rival's % SRC: w20-shapes-data-matched/FINDINGS.md
  retrain-without; ``reads no training row'' is not a size or a speed claim.
\item \textbf{Exact is expensive in bytes.} The deletable object at $\Kstar$ is 4.0768 to 4.0897
  times the reference (Table~\ref{tab:ledger-kstar}); across nine swept partitions the honest
  four-part total is U-shaped in $K$, with its minimum 3.9942 times the reference at $K=24$, so
  $\Kstar$ is not the cheapest point; and on the one configuration at $K\ge36$ whose composition
  was measured ($K=40$, 7 atoms per well, 0.991 times the pin) composition collapses to 0.0648.
  No object with a float64 sufficient statistic composes and forgets inside 524,288 bytes at rank
  64 on any swept partition (the validation sweep labels itself a smoke, not a claim).
  % SRC: w13-referee-2/REFEREE-v1.md; w13-seeds-and-tables/TABLES.md; w13-draft-revise/DRAFT-v1.md
\item \textbf{The rival search may be truncated.} Every data-matched family selected the largest
  learning rate and the longer epoch count of its grids, and the MLP at seed 0 still gains 1.47 tuples % SRC: w20-shapes-data-matched/FINDINGS.md
  per corner from ten to thirty times its steps (\S\ref{sec:shapes-t7}).
\item \textbf{The encoder loop licenses no store claim.} Training the encoder through the store
  fired the pre-declared falsifier on row 3 of the shortened leg, a falsifier that involves no rival % SRC: w13-seeds-and-tables/TABLES.md
  (Appendix~\ref{app:encloop}).
\item \textbf{No functioning generative baseline.} LACE and PoE, re-implemented at their own budgets
  in this feature space, render no exact-match request on the generation table and score at the level
  of our landscape controls (Table~\ref{tab:lace-poe}); the study compares naive units only and has no
  working compositional-generation baseline.
\item \textbf{Post-deletion composition on the maze fails as pre-registered} (0.1024 to 0.0000
  on CMAZE-2); the repaired mechanism exists at one validation measurement only.
  % SRC: w13-seeds-and-tables/TABLES.md
\item \textbf{Refusal on the maze was novelty detection, and the maze refusal displays are
  retired.} The store refused the reachable-but-unwritten pair (0.9781) as often as the walled
  one (0.9801), at a false-refusal rate of 0.8182 on grounded-possible requests against a
  pre-registered 0.20. % SRC: w13-seeds-and-tables/TABLES.md
  A ridge probe on the same observation at $t=0$ scores AUC 0.48--0.55 across all six seed and
  tier runs: deciding solvability within budget is close to solving the maze, a property of the
  task rather than a defect of the model. % SRC: w14-advisor-report.md; w14-refusal-displays-closeout/FINDINGS.md
  The Shapes3D coupling hole is a different object in a different environment and is untouched.
\item \textbf{At most three seeds on Shapes3D.} Seeds 3 and 4 cannot be judged there (no decoder, no
  external classifier), so no Shapes3D row can exceed three seeds; CMAZE-T prints six.
\item \textbf{Further limitations} (an unseen-valid row that belongs to a generic learner, an
  exhausted placement dial, and comparators not run) are listed in Appendix~\ref{app:morelimits}.
\item \textbf{Residual novelty risk.} Abstract-level screening cannot catch a match buried in a
  method section, and very recent preprints and non-indexed workshop papers are incompletely
  covered; we judge the risk low, not zero. % SRC: _thinkpad/w8/w8-novelty/NOVELTY.md
\end{enumerate}
'''

LEAVING = r'''
The naive unit buys exact deletion by subtraction and, at a stated partition, composition on
Shapes3D that scores above read-path-matched rivals fitted on its own rows, by about seven tuples per
corner over the strongest. It does not buy composition on an exact-judge route task, where the
operator is task-shaped and belongs to the interface; on the transition task its lead over matched
rivals is small and seed-dependent, and its best configuration is a request coding any learner can be
given; deleting a conjunction costs it what the book knew about the shared value; and it pays for
exactness in bytes and in the time of the book's refit. Switched on, the dynamics cost composition at
every strength that moves the state, while a law that no arm of ours fails predicts their relaxation:
so far a \emph{law about the store}, not a \emph{lever on the task}. The experiment that would make
the law a claim about a class, a comparator predicted in advance to be mispredicted in the
$h/h^\star\in[0.5,2]$ bucket, is named and not run. % SRC: w10-physics-section-evidence/EVIDENCE.md
'''

ETHICS = r'''
The study uses a public synthetic image dataset (Shapes3D) and procedurally generated mazes; it
involves no human subjects and no personal data. Exact deletion is relevant to data-protection
requests; we claim exactness on the store and its address book only and no certified unlearning, and
we state in \S\ref{sec:shapes-del} a case in which exact deletion of a conjunction changes what the
model answers about the values that survive it.
'''

T7MORE = r'''
The rivals of Table~\ref{tab:t7} are matched on the read path at 130,514 (MLP), 130,721 (LSTM), % SRC: w20-shapes-data-matched/FINDINGS.md
130,433 (transformer) and 131,891 (continuous MLP) parameters, and are fitted on the store's own % SRC: w20-shapes-data-matched/FINDINGS.md
131,072 rows, which were measured identical to the store's fit pool. Their selection used six short % SRC: w20-shapes-data-matched/FINDINGS.md
trials over three learning rates and two weight decays and then two epoch counts, on a fixed 10\%
held-out split; per-epoch training and held-out losses are recorded for every fit. % SRC: w20-shapes-data-matched/FINDINGS.md
The deletable object itself is larger than its read path: its four-part ledger is printed in
Table~\ref{tab:ledger-kstar} and is memory, not a matched quantity. Rivals are matched on read-path
parameters only; no rival matched to the four-part memory ledger is printed.
Per-seed geometry and the selection rule's val/test check are in Appendix~\ref{app:kcurve}.
'''

LACE_POE_ROWS = r'''
PoE (own budget, not matched) & 3 & 0 / 0 / 0 & 0.020 / 0 / 0 & rival $-$ store: $-47.49$ / $-47.40$ / $-49.12$ \\
LACE (own budget, 131,072 fit rows) & 3 & \multicolumn{2}{c}{all 109 corners: 0.018 / 0.046 / 0.073} & --- \\ % SRC: w20-shapes-data-matched/SHAPES-MATCHED.md
PoE (own budget, 131,072 fit rows) & 3 & \multicolumn{2}{c}{all 109 corners: 0.000 / 0.000 / 0.000} & --- \\ % SRC: w20-shapes-data-matched/SHAPES-MATCHED.md
'''

K5_APPENDIX = r'''

The $K=5$ endpoint is printed with the same data-matched rivals as the main text. On rows 3--4 the
$K=5$ store fitted on 131,072 rows reads 43.789 / 45.679 / 47.798 and leads the strongest data-matched % SRC: w20-shapes-data-matched/SHAPES-MATCHED.md
rival by +3.780 / +6.872 / +3.697 (mean +4.783). On the ladder, Table~\ref{tab:k5} prints the $K=5$ % SRC: w20-shapes-data-matched/SHAPES-MATCHED.md
store rebuilt on the first 32,768 rows of each rung's pool, the rows the earlier $K=5$ ladder store % SRC: w20-shapes-data-matched/PREREG-w20-shapes-data-matched-1.md
was fitted from. All four landscape controls, the across-wells control a derangement, read 0.000 on
every rung and seed. By the one-$K$ rule of this paper it is the sweep's endpoint and its sign is a % SRC: w20-shapes-data-matched/SHAPES-MATCHED.md
property of $K=5$, not of the store.

\begin{table}[h]
\centering\small
\caption{\textbf{Ladder at $K=5$, the sweep's endpoint, rebuilt store, 3 seeds.} Store fitted on the % SRC: w20-shapes-data-matched/SHAPES-MATCHED.md
first 32,768 rows of each rung's pool; paired margin against the strongest data-matched rival of % SRC: w20-shapes-data-matched/SHAPES-MATCHED.md
Table~\ref{tab:ladder} (each rung's own 131,072 rows) and against the strongest rival of the earlier % SRC: w20-shapes-data-matched/SHAPES-MATCHED.md
32,256-row set. $n=2$: 60 corners; $n\ge3$: the identical 120 corners. \emph{Inference note, as for % SRC: w12-composition-ladder/FINDINGS.md; w20-shapes-data-matched/FINDINGS.md
Table~\ref{tab:ladder}:} for $n\ge3$ the rungs differ in fit-pool coverage, not in composition
depth, so the store's own rise from $n=2$ to $n=6$ is not to be inferred.}
\label{tab:k5}
\setlength{\tabcolsep}{2.4pt}
\begin{adjustbox}{max width=\linewidth}\begin{tabular}{lccccc}
\toprule
rung & seeds & store, seeds 0 / 1 / 2 & $\Delta$ vs strongest data-matched rival & mean & $\Delta$ vs strongest earlier-set rival \\
\midrule
% SRCBLOCK: w20-shapes-data-matched/SHAPES-MATCHED.md
$n=2$ & 3 & 30.450 / 30.917 / 32.000 & +3.283 / +5.333 / +2.217 & +3.611 & +11.033 / +8.917 / +7.750 \\
$n=3$ & 3 & 57.392 / 60.175 / 56.858 & +10.133 / +13.342 / $-0.342$ & +7.711 & +12.958 / +18.608 / +4.867 \\
$n=4$ & 3 & 61.342 / 62.500 / 63.583 & +10.825 / +12.483 / +3.508 & +8.939 & +16.908 / +20.933 / +11.592 \\
$n=5$ & 3 & 63.392 / 63.742 / 66.050 & +9.750 / +10.467 / +5.283 & +8.500 & +18.958 / +22.175 / +14.058 \\
$n=6$ & 3 & 61.050 / 63.192 / 66.167 & +6.867 / +8.500 / +6.133 & +7.167 & +16.617 / +21.625 / +14.175 \\
% ENDSRC
\bottomrule
\end{tabular}\end{adjustbox}
\end{table}

\noindent At $n=3$, seed 2, the rebuilt $K=5$ store is behind the strongest data-matched rival % SRC: w20-shapes-data-matched/SHAPES-MATCHED.md
($-0.342$); it leads on every other rung and seed. % SRC: w20-shapes-data-matched/SHAPES-MATCHED.md

\paragraph{Why the earlier $K=5$ ladder arm is replaced.} The earlier arm (the ladder's own
$K=5$ store on the label-derived partition, reused as already fitted) carried only two of the four
landscape controls, and its strong \texttt{random\_atoms} control, which refits the address book
against random atoms, reads 1.24 at $n=4$, seed 2 and 2.03 at $n=5$, seed 2, above the 1.0 bar, so % SRC: w20-shapes-data-matched/FINDINGS.md
its $n=4$ and $n=5$ rows are not claimable. Its ledger is kept for the byte conventions of
Appendix~\ref{app:conventions}: on its own full-square convention it is 23.886 to 24.875 times the % SRC: w13-seeds-and-tables/TABLES.md
reference over 18 legs, and \texttt{pca\_tensor} 1.358 to 1.372 times. % SRC: w13-seeds-and-tables/TABLES.md
% w22 draft-rulings (Ruling 6/7): the 1x-ledger opponents (learned_1x, 536,270-536,972 params; the K=5 bracket and
% bridge learners) and the old appendix app:1x are removed from the paper; see w12b-nr6-and-kstar/KSTAR-LADDER.md,
% w13-referee-2/REFEREE-v1.md and w20-shapes-data-matched/FINDINGS.md section 7 item 6.
'''

FIGDEL_CAPTION_LINES = r'''
punches the hole, as in Table~\ref{tab:del}. (a)~The store against the store refit without the rows (largest
'''

DELBASE = r'''
\paragraph{A sphere hazard, measured.} A damaged render could be judged sphere for rendering
reasons and score spuriously on the deleted conjunction. On this instrument a degraded render
is judged cube, the held-out shape, so the judged-sphere excess is negative for both maximally
degraded arms ($-0.3291$ for \texttt{landscape\_removed}, $-0.3078$ for
\texttt{random\_atoms}) against $+0.0074$ after the red-sphere deletion.
% SRC: w13-seeds-and-tables/TABLES.md

\begin{table}[h]
\centering\small
\caption{\textbf{Deletion cost for the data-matched rivals: retrain without the rows, on the store's % SRC: w20-shapes-data-matched/SHAPES-MATCHED.md
131,072-row pool, at the selected schedule, seeds 0 / 1 / 2.} The store's own line is % SRC: w20-shapes-data-matched/SHAPES-MATCHED.md
Table~\ref{tab:del}.}
\label{tab:rivals-del}
\setlength{\tabcolsep}{3pt}
\begin{adjustbox}{max width=\linewidth}\begin{tabular}{lcccccc}
\toprule
rival, retrain without & seeds & value (red), s & red sphere, s & item (1\%), s & rows read & bytes rewritten \\
\midrule
% SRCBLOCK: w20-shapes-data-matched/SHAPES-MATCHED.md
MLP & 3 & 547.7 / 677.4 / 729.8 & 814.3 / 805.6 / 746.6 & 884.6 / 922.8 / 742.5 & 120,028 / 125,561 / 129,761 & 522,056 \\
LSTM & 3 & 757.5 / 712.4 / 1,016.0 & 772.8 / 767.5 / 1,198.2 & 773.2 / 1,093.8 / 1,034.6 & 120,028 / 125,561 / 129,761 & 522,884 \\
transformer & 3 & 1,298.0 / 1,772.2 / 1,705.3 & 1,255.8 / 1,987.7 / 2,064.5 & 1,246.0 / 1,754.9 / 1,562.8 & 120,028 / 125,561 / 129,761 & 521,732 \\
continuous MLP & 3 & 695.9 / 760.1 / 673.3 & 922.3 / 859.6 / 712.3 & 817.2 / 862.4 / 629.7 & 120,028 / 125,561 / 129,761 & 527,564 \\
% ENDSRC
\bottomrule
\end{tabular}\end{adjustbox}
\end{table}

\begin{table}[h]
\centering\scriptsize
\caption{\textbf{The store after deletion against each rival retrained without the same rows, all % SRC: w20-shapes-data-matched/SHAPES-MATCHED.md
109 corners, seeds 0 / 1 / 2.} Rival: as trained $\to$ retrained without. Store $-$ rival: the % SRC: w20-shapes-data-matched/SHAPES-MATCHED.md
paired difference after deletion, with the 95\% bootstrap interval where it crosses zero or comes
closest to it. The store after deletion reads 50.865 / 50.854 / 52.000 (value, survivor corners), % SRC: w20-shapes-data-matched/SHAPES-MATCHED.md
41.367 / 40.972 / 42.817 (red sphere) and 47.477 / 47.294 / 49.110 (item). Rival retrains run at the % SRC: w20-shapes-data-matched/SHAPES-MATCHED.md
selected schedule, so the MLP (seeds 0, 1) and continuous-MLP convergence flags of Table~\ref{tab:t7} travel with them.}
\label{tab:rivals-del-seeds}
\setlength{\tabcolsep}{2.4pt}
\begin{adjustbox}{max width=\linewidth}\begin{tabular}{llccc}
\toprule
scope & rival & seeds & rival, as trained $\to$ retrained without (0 / 1 / 2) & store $-$ rival after deletion (0 / 1 / 2) \\
\midrule
% SRCBLOCK: w20-shapes-data-matched/SHAPES-MATCHED.md
value & MLP & 3 & 39.607 $\to$ 39.124 / 40.449 $\to$ 39.640 / 46.315 $\to$ 44.528 & +11.742 / +11.213 / +7.472 \\
 & LSTM & 3 & 36.685 $\to$ 37.191 / 34.831 $\to$ 35.315 / 43.303 $\to$ 42.742 & +13.674 / +15.539 / +9.258 \\
 & transformer & 3 & 27.933 $\to$ 27.798 / 30.090 $\to$ 29.775 / 40.865 $\to$ 40.157 & +23.067 / +21.079 / +11.843 \\
 & continuous MLP & 3 & 21.393 $\to$ 19.933 / 20.213 $\to$ 26.157 / 29.652 $\to$ 19.393 & +30.933 / +24.697 / +32.607 \\
red sphere & MLP & 3 & 36.706 $\to$ 36.110 / 37.569 $\to$ 38.486 / 43.174 $\to$ 40.220 & +5.257 / \textbf{+2.486 [$-0.94$, 6.02]} / \textbf{+2.596 [$-0.50$, 5.67]} \\
 & LSTM & 3 & 33.972 $\to$ 33.064 / 32.358 $\to$ 33.734 / 40.697 $\to$ 40.339 & +8.303 / +7.239 / \textbf{+2.477 [$-0.87$, 5.82]} \\
 & transformer & 3 & 25.560 $\to$ 24.725 / 27.789 $\to$ 27.349 / 38.651 $\to$ 37.890 & +16.642 / +13.624 / +4.927 [1.29, 8.57] \\
 & continuous MLP & 3 & 17.917 $\to$ 17.697 / 16.541 $\to$ 17.303 / 26.844 $\to$ 17.275 & +23.670 / +23.670 / +25.541 \\
item & MLP & 3 & 36.706 $\to$ 35.881 / 37.569 $\to$ 37.505 / 43.174 $\to$ 42.156 & +11.596 / +9.789 / +6.954 [5.34, 8.66] \\
 & LSTM & 3 & 33.972 $\to$ 33.248 / 32.358 $\to$ 32.587 / 40.697 $\to$ 40.881 & +14.229 / +14.706 / +8.229 \\
 & transformer & 3 & 25.560 $\to$ 25.055 / 27.789 $\to$ 28.083 / 38.651 $\to$ 37.771 & +22.422 / +19.211 / +11.339 \\
 & continuous MLP & 3 & 17.917 $\to$ 19.138 / 16.541 $\to$ 18.450 / 26.844 $\to$ 28.514 & +28.339 / +28.844 / +20.596 \\
% ENDSRC
\bottomrule
\end{tabular}\end{adjustbox}
\end{table}

\noindent Value and item scopes: every interval lies above zero, the lowest lower bound being % SRC: w20-shapes-data-matched/SHAPES-MATCHED.md
+5.34. Approximate unlearning (gradient ascent, fine-tuning on the retained rows) was not run % SRC: w20-shapes-data-matched/SHAPES-MATCHED.md
against these rivals (Appendix~\ref{app:notrun}).
% w22 draft-rulings (Ruling 2): the 16,384-row comparators' retrain-without and approximate-unlearning tables
% (tab:rivals-del, tab:unlearn, tab:rivals-del-seeds as they stood) are retired; artefacts: w12-deletion-baselines/FINDINGS.md,
% w15-single-design-seeds/T7-3SEEDS.md.
'''

CMAZET_CAPTION_LINES = r'''
(\S\ref{sec:maze-wm}); the symmetric-input rows are Table~\ref{tab:cmazet-sym}, where a matched
transformer given the store's own full word design reads ahead of the store on every TEST seed, so the % SRC: w21-words-full-test/FINDINGS.md
pair block, not the store, carries this table's $n$-way margin, and that row is a store row only
'''

CMAZET_SYM = r'''
% ---- w22 draft-rulings (Rulings 8-11): the symmetric-input condition table, every CMAZE-T symmetric-input
% placeholder replaced. Own table, never merged with Table tab:cmazet (w16, asymmetric input).
\begin{table}[h]
\centering\scriptsize
\caption{\textbf{CMAZE-T with symmetric inputs: every condition, unseen $n$-way, exact judge.} Within a % SRC: w21-maze-t-seed4/TABLES.md
condition every arm receives the identical input; the instrument, banks, hold-outs, controls
(across-wells a derangement) and floors are those of Table~\ref{tab:cmazet}. Values are per seed; the
seeds column gives how many. ``Best matched rival'' is the best rival matched on read floats over both
training budgets (8,000 and 80,000 steps) on that seed. ``Ceiling'' is the best score of any arm % SRC: w21-maze-t-seed4/FINDINGS.md
fitted to the seed's training data, the unmatched ten-times-read-floats diagnostic included and the
rule-given planner columns excluded. \texttt{+rel}: rivals handed the relational request columns of
CLU-REL-A10, a diagnostic, never a bar. VAL rows are marked. This table contains no row of
Table~\ref{tab:cmazet} and no CMAZE-O, CMAZE-1, CMAZE-2 or CMAZE-2b row.}
\label{tab:cmazet-sym}
\setlength{\tabcolsep}{2pt}
\begin{adjustbox}{max width=\linewidth}\begin{tabular}{p{2.2cm}p{0.6cm}p{10.3cm}}
\toprule
condition & seeds & CMAZE-T, symmetric inputs, unseen $n$-way \\
\midrule
% SRCBLOCK: w19-maze-t-symmetric/FINDINGS.md
words, main effects (VAL) & 3 & store 0.242 / 0.316 / 0.446 (mean 0.334; the store's reading here is the learned blind-sequential arm's) against the best matched rival by mean, a transformer at 80,000 steps, 0.747; behind by 0.413; the control bar also fails (max control 0.187 against a bar of 0.184) \\
% ENDSRC
% SRCBLOCK: w21-words-full-test/FINDINGS.md; w21-maze-t-seed4/FINDINGS.md
words, full design (TEST) & 3 & store 0.8003 / 0.5660 / 0.8076 against the best matched rival by mean, a transformer at 8,000 steps, 0.8252 / 0.5674 / 0.8338: behind by 0.0250 / 0.0013 / 0.0262; per-seed best rival 0.827 / 0.567 / 0.834; ceiling 0.845 / 0.567 / 0.866 \\
% ENDSRC
% SRCBLOCK: w21-maze-t-seed4/FINDINGS.md; w21-maze-t-seed4/TABLES.md
observation (VAL) & 6 & store, reading words through a declared extractor, 0.241 / 0.373 / 0.440 / 0.279 / 0.124 / 0.204 against the best matched rival 0.464 / 0.455 / 0.455 / 0.458 / 0.417 / 0.485: behind on every seed; ceiling 0.479 / 0.467 / 0.477 / 0.458 / 0.417 / 0.485; the extractor recovers the whole word tuple on 0.065 / 0.224 / 0.191 / 0.140 / 0.121 / 0.212 of the rows \\
\midrule
\textbf{words + observation, A1 (TEST)} & 6 & store \textbf{0.800 / 0.566 / 0.808 / 0.674 / 0.533 / 0.682} against the best matched rival 0.627 / 0.567 / 0.784 / 0.412 / 0.531 / 0.867; store $-$ rival \textbf{+0.173 / $-0.001$ / +0.023 / +0.262 / +0.001 / $-0.185$}, mean \textbf{+0.046}: three clear wins, two ties at the ceiling, one loss. Seed 3's matched MLP did not converge at either budget \\
\quad per-seed ceiling & 6 & 0.830 / 0.567 / 0.808 / 0.714 / 0.535 / 0.939 (store $-$ ceiling $-0.030$ / $-0.001$ / +0.000 / $-0.040$ / $-0.003$ / $-0.258$) \\
\quad controls, blind arms, floor & 6 & highest landscape control 0.415 / 0.209 / 0.319 / 0.135 / 0.266 / 0.495; learned blind-sequential 0.239 / 0.334 / 0.452 / 0.452 / 0.320 / 0.103; cross-maze word lookup 0.000 on every seed \\
% ENDSRC
% SRCBLOCK: w21-rel-newseeds/FINDINGS.md; w21-maze-t-seed4/FINDINGS.md
\textbf{words + observation, CLU-REL-A10 (TEST)} & 6 & store 0.908 / 0.894 / 0.984 / 0.927 / 0.900 / 0.939 against the best matched rival 0.803 / 0.567 / 0.685 / 0.723 / 0.624 / 0.939; \textbf{beside it, \texttt{+rel} 0.915 / 0.869 / 0.819 / 0.921 / 0.900 / 0.939}; highest control 0.778 / 0.563 / 0.477 / 0.698 / \textbf{0.900} / 0.827 (seed 4: random atoms equals the store); seed 5: a matched transformer at 80,000 steps equals the store; the store reaches the ceiling on every seed. An input-representation result, not a store result \\
% ENDSRC
\midrule
% SRCBLOCK: w21-maze-t-seed4/FINDINGS.md; w21-maze-t-clu-improve/FINDINGS.md; w21-rel-newseeds/FINDINGS.md
rival budget & 6 & every matched rival trained at 8,000 and 80,000 steps, both printed and the better one used; a ten-times-read-floats diagnostic enters the ceiling only \\
deletion & 6 & A1, words + observation: store-scoped exactness on every seed (relative statistic change at most \sci{1.94}{-15}, atoms unchanged, reads after the downdate identical to the store fitted without the rows); adaptation 0.09 to 0.88 s with no gradient step, against 71 to 140 s and 8,000 steps for a rival's retrain-without \\
rollout validity & 6 & a limitation of every arm: A1 0.000 on every seed (per-step exactness 0.900 to 0.930); CLU-REL-A10 0.067 / 0.000 / 0.289 / 0.267 / 0.156 / 0.000 \\
% ENDSRC
\bottomrule
\end{tabular}\end{adjustbox}
\end{table}
'''

MORELIMITS_COMP = r'''
\item \textbf{Comparators not run.} Approximate unlearners against the data-matched rivals are not
  run; an extra abstain output for the non-sequence rivals is not built; the refusal head-to-head on
  the same post-deletion requests is not run; \texttt{pca\_tensor} is not run at $\Kstar$
  (Appendix~\ref{app:notrun}).
'''

PLACEHOLDER_INDEX = r'''
Four rows remain pending, each marked in the text by a red \textsc{placeholder} whose argument names
the row and states that it may land as a negative or not at all: the third ladder rival set, fitted
on 32,768 rows of each rung's pool (one; Table~\ref{tab:ladder}); and the per-arm generated states of % SRC: w20-shapes-data-matched/FINDINGS.md
the arms of Table~\ref{tab:cmazet} in Figures~\ref{fig:cmazet-families}, \ref{fig:cmazet-door-tint}
and~\ref{fig:cmazet-deletion} (three), which were never persisted and need figure work. Rows that
were pending earlier are now written in from their artefacts: the data-matched rival set
(Tables~\ref{tab:t7} and~\ref{tab:ladder}); the four deletion measurements on every scope
(Table~\ref{tab:del}); the CMAZE-T symmetric-input conditions at six seeds
(Table~\ref{tab:cmazet-sym}); LACE and PoE with the generation-table run (\S\ref{sec:adapt},
Tables~\ref{tab:lace-poe} and~\ref{tab:gentable-kstar}); the decomposition paragraph
(Appendix~\ref{app:encoder-main}, Table~\ref{tab:bard-gpu}); and the coverage control
(\S\ref{sec:ladder}). No figure is a placeholder figure and none carries an in-image banner.
Figures~\ref{fig:cmazet-families}, \ref{fig:cmazet-per-arm}, \ref{fig:cmazet-door-tint}
and~\ref{fig:cmazet-deletion} draw every arm of the words-plus-observation condition from its
persisted objects; their three placeholders name the per-arm outputs of the arms of
Table~\ref{tab:cmazet}.
'''

NOTRUNS = r'''
\begin{itemize}[leftmargin=*,itemsep=1pt]
\item Any Shapes3D row at more than three seeds: seeds 3 and 4 pass the generation-2 cache rule
  (480,000 rows, 59 block hashes each) but have no frozen decoder and no external classifier.
  % SRC: w12b-nr6-and-kstar/FINDINGS.md
\item The strong (refit) \texttt{random\_atoms} control at $\Kstar$ (about an hour of
  single-thread CPU per leg). % SRC: w13-referee-2/REFEREE-v1.md
\item \texttt{pca\_tensor} at $\Kstar$; therefore whether ``the statistic buys deletion, not
  composition'' holds at $\Kstar$ is not measured.
\item The third ladder rival set on 32,768 rows of each rung's pool (pending); LACE and PoE refit % SRC: w20-shapes-data-matched/FINDINGS.md
  on 131,072 rows on the ladder rungs (rows 3--4 only were run); the ten-times fits on the ladder % SRC: w20-shapes-data-matched/FINDINGS.md
  beyond the MLP, the continuous MLP and the spot checks; the rivals' retrain-without at ten or
  thirty times their steps.
\item Approximate unlearning of the data-matched rivals; the refusal head-to-head on the same
  post-deletion requests; the extra abstain output for the non-sequence rivals.
\item A re-render of the Shapes3D per-arm success and failure panels against the data-matched rivals.
\item The per-arm generated states of the CMAZE-T asymmetric-input arms (Table~\ref{tab:cmazet}),
  which were never persisted; the CLU-REL-A10 configuration on the observation-only and word-only
  conditions at seeds 3 to 5.
\item The repaired maze post-deletion composition on TEST at 3 seeds (one validation measurement
  exists).
\item The maze encoder loop (contingent on a positive encoder-loop result that did not occur); the
  earlier generation cell at the longer compute pin.
\item The one experiment that would upgrade the retention law from theory to a claim (a comparator
  predicted in advance to be mispredicted).
\item A figure for the plan-versus-latent result, for the retention law, and for the $K=5\to\Kstar$
  comparison (figure slots only).
\end{itemize}
'''
