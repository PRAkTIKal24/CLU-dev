## abstract-shapes — DRAFT-v2.tex lines 106..115
```
On its 109 held-out corners the store, fitted on 131,072 rows, reads 47.4954 / 47.4037 / 49.1193 % SRC: w12b-nr6-and-kstar/NR6-T7.md; w13-seeds-and-tables/TABLES.md
against 30.0275 / 28.9174 / 37.5138 for the strongest read-path-matched rival of either of our two % SRC: w13-seeds-and-tables/TABLES.md
rival sets, whose 32,256 fit rows carry held-out values the store never saw; on every rung of an % SRC: _thinkpad/w4/spoke-a-rivals/rivals/seed_0/mlp.rival.json
$n$-way ladder it scores above a read-path-matched MLP fitted on the store's own 32,768 fit rows on % SRC: w12-composition-ladder/runs/shipped_seed0/rows_learned_ref_ownpool.json; w12b-nr6-and-kstar/kstar_ladder/2_seed0_K32/LEG.json
every seed, and beats an MLP matched to its whole four-part ledger on those rows by at least
$+3.117$ tuples per corner. % SRC: w12b-nr6-and-kstar/FINDINGS.md
Deletion is store-scoped and exact, and it is not free: the object that forgets is 4.0768 to
4.0897 times the 524,288-byte reference in its four-part ledger across the three seeds, and we
show that on this object the deletion changes what the store holds but not what it answers.
% SRC: w13-referee-2/REFEREE-v1.md; w13-seeds-and-tables/TABLES.md
```

## abstract-maze-TODO — DRAFT-v2.tex lines 131..137
```
On a transition instrument over the same mazes the store reads 0.7246 on held-out $n$-way
transitions against 0.4035 for the best matched rival over three seeds, with every landscape % SRC: w16-maze-wm/FINDINGS.md
control collapsing; but the store reads a word request with a hand-built pair block that the
rivals, which read the observation, do not receive; when every arm is given the identical input
the margin does not survive, because a matched transformer given the store's own pair design
matches or beats it, so the pair block, not the store, carries it; and no matched arm, the store % SRC: w19-maze-t-symmetric/FINDINGS.md
included, yields a valid chained rollout.
```

## contributions — DRAFT-v2.tex lines 236..261
```
\item \textbf{Composition at a stated partition, against read-path-matched rivals.} On a
  Shapes3D~\citep{burgess2018shapes,kim2018disentangling} value hold-out at $\Kstar=32$ wells, three scored seeds, with every % SRC: w12b-nr6-and-kstar/NR6-T7.md
  pre-declared landscape control collapsing, the store scores above the strongest read-path-matched
  learned rival of either of the paper's two rival sets on every seed, although that rival's fit rows
  carry held-out values the store never saw (\S\ref{sec:shapes-t7}). On every rung of an $n$-way
  ladder it also scores above a read-path-matched MLP fitted on the store's own 32,768 fit rows, and % SRC: w12b-nr6-and-kstar/kstar_ladder/2_seed0_K32/LEG.json
  beats an MLP matched to its whole ledger on the same rows by at least $+3.117$ per seed % SRC: w12b-nr6-and-kstar/FINDINGS.md
  (\S\ref{sec:ladder}); for $n\ge3$ the rungs are a training-data axis and not a depth axis.
\item \textbf{Exact, store-scoped deletion, with its cost and its bound printed.} Value,
  conjunction and item deletion are light and data-free for the store and cost the rivals a
  full retrain; the four-part byte ledger is printed beside every row; and a probe shows the
  claim's limit: at $\Kstar$ the store forgets the deleted value but the object still answers
  it, because the value identity lives in the address book (\S\ref{sec:shapes-del}).
  Refusal after deletion is a separate, structural mechanism, a hole in a coupling table
  (\S\ref{sec:hole}).
\item \textbf{The same properties on a maze, where deletion transfers through the
  store and composition transfers only on transitions.} Deletion transfers exactly; composition on routes is recovered by a plan-space
  verb, which we report as an \emph{interface} result because it lifts a matched-bytes lookup
  floor more than it lifts the store (\S\ref{sec:maze}). On a transition instrument the store
  carries a composition row with every landscape control collapsing, under a word request with a
  pair block that its rivals do not receive (given the identical input, matched rivals match or
beat it, so that margin belongs to the pair block, not the store); on validation it % SRC: w19-maze-t-symmetric/FINDINGS.md
  stays ahead of rivals trained ten times longer, and its scored replication against such rivals
  keeps the margin while one landscape control misses its collapse bar on one seed; no matched
  arm's chained rollout is valid; and on an observation
  instrument the store gate holds while the margin over rivals is withdrawn (\S\ref{sec:maze-wm}).
```

## thickness — DRAFT-v2.tex lines 266..274
```
\noindent\textbf{Thickness, stated in the introduction.} The positive composition claim rests
on one environment (Shapes3D) at three scored seeds; no row in this study can exceed three
seeds, because seeds 3 and 4 have verified caches but no frozen decoder and no external
classifier, so no corner of theirs can be judged. % SRC: w12b-nr6-and-kstar/FINDINGS.md
On routes, where the maze's exact judge scores whole plans at three scored seeds, the result is
a negative for the store and a positive for the interface. A transition instrument on the same
engine, also with an exact judge and three scored seeds, adds one store row, scoped to a request
with a pair block that its rivals do not receive (\S\ref{sec:maze-wm}). Several rows are pending further runs; each is a
marked placeholder whose text names the row and states that it may land as a negative.
```

## object-deletion — DRAFT-v2.tex lines 329..342
```
\paragraph{The retained statistic and exact deletion.} Beside the content the store keeps, per
well, the running covariance, mean and count of the fit rows. Deleting rows subtracts their
contribution from the statistic; the atoms are re-eigendecomposed from the downdated
statistic; and the address book is carried across the basis change by a closed-form rotation
formed at delete time. No row of data is read. We use one definition of exactness throughout,
and state its scope before any table:

\begin{quote}\small
\textsc{Exact is store-scoped.} After deleting a set of rows, the store content equals, to
float roundoff, the content fitted from the retained rows alone, holding the frozen encoder and
decoder fixed. Neither is refit at any point. It is not retrain-equivalence of the generative
model and it is not a claim that the renderer forgets.
% SRC: w13-seeds-and-tables/TABLES.md
\end{quote}
```

## t7-intro — DRAFT-v2.tex lines 412..424
```
Table~\ref{tab:t7} is the scored result. Row 3 holds out 60 pair corners and row 4 holds out
49 six-factor corners; the arm is compared with each rival on all 109 corners with the corner
as the unit. % SRC: w12b-nr6-and-kstar/NR6-T7.md
The control gate is checked before any margin, and was measured, not assumed, on every seed:
all four landscape controls read exactly 0.0000 on both rows on all three legs.
% SRC: w12b-nr6-and-kstar/NR6-T7.md
The rivals of Table~\ref{tab:t7} are the weaker of the two rival sets in this paper; the
store's standing against the strongest available rival is given below (\emph{Two rival sets}).
Each of them is beaten beyond the declared 1.0 margin on 109 of 109 corners on every leg; the
smallest paired delta against any comparator on any $\Kstar$ leg is $+39.266$, against the
table floor on seed 0. % SRC: w12b-nr6-and-kstar/NR6-T7.md
The seed-0 row, scored first, sits inside the three-seed spread on both rows.
% SRC: w12b-nr6-and-kstar/FINDINGS.md
```

## t7-table — DRAFT-v2.tex lines 435..466
```
\begin{table}[t]
\centering\small
\caption{\textbf{Composition at $\Kstar=32$, Shapes3D value hold-out, TEST, $n=3$ seeds
(0, 1, 2).} Distinct valid novel tuples per corner (ceiling 36 on row 3, 72 on row 4), per % SRC: w13-seeds-and-tables/TABLES.md
seed. Rivals are matched on read-path parameters (about 131k) and are the weaker of the paper's two rival sets (\S\ref{sec:shapes-t7}, \emph{Two rival sets}); the table is a floor, never a % SRC: w13-seeds-and-tables/TABLES.md
rival. ``Paired $\Delta$'' is the arm minus the comparator over all 109 corners (CI95 lower % SRC: w12b-nr6-and-kstar/NR6-T7.md
bound in the source); every comparator is beaten on 109/109 corners on every seed. The % SRC: w12b-nr6-and-kstar/NR6-T7.md
\texttt{refit\_postfix} rows re-fit the two token families from scratch inside the scoring
process after a dtype repair; seed 0 predates them and is a declared not-run for those two
rows.}
\label{tab:t7}
\setlength{\tabcolsep}{3.2pt}
\begin{adjustbox}{max width=\linewidth}\begin{tabular}{lcccc}
\toprule
arm / comparator & seeds & row 3 (60 pair corners) & row 4 (49 six-factor corners) & paired $\Delta$, all 109 corners \\ % SRC: w12b-nr6-and-kstar/NR6-T7.md
\midrule
% SRCBLOCK: w12b-nr6-and-kstar/NR6-T7.md
\textbf{single design, $\Kstar=32$} & 3 & \textbf{31.6000 / 32.2333 / 33.2167} & \textbf{66.9592 / 65.9796 / 68.5918} & --- \\
\quad mean $\pm$ sd & 3 & 32.350 $\pm$ 0.815 & 67.177 $\pm$ 1.320 & \\
MLP (as trained) & 3 & 2.9333 / 3.4333 / 5.2667 & 7.9592 / 3.0000 / 11.7347 & +42.303 / +44.165 / +40.945 \\
LSTM (as trained) & 3 & 2.5000 / 2.7000 / 4.9167 & 7.5306 / 7.0000 / 14.8980 & +42.734 / +42.771 / +39.716 \\
transformer (as trained) & 3 & 0.6667 / 0.6667 / 1.7167 & 1.9184 / 0.6327 / 4.7347 & +46.266 / +46.752 / +46.046 \\
continuous MLP (as trained) & 3 & 0.0000 / 0.0000 / 0.0167 & 0.0612 / 0.0612 / 0.1020 & +47.468 / +47.376 / +49.064 \\
transformer (refit post-fix) & 2 & n/r / 0.3000 / 1.7333 & n/r / 1.1429 / 3.1633 & n/r / +46.725 / +46.743 \\
LSTM (refit post-fix) & 2 & n/r / 3.0833 / 4.3667 & n/r / 5.0204 / 8.0612 & n/r / +43.450 / +43.092 \\
table (floor) & 3 & 5.8333 / 3.4833 / 6.1833 & 11.1633 / 7.0204 / 14.1224 & +39.266 / +42.330 / +39.367 \\
\midrule
four landscape controls & 3 & 0.0000 (all) & 0.0000 (all) & +47.495 / +47.404 / +49.119 \\
% ENDSRC
\bottomrule
\end{tabular}\end{adjustbox}
\end{table}
```

## t7-rendered-panels — DRAFT-v2.tex lines 474..496
```
% ---- w19 draft-figure-slots: T7 success / failure panels, beside Table tab:t7 (w<=14 evidence) ----
% SRCBLOCK: ../../outputs/w17-paper/figures/rendered/REQUESTS-w19.json; ../../outputs/w17-paper/figures/rendered/SAME-DECODER-w19.json; ../../outputs/w17-paper/figures/rendered/README-w19.md; ../../outputs/w19-success-failure-panels/REPORT.md
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
```

## two-rival-sets — DRAFT-v2.tex lines 485..546
```
\paragraph{Two rival sets, and which is stronger.} Two sets of read-path-matched learned rivals
appear in the main text, and on the same corners they score up to an order of magnitude apart.
The rivals of Table~\ref{tab:t7} are the as-trained comparators built for the deletion study
(\S\ref{sec:del-comparators}): fitted on a 16,384-row build-scale subsample of the value % SRC: w12-deletion-baselines/FINDINGS.md
hold-out fit pool, at 130,433 to 131,891 parameters, with epochs, learning rate and weight % SRC: w12-deletion-baselines/FINDINGS.md
decay inherited from the earlier rival grids and not tuned. % SRC: w12-deletion-baselines/FINDINGS.md
The rivals of Table~\ref{tab:ladder} are that earlier set: fitted on 32,256 rows of the % SRC: _thinkpad/w4/spoke-a-rivals/rivals/seed_0/mlp.rival.json
combination-split pool, at 117,878 to 119,858 parameters, 40 epochs at batch 256 and learning % SRC: _thinkpad/w4/spoke-a-rivals/REPORT.md; _thinkpad/w4/spoke-a-rivals/rivals/seed_0/mlp.json
rate 0.003 (5,040 steps), the same for every family and seed. % SRC: _thinkpad/w4/spoke-a-rivals/rivals/seed_0/mlp.json
The combination-split pool holds out combinations but no factor value, so the earlier set was
fitted on rows carrying the held-out \texttt{object\_hue} and shape values that the value
hold-out store and the rivals of Table~\ref{tab:t7} never saw. Its own record states that such a
comparison is not like-for-like and flatters whichever arm saw more, here the rival; the overlap
of its fit rows with the held-out corners was not measured.
% SRC: _thinkpad/w4/spoke-a-rivals/rivals/seed_0/mlp.rival.json; w12-composition-ladder/PREREG.md
On the shared corners the earlier set is the stronger. On the 60 pair corners the MLP reads % SRC: w12b-nr6-and-kstar/NR6-T7.md
2.9333 / 3.4333 / 5.2667 in Table~\ref{tab:t7} and 17.0167 / 22.0000 / 23.3333 in % SRC: w12b-nr6-and-kstar/NR6-T7.md; w13-seeds-and-tables/TABLES.md
Table~\ref{tab:ladder}, while the table floor reads 5.8333 / 3.4833 / 6.1833 in both. % SRC: w12b-nr6-and-kstar/NR6-T7.md; w13-seeds-and-tables/TABLES.md
Against the strongest learned rival of either set the store still scores higher on every seed.
On all 109 corners of Table~\ref{tab:t7} the store reads 47.4954 / 47.4037 / 49.1193, and the % SRC: w12b-nr6-and-kstar/NR6-T7.md
strongest earlier-set rival 30.0275 (LSTM) / 28.9174 (LSTM) / 37.5138 (transformer), scored on % SRC: w13-seeds-and-tables/TABLES.md
the same corners in the ladder's reproduction leg (a rival's score does not depend on $K$). % SRC: w12-composition-ladder/FINDINGS.md
On the ladder's $n=2$ rung the store reads 31.800 / 32.500 / 33.633 against 19.4167 / % SRC: w12b-nr6-and-kstar/KSTAR-LADDER.md; w13-seeds-and-tables/TABLES.md
22.0000 / 24.2500; on $n=3$, its lowest rung on the identical 120 corners, 60.150 / 62.783 / % SRC: w12b-nr6-and-kstar/KSTAR-LADDER.md; w13-seeds-and-tables/TABLES.md
61.908 against 44.4333 / 41.5667 / 51.9917. % SRC: w12b-nr6-and-kstar/KSTAR-LADDER.md; w13-seeds-and-tables/TABLES.md
These are score comparisons, not paired margins, and we do not lead with the margins of
Table~\ref{tab:t7} over its own rivals.
\TODO{no artefact isolates which difference between the two rival sets (fit population, fit-pool
size, parameter count or schedule) causes the gap, and none pairs the $\Kstar$ store against the
earlier set with an interval; a refit of the earlier set on the value hold-out pool exists but is
scored in no artefact.}
% SRCBLOCK: w12-deletion-baselines/FINDINGS.md; _thinkpad/w4/spoke-a-rivals/rivals/seed_0/mlp.rival.json; w13-seeds-and-tables/TABLES.md; ../../tasks/w20/shapes-data-matched.md; w12b-nr6-and-kstar/kstar_ladder/2_seed0_K32/LEG.json
\PLACEHOLDER{w20 \texttt{shapes-data-matched}: the one data-matched rival set that replaces both
rival sets above. The fit pools as they stand, per table: in Table~\ref{tab:t7} the store is fit on
131,072 rows of the value hold-out pool and its rivals on 16,384 rows of that pool (the deletion
study's declared build/smoke scale, reused as the headline comparators); in Table~\ref{tab:ladder}
the store's content is fit on 32,768 rows of each rung's pool, its data-matched MLP on the same
32,768 rows, and its other rivals on 32,256 rows of the combination-split pool, which carries the
held-out values. Pending: the four naive units refit on 131,072 rows, which is the store's own pool
for rows 3--4 and four times the ladder store's 32,768 content rows, and the store refit on the
rivals' 16,384- and 32,256-row pools, rows 3--4 and every ladder rung, 3 seeds, each rival with its
convergence evidence; which of the two tables leads is not yet decided. May land as a NEGATIVE:
against data-matched rivals the store's margin may shrink, vanish or reverse on some or all corners}
% ENDSRC

\paragraph{A rival matched on both the read path and the data.} One learned rival is already
matched on the data as well as on the read path: an MLP at the read-path pin (130,514 parameters) % SRC: w12-composition-ladder/LADDER.json; w13-seeds-and-tables/TABLES.md
fitted, on every ladder rung and seed, on the same first 32,768 rows of the rung's pool from which % SRC: w12-composition-ladder/runs/shipped_seed0/rows_learned_ref_ownpool.json; w12b-nr6-and-kstar/kstar_ladder/2_seed0_K32/LEG.json
the store's content is fitted (row ``MLP, data-matched'' of Table~\ref{tab:ladder}). The store
scores above it on every rung and seed: at $n=2$ it reads 5.5667 / 7.5833 / 9.3000 against the % SRC: w13-seeds-and-tables/TABLES.md
store's 31.800 / 32.500 / 33.633, and at $n=6$ 10.6667 / 15.1167 / 21.0333 against 65.908 / % SRC: w13-seeds-and-tables/TABLES.md; w12b-nr6-and-kstar/KSTAR-LADDER.md
66.250 / 67.650. On the 109 corners of Table~\ref{tab:t7} it reads 9.1835 / 11.1193 / 15.1284 % SRC: w12b-nr6-and-kstar/KSTAR-LADDER.md; w13-seeds-and-tables/TABLES.md
against the store's 47.4954 / 47.4037 / 49.1193; there it is fitted on 32,768 rows of the value % SRC: w12b-nr6-and-kstar/NR6-T7.md; w12-composition-ladder/runs/shipped_seed0/rows_learned_ref_ownpool.json
hold-out pool, not on the 131,072 rows from which that store is fitted, so on those corners it is % SRC: w13-seeds-and-tables/TABLES.md
matched on the population but not on the row count. It is not the strongest rival: at $n=2$ it
reads about 13 tuples per corner below the best combination-pool rival, plausibly because its pool, % SRC: w12-composition-ladder/AMENDMENT-3-CORRECTION.md
like the store's, lacks the held-out values that the combination-split pool carries. These are
score comparisons, not paired margins. The conservative comparison on the same rows is the stress
test of Appendix~\ref{app:1x}: an MLP matched to the object's whole four-part ledger (536,270 to % SRC: w12b-nr6-and-kstar/KSTAR-LADDER.md
536,972 parameters, about four times the read path), fitted on the same 32,768 rows, is beaten % SRC: w12b-nr6-and-kstar/KSTAR-LADDER.md; w12b-nr6-and-kstar/kstar_ladder/2_seed0_K32/LEG.json
beyond the pre-declared 1.0 margin on every rung and seed, with the smallest per-seed margin $+3.117$; % SRC: w12b-nr6-and-kstar/FINDINGS.md
it is a replication, not a scored row, and an MLP is the only family fitted at that size.
```

## ladder-intro — DRAFT-v2.tex lines 521..531
```
The ladder asks whether the store keeps composing as more factors are pinned jointly. A rung
$n$ holds out corners in which $n$ factors are jointly pinned to held-out values; $n=2$ and
$n=6$ are the two ends. Table~\ref{tab:ladder} gives the store at $\Kstar$ on every rung and
seed beside the read-path-matched rivals on the same corners. The rivals' features are a
function of the request alone, not of the rung's fit pool, so their scores do not depend on
$K$, and on rungs $n\ge3$ they are exactly constant by construction.
% SRC: w12-composition-ladder/FINDINGS.md
The store's control gate at $\Kstar$ collapses on all 15 legs by the pre-declared 1.0 margin;
13 of 15 read exactly 0.0000 and two ($n=3$, seeds 0 and 1) read 0.0083 on
\texttt{landscape\_shuffled}, one surviving novel tuple over 120 corners, which we print
rather than round away. % SRC: w12b-nr6-and-kstar/FINDINGS.md
```

## ladder-table — DRAFT-v2.tex lines 533..565
```
\begin{table}[t]
\centering\small
\caption{\textbf{The ladder at $\Kstar=32$, $n=3$ seeds, distinct valid novel tuples per % SRC: w12b-nr6-and-kstar/KSTAR-LADDER.md
corner, per seed (0 / 1 / 2).} Store rows from the $\Kstar$ replication (a free replication,
no scored budget; every row held out). Rival rows are the earlier, stronger read-path-matched rival
set (\S\ref{sec:shapes-t7}, \emph{Two rival sets}) scored on
the same corners; a rival does not depend on $K$. The data-matched MLP (130,514 parameters) is % SRC: w12-composition-ladder/LADDER.json
fitted on the same 32,768 rows as the store's content (\S\ref{sec:shapes-t7}); LACE and PoE are in % SRC: w12b-nr6-and-kstar/kstar_ladder/2_seed0_K32/LEG.json
Table~\ref{tab:lace-poe}. $n=2$: 60 corners; $n\ge3$: the identical % SRC: w12-composition-ladder/FINDINGS.md
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
% SRCBLOCK: w13-seeds-and-tables/TABLES.md
MLP & 3 & 17.0167 / 22.0000 / 23.3333 & \multicolumn{4}{c}{35.2083 / 41.5667 / 49.7500 (identical on $n=3$--$6$)} \\
LSTM & 3 & 19.4167 / 19.3333 / 23.7500 & \multicolumn{4}{c}{44.4333 / 39.4000 / 46.6750 (identical)} \\
transformer & 3 & 12.7500 / 15.7833 / 24.2500 & \multicolumn{4}{c}{28.8000 / 33.9000 / 51.9917 (identical)} \\
continuous MLP & 3 & 3.8833 / 4.5667 / 6.5333 & \multicolumn{4}{c}{6.3333 / 9.1167 / 12.6667 (identical)} \\
table (floor) & 3 & 5.8333 / 3.4833 / 6.1833 & \multicolumn{4}{c}{8.4333 / 7.9167 / 12.2333 (identical)} \\
MLP, data-matched (store's own fit rows) & 3 & 5.5667 / 7.5833 / 9.3000 & 8.2167 / 10.9083 / 16.2583 & 8.5417 / 12.5333 / 18.7667 & 9.8667 / 15.3083 / 19.7083 & 10.6667 / 15.1167 / 21.0333 \\
% ENDSRC
\bottomrule
\end{tabular}\end{adjustbox}
\end{table}
```

## ladder-paired — DRAFT-v2.tex lines 594..607
```
On every rung and every seed the store's score in Table~\ref{tab:ladder} exceeds every
rival's printed score. The paired estimand with a bootstrap interval~\citep{efron1979bootstrap} against the read-path
rivals has been computed at the $K=5$ endpoint (margins $+7.0833$ to $+9.6500$ at $n=2$,
$+14.9917$ to $+22.4250$ at $n=6$, Appendix~\ref{app:k5}), but not on the $\Kstar$ legs.
% SRC: w13-seeds-and-tables/TABLES.md
\TODO{no artefact for the paired $\Kstar$ ladder margin (with CI) against the read-path-matched
rivals; the $\Kstar$ replication pairs the store only against the $1\times$ opponent of
Appendix~\ref{app:1x}.}
Two caveats travel with the rival rows: the ladder's rival columns are the earlier rival set,
whose \texttt{rival\_transformer} and \texttt{rival\_lstm} were fitted before a later dtype repair of
the token families and were not re-fitted on the ladder's rungs, yet that set is the
\emph{stronger} of the paper's two rival sets on shared corners (\S\ref{sec:shapes-t7}); and
nothing in this table is a composition-depth result.
% SRC: w12b-nr6-and-kstar/FINDINGS.md
```

## headline-steps-fig — DRAFT-v2.tex lines 642..651
```
\figslot{fig_headline_steps}{\textbf{The rivals were short of optimiser steps, not of data (rows
3--4, TEST, 109 corners, 3 seeds).} (a)~The MLP rival at every fit pool and step budget that was % SRC: w20-shapes-data-matched/SHAPES-MATCHED.json
run. At 20,480 steps it scores within one tuple per corner of itself on 16,384, 32,768 and 131,072 % SRC: w20-shapes-data-matched/SHAPES-MATCHED.json
rows, while the draft's comparator (16,384 rows, 1,280 steps) and the data-matched MLP (32,768 % SRC: w20-shapes-data-matched/LOG.md
rows, 2,560 steps) sit near the table floor; ten and thirty times the steps on 131,072 rows add % SRC: w20-shapes-data-matched/SHAPES-MATCHED.json
under two tuples per corner. (b)~All four families: 1,280 steps on 16,384 rows (dotted, because % SRC: w20-shapes-data-matched/LOG.md
it is another pool), then 20,480, 204,800 and, for the MLP and the continuous MLP, 614,400 steps % SRC: w20-shapes-data-matched/SHAPES-MATCHED.json; w20-shapes-data-matched/rivals/test30x/seed0/mlp/declared.json
on the store's 131,072 rows. The blue line is the store at $\Kstar=32$ on 131,072 rows; the band % SRC: w20-shapes-data-matched/SHAPES-MATCHED.json
spans its fits on 16,384, 32,768 and 131,072 rows (47.28, 47.83 and 48.01). Large markers are seed % SRC: w20-shapes-data-matched/FINDINGS.md
means; small open markers are the seeds.}{fig:headline-steps}
```

## del-intro — DRAFT-v2.tex lines 649..665
```
Three deletions are run on the $\Kstar$ object on TEST. \emph{Value}: every fit row carrying
red. \emph{Conjunction}: every fit row carrying red \emph{and} sphere; we call it the red
sphere everywhere. The conjunction first named in the design, red $\wedge$ cube, deletes zero rows,
because cube is the shape held out of the split entirely; it is printed as vacuous and never as
a deletion result. \emph{Item}: a random 1\% of fit rows. Table~\ref{tab:del} gives each with
its cost. \emph{Forgetting a value $c$} is three operations on one object: downdate the store,
downdate the address book's statistics, and punch the hole for $c$ (\S\ref{sec:hole}). Every
deletion row therefore prints four measurements: (a) the store equals the store refit without the
rows, a result; (b) the address book's statistics equal those refit without the rows, a result;
(c) requests for $c$ are refused through the hole, a display row only, because the hole is a flag
and symbolic refusal is not claimed; and (d) composition on the other values is unchanged, a
result. Post-deletion adaptation of the store is \emph{light and data-free}: the covariance
and count are downdated by subtraction and the atoms are re-eigendecomposed from the downdated
statistic, with no row of data read. The (a) and (d) rows printed here were measured with the
address book carried across the basis change, not downdated; the address-book downdate is new,
and its cells are named placeholders in Table~\ref{tab:del}.
The reference can delete only by refitting from scratch without the deleted rows.
```

## del-table — DRAFT-v2.tex lines 663..722
```
\begin{table}[t]
\centering\small
\caption{\textbf{Exact (store-scoped) deletion on the single design, $\Kstar=32$, TEST, fit pool 131,072 rows, seeds 0, 1 % SRC: w13-seeds-and-tables/TABLES.md
and 2, one row per seed ($n=1$ each): four measurements per scope.} Forgetting $c$ is: downdate the
store, downdate the address book's statistics, punch the hole. \textbf{(a) A result}: the store
equals the store refit without the rows (``exactness'', the store's maximum absolute mode deviation
against it). \textbf{(b) A result}: the address book's statistics equal those refit without the
rows; the address-book downdate is new, and every (b) cell is a named placeholder.
\textbf{(c) A display, never a result}: requests for $c$ refused through the hole, a flag in the
coupling table; symbolic refusal is not claimed. \textbf{(d) A result}: survivor $\Delta$, the
paired composition change on corners that do not request the deleted scope, measured with the
address book carried across the basis change, not downdated. Seeds 1 and 2 are the lower block, one row per seed ($n=1$ each), so
every scope prints three seeds; they re-read the single design's scored artefacts and spend no new scored run.}
\label{tab:del}
\setlength{\tabcolsep}{2.4pt}
\begin{adjustbox}{max width=\linewidth}\begin{tabular}{lccccccccc}
\toprule
scope & seeds & rows del. & store s & store rows read & ref. refit s & ref. rows read & (a) exactness & (d) survivor $\Delta$ & cos$_{\min}$ \\
\midrule
% SRCBLOCK: w13-seeds-and-tables/TABLES.md
value: red & 1 & 11,044 & 0.8998 & 0 & 9.259 & 120,028 & \sci{5.960}{-08} & +0.0000 (89) & 0.006538 \\
conj.: red $\wedge$ cube & 1 & 0 & \multicolumn{7}{c}{vacuous: cube is held out of the split; printed, never a result} \\
conj.: red sphere & 1 & 5,511 & 0.3916 & 0 & 9.262 & 125,561 & \sci{5.383}{-07} & +0.0000 (109) & 0.066688 \\
item: random 1\% & 1 & 1,311 & 0.1055 & 0 & 9.788 & 129,761 & \sci{2.338}{-07} & +0.0000 (109) & 0.079602 \\
% ENDSRC
\midrule
% SRCBLOCK: w15-single-design-seeds/T7-3SEEDS.md; w15-single-design-seeds/deletion/test_seed1_K32/deletion.json; w15-single-design-seeds/deletion/test_seed2_K32/deletion.json
value: red, seed 1 & 1 & 11,044 & 3.7503 & 0 & 18.595 & 120,028 & \sci{4.821}{-08} & 0.0000 (89) & 0.04069122680502917 \\
value: red, seed 2 & 1 & 11,044 & 1.3085 & 0 & 14.43 & 120,028 & \sci{2.011}{-08} & 0.0000 (89) & 0.28057461977018805 \\
conj.: red $\wedge$ cube, seeds 1, 2 & 2 & 0 & \multicolumn{7}{c}{vacuous at every seed: 0 rows deleted; printed, never a result} \\
conj.: red sphere, seed 1 & 1 & 5,511 & 0.7678 & 0 & 17.402 & 125,561 & \sci{1.074}{-08} & 0.0000 (109) & 0.20550653338433075 \\
conj.: red sphere, seed 2 & 1 & 5,511 & 0.8311 & 0 & 11.539 & 125,561 & \sci{6.253}{-08} & 0.0000 (109) & 0.618318259716018 \\
item: random 1\%, seed 1 & 1 & 1,311 & 0.4182 & 0 & 17.624 & 129,761 & \sci{1.486}{-08} & 0.0000 (109) & 0.06958072632551195 \\
item: random 1\%, seed 2 & 1 & 1,311 & 0.3837 & 0 & 18.649 & 129,761 & \sci{3.162}{-08} & 0.0000 (109) & 0.18117547035211387 \\
% ENDSRC
\midrule
\multicolumn{10}{l}{\textbf{(b) address-book statistics $=$ refit-without}: a result; the address-book downdate is new} \\
value: red & 3 & \multicolumn{8}{p{12cm}}{\PLACEHOLDER{w20 matched deletion, measurement (b) on the value red: the address book's statistics after the address-book downdate against those refit without the red rows, on the store's fit pool, seeds 0, 1 and 2, with measurement (d) re-read with the address book downdated rather than carried; may land as a negative}} \\
conj.: red sphere & 3 & \multicolumn{8}{p{12cm}}{\PLACEHOLDER{w20 matched deletion, measurement (b) on the red sphere: as for the value red, seeds 0, 1 and 2; may land as a negative}} \\
item: random 1\% & 3 & \multicolumn{8}{p{12cm}}{\PLACEHOLDER{w20 matched deletion, measurement (b) on the random 1\% of items: as for the value red, seeds 0, 1 and 2; may land as a negative}} \\
\midrule
\multicolumn{10}{l}{\textbf{(c) requests for $c$ refused through the hole}: a \textsc{display} row, never a result; the hole is a flag} \\
value: red & 3 & \multicolumn{8}{p{12cm}}{\PLACEHOLDER{w20 matched deletion, measurement (c) on the value red: a hole for the value red on the $\Kstar$ object; the holes punched there are red $\wedge$ cube and the red sphere, and none for the value red is in the artefacts}} \\
conj.: red sphere & 3 & \multicolumn{8}{l}{hole punched: 2 B rewritten for two holes (red $\wedge$ cube and the red sphere) in the 1,322 B coupling table, 0 rows read, no refit, every seed} \\ % SRC: w15-single-design-seeds/FINDINGS.md
item: random 1\% & 3 & \multicolumn{8}{l}{no hole: an item deletion makes no request impossible (0 of 164 on the comparator's 16,384-row pool, every seed)} \\ % SRC: w12-deletion-baselines/FINDINGS.md
\bottomrule
\end{tabular}\end{adjustbox}
\end{table}

\noindent Every row rewrites 2,081,024 bytes of the store and address book, reads 0 data rows
and refits nothing; the reference reads 120,028 to 131,072 rows. % SRC: w13-seeds-and-tables/TABLES.md
The ranks of the downdated and refit stores agree on every scope, and the basis did move
($\cos_{\min}$ well below 1). % SRC: w13-seeds-and-tables/TABLES.md
% SRCBLOCK: w15-single-design-seeds/T7-3SEEDS.md; w15-single-design-seeds/FINDINGS.md; w13-seeds-and-tables/TABLES.md
At seeds 1 and 2 (the lower block of Table~\ref{tab:del}) every scope reproduces the seed-0
reading: exactness between \sci{1.074}{-08} and \sci{6.253}{-08}, survivor change 0.0000, the
ranks of the downdated and refit stores agree, and 0 data rows are read. The 2,081,024 bytes above
are seed 0's; seeds 1 and 2 rewrite 2,074,368 and 2,078,464 bytes. The rivals' retrain-without
cost at seeds 1 and 2 is Table~\ref{tab:rivals-del-seeds} (Appendix~\ref{app:delbase}).
% ENDSRC
```

## del-comparators — DRAFT-v2.tex lines 752..771
```
A learned rival's only exact deletion route is to retrain from scratch without the rows. We
built that route for every rival family, at matched compute, and two approximate unlearners
beside it, measured on the fit pool (16,384 rows; no held-out corner and no scored run).
Retraining without the red rows costs each rival 8.86 to 34.62 s and reads 15,032 data rows
(Table~\ref{tab:rivals-del}); the approximate unlearners are appendix rows
(Appendix~\ref{app:delbase}) because they are not scored rows of this study, and because
they miss the exact answer. % SRC: w12-deletion-baselines/FINDINGS.md

\noindent Against that exact reference, NegGrad+ over-forgets by 9.4 to 20.8 times and
fine-tuning on the retained rows under-forgets on three of four arms, though both were given
every gradient row-visit the exact retrain got (296,960, asserted equal across all eight arm
and method combinations on all three seeds). % SRC: w12-deletion-baselines/FINDINGS.md
The conjunction row confirms, on the rivals themselves, that composition survives a
conjunction deletion: retraining a learned rival without the red sphere rows leaves red-sphere
reconstruction essentially unchanged (3-seed means $-0.00289$, $+0.01907$, $-0.00179$,
$-0.00026$), because red and sphere each survive separately and the rival composes them.
% SRC: w12-deletion-baselines/FINDINGS.md
That is composition working, not deletion failing, and it is a property of compositional
generation rather than of our store. It is also why the conjunction row cannot carry a
refusal claim; the hole does.
```

## maze-wm — DRAFT-v2.tex lines 1067..1135
```
The two instruments below are the maze counterpart of \S\ref{sec:shapes}: they ask the store to
\emph{generate} a world model's transitions (CMAZE-T) and states (CMAZE-O) under the exact judge,
on random mazes never seen at test, so that the within-maze lookup floor cannot transfer and the
cross-maze fragment lookup is the floor. Rivals are matched on read floats; the pre-declared
landscape controls and blind columns are printed; criteria were pre-registered before any number; every
row has three seeds; and the two instruments never share a table (Tables~\ref{tab:cmazet}
and~\ref{tab:cmazeo}, Appendix~\ref{app:mazewm}).

% SRCBLOCK: w16-maze-wm/FINDINGS.md; w16-maze-wm/TABLES.md
\paragraph{CMAZE-T: a store row on unseen $n$-way transitions, under a pair-design request.} One pinned design (Config A1:
latent 128, $K=2$, 6 atoms, rank/width 0.094, 4,584 read floats), three TEST seeds, one pre-registered scored run each: the store generates held-out $n$-way transitions at 0.7246 (0.800 / 0.566 / 0.808) against
0.4035 (0.394 / 0.395 / 0.421) for the best matched rival, a transformer, and is ahead on every
seed. All four pre-declared landscape controls collapse below the bar of 0.575 on every seed (landscape removed
0.0000, shuffled 0.2248, shuffled across wells 0.2435, random atoms 0.1655), the learned
composition-blind arms fail (0.3418 and 0.0716 against a bar of 0.675), and the cross-maze lookup
floor reads 0.0000. The row survives rivals trained ten times longer: on validation, every rival
trained for 80,000 steps, matched or at ten times the read floats, stays below the store on every
seed (closest: seed 1, LSTM 0.437 against 0.569), and a scored replication at a wider latent (Config
C-512, three further scored runs) against 80,000-step rivals reads 0.7251 against 0.4533. A caveat travels with the
replication: on its seed 1, random atoms reach 0.464 against the store's 0.567, a margin of 0.103,
below the 0.15 bar. The row also has a ceiling set by the request design, not by the store:
validation $n$-way is 0.7247 with the same per-seed triple across about 15 configurations, and
seed 1 caps at 0.569 in all of them.
% ENDSRC

% SRCBLOCK: w16-maze-wm/PREREG-w16-maze-wm-T1.md; w16-maze-wm/FINDINGS.md; w16-maze-wm/TABLES.md; w15-cmaze-t-harness/FINDINGS.md; w14-request-layer-fix/FINAL-REPORT.md; ../../outputs/w17-paper/figures/rendered/REQUESTS-w19.json
\paragraph{CMAZE-T: the inputs are not symmetric.} The store row is not an input-matched
comparison. The store reads the request as five transition words through a \texttt{[bias | main |
pair]} design whose pair block is hand-built interaction structure, and the learned composition-blind
arms read reduced forms of the same request (main effects only, or colour identity removed); the
matched rivals regress the next state from a 636-dimensional observation read through the frozen
state encoder and receive no pair block. The design fixes which held-out $n$-way tuples are
readable, and the store decides only whether they are read cleanly, which is why validation $n$-way
sits at 0.7247 across about 15 configurations. The row therefore shows that the store is
load-bearing under this request, not that the store rather than its request design carries the
margin over the rivals; when a task-shaped interface was given to every arm (Config B, below), it
lifted the best matched rival more than the store. The symmetric-input test, in which every arm
receives the identical input, has closed with no condition winning (Table~\ref{tab:cmazet-sym},
Appendix~\ref{app:mazewm}): given the store's own pair design, a matched transformer matches or
beats the store, so the pair block, not the store, carries this row's margin over the rivals. % SRC: w19-maze-t-symmetric/FINDINGS.md
What stands on CMAZE-T is the store gate (every landscape control collapses) and exact deletion,
not a margin over matched rivals.
Two further facts bound the row. First, the rule-knowing planner columns, relaxed oracles that are
handed the engine's rules and never trained, read 0.9071 for BLIND-SEQUENTIAL, above the store's
0.7246, 0.5509 for BLIND-COLOUR and 0.6047 for BLIND-FACTOR. On CMAZE-2 the same kind of planner at
0.9077 showed that the held-out routes rarely exercised the relaxed rule (\S\ref{sec:req-layer}), and
it reads the same way here: the rule order the blind-sequential planner ignores matters only on rows
with a tinted door, a minority of the held-out $n$-way rows, so the row tests held-out word
combinations mostly on rows that do not need the door-then-tint interaction. The learned
composition-blind arms, which must generalise, fail (0.3418 and 0.0716). Second, the held-out
$n$-way rows per seed are 345 / 328 / 309 \texttt{free}, 316 / 307 / 288 \texttt{rule\_b}, 70 / 79 /
55 \texttt{interaction}, 30 / 28 / 33 \texttt{open\_door\_tint} and 0 / 0 / 1 \texttt{rule\_a}, of 761
/ 742 / 686; no per-family score exists.
% ENDSRC

% SRCBLOCK: w16-maze-wm/FINDINGS.md; w16-maze-wm/TABLES.md
\paragraph{CMAZE-T: what is not claimed.} Rollout validity is 0.000 for the store and for every
matched rival on every seed: per-step exactness is 0.925 for the store and 0.818 for the best
matched rival, and a per-step error compounds over 43 chained steps (failing link: \emph{read}). One
qualifier travels with the row: a diagnostic transformer at ten times the read floats, which is not
a matched rival, reads a non-zero validity on seed 2 (0.022). On unseen 2-way the store's 0.410 is
beaten by the learned composition-blind sequential arm (0.557), so nothing is claimed there (link:
\emph{interface}); on the never-seen colour the store's 0.465 is below the best rival's 0.511 (link:
\emph{content}). A task-shaped match-bit interface given to every arm lifts the best matched rival by
$+0.40$ and the store by $+0.20$, and fails validation (link: \emph{interface}). Exact deletion holds
on the winner: store-scoped exactness \sci{7.7}{-16} to \sci{1.2}{-15} on the statistic, deployed
atoms unchanged, the downdated read equal to the fitted-without read, in 0.03--0.13 s with no data
read, against 13.8--14.9 s and 8,000 gradient steps for a rival's retrain-without.
% ENDSRC
```

## maze-close — DRAFT-v2.tex lines 1167..1172
```
The maze section therefore stands on deletion transfer (CMAZE-2 and CMAZE-2b), the verb as an
interface result (CMAZE-2b), the CMAZE-T store gate and exact deletion, whose margin over matched
rivals belongs to the pair-design request and does not survive symmetric inputs, with its % SRC: w19-maze-t-symmetric/FINDINGS.md
declared negatives, and the CMAZE-O store
gate with its withdrawn margin. The budget, the configuration ladder, the fallback and every negative
with its link are in Appendix~\ref{app:mazewm}.
```

## limits-list — DRAFT-v2.tex lines 1181..1227
```
\begin{enumerate}[leftmargin=*,itemsep=1pt]
\item \textbf{No store row on the route tasks (CMAZE-2, CMAZE-2b).} On CMAZE-2 a matched-parameter transformer beats the
  store on every composition row ($-0.0560$ / $-0.0209$ / $-0.0434$), and the landscape controls do
  not collapse (\texttt{landscape\_shuffled} 0.9509 and \texttt{random\_atoms} 0.9601 against
  1.0000 on the seen row), so no row is claimed for the store (Appendix~\ref{app:cmaze2}). On
  CMAZE-2b the controls again fail to collapse (\S\ref{sec:verb}). % SRC: w13-seeds-and-tables/TABLES.md
  On CMAZE-T the store row is scoped to a word request with a pair block that the rivals do not
  receive; given the identical input no condition wins, and the pair block, not the store, carries
  the margin (\S\ref{sec:maze-wm}, Table~\ref{tab:cmazet-sym}). % SRC: w19-maze-t-symmetric/FINDINGS.md
\item \textbf{The value identity is not in the store at $\Kstar$.} Exact deletion of every red
  row leaves the red-requesting corners at 32.2500 and judge accuracy at 0.8958
  (\S\ref{sec:shapes-del}); the store-scoped claim is the only one demonstrated there.
  % SRC: w13-seeds-and-tables/TABLES.md
\item \textbf{Exact is expensive in bytes.} The deletable object at $\Kstar$ is 4.0768 to 4.0897
  times the reference (Table~\ref{tab:ledger-kstar}); across nine swept partitions the honest
  four-part total is U-shaped in $K$, with its minimum 3.9942 times the reference at $K=24$, so
  $\Kstar$ is not the cheapest point; and on the one configuration at $K\ge36$ whose composition
  was measured ($K=40$, 7 atoms per well, 0.991 times the pin) composition collapses to 0.0648.
  No object with a float64 sufficient statistic composes and forgets inside 524,288 bytes at rank
  64 on any swept partition (the validation sweep labels itself a smoke, not a claim).
  % SRC: w13-referee-2/REFEREE-v1.md; w13-seeds-and-tables/TABLES.md; w13-draft-revise/DRAFT-v1.md
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
\item \textbf{At most three seeds.} Seeds 3 and 4 cannot be judged (no decoder, no external
  classifier), so no row can exceed three seeds.
\item \textbf{Further limitations} (an unseen-valid row that belongs to a generic learner, an
  exhausted placement dial, and comparators not run) are listed in Appendix~\ref{app:morelimits}.
\item \textbf{Residual novelty risk.} Abstract-level screening cannot catch a match buried in a
  method section, and very recent preprints and non-indexed workshop papers are incompletely
  covered; we judge the risk low, not zero. % SRC: _thinkpad/w8/w8-novelty/NOVELTY.md
\end{enumerate}
```

## leaving — DRAFT-v2.tex lines 1286..1294
```
The naive unit buys exact deletion by subtraction and, at a stated partition, composition on
Shapes3D that scores above the strongest read-path-matched learner of either of our rival sets
and above a read-path-matched MLP fitted on the store's own rows. It does not buy composition on an exact-judge route task, where the operator is
task-shaped and belongs to the interface; it does not stop the object answering a deleted value;
and it pays for exactness in bytes. Switched on, the dynamics cost composition at every strength
that moves the state, while a law that no arm of ours fails predicts their relaxation: so far a
\emph{law about the store}, not a \emph{lever on the task}. The experiment that would make the law a
claim about a class, a comparator predicted in advance to be mispredicted in the
$h/h^\star\in[0.5,2]$ bucket, is named and not run. % SRC: w10-physics-section-evidence/EVIDENCE.md
```

## ethics — DRAFT-v2.tex lines 1315..1319
```
The study uses a public synthetic image dataset (Shapes3D) and procedurally generated mazes; it
involves no human subjects and no personal data. Exact deletion is relevant to data-protection
requests; we claim store-scoped exactness only and no certified unlearning, and we state in
\S\ref{sec:shapes-del} a case in which exact deletion of a value does not change what the model
answers.
```

## t7more — DRAFT-v2.tex lines 1372..1383
```
The rivals are matched on the read path,
130,514 (MLP), 130,721 (LSTM), 130,433 (transformer) and 131,891 (continuous MLP) parameters
on seed 0; the two refit token families were refit at the same pins, width 45 / 130,433 and
width 51 / 130,721, on the same 16,384 fit rows. % SRC: w13-seeds-and-tables/TABLES.md; w12b-nr6-and-kstar/FINDINGS.md
The deletable object itself is larger than its read path: its four-part ledger is printed in
Table~\ref{tab:ledger-kstar} and is memory, not a matched quantity. The continuous-input MLP
scores at the floor (0.000 / 0.061 on seed 0), which is a fact about that rival's fit on this
population and is not leaned on. % SRC: w13-seeds-and-tables/TABLES.md
Per-seed geometry and the selection rule's val/test check are in Appendix~\ref{app:kcurve}.
A rival matched to the object's own four-part total would carry about 4.09 times the
read-path parameters and would be stronger; the only such evidence at $\Kstar$ is the
replication of Appendix~\ref{app:1x}, which is not a scored row. % SRC: w13-seeds-and-tables/TABLES.md
```

## lace-poe-131k — DRAFT-v2.tex lines 1432..1432
```
PoE (own budget, not matched) & 3 & 0 / 0 / 0 & 0.020 / 0 / 0 & rival $-$ store: $-47.49$ / $-47.40$ / $-49.12$ \\
```

## k5-and-1x — DRAFT-v2.tex lines 1475..1586
```

The ladder was first run at $K=5$ (MI partition, full square), before $\Kstar$ existed. By the
one-$K$ rule of this paper it is the sweep's endpoint and its sign is a property of $K=5$, not of the store.
Table~\ref{tab:k5} gives the store against the read-path rivals; the rival rows are those of
Table~\ref{tab:ladder}.

\begin{table}[h]
\centering\small
\caption{\textbf{Ladder at $K=5$ (MI partition), 3 seeds, the sweep's endpoint.} Store
(bit-identical to \texttt{pca\_tensor} on every rung and seed), paired margin against the best
read-path rival with the rung's own corners, and the two pre-declared landscape controls as paired deltas.
$n=2$: 60 corners; $n\ge3$: the identical 120 corners. \emph{Inference note, as for % SRC: w12-composition-ladder/FINDINGS.md
Table~\ref{tab:ladder}:} for $n\ge3$ the rungs differ in fit-pool coverage, not in composition
depth, so the store's own rise from $n=2$ to $n=6$ is not to be inferred.}
\label{tab:k5}
\setlength{\tabcolsep}{2.4pt}
\begin{adjustbox}{max width=\linewidth}\begin{tabular}{lcccccc}
\toprule
rung & seeds & store, seeds 0 / 1 / 2 & $\Delta$ vs best read-path rival & ceiling-norm. & \texttt{landscape\_shuffled} $\Delta$ & \texttt{random\_atoms} $\Delta$ \\
\midrule
% SRCBLOCK: w13-seeds-and-tables/TABLES.md
$n=2$ & 3 & 28.6333 / 31.6500 / 31.3333 & +9.2167 / +9.6500 / +7.0833 & 0.8483 & +28.6333 / +31.6500 / +31.3333 & +28.3833 / +31.5667 / +30.5833 \\
$n=3$ & 3 & 56.9500 / 59.4250 / 62.8750 & +12.5167 / +17.8583 / +10.8833 & 0.8299 & +56.9500 / +59.4250 / +62.8750 & +56.3833 / +59.2500 / +62.2250 \\
$n=4$ & 3 & 60.9333 / 62.2333 / 64.9583 & +16.5000 / +20.6667 / +12.9667 & 0.8709 & +60.9333 / +62.2333 / +64.9500 & +60.4500 / +62.0500 / +63.7167 \\
$n=5$ & 3 & 63.2417 / 64.5083 / 66.4083 & +18.8083 / +22.9417 / +14.4167 & 0.8989 & +63.2417 / +64.5083 / +66.4083 & +62.5583 / +63.9250 / +64.3750 \\
$n=6$ & 3 & 62.6917 / 63.9917 / 66.9833 & +18.2583 / +22.4250 / +14.9917 & 0.8966 & +62.6917 / +63.9917 / +66.9833 & +62.4333 / +63.8333 / +66.4917 \\
% ENDSRC
\bottomrule
\end{tabular}\end{adjustbox}
\end{table}

\noindent Nothing deteriorates as $n$ grows; ceiling-normalised the store runs 0.8483 to 0.8966,
and for $n\ge3$ that rise may be a held-out-fraction trend rather than a depth trend. No arm has a
breaking point. The strong \texttt{random\_atoms} control (address book refit against random
atoms) is used here and reads 0.25 novel tuples at $n=2$ seed 0, not zero; the paired delta is
printed rather than ``the arm's entire score''. % SRC: w13-seeds-and-tables/TABLES.md; w13-referee-2/REFEREE-v1.md
The rung $n=2$ and the 49-corner prefix of $n=6$ reproduce the shipped rows 3 and 4 within each
arm's across-corner spread on every seed (max abs difference 1.0833 and 0.5306 for the store, 0.0000
for all five rivals). % SRC: w12-composition-ladder/FINDINGS.md
The store and \texttt{pca\_tensor} span the same per-well subspace to \sci{1}{-8} on the three wide
wells (1,929 of 2,048 channels); on the two narrow wells their ranks differ sharply (15 vs 60, 41 vs
59) and not one of the 2,160 judged requests changes. % SRC: w12-composition-ladder/HEAD-RULING-K.md

\paragraph{The $K=5$ ledger on the paper's convention.} On its own full-square convention the
$K=5$ (MI partition) object is 23.886 to 24.875 times the reference over 18 legs, and
\texttt{pca\_tensor} 1.358 to 1.372 times. % SRC: w13-seeds-and-tables/TABLES.md
Re-priced on the symmetric-triangle float64 convention of the rest of the paper (arithmetic on the
recorded well widths, seed 0) its four-part total is 6,841,296 / 6,806,400 / 6,876,992 / 6,832,008
/ 6,835,248 B on $n=2..6$, 13.049 / 12.982 / 13.117 / 13.031 / 13.037 times the reference, and its
$1\times$-matched opponent would carry 1,710,324 / 1,701,600 / 1,719,248 / 1,708,002 / 1,708,812
parameters. % SRC: w13-referee-2/REFEREE-v1.md
The honest bracket for that opponent is 938,413 (float32 triangle) to 1,710,324 (float64
triangle) parameters. The opponent the ladder actually fitted carried 3,131,564 to 3,260,030
parameters, 1.9 to 3.5 times outside the bracket; we do not print its rows.
% SRC: w13-referee-2/REFEREE-v1.md; w12b-nr6-and-kstar/FINDINGS.md
An opponent inside the bracket was fitted: the bridge leg's 1,029,776-parameter learner, on the
identical corner set, seed, cache, fit rows and schedule. Against it the $K=5$ store's margin is
$-1.2333$ / $+10.6583$ / $-0.3917$ / $-0.1250$ / $-0.5833$ on $n=2..6$ (seed 0; one seed;
replication; not scored), so the store fails the every-seed bar at four rungs by at most 1.3 tuples
per corner, not by 2 to 6. % SRC: w13-referee-2/REFEREE-v1.md

% ---------------------------------------------------------------------------------------
\section{A $1\times$ stress comparison at $\Kstar$ (not the paper's matching rule)}
\label{app:1x}

The paper matches rivals on the read path. As a stress test only, the $\Kstar$ replication pairs the
store with an MLP matched to the object's own four-part total (symmetric-triangle convention),
536,270 to 536,972 parameters, fitted from the same 32,768 fit rows as the store's content.
% SRC: w12b-nr6-and-kstar/KSTAR-LADDER.md; w13-referee-2/REFEREE-v1.md
Under the ladder's own pre-declared bar, beat the opponent by 1.0 tuple per corner on every seed,
the store clears at all five rungs; all 15 per-seed margins are positive and the smallest is
$+3.117$. % SRC: w12b-nr6-and-kstar/FINDINGS.md

\begin{table}[h]
\centering\small
\caption{\textbf{$\Kstar$ replication against a $1\times$-total-bytes MLP (stress comparison),
3 seeds, held out, 0 scored runs.} Per-seed margins (seeds 0 / 1 / 2) and 3-seed mean.}
\label{tab:1x}
\begin{adjustbox}{max width=\linewidth}\begin{tabular}{lcccc}
\toprule
rung & seeds & opponent params & margin per seed & 3-seed mean \\
\midrule
% SRCBLOCK: w12b-nr6-and-kstar/KSTAR-LADDER.md; w13-draft-revise/DRAFT-v1.md
$n=2$ & 3 & 536,972 & +6.667 / +3.117 / +3.283 & +4.356 \\
$n=3$ & 3 & 536,270 & +17.875 / +12.808 / +11.533 & +14.072 \\
$n=4$ & 3 & 536,270 & +12.700 / +8.742 / +8.625 & +10.022 \\
$n=5$ & 3 & 536,972 & +10.350 / +6.517 / +8.200 & +8.356 \\
$n=6$ & 3 & 536,972 & +9.375 / +6.108 / +8.450 & +7.978 \\
% ENDSRC
\bottomrule
\end{tabular}\end{adjustbox}
\end{table}

\paragraph{What moved between $K=5$ and $\Kstar$, and how much of it is the opponent.} Going from
the ladder's $K=5$ store to $\Kstar$ changes the well count and the construction (partition rule,
store class and covariance convention together). A bridge arm holds the $\Kstar$ construction and
sets $K=5$ (seed 0 only): the margin moves $-4.567\to-0.250\to+6.667$ at $n=2$ and
$-5.775\to-2.550\to+9.375$ at $n=6$, and the bridge still loses at four of five rungs.
% SRC: w12b-nr6-and-kstar/KSTAR-LADDER.md
Both steps are opponent-budget cuts, because the opponent is defined as the object's own total: the
opponent falls 3,241,076 $\to$ 1,029,776 $\to$ 534,866 parameters at $n=2$. Across the construction
step the store's own score falls at four of five rungs ($+0.98$, $-2.23$, $-0.63$, $-0.39$, $-1.97$);
end to end 71.8\% to 84.2\% of the swing is the opponent falling and 15.8\% to 28.2\% is the store
reading better. % SRC: w13-referee-2/REFEREE-v1.md
The store's own score does rise at every rung, by $+1.86$ to $+3.38$ tuples per corner (3-seed
means). % SRC: w12b-nr6-and-kstar/FINDINGS.md
The control gate at $\Kstar$ is the weaker post-fit form of \texttt{random\_atoms} (atoms
randomised, address book not refit); the refit form is a declared not-run at $\Kstar$ and reads
0.25, not 0.0000, at $K=5$. % SRC: w13-referee-2/REFEREE-v1.md
The flip is present at both ends of the only genuine depth contrast, $n=2$ ($-2.783\to+4.356$) and
$n=6$ ($-4.061\to+7.978$), seed-matched. % SRC: w12b-nr6-and-kstar/FINDINGS.md

```

## fig-deletion-caption — DRAFT-v2.tex lines 1765..1766
```
punches the hole; unlike the (d) rows of Table~\ref{tab:del}, which carry the book across the basis
change, the book here is refit. (a)~The store against the store refit without the rows (largest
```

## delbase — DRAFT-v2.tex lines 1780..1881
```
\paragraph{A sphere hazard, measured.} A damaged render could be judged sphere for rendering
reasons and score spuriously on the deleted conjunction. On this instrument a degraded render
is judged cube, the held-out shape, so the judged-sphere excess is negative for both maximally
degraded arms ($-0.3291$ for \texttt{landscape\_removed}, $-0.3078$ for
\texttt{random\_atoms}) against $+0.0074$ after the red-sphere deletion.
% SRC: w13-seeds-and-tables/TABLES.md


\begin{table}[h]
\centering\small
\caption{\textbf{Deletion cost for the rivals, value scope (red; 1,352 of 16,384 fit rows), % SRC: w12-deletion-baselines/FINDINGS.md
seed 0.} Data rows read and bytes rewritten are seed-invariant. ``Forget $-$ retain'' is the
feature-MSE gap of the exact retrain, 3-seed mean; the seeds column gives the count for the
seconds and for the gap. The store's own line (Table~\ref{tab:del}):
0 rows read, no refit.}
\label{tab:rivals-del}
\begin{adjustbox}{max width=\linewidth}\begin{tabular}{lccccc}
\toprule
rival, retrain without & seeds (s / gap) & seconds & data rows read & bytes rewritten & forget $-$ retain (exact) \\
\midrule
% SRCBLOCK: w12-deletion-baselines/FINDINGS.md
MLP & 1 / 3 & 8.86 & 15,032 & 522,056 & +0.03661 \\
continuous MLP & 1 / 3 & 8.87 & 15,032 & 527,564 & +0.03817 \\
transformer & 1 / 3 & 34.62 & 15,032 & 521,732 & +0.06057 \\
LSTM & 1 / 3 & 15.99 & 15,032 & 522,884 & +0.04111 \\
% ENDSRC
\bottomrule
\end{tabular}\end{adjustbox}
\end{table}



All rows are measured on the value hold-out condition's fit pool (16,384 rows); no held-out corner % SRC: w12-deletion-baselines/FINDINGS.md
and no scored request was read. Forget minus retain is the feature-MSE gap of the deleted region
over the survivors, 3-seed mean; higher means more forgetting. Not a scored row of this study,
and printed here: approximate unlearning of a rival's head.

\begin{table}[h]
\centering\small
\caption{\textbf{Forget minus retain, 3-seed means.} Exact reference = retrain without.}
\label{tab:unlearn}
\setlength{\tabcolsep}{3pt}
\begin{adjustbox}{max width=\linewidth}\begin{tabular}{llccccc}
\toprule
scope & arm & as trained & retrain without (exact) & NegGrad+ & fine-tune on retained & seeds \\
\midrule
% SRCBLOCK: w12-deletion-baselines/FINDINGS.md
value (red; 1,352 rows) & MLP & $-0.00189$ & +0.03661 & +0.51518 & +0.00740 & 3 \\
 & continuous MLP & +0.00935 & +0.03817 & +0.35688 & +0.03832 & 3 \\
 & transformer & $-0.00054$ & +0.06057 & +1.26223 & +0.01394 & 3 \\
 & LSTM & $-0.00101$ & +0.04111 & +0.63285 & +0.01174 & 3 \\
red sphere (667 rows) & MLP & $-0.00914$ & $-0.00289$ & +0.48965 & +0.00128 & 3 \\
 & continuous MLP & +0.00750 & +0.01907 & +0.33962 & +0.01459 & 3 \\
 & transformer & $-0.00708$ & $-0.00179$ & +1.33207 & +0.00243 & 3 \\
 & LSTM & $-0.00649$ & $-0.00026$ & +0.59847 & +0.00168 & 3 \\
item (1\%; 164 rows) & MLP & $-0.00227$ & $-0.00210$ & +0.26322 & $-0.00243$ & 3 \\
 & continuous MLP & +0.00292 & +0.00304 & +0.08767 & +0.00385 & 3 \\
 & transformer & $-0.00222$ & $-0.00208$ & +0.83286 & $-0.00237$ & 3 \\
 & LSTM & $-0.00172$ & $-0.00152$ & +0.37437 & $-0.00115$ & 3 \\
% ENDSRC
\bottomrule
\end{tabular}\end{adjustbox}
\end{table}

\noindent NegGrad+ over-forgets by 9.4 (continuous MLP) to 20.8 (transformer) times the exact
reference on the value scope; fine-tuning under-forgets on three arms of four (0.20, 1.00, 0.23 and
0.29 of the exact answer); both miss in opposite directions on every seed, at matched gradient
row-visits (296,960). On the item scope the exact reference and the as-trained arm agree within
0.000354 over all arms and seeds, and NegGrad+ still destroys the deleted region for a deletion the
exact answer says costs nothing. The exact reference reads 15,032 / 15,717 / 16,220 rows (value /
conjunction / item) and rewrites about 522 kB of parameters, taking 8.5 to 39.7 s per arm; no learned
comparator here adapts after deletion while reading zero rows of data.
% SRC: w12-deletion-baselines/FINDINGS.md
% SRCBLOCK: w15-single-design-seeds/T7-3SEEDS.md; w15-single-design-seeds/FINDINGS.md
At seeds 1 and 2 every rival's retrain-without reads the same 15,032 rows, and its seconds vary by
seed (Table~\ref{tab:rivals-del-seeds}, the value scope on the 89 survivor corners). Two caveats
travel with it: no row here is a $1\times$ result, and the rivals were built on 16,384 fit rows
against the store's full pool.
% ENDSRC

\begin{table}[h]
\centering\small
\caption{\textbf{Rival retrain-without at three seeds, value scope (red), 89 survivor corners.} % SRC: w15-single-design-seeds/T7-3SEEDS.md
Composition is distinct valid novel tuples per survivor corner. ``Store $-$ rival'' is the paired
difference after deletion (the store downdated, the rival retrained without). Won / tied / lost
counts survivor corners, and is 89 / 0 / 0 for every rival on every seed.} % SRC: w15-single-design-seeds/T7-3SEEDS.md
\label{tab:rivals-del-seeds}
\setlength{\tabcolsep}{3pt}
\begin{adjustbox}{max width=\linewidth}\begin{tabular}{lccccc}
\toprule
rival & seeds & retrain seconds (0 / 1 / 2) & data rows read & rival after retrain (0 / 1 / 2) & store $-$ rival (0 / 1 / 2) \\
\midrule
% SRCBLOCK: w15-single-design-seeds/T7-3SEEDS.md
MLP & 3 & 8.86 / 16.74 / 6.35 & 15,032 & 4.022 / 2.169 / 6.697 & 46.899 / 48.360 / 45.910 \\
continuous MLP & 3 & 8.87 / 13.59 / 5.85 & 15,032 & 0.067 / 0.056 / 0.112 & 50.854 / 50.472 / 52.494 \\
transformer & 3 & 34.62 / 37.14 / 23.24 & 15,032 & 0.865 / 0.112 / 2.281 & 50.056 / 50.416 / 50.326 \\
LSTM & 3 & 15.99 / 19.91 / 11.57 & 15,032 & 3.708 / 3.921 / 8.449 & 47.213 / 46.607 / 44.157 \\
% ENDSRC
\bottomrule
\end{tabular}\end{adjustbox}
\end{table}

```

## cmazet-caption — DRAFT-v2.tex lines 2533..2535
```
(\S\ref{sec:maze-wm}); the symmetric-input rows are Table~\ref{tab:cmazet-sym}, where no condition
wins: a matched transformer given the store's own pair design matches or beats the store, so the % SRC: w19-maze-t-symmetric/FINDINGS.md
pair block, not the store, carries this table's $n$-way margin, and that row is a store row only
```

## cmazet-sym-table — DRAFT-v2.tex lines 2612..2639
```
% ---- w19 draft-figure-slots: w19 maze-t-symmetric rows, NOT LANDED. Own table, never merged with
% Table tab:cmazet (w16). Every row a PLACEHOLDER that names it and says it may land as a negative.
\begin{table}[h]
\centering\scriptsize
\caption{\textbf{CMAZE-T with symmetric inputs: pending.} The % SRC: ../../tasks/w19/maze-t-symmetric.md
instrument, banks, hold-outs, criteria, controls, floors and the store configuration are those of
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
```

## morelimits-comparators — DRAFT-v2.tex lines 2822..2826
```
\item \textbf{Comparators not run.} The approximate unlearners are measured on the fit pool and
  not scored; an extra abstain output for the non-sequence rivals is not built; the refusal
  head-to-head on the same post-deletion requests is not run; \texttt{pca\_tensor} is not run at
  $\Kstar$; a $1\times$-matched LSTM, transformer and continuous MLP are not fitted
  (Appendix~\ref{app:notrun}).
```

## placeholder-index — DRAFT-v2.tex lines 2832..2850
```
Fifteen rows remain pending, each marked in the text by a red \textsc{placeholder} whose argument names
the row and states that it may land as a negative or not at all, and each awaits a further run: the
data-matched rival set that replaces both rival sets of \S\ref{sec:shapes-t7} (one); the new
deletion cells of Table~\ref{tab:del}, measurement (b) on each scope and measurement (c) for the
value red, which the matched deletion measures (four); the
CMAZE-T per-arm generated states in Figures~\ref{fig:cmazet-families}, \ref{fig:cmazet-door-tint}
and~\ref{fig:cmazet-deletion} (three); and the seven rows of Table~\ref{tab:cmazet-sym}. Rows
that were pending earlier are now written in from their artefacts: LACE and PoE with the generation-table run
(\S\ref{sec:adapt}, Tables~\ref{tab:t7}, \ref{tab:ladder} and~\ref{tab:gentable-kstar});
the decomposition paragraph (Appendix~\ref{app:encoder-main}, Table~\ref{tab:bard-gpu}); and the
coverage control
(\S\ref{sec:ladder}). Figure slots are marked as placeholders where they depend on pending rows.
No figure is a placeholder figure and none carries an in-image banner: Figures~\ref{fig:ladder},
\ref{fig:crossover} and~\ref{fig:deletion} are drawn from the three-seed artefacts of
Tables~\ref{tab:ladder}, \ref{tab:kcurve3} and~\ref{tab:del}, and Figure~\ref{fig:ladder} takes one
panel per data-matched rival set as each set lands. Figures~\ref{fig:cmazet-families},
\ref{fig:cmazet-per-arm}, \ref{fig:cmazet-door-tint} and~\ref{fig:cmazet-deletion} draw every arm
of the words-plus-observation condition from its persisted objects; their three placeholders name
the per-arm outputs of the arms of Table~\ref{tab:cmazet}, which were never persisted.
```

## notruns — DRAFT-v2.tex lines 2853..2877
```
A row that was not run is declared with its reason, never printed as a blank or a zero.
\begin{itemize}[leftmargin=*,itemsep=1pt]
\item Any row at more than three seeds: seeds 3 and 4 pass the generation-2 cache rule
  (480,000 rows, 59 block hashes each) but have no frozen decoder and no external classifier.
  % SRC: w12b-nr6-and-kstar/FINDINGS.md
\item $K=5$ on TEST at seeds 1 and 2 (the $K=5$ TEST number is one seed: row 3 28.817, row 4
  62.122, judge accuracy 0.8285); the bridge arm at seeds 1 and 2. % SRC: w12b-nr6-and-kstar/NR6-T7.md; w13-draft-revise/DRAFT-v1.md
\item The strong (refit) \texttt{random\_atoms} control at $\Kstar$ (about an hour of
  single-thread CPU per leg). % SRC: w13-referee-2/REFEREE-v1.md
\item \texttt{pca\_tensor} at $\Kstar$; therefore whether ``the statistic buys deletion, not
  composition'' holds at $\Kstar$ is not measured.
\item A $1\times$-total-bytes LSTM, transformer, continuous MLP and table (one family, an MLP, was
  fitted at $1\times$).
\item The refusal head-to-head on the same post-deletion requests; the extra abstain output for the
  non-sequence rivals; scoring of the approximate unlearners on held-out corners (held for after
  submission).
\item The repaired maze post-deletion composition on TEST at 3 seeds (one validation measurement
  exists).
\item The maze encoder loop (contingent on a positive encoder-loop result that did not occur); the
  earlier generation cell at the longer compute pin.
\item The one experiment that would upgrade the retention law from theory to a claim (a comparator
  predicted in advance to be mispredicted).
\item A figure for the plan-versus-latent result, for the retention law, and for the $K=5\to\Kstar$
  comparison (figure slots only).
\end{itemize}
```

## sub intro-approx-unlearning — DRAFT-v2.tex: 1 replacement(s)

## sub hole-item-sentence — DRAFT-v2.tex: 1 replacement(s)

## sub c512-sentence — DRAFT-v2.tex: 1 replacement(s)

## abstract-shapes — DRAFT-v2-style.tex lines 109..117
```
On its 109 held-out corners the store, fitted on 131,072 rows, reads 47.4954 / 47.4037 / 49.1193 % SRC: w12b-nr6-and-kstar/NR6-T7.md; w13-seeds-and-tables/TABLES.md
against 30.0275 / 28.9174 / 37.5138 for the strongest read-path-matched rival of either of our two % SRC: w13-seeds-and-tables/TABLES.md
rival sets, whose 32,256 fit rows carry held-out values the store never saw; on every rung of an % SRC: _thinkpad/w4/spoke-a-rivals/rivals/seed_0/mlp.rival.json
$n$-way ladder it scores above a read-path-matched MLP fitted on the store's own 32,768 fit rows on % SRC: w12-composition-ladder/runs/shipped_seed0/rows_learned_ref_ownpool.json; w12b-nr6-and-kstar/kstar_ladder/2_seed0_K32/LEG.json
every seed, and beats an MLP matched to its whole four-part ledger on those rows by at least
$+3.117$ tuples per corner. % SRC: w12b-nr6-and-kstar/FINDINGS.md
Deletion is exact and store-scoped, but not free: across the three seeds, the object
that forgets is 4.0768 to 4.0897 times the 524,288-byte reference in its four-part ledger. We show that on this object deletion changes what the store holds but not what it answers.
% SRC: w13-referee-2/REFEREE-v1.md; w13-seeds-and-tables/TABLES.md
```

## abstract-maze-TODO — DRAFT-v2-style.tex lines 133..139
```
On a transition instrument over the same mazes the store reads 0.7246 on held-out $n$-way
transitions against 0.4035 for the best matched rival over three seeds, with every landscape % SRC: w16-maze-wm/FINDINGS.md
control collapsing; but the store reads a word request with a hand-built pair block that the
rivals, which read the observation, do not receive; when every arm is given the identical input
the margin does not survive, because a matched transformer given the store's own pair design
matches or beats it, so the pair block, not the store, carries it; and no matched arm, the store % SRC: w19-maze-t-symmetric/FINDINGS.md
included, yields a valid chained rollout.
```

## contributions — DRAFT-v2-style.tex lines 236..259
```
\item \emph{Composition at a stated partition, against read-path-matched rivals.} On a
  Shapes3D~\citep{burgess2018shapes,kim2018disentangling} value hold-out at $\Kstar=32$ wells, three scored seeds, with every % SRC: w12b-nr6-and-kstar/NR6-T7.md
  pre-declared landscape control collapsing, the store scores above the strongest read-path-matched
  learned rival of either of the paper's two rival sets on every seed, although that rival's fit rows
  carry held-out values the store never saw (\S\ref{sec:shapes-t7}). On every rung of an $n$-way
  ladder it also scores above a read-path-matched MLP fitted on the store's own 32,768 fit rows, and % SRC: w12b-nr6-and-kstar/kstar_ladder/2_seed0_K32/LEG.json
  beats an MLP matched to its whole ledger on the same rows by at least $+3.117$ per seed % SRC: w12b-nr6-and-kstar/FINDINGS.md
  (\S\ref{sec:ladder}); for $n\ge3$ the rungs are a training-data axis and not a depth axis.
\item \emph{Exact, store-scoped deletion, with its cost and its bound printed.} Value, conjunction and item deletion are light and data-free for the store and cost the rivals a full retrain. The four-part byte ledger is printed beside every row, and a probe exposes the
  limit of the claim: at $\Kstar$ the store forgets the deleted value but the object still
  answers it, because the value identity resides in the address book (\S\ref{sec:shapes-del}).
  Refusal after deletion is a separate, structural mechanism: a hole in a coupling table
  (\S\ref{sec:hole}).
\item \emph{The same properties on a maze, where deletion transfers through the
  store and composition transfers only on transitions.} Deletion transfers exactly. Composition on routes is recovered by a plan-space verb,
  which we report as an \emph{interface} result because it lifts a matched-bytes lookup
  floor more than it lifts the store (\S\ref{sec:maze}). On a transition instrument the store
  carries a composition row with every landscape control collapsing, under a word request with a
  pair block that its rivals do not receive (given the identical input, matched rivals match or
beat it, so that margin belongs to the pair block, not the store); on validation it % SRC: w19-maze-t-symmetric/FINDINGS.md
  stays ahead of rivals trained ten times longer, and its scored replication against such rivals
  keeps the margin while one landscape control misses its collapse bar on one seed; no matched
  arm's chained rollout is valid; and on an observation
  instrument the store gate holds while the margin over rivals is withdrawn (\S\ref{sec:maze-wm}).
```

## thickness — DRAFT-v2-style.tex lines 266..274
```
\noindent\emph{Thickness, stated in the introduction.} The positive composition claim rests
on a single environment (Shapes3D) at three scored seeds. No row in this study can exceed three
seeds: seeds 3 and 4 have verified caches but neither a frozen decoder nor an external
classifier, so none of their corners can be judged. % SRC: w12b-nr6-and-kstar/FINDINGS.md
On routes, where the maze's exact judge scores whole plans at three scored seeds, the result is
a negative for the store and a positive for the interface. A transition instrument on the same
engine, also with an exact judge and three scored seeds, adds one store row, scoped to a request
with a pair block that its rivals do not receive (\S\ref{sec:maze-wm}). Several rows await further runs; each is a
marked placeholder whose text names the row and states that it may land as a negative.
```

## object-deletion — DRAFT-v2-style.tex lines 331..347
```
\paragraph{The retained statistic and exact deletion.} Beside the content, the store keeps per well the running covariance, mean and count of the fit rows. A deletion proceeds in
three steps:
\begin{itemize}[leftmargin=*,nosep]
\item the contribution of the deleted rows is subtracted from the statistic;
\item the atoms are re-eigendecomposed from the downdated statistic;
\item the address book is carried across the basis change by a closed-form rotation formed at
  delete time.
\end{itemize}
No row of data is read. We use one definition of exactness throughout, scoped before any table:

\begin{quote}\small
\textsc{Exact is store-scoped.} After deleting a set of rows, the store content equals, to
float roundoff, the content fitted from the retained rows alone, holding the frozen encoder and
decoder fixed. Neither is refit at any point. It is not retrain-equivalence of the generative
model and it is not a claim that the renderer forgets.
% SRC: w13-seeds-and-tables/TABLES.md
\end{quote}
```

## t7-intro — DRAFT-v2-style.tex lines 412..424
```
Table~\ref{tab:t7} reports the scored result. Row 3 holds out 60 pair corners and row 4 holds
out 49 six-factor corners; we compare the arm with each rival on all 109 corners, with the corner
as the unit. % SRC: w12b-nr6-and-kstar/NR6-T7.md
The control gate is checked before any margin, and it was measured, not assumed, on every seed:
all four landscape controls read exactly 0.0000 on both rows on all three legs.
% SRC: w12b-nr6-and-kstar/NR6-T7.md
The rivals of Table~\ref{tab:t7} are the weaker of the two rival sets in this paper; the
store's standing against the strongest available rival is given below (\emph{Two rival sets}).
On every leg, each of them is beaten beyond the declared 1.0 margin on 109 of 109 corners; the
smallest paired delta against any comparator on any $\Kstar$ leg is $+39.266$, against the
table floor on seed 0. % SRC: w12b-nr6-and-kstar/NR6-T7.md
The seed-0 row, scored first, lies inside the three-seed spread on both rows.
% SRC: w12b-nr6-and-kstar/FINDINGS.md
```

## t7-table — DRAFT-v2-style.tex lines 435..466
```
\begin{table}[t]
\centering\small
\caption{\textbf{Composition at $\Kstar=32$, Shapes3D value hold-out, TEST, $n=3$ seeds
(0, 1, 2).} Distinct valid novel tuples per corner (ceiling 36 on row 3, 72 on row 4), per % SRC: w13-seeds-and-tables/TABLES.md
seed. Rivals are matched on read-path parameters (about 131k) and are the weaker of the paper's two rival sets (\S\ref{sec:shapes-t7}, \emph{Two rival sets}); the table is a floor, never a % SRC: w13-seeds-and-tables/TABLES.md
rival. ``Paired $\Delta$'' is the arm minus the comparator over all 109 corners (CI95 lower % SRC: w12b-nr6-and-kstar/NR6-T7.md
bound in the source); every comparator is beaten on 109/109 corners on every seed. The % SRC: w12b-nr6-and-kstar/NR6-T7.md
\texttt{refit\_postfix} rows re-fit the two token families from scratch inside the scoring
process after a dtype repair; seed 0 predates them and is a declared not-run for those two
rows.}
\label{tab:t7}
\setlength{\tabcolsep}{3.2pt}
\begin{adjustbox}{max width=\linewidth}\begin{tabular}{lcccc}
\toprule
arm / comparator & seeds & row 3 (60 pair corners) & row 4 (49 six-factor corners) & paired $\Delta$, all 109 corners \\ % SRC: w12b-nr6-and-kstar/NR6-T7.md
\midrule
% SRCBLOCK: w12b-nr6-and-kstar/NR6-T7.md
\textbf{single design, $\Kstar=32$} & 3 & \textbf{31.6000 / 32.2333 / 33.2167} & \textbf{66.9592 / 65.9796 / 68.5918} & --- \\
\quad mean $\pm$ sd & 3 & 32.350 $\pm$ 0.815 & 67.177 $\pm$ 1.320 & \\
MLP (as trained) & 3 & 2.9333 / 3.4333 / 5.2667 & 7.9592 / 3.0000 / 11.7347 & +42.303 / +44.165 / +40.945 \\
LSTM (as trained) & 3 & 2.5000 / 2.7000 / 4.9167 & 7.5306 / 7.0000 / 14.8980 & +42.734 / +42.771 / +39.716 \\
transformer (as trained) & 3 & 0.6667 / 0.6667 / 1.7167 & 1.9184 / 0.6327 / 4.7347 & +46.266 / +46.752 / +46.046 \\
continuous MLP (as trained) & 3 & 0.0000 / 0.0000 / 0.0167 & 0.0612 / 0.0612 / 0.1020 & +47.468 / +47.376 / +49.064 \\
transformer (refit post-fix) & 2 & n/r / 0.3000 / 1.7333 & n/r / 1.1429 / 3.1633 & n/r / +46.725 / +46.743 \\
LSTM (refit post-fix) & 2 & n/r / 3.0833 / 4.3667 & n/r / 5.0204 / 8.0612 & n/r / +43.450 / +43.092 \\
table (floor) & 3 & 5.8333 / 3.4833 / 6.1833 & 11.1633 / 7.0204 / 14.1224 & +39.266 / +42.330 / +39.367 \\
\midrule
four landscape controls & 3 & 0.0000 (all) & 0.0000 (all) & +47.495 / +47.404 / +49.119 \\
% ENDSRC
\bottomrule
\end{tabular}\end{adjustbox}
\end{table}
```

## t7-rendered-panels — DRAFT-v2-style.tex lines 474..496
```
% ---- w19 draft-figure-slots: T7 success / failure panels, beside Table tab:t7 (w<=14 evidence) ----
% SRCBLOCK: ../../outputs/w17-paper/figures/rendered/REQUESTS-w19.json; ../../outputs/w17-paper/figures/rendered/SAME-DECODER-w19.json; ../../outputs/w17-paper/figures/rendered/README-w19.md; ../../outputs/w19-success-failure-panels/REPORT.md
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
```

## two-rival-sets — DRAFT-v2-style.tex lines 485..546
```
\paragraph{Two rival sets, and which is stronger.} Two sets of read-path-matched learned rivals
appear in the main text, and on the same corners they score up to an order of magnitude apart.
The rivals of Table~\ref{tab:t7} are the as-trained comparators built for the deletion study
(\S\ref{sec:del-comparators}): fitted on a 16,384-row build-scale subsample of the value % SRC: w12-deletion-baselines/FINDINGS.md
hold-out fit pool, at 130,433 to 131,891 parameters, with epochs, learning rate and weight % SRC: w12-deletion-baselines/FINDINGS.md
decay inherited from the earlier rival grids and not tuned. % SRC: w12-deletion-baselines/FINDINGS.md
The rivals of Table~\ref{tab:ladder} are that earlier set: fitted on 32,256 rows of the % SRC: _thinkpad/w4/spoke-a-rivals/rivals/seed_0/mlp.rival.json
combination-split pool, at 117,878 to 119,858 parameters, 40 epochs at batch 256 and learning % SRC: _thinkpad/w4/spoke-a-rivals/REPORT.md; _thinkpad/w4/spoke-a-rivals/rivals/seed_0/mlp.json
rate 0.003 (5,040 steps), the same for every family and seed. % SRC: _thinkpad/w4/spoke-a-rivals/rivals/seed_0/mlp.json
The combination-split pool holds out combinations but no factor value, so the earlier set was
fitted on rows carrying the held-out \texttt{object\_hue} and shape values that the value
hold-out store and the rivals of Table~\ref{tab:t7} never saw. Its own record states that such a
comparison is not like-for-like and flatters whichever arm saw more, here the rival; the overlap
of its fit rows with the held-out corners was not measured.
% SRC: _thinkpad/w4/spoke-a-rivals/rivals/seed_0/mlp.rival.json; w12-composition-ladder/PREREG.md
On the shared corners the earlier set is the stronger. On the 60 pair corners the MLP reads % SRC: w12b-nr6-and-kstar/NR6-T7.md
2.9333 / 3.4333 / 5.2667 in Table~\ref{tab:t7} and 17.0167 / 22.0000 / 23.3333 in % SRC: w12b-nr6-and-kstar/NR6-T7.md; w13-seeds-and-tables/TABLES.md
Table~\ref{tab:ladder}, while the table floor reads 5.8333 / 3.4833 / 6.1833 in both. % SRC: w12b-nr6-and-kstar/NR6-T7.md; w13-seeds-and-tables/TABLES.md
Against the strongest learned rival of either set the store still scores higher on every seed.
On all 109 corners of Table~\ref{tab:t7} the store reads 47.4954 / 47.4037 / 49.1193, and the % SRC: w12b-nr6-and-kstar/NR6-T7.md
strongest earlier-set rival 30.0275 (LSTM) / 28.9174 (LSTM) / 37.5138 (transformer), scored on % SRC: w13-seeds-and-tables/TABLES.md
the same corners in the ladder's reproduction leg (a rival's score does not depend on $K$). % SRC: w12-composition-ladder/FINDINGS.md
On the ladder's $n=2$ rung the store reads 31.800 / 32.500 / 33.633 against 19.4167 / % SRC: w12b-nr6-and-kstar/KSTAR-LADDER.md; w13-seeds-and-tables/TABLES.md
22.0000 / 24.2500; on $n=3$, its lowest rung on the identical 120 corners, 60.150 / 62.783 / % SRC: w12b-nr6-and-kstar/KSTAR-LADDER.md; w13-seeds-and-tables/TABLES.md
61.908 against 44.4333 / 41.5667 / 51.9917. % SRC: w12b-nr6-and-kstar/KSTAR-LADDER.md; w13-seeds-and-tables/TABLES.md
These are score comparisons, not paired margins, and we do not lead with the margins of
Table~\ref{tab:t7} over its own rivals.
\TODO{no artefact isolates which difference between the two rival sets (fit population, fit-pool
size, parameter count or schedule) causes the gap, and none pairs the $\Kstar$ store against the
earlier set with an interval; a refit of the earlier set on the value hold-out pool exists but is
scored in no artefact.}
% SRCBLOCK: w12-deletion-baselines/FINDINGS.md; _thinkpad/w4/spoke-a-rivals/rivals/seed_0/mlp.rival.json; w13-seeds-and-tables/TABLES.md; ../../tasks/w20/shapes-data-matched.md; w12b-nr6-and-kstar/kstar_ladder/2_seed0_K32/LEG.json
\PLACEHOLDER{w20 \texttt{shapes-data-matched}: the one data-matched rival set that replaces both
rival sets above. The fit pools as they stand, per table: in Table~\ref{tab:t7} the store is fit on
131,072 rows of the value hold-out pool and its rivals on 16,384 rows of that pool (the deletion
study's declared build/smoke scale, reused as the headline comparators); in Table~\ref{tab:ladder}
the store's content is fit on 32,768 rows of each rung's pool, its data-matched MLP on the same
32,768 rows, and its other rivals on 32,256 rows of the combination-split pool, which carries the
held-out values. Pending: the four naive units refit on 131,072 rows, which is the store's own pool
for rows 3--4 and four times the ladder store's 32,768 content rows, and the store refit on the
rivals' 16,384- and 32,256-row pools, rows 3--4 and every ladder rung, 3 seeds, each rival with its
convergence evidence; which of the two tables leads is not yet decided. May land as a NEGATIVE:
against data-matched rivals the store's margin may shrink, vanish or reverse on some or all corners}
% ENDSRC

\paragraph{A rival matched on both the read path and the data.} One learned rival is already
matched on the data as well as on the read path: an MLP at the read-path pin (130,514 parameters) % SRC: w12-composition-ladder/LADDER.json; w13-seeds-and-tables/TABLES.md
fitted, on every ladder rung and seed, on the same first 32,768 rows of the rung's pool from which % SRC: w12-composition-ladder/runs/shipped_seed0/rows_learned_ref_ownpool.json; w12b-nr6-and-kstar/kstar_ladder/2_seed0_K32/LEG.json
the store's content is fitted (row ``MLP, data-matched'' of Table~\ref{tab:ladder}). The store
scores above it on every rung and seed: at $n=2$ it reads 5.5667 / 7.5833 / 9.3000 against the % SRC: w13-seeds-and-tables/TABLES.md
store's 31.800 / 32.500 / 33.633, and at $n=6$ 10.6667 / 15.1167 / 21.0333 against 65.908 / % SRC: w13-seeds-and-tables/TABLES.md; w12b-nr6-and-kstar/KSTAR-LADDER.md
66.250 / 67.650. On the 109 corners of Table~\ref{tab:t7} it reads 9.1835 / 11.1193 / 15.1284 % SRC: w12b-nr6-and-kstar/KSTAR-LADDER.md; w13-seeds-and-tables/TABLES.md
against the store's 47.4954 / 47.4037 / 49.1193; there it is fitted on 32,768 rows of the value % SRC: w12b-nr6-and-kstar/NR6-T7.md; w12-composition-ladder/runs/shipped_seed0/rows_learned_ref_ownpool.json
hold-out pool, not on the 131,072 rows from which that store is fitted, so on those corners it is % SRC: w13-seeds-and-tables/TABLES.md
matched on the population but not on the row count. It is not the strongest rival: at $n=2$ it
reads about 13 tuples per corner below the best combination-pool rival, plausibly because its pool, % SRC: w12-composition-ladder/AMENDMENT-3-CORRECTION.md
like the store's, lacks the held-out values that the combination-split pool carries. These are
score comparisons, not paired margins. The conservative comparison on the same rows is the stress
test of Appendix~\ref{app:1x}: an MLP matched to the object's whole four-part ledger (536,270 to % SRC: w12b-nr6-and-kstar/KSTAR-LADDER.md
536,972 parameters, about four times the read path), fitted on the same 32,768 rows, is beaten % SRC: w12b-nr6-and-kstar/KSTAR-LADDER.md; w12b-nr6-and-kstar/kstar_ladder/2_seed0_K32/LEG.json
beyond the pre-declared 1.0 margin on every rung and seed, with the smallest per-seed margin $+3.117$; % SRC: w12b-nr6-and-kstar/FINDINGS.md
it is a replication, not a scored row, and an MLP is the only family fitted at that size.
```

## ladder-intro — DRAFT-v2-style.tex lines 521..531
```
The ladder tests whether the store keeps composing as more factors are pinned jointly. A rung
$n$ holds out corners in which $n$ factors are jointly pinned to held-out values, with $n=2$ and
$n=6$ as the two ends. Table~\ref{tab:ladder} reports the store at $\Kstar$ on every rung and
seed, beside the read-path-matched rivals on the same corners. Because the rivals' features are
a function of the request alone and not of the rung's fit pool, their scores do not depend on
$K$; on rungs $n\ge3$ they are exactly constant by construction.
% SRC: w12-composition-ladder/FINDINGS.md
At $\Kstar$ the store's control gate collapses on all 15 legs by the pre-declared 1.0 margin:
13 of 15 read exactly 0.0000, and two ($n=3$, seeds 0 and 1) read 0.0083 on
\texttt{landscape\_shuffled}, one surviving novel tuple over 120 corners, which we print
rather than round away. % SRC: w12b-nr6-and-kstar/FINDINGS.md
```

## ladder-table — DRAFT-v2-style.tex lines 533..565
```
\begin{table}[t]
\centering\small
\caption{\textbf{The ladder at $\Kstar=32$, $n=3$ seeds, distinct valid novel tuples per % SRC: w12b-nr6-and-kstar/KSTAR-LADDER.md
corner, per seed (0 / 1 / 2).} Store rows from the $\Kstar$ replication (a free replication,
no scored budget; every row held out). Rival rows are the earlier, stronger read-path-matched rival
set (\S\ref{sec:shapes-t7}, \emph{Two rival sets}) scored on
the same corners; a rival does not depend on $K$. The data-matched MLP (130,514 parameters) is % SRC: w12-composition-ladder/LADDER.json
fitted on the same 32,768 rows as the store's content (\S\ref{sec:shapes-t7}); LACE and PoE are in % SRC: w12b-nr6-and-kstar/kstar_ladder/2_seed0_K32/LEG.json
Table~\ref{tab:lace-poe}. $n=2$: 60 corners; $n\ge3$: the identical % SRC: w12-composition-ladder/FINDINGS.md
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
% SRCBLOCK: w13-seeds-and-tables/TABLES.md
MLP & 3 & 17.0167 / 22.0000 / 23.3333 & \multicolumn{4}{c}{35.2083 / 41.5667 / 49.7500 (identical on $n=3$--$6$)} \\
LSTM & 3 & 19.4167 / 19.3333 / 23.7500 & \multicolumn{4}{c}{44.4333 / 39.4000 / 46.6750 (identical)} \\
transformer & 3 & 12.7500 / 15.7833 / 24.2500 & \multicolumn{4}{c}{28.8000 / 33.9000 / 51.9917 (identical)} \\
continuous MLP & 3 & 3.8833 / 4.5667 / 6.5333 & \multicolumn{4}{c}{6.3333 / 9.1167 / 12.6667 (identical)} \\
table (floor) & 3 & 5.8333 / 3.4833 / 6.1833 & \multicolumn{4}{c}{8.4333 / 7.9167 / 12.2333 (identical)} \\
MLP, data-matched (store's own fit rows) & 3 & 5.5667 / 7.5833 / 9.3000 & 8.2167 / 10.9083 / 16.2583 & 8.5417 / 12.5333 / 18.7667 & 9.8667 / 15.3083 / 19.7083 & 10.6667 / 15.1167 / 21.0333 \\
% ENDSRC
\bottomrule
\end{tabular}\end{adjustbox}
\end{table}
```

## ladder-paired — DRAFT-v2-style.tex lines 594..607
```
On every rung and every seed, the store's score in Table~\ref{tab:ladder} exceeds every
rival's printed score. The paired estimand with a bootstrap interval~\citep{efron1979bootstrap} against the read-path
rivals has been computed at the $K=5$ endpoint (margins $+7.0833$ to $+9.6500$ at $n=2$,
$+14.9917$ to $+22.4250$ at $n=6$, Appendix~\ref{app:k5}), but not on the $\Kstar$ legs.
% SRC: w13-seeds-and-tables/TABLES.md
\TODO{no artefact for the paired $\Kstar$ ladder margin (with CI) against the read-path-matched
rivals; the $\Kstar$ replication pairs the store only against the $1\times$ opponent of
Appendix~\ref{app:1x}.}
Two caveats accompany the rival rows: the ladder's rival columns are the earlier rival set,
whose \texttt{rival\_transformer} and \texttt{rival\_lstm} were fitted before a later dtype repair of
the token families and were not re-fitted on the ladder's rungs, yet that set is the
\emph{stronger} of the paper's two rival sets on shared corners (\S\ref{sec:shapes-t7}); and
nothing in this table is a composition-depth result.
% SRC: w12b-nr6-and-kstar/FINDINGS.md
```

## headline-steps-fig — DRAFT-v2-style.tex lines 642..651
```
\figslot{fig_headline_steps}{\textbf{The rivals were short of optimiser steps, not of data (rows
3--4, TEST, 109 corners, 3 seeds).} (a)~The MLP rival at every fit pool and step budget that was % SRC: w20-shapes-data-matched/SHAPES-MATCHED.json
run. At 20,480 steps it scores within one tuple per corner of itself on 16,384, 32,768 and 131,072 % SRC: w20-shapes-data-matched/SHAPES-MATCHED.json
rows, while the draft's comparator (16,384 rows, 1,280 steps) and the data-matched MLP (32,768 % SRC: w20-shapes-data-matched/LOG.md
rows, 2,560 steps) sit near the table floor; ten and thirty times the steps on 131,072 rows add % SRC: w20-shapes-data-matched/SHAPES-MATCHED.json
under two tuples per corner. (b)~All four families: 1,280 steps on 16,384 rows (dotted, because % SRC: w20-shapes-data-matched/LOG.md
it is another pool), then 20,480, 204,800 and, for the MLP and the continuous MLP, 614,400 steps % SRC: w20-shapes-data-matched/SHAPES-MATCHED.json; w20-shapes-data-matched/rivals/test30x/seed0/mlp/declared.json
on the store's 131,072 rows. The blue line is the store at $\Kstar=32$ on 131,072 rows; the band % SRC: w20-shapes-data-matched/SHAPES-MATCHED.json
spans its fits on 16,384, 32,768 and 131,072 rows (47.28, 47.83 and 48.01). Large markers are seed % SRC: w20-shapes-data-matched/FINDINGS.md
means; small open markers are the seeds.}{fig:headline-steps}
```

## del-intro — DRAFT-v2-style.tex lines 649..666
```
We run three deletions on the $\Kstar$ object on TEST. \emph{Value}: every fit row carrying
red. \emph{Conjunction}: every fit row carrying red \emph{and} sphere, which we call the red
sphere everywhere; the conjunction first named in the design, red $\wedge$ cube, deletes zero rows,
because cube is the shape held out of the split entirely, and it is printed as vacuous, never as
a deletion result. \emph{Item}: a random 1\% of fit rows.
Table~\ref{tab:del} gives each deletion with its cost. \emph{Forgetting a value $c$} consists of
three operations on one object: the store is downdated, the address book's statistics are
downdated, and the hole for $c$ is punched (\S\ref{sec:hole}). Each deletion row therefore prints
four measurements: (a) the store equals the store refit without the rows, a result; (b) the
address book's statistics equal those refit without the rows, a result; (c) requests for $c$ are
refused through the hole, only a display row, since the hole is a flag and symbolic refusal is
not claimed; and (d) composition on the other values is unchanged, a result. Post-deletion
adaptation of the store is \emph{light and data-free}: the covariance and count are downdated by
subtraction and the atoms are re-eigendecomposed from the downdated statistic, with no row of data
read. The (a) and (d) rows printed here were measured with the address book carried across the
basis change rather than downdated; the address-book downdate is new, and its cells are named
placeholders in Table~\ref{tab:del}. The reference, in contrast, can delete only by refitting from
scratch without the deleted rows.
```

## del-table — DRAFT-v2-style.tex lines 663..722
```
\begin{table}[t]
\centering\small
\caption{\textbf{Exact (store-scoped) deletion on the single design, $\Kstar=32$, TEST, fit pool 131,072 rows, seeds 0, 1 % SRC: w13-seeds-and-tables/TABLES.md
and 2, one row per seed ($n=1$ each): four measurements per scope.} Forgetting $c$ is: downdate the
store, downdate the address book's statistics, punch the hole. \textbf{(a) A result}: the store
equals the store refit without the rows (``exactness'', the store's maximum absolute mode deviation
against it). \textbf{(b) A result}: the address book's statistics equal those refit without the
rows; the address-book downdate is new, and every (b) cell is a named placeholder.
\textbf{(c) A display, never a result}: requests for $c$ refused through the hole, a flag in the
coupling table; symbolic refusal is not claimed. \textbf{(d) A result}: survivor $\Delta$, the
paired composition change on corners that do not request the deleted scope, measured with the
address book carried across the basis change, not downdated. Seeds 1 and 2 are the lower block, one row per seed ($n=1$ each), so
every scope prints three seeds; they re-read the single design's scored artefacts and spend no new scored run.}
\label{tab:del}
\setlength{\tabcolsep}{2.4pt}
\begin{adjustbox}{max width=\linewidth}\begin{tabular}{lccccccccc}
\toprule
scope & seeds & rows del. & store s & store rows read & ref. refit s & ref. rows read & (a) exactness & (d) survivor $\Delta$ & cos$_{\min}$ \\
\midrule
% SRCBLOCK: w13-seeds-and-tables/TABLES.md
value: red & 1 & 11,044 & 0.8998 & 0 & 9.259 & 120,028 & \sci{5.960}{-08} & +0.0000 (89) & 0.006538 \\
conj.: red $\wedge$ cube & 1 & 0 & \multicolumn{7}{c}{vacuous: cube is held out of the split; printed, never a result} \\
conj.: red sphere & 1 & 5,511 & 0.3916 & 0 & 9.262 & 125,561 & \sci{5.383}{-07} & +0.0000 (109) & 0.066688 \\
item: random 1\% & 1 & 1,311 & 0.1055 & 0 & 9.788 & 129,761 & \sci{2.338}{-07} & +0.0000 (109) & 0.079602 \\
% ENDSRC
\midrule
% SRCBLOCK: w15-single-design-seeds/T7-3SEEDS.md; w15-single-design-seeds/deletion/test_seed1_K32/deletion.json; w15-single-design-seeds/deletion/test_seed2_K32/deletion.json
value: red, seed 1 & 1 & 11,044 & 3.7503 & 0 & 18.595 & 120,028 & \sci{4.821}{-08} & 0.0000 (89) & 0.04069122680502917 \\
value: red, seed 2 & 1 & 11,044 & 1.3085 & 0 & 14.43 & 120,028 & \sci{2.011}{-08} & 0.0000 (89) & 0.28057461977018805 \\
conj.: red $\wedge$ cube, seeds 1, 2 & 2 & 0 & \multicolumn{7}{c}{vacuous at every seed: 0 rows deleted; printed, never a result} \\
conj.: red sphere, seed 1 & 1 & 5,511 & 0.7678 & 0 & 17.402 & 125,561 & \sci{1.074}{-08} & 0.0000 (109) & 0.20550653338433075 \\
conj.: red sphere, seed 2 & 1 & 5,511 & 0.8311 & 0 & 11.539 & 125,561 & \sci{6.253}{-08} & 0.0000 (109) & 0.618318259716018 \\
item: random 1\%, seed 1 & 1 & 1,311 & 0.4182 & 0 & 17.624 & 129,761 & \sci{1.486}{-08} & 0.0000 (109) & 0.06958072632551195 \\
item: random 1\%, seed 2 & 1 & 1,311 & 0.3837 & 0 & 18.649 & 129,761 & \sci{3.162}{-08} & 0.0000 (109) & 0.18117547035211387 \\
% ENDSRC
\midrule
\multicolumn{10}{l}{\textbf{(b) address-book statistics $=$ refit-without}: a result; the address-book downdate is new} \\
value: red & 3 & \multicolumn{8}{p{12cm}}{\PLACEHOLDER{w20 matched deletion, measurement (b) on the value red: the address book's statistics after the address-book downdate against those refit without the red rows, on the store's fit pool, seeds 0, 1 and 2, with measurement (d) re-read with the address book downdated rather than carried; may land as a negative}} \\
conj.: red sphere & 3 & \multicolumn{8}{p{12cm}}{\PLACEHOLDER{w20 matched deletion, measurement (b) on the red sphere: as for the value red, seeds 0, 1 and 2; may land as a negative}} \\
item: random 1\% & 3 & \multicolumn{8}{p{12cm}}{\PLACEHOLDER{w20 matched deletion, measurement (b) on the random 1\% of items: as for the value red, seeds 0, 1 and 2; may land as a negative}} \\
\midrule
\multicolumn{10}{l}{\textbf{(c) requests for $c$ refused through the hole}: a \textsc{display} row, never a result; the hole is a flag} \\
value: red & 3 & \multicolumn{8}{p{12cm}}{\PLACEHOLDER{w20 matched deletion, measurement (c) on the value red: a hole for the value red on the $\Kstar$ object; the holes punched there are red $\wedge$ cube and the red sphere, and none for the value red is in the artefacts}} \\
conj.: red sphere & 3 & \multicolumn{8}{l}{hole punched: 2 B rewritten for two holes (red $\wedge$ cube and the red sphere) in the 1,322 B coupling table, 0 rows read, no refit, every seed} \\ % SRC: w15-single-design-seeds/FINDINGS.md
item: random 1\% & 3 & \multicolumn{8}{l}{no hole: an item deletion makes no request impossible (0 of 164 on the comparator's 16,384-row pool, every seed)} \\ % SRC: w12-deletion-baselines/FINDINGS.md
\bottomrule
\end{tabular}\end{adjustbox}
\end{table}

\noindent Each row rewrites 2,081,024 bytes of the store and address book, reads 0 data rows
and refits nothing, whereas the reference reads 120,028 to 131,072 rows. % SRC: w13-seeds-and-tables/TABLES.md
On every scope the ranks of the downdated and refit stores agree, and the basis did move
($\cos_{\min}$ well below 1). % SRC: w13-seeds-and-tables/TABLES.md
% SRCBLOCK: w15-single-design-seeds/T7-3SEEDS.md; w15-single-design-seeds/FINDINGS.md; w13-seeds-and-tables/TABLES.md
At seeds 1 and 2 (the lower block of Table~\ref{tab:del}) every scope reproduces the seed-0
reading: exactness between \sci{1.074}{-08} and \sci{6.253}{-08}, survivor change 0.0000, the
ranks of the downdated and refit stores agree, and 0 data rows are read. The 2,081,024 bytes above
are seed 0's; seeds 1 and 2 rewrite 2,074,368 and 2,078,464 bytes. The rivals' retrain-without
cost at seeds 1 and 2 is Table~\ref{tab:rivals-del-seeds} (Appendix~\ref{app:delbase}).
% ENDSRC
```

## del-comparators — DRAFT-v2-style.tex lines 752..771
```
The only exact deletion route for a learned rival is to retrain from scratch without the rows.
We built that route for every rival family at matched compute, together with two approximate
unlearners, measured on the fit pool (16,384 rows; no held-out corner and no scored run).
Retraining without the red rows costs each rival 8.86 to 34.62 s and reads 15,032 data rows
(Table~\ref{tab:rivals-del}). The approximate unlearners are appendix rows
(Appendix~\ref{app:delbase}), both because they are not scored rows of this study and because
they miss the exact answer. % SRC: w12-deletion-baselines/FINDINGS.md

\noindent Against that exact reference, NegGrad+ over-forgets by 9.4 to 20.8 times and
fine-tuning on the retained rows under-forgets on three of four arms, although both received
every gradient row-visit that the exact retrain received (296,960, asserted equal across all
eight arm and method combinations on all three seeds). % SRC: w12-deletion-baselines/FINDINGS.md
On the rivals themselves, the conjunction row confirms that composition survives a conjunction
deletion: retraining a learned rival without the red sphere rows leaves red-sphere
reconstruction essentially unchanged (3-seed means $-0.00289$, $+0.01907$, $-0.00179$,
$-0.00026$), since red and sphere each survive separately and the rival composes them.
% SRC: w12-deletion-baselines/FINDINGS.md
This is composition working, not deletion failing, and it is a property of compositional
generation rather than of our store. It is also the reason why the conjunction row cannot carry a
refusal claim; the hole carries it.
```

## maze-wm — DRAFT-v2-style.tex lines 1062..1130
```
The two instruments below are the maze counterpart of \S\ref{sec:shapes}: they ask the store to
\emph{generate} a world model's transitions (CMAZE-T) and states (CMAZE-O) under the exact judge,
on random mazes never seen at test, so that the within-maze lookup floor cannot transfer and the
cross-maze fragment lookup is the floor. Rivals are matched on read floats; the pre-declared
landscape controls and blind columns are printed; criteria were pre-registered before any number; every
row has three seeds; and the two instruments never share a table (Tables~\ref{tab:cmazet}
and~\ref{tab:cmazeo}, Appendix~\ref{app:mazewm}).

% SRCBLOCK: w16-maze-wm/FINDINGS.md; w16-maze-wm/TABLES.md
\paragraph{CMAZE-T: a store row on unseen $n$-way transitions, under a pair-design request.} One pinned design (Config A1:
latent 128, $K=2$, 6 atoms, rank/width 0.094, 4,584 read floats), three TEST seeds, one pre-registered scored run each: the store generates held-out $n$-way transitions at 0.7246 (0.800 / 0.566 / 0.808) against
0.4035 (0.394 / 0.395 / 0.421) for the best matched rival, a transformer, and is ahead on every
seed. All four pre-declared landscape controls collapse below the bar of 0.575 on every seed (landscape removed
0.0000, shuffled 0.2248, shuffled across wells 0.2435, random atoms 0.1655), the learned
composition-blind arms fail (0.3418 and 0.0716 against a bar of 0.675), and the cross-maze lookup
floor reads 0.0000. The row survives rivals trained ten times longer: on validation, every rival
trained for 80,000 steps, matched or at ten times the read floats, stays below the store on every
seed (closest: seed 1, LSTM 0.437 against 0.569), and a scored replication at a wider latent (Config
C-512, three further scored runs) against 80,000-step rivals reads 0.7251 against 0.4533. A caveat travels with the
replication: on its seed 1, random atoms reach 0.464 against the store's 0.567, a margin of 0.103,
below the 0.15 bar. The row also has a ceiling set by the request design, not by the store:
validation $n$-way is 0.7247 with the same per-seed triple across about 15 configurations, and
seed 1 caps at 0.569 in all of them.
% ENDSRC

% SRCBLOCK: w16-maze-wm/PREREG-w16-maze-wm-T1.md; w16-maze-wm/FINDINGS.md; w16-maze-wm/TABLES.md; w15-cmaze-t-harness/FINDINGS.md; w14-request-layer-fix/FINAL-REPORT.md; ../../outputs/w17-paper/figures/rendered/REQUESTS-w19.json
\paragraph{CMAZE-T: the inputs are not symmetric.} The store row is not an input-matched
comparison. The store reads the request as five transition words through a \texttt{[bias | main |
pair]} design whose pair block is hand-built interaction structure, and the learned composition-blind
arms read reduced forms of the same request (main effects only, or colour identity removed); the
matched rivals regress the next state from a 636-dimensional observation read through the frozen
state encoder and receive no pair block. The design fixes which held-out $n$-way tuples are
readable, and the store decides only whether they are read cleanly, which is why validation $n$-way
sits at 0.7247 across about 15 configurations. The row therefore shows that the store is
load-bearing under this request, not that the store rather than its request design carries the
margin over the rivals; when a task-shaped interface was given to every arm (Config B, below), it
lifted the best matched rival more than the store. The symmetric-input test, in which every arm
receives the identical input, has closed with no condition winning (Table~\ref{tab:cmazet-sym},
Appendix~\ref{app:mazewm}): given the store's own pair design, a matched transformer matches or
beats the store, so the pair block, not the store, carries this row's margin over the rivals. % SRC: w19-maze-t-symmetric/FINDINGS.md
What stands on CMAZE-T is the store gate (every landscape control collapses) and exact deletion,
not a margin over matched rivals.
Two further facts bound the row. First, the rule-knowing planner columns, relaxed oracles that are
handed the engine's rules and never trained, read 0.9071 for BLIND-SEQUENTIAL, above the store's
0.7246, 0.5509 for BLIND-COLOUR and 0.6047 for BLIND-FACTOR. On CMAZE-2 the same kind of planner at
0.9077 showed that the held-out routes rarely exercised the relaxed rule (\S\ref{sec:req-layer}), and
it reads the same way here: the rule order the blind-sequential planner ignores matters only on rows
with a tinted door, a minority of the held-out $n$-way rows, so the row tests held-out word
combinations mostly on rows that do not need the door-then-tint interaction. The learned
composition-blind arms, which must generalise, fail (0.3418 and 0.0716). Second, the held-out
$n$-way rows per seed are 345 / 328 / 309 \texttt{free}, 316 / 307 / 288 \texttt{rule\_b}, 70 / 79 /
55 \texttt{interaction}, 30 / 28 / 33 \texttt{open\_door\_tint} and 0 / 0 / 1 \texttt{rule\_a}, of 761
/ 742 / 686; no per-family score exists.
% ENDSRC

% SRCBLOCK: w16-maze-wm/FINDINGS.md; w16-maze-wm/TABLES.md
\paragraph{CMAZE-T: what is not claimed.} Rollout validity is 0.000 for the store and for every
matched rival on every seed: per-step exactness is 0.925 for the store and 0.818 for the best
matched rival, and a per-step error compounds over 43 chained steps (failing link: \emph{read}). One
qualifier travels with the row: a diagnostic transformer at ten times the read floats, which is not
a matched rival, reads a non-zero validity on seed 2 (0.022). On unseen 2-way the store's 0.410 is
beaten by the learned composition-blind sequential arm (0.557), so nothing is claimed there (link:
\emph{interface}); on the never-seen colour the store's 0.465 is below the best rival's 0.511 (link:
\emph{content}). A task-shaped match-bit interface given to every arm lifts the best matched rival by
$+0.40$ and the store by $+0.20$, and fails validation (link: \emph{interface}). Exact deletion holds
on the winner: store-scoped exactness \sci{7.7}{-16} to \sci{1.2}{-15} on the statistic, deployed
atoms unchanged, the downdated read equal to the fitted-without read, in 0.03--0.13 s with no data
read, against 13.8--14.9 s and 8,000 gradient steps for a rival's retrain-without.
% ENDSRC
```

## maze-close — DRAFT-v2-style.tex lines 1162..1167
```
The maze section therefore stands on deletion transfer (CMAZE-2 and CMAZE-2b), the verb as an
interface result (CMAZE-2b), the CMAZE-T store gate and exact deletion, whose margin over matched
rivals belongs to the pair-design request and does not survive symmetric inputs, with its % SRC: w19-maze-t-symmetric/FINDINGS.md
declared negatives, and the CMAZE-O store
gate with its withdrawn margin. The budget, the configuration ladder, the fallback and every negative
with its link are in Appendix~\ref{app:mazewm}.
```

## limits-list — DRAFT-v2-style.tex lines 1177..1220
```
\begin{enumerate}[leftmargin=*,itemsep=1pt]
\item \emph{No store row on the route tasks (CMAZE-2, CMAZE-2b).} On CMAZE-2 a matched-parameter transformer beats the
  store on every composition row ($-0.0560$ / $-0.0209$ / $-0.0434$), and the landscape controls do
  not collapse (\texttt{landscape\_shuffled} 0.9509 and \texttt{random\_atoms} 0.9601 against
  1.0000 on the seen row), so no row is claimed for the store (Appendix~\ref{app:cmaze2}). On CMAZE-2b they again fail to collapse (\S\ref{sec:verb}). % SRC: w13-seeds-and-tables/TABLES.md
  On CMAZE-T the store row is scoped to a word request with a pair block that the rivals do not
  receive; given the identical input no condition wins, and the pair block, not the store, carries
  the margin (\S\ref{sec:maze-wm}, Table~\ref{tab:cmazet-sym}). % SRC: w19-maze-t-symmetric/FINDINGS.md
\item \emph{The value identity is not in the store at $\Kstar$.} Exact deletion of every red
  row leaves the red-requesting corners at 32.2500 and judge accuracy at 0.8958
  (\S\ref{sec:shapes-del}); there, only the store-scoped claim is demonstrated.
  % SRC: w13-seeds-and-tables/TABLES.md
\item \emph{Exactness is expensive in bytes.} At $\Kstar$ the deletable object is 4.0768 to 4.0897
  times the reference (Table~\ref{tab:ledger-kstar}). Across nine swept partitions the honest
  four-part total is U-shaped in $K$, with its minimum of 3.9942 times the reference at $K=24$, so
  $\Kstar$ is not the cheapest point; and on the one configuration at $K\ge36$ whose composition
  was measured ($K=40$, 7 atoms per well, 0.991 times the pin), composition collapses to 0.0648.
  On no swept partition does an object with a float64 sufficient statistic compose and forget inside 524,288 bytes at rank
  64 (the validation sweep labels itself a smoke, not a claim).
  % SRC: w13-referee-2/REFEREE-v1.md; w13-seeds-and-tables/TABLES.md; w13-draft-revise/DRAFT-v1.md
\item \emph{The encoder loop licenses no store claim.} Training the encoder through the store
  fired the pre-declared falsifier on row 3 of the shortened leg, a falsifier that involves no rival % SRC: w13-seeds-and-tables/TABLES.md
  (Appendix~\ref{app:encloop}).
\item \emph{No functioning generative baseline.} LACE and PoE, re-implemented at their own budgets
  in this feature space, render no exact-match request on the generation table and score at the level
  of our landscape controls (Table~\ref{tab:lace-poe}); the study compares naive units only and has no
  working compositional-generation baseline.
\item \emph{Post-deletion composition on the maze fails, as pre-registered} (0.1024 to 0.0000
  on CMAZE-2); the repaired mechanism rests on one validation measurement only.
  % SRC: w13-seeds-and-tables/TABLES.md
\item \emph{Refusal on the maze was novelty detection, and the maze refusal displays are
  retired.} The store refused the reachable-but-unwritten pair (0.9781) as often as the walled
  one (0.9801), at a false-refusal rate of 0.8182 on grounded-possible requests against a
  pre-registered 0.20. % SRC: w13-seeds-and-tables/TABLES.md
  A ridge probe on the same observation at $t=0$ scores AUC 0.48--0.55 across all six seed and
  tier runs: deciding solvability within budget is close to solving the maze, a property of the task rather than a defect of the model. % SRC: w14-advisor-report.md; w14-refusal-displays-closeout/FINDINGS.md
  The Shapes3D coupling hole, a different object in a different environment, is unaffected.
\item \emph{At most three seeds.} Seeds 3 and 4 cannot be judged (they have neither a decoder
  nor an external classifier), so no row can exceed three seeds.
\item \emph{Further limitations} (an unseen-valid row that belongs to a generic learner, an
  exhausted placement dial, and comparators not run) are listed in Appendix~\ref{app:morelimits}.
\item \emph{Residual novelty risk.} Abstract-level screening cannot catch a match buried in a
  method section, and very recent preprints and non-indexed workshop papers are incompletely covered; we judge the risk low, not zero. % SRC: _thinkpad/w8/w8-novelty/NOVELTY.md
\end{enumerate}
```

## leaving — DRAFT-v2-style.tex lines 1277..1284
```
The naive unit buys exact deletion by subtraction and, at a stated partition, composition on
Shapes3D that scores above the strongest read-path-matched learner of either of our rival sets
and above a read-path-matched MLP fitted on the store's own rows. However, it does not buy composition on an
exact-judge route task, where the operator is task-shaped and belongs to the interface; it does
not stop the object answering a deleted value; and it pays for exactness in bytes. Switched on, the dynamics cost composition at every strength that moves the state, while a law that no arm
of ours fails predicts their relaxation: so far, a \emph{law about the store}, not a \emph{lever on
the task}. The experiment that would make the law a claim about a class, a comparator
predicted in advance to be mispredicted in the $h/h^\star\in[0.5,2]$ bucket, is named but not run. % SRC: w10-physics-section-evidence/EVIDENCE.md
```

## ethics — DRAFT-v2-style.tex lines 1304..1308
```
This study uses a public synthetic image dataset (Shapes3D) and procedurally generated mazes; it
involves no human subjects and no personal data. Exact deletion is relevant to data-protection
requests. We claim store-scoped exactness only, and no certified unlearning, and
\S\ref{sec:shapes-del} states a case in which the exact deletion of a value does not change what
the model answers.
```

## t7more — DRAFT-v2-style.tex lines 1361..1372
```
This appendix states what Table~\ref{tab:t7} does and does not claim. The rivals are matched on
the read path, at 130,514 (MLP), 130,721 (LSTM), 130,433 (transformer) and 131,891 (continuous MLP)
parameters on seed 0; the two refit token families were refit at the same pins, width 45 / 130,433 and
width 51 / 130,721, on the same 16,384 fit rows. % SRC: w13-seeds-and-tables/TABLES.md; w12b-nr6-and-kstar/FINDINGS.md
The deletable object itself is larger than its read path: its four-part ledger, printed in
Table~\ref{tab:ledger-kstar}, is memory, not a matched quantity. The continuous-input MLP
scores at the floor (0.000 / 0.061 on seed 0); this is a fact about that rival's fit on this
population, and we do not lean on it. % SRC: w13-seeds-and-tables/TABLES.md
Appendix~\ref{app:kcurve} gives the per-seed geometry and the selection rule's val/test check.
A rival matched to the object's own four-part total would carry about 4.09 times the
read-path parameters and would be stronger; the only such evidence at $\Kstar$ is the
replication of Appendix~\ref{app:1x}, which is not a scored row. % SRC: w13-seeds-and-tables/TABLES.md
```

## lace-poe-131k — DRAFT-v2-style.tex lines 1421..1421
```
PoE (own budget, not matched) & 3 & 0 / 0 / 0 & 0.020 / 0 / 0 & rival $-$ store: $-47.49$ / $-47.40$ / $-49.12$ \\
```

## k5-and-1x — DRAFT-v2-style.tex lines 1464..1575
```

The ladder was first run at $K=5$ (MI partition, full square), before $\Kstar$ existed. Under
the one-$K$ rule of this paper it is the endpoint of the sweep, and its sign is a property of $K=5$, not of
the store. Table~\ref{tab:k5} reports the store against the read-path rivals, whose rows are
those of Table~\ref{tab:ladder}.

\begin{table}[h]
\centering\small
\caption{\textbf{Ladder at $K=5$ (MI partition), 3 seeds, the sweep's endpoint.} Store
(bit-identical to \texttt{pca\_tensor} on every rung and seed), paired margin against the best
read-path rival with the rung's own corners, and the two pre-declared landscape controls as paired deltas.
$n=2$: 60 corners; $n\ge3$: the identical 120 corners. \emph{Inference note, as for % SRC: w12-composition-ladder/FINDINGS.md
Table~\ref{tab:ladder}:} for $n\ge3$ the rungs differ in fit-pool coverage, not in composition
depth, so the store's own rise from $n=2$ to $n=6$ is not to be inferred.}
\label{tab:k5}
\setlength{\tabcolsep}{2.4pt}
\begin{adjustbox}{max width=\linewidth}\begin{tabular}{lcccccc}
\toprule
rung & seeds & store, seeds 0 / 1 / 2 & $\Delta$ vs best read-path rival & ceiling-norm. & \texttt{landscape\_shuffled} $\Delta$ & \texttt{random\_atoms} $\Delta$ \\
\midrule
% SRCBLOCK: w13-seeds-and-tables/TABLES.md
$n=2$ & 3 & 28.6333 / 31.6500 / 31.3333 & +9.2167 / +9.6500 / +7.0833 & 0.8483 & +28.6333 / +31.6500 / +31.3333 & +28.3833 / +31.5667 / +30.5833 \\
$n=3$ & 3 & 56.9500 / 59.4250 / 62.8750 & +12.5167 / +17.8583 / +10.8833 & 0.8299 & +56.9500 / +59.4250 / +62.8750 & +56.3833 / +59.2500 / +62.2250 \\
$n=4$ & 3 & 60.9333 / 62.2333 / 64.9583 & +16.5000 / +20.6667 / +12.9667 & 0.8709 & +60.9333 / +62.2333 / +64.9500 & +60.4500 / +62.0500 / +63.7167 \\
$n=5$ & 3 & 63.2417 / 64.5083 / 66.4083 & +18.8083 / +22.9417 / +14.4167 & 0.8989 & +63.2417 / +64.5083 / +66.4083 & +62.5583 / +63.9250 / +64.3750 \\
$n=6$ & 3 & 62.6917 / 63.9917 / 66.9833 & +18.2583 / +22.4250 / +14.9917 & 0.8966 & +62.6917 / +63.9917 / +66.9833 & +62.4333 / +63.8333 / +66.4917 \\
% ENDSRC
\bottomrule
\end{tabular}\end{adjustbox}
\end{table}

\noindent Nothing deteriorates as $n$ grows: ceiling-normalised, the store runs from 0.8483 to
0.8966, and for $n\ge3$ that rise may be a held-out-fraction trend rather than a depth trend. No
arm has a breaking point. We use the strong \texttt{random\_atoms} control here (address book
refit against random atoms); it reads 0.25 novel tuples at $n=2$ seed 0, not zero, so the paired
delta is printed rather than ``the arm's entire score''. % SRC: w13-seeds-and-tables/TABLES.md; w13-referee-2/REFEREE-v1.md
On every seed, the rung $n=2$ and the 49-corner prefix of $n=6$ reproduce the shipped rows 3 and
4 within each arm's across-corner spread (max abs difference 1.0833 and 0.5306 for the store,
0.0000 for all five rivals). % SRC: w12-composition-ladder/FINDINGS.md
The store and \texttt{pca\_tensor} span the same per-well subspace to \sci{1}{-8} on the three
wide wells (1,929 of 2,048 channels). On the two narrow wells their ranks differ sharply (15 vs 60,
41 vs 59), yet not one of the 2,160 judged requests changes. % SRC: w12-composition-ladder/HEAD-RULING-K.md

\paragraph{The $K=5$ ledger on the paper's convention.} On its own full-square convention, the
$K=5$ (MI partition) object is 23.886 to 24.875 times the reference over 18 legs, and
\texttt{pca\_tensor} is 1.358 to 1.372 times. % SRC: w13-seeds-and-tables/TABLES.md
Re-priced on the symmetric-triangle float64 convention used in the rest of the paper (arithmetic
on the recorded well widths, seed 0), its four-part total is 6,841,296 / 6,806,400 / 6,876,992 / 6,832,008
/ 6,835,248 B on $n=2..6$, i.e.\ 13.049 / 12.982 / 13.117 / 13.031 / 13.037 times the reference, and its
$1\times$-matched opponent would carry 1,710,324 / 1,701,600 / 1,719,248 / 1,708,002 / 1,708,812
parameters. % SRC: w13-referee-2/REFEREE-v1.md
The honest bracket for that opponent is 938,413 (float32 triangle) to 1,710,324 (float64
triangle) parameters. The opponent that the ladder actually fitted carried 3,131,564 to 3,260,030
parameters, 1.9 to 3.5 times outside the bracket; we do not print its rows.
% SRC: w13-referee-2/REFEREE-v1.md; w12b-nr6-and-kstar/FINDINGS.md
An opponent inside the bracket was fitted: the bridge leg's 1,029,776-parameter learner, on the
identical corner set, seed, cache, fit rows and schedule. Against it, the $K=5$ store's margin is
$-1.2333$ / $+10.6583$ / $-0.3917$ / $-0.1250$ / $-0.5833$ on $n=2..6$ (seed 0; one seed;
replication; not scored). The store therefore fails the every-seed bar at four rungs by at most
1.3 tuples per corner, not by 2 to 6. % SRC: w13-referee-2/REFEREE-v1.md

% ---------------------------------------------------------------------------------------
\section{A $1\times$ stress comparison at $\Kstar$ (not the paper's matching rule)}
\label{app:1x}

The paper matches rivals on the read path. As a stress test only, the $\Kstar$ replication pairs
the store with an MLP matched to the object's own four-part total (symmetric-triangle
convention): 536,270 to 536,972 parameters, fitted from the same 32,768 fit rows as the store's
content. % SRC: w12b-nr6-and-kstar/KSTAR-LADDER.md; w13-referee-2/REFEREE-v1.md
Under the ladder's own pre-declared bar, \emph{beat the opponent by 1.0 tuple per corner on every
seed}, the store clears at all five rungs; all 15 per-seed margins are positive, and the smallest
is $+3.117$. % SRC: w12b-nr6-and-kstar/FINDINGS.md

\begin{table}[h]
\centering\small
\caption{\textbf{$\Kstar$ replication against a $1\times$-total-bytes MLP (stress comparison),
3 seeds, held out, 0 scored runs.} Per-seed margins (seeds 0 / 1 / 2) and 3-seed mean.}
\label{tab:1x}
\begin{adjustbox}{max width=\linewidth}\begin{tabular}{lcccc}
\toprule
rung & seeds & opponent params & margin per seed & 3-seed mean \\
\midrule
% SRCBLOCK: w12b-nr6-and-kstar/KSTAR-LADDER.md; w13-draft-revise/DRAFT-v1.md
$n=2$ & 3 & 536,972 & +6.667 / +3.117 / +3.283 & +4.356 \\
$n=3$ & 3 & 536,270 & +17.875 / +12.808 / +11.533 & +14.072 \\
$n=4$ & 3 & 536,270 & +12.700 / +8.742 / +8.625 & +10.022 \\
$n=5$ & 3 & 536,972 & +10.350 / +6.517 / +8.200 & +8.356 \\
$n=6$ & 3 & 536,972 & +9.375 / +6.108 / +8.450 & +7.978 \\
% ENDSRC
\bottomrule
\end{tabular}\end{adjustbox}
\end{table}

\paragraph{What moved between $K=5$ and $\Kstar$, and how much of it is the opponent.} Moving
from the ladder's $K=5$ store to $\Kstar$ changes both the well count and the construction
(partition rule, store class and covariance convention together). A bridge arm holds the
$\Kstar$ construction and sets $K=5$ (seed 0 only): the margin moves $-4.567\to-0.250\to+6.667$
at $n=2$ and $-5.775\to-2.550\to+9.375$ at $n=6$, and the bridge still loses at four of five
rungs. % SRC: w12b-nr6-and-kstar/KSTAR-LADDER.md
Both steps are cuts of the opponent budget, because the opponent is defined as the object's own
total: at $n=2$ the opponent falls 3,241,076 $\to$ 1,029,776 $\to$ 534,866 parameters. Across the
construction step, the store's own score falls at four of five rungs ($+0.98$, $-2.23$, $-0.63$,
$-0.39$, $-1.97$); end to end, 71.8\% to 84.2\% of the swing is the opponent falling and 15.8\%
to 28.2\% is the store reading better. % SRC: w13-referee-2/REFEREE-v1.md
The store's own score does rise at every rung, by $+1.86$ to $+3.38$ tuples per corner (3-seed
means). % SRC: w12b-nr6-and-kstar/FINDINGS.md
The control gate at $\Kstar$ uses the weaker post-fit form of \texttt{random\_atoms} (atoms
randomised, address book not refit); the refit form is a declared not-run at $\Kstar$ and reads
0.25, not 0.0000, at $K=5$. % SRC: w13-referee-2/REFEREE-v1.md
The flip is present, seed-matched, at both ends of the only genuine depth contrast: $n=2$
($-2.783\to+4.356$) and $n=6$ ($-4.061\to+7.978$). % SRC: w12b-nr6-and-kstar/FINDINGS.md

```

## fig-deletion-caption — DRAFT-v2-style.tex lines 1754..1755
```
punches the hole; unlike the (d) rows of Table~\ref{tab:del}, which carry the book across the basis
change, the book here is refit. (a)~The store against the store refit without the rows (largest
```

## delbase — DRAFT-v2-style.tex lines 1769..1870
```
\paragraph{A sphere hazard, measured.} A damaged render could be judged a sphere for rendering
reasons and so score spuriously on the deleted conjunction. On this instrument, however, a
degraded render is judged a cube, the held-out shape, so the judged-sphere excess is negative for
both maximally degraded arms ($-0.3291$ for \texttt{landscape\_removed}, $-0.3078$ for
\texttt{random\_atoms}) against $+0.0074$ after the red-sphere deletion.
% SRC: w13-seeds-and-tables/TABLES.md


\begin{table}[h]
\centering\small
\caption{\textbf{Deletion cost for the rivals, value scope (red; 1,352 of 16,384 fit rows), % SRC: w12-deletion-baselines/FINDINGS.md
seed 0.} Data rows read and bytes rewritten are seed-invariant. ``Forget $-$ retain'' is the
feature-MSE gap of the exact retrain, 3-seed mean; the seeds column gives the count for the
seconds and for the gap. The store's own line (Table~\ref{tab:del}):
0 rows read, no refit.}
\label{tab:rivals-del}
\begin{adjustbox}{max width=\linewidth}\begin{tabular}{lccccc}
\toprule
rival, retrain without & seeds (s / gap) & seconds & data rows read & bytes rewritten & forget $-$ retain (exact) \\
\midrule
% SRCBLOCK: w12-deletion-baselines/FINDINGS.md
MLP & 1 / 3 & 8.86 & 15,032 & 522,056 & +0.03661 \\
continuous MLP & 1 / 3 & 8.87 & 15,032 & 527,564 & +0.03817 \\
transformer & 1 / 3 & 34.62 & 15,032 & 521,732 & +0.06057 \\
LSTM & 1 / 3 & 15.99 & 15,032 & 522,884 & +0.04111 \\
% ENDSRC
\bottomrule
\end{tabular}\end{adjustbox}
\end{table}



All rows are measured on the fit pool of the value hold-out condition (16,384 rows); no held-out corner % SRC: w12-deletion-baselines/FINDINGS.md
and no scored request was read. Forget minus retain is the feature-MSE gap of the deleted region
over the survivors (3-seed mean); a higher value means more forgetting. Approximate unlearning of a
rival's head, not a scored row of this study, is printed here.

\begin{table}[h]
\centering\small
\caption{\textbf{Forget minus retain, 3-seed means.} Exact reference = retrain without.}
\label{tab:unlearn}
\setlength{\tabcolsep}{3pt}
\begin{adjustbox}{max width=\linewidth}\begin{tabular}{llccccc}
\toprule
scope & arm & as trained & retrain without (exact) & NegGrad+ & fine-tune on retained & seeds \\
\midrule
% SRCBLOCK: w12-deletion-baselines/FINDINGS.md
value (red; 1,352 rows) & MLP & $-0.00189$ & +0.03661 & +0.51518 & +0.00740 & 3 \\
 & continuous MLP & +0.00935 & +0.03817 & +0.35688 & +0.03832 & 3 \\
 & transformer & $-0.00054$ & +0.06057 & +1.26223 & +0.01394 & 3 \\
 & LSTM & $-0.00101$ & +0.04111 & +0.63285 & +0.01174 & 3 \\
red sphere (667 rows) & MLP & $-0.00914$ & $-0.00289$ & +0.48965 & +0.00128 & 3 \\
 & continuous MLP & +0.00750 & +0.01907 & +0.33962 & +0.01459 & 3 \\
 & transformer & $-0.00708$ & $-0.00179$ & +1.33207 & +0.00243 & 3 \\
 & LSTM & $-0.00649$ & $-0.00026$ & +0.59847 & +0.00168 & 3 \\
item (1\%; 164 rows) & MLP & $-0.00227$ & $-0.00210$ & +0.26322 & $-0.00243$ & 3 \\
 & continuous MLP & +0.00292 & +0.00304 & +0.08767 & +0.00385 & 3 \\
 & transformer & $-0.00222$ & $-0.00208$ & +0.83286 & $-0.00237$ & 3 \\
 & LSTM & $-0.00172$ & $-0.00152$ & +0.37437 & $-0.00115$ & 3 \\
% ENDSRC
\bottomrule
\end{tabular}\end{adjustbox}
\end{table}

\noindent On the value scope, NegGrad+ over-forgets by 9.4 (continuous MLP) to 20.8 (transformer)
times the exact reference, while fine-tuning under-forgets on three arms of four (0.20, 1.00, 0.23
and 0.29 of the exact answer); both miss, in opposite directions, on every seed, at matched gradient
row-visits (296,960). On the item scope, the exact reference and the as-trained arm agree within
0.000354 over all arms and seeds, yet NegGrad+ still destroys the deleted region for a deletion
that the exact answer says costs nothing. The exact reference reads 15,032 / 15,717 / 16,220 rows
(value / conjunction / item) and rewrites about 522 kB of parameters, taking 8.5 to 39.7 s per
arm. No learned comparator here adapts after deletion while reading zero rows of data.
% SRC: w12-deletion-baselines/FINDINGS.md
% SRCBLOCK: w15-single-design-seeds/T7-3SEEDS.md; w15-single-design-seeds/FINDINGS.md
At seeds 1 and 2 every rival's retrain-without reads the same 15,032 rows, and its seconds vary by
seed (Table~\ref{tab:rivals-del-seeds}, the value scope on the 89 survivor corners). Two caveats
travel with it: no row here is a $1\times$ result, and the rivals were built on 16,384 fit rows
against the store's full pool.
% ENDSRC

\begin{table}[h]
\centering\small
\caption{\textbf{Rival retrain-without at three seeds, value scope (red), 89 survivor corners.} % SRC: w15-single-design-seeds/T7-3SEEDS.md
Composition is distinct valid novel tuples per survivor corner. ``Store $-$ rival'' is the paired
difference after deletion (the store downdated, the rival retrained without). Won / tied / lost
counts survivor corners, and is 89 / 0 / 0 for every rival on every seed.} % SRC: w15-single-design-seeds/T7-3SEEDS.md
\label{tab:rivals-del-seeds}
\setlength{\tabcolsep}{3pt}
\begin{adjustbox}{max width=\linewidth}\begin{tabular}{lccccc}
\toprule
rival & seeds & retrain seconds (0 / 1 / 2) & data rows read & rival after retrain (0 / 1 / 2) & store $-$ rival (0 / 1 / 2) \\
\midrule
% SRCBLOCK: w15-single-design-seeds/T7-3SEEDS.md
MLP & 3 & 8.86 / 16.74 / 6.35 & 15,032 & 4.022 / 2.169 / 6.697 & 46.899 / 48.360 / 45.910 \\
continuous MLP & 3 & 8.87 / 13.59 / 5.85 & 15,032 & 0.067 / 0.056 / 0.112 & 50.854 / 50.472 / 52.494 \\
transformer & 3 & 34.62 / 37.14 / 23.24 & 15,032 & 0.865 / 0.112 / 2.281 & 50.056 / 50.416 / 50.326 \\
LSTM & 3 & 15.99 / 19.91 / 11.57 & 15,032 & 3.708 / 3.921 / 8.449 & 47.213 / 46.607 / 44.157 \\
% ENDSRC
\bottomrule
\end{tabular}\end{adjustbox}
\end{table}

```

## cmazet-caption — DRAFT-v2-style.tex lines 2525..2527
```
(\S\ref{sec:maze-wm}); the symmetric-input rows are Table~\ref{tab:cmazet-sym}, where no condition
wins: a matched transformer given the store's own pair design matches or beats the store, so the % SRC: w19-maze-t-symmetric/FINDINGS.md
pair block, not the store, carries this table's $n$-way margin, and that row is a store row only
```

## cmazet-sym-table — DRAFT-v2-style.tex lines 2604..2631
```
% ---- w19 draft-figure-slots: w19 maze-t-symmetric rows, NOT LANDED. Own table, never merged with
% Table tab:cmazet (w16). Every row a PLACEHOLDER that names it and says it may land as a negative.
\begin{table}[h]
\centering\scriptsize
\caption{\textbf{CMAZE-T with symmetric inputs: pending.} The % SRC: ../../tasks/w19/maze-t-symmetric.md
instrument, banks, hold-outs, criteria, controls, floors and the store configuration are those of
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
```

## morelimits-comparators — DRAFT-v2-style.tex lines 2814..2818
```
\item \emph{Comparators not run.} The approximate unlearners are measured on the fit pool and
  not scored; an extra abstain output for the non-sequence rivals is not built; the refusal
  head-to-head on the same post-deletion requests is not run; \texttt{pca\_tensor} is not run at
  $\Kstar$; and a $1\times$-matched LSTM, transformer and continuous MLP are not fitted
  (Appendix~\ref{app:notrun}).
```

## placeholder-index — DRAFT-v2-style.tex lines 2824..2842
```
Fifteen rows remain pending, each marked in the text by a red \textsc{placeholder} whose argument names
the row and states that it may land as a negative or not at all, and each awaits a further run: the
data-matched rival set that replaces both rival sets of \S\ref{sec:shapes-t7} (one); the new
deletion cells of Table~\ref{tab:del}, measurement (b) on each scope and measurement (c) for the
value red, which the matched deletion measures (four); the
CMAZE-T per-arm generated states in Figures~\ref{fig:cmazet-families}, \ref{fig:cmazet-door-tint}
and~\ref{fig:cmazet-deletion} (three); and the seven rows of Table~\ref{tab:cmazet-sym}. Rows
that were pending earlier are now written in from their artefacts: LACE and PoE with the generation-table run
(\S\ref{sec:adapt}, Tables~\ref{tab:t7}, \ref{tab:ladder} and~\ref{tab:gentable-kstar});
the decomposition paragraph (Appendix~\ref{app:encoder-main}, Table~\ref{tab:bard-gpu}); and the
coverage control
(\S\ref{sec:ladder}). Figure slots are marked as placeholders where they depend on pending rows.
No figure is a placeholder figure and none carries an in-image banner: Figures~\ref{fig:ladder},
\ref{fig:crossover} and~\ref{fig:deletion} are drawn from the three-seed artefacts of
Tables~\ref{tab:ladder}, \ref{tab:kcurve3} and~\ref{tab:del}, and Figure~\ref{fig:ladder} takes one
panel per data-matched rival set as each set lands. Figures~\ref{fig:cmazet-families},
\ref{fig:cmazet-per-arm}, \ref{fig:cmazet-door-tint} and~\ref{fig:cmazet-deletion} draw every arm
of the words-plus-observation condition from its persisted objects; their three placeholders name
the per-arm outputs of the arms of Table~\ref{tab:cmazet}, which were never persisted.
```

## notruns — DRAFT-v2-style.tex lines 2845..2869
```
Each row that was not run is declared with its reason and is never printed as a blank or a zero.
\begin{itemize}[leftmargin=*,itemsep=1pt]
\item Any row at more than three seeds: seeds 3 and 4 pass the generation-2 cache rule
  (480,000 rows, 59 block hashes each) but have no frozen decoder and no external classifier.
  % SRC: w12b-nr6-and-kstar/FINDINGS.md
\item $K=5$ on TEST at seeds 1 and 2 (the $K=5$ TEST number is one seed: row 3 28.817, row 4
  62.122, judge accuracy 0.8285); the bridge arm at seeds 1 and 2. % SRC: w12b-nr6-and-kstar/NR6-T7.md; w13-draft-revise/DRAFT-v1.md
\item The strong (refit) \texttt{random\_atoms} control at $\Kstar$ (about an hour of
  single-thread CPU per leg). % SRC: w13-referee-2/REFEREE-v1.md
\item \texttt{pca\_tensor} at $\Kstar$; therefore whether ``the statistic buys deletion, not
  composition'' holds at $\Kstar$ is not measured.
\item A $1\times$-total-bytes LSTM, transformer, continuous MLP and table (one family, an MLP, was
  fitted at $1\times$).
\item The refusal head-to-head on the same post-deletion requests; the extra abstain output for the
  non-sequence rivals; scoring of the approximate unlearners on held-out corners (held for after
  submission).
\item The repaired maze post-deletion composition on TEST at 3 seeds (one validation measurement
  exists).
\item The maze encoder loop (contingent on a positive encoder-loop result that did not occur); the
  earlier generation cell at the longer compute pin.
\item The one experiment that would upgrade the retention law from theory to a claim (a comparator
  predicted in advance to be mispredicted).
\item A figure for the plan-versus-latent result, for the retention law, and for the $K=5\to\Kstar$
  comparison (figure slots only).
\end{itemize}
```

## sub intro-approx-unlearning — DRAFT-v2-style.tex: 1 replacement(s)

## sub hole-item-sentence — DRAFT-v2-style.tex: 1 replacement(s)

## sub c512-sentence — DRAFT-v2-style.tex: 1 replacement(s)
