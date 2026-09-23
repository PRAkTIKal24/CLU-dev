r"""w18 placeholder-fill, SECOND PASS: the text written into both drafts.
Sources (under EV = ~/clu/evidence/outputs/): w15-bar-d-score/, w15-lace-poe-rows/, w15-coverage-control/.
Every number is copied from the file its % SRC comment names; nothing is re-derived or rounded here."""

# ---------------------------------------------------------------- header comment (3 lines -> 3 lines)
HEADER_OLD = r"""%   * w18 placeholder-fill (2026-09-23): the landed w15/w16 rows are written in, each with
%     its SRC comment. Four PLACEHOLDERs remain (bar-d-score x2, lace-poe-rows x2: AWAITING,
%     second pass); coverage-control carries a TODO (artefact not staged on batmac).
"""
HEADER_NEW = r"""%   * w18 placeholder-fill (2026-09-23, two passes): the landed w15/w16 rows are written in, each
%     with its SRC comment; pass 2 filled bar-d-score x2 and lace-poe-rows x2 and closed the
%     coverage-control TODO. Ten PLACEHOLDERs remain, all w19 (w19-draft-figure-slots/FIGURE-SLOTS.md).
"""

# ---------------------------------------------------------------- tab:t7: LACE / PoE rows (replace the placeholder row)
T7_ROW_OLD = r"""LACE, PoE & 3 & \multicolumn{3}{l}{\PLACEHOLDER{w15 \texttt{lace-poe-rows}: LACE and PoE at their own budgets; may beat the store}} \\
"""
T7_ROW_NEW = r"""\midrule
% SRCBLOCK: w15-lace-poe-rows/FINDINGS.md
LACE (own budget, not matched) & 3 & 0.117 / 0.033 / 0.267 & 0.224 / 0.122 / 0.184 & rival $-$ store: $-47.33$ / $-47.33$ / $-48.89$ \\
PoE (own budget, not matched) & 3 & 0 / 0 / 0 & 0.020 / 0 / 0 & rival $-$ store: $-47.49$ / $-47.40$ / $-49.12$ \\
\multicolumn{5}{p{15cm}}{\emph{LACE and PoE (w15 \texttt{lace-poe-rows}) are added beside the rivals above, which are not replaced; each runs at its own declared budget (LACE 31.08$\times$, PoE 27.13$\times$ the 524,288 B reference, a parameter count against store-mode bytes), not on the read-path matching rule. Their $\Delta$ is printed as rival minus store, the artefact's sign; the store wins 109/109 corners against each, on every seed. Scope: \S\ref{sec:adapt}.}} \\
% ENDSRC
"""

# ---------------------------------------------------------------- tab:ladder: coverage-control row (pass-1 row, now filled)
LADDER_COV_OLD = r"""store, $n=6$ on $n=3$'s fit pool & 3 & \multicolumn{5}{l}{TEST cell \emph{derived, not measured}: the $n=3$ store cell, by construction; validation: \textsc{no rise} (text)} \\
"""
LADDER_COV_NEW = r"""store, $n=6$ on $n=3$'s fit pool & 3 & \multicolumn{4}{l}{TEST cell \emph{derived, not measured}: the $n=3$ store cell, by construction (text)} & \emph{60.150 / 62.783 / 61.908, derived} \\ % SRC: w15-coverage-control/FINDINGS.md
"""

# ---------------------------------------------------------------- tab:ladder: LACE / PoE rows on every rung (inserted after the floor row)
LADDER_FLOOR_ANCHOR = r"""table (floor) & 3 & 5.8333 / 3.4833 / 6.1833 & \multicolumn{4}{c}{8.4333 / 7.9167 / 12.2333 (identical)} \\
% ENDSRC
"""
LADDER_LACE_ROWS = r"""% SRCBLOCK: w15-lace-poe-rows/FINDINGS.md
LACE (own budget, not matched) & 3 & 0.067 / 0.050 / 0.067 & 0.133 / 0.125 / 0.208 & 0.117 / 0.217 / 0.183 & 0.083 / 0.142 / 0.183 & 0.125 / 0.100 / 0.175 \\
PoE (own budget, not matched) & 3 & 0.017 / 0.000 / 0.000 & 0.000 / 0.000 / 0.000 & 0.008 / 0.000 / 0.008 & 0.000 / 0.008 / 0.000 & 0.000 / 0.008 / 0.000 \\
% ENDSRC
"""

# ---------------------------------------------------------------- coverage paragraph: close the TODO
# Replaces pass 1's SRC-PENDING sentence (the val verdict transcribed from the Hub) and its TODO.
COV_OLD = r"""number is printed for it. On validation, the one measured leg, the verdict is \textsc{no rise}: the
restricted $n=6$ fit's rise over $n=3$ is below the pre-declared margin on two of three seeds.
% SRC-PENDING: w15-coverage-control (artefact not staged on the drafting machine; verdict per the Hub's w18 instruction)
\TODO{no artefact for w15 \texttt{coverage-control} on the drafting machine: the validation verdict
and the derived TEST cell are transcribed from the Hub's placeholder-fill instruction; the per-seed
validation scores are printed when the artefact is staged.}
"""
COV_NEW = r"""number is measured for it. % SRC: w15-coverage-control/FINDINGS.md
On TEST, $n=6$ at $n=3$'s coverage is therefore 61.614 against $n=6$'s 66.603 ($\Kstar=32$, 3 seeds): % SRC: w15-coverage-control/FINDINGS.md
the whole $n=3\to6$ rise of +4.989 (+5.758 / +3.467 / +5.742 per seed) is fit-pool coverage and % SRC: w15-coverage-control/FINDINGS.md
none of it is depth; the coverage share is 1 by construction. That TEST cell is \emph{derived, not
measured} (a new TEST configuration would be a scored run, and the \texttt{composition} target is
declared exhausted); the identity behind it was checked on the instrument: the same pool draw,
acceptance 1.000000 under either rung's bitmap, \texttt{distinct\_valid} equal to % SRC: w15-coverage-control/FINDINGS.md
\texttt{distinct\_valid\_novel} in 3,600 of 3,600 published TEST rows, and the identical 8,640 % SRC: w15-coverage-control/FINDINGS.md
corner rows. Validation, the one measured leg (60 corners of 24 rows, 3 seeds), measures it % SRC: w15-coverage-control/FINDINGS.md
directly: $n=6$ at $n=3$'s coverage equals $n=3$ to every printed decimal on every seed, 22.767 % SRC: w15-coverage-control/FINDINGS.md
(21.767 / 23.050 / 23.483), against $n=6$'s 23.478 (22.967 / 23.617 / 23.850); the coverage % SRC: w15-coverage-control/FINDINGS.md
share of the rise is 1.00 on every seed and the mask channel is 0.000. The rise itself, % SRC: w15-coverage-control/FINDINGS.md
+1.200 / +0.567 / +0.367, is below the pre-declared 1.0 margin on two of three seeds, so the % SRC: w15-coverage-control/FINDINGS.md
pre-registered validation verdict is \textsc{no rise to decompose on val}, and we report it as
that; validation sits near its 24-tuple ceiling. The inference note, verbatim, travels with % SRC: w15-coverage-control/FINDINGS.md
Table~\ref{tab:ladder}: \emph{``We beat the naive fair rivals at every $n$; the CLU's own 2-way
$\to$ 6-way rise is not to be inferred, because rungs 3--6 share corners and differ in coverage.''}
% SRC: w15-coverage-control/FINDINGS.md
"""

# ---------------------------------------------------------------- sec:adapt placeholder: LACE / PoE + generation table
ADAPT_PREFIX = r"""w15 \texttt{lace-poe-rows}: LACE~\citep{nie2021controllable} and PoE"""
ADAPT = r"""\paragraph{LACE and PoE, at their own budgets (w15 \texttt{lace-poe-rows}).}
LACE~\citep{nie2021controllable} and PoE~\citep{hinton2002training} were \emph{added} beside the
learned rivals on rows 3--4 (Table~\ref{tab:t7}) and on every ladder rung
(Table~\ref{tab:ladder}), 3 seeds; no learned-rival column was replaced. They do not beat the
store on rows 3--4, nor anywhere on the ladder: on rows 3--4 the $\Kstar$ store scores 47.50 / % SRC: w15-lace-poe-rows/FINDINGS.md
47.40 / 49.12 novel valid tuples per corner against LACE 0.165 / 0.073 / 0.229 and PoE 0.009 / % SRC: w15-lace-poe-rows/FINDINGS.md
0.000 / 0.000, and the store wins 109/109 corners against each, on every seed; on every rung it % SRC: w15-lace-poe-rows/FINDINGS.md
wins every corner against both on every seed (60 of 60 at $n=2$, 120 of 120 at $n\ge3$). % SRC: w15-lace-poe-rows/FINDINGS.md
Their budgets are their own and are not byte-matched: LACE 4,073,663 classifier parameters % SRC: w15-lace-poe-rows/FINDINGS.md
(31.08$\times$ the 524,288 B reference; 63.10$\times$ counting its whitening and mean), PoE six % SRC: w15-lace-poe-rows/FINDINGS.md
experts with 3,556,665 parameters (27.13$\times$), each fitted on 32,256 rows with 512 validation % SRC: w15-lace-poe-rows/FINDINGS.md
rows from the rung's own pool and 36 selection trials. % SRC: w15-lace-poe-rows/FINDINGS.md
\emph{This is LACE and PoE as instantiated in this feature space at their own declared budgets,
and it must be weighed that way; it is not a refutation of LACE in general.} Both render zero
exact-match requests on the generation table as well; in the lace-poe-rows spoke's own words, a
reader should weigh the result as ``LACE/PoE, as re-implemented at these budgets, do not produce
valid full tuples here'', not as ``the CLU out-composes LACE in general''. % SRC: w15-lace-poe-rows/FINDINGS.md

\paragraph{The generation table, scored run 3 of 3, on the $\Kstar$ single design.}
The $\Kstar$ single design was added to the generation-table registry and scored once
(\texttt{generation-table} run 3 of 3, 3 seeds, nothing not-run; Table~\ref{tab:gentable-kstar}, % SRC: w15-lace-poe-rows/FINDINGS.md
Appendix~\ref{app:t7more}). All four landscape controls collapse on every seed (bar S1 % SRC: w15-lace-poe-rows/FINDINGS.md
\textsc{pass}), so the store claim stands on this table. The pre-declared bar against the shipped
read (every seed, margin 0.020) \textsc{fails} on seed 0 ($-0.0185$; mean $+0.1551$), and the % SRC: w15-lace-poe-rows/FINDINGS.md
pre-registration's falsification clause fires: the claim that $\Kstar$ beats the shipped read is
\emph{not} made. $\Kstar$ also loses to the prototype ceiling \texttt{knn\_proto} on the mean % SRC: w15-lace-poe-rows/FINDINGS.md
($-0.1150$, S3 \textsc{lose}) at 4.09--4.11$\times$ the reference bytes, and beats LACE % SRC: w15-lace-poe-rows/FINDINGS.md
($+0.4468$) and PoE ($+0.5532$) on the mean (S4). % SRC: w15-lace-poe-rows/FINDINGS.md"""

# ---------------------------------------------------------------- app:t7more: the generation-table K* table (inserted at the end of the appendix)
T7MORE_ANCHOR = r"""replication of Appendix~\ref{app:1x}, which is not a scored row. % SRC: w13-seeds-and-tables/TABLES.md
"""
GENTABLE = r"""
\begin{table}[h]
\centering\scriptsize
\caption{\textbf{The generation table, $\Kstar$ single design, \texttt{generation-table} scored run 3 of 3, % SRC: w15-lace-poe-rows/FINDINGS.md
generation-2 cache.} A separate instrument from Table~\ref{tab:t7}, never merged with it:
composition-B aggregate under the external classifier over 72 held-out requests, per seed. % SRC: w15-lace-poe-rows/FINDINGS.md
Byte figures name their convention; no $\Kstar$ row is matched-bytes. Pre-declared bars: S1
(controls collapse) \textsc{pass}; S2 (beats the shipped read on every seed) \textsc{fail}, seed 0;
S3 (against \texttt{knn\_proto}) \textsc{lose}; S4 (against LACE, PoE) \textsc{win}.}
\label{tab:gentable-kstar}
\setlength{\tabcolsep}{2.4pt}
\begin{adjustbox}{max width=\linewidth}\begin{tabular}{lcccp{4.2cm}}
\toprule
arm & seeds & aggregate, seed 0 / 1 / 2 & mean & deployed bytes (convention) \\
\midrule
% SRCBLOCK: w15-lace-poe-rows/FINDINGS.md
\textbf{$\Kstar$ single design} & 3 & \textbf{0.5718 / 0.9977 / 0.6806} & \textbf{0.7500 $\pm$ 0.1807} & 2,145,738 / 2,153,018 / 2,155,618 B, four-part, never netted (4.09--4.11$\times$ ref) \\
\quad \texttt{landscape\_removed} & 3 & 0.0556 / 0.0556 / 0.2778 & 0.1296 & same object, atoms zeroed \\
\quad \texttt{landscape\_shuffled} & 3 & 0.1481 / 0.1389 / 0.0602 & 0.1157 & same bytes \\
\quad \texttt{landscape\_shuffled\_across\_wells} & 3 & 0.1065 / 0.0764 / 0.1435 & 0.1088 & same bytes \\
\quad \texttt{random\_atoms} & 3 & 0.1458 / 0.2523 / 0.1597 & 0.1860 & same bytes \\
shipped read (\texttt{clu}) & 3 & 0.5903 / 0.6065 / 0.5880 & 0.5949 $\pm$ 0.0082 & 524,288 B (store modes) \\
\texttt{store\_removed} & 3 & 0.1111 / 0.1389 / 0.1111 & 0.1204 & 524,288 B \\
\texttt{knn\_proto} (ceiling) & 3 & 0.8588 / 0.8542 / 0.8819 & 0.8650 $\pm$ 0.0122 & 475,136 B (retired-budget selection, declared) \\
LACE (\texttt{ode}) & 3 & 0.2940 / 0.3241 / 0.2917 & 0.3032 $\pm$ 0.0148 & not byte-matched \\
PoE (\texttt{ebm\_compose}) & 3 & 0.1759 / 0.1875 / 0.2269 & 0.1968 $\pm$ 0.0218 & 14,226,660 B (27.1$\times$ ref) \\
\midrule
$\Kstar$ $-$ shipped read (S2) & 3 & $-0.0185$ / $+0.3912$ / $+0.0926$ & $+0.1551$ & \textsc{fail} (seed 0) \\
$\Kstar$ $-$ \texttt{knn\_proto} (S3) & 3 & $-0.2870$ / $+0.1435$ / $-0.2014$ & $-0.1150$ & \textsc{lose} \\
% ENDSRC
\bottomrule
\end{tabular}\end{adjustbox}
\end{table}
"""

# ---------------------------------------------------------------- app:encloop placeholder: BAR D on the longer leg
BARD1_PREFIX = r"""w15 \texttt{bar-d-score}: the scored BAR D row of the longer leg"""
BARD1 = r"""\paragraph{Criterion D on the longer leg, scored (w15 \texttt{bar-d-score}).} One
\texttt{encoder-loop} scored run (2 of 3) under its own pre-registration and one amendment
written before any TEST number, 3 seeds, 0 \textsc{void} seeds, in its own table % SRC: w15-bar-d-score/FINDINGS.md
(Table~\ref{tab:bard-gpu}), never merged with Table~\ref{tab:encloop}. $M(X)$ is the mean paired
difference, store minus the matched-parameter learner refitted on arm $X$'s own features (178,952 % SRC: w15-bar-d-score/FINDINGS.md
parameters, a parameter count; the store's atom block is 524,288 B, 64 atoms $\times$ 2048 % SRC: w15-bar-d-score/FINDINGS.md
float32, with its address book printed separately and never netted); growth is
$M(\texttt{loop\_swap})-M(\texttt{alone})$. \textbf{By the pre-declared rule Criterion D is
\textsc{mixed} on row 4 and on row 3, so no claim is made in either direction.} Growth is positive
on two seeds and not on the third on each row, every row-4 growth interval spans 0, and the
margin is inside $\pm1$ novel tuple per corner in all six row-4 arm-seed cells. All three
landscape controls collapse on every scored \texttt{alone} and \texttt{loop\_swap} leg (at most % SRC: w15-bar-d-score/FINDINGS.md
0.490 per corner; the lookup argmax is a printed floor at 0.000, never a rival): the store reads % SRC: w15-bar-d-score/FINDINGS.md
what is in its atoms, but with the margin at about zero this licenses no claim of the store over
the baseline, in either direction.
\emph{Why growth could not show here (descriptive, not a bar).} \texttt{loop\_swap} saturates: the
store reaches 1.000 / 0.995 / 0.998 of the row-4 ceiling and the learner 1.000 / 0.991 / 0.997, so % SRC: w15-bar-d-score/FINDINGS.md
$M(\texttt{loop\_swap})\approx0$ by construction, and \texttt{alone}$\to$\texttt{loop\_swap} lifts
the learner as much as the store (row 4: +33.47 against +34.39, +7.18 against +6.84, +32.59 % SRC: w15-bar-d-score/FINDINGS.md
against +31.67). Better features help every arm; this is a result about the encoder, not the store. % SRC: w15-bar-d-score/FINDINGS.md
\emph{A wiring defect, found and repaired before any TEST number.} The scoring path loaded a
rival fitted on the \emph{frozen} encoder's features instead of the arm's refitted learner; the
repair loads the arm's own learner and refuses a feature-hash mismatch. The same scorer served
the shortened-decoder leg of Table~\ref{tab:encloop}; by the bar-d-score artefact, that leg's
\texttt{rival\_mlp\_180k} rows at 0.000 are the frozen-feature rival, not a measurement of the
refitted baseline, so Criterion D on the shortened leg is \emph{unmeasured}, not void, and
re-scoring it (the last \texttt{encoder-loop} run) is a declared not-run. % SRC: w15-bar-d-score/FINDINGS.md

\begin{table}[h]
\centering\small
\caption{\textbf{Criterion D and the decomposition on the longer leg only (4,000 joint steps, % SRC: w15-bar-d-score/FINDINGS.md
32,768 training images, 1,800 decoder-refit steps, 4,000 warm steps), TEST, 3 seeds, % SRC: w15-bar-d-score/FINDINGS.md
\texttt{encoder-loop} scored run 2 of 3.} Never shares a table with the shortened-decoder leg
(Table~\ref{tab:encloop}). Distinct valid novel tuples per corner (ceiling 36 on row 3, 72 on
row 4); store: the \texttt{pca\_tensor} atom block (524,288 B, float32); learner: the % SRC: w15-bar-d-score/FINDINGS.md
matched-parameter MLP refitted on each arm's features (178,952 parameters). Paired intervals are % SRC: w15-bar-d-score/FINDINGS.md
10,000-draw bootstraps over corners.} % SRC: w15-bar-d-score/FINDINGS.md
\label{tab:bard-gpu}
\setlength{\tabcolsep}{2.4pt}
\begin{adjustbox}{max width=\linewidth}\begin{tabular}{lcp{4.6cm}p{4.6cm}}
\toprule
quantity (seeds 0 / 1 / 2) & seeds & row 3 (60 corners) & row 4 (49 corners) \\ % SRC: w15-bar-d-score/FINDINGS.md
\midrule
% SRCBLOCK: w15-bar-d-score/FINDINGS.md
$M(\texttt{alone})$, store $-$ learner & 3 & $-0.517$ / +0.867 / $-2.783$ & +0.918 / $-0.061$ / $-0.837$ \\
$M(\texttt{loop\_swap})$ & 3 & +0.217 / $-0.017$ / +1.183 & +0.000 / +0.286 / +0.082 \\
growth & 3 & +0.733 / $-0.883$ / +3.967 & $-0.918$ / +0.347 / +0.918 \\
growth, CI95 & 3 & [$-0.950$, +2.450] / [$-2.133$, +0.400] / [+1.983, +6.117] & [$-3.245$, +1.694] / [$-1.163$, +1.898] / [$-2.163$, +4.000] \\
\textbf{Criterion D (pre-declared rule)} & 3 & \textsc{mixed}: no claim & \textsc{mixed}: no claim \\
store, \texttt{alone} / \texttt{loop\_swap} & 3 & 14.650 / 30.783 / 16.883; 35.817 / 35.767 / 35.650 & 38.531 / 64.449 / 39.286; 72.000 / 71.633 / 71.878 \\
learner, \texttt{alone} / \texttt{loop\_swap} & 3 & 15.167 / 29.917 / 19.667; 35.600 / 35.783 / 34.467 & 37.612 / 64.510 / 40.122; 72.000 / 71.347 / 71.796 \\
three landscape controls & 3 & \multicolumn{2}{c}{collapse on every scored leg, both rows (at most 0.490 per corner; bar $\le1.0$)} \\
\midrule
$A'$ (unfreeze: \texttt{loop\_nosw} $-$ \texttt{alone}) & 3 & +18.367 / $-10.867$ / +18.600 & +32.245 / $-19.367$ / +32.224 \\
$A''$ (swap loss: \texttt{loop\_swap} $-$ \texttt{loop\_nosw}) & 3 & +2.800 / +15.850 / +0.167 & +1.224 / +26.551 / +0.367 \\
ordering (pre-declared rule) & 3 & \multicolumn{2}{c}{$A'<0$ in 2/6 cells, $A''>0$ in 6/6: \textsc{not reproduced}} \\
% ENDSRC
\bottomrule
\end{tabular}\end{adjustbox}
\end{table}"""

# ---------------------------------------------------------------- app:encoder-main placeholder: the decomposition re-stated on the longer leg
BARD2_PREFIX = r"""w15 \texttt{bar-d-score} (one \texttt{encoder-loop} scored run, new"""
BARD2 = r"""\paragraph{The decomposition re-stated on the longer leg (w15 \texttt{bar-d-score}).} On the
longer, full-decoder leg (Table~\ref{tab:bard-gpu}, a block of its own, never merged with the
shortened-decoder leg of Table~\ref{tab:encloop}) the ordering is \textbf{not reproduced} by its
pre-declared rule, which asks for $A'<0$ and $A''>0$ each in at least four of six cells: $A'$ is
negative in only 2 of 6 cells, while $A''$ is positive in 6 of 6 (its interval excludes 0 in 5 of % SRC: w15-bar-d-score/FINDINGS.md
6). The swap-loss half survives and strengthens (positive in 4 of 6 cells on the shortened leg, 6 % SRC: w15-bar-d-score/FINDINGS.md
of 6 on the longer one); the ``unfreezing hurts'' half does not survive. Unfreezing \emph{helps} by
about +32 tuples per corner on row 4 at seeds 0 and 2 and hurts only at seed 1, the one seed whose % SRC: w15-bar-d-score/FINDINGS.md
\texttt{alone} control trained well (probe margin over random initialisation $-0.0505$ / % SRC: w15-bar-d-score/FINDINGS.md
$+0.0242$ / $-0.0862$; row-4 score 38.5 / 64.4 / 39.3). ``Unfreezing hurts'' is therefore not a % SRC: w15-bar-d-score/FINDINGS.md
general finding: at these pins it is a property of one seed's well-trained control. What both legs
support is that the swap loss helps. Criterion D is \textsc{mixed} on both rows, so the margin
does not grow in the pre-declared sense and no claim is made either way; the features improved
for every arm, \texttt{loop\_swap} lifting the matched learner as much as the store, both ending at
0.99--1.00 of the 72-tuple ceiling, which is a finding about the encoder, not about the store. % SRC: w15-bar-d-score/FINDINGS.md"""

# ---------------------------------------------------------------- Placeholder index (app:placeholders), rewritten to the true state
INDEX_OLD = r"""Four rows remain pending, each marked in the text by a red \textsc{placeholder} whose argument names
the row and states that it may land as a negative; they belong to runs that had not landed when the
other rows were written in: LACE and PoE rows and the generation-table run (\S\ref{sec:adapt},
Table~\ref{tab:t7}); Criterion D and its decomposition paragraph (Appendices~\ref{app:encoder-main}
and~\ref{app:encloop}). One filled row carries an orange \textsc{todo}: the coverage control
(\S\ref{sec:ladder}), whose verdict is written in but whose artefact was not available to the
drafting machine. Figure slots are marked as placeholders where they depend on pending rows.
"""
INDEX_NEW = r"""Ten rows remain pending, each marked in the text by a red \textsc{placeholder} whose argument names
the row and states that it may land as a negative or not at all; all ten await the w19 wave: the
CMAZE-T per-arm generated states in Figures~\ref{fig:cmazet-families}, \ref{fig:cmazet-door-tint}
and~\ref{fig:cmazet-deletion} (three), and the seven rows of Table~\ref{tab:cmazet-sym}
(\texttt{maze-t-symmetric}). The rows that were pending when the w15 and w16 rows were first
written in are now written in from their artefacts: LACE and PoE with the generation-table run
(\S\ref{sec:adapt}, Tables~\ref{tab:t7}, \ref{tab:ladder} and~\ref{tab:gentable-kstar});
Criterion D and its decomposition paragraph (Appendices~\ref{app:encloop}
and~\ref{app:encoder-main}, Table~\ref{tab:bard-gpu}); and the coverage control
(\S\ref{sec:ladder}). Figure slots are marked as placeholders where they depend on pending rows.
"""
