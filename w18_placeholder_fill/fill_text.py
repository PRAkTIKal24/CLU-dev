r"""The new LaTeX text for w18 placeholder-fill. Every number is copied from the artefact named in
the % SRC / % SRCBLOCK comment beside it (paths relative to ~/clu/evidence/outputs/). The same
text goes into DRAFT-v2.tex and DRAFT-v2-style.tex, so the style diff-check stays empty.
Nothing here is rounded or re-derived; the values are the ones printed in the cited files."""

# ---------------------------------------------------------------------------------------------
# P4 coverage-control (sec:ladder). The w15 artefact is NOT staged on batmac: the construction
# argument is sourced; the validation verdict is the Hub's (launch instruction) and carries a TODO.
COVERAGE = r"""\textbf{Result (w15 \texttt{coverage-control}).} For $n\ge3$ the ladder's rise is fit-pool
coverage, and by construction it can only be coverage: rungs 3--6 probe the identical 120 corners, % SRC: w12-composition-ladder/FINDINGS.md
so $n=6$ refitted on $n=3$'s fit pool \emph{is} rung $n=3$. Its TEST cell is therefore
\emph{derived, not measured}: it is the $n=3$ store cell of Table~\ref{tab:ladder}, and no new
number is printed for it. On validation, the one measured leg, the verdict is \textsc{no rise}: the
restricted $n=6$ fit's rise over $n=3$ is below the pre-declared margin on two of three seeds.
% SRC-PENDING: w15-coverage-control (artefact not staged on the drafting machine; verdict per the Hub's w18 instruction)
\TODO{no artefact for w15 \texttt{coverage-control} on the drafting machine: the validation verdict
and the derived TEST cell are transcribed from the Hub's placeholder-fill instruction; the per-seed
validation scores are printed when the artefact is staged.}"""

LADDER_ROW_OLD = r"""\quad 3-seed mean & 3 & 32.644 & 61.614 & 66.086 & 66.581 & 66.603 \\
% ENDSRC
"""
LADDER_ROW_NEW = r"""\quad 3-seed mean & 3 & 32.644 & 61.614 & 66.086 & 66.581 & 66.603 \\
% ENDSRC
store, $n=6$ on $n=3$'s fit pool & 3 & \multicolumn{5}{l}{TEST cell \emph{derived, not measured}: the $n=3$ store cell, by construction; validation: \textsc{no rise} (text)} \\
"""

# ---------------------------------------------------------------------------------------------
# P5 tab:del rows at seeds 1 and 2
DEL_ROW_OLD = r"""seeds 1, 2 & 2 & \multicolumn{8}{l}{\PLACEHOLDER{w15 \texttt{single-design-seeds}: the three deletion rows at seeds 1--2; any may land as a negative}} \\
"""
DEL_ROW_NEW = r"""\midrule
% SRCBLOCK: w15-single-design-seeds/T7-3SEEDS.md; w15-single-design-seeds/deletion/test_seed1_K32/deletion.json; w15-single-design-seeds/deletion/test_seed2_K32/deletion.json
value: red, seed 1 & 1 & 11,044 & 3.7503 & 0 & 18.595 & 120,028 & \sci{4.821}{-08} & 0.0000 (89) & 0.04069122680502917 \\
value: red, seed 2 & 1 & 11,044 & 1.3085 & 0 & 14.43 & 120,028 & \sci{2.011}{-08} & 0.0000 (89) & 0.28057461977018805 \\
conj.: red $\wedge$ cube, seeds 1, 2 & 2 & 0 & \multicolumn{7}{c}{vacuous at every seed: 0 rows deleted; printed, never a result} \\
conj.: red sphere, seed 1 & 1 & 5,511 & 0.7678 & 0 & 17.402 & 125,561 & \sci{1.074}{-08} & 0.0000 (109) & 0.20550653338433075 \\
conj.: red sphere, seed 2 & 1 & 5,511 & 0.8311 & 0 & 11.539 & 125,561 & \sci{6.253}{-08} & 0.0000 (109) & 0.618318259716018 \\
item: random 1\%, seed 1 & 1 & 1,311 & 0.4182 & 0 & 17.624 & 129,761 & \sci{1.486}{-08} & 0.0000 (109) & 0.06958072632551195 \\
item: random 1\%, seed 2 & 1 & 1,311 & 0.3837 & 0 & 18.649 & 129,761 & \sci{3.162}{-08} & 0.0000 (109) & 0.18117547035211387 \\
% ENDSRC
"""
DEL_TITLE_OLD = r"""single design, $\Kstar=32$, TEST, seed 0 % SRC: w13-seeds-and-tables/TABLES.md
($n=1$).}"""
DEL_TITLE_NEW = r"""single design, $\Kstar=32$, TEST, seeds 0, 1 % SRC: w13-seeds-and-tables/TABLES.md
and 2, one row per seed ($n=1$ each).}"""
DEL_CAP_OLD = r"""request the deleted scope. Seeds 1 and 2 are a placeholder row below.}"""
DEL_CAP_NEW = r"""request the deleted scope. Seeds 1 and 2 are the lower block, one row per seed ($n=1$ each), so
every scope prints three seeds; they re-read the single design's scored artefacts and spend no new scored run.}"""

# P6 prose after tab:del
DEL_PROSE = r"""% SRCBLOCK: w15-single-design-seeds/T7-3SEEDS.md; w15-single-design-seeds/FINDINGS.md; w13-seeds-and-tables/TABLES.md
At seeds 1 and 2 (the lower block of Table~\ref{tab:del}) every scope reproduces the seed-0
reading: exactness between \sci{1.074}{-08} and \sci{6.253}{-08}, survivor change 0.0000, the
ranks of the downdated and refit stores agree, and 0 data rows are read. The 2,081,024 bytes above
are seed 0's; seeds 1 and 2 rewrite 2,074,368 and 2,078,464 bytes. The rivals' retrain-without
cost at seeds 1 and 2 is Table~\ref{tab:rivals-del-seeds} (Appendix~\ref{app:delbase}).
% ENDSRC

\paragraph{Deletion is not capability erasure, at three seeds.} After every red row is deleted,
the 20 deleted (red-requesting) corners still compose red on all three seeds: 32.250 $\to$ 32.250 % SRC: w15-single-design-seeds/FINDINGS.md
novel valid tuples per corner (judge accuracy 0.8958 $\to$ 0.8958) on seed 0, 33.500 $\to$ 33.500 % SRC: w15-single-design-seeds/FINDINGS.md
(0.9306 $\to$ 0.9306) on seed 1, and 33.600 $\to$ 33.550 (0.9333 $\to$ 0.9319) on seed 2. % SRC: w15-single-design-seeds/FINDINGS.md
The deletion claim on this object is store-scoped exactness plus a data-free adaptation cost,
not capability erasure. Removing the capability is the hole's job (\S\ref{sec:hole}), and the
hole is a display row with a declared not-run comparator, so it is not claimed."""

# ---------------------------------------------------------------------------------------------
# P8 CMAZE-2b deletion replication: prose + its own table (CMAZE-2b ONLY)
MAZE2B = r"""The replication on CMAZE-2b, the environment of the verb result (\S\ref{sec:verb}), is its own
table, never merged with Table~\ref{tab:maze-del}, and spends no scored run (Table~\ref{tab:maze-del-2b},
Appendix~\ref{app:verb}).
% SRCBLOCK: w15-maze-deletion-2b/TABLE.md; w15-maze-deletion-2b/FINDINGS.md
The downdate matches the refit to a float64 statistic residual of \sci{7.212}{-16} to
\sci{9.093}{-16} on all three seeds, with a projector residual of 0.000, and a clean checkout
reproduces both. Deletion does not induce refusal of the deleted word: green refusal falls on all
three seeds, so the claim is store-scoped exactness and nothing more.
% ENDSRC"""

MAZE2B_APP = r"""\paragraph{Exact deletion on CMAZE-2b in full.} Table~\ref{tab:maze-del-2b} is the CMAZE-2b
replication of \S\ref{sec:maze-del}: its own table, never merged with Table~\ref{tab:maze-del}, and
spending no scored run.
% SRCBLOCK: w15-maze-deletion-2b/TABLE.md; w15-maze-deletion-2b/FINDINGS.md
The downdate matches the refit to a float64 statistic residual of \sci{7.212}{-16} to
\sci{9.093}{-16} on all three seeds, with a projector residual of 0.000; the projector is built
from float32 atoms and cannot resolve below float32, so the two are quoted together. A clean
checkout reproduces both. Exact is not light in extent: the exact-head downdate changes every atom
and about 99.98\% of the address book, in about 2 s and with 0 passes over retained data, and the
surviving targets' validation validity moves by $-0.0198$ / $-0.0021$ / $-0.0059$; the
pre-registered basis-change carry again destroys them (to 0.0000). Deletion does not induce refusal
of the deleted word: green refusal falls on all three seeds (0.6333 $\to$ 0.5333, 0.7248 $\to$
0.4404, 0.5638 $\to$ 0.5319), so the claim is store-scoped exactness and nothing more. Speed is not
the differentiator either: the store's downdate (0.802 to 0.917 s) is close to refitting its own
store and head from the survivors. The difference is that it is exact without keeping the items,
while the matched transformer must keep about 1,200 rows and read them 256,000 times (88.6 to
90.8 s). The TEST-leg adaptation composition and the MLP and LSTM retrain-without are declared
not-run: reading TEST would be a scored act, and only the strongest printed rival was timed.
% ENDSRC

\begin{table}[h]
\centering\small
\caption{\textbf{Exact deletion on the maze, CMAZE-2b, 3 seeds, 0 scored runs.} This table % SRC: w15-maze-deletion-2b/TABLE.md
contains no CMAZE-1 or CMAZE-2 row. Delete the green target; store-scoped. Byte convention: the
deployed read is float32 (atoms plus address book); the sufficient statistic is float64, retained,
printed beside it and never netted in. Adaptation rows are on validation.}
\label{tab:maze-del-2b}
\begin{adjustbox}{max width=\linewidth}\begin{tabular}{lcc}
\toprule
property & seeds & seed 0 / seed 1 / seed 2 \\
\midrule
% SRCBLOCK: w15-maze-deletion-2b/TABLE.md; w15-maze-deletion-2b/FINDINGS.md
rows removed / kept & 3 & 224/1218 $\cdot$ 225/1205 $\cdot$ 211/1240 \\
statistic, max relative residual (float64; downdate vs refit) & 3 & \sci{7.212}{-16} / \sci{9.093}{-16} / \sci{8.937}{-16} \\
subspace, max projector residual (float32 atoms) & 3 & 0.000 / 0.000 / 0.000 \\
clean checkout: statistic residual / projector & 3 & reproduced on every seed / 0.000 \\
ledger: deployed read, float32 & 3 & 410688 floats = 1642752 B (atoms 7936 + address book 402752) \\
ledger: sufficient statistic, float64 retained & 3 & 35904 B \\
exact-head downdate: seconds & 3 & 2.042 / 2.328 / 2.181 \\
\quad address-book floats changed (of 402752) & 3 & 402656 / 402656 / 402674 \\
\quad VAL other-target validity, before $\to$ after & 3 & 0.2609 $\to$ 0.2411 / 0.1849 $\to$ 0.1828 / 0.2490 $\to$ 0.2431 \\
\quad VAL green refusal, before $\to$ after & 3 & 0.6333 $\to$ 0.5333 / 0.7248 $\to$ 0.4404 / 0.5638 $\to$ 0.5319 \\
basis-change carry (pre-registered): seconds & 3 & 0.009 / 0.004 / 0.012 \\
\quad VAL other-target validity after & 3 & 0.0000 / 0.0000 / 0.0000 \\
store downdate, store + head: seconds & 3 & 0.875 / 0.917 / 0.802 \\
rival retrain-without (transformer, 405776 params; target 410688): seconds & 3 & 89.8 / 90.8 / 88.6 \\
\quad row reads of retained data; items kept & 3 & 256000; yes (every seed) \\
% ENDSRC
\bottomrule
\end{tabular}\end{adjustbox}
\end{table}"""

# ---------------------------------------------------------------------------------------------
# P9 crossover-seeds (app:kcurve)
KCURVE = r"""\textbf{At three seeds (w15 \texttt{crossover-seeds}, 0 scored runs).} Table~\ref{tab:kcurve3} prints
every $K$ at $n=3$ and supersedes the $n=1$ middle rows of Table~\ref{tab:kcurve}.
% SRCBLOCK: w15-crossover-seeds/CROSSOVER-3SEED.md; w15-crossover-seeds/FINDINGS.md; w13-seeds-and-tables/TABLES.md
The frozen rule, applied unchanged, still returns $\Kstar=32$ and no crossover: no $K$ drops more
than 0.1 below $K=5$. Two sentences above are seed-0 statements and do not survive three seeds: at
$n=3$ every drop is at most zero, the largest drop anywhere is 0.0000, at $K=5$ itself, and the
$+0.012731$ at $K=10$ printed above is a single-seed (seed 0) value. The rule is not resolvable at $n=3$: the seed sd
exceeds the frozen 0.02 tolerance at six of the nine $K$, including $K=32$ (0.0259), so $\Kstar=32$
is what the rule returns, not a $K$ that three seeds distinguish from $K=24$ or $K=28$. The honest
four-part total is U-shaped at $n=3$, with the per-seed minimum at $K=24$ on all three seeds, and
that minimum is resolvable: the $K=24$ seed sd of 7,060 B is cleared by both neighbour gaps (12,203
B and 30,507 B). The floor over the sweep and all three seeds is 3.9675 times the reference, at
$K=24$, seed 1. The $K=20$ row fails the strict control gate:
\texttt{landscape\_shuffled\_across\_wells} reads 0.0139 on seed 1 (worst of 24 corners; mean
0.000579) against the arm's 0.9213, so no row may be claimed at $K=20$; the primary
\texttt{landscape\_shuffled} control collapses at every $K$. The across-wells control here is
w12's, a uniform permutation with only the identity rejected, so about one well in $K$ keeps its
own atoms; a derangement version is a declared not-run, kept non-deranged for comparability with
w12. Seeds 3 and 4 are a declared not-run.
% ENDSRC

\begin{table}[h]
\centering\small
\caption{\textbf{The $K$-curve at $n=3$ seeds on every row, validation (a smoke, not a claim).} % SRC: w15-crossover-seeds/CROSSOVER-3SEED.md
Byte convention: the honest four-part total, summed and never netted: atoms (float32), covariance
and count (float64, as retained), address book, and the 6,610 B coupling/hole table; 524,288 B is a % SRC: w15-crossover-seeds/CROSSOVER-3SEED.md
reference, not a budget. The seed sd is printed beside every mean. $K=20$ fails the strict control gate (text).} % SRC: w15-crossover-seeds/CROSSOVER-3SEED.md
\label{tab:kcurve3}
\setlength{\tabcolsep}{2.4pt}
\begin{adjustbox}{max width=\linewidth}\begin{tabular}{rcccccccc}
\toprule
$K$ & seeds & rank/width & val comp. mean & seed sd & drop vs $K{=}5$ & four-part B mean & seed sd B & strict gate \\
\midrule
% SRCBLOCK: w15-crossover-seeds/CROSSOVER-3SEED.md
5 & 3 & 0.1561--0.1565 & 0.8953 & 0.0065 & +0.0000 & 4,118,034 & 0 & pass \\
8 & 3 & 0.2500--0.2500 & 0.9037 & 0.0197 & $-0.0085$ & 2,959,602 & 0 & pass \\
10 & 3 & 0.3122--0.3137 & 0.9014 & 0.0211 & $-0.0062$ & 2,606,754 & 0 & pass \\
12 & 3 & 0.3743--0.3765 & 0.9082 & 0.0141 & $-0.0129$ & 2,393,714 & 0 & pass \\
16 & 3 & 0.5000--0.5000 & 0.9111 & 0.0203 & $-0.0158$ & 2,177,330 & 0 & pass \\
20 & 3 & 0.5294--0.6275 & 0.9178 & 0.0285 & $-0.0226$ & 2,098,879 & 2,864 & \textbf{fail} \\
24 & 3 & 0.4353--0.7529 & 0.9223 & 0.0235 & $-0.0270$ & 2,086,677 & 7,060 & pass \\
28 & 3 & 0.3425--0.8767 & 0.9267 & 0.0276 & $-0.0314$ & 2,117,183 & 9,885 & pass \\
32 & 3 & 0.2656--0.9844 & 0.9290 & 0.0259 & $-0.0338$ & 2,140,538 & 3,640 & pass \\
% ENDSRC
\bottomrule
\end{tabular}\end{adjustbox}
\end{table}"""

# ---------------------------------------------------------------------------------------------
# P10 rival retrain-without at seeds 1 and 2 (app:delbase)
RIVALS = r"""% SRCBLOCK: w15-single-design-seeds/T7-3SEEDS.md; w15-single-design-seeds/FINDINGS.md
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
\end{table}"""

# ---------------------------------------------------------------------------------------------
# P12 harness smokes (app:mazewm)
HARNESS = r"""Both harnesses shipped ready.
% SRCBLOCK: w15-cmaze-t-harness/FINDINGS.md; w15-cmaze-t-harness/HARNESS-READY.json
\emph{CMAZE-T.} An independent replay of 164,767 transitions (55,285 / 54,710 / 54,772 over the
three seeds), by two drivers written from the rules, finds 0 disagreements with the harness's
stepper, and 0 of 164,767 rows lacks an accepting emission, so no impossible example reaches a
non-refuser arm. The seeded validation smoke of the Config-A single design reads, for the store,
0.6911 on unseen $n$-way (0.802 / 0.468 / 0.803), 0.3932 on unseen 2-way, 0.9709 seen and 0.5386
on the value hold-out. On unseen 2-way the across-wells control reads 0.2535 against a bar of
0.2432, so that row is an interface claim only. The harness ships three warnings, all printed here:\\
\texttt{composition\_blind\_PLANNER\_scores\_near\_the\_clu\_on\_unseen\_nway},\\
\texttt{charter\_C\_control\_did\_not\_collapse}, and\\
\texttt{cross\_maze\_lookup\_floor\_scores\_near\_the\_clu}.
% ENDSRC
% SRCBLOCK: w15-cmaze-o-harness/FINDINGS.md
\emph{CMAZE-O.} A second checker replays 105,408 judged states (35,136 per seed) with 0
disagreements and 0 layout-bank rebuild mismatches; 30,588 requests (10,196 per seed) are checked
and 0 are impossible. Its validation smoke says one thing against itself: the charter-C controls
do not collapse at the span fraction Config-A deploys (0.688), so the smoke is an interface claim,
not a store claim, and the interface there is 201,312 of 202,848 read floats (99.24\%).
% ENDSRC
% SRCBLOCK: w16-maze-wm/FINDINGS.md; w15-cmaze-t-harness/FINDINGS.md
\emph{A control defect these smokes carry.} Both harnesses shipped
\texttt{landscape\_shuffled\_across\_wells} dealing well blocks with a random permutation, which at
$K=2$ is the identity half the time; on the world-model runs' first Config-A validation run it was
the identity on seeds 0 and 1 (the control equal to the store to every digit, 0.8019 and 0.5689).
The world-model runs replaced it with a derangement before any scored run (deranged, the control
reads 0.2504 on validation $n$-way, against 0.489 before), and every across-wells cell in
Tables~\ref{tab:cmazet} and~\ref{tab:cmazeo} uses the derangement. The harness smoke numbers above
used the original control and were \emph{not} re-measured; at the CMAZE-T smoke's $K=4$ and $K=8$ a
random permutation still carries the same partial-identity risk, so its across-wells cells,
the 0.2535 above included, carry this caveat.
% ENDSRC"""

# ---------------------------------------------------------------------------------------------
# P13-P20 tab:cmazet (CMAZE-T ONLY) and P21-P27 tab:cmazeo (CMAZE-O ONLY): whole floats
CMAZET = r"""\begin{table}[h]
\centering\scriptsize
\caption{\textbf{Maze world-model rows: CMAZE-T (transition generation), TEST, Config A1, 3 seeds % SRC: w16-maze-wm/TABLES.md
(scored runs 1--3).} Given the grid state and an action, generate the next state; exact
transition judge. Config A1: latent 128, $K=2$, 6 atoms, one design pinned across seeds. % SRC: w16-maze-wm/TABLES.md
Rivals are matched on read floats (MLP 4,781, $+4.3\%$; LSTM 4,290, $-6.4\%$; transformer 5,129, % SRC: w16-maze-wm/TABLES.md
$+11.9\%$, against the store's 4,584). % SRC: w16-maze-wm/TABLES.md
Byte convention: deployed bytes $=4\times$(atom $+$
address-book floats), float32; the float64 sufficient statistic is printed beside it, never netted.
The across-wells control is the deranged one. Values are mean (seed 0 / 1 / 2). This table contains
no CMAZE-O, CMAZE-1, CMAZE-2 or CMAZE-2b row.}
\label{tab:cmazet}\label{tab:cmaze-del}
\setlength{\tabcolsep}{2pt}
\begin{adjustbox}{max width=\linewidth}\begin{tabular}{p{1.9cm}p{1.2cm}p{10.0cm}}
\toprule
row & seeds & CMAZE-T \\
\midrule
% SRCBLOCK: w16-maze-wm/TABLES.md; w16-maze-wm/FINDINGS.md
seen & 3 & sanity and retrieval row, \emph{no claim}: store 0.9682 (0.967 / 0.973 / 0.965); transformer 0.7861; cross-maze word lookup 0.9988; within-maze lookup 0.2845 \\
unseen 2-way & 3 & \textbf{declared negative, nothing claimed}: store 0.4097 (0.214 / 0.456 / 0.559) against transformer 0.3611, but the learned BLIND-SEQUENTIAL arm reads 0.5567; across-wells control 0.2596 against a bar of 0.2597. Failing link: \emph{interface} (a held-out pair cell is untrained) \\
unseen $n$-way & 3 & \textbf{STORE row}: store 0.7246 (0.800 / 0.566 / 0.808) against the best matched rival, transformer 0.4035 (0.394 / 0.395 / 0.421); MLP 0.2194, LSTM 0.1413; ahead on every seed, margin 0.321. Ceiling set by the request design, not the store: validation $n$-way 0.7247 with the same per-seed triple across about 15 configurations \\
\quad replication & 3 & Config C-512 (latent 512, 8 atoms; runs 7--9), rivals trained 80,000 steps: store 0.7251 (0.800 / 0.567 / 0.808) against transformer 0.4533 (0.468 / 0.430 / 0.462); caveat: on seed 1 random atoms reads 0.464 against the store's 0.567 (0.103 $<$ 0.15) \\
\quad budget diagnostic & 3 & validation, A1, every rival trained 80,000 steps, matched and at 10$\times$ read floats: all below the store's 0.802 / 0.569 / 0.803 on every seed (closest: seed 1, LSTM 0.437) \\
rollout validity & 3 & \textbf{declared negative}: 0.0000 for the store and every matched rival on every seed (45 rollouts per seed); per-step exactness store 0.9254 (0.930 / 0.917 / 0.928) against transformer 0.8180, compounding over 43 steps. Qualifier: the \emph{diagnostic} transformer at 10$\times$ read floats, not a matched rival, reads 0.0074 (0.000 / 0.000 / 0.022). Failing link: \emph{read} \\
value hold-out & 3 & \textbf{declared negative, nothing claimed}: store 0.4647 below transformer 0.5107 and learned BLIND-SEQUENTIAL 0.4848. Failing link: \emph{content} (a never-trained colour has no trained cells) \\
controls, floors & 3 & all four charter-C controls at most store $-$0.15 on every seed (bar 0.575): landscape removed 0.0000, shuffled 0.2248, shuffled across wells 0.2435, random atoms 0.1655. Learned BLIND-SEQUENTIAL 0.3418, BLIND-COLOUR 0.0716 (bar 0.675). Floors: cross-maze word lookup 0.0000, cross-maze raw lookup 0.0019, cross-maze fragment 0.0319, within-maze lookup 0.1030 (not transfer). Planner columns, rule-knowing relaxed oracles and never rivals: BLIND-SEQUENTIAL 0.9071, BLIND-COLOUR 0.5509, BLIND-FACTOR 0.6047 \\
deletion, value & 3 & colour 1; store-scoped exactness $S_k$ relative change \sci{7.80}{-16} / \sci{1.22}{-15} / \sci{1.09}{-15}, atoms change 0; downdated read equal to fitted-without: 0.4369 / 0.4253 / 0.6827, against rival retrain-without 0.3153 / 0.2329 / 0.1076 (as trained 0.2432 / 0.0000 / 0.0340); store adaptation 0.130 / 0.069 / 0.082 s, light and data-free, against rival retrain 14.3 / 14.1 / 14.9 s and 8,000 steps \\
deletion, conj. & 3 & $S_k$ relative change \sci{7.73}{-16} / \sci{1.04}{-15} / \sci{8.45}{-16}, atoms change 0; downdated read equal to fitted-without: 0.4369 / 0.4304 / 0.6827, against rival retrain-without 0.2252 / 0.2633 / 0.1020; store adaptation 0.033 / 0.043 / 0.038 s against rival retrain 13.8 / 13.8 / 14.7 s \\
ledger & 3 & 4,584 read floats (768 atom + 3,816 head); 18,336 B deployed (float32); 34,320 B statistic (float64); rank/width 0.094 per well; identical per seed \\
% ENDSRC
\bottomrule
\end{tabular}\end{adjustbox}
\end{table}"""

CMAZEO = r"""\begin{table}[h]
\centering\scriptsize
\caption{\textbf{Maze world-model rows: CMAZE-O (observation generation), TEST, Config C-field, 3 % SRC: w16-maze-wm/TABLES.md
seeds (scored runs 4--6). The rival margin is WITHDRAWN BY THE HUB; the store gate holds.} Generate
the grid state for a request of colours, object cells and layout; exact placement and legality.
Store and head are fitted in the observation field ($K=8$, 64 atoms) and read out by a parameterless % SRC: w16-maze-wm/TABLES.md
decode, with no autoencoder in the read path; every rival regresses the identical field at matched
read floats. The rival rows here use the recipe's 3,000 training steps, the budget whose margin is % SRC: w16-maze-wm/TABLES.md
withdrawn. Byte convention as Table~\ref{tab:cmazet}. Values are mean (seed 0 / 1 / 2). This table
contains no CMAZE-T, CMAZE-1, CMAZE-2 or CMAZE-2b row.}
\label{tab:cmazeo}
\setlength{\tabcolsep}{2pt}
\begin{adjustbox}{max width=\linewidth}\begin{tabular}{p{1.9cm}p{1.2cm}p{10.0cm}}
\toprule
row & seeds & CMAZE-O \\
\midrule
% SRCBLOCK: w16-maze-wm/TABLES.md; w16-maze-wm/FINDINGS.md; w16-maze-wm/MAZE-WM.json
seen & 3 & store 0.7889 (0.787 / 0.767 / 0.813) against recipe-budget MLP 0.3378; lookup 0.0650 \\
unseen 2-way & 3 & store 0.5210 (0.755 / 0.386 / 0.422) against recipe-budget MLP 0.0993; across-wells and random-atoms controls 0.0000; lookup 0.0008. The rival comparison uses the withdrawn recipe budget \\
unseen $n$-way & 3 & \textbf{composition over rivals: WITHDRAWN BY THE HUB.} By the pre-registered rule a win, store 0.6933 (0.637 / 0.730 / 0.713) against recipe-budget MLP 0.1667; but the same matched MLP trained 30,000 steps reads 0.988 / 0.990 / 0.990 on validation $n$-way against the store's 0.640 / 0.730 / 0.730, so the store loses on every seed and runs 4--6 are attached to a claim that does not stand. Failing link: \emph{interface} (the field output lifts a converged MLP further than the store) \\
rollout validity & --- & (no rollouts) \\
value hold-out & 3 & 0.0000 for every arm, the store included (store 0.0000; pipeline ceiling 1.0000): a judge-integrity row, not a claim either way \\
controls, floors & 3 & \textbf{the store gate holds}: all four charter-C controls 0.0000 on every seed (landscape removed, shuffled channels, shuffled across wells, random atoms); the store is load-bearing relative to its controls. The atom-row \texttt{landscape\_shuffled} equals the store (0.6933) because it is a proved identity for a re-solved signed-span read: printed, never evidence. Lookup 0.0000; best BLIND column 0.0600; pipeline ceiling 1.0000; autoencoder-latent ceiling, for reference only, 0.4550 \\
deletion, value & 3 & colour 2; 5530 / 5250 / 5959 rows deleted; relative change of the statistic 0.0 on every seed; after-deletion equal to refit-without in every cell. Deleted colour 0.6020 / 0.7483 / 0.7185 $\to$ 0.0000; other colours 0.6734 $\to$ 0.6231, 0.6792 $\to$ 0.7642, 0.6780 $\to$ 0.6610; store adaptation 2.92 / 3.07 / 3.28 s against recipe-budget MLP retrain-without 28.3 / 27.7 / 28.0 s \\
deletion, conj. & 3 & 2240 / 1435 / 2168 rows deleted; relative change 0.0; deleted side 0.5800 / 0.8000 / 0.7067 $\to$ 0.0000; other colours 0.6457 $\to$ 0.6343, 0.6880 $\to$ 0.6800, 0.7160 $\to$ 0.7680; adaptation 1.92 / 1.83 / 2.03 s against 29.3 / 28.9 / 29.0 s. Store-scoped exactness on an instrument where the rival claim does not stand \\
ledger & 3 & 1,232,064 read floats (158,400 atom + 1,073,664 head); 4,928,256 B deployed (float32); 3,092,584 B statistic (float64); rank/width 0.174 to 0.182; rivals within 2\% (MLP 1231290, continuous MLP 1233449, LSTM 1231661, transformer 1217195) \\
% ENDSRC
\bottomrule
\end{tabular}\end{adjustbox}
\end{table}"""

# P28 / P29 / P30 after the two tables
AFTER_TABLES = r"""% SRCBLOCK: w16-maze-wm/FINDINGS.md; w16-maze-wm/TABLES.md
\paragraph{Configuration ladder and budget.} Every configuration was measured on validation first,
and three scored seeds were spent only after its validation bars passed. \emph{CMAZE-T}: a sweep of
53 configurations on three seeds left 25/53 store-gate eligible, and the crossover rule picked
$\Kstar=2$, Config A1 (latent 128, 6 atoms, rank/width 0.094, 4,584 read floats), scored in runs
1--3; Config C-512 (latent 512, 8 atoms, rank/width 0.031, 9,184 read floats), scored in runs 7--9
against 80,000-step rivals, is its replication. Config B, a match-bit interface given identically to
every arm, failed validation: the store rose from 0.7247 to 0.9258 ($+0.20$) and the best matched
rival from 0.3891 to 0.7902 ($+0.40$), and the learned BLIND-SEQUENTIAL arm read 0.8782 against a bar
of 0.8758. The pre-amendment Config A was superseded by the control defect of the harness paragraph
above. \emph{CMAZE-O}: Config A (autoencoder latent, rank/width at most 0.3) failed validation, its
best unseen $n$-way 0.0208; Config C-field ($K=8$, 64 atoms, 1,232,064 read floats) was scored in
runs 4--6; Config C2 (latent, rank/width 0.5) failed validation against 30,000-step rivals, the store
0.1367 (0.087 / 0.150 / 0.172) against MLP 0.3967 (0.315 / 0.412 / 0.463). \emph{Budget}: 9 of 18
scored runs spent, 9 declared not-run. Runs 10--12 (CMAZE-O) are not run because no CMAZE-O
configuration passes validation against converged rivals; a field configuration would need $n$-way
of at least 1.02 to clear the 0.99 MLP, and seeding a configuration that fails validation is
forbidden. A ten-times-steps re-scoring of the C-field TEST leg, deletion rows for Configs B and C2
(validation only, not winners) and any seed beyond three are also not run.
% ENDSRC
\paragraph{The fallback route task.} Declared not-run (runs 13--18): the route task on CMAZE-2b % SRC: w16-maze-wm/FINDINGS.md
with the store against learned rivals and the merge as the shared interface was opened only if
neither CMAZE-T nor CMAZE-O yielded a win, and CMAZE-T yielded a store row that survives the
training-budget objection. Tables~\ref{tab:verb} and~\ref{tab:maze-del-2b} are unaffected.
\paragraph{Declared negatives, each with its failing link.} CMAZE-T: unseen 2-way
(\emph{interface}); value hold-out (\emph{content}); rollout validity (\emph{read}); Config B
(\emph{interface}). CMAZE-O: Config A (\emph{content/read}: the autoencoder latent is not carried by
a low-rank store); C-field against converged rivals (\emph{interface}); Config C2 (\emph{read}: the
latent pipeline caps the store). CMAZE-O therefore has no single failing link, and its C-field
store gate holds while its composition-over-rivals claim is withdrawn."""

# sec:maze-wm main text (replaces the whole subsection body, both drafts)
MAZEWM_MAIN = r"""\subsection{World-model rows: transition and observation generation}
\label{sec:maze-wm}

The two instruments below are the maze counterpart of \S\ref{sec:shapes}: they ask the store to
\emph{generate} a world model's transitions (CMAZE-T) and states (CMAZE-O) under the exact judge,
on random mazes never seen at test, so that the within-maze lookup floor cannot transfer and the
cross-maze fragment lookup is the floor. Rivals are matched on read floats; the pre-declared
landscape controls and blind columns are printed; bars were pre-registered before any number; every
row has three seeds; and the two instruments never share a table (Tables~\ref{tab:cmazet}
and~\ref{tab:cmazeo}, Appendix~\ref{app:mazewm}).

% SRCBLOCK: w16-maze-wm/FINDINGS.md; w16-maze-wm/TABLES.md
\paragraph{CMAZE-T: a store row on unseen $n$-way transitions.} One pinned design (Config A1:
latent 128, $K=2$, 6 atoms, rank/width 0.094, 4,584 read floats), three TEST seeds, scored runs
1--3: the store generates held-out $n$-way transitions at 0.7246 (0.800 / 0.566 / 0.808) against
0.4035 (0.394 / 0.395 / 0.421) for the best matched rival, a transformer, and is ahead on every
seed. All four charter-C controls collapse below the bar of 0.575 on every seed (landscape removed
0.0000, shuffled 0.2248, shuffled across wells 0.2435, random atoms 0.1655), the learned
composition-blind arms fail (0.3418 and 0.0716 against a bar of 0.675), and the cross-maze lookup
floor reads 0.0000. The row survives rivals trained ten times longer: on validation, every rival
trained for 80,000 steps, matched or at ten times the read floats, stays below the store on every
seed (closest: seed 1, LSTM 0.437 against 0.569), and a scored replication at a wider latent (Config
C-512, runs 7--9) against 80,000-step rivals reads 0.7251 against 0.4533. A caveat travels with the
replication: on its seed 1, random atoms reach 0.464 against the store's 0.567, a margin of 0.103,
below the 0.15 bar. The row also has a ceiling set by the request design, not by the store:
validation $n$-way is 0.7247 with the same per-seed triple across about 15 configurations, and
seed 1 caps at 0.569 in all of them.
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

% SRCBLOCK: w16-maze-wm/FINDINGS.md; w16-maze-wm/MAZE-WM.json
\paragraph{CMAZE-O: the store gate holds; the rival margin is withdrawn.} With the store fitted in
the observation field (Config C-field: $K=8$, 64 atoms, 1,232,064 read floats; runs 4--6), the store
composes held-out $n$-way observations at 0.6933 against 0.1667 for the best rival trained on the
recipe's budget, a win by the pre-registered rule. \textbf{That rival margin is withdrawn by the
Hub.} The same matched MLP trained ten times longer (30,000 steps) reads 0.988 / 0.990 / 0.990 on
validation $n$-way against the store's 0.640 / 0.730 / 0.730, so no CMAZE-O configuration can pass
against a converged matched MLP, and runs 4--6 are attached to a composition-over-rivals claim that
does not stand. Two statements must be kept apart. What is withdrawn is \emph{composition over
rivals}. What holds is the \emph{store gate}: all four charter-C controls read 0.000 on every seed,
so the store is load-bearing relative to its own controls. The failing link of the rival claim is
the \emph{interface}: the field output space lifts a plain MLP further than it lifts the store. The
other two CMAZE-O configurations fail validation with their own links (the autoencoder latent at
rank/width at most 0.3, \emph{content/read}; a latent at rank/width 0.5, \emph{read}), so CMAZE-O has
no single failing link. Exact deletion also runs on C-field (relative change of the statistic 0.0 on
every seed; after-deletion equal to refit-without in every cell), on an instrument where the rival
claim does not stand.
% ENDSRC

The maze section therefore stands on deletion transfer (CMAZE-2 and CMAZE-2b), the verb as an
interface result (CMAZE-2b), the CMAZE-T store row with its declared negatives, and the CMAZE-O store
gate with its withdrawn margin. The budget, the configuration ladder, the fallback and every negative
with its link are in Appendix~\ref{app:mazewm}.

"""

PLACEHOLDER_INDEX = r"""\section{Placeholder index}
\label{app:placeholders}

Four rows remain pending, each marked in the text by a red \textsc{placeholder} whose argument names
the row and states that it may land as a negative; they belong to runs that had not landed when the
other rows were written in: LACE and PoE rows and the generation-table run (\S\ref{sec:adapt},
Table~\ref{tab:t7}); Criterion D and its decomposition paragraph (Appendices~\ref{app:encoder-main}
and~\ref{app:encloop}). One filled row carries an orange \textsc{todo}: the coverage control
(\S\ref{sec:ladder}), whose verdict is written in but whose artefact was not available to the
drafting machine. Figure slots are marked as placeholders where they depend on pending rows.

"""

HEADER_OLD = r"""%   * NO WAVE-15 OR WAVE-16 NUMBER APPEARS ANYWHERE IN THIS FILE (Head ruling,
%     AMENDMENT-1, 2026-09-22). Every w15/w16 row is a \PLACEHOLDER{...} whose argument
%     names the row AND states that it may land as a negative. `grep -n PLACEHOLDER` finds
%     every one; FINDINGS.md (w17-draft-v2) lists them all."""
HEADER_NEW = r"""%   * w18 placeholder-fill (2026-09-23): the landed w15/w16 rows are written in, each with
%     its SRC comment. Four PLACEHOLDERs remain (bar-d-score x2, lace-poe-rows x2: AWAITING,
%     second pass); coverage-control carries a TODO (artefact not staged on batmac).
%     `grep -n PLACEHOLDER` finds every one; w18-placeholder-fill/FINDINGS.md lists them."""
