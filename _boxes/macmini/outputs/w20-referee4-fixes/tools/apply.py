#!/usr/bin/env python3
"""w20 referee4-fixes: apply MF-1..MF-8, MF-10, SF-7 and the M7(e) regression to BOTH drafts.
usage: apply.py SRC_DIR OUT_DIR   (reads DRAFT-v2.tex / DRAFT-v2-style.tex, writes the edited pair)
Every edit is (tag, old_v2, new_v2, old_style, new_style, count); None for the style side means
'same as v2'. Each old string must occur exactly `count` times in its file, or the script aborts."""
import sys, os

SRC, OUT = sys.argv[1], sys.argv[2]
V2, ST = 'DRAFT-v2.tex', 'DRAFT-v2-style.tex'
t = {V2: open(os.path.join(SRC, V2)).read(), ST: open(os.path.join(SRC, ST)).read()}
log = []

def E(tag, old, new, old_s=None, new_s=None, count=1):
    for f, o, n in ((V2, old, new), (ST, old if old_s is None else old_s, new if new_s is None else new_s)):
        c = t[f].count(o)
        if c != count:
            sys.exit('ABORT %s: %s occurs %d times in %s (want %d)\n---\n%s' % (tag, repr(o[:80]), c, f, count, o))
        t[f] = t[f].replace(o, n)
    log.append(tag)

# ============================ MF-10 / MF-8 / MF-3: abstract, Shapes sentence ==================
ABS_NEW = r'''On a Shapes3D value hold-out at $\Kstar=32$ wells, the finest partition on a validation plateau
($K=5\ldots32$) that three seeds do not resolve, % SRC: w15-crossover-seeds/CROSSOVER-3SEED.md
the store composes held-out pairs at 32.350~$\pm$~0.815 and held-out six-factor corners at
67.177~$\pm$~1.320 distinct valid novel tuples per corner over three seeds, with all four landscape
controls collapsing to exactly 0.0000. % SRC: w12b-nr6-and-kstar/NR6-T7.md
On its 109 held-out corners the store, fitted on 131,072 rows, reads 47.4954 / 47.4037 / 49.1193 % SRC: w12b-nr6-and-kstar/NR6-T7.md; w13-seeds-and-tables/TABLES.md
against 30.0275 / 28.9174 / 37.5138 for the strongest read-path-matched rival of either of our two % SRC: w13-seeds-and-tables/TABLES.md
rival sets, whose 32,256 fit rows carry held-out values the store never saw; on every rung of an % SRC: _thinkpad/w4/spoke-a-rivals/rivals/seed_0/mlp.rival.json
$n$-way ladder it scores above a read-path-matched MLP fitted on the store's own 32,768 fit rows on % SRC: w12-composition-ladder/runs/shipped_seed0/rows_learned_ref_ownpool.json; w12b-nr6-and-kstar/kstar_ladder/2_seed0_K32/LEG.json
every seed, and beats an MLP matched to its whole four-part ledger on those rows by at least
$+3.117$ tuples per corner. % SRC: w12b-nr6-and-kstar/FINDINGS.md
'''
E('MF-10/MF-8/MF-3 abstract (Shapes)',
  r'''On a Shapes3D value hold-out with the partition selected on
validation ($\Kstar=32$ wells), the store composes held-out pairs at 32.350~$\pm$~0.815 and
held-out six-factor corners at 67.177~$\pm$~1.320 distinct valid novel tuples per corner over
three seeds, beating every read-path-matched learned rival on 109 of 109 corners on every
seed, with all four landscape controls collapsing to exactly 0.0000. % SRC: w12b-nr6-and-kstar/NR6-T7.md
''', ABS_NEW,
  r'''On a Shapes3D value hold-out, with the partition selected on validation ($\Kstar=32$ wells),
the store composes held-out pairs at 32.350~$\pm$~0.815 and held-out six-factor corners at
67.177~$\pm$~1.320 distinct valid novel tuples per corner over three seeds. It beats every
read-path-matched learned rival on 109 of 109 corners on every seed, and all four landscape
controls collapse to exactly 0.0000. % SRC: w12b-nr6-and-kstar/NR6-T7.md
''')

# ============================ MF-1: abstract, CMAZE-T sentence ================================
E('MF-1 abstract (CMAZE-T)',
  r'''On a transition instrument over the same mazes the store does compose held-out $n$-way
transitions, 0.7246 against 0.4035 for the best matched rival over three seeds with every % SRC: w16-maze-wm/FINDINGS.md
landscape control collapsing, although no matched arm, the store included, yields a valid
chained rollout.
''',
  r'''On a transition instrument over the same mazes the store reads 0.7246 on held-out $n$-way
transitions against 0.4035 for the best matched rival over three seeds, with every landscape % SRC: w16-maze-wm/FINDINGS.md
control collapsing; but the store reads a word request with a hand-built pair block that the
rivals, which read the observation, do not receive, so the row is scoped to that request until a
symmetric-input test is run, and no matched arm, the store included, yields a valid chained
rollout.
''')

# ============================ MF-8 / MF-3: contribution 1 =====================================
C1 = r'''{Composition at a stated partition, against read-path-matched rivals.} On a
  Shapes3D~\citep{burgess2018shapes,kim2018disentangling} value hold-out at $\Kstar=32$ wells, three scored seeds, with every % SRC: w12b-nr6-and-kstar/NR6-T7.md
  pre-declared landscape control collapsing, the store scores above the strongest read-path-matched
  learned rival of either of the paper's two rival sets on every seed, although that rival's fit rows
  carry held-out values the store never saw (\S\ref{sec:shapes-t7}). On every rung of an $n$-way
  ladder it also scores above a read-path-matched MLP fitted on the store's own 32,768 fit rows, and % SRC: w12b-nr6-and-kstar/kstar_ladder/2_seed0_K32/LEG.json
  beats an MLP matched to its whole ledger on the same rows by at least $+3.117$ per seed % SRC: w12b-nr6-and-kstar/FINDINGS.md
  (\S\ref{sec:ladder}); for $n\ge3$ the rungs are a training-data axis and not a depth axis.
'''
E('MF-8/MF-3 contribution 1',
  r'''\item \textbf{Composition at a stated partition, against read-path-matched rivals.} On a
  Shapes3D~\citep{burgess2018shapes,kim2018disentangling} value hold-out at $\Kstar=32$ wells, three scored seeds, the store beats every % SRC: w12b-nr6-and-kstar/NR6-T7.md
  learned rival matched on read-path parameters on both composition rows and on 109 of 109 % SRC: w12b-nr6-and-kstar/NR6-T7.md
  corners per seed, with every pre-declared landscape control collapsing
  (\S\ref{sec:shapes-t7}). The same holds on every rung of an $n$-way ladder
  (\S\ref{sec:ladder}), read with the inference note that, for $n\ge3$, the rungs are a
  training-data axis and not a depth axis.
''', r'\item \textbf' + C1,
  r'''\item \emph{Composition at a stated partition, against read-path-matched rivals.} On a
  Shapes3D~\citep{burgess2018shapes,kim2018disentangling} value hold-out at $\Kstar=32$ wells and three scored seeds, the store beats every % SRC: w12b-nr6-and-kstar/NR6-T7.md
  learned rival matched on read-path parameters, on both composition rows and on 109 of 109 % SRC: w12b-nr6-and-kstar/NR6-T7.md
  corners per seed, while every pre-declared landscape control collapses
  (\S\ref{sec:shapes-t7}). The same holds on every rung of an $n$-way ladder
  (\S\ref{sec:ladder}), read with the inference note that, for $n\ge3$, the rungs form a
  training-data axis and not a depth axis.
''', r'\item \emph' + C1)

# ============================ SF-7 / MF-1: contribution 3 =====================================
E('SF-7/MF-1 contribution 3',
  r'''carries a composition row with every landscape control collapsing, and it survives rivals
  trained ten times longer; no matched arm's chained rollout is valid;''',
  r'''carries a composition row with every landscape control collapsing, under a word request with a
  pair block that its rivals do not receive (a symmetric-input test is pending); on validation it
  stays ahead of rivals trained ten times longer, and its scored replication against such rivals
  keeps the margin while one landscape control misses its collapse bar on one seed; no matched
  arm's chained rollout is valid;''')

# ============================ MF-5(b): thickness ==============================================
E('MF-5(b) thickness',
  r'''The one environment with an exact route judge and three scored seeds returns a negative for
the store and a positive for the interface. A transition instrument on the same mazes adds
one store row at three seeds (\S\ref{sec:maze-wm}).''',
  r'''On routes, where the maze's exact judge scores whole plans at three scored seeds, the result is
a negative for the store and a positive for the interface. A transition instrument on the same
engine, also with an exact judge and three scored seeds, adds one store row, scoped to a request
with a pair block that its rivals do not receive (\S\ref{sec:maze-wm}).''')

# ============================ MF-2: the w20 placeholder's fit pools ===========================
E('MF-2 placeholder pools',
  r'''% SRCBLOCK: w12-deletion-baselines/FINDINGS.md; _thinkpad/w4/spoke-a-rivals/rivals/seed_0/mlp.rival.json; w13-seeds-and-tables/TABLES.md; ../../tasks/w20/shapes-data-matched.md
\PLACEHOLDER{w20 \texttt{shapes-data-matched}: the one data-matched rival set that replaces both
rival sets above. The two sets it replaces, and their fit pools: the headline rivals of
Table~\ref{tab:t7}, fit on 16,384 rows (the deletion study's declared build/smoke scale, reused as
the headline comparators), and the ladder's rivals of Table~\ref{tab:ladder}, fit on 32,256 rows of
the combination-split pool; the store's own fit pool is 131,072 rows. Pending: the four naive units
refit on the store's exact 131,072-row pool, and the store refit on the rivals' 16,384- and
32,256-row pools, rows 3--4 and every ladder rung, 3 seeds, each rival with its convergence
evidence; which of the two tables leads is not yet decided. May land as a NEGATIVE: against
data-matched rivals the store's margin may shrink, vanish or reverse on some or all corners}''',
  r'''% SRCBLOCK: w12-deletion-baselines/FINDINGS.md; _thinkpad/w4/spoke-a-rivals/rivals/seed_0/mlp.rival.json; w13-seeds-and-tables/TABLES.md; ../../tasks/w20/shapes-data-matched.md; w12b-nr6-and-kstar/kstar_ladder/2_seed0_K32/LEG.json
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
against data-matched rivals the store's margin may shrink, vanish or reverse on some or all corners}''')

# ============================ MF-3: the data-matched rival, printed in 3.1 ====================
DM = r'''\paragraph{A rival matched on both the read path and the data.} One learned rival is already
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

'''
E('MF-3 data-matched paragraph (3.1)',
  '\\begin{table}[t]\n\\centering\\small\n\\caption{\\textbf{The four-part ledger of the $\\Kstar=32$ object',
  DM + '\\begin{table}[t]\n\\centering\\small\n\\caption{\\textbf{The four-part ledger of the $\\Kstar=32$ object')

# ============================ MF-3 / MF-7: tab:ladder =========================================
E('MF-3 tab:ladder caption',
  r'''the same corners; a rival does not depend on $K$. $n=2$: 60 corners; $n\ge3$: the identical % SRC: w12-composition-ladder/FINDINGS.md''',
  r'''the same corners; a rival does not depend on $K$. The data-matched MLP (130,514 parameters) is % SRC: w12-composition-ladder/LADDER.json
fitted on the same 32,768 rows as the store's content (\S\ref{sec:shapes-t7}); LACE and PoE are in % SRC: w12b-nr6-and-kstar/kstar_ladder/2_seed0_K32/LEG.json
Table~\ref{tab:lace-poe}. $n=2$: 60 corners; $n\ge3$: the identical % SRC: w12-composition-ladder/FINDINGS.md''')
E('MF-3 tab:ladder data-matched row',
  r'''table (floor) & 3 & 5.8333 / 3.4833 / 6.1833 & \multicolumn{4}{c}{8.4333 / 7.9167 / 12.2333 (identical)} \\
% ENDSRC''',
  r'''table (floor) & 3 & 5.8333 / 3.4833 / 6.1833 & \multicolumn{4}{c}{8.4333 / 7.9167 / 12.2333 (identical)} \\
MLP, data-matched (store's own fit rows) & 3 & 5.5667 / 7.5833 / 9.3000 & 8.2167 / 10.9083 / 16.2583 & 8.5417 / 12.5333 / 18.7667 & 9.8667 / 15.3083 / 19.7083 & 10.6667 / 15.1167 / 21.0333 \\
% ENDSRC''')
LADDER_LP = r'''% SRCBLOCK: w15-lace-poe-rows/FINDINGS.md
LACE (own budget, not matched) & 3 & 0.067 / 0.050 / 0.067 & 0.133 / 0.125 / 0.208 & 0.117 / 0.217 / 0.183 & 0.083 / 0.142 / 0.183 & 0.125 / 0.100 / 0.175 \\
PoE (own budget, not matched) & 3 & 0.017 / 0.000 / 0.000 & 0.000 / 0.000 / 0.000 & 0.008 / 0.000 / 0.008 & 0.000 / 0.008 / 0.000 & 0.000 / 0.008 / 0.000 \\
% ENDSRC
'''
E('MF-7 LACE/PoE out of tab:ladder', LADDER_LP, '')

# ============================ MF-7: LACE / PoE out of tab:t7 ==================================
T7_LP = r'''\midrule
% SRCBLOCK: w15-lace-poe-rows/FINDINGS.md
LACE (own budget, not matched) & 3 & 0.117 / 0.033 / 0.267 & 0.224 / 0.122 / 0.184 & rival $-$ store: $-47.33$ / $-47.33$ / $-48.89$ \\
PoE (own budget, not matched) & 3 & 0 / 0 / 0 & 0.020 / 0 / 0 & rival $-$ store: $-47.49$ / $-47.40$ / $-49.12$ \\
\multicolumn{5}{p{15cm}}{\emph{LACE and PoE are added beside the rivals above, which are not replaced; each runs at its own declared budget (LACE 31.08$\times$, PoE 27.13$\times$ the 524,288 B reference, a parameter count against store-mode bytes), not on the read-path matching rule. Their $\Delta$ is printed as rival minus store, the artefact's sign; the store wins 109/109 corners against each, on every seed. Scope: \S\ref{sec:adapt}.}} \\
% ENDSRC
'''
E('MF-7 LACE/PoE out of tab:t7', T7_LP, '')

# ============================ MF-4 / MF-5(d): 3.7 encoder, heading, LACE, generation table ====
E('MF-4 3.7 heading',
  r'\subsection{The encoder: frozen by design, lightly adapted, and not shaped by the store}',
  r'\subsection{The encoder: frozen by design, lightly adapted; the encoder loop licenses no store claim}')
ENC_NEW = r'''Training the encoder through the store licenses no store claim. Pre-registered criterion D, whether
the store's margin over a matched learner grows, is mixed on the longer leg, with no claim in either
direction, and unmeasured on the shortened leg, whose learner rows are a rival fitted on the frozen
encoder's features; the swap loss helps on both legs, and ``unfreezing hurts'' does not survive the
longer leg (Appendices~\ref{app:encoder-main} and~\ref{app:encloop}).
'''
E('MF-4/MF-5(d) 3.7 encoder paragraph',
  r'''Training the encoder through the store is a wash of two real effects, unfreezing hurts and a swap
loss repairs, and the store's margin over a matched learner does not grow, so the store does not
shape the representation. Both are in Appendix~\ref{app:encoder-main}, with the pending scored row of pre-registered
criterion D on the longer leg.
''', ENC_NEW,
  r'''Training the encoder through the store, in contrast, is a wash of two real effects: unfreezing
hurts and a swap loss repairs. The store's margin over a matched learner does not grow, so the
store does not shape the representation. Appendix~\ref{app:encoder-main} reports both results,
with the pending scored row of pre-registered criterion D on the longer leg.
''')

LACE_PARA_OLD = r'''\paragraph{LACE and PoE, at their own budgets.}
LACE~\citep{nie2021controllable} and PoE~\citep{hinton2002training} were \emph{added} beside the
learned rivals on rows 3--4 (Table~\ref{tab:t7}) and on every ladder rung
(Table~\ref{tab:ladder}), 3 seeds; no learned-rival column was replaced. They do not beat the
store on rows 3--4, nor anywhere on the ladder: on rows 3--4 the $\Kstar$ store scores 47.50 / % SRC: w15-lace-poe-rows/FINDINGS.md
'''
LACE_PARA_MOVED = r'''\paragraph{LACE and PoE in full.}
LACE~\citep{nie2021controllable} and PoE~\citep{hinton2002training} were scored beside the
learned rivals on rows 3--4 and on every ladder rung (Table~\ref{tab:lace-poe}), 3 seeds; no
learned-rival column was replaced. They do not beat the
store on rows 3--4, nor anywhere on the ladder: on rows 3--4 the $\Kstar$ store scores 47.50 / % SRC: w15-lace-poe-rows/FINDINGS.md
'''
LACE_REST = r'''47.40 / 49.12 novel valid tuples per corner against LACE 0.165 / 0.073 / 0.229 and PoE 0.009 / % SRC: w15-lace-poe-rows/FINDINGS.md
0.000 / 0.000, and the store wins 109/109 corners against each, on every seed; on every rung it % SRC: w15-lace-poe-rows/FINDINGS.md
wins every corner against both on every seed (60 of 60 at $n=2$, 120 of 120 at $n\ge3$). % SRC: w15-lace-poe-rows/FINDINGS.md
Their budgets are their own and are not byte-matched: LACE 4,073,663 classifier parameters % SRC: w15-lace-poe-rows/FINDINGS.md
(31.08$\times$ the 524,288 B reference; 63.10$\times$ counting its whitening and mean), PoE six % SRC: w15-lace-poe-rows/FINDINGS.md
experts with 3,556,665 parameters (27.13$\times$), each fitted on 32,256 rows with 512 validation % SRC: w15-lace-poe-rows/FINDINGS.md
rows from the rung's own pool and 36 selection trials. % SRC: w15-lace-poe-rows/FINDINGS.md
\emph{This is LACE and PoE as instantiated in this feature space at their own declared budgets,
and it must be weighed that way; it is not a refutation of LACE in general.} Both render zero
exact-match requests on the generation table as well; in the words of that experiment's own record, a
reader should weigh the result as ``LACE/PoE, as re-implemented at these budgets, do not produce
valid full tuples here'', not as ``the CLU out-composes LACE in general''. % SRC: w15-lace-poe-rows/FINDINGS.md
'''
LACE_MAIN = r'''\paragraph{LACE and PoE, at their own budgets.}
LACE~\citep{nie2021controllable} and PoE~\citep{hinton2002training}, re-implemented at their own
budgets in this feature space, score at most 0.267 and 0.020 distinct valid novel tuples per corner % SRC: w15-lace-poe-rows/FINDINGS.md
on rows 3--4, render no exact-match request on the generation table on any seed, and there score at
the level of our landscape controls; we report them for completeness, not as evidence, in
Table~\ref{tab:lace-poe} (Appendix~\ref{app:t7more}). \emph{This is LACE and PoE as instantiated in
this feature space at their own declared budgets; it is not a refutation of LACE in general}, and
the study has no functioning generative baseline (\S\ref{sec:limits-list}).
'''
E('MF-7 LACE paragraph -> main-text sentence', LACE_PARA_OLD + LACE_REST, LACE_MAIN)

GEN_OLD = r'''\paragraph{The generation table, scored run 3 of 3, on the $\Kstar$ single design.}
The $\Kstar$ single design was added to the arms of the generation table and scored once
(\texttt{generation-table} run 3 of 3, 3 seeds, nothing not-run; Table~\ref{tab:gentable-kstar}, % SRC: w15-lace-poe-rows/FINDINGS.md
Appendix~\ref{app:t7more}). All four landscape controls collapse on every seed (pre-registered criterion S1, % SRC: w15-lace-poe-rows/FINDINGS.md
\textsc{pass}), so the store claim stands on this table. Pre-registered criterion S2, against the shipped
read (every seed, margin 0.020), \textsc{fails} on seed 0 ($-0.0185$; mean $+0.1551$), and the % SRC: w15-lace-poe-rows/FINDINGS.md
pre-registration's falsification clause fires: the claim that $\Kstar$ beats the shipped read is
\emph{not} made. $\Kstar$ also loses to the prototype ceiling \texttt{knn\_proto} on the mean % SRC: w15-lace-poe-rows/FINDINGS.md
'''
GEN_NEW = r'''\paragraph{The generation table, on the $\Kstar$ single design.}
The generation table is a second Shapes3D instrument, never merged with Table~\ref{tab:t7}: 72 % SRC: w15-lace-poe-rows/FINDINGS.md
held-out requests per seed, scored as an aggregate under the external classifier. The $\Kstar$ single design was added
to its arms and scored once, in one of that table's three pre-registered evaluation runs (3 seeds,
nothing not-run; Table~\ref{tab:gentable-kstar}, % SRC: w15-lace-poe-rows/FINDINGS.md
Appendix~\ref{app:t7more}). All four landscape controls collapse on every seed (pre-registered criterion S1, % SRC: w15-lace-poe-rows/FINDINGS.md
\textsc{pass}), so the store claim stands on this table. Pre-registered criterion S2, against the earlier
store's read (every seed, margin 0.020), \textsc{fails} on seed 0 ($-0.0185$; mean $+0.1551$), and the % SRC: w15-lace-poe-rows/FINDINGS.md
pre-registration's falsification clause fires: the claim that $\Kstar$ beats that read is
\emph{not} made. $\Kstar$ also loses to a nearest-prototype ceiling on the mean % SRC: w15-lace-poe-rows/FINDINGS.md
'''
E('M7(e) generation-table paragraph', GEN_OLD, GEN_NEW)

# ============================ MF-5(e): 4.1 =====================================================
E('MF-5(e) 4.1 only deletion leg in scored runs',
  'It is the one\ndeletion leg in the study that sits inside scored runs,',
  'It was the\nfirst deletion leg in the study to sit inside scored runs (the deletion rows of the two\nworld-model instruments of \\S\\ref{sec:maze-wm} also do),',
  'It is the\nonly deletion leg in the study that sits inside scored runs,')

# ============================ MF-1 / M7(e): 4.4 ================================================
E('MF-1 4.4 CMAZE-T heading',
  r'\paragraph{CMAZE-T: a store row on unseen $n$-way transitions.}',
  r'\paragraph{CMAZE-T: a store row on unseen $n$-way transitions, under a pair-design request.}')
E('M7(e) 4.4 runs 1--3',
  'three TEST seeds, scored runs\n1--3: the store generates',
  'three TEST seeds, one pre-registered scored run each: the store generates')
E('M7(e) 4.4 runs 7--9', '(Config\nC-512, runs 7--9) against', '(Config\nC-512, three further scored runs) against')
ASYM = r'''% SRCBLOCK: w16-maze-wm/PREREG-w16-maze-wm-T1.md; w16-maze-wm/FINDINGS.md; w16-maze-wm/TABLES.md; w15-cmaze-t-harness/FINDINGS.md; w14-request-layer-fix/FINAL-REPORT.md; ../../outputs/w17-paper/figures/rendered/REQUESTS-w19.json
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
receives the identical input, is pending (Table~\ref{tab:cmazet-sym}, Appendix~\ref{app:mazewm}).
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

'''
E('MF-1 4.4 asymmetry paragraph',
  '% SRCBLOCK: w16-maze-wm/FINDINGS.md; w16-maze-wm/TABLES.md\n\\paragraph{CMAZE-T: what is not claimed.}',
  ASYM + '% SRCBLOCK: w16-maze-wm/FINDINGS.md; w16-maze-wm/TABLES.md\n\\paragraph{CMAZE-T: what is not claimed.}')
E('M7(e) 4.4 CMAZE-O runs 4--6 (a)', '1,232,064 read floats; runs 4--6), the store', '1,232,064 read floats; three scored runs), the store')
E('M7(e) 4.4 CMAZE-O runs 4--6 (b)', 'against a converged matched MLP, and runs 4--6 are attached',
  'against a converged matched MLP, and its three scored runs are attached')
E('MF-1 4.4 section summary', 'the CMAZE-T store row with its declared negatives,',
  'the CMAZE-T store row, scoped to its pair-design request, with its declared negatives,')

# ============================ MF-5(a), MF-4, MF-7: limitations =================================
E('MF-5(a) limitation 1 heading', r'\item \textbf{No store row on the maze.} On CMAZE-2',
  r'\item \textbf{No store row on the route tasks (CMAZE-2, CMAZE-2b).} On CMAZE-2',
  r'\item \emph{No store row on the maze.} On CMAZE-2',
  r'\item \emph{No store row on the route tasks (CMAZE-2, CMAZE-2b).} On CMAZE-2')
E('MF-5(a) limitation 1 CMAZE-T clause', 'fail to collapse (\\S\\ref{sec:verb}). % SRC: w13-seeds-and-tables/TABLES.md\n',
  'fail to collapse (\\S\\ref{sec:verb}). % SRC: w13-seeds-and-tables/TABLES.md\n'
  '  On CMAZE-T the store row is scoped to a word request with a pair block that the rivals do not\n'
  '  receive, and a symmetric-input test is pending (\\S\\ref{sec:maze-wm}).\n')
LIM4 = r'''{The encoder loop licenses no store claim.} Training the encoder through the store
  fired the pre-declared falsifier on row 3 of the shortened leg, a falsifier that involves no rival; % SRC: w13-seeds-and-tables/TABLES.md
  criterion D, growth of the store's margin over a matched learner, is mixed on the longer leg and
  unmeasured on the shortened one (\S\ref{sec:bard}).
'''
LIM_GEN = r'''{No functioning generative baseline.} LACE and PoE, re-implemented at their own budgets
  in this feature space, render no exact-match request on the generation table and score at the level
  of our landscape controls (Table~\ref{tab:lace-poe}); the study compares naive units only and has no
  working compositional-generation baseline.
'''
E('MF-4/MF-7 limitations 4 + new item',
  r'''\item \textbf{The store does not shape the representation.} Training the encoder through the
  store fired the pre-declared falsifier on row 3 (\S\ref{sec:bard}). % SRC: w13-seeds-and-tables/TABLES.md
''', r'\item \textbf' + LIM4 + r'\item \textbf' + LIM_GEN,
  r'''\item \emph{The store does not shape the representation.} When the encoder was trained through
  the store, the pre-declared falsifier fired on row 3 (\S\ref{sec:bard}). % SRC: w13-seeds-and-tables/TABLES.md
''', r'\item \emph' + LIM4 + r'\item \emph' + LIM_GEN)

# ============================ MF-8: leaving note ===============================================
E('MF-8 leaving note', 'composition that\nbeats read-path-matched learners on Shapes3D.',
  'composition on\nShapes3D that scores above the strongest read-path-matched learner of either of our rival sets\nand above a read-path-matched MLP fitted on the store\'s own rows.')

# ============================ App D: generation table de-jargoned + the LACE/PoE table ==========
E('M7(e) gentable caption run 3 of 3',
  r'''\caption{\textbf{The generation table, $\Kstar$ single design, \texttt{generation-table} scored run 3 of 3, % SRC: w15-lace-poe-rows/FINDINGS.md''',
  r'''\caption{\textbf{The generation table, $\Kstar$ single design, one pre-registered evaluation run, % SRC: w15-lace-poe-rows/FINDINGS.md''')
E('M7(e) gentable caption S2/S3',
  r'''S2 (beats the shipped read on every seed) \textsc{fail}, seed 0;
S3 (against \texttt{knn\_proto}) \textsc{lose};''',
  r'''S2 (beats the earlier store's read on every seed) \textsc{fail}, seed 0;
S3 (against the nearest-prototype ceiling) \textsc{lose};''')
E('M7(e) gentable row shipped read', r'shipped read (\texttt{clu}) & 3 &', r"earlier store's read & 3 &")
E('M7(e) gentable row knn', r'\texttt{knn\_proto} (ceiling) & 3 &', r'nearest-prototype ceiling & 3 &')
E('M7(e) gentable S2 row', r'$\Kstar$ $-$ shipped read (S2) & 3 &', r"$\Kstar$ $-$ earlier store's read (S2) & 3 &")
E('M7(e) gentable S3 row', r'$\Kstar$ $-$ \texttt{knn\_proto} (S3) & 3 &', r'$\Kstar$ $-$ nearest-prototype ceiling (S3) & 3 &')

LP_TABLE = r'''\begin{table}[h]
\centering\scriptsize
\caption{\textbf{LACE and PoE at their own budgets, Shapes3D value hold-out, TEST, $n=3$ seeds (0, 1,
2).} Distinct valid novel tuples per corner, on the corners of Table~\ref{tab:t7} (upper block) and of
every rung of Table~\ref{tab:ladder} (lower block). Neither is matched on the read path or on bytes;
both are re-implementations in this feature space, reported for completeness and not as evidence. On
the generation table (Table~\ref{tab:gentable-kstar}) their aggregates, 0.3032 and 0.1968, sit at the % SRC: w15-lace-poe-rows/FINDINGS.md
level of the store's four landscape controls (0.1088 to 0.1860) and of the store-removed arm % SRC: w15-lace-poe-rows/FINDINGS.md
(0.1204), and neither renders an exact-match request on any seed.} % SRC: w15-lace-poe-rows/FINDINGS.md
\label{tab:lace-poe}
\setlength{\tabcolsep}{2.6pt}
\begin{adjustbox}{max width=\linewidth}\begin{tabular}{lcccc}
\toprule
arm & seeds & row 3 (60 pair corners) & row 4 (49 six-factor corners) & paired $\Delta$, all 109 corners \\ % SRC: w12b-nr6-and-kstar/NR6-T7.md
\midrule
''' + T7_LP.replace('\\midrule\n', '', 1) + r'''\bottomrule
\end{tabular}\end{adjustbox}

\medskip
\begin{adjustbox}{max width=\linewidth}\begin{tabular}{lcccccc}
\toprule
arm & seeds & $n=2$ & $n=3$ & $n=4$ & $n=5$ & $n=6$ \\
\midrule
''' + LADDER_LP + r'''\bottomrule
\end{tabular}\end{adjustbox}
\end{table}

''' + LACE_PARA_MOVED + LACE_REST + '\n'
E('MF-7 LACE/PoE appendix table + moved paragraph',
  '\\end{tabular}\\end{adjustbox}\n\\end{table}\n\n\n% ---------------------------------------------------------------------------------------\n\\section{The $K=5$ endpoint of the sweep}',
  '\\end{tabular}\\end{adjustbox}\n\\end{table}\n\n' + LP_TABLE + '\n% ---------------------------------------------------------------------------------------\n\\section{The $K=5$ endpoint of the sweep}')

# ============================ App J: MF-4, MF-5(c), M7(e) ======================================
E('M7(e) tab:encloop caption', '3 seeds, 1 of 3 % SRC: w13-seeds-and-tables/TABLES.md\nscored runs.} A separate instrument;',
  '3 seeds, one % SRC: w13-seeds-and-tables/TABLES.md\npre-registered scored run.} A separate instrument;')
E('MF-4 tab:encloop margin row label', r'margin over matched learner, alone $\to$ arm & 3 &',
  r'margin over the frozen-feature rival (not a measurement of the refitted learner), alone $\to$ arm & 3 &')
E('MF-4 App J success-bar sentence',
  r'''\noindent The pre-declared success bar fails on both rows and the falsifier (net $\Delta\le0$ on two
or more seeds) fires on row 3. The matched learner sits at or within 0.02 of zero (0.006 on row 3).
''',
  r'''\noindent The falsifier, which involves no rival (net $\Delta\le0$ on two or more seeds), fires on
row 3. The pre-declared success bar, growth of the store's margin over the matched learner, is not
assessed on this leg: the learner rows its scorer loaded are the frozen-feature rival (below), which
sits at or within 0.02 of zero (0.006 on row 3), so criterion D on this leg is unmeasured.
''',
  r'''\noindent The pre-declared success bar fails on both rows, and the falsifier (net $\Delta\le0$ on two
or more seeds) fires on row 3. The matched learner sits at or within 0.02 of zero (0.006 on row 3).
''')
E('MF-5(c) App J stale not-run sentence',
  'its own encoder. Its scoring stages were not\nrun. % SRC: w8-BAR-D-fold.md',
  'its own encoder. % SRC: w8-BAR-D-fold.md',
  'its own encoder. Its scoring stages were\nnot run. % SRC: w8-BAR-D-fold.md',
  'its own encoder. % SRC: w8-BAR-D-fold.md')
E('M7(e) App J criterion D run 2 of 3',
  '\\texttt{encoder-loop} scored run (2 of 3) under its own pre-registration',
  'pre-registered scored run of the encoder loop, under its own pre-registration')
E('M7(e) App J last encoder-loop run', '(the last \\texttt{encoder-loop} run) is a declared not-run',
  '(its remaining pre-registered run) is a declared not-run')
E('M7(e) tab:bard-gpu caption', '\\texttt{encoder-loop} scored run 2 of 3.} Never shares',
  'one pre-registered scored run.} Never shares')

# ============================ App S: M7(e) + MF-1 caption =====================================
E('M7(e) harness warnings',
  'The harness ships three warnings, all printed here:\\\\\n\\texttt{composition\\_blind\\_PLANNER\\_scores\\_near\\_the\\_clu\\_on\\_unseen\\_nway},\\\\\n\\texttt{charter\\_C\\_control\\_did\\_not\\_collapse}, and\\\\\n\\texttt{cross\\_maze\\_lookup\\_floor\\_scores\\_near\\_the\\_clu}.',
  'The harness ships three warnings, all printed here: a composition-blind planner scores near the\nstore on unseen $n$-way; a landscape control did not collapse; and the cross-maze lookup floor\nscores near the store.')
E('M7(e)/MF-1 tab:cmazet caption',
  '(scored runs 1--3).} Given the grid state and an action, generate the next state; exact\ntransition judge.',
  '(one pre-registered scored run per seed).} Given the grid state and an action, generate the next\nstate; exact transition judge. The inputs are not symmetric: the store reads a \\texttt{[bias | main |\npair]} word request, the learned blind arms reduced forms of it, and the rivals the observation\n(\\S\\ref{sec:maze-wm}); the symmetric-input rows are Table~\\ref{tab:cmazet-sym}.')
E('M7(e) tab:cmazet replication row', 'Config C-512 (latent 512, 8 atoms; runs 7--9), rivals',
  'Config C-512 (latent 512, 8 atoms; three further scored runs), rivals')
E('M7(e) tab:cmazeo caption', 'seeds (scored runs 4--6). The rival margin is WITHDRAWN;',
  'seeds (one pre-registered scored run per seed). The rival margin is WITHDRAWN;')
E('M7(e) tab:cmazeo n-way row', 'so the store loses on every seed and runs 4--6 are attached',
  'so the store loses on every seed and its three scored runs are attached')
E('M7(e) config ladder runs 1--3 / 7--9', 'scored in runs\n1--3; Config C-512 (latent 512, 8 atoms, rank/width 0.031, 9,184 read floats), scored in runs 7--9\nagainst',
  'scored in three\nruns; Config C-512 (latent 512, 8 atoms, rank/width 0.031, 9,184 read floats), scored in three further\nruns against')
E('M7(e) config ladder runs 4--6', 'was scored in\nruns 4--6; Config C2', 'was scored in\nthree runs; Config C2')
E('M7(e) budget runs 10--12',
  '\\emph{Budget}: 9 of 18\nscored runs spent, 9 declared not-run. Runs 10--12 (CMAZE-O) are not run because',
  '\\emph{Budget}: of the eighteen\npre-registered scored runs, nine were spent and nine are declared not-run. A CMAZE-O scored\nreplication is not run because')
E('M7(e) fallback runs 13--18', 'Declared not-run (runs 13--18): the route task',
  'Declared not-run (six pre-registered runs): the route task')
E('M7(e) this programme', 'this programme', 'this study', count=2)

# ============================ App T: MF-4 ======================================================
E('MF-4 App T margin does not grow',
  r'''The store's margin over
the matched-parameter learner does not grow ($+12.078\to+10.311$ on row 3; $+24.687\to+26.048$
on row 4), so this is a finding about the encoder, not about the store, and ``the store shapes
the representation'' is not supported.''',
  r'''On this leg the store's margin
over the matched-parameter learner is not measured: the printed $+12.078\to+10.311$ (row 3) and
$+24.687\to+26.048$ (row 4) are margins over the frozen-feature rival, not over the refitted learner
(Appendix~\ref{app:encloop}), so criterion D here is unmeasured, and on the longer leg, where it was
measured, it is mixed with no claim either way. This is a finding about the encoder, and the claim
that the store shapes the representation is not made.''',
  None, None)

# ============================ MF-6: bannered figures, captions + index ========================
E('MF-6 fig_ladder caption', r'\textbf{The ladder at $\Kstar$ (placeholder figure).}',
  r'\textbf{The ladder at $\Kstar$ (placeholder figure, Appendix~\ref{app:placeholders}).}')
E('MF-6 fig_crossover caption',
  'Placeholder\nfigure, drawn from the single-seed middle rows; the three-seed curve is Table~\\ref{tab:kcurve3}.}',
  'Placeholder\nfigure (Appendix~\\ref{app:placeholders}), drawn from the single-seed middle rows, so its flat\nseed-0 line is not the three-seed result: three seeds do not resolve the validation plateau\n(\\S\\ref{sec:shapes}); the three-seed curve is Table~\\ref{tab:kcurve3}.}')
E('MF-6 fig_deletion caption', 'Placeholder figure, seed 0 only;',
  'Placeholder figure (Appendix~\\ref{app:placeholders}), seed 0 only;')
E('MF-6 placeholder index',
  '(\\S\\ref{sec:ladder}). Figure slots are marked as placeholders where they depend on pending rows.',
  '(\\S\\ref{sec:ladder}). Figure slots are marked as placeholders where they depend on pending rows.\n'
  'Three figures are placeholder figures in addition to the eleven rows, and each carries an in-image\n'
  'placeholder banner: Figure~\\ref{fig:ladder} (three seeds, but without the data-matched MLP of\n'
  'Table~\\ref{tab:ladder}), Figure~\\ref{fig:crossover} (the single-seed curve, which three seeds do not\n'
  'support as a selection) and Figure~\\ref{fig:deletion} (seed 0 only). The data each awaits has\n'
  'landed (Tables~\\ref{tab:ladder}, \\ref{tab:kcurve3} and~\\ref{tab:del}); each is to be redrawn from\n'
  'it, with the banner removed, before submission.')

os.makedirs(OUT, exist_ok=True)
for f in (V2, ST):
    open(os.path.join(OUT, f), 'w').write(t[f])
print('applied %d edits:' % len(log))
for x in log:
    print('  -', x)
