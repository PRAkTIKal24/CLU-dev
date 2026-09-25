r"""PW-2 item 5: every edit, as (id, section, [(scope, old, new)], moved).
scope: 'both' (same old/new in both drafts), 'v2' or 'style' (the style twin keeps its own wording).
Each `old` must occur exactly once in the draft it is applied to (apply.py asserts it).
`moved`: list of (numbers that left the prose, where they live now, SRC) -- written to NUMBERS-MOVED.md.
Every removed number also stays in the .tex as a `% PW2-5 moved -> ...; SRC: ...` pointer comment inside `new`."""

E = []
def add(id, section, pairs, moved):
    E.append(dict(id=id, section=section, pairs=pairs, moved=moved))

# ------------------------------------------------------------------ ABSTRACT
add('A1', 'Abstract', [('both',
r"""On a Shapes3D value hold-out at $\Kstar=32$ wells, the finest partition on a validation plateau
($K=5\ldots32$) that three seeds do not resolve, % SRC: w15-crossover-seeds/CROSSOVER-3SEED.md
the n-CLU composes held-out pairs at 32.350~$\pm$~0.815 and held-out six-factor combinations at
67.177~$\pm$~1.320 distinct valid novel tuples per held-out combination over three seeds, with all four landscape
controls collapsing to exactly 0.0000. % SRC: w12b-nr6-and-kstar/NR6-T7.md""",
r"""On a Shapes3D value hold-out, at the finest partition on a validation plateau
that the seeds do not resolve, % SRC: w15-crossover-seeds/CROSSOVER-3SEED.md
the n-CLU composes held-out pairs and held-out combinations of every factor, with every landscape
control collapsing. % SRC: w12b-nr6-and-kstar/NR6-T7.md
% PW2-5 moved -> Tab. t7 (caption: 32.350 +- 0.815, 67.177 +- 1.320; controls 0.000) and Sec. shapes (K*=32, plateau K=5..32, three seeds): Kstar=32; K=5..32; three seeds; 32.350 +- 0.815; six-factor; 67.177 +- 1.320; all four controls 0.0000; SRC: w12b-nr6-and-kstar/NR6-T7.md; w15-crossover-seeds/CROSSOVER-3SEED.md""")],
[('$\\Kstar=32$ wells; $K=5\\ldots32$; three seeds', 'Sec. 3.1 first paragraph (Head-held) and Tab. t7 caption', 'w15-crossover-seeds/CROSSOVER-3SEED.md'),
 ('32.350 ± 0.815; 67.177 ± 1.320', 'Tab. t7 (means in the table; the ± sample SD added to its caption)', 'w12b-nr6-and-kstar/NR6-T7.md'),
 ('six-factor; all four controls 0.0000', 'Tab. t7 (row 4 = six-factor; four landscape controls 0.000 (all))', 'w12b-nr6-and-kstar/NR6-T7.md')])

add('A2', 'Abstract', [('both',
r"""Against four read-path-matched baselines fitted on the n-CLU's own 131,072 rows, each also trained for
ten times its selected optimiser steps, the n-CLU leads the strongest, an MLP, by +7.03 distinct
valid novel tuples per held-out combination on its 109 held-out combinations (+7.49 / +8.60 / +5.02 over three seeds,
every per-seed 95\% interval above +3.73), by +6.54 against that baseline's best fit of any length, and
by +10.468 to +21.936 against the transformer; on every rung of an $n$-way ladder its mean margin
over the strongest baseline fitted on the rung's own rows is +5.133 to +12.550.""",
r"""Against read-path-matched baselines fitted on the n-CLU's own rows, each also trained far beyond
its selected optimiser steps, the n-CLU produces about seven more distinct
valid novel tuples per held-out combination than the strongest, an MLP, and on every rung of an $n$-way ladder
it stays ahead of the strongest baseline fitted on the rung's own rows.
% PW2-5 moved -> Tab. t7, Tab. ladder, Sec. 4.1 first paragraph (Head-held): four baselines; 131,072 rows; ten times; +7.03 on 109 combinations (+7.49 / +8.60 / +5.02, every 95% interval above +3.73); +6.54 best fit of any length; +10.468 to +21.936 transformer; ladder mean margin +5.133 to +12.550; SRC: w20-shapes-data-matched/FINDINGS.md; w20-shapes-data-matched/SHAPES-MATCHED.md""")],
[('four (baselines); 131,072 rows; ten times the steps', 'Tab. t7 (four families; caption: 131,072 rows, ten times the schedule)', 'w20-shapes-data-matched/SHAPES-MATCHED.md'),
 ('+7.03; 109 combinations; +7.49 / +8.60 / +5.02', 'Tab. t7 (paired Δ +7.486 / +8.596 / +5.018; all 109 combinations); mean +7.034 in Sec. 4.1 first paragraph (Head-held)', 'w20-shapes-data-matched/FINDINGS.md'),
 ('95% interval above +3.73; +6.54', 'Sec. 4.1 first paragraph (Head-held: "every 95% interval above +3.73", "mean +6.544")', 'w20-shapes-data-matched/FINDINGS.md'),
 ('+10.468 to +21.936', 'Tab. t7 (transformer paired Δ +21.936 / +19.615 / +10.468)', 'w20-shapes-data-matched/SHAPES-MATCHED.md'),
 ('+5.133 to +12.550', 'Tab. ladder (3-seed mean margin row)', 'w20-shapes-data-matched/SHAPES-MATCHED.md')])

add('A3', 'Abstract', [('both',
r"""After a value, a conjunction or a random 1\% of items is deleted, the n-CLU matches an n-CLU that
was never fitted on the deleted rows, and its address book likewise (content mode deviation at
most \sci{5.38}{-07}, book statistic deviation 0); the frozen encoder and decoder are outside the
object. Deletion has limits, which we print: the book's retained statistic is % SRC: w20-shapes-data-matched/SHAPES-MATCHED.md
0.97 GB and its refit takes 2,367 to 5,913 s; the deletable object is 4.0768 to 4.0897 times % SRC: w20-shapes-data-matched/FINDINGS.md; w13-seeds-and-tables/TABLES.md; w13-referee-2/REFEREE-v1.md
the 524,288-byte reference in its footprint breakdown; and deleting the conjunction red sphere % SRC: w13-referee-2/REFEREE-v1.md; w13-seeds-and-tables/TABLES.md
changes what the book knows about the shared value: survivors fall by 6.13 / 6.43 / 6.30 tuples per % SRC: w20-shapes-data-matched/FINDINGS.md
combination, against at most 0.11 for a random 1\% deletion. % SRC: w20-shapes-data-matched/FINDINGS.md""",
r"""After a value, a conjunction or a random subset of items is deleted, the n-CLU matches an n-CLU that
was never fitted on the deleted rows, and its address book likewise, to floating-point roundoff; the frozen encoder and decoder are outside the
object. Deletion has limits, which we print: the book's retained statistic is % SRC: w20-shapes-data-matched/SHAPES-MATCHED.md
as large as its fit data and its refit is slower than retraining a baseline; the deletable object is several times % SRC: w20-shapes-data-matched/FINDINGS.md; w13-seeds-and-tables/TABLES.md; w13-referee-2/REFEREE-v1.md
the earlier design's byte reference in its footprint breakdown; and deleting the conjunction red sphere % SRC: w13-referee-2/REFEREE-v1.md; w13-seeds-and-tables/TABLES.md
changes what the book knows about the shared value: survivors fall by about six tuples per % SRC: w20-shapes-data-matched/FINDINGS.md
combination, far more than after a random item deletion. % SRC: w20-shapes-data-matched/FINDINGS.md
% PW2-5 moved -> Tab. del (random 1%; mode deviation max 5.38e-07; statistic deviation 0; survivors -6.128 / -6.431 / -6.303; item at most -0.110; book refit 2,367.3 to 5,912.9 s; 967,304,800 B statistic), Tab. rivals-del (retrain 547.7 to 2,064.5 s), Tab. ledger-kstar (4.0768 to 4.0897 x the 524,288 B reference): 1%; 5.38e-07; 0; 0.97 GB; 2,367 to 5,913 s; 4.0768 to 4.0897; 524,288; 6.13 / 6.43 / 6.30; 0.11; SRC: w20-shapes-data-matched/FINDINGS.md; w20-shapes-data-matched/SHAPES-MATCHED.md; w13-seeds-and-tables/TABLES.md""")],
[('random 1% (item scope)', 'Tab. del (row "item: random 1%")', 'w20-shapes-data-matched/SHAPES-MATCHED.md'),
 ('mode deviation at most 5.38e-07; book statistic deviation 0', 'Tab. del columns (a), (b)', 'w20-shapes-data-matched/SHAPES-MATCHED.md'),
 ('0.97 GB', 'Tab. del cost row ("reads the 967,304,800 B statistic")', 'w20-shapes-data-matched/SHAPES-MATCHED.json'),
 ('2,367 to 5,913 s', 'Tab. del cost row (book refit 2,590.7 … 2,367.3 / 5,912.9)', 'w20-shapes-data-matched/SHAPES-MATCHED.md'),
 ('4.0768 to 4.0897; 524,288-byte reference', 'Tab. ledger-kstar (last row and caption)', 'w13-seeds-and-tables/TABLES.md; w13-referee-2/REFEREE-v1.md'),
 ('6.13 / 6.43 / 6.30; at most 0.11', 'Tab. del column (d) (−6.128 / −6.431 / −6.303; item −0.018 / −0.110 / −0.009)', 'w20-shapes-data-matched/FINDINGS.md')])

add('A4', 'Abstract', [('both',
r"""with the one measured clue toward a""",
r"""with the measured clue toward a""")],
[('one (the one measured clue)', 'numeral dropped, no value (determiner); the clue is Sec. 5.3', '—')])

# ------------------------------------------------------------------ INTRODUCTION
NOV_PTR = r"""% PW2-5 moved -> pointer only (no table carries the screen): approximately 4,050 unique works; 3,038 generation-side; 759 deletion-side; 254 from 21 orthogonal queries; about 100 from a final sweep; deletion language in 10 of 3,038 (0.3%), seven false positives; 0 of 3,038; 4 of 6 claim atoms; 759 deletion-side works; SRC: _thinkpad/w8/w8-novelty/NOVELTY.md"""
add('I1', 'Introduction', [('v2',
r"""approximately 4,050 unique works: 3,038 citing the generation-side anchors, 759 citing the
deletion-side anchors, 254 from 21 orthogonal queries that cite neither, and about 100 from
a final targeted sweep. % SRC: _thinkpad/w8/w8-novelty/NOVELTY.md
Deletion language appears in 10 of the 3,038 generation-side works (0.3\%), seven of them
false positives and none deleting training rows; no work among the 3,038 combines frozen-decoder,
tensor-head and label-index language (0 of 3,038); the highest number of the six claim
atoms reached by any single work is 4 of 6; and of the 759 deletion-side works none combines
principal-subspace, generative and frozen-model language. % SRC: _thinkpad/w8/w8-novelty/NOVELTY.md""",
r"""about four thousand unique works: those citing the generation-side anchors, those citing the
deletion-side anchors, those from orthogonal queries that cite neither, and those from
a final targeted sweep. % SRC: _thinkpad/w8/w8-novelty/NOVELTY.md
Deletion language appears in ten of the generation-side works, most of them
false positives and none deleting training rows; no generation-side work combines frozen-decoder,
tensor-head and label-index language; the highest number of the six claim
atoms reached by any single work is four; and of the deletion-side works none combines
principal-subspace, generative and frozen-model language. % SRC: _thinkpad/w8/w8-novelty/NOVELTY.md
""" + NOV_PTR),
('style',
r"""screened approximately 4,050 unique works: 3,038 citing the generation-side anchors, 759 citing
the deletion-side anchors, 254 from 21 orthogonal queries that cite neither, and about 100 from
a final targeted sweep. % SRC: _thinkpad/w8/w8-novelty/NOVELTY.md
Of the 3,038 generation-side works, deletion language appears in 10 (0.3\%), seven of them
false positives and none deleting training rows; no work among the 3,038 combines frozen-decoder, tensor-head
and label-index language (0 of 3,038); no single work reaches more than 4 of the 6 claim atoms;
and none of the 759 deletion-side works combines principal-subspace, generative and
frozen-model language. % SRC: _thinkpad/w8/w8-novelty/NOVELTY.md""",
r"""screened about four thousand unique works: those citing the generation-side anchors, those citing
the deletion-side anchors, those from orthogonal queries that cite neither, and those from
a final targeted sweep. % SRC: _thinkpad/w8/w8-novelty/NOVELTY.md
Of the generation-side works, deletion language appears in ten, most of them
false positives and none deleting training rows; no generation-side work combines frozen-decoder, tensor-head
and label-index language; no single work reaches more than four of the six claim atoms;
and none of the deletion-side works combines principal-subspace, generative and
frozen-model language. % SRC: _thinkpad/w8/w8-novelty/NOVELTY.md
""" + NOV_PTR)],
[('approximately 4,050; 3,038; 759; 254; 21; about 100; 10 of 3,038 (0.3%); seven; 0 of 3,038; 4 of 6; 759',
  '% pointer only (no table in the paper carries the literature screen); rounded prose keeps "about four thousand", "ten", "most", "four of the six"', '_thinkpad/w8/w8-novelty/NOVELTY.md')])

add('I2', 'Introduction', [('both',
r"""  fitted on its own 131,072 rows, with each baseline's convergence printed beside it: by +7.03 tuples % SRC: w20-shapes-data-matched/FINDINGS.md
  per held-out factor combination over the strongest (+6.54 against that baseline's best fit of any length) and by +10.468 % SRC: w20-shapes-data-matched/FINDINGS.md; w20-shapes-data-matched/SHAPES-MATCHED.md
  to +21.936 over the transformer (Sec.~\ref{sec:shapes-t7}). On every rung of an $n$-way ladder it % SRC: w20-shapes-data-matched/SHAPES-MATCHED.md
  scores above the strongest baseline fitted on the rung's own rows, by +5.133 to +12.550 on the seed % SRC: w20-shapes-data-matched/SHAPES-MATCHED.md
  mean (Sec.~\ref{sec:ladder}); for $n\ge3$ the rungs are a training-data axis and not a depth axis.""",
r"""  fitted on its own rows, with each baseline's convergence printed beside it: by about seven tuples % SRC: w20-shapes-data-matched/FINDINGS.md
  per held-out factor combination over the strongest, an MLP, and by more over the transformer % SRC: w20-shapes-data-matched/FINDINGS.md; w20-shapes-data-matched/SHAPES-MATCHED.md
  (Tab.~\ref{tab:t7}, Sec.~\ref{sec:shapes-t7}). On every rung of an $n$-way ladder it % SRC: w20-shapes-data-matched/SHAPES-MATCHED.md
  scores above the strongest baseline fitted on the rung's own rows % SRC: w20-shapes-data-matched/SHAPES-MATCHED.md
  (Tab.~\ref{tab:ladder}, Sec.~\ref{sec:ladder}); for $n\ge3$ the rungs are a training-data axis and not a depth axis.
  % PW2-5 moved -> Tab. t7, Tab. ladder, Sec. 4.1 first paragraph (Head-held): 131,072 rows; +7.03; +6.54; +10.468 to +21.936; +5.133 to +12.550; SRC: w20-shapes-data-matched/FINDINGS.md; w20-shapes-data-matched/SHAPES-MATCHED.md""")],
[('131,072 rows', 'Tab. t7 caption', 'w20-shapes-data-matched/SHAPES-MATCHED.md'),
 ('+7.03; +6.54', 'Tab. t7 (per seed) and Sec. 4.1 first paragraph (Head-held: mean +7.034, +6.544)', 'w20-shapes-data-matched/FINDINGS.md'),
 ('+10.468 to +21.936', 'Tab. t7 (transformer row)', 'w20-shapes-data-matched/SHAPES-MATCHED.md'),
 ('+5.133 to +12.550', 'Tab. ladder (3-seed mean margin row)', 'w20-shapes-data-matched/SHAPES-MATCHED.md')])

add('I3', 'Introduction', [('both',
r"""  training row, but the book's retained statistic is 0.97 GB, the size of its fit data, and its refit % SRC: w20-shapes-data-matched/FINDINGS.md
  takes 2,367 to 5,913 s, so this is neither a size nor a speed claim. Refusal after deletion is a % SRC: w20-shapes-data-matched/FINDINGS.md""",
r"""  training row, but the book's retained statistic is nearly a gigabyte, the size of its fit data, and its refit % SRC: w20-shapes-data-matched/FINDINGS.md
  takes longer than retraining a baseline (Tabs.~\ref{tab:del} and~\ref{tab:rivals-del}), so this is neither a size nor a speed claim. Refusal after deletion is a % SRC: w20-shapes-data-matched/FINDINGS.md
  % PW2-5 moved -> Tab. del (967,304,800 B statistic; refit 2,367.3 to 5,912.9 s), Tab. rivals-del (547.7 to 2,064.5 s): 0.97 GB; 2,367 to 5,913 s; SRC: w20-shapes-data-matched/FINDINGS.md; w20-shapes-data-matched/SHAPES-MATCHED.md""")],
[('0.97 GB', 'Tab. del cost row (967,304,800 B)', 'w20-shapes-data-matched/SHAPES-MATCHED.json'),
 ('2,367 to 5,913 s', 'Tab. del cost row; comparison against Tab. rivals-del', 'w20-shapes-data-matched/SHAPES-MATCHED.md')])

add('I4', 'Introduction', [('both',
r"""  the best matched baseline on three seeds, level with it on two and behind on one (mean
  +0.046). A colour-relational request coding lifts the n-CLU on every seed, but baselines handed the % SRC: w21-maze-t-seed4/FINDINGS.md""",
r"""  the best matched baseline on three seeds, level with it on two and behind on one
  (Tab.~\ref{tab:cmazet-sym}). A colour-relational request coding lifts the n-CLU on every seed, but baselines handed the % SRC: w21-maze-t-seed4/FINDINGS.md
  % PW2-5 moved -> Tab. cmazet-sym (words + observation row, mean +0.046): mean +0.046; SRC: w21-maze-t-seed4/FINDINGS.md""")],
[('mean +0.046', 'Tab. cmazet-sym (n-CLU words + observation row, "mean +0.046")', 'w21-maze-t-seed4/FINDINGS.md')])

# ------------------------------------------------------------------ METHODS
add('M1', 'Methods', [('both',
r"""with ceilings of 36 (pair combinations) and 72 (six-factor
combinations).""",
r"""with a fixed ceiling for pair and for six-factor
combinations (Tab.~\ref{tab:t7})."""),
('both',
r"""before any number. % SRC: w13-seeds-and-tables/TABLES.md""",
r"""before any number. % SRC: w13-seeds-and-tables/TABLES.md
% PW2-5 moved -> Tab. t7 caption (ceiling 36 on row 3, 72 on row 4): 36; 72; SRC: w13-seeds-and-tables/TABLES.md""")],
[('ceilings 36 and 72', 'Tab. t7 caption ("ceiling 36 on row 3, 72 on row 4"); Tab. ablations header', 'w13-seeds-and-tables/TABLES.md')])

# ------------------------------------------------------------------ EXPERIMENTS
add('X1', 'Experiments', [('both',
r"""At $\Kstar$ the fitted baselines carry 130,433 to 131,891 parameters.""",
r"""At $\Kstar$ the fitted baselines carry the parameter counts of Tab.~\ref{tab:t7}.
% PW2-5 moved -> Tab. t7 caption (130,433 to 131,891): 130,433 to 131,891 parameters; SRC: w13-seeds-and-tables/TABLES.md; w20-shapes-data-matched/SHAPES-MATCHED.md""")],
[('130,433 to 131,891 parameters', 'Tab. t7 caption', 'w20-shapes-data-matched/SHAPES-MATCHED.md')])

add('X2', 'Experiments', [('both',
r"""of read-path-matched baselines, fitted on 32,256 rows of a combination-split pool, at 117,878 to % SRC: _thinkpad/w4/spoke-a-rivals/rivals/seed_0/mlp.rival.json; _thinkpad/w4/spoke-a-rivals/REPORT.md
119,858 parameters, 40 epochs at batch 256 and learning rate 0.003 (5,040 steps), the same for % SRC: _thinkpad/w4/spoke-a-rivals/REPORT.md; _thinkpad/w4/spoke-a-rivals/rivals/seed_0/mlp.json
every family and seed. % SRC: _thinkpad/w4/spoke-a-rivals/rivals/seed_0/mlp.json""",
r"""of read-path-matched baselines, fitted on a smaller combination-split pool, % SRC: _thinkpad/w4/spoke-a-rivals/rivals/seed_0/mlp.rival.json; _thinkpad/w4/spoke-a-rivals/REPORT.md
with one schedule, the same for % SRC: _thinkpad/w4/spoke-a-rivals/REPORT.md; _thinkpad/w4/spoke-a-rivals/rivals/seed_0/mlp.json
every family and seed (Tab.~\ref{tab:ladder}). % SRC: _thinkpad/w4/spoke-a-rivals/rivals/seed_0/mlp.json
% PW2-5 moved -> Tab. ladder (caption and header: 32,256 rows, 5,040 steps; notes: 117,878 to 119,858 parameters, 40 epochs, batch 256, learning rate 0.003): 32,256; 117,878 to 119,858; 40 epochs; batch 256; learning rate 0.003; 5,040 steps; SRC: _thinkpad/w4/spoke-a-rivals/REPORT.md; _thinkpad/w4/spoke-a-rivals/rivals/seed_0/mlp.json""")],
[('32,256 rows; 5,040 steps', 'Tab. ladder caption and header (already there)', '_thinkpad/w4/spoke-a-rivals/rivals/seed_0/mlp.rival.json; w21-ladder-rivals-32k/CAPTION.md'),
 ('117,878 to 119,858 parameters; 40 epochs; batch 256; learning rate 0.003', 'Tab. ladder caption, Notes (ADDED)', '_thinkpad/w4/spoke-a-rivals/REPORT.md; _thinkpad/w4/spoke-a-rivals/rivals/seed_0/mlp.json')])

add('X3', 'Experiments', [('both',
r"""refit on each rung's own 131,072-row pool, and the earlier 32,256-row set, whose features % SRC: w20-shapes-data-matched/FINDINGS.md""",
r"""refit on each rung's own pool, and the earlier set, whose features % SRC: w20-shapes-data-matched/FINDINGS.md"""),
('both',
r"""The landscape controls at $\Kstar$ collapse on all 15 legs by the pre-declared 1.0 margin;
13 of 15 read exactly 0.0000 and two ($n=3$, seeds 0 and 1) read 0.0083 on
landscape shuffled, one surviving novel tuple over 120 combinations, which we print
rather than round away. % SRC: w12b-nr6-and-kstar/FINDINGS.md""",
r"""The landscape controls at $\Kstar$ collapse on every rung--seed leg by the pre-declared margin;
two legs keep a single surviving novel tuple on
landscape shuffled (Tab.~\ref{tab:ladder}), which we print
rather than round away. % SRC: w12b-nr6-and-kstar/FINDINGS.md
% PW2-5 moved -> Tab. ladder (caption and header: 131,072-row pool, 32,256 rows; notes: 15 legs, 13 of 15 at 0.0000, 0.0083 at n=3 seeds 0 and 1, 120 combinations): 131,072; 32,256; 15; 1.0; 13 of 15; 0.0000; 0.0083; 120; SRC: w12b-nr6-and-kstar/FINDINGS.md; w20-shapes-data-matched/FINDINGS.md""")],
[('131,072-row pool; 32,256-row set', 'Tab. ladder caption and section headers', 'w20-shapes-data-matched/FINDINGS.md'),
 ('1.0 margin', 'Methods ("margin of 1.0 tuple per held-out combination declared before any number")', 'w13-seeds-and-tables/TABLES.md'),
 ('all 15 legs; 13 of 15 at 0.0000; 0.0083 at n=3, seeds 0 and 1; 120 combinations', 'Tab. ladder caption, Notes (ADDED)', 'w12b-nr6-and-kstar/FINDINGS.md')])

add('X4', 'Experiments', [('both',
r"""Three deletions are run on the $\Kstar$ object on TEST, on its 131,072-row fit pool, at three % SRC: w20-shapes-data-matched/FINDINGS.md""",
r"""Three deletions are run on the $\Kstar$ object on TEST, on its fit pool (Tab.~\ref{tab:del}), at three % SRC: w20-shapes-data-matched/FINDINGS.md
% PW2-5 moved -> Tab. del caption (fit pool 131,072 rows): 131,072; SRC: w20-shapes-data-matched/FINDINGS.md"""),
('both',
r"""red $\wedge$ cube, deletes zero rows,""",
r"""red $\wedge$ cube, deletes no rows,""")],
[('131,072-row fit pool', 'Tab. del caption', 'w20-shapes-data-matched/FINDINGS.md'),
 ('zero rows (red ∧ cube)', 'Tab. del (rows del. 0)', 'w20-shapes-data-matched/SHAPES-MATCHED.md')])

# ------------------------------------------------------------------ RESULTS
add('R1', 'Results', [('v2',
r"""\noindent Baselines are matched on the read path (130,433 to 131,891 parameters); the deletable object""",
r"""\noindent Baselines are matched on the read path (Tab.~\ref{tab:t7}); the deletable object"""),
('style',
r"""\noindent The baselines are matched on the read path (130,433 to 131,891 parameters). The deletable""",
r"""\noindent The baselines are matched on the read path (Tab.~\ref{tab:t7}). The deletable"""),
('v2',
r"""matched. Baseline pins, geometry and the stress comparison are in App.~\ref{app:t7more}.
% SRC: w13-seeds-and-tables/TABLES.md""",
r"""matched. Baseline pins, geometry and the stress comparison are in App.~\ref{app:t7more}.
% SRC: w13-seeds-and-tables/TABLES.md
% PW2-5 moved -> Tab. t7 caption: 130,433 to 131,891 parameters; SRC: w20-shapes-data-matched/SHAPES-MATCHED.md"""),
('style',
r"""comparison. % SRC: w13-seeds-and-tables/TABLES.md
""",
r"""comparison. % SRC: w13-seeds-and-tables/TABLES.md
% PW2-5 moved -> Tab. t7 caption: 130,433 to 131,891 parameters; SRC: w20-shapes-data-matched/SHAPES-MATCHED.md
""")],
[('130,433 to 131,891 parameters', 'Tab. t7 caption', 'w20-shapes-data-matched/SHAPES-MATCHED.md')])

add('R2', 'Results', [('both',
r"""Transformer at their selected schedule, all fitted on the same 131,072 rows) and the lookup table as the % SRC: w20-shapes-data-matched/FINDINGS.md
floor. Every model's output on a request is drawn through the same frozen decoder, and each panel
carries the judge's verdict on that image. Over all 5,688 held-out requests at seed 0 the judge marks % SRC: ../../outputs/pw1-render-shapes/REQUESTS-pw1.json
the n-CLU valid on 5,177, the MLP on 4,361, the LSTM on 3,703, the Transformer on 2,786, the continuous % SRC: ../../outputs/pw1-render-shapes/REQUESTS-pw1.json
MLP on 3,110 and the lookup table on 897. % SRC: ../../outputs/pw1-render-shapes/REQUESTS-pw1.json
Where the n-CLU is invalid, the image is almost always wrong in one factor only (497 of its 511 invalid % SRC: ../../outputs/pw1-render-shapes/factor-miss-seed0.txt
renders), most often the scale (329) or the orientation (125), less often the shape (71). % SRC: ../../outputs/pw1-render-shapes/factor-miss-seed0.txt
Requests on which a baseline succeeds and the n-CLU fails exist for every baseline (310 requests at % SRC: ../../outputs/pw1-render-shapes/REQUESTS-pw1.json
seed 0 have at least one); they are not confined to the capsule: 170 of the 310 are capsules, 100 % SRC: ../../outputs/pw1-render-shapes/REQUESTS-pw1.json
cylinders and 40 spheres. % SRC: ../../outputs/pw1-render-shapes/REQUESTS-pw1.json""",
r"""Transformer at their selected schedule, all fitted on the same rows) and the lookup table as the % SRC: w20-shapes-data-matched/FINDINGS.md
floor. Every model's output on a request is drawn through the same frozen decoder, and each panel
carries the judge's verdict on that image. Over all held-out requests at seed 0 the judge marks % SRC: ../../outputs/pw1-render-shapes/REQUESTS-pw1.json
the n-CLU valid on about nine in ten, more often than any baseline, and the lookup table % SRC: ../../outputs/pw1-render-shapes/REQUESTS-pw1.json
least often (Fig.~\ref{fig:shapes-per-model}). % SRC: ../../outputs/pw1-render-shapes/REQUESTS-pw1.json
Where the n-CLU is invalid, the image is almost always wrong in one factor only % SRC: ../../outputs/pw1-render-shapes/factor-miss-seed0.txt
(most often the scale or the orientation, less often the shape). % SRC: ../../outputs/pw1-render-shapes/factor-miss-seed0.txt
Requests on which a baseline succeeds and the n-CLU fails exist for every baseline % SRC: ../../outputs/pw1-render-shapes/REQUESTS-pw1.json
and are not confined to the capsule % SRC: ../../outputs/pw1-render-shapes/REQUESTS-pw1.json
(Fig.~\ref{fig:shapes-baseline-wins}). % SRC: ../../outputs/pw1-render-shapes/REQUESTS-pw1.json
% PW2-5 moved -> Fig. shapes-per-model caption (ADDED: valid counts, factor misses) and Fig. shapes-baseline-wins caption (310; 170 / 100 / 40): 131,072; 5,688; 5,177; 4,361; 3,703; 2,786; 3,110; 897; 497 of 511; 329; 125; 71; 310; 170 of 310; 100; 40; SRC: ../../outputs/pw1-render-shapes/REQUESTS-pw1.json; ../../outputs/pw1-render-shapes/factor-miss-seed0.txt""")],
[('131,072 rows', 'Tab. t7 caption', 'w20-shapes-data-matched/FINDINGS.md'),
 ('5,688; 5,177; 4,361; 3,703; 2,786; 3,110; 897', 'Fig. shapes-per-model caption (ADDED); 5,688 also in Fig. shapes-baseline-wins caption', '../../outputs/pw1-render-shapes/REQUESTS-pw1.json'),
 ('497 of 511; 329; 125; 71', 'Fig. shapes-per-model caption (ADDED)', '../../outputs/pw1-render-shapes/factor-miss-seed0.txt'),
 ('310; 170 of 310; 100; 40', 'Fig. shapes-baseline-wins caption (already there)', '../../outputs/pw1-render-shapes/REQUESTS-pw1.json')])

add('R3', 'Results', [('v2',
r"""rungs $n=3,4,5,6$ probe the identical 120 combinations (the same 8,640 rows); what varies is how
coarsely the hold-out is carved, and with it the held-out row count (300,000 / 235,200 /
213,600 / 200,640). Every arm fitted on the rung's own pool tracks that count at
$r=-0.96$ to $-0.98$, and every arm not fitted on it is exactly constant.""",
r"""rungs $n=3,4,5,6$ probe the identical combinations and rows; what varies is how
coarsely the hold-out is carved, and with it the held-out row count. Every arm fitted on the rung's own pool tracks that count closely,
inversely, and every arm not fitted on it is exactly constant (Tab.~\ref{tab:ladder})."""),
('style',
r"""rungs $n=3,4,5,6$ probe the identical 120 combinations (the same 8,640 rows). What varies is how
coarsely the hold-out is carved and, with it, the held-out row count (300,000 / 235,200 /
213,600 / 200,640). Every arm fitted on the rung's own pool tracks that count at
$r=-0.96$ to $-0.98$, and every arm not fitted on it is exactly constant.""",
r"""rungs $n=3,4,5,6$ probe the identical combinations and rows. What varies is how
coarsely the hold-out is carved and, with it, the held-out row count. Every arm fitted on the rung's own pool tracks that count closely,
inversely, and every arm not fitted on it is exactly constant (Tab.~\ref{tab:ladder})."""),
('both',
r"""Only $n=2$ against $n=6$ is a genuine change of probe (60 combinations of 36 rows against 120 of
72, disjoint sets). % SRC: w12-composition-ladder/FINDINGS.md""",
r"""Only $n=2$ against $n=6$ is a genuine change of probe, on disjoint sets of combinations
(Tab.~\ref{tab:ladder}). % SRC: w12-composition-ladder/FINDINGS.md
% PW2-5 moved -> Tab. ladder (caption: 60 and 120 combinations; notes ADDED: 8,640 rows, row counts, r, 36 / 72 rows): 120; 8,640; 300,000 / 235,200 / 213,600 / 200,640; r=-0.96 to -0.98; 60 of 36 rows; 120 of 72; SRC: w12-composition-ladder/FINDINGS.md; w13-seeds-and-tables/TABLES.md"""),
('both',
r"""bootstrap interval~\citep{efron_bootstrap_1979} lies above zero on every rung and seed (lowest lower
bound +2.46, at $n=3$, seed 2), and the smallest seed-level margin is +3.85. Two caveats travel with % SRC: w20-shapes-data-matched/FINDINGS.md""",
r"""bootstrap interval~\citep{efron_bootstrap_1979} lies above zero on every rung and seed
(Tab.~\ref{tab:ladder}). Two caveats travel with % SRC: w20-shapes-data-matched/FINDINGS.md"""),
('both',
r"""moves the LSTM from 20.55 to 21.85 at $n=2$, seed 0, printed and not carried. The earlier set's % SRC: w20-shapes-data-matched/FINDINGS.md""",
r"""moves the LSTM by about one tuple at one rung and seed (Tab.~\ref{tab:ladder}), printed and not carried. The earlier set's % SRC: w20-shapes-data-matched/FINDINGS.md
% PW2-5 moved -> Tab. ladder (notes ADDED: +2.46, 20.55 -> 21.85; +3.850 in the margin row): +2.46 at n=3 seed 2; +3.85; 20.55 to 21.85 at n=2 seed 0; SRC: w20-shapes-data-matched/FINDINGS.md"""),
('both',
r"""The n-CLU-content set asks whether the data-matched set's 131,072 rows, four times the n-CLU's
32,768 content rows, favour the baselines. They do not: fitted on exactly the n-CLU's content rows, % SRC: w21-ladder-rivals-32k/FINDINGS.md
the baselines leave the margin where the data-matched set left it, +6.08 / +11.48 / +14.00 / +9.47 / % SRC: w21-ladder-rivals-32k/FINDINGS.md
+10.36 at $n=2,\ldots,6$ (mean over rungs +10.28, and +10.75 step-matched, against +9.97 for the % SRC: w21-ladder-rivals-32k/FINDINGS.md
data-matched set and +17.53 for the earlier set), and every 95\% interval of the n-CLU-content % SRC: w21-ladder-rivals-32k/FINDINGS.md
margins lies above zero (lowest lower bound +1.98, at $n=2$, seed 2, against the step-matched MLP). % SRC: w21-ladder-rivals-32k/FINDINGS.md
The MLP is the strongest n-CLU-content baseline on every rung and seed and is flat from 51,200 to % SRC: w21-ladder-rivals-32k/FINDINGS.md
204,800 steps; the continuous MLP is still gaining at 204,800 steps and never becomes the strongest % SRC: w21-ladder-rivals-32k/FINDINGS.md
baseline. One more caveat bears on every single-fit baseline number in this paper: two MLP fits with the
same rows, width, schedule and 2,560 steps, differing only in the random draw, differ per seed by % SRC: w21-ladder-rivals-32k/FINDINGS.md
$-2.15$ to $+4.13$ novel tuples per held-out combination, and their per-combination interval excludes zero on 12 of 15 % SRC: w21-ladder-rivals-32k/FINDINGS.md
rung--seed legs. Family differences smaller than that on a single seed are not read as real.""",
r"""The n-CLU-content set asks whether the data-matched set's rows, four times the n-CLU's
content rows, favour the baselines. They do not: fitted on exactly the n-CLU's content rows, % SRC: w21-ladder-rivals-32k/FINDINGS.md
the baselines leave the margin where the data-matched set left it % SRC: w21-ladder-rivals-32k/FINDINGS.md
(Tab.~\ref{tab:ladder}), % SRC: w21-ladder-rivals-32k/FINDINGS.md
and every 95\% interval of the n-CLU-content % SRC: w21-ladder-rivals-32k/FINDINGS.md
margins lies above zero. % SRC: w21-ladder-rivals-32k/FINDINGS.md
The MLP is the strongest n-CLU-content baseline on every rung and seed and is flat from its ten-times fit to its % SRC: w21-ladder-rivals-32k/FINDINGS.md
step-matched fit; the continuous MLP is still gaining at its step-matched fit and never becomes the strongest % SRC: w21-ladder-rivals-32k/FINDINGS.md
baseline. One more caveat bears on every single-fit baseline number in this paper: two MLP fits with the
same rows, width and schedule, differing only in the random draw, differ per seed by % SRC: w21-ladder-rivals-32k/FINDINGS.md
up to about four novel tuples per held-out combination, and their per-combination interval excludes zero on most % SRC: w21-ladder-rivals-32k/FINDINGS.md
rung--seed legs (Tab.~\ref{tab:ladder}). Family differences smaller than that on a single seed are not read as real.
% PW2-5 moved -> Tab. ladder (caption and header: 131,072, 32,768, 51,200, 204,800; per-rung means +6.08 / +11.48 / +14.00 / +9.47 / +10.36; notes ADDED: +10.28, +10.75, +9.97, +17.53, +1.98, 2,560 steps, -2.15 to +4.13, 12 of 15): 131,072; 32,768; +6.08 / +11.48 / +14.00 / +9.47 / +10.36; +10.28; +10.75; +9.97; +17.53; +1.98 at n=2 seed 2; 51,200; 204,800; 2,560 steps; -2.15 to +4.13; 12 of 15; SRC: w21-ladder-rivals-32k/FINDINGS.md""")],
[('120 combinations; 60 combinations', 'Tab. ladder caption (already there)', 'w12-composition-ladder/FINDINGS.md'),
 ('8,640 rows; 300,000 / 235,200 / 213,600 / 200,640; r = −0.96 to −0.98; 36 rows; 72 rows', 'Tab. ladder caption, Notes (ADDED)', 'w12-composition-ladder/FINDINGS.md; w13-seeds-and-tables/TABLES.md'),
 ('+2.46 (n=3, seed 2); 20.55 → 21.85 (n=2, seed 0)', 'Tab. ladder caption, Notes (ADDED)', 'w20-shapes-data-matched/FINDINGS.md'),
 ('+3.85', 'Tab. ladder margin row (+3.850, n=2 seed 2)', 'w20-shapes-data-matched/FINDINGS.md'),
 ('131,072; 32,768; 51,200; 204,800', 'Tab. ladder caption and set headers', 'w21-ladder-rivals-32k/CAPTION.md'),
 ('+6.08 / +11.48 / +14.00 / +9.47 / +10.36', 'Tab. ladder, n-CLU-content 3-seed mean row', 'w21-ladder-rivals-32k/TABLES.md'),
 ('+10.28; +10.75; +9.97; +17.53; +1.98 (n=2, seed 2); 2,560 steps; −2.15 to +4.13; 12 of 15', 'Tab. ladder caption, Notes (ADDED)', 'w21-ladder-rivals-32k/FINDINGS.md')])

add('R4', 'Results', [('both',
r"""On TEST, $n=6$ at $n=3$'s coverage is therefore 61.614 against $n=6$'s 66.603 ($\Kstar=32$, 3 seeds): % SRC: w15-coverage-control/FINDINGS.md
the whole $n=3\to6$ rise of +4.989 (+5.758 / +3.467 / +5.742 per seed) is fit-pool coverage and % SRC: w15-coverage-control/FINDINGS.md""",
r"""On TEST, $n=6$ at $n=3$'s coverage is therefore the $n=3$ cell of Tab.~\ref{tab:ladder}: % SRC: w15-coverage-control/FINDINGS.md
the whole $n=3\to6$ rise, nearly five tuples per held-out combination, is fit-pool coverage and % SRC: w15-coverage-control/FINDINGS.md"""),
('both',
r"""acceptance 1.000000 under either rung's bitmap, \texttt{distinct\_valid} equal to % SRC: w15-coverage-control/FINDINGS.md
\texttt{distinct\_valid\_novel} in 3,600 of 3,600 published TEST rows, and the identical 8,640 % SRC: w15-coverage-control/FINDINGS.md
combination rows. Validation, the one measured leg (60 combinations of 24 rows, 3 seeds), measures it % SRC: w15-coverage-control/FINDINGS.md
directly: $n=6$ at $n=3$'s coverage equals $n=3$ to every printed decimal on every seed, 22.767 % SRC: w15-coverage-control/FINDINGS.md
(21.767 / 23.050 / 23.483), against $n=6$'s 23.478 (22.967 / 23.617 / 23.850); the coverage % SRC: w15-coverage-control/FINDINGS.md
share of the rise is 1.00 on every seed and the mask channel is 0.000. The rise itself, % SRC: w15-coverage-control/FINDINGS.md
+1.200 / +0.567 / +0.367, is below the pre-declared 1.0 margin on two of three seeds, so the % SRC: w15-coverage-control/FINDINGS.md
declared validation verdict is \textsc{no rise to decompose on val}, and we report it as
that; validation sits near its 24-tuple ceiling. The inference note travels with % SRC: w15-coverage-control/FINDINGS.md""",
r"""full acceptance under either rung's bitmap, \texttt{distinct\_valid} equal to % SRC: w15-coverage-control/FINDINGS.md
\texttt{distinct\_valid\_novel} in every published TEST row, and the identical % SRC: w15-coverage-control/FINDINGS.md
combination rows. Validation, the one measured leg, measures it % SRC: w15-coverage-control/FINDINGS.md
directly: $n=6$ at $n=3$'s coverage equals $n=3$ to every printed decimal on every seed % SRC: w15-coverage-control/FINDINGS.md
(Tab.~\ref{tab:ladder}, caption); the coverage % SRC: w15-coverage-control/FINDINGS.md
share of the rise is complete on every seed and the mask channel contributes nothing. The rise itself % SRC: w15-coverage-control/FINDINGS.md
is below the pre-declared margin on two of three seeds, so the % SRC: w15-coverage-control/FINDINGS.md
declared validation verdict is \textsc{no rise to decompose on val}, and we report it as
that; validation sits near its ceiling. The inference note travels with % SRC: w15-coverage-control/FINDINGS.md
% PW2-5 moved -> Tab. ladder (3-seed means 61.614 and 66.603, per-seed cells; notes ADDED: validation leg, TEST identity check): 61.614; 66.603; Kstar=32; 3 seeds; +4.989; +5.758 / +3.467 / +5.742; 1.000000; 3,600 of 3,600; 8,640; 60 combinations of 24 rows; 22.767 (21.767 / 23.050 / 23.483); 23.478 (22.967 / 23.617 / 23.850); 1.00; 0.000; +1.200 / +0.567 / +0.367; 1.0 margin; 24-tuple ceiling; SRC: w15-coverage-control/FINDINGS.md""")],
[('61.614; 66.603', 'Tab. ladder (n-CLU 3-seed mean row, n=3 and n=6)', 'w12b-nr6-and-kstar/KSTAR-LADDER.md'),
 ('+4.989; +5.758 / +3.467 / +5.742', 'derivable from Tab. ladder (n=6 minus n=3 per seed); pointer comment', 'w15-coverage-control/FINDINGS.md'),
 ('1.000000; 3,600 of 3,600; 8,640; 60 combinations of 24 rows; 22.767 (21.767 / 23.050 / 23.483); 23.478 (22.967 / 23.617 / 23.850); 1.00; 0.000; +1.200 / +0.567 / +0.367', 'Tab. ladder caption, Notes (ADDED)', 'w15-coverage-control/FINDINGS.md'),
 ('1.0 margin; 24-tuple ceiling', 'Methods (1.0 margin); Tab. ladder notes (24 rows)', 'w15-coverage-control/FINDINGS.md')])

add('R5', 'Results', [('both',
r"""book matches a book refit without them: over every scope and seed the content's largest mode
deviation is at most \sci{5.38}{-07}, the book's statistic deviates by 0 and its rendered features by
at most \sci{7.81}{-08} (relative); the frozen encoder and decoder are outside the object. Check (d)
holds as well: the downdated object and the object refit without the rows compose identically on every
combination of all nine legs.""",
r"""book matches a book refit without them: over every scope and seed the content's largest mode
deviation and the relative deviation of the book's rendered features are at floating-point roundoff, and the book's statistic does not deviate
at all (Tab.~\ref{tab:del}); the frozen encoder and decoder are outside the object. Check (d)
holds as well: the downdated object and the object refit without the rows compose identically on every
combination of every leg.
% PW2-5 moved -> Tab. del columns (a), (b), (d): 5.38e-07; 0; 7.81e-08; nine legs; SRC: w20-shapes-data-matched/SHAPES-MATCHED.md"""),
('both',
r"""The deletion reads no training row: the n-CLU's downdate reads 0 data rows, and the % SRC: w20-shapes-data-matched/SHAPES-MATCHED.json
book's refit reads no pool row but reads its retained statistic, 967,304,800 B (0.97 GB). The book sees % SRC: w20-shapes-data-matched/SHAPES-MATCHED.json; w20-shapes-data-matched/FINDINGS.md
data only through per-tuple aggregates and every Shapes3D tuple is one row, so its statistic is the
per-row first moment of its 117,964 fit rows, the fit data itself; and its refit takes 2,367 to % SRC: w20-shapes-data-matched/FINDINGS.md
5,913 s, about as long as refitting the book without the rows. ``Reads no training row'' is % SRC: w20-shapes-data-matched/FINDINGS.md""",
r"""The deletion reads no training row: the n-CLU's downdate reads no data rows, and the % SRC: w20-shapes-data-matched/SHAPES-MATCHED.json
book's refit reads no pool row but reads its retained statistic, nearly a gigabyte (Tab.~\ref{tab:del}). The book sees % SRC: w20-shapes-data-matched/SHAPES-MATCHED.json; w20-shapes-data-matched/FINDINGS.md
data only through per-tuple aggregates and every Shapes3D tuple is one row, so its statistic is the
per-row first moment of its fit rows, the fit data itself; and its refit (Tab.~\ref{tab:del}) takes % SRC: w20-shapes-data-matched/FINDINGS.md
about as long as refitting the book without the rows. ``Reads no training row'' is % SRC: w20-shapes-data-matched/FINDINGS.md
% PW2-5 moved -> Tab. del (cost rows: 0 rows read, 967,304,800 B, refit 2,367.3 to 5,912.9 s; caption ADDED: 117,964 fit rows): 0 data rows; 967,304,800 B (0.97 GB); 117,964; 2,367 to 5,913 s; SRC: w20-shapes-data-matched/SHAPES-MATCHED.json; w20-shapes-data-matched/FINDINGS.md"""),
('both',
r"""measured limitation of the unit. Deleting the red sphere lowers composition on the surviving combinations
by 6.13 / 6.43 / 6.30 tuples per held-out combination, on all three seeds. At seed 0 the loss is concentrated where % SRC: w20-shapes-data-matched/FINDINGS.md
the deleted rows tied red to a shape: the 20 combinations that request red with another shape fall by % SRC: w20-shapes-data-matched/FINDINGS.md
32.1 each, the 36 combinations that request a sphere in another colour by 1.47, and the other 53 move % SRC: w20-shapes-data-matched/FINDINGS.md
by +0.51. The same pipeline deleting a random 1\% of items moves the survivors by $-0.02$ / % SRC: w20-shapes-data-matched/FINDINGS.md
$-0.11$ / $-0.01$, so the loss is the deleted content, not instability of the refit; and since the % SRC: w20-shapes-data-matched/FINDINGS.md""",
r"""measured limitation of the unit. Deleting the red sphere lowers composition on the surviving combinations
by about six tuples per held-out combination, on all three seeds (Tab.~\ref{tab:del}). At seed 0 the loss is concentrated where % SRC: w20-shapes-data-matched/FINDINGS.md
the deleted rows tied red to a shape: the combinations that request red with another shape fall by % SRC: w20-shapes-data-matched/FINDINGS.md
about thirty each, those that request a sphere in another colour by about one, and the others barely move % SRC: w20-shapes-data-matched/FINDINGS.md
(Tab.~\ref{tab:del}, caption). The same pipeline deleting a random 1\% of items moves the survivors by at most about a tenth of a tuple, % SRC: w20-shapes-data-matched/FINDINGS.md
so the loss is the deleted content, not instability of the refit; and since the % SRC: w20-shapes-data-matched/FINDINGS.md
% PW2-5 moved -> Tab. del (column (d): -6.128 / -6.431 / -6.303, item -0.018 / -0.110 / -0.009; caption ADDED: seed-0 breakdown 20 / 36 / 53 combinations): 6.13 / 6.43 / 6.30; 20; 32.1; 36; 1.47; 53; +0.51; -0.02 / -0.11 / -0.01; SRC: w20-shapes-data-matched/FINDINGS.md"""),
('both',
r"""not delete from the book at all. A value deletion moves the survivors by $-0.06$ / $+0.33$ / % SRC: w20-shapes-data-matched/FINDINGS.md
$-0.61$, within the declared one-tuple margin. % SRC: w20-shapes-data-matched/FINDINGS.md""",
r"""not delete from the book at all. A value deletion moves the survivors % SRC: w20-shapes-data-matched/FINDINGS.md
by less than the declared one-tuple margin (Tab.~\ref{tab:del}). % SRC: w20-shapes-data-matched/FINDINGS.md
% PW2-5 moved -> Tab. del (column (d), value: -0.056 / +0.326 / -0.607): -0.06 / +0.33 / -0.61; SRC: w20-shapes-data-matched/FINDINGS.md""")],
[('5.38e-07; 0; 7.81e-08', 'Tab. del columns (a), (b)', 'w20-shapes-data-matched/SHAPES-MATCHED.md'),
 ('nine legs', 'Tab. del (3 scopes × 3 seeds)', 'w20-shapes-data-matched/SHAPES-MATCHED.md'),
 ('0 data rows; 967,304,800 B (0.97 GB); 2,367 to 5,913 s', 'Tab. del cost rows', 'w20-shapes-data-matched/SHAPES-MATCHED.json'),
 ('117,964 fit rows', 'Tab. del caption (ADDED)', 'w20-shapes-data-matched/FINDINGS.md'),
 ('6.13 / 6.43 / 6.30; −0.02 / −0.11 / −0.01; −0.06 / +0.33 / −0.61', 'Tab. del column (d) (−6.128 / −6.431 / −6.303; −0.018 / −0.110 / −0.009; −0.056 / +0.326 / −0.607)', 'w20-shapes-data-matched/SHAPES-MATCHED.md'),
 ('20 combinations, 32.1; 36, 1.47; 53, +0.51', 'Tab. del caption (ADDED)', 'w20-shapes-data-matched/FINDINGS.md')])

PROBE_PTR = r"""% PW2-5 moved -> Tab. del caption (ADDED: probe and K=5 targeting): 0.610083; 720; 0.614277; 4,968; 6.194e-08; 6.163e-08; 1,015,808 of 1,060,864; 0 data rows; K=5 rank/width 0.156; cos_min 0.1999 against 0.9976; SRC: w13-seeds-and-tables/TABLES.md; w13-draft-revise/DRAFT-v1.md"""
add('R6', 'Results', [('v2',
r"""The n-CLU has changed: without the address-book carry, the read moves 0.610083 relative RMS on
the 720 red-requesting requests; but it moves 0.614277 on the 4,968 requests that do not ask
for red, so the basis change is generic, not targeted. With the carry the read moves
\sci{6.194}{-08} and \sci{6.163}{-08} respectively. The carry rewrites 1,015,808 of the
address book's 1,060,864 bytes and touches 0 data rows. % SRC: w13-seeds-and-tables/TABLES.md""",
r"""The n-CLU has changed: without the address-book carry, the read moves by about 0.6 relative RMS on
the requests that ask for red; but it moves as much on the requests that do not ask
for red, so the basis change is generic, not targeted. With the carry the read moves
only at roundoff. The carry rewrites nearly all of the
address book's bytes and touches no data rows. % SRC: w13-seeds-and-tables/TABLES.md"""),
('style',
r"""The n-CLU itself has changed. Without the address-book carry, the read moves by 0.610083
relative RMS on the 720 red-requesting requests, but also by 0.614277 on the 4,968 requests
that do not ask for red; the basis change is therefore generic, not targeted. With the carry,
the read moves by \sci{6.194}{-08} and \sci{6.163}{-08}, respectively. The carry rewrites
1,015,808 of the address book's 1,060,864 bytes and touches 0 data rows. % SRC: w13-seeds-and-tables/TABLES.md""",
r"""The n-CLU itself has changed. Without the address-book carry, the read moves by about 0.6
relative RMS on the requests that ask for red, but also by as much on the requests
that do not ask for red; the basis change is therefore generic, not targeted. With the carry,
the read moves only at roundoff. The carry rewrites
nearly all of the address book's bytes and touches no data rows. % SRC: w13-seeds-and-tables/TABLES.md"""),
('v2',
r"""rather than vanished: at $K=5$, rank/width 0.156, the same red deletion was concentrated,
$\cos_{\min}$ 0.1999 against 0.9976 for a size-matched random deletion of the same number of
rows. % SRC: w13-seeds-and-tables/TABLES.md; w13-draft-revise/DRAFT-v1.md""",
r"""rather than vanished: at $K=5$ the same red deletion was concentrated,
with a far lower minimum cosine than a size-matched random deletion of the same number of
rows (Tab.~\ref{tab:del}, caption). % SRC: w13-seeds-and-tables/TABLES.md; w13-draft-revise/DRAFT-v1.md
""" + PROBE_PTR),
('style',
r"""rather than vanished: at $K=5$ (rank/width 0.156), the same red deletion was concentrated, with
$\cos_{\min}$ 0.1999 against 0.9976 for a size-matched random deletion of the same number of
rows. % SRC: w13-seeds-and-tables/TABLES.md; w13-draft-revise/DRAFT-v1.md""",
r"""rather than vanished: at $K=5$ the same red deletion was concentrated, with
a far lower minimum cosine than a size-matched random deletion of the same number of
rows (Tab.~\ref{tab:del}, caption). % SRC: w13-seeds-and-tables/TABLES.md; w13-draft-revise/DRAFT-v1.md
""" + PROBE_PTR)],
[('0.610083 (720 requests); 0.614277 (4,968 requests); 6.194e-08; 6.163e-08; 1,015,808 of 1,060,864 bytes; 0 data rows', 'Tab. del caption (ADDED); 720 and 4,968 also in Tab. del (c) rows; 1,060,864 in Tab. ledger-kstar', 'w13-seeds-and-tables/TABLES.md'),
 ('K=5 rank/width 0.156; cos_min 0.1999 against 0.9976', 'Tab. del caption (ADDED)', 'w13-seeds-and-tables/TABLES.md; w13-draft-revise/DRAFT-v1.md')])

add('R7', 'Results', [('both',
r"""that route for every data-matched baseline family on the same deleted rows of the same 131,072-row % SRC: w20-shapes-data-matched/FINDINGS.md
pool, at the baseline's selected schedule, three seeds (Tabs.~\ref{tab:rivals-del}
and~\ref{tab:rivals-del-seeds}, App.~\ref{app:delbase}). Each retrain reads 120,028 to 129,761 % SRC: w20-shapes-data-matched/SHAPES-MATCHED.md
rows and takes 547.7 to 2,064.5 s. After the value and item deletions the n-CLU stays ahead of every % SRC: w20-shapes-data-matched/SHAPES-MATCHED.md
retrained baseline on every seed, by +6.954 to +32.607 tuples per held-out combination with every 95\% interval above % SRC: w20-shapes-data-matched/SHAPES-MATCHED.md
zero. After the red-sphere deletion it is ahead on the mean at every seed, but its interval crosses
zero against the MLP at seeds 1 and 2 (+2.486 and +2.596) and against the LSTM at seed 2 % SRC: w20-shapes-data-matched/SHAPES-MATCHED.md
(+2.477): the n-CLU's own loss on that deletion, about six tuples per held-out combination, is larger than the % SRC: w20-shapes-data-matched/SHAPES-MATCHED.md
change any retrained baseline shows except the continuous MLP at seed 2 (26.844 to 17.275), so on % SRC: w20-shapes-data-matched/SHAPES-MATCHED.md
collateral damage the baselines' crude retrain is cleaner than the n-CLU's deletion. Nor is the
n-CLU's route faster end to end: its downdate takes seconds, but the book's refit takes 2,367 to % SRC: w20-shapes-data-matched/FINDINGS.md
5,913 s against the baselines' 547.7 to 2,064.5 s. What the n-CLU's route has that the retrain does not % SRC: w20-shapes-data-matched/FINDINGS.md; w20-shapes-data-matched/SHAPES-MATCHED.md""",
r"""that route for every data-matched baseline family on the same deleted rows of the same % SRC: w20-shapes-data-matched/FINDINGS.md
pool, at the baseline's selected schedule, three seeds (Tabs.~\ref{tab:rivals-del}
and~\ref{tab:rivals-del-seeds}, App.~\ref{app:delbase}), which print each retrain's rows read % SRC: w20-shapes-data-matched/SHAPES-MATCHED.md
and time. After the value and item deletions the n-CLU stays ahead of every % SRC: w20-shapes-data-matched/SHAPES-MATCHED.md
retrained baseline on every seed, with every 95\% interval above % SRC: w20-shapes-data-matched/SHAPES-MATCHED.md
zero. After the red-sphere deletion it is ahead on the mean at every seed, but its interval crosses
zero against the MLP at seeds 1 and 2 and against the LSTM at seed 2: % SRC: w20-shapes-data-matched/SHAPES-MATCHED.md
the n-CLU's own loss on that deletion, about six tuples per held-out combination, is larger than the % SRC: w20-shapes-data-matched/SHAPES-MATCHED.md
change any retrained baseline shows except the continuous MLP at seed 2, so on % SRC: w20-shapes-data-matched/SHAPES-MATCHED.md
collateral damage the baselines' crude retrain is cleaner than the n-CLU's deletion. Nor is the
n-CLU's route faster end to end: its downdate takes seconds, but the book's refit takes longer % SRC: w20-shapes-data-matched/FINDINGS.md
than any baseline's retrain (Tabs.~\ref{tab:del} and~\ref{tab:rivals-del}). What the n-CLU's route has that the retrain does not % SRC: w20-shapes-data-matched/FINDINGS.md; w20-shapes-data-matched/SHAPES-MATCHED.md
% PW2-5 moved -> Tab. rivals-del (rows read, seconds), Tab. rivals-del-seeds (margins, intervals, 26.844 -> 17.275), Tab. del (book refit seconds), Tab. t7 caption (131,072): 131,072; 120,028 to 129,761; 547.7 to 2,064.5 s; +6.954 to +32.607; +2.486; +2.596; +2.477; 26.844 to 17.275; 2,367 to 5,913 s; SRC: w20-shapes-data-matched/SHAPES-MATCHED.md; w20-shapes-data-matched/FINDINGS.md""")],
[('131,072-row pool', 'Tab. rivals-del caption', 'w20-shapes-data-matched/SHAPES-MATCHED.md'),
 ('120,028 to 129,761 rows; 547.7 to 2,064.5 s', 'Tab. rivals-del', 'w20-shapes-data-matched/SHAPES-MATCHED.md'),
 ('+6.954 to +32.607; +2.486; +2.596; +2.477; 26.844 to 17.275', 'Tab. rivals-del-seeds', 'w20-shapes-data-matched/SHAPES-MATCHED.md'),
 ('2,367 to 5,913 s', 'Tab. del cost row', 'w20-shapes-data-matched/FINDINGS.md')])

HOLE_PTR = r"""% PW2-5 moved -> Tab. hole (ours: 2 bytes, 0 rows read; caption ADDED: 0.0000 s, 1,322-byte table, 1,322 pair slots), Tab. del (c) (1 B of the 1,322 B table), Tab. del caption (131,072): 2 bytes; 0 data rows; 0.0000 s; 1,322-byte; 1,322 pair slots; 131,072; SRC: w13-seeds-and-tables/TABLES.md; w20-shapes-data-matched/SHAPES-MATCHED.json"""
add('R8', 'Results', [('v2',
r"""$\Kstar$ object two holes cost 2 bytes, read 0 data rows, take 0.0000 s and refit nothing, in
a 1,322-byte coupling table whose layout is asserted equal to the field-mode n-CLU's over 1,322 pair
slots. % SRC: w13-seeds-and-tables/TABLES.md""",
r"""$\Kstar$ object two holes cost a byte each, read no data rows, take no measurable time and refit nothing, in
a coupling table whose layout is asserted equal to the field-mode n-CLU's over every pair
slot (Tabs.~\ref{tab:del} and~\ref{tab:hole}). % SRC: w13-seeds-and-tables/TABLES.md
""" + HOLE_PTR),
('style',
r"""$\Kstar$ object, two holes cost 2 bytes, read 0 data rows, take 0.0000 s and refit nothing, in
a 1,322-byte coupling table whose layout is asserted equal to the field-mode n-CLU's over 1,322 pair
slots. % SRC: w13-seeds-and-tables/TABLES.md""",
r"""$\Kstar$ object, two holes cost a byte each, read no data rows, take no measurable time and refit nothing, in
a coupling table whose layout is asserted equal to the field-mode n-CLU's over every pair
slot (Tabs.~\ref{tab:del} and~\ref{tab:hole}). % SRC: w13-seeds-and-tables/TABLES.md
""" + HOLE_PTR),
('both',
r"""on the n-CLU's own 131,072-row pool, deleting a random""",
r"""on the n-CLU's own pool, deleting a random""")],
[('2 bytes; 0 data rows', 'Tab. hole (row "ours: 2 holes": bytes 2, rows 0)', 'w13-seeds-and-tables/TABLES.md'),
 ('0.0000 s; 1,322-byte table; 1,322 pair slots', 'Tab. hole caption (ADDED); 1,322 B also in Tab. del (c)', 'w13-seeds-and-tables/TABLES.md'),
 ('131,072-row pool', 'Tab. del caption', 'w20-shapes-data-matched/SHAPES-MATCHED.json')])

CMP_PTR = r"""% PW2-5 moved -> Tab. hole (AUROC 1.0000, 4,096 rows, 521,916 bytes; caption: 130,479 parameters; caption ADDED: 8-byte bitmap at 1.0000 with 0 rows, 10.48 s): 8-byte; 1.0000; 0 rows; 130,479; 4,096; 521,916; 10.48 s; SRC: w12-deletion-baselines/FINDINGS.md"""
EVICT_PTR = r"""% PW2-5 moved -> Tab. ablations caption (ADDED: eviction and hole on the earlier item memory): 1.0000; -2.0899 / -0.9101 / -6.7416; +-1.0 bar; +0.0000; SRC: w13-seeds-and-tables/TABLES.md"""
add('R9', 'Results', [('v2',
r"""deletion the task is a one-token lookup, and a free 8-byte value-presence bitmap already scores
it at 1.0000 with 0 rows read; for a conjunction the free bitmap cannot express the question at
all, and the transformer learns the conjunction hole at AUROC 1.0000 for 130,479 parameters,
4,096 data rows read, 521,916 bytes rewritten and 10.48 s, where the coupling hole costs one
byte and no row. % SRC: w12-deletion-baselines/FINDINGS.md""",
r"""deletion the task is a one-token lookup, and a free value-presence bitmap of a few bytes already scores
it perfectly with no rows read; for a conjunction the free bitmap cannot express the question at
all, and the transformer learns the conjunction hole perfectly, at the parameter, row, byte and time
costs of Tab.~\ref{tab:hole}, where the coupling hole costs one
byte and no row. % SRC: w12-deletion-baselines/FINDINGS.md
""" + CMP_PTR),
('style',
r"""deletion the task is a one-token lookup, which a free 8-byte value-presence bitmap already scores
at 1.0000 with 0 rows read. For a conjunction the free bitmap cannot express the question at
all, and the transformer learns the conjunction hole at AUROC 1.0000 for 130,479 parameters,
4,096 data rows read, 521,916 bytes rewritten and 10.48 s, whereas the coupling hole costs one
byte and no row. % SRC: w12-deletion-baselines/FINDINGS.md""",
r"""deletion the task is a one-token lookup, which a free value-presence bitmap of a few bytes already scores
perfectly with no rows read. For a conjunction the free bitmap cannot express the question at
all, and the transformer learns the conjunction hole perfectly, at the parameter, row, byte and time
costs of Tab.~\ref{tab:hole}, whereas the coupling hole costs one
byte and no row. % SRC: w12-deletion-baselines/FINDINGS.md
""" + CMP_PTR),
('v2',
r"""hole, not deletion in general. On the earlier item memory, evicting red refuses at 1.0000 on all three
seeds and changes composition by $-2.0899$ / $-0.9101$ / $-6.7416$ against a declared $\pm1.0$ bar,
while punching a hole refuses at 1.0000 and changes it by $+0.0000$. % SRC: w13-seeds-and-tables/TABLES.md""",
r"""hole, not deletion in general. On the earlier item memory, evicting red refuses every red request on all three
seeds and changes composition beyond a declared one-tuple bar on two of them,
while punching a hole refuses every red request and leaves composition unchanged (Tab.~\ref{tab:ablations}, caption). % SRC: w13-seeds-and-tables/TABLES.md
""" + EVICT_PTR),
('style',
r"""leaves behaviour unchanged. On the earlier item memory, evicting red refuses at 1.0000 on all three
seeds but changes composition by $-2.0899$ / $-0.9101$ / $-6.7416$ against a declared $\pm1.0$ bar,
while punching a hole refuses at 1.0000 with a change of $+0.0000$. % SRC: w13-seeds-and-tables/TABLES.md""",
r"""leaves behaviour unchanged. On the earlier item memory, evicting red refuses every red request on all three
seeds but changes composition beyond a declared one-tuple bar on two of them,
while punching a hole refuses every red request with no change (Tab.~\ref{tab:ablations}, caption). % SRC: w13-seeds-and-tables/TABLES.md
""" + EVICT_PTR)],
[('8-byte bitmap; 1.0000; 0 rows; 10.48 s', 'Tab. hole caption (ADDED)', 'w12-deletion-baselines/FINDINGS.md'),
 ('AUROC 1.0000; 130,479 parameters; 4,096 rows; 521,916 bytes', 'Tab. hole (rows and caption)', 'w12-deletion-baselines/FINDINGS.md'),
 ('refusal 1.0000; −2.0899 / −0.9101 / −6.7416; ±1.0 bar; +0.0000', 'Tab. ablations caption (ADDED)', 'w13-seeds-and-tables/TABLES.md')])

ABL_PTR = r"""% PW2-5 moved -> Tab. ablations caption (ADDED: 712,072 to 719,352 B, 1.358 to 1.372 x) and App. w5 (713,632 / 719,352 / 719,352 against 715,808 B; parameters 178,408 / 179,838 / 179,838 against 178,952): 712,072 to 719,352 bytes; 1.358 to 1.372; +-0.5%; 713,632 / 719,352 / 719,352; 715,808; -0.304% to +0.495%; SRC: w13-seeds-and-tables/TABLES.md; w12-composition-ladder/FINDINGS.md"""
IM_PTR = r"""% PW2-5 moved -> Tab. ablations (row 4: item memory 32.3878 / 39.1224 / 41.7755 against learned reference 53.4694 / 55.6531 / 59.8571): -21.0816 / -16.5306 / -18.0816; 18.56; SRC: w13-seeds-and-tables/TABLES.md"""
add('R10', 'Results', [('v2',
r"""seed, at 712,072 to 719,352 bytes, 1.358 to 1.372 times the reference.
% SRC: w13-seeds-and-tables/TABLES.md; w12-composition-ladder/FINDINGS.md""",
r"""seed, at just under 1.4 times the reference in bytes (Tab.~\ref{tab:ablations}, caption).
% SRC: w13-seeds-and-tables/TABLES.md; w12-composition-ladder/FINDINGS.md"""),
('style',
r"""At the $K=5$ endpoint of the ladder it is bit-identical to the single design at every rung and seed, at 712,072 to 719,352 bytes, 1.358 to 1.372 times the reference.
% SRC: w13-seeds-and-tables/TABLES.md; w12-composition-ladder/FINDINGS.md""",
r"""At the $K=5$ endpoint of the ladder it is bit-identical to the single design at every rung and seed, at just under 1.4 times the reference in bytes (Tab.~\ref{tab:ablations}, caption).
% SRC: w13-seeds-and-tables/TABLES.md; w12-composition-ladder/FINDINGS.md"""),
('v2',
r"""learner on both rows beyond a margin declared in advance, at total deployed bytes matched to
within $\pm0.5$\% (713,632 / 719,352 / 719,352 against the baseline's 715,808) and parameters within
$-0.304$\% to $+0.495$\%, the arm carrying more parameters on two of three seeds.
% SRC: w13-seeds-and-tables/TABLES.md""",
r"""learner on both rows beyond a margin declared in advance, at total deployed bytes and parameters each matched to
within half a percent (App.~\ref{app:w5}), the arm carrying more parameters on two of three seeds.
% SRC: w13-seeds-and-tables/TABLES.md
""" + ABL_PTR),
('style',
r"""learner on both rows beyond a margin declared in advance, at total deployed bytes matched to within $\pm0.5$\% (713,632 / 719,352 / 719,352 against the baseline's 715,808) and parameters within $-0.304$\% to $+0.495$\%, the arm carrying more parameters on two of three seeds.
% SRC: w13-seeds-and-tables/TABLES.md""",
r"""learner on both rows beyond a margin declared in advance, at total deployed bytes and parameters each matched to within half a percent (App.~\ref{app:w5}), the arm carrying more parameters on two of three seeds.
% SRC: w13-seeds-and-tables/TABLES.md
""" + ABL_PTR),
('v2',
r"""composes poorly: on row 4 it sits $-21.0816$ / $-16.5306$ / $-18.0816$ below the learned
reference, a 3-seed mean of 18.56 tuples per held-out combination that is a lower bound on what an item memory
loses, because it is a ceiling and not an arm. % SRC: w13-seeds-and-tables/TABLES.md""",
r"""composes poorly: on row 4 it sits below the learned
reference by more than eighteen tuples per held-out combination on the 3-seed mean (Tab.~\ref{tab:ablations}), a lower bound on what an item memory
loses, because it is a ceiling and not an arm. % SRC: w13-seeds-and-tables/TABLES.md
""" + IM_PTR),
('style',
r"""composes poorly: on row 4 it sits $-21.0816$ / $-16.5306$ / $-18.0816$ below the learned
reference. This 3-seed mean of 18.56 tuples per held-out combination is a lower bound on what an item memory
loses, because this n-CLU is a ceiling and not an arm. % SRC: w13-seeds-and-tables/TABLES.md""",
r"""composes poorly: on row 4 it sits below the learned
reference by more than eighteen tuples per held-out combination on the 3-seed mean (Tab.~\ref{tab:ablations}). This is a lower bound on what an item memory
loses, because this n-CLU is a ceiling and not an arm. % SRC: w13-seeds-and-tables/TABLES.md
""" + IM_PTR)],
[('712,072 to 719,352 bytes; 1.358 to 1.372', 'Tab. ablations caption (ADDED)', 'w13-seeds-and-tables/TABLES.md; w12-composition-ladder/FINDINGS.md'),
 ('±0.5%; 713,632 / 719,352 / 719,352; 715,808; −0.304% to +0.495%', 'App. w5 prose (713,632 / 719,352 / 719,352 against 715,808 B; 99.70% / 100.50%; parameters 178,408 / 179,838 / 179,838 against 178,952)', 'w13-seeds-and-tables/TABLES.md'),
 ('−21.0816 / −16.5306 / −18.0816; 18.56', 'Tab. ablations row 4 (item memory minus learned reference, per seed)', 'w13-seeds-and-tables/TABLES.md')])

add('R11', 'Results', [('both',
r"""anchored full unfreeze ($+0.5688\to+1.3303$, $-0.0275\to-0.0092$, $+1.1009\to+1.3028$) and on two
of three under a low-rank adapter. % SRC: w13-encoder-adapt/CROSS-ARM.md""",
r"""anchored full unfreeze and on two
of three under a low-rank adapter (Tab.~\ref{tab:adapt}). % SRC: w13-encoder-adapt/CROSS-ARM.md
% PW2-5 moved -> Tab. adapt (margin column): +0.5688 -> +1.3303; -0.0275 -> -0.0092; +1.1009 -> +1.3028; SRC: w13-encoder-adapt/CROSS-ARM.md""")],
[('+0.5688→+1.3303, −0.0275→−0.0092, +1.1009→+1.3028', 'Tab. adapt (margin column)', 'w13-encoder-adapt/CROSS-ARM.md')])

add('R12', 'Results', [('both',
r"""settings in this feature space, score at most 0.267 and 0.020 distinct valid novel tuples per held-out combination % SRC: w15-lace-poe-rows/FINDINGS.md""",
r"""settings in this feature space, score far below one distinct valid novel tuple per held-out combination % SRC: w15-lace-poe-rows/FINDINGS.md
% PW2-5 moved -> Tab. lace-poe (LACE row 3 seed 2 0.267; PoE row 4 seed 0 0.020): 0.267; 0.020; SRC: w15-lace-poe-rows/FINDINGS.md"""),
('both',
r"""The generation table is a second Shapes3D instrument, never merged with Tab.~\ref{tab:t7}: 72 % SRC: w15-lace-poe-rows/FINDINGS.md
held-out requests per seed,""",
r"""The generation table is a second Shapes3D instrument, never merged with Tab.~\ref{tab:t7}: a fixed set of % SRC: w15-lace-poe-rows/FINDINGS.md
held-out requests per seed,"""),
('both',
r"""n-CLU's read (every seed, margin 0.020), \textsc{fails} on seed 0 ($-0.0185$; mean $+0.1551$), and the % SRC: w15-lace-poe-rows/FINDINGS.md""",
r"""n-CLU's read (every seed, declared margin), \textsc{fails} on seed 0 though its mean is positive, and the % SRC: w15-lace-poe-rows/FINDINGS.md"""),
('both',
r"""($-0.1150$, S3 \textsc{lose}) at 4.09--4.11$\times$ the reference bytes, and beats LACE % SRC: w15-lace-poe-rows/FINDINGS.md
($+0.4468$) and PoE ($+0.5532$) on the mean (S4). % SRC: w15-lace-poe-rows/FINDINGS.md""",
r"""(S3 \textsc{lose}) at about four times the reference bytes, and beats LACE % SRC: w15-lace-poe-rows/FINDINGS.md
and PoE on the mean (S4; Tab.~\ref{tab:gentable-kstar}). % SRC: w15-lace-poe-rows/FINDINGS.md
% PW2-5 moved -> Tab. gentable-kstar (caption: 72 requests; caption ADDED: S2 margin 0.020; S2 row -0.0185, +0.1551; S3 row -0.1150; 4.09--4.11x; LACE / PoE means 0.3032 / 0.1968 against 0.7500): 72; 0.020; -0.0185; +0.1551; -0.1150; 4.09--4.11; +0.4468; +0.5532; SRC: w15-lace-poe-rows/FINDINGS.md""")],
[('0.267; 0.020', 'Tab. lace-poe', 'w15-lace-poe-rows/FINDINGS.md'),
 ('72 requests', 'Tab. gentable-kstar caption', 'w15-lace-poe-rows/FINDINGS.md'),
 ('margin 0.020', 'Tab. gentable-kstar caption (ADDED to the S2 criterion)', 'w15-lace-poe-rows/FINDINGS.md'),
 ('−0.0185; +0.1551; −0.1150; 4.09–4.11×', 'Tab. gentable-kstar (S2 and S3 rows; bytes column)', 'w15-lace-poe-rows/FINDINGS.md'),
 ('+0.4468; +0.5532', 'derivable from Tab. gentable-kstar means (0.7500 − 0.3032; 0.7500 − 0.1968)', 'w15-lace-poe-rows/FINDINGS.md')])

add('R13', 'Results', [('both',
r"""The n-CLU here is the base design (latent 128, $K=2$, 6 atoms).""",
r"""The n-CLU here is the base design (Tab.~\ref{tab:cmazet}).""" ),
('both',
r"""to every arm the n-CLU is behind the best matched baseline on all three TEST seeds, by 0.0250 / 0.0013 /
0.0262; on main-effect words alone and on the observation alone it is behind on every seed. On words
plus observation, TEST, six seeds, it reads 0.800 / 0.566 / 0.808 / 0.674 / 0.533 / 0.682 against the
best matched baseline over both step counts, 0.627 / 0.567 / 0.784 / 0.412 / 0.531 / 0.867: three
clear wins (seeds 0, 2 and 3), two ties (seeds 1 and 4, $-0.001$ and $+0.001$) and
one loss (seed 5, $-0.185$), a six-seed mean of +0.046. Seed 3's lead is over a set of baselines weakened
by a matched MLP that did not converge at either step count, and on seed 5 the n-CLU scores 0.000 on two
held-out tuples that are fully supported in training and that eight other arms solve. This miss is""",
r"""to every arm the n-CLU is behind the best matched baseline on all three TEST seeds; on main-effect words alone and on the observation alone it is behind on every seed. On words
plus observation, TEST, six seeds, against the
best matched baseline over both step counts it has three
clear wins (seeds 0, 2 and 3), two ties (seeds 1 and 4) and
one loss (seed 5; Tab.~\ref{tab:cmazet-sym}). Seed 3's lead is over a set of baselines weakened
by a matched MLP that did not converge at either step count, and on seed 5 the n-CLU scores nothing on two
held-out tuples that are fully supported in training and that eight other arms solve. This miss is
% PW2-5 moved -> Tab. cmazet (caption: latent 128, K=2, 6 atoms), Tab. cmazet-sym (full word design and words + observation rows): latent 128; K=2; 6 atoms; 0.0250 / 0.0013 / 0.0262; 0.800 / 0.566 / 0.808 / 0.674 / 0.533 / 0.682; 0.627 / 0.567 / 0.784 / 0.412 / 0.531 / 0.867; -0.001; +0.001; -0.185; +0.046; 0.000; SRC: w21-maze-t-seed4/FINDINGS.md; w21-words-full-test/FINDINGS.md; w19-maze-t-symmetric/FINDINGS.md"""),
('both',
r"""side of a 0.04 margin in the deployed span and is already wrong on 106 of the 317 fit rows that carry it; % SRC: w23-seed5-diagnosis/FINDINGS.md; w23-seed5-diagnosis/Q1-CONTENT.json""",
r"""side of a small margin in the deployed span and is already wrong on about a third of the fit rows that carry it; % SRC: w23-seed5-diagnosis/FINDINGS.md; w23-seed5-diagnosis/Q1-CONTENT.json
% PW2-5 moved -> Tab. cmazet-sym (words + observation n-CLU row, note ADDED): 0.04 margin; 106 of 317; SRC: w23-seed5-diagnosis/FINDINGS.md; w23-seed5-diagnosis/Q1-CONTENT.json"""),
('both',
r"""control stays below the n-CLU on every seed (highest 0.495, seed 5).""",
r"""control stays below the n-CLU on every seed (Tab.~\ref{tab:cmazet-sym}).
% PW2-5 moved -> Tab. cmazet-sym (controls row, highest 0.495 at seed 5): 0.495; SRC: w21-maze-t-seed4/FINDINGS.md""")],
[('latent 128; K=2; 6 atoms', 'Tab. cmazet caption ("Base n-CLU: latent 128, $K=2$, 6 atoms")', 'w16-maze-wm/TABLES.md'),
 ('0.0250 / 0.0013 / 0.0262', 'Tab. cmazet-sym (full word design row)', 'w21-words-full-test/FINDINGS.md'),
 ('0.800 / … / 0.682; 0.627 / … / 0.867; −0.001; +0.001; −0.185; +0.046', 'Tab. cmazet-sym (words + observation, n-CLU row)', 'w21-maze-t-seed4/FINDINGS.md'),
 ('0.000 on two tuples', 'pointer (per-tuple replay in the SRC note already in the .tex); prose: "scores nothing"', 'w21-maze-t-seed4/FINDINGS.md'),
 ('0.04 margin; 106 of 317 fit rows', 'Tab. cmazet-sym, n-CLU words + observation row (note ADDED)', 'w23-seed5-diagnosis/FINDINGS.md; w23-seed5-diagnosis/Q1-CONTENT.json'),
 ('highest control 0.495 (seed 5)', 'Tab. cmazet-sym (controls row)', 'w21-maze-t-seed4/FINDINGS.md')])

add('R14', 'Results', [('both',
r"""(n-CLU-REL), lifts the n-CLU on every seed, to 0.908 / 0.894 / 0.984 / 0.927 / 0.900 / 0.939
against the best matched baseline's 0.803 / 0.567 / 0.685 / 0.723 / 0.624 / 0.939. The mechanism is the""",
r"""(n-CLU-REL), lifts the n-CLU on every seed (Tab.~\ref{tab:cmazet-sym}). The mechanism is the"""),
('both',
r"""Tab.~\ref{tab:cmazet-sym}) reach 0.915 / 0.869 / 0.819 / 0.921 / 0.900 / 0.939, so the n-CLU's lead
over them is $-0.007$ / $+0.024$ / $+0.165$ / $+0.006$ / $+0.000$ / $+0.000$, and on seeds 3 and 4 the
columns recover 0.97 and 1.00 of the n-CLU's margin over the baselines without them. On seed 4 a control
ties the n-CLU (random atoms, 0.9004, equal on every held-out tuple), so the n-CLU is not load-bearing
there; on seed 5 a matched baseline ties it (a transformer at 80,000 steps). The n-CLU's lead over the
relational baselines is sizeable only on seed 2 (+0.165). Relational coding has one limit: relations""",
r"""Tab.~\ref{tab:cmazet-sym}) come close to it, so the n-CLU's lead
over them is small or absent on every seed but one, and on seeds 3 and 4 the
columns recover all or nearly all of the n-CLU's margin over the baselines without them. On seed 4 a control
ties the n-CLU (random atoms, equal on every held-out tuple), so the n-CLU is not load-bearing
there; on seed 5 a matched baseline ties it (a transformer at $10\times$ training steps). The n-CLU's lead over the
relational baselines is sizeable only on seed 2. Relational coding has one limit: relations
% PW2-5 moved -> Tab. cmazet-sym (n-CLU-REL row: n-CLU, best matched baseline, +rel, highest control 0.900 at seed 4; seed 5: a matched transformer at 80,000 steps): 80,000 steps; 0.908 / 0.894 / 0.984 / 0.927 / 0.900 / 0.939; 0.803 / 0.567 / 0.685 / 0.723 / 0.624 / 0.939; 0.915 / 0.869 / 0.819 / 0.921 / 0.900 / 0.939; -0.007 / +0.024 / +0.165 / +0.006 / +0.000 / +0.000; 0.97; 1.00; 0.9004; +0.165; SRC: w21-rel-newseeds/FINDINGS.md; w21-maze-t-seed4/FINDINGS.md"""),
('both',
r"""loss: n-CLU-REL reads 0.554 against the best matched baseline's 0.797 (six-seed means) and is % SRC: w21-rel-newseeds/TABLES.md""",
r"""loss: n-CLU-REL stays well behind the best matched baseline (Tab.~\ref{tab:cmazet-sym}) and is % SRC: w21-rel-newseeds/TABLES.md
% PW2-5 moved -> Tab. cmazet-sym (value hold-out, n-CLU-REL row ADDED): 0.554; 0.797; six-seed means; SRC: w21-rel-newseeds/TABLES.md""")],
[('0.908 / … / 0.939; 0.803 / … / 0.939; 0.915 / … / 0.939', 'Tab. cmazet-sym (n-CLU-REL row)', 'w21-rel-newseeds/FINDINGS.md'),
 ('−0.007 / +0.024 / +0.165 / +0.006 / +0.000 / +0.000; +0.165', 'derivable from Tab. cmazet-sym (n-CLU-REL minus +rel per seed); pointer comment', 'w21-rel-newseeds/FINDINGS.md'),
 ('0.97; 1.00 (share of margin recovered)', 'pointer comment (derived; not printed in a table)', 'w21-rel-newseeds/FINDINGS.md'),
 ('0.9004 (random atoms, seed 4)', 'Tab. cmazet-sym (highest control 0.900, seed 4)', 'w21-rel-newseeds/FINDINGS.md'),
 ('0.554 against 0.797 (six-seed means)', 'Tab. cmazet-sym, value hold-out n-CLU-REL row (ADDED)', 'w21-rel-newseeds/TABLES.md')])

add('R15', 'Results', [('both',
r"""\paragraph{CMAZE-T: what is not claimed.} Chained rollouts are not valid: The base n-CLU's rollout validity
is 0.000 on all six seeds at a per-step accuracy of 0.900 to 0.930, and n-CLU-REL reaches 0.067 /
0.000 / 0.289 / 0.267 / 0.156 / 0.000, a limitation of every arm. On unseen two-way pairs a learned""",
r"""\paragraph{CMAZE-T: what is not claimed.} Chained rollouts are not valid: The base n-CLU produces no valid rollout
on any of the six seeds at a per-step accuracy of about nine in ten, and n-CLU-REL a minority on every seed
(Tab.~\ref{tab:cmazet-sym}), a limitation of every arm. On unseen two-way pairs a learned
% PW2-5 moved -> Tab. cmazet-sym (rollout validity row): 0.000; 0.900 to 0.930; 0.067 / 0.000 / 0.289 / 0.267 / 0.156 / 0.000; SRC: w21-maze-t-seed4/FINDINGS.md; w21-rel-newseeds/FINDINGS.md"""),
('both',
r"""deleted rows: the downdated statistic matches the statistic fitted without the rows to a relative
\sci{1.94}{-15} at worst, the deployed atoms do not change, and the downdated n-CLU reads identically
to the n-CLU fitted without the rows, in 0.09 to 0.88 s with no gradient step,
against 71 to 140 s and 8,000 gradient steps for a baseline's retrain-without.""",
r"""deleted rows: the downdated statistic matches the statistic fitted without the rows to floating-point
roundoff, the deployed atoms do not change, and the downdated n-CLU reads identically
to the n-CLU fitted without the rows, in under a second with no gradient step,
against a minute or more and thousands of gradient steps for a baseline's retrain-without (Tab.~\ref{tab:cmazet-sym}).
% PW2-5 moved -> Tab. cmazet-sym (deletion row): 1.94e-15; 0.09 to 0.88 s; 71 to 140 s; 8,000 steps; SRC: w21-maze-t-seed4/FINDINGS.md; w19-maze-t-symmetric/FINDINGS.md""")],
[('0.000; 0.900 to 0.930; 0.067 / 0.000 / 0.289 / 0.267 / 0.156 / 0.000', 'Tab. cmazet-sym (rollout validity row)', 'w21-maze-t-seed4/FINDINGS.md'),
 ('1.94e-15; 0.09 to 0.88 s; 71 to 140 s; 8,000 gradient steps', 'Tab. cmazet-sym (deletion row)', 'w21-maze-t-seed4/FINDINGS.md')])

# ------------------------------------------------------------------ DISCUSSION
add('D1', 'Discussion', [('both',
r"""seeds, returning the static read's held-out scores to every printed digit (row 4: 63.612 /
64.327 / 67.653; row 3: 29.717 / 31.617 / 31.450). % SRC: w12-clue-theorist/CLUE.md""",
r"""seeds, returning the static read's held-out scores to every printed digit. % SRC: w12-clue-theorist/CLUE.md
% PW2-5 moved -> pointer only (the same values print as the composition-only n-CLU row of Tab. ablations, 63.6122 / 64.3265 / 67.6531 and 29.7167 / 31.6167 / 31.4500; no table states the zero-step identity): row 4 63.612 / 64.327 / 67.653; row 3 29.717 / 31.617 / 31.450; SRC: w12-clue-theorist/CLUE.md"""),
('v2',
r"""strength at which they move the state: $-0.531$ / $-0.510$ / $-0.388$ six-factor tuples per
held-out combination at the gentlest setting that moves it (median displacement 0.009), and $-39.959$ /
$-46.918$ / $-39.061$ at full strength (median displacement 0.39), while incurring 81,312 to 84,216
bytes of read-side coupling that the static read does not. % SRC: w12-clue-theorist/CLUE.md""",
r"""strength at which they move the state: by less than one six-factor tuple per
held-out combination at the gentlest setting that moves it, and by about forty on the seed mean at full strength, while incurring
bytes of read-side coupling that the static read does not (App.~\ref{app:physics}). % SRC: w12-clue-theorist/CLUE.md
% PW2-5 moved -> App. physics prose ("The same dynamics lower composition": -0.531 / -0.510 / -0.388; -39.959 / -46.918 / -39.061; 81,312 B) and pointer (median displacement 0.009 and 0.39; 84,216): -0.531 / -0.510 / -0.388; 0.009; -39.959 / -46.918 / -39.061; 0.39; 81,312 to 84,216 bytes; SRC: w12-clue-theorist/CLUE.md"""),
('style',
r"""strength at which they move the state: $-0.531$ / $-0.510$ / $-0.388$ six-factor tuples per
held-out combination at the gentlest setting that moves it (median displacement 0.009), and $-39.959$ /
$-46.918$ / $-39.061$ at full strength (median displacement 0.39), while incurring 81,312 to 84,216 bytes of read-side coupling that the static read does not. % SRC: w12-clue-theorist/CLUE.md""",
r"""strength at which they move the state: by less than one six-factor tuple per
held-out combination at the gentlest setting that moves it, and by about forty on the seed mean at full strength, while incurring bytes of read-side coupling that the static read does not (App.~\ref{app:physics}). % SRC: w12-clue-theorist/CLUE.md
% PW2-5 moved -> App. physics prose ("The same dynamics lower composition": -0.531 / -0.510 / -0.388; -39.959 / -46.918 / -39.061; 81,312 B) and pointer (median displacement 0.009 and 0.39; 84,216): -0.531 / -0.510 / -0.388; 0.009; -39.959 / -46.918 / -39.061; 0.39; 81,312 to 84,216 bytes; SRC: w12-clue-theorist/CLUE.md""")],
[('row 4: 63.612 / 64.327 / 67.653; row 3: 29.717 / 31.617 / 31.450', 'pointer comment (the same values print in Tab. ablations, composition-only row; the zero-step identity itself is in no table)', 'w12-clue-theorist/CLUE.md'),
 ('−0.531 / −0.510 / −0.388; −39.959 / −46.918 / −39.061; 81,312', 'App. physics prose ("The same dynamics lower composition")', 'w10-physics-section-evidence/EVIDENCE.md'),
 ('median displacement 0.009 and 0.39; 84,216', 'pointer comment only', 'w12-clue-theorist/CLUE.md')])

LAW_PTR = r"""% PW2-5 moved -> Tab. crr (B1 0.0433 / 0.0435 / 0.0431 against 0.10; B2 49 of 49; 49 combinations x 3 seeds), App. laws prose (1.105; 2,054,866-byte image; 65.17% to 65.39%; 83.5%): 1.105; 49; 0.10; 0.0433 / 0.0435 / 0.0431; 49 of 49; 0 bytes; 2,054,866; 65.17% to 65.39%; 83.5%; SRC: w10-physics-section-evidence/EVIDENCE.md"""
add('D2', 'Discussion', [('v2',
r"""prediction is a \emph{time}, not a score: with nothing fitted, it predicts the settle half-life to
a median factor of 1.105 on 49 held-out combinations and three seeds (bar 0.10 on the log error; 0.0433
/ 0.0435 / 0.0431), and beats a one-parameter form fitted with sight of the held-out measurements
on 49 of 49 combinations on every seed; one of six declared criteria (the V-curve argmin) misses on
all three seeds, and the law predicts where its simplest form stops.""",
r"""prediction is a \emph{time}, not a score: with nothing fitted, it predicts the settle half-life
well within its declared bar on the held-out combinations and three seeds (Tab.~\ref{tab:crr}), and beats a one-parameter form fitted with sight of the held-out measurements
on every combination on every seed; one of six declared criteria (the V-curve argmin) misses on
all three seeds, and the law predicts where its simplest form stops."""),
('style',
r"""prediction is a \emph{time}, not a score. With nothing fitted, it predicts the settle half-life to
a median factor of 1.105 on 49 held-out combinations and three seeds (bar 0.10 on the log error; 0.0433
/ 0.0435 / 0.0431), and beats a one-parameter form fitted with sight of the held-out
measurements on 49 of 49 combinations on every seed.""",
r"""prediction is a \emph{time}, not a score. With nothing fitted, it predicts the settle half-life
well within its declared bar on the held-out combinations and three seeds (Tab.~\ref{tab:crr}), and beats a one-parameter form fitted with sight of the held-out
measurements on every combination on every seed."""),
('v2',
r"""Deletion commutes with scheduled decay on a learned memory, 0 bytes differing in a 2,054,866-byte
image, while the stronger claim that the post-delete memory equals the never-inserted one is
\textsc{refuted} (65.17\% to 65.39\% of the image differs, 83.5\% of it because the address book
and coupling table are fitted on the pool). % SRC: w10-physics-section-evidence/EVIDENCE.md""",
r"""Deletion commutes with scheduled decay on a learned memory, no byte differing in the memory
image, while the stronger claim that the post-delete memory equals the never-inserted one is
\textsc{refuted} (nearly two thirds of the image differs, most of it because the address book
and coupling table are fitted on the pool; App.~\ref{app:laws}). % SRC: w10-physics-section-evidence/EVIDENCE.md
""" + LAW_PTR),
('style',
r"""Deletion commutes with scheduled decay on a learned memory, 0 bytes differing in a
2,054,866-byte image. The stronger claim that the post-delete memory equals the never-inserted one is \textsc{refuted} (65.17\% to 65.39\% of the image differs, 83.5\% of it because the
address book and coupling table are fitted on the pool). % SRC: w10-physics-section-evidence/EVIDENCE.md""",
r"""Deletion commutes with scheduled decay on a learned memory, no byte differing in the
memory image. The stronger claim that the post-delete memory equals the never-inserted one is \textsc{refuted} (nearly two thirds of the image differs, most of it because the
address book and coupling table are fitted on the pool; App.~\ref{app:laws}). % SRC: w10-physics-section-evidence/EVIDENCE.md
""" + LAW_PTR)],
[('1.105; 49; bar 0.10; 0.0433 / 0.0435 / 0.0431; 49 of 49', 'Tab. crr (B1, B2 rows; caption 49 combinations × 3 seeds); 1.105 in App. laws prose', 'w10-physics-section-evidence/EVIDENCE.md'),
 ('0 bytes; 2,054,866-byte image; 65.17% to 65.39%; 83.5%', 'App. laws prose (2,054,866-byte image; 65.17% to 65.39%; 83.5%)', 'w10-physics-section-evidence/EVIDENCE.md')])

# ------------------------------------------------------------------ FLOATS (captions and tables; identical in both drafts)
add('T1', 'Tab. t7 caption', [('both',
r"""Tab.~\ref{tab:lace-poe}.}
\label{tab:t7}""",
r"""Tab.~\ref{tab:lace-poe}. The n-CLU's row-3 and row-4 means are 32.350~$\pm$~0.815 and 67.177~$\pm$~1.320 % SRC: w12b-nr6-and-kstar/NR6-T7.md
($\pm$ the sample standard deviation over the three seeds).}
\label{tab:t7}""")], [])

add('T2', 'Fig. shapes-per-model caption', [('both',
r"""of the target shape the draw prefers object, floor and wall hues not yet shown. The judge's six read-back
values for every panel are listed in the supplementary material.}""",
r"""of the target shape the draw prefers object, floor and wall hues not yet shown. The judge's six read-back
values for every panel are listed in the supplementary material. Of the 5,688 held-out requests at seed 0 the % SRC: ../../outputs/pw1-render-shapes/REQUESTS-pw1.json
judge marks valid: n-CLU 5,177, MLP 4,361, LSTM 3,703, Transformer 2,786, continuous MLP 3,110, lookup table 897. % SRC: ../../outputs/pw1-render-shapes/REQUESTS-pw1.json
Of the n-CLU's 511 invalid renders, 497 are wrong in one factor only: scale 329, orientation 125, shape 71.} % SRC: ../../outputs/pw1-render-shapes/factor-miss-seed0.txt""")], [])

add('T3', 'Tab. ladder caption', [('both',
r"""fit-pool coverage, not in composition depth.}
\label{tab:ladder}""",
r"""fit-pool coverage, not in composition depth. \emph{Notes.} Earlier set: 117,878 to 119,858 parameters, % SRC: _thinkpad/w4/spoke-a-rivals/rivals/seed_0/mlp.rival.json; _thinkpad/w4/spoke-a-rivals/REPORT.md
40 epochs at batch 256 and learning rate 0.003, the same for every family and seed. % SRC: _thinkpad/w4/spoke-a-rivals/REPORT.md; _thinkpad/w4/spoke-a-rivals/rivals/seed_0/mlp.json
Landscape controls at $\Kstar$: 0.0000 on 13 of the 15 rung--seed legs; landscape shuffled reads 0.0083 % SRC: w12b-nr6-and-kstar/FINDINGS.md
(one novel tuple over 120 combinations) at $n=3$, seeds 0 and 1. % SRC: w12b-nr6-and-kstar/FINDINGS.md
Rungs $n=3$--6 probe the same 8,640 rows; $n=2$ probes 60 combinations of 36 rows, $n\ge3$ 120 of 72, disjoint sets. % SRC: w12-composition-ladder/FINDINGS.md
Held-out row count at $n=3,4,5,6$: 300,000 / 235,200 / 213,600 / 200,640; every arm fitted on the rung's own % SRC: w12-composition-ladder/FINDINGS.md; w13-seeds-and-tables/TABLES.md
pool tracks it at $r=-0.96$ to $-0.98$. % SRC: w12-composition-ladder/FINDINGS.md; w13-seeds-and-tables/TABLES.md
Data-matched margins: lowest 95\% lower bound +2.46 ($n=3$, seed 2); a ten-times spot check moves the LSTM % SRC: w20-shapes-data-matched/FINDINGS.md
from 20.55 to 21.85 at $n=2$, seed 0 (not carried). % SRC: w20-shapes-data-matched/FINDINGS.md
n-CLU-content margins: mean over rungs +10.28 (+10.75 step-matched), against +9.97 for the data-matched set % SRC: w21-ladder-rivals-32k/FINDINGS.md
and +17.53 for the earlier set; lowest 95\% lower bound +1.98 ($n=2$, seed 2, against the step-matched MLP). % SRC: w21-ladder-rivals-32k/FINDINGS.md
Two MLP fits with the same rows, width, schedule and 2,560 steps, differing only in the random draw, differ per seed % SRC: w21-ladder-rivals-32k/FINDINGS.md
by $-2.15$ to $+4.13$; their per-combination interval excludes zero on 12 of 15 rung--seed legs. % SRC: w21-ladder-rivals-32k/FINDINGS.md
Coverage control (Sec.~\ref{sec:ladder}): on validation (60 combinations of 24 rows, 3 seeds) $n=6$ at $n=3$'s % SRC: w15-coverage-control/FINDINGS.md
coverage reads 22.767 (21.767 / 23.050 / 23.483), equal to $n=3$, against $n=6$'s 23.478 (22.967 / 23.617 / 23.850), % SRC: w15-coverage-control/FINDINGS.md
a rise of +1.200 / +0.567 / +0.367, coverage share 1.00 on every seed, mask channel 0.000; the TEST identity % SRC: w15-coverage-control/FINDINGS.md
check reads acceptance 1.000000 under either rung's bitmap and \texttt{distinct\_valid} $=$ \texttt{distinct\_valid\_novel} % SRC: w15-coverage-control/FINDINGS.md
in 3,600 of 3,600 published TEST rows, on the identical 8,640 combination rows.} % SRC: w15-coverage-control/FINDINGS.md
\label{tab:ladder}""")], [])

add('T4', 'Tab. del caption', [('both',
r"""composition on the surviving combinations, before $\to$ after, with the paired change, and the downdated
object minus the object refit without the rows.}
\label{tab:del}""",
r"""composition on the surviving combinations, before $\to$ after, with the paired change, and the downdated
object minus the object refit without the rows. \emph{Notes.} The book's statistic is the per-row first moment % SRC: w20-shapes-data-matched/FINDINGS.md
of its 117,964 fit rows. At seed 0 the red-sphere loss falls on the 20 combinations that request red with % SRC: w20-shapes-data-matched/FINDINGS.md
another shape ($-32.1$ each); the 36 that request a sphere in another colour move by $-1.47$ and the other 53 % SRC: w20-shapes-data-matched/FINDINGS.md
by $+0.51$. Probe (Sec.~\ref{sec:shapes-del}): without the address-book carry the read moves 0.610083 relative % SRC: w13-seeds-and-tables/TABLES.md
RMS on the 720 red-requesting requests and 0.614277 on the 4,968 others; with it, \sci{6.194}{-08} and % SRC: w13-seeds-and-tables/TABLES.md
\sci{6.163}{-08}; the carry rewrites 1,015,808 of the address book's 1,060,864 bytes and reads 0 data rows. % SRC: w13-seeds-and-tables/TABLES.md
At $K=5$ (rank/width 0.156) the red deletion's $\cos_{\min}$ is 0.1999, against 0.9976 for a size-matched % SRC: w13-seeds-and-tables/TABLES.md; w13-draft-revise/DRAFT-v1.md
random deletion.}
\label{tab:del}""")], [])

add('T5', 'Tab. hole caption', [('both',
r"""the same post-deletion requests is a declared not-run.}
\label{tab:hole}""",
r"""the same post-deletion requests is a declared not-run. Our two holes take 0.0000 s in a 1,322-byte % SRC: w13-seeds-and-tables/TABLES.md
coupling table (1,322 pair slots). A free 8-byte value-presence bitmap scores the value deletion % SRC: w13-seeds-and-tables/TABLES.md
at 1.0000 with 0 rows read; the comparator's conjunction fit takes 10.48 s.} % SRC: w12-deletion-baselines/FINDINGS.md
\label{tab:hole}""")], [])

add('T6', 'Tab. ablations caption', [('both',
r"""Tab.~\ref{tab:t7}; the two are never merged.}
\label{tab:ablations}""",
r"""Tab.~\ref{tab:t7}; the two are never merged. At the $K=5$ endpoint of the ladder the composition-only n-CLU is % SRC: w13-seeds-and-tables/TABLES.md
712,072 to 719,352 bytes, 1.358 to 1.372 times the reference. Evicting red from the earlier item memory % SRC: w13-seeds-and-tables/TABLES.md; w12-composition-ladder/FINDINGS.md
refuses at 1.0000 on all three seeds and changes composition by $-2.0899$ / $-0.9101$ / $-6.7416$ against a % SRC: w13-seeds-and-tables/TABLES.md
declared $\pm1.0$ bar; punching a hole refuses at 1.0000 and changes it by $+0.0000$.} % SRC: w13-seeds-and-tables/TABLES.md
\label{tab:ablations}""")], [])

add('T7', 'Tab. gentable-kstar caption', [('both',
r"""(controls collapse) \textsc{pass}; S2 (beats the earlier n-CLU's read on every seed) \textsc{fail}, seed 0;""",
r"""(controls collapse) \textsc{pass}; S2 (beats the earlier n-CLU's read on every seed, margin 0.020) \textsc{fail}, seed 0; % SRC: w15-lace-poe-rows/FINDINGS.md""")], [])

add('T8', 'Tab. cmazet-sym', [('both',
r"""mean \textbf{+0.046}: three clear wins, two ties, one loss. Seed 3's matched MLP did not converge at either step count \\""",
r"""mean \textbf{+0.046}: three clear wins, two ties, one loss. Seed 3's matched MLP did not converge at either step count;
seed 5: two held-out tuples fully supported in training score 0, their outcome code decoding at the wrong side of a 0.04 margin and wrong on 106 of its 317 fit rows \\ % SRC: w23-seed5-diagnosis/FINDINGS.md; w23-seed5-diagnosis/Q1-CONTENT.json"""),
('both',
r"""seed 5: a matched transformer at 80,000 steps equals the n-CLU. An input-representation result, not an n-CLU result \\
% ENDSRC""",
r"""seed 5: a matched transformer at 80,000 steps equals the n-CLU. An input-representation result, not an n-CLU result \\
% ENDSRC
\quad value hold-out, n-CLU-REL (TEST) & 6 & n-CLU-REL 0.554 against the best matched baseline 0.797 (six-seed means); behind on every seed \\ % SRC: w21-rel-newseeds/TABLES.md""")], [])

add('R16', 'Results', [('both',
r"""coverage, and by construction it can only be coverage: rungs 3--6 probe the identical 120 combinations, % SRC: w12-composition-ladder/FINDINGS.md""",
r"""coverage, and by construction it can only be coverage: rungs 3--6 probe the identical combinations (Tab.~\ref{tab:ladder}), % SRC: w12-composition-ladder/FINDINGS.md
% PW2-5 moved -> Tab. ladder caption (the identical 120 combinations): 120; SRC: w12-composition-ladder/FINDINGS.md""")],
[('120 combinations', 'Tab. ladder caption', 'w12-composition-ladder/FINDINGS.md')])

add('R17', 'Results', [('both',
r"""An item memory of 64 whole written rows at""",
r"""An item memory of whole written rows at"""),
('both',
r"""% PW2-5 moved -> Tab. ablations (row 4: item memory""",
r"""% PW2-5 moved -> Tab. ablations (row label: item memory, 64 rows) and Tab. ablations (row 4: item memory""")],
[('64 whole written rows', 'Tab. ablations (row label "item memory, 64 rows, oracle read (ceiling)")', 'w13-seeds-and-tables/TABLES.md')])

add('R18', 'Results', [('both',
r"""margin over a 180k-parameter learner on all three seeds""",
r"""margin over the best learned baseline on all three seeds"""),
('both',
r"""% PW2-5 moved -> Tab. adapt (margin column): """,
r"""% PW2-5 moved -> Tab. adapt (caption: a 180k-parameter MLP; margin column): 180k-parameter; """)],
[('180k-parameter learner', 'Tab. adapt caption ("the best learned baseline (a 180k-parameter MLP in every cell)")', 'w13-encoder-adapt/CROSS-ARM.md'),
 ('80,000 steps (seed-5 tie)', 'Tab. cmazet-sym (n-CLU-REL row: "a matched transformer at 80,000 steps"); prose now "$10\\times$ training steps"', 'w21-maze-t-seed4/FINDINGS.md')])
