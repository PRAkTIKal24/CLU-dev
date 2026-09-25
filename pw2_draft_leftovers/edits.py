r"""PW-2 item 4 (+ the Hub-added item-5 follow-up on the Discussion limitations list): every .tex edit, both drafts.
Format as pw2_draft_numbers/edits.py: (id, part, [(scope, old, new)], moved).
scope: 'both' (same old/new in both drafts), 'v2' or 'style'. Each `old` must occur exactly once in the draft it is
applied to (apply.py asserts it). `moved`: (numbers that left the prose, where they live now, SRC) for part 7.
Pointer comments are tagged `% PW2-4` (parts 2-6) and `% PW2-5 moved` (part 7, the item-5 rule)."""

E = []
def add(id, part, pairs, moved=()):
    E.append(dict(id=id, part=part, pairs=pairs, moved=list(moved)))

# ------------------------------------------------------------------ part 2: Figs. 5-8 cited from their result sentence
add('F5', '2 fig:ladder', [('both',
r"""On every rung and every seed the n-CLU's score in Tab.~\ref{tab:ladder} exceeds every baseline's
printed score. Paired""",
r"""On every rung and every seed the n-CLU's score in Tab.~\ref{tab:ladder} exceeds every baseline's
printed score (Fig.~\ref{fig:ladder}). Paired""")])

add('F6', '2 fig:crossover', [('both',
r"""is what the rule returns, not a $K$ that three seeds distinguish from $K=24$ or $K=28$.""",
r"""is what the rule returns, not a $K$ that three seeds distinguish from $K=24$ or $K=28$ (Fig.~\ref{fig:crossover}).""")])

add('F7a', '2 fig:deletion', [('both',
r"""by about six tuples per held-out combination, on all three seeds (Tab.~\ref{tab:del}).""",
r"""by about six tuples per held-out combination, on all three seeds (Tab.~\ref{tab:del}, Fig.~\ref{fig:deletion}).""")])

add('F7b', '2 fig:deletion', [('both',
r"""zero against the MLP at seeds 1 and 2 and against the LSTM at seed 2:""",
r"""zero against the MLP at seeds 1 and 2 and against the LSTM at seed 2 (Fig.~\ref{fig:deletion}):""")])

add('F8', '2 fig:adapt', [('both',
r"""of three under a low-rank adapter (Tab.~\ref{tab:adapt}).""",
r"""of three under a low-rank adapter (Tab.~\ref{tab:adapt}, Fig.~\ref{fig:adapt}).""")])

# ------------------------------------------------------------------ part 3: the composition-only byte-ratio TODO sentence
add('T1', '3 TODO', [('both',
r"""$\Kstar$ (a declared not-run, App.~\ref{app:notrun}). \TODO{no record prints the byte ratio of the composition-only n-CLU to the single design;
the records give the two totals printed here and in Tab.~\ref{tab:ledger-kstar}, and they are
not divided here.}
""",
r"""$\Kstar$ (a declared not-run, App.~\ref{app:notrun}).
% PW2-4 (pw2-draft-leftovers): dropped the sentence carrying the TODO "no record prints the byte ratio of the composition-only n-CLU to the single design; the records give the two totals printed here and in Tab. ledger-kstar, and they are not divided here" (no record supports a ratio; the two totals stay in Tab. ablations caption and Tab. ledger-kstar)
""")])

# ------------------------------------------------------------------ part 4: the two undrawable observation-only placeholders
add('P1', '4 placeholder (fig:cmazet-families)', [('both',
r"""Sec.~\ref{sec:maze-wm}. The models of Tab.~\ref{tab:cmazet} were not persisted:
\PLACEHOLDER{per-model $s_{t+1}$ for the observation-only input of Tab.~\ref{tab:cmazet}
(n-CLU, Transformer, MLP, LSTM) with the engine's judge's marks, one success and one failure per model:
not drawable, those models were not persisted; may remain undrawn, and if drawn may show the n-CLU
failing where a baseline succeeds}}{fig:cmazet-families}
""",
r"""Sec.~\ref{sec:maze-wm}.}{fig:cmazet-families}
% PW2-4 (pw2-draft-leftovers): cut the undrawable placeholder "per-model s_{t+1} for the observation-only input of Tab. cmazet (n-CLU, Transformer, MLP, LSTM) with the engine's judge's marks, one success and one failure per model" and its lead-in "The models of Tab. cmazet were not persisted:"; App. mazewm states once that those models were not persisted
""")])

add('P2', '4 placeholder (fig:cmazet-door-tint)', [('both',
r"""under each $s_{t+1}$ is read as described in Sec.~\ref{sec:maze-wm}. For the models of
Tab.~\ref{tab:cmazet}:
\PLACEHOLDER{per-model $s_{t+1}$ on the door-then-tint transitions for the
observation-only input of Tab.~\ref{tab:cmazet}, with the engine's judge's marks: not drawable, those
models were not persisted; may remain undrawn, and if drawn may show every model, the n-CLU included,
failing}}{fig:cmazet-door-tint}
""",
r"""under each $s_{t+1}$ is read as described in Sec.~\ref{sec:maze-wm}.}{fig:cmazet-door-tint}
% PW2-4 (pw2-draft-leftovers): cut the undrawable placeholder "per-model s_{t+1} on the door-then-tint transitions for the observation-only input of Tab. cmazet, with the engine's judge's marks" and its lead-in "For the models of Tab. cmazet:"; App. mazewm states once that those models were not persisted
""")])

add('P3', '4 the one appendix sentence', [('both',
r"""\end{table}
% ---- CMAZE-T per-arm panels (w21 figures-redraw, Part C), beside Table tab:cmazet ----
% Drawn from w19's persisted words-plus-observation (BOTH) TEST objects, hash-verified before loading;
% every arm's n-way re-read equals the scored artefact on every seed (figures/data/cmazet_items.REPRO.json).
% The two PLACEHOLDERs name the per-arm outputs of Table tab:cmazet's own arms, which w16 never persisted
""",
r"""\end{table}

\noindent The observation-only models of Tab.~\ref{tab:cmazet} were not persisted, so their per-model panels are not drawn.
% PW2-4 (pw2-draft-leftovers): this sentence replaces the two placeholders cut from Figs. cmazet-families and cmazet-door-tint.
% ---- CMAZE-T per-arm panels (w21 figures-redraw, Part C), beside Table tab:cmazet ----
% Drawn from w19's persisted words-plus-observation (BOTH) TEST objects, hash-verified before loading;
% every arm's n-way re-read equals the scored artefact on every seed (figures/data/cmazet_items.REPRO.json).
% The two placeholders that named the per-arm outputs of Table tab:cmazet's own arms, which w16 never persisted, were cut (PW2-4)
""")])

add('P4', '4 main-text sentence made false by part 4', [('both',
r"""(Sec.~\ref{sec:maze-wm}). The rows still pending a run are marked placeholders whose text names the row.
""",
r"""(Sec.~\ref{sec:maze-wm}).
% PW2-4 (pw2-draft-leftovers): dropped "The rows still pending a run are marked placeholders whose text names the row." (no placeholder remains)
""")])

add('P5', '4 placeholder index made false by part 4', [('both',
r"""Two rows remain, each marked in the text by a red \textsc{placeholder} whose argument names
the row: the per-arm generated states of the arms of Tab.~\ref{tab:cmazet} in
Figs.~\ref{fig:cmazet-families} and~\ref{fig:cmazet-door-tint}. They are
permanently blocked, not pending. The run behind Tab.~\ref{tab:cmazet} persisted no arm object and
no per-transition prediction (its run records hold aggregates only), so these panels cannot be drawn
from that run, and we keep them as named placeholders rather than draw them from any other run. Rows
that were pending earlier""",
r"""% PW2-4 (pw2-draft-leftovers): dropped the two sentences on the two remaining placeholders (cut, part 4) and the reason they stayed ("The run behind Tab. cmazet persisted no arm object and no per-transition prediction (its run records hold aggregates only)"); App. mazewm states it once
Rows
that were pending earlier""")])

add('P6', '4 placeholder index made false by part 4', [('both',
r"""and~\ref{fig:cmazet-door-tint} draw every arm of the words-plus-observation condition from its
persisted objects; their two placeholders name the per-arm outputs of the arms of
Tab.~\ref{tab:cmazet}.
""",
r"""and~\ref{fig:cmazet-door-tint} draw every arm of the words-plus-observation condition from its
persisted objects.
% PW2-4 (pw2-draft-leftovers): dropped "; their two placeholders name the per-arm outputs of the arms of Tab. cmazet" (cut, part 4)
""")])

# ------------------------------------------------------------------ part 5: machine paths in comments
add('M1', '5 absolute path', [('both',
r"""(called EV below; on the drafting machine it is ~/clu/evidence/outputs/).""",
r"""(called EV below: the outputs/ directory of the evidence tree).""")])
add('M2', '5 absolute path', [('both',
r"""(from ~/clu/outputs/w17-paper/, template in ./iclr2027/)""",
r"""(from the folder that holds this file, template in ./iclr2027/)""")])

# ------------------------------------------------------------------ part 6: the 9-pt quote block at body size
add('Q1', '6 quote', [('both', r"""\begin{quote}\small
""", r"""\begin{quote}
""")])

# ------------------------------------------------------------------ part 7: item 5 on the Discussion limitations list
add('L2', '7 limitation 2', [('both',
r"""  sphere lowers composition on
  the surviving combinations by 6.13 / 6.43 / 6.30 tuples per held-out combination, 32.1 on the combinations that request red % SRC: w20-shapes-data-matched/FINDINGS.md
  with another shape at seed 0, while a random 1\% deletion moves them by at most 0.11; retraining a % SRC: w20-shapes-data-matched/FINDINGS.md
  baseline without the same rows is cleaner on collateral damage (Sec.~\ref{sec:shapes-del},
  Sec.~\ref{sec:del-comparators}).
""",
r"""  sphere lowers composition on
  the surviving combinations by about six tuples per held-out combination, about thirty on the combinations that request red % SRC: w20-shapes-data-matched/FINDINGS.md
  with another shape at seed 0, while a random 1\% deletion moves them by at most about a tenth of a tuple; retraining a % SRC: w20-shapes-data-matched/FINDINGS.md
  baseline without the same rows is cleaner on collateral damage (Tab.~\ref{tab:del}, Sec.~\ref{sec:shapes-del},
  Sec.~\ref{sec:del-comparators}).
  % PW2-5 moved -> Tab. del (column (d): -6.128 / -6.431 / -6.303; item -0.018 / -0.110 / -0.009; caption notes: the 20 red-with-another-shape combinations -32.1 each at seed 0) and Fig. deletion caption (6.13, 6.43, 6.30; 0.02, 0.11, 0.01): 6.13 / 6.43 / 6.30; 32.1; 0.11; SRC: w20-shapes-data-matched/FINDINGS.md
""")],
[('6.13 / 6.43 / 6.30', 'Tab. del column (d) (−6.128 / −6.431 / −6.303) and Fig. deletion caption (6.13, 6.43, 6.30); text: "about six"', 'w20-shapes-data-matched/FINDINGS.md'),
 ('32.1 (seed 0, red with another shape)', 'Tab. del caption notes (the seed-0 breakdown, −32.1 each); text: "about thirty"', 'w20-shapes-data-matched/FINDINGS.md'),
 ('at most 0.11 (random 1% deletion)', 'Tab. del column (d), item row (−0.110) and Fig. deletion caption (0.11); text: "at most about a tenth of a tuple"', 'w20-shapes-data-matched/FINDINGS.md')])

add('L3', '7 limitation 3', [('both',
r"""The book's retained statistic is 0.97 GB, % SRC: w20-shapes-data-matched/FINDINGS.md
  the size of its fit data, and its refit takes 2,367 to 5,913 s, longer than a baseline's % SRC: w20-shapes-data-matched/FINDINGS.md
  retrain-without; ``reads no training row'' is not a size or a speed claim.
""",
r"""The book's retained statistic is nearly a gigabyte, % SRC: w20-shapes-data-matched/FINDINGS.md
  the size of its fit data, and its refit takes longer than a baseline's % SRC: w20-shapes-data-matched/FINDINGS.md
  retrain-without (Tabs.~\ref{tab:del} and~\ref{tab:rivals-del}); ``reads no training row'' is not a size or a speed claim.
  % PW2-5 moved -> Tab. del (cost rows: reads the 967,304,800 B statistic; book refit 2,590.7 ... 2,367.3 ... 5,912.9 s) and Tab. rivals-del (retrain times): 0.97 GB; 2,367 to 5,913 s; SRC: w20-shapes-data-matched/FINDINGS.md; w20-shapes-data-matched/SHAPES-MATCHED.json
""")],
[('0.97 GB', 'Tab. del cost row ("reads the 967,304,800 B statistic"); text: "nearly a gigabyte"', 'w20-shapes-data-matched/SHAPES-MATCHED.json'),
 ('2,367 to 5,913 s', 'Tab. del cost row (book refit, per scope and seed: 2,367.3 … 5,912.9); text: "longer than a baseline\'s retrain-without" (Tab. rivals-del)', 'w20-shapes-data-matched/FINDINGS.md')])

add('L4', '7 limitation 4', [('both',
r"""The deletable object at $\Kstar$ is 4.0768 to 4.0897
  times the reference (Tab.~\ref{tab:ledger-kstar}); across nine swept partitions the honest
  four-part total is U-shaped in $K$, with its minimum 3.9942 times the reference at $K=24$.
  $\Kstar$ is not the point of smallest footprint. On the one configuration at $K\ge36$ whose composition
  was measured ($K=40$, 7 atoms per well, 0.991 times the pin) composition collapses to 0.0648.
  No object with a float64 sufficient statistic composes and forgets inside 524,288 bytes at rank
  64 on any swept partition (the validation sweep labels itself a smoke).
""",
r"""The deletable object at $\Kstar$ is about four
  times the reference (Tab.~\ref{tab:ledger-kstar}); across nine swept partitions the honest
  four-part total is U-shaped in $K$, with its minimum at $K=24$ (Tab.~\ref{tab:kcurve}).
  $\Kstar$ is not the point of smallest footprint. On the one configuration at $K\ge36$ whose composition
  was measured, composition collapses (Tab.~\ref{tab:kcurve}, caption).
  No object with a float64 sufficient statistic composes and forgets inside the reference at rank
  64 on any swept partition (the validation sweep labels itself a smoke).
  % PW2-5 moved -> Tab. ledger-kstar (last row: 4.0897 / 4.0768 / 4.0847 x the 524,288 B reference), Tab. kcurve (K=24 row: 3.9942) and Tab. kcurve caption (ADDED: K=40, 7 atoms per well, 0.991 times the reference, validation composition 0.0648); 524,288 is the reference defined in Methods: 4.0768 to 4.0897; 3.9942; K=40; 7 atoms; 0.991; 0.0648; 524,288; SRC: w13-seeds-and-tables/TABLES.md; w13-referee-2/REFEREE-v1.md
""")],
[('4.0768 to 4.0897', 'Tab. ledger-kstar (last row, 4.0897 / 4.0768 / 4.0847); text: "about four times the reference"', 'w13-referee-2/REFEREE-v1.md'),
 ('3.9942 (minimum at K=24)', 'Tab. kcurve (K=24 row, four-part ×); text keeps the partition name $K=24$', 'w13-seeds-and-tables/TABLES.md'),
 ('K=40; 7 atoms per well; 0.991 times the pin; 0.0648', 'Tab. kcurve caption (ADDED); text: "composition collapses"', 'w13-seeds-and-tables/TABLES.md; w13-referee-2/REFEREE-v1.md'),
 ('524,288 bytes', 'not the reference definition here (that is Methods, "The 524,288-byte figure ... is a reference column"); text: "the reference"', 'w13-referee-2/REFEREE-v1.md')])

add('L4c', '7 caption that now carries K=40', [('both',
r"""which prints $K=5\ldots32$ as a plateau, not a selection. % SRC: w15-crossover-seeds/CROSSOVER-3SEED.md
}
\label{tab:kcurve}""",
r"""which prints $K=5\ldots32$ as a plateau, not a selection. % SRC: w15-crossover-seeds/CROSSOVER-3SEED.md
One configuration outside the sweep, $K=40$ with 7 atoms per well at 0.991 times the reference, was % SRC: w13-seeds-and-tables/TABLES.md; w13-referee-2/REFEREE-v1.md
measured on validation: its composition collapses to 0.0648. % SRC: w13-seeds-and-tables/TABLES.md; w13-referee-2/REFEREE-v1.md
}
\label{tab:kcurve}""")])

add('L5', '7 limitation 5', [('both',
r"""and the MLP at seed 0 still gains 1.47 tuples % SRC: w20-shapes-data-matched/FINDINGS.md
  per held-out combination from ten to thirty times its steps (Sec.~\ref{sec:shapes-t7}).
""",
r"""and the MLP at seed 0 still gains more than a tuple % SRC: w20-shapes-data-matched/FINDINGS.md
  per held-out combination from ten to thirty times its steps (Sec.~\ref{sec:shapes-t7}, Tab.~\ref{tab:t7}).
  % PW2-5 moved -> Sec. results first paragraph (Head-held: "+1.47") and Tab. t7 (diagnostic MLP row at thirty times the steps, 41.477 at seed 0): 1.47; SRC: w20-shapes-data-matched/FINDINGS.md
""")],
[('1.47 (MLP, seed 0, ten to thirty times its steps)', 'Sec. 4.1 first paragraph (Head-held, "+1.47") and Tab. t7 (diagnostic MLP row); text: "more than a tuple"', 'w20-shapes-data-matched/FINDINGS.md')])
