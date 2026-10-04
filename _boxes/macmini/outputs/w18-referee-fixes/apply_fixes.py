#!/usr/bin/env python3
"""w18 referee-fixes: apply REFEREE-v2 M2, M4, M5, M6, M7 to DRAFT-v2.tex and DRAFT-v2-style.tex.
Every replacement asserts its anchor occurs exactly once (or the stated count) so nothing lands silently.
usage: apply_fixes.py DIR   (edits DIR/DRAFT-v2.tex and DIR/DRAFT-v2-style.tex in place)"""
import re, sys, os

D = sys.argv[1]
V2, ST = os.path.join(D, 'DRAFT-v2.tex'), os.path.join(D, 'DRAFT-v2-style.tex')
txt = {V2: open(V2).read(), ST: open(ST).read()}
LOG = []

def rep(f, old, new, count=1, tag=''):
    n = txt[f].count(old)
    assert n == count, 'anchor count %d != %d in %s [%s]: %r' % (n, count, os.path.basename(f), tag, old[:90])
    txt[f] = txt[f].replace(old, new)
    LOG.append('%s\t%s\t%d' % (tag, os.path.basename(f), n))

def both(old, new, count=1, tag=''):
    for f in (V2, ST):
        rep(f, old, new, count, tag)

# =====================================================================================
# M7(e) preamble: the figure-slot fallback text named a pipeline role
both(r"""Figure slot \texttt{figures/\detokenize{#1}.pdf} (owned by the
  \texttt{plots} spoke; not yet on disk at draft time).}}}""",
     r"""Figure pending (\texttt{figures/\detokenize{#1}.pdf} not yet
  available).}}}""", tag='M7e-figslot-macro')

# =====================================================================================
# M4 -- positioning: narrow to factor-compositional generation with data-free exact deletion;
#       cite and distinguish SISA and CDM.
rep(V2, r"""Compositional generation is a crowded area
and exact unlearning is an established one, but their intersection is empty: a mechanical
screen of approximately 4,050 works finds no generative model that deletes training rows
exactly. % SRC: _thinkpad/w8/w8-novelty/NOVELTY.md""",
r"""Compositional generation is a crowded area
and exact unlearning is an established one, but their intersection is empty in a precise sense: a mechanical
screen of approximately 4,050 works finds no \emph{factor-compositional} generative model (one that generates held-out combinations of factor values) that deletes training rows
exactly while reading no data and retraining nothing. % SRC: _thinkpad/w8/w8-novelty/NOVELTY.md""", tag='M4-abstract')
rep(ST, r"""Compositional generation is extensively studied and exact unlearning established; their intersection, however, is empty: a mechanical
screen of approximately 4,050 works finds no generative model that deletes training rows
exactly. % SRC: _thinkpad/w8/w8-novelty/NOVELTY.md""",
r"""Compositional generation is extensively studied and exact unlearning established; their intersection, however, is empty in a precise sense: a mechanical
screen of approximately 4,050 works finds no \emph{factor-compositional} generative model (one that generates held-out combinations of factor values) that deletes training rows
exactly while reading no data and retraining nothing. % SRC: _thinkpad/w8/w8-novelty/NOVELTY.md""", tag='M4-abstract')

both(r"""\subsection{Positioning: compositional generation is crowded; with exact deletion it is empty}""",
     r"""\subsection{Positioning: compositional generation is crowded; with data-free exact deletion it is empty}""", tag='M4-heading')

M4_PARA = r"""Two works outside the screen answer a broader reading of that query, and they fix the form of
our claim. SISA training~\citep{bourtoule2021machine} makes any learned model, a generator
included, exactly unlearnable by retraining the shard that held the deleted rows: the deletion is
exact, but it reads that shard's retained data and retrains. Compartmentalized diffusion models
(Golatkar, Achille, Swaminathan and Soatto) train separate diffusion models on separate data
sources and compose them at inference, which gives selective forgetting of a source by dropping
or retraining the model that saw it; ``compositional'' there means composing models of data
sources, not composing held-out combinations of factor values.
% cite-tagger: bib entry needed -- Golatkar, Achille, Swaminathan, Soatto, "Training Data Protection with Compositional Diffusion Models", arXiv:2308.01937 (not in references.bib)
Neither deletes without reading data and retraining, and neither generates held-out
combinations of factor values."""

rep(V2, r"""\emph{generative} $\wedge$ \emph{exact deletion}, returns zero works.
% SRC: _thinkpad/w8/w8-novelty/NOVELTY.md
We therefore state the position in the only form the screen supports: \emph{the intersection
of compositional generation with exact deletion is empty} in the works we could screen.""",
r"""\emph{generative} $\wedge$ \emph{exact deletion}, returns zero works on the screen.
% SRC: _thinkpad/w8/w8-novelty/NOVELTY.md
""" + M4_PARA + r"""
We therefore state the position in the only form the screen and these two works support:
\emph{the intersection of factor-compositional generation with exact deletion that reads no data
and retrains nothing is empty} in the works we could screen.""", tag='M4-positioning')
rep(ST, r"""\emph{generative} $\wedge$ \emph{exact deletion}, returns zero works.
% SRC: _thinkpad/w8/w8-novelty/NOVELTY.md
We therefore state the position only in the form that the screen supports: \emph{the
intersection of compositional generation with exact deletion is empty} in the works we could
screen.""",
r"""\emph{generative} $\wedge$ \emph{exact deletion}, returns zero works on the screen.
% SRC: _thinkpad/w8/w8-novelty/NOVELTY.md
""" + M4_PARA + r"""
We therefore state the position only in the form that the screen and these two works support:
\emph{the intersection of factor-compositional generation with exact deletion that reads no data
and retrains nothing is empty} in the works we could screen.""", tag='M4-positioning')

rep(V2, r"""nothing it has not stored; a learned generator composes but can delete only by retraining; and""",
        r"""nothing it has not stored; a learned generator composes but can delete exactly only by retraining
(the whole model, or a shard of it~\citep{bourtoule2021machine}); and""", tag='M4-retrain')
rep(ST, r"""composes nothing it has not stored; a learned generator composes but can delete only by
retraining; and""",
        r"""composes nothing it has not stored; a learned generator composes but can delete exactly only by
retraining (the whole model, or a shard of it~\citep{bourtoule2021machine}); and""", tag='M4-retrain')

# =====================================================================================
# M2 -- two rival sets, an order of magnitude apart on the same corners
rep(V2, r"""Every rival is beaten beyond the declared 1.0 margin on 109 of 109 corners on every leg; the
smallest paired delta""",
r"""The rivals of Table~\ref{tab:t7} are the weaker of the two rival sets in this paper; the
store's standing against the strongest available rival is given below (\emph{Two rival sets}).
Each of them is beaten beyond the declared 1.0 margin on 109 of 109 corners on every leg; the
smallest paired delta""", tag='M2-lead')
rep(ST, r"""On every leg, every rival is beaten beyond the declared 1.0 margin on 109 of 109 corners; the
smallest paired delta""",
r"""The rivals of Table~\ref{tab:t7} are the weaker of the two rival sets in this paper; the
store's standing against the strongest available rival is given below (\emph{Two rival sets}).
On every leg, each of them is beaten beyond the declared 1.0 margin on 109 of 109 corners; the
smallest paired delta""", tag='M2-lead')

both(r"""seed. Rivals are matched on read-path parameters (about 131k); the table is a floor, never a % SRC: w13-seeds-and-tables/TABLES.md""",
     r"""seed. Rivals are matched on read-path parameters (about 131k) and are the weaker of the paper's two rival sets (\S\ref{sec:shapes-t7}, \emph{Two rival sets}); the table is a floor, never a % SRC: w13-seeds-and-tables/TABLES.md""", tag='M2-t7-caption')

M2_PARA = r"""\paragraph{Two rival sets, and which is stronger.} Two sets of read-path-matched learned rivals
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

"""
both(r"""\begin{table}[t]
\centering\small
\caption{\textbf{The four-part ledger of the $\Kstar=32$ object""",
     M2_PARA + r"""\begin{table}[t]
\centering\small
\caption{\textbf{The four-part ledger of the $\Kstar=32$ object""", tag='M2-paragraph')

both(r"""no scored budget; every row held out). Rival rows are the read-path-matched rivals scored on
the same corners; a rival does not depend on $K$.""",
     r"""no scored budget; every row held out). Rival rows are the earlier, stronger read-path-matched rival
set (\S\ref{sec:shapes-t7}, \emph{Two rival sets}) scored on
the same corners; a rival does not depend on $K$.""", tag='M2-ladder-caption')

CAVEAT_NEW = r"""the ladder's rival columns are the earlier rival set,
whose \texttt{rival\_transformer} and \texttt{rival\_lstm} were fitted before a later dtype repair of
the token families and were not re-fitted on the ladder's rungs, yet that set is the
\emph{stronger} of the paper's two rival sets on shared corners (\S\ref{sec:shapes-t7}); and
nothing in this table is a composition-depth result."""
CAVEAT_OLD = r"""the ladder's \texttt{rival\_transformer} and
\texttt{rival\_lstm} columns are the earlier rival set's pre-repair numbers and were not
re-fitted on the ladder's rungs; and nothing in this table is a composition-depth result."""
rep(V2, 'Two caveats travel with the rival rows: ' + CAVEAT_OLD, 'Two caveats travel with the rival rows: ' + CAVEAT_NEW, tag='M2-caveat')
rep(ST, 'Two caveats accompany the rival rows: ' + CAVEAT_OLD, 'Two caveats accompany the rival rows: ' + CAVEAT_NEW, tag='M2-caveat')

# =====================================================================================
# M5 -- environments never share a table
# (a) tab:req -> two single-environment tables
REQ_OLD = r"""\begin{table}[h]
\centering\small
\caption{\textbf{The request-layer fix. CMAZE-2 (before) and CMAZE-2b (after) are different
environments; this table is the only place they appear side by side, as the before/after of one
acceptance rule, and no result row of either appears in it.} Validity under the exact judge;
SLACK10 is every kept pair at CMAZE-1's own budget rule.}
\label{tab:req}
\setlength{\tabcolsep}{3pt}
\begin{adjustbox}{max width=\linewidth}\begin{tabular}{lcc}
\toprule
quantity & before (w9 layer) & after (seeds 0 / 1 / 2) \\
\midrule
% SRCBLOCK: w14-request-layer-fix/FINAL-REPORT.md
routes by keys: 1 / 2 / 3 / 4 & 1242 / 198 / 9 / --- (of 1449) & 0 / 3825 / 751 / 35 (of 4611) \\
fraction of routes needing $\ge2$ keys & 0.1428 & 1.0000 \\
BLIND-SEQUENTIAL, SLACK10, all pairs & 0.8295 & 0.1810 / 0.1553 / 0.1068 \\
BLIND-SEQUENTIAL, unseen pair, test & 0.9077 & 0.2344 / 0.2000 / 0.1818 \\
ORACLE & 1.0000 & 1.0000 \\
cross-maze fragment lookup, $r=1$ / $r=2$ & 0.0000 / 0.0000 & 0.0000 / 0.0000 \\
% ENDSRC
\bottomrule
\end{tabular}\end{adjustbox}
\end{table}"""
REQ_NEW = r"""\begin{table}[h]
\centering\small
\caption{\textbf{CMAZE-2 request-layer audit (the original request layer, before the one
acceptance rule), seed 0.} Validity under the exact judge; SLACK10 is every kept pair at % SRC: w14-request-layer-fix/FINAL-REPORT.md
CMAZE-1's own budget rule. An audit, not a result row. CMAZE-2b is audited in its own table
(Table~\ref{tab:req2b}); the before/after contrast is stated in prose (\S\ref{sec:req-layer}).}
\label{tab:req}
\setlength{\tabcolsep}{3pt}
\begin{adjustbox}{max width=\linewidth}\begin{tabular}{lcc}
\toprule
quantity & seeds & CMAZE-2 \\
\midrule
% SRCBLOCK: w14-request-layer-fix/FINAL-REPORT.md
routes by keys: 1 / 2 / 3 / 4 & 1 & 1242 / 198 / 9 / --- (of 1449) \\
fraction of routes needing $\ge2$ keys & 1 & 0.1428 \\
BLIND-SEQUENTIAL, SLACK10, all pairs & 1 & 0.8295 \\
BLIND-SEQUENTIAL, unseen pair, test & 1 & 0.9077 \\
ORACLE & 1 & 1.0000 \\
cross-maze fragment lookup, $r=1$ / $r=2$ & 1 & 0.0000 / 0.0000 \\
% ENDSRC
\bottomrule
\end{tabular}\end{adjustbox}
\end{table}

\begin{table}[h]
\centering\small
\caption{\textbf{CMAZE-2b request-layer audit (after the one acceptance rule), seeds 0 / 1 / 2.} % SRC: w14-request-layer-fix/FINAL-REPORT.md
Validity under the exact judge; SLACK10 is every kept pair at CMAZE-1's own budget rule. The
key-count histogram is pooled over the three seeds. An audit, not a result row. CMAZE-2 is
audited in its own table (Table~\ref{tab:req}).}
\label{tab:req2b}
\setlength{\tabcolsep}{3pt}
\begin{adjustbox}{max width=\linewidth}\begin{tabular}{lcc}
\toprule
quantity & seeds & CMAZE-2b (seeds 0 / 1 / 2, or pooled) \\
\midrule
% SRCBLOCK: w14-request-layer-fix/FINAL-REPORT.md
routes by keys: 1 / 2 / 3 / 4 & 3 & 0 / 3825 / 751 / 35 (of 4611) \\
fraction of routes needing $\ge2$ keys & 3 & 1.0000 \\
BLIND-SEQUENTIAL, SLACK10, all pairs & 3 & 0.1810 / 0.1553 / 0.1068 \\
BLIND-SEQUENTIAL, unseen pair, test & 3 & 0.2344 / 0.2000 / 0.1818 \\
ORACLE & 3 & 1.0000 \\
cross-maze fragment lookup, $r=1$ / $r=2$ & 3 & 0.0000 / 0.0000 \\
% ENDSRC
\bottomrule
\end{tabular}\end{adjustbox}
\end{table}"""
both(REQ_OLD, REQ_NEW, tag='M5-req-split')
both(r"""(Table~\ref{tab:req}, Appendix~\ref{app:verb})""",
     r"""(Tables~\ref{tab:req} and~\ref{tab:req2b}, Appendix~\ref{app:verb})""", tag='M5-req-ref')

# (b) the figure that plotted CMAZE-2 and CMAZE-2b together is dropped (plots owns figures/)
both(r"""\figslot{fig_maze_requestlayer}{\textbf{The request-layer fix.} Key-count histogram and blind-sequential validity before (CMAZE-2) and after (CMAZE-2b) the one acceptance rule.}{fig:reqlayer}
""",
     r"""% fig_maze_requestlayer (both request-layer environments on one figure) removed under REFEREE-v2 M5;
% re-insert only as two single-environment figures drawn by the figures owner.
""", tag='M5-fig-drop')

# (c) tab:cmazet -> CMAZE-T table and CMAZE-O table. Placeholder arguments are moved, not edited.
def ph(s):
    return r'\PLACEHOLDER{' + s + '}'
T = ['w16 CMAZE-T seen row, 3 seeds, or a declared negative',
     'w16 CMAZE-T 2-way composition: store vs rivals, or a declared negative with its failing link',
     'w16 CMAZE-T $n$-way composition, or a declared negative with its failing link',
     'w16 CMAZE-T rollout validity over chained generated states, or a declared negative',
     'w16 CMAZE-T never-seen colour (scored, no refusal display), or a declared negative',
     'w16 CMAZE-T charter-C controls (must collapse for a store row), within- and cross-maze lookup floors, BLIND columns; else interface row or nothing',
     'w16 CMAZE-T value deletion: exactness, post-deletion composition vs retrain-without, cost; or a declared negative',
     'w16 CMAZE-T conjunction deletion, same columns; or a declared negative']
O = ['w16 CMAZE-O seen row, 3 seeds, or a declared negative',
     'w16 CMAZE-O 2-way composition: store vs rivals, or a declared negative with its failing link',
     'w16 CMAZE-O $n$-way composition, or a declared negative with its failing link',
     None,
     'w16 CMAZE-O never-seen colour, or a declared negative',
     'w16 CMAZE-O controls, lookup floor, BLIND columns; else interface row or nothing',
     'w16 CMAZE-O value deletion, same columns; or a declared negative',
     'w16 CMAZE-O conjunction deletion, same columns; or a declared negative']
ROWS = ['seen', 'unseen 2-way', 'unseen $n$-way', 'rollout validity', 'value hold-out',
        'controls, floors', 'deletion, value', 'deletion, conj.']
old_rows = ''.join('%s & %s & %s \\\\\n' % (r, ph(t), ph(o) if o else '(no rollouts)') for r, t, o in zip(ROWS, T, O))
CM_OLD = r"""\begin{table}[h]
\centering\scriptsize
\caption{\textbf{Maze world-model rows, pending wave 16: CMAZE-T (transition generation) and % (no result number on this line: wave labels / file names only)
CMAZE-O (observation generation).} Store against MLP / LSTM / transformer (and, on CMAZE-O, a
continuous MLP) at matched parameters on the same frozen encoder, 3 seeds, bars pre-registered.
CMAZE-T: given the grid state and an action, generate the next state; a rollout is valid iff every
generated step is legal and the goal is reached within budget. CMAZE-O: generate the grid state for
a request of colours, object cells and layout; exact placement and legality. Deletion rows: delete
colour $c$ (value) or colour $c$ $\wedge$ tinted floor (conjunction); post-deletion composition on
the other colours against rival retrain-without and as-trained, cost printed. \emph{Every cell may
be filled by a result or by a declared negative with its failing link.}}
\label{tab:cmazet}\label{tab:cmaze-del}\label{tab:cmazeo}
\setlength{\tabcolsep}{2pt}
\begin{adjustbox}{max width=\linewidth}\begin{tabular}{p{1.9cm}p{5.6cm}p{5.6cm}}
\toprule
row & CMAZE-T & CMAZE-O \\
\midrule
""" + old_rows + r"""\bottomrule
\end{tabular}\end{adjustbox}
\end{table}"""
def one_env(name, what, labels, cells, extra_rows):
    rows = ''.join('%s & pending & %s \\\\\n' % (r, ph(c)) if c else '%s & --- & (no rollouts) \\\\\n' % r
                   for r, c in zip(ROWS, cells))
    return (r"""\begin{table}[h]
\centering\scriptsize
\caption{\textbf{Maze world-model rows, pending: %s.} Store against %s at matched parameters on
the same frozen encoder, 3 seeds, bars pre-registered. %s Deletion rows: delete
colour $c$ (value) or colour $c$ $\wedge$ tinted floor (conjunction); post-deletion composition on
the other colours against rival retrain-without and as-trained, cost printed. \emph{Every cell may
be filled by a result or by a declared negative with its failing link.}}
%s
\setlength{\tabcolsep}{2pt}
\begin{adjustbox}{max width=\linewidth}\begin{tabular}{p{1.9cm}p{1.2cm}p{10.0cm}}
\toprule
row & seeds & %s \\
\midrule
""" % (name, extra_rows, what, labels, name.split(' ')[0]) + rows + r"""\bottomrule
\end{tabular}\end{adjustbox}
\end{table}""")
CM_NEW = (one_env('CMAZE-T (transition generation)', r"""CMAZE-T: given the grid state and an action, generate the next state; a rollout is valid iff every
generated step is legal and the goal is reached within budget.""", r'\label{tab:cmazet}\label{tab:cmaze-del}', T, 'MLP / LSTM / transformer')
          + '\n\n' +
          one_env('CMAZE-O (observation generation)', r"""CMAZE-O: generate the grid state for
a request of colours, object cells and layout; exact placement and legality.""", r'\label{tab:cmazeo}', O, 'MLP / LSTM / transformer / continuous MLP'))
both(CM_OLD, CM_NEW, tag='M5-cmazet-split')
rep(V2, r"""negatives are laid out as placeholders in Table~\ref{tab:cmazet} (Appendix~\ref{app:mazewm}).""",
        r"""negatives are laid out as placeholders in Tables~\ref{tab:cmazet} and~\ref{tab:cmazeo} (Appendix~\ref{app:mazewm}).""", tag='M5-cmazet-ref')
rep(ST, r"""Table~\ref{tab:cmazet} (Appendix~\ref{app:mazewm}) lays out as placeholders the pending rows,""",
        r"""Tables~\ref{tab:cmazet} and~\ref{tab:cmazeo} (Appendix~\ref{app:mazewm}) lay out as placeholders the pending rows,""", tag='M5-cmazet-ref')

# (d) tab:impl mixed the maze store with the two Shapes3D implementations
IMPL_OLD = r"""\begin{table}[h]
\centering\small
\caption{\textbf{Three implementations of exact (store-scoped) deletion, three rows, bands not
shared.} Seeds and configurations are separate fields; the nine legs of row 3 are 3 configurations
$\times$ 3 seeds.}
\label{tab:impl}
\setlength{\tabcolsep}{2.4pt}
\begin{adjustbox}{max width=\linewidth}\begin{tabular}{p{3.6cm}cccp{5.2cm}}
\toprule
implementation & seeds & configs & scored runs & band \\
\midrule
% SRCBLOCK: w13-seeds-and-tables/TABLES.md
1. maze store (CMAZE-2, TEST) & 3 & 1 & 3 & \sci{7.522}{-16} .. \sci{8.826}{-16} on the statistic; projector exactly 0.000 \\
2. Shapes3D downdate build (TEST), assertion battery & 3 & 1 & 0 & \sci{2.842}{-14} .. \sci{1.126}{-09}, five orders coarser \\
3. Shapes3D single-design candidates (TEST), assertion battery & 3 & 3 & 0 & worst 0.094 atom ulp / 0.0251 subspace ulp over nine legs \\
% ENDSRC
\bottomrule
\end{tabular}\end{adjustbox}
\end{table}"""
IMPL_NEW = r"""\begin{table}[h]
\centering\small
\caption{\textbf{Exact (store-scoped) deletion, implementation 1: the maze store (CMAZE-2).} The
two Shapes3D implementations are in Table~\ref{tab:impl}; environments never share a table.}
\label{tab:impl-maze}
\setlength{\tabcolsep}{2.4pt}
\begin{adjustbox}{max width=\linewidth}\begin{tabular}{p{3.6cm}cccp{5.2cm}}
\toprule
implementation & seeds & configs & scored runs & band \\
\midrule
% SRCBLOCK: w13-seeds-and-tables/TABLES.md
1. maze store (CMAZE-2, TEST) & 3 & 1 & 3 & \sci{7.522}{-16} .. \sci{8.826}{-16} on the statistic; projector exactly 0.000 \\
% ENDSRC
\bottomrule
\end{tabular}\end{adjustbox}
\end{table}

\begin{table}[h]
\centering\small
\caption{\textbf{Exact (store-scoped) deletion on Shapes3D, implementations 2 and 3, bands not
shared.} Seeds and configurations are separate fields; the nine legs of row 3 are 3 configurations
$\times$ 3 seeds. Implementation 2's band is five orders coarser than the maze store's
(Table~\ref{tab:impl-maze}).}
\label{tab:impl}
\setlength{\tabcolsep}{2.4pt}
\begin{adjustbox}{max width=\linewidth}\begin{tabular}{p{3.6cm}cccp{5.2cm}}
\toprule
implementation & seeds & configs & scored runs & band \\
\midrule
% SRCBLOCK: w13-seeds-and-tables/TABLES.md
2. Shapes3D downdate build (TEST), assertion battery & 3 & 1 & 0 & \sci{2.842}{-14} .. \sci{1.126}{-09} \\
3. Shapes3D single-design candidates (TEST), assertion battery & 3 & 3 & 0 & worst 0.094 atom ulp / 0.0251 subspace ulp over nine legs \\
% ENDSRC
\bottomrule
\end{tabular}\end{adjustbox}
\end{table}"""
both(IMPL_OLD, IMPL_NEW, tag='M5-impl-split')

# =====================================================================================
# M6 -- a seeds column on every table that lacked one
def table_block(f, label):
    t = txt[f]
    j = t.index(r'\label{%s}' % label)
    a = t.rindex(r'\begin{table}', 0, j)
    b = t.index(r'\end{table}', j) + len(r'\end{table}')
    return a, b

def add_seeds(label, value, mode='first', colspec=None, header='seeds'):
    for f in (V2, ST):
        a, b = table_block(f, label)
        blk = txt[f][a:b]
        ts = blk.index(r'\begin{tabular}{') + len(r'\begin{tabular}{')
        te = blk.index('}', ts)
        # colspec may contain p{..}; find the full balanced spec
        depth, k = 1, ts
        while depth:
            depth += {'{': 1, '}': -1}.get(blk[k], 0); k += 1
        spec = blk[ts:k-1]
        newspec = colspec if colspec else (spec[0] + 'c' + spec[1:] if mode == 'first' else spec + 'c')
        blk = blk[:ts] + newspec + blk[k-1:]
        out, in_tab, first_row = [], False, True
        for line in blk.split('\n'):
            code, com = line, ''
            if re.search(r'(?<!\\)%', line):
                m = re.search(r'(?<!\\)%', line); code, com = line[:m.start()], line[m.start():]
            if r'\begin{tabular}' in code: in_tab = True
            if in_tab and '&' in code and code.rstrip().endswith('\\\\'):
                val = header if first_row else value
                first_row = False
                if mode == 'first':
                    i = code.index('&')
                    code = code[:i] + '& ' + val + ' ' + code[i:]
                else:
                    i = code.rstrip().rindex('\\\\')
                    code = code[:i].rstrip() + ' & ' + val + ' ' + code[i:]
                line = code + com
            out.append(line)
        blk = '\n'.join(out)
        txt[f] = txt[f][:a] + blk + txt[f][b:]
        LOG.append('M6-seeds-%s\t%s\tok' % (label, os.path.basename(f)))

add_seeds('tab:ladder', '3')
add_seeds('tab:ledger-kstar', '3')
add_seeds('tab:maze-del', '3')
add_seeds('tab:verb', '3', mode='last')
add_seeds('tab:cmaze2', '3')
add_seeds('tab:cmaze1', '3')
add_seeds('tab:rivals-del', '1 / 3', header='seeds (s / gap)')
add_seeds('tab:unlearn', '3', mode='last')
add_seeds('tab:k5', '3')
add_seeds('tab:1x', '3')
add_seeds('tab:encloop', '3')
add_seeds('tab:crr', '3')
both(r"""seed 0.} Data rows read and bytes rewritten are seed-invariant. ``Forget $-$ retain'' is the
feature-MSE gap of the exact retrain, 3-seed mean.""",
     r"""seed 0.} Data rows read and bytes rewritten are seed-invariant. ``Forget $-$ retain'' is the
feature-MSE gap of the exact retrain, 3-seed mean; the seeds column gives the count for the
seconds and for the gap.""", tag='M6-rivals-del-caption')

# =====================================================================================
# M7(e) -- internal process language that renders in the PDF
rep(V2, 'corners per seed, with every charter control collapsing', 'corners per seed, with every pre-declared landscape control collapsing', tag='M7e-charter')
rep(ST, 'corners per seed, while every charter control collapses', 'corners per seed, while every pre-declared landscape control collapses', tag='M7e-charter')
rep(V2, r"""of written items is a printed \emph{floor}, never a rival. Charter controls
(\texttt{landscape\_removed},""", r"""of written items is a printed \emph{floor}, never a rival. Pre-declared landscape controls
(\texttt{landscape\_removed},""", tag='M7e-charter')
rep(ST, r"""store, the charter controls (\texttt{landscape\_removed},""", r"""store, the pre-declared landscape controls (\texttt{landscape\_removed},""", tag='M7e-charter')
both('The charter controls do not collapse: on the seen row', 'The pre-declared landscape controls do not collapse: on the seen row', tag='M7e-charter')
both('charter controls and blind columns are printed', 'pre-declared landscape controls and blind columns are printed', tag='M7e-charter')
both('and the charter controls do\n  not collapse', 'and the landscape controls do\n  not collapse', tag='M7e-charter')
both('the two charter controls as paired deltas', 'the two pre-declared landscape controls as paired deltas', tag='M7e-charter')
both('row is claimed for the store: the charter controls do not collapse.}', 'row is claimed for the store: the pre-declared landscape controls do not collapse.}', tag='M7e-charter')

both(r'\paragraph{The bars.} BAR 1 (', r'\paragraph{The pre-registered criteria.} Criterion 1 (', tag='M7e-BAR')
both('BAR 2 (store gain', 'Criterion 2 (store gain', tag='M7e-BAR')
both('BAR 3 (controls', 'Criterion 3 (controls', tag='M7e-BAR')
both('BAR 5 (store', 'Criterion 5 (store', tag='M7e-BAR')
rep(V2, 'with the pending scored BAR\nD row of the longer leg.', 'with the pending scored row of pre-registered\ncriterion D on the longer leg.', tag='M7e-BAR')
rep(ST, 'with the pending scored BAR D row of the longer leg.', 'with the pending scored row of pre-registered criterion D on the longer leg.', tag='M7e-BAR')
both('BAR D and its decomposition paragraph', 'Criterion D and its decomposition paragraph', tag='M7e-BAR')

rep(V2, 'Several rows are pending a later wave; each is a', 'Several rows are pending further runs; each is a', tag='M7e-wave')
rep(ST, 'Several rows await a later wave; each is a', 'Several rows await further runs; each is a', tag='M7e-wave')
both(r'(placeholder figure; wave 18 regenerates).}', r'(placeholder figure).}', tag='M7e-wave')
rep(V2, 'The conjunction the task first named, red', 'The conjunction first named in the design, red', tag='M7e-task')
rep(ST, 'the conjunction the task first named, red', 'the conjunction first named in the design, red', tag='M7e-task')
rep(V2, 'because they are dropped from paper rows by ruling, and because', 'because they are not scored rows of this study, and because', tag='M7e-ruling')
rep(ST, 'both because a ruling drops them from paper rows and because', 'both because they are not scored rows of this study and because', tag='M7e-ruling')
rep(V2, 'Dropped from paper rows by ruling,\nand printed here', 'Not a scored row of this study,\nand printed here', tag='M7e-ruling')
rep(ST, "rival's head, dropped from paper rows by ruling, is printed here.", "rival's head, not a scored row of this study, is printed here.", tag='M7e-ruling')
rep(V2, 'By the\none-$K$ ruling it is', 'By the\none-$K$ rule of this paper it is', tag='M7e-ruling')
rep(ST, 'Under\nthe one-$K$ ruling it is', 'Under\nthe one-$K$ rule of this paper it is', tag='M7e-ruling')
both(r"""\TODO{no artefact for the
``$\sim$1/7$\times$'' byte ratio of the composition-only store named in the task spine; the
artefacts give the two totals printed here and in Table~\ref{tab:ledger-kstar}, and this spoke
does not divide them.}""", r"""\TODO{no artefact prints the byte ratio of the composition-only store to the single design;
the artefacts give the two totals printed here and in Table~\ref{tab:ledger-kstar}, and they are
not divided here.}""", tag='M7e-taskspine')
both(r'\TODO{AI use statement --- the Head to confirm which', r'\TODO{AI use statement --- the authors to confirm which', tag='M7e-head')
rep(V2, r"""declared
not-runs, corrections and the per-cell provenance conventions are in
Appendices~\ref{app:notrun}--\ref{app:corrections}.""", r"""declared
not-runs and the per-cell provenance conventions are in
Appendices~\ref{app:conventions} and~\ref{app:notrun}.""", tag='M7e-corrections-ref')
rep(ST, r"""Appendices~\ref{app:notrun}--\ref{app:corrections} give the declared not-runs, corrections and
per-cell provenance conventions.""", r"""Appendices~\ref{app:conventions} and~\ref{app:notrun} give the per-cell provenance conventions and
the declared not-runs.""", tag='M7e-corrections-ref')
both(r"""within-maze lookup floor. Placeholder figure drawn from wave-14 data; wave 18 % (no result number on this line: wave labels / file names only)
regenerates.}""", r"""within-maze lookup floor. Placeholder figure.}""", tag='M7e-wave')
both(r"""(placeholder).} Pending wave 16; drawn as a marked placeholder from wave-14 data until then. It % (no result number on this line: wave labels / file names only)
may show""", r"""(placeholder).} Pending; drawn as a marked placeholder until its rows land. It
may show""", tag='M7e-wave')
both(r"""\section{Maze world-model rows (pending wave 16)} % (no result number on this line: wave labels / file names only)""",
     r"""\section{Maze world-model rows (pending)}""", tag='M7e-wave')
both(r"""and states that it may land as a negative. Wave-15 rows: the coverage control % (no result number on this line: wave labels / file names only)""",
     r"""and states that it may land as a negative. Pending Shapes3D and maze rows: the coverage control""", tag='M7e-wave')
both(r"""(Appendix~\ref{app:mazewm}). Wave-16 rows: every cell of Table~\ref{tab:cmazet}; the configuration ladder and budget; the fallback route task; the declared % (no result number on this line: wave labels / file names only)
negatives (Appendix~\ref{app:mazewm}). Figure slots are drawn by the figures spoke and are marked as""",
     r"""(Appendix~\ref{app:mazewm}). Pending maze world-model rows: every cell of Tables~\ref{tab:cmazet} and~\ref{tab:cmazeo}; the configuration ladder and budget; the fallback route task; the declared
negatives (Appendix~\ref{app:mazewm}). Figure slots are marked as""", tag='M7e-wave')

both(r"""\section{CMAZE-2 in full (w9 request layer, TEST, 3 of 3 scored seeds)}""",
     r"""\section{CMAZE-2 in full (original request layer, TEST, 3 of 3 scored seeds)}""", tag='M7e-w9')

# Corrections appendix: out of the submission PDF, kept in the source for the record
both(r"""% ---------------------------------------------------------------------------------------
\section{Corrections absorbed}
\label{app:corrections}""", r"""% ---------------------------------------------------------------------------------------
% REFEREE-v2 M7(e): the "Corrections absorbed" appendix corrects earlier internal write-ups a
% reader has never seen. It is held OUT of the submission PDF by \iffalse and kept in the
% source for the record (provenance comments unchanged).
\iffalse
\section{Corrections absorbed}
\label{app:corrections}""", tag='M7e-corrections')
for f in (V2, ST):
    t = txt[f]
    assert t.rstrip().endswith(r'\end{document}')
    i = t.rindex(r'\end{document}')
    txt[f] = t[:i] + '\\fi\n\n' + t[i:]
    LOG.append('M7e-corrections-fi\t%s\t1' % os.path.basename(f))

for f in (V2, ST):
    open(f, 'w').write(txt[f])
print('\n'.join(LOG))
print('edits applied:', len(LOG))
