#!/usr/bin/env python3
"""Apply the w18 placeholder-fill to one draft.  usage: apply_fill.py IN.tex OUT.tex
Every edit asserts that its anchor occurs exactly once, so a drifted draft fails loudly.
The four AWAITING placeholders (lace-poe-rows x2, bar-d-score x2) are never touched."""
import re, sys
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import fill_text as F

src, dst = sys.argv[1], sys.argv[2]
t = open(src).read()
log = []

def once(old, new, tag):
    global t
    n = t.count(old)
    assert n == 1, '%s: anchor occurs %d times' % (tag, n)
    t = t.replace(old, new)
    log.append(tag)

def macro_span(start):
    """Return the end index (exclusive) of the balanced {...} that opens at t[start]=='{',
    skipping LaTeX comments (from an unescaped % to end of line)."""
    assert t[start] == '{'
    k, depth = start + 1, 1
    while depth:
        ch = t[k]
        if ch == '%' and t[k - 1] != '\\':
            k = t.index('\n', k)
            continue
        if ch == '{' and t[k - 1] != '\\':
            depth += 1
        elif ch == '}' and t[k - 1] != '\\':
            depth -= 1
        k += 1
    return k

def placeholder(prefix, new, tag, lead=''):
    """Replace the whole \\PLACEHOLDER{...} whose argument starts with `prefix`."""
    global t
    key = lead + '\\PLACEHOLDER{' + prefix
    n = t.count(key)
    assert n == 1, '%s: placeholder prefix occurs %d times' % (tag, n)
    i = t.index(key)
    j = macro_span(i + len(lead) + len('\\PLACEHOLDER'))
    t = t[:i] + new + t[j:]
    log.append(tag)

def span(start_anchor, end_anchor, new, tag, keep_end=True):
    """Replace from start_anchor up to (not including) the next end_anchor."""
    global t
    assert t.count(start_anchor) == 1, '%s: start anchor count %d' % (tag, t.count(start_anchor))
    i = t.index(start_anchor)
    j = t.index(end_anchor, i + len(start_anchor))
    if not keep_end:
        j += len(end_anchor)
    t = t[:i] + new + t[j:]
    log.append(tag)

# ---- header comment (same number of lines, so the preamble offsets of the checker hold)
once(F.HEADER_OLD, F.HEADER_NEW, 'header')

# ---- Shapes3D, w15
placeholder('w15 \\texttt{coverage-control}', F.COVERAGE, 'P4 coverage-control')
once(F.LADDER_ROW_OLD, F.LADDER_ROW_NEW, 'P4 ladder row')
once(F.DEL_ROW_OLD, F.DEL_ROW_NEW, 'P5 tab:del seeds 1-2')
once(F.DEL_CAP_OLD, F.DEL_CAP_NEW, 'P5 tab:del caption')
once(F.DEL_TITLE_OLD, F.DEL_TITLE_NEW, 'P5 tab:del caption title')
placeholder('w15 \\texttt{single-design-seeds} (\\texttt{single-design} scored runs', F.DEL_PROSE, 'P6 deletion prose')
placeholder('w15 \\texttt{maze-deletion-2b}', F.MAZE2B, 'P8 maze-deletion-2b')
placeholder('w15 \\texttt{crossover-seeds}', F.KCURVE, 'P9 crossover-seeds')
placeholder('w15 \\texttt{single-design-seeds}: the rival retrain-without', F.RIVALS, 'P10 rival cost seeds 1-2')

_h = '\\section{Maze refusal displays (retired)}'
assert t.count(_h) == 1
_i = t.index(_h)
_sep = t.rindex('\n% ---', 0, _i) + 1  # start of the separator line before the heading
t = t[:_sep] + F.MAZE2B_APP + '\n\n' + t[_sep:]
log.append('P8 CMAZE-2b table in app:verb')

# ---- maze world-model, w15 harness + w16
placeholder('w15 \\texttt{cmaze-t-harness}', F.HARNESS, 'P12 harness smokes')
span('\\begin{table}[h]\n\\centering\\scriptsize\n\\caption{\\textbf{Maze world-model rows, pending: CMAZE-T',
     '\\end{table}', F.CMAZET, 'P13-P20 tab:cmazet', keep_end=False)
span('\\begin{table}[h]\n\\centering\\scriptsize\n\\caption{\\textbf{Maze world-model rows, pending: CMAZE-O',
     '\\end{table}', F.CMAZEO, 'P21-P27 tab:cmazeo', keep_end=False)
# P28-P30: the three consecutive placeholders after the tables
i0 = t.index('\\noindent\\PLACEHOLDER{w16 \\texttt{maze-wm} configuration ladder')
assert t.count('\\noindent\\PLACEHOLDER{w16 \\texttt{maze-wm} configuration ladder') == 1
k = t.index('\\PLACEHOLDER{w16 declared negatives', i0)
j = macro_span(k + len('\\PLACEHOLDER'))
t = t[:i0] + F.AFTER_TABLES + t[j:]
log.append('P28-P30 ladder/budget, fallback, negatives')

# ---- headings and the prose that said "pending"
span('\\subsection{World-model rows: transition and observation generation (pending)}',
     '% ====', F.MAZEWM_MAIN, 'sec:maze-wm main text')
once('\\section{Maze world-model rows (pending)}', '\\section{Maze world-model rows}', 'app heading')
span('\\section{Placeholder index}', '% ----', F.PLACEHOLDER_INDEX, 'placeholder index')
if 'observation instruments whose rows are pending (\\S\\ref{sec:maze-wm}).' in t:
    once('observation instruments whose rows are pending (\\S\\ref{sec:maze-wm}).',
         'observation instruments whose rows are in \\S\\ref{sec:maze-wm}.', 'sec:maze env list')
else:
    once('transition and observation instruments with rows pending (\\S\\ref{sec:maze-wm}).',
         'transition and observation instruments, reported in \\S\\ref{sec:maze-wm}.', 'sec:maze env list')

# ---- abstract, contribution 3, thickness: scope "composition does not transfer" to routes
once('composition does not transfer through the store: moving',
     'composition on routes does not transfer through the store: moving', 'abstract scope')
once('so the composition operator belongs to the interface.\n% SRC: w14-read-verb/FINAL-REPORT.md\n',
     'so the composition operator belongs to the interface.\n% SRC: w14-read-verb/FINAL-REPORT.md\n'
     'On a transition instrument over the same mazes the store does compose held-out $n$-way\n'
     'transitions, 0.7246 against 0.4035 for the best matched rival over three seeds with every % SRC: w16-maze-wm/FINDINGS.md\n'
     'landscape control collapsing, although no matched arm, the store included, yields a valid\n'
     'chained rollout.\n', 'abstract CMAZE-T')
once('where only deletion transfers through the\n  store.}',
     'where deletion transfers through the\n  store and composition transfers only on transitions.}', 'contribution 3 title')
once('floor more than it lifts the store (\\S\\ref{sec:maze}).',
     'floor more than it lifts the store (\\S\\ref{sec:maze}). On a transition instrument the store\n'
     '  carries a composition row with every landscape control collapsing, and it survives rivals\n'
     '  trained ten times longer; no matched arm\'s chained rollout is valid; and on an observation\n'
     '  instrument the store gate holds while the margin over rivals is withdrawn (\\S\\ref{sec:maze-wm}).',
     'contribution 3 body')
once('the store and a positive for the interface.',
     'the store and a positive for the interface. A transition instrument on the same mazes adds\n'
     'one store row at three seeds (\\S\\ref{sec:maze-wm}).', 'thickness')

# ---- figure captions (the maze figures are regenerated from the landed data)
once('per arm and seed on the unseen-pair row, with the single-decode pipeline ceiling and the\n'
     'within-maze lookup floor. Placeholder figure.}{fig:verb}',
     'per arm and seed on the unseen-pair row, with the single-decode pipeline ceiling and the\n'
     'within-maze lookup floor. CMAZE-2b only, 3 seeds.}{fig:verb}', 'fig:verb caption')
span('\\figslot{fig_maze_wm}{', '{fig:maze-wm}',
     '\\figslot{fig_maze_wm}{\\textbf{CMAZE-T, Config A1, TEST, 3 seeds (Table~\\ref{tab:cmazet}).} (a) Unseen\n'
     '$n$-way: the store against the matched rivals, the four charter-C controls and the cross-maze\n'
     'lookup floor, per seed. (b) Rollout validity is zero for the store and every matched rival; the\n'
     'per-step exactness that compounds over the chained steps is drawn beside it. (c) Post-deletion\n'
     'composition on the surviving colours: the downdated store (equal to the store fitted without)\n'
     'against the rival retrained without, value and conjunction scopes. CMAZE-T only.}', 'fig:maze-wm caption')

once('the number of seeds is printed per point. Placeholder\nuntil the crossover seeds land.}{fig:crossover}',
     'the number of seeds is printed per point. Placeholder\nfigure, drawn from the single-seed middle rows; the three-seed curve is Table~\\ref{tab:kcurve3}.}{fig:crossover}',
     'fig:crossover caption')
once('rivals: retrain without). Placeholder until the seed-1--2 deletion rows land.}{fig:deletion}',
     'rivals: retrain without). Placeholder figure, seed 0 only; the seed-1--2 rows are in Table~\\ref{tab:del} and Table~\\ref{tab:rivals-del-seeds}.}{fig:deletion}',
     'fig:deletion caption')

# ---- guard: the four awaiting placeholders survive byte-identical, and nothing else remains
args = re.findall(r'\\PLACEHOLDER\{(w1[56][^\n]*)', '\n'.join(l.split('%')[0] if not l.lstrip().startswith('%') else '' for l in t.split('\n')))
open(dst, 'w').write(t)
print('applied %d edits: %s' % (len(log), '; '.join(log)))
