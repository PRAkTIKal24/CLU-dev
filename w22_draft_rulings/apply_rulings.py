#!/usr/bin/env python3
"""w22 spoke/draft-rulings: apply the Head's w20/w21 rulings (+ AMENDMENT-1) to both drafts.

usage: apply_rulings.py <in dir> <out dir>
Reads DRAFT-v2.tex and DRAFT-v2-style.tex from <in dir>, writes the edited pair to <out dir>.
Every region is located by regex anchors that must resolve in BOTH drafts (a start anchor that
matches exactly once in the file, an end anchor = first match at/after the start); the new text is
the same in both drafts (texts.py), so the drafts stay content-identical. Any anchor miss aborts
before anything is written. A log of every replaced region (old text, both drafts) is written to
<out dir>/REPLACED-REGIONS.md for review.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import texts as T  # noqa: E402


def uniq(lines, pat, where):
    hits = [i for i, l in enumerate(lines) if re.search(pat, l)]
    if len(hits) != 1:
        raise SystemExit('ANCHOR %r: %d hits in %s (need 1): %s' % (pat, len(hits), where, hits[:6]))
    return hits[0]


def first_after(lines, pat, start, where):
    for i in range(start, len(lines)):
        if re.search(pat, lines[i]):
            return i
    raise SystemExit('END ANCHOR %r not found after line %d in %s' % (pat, start + 1, where))


# (name, start_pat, start_offset, end_pat, end_offset, new_text)
#  region = lines[s+so : e+eo+1]  (inclusive), where s = unique start match, e = first end match >= s
EDITS = [
    ('abstract-shapes', r'^On its 109 held-out corners the store, fitted on 131,072 rows', 0,
     r'% SRC: w13-referee-2/REFEREE-v1.md; w13-seeds-and-tables/TABLES.md\s*$', 0, T.ABSTRACT_SHAPES),
    ('abstract-maze-TODO', r'On a transition instrument over the same mazes the store reads 0\.7246', 0,
     r'included, yields a valid chained rollout\.', 0, T.ABSTRACT_MAZE),
    ('contributions', r'^\\item \\(textbf|emph)\{Composition at a stated partition', 0,
     r'^\\item \\(textbf|emph)\{Two ablations and a leaving note', -1, T.CONTRIB),
    ('thickness', r'^\\noindent\\(textbf|emph)\{Thickness, stated in the introduction', 0,
     r'marked placeholder whose text names the row', 0, T.THICKNESS),
    ('object-deletion', r'^\\paragraph\{The retained statistic and exact deletion\.\}', 0,
     r'^\\end\{quote\}', 0, T.OBJECT_DELETION),
    ('t7-intro', r'^\\label\{sec:shapes-t7\}', 2, r'^\\begin\{table\}\[t\]', -2, T.T7_INTRO),
    ('t7-table', r'\\caption\{\\textbf\{Composition at \$\\Kstar=32\$, Shapes3D value hold-out, TEST', -2,
     r'^\\end\{table\}', 0, T.T7_TABLE),
    ('t7-rendered-panels', r'^% ---- w19 draft-figure-slots: T7 success / failure panels', 0,
     r'^% ENDSRC', 0, T.T7_PANELS_RETIRED),
    ('two-rival-sets', r'^\\paragraph\{Two rival sets, and which is stronger\.\}', 0,
     r'it is a replication, not a scored row, and an MLP is the only family fitted at that size', 0, T.EARLIER_SET),
    ('ladder-intro', r'^\\label\{sec:ladder\}', 2, r'rather than round away\. % SRC: w12b-nr6-and-kstar/FINDINGS\.md', 0,
     T.LADDER_INTRO),
    ('ladder-table', r'\\caption\{\\textbf\{The ladder at \$\\Kstar=32\$, \$n=3\$ seeds, distinct valid novel tuples per', -2,
     r'^\\end\{table\}', 0, T.LADDER_TABLE),
    ('ladder-paired', r'^On every rung and every seed,? the store.s score in Table~\\ref\{tab:ladder\}', 0,
     r'^% SRC: w12b-nr6-and-kstar/FINDINGS\.md\s*$', 0, T.LADDER_PAIRED),
    ('headline-steps-fig', r'^\\figslot\{fig_headline_steps\}', 0, r'\{fig:headline-steps\}', 0, T.HEADLINE_FIG_RETIRED),
    ('del-intro', r'^\\label\{sec:shapes-del\}', 2, r'^\\begin\{table\}\[t\]', -2, T.DEL_INTRO),
    ('del-table', r'\\caption\{\\textbf\{Exact \(store-scoped\) deletion on the single design, \$\\Kstar=32\$, TEST', -2,
     r'^\\paragraph\{The probe that bounds the claim\.\}', -2, T.DEL_TABLE),
    ('del-comparators', r'^\\label\{sec:del-comparators\}', 2, r'^\\subsection\{Refusal after deletion', -2, T.DEL_COMPARATORS),
    ('maze-wm', r'^\\label\{sec:maze-wm\}', 2,
     r'^% SRCBLOCK: w16-maze-wm/FINDINGS\.md; w16-maze-wm/MAZE-WM\.json', -2, T.MAZE_WM),
    ('maze-close', r'^The maze section therefore stands on deletion', 0, r'^% =====', -2, T.MAZE_CLOSE),
    ('limits-list', r'^\\label\{sec:limits-list\}', 2, r'^\\end\{enumerate\}', 0, T.LIMITS),
    ('leaving', r'^\\label\{sec:leaving\}', 2, r'% SRC: w10-physics-section-evidence/EVIDENCE\.md\s*$', 0, T.LEAVING),
    ('ethics', r'^\\subsubsection\*\{Ethics statement\}', 1, r'^\\subsubsection\*\{Reproducibility statement\}', -2, T.ETHICS),
    ('t7more', r'^\\label\{app:t7more\}', 1, r'^\\begin\{table\}\[h\]', -2, T.T7MORE),
    ('lace-poe-131k', r'^PoE \(own budget, not matched\) & 3 & 0 / 0 / 0 &', 0,
     r'^PoE \(own budget, not matched\) & 3 & 0 / 0 / 0 &', 0, T.LACE_POE_ROWS),
    ('k5-and-1x', r'^\\label\{app:k5\}', 1, r'^\\section\{The \$K\$-curve', -2, T.K5_APPENDIX),
    ('fig-deletion-caption', r'punches the hole; unlike the \(d\) rows of Table~\\ref\{tab:del\}, which carry the book across the basis', 0,
     r'change, the book here is refit\.', 0, T.FIGDEL_CAPTION_LINES),
    ('delbase', r'^\\paragraph\{A sphere hazard, measured\.\}', 0, r'^\\section\{Light encoder adaptation in full\}', -2,
     T.DELBASE),
    ('cmazet-caption', r'^\(\\S\\ref\{sec:maze-wm\}\); the symmetric-input rows are Table~\\ref\{tab:cmazet-sym\}, where no condition', 0,
     r'pair block, not the store, carries this table', 0, T.CMAZET_CAPTION_LINES),
    ('cmazet-sym-table', r'^% ---- w19 draft-figure-slots: w19 maze-t-symmetric rows, NOT LANDED', 0,
     r'^\\end\{table\}', 0, T.CMAZET_SYM),
    ('morelimits-comparators', r'^\\item \\(textbf|emph)\{Comparators not run', 0, r'^\\end\{enumerate\}', -1, T.MORELIMITS_COMP),
    ('placeholder-index', r'^\\label\{app:placeholders\}', 2, r'^\\section\{Declared not-runs\}', -3, T.PLACEHOLDER_INDEX),
    ('notruns', r'^\\label\{app:notrun\}', 2, r'^\\end\{itemize\}', 0, T.NOTRUNS),
]

# in-line substitutions: (name, regex, replacement, count) applied to the whole text, count must match
SUBS = [
    ('intro-approx-unlearning',
     r'misses\s+the\s+exact\s+answer\s+in\s+either\s+direction\s+\(\\S\\ref\{sec:del-comparators\}\)\.',
     r'is approximate by construction, not exact (\\S\\ref{sec:del-comparators}).', 1),
    ('hole-item-sentence',
     r'An\s+item\s+deletion(,\s+in\s+contrast,)?\s+makes\s+nothing\s+impossible:\s*0\s+of\s+164\s+requests\s+built\s+from\s+the\s+deleted\s+rows\s+are\s+certifiably\s+impossible,\s+on\s+all\s+three\s+seeds,\s+so\s+the\s+item\s+row\s+has\s+no\s+refusal\s+row\s+at\s+all\.\s+% SRC: w12-deletion-baselines/FINDINGS\.md',
     'An item deletion makes nothing impossible: on the store\'s own 131,072-row pool, deleting a random\n1\\\\% of items clears no value bit and punches no hole, so no request becomes certifiably impossible\nand the item row has no refusal row at all. % SRC: w20-shapes-data-matched/SHAPES-MATCHED.json', 1),
    ('c512-sentence',
     r'is\s+its\s+replication\.',
     'is its replication under the same asymmetric input, so it inherits that row\'s scope and is not\npromoted to a symmetric-input result.', 1),
]


def apply(text, where, log):
    lines = text.split('\n')
    for name, sp, so, ep, eo, new in EDITS:
        if new is None:
            continue
        s = uniq(lines, sp, where + ':' + name)
        e = first_after(lines, ep, s, where + ':' + name)
        a, b = s + so, e + eo
        if not (0 <= a <= b + 1 <= len(lines)):
            raise SystemExit('bad region %s %d..%d in %s' % (name, a, b, where))
        old = lines[a:b + 1]
        log.append('## %s — %s lines %d..%d\n```\n%s\n```\n' % (name, where, a + 1, b + 1, '\n'.join(old)))
        lines[a:b + 1] = new.strip('\n').split('\n')
    text = '\n'.join(lines)
    for name, pat, rep, cnt in SUBS:
        if pat is None or rep is None:
            continue
        new_text, n = re.subn(pat, rep, text)
        if n != cnt:
            raise SystemExit('SUB %s: %d matches in %s (need %d)' % (name, n, where, cnt))
        log.append('## sub %s — %s: %d replacement(s)\n' % (name, where, n))
        text = new_text
    return text


def main():
    src, dst = sys.argv[1], sys.argv[2]
    os.makedirs(dst, exist_ok=True)
    outs, log = {}, []
    for f in ('DRAFT-v2.tex', 'DRAFT-v2-style.tex'):
        outs[f] = apply(open(os.path.join(src, f)).read(), f, log)
    for f, t in outs.items():
        open(os.path.join(dst, f), 'w').write(t)
    open(os.path.join(dst, 'REPLACED-REGIONS.md'), 'w').write('\n'.join(log))
    print('written', dst)


if __name__ == '__main__':
    main()
