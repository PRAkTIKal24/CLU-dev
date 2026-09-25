"""Writes FINAL-INDEX-template.md: the pw1-draft-final index re-run after the template pass, with a before -> after column
for every page number.
usage: python make_index_template.py BEFORE_FINALDIR AFTER_FINALDIR OUT.md
Both dirs are run_final.sh outputs (FIGTAB-INDEX.json, PAGES-*.json, PH-*.json, COMPILE-*.txt, JOB1-compare.txt).
"""
import json
import os
import re
import sys

B, A, OUT = sys.argv[1:4]
D = ('DRAFT-v2', 'DRAFT-v2-style')
L = []


def w(s=''):
    L.append(s)


def js(d, p):
    return json.load(open(os.path.join(d, p)))


def arrow(x, y, fmt='%s'):
    return (fmt % x) if x == y else ('**%s → %s**' % (fmt % x, fmt % y))


w('# FINAL-INDEX-template — the drafts after the ICLR 2027 template pass (pw1-draft-template)')
w()
w('Before = the drafts as installed by pw1-draft-final (its `final-125012/`; sources cmp-equal to this pass\'s snapshot, labels equal). '
  'After = this pass\'s installed drafts (`%s/`). Same tooling (`pw1_draft_final/run_final.sh --inplace`). A page number that moved is printed **before → after**.' % os.path.basename(A))
w()
w('## 1. Compile (after)')
w()
w('```')
for f in ('COMPILE-scratch.txt', 'COMPILE-inplace.txt', 'JOB1-compare.txt'):
    w('# ' + f)
    w(open(os.path.join(A, f)).read().rstrip())
w('```')
w()
# ---- figures / tables
fb, fa = js(B, 'FIGTAB-INDEX.json'), js(A, 'FIGTAB-INDEX.json')
moved = 0
for kind, title in (('figure', 'Figures'), ('table', 'Tables')):
    w('## 2%s. %s (number, label, caption lead, page; both drafts)' % ('a' if kind == 'figure' else 'b', title))
    w()
    w('| DRAFT-v2 no. | label | caption, first line (bold lead) | DRAFT-v2 page | style no. | style page |')
    w('|---|---|---|---|---|---|')
    for lab, x in fa['DRAFT-v2']['items'].items():
        if x['kind'] != kind:
            continue
        xb = fb['DRAFT-v2']['items'][lab]
        s, sb = fa['DRAFT-v2-style']['items'][lab], fb['DRAFT-v2-style']['items'][lab]
        assert x['num'] == xb['num'] and s['num'] == sb['num'], lab
        moved += (x['page'] != xb['page']) + (s['page'] != sb['page'])
        lead = re.sub(r'\s+', ' ', x['lead']).replace('|', '/')[:110]
        w('| %s | `%s` | %s | %s | %s | %s |' % (x['num'], lab, lead, arrow(xb['page'], x['page']), s['num'], arrow(sb['page'], s['page'])))
    w()
w('Float numbers: identical before and after in both drafts (asserted). Float pages moved: %d. lof/lot cross-check after: '
  'DRAFT-v2 lo_without_label %s, label_without_lo %s; style %s, %s.' % (
      moved, fa['DRAFT-v2']['lo_without_label'], fa['DRAFT-v2']['label_without_lo'],
      fa['DRAFT-v2-style']['lo_without_label'], fa['DRAFT-v2-style']['label_without_lo']))
w()
# ---- pages per section
w('## 3. Pages per section (marker method C: the shipped page, inside the heading; lengths in pages)')
w()
w('| section | title | DRAFT-v2 start | DRAFT-v2 length | style start | style length |')
w('|---|---|---|---|---|---|')
P = {(t, d): js(t, 'PAGES-%s.json' % d) for t in (B, A) for d in D}
f2 = lambda x: '' if x is None else '%.2f' % x
for rb, ra, sb, sa in zip(P[B, 'DRAFT-v2']['rows'], P[A, 'DRAFT-v2']['rows'], P[B, 'DRAFT-v2-style']['rows'], P[A, 'DRAFT-v2-style']['rows']):
    assert rb['title'] == ra['title'] == sb['title'] == sa['title']
    t = re.sub(r'\s+', ' ', ra['title']).replace('|', '/')[:70]
    ind = '&nbsp;&nbsp;' if ra['kind'].startswith('subsection') else ''
    lab = ra['num'] or {'subsubsection*': '(stmt)', 'bibliography': '(refs)', 'end': '(end)'}.get(ra['kind'], '')
    w('| %s%s | %s | %s | %s | %s | %s |' % (ind, lab, t, arrow(rb['page'], ra['page']), arrow(f2(rb['length']), f2(ra['length'])),
                                          arrow(sb['page'], sa['page']), arrow(f2(sb['length']), f2(sa['length']))))
w()
w('## 4. Main text against the 9-page limit')
w()
w('| draft | main text, pages (title → "AI use statement") | ends on page | over 9 pp | statements, pages | references | appendices | total pages |')
w('|---|---|---|---|---|---|---|---|')
for d in D:
    cells = []
    for t in (B, A):
        p = P[t, d]
        f = lambda s: next(r for r in p['rows'] if r['title'].startswith(s))
        ai, bib, app = f('AI use statement'), f('References'), f('Appendices')
        end = [r for r in p['rows'] if r['kind'] == 'end'][0]
        cells.append(('%.2f' % ai['coord'], ai['page'], '%.2f' % (ai['coord'] - 9), '%.2f' % (bib['coord'] - ai['coord']),
                      'pp. %d–%d' % (bib['page'], app['page'] - 1), 'pp. %d–%d' % (app['page'], end['page']), p['pages_total']))
    w('| %s | %s |' % (d, ' | '.join(arrow(x, y) for x, y in zip(*cells))))
w()
# ---- placeholders
pb_, pa_ = {d: js(B, 'PH-%s.json' % d) for d in D}, {d: js(A, 'PH-%s.json' % d) for d in D}
w('## 5. `\\TODO` / `\\PLACEHOLDER` (by argument)')
w()
for d in D:
    same = [x['arg'] for x in pb_[d]['items']] == [x['arg'] for x in pa_[d]['items']]
    w('- %s: %s → %s; arguments, in order, %s; lines %s' % (d, pb_[d]['count_args'], pa_[d]['count_args'],
                                                          'BYTE-IDENTICAL' if same else 'DIFFER',
                                                          ', '.join(arrow(x['line'], y['line']) for x, y in zip(pb_[d]['items'], pa_[d]['items']))))
w()
open(OUT, 'w').write('\n'.join(L) + '\n')
print('written', OUT, 'float pages moved:', moved)
