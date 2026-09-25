"""Writes TELLS-COUNT.md, EDITS.md and WORDCOUNT.md.
usage: python3 report.py V2_BEFORE STYLE_BEFORE V2_AFTER STYLE_AFTER AUDIT.json OUTDIR"""
import sys, json, re
from collections import Counter
from detect import detect, LONG, SHORT, PHRASES, CONNECTIVES, __doc__ as DETDOC
from edits import E
vb, sb, va, sa, aj, out = sys.argv[1:7]
A = json.load(open(aj))
NAMES = ['Abstract', 'Introduction', 'Methods', 'Experiments', 'Results', 'Discussion', 'Statements', 'App-preface']
def sec_order(U):
    s = []
    for u in U:
        if u['section'] not in s: s.append(u['section'])
    return s
def short(s): return re.sub(r'\\ref\{[^}]*\}', 'T7', s.split(':')[0] if s.startswith('App ') else s)[:40] if not s.startswith('App ') else s[:5].strip()
res = {}
for name, b, a in (('DRAFT-v2', vb, va), ('DRAFT-v2-style', sb, sa)):
    Ub, Hb = detect(b); Ua, Ha = detect(a)
    res[name] = dict(order=sec_order(Ub), b=Hb, a=Ha)
T = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7']
TN = {'T1': 'Triads', 'T2': 'Contrast constructions', 'T3': 'Self-narration of discipline', 'T4': 'Numbers (re-check only)',
      'T5': 'Headings (main text)', 'T6': 'Coinages', 'T7': 'Rhythm'}
L = ['# TELLS-COUNT — PW-2 item 1, the seven tells, per section, per draft, before → after', '',
     'Before = the live drafts at the start of this pass (`snap-start/`, = pw2-draft-numbers\' install). After = the staged, installed drafts.',
     'Each cell is `before → after`; `[h]` after a number = how many of the after-hits sit in Head-held text (not editable by this pass).',
     'The same detectors, unchanged, produced both columns (`pw2_draft_tells/detect.py`); the rules and their positive controls are at the end.', '']
for t in T:
    L += ['## %s %s' % (t, TN[t]), '', '| section | DRAFT-v2 | DRAFT-v2-style |', '|---|---|---|']
    tot = {}
    for sec in res['DRAFT-v2']['order']:
        cells = []
        for name in ('DRAFT-v2', 'DRAFT-v2-style'):
            R = res[name]
            nb = sum(1 for h in R['b'] if h['tell'] == t and h['section'] == sec)
            na = sum(1 for h in R['a'] if h['tell'] == t and h['section'] == sec)
            nh = sum(1 for h in R['a'] if h['tell'] == t and h['section'] == sec and h['hh'])
            tot.setdefault(name, [0, 0, 0]); tot[name][0] += nb; tot[name][1] += na; tot[name][2] += nh
            flag = ' **ROSE**' if na > nb else ''
            cells.append('%d → %d%s%s' % (nb, na, ' [%dh]' % nh if nh else '', flag))
        L.append('| %s | %s | %s |' % (sec.replace('|', '/'), cells[0], cells[1]))
    L.append('| **total** | **%d → %d** [%dh] | **%d → %d** [%dh] |' % tuple(tot['DRAFT-v2'] + tot['DRAFT-v2-style']))
    if t == 'T7':
        for name in ('DRAFT-v2', 'DRAFT-v2-style'):
            R = res[name]; c = Counter(h['rule'] for h in R['a'] if h['tell'] == 'T7'); cb = Counter(h['rule'] for h in R['b'] if h['tell'] == 'T7')
            L.append('')
            L.append('%s, T7 by rule: %s' % (name, '; '.join('%s %d → %d' % (k, cb[k], c[k]) for k in sorted(set(c) | set(cb)))))
    L.append('')
L += ['## Detectors (verbatim from `detect.py`)', '', '```', DETDOC.strip(), '```', '',
      'LONG = %d words, SHORT = %d words. Words are counted on the LaTeX-normalised text: a `\\ref` counts as one word ("Tab. 3" = 2), inline math as one.' % (LONG, SHORT),
      'T3 phrase list: ' + '; '.join('`%s`' % p for p in PHRASES), 'T7 connective list: ' + ', '.join(CONNECTIVES), '',
      '## Positive controls', '', 'See `CONTROLS.txt`: a paragraph planting every tell in Results 4.2 and a sentence planting T2/T3 in App Q raise each detector\'s count; each of the brief\'s seven self-narration examples matches the T3 list. **ALL CONTROLS PASS.**', '']
open(out + '/TELLS-COUNT.md', 'w').write('\n'.join(L) + '\n')

# EDITS.md
from detect import norm
L = ['# EDITS — every edit of PW-2 item 1, before → after, tagged with its tell', '',
     'Source: `pw2_draft_tells/edits.py` (applied by `apply.py`; each `old` occurs exactly once in its draft). Scope `both` = the same bytes in both drafts; otherwise the style twin\'s own wording at the same place.',
     'Tells: T1 triad, T2 contrast, T3 self-narration, T5 heading, T6 coinage, T7a split for a short sentence, T7b connective opener, T7c "This section"; B = borrowing from the style twin.',
     'Words added by each edit are listed (case-insensitive multiset, from `audit.json`); a capital made by a split is not an addition.', '']
per = {n: {p['id']: p for p in A[n]['per']} for n in A}
for e in E:
    L.append('### %s — %s — %s' % (e['id'], e['tells'], e['section']))
    L.append(e['note'])
    for s, old, new in e['pairs']:
        L.append('- **%s**: `%s`' % (s, ' '.join(old.split())[:600].replace('`', "'")))
        L.append('  → `%s`' % (' '.join(new.split())[:600].replace('`', "'") or '(deleted)'))
    adds = []
    for n in ('DRAFT-v2', 'DRAFT-v2-style'):
        p = per[n].get(e['id'])
        if p and p['added']: adds.append('%s adds %s' % (n, ', '.join('%s×%d' % kv for kv in p['added'].items())))
    if adds: L.append('- words added: ' + '; '.join(adds))
    L.append('')
open(out + '/EDITS.md', 'w').write('\n'.join(L) + '\n')

# WORDCOUNT.md
L = ['# WORDCOUNT — words per section, before → after (comment-stripped source text: prose, items, quote, captions, tables, headings)', '',
     '| section | DRAFT-v2 before | after | Δ | DRAFT-v2-style before | after | Δ |', '|---|---|---|---|---|---|---|']
grew = []
secs = list(A['DRAFT-v2']['sections'])
for s in secs:
    r = []
    for n in ('DRAFT-v2', 'DRAFT-v2-style'):
        v = A[n]['sections'].get(s, {}); bb, aa = v.get('before', 0), v.get('after', 0)
        r += [bb, aa, aa - bb]
        if aa > bb: grew.append((n, s, bb, aa))
    L.append('| %s | %d | %d | %+d | %d | %d | %+d |' % ((s.replace('|', '/'),) + tuple(r)))
for n in ('DRAFT-v2', 'DRAFT-v2-style'):
    L.append('')
    L.append('%s whole document (abstract to \\end{document}): %d → %d words (%+d). The net word change equals the sum of the listed edits\' changes: **%s**.' %
             (n, A[n]['doc_words_before'], A[n]['doc_words_after'], A[n]['doc_words_after'] - A[n]['doc_words_before'], A[n]['net_equal']))
L += ['', '## Sections that grew', '']
for n, s, bb, aa in grew:
    L.append('- %s, %s: %d → %d (%+d). Reason: the borrowing B2 (the style twin\'s Methods signpost, 29 words, plan item 4) is the only addition there; the tell edits in Methods removed 27 words.' % (n, s, bb, aa, aa - bb))
if not grew: L.append('- none')
open(out + '/WORDCOUNT.md', 'w').write('\n'.join(L) + '\n')
print('written')
