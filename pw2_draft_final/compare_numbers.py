"""Compare this audit's independent number list (numclass.py JSON) with pw2-draft-leftovers' NUMBERS-LEFT.md (the
latest list, from pw2_draft_numbers' classifier with the lock-step class removed). Tokens are matched on
(line, number) with multiplicity; classes are compared on a coarse scale both taxonomies map onto:
HEAD-HELD | ID/PROTOCOL (identifier, protocol, definition) | ROUNDED | COUNT | NOT-A-NUMBER.
usage: python3 compare_numbers.py MINE.json NUMBERS-LEFT.md DRAFTNAME  -> markdown on stdout
"""
import json, re, sys
from collections import Counter

mine_p, theirs_p, name = sys.argv[1:4]
mine = json.load(open(mine_p))
md = open(theirs_p).read()
part = md.split('## %s' % name)[1].split('\n## ')[0]
part = part.split('### Every number')[1]
theirs = []
for l in part.split('\n'):
    m = re.match(r'\| (\d+) \| ([^|]*) \| (digit|word) \| `([^`]*)` \| ([^|]*) \|', l)
    if m: theirs.append(dict(line=int(m.group(1)), section=m.group(2).strip(), kind=m.group(3), tok=m.group(4), why=m.group(5).strip()))


def coarse_theirs(w):
    w = w.lower()
    if w.startswith('head-held'): return 'HEAD-HELD'
    if w.startswith('rounded'): return 'ROUNDED'
    if w.startswith('not a number'): return 'NOT-A-NUMBER'
    if 'seed count' in w or '3-seed mean' in w: return 'COUNT'   # "protocol / high-level seed count": a count of seeds
    if w.startswith(('identifier', 'protocol', 'definition')): return 'ID/PROTOCOL'
    if w.startswith('count'): return 'COUNT'
    return 'OTHER:' + w[:30]


def coarse_mine(c):
    return {'IDENTIFIER': 'ID/PROTOCOL', 'PROTOCOL/DEFINITION': 'ID/PROTOCOL'}.get(c, c)


km = Counter((t['line'], t['tok'].lower()) for t in mine)
kt = Counter((t['line'], t['tok'].lower()) for t in theirs)
only_m, only_t = km - kt, kt - km
print('### %s: token match' % name)
print('')
print('- this audit: %d tokens (%d digit, %d word); pw2-draft-leftovers NUMBERS-LEFT: %d tokens (%d digit, %d word)' % (
    len(mine), sum(t['kind'] == 'digit' for t in mine), sum(t['kind'] == 'word' for t in mine),
    len(theirs), sum(t['kind'] == 'digit' for t in theirs), sum(t['kind'] == 'word' for t in theirs)))
print('- matched on (line, number): %d; only in this audit: %d; only in NUMBERS-LEFT: %d' % (sum((km & kt).values()), sum(only_m.values()), sum(only_t.values())))
print('')
if only_m or only_t:
    print('| line | number | only in | this audit\'s class / their justification | code |')
    print('|---|---|---|---|---|')
    for (ln, tok), n in sorted(only_m.items()):
        t = next(x for x in mine if x['line'] == ln and x['tok'].lower() == tok)
        glued = t['kind'] == 'digit' and (t['word'] != t['tok'] or re.search(r'[A-Za-z]$', t['code'][:t['col']]))
        split = any(x['line'] == ln and x['tok'].lower() != tok and x['tok'].lower() in tok for x in theirs) or \
                any(x['line'] == ln and x['tok'].lower() == tok for x in theirs)
        why = ('tokeniser: digits glued to letters or a macro (their regex skips a digit preceded by a letter)' if glued else
               'tokeniser: a comma list or a repeated digit that the two tokenisers split or count differently' if split else
               ('word not in their word list' if t['kind'] == 'word' else 'REVIEW'))
        print('| %d | `%s` ×%d | this audit | %s: %s — **%s** | `%s` |' % (ln, tok, n, t['cls'], t['why'], why, ' '.join(t['code'][max(0, t['col'] - 30):t['col'] + 40].split()).replace('|', '\\|')))
    for (ln, tok), n in sorted(only_t.items()):
        t = next(x for x in theirs if x['line'] == ln and x['tok'].lower() == tok)
        sub = any(x['line'] == ln and tok in x['tok'].lower() and x['tok'].lower() != tok for x in mine) or \
              any(x['line'] == ln and tok in x['word'].lower() for x in mine)
        why = 'tokeniser: a piece of a token this audit lists whole' if sub else 'REVIEW'
        print('| %d | `%s` ×%d | NUMBERS-LEFT | %s — **%s** | |' % (ln, tok, n, t['why'], why))
    print('')
# class comparison on matched tokens, in order within (line, tok)
from collections import defaultdict
gm, gt = defaultdict(list), defaultdict(list)
for t in mine: gm[(t['line'], t['tok'].lower())].append(t)
for t in theirs: gt[(t['line'], t['tok'].lower())].append(t)
dis = []
agree = 0
for k in gm:
    for a, b in zip(gm[k], gt.get(k, [])):
        ca, cb = coarse_mine(a['cls']), coarse_theirs(b['why'])
        if ca == cb: agree += 1
        else: dis.append((k, a, b, ca, cb))
print('### %s: class agreement on matched tokens (coarse scale)' % name)
print('')
print('- agree: %d; disagree: %d' % (agree, len(dis)))
pairs = Counter((ca, cb) for _, _, _, ca, cb in dis)
for (ca, cb), n in sorted(pairs.items(), key=lambda x: -x[1]):
    print('  - this audit %s / NUMBERS-LEFT %s: %d' % (ca, cb, n))
print('')
if dis:
    print('| line | number | this audit | NUMBERS-LEFT | code |')
    print('|---|---|---|---|---|')
    for (ln, tok), a, b, ca, cb in sorted(dis, key=lambda x: x[0]):
        print('| %d | `%s` | %s — %s | %s | `%s` |' % (ln, a['tok'], a['cls'], a['why'], b['why'], ' '.join(a['code'][max(0, a['col'] - 30):a['col'] + 40].split()).replace('|', '\\|')))
