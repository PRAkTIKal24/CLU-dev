"""Per-tell counts per section NOW (pw2_draft_tells/detect.py, unchanged), beside pw2-draft-tells' own before -> after
columns read from its TELLS-COUNT.md. usage: python3 tells_now.py V2.tex STYLE.tex TELLS-COUNT.md > TELLS-NOW.md"""
import os, re, sys
from collections import Counter
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'pw2_draft_tells'))
from detect import detect

V2, ST, TC = sys.argv[1:4]
names = {'T1': 'Triads', 'T2': 'Contrast constructions', 'T3': 'Self-narration of discipline', 'T4': 'Numbers in prose',
         'T5': 'Headings', 'T6': 'Coinages', 'T7': 'Rhythm'}
tc = open(TC).read()
old = {}
for T in names:
    m = re.search(r'## %s [^\n]*\n(.*?)(?=\n## |\Z)' % T, tc, re.S)
    if not m: continue
    for l in m.group(1).split('\n'):
        mm = re.match(r'\| (.+?) \| \**(\d+) → (\d+)\** ?(?:\[(\d+)h\])?(?: \*\*ROSE\*\*)?\** \| \**(\d+) → (\d+)\** ?(?:\[(\d+)h\])?(?: \*\*ROSE\*\*)?\** \|', l)
        if mm:
            sec = mm.group(1).replace('**', '').strip()
            old[(T, sec)] = (mm.group(2), mm.group(3), mm.group(4) or '0', mm.group(5), mm.group(6), mm.group(7) or '0')
now = {}
secs = []
for d, p in (('v2', V2), ('style', ST)):
    U, hits = detect(p)
    for u in U:
        if u['section'] not in secs: secs.append(u['section'])
    c = Counter((h['tell'], h['section']) for h in hits)
    ch = Counter((h['tell'], h['section']) for h in hits if h['hh'])
    now[d] = (c, ch)
out = ['# TELLS-NOW — the seven tells per section, re-measured on the installed drafts', '',
       'Detectors: `pw2_draft_tells/detect.py`, unchanged (rules in its docstring). Columns: pw2-draft-tells\' **before → after** '
       '(from its `TELLS-COUNT.md`) and **now** (this audit, the live drafts after pw2-draft-leftovers). `[nh]` = hits inside Head-held text. '
       'A "now" that differs from "after" is marked ◆ (a later spoke changed that section).', '']
tot_change = []
for T, nm in names.items():
    out += ['## %s %s' % (T, nm), '', '| section | DRAFT-v2 before → after → **now** | DRAFT-v2-style before → after → **now** |', '|---|---|---|']
    sums = [0] * 8
    for s in secs:
        o = old.get((T, s))
        cells = []
        for k, d in enumerate(('v2', 'style')):
            c, ch = now[d]
            n, nh = c[(T, s)], ch[(T, s)]
            if o:
                b, a, ah = o[3 * k], o[3 * k + 1], o[3 * k + 2]
                mark = ' ◆' if int(a) != n else ''
                if mark: tot_change.append((T, s, d, a, n))
                cells.append('%s → %s%s → **%d**%s%s' % (b, a, ' [%sh]' % ah if ah != '0' else '', n, ' [%dh]' % nh if nh else '', mark))
                sums[4 * k] += int(b); sums[4 * k + 1] += int(a)
            else:
                cells.append('(not in TELLS-COUNT) → **%d**%s' % (n, ' [%dh]' % nh if nh else ''))
            sums[4 * k + 2] += n; sums[4 * k + 3] += nh
        out.append('| %s | %s | %s |' % (s, cells[0], cells[1]))
    out.append('| **total** | %d → %d → **%d** [%dh] | %d → %d → **%d** [%dh] |' % tuple(sums))
    out.append('')
out += ['## Sections where "now" differs from pw2-draft-tells\' "after"', '']
if tot_change:
    out += ['| tell | section | draft | after | now |', '|---|---|---|---|---|'] + ['| %s | %s | %s | %s | %d |' % x for x in tot_change]
else:
    out.append('none')
print('\n'.join(out))
