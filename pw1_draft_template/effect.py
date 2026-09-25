"""Page effect of each variant against V0: total pages, the page on which the AI-use statement heading starts (the end of
the main text), and every \\newlabel whose page moved.
usage: python effect.py VARDIR  -> prints a markdown table and writes VARDIR/EFFECT.json
"""
import json
import os
import re
import sys

V = sys.argv[1]
LAB = re.compile(r'\\newlabel\{([^}]*)\}\{\{([^}]*)\}\{([^}]*)\}')


def labels(p):
    out = {}
    for l in open(p):
        m = LAB.match(l)
        if m and not m.group(1).endswith('@cref'):
            out[m.group(1)] = (m.group(2), m.group(3))
    return out


def stats(v, d):
    t = open(f'{V}/{v}/{d}.pdf.txt').read()
    pages = t.split('\n=== PAGE ')
    ai = next((i for i, p in enumerate(pages) if re.search(r'AI\s*USE STATEMENT', p)), None)
    n = len(pages) - 1
    return n, ai


rows = []
for v in sorted(os.listdir(V)):
    if not os.path.isdir(f'{V}/{v}') or v == 'V0':
        continue
    for d in ('DRAFT-v2', 'DRAFT-v2-style'):
        if v.startswith('L2-') and d != 'DRAFT-v2' or v.startswith('Ls-') and d != 'DRAFT-v2-style':
            continue
        b, a = labels(f'{V}/V0/{d}.labels.txt'), labels(f'{V}/{v}/{d}.labels.txt')
        moved = [(k, b[k][1], a[k][1]) for k in b if k in a and b[k][1] != a[k][1]]
        renum = [(k, b[k][0], a[k][0]) for k in b if k in a and b[k][0] != a[k][0]]
        (n0, ai0), (n1, ai1) = stats('V0', d), stats(v, d)
        rows.append(dict(variant=v, draft=d, pages=[n0, n1], ai_page=[ai0, ai1], moved=moved, renumbered=renum,
                         missing=sorted(set(b) ^ set(a))))
print('| variant | draft | total pages V0 -> variant | AI-use statement page V0 -> variant | labels whose page moved | labels renumbered |')
print('|---|---|---|---|---|---|')
for r in rows:
    mv = ', '.join(f'`{k}` {x}->{y}' for k, x, y in r['moved']) or '0'
    print(f"| {r['variant']} | {r['draft']} | {r['pages'][0]} -> {r['pages'][1]} | {r['ai_page'][0]} -> {r['ai_page'][1]} | "
          f"{len(r['moved'])}: {mv} | {len(r['renumbered'])} |")
json.dump(rows, open(f'{V}/EFFECT.json', 'w'), indent=1)
