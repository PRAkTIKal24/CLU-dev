#!/usr/bin/env python3
"""Before -> after table (markdown) from edit.py's two REPORT.json files. usage: report.py V2.json STYLE.json OUT.md"""
import json, re, sys
from collections import Counter
a, b = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
def flat(s):
    s = '\n'.join(re.split(r'(?<!\\)%', l, 1)[0] for l in s.split('\n'))
    return ' '.join(s.split()).replace('|', '\\|')
out = ['| # | item | site | DRAFT-v2 line | style line | same text in both | before (DRAFT-v2, rendered source) | after (DRAFT-v2) | numbers leaving | numbers added |', '|---|---|---|---|---|---|---|---|---|---|']
for i, (x, y) in enumerate(zip(a, b), 1):
    assert x['what'] == y['what']
    same = 'yes' if (x['before'], x['after']) == (y['before'], y['after']) else ('after only' if x['after'] == y['after'] else 'no (style paraphrase kept)')
    out.append('| %d | %s | %s | %d | %d | %s | %s | %s | %s | %s |' % (i, x['item'], x['what'], x['line_before'], y['line_before'], same,
               flat(x['before']), flat(x['after']), ', '.join(x['numbers_leaving']) or '—', ', '.join(x['numbers_added']) or '—'))
c = Counter(str(x['item']) for x in a)
out += ['', 'Operations per item (each applied once in each draft): ' + '; '.join('item %s: %d' % kv for kv in sorted(c.items())) + '; total %d per draft.' % len(a)]
open(sys.argv[3], 'w').write('\n'.join(out) + '\n')
print('\n'.join(out[-1:]))
