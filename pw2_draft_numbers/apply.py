"""usage: python3 apply.py V2_IN STYLE_IN V2_OUT STYLE_OUT -- applies edits.E in order; every `old` must occur exactly once."""
import sys, re
from edits import E
v2, st = open(sys.argv[1]).read(), open(sys.argv[2]).read()
fails = []
for e in E:
    for scope, old, new in e['pairs']:
        for name in (['v2', 'style'] if scope == 'both' else [scope]):
            t = v2 if name == 'v2' else st
            n = t.count(old)
            if n != 1:
                fails.append((e['id'], name, n, old[:70])); continue
            last = new.split('\n')[-1]
            j = t.index(old) + len(old)
            if not old.lstrip().startswith('%') and re.search(r'(?<!\\)%', last) and j < len(t) and t[j] != '\n':
                fails.append((e['id'], name, 'COMMENT SWALLOWS REST OF LINE', repr(t[j:j+60]))); continue
            t = t.replace(old, new)
            if name == 'v2': v2 = t
            else: st = t
for f in fails: print('FAIL', f)
if fails: sys.exit(1)
open(sys.argv[3], 'w').write(v2); open(sys.argv[4], 'w').write(st)
print('applied', sum(len(e['pairs']) for e in E), 'pairs from', len(E), 'edits')
