"""usage: python3 apply.py V2_IN STYLE_IN V2_OUT STYLE_OUT [--upto ID] -- applies edits.E in order; each `old` must occur
exactly once in its draft (asserted); refuses an edit whose new text ends in a % comment that would swallow the rest
of its source line. Prints every failure and exits 1 without writing if any."""
import sys, re
from edits import E
v2, st = open(sys.argv[1]).read(), open(sys.argv[2]).read()
fails = []; n = 0
for e in E:
    for scope, old, new in e['pairs']:
        for name in (['v2', 'style'] if scope == 'both' else [scope]):
            t = v2 if name == 'v2' else st
            c = t.count(old)
            if c != 1:
                fails.append((e['id'], name, c, old[:80])); continue
            j = t.index(old) + len(old)
            last = new.split('\n')[-1]
            if re.search(r'(?<!\\)%', last) and not re.search(r'(?<!\\)%', old.split('\n')[-1]) and j < len(t) and t[j] != '\n':
                fails.append((e['id'], name, 'COMMENT SWALLOWS REST OF LINE', t[j:j + 50])); continue
            t = t.replace(old, new); n += 1
            if name == 'v2': v2 = t
            else: st = t
for f in fails: print('FAIL', f)
if fails: sys.exit(1)
open(sys.argv[3], 'w').write(v2); open(sys.argv[4], 'w').write(st)
print('applied', n, 'replacements from', len(E), 'edits')
