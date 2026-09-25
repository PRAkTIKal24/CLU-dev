"""usage: python3 worddiff.py BEFORE.txt AFTER.txt -- rendered word diff (PDF text, page markers and hyphen breaks
removed); prints every replaced / deleted / inserted run, so a swallowed sentence shows as a long deletion."""
import sys, re, difflib
def words(p):
    t = open(p).read()
    t = re.sub(r'=== PAGE \d+ ===', ' ', t)
    t = re.sub(r'-\n', '', t)
    return t.split()
a, b = words(sys.argv[1]), words(sys.argv[2])
sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
n = 0
for op, i1, i2, j1, j2 in sm.get_opcodes():
    if op == 'equal': continue
    n += 1
    print('%s | -%d +%d | - %s\n    + %s' % (op, i2 - i1, j2 - j1, ' '.join(a[i1:i2])[:600], ' '.join(b[j1:j2])[:600]))
print('RUNS', n, 'words before', len(a), 'after', len(b))
