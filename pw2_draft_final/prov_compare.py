"""Provenance findings now vs the PW-1 final state (pw1-draft-citations' installed drafts), check_provenance.py output
for both, compared as MULTISETS of findings with line numbers masked (numbers move between lines and floats in PW-2).
usage: python3 prov_compare.py PW1.txt NOW.txt  -> markdown; exit 1 if NOW has a finding PW-1 did not (a regression)"""
import re, sys
from collections import Counter


def parse(p):
    t = open(p).read()
    uns = Counter(); nf = Counter()
    sec = None
    for l in t.split('\n'):
        if l.startswith('== UNSOURCED'): sec = 'u'; continue
        if l.startswith('== NOT FOUND'): sec = 'n'; continue
        m = re.match(r'^(\d+) (.*)$', l)
        if m and sec == 'u': uns[m.group(2)] += 1
        elif m and sec == 'n': nf[m.group(2)] += 1
    chk = re.search(r'checked number tokens: (\d+)', t).group(1)
    return uns, nf, chk


a_u, a_n, a_c = parse(sys.argv[1]); b_u, b_n, b_c = parse(sys.argv[2])
print('| | PW-1 final | now |')
print('|---|---|---|')
print('| checked number tokens | %s | %s |' % (a_c, b_c))
print('| unsourced lines | %d | %d |' % (sum(a_u.values()), sum(b_u.values())))
print('| not-found tokens | %d | %d |' % (sum(a_n.values()), sum(b_n.values())))
reg = (b_u - a_u) + (b_n - a_n)
imp = (a_u - b_u) + (a_n - b_n)
print('')
print('- findings in both (line numbers masked): %d' % (sum((a_u & b_u).values()) + sum((a_n & b_n).values())))
print('- only now (regressions): %s' % (', '.join('`%s`' % k for k in reg.elements()) or 'none'))
print('- only in PW-1 (resolved since): %s' % (', '.join('`%s`' % k for k in imp.elements()) or 'none'))
print('- verdict: **%s**' % ('IDENTICAL' if not reg and not imp else ('IMPROVED' if not reg else 'REGRESSED')))
print('')
print('Findings now (line numbers masked):')
for k, v in list(b_u.items()) + list(b_n.items()):
    print('- `%s`%s' % (k, ' ×%d' % v if v > 1 else ''))
sys.exit(1 if reg else 0)
