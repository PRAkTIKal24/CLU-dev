#!/usr/bin/env python3
"""Re-read check_provenance.py output: a not-found token on line N counts as DERIVED-VERIFIED only if a
comment-only line directly below N states '<tok> is <A> - <B>' and A and B both occur in the sources cited
on line N and A - B == tok. usage: check_derived.py DRAFT.tex PROV.txt"""
import re, sys, os
EV = os.path.expanduser('~/clu/evidence/outputs/')
L = open(sys.argv[1]).read().split('\n')
nf = raw = 0; derived = []
for row in open(sys.argv[2]):
    m = re.match(r'(\d+) (\S+) not in (.*)', row.strip())
    if not m: continue
    raw += 1; n, tok, srcs = int(m.group(1)), m.group(2), [s.strip() for s in m.group(3).split(';')]
    nxt = L[n] if n < len(L) else ''
    d = re.search(r'\b%s is (\d+) - (\d+)' % re.escape(tok), nxt) if nxt.lstrip().startswith('%') else None
    if d:
        a, b = d.group(1), d.group(2)
        texts = [open(EV + s).read() for s in srcs]
        ok = int(a) - int(b) == int(tok) and all(any(re.search(r'(?<!\d)%s(?!\d)' % x, t) for t in texts) for x in (a, b))
        if ok:
            derived.append('%d %s = %s - %s (operands found in %s)' % (n, tok, a, b, '; '.join(srcs))); continue
    nf += 1
print('raw not-found tokens: %d ; derived-verified: %d ; not-found after derivation check: %d' % (raw, len(derived), nf))
for x in derived: print('DERIVED-VERIFIED', x)
