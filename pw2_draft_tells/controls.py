"""Positive controls: plant one instance of every tell into a copy of a draft (a new paragraph and a run-in heading at
the top of Results 4.2, a sentence into an appendix) and check that every detector's count rises where it was planted.
Also: the brief's seven self-narration examples must each match the T3 list.
usage: python3 controls.py DRAFT.tex OUTDIR"""
import sys, os, re
from collections import Counter
from detect import detect, PHRASES
src, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
t = open(src).read()
PLANT = r"""\paragraph{What the plant is: a test (control).} However, the naive unit and the naive unit give the leaving note and the thickness of the
result, which we print, and the scores are printed, never netted, so the gain is real, not noise, and it holds on the same seeds,
the same rows, the same judges and all of them as a single long sentence of more than eighteen words without a break at 12.345 and 3.1 to 4.2.
This section plants three things at once in the n-CLU, the n-CLU, the n-CLU, the n-CLU and the n-CLU.

"""
anchor = '\\subsection{The $n$-way ladder}\n\\label{sec:ladder}\n'
assert t.count(anchor) == 1
t2 = t.replace(anchor, anchor + '\n' + PLANT)
aanchor = '\\section{Further limitations}\n\\label{app:morelimits}\n'
assert t2.count(aanchor) == 1
t2 = t2.replace(aanchor, aanchor + 'The planted appendix row is measured, not hedged, and we state plainly that it is neither a row nor a claim.\n\n')
p = os.path.join(out, 'planted-' + os.path.basename(src)); open(p, 'w').write(t2)
_, h0 = detect(src); _, h1 = detect(p)
c0 = Counter((h['tell'], h['section'][:5]) for h in h0); c1 = Counter((h['tell'], h['section'][:5]) for h in h1)
ok = True
for tell, sec in [('T1', 'Resul'), ('T2', 'Resul'), ('T3', 'Resul'), ('T4', 'Resul'), ('T5', 'Resul'), ('T6', 'Resul'), ('T7', 'Resul'),
                  ('T2', 'App Q'), ('T3', 'App Q')]:
    r = c1[(tell, sec)] > c0[(tell, sec)]
    ok &= r
    print('CONTROL %-3s %-6s %d -> %d  %s' % (tell, sec, c0[(tell, sec)], c1[(tell, sec)], 'CAUGHT' if r else 'MISSED'))
rules = Counter(h['rule'] for h in h1) - Counter(h['rule'] for h in h0)
print('rules that fired on the plant:', dict(rules))
brief = ['which we print', 'every row prints its seed count', 'cited in one sentence below with the distinction stated',
         'was not extended to manufacture', 'measured, not hedged', 'printed, never netted', 'we state plainly']
for b in brief:
    m = any(re.search(p_, b) for p_ in PHRASES); ok &= m
    print('BRIEF EXAMPLE %-60s %s' % (repr(b), 'MATCHED' if m else 'MISSED'))
print('ALL CONTROLS', 'PASS' if ok else 'FAIL'); sys.exit(0 if ok else 1)
