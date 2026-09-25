"""Whole-document proof that the rendered PDF changed ONLY by the expected insertions (float moves and line breaks do
not matter): character multiset of the page text (pdftext.py output; whitespace and hyphens removed), after minus
before, must equal the characters of EXPECTED and nothing may be lost.
usage: python3 charproof.py BEFORE.pdf.txt AFTER.pdf.txt [--plant]"""
import sys, re
from collections import Counter
def norm(s): return re.sub(r'[\s\-‐‑–]', '', s)
EXPECTED = [
  # (1) the Methods opening as rendered (the plan's words; \& -> &)
  None,
  # (2) the one added citation outside Methods, as natbib prints \citep
  '(Jawahar & Pierini, 2026)',
  # (3) the bibliography entry, as the .bst prints it (lower-casing is the bib's, not ours)
  'Pratik Jawahar and Maurizio Pierini. Chlu: The causal hamiltonian learning unit as a symplectic primitive for deep learning. arXiv preprint arXiv:2603.01768, 2026.']
import os
plan = open(os.path.expanduser('~/clu/tasks/pw2/PLAN.md')).read().split('### PW-2 item 3')[1].split('###')[0]
EXPECTED[0] = re.search(r'> "(.*?)"\s*\n', plan, re.S).group(1).replace('\\&', '&')
b = Counter(norm(open(sys.argv[1]).read())); a = Counter(norm(open(sys.argv[2]).read()))
if '--plant' in sys.argv: a['x'] += 1; a['e'] -= 1
exp = Counter(norm(''.join(EXPECTED)))
added = a - b; lost = b - a
print('chars before %d, after %d, expected insertion %d' % (sum(b.values()), sum(a.values()), sum(exp.values())))
print('added - expected:', dict(added - exp) or 'EMPTY'); print('expected - added:', dict(exp - added) or 'EMPTY')
print('lost:', dict(lost) or 'EMPTY')
ok = added == exp and not lost
print('CHAR PROOF:', 'PASS (after = before + exactly the three expected insertions)' if ok else 'FAIL')
sys.exit(0 if ok else 1)
