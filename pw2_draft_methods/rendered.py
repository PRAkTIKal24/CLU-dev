"""Rendered Methods opening vs the plan's text, word by word (item 3 acceptance: EMPTY apart from the citation rendering).
usage: python3 rendered.py PDFTEXT.txt [--plant]   (PDFTEXT from pw1_draft_cuts/pdftext.py)
The plan's text is read from PLAN.md sec. 2 item 3 (between the quote marks), not retyped. Its only LaTeX escape, "\\&",
is rendered "&" (what \\citet prints). --plant swaps one word of the rendered text (positive control: must report a diff)."""
import re, sys, difflib, os
plan = open(os.path.expanduser('~/clu/tasks/pw2/PLAN.md')).read()
blk = plan.split('### PW-2 item 3')[1].split('###')[0]
quote = re.search(r'> "(.*?)"\s*\n', blk, re.S).group(1)
want = quote.replace('\\&', '&')
txt = open(sys.argv[1]).read()
m = re.search(r'\n2 METHODS\n(.*?itself\.)\n', txt, re.S)
got = m.group(1)
got = re.sub(r'-\n(?=[a-z])', '', got)   # LaTeX hyphenation at a line end (none expected: rendered words must be whole)
got = ' '.join(got.split())
if '--plant' in sys.argv: got = got.replace('virtue', 'vertue', 1)
W, G = want.split(), got.split()
ops = [o for o in difflib.SequenceMatcher(None, W, G, autojunk=False).get_opcodes() if o[0] != 'equal']
print('PLAN (PLAN.md, \\& rendered as &):\n  ' + want)
print('RENDERED (%s):\n  %s' % (sys.argv[1], got))
print('words: plan %d, rendered %d' % (len(W), len(G)))
for tag, a, b, c, d in ops: print('  DIFF %s plan[%d:%d]=%r rendered[%d:%d]=%r' % (tag, a, b, W[a:b], c, d, G[c:d]))
print('WORD DIFF:', 'EMPTY' if not ops else '%d NON-EMPTY' % len(ops))
print('CHAR EQUAL (exact string):', want == got)
sys.exit(1 if ops or want != got else 0)
