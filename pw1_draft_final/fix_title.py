"""The one mechanical defect fix of pw1-draft-final: the title's forced line break.

usage: python fix_title.py IN.tex OUT.tex
Item 13 shortened the title and kept the old `\\\\` after "Generation"; in the ICLR small caps the first line then overfills,
LaTeX hyphenates GENERA-/TION and the forced break leaves "TION" alone on a line (3 title lines). Moving the break
after "Compositional" changes no word and gives two lines. One anchored operation, asserted unique.
"""
import sys
src, dst = sys.argv[1:3]
s = open(src).read()
old = r'\title{naive-CLU: Deletable Compositional Generation\\ on a Frozen Encoder}'
new = r'\title{naive-CLU: Deletable Compositional\\ Generation on a Frozen Encoder}'
assert s.count(old) == 1, s.count(old)
open(dst, 'w').write(s.replace(old, new))
print('title line break moved:', old, '->', new)
