"""python3 restructure.py IN.tex OUT.tex {v2|style}: rebuild the main text in the item-2 layout.
Outside the main-text span every byte is kept. Every paragraph chunk is emitted verbatim once."""
import sys
from blocks import chunks
from layout import layout_for
src, dst, which = sys.argv[1:4]
L, s, e, items, P = chunks(src)
lay = layout_for(which)
used = [x for k, x in lay if k == 'P']
assert sorted(used) == list(range(len(P))), ('chunk use', sorted(set(range(len(P))) - set(used)), len(used), len(P))
body = []
for k, x in lay:
    body.append(P[x] if k == 'P' else x)
new = L[:s] + [''] + '\n\n'.join(body).split('\n') + [''] + L[e:]
open(dst, 'w').write('\n'.join(new))
print(which, 'chunks', len(P), 'emitted', len(used), 'lines', len(L), '->', len(new))
