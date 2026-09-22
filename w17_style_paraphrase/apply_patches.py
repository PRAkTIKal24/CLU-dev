#!/usr/bin/env python3
"""Build DRAFT-v2-style.tex = DRAFT-v2.tex with declared prose line ranges replaced.
Patch spec files (patches/*.tex): blocks introduced by a line '@@@ START END' (1-indexed,
inclusive, line numbers of DRAFT-v2.tex); the block's text replaces those lines.
'@@@ START END' with END = START-1 inserts before START.
Per block, the multiset of number tokens and of provenance comments must be conserved;
every \\PLACEHOLDER{} and \\TODO{} argument must survive verbatim. Any violation aborts."""
import sys, glob, re
from texnum import numbers, comments, macro_args, norm_ws

SRC = sys.argv[1]; OUT = sys.argv[2]
orig = open(SRC).read().split('\n')
blocks = []
for f in sorted(glob.glob('patches/*.tex')):
    cur = None
    for line in open(f).read().split('\n'):
        m = re.match(r'^@@@ (\d+) (\d+)\s*$', line)
        if m:
            cur = [int(m.group(1)), int(m.group(2)), [], f]; blocks.append(cur); continue
        if cur is not None:
            cur[2].append(line)
for b in blocks:  # trim trailing blank lines of each block's text (spec-file separators)
    while b[2] and b[2][-1].strip() == '':
        b[2].pop()
blocks.sort(key=lambda b: b[0])
bad = 0
for a, b in zip(blocks, blocks[1:]):
    if b[0] <= a[1]:
        print('OVERLAP', a[:2], b[:2]); bad += 1
for s, e, new, f in blocks:
    old = orig[s-1:e]
    if numbers(old) != numbers(new):
        on, nn = numbers(old), numbers(new)
        print('NUMBER DRIFT %s %d-%d  lost=%s  added=%s' % (f, s, e, dict(on - nn), dict(nn - on))); bad += 1
    if comments(old) != comments(new):
        print('COMMENT DRIFT %s %d-%d  lost=%s added=%s' % (f, s, e, dict(comments(old)-comments(new)), dict(comments(new)-comments(old)))); bad += 1
    for mac in ('PLACEHOLDER', 'TODO'):
        oa = [norm_ws(x) for x in macro_args('\n'.join(old), mac)]
        na = [norm_ws(x) for x in macro_args('\n'.join(new), mac)]
        if oa != na:
            print('%s DRIFT %s %d-%d' % (mac, f, s, e)); bad += 1
if bad:
    print('ABORT: %d violations' % bad); sys.exit(1)
out = list(orig)
for s, e, new, f in sorted(blocks, key=lambda b: -b[0]):
    out[s-1:e] = new
open(OUT, 'w').write('\n'.join(out))
changed = sum(e - s + 1 for s, e, _, _ in blocks)
print('OK: %d blocks, %d original lines replaced, %d -> %d lines' % (len(blocks), changed, len(orig), len(out)))
