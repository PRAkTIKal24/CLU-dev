#!/usr/bin/env python3
"""Residual scan of rendered (non-comment) text for the item 14/15/17 tokens. usage: scan_cuts.py FILE [--lines]"""
import re, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pw1_draft_sweeps'))
from texnum import split_comment
TOK = [('CMAZE-1', r'CMAZE-1'), ('CMAZE-2', r'CMAZE-2(?!b)'), ('CMAZE-2b', r'CMAZE-2b'), ('CMAZE-O', r'CMAZE-O'),
       ('read verb', r'read verb'), ('verb', r'\bverbs?\b'), ('route', r'\broutes?\b'), ('MERGE/AVG', r'\\texttt\{(MERGE|AVG)\}'),
       ('request layer', r'request[- ]layer'), ('maze-rel-tuples', r'maze-rel-tuples|maze\\?_rel\\?_tuples'),
       ('cmazet-deletion', r'cmazet-deletion|cmazet\\?_deletion'), ('readverb', r'readverb'), ('requestlayer', r'requestlayer'),
       ('CMAZE (bare suite)', r'CMAZE(?!-)')]
f = sys.argv[1]; show = '--lines' in sys.argv
lines = open(f).read().split('\n')
code = [split_comment(l)[0] for l in lines]   # labels/refs KEPT here: a dangling \ref key counts
for name, rx in TOK:
    hits = [(i+1, m.group(0)) for i, c in enumerate(code) for m in re.finditer(rx, c)]
    print('%-20s %4d' % (name, len(hits)))
    if show:
        for i, g in hits: print('      L%d [%s] %s' % (i, g, code[i-1].strip()[:220]))
