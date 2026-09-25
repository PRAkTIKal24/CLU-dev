#!/usr/bin/env python3
"""Residual scan of rendered (non-comment) text for banned tokens. usage: scan.py FILE [--lines]"""
import re, sys
from texnum import split_comment
TOK = [
 ('rival', r'\brivals?\b'), ('store', r'\bstor(e|es|ed|ing)\b'), ('gate', r'\bgat(e|es|ed|ing)\b'),
 ('artifact/artefact', r'\bartef?i?facts?\b|\bartefacts?\b'), ('pre-register', r'\bpre-?regist\w*'),
 ('A1', r'\bA1\b'), ('A10', r'\bA10\b'), ('C-512', r'\bC-512\b'), ('improved', r'\bimproved\b'),
 ('config', r'\bconfig\w*'), ('spoke', r'\bspokes?\b'), ('hub', r'\bhubs?\b'), ('wave', r'\bwaves?\b'),
 ('charter', r'\bcharter\w*'), ('ledger', r'\bledgers?\b'), ('bar k', r'\bbar [a-zA-Z]\b|\bbar-[a-z]\b'),
 ('scored run', r'\bscored runs?\b'), ('budget', r'\bbudgets?\b'), ('spend/spent', r'\bsp(end|ent|ends|ending)\b'),
 ('price/priced/pricing', r'\bpric(e|es|ed|ing)\b'), ('buy', r'\bbu(y|ys|ying|ought)\b'),
 ('pay', r'\bpa(y|ys|id|ying)\b'), ('cost', r'\bcosts?\b'), ('cheap', r'\bcheap\w*'), ('expensive', r'\bexpensive\b'),
 ('t_pay', r't_\{\\mathrm\{pay\}\}'), ('corner', r'\bcorners?\b'), ('held-out composition', r'held-out composition'),
 ('oblong', r'\boblongs?\b'), ('exact judge', r'\bexact(ly)? judge'), ('exact', r'\bexact(ly|ness)?\b'),
 ('10x', r'10x\b|10\$?\\times\$? steps\b'),
 ('landscape_removed', r'landscape\\?_removed'), ('landscape_shuffled', r'landscape\\?_shuffled'),
 ('random_atoms', r'random\\?_atoms'), ('Naive CLU', r'\bNaive CLU\b|\bnaive CLU\b'), ('CLU (bare)', r'(?<![-\w])CLU(?![-\w])'),
 ('\\S', r'\\S(?![a-zA-Z])'), ('Section~', r'Section~\\ref'), ('Figure~', r'Figures?~\\ref'), ('Table~', r'Tables?~\\ref'),
 ('Appendix~', r'Appendi(x|ces)~\\ref'), ('Sec.~', r'Secs?\.~\\ref'), ('Fig.~', r'Figs?\.~\\ref'), ('Tab.~', r'Tabs?\.~\\ref'),
 ('App.~', r'Apps?\.~\\ref'), ('Eq.~', r'Eqs?\.~\\ref'), ('Alg.~', r'Algs?\.~\\ref'), ('bare \\ref', r'(?<![~.])\\ref\{'),
]
f = sys.argv[1]; show = '--lines' in sys.argv; want = [a for a in sys.argv[2:] if not a.startswith('--')]
lines = open(f).read().split('\n')
code = [re.sub(r'\\(label|ref|eqref|pageref|citep|citet|cite|nocite)\{[^}]*\}', lambda m: '\\'+m.group(1)+'{}', split_comment(l)[0]) for l in lines]
for name, rx in TOK:
    if want and name not in want: continue
    hits = [(i+1, m.group(0)) for i, c in enumerate(code) for m in re.finditer(rx, c, re.I if name not in ('A1','A10','CLU (bare)','\\S') else 0)]
    print('%-22s %4d' % (name, len(hits)))
    if show and hits:
        for i, g in hits: print('      L%d [%s] %s' % (i, g, code[i-1].strip()[:200]))
