"""usage: residue.py WORDDIFF-neutral-X.txt -- the non-citation residue of the pass.
Takes worddiff.py's leftovers after move resolution (full lists) of the two citation-neutralised builds (whitespace-free strings), removes from
the BEFORE side, in order, each hand-typed author/year string this pass deletes (rules H1-H14 of cite_edit.py),
and requires the remainder to equal the AFTER side. Printed: each removal, and the residue (must be EMPTY)."""
import ast, re, sys

HAND = [  # (rule, hand-typed text as pypdf renders it, whitespace removed)
    ('H1', 'TenenbaumandFreeman,2000'), ('H2', ',VasilescuandTerzopoulos,2002'), ('H3', 'Vlasicetal.,2005'),
    ('H4', ',2020'), ('H5', ',2020'), ('H6', ',2021'), ('H7', ',2023'), ('H8', 'CaoandYang(2015)'),
    ('H9', '(Ginartetal.,2019)'), ('H10', '(Golatkar,Achille,SwaminathanandSoatto)'),
    ('H11', '(Raissi,PerdikarisandKarniadakis,2019)'), ('H12', '(Greydanus,DzambaandYosinski,2019)'),
    ('H13', 'Hall,MarshallandMartin,2002'), ('H14', 'Blelloch–Golovin'),
]
txt = open(sys.argv[1]).read()
m = re.search(r'^moved blocks resolved: .*?; leftover before (\[.*\]) after (\[.*\])$', txt, re.M)
before, after = (''.join(ast.literal_eval(m.group(1))), ''.join(ast.literal_eval(m.group(2)))) if m else ('', '')
rest, pos, ok = before, 0, True
for rule, h in HAND:
    i = rest.find(h, pos)
    print(f'{rule}: remove {h!r} -> {"found" if i >= 0 else "NOT FOUND"}')
    if i < 0:
        ok = False; continue
    rest, pos = rest[:i] + rest[i + len(h):], i
print(f'before minus hand-typed strings: {rest!r}\nafter:                          {after!r}')
res = 'EMPTY' if ok and rest == after else 'NOT EMPTY'
print(f'NON-CITATION RESIDUE: {res}')
sys.exit(res != 'EMPTY')
