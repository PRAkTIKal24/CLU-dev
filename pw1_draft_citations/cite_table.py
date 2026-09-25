"""usage: cite_table.py BEFORE.tex AFTER.tex AFTER.bbl KEYMAP.json OLD.bib NEW.bib OUT.md
Every citation after the pass, in order, with its source sentence, the before form (the same site in BEFORE: the
command, or the hand-typed text it replaces, or '(added)'), the after form, and the rendered string computed from the
.bbl labels; then the citation count of every bib entry before -> after; uncited entries; citations inside the
abstract; \\citet whose author surname also appears in plain text right before it (a doubled name)."""
import json, re, sys
sys.path.insert(0, __import__('os').path.dirname(__file__))
from bibparse import entries

bt, at, bbl, km = open(sys.argv[1]).read(), open(sys.argv[2]).read(), open(sys.argv[3]).read(), json.load(open(sys.argv[4]))
old, new = entries(sys.argv[5]), entries(sys.argv[6])
CITE = re.compile(r'\\(cite[a-z]*|nocite)\{([^}]*)\}')


def code(s):
    return '\n'.join(re.sub(r'(?<!\\)%.*', '', l) for l in s.split('\n'))


def cites(s):
    c = code(s)
    return [(m.start(), m.group(1), [k.strip() for k in m.group(2).split(',')], m) for m in CITE.finditer(c)], c


label = {}
for m in re.finditer(r'\\bibitem\[(.*?)\((\d{4}[a-z]?)\)(.*?)\]\{([^}]*)\}', bbl, re.S):
    label[m.group(4)] = (' '.join(m.group(1).replace('\\&', '&').replace('et~al.', 'et al.').split()), m.group(2))


def render(cmd, keys):
    parts = [label.get(k, ('??', '??')) for k in keys]
    if cmd == 'citep':
        return '(' + '; '.join(f'{a}, {y}' for a, y in parts) + ')'
    if cmd == 'citet':
        return '; '.join(f'{a} ({y})' for a, y in parts)
    if cmd == 'citealp':
        return '; '.join(f'{a}, {y}' for a, y in parts)
    return '(prints nothing)'


def sentence(c, pos):
    a = max(c.rfind('. ', 0, pos), c.rfind('\n\n', 0, pos), c.rfind('\\item', 0, pos), c.rfind('\\caption{', 0, pos)) + 1
    e = [x for x in (c.find('. ', pos), c.find('\n\n', pos)) if x >= 0]
    return ' '.join(c[a:min(e) + 1 if e else pos + 200].split())


B, bc = cites(bt)
A, ac = cites(at)
abs0, abs1 = ac.find('\\begin{abstract}'), ac.find('\\end{abstract}')
from rules import RULES
# the rule spans: in BEFORE (the text a rule replaces) and in AFTER (the text it wrote), located by the command inside
spans_b = []
for rid, kind, pat, repl in RULES:
    for mm in re.finditer(pat, bc):
        spans_b.append((mm.start(), mm.end(), rid, kind, ' '.join(mm.group(0).split())))
        break
# AFTER sites written by an 'add' rule: located by the rule's replacement text (keys mapped) in AFTER
added = set()
for rid, kind, pat, repl in RULES:
    if kind != 'add':
        continue
    txt = re.sub(r'\\\\', r'\\', repl)
    txt = re.sub(r'(\\cite[a-z]*\{)([^}]*)\}', lambda q: q.group(1) + ','.join(km.get(k, k) for k in q.group(2).split(',')) + '}', txt)
    i = ac.find(txt)
    assert i >= 0, (rid, txt)
    added |= {j for j, (p, _, _, _) in enumerate(A) if i <= p < i + len(txt)}
rest = [j for j in range(len(A)) if j not in added]
assert len(rest) == len(B), (len(rest), len(B))
pair = dict(zip(rest, range(len(B))))
mismatch = [(j, i) for j, i in pair.items() if tuple(A[j][2]) != tuple(km.get(k, k) for k in B[i][2])]
assert not mismatch, mismatch
used = set(pair.values())
rows, out = [], []
add_rules = [r for r in RULES if r[1] == 'add']
for n, (p, cmd, ks, m) in enumerate(A, 1):
    if n - 1 in pair:
        bp, bcmd, bks, bm = B[pair[n - 1]]
        rule = next((x for x in spans_b if x[0] <= bp < x[1]), None)
        before = (f'{rule[2]}: {rule[4]}' if rule else f'\\{bcmd}{{{bm.group(2)}}}')
    else:
        before = '(added)'
    region = 'ABSTRACT' if abs0 <= p <= abs1 else ('appendix' if p > ac.find('\\appendix') else 'main')
    line = ac.count('\n', 0, p) + 1
    rows.append(f'| {n} | {line} | {region} | `{before}` | `\\{cmd}{{{",".join(ks)}}}` | {render(cmd, ks)} | {sentence(ac, p)[:260].replace("|", "/")} |')
lost = [f'\\{B[i][1]}{{{",".join(B[i][2])}}} (line {bc.count(chr(10), 0, B[i][0]) + 1})' for i in range(len(B)) if i not in used]
out += ['## Every citation, before -> after (in order of the AFTER draft)', '',
        '| # | line | region | before (old key; hand-typed text shown) | after | rendered (from .bbl) | sentence (source, comments stripped) |',
        '|---|---|---|---|---|---|---|'] + rows
out += ['', f'BEFORE sites not matched to an AFTER site (a removed citation would show here): {lost or "none"}']
cb, ca = {}, {}
for _, _, ks, _ in B:
    for k in ks:
        cb[km.get(k, k)] = cb.get(km.get(k, k), 0) + 1
for _, _, ks, _ in A:
    for k in ks:
        ca[k] = ca.get(k, 0) + 1
inv = {v: k for k, v in km.items()}
out += ['', '## Citation count per bib entry (key occurrences in \\cite*/\\nocite on code), before -> after', '',
        '| new key | old key | before | after | change |', '|---|---|---|---|---|']
for k in sorted(new):
    out.append(f'| `{k}` | `{inv.get(k, "")}` | {cb.get(k, 0)} | {ca.get(k, 0)} | {ca.get(k, 0) - cb.get(k, 0):+d} |')
unc = [k for k in new if not ca.get(k)]
notbib = sorted(set(ca) - set(new))
out += ['', f'entries in the bib: {len(new)}; cited after: {sum(1 for k in new if ca.get(k))}; UNCITED after: {unc or "none"}; cited keys not in the bib: {notbib or "none"}',
        f'citation commands: before {len(B)} ({sum(1 for x in B if x[1] == "nocite")} \\nocite), after {len(A)}; key occurrences before {sum(cb.values())}, after {sum(ca.values())}',
        f'commands after by kind: ' + str({c: sum(1 for x in A if x[1] == c) for c in sorted({x[1] for x in A})}),
        f'citations inside the abstract after: {sum(1 for x in A if abs0 <= x[0] <= abs1)}']
dbl = []
for p, cmd, ks, m in A:
    if cmd == 'citet':
        sur = re.split(r'[ ,]', label.get(ks[0], ('', ''))[0])[0]
        if sur and sur in ac[max(0, p - 80):p]:
            dbl.append((ks, sur))
out.append(f'\\citet whose first surname also appears in the 80 characters before it (doubled name): {dbl or "none"}')
open(sys.argv[7], 'w').write('\n'.join(out) + '\n')
print('\n'.join(out[-6:]))
