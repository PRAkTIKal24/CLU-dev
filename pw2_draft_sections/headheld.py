"""Head-held paragraphs of a draft (COMMON.md): the abstract's opening sentence, the abstract's maze \\TODO, and the first
paragraph of each of the five main sections (first prose chunk after the \\section line; float-only chunks skipped;
the Methods slot line counts while it is there).
usage: python3 headheld.py DRAFT.tex [HEAD-HELD.md]  -- prints the rows; with HEAD-HELD.md, compares every sha256 to the
table of the matching draft there and exits 1 on any difference."""
import re, sys
from blocks import chunks, text, sha, first8
from layout import SLOT
SECTIONS = ['Introduction', 'Methods', 'Experiments', 'Results', 'Discussion']
def rows(path):
    L, s, e, items, P = chunks(path)
    out = []
    ab0 = next(i for i, l in enumerate(L) if l.startswith(r'\begin{abstract}'))
    ab1 = next(i for i, l in enumerate(L) if l.startswith(r'\end{abstract}'))
    abs_txt = '\n'.join(L[ab0 + 1:ab1])
    m = re.search(r'\.(\s)', abs_txt); fs = abs_txt[:m.start() + 1]
    out.append(('abstract, opening sentence', f'{ab0+2}–{ab0+2+fs.count(chr(10))}', first8(fs), sha(fs)))
    tl = [i for i in range(ab0, ab1) if r'\TODO{CMAZE-T sentence' in L[i]][0]; tj = tl
    while not L[tj].rstrip().endswith('}'): tj += 1
    todo = '\n'.join(L[tl:tj + 1])
    out.append(('abstract, maze `\\TODO` (whole lines)', f'{tl+1}–{tj+1}', first8(todo), sha(todo)))
    sec = None; pre = []; done = set()
    for k, idx in items:
        if k == 'H':
            for i in idx:
                mm = re.match(r'\\section\{(.*)\}\s*$', L[i])
                if mm: sec = mm.group(1); pre = []
            continue
        if sec not in SECTIONS or sec in done: continue
        t = text(L, idx)
        if t == SLOT:
            out.append((f'{sec}, first paragraph = the item-3 slot (to be replaced by the Head\'s opening)', f'{idx[0]+1}', SLOT, sha(t)))
            done.add(sec); continue
        prose = [l for l in t.split('\n') if not l.lstrip().startswith('%')]
        if prose and re.match(r'\\(figslot|begin\{figure|begin\{table)', prose[0]):
            pre.append(f'{idx[0]+1}–{idx[-1]+1}'); continue
        note = f' (preceded by the float chunk at lines {", ".join(pre)})' if pre else ''
        out.append((f'{sec}, first paragraph{note}', f'{idx[0]+1}–{idx[-1]+1}', first8(t), sha(t))); done.add(sec)
    missing = [x for x in SECTIONS if x not in done]
    assert not missing, ('sections not found', missing)
    return out
def table(r):
    return ['| paragraph | lines | first 8 words | sha256 |', '|---|---|---|---|'] + \
           [f'| {a} | {b} | `{c}` | {d} |' for a, b, c, d in r]
if __name__ == '__main__':
    path = sys.argv[1]; r = rows(path)
    print('\n'.join(table(r)))
    if len(sys.argv) > 2:
        name = path.rstrip('/').split('/')[-1]
        md = open(sys.argv[2]).read()
        part = md.split(f'## {name}')[1].split('\n## ')[0]
        want = re.findall(r'\| ([0-9a-f]{64}) \|', part); have = [x[3] for x in r]
        bad = [(x[0], w, h) for x, w, h in zip(r, want, have) if w != h] + ([('row count', len(want), len(have))] if len(want) != len(have) else [])
        print('HEAD-HELD CHECK:', 'PASS (all %d equal)' % len(have) if not bad else 'FAIL %s' % bad)
        sys.exit(1 if bad else 0)
