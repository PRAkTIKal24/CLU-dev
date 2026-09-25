"""Parse the main text of a draft (after \\end{abstract}, up to the rule line before the AI-use
statement) into heading runs ('H') and paragraph chunks ('P').
A blank-line block is cut (i) at every heading line (% ==== rule, \\section, \\subsection,
\\label{sec:...}), and (ii) at a \\paragraph line that follows text inside the block, the comment
lines just above that \\paragraph going with it. A \\paragraph starts a new TeX paragraph, so (ii)
never cuts a paragraph; nothing else is cut."""
import re, hashlib
HEAD = re.compile(r'^(% =+\s*$|\\section\{|\\subsection\{|\\label\{sec:)')
PARA = re.compile(r'^\\paragraph\{')
def load(p): return open(p).read().split('\n')
def span(L):
    s = next(i for i, l in enumerate(L) if l.startswith(r'\end{abstract}')) + 1
    e = next(i for i, l in enumerate(L) if l.startswith(r'% -----') and i + 1 < len(L)
             and L[i + 1].startswith(r'\subsubsection*{AI use statement}'))
    return s, e
def iscomment(l): return l.lstrip().startswith('%')
def parse(L):
    s, e = span(L)
    raw, cur = [], []
    for i in range(s, e):
        if L[i].strip() == '':
            if cur: raw.append(cur); cur = []
        else: cur.append(i)
    if cur: raw.append(cur)
    items = []
    for b in raw:
        segs, cur, kind = [], [], None
        for j, i in enumerate(b):
            k = 'H' if HEAD.match(L[i]) else 'P'
            if k != kind and cur:
                segs.append([kind, cur]); cur = []
            kind = k; cur.append(i)
        if cur: segs.append([kind, cur])
        for kind, idx in segs:
            if kind == 'H': items.append(('H', idx)); continue
            # cut before a \paragraph that follows text (with its leading comments)
            start = 0
            for j in range(1, len(idx)):
                if PARA.match(L[idx[j]]):
                    c = j
                    while c - 1 > start and iscomment(L[idx[c - 1]]): c -= 1
                    if any(not iscomment(L[idx[t]]) for t in range(start, c)):
                        items.append(('P', idx[start:c])); start = c
            items.append(('P', idx[start:]))
    return s, e, items
def text(L, idx): return '\n'.join(L[i] for i in idx)
def sha(t): return hashlib.sha256(t.encode()).hexdigest()
def first8(t):
    lines = [l for l in t.split('\n') if not iscomment(l)] or t.split('\n')
    return ' '.join(lines[0].split()[:8])
def chunks(path):
    L = load(path); s, e, items = parse(L)
    P = [text(L, idx) for k, idx in items if k == 'P']
    return L, s, e, items, P
if __name__ == '__main__':
    import sys
    L, s, e, items, P = chunks(sys.argv[1]); n = 0
    for kind, idx in items:
        t = text(L, idx)
        if kind == 'P':
            print(f'P{n:02d} L{idx[0]+1}-{idx[-1]+1} {sha(t)[:10]} | {first8(t)[:70]}'); n += 1
        else:
            print(f'   H L{idx[0]+1}: ' + ' / '.join(L[i][:60] for i in idx if not L[i].startswith('% =')))
