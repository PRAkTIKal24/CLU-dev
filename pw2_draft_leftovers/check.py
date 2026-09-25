r"""PW-2 item 4 acceptance checks, before -> after, both drafts.
usage: python3 check.py BASE_DIR STAGE_DIR > CHECKS.md ; exit 1 on any failed acceptance line.
BASE_DIR / STAGE_DIR each hold v2/DRAFT-v2.{tex,aux} and style/DRAFT-v2-style.{tex,aux} (built).
- \TODO / \PLACEHOLDER: macro USES in code (comments and the \newcommand lines excluded), each named by its argument's head;
- absolute paths anywhere in the file, comments included (/Users/, /home/, C:\, ~/, /c/chlu, /tmp/, /private/, /Volumes/);
- Figs. 5-8: \ref count per label, with the label's number and page from the .aux;
- the quote block: \begin{quote}\small and any \small inside the quote;
- every \newlabel: same number before and after (pages may move)."""
import re, sys
DR = [('v2', 'DRAFT-v2'), ('style', 'DRAFT-v2-style')]
FIGS = ['fig:ladder', 'fig:crossover', 'fig:deletion', 'fig:adapt']
ABS = re.compile(r'/Users/|/home/|[A-Za-z]:\\\\|[A-Za-z]:\\(?=[A-Za-z])|~/|/c/chlu|/tmp/|/private/|/Volumes/')

def code(line):
    m = re.search(r'(?<!\\)%', line)
    return line if not m else line[:m.start()]

def macros(t, name):
    out = []
    for i, l in enumerate(t.split('\n'), 1):
        c = code(l)
        if r'\newcommand' in c: continue
        for m in re.finditer(r'\\' + name + r'\{', c):
            out.append((i, c[m.end():m.end() + 70]))
    return out

def labels(aux):
    return {m.group(1): (m.group(2), m.group(3)) for m in
            re.finditer(r'\\newlabel\{([^}]*)\}\{\{([^}]*)\}\{([^}]*)\}', open(aux).read())}

bad = 0
def ok(cond, msg):
    global bad
    bad += not cond
    return ('PASS' if cond else 'FAIL') + ' ' + msg

B, S = sys.argv[1], sys.argv[2]
print('# CHECKS — pw2-draft-leftovers (PW-2 item 4 + item-5 follow-up), before -> after\n')
for d, n in DR:
    tb, ts = open(f'{B}/{d}/{n}.tex').read(), open(f'{S}/{d}/{n}.tex').read()
    lb, ls = labels(f'{B}/{d}/{n}.aux'), labels(f'{S}/{d}/{n}.aux')
    print(f'## {n}\n')
    for mac, want in (('TODO', 3), ('PLACEHOLDER', 0)):
        mb, ms = macros(tb, mac), macros(ts, mac)
        print(f'### \\{mac}: {len(mb)} -> {len(ms)}  ' + ok(len(ms) == want and len(mb) == want + (1 if mac == 'TODO' else 2), f'(expected {want + (1 if mac == "TODO" else 2)} -> {want})'))
        heads_after = {h for _, h in ms}
        for i, h in mb: print(f'- before line {i}: `{h}…` — ' + ('stays' if h in heads_after else '**removed**'))
        for i, h in ms: print(f'- after line {i}: `{h}…`')
        print()
    ab, as_ = [(i, l.strip()[:110]) for i, l in enumerate(tb.split('\n'), 1) if ABS.search(l)], \
              [(i, l.strip()[:110]) for i, l in enumerate(ts.split('\n'), 1) if ABS.search(l)]
    print(f'### Absolute paths (anywhere, comments included): {len(ab)} -> {len(as_)}  ' + ok(len(as_) == 0, 'expected 0 after'))
    for i, l in ab: print(f'- before line {i}: `{l}`')
    for i, l in as_: print(f'- AFTER line {i}: `{l}`')
    print()
    print('### Figs. 5–8: `\\ref` uses (code only)\n')
    print('| label | number | page before -> after | refs before | refs after | citing line (after) |\n|---|---|---|---|---|---|')
    for f in FIGS:
        rb = sum(len(re.findall(r'\\ref\{' + f + r'\}', code(l))) for l in tb.split('\n'))
        cit = [(i, code(l).strip()) for i, l in enumerate(ts.split('\n'), 1) if re.search(r'\\ref\{' + f + r'\}', code(l))]
        ra = sum(len(re.findall(r'\\ref\{' + f + r'\}', c)) for _, c in cit)
        print(f'| `{f}` | {ls[f][0]} | {lb[f][1]} -> {ls[f][1]} | {rb} | {ra} | ' +
              '; '.join(f'l.{i}: `…{c[max(0, c.find(f) - 60):c.find(f) + len(f) + 2]}`' for i, c in cit) + ' |')
        bad += not (ra >= 1)
    print()
    q = re.search(r'\\begin\{quote\}(.*?)\\end\{quote\}', ts, re.S)
    print('### Quote block\n\n' + ok(r'\begin{quote}\small' not in ts and r'\small' not in code(q.group(1)) and tb.count(r'\begin{quote}\small') == 1,
          f'`\\begin{{quote}}\\small` {tb.count(chr(92) + "begin{quote}" + chr(92) + "small")} -> {ts.count(chr(92) + "begin{quote}" + chr(92) + "small")}; no `\\small` inside the quote after') + '\n')
    miss = sorted(set(lb) - set(ls)); new = sorted(set(ls) - set(lb))
    ren = sorted(k for k in set(lb) & set(ls) if lb[k][0] != ls[k][0])
    mov = sorted(k for k in set(lb) & set(ls) if lb[k][1] != ls[k][1])
    print(f'### Labels: {len(lb)} -> {len(ls)}; missing {miss}; new {new}; renumbered {ren}  ' + ok(not miss and not new and not ren, 'none missing, new or renumbered'))
    print(f'Pages moved ({len(mov)}): ' + ', '.join(f'{k} p.{lb[k][1]}→{ls[k][1]}' for k in mov) + '\n')
print(f'\n**{bad} FAIL**')
sys.exit(1 if bad else 0)
