"""usage: cite_edit.py IN.tex OUT.tex KEYMAP.json REPORT.md
Citation pass of pw1-draft-citations. Each rule must match exactly once in the draft (else exit 2).
The line count is preserved: a line break inside a replaced span moves to the nearest space before it.
Then every key of every \\cite*{} / \\nocite{} is renamed via the keymap; the bibliography lines point at refs_clu_iclr."""
import json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

KM = json.load(open(sys.argv[3]))
from rules import RULES, FIRST, BIB


def code_mask(s):
    """True at positions that are TeX code (not inside a % comment)."""
    m, com = [True] * len(s), False
    for i, c in enumerate(s):
        if c == '\n':
            com = False
        elif c == '%' and (i == 0 or s[i - 1] != '\\'):
            com = True
        m[i] = not com
    return m


src = open(sys.argv[1]).read()
n_lines = src.count('\n')
rep = ['# cite_edit report: ' + sys.argv[1], '', '| rule | kind | line | before | after |', '|---|---|---|---|---|']
bad = []
for rid, kind, pat, repl in RULES:
    mask = code_mask(src)
    hits = [m for m in re.finditer(pat, src) if all(mask[m.start():m.end()])]
    if rid in FIRST and hits:  # first mention: the earliest match is the one edited
        rep.append(f'| {rid} | note | | {len(hits)} matches; the first (line {src.count(chr(10), 0, hits[0].start()) + 1}) is edited | |')
        hits = hits[:1]
    if len(hits) != 1:
        bad.append(f'{rid}: {len(hits)} matches'); continue
    m = hits[0]
    new = m.expand(repl)
    nl = m.group(0).count('\n')
    pre = src[:m.start()]
    for _ in range(nl):  # move each line break to the nearest space before the span, on code
        k = len(pre) - 1
        # never leave a citation command at a line start: skip the space right before the span if it opens with \cite
        while k >= 0 and not (pre[k] == ' ' and pre[k - 1] not in '~\\' and not (k == len(pre) - 1 and new.startswith('\\cite'))):
            if pre[k] == '\n':
                raise SystemExit(f'{rid}: no space to carry the line break')
            k -= 1
        pre = pre[:k] + '\n' + pre[k + 1:]
    line = src.count('\n', 0, m.start()) + 1
    rep.append(f'| {rid} | {kind} | {line} | `{" ".join(m.group(0).split())}` | `{new}` |')
    src = pre + new + src[m.end():]
for rid, a, b in BIB:
    if src.count(a) != 1:
        bad.append(f'{rid}: {src.count(a)} matches'); continue
    rep.append(f'| {rid} | bibliography | {src[:src.index(a)].count(chr(10)) + 1} | `{a.strip()}` | `{b.strip()}` |')
    src = src.replace(a, b)
# rename keys in every cite command on code (comments untouched)
mask = code_mask(src)
out, last, renamed = [], 0, 0
for m in re.finditer(r'\\(cite[a-z]*|nocite)\{([^}]*)\}', src):
    if not mask[m.start()]:
        continue
    keys = [k.strip() for k in m.group(2).split(',')]
    miss = [k for k in keys if k not in KM]
    if miss:
        bad.append(f'unmapped key(s) {miss} at line {src.count(chr(10), 0, m.start()) + 1}')
    out.append(src[last:m.start()] + f'\\{m.group(1)}{{' + ','.join(KM.get(k, k) for k in keys) + '}')
    last, renamed = m.end(), renamed + len(keys)
src = ''.join(out) + src[last:]
if src.count('\n') != n_lines:
    bad.append(f'line count {n_lines} -> {src.count(chr(10))}')
left = [m.group(0) for m in re.finditer(r'\\nocite\{[^}]*\}', src) if code_mask(src)[m.start()]]
rep += ['', f'keys renamed: {renamed}; \\nocite left on code: {left}; lines {n_lines} -> {src.count(chr(10))}', 'problems: ' + (str(bad) or 'none')]
open(sys.argv[4], 'w').write('\n'.join(rep) + '\n')
print('\n'.join(rep[-2:]))
if bad:
    sys.exit(2)
open(sys.argv[2], 'w').write(src)
