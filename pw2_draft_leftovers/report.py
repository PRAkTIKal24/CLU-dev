r"""Report tables for pw2-draft-leftovers.
usage: python3 report.py BASE_DIR STAGE_DIR OUTDIR
Writes EDITS.md (every edit, exact before/after, scope, per draft), LIMITATIONS.md (part 7: the Discussion limitations
list before -> after in both drafts, the numbers moved and where they live now) and NUMBERS-LEFT-DISCUSSION.md (the
Discussion rows of NUMBERS-LEFT, before -> after counts)."""
import os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'pw2_draft_numbers'))
from edits import E
from scan import scan
B, S, O = sys.argv[1:4]
DR = [('v2', 'DRAFT-v2'), ('style', 'DRAFT-v2-style')]

def fence(t): return '```latex\n' + t.rstrip('\n') + '\n```'

out = ['# EDITS — pw2-draft-leftovers: every .tex edit, both drafts (exact before → after)\n',
       f'{len(E)} edits, {sum(len(e["pairs"]) for e in E)} (scope, old, new) pairs; scope `both` = the same bytes in both drafts. '
       'Applied by `pw2_draft_leftovers/apply.py` (each `old` occurs exactly once per draft; comment-swallow guard).\n']
for e in E:
    out.append(f'## {e["id"]} — part {e["part"]}\n')
    for scope, old, new in e['pairs']:
        out.append(f'scope: `{scope}`\n\nbefore:\n{fence(old)}\n\nafter:\n{fence(new)}\n')
open(f'{O}/EDITS.md', 'w').write('\n'.join(out))

def lim(p):
    t = open(p).read()
    return t[t.index(r'\subsection{Limitations}'):t.index(r'\subsection{A planned CMAZE suite}')]
out = ['# LIMITATIONS — part 7 (item 5 applied to the Discussion limitations list), both drafts\n']
for d, n in DR:
    a, b = lim(f'{B}/{d}/{n}.tex'), lim(f'{S}/{d}/{n}.tex')
    out.append(f'## {n}: the list before\n\n{fence(a)}\n\n## {n}: the list after\n\n{fence(b)}\n')
va, sa = lim(f'{S}/v2/DRAFT-v2.tex'), lim(f'{S}/style/DRAFT-v2-style.tex')
out.append(f'**Lock-step:** the list is byte-identical between the two drafts after the edit: {va == sa} '
           f'(before: {lim(B + "/v2/DRAFT-v2.tex") == lim(B + "/style/DRAFT-v2-style.tex")}).\n')
out.append('## Numbers that left the list, and where each lives now\n\n| numbers | where they live now; what the text says | SRC |\n|---|---|---|')
for e in E:
    for num, where, src in e['moved']: out.append(f'| {num} | {where} | `{src}` |')
open(f'{O}/LIMITATIONS.md', 'w').write('\n'.join(out) + '\n')

out = ['# NUMBERS-LEFT (Discussion) — every number left in Sec. Discussion, both drafts\n',
       'From `pw2_draft_leftovers/numbers_left.py` (the full main-text list is `NUMBERS-LEFT.md`; 0 UNCLASSIFIED). '
       'Before = the installed drafts at the start of this pass, scanned the same way (the limitations list was then classed "lock-step").\n']
for d, n in DR:
    rb = [r for r in scan(f'{B}/{d}/{n}.tex') if r['section'] == 'Discussion']
    ra = [r for r in scan(f'{S}/{d}/{n}.tex') if r['section'] == 'Discussion']
    cnt = lambda rs, k, sub=None: sum(1 for r in rs if r['kind'] == k and (sub is None or r['sub'] == sub))
    out.append(f'## {n}\n\n| | digit tokens | word tokens | digit tokens in Limitations |\n|---|---|---|---|\n'
               f'| before | {cnt(rb, "digit")} | {cnt(rb, "word")} | {cnt(rb, "digit", "Limitations")} |\n'
               f'| after | {cnt(ra, "digit")} | {cnt(ra, "word")} | {cnt(ra, "digit", "Limitations")} |\n')
    out.append('Limitations digit tokens before: ' + ', '.join(f'`{r["token"]}`' for r in rb if r['kind'] == 'digit' and r['sub'] == 'Limitations') + '\n')
    out.append('Limitations digit tokens after: ' + ', '.join(f'`{r["token"]}`' for r in ra if r['kind'] == 'digit' and r['sub'] == 'Limitations') + '\n')
full = open(f'{O}/NUMBERS-LEFT.md').read().split('\n')
for d, n in DR:
    sec = False; rows = []
    for l in full:
        if l.startswith('## '): sec = l.strip() == f'## {n}.tex'
        if sec and l.startswith('| ') and '| Discussion |' in l and not re.match(r'\| Discussion \| \d', l): rows.append(l)
    out.append(f'### {n}: every Discussion number after (line, kind, token, justification, line text)\n\n'
               '| line | section | kind | number | justification | line text (code) |\n|---|---|---|---|---|---|\n' + '\n'.join(rows) + '\n')
open(f'{O}/NUMBERS-LEFT-DISCUSSION.md', 'w').write('\n'.join(out))
print('wrote EDITS.md, LIMITATIONS.md, NUMBERS-LEFT-DISCUSSION.md')
