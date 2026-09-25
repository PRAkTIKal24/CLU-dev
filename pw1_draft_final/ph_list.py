"""Every \\TODO and \\PLACEHOLDER by ARGUMENT (brace-balanced, comments stripped), with line and section.

usage: python ph_list.py DRAFT.tex DRAFT.aux OUT.json
Section = the last \\section / \\subsection / \\section* heading above the macro, numbered from the .aux label that
follows the heading. Also prints `grep -c`-style LINE counts beside the argument counts (they differ when two macros
share a line).
"""
import json, os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'pw1_draft_sweeps'))
from texnum import split_comment  # noqa: E402
tex, aux, oj = sys.argv[1:4]
L = open(tex).read().split('\n')
code = '\n'.join(split_comment(l)[0] for l in L)
A = open(aux).read()
num = {m.group(1): m.group(2) for m in re.finditer(r'\\newlabel\{((?:sec|app):[^}]*)\}\{\{([^}]*)\}', A)}
heads = []
for m in re.finditer(r'\\(section|subsection|subsubsection)(\*?)\{', code):
    d = 0
    for k in range(m.end() - 1, len(code)):
        if code[k] == '{': d += 1
        elif code[k] == '}':
            d -= 1
            if d == 0: break
    lab = re.match(r'\s*\\label\{([^}]*)\}', code[k + 1:k + 200])
    heads.append((m.start(), m.group(1) + m.group(2), ' '.join(code[m.end():k].split()), num.get(lab.group(1)) if lab else None))
if 'abstract' in code:
    pass
ab = (code.find('\\begin{abstract}'), code.find('\\end{abstract}'))
out = []
for kind in ('TODO', 'PLACEHOLDER'):
    for m in re.finditer(r'\\%s\{' % kind, code):
        if code[max(0, m.start() - 12):m.start()].endswith('newcommand{'): continue
        d = 0
        for j in range(m.end() - 1, len(code)):
            if code[j] == '{': d += 1
            elif code[j] == '}':
                d -= 1
                if d == 0: break
        prev = [h for h in heads if h[0] < m.start()]
        where = 'abstract' if ab[0] < m.start() < ab[1] else (('%s %s' % (prev[-1][3] or '(unnumbered)', prev[-1][2])) if prev else 'front matter')
        # inside a float: the float's own label (\\label{..} or \\figslot's third argument) comes before any new float/heading
        fl = re.search(r'(\\label\{|\}\{)((?:fig|tab):[^}]*)\}', code[j:j + 4000])
        stop = re.search(r'\\begin\{(figure|table)|\\(sub)*section|\\figslot|\\end\{(figure|table)', code[j:j + 4000])
        cap = fl.group(2) if fl and (not stop or fl.start() < stop.start() or code[j + stop.start():].startswith('\\end')) else None
        out.append(dict(pos=m.start(), kind=kind, in_float=cap, line=code.count('\n', 0, m.start()) + 1, section=where,
                        arg=' '.join(code[m.end():j].split())))
out.sort(key=lambda o: o['pos'])
lines = dict(TODO=sum(1 for l in code.split('\n') if re.search(r'\\TODO\{', l) and 'newcommand' not in l),
             PLACEHOLDER=sum(1 for l in code.split('\n') if re.search(r'\\PLACEHOLDER\{', l) and 'newcommand' not in l))
res = dict(items=out, count_args=dict(TODO=sum(o['kind'] == 'TODO' for o in out), PLACEHOLDER=sum(o['kind'] == 'PLACEHOLDER' for o in out)),
           count_lines=lines)
json.dump(res, open(oj, 'w'), indent=1)
print(os.path.basename(tex), 'arguments', res['count_args'], '| lines', lines)
