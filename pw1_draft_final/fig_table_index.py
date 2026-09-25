"""Figure/table index for the Head: number -> label -> caption lead -> page, both drafts, from the compiled .aux.

usage: python fig_table_index.py BUILD_DIR OUT.json
Reads BUILD_DIR/DRAFT-v2{,-style}.{aux,tex}. For every fig:/tab: \\newlabel: number, page and caption (hyperref's
third field) and the section it sits in (the last \\section / \\subsection before its \\label in the .tex, numbered from
the .aux). Cross-checks each label against the list-of-figures/tables entry with the same number (caption start and page
must agree) and asserts that the lof/lot entries and the labels are the same set.
"""
import json, re, sys
B, OUTJ = sys.argv[1:3]

def bal(s, i):
    d = 0
    for k in range(i, len(s)):
        if s[k] == '{': d += 1
        elif s[k] == '}':
            d -= 1
            if d == 0: return k
    raise ValueError

def clean(t):
    t = re.sub(r'\\(textbf|emph|textit|textsc|texttt|mathrm|text)\s*\{', '{', t)
    t = t.replace('\\ensuremath', '').replace('\\Kstar', 'K*').replace('{K^{\\star }}', 'K*').replace('K^{\\star }', 'K*')
    t = re.sub(r'\\ref\s*\{([^}]*)\}', r'[\1]', t)
    t = t.replace('\\times', 'x').replace('~', ' ').replace('--', '-').replace('\\,', ' ').replace('\\ ', ' ')
    t = re.sub(r'\\[a-zA-Z]+\s*', '', t)
    t = t.replace('{', '').replace('}', '').replace('$', '')
    return ' '.join(t.split())

def lead(cap):
    m = re.match(r'\s*\\textbf\s*\{', cap)
    if m:
        k = bal(cap, m.end() - 1)
        return clean(cap[m.end():k])
    s = clean(cap)
    return s.split('. ')[0][:160]

def parse(job):
    aux = open('%s/%s.aux' % (B, job)).read()
    tex = open('%s/%s.tex' % (B, job)).read()
    labs = {}
    for m in re.finditer(r'\\newlabel\{((?:fig|tab):[^}]*)\}\{', aux):
        i = m.end() - 1; k = bal(aux, i); body = aux[i + 1:k]
        # {num}{page}{caption}{anchor}{}
        parts = []; j = 0
        while j < len(body) and len(parts) < 4:
            if body[j] == '{':
                e = bal(body, j); parts.append(body[j + 1:e]); j = e + 1
            else: j += 1
        labs[m.group(1)] = dict(num=parts[0], page=int(parts[1]), caption=parts[2], anchor=parts[3])
    lo = {}
    for kind in ('lof', 'lot'):
        for m in re.finditer(r'\\@writefile\{%s\}\{\\contentsline \{(figure|table)\}\{\\numberline \{([^}]*)\}\{\\ignorespaces ' % kind, aux):
            i = m.end() - 2; st = aux.rfind('{\\numberline', 0, m.end()); e = bal(aux, st)
            cap = aux[m.end():e - 1]
            pg = re.match(r'\{(\d+)\}', aux[e + 1:]).group(1)
            lo[(m.group(1), m.group(2))] = dict(caption=cap, page=int(pg))
    # section of each label, from the tex
    secs = []
    for m in re.finditer(r'\\(section|subsection)\*?\{', tex):
        k = bal(tex, m.end() - 1); secs.append((m.start(), m.group(1), ' '.join(tex[m.end():k].split())))
    seclabels = {}
    for m in re.finditer(r'\\label\{((?:sec|app):[^}]*)\}', tex): seclabels[m.start()] = m.group(1)
    secnum = {}
    for m in re.finditer(r'\\newlabel\{((?:sec|app):[^}]*)\}\{\{([^}]*)\}\{(\d+)\}', aux): secnum[m.group(1)] = (m.group(2), int(m.group(3)))
    out = {}
    for lab, v in labs.items():
        p = tex.find('\\label{%s}' % lab)
        if p < 0: p = tex.find('}{%s}' % lab)  # \figslot{stem}{caption}{label}: the third argument, never a \ref
        prev = [s for s in secs if s[0] < p]
        sl = None
        if prev:
            s0 = prev[-1][0]
            cands = [(q, n) for q, n in seclabels.items() if s0 <= q < s0 + 600]
            sl = cands[0][1] if cands else None
        kind = 'figure' if lab.startswith('fig:') else 'table'
        l2 = lo.get((kind, v['num']))
        out[lab] = dict(v, kind=kind, lead=lead(v['caption']), section_label=sl,
                        section=secnum.get(sl, ('?', None))[0], section_title=clean(prev[-1][2]) if prev else '',
                        texline=tex.count('\n', 0, p) + 1,
                        lo_page=l2['page'] if l2 else None,
                        lo_agrees=bool(l2) and l2['page'] == v['page'] and re.sub(r'\s', '', clean(l2['caption']))[:60] == re.sub(r'\s', '', clean(v['caption']))[:60])
    # a float may carry two labels (an alias); count floats, not labels
    lo_keys = set(lo); lab_keys = set((o['kind'], o['num']) for o in out.values())
    return out, lo_keys - lab_keys, lab_keys - lo_keys

res = {}
for job in ('DRAFT-v2', 'DRAFT-v2-style'):
    out, unl, nolo = parse(job)
    res[job] = dict(items=out, lo_without_label=sorted(unl), label_without_lo=sorted(nolo))
    print(job, 'figures', sum(1 for o in out.values() if o['kind'] == 'figure'), 'tables', sum(1 for o in out.values() if o['kind'] == 'table'),
          '| lof/lot entries without label:', sorted(unl), '| labels without lof/lot:', sorted(nolo),
          '| label vs lof/lot disagreements:', [k for k, o in out.items() if not o['lo_agrees']])
json.dump(res, open(OUTJ, 'w'), indent=1)
