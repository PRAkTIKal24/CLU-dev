"""python3 index_md.py G OUTDIR : SECTIONS.md (section index + pages, before -> after), FIGTAB.md, REFS.md."""
import sys, json, re
G, OUT = sys.argv[1:3]
DR = ['DRAFT-v2', 'DRAFT-v2-style']
def labels(aux):
    d = {}
    for m in re.finditer(r'\\newlabel\{([^}]*)\}\{\{([^}]*)\}\{([^}]*)\}', open(aux).read()): d[m.group(1)] = (m.group(2), m.group(3))
    return d
S = ['# Section index and page counts, before -> after (PW-2 item 2)', '',
     'From exact heading markers at shipout (`page_markers_c.py`, marker builds with every `\\newlabel` identical to the unmarked build). "len" = text-flow length in pages from the heading to the next heading of the same or higher level (floats count where they land). Main text = Introduction heading to the References heading.', '']
for d in DR:
    for side in ['before', 'after']:
        J = json.load(open(f'{G}/measure-{side}/PAGES-{d}.json')); R = J['rows']
        intro = next(r for r in R if r['title'].startswith('Introduction'))
        ref = next(r for r in R if r['kind'] == 'bibliography')
        S += [f'## {d}, {side}: {J["pages_total"]} pages; main text (Introduction -> References) {ref["coord"] - intro["coord"]:.2f} pages (Introduction on p. {intro["page"]}, References on p. {ref["page"]})', '',
              '| # | heading | page | len (pages) |', '|---|---|---|---|']
        for r in R:
            if r['kind'] in ('end',): continue
            if r['coord'] > ref['coord'] and r['kind'] != 'bibliography' and not r['title'].startswith('Appendices'):
                if r['kind'] not in ('section',): continue
            ln = f"{r['length']:.2f}" if r['length'] is not None else ''
            S.append(f"| {r['num']} | {'&nbsp;&nbsp;' if r['kind'].startswith('sub') else ''}{r['title']} | {r['page']} | {ln} |")
        S.append('')
open(f'{OUT}/SECTIONS.md', 'w').write('\n'.join(S) + '\n')
F = ['# Figure / table index, after (both drafts), with the before number and page', '']
for d in DR:
    A = json.load(open(f'{G}/measure-after/FIGTAB-INDEX.json'))[d]['items']
    B = json.load(open(f'{G}/measure-before/FIGTAB-INDEX.json'))[d]['items']
    F += [f'## {d}', '', '| kind | number | label | caption lead | page | section | before: number, page |', '|---|---|---|---|---|---|---|']
    renum = 0
    for kind in ('figure', 'table'):
        for lab, v in sorted(((l, v) for l, v in A.items() if v['kind'] == kind), key=lambda x: int(x[1]['num'])):
            b = B[lab]; renum += b['num'] != v['num']
            F.append(f"| {kind} | {v['num']} | `{lab}` | {v['lead'][:80]} | {v['page']} | {v['section']} {v['section_title']} | {b['num']}, p. {b['page']} |")
    F += ['', f'numbers changed before -> after: {renum}; labels before {len(B)}, after {len(A)}, same set: {set(A) == set(B)}', '']
open(f'{OUT}/FIGTAB.md', 'w').write('\n'.join(F) + '\n')
R = ['# Section-label targets, before -> after (rendered number), and every `\\ref` to a section label', '']
for d, f in [('DRAFT-v2', 'v2'), ('DRAFT-v2-style', 'style')]:
    lb, la = labels(f'{G}/before/{f}/{d}.aux'), labels(f'{G}/{f}/{d}.aux')
    tex = open(f'{G}/{f}/{d}.tex').read().split('\n')
    R += [f'## {d}', '', '| label | before | after | refs (line: context) |', '|---|---|---|---|']
    for lab in sorted(set(lb) | set(la), key=lambda x: (la.get(x, ('~',))[0])):
        if not lab.startswith('sec:'): continue
        refs = [f'{i+1}: …{re.sub(r"[|]", "/", l[max(0, l.find(lab) - 45):l.find(lab) + len(lab) + 2])}…' for i, l in enumerate(tex) if '{' + lab + '}' in l and '\\label{' + lab not in l]
        R.append(f"| `{lab}` | {lb.get(lab, ('—',))[0]} | {la.get(lab, ('—',))[0]} | {'<br>'.join(refs) or '(none)'} |")
    R.append('')
open(f'{OUT}/REFS.md', 'w').write('\n'.join(R) + '\n')
print('ok')
