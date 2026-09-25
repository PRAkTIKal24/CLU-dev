"""Page counts per section from the marker builds (page_markers_c.py), cross-checked with the compiled .aux.

usage: python pages.py MARKS.json BUILD_DIR JOBNAME UNMARKED_AUX > PAGES-<job>.md ; also writes PAGES-<job>.json beside stdout's
Positions: B markers (shipout page + y of the heading baseline); page coordinate = (page-1) + (y_top - y)/textheight,
with y_top = paperheight - (1in + voffset + topmargin + headheight + headsep). A section's length is the coordinate
difference to the next heading of the same or a higher level. A markers (immediate \\typeout{\\thepage}) are printed beside.
"""
import json, re, sys
MK, B, JOB, UAUX = sys.argv[1:5]
marks = json.load(open(MK))
pw = open('%s/%s.pwm' % (B, JOB)).read().split('\n')
D = [l for l in pw if l.startswith('D|')][0].split('|')[1:]
pt = [float(x.replace('pt', '')) for x in D]
paper, voff, top, hh, hs, th = pt
ytop = paper - (72.27 + voff + top + hh + hs)
pos = {}
for l in pw:
    if l.startswith('B|'):
        _, mid, pg, y = l.split('|')
        if mid not in pos: pos[mid] = (int(pg), int(y) / 65536.0)
A = {}
for m in re.finditer(r'PW1MARK A\|(m\d+)\|(\d+)', open('%s/%s.log' % (B, JOB)).read()):
    A.setdefault(m.group(1), int(m.group(2)))
aux = open(UAUX).read()
toc = []
for l in aux.split('\n'):
    m1 = re.match(r'\\@writefile\{toc\}\{\\contentsline \{(section|subsection)\}\{\\numberline \{([^}]*)\}', l)
    m2 = re.search(r'\}\{(\d+)\}\{(?:section|subsection|appendix)\.[^{}]*\}\\protected@file@percent \}$', l)
    if m1 and m2: toc.append((m1.group(1), m1.group(2), '', m2.group(1), ''))
npages = int(re.search(r'\\gdef \\@abspage@last\{(\d+)\}', aux).group(1)) if '@abspage@last' in aux else None
lev = {'section': 1, 'section*': 1, 'bibliography': 1, 'end': 0, 'subsection': 2, 'subsection*': 2, 'subsubsection*': 1}
rows = []
tocs = [(k, n, pg) for k, n, _, pg, _ in toc]
ti = 0
for idx, m in enumerate(marks):
    pg, y = pos[m['id']]
    coord = (pg - 1) + (ytop - y) / th
    num = ''
    if m['kind'] in ('section', 'subsection'):
        k, n, tp = tocs[ti]; ti += 1
        assert k == m['kind'], (k, m)
        num = n
        m['toc_page'] = int(tp)
    rows.append(dict(m, page=pg, y_pt=round(y, 2), coord=coord, A_page=A.get(m['id']), num=num))
for i, r in enumerate(rows):
    L = lev[r['kind']]
    nxt = next((s for s in rows[i + 1:] if lev[s['kind']] <= L), None)
    r['length'] = (nxt['coord'] - r['coord']) if nxt else None
    r['end_page'] = nxt['page'] if nxt else None
json.dump(dict(rows=rows, ytop_pt=ytop, textheight_pt=th, pages_total=npages), open('%s/PAGES-%s.json' % (B, JOB), 'w'), indent=1)
mism = [r for r in rows if 'toc_page' in r and r['toc_page'] != r['page']]
print('toc/aux page vs marker page mismatches: %d' % len(mism))
for r in mism: print('  MISMATCH', r['num'], r['title'], r['toc_page'], r['page'])
print('A (immediate \\typeout{\\thepage}) differs from the shipped page at %d of %d markers' %
      (sum(1 for r in rows if r['A_page'] != r['page']), len(rows)))
