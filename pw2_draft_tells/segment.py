"""Segment a whole draft (\\begin{abstract} .. \\end{document}) into units for the tell detectors.
Unit kinds: 'prose' (a paragraph: blank-line block, cut at \\paragraph and headings; the \\paragraph title is kept
apart as .title), 'item' (one \\item of an enumerate/itemize in prose), 'quote' (a quote block), 'caption' (a float
caption or \\figslot caption), 'table' (tabular rows), 'heading' (\\section/\\subsection/\\paragraph title).
Every unit carries: section (Abstract, Introduction, Methods, Experiments, Results, Discussion, Statements,
App-preface, 'App X: title'), sub (subsection), lines (1-based), code (comments stripped), headheld (row name or '').
usage: python3 segment.py DRAFT.tex  -- prints the units."""
import re, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pw2_draft_sections'))
import headheld

def strip_comment(l):
    m = re.search(r'(?<!\\)%', l)
    return l if not m else l[:m.start()]

def hh_lines(path):
    out = {}
    try:
        for name, rng, _, _ in headheld.rows(path):
            a, b = [int(x) for x in rng.split('–')] if '–' in rng else (int(rng), int(rng))
            for k in range(a, b + 1): out[k] = name.split(' (')[0]
    except Exception as e:
        print('headheld rows failed:', e, file=sys.stderr)
    return out

MAIN = ['Introduction', 'Methods', 'Experiments', 'Results', 'Discussion']
def units(path, hhpath=None):
    L = open(path).read().split('\n')
    hh = hh_lines(path) if hhpath is None else hh_lines(hhpath)
    a0 = next(i for i, l in enumerate(L) if l.startswith(r'\begin{abstract}'))
    end = next(i for i, l in enumerate(L) if l.startswith(r'\end{document}'))
    U = []; sec = 'Abstract'; sub = ''; appx = False; appn = 0
    cur = []; env = []       # env stack of (name, start)
    fl = None                # float being collected: [kind, lines]
    def flush():
        nonlocal cur
        if cur:
            # cut at Head-held boundaries (line level) so a hit in a Head-held line is attributed to it
            runs = []
            for i in cur:
                h = (i + 1) in hh
                if runs and runs[-1][0] == h: runs[-1][1].append(i + 1)
                else: runs.append([h, [i + 1]])
            pid = len(U)
            for h, ls in runs:
                U.append(dict(kind='prose', section=sec, sub=sub, lines=ls, title='', para=pid, paralines=[i + 1 for i in cur]))
            cur = []
    i = a0 + 1
    while i < end:
        raw = L[i]; code = strip_comment(raw)
        s = code.strip()
        if raw.lstrip().startswith('%') or (s == '' and raw.strip() != ''):
            i += 1; continue            # whole-line comment: invisible, does not break a paragraph
        if s == '':
            flush(); i += 1; continue
        if s.startswith(r'\end{abstract}'):
            flush(); i += 1; continue
        if s.startswith(r'\appendix'):
            flush(); appx = True; sec = 'App-preface'; sub = ''; i += 1; continue
        m = re.match(r'\\(section|subsection|subsubsection)(\*?)\{(.*)\}\s*(\\label\{[^}]*\})?\s*$', s)
        if m:
            flush()
            lev, star, title = m.group(1), m.group(2), m.group(3)
            if lev == 'section':
                if appx and not star:
                    appn += 1; sec = 'App %s: %s' % (chr(64 + appn), title)
                elif appx: sec = 'App-preface'
                else: sec = title
                sub = ''
            elif lev == 'subsubsection' and not appx: sec = 'Statements'; sub = title
            else: sub = title
            U.append(dict(kind='heading', level=lev + star, section=sec, sub=sub, lines=[i + 1], title=title))
            i += 1; continue
        if re.match(r'\\label\{', s) and not cur:
            i += 1; continue
        # floats: table/figure environments and \figslot calls
        if re.match(r'\\begin\{(table|figure)\*?\}', s) or s.startswith(r'\figslot{'):
            flush()
            j = i
            if s.startswith(r'\figslot{'):
                depth = 0; args = 0; j = i
                while True:
                    for ch in strip_comment(L[j]):
                        if ch == '{': depth += 1
                        elif ch == '}':
                            depth -= 1
                            if depth == 0: args += 1
                    if args >= 3: break
                    j += 1
                U.append(dict(kind='caption', section=sec, sub=sub, lines=list(range(i + 1, j + 2)), title=''))
            else:
                while not re.search(r'\\end\{(table|figure)\*?\}', strip_comment(L[j])): j += 1
                # caption lines: from \caption{ to the line where its brace closes
                k = i
                while k <= j:
                    c = strip_comment(L[k])
                    if r'\caption{' in c:
                        depth = 0; started = False; k2 = k
                        while True:
                            cc = strip_comment(L[k2]); p0 = cc.index(r'\caption{') if k2 == k else 0
                            for ch in cc[p0:]:
                                if ch == '{': depth += 1; started = True
                                elif ch == '}': depth -= 1
                            if started and depth == 0: break
                            k2 += 1
                        U.append(dict(kind='caption', section=sec, sub=sub, lines=list(range(k + 1, k2 + 2)), title=''))
                        k = k2 + 1; continue
                    if '&' in c and c.rstrip().endswith('\\\\') or r'\multicolumn' in c:
                        U.append(dict(kind='table', section=sec, sub=sub, lines=[k + 1], title=''))
                    k += 1
            i = j + 1; continue
        if re.match(r'\\begin\{(enumerate|itemize)\}', s):
            flush()
            j = i + 1; itm = []
            depth = 1
            while True:
                c = strip_comment(L[j]).strip()
                if re.match(r'\\begin\{(enumerate|itemize)\}', c): depth += 1
                if re.match(r'\\end\{(enumerate|itemize)\}', c):
                    depth -= 1
                    if depth == 0: break
                if c.startswith(r'\item'):
                    if itm: U.append(dict(kind='item', section=sec, sub=sub, lines=itm, title=''))
                    itm = [j + 1]
                elif c and itm: itm.append(j + 1)
                j += 1
            if itm: U.append(dict(kind='item', section=sec, sub=sub, lines=itm, title=''))
            i = j + 1; continue
        if re.match(r'\\begin\{quote\}', s):
            flush(); j = i
            while not strip_comment(L[j]).strip().startswith(r'\end{quote}'): j += 1
            U.append(dict(kind='quote', section=sec, sub=sub, lines=list(range(i + 2, j + 1)), title=''))
            i = j + 1; continue
        if s.startswith(r'\paragraph{') or s.startswith(r'\noindent\textbf{'):
            flush()
            mm = re.match(r'\\paragraph\{(.*?)\}\s*', s) if s.startswith(r'\paragraph') else None
            if mm:
                # title may span lines: collect until braces balance
                t = s; j = i
                while t.count('{') > t.count('}'):
                    j += 1; t += ' ' + strip_comment(L[j]).strip()
                depth = 0; k = t.index('{')
                for p in range(k, len(t)):
                    if t[p] == '{': depth += 1
                    elif t[p] == '}':
                        depth -= 1
                        if depth == 0: break
                title = t[k + 1:p]
                U.append(dict(kind='heading', level='paragraph', section=sec, sub=sub, lines=list(range(i + 1, j + 2)), title=title))
        cur.append(i); i += 1
    flush()
    for u in U:
        u['code'] = '\n'.join(strip_comment(L[k - 1]) for k in u['lines'])
        names = {hh[k] for k in u['lines'] if k in hh}
        u['headheld'] = ', '.join(sorted(names))
    return U

if __name__ == '__main__':
    for u in units(sys.argv[1]):
        print(u['kind'][:4], u['section'][:28].ljust(28), u['sub'][:22].ljust(22), 'L%d-%d' % (u['lines'][0], u['lines'][-1]),
              ('HH ' if u['headheld'] else '') + (u['title'] or u['code'].replace('\n', ' ')[:70]))
