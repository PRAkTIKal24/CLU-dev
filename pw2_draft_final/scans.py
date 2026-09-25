"""Closing-audit scans (read-only; written fresh for pw2-draft-final).

  python3 scans.py todo DRAFT.tex [PDF.txt]      every \\TODO / \\PLACEHOLDER use (code, not comments), full argument,
                                                 line, section; rendered markers counted in the PDF text
  python3 scans.py cite DRAFT.tex                every sentence citing jawahar2026chlu, its topic and verdict;
                                                 exit 1 on VIOLATION (a disallowed topic not in the adjudication table)
  python3 scans.py headheld DRAFT.tex OLD.tex HEAD-HELD.md NOW.txt
                                                 raw-line byte comparison of every Head-held paragraph: the lines
                                                 HEAD-HELD.md names in OLD.tex (the draft it was written from) against
                                                 the lines headheld.py locates now (NOW.txt = its table on DRAFT.tex)
  python3 scans.py defects DRAFT.tex PDF.txt     mechanical defects: ?? / [?] in the rendered text, absolute paths in
                                                 the .tex, rendered title lines, doubled words, forced \\\\ in prose
"""
import hashlib, re, sys


def strip_comments(l):
    m = re.search(r'(?<!\\)%', l)
    return l if not m else l[:m.start()]


def bal(s, i):
    d = 0
    for k in range(i, len(s)):
        if s[k] == '{' and s[k - 1] != '\\': d += 1
        elif s[k] == '}' and s[k - 1] != '\\':
            d -= 1
            if d == 0: return k
    raise ValueError


def sections(L):
    cur, out = 'Preamble', []
    for l in L:
        if l.startswith('\\begin{abstract}'): cur = 'Abstract'
        m = re.match(r'\\(sub)*section\*?\{(.*)\}', l)
        if m and not m.group(1): cur = m.group(2)
        m2 = re.match(r'\\subsubsection\*\{(.*)\}', l)
        if m2: cur = 'Statements: ' + m2.group(1)
        if l.startswith('\\bibliography{') or 'bibliography{refs' in l: cur = 'References'
        out.append(cur)
    return out


def todo(path, pdf=None):
    L = open(path).read().split('\n')
    S = sections(L)
    code = '\n'.join(strip_comments(l) for l in L)
    starts = [0]
    for l in code.split('\n'): starts.append(starts[-1] + len(l) + 1)
    rows = []
    for mac in ('TODO', 'PLACEHOLDER'):
        for m in re.finditer(r'\\' + mac + r'\{', code):
            ln = next(i for i in range(len(starts)) if starts[i + 1] > m.start())
            if L[ln].lstrip().startswith('\\newcommand'): continue
            k = bal(code, m.end() - 1)
            arg = ' '.join(code[m.end():k].split())
            rows.append((mac, ln + 1, S[ln], arg))
    rows.sort(key=lambda r: r[1])
    print('| macro | line | section | argument (whole, whitespace collapsed) |')
    print('|---|---|---|---|')
    for r in rows: print('| `\\%s` | %d | %s | %s |' % (r[0], r[1], r[2], r[3].replace('|', '\\|')))
    n = {m: sum(r[0] == m for r in rows) for m in ('TODO', 'PLACEHOLDER')}
    cm = sum(1 for l in L if re.search(r'(?<!\\)%.*(TODO|PLACEHOLDER)', l))
    print('\n\\TODO uses: %d; \\PLACEHOLDER uses: %d; comment lines mentioning TODO/PLACEHOLDER (not uses): %d' % (n['TODO'], n['PLACEHOLDER'], cm))
    if pdf:
        t = open(pdf).read()
        print('rendered: "[TODO:" %d; "[PLACEHOLDER:" %d' % (t.count('[TODO:'), t.count('[PLACEHOLDER:')))
    return rows


ALLOWED = {'unit': r'\b(CHLU|CLU|unit|latent dynamics model)\b', 'dynamics': r'(Hamiltonian|symplectic|dynamics|settling)',
           'governor': r'(relativistic|governor)'}
FORBIDDEN = r'(memory|delet|forget|test-time|retention|GMOR|dissipat|unlearn)'
# sentences read by hand; key = the sentence's first six words. Each states WHY a forbidden term does not bind the cite.
ADJUDICATED = {
    'We build on top of this idea': 'memory unit / deletion attributed to "we" (this paper), not to the cited work; the '
                                    'citation sits in the two preceding sentences (the unit), this sentence carries none',
}


def sentences_with(code, key):
    out = []
    for m in re.finditer(r'\\cite[pt]?\*?(\[[^\]]*\])?\{([^}]*)\}', code):
        if key not in m.group(2): continue
        prev = [x.end() for x in re.finditer(r'(?<!~)(?<!\b[A-Z][a-z])(?<!\be\.g)(?<!\bi\.e)\.\s|\n\s*\n', code[:m.start()])]
        a = prev[-1] if prev else 0
        nx = re.search(r'(?<!\b[A-Z][a-z])\.(\s|$)', code[m.end():])
        b = m.end() + (nx.start() + 1 if nx else len(code) - m.end())
        out.append((m.start(), ' '.join(code[a:b].split())))
    return out


def cite(path, key='jawahar2026chlu'):
    L = open(path).read().split('\n')
    S = sections(L)
    code = '\n'.join(strip_comments(l) for l in L)
    starts = [0]
    for l in code.split('\n'): starts.append(starts[-1] + len(l) + 1)
    bad = 0
    print('| line | section | citing sentence | topics (allowed) | disallowed terms in the sentence | verdict |')
    print('|---|---|---|---|---|---|')
    rows = sentences_with(code, key)
    for pos, s in rows:
        ln = next(i for i in range(len(starts)) if starts[i + 1] > pos)
        tops = [k for k, rx in ALLOWED.items() if re.search(rx, s)]
        forb = sorted(set(x.lower() for x in re.findall(FORBIDDEN, s, re.I)))
        first6 = ' '.join(re.sub(r'\\cite[pt]?\{[^}]*\}', 'CITE', s).split()[:6])
        if not forb and tops: v = 'OK: cited for the %s' % ' / '.join(tops)
        elif forb and any(s.startswith(k) for k in ADJUDICATED): v = 'OK (read): ' + next(ADJUDICATED[k] for k in ADJUDICATED if s.startswith(k))
        elif forb: v = 'VIOLATION: disallowed topic %s in the citing sentence' % forb; bad += 1
        else: v = 'VIOLATION: no allowed topic'; bad += 1
        print('| %d | %s | %s | %s | %s | **%s** |' % (ln + 1, S[ln], s.replace('|', '\\|'), ', '.join(tops) or '—', ', '.join(forb) or '—', v))
    # the sentence right after each Methods citation that names memory / deletion (attribution check)
    for m in re.finditer(r'We build on top of this idea[^.]*\.', code):
        s = ' '.join(m.group(0).split())
        ln = next(i for i in range(len(starts)) if starts[i + 1] > m.start())
        forb = sorted(set(x.lower() for x in re.findall(FORBIDDEN, s, re.I)))
        print('| %d | %s | %s | — | %s | **no citation in this sentence; %s** |' % (ln + 1, S[ln], s, ', '.join(forb), ADJUDICATED['We build on top of this idea']))
    n_cite = len(re.findall(key, code))
    print('\ncitations of `%s` in code: %d (in %d citing sentences); VIOLATION: %d' % (key, n_cite, len(rows), bad))
    return bad


def headheld(path, old, md, nowtxt):
    L, O = open(path).read().split('\n'), open(old).read().split('\n')
    name = path.rstrip('/').split('/')[-1]
    part = open(md).read().split('## %s' % name)[1].split('\n## ')[0]
    want = re.findall(r'\| ([^|]+) \| (\d+)–(\d+) \| `[^`]*` \| ([0-9a-f]{64}) \|', part)
    now = re.findall(r'\| ([^|]+) \| (\d+)–(\d+) \| `[^`]*` \| ([0-9a-f]{64}) \|', open(nowtxt).read())
    assert len(want) == len(now) == 7, (len(want), len(now))
    bad = 0
    print('| paragraph | HEAD-HELD.md lines (in the draft it was written from) | lines now | raw lines byte-equal | sha256 (HEAD-HELD.md) = sha256 (now) |')
    print('|---|---|---|---|---|')
    for (n1, a1, b1, h1), (n2, a2, b2, h2) in zip(want, now):
        t_old = '\n'.join(O[int(a1) - 1:int(b1)]); t_now = '\n'.join(L[int(a2) - 1:int(b2)])
        eq = t_old == t_now
        if n1.startswith('abstract, opening'):   # a sentence, not whole lines: compare the sentence prefix
            eq = t_now.startswith(t_old.split('. ')[0]) and t_old.split('. ')[0] == t_now.split('. ')[0]
        bad += (not eq) or (h1 != h2)
        print('| %s | %s–%s | %s–%s | %s | %s |' % (n1.strip(), a1, b1, a2, b2, 'YES' if eq else '**NO**', 'YES' if h1 == h2 else '**NO**'))
    print('\nHEAD-HELD RAW CHECK: %s' % ('PASS (7 of 7)' if not bad else 'FAIL (%d)' % bad))
    return bad


def defects(path, pdf):
    L = open(path).read().split('\n')
    t = open(pdf).read()
    body = '\n'.join(l for l in t.split('\n') if not re.match(r'^\d{3}$', l))
    print('- rendered `??` (undefined reference): %d; `(?)` or `[?]` (undefined citation): %d' % (body.count('??'), body.count('(?)') + body.count('[?]')))
    ap = [(i + 1, l.strip()) for i, l in enumerate(L) if re.search(r'/Users/|/home/|C:\\\\|~/|/c/chlu|/tmp/|/private/|/Volumes/', l)]
    print('- absolute machine paths in the .tex (code and comments): %d%s' % (len(ap), ''.join('\n  - l.%d `%s`' % x for x in ap)))
    p1 = body.split('=== PAGE 2 ===')[0].split('\n')
    i = next(k for k, l in enumerate(p1) if 'ICLR 2027' in l)
    j = next(k for k, l in enumerate(p1) if l.startswith('Anonymous'))
    print('- rendered title lines (page 1, between the running head and the author line): %d: %s' % (j - i - 1, ' / '.join('`%s`' % x for x in p1[i + 1:j])))
    dw = []
    for m in re.finditer(r'\b([A-Za-z]{2,})\s+\1\b', body):
        if m.group(1).lower() in ('that', 'had'): continue
        pg = body[:m.start()].count('=== PAGE ')
        dw.append('p.%d `%s`' % (pg, ' '.join(body[max(0, m.start() - 40):m.end() + 25].split())))
    print('- doubled words in the rendered text (page, context): %d%s' % (len(dw), ''.join('\n  - ' + x for x in dw)))
    src = '\n'.join(strip_comments(l) for l in L)
    src = re.sub(r'\\[A-Za-z]+\*?', ' ', src)
    ds = []
    for m in re.finditer(r'\b([A-Za-z]{2,})\s+\1\b', src):
        if m.group(1).lower() in ('that', 'had'): continue
        ln = src[:m.start()].count('\n') + 1
        ds.append('l.%d `%s`' % (ln, ' '.join(src[max(0, m.start() - 30):m.end() + 20].split())))
    print('- doubled words in the .tex source (comments and macro names removed; prose, captions and tables): %d%s' % (len(ds), ''.join('\n  - ' + x for x in ds)))
    fb = []
    t0 = next(i for i, l in enumerate(L) if l.startswith('\\begin{document}'))
    envdepth = 0
    for i, l in enumerate(L[t0:], t0):
        c = strip_comments(l)
        envdepth += len(re.findall(r'\\begin\{(tabular|table|figure|array|align)', c)) - len(re.findall(r'\\end\{(tabular|table|figure|array|align)', c))
        if envdepth <= 0 and re.search(r'(?<!\\)\\\\(?!\w)', c) and not c.startswith('\\title'): fb.append((i + 1, c.strip()[:100]))
    print('- forced `\\\\` line breaks outside tables, figures and the title: %d%s' % (len(fb), ''.join('\n  - l.%d `%s`' % x for x in fb)))


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'todo': todo(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    elif cmd == 'cite': sys.exit(1 if cite(sys.argv[2]) else 0)
    elif cmd == 'headheld': sys.exit(1 if headheld(*sys.argv[2:6]) else 0)
    elif cmd == 'defects': defects(sys.argv[2], sys.argv[3])
