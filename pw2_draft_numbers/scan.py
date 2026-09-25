"""Every number in the MAIN TEXT of a draft (title through the end of Discussion; abstract included; captions, tables,
figures and \\figslot arguments excluded; LaTeX comments excluded; \\ref/\\label/\\cite keys and file paths masked).
Digit tokens use the provenance checker's regex (pw1_draft_sweeps/texnum.py) plus \\sci{m}{e} whole; spelled-out
cardinals (zero..twelve, twenty, thirty, hundred, thousand, half, twice, double, ...) are listed separately.
usage: python3 scan.py DRAFT.tex [--json OUT.json]"""
import re, sys, json, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pw1_draft_sweeps'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pw2_draft_sections'))
from texnum import split_comment, NUM, SCI
import headheld

WORDS = r'zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twenty|thirty|forty|fifty|sixty|hundred|thousand|million|billion|half|twice|double|triple|dozen'
WNUM = re.compile(r'(?<![A-Za-z\\])(' + WORDS + r')(?![A-Za-z])', re.I)
MASK = re.compile(r'\\(ref|label|eqref|cite[a-z]*|citet|citep|citealp|includegraphics|input|bibliography[a-z]*)\*?(\[[^\]]*\])?\{[^}]*\}')
SECT = re.compile(r'\\(section|subsection|subsubsection)\*?\{([^}]*)\}')

def region(L):
    t0 = next(i for i, l in enumerate(L) if l.startswith(r'\title{'))
    t1 = next(i for i, l in enumerate(L) if l.startswith(r'\subsubsection*{AI use statement}'))
    return t0, t1

def float_mask(L, t0, t1):
    """True for every line inside a table/figure environment or a \\figslot{...}{...}{...} call."""
    m = [False] * len(L); i = t0
    while i < t1:
        code = split_comment(L[i])[0]
        if re.search(r'\\begin\{(table|figure)\*?\}', code):
            j = i
            while not re.search(r'\\end\{(table|figure)\*?\}', split_comment(L[j])[0]): j += 1
            for k in range(i, j + 1): m[k] = True
            i = j + 1; continue
        if r'\figslot{' in code and not code.lstrip().startswith(r'\newcommand'):
            depth, args, j = 0, 0, i
            s = code[code.index(r'\figslot{') + len(r'\figslot'):]
            while True:
                for ch in s:
                    if ch == '{': depth += 1
                    elif ch == '}':
                        depth -= 1
                        if depth == 0: args += 1
                if args >= 3: break
                j += 1; s = split_comment(L[j])[0]
            for k in range(i, j + 1): m[k] = True
            i = j + 1; continue
        i += 1
    return m

def headheld_lines(path):
    out = {}
    for name, rng, _, _ in headheld.rows(path):
        a, b = [int(x) for x in rng.split('–')] if '–' in rng else (int(rng), int(rng))
        for k in range(a, b + 1): out[k] = name
    return out

def scan(path):
    L = open(path).read().split('\n')
    t0, t1 = region(L); fm = float_mask(L, t0, t1); hh = headheld_lines(path)
    sec, sub, rows = 'Title/Abstract', '', []
    for i in range(t0, t1):
        code = split_comment(L[i])[0]
        for m in SECT.finditer(code):
            if m.group(1) == 'section': sec, sub = m.group(2), ''
            else: sub = m.group(2)
        if r'\begin{abstract}' in code: sec = 'Abstract'
        if fm[i]: continue
        c = MASK.sub(lambda m: ' ' * len(m.group(0)), code)
        toks = []
        for m in SCI.finditer(c): toks.append((m.start(), 'SCI(%s,%s)' % (m.group(1), m.group(2)), 'digit'))
        c2 = SCI.sub(lambda m: ' ' * len(m.group(0)), c)
        for m in NUM.finditer(c2):
            t = m.group(1).rstrip('.,')
            if t: toks.append((m.start(), t, 'digit'))
        for m in WNUM.finditer(c2): toks.append((m.start(), m.group(1), 'word'))
        for pos, t, kind in sorted(toks):
            rows.append(dict(line=i + 1, pos=pos, section=sec, sub=sub, token=t, kind=kind, headheld=hh.get(i + 1, ''),
                             context=code))
    return rows

if __name__ == '__main__':
    rows = scan(sys.argv[1])
    if '--json' in sys.argv: json.dump(rows, open(sys.argv[sys.argv.index('--json') + 1], 'w'), indent=1)
    from collections import Counter
    d = Counter((r['section'], r['kind'], bool(r['headheld'])) for r in rows)
    for k, v in sorted(d.items()): print(k, v)
    print('TOTAL digit', sum(1 for r in rows if r['kind'] == 'digit'), 'word', sum(1 for r in rows if r['kind'] == 'word'))
