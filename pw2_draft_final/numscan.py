"""Independent main-text number scanner for the PW-2 closing audit (written fresh; shares no code with pw2_draft_numbers).

Main text = from the \\title line to the line before `\\subsubsection*{AI use statement}` (the abstract included).
Excluded, character by character (replaced with blanks so line numbers survive): LaTeX comments (from the first
unescaped %), figure/table environments (captions and tables with them), \\figslot{..}{..}{..} calls, and the arguments
of \\label \\ref \\eqref \\pageref \\cite \\citep \\citet \\includegraphics \\hspace \\vspace \\setlength \\begin{..}{..}.
Tokens: every digit run (with inner , or . : 2,367 / 0.97) and every spelled cardinal (zero ... twenty, thirty ...
ninety, hundred, thousand, million, billion, half, twice, dozen). Nothing is filtered at this stage.
usage: python3 numscan.py DRAFT.tex  -> JSON list on stdout
"""
import json, re, sys

WORDS = ('zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen '
         'seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand million '
         'billion half twice dozen').split()
WRE = re.compile(r'(?<![A-Za-z])(' + '|'.join(WORDS) + r')(?![A-Za-z])', re.I)
DRE = re.compile(r'\d+(?:[.,]\d+)*')


def bal(s, i):
    """index of the brace closing the one at s[i] (s[i] == '{')"""
    d = 0
    for k in range(i, len(s)):
        c = s[k]
        if c == '\\':
            continue
        if c == '{' and (k == 0 or s[k - 1] != '\\'): d += 1
        elif c == '}' and (k == 0 or s[k - 1] != '\\'):
            d -= 1
            if d == 0: return k
    raise ValueError('unbalanced at %d' % i)


def blank(s, a, b):
    return s[:a] + ''.join('\n' if c == '\n' else ' ' for c in s[a:b]) + s[b:]


def region(lines):
    t0 = next(i for i, l in enumerate(lines) if l.startswith('\\title{'))
    t1 = next(i for i, l in enumerate(lines) if l.startswith('\\subsubsection*{AI use statement}'))
    return t0, t1


def masked(path):
    L = open(path).read().split('\n')
    t0, t1 = region(L)
    # comments
    out = []
    for l in L:
        m = re.search(r'(?<!\\)%', l)
        out.append(l if not m else l[:m.start()] + ' ' * (len(l) - m.start()))
    s = '\n'.join(out)
    # restrict to region: blank everything outside
    starts = [0]
    for l in L: starts.append(starts[-1] + len(l) + 1)
    a, b = starts[t0], starts[t1]
    s = blank(s, 0, a); s = blank(s, b, len(s))
    # floats
    for env in ('figure\\*', 'figure', 'table\\*', 'table', 'wrapfigure'):
        while True:
            m = re.search(r'\\begin\{' + env + r'\}', s)
            if not m: break
            e = re.search(r'\\end\{' + env + r'\}', s[m.end():])
            s = blank(s, m.start(), m.end() + e.end())
    while True:
        m = re.search(r'\\figslot\s*\{', s)
        if not m: break
        k = m.end() - 1
        for _ in range(3):
            k = bal(s, k) + 1
            while k < len(s) and s[k] in ' \n': k += 1
            if _ < 2: assert s[k] == '{', s[k:k + 20]
        s = blank(s, m.start(), k)
    # macro arguments that are keys or lengths
    for mac in ('label', 'ref', 'eqref', 'pageref', 'cite', 'citep', 'citet', 'citealp', 'includegraphics',
                'hspace', 'vspace', 'setlength', 'begin', 'end', 'bibliographystyle', 'bibliography'):
        for m in list(re.finditer(r'\\' + mac + r'\*?\s*(\[[^\]]*\])?\s*\{', s)):
            k = bal(s, m.end() - 1)
            s = blank(s, m.start(), k + 1)
    return L, s, t0, t1


def section_of(L, i):
    sec = 'Title'
    for j in range(0, i + 1):
        l = L[j]
        if l.startswith('\\begin{abstract}'): sec = 'Abstract'
        m = re.match(r'\\section\{([^}]*)\}', l)
        if m: sec = m.group(1)
    return sec


def scan(path):
    L, s, t0, t1 = masked(path)
    ML = s.split('\n')
    secs = {}
    cur = 'Title'
    for i, l in enumerate(L):
        if l.startswith('\\begin{abstract}'): cur = 'Abstract'
        m = re.match(r'\\section\{([^}]*)\}', l)
        if m: cur = m.group(1)
        secs[i] = cur
    toks = []
    for i in range(t0, t1):
        ml = ML[i]
        for m in DRE.finditer(ml):
            a, b = m.span()
            # the enclosing word (letters/digits glued on either side)
            wa, wb = a, b
            while wa > 0 and (ml[wa - 1].isalnum()): wa -= 1
            while wb < len(ml) and (ml[wb].isalnum()): wb += 1
            toks.append(dict(line=i + 1, col=a, kind='digit', tok=m.group(0), word=ml[wa:wb], section=secs[i],
                             ctx=ml[max(0, a - 70):b + 70].strip(), code=ml.strip()))
        for m in WRE.finditer(ml):
            a, b = m.span()
            toks.append(dict(line=i + 1, col=a, kind='word', tok=m.group(0), word=m.group(0), section=secs[i],
                             ctx=ml[max(0, a - 70):b + 70].strip(), code=ml.strip()))
    toks.sort(key=lambda t: (t['line'], t['col']))
    return toks


if __name__ == '__main__':
    json.dump(scan(sys.argv[1]), sys.stdout, indent=0)
