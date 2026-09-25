"""The seven tells of PW-2 item 1, one detector each (rules below), run on segment.py units.
usage: python3 detect.py DRAFT.tex [--json OUT] [--list TELL]   (TELL in T1..T7)
Scope (brief): T2, T3, T6, T7 main text and appendices (prose, items, quote; T6 also captions and tables);
T5 main-text headings only (\\section, \\subsection, \\paragraph run-in titles; Title/Abstract..Discussion);
T4 main text running prose only (Abstract..Discussion); T1 main text and appendices (prose/items/quote).

T1 triad: (a) the phrase 'three things'; (b) an anaphoric triple: three list members opening with the same
   one or two words ('the same A, the same B, the same C'; 'they X, they Y, and we Z' counts via 'X, Y, and Z'
   with a repeated opener).
T2 contrast: (a) 'not X(,) but Y'; (b) 'X, not Y' and 'X, never Y' (a comma, then not/never; 'not only' and
   'not yet' excluded); (c) 'neither ... nor'. Counted per sentence occurrence.
T3 self-narration: a fixed phrase list (PHRASES below; the brief's seven examples + the same family found on
   reading both drafts).
T4 numbers: a digit token with more than two significant figures, or a numeric range ('A to B', 'A--B', 'A-B'
   between numbers), in running text (captions, tables and math-only identifiers are out of scope).
T5 heading: title contains ':' , '(' , a comma apposition, or opens with Why/What/How/Toward/Reading/Deleting,
   or contains a finite verb from VERBS.
T6 coinage: 'naive unit' after its first mention; 'leaving note'; 'thickness'; 'one measured clue';
   and paragraphs with 5 or more n-CLU tokens (n-CLU, n-CLU-REL).
T7 rhythm (list items only when they carry >= 40 words; short bullets are exempt): (a) a unit with a long sentence (>= LONG words) but no short sentence (<= 8 words) directly after
   a long one; (b) a paragraph opening with a connective (CONNECTIVES); (c) a sentence opening 'This section'.
"""
import re, sys, os, json
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(__file__))
from segment import units, MAIN

LONG = 18; SHORT = 8
PHRASES = [r'which we print', r'\bwe print\b', r'printed, never', r'printed and not', r'never netted', r'not netted',
           r'measured, not (assumed|hedged)', r'stated? plainly', r'cited in one sentence', r'one sentence each',
           r'in one sentence', r'worth a sentence', r'merits a sentence', r'extended to manufacture',
           r'prints its seed count', r'[Ss]eed count is a column', r'limitations printed', r'convergence printed',
           r'verbatim in substance', r'rather than round away', r'we report it as', r'for completeness',
           r'all printed here', r'[Ss]tatus discipline', r'[Aa]ppendix-maximalism', r'nothing is omitted',
           r'stated in the introduction', r'never pooled', r'reported, never tuned', r'not widened after the fact',
           r'travels with', r'travel with', r'we report (it|them)\b']
CONNECTIVES = ['Therefore', 'Hence', 'Thus', 'However', 'We now', 'So', 'Moreover', 'Furthermore', 'Consequently',
               'Accordingly', 'Also', 'Still', 'Yet', 'Indeed', 'In particular', 'Specifically', 'Concretely', 'Nor']
VERBS = r'\b(is|are|changes|knows|bounds|look|removes?|can)\b'

ABBR = re.compile(r'\b(Secs?|Tabs?|Figs?|Apps?|Eqs?|Alg|e\.g|i\.e|vs|et al|cf|resp)\.')
def norm(code, drop_title=True):
    t = code
    if drop_title:
        t = re.sub(r'\\paragraph\{', r'\\PARA{', t)
        t = strip_braced(t, r'\PARA')
    t = t.replace('\\ ', ' ')
    t = ABBR.sub(lambda m: m.group(1).replace('.', ''), t)
    t = re.sub(r'\\(cite[a-z]*|citet|citep|citealp|label)\*?(\[[^\]]*\])?\{[^}]*\}', '', t)
    t = re.sub(r'\\(ref|eqref)\{[^}]*\}', 'R', t)
    t = re.sub(r'\\sci\{[^}]*\}\{[^}]*\}', 'N', t)
    t = re.sub(r'\$[^$]*\$', 'M', t)
    t = re.sub(r'\\(emph|textbf|textsc|texttt|textit|mathrm|text)\{', '{', t)
    t = re.sub(r'\\[A-Za-z]+\*?', ' ', t)
    t = t.replace('{', '').replace('}', '').replace('~', ' ')
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def strip_braced(t, macro):
    while macro + '{' in t:
        k = t.index(macro + '{'); d = 0
        for p in range(k + len(macro), len(t)):
            if t[p] == '{': d += 1
            elif t[p] == '}':
                d -= 1
                if d == 0: break
        t = t[:k] + t[p + 1:]
    return t

SPLIT = re.compile(r'(?<=[.?!])[\'"’”)]*\s+(?=[A-Z0-9(`\'"])')
def sentences(t):
    return [s for s in SPLIT.split(t) if s.strip()]
def nwords(s):
    return len([w for w in s.split() if re.search(r'[A-Za-z0-9]', w)])

T2A = re.compile(r'\bnot\b(?! only\b| yet\b)[^.;:]{1,140}?,?\s+but\b')
T2B = re.compile(r',\s+(not|never)\b(?! only\b| yet\b)')
T2C = re.compile(r'\bneither\b.{1,160}?\bnor\b')
T1A = re.compile(r'\bthree things\b')
T1B = re.compile(r'\b((?:[A-Za-z]{2,} )?(?!the\b|and\b|of\b|to\b|in\b|on\b|a\b)[A-Za-z]{2,}) [^,;.]{1,40}, \1 [^,;.]{1,40},? (?:and )?\1 ')
SIGNUM = re.compile(r'(?<![\w.])[-+−]?\d[\d,]*(?:\.\d+)?')
RANGE = re.compile(r'(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?\s*(?:to|--|–|-)\s*[-+]?\d[\d,]*(?:\.\d+)?(?![\w])')
T6P = [('naive unit', re.compile(r'naive (AI )?unit\b(?!s)')), ('leaving note', re.compile(r'leaving note', re.I)),
       ('thickness', re.compile(r'\bthickness\b', re.I)), ('one measured clue', re.compile(r'one measured clue'))]
NCLU = re.compile(r'n-CLU')

def sigfigs(tok):
    d = re.sub(r'[^0-9.]', '', tok)
    if '.' in d:
        d = d.replace('.', '').lstrip('0')
        return len(d)
    return len(d.lstrip('0')) if d.strip('0') else 1

def is_main(sec): return sec in ['Abstract'] + MAIN
def detect(path):
    U = units(path)
    hits = []   # dict(tell, section, sub, line, hh, text)
    first_naive = True
    for u in U:
        sec, kind = u['section'], u['kind']
        prose = kind in ('prose', 'item', 'quote')
        ln = u['lines'][0]
        def H(tell, txt, extra=''):
            hits.append(dict(tell=tell, section=sec, sub=u['sub'], line=ln, kind=kind, hh=u['headheld'], text=txt, rule=extra))
        if kind == 'heading':
            if not (is_main(sec) and u['level'] in ('section', 'subsection', 'paragraph')): continue
            t = u['title']
            why = []
            if ':' in t: why.append('colon')
            if '(' in t: why.append('parenthetical')
            if re.search(r', (never|verbatim|at|on|in|a|an|the|what|and which|stated)\b', t): why.append('comma apposition')
            if re.match(r'(Why|What|How|Toward|Reading|Deleting)\b', t): why.append('clause opener')
            if re.search(VERBS, re.sub(r'\bthat\b.*', '', t)): why.append('finite verb')
            if why: H('T5', t, '; '.join(why))
            continue
        if kind in ('caption', 'table'):
            t = norm(u['code'], False)
            for name, rx in T6P:
                for m in rx.finditer(t):
                    if name == 'naive unit' and first_naive: first_naive = False; continue
                    H('T6', m.group(0), name)
            continue
        t = norm(u['code'])
        # T6
        for name, rx in T6P:
            for m in rx.finditer(u['code'] if name != 'naive unit' else t):
                if name == 'naive unit' and first_naive: first_naive = False; continue
                H('T6', m.group(0), name)
        nc = len(NCLU.findall(t))
        if nc >= 5: H('T6', '%d n-CLU tokens' % nc, 'n-CLU >= 5 per paragraph')
        # T1
        for m in T1A.finditer(t): H('T1', m.group(0), 'three things')
        for m in T1B.finditer(t): H('T1', m.group(0), 'anaphoric triple')
        # T2
        for rx, nm in ((T2A, 'not X but Y'), (T2B, 'X, not/never Y'), (T2C, 'neither nor')):
            for m in rx.finditer(t): H('T2', t[max(0, m.start() - 50):m.end() + 30], nm)
        # T3
        for p in PHRASES:
            for m in re.finditer(p, t): H('T3', t[max(0, m.start() - 40):m.end() + 30], p)
        # T4 (main-text running prose only)
        if is_main(sec):
            c = re.sub(r'\\(ref|label|cite[a-z]*)\{[^}]*\}', '', u['code'])
            c = re.sub(r'\$[^$]*\$', ' M ', c)
            c = strip_braced(c, r'\PARA') if False else c
            for m in RANGE.finditer(c): H('T4', m.group(0), 'range')
            for m in SIGNUM.finditer(RANGE.sub(' ', c)):
                if sigfigs(m.group(0)) > 2: H('T4', m.group(0), '>2 significant figures')
        # T7 (on the whole TeX paragraph when a Head-held line cut it into runs; evaluated once, at its first run)
        if 'para' in u:
            if u['lines'][0] != u['paralines'][0]: continue
            full = '\n'.join(x['code'] for x in U if x.get('para') == u['para'])
            t = norm(full)
            if u['headheld'] and not all(x['headheld'] for x in U if x.get('para') == u['para']):
                ln = u['lines'][0]; u = dict(u, headheld='')   # mixed paragraph: attributed to the editable part
        S = sentences(t)
        if prose and (kind != 'item' or nwords(t) >= 40):
            W = [nwords(s) for s in S]
            if any(w >= LONG for w in W) and not any(W[i] >= LONG and W[i + 1] <= SHORT for i in range(len(W) - 1)):
                H('T7', ' | '.join('%d' % w for w in W), 'no short sentence after a long one')
            if kind == 'prose' and S:
                for c in CONNECTIVES:
                    if re.match(re.escape(c) + r'\b', S[0]): H('T7', S[0][:60], 'connective opener')
            for s in S:
                if s.startswith('This section'): H('T7', s[:60], 'This section')
    return U, hits

def summary(hits):
    c = Counter((h['tell'], h['section']) for h in hits)
    return c

if __name__ == '__main__':
    U, hits = detect(sys.argv[1])
    if '--json' in sys.argv: json.dump(hits, open(sys.argv[sys.argv.index('--json') + 1], 'w'), indent=1)
    if '--list' in sys.argv:
        T = sys.argv[sys.argv.index('--list') + 1]
        for h in hits:
            if h['tell'] == T: print('L%-5d %-28s %-3s %-22s %s' % (h['line'], h['section'][:28], 'HH' if h['hh'] else '', h['rule'][:22], h['text'][:150]))
    else:
        c = summary(hits); secs = []
        for u in U:
            if u['section'] not in secs: secs.append(u['section'])
        print('section'.ljust(40), ' '.join('T%d' % i for i in range(1, 8)))
        for s in secs:
            print(s[:40].ljust(40), ' '.join('%2d' % c[('T%d' % i, s)] for i in range(1, 8)))
        print('TOTAL'.ljust(40), ' '.join('%2d' % sum(v for (t, s), v in c.items() if t == 'T%d' % i) for i in range(1, 8)))
