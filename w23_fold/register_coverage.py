"""w23 fold: how many number tokens in the installed drafts does the w18 register check as a DRAFT cell?
Read-only. Token regex = pass 5's scan_draft.py; scope = every SRC / SRCBLOCK (not only w15).
A (line, token) is COVERED if some register cell whose source is the pinned pass-5 copy of that draft, at that line,
has a value containing the token. usage: register_coverage.py <out json>"""
import json, re, sys, collections, hashlib
from pathlib import Path
REG = Path('/Users/batmac/clu/outputs/w18-seeds-and-tables'); PAPER = Path('/Users/batmac/clu/outputs/w17-paper')
TOK = re.compile(r"(?<![\w.\\])([+\-−]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|[+\-−]?\d+\.\d+|\d{4,}|\d+ of \d+|\d+/\d+)(?![\w])")
DRAFTS = {'DRAFT-v2.tex': '@here/draft_pass5/DRAFT-v2.pass5-6cecd36a.tex', 'DRAFT-v2-style.tex': '@here/draft_pass5/DRAFT-v2-style.pass5-72327303.tex'}
man = json.load(open(REG / 'TABLES-w18.json'))
bysrc = collections.defaultdict(list)
for cid, c in man['cells'].items():
    bysrc[(c['source'], c.get('line'))].append(c['value'])
def scan(path):
    out, block = [], None
    for i, raw in enumerate(Path(path).read_text(encoding='utf-8').splitlines(), 1):
        s = raw.strip()
        if s.startswith('% SRCBLOCK:'): block = [p.strip() for p in s[len('% SRCBLOCK:'):].split(';')]; continue
        if s.startswith('% ENDSRC'): block = None; continue
        body, _, com = raw.partition('% SRC:')
        body = re.split(r'(?<!\\)%', body)[0].replace('--', ' – ')
        srcs = [p.strip() for p in com.split(';')] if com else (block or [])
        if not srcs: continue
        for m in TOK.finditer(body): out.append((i, m.group(1), srcs))
    return out
res = {}
for d, pin in DRAFTS.items():
    live = (PAPER / d).read_bytes(); pinb = (REG / pin.replace('@here/', '')).read_bytes()
    toks = scan(PAPER / d); unit_all = collections.Counter(); unit_cov = collections.Counter()
    for line, t, srcs in toks:
        u = sorted({s.split('/')[0] for s in srcs})[0] if srcs else '(none)'
        unit_all[u] += 1
        vals = bysrc.get((pin, line), [])
        if any(t.lstrip('+−-') in v for v in vals): unit_cov[u] += 1
    res[d] = {'live_md5': hashlib.md5(live).hexdigest(), 'live_equals_pass5_pin': live == pinb,
              'tokens_under_any_SRC': len(toks), 'covered_by_a_draft_cell': sum(unit_cov.values()),
              'per_unit': {u: {'tokens': n, 'covered': unit_cov[u]} for u, n in sorted(unit_all.items(), key=lambda x: -x[1])}}
json.dump(res, open(sys.argv[1], 'w'), indent=1, ensure_ascii=False)
for d, r in res.items():
    print(d, {k: v for k, v in r.items() if k != 'per_unit'})
    for u, v in list(r['per_unit'].items())[:40]: print('   ', u, v)
