"""Verbatim re-check of the final draft's sourced numbers. Token regex, scope rules, variants and
matching are copied from pass 7 (w23/fold:w23_fold/gen_cells_pass7.py). Nothing computed or rounded.
A: token under a surviving % SRC / % SRCBLOCK scope in the final draft -> checked against the first file
   of that scope that resolves (then the others).
B: token with no scope -> provenance recovered from the previous full draft: lines there under a scope that
   contain the same token verbatim; the one most similar to the final line wins (similarity printed).
usage: verbatim_recheck.py <final.tex> <prev.tex> <evidence base dir> <out.json>"""
import difflib, json, re, sys
from pathlib import Path
FIN, PREV, BASE, OUT = map(Path, sys.argv[1:5])
TOK = re.compile(r"(?<![\w.\\])([+\-−]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|[+\-−]?\d+\.\d+|\d{4,}|\d+ of \d+|\d+/\d+)(?![\w])")
def scopes(lines):
    block, out = None, {}
    for i, raw in enumerate(lines, 1):
        s = raw.strip()
        if s.startswith('% SRCBLOCK:'):
            block = [p.strip() for p in s[len('% SRCBLOCK:'):].split(';')]; continue
        if s.startswith('% ENDSRC'):
            block = None; continue
        body, _, com = raw.partition('% SRC:')
        body = re.split(r'(?<!\\)%', body)[0].replace('--', ' – ')
        srcs = [p.strip() for p in com.split(';') if p.strip()] if com else (block or [])
        out[i] = (body, srcs)
    return out
def variants(t):
    v = [t]; core = t.lstrip('+-−')
    if core != t: v += [t.replace('−', '-'), t.replace('-', '−'), core]
    if re.fullmatch(r'\d{1,3}(?:,\d{3})+(?:\.\d+)?', core): v.append(core.replace(',', ''))
    if re.fullmatch(r'\d{5,}', core): v.append(f"{int(core):,}")
    m = re.fullmatch(r'(\d+) of (\d+)', core)
    if m: v += [f"{m.group(1)}/{m.group(2)}", f"{m.group(1)} / {m.group(2)}"]
    return list(dict.fromkeys(v))
_c = {}
def src_lines(rel):
    if rel not in _c:
        # SRC paths come in two styles: '../../outputs/x' relative to the paper dir, and bare 'x' relative to
        # the evidence root (the paper dir's parent). Try both; first hit wins.
        _c[rel] = None
        for base in (BASE, BASE.parent):
            q = (base / rel).resolve()
            if q.is_file():
                _c[rel] = q.read_text(encoding='utf-8', errors='replace').splitlines(); break
    return _c[rel]
def find(t, srcs):
    for rel in srcs:
        L = src_lines(rel)
        if L is None: continue
        for var in variants(t):
            pat = re.compile(r'(?:(?<=[0-9]\.\.)|(?<![0-9.,]))' + re.escape(var) + r'(?![0-9])')  # also accept the '..' range form (0.1561..0.1565)
            for i, ln in enumerate(L, 1):
                if pat.search(ln): return rel, i, var
    return None
fin = scopes(FIN.read_text().splitlines()); prev = scopes(PREV.read_text().splitlines())
prev_by_tok = {}
for j, (b, s) in prev.items():
    if s:
        for m in TOK.finditer(b): prev_by_tok.setdefault(m.group(1), []).append(j)
rows = []
for i, (body, srcs) in fin.items():
    for m in TOK.finditer(body):
        t = m.group(1); r = {'line': i, 'token': t, 'context': body.strip()[:160]}
        if srcs:
            r['tier'] = 'A'; r['scope'] = srcs
        else:
            cands = prev_by_tok.get(t, [])
            if cands:
                sim = [(difflib.SequenceMatcher(None, body.strip(), prev[j][0].strip()).ratio(), j) for j in cands]
                s_, j = max(sim); r['tier'] = 'B'; r['scope'] = prev[j][1]; r['prev_line'] = j; r['similarity'] = round(s_, 3)
            else:
                r['tier'] = 'U'; r['scope'] = []
        if r['scope']:
            unresolved = [p for p in r['scope'] if src_lines(p) is None]
            if unresolved: r['unresolved_paths'] = unresolved
            f = find(t, r['scope'])
            r['status'] = 'FOUND' if f else 'NOT-FOUND'
            if f: r['found_in'], r['found_line'], r['variant'] = f
        else:
            r['status'] = 'UNSOURCED'
        rows.append(r)
from collections import Counter
summ = {'tokens': len(rows), 'by_tier': Counter(r['tier'] for r in rows), 'by_tier_status': Counter(f"{r['tier']}:{r['status']}" for r in rows),
        'found_total': sum(r['status'] == 'FOUND' for r in rows)}
json.dump({'final_tex': str(FIN), 'prev_tex': str(PREV), 'summary': summ, 'rows': rows}, open(OUT, 'w'), indent=1, ensure_ascii=False, default=list)
print(json.dumps(summ, indent=1, default=dict))
