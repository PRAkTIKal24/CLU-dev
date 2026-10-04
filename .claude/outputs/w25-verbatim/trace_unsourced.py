"""For each UNSOURCED token (layout widths at final lines 35, 36, 295 excluded): its scope in the previous
full draft (inline or block), and every evidence file (*.md, *.json, *.txt, *.csv under .claude/outputs, paper and
this folder excluded) containing it verbatim with pass 7's boundary rule."""
import json, re, os, sys, importlib.util
from pathlib import Path
sys.argv = ['x', 'w25-verbatim/src/DRAFT-v2.tex', 'w25-verbatim/DRAFT-v2.prev-42pp.tex', 'w17-paper', '/tmp/x.json']
spec = importlib.util.spec_from_file_location('vr', 'w25-verbatim/verbatim_recheck.py'); vr = importlib.util.module_from_spec(spec); spec.loader.exec_module(vr)
prev = vr.scopes(Path('w25-verbatim/DRAFT-v2.prev-42pp.tex').read_text().splitlines())
d = json.load(open('w25-verbatim/recheck.json'))
EX = ('w25-verbatim', 'w17-paper', 'pw1', 'pw2', 'paper-visuals', 'w24-repro-package', 'pw1-paper', 'pw1-paper-cited', 'pw2-desktop-snapshots')
files = [p for p in Path('.').rglob('*') if p.suffix in ('.md', '.json', '.txt', '.csv') and p.parts[0] not in EX and p.is_file() and p.stat().st_size < 50_000_000]
txt = {}
out = []
for r in d['rows']:
    if r['status'] != 'UNSOURCED' or r['line'] in (35, 36, 295): continue
    t = r['token']; pats = [re.compile(r'(?:(?<=[0-9]\.\.)|(?<![0-9.,]))' + re.escape(v) + r'(?![0-9])') for v in vr.variants(t)]
    pl = [(j, s) for j, (b, s) in prev.items() if any(p.search(b) for p in pats)]
    hits = []
    for f in files:
        if f not in txt: txt[f] = f.read_text(errors='replace')
        if any(p.search(txt[f]) for p in pats): hits.append(str(f))
    out.append({'line': r['line'], 'token': t, 'prev': [(j, s) for j, s in pl][:3], 'n_evidence': len(hits), 'evidence': hits[:4]})
    print(f"{r['line']:>4} {t:>12} | prev {[(j, (s[0][-45:] if s else None)) for j, s in pl][:2]} | ev {len(hits)}: {hits[:2]}")
json.dump(out, open('w25-verbatim/unsourced_trace.json', 'w'), indent=1)
print('files searched:', len(files))
