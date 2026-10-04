"""Records (under .claude/outputs, paper/package/this folder excluded) containing ALL tokens of a group verbatim."""
import re, json
from pathlib import Path
G = {'deletion-degraded (l.574-575)': ['-0.3291', '-0.3078', '+0.0074'],
     'zero-dynamics (l.664-677)': ['1.7244', '2.6682', '0.050000', '7.447', '7.875', '7.882', '3.971', '1.720', '0.25000000', '-1.50000000'],
     'retention-law (l.686, 696-697)': ['0.902', '-1.0020', '+1.116'],
     'capacity range (l.686)': ['0.29', '1.71']}
EX = ('w25-verbatim', 'w17-paper', 'pw1', 'pw2', 'paper-visuals', 'w24-repro-package', 'pw1-paper', 'pw1-paper-cited', 'pw2-desktop-snapshots')
def pat(t):
    v = [t, t.replace('-', '−'), t.lstrip('+-')] if t[0] in '+-' else [t]
    return [re.compile(r'(?<![0-9.,])' + re.escape(x) + r'(?![0-9])') for x in dict.fromkeys(v)]
files = [p for p in Path('.').rglob('*') if p.suffix in ('.md', '.json', '.txt') and p.parts[0] not in EX and p.is_file() and p.stat().st_size < 50_000_000]
res = {}
for g, toks in G.items():
    ps = {t: pat(t) for t in toks}; hits = []
    for f in files:
        s = f.read_text(errors='replace')
        if all(any(p.search(s) for p in ps[t]) for t in toks): hits.append(str(f))
    res[g] = hits; print(g, len(hits), hits[:6])
json.dump(res, open('w25-verbatim/cooccur.json', 'w'), indent=1)
