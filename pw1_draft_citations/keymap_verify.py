"""usage: keymap_verify.py OLD.bib NEW.bib keymap.json -> markdown table old -> new -> matching field(s)."""
import json, sys
from bibparse import entries, norm_title, doi, arxiv

old, new, km = entries(sys.argv[1]), entries(sys.argv[2]), json.load(open(sys.argv[3]))
print(f'old entries {len(old)}, new entries {len(new)}, keymap pairs {len(km)}; '
      f'keymap old keys == old bib keys: {set(km) == set(old)}; keymap values == new bib keys: {set(km.values()) == set(new)}; '
      f'one-to-one: {len(set(km.values())) == len(km)}\n')
print('| # | old key | new key | DOI | arXiv | title (normalised) | verdict |\n|---|---|---|---|---|---|---|')
bad = 0
for n, (o, k) in enumerate(sorted(km.items()), 1):
    a, b = old.get(o, {}), new.get(k, {})
    cells, agree = [], []
    for name, fn in (('DOI', doi), ('arXiv', arxiv), ('title', lambda e: norm_title(e.get('title', '')))):
        x, y = fn(a), fn(b)
        if x and y:
            cells.append('agree' if x == y else f'DIFFER `{x[:40]}` / `{y[:40]}`')
            if x == y:
                agree.append(name)
        else:
            cells.append(f'n/a (old `{x[:30]}`, new `{y[:30]}`)' if (x or y) else 'n/a')
    v = 'MATCH on ' + ' + '.join(agree) if agree else 'NO MATCHING FIELD'
    bad += not agree
    print(f'| {n} | `{o}` | `{k}` | ' + ' | '.join(cells) + f' | {v} |')
print(f'\npairs with no agreeing field: {bad}')
