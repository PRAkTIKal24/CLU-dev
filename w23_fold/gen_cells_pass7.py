"""w23 fold (register pass 6 of w18 seeds-and-tables; registry pass 7): the numbers W22 wrote into the drafts, as cells.

Scope: every number token (pass 5's scan_draft.py regex) on a line of the pinned pass-5 draft that is NOT in the pinned
pass-4 draft (difflib insert/replace hunks), i.e. text written by w22 draft-rulings, figures-redraw2, figures3 and seeds5,
under a `% SRC` / `% SRCBLOCK` scope, and not already a pass-5 draft cell on that line.
For each token, two cells:
  1. a DRAFT cell: (pinned pass-5 draft, line, token), verbatim on the line;
  2. a SOURCE cell: (the first file its SRC names that resolves under the evidence root, first line, variant) where the
     variant is the token or a mechanical spelling of it (sign dropped, − for -, comma grouping added/removed, 'a of b' as
     'a/b' or 'a / b'), found verbatim and not glued to another digit.
Tokens with no source line are NOT registered; they are listed (unsourced.json) as findings.
Nothing is computed or rounded. usage: gen_cells_pass7.py <register dir> <evidence root> <out dir>"""
import difflib, json, re, sys, collections
from pathlib import Path

REG, EV, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)
TOK = re.compile(r"(?<![\w.\\])([+\-−]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|[+\-−]?\d+\.\d+|\d{4,}|\d+ of \d+|\d+/\d+)(?![\w])")
PAIRS = {'V': ('draft_pass5/DRAFT-v2.pass4-6caf80cd.tex', 'draft_pass5/DRAFT-v2.pass5-6cecd36a.tex', 'D5V'),
         'S': ('draft_pass5/DRAFT-v2-style.pass4-5a58d5d9.tex', 'draft_pass5/DRAFT-v2-style.pass5-72327303.tex', 'D5S')}
sys.path.insert(0, str(REG))
from cells_pass5 import CELLS5  # noqa: E402
p5_at = collections.defaultdict(list)
for cid, (a, ln, v) in CELLS5.items():
    p5_at[(a, ln)].append(v)


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
    v = [t]
    core = t.lstrip('+-−')
    if core != t:
        v += [t.replace('−', '-'), t.replace('-', '−'), core]
    if re.fullmatch(r'\d{1,3}(?:,\d{3})+(?:\.\d+)?', core):
        v.append(core.replace(',', ''))
    if re.fullmatch(r'\d{5,}', core):
        v.append(f"{int(core):,}")
    m = re.fullmatch(r'(\d+) of (\d+)', core)
    if m:
        v += [f"{m.group(1)}/{m.group(2)}", f"{m.group(1)} / {m.group(2)}"]
    return list(dict.fromkeys(v))


_cache = {}
def src_lines(rel):
    if rel not in _cache:
        p = EV / rel
        _cache[rel] = p.read_text(encoding='utf-8', errors='replace').splitlines() if p.is_file() else None
    return _cache[rel]


def find(t, srcs):
    for rel in srcs:
        L = src_lines(rel)
        if L is None:
            continue
        for var in variants(t):
            pat = re.compile(r'(?<![0-9.,])' + re.escape(var) + r'(?![0-9])')
            for i, ln in enumerate(L, 1):
                if pat.search(ln):
                    return rel, i, var
    return None


alias_of, SRC7, CELLS7, rows, unsourced, stats = {}, {'D5V': None, 'D5S': None}, {}, {'V': [], 'S': []}, [], {}
del SRC7['D5V'], SRC7['D5S']  # the pass-5 draft aliases are reused, not redefined
srcell = {}
for k, (p4, p5, dal) in PAIRS.items():
    A = (REG / p4).read_text(encoding='utf-8').splitlines(); B = (REG / p5).read_text(encoding='utf-8').splitlines()
    new = set()
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, A, B, autojunk=False).get_opcodes():
        if tag in ('insert', 'replace'):
            new.update(range(j1 + 1, j2 + 1))
    sc = scopes(B); n_tok = n_p5 = 0; no_scope = 0
    for ln in sorted(new):
        body, srcs = sc.get(ln, ('', []))
        toks = [m.group(1) for m in TOK.finditer(body)]
        if not toks:
            continue
        if not srcs:
            no_scope += len(toks); continue
        for t in toks:
            n_tok += 1
            if any(t.lstrip('+-−') in v for v in p5_at.get((dal, ln), [])):
                n_p5 += 1; continue
            assert t in B[ln - 1], (k, ln, t)
            hit = find(t, srcs)
            if not hit:
                unsourced.append({'draft': dal, 'line': ln, 'token': t, 'srcs': srcs,
                                  'resolvable': [s for s in srcs if src_lines(s) is not None]}); continue
            rel, sl, var = hit
            if rel not in alias_of:
                alias_of[rel] = f"S7N{len(alias_of) + 1}"; SRC7[alias_of[rel]] = rel
            key = (alias_of[rel], sl, var)
            if key not in srcell:
                srcell[key] = f"p7_f_{len(srcell) + 1}"; CELLS7[srcell[key]] = key
            did = f"p7_{k.lower()}_{len(rows[k]) + 1}"
            CELLS7[did] = (dal, ln, t)
            strong = len(re.sub(r'\D', '', t)) >= 4
            rows[k].append((did, srcell[key], strong))
    stats[k] = {'w22_lines': len(new), 'tokens_on_w22_lines_under_SRC': n_tok, 'tokens_on_w22_lines_without_SRC': no_scope,
                'already_pass5_draft_cells': n_p5, 'registered': len(rows[k]),
                'registered_strong_4plus_digits': sum(r[2] for r in rows[k]),
                'unsourced': sum(1 for u in unsourced if u['draft'] == dal)}
with open(OUT / 'cells_pass7.py', 'w', encoding='utf-8') as fh:
    fh.write('"""Pass-7 cell register (w23 fold; static, generated once by w23_fold/gen_cells_pass7.py, re-verified by every build).\n'
             'The numbers W22 wrote into the drafts (pinned pass-5 lines not in the pinned pass-4 text), each as a DRAFT cell on\n'
             'the pinned pass-5 draft and a SOURCE cell on the first file its SRC names. CELLS7: (alias, line, value) verbatim."""\n\n')
    fh.write(f"SRC7 = {json.dumps(SRC7, ensure_ascii=False)}\n\n")
    fh.write("CELLS7 = {\n" + ''.join(f"    {cid!r}: {tuple(v)!r},\n" for cid, v in CELLS7.items()) + "}\n")
json.dump({'stats': stats, 'n_source_files': len(SRC7), 'n_source_cells': len(srcell), 'n_cells': len(CELLS7)},
          open(OUT / 'gen_stats.json', 'w'), indent=1)
json.dump(unsourced, open(OUT / 'unsourced.json', 'w'), indent=1, ensure_ascii=False)
json.dump(rows, open(OUT / 'rows.json', 'w'), indent=1)
print(json.dumps({'stats': stats, 'n_source_files': len(SRC7), 'n_source_cells': len(srcell), 'n_cells': len(CELLS7)}, indent=1))
