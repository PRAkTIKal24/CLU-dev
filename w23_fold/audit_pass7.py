"""Number audit: every decimal and every 4+-digit token in INSERTED lines must appear verbatim (not glued to a preceding digit) in a source file.
Strict pass: each registry entry / matrix row / inserted hunk is checked only against the files it names (+ CA7 and this pass's own record).
usage: audit_pass7.py <base dir> <new dir> <out json>"""
import difflib, re, sys, json, pathlib, glob
base, new, outp = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
E = '/Users/batmac/clu/evidence/outputs/'; O = '/Users/batmac/clu/outputs/'; C = O + 'w23-fold/curator/'
G = {
 's5': [E + 'w22-seeds-and-tables5/*.md', E + 'w22-seeds-and-tables5/*.txt', E + 'w22-seeds-and-tables5/*.json', O + 'w18-seeds-and-tables/FINDINGS.md',
        O + 'w18-seeds-and-tables/LOG.md', O + 'w18-seeds-and-tables/TABLES.md'],
 'lp': [E + 'w15-lace-poe-rows/FINDINGS.md', E + 'w15-lace-poe-rows/GENTABLE-KSTAR.json'],
 'w20': [E + 'w20-shapes-data-matched/SHAPES-MATCHED.json', E + 'w20-shapes-data-matched/FINDINGS.md', E + 'w20-shapes-data-matched/SHAPES-MATCHED.md'],
 'dr': [O + 'w22-draft-rulings/FINDINGS.md', O + 'w22-draft-rulings/LOG.md', O + 'w22-draft-rulings/AMENDMENT-1.md'],
 'rd2': [O + 'w22-figures-redraw2/FINDINGS.md'], 'pv': [O + 'w22-paper-visuals/FINDINGS.md'],
 'seed4': [E + 'w21-maze-t-seed4/MAZE-T-SEED4.json', E + 'w21-maze-t-seed4/TABLES.md', E + 'w21-rel-newseeds/FINDINGS.md'],
 'ca5': [O + 'w21-registry-fold/curator/curator_arithmetic.json'],
 'drafts': [O + 'w17-paper/DRAFT-v2.tex', O + 'w17-paper/DRAFT-v2-style.tex'],
 'p6': [E + 'w22-registry-fold6/FINDINGS.md', E + 'w22-registry-fold6/END_UTC.txt', E + 'w22-registry-fold6/START_UTC.txt', E + 'w22-registry-fold6/negative_results.md'],
 'ca7': [C + 'curator_arithmetic_p7.json', C + 'gen_stats.json', C + 'register_coverage_before.json', C + 'register_coverage_after.json', C + 'install-build.txt',
         C + 'verify_register7.json', O + 'w23-fold/SHA256-TABLES-pass7.txt'],
 'self': [O + 'w23-fold/LOG.md', O + 'w23-fold/START_UTC.txt', '/Users/batmac/clu/tasks/w23/fold.md', C + 'NEGATIVE-TESTS-pass7.txt'],
}
files = {k: [f for g in v for f in glob.glob(g)] for k, v in G.items()}
corpus = {f: open(f, errors='replace').read() for v in files.values() for f in v}
def found(x, c): return re.search(r'(?<![0-9])' + re.escape(x), c) is not None
IDS = r'\bA\d+\.\d+|§I8\.\d+|◈\.\d+|\bv2\.\d+|\bN\d+|CM-[A-Z0-9-]+|\bl\.\d+(?:[–-]\d+)?(?:, l\.\d+(?:[–-]\d+)?)*|\bL\d+|\bw\d+(?:[a-z]\b)?|\bpass \d+|\bs\d\b'
TOK = r'\d{1,3}(?:,\d{3})+|\d+\.\d+|\d{4,}'
def toks(t): return re.findall(TOK, re.sub(IDS, ' ', t))
res = {'n_source_files': len(corpus), 'source_files': sorted(corpus)}; hunks_by = {}
for fname in ['negative_results.md', 'claims_matrix.md']:
    A = (base / fname).read_text().split('\n'); B = (new / fname).read_text().split('\n')
    op = difflib.SequenceMatcher(None, A, B, autojunk=False).get_opcodes()
    hunks_by[fname] = ['\n'.join(B[j1:j2]) for t, i1, i2, j1, j2 in op if t == 'insert']
    tk = toks('\n'.join(hunks_by[fname]))
    res[fname] = {'inserted_lines': sum(h.count('\n') + 1 for h in hunks_by[fname]), 'tokens': len(tk), 'distinct': len(set(tk)),
                  'missing_from_all_sources': sorted({x for x in tk if not any(found(x, c) for c in corpus.values())})}
KEYS = {'seeds-and-tables': 's5', 'seeds5': 's5', 'TABLES.md': 's5', 'GENTABLE': 'lp', 'lace-poe': 'lp', 'PoE': 'lp', 'SHAPES-MATCHED': 'w20', 'w20': 'w20',
        'AMENDMENT': 'dr', 'draft-rulings': 'dr', 'redraw2': 'rd2', 'figures-redraw': 'rd2', 'paper-visuals': 'pv', 'MAZE-T-SEED4': 'seed4', '0.6407': 'seed4',
        'CA5': 'ca5', 'N385': 'ca5', 'DRAFT-v2': 'drafts', 'draft': 'drafts', 'pass 6': 'p6', 'PASS 6': 'p6', 'registry6': 'p6',
        'N398': 's5', 'N399': 'lp', 'N400': 's5', 'N401': 's5', 'N402': 's5', 'N403': 'dr', 'CM-W22-2': 's5', 'CM-W22-3': 's5', 'CM-W22-4': 'lp', 'CM-W22-5': 'dr'}
strict = {}
for fname, pat in [('negative_results.md', r'(?m)^(?=### N\d+ |## ◈\.\d+ )'), ('claims_matrix.md', r'(?m)^(?=### \*\*CM-|## §I8\.\d+ )')]:
    for part in [q for h in hunks_by[fname] for q in re.split(pat, h)]:
        if not part.strip(): continue
        m = re.match(r'### (N\d+)|### \*\*(CM-[A-Z0-9-]+)|## (◈\.\d+|§I8\.\d+)', part)
        name = fname[:3] + ':' + (next(g for g in m.groups() if g) if m else part.strip().split('\n')[0][:70])
        while name in strict: name += '+'
        keys = {v for k, v in KEYS.items() if k in part} | {'ca7', 'self'}
        sub = [corpus[f] for k in keys for f in files[k]]
        tk = set(toks(part)); miss = sorted(x for x in tk if not any(found(x, c) for c in sub))
        strict[name] = {'keys': sorted(keys), 'tokens': len(tk), 'not_in_named_sources': miss}
res['strict_per_entry'] = strict
res['strict_failures'] = {k: v['not_in_named_sources'] for k, v in strict.items() if v['not_in_named_sources']}
res['PASS'] = not res['strict_failures'] and all(not res[f]['missing_from_all_sources'] for f in ['negative_results.md', 'claims_matrix.md'])
print(json.dumps({k: v for k, v in res.items() if k not in ('strict_per_entry', 'source_files')}, indent=1, ensure_ascii=False))
json.dump(res, open(outp, 'w'), indent=1, ensure_ascii=False)
