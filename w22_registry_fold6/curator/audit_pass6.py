"""Number audit: every decimal and every 4+-digit token in INSERTED lines must appear verbatim in a cited source file.
Strict pass: each registry entry / matrix row is checked only against the files it names. usage: audit_pass6.py <base> <new> <out json>"""
import difflib, re, sys, json, pathlib, glob
base, new, outp = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
E = '/Users/batmac/clu/evidence/outputs/'; C = str(pathlib.Path(__file__).parent) + '/'
G = {
 'rel': ['w21-rel-newseeds/*.md', 'w21-rel-newseeds/*.json', 'w21-rel-newseeds/*_UTC.txt'],
 'lad': ['w21-ladder-rivals-32k/*.md', 'w21-ladder-rivals-32k/*.json', 'w21-ladder-rivals-32k/checks/*.json', 'w21-ladder-rivals-32k/*_UTC.txt'],
 'fig': ['w22-figures3-FINDINGS.md', 'FIGURE-DEFECT-cmazet-legality.md'],
 'w20': ['w20-shapes-data-matched/SHAPES-MATCHED.json', 'w20-shapes-data-matched/FINDINGS.md'],
}
files = {k: [f for g in v for f in glob.glob(E + g)] for k, v in G.items()}
files['fig'] += [f for f in glob.glob(C + 'src_w22_figures3/**', recursive=True) if pathlib.Path(f).is_file()]
files['ca6'] = [C + 'curator_arithmetic_p6.json']
files['brief'] = ['/Users/batmac/clu/tasks/w22/registry6.md']
files['self'] = [C + '../START_UTC.txt', '/Users/batmac/clu/outputs/w21-registry-fold/FINDINGS.md', '/Users/batmac/clu/outputs/w21-registry-fold/LOG.md',
                 '/Users/batmac/clu/outputs/w21-registry-fold/START_UTC.txt', '/Users/batmac/clu/outputs/w21-registry-fold/END_UTC.txt',
                 '/Users/batmac/clu/evidence/outputs/w21-maze-t-seed4/MAZE-T-SEED4.json', '/Users/batmac/clu/evidence/outputs/w21-maze-t-seed4/FINDINGS.md']
corpus = {f: open(f, errors='replace').read() for v in files.values() for f in v}
def found(x, c):  # verbatim, and not glued to a preceding digit (e.g. '2.22' inside '12.22' does not count)
    return re.search(r'(?<![0-9])' + re.escape(x), c) is not None
IDS = r'\bA\d+\.\d+|§I8\.\d+|◈\.\d+|\bv2\.\d+|\bN\d+|CM-[A-Z0-9-]+|\bl\.\d+(?:[–-]\d+)?(?:, l\.\d+(?:[–-]\d+)?)*|\bL\d+|\bw\d+(?:[a-z]\b)?|\bpass \d+|\bs\d\b'
TOK = r'\d{1,3}(?:,\d{3})+|\d+\.\d+|\d{4,}'
def toks(t): return re.findall(TOK, re.sub(IDS, ' ', t))
res = {'n_source_files': len(corpus)}
ins_by = {}; hunks_by = {}
for fname in ['negative_results.md', 'claims_matrix.md']:
    A = (base / fname).read_text().split('\n'); B = (new / fname).read_text().split('\n'); ins = []
    for t, i1, i2, j1, j2 in difflib.SequenceMatcher(None, A, B, autojunk=False).get_opcodes():
        if t == 'insert': ins += B[j1:j2]
    ins_by[fname] = '\n'.join(ins); tk = toks(ins_by[fname])
    hunks_by[fname] = [ '\n'.join(B[j1:j2]) for t, i1, i2, j1, j2 in difflib.SequenceMatcher(None, A, B, autojunk=False).get_opcodes() if t == 'insert']
    res[fname] = {'inserted_lines': len(ins), 'tokens': len(tk), 'distinct': len(set(tk)),
                  'missing_from_all_sources': sorted({x for x in tk if not any(found(x, c) for c in corpus.values())})}
# strict: split the inserted text into entries / rows; each checked against the files it names
KEYS = {'w21-rel-newseeds': 'rel', 'rel-newseeds': 'rel', 'REL-NEWSEEDS': 'rel', 'ladder-rivals-32k': 'lad', 'LADDER-RIVALS-32K': 'lad',
        'figures3': 'fig', 'FIGURE-DEFECT': 'fig', 'AMBIGUITY': 'fig', 'SHAPES-MATCHED': 'w20', 'w20-shapes-data-matched': 'w20', 'CA6': 'ca6',
        'brief': 'brief', 'registry6': 'brief', 'PASS 6': 'self', 'pass 5': 'self', 'maze-t-seed4': 'self',
        # a cross-reference inherits the sources of the entry / row it cites
        'N391': 'rel', 'N392': 'rel', 'N393': 'lad', 'N394': 'lad', 'N395': 'fig', 'CM-W21-4': 'rel', 'CM-W21-5': 'lad', 'CM-W21-6': 'lad', 'CM-W22-1': 'fig'}
strict = {}
for fname, pat in [('negative_results.md', r'(?m)^(?=### N\d+ |## ◈\.\d+ |\*\*Reading key)'), ('claims_matrix.md', r'(?m)^(?=### \*\*CM-|## §I8\.\d+ )')]:
    for part in [q for h in hunks_by[fname] for q in re.split(pat, h)]:
        if not part.strip(): continue
        m = re.match(r'### (N\d+)|### \*\*(CM-[A-Z0-9-]+)|## (◈\.\d+|§I8\.\d+)|(\*\*Reading key \([a-zA-Z-]+\))', part)
        name = fname[:3] + ':' + (next(g for g in m.groups() if g) if m else part.strip().split('\n')[0][:70])
        while name in strict: name += '+'
        keys = {v for k, v in KEYS.items() if k in part} | {'ca6', 'self'}
        sub = [corpus[f] for k in keys for f in files[k]]
        tk = set(toks(part)); miss = sorted(x for x in tk if not any(found(x, c) for c in sub))
        strict[name] = {'keys': sorted(keys), 'tokens': len(tk), 'not_in_named_sources': miss}
res['strict_per_entry'] = strict
res['strict_failures'] = {k: v['not_in_named_sources'] for k, v in strict.items() if v['not_in_named_sources']}
res['PASS'] = not res['strict_failures'] and all(not res[f]['missing_from_all_sources'] for f in ['negative_results.md', 'claims_matrix.md'])
print(json.dumps({k: v for k, v in res.items() if k != 'strict_per_entry'}, indent=1, ensure_ascii=False))
json.dump(res, open(outp, 'w'), indent=1, ensure_ascii=False)
