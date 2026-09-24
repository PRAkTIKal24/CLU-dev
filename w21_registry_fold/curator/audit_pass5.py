"""Number audit: every decimal and every 4+-digit token in INSERTED lines must appear verbatim in a cited source file."""
import difflib, re, sys, json, pathlib, glob
base, new = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
E = '/Users/batmac/clu/evidence/outputs/'
src_globs = [
 'w20-shapes-data-matched/*.md', 'w20-shapes-data-matched/*.json', 'w20-shapes-data-matched/deletion_store/seed*/*.json',
 'w21-words-full-test/*.md', 'w21-words-full-test/test/BARS-WORDS-FULL-test.json', 'w21-words-full-test/repro/REPRO_COMPARE.txt',
 'w21-maze-t-clu-improve/*.md', 'w21-maze-t-clu-improve/test/LABEL-BOTH-test.json', 'w21-maze-t-clu-improve/val_full/LABEL-*.json',
 'w21-maze-t-seed4/*.md', 'w21-maze-t-seed4/MAZE-T-SEED4.json', 'w21-maze-t-seed4/test/BOTH_test_seed3.json',
 'w18-advisor-report.md', 'w19-advisor-report.md', 'w20-advisor-report.md',
 'w13-seeds-and-tables/TABLES.md', 'w12b-nr6-and-kstar/KSTAR-LADDER.md',
 'w19-maze-t-symmetric/FINDINGS.md', 'w19-maze-t-symmetric/val/BARS-*.json', 'w19-maze-t-symmetric/test/BARS-BOTH-test.json']
files = [f for g in src_globs for f in glob.glob(E + g)]
files += ['/Users/batmac/clu/outputs/w21-registry-fold/curator/curator_arithmetic.json', '/Users/batmac/clu/tasks/w21/registry-fold.md',
          '/Users/batmac/clu/outputs/w21-registry-fold/START_UTC.txt', '/Users/batmac/clu/outputs/w18-registry-fold/FINDINGS.md']
corpus = {f: open(f, errors='replace').read() for f in files}
a = (base / 'negative_results.md').read_text().split('\n') + ['@@SPLIT@@'] + (base / 'claims_matrix.md').read_text().split('\n')
res = {}
for fname in ['negative_results.md', 'claims_matrix.md']:
    A = (base / fname).read_text().split('\n'); B = (new / fname).read_text().split('\n')
    ins = []
    for t, i1, i2, j1, j2 in difflib.SequenceMatcher(None, A, B, autojunk=False).get_opcodes():
        if t == 'insert': ins += B[j1:j2]
    text = '\n'.join(ins)
    # drop identifiers that are not measurements: section ids, entry ids, row ids, line refs, versions, tuples like (2,0,1,2)
    t2 = re.sub(r'\bA\d+\.\d+|§I8\.\d+|◈\.\d+|\bv2\.\d+|\bN\d+|CM-[A-Z0-9-]+|\bl\.\d+(?:[–-]\d+)?(?:, l\.\d+(?:[–-]\d+)?)*|\bL\d+', ' ', text)
    toks = re.findall(r'\d{1,3}(?:,\d{3})+|\d+\.\d+|\d{4,}', t2)
    missing = sorted({x for x in toks if not any(x in c for c in corpus.values())})
    res[fname] = {'inserted_lines': len(ins), 'tokens': len(toks), 'distinct': len(set(toks)), 'missing': missing}
print(json.dumps(res, indent=1, ensure_ascii=False))
json.dump(res, open(new / 'curator' / 'audit_pass5.json', 'w'), indent=1, ensure_ascii=False)

# ---- strict pass: per registry entry, tokens must be in a file the entry itself names ----
KEYS = {'w20-shapes-data-matched': 'w20-shapes-data-matched/', 'SHAPES-MATCHED': 'w20-shapes-data-matched/', 'w21-words-full-test': 'w21-words-full-test/',
        'w21-maze-t-clu-improve': 'w21-maze-t-clu-improve/', 'w21-maze-t-seed4': 'w21-maze-t-seed4/', 'w20-advisor-report': 'w20-advisor-report',
        'w19-advisor-report': 'w19-advisor-report', 'w18-advisor-report': 'w18-advisor-report', 'w13-seeds-and-tables': 'w13-seeds-and-tables/',
        'KSTAR-LADDER': 'w12b-nr6-and-kstar/', 'w19-maze-t-symmetric': 'w19-maze-t-symmetric/', 'CA': 'curator_arithmetic.json',
        'brief': 'registry-fold.md', 'pass 4': 'w18-registry-fold/FINDINGS.md'}
B = (new / 'negative_results.md').read_text()
blk = B[B.index('## ◈.15 '):]
parts = re.split(r'(?m)^(?=### N\d+ )', blk)
strict = {}
for p_ in parts:
    m = re.match(r'### (N\d+)', p_)
    name = m.group(1) if m else p_.split('\n')[0][:20]
    keys = [v for k, v in KEYS.items() if k in p_]
    sub = [c for f, c in corpus.items() if any(k in f for k in keys)]
    t2 = re.sub(r'\bA\d+\.\d+|§I8\.\d+|◈\.\d+|\bv2\.\d+|\bN\d+|CM-[A-Z0-9-]+|\bl\.\d+(?:[–-]\d+)?(?:, l\.\d+(?:[–-]\d+)?)*|\bL\d+', ' ', p_)
    toks = set(re.findall(r'\d{1,3}(?:,\d{3})+|\d+\.\d+|\d{4,}', t2))
    miss = sorted(x for x in toks if not any(x in c for c in sub))
    strict[name] = {'n_sources': len(sub), 'tokens': len(toks), 'not_in_named_sources': miss}
print(json.dumps({k: v for k, v in strict.items() if v['not_in_named_sources']}, indent=1, ensure_ascii=False))
res['strict_per_entry'] = strict
json.dump(res, open(new / 'curator' / 'audit_pass5.json', 'w'), indent=1, ensure_ascii=False)
