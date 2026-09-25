"""Acceptance: difflib insert-only proof, unique ### N headings, unique CM rows and sections. usage: verify_pass7.py <base dir> <new dir> <out json>"""
import difflib, re, sys, json, collections, pathlib
base, new, outp = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
res = {}
for f in ['negative_results.md', 'claims_matrix.md']:
    a = (base / f).read_text().split('\n'); b = (new / f).read_text().split('\n')
    op = difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes(); ops = collections.Counter(t for t, *_ in op)
    res[f] = {'base_lines': len(a), 'new_lines': len(b), 'opcodes': dict(ops),
              'inserted_lines': sum(j2 - j1 for t, i1, i2, j1, j2 in op if t == 'insert'),
              'replaced': ops.get('replace', 0), 'deleted': ops.get('delete', 0),
              'base_is_subsequence': len(a) == sum(i2 - i1 for t, i1, i2, _, _ in op if t == 'equal')}
reg = (new / 'negative_results.md').read_text()
heads = re.findall(r'^### (N\d+)\b', reg, re.M); c = collections.Counter(heads)
res['N_headings'] = len(heads); res['N_duplicates'] = {k: v for k, v in c.items() if v > 1}
res['new_N_each_once'] = {f'N{n}': c[f'N{n}'] for n in range(398, 406)}
res['max_N'] = max(int(h[1:]) for h in heads)
cm = (new / 'claims_matrix.md').read_text()
rows = re.findall(r'^### \*\*(CM-[A-Z0-9-]+)\*\*', cm, re.M); cc = collections.Counter(rows)
res['CM_rows'] = len(rows); res['CM_duplicates'] = {k: v for k, v in cc.items() if v > 1}
res['new_CM_each_once'] = {k: cc[k] for k in ['CM-W22-2', 'CM-W22-3', 'CM-W22-4', 'CM-W22-5', 'CM-W17-10']}
secs = re.findall(r'^## (§I8\.\d+|◈\.\d+) ', cm + '\n' + reg, re.M); sc = collections.Counter(secs)
res['section_duplicates'] = {k: v for k, v in sc.items() if v > 1}
res['PASS'] = all(res[f]['replaced'] == 0 and res[f]['deleted'] == 0 and res[f]['base_is_subsequence'] for f in ['negative_results.md', 'claims_matrix.md']) \
    and not res['N_duplicates'] and not res['CM_duplicates'] and not res['section_duplicates'] and set(res['new_N_each_once'].values()) == {1} \
    and set(res['new_CM_each_once'].values()) == {1}
print(json.dumps(res, indent=1)); json.dump(res, open(outp, 'w'), indent=1)
