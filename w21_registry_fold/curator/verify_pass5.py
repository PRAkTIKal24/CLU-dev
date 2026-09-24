"""Acceptance: difflib insert-only proof, unique ### N headings, unique CM rows."""
import difflib, re, sys, json, collections, pathlib
base, new = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
res = {}
for f in ['negative_results.md', 'claims_matrix.md']:
    a = (base / f).read_text().split('\n'); b = (new / f).read_text().split('\n')
    ops = collections.Counter(); ins = 0
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        ops[tag] += 1
        if tag == 'insert': ins += j2 - j1
    res[f] = {'base_lines': len(a), 'new_lines': len(b), 'opcodes': dict(ops), 'inserted_lines': ins,
              'replaced': ops.get('replace', 0), 'deleted': ops.get('delete', 0),
              'base_is_subsequence': len(a) == sum(i2 - i1 for t, i1, i2, _, _ in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes() if t == 'equal')}
reg = (new / 'negative_results.md').read_text()
heads = re.findall(r'^### (N\d+)\b', reg, re.M)
c = collections.Counter(heads)
res['N_headings'] = len(heads); res['N_duplicates'] = {k: v for k, v in c.items() if v > 1}
res['new_N_each_once'] = all(c[f'N{n}'] == 1 for n in range(372, 391))
cm = (new / 'claims_matrix.md').read_text()
rows = re.findall(r'^### \*\*(CM-[A-Z0-9-]+)\*\*', cm, re.M)
cc = collections.Counter(rows); res['CM_rows'] = len(rows); res['CM_duplicates'] = {k: v for k, v in cc.items() if v > 1}
print(json.dumps(res, indent=1))
json.dump(res, open(new / 'curator' / 'verify_pass5.json', 'w'), indent=1)
