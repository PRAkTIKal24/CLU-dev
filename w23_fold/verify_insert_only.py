"""difflib insert-only proof: base file must be an exact subsequence of the new file. usage: verify_insert_only.py <out json> <base> <new> [<base> <new> ...]"""
import difflib, collections, json, sys
res = {}
a = sys.argv[2:]
for base, new in zip(a[::2], a[1::2]):
    A = open(base, encoding='utf-8').read().split('\n'); B = open(new, encoding='utf-8').read().split('\n')
    op = difflib.SequenceMatcher(None, A, B, autojunk=False).get_opcodes(); c = collections.Counter(t for t, *_ in op)
    res[new] = {'base': base, 'base_lines': len(A), 'new_lines': len(B), 'inserted_lines': sum(j2 - j1 for t, i1, i2, j1, j2 in op if t == 'insert'),
                'replaced': c.get('replace', 0), 'deleted': c.get('delete', 0),
                'base_is_subsequence': len(A) == sum(i2 - i1 for t, i1, i2, _, _ in op if t == 'equal')}
res['PASS'] = all(v['replaced'] == 0 and v['deleted'] == 0 and v['base_is_subsequence'] for v in res.values())
json.dump(res, open(sys.argv[1], 'w'), indent=1); print(json.dumps(res, indent=1))
