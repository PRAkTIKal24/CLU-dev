"""Tokens (decimals) in inserted lines that occur in the sources ONLY as the prefix of a longer number: check they equal the ROUNDED longer number."""
import json, re, sys, pathlib, importlib.util
spec = importlib.util.spec_from_file_location('a', str(pathlib.Path(__file__).parent / 'audit_pass6.py'))
sys.argv = ['x', 'base_pass5', 'build_out', '/dev/null']
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    a = importlib.util.module_from_spec(spec); spec.loader.exec_module(a)
out = []
for fname, text in a.ins_by.items():
    for x in sorted(set(t for t in a.toks(text) if '.' in t and ',' not in t)):
        exact = any(re.search(r'(?<![0-9])' + re.escape(x) + r'(?![0-9])', c) for c in a.corpus.values())
        if exact: continue
        longer = set()
        for c in a.corpus.values():
            longer |= set(re.findall(r'(?<![0-9])' + re.escape(x) + r'\d+', c))
        nd = len(x.split('.')[1])
        ok = [l for l in longer if f'{float(l):.{nd}f}' == x]
        out.append({'file': fname, 'token': x, 'longer_in_sources': sorted(longer)[:5], 'is_correct_rounding_of_some': bool(ok)})
bad = [o for o in out if not o['is_correct_rounding_of_some']]
print(json.dumps({'prefix_only_tokens': len(out), 'not_a_correct_rounding': bad}, indent=1))
json.dump({'prefix_only_tokens': out, 'not_a_correct_rounding': bad}, open('rounding_check_p6.json', 'w'), indent=1)
