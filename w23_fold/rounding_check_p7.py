"""Tokens (decimals) in inserted lines that occur in the sources ONLY as the prefix of a longer number: each must be a correct rounding of it.
usage: rounding_check_p7.py <base> <new> <out json>"""
import json, re, sys, pathlib, importlib.util, io, contextlib
spec = importlib.util.spec_from_file_location('a', str(pathlib.Path(__file__).parent / 'audit_pass7.py'))
out_json = sys.argv[3]; sys.argv = ['x', sys.argv[1], sys.argv[2], '/dev/null']
with contextlib.redirect_stdout(io.StringIO()):
    a = importlib.util.module_from_spec(spec); spec.loader.exec_module(a)
out = []
for fname, hs in a.hunks_by.items():
    for x in sorted(set(t for t in a.toks('\n'.join(hs)) if '.' in t and ',' not in t)):
        if any(re.search(r'(?<![0-9])' + re.escape(x) + r'(?![0-9])', c) for c in a.corpus.values()):
            continue
        longer = set()
        for c in a.corpus.values():
            longer |= set(re.findall(r'(?<![0-9])' + re.escape(x) + r'\d+', c))
        nd = len(x.split('.')[1])
        out.append({'file': fname, 'token': x, 'longer_in_sources': sorted(longer)[:5], 'is_correct_rounding_of_some': any(f'{float(l):.{nd}f}' == x for l in longer)})
bad = [o for o in out if not o['is_correct_rounding_of_some']]
print(json.dumps({'prefix_only_tokens': out, 'not_a_correct_rounding': bad}, indent=1))
json.dump({'prefix_only_tokens': out, 'not_a_correct_rounding': bad}, open(out_json, 'w'), indent=1)
