"""Negative control: for every FOUND token, change its last digit (+1 mod 10) and search the same scope with
the same matcher. A mutant that is still 'found' is a token the check could not have caught if it were wrong."""
import json, re, sys, importlib.util
sys.argv = ['x', 'src/DRAFT-v2.tex', 'DRAFT-v2.prev-42pp.tex', '/Users/user/Desktop/CHLU/.claude/outputs/w17-paper', '/tmp/w25-nc-dummy.json']
spec = importlib.util.spec_from_file_location('vr', 'verbatim_recheck.py'); vr = importlib.util.module_from_spec(spec); spec.loader.exec_module(vr)
d = json.load(open('recheck.json')); found = [r for r in d['rows'] if r['status'] == 'FOUND']
def mutate(t):
    i = max(k for k, ch in enumerate(t) if ch.isdigit()); return t[:i] + str((int(t[i]) + 1) % 10) + t[i+1:]
caught = [r for r in found if vr.find(mutate(r['token']), r['scope']) is None]
esc = [r for r in found if r not in caught]
def cls(t): c = re.sub(r'[^0-9]', '', t); return 'short(<4 digits)' if len(c) < 4 else 'long(>=4 digits)'
from collections import Counter
print(json.dumps({'found': len(found), 'mutants_caught': len(caught), 'mutants_escaped': len(esc),
  'escaped_by_class': Counter(cls(r['token']) for r in esc), 'found_by_class': Counter(cls(r['token']) for r in found)}, indent=1))
json.dump([{'line': r['line'], 'token': r['token'], 'mutant': mutate(r['token'])} for r in esc], open('negative_control_escapes.json', 'w'), indent=1)
