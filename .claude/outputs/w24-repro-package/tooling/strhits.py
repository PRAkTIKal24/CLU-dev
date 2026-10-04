"""List dirty string constants in a tree: prose (>=4 words) vs short, with counts."""
import ast, sys, re, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import scan
scan.GIT_EXTRA.extend(scan.git_identities('/Users/batmac/clu/worktrees/CHLU-w24-repro-package'))
R = scan.compile_rules()
short = collections.Counter(); prose = 0; docs = 0
where = collections.defaultdict(set)
for p in Path(sys.argv[1]).rglob('*.py'):
    t = ast.parse(p.read_text())
    for n in ast.walk(t):
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            s = n.value
            if any(rx.search(s) for _c, _l, rx in R):
                if len(s.split()) >= 4:
                    prose += 1
                else:
                    short[s] += 1; where[s].add(p.name)
print('prose', prose)
for s, c in short.most_common():
    print(c, repr(s)[:140], sorted(where[s])[:3])
