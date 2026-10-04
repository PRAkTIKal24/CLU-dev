"""Patch layer applied to a copy of stage0 before shaking (tooling only).

Two primitives, both refusing silently-failing edits:
  rep(file, old, new, count=1)   exact string replacement, asserts the occurrence count
  redef(file, name, src)         replace the top-level def/class/assignment `name` (found by
                                 AST, decorators included) with `src`
Patch groups live in patches_*.py modules next to this file; each defines apply(root).
"""
import ast
import importlib
import sys
from pathlib import Path

ROOT = None


def _p(f):
    return Path(ROOT) / f


def rep(f, old, new, count=1):
    p = _p(f)
    s = p.read_text(encoding='utf-8')
    n = s.count(old)
    if n != count:
        raise SystemExit('PATCH FAILED %s: expected %d occurrence(s), found %d of:\n%s' % (f, count, n, old[:300]))
    p.write_text(s.replace(old, new), encoding='utf-8')


def redef(f, name, src):
    p = _p(f)
    s = p.read_text(encoding='utf-8')
    lines = s.splitlines(keepends=True)
    tree = ast.parse(s)
    hit = []
    for st in tree.body:
        nm = None
        if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            nm = st.name
        elif isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
            nm = st.targets[0].id
        elif isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name):
            nm = st.target.id
        if nm == name:
            start = min([st.lineno] + [d.lineno for d in getattr(st, 'decorator_list', [])])
            hit.append((start, st.end_lineno))
    if len(hit) != 1:
        raise SystemExit('REDEF FAILED %s: %s found %d times' % (f, name, len(hit)))
    a, b = hit[0]
    if not src.endswith('\n'):
        src += '\n'
    lines[a - 1:b] = [src]
    p.write_text(''.join(lines), encoding='utf-8')


def delete_def(f, name):
    redef(f, name, '')


def write(f, src):
    p = _p(f)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(src, encoding='utf-8')


if __name__ == '__main__':
    ROOT = sys.argv[1]
    sys.path.insert(0, str(Path(__file__).parent))
    groups = sys.argv[2:] or sorted(q.stem for q in Path(__file__).parent.glob('patches_*.py'))
    for g in groups:
        mod = importlib.import_module(g)
        mod.apply(sys.modules[__name__])
        print('applied', g)
