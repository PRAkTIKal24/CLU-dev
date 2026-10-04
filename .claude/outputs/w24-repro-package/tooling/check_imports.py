"""Static check on a package tree: every `from <internal> import name` binds something that
exists, and every `<alias>.attr` on an internal module alias exists. Tooling only.

    check_imports.py PKG_ROOT
"""
import ast
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])
FLAT = ['scripts', 'figures']


def modfile(dotted):
    parts = dotted.split('.')
    p = ROOT.joinpath(*parts)
    if (p / '__init__.py').exists():
        return p / '__init__.py'
    if p.with_suffix('.py').exists():
        return p.with_suffix('.py')
    if len(parts) == 1:
        for d in FLAT:
            q = ROOT / d / (dotted + '.py')
            if q.exists():
                return q
    return None


_cache = {}


def bound(f):
    if f in _cache:
        return _cache[f]
    t = ast.parse(Path(f).read_text())
    names = set()
    for st in ast.walk(t):
        if isinstance(st, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)) and st in t.body:
            names.add(st.name)
    for st in t.body:
        for n in ast.walk(st) if isinstance(st, (ast.If, ast.Try, ast.With, ast.For)) else [st]:
            if isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                names.add(n.name)
            elif isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                for tg in (n.targets if isinstance(n, ast.Assign) else [n.target]):
                    for x in ast.walk(tg):
                        if isinstance(x, ast.Name):
                            names.add(x.id)
            elif isinstance(n, ast.Import):
                for a in n.names:
                    names.add(a.asname or a.name.split('.')[0])
            elif isinstance(n, ast.ImportFrom):
                for a in n.names:
                    names.add(a.asname or a.name)
    _cache[f] = names
    return names


def internal(dotted):
    return dotted.split('.')[0] == 'nclu' or modfile(dotted) is not None


errors = []
for f in sorted(ROOT.rglob('*.py')):
    t = ast.parse(f.read_text())
    alias = {}
    for n in ast.walk(t):
        if isinstance(n, ast.ImportFrom) and n.module and n.level == 0 and internal(n.module):
            mf = modfile(n.module)
            if mf is None:
                errors.append('%s:%d no module %s' % (f.relative_to(ROOT), n.lineno, n.module))
                continue
            b = bound(mf)
            for a in n.names:
                if a.name == '*':
                    continue
                if a.name not in b and modfile(n.module + '.' + a.name) is None:
                    errors.append('%s:%d %s has no %s' % (f.relative_to(ROOT), n.lineno, n.module, a.name))
                if modfile(n.module + '.' + a.name) is not None:
                    alias[a.asname or a.name] = n.module + '.' + a.name
        elif isinstance(n, ast.Import):
            for a in n.names:
                if internal(a.name):
                    if modfile(a.name) is None:
                        errors.append('%s:%d no module %s' % (f.relative_to(ROOT), n.lineno, a.name))
                    elif a.asname:
                        alias[a.asname] = a.name
                    elif '.' not in a.name:
                        alias[a.name] = a.name
    for n in ast.walk(t):
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id in alias and not n.attr.startswith('__'):
            mod = alias[n.value.id]
            mf = modfile(mod)
            if mf and n.attr not in bound(mf) and modfile(mod + '.' + n.attr) is None:
                errors.append('%s:%d %s.%s missing (module %s)' % (f.relative_to(ROOT), n.lineno, n.value.id, n.attr, mod))
for e in sorted(set(errors)):
    print(e)
print('IMPORT-CHECK errors:', len(set(errors)))
sys.exit(1 if errors else 0)
