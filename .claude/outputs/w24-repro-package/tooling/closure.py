"""Static import closure: given a source root (with chlu/ and scripts/) and entry files,
return every chlu module and scripts/ module transitively imported (all imports, including
function-local ones). Usage: closure.py ROOT entry1.py [entry2.py ...] [--extra-root R2]"""
import ast, sys, os
from pathlib import Path

def mod_to_path(root, mod):
    parts = mod.split('.')
    p = Path(root, *parts)
    if (p / '__init__.py').exists():
        return p / '__init__.py'
    if p.with_suffix('.py').exists():
        return p.with_suffix('.py')
    return None

def imports_of(path, pkgname):
    src = Path(path).read_text(encoding='utf-8', errors='replace')
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        print('SYNTAX', path, e, file=sys.stderr); return set()
    out = set()
    # package of this file for relative imports
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names: out.add(a.name)
        elif isinstance(n, ast.ImportFrom):
            if n.level and n.level > 0:
                base = pkgname.split('.')
                base = base[:len(base) - n.level + (1 if Path(path).name == '__init__.py' else 0)] if n.level else base
                m = '.'.join(base + ([n.module] if n.module else []))
            else:
                m = n.module or ''
            out.add(m)
            for a in n.names:
                out.add(m + '.' + a.name)  # may be a submodule
    return out

def modname_of(root, path):
    rel = Path(path).resolve().relative_to(Path(root).resolve())
    parts = list(rel.with_suffix('').parts)
    if parts[-1] == '__init__': parts = parts[:-1]
    if parts and parts[0] in ('scripts', 'w21_figures'): parts = parts[1:]
    return '.'.join(parts)

def closure(root, entries):
    root = Path(root)
    seen = {}
    todo = [Path(e).resolve() for e in entries]
    while todo:
        p = todo.pop()
        if str(p) in seen: continue
        seen[str(p)] = True
        mn = modname_of(root, p)
        if os.environ.get('SKIP_INIT') and p.name == '__init__.py': continue
        for m in imports_of(p, mn):
            top = m.split('.')[0]
            cand = None
            if top == 'chlu' or (root / top / '__init__.py').exists() or (root / (top + '.py')).exists():
                # add package __init__s along the way
                parts = m.split('.')
                for i in range(1, len(parts) + 1):
                    q = mod_to_path(root, '.'.join(parts[:i]))
                    if q and str(q.resolve()) not in seen: todo.append(q.resolve())
            else:
                q = root / 'scripts' / (top + '.py')
                if q.exists() and str(q.resolve()) not in seen: todo.append(q.resolve())
    return sorted(seen)

if __name__ == '__main__':
    root = sys.argv[1]
    files = closure(root, sys.argv[2:])
    tot = 0
    for f in files:
        n = sum(1 for _ in open(f, encoding='utf-8', errors='replace'))
        tot += n
        print(n, os.path.relpath(f, root))
    print('TOTAL', len(files), 'files', tot, 'lines')
