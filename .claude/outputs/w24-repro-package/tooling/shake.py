"""Top-level tree-shaker for the package build (tooling only; never shipped).

shake.py SRC_ROOT DST_ROOT ENTRIES_FILE [--keep-all MOD ...]

SRC_ROOT holds packages (dirs with __init__.py) and flat script dirs (SCRIPT_DIRS) whose
modules import each other by bare name. ENTRIES_FILE lists entry files (relative to SRC_ROOT),
one per line; each is kept whole. Every other module keeps only the top-level statements
reachable from kept code: defs/classes/assignments by name, imports as bindings (partial
`from X import a, b` rewritten to the needed names), other top-level statements kept as side
effects except a trailing `if __name__ == "__main__":` block. Output mirrors SRC_ROOT layout.
"""
import ast, os, sys, json
from pathlib import Path

SCRIPT_DIRS = ['scripts', 'figures', 'clue_checks']


class Mod:
    def __init__(self, name, path, is_pkg):
        self.name, self.path, self.is_pkg = name, path, is_pkg
        self.src = path.read_text(encoding='utf-8')
        self.lines = self.src.splitlines(keepends=True)
        self.tree = ast.parse(self.src)
        self.body = self.tree.body
        self.bind = {}      # name -> [stmt index]
        self.imports = {}   # alias -> ('mod', dotted) | ('from', dotted_mod, name, level)
        for i, st in enumerate(self.body):
            for nm in self._bound(st):
                self.bind.setdefault(nm, []).append(i)

    def _bound(self, st):
        out = []
        if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.append(st.name)
        elif isinstance(st, ast.Import):
            for a in st.names:
                al = a.asname or a.name.split('.')[0]
                out.append(al)
                self.imports[al] = ('mod', a.name if a.asname else a.name.split('.')[0])
        elif isinstance(st, ast.ImportFrom):
            for a in st.names:
                al = a.asname or a.name
                out.append(al)
                self.imports[al] = ('from', self._abs(st), a.name)
        elif isinstance(st, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            tg = st.targets if isinstance(st, ast.Assign) else [st.target]
            for t in tg:
                for n in ast.walk(t):
                    if isinstance(n, ast.Name):
                        out.append(n.id)
        return out

    def _abs(self, st):
        if not st.level:
            return st.module or ''
        base = self.name.split('.')
        if not self.is_pkg:
            base = base[:-1]
        base = base[:len(base) - (st.level - 1)] if st.level > 1 else base
        return '.'.join(base + ([st.module] if st.module else []))


class Shaker:
    def __init__(self, root, entries, keep_all=()):
        self.root = Path(root)
        self.mods = {}
        self.entries = [str(Path(e)) for e in entries]
        self.keep_all = set(keep_all)
        self.kept = {}          # modname -> set(stmt idx)
        self.need = set()       # (modname, name)
        self.work = []

    # ---------------- module resolution
    def resolve(self, dotted):
        if dotted in self.mods:
            return self.mods[dotted]
        parts = dotted.split('.')
        p = self.root.joinpath(*parts)
        if (p / '__init__.py').exists():
            m = Mod(dotted, p / '__init__.py', True)
        elif p.with_suffix('.py').exists():
            m = Mod(dotted, p.with_suffix('.py'), False)
        else:
            for sd in SCRIPT_DIRS:
                q = self.root / sd / (dotted + '.py')
                if len(parts) == 1 and q.exists():
                    m = Mod(dotted, q, False)
                    break
            else:
                return None
        self.mods[dotted] = m
        self.kept.setdefault(dotted, set())
        # importing a.b.c executes a and a.b
        if '.' in dotted:
            self.resolve(dotted.rsplit('.', 1)[0])
        return m

    def modname_for_file(self, rel):
        parts = list(Path(rel).with_suffix('').parts)
        if parts[0] in SCRIPT_DIRS:
            return parts[-1]
        if parts[-1] == '__init__':
            parts = parts[:-1]
        return '.'.join(parts)

    # ---------------- reachability
    def require(self, modname, name):
        key = (modname, name)
        if key in self.need:
            return
        self.need.add(key)
        m = self.resolve(modname)
        if m is None:
            return
        idxs = m.bind.get(name)
        if idxs:
            for i in idxs:
                self.keep_stmt(m, i)
            if name in m.imports:
                self.follow_alias(m.imports[name])
        elif m.is_pkg:
            # maybe a submodule
            sub = self.resolve(modname + '.' + name)
            if sub is not None:
                self.keep_module_side_effects(sub)

    def keep_module_side_effects(self, m):
        for i, st in enumerate(m.body):
            if self.is_side_effect(m, i, st):
                self.keep_stmt(m, i)

    def is_side_effect(self, m, i, st):
        if isinstance(st, ast.ImportFrom) and st.module == '__future__':
            return True
        if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)):
            return False
        if isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant) and i == 0:
            return True  # docstring
        if isinstance(st, (ast.Assign, ast.AnnAssign)):
            tg = st.targets if isinstance(st, ast.Assign) else [st.target]
            if all(isinstance(t, ast.Name) for t in tg):
                return any(t.id == '__all__' for t in tg) and False
            return True
        if isinstance(st, ast.AugAssign):
            return not isinstance(st.target, ast.Name)
        if isinstance(st, ast.If) and self.is_main_guard(st):
            return m.name in self.entry_mods or m.name in self.keep_all
        return True

    @staticmethod
    def is_main_guard(st):
        t = st.test
        return (isinstance(t, ast.Compare) and isinstance(t.left, ast.Name) and t.left.id == '__name__')

    def keep_stmt(self, m, i):
        if i in self.kept[m.name]:
            return
        self.kept[m.name].add(i)
        self.work.append((m, m.body[i]))

    def process(self, m, st):
        # imports in this statement (top-level or local) -> require modules
        local_alias = dict(m.imports)
        if isinstance(st, (ast.Import, ast.ImportFrom)):
            self._import_stmt(m, st, only=None)
            if m.name in self.entry_mods and isinstance(st, ast.ImportFrom):
                src = m._abs(st)
                for a in st.names:
                    self.follow_alias(('from', src, a.name))
            return
        for n in ast.walk(st):
            if isinstance(n, ast.Import):
                for a in n.names:
                    al = a.asname or a.name.split('.')[0]
                    local_alias[al] = ('mod', a.name if a.asname else a.name.split('.')[0])
                    if self.resolve(a.name) is not None:
                        self.keep_module_side_effects(self.mods[a.name])
            elif isinstance(n, ast.ImportFrom):
                src = m._abs(n)
                for a in n.names:
                    local_alias[a.asname or a.name] = ('from', src, a.name)
                    if self.resolve(src) is not None:
                        self.keep_module_side_effects(self.mods[src])
                        self.require(src, a.name)
        for n in ast.walk(st):
            if isinstance(n, ast.Name):
                self.use_name(m, n.id, local_alias)
            elif isinstance(n, ast.Attribute):
                dotted = self.dotted(n)
                if dotted:
                    self.use_dotted(m, dotted, local_alias)
            elif isinstance(n, ast.Constant) and isinstance(n.value, str):
                v = n.value
                if v.isidentifier() and v in m.bind and not v.startswith('__'):
                    for j in m.bind[v]:
                        if isinstance(m.body[j], (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                            self.keep_stmt(m, j)

    def _import_stmt(self, m, st, only):
        if isinstance(st, ast.Import):
            for a in st.names:
                if self.resolve(a.name) is not None:
                    self.keep_module_side_effects(self.mods[a.name])
        else:
            src = m._abs(st)
            if self.resolve(src) is not None:
                self.keep_module_side_effects(self.mods[src])

    @staticmethod
    def dotted(n):
        parts = []
        while isinstance(n, ast.Attribute):
            parts.append(n.attr)
            n = n.value
        if isinstance(n, ast.Name):
            parts.append(n.id)
            return '.'.join(reversed(parts))
        return None

    def use_name(self, m, name, alias):
        if name in m.bind:
            self.require(m.name, name)
        if name in alias:
            self.follow_alias(alias[name])

    def follow_alias(self, a):
        if a[0] == 'from':
            src, nm = a[1], a[2]
            if self.resolve(src) is not None:
                self.require(src, nm)
        else:
            if self.resolve(a[1]) is not None:
                self.keep_module_side_effects(self.mods[a[1]])

    def use_dotted(self, m, dotted, alias):
        parts = dotted.split('.')
        head = parts[0]
        if head in m.bind:
            self.require(m.name, head)
        a = alias.get(head)
        if a is not None and a[0] == 'from':
            self.follow_alias(a)
        if a is None:
            return
        # base module the alias denotes
        if a[0] == 'mod':
            base = a[1]
        else:
            src = a[1]
            cand = src + '.' + a[2] if src else a[2]
            if self.resolve(cand) is not None:
                base = cand
            else:
                if self.resolve(src) is not None:
                    self.require(src, a[2])
                return
        # walk down attribute chain through submodules
        cur = base
        rest = parts[1:]
        if self.resolve(cur) is None:
            return
        self.keep_module_side_effects(self.mods[cur])
        while rest:
            nxt = cur + '.' + rest[0]
            if self.resolve(nxt) is not None and not self.mods[cur].bind.get(rest[0]):
                cur = nxt
                self.keep_module_side_effects(self.mods[cur])
                rest = rest[1:]
                continue
            self.require(cur, rest[0])
            break

    def run(self):
        self.entry_mods = set()
        for e in self.entries:
            mn = self.modname_for_file(e)
            m = self.resolve(mn)
            assert m is not None, e
            self.entry_mods.add(mn)
        for mn in list(self.entry_mods) + list(self.keep_all):
            m = self.resolve(mn)
            for i in range(len(m.body)):
                self.keep_stmt(m, i)
        while True:
            while self.work:
                m, st = self.work.pop()
                self.process(m, st)
            for mm in list(self.mods.values()):
                self.keep_module_side_effects(mm)
            if not self.work:
                break

    # ---------------- emit
    def emit(self, dst):
        dst = Path(dst)
        report = {}
        for mn, m in self.mods.items():
            kept = sorted(self.kept.get(mn, ()))
            if not kept and not m.is_pkg:
                continue
            out = []
            for i in kept:
                st = m.body[i]
                if isinstance(st, ast.ImportFrom) and mn not in self.entry_mods:
                    names = [a for a in st.names if (mn, a.asname or a.name) in self.need
                             or (a.asname or a.name) in self.used_names(m)]
                    if st.module == '__future__':
                        names = st.names
                    if not names:
                        continue
                    if len(names) != len(st.names):
                        mod = '.' * st.level + (st.module or '')
                        out.append('from %s import %s\n' % (mod, ', '.join(
                            a.name + (' as ' + a.asname if a.asname else '') for a in names)))
                        continue
                if isinstance(st, ast.Import) and mn not in self.entry_mods:
                    names = [a for a in st.names if (a.asname or a.name.split('.')[0]) in self.used_names(m)]
                    if not names:
                        continue
                    if len(names) != len(st.names):
                        out.append('import %s\n' % ', '.join(
                            a.name + (' as ' + a.asname if a.asname else '') for a in names))
                        continue
                start = min([st.lineno] + [d.lineno for d in getattr(st, 'decorator_list', [])])
                out.append(''.join(m.lines[start - 1:st.end_lineno]))
                if not out[-1].endswith('\n'):
                    out[-1] += '\n'
                if isinstance(st, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                    out.append('\n\n')
            rel = m.path.relative_to(self.root)
            p = dst / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(''.join(out), encoding='utf-8')
            report[str(rel)] = [len(kept), len(m.body)]
        return report

    def used_names(self, m):
        if not hasattr(self, '_used'):
            self._used = {}
        if m.name in self._used:
            return self._used[m.name]
        s = set()
        for i in self.kept.get(m.name, ()):
            st = m.body[i]
            if isinstance(st, (ast.Import, ast.ImportFrom)):
                continue
            for n in ast.walk(st):
                if isinstance(n, ast.Name):
                    s.add(n.id)
                elif isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value.isidentifier():
                    pass
        # names required from this module by others
        for (mn, nm) in self.need:
            if mn == m.name:
                s.add(nm)
        self._used[m.name] = s
        return s


if __name__ == '__main__':
    src, dst, ef = sys.argv[1:4]
    keep_all = sys.argv[5:] if len(sys.argv) > 4 and sys.argv[4] == '--keep-all' else []
    entries = [l.strip() for l in open(ef) if l.strip() and not l.startswith('#')]
    sh = Shaker(src, entries, keep_all)
    sh.run()
    rep = sh.emit(dst)
    tot_src = tot_dst = 0
    for rel, (k, n) in sorted(rep.items()):
        pass
    print(json.dumps({'modules': len(rep)}, indent=0))
    for rel in sorted(rep):
        print(rel, rep[rel])
