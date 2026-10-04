"""stage1 (shaken, original names) -> package source tree (neutral names). Tooling only.

    transform.py STAGE1 PKG_DIR

* relocates every module by MODMAP (dotted names; flat script dirs use bare names);
* rewrites imports and every identifier in IDMAP (names, attributes, defs, args, keywords);
* rewrites string constants by STRMAP (ordered substring replacements);
* keeps a docstring only if the scanner finds nothing in it and SOFT finds nothing;
  otherwise the docstring is dropped (a module keeps DOCSTRINGS[new_path] if given);
* drops every comment (ast round trip), then `ruff format`.
"""
import ast
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import scan  # noqa: E402
from names import MODMAP, IDMAP, STRMAP, DOCSTRINGS, FILEMAP_EXTRA, FILEMAP_JSON, PATHMAP, OLD_OUT_PREFIXES, EXACT  # noqa: E402

RULES = scan.compile_rules()
SOFT = re.compile(r'brief|agent|worktree|branch\b|commit|bundle|\bbox\b|lane|PID|verdict|\bbar\b|'
                  r'declared|DECLARED|registry|advisor|Amendment|AMENDMENT|scored|SCORED|ledger|'
                  r'programme|wave|spoke|hub|peer|shipped|charter|referee|pin\b|PIN|smoke|'
                  r'\bM[0-9]\b|\bA[0-9]\.|\bR[0-9]+\b|N[0-9]{3}|MF[0-9]|task file|Head|user|'
                  r'C:|/c/|forgis|windows|Windows|cp1252|MSYS|this machine|on this box|w[0-9]+|'
                  r'[☀-➿⬀-⯿\U0001F300-\U0001FAFF]', re.I)


def clean(text):
    if any(rx.search(text) for _c, _l, rx in RULES):
        return False
    return not SOFT.search(text)


# ------------------------------------------------------------------ module names
def old_modname(rel: Path):
    parts = list(rel.with_suffix('').parts)
    if parts[0] in ('scripts', 'figures', 'clue_checks'):
        return parts[0] + ':' + parts[-1]
    if parts[-1] == '__init__':
        parts = parts[:-1]
    return '.'.join(parts)


def new_location(old):
    """old module id -> (new module id, new relative file path)."""
    new = MODMAP.get(old)
    if new is None:
        if ':' in old:
            raise SystemExit('no MODMAP entry for script ' + old)
        new = dotted_new(old)
        if new.split('.')[0] != 'nclu':
            raise SystemExit('no MODMAP entry for ' + old)
    if ':' in new:
        d, n = new.split(':')
        return new, Path(d) / (n + '.py')
    return new, None


def dotted_new(old_dotted, script_dir=None):
    """Map an import target. Package modules: longest-prefix match in MODMAP; bare
    script modules: '<dir>:<name>'."""
    if script_dir is not None:
        for d in [script_dir] + [x for x in ('scripts', 'figures', 'clue_checks') if x != script_dir]:
            if (d + ':' + old_dotted) in MODMAP:
                return MODMAP[d + ':' + old_dotted].split(':')[1]
    if script_dir is None and '.' not in old_dotted:
        for d in ('scripts', 'figures', 'clue_checks'):
            if (d + ':' + old_dotted) in MODMAP:
                return MODMAP[d + ':' + old_dotted].split(':')[1]
    parts = old_dotted.split('.')
    for i in range(len(parts), 0, -1):
        pre = '.'.join(parts[:i])
        if pre in MODMAP and ':' not in MODMAP[pre]:
            return '.'.join([MODMAP[pre]] + parts[i:])
    if parts[0] == 'chlu':
        return 'nclu' + old_dotted[len('chlu'):]
    return old_dotted


PROSE = [(re.compile(a), b) for a, b in [
    # review-round / review-item / amendment references: drop the preposition that introduced them,
    # then the reference itself (parenthesised, possessive, or leading a clause)
    (r'\b(per|by|in|see|under|at)\s+(?=(PW-?[0-9]|AMENDMENT-?[0-9]|item\s+[0-9]|R[0-9]{1,3}\b))', ''),
    (r'\bretired(?=\s*(PW-?[0-9]|item\s+[0-9]|R[0-9]{1,3}\b))', 'not drawn'),
    (r'\s*\(\s*(Head\s+)?AMENDMENT-?[0-9]+[^()]*\)', ''),
    (r'\bAMENDMENT-?[0-9]+\b', ''),
    (r"\bR[0-9]{1,3}(?:(?:\s+ruling)?\s+[0-9]+|\.[0-9]+)?(?:'s)?(?:\s*[/,]\s*R[0-9]{1,3}(?:\.[0-9]+)?)*(?=\s+(is|was|says|requires)\b)", 'this'),
    (r"\s*\(\s*R[0-9]{1,3}(?:(?:\s+ruling)?\s+[0-9]+|\.[0-9]+)?(?:'s)?(?:\s*[/,]\s*R[0-9]{1,3}(?:\.[0-9]+)?)*\s*\)", ''),
    (r"\bR[0-9]{1,3}(?:\.[0-9]+)?'s\s+", 'the '),
    (r"\s*\bR[0-9]{1,3}(?:(?:\s+ruling)?\s+[0-9]+|\.[0-9]+)?(?:'s)?(?:\s*[/,]\s*R[0-9]{1,3}(?:\.[0-9]+)?)*\s*[:—–]\s*", ' '),
    (r"(?<![\w.])\s*\bR[0-9]{1,3}(?:(?:\s+ruling)?\s+[0-9]+|\.[0-9]+)?(?:'s)?(?:\s*[/,]\s*R[0-9]{1,3}(?:\.[0-9]+)?)*\b(?![.\w])", ''),
    (r'\bthe retired\b', 'the prior'),
    (r'\bPW-?[0-9]+ vocabulary\b', 'figure vocabulary'),
    # whole-word replacements for words that contain a banned token (cutting the token out of
    # the word would leave a stub such as 'n' for 'spoken')
    (r'\bspoken\b', 'expressed'), (r'(?i)\bscored runs\b', 'runs'), (r'(?i)\bscored run\b', 'run'),
    (r'\bPRE-REGISTERED\b', 'DECLARED IN ADVANCE'), (r'\b[Pp]re-registered\b', 'declared in advance'),
    (r'\badvisory\b', 'non-binding'),
    (r"\bThe Head's (ruling|decision)", 'The protocol'), (r"\bthe Head's (ruling|decision)", 'the protocol'),
    (r'\bPW-?[0-9]+(\s+item\s+[0-9]+(\.[0-9]+)*)?\.?', ''), (r'(?i)(?<![a-z])pw[0-9]+_?', ''),
    (r'\bitem\s+[0-9]+(\.[0-9]+)*\b', ''),
    (r'\b(declared|fixed|dated|written)?\s*20[0-9]{2}-[01][0-9]-[0-3][0-9](T[0-9:.]+Z?)?', ''),
    (r'(?<![A-Za-z])-?[Rr]uled\b', ''), (r'\bretired_by\b', ''), (r'\bretired\b', 'not drawn'),
    (r'\bRETIRED\b', 'NOT USED'),
    (r'\b[Ss]poke[- ]?A\b', 'the earlier baseline set'),
    (r'\b(spoke|hub)/[A-Za-z0-9_-]+', 'an earlier study'),
    (r'\b[Ss]pokes?\b', 'study'), (r'(?<![A-Za-z])[Hh]ubs?(?![A-Za-z])', 'study'),
    (r'\bHead\b ?', ''), (r'\bAdvisor\b ?', ''),
    (r'\b(AMENDMENT|Amendment|amendment)[- ]?[0-9]*(\.[0-9]+)?', ''),
    (r'\bPREREG[^ ,;)]*', 'protocol'), (r'[Pp]re-?regist[a-z]*', 'declared'),
    (r'\b[Cc]harter\b', 'protocol'), (r'\b[Ss]cored runs?\b', 'test run'),
    (r'\b[Rr]eferees?\b', 'reviewer'), (r'\b[Rr]egistry\b', 'record'),
    (r'\bN[0-9]{3}\b', ''), (r'\bMF[0-9]+\b', ''), (r'\b[Ww]aves?\b', 'round'),
    (r'\bbase/iter[0-9]*', ''), (r'\biter[0-9]\b', ''), (r'\bpw[0-9]\b', ''),
    (r'[A-Za-z]:[\\/][^\s\'"]*', '<path>'), (r'/Users/[^\s\'"]*', '<path>'),
    (r'/c/[^\s\'"]*', '<path>'), (r'~/clu[^\s\'"]*', '<path>'),
    (r'(?<![A-Za-z0-9])[wW][0-9]{1,2}[a-z]?(?![0-9])[-_ /]?', ''),
    (r'\b[Cc]laude\b', ''),
]]
NEED_RUN_DIR = set()
UNRESOLVED = []


def map_path(rest):
    for a, b in PATHMAP:
        if rest == a or rest.startswith(a + '/') or rest.startswith(a + '\\'):
            return b + rest[len(a):]
    return rest


SCARS = [(re.compile(a), b) for a, b in [
    (r'\(\s*[,;:.\-]+\s*', '('), (r'\(\s+(?=[ivx]{1,4}\))', '('), (r'(?<=\w)\s+\)', ')'), (r'\s*[,;:]+\s*\)', ')'), (r'\(\s*\)', ''), (r',\s*,', ','),
    (r'\.\s*:\s*\+?\s*([a-z])', lambda m: '. ' + m.group(1).upper()), (r'\.\s*:\s*\+?\s*', '. '), (r'\s*\bstill running on -?[0-9]+ at build time\b', ''),
    (r'\b(the|a|an|of|to) \1\b', r'\1'),
    (r'(?<=[A-Z]{3})-\)', ')'),
    (r'PROTOCOL-(?=§)', 'PROTOCOL '), (r'\b(spent|run|fitted|landed|used|declared|fixed) (in|at|by|from|per|under)(?=[;,)])', r'\1'),
    (r'(?<=\S) {2,}', ' '), (r' +$', ''), (r'\s+([,;:.])(?=\s|$)', r'\1'), (r'^\s*[,;:]\s*', ''),
]]


def descar(s):
    for _ in range(3):
        for rx, b in SCARS:
            s = rx.sub(b, s)
    return s


def sanitize(s, fragment=False):
    if s in EXACT:
        return EXACT[s]
    if re.fullmatch(r'20[0-9]{2}-[01][0-9]-[0-3][0-9](T[0-9:.]+Z?)?', s):
        return 'not recorded'

    for a, b in PATHMAP:
        s = s.replace(a, b)
    for a, b in STRMAP:
        s = s.replace(a, b)
    if ' ' in s or '\n' in s:
        s = re.sub(r'(?<=[A-Za-z0-9%]) ([,;])(?=\s)', r'\1', s)
        s = re.sub(r'\b(the|a|an|of|to) \1\b', r'\1', s)
        if not clean_hard(s):
            orig = s
            for rx, b in PROSE:
                s = rx.sub(b, s)
            for _c, _l, rx in RULES:
                if _c != 'scrub scars':
                    s = rx.sub('', s)
            # an earlier-study name removed before a possessive leaves an orphan "'s"
            s = re.sub(r"\b([Tt]he) 's\s+", r'\1 ', s)
            s = re.sub(r"(?<=[\s(=|])'s\b", "the earlier study's", s)
            if s != orig:
                # descar never touches the string's edges: an f-string fragment's leading ': ' or
                # trailing ' ' sits next to a {field}, and a plain string's edge whitespace is kept
                if fragment:
                    pre = re.match(r'^[\s,;:.)\]]*', s).group()
                    rest = s[len(pre):]
                    post = re.search(r'[\s,;:.(\[]*$', rest).group()
                    core = rest[:len(rest) - len(post)]
                    s = pre + descar(core) + post
                else:
                    lead = re.match(r'^\s*', orig).group()
                    trail = re.search(r'\s*$', orig).group()
                    s = lead + descar(s).strip() + trail
    return s


def clean_hard(text):
    return not any(rx.search(text) for _c, _l, rx in RULES)


def ren(name):
    return IDMAP.get(name, name)


class T(ast.NodeTransformer):
    def __init__(self, script_dir, is_pkg_mod, modname, fname=''):
        self.script_dir = script_dir
        self.modname = modname
        self.fname = fname
        self.need_run_dir = False

    def _doc(self, node):
        b = node.body
        if b and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant) and isinstance(b[0].value.value, str):
            if not clean(b[0].value.value):
                node.body = b[1:] or [ast.Pass()]
            return True
        return False

    def visit_Module(self, node):
        b = node.body
        had = b and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant) and isinstance(b[0].value.value, str)
        self._doc(node)
        still = node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str)
        if had and not still:
            if Path(self.fname).name == '__init__.py':
                text = 'The %s package.' % '.'.join(Path(self.fname).parent.parts)
            else:
                text = Path(self.fname).stem.replace('_', ' ') + '.'
            node.body.insert(0, ast.Expr(ast.Constant(text)))
        self.generic_visit(node)
        return node

    def visit_FunctionDef(self, node):
        self._doc(node)
        node.name = ren(node.name)
        self.generic_visit(node)
        return node

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        self._doc(node)
        node.name = ren(node.name)
        self.generic_visit(node)
        return node

    def visit_Name(self, node):
        node.id = ren(node.id)
        return node

    def visit_Attribute(self, node):
        self.generic_visit(node)
        node.attr = ren(node.attr)
        return node

    def visit_arg(self, node):
        node.arg = ren(node.arg)
        self.generic_visit(node)
        return node

    def visit_keyword(self, node):
        if node.arg:
            node.arg = ren(node.arg)
        self.generic_visit(node)
        return node

    def visit_Global(self, node):
        node.names = [ren(n) for n in node.names]
        return node

    visit_Nonlocal = visit_Global

    def visit_JoinedStr(self, node):
        for v in node.values:
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                v.value = sanitize(v.value, fragment=True)
            else:
                self.visit(v)
        return node

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            s = node.value
            for pre in OLD_OUT_PREFIXES:
                if s.startswith(pre) or s + '/' == pre:
                    rest = sanitize(map_path(s[len(pre):].replace('\\', '/')))
                    NEED_RUN_DIR.add(self.modname or self.script_dir)
                    self.need_run_dir = True
                    call = ast.Call(ast.Name('run_dir', ast.Load()), [ast.Constant(rest)] if rest else [], [])
                    return ast.Call(ast.Name('str', ast.Load()), [call], [])
            node.value = sanitize(s)
            if not clean_hard(node.value):
                UNRESOLVED.append((self.fname, node.value[:120]))
        return node

    def visit_Import(self, node):
        out = []
        for a in node.names:
            nd = dotted_new(a.name, self.script_dir)
            if a.asname:
                out.append(ast.alias(nd, ren(a.asname)))
            elif '.' in a.name:
                out.append(ast.alias(nd, None))
            else:
                bound = ren(a.name)
                out.append(ast.alias(nd, None if bound == nd else bound))
        node.names = out
        return node

    def visit_ImportFrom(self, node):
        if node.level:
            base = self.modname.split('.')
            base = base[:len(base) - node.level]
            absmod = '.'.join(base + ([node.module] if node.module else []))
        else:
            absmod = node.module or ''
        if absmod == '__future__':
            return node
        newmod = dotted_new(absmod, self.script_dir)
        names, extra = [], []
        for a in node.names:
            sub_old = absmod + '.' + a.name
            sub_new = dotted_new(sub_old)
            bound = ren(a.asname or a.name)
            if sub_old in OLD_MODULES:
                parent, nn = sub_new.rsplit('.', 1)
                if parent != newmod:
                    extra.append(ast.ImportFrom(parent, [ast.alias(nn, None if bound == nn else bound)], 0))
                    continue
            else:
                nn = ren(a.name)
            names.append(ast.alias(nn, None if bound == nn else bound))
        node.module, node.level = newmod, 0
        node.names = names
        out = ([node] if names else []) + extra
        return out if len(out) != 1 else out[0]


def transform_file(src: Path, rel: Path, dst_root: Path):
    old = old_modname(rel)
    new, relnew = new_location(old)
    if relnew is None:
        parts = new.split('.')
        relnew = Path(*parts) / '__init__.py' if rel.name == '__init__.py' else Path(*parts).with_suffix('.py')
    script_dir = old.split(':')[0] if ':' in old else None
    tree = ast.parse(src.read_text(encoding='utf-8'))
    tr = T(script_dir, rel.name == '__init__.py', old if ':' not in old else '', str(relnew))
    tr.visit(tree)
    if tr.need_run_dir:
        imp = ast.ImportFrom('nclu.paths', [ast.alias('run_dir', None)], 0)
        body = tree.body
        i = 0
        while i < len(body) and ((isinstance(body[i], ast.Expr) and isinstance(getattr(body[i], 'value', None), ast.Constant))
                                 or (isinstance(body[i], ast.ImportFrom) and body[i].module == '__future__')):
            i += 1
        body.insert(i, imp)
    ast.fix_missing_locations(tree)
    code = ast.unparse(tree) + '\n'
    doc = DOCSTRINGS.get(str(relnew))
    if doc:
        b = tree.body
        if b and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant) and isinstance(b[0].value.value, str):
            tree.body = b[1:]
            code = ast.unparse(tree) + '\n'
        code = '"""' + doc.strip('\n') + '\n"""\n' + code
    out = dst_root / relnew
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(code, encoding='utf-8')
    return str(relnew)


OLD_MODULES = set()


def main(stage1, dst):
    stage1, dst = Path(stage1), Path(dst)
    for p in stage1.rglob('*.py'):
        om = old_modname(p.relative_to(stage1))
        if ':' not in om:
            OLD_MODULES.add(om)
    made = {}
    for p in sorted(stage1.rglob('*.py')):
        rel = p.relative_to(stage1)
        made[str(rel)] = transform_file(p, rel, dst)
    def jmap(o):
        if isinstance(o, dict):
            return {sanitize(k): jmap(v) for k, v in o.items()}
        if isinstance(o, list):
            return [jmap(v) for v in o]
        if isinstance(o, str):
            v = sanitize(o)
            if not clean_hard(v):
                UNRESOLVED.append(('json', v[:120]))
            return v
        return o
    def drop_history(o):
        if isinstance(o, dict):
            return {k: drop_history(v) for k, v in o.items()
                    if k not in ('retired', 'retired_by', 'retired_rows', '_doc')}
        if isinstance(o, list):
            return [drop_history(v) for v in o
                    if not (isinstance(v, dict) and v.get('status') == 'retired')]
        return o
    for a, b in FILEMAP_JSON.items():
        s, d = stage1 / a, dst / b
        if s.exists():
            d.parent.mkdir(parents=True, exist_ok=True)
            d.write_text(json.dumps(jmap(drop_history(json.loads(s.read_text(encoding='utf-8')))), indent=1) + '\n', encoding='utf-8')
            made[a] = b
    for a, b in FILEMAP_EXTRA.items():
        s, d = stage1 / a, dst / b
        if s.exists():
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(s, d)
            made[a] = b
    subprocess.run(['/Users/batmac/clu/workspace/.venv/bin/ruff', 'format', '--quiet', '--line-length', '100', str(dst)], check=True)
    return made


if __name__ == '__main__':
    made = main(sys.argv[1], sys.argv[2])
    print(json.dumps(made, indent=0))
    Path(sys.argv[2]).parent.joinpath('transform_unresolved.txt').write_text(
        '\n'.join('%s\t%r' % u for u in UNRESOLVED), encoding='utf-8')
    print('UNRESOLVED strings:', len(UNRESOLVED), file=sys.stderr)
