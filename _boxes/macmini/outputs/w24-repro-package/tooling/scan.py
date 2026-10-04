"""Anonymity scanner (tooling only -- it contains the banned list, so it never ships).

    scan.py TARGET_DIR [--json OUT.json] [--git REPO]

Scans every file under TARGET_DIR:
  * file and directory names (every path component);
  * text contents (any file that decodes as UTF-8/Latin-1 text, incl. .ipynb/.json/.md);
  * binary contents: printable runs (like `strings -n 4`), plus
      - PNG: tEXt / zTXt (zlib) / iTXt (optionally compressed) chunks, decoded;
      - PDF: every FlateDecode stream decompressed, the Info dictionary and XMP;
      - ZIP / NPZ: member names, archive comment, and every member decompressed and scanned
        recursively (an .npy member's header and any pickled object array included);
      - gzip: decompressed.
Matching is case-insensitive except the role word `Head` (see RULES).  Exit status 1 if
anything is found.
"""
import gzip
import io
import json
import re
import subprocess
import sys
import zipfile
import zlib
from pathlib import Path

# --------------------------------------------------------------------------- the list
# category -> list of (label, regex). Regexes are compiled case-insensitive unless the
# label ends with '[case-sensitive]'.
SEED = {
    'scrub scars': [
        r'\(\s*[,;:]', r',\s*,', r'\.\s*:\s', r'\bon -[0-9]+\b', r"(?<![\w\])}'\"`\\])'s\b", r'\b(spent|run|fitted|landed|used|declared|fixed) (in|at|by|from|per|under)[;,)]', r'PROTOCOL-§', r'[A-Z]{3}-\)[case-sensitive]', r'^\s*init\s*\.', r'\binit  ', r'\(\s+[ivx]{1,4}\)', r'\b(the|a|an|of|to) \1\b', r'\bthe (that|which|who|of|and)\b', r'\(\s+\)', r'[A-Za-z0-9] [,;](?=\s)', r'\(\s*-\s*\)', r'--\s*\)',
    ],
    'people/org': [
        'Pratik', 'Jawahar', 'Pierini', 'forgis', 'forgisprova', '@forgis\\.com', 'x10719pj', 'ORCID',
        'PRAkTIKal24', '55033105', 'batmac', 'Copilot', '198982749',
        r'\bgrant(s|ed)?\b', r'\bfund(ed|ing|er)\b', r'acknowledg', r'\bsponsor', r'\bEPSRC\b', r'\bUKRI\b',
        r'\bNSF\b', r'\bERC\b', r'\bDFG\b', r'\bARIA\b', r'\bSNSF\b', 'Royal Society', 'Leverhulme',
    ],
    'machines/paths': [
        'FORGIS_14', 'forgis-14', 'batmac', 'thinkpad', 'sshkey', 'tailscale', r'\bmsys', '/Users/', '/home/',
        r'C:\\', r'\b[A-Za-z]:/', '/c/chlu', 'farside', 'clu-dev', 'tailfb0a84', r'\.ts\.net', '/private/',
        '/var/folders', r'~/clu', 'Desktop/CHLU', r'\bWindows-1', 'PowerShell', r'\bhost\"?\s*:\s*\"',
    ],
    'tooling': [
        'Claude', 'Anthropic', 'Co-Authored-By', r'CLAUDE\.md', r'\.claude', 'noreply@', r'\bOpus\b', r'\bFable\b',
        r'\bChatGPT\b', r'\bGPT-?[0-9]', r'\bCodex\b', 'copilot-swe', r'\bLLM\b', 'experiment-engineer',
    ],
    'programme': [
        'spoke', r'(?<![a-z])hub(?![a-z])', r'\bHead\b[case-sensitive]', 'Advisor', 'PREREG', 'pre-?regist',
        'ledger', 'charter', 'wave', r'\bw[0-9]{1,2}[_\-/]', r'(?<![a-z0-9])w[0-9]{1,2}[a-z]?(?=[_\-/.:]|\b)',
        r'base/iter', r'iter[0-9]', 'CHLU', 'chlu', 'referee', 'placeholder', 'TODO', 'scored run',
        r'amendment', r'\bruling\b', r'registry', r'\bN[0-9]{3}\b[case-sensitive]', r'\bMF[0-9]{1,2}\b', r'\bc[0-9]w[0-9]',
        r'\bpw[0-9]\b', 'bprime', r'SCORED-', r'\bcampaign\b', r'Montero pool',
        r'PW-?[0-9]', r'(?<![a-z])pw[0-9]', r'\bitem [0-9]+(\.[0-9]+)*\b', r'\bruled\b', r'(?<![A-Za-z])-ruled',
        r'20[0-9]{2}-[01][0-9]-[0-3][0-9]', 'retired_by', r'\bR[0-9]{1,3}(\.[0-9]+)?\b[case-sensitive]', r'AMENDMENT-?[0-9]', r'\bnot drawn [0-9]', r'\bretired\b',
    ],
}
# names and e-mails of every author / committer in the source history (filled at run time)
GIT_EXTRA = []

WHITELIST_RULES = [
    "lowercase 'head' (the model's request-indexed head, e.g. 'closed-form head', 'refuse-token "
    "head') is legitimate: the role word is matched case-sensitively as \\bHead\\b, so 'head', "
    "'heads', 'HeadSpec' never match and the capitalised role word always does.",
]


def compile_rules():
    rules = []
    for cat, pats in SEED.items():
        for p in pats:
            cs = p.endswith('[case-sensitive]')
            core = p[:-len('[case-sensitive]')] if cs else p
            rules.append((cat, p, re.compile(core, 0 if cs else re.I)))
    for s in GIT_EXTRA:
        rules.append(('people/org (git log --all)', s, re.compile(re.escape(s), re.I)))
    return rules


def git_identities(repo):
    out = subprocess.run(['git', '-C', repo, 'log', '--all', '--format=%an%n%ae%n%cn%n%ce%n%B'],
                         capture_output=True, text=True, errors='replace').stdout
    ids = set()
    lines = out.splitlines()
    # author/committer fields: every 4 lines before the body are not separable after %B, so
    # re-run field-only for exactness
    out2 = subprocess.run(['git', '-C', repo, 'log', '--all', '--format=%an%x00%ae%x00%cn%x00%ce'],
                          capture_output=True, text=True, errors='replace').stdout
    for ln in out2.splitlines():
        for f in ln.split('\x00'):
            f = f.strip()
            if not f:
                continue
            ids.add(f)
            if '@' in f:
                local, dom = f.split('@', 1)
                ids.add(local)
                if dom not in ('users.noreply.github.com', 'github.com'):
                    ids.add(dom)
    for ln in lines:
        m = re.match(r'\s*co-authored-by:\s*(.+?)\s*<([^>]+)>', ln, re.I)
        if m:
            ids.add(m.group(1))
            ids.add(m.group(2))
    # a bare common word is not an identity; they are covered by the seed list where relevant
    # an e-mail's local part or domain that is an ordinary word ('info', 'local') is dropped as a
    # separate token; the full address that contains it stays on the list.
    GENERIC = ('github', 'claude', 'info', 'local', 'noreply')
    ids = {i for i in ids if len(i) >= 4 and i.lower() not in GENERIC}
    return sorted(ids)


# --------------------------------------------------------------------------- extraction
PRINTABLE = re.compile(rb'[\x20-\x7e]{4,}')


def strings_of(b):
    return '\n'.join(m.group().decode('ascii') for m in PRINTABLE.finditer(b))


def png_text(b):
    out = []
    if not b.startswith(b'\x89PNG'):
        return out
    i = 8
    while i + 8 <= len(b):
        n = int.from_bytes(b[i:i + 4], 'big')
        typ = b[i + 4:i + 8]
        data = b[i + 8:i + 8 + n]
        i += 12 + n
        try:
            if typ == b'tEXt':
                out.append(data.replace(b'\x00', b': ').decode('latin-1'))
            elif typ == b'zTXt':
                k, rest = data.split(b'\x00', 1)
                out.append(k.decode('latin-1') + ': ' + zlib.decompress(rest[1:]).decode('latin-1'))
            elif typ == b'iTXt':
                k, rest = data.split(b'\x00', 1)
                comp, _meth, rest = rest[0], rest[1], rest[2:]
                _lang, rest = rest.split(b'\x00', 1)
                _tk, txt = rest.split(b'\x00', 1)
                out.append(k.decode('latin-1') + ': ' + (zlib.decompress(txt) if comp else txt).decode('utf-8', 'replace'))
            elif typ in (b'eXIf',):
                out.append(strings_of(data))
        except Exception as e:  # a malformed chunk is itself reported
            out.append('UNPARSED-%s-chunk %r' % (typ, e))
    return out


def pdf_text(b):
    out = []
    for m in re.finditer(rb'stream\r?\n(.*?)\r?\nendstream', b, re.S):
        raw = m.group(1)
        for f in (zlib.decompress, lambda x: zlib.decompressobj().decompress(x)):
            try:
                out.append(strings_of(f(raw)))
                break
            except Exception:
                continue
    for m in re.finditer(rb'<<(.{0,4000}?)>>', b, re.S):  # dictionaries incl. /Info
        s = m.group(1)
        if b'/Author' in s or b'/Creator' in s or b'/Producer' in s or b'/Title' in s:
            out.append(strings_of(s))
            for h in re.finditer(rb'<([0-9A-Fa-f]{8,})>', s):  # hex strings (UTF-16)
                try:
                    out.append(bytes.fromhex(h.group(1).decode()).decode('utf-16', 'replace'))
                except Exception:
                    pass
    return out


def npy_blobs(b):
    """Parse a concatenation of .npy blobs (an .npy file, or an equinox leaf file).

    -> (texts, unaccounted_bytes). Headers (dicts of text) and object/str arrays are returned
    as text to scan; numeric payloads are checked to be exactly the declared arrays and are
    not text. Any byte not accounted for by a declared array is returned for full scanning.
    """
    import ast as _ast
    import numpy as _np
    texts, i = [], 0
    while i < len(b) and b[i:i + 6] == b'\x93NUMPY':
        major = b[i + 6]
        if major == 1:
            hl = int.from_bytes(b[i + 8:i + 10], 'little'); h0 = i + 10
        else:
            hl = int.from_bytes(b[i + 8:i + 12], 'little'); h0 = i + 12
        header = b[h0:h0 + hl].decode('latin-1')
        texts.append(header)
        d = _ast.literal_eval(header.strip())
        dt = _np.dtype(d['descr'])
        n = int(_np.prod(d['shape'])) if d['shape'] else 1
        start = h0 + hl
        if dt.hasobject:  # pickled payload: scan it all
            texts.append(strings_of(b[start:]))
            return texts, b''
        end = start + n * dt.itemsize
        if dt.kind in 'SUa':
            texts.append(b[start:end].decode('latin-1', 'replace'))
        i = end
    return texts, b[i:]


def zip_text(b, depth):
    out = []
    try:
        z = zipfile.ZipFile(io.BytesIO(b))
    except Exception:
        return out
    out.append(z.comment.decode('latin-1'))
    for info in z.infolist():
        out.append('MEMBER ' + info.filename)
        try:
            data = z.read(info)
        except Exception as e:
            out.append('UNREADABLE-MEMBER %s %r' % (info.filename, e))
            continue
        out.extend(extract(info.filename, data, depth + 1))
    return out


def extract(name, b, depth=0):
    """-> list of strings to scan for one file's bytes."""
    texts = []
    try:
        texts.append(b.decode('utf-8'))
        return texts
    except UnicodeDecodeError:
        pass
    if b[:6] == b'\x93NUMPY':
        t, rest = npy_blobs(b)
        texts.extend(t)
        if rest:
            texts.append('UNACCOUNTED-BYTES %d after the declared arrays' % len(rest))
            texts.append(strings_of(rest))
        return texts
    texts.append(strings_of(b))
    if depth > 3:
        return texts
    texts.extend(png_text(b))
    if b.startswith(b'%PDF'):
        texts.extend(pdf_text(b))
    if b[:2] == b'PK':
        texts.extend(zip_text(b, depth))
    if b[:2] == b'\x1f\x8b':
        try:
            texts.extend(extract(name, gzip.decompress(b), depth + 1))
        except Exception:
            pass
    return texts


# --------------------------------------------------------------------------- scan
def scan(target):
    target = Path(target)
    rules = compile_rules()
    hits = []
    nfiles = 0
    for p in sorted(target.rglob('*')):
        rel = p.relative_to(target)
        for comp in rel.parts:
            for cat, lab, rx in rules:
                for m in rx.finditer(comp):
                    hits.append({'file': str(rel), 'where': 'name', 'category': cat, 'rule': lab,
                                 'match': m.group(), 'context': comp})
        if not p.is_file():
            continue
        nfiles += 1
        b = p.read_bytes()
        for t in extract(p.name, b):
            for ln_no, line in enumerate(t.splitlines(), 1):
                for cat, lab, rx in rules:
                    for m in rx.finditer(line):
                        hits.append({'file': str(rel), 'where': 'content', 'line': ln_no,
                                     'category': cat, 'rule': lab, 'match': m.group(),
                                     'context': line.strip()[:160]})
    # de-duplicate (a name component is seen once per descendant)
    seen, uniq = set(), []
    for h in hits:
        k = (h['file'] if h['where'] != 'name' else h['context'], h['where'], h.get('line'), h['rule'], h['match'])
        if k in seen:
            continue
        seen.add(k)
        uniq.append(h)
    return {'target': str(target), 'files_scanned': nfiles, 'hits': uniq}


def rules_listing():
    out = []
    for cat, pats in SEED.items():
        out.append((cat, list(pats)))
    out.append(('people/org (git log --all)', list(GIT_EXTRA)))
    return out


if __name__ == '__main__':
    args = sys.argv[1:]
    tgt = args[0]
    js = args[args.index('--json') + 1] if '--json' in args else None
    repo = args[args.index('--git') + 1] if '--git' in args else None
    if repo:
        GIT_EXTRA.extend(git_identities(repo))
    res = scan(tgt)
    res['rules'] = rules_listing()
    res['whitelist'] = WHITELIST_RULES
    if js:
        Path(js).write_text(json.dumps(res, indent=1), encoding='utf-8')
    print('files scanned: %d   hits: %d' % (res['files_scanned'], len(res['hits'])))
    for h in res['hits'][:400]:
        print('%-8s %-24s %s:%s  [%s] %r  | %s' % (h['where'], h['category'], h['file'], h.get('line', ''),
                                                    h['rule'], h['match'], h['context'][:100]))
    sys.exit(1 if res['hits'] else 0)
