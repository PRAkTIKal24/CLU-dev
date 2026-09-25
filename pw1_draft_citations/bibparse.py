"""Minimal brace-balanced BibTeX reader (read-only; never writes a .bib)."""
import re, unicodedata


def entries(path):
    s = open(path, encoding='utf-8').read()
    out = {}
    for m in re.finditer(r'@(\w+)\s*\{\s*([^,\s]+)\s*,', s):
        typ, key = m.group(1).lower(), m.group(2)
        if typ in ('comment', 'string', 'preamble'):
            continue
        i, depth = m.end(), 1
        while depth:
            c = s[i]
            depth += {'{': 1, '}': -1}.get(c, 0)
            i += 1
        body = s[m.end():i - 1]
        out[key] = dict(_type=typ, **fields(body))
    return out


def fields(body):
    f, i = {}, 0
    while True:
        m = re.compile(r'\s*,?\s*([A-Za-z_-]+)\s*=\s*').match(body, i)
        if not m:
            break
        name, j = m.group(1).lower(), m.end()
        if body[j] == '{':
            depth, k = 1, j + 1
            while depth:
                depth += {'{': 1, '}': -1}.get(body[k], 0)
                k += 1
            val, i = body[j + 1:k - 1], k
        elif body[j] == '"':
            k = body.index('"', j + 1)
            val, i = body[j + 1:k], k + 1
        else:
            k = re.compile(r'[^,}\s]*').match(body, j).end()
            val, i = body[j:k], k
        f[name] = val
    return f


def norm_title(t):
    t = unicodedata.normalize('NFKD', t)
    t = re.sub(r'\\[a-zA-Z]+|[{}]', '', t)
    return re.sub(r'[^a-z0-9]', '', t.lower())


def doi(e):
    d = e.get('doi', '').strip().lower()
    return re.sub(r'^https?://(dx\.)?doi\.org/', '', d)


def arxiv(e):
    blob = ' '.join(e.get(k, '') for k in ('eprint', 'url', 'doi', 'note', 'journal', 'archiveprefix', 'howpublished'))
    m = re.search(r'(?:arxiv[.:/ ]*(?:abs/)?|10\.48550/arxiv\.)(\d{4}\.\d{4,5})', blob, re.I) or \
        (re.search(r'(\d{4}\.\d{4,5})', e.get('eprint', '')) if e.get('eprint') else None)
    return m.group(1) if m else ''
