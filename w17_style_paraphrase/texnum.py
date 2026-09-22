"""Shared extraction for the style-paraphrase checks. Operates on LaTeX CODE only
(comments stripped at the first unescaped %)."""
import re
from collections import Counter

def split_comment(line):
    m = re.search(r'(?<!\\)%', line)
    return (line, '') if not m else (line[:m.start()], line[m.start():])

# identical to the draft-v2 provenance checker's number regex (not preceded by a letter,
# underscore, backslash or '{'), plus \sci{m}{e} captured whole.
NUM = re.compile(r'(?<![A-Za-z_\\{])(\d[\d,]*\.?\d*)')
SCI = re.compile(r'\\sci\{([^}]*)\}\{([^}]*)\}')

def numbers(lines):
    """Multiset of number tokens in the code (non-comment) part of `lines`.
    Every digit run counts (no small-integer or year filter): the strictest reading."""
    c = Counter()
    for line in lines:
        code, _ = split_comment(line)
        for m in SCI.finditer(code):
            c['SCI(%s,%s)' % (m.group(1).strip(), m.group(2).strip())] += 1
        code = SCI.sub(' ', code)
        for m in NUM.finditer(code):
            tok = m.group(1).rstrip('.,')
            if tok:
                c[tok] += 1
    return c

def comments(lines):
    """Multiset of provenance comments: every comment containing SRC:, SRCBLOCK:, ENDSRC
    or the '(no result number ...)' marker, stripped of surrounding whitespace."""
    c = Counter()
    for line in lines:
        _, com = split_comment(line)
        if re.search(r'SRC(BLOCK)?:|ENDSRC|no result number', com):
            c[com.strip()] += 1
    return c

def src_paths(lines):
    c = Counter()
    for line in lines:
        _, com = split_comment(line)
        m = re.search(r'SRC(?:BLOCK)?:\s*(.*)', com)
        if m:
            for p in m.group(1).split(';'):
                if p.strip():
                    c[p.strip()] += 1
    return c

def macro_args(text, name):
    """All balanced-brace arguments of \\name{...} in the code part of text (comments stripped)."""
    code = '\n'.join(split_comment(l)[0] for l in text.split('\n'))
    out, i, key = [], 0, '\\' + name + '{'
    while True:
        j = code.find(key, i)
        if j < 0:
            return out
        k, depth = j + len(key), 1
        while depth:
            ch = code[k]
            if ch == '{' and code[k-1] != '\\': depth += 1
            elif ch == '}' and code[k-1] != '\\': depth -= 1
            k += 1
        out.append(code[j + len(key):k - 1])
        i = k

def norm_ws(s):
    return re.sub(r'\s+', ' ', s).strip()
