#!/usr/bin/env python3
"""Figure captions of a draft, keyed by label (code only: comments stripped, lines joined).

usage: captions.py DRAFT.tex OUT.json [OUT.tex]
OUT.tex holds one caption per line (for pw1_draft_sweeps/scan.py, the banned-word scan).
Covers \\figslot{stem}{caption}{label} and figure environments with \\includegraphics / \\caption / \\label.
"""
import json, re, sys

sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__file__), '..', 'pw1_draft_sweeps'))
from texnum import split_comment

src = open(sys.argv[1]).read().split('\n')
code = '\n'.join(split_comment(l)[0] for l in src)
code = re.sub(r'\\newcommand\{\\figslot\}.*?\\end\{figure\}\}', '', code, flags=re.S)


def arg(s, i):
    """balanced {...} starting at s[i] == '{'; returns (content, end index)."""
    assert s[i] == '{', s[i:i + 30]
    d = 0
    for j in range(i, len(s)):
        if s[j] == '{' and s[j - 1] != '\\':
            d += 1
        elif s[j] == '}' and s[j - 1] != '\\':
            d -= 1
            if d == 0:
                return s[i + 1:j], j + 1
    raise ValueError


out = {}
for m in re.finditer(r'\\figslot\{', code):
    stem, k = arg(code, m.end() - 1)
    cap, k = arg(code, k)
    lab, k = arg(code, k)
    out[lab] = dict(stem=stem, caption=' '.join(cap.split()), kind='figslot')
for m in re.finditer(r'\\begin\{figure\}(.*?)\\end\{figure\}', code, re.S):
    body = m.group(1)
    g = re.search(r'\\includegraphics(\[[^]]*\])?\{([^}]*)\}', body)
    c = body.find('\\caption{')
    lab = re.search(r'\\label\{([^}]*)\}', body)
    if g and c >= 0 and lab:
        cap, _ = arg(body, c + len('\\caption'))
        out[lab.group(1)] = dict(stem=re.sub(r'^figures/|\.pdf$', '', g.group(2)), caption=' '.join(cap.split()), kind='figure')
json.dump(out, open(sys.argv[2], 'w'), indent=1)
if len(sys.argv) > 3:
    open(sys.argv[3], 'w').write('\n'.join(v['caption'] for v in out.values()) + '\n')
print(len(out), 'figure captions:', ', '.join(out))
