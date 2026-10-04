#!/usr/bin/env python3
"""Prose byte-identity after stripping citations (template: w17 cite-tagger PROSE-CHECK.txt).
Strips ~?\\cite*/\\nocite{...} from BEFORE and AFTER and compares bytes, (a) on the full file and
(b) on prose = full file with LaTeX comments removed."""
import re, sys, difflib, hashlib, pathlib
before, after = map(pathlib.Path, sys.argv[1:3])
CITE = re.compile(r'~?\\(?:nocite|citep|citet|cite)\{[^}]*\}')
COMMENT = re.compile(r'(?<!\\)%.*$', re.M)
def strip(t): return CITE.sub('', t)
def prose(t): return COMMENT.sub('', strip(t))
out = []
for name in ["DRAFT-v2.tex", "DRAFT-v2-style.tex"]:
    b, a = (before / name).read_text(), (after / name).read_text()
    sb, sa = strip(b), strip(a)
    pb, pa = prose(b), prose(a)
    kb = [k for m in re.finditer(r'\\(?:nocite|citep|citet|cite)\{([^}]*)\}', b) for k in m.group(1).split(',')]
    ka = [k for m in re.finditer(r'\\(?:nocite|citep|citet|cite)\{([^}]*)\}', a) for k in m.group(1).split(',')]
    cmds = lambda t, c: len(re.findall(r'\\' + c + r'\{', t))
    out.append(f"== {name}")
    out.append(f"PROSE (comments removed) BYTE-IDENTICAL after stripping cites: {'YES' if pb.encode()==pa.encode() else 'NO'}")
    out.append(f"  sha256 before-stripped-prose {hashlib.sha256(pb.encode()).hexdigest()}")
    out.append(f"  sha256 after-stripped-prose  {hashlib.sha256(pa.encode()).hexdigest()}")
    out.append(f"FULL FILE BYTE-IDENTICAL after stripping cites: {'YES' if sb==sa else 'NO'}")
    diff = [l for l in difflib.unified_diff(sb.splitlines(), sa.splitlines(), lineterm='', n=0) if l[:1] in '+-' and l[:3] not in ('+++','---')]
    out.append(f"  full-file lines differing after strip: {len(diff)} (expected: the 2 refreshed '% cite-tagger:' comment lines, as -/+ pairs)")
    for l in diff: out.append("    " + l[:260])
    assert all('% cite-tagger:' in l for l in diff), "non-comment difference!"
    out.append(f"words (wc -w style) before-stripped {len(pb.split())}  after-stripped {len(pa.split())}")
    out.append(f"cite keys before {len(kb)} (distinct {len(set(kb))})  after {len(ka)} (distinct {len(set(ka))})")
    out.append(f"\\citep before {cmds(b,'citep')} after {cmds(a,'citep')};  \\nocite before {cmds(b,'nocite')} after {cmds(a,'nocite')}")
    out.append(f"keys removed: {sorted(set(kb)-set(ka))}")
    out.append(f"keys added:   {sorted(set(ka)-set(kb))}")
print("\n".join(out))
