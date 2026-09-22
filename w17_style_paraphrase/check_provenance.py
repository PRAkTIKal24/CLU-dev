#!/usr/bin/env python3
"""Provenance checker for DRAFT-v2.tex.

For every non-comment number token in the .tex, find the sources that govern its line
(a '% SRC:' comment on the same line, or an enclosing '% SRCBLOCK:' ... '% ENDSRC' block,
or the nearest '% SRC:' within the next 3 lines of the same paragraph) and check that the
number string occurs in at least one of those source files (evidence root EV).
Prints: unsourced numbers, and sourced numbers not found in their sources.
Ignored: integers <= 12, 4-digit years 1990-2030, tokens inside \\ref/\\label/\\cite, and
LaTeX layout lengths.
"""
import re, sys, os, unicodedata

EV = os.path.expanduser('~/clu/evidence/outputs/')
TEX = sys.argv[1] if len(sys.argv) > 1 else 'DRAFT-v2.tex'
lines = open(TEX).read().split('\n')

cache = {}
def load(path):
    if path not in cache:
        fp = os.path.join(EV, path.strip())
        try:
            t = open(fp, errors='replace').read()
        except Exception:
            t = None
        if t is not None:
            t = t.replace('\u2212', '-').replace('\u2009', '').replace('\u202f', '')
            t_nocomma = t.replace(',', '')
            t = t + '\n' + t_nocomma
        cache[path] = t
    return cache[path]

NUM = re.compile(r'(?<![A-Za-z_\\{])(\d[\d,]*\.?\d*)')

def strip_noise(code):
    code = re.sub(r'\\(ref|label|cite[pt]?|eqref|pageref)\{[^}]*\}', ' ', code)
    code = re.sub(r'\\(setlength|tabcolsep|begin|end|includegraphics|adjustbox|multicolumn|p)\{[^}]*\}', ' ', code)
    code = re.sub(r'\{p\{[\d.]+cm\}', ' ', code)
    code = re.sub(r'p\{[\d.]+cm\}', ' ', code)
    code = re.sub(r'[\d.]+(pt|cm|mm|em|ex|in)\b', ' ', code)
    code = re.sub(r'\\linewidth|width=[\d.]+', ' ', code)
    code = re.sub(r'\\sci\{([^}]*)\}\{([^}]*)\}', lambda m: ' SCI(%s,%s) ' % (m.group(1), m.group(2)), code)
    return code

def split_comment(line):
    m = re.search(r'(?<!\\)%', line)
    if not m:
        return line, ''
    return line[:m.start()], line[m.start():]

def srcs_of(comment):
    m = re.search(r'SRC(?:BLOCK)?:\s*(.*)', comment)
    if not m:
        return []
    return [x.strip() for x in m.group(1).split(';') if x.strip()]

def variants(tok):
    t = tok.rstrip('.').rstrip(',')
    v = {t, t.replace(',', '')}
    return v

block = []
unsourced, missing, checked = [], [], 0
for idx, line in enumerate(lines):
    code, com = split_comment(line)
    if 'SRCBLOCK:' in com:
        block = srcs_of(com); continue
    if 'ENDSRC' in com:
        block = []; continue
    if line.lstrip().startswith('%'):
        continue
    code = strip_noise(code)
    toks = []
    for m in re.finditer(r'SCI\(([^,]*),([^)]*)\)', code):
        toks.append(('sci', m.group(1).strip(), m.group(2).strip()))
    code2 = re.sub(r'SCI\([^)]*\)', ' ', code)
    for m in NUM.finditer(code2):
        tok = m.group(1).rstrip('.,')
        if not tok:
            continue
        plain = tok.replace(',', '')
        try:
            val = float(plain)
        except ValueError:
            continue
        if '.' not in plain and val <= 12:
            continue
        if '.' not in plain and 1990 <= val <= 2030 and len(plain) == 4:
            continue
        toks.append(('n', tok))
    if not toks:
        continue
    srcs = srcs_of(com) or block
    if not srcs:
        # look ahead up to 3 lines for a trailing SRC in the same paragraph
        for k in range(idx + 1, len(lines)):
            if lines[k].strip() == '' or lines[k].lstrip().startswith(('\\item','\\paragraph','\\section','\\subsection','\\begin{table','\\end{table')):
                break
            c2 = split_comment(lines[k])[1]
            if 'SRC:' in c2 and 'SRCBLOCK' not in c2:
                srcs = srcs_of(c2); break
    if not srcs:
        unsourced.append((idx + 1, [t[1] for t in toks]))
        continue
    texts = [load(s) for s in srcs]
    for s, t in zip(srcs, texts):
        if t is None:
            missing.append((idx + 1, 'SOURCE FILE NOT FOUND: ' + s))
    texts = [t for t in texts if t]
    for t in toks:
        checked += 1
        if t[0] == 'sci':
            mant, ex = t[1], t[2]
            e = int(ex)
            cands = {'%se%s' % (mant, ex), '%se%+03d' % (mant, e), '%se-%02d' % (mant, abs(e)) if e < 0 else '%se+%02d' % (mant, e),
                     '%s×10' % mant, mant}
            ok = any(any(c in tx for c in cands) for tx in texts)
            label = '%se%s' % (mant, ex)
        else:
            ok = any(any(v in tx for v in variants(t[1])) for tx in texts)
            label = t[1]
        if not ok:
            missing.append((idx + 1, '%s not in %s' % (label, '; '.join(srcs))))

print('checked number tokens:', checked)
print('\n== UNSOURCED lines (numbers with no SRC) ==')
for ln, toks in unsourced:
    print(ln, toks)
print('\n== NOT FOUND in cited source ==')
for ln, msg in missing:
    print(ln, msg)
print('\nunsourced lines: %d ; not-found tokens: %d' % (len(unsourced), len(missing)))
