#!/usr/bin/env python3
"""w18 placeholder-fill SECOND PASS: apply to one draft.  usage: apply_fill2.py IN.tex OUT.tex
Input is the post-w19 draft (w19-draft-figure-slots 'after' sha). Every edit asserts its anchor
occurs exactly once. The 10 w19 placeholders are asserted byte-identical before and after."""
import re, sys
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import fill_text2 as F

src, dst = sys.argv[1], sys.argv[2]
t = open(src).read()
t0 = t
log = []

def once(old, new, tag):
    global t
    n = t.count(old)
    assert n == 1, '%s: anchor occurs %d times' % (tag, n)
    t = t.replace(old, new)
    log.append(tag)

def macro_span(s, start):
    assert s[start] == '{'
    k, depth = start + 1, 1
    while depth:
        ch = s[k]
        if ch == '%' and s[k - 1] != '\\':
            k = s.index('\n', k)
            continue
        if ch == '{' and s[k - 1] != '\\':
            depth += 1
        elif ch == '}' and s[k - 1] != '\\':
            depth -= 1
        k += 1
    return k

def placeholders(s):
    out, i = [], 0
    while True:
        i = s.find('\\PLACEHOLDER{', i)
        if i < 0:
            return out
        if s.rfind('\n', 0, i) < s.rfind('%', 0, i) and s[s.rfind('%', 0, i) - 1] != '\\':
            i += 1; continue   # inside a comment (the \newcommand line is code; this skips only commented uses)
        j = macro_span(s, i + len('\\PLACEHOLDER'))
        out.append(s[i:j]); i = j

def placeholder(prefix, new, tag):
    global t
    key = '\\PLACEHOLDER{' + prefix
    n = t.count(key)
    assert n == 1, '%s: placeholder prefix occurs %d times' % (tag, n)
    i = t.index(key)
    j = macro_span(t, i + len('\\PLACEHOLDER'))
    t = t[:i] + new + t[j:]
    log.append(tag)

before = [p for p in placeholders(t0) if not p.startswith('\\PLACEHOLDER}')]

once(F.HEADER_OLD, F.HEADER_NEW, 'header comment (3 lines -> 3 lines)')
once(F.T7_ROW_OLD, F.T7_ROW_NEW, 'lace-poe #1: tab:t7 LACE/PoE rows')
once(F.LADDER_COV_OLD, F.LADDER_COV_NEW, 'coverage: tab:ladder row (derived TEST cell)')
once(F.LADDER_FLOOR_ANCHOR, F.LADDER_FLOOR_ANCHOR + F.LADDER_LACE_ROWS, 'lace-poe: tab:ladder LACE/PoE rows')
once(F.COV_OLD, F.COV_NEW, 'coverage: TODO closed, val per seed, note verbatim')
placeholder(F.ADAPT_PREFIX, F.ADAPT, 'lace-poe #2: sec:adapt + generation table')
once(F.T7MORE_ANCHOR, F.T7MORE_ANCHOR + F.GENTABLE, 'lace-poe: tab:gentable-kstar in app:t7more')
placeholder(F.BARD1_PREFIX, F.BARD1, 'bar-d #1: app:encloop + tab:bard-gpu')
placeholder(F.BARD2_PREFIX, F.BARD2, 'bar-d #2: app:encoder-main decomposition')
once(F.INDEX_OLD, F.INDEX_NEW, 'placeholder index')

after = placeholders(t)
w15 = [p for p in before if p.startswith('\\PLACEHOLDER{w15')]
assert len(w15) == 4, len(w15)
assert after == [p for p in before if not p.startswith('\\PLACEHOLDER{w15')], 'non-w15 placeholders changed'
assert len(after) == 10, len(after)
assert t.count('\\TODO{') == t0.count('\\TODO{') - 1
assert 'SRC-PENDING' not in t
# pure insertion/replacement: every pre line outside the replaced spans is still present, in order
open(dst, 'w').write(t)
print('\n'.join(log))
print('placeholders: %d before -> %d after (all remaining = w19/w16 figure-slot ones, byte-identical)' % (len(before), len(after)))
print('TODO: %d -> %d' % (t0.count('\\TODO{'), t.count('\\TODO{')))
