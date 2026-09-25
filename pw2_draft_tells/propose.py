"""For every unit that fails T7a, list split points that would put a sentence of <= 8 words directly after one of
>= 18 words (the split is made at '; ', ', and ', ', so ', ', but ', ': ' outside braces and math).
usage: python3 propose.py DRAFT.tex [--hhfrom DRAFT_FOR_HEADHELD]"""
import re, sys
from detect import detect, norm, nwords, LONG, SHORT
from segment import strip_comment
path = sys.argv[1]
L = open(path).read().split('\n')
U, hits = detect(path)
bad = {(h['line']) for h in hits if h['tell'] == 'T7' and h['rule'].startswith('no short')}
DEL = re.compile(r'(; |, and |, so |, but |: |, while |, which )')
for u in U:
    if u['lines'][0] not in bad or u['headheld']: continue
    raw = '\n'.join(L[k - 1] for k in u['lines'])
    m = '\n'.join(strip_comment(L[k - 1]).ljust(len(L[k - 1])) for k in u['lines'])
    ends = [0] + [x.end() for x in re.finditer(r'(?<=[.?!])[)\'"’”]*\s+(?=[A-Z0-9(`\\$])', m)] + [len(m)]
    print('=' * 100); print('L%d  %s' % (u['lines'][0], norm(u['code'])[:140]))
    for a, b in zip(ends, ends[1:]):
        seg = m[a:b]
        for x in DEL.finditer(seg):
            pre = seg[:x.start()]
            if pre.count('{') != pre.count('}') or pre.count('$') % 2: continue
            hw = nwords(norm(pre)); tw = nwords(norm(seg[x.end():]))
            if tw <= SHORT and hw >= LONG:
                print('   TAIL  %2d|%-2d %r ... %r' % (hw, tw, norm(pre)[-50:], norm(seg[x.end():])))
            elif hw <= SHORT and tw >= 1:
                print('   HEAD  %2d|%-2d %r' % (hw, tw, norm(pre)))
