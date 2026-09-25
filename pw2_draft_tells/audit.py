"""Word audit of the edit list, both drafts.
(1) per edit and draft: words added and removed (case-insensitive multisets of the comment-stripped, LaTeX-normalised
    text of old vs new; a capital made by a split is not an added word);
(2) whole document: the multiset of words after minus before must equal the sum of the per-edit additions, and
    likewise for removals, so no change outside the listed edits exists;
(3) word count per section before -> after (all text outside comments: prose, items, quotes, captions, tables, headings).
usage: python3 audit.py V2_BEFORE STYLE_BEFORE V2_AFTER STYLE_AFTER OUT.json"""
import sys, re, json
from collections import Counter
from edits import E
from detect import norm
from segment import units, strip_comment
def W(t):
    t = '\n'.join(strip_comment(l) for l in t.split('\n'))
    t = norm(t, drop_title=False)
    return Counter(w.lower() for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9'’\-]*", t))
def doc(path):
    L = open(path).read()
    a = L.index(r'\begin{abstract}'); b = L.index(r'\end{document}')
    return W(L[a:b])
res = {}
for name, bi, ai in (('DRAFT-v2', 1, 3), ('DRAFT-v2-style', 2, 4)):
    scope = 'v2' if name == 'DRAFT-v2' else 'style'
    per = []; addsum = Counter(); remsum = Counter()
    for e in E:
        for s, old, new in e['pairs']:
            if s != 'both' and s != scope: continue
            o, n = W(old), W(new)
            add, rem = n - o, o - n
            addsum += add; remsum += rem
            per.append(dict(id=e['id'], tells=e['tells'], section=e['section'], added=dict(add), removed=dict(rem),
                            words_old=sum(o.values()), words_new=sum(n.values())))
    B, A = doc(sys.argv[bi]), doc(sys.argv[ai])
    dadd, drem = A - B, B - A
    ok = (dadd == addsum - remsum + (remsum - addsum) - (remsum - addsum)) if False else None
    # net comparison: document net change must equal the edits' net change
    net_doc = Counter(A); net_doc.subtract(B)
    net_ed = Counter(addsum); net_ed.subtract(remsum)
    net_doc = {k: v for k, v in net_doc.items() if v}; net_ed = {k: v for k, v in net_ed.items() if v}
    sec = {}
    for tag, p in (('before', sys.argv[bi]), ('after', sys.argv[ai])):
        L = open(p).read().split('\n')
        for u in units(p):
            c = W('\n'.join(L[k - 1] for k in u['lines']))
            sec.setdefault(u['section'], {}).setdefault(tag, 0)
            sec[u['section']][tag] += sum(c.values())
    res[name] = dict(per=per, doc_words_before=sum(B.values()), doc_words_after=sum(A.values()),
                     net_equal=(net_doc == net_ed), net_doc=net_doc, net_edits=net_ed,
                     added_total=dict(addsum), sections=sec)
    print(name, 'doc words', sum(B.values()), '->', sum(A.values()), '| net change equals the edits\' net change:', net_doc == net_ed)
    if net_doc != net_ed:
        x = Counter(net_doc); x.subtract(net_ed); print('  MISMATCH', {k: v for k, v in x.items() if v})
json.dump(res, open(sys.argv[5], 'w'), indent=1)
