"""Word-level diff of the rendered PDF text, before -> after: the "no content change" proof.

usage: python worddiff.py BEFORE.pdf.txt AFTER.pdf.txt [--plant]   (text files written by pw1_draft_cuts/pdftext.py)
Page furniture is removed from each page, and counted, before comparison:
  * the ICLR line-number ruler: page p carries the lines 54(p-1) .. 54p-1 as bare 3-digit lines (removed only where
    such a line matches that page's own ruler range);
  * the running header line "Under review as a conference paper at ICLR 2027";
  * the page number (last non-empty line equal to p).
Then a hyphen at the end of a line is joined to the next line (LaTeX hyphenation), and the text is split into words.
Reported:
  (1) WORD diff: difflib opcodes over the word sequences. A delete/insert pair with the same words is a MOVED block
      (a float placed on another page); a replace whose two sides are equal once hyphens and spaces are removed is a
      HYPHENATION / EXTRACTION artefact (a line break moved, pypdf spacing); everything else is CONTENT.
  (2) CHARACTER stream: every whitespace and hyphen removed; equal -> no letter or digit of the page body changed,
      only its layout.
  (3) the bag of words (printed; pypdf spacing makes it noisy) and the order-free character multiset (must be equal).
--plant: positive control, changes one word and deletes another in AFTER in memory; both must show up as CONTENT.
--plant-move: control the other way, moves a 60-word block 3,000 words on in AFTER; must stay EMPTY (MOVED).
Exit 0 iff no CONTENT leftover and equal character multisets.
"""
import collections
import difflib
import re
import sys

HEADER = 'Under review as a conference paper at ICLR 2027'


def pages(path):
    t = open(path).read()
    parts = re.split(r'\n=== PAGE (\d+) ===\n', t)
    return [(int(parts[i]), parts[i + 1]) for i in range(1, len(parts), 2)]


def clean(path):
    body, furn = [], collections.Counter()
    for p, txt in pages(path):
        lines = txt.split('\n')
        ruler = {f'{n:03d}' for n in range(54 * (p - 1), 54 * p)}
        keep = []
        for l in lines:
            s = l.strip()
            if s in ruler:
                furn['ruler'] += 1
                continue
            if s == HEADER:
                furn['header'] += 1
                continue
            keep.append(l)
        while keep and not keep[-1].strip():
            keep.pop()
        if keep and keep[-1].strip() == str(p):
            keep.pop()
            furn['pagenum'] += 1
        body.append('\n'.join(keep))
    t = '\n'.join(body)
    t = re.sub(r'-\n', '-', t)  # keep the hyphen, join the line; normalised below
    return t.split(), furn


def norm(w):
    return w.replace('-', '').replace('‐', '')


def ca_eq_sorted(a, b):
    """order-free check: the same multiset of characters (spaces and hyphens removed)."""
    return collections.Counter(''.join(map(norm, a))) == collections.Counter(''.join(map(norm, b)))


def main():
    a, fa = clean(sys.argv[1])
    b, fb = clean(sys.argv[2])
    if '--plant' in sys.argv:
        i = len(b) // 3
        b = b[:i] + ['PLANTEDWORD'] + b[i + 1:]
        del b[2 * len(b) // 3]
    if '--plant-move' in sys.argv:  # a float-like move: 60 words cut from one page and pasted 3,000 words later
        i = len(b) // 2
        blk = b[i:i + 60]
        b = b[:i] + b[i + 60:]
        b = b[:i + 3000] + blk + b[i + 3000:]
    print(f'furniture removed  before: {dict(fa)}  after: {dict(fb)}')
    print(f'words  before: {len(a)}  after: {len(b)}')
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    ops = [o for o in sm.get_opcodes() if o[0] != 'equal']
    kinds = collections.Counter()
    resA, resB = [], []  # residual pieces (normalised, no spaces) that are not a local hyphenation/extraction artefact
    for tag, i1, i2, j1, j2 in ops:
        A, B = a[i1:i2], b[j1:j2]
        if A and B and ''.join(map(norm, A)) == ''.join(map(norm, B)):
            k = 'HYPHENATION/EXTRACTION'
        else:
            k = 'residual'
            if A:
                resA.append(''.join(map(norm, A)))
            if B:
                resB.append(''.join(map(norm, B)))
        kinds[k] += 1
        print(f'  {k:24s} {tag:7s} before[{i1}:{i2}] {" ".join(A)[:110]!r} -> after[{j1}:{j2}] {" ".join(B)[:110]!r}')
    # move resolution: repeatedly take the longest common substring (>= MOVE chars) between a before-piece and an
    # after-piece and remove it from both (a block of text that a float placement moved). What is left on each side,
    # joined in order, must be the same string (a word split by a float, e.g. con|Table ...|dition -> condition).
    MOVE = 20
    moved = []
    while True:
        best = (0, None)
        for x, pa in enumerate(resA):
            for y, pb in enumerate(resB):
                if min(len(pa), len(pb)) <= best[0]:
                    continue
                m = difflib.SequenceMatcher(None, pa, pb, autojunk=False).find_longest_match(0, len(pa), 0, len(pb))
                if m.size > best[0]:
                    best = (m.size, (x, y, m))
        if best[0] < MOVE:
            break
        x, y, m = best[1]
        pa, pb = resA[x], resB[y]
        moved.append(pa[m.a:m.a + m.size])
        resA[x:x + 1] = [q for q in (pa[:m.a], pa[m.a + m.size:]) if q]
        resB[y:y + 1] = [q for q in (pb[:m.b], pb[m.b + m.size:]) if q]
    left_a, left_b = ''.join(resA), ''.join(resB)
    print(f'moved blocks resolved: {len(moved)} ({sum(map(len, moved))} chars); '
          f'leftover before {resA!r} after {resB!r}')
    content = [] if left_a == left_b else [(left_a[:300], left_b[:300])]
    kinds['MOVED blocks'] = len(moved)
    kinds['CONTENT'] = len(content)
    ca = ''.join(map(norm, a))
    cb = ''.join(map(norm, b))
    bag = collections.Counter(map(norm, a)) - collections.Counter(map(norm, b)), \
        collections.Counter(map(norm, b)) - collections.Counter(map(norm, a))
    print(f'(1) word-diff opcodes: {len(ops)}  by kind: {dict(kinds)}')
    print(f'(2) character stream (no whitespace, no hyphens): {"EQUAL" if ca == cb else "DIFFERS"} ({len(ca)} / {len(cb)} chars)')
    print(f'(3) bag of words, before-only: {dict(bag[0])}  after-only: {dict(bag[1])}')
    for c in content:
        print('CONTENT (leftover before / after):', c)
    ok = not content and ca_eq_sorted(a, b)
    print('RESULT:', 'EMPTY (no rendered word changed)' if ok else 'NOT EMPTY')
    sys.exit(0 if ok else 1)


main()
