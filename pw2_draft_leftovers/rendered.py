"""Rendered spot check (pypdf): the edited sentences as they print, and the quote block's font size against the body.
usage: uv run --with pypdf python rendered.py BASE.pdf STAGE.pdf"""
import sys, re
from collections import Counter
from pypdf import PdfReader
PROBES = ['printed score (Fig', 'or K = 28 (Fig', 'K=28 (Fig', 'all three seeds (Tab', 'at seed 2 (Fig', 'adapter (Tab',
          'were not persisted, so their', 'about thirty on the', 'nearly a gigabyte', 'about four', 'collapses (Tab',
          'more than a tuple', 'outside the sweep', 'PLACEHOLDER', 'TODO:', 'byte ratio', 'pending a run', 'Two rows remain',
          'seed at 10']
def text(p): return [pg.extract_text() or '' for pg in PdfReader(p).pages]
def sizes(p, needle_words):
    """font sizes (Tf size x text-matrix scale) of the runs on the page that contains the quote's opening words"""
    r = PdfReader(p); out = Counter(); body = Counter()
    for pg in r.pages:
        t = pg.extract_text() or ''
        if needle_words not in t: continue
        runs = []
        def v(txt, cm, tm, fd, fs): runs.append((txt, round(fs * tm[0], 2)))
        pg.extract_text(visitor_text=v)
        inq = False
        for txt, s in runs:
            if 'HE MATCH IS ON THE' in txt.upper().replace('\n', ' ') or 'The match is on' in txt: inq = True
            if inq: out[s] += len(txt)
            else: body[s] += len(txt)
            if inq and 'unchanged.' in txt: inq = False
        return out.most_common(3), body.most_common(3)
    return None, None
for tag, p in (('before', sys.argv[1]), ('after', sys.argv[2])):
    T = text(p); full = ' '.join(T).replace('\n', ' ')
    print(f'== {tag}: {p.split("/")[-1]} pages={len(T)}')
    for pr in PROBES:
        hits = [i + 1 for i, t in enumerate(T) if pr in t.replace('\n', ' ')]
        m = re.search(re.escape(pr), full)
        print(f'  {pr!r}: pages {hits}' + (f' | …{full[max(0, m.start() - 90):m.end() + 60]}…' if m and tag == 'after' else ''))
    q, b = sizes(p, 'outside')
    print(f'  quote font sizes (size: chars) {q}; rest of that page {b}')
