"""Font size of the deletion-scope quote vs the body text, from the PDF (pypdf text runs; size = Tf x Tm scale).
usage: uv run --with pypdf python quotesize.py DRAFT.pdf  -> the quote's runs' sizes and the body's modal size"""
import sys
from collections import Counter
from pypdf import PdfReader
START, END = 'Neither frozen network is refit', 'on the surviving requests are unchanged'
for i, pg in enumerate(PdfReader(sys.argv[1]).pages, 1):
    if START not in (pg.extract_text() or '').replace('\n', ' '): continue
    runs = []
    pg.extract_text(visitor_text=lambda t, cm, tm, fd, fs: runs.append((t, round(fs * tm[3], 2))))
    q, body, inq = Counter(), Counter(), False
    for t, s in runs:
        if 'deleted rows, and the address book likewise' in t or 'After deleting a set of rows' in t: inq = True
        (q if inq else body)[s] += len(t.strip())
        if END in t or 'unchanged.' in t and inq: inq = False
    print(f'page {i}: quote runs (size: chars) {q.most_common(4)}; other text on the page {body.most_common(3)}')
