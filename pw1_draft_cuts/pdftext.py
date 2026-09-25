"""usage: uv run --with pypdf python pdftext.py IN.pdf OUT.txt  (page-separated text, for residual scans)"""
import sys
from pypdf import PdfReader
r = PdfReader(sys.argv[1])
with open(sys.argv[2], 'w') as f:
    for i, p in enumerate(r.pages, 1):
        f.write('\n=== PAGE %d ===\n' % i + (p.extract_text() or ''))
