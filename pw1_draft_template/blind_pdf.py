"""Double-blind scan of a built PDF (formatting level): document /Info, every embedded figure's PTEX.FileName and
PTEX.InfoDict (pdfTeX copies the included PDF's /Info), XMP metadata, and every decompressed stream and string for
identity tokens.
usage: uv run --with pypdf python blind_pdf.py FILE.pdf [TOKEN ...]
"""
import re
import sys

import pypdf
from pypdf.generic import DictionaryObject, IndirectObject, StreamObject

f = sys.argv[1]
toks = sys.argv[2:] or ['batmac', '/Users/', 'forgis', 'Forgis', 'clu/', 'chlu', 'CHLU', '@', 'github.com/', 'home/']
r = pypdf.PdfReader(f)
print('## /Info:', {k: str(v) for k, v in (r.metadata or {}).items()})
print('## XMP present:', r.xmp_metadata is not None)
seen, files, infos, hits = set(), [], [], {t: 0 for t in toks}
nstreams = 0


def walk(o):
    global nstreams
    if isinstance(o, IndirectObject):
        if (o.idnum, o.generation) in seen:
            return
        seen.add((o.idnum, o.generation))
        o = o.get_object()
    if isinstance(o, DictionaryObject):
        if '/PTEX.FileName' in o:
            files.append(str(o['/PTEX.FileName']))
        if '/PTEX.InfoDict' in o:
            infos.append({k: str(v) for k, v in o['/PTEX.InfoDict'].get_object().items()})
        if isinstance(o, StreamObject):
            nstreams += 1
            try:
                data = o.get_data()
            except Exception:
                data = b''
            for t in toks:
                hits[t] += data.count(t.encode())
        for k, v in o.items():
            if k != '/Parent':
                walk(v)
    elif isinstance(o, list):
        for v in o:
            walk(v)
    else:
        s = str(o)
        for t in toks:
            hits[t] += s.count(t)


walk(r.trailer)
print('## embedded figures (PTEX.FileName):', len(files))
for x in files:
    print('   ', x)
print('## PTEX.InfoDict of embedded figures:', len(infos))
for x in sorted({repr(sorted(i.items())) for i in infos}):
    print('   ', x)
print('## streams decompressed:', nstreams, ' token hits (streams + strings):', hits)
