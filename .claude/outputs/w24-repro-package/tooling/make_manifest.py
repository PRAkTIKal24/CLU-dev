"""Write MANIFEST.md into the package: every file, size, SHA-256; total. Tooling only."""
import hashlib
import sys
from pathlib import Path

P = Path(sys.argv[1])
rows, tot = [], 0
for p in sorted(P.rglob('*')):
    if not p.is_file() or p.name == 'MANIFEST.md':
        continue
    b = p.read_bytes()
    tot += len(b)
    rows.append((str(p.relative_to(P)), len(b), hashlib.sha256(b).hexdigest()))
L = ['# MANIFEST', '',
     'Every file of this folder except this one: path, size in bytes, SHA-256.', '',
     'Files: %d. Total size: %s bytes (%.1f MB), excluding MANIFEST.md.' % (len(rows), format(tot, ','), tot / 1e6), '',
     '| file | bytes | sha256 |', '|---|---:|---|']
L += ['| `%s` | %s | `%s` |' % (f, format(n, ','), h) for f, n, h in rows]
(P / 'MANIFEST.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
print('MANIFEST.md: %d files, %d bytes' % (len(rows), tot))
