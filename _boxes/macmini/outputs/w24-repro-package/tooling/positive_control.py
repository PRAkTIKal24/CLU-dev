"""Positive control for the anonymity scanner (tooling only).

    positive_control.py PKG_DIR WORK_DIR REPORT_JSON [--python PY_WITH_MATPLOTLIB]

Copies the package to WORK_DIR, plants one token of every category in (1) a file body, (2) a file
name, (3) a PDF metadata field, (4) a PNG metadata field (tEXt), and additionally (5) a zlib-
compressed PNG zTXt chunk, (6) bytes appended to an array file (.eqx), (7) an .npz member name,
then runs scan.scan() and checks every plant is reported at its file. Exit 1 if any plant is missed.
"""
import json
import shutil
import subprocess
import sys
import zipfile
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import scan  # noqa: E402

PKG, WORK, REP = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
PY = sys.argv[sys.argv.index('--python') + 1] if '--python' in sys.argv else sys.executable

# one token per category, taken from the list itself
TOKENS = {
    'people/org': 'Jawahar',
    'machines/paths': 'thinkpad',
    'tooling': 'Anthropic',
    'programme': 'PREREG',
}
NAME_TOKENS = {  # file-name plants (a name component must carry the token)
    'people/org': 'forgis_notes.txt',
    'machines/paths': 'batmac_run.sh',
    'tooling': 'claude_helper.py',
    'programme': 'spoke_outputs',
}


def png_with_chunk(path, key, text, compressed=False):
    import struct
    def chunk(t, d):
        c = struct.pack('>I', len(d)) + t + d
        return c + struct.pack('>I', zlib.crc32(t + d) & 0xffffffff)
    raw = b'\x00\x00\x00\x00'  # 1x1 grey, filter byte + 1 px (8-bit grey + alpha? keep grey)
    ihdr = struct.pack('>IIBBBBB', 1, 1, 8, 0, 0, 0, 0)
    idat = zlib.compress(b'\x00\x80')
    if compressed:
        meta = chunk(b'zTXt', key.encode() + b'\x00\x00' + zlib.compress(text.encode()))
    else:
        meta = chunk(b'tEXt', key.encode() + b'\x00' + text.encode())
    path.write_bytes(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', ihdr) + meta + chunk(b'IDAT', idat) + chunk(b'IEND', b''))


def main():
    if WORK.exists():
        raise SystemExit('work dir exists: %s' % WORK)
    shutil.copytree(PKG, WORK / 'pkg')
    root = WORK / 'pkg'
    plants = []
    # (1) file body: one token per category, one file per category
    for cat, tok in TOKENS.items():
        p = root / 'nclu' / ('plant_body_%s.py' % cat.split('/')[0])
        p.write_text('# neutral code\nVALUE = "%s"\n' % tok)
        plants.append(('body', cat, str(p.relative_to(root)), tok))
    # (1b) review-item IDs and a deletion scar (AMENDMENT-3): PW-1 in a comment, PW1_X as an
    # identifier, and the scar "(, 2026-09-25)" left in a docstring
    for tag, body, tok in [
        ('pwdash', '# see PW-1 for the change\nVALUE = 1\n', 'PW-1'),
        ('pwident', 'PW1_X = ("a", "b")\n', 'PW1'),
        ('scar', '"""Figure builder (, 2026-09-25) for the panel."""\nVALUE = 2\n', '(, 2026-09-25)'),
        ('scaronly', '"""Figure builder (, ) for the panel."""\nVALUE = 3\n', '(,'),
    ]:
        p = root / 'figures' / ('plant_%s.py' % tag)
        p.write_text(body)
        plants.append(('body (AMENDMENT-3)', 'programme' if tag.startswith('pw') else 'scrub scars',
                       str(p.relative_to(root)), tok))
    # (2) file / directory names
    for cat, nm in NAME_TOKENS.items():
        p = root / 'figures' / nm
        if nm.endswith(('.txt', '.sh', '.py')):
            p.write_text('neutral\n')
        else:
            p.mkdir()
            (p / 'x.txt').write_text('neutral\n')
        plants.append(('name', cat, str(p.relative_to(root)), nm.split('_')[0]))
    # (3) PDF metadata and (4) PNG tEXt metadata, via matplotlib in the package's venv
    gen = WORK / 'gen_meta.py'
    gen.write_text('''import sys, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
spec = json.loads(sys.argv[1])
for path, fmt, key, val in spec:
    fig = plt.figure(figsize=(1, 1)); plt.plot([0, 1], [0, 1])
    fig.savefig(path, format=fmt, metadata={key: val}); plt.close(fig)
''')
    spec = []
    for cat, tok in TOKENS.items():
        c = cat.split('/')[0]
        spec.append((str(root / 'figures' / ('plant_meta_%s.pdf' % c)), 'pdf', 'Author', tok))
        spec.append((str(root / 'figures' / ('plant_meta_%s.png' % c)), 'png', 'Author', tok))
        plants.append(('pdf-metadata', cat, 'figures/plant_meta_%s.pdf' % c, tok))
        plants.append(('png-metadata', cat, 'figures/plant_meta_%s.png' % c, tok))
    subprocess.run([PY, str(gen), json.dumps(spec)], check=True)
    # (5) compressed PNG zTXt
    png_with_chunk(root / 'figures' / 'plant_ztxt.png', 'Comment', 'Co-Authored-By: someone', compressed=True)
    plants.append(('png-zTXt (compressed)', 'tooling', 'figures/plant_ztxt.png', 'Co-Authored-By'))
    # (6) bytes appended to an array file
    eq = root / 'weights' / 'classifier' / 'seed_0' / 'external_classifier.eqx'
    assert eq.exists(), eq
    with open(eq, 'ab') as fh:
        fh.write(b'hidden note: forgisprova')
    plants.append(('bytes appended to .eqx', 'people/org', 'weights/classifier/seed_0/external_classifier.eqx', 'forgisprova'))
    # (7) an .npz member name
    with zipfile.ZipFile(root / 'weights' / 'plant.npz', 'w') as z:
        z.writestr('tailscale_member.npy', b'\x93NUMPY\x01\x00v\x00' + b"{'descr': '<f4', 'fortran_order': False, 'shape': (1,), }".ljust(118) + b'\n' + b'\x00\x00\x80?')
    plants.append(('npz member name', 'machines/paths', 'weights/plant.npz', 'tailscale'))

    scan.GIT_EXTRA.extend(scan.git_identities('/Users/batmac/clu/worktrees/CHLU-w24-repro-package'))
    res = scan.scan(root)
    out = []
    for where, cat, f, tok in plants:
        hits = [h for h in res['hits'] if (h['file'] == f or h['file'].startswith(f + '/') or h.get('context') == Path(f).name)
                and tok.lower() in h['match'].lower() or (h['file'] == f and tok.lower() in h.get('context', '').lower())]
        out.append({'placement': where, 'category': cat, 'file': f, 'token': tok, 'caught': bool(hits),
                    'hit_rules': sorted({h['rule'] for h in hits})})
    missed = [o for o in out if not o['caught']]
    REP.write_text(json.dumps({'plants': out, 'n_plants': len(out), 'n_caught': len(out) - len(missed),
                               'missed': missed, 'total_hits_in_planted_copy': len(res['hits'])}, indent=1))
    for o in out:
        print('%-26s %-15s %-45s %-16s %s' % (o['placement'], o['category'], o['file'], o['token'],
                                             'CAUGHT ' + ','.join(o['hit_rules']) if o['caught'] else 'MISSED'))
    print('plants %d caught %d missed %d' % (len(out), len(out) - len(missed), len(missed)))
    sys.exit(1 if missed else 0)


main()
