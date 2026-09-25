"""w23 fold: stage the register pass (INSERT-ONLY on build_tables.py and TABLES.template.md; adds cells_pass7.py).
usage: stage_register7.py <base register dir> <frag dir> <stage dir>"""
import shutil, sys
from pathlib import Path
B, F, S = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
if S.exists():
    raise SystemExit(f'stage dir exists: {S}')
shutil.copytree(B, S, ignore=shutil.ignore_patterns('__pycache__', '*.bundle'))
shutil.copy(F / 'cells_pass7.py', S / 'cells_pass7.py')
b = (S / 'build_tables.py').read_text(encoding='utf-8').split('\n')
def ins_after(prefix, new):
    idx = [i for i, l in enumerate(b) if l.startswith(prefix)]; assert len(idx) == 1, (prefix, idx); b[idx[0] + 1:idx[0] + 1] = new
def ins_before(prefix, new):
    idx = [i for i, l in enumerate(b) if l.startswith(prefix)]; assert len(idx) == 1, (prefix, idx); b[idx[0]:idx[0]] = new
ins_after('SRC["D4S"] = ', [
 '# Pass 6 of this register (w23 fold, 2026-09-25): the numbers W22 wrote into the drafts (lines of the pinned pass-5 draft',
 '# not in the pinned pass-4 text), each a DRAFT cell on the pinned pass-5 draft and a SOURCE cell on the first file its SRC',
 '# names. Plain verbatim cells (CELLS7); generated once by w23_fold/gen_cells_pass7.py + make_template7.py. Insert-only.',
 'from cells_pass7 import SRC7, CELLS7  # noqa: E402',
 'assert not set(SRC7) & set(SRC), sorted(set(SRC7) & set(SRC))',
 'SRC.update(SRC7)'])
ins_before('HIST_USED = set()', [
 'for _cid in CELLS7:  # w23 fold (register pass 6)',
 '    if _cid in CELLS:',
 '        raise SystemExit(f"duplicate cell id {_cid}")',
 'CELLS.update(CELLS7)'])
(S / 'build_tables.py').write_text('\n'.join(b), encoding='utf-8')
t = (S / 'TABLES.template.md').read_text(encoding='utf-8')
(S / 'TABLES.template.md').write_text(t.rstrip('\n') + '\n' + (F / 'TABLES.pass7.frag.md').read_text(encoding='utf-8'), encoding='utf-8')
print('staged', S)
