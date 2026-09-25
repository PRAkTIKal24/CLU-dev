"""Job 6: every placed figure stem is present (PDF + PNG) in both w17-paper/figures/ and paper-visuals/ with equal
hashes; each folder's installed two-build manifests agree with the files; each folder's README equals the README that
make_index.py regenerates in a scratch copy of the folder (so the README index matches the drafts and the files).

usage: python folder_audit.py W17_PAPER PAPER_VISUALS OUTDIR
"""
import hashlib, json, os, re, shutil, subprocess, sys
P, PV, O = sys.argv[1:4]
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(P, 'figures')
def sha(p): return hashlib.sha256(open(p, 'rb').read()).hexdigest()
stems = []
for d in ('DRAFT-v2.tex', 'DRAFT-v2-style.tex'):
    t = open(os.path.join(P, d)).read()
    code = '\n'.join(l.split('%')[0] if not l.lstrip().startswith('\\newcommand') else '' for l in t.split('\n'))
    s = re.findall(r'\\figslot\{([^}]*)\}', code) + [x[len('figures/'):-4] for x in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{(figures/[^}]*\.pdf)\}', code)]
    stems.append([x for x in s if '#' not in x])  # drop the \figslot macro's own figures/#1.pdf
assert stems[0] == stems[1], 'the two drafts place different stems'
rows = []
for s in stems[0]:
    for ext in ('pdf', 'png'):
        f = s + '.' + ext
        a = os.path.join(FIG, f); b = os.path.join(PV, f)
        ha = sha(a) if os.path.isfile(a) else None; hb = sha(b) if os.path.isfile(b) else None
        rows.append(dict(file=f, w17=ha, pv=hb, equal=bool(ha) and ha == hb))
print('placed stems per draft: %d (identical lists in both drafts); files %d; present+equal in both folders: %d'
      % (len(stems[0]), len(rows), sum(r['equal'] for r in rows)))
for r in rows:
    if not r['equal']: print('  PROBLEM', r)
# manifests
for name, F in (('w17-paper/figures', FIG), ('paper-visuals', PV)):
    for m in ('SHA256-w22-build1.txt', 'SHA256-w22-build2.txt', 'rendered/SHA256-pw1-render-build1.txt', 'rendered/SHA256-pw1-render-build2.txt'):
        base = os.path.dirname(os.path.join(F, m)); bad = 0; n = 0
        for l in open(os.path.join(F, m)):
            h, f = l.split(None, 1); f = f.strip(); n += 1
            p = os.path.join(base, f)
            if not os.path.isfile(p):
                p2 = os.path.join(F, 'retired-pw1', f)
                if os.path.isfile(p2): continue
                bad += 1; print('  MISSING', name, m, f); continue
            if sha(p) != h: bad += 1; print('  HASH MISMATCH', name, m, f)
        print('%s %s: %d entries, %d problems' % (name, m, n, bad))
# README regeneration in scratch copies
commit = subprocess.check_output(['git', '-C', R, 'log', '-1', '--format=%H', '--', 'w21_figures', 'w19_task_figures']).decode().strip()
for name, F in (('paper-visuals', PV), ('w17-paper-figures', FIG)):
    dst = os.path.join(O, 'readme-check', 'paper-visuals' if name == 'paper-visuals' else 'figures')
    if os.path.exists(dst): shutil.move(dst, dst + '.old')
    shutil.copytree(F, dst, ignore=shutil.ignore_patterns('__pycache__', 'engine'))
    r = subprocess.run(['python3', os.path.join(R, 'pw1_draft_figures', 'make_index.py'), dst, P, P, os.path.join(O, 'readme-check'), commit],
                       capture_output=True, text=True)
    a = open(os.path.join(F, 'README.md')).read(); b = open(os.path.join(dst, 'README.md')).read()
    # the copy's folder name differs only where make_index prints nothing path-specific; compare bytes
    print('%s README: make_index exit %d; installed README == regenerated: %s' % (name, r.returncode, a == b))
json.dump(rows, open(os.path.join(O, 'FOLDERS.json'), 'w'), indent=1)
