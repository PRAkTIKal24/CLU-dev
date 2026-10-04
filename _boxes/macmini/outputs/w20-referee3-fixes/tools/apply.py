import sys, re
sys.path.insert(0, '/Users/batmac/clu/outputs/w20-referee3-fixes/tools')
from common_edits import COMMON
import perfile_edits as PF
P = '/Users/batmac/clu/outputs/w17-paper/'
COMMON = COMMON + [("HDR", "Ten PLACEHOLDERs remain, all w19 (w19-draft-figure-slots/FIGURE-SLOTS.md).",
                    "Eleven PLACEHOLDERs remain: ten w19 (w19-draft-figure-slots/FIGURE-SLOTS.md), one w20.")]
dry = '--dry' in sys.argv
log = []
corr_saved = None
for f, per, sysblk in [('DRAFT-v2.tex', PF.MAIN, PF.SYS_MAIN), ('DRAFT-v2-style.tex', PF.STYLE, PF.SYS_STYLE)]:
    t = open(P + f).read()
    for k, o, n in COMMON + per:
        c = t.count(o)
        assert c == 1, (f, k, c)
        t = t.replace(o, n); log.append((f, k))
    # (a) systems block
    a = t.index(PF.BLOCK[0]); b = t.index(PF.BLOCK[1], a) + len(PF.BLOCK[1])
    old = t[a:b]
    assert old.count('three months is incomplete') == 1 and 'is empty' in old, (f, 'block')
    t = t[:a] + sysblk + t[b:].lstrip('\n')
    log.append((f, 'A-sys'))
    # (e) Corrections absorbed out of the submission source
    m = re.search(r"% -{80,}\n% REFEREE-v2 M7\(e\).*?\n\\fi\n", t, re.S)
    assert m, (f, 'corr')
    if corr_saved is None: corr_saved = m.group(0)
    else: assert corr_saved == m.group(0), 'corrections blocks differ'
    t = t[:m.start()] + ("% ---------------------------------------------------------------------------------------\n"
        "% \"Corrections absorbed\" is moved OUT of the submission source (REFEREE-v2 M7(e); w20 referee3-fixes\n"
        "% item (e)). The block is kept verbatim, provenance comments included, in CORRECTIONS-ABSORBED.tex.\n") + t[m.end():]
    log.append((f, 'E-corr'))
    # (d) tab:adapt seeds column
    a = t.index(r"\label{tab:adapt}"); b = t.index(r"\end{table}", a)
    blk = t[a:b]
    blk2 = blk.replace(r"\begin{tabular}{llccccc}", r"\begin{tabular}{lclccccc}", 1)
    blk2 = blk2.replace("seed & arm & enc. params trained", "seed & seeds & arm & enc. params trained", 1)
    blk2, nrow = re.subn(r"(?m)^([012]) & (frozen|adapter|anchor) & ", r"\1 & 1 & \2 & ", blk2)
    assert nrow == 9 and blk2.count('lclccccc') == 1 and 'seed & seeds & arm' in blk2, (f, 'adapt', nrow)
    t = t[:a] + blk2 + t[b:]
    log.append((f, 'M6-adapt'))
    if not dry:
        open(P + f, 'w').write(t)
if not dry:
    open(P + 'CORRECTIONS-ABSORBED.tex', 'w').write(
        "% CORRECTIONS-ABSORBED.tex -- moved verbatim out of DRAFT-v2.tex and DRAFT-v2-style.tex by w20\n"
        "% referee3-fixes item (e) (REFEREE-v2 M7(e)). NOT part of the submission. Byte-identical in both drafts\n"
        "% before the move. Paths in SRC comments are relative to ~/clu/evidence/outputs/.\n" + corr_saved)
for x in log: print(*x)
print('DRY' if dry else 'APPLIED')
