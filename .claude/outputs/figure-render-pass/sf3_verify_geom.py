"""Geometric verification that the reconstructed sf3 data == the data plotted in
the banked figure: render the reconstruction's DATA ONLY into axes boxes placed at
the banked figure's exact pixel frame, then compare colour masks inside the frame.
"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sf3_reconstruct import xs, ys, sl, ic, FLOOR, xb, yb, slb, icb, FIT

ORIG = ".claude/papers/neurreps-variants/v2/figs/fig2_anchor_cure_laws.png"
W, H, DPI = 1430, 546, 130
L1, R1, L2, R2, T, B = 91, 700, 801, 1410, 76, 464

fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
def rect(l, r):
    return [l / W, 1 - B / H, (r - l) / W, (B - T) / H]
a0 = fig.add_axes(rect(L1, R1)); a1 = fig.add_axes(rect(L2, R2))
a0.loglog(xs, ys, "o-", color="tab:green", ms=6)
a0.loglog(xs[:FIT], 10 ** (ic + sl * np.log10(xs[:FIT])), "--", color="0.4", lw=2)
a0.axhline(FLOOR, color="red", ls=":", lw=1.5)
a1.loglog(xb, yb, "o", color="tab:purple", ms=6)
a1.loglog(xb, 10 ** (icb + slb * np.log10(xb)), "--", color="0.4", lw=2)
for a in (a0, a1):
    a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values(): s.set_visible(False)
P = ".claude/scratch/figure-render-pass/sf3_geom.png"
fig.savefig(P, dpi=DPI)

A = np.asarray(Image.open(ORIG).convert("RGB"), int)
Bm = np.asarray(Image.open(P).convert("RGB"), int)
def mask(im, c, tol=45):
    return ((np.abs(im[..., 0] - c[0]) < tol) & (np.abs(im[..., 1] - c[1]) < tol)
            & (np.abs(im[..., 2] - c[2]) < tol))
def iou(m1, m2):
    u = (m1 | m2).sum()
    return (m1 & m2).sum() / u if u else float("nan"), u
# interiors, legend regions excluded
for name, c, box in [("panel-a green curve", (44, 160, 44), (T + 2, B - 2, L1 + 2, R1 - 2)),
                     ("panel-a red floor",   (214, 39, 40), (T + 2, B - 2, L1 + 2, R1 - 2)),
                     ("panel-b purple pts",  (148, 103, 189), (T + 2, B - 2, L2 + 2, R2 - 2))]:
    t, b, l, r = box
    m1 = mask(A[t:b, l:r], c).copy(); m2 = mask(Bm[t:b, l:r], c).copy()
    # blank the legend rectangles (top-right of each panel in the original)
    if name.startswith("panel-a"):
        m1[:100, -300:] = False; m2[:100, -300:] = False
    else:
        m1[:60, :250] = False; m2[:60, :250] = False
    j, u = iou(m1, m2)
    print(f"{name:22s} IoU = {j:.4f}   union px = {u}   orig={m1.sum()} recon={m2.sum()}")
