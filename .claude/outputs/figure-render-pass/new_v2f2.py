"""V2 variant Fig 2 (fig2_anchor_cure_laws.png) re-rendered at its FINAL PRINTED SIZE.

Printed footprint measured from the built PDF: 340.568 x 130.035 pt = 4.730 x 1.806 in
(0.86\linewidth).  Aspect preserved to the banked figure's 1430/546 = 2.619048.

NO generating script survived for this figure; the data are re-derived from the banked
npz that produced it (outputs/v2-referee-experiments/sf3_{gmor,ep}_sweep.npz) and the
re-derivation is validated against the banked PNG in sf3_verify_geom.py (median
per-column curve offset 0.37 px panel (a) / 0.70 px panel (b), and the three fitted
annotations on the banked figure -- slope -0.961, floor 27.03, slope 0.516 --
reproduced as -0.96062, 27.02681, 0.51645).
Internal report labels SF-3a / SF-3b removed from the panel titles.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from size import canvas
from sf3_reconstruct import xs, ys, sl, ic, FLOOR, xb, yb, slb, icb, FIT

OUTDIR = os.environ.get("FRP_OUT", os.path.join(".claude", "papers", "neurreps-variants", "v2", "figs"))
FIGSIZE, DPI, PX = canvas(1430, 546, 4.730, 400.0)
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 8,
    "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
    "xtick.major.size": 2.4, "ytick.major.size": 2.4,
    "xtick.minor.size": 1.4, "ytick.minor.size": 1.4,
    "xtick.major.pad": 1.4, "ytick.major.pad": 1.4,
    "axes.labelpad": 1.4, "axes.titlepad": 2.6, "savefig.dpi": DPI,
})
fig, ax = plt.subplots(1, 2, figsize=FIGSIZE, layout="constrained")
fig.get_layout_engine().set(w_pad=0.012, h_pad=0.008, wspace=0.05)

ax[0].loglog(xs, ys, "o-", color="tab:green", ms=3.4, lw=1.3,
             label="anchored 3000-ep $n_{1/2}$")
ax[0].loglog(xs[:FIT], 10 ** (ic + sl * np.log10(xs[:FIT])), "--", color="0.4", lw=1.4,
             label="slope %.3f (pred $-1$)" % sl)
ax[0].axhline(FLOOR, color="red", ls=":", lw=1.4, label="floor %.2f" % FLOOR)
ax[0].set_xlabel(r"GMOR tilt $\delta$")
ax[0].set_ylabel(r"retention $n_{1/2}$ (steps)")
ax[0].set_title("GMOR retention at 3000 ep")
ax[0].text(0.975, 0.95, "slope $%.3f$ (pred $-1$)" % sl, transform=ax[0].transAxes,
           fontsize=8, color="0.35", ha="right", va="top")
ax[0].text(0.975, 0.845, "floor $%.2f$" % FLOOR, transform=ax[0].transAxes,
           fontsize=8, color="red", ha="right", va="top")

ax[1].loglog(xb, yb, "o", color="tab:purple", ms=3.4,
             label=r"$\varphi$ (Jacobian), above EP")
ax[1].loglog(xb, 10 ** (icb + slb * np.log10(xb)), "--", color="0.4", lw=1.4,
             label="slope %.3f (pred $0.5$)" % slb)
ax[1].set_xlabel(r"$h-h^*$")
ax[1].set_ylabel(r"onset frequency $\varphi=|\arg\lambda|$")
ax[1].set_title(r"EP onset at 3000 ep")
ax[1].text(0.975, 0.05, "slope $%.3f$ (pred $0.5$)" % slb, transform=ax[1].transAxes,
           fontsize=8, color="0.35", ha="right", va="bottom")

os.makedirs(OUTDIR, exist_ok=True)
out = os.path.join(OUTDIR, "fig2_anchor_cure_laws.png")
fig.savefig(out, metadata={"Software": None})
print("wrote", out, plt.imread(out).shape)
