"""V5 variant Fig 2 (fig2_two_instruments.png) re-rendered at its FINAL PRINTED SIZE.

Printed footprint measured from the built PDF: 293.036 x 82.050 pt = 4.070 x 1.140 in
(0.74\linewidth).  Aspect preserved to the banked figure's 2400/672 = 3.571429.

Replot only: values read from outputs/v5-vcurve-validation/m1_emergent.json, the same
file the banked figure used (source script scratch/v5-vcurve-validation/a1_analyse.py).
Internal instrument IDs (I-J / I-R1 / I-R3, Gamma_jac/Gamma_R3, n_jac/n_R1) and seed
short-tags (s42/s43/s44) are removed; seeds map 42->1, 43->2, 44->3 in file order.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.abspath(os.path.join(".claude", "outputs", "v5-vcurve-validation"))
OUTDIR = os.environ.get("FRP_OUT", os.path.join(".claude", "papers", "palm-variant", "v5", "figs"))
E = json.load(open(os.path.join(OUT, "m1_emergent.json")))
SEEDS = ["emergent150_s42", "emergent150_s43", "emergent150_s44"]
DEL = "0.05"

import sys as _sys; _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from size import canvas
FIGSIZE, DPI, PX = canvas(2400, 672, 4.070, 400.0)
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.6, "grid.linewidth": 0.4,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "xtick.major.size": 2.2, "ytick.major.size": 2.2,
    "xtick.minor.size": 1.2, "ytick.minor.size": 1.2,
    "xtick.major.pad": 1.2, "ytick.major.pad": 1.2,
    "axes.labelpad": 1.0, "axes.titlepad": 2.2, "savefig.dpi": DPI,
    "lines.solid_capstyle": "round",
})
fig, ax = plt.subplots(1, 3, figsize=FIGSIZE, layout="constrained")
fig.get_layout_engine().set(w_pad=0.008, h_pad=0.004, wspace=0.045, hspace=0.0)
cols = ["C0", "C1", "C2"]
for i, t in enumerate(SEEDS):
    r = E["runs"][t]; gc = r["gamma_crit_soft"]; rows = r["deltas"][DEL]["rows"]
    g = np.array([v["gamma"] for v in rows])
    ax[0].loglog(g, [v["n_half_jac"] for v in rows], "-", color=cols[i], lw=1.4)
    ax[0].loglog(g, [v["n_half_R3"] for v in rows], "o", color=cols[i], ms=1.9, mfc="none", mew=0.6)
    ax[0].loglog(g, [v["n_half_R1"] for v in rows], ":", color=cols[i], lw=1.1)
    ax[0].axvline(gc, color=cols[i], ls="--", lw=0.6, alpha=0.55)
    ax[1].semilogx(g, [v["ratio_Gam"] for v in rows], "-", color=cols[i], lw=1.3)
    ax[1].semilogx(g, [v["ratio_jac_R1"] for v in rows], ":", color=cols[i], lw=1.2)
    ax[2].loglog(np.array([v["gamma"] for v in rows]) / gc,
                 np.array([v["n_half_jac"] for v in rows]) * gc, "-", color=cols[i], lw=1.4)
    ax[2].loglog(np.array([v["gamma"] for v in rows]) / gc,
                 np.array([v["n_half_R3"] for v in rows]) * gc, "o", color=cols[i], ms=1.9,
                 mfc="none", mew=0.6)
ax[0].set_xlabel(r"$\gamma$"); ax[0].set_ylabel(r"$n_{1/2}$ (steps)")
ax[0].set_title(r"(a) $T=0$ V-curve")
ax[0].grid(alpha=.3, which="major")
ax[1].axhline(1.0, color="k", lw=.7); ax[1].set_ylim(0, 2.4)
ax[1].set_yticks([0, 0.5, 1.0, 1.5, 2.0])
ax[1].set_xlabel(r"$\gamma$"); ax[1].set_ylabel("ratio")
ax[1].set_title("(b) instrument ratio")
ax[1].grid(alpha=.3, which="major")
ax[1].text(0.30, 0.80, "decay rate", fontsize=7, ha="center", va="top")
ax[1].text(0.20, 2.02, "threshold", fontsize=7, ha="center", va="bottom")
ax[2].set_xlabel(r"$\gamma/\gamma_{\rm crit}$")
ax[2].set_ylabel(r"$n_{1/2}\,\gamma_{\rm crit}$")
ax[2].set_title("(c) collapsed")
ax[2].grid(alpha=.3, which="major")

os.makedirs(OUTDIR, exist_ok=True)
out = os.path.join(OUTDIR, "fig2_two_instruments.png")
fig.savefig(out, metadata={"Software": None})
print("wrote", out, plt.imread(out).shape)
