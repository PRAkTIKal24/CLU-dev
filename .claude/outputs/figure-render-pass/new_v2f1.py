"""V2 variant Fig 1 (fig1_mo_headtohead.png) re-rendered at its FINAL PRINTED SIZE.

Printed footprint measured from the built PDF: 269.283 x 147.289 pt = 3.740 x 2.046 in
(0.68\linewidth).  Aspect preserved to the banked figure's 1022/559 = 1.828265.

Replot only: values read from outputs/v2-full-runs/gmor_sweep.npz, the same file the
banked figure used (source: scratch/v2-full-runs/make_figures.py::fig2_mo).
Seed short-tags in the legend are collapsed to one neutral entry; the per-seed curves
and their colours are unchanged.  Seed order in the file is 42,43,44,45,46 -> 1..5.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from size import canvas

OUT = os.path.abspath(os.path.join(".claude", "outputs", "v2-full-runs"))
OUTDIR = os.environ.get("FRP_OUT", os.path.join(".claude", "papers", "neurreps-variants", "v2", "figs"))
FIGSIZE, DPI, PX = canvas(1022, 559, 3.740, 550.0)
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 8,
    "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
    "xtick.major.size": 2.6, "ytick.major.size": 2.6,
    "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
    "xtick.major.pad": 1.6, "ytick.major.pad": 1.6,
    "axes.labelpad": 1.6, "axes.titlepad": 3.0, "savefig.dpi": DPI,
})
d = np.load(os.path.join(OUT, "gmor_sweep.npz"))
fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
fig.get_layout_engine().set(w_pad=0.012, h_pad=0.010)
hh = d["h"] / d["h_star"]
unc = ~d["mo_censored"]
for s in sorted(set(d["seed"].astype(int))):
    m = unc & (d["seed"] == s)
    ax.plot(hh[m], d["mo_ratio"][m], "o-", ms=2.6, lw=1.0, alpha=0.8)
ax.axhline(1.0, color="k", lw=0.9)
ax.axhspan(0.98, 1.05, color="tab:green", alpha=0.18,
           label="Mo's median 1.013")
ax.axvline(1.0, color="tab:purple", ls="--", lw=1.1, label="EP")
ax.plot([], [], "o-", color="0.35", ms=3.2, lw=1.0, label="5 trained models")
ax.set_xscale("log")
ax.set_xlabel(r"$h/h^*(\gamma)$   ($h=\varepsilon\mu$, measured tilted Hessian)")
ax.set_ylabel("lifetime: measured / predicted")
ax.set_title("Mo's lifetime protocol on trained CLUs")
handles, labels = ax.get_legend_handles_labels()
order = [2, 0, 1]
ax.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper left",
          handlelength=1.7, handletextpad=0.55, borderpad=0.35, labelspacing=0.3,
          borderaxespad=0.3, framealpha=0.95)
ax.annotate("censored ($\\delta\\leq3\\times10^{-4}$): 10/70 runs,\n"
            "same pattern as Mo's own $\\epsilon=10^{-4}$ row",
            xy=(0.025, 0.045), xycoords="axes fraction", fontsize=7, color="0.35",
            linespacing=1.15)
os.makedirs(OUTDIR, exist_ok=True)
out = os.path.join(OUTDIR, "fig1_mo_headtohead.png")
fig.savefig(out, metadata={"Software": None})
print("wrote", out, plt.imread(out).shape)
