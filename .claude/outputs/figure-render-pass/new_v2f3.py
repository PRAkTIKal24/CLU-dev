"""V2 variant Fig 3 (fig3_gmor_condensate.png) re-rendered at its FINAL PRINTED SIZE.

Printed footprint measured from the built PDF: 237.598 x 180.247 pt = 3.300 x 2.503 in
(0.60\textwidth).  Aspect preserved to the banked figure's 1856/1408 = 1.318182.

Replot only: values read from outputs/f1-gmor-condensate/{gmor_condensate,
angular_tilt_contrast}.npz, the same arrays the banked figure used
(source: scratch/f1-gmor-condensate/analyze_and_figure.py, figure block).
Internal task ID "F-1" (suptitle) and the checkpoint short-tags in the panel-(a)
legend ("a3000 s42" ... "d150 s46") are removed.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from size import canvas

OUT = ".claude/outputs/f1-gmor-condensate"
OUTDIR = os.environ.get("FRP_OUT", os.path.join(".claude", "papers", "neurreps-variants", "v2", "figs"))
DELTAS = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1]
EPS = float(np.finfo(np.float64).eps)
d = np.load(os.path.join(OUT, "gmor_condensate.npz"), allow_pickle=True)
tr = dict(np.load(os.path.join(OUT, "angular_tilt_contrast.npz"), allow_pickle=True))
tags = sorted(set(d["tag"].tolist()))

FIGSIZE, DPI, PX = canvas(1856, 1408, 3.300, 400.0)
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.6, "grid.linewidth": 0.35,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "xtick.major.size": 2.2, "ytick.major.size": 2.2,
    "xtick.minor.size": 1.2, "ytick.minor.size": 1.2,
    "xtick.major.pad": 1.2, "ytick.major.pad": 1.2,
    "axes.labelpad": 1.2, "axes.titlepad": 2.2, "savefig.dpi": DPI,
})
fig, axes = plt.subplots(2, 2, figsize=FIGSIZE, layout="constrained")
fig.get_layout_engine().set(w_pad=0.008, h_pad=0.008, wspace=0.045, hspace=0.055)
colors = plt.cm.viridis(np.linspace(0, 0.88, len(tags)))
ANCH = {t: (t.startswith("anchored")) for t in tags}

# (a) the collapse
ax = axes[0, 0]
for c, tag in zip(colors, tags):
    m = d["tag"] == tag
    ax.loglog(d["rhs"][m], d["lhs"][m], "o" if ANCH[tag] else "s", ms=2.8,
              color=c, mfc="none" if ANCH[tag] else c, mew=0.8)
lo, hi = d["rhs"].min(), d["rhs"].max()
ax.loglog([lo, hi], [lo, hi], "k-", lw=0.9, zorder=0)
ax.set_xlabel(r"$\delta\,\Sigma$")
ax.set_ylabel(r"$\mu^2 F^2$")
ax.set_title("(a) GMOR collapse")
ax.set_xticks([1e-8, 1e-5, 1e-2]); ax.set_yticks([1e-8, 1e-5, 1e-2])
ax.text(0.96, 0.06, r"$\mu^2F^2=\delta\Sigma$", transform=ax.transAxes, fontsize=7.5,
        ha="right", va="bottom")
ax.grid(alpha=0.25, which="major")

# (b) deviations + the eps/delta floor
ax = axes[0, 1]
for c, tag in zip(colors, tags):
    m = d["tag"] == tag
    ax.loglog(d["delta"][m], np.maximum(d["rel_dev"][m], 1e-17), ".-", color=c, ms=2.6, lw=0.7)
dd = np.array(DELTAS)
ax.loglog(dd, 2.2e-16 / dd * np.median(d["mu_rad_sq_0"]) * np.median(d["M_ch"]),
          "k--", lw=1.0)
ax.loglog(dd, [np.abs(d["abs_dev"][d["delta"] == x]).max() for x in dd], "r^-", ms=3.0, lw=1.0)
ax.axhline(EPS, color="gray", ls=":", lw=0.9)
ax.set_xlabel(r"spurion $\delta$")
ax.set_ylabel("deviation")
ax.set_title("(b) deviation")
ax.set_xticks([1e-8, 1e-5, 1e-2]); ax.set_yticks([1e-16, 1e-12, 1e-8])
ax.text(0.97, 0.95, "roundoff floor", transform=ax.transAxes,
        fontsize=7, ha="right", va="top")
ax.text(0.30, 0.015, "abs. dev.", transform=ax.transAxes, fontsize=7,
        color="red", ha="left", va="bottom")
ax.grid(alpha=0.25, which="major")

# (c) the running condensate
ax = axes[1, 0]
for c, tag in zip(colors, tags):
    m = d["tag"] == tag
    ax.semilogx(d["delta"][m], d["r_star"][m] / d["f"][m], "o-", ms=2.4, color=c, lw=0.9)
ax.semilogx(dd, np.ones_like(dd), "k--", lw=1.2)
ax.set_xlabel(r"spurion $\delta$")
ax.set_ylabel(r"$\Sigma(\delta)/\Sigma(0)$")
ax.set_title("(c) running condensate")
ax.set_xticks([1e-8, 1e-5, 1e-2])
ax.text(0.04, 0.94, r"angular tilt: $\Sigma\equiv f$", transform=ax.transAxes,
        fontsize=7, ha="left", va="top")
ax.grid(alpha=0.25, which="major")

# (d) NLO: resonance saturation of the leading LEC
ax = axes[1, 1]
for c, tag in zip(colors, tags):
    m = (d["tag"] == tag) & (d["lec_pred_lo"] < 0.25)
    ax.plot(d["lec_pred_lo"][m], d["lec_ratio_geom"][m], "o-", ms=2.4, color=c, lw=0.9)
ax.axhline(1.0, color="k", ls="--", lw=1.0)
ax.set_xscale("log")
ax.set_xlabel(r"$x=\delta/(M_{ch}\mu_{rad}^2 f)$")
ax.set_ylabel("LEC ratio")
ax.set_title("(d) LEC saturation")
ax.set_ylim(0.85, 1.05)
ax.set_xticks([1e-8, 1e-5, 1e-2]); ax.set_yticks([0.9, 1.0])
ax.text(0.04, 0.20, r"exact as $x\to0$", transform=ax.transAxes, fontsize=7,
        ha="left", va="bottom")
ax.grid(alpha=0.25, which="major")

os.makedirs(OUTDIR, exist_ok=True)
out = os.path.join(OUTDIR, "fig3_gmor_condensate.png")
fig.savefig(out, metadata={"Software": None})
print("wrote", out, plt.imread(out).shape)
