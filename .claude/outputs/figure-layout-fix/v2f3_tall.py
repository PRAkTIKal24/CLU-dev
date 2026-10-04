"""V2 Fig 3 (appendix) -- RE-LAID-OUT TALLER so its in-axes notes reach the >=8 pt
target and the four legends deleted by the render pass come back, including the
panel-(c) "max dev" number.

Layout change only.  Values read from outputs/f1-gmor-condensate/{gmor_condensate,
angular_tilt_contrast}.npz, the same arrays the banked figure used
(source: scratch/f1-gmor-condensate/analyze_and_figure.py, figure block).
Internal task ID "F-1" and the per-checkpoint short-tags stay removed.

Printed box: FLF_W x FLF_H inches (default 4.400 x 3.600 = 0.80\textwidth).
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from size2 import canvas_wh

OUT = ".claude/outputs/f1-gmor-condensate"
OUTDIR = os.environ.get("FRP_OUT", os.path.join(".claude", "papers", "neurreps-variants", "v2", "figs"))
W_IN = float(os.environ.get("FLF_W", 4.400))
H_IN = float(os.environ.get("FLF_H", 3.600))

DELTAS = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1]
EPS = float(np.finfo(np.float64).eps)
d = np.load(os.path.join(OUT, "gmor_condensate.npz"), allow_pickle=True)
tr = dict(np.load(os.path.join(OUT, "angular_tilt_contrast.npz"), allow_pickle=True))
tags = sorted(set(d["tag"].tolist()))

FIGSIZE, DPI, PX = canvas_wh(W_IN, H_IN)
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 8,
    "axes.linewidth": 0.6, "grid.linewidth": 0.35,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "xtick.major.size": 2.2, "ytick.major.size": 2.2,
    "xtick.minor.size": 1.2, "ytick.minor.size": 1.2,
    "xtick.major.pad": 1.2, "ytick.major.pad": 1.2,
    "axes.labelpad": 1.2, "axes.titlepad": 2.4, "savefig.dpi": DPI,
})
fig, axes = plt.subplots(2, 2, figsize=FIGSIZE, layout="constrained")
fig.get_layout_engine().set(w_pad=0.010, h_pad=0.008, wspace=0.055, hspace=0.065)
colors = plt.cm.viridis(np.linspace(0, 0.88, len(tags)))
ANCH = {t: (t.startswith("anchored")) for t in tags}

def small_legend(a, handles, **kw):
    lg = a.legend(handles=handles, framealpha=0.92, borderpad=0.24, labelspacing=0.2,
                  handlelength=1.5, handletextpad=0.4, borderaxespad=0.25, **kw)
    lg.get_frame().set_linewidth(0.5)
    lg.set_in_layout(False)
    return lg

# (a) the collapse
ax = axes[0, 0]
for c, tag in zip(colors, tags):
    m = d["tag"] == tag
    ax.loglog(d["rhs"][m], d["lhs"][m], "o" if ANCH[tag] else "s", ms=3.2,
              color=c, mfc="none" if ANCH[tag] else c, mew=0.9)
lo, hi = d["rhs"].min(), d["rhs"].max()
ax.loglog([lo, hi], [lo, hi], "k-", lw=0.9, zorder=0)
ax.set_xlabel(r"$\delta\,\Sigma$")
ax.set_ylabel(r"$\mu^2 F^2$")
ax.set_title("(a) GMOR collapse")
ax.set_xticks([1e-8, 1e-5, 1e-2]); ax.set_yticks([1e-8, 1e-5, 1e-2])
ax.set_ylim(top=ax.get_ylim()[1] * 60.0)
ax.grid(alpha=0.25, which="major")
small_legend(ax, [
    Line2D([], [], color="0.35", lw=0, marker="o", ms=3.2, mfc="none", mew=0.9,
           label="anchored, 3000 ep"),
    Line2D([], [], color="0.35", lw=0, marker="s", ms=3.2, mew=0.9, label="designed, 150 ep"),
    Line2D([], [], color="k", lw=0.9, label=r"$\mu^2F^2=\delta\Sigma$")],
    loc="upper left", ncol=1)

# (b) deviations + the eps/delta floor
ax = axes[0, 1]
for c, tag in zip(colors, tags):
    m = d["tag"] == tag
    ax.loglog(d["delta"][m], np.maximum(d["rel_dev"][m], 1e-17), ".-", color=c, ms=3.0, lw=0.8)
dd = np.array(DELTAS)
ax.loglog(dd, 2.2e-16 / dd * np.median(d["mu_rad_sq_0"]) * np.median(d["M_ch"]),
          "k--", lw=1.0)
ax.loglog(dd, [np.abs(d["abs_dev"][d["delta"] == x]).max() for x in dd], "r^-", ms=3.2, lw=1.0)
ax.axhline(EPS, color="gray", ls=":", lw=0.9)
ax.set_xlabel(r"spurion $\delta$")
ax.set_ylabel("deviation from GMOR")
ax.set_title("(b) deviation")
ax.set_xticks([1e-8, 1e-5, 1e-2]); ax.set_yticks([1e-16, 1e-12, 1e-8])
ax.set_ylim(1e-18, 1e-4)
ax.grid(alpha=0.25, which="major")
small_legend(ax, [
    Line2D([], [], color="k", lw=1.0, ls="--", label=r"roundoff floor $\propto\varepsilon/\delta$"),
    Line2D([], [], color="r", lw=1.0, marker="^", ms=3.2, label="max abs. deviation"),
    Line2D([], [], color="gray", lw=0.9, ls=":",
           label=r"$\varepsilon_{64}=2.2\times10^{-16}$")],
    loc="upper right", ncol=1)

# (c) the running condensate
ax = axes[1, 0]
for c, tag in zip(colors, tags):
    m = d["tag"] == tag
    ax.semilogx(d["delta"][m], d["r_star"][m] / d["f"][m], "o-", ms=2.8, color=c, lw=0.9)
ax.semilogx(dd, np.ones_like(dd), "k--", lw=1.2)
ax.set_xlabel(r"spurion $\delta$")
ax.set_ylabel(r"$\Sigma(\delta)/\Sigma(0)$")
ax.set_title("(c) running condensate")
ax.set_xticks([1e-8, 1e-5, 1e-2])
ax.grid(alpha=0.25, which="major")
small_legend(ax, [Line2D([], [], color="k", lw=1.2, ls="--",
                         label="angular tilt: $\\Sigma\\equiv f$\n(max dev $%.0e$)"
                               % np.abs(tr["r_tilt"] - tr["f"]).max())],
             loc="upper left", ncol=1)

# (d) NLO: resonance saturation of the leading LEC
ax = axes[1, 1]
for c, tag in zip(colors, tags):
    m = (d["tag"] == tag) & (d["lec_pred_lo"] < 0.25)
    ax.plot(d["lec_pred_lo"][m], d["lec_ratio_geom"][m], "o-", ms=2.8, color=c, lw=0.9)
ax.axhline(1.0, color="k", ls="--", lw=1.0)
ax.set_xscale("log")
ax.set_xlabel(r"$x=\delta/(M_{ch}\mu_{rad}^2 f)$")
ax.set_ylabel("LEC ratio")
ax.set_title("(d) LEC saturation")
ax.set_ylim(0.85, 1.08)
ax.set_xticks([1e-8, 1e-5, 1e-2]); ax.set_yticks([0.9, 1.0])
ax.grid(alpha=0.25, which="major")
small_legend(ax, [Line2D([], [], color="k", lw=1.0, ls="--",
                         label="resonance saturation\n(exact as $x\\to0$)")],
             loc="lower left", ncol=1)

os.makedirs(OUTDIR, exist_ok=True)
out = os.path.join(OUTDIR, "fig3_gmor_condensate.png")
fig.savefig(out, metadata={"Software": None})
print("wrote", out, PX, "printed", W_IN, "x", round(W_IN * PX[1] / PX[0], 4), "in")
