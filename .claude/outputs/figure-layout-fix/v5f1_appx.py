"""V5 NEW appendix figure: the complete, fully-annotated damping-optimum collapse.

This is the full-size companion to the reduced main-text Fig 1.  Same banked JSON,
same plotted values; it restores the annotations the main-text box cannot hold:
the mu->0 flat-coset callout, the colorbar's probe-floor annotation, the
gamma=gamma_crit marker label and the full two-family legend.

Printed box: FLF_W x FLF_H inches (default 4.950 x 2.600 = 0.90\linewidth).
"""
import json, math, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from size2 import canvas_wh

ROOT = os.path.abspath(".claude")
TL = os.path.join(ROOT, "outputs", "t-lever-forgetting")
VG = os.path.join(ROOT, "outputs", "v5-gate")
OUTDIR = os.environ.get("FRP_OUT", os.path.join(ROOT, "papers", "palm-variant", "v5", "figs"))
W_IN = float(os.environ.get("FLF_W", 4.950))
H_IN = float(os.environ.get("FLF_H", 2.900))

MU2_FLOOR = 1.7e-12
NORM = colors.LogNorm(vmin=1e-12, vmax=2.0)
CMAP = cm.viridis

des = json.load(open(os.path.join(TL, "s4b_jacobian.json")))
emg = json.load(open(os.path.join(VG, "e1c_vcurve.json")))

FIGSIZE, DPI, PX = canvas_wh(W_IN, H_IN)
plt.rcParams.update({
    "font.size": 8.5, "axes.labelsize": 9, "axes.titlesize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8.5,
    "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
    "xtick.major.size": 3.0, "ytick.major.size": 3.0,
    "xtick.minor.size": 1.8, "ytick.minor.size": 1.8,
    "xtick.major.pad": 2.0, "ytick.major.pad": 2.0,
    "axes.labelpad": 2.0, "axes.titlepad": 3.0, "savefig.dpi": DPI,
})
fig = plt.figure(figsize=FIGSIZE)
ax = fig.add_axes([0.086, 0.318, 0.635, 0.600])
cax = fig.add_axes([0.734, 0.318, 0.018, 0.600])

seeds = sorted({r["seed"] for r in des})
for s in seeds:
    rows = sorted([r for r in des if r["seed"] == s and "gamma" in r], key=lambda r: r["gamma"])
    g = np.array([r["gamma"] for r in rows]); n = np.array([r["n_half_massive"] for r in rows])
    gc = rows[0]["gamma_crit"]; mu2 = rows[0]["mu_rad"] ** 2
    ax.loglog(g / gc, n / n.min(), "o-", ms=3.4, lw=1.2, color=CMAP(NORM(mu2)), zorder=3)
for tag, v in emg.items():
    if not tag.startswith("emergent"):
        continue
    rows = [r for r in v["rows"] if math.isfinite(r["n_half_jac"])]
    g = np.array([r["gamma"] for r in rows]); n = np.array([r["n_half_jac"] for r in rows])
    gc = v["gamma_crit_soft"]; mu2 = v["mu2_soft"]
    ax.loglog(g / gc, n / n.min(), "s--", ms=3.4, lw=1.2, color=CMAP(NORM(mu2)), zorder=4)

xs_lo = np.geomspace(0.02, 0.5, 20); xs_hi = np.geomspace(2.0, 30.0, 20)
ax.loglog(xs_lo, 1.0 / xs_lo, "k:", lw=1.2, zorder=6)
ax.loglog(xs_hi, xs_hi, "k:", lw=1.2, zorder=6)
ax.text(0.020, 78.0, "slope $-1$\n(underdamped floor)", fontsize=8.5, ha="left", va="bottom")
ax.text(36.0, 1.9, "slope $+1$\n(overdamped, $\\propto\\mu^{-2}$)", fontsize=8.5,
        ha="right", va="center")
ax.axvline(1.0, color="gray", lw=0.9, zorder=1)
ax.text(1.15, 1.35, r"$\gamma=\gamma_{\rm crit}=2\varepsilon\mu$", fontsize=8.5, rotation=90,
        va="bottom", color="gray")

ax.annotate("designed flat coset: the $\\mu\\to0$ corner of this curve\n"
            "($\\gamma_{\\rm crit}\\to0$, overdamped at every $\\gamma>0$, $n_{1/2}=\\infty$)",
            xy=(34, 60.0), xytext=(0.022, 330), fontsize=8.5, ha="left", va="bottom",
            arrowprops=dict(arrowstyle="->", lw=1.0, color="k",
                            connectionstyle="arc3,rad=-0.18"),
            bbox=dict(fc="lightyellow", ec="gray", lw=0.6))

ax.set_xlabel(r"$\gamma/\gamma_{\rm crit}$")
ax.set_ylabel(r"$n_{1/2}/n_{1/2}^{\min}$")
ax.set_xlim(0.018, 40); ax.set_ylim(0.8, 1000)
ax.set_title("The damping optimum, collapsed: one curve, two families", fontsize=9.5, x=0.54)

ax.plot([], [], "o-", color="0.35", ms=4.0, lw=1.2,
        label="designed radial modes, 5 seeds (verification)")
ax.plot([], [], "s--", color="0.35", ms=4.0, lw=1.2,
        label="emergent (learned-MLP) coset, 3/3 seeds (evidence)")
h, l = ax.get_legend_handles_labels()
leg = fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.40, 0.012), ncol=1,
                 fontsize=8.5, framealpha=0.95, borderpad=0.35, columnspacing=1.4,
                 handlelength=2.0, handletextpad=0.5)
leg.get_frame().set_linewidth(0.6)

sm = cm.ScalarMappable(norm=NORM, cmap=CMAP)
cb = fig.colorbar(sm, cax=cax)
fig.text(0.962, 0.60, r"$\mu^2$ of the plotted mode", rotation=90, fontsize=9,
         ha="center", va="center")
cb.ax.axhline(MU2_FLOOR, color="crimson", lw=2.0)
cb.ax.tick_params(labelsize=8, pad=2.0, width=0.8, length=2.6)
fig.text(0.884, 0.60, "flat coset (probe floor, $1.7\\times10^{-12}$)", rotation=90,
         fontsize=8.5, color="crimson", ha="center", va="center")
cb.outline.set_linewidth(0.8)

os.makedirs(OUTDIR, exist_ok=True)
out = os.path.join(OUTDIR, "figA1_damping_optimum_full.png")
fig.savefig(out, metadata={"Software": None})
print("wrote", out, PX, "printed", W_IN, "x", round(W_IN * PX[1] / PX[0], 4), "in")
