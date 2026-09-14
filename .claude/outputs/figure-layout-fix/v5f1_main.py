"""V5 Fig 1 (main text) -- REDUCED money panel: the collapse, squarer and legible.

Layout change only.  Every plotted value is read from the same banked JSON the
banked figure used (outputs/t-lever-forgetting/s4b_jacobian.json,
outputs/v5-gate/e1c_vcurve.json); no measurement is performed here.

Printed box: FLF_W x FLF_H inches (defaults = the box this pass installs).
The main-text height budget is 1.42 in (measured: 1.44 in spills to 5 pp).
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
W_IN = float(os.environ.get("FLF_W", 3.190))
H_IN = float(os.environ.get("FLF_H", 1.400))

MU2_FLOOR = 1.7e-12
NORM = colors.LogNorm(vmin=1e-12, vmax=2.0)
CMAP = cm.viridis

des = json.load(open(os.path.join(TL, "s4b_jacobian.json")))
emg = json.load(open(os.path.join(VG, "e1c_vcurve.json")))

FIGSIZE, DPI, PX = canvas_wh(W_IN, H_IN)
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 8,
    "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
    "xtick.major.size": 2.6, "ytick.major.size": 2.6,
    "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
    "xtick.major.pad": 1.6, "ytick.major.pad": 1.6,
    "axes.labelpad": 1.4, "axes.titlepad": 2.6, "savefig.dpi": DPI,
})
fig = plt.figure(figsize=FIGSIZE)
# fractions chosen so the axes box fills the reduced canvas
ax = fig.add_axes([0.115, 0.215, 0.700, 0.645])
cax = fig.add_axes([0.838, 0.215, 0.024, 0.645])

seeds = sorted({r["seed"] for r in des})
for s in seeds:
    rows = sorted([r for r in des if r["seed"] == s and "gamma" in r], key=lambda r: r["gamma"])
    g = np.array([r["gamma"] for r in rows]); n = np.array([r["n_half_massive"] for r in rows])
    gc = rows[0]["gamma_crit"]; mu2 = rows[0]["mu_rad"] ** 2
    ax.loglog(g / gc, n / n.min(), "o-", ms=2.4, lw=0.9, color=CMAP(NORM(mu2)), zorder=3)
for tag, v in emg.items():
    if not tag.startswith("emergent"):
        continue
    rows = [r for r in v["rows"] if math.isfinite(r["n_half_jac"])]
    g = np.array([r["gamma"] for r in rows]); n = np.array([r["n_half_jac"] for r in rows])
    gc = v["gamma_crit_soft"]; mu2 = v["mu2_soft"]
    ax.loglog(g / gc, n / n.min(), "s--", ms=2.4, lw=0.9, color=CMAP(NORM(mu2)), zorder=4)

xs_lo = np.geomspace(0.02, 0.5, 20); xs_hi = np.geomspace(2.0, 30.0, 20)
ax.loglog(xs_lo, 1.0 / xs_lo, "k:", lw=1.1, zorder=6)
ax.loglog(xs_hi, xs_hi, "k:", lw=1.1, zorder=6)
ax.text(0.030, 3.4, "slope $-1$", fontsize=8, ha="left", va="center")
ax.text(26.0, 3.6, "slope $+1$", fontsize=8, ha="right", va="top")
ax.axvline(1.0, color="gray", lw=0.7, zorder=1)

ax.set_xlabel(r"$\gamma/\gamma_{\rm crit}$")
ax.set_ylabel(r"$n_{1/2}/n_{1/2}^{\min}$")
ax.set_xlim(0.018, 40); ax.set_ylim(0.8, 300)
ax.set_title("The damping optimum, collapsed")

# --- legend restored (2 entries, 8 pt) -------------------------------------
ax.plot([], [], "o-", color="0.35", ms=3.0, lw=0.9, label="designed, 5 seeds")
ax.plot([], [], "s--", color="0.35", ms=3.0, lw=0.9, label="emergent, 3 seeds")
leg = ax.legend(loc="upper center", bbox_to_anchor=(0.53, 1.00), fontsize=8,
                framealpha=0.92, borderpad=0.25, labelspacing=0.22,
                handlelength=1.5, handletextpad=0.45, borderaxespad=0.0)
leg.get_frame().set_linewidth(0.5)

sm = cm.ScalarMappable(norm=NORM, cmap=CMAP)
cb = fig.colorbar(sm, cax=cax)
cb.set_label(r"$\mu^2$", fontsize=8, labelpad=1.0)
cb.ax.axhline(MU2_FLOOR, color="crimson", lw=1.6)
cb.ax.tick_params(labelsize=7, pad=1.4, width=0.7, length=2.2)
cb.ax.set_yticks([1e-12, 1e-6, 1e0])
cb.outline.set_linewidth(0.7)

os.makedirs(OUTDIR, exist_ok=True)
out = os.path.join(OUTDIR, "fig1_damping_optimum.png")
fig.savefig(out, metadata={"Software": None})
print("wrote", out, PX, "printed", W_IN, "x", round(W_IN * PX[1] / PX[0], 4), "in")
