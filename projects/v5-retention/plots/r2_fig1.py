"""Round-2 restyle of V5 Fig 1 (main text) and, with R2_FULL=1, its appendix
companion figA1. Data calls are those of outputs/figure-layout-fix/v5f1_main.py /
v5f1_appx.py, unchanged (same JSON, same arrays, same call multiset).
Run from the repo root."""
import json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style as S
import matplotlib.pyplot as plt
from matplotlib import cm, colors

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "..", "data")
TL = os.path.join(_DATA, "t-lever-forgetting")
VG = os.path.join(_DATA, "v5-gate")
OUTDIR = os.environ.get("FRP_OUT", os.path.join(_HERE, "renders"))
FULL = os.environ.get("R2_FULL") == "1"

DPI = 400.0
if FULL:   # figA1: 1980 x 1160 px, printed at \linewidth = 5.5 in
    PX = (1980, 1160); PRINTED_W = 5.5; NAME = "figA1_damping_optimum_full.png"
else:      # fig1: 1276 x 560 px, printed at 0.6\linewidth = 3.3 in
    PX = (1276, 560); PRINTED_W = 3.3; NAME = "fig1_damping_optimum.png"
FIGSIZE = (PX[0] / DPI, PX[1] / DPI)
S.setup(FIGSIZE[0], PRINTED_W)

MU2_FLOOR = 1.7e-12
NORM = colors.LogNorm(vmin=1e-12, vmax=2.0)
CMAP = cm.viridis

des = json.load(open(os.path.join(TL, "s4b_jacobian.json")))
emg = json.load(open(os.path.join(VG, "e1c_vcurve.json")))

fig = plt.figure(figsize=FIGSIZE)
if FULL:
    ax = fig.add_axes([0.085, 0.115, 0.745, 0.865])
    cax = fig.add_axes([0.850, 0.115, 0.022, 0.865])
else:
    ax = fig.add_axes([0.125, 0.225, 0.700, 0.745])
    cax = fig.add_axes([0.845, 0.225, 0.026, 0.745])

ms = S.pt(2.6); lw = S.pt(1.0)
seeds = sorted({r["seed"] for r in des})
for s in seeds:
    rows = sorted([r for r in des if r["seed"] == s and "gamma" in r], key=lambda r: r["gamma"])
    g = np.array([r["gamma"] for r in rows]); n = np.array([r["n_half_massive"] for r in rows])
    gc = rows[0]["gamma_crit"]; mu2 = rows[0]["mu_rad"] ** 2
    ax.loglog(g / gc, n / n.min(), "o-", ms=ms, lw=lw, color=CMAP(NORM(mu2)), zorder=3)
for tag, v in emg.items():
    if not tag.startswith("emergent"):
        continue
    rows = [r for r in v["rows"] if math.isfinite(r["n_half_jac"])]
    g = np.array([r["gamma"] for r in rows]); n = np.array([r["n_half_jac"] for r in rows])
    gc = v["gamma_crit_soft"]; mu2 = v["mu2_soft"]
    ax.loglog(g / gc, n / n.min(), "s--", ms=ms, lw=lw, color=CMAP(NORM(mu2)), zorder=4)

xs_lo = np.geomspace(0.02, 0.5, 20); xs_hi = np.geomspace(2.0, 30.0, 20)
ax.loglog(xs_lo, 1.0 / xs_lo, ":", color="k", lw=S.pt(1.0), zorder=6)
ax.loglog(xs_hi, xs_hi, ":", color="k", lw=S.pt(1.0), zorder=6)
ax.axvline(1.0, color=S.GREY, lw=S.pt(0.6), ls=(0, (1.5, 1.5)), zorder=1)

ax.set_xlabel(r"$\gamma/\gamma_{\rm crit}$", labelpad=S.pt(1.0))
ax.set_ylabel(r"$n_{1/2}/n_{1/2}^{\min}$", labelpad=S.pt(1.5))
ax.set_xlim(0.018, 40); ax.set_ylim(0.8, 300)

# legend (two families): the same two empty proxy calls as the original generator
ax.plot([], [], "o-", color="0.35", ms=ms, lw=lw, label="designed")
ax.plot([], [], "s--", color="0.35", ms=ms, lw=lw, label="emergent")
ax.legend(loc="upper left", bbox_to_anchor=(0.19, 1.02), handlelength=2.2)

sm = cm.ScalarMappable(norm=NORM, cmap=CMAP)
cb = fig.colorbar(sm, cax=cax)
cb.set_label(r"$\mu^2$", labelpad=S.pt(1.0))
cb.ax.axhline(MU2_FLOOR, color=S.OI[1], lw=S.pt(1.6))
cb.ax.tick_params(which="both", width=S.pt(0.6), length=S.pt(2.0), pad=S.pt(1.5))
cb.ax.tick_params(which="minor", length=0)
cb.ax.set_yticks([1e-12, 1e-6, 1e0] if not FULL else [1e-12, 1e-9, 1e-6, 1e-3, 1e0])
cb.outline.set_linewidth(S.pt(0.5))

os.makedirs(OUTDIR, exist_ok=True)
out = os.path.join(OUTDIR, NAME)
S.save(fig, out, DPI)
print("wrote", out)
