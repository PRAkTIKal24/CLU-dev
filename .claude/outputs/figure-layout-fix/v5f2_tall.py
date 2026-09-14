"""V5 Fig 2 (appendix) -- RE-LAID-OUT TALLER so the type targets are reachable
and the legends deleted by the render pass can be restored.

Layout change only.  Values read from outputs/v5-vcurve-validation/m1_emergent.json,
the same file the banked figure used (source scratch/v5-vcurve-validation/a1_analyse.py).
Internal instrument IDs (I-J / I-R1 / I-R3, Gamma_jac/Gamma_R3, n_jac/n_R1) and seed
short-tags stay removed; seeds map 42->1, 43->2, 44->3 in file order.

Printed box: FLF_W x FLF_H inches (default 4.950 x 2.250 = 0.90\linewidth).
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from size2 import canvas_wh

OUT = os.path.abspath(os.path.join(".claude", "outputs", "v5-vcurve-validation"))
OUTDIR = os.environ.get("FRP_OUT", os.path.join(".claude", "papers", "palm-variant", "v5", "figs"))
W_IN = float(os.environ.get("FLF_W", 4.950))
H_IN = float(os.environ.get("FLF_H", 2.250))

E = json.load(open(os.path.join(OUT, "m1_emergent.json")))
SEEDS = ["emergent150_s42", "emergent150_s43", "emergent150_s44"]
DEL = "0.05"

FIGSIZE, DPI, PX = canvas_wh(W_IN, H_IN)
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 8,
    "axes.linewidth": 0.6, "grid.linewidth": 0.4,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "xtick.major.size": 2.2, "ytick.major.size": 2.2,
    "xtick.minor.size": 1.2, "ytick.minor.size": 1.2,
    "xtick.major.pad": 1.4, "ytick.major.pad": 1.4,
    "axes.labelpad": 1.2, "axes.titlepad": 2.4, "savefig.dpi": DPI,
    "lines.solid_capstyle": "round",
})
fig, ax = plt.subplots(1, 3, figsize=FIGSIZE, layout="constrained")
fig.get_layout_engine().set(w_pad=0.010, h_pad=0.006, wspace=0.055, hspace=0.0)
cols = ["C0", "C1", "C2"]
for i, t in enumerate(SEEDS):
    r = E["runs"][t]; gc = r["gamma_crit_soft"]; rows = r["deltas"][DEL]["rows"]
    g = np.array([v["gamma"] for v in rows])
    ax[0].loglog(g, [v["n_half_jac"] for v in rows], "-", color=cols[i], lw=1.4)
    ax[0].loglog(g, [v["n_half_R3"] for v in rows], "o", color=cols[i], ms=2.4, mfc="none", mew=0.7)
    ax[0].loglog(g, [v["n_half_R1"] for v in rows], ":", color=cols[i], lw=1.2)
    ax[0].axvline(gc, color=cols[i], ls="--", lw=0.6, alpha=0.55)
    ax[1].semilogx(g, [v["ratio_Gam"] for v in rows], "-", color=cols[i], lw=1.3)
    ax[1].semilogx(g, [v["ratio_jac_R1"] for v in rows], ":", color=cols[i], lw=1.2)
    ax[2].loglog(np.array([v["gamma"] for v in rows]) / gc,
                 np.array([v["n_half_jac"] for v in rows]) * gc, "-", color=cols[i], lw=1.4)
    ax[2].loglog(np.array([v["gamma"] for v in rows]) / gc,
                 np.array([v["n_half_R3"] for v in rows]) * gc, "o", color=cols[i], ms=2.4,
                 mfc="none", mew=0.7)

ax[0].set_xlabel(r"$\gamma$"); ax[0].set_ylabel(r"$n_{1/2}$ (steps)")
ax[0].set_title(r"(a) $T=0$ V-curve")
ax[0].grid(alpha=.3, which="major")
inst = [Line2D([], [], color="0.35", lw=1.4, ls="-", label="Jacobian"),
        Line2D([], [], color="0.35", lw=0, marker="o", ms=2.8, mfc="none", mew=0.7,
               label="rollout rate"),
        Line2D([], [], color="0.35", lw=1.2, ls=":", label="rollout threshold")]
l0 = ax[0].legend(handles=inst, loc="upper left", bbox_to_anchor=(0.005, 1.005),
                  fontsize=8, framealpha=0.92, borderpad=0.24, labelspacing=0.2,
                  handlelength=1.5, handletextpad=0.4, borderaxespad=0.0)
l0.get_frame().set_linewidth(0.5)

ax[1].axhline(1.0, color="k", lw=.7); ax[1].set_ylim(0, 2.6)
ax[1].set_yticks([0, 0.5, 1.0, 1.5, 2.0, 2.5])
ax[1].set_xlabel(r"$\gamma$"); ax[1].set_ylabel("instrument ratio")
ax[1].set_title("(b) instrument ratio")
ax[1].grid(alpha=.3, which="major")
rat = [Line2D([], [], color="0.35", lw=1.3, ls="-", label="decay-rate ratio"),
       Line2D([], [], color="0.35", lw=1.2, ls=":", label="threshold ratio")]
l1 = ax[1].legend(handles=rat, loc="lower center", bbox_to_anchor=(0.5, -0.012),
                  fontsize=8, framealpha=0.92, borderpad=0.24, labelspacing=0.2,
                  handlelength=1.5, handletextpad=0.4, borderaxespad=0.0)
l1.get_frame().set_linewidth(0.5)

ax[2].set_xlabel(r"$\gamma/\gamma_{\rm crit}$")
ax[2].set_ylabel(r"$n_{1/2}\,\gamma_{\rm crit}$")
ax[2].set_title("(c) collapsed")
ax[2].grid(alpha=.3, which="major")
sd = [Line2D([], [], color=cols[i], lw=1.4, label=f"seed {i+1}") for i in range(3)]
l2 = ax[2].legend(handles=sd, loc="upper center", bbox_to_anchor=(0.54, 1.005),
                  fontsize=8, framealpha=0.92, borderpad=0.24, labelspacing=0.2,
                  handlelength=1.5, handletextpad=0.4, borderaxespad=0.0)
l2.get_frame().set_linewidth(0.5)

os.makedirs(OUTDIR, exist_ok=True)
out = os.path.join(OUTDIR, "fig2_two_instruments.png")
fig.savefig(out, metadata={"Software": None})
print("wrote", out, PX, "printed", W_IN, "x", round(W_IN * PX[1] / PX[0], 4), "in")
