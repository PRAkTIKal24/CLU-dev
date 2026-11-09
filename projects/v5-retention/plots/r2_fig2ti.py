"""Round-2 restyle of fig2_two_instruments. Data calls identical to
figs-regen/_work/new_v5f2.py (source outputs/v5-vcurve-validation/m1_emergent.json).
Run from the repo root."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style as S
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(_HERE, "..", "data", "v5-vcurve-validation")
OUTDIR = os.environ.get("FRP_OUT", os.path.join(_HERE, "renders"))
PX = (1980, 900); DPI = 400.0; FIGSIZE = (PX[0] / DPI, PX[1] / DPI)
S.setup(FIGSIZE[0], 5.5)

E = json.load(open(os.path.join(OUT, "m1_emergent.json")))
SEEDS = ["emergent150_s42", "emergent150_s43", "emergent150_s44"]
DEL = "0.05"

fig, ax = plt.subplots(1, 3, figsize=FIGSIZE, layout="constrained")
fig.get_layout_engine().set(w_pad=S.pt(2)/72, h_pad=S.pt(2)/72, wspace=0.04)
cols = S.OI[:3]
lw = S.pt(1.2); ms = S.pt(2.6); mew = S.pt(0.6)
for i, t in enumerate(SEEDS):
    r = E["runs"][t]; gc = r["gamma_crit_soft"]; rows = r["deltas"][DEL]["rows"]
    g = np.array([v["gamma"] for v in rows])
    ax[0].loglog(g, [v["n_half_jac"] for v in rows], "-", color=cols[i], lw=lw)
    ax[0].loglog(g, [v["n_half_R3"] for v in rows], "o", color=cols[i], ms=ms, mfc="none", mew=mew)
    ax[0].loglog(g, [v["n_half_R1"] for v in rows], ":", color=cols[i], lw=lw)
    ax[0].axvline(gc, color=cols[i], ls=(0, (3, 2)), lw=S.pt(0.5), alpha=0.7)
    ax[1].semilogx(g, [v["ratio_Gam"] for v in rows], "-", color=cols[i], lw=lw)
    ax[1].semilogx(g, [v["ratio_jac_R1"] for v in rows], ":", color=cols[i], lw=lw)
    ax[2].loglog(np.array([v["gamma"] for v in rows]) / gc,
                 np.array([v["n_half_jac"] for v in rows]) * gc, "-", color=cols[i], lw=lw)
    ax[2].loglog(np.array([v["gamma"] for v in rows]) / gc,
                 np.array([v["n_half_R3"] for v in rows]) * gc, "o", color=cols[i], ms=ms,
                 mfc="none", mew=mew)

ax[0].set_xlabel(r"$\gamma$"); ax[0].set_ylabel(r"$n_{1/2}$ (steps)")
inst = [Line2D([], [], color="0.3", lw=lw, ls="-", label="Jacobian"),
        Line2D([], [], color="0.3", lw=0, marker="o", ms=ms, mfc="none", mew=mew,
               label="rollout rate"),
        Line2D([], [], color="0.3", lw=lw, ls=":", label="rollout threshold")]
ax[0].set_ylim(top=4e4)
ax[0].legend(handles=inst, loc="upper left", bbox_to_anchor=(0.0, 1.03),
             frameon=True, facecolor="white", edgecolor="none", framealpha=1.0)

ax[1].axhline(1.0, color="k", lw=S.pt(0.5))
ax[1].set_ylim(0, 2.6)
ax[1].set_yticks([0, 0.5, 1.0, 1.5, 2.0, 2.5])
ax[1].set_xlabel(r"$\gamma$"); ax[1].set_ylabel("instrument ratio")
rat = [Line2D([], [], color="0.3", lw=lw, ls="-", label="decay-rate ratio"),
       Line2D([], [], color="0.3", lw=lw, ls=":", label="threshold ratio")]
ax[1].legend(handles=rat, loc="lower center", bbox_to_anchor=(0.5, -0.01))

ax[2].set_xlabel(r"$\gamma/\gamma_{\rm crit}$")
ax[2].set_ylabel(r"$n_{1/2}\,\gamma_{\rm crit}$")
sd = [Line2D([], [], color=cols[i], lw=lw, label=f"seed {s}")
      for i, s in enumerate(int(t.rsplit("_s", 1)[1]) for t in SEEDS)]
ax[2].legend(handles=sd, loc="upper center", bbox_to_anchor=(0.5, 1.03))

S.letters(fig, ax)

os.makedirs(OUTDIR, exist_ok=True)
S.save(fig, os.path.join(OUTDIR, "fig2_two_instruments.png"), DPI)
print("wrote")
