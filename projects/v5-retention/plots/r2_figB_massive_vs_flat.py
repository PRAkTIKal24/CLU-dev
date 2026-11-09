"""NEW generator (original lost: t-lever-forgetting/s7_figures.py fig4_massive_vs_flat,
as re-run by scratch/v5-revision-1/regen_figs.py) for figB_massive_vs_flat.
Data: outputs/t-lever-forgetting/s4b_jacobian.json (5 designed seeds x 22 gamma).
(a) n_half_massive vs gamma per seed, dotted line at each seed's gamma_crit, dashed
2 ln2/gamma on the gamma grid; (b) |flat_lam_minus_1| per seed (exact zeros fall off
the log axis, as in the original), dashed line at float64 eps. Run from the repo root."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style as S
from lostcommon import TL, finish
import matplotlib.pyplot as plt

PX = (1594, 635); DPI = 140.0; FIGSIZE = (PX[0] / DPI, PX[1] / DPI)
S.setup(FIGSIZE[0], 5.5)
D = [r for r in json.load(open(os.path.join(TL, "s4b_jacobian.json"))) if "gamma" in r]
SEEDS = sorted({r["seed"] for r in D})

fig, ax = plt.subplots(1, 2, figsize=FIGSIZE, layout="constrained")
fig.get_layout_engine().set(w_pad=S.pt(2)/72, h_pad=S.pt(2)/72, wspace=0.04)
lw = S.pt(1.0); ms = S.pt(2.6)
for i, s in enumerate(SEEDS):
    rows = sorted([r for r in D if r["seed"] == s], key=lambda r: r["gamma"])
    g = np.array([r["gamma"] for r in rows])
    ax[0].loglog(g, [r["n_half_massive"] for r in rows], "o-", color=S.OI[i], ms=ms, lw=lw, zorder=3)
    ax[0].axvline(rows[0]["gamma_crit"], color=S.OI[i], ls=":", lw=S.pt(0.8), zorder=1)
    ax[1].semilogy(g, np.abs([r["flat_lam_minus_1"] for r in rows]), "o-", color=S.OI[i], ms=ms,
                   lw=lw, label=f"seed {s}", zorder=3)
g0 = np.array(sorted(r["gamma"] for r in D if r["seed"] == SEEDS[0]))
ax[0].loglog(g0, 2 * np.log(2) / g0, "--", color="k", lw=S.pt(0.8), zorder=2)
ax[1].axhline(np.finfo(float).eps, color="k", ls="--", lw=S.pt(0.8), label=r"machine $\epsilon$", zorder=2)
ax[1].set_xscale("log"); ax[1].set_ylim(1e-17, 1e-10)
ax[0].set_xlabel(r"$\gamma$"); ax[0].set_ylabel(r"massive $n_{1/2}$ (steps)")
ax[1].set_xlabel(r"$\gamma$"); ax[1].set_ylabel(r"$||\lambda_{\rm flat}|-1|$")
ax[1].legend(loc="upper right", ncol=2)
S.letters(fig, ax)
finish(fig, "figB_massive_vs_flat.png", DPI)
