"""NEW generator (original lost: scratch/v5-revision-1/regen_figs.py, v5-gate figs.py
lineage) for figC_lambda_coset. Data: outputs/v5-gate/e1c_vcurve.json (48 gamma).
(a) n_half_jac vs gamma per emergent seed, dotted line at each seed's gamma_crit_soft;
designed s44 has n_half = inf at every gamma and is drawn as a horizontal line near
the top at the height measured on the original (2.6e4; the value is a placeholder for
infinity, not a measurement). (b) |dev_from_1| = ||lambda_coset|-1| for all four
checkpoints; exact zeros are shown at 1e-17 (as measured on the original); dashed line
at float64 eps. Run from the repo root."""
import json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style as S
from lostcommon import VG, finish
import matplotlib.pyplot as plt

PX = (1650, 660); DPI = 150.0; FIGSIZE = (PX[0] / DPI, PX[1] / DPI)
S.setup(FIGSIZE[0], 5.5)
C = json.load(open(os.path.join(VG, "e1c_vcurve.json")))
TAGS = [("emergent150_s42", "emergent 42", S.OI[0]), ("emergent150_s43", "emergent 43", S.OI[1]),
        ("emergent150_s44", "emergent 44", S.OI[2]), ("designed150_s44", "designed 44", "k")]
INF_DRAW = 2.6e4
FLOOR = 1e-17

fig, ax = plt.subplots(1, 2, figsize=FIGSIZE, layout="constrained")
fig.get_layout_engine().set(w_pad=S.pt(2)/72, h_pad=S.pt(2)/72, wspace=0.04)
lw = S.pt(1.0); ms = S.pt(2.6)
for tag, lab, c in TAGS:
    v = C[tag]; rows = v["rows"]
    g = np.array([r["gamma"] for r in rows])
    if tag.startswith("emergent"):
        ax[0].loglog(g, [r["n_half_jac"] for r in rows], "o-", color=c, ms=ms, lw=lw, zorder=3)
        ax[0].axvline(v["gamma_crit_soft"], color=c, ls=":", lw=S.pt(0.8), zorder=1)
    else:
        ax[0].axhline(INF_DRAW, color=c, lw=S.pt(1.6), zorder=2)
    ax[1].semilogy(g, np.maximum([r["dev_from_1"] for r in rows], FLOOR), "o-", color=c, ms=ms,
                   lw=lw, label=lab, zorder=3)
ax[1].axhline(np.finfo(float).eps, color=S.GREY, ls="--", lw=S.pt(0.8), label=r"machine $\epsilon$", zorder=2)
ax[0].set_ylim(29.756, 29978)
ax[1].set_xscale("log")
ax[0].set_xlabel(r"$\gamma$"); ax[0].set_ylabel(r"$n_{1/2}$ (steps)")
ax[1].set_xlabel(r"$\gamma$"); ax[1].set_ylabel(r"$||\lambda_{\rm coset}|-1|$")
ax[1].legend(loc="center right")
S.letters(fig, ax)
finish(fig, "figC_lambda_coset.png", DPI)
