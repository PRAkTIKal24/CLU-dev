"""NEW generator (original lost: v5-gate fig4_emergent_Tstar via
scratch/v5-revision-1/regen_figs.py) for figC_Tstar.
Data: outputs/v5-gate/e2emg_halflife.json (emergent s42, s44) and e2des_halflife.json
(designed s44): (a) raw ratio_to_free = n_half_obs / n_half(exact-Goldstone) vs T for
gamma in {0.01, 0.05, 0.2}; (b) gamma=0.2 cells with l_theta/Delta < 0.06, bias-corrected
ratio_to_free / (1 + 3.099 l_theta/Delta), l_theta/Delta = eps sqrt(T/M_ch)/(gamma r* Delta)
(M_ch, r* from e1c_vcurve.json), against sigma_theta/Delta = sqrt(T/(F^2 mu2_soft))/Delta;
designed (mu2 = 0, sigma = inf) drawn at x = 3.0 as in the original. These reproduce
v5-gate.md s3.5's table to 3-4 decimals. Run from the repo root."""
import json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style as S
from lostcommon import VG, finish
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PX = (1725, 675); DPI = 150.0; FIGSIZE = (PX[0] / DPI, PX[1] / DPI)
S.setup(FIGSIZE[0], 5.5)
E = json.load(open(os.path.join(VG, "e2emg_halflife.json")))
D = json.load(open(os.path.join(VG, "e2des_halflife.json")))
V = json.load(open(os.path.join(VG, "e1c_vcurve.json")))
EPS, DELTA, BIAS, LMAX, X_INF = 0.05, 0.5, 3.099, 0.06, 3.0
SER = [("emergent150_s42", E, "emergent 42", S.OI[0], "-"),
       ("emergent150_s44", E, "emergent 44", S.OI[2], "-"),
       ("designed150_s44", D, "designed 44", "k", "--")]
MK = {0.01: "o", 0.05: "s", 0.2: "^"}

fig, ax = plt.subplots(1, 2, figsize=FIGSIZE, layout="constrained")
fig.get_layout_engine().set(w_pad=S.pt(2)/72, h_pad=S.pt(2)/72, wspace=0.04)
lw = S.pt(1.0); ms = S.pt(3.0)
for tag, src, lab, c, ls in SER:
    rows = src[tag]["rows"]
    for g, m in MK.items():
        rr = sorted([r for r in rows if r["gamma"] == g], key=lambda r: r["T"])
        ax[0].loglog([r["T"] for r in rr], [r["ratio_to_free"] for r in rr], ls, marker=m,
                     color=c, ms=ms, lw=lw, mfc="white" if tag.startswith("designed") else c, mew=S.pt(0.8))
ax[0].axhline(1.0, color="k", lw=S.pt(0.6))

ax[1].axvspan(0.0, 0.5, color="#EBEBEB", lw=0, zorder=0)
for tag, src, lab, c, ls in SER:
    M, rs = V[tag]["M_ch"], V[tag]["r_star"]
    xs, ys = [], []
    for r in src[tag]["rows"]:
        if r["gamma"] != 0.2:
            continue
        l = EPS * math.sqrt(r["T"] / M) / (r["gamma"] * rs * DELTA)
        if l >= LMAX:
            continue
        xs.append(math.sqrt(r["T"] / (r["F_sq"] * r["mu2_soft"])) / DELTA if r["mu2_soft"] > 0 else X_INF)
        ys.append(r["ratio_to_free"] / (1 + BIAS * l))
    ax[1].scatter(xs, ys, s=S.pt(4.2) ** 2, color=c, marker="o" if tag.startswith("emergent") else "D",
                  lw=0, zorder=3, label=lab + (r" ($\sigma_\theta=\infty$)" if tag.startswith("designed") else ""))
ax[1].axhline(1.0, color="k", ls="--", lw=S.pt(0.8), zorder=1)
ax[1].set_xlim(0.19866, 3.3996); ax[1].set_yscale("log"); ax[1].set_ylim(0.69976, 4.6241)
ax[0].set_xlim(0.0017332, 0.036812); ax[0].set_ylim(0.84759, 16.321)

ax[0].set_xlabel(r"$T$"); ax[0].set_ylabel(r"$n_{1/2}^{\rm obs}/n_{1/2}^{\rm Goldstone}$")
ax[1].set_xlabel(r"$\sigma_\theta/\Delta$")
ax[1].set_ylabel(r"bias-corrected $n_{1/2}^{\rm obs}/n_{1/2}^{\rm Goldstone}$")
h = [Line2D([], [], color=c, ls=ls, lw=lw, label=lab) for _, _, lab, c, ls in SER]
h += [Line2D([], [], color="0.3", lw=0, marker=m, ms=ms, label=rf"$\gamma={g:g}$") for g, m in MK.items()]
ax[0].legend(handles=h, loc="upper center", ncol=2)
ax[1].legend(loc="upper right")
S.letters(fig, ax)
finish(fig, "figC_Tstar.png", DPI)
