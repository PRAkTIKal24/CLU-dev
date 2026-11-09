"""NEW generator (original lost: v5-gate fig1_friction_hole_vault via
scratch/v5-revision-1/regen_figs.py) for fig2_vault (designed arm, T=1e-3, gamma=0.05).
Data: outputs/v5-gate/r3main_results.json and r3t6_g50_results.json.
(a) Var(p_i)/(M_i T) = ratio_ch_mean, mean +- std (ddof 0) over seeds 42-44, vs
    gamma_eff; absorb-only curve gamma(2-gamma)/(g_eff(2-g_eff)); coupled bath = 1.
(b) D_hat mean +- std over seeds vs gamma_eff; absorb-only D0 (gamma/g_eff)^2 and
    coupled-bath D0 gamma(2-g_eff)/(g_eff(2-gamma)) with D0 = mean D_pred_absorb at
    gamma_phi = 0; scalar gamma = 0.525 control = mean S D_hat.
(c) raw first-passage n_half outside (gamma_phi = 0) and inside (gamma_phi = 0.5)
    per seed (bars of width 0.36, measured on the original).
Run from the repo root."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style as S
from lostcommon import VG, finish
import matplotlib.pyplot as plt

PX = (2250, 660); DPI = 150.0; FIGSIZE = (PX[0] / DPI, PX[1] / DPI)
S.setup(FIGSIZE[0], 5.5)
R = json.load(open(os.path.join(VG, "r3main_results.json")))
T6 = json.load(open(os.path.join(VG, "r3t6_g50_results.json")))["T6"]
SE = ["42", "43", "44"]; G = R["config"]["gamma"]
nphi = len(R["V"]["42"])
ge = np.array([R["V"]["42"][i]["gamma_eff"] for i in range(nphi)])
V = np.array([[R["V"][s][i]["ratio_ch_mean"] for i in range(nphi)] for s in SE])
Dh = np.array([[R["D"][s][i]["D_hat"] for i in range(nphi)] for s in SE])
D0 = np.mean([R["D"][s][0]["D_pred_absorb"] for s in SE])
Ssc = np.mean([[r["D_hat"] for r in R["S"][s] if abs(r["gamma"] - 0.525) < 1e-9][0] for s in SE])
gg = np.linspace(G, 0.55, 200)

fig, ax = plt.subplots(1, 3, figsize=FIGSIZE, layout="constrained")
fig.get_layout_engine().set(w_pad=S.pt(2)/72, h_pad=S.pt(2)/72, wspace=0.04)
lw = S.pt(1.2); ms = S.pt(3.2); ew = S.pt(0.7)
ax[0].plot(gg, G * (2 - G) / (gg * (2 - gg)), "-", color=S.OI[0], lw=lw, label="absorb-only")
ax[0].axhline(1.0, color=S.GREY, ls="--", lw=lw, label="coupled bath")
ax[0].errorbar(ge, V.mean(0), yerr=V.std(0), fmt="o", color="k", ms=ms, elinewidth=ew,
               capsize=S.pt(1.2), label="measured", zorder=3)
ax[1].loglog(gg, D0 * (G / gg) ** 2, "-", color=S.OI[0], lw=lw, label="absorb-only")
ax[1].loglog(ge, D0 * G * (2 - ge) / (ge * (2 - G)), "--", color=S.GREY, lw=lw, label="coupled bath")
ax[1].errorbar(ge, Dh.mean(0), yerr=Dh.std(0), fmt="o", color="k", ms=ms, elinewidth=ew,
               capsize=S.pt(1.2), label="field", zorder=3)
ax[1].plot([0.525], [Ssc], "s", color=S.OI[1], ms=ms * 1.1, label="scalar friction", zorder=3)
w = 0.36
for i, s in enumerate(SE):
    rows = {r["arm"]: r for r in T6[s]["rows"]}
    ax[2].bar(i - w / 2, rows["outside"]["n_half_obs"], w, color=S.OI[4], label="outside" if i == 0 else None)
    ax[2].bar(i + w / 2, rows["inside"]["n_half_obs"], w, color=S.OI[0], label="inside" if i == 0 else None)
ax[2].set_yscale("log"); ax[2].set_ylim(502.44, 2.0407e7); ax[2].set_xlim(-0.4976, 2.4929)
ax[2].set_xticks(range(3)); ax[2].set_xticklabels(SE)
ax[0].set_xlim(0.024123, 0.57588); ax[0].set_ylim(-0.0007, 1.151)
ax[1].set_xlim(0.044237, 0.62007); ax[1].set_ylim(9.938e-06, 0.0019462)

ax[0].set_xlabel(r"$\gamma_{\rm eff}$"); ax[0].set_ylabel(r"$\mathrm{Var}(p_i)/(M_iT)$")
ax[1].set_xlabel(r"$\gamma_{\rm eff}$"); ax[1].set_ylabel(r"$D_\theta$")
ax[2].set_xlabel("seed"); ax[2].set_ylabel(r"$n_{1/2}$ (steps)")
ax[0].legend(loc="center right")
ax[1].legend(loc="lower left")
ax[2].legend(loc="upper left", ncol=2)
S.letters(fig, ax)
finish(fig, "fig2_vault.png", DPI)
