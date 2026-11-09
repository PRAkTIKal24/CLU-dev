"""Round-2 restyle of figC2_vault_emergent. Data calls identical to
figs-regen/_work/new_v5fC2.py (source outputs/v5-vcurve-validation/m3_*.json); the
in-plot text (panel titles, 9 hop-fraction labels, 3 vault-factor labels) is removed.
Run from the repo root."""
import json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style as S
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(_HERE, "..", "data", "v5-vcurve-validation")
OUTDIR = os.environ.get("FRP_OUT", os.path.join(_HERE, "renders"))
PX = (1892, 1420); DPI = 400.0; FIGSIZE = (PX[0] / DPI, PX[1] / DPI)
S.setup(FIGSIZE[0], 5.5)
EM = json.load(open(os.path.join(OUT, "m3_emergent_VDS.json")))
DE = json.load(open(os.path.join(OUT, "m3_designed_crosscheck.json")))
X = json.load(open(os.path.join(OUT, "m3_emergent_X.json")))
F = json.load(open(os.path.join(OUT, "m3_fpt_emergent.json")))
SS = json.load(open(os.path.join(OUT, "m3_fpt_samesite.json")))
G = 0.05; SE = [42, 43, 44]
ge = lambda gp: 1 - (1 - G) * (1 - gp)

fig, axg = plt.subplots(2, 2, figsize=FIGSIZE, layout="constrained")
fig.get_layout_engine().set(w_pad=S.pt(2)/72, h_pad=S.pt(2)/72, wspace=0.04, hspace=0.04)
COL = S.OI[:3]                                   # seeds 42, 43, 44
C_NOHOLE, C_SCALAR, C_HOLE = S.OI[4], S.GREY, S.OI[5]
LW = S.pt(1.2); MS = S.pt(3.0); MEW = S.pt(0.7)
ax = [axg[0, 0], axg[0, 1], axg[1, 0], axg[1, 1]]

def small_legend(a, handles, **kw):
    return a.legend(handles=handles, handlelength=1.4, **kw)

# ---------------- (a) refrigerator ----------------
gp = np.linspace(0, 0.6, 200); gg = ge(gp)
ax[0].plot(gg, G * (2 - G) / (gg * (2 - gg)), "-", color="k", lw=LW)
ax[0].axhline(1.0, color=S.GREY, ls="--", lw=LW)
for j, T in enumerate((4e-3, 8e-3)):
    for i, s in enumerate(SE):
        r = EM["V"][f"emergent150_s{s}|T={T}"]
        ax[0].plot([x["gamma_eff"] for x in r], [x["ratio_ch_mean"] for x in r],
                   "os"[j], color=COL[i], ms=MS, mfc="none", mew=MEW, zorder=3)
ax[0].set_yscale("log"); ax[0].set_xlabel(r"$\gamma_{\rm eff}$")
ax[0].set_ylabel(r"$\mathrm{Var}(p_i)/(M_iT)$")
ax[0].set_ylim(0.075, 6.5)
ax[0].set_yticks([0.1, 1.0]); ax[0].set_yticklabels(["0.1", "1.0"])
small_legend(ax[0], [
    Line2D([], [], color="k", lw=LW, label="absorb-only"),
    Line2D([], [], color=S.GREY, lw=LW, ls="--", label="coupled bath"),
    Line2D([], [], color="0.3", lw=0, marker="o", ms=MS, mfc="none", mew=MEW,
           label=r"$T=4\times10^{-3}$"),
    Line2D([], [], color="0.3", lw=0, marker="s", ms=MS, mfc="none", mew=MEW,
           label=r"$T=8\times10^{-3}$")],
    loc="upper center", ncol=2, columnspacing=0.8)

# ---------------- (b) gamma_eff^-2 law ----------------
for i, s in enumerate(SE):
    for j, T in enumerate((4e-3, 8e-3)):
        r = EM["D"][f"emergent150_s{s}|T={T}"]; F2 = EM["runs"][f"emergent150_s{s}"]["F_sq"]
        b = [x for x in r if x["sigma_theta_ou"] < 2 * math.sqrt(
            x["T"] * (G * (2 - G) / (ge(x["gamma_phi"]) * (2 - ge(x["gamma_phi"])))) /
            (F2 * EM["runs"][f"emergent150_s{s}"]["mu2_adiab"]))]
        ax[1].plot([x["gamma_phi"] for x in b], [x["D_ou_over_absorb"] for x in b], "os"[j],
                   color=COL[i], ms=MS, mfc="none", mew=MEW, zorder=3)
        ax[1].plot([x["gamma_phi"] for x in r if x not in b],
                   [x["D_ou_over_absorb"] for x in r if x not in b], "x", color=COL[i],
                   ms=MS, mew=MEW, zorder=3)
for i, s in enumerate(SE):
    r = DE["D"][f"designed150_s{s}|T=0.001"]
    ax[1].plot([x["gamma_phi"] for x in r], [x["D_lin_over_absorb"] for x in r], "^--",
               color=S.GREY, ms=MS * 0.9, mfc="none", lw=S.pt(0.7), mew=MEW)
ax[1].axhline(1, color="k", lw=S.pt(0.6)); ax[1].set_yscale("log"); ax[1].set_ylim(0.115, 14)
ax[1].set_xlabel(r"$\gamma_\phi$")
ax[1].set_ylabel(r"$\hat D_\theta/D^{\rm absorb}_\theta$")
ax[1].set_yticks([1.0, 10.0])
lb1 = small_legend(ax[1], [
    Line2D([], [], color="0.3", lw=0, marker="o", ms=MS, mfc="none", mew=MEW,
           label="bounded"),
    Line2D([], [], color="0.3", lw=0, marker="x", ms=MS, mew=MEW, label="delocalized"),
    Line2D([], [], color=S.GREY, lw=S.pt(0.7), ls="--", marker="^", ms=MS * 0.9, mfc="none",
           mew=MEW, label="designed")],
    loc="lower left", ncol=1)
ax[1].add_artist(lb1)

# ---------------- (c) confinement, with the 9 hop-fraction labels ----------------
w = 0.25
BARC = [("field_g0", C_NOHOLE, r"no hole"),
        ("scalar_0.525", C_SCALAR, r"scalar control"),
        ("field_g50", C_HOLE, r"hole")]
for i, s in enumerate(SE):
    d = {r["arm"]: r for r in X["X"][f"emergent150_s{s}|T=0.004"]}
    for k, (a, c, lab) in enumerate(BARC):
        ax[2].bar(i + (k - 1) * w, d[a]["sigma_theta_iqr"], w, color=c)
ax[2].set_xticks(range(3)); ax[2].set_xticklabels([str(s) for s in SE])
ax[2].set_xlabel("seed")
ax[2].set_yscale("log"); ax[2].set_ylim(0.045, 9.0)
ax[2].set_yticks([0.1, 1.0]); ax[2].set_yticklabels(["0.1", "1.0"])
ax[2].set_ylabel(r"$\sigma_\theta$ (rad)")
small_legend(ax[2], [Patch(facecolor=c, label=lab) for _, c, lab in BARC],
             loc="upper left", ncol=1)

# ---------------- (d) same-site first passage, with the 3 vault-factor labels ----
for i, (c, g_) in enumerate(zip(F["cells"], SS["cells"])):
    base = g_["n_half_nofield_theta0"]; nin = c["cells"]["inside"]["n_half"]
    cap = c["cells"]["inside"]["cap"]
    v = cap if not np.isfinite(nin) else nin
    ax[3].bar(i - w / 2, base, w, color=C_NOHOLE)
    ax[3].bar(i + w / 2, v, w, color=C_HOLE, edgecolor="k", linewidth=0,
              hatch="///" if not np.isfinite(nin) else None)
ax[3].set_xticks(range(3)); ax[3].set_xticklabels([str(s) for s in SE])
ax[3].set_xlim(-0.62, 2.72)
ax[3].set_xlabel("seed")
ax[3].set_yscale("log"); ax[3].set_ylim(1e2, 3e8)
ax[3].set_yticks([1e2, 1e4, 1e6, 1e8])
ax[3].set_ylabel(r"$n_{1/2}$ (steps)")
small_legend(ax[3], [
    Patch(facecolor=C_NOHOLE, label="no hole"),
    Patch(facecolor=C_HOLE, label="hole"),
    Patch(facecolor=C_HOLE, edgecolor="k", linewidth=0, hatch="///", label="censored")],
    loc="upper left", ncol=1)

fig.legend(handles=[Line2D([], [], color=COL[i], lw=LW, label=f"seed {s}")
                    for i, s in enumerate(SE)],
           loc="outside lower center", ncol=3, columnspacing=1.6)
S.letters(fig, ax)

os.makedirs(OUTDIR, exist_ok=True)
out = os.path.join(OUTDIR, "figC2_vault_emergent.png")
S.save(fig, out, DPI)
print("wrote", out)
