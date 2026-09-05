"""V5 Fig C.2 (appendix; printed as Figure 3) -- RE-LAID-OUT AS A 2x2 GRID so the
type targets are reachable, the four legends come back, and all twelve in-figure
numeric labels (9 hop fractions in (c), 3 vault factors in (d)) are restored.

Layout change only.  Values read from the same banked JSON the banked figure used
(source scratch/v5-vcurve-validation/a3_fig.py).
Pre-registration item names (Q1/Q2/Q3/Q5) stay removed from the panel titles;
seed short-tags s42/s43/s44 -> seed 1/2/3 (file order).

Printed box: FLF_W x FLF_H inches (default 4.400 x 3.400 = 0.80\linewidth).
"""
import json, math, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from size2 import canvas_wh

OUT = os.path.abspath(os.path.join(".claude", "outputs", "v5-vcurve-validation"))
OUTDIR = os.environ.get("FRP_OUT", os.path.join(".claude", "papers", "palm-variant", "v5", "figs"))
W_IN = float(os.environ.get("FLF_W", 4.730))
H_IN = float(os.environ.get("FLF_H", 3.550))

EM = json.load(open(os.path.join(OUT, "m3_emergent_VDS.json")))
DE = json.load(open(os.path.join(OUT, "m3_designed_crosscheck.json")))
X = json.load(open(os.path.join(OUT, "m3_emergent_X.json")))
F = json.load(open(os.path.join(OUT, "m3_fpt_emergent.json")))
SS = json.load(open(os.path.join(OUT, "m3_fpt_samesite.json")))
G = 0.05; SE = [42, 43, 44]
ge = lambda gp: 1 - (1 - G) * (1 - gp)

FIGSIZE, DPI, PX = canvas_wh(W_IN, H_IN)
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 8,
    "axes.linewidth": 0.6, "grid.linewidth": 0.35,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "xtick.major.size": 2.2, "ytick.major.size": 2.2,
    "xtick.minor.size": 1.2, "ytick.minor.size": 1.2,
    "xtick.major.pad": 1.2, "ytick.major.pad": 1.2,
    "axes.labelpad": 1.0, "axes.titlepad": 2.2, "savefig.dpi": DPI,
})
fig, axg = plt.subplots(2, 2, figsize=FIGSIZE, layout="constrained")
fig.get_layout_engine().set(w_pad=0.010, h_pad=0.008, wspace=0.060, hspace=0.070,
                            rect=(0.0, 0.062, 1.0, 0.938))
ax = [axg[0, 0], axg[0, 1], axg[1, 0], axg[1, 1]]

def small_legend(a, handles, **kw):
    lg = a.legend(handles=handles, framealpha=0.92, borderpad=0.24, labelspacing=0.2,
                  handlelength=1.4, handletextpad=0.4, borderaxespad=0.25, **kw)
    lg.get_frame().set_linewidth(0.5)
    return lg

# ---------------- (a) refrigerator ----------------
gp = np.linspace(0, 0.6, 200); gg = ge(gp)
ax[0].plot(gg, G * (2 - G) / (gg * (2 - gg)), "k-", lw=1.3)
ax[0].axhline(1.0, color="r", ls="--", lw=1.3)
for j, T in enumerate((4e-3, 8e-3)):
    for i, s in enumerate(SE):
        r = EM["V"][f"emergent150_s{s}|T={T}"]
        ax[0].plot([x["gamma_eff"] for x in r], [x["ratio_ch_mean"] for x in r],
                   "os"[j], color=f"C{i}", ms=3.0, mfc="none", mew=0.8)
ax[0].set_yscale("log"); ax[0].set_xlabel(r"$\gamma_{\rm eff}$")
ax[0].set_ylabel(r"$\mathrm{Var}(p_i)/(M_iT)$")
ax[0].set_title("(a) refrigerator law", loc="left")
ax[0].set_ylim(0.075, 6.5)
ax[0].set_yticks([0.1, 1.0]); ax[0].set_yticklabels(["0.1", "1.0"])
ax[0].grid(alpha=.3)
small_legend(ax[0], [
    Line2D([], [], color="k", lw=1.3, label="absorb-only"),
    Line2D([], [], color="r", lw=1.3, ls="--", label="coupled bath"),
    Line2D([], [], color="0.35", lw=0, marker="o", ms=3.0, mfc="none", mew=0.8,
           label=r"$T=4\times10^{-3}$"),
    Line2D([], [], color="0.35", lw=0, marker="s", ms=3.0, mfc="none", mew=0.8,
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
                   color=f"C{i}", ms=3.0, mfc="none", mew=0.8)
        ax[1].plot([x["gamma_phi"] for x in r if x not in b],
                   [x["D_ou_over_absorb"] for x in r if x not in b], "x", color=f"C{i}",
                   ms=3.0, mew=0.9)
for i, s in enumerate(SE):
    r = DE["D"][f"designed150_s{s}|T=0.001"]
    ax[1].plot([x["gamma_phi"] for x in r], [x["D_lin_over_absorb"] for x in r], "^--",
               color="0.5", ms=2.8, mfc="none", lw=0.7, mew=0.7)
ax[1].axhline(1, color="k", lw=1.1); ax[1].set_yscale("log"); ax[1].set_ylim(0.115, 14)
ax[1].set_xlabel(r"$\gamma_\phi$")
ax[1].set_ylabel(r"$\hat D_\theta/D^{\rm absorb}_\theta$")
ax[1].set_title(r"(b) $\gamma_{\rm eff}^{-2}$ diffusion law", loc="left")
ax[1].set_yticks([1.0, 10.0])
ax[1].grid(alpha=.3)
lb1 = small_legend(ax[1], [
    Line2D([], [], color="0.35", lw=0, marker="o", ms=3.0, mfc="none", mew=0.8,
           label="bounded cell"),
    Line2D([], [], color="0.35", lw=0, marker="x", ms=3.0, mew=0.9, label="delocalised"),
    Line2D([], [], color="0.5", lw=0.7, ls="--", marker="^", ms=2.8, mfc="none",
           label="designed cross-check")],
    loc="lower left", ncol=1)
ax[1].add_artist(lb1)

# ---------------- (c) confinement, with the 9 hop-fraction labels ----------------
w = 0.25
BARC = [("field_g0", "C3", r"no hole ($\gamma$=0.05)"),
        ("scalar_0.525", "C7", r"scalar control ($\gamma$=0.525)"),
        ("field_g50", "C0", r"$\gamma_\phi$ hole")]
for i, s in enumerate(SE):
    d = {r["arm"]: r for r in X["X"][f"emergent150_s{s}|T=0.004"]}
    for k, (a, c, lab) in enumerate(BARC):
        ax[2].bar(i + (k - 1) * w, d[a]["sigma_theta_iqr"], w, color=c)
        ax[2].text(i + (k - 1) * w, d[a]["sigma_theta_iqr"] * 1.10,
                   f"{100*d[a]['hop_frac']:.1f}%", ha="center", va="bottom", fontsize=8)
ax[2].set_xticks(range(3)); ax[2].set_xticklabels(["1", "2", "3"])
ax[2].set_xlabel("seed")
ax[2].set_yscale("log"); ax[2].set_ylim(0.045, 90.0)
ax[2].set_yticks([0.1, 1.0]); ax[2].set_yticklabels(["0.1", "1.0"])
ax[2].set_ylabel(r"$\sigma_\theta$ (rad)")
ax[2].set_title("(c) confinement", loc="left")
ax[2].grid(alpha=.3, axis="y")
small_legend(ax[2], [Patch(facecolor=c, label=lab) for _, c, lab in BARC],
             loc="upper left", ncol=1)

# ---------------- (d) same-site first passage, with the 3 vault-factor labels ----
for i, (c, g_) in enumerate(zip(F["cells"], SS["cells"])):
    base = g_["n_half_nofield_theta0"]; nin = c["cells"]["inside"]["n_half"]
    cap = c["cells"]["inside"]["cap"]
    v = cap if not np.isfinite(nin) else nin
    ax[3].bar(i - w / 2, base, w, color="C3")
    ax[3].bar(i + w / 2, v, w, color="C0", hatch="//" if not np.isfinite(nin) else None)
    ax[3].text(i + w / 2, v * 1.35, ("$>$" if not np.isfinite(nin) else "") + f"{v/base:.0f}$\\times$",
               ha="center", va="bottom", fontsize=8)
ax[3].set_xticks(range(3)); ax[3].set_xticklabels(["1", "2", "3"])
ax[3].set_xlim(-0.62, 2.72)
ax[3].set_xlabel("seed")
ax[3].set_yscale("log"); ax[3].set_ylim(1e2, 2e10)
ax[3].set_yticks([1e3, 1e5, 1e7, 1e9])
ax[3].set_ylabel(r"$n_{1/2}$ (steps)")
ax[3].set_title("(d) same-site first passage", loc="left")
ax[3].grid(alpha=.3, axis="y")
small_legend(ax[3], [
    Patch(facecolor="C3", label="no hole (same site)"),
    Patch(facecolor="C0", label=r"$\gamma_\phi$ hole"),
    Patch(facecolor="C0", hatch="//", label="censored lower bound")],
    loc="upper left", ncol=1)

seedleg = fig.legend(handles=[Line2D([], [], color=f"C{i}", lw=1.6, label=f"seed {i+1}")
                             for i in range(3)],
                     loc="lower center", bbox_to_anchor=(0.5, 0.004), ncol=3,
                     fontsize=8, framealpha=0.92, borderpad=0.24, columnspacing=1.4,
                     handlelength=1.6, handletextpad=0.4)
seedleg.get_frame().set_linewidth(0.5)

os.makedirs(OUTDIR, exist_ok=True)
out = os.path.join(OUTDIR, "figC2_vault_emergent.png")
fig.savefig(out, metadata={"Software": None})
print("wrote", out, PX, "printed", W_IN, "x", round(W_IN * PX[1] / PX[0], 4), "in")
