"""V5 variant Fig C.2 (figC2_vault_emergent.png) re-rendered at its FINAL PRINTED SIZE.

Printed footprint measured from the built PDF: 316.788 x 70.027 pt = 4.400 x 0.973 in
(0.80\linewidth).  Aspect preserved to the banked figure's 3040/672 = 4.523810.

Replot only: values read from the same banked JSON the banked figure used
(source script scratch/v5-vcurve-validation/a3_fig.py).
Pre-registration item names (Q1/Q2/Q3/Q5) removed from panel titles; seed short-tags
s42/s43/s44 -> seed 1/2/3 (file order).
"""
import json, math, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from size import canvas

OUT = os.path.abspath(os.path.join(".claude", "outputs", "v5-vcurve-validation"))
OUTDIR = os.environ.get("FRP_OUT", os.path.join(".claude", "papers", "palm-variant", "v5", "figs"))
EM = json.load(open(os.path.join(OUT, "m3_emergent_VDS.json")))
DE = json.load(open(os.path.join(OUT, "m3_designed_crosscheck.json")))
X = json.load(open(os.path.join(OUT, "m3_emergent_X.json")))
F = json.load(open(os.path.join(OUT, "m3_fpt_emergent.json")))
SS = json.load(open(os.path.join(OUT, "m3_fpt_samesite.json")))
G = 0.05; SE = [42, 43, 44]; LAB = {42: "seed 1", 43: "seed 2", 44: "seed 3"}
ge = lambda gp: 1 - (1 - G) * (1 - gp)

FIGSIZE, DPI, PX = canvas(3040, 672, 4.400, 400.0)
plt.rcParams.update({
    "font.size": 7.5, "axes.labelsize": 8, "axes.titlesize": 9,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.55, "grid.linewidth": 0.35,
    "xtick.major.width": 0.55, "ytick.major.width": 0.55,
    "xtick.minor.width": 0.35, "ytick.minor.width": 0.35,
    "xtick.major.size": 2.0, "ytick.major.size": 2.0,
    "xtick.minor.size": 1.1, "ytick.minor.size": 1.1,
    "xtick.major.pad": 1.0, "ytick.major.pad": 1.0,
    "axes.labelpad": 0.8, "axes.titlepad": 2.0, "savefig.dpi": DPI,
})
fig, ax = plt.subplots(1, 4, figsize=FIGSIZE, layout="constrained")
fig.get_layout_engine().set(w_pad=0.006, h_pad=0.004, wspace=0.035, hspace=0.0)

# ---------------- (a) refrigerator ----------------
gp = np.linspace(0, 0.6, 200); gg = ge(gp)
ax[0].plot(gg, G * (2 - G) / (gg * (2 - gg)), "k-", lw=1.3, label="absorb-only")
ax[0].axhline(1.0, color="r", ls="--", lw=1.3, label="coupled bath")
for j, T in enumerate((4e-3, 8e-3)):
    for i, s in enumerate(SE):
        r = EM["V"][f"emergent150_s{s}|T={T}"]
        ax[0].plot([x["gamma_eff"] for x in r], [x["ratio_ch_mean"] for x in r],
                   "os"[j], color=f"C{i}", ms=3.0, mfc="none", mew=0.8)
ax[0].set_yscale("log"); ax[0].set_xlabel(r"$\gamma_{\rm eff}$")
ax[0].set_ylabel(r"$\mathrm{Var}(p_i)/(M_iT)$")
ax[0].set_title("(a)", loc="left")
ax[0].set_ylim(0.09, 1.9)
ax[0].set_yticks([0.1, 1.0])
ax[0].set_yticklabels(["0.1", "1.0"])
ax[0].text(0.335, 1.12, "coupled bath", fontsize=7, color="r", ha="center", va="bottom")
ax[0].text(0.245, 0.30, "absorb-only", fontsize=7, ha="left", va="bottom")
ax[0].grid(alpha=.3)

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
               color="0.5", ms=2.6, mfc="none", lw=0.7, mew=0.7)
ax[1].axhline(1, color="k", lw=1.1); ax[1].set_yscale("log"); ax[1].set_ylim(0.5, 10)
ax[1].set_xlabel(r"$\gamma_\phi$")
ax[1].set_ylabel(r"$\hat D_\theta/D^{\rm absorb}_\theta$")
ax[1].set_title("(b)", loc="left")
ax[1].grid(alpha=.3)

# ---------------- (c) confinement ----------------
w = 0.25
for i, s in enumerate(SE):
    d = {r["arm"]: r for r in X["X"][f"emergent150_s{s}|T=0.004"]}
    for k, (a, c) in enumerate([("field_g0", "C3"), ("scalar_0.525", "C7"), ("field_g50", "C0")]):
        ax[2].bar(i + (k - 1) * w, d[a]["sigma_theta_iqr"], w, color=c,
                  label=["no hole", "scalar control", "friction hole"][k] if i == 0 else None)
ax[2].set_xticks(range(3)); ax[2].set_xticklabels(["1", "2", "3"])
ax[2].set_xlabel("seed")
ax[2].set_yscale("log"); ax[2].set_ylim(0.045, 3.0)
ax[2].set_yticks([0.1, 1.0])
ax[2].set_yticklabels(["0.1", "1.0"])
ax[2].set_ylabel(r"$\sigma_\theta$ (rad)")
ax[2].set_title("(c)", loc="left")
ax[2].grid(alpha=.3, axis="y")

# ---------------- (d) same-site first passage ----------------
for i, (c, g_) in enumerate(zip(F["cells"], SS["cells"])):
    base = g_["n_half_nofield_theta0"]; nin = c["cells"]["inside"]["n_half"]
    cap = c["cells"]["inside"]["cap"]
    ax[3].bar(i - w / 2, base, w, color="C3", label="no hole" if i == 0 else None)
    ax[3].bar(i + w / 2, cap if not np.isfinite(nin) else nin, w, color="C0",
              hatch="//" if not np.isfinite(nin) else None,
              label="friction hole" if i == 0 else None)
    v = cap if not np.isfinite(nin) else nin
ax[3].set_xticks(range(3)); ax[3].set_xticklabels(["1", "2", "3"])
ax[3].set_xlabel("seed")
ax[3].set_yscale("log"); ax[3].set_ylim(1e2, 5e6)
ax[3].set_yticks([1e3, 1e5])
ax[3].set_ylabel(r"$n_{1/2}$ (steps)")
ax[3].set_title("(d)", loc="left")
ax[3].grid(alpha=.3, axis="y")

os.makedirs(OUTDIR, exist_ok=True)
out = os.path.join(OUTDIR, "figC2_vault_emergent.png")
fig.savefig(out, metadata={"Software": None})
print("wrote", out, plt.imread(out).shape)
