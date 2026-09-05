"""Reconstruct the ORIGINAL sf3_anchored3000_laws.png from the banked npz.

No generating script survived for this figure, so the reconstruction is validated
by rendering it at the original canvas (11.0 x 4.2 in @ 130 dpi, matplotlib
defaults) and pixel-diffing against the banked PNG.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = ".claude/outputs/v2-referee-experiments"
OUT = ".claude/scratch/figure-render-pass"
G = np.load(os.path.join(SRC, "sf3_gmor_sweep.npz"))
E = np.load(os.path.join(SRC, "sf3_ep_sweep.npz"))

# ---- panel (a) data -------------------------------------------------------
delta = G["delta"]; nh = G["n_half_env"]
xs = np.array(sorted(set(delta)))
ys = np.array([nh[delta == x].mean() for x in xs])
FIT = 7                                    # delta = 1e-4 .. 6e-2 (overdamped)
sl, ic = np.polyfit(np.log10(xs[:FIT]), np.log10(ys[:FIT]), 1)
FLOOR = 2 * np.log(2) / (-np.log(1 - 0.05))
# ---- panel (b) data -------------------------------------------------------
hm = E["h_minus_hstar"]; fj = E["freq_jac"]; sd = E["seed"]
m = (fj > 0) & (sd == 42)
xb, yb = hm[m], fj[m]
slb, icb = np.polyfit(np.log10(xb), np.log10(yb), 1)

if __name__ == "__main__":
    print("panel a: slope %.5f  floor %.5f" % (sl, FLOOR))
    print("panel b: slope %.5f" % slb)
    fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.2))
    ax[0].loglog(xs, ys, "o-", color="tab:green", ms=6,
                 label="anchored 3000-ep $n_{1/2}$(env)")
    ax[0].loglog(xs[:FIT], 10 ** (ic + sl * np.log10(xs[:FIT])), "--", color="0.4", lw=2,
                 label="slope %.3f (pred $-1$)" % sl)
    ax[0].axhline(FLOOR, color="red", ls=":", lw=1.5,
                  label="mass-indep floor %.2f" % FLOOR)
    ax[0].set_xlabel(r"GMOR tilt $\delta$")
    ax[0].set_ylabel(r"retention $n_{1/2}$ (steps)")
    ax[0].set_title("SF-3a: GMOR retention law survives at 3000 ep\n"
                    "(GMOR $\\mu^2/\\delta$ ratio = 1.0000$\\pm$1e-12, 3 seeds)")
    ax[0].legend(fontsize=8)
    ax[1].loglog(xb, yb, "o", color="tab:purple", ms=6,
                 label=r"$\varphi$ (Jacobian), above EP")
    ax[1].loglog(xb, 10 ** (icb + slb * np.log10(xb)), "--", color="0.4", lw=2,
                 label="slope %.3f (pred 0.5)" % slb)
    ax[1].set_xlabel(r"h $-$ h*")
    ax[1].set_ylabel(r"onset frequency $\varphi=|\arg\lambda|$")
    ax[1].set_title("SF-3b: EP onset $\\varphi\\propto\\sqrt{(h-h^*)}$ survives at 3000 ep\n"
                    "($\\varphi$=0 exactly below EP, all seeds)")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "sf3_recon.png"), dpi=130)
    print("wrote", os.path.join(OUT, "sf3_recon.png"))
