"""NEW generator (original lost: t-lever-forgetting/s7_figures.py fig2_signflip, as
re-run by scratch/v5-revision-1/regen_figs.py) for figB_signflip.
Data: outputs/t-lever-forgetting/s3_halflife_cells.json (seed 44, 20 cells):
n_half_obs with CI [n_half_ci_lo, n_half_ci_hi] (points) and n_half_pred (lines);
massive mode: outputs/t-lever-forgetting/s4b_jacobian.json seed 44 n_half_massive
(dashed); dotted line at that seed's gamma_crit. Run from the repo root."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style as S
from lostcommon import TL, finish
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PX = (1594, 663); DPI = 140.0; FIGSIZE = (PX[0] / DPI, PX[1] / DPI)
S.setup(FIGSIZE[0], 5.5)
C = json.load(open(os.path.join(TL, "s3_halflife_cells.json")))
J = [r for r in json.load(open(os.path.join(TL, "s4b_jacobian.json"))) if r.get("seed") == 44 and "gamma" in r]
J = sorted(J, key=lambda r: r["gamma"])
GS = sorted({c["gamma"] for c in C}); TS = sorted({c["T"] for c in C})


def sci(v):
    e = int(np.floor(np.log10(v) + 1e-9)); m = v / 10 ** e
    return rf"$10^{{{e}}}$" if abs(m - 1) < 1e-9 else rf"${m:g}\times10^{{{e}}}$"


fig, ax = plt.subplots(1, 2, figsize=FIGSIZE, layout="constrained")
fig.get_layout_engine().set(w_pad=S.pt(2)/72, h_pad=S.pt(2)/72, wspace=0.04)
lw = S.pt(1.0); ms = S.pt(3.0); ew = S.pt(0.7)

ax[0].loglog([r["gamma"] for r in J], [r["n_half_massive"] for r in J], "--", color="k", lw=S.pt(1.3))
gc = J[0]["gamma_crit"]
ax[0].axvline(gc, color=S.GREY, ls=":", lw=S.pt(0.8))
for i, t in enumerate(TS):
    cs = sorted([c for c in C if c["T"] == t], key=lambda c: c["gamma"])
    x = np.array([c["gamma"] for c in cs]); y = np.array([c["n_half_obs"] for c in cs])
    err = np.array([y - [c["n_half_ci_lo"] for c in cs], np.array([c["n_half_ci_hi"] for c in cs]) - y])
    ax[0].errorbar(x, y, yerr=err, fmt="o", color=S.OI[i], ms=ms, elinewidth=ew, capsize=S.pt(1.2), capthick=ew)
    ax[0].plot(x, [c["n_half_pred"] for c in cs], "-", color=S.OI[i], lw=lw)
for i, g in enumerate(GS):
    cs = sorted([c for c in C if c["gamma"] == g], key=lambda c: c["T"])
    x = np.array([c["T"] for c in cs]); y = np.array([c["n_half_obs"] for c in cs])
    err = np.array([y - [c["n_half_ci_lo"] for c in cs], np.array([c["n_half_ci_hi"] for c in cs]) - y])
    ax[1].errorbar(x, y, yerr=err, fmt="o", color=S.OI[i], ms=ms, elinewidth=ew, capsize=S.pt(1.2), capthick=ew)
    ax[1].loglog(x, [c["n_half_pred"] for c in cs], "-", color=S.OI[i], lw=lw)
ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[1].set_xscale("log"); ax[1].set_yscale("log")
ax[0].set_xlabel(r"$\gamma$"); ax[0].set_ylabel(r"$n_{1/2}$ (steps)")
ax[1].set_xlabel(r"$T$"); ax[1].set_ylabel(r"$n_{1/2}$ (steps)")
h0 = [Line2D([], [], color="k", ls="--", lw=S.pt(1.3), label="massive"),
      Line2D([], [], color=S.GREY, ls=":", lw=S.pt(0.8), label=r"$\gamma_{\rm crit}$")]
h0 += [Line2D([], [], color=S.OI[i], marker="o", ms=ms, lw=lw, label=r"$T=$" + sci(t)) for i, t in enumerate(TS)]
ax[0].legend(handles=h0, loc="upper left", ncol=2)
h1 = [Line2D([], [], color=S.OI[i], marker="o", ms=ms, lw=lw, label=rf"$\gamma={g:g}$") for i, g in enumerate(GS)]
ax[1].legend(handles=h1, loc="upper right")
S.letters(fig, ax)
finish(fig, "figB_signflip.png", DPI)
