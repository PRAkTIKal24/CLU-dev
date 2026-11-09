"""NEW generator (original lost: t-lever-forgetting/s7_figures.py fig1_dlaw, as
re-run by scratch/v5-revision-1/regen_figs.py) for figB_dlaw.
Data: outputs/t-lever-forgetting/s2_dlaw_cells.json (seed 44, 5 gamma x 5 T).
(a) D_hat (dots) and D_pred (lines) vs T per gamma; (b) the same vs gamma per T;
(c) D_ratio +- D_ratio_sem per cell (file order), band = 1 +- std(ddof=1) of D_ratio
(0.0223; band geometry measured on the original), line at 1. Run from the repo root."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style as S
from lostcommon import TL, finish
import matplotlib.pyplot as plt

PX = (2083, 628); DPI = 140.0; FIGSIZE = (PX[0] / DPI, PX[1] / DPI)
S.setup(FIGSIZE[0], 5.5)
C = json.load(open(os.path.join(TL, "s2_dlaw_cells.json")))
GS = sorted({c["gamma"] for c in C}); TS = sorted({c["T"] for c in C})


def sci(v):
    e = int(np.floor(np.log10(v) + 1e-9)); m = v / 10 ** e
    return rf"$10^{{{e}}}$" if abs(m - 1) < 1e-9 else rf"${m:g}\times10^{{{e}}}$"


fig, ax = plt.subplots(1, 3, figsize=FIGSIZE, layout="constrained")
fig.get_layout_engine().set(w_pad=S.pt(2)/72, h_pad=S.pt(2)/72, wspace=0.04)
lw = S.pt(1.0); ms = S.pt(3.2)
for i, g in enumerate(GS):
    cs = sorted([c for c in C if c["gamma"] == g], key=lambda c: c["T"])
    T = [c["T"] for c in cs]
    ax[0].loglog(T, [c["D_pred"] for c in cs], "-", color=S.OI[i], lw=lw)
    ax[0].loglog(T, [c["D_hat"] for c in cs], "o", color=S.OI[i], ms=ms,
                 label=rf"$\gamma={g:g}$")
for i, t in enumerate(TS):
    cs = sorted([c for c in C if c["T"] == t], key=lambda c: c["gamma"])
    gg = [c["gamma"] for c in cs]
    ax[1].loglog(gg, [c["D_pred"] for c in cs], "-", color=S.OI[i], lw=lw)
    ax[1].loglog(gg, [c["D_hat"] for c in cs], "o", color=S.OI[i], ms=ms, label=r"$T=$" + sci(t))
r = np.array([c["D_ratio"] for c in C]); e = np.array([c["D_ratio_sem"] for c in C])
# band: measured on the original PNG it is centred on 1 (not on the mean) with
# half-width 0.0225 +- 0.0005, i.e. the sample std (ddof=1) 0.0223 of the 25 ratios
sd = r.std(ddof=1)
ax[2].axhspan(1.0 - sd, 1.0 + sd, color="#E6E6E6", lw=0, zorder=0)
ax[2].axhline(1.0, color="k", lw=S.pt(0.6), zorder=1)
ax[2].errorbar(np.arange(len(r)), r, yerr=e, fmt="o", color=S.OI[0], ms=S.pt(2.6),
               elinewidth=S.pt(0.7), capsize=S.pt(1.2), capthick=S.pt(0.7), zorder=3)

ax[0].set_xlabel(r"$T$"); ax[0].set_ylabel(r"$D_\theta$")
ax[1].set_xlabel(r"$\gamma$"); ax[1].set_ylabel(r"$D_\theta$")
ax[2].set_xlabel("cell index"); ax[2].set_ylabel(r"$\hat D_\theta/D_\theta^{\rm pred}$")
ax[2].set_ylim(0.9, 1.1)
ax[0].set_ylim(1e-6, 0.11667); ax[1].set_ylim(1e-6, 0.11667)
ax[0].legend(loc="lower right"); ax[1].legend(loc="lower left", ncol=2, columnspacing=0.8)
S.letters(fig, ax)
finish(fig, "figB_dlaw.png", DPI)
