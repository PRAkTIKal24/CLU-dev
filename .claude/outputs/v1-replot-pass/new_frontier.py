"""V1 replot pass — fig_frontier_clean.png, re-rendered at PRINTED size with the
Hopfield band REMOVED (the paper no longer makes a Hopfield comparison).

Data: re-aggregated from .claude/scratch/regime-remap-2000ep/runs/*.json exactly as
the recovered original generator does (orig_frontier.py reproduces the banked PNG
byte-for-byte, md5 bcc5f32dcd85e01740638c6608f26320).  n = 3 seeds {42,43,44}, ne3.
"""
import os, sys
S = "/Users/user/Desktop/CHLU/.claude/scratch/v1-replot-pass"
sys.path.insert(0, S)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from frontier_data import load, EPOCHS

OUT = os.environ.get("REPLOT_OUT", os.path.join(S, "new_renders"))

FIGSIZE = (4.68000, 1.80002)   # 1872 x 720 px at dpi 400 = banked aspect 1560:600 = 13:5 EXACTLY
DPI = 400.0                    # printed box 4.6800 in wide -> type scale 1.00000
TITLE, LABEL, TICK, LEG = 9.0, 8.0, 7.0, 8.0

plt.rcParams.update({
    "font.size": LABEL, "axes.titlesize": TITLE, "axes.labelsize": LABEL,
    "xtick.labelsize": TICK, "ytick.labelsize": TICK, "legend.fontsize": LEG,
    "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
})

D, _band = load()
fig, ax = plt.subplots(1, 2, figsize=FIGSIZE, dpi=DPI)
for kv in (32, 64, 96, 128):
    d = D[kv]
    ax[0].errorbar(EPOCHS, d["fid"], yerr=d["fid_err"], marker="o", capsize=2,
                   lw=1.1, ms=3.2, elinewidth=0.9, label=f"kv{kv}")
    ax[1].errorbar(EPOCHS, d["gate"], yerr=d["gate_err"], marker="s", capsize=2,
                   lw=1.1, ms=3.2, elinewidth=0.9, label=f"kv{kv}")
ax[0].set(xlabel="train epochs", ylabel="CLU-EBM storage\nfidelity",
          title="Fidelity vs epochs\n(corr=0, ne3, 3 seeds)")
ax[1].set(xlabel="train epochs", ylabel="CLU gated\naccuracy",
          title="Gated acc vs epochs")
for a in ax:
    a.grid(alpha=0.3, lw=0.4)
    a.legend(loc="lower right", ncol=1, handlelength=1.1, handletextpad=0.4,
             borderpad=0.25, labelspacing=0.15, borderaxespad=0.35, framealpha=0.95)
    a.set_ylim(0, 1.05)
    a.set_xticks(EPOCHS)
fig.tight_layout(pad=0.35, w_pad=1.0)
for a in ax:                      # drop the crowded "500" label onto a second row
    a.get_xticklabels()[0].set_y(-0.095)
    a.xaxis.labelpad = 7.0
from fitcheck import unclip
unclip(fig)
fig.savefig(f"{OUT}/fig_frontier_clean.png", dpi=DPI)
print("wrote fig_frontier_clean.png")

from fitcheck import report
fig.canvas.draw()
print("fit @ printed width 4.6800 in:")
report(fig, 4.6800)
