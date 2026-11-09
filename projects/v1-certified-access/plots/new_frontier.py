"""figs-round2 / V1 — fig_frontier_clean.png in the house style, RE-PLOTTED FROM TABLES.
The run data (.claude/scratch/regime-remap-2000ep/runs/*.json) and the shipped generator
(.claude/scratch/v1-figure-fix/fix_frontier.py) are lost.
  means : exact float64 values recovered by frontier_exact.py from the 6-decimal table in
          .claude/outputs/v1-replot-pass.md section 3; their tap hashes equal the banked
          digest of the shipped figure (e02915ed..., 8/8 calls).
  errors: the table's 6-decimal population std (|error| <= 5e-7 by rounding)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
from house import Style, OKABE, TEXTWIDTH_IN, occlusion
from frontier_table import table, EPOCHS
import matplotlib.pyplot as plt

OUT = os.environ.get("REPLOT_OUT", os.path.join(HERE, "new_renders"))
FIGSIZE, DPI = (5.33, 2.20), 400.0                 # = shipped PNG: 2132 x 880 px
S = Style(0.82 * TEXTWIDTH_IN / FIGSIZE[0], ms=3.2)  # \includegraphics[width=0.82\linewidth]

T = table()
EX = np.load(os.path.join(HERE, "..", "data", "frontier_exact.npy"), allow_pickle=True).item()
D = {kv: dict(fid=EX[f"{kv}_fid"][0], gate=EX[f"{kv}_gate"][0],
              fid_err=T[kv]["fid_err"], gate_err=T[kv]["gate_err"]) for kv in (32, 64, 96, 128)}
for kv in D:  # every plotted mean must equal the table to its 6 printed decimals
    assert np.all(np.round(D[kv]["fid"], 6) == T[kv]["fid"]) and np.all(np.round(D[kv]["gate"], 6) == T[kv]["gate"])

fig, ax = plt.subplots(1, 2, figsize=FIGSIZE, dpi=DPI)
for c, kv in zip(OKABE, (32, 64, 96, 128)):
    d = D[kv]
    ax[0].errorbar(EPOCHS, d["fid"], yerr=d["fid_err"], marker="o", color=c, ms=S.ms,
                   elinewidth=S.thin, capthick=S.thin, capsize=2.0 / S.s, markeredgewidth=0, label=f"kv{kv}")
    ax[1].errorbar(EPOCHS, d["gate"], yerr=d["gate_err"], marker="s", color=c, ms=S.ms,
                   elinewidth=S.thin, capthick=S.thin, capsize=2.0 / S.s, markeredgewidth=0, label=f"kv{kv}")
ax[0].set(xlabel="train epochs", ylabel="storage fidelity")
ax[1].set(xlabel="train epochs", ylabel="gated accuracy")
for a in ax:
    a.set_ylim(0, 1.05); a.set_xticks(EPOCHS); a.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
h, l = ax[0].get_legend_handles_labels()
leg = ax[0].legend([x[0] for x in h], l, loc="lower right", borderaxespad=0.2)  # line+marker handles only
fig.tight_layout(pad=0.3, w_pad=1.5, rect=(0, 0, 1, 0.96))
S.letter_(ax[0], "(a)"); S.letter_(ax[1], "(b)")
from fitcheck import unclip
unclip(fig)
fig.savefig(f"{OUT}/fig_frontier_clean.png", dpi=DPI)
print("wrote fig_frontier_clean.png")
r = fig.canvas.get_renderer(); A, Lg = ax[0].get_window_extent(r), leg.get_window_extent(r)
print("legend inside axes:", Lg.x0 >= A.x0 and Lg.x1 <= A.x1 and Lg.y0 >= A.y0 and Lg.y1 <= A.y1)
print("occlusion:", occlusion(fig, ax[0], leg))

# independent check: read every error-bar segment back from the drawn artists and compare
# it with mean +- std straight from the 6-decimal table
from matplotlib.collections import LineCollection
worst = 0.0; nseg = 0
for a, key in ((ax[0], "fid"), (ax[1], "gate")):
    lcs = [c for c in a.collections if isinstance(c, LineCollection)]
    for kv, lc in zip((32, 64, 96, 128), lcs):
        for e, seg in zip(EPOCHS, lc.get_segments()):
            i = EPOCHS.index(e); seg = np.asarray(seg)
            assert seg[0, 0] == e and seg[1, 0] == e
            lo, hi = sorted(seg[:, 1])
            m, s = T[kv][key][i], T[kv][key + "_err"][i]
            worst = max(worst, abs(lo - (m - s)), abs(hi - (m + s))); nseg += 1
print(f"error-bar segments checked: {nseg}, max |drawn - table| = {worst:.2e}")
