"""NEW generator (original lost: scratch/v5-revision-1/regen_figs.py over the v5-gate
harness) for figC_register_capacity. Data: outputs/v5-gate/e1_jacobian.json, key
'decay': written coset value delta in {0.1,0.3,0.5}, T=0, gamma=0.05, 20k steps;
retained value = |rel_final| (v5-gate.md s3.3 "Retained value |theta(N)|").
Run from the repo root."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style as S
from lostcommon import VG, finish
import matplotlib.pyplot as plt

PX = (840, 660); DPI = 150.0; FIGSIZE = (PX[0] / DPI, PX[1] / DPI)
S.setup(FIGSIZE[0], 5.5)
J = json.load(open(os.path.join(VG, "e1_jacobian.json")))
TAGS = [("emergent150_s42", "emergent 42", S.OI[0], "o", 4.2),
        ("emergent150_s43", "emergent 43", S.OI[1], "s", 3.2),
        ("emergent150_s44", "emergent 44", S.OI[2], "^", 2.4),
        ("designed150_s44", "designed 44", "k", "o", 3.5)]
GAMMA = 0.05

fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
fig.get_layout_engine().set(w_pad=S.pt(2)/72, h_pad=S.pt(2)/72, wspace=0.04)
for tag, lab, c, m, ms in TAGS:
    rows = sorted([r for r in J[tag]["decay"] if r["gamma"] == GAMMA], key=lambda r: r["delta"])
    x = np.array([r["delta"] for r in rows]); y = np.abs([r["rel_final"] for r in rows])
    filled = tag.startswith("designed")
    ax.plot(x, y, "-", marker=m, color=c, ms=S.pt(ms), lw=S.pt(1.0),
            mfc=c if filled else "white", mew=S.pt(0.8), label=lab, zorder=3)
ax.plot([0, 0.55], [0, 0.55], "--", color=S.GREY, lw=S.pt(0.7), zorder=1)
ax.set_xlabel(r"written coset value $\delta$ (rad)")
ax.set_ylabel(r"retained value (rad)")
ax.set_xlim(-0.028, 0.578); ax.set_ylim(-0.031, 0.561)
ax.legend(loc="upper left")
finish(fig, "figC_register_capacity.png", DPI)
