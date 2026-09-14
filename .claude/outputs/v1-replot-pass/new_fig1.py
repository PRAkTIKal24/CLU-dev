"""V1 replot pass — fig1_certificate.png, re-rendered at PRINTED size.

Data: identical to .claude/scratch/v1-revision-2/make_figs.py (byte-identical
re-run verified).  Only text, type size and layout change.
"""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SRC = "/Users/user/Desktop/CHLU/.claude/outputs/v1-certificate-payoff/paid_access_metrics.json"
OUT = os.environ.get("REPLOT_OUT", "/Users/user/Desktop/CHLU/.claude/scratch/v1-replot-pass/new_renders")
M = json.load(open(SRC))

FIGSIZE = (6.51000, 2.36252)  # 2604 x 945 px at dpi 400 = banked aspect 2480:900 EXACTLY
DPI = 400.0                  # printed box 6.5001 in wide -> type scale 0.99847
TITLE, LABEL, TICK, LEG = 9.02, 8.02, 7.02, 8.02   # +0.02 pt so the 0.99848
                                                  # canvas->print scale still clears 9/8/7

plt.rcParams.update({
    "font.size": LABEL, "axes.titlesize": TITLE, "axes.labelsize": LABEL,
    "xtick.labelsize": TICK, "ytick.labelsize": TICK, "legend.fontsize": LEG,
    "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 1.0,
})

d = M["reach"]["distances"]
L = M["reach"]["L"]
lr = M["reach"]["landing_rates"]

arms = [
    ("wormhole",          "wormhole  (det J = 1, $\\Delta V$ = 0)",  "#2ca02c", "o", "-",  +0.024),
    ("no_physics_router", "state-replacing map  (det J = 0)",        "#9467bd", "X", "--", +0.008),
    ("newtonian_squeeze", "Newtonian squeeze (control, det J = 1)",  "#d62728", "^", ":",  -0.008),
    ("squeeze",           "squeeze $S^{(M)}$  (det J = 1)",          "#ff7f0e", "s", "-",  -0.024),
    ("throat_denseV",     "dense / throat-$V$  (no jump)",           "#8c564b", "v", "-",   0.0),
    ("plain_relax",       "plain relaxation",                        "#1f77b4", "d", "-",   0.0),
]

fig, (ax, bx) = plt.subplots(1, 2, figsize=FIGSIZE, dpi=DPI)

for key, lab, col, mk, ls, off in arms:
    y = np.array([lr[key][f"{x}"] for x in d], dtype=float)
    ax.plot(d, y + off, marker=mk, ls=ls, color=col, label=lab, lw=1.0, ms=3.0,
            alpha=0.95, markeredgecolor="white", markeredgewidth=0.35)

ax.axvline(L, color="k", ls="--", lw=0.8)
ax.text(L - 0.05, 0.50, "causal box\n$L = 2.5$", rotation=90, fontsize=7.02,
        va="center", ha="right", ma="center", linespacing=1.0)
ax.axvspan(L, 3.4, color="k", alpha=0.06)
ax.text(2.93, 0.70, "crossover\nbracket", fontsize=7.02, ha="center", va="center", color="0.35")
ax.annotate("squeeze: beyond\nthe swept $\\zeta \\leq 2.0$",
            xy=(4.0, -0.024), xytext=(4.20, 0.22),
            fontsize=7.02, color="#ff7f0e", ha="center",
            arrowprops=dict(arrowstyle="->", color="#ff7f0e", lw=0.8))
ax.set_xlabel("basin distance $d$")
ax.set_ylabel("landing rate  (5 seeds)")
ax.set_title("(a) Landing rate vs. basin distance")
ax.set_ylim(-0.12, 2.50)
ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.legend(loc="upper center", ncol=1, framealpha=0.95, bbox_to_anchor=(0.5, 1.015),
          handlelength=1.3, handletextpad=0.45, borderpad=0.28, labelspacing=0.22,
          borderaxespad=0.15)
ax.grid(alpha=0.25, lw=0.5)

la = M["certificate_payoff"]["latch"]["arms"]
qin_std = M["certificate_payoff"]["latch"]["Q_in_std"]

spec = [
    ("wormhole_coset_tangent", "wormhole, coset-tangent (det J=1)",   "#2ca02c", "o"),
    ("wormhole_across_coset",  "wormhole, across-coset (det J=1)",    "#1f77b4", "s"),
    ("random_shift",           "random shift (det J=1, no channel)",  "#ff7f0e", "^"),
    ("no_physics_router",      "state-replacing map (det J = 0)",     "#9467bd", "X"),
]
for key, lab, col, mk in spec:
    qi = np.array(la[key]["Q_in"])
    qo = np.array(la[key]["Q_out"])
    bx.plot(qi, qo, mk, color=col, label=lab, ms=3.0, alpha=0.9,
            markeredgecolor="white", markeredgewidth=0.3)

lo, hi = 1.35, 1.63
bx.plot([lo, hi], [lo, hi], "k--", lw=0.7, label="identity ($Q$ preserved)")
bx.set_xlabel(r"incoming charge  $Q_{\rm in} = p^\top X q$")
bx.set_ylabel(r"outgoing charge  $Q_{\rm out}$")
bx.set_title("(b) Goldstone charge: transported vs. erased")
bx.legend(loc="lower right", framealpha=0.95, handlelength=1.0, handletextpad=0.4,
          borderpad=0.28, labelspacing=0.22, borderaxespad=0.25)
bx.set_ylim(0.55, 2.30)
bx.grid(alpha=0.25, lw=0.5)
bx.text(0.03, 0.965,
        f"std$(Q_{{\\rm in}})$ = {qin_std:.4f}\n"
        f"wormhole: std$(Q_{{\\rm out}})$ = {la['wormhole_across_coset']['Q_out_std']:.4f}  (transport)\n"
        f"state-replacing: std$(Q_{{\\rm out}})$ = {la['no_physics_router']['Q_out_std']:.1f}  (erasure)",
        transform=bx.transAxes, fontsize=7.02, va="top", family="monospace",
        bbox=dict(boxstyle="round,pad=0.25", fc="#f7f7f7", ec="0.7", lw=0.5))

fig.tight_layout(pad=0.35, w_pad=0.8)
sys.path.insert(0, "/Users/user/Desktop/CHLU/.claude/scratch/v1-replot-pass")
from fitcheck import unclip
unclip(fig)
fig.savefig(f"{OUT}/fig1_certificate.png", dpi=DPI)
print("wrote fig1_certificate.png")

# --- fit diagnostics (printed points) ---
sys.path.insert(0, "/Users/user/Desktop/CHLU/.claude/scratch/v1-replot-pass")
from fitcheck import report
fig.canvas.draw()
print("fit @ printed width 6.5001 in:")
report(fig, 6.5001)
