"""V1 replot pass — fig4_bibo.png, re-rendered at PRINTED size.
Data identical to .claude/scratch/v1-revision-2/make_figs.py (byte-identical re-run verified).
"""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = "/Users/user/Desktop/CHLU/.claude/outputs/v1-certificate-payoff/paid_access_metrics.json"
OUT = os.environ.get("REPLOT_OUT", "/Users/user/Desktop/CHLU/.claude/scratch/v1-replot-pass/new_renders")
M = json.load(open(SRC))

FIGSIZE = (4.41750, 2.94502)   # 1767 x 1178 px at dpi 400 = banked aspect 1320:880 = 3:2 EXACTLY
DPI = 400.0                    # printed box 4.4201 in wide -> type scale 1.00058
TITLE, LABEL, TICK, LEG = 9.0, 8.0, 7.0, 8.0

plt.rcParams.update({
    "font.size": LABEL, "axes.titlesize": TITLE, "axes.labelsize": LABEL,
    "xtick.labelsize": TICK, "ytick.labelsize": TICK, "legend.fontsize": LEG,
    "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.minor.size": 1.4, "ytick.minor.size": 1.4,
})

b = M["certificate_payoff"]["bibo"]
bd = b["exit_distances"]
r2 = b["r_star_2T"]

fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
ax.plot(bd, [r2["wormhole_blind"][f"{v}"] for v in bd], "s-", color="#e377c2",
        lw=3.5, alpha=0.35, ms=4.0, label="wormhole, screen ignored (ablation)")
ax.plot(bd, [r2["no_physics_router"][f"{v}"] for v in bd], "X--", color="#9467bd",
        lw=1.0, ms=4.0, label="state-replacing map (coincides with ablation)")
ax.plot(bd, [r2["wormhole_certified"][f"{v}"] for v in bd], "o-", color="#2ca02c",
        lw=1.2, ms=4.0, label="wormhole + screen (refuses exit)")
ax.axvline(b["x_b"], color="k", ls="--", lw=0.9, label=f"coercive edge $x_b$ = {b['x_b']:.2f}")
ax.axhline(b["escape_radius"], color="0.5", ls=":", lw=0.9, label="escape radius")
ax.set_yscale("log")
ax.set_xlabel("requested exit distance  $b$")
ax.set_ylabel(r"$r^\ast = \max_t \|q_t\|$  over $2T$ steps")
ax.set_title("Maximum excursion radius vs. destination\nlocus of the wormhole jump", linespacing=1.15)
ax.legend(loc="center left", framealpha=0.95, handlelength=1.5, handletextpad=0.5,
          borderpad=0.3, labelspacing=0.25, borderaxespad=0.4)
ax.grid(alpha=0.25, which="both", lw=0.4)
ax.text(0.025, 0.045,
        "$b=5.0$: energy change $\\Delta H = 0$ (free)\n"
        "— the unscreened exit still escapes",
        transform=ax.transAxes, fontsize=7.0, ha="left", va="bottom", color="#b22222",
        linespacing=1.15)
fig.tight_layout(pad=0.35)
sys.path.insert(0, "/Users/user/Desktop/CHLU/.claude/scratch/v1-replot-pass")
from fitcheck import unclip
unclip(fig)
fig.savefig(f"{OUT}/fig4_bibo.png", dpi=DPI)
print("wrote fig4_bibo.png")

sys.path.insert(0, "/Users/user/Desktop/CHLU/.claude/scratch/v1-replot-pass")
from fitcheck import report
fig.canvas.draw()
print("fit @ printed width 4.4201 in:")
report(fig, 4.4201)
