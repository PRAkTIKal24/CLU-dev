"""figs-round2 / V1 — fig4_bibo.png in the house style.
Data calls are those of orig_fig4_regen.py (= figs-regen/_work/regen_fig4.py, which
reproduces the shipped PNG byte-for-byte); only text, colours, sizes and layout change."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from house import Style, OKABE, TEXTWIDTH_IN, occlusion
import matplotlib.pyplot as plt

SRC = os.path.join(HERE, "..", "data", "paid_access_metrics.json")
OUT = os.environ.get("REPLOT_OUT", os.path.join(HERE, "new_renders"))
M = json.load(open(SRC))

FIGSIZE, DPI = (4.42, 3.84), 400.0                 # = shipped PNG: 1768 x 1536 px
S = Style(0.68 * TEXTWIDTH_IN / FIGSIZE[0])        # \includegraphics[width=0.68\linewidth]
B, V, G, P, O, SKY, K, GR = OKABE

b = M["certificate_payoff"]["bibo"]; bd = b["exit_distances"]; r2 = b["r_star_2T"]
fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
ax.plot(bd, [r2["wormhole_blind"][f"{v}"] for v in bd], "s-", color=B, ms=S.ms * 1.5,
        markerfacecolor="none", markeredgewidth=S.thin * 1.2, label="gated channel, screen ignored")
ax.plot(bd, [r2["no_physics_router"][f"{v}"] for v in bd], "x--", color=V, ms=S.ms,
        markeredgewidth=S.thin * 1.2, label="state-replacing map")
ax.plot(bd, [r2["wormhole_certified"][f"{v}"] for v in bd], "o-", color=G, ms=S.ms,
        markeredgewidth=0, label="gated channel + screen")
ax.axvline(b["x_b"], color=K, ls="--", lw=S.thin, label="coercive edge $x_b$")
ax.axhline(b["escape_radius"], color=GR, ls=":", lw=S.thin * 1.3, label="escape radius")
ax.set_yscale("log")
ax.grid(which="major", axis="y", color="0.92", lw=S.thin * 0.7)
ax.set_axisbelow(True)
ax.set_xlabel("requested exit distance $b$")
ax.set_ylabel(r"$r^\ast = \max_t \|q_t\|$")
leg = ax.legend(loc="lower left", borderaxespad=0.2)
fig.tight_layout(pad=0.3)
from fitcheck import unclip
unclip(fig)
fig.savefig(f"{OUT}/fig4_bibo.png", dpi=DPI)
print("wrote fig4_bibo.png")
r = fig.canvas.get_renderer(); A, Lg = ax.get_window_extent(r), leg.get_window_extent(r)
print("legend inside axes:", Lg.x0 >= A.x0 and Lg.x1 <= A.x1 and Lg.y0 >= A.y0 and Lg.y1 <= A.y1)
print("occlusion:", occlusion(fig, ax, leg))
