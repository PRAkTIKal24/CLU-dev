"""figs-round2 / V1 — fig1_certificate.png in the house style.
Data calls are those of orig_fig1_gc.py (which reproduces the shipped PNG byte-for-byte):
same arrays, same per-arm offsets; only text, colours, sizes and layout change."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
from house import Style, OKABE, TEXTWIDTH_IN, occlusion
import matplotlib.pyplot as plt

SRC = os.path.join(HERE, "..", "data", "paid_access_metrics.json")
OUT = os.environ.get("REPLOT_OUT", os.path.join(HERE, "new_renders"))
M = json.load(open(SRC))

FIGSIZE, DPI = (6.51000, 2.36252), 400.0          # = shipped PNG: 2604 x 945 px
S = Style(0.82 * TEXTWIDTH_IN / FIGSIZE[0], ms=3.0)        # \includegraphics[width=0.82\textwidth]
B, V, G, P, O, SKY, K, GR = OKABE

d = M["reach"]["distances"]; L = M["reach"]["L"]; lr = M["reach"]["landing_rates"]
arms = [  # key, legend, colour, marker, ls, offset (offsets unchanged: they are in the plotted y)
    ("wormhole",          "gated channel",            B,   "o", "-",  +0.024),
    ("no_physics_router", "state-replacing map",      V,   "X", "--", +0.008),
    ("newtonian_squeeze", "Newtonian squeeze",        G,   "^", ":",  -0.008),
    ("squeeze",           "squeeze $S^{(M)}$",        P,   "s", "-",  -0.024),
    ("throat_denseV",     "dense / throat-$V$",       O,   "v", "-",   0.0),
    ("plain_relax",       "plain relaxation",         SKY, "d", "-",   0.0),
]
fig, (ax, bx) = plt.subplots(1, 2, figsize=FIGSIZE, dpi=DPI, gridspec_kw=dict(width_ratios=[1.12, 1.0]))
for key, lab, col, mk, ls, off in arms:
    y = np.array([lr[key][f"{x}"] for x in d], dtype=float)
    ax.plot(d, y + off, marker=mk, ls=ls, color=col, label=lab, ms=S.ms, markeredgewidth=0)
ax.set_xlabel("basin distance $d$")
ax.set_ylabel("landing rate")
ax.set_ylim(-0.08, 1.62)
ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
# reference marks drawn from the axis floor up to y = 1.05 (just above the data) so they do
# not run through the legend; position/extent along x is unchanged (tap: same positional args)
YTOP = (1.05 - (-0.08)) / (1.62 - (-0.08))
ax.axvline(L, ymax=YTOP, color=K, ls="--", lw=S.thin, zorder=0.5)
ax.axvspan(L, 3.4, ymax=YTOP, color=GR, alpha=0.15, lw=0, zorder=0.4)
la_ = ax.legend(loc="upper left", ncol=2, bbox_to_anchor=(0.0, 1.0), handlelength=1.15,
                handletextpad=0.3, columnspacing=0.5, borderaxespad=0.1)

la = M["certificate_payoff"]["latch"]["arms"]
spec = [  # plotting/legend order chosen so 'state-replacing map' is the same colour in every figure
    ("wormhole_coset_tangent", "coset-tangent",        B, "o"),
    ("no_physics_router",      "state-replacing map",  V, "X"),
    ("wormhole_across_coset",  "across-coset",         G, "s"),
    ("random_shift",           "random shift",         P, "^"),
]
for key, lab, col, mk in spec:
    bx.plot(np.array(la[key]["Q_in"]), np.array(la[key]["Q_out"]), mk, color=col, label=lab,
            ms=S.ms, markeredgewidth=0)
lo, hi = 1.35, 1.63
bx.plot([lo, hi], [lo, hi], color=K, ls="--", lw=S.thin, label="identity")
bx.set_xlabel(r"incoming charge $Q_{\rm in} = p^\top X q$")
bx.set_ylabel(r"outgoing charge $Q_{\rm out}$")
bx.set_ylim(1.02, 2.50)
bx.set_yticks([1.25, 1.50, 1.75, 2.00])
h, l = bx.get_legend_handles_labels()
ordr = [0, 2, 3, 1, 4]   # column 1: the three det J = 1 maps; column 2: state-replacing map, identity
lb_ = bx.legend([h[i] for i in ordr], [l[i] for i in ordr], loc="upper left", ncol=2,
                bbox_to_anchor=(0.0, 1.0), handlelength=0.9, handletextpad=0.2, columnspacing=0.35,
                borderaxespad=0.1)

fig.tight_layout(pad=0.3, w_pad=1.2, rect=(0, 0, 1, 0.97))
S.letter_(ax, "(a)"); S.letter_(bx, "(b)")
from fitcheck import unclip
unclip(fig)
fig.savefig(f"{OUT}/fig1_certificate.png", dpi=DPI)
print("wrote fig1_certificate.png")
r = fig.canvas.get_renderer()
for a_, l_ in ((ax, la_), (bx, lb_)):
    A, Lg = a_.get_window_extent(r), l_.get_window_extent(r)
    print("legend inside axes:", Lg.x0 >= A.x0 and Lg.x1 <= A.x1 and Lg.y1 <= A.y1 + 1, [round(v) for v in (Lg.x0, Lg.x1, A.x0, A.x1)])
print("occlusion (a):", occlusion(fig, ax, la_)); print("occlusion (b):", occlusion(fig, bx, lb_))
