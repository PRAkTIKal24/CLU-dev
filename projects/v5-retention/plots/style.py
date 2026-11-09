"""House style for the round-2 figure pass (shared by all V5 figures).

Every size below is a PRINTED size in pt. `setup(fig_w_in, printed_w_in)` converts
it to in-figure pt with k = fig_w_in / printed_w_in, so text prints at 7-8 pt and
lines at ~1.3 pt whatever canvas the figure must keep.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OI = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000", "#7F7F7F"]
GREY = "#7F7F7F"

TICK, LABEL, LEGEND, LETTER = 7.0, 8.0, 7.0, 8.0
LW, MS, SPINE = 1.3, 3.5, 0.6

K = 1.0


def setup(fig_w_in, printed_w_in):
    """Set rcParams for a canvas `fig_w_in` wide that LaTeX prints `printed_w_in` wide."""
    global K
    K = fig_w_in / printed_w_in
    k = K
    plt.rcdefaults()
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.weight": "normal",
        "font.size": LABEL * k,
        "axes.labelsize": LABEL * k, "axes.titlesize": LETTER * k,
        "axes.labelweight": "normal", "axes.titleweight": "normal",
        "xtick.labelsize": TICK * k, "ytick.labelsize": TICK * k,
        "legend.fontsize": LEGEND * k,
        "axes.linewidth": SPINE * k,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False,
        "axes.prop_cycle": matplotlib.cycler(color=OI),
        "lines.linewidth": LW * k, "lines.markersize": MS * k,
        "lines.markeredgewidth": 0.8 * k,
        "errorbar.capsize": 1.5 * k,
        "patch.linewidth": 0.0,
        "hatch.linewidth": 0.8 * k,
        "xtick.major.width": SPINE * k, "ytick.major.width": SPINE * k,
        "xtick.minor.width": 0.45 * k, "ytick.minor.width": 0.45 * k,
        "xtick.major.size": 2.5 * k, "ytick.major.size": 2.5 * k,
        "xtick.minor.size": 1.4 * k, "ytick.minor.size": 1.4 * k,
        "xtick.major.pad": 1.8 * k, "ytick.major.pad": 1.8 * k,
        "axes.labelpad": 2.0 * k,
        "legend.frameon": False,
        "legend.borderpad": 0.2, "legend.labelspacing": 0.25,
        "legend.handlelength": 1.6, "legend.handletextpad": 0.45,
        "legend.borderaxespad": 0.3, "legend.columnspacing": 1.0,
        "grid.linewidth": 0.3 * k, "grid.color": "#DDDDDD",
        "savefig.facecolor": "white",
    })
    return k


def pt(x):
    """printed pt -> in-figure pt"""
    return x * K


def letter(ax, s, x=None, y=1.0):
    """small regular-weight panel letter at the top-left, just outside the axes."""
    if x is None:
        x = 0.0
    ax.annotate(f"({s})", xy=(x, y), xycoords="axes fraction",
                xytext=(-pt(LETTER) * 1.9, pt(2.0)), textcoords="offset points",
                ha="left", va="bottom", fontsize=pt(LETTER), fontweight="normal")


def save(fig, path, dpi):
    fig.savefig(path, dpi=dpi, metadata={"Software": None})


def letters(fig, axes, labels="abcdefgh"):
    """Panel letters '(a)'.. as left-loc titles, x aligned with the left edge of each
    panel's y-label, so they never collide with tick labels. Works with constrained
    layout (titles are part of the layout)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    for a, l in zip(axes, labels):
        bb = a.get_tightbbox(r)
        ab = a.get_window_extent(r)
        x = (bb.x0 - ab.x0) / ab.width
        a.set_title(f"({l})", loc="left", x=x, pad=pt(3.0), fontsize=pt(LETTER), fontweight="normal")
    fig.canvas.draw()
    # keep every letter inside the canvas (constrained layout does not see the x offset)
    fig.set_layout_engine("none")           # freeze the solved layout (savefig must not re-solve)
    margin = 1.5 * fig.dpi / 72.0          # 1.5 pt
    for _ in range(6):                      # layout re-solves after each nudge
        moved = False
        for a in axes:
            t = a._left_title
            e = t.get_window_extent(r)
            if e.x0 < margin - 0.1:
                ab = a.get_window_extent(r)
                t.set_x(t.get_position()[0] + (margin - e.x0) / ab.width); moved = True
        fig.canvas.draw()
        if not moved:
            break
