"""House style shared by the three workshop papers (figs-round2).
Sizes are given in PRINTED points and converted to canvas points with the
canvas->print scale s = printed width / canvas width."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OKABE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000", "#7F7F7F"]
TEXTWIDTH_IN = 5.5   # neurips_2026.sty: \newgeometry{textwidth=5.5in}


class Style:
    def __init__(self, scale, tick=7.0, label=8.0, legend=7.0, letter=8.0, lw=1.3, ms=3.5):
        self.s = scale
        c = lambda pt: pt / scale
        self.tick, self.label, self.legend, self.letter = c(tick), c(label), c(legend), c(letter)
        self.lw, self.ms = c(lw), c(ms)
        self.thin = c(0.7)       # reference lines, spines
        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.weight": "normal", "axes.titleweight": "normal", "axes.labelweight": "normal",
            "font.size": self.label, "axes.labelsize": self.label,
            "xtick.labelsize": self.tick, "ytick.labelsize": self.tick,
            "legend.fontsize": self.legend, "legend.frameon": False,
            "axes.spines.top": False, "axes.spines.right": False,
            "axes.linewidth": c(0.6), "axes.grid": False,
            "xtick.major.width": c(0.6), "ytick.major.width": c(0.6),
            "xtick.minor.width": c(0.45), "ytick.minor.width": c(0.45),
            "xtick.major.size": c(2.5), "ytick.major.size": c(2.5),
            "xtick.minor.size": c(1.4), "ytick.minor.size": c(1.4),
            "xtick.major.pad": c(2.0), "ytick.major.pad": c(2.0),
            "axes.labelpad": c(2.5),
            "lines.linewidth": self.lw, "lines.markersize": self.ms,
            "errorbar.capsize": c(1.5),
            "legend.handlelength": 1.6, "legend.handletextpad": 0.4,
            "legend.labelspacing": 0.25, "legend.borderaxespad": 0.3,
            "legend.columnspacing": 0.9, "legend.borderpad": 0.2,
            "savefig.dpi": 400,
        })

    def letter_(self, ax, s):
        """Small regular-weight panel letter at the top-left of the axes."""
        ax.annotate(s, xy=(0, 1), xycoords="axes fraction", xytext=(-self.letter * 2.6, 0),
                    textcoords="offset points", ha="left", va="bottom", fontsize=self.letter)

    def opaque_legend(self, leg):
        """Visually frameless legend that still masks reference lines running behind it."""
        fr = leg.get_frame()
        fr.set_facecolor("white"); fr.set_edgecolor("none"); fr.set_alpha(1.0)
        leg.set_frame_on(True)
        return leg


def occlusion(fig, ax, leg, n=400):
    """Count sampled points of every data Line2D (and errorbar segment) of `ax` that fall
    inside the legend's bbox (display coords).  Returns {label: count}."""
    fig.canvas.draw()
    bb = leg.get_window_extent(fig.canvas.get_renderer())
    out = {}
    from matplotlib.collections import LineCollection
    arts = list(ax.lines) + [c for c in ax.collections if isinstance(c, LineCollection)]
    for idx, a in enumerate(arts):
        segs = []
        if hasattr(a, "get_xydata") and not isinstance(a, LineCollection):
            xy = a.get_xydata()
            if a.get_transform() != ax.transData:   # axhline/axvline: blended transforms
                pts = a.get_transform().transform(xy)
            else:
                pts = ax.transData.transform(xy)
            segs = [pts]
        else:
            segs = [ax.transData.transform(np.asarray(s)) for s in a.get_segments()]
        cnt = 0
        for pts in segs:
            pts = np.asarray(pts, float)
            if len(pts) == 1:
                sm = pts
            else:
                sm = np.concatenate([np.linspace(pts[i], pts[i + 1], n) for i in range(len(pts) - 1)])
            cnt += int(((sm[:, 0] >= bb.x0) & (sm[:, 0] <= bb.x1) & (sm[:, 1] >= bb.y0) & (sm[:, 1] <= bb.y1)).sum())
        out[f"{idx}:{a.get_label()}"] = cnt
    return out
