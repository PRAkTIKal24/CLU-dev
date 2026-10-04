"""The PW-1 rendered grid, sized for a 5.5 in text width (fonts 6-7 pt at print size).

One row per request, one column per model; the lookup-table floor is set apart by a gap.
Inside a panel: the image and the judge's verdict glyph only (check = valid, cross = invalid;
glyph shape carries the verdict, colour is redundant).  Text is ink, never the model colour:
a model's colour appears only as the bar under its column name and the bar of its row group.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from w17_rendered_compositions.figure import INVALID, VALID, _to_hwc  # noqa: E402
matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "dejavusans",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.hashsalt": "pw1-render-shapes",
    "path.simplify": False,
})
INK, MUTED, RULE = "#1f1f1d", "#5f5e5a", "#9a9994"
P = 0.64          # inches per panel
GAP = 0.05        # inches between panels
FLOOR_GAP = 0.14  # extra inches before the floor column
def draw_grid(out_stem: Path, col_labels, col_sub, col_colours, row_labels, images, verdicts,
              row_groups=None, highlight=None, floor_col=None, left=1.30) -> list:
    nr, nc = len(row_labels), len(col_labels)
    highlight = highlight or {}
    gl = 0.26 if row_groups else 0.0
    left = left + gl
    top = 0.64
    xs = [left + c * P + (FLOOR_GAP if floor_col is not None and c >= floor_col else 0.0)
          for c in range(nc)]
    fig_w = xs[-1] + P + 0.04
    fig_h = top + nr * P + 0.04
    fig = plt.figure(figsize=(fig_w, fig_h))
    fx = lambda x: x / fig_w          # noqa: E731
    fy = lambda y: 1.0 - y / fig_h    # noqa: E731
    for r in range(nr):
        for c in range(nc):
            ax = fig.add_axes([fx(xs[c] + GAP / 2), fy(top + (r + 1) * P - GAP / 2),
                               (P - GAP) / fig_w, (P - GAP) / fig_h])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(_to_hwc(images[r][c]), interpolation="nearest")
            hc = highlight.get((r, c))
            for s in ax.spines.values():
                s.set_edgecolor(hc or RULE)
                s.set_linewidth(2.0 if hc else 0.5)
            ch, colr = VALID if verdicts[r][c] else INVALID
            ax.text(0.95, 0.95, ch, transform=ax.transAxes, ha="right", va="top",
                    fontsize=7.5, fontweight="bold", color="white",
                    bbox=dict(boxstyle="circle,pad=0.10", fc=colr, ec="white", lw=0.6))
        fig.text(fx(left - 0.06), fy(top + (r + 0.5) * P), row_labels[r], ha="right",
                 va="center", fontsize=5.4, linespacing=1.22, color=INK)
    for c in range(nc):
        xc = fx(xs[c] + P / 2)
        fig.text(xc, fy(top - 0.20), col_labels[c], ha="center", va="bottom", fontsize=6.4,
                 fontweight="bold", color=INK, linespacing=1.05)
        if col_sub[c]:
            fig.text(xc, fy(top - 0.185), col_sub[c], ha="center", va="top", fontsize=4.9,
                     color=MUTED)
        fig.add_artist(matplotlib.lines.Line2D(
            [fx(xs[c] + 0.08), fx(xs[c] + P - 0.08)], [fy(top - 0.035)] * 2,
            color=col_colours[c], lw=2.0, solid_capstyle="round"))
    if row_groups:
        for lab, colr, a, b in row_groups:
            ya, yb = fy(top + a * P + 0.06), fy(top + (b + 1) * P - 0.06)
            x = fx(0.20)
            fig.add_artist(matplotlib.lines.Line2D([x, x], [ya, yb], color=colr, lw=2.0,
                                                   solid_capstyle="round"))
            if not lab:
                continue
            fig.text(fx(0.13), (ya + yb) / 2, lab, ha="right", va="center", rotation=90,
                     fontsize=6.2, fontweight="bold", color=INK)
    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext, kw in (("pdf", {"metadata": {"CreationDate": None, "ModDate": None}}),
                    ("png", {"dpi": 300, "metadata": {"Software": None}})):
        p = out_stem.with_suffix("." + ext)
        fig.savefig(p, **kw)
        paths.append(p.name)
    plt.close(fig)
    return paths


