"""The w19 grid: w17's layout (one row per request, one column per arm, arm names and
colours from figures/FIGURES.md above the columns, request label left of each row), with
three additions and NO title:

* ``row_groups``  -- (label, colour, first_row, last_row): a bracket at the far left naming
  the arm a pair of rows is about ("MLP" -> its success row and its failure row);
* ``highlight``   -- {(r, c): colour}: a thick frame in the subject arm's colour around the
  one panel the row was selected for;
* ``overlays``    -- {(r, c): fn(ax)}: vector marks drawn on top of an image (the CMAZE-T
  action arrow).  Graphics only, never text.

Inside a panel there is only the image and the judge's verdict mark (w17's check / cross).
A panel with no image is hatched grey (DECLARED not buildable; reason in STATUS/README).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from w17_rendered_compositions.figure import INVALID, VALID, _to_hwc  # noqa: E402

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.hashsalt": "w19-success-failure-panels",
    "path.simplify": False,
})
PANEL = 1.05


def draw_grid(out_stem: Path, col_labels, col_colours, row_labels, images, verdicts,
              col_groups=None, row_groups=None, highlight=None, overlays=None,
              left=1.9) -> list:
    nr, nc = len(row_labels), len(col_labels)
    highlight = highlight or {}
    overlays = overlays or {}
    gl = 0.55 if row_groups else 0.0
    left = left + gl
    top = 0.55 + (0.3 if col_groups else 0.0)
    fig_w = left + nc * PANEL + 0.15
    fig_h = top + nr * PANEL + 0.15
    fig = plt.figure(figsize=(fig_w, fig_h))
    gap = 0.06
    for r in range(nr):
        for c in range(nc):
            x0 = (left + c * PANEL + gap / 2) / fig_w
            y0 = 1.0 - (top + (r + 1) * PANEL - gap / 2) / fig_h
            ax = fig.add_axes([x0, y0, (PANEL - gap) / fig_w, (PANEL - gap) / fig_h])
            ax.set_xticks([])
            ax.set_yticks([])
            img = images[r][c]
            if img is None:
                ax.set_facecolor("#e9e8e4")
                ax.add_patch(matplotlib.patches.Rectangle(
                    (0, 0), 1, 1, transform=ax.transAxes, fill=False,
                    hatch="////", edgecolor="#b5b4ae", linewidth=0))
            else:
                ax.imshow(_to_hwc(img), interpolation="nearest")
                if (r, c) in overlays:
                    overlays[(r, c)](ax)
            hc = highlight.get((r, c))
            for s in ax.spines.values():
                s.set_edgecolor(hc or "#9a9994")
                s.set_linewidth(2.6 if hc else 0.6)
            v = verdicts[r][c]
            if v is not None:
                ch, colr = VALID if v else INVALID
                ax.text(0.93, 0.93, ch, transform=ax.transAxes, ha="right", va="top",
                        fontsize=13, fontweight="bold", color="white",
                        bbox=dict(boxstyle="circle,pad=0.12", fc=colr, ec="white", lw=0.8))
        fig.text((left - 0.1) / fig_w, 1.0 - (top + (r + 0.5) * PANEL) / fig_h,
                 row_labels[r], ha="right", va="center", fontsize=6.6, linespacing=1.25)
    for c in range(nc):
        fig.text((left + (c + 0.5) * PANEL) / fig_w, 1.0 - (top - 0.12) / fig_h,
                 col_labels[c], ha="center", va="bottom", fontsize=7.2,
                 fontweight="bold", color=col_colours[c])
    if col_groups:
        for lab, a, b in col_groups:
            xa = (left + a * PANEL + 0.08) / fig_w
            xb = (left + (b + 1) * PANEL - 0.08) / fig_w
            y = 1.0 - (top - 0.42) / fig_h
            fig.add_artist(matplotlib.lines.Line2D([xa, xb], [y, y], color="#55544f", lw=0.8))
            fig.text((xa + xb) / 2, y + 0.04 / fig_h, lab, ha="center", va="bottom",
                     fontsize=7.4, color="#333")
    if row_groups:
        for lab, colr, a, b in row_groups:
            ya = 1.0 - (top + a * PANEL + 0.08) / fig_h
            yb = 1.0 - (top + (b + 1) * PANEL - 0.08) / fig_h
            x = 0.42 / fig_w
            fig.add_artist(matplotlib.lines.Line2D([x, x], [ya, yb], color=colr, lw=2.0))
            fig.text(x - 0.06 / fig_w, (ya + yb) / 2, lab, ha="right", va="center",
                     rotation=90, fontsize=7.2, fontweight="bold", color=colr,
                     linespacing=1.1)
    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext, kw in (("pdf", {"metadata": {"CreationDate": None, "ModDate": None}}),
                    ("png", {"dpi": 300, "metadata": {"Software": None}})):
        p = out_stem.with_suffix("." + ext)
        fig.savefig(p, **kw)
        paths.append(str(p))
    plt.close(fig)
    return paths
