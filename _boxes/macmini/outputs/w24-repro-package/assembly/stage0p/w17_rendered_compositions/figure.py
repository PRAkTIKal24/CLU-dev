"""The grid: one row per request, one column per arm, same layout for every panel set.

Inside a panel there is only the rendered image, the verdict mark (top-right corner) and
nothing else.  The request label sits to the LEFT of its row and the arm name ABOVE its
column, both outside the panels, in the arm names and colours `figures/FIGURES.md` fixes
for every figure in the paper.

The verdict mark is the judge's per-request verdict: a filled green check for valid, a
filled red cross for invalid.  Shape AND colour differ, so it never relies on colour alone.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.hashsalt": "w17-rendered-compositions",
    "path.simplify": False,
})

VALID = ("✓", "#1a7f37")     # check
INVALID = ("✗", "#c62828")   # cross
PANEL = 1.05                      # inches per panel


def _to_hwc(img: np.ndarray) -> np.ndarray:
    a = np.asarray(img, np.float32)
    if a.ndim == 3 and a.shape[0] in (1, 3) and a.shape[-1] not in (1, 3):
        a = np.transpose(a, (1, 2, 0))
    if a.shape[-1] == 1:
        a = a[..., 0]
    return np.clip(a, 0.0, 1.0)


def draw_grid(out_stem: Path, col_labels, col_colours, row_labels, images, verdicts,
              title: str = "", placeholder_banner: str = "", col_groups=None) -> list:
    """images[r][c]: HWC/CHW array or None; verdicts[r][c]: True / False / None.

    `col_groups`: optional list of (label, first_col, last_col) drawn as a bracket above
    the column names (used by the deletion panel: before / after).
    Writes <out_stem>.pdf and <out_stem>.png and returns both paths.
    """
    nr, nc = len(row_labels), len(col_labels)
    left = 1.9
    top = 0.55 + (0.3 if col_groups else 0.0) + (0.35 if title else 0.0)
    fig_w = left + nc * PANEL + 0.15
    fig_h = top + nr * PANEL + (0.6 if placeholder_banner else 0.15)
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
            for s in ax.spines.values():
                s.set_edgecolor("#9a9994")
                s.set_linewidth(0.6)
            v = verdicts[r][c]
            if v is not None:
                ch, colr = VALID if v else INVALID
                ax.text(0.93, 0.93, ch, transform=ax.transAxes, ha="right", va="top",
                        fontsize=13, fontweight="bold", color="white",
                        bbox=dict(boxstyle="circle,pad=0.12", fc=colr, ec="white", lw=0.8))
        fig.text((left - 0.1) / fig_w,
                 1.0 - (top + (r + 0.5) * PANEL) / fig_h, row_labels[r],
                 ha="right", va="center", fontsize=6.6, linespacing=1.25)
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
    if title:
        fig.text(0.5, 1.0 - 0.1 / fig_h, title, ha="center", va="top", fontsize=8.5,
                 fontweight="bold")
    if placeholder_banner:
        fig.text(0.5, 0.12 / fig_h, placeholder_banner, ha="center", va="bottom",
                 fontsize=7.4, color="#c62828", fontweight="bold")
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
