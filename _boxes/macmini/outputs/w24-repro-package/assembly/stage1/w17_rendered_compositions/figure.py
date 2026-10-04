"""The grid: one row per request, one column per arm, same layout for every panel set.

Inside a panel there is only the rendered image, the verdict mark (top-right corner) and
nothing else.  The request label sits to the LEFT of its row and the arm name ABOVE its
column, both outside the panels, in the arm names and colours `figures/FIGURES.md` fixes
for every figure in the paper.

The verdict mark is the judge's per-request verdict: a filled green check for valid, a
filled red cross for invalid.  Shape AND colour differ, so it never relies on colour alone.
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
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
def _to_hwc(img: np.ndarray) -> np.ndarray:
    a = np.asarray(img, np.float32)
    if a.ndim == 3 and a.shape[0] in (1, 3) and a.shape[-1] not in (1, 3):
        a = np.transpose(a, (1, 2, 0))
    if a.shape[-1] == 1:
        a = a[..., 0]
    return np.clip(a, 0.0, 1.0)


