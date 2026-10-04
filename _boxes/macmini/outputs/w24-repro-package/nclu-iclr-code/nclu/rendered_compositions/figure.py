"""figure."""

from __future__ import annotations
import matplotlib

matplotlib.use("Agg")
import numpy as np

matplotlib.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.hashsalt": "rendered-compositions",
        "path.simplify": False,
    }
)
VALID = ("✓", "#1a7f37")
INVALID = ("✗", "#c62828")


def _to_hwc(img: np.ndarray) -> np.ndarray:
    a = np.asarray(img, np.float32)
    if a.ndim == 3 and a.shape[0] in (1, 3) and (a.shape[-1] not in (1, 3)):
        a = np.transpose(a, (1, 2, 0))
    if a.shape[-1] == 1:
        a = a[..., 0]
    return np.clip(a, 0.0, 1.0)
