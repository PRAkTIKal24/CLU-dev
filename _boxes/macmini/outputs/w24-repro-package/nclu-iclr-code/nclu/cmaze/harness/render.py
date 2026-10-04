"""render."""

from __future__ import annotations
import numpy as np
from nclu.cmaze.maze_colour import FLOOR


def n_channels(k):
    return 3 + 3 * k


def render(inst):
    """(C, H, W) float32 image of one instance.  Deterministic, no learning."""
    k = int(inst.n_colours)
    H, W = inst.grid.shape
    img = np.zeros((n_channels(k), H, W), dtype=np.float32)
    img[0] = (inst.grid == FLOOR).astype(np.float32)
    img[1][inst.start] = 1.0
    img[2][inst.goal] = 1.0
    for c in range(1, k + 1):
        img[2 + c] = (inst.door == c).astype(np.float32)
        img[2 + k + c] = (inst.tint == c).astype(np.float32)
    for i, cell in enumerate(inst.key_at):
        img[2 + 2 * k + (i + 1)][cell] = 1.0
    return img
