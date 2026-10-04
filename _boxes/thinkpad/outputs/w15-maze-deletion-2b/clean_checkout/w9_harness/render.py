"""Maze -> image.  The ONE input every learned arm sees, via the shared encoder.

The image is the INITIAL maze only: walls, start, goal, door colours, tint colours
and key home cells.  Nothing about the agent's trajectory or pocket is in it, which
is what keeps the encoder arm-agnostic -- it is trained by reconstruction on these
images alone, so no arm's objective can reach it (see `encoder.py`).

Channel layout for k colours (3 + 3k planes; k=4 -> 15):
    0            floor
    1            start
    2            goal
    3      .. 2+k    door of colour c
    3+k    .. 2+2k   tint of colour c
    3+2k   .. 2+3k   key home of colour c
Every plane is 0/1, so reconstruction is a per-pixel Bernoulli problem and the
reconstruction loss is directly readable as bits.
"""

from __future__ import annotations

import numpy as np

from w8_envscout.maze_colour import FLOOR


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


def render_batch(insts):
    return np.stack([render(i) for i in insts])
