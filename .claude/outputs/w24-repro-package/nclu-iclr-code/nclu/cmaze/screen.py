"""screen."""

from __future__ import annotations
import numpy as np
from nclu.cmaze.maze_colour import FLOOR, UNTAKEN, USED, _pocket


def canon_features(inst, pos, ks, radius=2):
    """Colour-canonicalised local signature.

    Colours are recoded RELATIVE TO THE POCKET, so a fragment learnt on a red key
    transfers to a blue one.  This is the most generous reasonable featurisation for
    a lookup policy and is what gives fragment lookup its best shot.
    Codes: 0 wall, 1 plain floor, 2 tint == pocket colour, 3 tint != pocket,
           4 door == pocket colour, 5 door != pocket, 6 an untaken key.
    """
    g, tint, door = (inst.grid, inst.tint, inst.door)
    H, W = g.shape
    pk = _pocket(ks)
    key_cell = {inst.key_at[i]: i + 1 for i in range(inst.n_colours)}
    patch = []
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            r, c = (pos[0] + dr, pos[1] + dc)
            if not (0 <= r < H and 0 <= c < W) or g[r, c] != FLOOR:
                patch.append(0)
                continue
            dcol = int(door[r, c])
            if dcol:
                patch.append(1 if ks[dcol - 1] == USED else 4 if dcol == pk else 5)
                continue
            kc = key_cell.get((r, c), 0)
            if kc and ks[kc - 1] == UNTAKEN:
                patch.append(6)
                continue
            tcol = int(tint[r, c])
            if tcol:
                patch.append(2 if tcol == pk else 3)
                continue
            patch.append(1)
    gr = int(np.sign(inst.goal[0] - pos[0]))
    gc = int(np.sign(inst.goal[1] - pos[1]))
    return (tuple(patch), 1 if pk else 0, gr, gc)
