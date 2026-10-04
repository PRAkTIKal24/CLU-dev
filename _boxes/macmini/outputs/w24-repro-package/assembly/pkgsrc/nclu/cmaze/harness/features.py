"""features."""

from __future__ import annotations
import numpy as np
from nclu.cmaze.maze_colour import UNTAKEN

PATCH_R = 2
BUDGET_SCALE = 64.0


def dyn_dim(k):
    return 3 * k + (k + 1) + 4 + 2 + 4 + k


def obs_dim(feat_dim, k):
    side = 2 * PATCH_R + 1
    return 2 * feat_dim * side * side + dyn_dim(k)


def global_summary(S, side=5):
    """The feature map average-pooled to side x side.  A local patch alone cannot
    tell an arm where the far half of the maze is; this is the global half of the
    observation, and it is the same encoder, so no arm gets a private view."""
    D, H, W = S.shape
    bh, bw = (H // side, W // side)
    return S[:, : bh * side, : bw * side].reshape(D, side, bh, side, bw).mean((2, 4))


def patch(S, pos, r=PATCH_R):
    """(D, 2r+1, 2r+1) slice of the feature map around pos, zero outside the grid."""
    D, H, W = S.shape
    side = 2 * r + 1
    out = np.zeros((D, side, side), dtype=np.float32)
    r0, r1 = (pos[0] - r, pos[0] + r + 1)
    c0, c1 = (pos[1] - r, pos[1] + r + 1)
    sr0, sc0 = (max(r0, 0), max(c0, 0))
    sr1, sc1 = (min(r1, H), min(c1, W))
    if sr0 < sr1 and sc0 < sc1:
        out[:, sr0 - r0 : sr1 - r0, sc0 - c0 : sc1 - c0] = S[:, sr0:sr1, sc0:sc1]
    return out


def dyn(inst, st):
    k = int(inst.n_colours)
    H, W = inst.grid.shape
    v = np.zeros(dyn_dim(k), dtype=np.float32)
    o = 0
    for c in range(k):
        v[o + 3 * c + int(st.ks[c])] = 1.0
    o += 3 * k
    v[o + st.pocket] = 1.0
    o += k + 1
    dr = inst.goal[0] - st.pos[0]
    dc = inst.goal[1] - st.pos[1]
    v[o + 0] = dr / H
    v[o + 1] = dc / W
    v[o + 2] = np.sign(dr)
    v[o + 3] = np.sign(dc)
    o += 4
    v[o + 0] = st.pos[0] / H
    v[o + 1] = st.pos[1] / W
    o += 2
    v[o + 0] = st.t / max(int(inst.budget), 1)
    v[o + 1] = (int(inst.budget) - st.t) / max(int(inst.budget), 1)
    v[o + 2] = int(inst.budget) / BUDGET_SCALE
    v[o + 3] = (int(inst.budget) - st.t) / BUDGET_SCALE
    o += 4
    for c in range(k):
        v[o + c] = 1.0 if st.ks[c] == UNTAKEN else 0.0
    return v


def obs(inst, S, st, gsum=None):
    g = global_summary(S) if gsum is None else gsum
    return np.concatenate([patch(S, st.pos).reshape(-1), g.reshape(-1), dyn(inst, st)])
