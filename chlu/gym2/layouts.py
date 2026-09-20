"""Procedural obstacle layouts for GYM-2, built so that ROUTE CHOICE CAN DEPEND ON V.

The wave-7 question is "compose the speed limit with the obstacle to find a different
route". A speed limit is an upper bound, so raising V never makes a route infeasible --
the composition can therefore only be a TIME trade-off. This module builds exactly that
trade-off and then leaves it to be measured rather than assumed:

    * a SHORT route: a one-cell-wide zigzag tunnel bored through a solid block. Because
      the passage is one cell wide and corner-cutting is a collision, a fast agent cannot
      stay inside it -- traversal costs about one step per cell whatever V is.
    * a LONG route: an open detour over the top of the block, where an agent may cruise at
      its full speed limit.

Short-and-crawlable versus long-and-cruisable is the only shape of composition a speed
limit can have, and it is the one thing 1-D could not express. Whether the crossover
actually lands inside the range of V used is NOT assumed here -- `exp_w51_gym2_structure`
sweeps it exhaustively.

Every layout also carries the two cell sets that make "which route did this path take?" an
exact question: block `block_top` and the tunnel is the only way across; block
`block_tunnel` and the detour is. Route attribution is therefore two extra BFS runs, not a
heuristic on the trajectory.
"""

from __future__ import annotations

import random
from typing import NamedTuple

import numpy as np

from chlu.gym2.grid2d import G, Episode, register_layout, get_layout, move_ok_table

# --- generator ranges (pre-registered shape; see LOG.md) ---------------------

CX_CHOICES = (6, 7, 8, 9)              # left edge of the solid block
TW_CHOICES = (5, 6, 7)                 # block width in columns
HB_CHOICES = (13, 15, 17)              # block height: rows [0, hb) are solid
ZIG_CHOICES = (3, 4, 5)                # vertical jump of each serpentine leg
N_LEG_CHOICES = (1, 2, 3)              # number of legs: more legs = slower tunnel
N_DECOR = (0, 1, 2, 3)                 # extra random rectangles in the open regions

#: Tunnel-length buckets -- the "obstacle" half of the compositional axis.
T_BUCKETS = ((0, 11), (12, 15), (16, 99))   # terciles of the generator's own distribution

V_CHOICES = (1, 2, 3, 4)

#: THE compositional split. Every speed limit and every tunnel bucket appears in
#: training; these four PAIRINGS never do. A held-out episode therefore asks the agent to
#: decide between crawling and detouring for a (speed limit, tunnel length) combination it
#: has never been supervised on -- which is precisely "compose the speed limit with the
#: obstacle to find a different route".
HELD_VT = ((1, 2), (2, 0), (3, 1), (4, 2))


def t_bucket(tunnel_len: int) -> int:
    for i, (lo, hi) in enumerate(T_BUCKETS):
        if lo <= tunnel_len <= hi:
            return i
    return len(T_BUCKETS) - 1


def is_held_vt(V: int, tunnel_len: int) -> bool:
    return (V, t_bucket(tunnel_len)) in set(HELD_VT)


# ---------------------------------------------------------------------------

def _carve_tunnel(cx: int, tw: int, hb: int, ty: int, legs):
    """Cells of a one-cell-wide zigzag passage through the block.

    `legs` is a list of (column_offset, dy). At each listed column the passage runs
    vertically by `dy` before continuing right. Returns the carved cells and the exit row,
    or None if the passage would leave the block.
    """
    cells = []
    cur = ty
    leg_at = dict(legs)
    for i in range(tw):
        c = cx + i
        cells.append((c, cur))
        d = leg_at.get(i)
        if d:
            nxt = cur + d
            if not (1 <= nxt <= hb - 2):
                return None, None
            for yy in range(min(cur, nxt), max(cur, nxt) + 1):
                cells.append((c, yy))
            cur = nxt
    return sorted(set(cells)), cur


def _rect(blocked, x0, y0, w, h):
    blocked[max(0, x0):min(G, x0 + w), max(0, y0):min(G, y0 + h)] = True


def _connected_free(blocked) -> int:
    """Number of free cells in the largest 4-connected free component."""
    free = ~blocked
    seen = np.zeros_like(free)
    idx = np.argwhere(free)
    if idx.size == 0:
        return 0
    best = 0
    for sx, sy in idx:
        if seen[sx, sy]:
            continue
        stack = [(int(sx), int(sy))]
        seen[sx, sy] = True
        cnt = 0
        while stack:
            x, y = stack.pop()
            cnt += 1
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < G and 0 <= ny < G and free[nx, ny] and not seen[nx, ny]:
                    seen[nx, ny] = True
                    stack.append((nx, ny))
        best = max(best, cnt)
    return best


class LayoutSpec(NamedTuple):
    lid: int
    cx: int
    tw: int
    hb: int
    tunnel_len: int
    tunnel_cells: tuple
    block_top: tuple
    block_tunnel: tuple


def make_layout(rng: random.Random):
    """One layout, or None if the draw was degenerate."""
    cx = CX_CHOICES[rng.randrange(len(CX_CHOICES))]
    tw = TW_CHOICES[rng.randrange(len(TW_CHOICES))]
    hb = HB_CHOICES[rng.randrange(len(HB_CHOICES))]
    if cx + tw >= G - 2:
        return None
    zig = ZIG_CHOICES[rng.randrange(len(ZIG_CHOICES))]
    ty = rng.randint(1, max(1, hb - 2 - zig))

    # Legs alternate in sign, so the passage serpentines: a fast agent cannot hold a
    # straight line inside a one-cell-wide corridor that keeps turning.
    n_legs = min(N_LEG_CHOICES[rng.randrange(len(N_LEG_CHOICES))], tw - 2)
    cols = sorted(rng.sample(range(0, tw - 1), n_legs)) if n_legs > 0 else []
    sgn = rng.choice((1, -1))
    legs = []
    for i, c in enumerate(cols):
        legs.append((c, zig * (sgn if i % 2 == 0 else -sgn)))

    tunnel, exit_y = _carve_tunnel(cx, tw, hb, ty, legs)
    if tunnel is None:
        return None

    blocked = np.zeros((G, G), dtype=bool)
    _rect(blocked, cx, 0, tw, hb)
    for c, y in tunnel:
        blocked[c, y] = False

    # decoration: small rectangles in the open regions, never touching the block band or
    # the two tunnel mouths.
    for _ in range(N_DECOR[rng.randrange(len(N_DECOR))]):
        w = rng.randint(1, 3)
        h = rng.randint(1, 3)
        x0 = rng.randrange(0, G - w)
        y0 = rng.randrange(0, G - h)
        if cx - 2 <= x0 <= cx + tw + 1:
            continue
        blocked[x0:x0 + w, y0:y0 + h] = True

    if _connected_free(blocked) != int((~blocked).sum()):
        return None      # the decoration split the free space; drop the draw

    tunnel_len = len(tunnel)
    block_top = tuple((c, y) for c in range(cx, cx + tw) for y in range(hb, G))
    block_tunnel = tuple(tunnel)

    meta = dict(cx=cx, tw=tw, hb=hb, ty=ty, exit_y=exit_y, legs=legs,
                tunnel_len=tunnel_len, tunnel_cells=tuple(tunnel),
                block_top=block_top, block_tunnel=block_tunnel,
                t_bucket=t_bucket(tunnel_len))
    lid = register_layout(blocked, meta)
    return LayoutSpec(lid, cx, tw, hb, tunnel_len, tuple(tunnel), block_top,
                      tuple(tunnel))


def make_layouts(n: int, seed: int):
    """`n` distinct layouts, BALANCED across the tunnel-length buckets.

    Balance matters because the buckets are half of the compositional axis: if one bucket
    were rare, the held-out pairings using it would be thin and the split would be
    measuring sample size rather than composition. Distinctness is on the blocked map
    itself, exactly.
    """
    rng = random.Random(seed)
    nb = len(T_BUCKETS)
    quota = [n // nb + (1 if i < n % nb else 0) for i in range(nb)]
    got = [0] * nb
    out, seen = [], set()
    tries = 0
    while sum(got) < n and tries < n * 600:
        tries += 1
        sp = make_layout(rng)
        if sp is None:
            continue
        b = t_bucket(sp.tunnel_len)
        if got[b] >= quota[b]:
            continue
        key = get_layout(sp.lid).blocked.tobytes()
        if key in seen:
            continue
        seen.add(key)
        got[b] += 1
        out.append(sp)
    return out


def make_sealed_layout(lid: int) -> int:
    """A variant of `lid` with BOTH crossings closed, so the grid has two components.

    The refusal split needs episodes whose goal is genuinely unreachable while a
    fault-free behaviour still exists. Ordinary GYM-2 layouts cannot supply them -- the
    generator rejects any draw whose free space is not fully connected -- so the refusal
    set is built on sealed variants instead. A sealed map is an ordinary input to the gym
    (the obstacle field is an observation, not a constant), and it is the one place where
    reading the obstacle is the only way to know that the goal must be refused.
    """
    base = get_layout(lid)
    blocked = base.blocked.copy()
    for c, y in base.meta["block_tunnel"]:
        blocked[c, y] = True
    for c, y in base.meta["block_top"]:
        blocked[c, y] = True
    meta = dict(base.meta)
    meta.update(sealed_from=lid, sealed=True, tunnel_cells=(), block_tunnel=(),
                block_top=())
    return register_layout(blocked, meta)


def layout_bytes(lid: int) -> bytes:
    return get_layout(lid).blocked.tobytes()


def render(lid: int) -> str:
    """ASCII picture, y increasing upward, for the LOG."""
    lay = get_layout(lid)
    rows = []
    for y in range(G - 1, -1, -1):
        rows.append("".join("#" if lay.blocked[x, y] else "." for x in range(G)))
    return "\n".join(rows)
