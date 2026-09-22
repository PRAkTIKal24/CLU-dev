"""The door graph: the structure the w14 fix selects on.

Nodes are the connected components of the floor with every DOOR CELL removed
(``w9_maze.task.door_free_components``).  An edge joins two components if one door
cell is orthogonally adjacent to both -- crossing that cell is the only way to move
between them.

The one fact the fix rests on, and it is a proof, not a measurement:

    a walk from component X to component Y crosses a set of door cells whose induced
    subgraph connects X to Y, so it crosses at least ``d(X, Y)`` DISTINCT door cells;
    every door cell carries a distinct colour (``maze_colour.generate`` assigns colour
    i+1 to the i-th of ``n_colours`` door cells) and a key is CONSUMED by the door it
    opens, so at least ``d(X, Y)`` DISTINCT KEYS must be picked up.

Hence ``d(X, Y) >= 2`` forces every route between them to pick up two or more keys,
and with two keys there is a key ORDER for RULE B to trap.  Nothing in the engine, the
solver or the judge is involved in establishing this.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from w8_envscout.maze_colour import FLOOR, MOVES
from w9_maze.task import door_free_components


def door_adjacency(grid, door, comps):
    """``(edges, comp_of)``: edges[(x, y)] = sorted door colours joining components x, y."""
    comp_of = {}
    for n, comp in enumerate(comps):
        for cell in comp:
            comp_of[cell] = n
    H, W = grid.shape
    edges = {}
    for r, c in map(tuple, np.argwhere(door > 0)):
        r, c = int(r), int(c)
        touch = set()
        for dr, dc in MOVES.values():
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and grid[nr, nc] == FLOOR:
                if (nr, nc) in comp_of:
                    touch.add(comp_of[(nr, nc)])
        touch = sorted(touch)
        for a in range(len(touch)):
            for b in range(a + 1, len(touch)):
                edges.setdefault((touch[a], touch[b]), []).append(int(door[r, c]))
    return {k: sorted(set(v)) for k, v in edges.items()}, comp_of


def door_distances(n_comps, edges):
    """All-pairs shortest path in DOOR CROSSINGS.  ``inf`` where unreachable."""
    adj = {i: set() for i in range(n_comps)}
    for (a, b) in edges:
        adj[a].add(b)
        adj[b].add(a)
    dist = [[float("inf")] * n_comps for _ in range(n_comps)]
    for s in range(n_comps):
        dist[s][s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[s][v] == float("inf"):
                    dist[s][v] = dist[s][u] + 1
                    q.append(v)
    return dist


def analyse(grid, door):
    """``(comps, edges, dist)`` for one maze."""
    comps = door_free_components(grid, door)
    edges, _ = door_adjacency(grid, door, comps)
    return comps, edges, door_distances(len(comps), edges)
