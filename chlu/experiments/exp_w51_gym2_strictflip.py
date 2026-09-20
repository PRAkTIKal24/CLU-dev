"""Addendum to the structural sweep: the STRICT version of the route-flip count.

`exp_w51_gym2_structure` resolves cost ties to "tunnel", so a counted flip means the tunnel went
from *at least as good* to *strictly worse*. This runs the stricter test -- the tunnel is
strictly cheaper at some V and strictly more expensive at another -- so the headline number
cannot be an artefact of how ties are broken. Same exhaustive sweep in
(start cell x start velocity), on a subset of layouts.
"""

from __future__ import annotations

import json
import os
import pickle
import time
from multiprocessing import Pool

import numpy as np

from chlu.gym2 import grid2d as E2, layouts as LY

OUT = os.environ.get("CHLU_HUB_OUTPUTS", "C:/chlu/outputs") + "/w51-gym2"
GOALS = 40
V_LIST = (1, 2, 3, 4)


def _slice(lid, V, gx, gy, extra=()):
    d = E2._dist_field(lid, V, gx, gy, extra)
    if d is None:
        return None
    nv = 2 * V + 1
    d = d.reshape(E2.G * E2.G, nv, nv)
    return d[:, V - 1:V + 2, V - 1:V + 2].reshape(E2.G * E2.G, 9)


def work(lid):
    rng = np.random.default_rng(5000 + lid)
    lay = E2.get_layout(lid)
    meta = lay.meta
    cells = np.nonzero(~lay.blocked.reshape(-1))[0]
    goals = rng.choice(cells, size=min(GOALS, cells.size), replace=False)
    posed = strict = loose = 0
    for gf in goals:
        gx, gy = int(gf) // E2.G, int(gf) % E2.G
        tun, det, both = [], [], None
        for V in V_LIST:
            ct = _slice(lid, V, gx, gy, meta["block_top"])
            cd = _slice(lid, V, gx, gy, meta["block_tunnel"])
            if ct is None or cd is None:
                both = None
                break
            b = (ct >= 0) & (cd >= 0)
            both = b if both is None else (both & b)
            tun.append(ct < cd)          # tunnel strictly better
            det.append(cd < ct)          # detour strictly better
        if both is None or not len(tun):
            continue
        T = np.stack(tun, 0).any(0)
        D = np.stack(det, 0).any(0)
        posed += int(both.sum())
        strict += int((both & T & D).sum())
    return posed, strict, 0


def _init(entries):
    E2.reset_registry(entries)


def main():
    t0 = time.time()
    with open(f"{OUT}/splits.pkl", "rb") as f:
        d = pickle.load(f)
    entries = list(d["layouts"])
    lids = sorted(set(d["train_lids"]) | set(d["novel_lids"]))
    with Pool(2, initializer=_init, initargs=(entries,)) as p:
        res = p.map(work, lids, chunksize=4)
    posed = sum(r[0] for r in res)
    strict = sum(r[1] for r in res)
    out = {
        "n_layouts": len(lids), "goals_per_layout": GOALS,
        "configurations_with_both_routes_open_at_every_V": posed,
        "strict_route_flips": strict,
        "strict_route_flip_frac": strict / max(posed, 1),
        "definition": ("tunnel strictly cheaper at some V AND strictly more expensive at "
                       "another V; independent of how ties are broken"),
        "seconds": time.time() - t0,
    }
    with open(f"{OUT}/strictflip.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
