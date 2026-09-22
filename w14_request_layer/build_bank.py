"""Build one CMAZE-2 bank, BEFORE (w9 request layer) or AFTER (the w14 repair).

Usage:  python -m w14_request_layer.build_bank <before|after> <seed_index> <n_sites> <out_dir>

Seeds are w9's: consecutive from ``1_000_000 + 100_000 * seed_index``, so a BEFORE bank
built here is the same object w9 built.  PER_SIDE and MIN_PAIRS are w9's too
(``scripts/w9_maze_bank.py``), and the walled target cycles exactly as it does there.
"""

from __future__ import annotations

import os
import pickle
import sys
import time

from w9_maze.task import build_site
from w14_request_layer.task2 import build_site_v2, site_key_orders

PER_SIDE = 3
MIN_PAIRS = 12


def main(which: str, seed_index: int, n_sites: int, out_dir: str) -> None:
    assert which in ("before", "after"), which
    seed0 = 1_000_000 + 100_000 * int(seed_index)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "bank_%s_seed%d.pkl" % (which, seed_index))
    if os.path.exists(path):
        print("exists, nothing to do:", path, flush=True)
        return
    t0 = time.time()
    sites, s, tried = [], seed0, 0
    while len(sites) < n_sites:
        wt = len(sites) % (2 * PER_SIDE)
        if which == "before":
            st = build_site(s, per_side=PER_SIDE, min_pairs=MIN_PAIRS, wall_target=wt)
        else:
            st = build_site_v2(s, per_side=PER_SIDE, min_pairs=MIN_PAIRS,
                               wall_target=wt)
        s += 1
        tried += 1
        if st is not None:
            sites.append(st)
        if tried % 50 == 0:
            print("  %s seed_index=%d  %d/%d sites, %d seeds tried, %.0fs"
                  % (which, seed_index, len(sites), n_sites, tried, time.time() - t0),
                  flush=True)
    for st in sites:
        st.key_order = site_key_orders(st)      # BEFORE banks get theirs computed here
    with open(path, "wb") as fh:
        pickle.dump({"sites": sites, "which": which, "seed_index": int(seed_index),
                     "seed0": seed0, "n_tried": tried, "per_side": PER_SIDE,
                     "min_pairs": MIN_PAIRS}, fh)
    print("wrote %s: %d sites, %d seeds tried (acceptance %.4f), %.0fs"
          % (path, len(sites), tried, len(sites) / tried, time.time() - t0), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
