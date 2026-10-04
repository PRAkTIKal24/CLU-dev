"""Build and cache one CMAZE-2 maze bank.

Usage:  python scripts/w9_maze_bank.py <seed_index> <n_sites> <out_dir>

One bank per seed index.  Generator seeds are consecutive from
``1_000_000 + 100_000 * seed_index`` so the three seeds share no maze.
"""

from __future__ import annotations

import os
import pickle
import sys
import time

from w9_maze.task import build_site

PER_SIDE = 3
MIN_PAIRS = 12


def main(seed_index: int, n_sites: int, out_dir: str) -> None:
    seed0 = 1_000_000 + 100_000 * int(seed_index)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "bank_seed%d.pkl" % seed_index)
    if os.path.exists(path):
        print("exists, nothing to do:", path)
        return
    t0 = time.time()
    sites, s, tried = [], seed0, 0
    while len(sites) < n_sites:
        #: every accepted site has one target SEALED -- the second refusal condition.
        #: The preferred word cycles, so no target word is walled everywhere and every
        #: target word stays grounded in the mazes where it is not.
        wt = len(sites) % (2 * PER_SIDE)
        st = build_site(s, per_side=PER_SIDE, min_pairs=MIN_PAIRS, wall_target=wt)
        s += 1
        tried += 1
        if st is not None:
            sites.append(st)
        if tried % 25 == 0:
            print("  seed_index=%d  %d/%d sites, %d seeds tried, %.0fs"
                  % (seed_index, len(sites), n_sites, tried, time.time() - t0),
                  flush=True)
    with open(path, "wb") as fh:
        pickle.dump({"sites": sites, "seed_index": int(seed_index), "seed0": seed0,
                     "n_tried": tried, "per_side": PER_SIDE,
                     "min_pairs": MIN_PAIRS}, fh)
    print("wrote %s: %d sites, %d seeds tried (acceptance %.3f), %.0fs"
          % (path, len(sites), tried, len(sites) / tried, time.time() - t0))


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
