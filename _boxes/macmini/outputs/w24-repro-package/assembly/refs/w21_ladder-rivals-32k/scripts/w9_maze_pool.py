"""Build the ENCODER POOL: mazes disjoint from every evaluation bank.

The shared encoder is trained by reconstruction.  Trained on the 864 items of one
evaluation bank it simply learns them, and the measured consequence is a pipeline
ceiling of 1.0000 on training items and 0.41 on the unseen-pair row -- a cap no arm
can pass, which would make every row below it meaningless.

So the encoder is given a POOL: mazes from a seed range no evaluation bank touches,
with all their pairs.  It still sees no held-out item, and it now has to learn a
generic maze-and-route code rather than a list.  The pool is built without poisoning
(it is reconstruction material, not evaluation material) and without walling; that
difference is declared and is the same for every arm, since there is only one encoder.

    python scripts/w9_maze_pool.py <shard> <n_sites> <out_dir>
"""

from __future__ import annotations

import os
import pickle
import sys
import time

from w9_maze.task import build_site

POOL_SEED0 = 5_000_000


def main(shard: int, n_sites: int, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "pool_shard%d.pkl" % shard)
    if os.path.exists(path):
        print("exists:", path)
        return
    seed = POOL_SEED0 + 200_000 * int(shard)
    sites, tried, t0 = [], 0, time.time()
    while len(sites) < n_sites:
        st = build_site(seed, per_side=3, min_pairs=12, adversarial=0, wall_target=-1)
        seed += 1
        tried += 1
        if st is not None:
            sites.append(st)
        if tried % 100 == 0:
            print("  shard %d: %d/%d sites, %d tried, %.0fs"
                  % (shard, len(sites), n_sites, tried, time.time() - t0), flush=True)
    with open(path, "wb") as fh:
        pickle.dump({"sites": sites, "shard": shard, "n_tried": tried}, fh)
    print("wrote %s: %d sites, %d tried, %.0fs"
          % (path, len(sites), tried, time.time() - t0))


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
