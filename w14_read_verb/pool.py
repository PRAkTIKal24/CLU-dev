"""The ENCODER POOL for CMAZE-2b -- mazes from a seed range no evaluation bank touches.

Identical in role and construction to ``scripts/w9_maze_pool.py``: reconstruction
material, built WITHOUT poisoning and WITHOUT walling, so the shared encoder has to
learn a generic maze-and-route code rather than a list of the evaluation bank.

The ONE difference from w9's pool is the one thing wave 14 changed about the
environment: the sites are built by ``w14_request_layer.task2.build_site_v2``, so the
pool's routes are >= 2-key routes like the evaluation bank's.  A pool of single-key
routes would train the shared encoder on a different route distribution from the one
every arm is then graded on, which would be a confound carried by the encoder.

Seed range 7_000_000+ : disjoint from w9's pool (5_000_000+) and from every CMAZE-2b
evaluation bank (1_000_000 / 1_100_000 / 1_200_000 + at most a few thousand).

    python -m w14_read_verb.pool <shard> <n_sites> <out_dir>
"""

from __future__ import annotations

import os
import pickle
import sys
import time

from w14_request_layer.task2 import build_site_v2

POOL_SEED0 = 7_000_000


def main(shard: int, n_sites: int, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "pool_shard%d.pkl" % shard)
    if os.path.exists(path):
        print("exists:", path, flush=True)
        return
    seed = POOL_SEED0 + 200_000 * int(shard)
    seed0 = seed
    sites, tried, t0 = [], 0, time.time()
    while len(sites) < n_sites:
        st = build_site_v2(seed, per_side=3, min_pairs=12, adversarial=0,
                           wall_target=-1)
        seed += 1
        tried += 1
        if st is not None:
            sites.append(st)
        if tried % 100 == 0:
            print("  shard %d: %d/%d sites, %d tried, %.0fs"
                  % (shard, len(sites), n_sites, tried, time.time() - t0), flush=True)
    with open(path, "wb") as fh:
        pickle.dump({"sites": sites, "shard": shard, "n_tried": tried,
                     "seed0": seed0, "builder": "w14_request_layer.task2.build_site_v2",
                     "adversarial": 0, "wall_target": -1}, fh)
    print("wrote %s: %d sites, %d tried (acceptance %.4f), %.0fs"
          % (path, len(sites), tried, len(sites) / tried, time.time() - t0), flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
