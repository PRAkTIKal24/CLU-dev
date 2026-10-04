"""Train and cache the shared encoder for one seed.

The encoder depends only on (bank seed, split, pool) -- never on the store
configuration -- so it can be trained once per seed and reused by every leg and every
arm.  That is not a shortcut: the encoder is a CONSTANT of the comparison by wave-9
rule 1, and caching it is what makes a store sweep a sweep of the store.

    python scripts/w9_maze_ae_cache.py seed=1 ae_steps=12000
"""

from __future__ import annotations

import sys

from scripts.w9_maze_store_sweep import load_all


def main(seed=0, ae_steps=12000):
    load_all(int(seed), int(ae_steps))
    print("encoder ready for seed", seed)


if __name__ == "__main__":
    kw = dict(a.split("=") for a in sys.argv[1:])
    main(**{k: int(v) for k, v in kw.items()})
