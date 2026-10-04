"""hub/placement -- THE SCORED ENTRY POINT.  One arm, one bridge seed, the seven rows.

WHY THIS MODULE EXISTS AND WHAT IT DOES NOT DO.  ``exp_gen_factor_wells`` pins
``PINS["seed"] = 0`` and its CLI has no seed switch, because the hub that wrote it scored
one seed.  w4-common section 2 requires **three** (bridge seeds 0, 1, 2 on cache
generation 2).  This module is the seed switch and nothing else: it sets the seed, makes
sure the seed's own candidate partitions exist, hands them over through the additive
``extra_partitions.npz`` hook, and then calls ``exp_gen_factor_wells.run_arm``.  ⛔ It does
not re-implement the store fit, the head, the read, the judge, the corners, the accept
rule, the deletion leg or the estimand.  Every number it produces comes out of the
inherited code path.

⭐ WHAT IS SEED-DEPENDENT, AND IT IS THE RIGHT THING.  The placement METHOD is fixed --
rank-normalised ``I(bin(channel); factor)`` on the seed's own fit pool, the shipped merge,
the shipped capacity-constrained assignment.  The label VECTOR it produces is different on
every seed, because every seed is a different encoder.  That is what a seed axis is for: it
asks whether the METHOD wins on three encoders, not whether one lucky partition does.

⛔ THE GENERATION TRAVELS WITH THE NUMBER.  ``CHLU_GEN_FEATURES_ROOT`` must point at the
generation-2 cache for a scored run, and the run refuses to start on generation 1 unless
``--allow-generation1`` is passed for a DEV run.  Generation 2 seed 0 and generation 1 seed
0 differ (``ef6359a4`` vs ``d6728f01``) even at an identical encoder hash, so w4-common
section 2's ban on mixing generations is enforced here rather than trusted.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import exp_gen_factor_wells as W
from . import exp_gen_placement as P

#: The placement families a scored run may deploy.  A name outside this set cannot be
#: handed over, so a scored run cannot quietly deploy a partition the hub never screened.
EXPORTABLE = ("mi_balanced", "mi_argmax", "norm_rank", "norm_zscore", "norm_sqrt",
              "norm_share", "indep_lam0.5", "indep_lam1", "indep_lam2")


def prepare_seed(out_dir: Path, seed: int, log=print) -> dict:
    """Build this seed's candidate partitions and hand them to the scoring path.

    The candidates are built in ``out_dir`` itself, so one directory holds the partition
    that was deployed, the relevance it came from, the fit pool it was computed on and the
    rows that were scored -- and a replication needs that directory and nothing else.
    """
    from ..core import gen_swap_probe as SP
    W.PINS["seed"] = int(seed)
    cands = P.build_candidates(out_dir, log=log)
    have = cands["labels"]
    extra = {}
    for nm in EXPORTABLE:
        if nm not in have:
            continue
        v = np.asarray(have[nm], np.int64)
        extra[nm] = v
        # the size-matched laundering floor for every exportable placement, so the arm's
        # own floor is never borrowed from another arm's histogram
        extra["rand_" + nm] = SP.size_matched_random(v, int(seed))
    p = out_dir / "extra_partitions.npz"
    np.savez_compressed(p, **extra)
    meta = {
        "seed": int(seed),
        "feature_cache": W.feature_cache_manifest(int(seed)),
        "partitions": {k: {"well_sizes": np.bincount(
            v, minlength=int(W.PINS["K"])).tolist()} for k, v in extra.items()},
        "method_is_fixed_the_vector_is_not": (
            "the placement RULE is identical on every seed; the label vector it produces "
            "differs because every seed is a different encoder"),
    }
    W._jdump(meta, out_dir / "extra_partitions.json")
    log(f"  seed {seed}: {len(extra)} partitions handed over -> {p.name}")
    return meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--condition", default="test", choices=("val", "test"))
    ap.add_argument("--max-seconds", type=float, default=900.0)
    ap.add_argument("--allow-generation1", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    W.PINS["seed"] = int(a.seed)
    W.PINS["condition"] = a.condition
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)

    man = W.feature_cache_manifest(int(a.seed))
    if man.get("generation") != 2 and not a.allow_generation1:
        raise SystemExit(
            f"REFUSING: feature cache generation is {man.get('generation')!r}, not 2. "
            "w4-common section 2 scores on generation 2 only. Set "
            "CHLU_GEN_FEATURES_ROOT, or pass --allow-generation1 for a DEV run.")
    log(f"arm={a.arm} seed={a.seed} condition={a.condition} cache={json.dumps(man)}")
    if not (out / "extra_partitions.npz").exists():
        prepare_seed(out, int(a.seed), log=log)
    res = W.run_arm(a.arm, out, max_seconds=float(a.max_seconds), log=log)
    log(f"done: {json.dumps(res, default=str)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
