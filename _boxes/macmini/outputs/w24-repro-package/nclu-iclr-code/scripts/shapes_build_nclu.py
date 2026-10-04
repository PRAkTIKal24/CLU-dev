"""Build one n-CLU on Shapes3D: wells partition, per-well principal subspaces with the retained
sufficient statistic, and the request-indexed tensor address book (closed-form head).

    python scripts/shapes_build_nclu.py --where test --seed 0 --K 32 --npool 131072

--where test|val|rung:<n> selects the fit pool (the partition file of that condition, or a ladder
rung's pool); the first --npool rows are used. Writes runs/shapes3d/stores/<where>/seed<s>/K<K>_n<npool>/
arm.pkl and info.json (four-part memory footprint).
"""

from __future__ import annotations
import argparse
import pickle
import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shapes_common import D, OUT, assert_worktree, jdump, load_pool, log


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--where", required=True, help="val | test | rung:<k>")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--npool", type=int, required=True)
    ap.add_argument(
        "--reuse-address",
        default=None,
        help="load an EXISTING fitted address book (ladder leg pkl) instead of refitting",
    )
    a = ap.parse_args(argv)
    wt = assert_worktree()
    from nclu.core.gen_read import TensorAddress
    from nclu.core.gen_store import channel_correlation
    from nclu.experiments import factor_wells as fw
    from nclu.experiments import exp_gen_read as R
    from nclu.experiments.covariance_store import CovContentStore
    from shapes_nclu_lib import atoms_for, four_part_footprint, pinned_basis, wells_for

    seed, K, n = (int(a.seed), int(a.K), int(a.npool))
    d = OUT / "stores" / a.where.replace(":", "") / ("seed%d" % seed) / ("K%d_n%d" % (K, n))
    d.mkdir(parents=True, exist_ok=True)
    if (d / "arm.pkl").exists() and (d / "info.json").exists():
        log("already built: %s" % d)
        return 0
    fw.PINS["seed"] = seed
    fw.PINS["condition"] = "val" if a.where == "val" else "test"
    grid, _ = fw.build_grid()
    from nclu.experiments.gen_deletion import assert_generation2

    log("cache %s" % assert_generation2(seed))
    pool_full, dead, pfile = load_pool(a.where, seed)
    pool = pool_full[:n]
    log(
        "where=%s seed=%d K=%d npool=%d (of %d) from %s"
        % (a.where, seed, K, n, pool_full.size, pfile)
    )
    t = time.time()
    F = fw.features_mmap(seed)
    Xc = np.asarray(fw.gather(F, pool[: min(n, int(fw.PINS["n_corr"]))]), np.float32)
    corr = channel_correlation(Xc)
    del Xc
    X = np.asarray(fw.gather(F, pool), np.float32)
    wc = wells_for(corr, K, dead)
    sizes = [int(c.size) for c in wc]
    A = atoms_for(sizes)
    st = CovContentStore(X, wc, A, D)
    del X
    basis = pinned_basis(st)
    rw = [round(int(b.shape[1]) / s, 4) for b, s in zip(basis, sizes)]
    t_build = time.time() - t
    log(
        "store built %.1fs  A=%d wells %d..%d RANK/WIDTH %.4f..%.4f"
        % (t_build, A, min(sizes), max(sizes), min(rw), max(rw))
    )
    t = time.time()
    if a.reuse_address:
        with open(a.reuse_address, "rb") as fh:
            addr = pickle.load(fh)
        log("address book REUSED from %s (not refitted)" % a.reuse_address)
    else:
        addr = None
    if addr is None:
        Y, T = R._pool_span_targets(st, pool, basis, log=lambda *_: None)
        ntr = int(0.9 * Y.shape[0])
        addr = TensorAddress(
            [int(c) for c in grid.cardinalities],
            rank=int(R.READ_PINS["rank"]),
            ridge=float(R.READ_PINS["ridge"]),
            seed=int(R.READ_PINS["address_seed"]),
        )
        addr.multiplicative = True
        addr.fit(
            Y[:ntr],
            [t_[:ntr] for t_ in T],
            sweeps=int(R.READ_PINS["sweeps"]),
            log=lambda *_: None,
            Y_val=Y[ntr:],
            targets_val=[t_[ntr:] for t_ in T],
        )
    t_addr = time.time() - t
    abytes = int(addr.byte_footprint()["address_bytes"])
    led = four_part_footprint(st, abytes)
    log(
        "address book %.1fs %d B; four-part total %d B"
        % (t_addr, abytes, led["honest_total_bytes"])
    )
    with open(d / "arm.pkl", "wb") as fh:
        pickle.dump(
            {
                "store": st,
                "address": addr,
                "basis": basis,
                "well_channels": wc,
                "K": K,
                "A": A,
                "seed": seed,
                "where": a.where,
                "npool": n,
                "read": "tensor",
            },
            fh,
        )
    jdump(
        {
            "where": a.where,
            "seed": seed,
            "K": K,
            "npool": n,
            "pool_file": pfile,
            "n_fit_rows": int(pool.size),
            "A_requested": A,
            "well_sizes": sizes,
            "deployed_rank": [int(r) for r in st.deployed_rank],
            "rank_over_width": rw,
            "rank_over_width_max": float(max(rw)),
            "footprint_four_part": led,
            "build_seconds": round(t_build, 1),
            "address_fit_seconds": round(t_addr, 1),
            "package_file": wt,
            "address_reused_from": a.reuse_address,
        },
        d / "info.json",
    )
    log("wrote %s" % d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
