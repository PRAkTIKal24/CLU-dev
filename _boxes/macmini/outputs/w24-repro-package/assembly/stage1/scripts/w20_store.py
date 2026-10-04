"""w20 hub/shapes-data-matched -- BUILD one store (no scoring here).

The crossover hub's object, UNCHANGED: `CovContentStore` on the label-free
`ward_balanced` correlation cut, A = min(64, min_k d_k - 1), read by the product
`TensorAddress` at `exp_gen_read.READ_PINS` (rank 128, ridge 1e-2, 40 sweeps, 90/10
split of the fit pool) -- verbatim the code path of `w12b_scored_leg.py` (rows 3-4)
and `w12b_ladder_kstar.build_cov_store / address_book` (the rungs).

The ONLY thing this script varies is the fit pool: the store's pool truncated to its
first `--npool` rows (16,384 = the w12 comparators' rows, which ARE this pool's first
16,384 -- measured, `pool == fw.fit_pool(...)`; 32,256 = the size of spoke A's fit pool,
taken as the NESTED prefix of the same population, see PREREG §2.3).
The channel correlation that cuts the wells reads `pool[:min(npool, n_corr=16,384)]`
and the dead-channel mask is the partition file's (computed on `pool[:16,384]`), so a
16,384-row store sees no row outside its own 16,384.
"""
from __future__ import annotations
import argparse
import pickle
import sys
import time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from w20_lib import D, OUT, assert_worktree, jdump, load_pool, log  # noqa: E402
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--where", required=True, help="val | test | rung:<k>")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--npool", type=int, required=True)
    ap.add_argument("--reuse-address", default=None,
                    help="load an EXISTING fitted address book (w12b ladder leg pkl) instead of refitting")
    a = ap.parse_args(argv)
    wt = assert_worktree()

    from chlu.core.gen_read import TensorAddress
    from chlu.core.gen_store import channel_correlation
    from chlu.experiments import exp_gen_factor_wells as fw
    from chlu.experiments import exp_gen_read as R
    from chlu.experiments.exp_w8_single_design import CovContentStore
    from w12_crossover_lib import atoms_for, four_part_ledger, pinned_basis, wells_for

    seed, K, n = int(a.seed), int(a.K), int(a.npool)
    d = OUT / "stores" / a.where.replace(":", "") / ("seed%d" % seed) / ("K%d_n%d" % (K, n))
    d.mkdir(parents=True, exist_ok=True)
    if (d / "arm.pkl").exists() and (d / "info.json").exists():
        log("already built: %s" % d)
        return 0
    fw.PINS["seed"] = seed
    fw.PINS["condition"] = "val" if a.where == "val" else "test"
    grid, _ = fw.build_grid()
    from chlu.experiments.exp_gen_delete_row7 import assert_generation2
    log("cache %s" % assert_generation2(seed))
    pool_full, dead, pfile = load_pool(a.where, seed)
    pool = pool_full[:n]
    log("where=%s seed=%d K=%d npool=%d (of %d) from %s" % (a.where, seed, K, n,
                                                           pool_full.size, pfile))
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
    log("store built %.1fs  A=%d wells %d..%d RANK/WIDTH %.4f..%.4f"
        % (t_build, A, min(sizes), max(sizes), min(rw), max(rw)))

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
      addr = TensorAddress([int(c) for c in grid.cardinalities],
                           rank=int(R.READ_PINS["rank"]), ridge=float(R.READ_PINS["ridge"]),
                           seed=int(R.READ_PINS["address_seed"]))
      addr.multiplicative = True
      addr.fit(Y[:ntr], [t_[:ntr] for t_ in T], sweeps=int(R.READ_PINS["sweeps"]),
               log=lambda *_: None, Y_val=Y[ntr:], targets_val=[t_[ntr:] for t_ in T])
    t_addr = time.time() - t
    abytes = int(addr.byte_ledger()["address_bytes"])
    led = four_part_ledger(st, abytes)
    log("address book %.1fs %d B; four-part total %d B" % (t_addr, abytes,
                                                          led["honest_total_bytes"]))
    with open(d / "arm.pkl", "wb") as fh:
        pickle.dump({"store": st, "address": addr, "basis": basis, "well_channels": wc,
                     "K": K, "A": A, "seed": seed, "where": a.where, "npool": n,
                     "read": "tensor"}, fh)
    jdump({"where": a.where, "seed": seed, "K": K, "npool": n, "pool_file": pfile,
           "n_fit_rows": int(pool.size), "A_requested": A, "well_sizes": sizes,
           "deployed_rank": [int(r) for r in st.deployed_rank], "rank_over_width": rw,
           "rank_over_width_max": float(max(rw)), "ledger_four_part": led,
           "build_seconds": round(t_build, 1), "address_fit_seconds": round(t_addr, 1),
           "chlu_file": wt, "address_reused_from": a.reuse_address}, d / "info.json")
    log("wrote %s" % d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
