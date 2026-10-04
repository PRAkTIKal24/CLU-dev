"""Deletion assertion battery on covariance-store candidates (implementation 2).

python scripts/shapes_deletion_battery_candidates.py --K 5 --atoms 64 --seed 0
"""

from __future__ import annotations
from nclu.paths import run_dir
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nclu.experiments.covariance_store import (
    CovContentStore,
    WellStats,
    honest_byte_footprint,
    plan_partition,
    EIG_REL_TOL,
)

PART = str(run_dir("partitions/test/partitions.npz"))
D = 2048
BUDGET = 524288
DELETED_FACTOR_NAME = "object_hue"
DELETED_VALUE = 0.0


def log(m=""):
    print(m, flush=True)


def load_pool_features(seed: int, pool: np.ndarray) -> np.ndarray:
    from nclu.experiments import factor_wells as fw

    return np.asarray(fw.gather(fw.features_mmap(int(seed)), pool), np.float32)


def red_mask(pool: np.ndarray) -> np.ndarray:
    """Which pool rows carry red.  Read from the factor index, not from a label array."""
    from nclu.paths import data_root

    z = np.load(
        str(data_root() / "shapes3d" / "value_holdout" / "factor_index.npz"), allow_pickle=True
    )
    names = [str(x) for x in z["factor_names"]]
    cards = [int(c) for c in z["cardinalities"]]
    k = names.index(DELETED_FACTOR_NAME)
    vals = np.asarray(z[f"values__{DELETED_FACTOR_NAME}"], np.float64)
    vidx = int(np.argmin(np.abs(vals - DELETED_VALUE)))
    strides = np.ones(len(cards), np.int64)
    for i in range(len(cards) - 2, -1, -1):
        strides[i] = strides[i + 1] * cards[i + 1]
    idx = pool[:, None] // strides[None, :] % np.asarray(cards, np.int64)[None, :]
    return idx[:, k] == vidx


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=5, help="5 = the incumbent's own partition")
    ap.add_argument("--atoms", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-pool", type=int, default=0, help="0 = the whole pool")
    a = ap.parse_args(argv)
    z = dict(np.load(PART, allow_pickle=True))
    lab = np.asarray(z["factor"], np.int64)
    pool = np.asarray(z["pool"], np.int64)
    dead = np.asarray(z["dead"]).ravel()
    if a.n_pool:
        pool = pool[: a.n_pool]
    K = int(lab.max()) + 1
    if a.K != K:
        log(
            f"[!] this DEV pass uses the realised K={K} partition; --K {a.K} ignored (a re-cut partition is a separate build, not a relabelling)"
        )
    wc = [np.flatnonzero(lab == k).astype(np.int64) for k in range(K)]
    sizes = [int(c.size) for c in wc]
    dead_per = [int(dead[lab == k].sum()) for k in range(K)]
    log(f"partition (SCORED leg): K={K} well sizes {sizes} sum {sum(sizes)}")
    log(
        f"dead channels per well {dead_per} -> effective {[s - d for s, d in zip(sizes, dead_per)]}"
    )
    log(f"pool rows {pool.size}, atoms A={a.atoms}, eig rel tol {EIG_REL_TOL:g}")
    sat = [d for d in sizes if a.atoms >= d]
    if sat:
        log(
            f"[!] SATURATED wells at A={a.atoms}: {sat} -- rank == d_k there, the projection is the identity and the content carries no information"
        )
    t0 = time.time()
    X = load_pool_features(a.seed, pool)
    log(f"features {X.shape} {X.dtype} in {time.time() - t0:.1f}s")
    red = red_mask(pool)
    log(
        f"red rows in pool: {int(red.sum())} of {pool.size} ({100.0 * red.mean():.2f}%)   [{DELETED_FACTOR_NAME} = {DELETED_VALUE}]"
    )
    if red.sum() == 0:
        log("[X] no red rows in the pool -- the deletion would be a no-op; aborting")
        return 2
    t0 = time.time()
    st = CovContentStore(X, wc, a.atoms, D, allow_saturated=True)
    t_build = time.time() - t0
    led = st.byte_footprint(BUDGET)
    log(
        f"\nbuild {t_build:.1f}s   honest footprint {led['deployed_bytes']:,} B = {led['multiple_of_budget']:.3f}x budget (atoms {led['atom_bytes']:,} + statistic {led['statistic_bytes']:,})"
    )
    log(f"deployed rank per well {st.deployed_rank}")
    t0 = time.time()
    rec = st.delete_rows(X[red])
    t_down = time.time() - t0
    log(
        f"\nDOWNDATE {t_down:.3f}s  {rec['n_rows_removed']:,} rows  refit={rec['refit']}  passes over retained data={rec['passes_over_retained_data']}"
    )
    log(f"deployed rank per well after downdate {st.deployed_rank}")
    t0 = time.time()
    ref = CovContentStore(X[~red], wc, a.atoms, D, allow_saturated=True)
    t_refit = time.time() - t0
    log(
        f"REFIT-WITHOUT {t_refit:.1f}s ({int((~red).sum()):,} rows)  rank per well {ref.deployed_rank}"
    )
    log("\n== downdate vs refit-without, per well ==")
    log("  well   d_k  eff  rank_dd  rank_refit   max|dev|      subspace_dist")
    devs, ok = ([], True)
    for k in range(K):
        Ad, Ar = (st.modes[k], ref.modes[k])
        md = float(np.max(np.abs(Ad - Ar)))
        rd, rr = (st.deployed_rank[k], ref.deployed_rank[k])
        r = min(rd, rr)
        if r > 0:
            Ud = np.linalg.qr(Ad[1 : 1 + r].T)[0]
            Ur = np.linalg.qr(Ar[1 : 1 + r].T)[0]
            sd = float(np.linalg.norm(Ud @ Ud.T - Ur @ Ur.T, 2))
        else:
            sd = 0.0
        devs.append(md)
        flag = "" if md < 0.0001 and sd < 0.0001 else "   <-- MISMATCH"
        if flag:
            ok = False
        log(
            f"  {k:4d} {sizes[k]:5d} {sizes[k] - dead_per[k]:4d} {rd:8d} {rr:11d}   {md:.3e}   {sd:.3e}{flag}"
        )
    log(
        f"\n  max abs dev over wells = {max(devs):.3e}   ranks agree = {st.deployed_rank == ref.deployed_rank}"
    )
    log("\n== NEGATIVE CONTROLS (a check that cannot fail is not a check) ==")
    st2 = CovContentStore(X, wc, a.atoms, D, allow_saturated=True)
    st2.delete_rows(X[red])
    st2.stats[0].S[0, 0] += 0.001 * abs(float(st2.stats[0].S[0, 0])) + 0.001
    st2._refresh()
    nd_weak = float(np.max(np.abs(st2.modes[0] - ref.modes[0])))
    fired_weak = nd_weak >= 0.0001
    log(
        f"  (a) one-entry nudge, S[0,0] += 1e-3 rel -> max|dev| {nd_weak:.3e}  {('FIRED' if fired_weak else 'did not fire -- control too weak to bind')}"
    )
    rng = np.random.default_rng(20260920)
    notred = np.flatnonzero(~red)
    wrong = rng.choice(notred, size=int(red.sum()), replace=False)
    st3 = CovContentStore(X, wc, a.atoms, D, allow_saturated=True)
    st3.delete_rows(X[wrong])
    nd_wrong = max((float(np.max(np.abs(st3.modes[k] - ref.modes[k]))) for k in range(K)))
    fired_wrong = nd_wrong >= 0.0001
    log(
        f"  (b) downdated by {len(wrong):,} WRONG (non-red) rows -> max|dev| {nd_wrong:.3e}  {('FIRED (good)' if fired_wrong else 'DID NOT FIRE -- the check is blind')}"
    )
    sep = nd_wrong / max(devs) if max(devs) > 0 else float("inf")
    log(f"  separation = wrong-rows dev / true dev = {sep:.3g}x")
    fired = fired_wrong
    log("\n== cost line ==")
    log(f"  downdate      {t_down:9.3f} s   rows refitted 0        passes over data 0")
    log(
        f"  refit-without {t_refit:9.3f} s   rows refitted {int((~red).sum()):,}   passes over data 1"
    )
    if t_down > 0:
        log(f"  ratio refit/downdate = {t_refit / t_down:.1f}x")
    verdict = bool(ok and fired)
    log(
        f"\nVERDICT: downdate == refit-without to float precision: {('YES' if ok else 'NO')};  negative control fired: {('YES' if fired else 'NO')}"
    )
    out = Path(str(run_dir("shapes3d/deletion_battery_candidates")))
    out.mkdir(parents=True, exist_ok=True)
    (out / f"downdate_check_K{K}_A{a.atoms}_seed{a.seed}.json").write_text(
        json.dumps(
            {
                "NOT_A_SCORED_RUN": True,
                "leg": "test/scored partition, DEV only",
                "K": K,
                "well_sizes": sizes,
                "dead_per_well": dead_per,
                "n_atoms": a.atoms,
                "pool_rows": int(pool.size),
                "red_rows": int(red.sum()),
                "eig_rel_tol": EIG_REL_TOL,
                "footprint": led,
                "deployed_rank_downdated": st.deployed_rank,
                "deployed_rank_refit": ref.deployed_rank,
                "max_abs_dev_per_well": devs,
                "max_abs_dev": max(devs),
                "ranks_agree": st.deployed_rank == ref.deployed_rank,
                "negative_controls": {
                    "weak_one_entry_nudge": {
                        "dev": nd_weak,
                        "fired": fired_weak,
                        "note": "did not bind; kept and reported rather than deleted",
                    },
                    "binding_wrong_rows": {
                        "dev": nd_wrong,
                        "fired": fired_wrong,
                        "n_rows": int(len(wrong)),
                        "note": "downdated by a size-matched set of NON-red rows",
                    },
                    "separation_x": sep,
                },
                "negative_control_fired": fired,
                "seconds": {"build": t_build, "downdate": t_down, "refit_without": t_refit},
                "verdict_exactness_holds": verdict,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"\nwrote {out}")
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
