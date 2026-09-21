"""w8 hub/single-design -- BUILD THE MATCHED-BYTES OBJECT.  Selection on `val` ONLY.

Why this exists
---------------
The byte pin is **PRIMARY at tolerance 0**.  The object the spoke built carries both
properties but at a four-part honest ledger of **24.5466x** the pin, so ⛔ it cannot
produce a RESULT row at all -- only DISPLAY rows.  This builds the object that can.

The design, in one line: keep the winning content and the exact deletion, and pay for
the covariance by making the wells FINER, until the four-part ledger fits 524,288 B.

What is held fixed, so only one thing moves
-------------------------------------------
* content: per-well leading eigenvectors of a retained ``S_k, t_k, n_k`` -- unchanged
* deletion: rank-1 subtract, no data pass, items never kept -- unchanged
* read: ``TensorAddress`` product address -- unchanged
* the only dial is **K**, the number of wells, and ``A`` follows from the budget

⛔ THE PARTITION CRITERION CHANGES, and it must be said rather than slipped in.
``factor_aligned_balanced_labels`` REFUSES any ``k > 6`` ("a factor-aligned partition
cannot make more wells than there are factors to align them to"), so the inherited
label-MI criterion cannot reach a matched-bytes K.  The criterion used here is the
shipped **label-free** ``ward_balanced`` cut of the correlation dendrogram's leaf order
(``exp_gen_factor_wells.py:594``), which the target file sanctions explicitly.
⭐ It sees NO label, so this object is *strictly more* label-free than the incumbent.
⚠ It therefore differs from the incumbent in TWO ways at once -- criterion AND
granularity -- and the control that separates them (``ward_balanced`` at K=5, A=64) is
pre-registered in Amendment 1 §A1.4 and built here as ``--control-k5``.

The ledger is FOUR-PART and summed, never netted
------------------------------------------------
store (atoms) + covariance (PACKED triangle, float64) + counts + coupling, the same
four parts the spoke reports, so the two objects' ledgers are comparable line by line.
⭐ The covariance is counted as the packed triangle because it is now actually STORED
packed (``WellStats._Sp``); an earlier version counted a packing it did not implement.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from chlu.experiments.exp_w8_single_design import CovContentStore   # noqa: E402
from w8_matched_bytes_recut import ward_balanced_labels             # noqa: E402
from w8_single_design_dev import red_mask                           # noqa: E402

D = 2048
BUDGET = 524_288
OUT = Path("C:/chlu/outputs/w8-single-design/build")

#: The two fixed ledger parts, taken from the spoke's realised object so the two are
#: comparable line by line.  Both depend on the factor cardinalities, NOT on K.
COUNTS_BYTES = 11_488
COUPLING_BYTES = 5_288 + 1_322


def log(m=""):
    print(m, flush=True)


def plan(K: int) -> dict:
    d0, rem = divmod(D, K)
    ds = [d0 + 1] * rem + [d0] * (K - rem)
    tri = sum(d * (d + 1) // 2 for d in ds)
    cov = (tri + D + K) * 8                       # packed triangle + t_k + n_k, float64
    avail = BUDGET - cov - COUNTS_BYTES - COUPLING_BYTES
    A = avail // (D * 4) if avail > 0 else 0
    A = int(min(A, min(ds) - 1)) if ds else 0     # ⛔ never saturate a well
    return {"K": K, "well_sizes": ds, "A": A, "covariance_bytes": cov,
            "atom_bytes": A * D * 4,
            "total": A * D * 4 + cov + COUNTS_BYTES + COUPLING_BYTES,
            "rank_over_d": A / min(ds) if ds else 0.0,
            "span": sum(min(A, d) for d in ds)}


def four_part(st: CovContentStore) -> dict:
    atom = int(sum(M.size for M in st.modes)) * 4
    stat = int(sum(w.stat_bytes() for w in st.stats))
    total = atom + stat + COUNTS_BYTES + COUPLING_BYTES
    return {"store_atoms_bytes": atom, "covariance_bytes": stat,
            "counts_bytes": COUNTS_BYTES, "coupling_bytes": COUPLING_BYTES,
            "honest_total_bytes": total, "budget_bytes": BUDGET,
            "times_over_budget": round(total / BUDGET, 4),
            "within_budget": bool(total <= BUDGET),
            "note": "four parts, summed and never netted -- the same parts the spoke "
                    "reports, so the two ledgers compare line by line"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="val", choices=("val", "test"))
    ap.add_argument("--K", type=int, action="append")
    ap.add_argument("--control-k5", action="store_true",
                    help="also build ward_balanced at K=5, A=64 -- the control that "
                         "separates criterion from granularity")
    a = ap.parse_args(argv)
    Ks = a.K or [40, 44, 48]

    part = ("C:/chlu/outputs/hub-factor-wells/run-val/partitions.npz"
            if a.condition == "val"
            else "C:/chlu/outputs/hub-factor-wells/run/partitions.npz")
    z = dict(np.load(part, allow_pickle=True))
    dead = np.asarray(z["dead"]).ravel().astype(bool)
    pool = np.asarray(z["pool"], np.int64)
    lab5 = np.asarray(z["factor"], np.int64)
    inc = [int((lab5 == k).sum()) for k in range(int(lab5.max()) + 1)]
    log(f"condition={a.condition}  incumbent K=5 wells {inc}  pool {pool.size:,}")

    log("\n== the matched-bytes plan (four-part ledger, packed float64 covariance) ==")
    log("   K    d   A  rank/d  span   atoms_B     cov_B  counts+coup   TOTAL   xB")
    for K in Ks:
        p = plan(K)
        log(f" {K:3d} {p['well_sizes'][0]:4d} {p['A']:3d}  {p['rank_over_d']:5.3f} "
            f"{p['span']:5d} {p['atom_bytes']:9,d} {p['covariance_bytes']:9,d} "
            f"{COUNTS_BYTES + COUPLING_BYTES:11,d} {p['total']:9,d} "
            f"{p['total'] / BUDGET:5.3f}{'' if p['total'] <= BUDGET else '  OVER'}")

    from chlu.core.gen_store import channel_correlation
    from chlu.experiments import exp_gen_factor_wells as fw
    F = fw.features_mmap(0)
    Xc = np.asarray(fw.gather(F, pool[: int(fw.PINS["n_corr"])]), np.float32)
    corr = channel_correlation(Xc)
    del Xc
    X = np.asarray(fw.gather(F, pool), np.float32)
    red = red_mask(pool)
    log(f"\ncorrelation from {int(fw.PINS['n_corr']):,} pool rows (LABEL-FREE); "
        f"red rows {int(red.sum()):,}")

    builds = []
    jobs = [(K, plan(K)["A"]) for K in Ks]
    if a.control_k5:
        jobs.append((5, 64))
    for K, A in jobs:
        is_ctl = (K == 5)
        tag = "CONTROL ward_balanced K=5 A=64" if is_ctl else f"K={K} A={A}"
        log(f"\n===== {tag} =====")
        lab = ward_balanced_labels(corr, K, dead)
        wc = [np.flatnonzero(lab == k).astype(np.int64) for k in range(K)]
        sizes = [int(c.size) for c in wc]
        t0 = time.time()
        st = CovContentStore(X, wc, A, D, allow_saturated=is_ctl)
        tb = time.time() - t0
        led = four_part(st)
        log(f"  wells {min(sizes)}..{max(sizes)}  ranks {min(st.deployed_rank)}.."
            f"{max(st.deployed_rank)}  built {tb:.1f}s")
        log(f"  LEDGER  atoms {led['store_atoms_bytes']:,} + covariance "
            f"{led['covariance_bytes']:,} + counts {COUNTS_BYTES:,} + coupling "
            f"{COUPLING_BYTES:,}")
        log(f"          = {led['honest_total_bytes']:,} B = "
            f"{led['times_over_budget']}x the pin   WITHIN BUDGET = "
            f"{led['within_budget']}")
        sat = [d for d in sizes if A >= d]
        if sat:
            log(f"  ⛔ SATURATED wells {sorted(set(sat))} -- decorative content there")

        # exactness at THIS partition, with the size-matched control
        t0 = time.time()
        st.delete_rows(X[red])
        td = time.time() - t0
        ref = CovContentStore(X[~red], wc, A, D, allow_saturated=is_ctl)
        devs, sub = [], []
        for k in range(K):
            r = min(st.deployed_rank[k], ref.deployed_rank[k])
            devs.append(float(np.max(np.abs(st.modes[k] - ref.modes[k]))))
            if r > 0:
                Ud = np.linalg.qr(np.asarray(st.modes[k][1:1 + r], np.float64).T)[0]
                Ur = np.linalg.qr(np.asarray(ref.modes[k][1:1 + r], np.float64).T)[0]
                sub.append(float(np.linalg.norm(Ud @ Ud.T - Ur @ Ur.T, 2)))
        aud = st.basis_change_audit()
        log(f"  EXACTNESS  downdate {td:.3f}s, 0 rows refitted, 0 data passes")
        log(f"             max abs dev {max(devs):.3e}   max subspace dist "
            f"{max(sub):.3e}   ranks agree {st.deployed_rank == ref.deployed_rank}")
        log(f"  BASIS CHANGE  cos min {aud['principal_angle_cos_min']:.4f}  median "
            f"{aud['principal_angle_cos_median']:.4f}  frac>0.99 "
            f"{aud['frac_directions_cos_gt_0p99']:.3f}")

        d = OUT / f"{a.condition}_wardbal_K{K}_A{A}"
        d.mkdir(parents=True, exist_ok=True)
        rec = {"NOT_A_SCORED_RUN": True, "condition": a.condition,
               "criterion": "ward_balanced (LABEL-FREE correlation-dendrogram cut)",
               "is_control_k5": is_ctl, "K": K, "A": A, "well_sizes": sizes,
               "deployed_rank": st.deployed_rank,
               "rank_over_d": [round(r / s, 4)
                               for r, s in zip(st.deployed_rank, sizes)],
               "saturated_wells": sorted(set(sat)),
               "ledger_four_part": led,
               "exactness": {"max_abs_dev": max(devs), "per_well": devs,
                             "max_subspace_dist": max(sub),
                             "ranks_agree": st.deployed_rank == ref.deployed_rank,
                             "downdate_seconds": td, "rows_refitted": 0,
                             "data_passes": 0},
               "basis_change": aud}
        (d / "build.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
        builds.append(rec)
        log(f"  wrote {d / 'build.json'}")

    ok = [b for b in builds if b["ledger_four_part"]["within_budget"]
          and not b["saturated_wells"]]
    log(f"\n{len(ok)} of {len(builds)} builds are within budget and unsaturated")
    if ok:
        log("⭐ matched-bytes candidates: " + ", ".join(
            f"K={b['K']}/A={b['A']} at {b['ledger_four_part']['times_over_budget']}x"
            for b in ok))
    log("⛔ K is NOT selected here -- selection is on val corners, charter A.2.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
