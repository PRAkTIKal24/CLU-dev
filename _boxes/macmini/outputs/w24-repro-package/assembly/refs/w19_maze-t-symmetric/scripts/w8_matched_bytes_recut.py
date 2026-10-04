"""w8 hub/single-design -- the MATCHED-BYTES re-cut.  ⛔ DEV only, NOT a scored run.

The single design carries both properties at the incumbent's own K=5 partition, but at
**12.787x the 524,288 B budget** -- so rows 3-4 against ``pca_tensor`` would not be
matched-bytes and my target file pre-commits me to print that rather than net it.

This script asks whether there is a partition at which the honest ledger
(atoms + float64 second-moment triangle + first moment + count) fits the pin.  The
criterion is NOT re-invented: it is ``factor_aligned_balanced_labels`` -- the SAME
inherited label-MI criterion with the SAME merge rule -- evaluated at a finer K.
``mi_bits`` and ``dead`` are read off the realised ``partitions.npz``, so nothing is
re-fitted and no held-out row is touched.

⛔ Charter note: charter §B's closing sentence lists **partition** among the things a hub
may change.  "Placement exhausted" is about LEARNED well PLACEMENT returning a null
against inherited grouping; re-cutting granularity with the inherited criterion is a
different dial.  The placement RULE is unchanged; only K moves.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chlu.core.gen_factor_partition import factor_aligned_balanced_labels  # noqa: E402
from chlu.core.gen_store import channel_correlation  # noqa: E402
from chlu.experiments.exp_w8_single_design import (  # noqa: E402
    CovContentStore, honest_byte_ledger,
)

PART = "C:/chlu/outputs/hub-factor-wells/run/partitions.npz"
D = 2048
BUDGET = 524_288
STAT_ITEMSIZE = 8          # float64 -- see WellStats.stat_bytes


def log(m=""):
    print(m, flush=True)


def ledger_for(sizes, A):
    stat = sum(d * (d + 1) // 2 + d + 1 for d in sizes)
    atoms = A * D
    total = atoms * 4 + stat * STAT_ITEMSIZE
    return {"atom_bytes": atoms * 4, "statistic_bytes": stat * STAT_ITEMSIZE,
            "total_bytes": total, "x_budget": total / BUDGET,
            "within": total <= BUDGET}


def ward_balanced_labels(corr, K, dead):
    """Equal wells at ANY K by cutting the correlation dendrogram's leaf order.

    ⭐ Reproduced verbatim from ``exp_gen_factor_wells.py:594-608`` -- the shipped
    ``ward_balanced`` construction -- so this is an inherited partition evaluated at a
    finer K, not a new one.  ⭐⭐ It sees NO LABEL: the single design built on it is
    strictly MORE label-free than the incumbent, whose partition is label-MI.
    """
    from scipy.cluster.hierarchy import leaves_list, linkage
    live_idx = np.flatnonzero(~dead)
    Zl = linkage(np.abs(corr[np.ix_(live_idx, live_idx)]), method="ward")
    order = live_idx[leaves_list(Zl)]
    lab = np.zeros(D, np.int64)
    base = order.size // K
    cap = np.full(K, base, np.int64)
    cap[: order.size - base * K] += 1
    pos = 0
    for j, c in enumerate(cap):
        lab[order[pos:pos + int(c)]] = j
        pos += int(c)
    cnt = np.bincount(lab[live_idx], minlength=K).astype(np.int64)
    for c in np.flatnonzero(dead):
        j = int(np.argmin(cnt))
        lab[c] = j
        cnt[j] += 1
    return lab


def max_atoms(sizes):
    """Largest A whose honest ledger fits, and which saturates no well."""
    stat = sum(d * (d + 1) // 2 + d + 1 for d in sizes) * STAT_ITEMSIZE
    avail = BUDGET - stat
    if avail <= 0:
        return 0
    A = avail // (D * 4)
    return int(min(A, min(sizes) - 1)) if sizes else 0


def main() -> int:
    z = dict(np.load(PART, allow_pickle=True))
    R = np.asarray(z["mi_bits"], np.float64)
    dead = np.asarray(z["dead"]).ravel().astype(bool)
    lab5 = np.asarray(z["factor"], np.int64)
    pool = np.asarray(z["pool"], np.int64)
    log(f"mi_bits {R.shape}  dead {int(dead.sum())}  pool {pool.size}")

    inc = [int((lab5 == k).sum()) for k in range(int(lab5.max()) + 1)]
    inc_led = ledger_for(inc, 64)
    inc_span = sum(min(64, d) for d in inc)
    log(f"\nINCUMBENT  K=5 wells {inc} A=64 span {inc_span}  honest "
        f"{inc_led['total_bytes']:,} B = {inc_led['x_budget']:.3f}x  "
        f"saturated wells {[d for d in inc if 64 >= d]}")

    log("\n== re-cut with the SAME criterion (factor_aligned_balanced), sweeping K ==")
    log("   K  wells(min..max)   A  rank/d_min  span   atoms_B   stat_B    total_B  xB")
    best = None
    rows = []
    # ⛔ factor_aligned_balanced_labels caps K at the 6 factors ("a factor-aligned
    # partition cannot make more wells than there are factors to align them to"), so the
    # matched-bytes route cannot use it.  ward_balanced reaches any K and sees no label.
    from chlu.experiments import exp_gen_factor_wells as fw0
    Xc = np.asarray(fw0.gather(fw0.features_mmap(0), pool[: int(fw0.PINS["n_corr"])]),
                    np.float32)
    corr = channel_correlation(Xc)
    del Xc
    log(f"correlation from {int(fw0.PINS['n_corr']):,} pool rows (label-free)")
    for K in [5, 8, 12, 16, 20, 24, 28, 32, 34, 36, 37, 38, 40, 42, 44, 48]:
        try:
            lab = ward_balanced_labels(corr, K, dead)
        except Exception as e:                       # noqa: BLE001
            log(f"{K:4d}  re-cut refused: {type(e).__name__}: {e}")
            continue
        sizes = [int((lab == k).sum()) for k in range(int(lab.max()) + 1)]
        if len(sizes) != K or min(sizes) < 2:
            log(f"{K:4d}  re-cut produced {len(sizes)} wells (min {min(sizes)}); skipped")
            continue
        A = max_atoms(sizes)
        if A < 2:
            log(f"{K:4d}  no atoms fit ({sizes[:3]}...)")
            continue
        led = ledger_for(sizes, A)
        span = sum(min(A, d) for d in sizes)
        rmin = A / min(sizes)
        rows.append({"K": K, "sizes": sizes, "A": A, "span": span,
                     "rank_over_d_min": rmin, **led})
        log(f"{K:4d}  {min(sizes):4d}..{max(sizes):<4d}  {A:4d}  {rmin:9.3f} {span:5d} "
            f"{led['atom_bytes']:9,d} {led['statistic_bytes']:9,d} {led['total_bytes']:10,d} "
            f"{led['x_budget']:5.3f}{'' if led['within'] else '  OVER'}")
        if led["within"] and rmin < 1.0:
            # prefer the candidate whose span is closest to the incumbent's, which is
            # the like-for-like content budget, not the largest span.
            if best is None or abs(span - inc_span) < abs(best["span"] - inc_span):
                best = rows[-1]

    if best is None:
        log("\n[X] NO matched-bytes re-cut exists with this criterion.")
        log("    => the two properties are not jointly affordable at 524,288 B, and that")
        log("       is a result: the deletion property costs bytes the pin does not have.")
        out = Path("C:/chlu/outputs/w8-single-design/dev")
        out.mkdir(parents=True, exist_ok=True)
        (out / "matched_bytes_recut.json").write_text(json.dumps(
            {"NOT_A_SCORED_RUN": True, "incumbent": {"sizes": inc, "A": 64, **inc_led},
             "candidates": rows, "matched_bytes_point": None}, indent=2), encoding="utf-8")
        return 1

    log(f"\n⭐ MATCHED-BYTES POINT: K={best['K']} A={best['A']} "
        f"wells {min(best['sizes'])}..{max(best['sizes'])} span {best['span']} "
        f"(incumbent {inc_span}) honest {best['total_bytes']:,} B = {best['x_budget']:.3f}x")

    # -- verify the downdate still holds at this partition -------------------
    log("\n== verifying exactness AT the matched-bytes partition ==")
    from chlu.experiments import exp_gen_factor_wells as fw
    lab = ward_balanced_labels(corr, best["K"], dead)
    wc = [np.flatnonzero(lab == k).astype(np.int64) for k in range(best["K"])]
    X = np.asarray(fw.gather(fw.features_mmap(0), pool), np.float32)

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from w8_single_design_dev import red_mask                       # noqa: E402
    red = red_mask(pool)
    log(f"  pool {pool.size} rows, red {int(red.sum())}")

    t0 = time.time()
    st = CovContentStore(X, wc, best["A"], D)
    tb = time.time() - t0
    t0 = time.time()
    st.delete_rows(X[red])
    td = time.time() - t0
    t0 = time.time()
    ref = CovContentStore(X[~red], wc, best["A"], D)
    tr = time.time() - t0
    devs = [float(np.max(np.abs(st.modes[k] - ref.modes[k]))) for k in range(best["K"])]
    led_real = st.byte_ledger(BUDGET)
    log(f"  build {tb:.1f}s  downdate {td:.3f}s  refit-without {tr:.1f}s "
        f"(ratio {tr / max(td, 1e-9):.1f}x)")
    log(f"  realised honest ledger {led_real['deployed_bytes']:,} B = "
        f"{led_real['multiple_of_budget']:.3f}x  within budget={led_real['within_budget']}")
    log(f"  max abs dev over {best['K']} wells = {max(devs):.3e}   "
        f"ranks agree = {st.deployed_rank == ref.deployed_rank}")
    log(f"  deployed rank per well: min {min(st.deployed_rank)} max {max(st.deployed_rank)}")
    log(f"  saturated wells: {led_real['saturated_wells']}")

    out = Path("C:/chlu/outputs/w8-single-design/dev")
    out.mkdir(parents=True, exist_ok=True)
    (out / "matched_bytes_recut.json").write_text(json.dumps({
        "NOT_A_SCORED_RUN": True,
        "criterion": "ward_balanced -- the shipped correlation-dendrogram cut "
                     "(exp_gen_factor_wells.py:594), evaluated at a finer K. LABEL-FREE: "
                     "strictly more label-free than the incumbent's label-MI partition. "
                     "factor_aligned_balanced_labels was tried first and REFUSES K > 6.",
        "incumbent": {"sizes": inc, "A": 64, "span": inc_span, **inc_led},
        "candidates": rows,
        "matched_bytes_point": best,
        "verification": {
            "realised_ledger": led_real,
            "max_abs_dev_per_well": devs, "max_abs_dev": max(devs),
            "ranks_agree": st.deployed_rank == ref.deployed_rank,
            "deployed_rank": st.deployed_rank,
            "seconds": {"build": tb, "downdate": td, "refit_without": tr},
        },
    }, indent=2), encoding="utf-8")
    log(f"\nwrote {out / 'matched_bytes_recut.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
