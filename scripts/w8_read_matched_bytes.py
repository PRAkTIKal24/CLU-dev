"""Fit the address book on the matched-bytes object and read corners.

⛔ TWO PINS, AND THEY ARE NOT THE SAME PIN.  The spoke's object pins the ATOM BLOCK at
524,288 B and reports the covariance beside it (honest total 24.5x).  The matched-bytes
object pins the FOUR-PART HONEST TOTAL at 524,288 B and lets the atom block fall to
57,344 B (K=40) so the covariance fits inside the budget.  Both are "byte pin primary";
they pin different things, and `DowndatableStore.byte_ledger` asserts the first, which
is why this drives `CovContentStore` rather than reusing `stage_smoke` wholesale.

The read is the programme's own, unchanged: `TensorAddress` product address at R=128
over the per-well span, `exp_gen_read.arm_features`, `assert_read_request_only`.

Basis: `pinned_span_basis`-style (the store's own eigenvalue-ordered direction rows),
NOT `gen_read.span_basis`, whose SVD of a maximally-degenerate atom block is gauge-
arbitrary.  Measured by the spoke at 2.994e-03 vs 3.725e-09.

⛔ Selection (which K) is on `val` ONLY, charter A.2.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from chlu.experiments.exp_w8_single_design import CovContentStore   # noqa: E402
from w8_matched_bytes_recut import ward_balanced_labels             # noqa: E402

D = 2048
BUDGET = 524_288
OUT = Path("C:/chlu/outputs/w8-single-design/read")


def log(m=""):
    print(m, flush=True)


def pinned_basis(store):
    """The store's own direction rows as the basis, plus the mean's perpendicular part.

    ⭐ Zero bytes, no SVD of a degenerate block.  Admission of the mean's perpendicular
    component is at 1e-4 relative -- the spoke's declared threshold, adopted verbatim
    after it measured that 1e-8 readmits pure noise in rank-deficient wells.
    """
    out = []
    for M, r in zip(store.modes, store.deployed_rank):
        V = np.asarray(M[1:1 + r], np.float64).T          # (d, r) orthonormal columns
        mu = np.asarray(M[0], np.float64)
        if r > 0:
            perp = mu - V @ (V.T @ mu)
        else:
            perp = mu
        nrm = float(np.linalg.norm(perp))
        ref = float(np.linalg.norm(mu)) or 1.0
        if nrm / ref > 1e-4:
            V = np.hstack([V, (perp / nrm)[:, None]])
        out.append(np.ascontiguousarray(V))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="val", choices=("val", "test"))
    ap.add_argument("--K", type=int, default=40)
    ap.add_argument("--A", type=int, default=7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sweeps", type=int, default=40)
    a = ap.parse_args(argv)

    from chlu.core.gen_read import TensorAddress, assert_read_request_only
    from chlu.core.gen_store import channel_correlation
    from chlu.experiments import exp_gen_factor_wells as fw
    from chlu.experiments import exp_gen_read as R

    fw.PINS["condition"] = a.condition
    part = ("C:/chlu/outputs/hub-factor-wells/run-val/partitions.npz"
            if a.condition == "val"
            else "C:/chlu/outputs/hub-factor-wells/run/partitions.npz")
    z = dict(np.load(part, allow_pickle=True))
    dead = np.asarray(z["dead"]).ravel().astype(bool)
    pool = np.asarray(z["pool"], np.int64)

    F = fw.features_mmap(a.seed)
    Xc = np.asarray(fw.gather(F, pool[: int(fw.PINS["n_corr"])]), np.float32)
    corr = channel_correlation(Xc)
    del Xc
    lab = ward_balanced_labels(corr, a.K, dead)
    wc = [np.flatnonzero(lab == k).astype(np.int64) for k in range(a.K)]
    X = np.asarray(fw.gather(F, pool), np.float32)

    t0 = time.time()
    st = CovContentStore(X, wc, a.A, D)
    led = st.byte_ledger(BUDGET)
    log(f"store K={a.K} A={a.A} seed={a.seed} cond={a.condition}: built {time.time()-t0:.1f}s")
    log(f"  atoms {led['atom_bytes']:,} + statistic {led['statistic_bytes']:,} "
        f"(+counts/coupling 18,098) -> four-part "
        f"{led['atom_bytes']+led['statistic_bytes']+18098:,} B")
    basis = pinned_basis(st)
    ranks = [b.shape[1] for b in basis]
    log(f"  span ranks {min(ranks)}..{max(ranks)} (sum {sum(ranks)})")

    grid, _ = fw.build_grid()
    Y = grid.factor_indices_of(pool)
    cards = [int(c) for c in grid.cardinalities]
    T = R._pool_span_targets(st, pool, basis)[1] if hasattr(R, "_pool_span_targets") else None
    if T is None:
        raise SystemExit("exp_gen_read._pool_span_targets missing")
    t0 = time.time()
    addr = TensorAddress(cards, rank=128, ridge=1e-2, seed=0)
    addr.multiplicative = True
    addr.fit(Y, T, sweeps=int(a.sweeps), log=log)
    log(f"  address book fitted in {time.time()-t0:.1f}s; "
        f"bytes {addr.byte_ledger()['address_bytes']:,}")

    vals0 = grid.values_of(pool[:64])
    ro = assert_read_request_only(addr, st, vals0, grid.values, basis)
    log(f"  request-only assertion: {ro.get('signature_ok', ro)}")

    d = OUT / f"{a.condition}_K{a.K}_A{a.A}_seed{a.seed}"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "arm.pkl", "wb") as fh:
        pickle.dump({"store": st, "address": addr, "basis": basis,
                     "well_channels": wc, "K": a.K, "A": a.A,
                     "condition": a.condition, "seed": a.seed,
                     "read": "tensor"}, fh)
    (d / "arm.json").write_text(json.dumps({
        "NOT_A_SCORED_RUN_BY_ITSELF": True,
        "K": a.K, "A": a.A, "seed": a.seed, "condition": a.condition,
        "ledger_four_part_bytes": led['atom_bytes'] + led['statistic_bytes'] + 18098,
        "atom_bytes": led["atom_bytes"], "statistic_bytes": led["statistic_bytes"],
        "span_ranks": ranks, "address_bytes": addr.byte_ledger()["address_bytes"],
        "request_only": {k: str(v) for k, v in (ro or {}).items()},
    }, indent=2), encoding="utf-8")
    log(f"  wrote {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
