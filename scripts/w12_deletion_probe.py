"""w12 -- WHAT DID THE DELETION ACTUALLY MOVE AT THE READ?

THE ROW THIS EXISTS FOR.  On TEST at K* = 32, deleting every one of the 11,044 rows
carrying red changed the composition of the 20 red-requesting corners by EXACTLY 0.0000
(32.25 -> 32.25 novel tuples per corner, judge accuracy 0.8958 -> 0.8958), while the
store-scoped exactness check says the downdated store equals a store refit WITHOUT those
rows to 5.960e-08.  Both cannot be ignored, so this measures the thing in between: how far
the RENDERED ANSWER moved, in the store's own units, on exactly those requests.

It needs no judge and no decoder, so it is cheap and it isolates the read.
"""
from __future__ import annotations

import argparse
import copy
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from w12_crossover_lib import log, pinned_basis  # noqa: E402
from w12_deletion_rows import (  # noqa: E402
    RED_FACTOR, RED_INDEX, SPHERE_INDEX, SHAPE_FACTOR, corner_requests_value,
    factor_indices, match_basis_width,
)

OUT = Path("C:/chlu/outputs/w12-single-design-crossover")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--condition", default="test")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    from chlu.core.downdate_store import carry_address_book
    from chlu.experiments import exp_gen_factor_wells as fw
    from chlu.experiments import exp_gen_read as R
    from chlu.experiments.exp_w8_single_design import CovContentStore
    from w12_crossover_lib import load_partition

    fw.PINS["condition"] = a.condition
    fw.PINS["seed"] = int(a.seed)
    _z, dead, pool = load_partition(a.condition)
    cond = fw.load_condition()
    grid = cond["grid"]
    cards = [int(c) for c in grid.cardinalities]
    corners = (fw.pair_subcorners(cond) + fw.nway_corners(cond)
               if a.condition == "test" else fw.val_corners(cond))
    with open(Path(a.arm) / "arm.pkl", "rb") as fh:
        saved = pickle.load(fh)
    st, addr, basis = saved["store"], saved["address"], saved["basis"]
    wc = [np.asarray(c, np.int64) for c in saved["well_channels"]]
    K, A = int(saved["K"]), int(saved["A"])
    log(f"arm K={K} A={A} cond={a.condition} seed={a.seed}")

    F = fw.features_mmap(int(a.seed))
    X = np.asarray(fw.gather(F, pool), np.float32)
    idx = factor_indices(pool, cards)
    red = idx[:, RED_FACTOR] == RED_INDEX

    obj = {"arm": "cov_tensor", "read": "tensor", "kind": "span", "store": st,
           "basis": basis, "address": addr, "protos": None,
           "head_bytes": {"head_bytes": 0}}

    st2 = copy.deepcopy(st)
    U_pre = pinned_basis(st2)
    st2.delete_rows(X[red])
    U_post_raw = pinned_basis(st2)
    U_post, _ = match_basis_width(U_pre, U_post_raw)
    addr2, carry = carry_address_book(addr, U_pre, U_post)
    obj2 = dict(obj, store=st2, basis=U_post_raw, address=addr2)

    # a THIRD arm: the downdated store read through the UNCARRIED book, so the carry's
    # own contribution is visible rather than bundled into the deletion's.
    obj3 = dict(obj, store=st2, basis=U_post_raw)

    red_ids = {c["corner_id"] for c in corners
               if corner_requests_value(c, grid, RED_FACTOR, RED_INDEX)}
    out = {"condition": a.condition, "seed": int(a.seed), "K": K, "A": A,
           "n_rows_deleted": int(red.sum()),
           "n_corners_requesting_red": len(red_ids),
           "what": ("how far the RENDERED answer moved on exactly the requests whose "
                    "evidence was deleted, in the store's own units. No judge, no decoder."),
           "per_group": {}}
    for tag, sel in (("corners REQUESTING red", [c for c in corners
                                                 if c["corner_id"] in red_ids]),
                     ("corners NOT requesting red", [c for c in corners
                                                     if c["corner_id"] not in red_ids])):
        num = den = 0.0
        num_nc = 0.0
        mx = 0.0
        n = 0
        for c in sel:
            vals = fw._request_values_from_rows(grid, c["rows"])
            f0, _ = R.arm_features(obj, vals, grid)
            f1, _ = R.arm_features(obj2, vals, grid)
            f2, _ = R.arm_features(obj3, vals, grid)
            d = np.abs(f1 - f0)
            num += float((d ** 2).sum())
            num_nc += float(((f2 - f0) ** 2).sum())
            den += float((f0 ** 2).sum())
            mx = max(mx, float(d.max()))
            n += int(vals.shape[0])
        out["per_group"][tag] = {
            "n_requests": n,
            "relative_rms_change_carried": float(np.sqrt(num / max(den, 1e-30))),
            "relative_rms_change_uncarried": float(np.sqrt(num_nc / max(den, 1e-30))),
            "max_abs_change_carried": mx,
        }
        log(f"  {tag:30s} n={n:6d}  rel RMS change (carried) "
            f"{out['per_group'][tag]['relative_rms_change_carried']:.6f}  "
            f"(uncarried) {out['per_group'][tag]['relative_rms_change_uncarried']:.6f}  "
            f"max|delta| {mx:.4e}")

    out["carry"] = {k: (v if not isinstance(v, (list, np.ndarray))
                        else [float(x) for x in np.asarray(v).ravel()[:8]])
                    for k, v in carry.items()}
    sv = np.asarray(carry["correspondence_min_singular_value"], float)
    out["carry_min_singular_value_over_wells"] = float(sv.min())
    out["reading"] = (
        "If the relative RMS change on the red-requesting corners is ~0, the deletion "
        "removed red from the STATISTIC but not from the ANSWER, and the value-specific "
        "information lives in the address book -- which this deletion carries but does "
        "NOT refit. That is a limit of the claim, not a bug in the downdate, and it is "
        "what the store-scoped definition of exactness does and does not cover.")
    p = OUT / "deletion" / f"probe_{a.condition}_seed{a.seed}_K{K}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    log(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
