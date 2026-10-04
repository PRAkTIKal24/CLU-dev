"""w6.1 retry -- the three cheap closures, on the parent's already-paid rows.

1. ⭐ **Re-verify the headline** the retry is built on: row-4 novel valid tuples per
   corner for each scored leg, and the delta against `pca_tensor`, recomputed from
   `rows.npz` rather than read off the parent's table.
2. ⭐ **Is `pca_settle_f05_conv`'s loss real or noise?**  It is negative on all three
   seeds but inside the pre-registered margin.  Three signs is weak evidence on its own,
   so this pairs the legs **per corner** (49 corners x 3 seeds) and **per request**
   (10,584 requests, McNemar on the validity flips), which is the test the design
   actually supports.
3. ⚠ **What the 81,312 B of read-side overhead would have to buy.**  Priced against the
   store it sits beside and against the margin the incumbent already holds.

Reads only artifacts.  No judge, nothing scored.

    python scripts/w61_close.py --root DIR --out FILE.json
"""
from __future__ import annotations

import argparse
import json
from math import exp, lgamma, log
from pathlib import Path

import numpy as np

from chlu.experiments import exp_gen_factor_wells as fw
from chlu.experiments import exp_gen_read as R
from chlu.experiments import exp_w6_compose_physics as P

LEGS = ("pca_tensor", "pca_settle_zerostep", "pca_settle_f05_conv",
        "pca_settle_onestep", "pca_settle")


def sign_test_p(k: int, n: int) -> float:
    """Two-sided exact binomial p at q = 1/2 for k successes in n trials.

    Summed in log space: n runs to a few thousand here and 2**n overflows a float.
    """
    if n == 0:
        return float("nan")
    lo = min(int(k), n - int(k))
    terms = [lgamma(n + 1) - lgamma(i + 1) - lgamma(n - i + 1) - n * log(2.0)
             for i in range(lo + 1)]
    mx = max(terms)
    tail = mx + log(sum(exp(t - mx) for t in terms))
    return float(min(1.0, 2.0 * exp(tail)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="C:/chlu/outputs/w6-compose-physics/scored")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    P.install()

    out = {"per_seed": {}, "legs": LEGS}
    pooled = {l: {"corner_delta": [], "flip_lose": 0, "flip_gain": 0}
              for l in LEGS if l != "pca_tensor"}

    for seed in (0, 1, 2):
        leg = Path(a.root) / f"test_seed{seed}"
        R.set_condition(seed, "test")
        cond = fw.load_condition()
        grid = cond["grid"]
        cards = tuple(int(c) for c in grid.cardinalities)
        corners = fw.nway_corners(cond)
        seen = fw.seen_tuple_mask(grid, cond["train_rows"])
        per = {}
        base_novel = None
        base_valid = None
        for arm in LEGS:
            p = leg / "arms" / arm / "rows.npz"
            if not p.exists():
                continue
            cor = np.load(p, allow_pickle=True)["corners"].item()
            novel, valid = [], []
            for c in corners:
                r = cor[c["corner_id"]]
                v = np.asarray(r["valid"], bool)
                t = np.asarray(r["tuples"])
                keep = t[v]
                if keep.size:
                    lin = np.unique(np.ravel_multi_index(tuple(keep.T), cards))
                    novel.append(int((~seen[lin]).sum()))
                else:
                    novel.append(0)
                valid.append(v)
            novel = np.asarray(novel, np.int64)
            valid = np.concatenate(valid)
            if arm == "pca_tensor":
                base_novel, base_valid = novel, valid
            blk = {"novel_per_corner_mean": float(novel.mean()),
                   "validity": float(valid.mean())}
            if arm != "pca_tensor":
                dlt = novel - base_novel
                lose = int((base_valid & ~valid).sum())
                gain = int((~base_valid & valid).sum())
                blk.update({
                    "delta_novel_per_corner": float(dlt.mean()),
                    "n_corners_worse": int((dlt < 0).sum()),
                    "n_corners_better": int((dlt > 0).sum()),
                    "n_corners_tied": int((dlt == 0).sum()),
                    "sign_test_p_over_corners":
                        sign_test_p(int((dlt > 0).sum()),
                                    int((dlt != 0).sum())),
                    "requests_valid_to_invalid": lose,
                    "requests_invalid_to_valid": gain,
                    "mcnemar_p": sign_test_p(gain, lose + gain),
                })
                pooled[arm]["corner_delta"].extend(dlt.tolist())
                pooled[arm]["flip_lose"] += lose
                pooled[arm]["flip_gain"] += gain
            per[arm] = blk
        out["per_seed"][str(seed)] = per

    out["pooled_over_3_seeds"] = {}
    for arm, d in pooled.items():
        dlt = np.asarray(d["corner_delta"], np.float64)
        nb, nw = int((dlt > 0).sum()), int((dlt < 0).sum())
        out["pooled_over_3_seeds"][arm] = {
            "n_corner_pairs": int(dlt.size),
            "mean_delta_novel_per_corner": float(dlt.mean()),
            "median_delta_novel_per_corner": float(np.median(dlt)),
            "corners_better": nb, "corners_worse": nw,
            "corners_tied": int((dlt == 0).sum()),
            "sign_test_p_over_corners": sign_test_p(nb, nb + nw),
            "requests_valid_to_invalid": d["flip_lose"],
            "requests_invalid_to_valid": d["flip_gain"],
            "mcnemar_p": sign_test_p(d["flip_gain"],
                                     d["flip_lose"] + d["flip_gain"]),
        }

    # -- 3. the price of the read-side overhead ---------------------------
    lj = json.load(open(Path(a.root) / "test_seed0" / "span_coupling_r16.json",
                        encoding="utf-8"))["byte_ledger"]
    aj = json.load(open(Path(a.root) / "test_seed0" / "arms" / "pca_settle" /
                        "arm.json", encoding="utf-8"))
    store_b = int(aj["byte_ledger"]["deployed_bytes"])
    addr_b = int(aj["address_bytes"].get("address_bytes", 0))
    add = int(lj["total_read_side_bytes_added"])
    out["byte_price"] = {
        "store_bytes": store_b,
        "address_bytes": addr_b,
        "coupling_bytes": int(lj["coupling_bytes"]),
        "standardisation_bytes": int(lj["standardisation_bytes"]),
        "total_added": add,
        "added_as_frac_of_store": add / store_b,
        "added_as_frac_of_store_plus_address": add / (store_b + addr_b),
        "added_float32_count": add // 4,
    }
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out["pooled_over_3_seeds"], indent=1))
    print(json.dumps(out["byte_price"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
