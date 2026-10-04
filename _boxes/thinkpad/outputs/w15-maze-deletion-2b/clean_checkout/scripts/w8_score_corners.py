"""Score the matched-bytes arm on the corners, through the SHARED frozen decoder+judge.

Mirrors `exp_w8_downdate.stage_smoke`'s `_run` exactly, so the number is produced by the
same code path as the spoke's and the two are comparable.
⛔ On `val` this is a SMOKE and claims nothing (charter A.2's free selection condition).
"""
from __future__ import annotations
import argparse, json, pickle, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def log(m=""):
    print(m, flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--condition", default="val", choices=("val", "test"))
    a = ap.parse_args(argv)

    from chlu.experiments import exp_gen_factor_wells as fw
    from chlu.experiments import exp_gen_read as R
    fw.PINS["condition"] = a.condition
    with open(Path(a.arm) / "arm.pkl", "rb") as fh:
        A = pickle.load(fh)
    obj = {"arm": "cov_tensor_matched_bytes", "read": "tensor", "kind": "span",
           "store": A["store"], "basis": A["basis"], "address": A["address"],
           "protos": None, "head_bytes": {"head_bytes": 0}}
    cond = fw.load_condition()
    grid = cond["grid"]
    judge = fw.Judge(int(fw.PINS["seed"]), log=log)
    bitmap = fw.value_presence_bitmap(
        grid, np.setdiff1d(cond["train_rows"], fw.val_corner_rows(cond)))
    corners = (fw.val_corners(cond) if a.condition == "val"
               else fw.nway_corners(cond) + fw.pair_subcorners(cond))
    log(f"{len(corners)} {a.condition} corners through the shared frozen decoder+judge")
    rows = []
    t0 = time.time()
    for i, c in enumerate(corners):
        vals = fw._request_values_from_rows(grid, c["rows"])
        feats, oor = R.arm_features(obj, vals, grid)
        acc = fw.accept_rule(vals, oor, np.zeros(vals.shape[0], bool), bitmap, grid)
        tup = judge.read_back(feats, int(fw.PINS["eval_batch"]))
        valid = fw.judge_valid(vals, tup, grid) & acc
        rows.append({"corner_id": c["corner_id"], "n": int(vals.shape[0]),
                     "accept": float(acc.mean()), "valid": float(valid.mean())})
        if (i + 1) % 10 == 0:
            log(f"  {i+1}/{len(corners)} corners, {time.time()-t0:.0f}s")
    v = np.asarray([r["valid"] for r in rows], np.float64)
    log(f"  cov_tensor_matched_bytes  valid mean {v.mean():.4f}  sd {v.std(ddof=1):.4f}"
        f"  min {v.min():.4f}  max {v.max():.4f}  over {len(rows)} corners")
    out = Path(a.arm) / f"corners_{a.condition}.json"
    out.write_text(json.dumps({
        "SMOKE_NOT_A_CLAIM": a.condition == "val",
        "condition": a.condition, "n_corners": len(rows),
        "valid_mean": float(v.mean()), "valid_sd": float(v.std(ddof=1)),
        "valid_min": float(v.min()), "valid_max": float(v.max()),
        "per_corner": rows}, indent=2), encoding="utf-8")
    log(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
