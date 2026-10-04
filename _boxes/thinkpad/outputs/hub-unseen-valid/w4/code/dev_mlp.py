"""DEV: the continuous-input MLP rival vs the CLU read, on the second judge, on val."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import w4_common as C  # noqa: E402
import cont_mlp as M  # noqa: E402
import chlu.experiments.exp_gen_factor_wells as W  # noqa: E402
from dev_eval import make_anchors, second_judge_e, per_corner, paired  # noqa: E402

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="val")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--n-select", type=int, default=512)
    ap.add_argument("--batch", type=int, default=128)
    a = ap.parse_args()
    W.PINS["condition"] = a.condition

    from chlu.core.gen_request_head import head_read, request_design
    row2, grid, train_rows = C.build_row2(8192, seed=0)
    recs = row2["records"][4096:4096 + a.n_select]
    rows, hues, vals = C.bracket_rows(recs, grid, train_rows)
    cids = C.corner_ids(recs, grid, vals, hues)

    odir = C.OUT / "objects" / a.condition
    obj = W.build_arm("factor_aligned_balanced", odir, log=print)
    proto = W.build_arm("knn_proto", odir, log=print)
    target = (int(obj["byte_ledger"]["deployed_bytes"]) +
              int(obj["head_bytes"]["head_bytes"])) // 4
    pool = np.asarray(np.load(odir / "partitions.npz")["pool"], np.int64)
    F = W.features_mmap(a.seed)

    t0 = time.time()
    model, info = M.fit(grid, pool, F, target, seed=a.seed, steps=a.steps)
    print(f"[mlp] fitted in {time.time()-t0:.0f}s: {info}")

    Phi, _ = request_design(vals, grid.values, obj["spec"])
    feats = {"clu_linear_span": head_read(obj["heads"], obj["store"], Phi),
             "table_interp_knn_proto": W.arm_features(proto, vals, grid)[0],
             "cont_input_mlp_RIVAL": M.predict(model, vals, grid)}
    J = C.SecondJudge(a.seed)
    A = make_anchors(J, rows, hues, a.batch)
    pc, summ = {}, {}
    for k, X in feats.items():
        e, p = second_judge_e(J, X, A, a.batch)
        pc[k] = per_corner(cids, e)
        summ[k] = {"mean_e": float(e.mean()), "mean_p": float(p.mean()),
                   "frac_interior_20_80": float(((p > .2) & (p < .8)).mean())}
        print(f"[mlp] {k:26s} e={e.mean():.4f} p={p.mean():+.3f} "
              f"interior={((p>.2)&(p<.8)).mean():.3f}")
    res = {"condition": a.condition, "seed": a.seed, "mlp": info,
           "param_match_target": target, "reads": summ,
           "clu_vs_mlp": paired(pc["clu_linear_span"], pc["cont_input_mlp_RIVAL"]),
           "n_corners": len(set(cids))}
    C.jdump(res, C.OUT / "runs" / f"dev_mlp_{a.condition}_s{a.seed}.json")
    d = res["clu_vs_mlp"]
    print(f"[mlp] PAIRED clu - mlp: {d['mean_delta']:+.4f} ci95="
          f"[{d['ci95'][0]:+.4f},{d['ci95'][1]:+.4f}] "
          f"(negative = CLU better) corners {d['corners_better']}/{d['corners_worse']}")
