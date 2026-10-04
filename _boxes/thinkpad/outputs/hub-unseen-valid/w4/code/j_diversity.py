"""⭐ THE DISTINCT-OUTPUT CHECK — does `e = |p-0.5|` reward a BROKEN CLOCK?

`w4-hub-common.md` §1 and spoke A's instrument both insist an accuracy is never read
without its distinct-output count. My own run surfaced the reason: the zeroed-atom
control `landscape_removed` collapses to CHANCE in the mean (0.2514 vs 0.2500 exactly)
yet still beats the arm on 41 of 70 corners on seed 0. A CONSTANT render can sit near
the requested midpoint by accident.

⛔ The danger to my headline: if `cont_mlp_ARMMATCHED` also produces low-diversity
output, part of its win over the arm is the same artefact rather than a better read.
This measures diversity directly, per arm and per corner:
  * sd of the RAW oracle-referenced readout (a constant render gives ~0),
  * distinct decoded FACTOR TUPLES (judge 1), overall and per corner,
  * distinct rendered FEATURE ROWS.
⛔ Diagnostic only. It adds no comparison against the margin and spends no scored run.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import w4_common as C  # noqa: E402
import valley as V  # noqa: E402
import chlu.experiments.exp_gen_factor_wells as W  # noqa: E402
from j_readout_choice import hue_logits  # noqa: E402


def main() -> None:
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    W.PINS["condition"] = "test"
    W.PINS["seed"] = seed
    C.use_cache_generation2((seed,))
    from chlu.core.gen_request_head import head_coefficients, head_read, request_design
    import cont_mlp as M

    t0 = time.time()
    row2, grid, train_rows = C.build_row2(4096, seed=0)
    recs = row2["records"][:4096]
    rows, hues, vals = C.bracket_rows(recs, grid, train_rows)
    cids = C.corner_ids(recs, grid, vals, hues)
    odir = C.OUT / "objects" / f"test_s{seed}"
    obj = W.build_arm("factor_aligned_balanced", odir, log=print)
    rem = W.build_arm("landscape_removed", odir, log=print)
    Phi, _ = request_design(vals, grid.values, obj["spec"])
    store, heads = obj["store"], obj["heads"]

    feats = {"linear_span_ARM": head_read(heads, store, Phi),
             "landscape_removed_CONTROL": W.arm_features(rem, vals, grid)[0]}
    Cc = head_coefficients(heads, Phi)
    look = np.zeros((Phi.shape[0], int(store.dim)), np.float32)
    for c, w, G in zip(Cc, store.well_channels, store.modes):
        look[:, np.asarray(w, np.int64)] = np.asarray(G, np.float32)[np.argmax(c, 1)]
    feats["lookup_argmax_FLOOR"] = look
    pool = np.asarray(np.load(odir / "partitions.npz")["pool"], np.int64)
    F = W.features_mmap(seed)
    tgt = (int(obj["byte_ledger"]["deployed_bytes"])
           + int(obj["head_bytes"]["head_bytes"])) // 4
    mdl, info = M.fit(grid, pool, F, tgt, seed=seed, steps=4000)
    feats["cont_mlp_ARMMATCHED_RIVAL"] = M.predict(mdl, vals, grid)
    print(f"[div] mlp refit {info['n_params']:,} params, mse {info['final_mse']:.4f}")

    J = C.SecondJudge(seed)
    i_lo = np.round(hues[:, 0] / 0.1).astype(int)
    i_hi = np.round(hues[:, 1] / 0.1).astype(int)
    n = np.arange(len(recs))
    res = {"seed": seed, "note": "diagnostic only; no margin comparison, no scored run",
           "arms": {}}
    for name, X in feats.items():
        L = hue_logits(J.judge, X, J.k_hue, 128)
        r = L[n, i_hi] - L[n, i_lo]
        tup = J.judge.read_back(X, 256)
        lin = np.ravel_multi_index(tuple(tup.T), tuple(int(c) for c in grid.cardinalities))
        per_corner = [len(np.unique(lin[cids == c])) for c in sorted(set(cids))]
        rows_u = len(np.unique(np.round(np.asarray(X, np.float64), 4), axis=0))
        res["arms"][name] = {
            "readout_sd": float(r.std()),
            "readout_range": [float(r.min()), float(r.max())],
            "distinct_tuples_overall": int(len(np.unique(lin))),
            "distinct_tuples_per_corner_mean": float(np.mean(per_corner)),
            "distinct_tuples_per_corner_min": int(np.min(per_corner)),
            "distinct_feature_rows": int(rows_u),
            "n_requests": int(len(recs)),
        }
        print(f"[div] {name:32s} readout_sd={r.std():8.3f}  distinct_tuples={len(np.unique(lin)):5d}"
              f"  per_corner_mean={np.mean(per_corner):6.2f}  distinct_feature_rows={rows_u:5d}")
    res["seconds"] = time.time() - t0
    C.jdump(res, C.OUT / "runs" / f"diversity_s{seed}.json")
    print(f"[div] wrote diversity_s{seed}.json in {res['seconds']:.0f}s")


if __name__ == "__main__":
    main()
