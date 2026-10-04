"""Why does spoke A's `table` (0.1581) beat my `knn_proto` (0.2558) at identical bytes?

Both are additive FactorPrototypes composition at 475,136 B = (sum_k C_k + 1) x D.
The ONE declared difference is the fit pool: theirs 32,256 rows, mine the 131,072-row
pinned pool. This refits mine at a range of pool sizes and scores each on the SAME
requests with the SAME second judge. ⛔ Diagnostic only -- it changes no scored number.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import w4_common as C  # noqa: E402
import chlu.experiments.exp_gen_factor_wells as W  # noqa: E402
from dev_eval import make_anchors, second_judge_e, per_corner  # noqa: E402

if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    W.PINS["condition"] = "test"
    W.PINS["seed"] = seed
    C.use_cache_generation2((seed,))
    from chlu.core.gen_request import FactorPrototypes

    row2, grid, train_rows = C.build_row2(4096, seed=0)
    recs = row2["records"][:4096]
    rows, hues, vals = C.bracket_rows(recs, grid, train_rows)
    cids = C.corner_ids(recs, grid, vals, hues)
    odir = C.OUT / "objects" / f"test_s{seed}"
    pool = np.asarray(np.load(odir / "partitions.npz")["pool"], np.int64)
    F = W.features_mmap(seed)
    J = C.SecondJudge(seed)
    A = make_anchors(J, rows, hues, 128)

    res = {"seed": seed, "note": "diagnostic only, not a scored number", "by_pool": {}}
    for n in (32_256, 65_536, 131_072):
        sub = pool[:n]
        p = FactorPrototypes.fit(grid, W.gather(F, sub), sub)
        reqs = [[{"factor_index": f, "value": float(vals[i, f])}
                 for f in range(grid.n_factors)] for i in range(vals.shape[0])]
        e, pp = second_judge_e(J, p.embed_many(reqs), A, 128)
        res["by_pool"][str(n)] = {"mean_e": float(e.mean()), "mean_p": float(pp.mean()),
                                  "bytes": int(p.byte_ledger()["bytes"])}
        print(f"[tablegap] pool={n:>7d}  mean_e={e.mean():.4f}  mean_p={pp.mean():+.3f}  "
              f"bytes={p.byte_ledger()['bytes']:,}")
    z = np.load(C.OUT / "runs" / f"spokea_rivals_s{seed}_n4096.npz", allow_pickle=True)
    e, pp = second_judge_e(J, np.asarray(z["feats__table"], np.float32), A, 128)
    res["spokeA_table"] = {"mean_e": float(e.mean()), "mean_p": float(pp.mean())}
    print(f"[tablegap] spokeA_table    mean_e={e.mean():.4f}  mean_p={pp.mean():+.3f}")
    C.jdump(res, C.OUT / "runs" / f"table_gap_s{seed}.json")
