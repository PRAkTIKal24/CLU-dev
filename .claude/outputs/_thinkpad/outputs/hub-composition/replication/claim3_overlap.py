"""CLAIM 3 -- provenance of the 512 `val` rows.

"Ceiling" and "bound" are held-out words. This measures, with set arithmetic only,
which fitted stages have already SEEN the 512 rows the two numbers are measured on.
"""
import json
import sys

import numpy as np

from chlu.config import get_default_config, gen_row_split_config
from chlu.core import gen_bridge
from chlu.data import gen_rowsplit
from chlu.experiments import exp_gen_store

SEED = 0


def main():
    cfg = get_default_config()
    mod = gen_bridge.dataset_module("shapes3d")
    split, _ = mod.build_split(cfg.gen_data.data_root)
    rows = exp_gen_store.select_store_rows(split, cfg, SEED)
    pool = np.asarray(rows["store_train"], np.int64)
    rs = gen_row_split_config(cfg)
    fit_rows, val_rows, _ = gen_rowsplit.split_fit_val(
        pool, split.grid.factor_indices_of(pool), int(cfg.gen_cell.n_fit),
        int(cfg.gen_cell.n_val), SEED, mode=str(rs.mode),
        tiebreak_jitter=float(rs.tiebreak_jitter), seed_mult=int(rs.seed_mult),
        seed_offset=int(rs.seed_offset), log=None)
    V = np.asarray(val_rows, np.int64)
    sets = {
        "store_train (the rows the STORE was fitted on)": pool,
        "bridge_train (encoder/decoder training rows)": np.asarray(rows["bridge_train"], np.int64),
        "eval_a": np.asarray(rows["eval_a"], np.int64),
        "eval_b (the 72 held-out B)": np.asarray(rows["eval_b"], np.int64),
        "fit (prototype/rival fit rows)": np.asarray(fit_rows, np.int64),
        "split.test_rows": np.asarray(split.test_rows, np.int64),
    }
    out = {"n_val": int(V.size), "overlaps": {}}
    for k, s in sets.items():
        n = int(np.intersect1d(V, s).size)
        out["overlaps"][k] = {"n_in_that_set": n, "set_size": int(s.size),
                              "frac_of_val": n / float(V.size)}
        print("%-50s %5d / %d val rows" % (k, n, V.size))
    print("val rows disjoint from fit:", out["overlaps"]["fit (prototype/rival fit rows)"]["n_in_that_set"] == 0)
    return out


if __name__ == "__main__":
    r = main()
    if len(sys.argv) > 1:
        with open(sys.argv[1], "w") as fh:
            json.dump(r, fh, indent=1)
        print("wrote", sys.argv[1])
