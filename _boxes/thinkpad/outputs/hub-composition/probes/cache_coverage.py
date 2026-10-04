"""Exactly how many rows the CELL needs that its seed's frozen cache does not have.
⛔ Reads only. Encodes nothing. Writes only under $OUT."""
import json, os, sys
import numpy as np
from chlu.config import get_default_config
from chlu.core.gen_bridge import dataset_module, load_feature_cache
from chlu.experiments.exp_gen_store import bridge_dir, select_store_rows, matched_seen_rows
from chlu.data.gen_rowsplit import split_fit_val
from chlu.config import gen_row_split_config

cfg = get_default_config()
ds, px = "shapes3d", "recon"
mod = dataset_module(ds)
split, values = mod.build_split(cfg.gen_data.data_root)
g = cfg.gen_cell
rs = gen_row_split_config(cfg)
out = {}
for seed in (0, 1, 2):
    bdir = bridge_dir(cfg, ds, px, seed)
    crows, feats, cman = load_feature_cache(bdir, verify=False)
    have = set(np.asarray(crows, np.int64).tolist())
    rows = select_store_rows(split, cfg, seed)
    pool = np.asarray(rows["store_train"], np.int64)
    fit, val, _ = split_fit_val(pool, split.grid.factor_indices_of(pool),
                               int(g.n_fit), int(g.n_val), int(seed),
                               mode=str(rs.mode), tiebreak_jitter=float(rs.tiebreak_jitter),
                               seed_mult=int(rs.seed_mult), seed_offset=int(rs.seed_offset),
                               log=lambda *a: None)
    need = {"store_train": pool, "eval_b": np.asarray(rows["eval_b"], np.int64),
            "MS": np.asarray(matched_seen_rows(ds, split, values, rows["store_train"]), np.int64),
            "fit": fit, "val": val}
    allneed = np.unique(np.concatenate([np.asarray(v, np.int64).ravel() for v in need.values()]))
    miss = {k: int(sum(1 for r in np.asarray(v).ravel().tolist() if int(r) not in have))
            for k, v in need.items()}
    out[seed] = {"cache_scope": cman["scope"], "cache_n_rows": int(cman["n_rows"]),
                 "n_needed_union": int(allneed.size),
                 "n_missing_union": int(sum(1 for r in allneed.tolist() if int(r) not in have)),
                 "missing_by_set": miss,
                 "sizes": {k: int(np.asarray(v).size) for k, v in need.items()}}
    print(seed, json.dumps(out[seed]))
p = os.path.join(os.environ["OUT"], "probes", "cache_coverage.json")
json.dump(out, open(p, "w"), indent=2)
print("wrote", p)
