"""⛔ Is the novelty clause satisfiable at all, under the STRICTEST reference?
Not "not a stored ATOM's tuple" (64 rows) but "not the tuple of ANY row the store was
trained on" (32,768) and "not the tuple of ANY TRAIN row at all" (479,928).
⛔ Reads only. B's LABELS are used, which the split defines publicly; no render, no score."""
import os, json
import numpy as np
from chlu.config import get_default_config
from chlu.core.gen_bridge import dataset_module
from chlu.eval.gen_probe import factor_labels
from chlu.experiments.exp_gen_store import select_store_rows

cfg = get_default_config()
mod = dataset_module("shapes3d")
split, values = mod.build_split(cfg.gen_data.data_root)
grid = split.grid
out = {}
B = np.asarray(split.test_rows, np.int64)
labB = factor_labels(grid, B)
uB = np.unique(labB, axis=0)
train = np.asarray(split.train_rows, np.int64)
labTr = factor_labels(grid, train)
setTr = set(map(tuple, labTr.tolist()))
for seed in (0, 1, 2):
    rows = select_store_rows(split, cfg, seed)
    pool = np.asarray(rows["store_train"], np.int64)
    setPool = set(map(tuple, factor_labels(grid, pool).tolist()))
    out[seed] = {
        "n_B": int(B.size), "n_B_distinct_tuples": int(uB.shape[0]),
        "n_B_tuples_NOT_in_store_train_pool": int(sum(1 for t in map(tuple, uB.tolist()) if t not in setPool)),
        "n_B_tuples_NOT_in_ANY_train_row": int(sum(1 for t in map(tuple, uB.tolist()) if t not in setTr)),
        "n_store_train": int(pool.size),
    }
    print(f"seed {seed}: B rows={out[seed]['n_B']} distinct tuples={out[seed]['n_B_distinct_tuples']} "
          f"| novel vs store_train pool ({pool.size} rows): {out[seed]['n_B_tuples_NOT_in_store_train_pool']}/72 "
          f"| novel vs ALL {train.size} train rows: {out[seed]['n_B_tuples_NOT_in_ANY_train_row']}/72")
json.dump(out, open(os.path.join(os.environ["OUT"], "probes", "novelty_reference.json"), "w"), indent=2)
print("done")
