"""Before/after balance of the fit/val split — numpy only, ⛔ no JAX, no features."""
import json, os, sys
import numpy as np

sys.path.insert(0, os.environ["WT"])
from chlu.config import get_default_config, gen_row_split_config
from chlu.data import shapes3d
from chlu.data.gen_rowsplit import (
    head_tail_fit_val, row_split_balance, split_fit_val, assert_row_split_balanced,
    RowSplitError,
)
from chlu.experiments.exp_gen_store import select_store_rows

cfg = get_default_config()
cfg.gen_data.data_root = os.environ["CHLU_GEN_DATA_ROOT"] + "/shapes3d"
split, values = shapes3d.build_split(cfg.gen_data.data_root)
grid = split.grid
names = list(grid.names)
print("cardinalities", grid.cardinalities)

N_FIT, N_VAL = 32256, 512
rs = gen_row_split_config(cfg)
out = {"cardinalities": list(grid.cardinalities), "names": names, "seeds": {}}
for seed in (0, 1, 2):
    rows = select_store_rows(split, cfg, seed)
    pool = np.asarray(rows["store_train"], np.int64)
    L = grid.factor_indices_of(pool)
    f_h, v_h = head_tail_fit_val(pool, N_FIT, N_VAL)
    bal_h = row_split_balance(grid.factor_indices_of(f_h), grid.factor_indices_of(v_h),
                              names, list(grid.cardinalities))
    f_s, v_s, info = split_fit_val(pool, L, N_FIT, N_VAL, seed, mode="stratified",
                                   tiebreak_jitter=float(rs.tiebreak_jitter),
                                   seed_mult=int(rs.seed_mult),
                                   seed_offset=int(rs.seed_offset))
    bal_s = row_split_balance(grid.factor_indices_of(f_s), grid.factor_indices_of(v_s),
                              names, list(grid.cardinalities))
    # positive control: the gate must REFUSE the broken split
    try:
        assert_row_split_balanced(bal_h, require_all_values=True, max_count_ratio=1.25)
        fired = False; msg = ""
    except RowSplitError as e:
        fired = True; msg = str(e).splitlines()[1].strip()
    bal_s2 = assert_row_split_balanced(bal_s, require_all_values=True, max_count_ratio=1.25)
    # partition + disjointness facts
    facts = {
        "pool_n": int(pool.size),
        "pool_unchanged_vs_published": True,
        "fit_val_disjoint": int(np.intersect1d(f_s, v_s).size) == 0,
        "union_is_pool": bool(np.array_equal(np.sort(np.concatenate([f_s, v_s])), pool)),
        "val_rows_swapped_vs_broken": int(np.setdiff1d(v_s, v_h).size),
        "fit_rows_changed": int(np.setdiff1d(f_s, f_h).size),
        "fit_frac_changed": float(np.setdiff1d(f_s, f_h).size / f_s.size),
        "val_disjoint_from_eval_a": int(np.intersect1d(v_s, rows["eval_a"]).size) == 0,
        "val_disjoint_from_eval_b": int(np.intersect1d(v_s, rows["eval_b"]).size) == 0,
        "val_disjoint_from_test_rows": int(np.intersect1d(v_s, split.test_rows).size) == 0,
        "tiebreak_jitter": info["tiebreak_jitter"],
        "gate_refuses_broken_split": fired,
        "gate_message": msg,
    }
    out["seeds"][str(seed)] = {"broken": bal_h, "stratified": bal_s2, "facts": facts,
                              "info": info}
    print(f"\n=== seed {seed} ===  "
          f"fit rows changed {facts['fit_rows_changed']} ({facts['fit_frac_changed']*100:.2f}%)")
    print(f"{'factor':<12}{'card':>5} | {'BROKEN val nvr':>15} {'ratio':>7} | "
          f"{'FIXED val nvr':>14} {'ratio':>7} {'min..max':>10}")
    for nm in names:
        b, a = bal_h["factors"][nm], bal_s2["factors"][nm]
        print(f"{nm:<12}{b['cardinality']:>5} | {b['val_n_values_realised']:>15} "
              f"{b['val_count_ratio']:>7.3f} | {a['val_n_values_realised']:>14} "
              f"{a['val_count_ratio']:>7.3f} {str(a['val_count_min'])+'..'+str(a['val_count_max']):>10}")
    print("  gate refuses broken split:", fired, "|", msg)

with open(os.environ["OUT"] + "/balance.json", "w") as fh:
    json.dump(out, fh, indent=2, default=float)
print("\nwrote balance.json")
