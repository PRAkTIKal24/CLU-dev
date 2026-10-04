"""CLAIM 2 -- held-out B target tuples: count, distinctness, novelty vs train.

Independent: builds the split from the library primitives named in the charter and
does all counting with plain numpy set logic. No hub scripts imported.
"""
import json, sys
import numpy as np

from chlu.core.gen_bridge import dataset_module
from chlu.config import get_default_config
from chlu.eval.gen_probe import factor_labels


def tuple_set(arr):
    return set(map(tuple, np.asarray(arr).tolist()))


def main():
    cfg = get_default_config()
    mod = dataset_module("shapes3d")
    split, values = mod.build_split(cfg.gen_data.data_root)

    test_rows = np.asarray(split.test_rows, np.int64)
    train_rows = np.asarray(split.train_rows, np.int64)

    test_t = factor_labels(split.grid, test_rows)
    train_t = factor_labels(split.grid, train_rows)

    n_test = int(test_rows.size)
    test_set = tuple_set(test_t)
    n_distinct = len(test_set)
    train_set = tuple_set(train_t)
    overlap = sorted(test_set & train_set)
    n_overlap = len(overlap)

    # row-level: how many individual test rows carry a tuple that also occurs in train
    rows_overlapping = int(sum(1 for t in map(tuple, test_t.tolist()) if t in train_set))

    # sanity: is the row->tuple map injective on the WHOLE grid?
    n_rows_total = int(train_rows.size + test_rows.size)
    grid_injective = (len(train_set) == int(train_rows.size)) and (n_distinct == n_test)

    res = dict(
        n_test_rows=n_test,
        n_distinct_test_tuples=n_distinct,
        n_test_tuples_also_in_train=n_overlap,
        n_test_rows_whose_tuple_is_in_train=rows_overlapping,
        n_train_rows=int(train_rows.size),
        n_distinct_train_tuples=len(train_set),
        n_rows_total=n_rows_total,
        grid_row_to_tuple_injective=bool(grid_injective),
        n_factors=int(test_t.shape[1]),
        factor_names=list(getattr(split.grid, "factor_names", []) or []),
        test_tuple_examples=[list(map(int, t)) for t in sorted(test_set)[:5]],
        test_factor_value_ranges={
            i: sorted(set(int(v) for v in test_t[:, i])) for i in range(test_t.shape[1])
        },
    )
    for k, v in res.items():
        print(f"{k}: {v}")
    return res


if __name__ == "__main__":
    r = main()
    if len(sys.argv) > 1:
        json.dump(r, open(sys.argv[1], "w"), indent=1)
        print("wrote", sys.argv[1])
