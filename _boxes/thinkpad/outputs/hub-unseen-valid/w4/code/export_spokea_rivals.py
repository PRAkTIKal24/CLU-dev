"""Run SPOKE A's OWN fitted rivals through MY row-2 requests and save the features.

⭐ Run with PYTHONPATH pointing at spoke A's worktree, so the rival is evaluated by the
code that fitted it. Spoke A's worktree is at the SAME commit as mine (9dfed5e) with the
rival API as untracked ADDITIONS only (`git status` shows no tracked modification), so
every shared file -- including the row-2 builder -- is byte-identical.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

OUT = Path("C:/chlu/outputs/hub-unseen-valid/w4")


def main() -> None:
    seed = int(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
    dest = OUT / "runs" / f"spokea_rivals_s{seed}_n{n}.npz"
    if dest.exists():
        print(f"[spokeA] {dest.name} exists -- skipping")
        return

    import dataclasses as dc
    from chlu.data.gen_splits import GenSplit
    from chlu.data.gen_value_holdout import ValueHoldoutConfig, build_row2_unseen
    import chlu.experiments.exp_gen_factor_wells as W
    from chlu.eval.rivals.w4_api import available, load_rival

    grid, z = W.build_grid()
    train_rows = np.asarray(z["train_rows"], np.int64)
    test_rows = np.asarray(z["test_rows"], np.int64)
    cells = json.load(open(W.data_root() / "value_holdout" / "value_cells.json",
                           encoding="utf-8"))
    base = ValueHoldoutConfig(**{k: v for k, v in cells["config"].items()
                                 if k in {f.name for f in dc.fields(ValueHoldoutConfig)}})
    cfg = dc.replace(base, row2_n_requests=n, row2_seed=0)
    split = GenSplit(dataset=grid.dataset, condition="value_holdout",
                     variant=str(cells.get("variant", "value_holdout")), grid=grid,
                     train_rows=train_rows, test_rows=test_rows,
                     test_predicate=lambda v: np.zeros(len(v), bool))
    row2 = build_row2_unseen(grid, split, cfg)
    recs = row2["records"]
    vals = np.zeros((len(recs), grid.n_factors), np.float64)
    for i, rec in enumerate(recs):
        for t in rec["request"]:
            vals[i, int(t["factor_index"])] = float(t["value"])
    print(f"[spokeA] seed {seed}: {len(recs)} row-2 requests built")
    print(f"[spokeA] available(seed={seed}): {json.dumps(available(seed))[:400]}")

    out = {"values": vals}
    for name in ("table", "cont_mlp", "mlp", "transformer", "lstm"):
        try:
            r = load_rival(seed, name)
            assert list(r.names) == list(grid.names), (
                f"{name}: factor order {r.names} != grid order {grid.names}")
            feats, oor = r(vals)
            out[f"feats__{name}"] = np.asarray(feats, np.float32)
            out[f"oor__{name}"] = np.asarray(oor, bool)
            print(f"[spokeA]   {name}: feats {feats.shape} oor {int(np.sum(oor))}")
        except Exception as ex:                      # RivalRefusal and friends
            out[f"refused__{name}"] = np.array(str(ex)[:300])
            print(f"[spokeA]   {name}: REFUSED/UNAVAILABLE -- {type(ex).__name__}: {str(ex)[:160]}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, **out)
    print(f"[spokeA] wrote {dest}")


if __name__ == "__main__":
    main()
