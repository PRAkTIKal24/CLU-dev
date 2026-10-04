"""Export SPOKE A's rivals REFITTED ON MY VALUE-HOLDOUT POOL, through their own API.

⭐ Run with PYTHONPATH on spoke A's worktree (same commit 9dfed5e, untracked additions only),
so their object is evaluated by the code that fitted it.
⛔ Asserts the POPULATION off each checkpoint before exporting anything (PREREG-2 §3).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

OUT = Path("C:/chlu/outputs/hub-unseen-valid/w4")
ROOT = r"C:/chlu/outputs/spoke-a-rivals/rivals_vhpool"
WANT = ("cont_mlp_notune_572k", "cont_mlp_notune", "table", "cont_mlp")


def main() -> None:
    seed = int(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
    dest = OUT / "runs" / f"vhpool_rivals_s{seed}_n{n}.npz"
    if dest.exists():
        print(f"[vh] {dest.name} exists -- skipping")
        return

    import dataclasses as dc
    from chlu.data.gen_splits import GenSplit
    from chlu.data.gen_value_holdout import ValueHoldoutConfig, build_row2_unseen
    import chlu.experiments.exp_gen_factor_wells as W
    from chlu.eval.rivals.w4_api import load_rival

    grid, z = W.build_grid()
    tr = np.asarray(z["train_rows"], np.int64)
    te = np.asarray(z["test_rows"], np.int64)
    cells = json.load(open(W.data_root() / "value_holdout" / "value_cells.json", encoding="utf-8"))
    base = ValueHoldoutConfig(**{k: v for k, v in cells["config"].items()
                                 if k in {f.name for f in dc.fields(ValueHoldoutConfig)}})
    cfg = dc.replace(base, row2_n_requests=n, row2_seed=0)
    split = GenSplit(dataset=grid.dataset, condition="value_holdout",
                     variant=str(cells.get("variant", "value_holdout")), grid=grid,
                     train_rows=tr, test_rows=te,
                     test_predicate=lambda v: np.zeros(len(v), bool))
    recs = build_row2_unseen(grid, split, cfg)["records"]
    vals = np.zeros((len(recs), grid.n_factors), np.float64)
    for i, rec in enumerate(recs):
        for t in rec["request"]:
            vals[i, int(t["factor_index"])] = float(t["value"])

    # PREREG-2 §3: the frozen-superset assertion, here too
    fr = json.load(open(W.data_root() / "value_holdout" / "row2_unseen.json",
                        encoding="utf-8"))["records"]
    assert all(json.dumps(recs[i], sort_keys=True) == json.dumps(fr[i], sort_keys=True)
               for i in range(len(fr))), "not a superset of the frozen row2_unseen.json"
    print(f"[vh] seed {seed}: {len(recs)} requests, superset asserted")

    k_hue = grid.index_of("object_hue")
    k_shape = grid.index_of("shape")
    out = {"values": vals}
    proof = {}
    for name in WANT:
        try:
            r = load_rival(seed, name, root=ROOT)
        except Exception as ex:
            print(f"[vh]   {name}: UNAVAILABLE -- {type(ex).__name__}: {str(ex)[:140]}")
            out[f"absent__{name}"] = np.array(str(ex)[:300])
            continue
        # -- PREREG-2 §3 assertions, off the OBJECT, not the label -------------
        pop = None
        for attr in ("population", "meta", "info"):
            pop = getattr(r, attr, None) or (r.meta.get(attr) if hasattr(r, "meta") else None)
            if isinstance(pop, dict) and "n_per_value" in pop:
                break
            pop = None
        if pop is None and hasattr(r, "meta"):
            pop = (r.meta or {}).get("population")
        assert pop is not None, f"{name}: no population record on the checkpoint"
        npv = pop["n_per_value"]
        hue_c = npv[grid.names[k_hue]] if isinstance(npv, dict) else npv[k_hue]
        shp_c = npv[grid.names[k_shape]] if isinstance(npv, dict) else npv[k_shape]
        assert int(hue_c[5]) == 0, f"{name}: object_hue[5] = {hue_c[5]}, expected 0"
        assert int(shp_c[0]) == 0, f"{name}: shape[0] = {shp_c[0]}, expected 0"
        nrf = int(pop.get("n_rows_fit", 0))
        assert nrf == 131_072, f"{name}: n_rows_fit = {nrf}, expected 131072"
        proof[name] = {"object_hue_5": 0, "shape_0": 0, "n_rows_fit": nrf}
        feats, oor = r(vals)
        out[f"feats__{name}"] = np.asarray(feats, np.float32)
        out[f"oor__{name}"] = np.asarray(oor, bool)
        print(f"[vh]   {name}: ASSERTED population; feats {feats.shape}, oor {int(np.sum(oor))}")
    out["population_proof"] = np.array(json.dumps(proof))
    np.savez_compressed(dest, **out)
    print(f"[vh] wrote {dest}")


if __name__ == "__main__":
    main()
