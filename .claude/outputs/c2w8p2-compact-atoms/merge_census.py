"""Merge the per-seed arm-A artifacts into ONE `capture_armA.json`.

The three census seeds were run as three processes (wall-clock; the cells are
independent and each carries its own seed), so the arm-level aggregation is
re-emitted here **using the shipped functions from `exp_capture_armA`** rather
than re-implemented, so the merged artifact is the same arithmetic a single
`--seeds 0,1,2` run would have written.
"""
import json
import sys

from chlu.config import get_default_config
from chlu.experiments import exp_capture_armA as arm
from chlu.experiments import exp_well_lifecycle as ewl

paths = sys.argv[1:-1]
out_path = sys.argv[-1]

cfg = get_default_config()
cfg.experiment_capture_arm_a.atom_width_frac_spacing = 1.5

cells, base = [], None
for p in paths:
    d = json.load(open(p))
    base = base or d
    cells += d["arms"]["armA_compact"]["cells"]
cells.sort(key=lambda c: c["seed"])

legs = [arm.gate_legs(c) for c in cells]
all_pass = [bool(lg["G_CAP"]["pass"] and lg["G_DEC"]["pass"] and lg["G_DRIFT"]["pass"])
            for lg in legs]
merged = dict(base)
merged["seeds"] = [int(c["seed"]) for c in cells]
merged["merged_from"] = paths
merged["arms"] = {
    "armA_compact": {
        "label": "armA_compact",
        "seeds": [int(c["seed"]) for c in cells],
        "arm_config": base["arms"]["armA_compact"]["arm_config"],
        "store_flags": [c["flags"]["clu_system_non_defaults"] for c in cells],
        "gate": {
            "legs_by_seed": legs,
            "G_CAP_pass_seeds": int(sum(lg["G_CAP"]["pass"] for lg in legs)),
            "G_DEC_pass_seeds": int(sum(lg["G_DEC"]["pass"] for lg in legs)),
            "G_DRIFT_pass_seeds": int(sum(lg["G_DRIFT"]["pass"] for lg in legs)),
            "all_three_same_seed": int(sum(all_pass)),
            "n_seeds": len(cells),
            "gate_pass": bool(len(cells) >= 3 and all(all_pass)),
            "rule": base["arms"]["armA_compact"]["gate"]["rule"],
        },
        "own_foreign_by_seed": [arm.own_foreign(c) for c in cells],
        "bytes_by_seed": [arm.byte_ledger(c, cfg) for c in cells],
        "depth_raw_median_by_seed": [c["census"]["depth_raw_median"] for c in cells],
        "geometry_by_seed": [c["geometry"] for c in cells],
        "self_probe_by_seed": [c["self_probe"] for c in cells],
        "cells": cells,
        "wall_s": float(sum(c["wall_s"] for c in cells)),
    }
}
with open(out_path, "w") as f:
    json.dump(ewl._jsonable(merged), f, indent=2)
g = merged["arms"]["armA_compact"]["gate"]
print("gate_pass", g["gate_pass"], "CAP", g["G_CAP_pass_seeds"], "DEC",
      g["G_DEC_pass_seeds"], "DRIFT", g["G_DRIFT_pass_seeds"], "of", g["n_seeds"])
print("wrote", out_path)
