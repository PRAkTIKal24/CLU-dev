"""placement-landing (w26) — the rematch cell: canonical placement under the REAL read.

`controller-mvp`'s `on_sized` K=64 point, re-run with `placement="canonical"`, scored by
the shipped two-phase Verlet read (`evaluate_items` via `controller_line`). This closes the
theorist's one open scope gap (H5 used gradient-flow relaxation).

Also runs the incumbent `on_sized` (refuse-and-relocate) arm on the same seeds, so the
per-offered comparison is measured in one process, not quoted across reports.
"""
import argparse
import json
import os
import time

import numpy as np

from chlu.config import get_default_config
from chlu.experiments.exp_controller_mvp import controller_line

HERE = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", default=os.path.join(HERE, "rematch_cell.json"))
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]

    cfg = get_default_config().experiment_controller_mvp
    cells = []
    for arm, mult in (("canon_sized", 1.0), ("canon_sized", 1.05), ("on_sized", 1.0)):
        cfg.canonical_radius_mult = mult
        for s in seeds:
            t0 = time.time()
            r = controller_line(cfg, s, a.K, arm)
            r["mult"] = mult
            r["runtime_s"] = time.time() - t0
            cells.append(r)
            print(f"{arm} mult={mult} seed={s}: admitted={r['n_admitted']} "
                  f"per_adm={r['per_admitted']:.4f} per_off={r['per_offered']:.4f} "
                  f"spacing={r['min_spacing_live']:.6f} cells={r['n_cells']} "
                  f"({r['runtime_s']:.0f}s)", flush=True)

    summary = {}
    for arm, mult in (("canon_sized", 1.0), ("canon_sized", 1.05), ("on_sized", 1.0)):
        rs = [c for c in cells if c["arm"] == arm and c["mult"] == mult]
        summary[f"{arm}@{mult}"] = {
            "n_admitted": [float(np.mean([r["n_admitted"] for r in rs])),
                           float(np.std([r["n_admitted"] for r in rs]))],
            "per_admitted": [float(np.mean([r["per_admitted"] for r in rs])),
                             float(np.std([r["per_admitted"] for r in rs]))],
            "per_offered": [float(np.mean([r["per_offered"] for r in rs])),
                            float(np.std([r["per_offered"] for r in rs]))],
            "min_spacing_live": [float(np.mean([r["min_spacing_live"] for r in rs])),
                                 float(np.std([r["min_spacing_live"] for r in rs]))],
            "n_cells": rs[0]["n_cells"], "radius": rs[0]["radius"],
            "admitted_frac": float(np.mean([r["n_admitted"] for r in rs])) / a.K,
        }
    with open(a.out, "w") as fh:
        json.dump({"K": a.K, "seeds": seeds, "cells": cells, "summary": summary}, fh,
                  indent=2, default=float)
    print(json.dumps(summary, indent=2))
