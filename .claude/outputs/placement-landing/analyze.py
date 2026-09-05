"""Tables for placement-landing from placement_mia.json (direction-calibrated, as in mia)."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "placement_mia.json")
r = json.load(open(path))
m = r["meta"]
print("meta:", {k: m[k] for k in ("commit", "jax", "seeds", "n_targets", "n_worlds",
                                  "R_mia", "R_sized", "n_cells_mia", "n_cells_sized")})

STATS = ["hole", "n_live", "s1", "s2", "s4", "s5"]
for arm in [a for a in ("relocate", "canon_sized", "canon_native") if a in r]:
    A = r[arm]
    print(f"\n=== {arm}  (placement={A['placement']}, n_cells={A['n_cells']}, "
          f"R={A['lattice_radius']}) ===")
    print(f"  IN-after-removal vs OUT-history byte-equal fraction: "
          f"mean {A['byte_equal_frac'][0]:.4f}  min {A['byte_equal_frac'][1]:.4f}")
    print(f"  moves/delete: mean {A['moves_per_delete'][0]:.3f} "
          f"max {A['moves_per_delete'][1]:.0f}   runtime {A['runtime_s']:.0f}s")
    print(f"  {'stat':8s} | {'history AUC (cal)':>18s} {'raw':>8s} {'TPR@1%':>8s} "
          f"| {'paired AUC':>11s} {'TPR@1%':>8s}")
    for st in STATS:
        row = f"  {st:8s} |"
        for col in ("history", "paired"):
            d = A["columns"].get(col, {}).get(st)
            if d is None:
                row += f" {'-':>18s} {'-':>8s} {'-':>8s} |" if col == "history" else \
                       f" {'-':>11s} {'-':>8s}"
                continue
            raw = np.asarray(d["auc_all"])
            cal = np.maximum(raw, 1 - raw)
            t = d["tpr@fpr0.01"][0]
            if col == "history":
                row += (f" {cal.mean():>10.5f}±{cal.std():.4f} {raw.mean():>8.4f} "
                        f"{t:>8.4f} |")
            else:
                row += f" {cal.mean():>6.4f}±{cal.std():.4f} {t:>8.4f}"
        print(row)
    if "retention_post" in A["columns"]:
        d = A["columns"]["retention_post"]["-"]
        print(f"  retention after removal: {d['auc'][0]:.4f}")
