"""Merge per-(arm, seed) spine runs into ONE `CAPTURE-STRONG-PHI.json`.

Each cell is run in its own process (a 9-cell single process would lose every
completed cell to one crash), so the arm-level verdicts — the completed gate,
the branch, the D2a co-occurrence — are recomputed here **by the shipped
functions**, never re-typed:

    python merge.py <out.json> <run1.json> <run2.json> ...
"""
import json
import sys

from chlu.experiments import exp_capture_strong_phi as csp


def main(out_path, paths):
    runs = [json.load(open(p)) for p in paths]
    cells_by_arm = {}
    for r in runs:
        for a, arm in r["arms"].items():
            cells_by_arm.setdefault(a, [])
            for c in arm["cells"]:
                if all(int(x["seed"]) != int(c["seed"]) for x in cells_by_arm[a]):
                    cells_by_arm[a].append(c)
    base = runs[0]
    arms = {}
    for a, cells in cells_by_arm.items():
        cells = sorted(cells, key=lambda c: int(c["seed"]))
        gate = csp.completed_gate(cells)
        arms[a] = {
            "arm": a,
            "role": base["arms"][a]["role"] if a in base["arms"] else None,
            "seeds": [int(c["seed"]) for c in cells],
            "phi_dim_own": cells[0]["phi_dim_own"],
            "projection_form": cells[0]["projection_form"],
            "gate": gate,
            "daylight": csp.daylight_verdict(gate["g_addr_by_seed"],
                                             min_seeds=len(cells)),
            "d2a": {
                "by_seed": [c["d2a"] for c in cells],
                "cooccurrence": csp.d2a_cooccurrence(gate["legs_by_seed"],
                                                     gate["g_addr_by_seed"]),
            },
            "bytes_by_seed": [c["bytes_with_phi"] for c in cells],
            "launder_audit_by_seed": [c["launder_audit"] for c in cells],
            "geometry_by_seed": [c["geometry"] for c in cells],
            "self_probe_by_seed": [c["self_probe"] for c in cells],
            "depth_raw_median_by_seed": [c["census"]["depth_raw_median"] for c in cells],
            "store_inert_by_seed": [bool(float(c["census"]["depth_raw_median"]) < 1e-6)
                                    for c in cells],
            "stream_by_seed": [c["stream"] for c in cells],
            "phi_provenance_by_seed": [c["phi_provenance"] for c in cells],
            "stream_fingerprint_by_seed": [c["stream_fingerprint"] for c in cells],
            "store_flags_by_seed": [c["flags"]["clu_system_non_defaults"] for c in cells],
            "cells": cells,
            "wall_s": float(sum(float(c["wall_s"]) for c in cells)),
        }
    out = dict(base)
    out["arms"] = arms
    out["seeds"] = sorted({int(c["seed"]) for a in arms for c in arms[a]["cells"]})
    out["arm_roles"] = {a: arms[a]["role"] for a in arms}
    out["branch_by_arm"] = {a: arms[a]["daylight"]["branch"] for a in arms}
    out["gate_pass_by_arm"] = {a: arms[a]["gate"]["gate_pass"] for a in arms}
    out["store_inert_by_arm"] = {a: arms[a]["store_inert_by_seed"] for a in arms}
    out["wall_s"] = float(sum(arms[a]["wall_s"] for a in arms))
    out["merged_from"] = list(paths)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {out_path}: arms={list(arms)} "
          f"seeds={ {a: arms[a]['seeds'] for a in arms} }")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
