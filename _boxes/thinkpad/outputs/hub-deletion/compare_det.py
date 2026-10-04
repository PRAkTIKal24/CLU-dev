#!/usr/bin/env python
"""STEP 1 verdict: are two independent full-scale builds byte-identical?

Compares build.json digests AND re-hashes the saved arrays in modes.npz, so the
verdict does not rest on the digest function agreeing with itself.
PASS is all-of; there is no partial credit and no tolerance.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

out = Path(sys.argv[1])
tags = sys.argv[2:] or ["det_A", "det_B"]
mans = {}
for t in tags:
    with open(out / t / "build.json") as fh:
        mans[t] = json.load(fh)

KEYS = [("digest", "store_digest"), ("digest", "param_hash"), ("digest", "book_hash"),
        ("digest", "labels_sha256"), ("digest", "well_sizes"),
        ("live_sha256",), ("deleted_sha256",), ("n_live",)]

a, b = tags[0], tags[1]
rows, ok = [], True
for k in KEYS:
    va, vb = mans[a], mans[b]
    for part in k:
        va, vb = (va or {}).get(part), (vb or {}).get(part)
    same = va == vb
    ok &= same
    rows.append({"field": ".".join(k), a: va, b: vb, "equal": bool(same)})

# --- independent re-hash of the stored arrays (does not use store_digest) ----
arr_ok = True
arr_rows = []
za, zb = np.load(out / a / "modes.npz"), np.load(out / b / "modes.npz")
for f in sorted(set(za.files) & set(zb.files)):
    ha = hashlib.sha256(np.ascontiguousarray(za[f]).tobytes()).hexdigest()
    hb = hashlib.sha256(np.ascontiguousarray(zb[f]).tobytes()).hexdigest()
    same = ha == hb
    arr_ok &= same
    d = {"array": f, "shape": list(za[f].shape), "dtype": str(za[f].dtype),
         "sha_a": ha[:16], "sha_b": hb[:16], "equal": bool(same)}
    if not same and np.issubdtype(za[f].dtype, np.floating):
        d["max_abs_diff"] = float(np.max(np.abs(za[f].astype(np.float64) - zb[f].astype(np.float64))))
        d["n_differing"] = int((za[f] != zb[f]).sum())
    arr_rows.append(d)

verdict = "PASS" if (ok and arr_ok) else "FAIL"
res = {"schema": "chlu-hub-deletion/determinism/1", "tags": tags, "verdict": verdict,
       "manifest_fields_equal": bool(ok), "all_arrays_bitwise_equal": bool(arr_ok),
       "fields": rows, "arrays": arr_rows,
       "build_inputs": mans[a]["build_inputs"],
       "timing_s": {t: mans[t]["timing_s"] for t in tags},
       "unit_wall_s": {t: mans[t]["unit_wall_s"] for t in tags},
       "loss_first": {t: mans[t]["loss_first"] for t in tags},
       "loss_final": {t: mans[t]["loss_final"] for t in tags},
       "threads": {t: mans[t]["machine"]["threads"] for t in tags},
       "utc": {t: mans[t]["utc"] for t in tags}}
print(json.dumps(res, indent=2, sort_keys=True, default=float))
print(f"\n=== DETERMINISM AT REAL SCALE: {verdict} ===", file=sys.stderr)
for r in rows:
    if not r["equal"]:
        print(f"  DIFFERS  {r['field']}: {r[a]}  !=  {r[b]}", file=sys.stderr)
for r in arr_rows:
    if not r["equal"]:
        print(f"  DIFFERS  array {r['array']} {r['shape']} "
              f"n_diff={r.get('n_differing')} max_abs={r.get('max_abs_diff')}", file=sys.stderr)
sys.exit(0 if verdict == "PASS" else 1)
