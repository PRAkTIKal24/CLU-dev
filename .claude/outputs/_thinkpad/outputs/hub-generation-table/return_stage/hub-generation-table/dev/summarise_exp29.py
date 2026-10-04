"""Summarise whatever DEV-29 has finished so far, straight from its ledger.

⭐ DEV-29 writes each corner the moment it completes, so this can be run at any point and
reports **exactly** the corners on disk — ⛔ never a projection of the ones still running.
"""
import json
import os
import sys

import numpy as np

p = os.path.join(os.environ["CHLU_OUT"].replace("/c/", "C:/", 1), "dev", "exp29_seed0.json")
d = json.load(open(p, encoding="utf-8"))
rows = [v for k, v in sorted(d["corners"].items(), key=lambda kv: int(kv[0][1:]))]
n = len(rows)
KEYS = ("clu_argmax", "linear_additive", "linear_pairs16", "store_removed_argmax",
        "store_removed_linear", "atoms_medoid_linear", "knn_proto")
g = lambda k: np.array([r[k]["aggregate"] for r in rows])
t = lambda k: np.array([r[k]["distinct_factor_tuples"] for r in rows])

print(f"DEV-29 — the SHIPPED store (K=5, M=64, ward, train_store), n = {n} corners, seed 0")
print(f"store_bytes = {rows[0]['store_bytes']}  shipped_head = {rows[0]['shipped_head_bytes']}  "
      f"linear_head = {rows[0]['linear_additive']['head_bytes']}  "
      f"byte-identical = {rows[0]['linear_additive']['byte_identical_to_shipped_clu']}")
print(f"linear+p16 total = {rows[0]['linear_pairs16']['total_bytes']} "
      f"({rows[0]['linear_pairs16']['total_bytes'] / (rows[0]['store_bytes'] + rows[0]['shipped_head_bytes']):.3f}x "
      "the shipped clu ledger)")
print()
print(f"{'row':<24}{'mean':>9}{'sd':>8}{'tuples':>9}")
for k in KEYS:
    print(f"{k:<24}{g(k).mean():>9.4f}{g(k).std(ddof=1) if n > 1 else 0:>8.4f}{t(k).mean():>9.1f}")
print(f"{'best_constant':<24}{0.5278:>9.4f}{0:>8.4f}{1:>9.1f}")
print()
print(f"{'paired difference':<44}{'mean':>9}{'SE':>8}{'95% CI':>22}{'wins':>8}{'t':>7}")
for a, b in (("linear_additive", "clu_argmax"), ("linear_pairs16", "clu_argmax"),
             ("linear_additive", "atoms_medoid_linear"),
             ("linear_additive", "store_removed_linear"),
             ("linear_additive", "knn_proto"), ("linear_pairs16", "knn_proto")):
    v = g(a) - g(b)
    se = v.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
    print(f"{a + ' - ' + b:<44}{v.mean():>+9.4f}{se:>8.4f}"
          f"  [{v.mean() - 1.96 * se:>+7.4f},{v.mean() + 1.96 * se:>+7.4f}]"
          f"{int((v > 0).sum()):>5}/{n}{v.mean() / se:>7.2f}")
