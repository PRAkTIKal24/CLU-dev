"""EXP-30 — ⭐ LEVER ISOLATION for hub/generation-table-2.

Every row is the SAME arm with ONE thing changed, at a byte ledger recomputed from the
live objects and held under ``knn_proto``'s 475,136 B.

    base        the predecessor's arm, reproduced exactly (58-col design, one ridge,
                coefficient-metric truncation, k-means dictionary, uniform allocation)
    +reduced    52-col additive design (an exactly equivalent re-parameterisation)
                and the freed bytes handed to the interaction rank
    +metric     rank truncation in the FEATURE-space metric
    +ridge      a separate, stronger ridge on the interaction block
    +nmf        the dictionary fitted for the LINEAR read verb (non-negative atoms)
    +alloc      atoms allocated across wells by spectrum water-filling
    all         every lever together

Controls on every corner, at identical bytes and identical head configuration:
    atoms_medoid  §C.1 sharp      atoms_kpp  §C.1 adversarial    mean  §C.2
    pca_affine    ⛔ R4 control only        random  the D-matched null
    knn_proto     the ceiling arm           best_constant  the free floor

⛔ DEV corners only.  ``dev.dev_setup`` REFUSES (never trims) a corner touching eval_a/b.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arm2  # noqa: E402
import ctx  # noqa: E402
import dev  # noqa: E402
import fast  # noqa: E402

T0 = time.time()
SEED = 0
OUT = os.environ["CHLU_OUT2"].replace("/c/", "C:/", 1)
TAG = sys.argv[1] if len(sys.argv) > 1 else "sel"
LO = int(sys.argv[2]) if len(sys.argv) > 2 else 0
HI = int(sys.argv[3]) if len(sys.argv) > 3 else 12
OUTP = os.path.join(OUT, "dev", f"exp30_{TAG}.json")

CORNERS = [
    dict(hue_lo=0.2, scale_hi=True, shape_value=0.0, orient_hi=True),
    dict(hue_lo=0.4, scale_hi=True, shape_value=2.0, orient_hi=True),
    dict(hue_lo=0.6, scale_hi=False, shape_value=1.0, orient_hi=False),
    dict(hue_lo=0.2, scale_hi=False, shape_value=3.0, orient_hi=True),
    dict(hue_lo=0.4, scale_hi=False, shape_value=1.0, orient_hi=False),
    dict(hue_lo=0.6, scale_hi=True, shape_value=0.0, orient_hi=False),
    dict(hue_lo=0.2, scale_hi=True, shape_value=1.0, orient_hi=True),
    dict(hue_lo=0.4, scale_hi=True, shape_value=3.0, orient_hi=False),
    dict(hue_lo=0.6, scale_hi=False, shape_value=2.0, orient_hi=True),
    dict(hue_lo=0.2, scale_hi=False, shape_value=0.0, orient_hi=False),
    dict(hue_lo=0.4, scale_hi=False, shape_value=2.0, orient_hi=True),
    dict(hue_lo=0.6, scale_hi=True, shape_value=3.0, orient_hi=True),
] + [dict(hue_lo=h, scale_hi=sc, shape_value=sh, orient_hi=o)
     for h in (0.0, 0.2, 0.4, 0.6) for sh in (0.0, 1.0, 2.0, 3.0)
     for sc in (True, False) for o in (True, False)]
_seen, _C = set(), []
for _c in CORNERS:
    _k = tuple(sorted(_c.items()))
    if _k not in _seen:
        _seen.add(_k)
        _C.append(_c)
CORNERS = _C

c = ctx.build_context(SEED)
D = int(np.prod(c["feat_shape"]))
cards = c["cards"]

from chlu.core.gen_dream import requests_from_labels  # noqa: E402
from chlu.core.gen_factored_read import default_pairs  # noqa: E402
from chlu.core.gen_request import FactorPrototypes, requests_from_rows  # noqa: E402

PAIR_NAMES = ["floor_huexscale", "object_huexscale"]     # ⛔ INHERITED, not re-selected
PAIRS = default_pairs(c["names"], PAIR_NAMES)
N_PAIR = sum(cards[f] * cards[g] for f, g in PAIRS)
K = 32
D_WELL = D // K
N_ADD_FULL, N_ADD_RED = 1 + sum(cards), 1 + sum(x - 1 for x in cards)

A_BASE = K * 26                                            # 832, the inherited span
R_BASE = arm2.budget_rank(K, A_BASE, N_ADD_FULL, N_PAIR, D_WELL)
R_RED = arm2.budget_rank(K, A_BASE, N_ADD_RED, N_PAIR, D_WELL)

_full = dict(K=K, A=A_BASE, pairs=PAIRS, n_pair=N_PAIR, alloc="uniform", seed=SEED)
_base = dict(_full, method="kmeans", rank=R_BASE, ridge_add=10.0, ridge_pair=10.0,
             reduced=False, metric_correct=False, n_add=N_ADD_FULL)

VARIANTS = {
    "base":     dict(_base),
    "reduced":  dict(_base, reduced=True, n_add=N_ADD_RED, rank=R_RED),
    "metric":   dict(_base, metric_correct=True),
    "ridge":    dict(_base, ridge_pair=100.0),
    "nmf":      dict(_base, method="nmf"),
    "alloc":    dict(_base, alloc="waterfill"),
    "all":      dict(_full, method="nmf", reduced=True, n_add=N_ADD_RED, rank=R_RED,
                     ridge_add=10.0, ridge_pair=100.0, metric_correct=True,
                     alloc="waterfill"),
    "all_km":   dict(_full, method="kmeans", reduced=True, n_add=N_ADD_RED, rank=R_RED,
                     ridge_add=10.0, ridge_pair=100.0, metric_correct=True,
                     alloc="waterfill"),
}
#: controls, run at the WINNING head configuration (filled per corner below)
CONTROLS = ("atoms_medoid", "atoms_kpp", "atoms_random", "pca_affine", "random", "mean")

ctx.log(f"K={K} d_well={D_WELL} A={A_BASE} n_add full/reduced={N_ADD_FULL}/{N_ADD_RED} "
        f"n_pair={N_PAIR} rank base/reduced={R_BASE}/{R_RED}")

res = json.load(open(OUTP, encoding="utf-8")) if os.path.exists(OUTP) else {}
res.setdefault("config", {"K": K, "A": A_BASE, "seed": SEED, "pairs": PAIR_NAMES,
                          "n_add_full": N_ADD_FULL, "n_add_reduced": N_ADD_RED,
                          "n_pair": N_PAIR, "rank_base": R_BASE, "rank_reduced": R_RED,
                          "knn_bytes": arm2.KNN_BYTES})
res.setdefault("corners", {})
res.setdefault("skipped", {})

for ci in range(LO, min(HI, len(CORNERS))):
    ck = f"c{ci}"
    if ck in res["corners"] or ck in res["skipped"]:
        continue
    t0 = time.time()
    try:
        s = dev.dev_setup(c, CORNERS[ci])
    except (dev.HeldOutOverlap, AssertionError) as exc:
        ctx.log(f"c{ci} SKIPPED — {exc}")
        res["skipped"][ck] = {"corner": CORNERS[ci], "reason": str(exc)}
        json.dump(res, open(OUTP, "w", encoding="utf-8"), indent=2)
        continue
    X_store = c["gather"](s["pool"])
    lab_store = c["factor_labels"](c["grid"], s["pool"])
    lab_dev = s["lab_dev"]
    Zs = np.asarray(X_store, np.float32).reshape((-1, D))
    _, dead, ow, _ = fast.channel_geometry(X_store, c["cfg"],
                                           f"s{SEED}_c{ci}_n{s['pool'].size}", log=None)
    labels = fast.balanced_labels(ow, dead, K, D)
    perm = np.stack([np.flatnonzero(labels == k) for k in range(K)])
    req = requests_from_labels(lab_dev, "full", 0, SEED)
    views = arm2.well_views(Zs, perm)
    row = {"corner": CORNERS[ci],
           "best_constant": float(np.mean([1.0 / np.unique(lab_dev[:, j]).size
                                           for j in range(len(cards))]))}

    for name, spec in VARIANTS.items():
        row[name] = arm2.run_variant(c, s, Zs, lab_store, perm, req, lab_dev,
                                     views=views, **spec)

    # controls at the "all" head configuration — ⭐ identical bytes, identical head
    ctrl_spec = {k: v for k, v in VARIANTS["all"].items() if k != "method"}
    for m in CONTROLS:
        row[m] = arm2.run_variant(c, s, Zs, lab_store, perm, req, lab_dev,
                                  views=views, method=m, **ctrl_spec)

    protos = FactorPrototypes.fit(c["grid"], X_store, s["pool"])
    row["knn_proto"] = dev.score_features(
        c, np.asarray(protos.embed_many(requests_from_rows(c["grid"], s["dev_rows"])),
                      np.float32), lab_dev)
    row["knn_proto"]["deployed_bytes"] = int(protos.byte_ledger()["bytes"])
    row["seconds"] = round(time.time() - t0, 1)
    res["corners"][ck] = row
    json.dump(res, open(OUTP, "w", encoding="utf-8"), indent=2)
    ctx.log(f"[c{ci}] " + " ".join(
        f"{n}={row[n]['aggregate']:.4f}({row[n]['distinct_factor_tuples']})"
        for n in list(VARIANTS) + ["atoms_medoid", "pca_affine", "mean"])
        + f" knn={row['knn_proto']['aggregate']:.4f}"
          f"({row['knn_proto']['distinct_factor_tuples']})"
          f" bytes={row['all']['total_bytes']} [{row['seconds']}s]")

rows = [res["corners"][k] for k in sorted(res["corners"], key=lambda x: int(x[1:]))]
if rows:
    def g(k):
        return np.array([r[k]["aggregate"] for r in rows])

    def tup(k):
        return np.array([r[k]["distinct_factor_tuples"] for r in rows])

    kn = g("knn_proto")
    n = len(rows)
    summ = {"n_corners": n,
            "best_constant": float(np.mean([r["best_constant"] for r in rows]))}
    for k in list(VARIANTS) + list(CONTROLS) + ["knn_proto"]:
        d = g(k) - kn
        se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        summ[k] = {"mean": float(g(k).mean()), "sd": float(g(k).std(ddof=1) if n > 1 else 0.0),
                   "tuples_mean": float(tup(k).mean()),
                   "delta_vs_knn": float(d.mean()), "se_vs_knn": se,
                   "ci95_vs_knn": [float(d.mean() - 1.96 * se), float(d.mean() + 1.96 * se)],
                   "wins_vs_knn": int((g(k) > kn).sum()),
                   "total_bytes": int(rows[0][k].get("total_bytes", 0))}
    res["summary"] = summ
    res["total_s"] = round(time.time() - T0, 1)
    json.dump(res, open(OUTP, "w", encoding="utf-8"), indent=2)
    ctx.log(f"[SUMMARY n={n}] " + " | ".join(
        f"{k}={summ[k]['mean']:.4f}({summ[k]['tuples_mean']:.1f})"
        f" D={summ[k]['delta_vs_knn']:+.4f} {summ[k]['wins_vs_knn']}/{n}"
        for k in list(VARIANTS) + ["atoms_medoid", "pca_affine", "mean"]))
    ctx.log(f"knn={summ['knn_proto']['mean']:.4f}({summ['knn_proto']['tuples_mean']:.1f}) "
            f"best_const={summ['best_constant']:.4f}")
ctx.log(f"done in {round(time.time() - T0, 1)}s -> {OUTP}")
