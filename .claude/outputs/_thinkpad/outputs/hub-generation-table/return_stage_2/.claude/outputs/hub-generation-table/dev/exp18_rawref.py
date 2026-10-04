"""DEV-18 — the raw request-model references, PER CORNER.

⛔ No store anywhere.  One question: is the interaction advantage DEV-12 measured on corner
0 (+0.21) real across corners, or was it corner-specific like DEV-14's +0.13 turned out to
be?  This is the cheapest decisive experiment left.
"""
import itertools
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ctx  # noqa: E402
import dev  # noqa: E402

T0 = time.time()
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
OUTP = os.path.join(os.environ["CHLU_OUT"].replace("/c/", "C:/", 1), "dev",
                    f"exp18_seed{SEED}.json")
CORNERS = [
    dict(hue_lo=0.2, scale_hi=True, shape_value=0.0, orient_hi=True),
    dict(hue_lo=0.4, scale_hi=True, shape_value=2.0, orient_hi=True),
    dict(hue_lo=0.6, scale_hi=False, shape_value=1.0, orient_hi=False),
    dict(hue_lo=0.2, scale_hi=False, shape_value=3.0, orient_hi=True),
    dict(hue_lo=0.4, scale_hi=False, shape_value=1.0, orient_hi=False),
    dict(hue_lo=0.6, scale_hi=True, shape_value=0.0, orient_hi=False),
]
c = ctx.build_context(SEED)
D = int(np.prod(c["feat_shape"]))
cards = c["cards"]
F = len(cards)
ALLP = list(itertools.combinations(range(F), 2))
res = json.load(open(OUTP, encoding="utf-8")) if os.path.exists(OUTP) else {}
res.setdefault("corners", {})


def design(lab, pairs):
    N = lab.shape[0]
    offs = np.cumsum([0] + cards[:-1])
    A = np.zeros((N, 1 + int(sum(cards))))
    A[:, 0] = 1.0
    A[np.repeat(np.arange(N), F), 1 + (lab + offs[None, :]).reshape(-1)] = 1.0
    out = [A]
    for (f, g) in pairs:
        B = np.zeros((N, cards[f] * cards[g]))
        B[np.arange(N), lab[:, f] * cards[g] + lab[:, g]] = 1.0
        out.append(B)
    return np.concatenate(out, axis=1)


for ci, corner in enumerate(CORNERS):
    k = f"c{ci}"
    if k in res["corners"]:
        ctx.log(f"corner {ci} already done")
        continue
    s = dev.dev_setup(c, corner)
    Zs = np.asarray(c["gather"](s["pool"]), np.float32).reshape((-1, D)).astype(np.float64)
    Ls = c["factor_labels"](c["grid"], s["pool"])
    Ld = s["lab_dev"]
    row = {"corner": corner}
    for nm, pairs in (("additive", []), ("pairwise_all", ALLP)):
        best = None
        Phi = design(Ls, pairs)
        Phid = design(Ld, pairs)
        for r in (1.0, 10.0, 100.0, 300.0):
            W = np.linalg.solve(Phi.T @ Phi + r * np.eye(Phi.shape[1]), Phi.T @ Zs)
            sc = dev.score_features(c, (Phid @ W).astype(np.float32), Ld)
            sc["ridge"] = r
            sc["n_columns"] = int(Phi.shape[1])
            sc["bytes_raw"] = int(Phi.shape[1] * D * 4)
            if best is None or sc["aggregate"] > best["aggregate"]:
                best = sc
        row[nm] = best
        ctx.log(f"  [c{ci} {nm}] agg={best['aggregate']:.4f}"
                f"({best['distinct_factor_tuples']}/72) ridge={best['ridge']:g} "
                f"cols={best['n_columns']} bytes={best['bytes_raw']}")
    row["delta"] = row["pairwise_all"]["aggregate"] - row["additive"]["aggregate"]
    res["corners"][k] = row
    with open(OUTP, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)

a = [v["additive"]["aggregate"] for v in res["corners"].values()]
p = [v["pairwise_all"]["aggregate"] for v in res["corners"].values()]
ta = [v["additive"]["distinct_factor_tuples"] for v in res["corners"].values()]
tp = [v["pairwise_all"]["distinct_factor_tuples"] for v in res["corners"].values()]
res["summary"] = {
    "n": len(a),
    "additive_mean": float(np.mean(a)), "additive_sd": float(np.std(a)),
    "pairwise_mean": float(np.mean(p)), "pairwise_sd": float(np.std(p)),
    "additive_tuples": float(np.mean(ta)), "pairwise_tuples": float(np.mean(tp)),
    "delta_mean": float(np.mean(p) - np.mean(a)),
    "paired_wins": int(sum(1 for x, y in zip(p, a) if x > y)),
}
with open(OUTP, "w", encoding="utf-8") as fh:
    json.dump(res, fh, indent=2)
ss = res["summary"]
ctx.log(f"[SUMMARY] additive={ss['additive_mean']:.4f}+-{ss['additive_sd']:.4f}"
        f"({ss['additive_tuples']:.1f})  "
        f"pairwise={ss['pairwise_mean']:.4f}+-{ss['pairwise_sd']:.4f}"
        f"({ss['pairwise_tuples']:.1f})  "
        f"delta={ss['delta_mean']:+.4f}  paired {ss['paired_wins']}/{ss['n']}")
ctx.log(f"done in {time.time() - T0:.0f}s -> {OUTP}")
