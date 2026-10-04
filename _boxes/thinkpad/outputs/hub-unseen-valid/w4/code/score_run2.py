"""SCORED RUN 2/3 — spoke A's rivals, refitted on MY pool, on MY 70 corners.

⛔ Runs only from PREREG-w4-unseen-valid-2.md. The arm's per-corner values are REUSED
verbatim from run 1/3's artifacts -- the arm is not re-scored and cannot drift.
⭐ Every leg carries its distinct-output count beside its accuracy (§1).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import w4_common as C  # noqa: E402
import chlu.experiments.exp_gen_factor_wells as W  # noqa: E402
from dev_eval import make_anchors, second_judge_e, per_corner  # noqa: E402
from score_row2 import paired_estimand  # noqa: E402
from j_readout_choice import hue_logits  # noqa: E402

LEGS = ("cont_mlp_notune_572k", "cont_mlp_notune", "table", "cont_mlp")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prereg", required=True)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--margin", type=float, required=True)
    ap.add_argument("--batch", type=int, default=128)
    a = ap.parse_args()
    assert Path(a.prereg).exists(), "PREREG missing -- refusing to score"
    W.PINS["condition"] = "test"

    agg = {"prereg": a.prereg, "margin": a.margin, "per_seed": {}, "NOT_RUN": {}}
    for seed in [int(s) for s in a.seeds.split(",")]:
        out = C.OUT / "runs" / f"score_run2_s{seed}.json"
        if out.exists():
            agg["per_seed"][str(seed)] = json.load(open(out, encoding="utf-8"))
            print(f"[run2] seed {seed}: reusing artifact")
            continue
        W.PINS["seed"] = seed
        C.use_cache_generation2((seed,))
        t0 = time.time()
        row2, grid, train_rows = C.build_row2(a.n, seed=0)
        recs = row2["records"][:a.n]
        rows, hues, vals = C.bracket_rows(recs, grid, train_rows)
        cids = C.corner_ids(recs, grid, vals, hues)
        assert len(set(cids)) >= 49, f"only {len(set(cids))} corners"

        z = np.load(C.OUT / "runs" / f"vhpool_rivals_s{seed}_n{a.n}.npz", allow_pickle=True)
        assert np.allclose(z["values"], vals), "exported requests != mine"
        run1 = json.load(open(C.OUT / "runs" / f"score_run1_s{seed}.json", encoding="utf-8"))

        J = C.SecondJudge(seed)
        A = make_anchors(J, rows, hues, a.batch)
        i_lo = np.round(hues[:, 0] / 0.1).astype(int)
        i_hi = np.round(hues[:, 1] / 0.1).astype(int)
        nn = np.arange(len(recs))
        cards = tuple(int(c) for c in grid.cardinalities)

        res = {"seed": seed, "n_requests": len(recs), "n_corners": len(set(cids)),
               "population_proof": json.loads(str(z["population_proof"])),
               "reads": {}, "per_corner": {}, "not_run": {}}
        for leg in LEGS:
            if f"feats__{leg}" not in z:
                reason = str(z[f"absent__{leg}"]) if f"absent__{leg}" in z else "absent"
                res["not_run"][leg] = reason
                print(f"[run2] seed {seed} {leg}: NOT RUN -- {reason[:90]}")
                continue
            X = np.asarray(z[f"feats__{leg}"], np.float32)
            e, p = second_judge_e(J, X, A, a.batch)
            tup = J.judge.read_back(X, 256)
            lin = np.ravel_multi_index(tuple(tup.T), cards)
            res["per_corner"][leg] = per_corner(cids, e)
            res["reads"][leg] = {
                "mean_e": float(e.mean()), "mean_p": float(p.mean()),
                "distinct_tuples": int(len(np.unique(lin))),
                "distinct_feature_rows": int(len(np.unique(np.round(X.astype(np.float64), 4), axis=0))),
            }
            print(f"[run2] seed {seed} {leg:28s} e={e.mean():.4f}  "
                  f"distinct_tuples={len(np.unique(lin))}")
        res["arm_per_corner_from_run1"] = run1["per_corner"]["linear_span_ARM"]
        res["seconds"] = time.time() - t0
        C.jdump(res, out)
        agg["per_seed"][str(seed)] = res

    # ---- the paired estimands -------------------------------------------
    agg["estimands"] = {}
    seeds = sorted(agg["per_seed"])
    for leg in LEGS:
        per = []
        for s in seeds:
            r = agg["per_seed"][s]
            if leg not in r["per_corner"]:
                continue
            # ⭐ SAME orientation as run 1/3: positive delta = MY ARM better.
            per.append(paired_estimand(r["arm_per_corner_from_run1"],
                                       r["per_corner"][leg],
                                       a.margin, "linear_span_ARM(run1)", leg))
        if not per:
            agg["NOT_RUN"][leg] = "no seed produced this leg"
            continue
        agg["estimands"][leg] = {
            "per_seed": per, "n_seeds": len(per),
            "mean_delta_across_seeds": float(np.mean([x["mean_paired_delta"] for x in per])),
            "sd_delta_across_seeds": float(np.std([x["mean_paired_delta"] for x in per])),
        }
    C.jdump(agg, C.OUT / "runs" / "score_run2_AGG.json")
    print()
    for leg, v in agg["estimands"].items():
        print(f"{leg:28s} ARM_minus_leg {v['mean_delta_across_seeds']:+.4f} "
              f"sd {v['sd_delta_across_seeds']:.4f}  seeds {v['n_seeds']}  "
              f"(NEGATIVE = the leg BEATS my arm)")
    for leg, why in agg["NOT_RUN"].items():
        print(f"{leg:28s} NOT RUN -- {why}")


if __name__ == "__main__":
    main()
