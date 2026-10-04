"""hub/refusal -- the join: units on disk -> the reported numbers.

Scored by the SHIPPED protocol, re-implemented nowhere that matters:
``chlu.eval.gen_probe.binary_auroc`` for every AUROC,
``chlu.eval.rivals.gen_knn.threshold_from_valid`` / ``refusal_rates`` for the threshold
and the refuse predicate.  The A7 control map is read from ``query_sets.json`` on disk.

The rivals are re-scored from ``gen-val-split-fix/refusal_scores.json``'s per-request
table -- a JOIN, never a re-run.  ``gen-arms-a7-wiring/refusal_scores.json`` is the
SUPERSEDED broken-split file and is never opened; the path is asserted below.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import refusal_lib as R  # noqa: E402
from chlu.core.gen_request import control_map_of, load_query_set_registry  # noqa: E402
from chlu.eval.gen_probe import binary_auroc  # noqa: E402
from chlu.eval.rivals import gen_knn as knn  # noqa: E402

RUNS = R.OUT / "runs"
#: The GOOD file.  Checked by DIRECTORY, not by basename -- the two share one.
RIVALS = R.CLAUDE_ROOT / "outputs" / "gen-val-split-fix" / "refusal_scores.json"
SUPERSEDED = R.CLAUDE_ROOT / "outputs" / "gen-arms-a7-wiring" / "refusal_scores.json"
#: The shipped quantile.  ``GenBaselinesConfig.refusal_valid_quantile``.
Q = 0.95
SEEDS = (0, 1, 2)


def _assert_good_rivals_file() -> dict:
    if RIVALS.parent.name != "gen-val-split-fix":
        raise AssertionError(f"refusing: {RIVALS} is not in gen-val-split-fix/")
    if RIVALS.resolve() == SUPERSEDED.resolve():
        raise AssertionError("refusing: that is the SUPERSEDED broken-split file")
    with open(RIVALS, encoding="utf-8") as fh:
        d = json.load(fh)
    if str(d.get("status")) != "complete":
        raise AssertionError(f"rival table status is {d.get('status')!r}, not complete")
    return d


def au(neg, pos) -> float:
    neg, pos = np.asarray(neg, np.float64), np.asarray(pos, np.float64)
    s = np.concatenate([neg, pos])
    y = np.concatenate([np.zeros(neg.size, bool), np.ones(pos.size, bool)])
    return float(binary_auroc(s, y))


def ms(vals) -> dict:
    if any(isinstance(v, str) for v in vals):          # a declared label, not a number
        u = sorted({str(v) for v in vals})
        return {"label": u[0] if len(u) == 1 else u}
    a = np.asarray([float(v) for v in vals], np.float64)
    return {"mean": float(np.nanmean(a)), "std": float(np.nanstd(a)),
            "per_seed": [float(x) for x in a], "n_seeds": int(a.size)}


# ---------------------------------------------------------------------------
# The rivals -- a JOIN on the persisted per-request table
# ---------------------------------------------------------------------------


def rival_rows() -> dict:
    d = _assert_good_rivals_file()
    out = {}
    for leg in d["scores"]:
        per_variant = {}
        for seed_key in sorted(d["scores"][leg]):
            sc = d["scores"][leg][seed_key]["scores"]
            for fam in sc:
                for var, v in sc[fam].items():
                    kinds = np.asarray(v["_kinds_C"])
                    C = np.asarray(v["C"]["scores"], np.float64)
                    B = np.asarray(v["B"]["scores"], np.float64)
                    A = np.asarray(v["A"]["scores"], np.float64)
                    A7 = np.asarray(v["A7"]["scores"], np.float64)
                    A7D = np.asarray(v["A7D"]["scores"], np.float64)
                    MS = np.asarray(v["MS"]["scores"], np.float64)
                    oor, dup = kinds == "out_of_range", kinds == "duplicate_factor"
                    # the rivals' OWN published threshold rule: A at the shipped q
                    thr = knn.threshold_from_valid(A, Q)
                    rr = lambda x: knn.refusal_rates(x, thr)["refused_fraction"]  # noqa: E731
                    per_variant.setdefault(f"{fam}/{var}", {})[seed_key] = {
                        "auroc_pooled_C_vs_B": au(B, C),
                        "auroc_a7ctrl_out_of_range": au(B, C[oor]),
                        "auroc_a7ctrl_duplicate_factor": au(A7, C[dup]),
                        "auroc_uncontrolled_duplicate_factor": au(B, C[dup]),
                        "threshold": float(thr), "threshold_set": "A", "q": Q,
                        "reject_out_of_range": float(rr(C[oor])),
                        "reject_duplicate_factor": float(rr(C[dup])),
                        "reject_C_pooled": float(rr(C)),
                        "accept_B": float(1.0 - rr(B)),
                        "accept_A7": float(1.0 - rr(A7)),
                        "accept_A7D": float(1.0 - rr(A7D)),
                        "accept_MS": float(1.0 - rr(MS)),
                        "accept_A": float(1.0 - rr(A)),
                    }
        out[leg] = {k: v for k, v in per_variant.items() if len(v) == 3}
    return {leg: {var: {m: ms([per[s][m] for s in sorted(per)])
                        for m in next(iter(per.values()))}
                  for var, per in legv.items()}
            for leg, legv in out.items()}


# ---------------------------------------------------------------------------
# Ours
# ---------------------------------------------------------------------------

STATS = {
    "S_max": "S_max",            # PRIMARY
    "S_sum": "S_sum",            # C1 -- declared a term counter
    "S_mean": "S_mean",          # C2
    "e_total": "e_total",        # C3
    "e_store": "e_store",        # C4
    "neg_rel_tail_drop": None,   # C5 -- built below (higher = more refusable)
    "launch_S_max": "launch_S_max",   # C6
}


def load_unit(seed: int, budget: str, set_name: str, arm: str):
    p = RUNS / f"seed_{int(seed)}" / f"{budget}_{set_name}_{arm}.npz"
    if not p.exists():
        return None
    return np.load(p, allow_pickle=False)


def stat_of(z, name: str) -> np.ndarray:
    if name == "neg_rel_tail_drop":
        # rel is negative while descending; a chain still descending has NOT settled,
        # so -rel is "how far from settled", i.e. higher = more refusable.
        return -np.asarray(z["rel_tail_drop"], np.float64)
    return np.asarray(z[STATS[name]], np.float64)


def our_rows(budget: str, arm: str, seeds=SEEDS) -> dict:
    reg = load_query_set_registry(R.DATA_ROOT / R.DATASET / "query_sets.json")
    cmap = control_map_of(reg)
    per_stat: dict = {}
    per_seed_thr: dict = {}
    for stat in STATS:
        rows = {}
        for seed in seeds:
            z = {s: load_unit(seed, budget, s, arm)
                 for s in ("val", "A", "B", "MS", "C", "A7", "A7D")}
            if any(v is None for v in z.values()):
                return {"incomplete": True,
                        "missing": [k for k, v in z.items() if v is None],
                        "seed": int(seed), "arm": arm, "budget": budget}
            kinds = np.asarray([str(k) for k in z["C"]["kinds"]])
            S = {k: stat_of(v, stat) for k, v in z.items()}
            thr = knn.threshold_from_valid(S["val"], Q)
            rr = lambda x: knn.refusal_rates(x, thr)["refused_fraction"]  # noqa: E731
            oor, dup = kinds == "out_of_range", kinds == "duplicate_factor"
            neg_for = {"out_of_range": S["B"], "duplicate_factor": S["A7"]}
            rows[seed] = {
                "auroc_pooled_C_vs_B": au(S["B"], S["C"]),
                "auroc_a7ctrl_out_of_range": au(neg_for["out_of_range"], S["C"][oor]),
                "auroc_a7ctrl_duplicate_factor": au(neg_for["duplicate_factor"], S["C"][dup]),
                "auroc_uncontrolled_duplicate_factor": au(S["B"], S["C"][dup]),
                "threshold": float(thr), "threshold_set": "val", "q": Q,
                "reject_out_of_range": float(rr(S["C"][oor])),
                "reject_duplicate_factor": float(rr(S["C"][dup])),
                "reject_C_pooled": float(rr(S["C"])),
                "accept_B": float(1.0 - rr(S["B"])),
                "accept_A7": float(1.0 - rr(S["A7"])),
                "accept_A7D": float(1.0 - rr(S["A7D"])),
                "accept_MS": float(1.0 - rr(S["MS"])),
                "accept_A": float(1.0 - rr(S["A"])),
                "accept_val_BUILD_CHECK": float(1.0 - rr(S["val"])),
                "n_C_oor": int(oor.sum()), "n_C_dup": int(dup.sum()),
            }
            per_seed_thr.setdefault(stat, {})[seed] = float(thr)
        per_stat[stat] = {m: ms([rows[s][m] for s in seeds])
                          for m in rows[seeds[0]]}
    return {"incomplete": False, "budget": budget, "arm": arm,
            "control_map": cmap, "q": Q, "threshold_set": "val",
            "stats": per_stat, "thresholds": per_seed_thr}


def beyond_seed_spread(mine: dict, rival: dict) -> bool:
    """mean - 1 std > rival mean + 1 std."""
    return bool(mine["mean"] - mine["std"] > rival["mean"] + rival["std"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", default=R.HEADLINE_BUDGET)
    ap.add_argument("--arms", nargs="+",
                    default=["full", "request_off", "zero_step", "store_removed"])
    ap.add_argument("--out", default=str(R.OUT / "RESULTS.json"))
    a = ap.parse_args()

    res = {"schema": "hub-refusal/results/1",
           "rivals_file": str(RIVALS), "superseded_never_read": str(SUPERSEDED),
           "q": Q, "threshold_set_ours": "val", "threshold_set_rivals": "A",
           "budget": a.budget,
           "rivals": rival_rows(),
           "ours": {arm: our_rows(a.budget, arm) for arm in a.arms}}
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(f"wrote {a.out}")

    full = res["ours"].get("full", {})
    if full.get("incomplete"):
        print(f"full arm INCOMPLETE: {full}")
        return 0
    riv = res["rivals"]
    hist = riv.get("historical_688128", {})
    dep = riv.get("deployed_524288", {})
    print("\n=== RIVALS (3-seed mean +/- std) ===")
    hdr = ("row", "pooled C-vs-B", "A7ctrl oor", "A7ctrl dup", "accept B", "rej oor", "rej dup")
    print("  %-34s %-18s %-18s %-18s %-9s %-9s %-9s" % hdr)
    for leg, legv in (("historical_688128", hist), ("deployed_524288", dep)):
        for var, v in legv.items():
            print("  %-34s %.4f+-%.4f  %.4f+-%.4f  %.4f+-%.4f  %.4f    %.4f    %.4f" % (
                f"{leg[:4]}/{var}",
                v["auroc_pooled_C_vs_B"]["mean"], v["auroc_pooled_C_vs_B"]["std"],
                v["auroc_a7ctrl_out_of_range"]["mean"], v["auroc_a7ctrl_out_of_range"]["std"],
                v["auroc_a7ctrl_duplicate_factor"]["mean"], v["auroc_a7ctrl_duplicate_factor"]["std"],
                v["accept_B"]["mean"], v["reject_out_of_range"]["mean"],
                v["reject_duplicate_factor"]["mean"]))
    for arm in a.arms:
        o = res["ours"].get(arm, {})
        if o.get("incomplete"):
            print(f"\n=== OURS arm={arm}: INCOMPLETE {o.get('missing')} seed {o.get('seed')}")
            continue
        print(f"\n=== OURS  budget={a.budget} arm={arm}  (threshold: val @ q={Q}) ===")
        print("  %-20s %-18s %-18s %-18s %-9s %-9s %-9s %-9s" % (
            "statistic", "pooled C-vs-B", "A7ctrl oor", "A7ctrl dup",
            "accept B", "rej oor", "rej dup", "val-chk"))
        for stat, v in o["stats"].items():
            star = " <-- PRIMARY" if stat == "S_max" else ""
            print("  %-20s %.4f+-%.4f  %.4f+-%.4f  %.4f+-%.4f  %.4f    %.4f    %.4f    %.4f%s" % (
                stat,
                v["auroc_pooled_C_vs_B"]["mean"], v["auroc_pooled_C_vs_B"]["std"],
                v["auroc_a7ctrl_out_of_range"]["mean"], v["auroc_a7ctrl_out_of_range"]["std"],
                v["auroc_a7ctrl_duplicate_factor"]["mean"], v["auroc_a7ctrl_duplicate_factor"]["std"],
                v["accept_B"]["mean"], v["reject_out_of_range"]["mean"],
                v["reject_duplicate_factor"]["mean"],
                v["accept_val_BUILD_CHECK"]["mean"], star))

    # --- the verdict, against the STRICTEST rival on each kind (PREREG section 1) ---
    prim = full["stats"]["S_max"]
    def best(metric):
        cands = [(f"{leg}/{var}", v[metric])
                 for leg, legv in riv.items() for var, v in legv.items()]
        return max(cands, key=lambda kv: kv[1]["mean"])
    print("\n=== VERDICT (PRIMARY = S_max, full arm; strictest rival per kind) ===")
    for metric in ("auroc_a7ctrl_out_of_range", "auroc_a7ctrl_duplicate_factor",
                   "auroc_pooled_C_vs_B"):
        name, rv = best(metric)
        mine = prim[metric]
        print("  %-36s mine %.4f+-%.4f   best rival %-28s %.4f+-%.4f   beats=%s  "
              "beyond-seed-spread=%s" % (
                  metric, mine["mean"], mine["std"], name, rv["mean"], rv["std"],
                  mine["mean"] > rv["mean"], beyond_seed_spread(mine, rv)))
    print("  accept-unseen-valid (B): mine %.4f+-%.4f" % (
        prim["accept_B"]["mean"], prim["accept_B"]["std"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
