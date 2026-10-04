"""w6 hub/retention -- the BARS, the cross-arm comparison, and the cost decomposition.

Separate from ``exp_w6_retention`` on purpose: the scored seeds run in resumable chunks that
RE-IMPORT the experiment module, so this script only READS finished artifacts and never
changes the scored object.

It answers, per seed and across seeds:

* **BAR 1** the task file's falsifier -- decay's composition cost at the last horizon before
  death vs instant eviction's, paired on the 89 survivor corners, margin 1.0.
* **BAR 2** ``t_pay`` (where the cost is paid) against ``t_refuse`` (where the refusal
  arrives, DERIVED), plus the **free window**: the largest horizon whose |delta| is inside
  the margin.
* **the cost decomposition** -- how much of eviction's cost decay pays by WEIGHT (before the
  item dies) and how much by DEATH (the content zeroing at the refusal step).
* **the cross-arm comparison** -- the post-deletion field store against the table, retrain
  from scratch (GOLD), the 8-byte gate and the charter section C.1 floors, in ABSOLUTE
  novel-tuples-per-corner, which is the only way the "we still beat the rivals" half is
  readable beside the "we pay a cost" half.

Runnable::

    python scripts/w6_retention_bars.py --out C:/chlu/outputs/w6-retention/run1
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import numpy as np
from chlu.core import gen_retention as LAW
from chlu.experiments import exp_gen_delete_row7 as R7
from chlu.experiments import exp_w6_retention as W6
MARGIN = float(R7.MARGIN_NOVELTY)
def _rows_for(out: Path, seed: int):
    z = np.load(Path(out) / f"seed_{int(seed)}" / "scores.npz", allow_pickle=True)
    done = z["done"].item()
    cond = R7.load_condition()
    grid = cond["grid"]
    seen = R7.seen_tuple_mask(grid, cond["train_rows"])
    arms = sorted({a for (a, _) in done})
    rows = {a: {} for a in arms}
    for (a, cid), rec in done.items():
        if cid.startswith("probe/"):
            continue
        rows[a][cid] = R7._per_corner(rec, grid, seen)
    return rows, z, cond


def analyse(out: Path, seeds=(0, 1, 2), log=print) -> dict:
    out = Path(out)
    sched = W6.horizon_schedule()
    cond = R7.load_condition()
    cs = R7.scored_corners(cond)
    surv = [c["corner_id"] for c in cs["survivor"]]
    dele = [c["corner_id"] for c in cs["deleted"]]
    n_ref = int(sched["n_refuse_steps"])
    steps = sorted(int(r["n_steps"]) for r in sched["rows"])
    last_alive = max(s for s in steps if s < n_ref)

    res = {"margin": MARGIN, "schedule": {
        k: sched[k] for k in ("mu2", "rho", "n_half", "n_refuse_steps",
                              "n_refuse_half_lives", "band", "eps", "gamma")},
        "definitions": {
            "t_pay": "smallest horizon whose composition delta reaches 50 % of that seed's "
                     "instant-eviction delta (PREREG BAR 2)",
            "t_pay_90": "the same at 90 %, reported beside it",
            "free_window": "the largest horizon whose |composition delta| is still inside "
                           "the margin of 1.0 novel tuple/corner",
            "t_refuse": "the DERIVED refusal horizon, ceil(ln(1e-6)/ln(rho)) steps",
        },
        "per_seed": {}}

    for seed in seeds:
        p = out / f"seed_{int(seed)}" / "scores.npz"
        if not p.exists():
            res["per_seed"][str(seed)] = {"NOT_RUN": "no scores.npz -- a declared NOT-RUN "
                                                     "is never a null"}
            continue
        rows, z, _ = _rows_for(out, seed)
        S = {}

        def pair(a, b, ids=surv):
            return R7._paired(rows[a], rows[b], ids, "distinct_valid_novel_tuples")

        def refuse(a, ids):
            return float(np.mean([rows[a][i]["refuse_rate"] for i in ids]))

        def absolute(a, ids=surv):
            return float(np.mean([rows[a][i]["distinct_valid_novel_tuples"] for i in ids]))

        ev = pair("field/evict_red", "field/base")["mean_delta"]
        curve = []
        for r in sched["rows"]:
            n = int(r["n_steps"])
            a = W6.arm_name(n, r["t_half_lives"])
            if a not in rows:
                continue
            d = pair(a, "field/base")
            curve.append({"n_steps": n, "t": float(n) / sched["n_half"],
                          "lifetime": r["lifetime"], "delta": d["mean_delta"],
                          "ci95": d["ci95"],
                          "refuse_deleted": refuse(a, dele),
                          "refuse_survivor": refuse(a, surv),
                          "absolute_novel_per_corner": absolute(a),
                          "frac_of_eviction_cost": (d["mean_delta"] / ev) if ev else None})
        S["curve"] = curve

        # ---- BAR 1 ---------------------------------------------------------
        a_last = W6.arm_name(last_alive, 0.0)
        b1 = pair(a_last, "field/evict_red")
        S["BAR1_decay_vs_eviction_at_last_alive_step"] = {
            "step": last_alive, "t": last_alive / sched["n_half"],
            **b1, "margin": MARGIN,
            "PASS": bool(abs(b1["mean_delta"]) <= MARGIN),
            "means": ("PASS = decay's composition cost EQUALS instant eviction's at the last "
                      "horizon before the item dies => lifetimes buy nothing but latency, "
                      "which FALSIFIES this hub's dial declaration")}

        # ---- BAR 2 ---------------------------------------------------------
        def first_at(frac):
            for c in curve:
                if ev and c["delta"] / ev >= frac:
                    return c["t"]
            return None
        free = None
        for c in curve:
            if abs(c["delta"]) <= MARGIN:
                free = c["t"]
        t_ref_meas = next((c["t"] for c in curve if c["refuse_deleted"] >= 0.99), None)
        S["BAR2_pay_vs_refuse"] = {
            "t_pay_50pct": first_at(0.5), "t_pay_90pct": first_at(0.9),
            "t_pay_97pct": first_at(0.97),
            "free_window_half_lives": free,
            "t_refuse_measured": t_ref_meas,
            "t_refuse_derived": sched["n_refuse_half_lives"],
            "DOMINATED": (None if (first_at(0.5) is None or t_ref_meas is None)
                          else bool(first_at(0.5) < t_ref_meas)),
            "means": ("DOMINATED = the composition cost is paid strictly before any refusal "
                      "arrives, so instant eviction gives the same cost AND the refusal now")}

        # ---- the cost decomposition ---------------------------------------
        d_last = next(c["delta"] for c in curve if c["n_steps"] == last_alive)
        d_ref = next(c["delta"] for c in curve if c["n_steps"] == n_ref)
        S["cost_decomposition"] = {
            "paid_by_WEIGHT_before_death": d_last,
            "paid_at_DEATH_content_zeroing": d_ref - d_last,
            "total_at_refusal_step": d_ref,
            "instant_eviction": ev,
            "frac_paid_by_weight": (d_last / ev) if ev else None,
            "means": ("how much of eviction's cost decay has already paid while the item is "
                      "still LIVE (so still answering) versus how much arrives only when the "
                      "content is zeroed")}

        # ---- the cross-arm comparison, ABSOLUTE ----------------------------
        a_ref = W6.arm_name(n_ref, 0.0)
        S["vs_rivals_absolute_novel_per_corner"] = {
            "field/base (pre-deletion)": absolute("field/base"),
            "field/decay@refusal_step": absolute(a_ref),
            "field/evict_red": absolute("field/evict_red"),
            "field/hole": absolute("field/hole"),
            "table/base": absolute("table/base"),
            "table/row_removed": absolute("table/row_removed"),
            "table/retrain (GOLD)": absolute("table/retrain"),
            "field/lookup_argmax (C.1)": absolute("field/lookup_argmax"),
            "field/lookup_mean (C.1)": absolute("field/lookup_mean"),
            "field/landscape_removed (C.2)": absolute("field/landscape_removed"),
            "field/landscape_shuffled (C.2)": absolute("field/landscape_shuffled"),
        }
        S["vs_rivals_paired"] = {
            "decay@refusal vs table/retrain (GOLD)": pair(a_ref, "table/retrain"),
            "decay@refusal vs table/row_removed": pair(a_ref, "table/row_removed"),
            "decay@refusal vs lookup_argmax (C.1 FLOOR)": pair(a_ref, "field/lookup_argmax"),
            "decay@refusal vs lookup_mean (C.1 FLOOR)": pair(a_ref, "field/lookup_mean"),
            "decay@refusal vs landscape_removed (C.2)": pair(a_ref,
                                                             "field/landscape_removed"),
            "decay@refusal vs landscape_shuffled (C.2)": pair(a_ref,
                                                              "field/landscape_shuffled"),
        }
        S["refusal_on_deleted_set"] = {
            a: refuse(a, dele) for a in
            ("field/base", "field/evict_red", "field/hole", a_ref, "table/row_removed",
             "table/retrain", "table+gate8/clear_red", "table+joint/hole")
            if a in rows}
        S["generator_equalities"] = z["generator_equalities"].item() \
            if "generator_equalities" in z.files else None
        res["per_seed"][str(seed)] = S
        log(f"  seed {seed}: BAR1 {S['BAR1_decay_vs_eviction_at_last_alive_step']['PASS']} "
            f"(delta {b1['mean_delta']:+.4f}); BAR2 DOMINATED="
            f"{S['BAR2_pay_vs_refuse']['DOMINATED']} "
            f"(t_pay50 {S['BAR2_pay_vs_refuse']['t_pay_50pct']}, free window "
            f"{S['BAR2_pay_vs_refuse']['free_window_half_lives']}, t_refuse "
            f"{S['BAR2_pay_vs_refuse']['t_refuse_measured']})")

    ok = [s for s in res["per_seed"] if "NOT_RUN" not in res["per_seed"][s]]
    if ok:
        res["across_seeds"] = {
            "n_seeds_scored": len(ok),
            "BAR1_pass_all": all(res["per_seed"][s]
                                 ["BAR1_decay_vs_eviction_at_last_alive_step"]["PASS"]
                                 for s in ok),
            "BAR2_dominated_all": all(res["per_seed"][s]["BAR2_pay_vs_refuse"]["DOMINATED"]
                                      for s in ok),
            "eviction_delta_per_seed": {s: res["per_seed"][s]["cost_decomposition"]
                                        ["instant_eviction"] for s in ok},
            "free_window_per_seed": {s: res["per_seed"][s]["BAR2_pay_vs_refuse"]
                                     ["free_window_half_lives"] for s in ok},
            "t_pay50_per_seed": {s: res["per_seed"][s]["BAR2_pay_vs_refuse"]["t_pay_50pct"]
                                 for s in ok},
            "frac_paid_by_weight_per_seed": {
                s: res["per_seed"][s]["cost_decomposition"]["frac_paid_by_weight"]
                for s in ok},
        }
    with open(out / "bars.json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=lambda o: (
            float(o) if isinstance(o, (np.floating, np.integer)) else
            (o.tolist() if isinstance(o, np.ndarray) else str(o))))
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="0,1,2")
    a = ap.parse_args(argv)
    r = analyse(Path(a.out), tuple(int(s) for s in a.seeds.split(",")))
    print(json.dumps(r.get("across_seeds", {}), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
