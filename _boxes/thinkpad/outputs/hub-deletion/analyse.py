#!/usr/bin/env python
"""PREREG-01 §4/§5 analysis. Reads the per-row predictions each unit wrote and
produces the collateral statistic, the unrecoverability table and the preservation
table. Pure post-processing: it builds nothing and scores nothing.

THE ALGEBRA, stated because it matters
--------------------------------------
    C_arm(f) = mean_{f' != f} [ (acc_del - acc_pre) - (acc_plc - acc_pre) ]
             = mean_{f' != f} [  acc_del - acc_plc ]

The pre-deletion term **cancels exactly**, so the headline does not depend on the
undeleted build at all -- `det_A` enters only as the pre-deletion CONTROL for
condition 1, where it is mandatory. That cancellation is a property of the
pre-registered statistic, not a convenience chosen afterwards.

    Delta(f) = C_clu(f) - C_knn(f)

Pairing unit = the eval row. On B every one of the six deleted values is absent by
construction (PREREG §2), so all 72 rows enter for every f and the six can be paired
row-wise; on dev/a the excluded rows differ per f, so the six Delta(f) are treated as
six observations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load(out: Path, tag: str) -> dict:
    d = {"tag": tag}
    with open(out / tag / "eval.json") as fh:
        d["eval"] = json.load(fh)
    d["preds"] = np.load(out / tag / "preds.npz")
    rp = out / tag / "rival.json"
    if rp.exists():
        with open(rp) as fh:
            d["rival"] = json.load(fh)
        d["rival_preds"] = np.load(out / tag / "rival_preds.npz")
    return d


def correct(u: dict, s: str, arm: str) -> np.ndarray:
    """(n, F) boolean per-row correctness for one arm on one row set."""
    labs = u["preds"][f"{s}_labels"]
    if arm == "knn_proto":
        p = u["rival_preds"][f"{s}_knn_proto_preds"]
    else:
        p = u["preds"][f"{s}_{arm}_preds"]
    return np.asarray(p) == np.asarray(labs)


def boot_ci(z: np.ndarray, n_boot: int = 20000, seed: int = 20260917) -> tuple:
    """Percentile bootstrap over the pairing unit. Fixed seed: the CI never moves."""
    rng = np.random.default_rng(seed)
    n = z.shape[0]
    if n < 2:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, n, size=(n_boot, n))
    means = z[idx].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def verdict_r1(delta: float, lo: float, hi: float, m: float = 0.020) -> str:
    """Head R1, applied verbatim. Set before any scored run (PREREG-01 §4)."""
    if lo <= 0.0 <= hi:
        return "tie (CI includes 0)"
    if abs(delta) <= m:
        return "tie (|Delta| <= m)"
    if delta > 0:
        return "WIN"
    return "LOSS"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--pre", default="det_A")
    ap.add_argument("--sets", default="dev,a")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    out = Path(args.out)
    with open(args.meta) as fh:
        meta = json.load(fh)
    names = meta["factor_names"]
    cards = meta["cardinalities"]
    F = len(names)
    sets = [s for s in args.sets.split(",") if s]

    pre = load(out, args.pre)
    report = {"schema": "chlu-hub-deletion/analysis/1", "pre_tag": args.pre,
              "factor_names": names, "cardinalities": cards, "sets": sets,
              "m": 0.020, "arms": ["clu", "knn_proto"],
              "bytes": {"clu_store": 64 * 2048 * 4}, "collateral": {},
              "unrecoverability": {}, "preservation": {}, "rebuild_cost_s": {},
              "laundering_floor": {}, "missing": []}

    dels = [d for d in meta["deletions"] if d["f"] is not None]

    # ---- rebuild cost line ------------------------------------------------
    for d in dels + [d for d in meta["deletions"] if d["f"] is None]:
        p = out / d["tag"] / "build.json"
        if p.exists():
            with open(p) as fh:
                b = json.load(fh)
            report["rebuild_cost_s"][d["tag"]] = {
                "n_live": b["n_live"], "n_deleted": b["n_deleted"],
                "train_s": b["timing_s"]["train"], "book_s": b["timing_s"]["book"],
                "build_total_s": b["timing_s"]["total"], "unit_wall_s": b["unit_wall_s"]}

    for s in sets:
        coll, unrec, pres, floors = {}, {}, {}, {}
        per_row_z = {}
        for d in dels:
            f, v, tag, ptag = d["f"], d["value"], d["tag"], d["placebo"]
            try:
                ud, up = load(out, tag), load(out, ptag)
            except FileNotFoundError as e:
                report["missing"].append(f"{tag}/{ptag} ({s}): {e}")
                continue
            if f"{s}_labels" not in ud["preds"].files:
                report["missing"].append(f"{tag}: set {s} not scored")
                continue
            labs = np.asarray(ud["preds"][f"{s}_labels"])
            keep = labs[:, f] != v          # PREREG: compositions NOT naming v
            hit = labs[:, f] == v           # the deleted value itself

            entry = {"n_rows_total": int(labs.shape[0]),
                     "n_rows_not_naming_v": int(keep.sum()),
                     "n_rows_naming_v": int(hit.sum()),
                     "n_deleted_rows": d["n_deleted"], "factor": names[f], "value": v}

            have_rival = "rival" in ud and "rival" in up
            if not have_rival:
                report["missing"].append(f"{tag}/{ptag}: rival.json absent")
            keep_ok = True

            # ---- per-row z, averaged over f' != f --------------------------
            # ⛔ On a TARGETED set every row names v, so there is no surviving-space
            # sample and the collateral statistic is simply undefined there.  It is
            # left out rather than computed on an empty slice and reported as 0.
            others = [j for j in range(F) if j != f]
            if not keep.any():
                entry["collateral_undefined"] = "every row in this set names v"
                coll[names[f]] = entry
                keep_ok = False
            else:
                keep_ok = True
            cd_clu, cp_clu = correct(ud, s, "clu"), correct(up, s, "clu")
            dz_clu = (cd_clu.astype(np.float64) - cp_clu.astype(np.float64))
            if not keep_ok:
                dz_k = None
            entry["C_clu"] = float(dz_clu[keep][:, others].mean()) if keep_ok else None
            entry["C_clu_per_factor"] = {names[j]: float(dz_clu[keep][:, j].mean())
                                         for j in range(F)} if keep_ok else None
            if have_rival and keep_ok:
                cd_k, cp_k = correct(ud, s, "knn_proto"), correct(up, s, "knn_proto")
                dz_k = (cd_k.astype(np.float64) - cp_k.astype(np.float64))
                entry["C_knn"] = float(dz_k[keep][:, others].mean())
                entry["C_knn_per_factor"] = {names[j]: float(dz_k[keep][:, j].mean())
                                             for j in range(F)}
                z = (dz_clu - dz_k)[keep][:, others].mean(axis=1)
                per_row_z[f] = (z, keep)
                lo, hi = boot_ci(z)
                se = float(z.std(ddof=1) / np.sqrt(z.size)) if z.size > 1 else float("nan")
                entry.update({"Delta": float(z.mean()), "ci95_boot": [lo, hi],
                              "ci95_normal": [float(z.mean() - 1.96 * se),
                                              float(z.mean() + 1.96 * se)],
                              "se": se, "n_pairs": int(z.size),
                              "verdict_R1": verdict_r1(float(z.mean()), lo, hi)})
            coll[names[f]] = entry

            # ---- condition 1: unrecoverability, WITH the pre-deletion control
            u = {"chance": 1.0 / cards[f], "n_requests_naming_v": int(hit.sum())}
            if hit.any():
                for arm, src in (("clu", ud), ("knn_proto", ud if have_rival else None)):
                    if src is None:
                        continue
                    a_post = float(correct(ud, s, arm)[hit][:, f].mean())
                    a_pre = float(correct(pre, s, arm)[hit][:, f].mean()) \
                        if (arm != "knn_proto" or "rival" in pre) else None
                    u[arm] = {"acc_pre_on_v": a_pre, "acc_post_on_v": a_post}
                    if a_pre is not None:
                        u[arm]["vacuous"] = bool(a_pre <= 1.0 / cards[f] + 1e-9)
                # "no rendered tuple contains it": over ALL requests in the set
                pv = np.asarray(ud["preds"][f"{s}_clu_preds"])[:, f]
                pv_pre = np.asarray(pre["preds"][f"{s}_clu_preds"])[:, f]
                u["frac_all_renders_showing_v_post"] = float((pv == v).mean())
                u["frac_all_renders_showing_v_pre"] = float((pv_pre == v).mean())
            unrec[names[f]] = u

            # ---- condition 2: preservation, absolute post numbers ----------
            pres[names[f]] = None if not keep_ok else {
                "clu_post_not_naming_v": {names[j]: float(cd_clu[keep][:, j].mean())
                                          for j in range(F)},
                "clu_placebo_not_naming_v": {names[j]: float(cp_clu[keep][:, j].mean())
                                             for j in range(F)},
                "clu_pre_not_naming_v": {names[j]: float(correct(pre, s, "clu")[keep][:, j].mean())
                                         for j in range(F)},
            }
            if have_rival and keep_ok:
                pres[names[f]]["knn_post_not_naming_v"] = {
                    names[j]: float(correct(ud, s, "knn_proto")[keep][:, j].mean())
                    for j in range(F)}
            floors[names[f]] = {
                "store_removed_post": ud["eval"]["sets"][s]["reads"]["store_removed"]["aggregate"],
                "clu_post": ud["eval"]["sets"][s]["reads"]["clu"]["aggregate"],
                "feature_addr_post": ud["eval"]["sets"][s]["reads"]["feature_addr"]["aggregate"],
            }

        # ---- the overall statistic over the 6 factors ---------------------
        overall = {}
        ds = [coll[names[d["f"]]] for d in dels if names[d["f"]] in coll
              and "Delta" in coll[names[d["f"]]]]
        if ds:
            vals = np.array([e["Delta"] for e in ds])
            overall["per_factor_Delta"] = {e["factor"]: e["Delta"] for e in ds}
            overall["mean_Delta"] = float(vals.mean())
            lo, hi = boot_ci(vals)
            overall["ci95_boot_over_factors"] = [lo, hi]
            overall["n_factors"] = int(vals.size)
            overall["verdict_R1_over_factors"] = verdict_r1(float(vals.mean()), lo, hi)
            # row-paired when every factor kept the same rows (true on B)
            ks = [per_row_z[d["f"]] for d in dels if d["f"] in per_row_z]
            if ks and all(k[1].all() for k in ks) and len({k[0].size for k in ks}) == 1:
                Z = np.stack([k[0] for k in ks], axis=1).mean(axis=1)
                lo2, hi2 = boot_ci(Z)
                overall["row_paired"] = {
                    "mean_Delta": float(Z.mean()), "ci95_boot": [lo2, hi2],
                    "n_rows": int(Z.size),
                    "verdict_R1": verdict_r1(float(Z.mean()), lo2, hi2),
                    "note": "every row kept for every factor (no deleted value occurs in this set)"}
        report["collateral"][s] = {"per_deletion": coll, "overall": overall}
        report["unrecoverability"][s] = unrec
        report["preservation"][s] = pres
        report["laundering_floor"][s] = floors

    # ---- the churn band ---------------------------------------------------
    churn = {}
    for s in sets:
        vals = []
        for t in meta["churn_tags"]:
            p = out / t / "eval.json"
            if p.exists():
                with open(p) as fh:
                    e = json.load(fh)
                if s in e["sets"]:
                    vals.append(e["sets"][s]["reads"]["clu"]["aggregate"])
        if vals:
            churn[s] = {"tags": meta["churn_tags"][:len(vals)], "clu_aggregate": vals,
                        "mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1))
                        if len(vals) > 1 else None,
                        "spread": float(max(vals) - min(vals))}
    report["churn_band"] = churn

    txt = json.dumps(report, indent=2, sort_keys=True, default=float)
    if args.json:
        Path(args.json).write_text(txt, encoding="utf-8")
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
