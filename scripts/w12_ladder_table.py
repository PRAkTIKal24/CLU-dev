"""w12 hub/composition-ladder -- aggregate every leg into LADDER.json / LADDER.md / the figure.

Reads only what is on disk in ``runs/<rung>_seed<seed>/rows_<arm>.json``.
Anything absent is a DECLARED NOT-RUN with a reason -- never a zero and never a blank.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import chlu

if "CHLU-w12-composition-ladder" not in str(chlu.__file__):
    raise AssertionError("chlu resolves OUTSIDE this worktree: %s" % chlu.__file__)

from chlu.experiments import exp_gen_factor_wells as fw

RUNGS = ["2", "3", "4", "5", "6"]
SEEDS = [0, 1, 2]
STORE = ["single_design_K5", "pca_tensor"]
CONTROLS = ["landscape_shuffled", "random_atoms"]
LEARNED = ["rival_mlp", "rival_lstm", "rival_transformer", "rival_cont_mlp", "rival_table"]
#: AMENDMENT-1 arms: fitted by THIS hub on each rung's OWN pool.  They are reported as their
#: own columns and are DELIBERATELY NOT in LEARNED -- PREREG section 5's best-learned-baseline
#: rule is over the five pre-existing learned arms and must not move.
OWNPOOL = ["learned_ref_ownpool", "learned_1x"]
ARMS = (STORE + ["%s__%s" % (s, c) for s in STORE for c in CONTROLS]
        + LEARNED + OWNPOOL)
METRIC = "distinct_valid_novel_tuples"

LABEL = {
    "single_design_K5": "single design (K=5, DECLARED)",
    "pca_tensor": "pca_tensor (wave-5 subspace store)",
    "rival_mlp": "MLP", "rival_lstm": "LSTM", "rival_transformer": "transformer",
    "rival_cont_mlp": "continuous MLP", "rival_table": "table floor",
    "single_design_K5__landscape_shuffled": "single design - landscape_shuffled",
    "single_design_K5__random_atoms": "single design - random_atoms",
    "pca_tensor__landscape_shuffled": "pca_tensor - landscape_shuffled",
    "pca_tensor__random_atoms": "pca_tensor - random_atoms",
}


def load(runs: Path) -> dict:
    """{rung: {seed: {arm: {'summary':..., 'per_corner':...}}}} plus the NOT-RUN list."""
    out, missing = {}, []
    for rung in RUNGS + ["shipped"]:
        out[rung] = {}
        for s in SEEDS:
            leg = runs / ("%s_seed%d" % (rung, s))
            out[rung][s] = {}
            for arm in ARMS:
                p = leg / ("rows_%s.json" % arm)
                if not p.exists():
                    f = leg / ("FAILED_%s.json" % arm)
                    why = "leg directory absent" if not leg.exists() else (
                        json.load(open(f, encoding="utf-8"))["error"] if f.exists()
                        else "arm not scored in this leg")
                    missing.append({"rung": rung, "seed": s, "arm": arm, "reason": why})
                    continue
                with open(p, encoding="utf-8") as fh:
                    out[rung][s][arm] = json.load(fh)
    return {"data": out, "not_run": missing}


def _per(d, arm):
    return d.get(arm, {}).get("per_corner")


def best_learned(d: dict) -> tuple:
    """The PRE-DECLARED rule (PREREG section 5): the learned arm with the largest mean."""
    cand = [(a, float(np.mean([r[METRIC] for r in _per(d, a)]))) for a in LEARNED
            if _per(d, a) is not None]
    if not cand:
        return None, None
    cand.sort(key=lambda t: -t[1])
    return cand[0][0], cand[0][1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    runs, out = Path(a.runs), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    L = load(runs)
    D = L["data"]

    rep = {"n_seeds_declared": len(SEEDS), "seeds": SEEDS,
           "n_configurations_per_rung": None,
           "metric": METRIC,
           "margin_name": "DECORATION_MARGIN_NOVELTY",
           "margin": float(fw.DECORATION_MARGIN_NOVELTY),
           "estimand_source": ("LADDER-READY.json:estimand / w12-kway-ladder-harness "
                               "PREREG - pre-declared, not restated"),
           "best_learned_rule": ("PREREG section 5, fixed 2026-09-21T11:38:41Z: per (rung, "
                                 "seed), the learned arm with the LARGEST mean "
                                 "distinct_valid_novel_tuples"),
           "rungs": {}, "not_run": L["not_run"]}

    # ---- per rung -------------------------------------------------------
    for rung in RUNGS + ["shipped"]:
        r = {"rung": rung, "seeds_present": [], "arms": {}, "best_learned_per_seed": {},
             "n_corners_per_seed": {}, "ceiling_per_corner": None, "leg": {}}
        for s in SEEDS:
            d = D[rung][s]
            if not d:
                continue
            r["seeds_present"].append(s)
            any_arm = next(iter(d.values()))
            r["n_corners_per_seed"][s] = len(any_arm["per_corner"])
            # the CEILING of this metric is the corner's own row count -- a corner cannot
            # yield more distinct tuples than it has rows.  k=2 corners hold 36 rows and
            # k>=3 corners hold 72, so the ABSOLUTE scale changes at k=3 and the rungs are
            # NOT on one absolute scale.  Printed, never silently normalised away.
            r["ceiling_per_corner"] = int(max(x["n_requests"] for x in any_arm["per_corner"]))
            lp = runs / ("%s_seed%d" % (rung, s)) / "LEG.json"
            if lp.exists():
                with open(lp, encoding="utf-8") as fh:
                    lj = json.load(fh)
                r["leg"][s] = {k: lj[k] for k in ("n_train", "n_test", "n_corners",
                                                  "corner_check", "cache")
                               if k in lj}
            bl, blv = best_learned(d)
            r["best_learned_per_seed"][s] = {"arm": bl, "mean": blv}
        for arm in ARMS:
            per_seed = {}
            for s in r["seeds_present"]:
                pc = _per(D[rung][s], arm)
                if pc is None:
                    continue
                v = np.array([x[METRIC] for x in pc], float)
                e = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                     "n_corners": int(v.size),
                     "judge_accuracy_mean": float(np.mean([x["judge_accuracy"] for x in pc]))}
                summ = D[rung][s][arm]["summary"]
                for key in ("rank_width", "byte_ledger", "address_bytes", "param_count",
                            "population", "byte_matching", "perturbation"):
                    if key in summ:
                        e[key] = summ[key]
                # the paired margin against the PRE-DECLARED best learned baseline
                bl = r["best_learned_per_seed"][s]["arm"]
                if bl is not None and _per(D[rung][s], bl) is not None:
                    e["paired_vs_best_learned"] = fw.paired_estimand(
                        pc, _per(D[rung][s], bl), METRIC,
                        float(fw.DECORATION_MARGIN_NOVELTY), arm, bl)
                per_seed[s] = e
            if not per_seed:
                r["arms"][arm] = {"status": "DECLARED NOT-RUN",
                                  "reason": "no seed of this rung scored this arm"}
                continue
            m = np.array([per_seed[s]["mean"] for s in sorted(per_seed)], float)
            ceil = r["ceiling_per_corner"] or 1
            r["arms"][arm] = {
                "status": "RUN", "n_seeds": int(m.size),
                "per_seed": per_seed,
                "across_seed_mean": float(m.mean()),
                "across_seed_min": float(m.min()), "across_seed_max": float(m.max()),
                "across_seed_sd": float(m.std(ddof=1)) if m.size > 1 else None,
                "ceiling_per_corner": int(ceil),
                "across_seed_mean_frac_of_ceiling": float(m.mean() / ceil),
            }
        rep["rungs"][rung] = r

    # ---- the CONTROL verdict: does the store survive its own controls? ---
    verdict = {}
    for rung in RUNGS + ["shipped"]:
        verdict[rung] = {}
        for base in STORE:
            v = {}
            for ctl in CONTROLS:
                arm = "%s__%s" % (base, ctl)
                per = []
                for s in rep["rungs"][rung]["seeds_present"]:
                    pa = _per(D[rung][s], base)
                    pc = _per(D[rung][s], arm)
                    if pa is None or pc is None:
                        continue
                    est = fw.paired_estimand(pa, pc, METRIC,
                                             float(fw.DECORATION_MARGIN_NOVELTY), base, arm)
                    per.append({"seed": s, "arm_mean": est["arm_mean"],
                                "control_mean": est["control_mean"],
                                "mean_paired_delta": est["mean_paired_delta"],
                                "bootstrap_ci95": est["bootstrap_ci95"],
                                "sign_test_p_two_sided": est["sign_test_p_two_sided"],
                                "beats_beyond_margin": est["beats_beyond_margin"]})
                if not per:
                    v[ctl] = {"status": "DECLARED NOT-RUN",
                              "reason": "arm or control absent at this rung"}
                    continue
                collapses = all(p["beats_beyond_margin"] for p in per)
                v[ctl] = {"status": "RUN", "n_seeds": len(per), "per_seed": per,
                          "control_collapses_on_every_seed": bool(collapses),
                          "rule": ("the control COLLAPSES at this rung iff the arm beats it "
                                   "beyond the pre-declared margin +1.0 with the paired "
                                   "bootstrap CI95 lower bound also above +1.0, on EVERY seed")}
            decorative = any(
                (x.get("status") == "RUN" and not x["control_collapses_on_every_seed"])
                for x in v.values())
            v["VERDICT"] = ("DECORATIVE at this rung -- NO ROW IS CLAIMED"
                            if decorative else
                            "the store survives both charter-C.2 controls at this rung")
            verdict[rung][base] = v
    rep["control_verdicts"] = verdict

    # ---- the two END REPRODUCTION checks --------------------------------
    ends = {}
    for tag, rung, ship_slice, rung_slice in (
            ("k2_vs_shipped_row3", "2", slice(0, 60), slice(0, 60)),
            ("k6_vs_shipped_row4", "6", slice(60, 109), slice(0, 49))):
        t = {"rung": rung, "reference": "shipped value_holdout condition on disk",
             "per_arm": {}}
        for arm in ARMS:
            rows = []
            for s in SEEDS:
                pr = _per(D[rung][s], arm)
                ps = _per(D["shipped"][s], arm)
                if pr is None or ps is None:
                    continue
                vr = np.array([x[METRIC] for x in pr[rung_slice]], float)
                vs = np.array([x[METRIC] for x in ps[ship_slice]], float)
                rows.append({"seed": s, "rung_mean": float(vr.mean()),
                             "rung_sd": float(vr.std(ddof=1)),
                             "shipped_mean": float(vs.mean()),
                             "shipped_sd": float(vs.std(ddof=1)),
                             "abs_diff": float(abs(vr.mean() - vs.mean())),
                             "within_own_corner_sd": bool(
                                 abs(vr.mean() - vs.mean()) <= vs.std(ddof=1)),
                             "n_rung_corners": int(vr.size),
                             "n_shipped_corners": int(vs.size)})
            if not rows:
                t["per_arm"][arm] = {"status": "DECLARED NOT-RUN",
                                     "reason": "rung or shipped leg absent for this arm"}
                continue
            t["per_arm"][arm] = {
                "status": "RUN", "n_seeds": len(rows), "per_seed": rows,
                "reproduces_on_every_seed": bool(all(x["within_own_corner_sd"] for x in rows)),
                "max_abs_diff": float(max(x["abs_diff"] for x in rows))}
        runall = [v for v in t["per_arm"].values() if v.get("status") == "RUN"]
        t["VERDICT"] = ("NOT TESTED -- the reference leg is absent" if not runall else
                        ("the end reproduces on every arm and every seed"
                         if all(v["reproduces_on_every_seed"] for v in runall)
                         else "OFF-INSTRUMENT: at least one arm does not reproduce"))
        ends[tag] = t
    rep["end_reproduction"] = ends

    with open(out / "LADDER.json", "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1, default=str)
    print("wrote", out / "LADDER.json")

    # ---- a flat CSV, one row per (rung, seed, arm) ----------------------
    import csv
    with open(out / "LADDER.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rung", "k", "seed", "arm", "n_corners", "mean_" + METRIC, "sd_" + METRIC,
                    "judge_accuracy_mean", "rank_over_width_max", "param_count",
                    "byte_matching", "best_learned_baseline",
                    "paired_delta_vs_best_learned", "ci95_lo", "ci95_hi", "sign_p"])
        for rung in RUNGS + ["shipped"]:
            for arm in ARMS:
                A = rep["rungs"][rung]["arms"].get(arm, {})
                if A.get("status") != "RUN":
                    continue
                for s, e in sorted(A["per_seed"].items()):
                    pe = e.get("paired_vs_best_learned") or {}
                    w.writerow([rung, rung if rung != "shipped" else "",
                                s, arm, e["n_corners"],
                                "%.6f" % e["mean"], "%.6f" % e["sd"],
                                "%.6f" % e["judge_accuracy_mean"],
                                (e.get("rank_width") or {}).get("rank_over_width_max", ""),
                                e.get("param_count", ""), e.get("byte_matching", ""),
                                pe.get("control", ""),
                                "" if not pe else "%.6f" % pe["mean_paired_delta"],
                                "" if not pe else "%.6f" % pe["bootstrap_ci95"][0],
                                "" if not pe else "%.6f" % pe["bootstrap_ci95"][1],
                                "" if not pe else "%.4g" % pe["sign_test_p_two_sided"]])
    print("wrote", out / "LADDER.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
