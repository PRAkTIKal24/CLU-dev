"""hub/placement -- THE RESULTS LEDGER: every PREREG bar, every seed, every control.

One file answers "did each pre-registered bar pass, on each of the three seeds", and it
answers it from the artifacts on disk without re-running anything.  The unit is the CORNER,
paired; the statistic is the mean paired difference with a sign test and a paired bootstrap
(``exp_gen_factor_wells.paired_estimand``, inherited, not re-implemented).

⛔ NO BAR IS DEFINED HERE.  The bars come from ``PREREG-hub-placement-1.md`` and are
transcribed into ``BARS`` below with the PREREG's own numbers.  A bar this module could
choose is a bar that was not pre-registered.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import exp_gen_factor_wells as W

#: Row 4's headline metric, in the PREREG's words.
METRIC = "distinct_valid_novel_tuples"

#: The PREREG's margin, in the PREREG's currency.  Wave 3's factor-wells PREREG declared the
#: same number in the same units, so this hub's bar cannot be softer than the one before it.
MARGIN = 1.0

#: Every pre-registered bar, transcribed from PREREG-hub-placement-1.md section 4.
#: (bar, arm, control, family, rule)
BARS = (
    (1, "place_norm_rank", "factor_aligned_balanced", "nway",
     "delta >= +1.0 AND bootstrap CI95 low > +1.0, on all three seeds -- THE ABLATION HALF"),
    (2, "place_norm_rank", "ward_balanced", "nway",
     "delta >= +1.0 -- the gain is alignment, not equal wells"),
    (3, "place_norm_rank", "place_rand_norm_rank", "nway",
     "delta >= +1.0 beyond seed spread -- the arm beats its OWN size-matched floor"),
    (6, "place_norm_rank", "factor_aligned_balanced", "pair",
     "delta >= 0.0 -- row 3 is a HOLD, not a win; it must not be paid for"),
)

#: Charter section C.2 / C.1 controls.  Bar 4 wants a COLLAPSE, bar 5 a non-lookup.
COLLAPSE_CONTROLS = ("place_norm_rank_removed", "place_norm_rank_shuffled")
COLLAPSE_MAX_FRACTION = 0.25
LOOKUP_CONTROLS = ("place_norm_rank_lookupargmax", "place_norm_rank_lookupconvex",
                   "knn_proto")

#: Context rows: reported on every table, never a bar.
CONTEXT = ("basis_head", "random_dmatched_balanced")


def per_corner(seed_dir: Path, arm: str, seen, cards, family: str) -> list:
    """One row per corner of ``family``, in a fixed corner order, for the pairing."""
    p = seed_dir / "arms" / arm / "partial.npz"
    if not p.exists():
        return []
    z = np.load(p, allow_pickle=True)["done"].item()
    out = []
    for cid in sorted(z):
        d = z[cid]
        if str(d.get("family")) != family:
            continue
        v = d["valid"]
        keep = d["tuples"][v]
        if keep.size:
            lin = np.unique(np.ravel_multi_index(tuple(keep.T), cards))
            novel = lin[~seen[lin]]
        else:
            lin = novel = np.zeros(0, np.int64)
        out.append({"corner_id": cid, "n_requests": int(d["n"]),
                    "judge_accuracy": float(v.mean()),
                    "accept": float(d["accept"].mean()),
                    "distinct_valid_tuples": int(lin.size),
                    METRIC: int(novel.size)})
    return out


def build(root: Path, seeds=(0, 1, 2), log=print) -> dict:
    W.PINS["condition"] = "test"
    cond = W.load_condition()
    grid = cond["grid"]
    cards = tuple(int(c) for c in grid.cardinalities)
    seen = W.seen_tuple_mask(grid, cond["train_rows"])
    res = {"metric": METRIC, "margin_declared_in_advance": MARGIN,
           "prereg": "PREREG-hub-placement-1.md", "condition": "test",
           "seeds": list(seeds), "per_seed": {}, "bars": {}, "arms_present": {}}

    for s in seeds:
        sd = root / f"seed{s}"
        if not (sd / "arms").exists():
            log(f"  seed {s}: no arms yet")
            continue
        W.PINS["seed"] = int(s)
        arms = sorted(d.name for d in (sd / "arms").iterdir()
                      if (d / "partial.npz").exists())
        res["arms_present"][str(s)] = arms
        rows = {fam: {a: per_corner(sd, a, seen, cards, fam) for a in arms}
                for fam in ("nway", "pair")}
        entry = {"feature_cache": W.feature_cache_manifest(int(s)),
                 "n_corners": {fam: (len(next(iter(rows[fam].values())))
                                     if rows[fam] else 0)
                               for fam in rows},
                 "means": {}, "paired": {}}
        for fam in ("nway", "pair"):
            for a in arms:
                if not rows[fam][a]:
                    continue
                entry["means"].setdefault(fam, {})[a] = {
                    "novel_per_corner": float(np.mean([r[METRIC]
                                                       for r in rows[fam][a]])),
                    "valid_tuples_per_corner": float(np.mean(
                        [r["distinct_valid_tuples"] for r in rows[fam][a]])),
                    "judge_accuracy": float(np.mean([r["judge_accuracy"]
                                                     for r in rows[fam][a]])),
                    "accept": float(np.mean([r["accept"] for r in rows[fam][a]])),
                }
        for bar, arm, ctl, fam, rule in BARS:
            if not rows[fam].get(arm) or not rows[fam].get(ctl):
                continue
            m = MARGIN if bar != 6 else 0.0
            pe = W.paired_estimand(rows[fam][arm], rows[fam][ctl], METRIC, m, arm, ctl)
            pe["bar"] = bar
            pe["rule"] = rule
            pe["family"] = fam
            pe["passes"] = bool(pe["mean_paired_delta"] >= m
                                and (pe["bootstrap_ci95"][0] > m if bar == 1
                                     else pe["mean_paired_delta"] >= m))
            entry["paired"][f"bar{bar}_{arm}_vs_{ctl}_{fam}"] = pe
            log(f"  seed {s} bar {bar}: {arm} vs {ctl} ({fam}) delta="
                f"{pe['mean_paired_delta']:+.3f} ci95={np.round(pe['bootstrap_ci95'], 3).tolist()} "
                f"wins={pe['corners_arm_wins']}/{pe['n_corners']} p={pe['sign_test_p_two_sided']:.3g} "
                f"PASS={pe['passes']}")
        # bar 4: the charter C.2 collapse
        base = (entry["means"].get("nway", {}).get("place_norm_rank") or {}).get(
            "novel_per_corner")
        if base:
            c4 = {}
            for c in COLLAPSE_CONTROLS:
                v = (entry["means"].get("nway", {}).get(c) or {}).get("novel_per_corner")
                if v is None:
                    continue
                c4[c] = {"novel_per_corner": v, "fraction_of_arm": v / base,
                         "collapses": bool(v / base <= COLLAPSE_MAX_FRACTION)}
                log(f"  seed {s} bar 4: {c} = {v:.3f} ({v / base:.3f} of the arm) "
                    f"COLLAPSE={c4[c]['collapses']}")
            entry["bar4_collapse"] = c4
            c5 = {}
            for c in LOOKUP_CONTROLS:
                v = (entry["means"].get("nway", {}).get(c) or {}).get("novel_per_corner")
                if v is None:
                    continue
                c5[c] = {"novel_per_corner": v, "arm_minus_control": base - v,
                         "arm_at_least": bool(base >= v)}
                log(f"  seed {s} bar 5: {c} = {v:.3f}, arm - control = {base - v:+.3f}")
            entry["bar5_lookup"] = c5
        res["per_seed"][str(s)] = entry
        res.setdefault("per_corner", {})[str(s)] = rows

    # the cross-seed verdict: a bar passes only if it passes on EVERY seed present
    for bar, arm, ctl, fam, rule in BARS:
        key = f"bar{bar}_{arm}_vs_{ctl}_{fam}"
        got = [(s, e["paired"][key]) for s, e in res["per_seed"].items()
               if key in e.get("paired", {})]
        if not got:
            continue
        res["bars"][key] = {
            "bar": bar, "rule": rule, "family": fam,
            "n_seeds": len(got),
            "per_seed_delta": {s: p["mean_paired_delta"] for s, p in got},
            "per_seed_ci95_lo": {s: p["bootstrap_ci95"][0] for s, p in got},
            "per_seed_pass": {s: p["passes"] for s, p in got},
            "passes_on_every_seed_present": bool(all(p["passes"] for _, p in got)),
            "mean_delta_across_seeds": float(np.mean([p["mean_paired_delta"]
                                                      for _, p in got])),
            "seed_spread_delta": float(np.ptp([p["mean_paired_delta"] for _, p in got]))
            if len(got) > 1 else 0.0,
            "three_seeds_required_by_w4_common_s2": len(got) >= 3,
        }
    W._jdump(res, root / "table.json")
    log(f"  -> {root / 'table.json'}")
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--seeds", default="0,1,2")
    a = ap.parse_args(argv)
    build(Path(a.root), seeds=tuple(int(x) for x in a.seeds.split(",")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
