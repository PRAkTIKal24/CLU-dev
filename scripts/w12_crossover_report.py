"""w12 hub/single-design-crossover -- aggregate the val sweep, APPLY the frozen rule.

Reads every `val_sweep/val_K*_seed*/row.json` that exists, prints the crossover table,
and applies the selection rule EXACTLY as `SELECTION-RULE.md` states it.  The rule is
read from this file's constants, which mirror the dated file; it is not re-derived from
the data and it is not moved.

    K* = max { K in SWEEP : v(5) - mean_seeds v(K) <= 0.02 }
    K_cross = min { K in SWEEP : v(5) - mean_seeds v(K) > 0.10 }

A K with no row on disk is a DECLARED NOT-RUN and is printed as such -- never a zero,
never a blank, and never silently dropped from the max.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

OUT = Path("C:/chlu/outputs/w12-single-design-crossover")
SWEEP = (5, 8, 10, 12, 16, 20, 24, 28, 32)
WITHIN = 0.02          # the selection tolerance, declared 2026-09-21T11:10:24Z
CROSS = 0.10           # the crossover threshold, declared in the same file


def load(condition: str, seeds):
    rows = {}
    for K in SWEEP:
        per = {}
        for s in seeds:
            p = OUT / "val_sweep" / f"{condition}_K{K}_seed{s}" / "row.json"
            if p.exists():
                per[s] = json.loads(p.read_text(encoding="utf-8"))
        rows[K] = per
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="val")
    ap.add_argument("--seed", type=int, action="append", default=None)
    a = ap.parse_args(argv)
    seeds = a.seed or [0, 1, 2]
    rows = load(a.condition, seeds)

    out = {"SMOKE_NOT_A_CLAIM": a.condition == "val",
           "condition": a.condition, "seeds_requested": seeds,
           "rule": {"within": WITHIN, "crossover": CROSS,
                    "declared_at": "2026-09-21T11:10:24Z",
                    "file": str(OUT / "SELECTION-RULE.md")},
           "per_K": {}, "not_run": []}

    print(f"\n{'K':>4} {'n':>3} {'A':>3} {'width':>9} {'rank/width':>18} "
          f"{'val comp':>16} {'drop vs K=5':>12} {'4-part B':>12} {'xPin':>7} "
          f"{'1x params':>11}")
    print("-" * 121)
    v5 = None
    for K in SWEEP:
        per = rows[K]
        if not per:
            out["not_run"].append({"K": K, "reason":
                                   "DECLARED NOT-RUN -- no row.json on disk"})
            print(f"{K:>4} {'--':>3} {'--':>3} {'DECLARED NOT-RUN -- no row on disk':>60}")
            continue
        ss = sorted(per)
        comp = np.array([per[s]["composition_val"]["valid_mean"] for s in ss])
        A = per[ss[0]]["A_requested"]
        wmin = min(per[ss[0]]["well_sizes"])
        wmax = max(per[ss[0]]["well_sizes"])
        rw_lo = min(per[s]["rank_over_width_min"] for s in ss)
        rw_hi = max(per[s]["rank_over_width_max"] for s in ss)
        tot = per[ss[0]]["ledger_four_part"]["honest_total_bytes"]
        xp = per[ss[0]]["ledger_four_part"]["times_reference_pin"]
        m = float(comp.mean())
        if K == 5:
            v5 = m
        drop = (v5 - m) if v5 is not None else float("nan")
        sd = float(comp.std(ddof=1)) if comp.size > 1 else float("nan")
        led = per[ss[0]]["ledger_four_part"]
        out["per_K"][str(K)] = {
            "n_seeds": len(ss), "seeds": ss, "A_requested": int(A),
            # THE 1x MATCHING TARGET, and the REFERENCE column beside it.
            # The learned baselines are matched to the single design's OWN four-part
            # total this wave. The 524,288 B figure is a reference, not a budget.
            "baseline_match_1x": {
                "target_bytes": int(led["honest_total_bytes"]),
                "target_params_float32": int(led["honest_total_bytes"] // 4),
                "reference_pin_bytes": 524_288,
                "reference_pin_params_float32": 524_288 // 4,
                "times_reference": led["times_reference_pin"],
                "note": ("1x means 1x THIS OBJECT'S four-part total. The 524,288 B "
                         "column is printed for continuity with waves 5-8 and is NOT "
                         "a budget this wave."),
            },
            "well_width_min": int(wmin), "well_width_max": int(wmax),
            "rank_over_width_min": float(rw_lo), "rank_over_width_max": float(rw_hi),
            "val_composition_per_seed": {str(s): float(per[s]["composition_val"]
                                                       ["valid_mean"]) for s in ss},
            "val_composition_mean": m, "val_composition_sd": sd,
            "drop_from_K5": float(drop),
            "ledger_four_part": per[ss[0]]["ledger_four_part"],
            "controls": {s: per[s]["controls"] for s in ss},
            "exactness": {s: per[s]["exactness"] for s in ss},
        }
        print(f"{K:>4} {len(ss):>3} {A:>3} {wmin:>4}..{wmax:<4} "
              f"{rw_lo:>8.4f}..{rw_hi:<8.4f} {m:>8.4f}+-{sd:<6.4f} "
              f"{drop:>12.4f} {tot:>12,} {xp:>7.3f} {tot // 4:>11,}")

    ran = [K for K in SWEEP if rows[K]]
    if v5 is None:
        print("\nK=5 HAS NO ROW -- the rule references v(5) and cannot be applied. "
              "DECLARED NOT-RUN.")
        out["K_star"] = None
        out["K_star_reason"] = "K=5 reference row absent; rule not applicable"
    else:
        ok = [K for K in ran if (v5 - out["per_K"][str(K)]["val_composition_mean"])
              <= WITHIN]
        out["K_star"] = int(max(ok)) if ok else None
        out["K_star_candidates"] = ok
        print(f"\nRULE: K* = max{{K : v(5) - v(K) <= {WITHIN}}} over the K actually run "
              f"{ran}")
        print(f"      satisfying set = {ok}")
        print(f"      K* = {out['K_star']}")
        cross = [K for K in ran
                 if (v5 - out["per_K"][str(K)]["val_composition_mean"]) > CROSS]
        out["K_crossover"] = int(min(cross)) if cross else None
        out["K_crossover_note"] = (
            None if cross else
            f"no K in the swept range drops more than {CROSS} below K=5; the range "
            "is NOT extended to manufacture a crossover")
        print(f"      CROSSOVER (first K with drop > {CROSS}) = {out['K_crossover']}"
              + ("" if cross else "  -- none inside the swept range"))

    # --- the control gate --------------------------------------------------
    if out.get("K_star"):
        Ks = str(out["K_star"])
        gate = {}
        for s, cs in out["per_K"][Ks]["controls"].items():
            for name, sc in (cs or {}).items():
                gate.setdefault(name, []).append(sc["valid_mean"])
        out["control_gate_at_K_star"] = {
            k: {"per_seed": v, "mean": float(np.mean(v))} for k, v in gate.items()}
        base = out["per_K"][Ks]["val_composition_mean"]
        print(f"\nCONTROL GATE at K*={Ks} (val; the SCORED gate is re-run on test):")
        for k, v in out["control_gate_at_K_star"].items():
            print(f"  {k:34s} {v['mean']:.4f}   (clu {base:.4f}, "
                  f"drop {base - v['mean']:+.4f})")

    p = OUT / f"CROSSOVER-{a.condition}.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
