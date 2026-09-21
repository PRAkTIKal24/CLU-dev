"""w12b hub/nr6-and-kstar -- the K* ladder table, and the CORRECTION it forces.

⛔ RE-DERIVES NOTHING.  Every cell is read from a `rows_*.json` or a `LEG.json` on disk and
every cell names its file.  A rung/seed with no artefact is a DECLARED NOT-RUN carrying the
path that is missing -- never a zero, never a blank, never an interpolation.

WHAT THIS TABLE IS FOR
----------------------
The n-way ladder's published verdict -- *the store loses to a same-bytes MLP at 4 of 5
rungs* -- was measured at **K = 5** against an opponent matched to the K = 5 object's own
total.  The Head's ruling is that the main text carries **ONE K, K\\***.  So the verdict is
re-measured at K* here, and three columns are printed side by side:

  `ladder K=5`   the w12 ladder's own `single_design_K5` (a `DowndatableStore` on the
                 LABEL-MI `factor` partition) vs its own `learned_1x`
  `bridge K=5`   the CROSSOVER hub's construction (`CovContentStore` on the LABEL-FREE
                 `ward_balanced` cut) at K = 5 vs a `learned_1x` matched to ITS total
  `K* = 32`      the same crossover construction at K* vs a `learned_1x` matched to ITS
                 total

⭐ `K* vs bridge` isolates the effect of **K**.  `bridge vs ladder K=5` isolates the effect
of the **construction**.  ⛔ Without the bridge column a sign change could not be
attributed to either, and this hub would have replaced one confounded number with another.

⛔ AND THE AXIS IS LABELLED HONESTLY.  `spoke/seeds-and-tables` measured that rungs
n = 3, 4, 5, 6 probe the **IDENTICAL 120 corners** and that what varies across them is how
many rows the cell removed from the fit pool.  The rungs are reported, and the axis is
called what it is: for n >= 3 it is a TRAINING-DATA axis, and only n = 2 vs n = 6 is a
genuine depth contrast.
"""
from __future__ import annotations

import argparse
import json
import statistics as S
from pathlib import Path

LADDER = Path("C:/chlu/outputs/w12-composition-ladder/runs")        # READ ONLY
MY = Path("C:/chlu/outputs/w12b-nr6-and-kstar")
RUNGS = ("2", "3", "4", "5", "6")
SEEDS = (0, 1, 2)
NOT_RUN = "DECLARED NOT-RUN"
#: rungs 3..6 probe the identical corners -- measured by w12 spoke/seeds-and-tables
IDENTICAL_CORNERS = ("3", "4", "5", "6")
CONTROLS = ("landscape_removed", "landscape_shuffled", "random_atoms")


def jload(p: Path):
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:                                        # noqa: BLE001
        return {"__ERROR__": f"{type(exc).__name__}: {exc}"}


def summary(p: Path):
    d = jload(p)
    if d is None or "summary" not in d:
        return None
    return d["summary"]


def leg_kstar(root: Path, rung: str, seed: int, K: int) -> dict:
    """One (rung, seed, K) of THIS hub's replication."""
    leg = root / f"{rung}_seed{seed}_K{K}"
    store = summary(leg / f"rows_single_design_K{K}.json")
    lx = summary(leg / "rows_learned_1x.json")
    meta = jload(leg / "LEG.json") or {}
    ctl = {}
    for c in CONTROLS:
        s = summary(leg / f"rows_single_design_K{K}__{c}.json")
        ctl[c] = (s["distinct_valid_novel_tuples_mean"] if s
                  else NOT_RUN + f": {leg / f'rows_single_design_K{K}__{c}.json'}")
    out = {"dir": str(leg),
           "store": (store["distinct_valid_novel_tuples_mean"] if store
                     else NOT_RUN + f": {leg / f'rows_single_design_K{K}.json'}"),
           "learned_1x": (lx["distinct_valid_novel_tuples_mean"] if lx
                          else NOT_RUN + f": {leg / 'rows_learned_1x.json'}"),
           "learned_1x_params": lx["param_count"] if lx else NOT_RUN,
           "learned_1x_target_params": lx["target_params"] if lx else NOT_RUN,
           "one_x_target": meta.get("one_x_target", NOT_RUN),
           "ledger_four_part": meta.get("ledger_four_part", NOT_RUN),
           "rank_over_width_max": (meta.get("store_info") or {}).get(
               "rank_over_width_max", NOT_RUN),
           "n_corners": meta.get("n_corners", NOT_RUN),
           "controls": ctl,
           "controls_all_collapse": all(isinstance(v, float) and v == 0.0
                                        for v in ctl.values())}
    if isinstance(out["store"], float) and isinstance(out["learned_1x"], float):
        out["margin"] = out["store"] - out["learned_1x"]
    else:
        out["margin"] = NOT_RUN + ": one of the two arms is missing"
    return out


def leg_ladder(rung: str, seed: int) -> dict:
    """One (rung, seed) of the w12 ladder, at K = 5, read from ITS artefacts."""
    leg = LADDER / f"{rung}_seed{seed}"
    store = summary(leg / "rows_single_design_K5.json")
    lx = summary(leg / "rows_learned_1x.json")
    tgt = jload(leg / "ONEX_TARGET.json") or {}
    out = {"dir": str(leg),
           "store": (store["distinct_valid_novel_tuples_mean"] if store else NOT_RUN),
           "learned_1x": (lx["distinct_valid_novel_tuples_mean"] if lx else NOT_RUN),
           "learned_1x_params": lx["param_count"] if lx else NOT_RUN,
           "learned_1x_target_params": lx["target_params"] if lx else NOT_RUN,
           "one_x_total_bytes": tgt.get("total_bytes", NOT_RUN),
           "one_x_composition": ("DowndatableStore deployed atoms + FULL-SQUARE float64 "
                                 "statistics + address book; ⛔ NOT the crossover hub's "
                                 "four-part convention and not comparable to it line "
                                 "for line")}
    if isinstance(out["store"], float) and isinstance(out["learned_1x"], float):
        out["margin"] = out["store"] - out["learned_1x"]
    else:
        out["margin"] = NOT_RUN
    return out


def agg(vals):
    v = [x for x in vals if isinstance(x, float)]
    if not v:
        return {"n": 0, "mean": NOT_RUN, "sd": NOT_RUN}
    return {"n": len(v), "mean": float(S.fmean(v)),
            "sd": float(S.stdev(v)) if len(v) > 1 else None,
            "per_seed": v}


def fmt(x, nd=4):
    return f"{x:.{nd}f}" if isinstance(x, float) else "NOT-RUN"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(MY / "kstar_ladder"))
    ap.add_argument("--out", default=str(MY))
    a = ap.parse_args(argv)
    root, out = Path(a.root), Path(a.out)

    rep = {"SCHEMA": "chlu-w12b-kstar-ladder/1",
           "REPLICATION_NOT_SCORED": True,
           "hub": "w12b hub/nr6-and-kstar",
           "axis_label": ("⛔ rungs n = 3, 4, 5 and 6 probe the IDENTICAL 120 corners "
                          "(w12 spoke/seeds-and-tables). For n >= 3 this is a "
                          "TRAINING-DATA axis, not a composition-depth axis; only "
                          "n = 2 vs n = 6 is a genuine depth contrast. The rungs are "
                          "reported and the axis is named for what it is."),
           "identical_corner_rungs": list(IDENTICAL_CORNERS),
           "legs": {}, "by_rung": {}, "verdict": {}}

    for rung in RUNGS:
        rr = {"ladder_K5": {}, "bridge_K5": {}, "kstar_K32": {}}
        for s in SEEDS:
            rr["ladder_K5"][str(s)] = leg_ladder(rung, s)
            rr["bridge_K5"][str(s)] = leg_kstar(root, rung, s, 5)
            rr["kstar_K32"][str(s)] = leg_kstar(root, rung, s, 32)
        rep["legs"][rung] = rr
        rep["by_rung"][rung] = {
            col: {"store": agg([rr[col][str(s)]["store"] for s in SEEDS]),
                  "learned_1x": agg([rr[col][str(s)]["learned_1x"] for s in SEEDS]),
                  "margin": agg([rr[col][str(s)]["margin"] for s in SEEDS]),
                  "learned_1x_params": agg(
                      [float(rr[col][str(s)]["learned_1x_params"])
                       if isinstance(rr[col][str(s)]["learned_1x_params"], (int, float))
                       else NOT_RUN for s in SEEDS]),
                  "controls_all_collapse_every_seed": all(
                      rr[col][str(s)].get("controls_all_collapse", False)
                      for s in SEEDS) if col != "ladder_K5" else "see the w12 ladder",
                  }
            for col in ("ladder_K5", "bridge_K5", "kstar_K32")}

    # ---- the verdict, and whether the sign flipped -------------------------------
    for col in ("ladder_K5", "bridge_K5", "kstar_K32"):
        wins, losses, unrun = [], [], []
        for rung in RUNGS:
            m = rep["by_rung"][rung][col]["margin"]["mean"]
            if not isinstance(m, float):
                unrun.append(rung)
            elif m > 0:
                wins.append(rung)
            else:
                losses.append(rung)
        rep["verdict"][col] = {
            "store_wins_at_rungs": wins, "store_loses_at_rungs": losses,
            "DECLARED_NOT_RUN_rungs": unrun,
            "margins_per_rung": {r: rep["by_rung"][r][col]["margin"]["mean"]
                                 for r in RUNGS},
            "seeds_per_rung": {r: rep["by_rung"][r][col]["margin"]["n"] for r in RUNGS},
        }
    v5, vk = rep["verdict"]["ladder_K5"], rep["verdict"]["kstar_K32"]
    flipped = [r for r in RUNGS
               if isinstance(v5["margins_per_rung"][r], float)
               and isinstance(vk["margins_per_rung"][r], float)
               and (v5["margins_per_rung"][r] > 0) != (vk["margins_per_rung"][r] > 0)]
    rep["SIGN_FLIPPED_AT_RUNGS"] = flipped
    rep["SIGN_FLIP_READING"] = (
        "⭐⭐ THE K* REPLICATION FLIPS THE SIGN at rungs %s. Under the Head's ruling the "
        "main text carries K*, so this is a CORRECTION to the K = 5 verdict and must be "
        "reported as one, not buried." % flipped if flipped else
        "the sign does NOT flip at any rung for which both K's have a number; the K = 5 "
        "verdict carries to K*, and it is reported as measured rather than assumed.")

    out.mkdir(parents=True, exist_ok=True)
    (out / "KSTAR-LADDER.json").write_text(json.dumps(rep, indent=1, default=str),
                                           encoding="utf-8")

    L = ["# The K* replication of the composition ladder\n",
         "⛔ **A REPLICATION. FREE. No ledger row, no scored budget.**\n",
         "⛔ **AXIS LABEL.** " + rep["axis_label"] + "\n",
         "## The three columns\n",
         "| rung | n seeds | `ladder K=5` store − `learned_1x` | `bridge K=5` "
         "store − `learned_1x` | `K* = 32` store − `learned_1x` |",
         "|---|---|---|---|---|"]
    for rung in RUNGS:
        b = rep["by_rung"][rung]
        L.append("| n=%s%s | %d/%d/%d | %s | %s | %s |"
                 % (rung, " ⚠same corners" if rung in IDENTICAL_CORNERS else "",
                    b["ladder_K5"]["margin"]["n"], b["bridge_K5"]["margin"]["n"],
                    b["kstar_K32"]["margin"]["n"],
                    fmt(b["ladder_K5"]["margin"]["mean"]),
                    fmt(b["bridge_K5"]["margin"]["mean"]),
                    fmt(b["kstar_K32"]["margin"]["mean"])))
    L.append("\n## The opponent's size — the thing that changed\n")
    L.append("| rung | `ladder K=5` params | `bridge K=5` params | `K* = 32` params |")
    L.append("|---|---|---|---|")
    for rung in RUNGS:
        b = rep["by_rung"][rung]
        def p(c):
            m = b[c]["learned_1x_params"]["mean"]
            return f"{m:,.0f}" if isinstance(m, float) else "NOT-RUN"
        L.append(f"| n={rung} | {p('ladder_K5')} | {p('bridge_K5')} | {p('kstar_K32')} |")
    L.append("\n## The control gate at K* — checked BEFORE any margin\n")
    L.append("| rung | seed | " + " | ".join(f"`{c}`" for c in CONTROLS)
             + " | all collapse? |")
    L.append("|---|---|" + "---|" * (len(CONTROLS) + 1))
    for rung in RUNGS:
        for s in SEEDS:
            lg = rep["legs"][rung]["kstar_K32"][str(s)]
            L.append("| n=%s | %d | %s | **%s** |"
                     % (rung, s,
                        " | ".join(fmt(lg["controls"][c]) for c in CONTROLS),
                        "YES" if lg["controls_all_collapse"] else "⛔ NO / NOT-RUN"))
    L.append("\n## Verdict\n```\n"
             + json.dumps({"SIGN_FLIPPED_AT_RUNGS": rep["SIGN_FLIPPED_AT_RUNGS"],
                           "reading": rep["SIGN_FLIP_READING"],
                           "verdict": rep["verdict"]}, indent=1, default=str)
             + "\n```")
    (out / "KSTAR-LADDER.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\nwrote {out / 'KSTAR-LADDER.json'} and {out / 'KSTAR-LADDER.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
