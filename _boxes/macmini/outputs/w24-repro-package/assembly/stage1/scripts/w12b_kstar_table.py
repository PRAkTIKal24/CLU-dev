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


def ceiling_of(p: Path):
    """The per-corner CEILING = max requests in any corner of the rung.

    ⛔ This is a property of Shapes3D and of the rung's corner construction, NOT of any
    arm: `w12_ladder_table.py:134` takes it as `max(n_requests)` over the corners, and the
    same expression is used here so a normalised number of this hub's is comparable, cell
    for cell, with the ladder's published `frac_of_ceiling` column.
    """
    d = jload(p)
    if d is None or "per_corner" not in d:
        return None
    try:
        return int(max(int(r["n_requests"]) for r in d["per_corner"]))
    except Exception:                                               # noqa: BLE001
        return None


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
    ceil = ceiling_of(leg / f"rows_single_design_K{K}.json")
    out = {"dir": str(leg), "ceiling_per_corner": ceil if ceil else NOT_RUN,
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
           # ⚠ TWO DIFFERENT THINGS, KEPT APART ON PURPOSE.
           # `controls_exactly_zero` is the OBSERVATION the crossover hub reported at K*
           # on val ("all four read EXACTLY 0.0000"). It is not a bar and was never
           # declared as one.
           # `controls_collapse_beyond_declared_margin` is the PRE-DECLARED BAR:
           # `LADDER-READY.json:estimand` pins `DECORATION_MARGIN_NOVELTY = 1.0`, so a
           # control collapses when the arm beats it by at least one novel tuple per
           # corner. ⛔ The bar is not moved to accommodate a non-zero reading, and a
           # non-zero reading is not rounded away either -- both are printed.
           "controls_exactly_zero": all(isinstance(v, float) and v == 0.0
                                        for v in ctl.values())}
    if isinstance(store and store.get("distinct_valid_novel_tuples_mean"), float):
        arm_v = float(store["distinct_valid_novel_tuples_mean"])
        out["controls_collapse_beyond_declared_margin"] = all(
            isinstance(v, float) and (arm_v - v) >= 1.0 for v in ctl.values())
        out["control_margins"] = {c: (arm_v - v) if isinstance(v, float) else NOT_RUN
                                  for c, v in ctl.items()}
    else:
        out["controls_collapse_beyond_declared_margin"] = False
        out["control_margins"] = NOT_RUN + ": the arm itself has no number"
    out["controls_all_collapse"] = out["controls_collapse_beyond_declared_margin"]
    if isinstance(out["store"], float) and isinstance(out["learned_1x"], float):
        out["margin"] = out["store"] - out["learned_1x"]
        if ceil:
            out["store_frac_of_ceiling"] = out["store"] / ceil
            out["learned_1x_frac_of_ceiling"] = out["learned_1x"] / ceil
            out["margin_frac_of_ceiling"] = out["margin"] / ceil
    else:
        out["margin"] = NOT_RUN + ": one of the two arms is missing"
    return out


def leg_ladder(rung: str, seed: int) -> dict:
    """One (rung, seed) of the w12 ladder, at K = 5, read from ITS artefacts."""
    leg = LADDER / f"{rung}_seed{seed}"
    store = summary(leg / "rows_single_design_K5.json")
    lx = summary(leg / "rows_learned_1x.json")
    tgt = jload(leg / "ONEX_TARGET.json") or {}
    ceil = ceiling_of(leg / "rows_single_design_K5.json")
    out = {"dir": str(leg), "ceiling_per_corner": ceil if ceil else NOT_RUN,
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
        if ceil:
            out["store_frac_of_ceiling"] = out["store"] / ceil
            out["learned_1x_frac_of_ceiling"] = out["learned_1x"] / ceil
            out["margin_frac_of_ceiling"] = out["margin"] / ceil
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
                  "margin_frac_of_ceiling": agg(
                      [rr[col][str(s)].get("margin_frac_of_ceiling", NOT_RUN)
                       for s in SEEDS]),
                  "store_frac_of_ceiling": agg(
                      [rr[col][str(s)].get("store_frac_of_ceiling", NOT_RUN)
                       for s in SEEDS]),
                  "learned_1x_frac_of_ceiling": agg(
                      [rr[col][str(s)].get("learned_1x_frac_of_ceiling", NOT_RUN)
                       for s in SEEDS]),
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
    # ⛔ SEED-MATCHED, because the two columns need not carry the same seeds. Comparing a
    # 3-seed mean against a 1-seed mean would charge a SEED difference to K, which is the
    # same class of mistake as charging a CONSTRUCTION difference to K -- the mistake the
    # bridge column exists to prevent. Only seeds that have BOTH numbers are used.
    matched = {}
    for rung in RUNGS:
        rr = rep["legs"][rung]
        common = [str(x) for x in SEEDS
                  if isinstance(rr["ladder_K5"][str(x)]["margin"], float)
                  and isinstance(rr["kstar_K32"][str(x)]["margin"], float)]
        if not common:
            matched[rung] = {"seeds": [], "status": NOT_RUN
                             + ": no seed carries BOTH a K = 5 and a K* number"}
            continue
        l5 = [rr["ladder_K5"][x]["margin"] for x in common]
        lk = [rr["kstar_K32"][x]["margin"] for x in common]
        matched[rung] = {
            "seeds": [int(x) for x in common], "n_seeds": len(common),
            "ladder_K5_margin_mean": float(S.fmean(l5)),
            "kstar_K32_margin_mean": float(S.fmean(lk)),
            "ladder_K5_margin_per_seed": l5, "kstar_K32_margin_per_seed": lk,
            "delta": float(S.fmean(lk) - S.fmean(l5)),
            "sign_flipped": (S.fmean(l5) > 0) != (S.fmean(lk) > 0),
            "sign_flipped_on_every_matched_seed": all(
                (a > 0) != (b > 0) for a, b in zip(l5, lk)),
            # ⭐ THE LADDER'S OWN PRE-DECLARED BAR, applied unchanged.
            # `AMENDMENT-2 §3` of w12 hub/composition-ladder: the arm clears only if it
            # beats the opponent by `DECORATION_MARGIN_NOVELTY = 1.0` novel tuple per
            # corner on EVERY declared seed -- not on the seed mean. ⛔ The bar is not
            # moved; it is simply evaluated at the other K as well.
            "clears_declared_bar_every_seed_K5": all(v >= 1.0 for v in l5),
            "clears_declared_bar_every_seed_Kstar": all(v >= 1.0 for v in lk),
            "min_margin_K5": float(min(l5)), "min_margin_Kstar": float(min(lk)),
            "store_K5_mean": float(S.fmean(
                [rr["ladder_K5"][x]["store"] for x in common])),
            "store_Kstar_mean": float(S.fmean(
                [rr["kstar_K32"][x]["store"] for x in common])),
            "opponent_K5_params_mean": float(S.fmean(
                [float(rr["ladder_K5"][x]["learned_1x_params"]) for x in common])),
            "opponent_Kstar_params_mean": float(S.fmean(
                [float(rr["kstar_K32"][x]["learned_1x_params"]) for x in common])),
        }
        m_ = matched[rung]
        m_["opponent_shrank_by"] = (m_["opponent_K5_params_mean"]
                                    / max(m_["opponent_Kstar_params_mean"], 1.0))
    rep["seed_matched_K5_vs_Kstar"] = matched

    # ⭐ THE DECOMPOSITION THE BRIDGE BUYS.  Going from the ladder's K = 5 arm to the K*
    # arm changes TWO things, so the total swing is split into the two steps rather than
    # attributed to K wholesale:
    #     ladder K=5  ->  bridge K=5   = the CONSTRUCTION step (partition rule, store
    #                                    class, covariance convention).  K is held at 5.
    #     bridge K=5  ->  K* = 32      = the K step.  The construction is held fixed.
    # ⛔ Only seeds carrying all THREE numbers are used.
    decomp = {}
    for rung in RUNGS:
        rr = rep["legs"][rung]
        common = [str(x) for x in SEEDS
                  if all(isinstance(rr[c][str(x)]["margin"], float)
                         for c in ("ladder_K5", "bridge_K5", "kstar_K32"))]
        if not common:
            decomp[rung] = {"seeds": [], "status": NOT_RUN + ": no seed carries all "
                            "three of ladder K=5, bridge K=5 and K* = 32"}
            continue
        a = float(S.fmean([rr["ladder_K5"][x]["margin"] for x in common]))
        b = float(S.fmean([rr["bridge_K5"][x]["margin"] for x in common]))
        c_ = float(S.fmean([rr["kstar_K32"][x]["margin"] for x in common]))
        tot = c_ - a
        decomp[rung] = {
            "seeds": [int(x) for x in common], "n_seeds": len(common),
            "margin_ladder_K5": a, "margin_bridge_K5": b, "margin_kstar_K32": c_,
            "total_swing": tot,
            "construction_step": b - a, "K_step": c_ - b,
            "construction_share": (b - a) / tot if tot else None,
            "K_share": (c_ - b) / tot if tot else None,
            "store_ladder_K5": float(S.fmean(
                [rr["ladder_K5"][x]["store"] for x in common])),
            "store_bridge_K5": float(S.fmean(
                [rr["bridge_K5"][x]["store"] for x in common])),
            "store_kstar_K32": float(S.fmean(
                [rr["kstar_K32"][x]["store"] for x in common])),
            "opponent_params": {
                "ladder_K5": float(S.fmean(
                    [float(rr["ladder_K5"][x]["learned_1x_params"]) for x in common])),
                "bridge_K5": float(S.fmean(
                    [float(rr["bridge_K5"][x]["learned_1x_params"]) for x in common])),
                "kstar_K32": float(S.fmean(
                    [float(rr["kstar_K32"][x]["learned_1x_params"]) for x in common])),
            },
            "why": ("⛔ without the bridge the WHOLE swing would have been charged to K. "
                    "The construction step is the partition rule, the store class and "
                    "the covariance convention changing together; the K step is the only "
                    "one the Head's ruling is about."),
        }
    rep["margin_swing_decomposed"] = decomp

    v5, vk = rep["verdict"]["ladder_K5"], rep["verdict"]["kstar_K32"]
    flipped = [r for r in RUNGS if matched[r].get("sign_flipped")]
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
    L.append("| rung | seed | arm | " + " | ".join(f"`{c}`" for c in CONTROLS)
             + " | exactly 0? | collapses beyond the declared 1.0 margin? |")
    L.append("|---|---|---|" + "---|" * (len(CONTROLS) + 2))
    for rung in RUNGS:
        for s in SEEDS:
            lg = rep["legs"][rung]["kstar_K32"][str(s)]
            L.append("| n=%s | %d | %s | %s | %s | **%s** |"
                     % (rung, s, fmt(lg["store"], 3),
                        " | ".join(fmt(lg["controls"][c]) for c in CONTROLS),
                        "yes" if lg.get("controls_exactly_zero") else "no",
                        "YES" if lg["controls_all_collapse"] else "⛔ NO / NOT-RUN"))
    L.append("\n⚠ `exactly 0` is the crossover hub's OBSERVATION at K* on val, not a bar. "
             "The PRE-DECLARED bar is `LADDER-READY.json`'s "
             "`DECORATION_MARGIN_NOVELTY = 1.0` novel tuple per corner. ⛔ Neither is "
             "moved to accommodate the other, and a non-zero control reading is printed "
             "rather than rounded away.\n")
    L.append("\n## SEED-MATCHED K = 5 vs K\\* — the only honest comparison\n")
    L.append("| rung | seeds | store K=5 | store K\\* | opp K=5 params | opp K\\* params "
             "| opp shrank | margin K=5 | margin K\\* | sign flipped? |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for rung in RUNGS:
        m_ = matched[rung]
        if not m_.get("seeds"):
            L.append("| n=%s | — | %s | | | | | | | |" % (rung, m_["status"]))
            continue
        L.append("| n=%s | %s | %s | %s | %s | %s | %.2f× | %s | %s | **%s** |"
                 % (rung, m_["seeds"], fmt(m_["store_K5_mean"], 3),
                    fmt(m_["store_Kstar_mean"], 3),
                    format(int(m_["opponent_K5_params_mean"]), ","),
                    format(int(m_["opponent_Kstar_params_mean"]), ","),
                    m_["opponent_shrank_by"],
                    fmt(m_["ladder_K5_margin_mean"], 3),
                    fmt(m_["kstar_K32_margin_mean"], 3),
                    "YES" if m_["sign_flipped"] else "no"))
    L.append("\n### The ladder's OWN pre-declared bar, evaluated at both K\n")
    L.append("⛔ `AMENDMENT-2 §3` of `w12 hub/composition-ladder`: the arm clears only if "
             "it beats the opponent by `DECORATION_MARGIN_NOVELTY = 1.0` novel tuple per "
             "corner on **every** declared seed — not on the seed mean. The bar is not "
             "moved here; it is evaluated at the other K as well.\n")
    L.append("| rung | clears at K=5? | worst seed K=5 | clears at K\\*? | worst seed K\\* |")
    L.append("|---|---|---|---|---|")
    for rung in RUNGS:
        m_ = matched[rung]
        if not m_.get("seeds"):
            L.append("| n=%s | %s | | | |" % (rung, m_["status"]))
            continue
        L.append("| n=%s | %s | %s | **%s** | **%s** |"
                 % (rung,
                    "yes" if m_["clears_declared_bar_every_seed_K5"] else "**no**",
                    fmt(m_["min_margin_K5"], 3),
                    "YES" if m_["clears_declared_bar_every_seed_Kstar"] else "no",
                    fmt(m_["min_margin_Kstar"], 3)))

    L.append("\n## ⭐ THE SWING, DECOMPOSED — what the bridge column buys\n")
    L.append("| rung | seeds | margin: ladder K=5 → bridge K=5 → K\\* | CONSTRUCTION step "
             "| K step | total | K's share |")
    L.append("|---|---|---|---|---|---|---|")
    for rung in RUNGS:
        d_ = decomp[rung]
        if not d_.get("seeds"):
            L.append("| n=%s | — | %s | | | | |" % (rung, d_["status"]))
            continue
        L.append("| n=%s | %s | %s → %s → %s | **%+.3f** | **%+.3f** | %+.3f | %s |"
                 % (rung, d_["seeds"], fmt(d_["margin_ladder_K5"], 3),
                    fmt(d_["margin_bridge_K5"], 3), fmt(d_["margin_kstar_K32"], 3),
                    d_["construction_step"], d_["K_step"], d_["total_swing"],
                    ("%.0f%%" % (100 * d_["K_share"])) if d_["K_share"] is not None
                    else "n/a"))
    L.append("\n⛔ Without the bridge the WHOLE swing would have been charged to K. The "
             "construction step is the partition rule, the store class and the covariance "
             "convention moving together; only the K step is what the Head's ruling is "
             "about.\n")

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
