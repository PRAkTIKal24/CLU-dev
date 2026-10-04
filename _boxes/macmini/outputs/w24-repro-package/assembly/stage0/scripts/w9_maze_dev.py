"""Run one CMAZE-2 pipeline leg and print its table.

    python scripts/w9_maze_dev.py --seed 0 --which val [--ae-steps N] [--quick]

`--which val` is the DEV leg.  `--which test` is a SCORED leg and refuses to run
unless the harness gate file exists and a PREREG file for the run is present.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np

from w9_maze import pipeline

BANKS = "C:/chlu/outputs/w9-maze/banks2"
OUT = "C:/chlu/outputs/w9-maze/runs"
GATE = "C:/chlu/outputs/w9-harness/HARNESS-READY.json"


POOL = "C:/chlu/outputs/w9-maze/pool"


def load_pool(max_sites=None):
    """The encoder pool: mazes from a seed range no evaluation bank touches."""
    import glob
    out = []
    for p in sorted(glob.glob(os.path.join(POOL, "pool_shard*.pkl"))):
        with open(p, "rb") as fh:
            out.extend(pickle.load(fh)["sites"])
    return out if max_sites is None else out[:max_sites]


def load_bank(seed, n_sites=None):
    with open(os.path.join(BANKS, "bank_seed%d.pkl" % seed), "rb") as fh:
        bank = pickle.load(fh)["sites"]
    return bank if n_sites is None else bank[:n_sites]


def fmt_table(res):
    lines = []
    env = res.get("env_columns", {})
    blind = env.get("unseen_pair", {}).get("blind", {})
    seq = None
    for k, v in blind.items():
        if k.startswith("BLIND-SEQUENTIAL"):
            seq = v
    frag = env.get("fragment", {})
    lines.append("")
    lines.append("seed %d  leg %s  mazes %d  train requests %d  %.0fs"
                 % (res["seed"], res["which"], res["n_mazes"],
                    res["n_train_requests"], res["seconds"]))
    led = res["ledger"]
    lines.append("  ledger: read floats %d (atoms %d + address book %d) = %d B deployed;"
                 " float64 statistic %d B  [K=%d A=%d D=%d design width %d]"
                 % (led["read_floats"], led["store_atom_floats"], led["head_floats"],
                    led["deployed_bytes"], led["statistic_bytes"], led["K"],
                    led["n_atoms"], led["dim"], led["design_width"]))
    if "baseline_params" in res:
        lines.append("  matched parameters: " + "  ".join(
            "%s %d(%s)" % (k, v["params"], "OK" if v["matched"] else "OFF")
            for k, v in res["baseline_params"].items()))
    lines.append("  CEILINGS  representation %s"
                 % {r: round(v["validity"], 4)
                    for r, v in res["representation_ceiling"].items()})
    lines.append("            pipeline(train) %.4f   pipeline(rows) %s"
                 % (res["pipeline_ceiling_train"]["validity"],
                    {r: round(v["validity"], 4)
                     for r, v in res["pipeline_ceiling_rows"].items()}))
    if "span_ceiling_rows" in res:
        lines.append("            store-span(rows) %s   [encoder items %s]"
                     % ({r: round(v["validity"], 4)
                         for r, v in res["span_ceiling_rows"].items()},
                        res.get("ae_train_items")))
    if seq:
        lines.append("  ENV COLUMNS on THIS split (unseen_pair rows): "
                     "BLIND-SEQUENTIAL validity %.4f score %.4f | bends %.3f | "
                     "FRAG r1 %.4f r2 %.4f | published CMAZE-1: 0.4980 / 0.0000 / 0.816"
                     % (seq["validity"], seq["score"],
                        env["unseen_pair"]["bend"]["bend_fraction"],
                        frag.get("FRAG-r1", {}).get("score", float("nan")),
                        frag.get("FRAG-r2", {}).get("score", float("nan"))))
    for row, cols in res["table"].items():
        lines.append("  ROW %-14s %-20s %8s %8s %8s %8s %6s"
                     % (row, "arm", "valid", "score", "vld|bend", "vld|no", "n"))
        for name, v in cols.items():
            lines.append("  %-18s %-20s %8.4f %8.4f %8.4f %8.4f %6d"
                         % ("", name, v["validity"], v["score"],
                            v["bend"]["validity"], v["nobend"]["validity"], v["n"]))
    return "\n".join(lines)


def fmt_display(res):
    d = res.get("display")
    if not d:
        return ""
    L = ["", "  DISPLAY ROWS -- inductive bias.  NEVER a RESULT row.",
         "  %-22s %8s %8s %8s %6s" % ("refusal condition", "refusal", "valid",
                                      "sup.ref", "n")]
    sup = d.get("supervised_refuser", {}).get("refusal_rate", {})
    for k, v in d["refusal"].items():
        L.append("  %-22s %8.4f %8.4f %8s %6d"
                 % (k, v.get("refusal", float("nan")), v.get("validity", float("nan")),
                    ("%.4f" % sup[k]) if sup.get(k) is not None else "-",
                    v.get("n", 0)))
    ml = d["delete_move_left"]
    L.append("  delete move-left (INTERFACE deletion, no atom touched):")
    L.append("    refusal before %.4f | after, routes needing it %.4f (n=%d) | "
             "after, routes not needing it %.4f (n=%d), validity %.4f"
             % (ml["before"]["refusal"], ml["after_needs_W"]["refusal"],
                ml["after_needs_W"]["n"], ml["after_no_W"]["refusal"],
                ml["after_no_W"]["n"], ml["after_no_W"]["validity"]))
    g = d["delete_green"]
    e = d["delete_green_exactness"]
    L.append("  delete the green target (CONTENT deletion, exact rank-m downdate):")
    L.append("    downdate vs refit: statistic %.3e, subspace projector %.3e; "
             "rows removed %d, refit %s, passes %d, items kept %s"
             % (e["statistic_max_rel_residual"], e["subspace_max_projector_residual"],
                e["n_rows_removed"], e["receipt"]["refit"],
                e["receipt"]["passes_over_retained_data"], e["receipt"]["items_kept"]))
    L.append("    green requests: refusal %.4f -> %.4f, validity %.4f -> %.4f"
             % (g["green_before"]["refusal"], g["green_after"]["refusal"],
                g["green_before"]["validity"], g["green_after"]["validity"]))
    L.append("    OTHER targets, no retraining: validity %.4f -> %.4f (delta %+.4f), "
             "score %.4f -> %.4f (delta %+.4f)"
             % (g["other_before"]["validity"], g["other_after"]["validity"],
                g["composition_without_retraining"]["validity_delta_on_other_targets"],
                g["other_before"]["score"], g["other_after"]["score"],
                g["composition_without_retraining"]["score_delta_on_other_targets"]))
    return chr(10).join(L)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--which", default="val")
    ap.add_argument("--n-sites", type=int, default=None)
    ap.add_argument("--ae-steps", type=int, default=6000)
    ap.add_argument("--bl-steps", type=int, default=4000)
    ap.add_argument("--n-atoms", type=int, default=8)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--ridge", type=float, default=1.0)
    ap.add_argument("--tag", default="dev")
    ap.add_argument("--no-baselines", action="store_true")
    ap.add_argument("--display", action="store_true")
    ap.add_argument("--exact-head-delete", action="store_true")
    ap.add_argument("--pool-sites", type=int, default=None)
    ap.add_argument("--no-pool", action="store_true")
    ap.add_argument("--ae-cache", default=None)
    ap.add_argument("--lowrank-atoms", type=int, default=None)
    ap.add_argument("--prereg", default=None)
    a = ap.parse_args(argv)

    if a.which == "test":
        if not os.path.exists(GATE):
            sys.exit("SCORED leg refused: harness gate %s absent" % GATE)
        if a.prereg is None or not os.path.exists(a.prereg):
            sys.exit("SCORED leg refused: no PREREG file")
    os.makedirs(OUT, exist_ok=True)
    bank = load_bank(a.seed, a.n_sites)
    t0 = time.time()
    pool = None if a.no_pool else load_pool(a.pool_sites)
    cache = a.ae_cache
    if cache is None:
        os.makedirs(os.path.join(OUT, "ae"), exist_ok=True)
        cache = os.path.join(OUT, "ae", "ae_seed%d_steps%d%s.eqx"
                             % (a.seed, a.ae_steps,
                                "" if a.n_sites is None else "_n%d" % a.n_sites))
    res, art = pipeline.run(bank, a.seed, pool_sites=pool, ae_cache=cache,
                            lowrank_atoms=a.lowrank_atoms, n_atoms=a.n_atoms, K=a.K, ridge=a.ridge,
                             ae_steps=a.ae_steps, bl_steps=a.bl_steps, which=a.which,
                             do_baselines=not a.no_baselines)
    if a.display:
        from w9_maze import display_run
        res["display"] = display_run.run(bank, art, res, seed=a.seed,
                                         exact_head_delete=a.exact_head_delete)
    res["tag"] = a.tag
    res["prereg"] = a.prereg
    name = "%s_seed%d_%s.json" % (a.tag, a.seed, a.which)
    pipeline.to_json(res, os.path.join(OUT, name))
    txt = fmt_table(res) + fmt_display(res)
    print(txt)
    with open(os.path.join(OUT, name.replace(".json", ".txt")), "w") as fh:
        fh.write(txt + "\n")
    print("\nwrote %s (%.0fs)" % (os.path.join(OUT, name), time.time() - t0))


if __name__ == "__main__":
    main()
