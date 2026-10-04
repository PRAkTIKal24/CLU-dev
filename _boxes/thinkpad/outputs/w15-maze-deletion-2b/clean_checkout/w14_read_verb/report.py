"""Print the arm x verb tables from one or more result files, and check the bars.

    python -m w14_read_verb.report <run.json> [<run.json> ...]

Nothing is computed here that the leg did not already write: this module reads the
result JSONs and lays them out.  Bar verdicts are recomputed from the same numbers so
the verdict and the number can never drift apart.
"""

from __future__ import annotations

import json
import sys

ARMS = ("clu", "clu_lowrank", "clu_nopair", "landscape_removed", "landscape_shuffled",
        "random_atoms", "lookup_matched", "lookup_full", "mlp", "lstm", "transformer")
VERBS = ("FULL", "AVG", "MERGE_target_wins", "MERGE_agent_wins", "MERGE_agent_only",
         "MERGE_target_only")
SHORT = {"FULL": "FULL", "AVG": "AVG", "MERGE_target_wins": "MERGE",
         "MERGE_agent_wins": "M:agent", "MERGE_agent_only": "A only",
         "MERGE_target_only": "B only"}


def load(paths):
    out = []
    for p in paths:
        with open(p) as fh:
            out.append(json.load(fh))
    return out


def val(res, row, arm, verb, key="validity"):
    try:
        return res["table"][row][arm][verb][key]
    except KeyError:
        return None


def header(res):
    led = res["ledger"]
    print("\n%s  seed %d  %s  grid %s %s  mazes %d  train %d  %.0fs"
          % (res.get("tag", "-"), res["seed"], res["which"], res.get("grid_name", "-"),
             res["budgets"], res["n_mazes"], res["n_train_requests"], res["seconds"]))
    print("  rows %s" % res["n_rows"])
    print("  ledger: read floats %d (atoms %d + address book %d) = %d B deployed; "
          "rank/width %.4f; tolerance_used %s"
          % (led["read_floats"], led["store_atom_floats"], led["head_floats"],
             led["deployed_bytes"], led["rank_over_width"][0],
             res.get("tolerance_used")))
    print("  pipeline ceiling: train %.4f  rows %s"
          % (res["pipeline_ceiling_train"]["validity"],
             {r: round(v["validity"], 4)
              for r, v in res["pipeline_ceiling_rows"].items()}))
    print("  span ceiling (true latent on the atoms): %s"
          % {r: round(v["validity"], 4) for r, v in res["span_ceiling_rows"].items()})
    env = res.get("env_columns", {})
    for row in ("unseen_pair", "nway"):
        b = env.get(row, {}).get("blind", {})
        if b:
            cells = ["%s %.4f" % (k, v["validity"]) for k, v in sorted(b.items())]
            print("  env %-12s (THIS HUB'S OWN SPLIT, CMAZE-2b): %s"
                  % (row, "  ".join(cells)))
            bd = env.get(row, {}).get("bend", {})
            if bd:
                print("       bend stats: %s" % json.dumps(bd, default=float)[:200])


def table(res, row):
    if row not in res.get("table", {}):
        return
    print("\n  ROW %s  (n = %d)" % (row.upper(), res["n_rows"][row]))
    print("  %-20s %s   %s" % ("arm", "  ".join("%8s" % SHORT[v] for v in VERBS),
                               "MERGE-AVG"))
    for arm in ARMS:
        if arm not in res["table"][row]:
            continue
        cells = []
        for v in VERBS:
            x = val(res, row, arm, v)
            cells.append("%8.4f" % x if x is not None else "%8s" % "-")
        m = val(res, row, arm, "MERGE_target_wins")
        a = val(res, row, arm, "AVG")
        d = ("%+9.4f" % (m - a)) if (m is not None and a is not None) else " " * 9
        rw = res.get("control_ledgers", {}).get(arm, {}).get("rank_over_width")
        tag = ("  rank/width %.4f" % rw[0]) if rw else ""
        print("  %-20s %s   %s%s" % (arm, "  ".join(cells), d, tag))


def bars(runs, row="unseen_pair"):
    print("\n  === BARS on the PRIMARY row (%s) ===" % row)
    ok1 = ok2 = ok3 = ok5 = True
    for res in runs:
        m = val(res, row, "clu", "MERGE_target_wins")
        a = val(res, row, "clu", "AVG")
        tm = val(res, row, "transformer", "MERGE_target_wins")
        ta = val(res, row, "transformer", "AVG")
        d_clu = (m - a) if None not in (m, a) else None
        d_tr = (tm - ta) if None not in (tm, ta) else None
        b1 = d_clu is not None and d_clu >= 0.10
        b2 = (d_clu is not None and d_tr is not None and d_clu - d_tr >= 0.05)
        cs = val(res, "seen", "clu", "MERGE_target_wins")
        sh = val(res, "seen", "landscape_shuffled", "MERGE_target_wins")
        ra = val(res, "seen", "random_atoms", "MERGE_target_wins")
        b3 = (cs is not None and sh is not None and ra is not None
              and sh <= 0.5 * cs and ra <= 0.5 * cs)
        lk = val(res, row, "lookup_matched", "MERGE_target_wins")
        lr = val(res, row, "landscape_removed", "MERGE_target_wins")
        b5 = (m is not None and lk is not None and lr is not None
              and m > lk and m > lr)
        ok1 &= b1
        ok2 &= b2
        ok3 &= b3
        ok5 &= b5
        f = lambda x: ("%.4f" % x) if x is not None else "--"        # noqa: E731
        print("  seed %d: clu MERGE %s  AVG %s  delta %s | transformer delta %s"
              " | BAR1 %s BAR2 %s BAR3 %s BAR5 %s"
              % (res["seed"], f(m), f(a),
                 ("%+.4f" % d_clu) if d_clu is not None else "--",
                 ("%+.4f" % d_tr) if d_tr is not None else "--",
                 "PASS" if b1 else "FAIL", "PASS" if b2 else "FAIL",
                 "PASS" if b3 else "FAIL", "PASS" if b5 else "FAIL"))
        print("       MERGE on seen: clu %s  landscape_shuffled %s  random_atoms %s  "
              "landscape_removed %s"
              % (f(cs), f(sh), f(ra),
                 f(val(res, "seen", "landscape_removed", "MERGE_target_wins"))))
        print("       MERGE on %s: clu %s  lookup_matched %s  landscape_removed %s"
              % (row, f(m), f(lk), f(lr)))
        print("       ALL ARMS' MERGE-AVG on %s: %s" % (row, ", ".join(
            "%s %+.4f" % (aa, val(res, row, aa, "MERGE_target_wins")
                          - val(res, row, aa, "AVG"))
            for aa in ARMS if aa in res["table"].get(row, {}))))
    print("  ALL THREE SEEDS: BAR1 %s  BAR2 %s  BAR3 %s  BAR5 %s"
          % tuple("PASS" if x else "FAIL" for x in (ok1, ok2, ok3, ok5)))


def main(paths):
    runs = load(paths)
    for res in runs:
        header(res)
        for row in ("seen", "unseen_pair", "nway", "budget_interp"):
            table(res, row)
    if runs:
        for row in ("unseen_pair", "nway"):
            bars(runs, row)


if __name__ == "__main__":
    main(sys.argv[1:])
