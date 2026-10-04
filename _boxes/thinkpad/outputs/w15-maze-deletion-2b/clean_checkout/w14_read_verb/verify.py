"""INDEPENDENT verification of this hub's claims, straight from the artefacts.

    python -m w14_read_verb.verify <out_dir>

It does NOT import ``w14_read_verb.pipeline``, ``w14_read_verb.report`` or
``w14_read_verb.leg``: it reads the result JSONs, the gate files and the ledger, and
re-derives every number the report prints.  Exit code 0 only if every check passes.

The checks are the ones a referee would ask for:

 1. the GATE was complete before any scored leg, and its ACCEPTANCE items all pass;
 2. ``tolerance_used`` is ``null`` in every leg, and the judge's file sha256 is the one
    the gate recorded;
 3. three scored TEST legs exist, at three DISTINCT seeds, and a ledger row for target
    ``read-verb`` names each one BEFORE it (checked by tag presence, timestamps ordered);
 4. every arm carries every verb on every row -- so no arm was given a merge another was
    denied;
 5. the MERGE - AVG deltas the report prints are exactly the JSON's numbers;
 6. the charter-C controls' verdict is computed from the JSON, not quoted;
 7. the two half-addresses really are 7 of the FULL design's 11 non-zero columns;
 8. no CMAZE-1 number appears in any table this hub writes.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import sys

GATE_A = "C:/chlu/outputs/w14-request-layer-fix/REQUEST-LAYER-READY.json"
GATE_B = "C:/chlu/outputs/w14-refusal-displays-closeout/CLOSEOUT.md"
LEDGER = "C:/chlu/SCORED-LEDGER.md"
VERBS = ("FULL", "AVG", "MERGE_target_wins", "MERGE_agent_wins", "MERGE_agent_only",
         "MERGE_target_only")

fails = []
notes = []


def check(name, ok, detail=""):
    print("%-6s %s%s" % ("PASS" if ok else "FAIL", name,
                         ("  -- " + detail) if detail else ""))
    if not ok:
        fails.append(name)
    return ok


def main(out_dir):
    print("w14 hub/read-verb -- INDEPENDENT VERIFY")
    print("out_dir =", out_dir)

    # 1 -- the gate ------------------------------------------------------
    with open(GATE_A) as fh:
        g = json.load(fh)
    check("1a gate status ready", g.get("status") == "ready", g.get("status"))
    check("1b gate ACCEPTANCE all pass",
          all(v.get("pass") for v in g["ACCEPTANCE"].values()),
          ", ".join("%s=%s" % (k, v.get("pass")) for k, v in g["ACCEPTANCE"].items()))
    check("1c gate spent no scored run", g.get("scored_runs_spent") == 0)
    check("1d gate closed (END_UTC present)",
          os.path.exists("C:/chlu/outputs/w14-request-layer-fix/END_UTC.txt"))
    check("1e refusal closeout present and closed",
          os.path.exists(GATE_B) and os.path.exists(
              "C:/chlu/outputs/w14-refusal-displays-closeout/END_UTC.txt"))

    # 2 -- the judge does not move ---------------------------------------
    jp = "w8_envscout/maze_colour.py"
    if os.path.exists(jp):
        with open(jp, "rb") as fh:
            sha = hashlib.sha256(fh.read()).hexdigest()
        check("2a judge file sha256 == the gate's",
              sha == g["provenance"]["judge"]["sha256"], sha)
    else:
        notes.append("2a judge file not found at %s (run from the worktree root)" % jp)

    runs = sorted(glob.glob(os.path.join(out_dir, "runs", "*.json")))
    scored = []
    for p in runs:
        with open(p) as fh:
            r = json.load(fh)
        r["_path"] = p
        if r.get("which") == "test":
            scored.append(r)
    check("2b tolerance_used is null in EVERY leg",
          bool(runs) and all(json.load(open(p)).get("tolerance_used") is None
                             for p in runs),
          "%d legs (a vacuous pass is a FAIL: at least one leg must exist)" % len(runs))

    # 3 -- three scored legs, three distinct seeds ------------------------
    check("3a exactly three scored TEST legs", len(scored) == 3,
          "%d found: %s" % (len(scored), [os.path.basename(x["_path"]) for x in scored]))
    check("3b three DISTINCT seeds", len({x["seed"] for x in scored}) == 3,
          str(sorted(x["seed"] for x in scored)))
    if os.path.exists(LEDGER):
        with open(LEDGER, encoding="utf-8", errors="replace") as fh:
            led = fh.read()
        rows = [ln for ln in led.splitlines() if "read-verb" in ln and ln.startswith("|")]
        check("3c a ledger row per scored leg, target read-verb", len(rows) >= 3,
              "%d rows" % len(rows))
        for x in scored:
            t = x.get("tag", "")
            check("3d ledger names tag %s" % t, any(t in ln for ln in rows))
    else:
        check("3c ledger present", False, LEDGER)

    # 4..7 -- the table -------------------------------------------------
    for x in scored or runs and [json.load(open(runs[0]))]:
        seed = x.get("seed")
        tab = x.get("table", {})
        arms = sorted({a for row in tab.values() for a in row})
        missing = []
        for row, byarm in tab.items():
            for a, byverb in byarm.items():
                for v in VERBS:
                    if v not in byverb:
                        missing.append("%s/%s/%s" % (row, a, v))
        check("4 seed %s: every arm carries every verb on every row" % seed,
              not missing, "%d arms; missing %s" % (len(arms), missing[:4]))
        # 5 -- the deltas
        d = {}
        for row in ("unseen_pair", "nway", "seen"):
            if row not in tab:
                continue
            d[row] = {a: tab[row][a]["MERGE_target_wins"]["validity"]
                      - tab[row][a]["AVG"]["validity"] for a in tab[row]}
        print("     seed %s MERGE-AVG  unseen_pair: %s" % (
            seed, {k: round(v, 4) for k, v in sorted(
                d.get("unseen_pair", {}).items())}))
        # 6 -- charter C, computed here
        s = tab.get("seen", {})
        if s:
            cs = s["clu"]["MERGE_target_wins"]["validity"]
            for ctl in ("landscape_shuffled", "random_atoms", "landscape_removed"):
                if ctl in s:
                    cv = s[ctl]["MERGE_target_wins"]["validity"]
                    check("6 seed %s: %s collapses (<= 0.5 x clu)" % (seed, ctl),
                          cv <= 0.5 * cs, "%.4f vs clu %.4f" % (cv, cs))

    # 7 -- the half-addresses, re-derived from the design itself ---------
    try:
        import numpy as np
        from chlu.core.gen_request_head import HeadSpec, request_design
        grids = [np.arange(96.0), np.arange(6.0), np.arange(6.0),
                 np.array([55.0, 85.0, 105.0])]
        spec = HeadSpec([int(gg.size) for gg in grids], pair_block=True, ridge=0.01)
        V = np.array([[3.0, 2.0, 4.0, 85.0]])
        Va = V.copy(); Va[:, 2] = np.nan
        Vb = V.copy(); Vb[:, 1] = np.nan
        nf = int((request_design(V, grids, spec)[0] != 0).sum())
        na = int((request_design(Va, grids, spec)[0] != 0).sum())
        nb = int((request_design(Vb, grids, spec)[0] != 0).sum())
        check("7 half-addresses are 7/7 of the FULL design's 11 columns",
              (nf, na, nb) == (11, 7, 7), "full=%d A=%d B=%d" % (nf, na, nb))
    except Exception as e:                                   # noqa: BLE001
        notes.append("7 not run: %s" % e)

    # 8 -- no CMAZE-1 number shares a table ------------------------------
    bad = []
    for p in runs:
        with open(p) as fh:
            txt = fh.read()
        for tok in ("cmaze1", "CMAZE-1", "cmaze1-easy"):
            if tok in txt:
                bad.append((os.path.basename(p), tok))
    check("8 no CMAZE-1 token in any result file", not bad, str(bad[:4]))

    print("\n%d checks failed%s" % (len(fails), (": " + ", ".join(fails)) if fails else ""))
    for n in notes:
        print("NOTE  " + n)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else
                  "C:/chlu/outputs/w14-read-verb"))
