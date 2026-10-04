"""hub/store-content — REPLICATION CHECK (charter §A.5).

Re-runs the hub's headline comparison from ``PREREG-hub-store-content-1.md`` in a
FRESH process with nothing carried over, writes it to its own tag, and diffs it
numerically against the scored run.  ⛔ It does not re-use a single cached result:
the store is refit, the address book is refit, the read is redone, the decoder and
judge are reloaded and re-hashed.

⭐ What it proves and what it does not.  It proves the pipeline is deterministic from
its pinned seeds and that no in-process state leaked between arms — which is the
failure §A.5 exists to catch.  ⛔ It does NOT prove independence of implementation:
the same code runs both times.  That limitation is stated in LOG.md rather than
papered over.
"""
from __future__ import annotations

import json, os, subprocess, sys
from pathlib import Path

OUT = Path(os.environ.get("CHLU_HUB_OUTPUTS", "C:/chlu/outputs")) / "hub-store-content"
#: the two arms the headline verdict rests on, plus the like-for-like
ARMS = ("P", "WIDEN", "basis_head")
TOL = 1e-12


def main(argv=None) -> int:
    ref_tag, rep_tag = "scored_corners_seed0", "replication_corners_seed0"
    cmd = [sys.executable, "-u", str(OUT / "code" / "run_content.py"),
           "--specs", ",".join(ARMS), "--set", "corners", "--tag", rep_tag,
           "--prereg", str(OUT / "PREREG-hub-store-content-1.md"), "--force"]
    print("running:", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True,
                       env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    if r.returncode != 0:
        print(r.stdout[-3000:]); print(r.stderr[-3000:])
        return r.returncode

    report, ok = {}, True
    for a in ARMS:
        ref = json.load(open(OUT / "runs" / ref_tag / f"{a}.json"))
        rep = json.load(open(OUT / "runs" / rep_tag / f"{a}.json"))
        per = {}
        for row in ref["rows"]:
            if row not in rep["rows"]:
                per[row] = {"status": "MISSING in replication"}
                ok = False
                continue
            A, B = ref["rows"][row], rep["rows"][row]
            d = {"aggregate": abs(A["aggregate"] - B["aggregate"]),
                 "exact_match": abs(A["exact_match"] - B["exact_match"])}
            for f in A["per_factor"]:
                d[f] = abs(A["per_factor"][f] - B["per_factor"][f])
            worst = max(d.values())
            per[row] = {"max_abs_diff": worst, "identical": bool(worst <= TOL)}
            ok &= per[row]["identical"]
        lref, lrep = ref["byte_ledger"]["deployed_bytes"], rep["byte_ledger"]["deployed_bytes"]
        gref = ref.get("grouping", {}).get("well_sizes")
        grep = rep.get("grouping", {}).get("well_sizes")
        report[a] = {"rows": per, "ledger_ref": lref, "ledger_rep": lrep,
                     "ledger_identical": lref == lrep,
                     "well_sizes_ref": gref, "well_sizes_rep": grep,
                     "well_sizes_identical": gref == grep}
        ok &= (lref == lrep) and (gref == grep)
        print(f"  {a}: ledger {lref == lrep} wells {gref == grep} rows "
              f"{ {k: v.get('max_abs_diff') for k, v in per.items()} }")

    report["replicated"] = bool(ok)
    report["tolerance"] = TOL
    report["note"] = ("Fresh process, refit from the PREREG. Proves determinism and "
                      "no cross-arm state leakage. Does NOT prove implementation "
                      "independence - the same code runs both times.")
    json.dump(report, open(OUT / "runs" / "REPLICATION.json", "w"), indent=1)
    print("REPLICATED" if ok else "⛔ REPLICATION MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
