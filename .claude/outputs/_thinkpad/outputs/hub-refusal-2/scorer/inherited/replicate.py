"""hub/refusal -- INDEPENDENT REPLICATION from the clean worktree checkout (charter A.5).

Charter A.5: "a separate spoke re-runs the hub's winning configuration from its PREREG
file on a clean checkout of the hub's branch.  Unreplicated = not a result."

Run this with ``CHLU_ROOT`` pointing at the WORKTREE (``../CHLU-hub-refusal``) and
``CHLU_CLAUDE_ROOT`` / ``CHLU_GEN_DATA_ROOT`` / ``CHLU_GEN_BRIDGE_ROOT`` /
``CHLU_HUB_OUTPUTS`` pointing at the bundle's frozen data -- a worktree has no
``.claude/`` (it is gitignored), which ``exp_gen_dynamical_read``'s docstring warns about.
The CODE then comes from the clean checkout and the DATA from the frozen bundle.

It asserts ``chlu`` was imported from the worktree, re-runs one unit, and compares every
persisted statistic against the unit already on disk.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import refusal_lib as R  # noqa: E402


def main() -> int:
    import chlu

    where = str(Path(chlu.__file__).resolve()).replace("\\", "/")
    root = str(Path(os.environ["CHLU_ROOT"]).resolve()).replace("\\", "/")
    print(f"chlu imported from : {where}")
    print(f"CHLU_ROOT          : {root}")
    if not where.startswith(root):
        raise AssertionError(
            "chlu was NOT imported from CHLU_ROOT -- the import trap env_hub.sh warns "
            "about. This is not a replication from a clean checkout.")
    print("IMPORT-UNDER-ROOT: OK (the code is the worktree's)")

    ctx = R.build_context(0)
    built = R.build_land(ctx, "A", log=lambda *_: None)
    reqs = R.request_sets(ctx, ["MS"])
    rec = R.settle_unit(ctx, built, "MS", reqs["MS"], arm="full", budget="A",
                        log=lambda *_: None)

    ref_path = R.OUT / "runs" / "seed_0" / "A_MS_full.npz"
    ref = np.load(ref_path, allow_pickle=False)
    got = {
        "S_max": rec["settled"]["S_max"], "S_sum": rec["settled"]["S_sum"],
        "S_mean": rec["settled"]["S_mean"],
        "e_total": rec["settled"]["e_total"], "e_store": rec["settled"]["e_store"],
        "launch_S_max": rec["launch"]["S_max"],
        "rel_tail_drop": rec["conv"]["rel_tail_drop"],
        "x_settled": rec["x_settled"],
    }
    out = {"schema": "hub-refusal/replication/1",
           "rule": "charter A.5 -- clean checkout of the hub branch; unreplicated = not a result",
           "chlu_imported_from": where, "chlu_root": root,
           "reference_unit": str(ref_path), "fields": {}}
    all_ok = True
    for k, b in got.items():
        a = np.asarray(ref[k], np.float64)
        b = np.asarray(b, np.float64)
        same = bool(np.array_equal(a, b))
        d = float(np.max(np.abs(a - b))) if a.size else 0.0
        out["fields"][k] = {"bit_identical": same, "max_abs_delta": d,
                            "n": int(a.size)}
        all_ok &= same
        print(f"  {k:16s} bit-identical={str(same):5s}  max|delta|={d:.3e}  n={a.size}")
    out["experts_reproduction"] = rec["meta"]["experts_reproduction"]
    out["frac_still_descending"] = rec["meta"]["pooled_convergence"]["frac_still_descending"]
    out["ALL_BIT_IDENTICAL"] = bool(all_ok)
    print(f"  experts_reproduction : {out['experts_reproduction']}")
    print(f"  frac_still_descending: {out['frac_still_descending']}")
    print("\nREPLICATION: " + ("BIT-IDENTICAL" if all_ok else "NOT bit-identical -- see deltas"))
    p = R.OUT / "REPLICATION.json"
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
