"""hub/refusal-2 -- INDEPENDENT REPLICATION from the clean worktree (charter A.5).

Charter A.5: "a separate spoke re-runs the hub's winning configuration from its PREREG file
on a clean checkout of the hub's branch.  Unreplicated = not a result."

⭐ This hub has no WINNING configuration -- the go/no-go returned NO-GO.  What must
replicate instead is the NEGATIVE, because a negative result that does not reproduce is
just as worthless as a positive one that does not.  So this re-runs the GO/NO-GO's own
units from the clean checkout and compares every persisted array against the ones on disk.

Run with ``CHLU_ROOT`` pointing at the WORKTREE (``../CHLU-hub-refusal-2``) and the frozen
data roots at the bundle -- a worktree has no ``.claude/`` (it is gitignored).  The CODE is
then the clean checkout's and the DATA is the pin's.  Import-under-root is asserted first.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scorer"))

import numpy as np  # noqa: E402

# ⛔ ORDER MATTERS AND IT BIT ME.  ``refusal_lib`` does ``sys.path.insert(0, CHLU_ROOT)`` at
# import time; the venv also carries an editable ``chlu`` that resolves to the BUNDLE.  If
# ``import chlu`` happens first, it binds the bundle's copy and the A.5 guard (correctly)
# refuses to call that a clean-checkout replication.  These two imports must precede it.
import refusal_lib as R  # noqa: E402
import refusal2_lib as L  # noqa: E402
from synthetic_probe import N_BASES, RNG_SEED, build_sets  # noqa: E402

FIELDS = ("S", "M", "d_over_spread", "S_argmax_read", "d_argmax_read",
          "posterior_conf", "x_read")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", default="A")
    ap.add_argument("--arms", nargs="+", default=["linear", "linear_store_removed"])
    a = ap.parse_args()

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

    from chlu.core.gen_request import requests_from_rows

    ctx = R.build_context(int(a.seed))
    built = R.build_land(ctx, str(a.budget), log=lambda *_: None)
    grid = ctx["grid"]
    rng = np.random.default_rng(RNG_SEED)
    _fit = np.asarray(ctx["fit_rows"], np.int64)
    _take = np.sort(rng.choice(_fit.size, size=min(N_BASES, _fit.size), replace=False))
    sets = build_sets(requests_from_rows(grid, _fit[_take]), grid, rng)
    sets = {"val": requests_from_rows(grid, ctx["val_rows"]), **sets}

    rundir = R.OUT / "runs2" / f"gonogo_seed{a.seed}_{a.budget}"
    out = {"schema": "hub-refusal-2/replication/1",
           "rule": ("charter A.5 -- clean checkout of the hub branch. This hub has no "
                    "winning configuration; what replicates is the NEGATIVE."),
           "chlu_imported_from": where, "CHLU_ROOT": root,
           "import_under_root": True,
           "seed": int(a.seed), "budget": str(a.budget), "units": {}}
    all_ok = True
    for arm in a.arms:
        for name, reqs in sets.items():
            ref_path = rundir / f"{arm}__{name}.npz"
            if not ref_path.exists():
                out["units"][f"{arm}/{name}"] = {"skipped": f"no reference at {ref_path}"}
                continue
            rec = L.linear_unit(ctx, built, f"gng_{name}", reqs, arm=arm,
                                budget=str(a.budget), log=lambda *_: None)
            ref = np.load(ref_path, allow_pickle=False)
            row = {}
            for f in FIELDS:
                g = np.asarray(rec[f])
                r = np.asarray(ref[f])
                row[f] = {
                    "bit_identical": bool(np.array_equal(g, r)),
                    "max_abs_delta": float(np.max(np.abs(g.astype(np.float64)
                                                         - r.astype(np.float64))))
                    if g.size else 0.0,
                    "n": int(g.size),
                }
                all_ok &= row[f]["bit_identical"]
            out["units"][f"{arm}/{name}"] = row
            print(f"  {arm}/{name:<12} " + "  ".join(
                f"{f}:{'OK' if row[f]['bit_identical'] else 'delta=%.3e' % row[f]['max_abs_delta']}"
                for f in FIELDS))
    out["ALL_BIT_IDENTICAL"] = bool(all_ok)
    p = R.OUT / f"REPLICATION2_seed{a.seed}_{a.budget}.json"
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=str)
    print(f"\nALL_BIT_IDENTICAL = {all_ok}")
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
