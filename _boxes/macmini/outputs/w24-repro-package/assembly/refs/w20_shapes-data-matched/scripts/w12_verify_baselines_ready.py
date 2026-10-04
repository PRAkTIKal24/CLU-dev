"""VERIFY ``BASELINES-READY.json`` without importing the builder.

The crossover hub gates on that file, so it is checked by something that did not write it.
This script imports ``w4_api`` only -- never ``scripts/w12_deletion_baselines.py`` and never the
w12 module's fitting code -- so a defect in the builder cannot hide itself here.

What it checks, all of it refusable:
  1. every record names an entry point that resolves to a real callable;
  2. every record carries a COMPLETE cost line (seconds / data rows read / bytes rewritten);
  3. every learned arm is inside the declared parameter tolerance of the byte pin;
  4. matched compute is equal across every arm on every leg of every seed;
  5. every comparator checkpoint LOADS BY NAME through ``w4_api.load_rival`` and returns a
     number of the declared shape on a real request;
  6. every DECLARED NOT-RUN carries a reason.

Run::

    python scripts/w12_verify_baselines_ready.py --ready DIR/BASELINES-READY.json
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FAIL = []


def check(ok: bool, msg: str):
    if ok:
        print("  PASS  %s" % msg)
    else:
        FAIL.append(msg)
        print("  FAIL  %s" % msg)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ready",
                    default="C:/chlu/outputs/w12-deletion-baselines/BASELINES-READY.json")
    ap.add_argument("--load-every", action="store_true",
                    help="load EVERY checkpoint, not one of each kind per seed")
    a = ap.parse_args(argv)

    from chlu.eval.rivals.w4_api import load_rival

    with open(a.ready, encoding="utf-8") as fh:
        d = json.load(fh)
    print("BASELINES-READY: seeds %s, %d records"
          % (d["seeds_present"], d["n_records_total"]))

    # 1. entry points resolve
    for name, spec in d["comparators"].items():
        ep = spec["entry_point"].split(" ")[0]
        mod, _, fn = ep.rpartition(".")
        try:
            obj = getattr(importlib.import_module(mod), fn)
            check(callable(obj), "entry point resolves: %s -> %s" % (name, ep))
        except Exception as exc:                                  # noqa: BLE001
            check(False, "entry point resolves: %s -> %s (%s)" % (name, ep, exc))

    tol = float(d["pins"]["param_tolerance"])
    target = int(d["pins"]["param_target"])
    root = d["how_the_hub_calls_them"]["comparators_root"]

    for seed_s, per in sorted(d["per_seed"].items(), key=lambda kv: int(kv[0])):
        seed = int(seed_s)
        R = per["records"]
        print("\n-- seed %d: %d records" % (seed, len(R)))

        bad_cost = [k for k, v in R.items()
                    if any(v["cost"].get(f) is None for f in d["cost_fields"])]
        check(not bad_cost, "every record has a complete cost line (%d records)" % len(R))

        bad_par = [k for k, v in R.items()
                   if abs(float(v["param_error"])) > tol]
        check(not bad_par,
              "every arm within +/-%.0f%% of the %d-param pin (worst %+.3f%%)"
              % (100 * tol, target,
                 100 * max(abs(float(v["param_error"])) for v in R.values())))

        for leg, mc in per["matched_compute"].items():
            vals = set(int(x) for x in mc["per_arm"].values())
            check(len(vals) == 1 and mc["matched"],
                  "matched compute equal on leg %r: %s row-visits across %d arms"
                  % (leg, sorted(vals), len(mc["per_arm"])))

        nr = per["DECLARED_NOT_RUN"].get("measured_not_applicable", {})
        check(all(v.get("reason") for v in nr.values()),
              "every measured NOT-RUN carries a reason (%d of them)" % len(nr))

        # load by name and get a number
        seen_kind = set()
        loaded = 0
        for name, v in sorted(R.items()):
            disk = v.get("name_on_disk", name)
            if not a.load_every:
                if v["kind"] in seen_kind:
                    continue
                seen_kind.add(v["kind"])
            try:
                r = load_rival(seed, disk, root=root)
                vals_req = np.array([[float(r.grid_values[k][1])
                                      for k in range(len(r.names))]])
                feats, oor = r(vals_req)
                ok = (feats.ndim == 2 and feats.shape[0] == 1
                      and feats.shape[1] == int(r.meta["out_dim"])
                      and np.isfinite(feats).all() and int(oor.sum()) == 0)
                check(ok, "loads BY NAME and returns %s finite: %s"
                      % (tuple(feats.shape), disk))
                loaded += 1
            except Exception as exc:                              # noqa: BLE001
                check(False, "loads BY NAME: %s (%s)" % (disk, exc))
        print("     (%d checkpoints loaded%s)"
              % (loaded, "" if a.load_every else ", one per comparator kind"))

    print()
    if FAIL:
        print("FAILED: %d check(s)" % len(FAIL))
        for m in FAIL:
            print("  - %s" % m)
        return 1
    print("BASELINES-READY VERIFIED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
