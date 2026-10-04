"""w12 hub/composition-ladder -- THE BYTE LEDGERS, measured on the objects this hub fitted.

The single design's total is its ATOMS + its SUFFICIENT STATISTICS + its address book, all
three declared side by side and NEVER netted (w5's rule for `address_bytes`, w6's for
`coupling_bytes`, `downdate_store.byte_ledger`'s own note).  `pca_tensor`'s total is its
atoms + its address book; it keeps no statistics and therefore has no exact deletion.

This is what "1x" means on this hub's rows, and it is measured on THIS partition -- the
factor-MI wells are UNEQUAL, so the equal-width K=5 arithmetic in
`w12-single-design-crossover/LEDGER-ACROSS-K.json` (well widths 409..410) does not describe
the object that was actually fitted here.  Both numbers are printed.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

import chlu

if "CHLU-w12-composition-ladder" not in str(chlu.__file__):
    raise AssertionError("chlu resolves OUTSIDE this worktree: %s" % chlu.__file__)

REF_PIN = 524288


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    runs, out = Path(a.runs), Path(a.out)
    rep = {"reference_pin_bytes": REF_PIN,
           "never_netted": True,
           "what": ("every part declared beside the others; a total is the SUM and is "
                    "never reduced by netting one part against another"),
           "legs": {}}
    for leg in sorted(runs.glob("*_seed*")):
        e = {}
        sd = leg / "single_design_store.pkl"
        if sd.exists():
            with open(sd, "rb") as fh:
                st = pickle.load(fh)
            b1 = st.byte_ledger(REF_PIN)
            b2 = st.covariance_ledger()
            e["single_design_K5"] = {
                "p1_atom_bytes": int(b1["deployed_bytes"]),
                "p1_matches_524288": bool(b1["matches_budget"]),
                "rank_kept_per_well": b1["rank_kept_per_well"],
                "rank_over_d_per_well": b1["rank_over_d_per_well"],
                "any_well_saturated": b1["any_well_saturated"],
                "well_sizes": [int(x) for x in st.well_sizes],
                "p2_statistics_bytes": int(b2["total_statistics_bytes"]),
                "p2_covariance_bytes": int(b2["covariance_bytes"]),
                "p2_if_symmetric_triangle_float32": int(b2["if_symmetric_triangle_float32"]),
            }
        pc = leg / "pca_content.pkl"
        if pc.exists():
            with open(pc, "rb") as fh:
                obj = pickle.load(fh)
            e["pca_tensor"] = {
                "p1_atom_bytes": int(obj["byte_ledger"]["deployed_bytes"]),
                "p1_matches_524288": bool(obj["byte_ledger"]["matches_budget"]),
                "well_sizes": obj["well_sizes"],
                "p2_statistics_bytes": 0,
                "p2_note": ("pca_tensor keeps NO sufficient statistics -- it also has no "
                            "exact deletion.  The zero is the price it does not pay."),
            }
        for p in sorted(leg.glob("address_*_R*.pkl")):
            with open(p, "rb") as fh:
                addr = pickle.load(fh)
            e.setdefault("address_books", {})[p.name] = addr.byte_ledger()
        if e:
            rep["legs"][leg.name] = e

    # roll the totals up, per arm, over whatever legs exist
    tot = {}
    for legname, e in rep["legs"].items():
        for arm in ("single_design_K5", "pca_tensor"):
            if arm not in e:
                continue
            books = e.get("address_books", {})
            key = [k for k in books if k.startswith("address_%s_R" % arm)]
            ab = int(books[key[0]].get("address_bytes", 0)) if key else None
            t = e[arm]["p1_atom_bytes"] + e[arm]["p2_statistics_bytes"] + (ab or 0)
            tot.setdefault(arm, []).append({
                "leg": legname, "p1_atoms": e[arm]["p1_atom_bytes"],
                "p2_statistics": e[arm]["p2_statistics_bytes"],
                "p3_address_book": ab,
                "p3_status": ("measured" if ab is not None
                              else "DECLARED NOT-RUN -- no address book on disk for this leg"),
                "total_bytes": t,
                "times_reference_pin": round(t / REF_PIN, 4)})
    rep["per_arm"] = tot
    for arm, rows in tot.items():
        v = [r["times_reference_pin"] for r in rows]
        rep.setdefault("summary", {})[arm] = {
            "n_legs": len(rows), "times_reference_pin_min": min(v),
            "times_reference_pin_max": max(v),
            "total_bytes_min": min(r["total_bytes"] for r in rows),
            "total_bytes_max": max(r["total_bytes"] for r in rows)}
    p = out / "BYTES.json"
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1, default=str)
    print("wrote", p)
    print(json.dumps(rep.get("summary", {}), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
