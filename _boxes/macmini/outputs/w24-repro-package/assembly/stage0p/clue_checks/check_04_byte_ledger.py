"""check_04 -- the settle ladder's byte ledger, per seed, straight from the arm.json files.

Wave 6's dial rests on the settle legs being PARAMETER-IDENTICAL and BYTE-IDENTICAL end to
end, so that `pca_settle - pca_settle_zerostep` is the dynamics and nothing else.  This
check re-derives that from the artefacts rather than from the table, and prints the
per-seed byte ledger that `SCORED-TABLE.md` collapses into one column.

Run:  OMP_NUM_THREADS=2 python checks/check_04_byte_ledger.py
"""
import glob
import json
import os

EV = os.environ.get(
    "W12_COMPOSE",
    r"C:\chlu\evidence\outputs\_thinkpad\w6\w6-compose-physics")
RES = {}


def _banner(s):
    print("\n" + "=" * 78 + "\n" + s + "\n" + "=" * 78)


def main():
    _banner("per-seed byte ledger of every settle leg (source: scored/*/arms/*/arm.json)")
    table = {}
    for s in (0, 1, 2):
        legs = {}
        pat = os.path.join(EV, "scored", "test_seed%d" % s, "arms", "*", "arm.json")
        for p in sorted(glob.glob(pat)):
            name = os.path.basename(os.path.dirname(p))
            if not (name.startswith("pca_settle") or name == "pca_tensor"):
                continue
            d = json.load(open(p))
            legs[name] = {
                "store_bytes": int(d["byte_ledger"]["deployed_bytes"]),
                "matches_budget": bool(d["byte_ledger"]["matches_budget"]),
                "address_bytes": d.get("address_bytes", {}).get("address_bytes"),
                "read_side_added":
                    d.get("coupling_bytes", {}).get("total_read_side_bytes_added"),
                "settle_steps": d.get("settle_steps"),
                "zerostep_bit_identical_to_pca_tensor":
                    d.get("zerostep_bit_identical_to_pca_tensor"),
                "cert_min_eig": d.get("coupling_certificate", {})
                                 .get("certificate", {})
                                 .get("certificate_min_eigenvalue"),
                "lambda_star": d.get("coupling_certificate", {})
                                .get("certificate", {})
                                .get("max_safe_coupling_strength"),
                "n_pairs": len(d.get("coupling_certificate", {})
                               .get("certificate", {})
                               .get("pair_spectral_norms", [])),
            }
        table["test_seed%d" % s] = legs
        if not legs:
            print("  DECLARED NOT-RUN: no arm.json under %s" % pat)
            continue
        print("  test_seed%d" % s)
        for n, v in legs.items():
            print("    %-22s store=%d(pin %s)  addr=%s  read-side added=%s  steps=%s"
                  % (n, v["store_bytes"], v["matches_budget"], v["address_bytes"],
                     v["read_side_added"], v["settle_steps"]))
    RES["table"] = table

    _banner("1 -- is the settle ladder byte-identical WITHIN a seed?")
    ok_within = True
    for s, legs in table.items():
        st = {n: v for n, v in legs.items() if n.startswith("pca_settle")}
        if not st:
            continue
        addr = {v["address_bytes"] for v in st.values()}
        add = {v["read_side_added"] for v in st.values()}
        store = {v["store_bytes"] for v in st.values()}
        good = len(addr) == 1 and len(add) == 1 and len(store) == 1
        ok_within &= good
        print("  %s  %d settle legs  address=%s  read-side added=%s  store=%s  -> %s"
              % (s, len(st), addr, add, store, "IDENTICAL" if good else "DIFFER"))
    RES["byte_identical_within_seed"] = bool(ok_within)
    print("VERDICT  the parameter-identical dial is byte-identical within each seed: %s"
          % ok_within)

    _banner("2 -- does the ONE column SCORED-TABLE.md prints hold across seeds?")
    printed = {"address_bytes": 189344, "read_side_added": 81312, "store_bytes": 524288}
    rows = {}
    for s, legs in table.items():
        v = legs.get("pca_settle_zerostep")
        if v is None:
            continue
        rows[s] = {k: v[k] for k in printed}
        print("  %s  address=%s (table prints %d)   read-side added=%s (table prints %d)"
              "   store=%s (table prints %d)"
              % (s, v["address_bytes"], printed["address_bytes"],
                 v["read_side_added"], printed["read_side_added"],
                 v["store_bytes"], printed["store_bytes"]))
    store_ok = all(r["store_bytes"] == printed["store_bytes"] for r in rows.values())
    addr_ok = all(r["address_bytes"] == printed["address_bytes"] for r in rows.values())
    add_ok = all(r["read_side_added"] == printed["read_side_added"]
                 for r in rows.values())
    RES["store_pin_holds_all_seeds"] = bool(store_ok)
    RES["address_column_holds_all_seeds"] = bool(addr_ok)
    RES["read_side_column_holds_all_seeds"] = bool(add_ok)
    RES["read_side_range"] = sorted({r["read_side_added"] for r in rows.values()})
    RES["address_range"] = sorted({r["address_bytes"] for r in rows.values()})
    print("  store pin 524,288 B holds on every seed        : %s" % store_ok)
    print("  address-book column holds on every seed        : %s   (range %s)"
          % (addr_ok, RES["address_range"]))
    print("  read-side-added column holds on every seed     : %s   (range %s)"
          % (add_ok, RES["read_side_range"]))
    print("DISCLOSURE  the store pin is exact on all three seeds, but the read-side")
    print("            columns in SCORED-TABLE.md are the SEED-0 values.  Seeds 1 and 2")
    print("            pay 84,216 B, not 81,312 B.  This makes the settle arm's byte")
    print("            cost slightly WORSE, never better, and leaves the dial intact.")

    _banner("3 -- the no-op assertion and the certificate, per seed")
    for s, legs in table.items():
        v = legs.get("pca_settle_zerostep")
        if v is None:
            continue
        print("  %s  settle_steps=%s  zerostep_bit_identical_to_pca_tensor=%s"
              % (s, v["settle_steps"], v["zerostep_bit_identical_to_pca_tensor"]))
        print("        certificate min eig=%s   lambda*=%s   pairs=%d"
              % (v["cert_min_eig"], v["lambda_star"], v["n_pairs"]))
    RES["zerostep_flag_all_seeds"] = bool(
        all(legs.get("pca_settle_zerostep", {}).get(
            "zerostep_bit_identical_to_pca_tensor") is True
            for legs in table.values() if legs))

    _banner("SUMMARY")
    for k in ("byte_identical_within_seed", "store_pin_holds_all_seeds",
              "zerostep_flag_all_seeds", "address_column_holds_all_seeds",
              "read_side_column_holds_all_seeds"):
        print("  %-34s %s" % (k, RES[k]))
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "check_04_byte_ledger.json")
    with open(out, "w") as fh:
        json.dump(RES, fh, indent=1, sort_keys=True)
    print("  wrote %s" % out)
    return 0 if (RES["byte_identical_within_seed"] and RES["store_pin_holds_all_seeds"]
                 and RES["zerostep_flag_all_seeds"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
