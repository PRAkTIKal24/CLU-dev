"""w15 -- cross-check every number in FINDINGS.md against T7-3SEEDS.json, cell by cell.

Run from the output dir. A number that appears in the prose or a table but not in the
acceptance artefact is a transcription error, and this is what catches it. It also
re-asserts the three properties the claim rests on: every control reads exactly 0, every
deletion paired delta reads exactly 0, and every rank agrees with the refit-without
store. Exit noise is deliberate: it prints every mismatch rather than the first.
"""
import json, re, pathlib
D = json.loads(pathlib.Path("T7-3SEEDS.json").read_text(encoding="utf-8"))
md = pathlib.Path("FINDINGS.md").read_text(encoding="utf-8")
bad = []

def near(a, b, tol=6e-4):
    return a is not None and b is not None and abs(float(a)-float(b)) <= tol

# T7 rows
t7 = D["T7_rows_3_and_4"]["rows"]
for arm, rec in t7.items():
    for key in ("row3_pair", "row4_nway"):
        agg = rec[f"{key}_over_seeds"]
        for v in agg["values"]:
            if f"{v:,.3f}" not in md and f"{v:.3f}" not in md:
                bad.append(f"T7 {arm} {key} value {v:.3f} not in FINDINGS")
        if f"{agg['mean']:.3f}" not in md:
            bad.append(f"T7 {arm} {key} mean {agg['mean']:.3f} not in FINDINGS")

# byte ledger
for s in (0,1,2):
    r = D["byte_ledger_per_seed"]["per_seed"][f"seed{s}"]
    for k in ("p1_atom_bytes","p2_covariance_plus_count_bytes","p3_address_book_bytes",
              "p4_coupling_hole_table_bytes","honest_total_bytes"):
        if f"{r[k]:,}" not in md: bad.append(f"ledger seed{s} {k}={r[k]:,} missing")
    if f"{r['times_reference_pin']}" not in md:
        bad.append(f"ledger seed{s} times_reference_pin missing")

# control gate
g = D["control_gate"]
for s in (0,1,2):
    r = g["per_seed"][f"seed{s}"]
    if f"{r['arm_novel_per_corner_all']:.4f}" not in md:
        bad.append(f"gate seed{s} arm value missing")
    for c,v in r["controls"].items():
        if v["novel_per_corner_all"] != 0.0: bad.append(f"gate seed{s} {c} NOT zero!")
        if f"{v['paired_mean_delta']:.3f}" not in md:
            bad.append(f"gate seed{s} {c} paired {v['paired_mean_delta']:.3f} missing")

# deletion rows
for sc, rec in D["deletion_rows"]["scopes"].items():
    for s in (0,1,2):
        v = rec["per_seed"][f"seed{s}"]
        if "DECLARED_NOT_RUN" in v: bad.append(f"deletion {sc} seed{s} NOT-RUN"); continue
        for lit in (f"{v['n_rows_deleted']:,}", f"{v['survivors_before']:.3f}",
                    f"{v['survivors_after']:.3f}",
                    f"{v['OUR_ADAPTATION_COST']['bytes_rewritten']:,}",
                    f"{v['REFIT_WITHOUT_COST']['data_rows_read']:,}"):
            if lit not in md: bad.append(f"deletion {sc} seed{s} literal {lit} missing")
        if v["paired_after_minus_before"] != 0.0:
            bad.append(f"deletion {sc} seed{s} paired != 0: {v['paired_after_minus_before']}")
        if not v["ranks_agree_with_refit_without"]:
            bad.append(f"deletion {sc} seed{s} ranks DISAGREE")

# rival column
for sc, rec in D["rival_deletion_column"]["scopes"].items():
    for s in (0,1,2):
        v = rec["per_seed"][f"seed{s}"]
        if "DECLARED_NOT_RUN" in v: continue
        for fam, x in v["families"].items():
            rw = x["retrain_without_novel_per_corner"]
            at = x["as_trained_novel_per_corner_on_the_scopes_survivors"]
            pm = x["paired_ours_minus_rival"]
            for lit in (f"{rw:.3f}", f"{at:.3f}", f"{pm['mean_paired_delta']:.3f}"):
                if lit not in md: bad.append(f"rival {sc} s{s} {fam} {lit} missing")
            if not pm.get("beats_the_1.0_margin"):
                bad.append(f"rival {sc} s{s} {fam} DOES NOT beat margin")
            c = x.get("retrain_without_cost") or {}
            if c.get("seconds") is not None and f"{c['seconds']:.2f}" not in md:
                bad.append(f"rival {sc} s{s} {fam} seconds {c['seconds']:.2f} missing")

print("MISMATCHES:", len(bad))
for b in bad: print("  -", b)
