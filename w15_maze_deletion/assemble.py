"""Assemble MAZE-DELETION-2B.json + TABLE.md from the per-seed run JSONs.

    python -m w15_maze_deletion.assemble <out_dir>

Every row carries ``environment: CMAZE-2b``.  No CMAZE-1 or CMAZE-2 row is read or
written here; the banked CMAZE-2 figures are cited in FINDINGS.md only, in prose, as the
thing this replication replaces -- never in this table.
"""

from __future__ import annotations

import json
import os
import sys

SEEDS = (0, 1, 2)


def load(p):
    with open(p) as fh:
        return json.load(fh)


def main(out):
    prim = {s: load(os.path.join(out, "runs", "deletion_primary_seed%d.json" % s))
            for s in SEEDS}
    cp = {s: os.path.join(out, "clean", "runs", "deletion_clean_seed%d.json" % s)
          for s in SEEDS}
    clean = {s: load(p) for s, p in cp.items() if os.path.exists(p)}
    for d in list(prim.values()) + list(clean.values()):
        assert d["environment"] == "CMAZE-2b"
        assert all(r["environment"] == "CMAZE-2b" for r in d["deletion_rows"])

    rows = []
    for s in SEEDS:
        d = prim[s]
        p = d["deletion_rows"][0]
        assert p["primary"]
        ad = d["adaptation_cost"]
        ex = ad["exact_head_downdate (w9 Amendment 1)"]
        ca = ad["carry_W_R (w9 as pre-registered)"]
        rv = d.get("rival_retrain_without", {})
        row = {
            "environment": "CMAZE-2b", "seed": s, "split_items": "TRAIN (val==test)",
            "deletion": p["label"],
            "n_rows_removed": p["n_rows_removed"], "n_rows_kept": p["n_rows_kept"],
            "downdate_vs_refit_statistic_max_rel_residual": p["statistic_max_rel_residual"],
            "subspace_max_projector_residual": p["subspace_max_projector_residual"],
            "head_gram_A_residual": p["head_gram_A_max_rel_residual"],
            "head_gram_Mx_residual": p["head_gram_Mx_max_rel_residual"],
            "read_residual": p["read_max_rel_residual_on_probe"],
            "sanity_read_change_by_deletion": p["sanity_read_change_by_deletion"],
            "refit": p["refit"], "passes_over_retained_data": p["passes_over_retained_data"],
            "items_kept": p["items_kept"],
            "seconds_downdate": p["seconds_downdate_store_plus_head"],
            "seconds_refit_without_same_object": p["seconds_refit_without_store_plus_head"],
            "adaptation_exact_head": {
                "seconds": ex["seconds"],
                "address_book_floats_changed": ex["address_book_floats_changed"],
                "address_book_floats_total": ex["address_book_floats_total"],
                "atom_floats_changed": ex["atom_floats_changed"],
                "atom_floats_total": ex["atom_floats_total"],
                "val_other_targets_validity_before_after": [
                    ad["other_before"]["validity"], ex["other_after"]["validity"]],
                "val_other_targets_validity_delta": ex["validity_delta_other_targets"],
                "val_green_refusal_before_after": ex["refusal_green_before_after"],
                "n_val_other": ad["n_other_rows"], "n_val_green": ad["n_green_rows"],
                "retrained": False},
            "adaptation_carry_as_w9_preregistered": {
                "val_other_targets_validity_before_after": [
                    ad["other_before"]["validity"], ca["other_after"]["validity"]],
                "val_other_targets_validity_delta": ca["validity_delta_other_targets"],
                "val_green_refusal_before_after": ca["refusal_green_before_after"]},
            "adaptation_test_leg": ad["test_leg"],
            "rival_retrain_without": rv,
            "source": "runs/deletion_primary_seed%d.json" % s,
            "encoder": d["encoder"], "bank": d["bank"], "git_head": d["git_head"],
            "Ztrain_sha256": d["Ztrain_sha256"],
        }
        if s in clean:
            c = clean[s]
            cpri = c["deletion_rows"][0]
            row["clean_checkout_replication"] = {
                "checkout": c["checkout"], "git_head": c["git_head"],
                "encoder": "retrained from scratch, %d steps" % c["config"]["ae_steps"],
                "encoder_sha256": c["encoder"]["sha256"],
                "encoder_bit_identical_to_printed": c["encoder"]["sha256"] == d["encoder"]["sha256"],
                "Ztrain_bit_identical": c["Ztrain_sha256"] == d["Ztrain_sha256"],
                "statistic_max_rel_residual": cpri["statistic_max_rel_residual"],
                "subspace_max_projector_residual": cpri["subspace_max_projector_residual"],
                "head_gram_A_residual": cpri["head_gram_A_max_rel_residual"],
                "head_gram_Mx_residual": cpri["head_gram_Mx_max_rel_residual"],
                "read_residual": cpri["read_max_rel_residual_on_probe"],
                "n_rows_removed": cpri["n_rows_removed"],
                "identical_to_primary": (
                    cpri["statistic_max_rel_residual"] == p["statistic_max_rel_residual"]
                    and cpri["subspace_max_projector_residual"]
                    == p["subspace_max_projector_residual"]),
                "source": "clean/runs/deletion_clean_seed%d.json" % s}
        else:
            row["clean_checkout_replication"] = {"DECLARED_NOT_RUN": "no clean run JSON"}
        rows.append(row)

    supp = []
    for s in SEEDS:
        for r in prim[s]["deletion_rows"][1:]:
            supp.append({"environment": "CMAZE-2b", "seed": s, "deletion": r["label"],
                         "n_rows_removed": r["n_rows_removed"],
                         "statistic": r["statistic_max_rel_residual"],
                         "projector": r["subspace_max_projector_residual"],
                         "read_residual": r["read_max_rel_residual_on_probe"],
                         "sanity_read_change": r["sanity_read_change_by_deletion"],
                         "source": "runs/deletion_primary_seed%d.json" % s})
    st = [r["downdate_vs_refit_statistic_max_rel_residual"] for r in rows]
    pj = [r["subspace_max_projector_residual"] for r in rows]
    allst = st + [r["statistic"] for r in supp]
    out_json = {
        "artefact": "MAZE-DELETION-2B",
        "environment": "CMAZE-2b",
        "environment_definition": (
            "w14 fixed request layer: w14_request_layer.task2.build_site_v2 banks "
            "bank_after_seed{0,1,2} (96 sites), GRID_REBASED, CMAZE-2b encoder pool 270 "
            "sites (w14_read_verb.pool), K=8, n_atoms=31, ridge=0.01, ae_steps=12000 "
            "-- the configuration of the printed CMAZE-2b rows (w14_read_verb.leg)"),
        "contains_no_CMAZE1_or_CMAZE2_row": True,
        "n_seeds": len(rows), "seeds": list(SEEDS), "scored_runs_spent": 0,
        "deletable_object": "store sufficient statistic (S,t,n) + atoms derived from it "
                            "+ address-book Gram (A, Mx)",
        "outside_the_deletable_object": "the frozen encoder and its frozen decoder",
        "primary_rows": rows,
        "summary": {
            "statistic_residual_per_seed": st,
            "statistic_residual_range": [min(st), max(st)],
            "projector_per_seed": pj,
            "supplementary_rows": len(supp),
            "all_rows_statistic_max": max(allst),
            "all_rows_projector_max": max([r["projector"] for r in supp] + pj),
            "all_rows_read_residual_max": max([r["read_residual"] for r in supp]
                                              + [r["read_residual"] for r in rows]),
        },
        "supplementary_rows": supp,
    }
    with open(os.path.join(out, "MAZE-DELETION-2B.json"), "w") as fh:
        json.dump(out_json, fh, indent=1)

    L = ["# MAZE-DELETION-2B -- exact deletion on CMAZE-2b",
         "",
         "**This table contains no CMAZE-1 or CMAZE-2 row.** Every row: environment "
         "**CMAZE-2b**. Seeds = 3. Scored runs spent = 0.",
         "",
         "| env | seed | deletion | rows removed / kept | downdate vs refit: statistic "
         "(max rel, float64) | subspace projector | head Gram A / Mx | read | read moved "
         "by deletion | clean-checkout statistic / projector | source |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        c = r["clean_checkout_replication"]
        cs = ("%.3e / %.3e" % (c["statistic_max_rel_residual"],
                               c["subspace_max_projector_residual"])
              if "statistic_max_rel_residual" in c else "NOT RUN")
        L.append("| CMAZE-2b | %d | %s | %d / %d | %.3e | %.3e | %.1e / %.1e | %.1e | %.3f "
                 "| %s | `%s` |"
                 % (r["seed"], r["deletion"], r["n_rows_removed"], r["n_rows_kept"],
                    r["downdate_vs_refit_statistic_max_rel_residual"],
                    r["subspace_max_projector_residual"], r["head_gram_A_residual"],
                    r["head_gram_Mx_residual"], r["read_residual"],
                    r["sanity_read_change_by_deletion"], cs, r["source"]))
    L += ["", "## Adaptation cost after deletion (NEVER 'no changes')", "",
          "| env | seed | path | seconds | address-book floats changed | atom floats "
          "changed | VAL other-target validity before → after (Δ) | VAL green refusal "
          "before → after | retrained |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        e = r["adaptation_exact_head"]
        c = r["adaptation_carry_as_w9_preregistered"]
        L.append("| CMAZE-2b | %d | exact head downdate (A−=ΦdᵀΦd, Mx−=ΦdᵀXd, re-solve) | "
                 "%.3f | %d / %d | %d / %d | %.4f → %.4f (%+.4f) | %.4f → %.4f | no |"
                 % (r["seed"], e["seconds"], e["address_book_floats_changed"],
                    e["address_book_floats_total"], e["atom_floats_changed"],
                    e["atom_floats_total"], *e["val_other_targets_validity_before_after"],
                    e["val_other_targets_validity_delta"],
                    *e["val_green_refusal_before_after"]))
        L.append("| CMAZE-2b | %d | basis-change carry W←WR (w9 as pre-registered) | "
                 "— | — | — | %.4f → %.4f (%+.4f) | %.4f → %.4f | no |"
                 % (r["seed"], *c["val_other_targets_validity_before_after"],
                    c["val_other_targets_validity_delta"],
                    *c["val_green_refusal_before_after"]))
    L += ["", "## Rival deletes only by retrain-without (cost printed)", "",
          "| env | seed | rival | params (target) | steps × batch | row reads of retained "
          "data | seconds | items kept | CLU downdate seconds |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        v = r["rival_retrain_without"]
        if "seconds" in v:
            L.append("| CMAZE-2b | %d | %s | %d (%d) | %d × %d | %d | %.1f | yes (%d rows) | %.3f |"
                     % (r["seed"], v["arm"], v["params"], v["target_params"], v["steps"],
                        v["batch"], v["row_reads_of_retained_data"], v["seconds"],
                        v["n_rows_retained_and_required"], r["seconds_downdate"]))
        else:
            L.append("| CMAZE-2b | %d | DECLARED NOT-RUN | | | | | | |" % r["seed"])
    L += ["", "## Supplementary deletions (same object, other words) -- labelled supplementary",
          "", "| env | seed | deletion | rows removed | statistic | projector | read | "
          "read moved |", "|---|---|---|---|---|---|---|---|"]
    for r in supp:
        L.append("| CMAZE-2b | %d | %s | %d | %.3e | %.3e | %.1e | %.3f |"
                 % (r["seed"], r["deletion"], r["n_rows_removed"], r["statistic"],
                    r["projector"], r["read_residual"], r["sanity_read_change"]))
    with open(os.path.join(out, "TABLE.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1])
