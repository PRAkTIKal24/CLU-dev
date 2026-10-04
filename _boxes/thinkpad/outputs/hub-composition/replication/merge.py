"""Merge the four raw result files into replication.json and attach verdicts."""
import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))


def L(n):
    with open(os.path.join(HERE, n)) as fh:
        return json.load(fh)


def main():
    c1, c2, c3 = L("claim1.json"), L("claim2.json"), L("claim3.json")
    sw, ex, ov = L("claim3_sweep.json"), L("claim3_extra.json"), L("claim3_overlap.json")

    zero_arms = ["clu", "lace", "lace/ld_corrected", "lace/ld_published", "ebm_compose",
                 "dynread", "dynread/B", "dynread/A_request_off",
                 "dynread/B_request_off", "store_removed"]
    by_arm = {r["arm"]: r for r in c1["arms"]}
    all_zero = all(all(x == 0.0 for x in by_arm[a]["count72"]) for a in zero_arms)

    perm_draws = ([c3["c_controls"]["perm_rng%d" % i]["exact"] for i in range(3)]
                  + list(sw["perm"]))
    gauss_draws = ([c3["c_controls"]["gauss_rng%d" % i]["exact"] for i in range(3)]
                   + list(sw["gauss"]))

    out = {
        "what": "independent replication of three hub/composition headline numbers",
        "claim1": {
            "raw": c1,
            "hub": {"ceiling": [67.67, 2.49], "knn_proto": [14.33, 4.78],
                    "all_other_arms": 0.0},
            "mine": {"ceiling": [c1["ceiling"]["mean"], c1["ceiling"]["sd_pop"],
                                 c1["ceiling"]["sd_sample"]],
                     "knn_proto": [by_arm["knn_proto"]["mean"],
                                   by_arm["knn_proto"]["sd_pop"],
                                   by_arm["knn_proto"]["sd_sample"]],
                     "all_other_arms_exactly_zero": bool(all_zero)},
            "verdict": "CONFIRMED",
            "note": ("the hub's +/- is the POPULATION sd (ddof=0); the sample sd "
                     "(ddof=1) is 3.055 for the ceiling and 5.859 for knn_proto"),
        },
        "claim2": {
            "raw": c2,
            "hub": {"n_test_rows": 72, "n_distinct_tuples": 72, "n_in_train": 0},
            "mine": {"n_test_rows": c2["n_test_rows"],
                     "n_distinct_tuples": c2["n_distinct_test_tuples"],
                     "n_in_train": c2["n_test_tuples_also_in_train"]},
            "verdict": "CONFIRMED",
            "identity_distinct_valid_novel_eq_72_times_exact_match": "DOES NOT FOLLOW",
        },
        "claim3": {
            "raw": c3, "sweep": sw, "extra": ex, "val_row_provenance": ov,
            "a": {"hub": 491, "mine": c3["a_decoder_ceiling"]["exact"],
                  "verdict": "CONFIRMED"},
            "b": {"hub_exact": 10, "mine_exact": c3["b_roundtrip"]["exact"],
                  "hub_rel_err": 0.3486,
                  "mine_rel_err": c3["b_roundtrip"]["rel_err"]["mean_per_row"],
                  "verdict": "CONFIRMED"},
            "c_gauss": {
                "hub": 393, "mine_draws": gauss_draws,
                "mine_mean": statistics.fmean(gauss_draws),
                "mine_sd": statistics.stdev(gauss_draws),
                "mine_min": min(gauss_draws), "mine_max": max(gauss_draws),
                "hub_value_hit_exactly_by_a_draw": 393 in gauss_draws,
                "verdict": "CONFIRMED (within RNG spread)"},
            "c_minus_e": {"hub": 162, "mine": c3["c_controls"]["minus_e"]["exact"],
                          "verdict": "CONFIRMED"},
            "c_perm": {
                "hub": 95, "mine_draws": perm_draws,
                "mine_mean": statistics.fmean(perm_draws),
                "mine_sd": statistics.stdev(perm_draws),
                "mine_min": min(perm_draws), "mine_max": max(perm_draws),
                "gap_vs_my_mean": statistics.fmean(perm_draws) - 95,
                "gap_in_my_sds": (statistics.fmean(perm_draws) - 95)
                                 / statistics.stdev(perm_draws),
                "alt_conventions": {k: v["exact"] for k, v in ex["alt_perms"].items()},
                "verdict": "REFUTED as stated (no random permutation reproduced 95)"},
        },
    }
    with open(os.path.join(HERE, "replication.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print("perm mean %.2f sd %.2f min %d max %d"
          % (out["claim3"]["c_perm"]["mine_mean"], out["claim3"]["c_perm"]["mine_sd"],
             out["claim3"]["c_perm"]["mine_min"], out["claim3"]["c_perm"]["mine_max"]))
    print("gauss mean %.2f sd %.2f min %d max %d  hit393=%s"
          % (out["claim3"]["c_gauss"]["mine_mean"], out["claim3"]["c_gauss"]["mine_sd"],
             out["claim3"]["c_gauss"]["mine_min"], out["claim3"]["c_gauss"]["mine_max"],
             out["claim3"]["c_gauss"]["hub_value_hit_exactly_by_a_draw"]))
    print("all_other_arms_exactly_zero:", all_zero)
    print("wrote replication.json")


if __name__ == "__main__":
    main()
