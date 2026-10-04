"""w23 spoke/seed5-diagnosis -- machine summary.  Reads ONLY this spoke's own artefacts.

    python scripts/w23_summary.py <out_dir>
"""
import json
import os
import sys


def main(od):
    q1 = json.load(open(os.path.join(od, "Q1-TRACE.json")))
    q1b = json.load(open(os.path.join(od, "Q1-CONTENT.json")))
    q1c = json.load(open(os.path.join(od, "Q1-CONTENT-ALLSEEDS.json")))
    q2 = json.load(open(os.path.join(od, "Q2-REL-VALUE.json")))
    code = {tuple(r["code"]): r for r in q1b["codes"]}

    def tr(t):
        pr = q1["TRACE"][t]["per_request"]
        m = max(pr, key=lambda p: p["n_rows"])
        c = code[tuple(m["truth"])]
        return {"n": q1["TRACE"][t]["n"], "clu_exact": q1["TRACE"][t]["clu_exact"],
                "main_request_words": m["request_words"], "rows": m["n_rows"],
                "design_nonzero_cols": [x[1] for x in m["design_nonzero_cols"]],
                "c_head": m["c_head"], "c_oracle": m["c_oracle"],
                "c_err_norm": m["c_head_minus_oracle_norm"], "emitted": m["emitted"],
                "truth": m["truth"], "wrong_facts": m["wrong_facts"],
                "emitted_margins": m["emitted_margins"],
                "codec_roundtrip_ok": m["codec_roundtrip_ok"],
                "l2_oracle_decode": m["oracle_atoms_decode"],
                "l2_oracle_margins": m["oracle_margins"],
                "phi_null_fraction": m["estimable_null_fraction"],
                "ridge_leverage": m["ridge_leverage_phi_Hinv_phi"],
                "support_design_rank": m["support_design_rank"],
                "phi_in_support_rowspace_rel_resid": m["phi_in_rowspace_of_support_rel_resid"],
                "truth_code_span_rel_resid": c["clu.pkl"]["rel_resid"],
                "truth_code_fit_rows": c["n_train"],
                "clu_in_sample_on_truth_code": c["clu_in_sample_exact"]}

    out = {
        "spoke": "w23/seed5-diagnosis", "zero_fit": True, "scored_runs": 0,
        "Q1": {
            "locus": "CONTENT",
            "statement": ("seed-5 A1 n-CLU emits displacement N (1) instead of S (3) on "
                          "(2,0,1,2) and (2,0,3,2); both need outcome code (1,3,0,2); the "
                          "deployed atoms hold that code badly (span rel-resid 0.387; the "
                          "L2-best in-span point decodes displacement E with margin 0.04), "
                          "so the CLU is already wrong on 1/3 of the 317 FIT rows carrying "
                          "that exact code; design estimable, address error of ordinary size"),
            "failing": {t: tr(t) for t in ("2,0,1,2", "2,0,3,2")},
            "comparison_right": {t: tr(t) for t in ("2,0,3,3", "3,0,1,0")},
            "isolation_block_swaps": q1["ISOLATION"],
            "summary_all_tuples": q1["SUMMARY"],
            "design_global": q1["design_global"],
            "train_in_sample_exact": q1["train_in_sample_exact"],
            "pearson_span_resid_vs_in_sample": q1b["pearson_resid_vs_insample_ge20"],
            "controls_content": q1b["per_object"],
            "all_seeds_contingency_rows": q1c["contingency_rows"],
            "sources": ["Q1-TRACE.json", "Q1-CONTENT.json", "Q1-CONTENT-ALLSEEDS.json"]},
        "Q2": {
            "value_row_clu": q2["partA"]["clu"], "brief_clu": q2["partA"]["brief_clu"],
            "clu_mean6": q2["partA"]["clu_mean6"],
            "best_arm_by_definition": {k: {"mean6": v["mean6"], "mean_s0_2": v["mean_s0_2"]}
                                       for k, v in q2["partA"]["best_by_definition"].items()},
            "section9": q2["partA"]["section9"],
            "per_seed": {s: {k: v[k] for k in (
                "clu_value_exact_reproduced", "n_C4_cols", "C4_fit_rows_max_abs",
                "head_rows_C4_max_abs", "head_rows_other_rms",
                "read_invariance_zeroing_C4_max_abs", "emissions_identical_frac",
                "eq_columns", "eqx_crosses_naming_4", "by_truth", "store")}
                for s, v in q2["partB"].items()},
            "pooled_by_truth": q2["pooled_by_truth"],
            "source": "Q2-REL-VALUE.json"},
    }
    with open(os.path.join(od, "W23-SEED5-DIAGNOSIS.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote W23-SEED5-DIAGNOSIS.json")


if __name__ == "__main__":
    main(sys.argv[1])
