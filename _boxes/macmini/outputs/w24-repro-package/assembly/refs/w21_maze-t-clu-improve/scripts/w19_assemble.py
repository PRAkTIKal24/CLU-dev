"""w19 assembler -> MAZE-T-SYMMETRIC.json.  python scripts/w19_assemble.py <out_dir>"""
import json
import os
import sys

CONDS = ("WORDS-MAIN", "WORDS-FULL", "OBS", "BOTH")


def ld(p):
    return json.load(open(p)) if os.path.exists(p) else None


def main(out):
    res = {"artefact": "MAZE-T-SYMMETRIC.json", "hub": "w19 hub/maze-t-symmetric",
           "branch": "w19/maze-t-symmetric", "base": "w16/maze-wm 9f80f59",
           "prereg": "PREREG-w19-maze-t-symmetric.md (+ AMENDMENT 1)",
           "store_config": {"latent": 128, "K": 2, "n_atoms": 6, "ridge": 1.0,
                            "name": "A1 (UNCHANGED)"},
           "n_seeds": 3, "tolerance_used": None, "budget_scored_runs": 9,
           "conditions": {}, "diagnostics": {}}
    spent = 0
    for c in CONDS:
        v = ld(os.path.join(out, "val", "BARS-%s-val.json" % c))
        t = ld(os.path.join(out, "test", "BARS-%s-test.json" % c))
        e = {"val": None if v is None else {"PRIMARY_nway": v["PRIMARY_nway"],
                                            "nway": v["rows"]["nway"],
                                            "other_rows": {k: {"label": x["label"],
                                                               "CLU_mean": x["CLU_mean"],
                                                               "R_star_mean": x["R_star_mean"]}
                                                           for k, x in v["rows"].items()
                                                           if k != "nway"},
                                            "derangements": v["across_wells_derangement_per_seed"],
                                            "word_extractor_accuracy":
                                                v.get("word_extractor_accuracy"),
                                            "artefact": "val/BARS-%s-val.json" % c}}
        if t is not None:
            spent += 3
            e["test"] = {"PRIMARY_nway": t["PRIMARY_nway"], "nway": t["rows"]["nway"],
                         "other_rows": {k: {"label": x["label"], "CLU_mean": x["CLU_mean"],
                                            "R_star_mean": x["R_star_mean"]}
                                        for k, x in t["rows"].items() if k != "nway"},
                         "derangements": t["across_wells_derangement_per_seed"],
                         "word_extractor_accuracy": t.get("word_extractor_accuracy"),
                         "artefact": "test/BARS-%s-test.json" % c,
                         "per_seed_json": ["test/%s_test_seed%d.json" % (c, s) for s in (0, 1, 2)]}
            e["scored_runs"] = 3
        else:
            e["scored_runs"] = 0
            e["test"] = {"DECLARED_NOT_RUN": ("failed VAL (see val.PRIMARY_nway)"
                                              if v is not None and not v["PRIMARY_nway"]["PASS"]
                                              else "see FINDINGS.md s.budget")}
        res["conditions"][c] = e
    x = ld(os.path.join(out, "val", "BARS-OBS-X10-val.json"))
    if x is not None:
        res["diagnostics"]["OBS-X10 (PREREG AMENDMENT 1; printed only, never a bar)"] = {
            "nway": x["rows"]["nway"], "word_extractor_accuracy": x["word_extractor_accuracy"],
            "artefact": "val/BARS-OBS-X10-val.json"}
    res["budget"] = {"spent": spent, "declared_not_run": 9 - spent}
    json.dump(res, open(os.path.join(out, "MAZE-T-SYMMETRIC.json"), "w"), indent=1)
    print("wrote MAZE-T-SYMMETRIC.json; spent %d / 9" % spent)


if __name__ == "__main__":
    main(sys.argv[1])
