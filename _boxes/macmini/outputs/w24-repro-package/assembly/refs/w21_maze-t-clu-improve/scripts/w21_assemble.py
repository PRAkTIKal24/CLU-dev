"""w21 assembler -> MAZE-T-CLU-IMPROVE.json.   python scripts/w21_assemble.py <out_dir>"""
import glob
import json
import os
import sys


def ld(p):
    return json.load(open(p)) if os.path.exists(p) else None


def slim(lab):
    if lab is None:
        return None
    return {row: {k: v for k, v in p.items() if k not in ("means", "per_seed")}
            for row, p in lab["rows"].items()}


def main(out):
    val = {}
    for p in sorted(glob.glob(os.path.join(out, "val", "sweep_r*.json"))):
        for name, r in ld(p).items():
            val[name] = {"round": os.path.basename(p)[6:-5], "cand": r["cand"], "nway": r["nway"],
                         "nway_mean": r["nway_mean"], "pair_mean": r["pair_mean"],
                         "seen_mean": r["seen_mean"], "value_mean": r["value_mean"],
                         "read_floats": r["read_floats"], "gate_pass": r["gate_pass"]}
    res = {"artefact": "MAZE-T-CLU-IMPROVE.json", "hub": "w21 hub/maze-t-clu-improve",
           "branch": "w21/maze-t-clu-improve", "base": "w19/maze-t-symmetric 0455961",
           "prereg": "PREREG-w21-maze-t-clu-improve.md",
           "prereg_sha256": open(os.path.join(out, "PREREG-w21-maze-t-clu-improve.sha256")).read(),
           "question": "widen the CLU's CMAZE-T composition margin under identical inputs, by changing ONLY the CLU",
           "A1_val_nway": {"mean": 0.725, "per_seed": [0.8019, 0.5689, 0.8034]},
           "gate": "VAL n-way mean > A1 mean AND no seed below A1",
           "frozen_config": ld(os.path.join(out, "prereg", "CLU-REL-A10.json")),
           "n_seeds": 3, "tolerance_used": None,
           "val_candidates": val,
           "n_val_candidates": len([k for k in val if k != "A1-repro"]),
           "posthoc_controls_val": ld(os.path.join(out, "posthoc", "controls_val.json")),
           "BOTH_val": slim(ld(os.path.join(out, "val_full", "LABEL-BOTH-val.json"))),
           "BOTH_test_scored_runs_7_8_9": slim(ld(os.path.join(out, "test", "LABEL-BOTH-test.json"))),
           "WORDS_FULL_val": slim(ld(os.path.join(out, "val_full", "LABEL-WORDS-FULL-val.json"))),
           "WORDS_FULL_test": "DECLARED NOT-RUN by this hub (runs 4-6 are spoke/words-full-test's; brief asks VAL only)",
           "scored_runs_spent": 3 if ld(os.path.join(out, "test", "LABEL-BOTH-test.json")) else 0}
    json.dump(res, open(os.path.join(out, "MAZE-T-CLU-IMPROVE.json"), "w"), indent=1)
    print("wrote MAZE-T-CLU-IMPROVE.json")


if __name__ == "__main__":
    main(sys.argv[1])
