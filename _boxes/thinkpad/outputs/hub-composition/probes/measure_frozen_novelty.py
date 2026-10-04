"""⭐ VALID-and-NOVEL counts for the dynread rungs, from FROZEN Mac artefacts.
⛔ Reads only. No encode, no decode, no classifier call — the classifier's tuples were
persisted by the run that produced them. ⛔ Not a scored run: it spends nothing."""
import json, os, sys, glob
import numpy as np
sys.path.insert(0, os.path.join(os.environ["OUT"], "instrument"))
from valid_novel import valid_novel_report

ART = ".claude/outputs/gen-read-cell-arm/runs/artefacts/shapes3d_recon"
CELL = ".claude/outputs/gen-read-cell-arm/runs/cell"
LABEL_BUDGET = {"dynread": "A", "dynread__A_request_off": "A",
                "dynread__B": "B", "dynread__B_request_off": "B"}
PRETTY = {"dynread": "dynread (A, full)", "dynread__A_request_off": "dynread/A_request_off",
          "dynread__B": "dynread/B (full)", "dynread__B_request_off": "dynread/B_request_off"}

out = {}
for seed in (0, 1, 2):
    man = json.load(open(f"{CELL}/seed_{seed}/seed_{seed}/cell_manifest.json"))
    for safe, budget in LABEL_BUDGET.items():
        for rs in ("B", "MS"):
            p = f"{ART}/seed_{seed}/decoded_s{seed}_{safe}_{rs}.npz"
            if not os.path.exists(p):
                print("MISSING", p); continue
            z = np.load(p, allow_pickle=True)
            at = np.load(f"{ART}/seed_{seed}/stored_atom_tuples_s{seed}_{budget}.npz")
            rep = valid_novel_report(z["factor_tuples"], z["labels"],
                                     at["atom_factor_tuples"], z["factor_names"])
            key = f"{PRETTY[safe]}|{rs}|seed{seed}"
            out[key] = rep
            # ⭐ CROSS-CHECK against the harness's own numbers for the same row
            lbl = {"dynread": "dynread", "dynread__A_request_off": "dynread/A_request_off",
                   "dynread__B": "dynread/B", "dynread__B_request_off": "dynread/B_request_off"}[safe]
            row = man["arms"].get(lbl)
            chk = ""
            if row:
                h_dist = row["distinct"][rs]["factor_tuples"]
                ok_d = (h_dist == rep["distinct"])
                chk = f" | harness distinct={h_dist} {'OK' if ok_d else '⛔MISMATCH'}"
                if rs == "B":
                    h_em = row["external"]["composition_B"]["exact_match"]
                    ok_e = abs(h_em - rep["exact_match_frac"]) < 1e-9
                    chk += f" ; harness exact_match={h_em:.4f} {'OK' if ok_e else '⛔MISMATCH'}"
                rep["cross_check"] = {"harness_distinct": h_dist, "distinct_agrees": bool(ok_d)}
            print(f"{key:44s} distinct={rep['distinct']:3d} novel={rep['distinct_novel']:3d} "
                  f"valid={rep['distinct_valid']:3d} ⭐VALID∧NOVEL={rep['distinct_valid_novel']:3d}"
                  f"{chk}")
    print()
p = os.path.join(os.environ["OUT"], "probes", "frozen_novelty.json")
json.dump(out, open(p, "w"), indent=2)
print("wrote", p)
