"""GATE D -- path diversity. Reported FIRST, before any baseline (pre-registration P1).

If the median held-out path tiles from two stored fragments, no split will save the gym --
the parent's own stop condition. Measured exactly, against the corpus a stitching lookup
table actually attacks (the translation-invariant behaviour string), and also against the
parent's absolute-position corpus so the two are comparable and the gap is visible.
"""

from __future__ import annotations

import json
import os
import pickle
import time

import numpy as np

from chlu.gym2 import episodes as EP, grid2d as E2, layouts as LY

OUT = os.environ.get("CHLU_HUB_OUTPUTS", "C:/chlu/outputs") + "/w51-gym2"
N_CHECK = 1500


def load():
    with open(f"{OUT}/splits.pkl", "rb") as f:
        d = pickle.load(f)
    E2.reset_registry(d["layouts"])
    return d


def main():
    t0 = time.time()
    d = load()
    train, held, held_nl = d["train"], d["held"], d["held_novel_layout"]

    med_len = int(np.median([r.cost for r in held]))
    coarse = max(2, med_len // 2)      # MIN_FRAG_GATE, fixed by the split, not chosen
    print(f"median held-out optimal path length = {med_len} -> MIN_FRAG_GATE = {coarse}",
          flush=True)

    out = {"median_held_path_len": med_len, "MIN_FRAG_GATE": coarse,
           "n_train": len(train)}
    for name, hs in (("held", held), ("held_novel_layout", held_nl)):
        t1 = time.time()
        rep = EP.novelty_report(hs, train, min_frag=EP.MIN_FRAG, limit=N_CHECK,
                                coarse_min_frag=coarse)
        rep["seconds"] = time.time() - t1
        out[name] = rep
        print(f"--- {name} ({rep['seconds']:.0f}s)")
        print(json.dumps(rep, indent=2)[:2500], flush=True)

    h = out["held"]
    gate = {
        "N1_pass": h["N1_key_novel_frac"] == 1.0,
        "N2a_pass": h["N2a_position_path_novel_frac"] >= 0.95,
        "N2b_pass": h["N2b_action_seq_novel_frac"] >= 0.90,
        "N2c_pass": h["N2c_behaviour_novel_frac"] >= 0.90,
        "N3rel_median": h["N3rel_median"],
        "N3rel_pass": (h["N3rel_median"] or 0) >= 4,
        "N3rel_fail_hard": (h["N3rel_median"] or 0) <= 2,
    }
    gate["GATE_D_PASS"] = bool(gate["N1_pass"] and gate["N2a_pass"] and gate["N2b_pass"]
                               and gate["N2c_pass"] and gate["N3rel_pass"])
    out["GATE_D"] = gate
    print("GATE D:", json.dumps(gate, indent=2), flush=True)

    with open(f"{OUT}/diversity.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"TOTAL {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
