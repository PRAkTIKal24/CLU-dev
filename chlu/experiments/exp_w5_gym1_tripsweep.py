"""Tripwire diagnostics: how much recombination does GYM-1 actually cost a lookup table?

Two knobs, swept together:

  K         how many stored fragments may be concatenated (K=1 is whole-trajectory replay)
  min_frag  the shortest piece the table is allowed to reuse

`min_frag` is the honest separator. At min_frag=3 a stitching search over macro-actions is
arguably re-planning rather than lookup. At min_frag=8 or 12 the pieces are substantial
trajectory chunks, so if the held-out pairs still fall, they fall to genuine
trajectory recombination and nothing cleverer.
"""

from __future__ import annotations

import json
import os
import pickle
import time

from chlu.gym1 import episodes as E
from chlu.gym1 import tripwire as TW

OUT = os.environ.get("CHLU_GYM1_OUT", "C:/chlu/outputs/w5-gym1")
KS = (1, 2, 3, 4, 6, 8, 10, 12, 16)
MIN_FRAGS = (3, 5, 8, 12)


def main():
    path = os.path.join(OUT, "tripwire_sweep.json")
    if os.path.exists(path):
        print("[skip] tripwire_sweep.json exists")
        return
    blob = pickle.load(open(os.path.join(OUT, "splits.pkl"), "rb"))
    train, held = blob["train"], blob["held"]
    budget = json.load(open(os.path.join(OUT, "param_match.json")))["tripwire_budget_bytes"]

    rows = []
    for mf in MIN_FRAGS:
        t0 = time.time()
        scores = TW.t4_stitch(train, held, E.HORIZON, budget, ks=KS, min_frag=mf,
                              harvest_from=30000, sample=None)
        for s in scores:
            rows.append({"min_frag": mf, "K": s.detail["K"],
                         "fragments": s.detail["fragments"],
                         "valid_and_reached": s.valid_and_reached,
                         "all_valid": s.all_valid,
                         "bytes_used": s.bytes_used, "bytes_budget": s.bytes_budget})
        print("min_frag=%2d (%d frags, %.1fs): %s" % (
            mf, scores[0].detail["fragments"], time.time() - t0,
            " ".join("K%d=%.3f" % (s.detail["K"], s.valid_and_reached) for s in scores)))

    with open(path, "w") as f:
        json.dump({"rows": rows, "ks": list(KS), "min_frags": list(MIN_FRAGS),
                   "n_held": len(held)}, f, indent=2)
    print("[write]", path)


if __name__ == "__main__":
    main()
