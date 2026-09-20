"""At what byte budget does the coarse-band tripwire STOP firing? — instrument characterisation.

⛔ This does not harden GYM-2. GYM-2 is condemned by GATE T and stays condemned; nothing here
changes the gym, the split or the metric. What it measures is the **tripwire's operating point**,
because that is the number wave 7 actually needs before it picks an environment.

The hypothesis the whole result points at is arithmetic. A parameter-matched byte budget buys

    entries ~= budget / frag_bytes(min_frag)

macro-actions, while the environment has

    states = G^2 * (2V+1)^2

distinct (cell, velocity) states. If entries >> states the table can afford a useful macro-action
at *every* state, and stitching with a small K is then indistinguishable from planning — whatever
the gym looks like. GYM-2: 1,001,188 / 5 = 200,237 entries against 32,400 states, a factor of
**6.2 in the table's favour**. So the prediction is that the coarse band collapses once the
budget drops far enough that entries fall below the state count, and that the collapse point is a
property of the arithmetic rather than of this particular gym.

Sweeping the budget tests that prediction and turns it into a spec: an environment whose state
space is not large compared with `4 * n_params / frag_bytes` cannot be gated by this tripwire.
"""

from __future__ import annotations

import json
import os
import pickle
import time

import numpy as np

from chlu.gym2 import episodes as EP, grid2d as E2, layouts as LY, tripwire as TW

OUT = os.environ.get("CHLU_HUB_OUTPUTS", "C:/chlu/outputs") + "/w51-gym2"
BUDGETS = (1001188, 300000, 100000, 30000, 10000, 3000, 1000)
N_EVAL = int(os.environ.get("BUDGET_N_EVAL", "200"))
KS = (1, 2, 3)


def main():
    t0 = time.time()
    with open(f"{OUT}/splits.pkl", "rb") as f:
        d = pickle.load(f)
    E2.reset_registry(d["layouts"])
    train, held = d["train"], d["held"]
    with open(f"{OUT}/tripwire.json") as f:
        coarse = json.load(f)["MIN_FRAG_GATE"]

    rng = np.random.default_rng(23)
    sel = rng.choice(len(held), size=min(N_EVAL, len(held)), replace=False)
    recs = [held[int(i)] for i in sel]

    states_by_V = {V: E2.G * E2.G * (2 * V + 1) ** 2 for V in LY.V_CHOICES}
    res = {"min_frag": coarse, "n_eval": len(recs), "ks": list(KS),
           "frag_bytes_at_min_frag": TW.frag_bytes(coarse),
           "states_by_V": states_by_V,
           "states_at_Vmax": states_by_V[max(LY.V_CHOICES)],
           "rows": []}

    for B in BUDGETS:
        t1 = time.time()
        scores, F = TW.t4_stitch(train, recs, EP.HORIZON, B, ks=KS, min_frag=coarse,
                                 max_frag=max(12, coarse + 6), harvest_from=20000,
                                 sample=None, seed=7)
        row = {"budget": B, "fragments": len(F.seqs), "bytes_used": int(F.bytes_used),
               "entries_per_state_at_Vmax": len(F.seqs) / states_by_V[max(LY.V_CHOICES)],
               "seconds": time.time() - t1}
        for s in scores:
            row[f"K{s.detail['K']}"] = s.valid_and_reached
        res["rows"].append(row)
        print(json.dumps(row), flush=True)

    res["seconds"] = time.time() - t0
    with open(f"{OUT}/budget_sweep.json", "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2), flush=True)


if __name__ == "__main__":
    main()
