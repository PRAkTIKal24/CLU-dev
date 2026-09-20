"""Train the two GYM-2 baselines: LSTM and transformer, RULES VISIBLE, matched parameters.

One-step cross-entropy and nothing else (see `chlu.gym2.train` for how persistence at one
step is enforced). Parameter counts are printed, never estimated.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import time

import equinox as eqx
import jax
import numpy as np

from chlu.gym2 import episodes as EP, grid2d as E2, train as TR
from chlu.gym2.models import (LSTMPolicy, TransformerPolicy, count_params,
                              match_report, param_breakdown)

OUT = os.environ.get("CHLU_HUB_OUTPUTS", "C:/chlu/outputs") + "/w51-gym2"
STEPS = int(os.environ.get("GYM2_STEPS", "6000"))
BATCH = int(os.environ.get("GYM2_BATCH", "64"))
TIME_BUDGET = float(os.environ.get("GYM2_TIME_BUDGET", "2100"))


def main():
    t0 = time.time()
    with open(f"{OUT}/splits.pkl", "rb") as f:
        d = pickle.load(f)
    E2.reset_registry(d["layouts"])
    train = d["train"]

    key = jax.random.PRNGKey(20260920)
    k1, k2 = jax.random.split(key)
    lstm = LSTMPolicy(d=64, h=200, key=k1)
    tfm = TransformerPolicy(d=64, n_layers=3, heads=4, d_ff=400,
                            max_len=EP.HORIZON_LONG + 8, key=k2)
    mr = match_report(lstm, tfm)
    pm = {"lstm_params": mr.lstm_params, "tfm_params": mr.tfm_params,
          "rel_gap": mr.rel_gap, "tolerance": 0.02, "within_tolerance": mr.rel_gap <= 0.02,
          "budget_bytes": mr.budget_bytes,
          "lstm_breakdown": param_breakdown(lstm),
          "tfm_breakdown": param_breakdown(tfm)}
    print(json.dumps(pm, indent=2), flush=True)
    assert pm["within_tolerance"], "parameter match outside the pre-registered 2 %"
    with open(f"{OUT}/param_match.json", "w") as f:
        json.dump(pm, f, indent=2)

    data = TR.build_arrays(train)
    remap, maps, pads = TR.make_feature_tables(data["lid"])
    print(f"dataset {data['x'].shape}, {len(remap)} layouts, "
          f"{int(data['m'].sum())} supervised steps", flush=True)

    rep = {"param_match": pm, "steps_requested": STEPS, "batch": BATCH,
           "time_budget_s": TIME_BUDGET}
    for name, model in (("lstm", lstm), ("transformer", tfm)):
        print(f"--- training {name} ({count_params(model)} params)", flush=True)
        t1 = time.time()
        model, log = TR.train_model(model, data, remap, maps, pads, steps=STEPS,
                                    batch=BATCH, seed=1, ckpt_path=f"{OUT}/ckpt_{name}",
                                    time_budget_s=TIME_BUDGET)
        rep[name] = {"params": log.params, "steps": len(log.losses),
                     "seconds": log.seconds,
                     "final_loss": float(np.mean(log.losses[-200:])),
                     "loss_curve": [float(x) for x in log.losses[::100]]}
        print(f"  done in {log.seconds:.0f}s, final loss "
              f"{rep[name]['final_loss']:.4f}", flush=True)

    rep["seconds"] = time.time() - t0
    with open(f"{OUT}/train_report.json", "w") as f:
        json.dump(rep, f, indent=2)
    print(f"TOTAL {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
