"""Train the two GYM-1 baselines with the RULES VISIBLE, at matched parameters.

One-step cross-entropy only. Resumable: each model checkpoints every 500 steps and
restarts from its checkpoint. Each has a wall-clock budget so a stuck run cannot eat the
machine.
"""

from __future__ import annotations

import json
import os
import pickle
import time

import equinox as eqx
import jax

from chlu.gym1 import train as TR
from chlu.gym1.models import LSTMPolicy, TransformerPolicy, count_params

OUT = os.environ.get("CHLU_GYM1_OUT", "C:/chlu/outputs/w5-gym1")
STEPS = 8000
BATCH = 128
LR = 3e-4
BUDGET_S = {"lstm": 2700, "transformer": 4200}


def main():
    os.makedirs(OUT, exist_ok=True)
    blob = pickle.load(open(os.path.join(OUT, "splits.pkl"), "rb"))
    t0 = time.time()
    data = TR.build_arrays(blob["train"])
    print("[data] one-step arrays %s in %.1fs" % (data["x"].shape, time.time() - t0),
          flush=True)

    k = jax.random.PRNGKey(0)
    models = {
        "lstm": LSTMPolicy(d=64, h=205, key=k),
        "transformer": TransformerPolicy(d=64, n_layers=3, heads=4, d_ff=422,
                                         max_len=128, key=jax.random.PRNGKey(1)),
    }
    report = {}
    for name, model in models.items():
        print(f"\n=== training {name} ({count_params(model)} params) ===", flush=True)
        ck = os.path.join(OUT, f"ckpt_{name}")
        model, log = TR.train_model(model, data, steps=STEPS, batch=BATCH, lr=LR,
                                    seed=0 if name == "lstm" else 1,
                                    ckpt_path=ck, ckpt_every=500,
                                    time_budget_s=BUDGET_S[name])
        report[name] = {"params": log.params, "steps": log.steps,
                        "seconds": log.seconds,
                        "final_loss_mean_last200": float(
                            sum(log.losses[-200:]) / max(len(log.losses[-200:]), 1)),
                        "n_losses": len(log.losses)}
        print(f"[done] {name}: {report[name]}", flush=True)

    with open(os.path.join(OUT, "train_report.json"), "w") as f:
        json.dump({"report": report, "steps": STEPS, "batch": BATCH, "lr": LR,
                   "n_train_episodes": len(blob["train"]),
                   "n_hold_steps_supervised": TR.N_HOLD,
                   "max_len": TR.MAX_LEN}, f, indent=2)
    print("[write]", os.path.join(OUT, "train_report.json"))


if __name__ == "__main__":
    main()
