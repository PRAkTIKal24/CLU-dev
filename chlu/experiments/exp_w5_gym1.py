"""w5/gym1 driver -- INSTRUMENT ONLY. Makes no claim about the CLU; spends no scored run.

Stages, each resumable (every stage writes its artefact and is skipped if present):

    data      sample the splits, save them
    tripwire  matched-bytes trajectory-lookup tripwire       <- reported FIRST
    novelty   exact N1 / N2 / N3 checks on the held-out split
    train     LSTM + transformer, rules visible, params matched, one-step loss
    eval      free-running rollouts: exact validity, persistence, refusal, wall ablation

Usage:  python -m chlu.experiments.exp_w5_gym1 --stage all
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time

import numpy as np

from chlu.gym1 import episodes as E
from chlu.gym1 import tripwire as TW
from chlu.gym1.line1d import L, V_MAX_MAX, wall_is_uncrossable

OUT = os.environ.get("CHLU_GYM1_OUT", "C:/chlu/outputs/w5-gym1")

N_TRAIN = 120_000
N_HELD = 3_000
N_HELD_TRAINDIST = 3_000    # in-distribution control: same (V, bucket) pairings as train
N_REFUSAL = 1_500
SEED_TRAIN, SEED_HELD, SEED_CTRL, SEED_REF = 1, 7, 13, 11


def _p(name):
    return os.path.join(OUT, name)


def jdump(obj, name):
    os.makedirs(OUT, exist_ok=True)
    with open(_p(name), "w") as f:
        json.dump(obj, f, indent=2, default=float)
    print(f"[write] {_p(name)}")


# ---------------------------------------------------------------------------

def stage_data(force=False):
    path = _p("splits.pkl")
    if os.path.exists(path) and not force:
        print("[skip] splits.pkl exists")
        with open(path, "rb") as f:
            return pickle.load(f)
    t0 = time.time()
    train = E.sample_split(N_TRAIN, seed=SEED_TRAIN, held=False)
    held = E.sample_split(N_HELD, seed=SEED_HELD, held=True)
    ctrl = E.sample_split(N_HELD_TRAINDIST, seed=SEED_CTRL, held=False)
    ctrl_keys = {r.ep.key() for r in train}
    ctrl = [r for r in ctrl if r.ep.key() not in ctrl_keys]
    refusal = E.sample_refusal(N_REFUSAL, seed=SEED_REF)
    blob = {"train": train, "held": held, "ctrl": ctrl, "refusal": refusal}
    os.makedirs(OUT, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(blob, f, protocol=4)
    print("[data] train=%d held=%d ctrl=%d refusal=%d in %.1fs"
          % (len(train), len(held), len(ctrl), len(refusal), time.time() - t0))

    # The obstacle really is uncrossable, proved on the transition graph itself.
    bad = [(V, w0, ww) for V in E.V_CHOICES for ww in E.WW_CHOICES for w0 in E.W0_CHOICES
           if not wall_is_uncrossable(V, w0, ww)]
    summary = {
        "n_train": len(train), "n_held": len(held), "n_ctrl": len(ctrl),
        "n_refusal": len(refusal),
        "held_VD_pairs": [list(x) for x in E.HELD_VD],
        "D_buckets": [list(x) for x in E.D_BUCKETS],
        "V_choices": list(E.V_CHOICES), "ww_choices": list(E.WW_CHOICES),
        "w0_choices": list(E.W0_CHOICES),
        "horizon": E.HORIZON, "horizon_long": E.HORIZON_LONG,
        "path_len_train": {
            "min": min(len(r.actions) for r in train),
            "median": int(np.median([len(r.actions) for r in train])),
            "max": max(len(r.actions) for r in train)},
        "path_len_held": {
            "min": min(len(r.actions) for r in held),
            "median": int(np.median([len(r.actions) for r in held])),
            "max": max(len(r.actions) for r in held)},
        "wall_uncrossable_all_rule_settings": (len(bad) == 0),
        "rule_settings_where_wall_crossable": bad,
        "held_wall_raises_cost_frac": sum(r.wall_costs for r in held) / len(held),
        "train_wall_raises_cost_frac": sum(r.wall_costs for r in train) / len(train),
    }
    jdump(summary, "splits_summary.json")
    return blob


# ---------------------------------------------------------------------------

def stage_tripwire(blob, budget, force=False):
    path = _p("tripwire.json")
    if os.path.exists(path) and not force:
        print("[skip] tripwire.json exists")
        with open(path) as f:
            return json.load(f)
    train, held, ctrl = blob["train"], blob["held"], blob["ctrl"]
    t0 = time.time()
    scores = TW.run_all(train, held, E.HORIZON, budget)
    ctrl_scores = [TW.t3_nn_relative(train, ctrl, E.HORIZON, budget)]
    out = {
        "budget_bytes": budget,
        "n_train_stored_from": len(train),
        "held": [s._asdict() for s in scores],
        "ctrl_in_distribution": [s._asdict() for s in ctrl_scores],
        "best_held_valid_and_reached": max(s.valid_and_reached for s in scores),
        "best_held_strategy": max(scores, key=lambda s: s.valid_and_reached).name,
        "seconds": time.time() - t0,
    }
    jdump(out, "tripwire.json")
    return out


# ---------------------------------------------------------------------------

def stage_novelty(blob, limit=1500, force=False):
    path = _p("novelty.json")
    if os.path.exists(path) and not force:
        print("[skip] novelty.json exists")
        with open(path) as f:
            return json.load(f)
    t0 = time.time()
    rep = E.novelty_report(blob["held"], blob["train"], limit=limit)
    rep_ctrl = E.novelty_report(blob["ctrl"], blob["train"], limit=min(limit, 500))
    out = {"held": rep, "ctrl_in_distribution": rep_ctrl, "seconds": time.time() - t0}
    jdump(out, "novelty.json")
    return out


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["all", "data", "tripwire", "novelty", "train", "eval"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    import jax
    from chlu.gym1.models import LSTMPolicy, TransformerPolicy, match_report
    k = jax.random.PRNGKey(0)
    lstm = LSTMPolicy(d=64, h=205, key=k)
    tfm = TransformerPolicy(d=64, n_layers=3, heads=4, d_ff=422, max_len=128, key=k)
    mr = match_report(lstm, tfm)
    print("[params] LSTM=%d  Transformer=%d  gap=%.4f%%  tripwire budget=%d bytes"
          % (mr.lstm_params, mr.tfm_params, 100 * mr.rel_gap, mr.budget_bytes))
    jdump({"lstm_params": mr.lstm_params, "transformer_params": mr.tfm_params,
           "relative_gap": mr.rel_gap, "bytes_per_param": 4,
           "tripwire_budget_bytes": mr.budget_bytes}, "param_match.json")

    blob = stage_data(force=args.force and args.stage in ("all", "data"))
    if args.stage in ("all", "data"):
        pass
    if args.stage in ("all", "tripwire"):
        tw = stage_tripwire(blob, mr.budget_bytes, force=args.force)
        print("\n=== TRIPWIRE (reported first) ===")
        for s in tw["held"]:
            print("  %-18s valid_and_reached=%.4f reached=%.4f all_valid=%.4f "
                  "hit=%.3f bytes=%d/%d"
                  % (s["name"], s["valid_and_reached"], s["reached"], s["all_valid"],
                     s["hit_rate"], s["bytes_used"], s["bytes_budget"]))
        print("  BEST = %s at %.4f" % (tw["best_held_strategy"],
                                       tw["best_held_valid_and_reached"]))
    if args.stage in ("all", "novelty"):
        nv = stage_novelty(blob, force=args.force)
        print("\n=== NOVELTY ===")
        print(" ", json.dumps(nv["held"], indent=2, default=float))


if __name__ == "__main__":
    main()
