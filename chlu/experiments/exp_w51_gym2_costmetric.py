"""Does the metric I am about to RECOMMEND actually work? Test it, do not assert it.

Section 9.1 of the final report tells wave 7 to score composition with an OPTIMALITY GAP (steps
against the provably optimal cost) rather than with a feasibility metric, because a speed limit
is an upper bound and its composition with an obstacle can only change a route's cost. A
recommendation that has not been run against the data is worth nothing, so this runs it.

Three things are re-scored under `optimal_and_valid` = fault-free AND parked on the goal AND
using exactly the provably optimal number of steps (and a tolerance curve around it):

  * the coarse-band lookup table T4 (min_frag = MIN_FRAG_GATE, K <= 3);
  * the composition-blind oracle O3, which scored 1.0000 on the feasibility metric;
  * both trained baselines.

If the cost metric separates the table from the models, it rescues the TRIPWIRE. If it also drops
O3 below 0.90 it rescues GATE O. Those are two different questions and they can come out
differently, so both are reported.
"""

from __future__ import annotations

import json
import os
import pickle
import time

import numpy as np

from chlu.gym2 import episodes as EP, grid2d as E2, train as TR, tripwire as TW, vecroll as VR

OUT = os.environ.get("CHLU_HUB_OUTPUTS", "C:/chlu/outputs") + "/w51-gym2"
N_EVAL = int(os.environ.get("COST_N_EVAL", "300"))
SLACKS = (0, 1, 2, 4, 8)


def curve(extra, n):
    """Fraction within `s` extra steps of optimal, for each slack in SLACKS."""
    e = np.array([x for x in extra if x is not None], dtype=np.float64)
    return {str(s): float((e <= s).sum()) / max(n, 1) for s in SLACKS}


def main():
    t0 = time.time()
    with open(f"{OUT}/splits.pkl", "rb") as f:
        d = pickle.load(f)
    E2.reset_registry(d["layouts"])
    train, held = d["train"], d["held"]
    with open(f"{OUT}/tripwire.json") as f:
        coarse = json.load(f)["MIN_FRAG_GATE"]
        budget = json.load(open(f"{OUT}/tripwire.json"))["budget_bytes"]

    rng = np.random.default_rng(31)
    sel = rng.choice(len(held), size=min(N_EVAL, len(held)), replace=False)
    recs = [held[int(i)] for i in sel]
    opt = np.array([r.cost for r in recs], dtype=np.float64)
    res = {"n_eval": len(recs), "min_frag": coarse, "slacks": list(SLACKS),
           "mean_optimal_cost": float(opt.mean())}

    # --- the coarse-band lookup table ---------------------------------------
    F = TW.harvest_fragments(train, min_frag=coarse, max_frag=max(12, coarse + 6),
                             budget=budget, harvest_from=20000)
    by_lay = {}
    for i, r in enumerate(recs):
        by_lay.setdefault(r.ep.lid, []).append(i)
    extra_t4 = [None] * len(recs)
    for lid, ids in sorted(by_lay.items()):
        LI = TW.build_layout_index(F, lid)
        for i in ids:
            p = TW._stitch_one(recs[i].ep, F, LI, 3, EP.HORIZON)
            if p is None:
                continue
            rr = E2.rollout(recs[i].ep, E2.replay_policy(p), horizon=EP.HORIZON,
                            continue_after_fault=True)
            if rr.all_steps_valid and rr.reached_goal:
                extra_t4[i] = rr.reach_step - recs[i].cost
    n_ok = sum(x is not None for x in extra_t4)
    res["T4_coarse_K3"] = {
        "valid_and_reached": n_ok / len(recs),
        "mean_extra_steps_when_solved": float(np.mean([x for x in extra_t4 if x is not None]))
                                        if n_ok else None,
        "within_slack": curve(extra_t4, len(recs)),
    }
    print("T4:", json.dumps(res["T4_coarse_K3"]), flush=True)

    # --- O3, the oracle that scored 1.0000 on the feasibility metric --------
    s3, st3 = TR.oracle_composition_blind(recs)
    extra_o3 = []
    for r, p in zip(recs, s3):
        rr = E2.rollout(r.ep, E2.replay_policy(p), horizon=EP.HORIZON,
                        continue_after_fault=True)
        extra_o3.append(rr.reach_step - r.cost if (rr.all_steps_valid and rr.reached_goal)
                        else None)
    n_ok = sum(x is not None for x in extra_o3)
    res["O3_composition_blind"] = {
        "valid_and_reached": n_ok / len(recs),
        "mean_extra_steps_when_solved": float(np.mean([x for x in extra_o3 if x is not None]))
                                        if n_ok else None,
        "within_slack": curve(extra_o3, len(recs)),
        "stats": st3,
    }
    print("O3:", json.dumps(res["O3_composition_blind"]), flush=True)

    # --- the trained baselines ----------------------------------------------
    import equinox as eqx
    import jax
    from chlu.gym2.models import LSTMPolicy, TransformerPolicy
    k1, k2 = jax.random.split(jax.random.PRNGKey(20260920))
    remap, maps, pads = TR.make_feature_tables(
        [r.ep.lid for r in train] + [r.ep.lid for r in recs])
    BL = VR.blocked_stack()
    for name, proto, path in (
            ("lstm", LSTMPolicy(d=64, h=200, key=k1), f"{OUT}/ckpt_lstm.eqx"),
            ("transformer", TransformerPolicy(d=64, n_layers=3, heads=4, d_ff=400,
                                              max_len=EP.HORIZON_LONG + 8, key=k2),
             f"{OUT}/ckpt_transformer.eqx")):
        if not os.path.exists(path):
            continue
        model = eqx.tree_deserialise_leaves(path, proto)
        # scalar rollout, so the exact reach step is available per episode
        extra = []
        n = len(recs)
        pol = (TR.lstm_policy(model, n, remap, maps, pads) if name == "lstm"
               else TR.transformer_policy(model, n, EP.HORIZON, remap, maps, pads))
        P = VR.pack([r.ep for r in recs])
        x, y = P["sx"].copy(), P["sy"].copy()
        vx, vy = P["vx0"].copy(), P["vy0"].copy()
        acts = np.zeros((n, EP.HORIZON, 2), dtype=np.int64)
        for t in range(EP.HORIZON):
            a = np.asarray(pol(t, x, y, vx, vy, P), dtype=np.int64)
            acts[:, t] = a
            vx = vx + a[:, 0]
            vy = vy + a[:, 1]
            x, y = x + vx, y + vy
            off = (x < 0) | (x >= E2.G) | (y < 0) | (y >= E2.G)
            x = np.where(off, np.clip(x, 0, E2.G - 1), x)
            y = np.where(off, np.clip(y, 0, E2.G - 1), y)
            vx = np.where(off, 0, vx)
            vy = np.where(off, 0, vy)
        for i, r in enumerate(recs):
            rr = E2.rollout(r.ep, E2.replay_policy([tuple(u) for u in acts[i]]),
                            horizon=EP.HORIZON, continue_after_fault=True)
            extra.append(rr.reach_step - r.cost if (rr.all_steps_valid and rr.reached_goal)
                         else None)
        n_ok = sum(u is not None for u in extra)
        res[name] = {
            "valid_and_reached": n_ok / len(recs),
            "mean_extra_steps_when_solved": float(np.mean([u for u in extra if u is not None]))
                                            if n_ok else None,
            "within_slack": curve(extra, len(recs)),
        }
        print(f"{name}:", json.dumps(res[name]), flush=True)

    res["seconds"] = time.time() - t0
    with open(f"{OUT}/cost_metric.json", "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2), flush=True)


if __name__ == "__main__":
    main()
