"""GYM-2 evaluation: exact validity, persistence, refusal, rule ablations, and GATE O.

Order of business, all of it free-running and unmasked:

  0. Re-prove the environment applies illegal actions verbatim, and check the batched
     environment field-for-field against the scalar reference. Nothing below is trusted
     until both pass.
  1. GATE O -- the rule-blind oracles. O1 obstacle-blind, O2 speed-blind, O3
     composition-blind. Each plans optimally with one rule (or the interaction) removed and
     is replayed into the TRUE environment. PASS requires valid_and_reached < 0.90.
  2. The two trained baselines on `held` and on `held_novel_layout`, at horizon 48 and at
     the persistence horizon 96: exact fault counts by kind, survival curves, at-goal-over-
     time curves.
  3. In-distribution rule ablations: permute the obstacle map across episodes, and permute
     V, so every input stays a value the model has seen but no longer describes its own
     episode. This is the parent's `w0`-shuffle control, ported.
  4. The refusal split (goal unreachable, a fault-free behaviour exists).
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

from chlu.gym2 import episodes as EP, grid2d as E2, train as TR, vecroll as VR
from chlu.gym2.models import LSTMPolicy, TransformerPolicy

OUT = os.environ.get("CHLU_HUB_OUTPUTS", "C:/chlu/outputs") + "/w51-gym2"
N_EVAL = int(os.environ.get("GYM2_N_EVAL", "1500"))


def metrics_dict(vm):
    d = vm._asdict()
    d.pop("clean_flags", None)
    d.pop("reached_flags", None)
    d["survival"] = [float(x) for x in vm.survival]
    d["at_goal_frac"] = [float(x) for x in vm.at_goal_frac]
    d["first_fault_hist"] = {str(k): v for k, v in vm.first_fault_hist.items()}
    return d


def brief(vm):
    return {"valid_and_reached": vm.valid_and_reached, "all_steps_valid": vm.all_steps_valid,
            "reached_goal": vm.reached_goal, "held_at_goal": vm.held_at_goal,
            "n_speed": vm.n_speed, "n_obstacle": vm.n_obstacle, "n_out": vm.n_out}


def main():
    t0 = time.time()
    with open(f"{OUT}/splits.pkl", "rb") as f:
        d = pickle.load(f)
    E2.reset_registry(d["layouts"])
    train, held, held_nl, refusal = (d["train"], d["held"], d["held_novel_layout"],
                                     d["refusal"])
    held = held[:N_EVAL]
    held_nl = held_nl[:N_EVAL]
    res = {}

    # --- 0. adversarial checks ------------------------------------------------
    vu = E2.verify_unmasked_rollout()
    assert vu["ok"]
    res["verify_unmasked_rollout"] = vu
    chk = VR.check_against_scalar([r.ep for r in held[:300]],
                                  [list(r.actions) for r in held[:300]], EP.HORIZON)
    res["vec_vs_scalar"] = chk
    print("vec_vs_scalar:", json.dumps(chk), flush=True)
    assert chk["ok"], "the batched environment disagrees with the scalar reference"

    BL = VR.blocked_stack()

    # --- 1. GATE O: rule-blind oracles ---------------------------------------
    print("--- GATE O: rule-blind oracles", flush=True)
    oracles = {}
    seqs = TR.oracle_optimal(held)
    oracles["optimal_control"] = brief(
        VR.vec_rollout([r.ep for r in held], TR.replay_batch_policy(seqs), EP.HORIZON, BL))
    for nm, fn in (("O1_obstacle_blind", TR.oracle_obstacle_blind),
                   ("O2_speed_blind", TR.oracle_speed_blind)):
        s = fn(held)
        BL2 = VR.blocked_stack()      # O1 registers an extra empty layout
        oracles[nm] = brief(VR.vec_rollout([r.ep for r in held],
                                           TR.replay_batch_policy(s), EP.HORIZON, BL2))
    s3, st3 = TR.oracle_composition_blind(held)
    BL3 = VR.blocked_stack()
    oracles["O3_composition_blind"] = brief(
        VR.vec_rollout([r.ep for r in held], TR.replay_batch_policy(s3), EP.HORIZON, BL3))
    oracles["O3_stats"] = st3
    names = ("O1_obstacle_blind", "O2_speed_blind", "O3_composition_blind")
    gate_o = {k: {"valid_and_reached": oracles[k]["valid_and_reached"],
                  "pass_below_0.90": oracles[k]["valid_and_reached"] < 0.90}
              for k in names}
    gate_o["GATE_O_PASS"] = all(gate_o[k]["pass_below_0.90"] for k in names)
    res["oracles"] = oracles
    res["GATE_O"] = gate_o
    print(json.dumps(oracles, indent=2), flush=True)
    print("GATE O:", json.dumps(gate_o, indent=2), flush=True)

    # --- 2. trained baselines -------------------------------------------------
    key = jax.random.PRNGKey(20260920)
    k1, k2 = jax.random.split(key)
    models = {}
    p1 = f"{OUT}/ckpt_lstm.eqx"
    p2 = f"{OUT}/ckpt_transformer.eqx"
    if os.path.exists(p1):
        models["lstm"] = eqx.tree_deserialise_leaves(
            p1, LSTMPolicy(d=64, h=200, key=k1))
    if os.path.exists(p2):
        models["transformer"] = eqx.tree_deserialise_leaves(
            p2, TransformerPolicy(d=64, n_layers=3, heads=4, d_ff=400,
                                  max_len=EP.HORIZON_LONG + 8, key=k2))
    if not models:
        print("no checkpoints yet; oracle gate only", flush=True)
        with open(f"{OUT}/eval.json", "w") as f:
            json.dump(res, f, indent=2, default=str)
        return

    remap, maps, pads = TR.make_feature_tables(
        [r.ep.lid for r in train] + [r.ep.lid for r in held]
        + [r.ep.lid for r in held_nl] + [e.lid for e in refusal])

    def policy_for(name, model, n, horizon):
        if name == "lstm":
            return TR.lstm_policy(model, n, remap, maps, pads)
        return TR.transformer_policy(model, n, horizon, remap, maps, pads)

    res["models"] = {}
    for name, model in models.items():
        blk = {}
        for setname, recs in (("held", held), ("held_novel_layout", held_nl)):
            eps = [r.ep for r in recs]
            for H, tag in ((EP.HORIZON, "h48"), (EP.HORIZON_LONG, "h96")):
                t1 = time.time()
                vm = VR.vec_rollout(eps, policy_for(name, model, len(eps), H), H, BL)
                blk[f"{setname}_{tag}"] = metrics_dict(vm)
                print(f"  {name} {setname} {tag}: vr={vm.valid_and_reached:.4f} "
                      f"valid={vm.all_steps_valid:.4f} reach={vm.reached_goal:.4f} "
                      f"held={vm.held_at_goal:.4f} ({time.time()-t1:.0f}s)", flush=True)
        # in-distribution ablations on `held`
        eps = [r.ep for r in held]
        rng = np.random.default_rng(0)
        perm = rng.permutation(len(eps))
        eps_map = [E2.Episode(eps[perm[i]].lid, e.V, e.sx, e.sy, e.vx0, e.vy0, e.gx, e.gy)
                   for i, e in enumerate(eps)]
        eps_V = [E2.Episode(e.lid, eps[perm[i]].V, e.sx, e.sy, e.vx0, e.vy0, e.gx, e.gy)
                 for i, e in enumerate(eps)]
        for tag, fake in (("ablate_map_shuffled", eps_map), ("ablate_V_shuffled", eps_V)):
            pol = policy_for(name, model, len(eps), EP.HORIZON)
            # the POLICY sees the shuffled rule; the ENVIRONMENT keeps the true one
            P_fake = VR.pack(fake)
            real = VR.pack(eps)

            def wrapped(t, x, y, vx, vy, P, _p=pol, _f=P_fake):
                return _p(t, x, y, vx, vy, _f)

            vm = VR.vec_rollout(eps, wrapped, EP.HORIZON, BL)
            blk[tag] = brief(vm)
            print(f"  {name} {tag}: vr={vm.valid_and_reached:.4f}", flush=True)
        # refusal split
        pol = policy_for(name, model, len(refusal), EP.HORIZON)
        vm = VR.vec_rollout(refusal, pol, EP.HORIZON, BL)
        blk["refusal"] = brief(vm)
        print(f"  {name} refusal: all_valid={vm.all_steps_valid:.4f} "
              f"collisions={vm.n_obstacle}", flush=True)
        res["models"][name] = blk

    best = max(res["models"][m]["held_h48"]["valid_and_reached"] for m in res["models"])
    res["GATE_H"] = {"best_baseline_valid_and_reached": best,
                     "lower": 0.05, "upper": 0.95,
                     "GATE_H_PASS": bool(0.05 < best < 0.95)}
    print("GATE H:", json.dumps(res["GATE_H"]), flush=True)

    res["seconds"] = time.time() - t0
    with open(f"{OUT}/eval.json", "w") as f:
        json.dump(res, f, indent=2, default=str)
    print(f"TOTAL {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
