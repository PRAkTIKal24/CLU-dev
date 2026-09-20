"""GYM-1 evaluation: exact validity, persistence at one step, refusal, and the wall ablation.

Every number here comes from a FREE-RUNNING rollout with no action mask, no clamp of the
velocity to the speed limit, and no projection of the position back onto the line. The
batched environment is checked field-for-field against the scalar one before anything is
reported.
"""

from __future__ import annotations

import json
import os
import pickle
import time

import equinox as eqx
import jax
import numpy as np

from chlu.gym1 import episodes as E
from chlu.gym1 import train as TR
from chlu.gym1 import vecroll as VR
from chlu.gym1.line1d import Episode
from chlu.gym1.models import LSTMPolicy, TransformerPolicy, count_params

OUT = os.environ.get("CHLU_GYM1_OUT", "C:/chlu/outputs/w5-gym1")
N_EVAL = 1500        # episodes for the main horizon-48 metrics
N_LONG = 500         # episodes for the horizon-96 persistence probe


def load_models():
    """Load whichever checkpoints exist. A missing one is skipped and named in the
    output, so a partial evaluation is never silently reported as a complete one."""
    specs = {
        "lstm": (LSTMPolicy(d=64, h=205, key=jax.random.PRNGKey(0)), "ckpt_lstm.eqx"),
        "transformer": (TransformerPolicy(d=64, n_layers=3, heads=4, d_ff=422,
                                          max_len=128, key=jax.random.PRNGKey(1)),
                        "ckpt_transformer.eqx"),
    }
    out, missing = {}, []
    for name, (skeleton, fn) in specs.items():
        path = os.path.join(OUT, fn)
        if not os.path.exists(path):
            missing.append(name)
            continue
        out[name] = eqx.tree_deserialise_leaves(path, skeleton)
    if missing:
        print("[warn] NO CHECKPOINT, skipping:", missing)
    return out, missing


def m2d(m: VR.VecMetrics, keep_curves=True):
    d = m._asdict()
    d["survival"] = [float(z) for z in m.survival] if keep_curves else None
    d["at_goal_frac"] = [float(z) for z in m.at_goal_frac] if keep_curves else None
    return d


def policy_for(name, model, eps, horizon, override=None):
    """`override` replaces the rule tokens the model is shown, for the ablation."""
    n = len(eps)
    base = VR.pack(eps)

    if name == "lstm":
        raw = TR.lstm_policy(model, n)
    else:
        raw = TR.transformer_policy(model, n, horizon)
    if override is None:
        return raw

    def pol(t, x, v, P):
        P2 = dict(P)
        P2.update(override)
        return raw(t, x, v, P2)

    pol._unmasked = True
    return pol


def main():
    path = os.path.join(OUT, "eval.json")
    blob = pickle.load(open(os.path.join(OUT, "splits.pkl"), "rb"))
    held, ctrl, refusal = blob["held"], blob["ctrl"], blob["refusal"]
    out = {}

    # ---- 0a. the environment must apply illegal actions verbatim --------------
    gr = TR.verify_unmasked_rollout()
    print("[check] no guardrails in the rollout:", gr["ok"], gr)
    assert gr["ok"], "the rollout is sanitising illegal actions"
    out["no_guardrail_check"] = gr

    # ---- 0b. the batched environment must equal the scalar one ----------------
    sub = held[:300]
    chk = VR.check_against_scalar([r.ep for r in sub], [list(r.actions) for r in sub],
                                  E.HORIZON)
    print("[check] vec == scalar:", chk["ok"], chk["mismatches"][:3])
    assert chk["ok"], "batched environment disagrees with the scalar one"
    out["vec_scalar_check"] = chk

    heval = held[:N_EVAL]
    ceval = ctrl[:N_EVAL]
    eps_h = [r.ep for r in heval]
    eps_c = [r.ep for r in ceval]

    # ---- 1. oracles ----------------------------------------------------------
    t0 = time.time()
    opt_pol = lambda seqs: (lambda t, x, v, P: np.array(
        [s[t] if t < len(s) else 0 for s in seqs], dtype=np.int64))
    out["oracle_optimal_held"] = m2d(VR.vec_rollout(
        eps_h, opt_pol([list(r.actions) for r in heval]), E.HORIZON))
    out["oracle_wall_blind_held"] = m2d(VR.vec_rollout(
        eps_h, TR.wall_blind_oracle_policy(heval), E.HORIZON))
    print("[oracle] optimal=%.4f  wall-blind=%.4f  (%.1fs)" % (
        out["oracle_optimal_held"]["valid_and_reached"],
        out["oracle_wall_blind_held"]["valid_and_reached"], time.time() - t0))

    models, missing = load_models()
    out["params"] = {k: count_params(m) for k, m in models.items()}
    out["models_missing"] = missing

    for name, model in models.items():
        print(f"\n=== {name} ===", flush=True)
        t0 = time.time()
        p = policy_for(name, model, eps_h, E.HORIZON)
        assert TR.assert_no_guardrails(p), "rollout policy must be unmasked"
        mh = VR.vec_rollout(eps_h, p, E.HORIZON)
        out[f"{name}_held"] = m2d(mh)
        print("  held   valid_and_reached=%.4f all_valid=%.4f reached=%.4f held_at_goal=%.4f"
              " | speed=%d wall=%d teleport=%d over %d steps (%.1fs)"
              % (mh.valid_and_reached, mh.all_steps_valid, mh.reached_goal, mh.held_at_goal,
                 mh.n_speed, mh.n_wall, mh.n_teleport, mh.n_steps_total, time.time() - t0),
              flush=True)

        p = policy_for(name, model, eps_c, E.HORIZON)
        mc = VR.vec_rollout(eps_c, p, E.HORIZON)
        out[f"{name}_ctrl"] = m2d(mc)
        print("  ctrl   valid_and_reached=%.4f" % mc.valid_and_reached, flush=True)

        # ---- persistence at one step: rolled out twice as far as the horizon ---
        eps_l = eps_h[:N_LONG]
        p = policy_for(name, model, eps_l, E.HORIZON_LONG)
        ml = VR.vec_rollout(eps_l, p, E.HORIZON_LONG)
        out[f"{name}_held_long"] = m2d(ml)
        print("  long   H=%d all_valid=%.4f held_at_goal=%.4f speed=%d wall=%d tele=%d"
              % (E.HORIZON_LONG, ml.all_steps_valid, ml.held_at_goal,
                 ml.n_speed, ml.n_wall, ml.n_teleport), flush=True)

        # ---- wall ablation ------------------------------------------------------
        # Two probes, because they answer different questions and conflating them
        # would misreport the result:
        #   (a) w0 SHUFFLED across episodes -- every rule token stays in-distribution
        #       (a real wall width, a real wall position), but the position no longer
        #       corresponds to the wall the environment actually has. This is the
        #       honest test of whether the model USES the wall information.
        #   (b) ww := 0 -- "no wall", a value that NEVER occurs in training. Its
        #       embedding is untrained, so a collapse here measures out-of-distribution
        #       input, NOT dependence on the obstacle. Reported, and labelled as such.
        eps_a = eps_h[:N_LONG]
        Pa = VR.pack(eps_a)
        p = policy_for(name, model, eps_a, E.HORIZON)
        mb = VR.vec_rollout(eps_a, p, E.HORIZON)
        out[f"{name}_held_wall_shown"] = m2d(mb, keep_curves=False)

        rng = np.random.default_rng(0)
        perm = rng.permutation(len(eps_a))
        p = policy_for(name, model, eps_a, E.HORIZON,
                       override={"w0": Pa["w0"][perm]})
        ms = VR.vec_rollout(eps_a, p, E.HORIZON)
        out[f"{name}_held_wall_shuffled"] = m2d(ms, keep_curves=False)

        p = policy_for(name, model, eps_a, E.HORIZON,
                       override={"ww": np.zeros_like(Pa["ww"])})
        ma = VR.vec_rollout(eps_a, p, E.HORIZON)
        out[f"{name}_held_wall_ww_zero_OOD"] = m2d(ma, keep_curves=False)

        print("  ablate  true wall=%.4f | w0 SHUFFLED (in-distribution)=%.4f delta=%+.4f"
              " | ww:=0 (OOD input, not a wall probe)=%.4f"
              % (mb.valid_and_reached, ms.valid_and_reached,
                 ms.valid_and_reached - mb.valid_and_reached, ma.valid_and_reached),
              flush=True)

        # ---- refusal split: goal unreachable, a fault-free behaviour exists ----
        eps_r = refusal[:N_LONG]
        p = policy_for(name, model, eps_r, E.HORIZON)
        mr = VR.vec_rollout(eps_r, p, E.HORIZON)
        out[f"{name}_refusal"] = m2d(mr, keep_curves=False)
        print("  refusal all_valid=%.4f wall_crossings=%d (goal is unreachable by "
              "construction; reached=%.4f must be 0)"
              % (mr.all_steps_valid, mr.n_wall, mr.reached_goal), flush=True)

    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print("\n[write]", path)


if __name__ == "__main__":
    main()
