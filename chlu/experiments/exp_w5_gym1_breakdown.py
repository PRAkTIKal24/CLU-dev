"""Per-composition breakdown of the tripwire.

Which held-out (speed limit, displacement bucket) pairings does a matched-bytes lookup
table actually find hard? If some pairing resists lookup, that pairing is the only part
of GYM-1 with any instrument value left, and the Head should know which one.
"""

from __future__ import annotations

import collections
import json
import os
import pickle

import numpy as np

from chlu.gym1 import episodes as E
from chlu.gym1 import tripwire as TW
from chlu.gym1.line1d import replay_policy, rollout

OUT = os.environ.get("CHLU_GYM1_OUT", "C:/chlu/outputs/w5-gym1")


def per_episode_ok(eps, seqs, horizon):
    ok = []
    for ep, acts in zip(eps, seqs):
        if acts is None:
            ok.append(False)
            continue
        r = rollout(ep, replay_policy(acts), horizon=horizon, continue_after_fault=True)
        ok.append(bool(r.all_steps_valid and r.reached_goal))
    return np.array(ok)


def main():
    path = os.path.join(OUT, "tripwire_breakdown.json")
    if os.path.exists(path):
        print("[skip] exists")
        return
    blob = pickle.load(open(os.path.join(OUT, "splits.pkl"), "rb"))
    train, held = blob["train"], blob["held"]
    budget = json.load(open(os.path.join(OUT, "param_match.json")))["tripwire_budget_bytes"]
    eps = [r.ep for r in held]

    # T3: nearest neighbour on the relative key, per episode
    kept, _ = TW._fit_entries(train, budget, TW.entry_bytes_rel)
    arr = np.array([[train[i].ep.V, train[i].ep.ww, train[i].ep.v0,
                     train[i].ep.g - train[i].ep.s, train[i].ep.s - train[i].ep.w0]
                    for i in kept], dtype=np.int32)
    acts = [train[i].actions for i in kept]
    t3 = []
    for r in held:
        q = np.array([r.ep.V, r.ep.ww, r.ep.v0, r.ep.g - r.ep.s, r.ep.s - r.ep.w0],
                     dtype=np.int32)
        dV = arr[:, 0] - q[0]
        cV = np.where(dV > 0, 60 * dV, -6 * dV)
        dW = arr[:, 1] - q[1]
        cW = np.where(dW < 0, 25 * (-dW), 3 * dW)
        cost = (cV + cW + 30 * np.abs(arr[:, 2] - q[2])
                + 12 * np.abs(arr[:, 3] - q[3]) + np.abs(arr[:, 4] - q[4]))
        t3.append(acts[int(np.argmin(cost))])
    ok_t3 = per_episode_ok(eps, t3, E.HORIZON)

    # T4 at the conservative setting: long fragments, few of them
    F = TW.prune_dominated(TW.harvest_fragments(train[:30000], min_frag=12, budget=budget))
    groups = TW._group_by_v(F)
    t4 = [TW._stitch_one(r.ep, F, 3, E.HORIZON, groups) for r in held]
    ok_t4 = per_episode_ok(eps, t4, E.HORIZON)

    rows = []
    keys = [(r.ep.V, E.d_bucket(r.d)) for r in held]
    for key in sorted(set(keys)):
        m = np.array([k == key for k in keys])
        rows.append({"V": key[0], "bucket": key[1],
                     "d_range": list(E.D_BUCKETS[key[1]]),
                     "n": int(m.sum()),
                     "T3_nn_relative": float(ok_t3[m].mean()),
                     "T4_minfrag12_K3": float(ok_t4[m].mean())})
        print("V=%d bucket=%d |d|in%s n=%4d  T3=%.3f  T4(min_frag=12,K=3)=%.3f"
              % (key[0], key[1], E.D_BUCKETS[key[1]], m.sum(),
                 ok_t3[m].mean(), ok_t4[m].mean()))

    with open(path, "w") as f:
        json.dump({"rows": rows,
                   "overall": {"T3": float(ok_t3.mean()), "T4": float(ok_t4.mean())}}, f,
                  indent=2)
    print("[write]", path)


if __name__ == "__main__":
    main()
