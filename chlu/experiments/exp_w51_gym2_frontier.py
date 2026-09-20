"""The frontier between "lookup-proof" and "learnable" — the task's fourth falsification case.

If path diversity cannot be pushed high enough without making the task unlearnable, that is the
real finding, and the frontier has to be measured rather than asserted. Two axes:

  F1  PATH LENGTH. Longer held-out paths need more stored fragments to tile, so lookup should
      get harder — but they are also harder to learn. Reported side by side, per length bucket:
      N3-rel, coarse coverability, the coarse-band tripwire T4(min_frag=MIN_FRAG_GATE, K<=3), and
      each trained baseline. Where the two curves cross, if they do, is the usable window.

  F2  TRAINING-CORPUS SIZE. Is the stitching table's power data-bound or structural? The same
      coarse band, with fragments harvested from 2.5 k / 10 k / 40 k training episodes at a fixed
      byte budget. If the curve is flat, more data is not what makes the tripwire fire and no
      amount of split engineering will change it.
"""

from __future__ import annotations

import json
import os
import pickle
import time

import numpy as np

from chlu.gym2 import episodes as EP, grid2d as E2, tripwire as TW, vecroll as VR

OUT = os.environ.get("CHLU_HUB_OUTPUTS", "C:/chlu/outputs") + "/w51-gym2"
BUCKETS = ((10, 12), (13, 15), (16, 19), (20, 36))
PER_BUCKET = int(os.environ.get("FRONTIER_PER_BUCKET", "120"))


def stitch_flags(train, recs, budget, min_frag, K, max_frag, harvest_from, F=None):
    """Per-episode success of the coarse band, so results can be bucketed afterwards."""
    if F is None:
        F = TW.harvest_fragments(train, min_frag=min_frag, max_frag=max_frag,
                                 budget=budget, harvest_from=harvest_from)
    by_lay = {}
    for i, r in enumerate(recs):
        by_lay.setdefault(r.ep.lid, []).append(i)
    ok = np.zeros(len(recs), dtype=bool)
    for lid, ids in sorted(by_lay.items()):
        LI = TW.build_layout_index(F, lid)
        for i in ids:
            p = TW._stitch_one(recs[i].ep, F, LI, K, EP.HORIZON)
            if p is None:
                continue
            r = E2.rollout(recs[i].ep, E2.replay_policy(p), horizon=EP.HORIZON,
                           continue_after_fault=True)
            ok[i] = r.all_steps_valid and r.reached_goal
    return ok, F


def main():
    t0 = time.time()
    with open(f"{OUT}/splits.pkl", "rb") as f:
        d = pickle.load(f)
    E2.reset_registry(d["layouts"])
    train, held = d["train"], d["held"]

    with open(f"{OUT}/tripwire.json") as f:
        tw = json.load(f)
    budget = tw["budget_bytes"]
    coarse = tw["MIN_FRAG_GATE"]

    rng = np.random.default_rng(11)
    sel, tags = [], []
    for lo, hi in BUCKETS:
        ids = [i for i, r in enumerate(held) if lo <= r.cost <= hi]
        take = rng.choice(ids, size=min(PER_BUCKET, len(ids)), replace=False)
        sel.extend(int(i) for i in take)
        tags.extend([f"{lo}-{hi}"] * len(take))
    recs = [held[i] for i in sel]
    tags = np.array(tags)
    print(f"frontier sample {len(recs)} episodes over {len(BUCKETS)} length buckets",
          flush=True)

    res = {"buckets": [list(b) for b in BUCKETS], "n_sampled": len(recs),
           "MIN_FRAG_GATE": coarse, "budget_bytes": budget}

    # --- N3-rel and coarse coverability per bucket --------------------------
    beh = EP.Corpus([EP.behaviour_str(r.ep.vx0, r.ep.vy0, r.actions) for r in train])
    per = {}
    for tag in sorted(set(tags)):
        idx = np.nonzero(tags == tag)[0]
        n3, cov = [], 0
        for i in idx:
            r = recs[int(i)]
            s = EP.behaviour_str(r.ep.vx0, r.ep.vy0, r.actions)
            T = len(r.actions)
            c = EP.fragment_cover(s, beh, T, EP.MIN_FRAG, 2, 1)
            if c is not None:
                n3.append(c)
            cov += int(EP.fragment_cover(s, beh, T, coarse, 2, 1) is not None)
        per[tag] = {"n": len(idx),
                    "N3rel_mean": float(np.mean(n3)) if n3 else None,
                    "N3rel_median": float(np.median(n3)) if n3 else None,
                    "coarse_coverable_frac": cov / max(len(idx), 1)}
    print("N3 by bucket:", json.dumps(per, indent=1), flush=True)

    # --- F1: coarse-band tripwire per bucket --------------------------------
    t1 = time.time()
    ok, F = stitch_flags(train, recs, budget, coarse, 3, max(12, coarse + 6), 20000)
    print(f"  coarse-band stitch over the sample in {time.time()-t1:.0f}s "
          f"({len(F.seqs)} fragments, {F.bytes_used} bytes)", flush=True)
    for tag in sorted(set(tags)):
        idx = tags == tag
        per[tag]["T4_coarse_K3_valid_and_reached"] = float(ok[idx].mean())

    # --- F1: trained baselines per bucket -----------------------------------
    try:
        import equinox as eqx
        import jax
        from chlu.gym2 import train as TR
        from chlu.gym2.models import LSTMPolicy, TransformerPolicy
        k1, k2 = jax.random.split(jax.random.PRNGKey(20260920))
        remap, maps, pads = TR.make_feature_tables(
            [r.ep.lid for r in train] + [r.ep.lid for r in recs])
        BL = VR.blocked_stack()
        eps = [r.ep for r in recs]
        for name, proto, path in (
                ("lstm", LSTMPolicy(d=64, h=200, key=k1), f"{OUT}/ckpt_lstm.eqx"),
                ("transformer", TransformerPolicy(d=64, n_layers=3, heads=4, d_ff=400,
                                                  max_len=EP.HORIZON_LONG + 8, key=k2),
                 f"{OUT}/ckpt_transformer.eqx")):
            if not os.path.exists(path):
                continue
            model = eqx.tree_deserialise_leaves(path, proto)
            pol = (TR.lstm_policy(model, len(eps), remap, maps, pads) if name == "lstm"
                   else TR.transformer_policy(model, len(eps), EP.HORIZON, remap, maps, pads))
            vm = VR.vec_rollout(eps, pol, EP.HORIZON, BL)
            good = vm.clean_flags & vm.reached_flags
            for tag in sorted(set(tags)):
                per[tag][f"{name}_valid_and_reached"] = float(good[tags == tag].mean())
            print(f"  {name} per-bucket done", flush=True)
    except Exception as e:                      # baselines are optional for F1
        print("baselines unavailable for the frontier:", repr(e), flush=True)

    res["F1_by_path_length"] = per

    # --- F2: is the table data-bound? ---------------------------------------
    f2 = {}
    sub = recs[::3]
    for hf in (2500, 10000, 40000):
        t1 = time.time()
        ok2, F2 = stitch_flags(train, sub, budget, coarse, 3, max(12, coarse + 6), hf)
        f2[str(hf)] = {"valid_and_reached": float(ok2.mean()),
                       "fragments": len(F2.seqs), "bytes_used": int(F2.bytes_used),
                       "n_eval": len(sub), "seconds": time.time() - t1}
        print(f"  harvest_from={hf}: vr={ok2.mean():.4f} "
              f"frags={len(F2.seqs)} ({time.time()-t1:.0f}s)", flush=True)
    res["F2_by_training_corpus_size"] = f2

    res["seconds"] = time.time() - t0
    with open(f"{OUT}/frontier.json", "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2), flush=True)


if __name__ == "__main__":
    main()
