"""GYM-2 step 1: layouts, the compositional split, and the exactness checks on the split.

Writes `splits.pkl` + `splits_summary.json`. Nothing here is a gate; the gates run on top of
this in `exp_w51_gym2_diversity`, `_structure`, `_tripwire`, `_oracles`, `_train`, `_eval`.

Three evaluation sets, not one:
  held             -- held-out (V, tunnel-bucket) PAIRINGS on layouts seen in training.
                      This is the compositional split proper.
  held_novel_lay   -- the same held-out pairings on layouts that appear NOWHERE in
                      training. Strictly harder; reported alongside, so that "the model
                      memorised these maps" is a measurement rather than an objection.
  refusal          -- goal unreachable, but a fault-free behaviour exists. Inherited from
                      the parent: it is what proves the collision metric is able to fire.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import time
from collections import Counter

import numpy as np

from chlu.gym2 import episodes as EP, grid2d as E2, layouts as LY

OUT = os.environ.get("CHLU_HUB_OUTPUTS", "C:/chlu/outputs") + "/w51-gym2"

N_LAYOUTS_TRAIN = 150
N_LAYOUTS_NOVEL = 40
N_TRAIN = 60000
N_HELD = 3000
N_HELD_NOVEL = 1500
N_REFUSAL = 1000


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()

    # --- the unmasked-rollout adversarial check comes FIRST: if the environment repairs
    # illegal actions, every number downstream is vacuous.
    vu = E2.verify_unmasked_rollout()
    print("verify_unmasked_rollout:", json.dumps(vu), flush=True)
    assert vu["ok"], "the environment is not applying illegal actions verbatim"

    specs = LY.make_layouts(N_LAYOUTS_TRAIN + N_LAYOUTS_NOVEL, seed=20260920)
    train_specs = specs[:N_LAYOUTS_TRAIN]
    novel_specs = specs[N_LAYOUTS_TRAIN:]
    print(f"layouts: {len(train_specs)} train-visible + {len(novel_specs)} novel  "
          f"buckets={Counter(LY.t_bucket(s.tunnel_len) for s in specs)}", flush=True)

    print("sampling train ...", flush=True)
    train = EP.sample_split(train_specs, N_TRAIN, seed=101, held=False)
    print(f"  train {len(train)} in {time.time()-t0:.0f}s", flush=True)

    t1 = time.time()
    held = EP.sample_split(train_specs, N_HELD, seed=202, held=True)
    print(f"  held {len(held)} in {time.time()-t1:.0f}s", flush=True)

    t1 = time.time()
    held_nl = EP.sample_split(novel_specs, N_HELD_NOVEL, seed=303, held=True)
    print(f"  held_novel_layout {len(held_nl)} in {time.time()-t1:.0f}s", flush=True)

    t1 = time.time()
    refusal = EP.sample_refusal(train_specs, N_REFUSAL, seed=404)
    print(f"  refusal {len(refusal)} in {time.time()-t1:.0f}s", flush=True)

    # --- exactness checks on the split itself -------------------------------
    tr_keys = {r.ep.key() for r in train}
    he_keys = {r.ep.key() for r in held}
    hn_keys = {r.ep.key() for r in held_nl}
    tr_pair = {(r.ep.V, LY.t_bucket(r.tunnel_len)) for r in train}
    he_pair = {(r.ep.V, LY.t_bucket(r.tunnel_len)) for r in held}
    tr_lids = {r.ep.lid for r in train}
    hn_lids = {r.ep.lid for r in held_nl}

    checks = {
        "key_overlap_train_held": len(tr_keys & he_keys),
        "key_overlap_train_heldnovel": len(tr_keys & hn_keys),
        "pairing_overlap": len(tr_pair & he_pair),
        "held_pairings": sorted(he_pair),
        "train_pairings_n": len(tr_pair),
        "every_V_in_train": sorted({v for v, _ in tr_pair}) == sorted(LY.V_CHOICES),
        "every_bucket_in_train": sorted({b for _, b in tr_pair}) == list(range(len(LY.T_BUCKETS))),
        "layout_overlap_train_heldnovel": len(tr_lids & hn_lids),
        "n_train_layouts_used": len(tr_lids),
    }
    print("split checks:", json.dumps(checks, default=str), flush=True)
    assert checks["key_overlap_train_held"] == 0
    assert checks["key_overlap_train_heldnovel"] == 0
    assert checks["pairing_overlap"] == 0
    assert checks["every_V_in_train"] and checks["every_bucket_in_train"]
    assert checks["layout_overlap_train_heldnovel"] == 0

    lay_entries = [(E2.get_layout(i).blocked, E2.get_layout(i).meta)
                   for i in range(E2.n_layouts())]
    with open(f"{OUT}/splits.pkl", "wb") as f:
        pickle.dump({"layouts": lay_entries, "train": train, "held": held,
                     "held_novel_layout": held_nl, "refusal": refusal,
                     "train_lids": sorted(s.lid for s in train_specs),
                     "novel_lids": sorted(s.lid for s in novel_specs)}, f, protocol=4)

    def summ(recs, name):
        if not recs:
            return {}
        lens = [r.cost for r in recs]
        return {
            "name": name, "n": len(recs),
            "V_hist": {int(k): int(v) for k, v in sorted(Counter(r.ep.V for r in recs).items())},
            "bucket_hist": {int(k): int(v) for k, v in
                            sorted(Counter(LY.t_bucket(r.tunnel_len) for r in recs).items())},
            "route_hist": dict(Counter(r.route for r in recs)),
            "crossing_frac": float(np.mean([r.crossing for r in recs])),
            "path_len_mean": float(np.mean(lens)),
            "path_len_median": float(np.median(lens)),
            "path_len_max": int(max(lens)),
            "n_layouts": len({r.ep.lid for r in recs}),
        }

    summary = {
        "grid": E2.G, "n_actions": E2.N_ACTIONS, "V_choices": list(LY.V_CHOICES),
        "t_buckets": [list(b) for b in LY.T_BUCKETS], "held_pairings": list(map(list, LY.HELD_VT)),
        "min_path": EP.MIN_PATH, "max_path": EP.MAX_PATH,
        "horizon": EP.HORIZON, "horizon_long": EP.HORIZON_LONG,
        "verify_unmasked_rollout": vu,
        "checks": checks,
        "train": summ(train, "train"), "held": summ(held, "held"),
        "held_novel_layout": summ(held_nl, "held_novel_layout"),
        "n_refusal": len(refusal),
        "seconds": time.time() - t0,
    }
    with open(f"{OUT}/splits_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(json.dumps(summary, indent=2, default=str)[:4000], flush=True)
    print(f"TOTAL {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
