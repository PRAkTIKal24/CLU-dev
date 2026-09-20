"""GATE T -- the matched-bytes trajectory-lookup tripwire on GYM-2.

Run BEFORE the gym is adopted and before any baseline is trained, as pre-registered.
Reports every strategy T1-T4 with bytes used, a full (min_frag x K) grid for stitching, and
the three registered bands TW-A / TW-B / TW-C. TW-D needs the baselines and is decided in
the final report.

Also verifies, rather than asserts, the TW-C degeneracy claim made in the pre-registration:
is the harvested `min_frag = 3` fragment set behaviourally COMPLETE -- i.e. does it contain
every (start velocity, 3-action) macro-action the environment admits? If it does, stitching
with unbounded K is literally breadth-first search over the same transition graph that
produced the ground truth, and TW-C measures nothing about the gym.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import time

import numpy as np

from chlu.gym2 import episodes as EP, grid2d as E2, layouts as LY, tripwire as TW

OUT = os.environ.get("CHLU_HUB_OUTPUTS", "C:/chlu/outputs") + "/w51-gym2"
HARVEST_FROM = 20000
T4_SAMPLE = int(os.environ.get("T4_SAMPLE", "400"))
KS = (1, 2, 3, 4, 6)


def budget_bytes():
    import jax
    from chlu.gym2.models import LSTMPolicy, TransformerPolicy, match_report
    k = jax.random.PRNGKey(0)
    lstm = LSTMPolicy(d=64, h=200, key=k)
    tfm = TransformerPolicy(d=64, n_layers=3, heads=4, d_ff=400, key=k)
    mr = match_report(lstm, tfm)
    return mr


def completeness_check(F, V_list=(1, 2, 3, 4)):
    """Is the min_frag=3 fragment table behaviourally complete? Exact count."""
    have = set()
    for c, seq in zip(F.v_start, F.seqs):
        if len(seq) == 3:
            have.add((int(c), tuple(seq)))
    out = {}
    for V in V_list:
        need = 0
        got = 0
        for vx in range(-V, V + 1):
            for vy in range(-V, V + 1):
                c = TW.vcode(vx, vy)
                for a0 in E2.ACTIONS:
                    for a1 in E2.ACTIONS:
                        for a2 in E2.ACTIONS:
                            seq = (a0, a1, a2)
                            v = (vx, vy)
                            ok = True
                            for ax, ay in seq:     # only profiles legal under V matter
                                v = (v[0] + ax, v[1] + ay)
                                if abs(v[0]) > V or abs(v[1]) > V:
                                    ok = False
                                    break
                            if not ok:
                                continue
                            need += 1
                            got += int((c, seq) in have)
        out[str(V)] = {"legal_3step_macro_actions": need, "present_in_table": got,
                       "coverage": got / max(need, 1)}
    return out


def main():
    t0 = time.time()
    with open(f"{OUT}/splits.pkl", "rb") as f:
        d = pickle.load(f)
    E2.reset_registry(d["layouts"])
    train, held, held_nl = d["train"], d["held"], d["held_novel_layout"]

    mr = budget_bytes()
    budget = mr.budget_bytes
    med_len = int(np.median([r.cost for r in held]))
    coarse = max(2, med_len // 2)
    print(f"budget = 4 x {max(mr.lstm_params, mr.tfm_params)} params = {budget} bytes; "
          f"MIN_FRAG_GATE = {coarse}", flush=True)

    res = {"budget_bytes": budget, "lstm_params": mr.lstm_params,
           "tfm_params": mr.tfm_params, "rel_param_gap": mr.rel_gap,
           "MIN_FRAG_GATE": coarse, "median_held_path_len": med_len,
           "n_train": len(train), "horizon": EP.HORIZON, "strategies": {}}

    for name, hs in (("held", held), ("held_novel_layout", held_nl)):
        block = {}
        for fn in (TW.t1_exact, TW.t2_nn_absolute, TW.t3_nn_relative):
            t1 = time.time()
            s = fn(train, hs, EP.HORIZON, budget)
            block[s.name] = s._asdict()
            print(f"  [{name}] {s.name}: vr={s.valid_and_reached:.4f} "
                  f"all_valid={s.all_valid:.4f} hit={s.hit_rate:.3f} "
                  f"bytes={s.bytes_used}/{budget} ({time.time()-t1:.0f}s)", flush=True)
        res["strategies"][name] = block

    # --- T4 over the (min_frag, K) grid -------------------------------------
    mfs = sorted({3, 4, coarse, coarse + 2, coarse + 4})
    res["t4"] = {}
    res["completeness"] = None
    for mf in mfs:
        t1 = time.time()
        scores, F = TW.t4_stitch(train, held, EP.HORIZON, budget, ks=KS, min_frag=mf,
                                 max_frag=max(12, coarse + 6),
                                 harvest_from=HARVEST_FROM, sample=T4_SAMPLE, seed=7,
                                 verbose=True)
        res["t4"][str(mf)] = {s.name: s._asdict() for s in scores}
        print(f"  min_frag={mf}: frags={len(F.seqs)} (raw {F.n_before_prune}) "
              f"bytes={F.bytes_used}/{budget} ({time.time()-t1:.0f}s)", flush=True)
        for s in scores:
            print(f"     K={s.detail['K']}  vr={s.valid_and_reached:.4f} "
                  f"hit={s.hit_rate:.3f} all_valid={s.all_valid:.4f}", flush=True)
        if mf == 3:
            res["completeness"] = completeness_check(F)
            print("  TW-C completeness:", json.dumps(res["completeness"]), flush=True)

    # --- registered bands ---------------------------------------------------
    h = res["strategies"]["held"]
    tw_a = max([h[k]["valid_and_reached"] for k in h] +
               [res["t4"][str(m)][f"T4_stitch_mf{m}_K1"]["valid_and_reached"]
                for m in mfs])
    tw_b = max([res["t4"][str(m)][f"T4_stitch_mf{m}_K{K}"]["valid_and_reached"]
                for m in mfs if m >= coarse for K in KS if K <= 3] or [0.0])
    tw_c = max([res["t4"]["3"][f"T4_stitch_mf3_K{K}"]["valid_and_reached"] for K in KS])
    bands = {
        "TW_A_whole_trajectory": tw_a, "TW_A_threshold": 0.50, "TW_A_fires": tw_a > 0.50,
        "TW_B_coarse_recombination": tw_b, "TW_B_threshold": 0.60, "TW_B_fires": tw_b > 0.60,
        "TW_C_fine_recombination_not_a_gate": tw_c,
        "GATE_T_FIRES_pre_baseline": bool(tw_a > 0.50 or tw_b > 0.60),
    }
    res["bands"] = bands
    print("BANDS:", json.dumps(bands, indent=2), flush=True)

    res["seconds"] = time.time() - t0
    with open(f"{OUT}/tripwire.json", "w") as f:
        json.dump(res, f, indent=2, default=str)
    print(f"TOTAL {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
