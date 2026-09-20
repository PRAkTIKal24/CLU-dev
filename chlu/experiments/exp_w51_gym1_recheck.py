"""Re-verify, on the PARENT's own data, the two claims this retry builds on.

The brief says to treat the parent's numbers as reading, not truth, and to re-verify anything
built on them. Two things are load-bearing here and both are checked directly against
`w5-gym1/splits.pkl` using the parent's own code:

  R1  Is GYM-1's `min_frag = 3` fragment table behaviourally COMPLETE -- does it contain every
      (start velocity, 3-action) macro-action the dynamics admit? If so, the parent's headline
      row "T4 stitching, K=10 -> 1.0000" is the degenerate planner band (TW-C in this retry's
      pre-registration), not evidence about GYM-1 specifically, and the claim that a gym can be
      condemned by that row is too strong. The parent's *other* rows would still stand.

  R2  GYM-1's N3 measured the way a stitching table actually attacks it -- translation-invariant
      (start velocity + action string), not absolute positions. The parent reported N3-abs = 1.51
      mean. N3-rel is the number comparable to GYM-2's 2.22, and the two gyms can only be ranked
      on the same statistic.

Also re-runs the parent's coarse band (min_frag 12, K <= 3) so its 0.970 is reproduced rather
than quoted.
"""

from __future__ import annotations

import json
import os
import pickle
import time

import numpy as np

from chlu.gym1 import episodes as E1, line1d as L1, tripwire as T1

OUT = os.environ.get("CHLU_HUB_OUTPUTS", "C:/chlu/outputs") + "/w51-gym2"
PARENT = "C:/chlu/outputs/w5-gym1"
BUDGET = 997896          # the parent's own budget, from its LOG


def completeness(F, V_list=(2, 3, 4, 5, 6)):
    """Every (start velocity, 3-action) macro-action legal under V -- present or not."""
    have = set()
    for vs, seq in zip(F.v_start, F.seqs):
        if len(seq) == 3:
            have.add((int(vs), tuple(int(a) for a in seq)))
    out = {}
    for V in V_list:
        need = got = 0
        for v0 in range(-V, V + 1):
            for a0 in L1.ACTIONS:
                for a1 in L1.ACTIONS:
                    for a2 in L1.ACTIONS:
                        v = v0
                        ok = True
                        for a in (a0, a1, a2):
                            v += a
                            if abs(v) > V:
                                ok = False
                                break
                        if not ok:
                            continue
                        need += 1
                        got += int((v0, (a0, a1, a2)) in have)
        out[str(V)] = {"legal_3step_macro_actions": need, "present_in_table": got,
                       "coverage": got / max(need, 1)}
    return out


def n3_relative(held, train, min_frag=3, limit=1500):
    """N3 on the translation-invariant behaviour string, GYM-1 flavour.

    B = v0 a0 v1 a1 ... vT, so a fragment over steps [i, j) is exactly B[2i : 2j+1] and
    membership is an exact substring test -- the same construction used for GYM-2.
    """
    C_VEL, C_ACT, C_SEP = 0x800, 0x1000, 0x2000

    def bstr(v0, acts):
        out = [chr(C_VEL + int(v0) + 8)]
        v = int(v0)
        for a in acts:
            out.append(chr(C_ACT + int(a) + 1))
            v += int(a)
            out.append(chr(C_VEL + v + 8))
        return "".join(out)

    blob = chr(C_SEP).join(bstr(r.ep.v0, r.actions) for r in train)

    def contains(s):
        return s in blob

    def cover(s, T):
        INF = float("inf")
        dp = [INF] * (T + 1)
        dp[T] = 0
        for i in range(T - 1, -1, -1):
            lo, hi = 0, T - i
            while lo < hi:                       # exact: occurrence is monotone in length
                mid = (lo + hi + 1) // 2
                if contains(s[2 * i: 2 * (i + mid) + 1]):
                    lo = mid
                else:
                    hi = mid - 1
            if lo < min_frag:
                continue
            for ell in range(min_frag, lo + 1):
                j = i + ell
                if j != T and T - j < min_frag:
                    continue
                if dp[j] + 1 < dp[i]:
                    dp[i] = dp[j] + 1
        return None if dp[0] == INF else int(dp[0])

    covers, unc, novel = [], 0, 0
    sub = held[:limit]
    for r in sub:
        s = bstr(r.ep.v0, r.actions)
        novel += int(not contains(s))
        c = cover(s, len(r.actions))
        if c is None:
            unc += 1
        else:
            covers.append(c)
    hist = {int(k): int(covers.count(k)) for k in sorted(set(covers))}
    return {"n_checked": len(sub), "min_frag": min_frag,
            "N2c_behaviour_novel_frac": novel / max(len(sub), 1),
            "N3rel_mean": float(np.mean(covers)) if covers else None,
            "N3rel_median": float(np.median(covers)) if covers else None,
            "N3rel_hist": hist, "N3rel_uncoverable_frac": unc / max(len(sub), 1)}


def main():
    t0 = time.time()
    with open(f"{PARENT}/splits.pkl", "rb") as f:
        d = pickle.load(f)
    train, held = d["train"], d["held"]
    res = {"parent_train": len(train), "parent_held": len(held), "budget": BUDGET}

    print("R1: is GYM-1's min_frag=3 fragment table behaviourally complete?", flush=True)
    F3 = T1.prune_dominated(
        T1.harvest_fragments(train[:30000], min_frag=3, budget=BUDGET))
    res["R1_completeness_min_frag_3"] = completeness(F3)
    res["R1_fragments"] = len(F3.seqs)
    res["R1_bytes"] = F3.bytes_used
    print(json.dumps(res["R1_completeness_min_frag_3"], indent=1), flush=True)

    print("R2: GYM-1 N3 measured translation-invariantly", flush=True)
    res["R2_n3_relative"] = n3_relative(held, train, min_frag=3, limit=1500)
    print(json.dumps(res["R2_n3_relative"], indent=1), flush=True)

    print("R3: reproduce the parent's coarse band (min_frag 12, K<=3)", flush=True)
    sc = T1.t4_stitch(train, held, E1.HORIZON, BUDGET, ks=(1, 2, 3), min_frag=12,
                      harvest_from=30000, sample=1000, seed=0)
    res["R3_coarse_band"] = {s.name: {"valid_and_reached": s.valid_and_reached,
                                      "hit_rate": s.hit_rate,
                                      "bytes_used": s.bytes_used,
                                      "fragments": s.detail.get("fragments")}
                             for s in sc}
    print(json.dumps(res["R3_coarse_band"], indent=1), flush=True)

    res["seconds"] = time.time() - t0
    with open(f"{OUT}/gym1_recheck.json", "w") as f:
        json.dump(res, f, indent=2)
    print(f"TOTAL {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
