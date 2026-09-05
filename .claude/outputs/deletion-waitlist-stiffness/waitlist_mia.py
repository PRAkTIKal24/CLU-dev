"""deletion-waitlist-stiffness (w27) Part A — the ACCEPTANCE TEST for the P2 waitlist.

Same paired-world harness as `placement-landing/mia_placement.py` (adversary + statistics
imported VERBATIM from `mia-decay-measurement/mia_harness.py`), with three changes:

  1. `--offers N`  : total offers = 1 target + (N-1) background, so the LOAD is swept
                     (`carried-remeasurements`: the allocator leak is a curve, not a point).
  2. arms `p1`/`p2`: canonical placement on the un-inflated mia disk (7 cells) with the P2
                     waitlist OFF (w26 rung P1) / ON (w27).
  3. the delete verb is called UNCONDITIONALLY under canonical placement. The w26 harness
     only deleted when the target was *live*; at 8 offers with `target_id = -1` (the LOWEST
     priority of {-1,0..6}) the target is the key that fails to seat, so w26 fell through to
     `store.evict(tslot)` with a STALE `tslot` and deleted a background row instead. The P2
     waitlist makes `delete()` legal for an offered-but-unseated item, which is the correct
     counterfactual (the store never held it).

Also records `target_seated_frac` — the fraction of worlds in which the target actually got
a cell — because at overflow the two target ids exercise two different mechanisms.
"""
import argparse
import json
import os
import sys
import time

import jax
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "mia-decay-measurement"))

import mia_harness as MH  # noqa: E402

from chlu.core.admission import disk_proposer  # noqa: E402
from chlu.core.controller import Controller  # noqa: E402
from chlu.core.memory_potentials import AtomStorePotential  # noqa: E402
from chlu.core.placement import n_cells_for  # noqa: E402

CFG, DIM, CAP, NQ = MH.CFG, MH.DIM, MH.CAP, MH.NQ
D_SAFE, R_DISK, CODEBOOK = MH.D_SAFE, MH.R_DISK, MH.CODEBOOK

ARMS = {  # name -> (placement, waitlist)
    "p1": ("canonical", False),
    "p2": ("canonical", True),
    "relocate": ("relocate", False),
}


def _store(capacity):
    return AtomStorePotential(dim=DIM, capacity=capacity, alpha=CFG.atom_alpha,
                              s=CFG.atom_width, kappa=CFG.payload_kappa)


def _ctrl(arm, capacity):
    placement, waitlist = ARMS[arm]
    if placement == "relocate":
        return Controller(_store(capacity), d_safe=D_SAFE, budget=capacity,
                          amp=CFG.atom_amp, n_candidates=CFG.n_relocation_candidates)
    return Controller(_store(capacity), d_safe=D_SAFE, budget=capacity, amp=CFG.atom_amp,
                      evict_policy="depth", placement="canonical",
                      lattice_radius=R_DISK, waitlist=waitlist)


def build_worlds(arm, seed, c_i, a_i, bg_pay, n_worlds, target_id, n_bg, capacity):
    prop = lambda k, n: np.asarray(disk_proposer(R_DISK, DIM)(k, n))[:, :2]  # noqa: E731
    keys = ("in", "od", "op", "oh")
    out = {f"{p}_{s}": [] for p in keys for s in "cpam"}
    tslots, n_live_in, n_live_od, n_live_oh, moves, seated = [], [], [], [], [], []
    for w in range(n_worlds):
        key = jax.random.PRNGKey(1_000_003 * (seed + 1) + w)
        k_prop, k_in, _ = jax.random.split(key, 3)
        proposals = np.asarray(disk_proposer(R_DISK, DIM)(k_prop, n_bg))[:, :2]

        # --- IN ---
        c_in = _ctrl(arm, capacity)
        c_in.offer(item_id=target_id, q_new=c_i, payload=float(a_i), key=k_in,
                   proposer=prop)
        tslot = c_in.live_slots()[0] if c_in.n_live else 0
        kk = k_in
        for j in range(n_bg):
            kk, ko = jax.random.split(kk)
            c_in.offer(item_id=j, q_new=proposals[j], payload=float(bg_pay[j]),
                       key=ko, proposer=prop)
        rec = [r for r in c_in.records.values() if r.item_id == target_id]
        tslot = rec[0].slot if rec else tslot
        seated.append(1.0 if rec else 0.0)
        st = c_in.store
        for s, v in zip("cpam", (st.centers, st.payloads, st.amps, st.active)):
            out[f"in_{s}"].append(np.asarray(v))
        tslots.append(tslot)
        n_live_in.append(c_in.n_live)

        # --- IN after the REMOVAL verb (unconditional under canonical) ---
        if ARMS[arm][0] == "canonical":
            try:
                row = c_in.delete(target_id)
                moves.append(row["moves"])
            except KeyError:      # rung P1 with the target forgotten: nothing to delete
                moves.append(0)
            sd = c_in.store
        else:
            moves.append(0)
            sd = st.evict(tslot)
        for s, v in zip("cpam", (sd.centers, sd.payloads, sd.amps, sd.active)):
            out[f"od_{s}"].append(np.asarray(v))
        n_live_od.append(int(np.asarray(sd.active).sum()))

        # --- OUT paired: identical background placement, target slot masked ---
        op = st.evict(tslot)
        for s, v in zip("cpam", (op.centers, op.payloads, op.amps, op.active)):
            out[f"op_{s}"].append(np.asarray(v))

        # --- OUT history: the background sequence, target never offered ---
        c_oh = _ctrl(arm, capacity)
        kk = k_in
        for j in range(n_bg):
            kk, ko = jax.random.split(kk)
            c_oh.offer(item_id=j, q_new=proposals[j], payload=float(bg_pay[j]),
                       key=ko, proposer=prop)
        sh = c_oh.store
        for s, v in zip("cpam", (sh.centers, sh.payloads, sh.amps, sh.active)):
            out[f"oh_{s}"].append(np.asarray(v))
        n_live_oh.append(c_oh.n_live)

    W = {k: np.stack(v) for k, v in out.items()}
    W["tslot"] = np.array(tslots)
    W["n_live_in"] = np.array(n_live_in)
    W["n_live_od"] = np.array(n_live_od)
    W["n_live_oh"] = np.array(n_live_oh)
    W["moves"] = np.array(moves)
    W["seated"] = np.array(seated)
    return W


def _pack(W, pre):
    return {"centers": W[pre + "_c"], "payloads": W[pre + "_p"],
            "amps": W[pre + "_a"], "active": W[pre + "_m"]}


def run(arm, offers, seeds, n_targets, n_worlds, with_reads, target_id):
    t0 = time.time()
    n_bg = offers - 1
    capacity = max(CAP, offers)
    per_ex, byte_equal, moves_all, seated_all = {}, [], [], []

    def rec(k, d):
        per_ex.setdefault(k, []).append(d)

    for seed in seeds:
        rng_t = np.random.default_rng(4242 + seed)          # identical to mia_harness
        ang = rng_t.random(n_targets) * 2 * np.pi
        rad = R_DISK * np.sqrt(rng_t.random(n_targets)) * 0.8
        for ti in range(n_targets):
            c_i = np.array([rad[ti] * np.cos(ang[ti]), rad[ti] * np.sin(ang[ti])])
            a_i = float(CODEBOOK[ti % CAP])
            bg_pay = np.array([CODEBOOK[(ti + 1 + j) % CAP] for j in range(n_bg)])
            W = build_worlds(arm, seed, c_i, a_i, bg_pay, n_worlds, target_id, n_bg,
                             capacity)
            od, op, oh = _pack(W, "od"), _pack(W, "op"), _pack(W, "oh")
            moves_all += W["moves"].tolist()
            seated_all += W["seated"].tolist()
            byte_equal.append(float(np.mean([
                all(od[k][i].tobytes() == oh[k][i].tobytes() for k in od)
                for i in range(n_worlds)])))
            rec(("history", "hole"), MH.per_example(MH.hole_stat(od, c_i),
                                                    MH.hole_stat(oh, c_i)))
            rec(("paired", "hole"), MH.per_example(MH.hole_stat(op, c_i),
                                                   MH.hole_stat(op, c_i)))
            rec(("history", "n_live"), MH.per_example(W["n_live_od"], W["n_live_oh"]))
            s4_od, s5_od = MH.potential_probe(od, c_i)
            s4_op, s5_op = MH.potential_probe(op, c_i)
            s4_oh, s5_oh = MH.potential_probe(oh, c_i)
            rec(("history", "s4"), MH.per_example(s4_od, s4_oh))
            rec(("history", "s5"), MH.per_example(s5_od, s5_oh))
            rec(("paired", "s4"), MH.per_example(s4_op, s4_op))
            rec(("paired", "s5"), MH.per_example(s5_op, s5_op))
            if with_reads:
                Qn, Pn = MH.queries_native(seed, c_i, n_worlds)
                a_od, v_od = MH.read_batch(od, Qn, Pn)
                a_op, v_op = MH.read_batch(op, Qn, Pn)
                a_oh, v_oh = MH.read_batch(oh, Qn, Pn)
                s1d, s2d = MH.query_scores(a_od, v_od, c_i, a_i)
                s1p, s2p = MH.query_scores(a_op, v_op, c_i, a_i)
                s1h, s2h = MH.query_scores(a_oh, v_oh, c_i, a_i)
                rec(("history", "s1"), MH.per_example(s1d, s1h))
                rec(("history", "s2"), MH.per_example(s2d, s2h))
                rec(("paired", "s1"), MH.per_example(s1p, s1p))
                rec(("paired", "s2"), MH.per_example(s2p, s2p))
                rec(("retention_post", "-"), {"auc": float(
                    MH.retention(a_od, v_od, od, W["tslot"], a_i).mean())})
            print(f"  [{arm} off{offers} tid{target_id} seed {seed} target {ti}] "
                  f"{time.time()-t0:.0f}s", flush=True)

    agg = {}
    for (col, stat), lst in per_ex.items():
        d = {"n_examples": len(lst)}
        for f in lst[0]:
            d[f] = MH.agg([x[f] for x in lst])
            d[f + "_all"] = [float(x[f]) for x in lst]
        agg.setdefault(col, {})[stat] = d
    return {
        "arm": arm, "offers": offers, "waitlist": ARMS[arm][1], "target_id": target_id,
        "seeds": seeds, "n_targets": n_targets, "n_worlds": n_worlds,
        "with_reads": with_reads, "capacity": capacity,
        "n_cells": n_cells_for(R_DISK, D_SAFE),
        "byte_equal_frac": [float(np.mean(byte_equal)), float(np.min(byte_equal))],
        "moves_per_delete": [float(np.mean(moves_all)), float(np.max(moves_all))],
        "target_seated_frac": float(np.mean(seated_all)),
        "columns": agg, "runtime_s": time.time() - t0,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--arms", default="p1,p2")
    ap.add_argument("--offers", default="8")
    ap.add_argument("--target-ids", default="-5,-1")
    ap.add_argument("--no-reads", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    seeds = [0] if a.quick else [0, 1, 2]
    n_targets = 2 if a.quick else MH.N_TARGETS
    n_worlds = 16 if a.quick else MH.N_WORLDS
    res = {"meta": {
        "commit": os.popen("git rev-parse --short HEAD").read().strip(),
        "jax": jax.__version__, "quick": a.quick, "seeds": seeds,
        "n_targets": n_targets, "n_worlds": n_worlds, "n_query": NQ,
        "d_safe": D_SAFE, "R_mia": R_DISK, "n_cells": n_cells_for(R_DISK, D_SAFE),
        "with_reads": not a.no_reads,
        "store": {"alpha": CFG.atom_alpha, "s": CFG.atom_width,
                  "kappa": CFG.payload_kappa, "amp": CFG.atom_amp, "cap": CAP},
        "read": {"dt": CFG.dt, "gamma_address": CFG.gamma_address,
                 "gamma_read": CFG.gamma_read, "address_steps": CFG.address_steps,
                 "read_steps": CFG.read_steps},
    }}
    for arm in a.arms.split(","):
        for offers in [int(x) for x in a.offers.split(",")]:
            for tid in [int(x) for x in a.target_ids.split(",")]:
                k = f"{arm}|off{offers}|tid{tid}"
                print(f"== {k} ==", flush=True)
                res[k] = run(arm, offers, seeds, n_targets, n_worlds,
                             not a.no_reads, tid)
    out = os.path.join(HERE, f"waitlist_mia{a.tag}{'_quick' if a.quick else ''}.json")
    with open(out, "w") as fh:
        json.dump(res, fh, indent=2)
    print("wrote", out)
