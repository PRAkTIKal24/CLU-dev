"""placement-landing (w26) — the ACCEPTANCE TEST for canonical placement.

Re-runs `mia-decay-measurement` §2 (the post-eviction / post-deletion column) with the
controller's placement rule swapped from refuse-and-relocate to PGCP, and the removal verb
swapped from `store.evict(slot)` to the new `Controller.delete(item_id)`.

Everything about the adversary is imported VERBATIM from `mia_harness.py` (statistics,
queries, the shipped two-phase read, the per-example LiRA scorer) — the only thing that
changes is how the worlds are built.

Arms:
  relocate   — the shipped w23 allocator (reproduces the published 0.99985 / 0.811)
  canon_sized— canonical placement, lattice sized so n_cells >= n_offered (BELOW capacity:
               the theorist's clean claim)
  canon_native—canonical placement on the un-inflated mia disk (7 cells < 8 offers => the
               §4b overflow counterfactual gap, pre-registered as NOT exact)
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
from chlu.core.placement import n_cells_for, radius_for_cells  # noqa: E402

CFG, DIM, CAP, N_BG, NQ = MH.CFG, MH.DIM, MH.CAP, MH.N_BG, MH.NQ
D_SAFE, R_DISK, CODEBOOK = MH.D_SAFE, MH.R_DISK, MH.CODEBOOK

R_SIZED = radius_for_cells(CAP, D_SAFE)  # smallest lattice radius holding >= 8 cells

ARMS = {
    #  name          placement     lattice radius   n_cells
    "relocate":    ("relocate",    None,            None),
    "canon_sized": ("canonical",   R_SIZED,         n_cells_for(R_SIZED, D_SAFE)),
    "canon_native": ("canonical",  R_DISK,          n_cells_for(R_DISK, D_SAFE)),
}


def _store():
    return AtomStorePotential(dim=DIM, capacity=CAP, alpha=CFG.atom_alpha,
                              s=CFG.atom_width, kappa=CFG.payload_kappa)


def _ctrl(arm):
    placement, radius, _ = ARMS[arm]
    if placement == "relocate":
        return Controller(_store(), d_safe=D_SAFE, budget=CAP, amp=CFG.atom_amp,
                          n_candidates=CFG.n_relocation_candidates)
    return Controller(_store(), d_safe=D_SAFE, budget=CAP, amp=CFG.atom_amp,
                      evict_policy="depth", placement="canonical", lattice_radius=radius)


def build_worlds(arm, seed, c_i, a_i, bg_pay, n_worlds, target_id=-1):
    """IN / IN-after-removal / OUT-paired / OUT-history, n_worlds of each.

    Same world construction as `mia_harness.build_worlds` (same keys, same proposals),
    plus the `od` arm = IN after the controller's REMOVAL verb:
      relocate  -> store.evict(slot)          (mask only; placement history retained)
      canonical -> Controller.delete(item_id) (Theorem 2 fix-up cascade)
    """
    prop = lambda k, n: np.asarray(disk_proposer(R_DISK, DIM)(k, n))[:, :2]  # noqa: E731
    keys = ("in", "od", "op", "oh")
    out = {f"{p}_{s}": [] for p in keys for s in "cpam"}
    tslots, n_live_in, n_live_od, n_live_oh, moves = [], [], [], [], []
    for w in range(n_worlds):
        key = jax.random.PRNGKey(1_000_003 * (seed + 1) + w)
        k_prop, k_in, _ = jax.random.split(key, 3)
        proposals = np.asarray(disk_proposer(R_DISK, DIM)(k_prop, N_BG))[:, :2]

        # --- IN ---
        c_in = _ctrl(arm)
        c_in.offer(item_id=target_id, q_new=c_i, payload=float(a_i), key=k_in, proposer=prop)
        tslot = c_in.live_slots()[0] if c_in.n_live else -1
        kk = k_in
        for j in range(N_BG):
            kk, ko = jax.random.split(kk)
            c_in.offer(item_id=j, q_new=proposals[j], payload=float(bg_pay[j]),
                       key=ko, proposer=prop)
        # the target's slot can move under canonical re-packing: re-read it
        rec = [r for r in c_in.records.values() if r.item_id == target_id]
        tslot = rec[0].slot if rec else tslot
        st = c_in.store
        for s, v in zip("cpam", (st.centers, st.payloads, st.amps, st.active)):
            out[f"in_{s}"].append(np.asarray(v))
        tslots.append(tslot)
        n_live_in.append(c_in.n_live)

        # --- IN after the REMOVAL verb ---
        if ARMS[arm][0] == "canonical" and rec:
            row = c_in.delete(target_id)
            moves.append(row["moves"])
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
        c_oh = _ctrl(arm)
        kk = k_in
        for j in range(N_BG):
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
    return W


def _pack(W, pre):
    return {"centers": W[pre + "_c"], "payloads": W[pre + "_p"],
            "amps": W[pre + "_a"], "active": W[pre + "_m"]}


def run(arm, seeds, n_targets, n_worlds, with_reads=True, target_id=-1):
    t0 = time.time()
    per_ex = {}
    byte_equal, moves_all = [], []

    def rec(k, d):
        per_ex.setdefault(k, []).append(d)

    for seed in seeds:
        rng_t = np.random.default_rng(4242 + seed)          # identical to mia_harness
        ang = rng_t.random(n_targets) * 2 * np.pi
        rad = R_DISK * np.sqrt(rng_t.random(n_targets)) * 0.8
        for ti in range(n_targets):
            c_i = np.array([rad[ti] * np.cos(ang[ti]), rad[ti] * np.sin(ang[ti])])
            a_i = float(CODEBOOK[ti % CAP])
            bg_pay = np.array([CODEBOOK[(ti + 1 + j) % CAP] for j in range(N_BG)])
            W = build_worlds(arm, seed, c_i, a_i, bg_pay, n_worlds, target_id)
            od, op, oh = _pack(W, "od"), _pack(W, "op"), _pack(W, "oh")
            moves_all += W["moves"].tolist()

            # bit-identity of IN-after-removal vs OUT-history (the theorem, measured)
            byte_equal.append(float(np.mean([
                all(od[k][i].tobytes() == oh[k][i].tobytes() for k in od)
                for i in range(n_worlds)])))

            # --- allocator-trace statistics (no read needed) ---
            rec(("history", "hole"), MH.per_example(MH.hole_stat(od, c_i),
                                                    MH.hole_stat(oh, c_i)))
            rec(("paired", "hole"), MH.per_example(MH.hole_stat(op, c_i),
                                                   MH.hole_stat(op, c_i)))
            rec(("history", "n_live"), MH.per_example(W["n_live_od"], W["n_live_oh"]))
            # --- potential probes (numpy) ---
            s4_od, s5_od = MH.potential_probe(od, c_i)
            s4_op, s5_op = MH.potential_probe(op, c_i)
            s4_oh, s5_oh = MH.potential_probe(oh, c_i)
            rec(("history", "s4"), MH.per_example(s4_od, s4_oh))
            rec(("history", "s5"), MH.per_example(s5_od, s5_oh))
            rec(("paired", "s4"), MH.per_example(s4_op, s4_op))
            rec(("paired", "s5"), MH.per_example(s5_op, s5_op))
            # --- query statistics (the shipped two-phase read) ---
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
            print(f"  [{arm} seed {seed} target {ti}] {time.time()-t0:.0f}s", flush=True)

    agg = {}
    for (col, stat), lst in per_ex.items():
        d = {"n_examples": len(lst)}
        for f in lst[0]:
            d[f] = MH.agg([x[f] for x in lst])
            d[f + "_all"] = [float(x[f]) for x in lst]
        agg.setdefault(col, {})[stat] = d
    return {
        "arm": arm, "placement": ARMS[arm][0], "lattice_radius": ARMS[arm][1],
        "n_cells": ARMS[arm][2], "seeds": seeds, "n_targets": n_targets,
        "n_worlds": n_worlds, "with_reads": with_reads, "target_id": target_id,
        "byte_equal_frac": [float(np.mean(byte_equal)), float(np.min(byte_equal))],
        "moves_per_delete": [float(np.mean(moves_all)), float(np.max(moves_all))],
        "columns": agg, "runtime_s": time.time() - t0,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--arms", default="relocate,canon_sized,canon_native")
    ap.add_argument("--target-id", type=int, default=-1)
    ap.add_argument("--out", default=os.path.join(HERE, "placement_mia.json"))
    a = ap.parse_args()
    seeds = [0] if a.quick else [0, 1, 2]
    n_targets = 2 if a.quick else MH.N_TARGETS
    n_worlds = 16 if a.quick else MH.N_WORLDS
    res = {"meta": {
        "commit": os.popen("git rev-parse --short HEAD").read().strip(),
        "jax": jax.__version__, "quick": a.quick, "seeds": seeds,
        "n_targets": n_targets, "n_worlds": n_worlds, "n_query": NQ,
        "target_id": a.target_id,
        "d_safe": D_SAFE, "R_mia": R_DISK, "R_sized": R_SIZED,
        "n_cells_mia": n_cells_for(R_DISK, D_SAFE),
        "n_cells_sized": n_cells_for(R_SIZED, D_SAFE),
        "store": {"alpha": CFG.atom_alpha, "s": CFG.atom_width,
                  "kappa": CFG.payload_kappa, "amp": CFG.atom_amp, "cap": CAP},
        "read": {"dt": CFG.dt, "gamma_address": CFG.gamma_address,
                 "gamma_read": CFG.gamma_read, "address_steps": CFG.address_steps,
                 "read_steps": CFG.read_steps},
    }}
    for arm in a.arms.split(","):
        print(f"== {arm} ==", flush=True)
        res[arm] = run(arm, seeds, n_targets, n_worlds, target_id=a.target_id)
    out = a.out.replace(".json", "_quick.json") if a.quick else a.out
    with open(out, "w") as fh:
        json.dump(res, fh, indent=2)
    print("wrote", out)
