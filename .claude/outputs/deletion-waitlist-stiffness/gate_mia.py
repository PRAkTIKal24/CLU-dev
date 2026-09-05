"""deletion-waitlist-stiffness (w27) Part B — `mia-decay` §1 / §3(b) / §5 under option (d).

Re-runs the mia-decay measurement with the payload channel swapped for the **gated
stiffness** (`AtomStorePotential(payload_gate=True, payload_g0=g0)`), which is the
theorist's option (d) for mia-D3 (payload-dependent lifetimes, r = -0.846 with a_i^2).

The worlds are built by `mia_harness.build_worlds` VERBATIM (shipped relocate allocator),
so the geometry, the targets, the backgrounds and the queries are element-for-element the
published ones; the ONLY thing that changes is `V`'s payload term and the read length.

Panels
  A  amplitude ladder: retention (mean + per-payload), query MIA (s1/s2), white-box s4/s5
  B  radius sweep -> R50 (the decisive number: does the retrieval-geometry differentiator
     survive the gate?), plus the TTL vector-store control
  L  read-length law: value error vs read_mult at several (g0, A) -- the compute-adaptive
     read requirement tau_y

Arms: base (gate off) | g05 (g0 = amp_floor = 0.05) | g005 (g0 = 0.005) | g005x4 (x4 read)
"""
import argparse
import dataclasses
import json
import os
import sys
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "mia-decay-measurement"))

import mia_harness as MH  # noqa: E402

from chlu.core.memory_potentials import AtomStorePotential  # noqa: E402
from chlu.experiments.exp_learned_memory import model_for  # noqa: E402
from chlu.experiments.exp_sequential_write import two_phase  # noqa: E402

CFG, DIM, CAP, NQ = MH.CFG, MH.DIM, MH.CAP, MH.NQ
CODEBOOK, TOL = MH.CODEBOOK, MH.TOL

ARMS = {  # name -> (payload_gate, g0, read_mult)
    "base": (False, 0.0, 1),
    "g05": (True, 0.05, 1),
    "g005": (True, 0.005, 1),
    "g005x4": (True, 0.005, 4),
    "g05x4": (True, 0.05, 4),
    "g05x2": (True, 0.05, 2),
}


def _cfg(read_mult):
    return CFG if read_mult == 1 else dataclasses.replace(
        CFG, read_steps=CFG.read_steps * read_mult)


def _tail_idx(cfg):
    return np.linspace(int((1 - cfg.tail_frac) * cfg.read_steps),
                       cfg.read_steps - 1, cfg.n_subsample).astype(int)


def make_reader(gate, g0, read_mult):
    """The SHIPPED two-phase read against a store with (or without) the gate."""
    tmpl = AtomStorePotential(dim=DIM, capacity=CAP, alpha=CFG.atom_alpha,
                              s=CFG.atom_width, kappa=CFG.payload_kappa,
                              payload_gate=gate, payload_g0=g0)
    cfg = _cfg(read_mult)
    tail = jnp.asarray(_tail_idx(cfg))

    def _one(centers, payloads, amps, active, Q0, P0):
        V = eqx.tree_at(lambda t: (t.centers, t.payloads, t.amps, t.active), tmpl,
                        (centers, payloads, amps, active))
        addr_q, traj = two_phase(model_for(V, DIM), Q0, P0, cfg,
                                 cfg.gamma_address, cfg.gamma_read)
        return addr_q, traj[:, tail, 2]

    fn = eqx.filter_jit(jax.vmap(_one, in_axes=(0, 0, 0, 0, 0, 0)))

    def read_batch(worlds, Q0, P0, chunk=384):
        W = worlds["centers"].shape[0]
        A, V = [], []
        for i in range(0, W, chunk):
            sl = slice(i, min(i + chunk, W))
            a, pt = fn(jnp.asarray(worlds["centers"][sl]),
                       jnp.asarray(worlds["payloads"][sl]),
                       jnp.asarray(worlds["amps"][sl]),
                       jnp.asarray(worlds["active"][sl]),
                       jnp.asarray(Q0[sl]), jnp.asarray(P0[sl]))
            A.append(np.asarray(a))
            V.append(np.asarray(pt).mean(-1))
        return np.concatenate(A), np.concatenate(V)

    return read_batch


def gated_probe(pack, c_i, gate, g0, eps=1e-6):
    """s4 (address depth, gate-independent) and s5 (full V at q2=0) for the arm's V."""
    rho = CFG.atom_width
    pts = np.concatenate([c_i[None, :], c_i[None, :] + MH._probe_dirs(rho)])
    d2 = ((pts[None, :, None, :] - pack["centers"][:, None, :, :]) ** 2).sum(-1)
    m = pack["active"][:, None, :]
    w = m * np.exp(-d2 / (2 * CFG.atom_width ** 2))
    U = CFG.atom_alpha * (pts ** 2).sum(-1)[None, :] - (pack["amps"][:, None, :] * w).sum(-1)
    gw = pack["amps"][:, None, :] * w
    if gate:
        G = g0 + gw.sum(-1)
        abar = (pack["payloads"][:, None, :] * gw).sum(-1) / (eps + gw.sum(-1))
        V = U + 0.5 * CFG.payload_kappa * G * abar ** 2      # q2 = 0
    else:
        S = (pack["payloads"][:, None, :] * w).sum(-1)
        V = U + 0.5 * CFG.payload_kappa * S ** 2
    return U[:, 1:].mean(1) - U[:, 0], V[:, 1:].mean(1) - V[:, 0]


def run(arms, panels, seeds, n_targets, n_worlds, a_grid, r_grid, a_radius, out_path):
    t0 = time.time()
    readers = {a: make_reader(*ARMS[a]) for a in arms}
    res = {"meta": {
        "commit": os.popen("git rev-parse --short HEAD").read().strip(),
        "jax": jax.__version__, "seeds": seeds, "n_targets": n_targets,
        "n_worlds": n_worlds, "n_query": NQ, "arms": {a: ARMS[a] for a in arms},
        "panels": panels, "A_grid": a_grid, "R_grid": r_grid, "A_radius": a_radius,
        "payload_tol": TOL, "codebook": CODEBOOK.tolist(), "amp_floor": CFG.amp_floor,
        "store": {"alpha": CFG.atom_alpha, "s": CFG.atom_width,
                  "kappa": CFG.payload_kappa, "amp": CFG.atom_amp, "cap": CAP},
        "read": {"dt": CFG.dt, "gamma_address": CFG.gamma_address,
                 "gamma_read": CFG.gamma_read, "address_steps": CFG.address_steps,
                 "read_steps": CFG.read_steps, "tail_frac": CFG.tail_frac,
                 "n_subsample": CFG.n_subsample},
    }, "panelA": {}, "panelB": {}, "lengths": {}}

    accA, accB, accL = {}, {}, {}
    for seed in seeds:
        rng_t = np.random.default_rng(4242 + seed)
        ang = rng_t.random(n_targets) * 2 * np.pi
        rad = MH.R_DISK * np.sqrt(rng_t.random(n_targets)) * 0.8
        for ti in range(n_targets):
            c_i = np.array([rad[ti] * np.cos(ang[ti]), rad[ti] * np.sin(ang[ti])])
            a_i = float(CODEBOOK[ti % CAP])
            bg_pay = np.array([CODEBOOK[(ti + 1 + j) % CAP] for j in range(MH.N_BG)])
            W = MH.build_worlds(seed, c_i, a_i, bg_pay, n_worlds)
            base_in, base_op, base_oh = MH._pack(W, "in"), MH._pack(W, "op"), MH._pack(W, "oh")
            ts = W["tslot"]

            for arm in arms:
                gate, g0, rm = ARMS[arm]
                rb = readers[arm]
                if "A" in panels:
                    Qn, Pn = MH.queries_native(seed, c_i, n_worlds)
                    a_op, v_op = rb(base_op, Qn, Pn)
                    s1o, s2o = MH.query_scores(a_op, v_op, c_i, a_i)
                    s4o, s5o = gated_probe(base_op, c_i, gate, g0)
                    a_oh, v_oh = rb(base_oh, Qn, Pn)
                    s1h, s2h = MH.query_scores(a_oh, v_oh, c_i, a_i)
                    s4h, s5h = gated_probe(base_oh, c_i, gate, g0)
                    for A in a_grid:
                        pk = MH.set_amp(base_in, ts, A)
                        ad, vd = rb(pk, Qn, Pn)
                        s1, s2 = MH.query_scores(ad, vd, c_i, a_i)
                        s4, s5 = gated_probe(pk, c_i, gate, g0)
                        ret = MH.retention(ad, vd, pk, ts, a_i)
                        err = float(np.abs(vd - a_i).mean())
                        k = (arm, f"{A:g}")
                        accA.setdefault(k, []).append({
                            "retention": float(ret.mean()), "val_err": err,
                            "a_i": a_i, "a2": a_i ** 2,
                            "auc_s1_paired": MH.auc(s1, s1o), "auc_s2_paired": MH.auc(s2, s2o),
                            "auc_s4_paired": MH.auc(s4, s4o), "auc_s5_paired": MH.auc(s5, s5o),
                            "auc_s1_hist": MH.auc(s1, s1h), "auc_s2_hist": MH.auc(s2, s2h),
                            "auc_s4_hist": MH.auc(s4, s4h), "auc_s5_hist": MH.auc(s5, s5h),
                            "tpr_s1": MH.lira_tpr(s1, s1o)["tpr@fpr0.01"],
                            "tpr_s4": MH.lira_tpr(s4, s4o)["tpr@fpr0.01"],
                        })
                if "B" in panels:
                    for r in r_grid:
                        Qr, Pr = MH.queries_ring(seed, c_i, n_worlds, r)
                        for A in a_radius:
                            pk = MH.set_amp(base_in, ts, A)
                            a_, v_ = rb(pk, Qr, Pr)
                            ret = MH.retention(a_, v_, pk, ts, a_i)
                            accB.setdefault((arm, f"A{A:g}", f"r{r:g}"), []).append(
                                {"retention": float(ret.mean()), "a_i": a_i})
                if "L" in panels:
                    Qn, Pn = MH.queries_native(seed, c_i, n_worlds)
                    for A in a_radius:
                        pk = MH.set_amp(base_in, ts, A)
                        ad, vd = rb(pk, Qn, Pn)
                        accL.setdefault((arm, f"A{A:g}"), []).append({
                            "val_err": float(np.abs(vd - a_i).mean()),
                            "val_err_med": float(np.median(np.abs(vd - a_i))),
                            "retention": float(MH.retention(ad, vd, pk, ts, a_i).mean()),
                            "a_i": a_i, "read_steps": _cfg(ARMS[arm][2]).read_steps,
                        })
                print(f"[{arm} seed {seed} target {ti} a={a_i:+.3f}] "
                      f"{time.time()-t0:.0f}s", flush=True)

    def dump(acc):
        out = {}
        for k, lst in acc.items():
            d = {"n": len(lst)}
            for f in lst[0]:
                d[f] = [float(np.mean([x[f] for x in lst])),
                        float(np.std([x[f] for x in lst]))]
                d[f + "_all"] = [float(x[f]) for x in lst]
            out["|".join(k)] = d
        return out

    res["panelA"], res["panelB"], res["lengths"] = dump(accA), dump(accB), dump(accL)
    res["meta"]["runtime_s"] = time.time() - t0
    with open(out_path, "w") as fh:
        json.dump(res, fh, indent=2)
    print("wrote", out_path, f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--arms", default="base,g05")
    ap.add_argument("--panels", default="AB")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    seeds = [0] if a.quick else [0, 1, 2]
    n_targets = 2 if a.quick else MH.N_TARGETS
    n_worlds = 16 if a.quick else MH.N_WORLDS
    a_grid = [1.0, 0.2, 0.06] if a.quick else [
        1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.12, 0.10, 0.08, 0.07, 0.06, 0.055, 0.051]
    r_grid = [0.212, 0.8, 1.2] if a.quick else MH.R_GRID
    a_radius = [1.0, 0.06] if a.quick else MH.A_RADIUS
    out = os.path.join(HERE, f"gate_mia{a.tag}{'_quick' if a.quick else ''}.json")
    run(a.arms.split(","), a.panels, seeds, n_targets, n_worlds,
        a_grid, r_grid, a_radius, out)
