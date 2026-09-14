"""mia-decay-measurement (w25) — retention + per-example MIA vs leak*t on one axis.

Analysis-only harness (results-analyst). Uses the SHIPPED store/controller/read
verbatim (`chlu.core.controller.Controller`, `AtomStorePotential`,
`exp_sequential_write.two_phase`) and adds only the adversary + scoring.
Nothing under `chlu/` is modified.

Outputs a single metrics JSON; every number in the report is re-derived from it.
"""
import argparse
import json
import os
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from chlu.config import get_default_config
from chlu.core.admission import disk_proposer
from chlu.core.controller import Controller, packing_bound_disk, radius_for_capacity
from chlu.core.memory_potentials import AtomStorePotential, designed_payloads
from chlu.experiments.exp_learned_memory import model_for
from chlu.experiments.exp_sequential_write import (
    effective_payload_tol,
    make_queries_at,
    two_phase,
)

DIM = 3
CAP = 8            # store capacity = 1 target + 7 background
N_BG = 7
N_TARGETS = 8
N_WORLDS = 128
NQ = 16
SEEDS = [0, 1, 2]

CFG = get_default_config().experiment_controller_mvp
D_SAFE = CFG.d_safe_mult * CFG.atom_width          # 1.54
R_DISK = radius_for_capacity(CAP, D_SAFE)          # 2.287
CODEBOOK = np.asarray(designed_payloads(CAP, seed=CFG.payload_seed))
TOL = effective_payload_tol(CFG, CODEBOOK)         # 0.05
A_FLOOR = CFG.amp_floor                            # 0.05
TAIL_IDX = np.linspace(int((1 - CFG.tail_frac) * CFG.read_steps),
                       CFG.read_steps - 1, CFG.n_subsample).astype(int)

TEMPLATE = AtomStorePotential(dim=DIM, capacity=CAP, alpha=CFG.atom_alpha,
                              s=CFG.atom_width, kappa=CFG.payload_kappa)

# amplitude ladder (tau = -ln A); dense near the floor (the "neither present nor
# absent" region the recon predicted)
A_GRID = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15,
          0.12, 0.10, 0.08, 0.07, 0.06, 0.055, 0.051]
A_RADIUS = [1.0, 0.5, 0.2, 0.1, 0.06]
R_GRID = [0.0, 0.106, 0.212, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4]
SIGMAS = [0.01, 0.03, 0.1, 0.3]


# ---------------------------------------------------------------------------
# batched read (the SHIPPED two-phase read, vmapped over stores)
# ---------------------------------------------------------------------------
def _build(centers, payloads, amps, active):
    return eqx.tree_at(
        lambda t: (t.centers, t.payloads, t.amps, t.active), TEMPLATE,
        (centers, payloads, amps, active))


def _read_one(centers, payloads, amps, active, Q0, P0):
    m = model_for(_build(centers, payloads, amps, active), DIM)
    addr_q, traj = two_phase(m, Q0, P0, CFG, CFG.gamma_address, CFG.gamma_read)
    pay_tail = traj[:, jnp.asarray(TAIL_IDX), 2]        # (NQ, n_sub)
    return addr_q, pay_tail


_READ = eqx.filter_jit(jax.vmap(_read_one, in_axes=(0, 0, 0, 0, 0, 0)))


def read_batch(worlds, Q0, P0, chunk=384):
    """worlds: dict of stacked (W,...) arrays. Returns addr (W,NQ,3), val (W,NQ)."""
    W = worlds["centers"].shape[0]
    A, V = [], []
    for i in range(0, W, chunk):
        sl = slice(i, min(i + chunk, W))
        a, pt = _READ(jnp.asarray(worlds["centers"][sl]),
                      jnp.asarray(worlds["payloads"][sl]),
                      jnp.asarray(worlds["amps"][sl]),
                      jnp.asarray(worlds["active"][sl]),
                      jnp.asarray(Q0[sl]), jnp.asarray(P0[sl]))
        A.append(np.asarray(a))
        V.append(np.asarray(pt).mean(-1))
    return np.concatenate(A), np.concatenate(V)


# ---------------------------------------------------------------------------
# world construction (shipped Controller)
# ---------------------------------------------------------------------------
def _proposer(radius):
    return lambda k, n: np.asarray(disk_proposer(radius, DIM)(k, n))[:, :2]


def build_worlds(seed, target_center, target_payload, bg_payloads, n_worlds):
    """IN / OUT-paired / OUT-history triples, n_worlds of each.

    IN: the target is offered FIRST (so it is always admitted at exactly c_i),
    then N_BG background offers go through the shipped admission gate (they get
    relocated away from the target -> the history channel).
    OUT-paired: the SAME final background sites, target absent (V differs only
    by the target atom).
    OUT-history: the same background offer sequence replayed with the target
    never offered (background lands elsewhere).
    """
    prop = _proposer(R_DISK)
    out = {k: [] for k in ("in_c", "in_p", "in_a", "in_m", "op_c", "op_p", "op_a",
                           "op_m", "oh_c", "oh_p", "oh_a", "oh_m")}
    tslots, n_live_in, n_live_oh = [], [], []
    for w in range(n_worlds):
        key = jax.random.PRNGKey(1_000_003 * (seed + 1) + w)
        k_prop, k_in, k_oh = jax.random.split(key, 3)
        proposals = np.asarray(disk_proposer(R_DISK, DIM)(k_prop, N_BG))[:, :2]

        # --- IN ---
        c_in = Controller(AtomStorePotential(dim=DIM, capacity=CAP, alpha=CFG.atom_alpha,
                                             s=CFG.atom_width, kappa=CFG.payload_kappa),
                          d_safe=D_SAFE, budget=CAP, amp=CFG.atom_amp,
                          n_candidates=CFG.n_relocation_candidates)
        c_in.offer(item_id=-1, q_new=target_center, payload=float(target_payload),
                   key=k_in, proposer=prop)
        tslot = c_in.live_slots()[0]
        kk = k_in
        for j in range(N_BG):
            kk, ko = jax.random.split(kk)
            c_in.offer(item_id=j, q_new=proposals[j], payload=float(bg_payloads[j]),
                       key=ko, proposer=prop)
        st = c_in.store
        out["in_c"].append(np.asarray(st.centers))
        out["in_p"].append(np.asarray(st.payloads))
        out["in_a"].append(np.asarray(st.amps))
        out["in_m"].append(np.asarray(st.active))
        tslots.append(tslot)
        n_live_in.append(c_in.n_live)

        # --- OUT paired: identical background, target slot cleared ---
        op = st.evict(tslot)
        out["op_c"].append(np.asarray(op.centers))
        out["op_p"].append(np.asarray(op.payloads))
        out["op_a"].append(np.asarray(op.amps))
        out["op_m"].append(np.asarray(op.active))

        # --- OUT history: replay the background sequence, no target ---
        c_oh = Controller(AtomStorePotential(dim=DIM, capacity=CAP, alpha=CFG.atom_alpha,
                                             s=CFG.atom_width, kappa=CFG.payload_kappa),
                          d_safe=D_SAFE, budget=CAP, amp=CFG.atom_amp,
                          n_candidates=CFG.n_relocation_candidates)
        kk = k_in
        for j in range(N_BG):
            kk, ko = jax.random.split(kk)
            c_oh.offer(item_id=j, q_new=proposals[j], payload=float(bg_payloads[j]),
                       key=ko, proposer=prop)
        sh = c_oh.store
        out["oh_c"].append(np.asarray(sh.centers))
        out["oh_p"].append(np.asarray(sh.payloads))
        out["oh_a"].append(np.asarray(sh.amps))
        out["oh_m"].append(np.asarray(sh.active))
        n_live_oh.append(c_oh.n_live)

    W = {k: np.stack(v) for k, v in out.items()}
    W["tslot"] = np.array(tslots)
    W["n_live_in"] = np.array(n_live_in)
    W["n_live_oh"] = np.array(n_live_oh)
    return W


def _pack(W, pre):
    return {"centers": W[pre + "_c"], "payloads": W[pre + "_p"],
            "amps": W[pre + "_a"], "active": W[pre + "_m"]}


def set_amp(pack, tslot, A):
    p = {k: v.copy() for k, v in pack.items()}
    p["amps"][np.arange(len(tslot)), tslot] = A
    return p


# ---------------------------------------------------------------------------
# queries
# ---------------------------------------------------------------------------
def queries_native(seed, c_i, n_worlds):
    """The SHIPPED query distribution at the target site (sigma=0.15 jitter)."""
    site = np.zeros((1, DIM)); site[0, :2] = c_i
    Q, P = [], []
    for w in range(n_worlds):
        q, p, _ = make_queries_at(jax.random.PRNGKey(7_777_777 + 31 * seed + w),
                                  site, NQ, CFG, dim=DIM)
        Q.append(np.asarray(q)); P.append(np.asarray(p))
    return np.stack(Q), np.stack(P)


def queries_ring(seed, c_i, n_worlds, r):
    Q = np.zeros((n_worlds, NQ, DIM)); P = np.zeros((n_worlds, NQ, DIM))
    rng = np.random.default_rng(90_000 + 13 * seed)
    for w in range(n_worlds):
        th = 2 * np.pi * (np.arange(NQ) / NQ + rng.random())
        Q[w, :, 0] = c_i[0] + r * np.cos(th)
        Q[w, :, 1] = c_i[1] + r * np.sin(th)
    return Q, P


# ---------------------------------------------------------------------------
# adversary statistics
# ---------------------------------------------------------------------------
def query_scores(addr, val, c_i, a_i):
    """s1 value-return, s2 address-capture (per world, mean over queries)."""
    s1 = -np.abs(val - a_i).mean(1)
    s2 = -np.linalg.norm(addr[:, :, :2] - c_i[None, None, :], axis=-1).mean(1)
    return s1, s2


def retention(addr, val, pack, tslot, a_i):
    """The SHIPPED strict criterion: own basin among live sites AND |v-a|<tol."""
    d = np.linalg.norm(addr[:, :, None, :2] - pack["centers"][:, None, :, :], axis=-1)
    d = np.where(pack["active"][:, None, :] > 0, d, np.inf)
    basin_ok = d.argmin(-1) == tslot[:, None]
    return (basin_ok & (np.abs(val - a_i) < TOL)).mean(1)


def _probe_dirs(rho):
    th = 2 * np.pi * np.arange(8) / 8
    return rho * np.stack([np.cos(th), np.sin(th)], 1)      # (8,2)


def potential_probe(pack, c_i, rho=None, noise=None, rng=None):
    """s4 (address-channel depth) and s5 (full V at q2=0), both at scale rho.

    s4 = mean_dirs U(c+d) - U(c) with U = alpha|q|^2 - sum m A exp(-d^2/2s^2)
    s5 = same with the shipped V (which adds 0.5 kappa (0 - S(q))^2).
    `noise` (sigma) corrupts EVERY probe evaluation independently (TM-3).
    """
    rho = CFG.atom_width if rho is None else rho
    pts = np.concatenate([c_i[None, :], c_i[None, :] + _probe_dirs(rho)])   # (9,2)
    d2 = ((pts[None, :, None, :] - pack["centers"][:, None, :, :]) ** 2).sum(-1)
    m = pack["active"][:, None, :]
    w = m * np.exp(-d2 / (2 * CFG.atom_width ** 2))
    wpay = m * np.exp(-d2 / (2 * CFG.atom_width ** 2))
    U = CFG.atom_alpha * (pts ** 2).sum(-1)[None, :] - (pack["amps"][:, None, :] * w).sum(-1)
    S = (pack["payloads"][:, None, :] * wpay).sum(-1)
    V = U + 0.5 * CFG.payload_kappa * S ** 2
    if noise:
        U = U + rng.normal(0, noise, U.shape)
        V = V + rng.normal(0, noise, V.shape)
    s4 = U[:, 1:].mean(1) - U[:, 0]
    s5 = V[:, 1:].mean(1) - V[:, 0]
    return s4, s5


def hole_stat(pack, c_i):
    d = np.linalg.norm(pack["centers"] - c_i[None, None, :], axis=-1)
    d = np.where(pack["active"] > 0, d, np.inf)
    return d.min(1)


# ---------------------------------------------------------------------------
# per-example U-LiRA scoring
# ---------------------------------------------------------------------------
def auc(x_in, x_out):
    """Mann-Whitney AUC with tie-correction (0.5 for a fully tied pair set)."""
    x_in = np.asarray(x_in, float); x_out = np.asarray(x_out, float)
    n1, n0 = len(x_in), len(x_out)
    allx = np.concatenate([x_in, x_out])
    order = allx.argsort(kind="mergesort")
    ranks = np.empty(len(allx)); ranks[order] = np.arange(1, len(allx) + 1)
    # average ranks over ties
    s = np.sort(allx); i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[np.isin(allx, s[i])] = (i + j + 2) / 2.0
        i = j + 1
    r1 = ranks[:n1].sum()
    return float((r1 - n1 * (n1 + 1) / 2) / (n1 * n0))


def lira_tpr(x_in, x_out, fprs=(0.05, 0.01)):
    """Per-example LiRA: Gaussian fits to the IN/OUT score distributions, ROC
    from the log-likelihood-ratio statistic."""
    mi, si = float(np.mean(x_in)), float(np.std(x_in)) + 1e-12
    mo, so = float(np.mean(x_out)), float(np.std(x_out)) + 1e-12

    def llr(x):
        return (-0.5 * ((x - mi) / si) ** 2 - np.log(si)) - (
            -0.5 * ((x - mo) / so) ** 2 - np.log(so))

    li, lo = llr(np.asarray(x_in, float)), llr(np.asarray(x_out, float))
    out = {}
    for f in fprs:
        thr = np.quantile(lo, 1 - f)
        out[f"tpr@fpr{f}"] = float((li > thr).mean())
    return out


def per_example(x_in, x_out):
    d = {"auc": auc(x_in, x_out)}
    d.update(lira_tpr(x_in, x_out))
    return d


def agg(vals):
    v = np.asarray([x for x in vals if np.isfinite(x)], float)
    if not len(v):
        return [float("nan")] * 2
    return [float(v.mean()), float(v.std())]


# ---------------------------------------------------------------------------
# TTL vector-store control (the trivial substitute)
# ---------------------------------------------------------------------------
def ttl_dict_line(seed, c_i, a_i, bg_sites, bg_pays, r_grid, R_lookup, present):
    """Nearest-neighbour dict with row-delete. `present`: is the target's row live?

    Retention: nearest live key is the target AND its value matches within TOL.
    MIA score: -|returned - a_i| (penalty 1.0 when nothing is within R_lookup).
    Returns per-radius (retention, s1) over worlds.
    """
    ret, sc = [], []
    for r in r_grid:
        th = 2 * np.pi * np.arange(NQ) / NQ
        q = c_i[None, :] + r * np.stack([np.cos(th), np.sin(th)], 1)   # (NQ,2)
        keys = np.concatenate([c_i[None, :], bg_sites]) if present else bg_sites
        vals = np.concatenate([[a_i], bg_pays]) if present else bg_pays
        d = np.linalg.norm(q[:, None, :] - keys[None, :, :], axis=-1)
        nn = d.argmin(1)
        hit = d.min(1) <= R_lookup
        got = np.where(hit, vals[nn], np.nan)
        is_t = present & (nn == 0) & hit
        ret.append(float((is_t & (np.abs(np.nan_to_num(got, nan=9.9) - a_i) < TOL)).mean()))
        pen = np.where(np.isnan(got), 1.0, np.abs(np.nan_to_num(got) - a_i))
        sc.append(-float(pen.mean()))
    return ret, sc


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def run(out_path, quick=False):
    t_start = time.time()
    n_worlds = 16 if quick else N_WORLDS
    n_targets = 2 if quick else N_TARGETS
    seeds = [0] if quick else SEEDS
    a_grid = [1.0, 0.2, 0.06] if quick else A_GRID
    r_grid = [0.212, 0.8, 1.2] if quick else R_GRID
    a_radius = [1.0, 0.1] if quick else A_RADIUS

    res = {
        "meta": {
            "commit": os.popen("git rev-parse --short HEAD").read().strip(),
            "jax": jax.__version__,
            "seeds": seeds, "n_worlds": n_worlds, "n_targets": n_targets,
            "n_query": NQ, "cap": CAP, "n_bg": N_BG,
            "d_safe": D_SAFE, "disk_radius": R_DISK,
            "packing_bound": packing_bound_disk(R_DISK, D_SAFE),
            "payload_tol": TOL, "amp_floor": A_FLOOR,
            "codebook": CODEBOOK.tolist(),
            "A_grid": a_grid, "R_grid": r_grid, "A_radius": a_radius,
            "sigmas": SIGMAS,
            "read": {"dt": CFG.dt, "gamma_address": CFG.gamma_address,
                     "gamma_read": CFG.gamma_read, "address_steps": CFG.address_steps,
                     "read_steps": CFG.read_steps, "tail_frac": CFG.tail_frac,
                     "n_subsample": CFG.n_subsample},
            "store": {"alpha": CFG.atom_alpha, "s": CFG.atom_width,
                      "kappa": CFG.payload_kappa, "amp": CFG.atom_amp},
            "quick": quick,
        },
        "panelA": {}, "panelB": {}, "postevict": {}, "tm3": {}, "ttl": {},
        "decay_law": {}, "dump": {},
    }

    # ---- P9: the shipped decay law, run through Controller.tick ------------
    ctl = Controller(AtomStorePotential(dim=DIM, capacity=4, alpha=CFG.atom_alpha,
                                        s=CFG.atom_width, kappa=CFG.payload_kappa),
                     d_safe=D_SAFE, budget=4, amp=1.0, leak=0.35,
                     amp_floor=A_FLOOR, n_candidates=CFG.n_relocation_candidates)
    ctl.offer(item_id=0, q_new=np.array([0.0, 0.0]), payload=1.0,
              key=jax.random.PRNGKey(0), proposer=_proposer(R_DISK))
    amps_seen, live_seen = [], []
    for t in range(12):
        amps_seen.append(float(np.asarray(ctl.store.amps)[0]))
        live_seen.append(ctl.n_live)
        ctl.tick()
    res["decay_law"] = {
        "leak": 0.35, "amps_by_tick": amps_seen, "n_live_by_tick": live_seen,
        "exp_law": [float(np.exp(-0.35 * t)) for t in range(12)],
        "evict_tick": int(np.argmax(np.array(live_seen) == 0)) if 0 in live_seen else None,
    }

    per_ex = {}   # (metric, arm, level) -> list of per-example dicts
    def rec(key, d):
        per_ex.setdefault(key, []).append(d)

    ttl_acc = {}
    postA = {}
    for seed in seeds:
        rng_t = np.random.default_rng(4242 + seed)
        # target sites: drawn in the same disk, distinct per target
        ang = rng_t.random(n_targets) * 2 * np.pi
        rad = R_DISK * np.sqrt(rng_t.random(n_targets)) * 0.8
        for ti in range(n_targets):
            c_i = np.array([rad[ti] * np.cos(ang[ti]), rad[ti] * np.sin(ang[ti])])
            a_i = float(CODEBOOK[ti % CAP])
            bg_pay = np.array([CODEBOOK[(ti + 1 + j) % CAP] for j in range(N_BG)])
            W = build_worlds(seed, c_i, a_i, bg_pay, n_worlds)
            base_in, base_op, base_oh = _pack(W, "in"), _pack(W, "op"), _pack(W, "oh")
            ts = W["tslot"]

            # TM-4: does evict erase the data structure?
            dump_c = base_op["centers"][np.arange(n_worlds), ts]
            dump_p = base_op["payloads"][np.arange(n_worlds), ts]
            res["dump"].setdefault("center_max_err", []).append(
                float(np.abs(dump_c - c_i[None, :]).max()))
            res["dump"].setdefault("payload_max_err", []).append(
                float(np.abs(dump_p - a_i).max()))
            res["dump"].setdefault("amp_after_evict", []).append(
                float(np.abs(base_op["amps"][np.arange(n_worlds), ts]).max()))
            res["dump"].setdefault("active_after_evict", []).append(
                float(np.abs(base_op["active"][np.arange(n_worlds), ts]).max()))

            # ---------------- Panel A: native queries, amplitude ladder ------
            Qn, Pn = queries_native(seed, c_i, n_worlds)
            addr_op, val_op = read_batch(base_op, Qn, Pn)
            addr_oh, val_oh = read_batch(base_oh, Qn, Pn)
            s1_op, s2_op = query_scores(addr_op, val_op, c_i, a_i)
            s1_oh, s2_oh = query_scores(addr_oh, val_oh, c_i, a_i)
            s4_op, s5_op = potential_probe(base_op, c_i)
            s4_oh, s5_oh = potential_probe(base_oh, c_i)

            for A in a_grid:
                pk = set_amp(base_in, ts, A)
                addr, val = read_batch(pk, Qn, Pn)
                s1, s2 = query_scores(addr, val, c_i, a_i)
                s4, s5 = potential_probe(pk, c_i)
                ret = retention(addr, val, pk, ts, a_i)
                lvl = f"{A:g}"
                rec(("retention", lvl), {"auc": float(ret.mean())})
                for nm, xin, xout in (("s1", s1, s1_op), ("s2", s2, s2_op),
                                      ("s4", s4, s4_op), ("s5", s5, s5_op)):
                    rec((f"stat_{nm}", lvl), {
                        "auc": float(xin.mean()),          # IN mean (field reused)
                        "in_mean": float(xin.mean()), "in_std": float(xin.std()),
                        "out_mean": float(xout.mean()), "out_std": float(xout.std()),
                        "gap": float(xin.mean() - xout.mean()),
                        "dprime": float((xin.mean() - xout.mean())
                                        / (xout.std() + 1e-12)),
                    })
                for nm, xin, xout in (("s1", s1, s1_op), ("s2", s2, s2_op),
                                      ("s4", s4, s4_op), ("s5", s5, s5_op)):
                    rec((f"paired/{nm}", lvl), per_example(xin, xout))
                for nm, xin, xout in (("s1", s1, s1_oh), ("s2", s2, s2_oh),
                                      ("s4", s4, s4_oh), ("s5", s5, s5_oh)):
                    rec((f"history/{nm}", lvl), per_example(xin, xout))
                # TM-3: resolution-limited probe
                for sg in SIGMAS:
                    rng = np.random.default_rng(hash((seed, ti, A, sg)) % (2 ** 31))
                    n4, _ = potential_probe(pk, c_i, noise=sg, rng=rng)
                    n4o, _ = potential_probe(base_op, c_i, noise=sg, rng=rng)
                    rec((f"tm3/{sg}", lvl), per_example(n4, n4o))

            # ---------------- post-evict --------------------------------
            addr_pe, val_pe = read_batch(base_op, Qn, Pn)   # IN after evict == op
            s1_pe, s2_pe = query_scores(addr_pe, val_pe, c_i, a_i)
            ret_pe = retention(addr_pe, val_pe, base_op, ts, a_i)
            postA.setdefault("retention", []).append(float(ret_pe.mean()))
            for nm, xin, xout in (("s1", s1_pe, s1_op), ("s2", s2_pe, s2_op),
                                  ("s4", s4_op, s4_op), ("s5", s5_op, s5_op)):
                rec((f"postevict_paired/{nm}", "evicted"), per_example(xin, xout))
            for nm, xin, xout in (("s1", s1_pe, s1_oh), ("s2", s2_pe, s2_oh),
                                  ("s4", s4_op, s4_oh), ("s5", s5_op, s5_oh)):
                rec((f"postevict_history/{nm}", "evicted"), per_example(xin, xout))
            rec(("postevict_history/hole", "evicted"),
                per_example(hole_stat(base_op, c_i), hole_stat(base_oh, c_i)))
            rec(("postevict_history/n_live", "evicted"),
                per_example(W["n_live_in"] - 1, W["n_live_oh"]))
            rec(("postevict_paired/hole", "evicted"),
                per_example(hole_stat(base_op, c_i), hole_stat(base_op, c_i)))

            # ---------------- Panel B: radius sweep -----------------------
            for r in r_grid:
                Qr, Pr = queries_ring(seed, c_i, n_worlds, r)
                a_o, v_o = read_batch(base_op, Qr, Pr)
                s1o, s2o = query_scores(a_o, v_o, c_i, a_i)
                for A in a_radius:
                    pk = set_amp(base_in, ts, A)
                    a_, v_ = read_batch(pk, Qr, Pr)
                    s1_, s2_ = query_scores(a_, v_, c_i, a_i)
                    ret = retention(a_, v_, pk, ts, a_i)
                    lvl = f"A{A:g}|r{r:g}"
                    rec(("radius/retention", lvl), {"auc": float(ret.mean())})
                    rec(("radius/mia_s1", lvl), per_example(s1_, s1o))
                    rec(("radius/mia_s2", lvl), per_example(s2_, s2o))

            # ---------------- TTL vector-store control --------------------
            bg_sites = base_in["centers"][0][base_in["active"][0] > 0]
            bg_p = base_in["payloads"][0][base_in["active"][0] > 0]
            keep = ~np.all(np.isclose(bg_sites, c_i[None, :], atol=1e-6), axis=1)
            for present in (True, False):
                rt, sc = ttl_dict_line(seed, c_i, a_i, bg_sites[keep], bg_p[keep],
                                       r_grid, R_lookup=D_SAFE / 2, present=present)
                ttl_acc.setdefault(f"present{int(present)}_ret", []).append(rt)
                ttl_acc.setdefault(f"present{int(present)}_s1", []).append(sc)
            print(f"[seed {seed} target {ti}] done  t={time.time()-t_start:.0f}s",
                  flush=True)

    # ------------- aggregate over the 24 per-example values -----------------
    agg_out = {}
    for (metric, lvl), lst in per_ex.items():
        d = {"n_examples": len(lst)}
        for field in lst[0]:
            d[field] = agg([x[field] for x in lst])
            d[field + "_all"] = [float(x[field]) for x in lst]
        agg_out.setdefault(metric, {})[lvl] = d
    res["per_example"] = agg_out
    res["postevict"]["retention_mean"] = agg(postA.get("retention", []))
    res["ttl"] = {k: np.asarray(v).mean(0).tolist() for k, v in ttl_acc.items()}
    res["ttl"]["r_grid"] = r_grid
    res["ttl"]["R_lookup"] = D_SAFE / 2
    res["meta"]["runtime_s"] = time.time() - t_start
    with open(out_path, "w") as fh:
        json.dump(res, fh, indent=2)
    print("wrote", out_path, f"({time.time()-t_start:.0f}s)")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=".claude/outputs/mia-decay-measurement/mia_metrics.json")
    a = ap.parse_args()
    run(a.out if not a.quick else a.out.replace(".json", "_quick.json"), quick=a.quick)
