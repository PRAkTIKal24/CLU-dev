"""One-step training for the GYM-2 baselines, the free-running policies, and the ORACLES.

PERSISTENCE AT ONE STEP -- how it is enforced (inherited verbatim from w5/gym1, which
found it caught a failure that goal-reach hides):

  1. The loss is cross-entropy on the immediate next acceleration and nothing else. No
     multi-step term, no rollout inside the objective, no scheduled sampling. The model is
     never given credit or blame for anything beyond one step.
  2. The supervised target is the provably optimal path plus `N_HOLD = 4` steps of "stay
     parked at the goal". Four. The evaluation horizon is 48 and the persistence probe is
     96, so holding the goal for 44 or 92 further steps is never supervised.
  3. At rollout there is NO action mask, NO clamp of v to the speed limit, NO projection of
     the agent back onto the grid while the trajectory is still clean. The only thing
     between the model and a violation is the model. `grid2d.verify_unmasked_rollout`
     proves this in code rather than asserting it in prose.

THE RULE-BLIND ORACLES (pre-registration P4) live here too, because they are policies.
Each plans optimally with one rule deleted or relaxed and is then replayed into the TRUE
environment. A rule whose deletion costs nothing is a rule the gym does not test.
"""

from __future__ import annotations

import os
import pickle
import time
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from chlu.gym2 import episodes as EP, layouts as LY
from chlu.gym2.grid2d import ACTIONS, G, N_ACTIONS, Episode, get_layout, plan
from chlu.gym2.models import PAD, PATCH, V_EMB, count_params

N_HOLD = 4
MAX_LEN = EP.MAX_PATH + N_HOLD
_AIDX = {a: i for i, a in enumerate(ACTIONS)}


# ---------------------------------------------------------------------------
# Obstacle-map features. Both are exact views of the true map, never a summary.
# ---------------------------------------------------------------------------

def map_flat(lid: int) -> np.ndarray:
    return get_layout(lid).blocked.reshape(-1).astype(np.float32)


def padded_map(lid: int) -> np.ndarray:
    """(G+2P, G+2P) float, 1 = blocked. Outside the grid counts as blocked."""
    m = np.ones((G + 2 * PAD, G + 2 * PAD), dtype=np.float32)
    m[PAD:PAD + G, PAD:PAD + G] = get_layout(lid).blocked.astype(np.float32)
    return m


_PATCH_DI = np.arange(PATCH)


def patches_for(padmaps: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """(..., PATCH*PATCH) egocentric obstacle windows, gathered exactly."""
    xi = np.clip(x, -PAD, G - 1 + PAD)[..., None, None] + _PATCH_DI[None, :, None]
    yi = np.clip(y, -PAD, G - 1 + PAD)[..., None, None] + _PATCH_DI[None, None, :]
    idx = np.arange(padmaps.shape[0]).reshape((-1,) + (1,) * (xi.ndim - 1))
    return padmaps[idx, xi, yi].reshape(*x.shape, PATCH * PATCH)


# ---------------------------------------------------------------------------

def build_arrays(records, max_len=MAX_LEN):
    """Teacher-forced one-step dataset. Each position is an independent one-step target."""
    n = len(records)
    x = np.zeros((n, max_len), dtype=np.int32)
    y = np.zeros((n, max_len), dtype=np.int32)
    vx = np.zeros((n, max_len), dtype=np.int32)
    vy = np.zeros((n, max_len), dtype=np.int32)
    tgt = np.zeros((n, max_len), dtype=np.int32)
    m = np.zeros((n, max_len), dtype=np.float32)
    gx = np.zeros(n, dtype=np.int32)
    gy = np.zeros(n, dtype=np.int32)
    V = np.zeros(n, dtype=np.int32)
    lid = np.zeros(n, dtype=np.int32)

    for i, r in enumerate(records):
        ep = r.ep
        xs = list(r.positions)
        vs = [(ep.vx0, ep.vy0)]
        for ax, ay in r.actions:
            vs.append((vs[-1][0] + ax, vs[-1][1] + ay))
        T = len(r.actions)
        hold = min(N_HOLD, max_len - T)
        for t in range(T):
            x[i, t], y[i, t] = xs[t]
            vx[i, t], vy[i, t] = vs[t]
            tgt[i, t] = _AIDX[r.actions[t]]
            m[i, t] = 1.0
        for t in range(T, T + hold):        # stay parked: the one-step persistence rule
            x[i, t], y[i, t] = ep.gx, ep.gy
            vx[i, t] = vy[i, t] = 0
            tgt[i, t] = _AIDX[(0, 0)]
            m[i, t] = 1.0
        gx[i], gy[i], V[i], lid[i] = ep.gx, ep.gy, ep.V, ep.lid
    return dict(x=x, y=y, vx=vx, vy=vy, tgt=tgt, m=m, gx=gx, gy=gy, V=V, lid=lid)


def make_feature_tables(lids):
    """Per-layout map vectors and padded maps, indexed by a compact table id."""
    uniq = sorted(set(int(l) for l in lids))
    remap = {l: i for i, l in enumerate(uniq)}
    maps = np.stack([map_flat(l) for l in uniq], 0)
    pads = np.stack([padded_map(l) for l in uniq], 0)
    return remap, maps, pads


def batch_tokens(data, idx, remap, maps, pads):
    """Token tuple for a mini-batch, including both obstacle-map features."""
    x, y = data["x"][idx], data["y"][idx]
    vx, vy = data["vx"][idx], data["vy"][idx]
    T = x.shape[1]
    li = np.array([remap[int(l)] for l in data["lid"][idx]], dtype=np.int64)
    patch = patches_for(pads[li], x, y)
    tile = lambda z: np.repeat(z[:, None], T, axis=1)
    return (jnp.asarray(np.clip(x, 0, G - 1)), jnp.asarray(np.clip(y, 0, G - 1)),
            jnp.asarray(np.clip(vx, -V_EMB, V_EMB) + V_EMB),
            jnp.asarray(np.clip(vy, -V_EMB, V_EMB) + V_EMB),
            jnp.asarray(tile(data["gx"][idx])), jnp.asarray(tile(data["gy"][idx])),
            jnp.asarray(tile(data["V"][idx])),
            jnp.asarray(np.repeat(maps[li][:, None, :], T, axis=1)),
            jnp.asarray(patch))


def loss_fn(model, toks, tgt, mask):
    logits = jax.vmap(model)(toks)
    ll = jax.nn.log_softmax(logits, axis=-1)
    picked = jnp.take_along_axis(ll, tgt[..., None], axis=-1)[..., 0]
    return -(picked * mask).sum() / jnp.maximum(mask.sum(), 1.0)


@eqx.filter_jit
def _train_step(model, opt_state, toks, tgt, mask, opt):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, toks, tgt, mask)
    updates, opt_state = opt.update(grads, opt_state,
                                    eqx.filter(model, eqx.is_inexact_array))
    return eqx.apply_updates(model, updates), opt_state, loss


class TrainLog(NamedTuple):
    steps: int
    seconds: float
    losses: list
    params: int


def train_model(model, data, remap, maps, pads, steps=6000, batch=64, lr=3e-4, seed=0,
                log_every=250, ckpt_path=None, ckpt_every=500, time_budget_s=None):
    opt = optax.chain(optax.clip_by_global_norm(1.0),
                      optax.adamw(optax.cosine_decay_schedule(lr, steps),
                                  weight_decay=1e-4))
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))
    rng = np.random.default_rng(seed)
    n = data["x"].shape[0]
    losses = []
    t0 = time.time()
    for step in range(steps):
        idx = rng.integers(0, n, size=batch)
        toks = batch_tokens(data, idx, remap, maps, pads)
        model, opt_state, loss = _train_step(
            model, opt_state, toks, jnp.asarray(data["tgt"][idx]),
            jnp.asarray(data["m"][idx]), opt)
        losses.append(float(loss))
        if (step + 1) % log_every == 0:
            print("  step %5d/%d  loss %.5f  (%.0fs)"
                  % (step + 1, steps, float(np.mean(losses[-log_every:])),
                     time.time() - t0), flush=True)
        if ckpt_path and (step + 1) % ckpt_every == 0:
            eqx.tree_serialise_leaves(ckpt_path + ".eqx", model)
        if time_budget_s and time.time() - t0 > time_budget_s:
            print(f"[stop] wall-clock budget {time_budget_s}s reached at step {step+1}",
                  flush=True)
            break
    if ckpt_path:
        eqx.tree_serialise_leaves(ckpt_path + ".eqx", model)
    return model, TrainLog(steps, time.time() - t0, losses, count_params(model))


# ---------------------------------------------------------------------------
# Free-running rollout policies. NO guardrails.
# ---------------------------------------------------------------------------

def _obs_inputs(P, remap, maps, pads, x, y):
    li = np.array([remap[int(l)] for l in P["lid"]], dtype=np.int64)
    return maps[li], patches_for(pads[li], np.asarray(x), np.asarray(y))


def lstm_policy(model, n, remap, maps, pads):
    state = [(jnp.zeros((n, model.h)), jnp.zeros((n, model.h)))]

    @eqx.filter_jit
    def _step(model, st, toks):
        def one(sh, sc, *tk):
            ns, logits = model.step((sh, sc), tk)
            return ns, logits
        return jax.vmap(one)(st[0], st[1], *toks)

    def pol(t, x, y, vx, vy, P):
        mp, pt = _obs_inputs(P, remap, maps, pads, x, y)
        toks = (jnp.asarray(np.clip(x, 0, G - 1), dtype=jnp.int32),
                jnp.asarray(np.clip(y, 0, G - 1), dtype=jnp.int32),
                jnp.asarray(np.clip(vx, -V_EMB, V_EMB) + V_EMB, dtype=jnp.int32),
                jnp.asarray(np.clip(vy, -V_EMB, V_EMB) + V_EMB, dtype=jnp.int32),
                jnp.asarray(P["gx"], dtype=jnp.int32), jnp.asarray(P["gy"], dtype=jnp.int32),
                jnp.asarray(P["V"], dtype=jnp.int32),
                jnp.asarray(mp), jnp.asarray(pt))
        ns, logits = _step(model, state[0], toks)
        state[0] = ns
        return np.array(ACTIONS, dtype=np.int64)[np.asarray(jnp.argmax(logits, axis=-1))]

    pol._unmasked = True
    return pol


def transformer_policy(model, n, horizon, remap, maps, pads):
    """Re-runs the causal prefix each step; keeps the history and nothing else.

    The buffer is a fixed `horizon`-long array rather than a growing prefix. Attention is
    causal, so position t cannot see anything after t and the still-unwritten tail cannot
    influence the logits read at t -- the answer is identical to running the ragged prefix,
    and jax compiles the forward pass once instead of once per step.
    """
    bx = np.zeros((n, horizon), dtype=np.int32)
    by = np.zeros((n, horizon), dtype=np.int32)
    bvx = np.zeros((n, horizon), dtype=np.int32)
    bvy = np.zeros((n, horizon), dtype=np.int32)
    bpa = np.zeros((n, horizon, PATCH * PATCH), dtype=np.float32)

    @eqx.filter_jit
    def _fwd(model, toks, t):
        logits = jax.vmap(model)(toks)
        return jnp.take(logits, t, axis=1)

    def pol(t, x, y, vx, vy, P):
        mp, pt = _obs_inputs(P, remap, maps, pads, x, y)
        bx[:, t] = np.clip(x, 0, G - 1)
        by[:, t] = np.clip(y, 0, G - 1)
        bvx[:, t] = np.clip(vx, -V_EMB, V_EMB) + V_EMB
        bvy[:, t] = np.clip(vy, -V_EMB, V_EMB) + V_EMB
        bpa[:, t] = pt
        T = horizon
        tile = lambda z: jnp.repeat(jnp.asarray(z)[:, None], T, axis=1)
        toks = (jnp.asarray(bx), jnp.asarray(by), jnp.asarray(bvx), jnp.asarray(bvy),
                tile(P["gx"]), tile(P["gy"]), tile(P["V"]),
                jnp.repeat(jnp.asarray(mp)[:, None, :], T, axis=1), jnp.asarray(bpa))
        logits = _fwd(model, toks, jnp.asarray(t))
        return np.array(ACTIONS, dtype=np.int64)[np.asarray(jnp.argmax(logits, axis=-1))]

    pol._unmasked = True
    return pol


def replay_batch_policy(seqs):
    """A list of fixed action sequences as a batched policy."""
    def pol(t, x, y, vx, vy, P):
        return np.array([s[t] if t < len(s) else (0, 0) for s in seqs], dtype=np.int64)
    pol._unmasked = True
    return pol


# ---------------------------------------------------------------------------
# GATE O -- the rule-blind oracles (pre-registration P4).
# ---------------------------------------------------------------------------

def oracle_optimal(records):
    """Control: the provably optimal plan, replayed. Should score 1.0 by construction."""
    return [list(r.actions) for r in records]


def oracle_obstacle_blind(records):
    """O1 -- plan with EVERY obstacle deleted, then replay into the true field."""
    from chlu.gym2.grid2d import register_layout
    import numpy as _np
    free_lid = register_layout(_np.zeros((G, G), dtype=bool), {"purpose": "obstacle_free"})
    out = []
    for r in records:
        e = r.ep
        p = plan(Episode(free_lid, e.V, e.sx, e.sy, e.vx0, e.vy0, e.gx, e.gy))
        out.append(list(p) if p else [(0, 0)])
    return out


def oracle_speed_blind(records, v_max=None):
    """O2 -- plan under the loosest speed limit, then replay under the episode's true V."""
    vmax = v_max or max(LY.V_CHOICES)
    out = []
    for r in records:
        e = r.ep
        p = plan(Episode(e.lid, vmax, e.sx, e.sy,
                         int(np.clip(e.vx0, -vmax, vmax)),
                         int(np.clip(e.vy0, -vmax, vmax)), e.gx, e.gy))
        out.append(list(p) if p else [(0, 0)])
    return out


def oracle_composition_blind(records, v_max=None):
    """O3 -- THE decisive oracle for wave 7.

    Choose the ROUTE (tunnel or detour) that is optimal at `V = V_MAX`, then re-plan
    OPTIMALLY under the episode's true `V` restricted to that route. Speed limits and
    obstacles are both fully respected; the only thing blinded is the *interaction* -- the
    fact that which route is cheaper depends on how fast you are allowed to go.

    If O3 scores ~1.0 and its cost equals the unrestricted optimum, then the route does not
    depend on the speed limit and the composition wave 7 wants to test is empty, even
    though O1 and O2 both fail loudly. This is the 2-D analogue of the parent's wall-blind
    oracle, and it is the gate GYM-1 would have failed one dimension up.
    """
    vmax = v_max or max(LY.V_CHOICES)
    out, cost_gap, n_route_differs, n_posed = [], [], 0, 0
    for r in records:
        e = r.ep
        meta = get_layout(e.lid).meta
        if not r.crossing or meta.get("sealed"):
            out.append(list(r.actions))
            continue
        vx0 = int(np.clip(e.vx0, -vmax, vmax))
        vy0 = int(np.clip(e.vy0, -vmax, vmax))
        big = Episode(e.lid, vmax, e.sx, e.sy, vx0, vy0, e.gx, e.gy)
        from chlu.gym2.grid2d import opt_cost
        ct_big = opt_cost(big, meta["block_top"])
        cd_big = opt_cost(big, meta["block_tunnel"])
        if ct_big < 0 and cd_big < 0:
            out.append(list(r.actions))
            continue
        pick_tunnel = ct_big >= 0 and (cd_big < 0 or ct_big <= cd_big)
        extra = meta["block_top"] if pick_tunnel else meta["block_tunnel"]
        p = plan(e, block_extra=extra)
        n_posed += 1
        if p is None:
            out.append([(0, 0)])
            continue
        out.append(list(p))
        cost_gap.append(len(p) - r.cost)
        true_route = r.route
        if (true_route == "tunnel") != bool(pick_tunnel):
            n_route_differs += 1
    stats = {
        "n_route_decisions": n_posed,
        "n_route_choice_differs_from_true_optimum": n_route_differs,
        "frac_route_differs": n_route_differs / max(n_posed, 1),
        "mean_extra_steps_vs_optimum": float(np.mean(cost_gap)) if cost_gap else None,
        "frac_strictly_costlier": float(np.mean([c > 0 for c in cost_gap]))
                                   if cost_gap else None,
    }
    return out, stats
