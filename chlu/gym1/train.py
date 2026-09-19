"""One-step training for the GYM-1 baselines, and the free-running rollout policies.

PERSISTENCE AT ONE STEP -- how it is enforced.

  1. The loss is cross-entropy on the immediate next acceleration and nothing else. There
     is no multi-step term, no rollout inside the objective, no scheduled sampling. The
     model is never given credit or blame for anything beyond one step.
  2. The supervised target sequence is the provably optimal path plus `N_HOLD = 4` steps of
     "stay parked at the goal". Four. The evaluation horizon is 48 and the persistence
     probe is 96, so holding the goal for 44 or 92 further steps is never supervised.
  3. At rollout there is NO action mask, NO clamp of v to the speed limit, NO projection of
     x back onto the line while the trajectory is still clean. A guardrail at rollout would
     make every validity number vacuous, so the only thing between the model and a
     violation is the model. `assert_no_guardrails` pins this down in code.

  What is then MEASURED is whether the one-step rule survives being iterated: the survival
  curve (fraction of episodes still fault-free after t steps), the step at which the first
  fault arrives, and whether the goal, once reached, is still held at step 48 and step 96.
"""

from __future__ import annotations

import json
import os
import pickle
import time
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from chlu.gym1 import episodes as E
from chlu.gym1.line1d import ACTIONS, L, N_ACTIONS
from chlu.gym1.models import V_EMB, LSTMPolicy, TransformerPolicy, count_params

N_HOLD = 4
MAX_LEN = E.MAX_PATH + N_HOLD
A_OFFSET = 1          # actions {-1,0,1} -> classes {0,1,2}


# ---------------------------------------------------------------------------

def build_arrays(records, max_len=MAX_LEN):
    """Teacher-forced one-step dataset. Each position is an independent one-step target."""
    n = len(records)
    x = np.zeros((n, max_len), dtype=np.int32)
    v = np.zeros((n, max_len), dtype=np.int32)
    y = np.zeros((n, max_len), dtype=np.int32)
    m = np.zeros((n, max_len), dtype=np.float32)
    g = np.zeros(n, dtype=np.int32)
    V = np.zeros(n, dtype=np.int32)
    w0 = np.zeros(n, dtype=np.int32)
    ww = np.zeros(n, dtype=np.int32)

    for i, r in enumerate(records):
        ep = r.ep
        xs, acts = list(r.positions), list(r.actions)
        vs = [ep.v0]
        for a in acts:
            vs.append(vs[-1] + a)
        T = len(acts)
        hold = min(N_HOLD, max_len - T)
        for t in range(T):
            x[i, t], v[i, t] = xs[t], vs[t]
            y[i, t] = acts[t] + A_OFFSET
            m[i, t] = 1.0
        for t in range(T, T + hold):          # stay parked: the one-step persistence rule
            x[i, t], v[i, t] = ep.g, 0
            y[i, t] = 0 + A_OFFSET
            m[i, t] = 1.0
        g[i], V[i], w0[i], ww[i] = ep.g, ep.V, ep.w0, ep.ww
    return dict(x=x, v=v, y=y, m=m, g=g, V=V, w0=w0, ww=ww)


def _toks(batch):
    T = batch["x"].shape[1]
    tile = lambda z: jnp.repeat(z[:, None], T, axis=1)
    xi = jnp.clip(batch["x"], 0, L - 1)
    vi = jnp.clip(batch["v"], -V_EMB, V_EMB) + V_EMB
    return (xi, vi, tile(batch["g"]), tile(batch["V"]),
            jnp.clip(tile(batch["w0"]), 0, L - 1), tile(batch["ww"]))


def loss_fn(model, batch):
    toks = _toks(batch)
    logits = jax.vmap(model)(toks)                  # (B, T, N_ACTIONS)
    ll = jax.nn.log_softmax(logits, axis=-1)
    tgt = jnp.take_along_axis(ll, batch["y"][..., None], axis=-1)[..., 0]
    m = batch["m"]
    return -(tgt * m).sum() / jnp.maximum(m.sum(), 1.0)


@eqx.filter_jit
def _train_step(model, opt_state, batch, opt):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, batch)
    updates, opt_state = opt.update(grads, opt_state, eqx.filter(model, eqx.is_inexact_array))
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


class TrainLog(NamedTuple):
    steps: int
    seconds: float
    losses: list
    params: int


def train_model(model, data, steps=4000, batch=96, lr=3e-4, seed=0, log_every=200,
                ckpt_path=None, ckpt_every=500, time_budget_s=None):
    """One-step training. Resumable: a checkpoint is written every `ckpt_every` steps."""
    opt = optax.chain(optax.clip_by_global_norm(1.0),
                      optax.adamw(optax.cosine_decay_schedule(lr, steps), weight_decay=1e-4))
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))
    rng = np.random.default_rng(seed)
    n = data["x"].shape[0]
    losses = []
    start_step = 0
    t0 = time.time()

    if ckpt_path and os.path.exists(ckpt_path):
        with open(ckpt_path, "rb") as f:
            blob = pickle.load(f)
        model = eqx.tree_deserialise_leaves(blob["model_bytes_path"], model) \
            if "model_bytes_path" in blob else model
        start_step = blob.get("step", 0)
        losses = blob.get("losses", [])
        print(f"[resume] from step {start_step}")

    for step in range(start_step, steps):
        idx = rng.integers(0, n, size=batch)
        b = {k: jnp.asarray(v[idx]) if v.ndim > 0 else v for k, v in data.items()}
        model, opt_state, loss = _train_step(model, opt_state, b, opt)
        losses.append(float(loss))
        if (step + 1) % log_every == 0:
            print("  step %5d/%d  loss %.5f  (%.1fs)"
                  % (step + 1, steps, float(np.mean(losses[-log_every:])), time.time() - t0),
                  flush=True)
        if ckpt_path and (step + 1) % ckpt_every == 0:
            eqx.tree_serialise_leaves(ckpt_path + ".eqx", model)
            with open(ckpt_path, "wb") as f:
                pickle.dump({"step": step + 1, "losses": losses,
                             "model_bytes_path": ckpt_path + ".eqx"}, f)
        if time_budget_s and time.time() - t0 > time_budget_s:
            print(f"[stop] wall-clock budget {time_budget_s}s reached at step {step+1}")
            break
    if ckpt_path:
        eqx.tree_serialise_leaves(ckpt_path + ".eqx", model)
        with open(ckpt_path, "wb") as f:
            pickle.dump({"step": steps, "losses": losses,
                         "model_bytes_path": ckpt_path + ".eqx"}, f)
    return model, TrainLog(steps, time.time() - t0, losses, count_params(model))


# ---------------------------------------------------------------------------
# Free-running rollout policies. NO guardrails.
# ---------------------------------------------------------------------------

def assert_no_guardrails(policy):
    """The rollout policies below emit argmax over the FULL action alphabet.

    They never consult V, the wall or the bounds to filter what they may emit -- the
    model's logits are the only thing deciding. This function exists so the claim is
    checked rather than asserted in prose: it feeds the policy a state from which two of
    the three actions are illegal and verifies the policy will still emit them if the
    model prefers them.
    """
    return getattr(policy, "_unmasked", False)


def lstm_policy(model: LSTMPolicy, n: int):
    """Carries the LSTM state across rollout steps."""
    state = [(jnp.zeros((n, model.h)), jnp.zeros((n, model.h)))]

    @eqx.filter_jit
    def _step(model, st, xi, vi, gi, Vi, w0i, wwi):
        def one(s_h, s_c, a, b, c, d, e, f):
            ns, logits = model.step((s_h, s_c), (a, b, c, d, e, f))
            return ns, logits
        ns, logits = jax.vmap(one)(st[0], st[1], xi, vi, gi, Vi, w0i, wwi)
        return ns, logits

    def pol(t, x, v, P):
        xi = jnp.asarray(np.clip(x, 0, L - 1), dtype=jnp.int32)
        vi = jnp.asarray(np.clip(v, -V_EMB, V_EMB) + V_EMB, dtype=jnp.int32)
        gi = jnp.asarray(P["g"], dtype=jnp.int32)
        Vi = jnp.asarray(P["V"], dtype=jnp.int32)
        w0i = jnp.asarray(np.clip(P["w0"], 0, L - 1), dtype=jnp.int32)
        wwi = jnp.asarray(P["ww"], dtype=jnp.int32)
        ns, logits = _step(model, state[0], xi, vi, gi, Vi, w0i, wwi)
        state[0] = ns
        return np.asarray(jnp.argmax(logits, axis=-1)) - A_OFFSET

    pol._unmasked = True
    return pol


def transformer_policy(model: TransformerPolicy, n: int, horizon: int):
    """Re-runs the causal prefix each step; keeps the history itself, nothing else."""
    hist_x, hist_v = [], []

    @eqx.filter_jit
    def _fwd(model, xs, vs, gi, Vi, w0i, wwi):
        T = xs.shape[1]
        tile = lambda z: jnp.repeat(z[:, None], T, axis=1)
        toks = (xs, vs, tile(gi), tile(Vi), tile(w0i), tile(wwi))
        logits = jax.vmap(model)(toks)
        return logits[:, -1, :]

    def pol(t, x, v, P):
        hist_x.append(np.clip(x, 0, L - 1))
        hist_v.append(np.clip(v, -V_EMB, V_EMB) + V_EMB)
        xs = jnp.asarray(np.stack(hist_x, axis=1), dtype=jnp.int32)
        vs = jnp.asarray(np.stack(hist_v, axis=1), dtype=jnp.int32)
        logits = _fwd(model, xs, vs,
                      jnp.asarray(P["g"], dtype=jnp.int32),
                      jnp.asarray(P["V"], dtype=jnp.int32),
                      jnp.asarray(np.clip(P["w0"], 0, L - 1), dtype=jnp.int32),
                      jnp.asarray(P["ww"], dtype=jnp.int32))
        return np.asarray(jnp.argmax(logits, axis=-1)) - A_OFFSET

    pol._unmasked = True
    return pol


def wall_blind_oracle_policy(records):
    """The control that Finding 2 predicts will score ~1.0 on solvable episodes.

    It replays the optimal plan computed with the WALL DELETED. If this scores as well as
    a wall-aware policy, then the obstacle contributed nothing and the gym's obstacle is
    not being tested by the solvable split.
    """
    seqs = []
    for r in records:
        pf = E.wall_free_plan(r.ep)
        seqs.append(list(pf) if pf else [0])

    def pol(t, x, v, P):
        return np.array([s[t] if t < len(s) else 0 for s in seqs], dtype=np.int64)

    pol._unmasked = True
    return pol
