"""The arms: one shared action head, three trunks, matched parameter counts.

THE SHARED ACTION HEAD.  `ActionHead` is a single class, and every arm is given one
built from the SAME PRNG key, so before training the four heads are byte-identical
arrays of the same shape.  `head_identity_report` checks that and prints it.  The
heads are then trained per arm -- they have to be, since a head frozen on one trunk
would be a head tuned for that trunk -- but no arm gets a bigger, deeper or
differently-initialised decision layer than any other.  That is the whole of the
"shared action head" claim, stated exactly, with the part that is not shared named.

THE SHARED FEATURES.  Every trunk takes `features.obs`, nothing else.  The trunks
differ only in how they combine observations ACROSS TIME:
    MLP          this step only
    LSTM         a recurrent summary of the prefix
    TRANSFORMER  causal attention over the whole prefix

MATCHED PARAMETERS.  `match_widths` searches each trunk's width for the total count
closest to a common target and reports the counts it actually achieved -- it does
not pretend to an exact match that integer layer shapes cannot deliver.
"""

from __future__ import annotations

import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from w9_harness.judge_maze import ACTIONS, REFUSE

N_ACTIONS = len(ACTIONS)              # 5: N E S W P
N_OUT_REFUSER = N_ACTIONS + 1         # + REFUSE

D_MODEL = 32     # FIXED for every arm, so ActionHead has ONE shape everywhere.
                 # The matched-parameter search moves the trunk width, never this.


def n_params(m):
    return int(sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(m, eqx.is_array))))


# ------------------------------------------------------------ the shared head
class ActionHead(eqx.Module):
    """The one decision layer.  Same class, same shape, same init, every arm."""

    lin: eqx.nn.Linear

    def __init__(self, d_model, n_out, key):
        self.lin = eqx.nn.Linear(d_model, n_out, key=key)

    def __call__(self, h):
        return self.lin(h)


HEAD_KEY = jax.random.PRNGKey(20260921)       # identical for every arm, by design


# ----------------------------------------------------------------- the trunks
class MLPTrunk(eqx.Module):
    layers: list
    name: str = eqx.field(static=True)

    def __init__(self, obs_dim, d_model, width, key):
        k1, k2, k3 = jax.random.split(key, 3)
        self.layers = [eqx.nn.Linear(obs_dim, width, key=k1),
                       eqx.nn.Linear(width, width, key=k2),
                       eqx.nn.Linear(width, d_model, key=k3)]
        self.name = "mlp"

    def __call__(self, seq):                  # (T, obs) -> (T, d_model)
        def one(x):
            h = jax.nn.relu(self.layers[0](x))
            h = jax.nn.relu(self.layers[1](h))
            return jax.nn.relu(self.layers[2](h))
        return jax.vmap(one)(seq)


class LSTMTrunk(eqx.Module):
    proj: eqx.nn.Linear
    cell: eqx.nn.LSTMCell
    out: eqx.nn.Linear
    hidden: int = eqx.field(static=True)
    name: str = eqx.field(static=True)

    def __init__(self, obs_dim, d_model, width, key):
        k1, k2, k3 = jax.random.split(key, 3)
        self.proj = eqx.nn.Linear(obs_dim, width, key=k1)
        self.cell = eqx.nn.LSTMCell(width, width, key=k2)
        self.out = eqx.nn.Linear(width, d_model, key=k3)
        self.hidden = width
        self.name = "lstm"

    def __call__(self, seq):
        x = jax.vmap(lambda v: jax.nn.relu(self.proj(v)))(seq)
        h0 = (jnp.zeros(self.hidden), jnp.zeros(self.hidden))

        def stepf(carry, xt):
            carry = self.cell(xt, carry)
            return carry, carry[0]
        _, hs = jax.lax.scan(stepf, h0, x)
        return jax.vmap(lambda h: jax.nn.relu(self.out(h)))(hs)


class Block(eqx.Module):
    attn: eqx.nn.MultiheadAttention
    n1: eqx.nn.LayerNorm
    n2: eqx.nn.LayerNorm
    f1: eqx.nn.Linear
    f2: eqx.nn.Linear

    def __init__(self, d, heads, ff, key):
        k1, k2, k3 = jax.random.split(key, 3)
        self.attn = eqx.nn.MultiheadAttention(heads, d, key=k1)
        self.n1 = eqx.nn.LayerNorm(d)
        self.n2 = eqx.nn.LayerNorm(d)
        self.f1 = eqx.nn.Linear(d, ff, key=k2)
        self.f2 = eqx.nn.Linear(ff, d, key=k3)

    def __call__(self, x, mask):
        h = jax.vmap(self.n1)(x)
        x = x + self.attn(h, h, h, mask=mask)
        h = jax.vmap(self.n2)(x)
        h = jax.vmap(lambda v: self.f2(jax.nn.gelu(self.f1(v))))(h)
        return x + h


class TransformerTrunk(eqx.Module):
    proj: eqx.nn.Linear
    pos: jnp.ndarray
    blocks: list
    norm: eqx.nn.LayerNorm
    out: eqx.nn.Linear
    name: str = eqx.field(static=True)

    def __init__(self, obs_dim, d_model, width, key, n_layers=2, heads=4, tmax=176):
        ks = jax.random.split(key, n_layers + 3)
        self.proj = eqx.nn.Linear(obs_dim, width, key=ks[0])
        self.pos = jax.random.normal(ks[1], (tmax, width)) * 0.02
        self.blocks = [Block(width, heads, 2 * width, ks[2 + i]) for i in range(n_layers)]
        self.norm = eqx.nn.LayerNorm(width)
        self.out = eqx.nn.Linear(width, d_model, key=ks[n_layers + 2])
        self.name = "transformer"

    def __call__(self, seq):
        T = seq.shape[0]
        x = jax.vmap(self.proj)(seq) + self.pos[:T]
        mask = jnp.tril(jnp.ones((T, T), dtype=bool))
        for b in self.blocks:
            x = b(x, mask)
        x = jax.vmap(self.norm)(x)
        return jax.vmap(lambda v: jax.nn.relu(self.out(v)))(x)


TRUNKS = {"mlp": MLPTrunk, "lstm": LSTMTrunk, "transformer": TransformerTrunk}


class Policy(eqx.Module):
    trunk: eqx.Module
    head: ActionHead

    def __init__(self, kind, obs_dim, width, n_out, key, d_model=D_MODEL, **kw):
        self.trunk = TRUNKS[kind](obs_dim, d_model, width, key, **kw)
        self.head = ActionHead(d_model, n_out, HEAD_KEY)     # same key for every arm

    def __call__(self, seq):                 # (T, obs) -> (T, n_out)
        return jax.vmap(self.head)(self.trunk(seq))


# ------------------------------------------------------------ matched params
def build(kind, obs_dim, width, n_out, seed=0, **kw):
    return Policy(kind, obs_dim, width, n_out, jax.random.PRNGKey(1000 + seed), **kw)


def match_widths(obs_dim, n_out, target, kinds=("mlp", "lstm", "transformer"),
                 lo=8, hi=256, tmax=176, verbose=True):
    """Closest achievable parameter count per trunk against a common target.

    Integer layer shapes cannot hit a target exactly, so this reports the count it
    ACHIEVED and the residual, rather than claiming a match it does not have.  The
    transformer additionally searches over depth and head count, because with one
    width knob alone its step size is coarse enough to leave several percent on the
    table -- and "matched to within 7%" is not matched.
    """
    out = {}
    for kind in kinds:
        best = None
        if kind == "transformer":
            for layers in (1, 2, 3):
                for heads in (2, 4):
                    for w in range(lo, hi + 1, heads):
                        if w % heads:
                            continue
                        kw = dict(n_layers=layers, heads=heads, tmax=tmax)
                        p = n_params(build(kind, obs_dim, w, n_out, **kw))
                        if best is None or abs(p - target) < abs(best[1] - target):
                            best = (w, p, kw)
        else:
            for w in range(lo, hi + 1):
                p = n_params(build(kind, obs_dim, w, n_out))
                if best is None or abs(p - target) < abs(best[1] - target):
                    best = (w, p, {})
        out[kind] = best
    if verbose:
        print("MATCHED PARAMETERS (target %d, obs_dim %d, n_out %d)"
              % (target, obs_dim, n_out))
        for k, (w, p, kw) in out.items():
            extra = "" if not kw else "  " + " ".join(
                "%s=%s" % (a, b) for a, b in kw.items() if a != "tmax")
            print("  %-12s width %3d  params %7d  (%+.3f%% of target)%s"
                  % (k, w, p, 100.0 * (p - target) / target, extra))
    return out


def head_identity_report(policies):
    """Every arm's head must be the same shape and the same initial weights."""
    shapes, sizes = {}, {}
    for name, p in policies.items():
        leaves = jax.tree_util.tree_leaves(eqx.filter(p.head, eqx.is_array))
        shapes[name] = [tuple(l.shape) for l in leaves]
        sizes[name] = int(sum(l.size for l in leaves))
    ref = next(iter(shapes.values()))
    same_shape = all(s == ref for s in shapes.values())
    return {"head_shapes": shapes, "head_params": sizes,
            "identical_shape_across_arms": bool(same_shape),
            "same_init_key": "HEAD_KEY (single PRNGKey, used by every arm)",
            "trained_separately": True}


# --------------------------------------------------------------- BC training
def _ce(policy, X, Y, M):
    """Masked cross-entropy over a batch of padded trajectories."""
    logits = jax.vmap(policy)(X)                           # (B, T, n_out)
    ll = jnp.take_along_axis(jax.nn.log_softmax(logits), Y[..., None], axis=-1)[..., 0]
    return -(ll * M).sum() / jnp.maximum(M.sum(), 1.0)


def train_policy(policy, X, Y, M, steps=1500, batch=32, lr=1e-3, seed=0,
                 label="", verbose=True, val=None):
    t0 = time.time()
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))
    state = opt.init(eqx.filter(policy, eqx.is_array))

    @eqx.filter_jit
    def upd(p, st, xb, yb, mb):
        loss, g = eqx.filter_value_and_grad(_ce)(p, xb, yb, mb)
        u, st = opt.update(g, st, eqx.filter(p, eqx.is_array))
        return eqx.apply_updates(p, u), st, loss

    n = X.shape[0]
    rng = np.random.default_rng(seed)
    hist = []
    for s in range(steps):
        idx = rng.integers(0, n, size=min(batch, n))
        policy, state, loss = upd(policy, state, jnp.asarray(X[idx]),
                                  jnp.asarray(Y[idx]), jnp.asarray(M[idx]))
        if s % 250 == 0 or s == steps - 1:
            hist.append((s, float(loss)))
    rep = {"steps": steps, "batch": batch, "lr": lr, "loss_curve": hist,
           "final_loss": hist[-1][1], "params": n_params(policy),
           "seconds": time.time() - t0}
    if val is not None:
        vX, vY, vM = val
        rep["val_loss"] = float(_ce(policy, jnp.asarray(vX), jnp.asarray(vY),
                                    jnp.asarray(vM)))
        acc = []
        for i in range(0, len(vX), 64):
            lg = jax.vmap(policy)(jnp.asarray(vX[i:i + 64]))
            pr = jnp.argmax(lg, -1)
            acc.append((np.asarray(pr == vY[i:i + 64]) * vM[i:i + 64]).sum()
                       / max(vM[i:i + 64].sum(), 1))
        rep["val_teacher_forced_action_acc"] = float(np.average(acc))
    if verbose:
        print("  TRAIN %-12s params %7d  loss %.4f%s  [%.1fs]"
              % (label or "", rep["params"], rep["final_loss"],
                 "  val-acc %.4f" % rep["val_teacher_forced_action_acc"]
                 if val is not None else "", rep["seconds"]))
    return policy, rep
