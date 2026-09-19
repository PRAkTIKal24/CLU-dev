"""LSTM and transformer baselines for GYM-1, trained with the RULES VISIBLE.

Both models see, at every step, the speed limit `V`, the wall left edge `w0` and the wall
width `ww` as embedded tokens alongside the current position, the current velocity and
the goal. Nothing about the rules is hidden: the question this gym asks is not whether a
model can infer the rules, it is whether a model that has been TOLD the rules obeys them
when it is rolled out.

The supervision is one step wide. The loss is cross-entropy on the immediate next
acceleration and nothing else -- no multi-step term, no rollout in the loss, no scheduled
sampling. Whether the rules survive a free-running rollout is then a measurement, not
something the objective bought.

Parameters are matched between the two families (see `match_report`).
"""

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

from chlu.gym1.line1d import L, N_ACTIONS, V_MAX_MAX

V_EMB = 16          # velocity embedding is clipped to [-V_EMB, V_EMB]
N_V_TOK = 2 * V_EMB + 1
N_W0_TOK = L
N_WW_TOK = 5
N_VLIM_TOK = V_MAX_MAX + 1


def encode_step(x, v, g, V, w0, ww):
    """Token indices for one step. Arrays in, arrays out (works batched)."""
    xi = jnp.clip(x, 0, L - 1)
    vi = jnp.clip(v, -V_EMB, V_EMB) + V_EMB
    return xi, vi, jnp.clip(g, 0, L - 1), V, jnp.clip(w0, 0, L - 1), ww


class _Embed(eqx.Module):
    """Sum of the six token embeddings. The three rule tokens are always present."""

    e_x: eqx.nn.Embedding
    e_v: eqx.nn.Embedding
    e_g: eqx.nn.Embedding
    e_V: eqx.nn.Embedding
    e_w0: eqx.nn.Embedding
    e_ww: eqx.nn.Embedding

    def __init__(self, d, key):
        k = jax.random.split(key, 6)
        self.e_x = eqx.nn.Embedding(L, d, key=k[0])
        self.e_v = eqx.nn.Embedding(N_V_TOK, d, key=k[1])
        self.e_g = eqx.nn.Embedding(L, d, key=k[2])
        self.e_V = eqx.nn.Embedding(N_VLIM_TOK, d, key=k[3])
        self.e_w0 = eqx.nn.Embedding(N_W0_TOK, d, key=k[4])
        self.e_ww = eqx.nn.Embedding(N_WW_TOK, d, key=k[5])

    def __call__(self, xi, vi, gi, Vi, w0i, wwi):
        return (self.e_x(xi) + self.e_v(vi) + self.e_g(gi)
                + self.e_V(Vi) + self.e_w0(w0i) + self.e_ww(wwi))


class LSTMPolicy(eqx.Module):
    """One-step action predictor with an LSTM over the trajectory prefix."""

    embed: _Embed
    cell: eqx.nn.LSTMCell
    head: eqx.nn.Linear
    d: int = eqx.field(static=True)
    h: int = eqx.field(static=True)

    def __init__(self, d=64, h=205, *, key):
        ke, kc, kh = jax.random.split(key, 3)
        self.embed = _Embed(d, ke)
        self.cell = eqx.nn.LSTMCell(d, h, key=kc)
        self.head = eqx.nn.Linear(h, N_ACTIONS, key=kh)
        self.d, self.h = d, h

    def init_state(self):
        return (jnp.zeros(self.h), jnp.zeros(self.h))

    def step(self, state, toks):
        """One step: (state, tokens) -> (new_state, logits)."""
        e = self.embed(*toks)
        state = self.cell(e, state)
        return state, self.head(state[0])

    def __call__(self, toks_seq):
        """Teacher-forced over a whole prefix. toks_seq: tuple of (T,) int arrays."""
        def body(state, toks):
            state, logits = self.step(state, toks)
            return state, logits

        _, logits = jax.lax.scan(body, self.init_state(), toks_seq)
        return logits


class _Block(eqx.Module):
    ln1: eqx.nn.LayerNorm
    attn: eqx.nn.MultiheadAttention
    ln2: eqx.nn.LayerNorm
    ff1: eqx.nn.Linear
    ff2: eqx.nn.Linear

    def __init__(self, d, heads, d_ff, key):
        ka, k1, k2 = jax.random.split(key, 3)
        self.ln1 = eqx.nn.LayerNorm(d)
        self.attn = eqx.nn.MultiheadAttention(heads, d, key=ka)
        self.ln2 = eqx.nn.LayerNorm(d)
        self.ff1 = eqx.nn.Linear(d, d_ff, key=k1)
        self.ff2 = eqx.nn.Linear(d_ff, d, key=k2)

    def __call__(self, h, mask):
        z = jax.vmap(self.ln1)(h)
        h = h + self.attn(z, z, z, mask=mask)
        z = jax.vmap(self.ln2)(h)
        z = jax.vmap(self.ff2)(jax.nn.gelu(jax.vmap(self.ff1)(z)))
        return h + z


class TransformerPolicy(eqx.Module):
    """One-step action predictor with a causal transformer over the trajectory prefix."""

    embed: _Embed
    pos: eqx.nn.Embedding
    blocks: list
    ln_f: eqx.nn.LayerNorm
    head: eqx.nn.Linear
    d: int = eqx.field(static=True)
    max_len: int = eqx.field(static=True)

    def __init__(self, d=64, n_layers=3, heads=4, d_ff=450, max_len=128, *, key):
        ke, kp, kh, *kb = jax.random.split(key, 3 + n_layers)
        self.embed = _Embed(d, ke)
        self.pos = eqx.nn.Embedding(max_len, d, key=kp)
        self.blocks = [_Block(d, heads, d_ff, kb[i]) for i in range(n_layers)]
        self.ln_f = eqx.nn.LayerNorm(d)
        self.head = eqx.nn.Linear(d, N_ACTIONS, key=kh)
        self.d, self.max_len = d, max_len

    def __call__(self, toks_seq):
        T = toks_seq[0].shape[0]
        h = jax.vmap(self.embed)(*toks_seq) + jax.vmap(self.pos)(jnp.arange(T))
        mask = jnp.tril(jnp.ones((T, T), dtype=bool))
        for b in self.blocks:
            h = b(h, mask)
        return jax.vmap(self.head)(jax.vmap(self.ln_f)(h))


# ---------------------------------------------------------------------------
# Parameter accounting. Printed, never estimated.
# ---------------------------------------------------------------------------

def count_params(model) -> int:
    return int(sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_inexact_array))))


def param_breakdown(model) -> dict:
    out = {}
    for name in ("embed", "pos", "cell", "blocks", "ln_f", "head"):
        sub = getattr(model, name, None)
        if sub is None:
            continue
        leaves = jax.tree_util.tree_leaves(eqx.filter(sub, eqx.is_inexact_array))
        out[name] = int(sum(x.size for x in leaves))
    return out


class MatchReport(NamedTuple):
    lstm_params: int
    tfm_params: int
    rel_gap: float
    budget_bytes: int


def match_report(lstm, tfm, bytes_per_param: int = 4) -> MatchReport:
    a, b = count_params(lstm), count_params(tfm)
    gap = abs(a - b) / max(a, b)
    # The lookup tripwire is given the SAME bytes as the parameter-matched baselines.
    budget = bytes_per_param * max(a, b)
    return MatchReport(a, b, gap, budget)
