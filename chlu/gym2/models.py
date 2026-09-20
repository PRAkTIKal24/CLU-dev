"""LSTM and transformer baselines for GYM-2, trained with the RULES VISIBLE.

"Rules visible" is taken literally and generously. At every step both models see

    * the speed limit V, as an embedded token;
    * the WHOLE obstacle map, as a learned linear projection of the 20x20 occupancy grid;
    * a 7x7 egocentric patch of that map (out-of-grid counts as blocked), so local
      collision geometry does not have to be decoded out of the global projection;
    * position, velocity and goal.

Nothing about the rules is hidden. The question this gym asks is not whether a model can
infer the rules, it is whether a model that has been TOLD the rules obeys them when it is
rolled out free-running.

The supervision is one step wide: cross-entropy on the immediate next acceleration and
nothing else -- no multi-step term, no rollout in the loss, no scheduled sampling. Whether
the rules survive iteration is then a measurement, not something the objective bought.

Parameters are matched between the two families; `match_report` prints the counts.
"""

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

from chlu.gym2.grid2d import G, N_ACTIONS, V_MAX_MAX

V_EMB = 8                 # velocity embedding clipped to [-V_EMB, V_EMB]
N_V_TOK = 2 * V_EMB + 1
N_VLIM_TOK = V_MAX_MAX + 1
PATCH = 7                 # egocentric obstacle patch, odd
PAD = PATCH // 2
N_CELLS = G * G


class _Embed(eqx.Module):
    """Sum of the token embeddings and the two obstacle-map features."""

    e_x: eqx.nn.Embedding
    e_y: eqx.nn.Embedding
    e_vx: eqx.nn.Embedding
    e_vy: eqx.nn.Embedding
    e_gx: eqx.nn.Embedding
    e_gy: eqx.nn.Embedding
    e_V: eqx.nn.Embedding
    w_map: eqx.nn.Linear       # the WHOLE obstacle grid
    w_patch: eqx.nn.Linear     # egocentric window on it

    def __init__(self, d, key):
        k = jax.random.split(key, 9)
        self.e_x = eqx.nn.Embedding(G, d, key=k[0])
        self.e_y = eqx.nn.Embedding(G, d, key=k[1])
        self.e_vx = eqx.nn.Embedding(N_V_TOK, d, key=k[2])
        self.e_vy = eqx.nn.Embedding(N_V_TOK, d, key=k[3])
        self.e_gx = eqx.nn.Embedding(G, d, key=k[4])
        self.e_gy = eqx.nn.Embedding(G, d, key=k[5])
        self.e_V = eqx.nn.Embedding(N_VLIM_TOK, d, key=k[6])
        self.w_map = eqx.nn.Linear(N_CELLS, d, key=k[7])
        self.w_patch = eqx.nn.Linear(PATCH * PATCH, d, key=k[8])

    def __call__(self, xi, yi, vxi, vyi, gxi, gyi, Vi, mapv, patch):
        return (self.e_x(xi) + self.e_y(yi) + self.e_vx(vxi) + self.e_vy(vyi)
                + self.e_gx(gxi) + self.e_gy(gyi) + self.e_V(Vi)
                + self.w_map(mapv) + self.w_patch(patch))


class LSTMPolicy(eqx.Module):
    embed: _Embed
    cell: eqx.nn.LSTMCell
    head: eqx.nn.Linear
    d: int = eqx.field(static=True)
    h: int = eqx.field(static=True)

    def __init__(self, d=64, h=160, *, key):
        ke, kc, kh = jax.random.split(key, 3)
        self.embed = _Embed(d, ke)
        self.cell = eqx.nn.LSTMCell(d, h, key=kc)
        self.head = eqx.nn.Linear(h, N_ACTIONS, key=kh)
        self.d, self.h = d, h

    def init_state(self):
        return (jnp.zeros(self.h), jnp.zeros(self.h))

    def step(self, state, toks):
        e = self.embed(*toks)
        state = self.cell(e, state)
        return state, self.head(state[0])

    def __call__(self, toks_seq):
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
    embed: _Embed
    pos: eqx.nn.Embedding
    blocks: list
    ln_f: eqx.nn.LayerNorm
    head: eqx.nn.Linear
    d: int = eqx.field(static=True)
    max_len: int = eqx.field(static=True)

    def __init__(self, d=64, n_layers=3, heads=4, d_ff=256, max_len=128, *, key):
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
    return int(sum(x.size for x in jax.tree_util.tree_leaves(
        eqx.filter(model, eqx.is_inexact_array))))


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
    return MatchReport(a, b, gap, bytes_per_param * max(a, b))
