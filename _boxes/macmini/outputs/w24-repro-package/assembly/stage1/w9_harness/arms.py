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
import equinox as eqx
import jax
import jax.numpy as jnp
D_MODEL = 32     # FIXED for every arm, so ActionHead has ONE shape everywhere.
def n_params(m):
    return int(sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(m, eqx.is_array))))


class ActionHead(eqx.Module):
    """The one decision layer.  Same class, same shape, same init, every arm."""

    lin: eqx.nn.Linear

    def __init__(self, d_model, n_out, key):
        self.lin = eqx.nn.Linear(d_model, n_out, key=key)

    def __call__(self, h):
        return self.lin(h)


HEAD_KEY = jax.random.PRNGKey(20260921)       # identical for every arm, by design
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
