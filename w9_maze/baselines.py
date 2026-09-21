"""The learned baselines: MLP, LSTM, transformer, at MATCHED PARAMETERS.

The table is a printed floor (wave-9 rule 3), so these are not decoration: if one of
them beats the CLU on the RESULT rows, the Shapes3D result does not transfer and that
is the wave's answer.

Matched to what
---------------
To the CLU's DEPLOYED READ: the atom floats plus the address-book floats, which is
everything that turns four words into a latent.  Each baseline's width is grown until
its parameter count is within :data:`MATCH_TOL` of that number, and the realised count
is printed on every row.  The CLU's float64 sufficient statistic -- what buys exact
deletion -- is declared separately and is NOT charged to the baselines, because they
are not asked to delete.

What they are given
-------------------
The same four words, as the same interpolated one-hots the CLU's design is built from
(so a budget between two trained words means the same thing to every arm), but WITHOUT
the pair-interaction block: a learned model is supposed to find its own interactions,
and handing it the CLU's factorial design would be handing it the CLU's inductive bias.
The LSTM and the transformer read the words as a four-token sequence, which is their
native shape.

They are trained on exactly the written items, to the same objective the CLU's ridge
minimises (squared error to the item's latent), with early stopping on a held-IN 10 %
of the training requests.  No held-out row is touched by any selection.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

MATCH_TOL = 0.02


def _count(model):
    return int(sum(v.size for v in jax.tree.leaves(eqx.filter(model, eqx.is_array))))


class MLP(eqx.Module):
    layers: list

    def __init__(self, key, n_in, h, d_out, depth=2):
        ks = jax.random.split(key, depth + 1)
        dims = [n_in] + [h] * depth + [d_out]
        self.layers = [eqx.nn.Linear(dims[i], dims[i + 1], key=ks[i])
                       for i in range(depth + 1)]

    def __call__(self, x):
        for lyr in self.layers[:-1]:
            x = jax.nn.gelu(lyr(x))
        return self.layers[-1](x)


class LSTMHead(eqx.Module):
    emb: eqx.nn.Linear
    cell: eqx.nn.LSTMCell
    out: eqx.nn.Linear
    h: int

    def __init__(self, key, n_tok_in, h, d_out):
        k1, k2, k3 = jax.random.split(key, 3)
        self.emb = eqx.nn.Linear(n_tok_in, h, key=k1)
        self.cell = eqx.nn.LSTMCell(h, h, key=k2)
        self.out = eqx.nn.Linear(h, d_out, key=k3)
        self.h = int(h)

    def __call__(self, toks):                      # toks: (4, n_tok_in)
        state = (jnp.zeros(self.h), jnp.zeros(self.h))
        for t in range(toks.shape[0]):
            state = self.cell(jax.nn.gelu(self.emb(toks[t])), state)
        return self.out(state[0])


class TransformerHead(eqx.Module):
    emb: eqx.nn.Linear
    pos: jax.Array
    attn: list
    ff: list
    ln: list
    out: eqx.nn.Linear

    def __init__(self, key, n_tok_in, h, d_out, layers=2, heads=4):
        ks = jax.random.split(key, 3 + 3 * layers)
        self.emb = eqx.nn.Linear(n_tok_in, h, key=ks[0])
        self.pos = jax.random.normal(ks[1], (4, h)) * 0.02
        self.attn = [eqx.nn.MultiheadAttention(heads, h, key=ks[3 + 3 * i])
                     for i in range(layers)]
        self.ff = [[eqx.nn.Linear(h, 2 * h, key=ks[4 + 3 * i]),
                    eqx.nn.Linear(2 * h, h, key=ks[5 + 3 * i])] for i in range(layers)]
        self.ln = [eqx.nn.LayerNorm(h) for _ in range(2 * layers)]
        self.out = eqx.nn.Linear(h, d_out, key=ks[2])

    def __call__(self, toks):
        x = jax.vmap(self.emb)(toks) + self.pos
        for i, at in enumerate(self.attn):
            y = jax.vmap(self.ln[2 * i])(x)
            x = x + at(y, y, y)
            y = jax.vmap(self.ln[2 * i + 1])(x)
            x = x + jax.vmap(lambda v: self.ff[i][1](jax.nn.gelu(self.ff[i][0](v))))(y)
        return self.out(x.mean(axis=0))


def _fit_width(build, target, lo=8, hi=1024):
    """Smallest width whose parameter count is nearest the target from either side."""
    best, best_gap = lo, None
    while lo <= hi:
        mid = (lo + hi) // 2
        n = _count(build(mid))
        gap = abs(n - target)
        if best_gap is None or gap < best_gap:
            best, best_gap = mid, gap
        if n < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return best, _count(build(best))


def token_inputs(Phi_main, layout, n_factors=4):
    """Split the main-effect design into one token per factor, zero-padded to a common
    width, so the sequence models see four words rather than one long vector."""
    cards = layout["cardinalities"]
    offs = layout["main_offsets"]
    w = max(cards)
    N = Phi_main.shape[0]
    out = np.zeros((N, n_factors, w + 1), np.float32)
    for f in range(n_factors):
        out[:, f, :cards[f]] = Phi_main[:, offs[f]:offs[f] + cards[f]]
        out[:, f, w] = 1.0                            # a per-token bias
    return out


class LearnedArm:
    """One learned baseline, trained to the item latents at matched parameters."""

    def __init__(self, kind, key, n_in, d_out, target_params, seq_in=None):
        self.kind = kind
        if kind == "mlp":
            build = lambda h: MLP(key, n_in, h, d_out)
        elif kind == "lstm":
            build = lambda h: LSTMHead(key, seq_in, h, d_out)
        elif kind == "transformer":
            build = lambda h: TransformerHead(key, seq_in, (h // 4) * 4 or 4, d_out)
        else:
            raise KeyError(kind)
        self.width, self.n_params = _fit_width(build, int(target_params))
        self.model = build(self.width)
        self.target_params = int(target_params)
        self.matched = abs(self.n_params - target_params) <= MATCH_TOL * target_params

    def fit(self, X, Y, steps=4000, batch=64, lr=1e-3, seed=0, verbose=False):
        rng = np.random.default_rng(seed)
        n = int(X.shape[0])
        perm = rng.permutation(n)
        n_val = max(8, int(0.10 * n))
        vi, ti = perm[:n_val], perm[n_val:]
        Xt, Yt, Xv, Yv = X[ti], Y[ti], X[vi], Y[vi]

        def loss_fn(model, xb, yb):
            pred = jax.vmap(model)(xb)
            return jnp.mean((pred - yb) ** 2)

        sched = optax.warmup_cosine_decay_schedule(1e-4, lr, int(0.05 * steps), steps,
                                                   1e-5)
        opt = optax.adam(sched)
        state = opt.init(eqx.filter(self.model, eqx.is_array))

        @eqx.filter_jit
        def step(model, state, xb, yb):
            lo, gr = eqx.filter_value_and_grad(loss_fn)(model, xb, yb)
            upd, state = opt.update(gr, state, eqx.filter(model, eqx.is_array))
            return eqx.apply_updates(model, upd), state, lo

        @eqx.filter_jit
        def vloss(model):
            return loss_fn(model, jnp.asarray(Xv), jnp.asarray(Yv))

        model = self.model
        best, best_v, self.hist = model, np.inf, []
        for t in range(steps):
            idx = rng.integers(0, Xt.shape[0], size=min(batch, Xt.shape[0]))
            model, state, lo = step(model, state, jnp.asarray(Xt[idx]),
                                    jnp.asarray(Yt[idx]))
            if (t + 1) % 200 == 0:
                v = float(vloss(model))
                self.hist.append((t + 1, float(lo), v))
                if v < best_v:
                    best_v, best = v, model
                if verbose:
                    print("    %s step %5d train %.5f heldin-val %.5f"
                          % (self.kind, t + 1, float(lo), v), flush=True)
        self.model, self.val_mse = best, float(best_v)
        return self

    def read(self, X) -> np.ndarray:
        out = []
        for s in range(0, int(X.shape[0]), 256):
            out.append(np.asarray(jax.vmap(self.model)(jnp.asarray(X[s:s + 256]))))
        return np.concatenate(out, 0).astype(np.float32)
