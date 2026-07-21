"""Interchangeable sequence-mixing primitives ("the drop-in slot").

The primitive-harness (w20) evaluates the CLU as a *general* AI primitive rather
than a special case of one. That framing dictates the evaluation: every primitive
— MLP, GRU, S4D/Mamba-style SSM, causal self-attention, and the CLU — implements
ONE interface,

    block(x: (T, d_model)) -> (T, d_model),   causal in T,

and is dropped into the identical model (embedding + learned positional
embedding -> n_layers x [block + residual + LayerNorm] -> linear head). Nothing
else varies: same optimizer, schedule, data, LR grid, parameter budget.

**What the CLU had to concede to fit** (reported, not engineered around; see
`.claude/outputs/primitive-harness.md` Item 1):

1. *Driven, not autonomous.* Input enters as a momentum impulse ``p += W_in x_t``
   ("write current"). A driven Hamiltonian does not conserve H — the slot
   requires the unit to absorb an exogenous token stream, which the autonomous
   formulation has no channel for.
2. *Dissipative, not symplectic.* ``gamma > 0`` is required for the state to be
   readable (clu-retrieval-demo §6: retrieval accuracy 1.000 at gamma=0.02 vs
   0.813 at gamma=0), so the block is conformal-symplectic, not symplectic.
3. *Carry width 2*d_clu.* The state is (q, p), so parameter matching solves for
   ``d_clu`` separately from ``d_model``.

Everything else about the slot is shared. Blocks are ``eqx.Module`` PyTrees;
randomness is explicit PRNGKey threading; the recurrent blocks use ``lax.scan``.
"""

from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp

from chlu.core.chlu_unit import CHLU

#: Primitives registered for the harness (order is the canonical report order).
PRIMITIVES = ("mlp", "gru", "ssm", "attention", "clu")


class MLPBlock(eqx.Module):
    """Token-wise 2-layer MLP. **Control primitive: it cannot mix across time.**

    Included deliberately as the floor of the harness — any primitive that fails
    to beat it on a recall task has not learned to move information in time.
    """

    l1: eqx.nn.Linear
    l2: eqx.nn.Linear

    def __init__(self, d_model: int, width: int, *, key):
        k1, k2 = jax.random.split(key)
        self.l1 = eqx.nn.Linear(d_model, width, key=k1)
        self.l2 = eqx.nn.Linear(width, d_model, key=k2)

    def __call__(self, x, *, key=None):
        return jax.vmap(lambda t: self.l2(jax.nn.gelu(self.l1(t))))(x)


class GRUBlock(eqx.Module):
    """Single-direction GRU over the sequence + linear read-out."""

    cell: eqx.nn.GRUCell
    out: eqx.nn.Linear
    hidden: int = eqx.field(static=True)

    def __init__(self, d_model: int, width: int, *, key):
        k1, k2 = jax.random.split(key)
        self.hidden = width
        self.cell = eqx.nn.GRUCell(d_model, width, key=k1)
        self.out = eqx.nn.Linear(width, d_model, key=k2)

    def __call__(self, x, *, key=None):
        def step(h, x_t):
            h = self.cell(x_t, h)
            return h, h

        _, hs = jax.lax.scan(step, jnp.zeros(self.hidden), x)
        return jax.vmap(self.out)(hs)


class SSMBlock(eqx.Module):
    """Diagonal state-space block (S4D-style), optionally *selective* (Mamba-like).

    Real diagonal recurrence ``h_t = a * h_{t-1} + b * u_t`` with
    ``a = exp(-dt * softplus(log_decay))``, per-channel. With
    ``selective=True`` the timescale is input-dependent,
    ``dt_t = softplus(w_dt . x_t + b_dt)``, which is the one ingredient that
    distinguishes Mamba (S6) from S4D: a content-dependent gate on how much of
    the past is retained. Followed by a gated output projection (the standard
    Mamba-block output gate).
    """

    in_proj: eqx.nn.Linear
    out_proj: eqx.nn.Linear
    gate_proj: eqx.nn.Linear
    dt_proj: Optional[eqx.nn.Linear]
    log_decay: jnp.ndarray
    b: jnp.ndarray
    d_skip: jnp.ndarray
    selective: bool = eqx.field(static=True)

    def __init__(self, d_model: int, width: int, *, selective: bool = True, key):
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        self.in_proj = eqx.nn.Linear(d_model, width, key=k1)
        self.gate_proj = eqx.nn.Linear(d_model, width, key=k2)
        self.out_proj = eqx.nn.Linear(width, d_model, key=k3)
        self.selective = selective
        self.dt_proj = eqx.nn.Linear(d_model, 1, key=k4) if selective else None
        # Log-spaced decay rates: covers fast and slow timescales at init
        # (the HiPPO-lite trick that makes a diagonal SSM usable out of the box).
        self.log_decay = jnp.linspace(jnp.log(0.01), jnp.log(1.0), width)
        self.b = jax.random.normal(k5, (width,)) * 0.5
        self.d_skip = jnp.ones(width)

    def __call__(self, x, *, key=None):
        u = jax.vmap(self.in_proj)(x)  # (T, width)
        gate = jax.nn.silu(jax.vmap(self.gate_proj)(x))
        decay = jax.nn.softplus(self.log_decay)  # (width,) > 0

        if self.selective:
            dt = jax.nn.softplus(jax.vmap(self.dt_proj)(x))  # (T, 1)
        else:
            dt = jnp.ones((x.shape[0], 1))

        def step(h, inp):
            u_t, dt_t = inp
            a = jnp.exp(-dt_t * decay)
            h = a * h + (1.0 - a) * self.b * u_t
            return h, h

        _, hs = jax.lax.scan(step, jnp.zeros(u.shape[-1]), (u, dt))
        y = hs + self.d_skip * u  # local skip (the "D" term)
        return jax.vmap(self.out_proj)(y * gate)


class AttentionBlock(eqx.Module):
    """Causal multi-head self-attention (exact softmax attention)."""

    attn: eqx.nn.MultiheadAttention
    n_heads: int = eqx.field(static=True)

    def __init__(self, d_model: int, width: int, *, n_heads: int = 4, key):
        # width is the total qkv size; round to a multiple of n_heads.
        width = max(n_heads, (width // n_heads) * n_heads)
        self.n_heads = n_heads
        self.attn = eqx.nn.MultiheadAttention(
            num_heads=n_heads,
            query_size=d_model,
            qk_size=width // n_heads,
            vo_size=width // n_heads,
            output_size=d_model,
            key=key,
        )

    def __call__(self, x, *, key=None):
        T = x.shape[0]
        causal = jnp.tril(jnp.ones((T, T), dtype=bool))
        return self.attn(x, x, x, mask=causal)


class CLUBlock(eqx.Module):
    """The CLU in the drop-in slot: an input-driven CHLU recurrence.

    Per token, ``clu_steps`` Verlet steps of the learned Hamiltonian are taken,
    with the token injected as a momentum impulse before the first one::

        p <- p + W_in x_t                  (write current)
        (q, p) <- CHLU.step((q, p), dt, gamma)   x clu_steps
        y_t = W_out [q ; p]                (linear read of the full state)

    The read is linear on the *whole* state, matching every other block's
    linear read-out. The CHLU unit itself (``potential_net``, ``log_mass``) is
    used unmodified — this file adds no physics, only the input/output shell.
    """

    clu: CHLU
    w_in: eqx.nn.Linear
    w_out: eqx.nn.Linear
    dt: float = eqx.field(static=True)
    gamma: float = eqx.field(static=True)
    clu_steps: int = eqx.field(static=True)
    d_clu: int = eqx.field(static=True)

    def __init__(
        self,
        d_model: int,
        width: int,
        *,
        hidden: int = 32,
        dt: float = 0.1,
        gamma: float = 0.05,
        clu_steps: int = 1,
        kinetic_mode: str = "newtonian_learned",
        potential_type: str = "mlp",
        key,
    ):
        k1, k2, k3 = jax.random.split(key, 3)
        self.d_clu = width
        self.dt = dt
        self.gamma = gamma
        self.clu_steps = clu_steps
        self.clu = CHLU(
            dim=width,
            hidden=hidden,
            kinetic_mode=kinetic_mode,
            potential_type=potential_type,
            key=k1,
        )
        self.w_in = eqx.nn.Linear(d_model, width, key=k2)
        self.w_out = eqx.nn.Linear(2 * width, d_model, key=k3)

    def __call__(self, x, *, key=None):
        kicks = jax.vmap(self.w_in)(x)  # (T, d_clu)

        def step(state, kick):
            q, p = state
            p = p + kick  # driven Hamiltonian: the write current
            for _ in range(self.clu_steps):
                q, p = self.clu.step((q, p), self.dt, self.gamma)
            return (q, p), jnp.concatenate([q, p])

        z0 = jnp.zeros(self.d_clu)
        _, states = jax.lax.scan(step, (z0, z0), kicks)
        return jax.vmap(self.w_out)(states)


def make_block(name: str, d_model: int, width: int, *, key, **kwargs):
    """Construct a primitive by name. `width` is the primitive's capacity knob.

    ``width`` is what the harness solves for when matching parameter budgets; its
    meaning is primitive-specific by necessity (GRU hidden units, SSM state
    channels, attention qkv width, CLU latent dimension) — matching is on the
    resulting *parameter count*, which is the comparable quantity.
    """
    if name == "mlp":
        return MLPBlock(d_model, width, key=key)
    if name == "gru":
        return GRUBlock(d_model, width, key=key)
    if name == "ssm":
        return SSMBlock(d_model, width, selective=kwargs.get("ssm_selective", True), key=key)
    if name == "attention":
        return AttentionBlock(d_model, width, n_heads=kwargs.get("n_heads", 4), key=key)
    if name == "clu":
        return CLUBlock(
            d_model,
            width,
            hidden=kwargs.get("clu_hidden", 32),
            dt=kwargs.get("clu_dt", 0.1),
            gamma=kwargs.get("clu_gamma", 0.05),
            clu_steps=kwargs.get("clu_steps", 1),
            kinetic_mode=kwargs.get("clu_kinetic_mode", "newtonian_learned"),
            potential_type=kwargs.get("clu_potential_type", "mlp"),
            key=key,
        )
    raise ValueError(f"Unknown primitive: {name}. Must be one of {PRIMITIVES}.")


class SequenceModel(eqx.Module):
    """Embedding -> n_layers x [block + residual + LayerNorm] -> linear head.

    Held FIXED across primitives; only ``blocks`` changes. Supports both a
    discrete-token input (``vocab_size`` set, embedding lookup) and a continuous
    vector input (``in_dim`` set, linear encoder), so one model class serves the
    recall and the regression/state-tracking families.
    """

    embed: eqx.Module
    pos: jnp.ndarray
    blocks: list
    norms: list
    head: eqx.nn.Linear
    discrete: bool = eqx.field(static=True)

    def __init__(
        self,
        primitive: str,
        *,
        d_model: int,
        width: int,
        out_dim: int,
        max_len: int,
        vocab_size: Optional[int] = None,
        in_dim: Optional[int] = None,
        n_layers: int = 2,
        key,
        **block_kwargs,
    ):
        if (vocab_size is None) == (in_dim is None):
            raise ValueError("Specify exactly one of vocab_size (discrete) or in_dim (continuous).")
        keys = jax.random.split(key, n_layers + 3)
        self.discrete = vocab_size is not None
        if self.discrete:
            self.embed = eqx.nn.Embedding(vocab_size, d_model, key=keys[0])
        else:
            self.embed = eqx.nn.Linear(in_dim, d_model, key=keys[0])
        # Learned positional embedding, identical for every primitive. Attention
        # requires one; the recurrent primitives do not but receive it anyway so
        # the slot is genuinely identical (this is a concession to fairness that
        # slightly *helps* the recurrent baselines).
        self.pos = jax.random.normal(keys[1], (max_len, d_model)) * 0.02
        self.blocks = [
            make_block(primitive, d_model, width, key=keys[2 + i], **block_kwargs)
            for i in range(n_layers)
        ]
        self.norms = [eqx.nn.LayerNorm(d_model) for _ in range(n_layers)]
        self.head = eqx.nn.Linear(d_model, out_dim, key=keys[-1])

    def __call__(self, x, *, key=None):
        """x: (T,) int tokens or (T, in_dim) floats -> (T, out_dim)."""
        h = jax.vmap(self.embed)(x)
        h = h + self.pos[: h.shape[0]]
        for block, norm in zip(self.blocks, self.norms, strict=True):
            h = jax.vmap(norm)(h + block(h))
        return jax.vmap(self.head)(h)
