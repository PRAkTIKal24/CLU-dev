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
2. *Dissipative, not symplectic.* ``gamma > 0`` was assumed to be required for
   the state to be readable (w19 clu-retrieval-demo §6: retrieval accuracy 1.000
   at gamma=0.02 vs 0.813 at gamma=0), so the block is conformal-symplectic, not
   symplectic. ⚠ **That justification was superseded**: w20
   ``learned-landscape-write-read`` §5 showed the 0.813-at-gamma=0 figure is a
   SINGLE-PHASE artifact (with a relaxation phase, fidelity is exactly invariant
   to gamma_read), and ``address-space-dimension-scaling`` §4 showed identity
   retrieval at gamma=0 is fine (0.969-1.000) while only *value* retrieval needs
   dissipation. ``gamma`` is therefore a swept knob, not a fixed concession —
   see ``chlu/experiments/exp_primitive_harness.py:run_gamma_read_sweep``.
3. *Carry width 2*d_clu.* The state is (q, p), so parameter matching solves for
   ``d_clu`` separately from ``d_model``.

Everything else about the slot is shared. Blocks are ``eqx.Module`` PyTrees;
randomness is explicit PRNGKey threading; the recurrent blocks use ``lax.scan``.
"""

from dataclasses import dataclass, fields
from typing import Any, Dict, NamedTuple, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from chlu.core.chlu_unit import CHLU

#: Primitives registered for the harness (order is the canonical report order).
PRIMITIVES = ("mlp", "gru", "ssm", "attention", "clu")

#: ⭐ C2W4 tier-iii memory cells (``StreamBlock``'s swappable slot). ⛔ These are
#: NOT ``CLUBlock``: ``"clu_store"`` holds the **full C2W1 store**, the others are
#: the system-level swap controls. See :class:`StreamBlock`.
MEMORY_CELLS = ("clu_store", "gru_matched", "ttt_matched", "none", "echo")

#: CLU-internal read modes (w21). See ``CLUBlock`` for the semantics.
CLU_READ_MODES = ("endpoint", "trajectory")

#: CLU-internal write-current modes (w21, EXPLORATORY). See ``CLUBlock``.
CLU_WRITE_MODES = ("linear", "gated")


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

    **Read modes** (w21 ``gamma-read-sweep`` Item 2). The two read types want
    opposite dissipation, so the read is a knob, not a constant:

    - ``"endpoint"`` (default, the shipped behaviour): read the *settled* state
      after all ``clu_steps`` sub-steps, ``y_t = W_out [q_K ; p_K]``. Wants
      ``gamma > 0`` — the value is only readable once it is a fixed point.
    - ``"trajectory"``: read the whole intra-token rollout — the Prop-11 fiber,
      ``y_t = W_out [q_1 ; p_1 ; ... ; q_K ; p_K]``. Wants ``gamma -> 0``,
      because the oscillation *is* the signal and dissipation erases it.

    ⚠ **At ``clu_steps == 1`` the two modes are the SAME MAP, exactly** (the
    fiber has one element), including bit-identical initialisation, since
    ``w_out`` has the same shape and consumes the same key. The read-mode axis
    is only non-degenerate for ``clu_steps > 1``; this is asserted by
    ``test_trajectory_read_is_identity_at_one_step``.

    **Write modes** (w21, EXPLORATORY — ``"linear"`` is the shipped default and
    the only mode any published number uses):

    - ``"linear"``: ``p += W_in x_t``. The write current is *unconditionally*
      linear in the token, so the state accumulates ``sum_t a v_t`` and
      ``sum_t b m_t`` but never their product. Measured consequence
      (`gamma-read-sweep` memory probe): at the readout the state decodes
      ``sum_t v_t`` at R^2 = 1.000 and the marker positions at R^2 = 0.99 at
      EVERY gamma, but the adding target -- a value x marker *conjunction* --
      at R^2 ~ 0.07.
    - ``"gated"``: ``p += (W_in x_t) * sigmoid(W_gate x_t)``. Supplies the one
      ingredient the GRU (gates), the selective SSM (input-dependent Delta) and
      attention (softmax QK) all have and the CLU did not: a multiplicative,
      input-conditioned write. Costs ``d_clu`` under the matched-parameter
      search, exactly like the fiber read.
    """

    clu: CHLU
    w_in: eqx.nn.Linear
    w_gate: Optional[eqx.nn.Linear]
    w_out: eqx.nn.Linear
    dt: float = eqx.field(static=True)
    gamma: float = eqx.field(static=True)
    clu_steps: int = eqx.field(static=True)
    d_clu: int = eqx.field(static=True)
    read_mode: str = eqx.field(static=True)
    write_mode: str = eqx.field(static=True)

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
        read_mode: str = "endpoint",
        write_mode: str = "linear",
        key,
    ):
        if read_mode not in CLU_READ_MODES:
            raise ValueError(
                f"Unknown CLU read_mode {read_mode!r}. Must be one of {CLU_READ_MODES}."
            )
        if write_mode not in CLU_WRITE_MODES:
            raise ValueError(
                f"Unknown CLU write_mode {write_mode!r}. Must be one of {CLU_WRITE_MODES}."
            )
        # NOTE: still a 3-way split, and the gate key is folded in separately, so
        # the default ("linear") block is initialised BIT-IDENTICALLY to the w20
        # shipped harness. A 4-way split here would silently re-randomise every
        # published CLU cell.
        k1, k2, k3 = jax.random.split(key, 3)
        self.d_clu = width
        self.dt = dt
        self.gamma = gamma
        self.clu_steps = clu_steps
        self.read_mode = read_mode
        self.write_mode = write_mode
        self.clu = CHLU(
            dim=width,
            hidden=hidden,
            kinetic_mode=kinetic_mode,
            potential_type=potential_type,
            key=k1,
        )
        self.w_in = eqx.nn.Linear(d_model, width, key=k2)
        self.w_gate = (
            eqx.nn.Linear(d_model, width, key=jax.random.fold_in(k2, 1))
            if write_mode == "gated"
            else None
        )
        # The trajectory read consumes the whole fiber, so its read-in width is
        # clu_steps x larger. Parameter matching then solves for a SMALLER
        # d_clu at the same budget -- the fiber is paid for out of state width,
        # which is exactly the trade-off the sweep is meant to expose.
        n_read = clu_steps if read_mode == "trajectory" else 1
        self.w_out = eqx.nn.Linear(2 * width * n_read, d_model, key=k3)

    def __call__(self, x, *, key=None):
        kicks = jax.vmap(self.w_in)(x)  # (T, d_clu)
        if self.w_gate is not None:  # EXPLORATORY: input-conditioned write current
            kicks = kicks * jax.nn.sigmoid(jax.vmap(self.w_gate)(x))
        trajectory = self.read_mode == "trajectory"

        def step(state, kick):
            q, p = state
            p = p + kick  # driven Hamiltonian: the write current
            fiber = []
            for _ in range(self.clu_steps):
                q, p = self.clu.step((q, p), self.dt, self.gamma)
                if trajectory:
                    fiber += [q, p]
            return (q, p), jnp.concatenate(fiber if trajectory else [q, p])

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
            read_mode=kwargs.get("clu_read_mode", "endpoint"),
            write_mode=kwargs.get("clu_write_mode", "linear"),
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


# ==========================================================================
# ⭐ TIER III (C2W4) — the FULL C2W1 CLU store as a streaming block's memory
# ==========================================================================
"""
⛔ **Read this before using anything below.** ``CLUBlock`` (above) is *not* the
memory. It is a driven-Hamiltonian recurrence with **no store, no admission, no
placement** — the w20/w21 object — and the Head's C2W4 ruling rules it out as a
tier-iii arm. The memory slot below holds the **C2W1 full store**
(:class:`chlu.core.clu_system.LearnedVStore` + the controller + the 13 monitors),
used through its public API and never edited.

**Why the design is shaped the way it is** (each rule is a measured lesson):

1. ``d q*/d q0 = 0`` **exactly**, so a settled-point read sends *no* gradient to
   its read-in (measured ``||dL/dphi||``: 0.0 implicit · 2.654e-9 unroll ·
   6.421e-3 trajectory). ⇒ the block trains end-to-end **only** through the
   trajectory read. ``read_mode="settled_point"`` is retained solely as the
   pre-registered *control* that must measure ~0.
2. ``orgdiv-prereg`` **Theorem O1**: under a settled-point read the image of
   ``x -> q*`` is exactly the set of minima of ``V_theta`` (measured: 2 distinct
   settled points from 4000 queries), so a settled-point-read block is
   reproducible by an ``N_min``-row table **for every reader**. The trajectory
   read is mandatory twice over.
3. ``gamma`` / ``M`` are trainable selectors **only** through that same channel;
   friction is the ~14x stronger channel, so ``gamma`` is the selector this
   block trains by default.
4. **Chunk granularity is mandatory, not a concession** (charter §2.2: memory
   operations at chunk granularity, "as Titans-class memories do — fair"). A
   settle per token does not run at sequence scale. ⭐ **Every swap-control cell
   gets the identical chunk convention** or the comparison is not matched.
5. **Read before write, always** (:class:`StreamBlock`): the query for chunk
   ``c`` is ``phi(pool(chunk c-1))`` run against a store holding chunks
   ``0..c-2``. Reading a store that was just handed the answer is an echo, not a
   retrieval, and it is the cheapest way to launder a win. It is also what makes
   the block causal: no token ever sees its own chunk's pooled summary.
"""


@dataclass(frozen=True)
class StreamMemoryConfig:
    """Static knobs of the streaming memory slot (⚠ **not** ``chlu/config.py``).

    ``chlu/config.py`` is standing read-only to C2 engineers, so — exactly as
    :class:`~chlu.core.clu_system.CluSystemConfig` and
    :class:`~chlu.core.psi_readout.PsiSpec` do — the config lives next to the
    code that reads it, with a ``from_mapping`` override path for a
    ``cluformer:`` block in a project YAML.
    """

    #: Tokens per memory operation (§0.3 mitigation 1). One read + one write per
    #: chunk, in EVERY arm.
    chunk: int = 64
    #: Verlet steps of phase 1 (address acquisition, ``gamma_address``).
    address_steps: int = 64
    #: Verlet steps of phase 2 (the read rollout, ``gamma_read``).
    read_steps: int = 64
    #: Strided trajectory buffer handed to psi.
    traj_stride: int = 8
    #: ⭐ ``"trajectory"`` is the only mode that trains ``phi`` (design rule 1).
    #: ``"settled_point"`` exists as the pre-registered zero-gradient control.
    read_mode: str = "trajectory"
    #: Inner masked-write steps per chunk. ⚠ The shipped store uses 300; at chunk
    #: granularity that is unaffordable, so this is the §0.3 re-budget — the same
    #: objective, fewer steps, differentiably unrolled (the Titans/TTT
    #: convention). **Monitor #13 (maturity, floor 40) trips as a result and that
    #: is a reported artifact, not a bug.**
    write_inner_steps: int = 4
    #: Inner-write step size. ⭐ The inner optimiser is **sign-SGD**, not plain
    #: SGD, and that is load-bearing, not a detail. The shipped store digs its
    #: wells with **300 Adam steps at 3e-3** — and Adam moves a parameter by
    #: ~``lr`` per step *regardless of gradient magnitude*. At
    #: ``atom_depth_init = 1e-4`` the landscape starts essentially flat, so the
    #: write objective's gradients are ~1e-4 and plain SGD moves the atoms by
    #: ~1e-6 per step: MEASURED, the read output then moved by **7.5e-9**, i.e.
    #: below float32 resolution — the memory was arithmetically inert and would
    #: have been reported as a null. Sign-SGD reproduces Adam's step-size
    #: behaviour **statelessly**; a real Adam would add two moment tensors per
    #: sequence (2x the store's state bytes) and break the D2 byte ledger.
    write_lr: float = 0.05
    #: Monte-Carlo perturbations in the masked write objective.
    write_n_perturb: int = 8
    #: ⭐ ``True`` = sign-SGD inner write (the default and the shipped
    #: configuration; see ``write_lr``). ``False`` = plain SGD, which is kept
    #: **only** as a diagnostic: ``jnp.sign`` has zero derivative, so sign-SGD
    #: severs ``d(store state)/d(phi)`` and the trajectory read becomes the
    #: *only* channel to ``phi`` by construction as well as by theorem. Running
    #: the gradient probe with ``write_sign=False`` separates the two causes.
    write_sign: bool = True
    #: psi hidden width. ⭐ Chosen so a **two-sided-matched** TTT-class swap
    #: exists (see :func:`solve_matched_ttt`); declared in PREREG §0.
    psi_hidden: int = 128
    #: ⭐ Output gain of the shared ``phi``. **The anti-collapse initialisation**
    #: (§A13 design rule: allocation collapse is a gradient ATTRACTOR — once the
    #: policy reaches a degenerate corner its own gradient dies and it cannot
    #: leave, so it must be *initialised away from it*). At gain 1.0 a tanh MLP
    #: on LayerNormed pooled summaries emits tightly-clustered addresses and the
    #: admission gate refuses most chunks on the first step; the gain spreads the
    #: chunk latents across the address ball so admission starts LIVE. Identical
    #: in every arm (it is part of the shared shell) and reported as a flag.
    phi_gain: float = 1.0
    #: ⭐ Confidence-gated retry rounds (0 = off). Live per ruling 0.1; the round
    #: is always computed and gated by the plan's verdict (a data-dependent step
    #: count is not a static shape), so it is charged to the Verlet budget.
    retry_rounds: int = 1
    #: Trainable friction selector (the 14x channel). ``False`` pins gamma.
    trainable_gamma: bool = True
    #: Trainable mass selector (the weaker channel; kept live per ruling 0.1).
    trainable_mass: bool = True
    #: Intra-chunk causal depthwise conv width. Identical in every arm, so the
    #: memory carries only CROSS-chunk information and the swap is meaningful.
    conv_kernel: int = 4
    #: Token-wise MLP expansion (the "assimilation/decoder" of §A13). Identical
    #: in every arm.
    mlp_mult: int = 4

    # -- `pilot-placement-probe` H1: the N98 LOCALIZED ATOM INIT --------------
    #: ⭐ **H1.** Radius of the ball each atom group is initialised in, around
    #: its own ``atom_group_centers`` row, **on the leading ``L`` (address)
    #: coordinates only** (N46: localizing the payload axis would hand the write
    #: the value it is supposed to learn). ``0.0`` (default) = the historical
    #: scatter and the construction is **bit-identical** — the localized draw
    #: uses a folded key, so even the default RNG stream is untouched.
    #: N98's designed band is ``~2 * atom_width``.
    #: ⚠ ``CluSystemConfig.atom_local_radius`` exists but is **not read** by
    #: :class:`~chlu.core.clu_system.LearnedVStore` (it never forwards it to
    #: ``DesignFreedomPotential``); this knob is the streaming block's own path
    #: to the same shipped mechanism and reproduces it bit-for-bit.
    atom_local_radius: float = 0.0
    #: ``(capacity, L)`` localization targets as a **tuple of tuples of floats**
    #: (the config is a static field, so it must stay hashable). ``None`` =>
    #: no localization regardless of ``atom_local_radius``.
    atom_group_centers: Optional[tuple] = None
    #: ⭐ **H1b — localized placement AT WRITE** (the streaming form of H1).
    #: ``atom_local_radius`` is N98 as it ships: a *static* init localization
    #: around targets fixed before the stream starts. In a streaming block the
    #: site an item will occupy is **not known at init** — the controller decides
    #: it when the chunk arrives — so the static lever localizes a group's atoms
    #: around a point the item it later holds has no reason to be near. This
    #: knob re-draws the written slot's atom ADDRESS coordinates into a ball of
    #: this radius around the **incoming chunk's own address** at write time,
    #: which is what "atoms seeded near the phi-image of the chunk" means once
    #: the stream is live. Payload coordinates are untouched (N46). The offsets
    #: are a fixed per-atom pattern (a parameter-free geometric jig), so no byte
    #: of the STATE column changes and C3 locality is preserved: only the
    #: slot's own rows move, and a refused offer still leaves ``V_theta``
    #: bit-identical. ``0.0`` (default) = off, bit-identical.
    atom_place_radius: float = 0.0

    # -- `pilot-placement-probe` H2: the C2W2 TRAJECTORY WRITE TERM ----------
    #: ⭐ **H2.** Coefficient of :func:`~chlu.training.train_memory.
    #: trajectory_margin_penalty` inside the inner write objective. ``0.0``
    #: (default) leaves the shipped objective **bit-identical** (the term is
    #: added only when the coefficient is a Python float > 0, so not one extra
    #: op is traced). C2W2's measured band is ``{0.03, 0.3, 3, 30}``.
    write_lambda_traj: float = 0.0
    #: Verlet steps of the trajectory term's own rollout (defaults to the read's
    #: phase-2 budget when 0, so the path the write shapes is the path the read
    #: traverses).
    write_traj_steps: int = 0
    #: Launches per item in the trajectory term (kept small: the term costs
    #: ``n_launch * steps`` differentiated force evaluations per inner step).
    write_traj_n_launch: int = 2

    @classmethod
    def from_mapping(cls, overrides: Optional[dict] = None) -> "StreamMemoryConfig":
        known = {f.name for f in fields(cls)}
        kw = {k: v for k, v in dict(overrides or {}).items() if k in known}
        return cls(**kw)

    def as_flag_table(self) -> Dict[str, Any]:
        base = StreamMemoryConfig()
        return {f.name: getattr(self, f.name) for f in fields(self)
                if getattr(self, f.name) != getattr(base, f.name)}


class WritePlan(NamedTuple):
    """⭐ The controller's decisions for one sequence, as traceable arrays.

    **Why this exists — the decision-replay design.** The C2W1 controller
    (:class:`~chlu.core.clu_controller.CluControllerV0`) is a mutable Python
    object whose verbs branch on ``numpy`` values: admission, placement,
    eviction, decay and retry are *discrete* decisions. It cannot run inside a
    traced/differentiated forward pass — ``np.asarray(tracer)`` raises.

    Rather than delete the controller (forbidden: ruling 0.1) or fake it, the
    forward is split in two:

    * **pass 1 (concrete, no gradient)** — the real controller runs on detached
      latents and emits this plan;
    * **pass 2 (differentiable)** — the block replays the plan with the slot
      assignments held fixed.

    Nothing about the controller is weakened: every guard fires, every verb is
    exercised, the monitors see the real trip states. What is given up is
    ``d(decision)/d(theta)`` — which is zero anyway (the verbs are discrete), and
    which is exactly the quantity T3's corollary says is a gradient attractor.

    Fields are all ``(n_chunks, ...)`` and are ``stop_gradient`` by construction
    (they come from a detached pass).

    Attributes:
        slot: ``(n_chunks,)`` int32 — the atom group this chunk writes into.
        admitted: ``(n_chunks,)`` float32 in {0, 1} — the admission gate's verdict.
        group_scale: ``(n_chunks, K)`` float32 — per-item amplitude multiplier
            applied BEFORE the write. Carries per-item **lifetimes** (leak) and
            the ``decay`` verb.
        reset: ``(n_chunks, K)`` float32 in {0, 1} — 1 = re-draw this group from
            the initialisation distribution (eviction/deletion). ⚠ Re-draw, never
            zero: zeroing starves the next item in the slot AND leaves a
            membership trace (``LearnedVStore.reinit_group``).
        sites: ``(n_chunks, K, dim)`` float32 — the live codebook, used as the
            write objective's crowding/barrier targets. Dead slots are pushed to
            ``+1e3`` so they are never anyone's nearest neighbour.
        live: ``(n_chunks, K)`` float32 in {0, 1}.
        retry: ``(n_chunks,)`` int32 — extra phase-2 rounds the confidence gate
            asked for (0 = none).
    """

    slot: jnp.ndarray
    admitted: jnp.ndarray
    group_scale: jnp.ndarray
    reset: jnp.ndarray
    sites: jnp.ndarray
    live: jnp.ndarray
    retry: jnp.ndarray


def blank_plan(n_chunks: int, capacity: int, dim: int) -> WritePlan:
    """A plan that admits nothing — the **blank-store control** (collapse #4).

    The store is read exactly as in the live arm but never written, so any bpc
    the model still achieves through the memory path is a leak, not a retrieval.
    """
    return WritePlan(
        slot=jnp.zeros((n_chunks,), dtype=jnp.int32),
        admitted=jnp.zeros((n_chunks,), dtype=jnp.float32),
        group_scale=jnp.ones((n_chunks, capacity), dtype=jnp.float32),
        reset=jnp.zeros((n_chunks, capacity), dtype=jnp.float32),
        sites=jnp.full((n_chunks, capacity, dim), 1e3, dtype=jnp.float32),
        live=jnp.zeros((n_chunks, capacity), dtype=jnp.float32),
        retry=jnp.zeros((n_chunks,), dtype=jnp.int32),
    )


def round_robin_plan(n_chunks: int, capacity: int, dim: int) -> WritePlan:
    """A controller-free plan (every chunk admitted, slots cycled).

    ⚠ **Not a substitute for the controller** — it is the *instrument* used by
    unit tests and by the gradient measurements, where the point is the gradient
    path and running B x L Python controllers per step would dominate the clock.
    Any reported performance number uses the real controller's plan.
    """
    return WritePlan(
        slot=jnp.arange(n_chunks, dtype=jnp.int32) % int(capacity),
        admitted=jnp.ones((n_chunks,), dtype=jnp.float32),
        group_scale=jnp.ones((n_chunks, capacity), dtype=jnp.float32),
        reset=jnp.zeros((n_chunks, capacity), dtype=jnp.float32),
        sites=jnp.full((n_chunks, capacity, dim), 1e3, dtype=jnp.float32),
        live=jnp.zeros((n_chunks, capacity), dtype=jnp.float32),
        retry=jnp.zeros((n_chunks,), dtype=jnp.int32),
    )


# --------------------------------------------------------------------------
# the memory cells — one interface, three real implementations + 2 controls
# --------------------------------------------------------------------------
class NullMemoryCell(eqx.Module):
    """⭐ **The +0 B trivial substitute** (D3): no state, no read, no parameters.

    The block still has its embedding, conv, MLP, norms and residual, so this arm
    measures *"what does the memory buy at all?"*. If a real cell does not beat
    it, that cell's memory is inert and no swap comparison between two inert
    cells means anything.
    """

    latent_dim: int = eqx.field(static=True)

    def init_state(self):
        return jnp.zeros((0,))

    def read(self, state, z, plan_c=None, read_mode=None, verlet=None):
        return jnp.zeros((self.latent_dim,))

    def write(self, state, z, plan_c):
        return state

    def cell_ledger(self) -> Dict[str, int]:
        return {"params": 0, "state_floats": 0, "state_bytes": 0}


class EchoMemoryCell(eqx.Module):
    """⭐ **The other +0 B substitute**: the read returns the query unchanged.

    Costs zero bytes of storage and zero parameters, and it is the laundering
    control for "the memory output is just a re-encoding of the current chunk".
    ⚠ Note this cell is **one chunk behind** by the block's read-before-write
    convention, so it is a genuine (if trivial) recurrence, not the identity.
    """

    latent_dim: int = eqx.field(static=True)

    def init_state(self):
        return jnp.zeros((self.latent_dim,))

    def read(self, state, z, plan_c=None, read_mode=None, verlet=None):
        return state

    def write(self, state, z, plan_c):
        return z

    def cell_ledger(self) -> Dict[str, int]:
        return {"params": 0, "state_floats": 0, "state_bytes": 0}


class MatchedGRUCell(eqx.Module):
    """The **mandatory system-level swap control**: a matched-state GRU cell.

    Same chunk convention, same latent width in and out, same block around it.
    ``read`` is query-dependent (``[h ; z] -> r``) so the control is not
    artificially handicapped relative to an associative memory.

    ⛔ **Arithmetic finding, not a choice** (task §5, D2): for a GRU
    ``params = Theta(h^2)`` while ``state = Theta(h)``, whereas the CLU store's
    parameters *are* its state (``params/state ~ 1.02``). Matched parameters and
    matched state-bytes therefore **cannot both be hit**; see
    :func:`solve_matched_gru` and :func:`swap_ledger`, which publish both columns.
    """

    cell: eqx.nn.GRUCell
    out: eqx.nn.Linear
    h0: jnp.ndarray
    hidden: int = eqx.field(static=True)
    latent_dim: int = eqx.field(static=True)

    def __init__(self, latent_dim: int, hidden: int, *, key):
        k1, k2 = jax.random.split(key, 2)
        self.latent_dim = int(latent_dim)
        self.hidden = int(hidden)
        self.cell = eqx.nn.GRUCell(int(latent_dim), int(hidden), key=k1)
        self.out = eqx.nn.Linear(int(hidden) + int(latent_dim), int(latent_dim), key=k2)
        # A LEARNED initial state: parameters, per PREREG-Bprime §4's
        # learned-initial-state rule. Only the per-sequence deviation is STATE.
        self.h0 = jnp.zeros((int(hidden),))

    def init_state(self):
        return self.h0

    def read(self, state, z, plan_c=None, read_mode=None, verlet=None):
        return self.out(jnp.concatenate([state, z]))

    def write(self, state, z, plan_c):
        return self.cell(z, state)

    def cell_ledger(self) -> Dict[str, int]:
        p = int(sum(x.size for x in jax.tree_util.tree_leaves(
            eqx.filter(self, eqx.is_inexact_array))))
        return {"params": p, "state_floats": self.hidden,
                "state_bytes": 4 * self.hidden}


class MatchedTTTCell(eqx.Module):
    """A minimal faithful **TTT-class** test-time-memory cell (the second swap).

    The hidden state is a linear map ``W in R^{k x n}`` updated at test time by
    one gradient step on a self-supervised reconstruction loss (Sun et al. 2024,
    TTT-Linear), at **chunk granularity**::

        k_c = theta_K z ;  v_c = theta_V z
        W <- W - eta * grad_W || W k_c - v_c ||^2      (one closed-form step)
        r  = theta_O (W q_c),   q_c = theta_Q z

    ⭐ **This is the arm that can match BOTH axes.** Its state ``W`` is ``k*n``
    floats and its parameters are ``W0`` (the learned initial state — parameters,
    per the learned-initial-state rule) plus four thin projections, so
    ``params/state -> 1`` exactly as the CLU store's does. ``k`` and ``n`` are
    solved for by :func:`solve_matched_ttt`.
    """

    W0: jnp.ndarray
    theta_K: jnp.ndarray
    theta_V: jnp.ndarray
    theta_Q: jnp.ndarray
    theta_O: jnp.ndarray
    log_eta: jnp.ndarray
    k: int = eqx.field(static=True)
    n: int = eqx.field(static=True)
    latent_dim: int = eqx.field(static=True)

    def __init__(self, latent_dim: int, k: int, n: int, *, key):
        ks = jax.random.split(key, 5)
        d = int(latent_dim)
        self.latent_dim, self.k, self.n = d, int(k), int(n)
        self.W0 = jnp.zeros((int(k), int(n)))
        s = 1.0 / max(d, 1) ** 0.5
        self.theta_K = jax.random.normal(ks[0], (int(n), d)) * s
        self.theta_V = jax.random.normal(ks[1], (int(k), d)) * s
        self.theta_Q = jax.random.normal(ks[2], (int(n), d)) * s
        self.theta_O = jax.random.normal(ks[3], (d, int(k))) / max(int(k), 1) ** 0.5
        self.log_eta = jnp.asarray(0.0)

    def init_state(self):
        return self.W0

    def read(self, state, z, plan_c=None, read_mode=None, verlet=None):
        return self.theta_O @ (state @ (self.theta_Q @ z))

    def write(self, state, z, plan_c):
        kk = self.theta_K @ z            # (n,)
        vv = self.theta_V @ z            # (k,)
        err = state @ kk - vv            # (k,)
        eta = jax.nn.softplus(self.log_eta)
        return state - eta * jnp.outer(err, kk)

    def cell_ledger(self) -> Dict[str, int]:
        p = int(sum(x.size for x in jax.tree_util.tree_leaves(
            eqx.filter(self, eqx.is_inexact_array))))
        return {"params": p, "state_floats": self.k * self.n,
                "state_bytes": 4 * self.k * self.n}


class StoreState(NamedTuple):
    """The CLU store's **per-sequence writable state** — the honest STATE column.

    ``centers``/``log_width``/``amp`` are the three learned atom leaves. Their
    *initialisation* lives in the cell's ``V0`` and is **PARAMETERS**; what is
    carried here and mutated by the stream is the **STATE** (PREREG-Bprime §4's
    learned-initial-state rule, applied to ``V_theta`` exactly as to a GRU's
    ``h0``). ``codebook`` is the retained derived address per live item
    (`controller-doctrine` I-1) — without it monitor #2 has no runtime form and
    the same-keys launder cannot be built, so retaining it is *conservative*: it
    strengthens the control we are scored against, and it is counted as state.
    """

    centers: jnp.ndarray
    log_width: jnp.ndarray
    amp: jnp.ndarray
    codebook: jnp.ndarray


def localize_atom_init(store, centers, radius: float, *, key):
    """⭐ **H1 — the N98 localized atom init, applied to a built ``LearnedVStore``.**

    Reproduces :class:`~chlu.core.memory_potentials.AtomDictionaryPotential`'s own
    localization **bit-for-bit** (same ``_uniform_ball``, same owner map, same
    ``fold_in(key, 1)`` sub-stream, same "leading ``L`` coordinates only" rule),
    but from the outside — because
    :class:`~chlu.core.clu_system.LearnedVStore` never forwards
    ``CluSystemConfig.atom_local_radius`` to ``DesignFreedomPotential``, so the
    shipped lever is unreachable through the store's own config. Bit-identity
    with the direct construction is asserted in ``tests/test_placement_probe.py``.

    Args:
        store: a built ``LearnedVStore`` (or anything with ``.V.learned``).
        centers: ``(n_groups, L)`` localization targets, ``L <= dim``. ⭐ **Address
            axes only** (N46): the payload axis keeps its ``N(0, init_scale)``
            scatter, or the write is handed the value it is supposed to learn.
        radius: ball radius; ``<= 0`` returns ``store`` unchanged.
        key: the SAME key the store's atoms were drawn with (``LearnedVStore``'s
            ``key``), or the reproduction is not bit-identical.
    """
    from chlu.core.memory_potentials import _atom_group_owner, _uniform_ball

    if radius is None or float(radius) <= 0.0 or centers is None:
        return store
    atoms = store.V.learned
    gc = jnp.asarray(np.asarray(centers, dtype=np.float32), dtype=atoms.centers.dtype)
    if gc.ndim != 2:
        raise ValueError(f"centers must be 2-D, got shape {gc.shape}")
    n_atoms = int(atoms.centers.shape[0])
    n_g = int(atoms.n_groups)
    if int(gc.shape[0]) != n_g:
        raise ValueError(f"centers has {gc.shape[0]} rows for {n_g} groups")
    local_dims = int(gc.shape[1])
    if local_dims > int(atoms.centers.shape[1]):
        raise ValueError(f"centers width {local_dims} exceeds dim "
                         f"{atoms.centers.shape[1]}")
    owner = jnp.asarray(_atom_group_owner(n_atoms, n_g))
    k_ball = jax.random.fold_in(key, 1)     # default path's stream untouched
    offs = _uniform_ball(k_ball, n_atoms, local_dims) * float(radius)
    new = atoms.centers.at[:, :local_dims].set(gc[owner] + offs)
    return eqx.tree_at(lambda s: s.V.learned.centers, store, new)


class CluStoreCell(eqx.Module):
    """⭐⭐ **The full C2W1 CLU store as a block's memory.**

    Every full-CLU lever of intervention §4 is live (ruling 0.1 — none turned
    off): items held in a learned ``V_theta`` (never arrays), derived addressing,
    admission, per-item lifetimes, masked/local write, permitted basin
    interaction, learned ``phi`` in (owned by :class:`StreamBlock`, shared with
    every arm), **learned trajectory ``psi`` out**, two-phase relaxation, mass and
    friction as trainable selectors, trajectory *and* settled point available to
    ``psi``, confidence-gated retry, and the controller's verb set (replayed
    through :class:`WritePlan`).

    **The write** is the shipped masked objective
    (:func:`chlu.training.train_memory.write_loss`) applied for
    ``write_inner_steps`` differentiably-unrolled SGD steps, with the update
    masked to the slot's own atom rows by
    :func:`chlu.core.memory_potentials.atom_write_mask_fn` — i.e. C3-local in
    parameter space, exactly as :meth:`CluSystem._write_item` is.

    **The read** is the shipped two-phase relaxation: phase 1 at
    ``gamma_address`` (the address of an item is *derived* — wherever the query
    relaxes to), phase 2 at ``gamma_read`` (where the value is read), both
    recorded into a strided buffer that ``psi`` pools.
    """

    clu: CHLU                       # potential_net = V_theta init (PARAMETERS)
    psi: eqx.Module                 # learned trajectory read-out
    log_gamma_addr: jnp.ndarray
    log_gamma_read: jnp.ndarray
    #: (K, n_atoms) BOOL buffer, one row per atom group. Held as a non-static
    #: **integer** leaf on purpose: static numpy arrays are unhashable (equinox
    #: warns) and a float leaf would be counted as a parameter by the ledger.
    group_matrix: jnp.ndarray
    cfg: Any = eqx.field(static=True)        # CluSystemConfig
    mcfg: StreamMemoryConfig = eqx.field(static=True)
    latent_dim: int = eqx.field(static=True)

    def __init__(self, cfg, mcfg: StreamMemoryConfig, *, key):
        from chlu.core.clu_system import LearnedVStore
        from chlu.experiments.goldstone_harness import clu_with_potential
        from chlu.core.psi_readout import DeepSetsPsi, PsiSpec

        k_store, k_psi = jax.random.split(key, 2)
        self.cfg = cfg
        self.mcfg = mcfg
        self.latent_dim = int(cfg.dim)
        store = LearnedVStore(cfg, k_store)
        # ⭐ H1 (`pilot-placement-probe`): the N98 localized atom init. A no-op
        # (bit-identical) at the default `atom_local_radius = 0.0`.
        store = localize_atom_init(store, mcfg.atom_group_centers,
                                   float(mcfg.atom_local_radius), key=k_store)
        self.clu = clu_with_potential(
            store.V, dim=int(cfg.dim), kinetic_mode=str(cfg.kinetic_mode),
            inertia=jnp.ones(int(cfg.dim)),
        )
        # ⚠ psi's ``input_mode`` is fixed to "trajectory" at CONSTRUCTION so the
        # settled-point control has **bit-identical parameters** (matched_pair's
        # rule): the two read modes must differ only in what the dynamics hand
        # psi, never in psi itself. The mode is a runtime argument of ``read``.
        spec = PsiSpec(dim=int(cfg.dim), addr_dim=int(cfg.addr_dim),
                       payload_dim=int(cfg.dim),   # psi emits the whole latent
                       hidden=int(mcfg.psi_hidden), depth=2,
                       input_mode="trajectory", representation="raw",
                       include_momentum=True, include_time=True, stride=1)
        self.psi = DeepSetsPsi(spec, k_psi)
        self.log_gamma_addr = jnp.log(jnp.asarray(float(cfg.gamma_address)))
        self.log_gamma_read = jnp.log(jnp.asarray(float(cfg.gamma_read)))
        n_atoms = int(store.V.learned.centers.shape[0])
        K = int(cfg.capacity)
        gm = np.zeros((K, n_atoms), dtype=bool)
        for s in range(K):
            gm[s] = np.asarray(store.V.learned.group_rows(s), dtype=bool)
        self.group_matrix = jnp.asarray(gm)

    # -- state -------------------------------------------------------------
    @property
    def _atoms(self):
        return self.clu.potential_net.learned

    def init_state(self) -> StoreState:
        a = self._atoms
        return StoreState(centers=a.centers, log_width=a.log_width, amp=a.amp,
                          codebook=jnp.zeros((int(self.cfg.capacity),
                                              int(self.cfg.dim))))

    def _model(self, state: StoreState):
        return eqx.tree_at(
            lambda m: [m.potential_net.learned.centers,
                       m.potential_net.learned.log_width,
                       m.potential_net.learned.amp],
            self.clu, replace=[state.centers, state.log_width, state.amp])

    @property
    def gammas(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """``(gamma_address, gamma_read)`` — the **trainable friction selectors**.

        Friction is the ~14x stronger of the two selector channels (§A13 design
        rule) and, like the mass, is trainable ONLY through the trajectory read.
        """
        ga = jnp.exp(self.log_gamma_addr)
        gr = jnp.exp(self.log_gamma_read)
        if not self.mcfg.trainable_gamma:
            ga, gr = jax.lax.stop_gradient(ga), jax.lax.stop_gradient(gr)
        return ga, gr

    # -- the read ----------------------------------------------------------
    def read(self, state: StoreState, z: jnp.ndarray, plan_c=None,
             read_mode: Optional[str] = None,
             verlet: Optional[Tuple[int, int]] = None) -> jnp.ndarray:
        """Two-phase relaxation read; ``psi`` pools the strided trajectory.

        ⭐ **The confidence-gated retry is live** (ruling 0.1: no lever off). The
        gate's verdict is a *discrete* decision, so — like every other verb — it
        is taken concretely in the plan pass and replayed here as
        ``plan_c.retry``. The extra phase-2 round is **always computed and then
        gated**, because a data-dependent number of Verlet steps is not a static
        shape. ⚠ The compute is therefore paid on every chunk and the benefit
        only where the gate fires: that price is declared in the report's Verlet
        budget, not hidden.
        """
        from chlu.core.clu_system import ReadState
        from chlu.core.implicit_grad import SettleSpec, implicit_settle, truncated_rollout

        m = self._model(state)
        d = int(self.cfg.dim)
        # launch on the payload-zero manifold: the read must RECOVER the payload
        # from V_theta, never be handed it (the anti-decoration guard).
        q0 = z.at[int(self.cfg.addr_dim):].set(0.0)
        p0 = jnp.zeros_like(q0)
        ga, gr = self.gammas
        st = int(self.mcfg.traj_stride)
        mode = self.mcfg.read_mode if read_mode is None else str(read_mode)
        # ⭐ D5's compute dial: the per-read Verlet budget as a RUNTIME override.
        # Step counts must be static (lax.scan), so an override is a Python pair
        # and re-triggers compilation — fine for an evaluation sweep, which is
        # the only place it is used. ⚠ SHAPE claim only (§A3): the anytime figure
        # is occupied (DEQs, EBTs, Titans-Revisited); no uniqueness is claimed.
        n_addr = int(self.mcfg.address_steps if verlet is None else verlet[0])
        n_read = int(self.mcfg.read_steps if verlet is None else verlet[1])
        tr1, q_addr, p_addr = truncated_rollout(
            m, q0, p0, n_addr, float(self.cfg.dt), ga,
            retain=None, stride=st, return_endpoint=True)
        tr2, q_star, p_star = truncated_rollout(
            m, q_addr, p_addr, n_read, float(self.cfg.dt), gr,
            retain=None, stride=st, return_endpoint=True)
        if mode == "settled_point":
            # ⭐ the pre-registered ZERO-GRADIENT control: an implicit settle, so
            # d q*/d q0 = 0 EXACTLY rather than as a 1e-9 numerical accident.
            # ⚠ SettleSpec needs a CONCRETE gamma. Using the configured constant
            # rather than the (traced, trainable) selector is exact, not a
            # shortcut: the fixed point of the damped map is gamma-INDEPENDENT
            # (trainability-spike measured the (gamma, dt) spread of the implicit
            # answer at 2.50e-11 over gamma in {0.02..0.3}), so gamma changes only
            # how fast the settle arrives, never where.
            spec = SettleSpec(steps=n_read, dt=float(self.cfg.dt),
                              gamma=float(self.cfg.gamma_read), ridge=0.0)
            q_star = implicit_settle(m, jax.lax.stop_gradient(q_addr),
                                     jax.lax.stop_gradient(p_addr), spec)
            p_star = jax.lax.stop_gradient(p_star)
        if plan_c is not None and self.mcfg.retry_rounds > 0:
            w = jnp.asarray(plan_c.retry, dtype=jnp.float32)
            tr3, q3, p3 = truncated_rollout(
                m, q_star, p_star, n_read, float(self.cfg.dt), gr,
                retain=None, stride=st, return_endpoint=True)
            tr2 = jnp.concatenate([tr2, w * tr3 + (1.0 - w) * tr2[-1:][
                jnp.zeros((tr3.shape[0],), dtype=jnp.int32)]], axis=0)
            q_star = w * q3 + (1.0 - w) * q_star
            p_star = w * p3 + (1.0 - w) * p_star
        if mode == "settled_point":
            traj = jnp.concatenate([q_star, p_star])[None, None, :]
        else:
            traj = jnp.concatenate([tr1, tr2], axis=0)[None, ...]
        rs = ReadState(q0=q0[None], p0=p0[None], q_addr=q_addr[None],
                       p_addr=p_addr[None], q_star=q_star[None], p_star=p_star[None])
        _ = d
        return self.psi(traj, rs)[0]

    def read_diag(self, state: StoreState, z: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """Residual / rho_conv at this read — the monitors' runtime inputs."""
        m = self._model(state)
        q0 = z.at[int(self.cfg.addr_dim):].set(0.0)
        p0 = jnp.zeros_like(q0)
        ga, gr = self.gammas
        _, q_addr, p_addr = truncated_rollout_endpoint(
            m, q0, p0, int(self.mcfg.address_steps), float(self.cfg.dt), ga)
        _, q_star, _ = truncated_rollout_endpoint(
            m, q_addr, p_addr, int(self.mcfg.read_steps), float(self.cfg.dt), gr)
        g0 = jnp.linalg.norm(jax.grad(lambda q: m.potential_net(q))(q0))
        gs = jnp.linalg.norm(jax.grad(lambda q: m.potential_net(q))(q_star))
        return {"residual": gs, "grad0": g0, "rho_conv": gs / jnp.maximum(g0, 1e-12),
                "q_star": q_star}

    # -- the write ---------------------------------------------------------
    def write(self, state: StoreState, z: jnp.ndarray, plan_c) -> StoreState:
        """Masked, local, differentiably-unrolled write into ``V_theta``."""
        from chlu.core.memory_potentials import atom_write_mask_fn
        from chlu.training.train_memory import write_loss

        K = int(self.cfg.capacity)
        gm = self.group_matrix.astype(jnp.float32)              # (K, n_atoms)
        slot_oh = jax.nn.one_hot(plan_c.slot, K)                # (K,)
        row_mask = slot_oh @ gm                                  # (n_atoms,)

        # 1. lifetimes + eviction, applied BEFORE the write (physical: the item's
        #    own wells shallow; an evicted group is RE-DRAWN from the init
        #    distribution, never zeroed).
        amp_scale = plan_c.group_scale @ gm                      # (n_atoms,)
        reset_rows = plan_c.reset @ gm                           # (n_atoms,) in {0,1}
        a0 = self._atoms
        centers = jnp.where(reset_rows[:, None] > 0.5, a0.centers, state.centers)
        log_width = jnp.where(reset_rows > 0.5, a0.log_width, state.log_width)
        amp = jnp.where(reset_rows > 0.5, a0.amp, state.amp * amp_scale)

        st = StoreState(centers, log_width, amp, state.codebook)

        # 1b. ⭐ H1b — LOCALIZED PLACEMENT AT WRITE. The slot's own atoms have
        #     their ADDRESS coordinates re-drawn into a ball of
        #     `atom_place_radius` around the INCOMING chunk's address (payload
        #     coordinates untouched — N46). Off by default, bit-identical.
        #     The offsets are a fixed, key-free geometric jig, so this adds no
        #     parameter and no state byte; the update is restricted to
        #     `row_mask`, so C3 locality holds.
        #     ⚠ It is applied to `st_w` — the state the write STARTS FROM — and
        #     NOT to `st`, because `st` is what the admission blend at step 3
        #     falls back to: a refused offer must leave `V_theta` bit-identical,
        #     and placing atoms for an offer that is then refused would break
        #     exactly that contract (caught by
        #     `test_a_refused_offer_leaves_the_landscape_bit_identical_under_placement`).
        st_w = st
        if float(self.mcfg.atom_place_radius) > 0.0:
            from chlu.core.memory_potentials import _uniform_ball

            d_addr = int(self.cfg.addr_dim)
            n_atoms = int(centers.shape[0])
            jig = _uniform_ball(jax.random.PRNGKey(0xB0CA), n_atoms, d_addr)
            jig = jig * float(self.mcfg.atom_place_radius)
            placed = z[:d_addr][None, :] + jig                   # (n_atoms, d)
            sel = (row_mask > 0.5)[:, None]
            st_w = StoreState(
                centers.at[:, :d_addr].set(
                    jnp.where(sel, placed, centers[:, :d_addr])),
                log_width, amp, state.codebook)

        # 2. the masked local write, unrolled `write_inner_steps` times.
        target = z[None, :]                                      # (1, dim)
        crowd = plan_c.sites                                     # (K, dim)
        lr = float(self.mcfg.write_lr)
        mask_apply = atom_write_mask_fn(row_mask)
        key = jax.random.PRNGKey(0)

        # ⭐ H2 (`pilot-placement-probe`): the C2W2 trajectory-information term.
        # Added ONLY when the coefficient is a Python float > 0, so at the
        # default 0.0 not one extra op is traced and the written V is
        # bit-identical to the shipped objective (`write_loss`'s own
        # coefficient-zero regression gate). The rollout is the READ's phase-2
        # budget and friction, so the path the write is asked to shape is the
        # path the read actually traverses.
        lam_traj = float(self.mcfg.write_lambda_traj)
        traj_kw = dict(
            rollout_steps=int(self.mcfg.write_traj_steps
                              or self.mcfg.read_steps),
            stride=int(self.mcfg.traj_stride),
            gamma=float(self.cfg.gamma_read), dt=float(self.cfg.dt),
            n_launch=int(self.mcfg.write_traj_n_launch),
        )

        def loss_of(V):
            return write_loss(
                V, target, key,
                n_perturb=int(self.mcfg.write_n_perturb),
                sigma_addr=float(self.cfg.write_sigma_addr),
                sigma_pay=float(self.cfg.write_sigma_pay),
                margin=float(self.cfg.write_margin),
                barrier=float(self.cfg.write_barrier),
                payload_index=int(self.cfg.addr_dim),
                barrier_pairs="nn", crowd_targets=crowd,
                payload_dim=int(self.cfg.payload_dim),
                lambda_traj=lam_traj,
                traj_kwargs=(traj_kw if lam_traj > 0.0 else None),
            )

        V = self._model(st_w).potential_net
        for _ in range(int(self.mcfg.write_inner_steps)):
            g = eqx.filter_grad(loss_of)(V)
            g = mask_apply(g)
            V = eqx.apply_updates(V, jax.tree_util.tree_map(
                lambda x: (-lr * (jnp.sign(x) if self.mcfg.write_sign else x))
                if eqx.is_inexact_array(x) else None, g))
        new = V.learned

        # 3. admission: a refused offer leaves the landscape bit-identical.
        adm = plan_c.admitted
        blend = lambda a, b: adm * a + (1.0 - adm) * b  # noqa: E731
        cb = st.codebook * (1.0 - slot_oh[:, None]) + \
            adm * slot_oh[:, None] * z[None, :]
        return StoreState(
            centers=blend(new.centers, st.centers),
            log_width=blend(new.log_width, st.log_width),
            amp=blend(new.amp, st.amp),
            codebook=cb,
        )

    # -- ledger ------------------------------------------------------------
    def cell_ledger(self) -> Dict[str, int]:
        """Two-sided byte ledger for this cell (learned-initial-state rule).

        ``params`` = ``V_theta`` **initialisation** + ``psi`` + the selectors.
        ``state`` = the per-sequence ``V_theta`` **deviation** + the retained
        codebook. Both declared, exactly as the GRU's ``h0``/``h`` are.
        """
        p = int(sum(x.size for x in jax.tree_util.tree_leaves(
            eqx.filter(self, eqx.is_inexact_array))))
        a = self._atoms
        v_floats = int(a.centers.size + a.log_width.size + a.amp.size)
        s = v_floats + int(self.cfg.capacity) * int(self.cfg.dim)
        return {"params": p, "state_floats": s, "state_bytes": 4 * s,
                "v_theta_floats": v_floats, "n_atoms": int(a.centers.shape[0])}


def truncated_rollout_endpoint(model, q0, p0, steps, dt, gamma):
    """Forward-only rollout returning ``(None, q_end, p_end)`` (diagnostics)."""
    from chlu.core.implicit_grad import truncated_rollout

    tr, q, p = truncated_rollout(model, jax.lax.stop_gradient(q0),
                                 jax.lax.stop_gradient(p0), int(steps), float(dt),
                                 jax.lax.stop_gradient(gamma), retain=0,
                                 stride=max(int(steps), 1), return_endpoint=True)
    return tr, q, p


# --------------------------------------------------------------------------
# the shared shell: identical in EVERY arm, so the swap is genuinely a swap
# --------------------------------------------------------------------------
class StreamPhi(eqx.Module):
    """The shared read-in ``d_model -> latent``. ⭐ **Bit-identical in every arm.**

    One MLP with two *uses*, never two parameter sets:

    * :meth:`query` — the read launch, with the **payload channels forced to
      zero**. This is ``LearnedPhi``'s anti-decoration rule: the read must
      recover the payload from ``V_theta``, and a phi that could write the
      payload into its own launch point would be reading the answer off its
      input.
    * :meth:`__call__` — the full latent, which is what gets *written* (address
      block + payload block) and what the swap cells consume.

    Its byte count is ledgered on **every** arm (task dial declaration:
    "identical embedding + identical phi, with phi-bytes ledgered on every arm").
    """

    dim: int = eqx.field(static=True)
    addr_dim: int = eqx.field(static=True)
    payload_dim: int = eqx.field(static=True)
    net: eqx.nn.MLP
    gain: float = eqx.field(static=True)

    def __init__(self, d_model: int, dim: int, addr_dim: int, payload_dim: int,
                 hidden: int = 64, gain: float = 1.0, *, key):
        self.dim, self.addr_dim, self.payload_dim = int(dim), int(addr_dim), int(payload_dim)
        self.gain = float(gain)
        self.net = eqx.nn.MLP(int(d_model), int(dim), int(hidden), 1,
                              activation=jax.nn.tanh, key=key)

    def __call__(self, u: jnp.ndarray) -> jnp.ndarray:
        return self.gain * self.net(u)

    def query(self, u: jnp.ndarray) -> jnp.ndarray:
        z = self(u)
        return z.at[self.addr_dim: self.addr_dim + self.payload_dim].set(0.0)


class StreamBlock(eqx.Module):
    """⭐ The tier-iii block: ``norm -> causal conv -> [MEMORY CELL] -> MLP``, residual.

    Everything except ``cell`` is **bit-identical across arms** — the same
    embedding upstream, the same LayerNorms, the same intra-chunk causal
    depthwise convolution, the same ``phi``, the same assimilation projection,
    the same token-wise MLP, the same residual structure. That is what makes the
    system-level swap a *swap* and not a redesign.

    **The chunk contract** (§0.3, identical in every arm):

    1. an intra-chunk causal depthwise conv does the *local* mixing, so the
       memory carries only **cross-chunk** information;
    2. chunk ``c-1`` is pooled to ``u_{c-1}``, and ``z_{c-1} = phi(u_{c-1})``;
    3. **read first**: ``r_{c-1} = cell.read(state, z_{c-1})`` against a store
       holding chunks ``0..c-2``;
    4. **then write**: ``state <- cell.write(state, z_{c-1}, plan[c-1])``;
    5. ``r_{c-1}`` is assimilated and added to **every token of chunk ``c``**.

    Step 3-before-4 is not a stylistic choice: reading a store that was just
    handed the current chunk is an echo, and it would let a memory that stores
    nothing score like a memory that stores everything. Step 5's one-chunk shift
    is what makes the whole block causal.
    """

    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm
    norm3: eqx.nn.LayerNorm
    conv_w: jnp.ndarray
    phi: StreamPhi
    cell: eqx.Module
    assim: eqx.nn.Linear
    mlp_in: eqx.nn.Linear
    mlp_out: eqx.nn.Linear
    mcfg: StreamMemoryConfig = eqx.field(static=True)
    latent_dim: int = eqx.field(static=True)

    def __init__(self, d_model: int, cell: eqx.Module, mcfg: StreamMemoryConfig,
                 *, latent_dim: int, addr_dim: int, payload_dim: int, key):
        ks = jax.random.split(key, 4)
        self.mcfg = mcfg
        self.latent_dim = int(latent_dim)
        self.cell = cell
        self.norm1 = eqx.nn.LayerNorm(d_model)
        self.norm2 = eqx.nn.LayerNorm(d_model)
        self.norm3 = eqx.nn.LayerNorm(d_model)
        # depthwise causal conv, zero-initialised beyond the identity tap so the
        # block starts as a clean residual stream.
        self.conv_w = jnp.zeros((int(mcfg.conv_kernel), d_model)).at[-1].set(1.0)
        # ⭐ addr/payload are declared by the MODEL, not read off the cell, so
        # every arm's phi is constructed with identical arguments and identical
        # key -> bit-identical parameters (the dial declaration's requirement).
        self.phi = StreamPhi(d_model, latent_dim, addr_dim, payload_dim,
                             gain=float(mcfg.phi_gain), key=ks[0])
        self.assim = eqx.nn.Linear(int(latent_dim), d_model, key=ks[1])
        h = int(mcfg.mlp_mult) * d_model
        self.mlp_in = eqx.nn.Linear(d_model, h, key=ks[2])
        self.mlp_out = eqx.nn.Linear(h, d_model, key=ks[3])

    def _causal_conv(self, x):
        k = int(self.mcfg.conv_kernel)
        pad = jnp.zeros((k - 1, x.shape[-1]), dtype=x.dtype)
        xp = jnp.concatenate([pad, x], axis=0)
        wins = jnp.stack([xp[i: i + x.shape[0]] for i in range(k)], axis=0)  # (k,T,D)
        return jnp.einsum("ktd,kd->td", wins, self.conv_w)

    def chunk_latents(self, x: jnp.ndarray) -> jnp.ndarray:
        """``(n_chunks, latent)`` — steps 1-2 only, for the **concrete plan pass**.

        Shared verbatim with :meth:`__call__`, so the controller sees exactly the
        latents the differentiable pass will write.
        """
        T, D = x.shape
        C = int(self.mcfg.chunk)
        x = x + self._causal_conv(jax.vmap(self.norm1)(x))
        u = jnp.mean(jax.vmap(self.norm2)(x).reshape(T // C, C, D), axis=1)
        return jax.vmap(self.phi)(u)

    def __call__(self, x: jnp.ndarray, plan, read_mode: Optional[str] = None,
                 verlet: Optional[Tuple[int, int]] = None) -> jnp.ndarray:
        """``x`` is ``(T, d_model)`` for ONE sequence; ``plan`` is a :class:`WritePlan`.

        ``read_mode`` overrides the cell's configured read at call time — that is
        how the S2 gradient probe compares the trajectory read against the
        settled-point read **on the same model with the same parameters**.
        """
        T, D = x.shape
        C = int(self.mcfg.chunk)
        n_chunks = T // C
        # 1. local mixing (identical in every arm)
        x = x + self._causal_conv(jax.vmap(self.norm1)(x))
        # 2. per-chunk pooled summary -> latent
        u = jnp.mean(jax.vmap(self.norm2)(x).reshape(n_chunks, C, D), axis=1)
        z = jax.vmap(self.phi)(u)                                   # (n_chunks, dim)

        # 3-4. read-before-write, scanned over chunk boundaries
        def step(state, inp):
            z_c, plan_c = inp
            r = self.cell.read(state, z_c, plan_c, read_mode, verlet)
            state = self.cell.write(state, z_c, plan_c)
            return state, r

        _, r = jax.lax.scan(step, self.cell.init_state(), (z, plan))  # (n_chunks, dim)
        # 5. shift by one chunk: chunk c is told what the store knew BEFORE it.
        r = jnp.concatenate([jnp.zeros((1, r.shape[-1]), dtype=r.dtype), r[:-1]], axis=0)
        m = jax.vmap(self.assim)(r)                                  # (n_chunks, D)
        x = x + jnp.repeat(m, C, axis=0)
        # 6. token-wise MLP (identical in every arm)
        hh = jax.vmap(self.norm3)(x)
        return x + jax.vmap(lambda t: self.mlp_out(jax.nn.gelu(self.mlp_in(t))))(hh)


class StreamModel(eqx.Module):
    """Embedding -> ``n_layers`` x :class:`StreamBlock` -> norm -> head.

    The tier-iii shell. Held FIXED across arms; only ``blocks[i].cell`` changes.
    Discrete tokens only (enwik8/WT-103 are byte/token streams).
    """

    embed: eqx.nn.Embedding
    pos: jnp.ndarray
    blocks: list
    norm_f: eqx.nn.LayerNorm
    head: eqx.nn.Linear
    n_layers: int = eqx.field(static=True)
    chunk: int = eqx.field(static=True)

    def __init__(self, *, vocab_size: int, d_model: int, n_layers: int,
                 max_len: int, cells: list, mcfg: StreamMemoryConfig,
                 latent_dim: int, addr_dim: int, payload_dim: int, key):
        ks = jax.random.split(key, n_layers + 3)
        self.n_layers = int(n_layers)
        self.chunk = int(mcfg.chunk)
        self.embed = eqx.nn.Embedding(vocab_size, d_model, key=ks[0])
        self.pos = jax.random.normal(ks[1], (max_len, d_model)) * 0.02
        self.blocks = [
            StreamBlock(d_model, cells[i], mcfg, latent_dim=latent_dim,
                        addr_dim=addr_dim, payload_dim=payload_dim, key=ks[2 + i])
            for i in range(n_layers)
        ]
        self.norm_f = eqx.nn.LayerNorm(d_model)
        self.head = eqx.nn.Linear(d_model, vocab_size, key=ks[-1])

    def __call__(self, tokens: jnp.ndarray, plans, read_mode: Optional[str] = None,
                 verlet: Optional[Tuple[int, int]] = None) -> jnp.ndarray:
        """``tokens`` ``(T,)`` int -> logits ``(T, vocab)``. ``plans``: one per layer."""
        h = jax.vmap(self.embed)(tokens)
        h = h + self.pos[: h.shape[0]]
        for blk, plan in zip(self.blocks, plans, strict=True):
            h = blk(h, plan, read_mode, verlet)
        return jax.vmap(self.head)(jax.vmap(self.norm_f)(h))


# --------------------------------------------------------------------------
# ⭐ the matched-swap solvers and the two-sided byte ledger (D2)
# --------------------------------------------------------------------------
def _count(mod) -> int:
    return int(sum(x.size for x in jax.tree_util.tree_leaves(
        eqx.filter(mod, eqx.is_inexact_array))))


def solve_matched_gru(target_params: int, latent_dim: int) -> int:
    """``hidden`` whose GRU-cell parameter count is closest to ``target_params``.

    Closed form (``params(h) = 3h^2 + (4*dim + 5)h + dim^2 + dim``) followed by a
    **verification sweep over the integer neighbourhood, by construction** — so
    the number in the ledger is the number in the model, not the number in a
    formula that might drift from ``eqx.nn.GRUCell``.
    """
    d = int(latent_dim)
    a, b, c = 3.0, 4.0 * d + 5.0, float(d * d + d - int(target_params))
    h0 = max(1, int(round((-b + (b * b - 4 * a * c) ** 0.5) / (2 * a))))
    best, best_err = h0, float("inf")
    for h in range(max(1, h0 - 4), h0 + 5):
        p = _count(MatchedGRUCell(d, h, key=jax.random.PRNGKey(0)))
        e = abs(p - int(target_params))
        if e < best_err:
            best, best_err = h, e
    return best


def gru_hidden_for_state(target_state_floats: int) -> int:
    """``hidden`` that matches the CLU cell's STATE bytes — the other column.

    ⛔ Published alongside :func:`solve_matched_gru` precisely because the two
    answers are hundreds of times apart. See :func:`swap_ledger`.
    """
    return int(target_state_floats)


def solve_matched_ttt(target_params: int, target_state: int, latent_dim: int
                      ) -> Tuple[int, int]:
    """``(k, n)`` matching BOTH the CLU cell's parameters and its state bytes.

    ⭐ A TTT-class cell can do what a GRU provably cannot. Its state
    ``W in R^{k x n}`` is ``k*n`` floats and its parameters are ``W0`` (the
    learned initial state — PARAMETERS, per the learned-initial-state rule) plus
    four thin ``O(dim)`` projections::

        params = k*n + 2*dim*(k+n) + 1 ,   state = k*n

    so the two constraints decouple: ``k*n = S`` fixes the state and
    ``k + n = (P - S - 1) / (2*dim)`` fixes the parameters. Both are hit whenever
    the discriminant ``(k+n)^2 - 4*k*n >= 0``, i.e. whenever the CLU cell's
    parameter surplus over its state is large enough to pay for the projections.
    ⚠ **If it is not, the two-sided match is impossible for this cell too** and
    the caller must report that (task §5): we return the best available ``(k, n)``
    and :func:`swap_ledger` publishes the residual on both axes.
    """
    P, S, d = int(target_params), int(target_state), int(latent_dim)
    ssum = (P - S - 1) / (2.0 * d)
    disc = ssum * ssum - 4.0 * S
    if disc >= 0:
        k = max(2, int(round((ssum + disc ** 0.5) / 2.0)))
        n = max(1, int(round((ssum - disc ** 0.5) / 2.0)))
        cands = [(kk, nn) for kk in range(max(2, k - 3), k + 4)
                 for nn in range(max(1, n - 3), n + 4)]
    else:  # unreachable two-sided match: fall back to the state-matched square
        k = max(2, int(round(S ** 0.5)))
        cands = [(kk, max(1, int(round(S / kk)))) for kk in range(max(2, k - 40), k + 41)]
    best, best_err = cands[0], float("inf")
    for kk, nn in cands:
        cell = MatchedTTTCell(d, kk, nn, key=jax.random.PRNGKey(0))
        e = abs(_count(cell) - P) / max(P, 1) + abs(kk * nn - S) / max(S, 1)
        if e < best_err:
            best, best_err = (kk, nn), e
    return best


def swap_ledger(cells: Dict[str, eqx.Module], shared: Optional[Dict[str, int]] = None
                ) -> Dict[str, Dict[str, Any]]:
    """⭐ **The two-sided byte ledger**, per arm, with the matching verdicts.

    Conventions (``PREREG-Bprime.md`` §4, applied to ``V_theta`` exactly as to a
    GRU's ``h0``): **an initialisation is PARAMETERS; only the per-sequence
    deviation is STATE. Both declared.** ``shared`` is the arm-identical shell
    (embedding, positional, norms, conv, phi, assimilation, MLP, head), reported
    on every row so no arm can hide budget in it.
    """
    ref = cells.get("clu_store")
    out: Dict[str, Dict[str, Any]] = {}
    rp = ref.cell_ledger()["params"] if ref is not None else None
    rs = ref.cell_ledger()["state_floats"] if ref is not None else None
    for name, cell in cells.items():
        led = dict(cell.cell_ledger())
        if rp:
            led["params_vs_clu"] = led["params"] / rp
            led["params_matched_pct"] = 100.0 * (led["params"] - rp) / rp
        if rs:
            led["state_vs_clu"] = (led["state_floats"] / rs) if led["state_floats"] else 0.0
            led["clu_state_over_arm"] = (rs / led["state_floats"]) if led["state_floats"] else float("inf")
        if shared:
            led.update({f"shared_{k}": v for k, v in shared.items()})
        out[name] = led
    return out


def store_byte_law(atoms_per_item: int, dim: int, addr_dim: int, payload_dim: int
                   ) -> float:
    """The corrected per-item byte law ``[A(D+2)+d]/(d+m)`` (errata 24/28).

    Cost of one stored item **relative to one table row**. ⚠ The floor is
    **2.40x at ``n_spec = 1``** (``A = 1``); anything above that is the price of
    holding the item in a learned landscape instead of an array.
    """
    return (int(atoms_per_item) * (int(dim) + 2) + int(addr_dim)) / \
        float(int(addr_dim) + int(payload_dim))


def make_memory_cell(name: str, *, latent_dim: int, clu_cfg=None,
                     mcfg: Optional[StreamMemoryConfig] = None,
                     hidden: Optional[int] = None,
                     ttt_shape: Optional[Tuple[int, int]] = None, key):
    """Construct one memory cell by name (``MEMORY_CELLS``)."""
    mcfg = mcfg or StreamMemoryConfig()
    if name == "clu_store":
        return CluStoreCell(clu_cfg, mcfg, key=key)
    if name == "gru_matched":
        return MatchedGRUCell(latent_dim, int(hidden), key=key)
    if name == "ttt_matched":
        k, n = ttt_shape
        return MatchedTTTCell(latent_dim, int(k), int(n), key=key)
    if name == "none":
        return NullMemoryCell(latent_dim=int(latent_dim))
    if name == "echo":
        return EchoMemoryCell(latent_dim=int(latent_dim))
    raise ValueError(f"Unknown memory cell: {name}. Must be one of {MEMORY_CELLS}.")
