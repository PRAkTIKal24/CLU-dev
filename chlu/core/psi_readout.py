"""Learned read-outs ``psi`` over the **strided read trajectory** (pillar 1).

In 26 waves the read-out was never learned and the trajectory was never read
(intervention §3.2, §3.7). ``CluSystem.read()`` returns a strided trajectory
buffer *and* ``q*`` precisely so that point-vs-trajectory is a **configuration
change, not a rewrite** — and so the ablation is **internal**: the settled-point
read *is* the "trajectory deleted" substitute, on the same harness, same bytes,
same ``phi``, same parameters.

Two pooling families (charter §6.4 names both):

* :class:`DeepSetsPsi` — permutation-invariant ``rho(pool_i(enc(x_i)))``.
* :class:`AttentionPsi` — a learned query attends over the encoded points.

⭐ **Both accept the settled-point-only input as a degenerate case** — the same
module, the same parameters, the same parameter count; only ``input_mode``
changes which points enter the set. That is what makes the ablation fair:

===================  ====================================================
``input_mode``       the set fed to the pooling
===================  ====================================================
``settled_point``    ``{[q*, p*]}``                       (1 point)
``endpoints``        ``{[q0, p0], [q*, p*]}``             (2 points)
``trajectory``       the whole strided buffer             (n points)
===================  ====================================================

⚠ **The trajectory contains ``q0 = phi(x)``.** A psi over the raw buffer has
direct access to the query embedding, which is exactly the N68 configuration
(blank stores scored 0.992–1.000). ``representation="store_relative"``
implements the doctrine's I-2 form (``traj - q0``), and **every** accuracy
number produced with a learned psi must travel with
``chlu.eval.dividend.trajectory_launder``'s three-way split
(``full`` / ``q0_only`` / ``endpoints``). This module deliberately does not hide
the raw form — it makes both available so the launder has something to measure.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    "PsiSpec",
    "DeepSetsPsi",
    "AttentionPsi",
    "LearnedPhi",
    "make_psi",
    "matched_pair",
    "psi_param_count",
    "select_points",
]

INPUT_MODES = ("trajectory", "settled_point", "endpoints")
REPRESENTATIONS = ("raw", "store_relative")


@dataclass(frozen=True)
class PsiSpec:
    """Static configuration of a learned read-out.

    ⚠ Config lives **here**, not in ``chlu/config.py`` (C2W1 file-ownership rule:
    C1W27 owns two blocks of that file this wave). Override from a project YAML
    via :meth:`from_mapping`.
    """

    dim: int
    addr_dim: int
    payload_dim: int = 1
    hidden: int = 32
    depth: int = 2
    input_mode: str = "trajectory"
    representation: str = "raw"
    include_momentum: bool = True
    include_time: bool = True
    stride: int = 1  # further subsampling of the buffer handed to psi
    n_heads: int = 1  # attention only

    def __post_init__(self):
        if self.input_mode not in INPUT_MODES:
            raise ValueError(f"input_mode must be one of {INPUT_MODES}, got {self.input_mode!r}")
        if self.representation not in REPRESENTATIONS:
            raise ValueError(
                f"representation must be one of {REPRESENTATIONS}, got {self.representation!r}"
            )
        if int(self.stride) < 1:
            raise ValueError("stride must be >= 1")

    @property
    def point_features(self) -> int:
        """Per-point feature width the encoder sees."""
        n = int(self.dim) * (2 if self.include_momentum else 1)
        return n + (1 if self.include_time else 0)

    @classmethod
    def from_mapping(cls, dim: int, addr_dim: int, overrides: Optional[dict] = None
                     ) -> "PsiSpec":
        known = {"payload_dim", "hidden", "depth", "input_mode", "representation",
                 "include_momentum", "include_time", "stride", "n_heads"}
        kw = {k: v for k, v in dict(overrides or {}).items() if k in known}
        return cls(dim=int(dim), addr_dim=int(addr_dim), **kw)

    def as_flags(self) -> Dict[str, Any]:
        return {"psi_input_mode": self.input_mode,
                "psi_representation": self.representation,
                "psi_hidden": int(self.hidden), "psi_depth": int(self.depth),
                "psi_stride": int(self.stride),
                "psi_include_momentum": bool(self.include_momentum),
                "psi_n_heads": int(self.n_heads)}


# --------------------------------------------------------------------------
# point selection — the ONLY thing the ablation changes
# --------------------------------------------------------------------------
def select_points(traj: jnp.ndarray, state, spec: PsiSpec) -> jnp.ndarray:
    """Build the ``(B, n_points, point_features)`` set psi pools over.

    ``traj`` is ``(B, n, 2*dim)`` (``[q | p]`` per point, the layout
    :attr:`chlu.core.clu_system.ReadResult.traj` uses); ``state`` is a
    :class:`~chlu.core.clu_system.ReadState`.
    """
    d = int(spec.dim)
    if spec.input_mode == "trajectory":
        pts = traj[:, :: int(spec.stride), :]
    elif spec.input_mode == "settled_point":
        pts = jnp.concatenate([state.q_star, state.p_star], axis=-1)[:, None, :]
    else:  # endpoints
        pts = jnp.stack([
            jnp.concatenate([state.q0, state.p0], axis=-1),
            jnp.concatenate([state.q_star, state.p_star], axis=-1),
        ], axis=1)

    if spec.representation == "store_relative":
        # doctrine I-2: subtract the launch point so psi cannot simply be a
        # classifier on phi(x). Applied identically in every input_mode, so the
        # ablation stays internal.
        ref = jnp.concatenate([state.q0, state.p0], axis=-1)[:, None, :]
        pts = pts - ref

    if not spec.include_momentum:
        pts = pts[..., :d]
    if spec.include_time:
        n = pts.shape[1]
        t = (jnp.arange(n, dtype=pts.dtype) / max(n - 1, 1))[None, :, None]
        pts = jnp.concatenate([pts, jnp.broadcast_to(t, pts.shape[:2] + (1,))], axis=-1)
    return pts


# --------------------------------------------------------------------------
# the two pooling families
# --------------------------------------------------------------------------
class DeepSetsPsi(eqx.Module):
    """``rho( [mean_i enc(x_i) ; max_i enc(x_i)] )`` — permutation invariant.

    Degenerate at ``input_mode="settled_point"``: the set has one element, mean
    and max coincide, and the network reduces to an MLP on ``[q*, p*]`` — i.e.
    exactly the classical settled-point read, **at identical parameter count**.
    """

    enc: eqx.nn.MLP
    dec: eqx.nn.MLP
    spec: PsiSpec = eqx.field(static=True)
    representation: str = eqx.field(static=True)

    def __init__(self, spec: PsiSpec, key):
        k1, k2 = jax.random.split(key, 2)
        self.spec = spec
        self.representation = f"deepsets:{spec.input_mode}:{spec.representation}"
        self.enc = eqx.nn.MLP(spec.point_features, spec.hidden, spec.hidden,
                              max(int(spec.depth) - 1, 1), activation=jax.nn.tanh, key=k1)
        self.dec = eqx.nn.MLP(2 * spec.hidden, spec.payload_dim, spec.hidden,
                              max(int(spec.depth) - 1, 1), activation=jax.nn.tanh, key=k2)

    def __call__(self, traj: jnp.ndarray, state) -> jnp.ndarray:
        pts = select_points(traj, state, self.spec)
        h = jax.vmap(jax.vmap(self.enc))(pts)  # (B, n, hidden)
        pooled = jnp.concatenate([jnp.mean(h, axis=1), jnp.max(h, axis=1)], axis=-1)
        return jax.vmap(self.dec)(pooled)


class AttentionPsi(eqx.Module):
    """A learned query attends over the encoded trajectory points.

    ``a = softmax(<Wq q_learned, Wk h_i> / sqrt(h))``, ``out = rho(sum_i a_i Wv h_i)``.
    Degenerate at ``input_mode="settled_point"``: one point, so ``a = 1`` and the
    module is again an MLP on ``[q*, p*]`` — same parameters, same count.
    """

    enc: eqx.nn.MLP
    q_tok: jnp.ndarray  # (n_heads, head_dim)
    W_k: jnp.ndarray  # (n_heads, head_dim, hidden)
    W_v: jnp.ndarray  # (n_heads, head_dim, hidden)
    dec: eqx.nn.MLP
    spec: PsiSpec = eqx.field(static=True)
    representation: str = eqx.field(static=True)

    def __init__(self, spec: PsiSpec, key):
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        self.spec = spec
        self.representation = f"attention:{spec.input_mode}:{spec.representation}"
        nh = int(spec.n_heads)
        hd = max(int(spec.hidden) // nh, 1)
        self.enc = eqx.nn.MLP(spec.point_features, spec.hidden, spec.hidden,
                              max(int(spec.depth) - 1, 1), activation=jax.nn.tanh, key=k1)
        scale = 1.0 / np.sqrt(spec.hidden)
        self.q_tok = jax.random.normal(k2, (nh, hd)) * scale
        self.W_k = jax.random.normal(k3, (nh, hd, spec.hidden)) * scale
        self.W_v = jax.random.normal(k4, (nh, hd, spec.hidden)) * scale
        self.dec = eqx.nn.MLP(nh * hd, spec.payload_dim, spec.hidden,
                              max(int(spec.depth) - 1, 1), activation=jax.nn.tanh, key=k5)

    def __call__(self, traj: jnp.ndarray, state) -> jnp.ndarray:
        pts = select_points(traj, state, self.spec)
        h = jax.vmap(jax.vmap(self.enc))(pts)  # (B, n, hidden)
        k = jnp.einsum("hdc,bnc->bhnd", self.W_k, h)
        v = jnp.einsum("hdc,bnc->bhnd", self.W_v, h)
        logits = jnp.einsum("hd,bhnd->bhn", self.q_tok, k) / np.sqrt(k.shape[-1])
        a = jax.nn.softmax(logits, axis=-1)
        pooled = jnp.einsum("bhn,bhnd->bhd", a, v).reshape(h.shape[0], -1)
        return jax.vmap(self.dec)(pooled)


class LearnedPhi(eqx.Module):
    """Read-in ``x -> q0``, the first link of ``query -> phi -> settle -> psi -> loss``.

    Emits a launch point in the full latent, with the **payload channels forced
    to zero** — the shipped read launches on the payload-zero manifold
    (``CluSystem.read`` does ``q0[:, addr:addr+m] = 0``), and a phi that could
    write the payload directly would be reading the answer off its own input.
    """

    net: eqx.nn.MLP
    dim: int = eqx.field(static=True)
    addr_dim: int = eqx.field(static=True)
    payload_dim: int = eqx.field(static=True)
    residual: bool = eqx.field(static=True)

    def __init__(self, in_dim: int, dim: int, addr_dim: int, payload_dim: int = 1,
                 hidden: int = 32, depth: int = 2, *, residual: bool = True, key=None):
        self.dim, self.addr_dim, self.payload_dim = int(dim), int(addr_dim), int(payload_dim)
        self.residual = bool(residual)
        self.net = eqx.nn.MLP(int(in_dim), int(dim), int(hidden), max(int(depth) - 1, 1),
                              activation=jax.nn.tanh, key=key)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.atleast_2d(jnp.asarray(x))
        out = jax.vmap(self.net)(x)
        if self.residual:
            # start life as (near-)identity on the address block, so the pilot
            # begins from the shipped read rather than from a random embedding
            pad = jnp.zeros((x.shape[0], self.dim - x.shape[-1]), dtype=out.dtype)
            out = 0.1 * out + jnp.concatenate([x, pad], axis=-1)[:, : self.dim]
        z = jnp.zeros((x.shape[0], self.payload_dim), dtype=out.dtype)
        return jnp.concatenate(
            [out[:, : self.addr_dim], z, out[:, self.addr_dim + self.payload_dim:]], axis=-1
        )


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def make_psi(family: str, spec: PsiSpec, key) -> eqx.Module:
    """``family in {"deepsets", "attention"}``."""
    if family == "deepsets":
        return DeepSetsPsi(spec, key)
    if family == "attention":
        return AttentionPsi(spec, key)
    raise ValueError(f"unknown psi family {family!r} (deepsets | attention)")


def psi_param_count(psi: eqx.Module) -> int:
    """Number of learnable scalars — the matched-parameter denominator."""
    leaves = jax.tree_util.tree_leaves(eqx.filter(psi, eqx.is_inexact_array))
    return int(sum(int(np.asarray(x).size) for x in leaves))


def matched_pair(family: str, spec: PsiSpec, key) -> Tuple[eqx.Module, eqx.Module]:
    """``(point_psi, trajectory_psi)`` from the **same key** — identical
    initial parameters, identical parameter count, only ``input_mode`` differs.

    This is the ablation's fairness guarantee, in one function: a trajectory read
    that wins by being bigger is not a result.
    """
    p = make_psi(family, replace(spec, input_mode="settled_point"), key)
    t = make_psi(family, replace(spec, input_mode="trajectory"), key)
    assert psi_param_count(p) == psi_param_count(t)
    return p, t
