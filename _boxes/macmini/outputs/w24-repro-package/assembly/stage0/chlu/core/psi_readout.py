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
    # C2W3 reconciliation 1 — the AttentionPsi quarantine (charter §A11 rider)
    "AttentionPsiLeakError",
    "LeakProbe",
    "ATTENTION_PSI_LEAK_BAR",
    "ATTENTION_PSI_LEAK_EVIDENCE",
    "LearnedPhi",
    "make_psi",
    "matched_pair",
    "psi_param_count",
    "select_points",
    # C2W2 (charter §A4.3) — the read-in that parametrizes the particle
    "PHI_FAMILIES",
    "PhiSpec",
    "ParticleLaunch",
    "ParticlePhi",
    "PhiMismatchError",
    "make_phi",
    "phi_fingerprint",
    "assert_identical_phi",
    "SharedPhi",
    "phi_ledger",
    "joint_dial",
    "assert_joint_dial",
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


#: ⛔ The registered `AttentionPsi` leak bar (C2W2 `traj-write-objective` D6 /
#: spike R-4). ``chance + 3 SE`` on the family the leak was measured on.
ATTENTION_PSI_LEAK_BAR = 0.1902

#: The measured evidence, kept next to the code that must not be used without it.
ATTENTION_PSI_LEAK_EVIDENCE: Dict[str, Any] = {
    "source": "C2W2 traj-write-objective D6 (spike R-4), K=8, 2000 fit steps, "
              "params matched 4609, chance 0.1386, bar 0.1902",
    "q0_only_by_stride": {1: 0.4134, 2: 0.4332, 4: 0.4480, 8: 0.4257,
                          16: 0.3515, 32: 0.3911},
    "blank_store_by_stride": {1: 0.4059, 2: 0.4381, 4: 0.4728, 8: 0.4653,
                              16: 0.3762, 32: 0.3713},
    "full_by_stride": {1: 0.6658, 2: 0.6460, 4: 0.6460, 8: 0.6510,
                       16: 0.6386, 32: 0.6658},
    "fired": "1/1 at EVERY stride — the leak is stride-independent",
}


class AttentionPsiLeakError(RuntimeError):
    """⛔ Raised when an ``AttentionPsi`` is asked for a **store-relative**
    (trajectory) reading without a passing leak probe.

    **The evidence (C2W2 D6 / spike R-4, closed and FIRED).** An attention psi
    *selects* the launch point out of the buffer instead of diluting it, so it
    reads ``phi(x)`` rather than the store:

    ======  ======  =========  ===========  ============
    stride  full    q0_only    blank_store  leak (bar 0.1902)
    ======  ======  =========  ===========  ============
    1       0.6658  **0.4134**  **0.4059**  YES
    2       0.6460  **0.4332**  **0.4381**  YES
    4       0.6460  **0.4480**  **0.4728**  YES
    8       0.6510  **0.4257**  **0.4653**  YES
    16      0.6386  **0.3515**  **0.3762**  YES
    32      0.6658  **0.3911**  **0.3713**  YES
    ======  ======  =========  ===========  ============

    ``q0_only`` is 3.0x chance and 2.2x the bar **at every stride**, and a
    **blank store** read by an attention psi scores 0.37-0.47 — i.e. the number
    survives deleting the store entirely. A pooled DeepSets psi does not do this
    (C2W1: ``q0_only`` 0.129 vs chance 0.125, no leak): pooling dilutes ``q0`` to
    1 of 150 points, attention selects it. ⇒ **no attention-psi trajectory number
    is quotable store-relative**, and the C2W2 gate's race card is unaffected only
    because it used the gym's handcrafted psi.

    The quarantine raises rather than warns, per the ``PhiMismatchError``
    precedent: an invariant enforced in prose is not enforced. To use the module
    anyway (e.g. the bit-identity regression test, or a deliberately
    non-store-relative use) pass ``quarantine=False`` **explicitly** — that is
    greppable; a silently ignored warning is not.

    ⚠ **This does not bar a table reader that happens to use softmax attention**
    over a launder's own ``(key, payload)`` rows (``bprime-fb4-gate``'s
    ``attention`` arm). That object never sees a trajectory, cannot select ``q0``
    out of a buffer that it is not given, and is a different object entirely.
    """


@dataclass(frozen=True)
class LeakProbe:
    """The trajectory launder's verdict for one psi, as a first-class object.

    ``q0_only``/``blank_store`` are the same-scorer scores of the leak controls;
    the probe **passes** only when both sit at or below ``bar``.
    """

    q0_only: float
    blank_store: float
    bar: float = ATTENTION_PSI_LEAK_BAR
    stride: Optional[int] = None
    source: str = ""

    @property
    def leak(self) -> float:
        return float(max(self.q0_only, self.blank_store) - self.bar)

    def passed(self) -> bool:
        vals = (float(self.q0_only), float(self.blank_store), float(self.bar))
        return bool(all(np.isfinite(v) for v in vals) and self.leak <= 0.0)

    def as_dict(self) -> Dict[str, Any]:
        return {"q0_only": float(self.q0_only),
                "blank_store": float(self.blank_store), "bar": float(self.bar),
                "stride": self.stride, "source": self.source,
                "leak": self.leak, "passed": self.passed()}


class AttentionPsi(eqx.Module):
    """A learned query attends over the encoded trajectory points.

    ``a = softmax(<Wq q_learned, Wk h_i> / sqrt(h))``, ``out = rho(sum_i a_i Wv h_i)``.
    Degenerate at ``input_mode="settled_point"``: one point, so ``a = 1`` and the
    module is again an MLP on ``[q*, p*]`` — same parameters, same count.

    ⛔ **QUARANTINED for trajectory input (C2W2 reconciliation 1).** In
    ``input_mode="trajectory"`` this module reads the launch point, not the store
    — ``q0_only`` **0.35-0.45** against a bar of **0.19**, blank store
    **0.37-0.47**, *at every stride*. It therefore **refuses** to produce a
    reading in that mode unless it is handed a passing
    :class:`LeakProbe` (``__call__(..., leak_probe=probe)``) or is constructed
    with ``quarantine=False``. See :class:`AttentionPsiLeakError` for the numbers
    and for why the pooled DeepSets psi is *not* quarantined.

    ⭐ Everything else is bit-identical to the shipped module: the quarantine is
    a pure precondition check, the parameters, the maths and the outputs are
    untouched (``tests/test_psi_readout.py`` asserts bit-identity with the
    quarantine disabled).
    """

    enc: eqx.nn.MLP
    q_tok: jnp.ndarray  # (n_heads, head_dim)
    W_k: jnp.ndarray  # (n_heads, head_dim, hidden)
    W_v: jnp.ndarray  # (n_heads, head_dim, hidden)
    dec: eqx.nn.MLP
    spec: PsiSpec = eqx.field(static=True)
    representation: str = eqx.field(static=True)
    quarantine: bool = eqx.field(static=True)
    leak_probe: Optional[LeakProbe] = eqx.field(static=True)

    def __init__(self, spec: PsiSpec, key, *, quarantine: bool = True,
                 leak_probe: Optional[LeakProbe] = None):
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        self.spec = spec
        self.quarantine = bool(quarantine)
        self.leak_probe = leak_probe
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

    def _guard(self, leak_probe: Optional[LeakProbe]) -> None:
        """⛔ The quarantine. Raises; never warns (the ``PhiMismatchError``
        precedent — an invariant enforced in prose is not enforced)."""
        if not self.quarantine or self.spec.input_mode != "trajectory":
            return
        probe = leak_probe if leak_probe is not None else self.leak_probe
        if probe is not None and probe.passed():
            return
        raise AttentionPsiLeakError(
            "AttentionPsi refuses a store-relative TRAJECTORY reading: the leak "
            "probe is "
            + ("absent" if probe is None
               else f"failing ({probe.as_dict()})")
            + f". C2W2 D6 measured q0_only 0.3515-0.4480 and blank_store "
              f"0.3713-0.4728 against a bar of {ATTENTION_PSI_LEAK_BAR} at EVERY "
              "stride, so no attention-psi trajectory number is quotable "
              "store-relative. Supply a passing LeakProbe "
              "(__call__(..., leak_probe=...) or AttentionPsi(..., "
              "leak_probe=...)), or pass quarantine=False explicitly if the "
              "reading is deliberately not store-relative.")

    def __call__(self, traj: jnp.ndarray, state, *,
                 leak_probe: Optional[LeakProbe] = None) -> jnp.ndarray:
        self._guard(leak_probe)
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
def make_psi(family: str, spec: PsiSpec, key, *, quarantine: bool = True,
             leak_probe: Optional[LeakProbe] = None) -> eqx.Module:
    """``family in {"deepsets", "attention"}``.

    ``quarantine``/``leak_probe`` are forwarded to :class:`AttentionPsi` only —
    ``DeepSetsPsi`` is **not** quarantined (its own trajectory launder did not
    fire: C2W1 measured ``q0_only`` 0.129 against a chance of 0.125).
    """
    if family == "deepsets":
        return DeepSetsPsi(spec, key)
    if family == "attention":
        return AttentionPsi(spec, key, quarantine=quarantine,
                            leak_probe=leak_probe)
    raise ValueError(f"unknown psi family {family!r} (deepsets | attention)")


def psi_param_count(psi: eqx.Module) -> int:
    """Number of learnable scalars — the matched-parameter denominator."""
    leaves = jax.tree_util.tree_leaves(eqx.filter(psi, eqx.is_inexact_array))
    return int(sum(int(np.asarray(x).size) for x in leaves))


def matched_pair(family: str, spec: PsiSpec, key, *, quarantine: bool = True,
                 leak_probe: Optional[LeakProbe] = None
                 ) -> Tuple[eqx.Module, eqx.Module]:
    """``(point_psi, trajectory_psi)`` from the **same key** — identical
    initial parameters, identical parameter count, only ``input_mode`` differs.

    This is the ablation's fairness guarantee, in one function: a trajectory read
    that wins by being bigger is not a result.
    """
    p = make_psi(family, replace(spec, input_mode="settled_point"), key,
                 quarantine=quarantine, leak_probe=leak_probe)
    t = make_psi(family, replace(spec, input_mode="trajectory"), key,
                 quarantine=quarantine, leak_probe=leak_probe)
    assert psi_param_count(p) == psi_param_count(t)
    return p, t


# ==========================================================================
# ⭐ C2W2 — THE READ-IN THAT PARAMETRIZES THE PARTICLE (charter §A4.3)
# ==========================================================================
# Charter §A4.3, verbatim: *"Strong standard encoders (small CNN/ResNet,
# RNN/transformer, SSL-pretrained where weight class allows) — weak phi is a
# measured failure mode (CIFAR null; `full_pca`). Fairness invariants: identical
# phi for CLU/baselines/launder; phi params in the byte ledger, all arms. phi's
# output head widens to parametrize the particle — launch q0 + per-particle mass
# + friction (mass is live under trajectory reads, per A2.2)."*
#
# Three things live below and nothing else:
#   1. :class:`PhiSpec` / :func:`make_phi` — the phi *interface*, with the strong
#      families wired in (the CNN trunk is `experiments/phi_encoders.ConvTrunk`,
#      imported, never forked).
#   2. :class:`ParticlePhi` — the widened head: ``x -> (q0, log_mass, friction)``.
#      **Default OFF** (``particle_head=False``), and off it is bit-identical to
#      the shipped launch: payload channels zeroed, no mass override, no friction
#      override.
#   3. The **fairness invariant, enforced in code**: :func:`assert_identical_phi`
#      RAISES (never warns) and :class:`SharedPhi` is the object every arm of a
#      race must draw its phi from. Plus :func:`phi_ledger`, which emits
#      ``phi_id`` / ``phi_bytes`` for the race card, and the ``(d, n_atoms)``
#      JOINT DIAL (:func:`joint_dial`).
#
# ⚠ Monitors are guards, never losses — nothing here reads a monitor, and no
# quantity here may enter an objective other than through the loss on psi.

#: phi families. ``identity``/``pca`` are the WEAK read-ins whose failure is
#: measured (the CIFAR null; ``full_pca`` under the 5-d ``q0_only`` baseline);
#: ``mlp``/``cnn``/``gru`` are the standard strong encoders §A4.3 asks for.
PHI_FAMILIES = ("identity", "pca", "mlp", "cnn", "gru")


class PhiMismatchError(RuntimeError):
    """Raised when two arms of the same comparison do not share one phi.

    §A4.3's fairness invariant is *"identical phi for CLU / baselines / launder"*.
    A warning would be silently ignorable, and every C2W2 dividend depends on
    this, so the mismatch is an exception.
    """


@dataclass(frozen=True)
class PhiSpec:
    """Static configuration of a read-in.

    ⚠ Config lives **here**, not in ``chlu/config.py`` — same C2W1/C2W2
    file-ownership rule as :class:`PsiSpec`. Override from a project YAML via
    :meth:`from_mapping`.

    Attributes:
        in_dim: raw query width (for ``cnn``: ``prod(image_shape)``).
        dim: latent width of the store (``addr_dim + payload_dim + spectator``).
        addr_dim: ``d``, the address block — **a capacity lever, not a free
            choice** (see :func:`joint_dial`).
        payload_dim: ``m``; these channels are forced to zero in ``q0`` because
            the shipped read launches on the payload-zero manifold.
        family: one of :data:`PHI_FAMILIES`.
        particle_head: ⭐ §A4.3's widened head. ``False`` (**default**) = the
            shipped behaviour, launch point only.
        log_mass_center / log_mass_span: the per-particle mass is
            ``M = softplus(log_mass_center + log_mass_span * tanh(head))``
            (softplus = the repo's positivity convention, ``chlu_unit.mass_vector``).
            The default centre is ``softplus^-1(1) = 0.5413``, so at init
            ``M ≈ 1`` — i.e. the shipped identity inertia.
        friction_lo / friction_hi / friction_init: the per-particle friction is
            ``gamma = lo + (hi-lo) * sigmoid(head + b0)`` with ``b0`` set so
            ``gamma = friction_init`` at init. The band is DECLARED, not
            discovered: monitor #1 trips at ``rho_conv > 1e-6`` and the shipped
            read sits at 4.3e-7 (doctrine R2), i.e. 2.3x inside the edge, so a
            head allowed to shrink gamma without bound would drive the store
            straight into an overdamping trip.
        n_atoms: the OTHER half of the joint dial (see :func:`joint_dial`).
            ``None`` = derive it from ``addr_dim`` by the banked co-scaling law.
    """

    in_dim: int
    dim: int
    addr_dim: int
    payload_dim: int = 1
    family: str = "mlp"
    hidden: int = 32
    depth: int = 2
    residual: bool = True
    # -- the particle head (§A4.3) — DEFAULT OFF -------------------------
    particle_head: bool = False
    log_mass_center: float = 0.5413248546129181  # softplus^-1(1.0)
    log_mass_span: float = 1.0
    friction_lo: float = 0.02
    friction_hi: float = 0.20
    friction_init: float = 0.05  # the shipped ``gamma_address``
    # -- family-specific ---------------------------------------------------
    image_shape: Optional[Tuple[int, int, int]] = None  # (C, H, W), family="cnn"
    seq_shape: Optional[Tuple[int, int]] = None  # (T, F), family="gru"
    cnn_channels: Tuple[int, ...] = (16, 32)
    cnn_pool: int = 2
    cnn_groups: int = 8
    # -- the joint dial ----------------------------------------------------
    n_atoms: Optional[int] = None
    capacity: int = 8

    def __post_init__(self):
        if self.family not in PHI_FAMILIES:
            raise ValueError(f"family must be one of {PHI_FAMILIES}, got {self.family!r}")
        if not (0 < self.friction_lo <= self.friction_init <= self.friction_hi < 2.0):
            raise ValueError(
                "require 0 < friction_lo <= friction_init <= friction_hi < 2 "
                f"(got {self.friction_lo}, {self.friction_init}, {self.friction_hi})"
            )
        if self.addr_dim + self.payload_dim > self.dim:
            raise ValueError("addr_dim + payload_dim must be <= dim")
        if self.family == "cnn" and self.image_shape is None:
            raise ValueError("family='cnn' requires image_shape=(C, H, W)")
        if self.family == "gru" and self.seq_shape is None:
            raise ValueError("family='gru' requires seq_shape=(T, F)")

    @classmethod
    def from_mapping(cls, in_dim: int, dim: int, addr_dim: int,
                     overrides: Optional[dict] = None) -> "PhiSpec":
        known = {f for f in (
            "payload_dim", "family", "hidden", "depth", "residual", "particle_head",
            "log_mass_center", "log_mass_span", "friction_lo", "friction_hi",
            "friction_init", "image_shape", "seq_shape", "cnn_channels", "cnn_pool",
            "cnn_groups", "n_atoms", "capacity")}
        kw = {k: v for k, v in dict(overrides or {}).items() if k in known}
        for k in ("image_shape", "seq_shape", "cnn_channels"):
            if isinstance(kw.get(k), list):
                kw[k] = tuple(kw[k])
        return cls(in_dim=int(in_dim), dim=int(dim), addr_dim=int(addr_dim), **kw)

    @property
    def n_head_out(self) -> int:
        """Width of the output head: ``dim`` (+ ``dim`` log-masses + 1 friction)."""
        return int(self.dim) + (int(self.dim) + 1 if self.particle_head else 0)

    def as_flags(self) -> Dict[str, Any]:
        """The flag-provenance row for this read-in."""
        return {"phi_family": self.family, "phi_hidden": int(self.hidden),
                "phi_depth": int(self.depth), "phi_residual": bool(self.residual),
                "phi_particle_head": bool(self.particle_head),
                "phi_addr_dim": int(self.addr_dim),
                "phi_friction_band": (float(self.friction_lo), float(self.friction_hi)),
                "phi_friction_init": float(self.friction_init),
                "phi_log_mass_center": float(self.log_mass_center),
                "phi_log_mass_span": float(self.log_mass_span)}


# --------------------------------------------------------------------------
# the (d, atom-budget) JOINT DIAL — charter §A4.3, a DECLARATION not a sweep
# --------------------------------------------------------------------------
#: The banked co-scaling constants (w23, ``CluSystemConfig.min_atoms*``). Kept
#: here so this module can state the law without importing the frozen harness;
#: ``tests/test_phi_particle.py`` asserts the two agree for every ``d`` in 1..8.
MIN_ATOMS_BASE = 512
MIN_ATOMS_C = 1.4142135623730951  # sqrt(2)
MIN_ATOMS_FLOOR = 384


def joint_dial(addr_dim: int, capacity: int = 8, atoms_per_item: int = 32,
               n_atoms: Optional[int] = None) -> Dict[str, Any]:
    """⭐ ``(d, atom budget)`` as **one declared dial** (charter §A4.3).

    *"phi's address-block dimension ``d`` is a capacity lever, not a free choice
    — banked law ``K_learned(d) = min(2^d, write ceiling)``; but atom budget must
    co-scale (``min_atoms ∝ c^d``), bytes/well grow ∝ d (capacity-per-byte is the
    honest metric), and reach tightens as ``σ√d`` — so ``(d, atom budget)`` is a
    single declared joint dial in the byte ledger, launder included."*

    The co-scaling law (identical to ``CluSystemConfig.n_atoms``, w23):

        ``n_atoms(d) = ceil_to_K( max(atoms_per_item*K, 384, round(512 * sqrt(2)^d)) )``

    Returns the whole dial as one record, and **asserts** the co-scaling when an
    explicit ``n_atoms`` is supplied: an under-budgeted high-``d`` cell reads as
    a capacity result when it is an optimizer artefact, which is the exact error
    the law exists to prevent.

    ⛔ ``K_learned(8)`` is **lower-bounded**, never bracketed — ``K = 2048`` was
    never run. Do not sweep ``d`` here; that is C2W3's business.
    """
    d, K = int(addr_dim), int(capacity)
    geo = round(MIN_ATOMS_BASE * MIN_ATOMS_C ** d)
    need = max(int(atoms_per_item) * K, MIN_ATOMS_FLOOR, int(geo))
    derived = int(K * int(np.ceil(need / K)))
    used = derived if n_atoms is None else int(n_atoms)
    ok = used >= derived
    return {
        "d": d,
        "capacity_K": K,
        "n_atoms": used,
        "n_atoms_required": derived,
        "atoms_per_item": float(used) / max(K, 1),
        "co_scaling_ok": bool(ok),
        "co_scaling_law": "n_atoms >= max(atoms_per_item*K, 384, round(512*sqrt(2)^d))",
        "k_learned_designed": int(2 ** d),  # min(2^d, write ceiling); LOWER BOUND at d=8
        "reach_sigma_scale": float(np.sqrt(d)),  # reach tightens as sigma*sqrt(d)
    }


def assert_joint_dial(addr_dim: int, n_atoms: int, capacity: int = 8,
                      atoms_per_item: int = 32) -> Dict[str, Any]:
    """:func:`joint_dial`, but **raises** when the atom budget has not co-scaled."""
    rec = joint_dial(addr_dim, capacity, atoms_per_item, n_atoms)
    if not rec["co_scaling_ok"]:
        raise ValueError(
            f"(d, atom-budget) joint dial violated: d={rec['d']} requires "
            f"n_atoms >= {rec['n_atoms_required']}, got {rec['n_atoms']}. "
            "d and the atom budget move together (charter §A4.3) — an "
            "under-budgeted high-d cell reads as a capacity result when it is an "
            "optimizer artefact."
        )
    return rec


# --------------------------------------------------------------------------
# the widened head
# --------------------------------------------------------------------------
class ParticleLaunch(eqx.Module):
    """What a particle-parametrizing phi emits: ``(q0, log_mass, friction)``.

    ``q0`` is the launch point on the payload-zero manifold; ``log_mass`` is
    per-coordinate (the store's inertia is diagonal, ``chlu_unit.mass_vector``);
    ``friction`` is one scalar per query, inside the declared band.

    ⭐ **``log_mass`` and ``friction`` are ``None`` when the particle head is
    off.** That is what makes "ship default-off" structural rather than
    numerical: a reader sees "no per-particle mass was emitted" and falls back to
    the model's own global ``M`` and the config's ``gamma`` — it does **not**
    override them with a nominal value that happens to be close.
    """

    q0: jnp.ndarray  # (B, dim)
    log_mass: Optional[jnp.ndarray]  # (B, dim) or None (head off)
    friction: Optional[jnp.ndarray]  # (B,) or None (head off)

    @property
    def has_particle(self) -> bool:
        """``True`` iff the head actually emitted particle attributes."""
        return self.log_mass is not None or self.friction is not None

    def mass(self) -> Optional[jnp.ndarray]:
        """``M = softplus(log_mass)`` — the repo's positivity convention."""
        return None if self.log_mass is None else jax.nn.softplus(self.log_mass)


def _phi_trunk(spec: PhiSpec, key):
    """``(module_or_None, h_dim)`` for the family. The CNN trunk is imported."""
    if spec.family in ("identity", "pca"):
        return None, int(spec.in_dim)
    if spec.family == "mlp":
        return eqx.nn.MLP(int(spec.in_dim), int(spec.hidden), int(spec.hidden),
                          max(int(spec.depth) - 1, 1), activation=jax.nn.tanh,
                          key=key), int(spec.hidden)
    if spec.family == "cnn":
        # the w26 CL encoder trunk, imported and NOT forked (lazy import: core
        # must not depend on `experiments` at module load).
        from chlu.experiments.phi_encoders import ConvTrunk

        c = int(spec.image_shape[0])
        trunk = ConvTrunk(c, tuple(int(x) for x in spec.cnn_channels),
                          int(spec.cnn_pool), int(spec.cnn_groups), key)
        return trunk, int(trunk.h_dim)
    if spec.family == "gru":
        cell = eqx.nn.GRUCell(int(spec.seq_shape[1]), int(spec.hidden), key=key)
        return cell, int(spec.hidden)
    raise ValueError(f"unknown phi family {spec.family!r}")


class ParticlePhi(eqx.Module):
    """⭐ The read-in of charter §A4.3: ``x -> q0`` **and** ``x -> (q0, M, gamma)``.

    ``__call__(x) -> q0`` keeps the shipped signature (drop-in for
    :class:`LearnedPhi`); :meth:`launch` returns the full
    :class:`ParticleLaunch`. With ``spec.particle_head=False`` the mass/friction
    fields are the *declared defaults* (``M = softplus(center)``, ``gamma =
    friction_init``) and carry **no gradient**, so shipped behaviour is
    unchanged — that is what "ship default-off" means here.

    Positivity/band conventions (both are reparameterizations, never penalties —
    the same trick as ``tie_channel_mass``):

    * ``M = softplus(center + span * tanh(head_m))`` — positive by construction;
    * ``gamma = lo + (hi - lo) * sigmoid(head_g + b0)`` — inside the declared
      band by construction, and ``= friction_init`` at init.
    """

    trunk: Optional[eqx.Module]
    head: Optional[eqx.nn.Linear]
    proj: Optional[jnp.ndarray]  # frozen (in_dim, addr_dim) map, family="pca"
    proj_mean: Optional[jnp.ndarray]
    spec: PhiSpec = eqx.field(static=True)
    friction_bias: float = eqx.field(static=True)

    def __init__(self, spec: PhiSpec, key=None, *, proj=None, proj_mean=None):
        self.spec = spec
        key = jax.random.PRNGKey(0) if key is None else key
        k_t, k_h = jax.random.split(key, 2)
        trunk, h_dim = _phi_trunk(spec, k_t)
        self.trunk = trunk
        need_head = (spec.family not in ("identity", "pca")) or spec.particle_head
        if need_head:
            head = eqx.nn.Linear(h_dim, spec.n_head_out, key=k_h)
            # start small: a strong trunk should not throw the launch point
            # across the ball on step 0 (and with `residual` the launch begins
            # at the shipped embedding).
            head = eqx.tree_at(lambda m: m.weight, head, 0.1 * head.weight)
            head = eqx.tree_at(lambda m: m.bias, head, jnp.zeros_like(head.bias))
            self.head = head
        else:
            self.head = None
        self.proj = None if proj is None else jnp.asarray(proj)
        self.proj_mean = None if proj_mean is None else jnp.asarray(proj_mean)
        lo, hi, g0 = spec.friction_lo, spec.friction_hi, spec.friction_init
        u = (float(g0) - float(lo)) / (float(hi) - float(lo))
        u = min(max(u, 1e-6), 1.0 - 1e-6)
        self.friction_bias = float(np.log(u / (1.0 - u)))

    # -- pieces ------------------------------------------------------------
    def _features(self, x: jnp.ndarray) -> jnp.ndarray:
        s = self.spec
        if s.family in ("identity", "pca"):
            z = x if self.proj is None else (x - (0.0 if self.proj_mean is None
                                                  else self.proj_mean)) @ self.proj
            return z
        if s.family == "mlp":
            return jax.vmap(self.trunk)(x)
        if s.family == "cnn":
            img = jnp.reshape(x, (x.shape[0],) + tuple(int(v) for v in s.image_shape))
            return jax.vmap(self.trunk)(img)
        if s.family == "gru":
            t, f = (int(v) for v in s.seq_shape)
            seq = jnp.reshape(x, (x.shape[0], t, f))

            def run(u):
                def step(h, ut):
                    return self.trunk(ut, h), None
                h0 = jnp.zeros((self.trunk.hidden_size,), dtype=u.dtype)
                h, _ = jax.lax.scan(step, h0, u)
                return h

            return jax.vmap(run)(seq)
        raise ValueError(f"unknown phi family {s.family!r}")

    def _pad_to_latent(self, z: jnp.ndarray) -> jnp.ndarray:
        d = int(self.spec.dim)
        if z.shape[-1] >= d:
            return z[:, :d]
        pad = jnp.zeros((z.shape[0], d - z.shape[-1]), dtype=z.dtype)
        return jnp.concatenate([z, pad], axis=-1)

    def _zero_payload(self, q: jnp.ndarray) -> jnp.ndarray:
        a, m = int(self.spec.addr_dim), int(self.spec.payload_dim)
        z = jnp.zeros((q.shape[0], m), dtype=q.dtype)
        return jnp.concatenate([q[:, :a], z, q[:, a + m:]], axis=-1)

    # -- the two public forms ---------------------------------------------
    def launch(self, x: jnp.ndarray) -> ParticleLaunch:
        """``x -> (q0, log_mass, friction)``."""
        s = self.spec
        x = jnp.atleast_2d(jnp.asarray(x))
        h = self._features(x)
        if self.head is None:
            out = self._pad_to_latent(h)
            raw_m = None
            raw_g = None
        else:
            y = jax.vmap(self.head)(h)
            out = y[:, : s.dim]
            if s.particle_head:
                raw_m = y[:, s.dim: 2 * s.dim]
                raw_g = y[:, 2 * s.dim]
            else:
                raw_m = None
                raw_g = None
        if s.residual:
            # begin life at the shipped embedding (near-identity on the address
            # block), so a pilot starts from the shipped read.
            out = out + self._pad_to_latent(jnp.asarray(x, dtype=out.dtype))
        q0 = self._zero_payload(out)
        if raw_m is None:  # head off => NOTHING is overridden (default-off)
            return ParticleLaunch(q0=q0, log_mass=None, friction=None)
        log_mass = s.log_mass_center + s.log_mass_span * jnp.tanh(raw_m)
        u = jax.nn.sigmoid(raw_g + self.friction_bias)
        friction = s.friction_lo + (s.friction_hi - s.friction_lo) * u
        return ParticleLaunch(q0=q0, log_mass=log_mass, friction=friction)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """``x -> q0`` — the shipped signature (drop-in for :class:`LearnedPhi`)."""
        return self.launch(x).q0


def make_phi(spec: PhiSpec, key=None, *, proj=None, proj_mean=None) -> ParticlePhi:
    """Build the phi named by ``spec``. One constructor for every arm."""
    return ParticlePhi(spec, key, proj=proj, proj_mean=proj_mean)


# --------------------------------------------------------------------------
# ⭐ the fairness invariant, ENFORCED IN CODE
# --------------------------------------------------------------------------
def phi_fingerprint(phi) -> str:
    """A stable content hash of a phi: architecture **and** parameter values.

    Two arms with the same architecture but different weights are NOT the same
    read-in, so the fingerprint hashes the parameter bytes too.
    """
    import hashlib

    h = hashlib.sha256()
    h.update(type(phi).__name__.encode())
    spec = getattr(phi, "spec", None)
    if spec is not None:
        h.update(repr(spec).encode())
    for leaf in jax.tree_util.tree_leaves(eqx.filter(phi, eqx.is_inexact_array)):
        a = np.asarray(leaf)
        h.update(str(a.shape).encode())
        h.update(np.ascontiguousarray(a, dtype=np.float64).tobytes())
    return h.hexdigest()[:16]


def assert_identical_phi(phis: Dict[str, Any]) -> str:
    """⭐ §A4.3's fairness invariant. ``{arm_name: phi}`` -> the shared ``phi_id``.

    **Raises** :class:`PhiMismatchError` on any mismatch — it does not warn. If
    some arm structurally needs a different read-in, every number in the
    comparison inherits a confound, and that must stop the race, not annotate it.
    """
    if not phis:
        raise PhiMismatchError("no arms given: the identical-phi invariant is vacuous")
    ids = {name: phi_fingerprint(p) for name, p in phis.items()}
    uniq = sorted(set(ids.values()))
    if len(uniq) != 1:
        groups: Dict[str, list] = {}
        for name, fid in ids.items():
            groups.setdefault(fid, []).append(name)
        raise PhiMismatchError(
            "identical-phi invariant VIOLATED (charter §A4.3: identical phi for "
            "CLU / baselines / launder). Distinct read-ins: "
            + "; ".join(f"{fid}: {sorted(v)}" for fid, v in sorted(groups.items()))
        )
    return uniq[0]


class SharedPhi:
    """The one phi every arm of a comparison must draw from.

    Usage::

        shared = SharedPhi(make_phi(spec, key))
        phi_clu = shared.for_arm("clu")
        phi_launder = shared.for_arm("trajectory_launder")
        ...
        shared.assert_invariant()      # raises if any arm was handed another phi
        card = shared.ledger()         # phi_id / phi_bytes for the race card

    It hands out the *same instance*, records which arms took it, and can check
    an externally-built phi against itself (:meth:`check`). ``assert_invariant``
    is cheap and should be called immediately before any number is emitted.
    """

    def __init__(self, phi, spec: Optional[PhiSpec] = None):
        self.phi = phi
        self.spec = spec if spec is not None else getattr(phi, "spec", None)
        self.phi_id = phi_fingerprint(phi)
        self.arms: Dict[str, Any] = {}

    def for_arm(self, name: str):
        """Hand this arm the shared phi (and record that it took it)."""
        self.arms[str(name)] = self.phi
        return self.phi

    def check(self, name: str, phi) -> None:
        """Register an externally-built phi for ``name`` — **raises** if it differs."""
        self.arms[str(name)] = phi
        if phi_fingerprint(phi) != self.phi_id:
            raise PhiMismatchError(
                f"arm {name!r} was built with a DIFFERENT phi "
                f"({phi_fingerprint(phi)} != {self.phi_id}); charter §A4.3 "
                "requires identical phi for CLU / baselines / launder"
            )

    def assert_invariant(self) -> str:
        """Re-check every registered arm. Returns the shared ``phi_id``."""
        fid = assert_identical_phi(self.arms if self.arms else {"_": self.phi})
        if fid != self.phi_id:  # a caller mutated the shared phi in place
            raise PhiMismatchError(
                f"the shared phi changed under the arms ({fid} != {self.phi_id})")
        return fid

    def ledger(self, **extra) -> Dict[str, Any]:
        """The byte-ledger row, with the arms that drew this phi."""
        return phi_ledger(self.phi, self.spec, arms=sorted(self.arms), **extra)


# --------------------------------------------------------------------------
# the byte ledger (the C2W2 race card reads `phi_id` / `phi_bytes`)
# --------------------------------------------------------------------------
#: float32 parameters — the store's own precision (``CluSystem`` is float32).
BYTES_PER_PARAM = 4


def phi_ledger(phi, spec: Optional[PhiSpec] = None, *, arms=None,
               n_atoms: Optional[int] = None, capacity: Optional[int] = None,
               **extra) -> Dict[str, Any]:
    """⭐ ``phi_id`` / ``phi_bytes`` **on every arm** (§A4.3), plus the joint dial.

    ``phi_bytes = 4 * (learnable scalars)``. A frozen PCA/identity read-in still
    costs its bytes: ``full_pca``'s projection is data-derived, so it is paid
    for, and a launder that reads the same phi pays the same price. The dial
    prints ``d`` **and** ``n_atoms`` together because §A4.3 makes them one
    declared dial, and prints ``bytes_per_well ∝ d`` because capacity-per-byte
    is the honest metric.
    """
    spec = spec if spec is not None else getattr(phi, "spec", None)
    n_par = psi_param_count(phi)
    d = int(spec.addr_dim) if spec is not None else -1
    m = int(spec.payload_dim) if spec is not None else 1
    row: Dict[str, Any] = {
        "phi_id": phi_fingerprint(phi),
        "phi_family": (spec.family if spec is not None else type(phi).__name__),
        "phi_particle_head": bool(getattr(spec, "particle_head", False)),
        "phi_params": int(n_par),
        "phi_bytes": int(n_par * BYTES_PER_PARAM),
    }
    if spec is not None:
        dial = joint_dial(d, capacity=int(spec.capacity if capacity is None else capacity),
                          n_atoms=(spec.n_atoms if n_atoms is None else int(n_atoms)))
        row.update({
            "d": d,
            "n_atoms": dial["n_atoms"],
            "n_atoms_required": dial["n_atoms_required"],
            "joint_dial_ok": dial["co_scaling_ok"],
            # bytes/well grow ∝ d (the address) + m (the payload)
            "bytes_per_well": int((d + m) * BYTES_PER_PARAM),
            "capacity_per_byte_note": "capacity-per-byte is the honest metric (§A4.3)",
        })
    if arms is not None:
        row["phi_arms"] = list(arms)
    row.update(extra)
    return row


# ==========================================================================
# ⭐⭐ C2W11 — THE SET-LEVEL READ (§A34.1: *binding is the READ + psi's job*)
# ==========================================================================
# The `k` feature-factored particles land; a **DeepSets pooled psi** binds them
# into the downstream answer with a **likelihood weighted by captured-vs-
# scattered particles**.
#
# ⛔ **DeepSets ONLY.** `AttentionPsi` is QUARANTINED for trajectory input
# (C2W2 reconciliation 1, `AttentionPsiLeakError` above) — the pooled DeepSets
# psi is explicitly *not* quarantined, and any attention-psi number is a
# declared NOT-RUN this wave.
# ⛔ **No binding structure is built**: the latent space may be disjoint /
# independent per feature. Co-activation / wormhole edges are the C2W9 pointer.
# ⛔⛔ psi's capacity is NOT chosen — it is **set by the measured leak**
# (K4-at-full-psi legs 1 and 2, `k4_full_psi_obligation`). psi carries more
# parameters than the reader class's `N_a*m` SP-1 bound, which is exactly why
# K4-at-full-psi and the K8 structural cell carry this wave's false-positive
# load.
# ==========================================================================


class ParticleSetPsi(eqx.Module):
    """``rho( [ sum_f h(u_f) ; sum_f w_f h(u_f) ] ) -> R^m`` — pooled DeepSets.

    ``u_f`` is the per-particle descriptor of :func:`chlu.core.novelty_read.
    particle_descriptors`; ``w_f`` is the particle's **capture likelihood**, so a
    scattered particle is down-weighted in the binding rather than silently
    counted (§A34.1's *likelihood weighted by captured-vs-scattered particles*).

    ⭐ **Pooled SUM, not mean/max**: the family's target is ``y = sum_{j in A}
    v_j``, so a sum pool is the permutation-invariant statistic the task is
    literally written in. The second (weighted) pool is the only place capture
    enters the binding.
    """

    enc: eqx.nn.MLP
    dec: eqx.nn.MLP
    u_dim: int = eqx.field(static=True)
    hidden: int = eqx.field(static=True)
    out_dim: int = eqx.field(static=True)

    def __init__(self, u_dim: int, out_dim: int, key, *, hidden: int = 16,
                 depth: int = 2):
        k1, k2 = jax.random.split(key, 2)
        self.u_dim = int(u_dim)
        self.hidden = int(hidden)
        self.out_dim = int(out_dim)
        self.enc = eqx.nn.MLP(int(u_dim), int(hidden), int(hidden),
                              max(int(depth) - 1, 1), activation=jax.nn.tanh, key=k1)
        self.dec = eqx.nn.MLP(2 * int(hidden), int(out_dim), int(hidden),
                              max(int(depth) - 1, 1), activation=jax.nn.tanh, key=k2)

    def __call__(self, u: jnp.ndarray, w: Optional[jnp.ndarray] = None
                 ) -> jnp.ndarray:
        """``(B, k, u_dim) [, (B, k)] -> (B, out_dim)``."""
        h = jax.vmap(jax.vmap(self.enc))(u)          # (B, k, hidden)
        pooled = jnp.sum(h, axis=1)
        if w is None:
            w = jnp.ones(u.shape[:2], dtype=u.dtype)
        wpool = jnp.sum(h * w[..., None], axis=1)
        return jax.vmap(self.dec)(jnp.concatenate([pooled, wpool], axis=-1))


def set_psi_param_count(psi: "ParticleSetPsi") -> int:
    """Fitted-parameter count of the set-level psi — ledgered on every arm."""
    return int(sum(x.size for x in jax.tree_util.tree_leaves(
        eqx.filter(psi, eqx.is_inexact_array))))
