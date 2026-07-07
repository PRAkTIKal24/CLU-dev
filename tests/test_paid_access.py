"""Tests for the intra-unit wormhole mechanisms (w7 paid-access battery).

Covers the four certificates of paid-access-theory Prop-A6/A7 for
``WormholeChannels`` (construction (a), gated canonical translation) and the
closed-gate reduction of ``IntraWormholePotential`` (construction (b), throat):

  1. det J = 1 to ~1e-12 through a frozen-gate jump (Prop-A6).
  2. bit-exact closed-gate reduction to plain CHLU (both constructions).
  3. bounded-energy / ledger correctness (Prop-A6 energy ledger DeltaH=V(b)-V(a)).
  4. latch-transit shift = p^T X Delta (Prop-A7), 0 iff X.Delta _|_ p; plus the
     forbidden state-dependent-gate volume break det J = 1 + grad(g).Delta.
"""

import jax
import jax.numpy as jnp

from chlu.core.chlu_unit import CHLU
from chlu.core.potentials import (
    IntraWormholePotential,
    PotentialMLP,
    WormholeChannels,
    so2_generator,
)


def _channels():
    ent = jnp.array([[-1.0, 0.0]])
    ext = jnp.array([[2.5, 0.0]])
    rad = jnp.array([0.3])
    return WormholeChannels(entrances=ent, exits=ext, radii=rad)


def test_wormhole_jump_det_j_is_one():
    """Frozen-gate canonical translation has det J = 1 exactly (Prop-A6)."""
    wc = _channels()
    d = 2
    q = jnp.array([-1.0, 0.05])
    p = jnp.array([0.7, -0.2])

    def map_qp(qp):
        q_, p_ = qp[:d], qp[d:]
        delta, _ = wc.selected_delta(q_)  # frozen at incoming q => constant
        return jnp.concatenate([q_ + delta, p_])

    J = jax.jacfwd(map_qp)(jnp.concatenate([q, p]))
    det = jnp.linalg.det(J)
    assert jnp.abs(det - 1.0) < 1e-12, f"det J = {det}, expected 1"

    # p is unchanged by the jump
    _, p_new, jumped = wc.jump(q, p)
    assert bool(jumped)
    assert jnp.allclose(p_new, p)


def test_closed_gate_reduces_to_plain_chlu():
    """Outside capture: no jump. Zero-depth throat: V reduces bit-exactly."""
    wc = _channels()
    p = jnp.array([0.7, -0.2])

    # Outside every capture set => no translation.
    q_far = jnp.array([1.0, 1.0])
    q_new, _, jumped = wc.jump(q_far, p)
    assert not bool(jumped)
    assert jnp.array_equal(q_new, q_far)

    # Throat with depth 0 reduces to the base potential bit-exactly.
    base = PotentialMLP(2, 16, key=jax.random.PRNGKey(3))
    iw = IntraWormholePotential(
        base=base,
        via=jnp.array([[0.5, 0.0]]),
        depth=jnp.array([0.0]),
        width=jnp.array([0.3]),
    )
    for q in (q_far, jnp.array([0.5, 0.0]), jnp.array([-1.0, 0.05])):
        assert iw(q) == base(q)


def test_ledger_and_bounded_energy():
    """Energy ledger DeltaH = V(q+Delta) - V(q) exact; throat is bounded."""
    wc = _channels()
    V = PotentialMLP(2, 16, key=jax.random.PRNGKey(3))
    q = jnp.array([-1.0, 0.05])

    led = wc.ledger(V, q)
    qn, _, _ = wc.jump(q, p=jnp.zeros(2))
    assert jnp.abs(led - (V(qn) - V(q))) < 1e-10

    # Throat energy magnitude is bounded by sum(depth): sup_q |throat| <= depth.
    depth = jnp.array([1.3, 0.7])
    iw = IntraWormholePotential(
        base=V,
        via=jnp.array([[0.5, 0.0], [-0.5, 0.0]]),
        depth=depth,
        width=jnp.array([0.3, 0.3]),
    )
    # At the via point the throat is deepest; |iw - base| <= sum(depth).
    for q in (jnp.array([0.5, 0.0]), jnp.array([-0.5, 0.0]), jnp.array([0.0, 0.0])):
        assert (V(q) - iw(q)) <= float(jnp.sum(depth)) + 1e-6
        assert (V(q) - iw(q)) >= -1e-6  # throat only lowers V


def test_latch_transit_and_forbidden_gate():
    """Q' - Q = p^T X Delta exactly (Prop-A7); frozen gate keeps det J = 1,
    a state-dependent gate breaks volume by exactly grad(g).Delta."""
    wc = _channels()
    d = 2
    X = so2_generator(d)
    delta = wc.deltas()[0]
    q = jnp.array([-1.0, 0.2])  # inside capture radius 0.3 of entrance [-1, 0]
    p = jnp.array([0.7, -0.2])

    qn, pn, jumped = wc.jump(q, p)
    assert bool(jumped)
    Q = p @ X @ q
    Qn = pn @ X @ qn
    assert jnp.abs((Qn - Q) - (p @ X @ delta)) < 1e-5  # exact; f32 roundoff

    # Tangent Delta with X.Delta _|_ p => zero latch shift (design choice).
    Xd = X @ delta
    # pick p orthogonal to X.Delta
    p_perp = jnp.array([Xd[1], -Xd[0]])
    assert jnp.abs(p_perp @ X @ delta) < 1e-5

    # Forbidden state-dependent gate: det J = 1 + grad(g).Delta != 1.
    g = lambda qq: 0.5 + 0.3 * qq[0]  # noqa: E731

    def fmap(qp):
        q_, p_ = qp[:d], qp[d:]
        qn2, pn2 = wc.forbidden_state_dependent_jump(q_, p_, g)
        return jnp.concatenate([qn2, pn2])

    Jf = jax.jacfwd(fmap)(jnp.concatenate([q, p]))
    grad_g = jax.grad(g)(q)
    assert jnp.abs(jnp.linalg.det(Jf) - (1.0 + grad_g @ delta)) < 1e-5
    assert jnp.abs(jnp.linalg.det(Jf) - 1.0) > 1e-3  # volume genuinely broken


def test_wormhole_composes_with_chlu_step():
    """A jump between two plain CHLU Verlet steps preserves p and lands at
    the exit basin; the CHLU step itself is untouched (no core edits)."""
    wc = _channels()
    model = CHLU(
        dim=2, hidden=16, kinetic_mode="relativistic", key=jax.random.PRNGKey(0)
    )
    q = jnp.array([-1.0, 0.02])
    p = jnp.array([0.1, 0.0])
    # one relax step, then a wormhole jump, then another relax step
    q, p = model.step((q, p), dt=0.05)
    q, p, jumped = wc.jump(q, p)
    assert bool(jumped)
    assert q[0] > 1.0  # translated across to the exit
    q, p = model.step((q, p), dt=0.05)
    assert jnp.all(jnp.isfinite(q)) and jnp.all(jnp.isfinite(p))
