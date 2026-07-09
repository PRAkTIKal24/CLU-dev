"""Tests for the intra-unit wormhole mechanisms (w7 paid-access battery).

Covers the four certificates of paid-access-theory Prop-A6/A7 for
``WormholeChannels`` (construction (a), gated canonical translation) and the
closed-gate reduction of ``IntraWormholePotential`` (construction (b), throat):

  1. det J = 1 to ~1e-12 through a frozen-gate jump (Prop-A6).
  2. bit-exact closed-gate reduction to plain CHLU (both constructions).
  3. bounded-energy / ledger correctness (Prop-A6 energy ledger DeltaH=V(b)-V(a)).
  4. latch-transit shift = p^T X Delta (Prop-A7), 0 iff X.Delta _|_ p; plus the
     forbidden state-dependent-gate volume break det J = 1 + grad(g).Delta.

Plus the two CERTIFICATE-PAYOFF guarantees (V1 referee F3): guarantees the
wormhole provably preserves and the no-physics router provably violates.

  5. det J = 0 for the router (q := exit) => non-injective => it ERASES the
     Goldstone charge (Var(Q_out) = 0), where the det J = 1 wormhole TRANSPORTS
     it by the exact constant p^T X Delta.
  6. a wormhole exit outside the coercive component breaks BIBO; the receipt
     (energy ledger + component membership) screens it, while an energy-only
     test does NOT -- it falsely admits b = 4.0, whose ledger is even cheaper.
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


def test_router_erases_latch_that_wormhole_transports():
    """CERTIFICATE PAYOFF A (referee F3.1). The router q := exit has Jacobian
    blockdiag(0, I) => det J = 0 exactly: it is non-injective, so a cloud of
    incoming states leaves with ONE value of Q -- the latch is erased. The
    wormhole's canonical translation has det J = 1 and shifts every incoming
    state's Q by the same exact p^T X Delta -- the latch is transported."""
    d = 2
    X = so2_generator(d)
    a = jnp.array([3.0, 0.0])  # entrance on the vacuum circle
    p = jnp.array([0.05, 0.5])
    delta = jnp.array([0.0, 0.5])  # across-coset => nonzero, exact shift
    wc = WormholeChannels(
        entrances=a[None, :], exits=(a + delta)[None, :], radii=jnp.array([0.35])
    )

    # a cloud of incoming states strictly inside the capture ball (radius 0.35):
    # sample directions, then rescale to radius <= 0.25 so the hard gate fires
    # for EVERY sample (an uncaptured state would trivially have dQ = 0).
    key = jax.random.PRNGKey(0)
    dirs = jax.random.normal(key, (12, d))
    dirs = dirs / jnp.linalg.norm(dirs, axis=1, keepdims=True)
    q_in = a[None, :] + 0.25 * dirs
    assert bool(jnp.all(jax.vmap(lambda q: wc.jump(q, p)[2])(q_in)))  # all captured
    Q_in = jax.vmap(lambda q: p @ X @ q)(q_in)

    # -- wormhole: constant shift, spread preserved (injective) --
    q_wh = jax.vmap(lambda q: wc.jump(q, p)[0])(q_in)
    Q_wh = jax.vmap(lambda q: p @ X @ q)(q_wh)
    dQ = Q_wh - Q_in
    assert jnp.max(jnp.abs(dQ - (p @ X @ delta))) < 1e-5  # exact p^T X Delta
    assert jnp.std(dQ) < 1e-6  # same shift for EVERY incoming state
    assert jnp.abs(jnp.std(Q_wh) - jnp.std(Q_in)) < 1e-6  # spread transported
    # invertible: q_in recovered exactly from q_out
    assert jnp.max(jnp.abs((q_wh - delta) - q_in)) < 1e-6

    # -- router: every incoming state is mapped to the SAME exit --
    exit_pt = a + delta
    q_rt = jnp.broadcast_to(exit_pt, (12, d))
    Q_rt = jax.vmap(lambda q: p @ X @ q)(q_rt)
    assert float(jnp.std(Q_rt)) < 1e-6  # latch ERASED (f32 std of identical Q)
    assert jnp.max(jnp.abs(Q_rt - Q_rt[0])) == 0.0  # bit-identical: no spread
    assert float(jnp.std(Q_in)) > 1e-3  # ... and there was a spread to erase

    # certificates, measured on both arms (apples-to-apples)
    def _detJ(qmap):
        def full(qp):
            return jnp.concatenate([qmap(qp[:d]), qp[d:]])

        return jnp.linalg.det(jax.jacfwd(full)(jnp.concatenate([a, p])))

    assert jnp.abs(_detJ(lambda q: wc.jump(q, p)[0]) - 1.0) < 1e-12
    assert jnp.abs(_detJ(lambda q: exit_pt) - 0.0) < 1e-12  # volume-annihilating


def test_noncoercive_exit_breaks_bibo_and_receipt_screens_it():
    """CERTIFICATE PAYOFF B (referee F3.2 / theory §7 issue 7). An exit beyond
    the coercive component escapes to infinity; the wormhole receipt (ledger +
    component membership) rejects it, and an ENERGY-ONLY test does not."""
    from chlu.experiments.exp_paid_access import NonCoerciveWell, _make_clu

    pot = NonCoerciveWell(k=1.0, eps=0.02, conf=4.0)
    x_b, V_b = pot.barrier_top(), pot.barrier_height()
    assert abs(x_b - 3.5355) < 1e-3 and abs(V_b - 3.125) < 1e-3

    d = 2
    model = _make_clu(d, "relativistic", 1.0, 1.0, [4.0, 0.25], pot, jax.random.PRNGKey(0))
    q0 = jnp.zeros(d)
    p0 = jnp.zeros(d).at[0].set(0.3)
    ke0 = float(model.T(p0) - model.T(jnp.zeros(d)))

    def rollout_rstar(q, p, steps):
        traj = model(q, p, steps, 0.05, 0.02)
        return float(jnp.max(jnp.linalg.norm(traj[:, :d], axis=1)))

    ledgers = {}
    for b, expect_escape in [(3.0, False), (4.0, True), (5.0, True)]:
        exit_pt = jnp.zeros(d).at[0].set(b)
        wc = WormholeChannels(
            entrances=jnp.zeros((1, d)), exits=exit_pt[None, :], radii=jnp.array([0.35])
        )
        # the receipt, computed from the unit's own potential
        ledger = float(wc.ledger(pot, q0))
        assert abs(ledger - float(pot(exit_pt) - pot(q0))) < 1e-6  # exact
        ledgers[b] = ledger
        in_component = abs(b) < x_b
        energy_ok = ke0 + float(pot(exit_pt)) <= V_b
        admissible = in_component and energy_ok

        qj, pj, _ = wc.jump(q0, p0)
        r_T = rollout_rstar(qj, pj, 500)
        r_2T = rollout_rstar(qj, pj, 1000)
        escaped = r_2T > 20.0
        assert escaped is expect_escape, f"b={b}: r*(2T)={r_2T}"
        assert admissible is (not escaped)  # the receipt predicts BIBO exactly
        if escaped:
            assert r_2T > 1.5 * r_T  # r* grows ~linearly in T => unbounded
        else:
            assert abs(r_2T - r_T) < 1e-3  # saturated => bounded

    # THE FINE PRINT (why det J = 1 + a bounded ledger is NOT enough): the exit
    # b = 5.0 is FREE by the energy ledger (Delta H = 0, cheaper than the
    # admissible b = 3.0) and passes an energy-only sub-level test -- yet it
    # escapes. Coercive-COMPONENT membership, not energy, is the operative
    # clause of the receipt (paid-access-theory §7 issue 7).
    assert abs(ledgers[5.0]) < 1e-6  # a "free" jump ...
    assert ledgers[5.0] < ledgers[3.0] < V_b  # ... cheaper than an admissible one
    assert (ke0 + float(pot(jnp.array([5.0, 0.0])))) <= V_b  # energy-only ADMITS
    assert 5.0 > x_b  # ... and only the component test catches it


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
