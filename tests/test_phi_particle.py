"""Tests for the C2W2 read-in that parametrizes the particle (charter §A4.3).

Four families, in the order they matter:

* **the fairness invariant** — ``assert_identical_phi`` must RAISE on a mismatch
  (a warning would be silently ignorable, and every C2W2 dividend depends on it);
* **the particle head** — mass positive by construction, friction inside the
  declared band by construction, and **default-off means structurally off**
  (``log_mass = friction = None``), not "off to within round-off";
* **the ``(d, atom-budget)`` joint dial** — the co-scaling law stated here must
  agree with the harness's own ``CluSystemConfig.n_atoms`` at every ``d``,
  because a law that drifts from the code it describes is worse than no law;
* **the mass gauge** — the D3 claim in miniature, on the controlled toy: an
  **endpoint** loss has a numerically-zero mass/friction gradient (Prop Q1.1:
  ``Fix(T) = {(q,0): grad V = 0}`` contains neither ``M`` nor ``gamma``) while a
  **trajectory** loss does not.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chlu.core.psi_readout import (
    MIN_ATOMS_BASE,
    MIN_ATOMS_C,
    ParticleLaunch,
    PhiMismatchError,
    PhiSpec,
    SharedPhi,
    assert_identical_phi,
    assert_joint_dial,
    joint_dial,
    make_phi,
    phi_fingerprint,
    phi_ledger,
    psi_param_count,
)


@pytest.fixture(autouse=True)
def float32_dynamics():
    """Pin float32 (repo-wide isolation hazard: other modules enable x64)."""
    was = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", False)
    yield
    jax.config.update("jax_enable_x64", was)


IN, DIM, ADDR, PAY = 4, 5, 4, 1


def _spec(**kw):
    return PhiSpec(in_dim=IN, dim=DIM, addr_dim=ADDR, payload_dim=PAY, **kw)


# --------------------------------------------------------------------------
# the fairness invariant
# --------------------------------------------------------------------------
def test_identical_phi_invariant_raises_on_a_mismatch():
    """§A4.3: identical phi for CLU / baselines / launder. **Raises**, not warns."""
    a = make_phi(_spec(), jax.random.PRNGKey(0))
    b = make_phi(_spec(), jax.random.PRNGKey(1))
    assert assert_identical_phi({"clu": a, "launder": a, "baseline": a})
    with pytest.raises(PhiMismatchError):
        assert_identical_phi({"clu": a, "baseline": b})


def test_identical_phi_invariant_sees_through_equal_architectures():
    """Same architecture, different weights, is NOT the same read-in."""
    a = make_phi(_spec(), jax.random.PRNGKey(0))
    b = make_phi(_spec(), jax.random.PRNGKey(1))
    assert phi_fingerprint(a) != phi_fingerprint(b)
    # ...and a fingerprint is stable under re-computation
    assert phi_fingerprint(a) == phi_fingerprint(a)


def test_shared_phi_hands_out_one_instance_and_checks_rebuilds():
    a = make_phi(_spec(), jax.random.PRNGKey(0))
    shared = SharedPhi(a)
    assert shared.for_arm("clu") is a
    assert shared.for_arm("launder") is a
    assert shared.assert_invariant() == shared.phi_id
    with pytest.raises(PhiMismatchError):
        shared.check("baseline", make_phi(_spec(), jax.random.PRNGKey(2)))


def test_phi_ledger_emits_the_race_card_fields():
    phi = make_phi(_spec(family="mlp"), jax.random.PRNGKey(0))
    row = phi_ledger(phi, _spec(family="mlp"))
    for k in ("phi_id", "phi_bytes", "phi_params", "phi_family", "d", "n_atoms"):
        assert k in row
    assert row["phi_bytes"] == 4 * row["phi_params"]  # float32
    assert row["phi_params"] == psi_param_count(phi)


def test_identity_phi_costs_zero_bytes_and_a_head_costs_more():
    ident = make_phi(_spec(family="identity"), jax.random.PRNGKey(0))
    assert phi_ledger(ident, _spec(family="identity"))["phi_bytes"] == 0
    off = _spec(family="mlp", particle_head=False)
    on = _spec(family="mlp", particle_head=True)
    b_off = phi_ledger(make_phi(off, jax.random.PRNGKey(0)), off)["phi_bytes"]
    b_on = phi_ledger(make_phi(on, jax.random.PRNGKey(0)), on)["phi_bytes"]
    assert b_on > b_off  # the widened head is PAID FOR, in the ledger


# --------------------------------------------------------------------------
# the particle head
# --------------------------------------------------------------------------
def test_default_off_emits_no_particle_attributes():
    """Ship default-off: the head-off read-in overrides NOTHING (structural)."""
    phi = make_phi(_spec(family="mlp", particle_head=False), jax.random.PRNGKey(0))
    lau = phi.launch(jnp.zeros((3, IN)))
    assert lau.log_mass is None and lau.friction is None
    assert lau.has_particle is False
    assert lau.mass() is None
    assert lau.q0.shape == (3, DIM)


def test_particle_head_mass_is_positive_and_friction_is_inside_the_band():
    spec = _spec(family="mlp", particle_head=True, friction_lo=0.02, friction_hi=0.2)
    phi = make_phi(spec, jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (16, IN)) * 10.0  # extreme inputs
    lau = phi.launch(x)
    assert lau.has_particle
    assert float(jnp.min(lau.mass())) > 0.0
    assert float(jnp.min(lau.friction)) >= spec.friction_lo
    assert float(jnp.max(lau.friction)) <= spec.friction_hi


def test_particle_head_starts_at_the_shipped_friction():
    """``friction_init`` is the shipped ``gamma_address``: the head begins where
    the shipped read is, so turning it on is a perturbation, not a jump."""
    spec = _spec(family="mlp", particle_head=True, friction_init=0.05)
    phi = make_phi(spec, jax.random.PRNGKey(0))
    lau = phi.launch(jnp.zeros((4, IN)))
    # the bias is zeroed and the head weight scaled by 0.1, so the head starts
    # within a fraction of a percent of the shipped values (not exactly at them:
    # the trunk's own biases make h != 0 even at x = 0).
    assert float(jnp.max(jnp.abs(lau.friction - 0.05))) < 1e-3   # 0.2 % of gamma
    assert float(jnp.max(jnp.abs(lau.mass() - 1.0))) < 3e-2      # 3 % of M


def test_launch_is_on_the_payload_zero_manifold():
    for head in (False, True):
        phi = make_phi(_spec(family="mlp", particle_head=head), jax.random.PRNGKey(0))
        q0 = phi(jax.random.normal(jax.random.PRNGKey(2), (5, IN)))
        assert np.allclose(np.asarray(q0)[:, ADDR:ADDR + PAY], 0.0)


def test_particle_attributes_carry_gradient():
    """The whole point of the head: ``dL/d(log_mass, friction)`` must exist."""
    spec = _spec(family="mlp", particle_head=True)
    phi = make_phi(spec, jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(3), (4, IN))

    def loss(p):
        lau = p.launch(x)
        return jnp.sum(lau.log_mass ** 2) + jnp.sum(lau.friction ** 2)

    g = eqx.filter_grad(loss)(phi)
    assert float(jnp.linalg.norm(g.head.weight)) > 0.0


@pytest.mark.parametrize("family,kw", [
    ("identity", {}),
    ("pca", {}),
    ("mlp", {}),
    ("cnn", dict(image_shape=(1, 8, 8), in_dim=64, cnn_channels=(4, 8), cnn_pool=1,
                 cnn_groups=2)),
    ("gru", dict(seq_shape=(4, 2), in_dim=8)),
])
def test_every_phi_family_builds_and_launches(family, kw):
    in_dim = kw.pop("in_dim", IN)
    spec = PhiSpec(in_dim=in_dim, dim=DIM, addr_dim=ADDR, payload_dim=PAY,
                   family=family, particle_head=True, **kw)
    phi = make_phi(spec, jax.random.PRNGKey(0))
    lau = phi.launch(jax.random.normal(jax.random.PRNGKey(1), (3, in_dim)))
    assert lau.q0.shape == (3, DIM)
    assert lau.log_mass.shape == (3, DIM)
    assert lau.friction.shape == (3,)
    assert bool(jnp.all(jnp.isfinite(lau.q0)))


def test_phi_spec_rejects_an_inverted_friction_band():
    with pytest.raises(ValueError):
        _spec(friction_lo=0.5, friction_hi=0.1)
    with pytest.raises(ValueError):
        _spec(friction_lo=0.1, friction_init=0.01, friction_hi=0.2)


def test_phi_spec_rejects_an_unknown_family_and_missing_shapes():
    with pytest.raises(ValueError):
        _spec(family="transformer")
    with pytest.raises(ValueError):
        _spec(family="cnn")  # no image_shape


# --------------------------------------------------------------------------
# the (d, atom-budget) joint dial
# --------------------------------------------------------------------------
@pytest.mark.parametrize("d", [1, 2, 3, 4, 5, 6, 7, 8])
def test_joint_dial_matches_the_harness_atom_law_at_every_d(d):
    """The law written in ``psi_readout`` must BE the law the harness runs."""
    from chlu.core.clu_system import CluSystemConfig

    cfg = CluSystemConfig(addr_dim=d)
    assert joint_dial(d, capacity=cfg.capacity,
                      atoms_per_item=cfg.atoms_per_item)["n_atoms_required"] == cfg.n_atoms


def test_joint_dial_constants_are_the_banked_ones():
    from chlu.core.clu_system import CluSystemConfig

    cfg = CluSystemConfig()
    assert MIN_ATOMS_BASE == cfg.min_atoms_base
    assert MIN_ATOMS_C == pytest.approx(cfg.min_atoms_c)


def test_joint_dial_raises_when_the_atom_budget_does_not_co_scale():
    """d and the atom budget are ONE dial: raising d without atoms must fail."""
    assert_joint_dial(4, n_atoms=2048)  # in-band
    with pytest.raises(ValueError):
        assert_joint_dial(8, n_atoms=2048)  # d=8 needs ~8192


def test_joint_dial_reports_reach_and_capacity_together():
    rec = joint_dial(4)
    assert rec["reach_sigma_scale"] == pytest.approx(2.0)  # sqrt(d)
    assert rec["k_learned_designed"] == 16  # 2^d, the DESIGNED wall


# --------------------------------------------------------------------------
# ⭐ the mass gauge (D3 in miniature, on the controlled toy)
# --------------------------------------------------------------------------
def test_mass_and_friction_gauge_dissolves_under_a_trajectory_loss():
    """Charter §A2.2, at test scale and in float64.

    Endpoint loss: ``dL/dlog_mass`` and ``dL/dgamma`` are numerically zero (the
    settled point is M- and gamma-independent — Prop Q1.1). Trajectory loss:
    both are O(1e-3). If this ever inverts, §A2.2 is refuted.
    """
    from chlu.core.implicit_grad import GaussianWellsPotential, truncated_rollout
    from chlu.experiments.goldstone_harness import clu_with_potential

    was = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        centers = jnp.asarray([[np.cos(t), np.sin(t)]
                               for t in np.linspace(0, 2 * np.pi, 5)[:4]],
                              dtype=jnp.float64)
        V = GaussianWellsPotential(centers,
                                   jnp.asarray([1.0, 0.9, 1.1, 0.95], dtype=jnp.float64),
                                   s=0.35, alpha=0.05)
        model = clu_with_potential(V, dim=2, kinetic_mode="newtonian_learned",
                                   inertia=jnp.ones(2, dtype=jnp.float64))
        q0 = centers[0] + jnp.asarray([0.2, -0.15], dtype=jnp.float64)
        p0 = jnp.zeros(2, dtype=jnp.float64)

        def make(loss_kind):
            def f(log_m, g):
                # N = 1500: the same settle budget the spike used, i.e. deep
                # enough for the endpoint's mass gradient to reach machine zero
                # (rho^N with rho = sqrt(1-gamma) = 0.97468).
                tr = truncated_rollout(model, q0, p0, 1500, 0.05, g, retain=None,
                                       stride=10, mass_override=jax.nn.softplus(log_m))
                if loss_kind == "endpoint":
                    return 0.5 * jnp.sum(tr[-1, :2] ** 2)
                return jnp.mean(jnp.sum(tr[:, :2] ** 2, axis=-1))
            return f

        lm = jnp.asarray([0.5413248546129181] * 2, dtype=jnp.float64)
        gam = jnp.asarray(0.05, dtype=jnp.float64)
        g_end = jax.grad(make("endpoint"), argnums=(0, 1))(lm, gam)
        g_traj = jax.grad(make("trajectory"), argnums=(0, 1))(lm, gam)
        end_m = float(jnp.linalg.norm(g_end[0]))
        end_g = float(jnp.abs(g_end[1]))
        traj_m = float(jnp.linalg.norm(g_traj[0]))
        traj_g = float(jnp.abs(g_traj[1]))
        assert end_m < 1e-10, f"the settled point is NOT M-independent: {end_m}"
        assert end_g < 1e-10, f"the settled point is NOT gamma-independent: {end_g}"
        assert traj_m > 1e-6, f"A2.2 refuted at test scale: {traj_m}"
        assert traj_g > 1e-4, f"A2.2 refuted at test scale (friction): {traj_g}"
        assert traj_m / max(end_m, 1e-300) > 1e3
    finally:
        jax.config.update("jax_enable_x64", was)


def test_implicit_settle_sends_exactly_zero_gradient_to_the_model_mass():
    """The point arm's zero is EXACT by the theorem, not small by luck."""
    from chlu.core.implicit_grad import SettleSpec, implicit_settle
    from chlu.core.psi_readout import ParticleLaunch as _PL  # noqa: F401
    from chlu.experiments.goldstone_harness import clu_with_potential
    from chlu.core.implicit_grad import GaussianWellsPotential

    centers = jnp.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    V = GaussianWellsPotential(centers, jnp.asarray([1.0, 0.9, 1.1, 0.95]),
                               s=0.35, alpha=0.05)
    model = clu_with_potential(V, dim=2, kinetic_mode="newtonian_learned",
                               inertia=jnp.ones(2))
    q0 = centers[0] + jnp.asarray([0.2, -0.15])
    p0 = jnp.zeros(2)
    spec = SettleSpec(steps=400, dt=0.05, gamma=0.05)

    def loss(m):
        return 0.5 * jnp.sum(implicit_settle(m, q0, p0, spec) ** 2)

    g = eqx.filter_grad(loss)(model)
    assert np.all(np.asarray(g.log_mass) == 0.0)


def test_truncated_rollout_accepts_a_traced_gamma_and_mass_without_changing_the_default():
    """The two new arguments must not perturb the historical path."""
    from chlu.core.implicit_grad import truncated_rollout
    from chlu.experiments.goldstone_harness import clu_with_potential
    from chlu.core.implicit_grad import GaussianWellsPotential

    centers = jnp.asarray([[1.0, 0.0], [0.0, 1.0]])
    V = GaussianWellsPotential(centers, jnp.asarray([1.0, 0.9]), s=0.35, alpha=0.05)
    model = clu_with_potential(V, dim=2, kinetic_mode="newtonian_learned",
                               inertia=jnp.ones(2))
    q0, p0 = jnp.asarray([0.5, 0.1]), jnp.zeros(2)
    base = truncated_rollout(model, q0, p0, 50, 0.05, 0.05, stride=5)
    # the Python-float path is untouched: `_maybe_float` casts it exactly as before
    again = truncated_rollout(model, q0, p0, 50, 0.05, float(0.05), stride=5)
    assert np.array_equal(np.asarray(base), np.asarray(again))
    # a TRACED gamma is a different computation (a float32 array vs a weak-typed
    # Python constant the compiler may fold), so it agrees to float32 round-off,
    # not bit-for-bit — stated rather than asserted away.
    traced = jax.jit(lambda g: truncated_rollout(model, q0, p0, 50, 0.05, g, stride=5)
                     )(jnp.asarray(0.05))
    assert float(np.max(np.abs(np.asarray(base) - np.asarray(traced)))) < 1e-6
    # ...and mass_override=M_model reproduces it too (M = 1 here)
    with_mass = truncated_rollout(model, q0, p0, 50, 0.05, 0.05, stride=5,
                                  mass_override=model.mass_vector())
    assert float(np.max(np.abs(np.asarray(base) - np.asarray(with_mass)))) < 1e-6


def test_particle_launch_is_a_pytree_with_optional_fields():
    lau = ParticleLaunch(q0=jnp.zeros((2, DIM)), log_mass=None, friction=None)
    leaves = jax.tree_util.tree_leaves(lau)
    assert len(leaves) == 1  # q0 only
    assert lau.has_particle is False
