"""Tests for the 'CLU minus the physics' controls (G2 / P6)."""

import jax

jax.config.update("jax_enable_x64", True)  # det J / bit-level checks (test_goldstone convention)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from chlu.core.chlu_unit import CHLU  # noqa: E402
from chlu.core.twins import (  # noqa: E402
    BrokenVolumeCHLU,
    UnconstrainedTwin,
    build_arms,
    matched_twin_hidden,
)
from chlu.utils.metrics import count_params  # noqa: E402
from chlu.experiments.goldstone_harness import (  # noqa: E402
    rollout_from,
    settle,
    step_jacobian,
)

DIM = 4
HIDDEN = 32


def _chlu():
    return CHLU(
        dim=DIM, hidden=HIDDEN, kinetic_mode="newtonian_learned",
        potential_type="mlp", key=jax.random.PRNGKey(0),
    )


def test_duck_type_surface():
    """All three arms expose the CHLU surface the harness/trainer use."""
    arms = build_arms(
        jax.random.PRNGKey(1), DIM, HIDDEN, "newtonian_learned", "mlp"
    )
    q = jnp.ones(DIM)
    p = jnp.ones(DIM) * 0.3
    for m in (arms[k] for k in ("chlu", "broken_volume", "twin")):
        assert m.dim == DIM
        assert isinstance(m.kinetic_mode, str)
        assert isinstance(m.potential_type, str)
        # scalar energy + kinetic
        assert jnp.asarray(m.H(q, p)).shape == ()
        assert jnp.asarray(m.T(p)).shape == ()
        # mass accessors
        assert m.effective_inertia().shape == (DIM,)
        assert m.mass_vector().shape == (DIM,)
        # dynamics
        qn, pn = m.step((q, p), 0.05, 0.0)
        assert qn.shape == (DIM,) and pn.shape == (DIM,)
        traj = m(q, p, 5, 0.05, 0.0)
        assert traj.shape == (5, 2 * DIM)
        # potential_net callable
        assert jnp.asarray(m.potential_net(q)).shape == ()


def test_param_match_report():
    """Twin param count is close to CHLU; broken-volume adds exactly 2*dim."""
    arms = build_arms(
        jax.random.PRNGKey(2), DIM, HIDDEN, "newtonian_learned", "mlp"
    )
    p = arms["params"]
    assert p["broken_volume"] == p["chlu"] + 2 * DIM
    # matched within the achievable granularity (twin count is quadratic in
    # hidden, so the gap shrinks with width): loose here, tight at exp scale.
    rel = abs(p["twin"] - p["chlu"]) / p["chlu"]
    assert rel < 0.03, (p, rel)
    # at the experiment scale (hidden=64) the match is <0.5%
    big = build_arms(jax.random.PRNGKey(2), DIM, 64, "newtonian_learned", "mlp")
    bp = big["params"]
    assert abs(bp["twin"] - bp["chlu"]) / bp["chlu"] < 0.005, bp


def test_matched_twin_hidden_is_closest():
    """matched_twin_hidden returns the exact nearest-count width."""
    target = count_params(_chlu())
    h, cnt = matched_twin_hidden(target, DIM)
    # neighbours are not strictly better
    from chlu.core.twins import _twin_param_count

    assert abs(cnt - target) <= abs(_twin_param_count(DIM, h - 1) - target)
    assert abs(cnt - target) <= abs(_twin_param_count(DIM, h + 1) - target)


def test_broken_volume_init_bit_identical_to_chlu():
    """At init (log_scale=0) BrokenVolumeCHLU.step == CHLU.step, bit-level."""
    clu = _chlu()
    bv = BrokenVolumeCHLU(clu)
    q = jnp.array([0.7, -0.3, 0.1, 0.2])
    p = jnp.array([0.2, 0.4, -0.1, 0.0])
    for gamma in (0.0, 0.1):
        qc, pc = clu.step((q, p), 0.05, gamma)
        qb, pb = bv.step((q, p), 0.05, gamma)
        assert jnp.array_equal(qc, qb)
        assert jnp.array_equal(pc, pb)


def test_broken_volume_breaks_det_jacobian():
    """A non-zero log_scale makes det J != 1 (CHLU keeps det J == 1)."""
    import equinox as eqx

    clu = _chlu()
    q = jnp.array([1.0, 0.0, 0.0, 0.0])
    p = jnp.zeros(DIM)
    Jc = np.asarray(step_jacobian(clu, q, p, 0.05, 0.0))
    assert abs(np.linalg.det(Jc) - 1.0) < 1e-8  # symplectic, gamma=0

    bv = BrokenVolumeCHLU(clu)
    ls = jnp.array([0.1, -0.05, 0.2, 0.0, -0.1, 0.15, 0.0, 0.05])
    bv = eqx.tree_at(lambda m: m.log_scale, bv, replace=ls)
    Jb = np.asarray(step_jacobian(bv, q, p, 0.05, 0.0))
    det = np.linalg.det(Jb)
    assert abs(det - 1.0) > 1e-3  # volume broken
    # matches the analytic log|det J| = sum(log_scale) (Verlet det = 1)
    assert np.isclose(np.log(abs(det)), float(jnp.sum(ls)), atol=1e-6)
    assert np.isclose(bv.volume_log_jac(), float(jnp.sum(ls)))


def test_twin_free_update_and_zero_energy():
    """Twin advances the state and has identically-zero energy (sleep-inert)."""
    twin = UnconstrainedTwin(DIM, hidden=16, key=jax.random.PRNGKey(3))
    q = jnp.ones(DIM)
    p = jnp.zeros(DIM)
    qn, pn = twin.step((q, p), 0.05, 0.0)
    assert not jnp.allclose(jnp.concatenate([qn, pn]), jnp.concatenate([q, p]))
    assert float(twin.H(q, p)) == 0.0
    assert float(twin.T(p)) == 0.0
    # gamma damps momentum
    _, pn_g = twin.step((q, jnp.ones(DIM)), 0.05, 0.5)
    _, pn_0 = twin.step((q, jnp.ones(DIM)), 0.05, 0.0)
    assert jnp.all(jnp.abs(pn_g) <= jnp.abs(pn_0) + 1e-9)


def test_harness_runs_on_all_arms():
    """settle + rollout_from work verbatim on every arm."""
    arms = build_arms(
        jax.random.PRNGKey(4), DIM, HIDDEN, "newtonian_learned", "mlp"
    )
    q0 = jnp.array([1.0, 0.0, 0.0, 0.0])
    for name in ("chlu", "broken_volume", "twin"):
        q_star, p_star = settle(arms[name], q0, dt=0.05, gamma=0.1, steps=50)
        assert q_star.shape == (DIM,)
        traj = rollout_from(arms[name], q_star, p_star, steps=20, dt=0.05, gamma=0.05)
        assert traj.shape == (21, 2 * DIM)
        assert np.all(np.isfinite(np.asarray(traj)))


def test_train_chlu_runs_on_all_arms():
    """train_chlu runs verbatim on all three arms (tiny smoke)."""
    from chlu.config import get_default_config
    from chlu.core.twins import build_arms as ba
    from chlu.data.circle_vacuum import generate_circle_vacuum
    from chlu.training.train import train_chlu

    cfg = get_default_config()
    cfg.training.sleep_frequency = 10**9  # wake-only
    data = generate_circle_vacuum(
        jax.random.PRNGKey(5), n_points=16, seq_len=9, dim=DIM, radius=1.0
    )
    arms = ba(jax.random.PRNGKey(6), DIM, HIDDEN, "newtonian_learned", "mlp")
    for name in ("chlu", "broken_volume", "twin"):
        model, losses, _ = train_chlu(
            arms[name], data, key=jax.random.PRNGKey(7),
            config=cfg, epochs=3, window_size=8, dt=0.05,
        )
        assert np.isfinite(float(losses[-1]))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
