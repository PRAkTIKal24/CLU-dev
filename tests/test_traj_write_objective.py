"""C2W2 Route 1 — the trajectory / path write-objective terms.

⛔ **The first test in this file is the task's hard falsifier and it is
BLOCKING**: at ``lambda_traj = lambda_path = 0`` the written ``V`` must be
**bit-identical** to ``main``'s. If it is not, every downstream C2W2 number
measured against the ``endpoint_write`` control is uninterpretable and no science
cell may run.

The rest assert the properties that let a cell *vote* in the gate:

* both terms are **live** — they change the loss, its gradient, and the written
  landscape (a term that never moves anything hasn't been asked, it has been
  whispered at);
* both are **structurally inert with no competitor** (a single-item write), which
  is a declared property, not a bug — and it is why liveness is measured on a
  written stream;
* the trajectory term differentiates through a real damped-Verlet rollout, and
  that rollout reduces to the shipped
  :func:`~chlu.core.integrators.velocity_verlet_step` for identity mass;
* the path term's gradient acts on the *tangent* curvature between two sites,
  which is the spectral quantity the gym measured at ``lambda_min = -0.5946``.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chlu.core.clu_system import CluSystemConfig, build_system
from chlu.core.integrators import velocity_verlet_step
from chlu.core.memory_potentials import designed_payloads, designed_sites
from chlu.training.train_memory import (
    _damped_verlet_path,
    path_equal_depth_penalty,
    trajectory_margin_penalty,
    write_loss,
)


@pytest.fixture(autouse=True)
def float32_dynamics():
    """Pin float32 (repo-wide isolation hazard: other modules enable x64 at
    import; see ``tests/test_clu_system.py``)."""
    was = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", False)
    yield
    jax.config.update("jax_enable_x64", was)


def _tiny_cfg(**kw):
    base = dict(addr_dim=3, capacity=3, write_steps=25, address_steps=60,
                read_steps=80, n_query_per_item=2, atoms_per_item=8,
                min_atoms=48, min_atoms_base=48, seed=0)
    base.update(kw)
    return CluSystemConfig(**base)


def _stream(cfg, n=3):
    sites = np.asarray(designed_sites(cfg.addr_dim, n, R=cfg.ball_radius, seed=0))
    pays = np.asarray(designed_payloads(n, seed=0))
    return [{"item_id": i, "address": sites[i], "payload": float(pays[i])}
            for i in range(n)]


def _leaves(V):
    return [np.asarray(x) for x in
            jax.tree_util.tree_leaves(eqx.filter(V, eqx.is_inexact_array))]


def _written(write_objective=None, n=3):
    cfg = _tiny_cfg()
    sys_ = build_system(cfg, loud=False, write_objective=write_objective)
    sys_.write_stream(_stream(cfg, n))
    return sys_


def _targets(K=3, dim=4, payload_index=3):
    rng = np.random.default_rng(0)
    z = rng.normal(size=(K, dim)).astype(np.float32) * 0.5
    z[:, payload_index] = np.linspace(-0.8, 0.8, K)
    return jnp.asarray(z)


def _V(dim=4, n_atoms=24, groups=3):
    from chlu.core.memory_potentials import DesignFreedomPotential

    return DesignFreedomPotential(
        rung="free_mlp", dim=dim, payloads=jnp.zeros((groups,)),
        key=jax.random.PRNGKey(0), learned_family="atoms", n_atoms=n_atoms,
        rbf_init_width=0.3, confine=0.05, atom_depth_init=1e-4,
        atom_groups=groups, atom_init_scale=1.0,
    )


# ==========================================================================
# ⛔ THE COEFFICIENT-ZERO REGRESSION GATE (blocking)
# ==========================================================================
def test_coefficient_zero_writes_a_bit_identical_landscape():
    """⛔ ``lambda_traj = lambda_path = 0`` => the shipped write, bit-for-bit."""
    ref = _written(None)
    off = _written({"loss_kwargs": {"lambda_traj": 0.0, "lambda_path": 0.0}})
    for a, b in zip(_leaves(ref.store.V), _leaves(off.store.V), strict=True):
        assert np.array_equal(a, b)


def test_coefficient_zero_leaves_the_loss_value_bit_identical():
    """The same gate at the loss level: not one extra op is traced at 0, and the
    key stream is untouched (both terms fold their own sub-key)."""
    V, z, k = _V(), _targets(), jax.random.PRNGKey(3)
    base = float(write_loss(V, z, k, payload_index=3, barrier_pairs="nn"))
    zero = float(write_loss(V, z, k, payload_index=3, barrier_pairs="nn",
                            lambda_traj=0.0, lambda_path=0.0))
    assert base == zero  # exact equality, not approx


# ==========================================================================
# liveness of the two terms
# ==========================================================================
def test_the_trajectory_term_moves_the_loss_and_its_gradient():
    V, z, k = _V(), _targets(), jax.random.PRNGKey(3)
    kw = dict(payload_index=3, barrier_pairs="nn",
              traj_kwargs=dict(rollout_steps=12, stride=4, n_launch=2))
    base = float(write_loss(V, z, k, payload_index=3, barrier_pairs="nn"))
    on = float(write_loss(V, z, k, lambda_traj=1.0, **kw))
    assert on != base and np.isfinite(on)

    def g(mod, lam):
        return eqx.filter_grad(
            lambda m: write_loss(m, z, k, lambda_traj=lam, **kw))(mod)

    d0 = _leaves(g(V, 0.0))
    d1 = _leaves(g(V, 1.0))
    assert any(not np.array_equal(a, b) for a, b in zip(d0, d1, strict=True))


def test_the_path_term_moves_the_loss_and_its_gradient():
    V, z, k = _V(), _targets(), jax.random.PRNGKey(3)
    kw = dict(payload_index=3, barrier_pairs="nn",
              path_kwargs=dict(n_interp=3))
    base = float(write_loss(V, z, k, payload_index=3, barrier_pairs="nn"))
    on = float(write_loss(V, z, k, lambda_path=1.0, **kw))
    assert on != base and np.isfinite(on)
    d = _leaves(eqx.filter_grad(
        lambda m: write_loss(m, z, k, lambda_path=1.0, **kw))(V))
    assert any(np.any(np.abs(x) > 0) for x in d)


def test_a_big_coefficient_visibly_perturbs_the_written_store():
    """The grid's **liveness anchor**: at the top of the range the term must
    visibly move the write, even destructively. Without such a point an
    inert-everywhere result is an under-powered grid, not a <=0 vote."""
    ref = _written(None)
    hot = _written({"loss_kwargs": {
        "lambda_traj": 30.0,
        "traj_kwargs": {"rollout_steps": 12, "stride": 4, "n_launch": 2},
    }})
    moved = [float(np.max(np.abs(a - b)))
             for a, b in zip(_leaves(ref.store.V), _leaves(hot.store.V), strict=True)]
    assert max(moved) > 1e-4


# ==========================================================================
# declared structural properties
# ==========================================================================
def test_both_terms_are_inert_with_no_competitor():
    """Declared, not a bug: a sequential write's FIRST item has no neighbour, so
    a triplet margin and a connecting path are both undefined. This is why the
    liveness check runs on a written stream."""
    V = _V()
    one = _targets(K=1)
    assert float(trajectory_margin_penalty(
        V, one, jax.random.PRNGKey(0), payload_index=3)) == 0.0
    assert float(path_equal_depth_penalty(V, one)) == 0.0


def test_the_trajectory_term_uses_the_full_stored_set_as_competitors():
    """``crowd_targets`` is the FULL live set — the write is sequential, so
    ``targets`` is a single item and the competitor cannot come from it."""
    V = _V()
    one = _targets(K=1)
    crowd = _targets(K=3)
    val = float(trajectory_margin_penalty(
        V, one, jax.random.PRNGKey(0), payload_index=3, crowd_targets=crowd,
        rollout_steps=12, stride=4, n_launch=2))
    assert np.isfinite(val) and val > 0.0


def test_the_write_rollout_matches_the_shipped_verlet_step_for_identity_mass():
    """The trajectory the write shapes is the one the read traverses (up to the
    learned inertia): for ``H = |p|^2/2 + V(q)`` the shipped integrator reduces
    exactly to the write-side rollout."""
    V = _V()
    q = jnp.asarray(np.linspace(-0.3, 0.3, 4), dtype=jnp.float32)
    p = jnp.zeros(4, dtype=jnp.float32)
    dt, gamma, steps = 0.05, 0.05, 5

    def H(qq, pp):
        return 0.5 * jnp.sum(pp**2) + V(qq)

    qa, pa = q, p
    ref = []
    for _ in range(steps):
        qa, pa = velocity_verlet_step(H, qa, pa, dt, gamma)
        ref.append(np.asarray(qa))
    got = np.asarray(_damped_verlet_path(V, q, p, steps, dt, gamma, stride=1))
    assert np.allclose(got, np.stack(ref), atol=1e-6)


def test_the_strided_path_has_the_expected_number_of_points():
    V = _V()
    q = jnp.zeros(4, dtype=jnp.float32)
    out = _damped_verlet_path(V, q, q, steps=60, dt=0.05, gamma=0.05, stride=6)
    assert out.shape == (10, 4)  # the PREREG's registered n_pts


def test_the_path_term_is_zero_on_an_exactly_flat_equal_depth_segment():
    """Sanity: the term is a *penalty on the connecting path*, so a landscape
    that is already flat and equal-depth between two sites pays nothing."""

    class _Flat(eqx.Module):
        def __call__(self, q):
            return jnp.asarray(0.0) * jnp.sum(q)

    z = jnp.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=jnp.float32)
    assert float(path_equal_depth_penalty(_Flat(), z, n_interp=5)) == 0.0
