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


# ==========================================================================
# D4 — the recency-family harness fix (default OFF)
# ==========================================================================
def test_restrict_to_pair_chooses_between_the_querys_own_two_candidates():
    """⛔ The measured defect: ``queries_recency`` labels a **2-way** question and
    ``score_index`` grades it against a 2-way chance of 0.5, but the CLU arms
    arg-max over all ``K`` live sites. Measured (seed 0, K=5, 72 queries): the
    CLU answered outside its own pair **19.4 %** of the time."""
    from chlu.experiments.memory_gym import GymConfig, restrict_to_pair

    scores = np.array([[0.9, 0.1, 0.2],       # global argmax = 0, pair = (1,2)
                       [0.1, 0.3, 0.8]])      # global argmax = 2, pair = (0,1)
    pairs = np.array([[1, 2], [0, 1]])
    assert np.array_equal(restrict_to_pair(scores, pairs), np.array([2, 1]))
    assert np.array_equal(np.argmax(scores, axis=1), np.array([0, 2]))  # the defect
    # ⛔ default OFF => shipped behaviour
    assert GymConfig().restrict_index_to_pair is False


def test_restrict_to_pair_handles_a_lower_is_better_score():
    from chlu.experiments.memory_gym import restrict_to_pair

    d = np.array([[0.1, 5.0, 0.2]])
    pairs = np.array([[1, 2]])
    assert restrict_to_pair(d, pairs, higher_is_better=False)[0] == 2


# ==========================================================================
# the race-card runner (C2W2 D3)
# ==========================================================================
def test_objective_spec_builds_the_declared_arms_and_the_control():
    from chlu.experiments.exp_traj_write import objective_spec

    assert objective_spec("endpoint_write", 0.0) is None
    t = objective_spec("traj_write", 0.3)["loss_kwargs"]
    assert t["lambda_traj"] == 0.3 and "lambda_path" not in t
    p = objective_spec("path_write", 3.0)["loss_kwargs"]
    assert p["lambda_path"] == 3.0 and "lambda_traj" not in p
    b = objective_spec("traj+path", 0.3)["loss_kwargs"]
    assert b["lambda_traj"] == 0.3 and b["lambda_path"] == 0.3
    with pytest.raises(ValueError, match="unknown race arm"):
        objective_spec("nope", 1.0)


def test_the_pre_registered_grid_spans_two_decades_and_carries_a_zero_point():
    """§1(ii): >= 3 non-zero coefficients spanning >= 2 decades, plus the
    perturbing anchor and the mandatory zero point (the D1 regression gate)."""
    from chlu.experiments.exp_traj_write import COEFF_GRID, plan

    nz = [c for c in COEFF_GRID if c > 0]
    assert len(nz) >= 3
    assert max(nz) / min(nz) >= 100.0
    arms = plan()
    assert ("endpoint_write", 0.0) in arms  # the zero point
    assert {a for a, _ in arms} == {"endpoint_write", "traj_write", "path_write",
                                    "traj+path"}


def test_a_route1_arm_is_not_marked_unconverged_just_for_carrying_its_own_term():
    """⛔ The gate ruling (i) must judge convergence on the ENDPOINT part: a
    Route-1 arm's recorded loss includes ``lambda * L_term``, which is not
    required to reach zero. Judging on the total would exclude exactly the arms
    under test — the failure mode that would gut coverage until B' can never
    fire."""
    from chlu.experiments.exp_traj_write import _write_record

    rec = {"write_losses": [1.9, 1.88, 1.885],          # total, includes 3*0.55
           "endpoint_write_losses": [0.004, 0.002, 0.003],
           "endpoint_write_loss": 0.003,
           "certificates": {"lambda_min": 3.3},
           "clu_config_non_default": {"write_steps": 300}}
    w = _write_record(rec)
    assert w.converged and w.admissible()
    assert w.final_loss == pytest.approx(1.885)  # the total is still reported
    # ...and a genuinely unconverged write is still excluded
    rec2 = dict(rec, endpoint_write_losses=[0.22], endpoint_write_loss=0.22)
    assert not _write_record(rec2).admissible()


def test_to_cell_emits_the_frozen_schema_with_the_architectural_byte_note():
    from chlu.eval.race import RaceCell
    from chlu.experiments.exp_traj_write import to_cell

    rec = {
        "family": "overload", "seed": 1, "arm": "load1x_shipped",
        "primary_metric": "decode",
        "dividend": {"full": 0.75, "launder": 1.0, "dividend": -0.25,
                     "controls": {"same_keys_null": 1.0, "blank_store": 0.16}},
        "trivial_substitute_audit": {"best_zero_byte": 0.9167},
        "trajectory_launder": {"full": 0.75, "q0_only": 0.129, "endpoints": 0.74,
                               "blank_store": 0.148, "chance": 0.125, "bar": 0.24},
        "byte_ledger": {"full_bytes": 47820, "launder_bytes": 100, "ratio": 478.2,
                        "breakdown": {}},
        "write_losses": [0.001], "endpoint_write_losses": [0.001],
        "certificates": {"lambda_min": 3.29},
        "clu_config_non_default": {"write_steps": 300},
        "gym_config_non_default": {}, "monitors": [],
    }
    c = to_cell(rec, "traj_write", 3.0)
    assert isinstance(c, RaceCell) and c.route == "route1"
    assert c.arm == "traj_write@3" and c.family == "overload" and c.seed == 1
    assert c.dividend == pytest.approx(-0.25)
    assert c.substitute_margin == pytest.approx(0.75 - 0.9167)
    assert c.bytes.architectural and "NOT quotable" in c.notes
    assert not c.trajectory_launder.fired()
    assert c.gate_admissible is True
