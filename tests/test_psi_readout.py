"""Tests for ``chlu/core/psi_readout.py`` and the traceable two-phase read.

The load-bearing property is **fairness of the ablation**: the settled-point arm
and the trajectory arm must be the same module, the same parameters and the same
parameter count, differing only in which points enter the pooled set.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chlu.core.clu_system import ReadState
from chlu.core.psi_readout import (
    AttentionPsi,
    DeepSetsPsi,
    LearnedPhi,
    PsiSpec,
    make_psi,
    matched_pair,
    psi_param_count,
    select_points,
)


@pytest.fixture(autouse=True)
def float32_dynamics():
    """Pin float32 for the whole module, restoring the global flag after.

    ⚠ Repo-wide test-isolation hazard (handover §7.2): several test modules
    enable ``jax_enable_x64`` at MODULE import, so x64 is globally ON in a
    full-suite run. The harness stores and reads in float32 and
    ``test_differentiable_read_reproduces_the_frozen_harness_read`` compares
    against it at a float32 tolerance.
    """
    was = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", False)
    yield
    jax.config.update("jax_enable_x64", was)


DIM, ADDR, PAY, B, N = 5, 4, 1, 3, 7


def _state(key):
    k = jax.random.split(key, 6)
    return ReadState(q0=jax.random.normal(k[0], (B, DIM)),
                     p0=jnp.zeros((B, DIM)),
                     q_addr=jax.random.normal(k[2], (B, DIM)),
                     p_addr=jax.random.normal(k[3], (B, DIM)),
                     q_star=jax.random.normal(k[4], (B, DIM)),
                     p_star=jax.random.normal(k[5], (B, DIM)))


def _traj(key):
    return jax.random.normal(key, (B, N, 2 * DIM))


# --------------------------------------------------------------------------
# point selection — the ONLY thing the ablation changes
# --------------------------------------------------------------------------
@pytest.mark.parametrize("mode,n_pts", [("trajectory", N), ("settled_point", 1),
                                        ("endpoints", 2)])
def test_select_points_shapes(mode, n_pts):
    spec = PsiSpec(dim=DIM, addr_dim=ADDR, payload_dim=PAY, input_mode=mode)
    pts = select_points(_traj(jax.random.PRNGKey(0)),
                        _state(jax.random.PRNGKey(1)), spec)
    assert pts.shape == (B, n_pts, spec.point_features)
    assert spec.point_features == 2 * DIM + 1  # q, p, time


def test_settled_point_mode_ignores_the_trajectory_entirely():
    """The degenerate case must be *exactly* the classical read, so a change to
    the buffer cannot move it. (This is also why ``traj_stride`` is a dead axis
    for the shipped ``settled_point_psi`` — monitor #10's tier-b reading.)"""
    spec = PsiSpec(dim=DIM, addr_dim=ADDR, payload_dim=PAY,
                   input_mode="settled_point")
    st = _state(jax.random.PRNGKey(1))
    psi = DeepSetsPsi(spec, jax.random.PRNGKey(2))
    a = psi(_traj(jax.random.PRNGKey(0)), st)
    b = psi(_traj(jax.random.PRNGKey(99)) * 100.0, st)
    assert np.allclose(np.asarray(a), np.asarray(b))


def test_store_relative_representation_subtracts_the_launch_point():
    """Doctrine I-2: the trajectory CONTAINS ``q0 = phi(x)``."""
    spec = PsiSpec(dim=DIM, addr_dim=ADDR, input_mode="trajectory",
                   representation="store_relative", include_time=False)
    st = _state(jax.random.PRNGKey(1))
    ref = jnp.concatenate([st.q0, st.p0], axis=-1)
    traj = jnp.broadcast_to(ref[:, None, :], (B, N, 2 * DIM))
    pts = select_points(traj, st, spec)
    assert np.allclose(np.asarray(pts), 0.0, atol=1e-6)


def test_psi_spec_rejects_unknown_modes():
    with pytest.raises(ValueError):
        PsiSpec(dim=DIM, addr_dim=ADDR, input_mode="magic")
    with pytest.raises(ValueError):
        PsiSpec(dim=DIM, addr_dim=ADDR, representation="magic")
    with pytest.raises(ValueError):
        PsiSpec(dim=DIM, addr_dim=ADDR, stride=0)


# --------------------------------------------------------------------------
# the ablation's fairness guarantee
# --------------------------------------------------------------------------
@pytest.mark.parametrize("family", ["deepsets", "attention"])
def test_matched_pair_has_identical_parameters_not_merely_identical_counts(family):
    """⭐ Matched parameters and matched bytes are non-negotiable: a trajectory
    read that wins by being bigger is not a result."""
    spec = PsiSpec(dim=DIM, addr_dim=ADDR, payload_dim=PAY)
    point, traj = matched_pair(family, spec, jax.random.PRNGKey(3))
    assert psi_param_count(point) == psi_param_count(traj)
    a = jax.tree_util.tree_leaves(eqx.filter(point, eqx.is_inexact_array))
    b = jax.tree_util.tree_leaves(eqx.filter(traj, eqx.is_inexact_array))
    for x, y in zip(a, b, strict=True):
        assert np.allclose(np.asarray(x), np.asarray(y))


@pytest.mark.parametrize("family", ["deepsets", "attention"])
def test_psi_is_permutation_invariant_over_the_point_set(family):
    """Both pooling families are set functions — required, or the read depends on
    the storage order of the buffer rather than on its content."""
    spec = PsiSpec(dim=DIM, addr_dim=ADDR, payload_dim=PAY,
                   input_mode="trajectory", include_time=False)
    psi = make_psi(family, spec, jax.random.PRNGKey(4))
    st = _state(jax.random.PRNGKey(1))
    traj = _traj(jax.random.PRNGKey(0))
    perm = np.random.default_rng(0).permutation(N)
    a = np.asarray(psi(traj, st))
    b = np.asarray(psi(traj[:, perm, :], st))
    assert np.allclose(a, b, atol=1e-5)


@pytest.mark.parametrize("family", ["deepsets", "attention"])
def test_psi_output_shape_and_gradients(family):
    spec = PsiSpec(dim=DIM, addr_dim=ADDR, payload_dim=PAY)
    psi = make_psi(family, spec, jax.random.PRNGKey(5))
    st = _state(jax.random.PRNGKey(1))
    traj = _traj(jax.random.PRNGKey(0))
    assert psi(traj, st).shape == (B, PAY)
    g = eqx.filter_grad(lambda m: jnp.sum(m(traj, st) ** 2))(psi)
    assert psi_param_count(g) == psi_param_count(psi)
    flat = np.concatenate([np.asarray(x).ravel() for x in
                           jax.tree_util.tree_leaves(eqx.filter(g, eqx.is_inexact_array))])
    assert np.isfinite(flat).all() and np.abs(flat).max() > 0


def test_psi_stride_actually_subsamples():
    spec = PsiSpec(dim=DIM, addr_dim=ADDR, input_mode="trajectory", stride=3)
    pts = select_points(_traj(jax.random.PRNGKey(0)),
                        _state(jax.random.PRNGKey(1)), spec)
    assert pts.shape[1] == len(range(0, N, 3))


def test_attention_weights_are_a_simplex_over_points():
    spec = PsiSpec(dim=DIM, addr_dim=ADDR, input_mode="trajectory")
    psi = AttentionPsi(spec, jax.random.PRNGKey(6))
    st = _state(jax.random.PRNGKey(1))
    pts = select_points(_traj(jax.random.PRNGKey(0)), st, spec)
    h = jax.vmap(jax.vmap(psi.enc))(pts)
    k = jnp.einsum("hdc,bnc->bhnd", psi.W_k, h)
    logits = jnp.einsum("hd,bhnd->bhn", psi.q_tok, k) / np.sqrt(k.shape[-1])
    a = np.asarray(jax.nn.softmax(logits, axis=-1))
    assert np.allclose(a.sum(-1), 1.0, atol=1e-5)


# --------------------------------------------------------------------------
# phi
# --------------------------------------------------------------------------
def test_learned_phi_launches_on_the_payload_zero_manifold():
    """``CluSystem.read`` zeroes the payload channels of ``q0``; a ``phi`` that
    could write them would be reading the answer off its own input."""
    phi = LearnedPhi(ADDR, DIM, ADDR, PAY, key=jax.random.PRNGKey(7))
    q0 = phi(jax.random.normal(jax.random.PRNGKey(8), (B, ADDR)))
    assert q0.shape == (B, DIM)
    assert np.allclose(np.asarray(q0[:, ADDR:ADDR + PAY]), 0.0)


def test_learned_phi_is_near_identity_at_init_on_the_address_block():
    phi = LearnedPhi(ADDR, DIM, ADDR, PAY, residual=True, key=jax.random.PRNGKey(7))
    x = jax.random.normal(jax.random.PRNGKey(8), (B, ADDR))
    q0 = np.asarray(phi(x))[:, :ADDR]
    assert np.linalg.norm(q0 - np.asarray(x)) < 0.5 * np.linalg.norm(np.asarray(x))


# --------------------------------------------------------------------------
# the traceable read must reproduce the FROZEN harness read
# --------------------------------------------------------------------------
def test_differentiable_read_reproduces_the_frozen_harness_read():
    """``CluSystem.read`` is frozen API and is not traceable (numpy diagnostics),
    so ``differentiable_read`` re-implements its dynamics. It must agree."""
    from chlu.core.clu_system import CluSystemConfig, build_system
    from chlu.experiments.exp_trajectory_read import differentiable_read

    cfg = CluSystemConfig(seed=0, capacity=4, address_steps=40, read_steps=60,
                          traj_stride=8, atoms_per_item=8, min_atoms=32,
                          min_atoms_base=32, n_query_per_item=2)
    system = build_system(cfg, key=jax.random.PRNGKey(0), loud=False)
    q0 = np.zeros((3, system.store.dim), dtype=np.float32)
    q0[:, : cfg.addr_dim] = np.array([[0.3, -0.2, 0.1, 0.0],
                                      [-0.4, 0.5, -0.1, 0.2],
                                      [0.0, 0.0, 0.6, -0.3]], dtype=np.float32)
    ref = system.read(q0)
    traj, phase, state = differentiable_read(system.model(), jnp.asarray(q0), cfg,
                                             implicit_q_star=False)
    assert traj.shape == ref.traj.shape
    assert list(phase) == list(ref.phase)
    assert np.allclose(np.asarray(traj), np.asarray(ref.traj), atol=1e-5)
    assert np.allclose(np.asarray(state.q_star), np.asarray(ref.state.q_star),
                       atol=1e-5)
    assert np.allclose(np.asarray(state.q_addr), np.asarray(ref.state.q_addr),
                       atol=1e-5)


def test_differentiable_read_with_implicit_q_star_sends_no_gradient_to_phi():
    """The structural claim, on the real harness read path (not the toy)."""
    from chlu.core.clu_system import CluSystemConfig, build_system
    from chlu.experiments.exp_trajectory_read import differentiable_read

    cfg = CluSystemConfig(seed=0, capacity=4, address_steps=40, read_steps=60,
                          traj_stride=8, atoms_per_item=8, min_atoms=32,
                          min_atoms_base=32)
    system = build_system(cfg, key=jax.random.PRNGKey(0), loud=False)
    model = system.model()
    phi = LearnedPhi(cfg.addr_dim, system.store.dim, cfg.addr_dim, cfg.payload_dim,
                     key=jax.random.PRNGKey(11))
    x = jnp.asarray(np.array([[0.3, -0.2, 0.1, 0.0]], dtype=np.float32))

    def point_loss(p):
        _t, _ph, st = differentiable_read(model, p(x), cfg, implicit_q_star=True)
        return jnp.sum(st.q_star**2)

    def traj_loss(p):
        t, _ph, _st = differentiable_read(model, p(x), cfg, implicit_q_star=True)
        return jnp.sum(t**2)

    def norm(g):
        lv = jax.tree_util.tree_leaves(eqx.filter(g, eqx.is_inexact_array))
        return float(np.sqrt(sum(float(np.sum(np.asarray(v) ** 2)) for v in lv)))

    assert norm(eqx.filter_grad(point_loss)(phi)) == 0.0
    assert norm(eqx.filter_grad(traj_loss)(phi)) > 1e-6
