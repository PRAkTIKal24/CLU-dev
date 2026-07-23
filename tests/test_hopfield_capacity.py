"""Tests for the Hopfield-capacity benchmark (w22).

Everything here runs on tiny SYNTHETIC patterns so the suite stays fast and has
no dataset dependency. The MNIST/CIFAR arms are exercised by the experiment CLI,
not by pytest.
"""

import jax
import jax.numpy as jnp
import numpy as np

from chlu.config import get_default_config
from chlu.core.memory_potentials import GaussianMemoryPotential
from chlu.experiments.exp_hopfield_capacity import (
    ACTIVATIONS,
    _entmax15,
    _median_nn_distance,
    _sparsemax,
    capacity_sweep,
    clu_retrieve,
    dropout_query,
    hopfield_retrieve,
    score_retrieval,
)


def _orthogonalish_patterns(M, D, seed=0):
    """Well-separated random binary-ish patterns in [0,1]."""
    rng = np.random.default_rng(seed)
    return jnp.asarray((rng.random((M, D)) > 0.5).astype(np.float32))


def test_sparsemax_and_entmax_are_simplex():
    key = jax.random.PRNGKey(1)
    S = jax.random.normal(key, (7, 5))
    for act in (_sparsemax, _entmax15):
        p = act(S)
        assert p.shape == S.shape
        assert np.all(np.asarray(p) >= -1e-5)
        np.testing.assert_allclose(np.sum(np.asarray(p), axis=-1), 1.0, atol=1e-4)


def test_sparsemax_is_sparser_than_softmax():
    key = jax.random.PRNGKey(2)
    S = jax.random.normal(key, (10, 8)) * 3.0
    soft = np.asarray(ACTIVATIONS["softmax"](S))
    sparse = np.asarray(ACTIVATIONS["sparsemax"](S))
    # sparsemax puts exact zeros; softmax never does
    assert np.mean(sparse == 0.0) > 0.0
    assert np.mean(soft == 0.0) == 0.0


def test_dropout_query_zeros_and_rescales():
    X = jnp.ones((100, 50))
    q = np.asarray(dropout_query(X, 0.5, jax.random.PRNGKey(0)))
    # survivors are scaled to 2.0, dropped to 0.0
    vals = np.unique(np.round(q, 4))
    assert set(vals.tolist()).issubset({0.0, 2.0})
    frac_kept = np.mean(q > 0)
    assert 0.4 < frac_kept < 0.6  # ~50% kept


def test_hopfield_retrieves_clean_patterns():
    P = _orthogonalish_patterns(8, 64)
    # clean query = the patterns themselves; a tuned Hopfield must return them
    xhat = hopfield_retrieve(P, P, 1.0, 1.0, 1, ACTIVATIONS["softmax"])
    m, nn, _ = score_retrieval(P, xhat, np.arange(8), 0.9)
    # with sharp-enough separation identity accuracy is high
    assert m["identity_acc"] >= 0.5


def test_gaussian_memory_potential_has_wells_at_patterns():
    P = _orthogonalish_patterns(4, 16)
    s = 0.5 * _median_nn_distance(P)
    V = GaussianMemoryPotential(P, s=s, b=1.0, alpha=1e-3)
    # potential is lower AT a stored pattern than at a random point
    v_at = float(V(P[0]))
    v_rand = float(V(jax.random.normal(jax.random.PRNGKey(0), (16,))))
    assert v_at < v_rand


def test_clu_register_settles_toward_nearest_well():
    P = _orthogonalish_patterns(6, 32, seed=3)
    cfg = get_default_config().experiment_hopfield_capacity
    cfg.clu_steps = 80
    s = cfg.clu_s_frac * _median_nn_distance(P)
    # launch from lightly-masked queries; the register should recover identity
    Q = dropout_query(P, 0.3, jax.random.PRNGKey(4))
    xhat, dt = clu_retrieve(P, Q, s, cfg)
    m, _, _ = score_retrieval(P, xhat, np.arange(6), 0.9)
    assert dt > 0
    assert xhat.shape == P.shape
    assert m["identity_acc"] >= 0.5  # settles to the right well most of the time


def test_capacity_sweep_synthetic_runs_and_has_all_arms():
    cfg = get_default_config().experiment_hopfield_capacity
    cfg.load_grid = [4, 8]
    cfg.n_data_pool = 64
    cfg.clu_steps = 40
    cfg.activations = ["softmax", "sparsemax"]
    cfg.rollout_chunk = 32
    res = capacity_sweep(cfg, "synthetic", seed=0)
    assert len(res["rows"]) == 2
    row = res["rows"][0]
    for arm in (
        "hopfield_softmax_repo",
        "hopfield_softmax_tuned",
        "nearest_neighbor",
        "clu_register",
    ):
        assert arm in row
        assert 0.0 <= row[arm]["identity_acc"] <= 1.0
