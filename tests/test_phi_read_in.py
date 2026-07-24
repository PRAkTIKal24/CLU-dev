"""Tests for the φ read-in around a designed store (w23).

Everything runs on tiny SYNTHETIC patterns (i.i.d. Gaussian, no dataset
dependency) so the suite stays fast. The MNIST/CIFAR arms are exercised via the
experiment CLI, not by pytest.
"""

import jax
import numpy as np
import pytest

from chlu.config import get_default_config
from chlu.experiments.exp_phi_read_in import (
    AEReadIn,
    PCAReadIn,
    _auroc,
    _nearest_store_index,
    capacity_sweep_phi,
    knn_in_phi,
    laundering_control,
    load_store_and_fit_pools,
    retry_confidence_probe,
)


@pytest.fixture
def float32_dynamics():
    """Pin float32 for the CLU-dynamics tests (handover §7.2 x64 isolation)."""
    was = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", False)
    yield
    jax.config.update("jax_enable_x64", was)


def _cfg():
    cfg = get_default_config().experiment_phi_read_in
    cfg.n_data_pool = 48
    cfg.n_fit_pool = 96
    cfg.phi_dim = 8
    cfg.ae_hidden = 32
    cfg.ae_epochs = 20
    cfg.ae_batch = 32
    cfg.load_grid = [4, 8]
    cfg.clu_steps = 30
    cfg.activations = ["softmax", "sparsemax"]
    cfg.rollout_chunk = 32
    cfg.noise_fixed_load = 8
    return cfg


def test_store_and_fit_pools_are_disjoint():
    store, fit = load_store_and_fit_pools("synthetic", 40, 80, seed=0)
    # φ must never see the store: the two pools share no row
    S = np.asarray(store)
    F = np.asarray(fit)
    d2 = np.sum((S[:, None, :] - F[None, :, :]) ** 2, axis=-1)
    assert np.all(np.min(d2, axis=1) > 1e-8)  # no store row appears in the fit pool


def test_pca_read_in_shape_and_frozen():
    _, fit = load_store_and_fit_pools("synthetic", 40, 80, seed=1)
    phi = PCAReadIn(fit, k=8)
    f1 = np.asarray(phi(fit[:5]))
    f2 = np.asarray(phi(fit[:5]))
    assert f1.shape == (5, 8)
    np.testing.assert_allclose(f1, f2)  # frozen: deterministic


def test_ae_read_in_trains_and_encodes():
    _, fit = load_store_and_fit_pools("synthetic", 40, 120, seed=2)
    dim = int(fit.shape[1])
    phi = AEReadIn(fit, dim=dim, hidden=32, k=8, epochs=40, lr=1e-3, batch=32, seed=0)
    f = np.asarray(phi(fit[:6]))
    assert f.shape == (6, 8)
    assert phi.final_loss is not None and np.isfinite(phi.final_loss)


def test_knn_in_phi_matches_nearest_key():
    keys = np.array([[0.0, 0.0], [10.0, 10.0], [0.0, 10.0]], np.float32)
    feat_q = np.array([[0.1, 0.1], [9.0, 9.0]], np.float32)
    idx = knn_in_phi(keys, feat_q)
    assert list(idx) == [0, 1]
    idx2, d2 = _nearest_store_index(feat_q, keys)
    assert d2.shape == (2, 3)


def test_auroc_perfect_and_random():
    # perfect separation: score orders labels exactly
    scores = np.array([0.0, 1.0, 2.0, 3.0])
    labels = np.array([0, 0, 1, 1])
    assert _auroc(scores, labels) == pytest.approx(1.0)
    # single-class -> degenerate 0.5
    assert _auroc(scores, np.zeros(4)) == 0.5


def test_laundering_control_flags_tie_as_laundered():
    # construct a capacity result where CLU exactly ties kNN at both loads
    cap = [{
        "dataset": "synthetic", "arm": "pca",
        "rows": [
            {"M": 4, "clu_in_phi": {"identity_acc": 0.75},
             "knn_in_phi": {"identity_acc": 0.75}},
            {"M": 8, "clu_in_phi": {"identity_acc": 0.60},
             "knn_in_phi": {"identity_acc": 0.61}},
        ],
    }]
    out = laundering_control(cap)[0]
    assert out["laundered"] is True
    assert out["n_clu_wins"] == 0
    assert "the win is φ's" in out["verdict"]


def test_laundering_control_detects_clu_win():
    cap = [{
        "dataset": "synthetic", "arm": "ae",
        "rows": [
            {"M": 4, "clu_in_phi": {"identity_acc": 0.90},
             "knn_in_phi": {"identity_acc": 0.70}},
        ],
    }]
    out = laundering_control(cap)[0]
    assert out["laundered"] is False
    assert out["n_clu_wins"] == 1
    assert out["max_clu_margin"] == pytest.approx(0.20)


def test_capacity_sweep_phi_has_all_four_lines(float32_dynamics):
    cfg = _cfg()
    _, fit = load_store_and_fit_pools("synthetic", cfg.n_data_pool, cfg.n_fit_pool, 0)
    store, _ = load_store_and_fit_pools("synthetic", cfg.n_data_pool, cfg.n_fit_pool, 0)
    phi = PCAReadIn(fit, cfg.phi_dim)
    res = capacity_sweep_phi(cfg, "synthetic", "pca", phi, store, seed=0)
    assert len(res["rows"]) == 2
    row = res["rows"][0]
    for line in ("clu_in_phi", "knn_in_phi", "hopfield_softmax_in_phi",
                 "raw_space_clu"):
        assert line in row
        assert 0.0 <= row[line]["identity_acc"] <= 1.0
    # packing report present and sensible
    assert res["packing"]["d"] == cfg.phi_dim
    assert res["packing"]["well_width_s"] > 0


def test_retry_confidence_probe_returns_auroc(float32_dynamics):
    cfg = _cfg()
    store, fit = load_store_and_fit_pools("synthetic", cfg.n_data_pool,
                                          cfg.n_fit_pool, 0)
    phi = PCAReadIn(fit, cfg.phi_dim)
    out = retry_confidence_probe(cfg, "synthetic", "pca", phi, store, seed=0)
    assert 0.0 <= out["confidence_auroc"] <= 1.0
    assert out["n_total"] == cfg.noise_fixed_load
