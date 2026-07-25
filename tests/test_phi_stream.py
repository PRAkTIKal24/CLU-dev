"""Tests for the φ STREAM-DISCIPLINE study (w24).

Everything runs on a tiny SYNTHETIC *labelled* dataset injected through
``build_stream(..., data=(X, y))`` — no MNIST download, so the suite stays fast.
The real MNIST stream is exercised via the experiment CLI, not by pytest.
"""

import jax
import numpy as np
import pytest

from chlu.config import get_default_config
from chlu.experiments.exp_phi_stream import (
    ONLINE_STUB_NOTE,
    PHI_REGIMES,
    OnlineReadIn,
    build_stream,
    build_stream_read_in,
    cost_of_strictness,
    fit_pool_for_regime,
    run_experiment_phi_stream,
    run_stream_regime,
    store_advantage_watch_item,
    stream_laundering_control,
    task_classes,
)


@pytest.fixture
def float32_dynamics():
    """Pin float32 for the CLU-dynamics tests (handover §7.2 x64 isolation)."""
    was = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", False)
    yield
    jax.config.update("jax_enable_x64", was)


def _cfg():
    cfg = get_default_config().experiment_phi_stream
    cfg.n_tasks = 3
    cfg.classes_per_task = 2
    cfg.items_per_task = 4
    cfg.n_fit_pool = 40
    cfg.n_store_region = 120
    cfg.phi_dim = 6
    cfg.ae_hidden = 16
    cfg.ae_epochs = 15
    cfg.ae_batch = 16
    cfg.clu_steps = 25
    cfg.rollout_chunk = 32
    cfg.seeds = [0]
    return cfg


def _toy_data(n=240, dim=12, n_classes=6, seed=0):
    """Class-structured Gaussian blobs in [0,1] with integer labels 0..n_classes-1."""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(0.2, 0.8, size=(n_classes, dim))
    y = np.repeat(np.arange(n_classes), n // n_classes)
    X = np.clip(centers[y] + rng.normal(0, 0.08, size=(len(y), dim)), 0.0, 1.0)
    return X.astype(np.float32), y


def test_task_classes_are_split_mnist_shaped():
    cfg = _cfg()
    assert task_classes(cfg, 0) == [0, 1]
    assert task_classes(cfg, 2) == [4, 5]


def test_stream_fit_pools_respect_the_regimes():
    cfg = _cfg()
    X, y = _toy_data()
    stream = build_stream(cfg, seed=0, data=(X, y))
    # task-1-only φ may see ONLY task-1 classes; generic sees more than task 1
    pool_t1, prov_t1 = fit_pool_for_regime("task1_only", stream, cfg)
    pool_g, prov_g = fit_pool_for_regime("generic_frozen", stream, cfg)
    assert prov_t1["regime"] == "task1_only" and "PRIMARY" in prov_t1["role"]
    assert "upper bound" in prov_g["role"]
    # every task-1 fit row must be one of the task-1 class blobs: check by nearest
    # class centroid label from the raw data
    def _labels_of(pool):
        d2 = np.sum((np.asarray(pool)[:, None, :] - X[None, :, :]) ** 2, axis=-1)
        return y[np.argmin(d2, axis=1)]

    assert set(np.unique(_labels_of(pool_t1))) <= set(stream["task_classes"][0])
    assert len(set(np.unique(_labels_of(pool_g)))) > cfg.classes_per_task


def test_phi_never_sees_a_stored_pattern():
    """The fairness guarantee: store region and fit region are disjoint."""
    cfg = _cfg()
    X, y = _toy_data()
    stream = build_stream(cfg, seed=1, data=(X, y))
    stored = np.concatenate(stream["task_patterns"], axis=0)
    for pool in (stream["fit_pool_task1"], stream["fit_pool_generic"]):
        d2 = np.sum((stored[:, None, :] - np.asarray(pool)[None, :, :]) ** 2, axis=-1)
        assert np.all(np.min(d2, axis=1) > 1e-10)


def test_queries_are_identical_across_regimes():
    """Same store, same queries — the regimes may differ only in φ."""
    cfg = _cfg()
    X, y = _toy_data()
    a = build_stream(cfg, seed=3, data=(X, y))
    b = build_stream(cfg, seed=3, data=(X, y))
    for qa, qb in zip(a["task_queries"], b["task_queries"], strict=True):
        np.testing.assert_allclose(qa, qb)


def test_online_regime_is_a_stub_not_a_run():
    cfg = _cfg()
    X, y = _toy_data()
    stream = build_stream(cfg, seed=0, data=(X, y))
    assert "online" in PHI_REGIMES
    stub = OnlineReadIn(cfg)
    assert stub.implemented is False
    with pytest.raises(NotImplementedError):
        stub.observe_task(X[:4], 0)
    with pytest.raises(NotImplementedError):
        stub(X[:4])
    with pytest.raises(NotImplementedError):
        fit_pool_for_regime("online", stream, cfg)
    with pytest.raises(NotImplementedError):
        build_stream_read_in("online", "pca", stream, cfg, seed=0)
    assert "not implemented" in ONLINE_STUB_NOTE
    with pytest.raises(ValueError):
        fit_pool_for_regime("nonsense", stream, cfg)


def test_run_stream_regime_scores_every_seen_task(float32_dynamics):
    cfg = _cfg()
    X, y = _toy_data()
    stream = build_stream(cfg, seed=0, data=(X, y))
    run = run_stream_regime(cfg, stream, "task1_only", "pca", seed=0)
    assert len(run["rows"]) == cfg.n_tasks
    for t, row in enumerate(run["rows"], start=1):
        assert row["M"] == t * cfg.items_per_task
        assert len(row["per_task"]) == t  # every task seen so far is scored
        for tau in range(t):
            for line in ("clu_in_phi", "knn_in_phi"):
                m = row["per_task"][str(tau)][line]
                assert 0.0 <= m["identity_acc"] <= 1.0
                assert 0.0 <= m["class_acc"] <= 1.0
                assert m["n"] == cfg.items_per_task


def test_cost_of_strictness_reports_gap_and_slope(float32_dynamics):
    cfg = _cfg()
    X, y = _toy_data()
    stream = build_stream(cfg, seed=0, data=(X, y))
    runs = [
        run_stream_regime(cfg, stream, regime, "pca", seed=0)
        for regime in ("task1_only", "generic_frozen")
    ]
    cost = cost_of_strictness(runs, cfg, metric="identity_acc")["pca"]
    gap = cost["gap_generic_minus_task1"]
    assert len(gap["by_task_index"]) == cfg.n_tasks
    assert len(gap["by_stream_position"]) == cfg.n_tasks
    assert np.isfinite(gap["slope_per_task_index"])
    # the gap is a difference of accuracies -> bounded
    assert all(-1.0 <= g <= 1.0 for g in gap["by_task_index"])


def test_laundering_control_is_reported_in_every_regime(float32_dynamics):
    cfg = _cfg()
    X, y = _toy_data()
    stream = build_stream(cfg, seed=0, data=(X, y))
    runs = [
        run_stream_regime(cfg, stream, regime, "pca", seed=0)
        for regime in ("task1_only", "generic_frozen")
    ]
    out = stream_laundering_control(runs, cfg)
    assert {e["regime"] for e in out} == {"task1_only", "generic_frozen"}
    for e in out:
        assert e["n_points"] == cfg.n_tasks
        assert e["laundered"] == (e["n_clu_wins"] == 0)
        assert ("the win is φ's" in e["verdict"]) == e["laundered"]
    watch = store_advantage_watch_item(out)
    assert len(watch) == 1
    w = watch[0]
    assert w["delta_strict_minus_generic"] == pytest.approx(
        w["margin_task1_only"] - w["margin_generic_frozen"]
    )


def test_watch_item_fires_only_when_strict_regime_is_not_laundered():
    laundering = [
        {"arm": "pca", "regime": "task1_only", "metric": "identity_acc",
         "mean_clu_minus_knn": 0.10, "laundered": False},
        {"arm": "pca", "regime": "generic_frozen", "metric": "identity_acc",
         "mean_clu_minus_knn": -0.05, "laundered": True},
    ]
    w = store_advantage_watch_item(laundering)[0]
    assert w["store_advantage_under_strictness"] is True
    assert w["delta_strict_minus_generic"] == pytest.approx(0.15)
    laundering[0]["laundered"] = True
    assert store_advantage_watch_item(laundering)[0][
        "store_advantage_under_strictness"
    ] is False


def test_driver_end_to_end_on_toy_data(tmp_path, float32_dynamics):
    config = get_default_config()
    config.experiment_phi_stream = _cfg()
    res = run_experiment_phi_stream(
        config=config, save_dir=str(tmp_path / "plots"), data=_toy_data()
    )
    assert res["head_ruling"]["primary"] == "task1_only"
    assert res["online_regime_skipped"] is False  # default regimes exclude online
    assert set(res["cost_of_strictness"]) == {"identity_acc", "class_acc"}
    assert len(res["runs"]) == 2 * len(config.experiment_phi_stream.phi_arms)
    assert res["metrics_path"].endswith("exp_phi_stream_metrics.json")
