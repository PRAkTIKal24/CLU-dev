"""Tests for the mandatory statistical baselines + the harness end-to-end."""

import numpy as np
import pytest

from chlu.data.industrial.base import IndustrialDataset, UnitRecord
from chlu.eval.baselines import (
    IForestBaseline,
    KNNBaseline,
    LOFBaseline,
    PCAReconBaseline,
    make_default_baselines,
    sliding_windows,
    window_scores_to_point_scores,
)
from chlu.eval.config import EvalConfig, WindowConfig
from chlu.eval.harness import evaluate_dataset, load_eval_npz

MANDATORY = ("pca_recon", "iforest", "lof", "knn")


# ---------------------------------------------------------------------------
# windowing
# ---------------------------------------------------------------------------


def test_sliding_windows_shapes_and_values():
    x = np.arange(10, dtype=np.float32).reshape(5, 2)
    w = sliding_windows(x, size=2, stride=1)
    assert w.shape == (4, 4)
    np.testing.assert_array_equal(w[0], [0, 1, 2, 3])
    assert sliding_windows(x, size=2, stride=2).shape == (2, 4)
    with pytest.raises(ValueError, match="window size"):
        sliding_windows(x, size=6)


def test_window_scores_to_point_scores_overlap_average():
    # windows of size 2, stride 1 over 3 points; scores [1, 3]
    point = window_scores_to_point_scores(np.array([1.0, 3.0]), 3, size=2, stride=1)
    np.testing.assert_allclose(point, [1.0, 2.0, 3.0])


def test_window_scores_tail_fill_with_stride():
    # size 2, stride 2 over 5 points: windows cover [0,2) and [2,4); point 4 filled
    point = window_scores_to_point_scores(np.array([1.0, 2.0]), 5, size=2, stride=2)
    np.testing.assert_allclose(point, [1.0, 1.0, 2.0, 2.0, 2.0])


# ---------------------------------------------------------------------------
# baselines
# ---------------------------------------------------------------------------


def _toy_train_test(seed=0):
    rng = np.random.default_rng(seed)
    train = rng.normal(size=(400, 8)).astype(np.float32)
    normal = rng.normal(size=(100, 8)).astype(np.float32)
    anomalous = normal + 6.0  # gross offset anomaly
    return train, normal, anomalous


@pytest.mark.parametrize(
    "scorer",
    [
        PCAReconBaseline(n_components=0.9, seed=0),
        IForestBaseline(n_estimators=100, seed=0),
        LOFBaseline(n_neighbors=10),
        KNNBaseline(n_neighbors=5),
    ],
    ids=MANDATORY,
)
def test_each_baseline_separates_gross_anomaly(scorer):
    train, normal, anomalous = _toy_train_test()
    scorer.fit(train)
    s_norm = scorer.score(normal)
    s_anom = scorer.score(anomalous)
    assert s_norm.shape == (100,) and s_anom.shape == (100,)
    assert np.mean(s_anom) > np.mean(s_norm), scorer.name


def test_seeded_baselines_are_deterministic():
    train, normal, _ = _toy_train_test()
    a = IForestBaseline(n_estimators=50, seed=3).fit(train).score(normal)
    b = IForestBaseline(n_estimators=50, seed=3).fit(train).score(normal)
    np.testing.assert_array_equal(a, b)


def test_make_default_baselines_contains_exactly_the_mandatory_four():
    assert tuple(make_default_baselines(EvalConfig())) == MANDATORY


# ---------------------------------------------------------------------------
# harness end-to-end on synthetic in-memory datasets
# ---------------------------------------------------------------------------


class _ToyPointDataset(IndustrialDataset):
    """3 normal train units, 2 test units with one injected anomaly span."""

    name = "toy_point"
    label_kind = "point"
    protocol = "cross_unit"

    def __init__(self):  # bypass the file-root machinery
        self.root = None

    def is_available(self):
        return True

    def _make(self, uid, seed, anomalous):
        rng = np.random.default_rng(seed)
        t = np.arange(300, dtype=np.float32)
        base = np.stack([np.sin(0.2 * t + p) for p in (0.0, 1.0)], axis=1)
        data = base + 0.05 * rng.normal(size=base.shape).astype(np.float32)
        labels = np.zeros(300, np.int8)
        if anomalous:
            data[150:170] += 4.0
            labels[150:170] = 1
        return UnitRecord(uid, data, ("c1", "c2"), point_labels=labels)

    def unit_ids(self):
        return ("te0", "te1", "tr0", "tr1", "tr2")

    def train_ids(self):
        return ("tr0", "tr1", "tr2")

    def test_ids(self):
        return ("te0", "te1")

    @staticmethod
    def _seed(uid):
        return int.from_bytes(uid.encode(), "little") % 1000  # stable across runs

    def load_unit(self, uid):
        return self._make(uid, seed=self._seed(uid), anomalous=uid.startswith("te"))


class _ToyPrefixDataset(_ToyPointDataset):
    """TSB-AD style: one series, normal prefix + labelled suffix."""

    name = "toy_prefix"
    protocol = "per_unit_prefix"

    def unit_ids(self):
        return ("u0", "u1")

    def train_ids(self):
        return ()

    def test_ids(self):
        return self.unit_ids()

    def load_unit(self, uid):
        rec = self._make(uid, seed=self._seed(uid), anomalous=True)
        return UnitRecord(
            rec.unit_id,
            rec.data,
            rec.channels,
            point_labels=rec.point_labels,
            meta={"train_len": 100},
        )


CFG = EvalConfig(
    window=WindowConfig(size=16, stride=1, train_stride=2),
    metrics_sliding_window=16,
    metrics_mode="fast",
    seed=0,
)


def test_harness_cross_unit_end_to_end(tmp_path):
    res = evaluate_dataset(
        _ToyPointDataset(), config=CFG, out_dir=tmp_path, verbose=False
    )
    assert res.methods == MANDATORY
    assert res.values.shape == (4, 2, 4)  # methods x test units x fast metrics
    assert not np.isnan(res.values).any()
    # gross anomaly: every baseline should beat a random scorer handily
    agg = res.aggregate()
    for method in MANDATORY:
        mean_vus_pr = agg[method]["VUS-PR"][0]
        assert mean_vus_pr > 0.3, (method, mean_vus_pr)

    # npz round trip + markdown
    loaded = load_eval_npz(tmp_path / "eval_toy_point.npz")
    np.testing.assert_array_equal(loaded.values, res.values)
    assert loaded.methods == res.methods
    md = res.to_markdown()
    assert "toy_point" in md and "VUS-PR" in md and "pca_recon" in md
    assert "PA-F1" not in md


def test_harness_rejects_factory_without_mandatory_baselines():
    with pytest.raises(ValueError, match="mandatory"):
        evaluate_dataset(
            _ToyPointDataset(),
            config=CFG,
            verbose=False,
            scorer_factory=lambda: {"knn": KNNBaseline()},
        )


def test_harness_checks_unit_leakage():
    with pytest.raises(ValueError, match="leakage"):
        evaluate_dataset(
            _ToyPointDataset(),
            config=CFG,
            verbose=False,
            train_ids=("tr0", "te0"),
            test_ids=("te0", "te1"),
        )


def test_harness_per_unit_prefix_protocol():
    res = evaluate_dataset(_ToyPrefixDataset(), config=CFG, verbose=False)
    assert res.protocol == "per_unit_prefix"
    assert res.values.shape == (4, 2, 4)
    assert not np.isnan(res.values).any()
    assert any("per-unit-prefix" in n for n in res.notes)


def test_harness_episode_mode():
    class _ToyEpisodeDataset(_ToyPointDataset):
        name = "toy_episode"
        label_kind = "episode"

        def load_unit(self, uid):
            rec = self._make(uid, seed=self._seed(uid), anomalous=uid.startswith("te"))
            return UnitRecord(
                rec.unit_id,
                rec.data,
                rec.channels,
                episode_label=int(uid.startswith("te")),
            )

        def test_ids(self):
            return ("te0", "te1", "tr2")  # mixed labels for a defined AUROC

        def train_ids(self):
            return ("tr0", "tr1")

    res = evaluate_dataset(_ToyEpisodeDataset(), config=CFG, verbose=False)
    assert res.metric_names == ("AUC-ROC", "AUC-PR")
    assert res.values.shape == (4, 1, 2)
    assert not np.isnan(res.values).any()
    for i, method in enumerate(res.methods):
        assert res.values[i, 0, 0] == pytest.approx(1.0), method  # separable
