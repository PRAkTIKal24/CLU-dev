"""Tests for the CLU anomaly scorer (the CLU->harness bridge, G7b)."""

import numpy as np
import pytest

from chlu.core.chlu_unit import CHLU
from chlu.core.lattice import CLULattice
from chlu.data.industrial.base import IndustrialDataset, UnitRecord
from chlu.eval.clu_scorer import CHLUScorer, _SharedCLUFit, make_clu_scorers
from chlu.eval.config import (
    CLU_DEFAULT_SCORE_MODES,
    CLULatticeConfig,
    CLUScorerConfig,
    EvalConfig,
    WindowConfig,
)
from chlu.eval.harness import evaluate_dataset, load_eval_npz

MANDATORY = ("pca_recon", "iforest", "lof", "knn")

# Tiny config so the CLU trains in ~a second (quality is a downstream question).
TINY = CLUScorerConfig(
    hidden=8,
    epochs=3,
    batch_size=8,
    max_fit_windows=64,
    predict_horizon=4,
    relax_steps=3,
    residual_anchors=3,
    seed=0,
)
SIZE, C = 16, 3


def _windows(n, size, c, seed, anomalous=False):
    rng = np.random.default_rng(seed)
    t = np.arange(size)
    out = np.empty((n, size, c), np.float32)
    for i in range(n):
        phase = rng.uniform(0, 6.28, c)
        w = np.stack([np.sin(0.3 * t + phase[k]) for k in range(c)], axis=1)
        w += 0.02 * rng.normal(size=w.shape)
        if anomalous:
            w += 5.0  # gross offset
        out[i] = w
    return out.reshape(n, size * c).astype(np.float32)


# ---------------------------------------------------------------------------
# factory / ABC compliance
# ---------------------------------------------------------------------------


def test_make_clu_scorers_keeps_baselines_and_adds_arms():
    cfg = EvalConfig(window=WindowConfig(size=SIZE))
    scorers = make_clu_scorers(cfg, TINY)
    for m in MANDATORY:
        assert m in scorers
    for mode in CLU_DEFAULT_SCORE_MODES:
        assert f"clu_{mode}" in scorers
        assert isinstance(scorers[f"clu_{mode}"], CHLUScorer)


def test_make_clu_scorers_rejects_unknown_mode():
    cfg = EvalConfig(window=WindowConfig(size=SIZE))
    with pytest.raises(ValueError, match="unknown CLU score mode"):
        make_clu_scorers(cfg, TINY, modes=("bogus",))


def test_arms_share_one_trained_model():
    cfg = EvalConfig(window=WindowConfig(size=SIZE))
    scorers = make_clu_scorers(cfg, TINY, modes=("energy", "predict"))
    shared = scorers["clu_energy"]._shared
    assert scorers["clu_predict"]._shared is shared
    train = _windows(40, SIZE, C, seed=1)
    scorers["clu_energy"].fit(train)
    model_after_first = shared.model
    scorers["clu_predict"].fit(train)  # must reuse, not retrain
    assert shared.model is model_after_first


# ---------------------------------------------------------------------------
# fit / score numerics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["energy", "residual", "predict", "hybrid"])
def test_each_arm_produces_finite_scores(mode):
    shared = _SharedCLUFit(TINY, window_size=SIZE)
    train = _windows(48, SIZE, C, seed=2)
    scorer = CHLUScorer(mode, shared, name=f"clu_{mode}")
    scorer.fit(train)
    test = _windows(20, SIZE, C, seed=3)
    s = scorer.score(test)
    assert s.shape == (20,)
    assert np.all(np.isfinite(s))
    assert np.std(s) > 0  # not a constant


def test_energy_arm_flags_gross_anomaly():
    # A gross offset anomaly should not be *harder* than normal for the energy
    # arm; require directionally-sane separation (not superiority vs baselines).
    shared = _SharedCLUFit(TINY, window_size=SIZE)
    scorer = CHLUScorer("energy", shared)
    scorer.fit(_windows(64, SIZE, C, seed=4))
    s_norm = scorer.score(_windows(30, SIZE, C, seed=5, anomalous=False))
    s_anom = scorer.score(_windows(30, SIZE, C, seed=6, anomalous=True))
    assert np.mean(s_anom) > np.mean(s_norm)


def test_scorer_before_fit_raises():
    shared = _SharedCLUFit(TINY, window_size=SIZE)
    with pytest.raises(RuntimeError, match="before fit"):
        CHLUScorer("energy", shared).score(_windows(4, SIZE, C, seed=7))


def test_non_divisible_window_width_raises():
    shared = _SharedCLUFit(TINY, window_size=SIZE)
    bad = np.zeros((5, SIZE * C + 1), np.float32)  # not divisible by SIZE
    with pytest.raises(ValueError, match="not divisible"):
        shared.ensure_fit(bad)


# ---------------------------------------------------------------------------
# lattice (G7b torus-coset) hook
# ---------------------------------------------------------------------------


def test_single_unit_builds_chlu():
    shared = _SharedCLUFit(TINY, window_size=SIZE)
    shared.ensure_fit(_windows(16, SIZE, C, seed=8))
    assert isinstance(shared.model, CHLU)
    assert shared.model.dim == C


def test_lattice_hook_builds_lattice_of_matching_dim():
    cfg = CLUScorerConfig(
        hidden=8, epochs=2, batch_size=8, max_fit_windows=32,
        predict_horizon=3, relax_steps=2, residual_anchors=2, seed=0,
        kinetic_mode="newtonian_learned",
        lattice=CLULatticeConfig(unit_dim=2, topology="chain",
                                 potential_type="so2_invariant"),
    )
    shared = _SharedCLUFit(cfg, window_size=SIZE)
    shared.ensure_fit(_windows(16, SIZE, 4, seed=9))  # C=4 -> 2 units of dim 2
    assert isinstance(shared.model, CLULattice)
    assert shared.model.dim == 4
    assert shared.model.n_units == 2


def test_lattice_hook_requires_exact_tiling():
    cfg = CLUScorerConfig(lattice=CLULatticeConfig(unit_dim=2))
    shared = _SharedCLUFit(cfg, window_size=SIZE)
    with pytest.raises(ValueError, match="not divisible by"):
        shared.ensure_fit(_windows(8, SIZE, 3, seed=10))  # C=3 not divisible by 2


# ---------------------------------------------------------------------------
# literal joint-angle -> so2-coset torus map (G7b flagship)
# ---------------------------------------------------------------------------

# voraus-like: 6 joints -> 12 cos/sin channels + 15 aux = 27 (NOT a clean tile)
LIT_C, LIT_NSO2 = 27, 6


def _literal_cfg(topology="ring", shuffle=False, **kw):
    base = dict(
        hidden=8, epochs=2, batch_size=8, max_fit_windows=32,
        predict_horizon=3, relax_steps=2, residual_anchors=2, seed=0,
    )
    base.update(kw)
    return CLUScorerConfig(
        lattice=CLULatticeConfig(
            layout="literal", n_so2_units=LIT_NSO2, aux_unit_dim=4,
            topology=topology, shuffle_angles=shuffle, shuffle_seed=1,
        ),
        **base,
    )


def test_literal_lattice_so2_plus_aux_no_padding():
    shared = _SharedCLUFit(_literal_cfg(topology="chain"), window_size=SIZE)
    shared.ensure_fit(_windows(24, SIZE, LIT_C, seed=11))
    m = shared.model
    assert isinstance(m, CLULattice)
    assert m.dim == LIT_C  # total dim == C: nothing padded, nothing dropped
    # 6 so2 units (dim 2) + aux tiling of 15 -> [4, 4, 7] (last absorbs remainder)
    assert m.unit_dims == (2, 2, 2, 2, 2, 2, 4, 4, 7)
    assert m.n_units == 9


def test_literal_ring_closes_the_chain():
    chain = _SharedCLUFit(_literal_cfg(topology="chain"), window_size=SIZE)
    chain.ensure_fit(_windows(24, SIZE, LIT_C, seed=12))
    ring = _SharedCLUFit(_literal_cfg(topology="ring"), window_size=SIZE)
    ring.ensure_fit(_windows(24, SIZE, LIT_C, seed=12))
    # ring = chain + the closing (n-1, 0) bond
    assert len(ring.model.edges) == len(chain.model.edges) + 1
    assert (LIT_NSO2 - 1, 0) in ring.model.edges


def test_literal_shuffle_control_changes_topology():
    ordered = _SharedCLUFit(_literal_cfg(shuffle=False), window_size=SIZE)
    ordered.ensure_fit(_windows(24, SIZE, LIT_C, seed=13))
    shuffled = _SharedCLUFit(_literal_cfg(shuffle=True), window_size=SIZE)
    shuffled.ensure_fit(_windows(24, SIZE, LIT_C, seed=13))
    # same number of bonds, but connecting different (permuted) cosets
    assert len(ordered.model.edges) == len(shuffled.model.edges)
    assert set(ordered.model.edges) != set(shuffled.model.edges)


@pytest.mark.parametrize("mode", ["energy", "residual", "predict"])
def test_literal_arms_finite(mode):
    shared = _SharedCLUFit(_literal_cfg(), window_size=SIZE)
    train = _windows(32, SIZE, LIT_C, seed=14)
    scorer = CHLUScorer(mode, shared)
    scorer.fit(train)
    s = scorer.score(_windows(10, SIZE, LIT_C, seed=15))
    assert s.shape == (10,)
    assert np.all(np.isfinite(s))


def test_literal_torus_needs_perfect_square():
    # 6 joints is not a perfect square -> 2-D torus is impossible; must guide to ring
    shared = _SharedCLUFit(_literal_cfg(topology="torus"), window_size=SIZE)
    with pytest.raises(ValueError, match="perfect square"):
        shared.ensure_fit(_windows(16, SIZE, LIT_C, seed=16))


def test_literal_rejects_too_many_so2():
    cfg = _literal_cfg()
    shared = _SharedCLUFit(cfg, window_size=SIZE)
    with pytest.raises(ValueError, match="exceeds C"):
        shared.ensure_fit(_windows(8, SIZE, 4, seed=17))  # C=4 < 2*6


def test_embed_joint_angles_layout():
    from chlu.data.industrial.voraus_ad import embed_joint_angles

    T = 5
    channels = ("robot_voltage", "joint_position_1", "motor_torque_1",
                "joint_position_2")
    data = np.zeros((T, 4), np.float32)
    data[:, 1] = 0.0  # theta_1 = 0 -> (cos,sin)=(1,0)
    data[:, 3] = np.pi / 2  # theta_2 = pi/2 -> (cos,sin)=(0,1)
    data[:, 0] = 7.0
    data[:, 2] = -3.0
    out, names, n_so2 = embed_joint_angles(data, channels)
    assert n_so2 == 2
    assert names == ("cos_joint_1", "sin_joint_1", "cos_joint_2", "sin_joint_2",
                     "robot_voltage", "motor_torque_1")
    np.testing.assert_allclose(out[0, :4], [1.0, 0.0, 0.0, 1.0], atol=1e-6)
    assert out[0, 4] == 7.0 and out[0, 5] == -3.0


# ---------------------------------------------------------------------------
# episode-labelled datasets report AUC-ROC primary (voraus protocol; Blocker 2)
# ---------------------------------------------------------------------------


class _ToyEpisodeDataset(IndustrialDataset):
    name = "toy_episode"
    label_kind = "episode"
    protocol = "cross_unit"

    def __init__(self):
        self.root = None

    def is_available(self):
        return True

    def unit_ids(self):
        return tuple(f"tr{i}" for i in range(4)) + tuple(f"te{i}" for i in range(6))

    def train_ids(self):
        return tuple(f"tr{i}" for i in range(4))  # all normal

    def test_ids(self):
        return tuple(f"te{i}" for i in range(6))

    def load_unit(self, uid):
        i = int(uid[2:])
        anomalous = uid.startswith("te") and i % 2 == 0
        rng = np.random.default_rng(hash(uid) % 10_000)
        t = np.arange(160, dtype=np.float32)
        data = np.stack([np.sin(0.2 * t + p) for p in (0.0, 1.0, 2.0)], axis=1)
        data = data + 0.05 * rng.normal(size=data.shape).astype(np.float32)
        if anomalous:
            data += 3.0
        return UnitRecord(
            uid, data.astype(np.float32), ("c1", "c2", "c3"),
            episode_label=int(anomalous),
        )


def test_episode_dataset_reports_aucroc_primary_not_vuspr():
    eval_cfg = EvalConfig(window=WindowConfig(size=SIZE), metrics_mode="fast", seed=0)
    res = evaluate_dataset(
        _ToyEpisodeDataset(),
        config=eval_cfg,
        scorer_factory=lambda: make_clu_scorers(eval_cfg, TINY),
        verbose=False,
    )
    # Blocker 2: episode-labelled -> AUC-ROC is primary, VUS-PR is NOT computed
    assert res.primary_metric == "AUC-ROC"
    assert "VUS-PR" not in res.metric_names
    assert res.metric_names[0] == "AUC-ROC"
    # markdown puts the primary (AUC-ROC) first + bold
    md = res.to_markdown()
    assert "**AUC-ROC**" in md
    # episode reduce == mean (matches the voraus baseline-floors protocol)
    assert eval_cfg.episode_reduce == "mean"


# ---------------------------------------------------------------------------
# harness end-to-end + raw-score collection
# ---------------------------------------------------------------------------


class _ToyPointDataset(IndustrialDataset):
    name = "toy_clu"
    label_kind = "point"
    protocol = "cross_unit"

    def __init__(self):
        self.root = None

    def is_available(self):
        return True

    def _make(self, uid, seed, anomalous):
        rng = np.random.default_rng(seed)
        t = np.arange(200, dtype=np.float32)
        base = np.stack([np.sin(0.2 * t + p) for p in (0.0, 1.0, 2.0)], axis=1)
        data = base + 0.05 * rng.normal(size=base.shape).astype(np.float32)
        labels = np.zeros(200, np.int8)
        if anomalous:
            data[100:120] += 4.0
            labels[100:120] = 1
        return UnitRecord(uid, data, ("c1", "c2", "c3"), point_labels=labels)

    def unit_ids(self):
        return ("te0", "te1", "tr0", "tr1")

    def train_ids(self):
        return ("tr0", "tr1")

    def test_ids(self):
        return ("te0", "te1")

    @staticmethod
    def _seed(uid):
        return int.from_bytes(uid.encode(), "little") % 1000

    def load_unit(self, uid):
        return self._make(uid, self._seed(uid), anomalous=uid.startswith("te"))


def test_harness_end_to_end_with_clu_and_raw_scores(tmp_path):
    eval_cfg = EvalConfig(
        window=WindowConfig(size=SIZE, stride=1, train_stride=4),
        metrics_sliding_window=SIZE,
        metrics_mode="fast",
        seed=0,
    )
    raw = {}
    res = evaluate_dataset(
        _ToyPointDataset(),
        config=eval_cfg,
        scorer_factory=lambda: make_clu_scorers(eval_cfg, TINY),
        out_dir=tmp_path,
        raw_scores=raw,
        verbose=False,
    )
    # baselines + 3 CLU arms all present, no NaNs, finite VUS-PR
    for m in MANDATORY:
        assert m in res.methods
    for mode in CLU_DEFAULT_SCORE_MODES:
        assert f"clu_{mode}" in res.methods
    agg = res.aggregate()
    for m in res.methods:
        vus = agg[m]["VUS-PR"][0]
        assert np.isfinite(vus), (m, vus)
    # raw scores collected for every method, ROC-plottable
    for m in res.methods:
        assert m in raw
        assert raw[m]["scores"].shape == raw[m]["labels"].shape
        assert raw[m]["scores"].size > 0
    # npz round trip
    loaded = load_eval_npz(tmp_path / "eval_toy_clu.npz")
    np.testing.assert_array_equal(loaded.values, res.values)
