"""Tests for the CLU->CAFE bridge (``chlu.eval.cafe_model``).

The CAFE harness itself is an external, separately-cloned repo, so these tests
exercise the bridge against a LOCAL stand-in for ``cafe_bench.models.BaseModel``
that reproduces the harness contract verbatim (see ``_BaseModelStub``). That
keeps the suite runnable with no CAFE checkout while still pinning the two
things that can silently break the integration:

  * ``encode`` really returns ``(N, D)`` for ``(N, T, C)`` input, finite, with
    ``feature_names()`` exactly describing the columns; and
  * the lazy fit happens on the FIRST encode (the train split), so train and
    test are embedded by the SAME frozen CLU.
"""

import numpy as np
import pytest

from chlu.eval.cafe_model import CLUCafeMixin, CLUValleyMixin, _trend
from chlu.eval.config import CLUCafeEncodeConfig, CLUScorerConfig

# Tiny config so the CLU trains in ~a second (score quality is a downstream
# science question — these tests pin the CONTRACT).
TINY = CLUScorerConfig(
    hidden=8,
    epochs=2,
    batch_size=8,
    max_fit_windows=32,
    predict_horizon=4,
    relax_steps=3,
    residual_anchors=3,
    seed=0,
)
T, C = 12, 3


class _BaseModelStub:
    """Mirrors ``cafe_bench.models.base.BaseModel``'s default probes.

    Only the parts the bridge relies on: the probes encode TRAIN then TEST.
    """

    name = "stub"

    def anomaly_score(self, X_train, X_test):
        from sklearn.neighbors import NearestNeighbors

        Z_train = self.encode(X_train)
        Z_test = self.encode(X_test)
        nn = NearestNeighbors(n_neighbors=min(5, len(Z_train)))
        nn.fit(Z_train)
        d, _ = nn.kneighbors(Z_test)
        return d.mean(axis=1)


def _windows(n, seed, offset=0.0):
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    out = np.empty((n, T, C), np.float32)
    for i in range(n):
        phase = rng.uniform(0, 6.28, C)
        w = np.stack([np.sin(0.3 * t + phase[k]) for k in range(C)], axis=1)
        out[i] = w + 0.02 * rng.normal(size=w.shape) + offset
    return out


def _model(cls=CLUCafeMixin, **enc_kw):
    m = type("M", (cls, _BaseModelStub), {})
    return m(clu_config=TINY, encode_config=CLUCafeEncodeConfig(**enc_kw))


def test_encode_shape_and_feature_names_agree():
    m = _model()
    Z = m.encode(_windows(6, seed=1))
    assert Z.ndim == 2 and Z.shape[0] == 6
    assert np.isfinite(Z).all()
    # feature_names must describe the columns EXACTLY — the anomaly arms index
    # the embedding by name, so a silent drift here mislabels the score.
    assert len(m.feature_names()) == Z.shape[1]
    assert len(set(m.feature_names())) == Z.shape[1]


def test_lazy_fit_binds_to_the_first_encode_call():
    """Train and test must be embedded by the same frozen CLU."""
    m = _model()
    Xtr, Xte = _windows(8, seed=1), _windows(5, seed=2)
    m.encode(Xtr)
    fitted = m._shared.model
    m.encode(Xte)
    assert m._shared.model is fitted, "second encode must not refit"


def test_explicit_fit_prevents_test_split_contamination():
    """``fit`` pins the CLU to train even if test is encoded first."""
    m = _model()
    Xtr, Xte = _windows(8, seed=1), _windows(5, seed=2)
    m.fit(Xtr)
    fitted = m._shared.model
    m.encode(Xte)
    assert m._shared.model is fitted


def test_feature_group_selection_shrinks_the_embedding():
    full = _model()
    small = _model(feature_groups=("energy",))
    Zf = full.encode(_windows(4, seed=3))
    Zs = small.encode(_windows(4, seed=3))
    assert Zs.shape[1] == 4  # energy_mean/last/std/trend
    assert Zs.shape[1] < Zf.shape[1]
    assert small.feature_names() == [
        "energy_mean",
        "energy_last",
        "energy_std",
        "energy_trend",
    ]


def test_basin_coords_width_matches_channel_count():
    m = _model(feature_groups=("basin_coords",))
    Z = m.encode(_windows(4, seed=4))
    assert Z.shape[1] == C
    assert m.feature_names() == [f"q_star_{i}" for i in range(C)]


def test_valley_anomaly_arm_overrides_the_default_probe():
    """``clu_valley`` must produce its own score, not the kNN default."""
    valley = _model(CLUValleyMixin)
    plain = _model(CLUCafeMixin)
    Xtr, Xte = _windows(16, seed=5), _windows(6, seed=6, offset=4.0)
    s_valley = valley.anomaly_score(Xtr, Xte)
    s_knn = plain.anomaly_score(Xtr, Xte)
    assert s_valley.shape == (6,)
    assert np.isfinite(s_valley).all()
    assert not np.allclose(s_valley, s_knn)


def test_valley_arm_rejects_missing_features():
    m = _model(CLUValleyMixin, feature_groups=("energy",))
    with pytest.raises(ValueError, match="relax_residual"):
        m.anomaly_score(_windows(8, seed=7), _windows(4, seed=8))


def test_relax_budget_reports_the_dimensionless_product():
    """gamma*steps*dt is what controls settling — the knob we actually tune."""
    enc = CLUCafeEncodeConfig(relax_gamma=0.5, relax_steps=64)
    cfg = CLUScorerConfig(dt=0.05, gamma=0.1, relax_steps=32)
    assert enc.relax_budget(cfg) == pytest.approx(0.5 * 64 * 0.05)
    # inheriting reproduces the (badly under-damped) scorer default
    assert CLUCafeEncodeConfig().relax_budget(cfg) == pytest.approx(0.16)


def test_relax_override_changes_the_settled_point():
    """A real relaxation must move q* relative to a near-zero-damping one."""
    lo = _model(feature_groups=("basin_coords",), relax_gamma=0.0, relax_steps=2)
    hi = _model(feature_groups=("basin_coords",), relax_gamma=5.0, relax_steps=64)
    X = _windows(6, seed=9)
    assert not np.allclose(lo.encode(X), hi.encode(X))


def test_trend_recovers_a_known_slope():
    import jax.numpy as jnp

    y = jnp.asarray([1.0, 3.0, 5.0, 7.0])  # slope 2 per index
    assert float(_trend(y)) == pytest.approx(2.0, rel=1e-5)


def test_encode_rejects_non_3d_input():
    m = _model()
    with pytest.raises(ValueError, match=r"\(N, T, C\)"):
        m.encode(np.zeros((4, 10), np.float32))


def test_unknown_feature_group_rejected():
    with pytest.raises(ValueError, match="unknown CAFE feature group"):
        CLUCafeEncodeConfig(feature_groups=("energy", "nonsense"))


def test_unknown_anomaly_mode_rejected():
    with pytest.raises(ValueError, match="anomaly_mode"):
        CLUCafeEncodeConfig(anomaly_mode="nope")
