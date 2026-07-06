"""Tests for the calibration head, LTT wrapper, and selective metrics."""

import numpy as np
import pytest

from chlu.training.calibration import (
    CalibrationHead,
    fit_calibration_head,
    ltt_select_threshold,
)
from chlu.utils.metrics import (
    aurc,
    coverage_at_risk,
    expected_calibration_error,
    interpolate_risk_coverage,
    risk_coverage_curve,
)


# ---------------------------------------------------------------------------
# CalibrationHead
# ---------------------------------------------------------------------------


def _separable_split(n=100, seed=0):
    rng = np.random.default_rng(seed)
    R_right = rng.normal(-2.0, 0.3, n)
    R_wrong = rng.normal(2.0, 0.3, n)
    R = np.concatenate([R_right, R_wrong])
    margin = rng.normal(1.0, 0.2, 2 * n)  # uninformative
    wrong = np.concatenate([np.zeros(n, bool), np.ones(n, bool)])
    return R, margin, wrong


def test_head_separable_r_only():
    R, margin, wrong = _separable_split()
    head = fit_calibration_head(R=R, margin=margin, wrong=wrong, features="r")
    p = head.p_wrong(R=R)
    # p_wrong must increase with R and separate the classes
    assert p[wrong].min() > p[~wrong].max()
    from sklearn.metrics import roc_auc_score

    assert roc_auc_score(wrong.astype(int), p) == 1.0
    # learned threshold sits between the clusters
    tau = head.tau_r(0.5)
    assert -2.0 < tau < 2.0
    # monotone in R
    probe = head.p_wrong(R=np.array([-3.0, 0.0, 3.0]))
    assert probe[0] < probe[1] < probe[2]


def test_head_two_feature_uses_informative_feature():
    R, margin, wrong = _separable_split()
    head = fit_calibration_head(R=R, margin=margin, wrong=wrong, features="r_margin")
    # weight on the informative standardized feature (R) dominates
    assert abs(head.w[0]) > 5 * abs(head.w[1])
    p = head.p_wrong(R=R, margin=margin)
    assert (p[wrong].mean() - p[~wrong].mean()) > 0.9


def test_head_degenerate_single_class():
    R = np.linspace(-1, 1, 20)
    margin = np.ones(20)
    wrong = np.zeros(20, bool)  # self-test saw no failures
    head = fit_calibration_head(R=R, margin=margin, wrong=wrong, features="r")
    assert head.degenerate
    p = head.p_wrong(R=np.array([[0.0, 5.0]]))
    assert p.shape == (1, 2)
    assert np.allclose(p, head.p_const)
    assert head.p_const < 0.01
    assert np.isnan(head.tau_r())


def test_head_shape_preserved_and_round_trip():
    R, margin, wrong = _separable_split(n=30)
    head = fit_calibration_head(R=R, margin=margin, wrong=wrong, features="r_margin")
    R2 = R[:24].reshape(6, 4)
    m2 = margin[:24].reshape(6, 4)
    p = head.p_wrong(R=R2, margin=m2)
    assert p.shape == (6, 4)
    clone = CalibrationHead.from_dict(head.to_dict())
    assert np.allclose(clone.p_wrong(R=R2, margin=m2), p)
    assert clone.features == head.features and clone.n_fit == head.n_fit


def test_head_input_validation():
    with pytest.raises(ValueError):
        fit_calibration_head(R=np.zeros(3), wrong=np.zeros(3, bool), features="bogus")
    with pytest.raises(ValueError):
        fit_calibration_head(R=None, margin=None, wrong=np.zeros(3, bool), features="r")
    with pytest.raises(ValueError):
        fit_calibration_head(R=np.zeros(4), wrong=np.zeros(3, bool), features="r")


# ---------------------------------------------------------------------------
# Learn-then-Test wrapper
# ---------------------------------------------------------------------------


def test_ltt_certifies_calibrated_gate():
    # Well-calibrated p_wrong: errors only above p_wrong = 0.5
    n = 200
    p_wrong = np.linspace(0.01, 0.99, n)
    wrong = p_wrong > 0.5
    t, info = ltt_select_threshold(p_wrong, wrong, target_risk=0.05, delta=0.1)
    assert info["certified"] and t is not None
    # certified threshold keeps selective risk within target on the fit set
    sel = p_wrong <= t
    assert sel.sum() > 0
    assert wrong[sel].mean() <= 0.05
    # and it does not run far into the error region
    assert t < 0.6


def test_ltt_refuses_impossible_target():
    # errors everywhere: no threshold can achieve 0.1% selective risk
    rng = np.random.default_rng(1)
    p_wrong = rng.uniform(0, 1, 300)
    wrong = rng.uniform(0, 1, 300) < 0.5
    t, info = ltt_select_threshold(p_wrong, wrong, target_risk=0.001, delta=0.1)
    assert t is None and not info["certified"]


def test_ltt_empty_split():
    t, info = ltt_select_threshold(np.array([]), np.array([], bool), 0.05)
    assert t is None


# ---------------------------------------------------------------------------
# Selective-prediction metrics
# ---------------------------------------------------------------------------


def test_risk_coverage_hand_case():
    conf = np.array([0.9, 0.8, 0.7, 0.6])
    correct = np.array([True, True, False, True])
    cov, risk = risk_coverage_curve(conf, correct)
    assert np.allclose(cov, [0.25, 0.5, 0.75, 1.0])
    assert np.allclose(risk, [0.0, 0.0, 1 / 3, 1 / 4])
    assert np.isclose(aurc(cov, risk), (0 + 0 + 1 / 3 + 1 / 4) / 4)
    assert coverage_at_risk(cov, risk, 0.05) == 0.5
    assert coverage_at_risk(cov, risk, 0.30) == 1.0
    assert coverage_at_risk(cov, risk, -1.0) == 0.0  # nothing qualifies


def test_risk_coverage_order_invariance():
    rng = np.random.default_rng(0)
    conf = rng.uniform(size=50)
    correct = rng.uniform(size=50) < 0.7
    perm = rng.permutation(50)
    a = aurc(*risk_coverage_curve(conf, correct))
    b = aurc(*risk_coverage_curve(conf[perm], correct[perm]))
    assert np.isclose(a, b)


def test_ece_hand_case_uniform_bins():
    p = np.array([0.8, 0.8, 0.2, 0.2])
    correct = np.array([True, True, False, True])
    # 2 uniform bins: low bin conf .2 / acc .5 -> .3 * .5; high bin conf .8 /
    # acc 1. -> .2 * .5
    ece = expected_calibration_error(p, correct, n_bins=2, strategy="uniform")
    assert np.isclose(ece, 0.25)


def test_ece_perfect_and_constant():
    p = np.array([1.0, 1.0, 0.0, 0.0])
    correct = np.array([True, True, False, False])
    assert expected_calibration_error(p, correct, n_bins=2, strategy="uniform") == 0.0
    # all-identical predictions degrade to a single bin
    p_const = np.full(10, 0.7)
    correct = np.ones(10, bool)
    ece = expected_calibration_error(p_const, correct, n_bins=5)
    assert np.isclose(ece, 0.3)


def test_interpolate_risk_coverage():
    cov = np.array([0.5, 1.0])
    risk = np.array([0.0, 0.2])
    grid = np.array([0.25, 0.5, 0.75, 1.0])
    out = interpolate_risk_coverage(cov, risk, grid)
    assert np.allclose(out, [0.0, 0.0, 0.1, 0.2])
