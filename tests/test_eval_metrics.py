"""Tests for the TSB-AD metric wrapper (VUS-PR primary; point-adjust banned)."""

from pathlib import Path

import numpy as np
import pytest

from chlu.eval.metrics import (
    EPISODE_METRICS,
    POINT_METRICS,
    assert_metric_allowed,
    compute_episode_metrics,
    compute_point_metrics,
)


def _lcg_floats(n, seed=123456789):
    """Pure-python LCG — bit-identical across numpy versions/envs."""
    x = seed
    out = []
    for _ in range(n):
        x = (1664525 * x + 1013904223) % (2**32)
        out.append(x / 2**32)
    return np.asarray(out, dtype=np.float64)


def _reference_case():
    score = _lcg_floats(500)
    labels = np.zeros(500, dtype=int)
    labels[120:150] = 1
    labels[400:410] = 1
    score[120:150] += 0.35
    score[400:410] += 0.2
    return score, labels


#: Reference output of upstream `TSB_AD.evaluation.metrics.get_metrics`
#: (TSB-AD==1.5 from PyPI, isolated env, python 3.11) on `_reference_case()`
#: with slidingWindow=25 — computed 2026-07-06 via
#: .claude/scratch/f2-eval-harness/tsb_reference.py. Proves the vendored copy
#: is numerically identical to the real TSB-AD harness.
#: (Upstream also reported PA-F1 = 1.0 — a *perfect* score on this barely
#: detectable case, AUC-ROC 0.727: the point-adjust inflation pathology that
#: justifies the project-wide ban. We do not produce that key at all.)
TSB_AD_15_REFERENCE = {
    "AUC-PR": 0.3674013771929051,
    "AUC-ROC": 0.7270108695652173,
    "VUS-PR": 0.40871685759697807,
    "VUS-ROC": 0.7908045552367258,
    "Standard-F1": 0.3999968000255998,
    "Event-based-F1": 0.9999999999999996,
    "R-based-F1": 0.4070796460176992,
    "Affiliation-F": 0.9965894175657659,
}


def test_vendored_matches_upstream_tsb_ad_reference():
    """Wrapper (vendored TSB-AD v1.5) == real TSB-AD v1.5 on identical input."""
    score, labels = _reference_case()
    got = compute_point_metrics(score, labels, sliding_window=25, mode="full")
    for name, expected in TSB_AD_15_REFERENCE.items():
        assert got[name] == pytest.approx(expected, abs=1e-10), name


def test_fast_mode_is_subset_of_full():
    score, labels = _reference_case()
    full = compute_point_metrics(score, labels, sliding_window=25, mode="full")
    fast = compute_point_metrics(score, labels, sliding_window=25, mode="fast")
    assert set(fast) == {"VUS-PR", "VUS-ROC", "AUC-PR", "AUC-ROC"}
    for k, v in fast.items():
        assert v == pytest.approx(full[k], abs=1e-12)


def test_point_metrics_keys_and_no_point_adjust():
    score, labels = _reference_case()
    m = compute_point_metrics(score, labels, sliding_window=25)
    assert tuple(m) == POINT_METRICS
    assert not any(
        "pa" in k.lower().replace("-", "") and "f1" in k.lower()
        for k in m
        if k != "Standard-F1"
    ), m.keys()


@pytest.mark.parametrize(
    "name", ["PA-F1", "pa_f1", "PointF1PA", "point-adjust-F1", "F1-PA"]
)
def test_forbidden_metric_names_raise(name):
    with pytest.raises(ValueError, match="forbidden"):
        assert_metric_allowed(name)


def test_allowed_metric_names_pass():
    for name in POINT_METRICS + EPISODE_METRICS:
        assert_metric_allowed(name)


def test_point_adjust_absent_from_vendored_source():
    """The excision is physical: no PA implementation exists in the harness."""
    vendor_dir = Path(__file__).parent.parent / "chlu" / "eval" / "_tsb_vendor"
    assert vendor_dir.is_dir()
    for py in vendor_dir.rglob("*.py"):
        text = py.read_text()
        assert "PointF1PA" not in text, py
        assert "_adjust_predicts" not in text, py
        assert "PA-F1" not in text, py


def test_degenerate_labels_give_nans():
    score = _lcg_floats(100)
    with pytest.warns(UserWarning, match="degenerate"):
        m = compute_point_metrics(score, np.zeros(100, int))
    assert all(np.isnan(v) for v in m.values())
    with pytest.warns(UserWarning, match="degenerate"):
        e = compute_episode_metrics(score[:10], np.ones(10, int))
    assert all(np.isnan(v) for v in e.values())


def test_input_validation():
    with pytest.raises(ValueError, match="differ"):
        compute_point_metrics(np.ones(5), np.zeros(4, int))
    bad = np.ones(6)
    bad[0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        compute_point_metrics(bad, np.array([0, 1, 0, 1, 0, 1]))
    with pytest.raises(ValueError, match="binary"):
        compute_point_metrics(np.ones(3), np.array([0, 1, 2]))


def test_episode_metrics_perfect_and_inverted():
    labels = np.array([0, 0, 0, 1, 1])
    perfect = compute_episode_metrics(np.array([0.1, 0.2, 0.3, 0.8, 0.9]), labels)
    assert perfect["AUC-ROC"] == pytest.approx(1.0)
    assert perfect["AUC-PR"] == pytest.approx(1.0)
    inverted = compute_episode_metrics(np.array([0.9, 0.8, 0.7, 0.2, 0.1]), labels)
    assert inverted["AUC-ROC"] == pytest.approx(0.0)
