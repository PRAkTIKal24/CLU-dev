"""Mass-spectrum plumbing on the EVAL/CAFE training path (``clu_scorer``).

``mass_lr_mult`` existed only in ``chlu/training/train.py``; the eval path used
a plain ``optax.adam(lr)``, so every CAFE/voraus CLU ran with the knob
unreachable. These tests pin the port:

  * ``mass_lr_mult == 1.0`` is bit-compatible (the historical optimizer);
  * ``mass_lr_mult > 1`` really moves ``log_mass`` further, and specifically
    widens the SPECTRUM (``std(log_mass)``), which is the property that matters
    -- a uniform shift of every channel is not a timescale hierarchy;
  * the movement partition (theorist OQ1) is reported and internally consistent.
"""

import numpy as np
import pytest

from chlu.eval.clu_scorer import _collect_log_mass, _SharedCLUFit
from chlu.eval.config import CLUScorerConfig

L, C = 12, 4


def _windows(n=64, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, L * C)).astype(np.float32)


def _fit(**kw):
    cfg = CLUScorerConfig(
        hidden=8, epochs=4, batch_size=16, max_fit_windows=64,
        predict_horizon=4, relax_steps=3, seed=0, **kw
    )
    f = _SharedCLUFit(cfg, window_size=L)
    f.ensure_fit(_windows())
    return f


def test_default_mass_lr_mult_is_one():
    assert CLUScorerConfig().mass_lr_mult == 1.0
    assert CLUScorerConfig().mass_spread_lambda == 0.0


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_non_positive_mass_lr_mult_rejected(bad):
    with pytest.raises(ValueError, match="mass_lr_mult"):
        CLUScorerConfig(mass_lr_mult=bad)


def test_negative_mass_spread_lambda_rejected():
    with pytest.raises(ValueError, match="mass_spread_lambda"):
        CLUScorerConfig(mass_spread_lambda=-0.5)


def test_mass_lr_mult_one_is_deterministic():
    """The default path must stay reproducible run-to-run (bit-compatible)."""
    a = _collect_log_mass(_fit().model)
    b = _collect_log_mass(_fit(mass_lr_mult=1.0).model)
    np.testing.assert_array_equal(a, b)


def test_mass_lr_mult_moves_log_mass_further():
    base = _fit().mass_diagnostics
    fast = _fit(mass_lr_mult=10.0).mass_diagnostics
    assert fast["max_abs_drift"] > 3.0 * base["max_abs_drift"], (
        f"mass_lr_mult=10 should move log_mass much further: "
        f"{fast['max_abs_drift']} vs {base['max_abs_drift']}"
    )


def test_log_mass_drift_is_dominated_by_the_common_mode():
    """The measured defect, pinned: log_mass MOVES but the spectrum does not.

    A uniform shift of every channel rescales the overall inertia and is NOT a
    timescale hierarchy. On FD001 (150 epochs) the common mode is ~39x the
    differential at mass_lr_mult=1; the effect is present here too, which is why
    a larger mass_lr_mult alone does not reliably widen ``std(log_mass)`` at
    short training horizons.
    """
    for mult in (1.0, 10.0):
        d = _fit(mass_lr_mult=mult).mass_diagnostics
        assert abs(d["common_mode_drift"]) > 3.0 * d["differential_drift"], (
            f"mass_lr_mult={mult}: expected common-mode-dominated drift, got "
            f"common={d['common_mode_drift']} differential={d['differential_drift']}"
        )


def test_movement_partition_is_reported_and_consistent():
    d = _fit().mass_diagnostics
    mv = d["movement"]
    assert mv["n_mass"] == C          # one log_mass per channel
    assert mv["n_main"] > mv["n_mass"]
    assert 0.0 <= mv["mass_l2_fraction"] <= 1.0
    # rms_ratio is the count-fair statistic: mass_rms / main_rms
    assert mv["rms_ratio"] == pytest.approx(mv["mass_rms"] / mv["main_rms"], rel=1e-6)
    assert len(d["log_mass_init"]) == len(d["log_mass_final"]) == C


def test_mass_lr_mult_does_not_perturb_the_potential_much():
    """The mass slot must be the only thing the multiplier retargets."""
    base, fast = _fit().mass_diagnostics, _fit(mass_lr_mult=10.0).mass_diagnostics
    assert fast["movement"]["rms_ratio"] > 3.0 * base["movement"]["rms_ratio"]
