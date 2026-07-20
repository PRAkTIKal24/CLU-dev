"""Tests for the bounded-vs-informative rollout diagnostic.

The diagnostic exists to separate two things that a naive "is the rollout
stable?" check conflates: a rollout that stays BOUNDED because it is well
behaved, and one that stays bounded because it has COLLAPSED onto a fixed point
and carries no information about its input. These tests pin exactly that
discrimination, using analytic stand-in dynamics so the expected answer is known
in closed form rather than being whatever the current checkpoint happens to do.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from chlu.eval.rollout_diag import (
    RolloutDiagConfig,
    collapse_length,
    rollout_diagnostic,
    rollout_spread,
)


class _Collapsing:
    """Contracts every input toward the origin by ``rate`` per step.

    Bounded AND uninformative — the failure mode the diagnostic must catch.
    """

    def __init__(self, rate=0.5):
        self.rate = rate

    def __call__(self, q0, p0, steps, dt, gamma):
        # jnp (not np): these stubs are called inside a jax.vmap trace.
        qs = [q0 * self.rate ** (n + 1) for n in range(steps)]
        return jnp.stack([jnp.concatenate([q, jnp.zeros_like(q)]) for q in qs])


class _Rigid:
    """Translates every input by a constant — bounded and fully informative."""

    def __call__(self, q0, p0, steps, dt, gamma):
        qs = [q0 + (n + 1) * dt for n in range(steps)]
        return jnp.stack([jnp.concatenate([q, jnp.zeros_like(q)]) for q in qs])


def _anchors(n=64, c=3, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, c)), np.zeros((n, c))


# ── collapse_length ──────────────────────────────────────────────────────


def test_collapse_length_finds_first_crossing():
    s_rel = np.array([1.0, 0.5, 0.02, 0.005, 0.001])
    assert collapse_length(s_rel, 0.01) == 4  # 1-indexed


def test_collapse_length_is_none_when_never_collapsing():
    assert collapse_length(np.ones(10), 0.01) is None


def test_collapse_length_respects_threshold():
    s_rel = np.array([1.0, 0.3, 0.05, 0.005])
    assert collapse_length(s_rel, 0.01) == 4
    assert collapse_length(s_rel, 0.1) == 3


# ── the discrimination the diagnostic is FOR ─────────────────────────────


def test_collapsed_rollout_is_bounded_but_flagged():
    """A collapsing rollout must read as BOUNDED yet have a finite n*.

    This is the whole point: boundedness alone would score it as a success.
    """
    q0, p0 = _anchors()
    cfg = RolloutDiagConfig(horizon=32, gammas=(0.5,), n_anchors=64)
    out = rollout_diagnostic(_Collapsing(0.5), q0, p0, dt=0.05, cfg=cfg)["gamma=0.5"]

    assert out["bounded"] is True           # naive check says "stable"
    assert out["collapse_length"] is not None   # diagnostic says "dead"
    # rate 0.5/step => S_rel(n) = 0.5**n; first n with 0.5**n < 0.01 is n=7
    assert out["collapse_length"] == 7
    assert out["collapse_budget"] == pytest.approx(0.5 * 7 * 0.05)


def test_informative_rollout_has_no_collapse_length():
    q0, p0 = _anchors()
    cfg = RolloutDiagConfig(horizon=32, gammas=(0.0,), n_anchors=64)
    out = rollout_diagnostic(_Rigid(), q0, p0, dt=0.05, cfg=cfg)["gamma=0.0"]

    assert out["bounded"] is True
    assert out["collapse_length"] is None       # spread is preserved exactly
    assert out["S_rel"][-1] == pytest.approx(1.0, abs=1e-6)  # float32 rollout


def test_spread_curve_matches_closed_form():
    q0, p0 = _anchors()
    r = rollout_spread(_Collapsing(0.5), q0, p0, steps=8, dt=0.05, gamma=0.5)
    expected = np.array([0.5 ** (n + 1) for n in range(8)])
    assert np.allclose(r["S_rel"], expected, rtol=1e-6)


# ── error curve + persistence baseline ───────────────────────────────────


def test_persistence_baseline_is_reported_with_truth():
    q0, p0 = _anchors()
    cfg = RolloutDiagConfig(horizon=4, gammas=(0.0,), n_anchors=64)
    truth = np.stack([q0 + (n + 1) * 0.05 for n in range(4)], axis=1)
    out = rollout_diagnostic(_Rigid(), q0, p0, dt=0.05, cfg=cfg, truth=truth)["gamma=0.0"]

    # _Rigid reproduces `truth` exactly; persistence does not.
    assert np.allclose(out["mse"], 0.0, atol=1e-12)
    assert out["mse_persistence"][-1] > out["mse"][-1]


def test_collapsed_rollout_can_beat_persistence_and_still_be_dead():
    """The trap, made explicit: error alone does not detect collapse.

    Anchors centred near zero mean the collapsed prediction (-> 0) is a decent
    constant predictor, so a pure error check can rank the DEAD rollout above
    persistence. Only the spread curve exposes it.
    """
    q0, p0 = _anchors()
    cfg = RolloutDiagConfig(horizon=16, gammas=(0.5,), n_anchors=64)
    truth = np.zeros((len(q0), 16, q0.shape[1]))  # everything decays to 0
    out = rollout_diagnostic(_Collapsing(0.5), q0, p0, dt=0.05, cfg=cfg, truth=truth)
    out = out["gamma=0.5"]

    assert out["mse"][-1] < out["mse_persistence"][-1]   # "wins" on error
    assert out["bounded"] is True                        # "wins" on stability
    assert out["collapse_length"] == 7                   # but is uninformative


# ── config validation ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "kwargs",
    [
        {"horizon": 0},
        {"spread_threshold": 0.0},
        {"spread_threshold": 1.0},
        {"n_anchors": 1},
        {"gammas": (-0.1,)},
    ],
)
def test_config_rejects_invalid(kwargs):
    with pytest.raises(ValueError):
        RolloutDiagConfig(**kwargs)


def test_config_to_json_roundtrips():
    import json

    cfg = RolloutDiagConfig(horizon=8, gammas=(0.0, 0.5))
    assert json.loads(cfg.to_json())["horizon"] == 8
