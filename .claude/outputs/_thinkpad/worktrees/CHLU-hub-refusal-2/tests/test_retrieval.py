"""Tests for the hand-built write -> address -> retrieve loop (exp_retrieval).

Everything under test here is DESIGNED, not learned — these tests pin the
designed physics (the loop retrieves; the guards actually guard) and lock in the
two measurement bugs found while building it:

  * the radial half-life must be read off the ENVELOPE, not the first zero
    crossing of an oscillating |r - f|;
  * a ridge-to-one-hot linear probe on a 1-D non-monotone code fails for
    ESTIMATOR reasons, not retrieval reasons — ``linear_codebook_read`` must
    not.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chlu.config import get_default_config
from chlu.core.memory_potentials import (
    RingRegisterPotential,
    ThreeModePotential,
    designed_payloads,
)
from chlu.experiments.exp_retrieval import (
    build_ring_model,
    item3_write_modes,
    linear_codebook_read,
    linear_probe,
    make_queries,
)


def _cfg():
    return get_default_config().experiment_retrieval


# ---------------------------------------------------------------------------
# Designed potential
# ---------------------------------------------------------------------------


def test_ring_potential_finite_and_differentiable_at_origin():
    """The r=0 arctan2 singularity must be killed by the radial envelope."""
    V = RingRegisterPotential(designed_payloads(4, seed=0))
    for q in (
        jnp.zeros(3),
        jnp.array([0.0, 0.0, 0.5]),
        jnp.array([1e-8, -1e-8, 0.0]),
    ):
        v = V(q)
        g = jax.grad(V)(q)
        assert jnp.isfinite(v), f"V not finite at {q}"
        assert jnp.all(jnp.isfinite(g)), f"grad V not finite at {q}"


def test_payload_profile_reproduces_stored_values_at_sites():
    K = 6
    pay = designed_payloads(K, seed=0)
    V = RingRegisterPotential(pay, bump_width=0.02)
    theta = jnp.arange(K) * (2 * jnp.pi / K)
    s = V.payload_profile(theta)
    np.testing.assert_allclose(np.asarray(s), np.asarray(pay), atol=2e-2)


def test_designed_payloads_are_non_monotone():
    """The codebook must NOT be a monotone function of site index — that is what
    makes the payload read a real decode rather than a readout of the angle."""
    pay = np.asarray(designed_payloads(8, seed=0))
    assert not np.all(np.diff(pay) > 0)
    assert not np.all(np.diff(pay) < 0)
    assert len(set(pay.tolist())) == 8


# ---------------------------------------------------------------------------
# The loop itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("K", [2, 4])
def test_exact_launch_retrieves_the_stored_payload(K):
    """Launch at site k with q2(0)=0 -> the tail reads back a_k."""
    cfg = _cfg()
    pay = designed_payloads(K, seed=cfg.payload_seed)
    model = build_ring_model(pay, cfg)
    for k in range(K):
        th = 2 * np.pi * k / K
        q0 = jnp.array([cfg.f * np.cos(th), cfg.f * np.sin(th), 0.0])
        traj = model(q0, jnp.zeros(3), 800, cfg.dt, cfg.gamma)
        assert abs(float(traj[-1, 2]) - float(pay[k])) < 0.02


def test_blank_landscape_retrieves_nothing():
    """The anti-decoration guard: with nothing written, the payload channel must
    stay at ~0 no matter which address is used."""
    cfg = _cfg()
    K = 4
    pay = designed_payloads(K, seed=cfg.payload_seed)
    blank = build_ring_model(jnp.zeros_like(pay), cfg)
    for k in range(K):
        th = 2 * np.pi * k / K
        q0 = jnp.array([cfg.f * np.cos(th), cfg.f * np.sin(th), 0.0])
        traj = blank(q0, jnp.zeros(3), 800, cfg.dt, cfg.gamma)
        assert abs(float(traj[-1, 2])) < 1e-3


def test_queries_never_carry_payload_information():
    """q0[2] and p0[2] must be identically zero for every query."""
    cfg = _cfg()
    Q0, P0, labels = make_queries(jax.random.PRNGKey(0), 4, 8, cfg)
    assert np.allclose(np.asarray(Q0[:, 2]), 0.0)
    assert np.allclose(np.asarray(P0[:, 2]), 0.0)
    assert set(labels.tolist()) == {0, 1, 2, 3}


# ---------------------------------------------------------------------------
# Read-out estimators
# ---------------------------------------------------------------------------


def test_codebook_read_beats_onehot_probe_on_a_nonmonotone_code():
    """Regression test for the estimator artifact: on a clean 1-D non-monotone
    code, ridge-to-one-hot is crippled while the linear codebook read is exact.
    A drop in one-hot accuracy therefore must NOT be reported as interference."""
    pay = np.array([0.333, -1.0, -0.333, 1.0])  # non-monotone in index
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(4), 32)
    X = (pay[y] + rng.normal(0, 1e-3, size=y.shape))[:, None]
    n = len(y)
    perm = rng.permutation(n)
    tr, te = perm[: n // 2], perm[n // 2 :]

    acc_cb, r2 = linear_codebook_read(X[tr], y[tr], X[te], y[te], pay)
    acc_oh, _ = linear_probe(X[tr], y[tr], X[te], y[te], 4)

    assert acc_cb > 0.99, f"codebook read should be exact, got {acc_cb}"
    assert r2 > 0.99
    assert acc_oh < 0.9, (
        "one-hot probe is expected to FAIL on a 1-D non-monotone code; if it no "
        f"longer does, the interference numbers need re-reading (got {acc_oh})"
    )


# ---------------------------------------------------------------------------
# Three write modes
# ---------------------------------------------------------------------------


def test_permanent_mode_is_exactly_flat():
    """SO(2)-invariant channel: grad V is radial, so a purely radial write
    exerts no torque and the stored angle cannot drift."""
    V = ThreeModePotential()
    q = jnp.array([0.8, 0.6, 1.0, 0.0])  # off the vacuum radius on purpose
    g = jax.grad(V)(q)
    # angular component of the gradient in the (0,1) plane must vanish
    ang = float(-q[1] * g[0] + q[0] * g[1])
    assert abs(ang) < 1e-6, f"channel gradient has a torque component: {ang}"


def test_decaying_write_does_not_corrupt_its_permanent_neighbour():
    cfg = _cfg()
    cfg.tm_steps = 600
    res = item3_write_modes(cfg, seed=0)
    assert res["corruption_delta_theta"] < 1e-5
    assert res["uncorrelated_sign_retained"]


def test_radial_half_life_matches_2ln2_over_gamma():
    """The envelope half-life must follow n_1/2 = 2*ln2/gamma (per-step momentum
    damping (1-gamma) => amplitude envelope exp(-gamma*n/2)).

    Guards the metric bug: reading the first crossing of |r-f| < dr/2 instead of
    the envelope gave 6 steps against a true value near 69."""
    cfg = _cfg()
    cfg.tm_steps = 2000
    res = item3_write_modes(cfg, seed=0)
    predicted = 2 * np.log(2) / cfg.gamma
    measured = res["decaying_half_life_steps"]
    assert measured is not None
    assert 0.75 * predicted < measured < 1.25 * predicted, (
        f"half-life {measured} far from predicted {predicted}"
    )


# ---------------------------------------------------------------------------
# Config wiring
# ---------------------------------------------------------------------------


def test_retrieval_config_present_and_round_trips(tmp_path):
    from chlu.config import load_config, save_config

    cfg = get_default_config()
    assert hasattr(cfg, "experiment_retrieval")
    cfg.experiment_retrieval.barrier = 0.37
    cfg.experiment_retrieval.item_counts = [2, 3]
    p = tmp_path / "c.yaml"
    save_config(cfg, p)
    back = load_config(p)
    assert back.experiment_retrieval.barrier == pytest.approx(0.37)
    assert back.experiment_retrieval.item_counts == [2, 3]


def test_cli_exposes_exp_retrieval():
    import argparse

    from chlu.cli.experiment_cmd import setup_experiment_parsers

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    setup_experiment_parsers(sub)
    args = parser.parse_args(["exp-retrieval", "--quick"])
    assert args.quick is True
    assert hasattr(args, "func")
