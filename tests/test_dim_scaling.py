"""Tests for the address-space dimension-scaling measurement (exp_dim_scaling).

Everything under test here is DESIGNED, not learned. These tests pin the
d-dimensional generalization of the w19 ring register and — more importantly —
the guards that decide whether a cell counts as a measurement at all:

  * the **blank-landscape control**, which is what separates reading the stored
    memory from reading the address back (w19's full-state read scored 1.000 on
    a blank landscape for exactly that reason);
  * the **anti-decoration guard** — the payload channel is always launched at
    zero, so a payload-only read is structurally blind to the query;
  * the **query-noise convention**, where a per-axis sigma silently changes
    query precision with d and would confound the entire scaling curve.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chlu.config import get_default_config
from chlu.core.memory_potentials import (
    BallRegisterPotential,
    designed_payloads,
    designed_sites,
    site_separation,
)
from chlu.experiments.exp_dim_scaling import (
    apply_quick,
    evaluate_cell,
    make_ball_queries,
    measure_landscape_scales,
)


def _cfg():
    return get_default_config().experiment_dim_scaling


def _fast_cfg(**over):
    """A cheap cell: short rollouts, few queries. Same code path."""
    cfg = _cfg()
    cfg.steps = 200
    cfg.n_query_per_item = 8
    cfg.capture_n_dirs = 4
    cfg.capture_n_offsets = 4
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


# ---------------------------------------------------------------------------
# The designed d-dimensional potential
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("d", [1, 2, 3, 5])
def test_ball_potential_finite_and_differentiable(d):
    centers = designed_sites(d, 4, R=1.0, seed=0)
    V = BallRegisterPotential(designed_payloads(4, seed=0), centers)
    for q in (
        jnp.zeros(d + 1),
        jnp.ones(d + 1) * 1e-8,
        jnp.concatenate([centers[0], jnp.zeros(1)]),
        jnp.ones(d + 1) * 3.0,  # well outside the wall
    ):
        v = V(q)
        g = jax.grad(V)(q)
        assert jnp.isfinite(v), f"V not finite at {q}"
        assert jnp.all(jnp.isfinite(g)), f"grad V not finite at {q}"


def test_confinement_is_exactly_flat_inside_the_ball():
    """The whole point of the relu wall: inside the ball the ONLY structure is
    the item wells. A harmonic/quartic confinement would add a d-dependent
    restoring force and confound the dimension scaling with a geometry change."""
    d = 3
    R = 1.0
    # No wells, no payload -> V must be identically zero inside the ball.
    V = BallRegisterPotential(
        jnp.zeros(1), jnp.zeros((1, d)), R=R, b=0.0, kappa=0.0, c_conf=10.0
    )
    rng = np.random.default_rng(0)
    for _ in range(20):
        x = rng.normal(size=d)
        x = x / np.linalg.norm(x) * rng.uniform(0.0, 0.98 * R)
        q = jnp.asarray(np.concatenate([x, [0.0]]), dtype=jnp.float32)
        assert abs(float(V(q))) < 1e-6
        assert float(jnp.linalg.norm(jax.grad(V)(q))) < 1e-5
    # ...and strictly positive with a real restoring force outside.
    q_out = jnp.asarray(np.concatenate([np.ones(d) * R, [0.0]]), dtype=jnp.float32)
    assert float(V(q_out)) > 0.0


def test_payload_profile_reproduces_stored_values_at_sites():
    """s(c_k) ~ a_k: the payload really is written at each site."""
    d, K = 3, 6
    pay = designed_payloads(K, seed=0)
    centers = designed_sites(d, K, R=1.0, seed=0)
    # Narrow wells so neighbouring bumps do not bleed into each other.
    V = BallRegisterPotential(pay, centers, w=0.05)
    s = jnp.array([V.payload_profile(c) for c in centers])
    np.testing.assert_allclose(np.asarray(s), np.asarray(pay), atol=2e-2)


def test_ball_potential_rejects_undersized_dim():
    with pytest.raises(ValueError):
        BallRegisterPotential(jnp.zeros(2), jnp.zeros((2, 3)), dim=3)


def test_ball_potential_rejects_payload_center_mismatch():
    with pytest.raises(ValueError):
        BallRegisterPotential(jnp.zeros(5), jnp.zeros((2, 3)))


# ---------------------------------------------------------------------------
# Site packing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("d", [2, 3, 4])
def test_designed_sites_lie_in_the_ball_and_are_deterministic(d):
    a = designed_sites(d, 16, R=1.0, seed=0)
    b = designed_sites(d, 16, R=1.0, seed=0)
    np.testing.assert_allclose(np.asarray(a), np.asarray(b))
    assert a.shape == (16, d)
    assert float(jnp.linalg.norm(a, axis=1).max()) <= 1.0 + 1e-6


@pytest.mark.parametrize("d", [2, 3, 4])
def test_achieved_separation_tracks_the_volume_argument(d):
    """Farthest-point sampling should achieve Delta ~ 2 R K^(-1/d). This is the
    packing assumption the whole capacity law rests on, so it is measured rather
    than assumed (loosely bounded — it is an order-of-magnitude claim)."""
    R, K = 1.0, 32
    sep = site_separation(designed_sites(d, K, R=R, seed=0))
    predicted = 2.0 * R * K ** (-1.0 / d)
    assert 0.4 * predicted < sep < 2.0 * predicted, (
        f"d={d}: achieved separation {sep:.3f} vs predicted {predicted:.3f}"
    )


def test_site_separation_of_a_single_site_is_infinite():
    assert site_separation(np.zeros((1, 3))) == float("inf")


# ---------------------------------------------------------------------------
# Queries — the anti-decoration guard and the noise convention
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("d", [2, 4])
def test_queries_never_carry_payload_information(d):
    """The payload channel must be launched at exactly zero in both q and p,
    otherwise a payload-only read could be reading the query back."""
    cfg = _cfg()
    centers = designed_sites(d, 4, R=cfg.R, seed=0)
    Q0, P0, labels = make_ball_queries(jax.random.PRNGKey(0), centers, 8, cfg)
    assert np.allclose(np.asarray(Q0[:, d]), 0.0)
    assert np.allclose(np.asarray(P0[:, d]), 0.0)
    assert Q0.shape == (32, d + 1)
    assert set(labels.tolist()) == {0, 1, 2, 3}


def test_fixed_norm_mode_holds_query_precision_constant_across_d():
    """The confound this guards: a per-axis sigma gives a jitter NORM growing as
    sigma*sqrt(d), so 'the same sigma' silently degrades query precision as d
    grows and would masquerade as a capacity limit."""
    cfg = _cfg()
    cfg.query_noise_mode = "fixed_norm"
    norms = {}
    for d in (2, 8, 16):
        centers = jnp.zeros((1, d))
        Q0, _, _ = make_ball_queries(jax.random.PRNGKey(0), centers, 512, cfg)
        norms[d] = float(np.mean(np.linalg.norm(np.asarray(Q0)[:, :d], axis=1)))
    # All within 15% of query_sigma, and of each other, regardless of d.
    for d, v in norms.items():
        assert 0.85 * cfg.query_sigma < v < 1.15 * cfg.query_sigma, (d, norms)

    cfg.query_noise_mode = "per_axis"
    centers = jnp.zeros((1, 16))
    Q0, _, _ = make_ball_queries(jax.random.PRNGKey(0), centers, 512, cfg)
    n16 = float(np.mean(np.linalg.norm(np.asarray(Q0)[:, :16], axis=1)))
    # per_axis at d=16 must be ~sqrt(16)=4x larger — the two arms are genuinely
    # different questions, which is why both are reported.
    assert n16 > 3.0 * norms[16]


def test_unknown_query_noise_mode_raises():
    cfg = _cfg()
    cfg.query_noise_mode = "gaussian-ish"
    with pytest.raises(ValueError, match="query_noise_mode"):
        make_ball_queries(jax.random.PRNGKey(0), jnp.zeros((2, 2)), 4, cfg)


# ---------------------------------------------------------------------------
# Landscape scales are MEASURED, not assumed
# ---------------------------------------------------------------------------


def test_measured_basin_width_recovers_the_gaussian_shape_parameter():
    """w is read off the landscape as the radius of maximal restoring force,
    which for a Gaussian well recovers the shape parameter. That is what makes
    it a measurement rather than a redefinition of the config value.

    Also guards the payload-force artifact: probing with y=0 instead of the
    payload equilibrium y=s(x) inflated the measured width by ~45%."""
    cfg = _fast_cfg()
    for w in (0.15, 0.30):
        sc = measure_landscape_scales(cfg, d=2, w=w, seed=0)
        assert abs(sc["w_measured_force_max"] - w) < 0.25 * w, sc
        assert abs(sc["R_measured_wall"] - (cfg.R + cfg.wall_margin)) < 0.1


# ---------------------------------------------------------------------------
# The loop, and the load-bearing blank control
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("d", [2, 3])
def test_small_K_retrieves_and_blank_control_passes(d):
    """A cell that retrieves must ALSO have a passing blank control; a cell
    without one is not a measurement."""
    cell = evaluate_cell(_fast_cfg(), d=d, K=4, seed=0)
    assert cell["written"]["acc_codebook"] > 0.9
    assert cell["written"]["selectivity"] > 0.9
    assert cell["blank_passes"], cell["blank"]
    assert cell["retrieved"]


def test_blank_landscape_reads_at_chance():
    """With nothing written, the payload-only read must be at chance. If this
    ever passes above chance the read is leaking the address."""
    cfg = _fast_cfg()
    cell = evaluate_cell(cfg, d=3, K=8, seed=0)
    assert cell["blank"]["acc_codebook"] <= cell["chance"] + cfg.blank_margin
    # And the blank payload channel stores nothing at all. NOTE: R^2 is
    # deliberately None here — on an all-zero codebook the regression target is
    # constant, so R^2 is a degenerate 0/0 that evaluates to a misleading 1.000.
    # The honest "nothing stored" check is the channel magnitude.
    assert cell["blank"]["payload_r2"] is None
    assert cell["blank"]["payload_r2_undefined_constant_target"]
    assert cell["blank"]["payload_abs_level"] < 1e-2


def test_blank_guard_actually_gates_retrieved():
    """Regression: `retrieved` must be False when the blank control fails, even
    if the written accuracy is perfect. Guards against the guard being cosmetic."""
    cfg = _fast_cfg()
    cfg.blank_margin = -1.0  # force every blank control to fail
    cell = evaluate_cell(cfg, d=2, K=4, seed=0)
    assert not cell["blank_passes"]
    assert not cell["retrieved"], "a failing blank control must veto `retrieved`"


def test_capacity_exceeds_the_w19_ring_ceiling_in_higher_d():
    """The headline in miniature: the 2-D ring ceiling was 8 items. A 4-D address
    ball must retrieve materially more than that at the same fidelity criterion.

    This is the cheap regression form of the K_max-vs-d curve — if it ever fails,
    the scaling result has changed and the report needs re-reading."""
    cell = evaluate_cell(_fast_cfg(), d=4, K=32, seed=0)
    assert cell["blank_passes"], cell["blank"]
    assert cell["written"]["acc_codebook"] > 0.9, cell["written"]
    assert cell["written"]["selectivity"] > 0.9, cell["written"]


# ---------------------------------------------------------------------------
# Config / CLI wiring
# ---------------------------------------------------------------------------


def test_dim_scaling_config_present_and_round_trips(tmp_path):
    from chlu.config import load_config, save_config

    cfg = get_default_config()
    assert hasattr(cfg, "experiment_dim_scaling")
    cfg.experiment_dim_scaling.well_width = 0.37
    cfg.experiment_dim_scaling.dims = [2, 5]
    p = tmp_path / "c.yaml"
    save_config(cfg, p)
    back = load_config(p)
    assert back.experiment_dim_scaling.well_width == pytest.approx(0.37)
    assert back.experiment_dim_scaling.dims == [2, 5]


def test_apply_quick_shrinks_the_sweep():
    cfg = get_default_config()
    apply_quick(cfg)
    assert cfg.experiment_dim_scaling.dims == [2, 3]
    assert max(cfg.experiment_dim_scaling.k_ladder) <= 16


def test_cli_exposes_exp_dim_scaling():
    import argparse

    from chlu.cli.experiment_cmd import setup_experiment_parsers

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    setup_experiment_parsers(sub)
    args = parser.parse_args(["exp-dim-scaling", "--quick"])
    assert args.quick is True
    assert hasattr(args, "func")
