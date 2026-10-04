"""Tests for the RETRY-COMPUTE study (w23) — the accuracy-vs-compute curve.

Everything runs on tiny SYNTHETIC patterns so the suite stays fast and has no
dataset dependency (the MNIST arm is exercised by the CLI). The float32 fixture
matches the reported numerics and dodges the repo-wide x64 test-isolation hazard
(handover §7.2); the experiment code itself is x64-safe.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chlu.config import get_default_config
from chlu.experiments.exp_hopfield_capacity import (
    _median_nn_distance,
    build_clu_memory,
    noise_query,
)
from chlu.experiments.exp_retry_compute import (
    _confidence_and_nn,
    _dt_of,
    _ensemble_ladder,
    _feedforward_ladder,
    _hopfield_ladder,
    _majority,
    _retry_ladder,
    _settle,
    aggregate_cells,
    block_query,
    erasure_mask,
    headroom_probe,
    levels_for,
    make_query,
    masked_nn_identity,
    packing_slack,
    parse_regimes,
    run_cell,
    select_store,
    survivor_scale,
)


@pytest.fixture
def float32_dynamics():
    was = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", False)
    yield
    jax.config.update("jax_enable_x64", was)


def _patterns(M, D, seed=0):
    rng = np.random.default_rng(seed)
    return jnp.asarray((rng.random((M, D)) > 0.5).astype(np.float32))


def _quick_cfg(M_pool=32):
    cfg = get_default_config().experiment_retry_compute
    cfg.retry_ladder = [0, 1, 2]
    cfg.conf_thresholds = [0.97, 1.0]
    cfg.main_threshold = 0.99
    cfg.retry_step_frac = 0.25
    cfg.clu_steps = 30
    cfg.rollout_chunk = 32
    cfg.n_data_pool = M_pool
    return cfg


def test_confidence_is_label_free_and_bounded():
    P = _patterns(6, 32, seed=1)
    reads = np.asarray(P)  # perfect reads -> confidence 1, nn == truth
    cos, nn = _confidence_and_nn(reads, P)
    np.testing.assert_allclose(cos, 1.0, atol=1e-4)
    np.testing.assert_array_equal(nn, np.arange(6))


def test_majority_vote():
    votes = np.array([[0, 1, 2], [0, 1, 5], [3, 1, 2]])  # (R=3, Nq=3)
    maj = _majority(votes)
    np.testing.assert_array_equal(maj, [0, 1, 2])


def test_settle_is_finite_and_shape_stable(float32_dynamics):
    P = _patterns(5, 24, seed=2)
    cfg = _quick_cfg()
    s = cfg.clu_s_frac * _median_nn_distance(P)
    model = build_clu_memory(P, s, cfg)
    Q = np.asarray(noise_query(P, 0.3, jax.random.PRNGKey(0)))
    P0 = np.zeros_like(Q)
    out = _settle(model, Q, P0, cfg.clu_steps, _dt_of(cfg, s), cfg.clu_gamma, 3, 32)
    assert out.shape == Q.shape
    assert np.all(np.isfinite(out))


def test_ladders_return_all_rungs_and_monotone_compute(float32_dynamics):
    P = _patterns(8, 32, seed=3)
    cfg = _quick_cfg()
    s = cfg.clu_s_frac * _median_nn_distance(P)
    dt = _dt_of(cfg, s)
    model = build_clu_memory(P, s, cfg)
    Q = noise_query(P, 0.3, jax.random.PRNGKey(1))
    true = np.arange(8)
    reads0 = _settle(model, np.asarray(Q), np.zeros_like(np.asarray(Q)),
                     cfg.clu_steps, dt, cfg.clu_gamma, 3, 32)
    rng = np.random.default_rng(0)

    gated = _retry_ladder(model, Q, reads0, P, true, cfg, dt, 0.99, "gated", rng)
    ung = _retry_ladder(model, Q, reads0, P, true, cfg, dt, 0.99, "ungated", rng)
    kick = _retry_ladder(model, Q, reads0, P, true, cfg, dt, 0.99, "kick", rng)
    ens = _ensemble_ladder(model, Q, reads0, P, true, cfg, dt, rng)
    ff = _feedforward_ladder(Q, P, true, cfg, rng)
    hop = _hopfield_ladder(Q, P, true, cfg)

    for ladder in (gated, ung, kick, ens, ff, hop):
        assert set(ladder.keys()) == set(cfg.retry_ladder)
        accs = [ladder[k][0] for k in sorted(ladder)]
        comps = [ladder[k][1] for k in sorted(ladder)]
        assert all(0.0 <= a <= 1.0 for a in accs)
        # compute is non-decreasing along the ladder
        assert all(comps[i] <= comps[i + 1] + 1e-9 for i in range(len(comps) - 1))

    # the gate is adaptive: gated spends <= ungated at the top of the ladder
    top = max(cfg.retry_ladder)
    assert gated[top][1] <= ung[top][1] + 1e-9
    # ungated pays the full (k+1)x budget
    np.testing.assert_allclose(ung[top][1], top + 1, atol=1e-6)


@pytest.mark.parametrize("query_type,level", [("mask", 0.5), ("noise", 0.3)])
def test_run_cell_has_six_lines_and_threshold_sweep(
    float32_dynamics, query_type, level
):
    P = _patterns(16, 32, seed=4)
    cfg = _quick_cfg()
    cell = run_cell(cfg, P, query_type, level, seed=0)
    expected = {
        "clu_gated", "ungated_all", "ensemble",
        "random_kick", "feedforward_nn", "hopfield_ksteps",
    }
    # w24: the erasure protocols carry the ML-optimal observed-dims-only NN floor
    if query_type in ("mask", "block"):
        expected |= {"feedforward_nn_masked"}
    assert set(cell["lines"].keys()) == expected
    assert set(cell["threshold_sweep"].keys()) == {"0.97", "1.00"}
    assert cell["query_type"] == query_type
    assert 0.0 <= cell["first_pass_acc"] <= 1.0


def test_config_round_trip_includes_retry_compute(tmp_path):
    from chlu.config import get_default_config, load_config, save_config

    cfg = get_default_config()
    cfg.experiment_retry_compute.retry_ladder = [0, 1, 3]
    cfg.experiment_retry_compute.main_threshold = 0.42
    p = tmp_path / "cfg.yaml"
    save_config(cfg, p)
    loaded = load_config(p)
    assert loaded.experiment_retry_compute.retry_ladder == [0, 1, 3]
    assert loaded.experiment_retry_compute.main_threshold == 0.42


# ---------------------------------------------------------------------------
# w24 — the AMBIGUITY regimes (block occlusion, crowded store) + headroom gate
# ---------------------------------------------------------------------------


def test_block_query_erases_a_contiguous_region_of_the_right_size():
    """The block is CONTIGUOUS (this is the whole point vs iid dropout) and covers
    ~frac of the pattern; survivors are rescaled by 1/(1-frac_eff)."""
    X = jnp.ones((8, 64))  # 8x8 image, so the 2-D square-block branch is taken
    Q = np.asarray(block_query(X, 0.5, jax.random.PRNGKey(0)))
    zeroed = Q == 0.0
    frac = zeroed.mean()
    assert 0.3 <= frac <= 0.7  # nominal 0.5, quantised to a 6x6 block (36/64)
    # survivors are scaled by 1/(1-frac_eff), constant across the pattern
    surv = Q[~zeroed]
    np.testing.assert_allclose(surv, surv[0], rtol=1e-5)
    # contiguity: per row, the zeroed columns form ONE run
    for r in range(8):
        img = zeroed[r].reshape(8, 8)
        for row in img:
            idx = np.flatnonzero(row)
            if idx.size:
                assert idx.max() - idx.min() + 1 == idx.size


def test_block_query_is_more_correlated_than_iid_dropout():
    """Sanity for the design constraint: at equal erased fraction, the block mask
    has far fewer 0/1 transitions along the vector than iid dropout."""
    from chlu.experiments.exp_hopfield_capacity import dropout_query

    X = jnp.ones((4, 64))
    blk = np.asarray(block_query(X, 0.5, jax.random.PRNGKey(0))) == 0.0
    iid = np.asarray(dropout_query(X, 0.5, jax.random.PRNGKey(0))) == 0.0
    assert np.abs(np.diff(blk, axis=1)).sum() < np.abs(np.diff(iid, axis=1)).sum()


def test_crowded_store_is_tighter_than_iid_and_slack_falls():
    """The crowded store is a nearest-neighbour cluster ⇒ smaller median spacing
    ⇒ smaller packing slack (the R-CROWD design target)."""
    from chlu.experiments.exp_hopfield_capacity import _median_nn_distance

    rng = np.random.default_rng(0)
    pool = jnp.asarray(rng.normal(size=(200, 16)).astype(np.float32))
    cfg = _quick_cfg()
    iid = select_store(pool, 16, "iid", cfg)
    crowd = select_store(pool, 16, "crowded", cfg)
    d_iid = float(_median_nn_distance(iid))
    d_crowd = float(_median_nn_distance(crowd))
    assert d_crowd < d_iid
    Q = np.asarray(crowd) + 0.1
    s_iid, _, _ = packing_slack(iid, np.asarray(iid) + 0.1, 0.3 * d_iid)
    s_crowd, _, _ = packing_slack(crowd, Q, 0.3 * d_crowd)
    assert s_crowd <= s_iid + 1e-9
    with pytest.raises(ValueError):
        select_store(pool, 8, "nonsense", cfg)


def test_parse_regimes_and_levels_default_to_the_w23_grid():
    cfg = _quick_cfg()
    cfg.regimes = []
    cfg.query_types = ["mask", "noise"]
    assert parse_regimes(cfg) == [("iid", "mask"), ("iid", "noise")]
    assert levels_for(cfg, "iid", "mask") == cfg.mask_fracs
    assert levels_for(cfg, "iid", "noise") == cfg.noise_levels

    cfg.regimes = ["iid:block", "crowded:mask"]
    assert parse_regimes(cfg) == [("iid", "block"), ("crowded", "mask")]
    assert levels_for(cfg, "iid", "block") == cfg.block_fracs
    assert levels_for(cfg, "crowded", "mask") == cfg.crowd_mask_fracs


def test_headroom_probe_reports_the_gate(float32_dynamics):
    P = _patterns(16, 64, seed=7)
    cfg = _quick_cfg()
    r = headroom_probe(cfg, P, "block", 0.5, seed=0, store_mode="iid")
    assert r["regime"] == "iid:block"
    assert 0.0 <= r["first_pass_acc"] <= 1.0
    assert 0.0 <= r["nn_floor"] <= 1.0
    lo, hi = cfg.headroom_band
    assert r["first_pass_in_band"] == (lo <= r["first_pass_acc"] <= hi)
    assert r["nn_off_ceiling"] == (r["nn_floor"] < cfg.headroom_nn_ceiling)
    assert r["passed"] == (r["first_pass_in_band"] and r["nn_off_ceiling"])


def test_run_cell_block_regime_and_sweep_switch(float32_dynamics):
    P = _patterns(16, 64, seed=8)
    cfg = _quick_cfg()
    cell = run_cell(cfg, P, "block", 0.5, seed=0, store_mode="iid", with_sweep=False)
    assert cell["regime"] == "iid:block"
    assert cell["threshold_sweep"] == {}
    assert cell["packing_slack"] > 0.0
    assert len(cell["lines"]) == 7  # 6 + the w24 masked-NN (ML-optimal) floor
    assert 0.0 <= cell["nn_floor_masked"] <= 1.0


def test_aggregate_cells_averages_over_seeds_and_scores_the_nn_margin():
    def _cell(seed, g, f):
        return {
            "dataset": "d", "regime": "iid:block", "M": 8, "level": 0.5,
            "seed": seed, "first_pass_acc": 0.5, "packing_slack": 1.0,
            "lines": {
                "clu_gated": {"0": {"acc": 0.5, "compute_mult": 1.0},
                              "1": {"acc": g, "compute_mult": 1.5}},
                "feedforward_nn": {"0": {"acc": f, "compute_mult": 1.0},
                                   "1": {"acc": f, "compute_mult": 2.0}},
            },
        }

    agg = aggregate_cells([_cell(0, 0.8, 0.6), _cell(1, 0.6, 0.6)])
    assert len(agg) == 1
    a = agg[0]
    assert a["n_seeds"] == 2
    np.testing.assert_allclose(a["lines"]["clu_gated"]["1"]["acc"], 0.7)
    np.testing.assert_allclose(a["lines"]["clu_gated"]["1"]["acc_std"], 0.1)
    # per-seed margins are +20pp and 0pp -> mean 10, sd 10
    np.testing.assert_allclose(a["margin_vs_nn_pp"], 10.0)
    np.testing.assert_allclose(a["margin_vs_nn_pp_std"], 10.0)


def test_config_round_trip_includes_the_w24_ambiguity_knobs(tmp_path):
    from chlu.config import get_default_config, load_config, save_config

    cfg = get_default_config()
    cfg.experiment_retry_compute.regimes = ["iid:block", "crowded:mask"]
    cfg.experiment_retry_compute.block_fracs = [0.42]
    cfg.experiment_retry_compute.crowd_mask_fracs = [0.11]
    cfg.experiment_retry_compute.headroom_band = [0.4, 0.8]
    cfg.experiment_retry_compute.n_seeds = 3
    p = tmp_path / "cfg.yaml"
    save_config(cfg, p)
    loaded = load_config(p).experiment_retry_compute
    assert loaded.regimes == ["iid:block", "crowded:mask"]
    assert loaded.block_fracs == [0.42]
    assert loaded.crowd_mask_fracs == [0.11]
    assert loaded.headroom_band == [0.4, 0.8]
    assert loaded.n_seeds == 3


def test_default_config_reproduces_the_w23_grid():
    """Regression guard: the w24 knobs must NOT change the shipped default."""
    cfg = get_default_config().experiment_retry_compute
    assert cfg.regimes == []
    assert parse_regimes(cfg) == [("iid", "mask"), ("iid", "noise")]
    assert cfg.n_seeds == 1


# ---------------------------------------------------------------------------
# w24 gate-iteration-2 amendments: rescale switch, store contraction, the
# unit-corrected packing slack, and the ML-optimal masked-NN floor
# ---------------------------------------------------------------------------


def test_block_rescale_switch_only_changes_the_survivor_scale():
    """rescale=False is the partial-key reading: identical support, unamplified."""
    X = jnp.asarray(np.random.default_rng(0).random((4, 64)).astype(np.float32)) + 0.1
    k = jax.random.PRNGKey(3)
    a = np.asarray(block_query(X, 0.5, k, rescale=True))
    b = np.asarray(block_query(X, 0.5, k, rescale=False))
    np.testing.assert_array_equal(a == 0.0, b == 0.0)  # same erased support
    ratio = a[b != 0.0] / b[b != 0.0]
    np.testing.assert_allclose(ratio, ratio[0], rtol=1e-5)
    assert ratio[0] > 1.0
    # and make_query honours the config flag
    cfg = _quick_cfg()
    cfg.block_rescale = False
    np.testing.assert_allclose(
        np.asarray(make_query(X, "block", 0.5, k, cfg)), b, rtol=1e-6
    )


def test_crowd_rho_contracts_the_store_and_shrinks_median_nn():
    from chlu.experiments.exp_hopfield_capacity import _median_nn_distance

    rng = np.random.default_rng(1)
    pool = jnp.asarray(rng.normal(size=(200, 16)).astype(np.float32))
    cfg = _quick_cfg()
    cfg.crowd_rho = 1.0
    plain = select_store(pool, 16, "crowded", cfg)
    cfg.crowd_rho = 0.25
    tight = select_store(pool, 16, "crowded", cfg)
    # centroid preserved, spacing scaled by rho
    np.testing.assert_allclose(
        np.asarray(jnp.mean(plain, 0)), np.asarray(jnp.mean(tight, 0)), atol=1e-5
    )
    np.testing.assert_allclose(
        float(_median_nn_distance(tight)),
        0.25 * float(_median_nn_distance(plain)),
        rtol=1e-4,
    )


def test_packing_slack_uses_the_displacement_NORM_not_a_per_element_rms():
    """Regression guard for the w24 unit fix: a per-element sigma_q is sqrt(D) too
    small, which pins slack at the tautology 1/(3.1*clu_s_frac) for every store."""
    rng = np.random.default_rng(2)
    P = jnp.asarray(rng.normal(size=(32, 64)).astype(np.float32))
    Q = np.asarray(P) + rng.normal(size=(32, 64)).astype(np.float32)
    s = 0.3 * float(_median_nn_distance(P))
    slack, d_nn, sigma_q = packing_slack(P, Q, s)
    np.testing.assert_allclose(
        sigma_q, float(np.sqrt(np.mean(np.sum((Q - np.asarray(P)) ** 2, axis=1)))),
        rtol=1e-6,
    )
    assert sigma_q > s  # a unit-1 displacement per axis in D=64 dominates the well
    np.testing.assert_allclose(slack, d_nn / (3.1 * sigma_q), rtol=1e-6)
    assert abs(slack - 1.0 / (3.1 * 0.3)) > 1e-3  # NOT the degenerate value


def test_masked_nn_is_the_ml_rule_and_beats_naive_nn_under_erasure():
    """Observed-dims-only NN must be exact when the survivors are unambiguous, and
    at least as good as the naive full-vector NN under block erasure."""
    rng = np.random.default_rng(4)
    P = jnp.asarray(rng.normal(size=(24, 64)).astype(np.float32))
    key = jax.random.PRNGKey(5)
    cfg = _quick_cfg()
    Q = make_query(P, "block", 0.5, key, cfg)
    keep = erasure_mask(P, "block", 0.5, key, cfg)
    assert keep.dtype == bool and keep.shape == (24, 64)
    scale = survivor_scale("block", 0.5, cfg)
    idx = masked_nn_identity(Q, P, keep, scale)
    np.testing.assert_array_equal(idx, np.arange(24))  # exact on separable patterns
    naive = np.argmin(
        np.sum((np.asarray(Q)[:, None, :] - np.asarray(P)[None, :, :]) ** 2, -1), 1
    )
    assert (idx == np.arange(24)).mean() >= (naive == np.arange(24)).mean()
    assert erasure_mask(P, "noise", 0.3, key, cfg) is None
