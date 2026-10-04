"""Tests for the two competitor arms (``gen-baselines``).

⭐ **What these tests are FOR.**  The kNN arm is the comparator for two kill rules,
so the failure mode that matters here is not "the arm crashed" but **"the arm was
quietly hobbled"** — a store that grew past the budget, a C request clipped onto
the grid until it became satisfiable, a whitening that silently degraded, an OR
that lost its published asymmetry.  Each of those has a test below that fires.

⚠ §7.23: six test modules enable ``jax_enable_x64`` at **module import** and sort
before this one, so ``equinox`` builds float64 weights.  Every module under test
casts its input to its **own** parameter dtype (the ``gen_bridge`` exemplar), which
is why these pass in the full suite and not only in isolation.  ⛔ A float32-pinning
fixture is not the fix.
"""

from __future__ import annotations

import json

import jax
import numpy as np
import pytest

from chlu.config import get_default_config, load_config, save_config
from chlu.core.gen_request import (
    FactorPrototypes,
    _extrapolating_index,
    dense_request_arrays,
    normalise_value,
    requests_from_records,
    requests_from_rows,
)
from chlu.data.gen_splits import FactorGrid, build_c_invalid, validate_c_invalid
from chlu.eval.rivals import gen_knn as knn
from chlu.eval.rivals import gen_lace as lace

STORE_BYTES = 524_288      # gen-store-grouping/runs/shapes3d_recon/store_bytes.json
COUPLING_BYTES = 163_840   # gen_dream.coupling_bytes: K(K-1)/2 * M^2 * 4 = 10 * 64^2 * 4
TOTAL_BYTES = 688_128      # ⭐ the landscape = store + inter-well PMI coupling (Head, 2026-09-12)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def grid():
    return FactorGrid(
        names=("hue", "scale", "shape"),
        values=(
            np.array([0.0, 0.25, 0.5, 0.75], np.float64),
            np.array([1.0, 1.5, 2.0], np.float64),
            np.array([0.0, 1.0], np.float64),
        ),
        dataset="toy",
    )


@pytest.fixture
def toy_features(grid):
    """Additive-by-construction features, so a *correct* composition is recoverable."""
    rng = np.random.default_rng(0)
    d = 16
    basis = [rng.normal(size=(c, d)).astype(np.float32) for c in grid.cardinalities]
    rows = np.arange(grid.n_rows, dtype=np.int64)
    fi = grid.factor_indices_of(rows)
    X = np.zeros((rows.size, d), np.float32)
    for k in range(grid.n_factors):
        X += basis[k][fi[:, k]]
    return rows, X


# ---------------------------------------------------------------------------
# R8 — the request embedding, declared once
# ---------------------------------------------------------------------------


def test_valid_requests_name_each_factor_exactly_once(grid):
    reqs = requests_from_rows(grid, np.array([0, 5, 11]))
    for r in reqs:
        seen = [t["factor"] for t in r["request"]]
        assert len(seen) == len(set(seen)) == grid.n_factors
        assert r["kind"] == "valid"


def test_prototype_composition_recovers_an_additive_feature(grid, toy_features):
    """⭐ The composition rule is not arbitrary: on additive data it is exact."""
    rows, X = toy_features
    protos = FactorPrototypes.fit(grid, X, rows)
    reqs = requests_from_rows(grid, rows)
    E = protos.embed_many(reqs)
    assert np.max(np.abs(E - X)) < 1e-4


def test_out_of_range_terms_are_extrapolated_and_NEVER_clipped(grid, toy_features):
    """⛔ Clipping would turn an unsatisfiable request into a satisfiable one.

    This is the single most dangerous silent hobbling available to the kNN arm:
    a clipped C item *is* a valid request, so the arm would refuse nothing and
    still score well.
    """
    rows, X = toy_features
    protos = FactorPrototypes.fit(grid, X, rows)
    on_grid = protos.prototype(0, 0.0)
    far = protos.prototype(0, -1.0)  # 4 steps below the grid
    near = protos.prototype(0, -0.25)  # 1 step below
    assert not np.allclose(far, on_grid)
    # ⭐ monotone in how invalid the request is — that is what makes distance work
    assert np.linalg.norm(far - on_grid) > np.linalg.norm(near - on_grid)


def test_extrapolating_index_is_exact_on_grid_and_signed_off_grid(grid):
    v = grid.values[0]
    i, j, w = _extrapolating_index(v, 0.5)
    assert (i, j, w) == (2, 2, 0.0)
    _, _, w_below = _extrapolating_index(v, -0.25)
    _, _, w_above = _extrapolating_index(v, 1.0)
    assert w_below < 0.0 and w_above > 1.0


def test_duplicate_factor_request_contributes_BOTH_terms(grid, toy_features):
    """"Requested simultaneously" means both deviations enter the composition."""
    rows, X = toy_features
    protos = FactorPrototypes.fit(grid, X, rows)
    base = [{"factor": "hue", "factor_index": 0, "value": 0.0},
            {"factor": "scale", "factor_index": 1, "value": 1.0},
            {"factor": "shape", "factor_index": 2, "value": 0.0}]
    dup = base + [{"factor": "hue", "factor_index": 0, "value": 0.75}]
    e_base, e_dup = protos.embed(base), protos.embed(dup)
    expected = e_base + (protos.prototype(0, 0.75) - protos.mu_global)
    assert np.allclose(e_dup, expected, atol=1e-5)
    assert not np.allclose(e_dup, e_base)


def test_offgrid_report_fires_on_a_real_c_set(grid):
    """⭐ The guard the experiment asserts on: a C set with zero off-grid terms
    means ``out_of_range`` was silently clipped somewhere upstream."""
    payload = build_c_invalid(grid, n_per_kind=4, seed=0)
    validate_c_invalid(payload, grid)
    reqs = requests_from_records(payload["records"])
    rows = np.arange(grid.n_rows, dtype=np.int64)
    rng = np.random.default_rng(1)
    protos = FactorPrototypes.fit(grid, rng.normal(size=(rows.size, 8)).astype(np.float32), rows)
    rep = protos.offgrid_report(reqs)
    assert rep["n_offgrid_terms"] > 0
    assert rep["max_distance_outside_range"] > 0.0


def test_dense_request_arrays_carry_the_duplicate_in_the_extra_slot(grid):
    payload = build_c_invalid(grid, n_per_kind=3, seed=0)
    reqs = requests_from_records(payload["records"])
    dense = dense_request_arrays(grid, reqs)
    dup = np.asarray([r["kind"] == "duplicate_factor" for r in reqs])
    assert np.all(dense.extra_factor[dup] >= 0)
    assert np.all(dense.extra_factor[~dup] == -1)


def test_offgrid_value_normalises_outside_the_unit_interval(grid):
    v = np.asarray(grid.values[0])
    assert normalise_value(v, 0.0) == 0.0
    assert normalise_value(v, 0.75) == 1.0
    assert normalise_value(v, -0.25) < 0.0


# ---------------------------------------------------------------------------
# ⭐ The byte ledger — shrink to match, NEVER grow
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dep,expected", [("fp32", 64), ("fp16", 128), ("int8", 248)])
def test_entries_for_budget_matches_the_store_component_exactly(dep, expected):
    assert knn.entries_for_budget(STORE_BYTES, 2048, dep) == expected


@pytest.mark.parametrize("dep,expected", [("fp32", 84), ("fp16", 168), ("int8", 328)])
def test_entries_for_budget_at_the_CORRECTED_total(dep, expected):
    """⭐ The byte match is to store + coupling, not to the store alone.

    ⛔ Matching ``store_bytes`` alone under-resources the comparator by 24 % — and
    the kNN arm is what kill rules (ii) and (iv) read, so a weak one hands BOTH a
    false pass. Every deployment lands on the budget exactly, zero slack.
    """
    assert knn.entries_for_budget(TOTAL_BYTES, 2048, dep) == expected
    assert knn.byte_ledger(expected, 2048, dep, TOTAL_BYTES)["slack_bytes"] == 0


def test_the_coupling_component_is_derived_not_hand_typed():
    """``K(K-1)/2 * M^2 * 4`` at the shipped cell (K=5, M=64)."""
    k, m = 5, 64
    assert (k * (k - 1) // 2) * m * m * 4 == COUPLING_BYTES
    assert STORE_BYTES + COUPLING_BYTES == TOTAL_BYTES


def test_the_corrected_budget_does_not_widen_the_read_channel():
    """⭐ R20's 30-bit design constant is a CHANNEL fact and is untouched by a
    CAPACITY correction: the coupling shapes which codes are chosen, it is never
    transmitted to the decoder."""
    assert knn.bits_per_read(64, 5) == 30.0  # the CLU's K*log2(M)
    # the kNN's own channel does move, because its entry count moved — reported, not hidden
    assert knn.bits_per_read(84, 1) == pytest.approx(np.log2(84))


@pytest.mark.parametrize("dep", knn.DEPLOYMENTS)
def test_realised_ledger_never_exceeds_the_budget(dep):
    n = knn.entries_for_budget(STORE_BYTES, 2048, dep)
    led = knn.byte_ledger(n, 2048, dep, STORE_BYTES)
    assert led["within_budget"] and led["slack_bytes"] >= 0
    # ⭐ one more entry would BREAK the budget: the floor is tight, not slack
    assert not knn.byte_ledger(n + 1, 2048, dep, STORE_BYTES)["within_budget"]


def test_int8_overhead_is_inside_the_budget_not_beside_it():
    """The per-dimension scale/offset ships, so it is counted."""
    assert knn.deployment_overhead_bytes("int8", 2048) == 2 * 2048 * 4
    led = knn.byte_ledger(knn.entries_for_budget(STORE_BYTES, 2048, "int8"), 2048, "int8", STORE_BYTES)
    assert led["overhead_bytes"] > 0 and led["realised_bytes"] <= STORE_BYTES


def test_build_store_is_byte_matched_and_shrinks_rather_than_grows():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 64)).astype(np.float32)
    budget = 64 * 64 * 4  # 64 fp32 entries
    st = knn.build_store(X, budget, "fp32", seed=0, iters=3)
    assert st.n_entries == 64
    assert st.deployed_bytes() == budget
    assert st.info["byte_ledger"]["within_budget"]


def test_quantised_store_roundtrips_within_its_own_resolution():
    rng = np.random.default_rng(3)
    C = rng.normal(size=(32, 64)).astype(np.float32)
    q = knn.quantise(C, "int8")
    # ⚠ numpy 2 removed ``ndarray.ptp``; the function form is the portable one.
    err = np.max(np.abs(q.values() - C) / np.maximum(np.ptp(C, axis=0)[None, :], 1e-9))
    assert err < 0.01  # one 255th of the per-dimension range, plus rounding


def test_bits_per_read_is_the_retrieval_channel_width():
    assert knn.bits_per_read(64, 1) == 6.0
    assert knn.bits_per_read(64, 5) == 30.0  # ⭐ exactly the CLU's 30-bit channel


# ---------------------------------------------------------------------------
# kNN mechanics
# ---------------------------------------------------------------------------


def test_knn_read_does_not_take_the_query_as_an_argument():
    """⭐ The same structural property ``gen_store.read_from_codes`` has: the
    request embedding cannot reach the decoder, only retrieved store content can."""
    import inspect

    assert list(inspect.signature(knn.knn_read).parameters) == ["store_values_raw", "idx"]


def test_knn_search_finds_the_planted_neighbour():
    S = np.eye(8, dtype=np.float32) * 3.0
    Q = np.asarray([[0, 0, 3.1, 0, 0, 0, 0, 0]], np.float32)
    for metric in knn.METRICS:
        idx, d = knn.knn_search(S, Q, 1, metric)
        assert int(idx[0, 0]) == 2
        assert d[0, 0] >= 0.0


def test_normaliser_survives_dead_coordinates():
    """⚠ 71 of 2048 coordinates are constant post-ReLU on the shipped cache; a
    bare 1/std would divide by zero and poison every distance."""
    X = np.concatenate(
        [np.random.default_rng(0).normal(size=(50, 4)), np.zeros((50, 2))], axis=1
    ).astype(np.float32)
    out = knn.fit_normaliser(X, "zscore")(X)
    assert np.all(np.isfinite(out))


def test_threshold_uses_valid_data_only_and_refuses_an_empty_set():
    valid = np.linspace(0.0, 1.0, 101)
    assert knn.threshold_from_valid(valid, 0.95) == pytest.approx(0.95, abs=1e-6)
    with pytest.raises(ValueError):
        knn.threshold_from_valid(np.zeros(0), 0.95)


def test_tuning_grid_size_is_the_reported_budget():
    g = knn.tuning_grid((1, 3, 5), knn.METRICS, knn.NORMALISERS, knn.DEPLOYMENTS)
    assert len(g) == 36 == len({tuple(sorted(t.items())) for t in g})


# ---------------------------------------------------------------------------
# ⭐ LACE — whitening is mandatory, and the published constants are the published ones
# ---------------------------------------------------------------------------


def test_whitening_makes_the_prior_exact():
    """⭐ The identity that licenses LACE's ODE drift to omit the prior score."""
    rng = np.random.default_rng(0)
    A = rng.normal(size=(24, 24))
    X = (rng.normal(size=(4000, 24)) @ A + 5.0).astype(np.float32)
    wh = lace.fit_whitening(X, 1e-6)
    chk = lace.whitening_check(wh, X)
    assert chk["mean_abs_diag_minus_one"] < 0.05
    assert chk["mean_abs_offdiag"] < 0.05
    assert chk["roundtrip_max_abs_err"] < 1e-2


def test_whitening_refuses_to_degrade_silently_above_the_D_limit():
    """⛔ R14.3: above the limit LACE gets a DECLARED PCA projection, never a
    silent drop to channel-only whitening."""
    with pytest.raises(ValueError, match="DECLARED PCA"):
        lace.fit_whitening(np.zeros((4, lace.WHITENING_D_LIMIT + 1), np.float32))


def test_shrinkage_is_bounded_and_regularises_a_singular_covariance():
    """The shipped cache has 71 dead (constant) coordinates -> singular Sigma."""
    rng = np.random.default_rng(1)
    X = np.concatenate([rng.normal(size=(200, 10)), np.zeros((200, 4))], axis=1).astype(np.float32)
    wh = lace.fit_whitening(X, "ledoit_wolf")
    assert 0.0 <= wh.meta["shrinkage_intensity"] <= 1.0
    assert np.all(np.isfinite(wh.W)) and wh.meta["eigenvalue_min"] > 0.0


def test_dense_embedder_reproduces_the_published_layer_widths():
    """``PAPER:`` App. A.3 "Linear 384 / 256 / 128 / #logits" == ``OFFICIAL
    IMPLEMENTATION:`` ``np.linspace(512, 128, 4).astype(int)``."""
    assert lace.dense_embedder_dims(512, 128, 4) == [512, 384, 256, 128]


def test_published_constants_are_the_shipped_ones_not_the_argparse_defaults():
    """⛔ R9.6's constant traps, pinned so a future edit cannot re-introduce them."""
    assert lace.LACE_N_EPOCHS == 50      # ⛔ not the appendix's 100, not argparse's 200
    assert lace.LACE_BATCH == 128        # ⛔ not argparse's 64
    assert lace.LACE_SGLD_STD == 1e-2 and lace.LACE_SGLD_LR == 1e-2
    assert lace.LACE_MAX_OPERANDS == 10
    assert (lace.LACE_BETA_0, lace.LACE_BETA_1, lace.LACE_T) == (0.1, 20.0, 1.0)


def test_the_published_langevin_is_200x_colder_than_its_own_target():
    """``DERIVED:`` gen-lace-spec F2 — the reason R9.4 mandates two temperatures."""
    t_eff = lace.LACE_SGLD_STD**2 / (2 * lace.LACE_SGLD_LR)
    assert t_eff == pytest.approx(0.005)
    assert lace.LACE_SGLD_STD_CORRECTED == pytest.approx(np.sqrt(2 * lace.LACE_SGLD_LR))
    assert lace.LACE_SGLD_STD_CORRECTED / lace.LACE_SGLD_STD == pytest.approx(np.sqrt(200.0), rel=1e-6)


def test_or_is_non_commutative_and_caps_at_ten_operands():
    """``DERIVED:`` gen-lace-spec F5 — ``cof`` multiplies the LEFT operand only."""
    import jax.numpy as jnp

    neg = [jnp.asarray(-1.0), jnp.asarray(-3.0)]
    assert float(lace.logical_comb(neg, "0*1")) != pytest.approx(float(lace.logical_comb(neg, "1*0")))
    assert float(lace.logical_comb(neg, "0+1")) == pytest.approx(-4.0, abs=1e-5)
    with pytest.raises(ValueError, match="structural limit"):
        lace.logical_comb([jnp.asarray(-1.0)] * 11, "+".join(str(i) for i in range(11)))


def test_lace_energy_penalises_a_conflicting_duplicate_request(grid):
    """⭐ The mechanism behind LACE's refusal: AND is a SUM, so two incompatible
    terms on one factor leave an irreducible residual."""
    key = jax.random.PRNGKey(0)
    model = lace.LaceClassifier(12, grid.cardinalities, key, up_dim=6, depth=3)
    base = [{"factor": "hue", "factor_index": 0, "value": 0.0},
            {"factor": "scale", "factor_index": 1, "value": 1.0},
            {"factor": "shape", "factor_index": 2, "value": 0.0}]
    dup = base + [{"factor": "hue", "factor_index": 0, "value": 0.75}]
    req = dense_request_arrays(grid, [{"request": base}, {"request": dup}])
    U = np.zeros((2, 12), np.float32)
    e = lace.energy_of(model, U, req, "categorical", lace.LACE_SIGMA)
    assert e[1] > e[0]  # the duplicate term can only ADD energy


def test_lace_energy_is_higher_for_an_out_of_range_continuous_request(grid):
    """⚠ The categorical head has a CLOSED vocabulary and cannot express an
    off-grid value; the continuous head is the form that sees the violation."""
    key = jax.random.PRNGKey(1)
    model = lace.LaceClassifier(12, grid.cardinalities, key, up_dim=6, depth=3)
    ok = [{"factor": "hue", "factor_index": 0, "value": 0.0},
          {"factor": "scale", "factor_index": 1, "value": 1.0},
          {"factor": "shape", "factor_index": 2, "value": 0.0}]
    oor = [dict(ok[0], value=-3.0), ok[1], ok[2]]
    req = dense_request_arrays(grid, [{"request": ok}, {"request": oor}])
    U = np.zeros((2, 12), np.float32)
    e_cont = lace.energy_of(model, U, req, "continuous", lace.LACE_SIGMA)
    e_cat = lace.energy_of(model, U, req, "categorical", lace.LACE_SIGMA)
    assert e_cont[1] > e_cont[0]
    assert e_cat[1] == pytest.approx(e_cat[0], abs=1e-6)  # ⛔ closed vocabulary, blind


def test_both_samplers_run_and_reduce_the_conditional_energy(grid):
    key = jax.random.PRNGKey(2)
    model = lace.LaceClassifier(12, grid.cardinalities, key, up_dim=6, depth=3)
    reqs = requests_from_rows(grid, np.array([0, 7, 13]))
    req = dense_request_arrays(grid, reqs)
    U0 = np.asarray(jax.random.normal(jax.random.PRNGKey(3), (req.n, 12)), np.float32)
    e0 = lace.energy_of(model, U0, req, "both", lace.LACE_SIGMA)
    for name, kw in (("ld", dict(steps=40, std=lace.LACE_SGLD_STD_CORRECTED)),):
        U = lace.sample_ld(model, req, jax.random.PRNGKey(4), form="both", **kw)
        assert np.all(np.isfinite(U)), name
    U_ode = lace.sample_ode(model, req, jax.random.PRNGKey(5), form="both")
    assert np.all(np.isfinite(U_ode))
    e_ode = lace.energy_of(model, U_ode, req, "both", lace.LACE_SIGMA)
    assert e_ode.mean() < e0.mean()  # the ODE descends the composed energy


def test_lace_samplers_cover_both_temperatures_R9_4():
    assert set(lace.SAMPLERS) == {"ode", "ld_published", "ld_corrected"}
    assert lace.SAMPLERS["ld_published"]["std"] == lace.LACE_SGLD_STD
    assert lace.SAMPLERS["ld_corrected"]["std"] == lace.LACE_SGLD_STD_CORRECTED


def test_lace_tuning_grid_matches_the_knn_budget():
    """⭐ The Head's constraint: the tuning-trial budget is equal across arms."""
    n_lace = len(lace.tuning_grid((0.03, 0.1, 0.3, 1.0), lace.ENERGY_FORMS,
                                  ("ledoit_wolf", "1e-3", "1e-2")))
    n_knn = len(knn.tuning_grid((1, 3, 5), knn.METRICS, knn.NORMALISERS, knn.DEPLOYMENTS))
    assert n_lace == n_knn == 36


# ---------------------------------------------------------------------------
# experiment plumbing / config
# ---------------------------------------------------------------------------


def test_config_roundtrips_and_selects_the_WINNING_decoder(tmp_path):
    """⛔ The trap: ``dec_recipe`` still defaults to the SUPERSEDED ``baseline``."""
    c = get_default_config()
    assert c.gen_baselines.decoder_source == "decoder_lift_v1"
    assert c.gen_baselines.assert_decoder_hash is True
    assert c.gen_bridge.dec_recipe == "baseline"  # ⛔ deliberately left alone
    p = tmp_path / "cfg.yaml"
    save_config(c, p)
    c2 = load_config(p)
    assert c2.gen_baselines.lace_n_epochs == 50
    assert c2.gen_baselines.knn_deployments == ["fp32", "fp16", "int8"]
    assert c2.gen_baselines.n_fit + c2.gen_baselines.n_val == c2.gen_store.n_store_train


def test_arms_module_refuses_a_stage_that_is_not_its_own():
    from chlu.experiments.exp_gen_baselines import main

    assert main(["--stage", "store", "--seed", "0"]) == 2


def test_a_missing_byte_count_stops_the_run_rather_than_guessing(tmp_path):
    """⛔ An unmatched kNN arm is worthless: no byte count means stop and report."""
    from chlu.experiments.exp_gen_baselines import resolve_store_bytes

    c = get_default_config()
    c.gen_baselines.store_runs_root = str(tmp_path)
    with pytest.raises(FileNotFoundError, match="byte-matched"):
        resolve_store_bytes(c, "shapes3d", "recon", lambda *a: None)


def test_resolve_store_bytes_derives_the_coupling_from_the_plans_own_K_and_M(tmp_path):
    """⭐ Both components mechanical: store from ``store_bytes.json``, coupling from
    ``store_plan.json``'s own ``K``/``n_modes``. ⛔ Neither hand-typed.

    ⚠ **Amended for R26 ruling 3:** the coupling is no longer part of the deployed
    store, so reaching it now takes an explicit opt-in *and* an explicit expectation.
    ⛔ The derivation itself is unchanged — that is what this test still pins.
    """
    from chlu.experiments.exp_gen_baselines import resolve_store_bytes

    d = tmp_path / "shapes3d_recon"
    d.mkdir(parents=True)
    (d / "store_bytes.json").write_text(json.dumps(
        {"store_bytes": STORE_BYTES, "n_wells": 5, "n_modes": 64}))
    (d / "store_plan.json").write_text(json.dumps({"K": 5, "n_modes": 64}))
    c = get_default_config()
    c.gen_baselines.store_runs_root = str(tmp_path)
    c.gen_baselines.include_coupling_bytes = True          # ⛔ opt-in, never inherited
    c.gen_baselines.expect_byte_target_bytes = TOTAL_BYTES
    t = resolve_store_bytes(c, "shapes3d", "recon", lambda *a: None)
    assert t["store_bytes"] == STORE_BYTES
    assert t["coupling_bytes"] == COUPLING_BYTES
    assert t["total_bytes"] == TOTAL_BYTES
    # ⛔ and the default excludes it explicitly, never silently
    c.gen_baselines.include_coupling_bytes = False
    c.gen_baselines.expect_byte_target_bytes = STORE_BYTES
    t2 = resolve_store_bytes(c, "shapes3d", "recon", lambda *a: None)
    assert t2["coupling_bytes"] == 0 and t2["total_bytes"] == STORE_BYTES


def test_a_missing_decoder_lift_winner_refuses_to_fall_back(tmp_path):
    """⛔ Falling back to the bridge decoder would look normal in the logs and be void."""
    from chlu.experiments.exp_gen_baselines import load_shared_decoder

    c = get_default_config()
    c.gen_baselines.decoder_lift_dir = str(tmp_path)
    with pytest.raises(FileNotFoundError, match="FAILED the gate"):
        load_shared_decoder(c, tmp_path, {"decoder_param_hash": "x"}, 0, lambda *a: None)


def test_refusal_metrics_report_the_primary_and_both_companions():
    from chlu.experiments.exp_gen_baselines import refusal_metrics

    rng = np.random.default_rng(0)
    scores = {"A": rng.normal(0, 1, 200), "B": rng.normal(0.5, 1, 72),
              "MS": rng.normal(0.2, 1, 50), "C": rng.normal(3.0, 1, 72)}
    kinds = ["out_of_range"] * 36 + ["duplicate_factor"] * 36
    out = refusal_metrics(scores, 0.95, kinds)
    assert out["auroc_c_vs_b"] > 0.9  # ⭐ PRIMARY
    assert "auroc_c_vs_a" in out and "auroc_c_vs_a_union_b" in out
    assert set(out["auroc_c_vs_b_by_kind"]) == {"out_of_range", "duplicate_factor"}
    # ⛔ the threshold comes from A only
    assert out["threshold"]["value"] == pytest.approx(np.quantile(scores["A"], 0.95))


def test_aggregate_reports_a_missing_seed_as_absent_never_as_zero(tmp_path):
    """⭐ Protocol §6: ⛔ never a zero, never a silent 1-seed number presented as 3."""
    from chlu.experiments.exp_gen_baselines import aggregate_runs

    d = tmp_path / "seed_0"
    d.mkdir()
    man = {
        "dataset": "shapes3d", "pretext": "recon", "condition": "recomb2element", "seed": 0,
        "provenance": {"decoder": {"recipe": "v1_capacity_schedule"}},
        "ceilings": {"store_ceiling_balanced_accuracy": 0.6866,
                     "decoder_ceiling": {"composition_A": {"aggregate": 0.95},
                                         "in_corner_discrimination": {"balanced_accuracy": 0.94,
                                                                      "auroc": 0.99}}},
        "arms": {}, "NO_KILL_RULE_VERDICT": "x",
    }
    (d / "baselines_manifest.json").write_text(json.dumps(man))
    agg = aggregate_runs([d, tmp_path / "seed_1"], log=lambda *a: None)
    assert agg["n_seeds"] == 1 and agg["seeds"] == [0]


# ---------------------------------------------------------------------------
# ⭐ A7 — the C set's per-kind VALID CONTROL, wired into both arms
#
# ⛔ The defect these pin: all 36 ``duplicate_factor`` C records carry one term MORE
# than the valid pool, so a bare term counter scores AUROC 1.0000 on that half
# (`gen-c-set-a7` §1).  Scoring that kind against ``a7_redundant`` — same term count,
# same ordering, same repeated-factor pattern, still satisfiable — removes it.
# ⛔ ``out_of_range`` declares ``base`` and must therefore be UNCHANGED: that is the
# regression which catches a wiring change that moved a rival.
# ---------------------------------------------------------------------------


def _a7_style_record(grid, row: int, rec_id: str = "toy-a7-000") -> dict:
    """One valid request carrying a redundant (same-value) term, A7's construction."""
    req = requests_from_rows(grid, np.array([row]))[0]
    terms = [dict(t) for t in req["request"]]
    return {"id": rec_id, "kind": "valid_redundant", "row": int(row),
            "redundant_factor": grid.names[0], "request": terms + [dict(terms[0])]}


def test_a7_records_load_verbatim_and_carry_their_row(grid, tmp_path):
    from chlu.core.gen_request import (
        control_map_of,
        load_query_set_registry,
        load_registered_control_requests,
    )

    recs = [_a7_style_record(grid, r, f"toy-a7-{r:03d}") for r in (0, 3, 7)]
    (tmp_path / "a7_redundant.json").write_text(json.dumps({"records": recs}))
    reg = {"sets": {
        "base": {"role": "valid", "derived_from": "factor_index.npz"},
        "a7_redundant": {"role": "valid", "file": "a7_redundant.json"},
        "c_invalid": {"role": "invalid", "file": "c_invalid.json",
                      "controls": {"duplicate_factor": "a7_redundant",
                                   "out_of_range": "base"}},
    }}
    (tmp_path / "query_sets.json").write_text(json.dumps(reg))

    registry = load_query_set_registry(tmp_path / "query_sets.json")
    cmap = control_map_of(registry)
    # ⛔ the map is READ from disk, never hardcoded by the arm that is about to be scored
    assert cmap == {"duplicate_factor": "a7_redundant", "out_of_range": "base"}
    loaded = load_registered_control_requests(tmp_path, registry, sorted(set(cmap.values())))
    # ⛔ ``base`` is derived, held by the arm as its own B pool, and never duplicated here
    assert set(loaded) == {"a7_redundant"}
    a7 = loaded["a7_redundant"]
    assert [r["id"] for r in a7] == ["toy-a7-000", "toy-a7-003", "toy-a7-007"]
    assert [r["row"] for r in a7] == [0, 3, 7]          # ⭐ so the caller can assert no leak
    assert all(len(r["request"]) == grid.n_factors + 1 for r in a7)


def test_a_registered_control_missing_from_disk_is_not_a_silent_skip(tmp_path):
    """⛔ 'declared not-run reported as null' in code form (R26.7) — it must RAISE."""
    from chlu.core.gen_request import load_registered_control_requests

    reg = {"sets": {"a7_redundant": {"file": "a7_redundant.json"}}}
    with pytest.raises(FileNotFoundError, match="must not be faked"):
        load_registered_control_requests(tmp_path, reg, ["a7_redundant"])


def test_a7_embedding_double_counts_the_redundant_term_BY_THE_DECLARED_RULE(
        grid, toy_features):
    """⭐ R8 says *both* terms enter the composition, so a repeated term is added twice.

    ⛔ Not a defect to patch silently: the same rule produces the C ``duplicate_factor``
    embedding, so the control and the thing it controls are displaced the same way.
    The dedup set measures how much of a controlled number is this doubling.
    """
    from chlu.core.gen_request import deduplicate_terms, requests_from_valid_records

    rows, X = toy_features
    protos = FactorPrototypes.fit(grid, X, rows)
    a7 = requests_from_valid_records([_a7_style_record(grid, 5)])
    plain = requests_from_rows(grid, np.array([5]))
    e_a7, e_plain = protos.embed_many(a7)[0], protos.embed_many(plain)[0]
    dev = protos.prototype(0, float(plain[0]["request"][0]["value"])) - protos.mu_global
    assert np.allclose(e_a7 - e_plain, dev, atol=1e-5)
    # ⭐ and the declared sensitivity removes exactly that, nothing else
    dd = protos.embed_many(deduplicate_terms(a7))[0]
    assert np.allclose(dd, e_plain, atol=1e-5)


def test_out_of_range_is_A7_CLEAN_and_duplicate_factor_is_the_only_kind_that_moves():
    """⭐ THE regression check: A7 leaves ``out_of_range``'s control at the valid pool.

    ⛔ If this ever fails, the wiring changed a number that R26/R27 say cannot move,
    and the published 0.8045 / 0.4694 ``out_of_range`` figures are no longer comparable.
    """
    from chlu.experiments.exp_gen_baselines import refusal_metrics

    rng = np.random.default_rng(3)
    scores = {"A": rng.normal(0, 1, 200), "B": rng.normal(0.5, 1, 72),
              "MS": rng.normal(0.2, 1, 50), "C": rng.normal(3.0, 1, 72),
              # a control that is systematically FARTHER from the store than B
              "A7": rng.normal(2.0, 1, 72)}
    kinds = ["out_of_range"] * 36 + ["duplicate_factor"] * 36
    cmap = {"duplicate_factor": "a7_redundant", "out_of_range": "base"}
    out = refusal_metrics(scores, 0.95, kinds, cmap)
    ctrl = out["a7_controlled"]["auroc_by_kind"]
    raw = out["auroc_c_vs_b_by_kind"]
    # ⭐ bit-identical, not merely close
    assert ctrl["out_of_range"]["auroc"] == raw["out_of_range"]["auroc"]
    assert ctrl["out_of_range"]["control"] == "base"
    assert ctrl["duplicate_factor"]["control"] == "a7_redundant"
    assert ctrl["duplicate_factor"]["auroc"] != raw["duplicate_factor"]["auroc"]
    # ⛔ both columns survive side by side; neither overwrites the other
    assert set(out["a7_controlled"]["auroc_by_kind_PER_CONTROL"]) == {"B", "A7"}


def test_a_kind_with_no_declared_control_is_reported_as_an_error_not_scored():
    """⛔ The inheritance mechanism: a NEW C kind must not slip through uncontrolled."""
    from chlu.experiments.exp_gen_baselines import refusal_metrics

    rng = np.random.default_rng(4)
    scores = {"A": rng.normal(0, 1, 50), "B": rng.normal(0, 1, 20),
              "C": rng.normal(1.0, 1, 20), "A7": rng.normal(0.5, 1, 20)}
    kinds = ["duplicate_factor"] * 10 + ["brand_new_kind"] * 10
    out = refusal_metrics(scores, 0.95, kinds,
                          {"duplicate_factor": "a7_redundant"})
    assert "error" in out["a7_controlled"]["auroc_by_kind"]["brand_new_kind"]
    assert "auroc" in out["a7_controlled"]["auroc_by_kind"]["duplicate_factor"]


def test_the_pooled_controlled_number_is_a_weighted_mean_and_says_so():
    """⛔ Pooling two kinds with DIFFERENT controls into one ranking would re-create
    the giveaway A7 removes, so the companion is a weighted mean of the per-kind
    numbers — and it is labelled COMPANION (R27.4)."""
    from chlu.experiments.exp_gen_baselines import refusal_metrics

    rng = np.random.default_rng(5)
    scores = {"A": rng.normal(0, 1, 100), "B": rng.normal(0, 1, 72),
              "C": rng.normal(1.5, 1, 72), "A7": rng.normal(0.7, 1, 72)}
    kinds = ["out_of_range"] * 36 + ["duplicate_factor"] * 36
    out = refusal_metrics(scores, 0.95, kinds,
                          {"duplicate_factor": "a7_redundant", "out_of_range": "base"})
    by = out["a7_controlled"]["auroc_by_kind"]
    pooled = out["a7_controlled"]["pooled_COMPANION"]
    assert pooled["value"] == pytest.approx(
        0.5 * (by["out_of_range"]["auroc"] + by["duplicate_factor"]["auroc"]))
    assert "COMPANION" in pooled["definition"]


def test_per_request_scores_are_persisted_so_a_re_score_is_a_JOIN_not_a_re_run():
    """⭐ The acceptance item. ⛔ Their absence is what blocked the first A7 re-score:
    with only ``score_summary`` on disk there is nothing to re-score."""
    from chlu.experiments.exp_gen_baselines import refusal_metrics

    rng = np.random.default_rng(6)
    scores = {"A": rng.normal(0, 1, 5), "B": rng.normal(0, 1, 4),
              "C": rng.normal(1.0, 1, 3), "A7": rng.normal(0.5, 1, 3)}
    ids = {"A": [f"row-{i}" for i in range(5)], "B": [f"row-{i}" for i in range(4)],
           "C": [f"c-{i}" for i in range(3)], "A7": [f"a7-{i}" for i in range(3)]}
    kinds = ["duplicate_factor"] * 2 + ["out_of_range"]
    out = refusal_metrics(scores, 0.95, kinds, {"duplicate_factor": "a7_redundant",
                                                "out_of_range": "base"}, ids)
    tab = out["per_request_scores"]
    for k in ("A", "B", "C", "A7"):
        assert tab[k]["ids"] == ids[k]
        assert np.allclose(tab[k]["scores"], scores[k])
    assert tab["_kinds_C"] == kinds  # ⭐ the join key for a per-kind re-score


def test_extracting_the_per_request_tables_empties_them_out_of_the_manifest():
    from chlu.experiments.exp_gen_baselines import _extract_per_request_scores

    arms = {"knn": {"variants": {"knn": {"refusal": {"auroc_c_vs_b": 0.5,
                                                     "per_request_scores": {"C": {}}}}}},
            "lace": {"samplers": {"ode": {"refusal": {"auroc_c_vs_b": 0.6,
                                                      "per_request_scores": {"C": {}}}}}}}
    tab = _extract_per_request_scores(arms)
    assert set(tab) == {"knn", "lace"}
    assert "per_request_scores" not in arms["knn"]["variants"]["knn"]["refusal"]
    assert "per_request_scores" not in arms["lace"]["samplers"]["ode"]["refusal"]


# ---------------------------------------------------------------------------
# ⛔ R26 ruling 3 — the byte budget reverts to the deployed store
# ---------------------------------------------------------------------------


def test_the_default_byte_target_is_the_DEPLOYED_store_and_is_asserted(tmp_path):
    """⛔ R26.3 took the coupling field out of the deployed store: 688,128 -> 524,288.

    The default must be the deployed budget, the realised total must be ASSERTED,
    and the historical budget must still be reachable DELIBERATELY.
    """
    from chlu.experiments.exp_gen_baselines import resolve_store_bytes

    d = tmp_path / "shapes3d_recon"
    d.mkdir(parents=True)
    (d / "store_bytes.json").write_text(json.dumps(
        {"store_bytes": STORE_BYTES, "n_wells": 5, "n_modes": 64}))
    (d / "store_plan.json").write_text(json.dumps({"K": 5, "n_modes": 64}))
    c = get_default_config()
    assert c.gen_baselines.include_coupling_bytes is False       # ⛔ the flipped default
    assert c.gen_baselines.expect_byte_target_bytes == STORE_BYTES
    c.gen_baselines.store_runs_root = str(tmp_path)
    t = resolve_store_bytes(c, "shapes3d", "recon", lambda *a: None)
    assert t["total_bytes"] == STORE_BYTES and t["coupling_bytes"] == 0
    assert t["expected_total_bytes"] == STORE_BYTES
    assert t["superseded_by_R26_ruling_3"] is False

    # ⛔ inheriting the old flag without saying so now STOPS the run
    c.gen_baselines.include_coupling_bytes = True
    with pytest.raises(AssertionError, match="R26 ruling 3"):
        resolve_store_bytes(c, "shapes3d", "recon", lambda *a: None)

    # ⭐ the historical 688,128 B run still reproduces — deliberately
    c.gen_baselines.expect_byte_target_bytes = None
    t2 = resolve_store_bytes(c, "shapes3d", "recon", lambda *a: None)
    assert t2["total_bytes"] == TOTAL_BYTES and t2["superseded_by_R26_ruling_3"] is True


def test_the_cli_reaches_the_historical_budget_only_by_asking_for_it():
    from chlu.experiments.exp_gen_baselines import _config_from_args, build_arg_parser

    p = build_arg_parser()
    g = _config_from_args(p.parse_args([])).gen_baselines
    assert g.include_coupling_bytes is False and g.expect_byte_target_bytes == STORE_BYTES
    g2 = _config_from_args(p.parse_args(["--byte-budget", "historical"])).gen_baselines
    assert g2.include_coupling_bytes is True and g2.expect_byte_target_bytes is None
    g3 = _config_from_args(
        p.parse_args(["--expect-byte-target", str(TOTAL_BYTES)])).gen_baselines
    assert g3.expect_byte_target_bytes == TOTAL_BYTES


# ---------------------------------------------------------------------------
# ⭐ R27.3 — the LACE whitening-exactness gate picks the FALLBACK, never a weaker rival
# ---------------------------------------------------------------------------


def test_the_whitening_gate_is_config_backed_and_separates_the_measured_shrinkages():
    """MEASURED on the shipped cache: none 0.051-0.057 · ledoit_wolf 0.143-0.150 ·
    1e-3 0.216-0.238.  ⛔ The tolerance must call exactly the first one exact."""
    tol = get_default_config().gen_baselines.lace_whitening_exact_tol
    assert get_default_config().gen_baselines.lace_whitening_exactness_fallback is True
    assert 0.0570 <= tol and tol < 0.1425     # separates `none` from `ledoit_wolf`
    for measured, exact in ((0.0570, True), (0.1438, False), (0.2382, False)):
        assert (measured <= tol) is exact


def test_the_fallback_is_the_nearest_exact_config_in_the_same_grid():
    """⭐ R27.3: same sigma/form where it exists, shrinkage swapped to the exact one —
    ⛔ not a free re-tune, and ⛔ never chosen with C in hand."""
    trials = [
        {"sigma": 0.1, "form": "both", "shrinkage": "1e-3", "val_aggregate": 0.25},
        {"sigma": 0.1, "form": "both", "shrinkage": "none", "val_aggregate": 0.23},
        {"sigma": 0.03, "form": "both", "shrinkage": "none", "val_aggregate": 0.24},
    ]
    exact = {"none": True, "ledoit_wolf": False, "1e-3": False}
    best = trials[0]
    cand = [t for t in trials if exact[t["shrinkage"]]]
    near = [t for t in cand if t["sigma"] == best["sigma"] and t["form"] == best["form"]]
    pool = near or cand
    chosen = pool[int(np.argmax([t["val_aggregate"] for t in pool]))]
    # ⛔ NOT the higher-scoring 0.03/none: the nearest config wins, the tuner does not
    assert chosen["shrinkage"] == "none" and chosen["sigma"] == 0.1


def test_the_fallback_leg_never_replaces_the_pre_registered_one():
    """⛔ 'Both reported, the stronger quoted' — a leg that overwrote the other would
    make the pre-registered number unrecoverable."""
    from chlu.experiments.exp_gen_baselines import _arm_rows

    man = {
        "arms": {"lace": {
            "samplers": {
                "ode": {"in_corner_discrimination": {"balanced_accuracy": 0.68,
                                                     "auroc": 0.78},
                        "composition_A": {"aggregate": 0.9, "per_factor": {"shape": 0.5}},
                        "composition_B": {"aggregate": 0.8, "per_factor": {"shape": 0.4}},
                        "refusal": {"auroc_c_vs_b": 0.70, "auroc_c_vs_a": 0.7,
                                    "auroc_c_vs_a_union_b": 0.7}},
                "ode_wexact": {"in_corner_discrimination": {"balanced_accuracy": 0.66,
                                                            "auroc": 0.75},
                               "composition_A": {"aggregate": 0.9,
                                                 "per_factor": {"shape": 0.5}},
                               "composition_B": {"aggregate": 0.8,
                                                 "per_factor": {"shape": 0.4}},
                               "refusal": {"auroc_c_vs_b": 0.72, "auroc_c_vs_a": 0.7,
                                           "auroc_c_vs_a_union_b": 0.7}},
            },
            "best_sampler_composition": "ode",
            "tuning": {"ORACLE_UPPER_BOUND_auroc_c_vs_b": {
                "value": 0.9, "max_auroc_c_vs_a_union_b_over_all_trials": 0.9}},
        }},
        "ceilings": {"decoder_ceiling": {"composition_A": {"aggregate": 0.94},
                                         "in_corner_discrimination":
                                             {"balanced_accuracy": 0.94, "auroc": 0.99}}},
    }
    rows = _arm_rows(man)
    assert "lace_ode" in rows and "lace_ode_wexact" in rows      # ⭐ both survive
    assert rows["lace_ode"]["in_corner_balacc"] == 0.68
    assert rows["lace_ode_wexact"]["auroc_c_vs_b"] == 0.72
