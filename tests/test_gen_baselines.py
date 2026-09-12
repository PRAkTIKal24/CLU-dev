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
    ``store_plan.json``'s own ``K``/``n_modes``. ⛔ Neither hand-typed."""
    from chlu.experiments.exp_gen_baselines import resolve_store_bytes

    d = tmp_path / "shapes3d_recon"
    d.mkdir(parents=True)
    (d / "store_bytes.json").write_text(json.dumps(
        {"store_bytes": STORE_BYTES, "n_wells": 5, "n_modes": 64}))
    (d / "store_plan.json").write_text(json.dumps({"K": 5, "n_modes": 64}))
    c = get_default_config()
    c.gen_baselines.store_runs_root = str(tmp_path)
    t = resolve_store_bytes(c, "shapes3d", "recon", lambda *a: None)
    assert t["store_bytes"] == STORE_BYTES
    assert t["coupling_bytes"] == COUPLING_BYTES
    assert t["total_bytes"] == TOTAL_BYTES
    # ⛔ and the escape hatch still excludes it explicitly, never silently
    c.gen_baselines.include_coupling_bytes = False
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
