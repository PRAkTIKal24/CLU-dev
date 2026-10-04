"""The composition certificate: the derived barrier, the controls, and the audit.

⭐ The load-bearing tests here are the ones that can **fail on purpose**:

- ``test_the_audit_fires_when_a_C_id_reaches_a_fit`` — the zero-invalid-supervision
  assertion is handed a fit that *did* see a C record, and must FIRE.  An assertion
  never seen to fail is not evidence, and "a threshold touched invalid data" is
  precisely the failure that would void this spoke's headline.
- ``test_the_certificate_path_refuses_a_decoder`` — the structural guard is shown to
  fire on a function that takes one.
- ``test_hessian_closed_form_matches_finite_differences`` — the curvature component is
  an analytic Hessian; if the algebra were wrong nothing downstream would complain,
  so it is checked against numerical differentiation.
- ``test_reach_cannot_see_a_duplicate_and_the_pair_barrier_can`` — pins the honest
  decomposition the report rests on.

Everything runs on synthetic tensors; the real-data path is exercised by the harness
runs reported in ``.claude/outputs/gen-certificate.md``.

⚠ §7.23: ``test_certificate_is_stable_under_x64`` runs the whole scoring path with
``jax_enable_x64`` on against a float32 store, because six shipped test modules enable
x64 at import and sort before this one.
"""

import json

import jax
import numpy as np
import pytest

from chlu.config import GenCertificateConfig, get_default_config, load_config, save_config
from chlu.core.gen_certificate import (
    ARMS,
    COMPONENTS,
    C_KINDS,
    Request,
    assert_no_decoder_in_path,
    build_term_statistics,
    certificate_components,
    codes_for,
    compose,
    composed_point,
    factor_support,
    load_c_requests,
    naive_distance,
    pair_conflict,
    reach_energy,
    redundant_requests,
    requests_from_rows,
    score_requests,
    self_conflict_is_non_positive,
    store_energy,
    term_distribution,
    well_curvature,
)
from chlu.core.gen_store import GenStore, init_store
from chlu.eval.gen_cert_eval import (
    ValidOnlyFitter,
    auroc,
    certificate_metrics,
    operating_point,
    risk_coverage,
    weighted_auroc,
)

FACTORS = ("f0", "f1", "f2")
CARDS = (4, 3, 5)


# ---------------------------------------------------------------------------
# Fixtures: a tiny grid, a tiny store, and features that carry the factors
# ---------------------------------------------------------------------------


class _Grid:
    """Minimal stand-in for :class:`chlu.data.gen_splits.FactorGrid`."""

    names = FACTORS
    values = tuple(np.arange(c, dtype=np.float64) * 0.5 for c in CARDS)
    cardinalities = CARDS

    def factor_indices_of(self, rows):
        return np.stack(np.unravel_index(np.asarray(rows, np.int64), CARDS), axis=-1)

    def values_of(self, rows):
        fi = self.factor_indices_of(rows)
        return np.stack([self.values[k][fi[:, k]] for k in range(3)], axis=-1)


def _synthetic(n=600, d=12, seed=0):
    """Features in which each factor drives its own disjoint block of coordinates."""
    rng = np.random.default_rng(seed)
    rows = rng.integers(0, int(np.prod(CARDS)), size=n)
    grid = _Grid()
    fi = grid.factor_indices_of(rows)
    x = 0.05 * rng.standard_normal((n, d)).astype(np.float32)
    for j in range(3):
        blk = slice(j * 4, (j + 1) * 4)
        x[:, blk] += (fi[:, j][:, None] + 1.0).astype(np.float32)
    return grid, rows, fi, x


def _store(d=12, m=8, seed=0):
    wc = (tuple(range(0, 4)), tuple(range(4, 8)), tuple(range(8, 12)))
    return GenStore(wc, m, (d,), jax.random.PRNGKey(seed))


def _fitted_store(x, m=6, seed=0):
    wc = (tuple(range(0, 4)), tuple(range(4, 8)), tuple(range(8, 12)))
    return init_store(x, wc, m, (x.shape[1],), jax.random.PRNGKey(seed), seed=seed)


def _stats(store, x, fi, with_features=True):
    codes = codes_for(store, x)
    return build_term_statistics(codes, fi, CARDS,
                                 features=x if with_features else None,
                                 alpha=1.0, row_ids=np.arange(x.shape[0]))


# ---------------------------------------------------------------------------
# 1. Requests
# ---------------------------------------------------------------------------


def test_a_and_b_requests_are_built_from_the_grids_own_values():
    grid, rows, fi, _ = _synthetic(n=8)
    reqs = requests_from_rows(grid, rows[:3], "A")
    assert [r.source for r in reqs] == ["A", "A", "A"]
    assert all(r.n_terms == 3 for r in reqs)
    for r, row in zip(reqs, rows[:3], strict=True):
        assert r.id == f"A:{int(row)}"
        vals = tuple(float(v) for _, v in r.terms)
        assert vals == tuple(grid.values_of(np.array([row]))[0])


def test_c_records_are_read_verbatim_with_their_kinds(tmp_path):
    payload = {
        "schema": {"version": 1},
        "factors": {"names": list(FACTORS)},
        "records": [
            {"id": "x-dup-0", "kind": "duplicate_factor", "violated_factor": "f0",
             "reason": "", "request": [
                 {"factor": "f0", "factor_index": 0, "value": 0.0},
                 {"factor": "f1", "factor_index": 1, "value": 0.5},
                 {"factor": "f2", "factor_index": 2, "value": 1.0},
                 {"factor": "f0", "factor_index": 0, "value": 1.5}]},
            {"id": "x-oor-0", "kind": "out_of_range", "violated_factor": "f2",
             "reason": "", "request": [
                 {"factor": "f0", "factor_index": 0, "value": 0.0},
                 {"factor": "f1", "factor_index": 1, "value": 0.5},
                 {"factor": "f2", "factor_index": 2, "value": 9.0}]},
        ],
    }
    p = tmp_path / "c_invalid.json"
    p.write_text(json.dumps(payload))
    reqs = load_c_requests(p, FACTORS)
    assert [r.id for r in reqs] == ["x-dup-0", "x-oor-0"]
    assert [r.kind for r in reqs] == list(C_KINDS)
    assert reqs[0].n_terms == 4 and reqs[1].n_terms == 3
    assert reqs[0].violated_factor == "f0"


def test_c_loader_refuses_a_file_whose_factors_disagree(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"schema": {"version": 1},
                             "factors": {"names": ["a", "b"]}, "records": []}))
    with pytest.raises(ValueError, match="single-sourced"):
        load_c_requests(p, FACTORS)


# ---------------------------------------------------------------------------
# 2. Reach — and the honest limit of what it can see
# ---------------------------------------------------------------------------


def test_reach_is_exactly_zero_on_grid_and_grows_off_it():
    grid = _Grid()
    sup = factor_support(grid, 0.5)
    assert sup.radius[0] == pytest.approx(0.25)
    on = Request("a", "A", ((0, 0.5), (1, 1.0), (2, 2.0)))
    assert reach_energy(on, sup)["reach"] == 0.0
    off = Request("c", "C", ((0, 0.5), (1, 1.0), (2, 9.0)), kind="out_of_range",
                  violated_factor="f2")
    e = reach_energy(off, sup)
    assert e["reach"] > 0
    assert e["reach_argmax_factor"] == 2


def test_reach_cannot_see_a_duplicate_and_the_pair_barrier_can():
    """⭐ The decomposition the whole report rests on: ``reach`` is identically 0 on a
    ``duplicate_factor`` request (every value is on-grid), so only the composition
    barrier can refuse it."""
    grid, rows, fi, x = _synthetic()
    sup = factor_support(grid, 0.5)
    store = _fitted_store(x)
    stats = _stats(store, x, fi)
    valid = Request("A:0", "A", ((0, 0.0), (1, 0.5), (2, 1.0)))
    dup = Request("C:0", "C", ((0, 0.0), (1, 0.5), (2, 1.0), (0, 1.5)),
                  kind="duplicate_factor", violated_factor="f0")
    assert reach_energy(dup, sup)["reach"] == 0.0
    cv = certificate_components(valid, store, stats, sup, 0.5)
    cd = certificate_components(dup, store, stats, sup, 0.5)
    assert cd["pair"] > cv["pair"], (cd["pair"], cv["pair"])
    assert cd["pair_argmax_factors"] == [0, 0]


def test_pair_conflict_is_zero_for_the_prior_and_large_for_disjoint_terms():
    prior = np.full((2, 4), 0.25)
    assert pair_conflict(prior, prior, prior) == pytest.approx(0.0)
    q1 = np.array([[0.97, 0.01, 0.01, 0.01], [0.25, 0.25, 0.25, 0.25]])
    q2 = np.array([[0.01, 0.97, 0.01, 0.01], [0.25, 0.25, 0.25, 0.25]])
    assert pair_conflict(q1, q2, prior) > 1.0
    assert pair_conflict(q1, q1, prior) < pair_conflict(q1, q2, prior)


def test_term_distribution_is_a_simplex_per_well():
    d = term_distribution(np.zeros((3, 5)), 1.0)
    assert d.shape == (3, 5)
    assert np.allclose(d.sum(axis=1), 1.0)
    assert np.allclose(d, 0.2)


# ---------------------------------------------------------------------------
# 3. The curvature component — analytic Hessian vs finite differences
# ---------------------------------------------------------------------------


def test_hessian_closed_form_matches_finite_differences():
    """``Hess Phi = 2 I - (2/tau^2) Cov_w`` — checked numerically, because a wrong
    constant here would silently reweight the certificate and nothing would complain."""
    rng = np.random.default_rng(0)
    d, m, tau = 5, 4, 0.7
    mu = rng.standard_normal((m, d))
    store = GenStore((tuple(range(d)),), m, (d,), jax.random.PRNGKey(0),
                     init=[mu.astype(np.float32)])
    x = rng.standard_normal(d) * 0.3

    def phi(z):
        d2 = np.sum((mu - z[None, :]) ** 2, axis=1)
        a = -d2 / (2 * tau * tau)
        return -2 * tau * tau * (np.log(np.sum(np.exp(a - a.max()))) + a.max())

    h = 1e-4
    H = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            ei, ej = np.zeros(d), np.zeros(d)
            ei[i] = h
            ej[j] = h
            H[i, j] = (phi(x + ei + ej) - phi(x + ei - ej)
                       - phi(x - ei + ej) + phi(x - ei - ej)) / (4 * h * h)
    lam_fd = float(np.linalg.eigvalsh(0.5 * (H + H.T))[0])
    got = well_curvature(store, [x.astype(np.float32)], tau)["lambda_min_hess"][0]
    assert got == pytest.approx(lam_fd, abs=2e-3), (got, lam_fd)


def test_curvature_is_larger_between_two_distant_modes_than_on_one():
    d, m, tau = 4, 2, 0.5
    mu = np.zeros((m, d), np.float32)
    mu[0, 0] = -3.0
    mu[1, 0] = 3.0
    store = GenStore((tuple(range(d)),), m, (d,), jax.random.PRNGKey(0), init=[mu])
    on_mode = well_curvature(store, [mu[0]], tau)["curv"]
    between = well_curvature(store, [np.zeros(d, np.float32)], tau)["curv"]
    assert between > on_mode


def test_compose_is_the_prior_when_no_term_is_supplied():
    prior = np.array([[0.1, 0.2, 0.7], [0.5, 0.25, 0.25]])
    r, _ = compose([], prior)
    assert np.allclose(r, prior)


def test_composed_point_is_a_convex_combination_of_the_modes():
    store = _store(d=12, m=8)
    r = np.zeros((3, 8))
    r[:, 2] = 1.0
    x = composed_point(store, r)
    for k in range(3):
        assert np.allclose(np.asarray(x[k]), np.asarray(store.modes[k])[2], atol=1e-6)
    assert store_energy(store, x) == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# 4. The laundering control
# ---------------------------------------------------------------------------


def test_the_naive_control_is_scored_on_the_same_store_and_features():
    grid, rows, fi, x = _synthetic()
    sup = factor_support(grid, 0.5)
    store = _fitted_store(x)
    stats = _stats(store, x, fi)
    req = Request("A:0", "A", ((0, 0.0), (1, 0.5), (2, 1.0)))
    assert naive_distance(req, store, stats, sup) >= 0.0


def test_the_naive_control_refuses_to_run_without_prototype_features():
    grid, rows, fi, x = _synthetic()
    store = _fitted_store(x)
    stats = _stats(store, x, fi, with_features=False)
    req = Request("A:0", "A", ((0, 0.0), (1, 0.5), (2, 1.0)))
    with pytest.raises(ValueError, match="same features"):
        naive_distance(req, store, stats, factor_support(grid, 0.5))


# ---------------------------------------------------------------------------
# 5. ⭐ The audit — including the positive control
# ---------------------------------------------------------------------------


def test_the_audit_certifies_a_run_that_only_saw_valid_ids():
    f = ValidOnlyFitter()
    f.fit_standardiser("z", ["A:1", "A:2"], [1.0, 3.0])
    f.fit_quantile("t", ["A:1", "A:2"], [1.0, 3.0], 0.95)
    out = f.assert_no_invalid(["c-dup-0", "c-oor-0"])
    assert out["CERTIFIED_no_invalid_supervision"] is True
    assert out["n_fits_audited"] == 2
    assert out["n_ids_audited"] == 4
    assert out["n_invalid_ids_found_in_any_fit"] == 0
    assert f.certified


def test_the_audit_fires_when_a_C_id_reaches_a_fit():
    """⛔ POSITIVE CONTROL.  If this ever stops raising, the headline claim of the
    spoke is unprotected."""
    f = ValidOnlyFitter()
    f.fit_quantile("t", ["A:1", "c-dup-0"], [1.0, 3.0], 0.95)
    with pytest.raises(AssertionError, match="INVALID-EXAMPLE SUPERVISION"):
        f.assert_no_invalid(["c-dup-0"])
    assert not f.certified


def test_the_audit_refuses_an_empty_invalid_set():
    f = ValidOnlyFitter()
    f.fit_quantile("t", ["A:1"], [1.0], 0.5)
    with pytest.raises(ValueError, match="cannot fail is not evidence"):
        f.assert_no_invalid([])


def test_a_zero_variance_component_is_recorded_not_silently_patched():
    f = ValidOnlyFitter()
    s = f.fit_standardiser("z[reach]", ["A:1", "A:2"], [0.0, 0.0])
    assert s["passthrough"] is True
    assert f.records[0]["passthrough_zero_variance"] is True


def test_every_fit_records_the_hash_of_exactly_the_ids_it_saw():
    f = ValidOnlyFitter()
    f.fit_quantile("t", ["A:2", "A:1"], [1.0, 3.0], 0.5)
    g = ValidOnlyFitter()
    g.fit_quantile("t", ["A:1", "A:2"], [1.0, 3.0], 0.5)
    assert f.records[0]["ids_sha256"] == g.records[0]["ids_sha256"]
    h = ValidOnlyFitter()
    h.fit_quantile("t", ["A:1", "A:3"], [1.0, 3.0], 0.5)
    assert h.records[0]["ids_sha256"] != f.records[0]["ids_sha256"]


# ---------------------------------------------------------------------------
# 6. Metrics
# ---------------------------------------------------------------------------


def test_auroc_is_one_for_a_perfect_separation_and_half_for_ties():
    assert auroc([0, 0, 1, 1], [0.0, 0.1, 0.2, 0.3]) == 1.0
    assert auroc([0, 0, 1, 1], [1.0, 1.0, 1.0, 1.0]) == 0.5
    assert auroc([0, 1], [1.0, 0.0]) == 0.0


def test_the_balanced_pooled_auroc_is_the_mean_of_the_two():
    """⭐ The weighted Mann-Whitney identity R23's third number relies on."""
    rng = np.random.default_rng(0)
    a, b, c = rng.normal(0, 1, 500), rng.normal(0.5, 1, 7), rng.normal(2, 1, 7)
    m = certificate_metrics(a, b, c, ["duplicate_factor"] * 4 + ["out_of_range"] * 3)
    assert m["AB_vs_C_balanced"] == pytest.approx(
        0.5 * (m["A_vs_C"] + m["B_vs_C"]), abs=1e-12)
    assert m["AB_vs_C_balanced_identity_check"] < 1e-12
    assert m["n_C[duplicate_factor]"] == 4 and m["n_C[out_of_range]"] == 3


def test_the_unbalanced_pooled_auroc_is_reported_but_labelled():
    rng = np.random.default_rng(1)
    a, b, c = rng.normal(0, 1, 400), rng.normal(0, 1, 4), rng.normal(3, 1, 4)
    m = certificate_metrics(a, b, c, ["out_of_range"] * 4)
    assert "pooled_UNBALANCED_do_not_quote" in m
    assert "do not quote" in m["pooled_note"].lower() or "⛔" in m["pooled_note"]


def test_weighted_auroc_reduces_to_plain_auroc_at_uniform_weights():
    rng = np.random.default_rng(2)
    s = rng.normal(size=40)
    y = (rng.random(40) > 0.5).astype(int)
    assert weighted_auroc(y, s, np.ones(40)) == pytest.approx(auroc(y, s), abs=1e-12)


def test_risk_coverage_endpoints_and_the_random_baseline():
    rc = risk_coverage([0.0, 1.0, 2.0, 3.0], [0, 0, 1, 1])
    assert rc["coverage"][-1] == pytest.approx(1.0)
    assert rc["risk"][-1] == pytest.approx(0.5)
    assert rc["risk"][0] == 0.0
    assert rc["aurc"] < rc["aurc_random"]
    bad = risk_coverage([3.0, 2.0, 1.0, 0.0], [0, 0, 1, 1])
    assert bad["aurc"] > bad["aurc_random"]


def test_risk_coverage_collapses_ties_instead_of_trusting_input_order():
    """⛔ REGRESSION.  A score that ties every valid request with half the invalid ones
    is uninformative, and its AURC must equal the base rate.  Resolving ties in input
    order instead gave the degenerate ``reach`` arm a near-optimal AURC it had not
    earned (found in this spoke's own smoke run)."""
    rc = risk_coverage([0.0, 0.0, 0.0, 0.0], [0, 0, 1, 1])
    assert rc["n_distinct_scores"] == 1
    assert rc["coverage"] == [1.0]
    assert rc["aurc"] == pytest.approx(rc["aurc_random"])
    assert rc["aurc"] == pytest.approx(0.5)
    # a tied valid/invalid block in the middle is admitted together
    rc2 = risk_coverage([0.0, 1.0, 1.0, 2.0], [0, 0, 1, 1])
    assert rc2["n_distinct_scores"] == 3
    assert rc2["risk"][1] == pytest.approx(1.0 / 3.0)


def test_operating_point_reports_a_refusal_rate_per_set():
    op = operating_point(1.0, {"A": [0.0, 0.5], "B": [2.0], "C": [2.0, 3.0]})
    assert op["refusal_rate_A"] == 0.0
    assert op["refusal_rate_B"] == 1.0
    assert op["refusal_rate_C"] == 1.0
    assert op["n_C"] == 2


# ---------------------------------------------------------------------------
# 7. Structural guards
# ---------------------------------------------------------------------------


def test_the_certificate_path_contains_no_decoder():
    out = assert_no_decoder_in_path()
    assert out["decoder_in_certificate_path"] is False


def test_the_certificate_path_refuses_a_decoder(monkeypatch):
    """⛔ POSITIVE CONTROL for the structural guard."""
    import chlu.core.gen_certificate as mod

    def store_energy_with_a_decoder(store, xhat, decoder=None):
        return 0.0

    monkeypatch.setattr(mod, "store_energy", store_energy_with_a_decoder)
    with pytest.raises(AssertionError, match="no decoder"):
        assert_no_decoder_in_path()


def test_scoring_a_set_preserves_order_and_emits_every_component():
    grid, rows, fi, x = _synthetic()
    store = _fitted_store(x)
    stats = _stats(store, x, fi)
    sup = factor_support(grid, 0.5)
    reqs = requests_from_rows(grid, rows[:5], "A")
    out = score_requests(reqs, store, stats, sup, 0.5)
    assert [r["id"] for r in out] == [r.id for r in reqs]
    for row in out:
        for c in COMPONENTS:
            assert np.isfinite(row[c])
        assert np.isfinite(row["dist"])
        assert row["n_terms"] == 3


def test_certificate_is_stable_under_x64():
    """⚠ §7.23: shipped modules enable x64 and this one must survive that.

    ⛔ **And must not cause it.**  ``test_goldstone`` / ``test_kt`` / ``test_lattice``
    / ``test_twins`` / ``test_friction_field`` / ``test_wormhole`` enable
    ``jax_enable_x64`` at **module import**, i.e. during collection — *before any
    test runs*, and regardless of how the file names sort.  An earlier version of
    this test restored the flag to a hard-coded ``False`` and turned x64 off under
    all of them: 9 failures in ``test_goldstone.py`` alone, reproducible with just
    these two files.  ⭐ Save and restore the **observed** value, never a literal.
    """
    grid, rows, fi, x = _synthetic()
    store = _fitted_store(x)
    stats = _stats(store, x, fi)
    sup = factor_support(grid, 0.5)
    req = requests_from_rows(grid, rows[:1], "A")[0]
    before = certificate_components(req, store, stats, sup, 0.5)
    prior = bool(jax.config.jax_enable_x64)
    jax.config.update("jax_enable_x64", True)
    try:
        after = certificate_components(req, store, stats, sup, 0.5)
        codes = codes_for(store, x)
        assert codes.dtype == np.int64
    finally:
        jax.config.update("jax_enable_x64", prior)
    assert bool(jax.config.jax_enable_x64) is prior
    for c in COMPONENTS:
        assert after[c] == pytest.approx(before[c], rel=1e-9, abs=1e-12)


# ---------------------------------------------------------------------------
# 8. Config + CLI contract
# ---------------------------------------------------------------------------


def test_gen_certificate_config_defaults():
    c = GenCertificateConfig()
    assert c.groupings == ["correlation", "random"]
    assert c.accept_quantile == 0.95
    assert c.reach_radius_frac == 0.5
    assert c.tau_curvature is None
    assert c.seeds == [0, 1, 2]
    assert get_default_config().gen_certificate.dataset == "shapes3d"


def test_gen_certificate_config_round_trips_through_yaml(tmp_path):
    cfg = get_default_config()
    cfg.gen_certificate.accept_quantile = 0.9
    cfg.gen_certificate.laplace_alpha = 0.5
    cfg.gen_certificate.groupings = ["correlation"]
    p = tmp_path / "c.yaml"
    save_config(cfg, p)
    back = load_config(p)
    assert back.gen_certificate.accept_quantile == 0.9
    assert back.gen_certificate.laplace_alpha == 0.5
    assert back.gen_certificate.groupings == ["correlation"]


def test_cli_exposes_the_binding_contract():
    from chlu.experiments.exp_gen_certificate import build_arg_parser

    dest = {a.dest for a in build_arg_parser()._actions}
    for required in ("dataset", "pretext", "stage", "arm", "seed", "out", "condition"):
        assert required in dest, required


def test_module_refuses_a_stage_that_is_not_its_own(capsys):
    from chlu.experiments.exp_gen_certificate import main

    assert main(["--stage", "encoder"]) == 2
    assert "not this module" in capsys.readouterr().err


def test_arms_include_both_laundering_controls():
    assert "distance" in ARMS and "distance_reach" in ARMS and "count" in ARMS


def test_the_term_count_control_is_valid_and_the_same_length_as_a_C_duplicate():
    """⭐ A7: a redundant term repeats a factor **at the value it already has**, so the
    request stays satisfiable while carrying the 7th term a ``duplicate_factor`` C
    record carries.  ⛔ Built from valid requests alone — no invalid example is used."""
    grid, rows, fi, _ = _synthetic(n=12)
    reqs = requests_from_rows(grid, rows[:6], "A")
    a7 = redundant_requests(reqs, len(FACTORS))
    assert len(a7) == len(reqs)
    for base, r in zip(reqs, a7, strict=True):
        assert r.source == "A7" and r.kind == "valid_redundant"
        assert r.n_terms == base.n_terms + 1
        # the extra term is a REPEAT, not a conflict: one value per factor still
        per_factor = {}
        for j, u in r.terms:
            per_factor.setdefault(j, set()).add(u)
        assert all(len(v) == 1 for v in per_factor.values())


def test_the_count_control_is_blind_to_the_term_count_control():
    """The syntactic ``count`` statistic separates A from C-duplicates for free; on A7
    it is exactly at chance, which is the whole point of A7."""
    grid, rows, fi, _ = _synthetic(n=12)
    a7 = redundant_requests(requests_from_rows(grid, rows[:6], "A"), len(FACTORS))
    n_a7 = [float(r.n_terms) for r in a7]
    n_cdup = [4.0] * 6
    assert auroc([0] * 6 + [1] * 6, n_a7 + n_cdup) == 0.5


def test_the_aggregate_plot_draws_the_controls_in_the_same_axes(tmp_path):
    """⛔ A laundering control that lives in an appendix panel is easy to skip; the
    plot puts the certificate and both controls on the same axes."""
    from chlu.experiments.exp_gen_certificate import plot_aggregate

    def ms(v):
        return {"mean": v, "std": 0.01, "per_seed": [v], "n_seeds": 1}

    per = {a: {k: ms(0.8) for k in ("A_vs_C", "B_vs_C", "AB_vs_C_balanced",
                                    "B_vs_C[duplicate_factor]",
                                    "A7_vs_C[duplicate_factor]")}
           for a in ("clu", "distance", "distance_reach", "count")}
    agg = {"seeds": [0], "arms": {"correlation": {"per_arm": per}}}
    p = plot_aggregate(agg, tmp_path / "fig.png")
    assert p.exists() and p.stat().st_size > 0


def test_the_pair_barrier_is_invariant_to_a_redundant_term():
    """⭐ The structural reason ``pair`` is the term-count-clean statistic.

    ``chi(t,t) <= 0`` by Cauchy-Schwarz, and every other new pair repeats a value the
    request already had, so ``max over pairs`` does not move when a term is repeated.
    ⛔ The composed-point components (``curv``, ``energy``) have no such property —
    they read a distribution that sharpens with each extra term, which is exactly how
    the C set's 7-terms-vs-6 giveaway leaks into a score."""
    grid, rows, fi, x = _synthetic()
    sup = factor_support(grid, 0.5)
    store = _fitted_store(x)
    stats = _stats(store, x, fi)
    base = requests_from_rows(grid, rows[:16], "A")
    a7 = redundant_requests(base, len(FACTORS))
    moved_pair, moved_curv = 0, 0
    for b, r in zip(base, a7, strict=True):
        cb = certificate_components(b, store, stats, sup, 0.5)
        cr = certificate_components(r, store, stats, sup, 0.5)
        assert cr["n_terms"] == cb["n_terms"] + 1
        moved_pair += cr["pair"] != cb["pair"]
        moved_curv += cr["curv"] != cb["curv"]
    assert moved_pair == 0, f"{moved_pair}/16 pair scores moved under a redundant term"
    assert moved_curv > 0, "the curvature component should move — that is the point"
    for j in range(len(FACTORS)):
        for v in range(CARDS[j]):
            assert self_conflict_is_non_positive(stats.term(j, v), stats.prior)
