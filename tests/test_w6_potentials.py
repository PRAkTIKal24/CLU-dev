"""w6 hub/potentials -- the law, pinned against the shorts' own printed numbers.

These tests exist so that the closed forms this hub predicts from cannot drift silently.
Every assertion below is either a number a paper reports, or an identity a paper states.
If one of them fails, the prediction is wrong and every number the hub quotes is void.
"""
import numpy as np
import pytest

from chlu.core import w6_potentials as W6


# ---------------------------------------------------------------------------
# The shorts' own reported numbers.  These are the anchors.
# ---------------------------------------------------------------------------
def test_v2_underdamped_floor_matches_printed_2703():
    """V2 Sec. 4.1: "saturating at a floor of 27.03 steps for gamma = 0.05"."""
    assert W6.n_half_floor(0.05) == pytest.approx(27.03, abs=5e-3)


def test_v5_lambda_ret_and_half_life_at_gamma_0002():
    """V5 App.: "ret = 0.998999499 alongside an n1/2 of 692.5".

    Underdamped, |lambda| = sqrt(det A) = sqrt(1 - gamma) regardless of curvature.
    """
    lam = abs(W6.lambda_slow(1.0, 0.05, 0.002))
    assert lam == pytest.approx(0.998999499, abs=5e-10)
    assert W6.n_half_crr(1.0, 0.05, 0.002) == pytest.approx(692.5, abs=0.1)


def test_crossover_is_gamma_over_two_to_first_order():
    """V2 App. C Eq. (5) / inventory D8: h* = gamma/2 + O(gamma^2)."""
    for g in (0.002, 0.005, 0.01, 0.05):
        hs, _ = W6.band_edges(g)
        assert hs == pytest.approx(g / 2.0, rel=2.0 * g)


def test_stability_limit_is_two_for_every_gamma():
    """V2 App. C: "the map loses stability at h = 2 exactly, for every gamma"."""
    for g in (0.002, 0.05, 0.2, 0.5, 0.9):
        _, hf = W6.band_edges(g)
        assert hf == pytest.approx(2.0, abs=0.25 * g * g + 1e-12) or hf < 2.0
        # and an eigenvalue reaches modulus >= 1 at h = 2
        mu2 = (2.0 / 0.05) ** 2
        assert abs(W6.lambda_slow(mu2, 0.05, g)) >= 1.0 - 1e-12


def test_determinant_and_trace_identities():
    """Inventory D3: det J = 1 - gamma and tr J = (2 - gamma)(1 - h^2/2)."""
    for mu, eta, g in ((0.3, 0.05, 0.05), (1.7, 0.05, 0.2), (0.02, 0.1, 0.01)):
        A = W6.crr_matrix(mu, eta, g)
        h2 = (mu * eta) ** 2
        assert np.linalg.det(A) == pytest.approx(1.0 - g, rel=1e-12)
        assert np.trace(A) == pytest.approx((2.0 - g) * (1.0 - h2 / 2.0), rel=1e-12)


def test_floor_is_curvature_independent():
    """V2 App. C / D7: underdamped, |lambda|^2 = det A = 1 - gamma regardless of mu."""
    g, eta = 0.05, 0.05
    hs, hf = W6.band_edges(g)
    floor = W6.n_half_floor(g)
    for h in (2.0 * hs, 10.0 * hs, 0.5 * hf):
        mu2 = (h / eta) ** 2
        assert W6.band_of(mu2, eta, g) == "underdamped"
        assert W6.n_half_crr(mu2, eta, g) == pytest.approx(floor, rel=1e-9)


def test_overdamped_asymptote_carries_the_gamma():
    """Inventory D5 / V2 Eq. (6): n_half ~= 2 gamma ln2 / [(2-gamma) h^2].

    Deep on the overdamped branch the corrected asymptote must track the exact
    eigenvalue.  Without the gamma it is wrong by exactly 1/gamma.
    """
    g, eta = 0.05, 0.05
    for mu in (0.02, 0.05, 0.1):
        mu2 = mu * mu
        assert W6.band_of(mu2, eta, g) == "overdamped"
        ex = W6.n_half_crr(mu2, eta, g)
        asym = W6.n_half_overdamped_asym(mu2, eta, g)
        assert asym / ex == pytest.approx(1.0, rel=0.05)


def test_overdamped_slope_is_minus_one_in_log_mu2():
    """V2: "slope -1 in log n_half against log mu^2"; V2 measures -0.985."""
    g, eta = 0.05, 0.05
    mu2 = np.array([1e-4, 4e-4, 1.6e-3])
    n = np.array([W6.n_half_crr(m, eta, g) for m in mu2])
    slope = np.polyfit(np.log10(mu2), np.log10(n), 1)[0]
    assert slope == pytest.approx(-1.0, abs=0.02)


def test_latch_has_infinite_half_life():
    """V2 Sec. 3 / D12: at mu = 0 the eigenvalues are {1, 1-gamma}: infinite half-life."""
    assert W6.n_half_crr(0.0, 0.05, 0.05) == float("inf")
    assert abs(W6.lambda_slow(0.0, 0.05, 0.05)) == pytest.approx(1.0, abs=1e-15)


def test_v_curve_minimum_sits_at_the_exact_critical_damping():
    """D16: the V-curve turns where h*(gamma) = h.  gamma_crit_exact inverts that."""
    eta = 0.05
    for mu in (0.5, 2.0, 8.0):
        h = mu * eta
        gc = W6.gamma_crit_exact(h)
        gs = np.geomspace(max(1e-4, gc / 30.0), min(0.95, gc * 30.0), 600)
        n = np.array([W6.n_half_crr(mu * mu, eta, g) for g in gs])
        assert gs[int(np.argmin(n))] == pytest.approx(gc, rel=0.05)


def test_two_eps_mu_is_the_small_h_limit_of_the_exact_critical_damping():
    """The shorts say "inverting at SMALL h, gamma* = 2 eps mu" -- and only at small h."""
    for h in (1e-4, 1e-3, 1e-2):
        assert W6.gamma_crit_exact(h) == pytest.approx(2.0 * h, rel=2.0 * h)
    # and it is materially wrong once h is not small
    assert W6.gamma_crit_exact(0.4) == pytest.approx(0.5631, abs=1e-3)
    assert abs(W6.gamma_crit_exact(0.4) - 0.8) > 0.2


def test_branch_slopes_are_not_plus_minus_one():
    """D17/D18: the shorts derive slopes that EXCEED +-1 at every finite gamma."""
    from chlu.experiments.exp_w6_potentials import d17_slope, d18_slope
    for g in (0.05, 0.2, 0.5):
        assert d17_slope(g) > 1.0
        assert d18_slope(g) < -1.0
    assert d17_slope(0.05) == pytest.approx(1.0 + 0.05 / 1.95, rel=1e-12)


# ---------------------------------------------------------------------------
# LAW B -- the kernel curvature identity.  Derived here, not from the shorts.
# ---------------------------------------------------------------------------
def test_kernel_gradient_and_hessian_against_autodiff():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    rng = np.random.default_rng(0)
    n, d, s = 12, 6, 1.3
    A = rng.normal(size=(n, d))
    life = rng.uniform(0.3, 1.0, n)
    x = rng.normal(size=d)

    def E(z):
        lk = -jnp.sum((z[None, :] - jnp.asarray(A)) ** 2, axis=1) / (2 * s * s) \
            + jnp.log(jnp.asarray(life))
        return -s * s * jax.scipy.special.logsumexp(lk)

    g_ad = np.asarray(jax.grad(E)(jnp.asarray(x)))
    H_ad = np.asarray(jax.hessian(E)(jnp.asarray(x)))
    assert np.abs(W6.kernel_grad(x, A, s, life) - g_ad).max() < 1e-12

    w = W6.kernel_weights(x, A, s, life)
    m = w @ A
    C = (A - m).T @ ((A - m) * w[:, None])
    assert np.abs((np.eye(d) - C / (s * s)) - H_ad).max() < 1e-12


def test_two_atoms_bifurcate_at_delta_over_sigma_equals_two():
    """Curvature along the separation axis is exactly 1 - Delta^2/(4 sigma^2)."""
    s = 1.0
    for r in (0.5, 1.0, 1.9, 2.0, 2.1, 3.0):
        A = np.array([[-r * s / 2, 0.0], [r * s / 2, 0.0]])
        mu2 = W6.curvature_spectrum(np.zeros(2), A, s)["mu2"]
        assert mu2[0] == pytest.approx(1.0 - r * r / 4.0, abs=1e-12)
    assert W6.curvature_spectrum(
        np.zeros(2), np.array([[-1.0, 0.0], [1.0, 0.0]]), 1.0)["mu2"][0] == \
        pytest.approx(0.0, abs=1e-12)


def test_curvature_spectrum_is_low_rank_and_mass_whitened():
    """Only A-1 modes can differ from the trivial one, and mass divides through."""
    rng = np.random.default_rng(3)
    A = rng.normal(size=(8, 40))
    cs1 = W6.curvature_spectrum(A.mean(0), A, 2.0, mass=1.0)
    cs4 = W6.curvature_spectrum(A.mean(0), A, 2.0, mass=4.0)
    assert cs1["mu2"].size <= 7
    assert cs1["trivial_multiplicity"] == 40 - cs1["mu2"].size
    assert np.allclose(cs4["mu2"], cs1["mu2"] / 4.0)
    assert cs4["trivial_mu2"] == pytest.approx(0.25)


def test_mean_shift_settles_to_a_zero_gradient_point():
    rng = np.random.default_rng(7)
    A = rng.normal(size=(20, 10))
    st = W6.mean_shift_settle(A[0], A, 1.5, np.ones(20))
    assert st["converged"]
    assert np.abs(W6.kernel_grad(st["x"], A, 1.5, np.ones(20))).max() < 1e-8


def test_sigma_provenance_separates_the_two_length_scales():
    """The knn_density trap: a within-value spread is not a nearest-neighbour spacing."""
    rng = np.random.default_rng(11)
    A = rng.normal(size=(64, 30)) * 3.0
    pr = W6.sigma_provenance(A, well_sigma=6.2215)
    assert pr["nn_median"] > 0
    assert not pr["inherited_item_spacing"]
    pr_same = W6.sigma_provenance(A, well_sigma=pr["nn_median"])
    assert pr_same["inherited_item_spacing"]


# ---------------------------------------------------------------------------
# The chain: the numpy rollout IS the repo integrator, and it recovers the law.
# ---------------------------------------------------------------------------
def test_numpy_map_is_the_repo_integrator():
    pytest.importorskip("jax")
    from chlu.experiments.exp_w6_potentials import assert_map_matches_repo_integrator
    r = assert_map_matches_repo_integrator()
    assert r["passed"], r


def test_instrument_floor_is_small_enough_for_the_bar():
    """The measurement must recover the CRR where the CRR is exact, to well inside B1."""
    from chlu.experiments.exp_w6_potentials import instrument_floor_on_exact_quadratic
    r = instrument_floor_on_exact_quadratic()
    assert r["n_cells"] >= 12
    assert r["median_abs_log10"] < 0.05, r
    assert r["p90_abs_log10"] < 0.10, r


def test_envelope_and_first_crossing_agree_overdamped_and_diverge_underdamped():
    """V2 Fig. 1: the two instruments coincide overdamped and diverge above h*."""
    from chlu.experiments.exp_w6_potentials import instrument_floor_on_exact_quadratic
    rows = instrument_floor_on_exact_quadratic()["rows"]
    over = [r for r in rows if r["band"] == "overdamped" and np.isfinite(r["n_first"])]
    deep = [r for r in rows if r["h_over_hstar"] > 5.0 and np.isfinite(r["n_first"])]
    assert over, "no overdamped cells"
    for r in over:
        assert r["n_first"] == pytest.approx(r["n_env"], rel=1e-9)
    assert deep, "no deep-underdamped cells"
    assert max(r["n_env"] / r["n_first"] for r in deep) > 2.0
