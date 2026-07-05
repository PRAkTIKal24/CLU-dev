"""F5 Appendix-N smoke checks for the Goldstone measurement harness (V2/Exp D).

These validate the HARNESS against the exact quadratic predictions of the
formalism note (F5 §3.3–§3.4, §4.1) on hand-built potentials: the latch, the
1/mu^2 half-life law, underdamped saturation, Noether-charge decay, and the
kinetic-isotropy requirement. Per the task contract: if these fail, the
harness (or integrator) is wrong — not F5.

Precision: F5's Appendix-N numbers are float64; this module enables JAX x64
at import. The flag is process-global (pytest imports all test modules at
collection), but the pre-existing tests are tolerance-based and only get
*more* accurate under x64 — verified by running the full suite.
"""

import math

import jax

jax.config.update("jax_enable_x64", True)

import equinox as eqx  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from chlu.core.chlu_unit import CHLU  # noqa: E402
from chlu.core.potentials import SO2InvariantPotential, TiltedPotential  # noqa: E402
from chlu.data.circle_vacuum import generate_circle_vacuum  # noqa: E402
from chlu.experiments.goldstone_harness import (  # noqa: E402
    MexicanHatPotential,
    QuadraticPotential,
    classify_mode,
    clu_with_potential,
    fit_decay_rate,
    half_life_first_crossing,
    mode_coordinates,
    noether_charge,
    perturb_and_track,
    predicted_half_life,
    rollout_from,
    settle,
    spectrum_probe,
    step_jacobian,
)

DT = 0.05


def _exact_lambda_slow(mu_sq: float, dt: float, gamma: float) -> float:
    """Exact slow eigenvalue of the per-mode 2x2 damped-Verlet map (F5 §3.3),
    written out independently of the harness helpers (cross-validation)."""
    h2 = dt * dt * mu_sq
    tr = (2.0 - gamma) * (1.0 - h2 / 2.0)
    det = 1.0 - gamma
    disc = math.sqrt(tr * tr - 4.0 * det)  # real in the overdamped regime
    return (tr + disc) / 2.0


# ---------------------------------------------------------------------------
# (a) The Goldstone latch: q_inf = q0 + dt*p0/(M*gamma), exactly (F5 §3.3a)
# ---------------------------------------------------------------------------


def test_goldstone_latch_exact():
    gamma = 0.05
    inertia = 1.5
    # One exactly flat direction (k=0) + one curved companion (k=1)
    pot = QuadraticPotential(K=jnp.diag(jnp.array([0.0, 1.0])))
    model = clu_with_potential(
        pot, dim=2, kinetic_mode="newtonian_learned", inertia=(inertia, inertia)
    )
    m_eff = model.effective_inertia()
    assert abs(float(m_eff[0]) - inertia) < 1e-12  # log_mass_for_inertia is exact

    q0 = jnp.array([0.3, 0.2])
    p0 = jnp.array([0.5, -0.4])
    steps = 4000
    traj = rollout_from(model, q0, p0, steps=steps, dt=DT, gamma=gamma)

    # Exact latch: dissipation FREEZES the flat-direction displacement
    q_inf_pred = float(q0[0]) + DT * float(p0[0]) / (float(m_eff[0]) * gamma)
    latch_err = abs(float(traj[-1, 0]) - q_inf_pred)
    assert latch_err < 1e-11, f"latch error {latch_err:.3e} (F5 App-N: 1e-15)"

    # Frozen between steps 2000 -> 4000 (deadbeat memory)
    assert abs(float(traj[-1, 0]) - float(traj[2000, 0])) < 1e-13

    # Flat-mode momentum dies exactly geometrically: p_n = (1-gamma)^n p0
    n = np.arange(301)
    p_flat = np.asarray(traj[:301, 2])
    p_pred = (1.0 - gamma) ** n * float(p0[0])
    assert np.max(np.abs(p_flat - p_pred)) < 1e-12

    # Curved companion decayed to numerical zero (F5: ~1e-45)
    assert abs(float(traj[-1, 1])) < 1e-12


# ---------------------------------------------------------------------------
# (b) Overdamped half-life law: n_1/2 ∝ 1/mu^2, ratio 4.0 for mu^2-ratio 4
#     (F5 §3.3b–c; App-N measured 1544/6165, ratios 3.993/3.998)
# ---------------------------------------------------------------------------


def test_overdamped_half_life_ratio_four():
    gamma = 0.2  # h*(0.2) = 0.111284; both modes far below => register band
    pot = QuadraticPotential(K=jnp.diag(jnp.array([0.04, 0.01])))
    model = clu_with_potential(pot, dim=2, kinetic_mode="newtonian_identity")
    probe = spectrum_probe(model, jnp.zeros(2))

    # eigh ascending: mode 0 -> mu^2=0.01, mode 1 -> mu^2=0.04
    assert np.allclose(np.asarray(probe.mu_sq), [0.01, 0.04], atol=1e-12)
    assert classify_mode(0.01, DT, gamma) == "register"
    assert classify_mode(0.04, DT, gamma) == "register"

    res_slow = perturb_and_track(
        model,
        probe,
        mode_idx=0,
        kick=0.1,
        kick_type="position",
        steps=8000,
        dt=DT,
        gamma=gamma,
    )
    res_fast = perturb_and_track(
        model,
        probe,
        mode_idx=1,
        kick=0.1,
        kick_type="position",
        steps=2500,
        dt=DT,
        gamma=gamma,
    )
    n_slow = half_life_first_crossing(res_slow["retention"])
    n_fast = half_life_first_crossing(res_fast["retention"])

    # Ratio 4.0 for mu^2-ratio 4 (F5 measured 3.993)
    ratio = n_slow / n_fast
    assert abs(ratio - 4.0) < 0.05, f"half-life ratio {ratio} != 4.0"

    # Absolute agreement with the EXACT eigenvalue prediction (in-test formula)
    for mu_sq, n_meas in [(0.04, n_fast), (0.01, n_slow)]:
        lam = _exact_lambda_slow(mu_sq, DT, gamma)
        n_pred = math.log(2.0) / (-math.log(lam))
        assert abs(n_meas - n_pred) / n_pred < 0.02, (
            f"mu^2={mu_sq}: measured {n_meas} vs exact {n_pred:.1f}"
        )
        # And the harness helper must agree with the in-test exact formula
        assert abs(predicted_half_life(mu_sq, DT, gamma) - n_pred) / n_pred < 1e-9

    # F5 App-N absolute anchor: ~1544 steps for mu^2=0.04 at gamma=0.2, dt=0.05
    assert abs(n_fast - 1544) < 40


# ---------------------------------------------------------------------------
# (c) Underdamped saturation at 2*ln2/(-ln(1-gamma)) steps, mass-independent
#     (F5 §3.3b; App-N: first-crossing 23–26 vs 27; |lambda| = sqrt(1-gamma))
# ---------------------------------------------------------------------------


def test_underdamped_saturation_and_mass_independence():
    gamma = 0.05  # h*(0.05) = 0.025643; mu=1 -> h=0.05 => underdamped
    pot = QuadraticPotential(K=jnp.array([[1.0]]))
    model = clu_with_potential(pot, dim=1, kinetic_mode="newtonian_identity")
    probe = spectrum_probe(model, jnp.zeros(1))
    assert classify_mode(float(probe.mu_sq[0]), DT, gamma) == "working_memory"

    res = perturb_and_track(
        model,
        probe,
        mode_idx=0,
        kick=0.1,
        kick_type="position",
        steps=600,
        dt=DT,
        gamma=gamma,
    )

    # Saturated half-life = 2*ln2/(-ln(1-gamma)) = 27.03 steps. The raw
    # first-crossing is phase-convention dependent (F5 App-N "known diagnostic
    # artifacts"): friction only bites momentum, so log-amplitude ripples with
    # amplitude gamma/(2h) = 0.5 in log E, i.e. a crossing jitter of
    # +-(gamma/2h)/|ln sqrt(1-gamma)| ~ +-10 steps around 27 (F5's own kick
    # phase measured 23–26; a position kick lags to ~37). The window below
    # covers the ripple band while excluding the overdamped law, which would
    # predict 14.2 steps for these parameters.
    n_half = half_life_first_crossing(res["retention"])
    n_sat = 2.0 * math.log(2.0) / (-math.log(1.0 - gamma))
    assert 20 <= n_half <= 42, f"underdamped n_1/2 = {n_half}, expected {n_sat:.1f}±10"
    assert abs(predicted_half_life(1.0, DT, gamma) - n_sat) < 1e-9

    # The LAW, asserted tightly: envelope decay rate ln|lambda| =
    # 0.5*ln(1-gamma) (mass-independent saturation), fit over 4 periods
    slope = fit_decay_rate(res["retention"], n_fit=504)
    target = 0.5 * math.log(1.0 - gamma)
    assert abs(slope - target) / abs(target) < 0.01, f"fit {slope} vs {target}"
    # ... equivalently, the fit-derived half-life sits on the saturation value
    assert abs(math.log(2.0) / (-slope) - n_sat) / n_sat < 0.05

    # Mass-independence at the eigenvalue level (F5's ground truth): the
    # step-Jacobian eigen-moduli equal sqrt(1-gamma) for BOTH inertial masses
    for inertia in (1.0, 0.25):
        m = clu_with_potential(
            pot, dim=1, kinetic_mode="newtonian_learned", inertia=(inertia,)
        )
        J = step_jacobian(m, jnp.zeros(1), jnp.zeros(1), DT, gamma)
        mods = np.abs(np.linalg.eigvals(np.asarray(J)))
        assert np.max(np.abs(mods - math.sqrt(1.0 - gamma))) < 1e-9, (
            f"inertia={inertia}: |lambda| = {mods}"
        )


# ---------------------------------------------------------------------------
# (d) Noether-charge decay Q_n = (1-gamma)^n Q_0, exact (F5 §4.1)
# ---------------------------------------------------------------------------


def test_noether_charge_exact_decay():
    gamma = 0.05
    hat = MexicanHatPotential(lam=1.0, f=1.0, k_spec=None)
    model = clu_with_potential(
        hat, dim=2, kinetic_mode="newtonian_learned", inertia=(1.0, 1.0)
    )
    q0 = jnp.array([1.0, 0.0])
    p0 = jnp.array([0.15, 0.3])  # tangential + radial write
    traj = rollout_from(model, q0, p0, steps=1000, dt=DT, gamma=gamma)
    Q = np.asarray(noether_charge(traj, dim=2))
    n = np.arange(len(Q))
    Q_pred = (1.0 - gamma) ** n * Q[0]
    err = np.max(np.abs(Q - Q_pred)) / abs(Q[0])
    assert err < 1e-9, f"Noether decay-law error {err:.3e} (F5 App-N: 9.3e-16)"


# ---------------------------------------------------------------------------
# (e) Kinetic isotropy falsifiable: equal channel inertial masses conserve Q;
#     unequal masses drift O(1) (F5 §4.1; App-N: 3.0e-14 vs 2.6)
# ---------------------------------------------------------------------------


def test_kinetic_isotropy_requirement():
    hat = MexicanHatPotential(lam=1.0, f=1.0, k_spec=None)
    q0 = jnp.array([1.0, 0.0])
    p0 = jnp.array([0.2, 0.3])
    steps = 20000

    drifts = {}
    for label, inertia in [("equal", (1.0, 1.0)), ("unequal", (1.0, 2.0))]:
        model = clu_with_potential(
            hat, dim=2, kinetic_mode="newtonian_learned", inertia=inertia
        )
        traj = rollout_from(model, q0, p0, steps=steps, dt=DT, gamma=0.0)
        Q = np.asarray(noether_charge(traj, dim=2))
        drifts[label] = np.max(np.abs(Q - Q[0])) / abs(Q[0])

    assert drifts["equal"] < 1e-10, f"equal-mass drift {drifts['equal']:.3e}"
    assert drifts["unequal"] > 0.05, f"unequal-mass drift {drifts['unequal']:.3e}"
    assert drifts["unequal"] / drifts["equal"] > 1e6


# ---------------------------------------------------------------------------
# (f) Spectrum probe + GMOR tilt: mu^2 = {delta*n^2/(M f^2), 8*lam*f^2/M}
#     at the tilted vacuum (F5 §3.3c), and settle() finds the vacuum circle
# ---------------------------------------------------------------------------


def test_spectrum_probe_gmor_tilt():
    lam, f, inertia, delta, n_tilt = 0.7, 1.2, 1.3, 0.01, 2
    hat = MexicanHatPotential(lam=lam, f=f, k_spec=None)
    tilted = TiltedPotential(hat, tilt_delta=delta, tilt_n=n_tilt)
    model = clu_with_potential(
        tilted, dim=2, kinetic_mode="newtonian_learned", inertia=(inertia, inertia)
    )

    # Analytic minimum of delta*cos(2*theta): theta = pi/2 -> q* = (0, f)
    q_star = jnp.array([0.0, f])
    probe = spectrum_probe(model, q_star)
    assert float(probe.grad_norm) < 1e-10

    mu_ang_pred = delta * n_tilt**2 / (inertia * f**2)  # GMOR pseudo-Goldstone
    mu_rad_pred = 8.0 * lam * f**2 / inertia
    assert np.allclose(
        np.asarray(probe.mu_sq), [mu_ang_pred, mu_rad_pred], rtol=1e-8
    ), f"mu^2 = {np.asarray(probe.mu_sq)} vs pred {[mu_ang_pred, mu_rad_pred]}"

    # Untilted: the angular direction is EXACTLY flat anywhere on the circle
    model0 = clu_with_potential(
        hat, dim=2, kinetic_mode="newtonian_learned", inertia=(inertia, inertia)
    )
    phi = 0.7
    q_on = jnp.array([f * jnp.cos(phi), f * jnp.sin(phi)])
    probe0 = spectrum_probe(model0, q_on)
    assert abs(float(probe0.mu_sq[0])) < 1e-10
    assert abs(float(probe0.mu_sq[1]) - mu_rad_pred) < 1e-8 * mu_rad_pred

    # settle() lands on the vacuum circle (radial mode fully damped)
    q_s, _ = settle(model0, jnp.array([1.5, 0.4]), dt=DT, gamma=0.1, steps=3000)
    r_s = float(jnp.sqrt(q_s[0] ** 2 + q_s[1] ** 2))
    assert abs(r_s - f) < 1e-6


# ---------------------------------------------------------------------------
# SO(2)-invariant learned potential + tied channel inertial masses
# ---------------------------------------------------------------------------


def _rotate_channel(q: jnp.ndarray, phi: float) -> jnp.ndarray:
    c, s = jnp.cos(phi), jnp.sin(phi)
    return q.at[0].set(c * q[0] - s * q[1]).at[1].set(s * q[0] + c * q[1])


def test_so2_invariant_potential_exact_invariance():
    pot = SO2InvariantPotential(dim=4, hidden=16, key=jax.random.PRNGKey(7))
    q = jax.random.normal(jax.random.PRNGKey(8), (4,))
    for phi in (0.7317, 2.0, -1.234):
        dv = abs(float(pot(_rotate_channel(q, phi)) - pot(q)))
        assert dv < 1e-10, f"invariance violated: dV = {dv:.3e} at phi={phi}"

    # Tilt breaks it (except at its own periodicity 2*pi/n)
    tilted = TiltedPotential(pot, tilt_delta=0.3, tilt_n=3)
    q1 = jnp.array([1.0, 0.0, 0.1, -0.2])
    assert abs(float(tilted(_rotate_channel(q1, 0.5)) - tilted(q1))) > 0.1
    assert abs(float(tilted(_rotate_channel(q1, 2 * math.pi / 3)) - tilted(q1))) < 1e-10


def test_tied_channel_inertial_mass():
    model = CHLU(
        dim=4,
        hidden=8,
        kinetic_mode="newtonian_learned",
        potential_type="so2_invariant",
        tie_channel_mass=True,
        key=jax.random.PRNGKey(3),
    )
    mv = model.mass_vector()
    assert float(mv[0]) == float(mv[1]), "channel inertial masses not tied"
    m_eff = model.effective_inertia()
    assert float(m_eff[0]) == float(m_eff[1])

    untied = CHLU(
        dim=4,
        hidden=8,
        kinetic_mode="newtonian_learned",
        potential_type="so2_invariant",
        tie_channel_mass=False,
        key=jax.random.PRNGKey(3),
    )
    mv_u = untied.mass_vector()
    assert float(mv_u[0]) != float(mv_u[1]), "broken-isotropy switch has no effect"


def test_so2_gradients_flow():
    model = CHLU(
        dim=4,
        hidden=8,
        kinetic_mode="newtonian_learned",
        potential_type="so2_invariant",
        tie_channel_mass=True,
        key=jax.random.PRNGKey(0),
    )
    q0 = jnp.array([1.0, 0.0, 0.1, -0.1])
    p0 = jnp.zeros(4)

    def loss_fn(m):
        traj = m(q0, p0, steps=10, dt=DT)
        return jnp.sum(traj**2)

    grads = eqx.filter_grad(loss_fn)(model)
    leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_array))
    assert any(jnp.any(leaf != 0) for leaf in leaves if leaf.size > 0)


# ---------------------------------------------------------------------------
# Dataset + mode-coordinate helpers
# ---------------------------------------------------------------------------


def test_circle_vacuum_dataset():
    data = generate_circle_vacuum(
        jax.random.PRNGKey(0), n_points=8, seq_len=5, dim=4, radius=1.5
    )
    assert data.shape == (8, 5, 8)
    # Constant in time
    assert jnp.max(jnp.abs(data - data[:, :1, :])) == 0.0
    # On the circle, spectators and momenta zero
    r = jnp.sqrt(data[:, 0, 0] ** 2 + data[:, 0, 1] ** 2)
    assert jnp.max(jnp.abs(r - 1.5)) < 1e-12
    assert jnp.max(jnp.abs(data[:, :, 2:4])) == 0.0
    assert jnp.max(jnp.abs(data[:, :, 4:])) == 0.0


def test_mode_coordinates_roundtrip():
    # Canonical projections must recover a hand-planted mode displacement
    pot = QuadraticPotential(K=jnp.diag(jnp.array([0.09, 0.25])))
    model = clu_with_potential(
        pot, dim=2, kinetic_mode="newtonian_learned", inertia=(0.5, 2.0)
    )
    probe = spectrum_probe(model, jnp.zeros(2))
    res = perturb_and_track(
        model,
        probe,
        mode_idx=1,
        kick=0.2,
        kick_type="position",
        steps=1,
        dt=DT,
        gamma=0.0,
    )
    d, pc = mode_coordinates(probe, res["traj"])
    assert abs(float(d[0, 1]) - 0.2) < 1e-12  # kicked mode reads back the kick
    assert abs(float(d[0, 0])) < 1e-12  # other mode untouched
    assert abs(float(res["retention"][0]) - 1.0) < 1e-12
