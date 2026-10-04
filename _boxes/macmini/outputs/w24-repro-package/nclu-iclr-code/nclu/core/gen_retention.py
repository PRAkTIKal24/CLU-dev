"""gen retention."""

from __future__ import annotations
import math
from typing import Sequence
import numpy as np

LIFETIME_DEAD_EPS = 1e-06
PAPER = {
    "source": "F_v2MemShort_bento_final2.pdf -- 'The Price of a Flat Direction: Transverse Curvature sets Retention in a Trained Recurrent Memory'",
    "sha256": "5a0fc126c07975a41ef4a8a139594b010ae95f2c9aa2e1e39d2ffbd2e48b4e11",
    "sections": "Sec. 3 (bands), Sec. 4.1 (results), App. C Eqs. (3)-(6), App. M",
    "floor_steps_at_gamma_0p05": 27.03,
    "crossover_is_gamma_over_2": True,
    "stability_h": 2.0,
    "onset_exponent_predicted": 0.5,
    "onset_exponent_fitted": 0.5165,
    "overdamped_slope_predicted": -1.0,
    "overdamped_slope_fitted": -0.985,
    "declared_negative_on_learned_multi_atom_store": "App. F 'Tilt Limitations' / App. K: explicit breaking on a learned, multi-atom store failed to yield predictable pseudo-Goldstone scaling.",
}


def h_of(omega2: float, eps: float) -> float:
    """``h = omega * eps``, the leapfrog stability parameter."""
    return float(math.sqrt(max(float(omega2), 0.0)) * float(eps))


def verlet_block(omega2: float, eps: float, gamma: float, mass: float = 1.0) -> np.ndarray:
    e, g, M = (float(eps), float(gamma), float(mass))
    k = M * float(omega2)
    K = np.array([[1.0, 0.0], [-0.5 * e * k, 1.0]], np.float64)
    D = np.array([[1.0, e / M], [0.0, 1.0]], np.float64)
    G = np.array([[1.0, 0.0], [0.0, 1.0 - g]], np.float64)
    return G @ K @ D @ K


def spectrum(omega2: float, eps: float, gamma: float) -> tuple:
    g = float(gamma)
    s = math.sqrt(1.0 - g)
    h = h_of(omega2, eps)
    u = 1.0 - 0.5 * h * h
    left = (1.0 - s) * (1.0 - s) - 0.5 * (2.0 - g) * h * h
    right = (2.0 - g) * u + 2.0 * s
    disc = left * right
    t = (2.0 - g) * u
    if disc >= 0.0:
        r = math.sqrt(disc)
        big = 0.5 * (t + r) if t >= 0.0 else 0.5 * (t - r)
        small = (1.0 - g) / big if big != 0.0 else 0.5 * (t - r)
        return (big, small, disc) if t >= 0.0 else (small, big, disc)
    r = math.sqrt(-disc)
    return (complex(0.5 * t, 0.5 * r), complex(0.5 * t, -0.5 * r), disc)


def crossover_h(gamma: float) -> float:
    """``h*``, the exceptional point.  App. C Eq. (5).  ``= gamma/2 + O(gamma^2)``."""
    g = float(gamma)
    return math.sqrt(2.0) * (1.0 - math.sqrt(1.0 - g)) / math.sqrt(2.0 - g)


def stability_h(gamma: float) -> float:
    g = float(gamma)
    return math.sqrt(2.0) * (1.0 + math.sqrt(1.0 - g)) / math.sqrt(2.0 - g)


def band(omega2: float, eps: float, gamma: float) -> str:
    """Which of the paper's three retention bands this mode is in (Sec. 3)."""
    if float(omega2) <= 0.0:
        return "latch"
    h = h_of(omega2, eps)
    if h >= 2.0:
        return "unstable"
    return "overdamped" if h < crossover_h(gamma) else "underdamped"


def retention_factor(omega2: float, eps: float, gamma: float, mass: float = 1.0) -> float:
    lp, lm, _ = spectrum(omega2, eps, gamma)
    return float(max(abs(lp), abs(lm)))


def n_half_closed_form(omega2: float, eps: float, gamma: float, mass: float = 1.0) -> float:
    rho = retention_factor(omega2, eps, gamma, mass)
    if rho >= 1.0:
        return math.inf
    return math.log(2.0) / -math.log(rho)


def n_half_paper_asymptotic(omega2: float, eps: float, gamma: float) -> float:
    h = h_of(omega2, eps)
    if h <= 0.0:
        return math.inf
    g = float(gamma)
    return 2.0 * g * math.log(2.0) / ((2.0 - g) * h * h)


def floor_half_life(gamma: float) -> float:
    """The curvature-independent saturation floor, ``2 ln2 / (-ln(1-gamma))`` (Sec. 3)."""
    return 2.0 * math.log(2.0) / -math.log(1.0 - float(gamma))


def latch_displacement(q0: float, p0: float, eps: float, gamma: float, mass: float = 1.0) -> float:
    """The latch's frozen displacement ``q_inf = q_0 + eps p_0 / (gamma M)`` (Sec. 3)."""
    return float(q0) + float(eps) * float(p0) / (float(gamma) * float(mass))


def omega2_for_half_life(n_half: float, eps: float, gamma: float, mass: float = 1.0) -> float:
    if not math.isfinite(n_half):
        return 0.0
    fl = floor_half_life(gamma)
    if float(n_half) < fl:
        raise ValueError(
            f"requested half-life {n_half} is below the curvature-independent floor {fl:.6f} at gamma={gamma}; the law saturates there and no omega2 reaches it"
        )
    lo, hi = (0.0, (crossover_h(gamma) / float(eps)) ** 2)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if n_half_closed_form(mid, eps, gamma, mass) > float(n_half):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def refusal_horizon_half_lives(dead_eps: float = LIFETIME_DEAD_EPS) -> float:
    return math.log(float(dead_eps)) / math.log(0.5)


def simulate(
    omega2: float,
    eps: float,
    gamma: float,
    n_steps: int,
    q0: float = 1.0,
    p0: float = 0.0,
    mass: float = 1.0,
) -> dict:
    A = verlet_block(omega2, eps, gamma, mass)
    z = np.array([float(q0), float(p0)], np.float64)
    Q = np.empty(int(n_steps) + 1, np.float64)
    P = np.empty(int(n_steps) + 1, np.float64)
    Q[0], P[0] = z
    for n in range(int(n_steps)):
        z = A @ z
        Q[n + 1], P[n + 1] = z
    return {"q": Q, "p": P, "A": A, "band": band(omega2, eps, gamma), "h": h_of(omega2, eps)}


def n_half_simulated(
    omega2: float,
    eps: float,
    gamma: float,
    n_steps: int = 400000,
    q0: float = 1.0,
    p0: float = 0.0,
    mass: float = 1.0,
) -> dict:
    tr = simulate(omega2, eps, gamma, n_steps, q0, p0, mass)
    Q, P = (tr["q"], tr["p"])
    a = np.abs(Q)
    cross = np.flatnonzero(a <= 0.5 * abs(q0))
    first = float(cross[0]) if cross.size else math.inf
    nrm = np.sqrt(Q * Q + P * P)
    lo = int(0.5 * len(nrm))
    hi = int(len(nrm))
    m = nrm[lo:hi] > 0
    if m.sum() > 8:
        x = np.arange(lo, hi, dtype=np.float64)[m]
        y = np.log(nrm[lo:hi][m])
        y = np.maximum.accumulate(y[::-1])[::-1]
        sl = float(np.polyfit(x, y, 1)[0])
        env = math.log(2.0) / -sl if sl < 0 else math.inf
    else:
        env = math.nan
    return {
        "first_crossing": first,
        "envelope": env,
        "band": tr["band"],
        "h": tr["h"],
        "closed_form": n_half_closed_form(omega2, eps, gamma, mass),
        "paper_asymptotic": n_half_paper_asymptotic(omega2, eps, gamma),
    }


def assert_reparameterisation_invariance(
    eps: float,
    gamma: float,
    half_lives: Sequence[float] = (0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 19.9316, 30.0),
    n_half_targets: Sequence[float] = (60.0, 400.0, 5000.0),
    tol: float = 1e-09,
) -> dict:
    rows, worst = ([], 0.0)
    for nh in n_half_targets:
        wb = omega2_for_half_life(nh, eps, gamma)
        rho = retention_factor(wb, eps, gamma)
        got = [rho ** (float(t) * nh) for t in half_lives]
        want = [2.0 ** (-float(t)) for t in half_lives]
        d = float(max((abs(a - b) for a, b in zip(got, want))))
        worst = max(worst, d)
        rows.append(
            {
                "n_half_target": float(nh),
                "omega2": float(wb),
                "h": h_of(wb, eps),
                "band": band(wb, eps, gamma),
                "rho": float(rho),
                "max_abs_dev_from_2_pow_minus_t": d,
            }
        )
    if worst > tol:
        raise AssertionError(
            f"the half-life reparameterisation does not collapse: worst deviation {worst:g} > {tol:g}. The claim 'omega2 only sets the clock rate' would be FALSE."
        )
    return {
        "rows": rows,
        "half_lives": [float(t) for t in half_lives],
        "worst_abs_dev": worst,
        "tol": float(tol),
        "statement": f"lifetime(t * n_half) = 2**-t for every curvature, to {worst:.3g}; omega2 is a pure time reparameterisation and cannot move the refusal horizon relative to the composition horizon",
    }


def self_check(eps: float = 0.05, gamma: float = 0.05) -> dict:
    out = {"eps": float(eps), "gamma": float(gamma)}
    g = float(gamma)
    worst_det, worst_tr = (0.0, 0.0)
    for wb in np.logspace(-12, 1.5, 40):
        A = verlet_block(float(wb), eps, g)
        h = h_of(float(wb), eps)
        worst_det = max(worst_det, abs(float(np.linalg.det(A)) - (1.0 - g)))
        worst_tr = max(worst_tr, abs(float(A[0, 0] + A[1, 1]) - (2.0 - g) * (1.0 - 0.5 * h * h)))
    out["block_det_max_abs_err"] = worst_det
    out["block_trace_max_abs_err"] = worst_tr
    if worst_det > 1e-12 or worst_tr > 1e-12:
        raise AssertionError("the Verlet block does not satisfy App. C Eq. (3)'s det/trace")
    fl = floor_half_life(0.05)
    out["floor_at_gamma_0p05"] = fl
    out["paper_floor_at_gamma_0p05"] = PAPER["floor_steps_at_gamma_0p05"]
    out["floor_abs_err_vs_paper"] = abs(fl - PAPER["floor_steps_at_gamma_0p05"])
    if out["floor_abs_err_vs_paper"] > 0.005:
        raise AssertionError(
            f"the transcribed floor {fl:.4f} does not reproduce the paper's printed {PAPER['floor_steps_at_gamma_0p05']} at gamma=0.05"
        )
    hs = crossover_h(0.05)
    und = [n_half_closed_form((x / 0.05) ** 2, 0.05, 0.05) for x in np.linspace(1.5 * hs, 1.9, 25)]
    out["floor_max_dev_over_underdamped_band"] = float(max((abs(u - fl) for u in und)))
    if out["floor_max_dev_over_underdamped_band"] > 1e-09:
        raise AssertionError("the underdamped half-life is not curvature-independent")
    out["crossover_h"] = {}
    for gg in (0.5, 0.2, 0.05, 0.01, 0.001):
        hs = crossover_h(gg)
        out["crossover_h"][str(gg)] = {
            "h_star": hs,
            "gamma_over_2": 0.5 * gg,
            "rel_err": abs(hs - 0.5 * gg) / (0.5 * gg),
        }
    if out["crossover_h"]["0.001"]["rel_err"] > 0.001:
        raise AssertionError("h* does not approach gamma/2")
    out["crossover_delta_at_h_star"] = float(
        max(
            (
                abs(spectrum((crossover_h(gg) / eps) ** 2, eps, gg)[2])
                for gg in (0.5, 0.2, 0.05, 0.01)
            )
        )
    )
    if out["crossover_delta_at_h_star"] > 1e-12:
        raise AssertionError("Delta does not vanish at h*")
    st = {}
    for gg in (0.5, 0.2, 0.05, 0.01):
        A = verlet_block((2.0 / eps) ** 2, eps, gg)
        st[str(gg)] = float(1.0 + (A[0, 0] + A[1, 1]) + np.linalg.det(A))
    out["one_plus_tr_plus_det_at_h_eq_2"] = st
    if max((abs(v) for v in st.values())) > 1e-11:
        raise AssertionError("an eigenvalue does not reach -1 at h = 2")
    hs = crossover_h(g)
    wbs = np.logspace(math.log10((0.001 * hs / eps) ** 2), math.log10((0.3 * hs / eps) ** 2), 40)
    nh = np.array([n_half_closed_form(float(w), eps, g) for w in wbs])
    sl = float(np.polyfit(np.log(wbs), np.log(nh), 1)[0])
    out["overdamped_slope_log_nhalf_vs_log_omega2"] = sl
    out["paper_overdamped_slope_fitted"] = PAPER["overdamped_slope_fitted"]
    if abs(sl + 1.0) > 0.002:
        raise AssertionError(f"the overdamped slope is {sl:.5f}, not -1")
    rel = np.abs(np.array([n_half_paper_asymptotic(float(w), eps, g) for w in wbs]) / nh - 1.0)
    out["paper_asymptotic_max_rel_dev_deep_overdamped"] = float(rel.min())
    out["paper_asymptotic_rel_dev_at_0p30_h_star"] = float(rel[-1])
    d = np.logspace(-8, -4, 30)
    phi = []
    for dd in d:
        lp, _, _ = spectrum(((hs + dd) / eps) ** 2, eps, g)
        phi.append(abs(math.atan2(lp.imag, lp.real)) if isinstance(lp, complex) else 0.0)
    phi = np.array(phi)
    m = phi > 0
    out["onset_exponent"] = (
        float(np.polyfit(np.log(d[m]), np.log(phi[m]), 1)[0]) if m.sum() > 4 else math.nan
    )
    out["paper_onset_exponent_fitted"] = PAPER["onset_exponent_fitted"]
    if not abs(out["onset_exponent"] - 0.5) < 0.005:
        raise AssertionError(f"the onset exponent is {out['onset_exponent']}, not 1/2")
    lp, _, _ = spectrum((0.5 * hs / eps) ** 2, eps, g)
    out["phase_below_h_star_is_exactly_zero"] = bool(not isinstance(lp, complex))
    out["latch_rho"] = retention_factor(0.0, eps, g)
    out["latch_n_half"] = n_half_closed_form(0.0, eps, g)
    tr = simulate(0.0, eps, g, 20000, q0=0.3, p0=0.7)
    qinf = latch_displacement(0.3, 0.7, eps, g)
    out["latch_frozen_displacement_abs_err"] = abs(float(tr["q"][-1]) - qinf)
    if out["latch_rho"] != 1.0 or not math.isinf(out["latch_n_half"]):
        raise AssertionError("the latch does not have rho = 1 / infinite half-life")
    if out["latch_frozen_displacement_abs_err"] > 1e-12:
        raise AssertionError("the latch does not freeze at q_0 + eps p_0/(gamma M)")
    from nclu.core.gen_field import LIFETIME_EPS as _FLE

    out["gen_field_LIFETIME_EPS"] = float(_FLE)
    if float(_FLE) != float(LIFETIME_DEAD_EPS):
        raise AssertionError(
            f"gen_field.LIFETIME_EPS is {_FLE}, not {LIFETIME_DEAD_EPS}; the derived refusal horizon would be wrong"
        )
    out["refusal_horizon_half_lives"] = refusal_horizon_half_lives()
    out["invariance"] = assert_reparameterisation_invariance(eps, g)
    out["PASS"] = True
    return out


def verification_table(
    eps: float = 0.05, gamma: float = 0.05, n_points: int = 25, n_steps: int = 2000000
) -> list:
    hs = crossover_h(gamma)
    hs_grid = np.concatenate(
        [
            np.logspace(math.log10(0.001 * hs), math.log10(0.9 * hs), n_points // 2),
            np.logspace(math.log10(1.2 * hs), math.log10(1.6), n_points - n_points // 2),
        ]
    )
    rows = []
    for h in hs_grid:
        wb = (float(h) / eps) ** 2
        nh = n_half_closed_form(wb, eps, gamma)
        steps = int(min(n_steps, max(64, 40 * nh)))
        sim = n_half_simulated(wb, eps, gamma, n_steps=steps)
        rows.append(
            {
                "omega2": wb,
                "h": float(h),
                "h_over_h_star": float(h) / hs,
                "band": band(wb, eps, gamma),
                "n_half_closed_form": nh,
                "n_half_paper_asymptotic": n_half_paper_asymptotic(wb, eps, gamma),
                "n_half_sim_first_crossing": sim["first_crossing"],
                "n_half_sim_envelope": sim["envelope"],
                "sim_envelope_rel_err": abs(sim["envelope"] / nh - 1.0)
                if math.isfinite(sim["envelope"]) and nh > 0
                else math.nan,
                "crossing_over_envelope": sim["first_crossing"] / sim["envelope"]
                if math.isfinite(sim["envelope"])
                and math.isfinite(sim["first_crossing"])
                and (sim["envelope"] > 0)
                else math.nan,
            }
        )
    return rows
