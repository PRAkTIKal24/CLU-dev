"""check_02 -- THE CRR SIDE: what the retention law says when settling is switched ON.

Everything here is closed-form arithmetic on the damped velocity-Verlet map, plus one
Jacobian taken from the PRODUCTION integrator (`chlu.core.integrators.velocity_verlet_step`).
No scored run, no store, no GPU.  Runs in a few seconds.

Probes
------
G  the production integrator's Jacobian == the shorts' map:
   tr J = (2-gamma)(1-x/2),  det J = 1-gamma,   x = (eta*mu)^2
H  n_half = ln2 / (-ln |lambda_slow|); the overdamped branch, and D17's
   d ln n_half / d ln gamma = 1 + gamma/(2-gamma) at FIXED h
I  the sign trap: the same derivative at fixed h/h* returns MINUS the same number
   (EVIDENCE.md 4.1 defect 2 reports -1.026 against +1.026; reproduced here)
J  invert 1 + gamma/(2-gamma) = 1.1385 and print the derived slope on the RECORDED
   gamma grid (test/s0/results.json -> pins.gammas)
K  the underdamped floor is CURVATURE-INDEPENDENT: |lambda| = sqrt(1-gamma) exactly,
   so n_half = 2 ln2 / (-ln(1-gamma)) and D18 is its exact log-derivative
L  the '+-factor 1.10' claim: 10 ** (B1 median |log10 ratio|)
M  D14: the 2-alpha Hessian lift is ALGEBRAIC in the recorded check (tautology), and the
   finite-difference check that replaced it is not

Run:  OMP_NUM_THREADS=2 python checks/check_02_crr.py
"""
import json
import os

import numpy as np

LN2 = float(np.log(2.0))
RES = {}
EV = os.environ.get("W12_EVIDENCE",
                    r"C:\chlu\evidence\outputs\_thinkpad\w6\w6-potentials")


def _banner(s):
    print("\n" + "=" * 78 + "\n" + s + "\n" + "=" * 78)


# ------------------------------------------------------------------ map -------
def tr_det(gamma, x):
    """x = (h*mu)^2 = (eta*mu)^2, the dimensionless step-squared-times-curvature."""
    return (2.0 - gamma) * (1.0 - x / 2.0), 1.0 - gamma


def lam_slow(gamma, x):
    tr, det = tr_det(gamma, x)
    disc = tr * tr - 4.0 * det
    if disc >= 0.0:
        s = np.sqrt(disc)
        return max(abs((tr + s) / 2.0), abs((tr - s) / 2.0)), True
    return float(np.sqrt(det)), False                  # complex pair, modulus sqrt(det)


def n_half(gamma, x):
    lam, real = lam_slow(gamma, x)
    return LN2 / (-np.log(lam)), real


def dlog_dloggamma(gamma, x_of_gamma, rel=1e-5):
    h = gamma * rel
    f = lambda g: np.log(n_half(g, x_of_gamma(g))[0])
    return (f(gamma + h) - f(gamma - h)) / (np.log(gamma + h) - np.log(gamma - h))


# ------------------------------------------------------------------ G ---------
def probe_G():
    try:
        import jax
        import jax.numpy as jnp
        from chlu.core.integrators import velocity_verlet_step
    except Exception as e:                                        # pragma: no cover
        print("DECLARED NOT-RUN: jax/chlu integrator unavailable (%s)" % e)
        RES["G_pass"] = None
        return
    rows = []
    for mass, gamma, eta in ((1.0, 0.05, 0.05), (0.1, 0.2, 0.05), (10.0, 0.01, 0.1)):
        k = 1.0 / mass                       # H = p^2/2 + (k/2) q^2 in the reduced form
        H = lambda q, p: 0.5 * jnp.sum(p * p) + 0.5 * k * jnp.sum(q * q)
        step = lambda qp: jnp.concatenate(
            velocity_verlet_step(H, qp[:1], qp[1:], eta, gamma))
        J = np.asarray(jax.jacfwd(step)(jnp.array([0.3, -0.7])), np.float64)
        x = eta * eta * k
        tr_p, det_p = tr_det(gamma, x)
        d = max(abs(np.trace(J) - tr_p), abs(np.linalg.det(J) - det_p))
        rows.append({"mass": mass, "gamma": gamma, "eta": eta, "x": float(x),
                     "tr_measured": float(np.trace(J)), "tr_predicted": float(tr_p),
                     "det_measured": float(np.linalg.det(J)), "det_predicted": float(det_p),
                     "max_abs_diff": float(d)})
        print("  mass=%-7g gamma=%-6g eta=%-5g   tr %.9f vs %.9f   det %.9f vs %.9f   "
              "max|diff|=%.2e" % (mass, gamma, eta, np.trace(J), tr_p,
                                  np.linalg.det(J), det_p, d))
    mx = max(r["max_abs_diff"] for r in rows)
    RES["G_rows"] = rows
    RES["G_max_abs_diff"] = mx
    RES["G_pass"] = bool(mx < 1e-5)
    print("  max over rows = %.3e   (wave 6 recorded integrator_check.max_abs_diff ="
          " 1.884681927188936e-08, tol 1e-05)" % mx)
    print("VERDICT  the production integrator IS the shorts' damped-Verlet map: %s"
          % RES["G_pass"])


# ------------------------------------------------------------------ H ---------
def probe_H():
    rows = []
    for g in (0.05, 0.1, 0.2, 0.243303, 0.35, 0.5):
        for x in (1e-6, 1e-4, 1e-3):
            meas = dlog_dloggamma(g, lambda gg, x=x: x)
            der = 1.0 + g / (2.0 - g)
            _, real = n_half(g, x)
            rows.append({"gamma": g, "x": x, "measured": float(meas),
                         "derived_1_plus_g_over_2mg": float(der),
                         "abs_diff": float(abs(meas - der)), "overdamped": bool(real)})
            print("  gamma=%-9g x=%-8g  d ln n/d ln g = %+0.6f   1+g/(2-g) = %+0.6f"
                  "   |diff|=%.2e   overdamped=%s"
                  % (g, x, meas, der, abs(meas - der), real))
    small = [r for r in rows if r["x"] == 1e-6]
    mx = max(r["abs_diff"] for r in small)
    RES["H_rows"] = rows
    RES["H_max_abs_diff_at_x_1e-6"] = mx
    RES["H_pass"] = bool(mx < 1e-3)
    print("  max |measured - derived| over the x = 1e-6 rows = %.2e" % mx)
    print("VERDICT  D17 is the EXACT log-derivative of the exact slow root in the")
    print("         small-step limit, and the error grows with x (step x curvature): %s"
          % RES["H_pass"])


# ------------------------------------------------------------------ I ---------
def xstar(gamma):
    """The exact collision (critical) point: (2-g)(1-x/2) = 2 sqrt(1-g)."""
    return 2.0 * (1.0 - 2.0 * np.sqrt(1.0 - gamma) / (2.0 - gamma))


def probe_I():
    rows = []
    for g in (0.05, 0.1, 0.2):
        for c in (0.2, 0.3, 0.5):
            meas = dlog_dloggamma(g, lambda gg, c=c: c * c * xstar(gg))
            der = 1.0 + g / (2.0 - g)
            rows.append({"gamma": g, "h_over_hstar": c, "measured_at_fixed_h_over_hstar":
                         float(meas), "correct_at_fixed_h": float(der)})
            print("  gamma=%-6g h/h*=%-5g   at fixed h/h*: %+0.4f     at fixed h: %+0.4f"
                  % (g, c, meas, der))
    RES["I_rows"] = rows
    hit = [r for r in rows if r["gamma"] == 0.05]
    RES["I_gamma005_fixed_ratio"] = float(np.mean([r["measured_at_fixed_h_over_hstar"]
                                                   for r in hit]))
    RES["I_pass"] = bool(abs(RES["I_gamma005_fixed_ratio"] + 1.026) < 5e-4)
    print("  at gamma = 0.05 the fixed-h/h* derivative is %+0.4f"
          % RES["I_gamma005_fixed_ratio"])
    print("  EVIDENCE.md 4.1 defect 2 reports exactly -1.026 against +1.026.")
    print("VERDICT  the sign trap reproduces independently here: %s" % RES["I_pass"])


# ------------------------------------------------------------------ J ---------
def probe_J():
    target = 1.1385
    g = 2.0 * (target - 1.0) / target
    RES["J_gamma_for_1_1385"] = float(g)
    print("  1 + gamma/(2-gamma) = %.4f  <=>  gamma = %.6f   "
          "(EVIDENCE.md 4.2: 0.243303)" % (target, g))
    grid = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5]
    try:
        pins = json.load(open(os.path.join(EV, "test", "s0", "results.json")))["pins"]
        grid = list(pins["gammas"])
        print("  recorded grid read from test/s0/results.json -> pins.gammas: %s" % grid)
        print("  pins.eta = %s   pins.masses = %s   gamma_refine = %s (span %s)"
              % (pins["eta"], pins["masses"], pins["gamma_refine"],
                 pins["gamma_refine_span"]))
    except Exception as e:
        print("  DECLARED NOT-RUN: could not read the recorded grid (%s); using the"
              " grid EVIDENCE.md prints" % e)
    der = [1.0 + gg / (2.0 - gg) for gg in grid]
    print("  derived D17 slope on the recorded grid:")
    print("    gamma  " + "  ".join("%7g" % gg for gg in grid))
    print("    slope  " + "  ".join("%7.4f" % d for d in der))
    RES["J_grid"] = grid
    RES["J_derived_slope_on_grid"] = [float(d) for d in der]
    RES["J_gamma_on_grid"] = bool(any(abs(gg - g) < 1e-9 for gg in grid))
    print("  is gamma = %.6f a grid point?  %s" % (g, RES["J_gamma_on_grid"]))
    RES["J_pass"] = bool(abs(g - 0.243303) < 1e-6 and not RES["J_gamma_on_grid"])
    print("VERDICT  +1.1385 sits BETWEEN grid points 0.2 (1.1111) and 0.35 (1.2121);")
    print("         it is an aggregate over per-cell fit windows, not a grid value: %s"
          % RES["J_pass"])


# ------------------------------------------------------------------ K ---------
def d18(gamma):
    return -gamma / ((1.0 - gamma) * (-np.log(1.0 - gamma)))


def probe_K_safe():
    errs = []
    cells = 0
    for g in (0.002, 0.05, 0.2, 0.5):
        for x in (2e-3, 1e-2, 5e-2):
            lam, real = lam_slow(g, x)
            if real:
                continue
            cells += 1
            errs.append(abs(lam - np.sqrt(1.0 - g)))
    RES["K_floor_max_abs_err"] = float(max(errs)) if errs else None
    RES["K_floor_cells"] = cells
    print("  underdamped |lambda_slow| - sqrt(1-gamma):  max %.3e over %d cells"
          % (max(errs), cells))
    rows = []
    for g in (0.002, 0.05, 0.2, 0.5):
        for x in (2e-3, 1e-2, 5e-2):
            if lam_slow(g, x)[1]:
                continue
            meas = dlog_dloggamma(g, lambda gg, x=x: x)
            rows.append({"gamma": g, "x": x, "measured": float(meas),
                         "d18_derived": float(d18(g)),
                         "abs_diff": float(abs(meas - d18(g)))})
    mx = max(r["abs_diff"] for r in rows)
    RES["K_rows"] = rows
    RES["K_d18_max_abs_diff"] = mx
    print("  D18 = -g/[(1-g)(-ln(1-g))] vs the numeric log-derivative of the floor:")
    print("    max |diff| = %.2e over %d cells" % (mx, len(rows)))
    grid = RES.get("J_grid", [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5])
    print("  derived D18 on the recorded grid   (EVIDENCE.md 4.3 prints the same row):")
    print("    gamma  " + "  ".join("%7g" % gg for gg in grid))
    print("    slope  " + "  ".join("%7.4f" % d18(gg) for gg in grid))
    RES["K_d18_on_grid"] = {str(gg): float(d18(gg)) for gg in grid}

    def invert(target, lo=1e-6, hi=0.999999):
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if d18(mid) > target:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    g_der = invert(-1.0050)
    g_meas = invert(-1.1090)
    RES["K_gamma_for_derived_minus1_0050"] = float(g_der)
    RES["K_gamma_for_measured_minus1_1090"] = float(g_meas)
    print("  D18 = -1.0050 at gamma = %.6f   (EVIDENCE.md 4.3: 0.009917)" % g_der)
    print("  D18 = -1.1090 at gamma = %.6f   (EVIDENCE.md 4.3: 0.184104)" % g_meas)
    RES["K_pass"] = bool(mx < 1e-6 and abs(g_der - 0.009917) < 1e-5
                         and abs(g_meas - 0.184104) < 1e-5)
    print("VERDICT  D18 is exact, the floor is curvature-INDEPENDENT, and the two quoted")
    print("         D18 numbers sit at gamma = 0.0099 and gamma = 0.1841 -- different")
    print("         points of the same curve: %s" % RES["K_pass"])


# ------------------------------------------------------------------ L ---------
def probe_L():
    b1 = {"seed0": 0.0433, "seed1": 0.0435, "seed2": 0.0431}
    floor = 0.0349
    print("  B1 median |log10(n_meas / n_pred)| (RESULTS-LEDGER.md):")
    out = {}
    for k, v in b1.items():
        out[k] = float(10.0 ** v)
        print("    %s  %.4f  ->  ratio %.4f" % (k, v, 10.0 ** v))
    print("  instrument floor 0.0349 -> ratio %.4f  (the measurement's own error floor;"
          " results.json -> instrument_floor.median_abs_log10 = 0.03491890826972267)"
          % (10.0 ** floor))
    print("  B1's declared bar 0.10 -> ratio %.4f" % (10.0 ** 0.10))
    RES["L_ratios"] = out
    RES["L_floor_ratio"] = float(10.0 ** floor)
    RES["L_bar_ratio"] = float(10.0 ** 0.10)
    RES["L_pass"] = bool(abs(max(out.values()) - 1.1054) < 5e-4)
    print("VERDICT  'a factor 1.10' = 10**0.0433 = %.4f, i.e. 1.105 to 3 d.p.  The"
          % out["seed0"])
    print("         prediction is ~10.5 per cent off at the median corner, and the")
    print("         instrument's own floor is already %.1f per cent of that."
          % (100 * (10 ** floor - 1) / (10 ** 0.0433 - 1)))


# ------------------------------------------------------------------ M ---------
def probe_M():
    print("  D14 recorded check:  hess_eig(alpha) - [hess_eig(0) + 2 alpha]")
    print("  If the confined energy is E(x) + alpha||x||^2 then its Hessian is")
    print("  H + 2 alpha I by ALGEBRA, so the residual is 0 whatever H is. Demonstration")
    print("  on a random symmetric H, 200 draws:")
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(200):
        n = int(rng.integers(2, 12))
        A = rng.normal(size=(n, n))
        H = A + A.T
        alpha = float(rng.uniform(0, 5))
        e0 = np.sort(np.linalg.eigvalsh(H))
        ea = np.sort(np.linalg.eigvalsh(H + 2 * alpha * np.eye(n)))
        worst = max(worst, float(np.abs(ea - (e0 + 2 * alpha)).max()))
    print("    max |residual| over 200 random Hessians = %.3e  -- it CANNOT fail" % worst)
    RES["M_tautology_max_residual"] = worst
    rows = {}
    for s in (0, 1, 2):
        p = os.path.join(EV, "test", "s%d" % s, "results.json")
        try:
            d = json.load(open(p))
        except Exception as e:
            print("    DECLARED NOT-RUN: cannot read %s (%s)" % (p, e))
            continue
        rows["s%d" % s] = {
            "d14_confinement_n": d["d14_confinement"]["n"],
            "d14_confinement_max_abs_residual": d["d14_confinement"]["max_abs_residual"],
            "d14_confinement_law": d["d14_confinement"]["law"],
            "hessian_fd_check_n": d["hessian_fd_check"]["n"],
            "hessian_fd_check_max_abs_residual":
                d["hessian_fd_check"]["max_abs_residual"],
            "hessian_fd_check_eps_fd": d["hessian_fd_check"]["eps_fd"],
        }
        print("    s%d  d14_confinement: n=%d max_abs_residual=%s   |   "
              "hessian_fd_check: n=%d max_abs_residual=%.6e eps=%s"
              % (s, rows["s%d" % s]["d14_confinement_n"],
                 rows["s%d" % s]["d14_confinement_max_abs_residual"],
                 rows["s%d" % s]["hessian_fd_check_n"],
                 rows["s%d" % s]["hessian_fd_check_max_abs_residual"],
                 rows["s%d" % s]["hessian_fd_check_eps_fd"]))
    RES["M_recorded"] = rows
    taut = all(r["d14_confinement_max_abs_residual"] == 0.0 for r in rows.values()) \
        if rows else None
    RES["M_pass"] = bool(taut) if rows else None
    print("VERDICT  d14_confinement is exactly 0.0 on %s cells per seed and can only ever"
          % (rows.get("s0", {}).get("d14_confinement_n", "?")))
    print("         pass.  Quote hessian_fd_check (3.519e-11 / 1.202e-10 / 2.699e-11)"
          " instead: %s" % RES["M_pass"])



# ------------------------------------------------------------------ N ---------
def gamma_crit(x, lo=1e-9, hi=0.999999):
    """The branch boundary: tr^2 = 4 det, i.e. (2-g)^2 (1-x/2)^2 = 4(1-g).

    Below gamma_crit(x) the pair is complex (UNDERDAMPED, the floor branch);
    above it the roots are real (OVERDAMPED, the D17 branch).
    """
    c = (1.0 - x / 2.0) ** 2
    f = lambda g: c * (2.0 - g) ** 2 - 4.0 * (1.0 - g)
    if f(lo) * f(hi) > 0:
        return float("nan")
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def probe_N():
    """D18's discrepancy has EXACTLY ONE free parameter, and it is gamma.

    In the underdamped branch |lambda_slow| = sqrt(1-gamma) exactly (probe K), so
    n_half = 2 ln2 / (-ln(1-gamma)) depends on gamma ALONE -- not on the step, not on
    the curvature.  A measured floor slope of -1.1090 therefore PINS the fit window's
    effective gamma at 0.1841 with no other degree of freedom, and the underdamped
    branch only reaches that gamma above a definite step-times-curvature.
    """
    print("  gamma_crit(x): below it the pair is complex (the floor branch)")
    print("    x        " + "  ".join("%9.2e" % x for x in
                                      (1e-6, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1)))
    print("    gamma_c  " + "  ".join("%9.4f" % gamma_crit(x) for x in
                                      (1e-6, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1)))
    g_meas = RES.get("K_gamma_for_measured_minus1_1090", 0.184104)
    lo, hi = 1e-8, 1.9
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if gamma_crit(mid) < g_meas:
            lo = mid
        else:
            hi = mid
    x_need = 0.5 * (lo + hi)
    eta = 0.05
    print("  for the underdamped branch to REACH gamma = %.6f you need" % g_meas)
    print("    x = (eta*mu)^2 >= %.6f   i.e. eta*mu >= %.4f" % (x_need, np.sqrt(x_need)))
    print("    at the recorded pins.eta = %s that is mu >= %.3f"
          % (eta, np.sqrt(x_need) / eta))
    print("  EVIDENCE.md 4.3 reports gamma*_exact around 0.1185 for the modes in this")
    print("  grid, i.e. BELOW 0.1841 -- the wrong side of the branch.")
    RES["N_gamma_crit_table"] = {("%.0e" % x): float(gamma_crit(x))
                                 for x in (1e-6, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1)}
    RES["N_x_needed_for_measured_gamma"] = float(x_need)
    RES["N_mu_needed_at_eta_0_05"] = float(np.sqrt(x_need) / eta)
    RES["N_pass"] = bool(np.isfinite(x_need) and x_need > 0)
    print("VERDICT  D18's gap is a FIT-WINDOW question with one free parameter, and ONE")
    print("         recorded row (the window's gamma and x) would settle it: %s"
          % RES["N_pass"])


def main():
    _banner("G -- the production integrator is the shorts' map")
    probe_G()
    _banner("H -- D17: the overdamped slope, at FIXED h")
    probe_H()
    _banner("I -- the sign trap: the same derivative at fixed h/h*")
    probe_I()
    _banner("J -- where +1.1385 comes from, against the RECORDED gamma grid")
    probe_J()
    _banner("K -- D18 and the curvature-INDEPENDENT underdamped floor")
    probe_K_safe()
    _banner("L -- what 'parameter-free to a factor 1.10' actually is")
    probe_L()
    _banner("M -- D14: the tautological field, and the check that replaced it")
    probe_M()
    _banner("N -- D18: the floor has ONE free parameter, and it is gamma")
    probe_N()

    _banner("SUMMARY")
    keys = [k for k in RES if k.endswith("_pass")]
    for k in keys:
        print("  %-10s %s" % (k, RES[k]))
    allp = all(RES[k] is not False for k in keys)
    print("  NOTHING FAILED   %s" % allp)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "check_02_crr.json")
    with open(out, "w") as fh:
        json.dump(RES, fh, indent=1, sort_keys=True)
    print("  wrote %s" % out)
    return 0 if allp else 1


if __name__ == "__main__":
    raise SystemExit(main())
