"""w6 ``hub/retention`` -- THE EXACT-DECAY LAW, transcribed from the paper, then verified.

⛔⛔ READ THIS BEFORE READING A NUMBER THIS MODULE PRODUCES
===========================================================
Nothing in this module is fitted.  Every closed form below is copied out of

    *The Price of a Flat Direction: Transverse Curvature sets Retention in a Trained
    Recurrent Memory* -- ``/c/chlu/shorts/F_v2MemShort_bento_final2.pdf``
    (Sec. 3 "Dynamical mapping and retention bands", Sec. 4.1, App. C Eqs. (3)-(6),
    App. M),

which is the one of the three distinct shorts that carries a retention law.  ⭐ **The law
has ZERO free parameters**: given ``(eps, gamma, omega2)`` every quantity here is
determined.  ⛔ There is no constant in this file that was chosen to make a curve fit, and
:func:`self_check` asserts the transcription against the paper's own printed numbers
(the ``27.03``-step floor at ``gamma = 0.05`` and the ``h* = gamma/2`` crossover) before
anything downstream is allowed to use it.

The paper's object
------------------
One dissipative velocity-Verlet step -- step ``eps``, damping ``gamma`` -- advances the
latent ``(q, p)`` by a kick-drift-kick of ``H = T(p) + V(q)`` followed by the damping map
``p -> (1 - gamma) p``.  The map is conformally symplectic, ``J^T Omega J = (1-gamma)
Omega``.  Restricted to one mass-whitened normal mode of transverse curvature ``omega2``
(the Hessian eigenvalue of ``V`` across the mode), with ``h = omega * eps``, the one-step
map is the 2x2 block of App. C Eq. (3)::

    A = [[      1 - h^2/2          ,        eps        ],
         [ -(1-g) omega2 eps (1 - h^2/4) , (1-g)(1 - h^2/2) ]]

    det A = 1 - gamma      (the per-mode statement of conformal symplecticity)
    tr  A = (2 - gamma) (1 - h^2/2)

⭐⭐ THE WHOLE LAW IN ONE LINE
-----------------------------
All three of the paper's retention bands are the same statement about the **spectral
radius** ``rho = max(|lambda_+|, |lambda_-|)`` of that block::

    n_half = ln 2 / ( -ln rho(A) )

- **The Latch** (``omega2 = 0``, symmetry-protected): ``A`` is upper-triangular with
  eigenvalues ``{1, 1-gamma}``, so ``rho = 1`` and ``n_half = inf``.  The written value
  freezes at ``q_inf = q_0 + eps p_0 / (gamma M)`` (Sec. 3, App. C).
- **The Overdamped Register** (``0 < h < h*``): two real roots in ``(0,1)``; the slow one
  is ``lambda_+ = 1 - (2-gamma) h^2 / (2 gamma) + O(h^4/gamma^3)``, giving the paper's
  Eq. (6) ``n_half ~= 2 gamma ln2 / [(2-gamma)(omega eps)^2]``, i.e. **n_half prop.
  omega^-2**, slope ``-1`` in ``log n_half`` against ``log omega2``.
- **The Underdamped Working Memory** (``h* < h < h_f``): a complex-conjugate pair with
  ``|lambda|^2 = det A = 1-gamma`` **regardless of omega2**, so ``rho = sqrt(1-gamma)`` and
  ``n_half = 2 ln2 / (-ln(1-gamma))`` -- the **curvature-independent floor**.

Boundaries (App. C Eq. (5)), also exact::

    h*  = sqrt(2) (1 - sqrt(1-gamma)) / sqrt(2-gamma)  = gamma/2 + O(gamma^2)
    h_f = sqrt(2) (1 + sqrt(1-gamma)) / sqrt(2-gamma)  = 2 - O(gamma^2)

and an eigenvalue reaches ``-1`` iff ``1 + tr A + det A = 0``, i.e. at ``h = 2`` **exactly,
for every gamma** -- the leapfrog stability limit.

⚠ WHAT THE PAPER ITSELF SAYS ABOUT MY SUBSTRATE, recorded here and not buried
-----------------------------------------------------------------------------
App. F ("Tilt Limitations"): *"the tilt-as-a-dial construction is exclusively verified on
designed geometries.  As detailed in Appendix K, explicit breaking on a learned,
multi-atom store failed to yield predictable pseudo-Goldstone scaling."*  ⛔ The field
store **is** a learned, multi-atom store.  App. G.1 adds that a generic (emergent) MLP
potential's softest curvature is ``5.1/5.9/5.4e-2`` against the designed flat mode's
``2.4e-15`` -- 13-14 orders -- and that the emergent arm lives on discrete washboard
minima where any continuous written coordinate fully relaxes.

⛔ So this module does **not** try to recover a pseudo-Goldstone continuous register out of
the field store's learned potential.  That route is reported absent by the paper and
inventing a fit for it is exactly what this hub's task file forbids.  What the paper does
license on a learned store is its own positioning sentence (App. L.4): *"a physics-
structured associative memory where retention is governed as a designed, **per-item
property controlled by omega2**."*  ⭐ This module therefore instantiates the law as an
**exact per-item decay operator**: item ``i`` is given its own ``omega2_i``, hence its own
``rho_i``, hence its own half-life, and its stored **lifetime** is the retained amplitude

    lifetime_i(n) = lifetime_i(0) * rho_i ** n

which is what ``gen_field.decay_items`` already multiplies into every atom's kernel
weight.  The law is used as a **generator** of the decay, never as a curve fitted to it.

⭐⭐ THE CONSEQUENCE THAT DECIDES THIS HUB, and it is derivable BEFORE any measurement
--------------------------------------------------------------------------------------
``gen_field`` calls an item dead at ``LIFETIME_EPS = 1e-6`` -- a constant declared in that
module, not chosen here -- and refusal (``gap_read``'s ``empty_basin``) reads exactly that
liveness bit.  So the refusal horizon of a decaying item is

    n_refuse = ln(LIFETIME_EPS) / ln(rho) = n_half * ln(LIFETIME_EPS) / ln(1/2)
             = 19.9316 * n_half      for every omega2, exactly.

⛔ **The ratio is a pure constant.** Both horizons carry the same ``1/(-ln rho)`` factor, so
**no choice of curvature can move the refusal horizon relative to the composition horizon**
-- ``omega2`` sets the clock rate and nothing else.  Equivalently: plotted against
``n / n_half``, *every* lifetime in the family is the **same** curve, ``2 ** (-n/n_half)``.
:func:`refusal_horizon_half_lives` returns that constant and
:func:`assert_reparameterisation_invariance` proves the collapse numerically.

That is a structural statement, and it is the reason this hub measures the **composition
cost as a function of decay** rather than the refusal alone: the refusal half is settled in
closed form, and the only open question is where along the decay the composition cost has
already converged to instant eviction's.

Runnable::

    python -m chlu.core.gen_retention            # self-check + the verification table
"""
from __future__ import annotations
import math
from typing import Sequence
import numpy as np
LIFETIME_DEAD_EPS = 1e-6
PAPER = {
    "source": "F_v2MemShort_bento_final2.pdf -- 'The Price of a Flat Direction: Transverse "
              "Curvature sets Retention in a Trained Recurrent Memory'",
    "sha256": "5a0fc126c07975a41ef4a8a139594b010ae95f2c9aa2e1e39d2ffbd2e48b4e11",
    "sections": "Sec. 3 (bands), Sec. 4.1 (results), App. C Eqs. (3)-(6), App. M",
    # Sec. 4.1 / Fig. 1 / Fig. 4: "saturating at a floor of 27.03 steps for gamma = 0.05"
    "floor_steps_at_gamma_0p05": 27.03,
    # Sec. 3 / App. C Eq. (5): the exceptional point h* = gamma/2 + O(gamma^2)
    "crossover_is_gamma_over_2": True,
    # App. C: "the map loses stability at h = 2 exactly, for every gamma"
    "stability_h": 2.0,
    # Sec. 4.1 / App. G: the exceptional-point onset exponent, predicted 1/2, fitted 0.5165
    "onset_exponent_predicted": 0.5,
    "onset_exponent_fitted": 0.5165,
    # Sec. 4.1: the fitted overdamped slope, predicted -1
    "overdamped_slope_predicted": -1.0,
    "overdamped_slope_fitted": -0.985,
    # App. F: the paper's OWN declared negative on a learned, multi-atom store
    "declared_negative_on_learned_multi_atom_store":
        "App. F 'Tilt Limitations' / App. K: explicit breaking on a learned, multi-atom "
        "store failed to yield predictable pseudo-Goldstone scaling.",
}
def h_of(omega2: float, eps: float) -> float:
    """``h = omega * eps``, the leapfrog stability parameter."""
    return float(math.sqrt(max(float(omega2), 0.0)) * float(eps))


def verlet_block(omega2: float, eps: float, gamma: float, mass: float = 1.0) -> np.ndarray:
    """The one-step dissipative velocity-Verlet block ``A`` of App. C Eq. (3).

    ``kick(eps/2) -> drift(eps) -> kick(eps/2)`` on ``H = p^2/(2 mass) + (1/2) mass omega2
    q^2``, followed by ``p -> (1-gamma) p``.  Built from the kick/drift factors rather than
    typed in from the paper's matrix, so ``det A = 1-gamma`` and ``tr A = (2-gamma)(1-h^2/2)``
    are *consequences* here and can be checked against the paper (:func:`self_check` does).
    """
    e, g, M = float(eps), float(gamma), float(mass)
    k = M * float(omega2)                      # the spring constant of this normal mode
    K = np.array([[1.0, 0.0], [-0.5 * e * k, 1.0]], np.float64)   # half kick
    D = np.array([[1.0, e / M], [0.0, 1.0]], np.float64)          # full drift
    G = np.array([[1.0, 0.0], [0.0, 1.0 - g]], np.float64)        # the damping map
    return G @ K @ D @ K


def spectrum(omega2: float, eps: float, gamma: float) -> tuple:
    """⭐ ``(lambda_+, lambda_-, Delta)`` formed CANCELLATION-FREE from ``(omega2, eps, gamma)``.

    With ``s = sqrt(1-gamma)`` and ``u = 1 - h^2/2``, App. C Eq. (4)'s discriminant factorises

        Delta = [ (2-gamma) u - 2 s ] [ (2-gamma) u + 2 s ]
              = [ (1-s)^2 - (2-gamma) h^2 / 2 ] [ (2-gamma) u + 2 s ]

    using ``(2-gamma) - 2 s = 1 + s^2 - 2 s = (1-s)^2`` exactly.  ⭐ Written this way the
    latch is EXACT: at ``h = 0`` the left factor is ``(1-s)^2``, the right ``(1+s)^2``, so
    ``Delta = gamma^2`` and ``lambda_+ = 1`` to the last bit -- which matters, because the
    latch's ``rho = 1`` is the paper's infinite half-life and a ``1 + 1e-15`` there turns an
    exact latch into a spurious growth.  The left factor also vanishes exactly at ``h = h*``,
    so the exceptional point is located to machine precision rather than to ``sqrt(eps)``.
    """
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
        # stable pairing: form the larger root by addition, the other from the product
        big = 0.5 * (t + r) if t >= 0.0 else 0.5 * (t - r)
        small = ((1.0 - g) / big) if big != 0.0 else 0.5 * (t - r)
        return (big, small, disc) if t >= 0.0 else (small, big, disc)
    r = math.sqrt(-disc)
    return complex(0.5 * t, 0.5 * r), complex(0.5 * t, -0.5 * r), disc


def crossover_h(gamma: float) -> float:
    """``h*``, the exceptional point.  App. C Eq. (5).  ``= gamma/2 + O(gamma^2)``."""
    g = float(gamma)
    return math.sqrt(2.0) * (1.0 - math.sqrt(1.0 - g)) / math.sqrt(2.0 - g)


def stability_h(gamma: float) -> float:
    """``h_f``, the upper root of ``Delta = 0``.  App. C Eq. (5).  ``= 2 - O(gamma^2)``.

    ⚠ ⛔ NOT the stability limit: an eigenvalue reaches ``-1`` at ``h = 2`` exactly, for
    every ``gamma`` (App. C).  Above ``h_f`` the pair is real again and one root is below
    ``-1`` in modulus only past ``h = 2``.
    """
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
    """⭐ ``rho(A)`` -- the per-step retention factor.  **This is the decay operator.**

    ``latch`` -> ``1.0``;  ``overdamped`` -> the slow real root;  ``underdamped`` ->
    ``sqrt(1-gamma)`` (curvature-independent, because ``|lambda|^2 = det A``).
    """
    lp, lm, _ = spectrum(omega2, eps, gamma)
    return float(max(abs(lp), abs(lm)))


def n_half_closed_form(omega2: float, eps: float, gamma: float,
                       mass: float = 1.0) -> float:
    """⭐⭐ THE LAW: ``n_half = ln2 / (-ln rho(A))``.  Exact in all three bands."""
    rho = retention_factor(omega2, eps, gamma, mass)
    if rho >= 1.0:
        return math.inf
    return math.log(2.0) / (-math.log(rho))


def n_half_paper_asymptotic(omega2: float, eps: float, gamma: float) -> float:
    """The paper's *quoted* overdamped closed form, Eq. (6).

    ``n_half ~= 2 gamma ln2 / [(2-gamma)(omega eps)^2]``.  Reported BESIDE
    :func:`n_half_closed_form` so the ``O(h^4/gamma^3)`` truncation is visible rather than
    hidden -- ⛔ it is an asymptotic, and this hub never uses it as the operator.
    """
    h = h_of(omega2, eps)
    if h <= 0.0:
        return math.inf
    g = float(gamma)
    return 2.0 * g * math.log(2.0) / ((2.0 - g) * h * h)


def floor_half_life(gamma: float) -> float:
    """The curvature-independent saturation floor, ``2 ln2 / (-ln(1-gamma))`` (Sec. 3)."""
    return 2.0 * math.log(2.0) / (-math.log(1.0 - float(gamma)))


def latch_displacement(q0: float, p0: float, eps: float, gamma: float,
                       mass: float = 1.0) -> float:
    """The latch's frozen displacement ``q_inf = q_0 + eps p_0 / (gamma M)`` (Sec. 3)."""
    return float(q0) + float(eps) * float(p0) / (float(gamma) * float(mass))


def omega2_for_half_life(n_half: float, eps: float, gamma: float,
                         mass: float = 1.0) -> float:
    """Invert the law: the ``omega2`` whose half-life is ``n_half``.

    ⭐ Used to put a *requested* lifetime on an item.  Bisected on the exact
    :func:`n_half_closed_form`, ⛔ never on the asymptotic.  Returns ``0.0`` (the latch) for
    ``n_half = inf`` and raises below the floor, which is unreachable by construction.
    """
    if not math.isfinite(n_half):
        return 0.0
    fl = floor_half_life(gamma)
    if float(n_half) < fl:
        raise ValueError(
            f"requested half-life {n_half} is below the curvature-independent floor "
            f"{fl:.6f} at gamma={gamma}; the law saturates there and no omega2 reaches it")
    lo, hi = 0.0, (crossover_h(gamma) / float(eps)) ** 2
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if n_half_closed_form(mid, eps, gamma, mass) > float(n_half):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def refusal_horizon_half_lives(dead_eps: float = LIFETIME_DEAD_EPS) -> float:
    """⭐⭐ ``n_refuse / n_half = ln(dead_eps) / ln(1/2)`` -- a pure constant, ``19.9316``.

    Both horizons carry the same ``1/(-ln rho)``, so ``omega2`` cancels.  ⛔ **No curvature
    separates the refusal horizon from the composition horizon.**
    """
    return math.log(float(dead_eps)) / math.log(0.5)


def simulate(omega2: float, eps: float, gamma: float, n_steps: int,
             q0: float = 1.0, p0: float = 0.0, mass: float = 1.0) -> dict:
    """Iterate the actual 2x2 map ``n_steps`` times and return the trajectory.

    ⭐ This is what makes the reproduction non-tautological: the half-life is *measured* off
    a simulated trajectory of the paper's own map and compared with the closed form, rather
    than the closed form being re-derived from itself.
    """
    A = verlet_block(omega2, eps, gamma, mass)
    z = np.array([float(q0), float(p0)], np.float64)
    Q = np.empty(int(n_steps) + 1, np.float64)
    P = np.empty(int(n_steps) + 1, np.float64)
    Q[0], P[0] = z
    for n in range(int(n_steps)):
        z = A @ z
        Q[n + 1], P[n + 1] = z
    return {"q": Q, "p": P, "A": A, "band": band(omega2, eps, gamma),
            "h": h_of(omega2, eps)}


def n_half_simulated(omega2: float, eps: float, gamma: float, n_steps: int = 400_000,
                     q0: float = 1.0, p0: float = 0.0, mass: float = 1.0) -> dict:
    """Measure the half-life off the simulated map, two ways, as the paper does.

    ``first_crossing``  the first ``n`` with ``|q_n| <= |q_0|/2`` -- the paper's squares.
    ``envelope``        the half-life of the *envelope* ``||(q_n, p_n)||``, fitted by the
                        late-time log-slope -- the paper's circles.  ⭐ Overdamped the two
                        agree; underdamped they diverge, which is the paper's own
                        second-order signature of a genuine exceptional point (App. G).
    """
    tr = simulate(omega2, eps, gamma, n_steps, q0, p0, mass)
    Q, P = tr["q"], tr["p"]
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
        # the envelope decays as rho^n with a bounded oscillation on top; the UPPER
        # envelope is what "retention" means, so take the running max before fitting
        y = np.maximum.accumulate(y[::-1])[::-1]
        sl = float(np.polyfit(x, y, 1)[0])
        env = math.log(2.0) / (-sl) if sl < 0 else math.inf
    else:
        env = math.nan
    return {"first_crossing": first, "envelope": env, "band": tr["band"], "h": tr["h"],
            "closed_form": n_half_closed_form(omega2, eps, gamma, mass),
            "paper_asymptotic": n_half_paper_asymptotic(omega2, eps, gamma)}


def assert_reparameterisation_invariance(eps: float, gamma: float,
                                         half_lives: Sequence[float] = (0.25, 0.5, 1.0,
                                                                        2.0, 5.0, 10.0,
                                                                        19.9316, 30.0),
                                         n_half_targets: Sequence[float] = (60.0, 400.0,
                                                                            5000.0),
                                         tol: float = 1e-9) -> dict:
    """⭐⭐ Prove the collapse: in HALF-LIFE units every lifetime is the same curve.

    For several genuinely different curvatures (hence genuinely different clock rates), the
    lifetime at ``n = t * n_half`` is ``2 ** -t`` for all of them.  ⛔ This is why ``omega2``
    cannot trade composition against refusal: it is a pure time reparameterisation.
    """
    rows, worst = [], 0.0
    for nh in n_half_targets:
        w2 = omega2_for_half_life(nh, eps, gamma)
        rho = retention_factor(w2, eps, gamma)
        got = [rho ** (float(t) * nh) for t in half_lives]
        want = [2.0 ** (-float(t)) for t in half_lives]
        d = float(max(abs(a - b) for a, b in zip(got, want)))
        worst = max(worst, d)
        rows.append({"n_half_target": float(nh), "omega2": float(w2),
                     "h": h_of(w2, eps), "band": band(w2, eps, gamma),
                     "rho": float(rho), "max_abs_dev_from_2_pow_minus_t": d})
    if worst > tol:
        raise AssertionError(
            f"the half-life reparameterisation does not collapse: worst deviation {worst:g} "
            f"> {tol:g}. The claim 'omega2 only sets the clock rate' would be FALSE.")
    return {"rows": rows, "half_lives": [float(t) for t in half_lives],
            "worst_abs_dev": worst, "tol": float(tol),
            "statement": ("lifetime(t * n_half) = 2**-t for every curvature, to "
                          f"{worst:.3g}; omega2 is a pure time reparameterisation and "
                          "cannot move the refusal horizon relative to the composition "
                          "horizon")}


def self_check(eps: float = 0.05, gamma: float = 0.05) -> dict:
    """⛔ Assert the transcription against the PAPER'S OWN PRINTED NUMBERS.

    Nothing downstream may use this module until this passes.  It checks:

    1. ``det A = 1-gamma`` and ``tr A = (2-gamma)(1-h^2/2)`` -- App. C Eq. (3).
    2. the floor ``2 ln2/(-ln(1-gamma))`` equals the paper's printed **27.03** at
       ``gamma = 0.05`` -- Sec. 4.1, Fig. 1, Fig. 4.
    3. ``h* = gamma/2`` to ``O(gamma^2)`` -- App. C Eq. (5).
    4. an eigenvalue reaches ``-1`` at ``h = 2`` exactly, for several ``gamma``.
    5. the overdamped ``log n_half`` vs ``log omega2`` slope is ``-1`` -- Sec. 4.1.
    6. the exceptional-point onset exponent ``arg(lambda) prop. sqrt(h - h*)`` is ``1/2``.
    7. the latch: ``omega2 = 0`` gives ``rho = 1``, an infinite half-life and the frozen
       displacement ``q_0 + eps p_0/(gamma M)``.
    8. ``LIFETIME_DEAD_EPS`` still equals ``gen_field.LIFETIME_EPS``.
    """
    out = {"eps": float(eps), "gamma": float(gamma)}
    g = float(gamma)

    # 1. the block's two invariants ----------------------------------------
    worst_det, worst_tr = 0.0, 0.0
    for w2 in np.logspace(-12, 1.5, 40):
        A = verlet_block(float(w2), eps, g)
        h = h_of(float(w2), eps)
        worst_det = max(worst_det, abs(float(np.linalg.det(A)) - (1.0 - g)))
        worst_tr = max(worst_tr, abs(float(A[0, 0] + A[1, 1])
                                     - (2.0 - g) * (1.0 - 0.5 * h * h)))
    out["block_det_max_abs_err"] = worst_det
    out["block_trace_max_abs_err"] = worst_tr
    if worst_det > 1e-12 or worst_tr > 1e-12:
        raise AssertionError("the Verlet block does not satisfy App. C Eq. (3)'s det/trace")

    # 2. the floor, against the paper's printed 27.03 ----------------------
    fl = floor_half_life(0.05)
    out["floor_at_gamma_0p05"] = fl
    out["paper_floor_at_gamma_0p05"] = PAPER["floor_steps_at_gamma_0p05"]
    out["floor_abs_err_vs_paper"] = abs(fl - PAPER["floor_steps_at_gamma_0p05"])
    if out["floor_abs_err_vs_paper"] > 5e-3:
        raise AssertionError(
            f"the transcribed floor {fl:.4f} does not reproduce the paper's printed "
            f"{PAPER['floor_steps_at_gamma_0p05']} at gamma=0.05")
    # and it really is curvature-independent: every underdamped mode sits on it
    hs = crossover_h(0.05)
    und = [n_half_closed_form((x / 0.05) ** 2, 0.05, 0.05)
           for x in np.linspace(1.5 * hs, 1.9, 25)]
    out["floor_max_dev_over_underdamped_band"] = float(max(abs(u - fl) for u in und))
    if out["floor_max_dev_over_underdamped_band"] > 1e-9:
        raise AssertionError("the underdamped half-life is not curvature-independent")

    # 3. the crossover -----------------------------------------------------
    out["crossover_h"] = {}
    for gg in (0.5, 0.2, 0.05, 0.01, 1e-3):
        hs = crossover_h(gg)
        out["crossover_h"][str(gg)] = {"h_star": hs, "gamma_over_2": 0.5 * gg,
                                       "rel_err": abs(hs - 0.5 * gg) / (0.5 * gg)}
    if out["crossover_h"]["0.001"]["rel_err"] > 1e-3:
        raise AssertionError("h* does not approach gamma/2")
    # and Delta really vanishes there
    out["crossover_delta_at_h_star"] = float(max(
        abs(spectrum((crossover_h(gg) / eps) ** 2, eps, gg)[2])
        for gg in (0.5, 0.2, 0.05, 0.01)))
    if out["crossover_delta_at_h_star"] > 1e-12:
        raise AssertionError("Delta does not vanish at h*")

    # 4. stability at h = 2 exactly, for every gamma ------------------------
    st = {}
    for gg in (0.5, 0.2, 0.05, 0.01):
        A = verlet_block((2.0 / eps) ** 2, eps, gg)
        st[str(gg)] = float(1.0 + (A[0, 0] + A[1, 1]) + np.linalg.det(A))
    out["one_plus_tr_plus_det_at_h_eq_2"] = st
    if max(abs(v) for v in st.values()) > 1e-11:
        raise AssertionError("an eigenvalue does not reach -1 at h = 2")

    # 5. the overdamped slope ----------------------------------------------
    hs = crossover_h(g)
    w2s = np.logspace(math.log10(((1e-3 * hs) / eps) ** 2),
                      math.log10(((0.30 * hs) / eps) ** 2), 40)
    nh = np.array([n_half_closed_form(float(w), eps, g) for w in w2s])
    sl = float(np.polyfit(np.log(w2s), np.log(nh), 1)[0])
    out["overdamped_slope_log_nhalf_vs_log_omega2"] = sl
    out["paper_overdamped_slope_fitted"] = PAPER["overdamped_slope_fitted"]
    if abs(sl + 1.0) > 2e-3:
        raise AssertionError(f"the overdamped slope is {sl:.5f}, not -1")
    # the paper's asymptotic agrees with the exact root deep in the band
    rel = np.abs(np.array([n_half_paper_asymptotic(float(w), eps, g) for w in w2s]) / nh - 1.0)
    out["paper_asymptotic_max_rel_dev_deep_overdamped"] = float(rel.min())
    out["paper_asymptotic_rel_dev_at_0p30_h_star"] = float(rel[-1])

    # 6. the exceptional-point onset exponent ------------------------------
    d = np.logspace(-8, -4, 30)
    phi = []
    for dd in d:
        lp, _, _ = spectrum(((hs + dd) / eps) ** 2, eps, g)
        phi.append(abs(math.atan2(lp.imag, lp.real)) if isinstance(lp, complex) else 0.0)
    phi = np.array(phi)
    m = phi > 0
    out["onset_exponent"] = float(np.polyfit(np.log(d[m]), np.log(phi[m]), 1)[0]) if m.sum() > 4 else math.nan
    out["paper_onset_exponent_fitted"] = PAPER["onset_exponent_fitted"]
    if not (abs(out["onset_exponent"] - 0.5) < 5e-3):
        raise AssertionError(f"the onset exponent is {out['onset_exponent']}, not 1/2")
    # and below h* the phase is exactly zero
    lp, _, _ = spectrum(((0.5 * hs) / eps) ** 2, eps, g)
    out["phase_below_h_star_is_exactly_zero"] = bool(not isinstance(lp, complex))

    # 7. the latch ---------------------------------------------------------
    out["latch_rho"] = retention_factor(0.0, eps, g)
    out["latch_n_half"] = n_half_closed_form(0.0, eps, g)
    tr = simulate(0.0, eps, g, 20_000, q0=0.3, p0=0.7)
    qinf = latch_displacement(0.3, 0.7, eps, g)
    out["latch_frozen_displacement_abs_err"] = abs(float(tr["q"][-1]) - qinf)
    if out["latch_rho"] != 1.0 or not math.isinf(out["latch_n_half"]):
        raise AssertionError("the latch does not have rho = 1 / infinite half-life")
    if out["latch_frozen_displacement_abs_err"] > 1e-12:
        raise AssertionError("the latch does not freeze at q_0 + eps p_0/(gamma M)")

    # 8. the store's own death threshold -----------------------------------
    from .gen_field import LIFETIME_EPS as _FLE
    out["gen_field_LIFETIME_EPS"] = float(_FLE)
    if float(_FLE) != float(LIFETIME_DEAD_EPS):
        raise AssertionError(
            f"gen_field.LIFETIME_EPS is {_FLE}, not {LIFETIME_DEAD_EPS}; the derived refusal "
            "horizon would be wrong")
    out["refusal_horizon_half_lives"] = refusal_horizon_half_lives()

    out["invariance"] = assert_reparameterisation_invariance(eps, g)
    out["PASS"] = True
    return out


def verification_table(eps: float = 0.05, gamma: float = 0.05,
                       n_points: int = 25, n_steps: int = 2_000_000) -> list:
    """⭐ The law against the SIMULATED map, across the bands.  Not a fit.

    One row per curvature: the band, ``h/h*``, the closed form, the paper's asymptotic, the
    simulated first-crossing and the simulated envelope half-life.  ⚠ The envelope and the
    first crossing agree in the overdamped band and DIVERGE above the exceptional point --
    the paper's own second-order signature (App. G), reproduced here rather than asserted.
    """
    hs = crossover_h(gamma)
    hs_grid = np.concatenate([
        np.logspace(math.log10(1e-3 * hs), math.log10(0.9 * hs), n_points // 2),
        np.logspace(math.log10(1.2 * hs), math.log10(1.6), n_points - n_points // 2)])
    rows = []
    for h in hs_grid:
        w2 = (float(h) / eps) ** 2
        nh = n_half_closed_form(w2, eps, gamma)
        steps = int(min(n_steps, max(64, 40 * nh)))
        sim = n_half_simulated(w2, eps, gamma, n_steps=steps)
        rows.append({
            "omega2": w2, "h": float(h), "h_over_h_star": float(h) / hs,
            "band": band(w2, eps, gamma),
            "n_half_closed_form": nh,
            "n_half_paper_asymptotic": n_half_paper_asymptotic(w2, eps, gamma),
            "n_half_sim_first_crossing": sim["first_crossing"],
            "n_half_sim_envelope": sim["envelope"],
            "sim_envelope_rel_err": (abs(sim["envelope"] / nh - 1.0)
                                     if math.isfinite(sim["envelope"]) and nh > 0 else math.nan),
            "crossing_over_envelope": (sim["first_crossing"] / sim["envelope"]
                                       if math.isfinite(sim["envelope"])
                                       and math.isfinite(sim["first_crossing"])
                                       and sim["envelope"] > 0 else math.nan),
        })
    return rows


