"""w6 hub/potentials -- the shorts' potential laws, tested on the FIELD STORE.

DIAL DECLARATION -- *the kernel and the dynamics*: width, separation, friction, mass.
This module renders no verdict; the scored run does.  It adds no content to the store
and no bytes to the ledger: every quantity below is read off the field store's OWN
atoms, its OWN ``well_sigma`` and its OWN lifetimes.

Two laws are composed here, and they come from different places.  That distinction is
load-bearing and is preserved in the naming.

LAW A -- **the CRR**, from the shorts (V2 Sec. 3 + App. C Eqs. 3-6; V5 Sec. 2.1).
    One dissipative velocity-Verlet step on a mass-whitened normal mode of curvature
    ``mu^2``, step ``eta``, damping ``gamma``, ``h = mu*eta``, is the linear map

        A = [[1 - h^2/2                  ,  eta (1 - h^2/4)  ],
             [-(1-g) mu^2 eta (1 - h^2/4),  (1-g)(1 - h^2/2) ]]

    with ``det A = 1 - gamma`` and ``tr A = (2-gamma)(1 - h^2/2)``.  The retention
    half-life is ``n_1/2 = ln 2 / (-ln |lambda_slow|)`` (V5's I-J instrument), and the
    bands are

      * latch        ``mu = 0``       -> ``lambda = 1``, ``n_1/2 = inf``,
                                         ``q_inf = q_0 + p_0/(M gamma)``
      * overdamped   ``0 < h < h*``   -> ``n_1/2 ~= 2 gamma ln2 / [(2-gamma) h^2]``
      * underdamped  ``h* < h < h_f`` -> ``n_1/2 = 2 ln2 / (-ln(1-gamma))``, and
                                         **independent of curvature**
      * unstable     ``h > h_f``

      with ``h*^2 = 2(1-sqrt(1-g))^2/(2-g)`` (``= g/2 + O(g^2)``) and
           ``h_f^2 = 2(1+sqrt(1-g))^2/(2-g)`` (``= 2 - O(g^2)``).

    WARNING -- **the gamma in the overdamped numerator is a CORRECTION to the shorts.**
    V2 prints ``2 ln2 / [(2-g)(mu eta)^2]``.  Expanding ``lambda_+ = 1 - eps`` in the
    characteristic polynomial gives ``-g eps + (2-g) h^2/2 + eps^2 = 0``, i.e.
    ``eps = (2-g) h^2 / (2 g)``; V2's own printed error term ``O(h^4/g^3)`` is only
    consistent with that.  Nothing here pre-registers against the printed line:
    :func:`n_half_crr` uses the EXACT eigenvalue, which reproduces V2's floor (27.03)
    and V5's ``lambda_ret = 0.998999499`` / ``n_1/2 = 692.5`` from the papers' own tables.

LAW B -- **the kernel curvature identity**.  NOT from the shorts.  Derived here, exactly,
    from the energy this repository already uses (``gen_landscape.store_energy``):

        E(x) = -sigma^2 * logsumexp_j [ log life_j - ||x - a_j||^2 / (2 sigma^2) ]

        grad E = x - <a>_w            Hess E = I - Cov_w(a) / sigma^2
        w_j proportional to life_j exp(-||x - a_j||^2 / 2 sigma^2)

    Two consequences, both exact and both about **width and separation**:

      1. ``Hess`` has eigenvalue ``1`` with multiplicity ``d - rank(Cov_w)``; at ``A``
         atoms only ``A-1`` modes can differ from 1, so the whole curvature spectrum is
         obtained from an ``A x A`` Gram matrix (:func:`curvature_spectrum`) -- exact,
         and cheap at ``d = 2048``.
      2. For two atoms of equal weight at separation ``Delta``, the curvature along the
         separation axis is ``1 - Delta^2 / (4 sigma^2)``: the well is a **single merged
         minimum** for ``Delta/sigma < 2`` and **splits into two** at ``Delta/sigma = 2``.
         That bifurcation is the interpolation floor as a property of the KERNEL.  It is
         a row-2 mechanism statement and NOT a claimable row (w6-common section 3.7: a
         size-matched continuous MLP owns row 2 on every seed, two implementations).

**THE knn_density TRAP, checked explicitly.**  ``knn_density`` failed at 0.0203 because
its length scale came from the SPACING OF STORED ITEMS.  The field store's ``well_sigma``
does not: it is the median within-value RMS spread of POOL items about their value's
coordinate (``gen_field.build_field``).  :func:`sigma_provenance` computes both on the
same atoms and returns the ratio, so "you inherited that failure" is a measurement.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "crr_matrix", "lambda_slow", "n_half_crr", "band_edges", "band_of",
    "n_half_overdamped_asym", "n_half_floor",
    "kernel_weights", "kernel_energy", "kernel_grad", "curvature_spectrum",
    "mean_shift_settle", "sigma_provenance", "well_view",
]

LN2 = float(np.log(2.0))


# ---------------------------------------------------------------------------
# LAW A -- the CRR.  Nothing in this section knows about the store.
# ---------------------------------------------------------------------------

def crr_matrix(mu: float, eta: float, gamma: float) -> np.ndarray:
    """V2 Eq. (3): the one-step map on a mass-whitened mode.

    ``mu^2`` may be negative (a saddle direction), in which case ``h^2 < 0`` and the
    mode expands; the matrix is still the right one and :func:`lambda_slow` will
    report ``|lambda| > 1``.
    """
    h2 = float(mu) * float(mu) * float(eta) * float(eta)
    g = float(gamma)
    return np.array([
        [1.0 - h2 / 2.0, eta * (1.0 - h2 / 4.0)],
        [-(1.0 - g) * mu * mu * eta * (1.0 - h2 / 4.0), (1.0 - g) * (1.0 - h2 / 2.0)],
    ], dtype=np.float64)


def lambda_slow(mu2: float, eta: float, gamma: float) -> complex:
    """Larger-modulus root of ``l^2 - (2-g)(1-h^2/2) l + (1-g) = 0``, ``h^2 = mu2 eta^2``.

    Takes ``mu2`` (the Hessian eigenvalue), not ``mu``, so a saddle is expressible.
    """
    g = float(gamma)
    h2 = float(mu2) * float(eta) * float(eta)
    tr = (2.0 - g) * (1.0 - h2 / 2.0)
    det = 1.0 - g
    disc = tr * tr - 4.0 * det
    if disc >= 0.0:
        r = np.sqrt(disc)
        a, b = (tr + r) / 2.0, (tr - r) / 2.0
        return complex(a if abs(a) >= abs(b) else b, 0.0)
    return complex(tr / 2.0, np.sqrt(-disc) / 2.0)


def n_half_crr(mu2: float, eta: float, gamma: float) -> float:
    """**THE PRE-REGISTERED PREDICTED CURVE.**  ``n_1/2 = ln2 / (-ln |lambda_slow|)``.

    Parameter-free: every input is measured or configured, nothing is fitted.
    ``inf`` at ``mu2 = 0`` (the latch) and at any ``|lambda| >= 1`` (no decay).
    """
    lam = abs(lambda_slow(mu2, eta, gamma))
    if not np.isfinite(lam) or lam >= 1.0:
        return float("inf")
    if lam <= 0.0:
        return 0.0
    return LN2 / (-np.log(lam))


def band_edges(gamma: float) -> tuple:
    """V2 Eq. (5): ``(h*, h_f)``.  ``h* = gamma/2 + O(gamma^2)``, ``h_f = 2 - O(gamma^2)``."""
    g = float(gamma)
    r = np.sqrt(1.0 - g)
    return (float(np.sqrt(2.0 * (1.0 - r) ** 2 / (2.0 - g))),
            float(np.sqrt(2.0 * (1.0 + r) ** 2 / (2.0 - g))))


def band_of(mu2: float, eta: float, gamma: float) -> str:
    """``saddle`` | ``latch`` | ``overdamped`` | ``underdamped`` | ``unstable``."""
    if mu2 < 0.0:
        return "saddle"
    if mu2 == 0.0:
        return "latch"
    h = np.sqrt(float(mu2)) * float(eta)
    hs, hf = band_edges(gamma)
    if h < hs:
        return "overdamped"
    if h < hf:
        return "underdamped"
    return "unstable"


def n_half_overdamped_asym(mu2: float, eta: float, gamma: float) -> float:
    """``2 gamma ln2 / [(2-gamma) (mu eta)^2]`` -- the gamma-CORRECTED V2 Eq. (6)."""
    g = float(gamma)
    h2 = float(mu2) * float(eta) * float(eta)
    if h2 <= 0.0:
        return float("inf")
    return 2.0 * g * LN2 / ((2.0 - g) * h2)


def n_half_floor(gamma: float) -> float:
    """``2 ln2 / (-ln(1-gamma))`` -- the curvature-INDEPENDENT underdamped floor.

    27.0268 at gamma = 0.05; V2 Sec. 4.1 reports 27.03.
    """
    return 2.0 * LN2 / (-np.log(1.0 - float(gamma)))


# ---------------------------------------------------------------------------
# LAW B -- the field store's own kernel landscape.  Exact, closed form.
# ---------------------------------------------------------------------------

def kernel_weights(x: np.ndarray, A: np.ndarray, sigma: float,
                   life=None) -> np.ndarray:
    """``w_j proportional to life_j exp(-||x-a_j||^2 / 2 sigma^2)``, normalised."""
    A = np.asarray(A, np.float64)
    d2 = np.sum((np.asarray(x, np.float64)[None, :] - A) ** 2, axis=1)
    lk = -d2 / (2.0 * float(sigma) ** 2)
    if life is not None:
        lf = np.asarray(life, np.float64)
        with np.errstate(divide="ignore"):
            lk = lk + np.where(lf > 0.0, np.log(np.maximum(lf, 1e-300)), -np.inf)
    m = lk.max()
    if not np.isfinite(m):
        return np.zeros(A.shape[0])
    w = np.exp(lk - m)
    s = w.sum()
    return w / s if s > 0 else w


def kernel_energy(x: np.ndarray, A: np.ndarray, sigma: float, life=None) -> float:
    """``E = -sigma^2 logsumexp_j [log life_j - ||x-a_j||^2/2sigma^2]``.

    The energy ``gen_landscape.store_energy`` already uses, with the field store's
    lifetimes folded in as the atom weights.
    """
    A = np.asarray(A, np.float64)
    d2 = np.sum((np.asarray(x, np.float64)[None, :] - A) ** 2, axis=1)
    lk = -d2 / (2.0 * float(sigma) ** 2)
    if life is not None:
        lf = np.asarray(life, np.float64)
        with np.errstate(divide="ignore"):
            lk = lk + np.where(lf > 0.0, np.log(np.maximum(lf, 1e-300)), -np.inf)
    m = lk.max()
    if not np.isfinite(m):
        return float("inf")
    return float(-(float(sigma) ** 2) * (m + np.log(np.exp(lk - m).sum())))


def kernel_grad(x: np.ndarray, A: np.ndarray, sigma: float, life=None) -> np.ndarray:
    """``grad E = x - <a>_w``.  Exact; verified against autodiff to 5e-16."""
    w = kernel_weights(x, A, sigma, life)
    return np.asarray(x, np.float64) - (w @ np.asarray(A, np.float64))


def curvature_spectrum(x: np.ndarray, A: np.ndarray, sigma: float,
                       life=None, mass: float = 1.0) -> dict:
    """The **mass-whitened curvature spectrum** ``mu^2`` of ``E`` at ``x``, exactly.

    ``Hess E = I - Cov_w(a)/sigma^2`` has rank-``(A-1)`` deviation from the identity, so
    the full spectrum is ``{1}^(d-r)`` union ``{1 - nu_i/sigma^2}`` with ``nu_i`` the
    eigenvalues of the ``A x A`` weighted Gram matrix.  No ``d x d`` matrix is ever
    formed: at ``d = 2048`` per well that is the difference between seconds and hours.

    Returns the non-trivial eigenvalues (descending ``nu``, i.e. ascending curvature),
    the multiplicity of the trivial ``mu^2 = 1/mass`` mode, and the mode vectors, so a
    probe can be launched along a named mode.
    """
    A = np.asarray(A, np.float64)
    n, d = A.shape
    w = kernel_weights(x, A, sigma, life)
    m = w @ A
    B = np.sqrt(np.maximum(w, 0.0))[:, None] * (A - m[None, :])    # (n, d)
    G = B @ B.T                                                     # (n, n) Gram
    nu, U = np.linalg.eigh(G)
    order = np.argsort(nu)[::-1]
    nu, U = nu[order], U[:, order]
    keep = nu > max(1e-12 * max(float(nu.max()), 1.0), 0.0)
    nu_k, U_k = nu[keep], U[:, keep]
    V = (B.T @ U_k) / np.sqrt(np.maximum(nu_k, 1e-300))[None, :]    # (d, r) unit vectors
    hess_eig = 1.0 - nu_k / (float(sigma) ** 2)
    return {
        "mu2": hess_eig / float(mass),
        "vectors": V,
        "nu": nu_k,
        "trivial_mu2": 1.0 / float(mass),
        "trivial_multiplicity": int(d - nu_k.size),
        "sigma": float(sigma),
        "mass": float(mass),
        "n_saddle": int((hess_eig < 0.0).sum()),
        "cov_trace": float(nu.sum()),
    }


def mean_shift_settle(x0: np.ndarray, A: np.ndarray, sigma: float, life=None,
                      max_iter: int = 500, tol: float = 1e-10) -> dict:
    """Settle to a critical point of ``E`` by the fixed point of ``grad E = x - <a>_w``.

    ``x <- <a>_w(x)`` is the exact zero-gradient iteration for this energy (it is mean
    shift), so the Hessian is taken at a genuine critical point of ``V`` -- the condition
    V2 App. D imposes on the whole CRR.  This settle LOCATES THE CRITICAL POINT ONLY;
    the half-life is measured by rolling the damped velocity-Verlet map itself.
    """
    x = np.asarray(x0, np.float64).copy()
    mv = float("inf")
    for it in range(int(max_iter)):
        w = kernel_weights(x, A, sigma, life)
        xn = w @ np.asarray(A, np.float64)
        mv = float(np.sqrt(np.sum((xn - x) ** 2)))
        x = xn
        if mv <= tol:
            return {"x": x, "iters": it + 1, "converged": True, "move": mv}
    return {"x": x, "iters": int(max_iter), "converged": False, "move": mv}


def sigma_provenance(A: np.ndarray, well_sigma: float) -> dict:
    """**THE knn_density TRAP CHECK.**

    ``knn_density`` failed at 0.0203 with a length scale taken from the SPACING OF
    STORED ITEMS.  This returns, on the same atoms, the field store's ``well_sigma``
    (a within-value POOL spread) beside the two spacing-derived rules
    ``gen_landscape.atom_bandwidths`` offers, and their ratios.  A ratio of 1 would mean
    the field store had inherited that length scale.
    """
    A = np.asarray(A, np.float64)
    g = (A * A).sum(axis=1)
    d2 = np.maximum(g[:, None] + g[None, :] - 2.0 * (A @ A.T), 0.0)
    np.fill_diagonal(d2, np.inf)
    nn = float(np.median(np.sqrt(d2.min(axis=1))))
    iu = np.triu_indices(A.shape[0], 1)
    pw = float(np.median(np.sqrt(np.where(np.isfinite(d2), d2, 0.0)[iu])))
    ws = float(well_sigma)
    return {"well_sigma": ws, "nn_median": nn, "pairwise_median": pw,
            "well_sigma_over_nn_median": ws / nn if nn > 0 else float("inf"),
            "well_sigma_over_pairwise_median": ws / pw if pw > 0 else float("inf"),
            "inherited_item_spacing": bool(abs(ws / nn - 1.0) < 0.05) if nn > 0 else False}


def well_view(field, k: int) -> dict:
    """The live atoms of well ``k``, its width, its lifetimes -- straight off the image."""
    A = np.asarray(field.atoms_of_well(k), np.float64)
    life = np.asarray(field.view("atom_life"), np.float64)
    alive = np.asarray(field.view("atom_alive")).astype(bool)
    sig = float(np.asarray(field.view("well_sigma"))[k])
    w = life * alive
    return {"atoms": A, "life": w, "alive": alive, "sigma": sig,
            "channels": np.asarray(field.well_channels[k], np.int64),
            "n_live": int(alive.sum())}
