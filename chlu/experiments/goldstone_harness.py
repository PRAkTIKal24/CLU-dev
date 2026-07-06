"""Goldstone-memory measurement harness (V2 / Experiment D).

Reusable instruments that measure, on any CHLU model, the F5 §3–§4 mode-budget
quantities: the spectrum probe (spectral masses mu^2 of the canonical Hessian
W = M_eff^{-1/2} · Hess V · M_eff^{-1/2}), perturb-and-track retention along
eigendirections, half-lives, the Noether charge Q = q0*p1 - q1*p0 and its
exact (1-gamma)^n decay, the coset angle theta(t), the Goldstone latch, and
one-step Jacobians.

Nomenclature (F5 Def-2, binding): **inertial mass M** is the kinetic-term
diagonal (larger M = slower, lower speed cap); **spectral mass mu** is
sqrt(eigenvalue of M_eff^{-1} Hess V) (larger mu = faster oscillation, shorter
memory). Never "mass" unqualified.

Also provides hand-built calibration potentials (quadratic, Mexican hat) and a
builder that wires them into a CHLU, so the harness can be validated against
the exact F5 §3.3–§3.4 predictions before touching any learned potential
(tests/test_goldstone.py does exactly that).

All measurements are deterministic given their inputs; precision follows the
ambient JAX dtype — enable x64 for F5-Appendix-N-grade comparisons.
"""

import math
from typing import NamedTuple, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from chlu.core.chlu_unit import CHLU


# ---------------------------------------------------------------------------
# Hand-built calibration potentials (exact-law smoke tests, F5 Appendix N)
# ---------------------------------------------------------------------------


class QuadraticPotential(eqx.Module):
    """V(q) = 1/2 q^T K q — the exact-law calibration potential (F5 §3.3)."""

    K: jnp.ndarray  # (dim, dim) symmetric stiffness matrix

    def __call__(self, q: jnp.ndarray) -> float:
        return 0.5 * q @ (self.K @ q)


class MexicanHatPotential(eqx.Module):
    """
    Central potential V = lam * (r^2 - f^2)^2 on the channel (q0, q1), plus an
    optional quadratic well 1/2 * k_spec q_spec^2 on spectator coordinates.

    Exactly SO(2)-invariant on the channel (grad V is parallel to q there, so
    every Verlet kick is torque-free); vacuum circle of radius f; radial
    spectral mass at the vacuum mu_rad^2 = 8*lam*f^2 / M_eff and an exactly
    flat angular direction (the F5 §4.1 check-(g) geometry).
    """

    lam: float = eqx.field(static=True)
    f: float = eqx.field(static=True)
    k_spec: Optional[jnp.ndarray]  # (dim-2,) or None

    def __call__(self, q: jnp.ndarray) -> float:
        r2 = q[0] ** 2 + q[1] ** 2
        v = self.lam * (r2 - self.f**2) ** 2
        if self.k_spec is not None:
            v = v + 0.5 * jnp.sum(self.k_spec * q[2:] ** 2)
        return v


def log_mass_for_inertia(inertia) -> jnp.ndarray:
    """
    log_mass such that the unit's dynamics use exactly the requested rest
    inertia: softplus(log_mass) + 1e-6 == inertia (see CHLU.effective_inertia;
    the 1e-6 is the numerical-stability epsilon inside CHLU.H).

    Only meaningful for kinetic modes that read log_mass (newtonian_learned,
    relativistic — for the latter the rest inertia is rest_mass * inertia).
    """
    target = jnp.asarray(inertia) - 1e-6  # softplus target
    return jnp.log(jnp.expm1(target))  # inverse softplus


def clu_with_potential(
    potential: eqx.Module,
    dim: int,
    kinetic_mode: str = "newtonian_identity",
    inertia=None,
    rest_mass: float = 1.0,
    c: float = 1.0,
    tie_channel_mass: bool = False,
    key: jax.random.PRNGKey = None,
) -> CHLU:
    """
    Build a CHLU wired to a hand-built potential (bypassing the learned nets).

    Args:
        potential: any callable eqx.Module q -> scalar (e.g. QuadraticPotential).
        dim: latent dimension.
        kinetic_mode: as in CHLU.
        inertia: optional per-coordinate target effective rest inertia,
            converted through log_mass so model.effective_inertia() == inertia
            up to fp rounding (ignored by newtonian_identity).
        tie_channel_mass: kinetic isotropy switch (F5 §4.1).
        key: PRNG key (only used for the throwaway learned nets).

    Returns:
        CHLU whose potential_net is ``potential``.
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    model = CHLU(
        dim=dim,
        hidden=4,
        rest_mass=rest_mass,
        c=c,
        kinetic_mode=kinetic_mode,
        potential_type="mlp",
        tie_channel_mass=tie_channel_mass,
        key=key,
    )
    model = eqx.tree_at(lambda m: m.potential_net, model, replace=potential)
    if inertia is not None:
        model = eqx.tree_at(
            lambda m: m.log_mass,
            model,
            replace=log_mass_for_inertia(jnp.asarray(inertia)),
        )
    return model


# ---------------------------------------------------------------------------
# Spectrum probe (F5 §3.1)
# ---------------------------------------------------------------------------


class SpectrumProbe(NamedTuple):
    """Local mode decomposition at a probe point q* (F5 §3.1)."""

    q_star: jnp.ndarray  # (dim,) probe point
    K: jnp.ndarray  # (dim, dim) stiffness matrix Hess V(q*)
    M_eff: jnp.ndarray  # (dim,) rest inertial-mass diagonal (F5 §2.1 table)
    W: jnp.ndarray  # (dim, dim) canonical M_eff^{-1/2} K M_eff^{-1/2}
    mu_sq: jnp.ndarray  # (dim,) spectral masses^2, ascending (negatives = saddle)
    eigvecs: jnp.ndarray  # (dim, dim) eigenvector columns, canonical coords
    grad_norm: jnp.ndarray  # |grad V(q*)| — how settled the probe point is


def spectrum_probe(model: CHLU, q_star: jnp.ndarray) -> SpectrumProbe:
    """
    Exact local spectrum at q*: W = M_eff^{-1/2} · Hess V(q*) · M_eff^{-1/2},
    eigenvalues mu_k^2 (spectral masses squared) and eigenvectors in
    canonically normalized coordinates (q_tilde = M_eff^{1/2} q).

    Exact only near p ≈ 0 (F5 Prop-2: the relativistic kinetic term couples
    modes at finite momentum). Uses jax.hessian — fine for small dims.
    """

    def V(q):
        return model.potential_net(q)

    K = jax.hessian(V)(q_star)
    grad_norm = jnp.linalg.norm(jax.grad(V)(q_star))
    m_eff = model.effective_inertia()
    s = 1.0 / jnp.sqrt(m_eff)
    W = (K * s[:, None]) * s[None, :]
    W = 0.5 * (W + W.T)  # symmetrize fp noise
    mu_sq, eigvecs = jnp.linalg.eigh(W)
    return SpectrumProbe(q_star, K, m_eff, W, mu_sq, eigvecs, grad_norm)


def settle(
    model: CHLU,
    q0: jnp.ndarray,
    p0: Optional[jnp.ndarray] = None,
    dt: float = 0.05,
    gamma: float = 0.1,
    steps: int = 2000,
) -> tuple:
    """Damped rollout to the nearest attractor. Returns (q*, p*≈0)."""
    if p0 is None:
        p0 = jnp.zeros_like(q0)
    traj = model(q0, p0, steps=steps, dt=dt, gamma=gamma)
    d = q0.shape[0]
    return traj[-1, :d], traj[-1, d:]


# ---------------------------------------------------------------------------
# Perturb-and-track (F5 §3.3 retention measurements)
# ---------------------------------------------------------------------------


def rollout_from(
    model: CHLU,
    q0: jnp.ndarray,
    p0: jnp.ndarray,
    steps: int,
    dt: float,
    gamma: float = 0.0,
) -> jnp.ndarray:
    """
    (steps+1, 2*dim) trajectory INCLUDING the initial state as row 0, so a
    series index equals the number of map applications (F5's n counting) —
    CHLU.__call__ alone returns post-step states only.
    """
    traj = model(q0, p0, steps=steps, dt=dt, gamma=gamma)
    z0 = jnp.concatenate([q0, p0])[None, :]
    return jnp.concatenate([z0, traj], axis=0)


def mode_coordinates(probe: SpectrumProbe, traj: jnp.ndarray) -> tuple:
    """
    Canonical per-mode coordinates of a trajectory:
        d  = V^T M_eff^{1/2} (q - q*)   (mode positions)
        pc = V^T M_eff^{-1/2} p         (mode momenta)
    Returns (d, pc), each (T, dim).
    """
    dim = probe.q_star.shape[0]
    sqm = jnp.sqrt(probe.M_eff)
    d = ((traj[:, :dim] - probe.q_star) * sqm) @ probe.eigvecs
    pc = (traj[:, dim:] / sqm) @ probe.eigvecs
    return d, pc


def mode_amplitude(
    probe: SpectrumProbe, d: jnp.ndarray, pc: jnp.ndarray, mu_floor: float = 1e-8
) -> jnp.ndarray:
    """
    Oscillation-envelope amplitude per mode: sqrt(d^2 + (pc/mu)^2) for modes
    with mu > mu_floor, |d| for (near-)flat or unstable modes.

    Feed THIS (not raw |d|) to half-life extractors for underdamped modes:
    first-crossing of raw |d| measures oscillation phase, not retention
    (F5 Appendix N, "known diagnostic artifacts").
    """
    mu = jnp.sqrt(jnp.clip(probe.mu_sq, 0.0, None))
    safe_mu = jnp.where(mu > mu_floor, mu, 1.0)
    a_osc = jnp.sqrt(d**2 + (pc / safe_mu) ** 2)
    return jnp.where(mu > mu_floor, a_osc, jnp.abs(d))


def perturb_and_track(
    model: CHLU,
    probe: SpectrumProbe,
    mode_idx: int,
    kick: float = 0.1,
    kick_type: str = "position",
    steps: int = 4000,
    dt: float = 0.05,
    gamma: float = 0.05,
) -> dict:
    """
    Kick one canonical mode at q* and roll out at fixed friction gamma.

    kick_type:
        "position": q0 = q* + kick * M_eff^{-1/2} v_k  (canonical displacement
                    of size ``kick``; p0 = 0) — retention measurement.
        "momentum": p0 = kick * M_eff^{1/2} v_k (canonical impulse of size
                    ``kick``; q0 = q*) — the Goldstone *write* operation
                    (F5 §3.3a: the charge is the write current).

    Returns dict with:
        "traj":      (steps+1, 2*dim) trajectory incl. initial state,
        "d", "pc":   canonical mode positions/momenta (steps+1, dim),
        "amplitude": per-mode envelope amplitudes (steps+1, dim),
        "retention": amplitude of the kicked mode normalized to 1 at n=0.
    """
    v = probe.eigvecs[:, mode_idx]
    sqm = jnp.sqrt(probe.M_eff)
    if kick_type == "position":
        q0 = probe.q_star + kick * v / sqm
        p0 = jnp.zeros_like(probe.q_star)
    elif kick_type == "momentum":
        q0 = probe.q_star
        p0 = kick * v * sqm
    else:
        raise ValueError(
            f"Unknown kick_type: {kick_type}. Must be 'position' or 'momentum'."
        )

    traj = rollout_from(model, q0, p0, steps=steps, dt=dt, gamma=gamma)
    d, pc = mode_coordinates(probe, traj)
    amp = mode_amplitude(probe, d, pc)
    retention = amp[:, mode_idx] / amp[0, mode_idx]
    return {"traj": traj, "d": d, "pc": pc, "amplitude": amp, "retention": retention}


# ---------------------------------------------------------------------------
# Extractors: half-life, decay rate, latch (F5 §3.3–§3.4)
# ---------------------------------------------------------------------------


def half_life_first_crossing(series) -> float:
    """
    First index n >= 1 with series[n] <= series[0]/2 (F5's first-crossing
    n_1/2). Returns math.inf if the series never crosses. Feed an envelope
    amplitude, not raw |d|, for underdamped modes.
    """
    s = np.asarray(series)
    below = np.nonzero(s <= s[0] / 2.0)[0]
    below = below[below > 0]
    return float(below[0]) if below.size else math.inf


def fit_decay_rate(series, n_fit: Optional[int] = None) -> float:
    """
    Least-squares slope of log(series[n]) vs n — the per-step log decay rate.
    For an underdamped mode this estimates ln|lambda| = 0.5*ln(1-gamma)
    (F5 §3.3b, mass-independent). Choose n_fit spanning >= 2 oscillation
    periods so the phase ripple of the amplitude reconstruction averages out.
    """
    s = np.asarray(series)
    if n_fit is not None:
        s = s[:n_fit]
    mask = s > 0
    n = np.arange(len(s))[mask]
    return float(np.polyfit(n, np.log(s[mask]), 1)[0])


def exact_mode_eigenvalues(mu_sq: float, dt: float, gamma: float) -> tuple:
    """
    Exact eigenvalues of the per-mode 2x2 damped-Verlet map (F5 §3.3):
        tr A = (2-gamma) * (1 - h^2/2),  det A = 1-gamma,  h^2 = dt^2 * mu^2.
    Returns a complex pair (lam_plus, lam_minus).
    """
    h2 = dt * dt * mu_sq
    tr = (2.0 - gamma) * (1.0 - h2 / 2.0)
    det = 1.0 - gamma
    disc = complex(tr * tr - 4.0 * det) ** 0.5
    return ((tr + disc) / 2.0, (tr - disc) / 2.0)


def h_star(gamma: float) -> float:
    """Exact over/underdamped crossover h*(gamma) = (1-sqrt(1-gamma))*sqrt(2/(2-gamma))."""
    return (1.0 - math.sqrt(1.0 - gamma)) * math.sqrt(2.0 / (2.0 - gamma))


def classify_mode(mu_sq: float, dt: float, gamma: float, flat_tol: float = 1e-8) -> str:
    """F5 §3.4 budget-table band of one mode at step dt, friction gamma."""
    if mu_sq < -flat_tol:
        return "unstable"
    if mu_sq <= flat_tol:
        return "latch"
    h = dt * math.sqrt(mu_sq)
    if h >= 2.0:
        return "forbidden"
    if gamma > 0.0 and h < h_star(gamma):
        return "register"
    return "working_memory"


def predicted_half_life(
    mu_sq: float, dt: float, gamma: float, flat_tol: float = 1e-8
) -> float:
    """
    Exact-map half-life prediction (steps) per F5 §3.4:
        latch: inf;  register/working memory: ln 2 / (-ln |lam_slow|)
    (for underdamped modes |lam| = sqrt(1-gamma), giving 2*ln2 / (-ln(1-gamma));
    unstable/forbidden bands have no half-life — returns inf).
    """
    band = classify_mode(mu_sq, dt, gamma, flat_tol)
    if band in ("latch", "unstable", "forbidden"):
        return math.inf
    lam_plus, _ = exact_mode_eigenvalues(mu_sq, dt, gamma)
    mod = abs(lam_plus)  # slow (memory) eigenvalue: the larger modulus
    if mod >= 1.0:
        return math.inf
    return math.log(2.0) / (-math.log(mod))


def latch_prediction(d0: float, pc0: float, dt: float, gamma: float) -> float:
    """
    Exact Newtonian flat-mode latch in canonical coordinates (unit inertia,
    F5 §3.3a): d_inf = d0 + dt * pc0 / gamma. In physical coordinates this is
    q_inf = q0 + dt * p0 / (M_eff * gamma).
    """
    return d0 + dt * pc0 / gamma


# ---------------------------------------------------------------------------
# Symmetry observables (F5 §4.1)
# ---------------------------------------------------------------------------


def noether_charge(traj: jnp.ndarray, dim: int, channel: tuple = (0, 1)) -> jnp.ndarray:
    """
    SO(2) Noether charge Q = q_i p_j - q_j p_i per step (F5 §4.1). Conserved
    to machine precision by the conservative map iff V is channel-invariant
    AND the channel inertial masses are equal; decays exactly as (1-gamma)^n
    under friction. Q is the *write current* — memory itself lives in the
    coset angle (see ``coset_angle``).
    """
    i, j = channel
    q = traj[:, :dim]
    p = traj[:, dim:]
    return q[:, i] * p[:, j] - q[:, j] * p[:, i]


def coset_angle(
    traj: jnp.ndarray, dim: int, channel: tuple = (0, 1), unwrap: bool = True
) -> np.ndarray:
    """
    Coset coordinate theta(t) = atan2(q_j, q_i) on the channel plane — where
    the memory actually lives (F5 §4.1 "charge vs coordinate"). Unwrapped by
    default so latch freezing is visible as a plateau.
    """
    i, j = channel
    q = np.asarray(traj[:, :dim])
    th = np.arctan2(q[:, j], q[:, i])
    return np.unwrap(th) if unwrap else th


# ---------------------------------------------------------------------------
# Jacobian probe (F5 Props 3–5 ground truth)
# ---------------------------------------------------------------------------


def step_jacobian(
    model: CHLU, q: jnp.ndarray, p: jnp.ndarray, dt: float, gamma: float = 0.0
) -> jnp.ndarray:
    """
    Jacobian of one dissipative-Verlet step at (q, p) — the object of
    F5 Props 3–5. Its exact eigenvalues are the retention ground truth
    (F5 Appendix N: measured-crossing artifacts are diagnostic, the
    eigenvalues are not).
    """
    dim = q.shape[0]

    def f(z):
        qn, pn = model.step((z[:dim], z[dim:]), dt, gamma)
        return jnp.concatenate([qn, pn])

    return jax.jacobian(f)(jnp.concatenate([q, p]))
