"""Implicit (DEQ-style) gradients through the shipped dissipative settle.

**A settle is a fixed point; differentiate through the equilibrium, not the
unroll** (charter §2.4). This module implements that for the *shipped* damped
velocity-Verlet map (``chlu/core/integrators.py::velocity_verlet_step``: three
Verlet substeps, then ``p <- (1-gamma) p``) — it does **not** fork the
integrator, it re-uses ``CHLU``'s rollout for the forward pass and attaches a
custom VJP to the settled point.

**The theorem it implements** (`trainability-spike-theory`, §Q1, proven and
verified there to 1.3e-8 against re-settled finite differences):

* ``Fix(T_theta) = {(q, 0) : grad V_theta(q) = 0}`` for every ``gamma in (0,2)``,
  every ``dt > 0``, every ``M > 0``, and for **both** the Newtonian and the
  relativistic kinetic. The fixed-point *set* contains neither ``gamma``, ``dt``
  nor ``M``.
* ``det(I - dT/dz)|_{z*} = ((2-gamma) dt^2 / 2)^d det(M^-1 Hess V(q*))``, so
  ``I - dT/dz`` is invertible **iff** ``Hess V(q*)`` is nonsingular. *The
  discrete dissipative map adds no degeneracy of its own.*
* Therefore

      d q*/d theta = -(Hess V_theta(q*))^-1 d_theta grad V_theta(q*),
      d p*/d theta = 0

  **exactly**, with no ``(gamma, dt, M)`` correction. The dissipation and the
  discretisation enter the *conditioning*, never the *answer*.

Two consequences the implementation takes literally (theory §7 requests 1 and 3):

1. **Solve the ``d x d`` Hessian system, not the ``2d x 2d`` map system** — the
   identical answer, half the dimension, and ``d p*/d theta`` is skipped
   entirely because it is exactly zero.
2. ⭐ **``d q*/d q0 = 0`` almost everywhere.** The fixed-point set does not
   contain ``q0``; ``q0`` only selects *which* critical point is reached, and
   basin identity is piecewise constant in ``q0``. So the implicit VJP returns a
   **zero cotangent for the launch point**, i.e. **a settled-point read-out
   sends no gradient to ``phi``.** That is not a bug and not an approximation:
   it is the exact statement of N61 ("gradient search for an address is dead",
   measured contraction ``10^-32.9`` over 3000 damped steps). A read-out that
   wants to train its own read-in must look at the **trajectory**
   (:func:`truncated_rollout`).

**Ridge.** Permitted (standard DEQ practice) and **never silently enabled**:
``ridge`` defaults to ``0.0`` and every reported number carries it as a flag.
:func:`theory_ridge` gives the theorist's zero-free-parameter value
``lambda_ridge = 2 gamma m ln(1/tol) / (N (2-gamma) dt^2)`` (= 0.354 at the
shipped ``gamma=0.05, dt=0.05, N=400, tol=1e-3``), and
:func:`ridge_alarm` implements the theorist's guard: do not proceed silently if
``lambda_ridge > 0.1 * median(lambda)``.

**Flat directions.** On a *designed* symmetry block, pass ``null_basis`` (the
known generator(s), theory §7 request 5 — never an SVD ``rcond``); the solve is
then performed on the orthogonal complement (Prop Q2.1's ``H^+``). The orbit
coordinate itself is closed-form transport, not an implicit solve
(:func:`coset_transport`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    "SettleSpec",
    "theory_ridge",
    "ridge_alarm",
    "settle_forward",
    "implicit_settle",
    "truncated_rollout",
    "settle_telemetry",
    "coset_transport",
    "GaussianWellsPotential",
]


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class SettleSpec:
    """Everything the settle and its implicit backward pass need.

    Attributes:
        steps: number of damped Verlet steps in the forward settle.
        dt: integrator step (shipped 0.05).
        gamma: friction of this phase (shipped ``gamma_address = 0.05``).
        ridge: ``lambda_ridge`` added to the Hessian before the solve.
            **0.0 = OFF (the default).** Never enable it implicitly; pass
            :func:`theory_ridge` explicitly and report it as a flag.
        mass: scalar inertial mass used by :func:`theory_ridge` only (the
            forward pass uses the model's own ``M``).
        tol: the read-budget tolerance used by :func:`theory_ridge`.
    """

    steps: int = 400
    dt: float = 0.05
    gamma: float = 0.05
    ridge: float = 0.0
    mass: float = 1.0
    tol: float = 1e-3

    def as_flags(self) -> Dict[str, Any]:
        return {"settle_steps": int(self.steps), "dt": float(self.dt),
                "gamma": float(self.gamma), "ridge": float(self.ridge),
                "ridge_enabled": bool(self.ridge > 0.0)}


def theory_ridge(gamma: float, dt: float, n_settle: int, tol: float = 1e-3,
                 mass: float = 1.0) -> float:
    """``lambda_ridge = 2 gamma m ln(1/tol) / (N (2-gamma) dt^2)``.

    The theorist's zero-free-parameter ridge (§7 request 2): ridge at exactly the
    softness the read budget cannot resolve. At the shipped
    ``(gamma=0.05, dt=0.05, N=400, tol=1e-3, m=1)`` this is **0.354**, costing
    4.1 % bias on a healthy well mode and capping gradient amplification at
    2.82x.
    """
    g, m, N = float(gamma), float(mass), max(int(n_settle), 1)
    return float(2.0 * g * m * np.log(1.0 / float(tol)) / (N * (2.0 - g) * float(dt) ** 2))


def ridge_alarm(ridge: float, eigenvalues) -> bool:
    """Theorist's guard: ``True`` (do NOT proceed silently) if the ridge exceeds
    ``0.1 * median(lambda)`` — the read budget is too small for the landscape."""
    lam = np.asarray(eigenvalues, dtype=float)
    if lam.size == 0:
        return False
    return bool(float(ridge) > 0.1 * float(np.median(lam)))


# --------------------------------------------------------------------------
# forward settle (the shipped map, via CHLU's own rollout)
# --------------------------------------------------------------------------
def settle_forward(model, q0: jnp.ndarray, p0: jnp.ndarray, spec: SettleSpec
                   ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Run the shipped damped Verlet settle. Returns ``(q_star, p_star)``.

    ``model`` is a :class:`~chlu.core.chlu_unit.CHLU` (single-sample call
    signature ``model(q, p, steps, dt, gamma) -> (steps, 2*dim)``).
    """
    tr = model(q0, p0, int(spec.steps), float(spec.dt), float(spec.gamma))
    d = q0.shape[-1]
    return tr[-1, :d], tr[-1, d:]


# --------------------------------------------------------------------------
# the implicit / DEQ settle
# --------------------------------------------------------------------------
def implicit_settle(model, q0: jnp.ndarray, p0: jnp.ndarray, spec: SettleSpec,
                    *, null_basis: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    """``q*`` of the shipped settle, with an **implicit** VJP to the parameters.

    Forward: the ordinary damped Verlet rollout (no tape).
    Backward: given a cotangent ``g`` on ``q*``, solve the ``d x d`` system

        (Hess V(q*) + ridge I) w = g,      theta_bar = -VJP_theta[grad V(., q*)](w)

    and return **zero** cotangents for ``q0`` and ``p0`` (``d q*/d q0 = 0`` a.e.;
    see the module docstring). ``d p*/d theta = 0`` is skipped, not approximated.

    Args:
        model: a CHLU whose ``potential_net`` is the differentiated ``V_theta``.
        q0, p0: launch state, shape ``(dim,)`` (vmap for a batch).
        spec: :class:`SettleSpec`.
        null_basis: optional ``(dim, r)`` **known** symmetry generators spanning
            ``ker H`` on a designed flat block (theory §7 request 5). When given,
            the solve is restricted to the orthogonal complement — never an SVD
            ``rcond`` threshold.
    """
    params, static = eqx.partition(model, eqx.is_inexact_array)
    ridge = float(spec.ridge)
    nb = None if null_basis is None else jnp.asarray(null_basis)

    def _V(prm, q):
        m = eqx.combine(prm, static)
        return jnp.reshape(m.potential_net(q), ())

    def _grad_V(prm, q):
        return jax.grad(lambda z: _V(prm, z))(q)

    @jax.custom_vjp
    def _settle(prm, qa, pa):
        m = eqx.combine(jax.lax.stop_gradient(prm), static)
        q_star, _ = settle_forward(m, qa, pa, spec)
        return q_star

    def _fwd(prm, qa, pa):
        q_star = _settle(prm, qa, pa)
        return q_star, (prm, q_star, qa, pa)

    def _bwd(res, g):
        prm, q_star, qa, pa = res
        H = jax.hessian(lambda z: _V(prm, z))(q_star)
        H = 0.5 * (H + H.T)
        d = H.shape[0]
        if ridge != 0.0:
            H = H + ridge * jnp.eye(d, dtype=H.dtype)
        if nb is None:
            w = jnp.linalg.solve(H, g)
        else:
            # Prop Q2.1: project onto (ker H)^perp with the KNOWN generator, then
            # solve there. `P` is the orthogonal projector off the null space.
            Q, _ = jnp.linalg.qr(nb)
            P = jnp.eye(d, dtype=H.dtype) - Q @ Q.T
            Hp = P @ H @ P + (jnp.eye(d, dtype=H.dtype) - P)  # identity on ker
            w = P @ jnp.linalg.solve(Hp, P @ g)
        _, vjp = jax.vjp(lambda pr: _grad_V(pr, q_star), prm)
        (prm_bar,) = vjp(w)
        prm_bar = jax.tree_util.tree_map(lambda x: -x, prm_bar)
        # d q*/d q0 = d q*/d p0 = 0 exactly (the fixed-point set has no q0 in it)
        return prm_bar, jnp.zeros_like(qa), jnp.zeros_like(pa)

    _settle.defvjp(_fwd, _bwd)
    return _settle(params, q0, p0)


# --------------------------------------------------------------------------
# truncated backprop over the trajectory read
# --------------------------------------------------------------------------
def truncated_rollout(model, q0: jnp.ndarray, p0: jnp.ndarray, steps: int,
                      dt: float, gamma: float, *, retain: Optional[int] = None,
                      stride: int = 1) -> jnp.ndarray:
    """Strided trajectory whose gradient is truncated to the last ``retain`` steps.

    ⚠ **The trajectory read is not at a fixed point, so the implicit theorem does
    not apply to it** — the theory's answer is truncation at a *derived* depth
    (§Q4.2): ``k* = ln(1/eps)/ln(1/rho)`` with
    ``rho = max(sqrt(1-gamma), 1 - (2-gamma) dt^2 lambda_min/(2 gamma m))``.
    At the shipped address phase (``gamma=0.05, dt=0.05``) ``rho = 0.97479`` and
    ``k*(1e-3) = 269``; **unroll depth beyond ~270 steps is numerically
    worthless** and is spent here as a pure forward pass (``stop_gradient`` on
    both the state and the parameters entering the free window).

    ``retain=None`` retains everything (full backprop). ``retain=0`` retains
    nothing (a pure forward read).

    Returns a ``(n_points, 2*dim)`` buffer strided by ``stride``, matching the
    layout of :attr:`chlu.core.clu_system.ReadResult.traj` (which strides
    ``tr[:, ::stride, :]`` of the ``(steps, 2*dim)`` rollout).
    """
    steps = int(steps)
    retain = steps if retain is None else int(max(0, min(retain, steps)))
    n_free = steps - retain
    params, static = eqx.partition(model, eqx.is_inexact_array)
    frozen = eqx.combine(jax.lax.stop_gradient(params), static)
    d = q0.shape[-1]

    parts = []
    q, p = q0, p0
    if n_free > 0:
        tr = frozen(jax.lax.stop_gradient(q), jax.lax.stop_gradient(p),
                    n_free, float(dt), float(gamma))
        tr = jax.lax.stop_gradient(tr)
        parts.append(tr[::stride])
        q, p = tr[-1, :d], tr[-1, d:]
    if retain > 0:
        tr2 = model(q, p, retain, float(dt), float(gamma))
        # keep the stride phase continuous across the seam
        off = (-n_free) % stride if n_free > 0 else 0
        parts.append(tr2[off::stride])
    return jnp.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]


# --------------------------------------------------------------------------
# conditioning telemetry — the theorist's Q3.5 TRIPLE
# --------------------------------------------------------------------------
def settle_telemetry(model, q_star, *, centers=None, ridge: float = 0.0,
                     d_capture: Optional[float] = None) -> Dict[str, Any]:
    """The trainer's health check: **(residual, lambda_min, basin identity)**.

    Theory §Q3.5: "reach failure = implicit-gradient ill-conditioning" is a
    *half*-identity. The well-loss half (merger / decay / spurious-minimum
    annihilation) is the same object and shows up in ``lambda_min``; the
    **address/separatrix** half — the one that actually binds on the trained
    shipped ``V`` (31/32) — is **invisible to ``lambda_min``** (constant to six
    significant figures across the whole reach sweep) and visible only in the
    **settle residual** (14 orders of dynamic range) and in **basin identity**.
    Monitor #11 supplies one leg; this function computes all three, from the one
    ``d x d`` eigendecomposition the implicit solve needs anyway.

    Returns per-query arrays plus the ridge alarm (§7 request 2).
    """
    V = model.potential_net
    q = jnp.atleast_2d(jnp.asarray(q_star))
    g = jax.vmap(jax.grad(lambda z: jnp.reshape(V(z), ())))(q)
    Hs = jax.vmap(jax.hessian(lambda z: jnp.reshape(V(z), ())))(q)
    eig = np.linalg.eigvalsh(np.asarray(Hs))
    residual = np.asarray(jnp.linalg.norm(g, axis=-1))
    lam_min = eig[:, 0]
    out: Dict[str, Any] = {
        "residual": residual,
        "lambda_min": lam_min,
        "lambda_median": np.median(eig, axis=1),
        "lambda_max": eig[:, -1],
        "cond": eig[:, -1] / np.maximum(np.abs(lam_min), 1e-30),
        "n_negative_modes": np.sum(eig < 0.0, axis=1),
        "ridge": float(ridge),
        "ridge_alarm": ridge_alarm(ridge, np.median(eig, axis=1)),
    }
    if centers is not None:
        c = np.asarray(centers, dtype=float)
        qa = np.asarray(q)[:, : c.shape[1]]
        dist = np.linalg.norm(qa[:, None, :] - c[None, :, :], axis=-1)
        out["basin"] = np.argmin(dist, axis=1)
        out["d_nearest"] = np.min(dist, axis=1)
        if d_capture is not None:
            out["basin_ok"] = out["d_nearest"] <= float(d_capture)
    return out


# --------------------------------------------------------------------------
# project-and-transport: the coset half (theory §Q2.2 / §7 request 11)
# --------------------------------------------------------------------------
def coset_transport(L0: float, r_star: float, gamma: float, dt: float,
                    mass: float = 1.0, theta0: float = 0.0) -> float:
    """``theta_inf = theta_0 + dt L_0 / (m r*^2 gamma)`` — the exact geometric sum.

    On an exactly channel-invariant ``V`` with isotropic ``M`` the angular
    momentum obeys ``L_n = (1-gamma)^n L_0`` **exactly** (verified 6.1e-14, and
    also under the relativistic kinetic), so the orbit coordinate is a closed
    form: ``O(1)``, no unrolling and no ``1/lambda_min``. Accuracy 0.02–0.75 %
    on the theorist's sweep. ⚠ Requires isotropic mass on the symmetry channel
    (a 2.5x anisotropy breaks it by 2.5 % within 150 steps).
    """
    return float(theta0 + dt * L0 / (mass * r_star**2 * gamma))


# --------------------------------------------------------------------------
# a controlled toy with a known answer (the gradcheck fixture)
# --------------------------------------------------------------------------
class GaussianWellsPotential(eqx.Module):
    """``V(q) = alpha ||q||^2 - sum_i A_i exp(-||q - c_i||^2 / 2 s^2)``.

    The theory's own landscape family (`readout-channel-theory` / §Q1, §Q3), used
    here as the *controlled toy with a known answer* the gradcheck runs on:
    ``grad V`` and ``Hess V`` are analytic, the fixed points are isolated and
    ``lambda_min`` is known, so implicit / truncated-unroll / finite-difference
    gradients can all be computed on the same object.
    """

    amp: jnp.ndarray  # (n_wells,) the differentiated parameter theta
    centers: jnp.ndarray  # (n_wells, dim)
    log_s: jnp.ndarray  # scalar
    alpha: float = eqx.field(static=True)

    def __init__(self, centers, amp, s: float = 0.35, alpha: float = 0.05):
        self.centers = jnp.asarray(centers)
        self.amp = jnp.asarray(amp)
        self.log_s = jnp.asarray(float(np.log(s)))
        self.alpha = float(alpha)

    def __call__(self, q: jnp.ndarray) -> jnp.ndarray:
        s2 = jnp.exp(2.0 * self.log_s)
        d2 = jnp.sum((q[None, :] - self.centers) ** 2, axis=-1)
        return self.alpha * jnp.sum(q**2) - jnp.sum(self.amp * jnp.exp(-d2 / (2.0 * s2)))


def toy_model(centers, amp, s: float = 0.35, alpha: float = 0.05,
              kinetic_mode: str = "newtonian_identity"):
    """A CHLU wired to :class:`GaussianWellsPotential` (the gradcheck harness)."""
    from chlu.experiments.goldstone_harness import clu_with_potential

    V = GaussianWellsPotential(centers, amp, s=s, alpha=alpha)
    dim = int(np.asarray(centers).shape[1])
    return clu_with_potential(V, dim=dim, kinetic_mode=kinetic_mode,
                              inertia=jnp.ones(dim))


def unroll_grad(model, q0, p0, spec: SettleSpec, loss_fn: Callable,
                *, retain: Optional[int] = None):
    """``d loss(q_N)/d theta`` by **truncated unroll** — the substitute-gradient
    control the implicit path is checked against."""
    params, static = eqx.partition(model, eqx.is_inexact_array)

    def L(prm):
        m = eqx.combine(prm, static)
        tr = truncated_rollout(m, q0, p0, spec.steps, spec.dt, spec.gamma,
                               retain=retain, stride=1)
        d = q0.shape[-1]
        return loss_fn(tr[-1, :d])

    return jax.grad(L)(params)


def implicit_grad(model, q0, p0, spec: SettleSpec, loss_fn: Callable,
                  *, null_basis=None):
    """``d loss(q*)/d theta`` by the **implicit** path."""
    params, static = eqx.partition(model, eqx.is_inexact_array)

    def L(prm):
        m = eqx.combine(prm, static)
        return loss_fn(implicit_settle(m, q0, p0, spec, null_basis=null_basis))

    return jax.grad(L)(params)
