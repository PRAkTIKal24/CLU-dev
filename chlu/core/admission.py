"""MVC-0 admission control: the write-time certificates a CLU controller checks.

This module implements the two load-bearing gates of the minimum viable
controller (`.claude/outputs/clu-controller-spec.md` §C3/§C5/§4) as plain,
hand-coded, **non-learned** functions over quantities the physics already
exposes. Nothing here is trained; nothing here is differentiable-by-design (the
non-differentiability debt is itemized in the controller spec, and items 2/3 of
that table never sit on an inference gradient path).

Two gates, and they are **not** interchangeable:

``spacing_gate`` (C5-A1/A2 — *placement*)
    Admit a new site iff ``min_j d(q_new, q_j) >= d_safe``. If it fails,
    **refuse-and-relocate**: draw candidates and take the first admissible one;
    if none exists, *refuse*. Refusal is a correct controller output, not a
    failure (theorist A2: 7/20 proposals refused at zero accuracy cost).

``c3_drift`` / ``c3_admissible`` (C3 — *damage to what is already stored*)
    A write with landscape change ``dV`` is admissible w.r.t. stored item ``i``
    iff the induced fixed-point drift ``||H_i^-1 grad dV(q*_i)|| <= delta_budget``,
    where ``H_i`` is the Hessian of the *pre-write* landscape at the stored
    minimum. This is the implicit-function-theorem law ``dq* = -H^-1 grad dV``,
    verified by the theorist to 0.017 % relative at perturbation 1e-3.

⚠ **The distinction that the whole w21 measurement turns on.** The spacing gate
constrains *where* you write; it says nothing about the *support* of the write
operator. For a local (atom/dictionary) write the two are equivalent, because
``||grad dV||`` at distance ``d`` carries a factor ``exp(-d^2/2s^2)`` and
``d >= 4.4 s`` puts it at ``6.3e-5``. For a **global-support** write — any free
MLP ``V_theta`` trained by gradient descent — there is no such factor and the
spacing gate buys nothing. C3 can still *detect* the damage (it is a dot
product), but detection only licenses **refusal**, not locality.
"""

from typing import Callable, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

#: MVC-0 default: the spacing gate in units of the write's spatial width `s`.
#: At this separation a Gaussian atom write contributes exp(-4.4^2/2) = 6.3e-5
#: of its own gradient scale at a neighbouring minimum -- the "5 orders" of the
#: controller spec are this number, not a property of the gate itself.
D_SAFE_MULT = 4.4


# ---------------------------------------------------------------------------
# C5-A1/A2 -- the spacing gate on placement
# ---------------------------------------------------------------------------
def min_separation(q_new, stored) -> float:
    """Distance from ``q_new`` to the nearest stored site (``inf`` if none)."""
    s = np.atleast_2d(np.asarray(stored, dtype=float))
    if s.size == 0 or s.shape[0] == 0:
        return float("inf")
    q = np.asarray(q_new, dtype=float).reshape(-1)
    return float(np.sqrt(((s - q[None, :]) ** 2).sum(-1)).min())


def spacing_ok(q_new, stored, d_safe: float) -> bool:
    """C5-A1: is ``q_new`` far enough from every stored site?"""
    return bool(min_separation(q_new, stored) >= d_safe)


def admit_site(
    q_new,
    stored,
    d_safe: float,
    key=None,
    proposer: Optional[Callable] = None,
    n_candidates: int = 400,
):
    """Spacing gate with **refuse-and-relocate** (C5-A1 + A2).

    Args:
        q_new: proposed site (address coordinates only).
        stored: (n, d) already-admitted sites.
        d_safe: admission radius.
        key: PRNG key for relocation candidates (required if ``proposer`` given).
        proposer: ``key -> (n_candidates, d)`` candidate sampler. ``None``
            disables relocation, so the gate can only admit or refuse.
        n_candidates: candidates drawn in one relocation attempt.

    Returns:
        dict with ``decision`` in ``{"admit", "relocate", "refuse"}``, the
        ``site`` actually written (``None`` when refused), the separation before
        and after, and the number of candidates that had to be examined.

    Refusal is a *correct* output: the controller declining to damage what it
    already holds. It is reported, never silently retried.
    """
    d0 = min_separation(q_new, stored)
    if d0 >= d_safe:
        return {
            "decision": "admit",
            "site": np.asarray(q_new, dtype=float),
            "d_min_proposed": d0,
            "d_min_written": d0,
            "n_candidates_examined": 0,
        }
    if proposer is None or key is None:
        return {
            "decision": "refuse",
            "site": None,
            "d_min_proposed": d0,
            "d_min_written": float("nan"),
            "n_candidates_examined": 0,
        }
    cands = np.asarray(proposer(key, n_candidates), dtype=float)
    for i, c in enumerate(cands):
        d = min_separation(c, stored)
        if d >= d_safe:
            return {
                "decision": "relocate",
                "site": c,
                "d_min_proposed": d0,
                "d_min_written": d,
                "n_candidates_examined": i + 1,
            }
    return {
        "decision": "refuse",
        "site": None,
        "d_min_proposed": d0,
        "d_min_written": float("nan"),
        "n_candidates_examined": int(cands.shape[0]),
    }


# ---------------------------------------------------------------------------
# C3 -- the admissibility check on what is already stored
# ---------------------------------------------------------------------------
def _hess_inv_grad(H: jnp.ndarray, g: jnp.ndarray, ridge: float) -> jnp.ndarray:
    """Solve ``(H + ridge I) x = g`` -- ridge only guards a singular Hessian."""
    n = H.shape[0]
    return jnp.linalg.solve(H + ridge * jnp.eye(n), g)


def c3_drift(
    V_old: Callable,
    V_new: Callable,
    q_stars,
    ridge: float = 1e-6,
    coords: Optional[Sequence[int]] = None,
):
    """First-order fixed-point drift ``||H_i^-1 grad dV(q*_i)||`` per stored item.

    ``H_i`` is the Hessian of the **pre-write** landscape at the stored minimum
    ``q*_i`` and ``dV = V_new - V_old``; the law is the implicit function
    theorem applied to ``grad V(q*) = 0`` and is exact to first order in the
    perturbation. This is the quantity the controller compares against
    ``delta_budget``; the controller can always *compute* it, even on a learned
    landscape where it cannot *guarantee* it.

    Args:
        coords: restrict the drift to a coordinate subset (e.g. the address
            plane). ``None`` uses the full state.

    Returns:
        ``np.ndarray`` of per-item predicted drift magnitudes.
    """
    qs = jnp.asarray(q_stars, dtype=jnp.float32)
    if qs.ndim == 1:
        qs = qs[None, :]
    hess = jax.hessian(lambda q: V_old(q))
    gnew, gold = jax.grad(lambda q: V_new(q)), jax.grad(lambda q: V_old(q))

    def per_item(q):
        H = hess(q)
        g = gnew(q) - gold(q)
        dq = _hess_inv_grad(H, g, ridge)
        if coords is not None:
            dq = dq[jnp.asarray(coords)]
        return jnp.linalg.norm(dq)

    return np.asarray(jax.vmap(per_item)(qs))


def c3_admissible(V_old, V_new, q_stars, delta_budget: float, ridge: float = 1e-6):
    """C3: is a write admissible w.r.t. every stored item?

    Returns ``(ok, drifts)``. ``ok`` is ``True`` iff ``max_i drift_i <=
    delta_budget``. With no stored items the write is trivially admissible.
    """
    qs = np.atleast_2d(np.asarray(q_stars))
    if qs.size == 0 or qs.shape[0] == 0:
        return True, np.zeros((0,))
    d = c3_drift(V_old, V_new, qs, ridge=ridge)
    return bool(np.max(d) <= delta_budget), d


def measured_drift(relax_fn: Callable, q_stars) -> np.ndarray:
    """Actual drift: ``||R_gamma_new(q*_i) - q*_i||`` per stored item.

    ``relax_fn`` maps a state to its relaxation endpoint under the **post-write**
    landscape. This is the ground truth that :func:`c3_drift` predicts, and the
    pair is what turns "the bound holds" into a measured ratio rather than an
    assertion.
    """
    qs = np.atleast_2d(np.asarray(q_stars, dtype=float))
    if qs.size == 0 or qs.shape[0] == 0:
        return np.zeros((0,))
    end = np.asarray(relax_fn(jnp.asarray(qs, dtype=jnp.float32)))
    return np.sqrt(((end - qs) ** 2).sum(-1))


def disk_proposer(radius: float, dim: int, center=None):
    """Uniform-in-disk candidate sampler (address plane), for relocation."""
    c = np.zeros(dim) if center is None else np.asarray(center, dtype=float)

    def sample(key, n):
        k_r, k_t = jax.random.split(key, 2)
        r = radius * jnp.sqrt(jax.random.uniform(k_r, (n,)))
        th = jax.random.uniform(k_t, (n,), minval=0.0, maxval=2.0 * jnp.pi)
        pts = jnp.stack([r * jnp.cos(th), r * jnp.sin(th)], axis=-1)
        out = jnp.zeros((n, dim)).at[:, :2].set(pts)
        return np.asarray(out) + c[None, :]

    return sample


def ring_proposer(f: float, dim: int):
    """Uniform-on-ring candidate sampler -- the w19/w20 address manifold."""

    def sample(key, n):
        th = jax.random.uniform(key, (n,), minval=0.0, maxval=2.0 * jnp.pi)
        out = (
            jnp.zeros((n, dim))
            .at[:, 0]
            .set(f * jnp.cos(th))
            .at[:, 1]
            .set(f * jnp.sin(th))
        )
        return np.asarray(out)

    return sample
