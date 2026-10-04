"""Experiment: the paid-access battery (w7, V1 pillar-4 gate).

The discriminating end-to-end test of intra-unit *access* mechanisms
(paid-access-theory §7.1-7.3). A relativistic CLU has a causal box C_T
(Prop-A2) whose coord-i half-width is L_i = T*eps*c/sqrt(M_i): the position
shadow of a T-step rollout cannot leave it, *no matter how much energy is
injected* (the relativistic velocity cap c/sqrt(M_i) is energy-blind). The two
failure modes split (Def-A4): ESCAPE (barrier inside the box) is cured by a
bounded-energy squeeze; REACH (target outside the box) is cured only by a
non-local jump (the wormhole, det J = 1 with an energy ledger).

§7.1 REACH task: a double-well along coord 0 with wells at 0 and d, barrier
Delta V_b; d swept below AND above L. Arms:
  - plain_relax            : governed relaxation (KE0 < Delta V_b => escape-blocked)
  - squeeze                : S^(M) mass-weighted squeeze, line-searched zeta
  - wormhole               : matched constant-translation channel (Prop-A6)
  - newtonian_squeeze      : Newtonian mode + squeeze (CONTROL: energy buys reach)
  - no_physics_router      : oracle sets q := target (CM-7 mandatory control)
  - throat_denseV          : nonlocal throat lowers the barrier (dense-V discriminator)
Sharp predictions: squeeze landing ~1 for d<L (above zeta*), drops to ~0 for
d>L; wormhole flat ~1 all d; newtonian_squeeze rises even d>L; router flat ~1
but no volume certificate; throat helps d<L escape but still fails d>L.

§7.2 latch transit: SO(2) sector, Q = p^T X q; wormhole Delta tangent vs across
the coset => Q'-Q = p^T X Delta (Prop-A7). §7.3 certificates on every arm.

Mass banding (theory §3.3 reason 2) is a PREREQUISITE and is applied to the
CLU's inertial mass so S^(M) is directional; the band is in the provenance.

Config: ``config.experiment_paid_access``. NO training — analytic potentials,
oracle channel placement (learned entrance-steering is explicitly out of scope).
"""

import json
import os

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from chlu.core.chlu_unit import CHLU
from chlu.core.potentials import (
    IntraWormholePotential,
    WormholeChannels,
    so2_generator,
)
from chlu.core.transforms import (
    effective_mass,
    mass_weighted_squeeze,
    squeeze,
    squeeze_matrix,
    symplectic_form,
)


# --------------------------------------------------------------------------
# Analytic potentials (eqx.Modules; not learned — geometry is controlled)
# --------------------------------------------------------------------------
class DoubleWellReach(eqx.Module):
    """Double well along coord 0 (wells at 0 and d, barrier Delta V_b at d/2),
    harmonic confinement on all other coords. Barrier height is exactly
    ``barrier``; minima are exactly at x=0 and x=d."""

    d: float = eqx.field(static=True)
    barrier: float = eqx.field(static=True)
    conf: float = eqx.field(static=True)

    def __call__(self, q):
        x = q[0]
        u = (x - self.d / 2.0) / (self.d / 2.0)  # -1 at 0, +1 at d
        v0 = self.barrier * (u * u - 1.0) ** 2
        v_rest = 0.5 * self.conf * jnp.sum(q[1:] ** 2)
        return v0 + v_rest


class NonCoerciveWell(eqx.Module):
    """V(q) = 0.5*k*q0^2 - eps*q0^4 + 0.5*conf*||q_{1:}||^2 — the Deep/Conv-style
    NON-coercive potential (F5 §7 issue 7 / paid-access-theory §7.4): coercive
    only inside the connected sub-level component |q0| < x_b, unbounded below
    outside it. Barrier top x_b = sqrt(k/(4 eps)) with V(x_b) = V_b = k^2/(16 eps).

    A wormhole/router exit placed beyond x_b breaks BIBO: the state accelerates
    down the quartic tail forever (the relativistic cap bounds its SPEED at
    c/sqrt(M_0), not its excursion, so ||q|| grows linearly in T). Note V is NOT
    monotone in |q0|: V(4.0) = 2.88 < V_b = 3.125, so an energy-only admissibility
    test passes an exit that in fact escapes — the receipt must test component
    membership as well."""

    k: float = eqx.field(static=True)
    eps: float = eqx.field(static=True)
    conf: float = eqx.field(static=True)

    def __call__(self, q):
        x = q[0]
        v0 = 0.5 * self.k * x**2 - self.eps * x**4
        v_rest = 0.5 * self.conf * jnp.sum(q[1:] ** 2)
        return v0 + v_rest

    def barrier_top(self):
        """x_b: edge of the coercive component containing the origin."""
        return float(np.sqrt(self.k / (4.0 * self.eps)))

    def barrier_height(self):
        """V_b = V(x_b): the escape energy out of the coercive component."""
        return float(self.k**2 / (16.0 * self.eps))


class HarmonicPotential(eqx.Module):
    """Isotropic harmonic V = 0.5 * k * ||q||^2 — a positive quadratic form,
    the setting in which the squeeze injection bound H(Sz) <= e^{2|zeta|} H(z)
    (Prop-12 C2) is exactly a theorem. (The quartic DoubleWellReach can exceed
    it, which is expected — the bound is a quadratic-energy certificate.)"""

    k: float = eqx.field(static=True)

    def __call__(self, q):
        return 0.5 * self.k * jnp.sum(q**2)


def _inv_softplus(y):
    # log(exp(y) - 1); stable for the modest masses used here
    return np.log(np.expm1(y))


def _band_log_mass(dim, band):
    """log_mass such that softplus(log_mass) == the designed band: coord 0 gets
    band[0] (the reach coord), all others band[1] (cycled if longer)."""
    m = np.empty(dim)
    m[0] = band[0]
    for i in range(1, dim):
        m[i] = band[i % len(band)] if len(band) > 1 else band[0]
    return jnp.asarray(_inv_softplus(m), dtype=jnp.float32)


def _make_clu(dim, kinetic_mode, c, rest_mass, band, potential, key):
    model = CHLU(
        dim=dim,
        hidden=8,
        rest_mass=rest_mass,
        c=c,
        kinetic_mode=kinetic_mode,
        potential_type="mlp",
        key=key,
    )
    model = eqx.tree_at(lambda m: m.log_mass, model, _band_log_mass(dim, band))
    model = eqx.tree_at(lambda m: m.potential_net, model, potential)
    return model


def _rollout(model, q0, p0, steps, dt, gamma):
    """Deterministic Verlet rollout -> trajectory (steps, 2*dim)."""
    return model(q0, p0, steps, dt, gamma)


def _landed(traj, target_x0, tol):
    """True iff coord-0 of the trajectory enters within tol of target at any
    step (Def-A1 reachability: Q_T intersect basin != empty)."""
    x0 = traj[:, 0]
    return bool(jnp.min(jnp.abs(x0 - target_x0)) < tol)


# --------------------------------------------------------------------------
# §7.1 reach battery
# --------------------------------------------------------------------------
def _reach_battery(cfg, distances, seeds, quick=False):
    dim = cfg.dim
    dt = cfg.dt
    T = cfg.reach_steps
    band = list(cfg.mass_band)

    # Causal box half-width for coord 0 (Prop-A2): L = T*dt*c/sqrt(M_eff,0)
    M0 = band[0] * (cfg.rest_mass)  # relativistic M_eff,0 = m0 * M
    L = T * dt * cfg.c / np.sqrt(M0)

    arms = [
        "plain_relax",
        "squeeze",
        "wormhole",
        "newtonian_squeeze",
        "no_physics_router",
        "throat_denseV",
    ]
    # landing[arm][d] = list over seeds of 0/1
    landing = {a: {float(d): [] for d in distances} for a in arms}
    cert = {"wormhole_detJ": [], "wormhole_ledger_err": [], "router_detJ": []}

    for d in distances:
        pot = DoubleWellReach(
            d=float(d), barrier=cfg.barrier_height, conf=cfg.basin_curvature
        )
        # oracle wormhole channel: entrance at start well (0), exit at target well (d)
        ent = jnp.zeros((1, dim)).at[0, 0].set(0.0)
        ext = jnp.zeros((1, dim)).at[0, 0].set(float(d))
        rad = jnp.array([cfg.capture_radius])
        channels = WormholeChannels(entrances=ent, exits=ext, radii=rad)
        # throat: lower the barrier at the midpoint d/2
        via = jnp.zeros((1, dim)).at[0, 0].set(float(d) / 2.0)
        throat_pot = IntraWormholePotential(
            base=pot,
            via=via,
            depth=jnp.array([cfg.throat_depth]),
            width=jnp.array([max(0.25, float(d) / 6.0)]),
        )

        for s in seeds:
            key = jax.random.PRNGKey(s)
            k_init, k_rel, k_new, k_th = jax.random.split(key, 4)
            # init at start well, momentum along coord 0 (KE0 < barrier by design)
            q0 = jnp.zeros(dim)
            p0 = jnp.zeros(dim).at[0].set(cfg.init_momentum)
            # small per-seed perturbation for seed variance
            p0 = p0 + 0.02 * jax.random.normal(k_init, (dim,))

            rel = _make_clu(dim, "relativistic", cfg.c, cfg.rest_mass, band, pot, k_rel)
            m_eff = effective_mass(rel)

            # -- plain relax (conservative reach rollout: gamma=0 => sharp box L) --
            traj = _rollout(rel, q0, p0, T, dt, 0.0)
            landing["plain_relax"][float(d)].append(
                int(_landed(traj, d, cfg.landing_tol))
            )

            # -- squeeze: line-search zeta; success if ANY zeta lands --
            sq_land = 0
            for zeta in cfg.zeta_grid:
                qs, ps = mass_weighted_squeeze(q0, p0, float(zeta), m_eff)
                traj = _rollout(rel, qs, ps, T, dt, 0.0)
                if _landed(traj, d, cfg.landing_tol):
                    sq_land = 1
                    break
            landing["squeeze"][float(d)].append(sq_land)

            # -- wormhole: matched channel jump, then relax --
            qj, pj, jumped = channels.jump(q0, p0)
            traj = _rollout(rel, qj, pj, T, dt, 0.0)
            landing["wormhole"][float(d)].append(int(_landed(traj, d, cfg.landing_tol)))

            # -- newtonian squeeze control (energy DOES buy reach) --
            newt = _make_clu(
                dim, "newtonian_learned", cfg.c, cfg.rest_mass, band, pot, k_new
            )
            m_eff_n = effective_mass(newt)
            nt_land = 0
            for zeta in cfg.zeta_grid:
                qs, ps = mass_weighted_squeeze(q0, p0, float(zeta), m_eff_n)
                traj = _rollout(newt, qs, ps, T, dt, 0.0)
                if _landed(traj, d, cfg.landing_tol):
                    nt_land = 1
                    break
            landing["newtonian_squeeze"][float(d)].append(nt_land)

            # -- no-physics router (oracle): sets q := target, no dynamics --
            q_router = q0.at[0].set(float(d))
            landing["no_physics_router"][float(d)].append(
                int(bool(jnp.abs(q_router[0] - d) < cfg.landing_tol))
            )

            # -- throat / dense-V discriminator (relativistic, plain relax) --
            th = _make_clu(
                dim, "relativistic", cfg.c, cfg.rest_mass, band, throat_pot, k_th
            )
            traj = _rollout(th, q0, p0, T, dt, 0.0)
            landing["throat_denseV"][float(d)].append(
                int(_landed(traj, d, cfg.landing_tol))
            )

        # certificates for this d (seed 0): det J of wormhole jump, ledger, router
        q0 = jnp.zeros(dim)

        def jmap(qp, ch=channels, q_capture=q0):
            q_, p_ = qp[:dim], qp[dim:]
            delta, _ = ch.selected_delta(q_capture)  # frozen gate at capture
            return jnp.concatenate([q_ + delta, p_])

        J = jax.jacfwd(jmap)(jnp.concatenate([q0, jnp.zeros(dim)]))
        cert["wormhole_detJ"].append(float(jnp.linalg.det(J)))
        led = float(channels.ledger(pot, q0))
        qj, _, _ = channels.jump(q0, jnp.zeros(dim))
        cert["wormhole_ledger_err"].append(abs(led - float(pot(qj) - pot(q0))))
        # router: q := target. This IS differentiable, and its Jacobian is
        # blockdiag(0_d, I_d) => det J = 0 EXACTLY: the map is volume-ANNIHILATING
        # and non-invertible (not merely "uncertified"). Measured, not asserted.
        target = jnp.zeros(dim).at[0].set(float(d))

        def rmap(qp, tgt=target):
            return jnp.concatenate([tgt, qp[dim:]])

        Jr = jax.jacfwd(rmap)(jnp.concatenate([q0, jnp.zeros(dim)]))
        cert["router_detJ"].append(float(jnp.linalg.det(Jr)))

    rates = {
        a: {str(d): float(np.mean(landing[a][float(d)])) for d in distances}
        for a in arms
    }
    return {
        "L": float(L),
        "M_eff_0": float(M0),
        "v_max_0": float(cfg.c / np.sqrt(M0)),
        "distances": [float(d) for d in distances],
        "landing_rates": rates,
        "raw_landing": {
            a: {str(d): landing[a][float(d)] for d in distances} for a in arms
        },
        "certificates": cert,
    }


# --------------------------------------------------------------------------
# §7.2 latch transit + §7.3 squeeze/injection certificates
# --------------------------------------------------------------------------
def _latch_and_certs(cfg):
    dim = max(cfg.dim, 2)
    band = list(cfg.mass_band)
    X = so2_generator(dim)
    f = cfg.latch_radius
    # state on the vacuum circle radius f, momentum tangent-ish
    q = jnp.zeros(dim).at[0].set(f)
    p = jnp.zeros(dim).at[1].set(cfg.latch_momentum).at[0].set(0.1 * cfg.latch_momentum)

    # Latch shift is Q'-Q = p^T X Delta (Prop-A7). It is ZERO iff X.Delta _|_ p,
    # i.e. Delta _|_ X^T p = -X p. Construct that zero-shift channel explicitly,
    # and an "across" channel (radial) that shifts by the full charge scale.
    Xp = X @ p
    # zero-shift Delta: orthogonal to X p within the channel plane (coords 0,1)
    delta_zero = jnp.zeros(dim).at[0].set(Xp[1]).at[1].set(-Xp[0])
    delta_zero = delta_zero / (jnp.linalg.norm(delta_zero) + 1e-9) * 0.5
    # across the coset: radial (moves off the vacuum circle) => nonzero shift
    delta_rad = q / (jnp.linalg.norm(q) + 1e-9) * 0.5

    def q_shift(delta):
        return float(p @ X @ (q + delta) - p @ X @ q), float(p @ X @ delta)

    tan_meas, tan_pred = q_shift(delta_zero)
    rad_meas, rad_pred = q_shift(delta_rad)

    # A channel-ISOTROPIC squeeze (equal rapidity, equal channel mass) commutes
    # with X and preserves Q exactly (proven: Q'=Q). The raw squeeze is the
    # mass-blind, uniform version — use it here. NOTE a *banded* S^(M) (unequal
    # channel mass) does NOT commute with X and legitimately changes Q; the
    # latch-preserving arm requires channel isotropy (F5 §4.1).
    qs, ps = squeeze(q, p, 0.5)
    Q0 = float(p @ X @ q)
    Qsq = float(ps @ X @ qs)

    # random-shift baseline erases Q unpredictably
    rng = np.random.default_rng(cfg.seed0)
    rand_shifts = []
    for _ in range(5):
        dr = jnp.asarray(rng.normal(size=dim) * 0.5, dtype=jnp.float32)
        rand_shifts.append(float(p @ X @ (q + dr) - p @ X @ q))

    # §7.3 squeeze injection: (i) S^(M) is exactly symplectic with det=1 for the
    # BANDED m_eff (structure certificate), and (ii) the injected energy obeys
    # H(S^(M) z) <= e^{2|zeta|} H(z) (Prop-12 C2). The bound is a theorem for the
    # mass-MATCHED quadratic H = 1/2 p^T M^-1 p + 1/2 q^T M q (in the normalized
    # coordinates the mass-weighted squeeze is the standard isotropic squeeze).
    # A mismatched curvature (e.g. isotropic k with banded M) legitimately
    # exceeds it — the certificate is for the matched quadratic energy.
    rel = _make_clu(
        dim,
        "newtonian_learned",
        cfg.c,
        cfg.rest_mass,
        band,
        HarmonicPotential(1.0),
        jax.random.PRNGKey(0),
    )
    m_eff = effective_mass(rel)
    Omega = symplectic_form(dim)

    def H_matched(qq, pp):
        return 0.5 * float(jnp.sum(pp**2 / m_eff) + jnp.sum(m_eff * qq**2))

    inj = []
    for zeta in [0.25, 0.5, 1.0, 2.0]:
        S = squeeze_matrix(zeta, dim, m_eff=m_eff)
        detS = float(jnp.linalg.det(S))
        symp_err = float(jnp.max(jnp.abs(S.T @ Omega @ S - Omega)))
        H0 = H_matched(q, p)
        qn, pn = mass_weighted_squeeze(q, p, zeta, m_eff)
        H1 = H_matched(qn, pn)
        inj.append(
            {
                "zeta": zeta,
                "detS": detS,
                "symplectic_err": symp_err,
                "H_ratio": H1 / H0,
                "e2zeta_bound": float(np.exp(2 * zeta)),
                "bound_holds": bool(H1 / H0 <= np.exp(2 * zeta) + 1e-6),
            }
        )

    return {
        "latch": {
            "tangent": {
                "measured": tan_meas,
                "predicted_pXDelta": tan_pred,
                "err": abs(tan_meas - tan_pred),
            },
            "across": {
                "measured": rad_meas,
                "predicted_pXDelta": rad_pred,
                "err": abs(rad_meas - rad_pred),
            },
            "squeeze_preserves_Q": {
                "Q_before": Q0,
                "Q_after": Qsq,
                "abs_change": abs(Qsq - Q0),
            },
            "random_shift_Q_changes": rand_shifts,
        },
        "squeeze_injection": inj,
    }


# --------------------------------------------------------------------------
# §7.2b certificate payoff A — the router ERASES the latch, the wormhole
# TRANSPORTS it (referee F3, item 1). This is the measured downstream
# consequence of det J: det J = 1 (translation) is injective => the spread of
# the Goldstone charge Q survives the jump; det J = 0 (router, q := exit) is
# non-injective => every incoming state exits with the SAME Q.
# --------------------------------------------------------------------------
def _latch_payoff(cfg):
    dim = max(cfg.dim, 2)
    X = so2_generator(dim)
    f = cfg.latch_radius

    # entrance on the vacuum circle; fixed momentum (the charge carrier)
    a = jnp.zeros(dim).at[0].set(f)
    p = jnp.zeros(dim).at[1].set(cfg.latch_momentum).at[0].set(0.1 * cfg.latch_momentum)

    # coset-tangent channel: Delta _|_ X^T p  =>  p^T X Delta = 0  =>  Delta Q = 0
    Xp = X @ p
    d_tan = jnp.zeros(dim).at[0].set(Xp[1]).at[1].set(-Xp[0])
    d_tan = d_tan / (jnp.linalg.norm(d_tan) + 1e-9) * 0.5
    # across-coset channel: radial => Delta Q = p^T X Delta != 0, but EXACT
    d_rad = a / (jnp.linalg.norm(a) + 1e-9) * 0.5

    # a cloud of incoming states inside the capture ball around the entrance
    rng = np.random.default_rng(cfg.seed0)
    n = cfg.payoff_latch_samples
    jit = rng.normal(size=(n, dim))
    jit = jit / np.linalg.norm(jit, axis=1, keepdims=True)
    jit = jit * (cfg.payoff_capture_jitter * rng.uniform(size=(n, 1)) ** (1.0 / dim))
    q_in = jnp.asarray(np.asarray(a)[None, :] + jit, dtype=jnp.float32)

    Q_in = np.asarray(jax.vmap(lambda qq: p @ X @ qq)(q_in))

    def _arm(q_out, predicted):
        Q_out = np.asarray(jax.vmap(lambda qq: p @ X @ qq)(q_out))
        dQ = Q_out - Q_in
        row = {
            "dQ_mean": float(np.mean(dQ)),
            "dQ_std": float(np.std(dQ)),
            "dQ_min": float(np.min(dQ)),
            "dQ_max": float(np.max(dQ)),
            "Q_out_std": float(np.std(Q_out)),
            "Q_out_std_over_Q_in_std": float(np.std(Q_out) / (np.std(Q_in) + 1e-12)),
            "Q_in": [float(x) for x in Q_in],
            "Q_out": [float(x) for x in Q_out],
        }
        if predicted is None:
            row["predicted_pXDelta"] = None
            row["max_abs_err_vs_prediction"] = None  # no receipt to check against
        else:
            row["predicted_pXDelta"] = float(predicted)
            row["max_abs_err_vs_prediction"] = float(np.max(np.abs(dQ - predicted)))
        return row

    arms = {}

    # -- wormhole, coset-tangent channel: Delta Q = 0 for EVERY incoming state --
    wc_t = WormholeChannels(
        entrances=a[None, :],
        exits=(a + d_tan)[None, :],
        radii=jnp.array([cfg.capture_radius]),
    )
    q_out = jax.vmap(lambda qq: wc_t.jump(qq, p)[0])(q_in)
    arms["wormhole_coset_tangent"] = _arm(q_out, float(p @ X @ d_tan))

    # -- wormhole, across-coset channel: Delta Q = p^T X Delta, exact, constant --
    wc_r = WormholeChannels(
        entrances=a[None, :],
        exits=(a + d_rad)[None, :],
        radii=jnp.array([cfg.capture_radius]),
    )
    q_out = jax.vmap(lambda qq: wc_r.jump(qq, p)[0])(q_in)
    arms["wormhole_across_coset"] = _arm(q_out, float(p @ X @ d_rad))

    # -- random-shift translation: det J = 1 but NO matched channel => Q scrambled.
    #    (Volume preservation alone is NOT the latch certificate -- report honestly.)
    shifts = jnp.asarray(rng.normal(size=(n, dim)) * 0.5, dtype=jnp.float32)
    arms["random_shift"] = _arm(q_in + shifts, None)

    # -- no-physics router: q := exit for EVERY incoming state (det J = 0) --
    exit_pt = a + d_tan  # same landing site as the certified wormhole arm
    q_out = jnp.broadcast_to(exit_pt, (n, dim))
    arms["no_physics_router"] = _arm(q_out, None)

    # det J per arm, measured (not asserted)
    def _detJ(qmap):
        def full(qp):
            return jnp.concatenate([qmap(qp[:dim]), qp[dim:]])

        return float(jnp.linalg.det(jax.jacfwd(full)(jnp.concatenate([a, p]))))

    detJ = {
        "wormhole_coset_tangent": _detJ(lambda qq: wc_t.jump(qq, p)[0]),
        "wormhole_across_coset": _detJ(lambda qq: wc_r.jump(qq, p)[0]),
        "random_shift": _detJ(lambda qq: qq + shifts[0]),
        "no_physics_router": _detJ(lambda qq: exit_pt),
    }

    # invertibility: can the incoming state be recovered from the outgoing one?
    # wormhole: q_in = q_out - Delta (exact). router: information destroyed.
    q_back = jax.vmap(lambda qq: wc_t.jump(qq, p)[0])(q_in) - d_tan
    recon_err = {
        "wormhole_coset_tangent": float(np.max(np.abs(np.asarray(q_back - q_in)))),
        "no_physics_router": None,  # non-invertible: det J = 0
    }

    return {
        "n_samples": n,
        "capture_jitter": cfg.payoff_capture_jitter,
        "Q_in_std": float(np.std(Q_in)),
        "arms": arms,
        "det_J": detJ,
        "reconstruction_err": recon_err,
    }


# --------------------------------------------------------------------------
# §7.4 certificate payoff B — coercive-exit BIBO (referee F3 item 2 /
# paid-access-theory §7 issue 7). The wormhole's receipt (energy ledger
# Delta H = V(b) - V(a) + coercive-component membership of the exit) SCREENS
# inadmissible exits; the no-physics router has neither V nor a Jacobian, so it
# cannot form the receipt at all and blows up BIBO on the non-coercive exits.
# The `wormhole_blind` ablation isolates *the receipt* as the operative cause.
# --------------------------------------------------------------------------
def _bibo_battery(cfg, seeds):
    dim = cfg.dim
    dt = cfg.dt
    band = list(cfg.mass_band)
    T = cfg.bibo_steps
    gam = cfg.bibo_gamma

    pot = NonCoerciveWell(k=cfg.bibo_k, eps=cfg.bibo_quartic, conf=cfg.basin_curvature)
    x_b = pot.barrier_top()
    V_b = pot.barrier_height()

    arms = ["wormhole_certified", "wormhole_blind", "no_physics_router"]
    rstar_T = {a: {} for a in arms}
    rstar_2T = {a: {} for a in arms}
    escaped = {a: {} for a in arms}
    receipts = {}

    for b in cfg.bibo_exit_distances:
        b = float(b)
        exit_pt = jnp.zeros(dim).at[0].set(b)
        channels = WormholeChannels(
            entrances=jnp.zeros((1, dim)),
            exits=exit_pt[None, :],
            radii=jnp.array([cfg.capture_radius]),
        )
        for a in arms:
            rstar_T[a][str(b)] = []
            rstar_2T[a][str(b)] = []
            escaped[a][str(b)] = []

        for s in seeds:
            k_init, k_rel = jax.random.split(jax.random.PRNGKey(s), 2)
            q0 = jnp.zeros(dim)
            p0 = jnp.zeros(dim).at[0].set(cfg.bibo_init_momentum)
            p0 = p0 + 0.02 * jax.random.normal(k_init, (dim,))

            rel = _make_clu(dim, "relativistic", cfg.c, cfg.rest_mass, band, pot, k_rel)
            # kinetic energy above rest (relativistic T carries a c^2 m0 offset)
            ke0 = float(rel.T(p0) - rel.T(jnp.zeros(dim)))

            # ---- THE RECEIPT (only the wormhole can compute it) ----
            # (i) energy ledger: Delta H = V(exit) - V(entrance), exact, bounded
            ledger = float(channels.ledger(pot, q0))
            # (ii) exit lies in the coercive component containing the origin
            in_component = bool(abs(b) < x_b - cfg.bibo_margin)
            # (iii) post-jump energy cannot climb out of that component
            energy_ok = bool(ke0 + float(pot(exit_pt)) <= V_b - cfg.bibo_margin)
            admissible = in_component and energy_ok
            receipts[str(b)] = {
                "V_exit": float(pot(exit_pt)),
                "ledger_dH": ledger,
                "ledger_err": abs(ledger - float(pot(exit_pt) - pot(q0))),
                "KE0": ke0,
                "H_after_minus_rest": ke0 + float(pot(exit_pt)),
                "in_coercive_component": in_component,
                "energy_below_barrier": energy_ok,
                "admissible": admissible,
                "energy_only_test_would_admit": energy_ok,
            }

            starts = {
                # certified: jump only if the receipt admits the exit; else abort
                "wormhole_certified": (
                    channels.jump(q0, p0)[:2] if admissible else (q0, p0)
                ),
                # blind: same physics, receipt ignored (ablation)
                "wormhole_blind": channels.jump(q0, p0)[:2],
                # router: q := exit, unconditionally (no V, no ledger, no det J)
                "no_physics_router": (exit_pt, p0),
            }
            for a, (qs, ps) in starts.items():
                traj = _rollout(rel, qs, ps, 2 * T, dt, gam)
                r = np.asarray(jnp.linalg.norm(traj[:, :dim], axis=1))
                rT = float(np.max(r[:T]))
                r2T = float(np.max(r))
                rstar_T[a][str(b)].append(rT)
                rstar_2T[a][str(b)].append(r2T)
                escaped[a][str(b)].append(int(r2T > cfg.bibo_escape_radius))

    # receipt-prediction accuracy: does `admissible` predict boundedness?
    n_ok, n_tot = 0, 0
    for b in cfg.bibo_exit_distances:
        adm = receipts[str(float(b))]["admissible"]
        esc = float(np.mean(escaped["wormhole_blind"][str(float(b))])) > 0.5
        n_ok += int(adm == (not esc))
        n_tot += 1

    def _agg(dd):
        return {a: {b: float(np.mean(v)) for b, v in dd[a].items()} for a in arms}

    return {
        "x_b": x_b,
        "V_b": V_b,
        "exit_distances": [float(b) for b in cfg.bibo_exit_distances],
        "steps_T": T,
        "gamma": gam,
        "escape_radius": cfg.bibo_escape_radius,
        "r_star_T": _agg(rstar_T),
        "r_star_2T": _agg(rstar_2T),
        "escape_rate": _agg(escaped),
        "growth_2T_over_T": {
            a: {
                b: float(np.mean(rstar_2T[a][b]) / max(np.mean(rstar_T[a][b]), 1e-9))
                for b in rstar_T[a]
            }
            for a in arms
        },
        "receipts": receipts,
        "receipt_predicts_escape": {"correct": n_ok, "total": n_tot},
    }


def run_experiment_paid_access(
    config=None, save_dir=None, models_dir=None, seed=0, quick=False
):
    """Run the w7 paid-access battery. Writes results JSON + a landing-rate plot.

    Args:
        config: CHLUConfig (uses .experiment_paid_access). Defaults to
            ``get_default_config()``.
        save_dir: directory for plots/results (defaults to cwd/paid_access).
        models_dir: unused (no checkpoints — analytic, no training).
        seed: base seed (offsets cfg.seed0).
        quick: reduced seeds/distances smoke run.
    """
    from chlu.config import get_default_config

    if config is None:
        config = get_default_config()
    cfg = config.experiment_paid_access

    if save_dir is None:
        save_dir = os.path.join(os.getcwd(), "paid_access")
    os.makedirs(save_dir, exist_ok=True)

    distances = list(cfg.basin_distances)
    n_seeds = cfg.n_seeds
    if quick:
        # subsample so the smoke run still straddles the causal box L
        step = max(1, len(distances) // cfg.quick_distances)
        distances = distances[::step][: cfg.quick_distances]
        n_seeds = cfg.quick_seeds
    seeds = [cfg.seed0 + seed + i for i in range(n_seeds)]

    print(
        f"[paid-access] band={cfg.mass_band} c={cfg.c} T={cfg.reach_steps} "
        f"dt={cfg.dt} gamma(reach)=0 seeds={seeds} quick={quick}"
    )

    reach = _reach_battery(cfg, distances, seeds, quick=quick)
    lat = _latch_and_certs(cfg)
    payoff_latch = _latch_payoff(cfg)
    payoff_bibo = _bibo_battery(cfg, seeds)

    results = {
        "config": {
            "dim": cfg.dim,
            "c": cfg.c,
            "rest_mass": cfg.rest_mass,
            "dt": cfg.dt,
            "reach_steps": cfg.reach_steps,
            "mass_band": list(cfg.mass_band),
            "barrier_height": cfg.barrier_height,
            "gamma_reach": 0.0,
            "init_momentum": cfg.init_momentum,
            "zeta_grid": list(cfg.zeta_grid),
            "capture_radius": cfg.capture_radius,
            "throat_depth": cfg.throat_depth,
            "landing_tol": cfg.landing_tol,
            "seeds": seeds,
            "quick": quick,
            "bibo_k": cfg.bibo_k,
            "bibo_quartic": cfg.bibo_quartic,
            "bibo_gamma": cfg.bibo_gamma,
            "bibo_steps": cfg.bibo_steps,
            "bibo_init_momentum": cfg.bibo_init_momentum,
            "payoff_latch_samples": cfg.payoff_latch_samples,
            "payoff_capture_jitter": cfg.payoff_capture_jitter,
        },
        "reach": reach,
        "latch_and_certs": lat,
        "certificate_payoff": {"latch": payoff_latch, "bibo": payoff_bibo},
    }

    # write results
    results_dir = (
        os.path.join(os.path.dirname(save_dir.rstrip("/")), "results")
        if os.path.basename(save_dir.rstrip("/")) == "plots"
        else save_dir
    )
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "paid_access_metrics.json")
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)

    _print_summary(results)
    _plot_landing(results, save_dir)
    _plot_certificate_payoff(results, save_dir)
    print(f"[paid-access] results -> {out_path}")
    return results


def _print_summary(results):
    r = results["reach"]
    L = r["L"]
    print(f"\n=== §7.1 REACH  (L = {L:.3f}, v_max,0 = {r['v_max_0']:.3f}) ===")
    dists = r["distances"]
    header = "arm".ljust(20) + "".join(f"{d:>7.2f}" for d in dists)
    print(header)
    for arm, rates in r["landing_rates"].items():
        row = arm.ljust(20) + "".join(f"{rates[str(d)]:>7.2f}" for d in dists)
        print(row)
    print(f"{'(L)':<20}" + "".join(("  <L " if d < L else "  >L ") for d in dists))
    c = r["certificates"]
    print(f"wormhole det J: {[round(x, 12) for x in c['wormhole_detJ']]}")
    print(f"wormhole ledger err: {[f'{x:.2e}' for x in c['wormhole_ledger_err']]}")

    lat = results["latch_and_certs"]["latch"]
    print("\n=== §7.2 LATCH TRANSIT ===")
    print(
        f"tangent Delta:  measured={lat['tangent']['measured']:.4f} "
        f"pXDelta={lat['tangent']['predicted_pXDelta']:.4f} "
        f"err={lat['tangent']['err']:.2e}"
    )
    print(
        f"across  Delta:  measured={lat['across']['measured']:.4f} "
        f"pXDelta={lat['across']['predicted_pXDelta']:.4f} "
        f"err={lat['across']['err']:.2e}"
    )
    print(f"squeeze preserves Q: dQ={lat['squeeze_preserves_Q']['abs_change']:.2e}")
    print(
        f"random-shift dQ (erases): "
        f"{[round(x, 3) for x in lat['random_shift_Q_changes']]}"
    )

    print("\n=== §7.3 SQUEEZE INJECTION CERTIFICATE ===")
    for row in results["latch_and_certs"]["squeeze_injection"]:
        print(
            f"  zeta={row['zeta']:<4} detS={row['detS']:.4f} "
            f"symp_err={row['symplectic_err']:.1e} H_ratio={row['H_ratio']:.3f} "
            f"<= e^2z={row['e2zeta_bound']:.2f} : {row['bound_holds']}"
        )

    pay = results.get("certificate_payoff")
    if not pay:
        return
    pl = pay["latch"]
    print(
        f"\n=== §7.2b CERTIFICATE PAYOFF A — latch transport vs erasure "
        f"(n={pl['n_samples']} incoming states, std(Q_in)={pl['Q_in_std']:.4f}) ==="
    )
    hdr = (
        "arm".ljust(24)
        + "det J".rjust(8)
        + "dQ mean".rjust(11)
        + "dQ std".rjust(11)
        + "pred pXD".rjust(11)
        + "err".rjust(10)
        + "std(Q_out)".rjust(12)
    )
    print(hdr)
    for arm, row in pl["arms"].items():
        pred = row["predicted_pXDelta"]
        err = row["max_abs_err_vs_prediction"]
        print(
            arm.ljust(24)
            + f"{pl['det_J'][arm]:>8.4f}"
            + f"{row['dQ_mean']:>11.4f}"
            + f"{row['dQ_std']:>11.4f}"
            + (f"{pred:>11.4f}" if pred is not None else "  (none)  ".rjust(11))
            + (f"{err:>10.1e}" if err is not None else "     -    ".rjust(10))
            + f"{row['Q_out_std']:>12.2e}"
        )
    print(
        "  -> router det J = 0 (non-injective): std(Q_out) = 0 => the latch is "
        "ERASED. Wormhole det J = 1: std(Q_out) = std(Q_in) => TRANSPORTED."
    )

    pb = pay["bibo"]
    print(
        f"\n=== §7.4 CERTIFICATE PAYOFF B — coercive-exit BIBO "
        f"(x_b={pb['x_b']:.3f}, V_b={pb['V_b']:.3f}, gamma={pb['gamma']}, "
        f"T={pb['steps_T']}) ==="
    )
    exits = pb["exit_distances"]
    print("arm".ljust(22) + "".join(f"{b:>9.2f}" for b in exits) + "   metric")
    for arm in pb["r_star_2T"]:
        row = pb["r_star_2T"][arm]
        print(
            arm.ljust(22)
            + "".join(f"{row[str(b)]:>9.2f}" for b in exits)
            + "   r*(2T)"
        )
    for arm in pb["escape_rate"]:
        row = pb["escape_rate"][arm]
        print(
            arm.ljust(22)
            + "".join(f"{row[str(b)]:>9.2f}" for b in exits)
            + "   escape rate"
        )
    print(
        "receipt".ljust(22)
        + "".join(
            f"{('ADMIT' if pb['receipts'][str(b)]['admissible'] else 'REJECT'):>9}"
            for b in exits
        )
        + "   wormhole receipt"
    )
    print(
        "energy-only test".ljust(22)
        + "".join(
            f"{('admit' if pb['receipts'][str(b)]['energy_below_barrier'] else 'reject'):>9}"
            for b in exits
        )
        + "   (INSUFFICIENT)"
    )
    rp = pb["receipt_predicts_escape"]
    print(f"  -> receipt predicts BIBO blow-up {rp['correct']}/{rp['total']} exits.")


def _plot_landing(results, save_dir):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"[paid-access] plotting skipped: {e}")
        return
    r = results["reach"]
    dists = r["distances"]
    L = r["L"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for arm, rates in r["landing_rates"].items():
        ys = [rates[str(d)] for d in dists]
        ax.plot(dists, ys, marker="o", label=arm)
    ax.axvline(L, color="k", ls="--", lw=1, label=f"causal box L={L:.2f}")
    ax.set_xlabel("basin distance d")
    ax.set_ylabel("landing rate")
    ax.set_ylim(-0.05, 1.08)
    ax.set_title("§7.1 reach: landing vs distance (crossover at d=L)")
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    path = os.path.join(save_dir, "paid_access_reach.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[paid-access] plot -> {path}")


def _plot_certificate_payoff(results, save_dir):
    """Two-panel figure the V1 short can embed: (A) the router erases the latch
    the wormhole transports (det J = 0 vs 1); (B) an uncertified exit into the
    non-coercive region blows up BIBO, the receipt-screened wormhole stays
    bounded."""
    pay = results.get("certificate_payoff")
    if not pay:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"[paid-access] payoff plot skipped: {e}")
        return

    pl, pb = pay["latch"], pay["bibo"]
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.2))

    # -- Panel A: Q_out vs Q_in (slope 1 = transported, slope 0 = erased) --
    style = {
        "wormhole_coset_tangent": ("tab:green", "o", "wormhole, coset-tangent (det J=1)"),
        "wormhole_across_coset": ("tab:blue", "s", "wormhole, across-coset (det J=1)"),
        "random_shift": ("tab:orange", "^", "random shift (det J=1, no channel)"),
        "no_physics_router": ("tab:purple", "x", "no-physics router (det J=0)"),
    }
    for arm, row in pl["arms"].items():
        col, mk, lab = style[arm]
        axA.plot(row["Q_in"], row["Q_out"], mk, color=col, ms=5, ls="none", label=lab)
    lo, hi = min(pl["arms"]["wormhole_coset_tangent"]["Q_in"]), max(
        pl["arms"]["wormhole_coset_tangent"]["Q_in"]
    )
    axA.plot([lo, hi], [lo, hi], "k--", lw=0.8, label="identity (Q preserved)")
    axA.set_xlabel("incoming charge  $Q_{in} = p^\\top X q$")
    axA.set_ylabel("outgoing charge  $Q_{out}$")
    axA.set_title("A. latch: transported (slope 1) vs erased (slope 0)")
    axA.legend(fontsize=6.5, loc="best")

    # -- Panel B: r*(2T) vs requested exit distance --
    # NOTE: wormhole_blind and no_physics_router coincide exactly (same landing
    # site, neither screened) -- draw the ablation as a thick translucent band
    # UNDER the router so neither curve is occluded (referee F1 lesson).
    exits = pb["exit_distances"]
    styleB = {
        "wormhole_blind": (
            dict(color="tab:red", marker="s", lw=5, alpha=0.35, ms=9),
            "wormhole, receipt ignored (ablation)",
        ),
        "no_physics_router": (
            dict(color="tab:purple", marker="x", lw=1.4, ls="--", ms=7),
            "no-physics router, no receipt (overlaps ablation)",
        ),
        "wormhole_certified": (
            dict(color="tab:green", marker="o", lw=1.8, ms=6),
            "wormhole + receipt (screened)",
        ),
    }
    for arm, (kw, lab) in styleB.items():
        rows = pb["r_star_2T"][arm]
        axB.plot(exits, [rows[str(b)] for b in exits], label=lab, **kw)
    axB.axvline(pb["x_b"], color="k", ls="--", lw=1, label=f"coercive edge $x_b$={pb['x_b']:.2f}")
    axB.axhline(
        pb["escape_radius"], color="grey", ls=":", lw=1, label="escape radius"
    )
    axB.set_yscale("log")
    axB.set_xlabel("requested exit distance  $b$")
    axB.set_ylabel("$r^* = \\max_t \\|q_t\\|$  over 2T steps")
    axB.set_title("B. BIBO: uncertified exit escapes, receipt keeps it bounded")
    axB.legend(fontsize=6.5, loc="best")

    fig.tight_layout()
    path = os.path.join(save_dir, "paid_access_certificate_payoff.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[paid-access] payoff plot -> {path}")
