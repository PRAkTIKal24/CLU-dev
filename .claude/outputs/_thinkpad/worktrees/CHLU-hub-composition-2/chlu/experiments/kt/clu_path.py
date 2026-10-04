"""
The REAL CLU-path KT measurements (JAX).

Real code path: ``CLULattice`` + ``channel_spring_coupling(kappa)`` +
``langevin_step(noise_mode="fdt")``, ``newtonian_learned`` units carrying a
``MexicanHatPotential`` vacuum ring, **no governor**. Extends ``xy-1d-control``'s
s4b equilibrium-start protocol from N=2 to a 2-D torus.

Three measurements:

``run_bridge``       L x L torus: CLU-Langevin stationary rho_s & <cos dtheta>
                     vs reduced-XY  -> the KILL CRITERION (does the reduction
                     hold at scale on the real path?).
``run_winding_msd``  1-D CLU ring: winding MSD slope (bias-free slip rate) vs N
                     -> ``tau ~ 1/N`` (memory DEGRADES with size). **This is
                     CSF3 soft exponent (b), and it is NOT simply a compute
                     shortfall** — see the warning on ``run_winding_msd``: the
                     laptop's -0.7 at T/J=1.0 is very likely a SATURATION
                     artifact of the fit window, not the xi~1.2 effect it was
                     ascribed to, and lowering T to 0.5 makes the slope flatter,
                     not steeper. Read that docstring before sizing a run.
``run_winding_count`` the same null via slip counting (biased low by intra-chunk
                     slip cancellation; kept for cross-check only — MSD is the
                     estimator of record).

⚠ Settings discipline (handover §7.22): ``assert_kt_settings`` enforces float64
and ``noise_mode="fdt"``. Under the repo-default ``"legacy"`` noise, T is NOT in
energy units and none of this physics holds.
"""

import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import i0, i1

from ...core.chlu_unit import CHLU
from ...core.integrators import langevin_step
from ...core.lattice import CLULattice, channel_spring_coupling, torus_edges
from ..goldstone_harness import MexicanHatPotential, log_mass_for_inertia
from .reduced_xy import reduced_rho_s, xy_equilibrium_2d

TWO_PI = 2 * np.pi


# ------------------------------------------------------------- guardrails ----
def assert_kt_settings(noise_mode: str, kinetic_mode: str, use_governor: bool = False):
    """Fail loudly on a misconfigured run rather than silently produce garbage.

    A KT run under ``legacy`` noise or a governor is not a KT run: T stops being
    a temperature in energy units (§7.22) and the governor breaks detailed
    balance, so neither ``rho_s`` nor ``T_KT`` means anything.
    """
    if not jax.config.read("jax_enable_x64"):
        raise RuntimeError(
            "KT runs require float64: call jax.config.update('jax_enable_x64', True) "
            "BEFORE any array is created. float32 Langevin at gamma=0.1, dt=0.02 "
            "does not resolve the winding-slip rates this experiment measures."
        )
    if noise_mode != "fdt":
        raise RuntimeError(
            f"langevin_noise={noise_mode!r} but KT requires 'fdt' (handover §7.22). "
            "Under the repo-default 'legacy' noise T is NOT in energy units, so "
            "T_KT = 1.786 kappa r*^2 = 0.0893 and rho_s/T -> 2/pi are both void."
        )
    if kinetic_mode != "newtonian_learned":
        raise RuntimeError(
            f"kinetic_mode={kinetic_mode!r} but the validated KT path is "
            "'newtonian_learned' (relativistic T(p) changes the vacuum inertia "
            "and hence the XY dictionary J = 2 kappa r*^2)."
        )
    if use_governor:
        raise RuntimeError(
            "use_governor=True breaks detailed balance; the KT sampler must be "
            "a plain Langevin thermostat (no governor)."
        )


# ------------------------------------------------------------- construction --
def make_unit(lam: float, f: float, unit_key_seed: int = 0):
    """One designed SO(2) register: Mexican-hat vacuum ring, unit channel inertia."""
    u = CHLU(
        dim=2,
        hidden=4,
        key=jax.random.PRNGKey(unit_key_seed),
        kinetic_mode="newtonian_learned",
    )
    u = eqx.tree_at(
        lambda m: m.potential_net, u, MexicanHatPotential(lam=lam, f=f, k_spec=None)
    )
    u = eqx.tree_at(lambda m: m.log_mass, u, log_mass_for_inertia(jnp.array([1.0, 1.0])))
    return u


def make_lattice(edges, n_units: int, kappa: float, lam: float = 1.0, f: float = 1.0):
    units = tuple(make_unit(lam, f) for _ in range(n_units))
    cps = tuple(channel_spring_coupling(2, 2, kappa, channel=(0, 1)) for _ in edges)
    return CLULattice(units=units, edges=tuple(edges), couplings=cps)


def make_ring(N: int, kappa: float, lam: float = 1.0, f: float = 1.0):
    edges = [(i, (i + 1) % N) for i in range(N)]
    return make_lattice(edges, N, kappa, lam, f)


def angles(q, N):
    return np.arctan2(q[..., 1::2], q[..., 0::2])  # (..., N)


def u_xy(T, J_xy):
    """1-D XY nearest-neighbour bond correlation <cos dtheta> = I1(J/T)/I0(J/T);
    carried as a reference column in the bridge rows (not used by postproc)."""
    return float(i1(J_xy / T) / i0(J_xy / T))


def ring_winding(q, N):
    th = np.arctan2(q[:, 1::2], q[:, 0::2])  # (NW, N)
    d = th - np.roll(th, -1, axis=1)
    pv = (d + np.pi) % TWO_PI - np.pi
    return np.sum(pv, axis=1) / TWO_PI


# --------------------------------------------------------------- sampler -----
def make_chunker(lat, T, gamma, dt, NW, noise_mode="fdt"):
    """jit-compiled `CH`-step Langevin chunk, vmapped over NW walkers."""
    m_eff = lat.effective_mass()

    def step_all(state, k):
        q, p = state
        ks = jax.random.split(k, NW)
        qn, pn, _ = jax.vmap(
            lambda a, b, kk: langevin_step(
                lat.H, a, b, dt, gamma, T, kk, noise_mode=noise_mode, m_eff=m_eff
            )
        )(q, p, ks)
        return (qn, pn), None

    @jax.jit
    def chunk(state, keys):
        s, _ = jax.lax.scan(step_all, state, keys)
        return s

    return chunk, m_eff


def _init_momentum(T, m_eff, key, NW, dim):
    return jnp.sqrt(T * m_eff) * jax.random.normal(key, (NW, dim))


# ============================================================ E: kill crit ====
def run_bridge(
    L=8,
    tjs=(0.70, 0.85, 1.00),
    kappa=0.05,
    rstar=1.0,
    lam=1.0,
    f=1.0,
    dt=0.02,
    gamma=0.10,
    NW=256,
    n_chunks=40,
    burn_chunks=10,
    CH=200,
    seed=7,
    equil_seed=1234,
    equil_sweeps=1500,
    noise_mode="fdt",
    log=print,
):
    """L x L torus: does CLU-Langevin reproduce reduced-XY rho_s? (kill criterion)

    Started from the reduced-XY equilibrium; stationarity checked by first/second
    half drift. Laptop result at L=8: rho ratio 0.980/0.957/0.931 at
    T/J=0.70/0.85/1.00 — a monotone deficit growing with T, the pre-declared
    Born-Oppenheimer + thermal radial dressing signature.
    """
    assert_kt_settings(noise_mode, "newtonian_learned")
    J_xy = 2 * kappa * rstar**2
    N = L * L
    edges = torus_edges(L)
    lat = make_lattice(edges, N, kappa, lam, f)
    idx = np.arange(N).reshape(L, L)
    xbonds = [
        (int(idx[y, x]), int(idx[y, (x + 1) % L])) for y in range(L) for x in range(L)
    ]
    all_bonds = [(i, j) for (i, j) in edges]

    rows = []
    for TJ in tjs:
        T = TJ * J_xy
        t0 = time.time()
        th0 = xy_equilibrium_2d(L, TJ, NW, seed=equil_seed, nsweeps=equil_sweeps)
        rho_red = reduced_rho_s(th0, L, TJ)
        cos_red = float(
            np.mean([np.mean(np.cos(th0[:, i] - th0[:, j])) for (i, j) in all_bonds])
        )
        q0 = np.zeros((NW, 2 * N))
        q0[:, 0::2] = rstar * np.cos(th0)
        q0[:, 1::2] = rstar * np.sin(th0)

        chunk, m_eff = make_chunker(lat, T, gamma, dt, NW, noise_mode)
        key = jax.random.PRNGKey(seed)
        k2, key = jax.random.split(key)
        state = (jnp.asarray(q0), _init_momentum(T, m_eff, k2, NW, lat.dim))

        e1 = e2 = I1 = I2 = 0.0
        cn = [0, 0]
        cosb = [0.0, 0.0]
        for it in range(n_chunks):
            key, sub = jax.random.split(key)
            state = chunk(state, jax.random.split(sub, CH))
            if it >= burn_chunks:
                th = angles(np.asarray(state[0]), N)  # (NW, N)
                dx = np.stack([th[:, i] - th[:, j] for (i, j) in xbonds], 1)
                e = np.mean(np.sum(np.cos(dx), 1))
                Ix = np.mean(np.sum(np.sin(dx), 1) ** 2)
                cb = np.mean(
                    [np.mean(np.cos(th[:, i] - th[:, j])) for (i, j) in all_bonds]
                )
                half = 0 if (it - burn_chunks) < (n_chunks - burn_chunks) / 2 else 1
                if half == 0:
                    e1 += e
                    I1 += Ix
                    cn[0] += 1
                    cosb[0] += cb
                else:
                    e2 += e
                    I2 += Ix
                    cn[1] += 1
                    cosb[1] += cb
        e_m = (e1 + e2) / sum(cn)
        I_m = (I1 + I2) / sum(cn)
        rho_clu = float((e_m - I_m / TJ) / N)
        cos_clu = float((cosb[0] + cosb[1]) / sum(cn))
        drift = float(abs((e2 / cn[1] - e1 / cn[0]) / N))
        row = dict(
            L=L,
            TJ=TJ,
            T=T,
            rho_reduced=rho_red,
            rho_clu=rho_clu,
            rho_ratio=rho_clu / rho_red,
            cos_reduced=cos_red,
            cos_clu=cos_clu,
            cos_ratio=cos_clu / cos_red,
            u_xy=u_xy(T, J_xy),
            drift=drift,
            secs=time.time() - t0,
        )
        rows.append(row)
        log(
            f"  bridge L={L} T/J={TJ:.2f} rho: clu={rho_clu:.4f} red={rho_red:.4f} "
            f"ratio={rho_clu / rho_red:.3f} | cos ratio={cos_clu / cos_red:.3f} "
            f"| drift={drift:.4f} ({time.time() - t0:.0f}s)"
        )
    return rows


# ================================================== A: 1-D winding (MSD) ======
def run_winding_msd(
    N,
    TJ,
    n_chunks,
    CH=100,
    seed=31,
    kappa=0.05,
    rstar=1.0,
    lam=1.0,
    f=1.0,
    dt=0.02,
    gamma=0.10,
    NW=256,
    noise_mode="fdt",
    msd_fit_max=None,
):
    """Bias-free 1-D winding slip rate: MSD(t) = <(W(t)-W(0))^2> grows as rate*t
    for a Poisson +-1 winding walk. Unlike counting |Delta round(W)| per chunk,
    MSD is immune to intra-chunk slip cancellation. Predict rate ~ N (=> tau ~ 1/N).

    ⚠ **The through-origin fit is only valid while the walk is DIFFUSIVE.** Once
    MSD saturates (the winding has fully decorrelated, MSD ~ O(1)), the fitted
    "rate" is a saturation artifact that decreases with the fit window: measured
    at N=8, T/J=1.0, seed 31, the same run gives rate 2.5e-4 over t<=2500 but
    4.0e-5 over t<=50000 (6x), and the apparent N-scaling collapses from 0.39
    to ~0 as the window shortens. At T/J=1.0 the ring winding decorrelates in
    ~10^3 steps (E_wind(N=8,w=1)/T = 2.3, i.e. barely metastable), so the
    laptop's 3x10^4-step runs were saturation-dominated — this is the likeliest
    origin of the soft -0.7 slope, not the xi~1.2 explanation.

    ``msd_fit_max``: if set, fit only the initial segment with MSD <= this value
    (e.g. 0.3), i.e. the genuinely diffusive window. ``None`` (default)
    reproduces the original full-range fit bit-exactly. The reported
    ``fit_points``/``fit_tmax`` say which window was used — if ``fit_points``
    is small, the run is under-resolved rather than the rate being small.
    """
    assert_kt_settings(noise_mode, "newtonian_learned")
    J_xy = 2 * kappa * rstar**2
    T = TJ * J_xy
    lat = make_ring(N, kappa, lam, f)
    # init winding w = +1 : theta_i = -2*pi*i/N gives sum pv = +1
    th0 = (-TWO_PI * np.arange(N) / N)[None, :].repeat(NW, 0)
    q0 = np.zeros((NW, 2 * N))
    q0[:, 0::2] = rstar * np.cos(th0)
    q0[:, 1::2] = rstar * np.sin(th0)

    chunk, m_eff = make_chunker(lat, T, gamma, dt, NW, noise_mode)
    key = jax.random.PRNGKey(seed)
    k2, key = jax.random.split(key)
    state = (jnp.asarray(q0), _init_momentum(T, m_eff, k2, NW, lat.dim))

    W0 = ring_winding(np.asarray(state[0]), N)
    ts, msd = [], []
    for it in range(1, n_chunks + 1):
        key, sub = jax.random.split(key)
        state = chunk(state, jax.random.split(sub, CH))
        W = ring_winding(np.asarray(state[0]), N)
        ts.append(it * CH)
        msd.append(float(np.mean((W - W0) ** 2)))
    ts = np.array(ts)
    msd = np.array(msd)
    # fit through origin: rate = <t*msd>/<t*t>, over the diffusive window only
    tsf, msdf = ts, msd
    if msd_fit_max is not None:
        keep = np.searchsorted(np.maximum.accumulate(msd), msd_fit_max, side="left")
        keep = int(max(2, min(keep, len(msd))))
        tsf, msdf = ts[:keep], msd[:keep]
    rate = float(np.sum(tsf * msdf) / np.sum(tsf * tsf))
    return dict(
        N=N,
        TJ=TJ,
        T=T,
        seed=seed,
        nwalk=NW,
        steps=int(ts[-1]),
        slip_rate_msd=rate,
        msd_fit_max=msd_fit_max,
        fit_points=int(len(tsf)),
        fit_tmax=int(tsf[-1]),
        rate_over_N=float(rate / N),
        msd_final=float(msd[-1]),
        ts=ts.tolist(),
        msd=msd.tolist(),
    )


# ============================================ A': 1-D winding (counting) ======
def run_winding_count(
    N,
    TJ,
    n_chunks,
    CH=200,
    seed=21,
    kappa=0.05,
    rstar=1.0,
    lam=1.0,
    f=1.0,
    dt=0.02,
    gamma=0.10,
    NW=256,
    noise_mode="fdt",
):
    """Slip-counting cross-check of the 1-D null (biased LOW vs MSD)."""
    assert_kt_settings(noise_mode, "newtonian_learned")
    J_xy = 2 * kappa * rstar**2
    T = TJ * J_xy
    lat = make_ring(N, kappa, lam, f)
    th0 = (-TWO_PI * np.arange(N) / N)[None, :].repeat(NW, 0)
    q0 = np.zeros((NW, 2 * N))
    q0[:, 0::2] = rstar * np.cos(th0)
    q0[:, 1::2] = rstar * np.sin(th0)

    chunk, m_eff = make_chunker(lat, T, gamma, dt, NW, noise_mode)
    key = jax.random.PRNGKey(seed)
    k2, key = jax.random.split(key)
    state = (jnp.asarray(q0), _init_momentum(T, m_eff, k2, NW, lat.dim))

    W_prev = np.round(ring_winding(np.asarray(state[0]), N))
    w0 = float(np.mean(W_prev))
    total = 0
    steps = 0
    surv = np.ones(NW, bool)
    for _ in range(n_chunks):
        key, sub = jax.random.split(key)
        state = chunk(state, jax.random.split(sub, CH))
        steps += CH
        W = np.round(ring_winding(np.asarray(state[0]), N))
        total += int(np.sum(np.abs(W - W_prev)))
        W_prev = W
        surv &= W == 1
    rate = total / (NW * steps)
    return dict(
        N=N,
        TJ=TJ,
        T=T,
        w0=w0,
        seed=seed,
        total_slips=total,
        steps=steps,
        nwalk=NW,
        slip_rate=float(rate),
        N_times_rate=float(N * rate),
        frac_surv=float(np.mean(surv)),
    )


def loglog_slope(xs, ys):
    """d ln y / d ln x by least squares; nan if fewer than 2 positive points."""
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    good = (ys > 0) & (xs > 0)
    if good.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(xs[good]), np.log(ys[good]), 1)[0])
