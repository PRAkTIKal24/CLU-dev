"""
Reduced-XY Monte-Carlo phase diagram (pure numpy, no JAX).

The theorist's route (a): equilibrium KT observables from the reduced model at
``J = 2 kappa r*^2``. Work in T/J units (J=1). The CLU dictionary maps back via

    T_KT = 1.786 kappa r*^2 = **0.0893** CLU units at kappa=0.05  (= 0.8929 J).

⚠ An earlier note stated "0.1786" here; that is wrong by a factor 2 and is
retracted (``kt-2d-csf3`` reconciliation list). Never propagate 0.1786.

Sections:
  B  rho_s(T,L) two routes (fluctuation + twist-response), C(r)/eta, n_v
     -> the 2/pi jump
  D  winding survival tau(L) at T below/above T_KT
     -> the memory exponent sign change (the CSF3 soft exponent (a))
  F  broken-symmetry null: XY + h2 cos(2 theta) -> no 2/pi jump  (P5)

The update scheme (checkerboard overrelaxation + Metropolis, cold/aligned start
so the ensemble stays in the w=0 winding sector) is preserved verbatim from the
validated laptop run: a hot start traps metastable winding at low T, which
blows up <I^2> and makes rho_s spuriously large/negative. Helicity modulus is a
w=0 observable.
"""

import numpy as np

TWO_PI = 2 * np.pi


# ------------------------------------------------------------------ lattice ---
def neighbors(L):
    """index maps for +x,+y,-x,-y on an L x L torus. site = x + L*y."""
    idx = np.arange(L * L).reshape(L, L)  # [y, x]
    xp = np.roll(idx, -1, axis=1)
    xm = np.roll(idx, 1, axis=1)
    yp = np.roll(idx, -1, axis=0)
    ym = np.roll(idx, 1, axis=0)
    return idx, xp.ravel(), xm.ravel(), yp.ravel(), ym.ravel()


def checkerboard(L):
    yy, xx = np.mgrid[0:L, 0:L]
    black = ((xx + yy) % 2 == 0).ravel()
    return black, ~black


# --------------------------------------------------------------- MC updates ---
def sweep_metropolis(th, L, beta, nbrs, masks, delta, h2=0.0, rng=None):
    _, xp, xm, yp, ym = nbrs
    for m in masks:
        prop = th[m] + rng.uniform(-delta, delta, m.sum())
        # local field energy: -J sum cos(th_i - th_nbr)  (+ h2 cos 2th_i for null)
        s = th
        old = -(
            np.cos(s[m] - s[xp[m]])
            + np.cos(s[m] - s[xm[m]])
            + np.cos(s[m] - s[yp[m]])
            + np.cos(s[m] - s[ym[m]])
        )
        new = -(
            np.cos(prop - s[xp[m]])
            + np.cos(prop - s[xm[m]])
            + np.cos(prop - s[yp[m]])
            + np.cos(prop - s[ym[m]])
        )
        if h2 != 0.0:
            old = old + h2 * np.cos(2 * s[m])
            new = new + h2 * np.cos(2 * prop)
        dE = new - old
        acc = rng.random(m.sum()) < np.exp(-beta * dE)
        th[m] = np.where(acc, prop, th[m])
    return th


def sweep_overrelax(th, L, nbrs, masks):
    _, xp, xm, yp, ym = nbrs
    for m in masks:
        s = th
        fx = np.cos(s[xp[m]]) + np.cos(s[xm[m]]) + np.cos(s[yp[m]]) + np.cos(s[ym[m]])
        fy = np.sin(s[xp[m]]) + np.sin(s[xm[m]]) + np.sin(s[yp[m]]) + np.sin(s[ym[m]])
        phi = np.arctan2(fy, fx)
        th[m] = np.mod(2 * phi - th[m], TWO_PI)
    return th


# --------------------------------------------------------------- observables --
def helicity_fluct(th, L, beta, nbrs):
    """Route A: rho_s = (1/N)[<e_x> - beta <Ix^2>], x-bonds. (J=1)"""
    _, xp, _, _, _ = nbrs
    dx = th - th[xp]
    e = np.sum(np.cos(dx))
    Ix = np.sum(np.sin(dx))
    return e, Ix  # accumulate <e>, <I^2>; combine later


def sweep_twist(th, L, beta, nbrs, masks, delta, a, rng, overrelax=False):
    """One sweep of the ENSEMBLE EQUILIBRATED AT per-x-bond twist `a`:
    energy = -J sum_x cos(dtheta_x - a) - J sum_y cos(dtheta_y). +x neighbor gets
    effective angle theta+a, -x neighbor gets theta-a (opposite side of the bond)."""
    _, xp, xm, yp, ym = nbrs
    s = th
    for m in masks:
        if overrelax:
            fx = (
                np.cos(s[xp[m]] + a)
                + np.cos(s[xm[m]] - a)
                + np.cos(s[yp[m]])
                + np.cos(s[ym[m]])
            )
            fy = (
                np.sin(s[xp[m]] + a)
                + np.sin(s[xm[m]] - a)
                + np.sin(s[yp[m]])
                + np.sin(s[ym[m]])
            )
            th[m] = np.mod(2 * np.arctan2(fy, fx) - th[m], TWO_PI)
        else:
            prop = th[m] + rng.uniform(-delta, delta, m.sum())

            def E(u, m=m):
                return -(
                    np.cos(u - s[xp[m]] - a)
                    + np.cos(u - s[xm[m]] + a)
                    + np.cos(u - s[yp[m]])
                    + np.cos(u - s[ym[m]])
                )

            dE = E(prop) - E(th[m])
            acc = rng.random(m.sum()) < np.exp(-beta * dE)
            th[m] = np.where(acc, prop, th[m])
    return th


def torque_x(th, a, nbrs):
    """g(a) = dF/da = <dH/da> = -J <sum_x sin(dtheta_x - a)>  (per config)."""
    _, xp, _, _, _ = nbrs
    dx = th - th[xp]
    return -np.sum(np.sin(dx - a))


def winding_x(th, L):
    """mean over rows of the x-winding n_x(y) = (1/2pi) sum_x pv(theta_i - theta_{i+x})."""
    grid = th.reshape(L, L)  # [y, x]
    d = grid - np.roll(grid, -1, axis=1)
    pv = (d + np.pi) % TWO_PI - np.pi
    n = np.sum(pv, axis=1) / TWO_PI  # per-row winding (near integer)
    return float(np.mean(n))


def run_winding(L, TJ, seed, nwalk, n_max, delta=1.0, check_every=2):
    """Model-A dynamics: init winding w_x=1, single-spin Metropolis (local, winding-
    preserving except by phase slips). Return mean first-passage time (sweeps) for
    the mean winding to drop below 0.5. Censored at n_max.

    This is the estimator behind the 2-D "memory improves with L" result and the
    CSF3 soft exponent (a): above ``T_KT`` the AHNS prediction is a NEGATIVE
    exponent ``pi rho_s/T - 2``, which at L <= 16 is masked by vortex-diffusion
    traversal (``~L^2``, positive). Resolving it needs L >= 32.
    """
    rng = np.random.default_rng(seed)
    beta = 1.0 / TJ
    nbrs = neighbors(L)
    black, white = checkerboard(L)
    masks = [black, white]
    N = L * L
    xg = np.arange(L * L) % L  # x-coordinate of each site
    taus = []
    censored = 0
    for _ in range(nwalk):
        s = (TWO_PI * xg / L).astype(float) + 0.01 * rng.standard_normal(N)
        s = np.mod(s, TWO_PI)
        tau = n_max
        for it in range(1, n_max + 1):
            s = sweep_metropolis(s, L, beta, nbrs, masks, delta, rng=rng)
            if it % check_every == 0:
                Wm = winding_x(s, L)
                if abs(Wm) < 0.5:
                    tau = it
                    break
        if tau >= n_max:
            censored += 1
        taus.append(tau)
    return dict(
        L=L,
        TJ=TJ,
        seed=seed,
        tau_mean=float(np.mean(taus)),
        tau_med=float(np.median(taus)),
        censored=censored,
        nwalk=nwalk,
        n_max=n_max,
    )


def corr_fn_x(th, L, rmax):
    """C(r) = <cos(theta(x,y)-theta(x+r,y))> averaged over lattice + walkers."""
    grid = th.reshape(-1, L, L)  # [walker, y, x]
    C = np.zeros(rmax + 1)
    C[0] = 1.0
    for r in range(1, rmax + 1):
        shifted = np.roll(grid, -r, axis=2)
        C[r] = np.mean(np.cos(grid - shifted))
    return C


def vortex_density(th, L, nbrs):
    """plaquette circulation / 2pi, fraction of |charge|=1 plaquettes."""
    grid = th.reshape(-1, L, L)

    def d(a, b):
        x = a - b
        return (x + np.pi) % TWO_PI - np.pi  # principal value

    # plaquette (x,y)->(x+1,y)->(x+1,y+1)->(x,y+1)
    t00 = grid
    t10 = np.roll(grid, -1, axis=2)
    t11 = np.roll(np.roll(grid, -1, axis=2), -1, axis=1)
    t01 = np.roll(grid, -1, axis=1)
    circ = d(t10, t00) + d(t11, t10) + d(t01, t11) + d(t00, t01)
    charge = np.round(circ / TWO_PI)
    return np.mean(np.abs(charge))


# --------------------------------------------------------------- driver -------
def run_cell(L, TJ, seed, nwalk, n_therm, n_meas, meas_every, delta_tw=0.1, h2=0.0):
    rng = np.random.default_rng(seed)
    beta = 1.0 / TJ
    nbrs = neighbors(L)
    black, white = checkerboard(L)
    masks = [black, white]
    N = L * L
    # init: ALIGNED (cold) start -> stays in the w=0 winding sector; a hot start can
    # quench a metastable winding at low T, whose net current blows up <I_x^2> and
    # makes rho_s spuriously large/negative. Helicity modulus is a w=0 observable.
    th = 0.1 * rng.standard_normal((nwalk, N))
    e_acc = I2_acc = nv_acc = 0.0
    C_acc = np.zeros(9)
    cnt = 0
    drift = np.zeros(2)  # first/second half <e>
    dcnt = [0, 0]
    delta_met = 2.0 if TJ > 0.9 else 1.2
    for w in range(nwalk):
        s = th[w].copy()
        for _ in range(n_therm):
            s = sweep_overrelax(s, L, nbrs, masks)
            s = sweep_metropolis(s, L, beta, nbrs, masks, delta_met, h2=h2, rng=rng)
        m = 0
        for it in range(n_meas):
            s = sweep_overrelax(s, L, nbrs, masks)
            s = sweep_metropolis(s, L, beta, nbrs, masks, delta_met, h2=h2, rng=rng)
            if it % meas_every == 0:
                e, Ix = helicity_fluct(s, L, beta, nbrs)
                e_acc += e
                I2_acc += Ix * Ix
                nv_acc += vortex_density(s.reshape(1, N), L, nbrs)
                C_acc += corr_fn_x(s.reshape(1, N), L, 8)
                half = 0 if m < n_meas / (2 * meas_every) else 1
                drift[half] += e
                dcnt[half] += 1
                cnt += 1
                m += 1
    e_mean = e_acc / cnt
    I2_mean = I2_acc / cnt
    rho_fluct = (e_mean - beta * I2_mean) / N
    nv = nv_acc / cnt
    C = C_acc / cnt
    dr = abs(drift[1] / max(dcnt[1], 1) - drift[0] / max(dcnt[0], 1)) / N
    return dict(
        L=L,
        TJ=TJ,
        seed=seed,
        rho_fluct=float(rho_fluct),
        two_T_over_pi=float(2 * TJ / np.pi),
        rho_over_T=float(rho_fluct / TJ),
        n_vortex=float(nv),
        C=[float(x) for x in C],
        drift=float(dr),
    )


def run_twist(L, TJ, seed, a, n_therm, n_meas, meas_every):
    """Route B: rho_s from the twist-response at per-bond twist `a`, ensemble
    equilibrated AT the twist (short local dynamics preserves winding sector).
    rho_s = (1/N) g(a)/a,  g(a) = -J<sum_x sin(dtheta_x - a)>.

    ⚠ Known limitation (kt-2d-csf3 §4): at L >= 16 near T_KT the imposed twist
    promotes vortex crossing and leaks the ensemble out of the w=0 sector, so
    g(a) crashes. Estimator limitation, not physics; the E-bridge (clu_path) is
    the clean "CLU is XY" test.
    """
    rng = np.random.default_rng(seed)
    beta = 1.0 / TJ
    nbrs = neighbors(L)
    black, white = checkerboard(L)
    masks = [black, white]
    N = L * L
    s = 0.1 * rng.standard_normal(N)  # aligned (cold) start: w=0 sector
    dmet = 2.0 if TJ > 0.9 else 1.2
    for _ in range(n_therm):
        s = sweep_twist(s, L, beta, nbrs, masks, dmet, a, rng, overrelax=True)
        s = sweep_twist(s, L, beta, nbrs, masks, dmet, a, rng, overrelax=False)
    g_acc = 0.0
    cnt = 0
    for it in range(n_meas):
        s = sweep_twist(s, L, beta, nbrs, masks, dmet, a, rng, overrelax=True)
        s = sweep_twist(s, L, beta, nbrs, masks, dmet, a, rng, overrelax=False)
        if it % meas_every == 0:
            g_acc += torque_x(s, a, nbrs)
            cnt += 1
    rho_twist = (g_acc / cnt) / a / N
    return float(rho_twist)


def xy_equilibrium_2d(L, TJ, nwalk, seed, nsweeps=2000):
    """Equilibrium reduced-XY configurations used to INITIALISE the CLU-Langevin
    bridge (clu_path.run_bridge). Cold (aligned) start => w=0 sector."""
    rng = np.random.default_rng(seed)
    idx = np.arange(L * L).reshape(L, L)
    xp = np.roll(idx, -1, 1).ravel()
    xm = np.roll(idx, 1, 1).ravel()
    yp = np.roll(idx, -1, 0).ravel()
    ym = np.roll(idx, 1, 0).ravel()
    yy, xx = np.mgrid[0:L, 0:L]
    black = ((xx + yy) % 2 == 0).ravel()
    masks = [black, ~black]
    beta = 1.0 / TJ
    d = 1.5
    out = np.zeros((nwalk, L * L))
    for w in range(nwalk):
        s = 0.1 * rng.standard_normal(L * L)  # aligned start: w=0 sector for rho_s
        for _ in range(nsweeps):
            for m in masks:  # overrelax
                fx = (
                    np.cos(s[xp[m]])
                    + np.cos(s[xm[m]])
                    + np.cos(s[yp[m]])
                    + np.cos(s[ym[m]])
                )
                fy = (
                    np.sin(s[xp[m]])
                    + np.sin(s[xm[m]])
                    + np.sin(s[yp[m]])
                    + np.sin(s[ym[m]])
                )
                s[m] = np.mod(2 * np.arctan2(fy, fx) - s[m], TWO_PI)
            for m in masks:  # metropolis
                pr = s[m] + rng.uniform(-d, d, m.sum())
                old = -(
                    np.cos(s[m] - s[xp[m]])
                    + np.cos(s[m] - s[xm[m]])
                    + np.cos(s[m] - s[yp[m]])
                    + np.cos(s[m] - s[ym[m]])
                )
                new = -(
                    np.cos(pr - s[xp[m]])
                    + np.cos(pr - s[xm[m]])
                    + np.cos(pr - s[yp[m]])
                    + np.cos(pr - s[ym[m]])
                )
                acc = rng.random(m.sum()) < np.exp(-beta * (new - old))
                s[m] = np.where(acc, pr, s[m])
        out[w] = s
    return out


def reduced_rho_s(th, L, TJ):
    """rho_s from reduced angles (fluctuation), x-bonds. th: (nwalk, N)."""
    idx = np.arange(L * L).reshape(L, L)
    xp = np.roll(idx, -1, 1).ravel()
    dx = th - th[:, xp]
    e = np.mean(np.sum(np.cos(dx), 1))
    I2 = np.mean(np.sum(np.sin(dx), 1) ** 2)
    return float((e - I2 / TJ) / (L * L))
