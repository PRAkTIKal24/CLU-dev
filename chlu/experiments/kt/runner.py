"""
KT tranche driver: mode dispatch, array-job sharding, JSON emission.

One process = one ``mode``, optionally one ``task_id`` (= one cell of that
mode's sweep grid), so a Slurm array maps 1:1 onto cells:

    chlu exp-kt --mode winding2d --task-id $SLURM_ARRAY_TASK_ID --out $OUT

Cells write ``<out>/<mode>_task<k>.json``; ``--mode postproc`` merges the shards
into the canonical ``reduced_xy.json`` / ``kt_clu.json`` / ``kt_winding*.json``
and writes ``summary.json`` + figures. Running a mode with no ``--task-id``
executes every cell sequentially and writes the canonical file directly.
"""

import json
import pathlib
import time

import numpy as np

KT_MODES = ("winding1d", "winding2d", "bridge", "reduced", "postproc")


def _now():
    return time.strftime("%H:%M:%S")


# ------------------------------------------------------------------ cells ----
def cells(mode, kt):
    """The sweep grid for ``mode`` as an ordered list of cell dicts."""
    if mode == "winding1d":
        return [dict(N=int(N)) for N in kt.winding1d_n_values]
    if mode == "winding2d":
        return [
            dict(TJ=float(TJ), L=int(L))
            for TJ in kt.winding2d_tj_values
            for L in kt.winding2d_l_values
        ]
    if mode == "bridge":
        return [dict(TJ=float(TJ)) for TJ in kt.bridge_tj_values]
    if mode == "reduced":
        return [dict(L=int(L)) for L in kt.reduced_l_values]
    return []


def _meta(kt, mode, extra=None):
    m = dict(
        mode=mode,
        kappa=kt.kappa,
        rstar=kt.rstar,
        lam=kt.lam,
        f=kt.f,
        J_xy=2 * kt.kappa * kt.rstar**2,
        k_r=8 * kt.lam * kt.f**2,
        dt=kt.dt,
        gamma=kt.gamma,
        nwalk=kt.n_walkers,
        kinetic_mode=kt.kinetic_mode,
        langevin_noise=kt.langevin_noise,
        use_governor=False,
        float64=True,
        # T_KT = 1.786 kappa r*^2 (0.0893 CLU units at kappa=0.05; NOT 0.1786)
        T_KT_pred_clu_units=1.786 * kt.kappa * kt.rstar**2,
        T_KT_pred_over_J=1.786 / 2.0,
    )
    if extra:
        m.update(extra)
    return m


# ------------------------------------------------------------------- run -----
def run_kt(
    config,
    mode="winding1d",
    out_dir="results/kt",
    seed=None,
    quick=False,
    task_id=None,
    make_figures=True,
    log=print,
):
    """Run one KT mode (optionally one array cell). Returns the result dict."""
    if mode not in KT_MODES:
        raise ValueError(f"mode must be one of {KT_MODES}, got {mode!r}")
    kt = config.experiment_kt
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if mode == "postproc":
        from .postproc import postprocess

        return postprocess(out, make_figures=make_figures, log=log)

    # float64 must be set before ANY jax array is created (clu_path asserts it).
    if mode in ("winding1d", "bridge"):
        import jax

        jax.config.update("jax_enable_x64", True)

    if quick:
        kt = _quicken(kt)

    grid = cells(mode, kt)
    if task_id is not None:
        if not (0 <= task_id < len(grid)):
            raise IndexError(
                f"task_id {task_id} out of range for mode {mode!r}: "
                f"{len(grid)} cells (0..{len(grid) - 1})"
            )
        grid = [grid[task_id]]

    log(f"[{_now()}] KT mode={mode} cells={len(grid)} task_id={task_id} out={out}")
    runner = {
        "winding1d": _run_winding1d,
        "winding2d": _run_winding2d,
        "bridge": _run_bridge,
        "reduced": _run_reduced,
    }[mode]
    res = runner(kt, grid, seed, task_id, out, log)

    fname = (
        f"{mode}_task{task_id}.json"
        if task_id is not None
        else _CANONICAL[mode]
    )
    (out / fname).write_text(json.dumps(res, indent=2, default=float))
    log(f"[{_now()}] KT mode={mode} DONE -> {out / fname}")
    return res


_CANONICAL = {
    "winding1d": "kt_winding_msd.json",
    "winding2d": "reduced_xy.json",
    "bridge": "kt_clu.json",
    "reduced": "reduced_xy.json",
}


def _quicken(kt):
    """Tiny-but-real smoke sizing: same code path, seconds not hours."""
    import dataclasses

    return dataclasses.replace(
        kt,
        n_walkers=8,
        winding1d_n_values=[4, 8],
        winding1d_chunks=4,
        winding1d_chunks_large=3,
        winding1d_chunk_steps=20,
        winding2d_l_values=[4, 6],
        winding2d_tj_values=[1.10],
        winding2d_nwalk=2,
        winding2d_nmax_below=200,
        winding2d_nmax_above=200,
        bridge_l=4,
        bridge_tj_values=[1.00],
        bridge_chunks=4,
        bridge_burn_chunks=1,
        bridge_chunk_steps=20,
        bridge_equil_sweeps=30,
        reduced_l_values=[4],
        reduced_tj_values=[0.80, 1.00],
        reduced_seeds=[100],
        reduced_nwalk_small=1,
        reduced_nwalk_large=1,
        reduced_therm_small=30,
        reduced_therm_large=30,
        reduced_meas_small=60,
        reduced_meas_large=60,
        reduced_twist_tj_values=[0.80],
        reduced_broken_tj_values=[0.80],
    )


# ------------------------------------------------------------ mode: 1-D ------
def _run_winding1d(kt, grid, seed, task_id, out, log):
    from .clu_path import loglog_slope, run_winding_msd

    res = {
        "meta": _meta(
            kt,
            "winding1d",
            dict(
                TJ=kt.winding1d_tj,
                method="winding MSD slope (bias-free)",
                target="soft exponent (b): tau ~ 1/N, clean slope -1 at low T",
            ),
        )
    }
    rows = []
    base_seed = kt.winding1d_seed if seed is None else int(seed)
    for cell in grid:
        N = cell["N"]
        nch = kt.winding1d_chunks if N <= 16 else kt.winding1d_chunks_large
        t0 = time.time()
        r = run_winding_msd(
            N=N,
            TJ=kt.winding1d_tj,
            n_chunks=nch,
            CH=kt.winding1d_chunk_steps,
            seed=base_seed,
            kappa=kt.kappa,
            rstar=kt.rstar,
            lam=kt.lam,
            f=kt.f,
            dt=kt.dt,
            gamma=kt.gamma,
            NW=kt.n_walkers,
            noise_mode=kt.langevin_noise,
            msd_fit_max=kt.winding1d_msd_fit_max,
        )
        rows.append(r)
        log(
            f"[{_now()}]  N={N:3d} T/J={kt.winding1d_tj} rate_msd={r['slip_rate_msd']:.3e} "
            f"rate/N={r['rate_over_N']:.3e} msd_final={r['msd_final']:.2f} "
            f"fit_pts={r['fit_points']}/{len(r['ts'])} ({time.time() - t0:.0f}s)"
        )
        if r["msd_final"] > 1.0 and kt.winding1d_msd_fit_max is None:
            log(
                f"[{_now()}]  ⚠ N={N}: msd_final={r['msd_final']:.2f} > 1 with no "
                "diffusive-window cut — this fit is SATURATION-DOMINATED and the "
                "resulting exponent is not trustworthy (set winding1d_msd_fit_max)."
            )
    res["N_scan"] = rows
    if task_id is None and len(rows) >= 2:
        sl = loglog_slope([r["N"] for r in rows], [r["slip_rate_msd"] for r in rows])
        res["lograte_vs_logN_slope"] = sl
        res["tau_slope"] = -sl
        log(f"[{_now()}]  MSD slope d ln(rate)/d ln N = {sl:.3f} => tau ~ N^{-sl:.3f}")
    return res


# ------------------------------------------------------------ mode: 2-D ------
def _run_winding2d(kt, grid, seed, task_id, out, log):
    from .reduced_xy import run_winding

    res = {
        "meta": _meta(
            kt,
            "winding2d",
            dict(
                method="reduced-XY Model-A single-spin Metropolis, init w_x=1",
                target=(
                    "soft exponent (a): sign change of tau ~ L^(pi rho_s/T - 2) "
                    "above T_KT, unresolved at L<=16 (vortex diffusion ~L^2 masks it)"
                ),
                T_KT_over_J_measured=0.898,
            ),
        )
    }
    rows = []
    base_seed = kt.winding2d_seed if seed is None else int(seed)
    for cell in grid:
        TJ, L = cell["TJ"], cell["L"]
        nmax = (
            kt.winding2d_nmax_below
            if TJ < kt.winding2d_tkt_over_j
            else kt.winding2d_nmax_above
        )
        t0 = time.time()
        r = run_winding(
            L=L, TJ=TJ, seed=base_seed, nwalk=kt.winding2d_nwalk, n_max=nmax
        )
        r["secs"] = time.time() - t0
        rows.append(r)
        log(
            f"[{_now()}]  wind T/J={TJ:.2f} L={L:3d} tau_med={r['tau_med']:.0f} "
            f"tau_mean={r['tau_mean']:.0f} censored={r['censored']}/{r['nwalk']} "
            f"({r['secs']:.0f}s)"
        )
    res["D_winding"] = rows
    return res


# --------------------------------------------------------- mode: bridge ------
def _run_bridge(kt, grid, seed, task_id, out, log):
    from .clu_path import run_bridge

    res = {
        "meta": _meta(
            kt, "bridge", dict(L=kt.bridge_l, method="CLU-Langevin vs reduced-XY rho_s")
        )
    }
    res["E_bridge"] = run_bridge(
        L=kt.bridge_l,
        tjs=[c["TJ"] for c in grid],
        kappa=kt.kappa,
        rstar=kt.rstar,
        lam=kt.lam,
        f=kt.f,
        dt=kt.dt,
        gamma=kt.gamma,
        NW=kt.n_walkers,
        n_chunks=kt.bridge_chunks,
        burn_chunks=kt.bridge_burn_chunks,
        CH=kt.bridge_chunk_steps,
        seed=kt.bridge_seed if seed is None else int(seed),
        equil_seed=kt.bridge_equil_seed,
        equil_sweeps=kt.bridge_equil_sweeps,
        noise_mode=kt.langevin_noise,
        log=lambda m: log(f"[{_now()}]{m}"),
    )
    return res


# -------------------------------------------------------- mode: reduced ------
def _run_reduced(kt, grid, seed, task_id, out, log):
    """Reduced-XY phase diagram: B (rho_s/C(r)/n_v) sharded by L; C (two-route)
    and F (broken-symmetry null) run in the unsharded/first shard only."""
    from .reduced_xy import run_cell, run_twist

    res = {"meta": _meta(kt, "reduced", dict(note="reduced XY, T in units of J"))}
    seeds = kt.reduced_seeds if seed is None else [int(seed)]

    def sizing(L):
        small = L <= 16
        return (
            kt.reduced_nwalk_small if small else kt.reduced_nwalk_large,
            kt.reduced_therm_small if small else kt.reduced_therm_large,
            kt.reduced_meas_small if small else kt.reduced_meas_large,
        )

    # ---- B ----
    B = []
    for cell in grid:
        L = cell["L"]
        nwalk, n_therm, n_meas = sizing(L)
        for TJ in kt.reduced_tj_values:
            t0 = time.time()
            rows = [
                run_cell(L, TJ, s, nwalk, n_therm, n_meas, kt.reduced_meas_every)
                for s in seeds
            ]
            rf = np.array([r["rho_fluct"] for r in rows])
            nv = np.array([r["n_vortex"] for r in rows])
            Cm = np.mean([r["C"] for r in rows], axis=0)
            B.append(
                dict(
                    L=L,
                    TJ=TJ,
                    rho_fluct=float(rf.mean()),
                    rho_fluct_sem=float(rf.std() / np.sqrt(len(rf))),
                    two_T_over_pi=float(2 * TJ / np.pi),
                    n_vortex=float(nv.mean()),
                    C=[float(v) for v in Cm],
                    drift=float(np.mean([r["drift"] for r in rows])),
                )
            )
            log(
                f"[{_now()}]  L={L:3d} T/J={TJ:.2f} rho_fl={rf.mean():.4f}"
                f"+-{rf.std() / np.sqrt(len(rf)):.4f} 2T/pi={2 * TJ / np.pi:.4f} "
                f"nv={nv.mean():.2e} ({time.time() - t0:.0f}s)"
            )
    res["B_rho_s"] = B

    if task_id not in (None, 0):
        return res

    # ---- C: two-route agreement ----
    C = []
    for L in kt.reduced_twist_l_values:
        nwalk, nt, nm = sizing(L)
        for TJ in kt.reduced_twist_tj_values:
            rf = np.mean(
                [
                    run_cell(L, TJ, s, 2, nt, nm, kt.reduced_meas_every)["rho_fluct"]
                    for s in seeds[:2]
                ]
            )
            rt = np.mean(
                [
                    run_twist(L, TJ, 500 + i, kt.reduced_twist_a, nt, nm, kt.reduced_meas_every)
                    for i in range(len(seeds[:2]))
                ]
            )
            rel = abs(rt - rf) / max(abs(rf), 1e-6)
            C.append(
                dict(L=L, TJ=TJ, rho_fluct=float(rf), rho_twist=float(rt), rel_diff=float(rel))
            )
            log(
                f"[{_now()}]  two-route L={L} T/J={TJ:.2f} fluct={rf:.4f} "
                f"twist={rt:.4f} rel={rel:.3f}"
            )
    res["C_two_route"] = C

    # ---- F: broken-symmetry null (XY + h2 cos 2theta) ----
    F = []
    L = kt.reduced_broken_l
    nwalk, nt, nm = sizing(L)
    for TJ in kt.reduced_broken_tj_values:
        rows = [
            run_cell(
                L, TJ, 300 + i, nwalk, nt, nm, kt.reduced_meas_every, h2=kt.reduced_broken_h2
            )
            for i in range(len(seeds))
        ]
        rf = np.array([r["rho_fluct"] for r in rows])
        F.append(
            dict(
                L=L,
                TJ=TJ,
                h2=kt.reduced_broken_h2,
                rho_fluct=float(rf.mean()),
                rho_fluct_sem=float(rf.std() / np.sqrt(len(rf))),
                two_T_over_pi=float(2 * TJ / np.pi),
            )
        )
        log(
            f"[{_now()}]  broken-sym h2={kt.reduced_broken_h2} T/J={TJ:.2f} "
            f"rho_fl={rf.mean():.4f} 2T/pi={2 * TJ / np.pi:.4f}"
        )
    res["F_broken_sym"] = F
    return res
