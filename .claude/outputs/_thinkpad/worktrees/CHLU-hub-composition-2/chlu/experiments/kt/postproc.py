"""
Post-process a KT tranche: Weber-Minnhagen T_KT fit, eta, winding slopes, figures.

Consumes the same filenames the laptop run wrote, so it reads
``.claude/outputs/kt-2d-csf3/`` unchanged:

    reduced_xy.json      B_rho_s / C_two_route / D_winding / F_broken_sym
    kt_clu.json          E_bridge / A_winding_1d
    kt_winding_msd.json  N_scan  (bias-free 1-D slip rate)
    kt_winding1d.json    N_scan  (slip counting cross-check)

and additionally merges array-job shards ``<mode>_task*.json`` written by
``runner.run_kt(task_id=...)`` into those same structures first.

Every section is optional: a winding-only tranche post-processes fine.
"""

import json
import pathlib

import numpy as np

# Figures are bespoke KT diagnostics (universal-jump line, WM extrapolation),
# not part of the reusable utils/plotting.py trajectory/phase-space vocabulary.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

T_KT_LIT_OVER_J = 0.8929  # Hasenbusch
# T_KT = 1.786 kappa r*^2 = 0.0893 CLU units at kappa=0.05 (NOT 0.1786).
J_XY_DEFAULT = 0.10


# --------------------------------------------------------------- shards ------
SHARD_TARGET = {
    "winding2d": ("reduced_xy.json", "D_winding"),
    "reduced": ("reduced_xy.json", None),
    "bridge": ("kt_clu.json", "E_bridge"),
    "winding1d": ("kt_winding_msd.json", "N_scan"),
    "winding1d_count": ("kt_winding1d.json", "N_scan"),
}


def merge_shards(out: pathlib.Path, log=print):
    """Fold ``<mode>_task<k>.json`` array-job shards into the canonical files."""
    for mode, (fname, _section) in SHARD_TARGET.items():
        shards = sorted(out.glob(f"{mode}_task*.json"))
        if not shards:
            continue
        target = out / fname
        base = json.loads(target.read_text()) if target.exists() else {}
        for sh in shards:
            d = json.loads(sh.read_text())
            base.setdefault("meta", {}).update(d.get("meta", {}))
            for key, rows in d.items():
                if key == "meta" or not isinstance(rows, list):
                    continue
                merged = base.setdefault(key, [])
                for r in rows:
                    if r not in merged:
                        merged.append(r)
        target.write_text(json.dumps(base, indent=2, default=float))
        log(f"  merged {len(shards)} {mode} shard(s) -> {fname}")


# ------------------------------------------------------------- postprocess ---
def postprocess(out_dir, make_figures=True, log=print):
    out = pathlib.Path(out_dir)
    merge_shards(out, log=log)

    def load(name):
        p = out / name
        return json.loads(p.read_text()) if p.exists() else {}

    R = load("reduced_xy.json")
    summary = {}
    B = R.get("B_rho_s", [])
    Ls, Tx, T_inf, x, y, A = [], {}, float("nan"), None, None, None

    # ---------- B: rho_s crossings + Weber-Minnhagen T_KT ----------
    if B:
        Ls = sorted(set(r["L"] for r in B))

        def crossing(L):
            rows = sorted([r for r in B if r["L"] == L], key=lambda r: r["TJ"])
            T = np.array([r["TJ"] for r in rows])
            rho = np.array([r["rho_fluct"] for r in rows])
            line = 2 * T / np.pi
            diff = rho - line  # positive below T_KT, negative above
            for i in range(len(T) - 1):
                if diff[i] > 0 >= diff[i + 1]:
                    f = diff[i] / (diff[i] - diff[i + 1])  # linear interp
                    return float(T[i] + f * (T[i + 1] - T[i]))
            return float("nan")

        Tx = {L: crossing(L) for L in Ls}
        # WM: T_x(L) = T_inf + a / ln^2 L
        x = np.array([1.0 / np.log(L) ** 2 for L in Ls])
        y = np.array([Tx[L] for L in Ls])
        if len(Ls) >= 2 and np.all(np.isfinite(y)):
            A = np.polyfit(x, y, 1)  # y = A[0]*x + A[1]
            T_inf = float(A[1])
            summary["WM_slope_a"] = float(A[0])
        summary["crossings_TJ"] = {int(L): Tx[L] for L in Ls}
        summary["WM_T_inf_over_J"] = T_inf
        summary["T_KT_lit_over_J"] = T_KT_LIT_OVER_J
        summary["T_KT_CLU_units"] = T_inf * J_XY_DEFAULT
        summary["universal_jump_2_over_pi"] = 2 / np.pi

        # ---------- eta at crossing from C(r) ----------
        def eta_at(L, Ttarget):
            rows = sorted([r for r in B if r["L"] == L], key=lambda r: r["TJ"])
            if not rows or not np.isfinite(Ttarget):
                return float("nan"), float("nan")
            row = min(rows, key=lambda r: abs(r["TJ"] - Ttarget))  # nearest T
            C = np.array(row["C"])
            rr = np.arange(len(C))
            use = (rr >= 1) & (rr <= L // 2) & (C > 0.02)
            if use.sum() < 2:
                return float("nan"), row["TJ"]
            sl = np.polyfit(np.log(rr[use]), np.log(C[use]), 1)
            return float(-sl[0]), row["TJ"]

        for L in (16, 32):
            if L in Ls:
                eta, Te = eta_at(L, T_inf)
                summary[f"eta_L{L}_at_TKT"] = dict(eta=eta, at_TJ=Te)

    # ---------- C: two-route agreement ----------
    if R.get("C_two_route"):
        summary["two_route"] = R["C_two_route"]
        off = [
            r["rel_diff"]
            for r in R["C_two_route"]
            if not np.isfinite(T_inf) or abs(r["TJ"] - T_inf) > 0.08
        ]
        if off:
            summary["two_route_max_reldiff_offcrit"] = float(max(off))

    # ---------- D: 2-D winding survival slopes (soft exponent (a)) ----------
    if R.get("D_winding"):
        Dsum = {}
        for TJ in sorted(set(r["TJ"] for r in R["D_winding"])):
            rows = sorted(
                [r for r in R["D_winding"] if r["TJ"] == TJ], key=lambda r: r["L"]
            )
            Lw = np.array([r["L"] for r in rows])
            tau = np.array([r["tau_med"] for r in rows])
            cens = [r["censored"] for r in rows]
            # censored points bias the slope DOWN, so a positive slope is conservative
            sl = (
                float(np.polyfit(np.log(Lw), np.log(np.maximum(tau, 1)), 1)[0])
                if len(Lw) >= 2
                else float("nan")
            )
            Dsum[TJ] = dict(
                L=Lw.tolist(), tau_med=tau.tolist(), censored=cens, loglog_slope=sl
            )
        summary["winding_2d"] = Dsum

    # ---------- F: broken symmetry ----------
    if R.get("F_broken_sym"):
        summary["broken_sym"] = [
            dict(
                TJ=r["TJ"],
                rho=r["rho_fluct"],
                two_T_over_pi=r["two_T_over_pi"],
                above_line=r["rho_fluct"] > r["two_T_over_pi"],
            )
            for r in R["F_broken_sym"]
        ]

    # ---------- E: CLU bridge + legacy A_winding_1d ----------
    K = load("kt_clu.json")
    if K.get("E_bridge"):
        summary["E_bridge"] = K["E_bridge"]
    if K.get("A_winding_1d"):
        Aw = K["A_winding_1d"]
        sl = _slope([r["N"] for r in Aw], [r["slip_rate"] for r in Aw])
        summary["winding_1d"] = dict(rows=Aw, lograte_vs_logN_slope=sl, tau_slope=-sl)

    # ---------- 1-D winding: MSD (bias-free) + counting ----------
    for fn, key, ratekey in [
        ("kt_winding_msd.json", "winding_1d_msd", "slip_rate_msd"),
        ("kt_winding1d.json", "winding_1d_count", "slip_rate"),
    ]:
        W = load(fn)
        rows = W.get("N_scan", [])
        if not rows:
            continue
        slope = W.get("lograte_vs_logN_slope")
        if slope is None:
            slope = _slope([r["N"] for r in rows], [r[ratekey] for r in rows])
        summary[key] = dict(
            slope=slope,
            tau_slope=-slope if slope == slope else float("nan"),
            TJ=sorted(set(r["TJ"] for r in rows)),
            rows=[
                {k: v for k, v in r.items() if k not in ("ts", "msd")} for r in rows
            ],
        )

    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    if make_figures:
        _figures(out, summary, B, Ls, T_inf, x, y, A, log=log)

    log(f"  summary.json written -> {out / 'summary.json'}")
    return summary


def _slope(xs, ys):
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    good = (ys > 0) & (xs > 0)
    if good.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(xs[good]), np.log(ys[good]), 1)[0])


# ---------------------------------------------------------------- figures ----
def _figures(out, summary, B, Ls, T_inf, x, y, A, log=print):
    # --- winding 1D (degrades) vs 2D (improves) contrast ---
    try:
        msd = json.loads((out / "kt_winding_msd.json").read_text())["N_scan"]
        have2d = bool(summary.get("winding_2d"))
        fig, axes = plt.subplots(1, 2 if have2d else 1, figsize=(13 if have2d else 6.5, 5))
        axes = np.atleast_1d(axes)
        ax = axes[0]
        Nn = np.array([r["N"] for r in msd])
        rt = np.array([r["slip_rate_msd"] for r in msd])
        ax.loglog(Nn, rt, "o-", label="CLU ring slip rate (MSD)")
        ax.loglog(
            Nn, rt[0] * (Nn / Nn[0]), "k--",
            label=r"$\propto N$ (slope 1, $\tau\propto1/N$)",
        )
        ax.set_xlabel("N (ring size)")
        ax.set_ylabel("winding slip rate / step")
        sl = summary.get("winding_1d_msd", {}).get("slope", float("nan"))
        ax.set_title(f"1-D: memory DEGRADES with size (slope {sl:.2f})")
        ax.legend()
        ax.grid(alpha=0.3, which="both")
        if have2d:
            ax = axes[1]
            for TJ, d in summary["winding_2d"].items():
                below = float(TJ) < T_inf if np.isfinite(T_inf) else float(TJ) < 0.898
                ax.loglog(
                    d["L"], d["tau_med"], "o-",
                    label=f"T/J={TJ} ({'below' if below else 'above'} $T_{{KT}}$)",
                )
            ax.set_xlabel("L (torus size)")
            ax.set_ylabel(r"winding survival $\tau$ (sweeps)")
            ax.set_title("2-D: memory IMPROVES with L below $T_{KT}$")
            ax.legend()
            ax.grid(alpha=0.3, which="both")
        fig.tight_layout()
        fig.savefig(out / "winding_contrast.png", dpi=130)
        plt.close(fig)
    except Exception as e:
        log(f"  winding fig skipped: {e}")

    # --- broken-symmetry null ---
    try:
        bs = summary["broken_sym"]
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(
            [r["TJ"] for r in bs], [r["rho"] for r in bs], "s-", color="crimson",
            label=r"random-$W$ ($h_2/J=1$)",
        )
        xy = sorted([r for r in B if r["L"] == 16], key=lambda r: r["TJ"])
        if xy:
            ax.plot(
                [r["TJ"] for r in xy], [r["rho_fluct"] for r in xy], "o-",
                color="steelblue", label="channel-spring (XY), L=16",
            )
        Tg = np.linspace(0.5, 1.2, 50)
        ax.plot(Tg, 2 * Tg / np.pi, "k--", label=r"$2T/\pi$")
        ax.set_xlabel("T/J")
        ax.set_ylabel(r"$\rho_s$")
        ax.set_title("Broken-symmetry null: no KT $2/\\pi$ jump (Ising-like collapse)")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "broken_sym.png", dpi=130)
        plt.close(fig)
    except Exception as e:
        log(f"  broken-sym fig skipped: {e}")

    # --- the 2/pi jump + rho_s/T ---
    if not B:
        return
    try:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        ax = axes[0]
        for L in Ls:
            rows = sorted([r for r in B if r["L"] == L], key=lambda r: r["TJ"])
            ax.errorbar(
                [r["TJ"] for r in rows], [r["rho_fluct"] for r in rows],
                yerr=[r.get("rho_fluct_sem", 0.0) for r in rows],
                marker="o", label=f"L={L}",
            )
        Tg = np.linspace(min(r["TJ"] for r in B), max(r["TJ"] for r in B), 100)
        ax.plot(Tg, 2 * Tg / np.pi, "k--", label=r"$2T/\pi$ (univ. jump)")
        if np.isfinite(T_inf):
            ax.axvline(T_inf, color="gray", ls=":", label=f"$T_{{KT}}/J$={T_inf:.3f}")
        ax.set_xlabel("T/J")
        ax.set_ylabel(r"$\rho_s$ (spin stiffness)")
        ax.set_title("KT universal jump: reduced-XY")
        ax.legend()
        ax.grid(alpha=0.3)

        ax = axes[1]
        for L in Ls:
            rows = sorted([r for r in B if r["L"] == L], key=lambda r: r["TJ"])
            T = np.array([r["TJ"] for r in rows])
            rho = np.array([r["rho_fluct"] for r in rows])
            ax.plot(T, rho / T, marker="o", label=f"L={L}")
        ax.axhline(2 / np.pi, color="k", ls="--", label=r"$2/\pi$")
        ax.set_xlabel("T/J")
        ax.set_ylabel(r"$\rho_s/T$")
        ax.set_title(r"$\rho_s/T$ crosses $2/\pi$")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "kt_jump.png", dpi=130)
        plt.close(fig)
    except Exception as e:
        log(f"  kt_jump fig skipped: {e}")

    # --- Weber-Minnhagen crossing extrapolation ---
    if A is None:
        return
    try:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(x, y, "o", ms=10)
        xx = np.linspace(0, max(x) * 1.1, 50)
        ax.plot(xx, A[0] * xx + A[1], "-")
        ax.axhline(T_KT_LIT_OVER_J, color="r", ls="--", label="Hasenbusch 0.8929")
        ax.plot(0, T_inf, "s", ms=12, label=f"WM extrap {T_inf:.4f}")
        for i, L in enumerate(Ls):
            ax.annotate(f"L={L}", (x[i], y[i]))
        ax.set_xlabel(r"$1/\ln^2 L$")
        ax.set_ylabel(r"crossing $T_\times/J$")
        ax.set_title("Weber-Minnhagen extrapolation")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "wm_extrap.png", dpi=130)
        plt.close(fig)
    except Exception as e:
        log(f"  wm_extrap fig skipped: {e}")
