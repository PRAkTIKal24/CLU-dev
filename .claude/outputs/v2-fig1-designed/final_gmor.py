"""v2-full-runs: paper-grade figures from the saved npz/json tables.

Writes PNGs into .claude/outputs/v2-full-runs/.
Command: PYTHONPATH=<repo> uv run --no-sync python .claude/scratch/v2-full-runs/make_figures.py
"""

import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(BASE, "..", "..", "outputs", "v2-full-runs"))
GAMMA = 0.05
DT = 0.05
H_STAR = (1 - math.sqrt(1 - GAMMA)) * math.sqrt(2 / (2 - GAMMA))
FLOOR = 2 * math.log(2) / (-math.log(1 - GAMMA))

plt.rcParams.update({"font.size": 9, "figure.dpi": 150})


def agg(d, key, mask=None):
    """mean/std of d[key] grouped by delta (or gamma)."""
    xs = d["delta"] if "delta" in d else d["gamma"]
    if mask is None:
        mask = np.ones(len(xs), bool)
    vals = {}
    for x in sorted(set(np.round(xs[mask], 9))):
        m = mask & (np.round(xs, 9) == x)
        vals[x] = (d[key][m].mean(), d[key][m].std())
    x = np.array(list(vals))
    mu = np.array([v[0] for v in vals.values()])
    sd = np.array([v[1] for v in vals.values()])
    return x, mu, sd


def fig1_gmor():
    d = np.load(os.path.join(OUT, "gmor_sweep.npz"))
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))

    ax = axes[0]
    x, me, se = agg(d, "n_half_env")
    _, mr, sr = agg(d, "n_half_raw")
    _, mp, _ = agg(d, "pred_n_half")
    ax.errorbar(x, me, se, fmt="o", ms=4, capsize=2, label="measured envelope $n_{1/2}$", zorder=5)
    ax.errorbar(x, mr, sr, fmt="s", ms=3.5, capsize=2, color="tab:red", alpha=0.8,
                label="measured raw-$|d|$ first crossing (readout)", zorder=4)
    xx = np.logspace(np.log10(x.min()), np.log10(x.max()), 200)
    # prediction from the mean GMOR slope mu2 = delta * <1/(M r^2)>
    slope = float(np.mean(d["mu2_meas"] / d["delta"]))
    from chlu.experiments.goldstone_harness import exact_mode_eigenvalues
    pred = []
    for dl in xx:
        lp, lm = exact_mode_eigenvalues(dl * slope, DT, GAMMA)
        mod = max(abs(lp), abs(lm))
        pred.append(math.log(2) / (-math.log(mod)) if mod < 1 else np.inf)
    ax.plot(xx, pred, "k-", lw=1, label="exact-map prediction", zorder=3)
    ax.axhline(FLOOR, color="gray", ls=":", lw=1, label=f"saturation floor $2\\ln2/(-\\ln(1-\\gamma))$ = {FLOOR:.1f}")
    dstar = (H_STAR / DT) ** 2 / slope
    ax.axvline(dstar, color="tab:purple", ls="--", lw=1, label=f"crossover $h=h^*(\\gamma)$ ($\\delta^*\\approx${dstar:.2f})")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"explicit breaking $\delta$  [tilt $\delta\cos\theta$]")
    ax.set_ylabel(r"retention half-life $n_{1/2}$ (steps)")
    ax.set_title("GMOR retention law on trained CLUs (5 seeds)\n"
                 r"overdamped slope $-0.985$ (predicted $-1$); metric bifurcation past $h^*$")
    ax.legend(fontsize=6.5, loc="lower left")

    ax = axes[1]
    x, mg, sg = agg(d, "gmor_ratio")
    ax.errorbar(x, mg, sg, fmt="o", ms=4, capsize=2, color="tab:green")
    ax.axhline(1.0, color="k", lw=0.8)
    ax.set_xscale("log")
    ax.set_ylim(1 - 1e-9, 1 + 1e-9)
    ax.set_xlabel(r"$\delta$")
    ax.set_ylabel(r"$\mu^2_{\rm meas} \,/\, [\delta n^2/(M_{\rm ch} r_*^2)]$")
    ax.set_title("GMOR spectral-mass law: exact on the designed vacuum\n(max deviation $\\sim 10^{-12}$, all seeds,\n4.5 orders of magnitude)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_gmor.png"), bbox_inches="tight")
    print("fig1 done")


def fig2_mo():
    d = np.load(os.path.join(OUT, "gmor_sweep.npz"))
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    hh = d["h"] / d["h_star"]
    unc = ~d["mo_censored"]
    for s in sorted(set(d["seed"].astype(int))):
        m = unc & (d["seed"] == s)
        ax.plot(hh[m], d["mo_ratio"][m], "o-", ms=3, lw=0.6, alpha=0.65, label=f"seed {s}")
    ax.axhline(1.0, color="k", lw=0.8)
    ax.axhspan(0.98, 1.05, color="tab:green", alpha=0.15, label="Mo's reported median ratio 1.013")
    ax.axvline(1.0, color="tab:purple", ls="--", lw=1, label=r"$h = h^*(\gamma)$ (EP)")
    ax.set_xscale("log")
    ax.set_xlabel(r"$h/h^*(\gamma)$   ($h = \varepsilon\mu$ from the measured tilted Hessian)")
    ax.set_ylabel("Mo lifetime: measured / predicted")
    ax.set_title("Mo's exact lifetime protocol run on trained CLUs\n"
                 "overdamped: 1.012–1.03 (his median 1.013) · EP: 2.20±0.16 · deep underdamped: 0.31")
    ax.legend(fontsize=6.5)
    ax.annotate("censored ($\\delta\\leq 3\\times10^{-4}$): 10/70 runs,\nsame pattern as Mo's own $\\epsilon=10^{-4}$ row",
                xy=(0.03, 0.06), xycoords="axes fraction", fontsize=6.5, color="gray")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_mo.png"), bbox_inches="tight")
    print("fig2 done")


def fig3_gamma():
    d = np.load(os.path.join(OUT, "gamma_sweep.npz"))
    ps = json.load(open(os.path.join(OUT, "gamma_per_seed.json")))
    fim = json.load(open(os.path.join(OUT, "flat_immunity_raw.json")))
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))

    ax = axes[0]
    from chlu.experiments.goldstone_harness import exact_mode_eigenvalues
    cmap = plt.get_cmap("viridis")
    seeds = sorted(set(d["seed"].astype(int)))
    for i, s in enumerate(seeds):
        m = d["seed"] == s
        c = cmap(i / max(len(seeds) - 1, 1))
        ax.plot(d["gamma"][m], d["n_half_env"][m], "o", ms=3.5, color=c, label=f"s{s} measured")
        mu2 = ps[f"designed150_s{s}"]["mu2_rad"]
        gg = np.logspace(math.log10(0.004), math.log10(0.7), 200)
        pred = []
        for g in gg:
            lp, lm = exact_mode_eigenvalues(mu2, DT, g)
            mod = max(abs(lp), abs(lm))
            pred.append(math.log(2) / (-math.log(mod)) if mod < 1 else np.inf)
        ax.plot(gg, pred, "-", lw=0.8, color=c, alpha=0.6)
        ax.axvline(ps[f"designed150_s{s}"]["gamma_star_exact"], color=c, ls=":", lw=0.8, alpha=0.7)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"friction $\gamma$")
    ax.set_ylabel(r"radial-mode $n_{1/2}$ (steps)")
    ax.set_title("C1: retention minimum at critical damping\n(points = measured; lines = exact-map prediction; dotted = exact $\\gamma^*$)")
    ax.legend(fontsize=6)

    ax = axes[1]
    x, mm, sm = agg(d, "dtheta_meas")
    _, pp, _ = agg(d, "dtheta_pred")
    ratio_x, ratio_m, ratio_s = agg({"gamma": d["gamma"], "r": d["dtheta_meas"] / d["dtheta_pred"]}, "r")
    ax.errorbar(ratio_x, ratio_m, ratio_s, fmt="o", ms=4, capsize=2, color="tab:blue",
                label=r"latch transport $\Delta\theta_{\rm meas}/\Delta\theta_{\rm pred}$")
    ax.axhline(1.0, color="k", lw=0.8)
    gs = [fim[k][g]["theta_drift_total"] for k in fim for g in fim[k]]
    ax.set_xscale("log")
    ax.set_ylim(0.97, 1.03)
    ax.set_xlabel(r"friction $\gamma$")
    ax.set_ylabel("ratio")
    ax.set_title("Goldstone write/latch across the whole $\\gamma$ sweep\n"
                 r"transport law $\Delta\theta=\varepsilon p_0/(\sqrt{M}r\gamma)$ to $\leq$1%;"
                 "\nwritten angle frozen to $\\leq 10^{-15}$ rad (flat-mode immunity, all $\\gamma$)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_gamma.png"), bbox_inches="tight")
    print("fig3 done")


def fig4_emergent():
    es = json.load(open(os.path.join(OUT, "emergent_summary.json")))
    rp = np.load(os.path.join(OUT, "emergent_ring_profiles.npz"))
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.4))

    ax = axes[0]
    for tag in es:
        th = rp[f"{tag}_ring_thetas"]; v = rp[f"{tag}_ring_V"]
        style = "-" if tag.startswith("emergent") else ":"
        lw = 1.2 if tag.startswith("emergent") else 0.9
        ax.plot(th, v - v.min() + 1e-18, style, lw=lw, label=tag.replace("150", ""))
    ax.set_yscale("log")
    ax.set_xlabel(r"ring angle $\theta$ (at $r_*$)")
    ax.set_ylabel(r"$V(\theta) - V_{\min}$")
    ax.set_title("Self-induced washboard on the vacuum ring\nemergent (mlp): ripple 1.2–5.7e-2 · designed: $\\leq 10^{-16}$ (exact)")
    ax.legend(fontsize=5.5, ncol=2)

    ax = axes[1]
    tags = list(es)
    for i, tag in enumerate(tags):
        mu = np.abs(np.array(es[tag]["mu_sq"])) + 1e-17
        col = "tab:red" if tag.startswith("emergent") else "tab:blue"
        ax.scatter([i] * len(mu), mu, s=18, color=col)
    ax.set_yscale("log")
    ax.set_xticks(range(len(tags)))
    ax.set_xticklabels([t.replace("150", "") for t in tags], rotation=45, ha="right", fontsize=6)
    ax.axhline(H_STAR**2 / DT**2, color="tab:purple", ls="--", lw=0.8,
               label=r"register/working-memory crossover $\mu^2$ at $\gamma=0.05$")
    ax.set_ylabel(r"$|\mu_k^2|$ spectrum at the settled vacuum")
    ax.set_title("Designed vs emergent spectral gap:\nflat mode $10^{-16}$ vs softest emergent $5\\times10^{-2}$ (13–14 orders)")
    ax.legend(fontsize=6)

    ax = axes[2]
    names, vals, cols = [], [], []
    for tag in tags:
        names.append(tag.replace("150", ""))
        vals.append(es[tag]["eeq_v"]["max"] + 1e-18)
        cols.append("tab:red" if tag.startswith("emergent") else "tab:blue")
    ax.bar(range(len(names)), vals, color=cols)
    ax.set_yscale("log")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=6)
    ax.set_ylabel(r"$E^{V}_{\rm eq}$ (max over 192 pairs)")
    ax.set_title("Equivariance error of $\\nabla V$ (Mo Eq. 4 refined)\narchitectural $\\sim 3\\times10^{-16}$ vs emergent $\\sim 10^{-1}$")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_emergent.png"), bbox_inches="tight")
    print("fig4 done")


def fig5_isotropy():
    iso = json.load(open(os.path.join(OUT, "isotropy_summary.json")))
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))

    ax = axes[0]
    for tag, r in iso.items():
        broken = not r["tie"]
        col = "tab:red" if broken else "tab:blue"
        ax.plot([0, 1], [r["split_log_init"], r["split_log_final"]], "o-", color=col,
                label=tag.replace("150", "") if True else None, alpha=0.8)
        ax.annotate(tag.replace("150", ""), xy=(1.02, r["split_log_final"]), fontsize=6, color=col)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["init", "trained (150 ep)"])
    ax.set_xlim(-0.15, 1.45)
    ax.set_ylabel(r"channel split $|\log m_0 - \log m_1|$ (raw entries)")
    ax.set_title("F5 §4.1 falsifiable: NO isotropization on symmetric data\n"
                 "red = broken-isotropy (split persists); blue = tied control (split exactly preserved)")

    ax = axes[1]
    sp = [iso[t]["split_M_final"] for t in iso if not iso[t]["tie"]]
    q0 = [iso[t]["Q_drift_gamma0"] for t in iso if not iso[t]["tie"]]
    ql = [iso[t]["Q_law_err_gamma"] for t in iso if not iso[t]["tie"]]
    sp_t = [max(iso[t]["split_M_final"], 1e-18) for t in iso if iso[t]["tie"]]
    q0_t = [iso[t]["Q_drift_gamma0"] for t in iso if iso[t]["tie"]]
    ax.loglog(sp, q0, "o", color="tab:red", label=r"$\gamma=0$ charge drift (2e4 steps), broken")
    ax.loglog(sp, ql, "s", color="tab:orange", label=r"$\gamma=0.05$ $(1-\gamma)^n$-law error, broken")
    ax.loglog(sp_t, q0_t, "^", color="tab:blue", label="tied controls (machine floor)")
    xx = np.logspace(math.log10(min(sp)) - 0.3, math.log10(max(sp)) + 0.3, 10)
    ax.plot(xx, 0.45 * xx, "k:", lw=0.8, label=r"$\propto$ split (guide)")
    ax.set_xlabel(r"trained inertial-mass split $|M_0 - M_1|$")
    ax.set_ylabel("Noether-charge law violation")
    ax.set_title("The price of kinetic breaking is the charge law\n(Hessian $\\mu^2_{\\rm ang}$ stays $\\sim10^{-15}$: invisible to statics)")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5_isotropy.png"), bbox_inches="tight")
    print("fig5 done")


def fig6_ep():
    d = np.load(os.path.join(OUT, "ep_sweep.npz"))
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    above = d["h_minus_hstar"] > 0
    x = d["h_minus_hstar"][above]; y = d["freq_jac"][above]
    ax.loglog(x, y, "o", ms=4, label="measured (map Jacobian eigenvalue), 5 seeds")
    tr = np.isfinite(d["freq_traj"]) & above
    ax.loglog(d["h_minus_hstar"][tr], d["freq_traj"][tr], "x", ms=7, color="tab:red",
              label="trajectory corroboration")
    sl, ic = np.polyfit(np.log(x), np.log(y), 1)
    xx = np.logspace(math.log10(x.min()), math.log10(x.max()), 50)
    ax.plot(xx, np.exp(ic) * xx**sl, "k-", lw=0.8,
            label=f"fit: slope {sl:.4f} (C3 predicts 1/2), prefactor {math.exp(ic):.3f}")
    ax.set_xlabel(r"$h - h^*(\gamma)$")
    ax.set_ylabel(r"oscillation onset $\varphi$ (rad/step)")
    ax.set_title("C3 exceptional-point signature on trained CLUs\n"
                 r"$\varphi = 0$ exactly below $h^*$ (all 15 below-EP rows); onset $\propto\sqrt{h-h^*}$ above")
    ax.legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig6_ep.png"), bbox_inches="tight")
    print("fig6 done, fitted slope", sl, "prefactor", math.exp(ic))


def fig7_collapse():
    import jax
    import jax.numpy as jnp
    import sys
    sys.path.insert(0, BASE)
    from probe_common import to_x64
    from chlu.utils.checkpoints import load_model

    cells = [
        ("150 ep (battery-2)", os.path.join(BASE, "runs", "diag_max_150", "models", "exp_d_chlu.pkl")),
        ("300 ep", os.path.join(BASE, "runs", "diag2_max_300", "models", "exp_d_chlu.pkl")),
        ("600 ep", os.path.join(BASE, "runs", "diag2_max_600", "models", "exp_d_chlu.pkl")),
        ("1000 ep (defaults)", os.path.join(BASE, "runs", "diag_max_1000", "models", "exp_d_chlu.pkl")),
        ("1000 ep, wake-only", os.path.join(BASE, "runs", "diag2_wakeonly_1000", "models", "exp_d_chlu.pkl")),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
    ax = axes[0]
    rs = np.linspace(0, 1.6, 161)
    depths, labels = [], []
    for name, path in cells:
        model = to_x64(load_model(path))
        V = jax.jit(lambda q: model.potential_net(q))
        prof = np.array([float(V(jnp.array([r, 0.0, 0.0, 0.0]))) for r in rs])
        ax.plot(rs, prof - prof[0], lw=1.2, label=name)
        ring = prof[0] - prof[rs > 0.4].min()
        depths.append(ring); labels.append(name)
    ax.axvline(1.0, color="gray", ls=":", lw=0.8)
    ax.annotate("data ring R=1", xy=(1.01, ax.get_ylim()[0] * 0.9), fontsize=6, color="gray")
    ax.set_xlabel("channel radius $r$")
    ax.set_ylabel("$V(r) - V(0)$")
    ax.set_title("Sleep-phase erosion of the designed vacuum\n(radial profile of trained $V$, seed 42)")
    ax.legend(fontsize=6.5)

    ax = axes[1]
    ep = [150, 300, 600, 1000]
    dd = depths[:4]
    ax.plot(ep, dd, "o-", color="tab:red", label="with sleep (defaults, freq 5)")
    ax.plot([1000], [depths[4]], "s", ms=8, color="tab:blue", label="wake-only, 1000 ep")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("training epochs")
    ax.set_ylabel("ring depth $V(0) - V(r_{\\rm ring})$")
    ax.set_title("Ring depth vs training length\ninversion between 300 and 600 epochs; wake-only intact")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig7_collapse.png"), bbox_inches="tight")
    print("fig7 done; depths:", dict(zip(labels, [round(float(x), 5) for x in depths])))


if __name__ == "__main__":
    fig1_gmor()
    print("FIG1 ONLY DONE")
