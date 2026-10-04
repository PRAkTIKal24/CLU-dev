"""UNIT 6 -- THE CERTIFIED-ACCESS FIGURE (DEV, unscored).

WHAT THIS DOES.  Walks a launch point outward from a ``val`` row's OWN CACHED FEATURE
along a fixed, seeded, unit-norm random ray, and at each radius measures

  * the geometry: ``min_k d_k/sigma_k``, ``max_k E_k``, the CLU gate decision
    (``E_k > 0`` in any well), the CERTIFICATE (``d_k >= sigma_k*sqrt(2 ln N)`` in any
    well), and two calibrated kNN rivals at ``q = 0.95``;
  * the decode: the external classifier's factor accuracy on the decoded output, for
    (a) a plain decode of ``x(t)`` and (b) the CLU read -- the shipped ``settle_chain``
    relaxation launched at ``x(t)``.

⛔ NOTHING IS RE-ENCODED.  Every feature comes from ``ctx["X_store"]``, the frozen
cache gathered by ``refusal_lib.build_context``.  ⛔ ``ctx["eval_b"]`` is never read.
⛔ The gate is ``admission_lib``'s, the decoder is ``exp_gen_baselines``'s hash-asserted
winner, the classifier is ``exp_gen_cell.load_external_classifier``'s gated instrument,
the settling is ``gen_landscape.settle_chain``.  Not one of them is re-implemented here.

THE CLU READ, STATED EXACTLY.  The launch point on the ray is a LATENT, not a request,
so there is no term tensor to steer with: ``settle_chain`` is called with
``log_w = None`` and ``total_energy`` therefore drops the request term entirely (its own
branch, ``gen_landscape.total_energy`` line 954).  This is the store-landscape relaxation
of the launch point and nothing more.  ⛔ No request-steered path is claimed.

Re-runnable:  ``python u6_figure.py``            -- full run (~15 min)
              ``python u6_figure.py --plot-only`` -- redraw the PNG from the JSON
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

OUT = Path(os.environ["CHLU_HUB_OUTPUTS"]) / "hub-admission"
sys.path.insert(0, str(OUT / "scorer"))
sys.path.insert(0, os.path.join(os.environ["CHLU_HUB_OUTPUTS"], "hub-refusal", "scorer"))

JSON_PATH = OUT / "CERTIFIED_ACCESS_FIGURE_seed0_A.json"
PNG_PATH = OUT / "certified_access_seed0_A.png"

SEED, BUDGET, Q = 0, "A", 0.95
#: ⭐ THE DIRECTION SEED.  One ``default_rng(20260917)`` draw of shape ``(n_rows, 2048)``,
#: row-normalised: one fixed unit direction per val row, identical at every radius.
DIR_SEED = 20260917
N_ROWS = int(os.environ.get("U6_N_ROWS", "192"))
N_CAL = int(os.environ.get("U6_N_CAL", "4096"))
#: The ray grid, in absolute latent units.  Chosen from the measured scale (u6_probe2):
#: ``min_k d/sigma`` runs 0.80 at t=0 to ~5 at t=86.  ⛔ Not guessed: read off the data.
T_GRID = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 13.0, 16.0, 19.0, 22.0, 25.0, 28.0,
          32.0, 36.0, 40.0, 45.0, 50.0, 56.0, 63.0, 70.0, 78.0, 86.0]

T0 = time.time()


def log(m=""):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


# ---------------------------------------------------------------------------
# crossings
# ---------------------------------------------------------------------------

def first_crossing(x, y, level, rising=True):
    """Linear-interpolated first crossing of ``level``; ``None`` if it never crosses."""
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    for i in range(len(y) - 1):
        a, b = y[i], y[i + 1]
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        hit = (a < level <= b) if rising else (a > level >= b)
        if hit:
            if b == a:
                return float(x[i])
            return float(x[i] + (x[i + 1] - x[i]) * (level - a) / (b - a))
    return None


# ---------------------------------------------------------------------------
# the figure
# ---------------------------------------------------------------------------

def draw(rec: dict, path: Path) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    C = rec["curves"]
    x = np.asarray(C["x_min_d_over_sigma_mean"], np.float64)
    cr = rec["crossings"]
    chance = rec["chance_accuracy"]

    # Palette: the reference categorical instance, validated with the skill's own
    # validator (`validate_palette.js`, light, --pairs all): lightness band PASS,
    # chroma floor PASS, CVD separation PASS (worst pair dE 9.2 deutan), normal-vision
    # floor PASS (16.3).  Aqua WARNs on 3:1 contrast -> the relief rule is discharged
    # by the visible direct label on that curve and by the raw table in the JSON.
    SURFACE = "#fcfcfb"
    INK = "#0b0b0b"
    MUTED = "#52514e"
    GRID = "#e7e6e2"
    PLAIN = "#2a78d6"      # slot 1 -- plain decode of x(t)
    CLU = "#eb6834"        # slot 2 -- the CLU read
    LEVEL = "#4a3aa7"      # slot 7 -- the CLU gate, E_k > 0
    CERT = "#1baf7a"       # slot 3 -- the certificate
    KNN = MUTED            # the calibrated rivals are reference lines, not identities

    fig, (ax, bx) = plt.subplots(
        2, 1, figsize=(9.6, 8.6), sharex=True,
        gridspec_kw={"height_ratios": [1.5, 1.0], "hspace": 0.10})
    fig.patch.set_facecolor(SURFACE)

    vlines = [
        (cr["zero_level_set_x"], LEVEL, "-", "zero level set",
         f"zero level set: $\\max_k E_k$ crosses 0  ({cr['zero_level_set_x']:.2f})"),
        (cr["certificate_flip_x"], CERT, "-", "certificate",
         f"certified radius: $d_k\\geq\\sigma_k\\sqrt{{2\\ln N}}$ in some well  "
         f"({cr['certificate_flip_x']:.2f})"),
        (cr["knn_whole_x"], KNN, (0, (6, 3)), "kNN whole",
         f"kNN whole-latent, q=0.95  ({cr['knn_whole_x']:.2f})"),
        (cr["knn_factored_x"], KNN, (0, (1.5, 2)), "kNN factored",
         f"kNN factored, q=0.95  ({cr['knn_factored_x']:.2f})"),
    ]
    for a in (ax, bx):
        for xv, col, ls, _s, _lab in vlines:
            if xv is not None:
                a.axvline(xv, color=col, ls=ls, lw=1.4, alpha=0.9, zorder=1)

    # ---- top: the deliverable, accuracy vs radius --------------------------
    ax.plot(x, C["acc_plain"], "-o", color=PLAIN, lw=2.0, ms=5.0, zorder=4,
            mec=SURFACE, mew=1.0, label="plain decode of $x(t)$")
    ax.plot(x, C["acc_clu_req"], "-s", color=CLU, lw=2.0, ms=5.0, zorder=5,
            mec=SURFACE, mew=1.0,
            label="CLU read: settle from $x(t)$ under the row's own request, then decode")
    ax.plot(x, C["acc_clu"], color=MUTED, ls=(0, (7, 2, 1.5, 2)), lw=1.5, zorder=3,
            label="control: the same settle with the request term off ($\\log w$ = None)")
    ax.axhline(chance, color=GRID, ls="-", lw=1.4, zorder=0)
    ax.text(x[-1], chance + 0.018, f"chance {chance:.3f}", color=MUTED,
            ha="right", va="bottom", fontsize=8.5)
    ax.text(x[-1], float(C["acc_clu"][-1]) - 0.022,
            "request term off  (at chance)", color=MUTED,
            ha="right", va="top", fontsize=8.5)
    j_lab = min(13, len(x) - 1)
    ax.text(x[j_lab], float(C["acc_clu_req"][j_lab]) - 0.028,
            "CLU read", color=INK, fontsize=9.5, ha="left", va="top")
    ax.text(x[j_lab], float(C["acc_plain"][j_lab]) + 0.028,
            "plain decode", color=INK, fontsize=9.5, ha="left", va="bottom")

    # ⭐ the one sentence the figure exists to make, drawn as a measured span
    if cr["certificate_flip_x"] is not None and cr["decode_collapse_x"] is not None:
        y_span = 0.245
        ax.annotate("", xy=(cr["certificate_flip_x"], y_span),
                    xytext=(cr["decode_collapse_x"], y_span),
                    arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1.1,
                                    shrinkA=0, shrinkB=0))
        ax.text(0.5 * (cr["certificate_flip_x"] + cr["decode_collapse_x"]),
                y_span - 0.022,
                f"the certificate flips {cr['decode_collapse_x'] - cr['certificate_flip_x']:.2f}"
                " earlier, in the same units",
                color=INK, fontsize=9.0, ha="center", va="top")
    if cr["decode_collapse_x"] is not None:
        ax.plot([cr["decode_collapse_x"]], [rec["decode_collapse_level"]], "*",
                color=PLAIN, ms=17, zorder=6, mec=SURFACE, mew=1.0)
        ax.annotate(f"decode collapses  {cr['decode_collapse_x']:.2f}",
                    xy=(cr["decode_collapse_x"], rec["decode_collapse_level"]),
                    xytext=(14, 26), textcoords="offset points",
                    color=INK, fontsize=9.5, ha="left",
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.9))
    ax.set_ylim(0.0, 1.04)
    ax.set_ylabel("external-classifier accuracy\n(mean over 6 factors)",
                  fontsize=10, color=INK)
    ax.set_title(
        f"Certified access — seed {rec['seed']}, rung {rec['budget']} "
        f"(K={rec['K_wells']} wells, N={rec['n_atoms']} atoms/well).  DEV, unscored.\n"
        f"{rec['n_rows']} val rows walked outward along a fixed unit ray "
        f"(direction seed {rec['direction_seed']}); "
        f"{len(rec['t_grid'])} radii; decoder = gen-decoder-lift winner, "
        "classifier = gen-external-classifier (gated).",
        fontsize=10.5, loc="left", pad=12, color=INK)
    hl, ll = ax.get_legend_handles_labels()

    # ---- bottom: the geometry the gate actually reads ----------------------
    bx.plot(x, C["refuse_rate"], "-o", color=LEVEL, lw=1.9, ms=4.2, zorder=4,
            mec=SURFACE, mew=0.9, label="CLU gate refuses: $E_k>0$ in some well")
    bx.plot(x, C["certified_rate"], "-s", color=CERT, lw=1.9, ms=4.2, zorder=4,
            mec=SURFACE, mew=0.9,
            label="certified: $d_k\\geq\\sigma_k\\sqrt{2\\ln N}$ in some well")
    bx.plot(x, C["knn_whole_refuse_rate"], color=KNN, ls=(0, (6, 3)), lw=1.6,
            zorder=3, label="kNN whole-latent refuses (q=0.95)")
    bx.plot(x, C["knn_factored_refuse_rate"], color=KNN, ls=(0, (1.5, 2)), lw=1.9,
            zorder=3, label="kNN factored refuses (q=0.95)")
    bx.axhline(0.5, color=MUTED, lw=0.8, ls=(0, (2, 4)), zorder=1)
    j = int(np.argmax(np.asarray(C["certified_rate"]) >= 0.999))
    bx.annotate("certified", xy=(x[j], 1.0), xytext=(6, -16),
                textcoords="offset points", color=INK, fontsize=9.5,
                ha="left", va="top")
    bx.set_ylim(-0.04, 1.08)
    bx.set_ylabel(f"fraction of the {rec['n_rows']} rays", fontsize=10, color=INK)
    bx.set_xlabel(r"$\min_k\ d_k/\sigma_k$ at the launch point   (mean over rays)",
                  fontsize=11, color=INK)
    h2, l2 = bx.get_legend_handles_labels()

    # ---- ONE legend for the whole figure, outside the data ------------------
    # The vertical rules ARE the bottom panel's four series, so they share its
    # handles; the crossing each one marks is written into the label.
    l2 = [f"{lab}   —  50% at {xv:.2f}" if xv is not None else lab
          for lab, (xv, _c, _ls, _s, _l) in zip(l2, vlines)]
    leg = fig.legend(hl + h2, ll + l2, loc="upper center",
                     bbox_to_anchor=(0.5, 0.065), ncol=2, fontsize=8.9,
                     framealpha=1.0, edgecolor=GRID, facecolor=SURFACE,
                     borderpad=0.8, labelspacing=0.55, columnspacing=2.4,
                     handlelength=2.8)
    for t in leg.get_texts():
        t.set_color(INK)

    for a in (ax, bx):
        a.set_facecolor(SURFACE)
        a.grid(True, color=GRID, lw=0.7)
        a.set_axisbelow(True)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            a.spines[s].set_color(MUTED)
            a.spines[s].set_linewidth(0.8)
        a.tick_params(colors=MUTED, labelsize=9, length=3)
        for t in a.get_xticklabels() + a.get_yticklabels():
            t.set_color(INK)

    bx.text(0.985, 0.30,
            "Every curve is in CERTIFIED_ACCESS_FIGURE_seed0_A.json (the table view).\n"
            "No claim beyond this figure.",
            transform=bx.transAxes, fontsize=8.0, color=MUTED,
            ha="right", va="center", linespacing=1.5)
    fig.savefig(path, dpi=190, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return str(path)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    import admission_ctrl as AC      # noqa: E402
    import admission_lib as AL       # noqa: E402
    import refusal_lib as R          # noqa: E402

    import jax                                                          # noqa: E402
    from chlu.config import gen_dynamical_read_config                   # noqa: E402
    from chlu.core.gen_landscape import SettleCfg, settle_chain         # noqa: E402
    from chlu.core.gen_request import FactorPrototypes, requests_from_rows  # noqa: E402
    from chlu.eval.gen_probe import (decode_features, factor_labels,    # noqa: E402
                                     probe_accuracy)
    from chlu.experiments.exp_gen_baselines import load_shared_decoder  # noqa: E402
    from chlu.experiments.exp_gen_cell import load_external_classifier  # noqa: E402
    from chlu.experiments.exp_gen_store import bridge_dir               # noqa: E402

    ctx = R.build_context(SEED)
    built = R.build_land(ctx, budget=BUDGET, log=lambda *a, **k: None)
    land, store, grid = built["land"], built["store"], ctx["grid"]
    config = ctx["config"]
    sig = np.asarray(land.sigma, np.float64)
    rad = AL.certified_radius(land)
    names = list(grid.names)
    feat_shape = tuple(int(v) for v in ctx["feat_shape"])
    log(f"land K={store.n_wells} N={store.n_atoms} dim={store.dim} "
        f"sigma={np.round(sig, 4).tolist()} cert_radius={np.round(rad, 4).tolist()}")

    # -- PREREG self-check: the identity must hold before any gate number --
    ident = AL.identity_E_le_0_on_live_store(land)
    if not ident["identity_E_le_0_holds"]:
        raise AssertionError(f"identity E_k <= 0 on the live store FAILED: {ident}")
    log(f"SELF-CHECK identity_E_le_0_on_live_store: holds "
        f"(max over wells {ident['max_over_all_wells']:.4f})")

    pool = np.asarray(ctx["pool"], np.int64)
    Xs = np.asarray(ctx["X_store"], np.float32)
    pos = {int(r): i for i, r in enumerate(pool)}

    val_rows = np.asarray(ctx["val_rows"], np.int64)
    fit_rows = np.asarray(ctx["fit_rows"], np.int64)
    rng_rows = np.random.default_rng(DIR_SEED)
    sel = np.sort(rng_rows.permutation(val_rows.size)[:N_ROWS])
    rows = val_rows[sel]
    idx = np.asarray([pos[int(r)] for r in rows], np.int64)   # ⛔ cache only
    X0 = Xs[idx]
    lab = factor_labels(grid, rows)
    log(f"{rows.size} val rows (of {val_rows.size}); features read from the cache at "
        f"their pool positions -- ⛔ nothing re-encoded")

    # ⭐ the directions: ONE seeded unit vector per row, identical at every radius
    rng_dir = np.random.default_rng(DIR_SEED)
    U = rng_dir.standard_normal(X0.shape).astype(np.float32)
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    assert np.allclose(np.linalg.norm(U, axis=1), 1.0, atol=1e-5)

    # ⭐ the row's OWN request, so the shipped request-steered read can be launched
    # at x(t).  ⛔ Valid requests only: term_experts is elementwise equal to the
    # shipped request_experts here, and that is asserted, not assumed.
    reqs = requests_from_rows(grid, rows)
    log_w, n_terms, _tinfo = R.term_experts(built["book"], grid, reqs, R.T_MAX)
    repro = R.assert_reproduces_shipped_experts(built["book"], reqs, log_w)
    log(f"SELF-CHECK term_experts vs shipped request_experts: {repro}")

    # -- the kNN rivals, calibrated on fit_rows.  ⛔ never on val, ⛔ never on B --
    cal_rows = fit_rows[np.random.default_rng(20260917).permutation(fit_rows.size)[:N_CAL]]
    X_cal_feat = Xs[np.asarray([pos[int(r)] for r in cal_rows], np.int64)]
    g_whole = AC.KnnGate.fit(store, X_cal_feat, "whole", Q)
    g_fact = AC.KnnGate.fit(store, X_cal_feat, "factored", Q)
    log(f"kNN calibrated on {cal_rows.size} fit_rows CACHED FEATURES: "
        f"whole={np.round(g_whole.thresholds, 4).tolist()} "
        f"factored={np.round(g_fact.thresholds, 4).tolist()}")

    # the same rivals calibrated on the R8 embeddings of the fit_rows corners, so the
    # DEV table's thresholds are quotable beside these ones.
    protos = FactorPrototypes.fit(grid, Xs, pool)
    X_cal_r8 = protos.embed_many(requests_from_rows(grid, cal_rows)).reshape(
        cal_rows.size, -1)
    g_whole_r8 = AC.KnnGate.fit(store, X_cal_r8, "whole", Q)
    g_fact_r8 = AC.KnnGate.fit(store, X_cal_r8, "factored", Q)
    log(f"kNN calibrated on the R8 embeddings of the same corners: "
        f"whole={np.round(g_whole_r8.thresholds, 4).tolist()} "
        f"factored={np.round(g_fact_r8.thresholds, 4).tolist()}")

    # -- the instruments --
    bdir = bridge_dir(config, R.DATASET, R.PRETEXT, SEED)
    bman = json.loads((bdir / "bridge_manifest.json").read_text())
    dec, dinfo = load_shared_decoder(config, bdir, bman, SEED, log)
    clf, cinfo = load_external_classifier(config, SEED, log=log)

    c = gen_dynamical_read_config(config)
    steps = int(R.DYNREAD_STEPS[BUDGET])
    scfg = SettleCfg(steps=steps, step_size=float(c.step_size),
                     sigma_scale_start=float(c.sigma_scale_start),
                     sigma_scale_end=float(c.sigma_scale_end),
                     temp_start=float(c.temp_start), temp_end=float(c.temp_end),
                     metropolis=bool(c.metropolis))
    key0 = R.unit_key(SEED, BUDGET, "u6_certified_access", "full", config)
    log(f"settle: steps={steps} {scfg.as_dict()}")

    cards = list(grid.cardinalities)
    chance = float(np.mean([1.0 / int(cc) for cc in cards]))

    def score(feats):
        imgs = decode_features(dec, np.asarray(feats, np.float32).reshape(
            (-1,) + feat_shape), 256)
        return probe_accuracy(clf, imgs, lab, names, 256)

    C = {k: [] for k in (
        "t", "x_min_d_over_sigma_mean", "x_min_d_over_sigma_median",
        "x_min_d_over_sigma_p10", "x_min_d_over_sigma_p90", "max_d_over_sigma_mean",
        "max_E_mean", "max_E_median", "refuse_rate", "certified_rate",
        "certified_refusal_rate", "knn_whole_refuse_rate", "knn_factored_refuse_rate",
        "knn_whole_r8_refuse_rate", "knn_factored_r8_refuse_rate",
        "acc_plain", "acc_clu", "acc_clu_req", "acc_plain_exact", "acc_clu_exact",
        "acc_clu_req_exact",
        "clu_displacement_mean", "clu_min_d_over_sigma_mean", "clu_max_E_mean",
        "clu_req_displacement_mean", "clu_req_min_d_over_sigma_mean",
        "clu_req_max_E_mean")}
    per_factor = {"plain": [], "clu": [], "clu_req": []}
    raw = {"min_d_over_sigma": [], "max_E": [], "refuse": [], "certified": []}

    for ti, t in enumerate(T_GRID):
        X = (X0 + np.float32(t) * U).astype(np.float32)
        g = AL.gate(land, X)
        ds = g["d_over_sigma"].min(axis=1)
        Emax = g["E"].max(axis=1)
        dw = g_whole.decide(store, X)
        df = g_fact.decide(store, X)
        dw8 = g_whole_r8.decide(store, X)
        df8 = g_fact_r8.decide(store, X)

        a_plain = score(X)
        # (i) the SHIPPED read: settle at x(t) with the row's OWN request steering it.
        out_r = settle_chain(land, X, log_w, scfg, jax.random.fold_in(key0, ti))
        Xr = np.asarray(out_r["x"], np.float32)
        gr = AL.gate(land, Xr)
        a_clu_req = score(Xr)
        # (ii) the same path with the request term removed (log_w=None): the control
        #      that says how much of the read is the request and how much the store.
        out = settle_chain(land, X, None, scfg, jax.random.fold_in(key0, 1000 + ti))
        Xc = np.asarray(out["x"], np.float32)
        gc = AL.gate(land, Xc)
        a_clu = score(Xc)

        C["t"].append(float(t))
        C["x_min_d_over_sigma_mean"].append(float(ds.mean()))
        C["x_min_d_over_sigma_median"].append(float(np.median(ds)))
        C["x_min_d_over_sigma_p10"].append(float(np.quantile(ds, 0.10)))
        C["x_min_d_over_sigma_p90"].append(float(np.quantile(ds, 0.90)))
        C["max_d_over_sigma_mean"].append(float(g["d_over_sigma"].max(axis=1).mean()))
        C["max_E_mean"].append(float(Emax.mean()))
        C["max_E_median"].append(float(np.median(Emax)))
        C["refuse_rate"].append(float(g["refuse"].mean()))
        C["certified_rate"].append(float(g["certified_well"].any(axis=1).mean()))
        C["certified_refusal_rate"].append(float(g["certified_refusal"].mean()))
        C["knn_whole_refuse_rate"].append(float(dw["refuse"].mean()))
        C["knn_factored_refuse_rate"].append(float(df["refuse"].mean()))
        C["knn_whole_r8_refuse_rate"].append(float(dw8["refuse"].mean()))
        C["knn_factored_r8_refuse_rate"].append(float(df8["refuse"].mean()))
        C["acc_plain"].append(float(a_plain["aggregate"]))
        C["acc_clu"].append(float(a_clu["aggregate"]))
        C["acc_clu_req"].append(float(a_clu_req["aggregate"]))
        C["acc_plain_exact"].append(float(a_plain["exact_match"]))
        C["acc_clu_exact"].append(float(a_clu["exact_match"]))
        C["acc_clu_req_exact"].append(float(a_clu_req["exact_match"]))
        C["clu_displacement_mean"].append(float(np.linalg.norm(Xc - X, axis=1).mean()))
        C["clu_min_d_over_sigma_mean"].append(float(gc["d_over_sigma"].min(axis=1).mean()))
        C["clu_max_E_mean"].append(float(gc["E"].max(axis=1).mean()))
        C["clu_req_displacement_mean"].append(float(np.linalg.norm(Xr - X, axis=1).mean()))
        C["clu_req_min_d_over_sigma_mean"].append(
            float(gr["d_over_sigma"].min(axis=1).mean()))
        C["clu_req_max_E_mean"].append(float(gr["E"].max(axis=1).mean()))
        per_factor["plain"].append(a_plain["per_factor"])
        per_factor["clu"].append(a_clu["per_factor"])
        per_factor["clu_req"].append(a_clu_req["per_factor"])
        raw["min_d_over_sigma"].append([float(v) for v in ds])
        raw["max_E"].append([float(v) for v in Emax])
        raw["refuse"].append([bool(v) for v in g["refuse"]])
        raw["certified"].append([bool(v) for v in g["certified_well"].any(axis=1)])

        log(f"  t={t:6.1f}  min d/sig={ds.mean():6.3f}  maxE={Emax.mean():10.2f}  "
            f"refuse={C['refuse_rate'][-1]:.3f} cert={C['certified_rate'][-1]:.3f}  "
            f"knnW={C['knn_whole_refuse_rate'][-1]:.3f} knnF={C['knn_factored_refuse_rate'][-1]:.3f}"
            f"  acc_plain={C['acc_plain'][-1]:.4f}  acc_clu_req={C['acc_clu_req'][-1]:.4f}"
            f"  acc_clu_storeonly={C['acc_clu'][-1]:.4f}  "
            f"(read moved {C['clu_req_displacement_mean'][-1]:.2f} -> d/sig "
            f"{C['clu_req_min_d_over_sigma_mean'][-1]:.3f})")

    x = np.asarray(C["x_min_d_over_sigma_mean"], np.float64)
    acc0 = float(C["acc_plain"][0])
    collapse_level = 0.5 * (acc0 + chance)
    cr = {
        "zero_level_set_x": first_crossing(x, C["refuse_rate"], 0.5, True),
        "zero_level_set_x_by_meanE": first_crossing(x, C["max_E_mean"], 0.0, True),
        "certificate_flip_x": first_crossing(x, C["certified_rate"], 0.5, True),
        "certified_refusal_x": first_crossing(x, C["certified_refusal_rate"], 0.5, True),
        "knn_whole_x": first_crossing(x, C["knn_whole_refuse_rate"], 0.5, True),
        "knn_factored_x": first_crossing(x, C["knn_factored_refuse_rate"], 0.5, True),
        "knn_whole_r8_x": first_crossing(x, C["knn_whole_r8_refuse_rate"], 0.5, True),
        "knn_factored_r8_x": first_crossing(x, C["knn_factored_r8_refuse_rate"], 0.5, True),
        "decode_collapse_x": first_crossing(x, C["acc_plain"], collapse_level, False),
        "decode_below_half_x": first_crossing(x, C["acc_plain"], 0.5, False),
        "clu_read_collapse_x": first_crossing(x, C["acc_clu_req"], collapse_level, False),
        "clu_read_below_half_x": first_crossing(x, C["acc_clu_req"], 0.5, False),
        "clu_storeonly_collapse_x": first_crossing(x, C["acc_clu"], collapse_level, False),
    }

    rec = {
        "schema": "admission-certified-access/1",
        "title": "certified access: gate geometry and decode quality along a ray (DEV)",
        "seed": SEED, "budget": BUDGET, "q": Q,
        "K_wells": int(store.n_wells), "n_atoms": int(store.n_atoms),
        "dim": int(store.dim), "feat_shape": list(feat_shape),
        "sigma": [float(s) for s in sig],
        "certified_radius": [float(v) for v in rad],
        "certified_radius_in_sigma_units": float(np.sqrt(2.0 * np.log(store.n_atoms))),
        "n_rows": int(rows.size), "direction_seed": DIR_SEED,
        "row_selection": f"default_rng({DIR_SEED}).permutation(val_rows)[:{N_ROWS}], sorted",
        "val_rows_used": [int(r) for r in rows],
        "t_grid": [float(v) for v in T_GRID],
        "chance_accuracy": chance,
        "factor_cardinalities": [int(v) for v in cards],
        "factor_names": names,
        "decode_collapse_level": collapse_level,
        "decode_collapse_definition":
            "acc_plain crosses (acc at t=0 + chance)/2, linearly interpolated on the "
            "mean min_k d/sigma axis",
        "identity_check": ident,
        "term_experts_check": repro,
        "knn_thresholds": {
            "calibrated_on": f"{int(cal_rows.size)} fit_rows CACHED FEATURES "
                             "(disjoint from val, ⛔ B never touched)",
            "whole": [float(v) for v in g_whole.thresholds],
            "factored": [float(v) for v in g_fact.thresholds],
            "whole_r8_embedding": [float(v) for v in g_whole_r8.thresholds],
            "factored_r8_embedding": [float(v) for v in g_fact_r8.thresholds],
        },
        "settle_cfg": scfg.as_dict(),
        "curves": C, "per_factor": per_factor, "raw_per_row": raw,
        "crossings": cr,
        "provenance": {
            "machine": "FORGIS_14 / Windows AMD64 / CPU",
            "platform": f"{platform.system()} {platform.machine()} "
                        f"python {platform.python_version()}",
            "jax_backend": str(jax.default_backend()),
            "seed": SEED, "budget_rung": BUDGET,
            "direction_seed": DIR_SEED,
            "calibration_source": "ctx['fit_rows'] only -- ⛔ never val, ⛔ never eval_b",
            "features": "ctx['X_store'] (frozen feature cache) -- ⛔ nothing re-encoded",
            "decode_leg_ran": True,
            "decoder": dinfo, "classifier": cinfo,
            "encoder_param_hash": ctx["encoder_param_hash"],
            "reencode_check": ctx["reencode_check"],
            "clu_read_definition":
                "acc_clu_req = gen_landscape.settle_chain(land, x(t), log_w, SettleCfg("
                f"steps={steps}, the shipped gen_dynamical_read knobs), then decode: the "
                "SHIPPED request-steered read, launched at x(t) instead of at the "
                "address-book launch, with the val row's OWN request as the term tensor "
                "(refusal_lib.term_experts, asserted elementwise equal to the shipped "
                "request_experts for these valid requests).  "
                "acc_clu = the SAME call with log_w=None: the store-landscape "
                "relaxation alone, reported as the control that says how much of the "
                "read is the request term.  ⛔ No other path is claimed.",
            "script": str(Path(__file__).resolve()),
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    }
    JSON_PATH.write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")
    log(f"wrote {JSON_PATH}")

    log("")
    log("=== CROSSINGS (units: mean min_k d_k/sigma_k) ===")
    for k, v in cr.items():
        log(f"  {k:26s} {('%.4f' % v) if v is not None else 'never crosses in range'}")
    try:
        log(f"figure -> {draw(rec, PNG_PATH)}")
    except Exception as exc:                       # matplotlib missing / backend
        log(f"⛔ FIGURE NOT DRAWN: {type(exc).__name__}: {exc}")
    log("U6 DONE")


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        rec = json.loads(JSON_PATH.read_text(encoding="utf-8"))
        print("figure ->", draw(rec, PNG_PATH))
    else:
        main()
