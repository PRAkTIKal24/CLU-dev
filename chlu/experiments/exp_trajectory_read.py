"""C2W1 `trainability-spike` — end-to-end gradients and the learned-psi trajectory read.

Four parts, run in the PREREG'd compute order
(``.claude/outputs/trainability-spike/PREREG.md`` §5):

``--part a``
    **A1-A2 gradcheck.** Implicit (DEQ) gradients through the shipped dissipative
    settle vs (a) truncated unroll at the theorist's registered depth and (b)
    re-settled central finite differences, on a controlled toy with a known
    answer (``GaussianWellsPotential``). Also the ``(gamma, dt)``-independence
    check, ``||p*||``, the ridge bias, and the Q3.5 conditioning triple.

``--part e2e``
    **Acceptance half 1.** Gradients flow ``query -> phi -> settle -> psi ->
    loss`` on the real harness store, at a measured wall-clock per training step,
    scored against the budget declared in PREREG §3. Also measures the PREREG §4c
    prediction that the settled-point read sends **no** gradient to ``phi``.

``--part stage0``
    ⭐ **The blocking axis-liveness gate** (Hub amendment, monitor #10). Does the
    strided trajectory carry information the settled point does not, at the
    **healthy** S0 geometry (``sep/sigma_q = 6.83``)? Probes (linear + kNN) on
    matched-capacity feature sets ``q0_only`` / ``endpoints`` / ``full``, over a
    sweep of query ambiguity x ``traj_stride`` x ``gamma_read``. Part B does not
    run unless this passes its registered gate.

``--part b``
    **The point-vs-trajectory ablation** — matched psi family, matched parameter
    count, matched bytes, matched phi, matched seeds — plus the **trajectory
    launder** (``eval/dividend.py``) and the blank-store control, which become
    mandatory the moment a learned psi can see the address block.

Run by module invocation (this wave's CLI file is owned by ``memory-gym-v0``;
**no CLI hook is added**)::

    PYTHONPATH=. python -u -m chlu.experiments.exp_trajectory_read --part a --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import replace
from typing import Any, Dict, Optional

import numpy as np

OUT_DIR = os.environ.get(
    "TRAJ_READ_OUT",
    os.path.join(os.path.expanduser("~"), "Desktop", "CHLU", ".claude", "outputs",
                 "trainability-spike"),
)


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return float(o)
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    return o


def _save(name: str, payload: dict) -> str:
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    with open(path, "w") as f:
        json.dump(_jsonable(payload), f, indent=2)
    return path


# ==========================================================================
# PART A — implicit/DEQ gradients through the settle
# ==========================================================================
def part_a(seed: int = 0, quick: bool = False) -> dict:
    """A1-A2: gradcheck implicit vs truncated unroll vs finite differences.

    Runs in **float64** (enabled here, never at module import — the repo's
    x64-at-import hazard, handover §7.2).
    """
    import jax

    jax.config.update("jax_enable_x64", True)
    import equinox as eqx
    import jax.numpy as jnp

    from chlu.core.implicit_grad import (
        SettleSpec,
        implicit_grad,
        implicit_settle,
        ridge_alarm,
        settle_forward,
        settle_telemetry,
        theory_ridge,
        toy_model,
        unroll_grad,
    )

    # -- the controlled toy: 4 Gaussian wells on a ring (the theory's family) --
    n_w, dim = 4, 2
    ang = 2.0 * np.pi * np.arange(n_w) / n_w
    centers = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    amp0 = np.array([1.0, 0.9, 1.1, 0.95])
    s_well, alpha = 0.35, 0.05

    def build(amp):
        return toy_model(centers, jnp.asarray(amp), s=s_well, alpha=alpha)

    q0 = jnp.asarray(centers[0] + np.array([0.2, -0.15]))
    p0 = jnp.zeros_like(q0)

    def loss_fn(q):
        return 0.5 * jnp.sum(q**2)

    model = build(amp0)
    N_settle = 300 if quick else 1500
    spec = SettleSpec(steps=N_settle, dt=0.05, gamma=0.05, ridge=0.0)

    # ---- settle quality: ||p*||, residual, lambda_min (the Q3.5 triple) ----
    q_star, p_star = settle_forward(model, q0, p0, spec)
    tele = settle_telemetry(model, q_star[None, :], centers=centers, ridge=0.0)
    residual = float(tele["residual"][0])
    lam_min = float(tele["lambda_min"][0])
    lam_med = float(tele["lambda_median"][0])
    p_star_norm = float(jnp.linalg.norm(p_star))

    def _flat(tree):
        leaves = jax.tree_util.tree_leaves(eqx.filter(tree, eqx.is_inexact_array))
        return np.concatenate([np.asarray(x).ravel() for x in leaves]) if leaves else np.zeros(0)

    def _amp_grad(tree):
        """The differentiated parameter is ``V.amp``; pull just that leaf."""
        return np.asarray(tree.potential_net.amp)

    # ---- the implicit gradient ----
    t0 = time.time()
    g_imp = _amp_grad(implicit_grad(model, q0, p0, spec, loss_fn))
    t_imp = time.time() - t0

    # ---- finite differences: RE-SETTLE at theta +/- h (the ground truth) ----
    h = 1e-5
    g_fd = np.zeros_like(amp0)
    t0 = time.time()
    for i in range(n_w):
        ap, am = amp0.copy(), amp0.copy()
        ap[i] += h
        am[i] -= h
        qp, _ = settle_forward(build(ap), q0, p0, spec)
        qm, _ = settle_forward(build(am), q0, p0, spec)
        g_fd[i] = float((loss_fn(qp) - loss_fn(qm)) / (2.0 * h))
    t_fd = time.time() - t0

    def relerr(a, b):
        b = np.asarray(b, dtype=float)
        return float(np.linalg.norm(np.asarray(a, dtype=float) - b) / max(np.linalg.norm(b), 1e-300))

    # ---- truncated unroll at the theorist's registered depths ----
    depths = [180, 270, 449] if not quick else [30, 60]
    unroll = {}
    for k in depths:
        t0 = time.time()
        g_k = _amp_grad(unroll_grad(model, q0, p0, spec, loss_fn, retain=k))
        dt_k = time.time() - t0
        unroll[str(k)] = {
            "rel_err_vs_implicit": relerr(g_k, g_imp),
            "rel_err_vs_fd": relerr(g_k, g_fd),
            "wall_s": dt_k,
            "grad": np.asarray(g_k).tolist(),
        }
    # full backprop through the whole settle (the k = N limit)
    g_full = _amp_grad(unroll_grad(model, q0, p0, spec, loss_fn, retain=None))
    unroll["full_%d" % N_settle] = {
        "rel_err_vs_implicit": relerr(g_full, g_imp),
        "rel_err_vs_fd": relerr(g_full, g_fd),
        "grad": np.asarray(g_full).tolist(),
    }

    # ---- (gamma, dt) independence of the implicit answer (theory Q1.3) ----
    # ⚠ The step budget must use the theorist's TWO-branch rho (Q4.2), not the
    # underdamped-only `N = 2 ln(1/tol)/gamma`: at (gamma=0.3, dt=0.02) the well
    # mode is OVERDAMPED (lambda_crit = gamma^2 m / (2(2-gamma) dt^2) = 66.2 >
    # lambda = 8.21) and the naive budget is ~16x too short. Both budgets are
    # reported so the failure of the naive one is on the record.
    def rho_two_branch(gam, dtv, lam, m=1.0):
        lam_crit = gam**2 * m / (2.0 * (2.0 - gam) * dtv**2)
        under = np.sqrt(max(1.0 - gam, 0.0))
        over = 1.0 - (2.0 - gam) * dtv**2 * lam / (2.0 * gam * m)
        return (under if lam > lam_crit else max(over, under)), lam_crit

    grid = []
    for gam in ([0.05] if quick else [0.02, 0.05, 0.1, 0.3]):
        for dtv in ([0.05] if quick else [0.02, 0.05, 0.1]):
            rho, lam_crit = rho_two_branch(gam, dtv, lam_min)
            n_naive = int(np.ceil(2.0 * np.log(1e12) / gam))
            n_needed = int(np.ceil(np.log(1e12) / max(-np.log(rho), 1e-12)))
            for tag, nsteps in (("naive", n_naive), ("two_branch", n_needed)):
                sp = SettleSpec(steps=min(nsteps, 20000), dt=dtv, gamma=gam)
                g = _amp_grad(implicit_grad(model, q0, p0, sp, loss_fn))
                qs, ps = settle_forward(model, q0, p0, sp)
                grid.append({
                    "gamma": gam, "dt": dtv, "budget": tag,
                    "steps": int(min(nsteps, 20000)), "rho": float(rho),
                    "lambda_crit": float(lam_crit),
                    "overdamped": bool(lam_min < lam_crit),
                    "grad": np.asarray(g).tolist(),
                    "p_star_norm": float(jnp.linalg.norm(ps)),
                    "residual": float(jnp.linalg.norm(
                        jax.grad(lambda z: jnp.reshape(model.potential_net(z), ()))(qs)))})

    def _spread(rows):
        if not rows:
            return float("nan")
        G = np.array([r["grad"] for r in rows])
        return float(np.max(np.abs(G - G.mean(axis=0, keepdims=True))) /
                     max(np.max(np.abs(G)), 1e-300))

    spread = _spread([r for r in grid if r["budget"] == "two_branch"])
    spread_naive = _spread([r for r in grid if r["budget"] == "naive"])

    # ---- the ridge (never silently enabled) ----
    lam_r = theory_ridge(0.05, 0.05, 400, 1e-3, 1.0)
    g_ridged = _amp_grad(implicit_grad(model, q0, p0,
                                       replace(spec, ridge=lam_r), loss_fn))
    ridge_bias = relerr(g_ridged, g_imp)

    # ---- d q*/d q0 = 0 (PREREG §4c, the structural claim) ----
    def q_star_of_q0(z):
        return jnp.sum(implicit_settle(model, z, p0, spec) ** 2)

    dq0_implicit = float(jnp.linalg.norm(jax.grad(q_star_of_q0)(q0)))

    def _dq0_unrolled(nsteps: int) -> float:
        sp = replace(spec, steps=int(nsteps))

        def f(z):
            qs, _ = settle_forward(model, z, p0, sp)
            return jnp.sum(qs**2)

        return float(jnp.linalg.norm(jax.grad(f)(q0)))

    dq0_unrolled = _dq0_unrolled(N_settle)
    # Verify the theory's geometric-death law rho^N on the SAME object.
    rho_addr = float(np.sqrt(1.0 - spec.gamma))
    ladder = [50, 100, 200, 400, 800, 1500] if not quick else [50, 100, 200]
    dq0_ladder = []
    for n in ladder:
        v = _dq0_unrolled(n)
        dq0_ladder.append({"N": n, "norm": v, "rho_pow_N": float(rho_addr**n)})

    res = {
        "toy": {"n_wells": n_w, "dim": dim, "s": s_well, "alpha": alpha,
                "amp": amp0.tolist(), "q0": np.asarray(q0).tolist(),
                "N_settle": N_settle, "dtype": "float64"},
        "settle": {"p_star_norm": p_star_norm, "residual": residual,
                   "lambda_min": lam_min, "lambda_median": lam_med,
                   "cond": float(tele["cond"][0]),
                   "basin": int(tele["basin"][0]),
                   "d_nearest": float(tele["d_nearest"][0])},
        "grad_implicit": np.asarray(g_imp).tolist(),
        "grad_fd": g_fd.tolist(),
        "rel_err_implicit_vs_fd": relerr(g_imp, g_fd),
        "wall_s_implicit": t_imp,
        "wall_s_fd_4param": t_fd,
        "unroll": unroll,
        "gamma_dt_grid": grid,
        "gamma_dt_spread": spread,
        "gamma_dt_spread_naive_budget": spread_naive,
        "ridge": {"lambda_ridge": lam_r, "rel_bias_vs_unridged": ridge_bias,
                  "alarm": bool(ridge_alarm(lam_r, [lam_med]))},
        "dq_star_dq0": {"implicit": dq0_implicit, "unrolled_full": dq0_unrolled,
                        "rho_address": rho_addr, "ladder": dq0_ladder},
    }
    return res


# ==========================================================================
# shared: the HEALTHY store, and the ambiguity-graded query set
# ==========================================================================
def build_store(seed: int = 0, stage: str = "S0_baseline", quick: bool = False):
    """The shipped harness store at the **healthy** geometry (``sep/sigma_q = 6.83``).

    ⚠ Never S4's collapsed 3.07: the harness showed a trajectory psi cannot
    demonstrate value where addressing has already failed.
    """
    import jax

    from chlu.core.clu_system import CluSystemConfig, build_system
    from chlu.experiments.exp_clu_system import make_stream, stage_config

    base = CluSystemConfig(seed=int(seed), quick=bool(quick))
    cfg = stage_config(stage, base)
    system = build_system(cfg, key=jax.random.PRNGKey(int(seed)), loud=False)
    items, sites, pays = make_stream(cfg, n_offer=cfg.capacity + 2, seed=int(seed))
    rep = system.write_stream(items)
    ids, centers, live_pays = system.codebook()
    sep = float(np.min([np.linalg.norm(centers[a] - centers[b])
                        for a in range(len(ids)) for b in range(len(ids)) if a != b]))
    info = {
        "stage": stage, "n_live": int(len(ids)),
        "admitted": len(rep.admitted), "refused": len(rep.refused),
        "evicted": len(rep.evicted), "deleted": len(rep.deleted),
        "sep": sep, "sigma_q": float(cfg.query_sigma),
        "sep_over_sigma_q": sep / float(cfg.query_sigma),
        "n_bytes": int(system.n_bytes()),
    }
    return system, cfg, info


AMBIGUITY_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.45)


def ambiguity_queries(centers: np.ndarray, sigma_q: float, dim: int, addr_dim: int,
                      seed: int = 0, t_grid=AMBIGUITY_GRID, n_rep: int = 3):
    """Queries interpolated between an item ``i`` and a **random** competitor ``j``.

    ⭐ Charter §2.1(c) is the hypothesis under test: *a trajectory passing near
    competing wells encodes a distribution over answers; a settled point cannot.*
    So the query set is graded by ambiguity ``t``: ``q = (1-t) c_i + t c_j``, plus
    the shipped query jitter. ``j`` is drawn **uniformly among the other live
    items**, not taken as ``i``'s nearest neighbour — otherwise ``j`` would be a
    deterministic function of ``i`` and "predict the competitor" would collapse
    into "predict the winner".
    """
    rng = np.random.default_rng(int(seed) + 31337)
    K = centers.shape[0]
    rows, wi, wj, tt = [], [], [], []
    for i in range(K):
        for j in range(K):
            if j == i:
                continue
            for t in t_grid:
                for _ in range(int(n_rep)):
                    a = (1.0 - t) * centers[i] + t * centers[j]
                    a = a + rng.normal(size=addr_dim) * float(sigma_q)
                    rows.append(a)
                    wi.append(i)
                    wj.append(j)
                    tt.append(float(t))
    q0 = np.zeros((len(rows), dim), dtype=np.float32)
    q0[:, :addr_dim] = np.asarray(rows, dtype=np.float32)
    return q0, {"winner": np.asarray(wi), "competitor": np.asarray(wj),
                "t": np.asarray(tt)}


# ==========================================================================
# STAGE 0 — the blocking axis-liveness gate (Hub amendment, monitor #10)
# ==========================================================================
CLASS_PROBES = ("linear", "knn")
REG_PROBES = ("ridge", "knn_reg")


def _probe_scores(X: np.ndarray, y: np.ndarray, *, kind: str, seed: int = 0,
                  n_folds: int = 5, n_pca: Optional[int] = None):
    """5-fold CV out-of-fold predictions of a fixed-capacity probe.

    ⚠ Any dimensionality reduction is fit **inside** the fold, so the
    matched-capacity ``full_pca`` feature set carries no transductive leakage.
    """
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression, RidgeCV
    from sklearn.model_selection import KFold
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=int(seed))
    preds = np.zeros(y.shape[0], dtype=np.float64 if kind in REG_PROBES else y.dtype)
    folds = []
    for tr, te in kf.split(X):
        steps = [StandardScaler()]
        if n_pca is not None and n_pca < X.shape[1]:
            steps.append(PCA(n_components=int(n_pca), random_state=int(seed)))
            steps.append(StandardScaler())
        if kind == "linear":
            steps.append(LogisticRegression(max_iter=3000, C=1.0))
        elif kind == "knn":
            steps.append(KNeighborsClassifier(n_neighbors=5))
        elif kind == "ridge":
            steps.append(RidgeCV(alphas=np.logspace(-3, 5, 17)))
        elif kind == "knn_reg":
            steps.append(KNeighborsRegressor(n_neighbors=5))
        else:
            raise ValueError(kind)
        est = make_pipeline(*steps)
        est.fit(X[tr], y[tr])
        p = est.predict(X[te])
        preds[te] = p
        if kind in CLASS_PROBES:
            folds.append(float(np.mean(p == y[te])))
        else:
            ss = float(np.sum((y[te] - y[te].mean()) ** 2))
            folds.append(float(1.0 - np.sum((p - y[te]) ** 2) / max(ss, 1e-12)))
    score = _score_from_preds(preds, y, kind)
    return score, folds, preds


def _score_from_preds(preds, y, kind) -> float:
    if kind in CLASS_PROBES:
        return float(np.mean(preds == y))
    return float(1.0 - np.sum((preds - y) ** 2) /
                 max(float(np.sum((y - y.mean()) ** 2)), 1e-12))


def _paired_se(pred_a: np.ndarray, pred_b: np.ndarray, y: np.ndarray,
               kind: str = "linear", n_boot: int = 400, seed: int = 0) -> float:
    """Paired s.e. of the score DIFFERENCE ``score(a) - score(b)``.

    Classification: McNemar (exact, no resampling). Regression: paired bootstrap
    over the out-of-fold residuals (R^2 is not a per-sample statistic).
    """
    if kind in CLASS_PROBES:
        a, b = (pred_a == y), (pred_b == y)
        n01 = int(np.sum(a & ~b))
        n10 = int(np.sum(~a & b))
        return float(np.sqrt(max(n01 + n10, 1)) / y.shape[0])
    rng = np.random.default_rng(int(seed))
    n = y.shape[0]
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[b] = (_score_from_preds(pred_a[idx], y[idx], kind)
                    - _score_from_preds(pred_b[idx], y[idx], kind))
    return float(np.std(diffs))


def _feature_sets(traj: np.ndarray, q0: np.ndarray, q_addr: np.ndarray,
                  q_star: np.ndarray, p_star: np.ndarray, *, stride: int,
                  n_pca: int = 20, window=None):
    """``q0_only`` / ``endpoints`` / ``full_pca`` (matched capacity) / ``full_raw``.

    ``endpoints`` is the **fair point baseline**: it holds the query embedding
    ``q0`` *and* the settled point, so any trajectory gain is attributable to the
    intermediate points and not to the ``q0 = phi(x)`` leak. ``full_pca`` is
    ``full_raw`` projected (inside the CV fold) to exactly ``dim(endpoints)``, so
    the comparison is **capacity-matched** as well as byte-matched.
    """
    sub = traj if window is None else traj[:, window, :]
    sub = sub[:, ::int(stride), :]
    flat = sub.reshape(sub.shape[0], -1)
    endpoints = np.concatenate([q0, q_addr, q_star, p_star], axis=1)
    q_star_only = np.concatenate([q_star, p_star], axis=1)
    k = int(min(n_pca, flat.shape[1]))
    return (
        {"q0_only": (q0, None),
         # the classical 26-wave read, and the INSTRUMENT POSITIVE CONTROL:
         # the trajectory provably contains q0 and q_star_only provably does not,
         # so `full - q_star_only` must be large or the probe pipeline is broken.
         "q_star_only": (q_star_only, None),
         "endpoints": (endpoints, None),
         "full_pca": (flat, k), "full_raw": (flat, None)},
        {"n_points": int(sub.shape[1]), "dim_full_raw": int(flat.shape[1]),
         "dim_endpoints": int(endpoints.shape[1]), "dim_full_pca": int(k),
         "dim_q_star_only": int(q_star_only.shape[1])},
    )


def part_stage0(seed: int = 0, quick: bool = False, n_rep: int = 6) -> dict:
    """⭐ Does the strided trajectory carry information the settled point does not?

    Runs at the **healthy** S0 geometry. Reports, for every cell of the swept
    regime, the probe score of each feature set and the **gain**
    ``full - endpoints`` with its paired 3-sigma bar. The registered gate
    (PREREG §4a) is applied at the end; Part B does not run unless it passes.
    """

    from chlu.core.clu_system import settled_point_psi, tail_mean_psi

    t0 = time.time()
    system, cfg, info = build_store(seed=seed, quick=quick)
    ids, centers, pays = system.codebook()
    K = len(ids)
    n_rep = 1 if quick else int(n_rep)
    q0, meta = ambiguity_queries(centers, cfg.query_sigma, system.store.dim,
                                 system.store.addr_dim, seed=seed, n_rep=n_rep)
    info["n_queries"] = int(q0.shape[0])
    info["build_wall_s"] = time.time() - t0

    strides = [8] if quick else [1, 2, 4, 8, 16, 32]
    # ⚠ The regime sweep must include the knobs that *should* wake the axis, not
    # only the shipped ones (Hub amendment bullet 2):
    #   * `gamma_read = 0` — the conservative read. Theory Q1.1b: at gamma = 0 the
    #     map is symplectic, Liouville forbids an attractor, so there IS no
    #     settled point to read and the trajectory is all there is. If the
    #     trajectory channel is ever alive, it is alive here.
    #   * short reads — an unconverged settle makes q* arbitrary.
    # (gamma_read, address_steps, read_steps)
    if quick:
        regimes = [(cfg.gamma_read, cfg.address_steps, cfg.read_steps)]
    else:
        regimes = [
            (0.0, 400, 800), (0.005, 400, 800), (0.02, 400, 800),
            (0.08, 400, 800), (0.2, 400, 800),
            (0.02, 400, 50), (0.02, 400, 200), (0.02, 400, 1600),
            (0.02, 50, 800), (0.02, 100, 800),
            (0.0, 400, 200), (0.0, 100, 800),
        ]
    reference = (cfg.gamma_read, cfg.address_steps, cfg.read_steps)
    # (target -> (probe kinds, y)). The task asks for "linear probe / kNN"; both
    # run on the two competing-well targets, the cheap ones only linear/ridge.
    targets = {
        "winner_id": (("linear",), meta["winner"]),
        "competitor_id": (("linear", "knn"), meta["competitor"]),
        "competitor_payload": (("ridge", "knn_reg"), pays[meta["competitor"], 0]),
        "ambiguity_t": (("ridge",), meta["t"]),
    }

    rows = []
    axis_liveness = []
    saved = (cfg.traj_stride, cfg.gamma_read, cfg.address_steps, cfg.read_steps)
    try:
        cfg.traj_stride = 1  # read once at full resolution; stride is post-hoc
        for gr, n_addr, n_read in regimes:
            is_ref = (gr, n_addr, n_read) == reference
            cfg.gamma_read = float(gr)
            cfg.address_steps = int(n_addr)
            cfg.read_steps = int(n_read)
            tr0 = time.time()
            res = system.read(q0)
            read_s = time.time() - tr0
            traj = np.asarray(res.traj)
            phase = np.asarray(res.phase)
            st = res.state
            q0n = np.asarray(st.q0)
            q_addr = np.asarray(st.q_addr)
            q_star = np.asarray(st.q_star)
            p_star = np.asarray(st.p_star)
            reg = {"gamma_read": float(gr), "address_steps": int(n_addr),
                   "read_steps": int(n_read)}

            # --- axis liveness in monitor #10's own noise units --------------
            for name, psi in (("settled_point", settled_point_psi(cfg.addr_dim,
                                                                  cfg.payload_dim)),
                              ("tail_mean", tail_mean_psi(cfg.addr_dim,
                                                          cfg.payload_dim))):
                base = np.asarray(psi(res.traj, st))
                noise = max(float(np.std(base)) * 1e-3, 1e-9)
                for s in strides:
                    sub_t = res.traj[:, ::s, :]
                    v = np.asarray(psi(sub_t, st))
                    axis_liveness.append({
                        **reg, "psi": name, "knob": "traj_stride", "stride": int(s),
                        "movement_noise_units": float(np.max(np.abs(v - base)) / noise),
                    })

            sweep_strides = strides if is_ref else [8]
            windows = {"both": None,
                       "phase1": np.where(phase == 1)[0],
                       "phase2": np.where(phase == 2)[0]}
            if not is_ref:
                windows.pop("phase2")
            for wname, widx in windows.items():
                for s in sweep_strides:
                    feats, dims = _feature_sets(traj, q0n, q_addr, q_star, p_star,
                                                stride=s, window=widx)
                    for tname, (kinds, y) in targets.items():
                        if not is_ref and tname in ("winner_id", "ambiguity_t"):
                            continue  # cost guard: the gate lives on the competitor
                        for kind in kinds:
                            scores, preds = {}, {}
                            for fname, (X, npca) in feats.items():
                                if fname == "full_raw" and X.shape[1] > 4000:
                                    continue  # cost guard; full_pca is the matched one
                                sc, _f, pr = _probe_scores(X, y, kind=kind, seed=seed,
                                                           n_pca=npca)
                                scores[fname] = sc
                                preds[fname] = pr
                            for band, mask in (
                                ("all", np.ones_like(meta["t"], dtype=bool)),
                                ("unambiguous", meta["t"] <= 0.10),
                                ("ambiguous", meta["t"] >= 0.35),
                            ):
                                row = {**reg, "window": wname, "stride": int(s),
                                       "target": tname, "probe": kind, "band": band,
                                       "n": int(mask.sum()), "read_wall_s": read_s,
                                       "is_reference": bool(is_ref), **dims}
                                for fname in preds:
                                    row[f"score_{fname}"] = _score_from_preds(
                                        preds[fname][mask], y[mask], kind)
                                row["gain_full_pca_minus_endpoints"] = (
                                    row["score_full_pca"] - row["score_endpoints"])
                                row["se_paired"] = _paired_se(
                                    preds["full_pca"][mask], preds["endpoints"][mask],
                                    y[mask], kind=kind, seed=seed)
                                # the INSTRUMENT POSITIVE CONTROL
                                row["control_full_minus_qstar_only"] = (
                                    row["score_full_pca"] - row["score_q_star_only"])
                                row["control_se"] = _paired_se(
                                    preds["full_pca"][mask], preds["q_star_only"][mask],
                                    y[mask], kind=kind, seed=seed)
                                if "full_raw" in preds:
                                    row["gain_full_raw_minus_endpoints"] = (
                                        row["score_full_raw"] - row["score_endpoints"])
                                if kind in CLASS_PROBES:
                                    _, cnt = np.unique(y[mask], return_counts=True)
                                    row["chance"] = float(np.max(cnt) / max(mask.sum(), 1))
                                rows.append(row)
    finally:
        (cfg.traj_stride, cfg.gamma_read, cfg.address_steps,
         cfg.read_steps) = saved

    # --- the registered gate (PREREG §4a) --------------------------------
    qualifying = [
        r for r in rows
        if r.get("band") == "ambiguous"
        and r["target"] in ("competitor_id", "competitor_payload")
        and r.get("gain_full_pca_minus_endpoints", -1) >= 0.10
        and r["gain_full_pca_minus_endpoints"] >= 3.0 * r["se_paired"]
    ]
    best = max(rows, key=lambda r: r.get("gain_full_pca_minus_endpoints", -9)
               if r.get("band") == "ambiguous" else -9)
    # instrument positive control: `full - q_star_only` on winner_id must be
    # large (the trajectory provably contains q0, q_star_only provably does not).
    pc = [r for r in rows if r["target"] == "winner_id" and r["band"] == "all"
          and r["window"] == "both"]
    pc_best = max(pc, key=lambda r: r["control_full_minus_qstar_only"]) if pc else None
    return {
        "store": info, "K": int(K),
        "sweep": {"regimes": [list(x) for x in regimes], "strides": strides,
                  "t_grid": list(AMBIGUITY_GRID), "n_rep": n_rep,
                  "reference": list(reference)},
        "rows": rows,
        "axis_liveness": axis_liveness,
        "instrument_positive_control": pc_best,
        "gate": {"passed": bool(len(qualifying) > 0),
                 "n_qualifying_cells": len(qualifying),
                 "n_cells_examined": len(rows),
                 "best_ambiguous_cell": best,
                 "qualifying": qualifying[:20]},
        "wall_s": time.time() - t0,
    }


# ==========================================================================
# a TRACEABLE two-phase read (CluSystem.read mixes numpy into its diagnostics)
# ==========================================================================
def differentiable_read(model, q0, cfg, *, retain_phase1=None,
                        retain_phase2=None, stride=None, implicit_q_star: bool = True,
                        ridge: float = 0.0):
    """A fully traceable re-implementation of ``CluSystem.read``'s two phases.

    ⚠ ``CluSystem.read`` is **frozen API and must not be edited**, and it calls
    ``np.asarray`` on its diagnostics, so it is not traceable end to end. This
    function reproduces its *dynamics* exactly (phase 1 at ``gamma_address`` for
    ``address_steps``, phase 2 at ``gamma_read`` for ``read_steps``, each strided
    by ``traj_stride`` and concatenated) and is verified against it to float32
    round-off in ``tests/test_psi_readout.py``. No anneal schedule, no retry.

    Gradient plumbing (theory §7 requests 1 and 7):

    * phase 1 — ``retain_phase1`` steps of backprop (``None`` = the whole
      400-step window, which is only 1.5x the theorist's useful depth
      ``k* = 269``, so nothing is bought by truncating it);
    * phase 2 — full backprop (``rho = 0.98995`` at the shipped
      ``gamma_read = 0.02``, so ``k*(1e-3) = 684`` and the whole 800-step window
      is inside the useful depth);
    * ``q_star`` — the **implicit** gradient when ``implicit_q_star`` (the
      settle is a fixed point), which is also what makes ``d q*/d q0 = 0``
      explicit rather than a 1e-8 numerical accident.

    ⚠ **Truncation direction is not a free choice, and the theorist's depth is
    stated for the wrong end if you are training ``phi``.** Tail truncation
    (retain the LAST ``k`` steps) is correct for ``theta``: the late steps
    dominate the gradient to the endpoint. But it severs the path to ``q0``
    *completely* — the retained window is entered through a ``stop_gradient`` —
    so ``d loss / d phi`` becomes **exactly 0**, not ``rho^k``. Measured in
    ``part_e2e``'s truncation study. Keep ``retain_phase1=None`` whenever ``phi``
    is being trained.
    """
    import jax
    import jax.numpy as jnp

    from chlu.core.clu_system import ReadState
    from chlu.core.implicit_grad import SettleSpec, implicit_settle, truncated_rollout

    st = int(cfg.traj_stride if stride is None else stride)
    q0 = jnp.asarray(q0, dtype=jnp.float32)
    p0 = jnp.zeros_like(q0)
    d = q0.shape[-1]

    def one(qa, pa):
        tr1, q_addr, p_addr = truncated_rollout(
            model, qa, pa, int(cfg.address_steps), float(cfg.dt),
            float(cfg.gamma_address), retain=retain_phase1, stride=st,
            return_endpoint=True)
        r2 = int(cfg.read_steps) if retain_phase2 is None else int(retain_phase2)
        tr2, q_star, p_star = truncated_rollout(
            model, q_addr, p_addr, int(cfg.read_steps), float(cfg.dt),
            float(cfg.gamma_read), retain=r2, stride=st, return_endpoint=True)
        if implicit_q_star:
            spec = SettleSpec(steps=int(cfg.read_steps), dt=float(cfg.dt),
                              gamma=float(cfg.gamma_read), ridge=float(ridge))
            q_star = implicit_settle(model, jax.lax.stop_gradient(q_addr),
                                     jax.lax.stop_gradient(p_addr), spec)
            p_star = jax.lax.stop_gradient(p_star)
        return jnp.concatenate([tr1, tr2], axis=0), q_addr, p_addr, q_star, p_star

    traj, q_addr, p_addr, q_star, p_star = jax.vmap(one)(q0, p0)
    n1 = traj.shape[1] - int(np.ceil(int(cfg.read_steps) / st))
    phase = np.concatenate([np.ones(n1, dtype=int),
                            2 * np.ones(traj.shape[1] - n1, dtype=int)])
    state = ReadState(q0=q0, p0=p0, q_addr=q_addr, p_addr=p_addr,
                      q_star=q_star, p_star=p_star)
    _ = d
    return traj, phase, state


# ==========================================================================
# ACCEPTANCE HALF 1 — gradients flow query -> phi -> settle -> psi -> loss
# ==========================================================================
def part_e2e(seed: int = 0, quick: bool = False, n_steps: int = 20,
             batch: int = 32) -> dict:
    """End-to-end trainability + the declared wall-clock budget (PREREG §3).

    Also measures the PREREG §4c structural prediction: with the implicit settle
    the **settled-point** read sends *no* gradient to ``phi``, while the
    trajectory read sends an O(1) one.
    """
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import optax

    from chlu.core.implicit_grad import settle_telemetry, theory_ridge
    from chlu.core.psi_readout import (
        DeepSetsPsi,
        LearnedPhi,
        PsiSpec,
        psi_param_count,
    )

    t_all = time.time()
    system, cfg, info = build_store(seed=seed, quick=quick)
    ids, centers, pays = system.codebook()
    K = len(ids)
    model = system.model()
    n_steps = 4 if quick else int(n_steps)

    key = jax.random.PRNGKey(seed + 77)
    k_phi, k_psi, k_q = jax.random.split(key, 3)
    phi = LearnedPhi(cfg.addr_dim, system.store.dim, cfg.addr_dim, cfg.payload_dim,
                     hidden=32, depth=2, key=k_phi)
    base_spec = PsiSpec(dim=system.store.dim, addr_dim=cfg.addr_dim,
                        payload_dim=cfg.payload_dim, hidden=32, depth=2)

    # the query batch (raw x -> phi -> q0)
    lab = np.asarray(jax.random.randint(k_q, (batch,), 0, K))
    x = jnp.asarray(centers[lab] + np.asarray(
        jax.random.normal(jax.random.fold_in(k_q, 1), (batch, cfg.addr_dim))
    ) * cfg.query_sigma, dtype=jnp.float32)
    y = jnp.asarray(pays[lab], dtype=jnp.float32)

    lam_r = theory_ridge(cfg.gamma_read, cfg.dt, cfg.read_steps, 1e-3, 1.0)
    out: Dict[str, Any] = {"store": info, "K": int(K), "batch": int(batch),
                           "n_steps": n_steps, "lambda_ridge_phase2": lam_r}

    def make_loss(input_mode: str, implicit: bool, retain1=None):
        spec = replace(base_spec, input_mode=input_mode)
        psi = DeepSetsPsi(spec, k_psi)

        def loss(phi_, psi_, model_):
            q0 = phi_(x)
            traj, _ph, st = differentiable_read(
                model_, q0, cfg, retain_phase1=retain1, implicit_q_star=implicit)
            v = psi_(traj, st)
            return jnp.mean((v - y) ** 2)

        return psi, loss

    # ---- the PREREG §4c measurement: which arm reaches phi? --------------
    def _grads(loss_fn, psi_):
        return eqx.filter_grad(lambda tr: loss_fn(*tr))((phi, psi_, model))

    grad_norms = {}
    for mode in ("settled_point", "trajectory", "endpoints"):
        for impl in (True, False):
            psi_, loss = make_loss(mode, impl)
            gphi, gpsi, gth = _grads(loss, psi_)
            grad_norms[f"{mode}|{'implicit' if impl else 'unrolled'}"] = {
                "phi": _tree_norm(gphi), "psi": _tree_norm(gpsi),
                "theta": _tree_norm(gth),
            }
    out["grad_norms"] = grad_norms

    # ---- A4: the TRUNCATION STUDY (theory Q4.2 / §7 request 7) -----------
    # Retain the last `k` steps of the 400-step address phase and watch what the
    # theorist's `rho^k` law does to (a) the theta gradient and (b) the phi
    # gradient. They behave completely differently, and that is the finding.
    trunc, gth_full = [], None
    for k in ([0, 400] if quick else [0, 50, 100, 180, 270, 400]):
        psi_, loss = make_loss("trajectory", True, retain1=k)
        gphi, gpsi, gth = _grads(loss, psi_)
        if k == 400:
            gth_full = gth
        trunc.append({"retain_phase1": k, "phi": _tree_norm(gphi),
                      "psi": _tree_norm(gpsi), "theta": _tree_norm(gth),
                      "_gth": gth,
                      "rho_pow_k": float(np.sqrt(1.0 - cfg.gamma_address) ** k)})
    for rec in trunc:
        rec["theta_rel_err_vs_full"] = _tree_relerr(rec.pop("_gth"), gth_full)
    out["truncation_study"] = trunc

    # ---- the training loop + wall clock ---------------------------------
    def _train_arm(loss, psi, n):
        """Plain Adam on (phi, psi) — charter §2.4: standard machinery, no
        bespoke optimizer. Returns the timing/loss record for one arm."""
        params = (phi, psi)
        opt = optax.adam(1e-3)
        opt_state = opt.init(eqx.filter(params, eqx.is_inexact_array))

        @eqx.filter_jit
        def step(prm, ost):
            val, grads = eqx.filter_value_and_grad(
                lambda pr: loss(pr[0], pr[1], model))(prm)
            upd, ost = opt.update(grads, ost, eqx.filter(prm, eqx.is_inexact_array))
            return eqx.apply_updates(prm, upd), ost, val

        t0 = time.time()
        params, opt_state, val = step(params, opt_state)
        val.block_until_ready()
        compile_s = time.time() - t0
        losses, per_step = [float(val)], []
        for _ in range(n):
            t0 = time.time()
            params, opt_state, val = step(params, opt_state)
            val.block_until_ready()
            per_step.append(time.time() - t0)
            losses.append(float(val))
        return {"compile_s": compile_s,
                "median_step_s": float(np.median(per_step)),
                "mean_step_s": float(np.mean(per_step)),
                "loss_first": losses[0], "loss_last": losses[-1],
                "losses": losses,
                "psi_params": psi_param_count(psi),
                "phi_params": psi_param_count(phi)}

    timings = {}
    for mode in ("settled_point", "trajectory"):
        psi_, loss = make_loss(mode, True)
        timings[mode] = _train_arm(loss, psi_, n_steps)
    out["training"] = timings
    out["budget_s"] = 30.0
    out["falsifier_s"] = 300.0
    out["budget_met"] = bool(
        max(v["median_step_s"] for v in timings.values()) <= 30.0)
    out["falsifier_fired"] = bool(
        max(v["median_step_s"] for v in timings.values()) > 300.0)

    # ---- A3/A4 telemetry: the Q3.5 triple at the read -------------------
    q0 = phi(x)
    _tr, _ph, st = differentiable_read(model, q0, cfg, retain_phase1=270)
    tele = settle_telemetry(model, st.q_star, centers=centers, ridge=lam_r)
    out["telemetry"] = {
        "residual_median": float(np.median(tele["residual"])),
        "residual_max": float(np.max(tele["residual"])),
        "lambda_min_median": float(np.median(tele["lambda_min"])),
        "lambda_min_min": float(np.min(tele["lambda_min"])),
        "lambda_median_median": float(np.median(tele["lambda_median"])),
        "cond_median": float(np.median(tele["cond"])),
        "n_negative_modes_max": int(np.max(tele["n_negative_modes"])),
        "basin_correct_frac": float(np.mean(tele["basin"] == lab)),
        "ridge": lam_r, "ridge_alarm": bool(tele["ridge_alarm"]),
    }
    out["wall_s"] = time.time() - t_all
    return out


# ==========================================================================
# PART B — the point-vs-trajectory ablation (GATED on Stage 0)
# ==========================================================================
def _read_cache(system, cfg, q0, stride: int):
    """One read at ``stride``; returns ``(traj, ReadState)`` ready for a psi."""
    old = cfg.traj_stride
    try:
        cfg.traj_stride = int(stride)
        res = system.read(q0)
        return res.traj, res.state
    finally:
        cfg.traj_stride = old


def _fit_psi(psi, traj, state, y, *, steps: int, lr: float = 3e-3, seed: int = 0):
    """Plain Adam on psi over a FIXED read (charter §2.4: standard machinery).

    The store, ``phi``, the bytes and the read are identical across arms; only
    ``psi.spec.input_mode`` differs, and the parameter count is identical by
    construction (:func:`chlu.core.psi_readout.matched_pair`).
    """
    import equinox as eqx
    import jax.numpy as jnp
    import optax

    opt = optax.adam(lr)
    opt_state = opt.init(eqx.filter(psi, eqx.is_inexact_array))
    yj = jnp.asarray(y, dtype=jnp.float32)

    @eqx.filter_jit
    def step(p, ost):
        val, g = eqx.filter_value_and_grad(
            lambda m: jnp.mean((m(traj, state) - yj) ** 2))(p)
        upd, ost = opt.update(g, ost, eqx.filter(p, eqx.is_inexact_array))
        return eqx.apply_updates(p, upd), ost, val

    hist = []
    for _ in range(int(steps)):
        psi, opt_state, val = step(psi, opt_state)
        hist.append(float(val))
    return psi, hist


def _decode(vals: np.ndarray, pays: np.ndarray, labels: np.ndarray) -> float:
    d = np.linalg.norm(np.asarray(vals)[:, None, :] - pays[None, :, :], axis=-1)
    return float(np.mean(np.argmin(d, axis=1) == labels))


def part_b(seed: int = 0, quick: bool = False, family: str = "deepsets",
           fit_steps: int = 2000, gate_passed: Optional[bool] = None) -> dict:
    """The ablation: settled-point psi vs trajectory psi, everything else matched.

    ⛔ **GATED on Stage 0.** If Stage 0 did not find a live trajectory axis, a
    flat ablation here is monitor #10 firing, not a pillar-1 result; the runner
    records ``gate_passed`` so the number can never be quoted without it.
    """
    import equinox as eqx
    import jax
    import numpy as _np

    from chlu.core.clu_system import build_system
    from chlu.core.psi_readout import PsiSpec, make_psi, psi_param_count
    from chlu.eval.dividend import byte_account, dividend, trajectory_launder

    t_all = time.time()
    system, cfg, info = build_store(seed=seed, quick=quick)
    ids, centers, pays = system.codebook()
    K = len(ids)
    fit_steps = 60 if quick else int(fit_steps)

    q0, meta = ambiguity_queries(centers, cfg.query_sigma, system.store.dim,
                                 system.store.addr_dim, seed=seed,
                                 n_rep=1 if quick else 4)
    lab = meta["winner"]
    y_win = pays[lab]
    y_cmp = pays[meta["competitor"]]
    n = q0.shape[0]
    rng = _np.random.default_rng(seed)
    tr_idx = rng.permutation(n)
    n_tr = int(0.7 * n)
    itr, ite = tr_idx[:n_tr], tr_idx[n_tr:]

    strides = [8] if quick else [1, 2, 4, 8, 16, 32]
    base_spec = PsiSpec(dim=system.store.dim, addr_dim=cfg.addr_dim,
                        payload_dim=cfg.payload_dim, hidden=32, depth=2)
    k_psi = jax.random.PRNGKey(seed + 909)

    rows, launder_rows = [], []
    for stride in strides:
        traj, state = _read_cache(system, cfg, q0, stride)
        tr_sub = jax.tree_util.tree_map(lambda a: a[itr], traj)
        st_tr = jax.tree_util.tree_map(lambda a: a[itr], state)
        te_sub = jax.tree_util.tree_map(lambda a: a[ite], traj)
        st_te = jax.tree_util.tree_map(lambda a: a[ite], state)
        for target, yy in (("winner_payload", y_win), ("competitor_payload", y_cmp)):
            arms = {}
            for mode in ("settled_point", "endpoints", "trajectory"):
                spec = replace(base_spec, input_mode=mode)
                psi = make_psi(family, spec, k_psi)  # SAME key => same init
                psi, hist = _fit_psi(psi, tr_sub, st_tr, yy[itr], steps=fit_steps,
                                     seed=seed)
                v = _np.asarray(psi(te_sub, st_te))
                arms[mode] = {
                    "decode": _decode(v, pays, (lab if target == "winner_payload"
                                                else meta["competitor"])[ite]),
                    "rmse": float(_np.sqrt(_np.mean((v - _np.asarray(yy)[ite]) ** 2))),
                    "params": psi_param_count(psi),
                    "loss_first": hist[0], "loss_last": hist[-1],
                    "psi": psi,
                }
            assert arms["settled_point"]["params"] == arms["trajectory"]["params"]
            row = {"stride": int(stride), "target": target, "family": family,
                   "n_train": int(n_tr), "n_test": int(len(ite)),
                   "params_matched": True,
                   "params": arms["trajectory"]["params"],
                   "chance": 1.0 / K}
            for m, a in arms.items():
                row[f"decode_{m}"] = a["decode"]
                row[f"rmse_{m}"] = a["rmse"]
                row[f"loss_last_{m}"] = a["loss_last"]
            row["ablation_traj_minus_point"] = (
                arms["trajectory"]["decode"] - arms["settled_point"]["decode"])
            row["ablation_traj_minus_endpoints"] = (
                arms["trajectory"]["decode"] - arms["endpoints"]["decode"])
            se = float(_np.sqrt(2.0 * 0.5 * 0.5 / max(len(ite), 1)))
            row["se_conservative"] = se
            # ambiguity-resolved
            for band, mask in (("unambiguous", meta["t"][ite] <= 0.10),
                               ("ambiguous", meta["t"][ite] >= 0.35)):
                for m, a in arms.items():
                    vv = _np.asarray(a["psi"](te_sub, st_te))[mask]
                    tgt = (lab if target == "winner_payload"
                           else meta["competitor"])[ite][mask]
                    row[f"decode_{m}_{band}"] = _decode(vv, pays, tgt)
                row[f"ablation_{band}"] = (row[f"decode_trajectory_{band}"]
                                           - row[f"decode_settled_point_{band}"])
                row[f"n_{band}"] = int(mask.sum())
            rows.append(row)

            # ---- THE TRAJECTORY LAUNDER (mandatory; never run before) ------
            if target == "winner_payload":
                psi_t = arms["trajectory"]["psi"]
                lau = trajectory_launder(psi_t, te_sub, st_te)
                tgt = lab[ite]
                lr_ = {"stride": int(stride), "psi": "trajectory",
                       "representation": psi_t.representation}
                for k_, v_ in lau.items():
                    lr_[f"decode_{k_}"] = _decode(_np.asarray(v_), pays, tgt)
                # blank-store control at the SAME learned psi (the N68 config)
                blank = build_system(replace(cfg, seed=seed + 991),
                                     key=jax.random.PRNGKey(seed + 991),
                                     psi=psi_t, loud=False)
                b_old = blank.cfg.traj_stride
                blank.cfg.traj_stride = int(stride)
                b_res = blank.read(q0[ite])
                blank.cfg.traj_stride = b_old
                lr_["decode_blank_store"] = _decode(
                    _np.asarray(b_res.value).reshape(len(ite), -1), pays, tgt)
                _, cnt = _np.unique(tgt, return_counts=True)
                chance = float(_np.max(cnt) / len(tgt))
                lr_["chance"] = chance
                lr_["bar"] = chance + 3.0 * float(
                    _np.sqrt(max(chance * (1 - chance), 1e-12) / len(tgt)))
                lr_["blank_leak"] = bool(lr_["decode_blank_store"] > lr_["bar"])
                launder_rows.append(lr_)

    ba = byte_account(system, centers, pays)
    ref = [r for r in rows if r["stride"] == 8 and r["target"] == "winner_payload"]
    div = None
    if ref:
        r = ref[0]
        div = dividend(r["decode_trajectory"], r["decode_settled_point"],
                       metric="decode (trajectory psi vs settled-point psi)",
                       controls={"endpoints_psi": r["decode_endpoints"],
                                 "chance": r["chance"]},
                       bytes_account=ba,
                       flags={"stride": 8, "family": family,
                              "fit_steps": fit_steps, "seed": seed}).as_dict()
    _ = eqx
    return {"store": info, "K": int(K), "gate_passed": gate_passed,
            "family": family, "fit_steps": fit_steps,
            "rows": [{k: v for k, v in r.items() if k != "psi"} for r in rows],
            "trajectory_launder": launder_rows,
            "bytes": ba.as_dict(), "internal_dividend": div,
            "wall_s": time.time() - t_all}


def _tree_flat(tree) -> np.ndarray:
    import equinox as eqx
    import jax

    leaves = jax.tree_util.tree_leaves(eqx.filter(tree, eqx.is_inexact_array))
    if not leaves:
        return np.zeros(0)
    return np.concatenate([np.asarray(x).ravel() for x in leaves])


def _tree_relerr(a, b) -> float:
    """Relative error of two gradient PyTrees, as vectors (not norms)."""
    fa, fb = _tree_flat(a), _tree_flat(b)
    return float(np.linalg.norm(fa - fb) / max(np.linalg.norm(fb), 1e-30))


def _tree_norm(tree) -> float:
    import equinox as eqx
    import jax

    leaves = jax.tree_util.tree_leaves(eqx.filter(tree, eqx.is_inexact_array))
    if not leaves:
        return 0.0
    return float(np.sqrt(sum(float(np.sum(np.asarray(x) ** 2)) for x in leaves)))


# ==========================================================================
# main
# ==========================================================================
def main(argv: Optional[list] = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--part", default="a", choices=["a", "e2e", "stage0", "b", "all"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n-rep", dest="n_rep", type=int, default=6,
                    help="repetitions per (item, competitor, ambiguity) cell")
    ap.add_argument("--family", default="deepsets", choices=["deepsets", "attention"])
    ap.add_argument("--force-b", action="store_true",
                    help="run Part B even if Stage 0's gate did not pass "
                         "(the result is labelled gate_passed=False)")
    args = ap.parse_args(argv)

    out: Dict[str, Any] = {"part": args.part, "seed": args.seed, "quick": args.quick}
    t0 = time.time()
    if args.part in ("a", "all"):
        out["A"] = part_a(seed=args.seed, quick=args.quick)
        print(json.dumps(_jsonable(out["A"]), indent=2)[:4000])
    if args.part in ("e2e", "all"):
        out["E2E"] = part_e2e(seed=args.seed, quick=args.quick)
        print(json.dumps(_jsonable({k: v for k, v in out["E2E"].items()
                                    if k != "training"}), indent=2))
        for m, v in out["E2E"]["training"].items():
            print(f"  {m:16s} median step {v['median_step_s']:.3f} s  "
                  f"compile {v['compile_s']:.1f} s  loss {v['loss_first']:.4f} "
                  f"-> {v['loss_last']:.4f}")
    if args.part in ("stage0", "all"):
        out["STAGE0"] = part_stage0(seed=args.seed, quick=args.quick,
                                    n_rep=args.n_rep)
        g = out["STAGE0"]["gate"]
        print(f"\n[STAGE 0] gate passed = {g['passed']}  "
              f"({g['n_qualifying_cells']} qualifying cells)")
        print(json.dumps(_jsonable(g["best_ambiguous_cell"]), indent=2))
    if args.part in ("b", "all"):
        gp = out.get("STAGE0", {}).get("gate", {}).get("passed")
        if gp is False and not args.force_b:
            print("\n[PART B] ⛔ NOT RUN — Stage 0's gate did not pass. A flat "
                  "ablation over a dead axis is monitor #10 firing, not a result. "
                  "Use --force-b to run it anyway (it will be labelled).")
        else:
            out["B"] = part_b(seed=args.seed, quick=args.quick, family=args.family,
                              gate_passed=gp)
            for r in out["B"]["rows"]:
                print(f"  stride {r['stride']:2d} {r['target']:20s} "
                      f"point {r['decode_settled_point']:.4f} "
                      f"endpoints {r['decode_endpoints']:.4f} "
                      f"traj {r['decode_trajectory']:.4f}  "
                      f"ablation {r['ablation_traj_minus_point']:+.4f} "
                      f"(params {r['params']})")
            for lr_ in out["B"]["trajectory_launder"]:
                print(f"  LAUNDER stride {lr_['stride']:2d} full "
                      f"{lr_['decode_full']:.4f} q0_only {lr_['decode_q0_only']:.4f} "
                      f"endpoints {lr_['decode_endpoints']:.4f} blank "
                      f"{lr_['decode_blank_store']:.4f} (bar {lr_['bar']:.4f}, "
                      f"leak={lr_['blank_leak']})")
    out["wall_s"] = time.time() - t0
    path = _save(f"exp_trajectory_read_{args.part}_seed{args.seed}.json", out)
    print(f"\n[exp_trajectory_read] wrote {path}  ({out['wall_s']:.1f} s)")
    return out


if __name__ == "__main__":
    main()
