"""Experiment RETRY-COMPUTE: the accuracy-vs-compute curve for CLU retrieval,
done properly (w23 — the novel-property flagship).

⭐ **What this promotes.** w22's `exp_hopfield_capacity` retry was a SINGLE rung
(one boosted re-relaxation on the low-confidence half, +46.9pp at ×1.5 compute).
A referee asks three questions that a single point cannot answer:

  1. is the *curve* real — multiple compute budgets, monotone?
  2. is it the *physics* (the Lorentz-boost-style directed re-launch) or would any
     stochastic-restart heuristic match it?
  3. can a feedforward memory given the SAME extra compute draw the same curve?

This experiment turns the one demo point into a defensible **accuracy-vs-compute
curve with five pre-registered controls**, on the w22 retrieval protocol (Gaussian
noise queries, matched to ``MAGICS-LAB/UHop`` ``memory_retrieval_noise.py``), at
≥2 load levels (M) and ≥2 noise levels (σ). The honest compute unit is the
**relaxation-step count**; wall-clock is secondary.

**The six lines (Item 1 + Item 2 controls):**

- ``clu_gated``      — CLU retry: k boosted re-relaxation rounds, **confidence
                       gated** (retry only reads below a cosine threshold, re-gated
                       each round). The headline: adaptive spend.
- ``ungated_all``    — the SAME boost applied to ALL reads (no gate). Quantifies the
                       gate's contribution (w22: −38pp blank guard says gating is
                       load-bearing).
- ``ensemble``       — k+1 independent CLU settles from the query with random start
                       momenta, keep the best-confidence read. The fair "is it the
                       boost or just k tries?" rival.
- ``random_kick``    — CLU retry with the directed boost replaced by an
                       **equal-energy random** perturbation. Is the boost doing
                       anything a kick doesn't? (N1: test this honestly.)
- ``feedforward_nn`` — the trivial NN baseline given the same budget: k+1 augmented
                       nearest-neighbour reads, majority-vote the identity (TTA).
                       CM-23's "a curve feedforward memories cannot draw" survives
                       ONLY if this is flat-or-worse.
- ``hopfield_ksteps``— the closed-form modern-Hopfield line iterated k+1 steps.

⚠ Compute-axis honesty: CLU methods are placed at their MEASURED relaxation-step
multiplier (gated/kick are **sub-linear** in k — the adaptive-compute advantage).
The feedforward-NN and Hopfield lines are placed at their matched *budget*
multiplier (k+1); mapping one NN/Hopfield read to one CLU settle is **generous to
the baselines** and is stated in the report.

Runnable: ``uv run python -m chlu.experiments.exp_retry_compute --quick`` or via the
CLI ``chlu exp-retry-compute [--project N] [--seed I] [--quick]``.
"""

import json
import os
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from chlu.config import CHLUConfig, get_default_config
from chlu.experiments.exp_hopfield_capacity import (
    _cosine,
    _median_nn_distance,
    build_clu_memory,
    dropout_query,
    hopfield_retrieve,
    load_patterns,
    noise_query,
)
from chlu.experiments.exp_hopfield_capacity import _softmax as _softmax_act


# ---------------------------------------------------------------------------
# CLU settle with an explicit per-query initial momentum (x64-safe, memory-safe)
# ---------------------------------------------------------------------------


def _settle(model, Q0, P0, steps, dt, gamma, tail_steps, chunk):
    """Damped Verlet from each (q0, p0); read = mean q over the last tail_steps.

    Generalizes ``exp_hopfield_capacity._settle_read`` to accept a launch
    momentum (needed for the boost / kick / ensemble start). x64-safe: launch in
    the ambient float dtype so the lax.scan carry types agree (the w22 landmine).
    NaN-guard: any query whose rollout diverged reads back its launch point.
    """
    tail_steps = int(max(1, tail_steps))
    settle_steps = int(max(1, steps - tail_steps))
    fdtype = jnp.result_type(float)
    Q0 = jnp.asarray(Q0, dtype=fdtype)
    P0 = jnp.asarray(P0, dtype=fdtype)

    def per_query(q0, p0):
        def step_fn(state, _):
            return model.step(state, dt, gamma), None

        (q1, p1), _ = jax.lax.scan(step_fn, (q0, p0), None, length=settle_steps)
        traj = model(q1, p1, tail_steps, dt, gamma)
        dim = q0.shape[0]
        read = jnp.mean(traj[:, :dim], axis=0)
        bad = jnp.any(~jnp.isfinite(read))
        return jnp.where(bad, q0, read)

    f = eqx.filter_jit(jax.vmap(per_query))
    outs = []
    for i in range(0, Q0.shape[0], chunk):
        outs.append(np.asarray(f(Q0[i : i + chunk], P0[i : i + chunk])))
    return np.concatenate(outs, axis=0)


# ---------------------------------------------------------------------------
# Label-free confidence + accuracy
# ---------------------------------------------------------------------------


def _confidence_and_nn(reads, patterns):
    """Return (cosine-to-nearest-well, nearest-well-index). Confidence is
    LABEL-FREE (cosine to the retrieved pattern, not to the truth)."""
    R = np.asarray(reads)
    P = np.asarray(patterns)
    d2 = np.sum((R[:, None, :] - P[None, :, :]) ** 2, axis=-1)  # (Nq, M)
    nn = np.argmin(d2, axis=1)
    cos = _cosine(R, P[nn])
    return np.asarray(cos), nn


def _acc(nn, true_idx):
    return float(np.mean(np.asarray(nn) == np.asarray(true_idx)))


def _dt_of(cfg, s):
    return cfg.clu_dt if cfg.clu_dt > 0 else 0.5 * s / np.sqrt(cfg.clu_b)


# ---------------------------------------------------------------------------
# CLU-gated / ungated / random-kick retry ladder (directed boost vs kick vs all)
# ---------------------------------------------------------------------------


def _retry_ladder(
    model, Q, reads0, patterns, true_idx, cfg, dt, threshold, mode, rng
):
    """Confidence-gated, lock-on-retry boosted re-relaxation ladder.

    Returns {k: (identity_acc, compute_mult)} for each k in ``cfg.retry_ladder``.

    Per round we re-relax the ``retry_step_frac`` lowest-confidence *eligible*
    reads with a directed boost (or, in "kick" mode, an equal-energy random
    perturbation) launched from the current settled point toward the query, and
    replace them **unconditionally**. Eligibility = ``(not yet retried) and
    (confidence < threshold)``; a retried read is **locked** so it is never
    retried again. Confidence = cosine to the nearest stored well.

    ⚠ **Why this design (measured, not assumed).** After a damped settle every
    particle sits *in some well*, so cosine-to-nearest is high (≈0.95–1.0) whether
    the well is right or wrong — it is a good *ranking* signal (correct 0.998 vs
    wrong 0.949) but a useless *acceptance* signal (a boost into the RIGHT well
    does not raise cosine above the wrong well it left). So no per-item accept rule
    works; the GATE (retry the low-confidence, wrong-enriched tail) is the
    load-bearing element, exactly w22's finding (−38pp blank guard). Re-gating the
    same reads every round without locking OSCILLATES (boosting an already-correct
    read corrupts it); locking makes the ladder monotone, and the threshold makes
    the gate **auto-stop** spending compute once the low-confidence tail is
    exhausted — the honest adaptive-compute signature.

    mode:
      "gated"   — the eligibility gate above (directed boost). Sub-linear compute.
      "ungated" — retry ALL reads every round, no lock, no threshold (the no-gate
                  control; compute (k+1)×). Quantifies the gate's contribution.
      "kick"    — the SAME gate/lock as "gated" but the directed boost is replaced
                  by an equal-energy random-direction kick (the mechanism control).

    ⚠ JIT-shape-stability: each round settles the FULL population so the vmap batch
    dim is constant (one compile per (M, s)); updates and the compute count are
    restricted to the retried subset. The reported compute (relaxation steps) is
    the honest retried count, not the full-population wall-clock."""
    S = cfg.clu_steps
    tail = int(max(1, cfg.clu_tail_frac * S))
    chunk = cfg.rollout_chunk
    Nq = Q.shape[0]
    base = Nq * S
    step_n = max(1, int(round(cfg.retry_step_frac * Nq)))

    reads = np.array(reads0, dtype=float)
    locked = np.zeros(Nq, dtype=bool)
    total_steps = base
    out = {0: (_acc(_confidence_and_nn(reads, patterns)[1], true_idx), 1.0)}
    max_k = max(cfg.retry_ladder)
    Qn = np.asarray(Q, dtype=float)

    for j in range(1, max_k + 1):
        cos, _ = _confidence_and_nn(reads, patterns)
        if mode == "ungated":
            cand = np.arange(Nq)  # no gate, no lock
        else:
            eligible = (~locked) & (cos < threshold)
            cm = np.where(eligible, cos, np.inf)
            cand = np.argsort(cm)[:step_n]
            cand = cand[np.isfinite(cm[cand])]  # drop padding when eligible < step_n

        if len(cand) > 0:
            direction = Qn - reads  # full population (constant batch dim)
            if mode == "kick":
                mag = cfg.retry_boost * np.linalg.norm(direction, axis=1, keepdims=True)
                d = rng.normal(size=reads.shape)
                d = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
                p0 = mag * d
            else:  # gated / ungated — directed boost toward the query
                p0 = cfg.retry_boost * direction
            new = _settle(model, reads, p0, S, dt, cfg.clu_gamma, tail, chunk)
            reads[cand] = new[cand]  # unconditional within the retried set
            if mode != "ungated":
                locked[cand] = True
            total_steps += len(cand) * S  # honest: only the retried subset charged

        if j in cfg.retry_ladder:
            _, nn_j = _confidence_and_nn(reads, patterns)
            out[j] = (_acc(nn_j, true_idx), total_steps / base)
    return out


# ---------------------------------------------------------------------------
# Ensemble-of-k-reads (k independent random starts, keep best confidence)
# ---------------------------------------------------------------------------


def _ensemble_ladder(model, Q, reads0, patterns, true_idx, cfg, dt, rng):
    """k+1 independent CLU settles from the query with random launch momenta
    (energy-matched to the boost), keep the best-confidence read. Compute (k+1)×."""
    S = cfg.clu_steps
    tail = int(max(1, cfg.clu_tail_frac * S))
    chunk = cfg.rollout_chunk
    Qn = np.asarray(Q, dtype=float)

    best = np.array(reads0, dtype=float)
    best_cos, best_nn = _confidence_and_nn(best, patterns)
    # per-query energy scale matched to the round-1 boost magnitude
    scale = cfg.retry_boost * np.linalg.norm(Qn - best, axis=1, keepdims=True)

    out = {0: (_acc(best_nn, true_idx), 1.0)}
    max_k = max(cfg.retry_ladder)
    for j in range(1, max_k + 1):
        d = rng.normal(size=Qn.shape)
        d = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
        p0 = scale * d
        r = _settle(model, Qn, p0, S, dt, cfg.clu_gamma, tail, chunk)
        r_cos, _ = _confidence_and_nn(r, patterns)
        take = r_cos > best_cos
        best[take] = r[take]
        best_cos[take] = r_cos[take]
        if j in cfg.retry_ladder:
            _, nn_j = _confidence_and_nn(best, patterns)
            out[j] = (_acc(nn_j, true_idx), float(j + 1))
    return out


# ---------------------------------------------------------------------------
# Feedforward-NN matched compute (TTA augmentation + majority vote)
# ---------------------------------------------------------------------------


def _majority(votes):
    """Majority vote over axis 0 of an (R, Nq) int array."""
    R, Nq = votes.shape
    out = np.empty(Nq, dtype=int)
    for i in range(Nq):
        vals, counts = np.unique(votes[:, i], return_counts=True)
        out[i] = vals[np.argmax(counts)]
    return out


def _feedforward_ladder(Q, patterns, true_idx, cfg, rng):
    """The trivial feedforward memory given the same budget: k+1 nearest-neighbour
    reads over test-time-augmented queries, majority-vote the identity. Placed at
    the matched budget multiplier (k+1). One NN read ≪ one CLU settle — GENEROUS to
    the baseline (stated in the report)."""
    P = np.asarray(patterns, dtype=float)
    Qn = np.asarray(Q, dtype=float)

    def nn_idx(q):
        d2 = np.sum((q[:, None, :] - P[None, :, :]) ** 2, axis=-1)
        return np.argmin(d2, axis=1)

    votes = [nn_idx(Qn)]
    out = {0: (_acc(votes[0], true_idx), 1.0)}
    max_k = max(cfg.retry_ladder)
    for j in range(1, max_k + 1):
        aug = np.clip(np.abs(Qn + rng.normal(size=Qn.shape) * cfg.ff_aug_sigma), 0, 1)
        votes.append(nn_idx(aug))
        if j in cfg.retry_ladder:
            maj = _majority(np.stack(votes, axis=0))
            out[j] = (_acc(maj, true_idx), float(j + 1))
    return out


# ---------------------------------------------------------------------------
# Hopfield-k-steps (the closed-form line iterated)
# ---------------------------------------------------------------------------


def _hopfield_ladder(Q, patterns, true_idx, cfg):
    """Modern-Hopfield update iterated k+1 steps. beta = auto-sharpened rule
    (β·⟨x,x⟩≈200, floored at the repo β), matching exp_hopfield_capacity."""
    P = np.asarray(patterns, dtype=float)
    self_overlap = float(np.mean(np.sum(P * P, axis=-1)))
    beta = (
        cfg.hopfield_beta_tuned
        if cfg.hopfield_beta_tuned > 0
        else max(cfg.hopfield_beta, 200.0 / (self_overlap + 1e-9))
    )
    out = {}
    for k in cfg.retry_ladder:
        xhat = hopfield_retrieve(patterns, Q, beta, 1.0, k + 1, _softmax_act)
        X = np.asarray(xhat)
        d2 = np.sum((X[:, None, :] - P[None, :, :]) ** 2, axis=-1)
        nn = np.argmin(d2, axis=1)
        out[k] = (_acc(nn, true_idx), float(k + 1))
    return out


# ---------------------------------------------------------------------------
# One (M, σ) cell — all six lines
# ---------------------------------------------------------------------------


def make_query(patterns, query_type, level, key):
    """mask -> torch.dropout(level); noise -> clamp(|x+N(0,level)|,0,1)."""
    if query_type == "mask":
        return dropout_query(patterns, level, key)
    if query_type == "noise":
        return noise_query(patterns, level, key)
    raise ValueError(f"unknown query_type {query_type!r}")


#: deterministic per-protocol seed offset (Python's ``hash`` is process-salted, so
#: it must NOT be used for reproducible PRNG seeding — handover reproducibility rule)
_QT_OFFSET = {"mask": 101, "noise": 202}


def run_cell(cfg, patterns, query_type, level, seed):
    M = patterns.shape[0]
    true_idx = np.arange(M)
    qt_off = _QT_OFFSET.get(query_type, 0)
    key = jax.random.PRNGKey(seed + int(1000 * level) + M + qt_off)
    key, kq = jax.random.split(key)
    Q = make_query(patterns, query_type, level, kq)

    s = cfg.clu_s_frac * _median_nn_distance(patterns)
    dt = _dt_of(cfg, s)
    model = build_clu_memory(patterns, s, cfg)

    # first pass (p0 = 0) — shared by every CLU method
    S = cfg.clu_steps
    tail = int(max(1, cfg.clu_tail_frac * S))
    reads0 = _settle(
        model,
        np.asarray(Q),
        np.zeros_like(np.asarray(Q)),
        S,
        dt,
        cfg.clu_gamma,
        tail,
        cfg.rollout_chunk,
    )
    first_acc = _acc(_confidence_and_nn(reads0, patterns)[1], true_idx)

    rng = np.random.default_rng(seed + M + int(1000 * level) + qt_off)
    thr = cfg.main_threshold
    lines = {
        "clu_gated": _retry_ladder(
            model, Q, reads0, patterns, true_idx, cfg, dt, thr, "gated", rng
        ),
        "ungated_all": _retry_ladder(
            model, Q, reads0, patterns, true_idx, cfg, dt, thr, "ungated", rng
        ),
        "random_kick": _retry_ladder(
            model, Q, reads0, patterns, true_idx, cfg, dt, thr, "kick", rng
        ),
        "ensemble": _ensemble_ladder(
            model, Q, reads0, patterns, true_idx, cfg, dt, rng
        ),
        "feedforward_nn": _feedforward_ladder(Q, patterns, true_idx, cfg, rng),
        "hopfield_ksteps": _hopfield_ladder(Q, patterns, true_idx, cfg),
    }

    # threshold sweep (gated only), reported at the full ladder
    thr_sweep = {}
    for t in cfg.conf_thresholds:
        thr_sweep[f"{t:.2f}"] = _retry_ladder(
            model, Q, reads0, patterns, true_idx, cfg, dt, t, "gated", rng
        )

    return {
        "M": int(M),
        "query_type": query_type,
        "level": float(level),
        "s": float(s),
        "dt": float(dt),
        "first_pass_acc": first_acc,
        "lines": {
            name: {str(k): {"acc": v[0], "compute_mult": v[1]} for k, v in d.items()}
            for name, d in lines.items()
        },
        "threshold_sweep": {
            t: {str(k): {"acc": v[0], "compute_mult": v[1]} for k, v in d.items()}
            for t, d in thr_sweep.items()
        },
    }


# ---------------------------------------------------------------------------
# Figure — grid of (M, σ) cells, six lines each
# ---------------------------------------------------------------------------


LINE_STYLE = {
    "clu_gated": ("o-", "#d62728", 2.0),
    "ungated_all": ("s--", "#ff7f0e", 1.2),
    "ensemble": ("^--", "#1f77b4", 1.2),
    "random_kick": ("v--", "#9467bd", 1.2),
    "feedforward_nn": ("D:", "#2ca02c", 1.2),
    "hopfield_ksteps": ("x:", "#7f7f7f", 1.2),
}


def _plot_grid(cells, loads, levels, dataset, query_type, save_dir):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    nrow, ncol = len(loads), len(levels)
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(4.2 * ncol, 3.4 * nrow), squeeze=False
    )
    lvl_name = "mask p" if query_type == "mask" else "σ"
    by_key = {(c["M"], round(c["level"], 4)): c for c in cells}
    for i, M in enumerate(loads):
        for jx, lv in enumerate(levels):
            ax = axes[i][jx]
            c = by_key.get((M, round(lv, 4)))
            if c is None:
                ax.axis("off")
                continue
            for name, (mk, col, lw) in LINE_STYLE.items():
                d = c["lines"][name]
                ks = sorted(int(k) for k in d)
                xs = [d[str(k)]["compute_mult"] for k in ks]
                ys = [d[str(k)]["acc"] for k in ks]
                ax.plot(xs, ys, mk, color=col, lw=lw, ms=4, label=name)
            ax.set_title(f"M={M}, {lvl_name}={lv}", fontsize=9)
            ax.set_xlabel("compute (× first-pass budget)")
            ax.set_ylabel("identity accuracy")
            ax.grid(alpha=0.25)
            if i == 0 and jx == 0:
                ax.legend(fontsize=6)
    fig.suptitle(
        f"Accuracy vs test-time compute — {dataset} / {query_type} query "
        "(CLU-gated retry + 5 controls)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = os.path.join(save_dir, f"retry_compute_grid_{dataset}_{query_type}.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return [p]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_experiment_retry_compute(
    config: Optional[CHLUConfig] = None,
    save_dir: str = "results",
    models_dir: Optional[str] = None,
    seed: Optional[int] = None,
):
    config = config or get_default_config()
    cfg = config.experiment_retry_compute
    seed = cfg.seed if seed is None else seed
    os.makedirs(save_dir, exist_ok=True)

    results = {
        "seed": seed,
        "designed_not_learned": True,
        "protocol": {
            "query": "gaussian noise clamp(|x+N(0,sigma)|,0,1) (UHop noise harness)",
            "compute_unit": "relaxation steps / (Nq*clu_steps) first-pass budget",
            "note": "feedforward_nn and hopfield_ksteps placed at matched budget "
            "multiplier (k+1); one NN/Hopfield read << one CLU settle "
            "(generous to the baselines).",
        },
        "config": {
            k: getattr(cfg, k)
            for k in (
                "datasets",
                "load_grid",
                "noise_levels",
                "query_types",
                "mask_fracs",
                "retry_ladder",
                "retry_step_frac",
                "conf_thresholds",
                "main_threshold",
                "retry_boost",
                "clu_s_frac",
                "clu_b",
                "clu_alpha",
                "clu_gamma",
                "clu_steps",
                "clu_kinetic_mode",
                "ff_aug_sigma",
                "n_data_pool",
            )
        },
    }

    all_cells = []
    figs = []
    for ds in cfg.datasets:
        try:
            pool = load_patterns(ds, cfg.n_data_pool, seed)
        except FileNotFoundError as e:
            all_cells.append({"dataset": ds, "skipped": str(e)})
            continue
        loads = [M for M in cfg.load_grid if M <= pool.shape[0]]
        for qt in cfg.query_types:
            levels = cfg.mask_fracs if qt == "mask" else cfg.noise_levels
            cells = []
            for M in loads:
                patterns = pool[:M]
                for lv in levels:
                    cell = run_cell(cfg, patterns, qt, lv, seed)
                    cell["dataset"] = ds
                    cells.append(cell)
            all_cells.extend(cells)
            figs += _plot_grid(cells, loads, levels, ds, qt, save_dir)

    results["cells"] = all_cells
    results["figures"] = figs

    # verdict summary: per cell, best gated accuracy over the ladder and the gap of
    # each control's best to it (the honest "does gated dominate at equal compute?"
    # is in the figure; this is the summary scalar).
    verdict = []
    for c in all_cells:
        if "lines" not in c:
            continue
        gated = c["lines"]["clu_gated"]
        g_best = max(v["acc"] for v in gated.values())
        g_at = {k: v for k, v in gated.items()}
        row = {
            "M": c["M"],
            "query_type": c["query_type"],
            "level": c["level"],
            "first_pass_acc": c["first_pass_acc"],
            "gated_best_acc": round(g_best, 4),
            "gated_lift_pp": round(100.0 * (g_best - c["first_pass_acc"]), 2),
            "gated_compute_at_best": round(
                min(v["compute_mult"] for v in g_at.values() if v["acc"] == g_best), 3
            ),
        }
        for name in c["lines"]:
            if name == "clu_gated":
                continue
            best = max(v["acc"] for v in c["lines"][name].values())
            row[f"{name}_best_acc"] = round(best, 4)
            row[f"{name}_gap_pp"] = round(100.0 * (g_best - best), 2)
        verdict.append(row)
    results["verdict"] = verdict

    results_dir = os.path.join(os.path.dirname(os.path.abspath(save_dir)), "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "exp_retry_compute_metrics.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    results["metrics_path"] = out_path
    return results


def apply_quick(config: CHLUConfig) -> None:
    """Quick smoke settings — same code path, smaller sweeps."""
    cfg = config.experiment_retry_compute
    cfg.load_grid = [32, 64]
    cfg.noise_levels = [0.3]
    cfg.mask_fracs = [0.5]
    cfg.query_types = ["mask", "noise"]
    cfg.retry_ladder = [0, 1, 2]
    cfg.conf_thresholds = [0.97, 1.0]
    cfg.clu_steps = 40
    cfg.n_data_pool = 200
    cfg.rollout_chunk = 64


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Retry-compute: accuracy-vs-compute curve, CLU-gated + 5 controls"
    )
    parser.add_argument("--project", help="Project name to use (default: ./results)")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--quick", action="store_true", help="Quick smoke mode")
    parser.add_argument("--dataset", help="Override datasets (comma-separated)")
    args = parser.parse_args()

    if args.project:
        from chlu.project import ProjectManager

        pm = ProjectManager()
        config = pm.load(args.project)
        paths = pm.get_paths(args.project)
        save_dir = str(paths["plots"])
    else:
        config = get_default_config()
        save_dir = "results"
        os.makedirs(save_dir, exist_ok=True)

    if args.quick:
        apply_quick(config)
    if args.dataset:
        config.experiment_retry_compute.datasets = args.dataset.split(",")

    res = run_experiment_retry_compute(config=config, save_dir=save_dir, seed=args.seed)
    print(
        json.dumps(
            {"verdict": res["verdict"], "figures": res["figures"]},
            indent=2,
        )[:8000]
    )


if __name__ == "__main__":
    main()
