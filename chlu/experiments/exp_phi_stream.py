"""Experiment PHI-STREAM: what may the learned read-in ``φ`` see in a CONTINUAL
stream? (w24 — the blocker that de-risks the w25 continual-learning entry.)

w23's ``φ`` (:mod:`chlu.experiments.exp_phi_read_in`) was fit on a disjoint pool
that saw **all ten classes**. For static recall that is fair. In **Class-IL it is
data leakage**: ``φ`` must not be trained on data from tasks the model has not yet
reached — a referee kills the entry in one line otherwise.

⭐ **Head ruling (binding), implemented here as three regimes behind one flag:**
  - ``task1_only`` — **PRIMARY.** ``φ`` is fit on **task 1's classes only** and then
    frozen for the whole stream. The defensible arm; every headline number comes
    from here.
  - ``generic_frozen`` — the w23-style all-classes pool, carried as a **declared
    upper bound**. Clearly labelled; **never quoted as the headline**.
  - ``online`` — **stub + interface only, NOT run** (:class:`OnlineReadIn`). Its own
    experiment later; this module leaves a clean extension point and nothing more.

**Unchanged laws.** The store stays **DESIGNED** (Gaussian wells over ``φ(x)``,
payload = the raw ``x``) and ``φ`` is **never trained through the store** (the w20
law). ``kNN-in-φ`` is reported in **every** regime — the mandatory laundering
control (N89, CM-22(i)).

**⭐ The deliverable (Item 2) — the cost-of-strictness curve.** Over a
Split-MNIST-shaped stream (5 tasks × 2 classes), with the **same store and the same
queries** in both regimes, report identity-retrieval and downstream (class-label)
accuracy **per task index**, and how the ``generic_frozen − task1_only`` gap grows
as tasks accumulate. A steeply widening gap means a task-1 ``φ`` cannot represent
later classes and the w25 entry needs online ``φ``.

**⚠ Standing scope caveat (Head, binding).** Masked/static retrieval is a task where
equalling a simple baseline is our best case, because CLU *approximates* the
nearest-neighbour method that wins it. Every retrieval number here is **diagnostic
of φ's stream discipline**, not a competitive claim; masked recall is permanently
appendix-only.

Runnable: ``uv run python -m chlu.experiments.exp_phi_stream --quick`` or via the CLI
``chlu exp-phi-stream [--project N] [--seed I] [--quick] [--regimes …] [--arms …]``.
"""

import json
import os
from typing import Optional

import jax
import numpy as np

from chlu.config import CHLUConfig, get_default_config
from chlu.experiments.exp_hopfield_capacity import (
    RAMSAUER_COMMIT,
    UHOP_COMMIT,
    _median_nn_distance,
    dropout_query,
    score_retrieval,
)
from chlu.experiments.exp_phi_read_in import (
    build_read_in,
    clu_in_phi,
    knn_in_phi,
)

#: The three regimes. ``online`` is declared but deliberately NOT runnable here.
PHI_REGIMES = ("task1_only", "generic_frozen", "online")

REGIME_ROLE = {
    "task1_only": "PRIMARY — the defensible arm; all headline numbers",
    "generic_frozen": "REFERENCE — declared upper bound; never the headline",
    "online": "NOT RUN — stub/interface only (its own experiment later)",
}

ONLINE_STUB_NOTE = (
    "online φ is a deliberate extension point, not implemented in w24 (Head "
    "ruling): φ would be refit/updated as each task arrives, using only data "
    "already seen by the stream. Implementing it means (a) a per-task update "
    "hook (OnlineReadIn.observe_task) called BEFORE the task's items are written, "
    "(b) re-keying every already-stored well through the new φ (or accepting a "
    "stale-address penalty), and (c) a decision about whether re-keying counts as "
    "replay. Until that is settled it must not enter the CL results."
)


# ---------------------------------------------------------------------------
# Data — a Split-MNIST-shaped class-incremental stream with labels
# ---------------------------------------------------------------------------


def load_labeled_images(dataset: str, split: str = "all"):
    """Return ``(X, y)`` — ``(N, D)`` float32 images in ``[0,1]`` and int labels.

    Labels are required here (unlike w23's ``load_patterns``) because the stream is
    **class**-incremental and the downstream metric is a class read-out.

    Args:
        dataset: ``"mnist"`` (784-d) or ``"cifar10"`` (3072-d, w25 — the hard rung
            of the continual-learning entry).
        split: ``"all"`` (default, the w24 behaviour — the whole labelled set),
            ``"train"`` or ``"test"``. The canonical splits are MNIST 60k/10k and
            CIFAR-10 50k/10k; a Class-IL entry must evaluate on held-out test data,
            so the split argument exists for w25 and changes nothing for w24 callers.
    """
    if split not in ("all", "train", "test"):
        raise ValueError(f"split must be all|train|test, got {split!r}")
    if dataset == "mnist":
        from sklearn.datasets import fetch_openml

        mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
        X = np.asarray(mnist.data, dtype=np.float32) / 255.0
        y = np.asarray(mnist.target).astype(int)
        n_train = 60000
    elif dataset == "cifar10":
        X, y, n_train = _load_cifar10_labeled()
    else:
        raise ValueError(
            f"labelled stream data is wired for 'mnist'|'cifar10' (got {dataset!r})"
        )
    if split == "train":
        return X[:n_train], y[:n_train]
    if split == "test":
        return X[n_train:], y[n_train:]
    return X, y


def _load_cifar10_labeled():
    """CIFAR-10 **with labels**, ``(X, y, n_train)``, train batches then test batch.

    Reads the canonical python tarball (the same file
    :func:`chlu.experiments.exp_hopfield_capacity._load_cifar10` uses; openml's
    copy is checksum-blocked on this machine). w24's loader dropped the labels and
    read only one batch — a Class-IL stream needs both.
    """
    import pickle
    import tarfile

    here = os.path.dirname(os.path.abspath(__file__))
    scratch = os.path.join(
        here, "..", "..", ".claude", "scratch", "hopfield-capacity-benchmark"
    )
    candidates = [
        p
        for p in (
            os.environ.get("CHLU_CIFAR10_TARBALL"),  # explicit override (worktrees:
            # ``.claude/`` is gitignored, so a worktree has no scratch dir of its own)
            os.path.join(scratch, "cifar10.tar.gz"),
            os.path.expanduser("~/cifar-10-python.tar.gz"),
            "cifar-10-python.tar.gz",
        )
        if p
    ]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        raise FileNotFoundError(
            "CIFAR-10 not available locally. Place cifar-10-python.tar.gz in "
            ".claude/scratch/hopfield-capacity-benchmark/ to enable the CIFAR arm."
        )
    # decoding the 170 MB gzip takes minutes; cache the decoded arrays next to it
    cache = os.path.join(os.path.dirname(os.path.abspath(path)), "cifar10_labeled.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        return z["X"], z["y"], int(z["n_train"])

    Xtr, ytr, Xte, yte = [], [], [], []
    with tarfile.open(path, "r:gz") as tar:
        members = {m.name.split("/")[-1]: m for m in tar.getmembers()}
        for i in range(1, 6):
            name = f"data_batch_{i}"
            if name not in members:
                continue
            d = pickle.load(tar.extractfile(members[name]), encoding="bytes")
            Xtr.append(np.asarray(d[b"data"], dtype=np.float32) / 255.0)
            ytr.append(np.asarray(d[b"labels"], dtype=int))
        if "test_batch" in members:
            d = pickle.load(tar.extractfile(members["test_batch"]), encoding="bytes")
            Xte.append(np.asarray(d[b"data"], dtype=np.float32) / 255.0)
            yte.append(np.asarray(d[b"labels"], dtype=int))
    if not Xtr:
        raise FileNotFoundError(f"no CIFAR-10 data batches found in {path}")
    Xtr_a, ytr_a = np.concatenate(Xtr), np.concatenate(ytr)
    if Xte:
        Xte_a, yte_a = np.concatenate(Xte), np.concatenate(yte)
    else:  # degenerate archive: fall back to a tail slice as the test split
        Xte_a, yte_a = Xtr_a[-10000:], ytr_a[-10000:]
        Xtr_a, ytr_a = Xtr_a[:-10000], ytr_a[:-10000]
    X = np.concatenate([Xtr_a, Xte_a])
    y = np.concatenate([ytr_a, yte_a])
    n_train = int(len(Xtr_a))
    try:
        np.savez(cache, X=X, y=y, n_train=n_train)
    except OSError:
        pass  # read-only location: just pay the decode cost next time
    return X, y, n_train


def task_classes(cfg, t: int):
    """Classes of task ``t`` (0-indexed): Split-MNIST = {2t, 2t+1}."""
    c = cfg.classes_per_task
    return list(range(t * c, (t + 1) * c))


def build_stream(cfg, seed: int, data=None):
    """Build the class-incremental stream.

    ``data`` may be an explicit ``(X, y)`` pair (used by the tests to run the whole
    pipeline on tiny synthetic labelled data); otherwise it is loaded from
    ``cfg.dataset``.

    Disjointness (the fairness guarantee): a single permutation is split into a
    **store region** (the only source of stored items) and a **fit region** (the
    only source of ``φ`` fit pools), so no ``φ`` — in either regime — is ever fit on
    a stored pattern. The ``task1_only`` fit pool is additionally restricted to
    task-1 classes; both pools have the SAME size so the regimes differ only in
    *which classes* ``φ`` may see.

    Queries (50 %-masked, repo-verbatim dropout) are generated ONCE per stored item
    here, so both regimes and every stream position see **identical queries**.
    """
    X, y = load_labeled_images(cfg.dataset) if data is None else data
    X = np.asarray(X, np.float32)
    y = np.asarray(y).astype(int)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    n_store_region = min(cfg.n_store_region, len(X) // 2)
    store_region, fit_region = perm[:n_store_region], perm[n_store_region:]

    key = jax.random.PRNGKey(seed + 991)
    patterns, labels, queries, classes = [], [], [], []
    for t in range(cfg.n_tasks):
        cls = task_classes(cfg, t)
        pool = store_region[np.isin(y[store_region], cls)][: cfg.items_per_task]
        Xt = X[pool]
        key, kq = jax.random.split(key)
        patterns.append(Xt)
        labels.append(y[pool])
        queries.append(np.asarray(dropout_query(Xt, cfg.mask_p, kq)))
        classes.append(cls)

    fit_generic = X[fit_region[: cfg.n_fit_pool]]
    t1 = fit_region[np.isin(y[fit_region], classes[0])][: cfg.n_fit_pool]
    fit_task1 = X[t1]
    return {
        "task_patterns": patterns,
        "task_labels": labels,
        "task_queries": queries,
        "task_classes": classes,
        "fit_pool_generic": fit_generic,
        "fit_pool_task1": fit_task1,
        "dim": int(X.shape[1]),
        "n_store_region": int(n_store_region),
    }


# ---------------------------------------------------------------------------
# The three φ-stream regimes (one flag) — Item 1
# ---------------------------------------------------------------------------


class OnlineReadIn:
    """⚠ **Stub — the declared extension point for an ONLINE ``φ`` (NOT run).**

    Constructing this object is allowed so the interface is inspectable and
    testable; every operation raises :class:`NotImplementedError`. A real online
    ``φ`` would implement :meth:`observe_task` (called with a task's data *before*
    that task's items are written) and re-key the already-stored wells. See
    :data:`ONLINE_STUB_NOTE` for the three open design decisions.
    """

    regime = "online"
    implemented = False

    def __init__(self, cfg=None, **kwargs):
        self.cfg = cfg
        self.kwargs = kwargs

    def observe_task(self, X_task, task_index: int):
        raise NotImplementedError(ONLINE_STUB_NOTE)

    def __call__(self, X):
        raise NotImplementedError(ONLINE_STUB_NOTE)


def fit_pool_for_regime(regime: str, stream, cfg):
    """Return ``(fit_pool, provenance)`` — **exactly what this regime may see**."""
    if regime == "task1_only":
        return stream["fit_pool_task1"], {
            "regime": "task1_only",
            "role": REGIME_ROLE["task1_only"],
            "may_see": f"task-1 classes only {stream['task_classes'][0]}",
            "frozen_from": "end of task 1 (never updated again)",
        }
    if regime == "generic_frozen":
        allc = sorted(c for cl in stream["task_classes"] for c in cl)
        return stream["fit_pool_generic"], {
            "regime": "generic_frozen",
            "role": REGIME_ROLE["generic_frozen"],
            "may_see": f"all stream classes {allc} (LEAKS future tasks)",
            "frozen_from": "before the stream starts (never updated)",
        }
    if regime == "online":
        raise NotImplementedError(ONLINE_STUB_NOTE)
    raise ValueError(f"unknown φ regime {regime!r} (expected one of {PHI_REGIMES})")


def build_stream_read_in(regime: str, arm: str, stream, cfg, seed: int):
    """Build the frozen ``φ`` for one (regime, arm). ``φ`` is fit ONLY on the
    regime's fit pool — never on a stored pattern, never through the store."""
    pool, prov = fit_pool_for_regime(regime, stream, cfg)
    dim_probe = stream["task_patterns"][0]
    phi, arm_prov = build_read_in(arm, cfg.dataset, dim_probe, pool, cfg, seed)
    prov = {**prov, **arm_prov, "n_fit_pool": int(len(pool))}
    return phi, prov


# ---------------------------------------------------------------------------
# One stream run: write task by task, evaluate every task seen so far
# ---------------------------------------------------------------------------


def _metrics(patterns, labels, payload_idx, mask, true_idx, cfg):
    """Retrieval metrics for one task slice. ``identity_acc``/``mean_sqdiff`` are the
    w22/w23 lineage (pixel space, on the returned payload); ``class_acc`` is the
    downstream Class-IL-shaped read-out (label of the retrieved payload)."""
    idx = payload_idx[mask]
    payloads = np.asarray(patterns)[idx]
    m, _, _ = score_retrieval(patterns, payloads, true_idx[mask], cfg.success_cosine)
    m["class_acc"] = float(np.mean(np.asarray(labels)[idx] == np.asarray(labels)[mask]))
    m["n"] = int(mask.sum())
    return m


def run_stream_regime(cfg, stream, regime: str, arm: str, seed: int):
    """Walk the stream for one (regime, arm): after each task is written, retrieve
    every stored item with its (fixed) masked query and score per task index."""
    phi, prov = build_stream_read_in(regime, arm, stream, cfg, seed)
    rows, s_task1 = [], None
    for t in range(1, cfg.n_tasks + 1):
        patterns = np.concatenate(stream["task_patterns"][:t], axis=0)
        labels = np.concatenate(stream["task_labels"][:t], axis=0)
        Q = np.concatenate(stream["task_queries"][:t], axis=0)
        task_of = np.concatenate(
            [np.full(len(p), i) for i, p in enumerate(stream["task_patterns"][:t])]
        )
        true_idx = np.arange(len(patterns))

        keys = np.asarray(phi(patterns))
        s_now = cfg.clu_s_frac * _median_nn_distance(keys)
        if t == 1:
            s_task1 = s_now
        s = s_task1 if cfg.s_policy == "task1_frozen" else s_now
        feat_q = np.asarray(phi(Q))

        idx_clu, dist_clu, dt = clu_in_phi(keys, feat_q, s, cfg)
        idx_knn = knn_in_phi(keys, feat_q)

        row = {
            "position": t,
            "M": int(len(patterns)),
            "well_width_s": float(s),
            "clu_dt": float(dt),
            "median_nn_keys": float(_median_nn_distance(keys)),
            "per_task": {},
            "overall": {},
        }
        all_mask = np.ones(len(patterns), bool)
        for line, idx in (("clu_in_phi", idx_clu), ("knn_in_phi", idx_knn)):
            row["overall"][line] = _metrics(
                patterns, labels, idx, all_mask, true_idx, cfg
            )
            for tau in range(t):
                row["per_task"].setdefault(str(tau), {})[line] = _metrics(
                    patterns, labels, idx, task_of == tau, true_idx, cfg
                )
        row["mean_dist_to_well"] = float(np.mean(dist_clu))
        rows.append(row)
    return {"regime": regime, "arm": arm, "seed": int(seed),
            "phi_provenance": prov, "rows": rows}


# ---------------------------------------------------------------------------
# Item 2 ⭐ — the cost-of-strictness curve
# ---------------------------------------------------------------------------


def _seed_mean(values):
    a = np.asarray(values, float)
    return float(a.mean()), float(a.std(ddof=0))


def _slope(xs, ys):
    if len(xs) < 2:
        return 0.0
    return float(np.polyfit(np.asarray(xs, float), np.asarray(ys, float), 1)[0])


def cost_of_strictness(runs, cfg, metric="identity_acc", line="clu_in_phi"):
    """⭐ The deliverable. For each ``φ`` arm: accuracy **per task index** at the END
    of the stream and **per stream position** (mean over tasks seen), for both
    regimes, plus the ``generic_frozen − task1_only`` gap and its growth slope.

    A gap that widens steeply with task index means the task-1 ``φ`` cannot
    represent later classes ⇒ the w25 entry needs an online ``φ``.
    """
    out = {}
    arms = sorted({r["arm"] for r in runs})
    for arm in arms:
        per_regime_final, per_regime_stream = {}, {}
        for regime in sorted({r["regime"] for r in runs}):
            sel = [r for r in runs if r["arm"] == arm and r["regime"] == regime]
            if not sel:
                continue
            final_by_task, stream_by_pos = [], []
            for tau in range(cfg.n_tasks):
                vals = [
                    r["rows"][-1]["per_task"][str(tau)][line][metric]
                    for r in sel
                    if str(tau) in r["rows"][-1]["per_task"]
                ]
                mu, sd = _seed_mean(vals)
                final_by_task.append({"task": tau, "mean": mu, "std": sd,
                                      "n_seeds": len(vals)})
            for t in range(1, cfg.n_tasks + 1):
                vals = [
                    float(np.mean([
                        r["rows"][t - 1]["per_task"][str(tau)][line][metric]
                        for tau in range(t)
                    ]))
                    for r in sel
                ]
                mu, sd = _seed_mean(vals)
                stream_by_pos.append({"position": t, "mean": mu, "std": sd,
                                      "n_seeds": len(vals)})
            per_regime_final[regime] = final_by_task
            per_regime_stream[regime] = stream_by_pos

        entry = {"metric": metric, "line": line,
                 "final_by_task": per_regime_final,
                 "stream_by_position": per_regime_stream}
        if "task1_only" in per_regime_final and "generic_frozen" in per_regime_final:
            gap_task = [
                g["mean"] - s["mean"]
                for g, s in zip(per_regime_final["generic_frozen"],
                                per_regime_final["task1_only"], strict=True)
            ]
            gap_pos = [
                g["mean"] - s["mean"]
                for g, s in zip(per_regime_stream["generic_frozen"],
                                per_regime_stream["task1_only"], strict=True)
            ]
            entry["gap_generic_minus_task1"] = {
                "by_task_index": gap_task,
                "slope_per_task_index": _slope(range(cfg.n_tasks), gap_task),
                "by_stream_position": gap_pos,
                "slope_per_stream_position": _slope(
                    range(1, cfg.n_tasks + 1), gap_pos
                ),
                "gap_at_task1": float(gap_task[0]),
                "gap_at_last_task": float(gap_task[-1]),
                "gap_end_of_stream": float(gap_pos[-1]),
                "note": "positive = the (leaky) generic φ is better; this is the "
                        "COST OF STRICTNESS paid by the primary task-1-only arm",
            }
        out[arm] = entry
    return out


# ---------------------------------------------------------------------------
# Item 4 — the laundering control (kNN-in-φ), MANDATORY in every regime
# ---------------------------------------------------------------------------


def stream_laundering_control(runs, cfg, metric="identity_acc"):
    """N89 / CM-22(i): same ``φ``, trivial store swap. Per (regime, arm), does
    CLU-in-φ ever beat kNN-in-φ (outside the tie band) on the per-task-index axis at
    the end of the stream? If not, the win is ``φ``'s, not the store's."""
    band = cfg.laundering_tie_band
    out = []
    for arm in sorted({r["arm"] for r in runs}):
        for regime in sorted({r["regime"] for r in runs}):
            sel = [r for r in runs if r["arm"] == arm and r["regime"] == regime]
            if not sel:
                continue
            per_task = []
            for tau in range(cfg.n_tasks):
                d = [
                    r["rows"][-1]["per_task"][str(tau)]["clu_in_phi"][metric]
                    - r["rows"][-1]["per_task"][str(tau)]["knn_in_phi"][metric]
                    for r in sel
                    if str(tau) in r["rows"][-1]["per_task"]
                ]
                per_task.append({"task": tau, "clu_minus_knn": _seed_mean(d)[0]})
            arr = np.array([p["clu_minus_knn"] for p in per_task])
            n_clu = int(np.sum(arr > band))
            laundered = n_clu == 0
            out.append({
                "arm": arm, "regime": regime, "metric": metric, "tie_band": band,
                "per_task": per_task,
                "n_points": len(arr),
                "n_clu_wins": n_clu,
                "n_knn_wins": int(np.sum(arr < -band)),
                "n_tie": int(np.sum(np.abs(arr) <= band)),
                "mean_clu_minus_knn": float(arr.mean()),
                "max_clu_margin": float(arr.max()),
                "laundered": laundered,
                "verdict": (
                    "LAUNDERED — CLU-in-φ never beats kNN-in-φ in this regime: the "
                    "win is φ's, not ours."
                    if laundered else
                    f"CLU-in-φ beats kNN-in-φ on {n_clu}/{len(arr)} task indices "
                    f"(max +{float(arr.max()):.3f}) — a margin that exists ONLY "
                    "with the designed store."
                ),
            })
    return out


def store_advantage_watch_item(laundering, metric="identity_acc"):
    """⚠ **Pre-registered WATCH-ITEM, not a hunt (Item 4).** If the strict
    (task-1-only) ``φ`` makes kNN-in-φ *worse* while CLU-in-φ holds up, the CLU−kNN
    margin RISES under strictness — the first evidence of a store advantage. We
    only report the margin difference; we do not go looking for it."""
    out = []
    by = {(e["arm"], e["regime"]): e for e in laundering if e["metric"] == metric}
    for arm in sorted({a for a, _ in by}):
        s, g = by.get((arm, "task1_only")), by.get((arm, "generic_frozen"))
        if s is None or g is None:
            continue
        delta = s["mean_clu_minus_knn"] - g["mean_clu_minus_knn"]
        out.append({
            "arm": arm, "metric": metric,
            "margin_task1_only": s["mean_clu_minus_knn"],
            "margin_generic_frozen": g["mean_clu_minus_knn"],
            "delta_strict_minus_generic": float(delta),
            "store_advantage_under_strictness": bool(
                delta > 0 and not s["laundered"]
            ),
            "note": "watch-item only: a positive delta with laundering NOT firing "
                    "in the strict regime would be the first store-advantage "
                    "evidence; a positive delta alone is not a claim.",
        })
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _plot_cost(cost, save_dir):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    paths = []
    for metric_key, cost_d in cost.items():
        for arm, entry in cost_d.items():
            fin = entry["final_by_task"]
            if len(fin) < 1:
                continue
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
            for regime, style in (("task1_only", "o-"), ("generic_frozen", "s--")):
                if regime not in fin:
                    continue
                xs = [d["task"] for d in fin[regime]]
                ys = [d["mean"] for d in fin[regime]]
                es = [d["std"] for d in fin[regime]]
                axes[0].errorbar(xs, ys, yerr=es, fmt=style, capsize=2,
                                 label=f"CLU-in-φ / {regime}")
                st = entry["stream_by_position"][regime]
                axes[1].errorbar([d["position"] for d in st],
                                 [d["mean"] for d in st],
                                 yerr=[d["std"] for d in st],
                                 fmt=style, capsize=2, label=regime)
            axes[0].set_xlabel("task index τ (evaluated at end of stream)")
            axes[0].set_ylabel(metric_key)
            axes[0].set_title(f"cost of strictness — φ={arm} ({metric_key})")
            axes[0].legend(fontsize=7)
            axes[1].set_xlabel("stream position t (tasks written)")
            axes[1].set_ylabel(f"mean {metric_key} over tasks seen")
            axes[1].set_title("stream view")
            axes[1].legend(fontsize=7)
            fig.tight_layout()
            p = os.path.join(save_dir, f"phi_stream_cost_{arm}_{metric_key}.png")
            fig.savefig(p, dpi=140)
            plt.close(fig)
            paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_experiment_phi_stream(
    config: Optional[CHLUConfig] = None,
    save_dir: str = "results",
    models_dir: Optional[str] = None,
    seed: Optional[int] = None,
    data=None,
):
    config = config or get_default_config()
    cfg = config.experiment_phi_stream
    seeds = [seed] if seed is not None else list(cfg.seeds)
    os.makedirs(save_dir, exist_ok=True)

    regimes = [r for r in cfg.phi_regimes if r != "online"]
    skipped_online = "online" in cfg.phi_regimes

    runs = []
    for sd in seeds:
        stream = build_stream(cfg, sd, data=data)
        for regime in regimes:
            for arm in cfg.phi_arms:
                runs.append(run_stream_regime(cfg, stream, regime, arm, sd))

    cost = {
        m: cost_of_strictness(runs, cfg, metric=m, line="clu_in_phi")
        for m in ("identity_acc", "class_acc")
    }
    laundering = [
        e
        for m in ("identity_acc", "class_acc")
        for e in stream_laundering_control(runs, cfg, metric=m)
    ]
    results = {
        "seeds": [int(s) for s in seeds],
        "head_ruling": {
            "primary": "task1_only",
            "reference_upper_bound": "generic_frozen",
            "not_run": "online",
            "roles": REGIME_ROLE,
            "online_stub_note": ONLINE_STUB_NOTE,
        },
        "protocol": {
            "stream": f"Split-{cfg.dataset.upper()}-shaped: {cfg.n_tasks} tasks × "
                      f"{cfg.classes_per_task} classes, class-incremental",
            "store": "DESIGNED Gaussian wells over φ(x); payload = raw x; read-out "
                     "ψ = damped-Verlet settle → payload of nearest well",
            "phi_law": "φ is NEVER trained through the store (w20) and NEVER fit on "
                       "a stored pattern (store/fit regions are disjoint)",
            "queries": f"torch.dropout(x, p={cfg.mask_p}) — generated once per "
                       "stored item, IDENTICAL across regimes/arms/positions",
            "laundering_control": "kNN-in-φ reported in EVERY regime (N89, CM-22(i))",
            "uhop_commit": UHOP_COMMIT,
            "ramsauer_commit": RAMSAUER_COMMIT,
            "scope_caveat": "diagnostic of φ's stream discipline, NOT a competitive "
                            "claim; masked recall is appendix-only (Head ruling)",
        },
        "config": {
            k: getattr(cfg, k)
            for k in (
                "dataset", "n_tasks", "classes_per_task", "phi_regimes", "phi_arms",
                "seeds", "items_per_task", "n_fit_pool", "n_store_region", "mask_p",
                "phi_dim", "ae_hidden", "ae_epochs", "ae_lr", "ae_batch",
                "clu_s_frac", "clu_b", "clu_alpha", "clu_gamma", "clu_steps",
                "clu_tail_frac", "clu_kinetic_mode", "s_policy",
                "laundering_tie_band", "success_cosine",
            )
        },
        "online_regime_skipped": skipped_online,
        "runs": runs,
        "cost_of_strictness": cost,
        "laundering_control": laundering,
        "store_advantage_watch_item": {
            m: store_advantage_watch_item(laundering, metric=m)
            for m in ("identity_acc", "class_acc")
        },
    }
    results["figures"] = _plot_cost(cost, save_dir)

    results_dir = os.path.join(os.path.dirname(os.path.abspath(save_dir)), "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "exp_phi_stream_metrics.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    results["metrics_path"] = out_path
    return results


def apply_quick(config: CHLUConfig) -> None:
    """Quick smoke settings — same code path, one seed and a small store."""
    cfg = config.experiment_phi_stream
    cfg.seeds = [0]
    cfg.items_per_task = 8
    cfg.n_fit_pool = 400
    cfg.n_store_region = 4000
    cfg.phi_dim = 16
    cfg.ae_hidden = 64
    cfg.ae_epochs = 60
    cfg.clu_steps = 60
    cfg.rollout_chunk = 64


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="φ stream discipline: task-1-only vs generic-frozen φ on a "
                    "class-incremental stream (cost-of-strictness curve)"
    )
    parser.add_argument("--project", help="Project name to use (default: ./results)")
    parser.add_argument("--seed", type=int, help="Single seed (overrides cfg.seeds)")
    parser.add_argument("--quick", action="store_true", help="Quick smoke mode")
    parser.add_argument("--regimes", help="Override φ regimes (comma-separated)")
    parser.add_argument("--arms", help="Override φ arms (comma-separated: pca,ae)")
    args = parser.parse_args()

    if args.project:
        from chlu.project import ProjectManager

        pm = ProjectManager()
        config = pm.load(args.project)
        paths = pm.get_paths(args.project)
        save_dir, models_dir = str(paths["plots"]), str(paths["models"])
    else:
        config = get_default_config()
        save_dir, models_dir = "results", None
        os.makedirs(save_dir, exist_ok=True)

    if args.quick:
        apply_quick(config)
    if args.regimes:
        config.experiment_phi_stream.phi_regimes = args.regimes.split(",")
    if args.arms:
        config.experiment_phi_stream.phi_arms = args.arms.split(",")

    res = run_experiment_phi_stream(
        config=config, save_dir=save_dir, models_dir=models_dir, seed=args.seed
    )
    print(
        json.dumps(
            {
                "cost_of_strictness": res["cost_of_strictness"],
                "laundering_control": res["laundering_control"],
                "store_advantage_watch_item": res["store_advantage_watch_item"],
            },
            indent=2,
            default=float,
        )[:8000]
    )


if __name__ == "__main__":
    main()
