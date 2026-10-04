"""Metric wrappers around the vendored TSB-AD reference implementation.

Binding rules implemented here:
- **VUS-PR is the primary metric** for point-labelled series (Liu &
  Paparrizos, NeurIPS 2024); VUS-ROC / AUC-PR / AUC-ROC and the range-aware /
  event-wise F1 family are secondary.
- **Point-adjust F1 is forbidden.** It is excised from the vendored code
  (``_tsb_vendor/README.md``) and ``assert_metric_allowed`` refuses PA names,
  so it cannot be quietly (re)introduced.
- VUS-PR is **not reimplemented** — computation is delegated to TSB-AD's own
  code, vendored and pinned at v1.5.

Episode-labelled datasets (one anomaly label per run/episode, e.g. voraus-AD)
have no per-timestep ground truth, so volume-under-surface metrics are
undefined; the convention there (matching the voraus-AD paper) is episode-level
AUROC (primary) + AUPR.
"""

import warnings

import numpy as np

#: Metric names produced for point-labelled (per-timestep) ground truth.
POINT_METRICS = (
    "VUS-PR",
    "VUS-ROC",
    "AUC-PR",
    "AUC-ROC",
    "Standard-F1",
    "Event-based-F1",
    "R-based-F1",
    "Affiliation-F",
)

#: Threshold-independent subset (metrics_mode="fast").
POINT_METRICS_FAST = ("VUS-PR", "VUS-ROC", "AUC-PR", "AUC-ROC")

#: Metric names produced for episode-labelled ground truth.
EPISODE_METRICS = ("AUC-ROC", "AUC-PR")

#: Primary metric per label kind (first column of every report).
PRIMARY_METRIC = {"point": "VUS-PR", "episode": "AUC-ROC"}

#: Point-adjust aliases that must never appear in this harness.
FORBIDDEN_METRIC_NAMES = frozenset(
    {
        "pa-f1",
        "paf1",
        "pa_f1",
        "pointf1pa",
        "point-adjust-f1",
        "point_adjust_f1",
        "adjusted-f1",
        "adjusted_f1",
        "f1-pa",
        "f1_pa",
    }
)


def assert_metric_allowed(name: str) -> None:
    """Raise ``ValueError`` if ``name`` is a point-adjust metric alias.

    Point-adjust evaluation can turn a random score into apparent SOTA
    (Kim et al., AAAI 2022, arXiv:2109.05257) and is forbidden project-wide.
    """
    if name.strip().lower().replace(" ", "") in FORBIDDEN_METRIC_NAMES:
        raise ValueError(
            f"metric '{name}' is point-adjust F1 — forbidden by the CHLU "
            "evaluation protocol (Kim et al., AAAI 2022). Use VUS-PR (primary) "
            "or the range-aware/event secondaries instead."
        )


def _validate_inputs(score: np.ndarray, labels: np.ndarray) -> tuple:
    score = np.asarray(score, dtype=np.float64).ravel()
    labels = np.asarray(labels).ravel().astype(np.int64)
    if score.shape != labels.shape:
        raise ValueError(f"score {score.shape} and labels {labels.shape} differ")
    if not np.all(np.isfinite(score)):
        raise ValueError("anomaly scores contain NaN/Inf")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("labels must be binary 0/1")
    return score, labels


def _nan_metrics(names: tuple) -> dict:
    return {name: float("nan") for name in names}


def compute_point_metrics(
    score: np.ndarray,
    labels: np.ndarray,
    sliding_window: int = 100,
    mode: str = "full",
) -> dict:
    """Compute the F2 metric set for one point-labelled series.

    Args:
        score: (T,) anomaly scores, higher = more anomalous.
        labels: (T,) binary ground truth per timestep.
        sliding_window: TSB-AD's ``slidingWindow`` for the VUS buffer region
            (explicit per dataset — record it alongside results).
        mode: "full" (TSB-AD ``get_metrics`` set) or "fast"
            (threshold-independent metrics only).

    Returns:
        dict metric-name -> float. Degenerate ground truth (all-normal or
        all-anomalous) yields NaNs with a warning, so callers can aggregate
        with ``nanmean`` and report the valid count.
    """
    score, labels = _validate_inputs(score, labels)
    names = POINT_METRICS if mode == "full" else POINT_METRICS_FAST
    if labels.min() == labels.max():
        warnings.warn(
            "degenerate ground truth (single class) — point metrics undefined, "
            "returning NaNs",
            stacklevel=2,
        )
        return _nan_metrics(names)

    # Delegate to the vendored TSB-AD implementation (never reimplement VUS).
    from chlu.eval._tsb_vendor.basic_metrics import basic_metricor, generate_curve
    from chlu.eval._tsb_vendor.metrics import get_metrics

    if mode == "full":
        raw = get_metrics(score, labels, slidingWindow=sliding_window)
    elif mode == "fast":
        grader = basic_metricor()
        _, _, _, _, _, _, vus_roc, vus_pr = generate_curve(
            labels.astype(int), score, sliding_window
        )
        raw = {
            "VUS-PR": vus_pr,
            "VUS-ROC": vus_roc,
            "AUC-PR": grader.metric_PR(labels, score),
            "AUC-ROC": grader.metric_ROC(labels, score),
        }
    else:
        raise ValueError(f"mode must be full|fast, got {mode}")

    # Defense in depth: no forbidden name may ever leave this wrapper.
    for key in raw:
        assert_metric_allowed(key)
    return {name: float(raw[name]) for name in names}


def compute_episode_metrics(scores: np.ndarray, labels: np.ndarray) -> dict:
    """Episode-level AUROC/AUPR (datasets labelled per run, not per timestep).

    Args:
        scores: (n_episodes,) one anomaly score per episode.
        labels: (n_episodes,) binary episode labels.

    Returns:
        dict metric-name -> float (NaNs when ground truth is single-class).
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    scores, labels = _validate_inputs(scores, labels)
    if labels.min() == labels.max():
        warnings.warn(
            "degenerate episode ground truth (single class) — returning NaNs",
            stacklevel=2,
        )
        return _nan_metrics(EPISODE_METRICS)
    return {
        "AUC-ROC": float(roc_auc_score(labels, scores)),
        "AUC-PR": float(average_precision_score(labels, scores)),
    }
