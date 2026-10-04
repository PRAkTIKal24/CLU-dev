"""Dataset-agnostic industrial anomaly-detection evaluation harness (F2).

Binding evaluation rules (scout-industrial-datasets, 2026-07-05):
- VUS-PR is the primary metric; range-aware/event metrics and AUROC/AUPR are
  secondary. Metrics are computed by TSB-AD's reference implementation
  (vendored, pinned — see ``chlu/eval/_tsb_vendor/README.md``).
- Point-adjust F1 is FORBIDDEN and deliberately absent from this package.
- The four statistical baselines (PCA-recon, IsolationForest, LOF, KNN) are
  wired into every evaluation run by default.
- Splits are leakage-safe: by physical unit / run / simulation seed, never by
  window.
- Results are reported per dataset, never as a single grand mean.

No CLU modeling lives here — this is the measurement backbone.
"""

from chlu.eval.baselines import (
    BaselineScorer,
    IForestBaseline,
    KNNBaseline,
    LOFBaseline,
    PCAReconBaseline,
    make_default_baselines,
    sliding_windows,
    window_scores_to_point_scores,
)
from chlu.eval.config import EvalConfig, WindowConfig
from chlu.eval.harness import EvalRunResult, evaluate_dataset, results_to_markdown
from chlu.eval.metrics import (
    EPISODE_METRICS,
    FORBIDDEN_METRIC_NAMES,
    POINT_METRICS,
    PRIMARY_METRIC,
    assert_metric_allowed,
    compute_episode_metrics,
    compute_point_metrics,
)
from chlu.eval.splits import (
    assert_no_unit_leakage,
    cross_condition_split,
    unit_split,
)

__all__ = [
    "BaselineScorer",
    "EPISODE_METRICS",
    "EvalConfig",
    "EvalRunResult",
    "FORBIDDEN_METRIC_NAMES",
    "IForestBaseline",
    "KNNBaseline",
    "LOFBaseline",
    "PCAReconBaseline",
    "POINT_METRICS",
    "PRIMARY_METRIC",
    "WindowConfig",
    "assert_metric_allowed",
    "assert_no_unit_leakage",
    "compute_episode_metrics",
    "compute_point_metrics",
    "cross_condition_split",
    "evaluate_dataset",
    "make_default_baselines",
    "results_to_markdown",
    "sliding_windows",
    "unit_split",
    "window_scores_to_point_scores",
]
