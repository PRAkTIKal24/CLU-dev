"""Dataset-agnostic evaluation runner: loaders x baselines x TSB-AD metrics.

One call = one dataset (binding rule: per-dataset reporting, never a grand
mean). The four statistical baselines are always included; splits are
unit-level and re-checked for leakage on every run; results go to
``results/eval_<dataset>.npz`` plus a paper-ready markdown table.
"""

import json
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from chlu.data.industrial.base import IndustrialDataset, UnitRecord
from chlu.eval.baselines import (
    make_default_baselines,
    sliding_windows,
    window_scores_to_point_scores,
)
from chlu.eval.config import EvalConfig
from chlu.eval.metrics import (
    EPISODE_METRICS,
    POINT_METRICS,
    POINT_METRICS_FAST,
    PRIMARY_METRIC,
    compute_episode_metrics,
    compute_point_metrics,
)
from chlu.eval.splits import assert_no_unit_leakage

EPISODE_AXIS = "__all_episodes__"


@dataclass
class EvalRunResult:
    """Results of one harness run on one dataset."""

    dataset: str
    label_kind: str
    protocol: str
    methods: tuple
    unit_ids: tuple  # evaluated test units; episode mode: (EPISODE_AXIS,)
    metric_names: tuple
    values: np.ndarray  # (n_methods, n_units, n_metrics), NaN where undefined
    config_json: str
    n_train_windows: int
    timings_s: dict = field(default_factory=dict)
    notes: tuple = ()

    @property
    def primary_metric(self) -> str:
        return PRIMARY_METRIC[self.label_kind]

    def aggregate(self) -> dict:
        """Per-method nanmean/nanstd/valid-count across test units."""
        out = {}
        for i, method in enumerate(self.methods):
            per_metric = {}
            for k, metric in enumerate(self.metric_names):
                col = self.values[i, :, k]
                n_valid = int(np.sum(~np.isnan(col)))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    mean = float(np.nanmean(col)) if n_valid else float("nan")
                    std = float(np.nanstd(col)) if n_valid else float("nan")
                per_metric[metric] = (mean, std, n_valid)
            out[method] = per_metric
        return out

    def to_markdown(self) -> str:
        """Paper-ready per-dataset table (primary metric first, bold)."""
        ordered = [self.primary_metric] + [
            m for m in self.metric_names if m != self.primary_metric
        ]
        agg = self.aggregate()
        lines = [
            f"### {self.dataset} — {self.label_kind}-labelled, protocol={self.protocol}",
            "",
            "| method | "
            + " | ".join(
                f"**{ordered[0]}**" if i == 0 else m for i, m in enumerate(ordered)
            )
            + " |",
            "|" + "---|" * (len(ordered) + 1),
        ]
        for method in self.methods:
            cells = []
            for metric in ordered:
                mean, std, n = agg[method][metric]
                if np.isnan(mean):
                    cells.append("n/a")
                elif len(self.unit_ids) > 1:
                    cells.append(f"{mean:.3f} ± {std:.3f} (n={n})")
                else:
                    cells.append(f"{mean:.3f}")
            lines.append(f"| {method} | " + " | ".join(cells) + " |")
        cfg = json.loads(self.config_json)
        lines += [
            "",
            f"_config: window={cfg['window']}, metrics_sliding_window="
            f"{cfg['metrics_sliding_window']}, seed={cfg['seed']}; "
            f"{self.n_train_windows} train windows; per-dataset table — do not "
            "average across datasets._",
        ]
        if self.notes:
            lines += [""] + [f"_note: {n}_" for n in self.notes]
        return "\n".join(lines) + "\n"

    def save_npz(self, out_dir: str | Path) -> Path:
        """Write ``eval_<dataset>.npz`` (self-describing, pickle-free)."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"eval_{self.dataset}.npz"
        np.savez(
            path,
            dataset=np.array(self.dataset),
            label_kind=np.array(self.label_kind),
            protocol=np.array(self.protocol),
            methods=np.array(self.methods, dtype=str),
            unit_ids=np.array(self.unit_ids, dtype=str),
            metric_names=np.array(self.metric_names, dtype=str),
            values=self.values,
            config_json=np.array(self.config_json),
            n_train_windows=np.array(self.n_train_windows),
            timings_json=np.array(json.dumps(self.timings_s, sort_keys=True)),
            notes=np.array(self.notes, dtype=str),
        )
        return path


def load_eval_npz(path: str | Path) -> EvalRunResult:
    """Round-trip loader for ``EvalRunResult.save_npz`` files."""
    with np.load(path) as z:
        return EvalRunResult(
            dataset=str(z["dataset"]),
            label_kind=str(z["label_kind"]),
            protocol=str(z["protocol"]),
            methods=tuple(z["methods"]),
            unit_ids=tuple(z["unit_ids"]),
            metric_names=tuple(z["metric_names"]),
            values=z["values"],
            config_json=str(z["config_json"]),
            n_train_windows=int(z["n_train_windows"]),
            timings_s=json.loads(str(z["timings_json"])),
            notes=tuple(z["notes"]),
        )


def results_to_markdown(results: list) -> str:
    """Emit one markdown section per dataset — never merged, never averaged."""
    return "\n".join(r.to_markdown() for r in results)


# ---------------------------------------------------------------------------
# run logic
# ---------------------------------------------------------------------------


def _normal_train_windows(rec: UnitRecord, size: int, stride: int) -> np.ndarray:
    """Training windows from one unit, excluding any window that overlaps a
    labelled-anomalous timestep (semi-supervised normal-only protocol)."""
    windows = sliding_windows(rec.data, size, stride)
    labels = rec.point_labels
    if labels is None or not np.any(labels):
        return windows
    cum = np.concatenate([[0], np.cumsum(labels.astype(np.int64))])
    starts = np.arange(len(windows)) * stride
    n_anom = cum[starts + size] - cum[starts]
    return windows[n_anom == 0]


def _subsample(windows: np.ndarray, limit: int | None, seed: int) -> np.ndarray:
    if limit is None or len(windows) <= limit:
        return windows
    idx = np.random.default_rng(seed).choice(len(windows), size=limit, replace=False)
    return windows[np.sort(idx)]


def _fit_scaler(train: np.ndarray):
    from sklearn.preprocessing import StandardScaler

    return StandardScaler().fit(train)


def evaluate_dataset(
    dataset: IndustrialDataset,
    config: EvalConfig | None = None,
    train_ids: tuple | None = None,
    test_ids: tuple | None = None,
    scorer_factory=None,
    out_dir: str | Path | None = None,
    verbose: bool = True,
) -> EvalRunResult:
    """Run the mandatory baselines + metric suite on one dataset.

    Args:
        dataset: An ``IndustrialDataset`` instance.
        config: Explicit run configuration (default ``EvalConfig()``).
        train_ids / test_ids: Optional unit-level split override; defaults to
            the dataset's canonical split. Leakage-checked either way.
            (Ignored for ``per_unit_prefix`` datasets, which carry their own
            in-series temporal split.)
        scorer_factory: ``() -> dict[name, BaselineScorer]``. Defaults to the
            four mandatory statistical baselines; custom factories must keep
            them (enforced).
        out_dir: If given, results are written to ``out_dir/eval_<name>.npz``.
        verbose: Print progress.

    Returns:
        ``EvalRunResult`` (per-unit metric tensor + aggregation/markdown/npz).
    """
    config = config or EvalConfig()
    factory = scorer_factory or (lambda: make_default_baselines(config))
    required = set(make_default_baselines(config))
    probe = factory()
    missing = required - set(probe)
    if missing:
        raise ValueError(
            f"scorer_factory must include the mandatory statistical baselines "
            f"{sorted(required)}; missing {sorted(missing)}"
        )
    methods = tuple(probe)

    metric_names = (
        (POINT_METRICS if config.metrics_mode == "full" else POINT_METRICS_FAST)
        if dataset.label_kind == "point"
        else EPISODE_METRICS
    )
    log = print if verbose else (lambda *a, **k: None)

    if dataset.protocol == "per_unit_prefix":
        result = _run_per_unit_prefix(
            dataset, config, factory, methods, metric_names, log
        )
    else:
        result = _run_cross_unit(
            dataset, config, factory, methods, metric_names, train_ids, test_ids, log
        )

    if out_dir is not None:
        path = result.save_npz(out_dir)
        log(f"[harness] wrote {path}")
    return result


def _run_cross_unit(
    dataset, config, factory, methods, metric_names, train_ids, test_ids, log
):
    size, stride = config.window.size, config.window.stride
    train_ids = tuple(train_ids) if train_ids is not None else dataset.train_ids()
    test_ids = tuple(test_ids) if test_ids is not None else dataset.test_ids()
    assert_no_unit_leakage(train_ids, test_ids)

    train_parts, notes = [], []
    for rec in dataset.iter_units(train_ids):
        if len(rec.data) < size:
            notes.append(f"train unit {rec.unit_id} shorter than window — skipped")
            continue
        train_parts.append(_normal_train_windows(rec, size, config.window.train_stride))
    if not train_parts:
        raise ValueError("no usable training windows")
    train = np.concatenate(train_parts, axis=0)
    train = _subsample(train, config.max_train_windows, config.seed)
    log(
        f"[harness] {dataset.name}: {len(train)} train windows "
        f"(size={size}, train_stride={config.window.train_stride})"
    )

    scaler = _fit_scaler(train)
    train = scaler.transform(train)
    scorers = factory()
    timings = {}
    for name, scorer in scorers.items():
        t0 = time.perf_counter()
        scorer.fit(train)
        timings[name] = {"fit_s": round(time.perf_counter() - t0, 3)}

    if dataset.label_kind == "point":
        unit_axis = tuple(test_ids)
        values = np.full((len(methods), len(unit_axis), len(metric_names)), np.nan)
        for j, rec in enumerate(dataset.iter_units(test_ids)):
            if len(rec.data) < size:
                notes.append(f"test unit {rec.unit_id} shorter than window — NaN row")
                continue
            windows = scaler.transform(sliding_windows(rec.data, size, stride))
            labels = rec.point_labels
            if labels is None:
                notes.append(f"test unit {rec.unit_id} has no point labels — NaN row")
                continue
            for i, name in enumerate(methods):
                t0 = time.perf_counter()
                w_scores = scorers[name].score(windows)
                point_scores = window_scores_to_point_scores(
                    w_scores, len(rec.data), size, stride
                )
                m = compute_point_metrics(
                    point_scores,
                    labels,
                    sliding_window=config.metrics_sliding_window,
                    mode=config.metrics_mode,
                )
                values[i, j] = [m[k] for k in metric_names]
                timings[name]["score_s"] = round(
                    timings[name].get("score_s", 0.0) + time.perf_counter() - t0, 3
                )
            log(f"[harness]   scored unit {rec.unit_id}")
    else:  # episode labels
        unit_axis = (EPISODE_AXIS,)
        episode_scores = {name: [] for name in methods}
        episode_labels = []
        for rec in dataset.iter_units(test_ids):
            if len(rec.data) < size:
                notes.append(f"test unit {rec.unit_id} shorter than window — skipped")
                continue
            if rec.episode_label is None:
                notes.append(f"test unit {rec.unit_id} lacks episode label — skipped")
                continue
            windows = scaler.transform(sliding_windows(rec.data, size, stride))
            episode_labels.append(int(rec.episode_label))
            reduce = np.mean if config.episode_reduce == "mean" else np.max
            for name in methods:
                t0 = time.perf_counter()
                episode_scores[name].append(float(reduce(scorers[name].score(windows))))
                timings[name]["score_s"] = round(
                    timings[name].get("score_s", 0.0) + time.perf_counter() - t0, 3
                )
        values = np.full((len(methods), 1, len(metric_names)), np.nan)
        for i, name in enumerate(methods):
            m = compute_episode_metrics(
                np.asarray(episode_scores[name]), np.asarray(episode_labels)
            )
            values[i, 0] = [m[k] for k in metric_names]
        notes.append(
            f"episode scores pooled with '{config.episode_reduce}' over "
            f"{len(episode_labels)} test episodes"
        )

    return EvalRunResult(
        dataset=dataset.name,
        label_kind=dataset.label_kind,
        protocol=dataset.protocol,
        methods=methods,
        unit_ids=unit_axis,
        metric_names=metric_names,
        values=values,
        config_json=config.to_json(),
        n_train_windows=len(train),
        timings_s=timings,
        notes=tuple(notes),
    )


def _run_per_unit_prefix(dataset, config, factory, methods, metric_names, log):
    """TSB-AD-style protocol: each series carries its own normal prefix."""
    size, stride = config.window.size, config.window.stride
    unit_axis = dataset.test_ids()
    values = np.full((len(methods), len(unit_axis), len(metric_names)), np.nan)
    notes, timings, n_train_total = [], {m: {} for m in methods}, 0

    for j, rec in enumerate(dataset.iter_units(unit_axis)):
        train_len = int(rec.meta.get("train_len", 0))
        if rec.point_labels is None:
            notes.append(f"unit {rec.unit_id}: no point labels — NaN row")
            continue
        if train_len < size or len(rec.data) - train_len < size:
            notes.append(
                f"unit {rec.unit_id}: segment(s) shorter than window — NaN row"
            )
            continue
        train = sliding_windows(rec.data[:train_len], size, config.window.train_stride)
        train = _subsample(train, config.max_train_windows, config.seed)
        n_train_total += len(train)
        scaler = _fit_scaler(train)
        train_scaled = scaler.transform(train)
        eval_data = rec.data[train_len:]
        eval_labels = rec.point_labels[train_len:]
        windows = scaler.transform(sliding_windows(eval_data, size, stride))
        scorers = factory()
        for i, name in enumerate(methods):
            t0 = time.perf_counter()
            scorers[name].fit(train_scaled)
            w_scores = scorers[name].score(windows)
            point_scores = window_scores_to_point_scores(
                w_scores, len(eval_data), size, stride
            )
            m = compute_point_metrics(
                point_scores,
                eval_labels,
                sliding_window=config.metrics_sliding_window,
                mode=config.metrics_mode,
            )
            values[i, j] = [m[k] for k in metric_names]
            timings[name]["total_s"] = round(
                timings[name].get("total_s", 0.0) + time.perf_counter() - t0, 3
            )
        log(f"[harness]   scored unit {rec.unit_id} (train prefix {train_len})")

    notes.append(
        "per-unit-prefix protocol: baselines fit per series on its "
        "normal training prefix (TSB-AD convention)"
    )
    return EvalRunResult(
        dataset=dataset.name,
        label_kind=dataset.label_kind,
        protocol=dataset.protocol,
        methods=methods,
        unit_ids=unit_axis,
        metric_names=metric_names,
        values=values,
        config_json=config.to_json(),
        n_train_windows=n_train_total,
        timings_s=timings,
        notes=tuple(notes),
    )
