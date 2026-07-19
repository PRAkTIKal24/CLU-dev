"""``chlu eval`` — score a real anomaly dataset with the CLU + baselines.

Entry point the CSF3 job (``scripts/csf3/job_gpu_eval.sh``) calls. Runs the
model-agnostic evaluation harness (``chlu.eval.harness.evaluate_dataset``) with
a scorer factory that includes the CLU anomaly-score arms alongside the four
mandatory statistical baselines, then writes:

  - ``eval_<dataset>.npz``      — the ``EvalRunResult`` metric tensor (VUS-PR
                                  primary for point labels, AUROC for episode),
  - ``eval_<dataset>.md``       — the paper-ready per-dataset table,
  - ``eval_<dataset>_raw.npz``  — pooled raw per-arm score/label arrays (so ROC
                                  / PR curves can be re-plotted without rerun),
  - ``eval_<dataset>_roc.npz``  — per-arm ROC curve (fpr, tpr) + AUROC/AUPR.

Both first-pass CLU arms (``energy``/``residual`` and ``predict``) are emitted
by default so the Head can compare the score modes from the data (the score
mode IS the experiment; Head 2026-07-19).
"""

from pathlib import Path

import numpy as np
from rich.console import Console

from ..eval.config import (
    CLU_DEFAULT_SCORE_MODES,
    CLU_SCORE_MODES,
    CLULatticeConfig,
    CLUScorerConfig,
    EvalConfig,
    WindowConfig,
)
from ..eval.harness import evaluate_dataset

console = Console()

#: dataset key -> (module path, class name)
_DATASETS = {
    "voraus": ("chlu.data.industrial.voraus_ad", "VorausAD"),
    "skab": ("chlu.data.industrial.skab", "SKAB"),
    "tep": ("chlu.data.industrial.tep_rieth", "TEPRieth"),
    "smd": ("chlu.data.industrial.smd_tsb", "SMDTSB"),
}


def setup_eval_parser(subparsers):
    """Register the ``eval`` subcommand."""
    p = subparsers.add_parser(
        "eval",
        help="Score a real anomaly dataset with the CLU + statistical baselines",
    )
    p.add_argument(
        "--dataset", required=True, choices=sorted(_DATASETS),
        help="Which industrial anomaly dataset to score",
    )
    p.add_argument(
        "--score-mode", default="default",
        choices=["default", "all", *CLU_SCORE_MODES],
        help="CLU arm(s): 'default' = energy+residual+predict (both first-pass "
             "arms), 'all' adds hybrid, or a single mode",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed (init/train/subsample)")
    p.add_argument(
        "--limit", type=int, default=None,
        help="Cap the number of train AND test units (small real-slice smoke)",
    )
    p.add_argument("--out", default="results", help="Output directory")
    p.add_argument("--root", default=None, help="Dataset data root (else default cache)")
    p.add_argument("--download", action="store_true", help="Download the dataset if missing")
    p.add_argument("--variant", default="100hz", help="voraus variant (100hz|500hz)")
    p.add_argument("--window", type=int, default=100, help="Sliding window size")
    p.add_argument("--stride", type=int, default=1, help="Test-window stride")
    p.add_argument("--train-stride", type=int, default=1, help="Train-window stride")
    p.add_argument(
        "--metrics-mode", default="full", choices=["full", "fast"],
        help="'full' = VUS + F1 family; 'fast' = threshold-independent only",
    )
    p.add_argument(
        "--max-train-windows", type=int, default=100_000,
        help="Cap on statistical-baseline train windows",
    )
    # CLU knobs (rest live at CLUScorerConfig defaults; override the common ones)
    p.add_argument("--epochs", type=int, default=None, help="CLU training epochs")
    p.add_argument("--kinetic-mode", default=None,
                   choices=["newtonian_identity", "newtonian_learned", "relativistic"])
    p.add_argument("--max-fit-windows", type=int, default=None,
                   help="Cap on windows fed to the CLU fit")
    p.add_argument("--quick", action="store_true",
                   help="Fast smoke: few epochs, small window/caps")
    # G7b torus-coset lattice hook (flag, not default)
    p.add_argument("--lattice", action="store_true",
                   help="Fit a CLULattice (torus-coset hook) instead of a single CHLU")
    p.add_argument("--lattice-topology", default="chain", choices=["chain", "torus"])
    p.add_argument("--lattice-unit-dim", type=int, default=2,
                   help="Channels per lattice unit (2 = one SO(2) coset)")
    p.set_defaults(func=cmd_eval)


def _make_dataset(args):
    import importlib

    mod_path, cls_name = _DATASETS[args.dataset]
    cls = getattr(importlib.import_module(mod_path), cls_name)
    kwargs = {"download": args.download}
    if args.root is not None:
        kwargs["root"] = args.root
    if args.dataset == "voraus":
        kwargs["variant"] = args.variant
    return cls(**kwargs)


def _resolve_modes(score_mode: str) -> tuple:
    if score_mode == "default":
        return CLU_DEFAULT_SCORE_MODES
    if score_mode == "all":
        return CLU_SCORE_MODES
    return (score_mode,)


def _clu_config(args) -> CLUScorerConfig:
    kw = {"seed": args.seed}
    if args.kinetic_mode is not None:
        kw["kinetic_mode"] = args.kinetic_mode
    if args.epochs is not None:
        kw["epochs"] = args.epochs
    if args.max_fit_windows is not None:
        kw["max_fit_windows"] = args.max_fit_windows
    if args.quick:
        kw.setdefault("epochs", 20)
        kw.setdefault("max_fit_windows", 800)
    if args.lattice:
        kw["lattice"] = CLULatticeConfig(
            unit_dim=args.lattice_unit_dim, topology=args.lattice_topology
        )
    return CLUScorerConfig(**kw)


def _compute_roc(raw_scores: dict) -> dict:
    """Per-arm ROC curve + AUROC/AUPR from pooled raw score/label arrays."""
    from sklearn.metrics import (
        average_precision_score,
        roc_auc_score,
        roc_curve,
    )

    out = {}
    for method, arr in raw_scores.items():
        s, y = np.asarray(arr["scores"], float), np.asarray(arr["labels"], int)
        if y.min() == y.max():  # single-class — ROC undefined
            out[method] = {"fpr": np.array([]), "tpr": np.array([]),
                           "auroc": float("nan"), "aupr": float("nan")}
            continue
        fpr, tpr, _ = roc_curve(y, s)
        out[method] = {
            "fpr": fpr, "tpr": tpr,
            "auroc": float(roc_auc_score(y, s)),
            "aupr": float(average_precision_score(y, s)),
        }
    return out


def cmd_eval(args) -> int:
    from ..eval.clu_scorer import make_clu_scorers

    window = args.window
    max_train = args.max_train_windows
    if args.quick:
        window = min(window, 32)
        max_train = min(max_train, 2000)

    eval_cfg = EvalConfig(
        window=WindowConfig(size=window, stride=args.stride, train_stride=args.train_stride),
        metrics_sliding_window=window,
        metrics_mode=args.metrics_mode,
        seed=args.seed,
        max_train_windows=max_train,
    )
    clu_cfg = _clu_config(args)
    modes = _resolve_modes(args.score_mode)

    console.print(f"[bold cyan]chlu eval[/bold cyan] dataset={args.dataset} "
                  f"modes={modes} window={window} seed={args.seed}")
    dataset = _make_dataset(args)

    train_ids = test_ids = None
    if args.limit is not None:
        try:
            train_ids = dataset.train_ids()[: args.limit]
            test_ids = dataset.test_ids()[: args.limit]
            console.print(f"[dim]--limit {args.limit}: {len(train_ids)} train / "
                          f"{len(test_ids)} test units[/dim]")
        except NotImplementedError:
            console.print("[yellow]--limit ignored (dataset has no unit split)[/yellow]")

    def factory():
        return make_clu_scorers(eval_cfg, clu_cfg, modes=modes)

    raw_scores: dict = {}
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = evaluate_dataset(
        dataset,
        config=eval_cfg,
        train_ids=train_ids,
        test_ids=test_ids,
        scorer_factory=factory,
        out_dir=out_dir,
        raw_scores=raw_scores,
        verbose=True,
    )

    # markdown table
    md_path = out_dir / f"eval_{result.dataset}.md"
    md_path.write_text(result.to_markdown() + "\n_CLU config: " + clu_cfg.to_json() + "_\n")

    # raw scores + ROC curves
    if raw_scores:
        np.savez(
            out_dir / f"eval_{result.dataset}_raw.npz",
            **{f"{m}__scores": raw_scores[m]["scores"] for m in raw_scores},
            **{f"{m}__labels": raw_scores[m]["labels"] for m in raw_scores},
        )
        roc = _compute_roc(raw_scores)
        np.savez(
            out_dir / f"eval_{result.dataset}_roc.npz",
            **{f"{m}__fpr": roc[m]["fpr"] for m in roc},
            **{f"{m}__tpr": roc[m]["tpr"] for m in roc},
            auroc=np.array([roc[m]["auroc"] for m in roc]),
            aupr=np.array([roc[m]["aupr"] for m in roc]),
            methods=np.array(list(roc), dtype=str),
        )

    # console summary — primary metric + per-arm AUROC
    console.print("\n[bold]== summary ==[/bold]")
    agg = result.aggregate()
    primary = result.primary_metric
    for method in result.methods:
        mean, std, n = agg[method][primary]
        auroc = agg[method].get("AUC-ROC", (float("nan"),))[0]
        console.print(f"  {method:14s} {primary}={mean:.4f}  AUROC={auroc:.4f}")
    console.print(f"\n[green]wrote {md_path} and eval_{result.dataset}[_raw|_roc].npz "
                  f"under {out_dir}[/green]")
    return 0
