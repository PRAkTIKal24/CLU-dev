#!/usr/bin/env python
"""Run CLU on the CAFE benchmark (``cafe-bench``) — the CLU->CAFE entry point.

CAFE is an EXTERNAL, separately-cloned harness (private sibling repo). It is
deliberately NOT vendored here; put its checkout on ``PYTHONPATH``::

    git clone git@github.com:Forgis-Labs/CAFE.git ~/cafe-bench
    PYTHONPATH=~/cafe-bench uv run python scripts/cafe/run_clu_cafe.py \
        --dataset cmapss_fd001 --data-root ~/cafe-data

The harness owns splits/probes/metrics; this script only builds a configured
CLU, registers it (``chlu.eval.cafe_model.register``) and calls
``cafe_bench.pipeline.run``, so the number produced is the harness's own.

FAIRNESS (do not "fix" this without reading it): CAFE's own HEPA wrapper is
``encode()``-only and scores Event Prediction through the DEFAULT CoxPH probe.
CLU therefore does the same by default. ``--model clu_valley`` overrides only
the *anomaly* probe, and is a separate leaderboard identity.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="cmapss_fd001",
                   help="CAFE dataset key (e.g. cmapss_fd001).")
    p.add_argument("--model", default="clu", choices=("clu", "clu_valley"),
                   help="Which registered CLU identity to run.")
    p.add_argument("--data-root", default="data",
                   help="CAFE data root (contains cmapss/, etc).")
    p.add_argument("--results-dir", default="results",
                   help="Where the harness writes results JSON.")
    p.add_argument("--cafe-root", default=None,
                   help="Path to the CAFE checkout (prepended to sys.path).")
    # --- CLU training knobs (config-driven; defaults = CLUScorerConfig) ---
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--max-fit-windows", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--hidden", type=int, default=None)
    p.add_argument("--kinetic-mode", default=None,
                   choices=("newtonian_identity", "newtonian_learned", "relativistic"))
    p.add_argument("--dt", type=float, default=None)
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--relax-steps", type=int, default=None)
    p.add_argument("--predict-horizon", type=int, default=None)
    p.add_argument("--mass-lr-mult", type=float, default=None,
                   help="Run log_mass on its own Adam slot at lr*MULT. The "
                        "default (1.0) is what every CLU run in this program "
                        "has used, and it leaves the mass spectrum at init "
                        "(see clu-latent-io-audit). 10 is the known-safe "
                        "setting; 100 inverts the ordering (CM-5/N8).")
    p.add_argument("--mass-spread-lambda", type=float, default=None,
                   help="R-1 mass-spread term: subtract LAMBDA*Var(log_mass) "
                        "from the loss, forcing a non-degenerate timescale "
                        "hierarchy. On from epoch 0 (T3). Default 0.0 = off.")
    p.add_argument("--seed", type=int, default=42)
    # --- encode knobs ---
    p.add_argument("--feature-groups", default=None,
                   help="Comma-separated subset of CAFE_FEATURE_GROUPS.")
    p.add_argument("--encode-batch-size", type=int, default=None)
    p.add_argument("--anomaly-mode", default=None)
    p.add_argument("--no-standardize", action="store_true")
    p.add_argument("--relax-gamma", type=float, default=None,
                   help="Override the encode-side relaxation dissipation. The "
                        "damping BUDGET gamma*steps*dt is the strongest measured "
                        "lever: the inherited default is ~0.16 (barely damped, "
                        "q* free-streams); ~1.6 measured best on C-MAPSS FD001; "
                        ">~60 collapses every window onto one settled point.")
    p.add_argument("--relax-steps-encode", type=int, default=None,
                   help="Override the encode-side relaxation rollout length.")
    # --- smoke ---
    p.add_argument("--subsample", type=int, default=None,
                   help="QUICK/SMOKE ONLY: cap train+test windows (seeded, "
                        "stratified by nothing — uniform). Registers a "
                        "'<dataset>_sub' dataset so the harness path is real.")
    p.add_argument("--quick", action="store_true",
                   help="Laptop smoke preset: few epochs + small subsample.")
    return p.parse_args(argv)


def _subsampled_dataset(base_key: str, n_cap: int, seed: int):
    """Wrap a registered EVENT dataset so ``load`` returns a capped subset.

    Used only for local smokes — it keeps the real loader, evaluator and probe
    in the loop while cutting the window count. Any number produced this way is
    a PIPELINE CHECK, not a benchmark result (CM-3).
    """
    import numpy as np
    from cafe_bench.datasets.base import EventBatch
    from cafe_bench.registry import get_dataset, register_dataset

    base = get_dataset(base_key)

    class _Sub:
        info = base.info

        def load(self, data_root):
            b = base.load(data_root)
            if not isinstance(b, EventBatch):
                raise ValueError("--subsample currently supports event datasets only")
            rng = np.random.default_rng(seed)

            def take(X, t, e, n):
                if len(X) <= n:
                    return X, t, e
                idx = np.sort(rng.choice(len(X), size=n, replace=False))
                return X[idx], t[idx], e[idx]

            Xtr, ttr, etr = take(b.X_train, b.t_train, b.e_train, n_cap)
            Xte, tte, ete = take(b.X_test, b.t_test, b.e_test, n_cap)
            return EventBatch(Xtr, ttr, etr, Xte, tte, ete, horizon_max=b.horizon_max)

    key = f"{base_key}_sub"
    register_dataset(key, _Sub())
    return key


def main(argv=None) -> int:
    args = _parse_args(argv)

    if args.cafe_root:
        sys.path.insert(0, str(Path(args.cafe_root).expanduser()))

    try:
        import cafe_bench  # noqa: F401
    except ImportError as e:
        print(
            "ERROR: cafe_bench not importable. Clone the CAFE benchmark and put "
            "it on PYTHONPATH (or pass --cafe-root):\n"
            "  git clone git@github.com:Forgis-Labs/CAFE.git ~/cafe-bench\n"
            f"  PYTHONPATH=~/cafe-bench {' '.join(sys.argv)}\n"
            f"(underlying error: {e})",
            file=sys.stderr,
        )
        return 2

    # Populate the dataset registry (the harness's CLI does the same).
    import cafe_bench.datasets.event.cmapss      # noqa: F401
    import cafe_bench.datasets.event.physionet   # noqa: F401

    from chlu.eval.cafe_model import register
    from chlu.eval.config import CLUCafeEncodeConfig, CLUScorerConfig

    registered = register()

    # --- build configs (only override what was asked for) ---------------
    quick = args.quick
    clu_kw = {"seed": args.seed}
    if quick:
        clu_kw.update(epochs=5, max_fit_windows=256, batch_size=32)
    for name, val in (
        ("epochs", args.epochs), ("max_fit_windows", args.max_fit_windows),
        ("batch_size", args.batch_size), ("lr", args.lr), ("hidden", args.hidden),
        ("kinetic_mode", args.kinetic_mode), ("dt", args.dt), ("gamma", args.gamma),
        ("relax_steps", args.relax_steps), ("predict_horizon", args.predict_horizon),
        ("mass_lr_mult", args.mass_lr_mult),
        ("mass_spread_lambda", args.mass_spread_lambda),
    ):
        if val is not None:
            clu_kw[name] = val
    clu_cfg = CLUScorerConfig(**clu_kw)

    enc_kw = {}
    if args.feature_groups:
        enc_kw["feature_groups"] = tuple(
            g.strip() for g in args.feature_groups.split(",") if g.strip()
        )
    if args.encode_batch_size is not None:
        enc_kw["batch_size"] = args.encode_batch_size
    if args.anomaly_mode is not None:
        enc_kw["anomaly_mode"] = args.anomaly_mode
    if args.no_standardize:
        enc_kw["standardize"] = False
    if args.relax_gamma is not None:
        enc_kw["relax_gamma"] = args.relax_gamma
    if args.relax_steps_encode is not None:
        enc_kw["relax_steps"] = args.relax_steps_encode
    enc_cfg = CLUCafeEncodeConfig(**enc_kw)

    model = registered[args.model](clu_config=clu_cfg, encode_config=enc_cfg)

    dataset_key = args.dataset
    n_cap = args.subsample if args.subsample is not None else (2000 if quick else None)
    if n_cap is not None:
        dataset_key = _subsampled_dataset(args.dataset, n_cap, args.seed)

    # --- provenance (flag-provenance rule) ------------------------------
    print("── CLU/CAFE run ─────────────────────────────────────────────")
    print(f"dataset      : {dataset_key}")
    print(f"model        : {args.model}")
    print(f"subsample    : {n_cap}")
    print(f"clu_config   : {clu_cfg.to_json()}")
    print(f"encode_config: {enc_cfg.to_json()}")
    print(f"relax_budget : {enc_cfg.relax_budget(clu_cfg):.3f}  (gamma*steps*dt)")
    print("─────────────────────────────────────────────────────────────")

    from cafe_bench.pipeline import run as _run

    record = _run(
        dataset_key,
        args.model,
        args.data_root,
        results_dir=args.results_dir,
        model_instance=model,
    )
    record["clu_config"] = json.loads(clu_cfg.to_json())
    record["encode_config"] = json.loads(enc_cfg.to_json())
    record["embedding_dim"] = len(model.feature_names())
    record["feature_names"] = model.feature_names()
    record["subsample"] = n_cap
    shared = getattr(model, "_shared", None)
    if shared is not None and getattr(shared, "mass_diagnostics", None):
        record["mass_diagnostics"] = shared.mass_diagnostics
        md = shared.mass_diagnostics
        print(
            f"log_mass: std {md['std_init']:.4g} -> {md['std_final']:.4g} | "
            f"max drift {md['max_abs_drift']:.4g} | "
            f"movement rms_ratio {md['movement']['rms_ratio']:.4g}"
        )

    out = Path(args.results_dir) / args.model / f"{dataset_key}.json"
    out.write_text(json.dumps(record, indent=2))
    print(f"h_auroc = {record['metrics'].get('h_auroc')}   -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
