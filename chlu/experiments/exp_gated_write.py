"""Experiment: gated-write performance test (w22).

**Question.** `primitive-harness` scored the CLU 0/3 as a sequence primitive;
`gamma-read-sweep §2/§5` diagnosed the cause — the shipped write current
``p += W_in x_t`` is *unconditionally linear* in the token, so the state can
carry ``Σv_t`` and ``Σm_t`` but never their product, and every recall/parity/
adding task needs an input-conditioned conjunction. A multiplicative write gate
``p += (W_in x_t)⊙σ(W_gate x_t)`` (already in ``CLUBlock`` as
``write_mode="gated"``) restored the conjunction in a 3-cell exploratory arm.

This module runs that fix **properly**:

- **Item 1** — the corrected three-family table: baselines + linear-CLU +
  gated-CLU, all at the published-numbers budget with the **symmetric monotone
  LR-rescue** applied to every variant (``primitive-harness §4``). γ=0 for MQAR.
- **Item 3** — the edge search: is there any axis where CLU's physics beats a
  matched, *equally-gated* GRU/SSM?
    - **3a** long-horizon extrapolation (train at T, test at 2T, 4T);
    - **3b** capacity under item load (MQAR kv-sweep, gate + γ=0);
    - **3c** robustness to input noise injected at inference.
- **Item 4** — the honest wall-clock/FLOP multiple of gated CLU vs baselines.

⚠ **Fairness (Item 2).** The gate imports a capability every baseline already
has. This is **levelling, not beating**. Item 1 answers only "is CLU now
competitive". An edge (Item 3) that survives matched gating is the real question.

Nothing here touches the shared slot, the shipped harness defaults, or any
baseline's tuning budget: the CLU-internal ``write_mode`` / ``gamma`` are the
only knobs varied, and every primitive gets the identical LR grid, step budget,
seeds and rescue pass.
"""

import contextlib
import json
import os
from typing import Optional

import equinox as eqx
import jax
import numpy as np

from chlu.config import CHLUConfig, get_default_config
from chlu.experiments.exp_primitive_harness import (
    AddingFamily,
    MQARFamily,
    ParityFamily,
    _eval_step,
    _make_opt,
    _train_step,
    benchmark_cost,
    build_model,
    cfg_seed,
    match_width,
    run_family,
    run_lr_rescue,
)


# --------------------------------------------------------------------------
# cfg-override context (same pattern as exp_primitive_harness._sweep_cell):
# CLU-internal overrides are written onto cfg because build_model reads the CLU
# physics from there, then restored so variants never leak into each other.
# --------------------------------------------------------------------------
@contextlib.contextmanager
def cfg_overrides(cfg, overrides):
    saved = {k: getattr(cfg, k) for k in overrides}
    for k, v in overrides.items():
        setattr(cfg, k, v)
    try:
        yield
    finally:
        for k, v in saved.items():
            setattr(cfg, k, v)


def _family_ctor(name):
    """Map a family name to a constructor of one seq_len argument."""
    if name == "adding":
        return AddingFamily
    if name == "parity":
        return ParityFamily
    raise ValueError(f"gated-write extrapolation supports adding/parity, got {name!r}")


def _clu_gamma_for(cfg, family_name):
    return cfg.gw_mqar_gamma if family_name.startswith("mqar") else cfg.clu_gamma


def item1_variants(cfg, family_name):
    """The six variants of Item 1 for one family.

    Baselines are write-mode-agnostic (run once). The CLU appears twice —
    ``linear`` (the shipped write) and ``gated`` (the fix) — at the SAME gamma
    for that family, so the linear/gated contrast is clean and the only thing
    that differs between the two CLU rows is the write current.
    """
    g = _clu_gamma_for(cfg, family_name)
    return [
        {"label": "mlp", "primitive": "mlp", "overrides": {}},
        {"label": "gru", "primitive": "gru", "overrides": {}},
        {"label": "ssm", "primitive": "ssm", "overrides": {}},
        {"label": "attention", "primitive": "attention", "overrides": {}},
        {"label": "clu_linear", "primitive": "clu",
         "overrides": {"clu_write_mode": "linear", "clu_gamma": g}},
        {"label": "clu_gated", "primitive": "clu",
         "overrides": {"clu_write_mode": "gated", "clu_gamma": g}},
    ]


def _run_variant(cfg, family, variant, rescue, log):
    """LR-select + n_seeds final (+ optional symmetric rescue) for one variant.

    Reuses ``run_family`` / ``run_lr_rescue`` verbatim under the variant's cfg
    override, so the CLU rows get exactly the tuning the baselines get.
    """
    with cfg_overrides(cfg, variant["overrides"]):
        entries = []
        run_family(cfg, family, [variant["primitive"]], entries, log=log)
        entry = entries[0]
        entry["label"] = variant["label"]
        entry["overrides"] = dict(variant["overrides"])
        if rescue and not entry.get("all_diverged"):
            entry = run_lr_rescue(cfg, {family.name: family}, [entry], log=log)[0]
    return entry


# --------------------------------------------------------------------------
# Item 1 — the corrected three-family table
# --------------------------------------------------------------------------
def run_item1(cfg, families, results, log=print):
    for family in families:
        log(f"\n=== Item 1: {family.name} ===")
        for variant in item1_variants(cfg, family.name):
            entry = _run_variant(cfg, family, variant, rescue=True, log=log)
            entry["item"] = "item1"
            entry["seq_len"] = family.seq_len
            results.append(entry)
    return results


# --------------------------------------------------------------------------
# Item 3a — long-horizon extrapolation (train at T, test at 2T, 4T)
# --------------------------------------------------------------------------
def _build_extrap_family(cfg, family_ctor, seq_len, max_len, overrides):
    """A family at ``seq_len`` whose model is built with pos-embedding room for
    ``max_len`` tokens, so the SAME trained block can be evaluated at longer T."""
    with cfg_overrides(cfg, overrides):
        fam = family_ctor(seq_len)
        fam.model_kwargs = dict(fam.model_kwargs)
        fam.model_kwargs["max_len"] = max_len
    return fam


def _train_extrapolate(cfg, primitive, family_ctor, train_T, eval_Ts, width,
                       lr, seed, overrides):
    """Train at ``train_T``; evaluate the SAME block at every ``eval_Ts``.

    The recurrent primitives (GRU/SSM/CLU) carry a fixed-width state and can
    ingest a longer sequence unchanged; attention's learned positional
    embedding beyond ``train_T`` is untrained, which is exactly the founding
    CHLU extrapolation stress. Returns {T: metric} plus divergence.
    """
    max_T = max(eval_Ts)
    key = jax.random.PRNGKey(seed)
    mkey, dkey = jax.random.split(key)
    train_fam = _build_extrap_family(cfg, family_ctor, train_T, max_T, overrides)
    with cfg_overrides(cfg, overrides):
        model = build_model(primitive, cfg, train_fam, width, mkey)
        opt = _make_opt(lr, cfg.grad_clip)
        opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))
        diverged = False
        for i in range(cfg.train_steps):
            bkey = jax.random.fold_in(dkey, i)
            x, y, m = train_fam.jbatch(bkey, cfg.batch_size)
            model, opt_state, loss = _train_step(
                model, opt_state, x, y, m, opt, train_fam
            )
            if not np.isfinite(float(loss)):
                diverged = True
                break
        evals = {}
        for T in eval_Ts:
            efam = family_ctor(T)
            ekey = jax.random.fold_in(jax.random.PRNGKey(seed), 900000 + T)
            xe, ye, me = efam.jbatch(ekey, cfg.eval_batch)
            evals[T] = (
                float("nan") if diverged else float(_eval_step(model, xe, ye, me, efam))
            )
    return {"metrics": evals, "diverged": diverged}


def run_item3a(cfg, results, primitives=None, log=print):
    """Extrapolation edge. LR is selected on the IN-DISTRIBUTION (train_T) score
    at ``tune_steps`` — you may only tune on what you can see — then all eval_Ts
    are reported at the winning LR for n_seeds fresh runs."""
    primitives = primitives or [
        {"label": "clu_gated", "primitive": "clu",
         "overrides": {"clu_write_mode": "gated", "clu_gamma": 0.0}},
        {"label": "gru", "primitive": "gru", "overrides": {}},
        {"label": "ssm", "primitive": "ssm", "overrides": {}},
    ]
    train_T = cfg.gw_extrap_train_T
    eval_Ts = [train_T * m for m in cfg.gw_extrap_mults]
    for fam_name in cfg.gw_extrap_families:
        ctor = _family_ctor(fam_name)
        higher = not fam_name.startswith("adding")
        log(f"\n=== Item 3a: extrapolation {fam_name} train_T={train_T} "
            f"eval_Ts={eval_Ts} ===")
        for variant in primitives:
            width, bp, tp, err = match_width(
                variant["primitive"], cfg,
                _build_extrap_family(cfg, ctor, train_T, max(eval_Ts), variant["overrides"]),
                jax.random.PRNGKey(0),
            )
            # LR selection on in-distribution score (train_T), short budget.
            full = cfg.train_steps
            cfg.train_steps = min(cfg.gw_tune_steps, full)
            tune = []
            for lr in cfg.lr_grid:
                r = _train_extrapolate(cfg, variant["primitive"], ctor, train_T,
                                       [train_T], width, lr, cfg_seed(cfg, 0),
                                       variant["overrides"])
                tune.append((lr, r["metrics"][train_T], r["diverged"]))
            cfg.train_steps = full
            valid = [(lr, v) for lr, v, d in tune if np.isfinite(v)]
            if not valid:
                log(f"    !! all LRs diverged {fam_name}/{variant['label']}")
                continue
            best_lr = (max if higher else min)(valid, key=lambda t: t[1])[0]
            # n_seeds full-length runs, all eval_Ts.
            per_seed = [
                _train_extrapolate(cfg, variant["primitive"], ctor, train_T,
                                   eval_Ts, width, best_lr, cfg_seed(cfg, s),
                                   variant["overrides"])
                for s in range(cfg.n_seeds)
            ]
            by_T = {
                T: [r["metrics"][T] for r in per_seed] for T in eval_Ts
            }
            entry = {
                "item": "item3a_extrapolation", "family": fam_name,
                "label": variant["label"], "overrides": dict(variant["overrides"]),
                "train_T": train_T, "eval_Ts": eval_Ts, "best_lr": best_lr,
                "width": width, "block_params": bp, "total_params": tp,
                "param_err": err, "metric_name": "accuracy" if higher else "mse",
                "metric_by_T": {str(T): float(np.nanmean(v)) for T, v in by_T.items()},
                "std_by_T": {str(T): float(np.nanstd(v)) for T, v in by_T.items()},
                "n_diverged": int(sum(r["diverged"] for r in per_seed)),
            }
            m = entry["metric_by_T"]
            log(f"    {variant['label']:12s} lr={best_lr:g} "
                + " ".join(f"T={T}:{m[str(T)]:.4f}" for T in eval_Ts))
            results.append(entry)
    return results


# --------------------------------------------------------------------------
# Item 3b — capacity under item load (MQAR kv-sweep, gate + gamma=0)
# --------------------------------------------------------------------------
def run_item3b(cfg, results, log=print):
    """Re-run the primitive-harness capacity axis with the CLU gated and gamma=0.

    Variants: gated-CLU (γ=0), linear-CLU (γ=0, the pre-gate reference), GRU,
    SSM, attention. The GRU/SSM are already gated, so this is 'matched, equally
    gated' by construction — the crossover, if it survives, is a physics edge.
    """
    variants = [
        {"label": "gru", "primitive": "gru", "overrides": {}},
        {"label": "ssm", "primitive": "ssm", "overrides": {}},
        {"label": "attention", "primitive": "attention", "overrides": {}},
        {"label": "clu_linear_g0", "primitive": "clu",
         "overrides": {"clu_write_mode": "linear", "clu_gamma": 0.0}},
        {"label": "clu_gated_g0", "primitive": "clu",
         "overrides": {"clu_write_mode": "gated", "clu_gamma": 0.0}},
    ]
    for kv in cfg.mqar_kv_sweep:
        family = MQARFamily(cfg.mqar_seq_len_fixed, kv, cfg.mqar_vocab)
        log(f"\n=== Item 3b: capacity kv={kv} T={cfg.mqar_seq_len_fixed} ===")
        for variant in variants:
            entry = _run_variant(cfg, family, variant, rescue=True, log=log)
            entry["item"] = "item3b_capacity"
            entry["kv"] = kv
            entry["seq_len"] = cfg.mqar_seq_len_fixed
            results.append(entry)
    return results


# --------------------------------------------------------------------------
# Item 3c — robustness to input noise injected at inference
# --------------------------------------------------------------------------
def _noisy_adding_metric(model, family, noise_std, key, n):
    """Adding-problem MSE with Gaussian noise added to the VALUE channel only
    (channel 0) at inference; markers (channel 1) are left intact so the task is
    unchanged, only the observed values are corrupted."""
    x, y, mask = family.jbatch(key, n)
    nkey = jax.random.fold_in(key, 777)
    noise = jax.random.normal(nkey, x.shape) * noise_std
    noise = noise.at[..., 1].set(0.0)  # never corrupt the marker channel
    return float(_eval_step(model, x + noise, y, mask, family))


def run_item3c(cfg, results, log=print):
    """Train clean on the adding problem, evaluate under an input-noise sweep.

    Barrier confinement (Prop 2) predicts a flatter MSE-vs-noise curve for the
    CLU than for a matched gated GRU/SSM. Reported as the degradation curve.
    """
    variants = [
        {"label": "clu_gated", "primitive": "clu",
         "overrides": {"clu_write_mode": "gated", "clu_gamma": cfg.clu_gamma}},
        {"label": "gru", "primitive": "gru", "overrides": {}},
        {"label": "ssm", "primitive": "ssm", "overrides": {}},
    ]
    family = AddingFamily(cfg.adding_seq_len)
    log(f"\n=== Item 3c: robustness (adding T={cfg.adding_seq_len}) "
        f"noise_grid={cfg.gw_noise_grid} ===")
    for variant in variants:
        with cfg_overrides(cfg, variant["overrides"]):
            width, bp, tp, err = match_width(
                variant["primitive"], cfg, family, jax.random.PRNGKey(0)
            )
            # LR-select clean, short budget.
            full = cfg.train_steps
            cfg.train_steps = min(cfg.gw_tune_steps, full)
            from chlu.experiments.exp_primitive_harness import train_one
            tune = [
                train_one(variant["primitive"], cfg, family, width, lr, cfg_seed(cfg, 0))
                for lr in cfg.lr_grid
            ]
            cfg.train_steps = full
            valid = [r for r in tune if np.isfinite(r["metric"])]
            best_lr = min(valid, key=lambda r: r["metric"])["lr"]
            # Retrain at full length per seed, then sweep the eval noise.
            curves = {str(s): {} for s in range(cfg.n_seeds)}
            for s in range(cfg.n_seeds):
                seed = cfg_seed(cfg, s)
                mkey, dkey = jax.random.split(jax.random.PRNGKey(seed))
                model = build_model(variant["primitive"], cfg, family, width, mkey)
                opt = _make_opt(best_lr, cfg.grad_clip)
                opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))
                for i in range(cfg.train_steps):
                    bkey = jax.random.fold_in(dkey, i)
                    x, y, m = family.jbatch(bkey, cfg.batch_size)
                    model, opt_state, _ = _train_step(model, opt_state, x, y, m, opt, family)
                ekey = jax.random.fold_in(jax.random.PRNGKey(seed), 55555)
                for ns in cfg.gw_noise_grid:
                    curves[str(s)][str(ns)] = _noisy_adding_metric(
                        model, family, ns, ekey, cfg.eval_batch
                    )
        mean_curve = {
            str(ns): float(np.nanmean([curves[str(s)][str(ns)] for s in range(cfg.n_seeds)]))
            for ns in cfg.gw_noise_grid
        }
        std_curve = {
            str(ns): float(np.nanstd([curves[str(s)][str(ns)] for s in range(cfg.n_seeds)]))
            for ns in cfg.gw_noise_grid
        }
        log(f"    {variant['label']:12s} lr={best_lr:g} "
            + " ".join(f"σ={ns}:{mean_curve[str(ns)]:.4f}" for ns in cfg.gw_noise_grid))
        results.append({
            "item": "item3c_robustness", "family": family.name,
            "label": variant["label"], "overrides": dict(variant["overrides"]),
            "best_lr": best_lr, "width": width, "block_params": bp,
            "metric_name": "mse", "mse_by_noise": mean_curve, "std_by_noise": std_curve,
        })
    return results


# --------------------------------------------------------------------------
# Item 4 — cost (gated CLU vs baselines)
# --------------------------------------------------------------------------
def run_item4_cost(cfg, log=print):
    """Wall-clock + FLOP multiple of gated CLU vs the baselines, on MQAR kv=4.

    Baselines + linear CLU come from one interleaved round-robin benchmark; the
    gated CLU is timed in a second pass under the gated override (its extra
    ``w_gate`` is the only structural difference). Ratios are quoted vs GRU.
    """
    family = MQARFamily(cfg.mqar_seq_len_fixed, cfg.mqar_kv_fixed, cfg.mqar_vocab)
    base = benchmark_cost(cfg, family, ["gru", "ssm", "attention", "clu"], log=log)
    with cfg_overrides(cfg, {"clu_write_mode": "gated", "clu_gamma": 0.0}):
        gated = benchmark_cost(cfg, family, ["gru", "clu"], log=log)
    out = {"linear": base, "gated_clu": gated["clu"], "gru_ref": gated["gru"]}
    return out


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def _apply_quick(cfg):
    cfg.gw_train_steps = 20
    cfg.gw_tune_steps = 10
    cfg.lr_grid = [1e-3]
    cfg.n_seeds = 1
    cfg.eval_batch = 32
    cfg.target_block_params = 4000
    cfg.adding_seq_len = 32
    cfg.parity_seq_len = 32
    cfg.mqar_seq_len_fixed = 32
    cfg.mqar_kv_sweep = [2, 4]
    cfg.gw_extrap_train_T = 16
    cfg.gw_extrap_mults = [1, 2]
    cfg.gw_noise_grid = [0.0, 0.2]


def run_gated_write(
    config: Optional[CHLUConfig] = None,
    save_dir: Optional[str] = None,
    items: Optional[list] = None,
    families: Optional[list] = None,
    out_name: str = "gated_write.json",
    quick: bool = False,
) -> dict:
    """Run the w22 gated-write performance test.

    ``items`` subsets {"item1", "3a", "3b", "3c", "cost"} (default: all).
    ``families`` subsets Item 1's families {"adding", "parity", "mqar"}.
    """
    if config is None:
        config = get_default_config()
    if save_dir is not None:
        config.project.save_dir = save_dir
    cfg = config.experiment_primitive_harness

    if quick:
        _apply_quick(cfg)

    # The gated-write experiment runs at the published-numbers budget.
    cfg.train_steps = cfg.gw_train_steps
    cfg.tune_steps = cfg.gw_tune_steps

    want = set(items or ["item1", "3a", "3b", "3c", "cost"])
    want_fams = set(families or ["adding", "parity", "mqar"])
    save_dir = config.project.save_dir or "results/"
    results_dir = os.path.join(save_dir, "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, out_name)

    print("=" * 70)
    print("w22 GATED-WRITE PERFORMANCE TEST — CLU vs MLP/GRU/SSM/attention")
    print("=" * 70)
    print(f"items={sorted(want)} families={sorted(want_fams)} "
          f"train_steps={cfg.train_steps} tune_steps={cfg.tune_steps} "
          f"seeds={cfg.n_seeds} lr_grid={cfg.lr_grid} "
          f"budget={cfg.target_block_params}")

    results, cost = [], None

    def _flush():
        summary = {
            "experiment": "gated_write",
            "shared_slot": {
                "d_model": cfg.d_model, "n_layers": cfg.n_layers,
                "target_block_params": cfg.target_block_params,
                "train_steps": cfg.train_steps, "tune_steps": cfg.tune_steps,
                "batch_size": cfg.batch_size, "eval_batch": cfg.eval_batch,
                "lr_grid": list(cfg.lr_grid), "n_seeds": cfg.n_seeds,
                "clu_dt": cfg.clu_dt, "clu_hidden": cfg.clu_hidden,
                "clu_kinetic_mode": cfg.clu_kinetic_mode,
                "clu_potential_type": cfg.clu_potential_type,
                "clu_read_mode": cfg.clu_read_mode,
                "shipped_clu_gamma": cfg.clu_gamma, "mqar_gamma": cfg.gw_mqar_gamma,
            },
            "results": [{k: v for k, v in r.items() if k != "seed_runs"} for r in results],
            "cost": cost,
        }
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        return summary

    if "item1" in want:
        fams = []
        if "adding" in want_fams:
            fams.append(AddingFamily(cfg.adding_seq_len))
        if "parity" in want_fams:
            fams.append(ParityFamily(cfg.parity_seq_len))
        if "mqar" in want_fams:
            fams.append(MQARFamily(cfg.mqar_seq_len_fixed, cfg.mqar_kv_fixed, cfg.mqar_vocab))
        run_item1(cfg, fams, results)
        _flush()
    if "3a" in want:
        run_item3a(cfg, results)
        _flush()
    if "3b" in want:
        run_item3b(cfg, results)
        _flush()
    if "3c" in want:
        run_item3c(cfg, results)
        _flush()
    if "cost" in want:
        cost = run_item4_cost(cfg)
        _flush()

    summary = _flush()
    print(f"\nWrote {out_path}")
    return {"results": results, "cost": cost, "summary": summary, "out_path": out_path}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Gated-write performance test (w22)")
    parser.add_argument("--project")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--items", nargs="+",
                        choices=["item1", "3a", "3b", "3c", "cost"])
    parser.add_argument("--families", nargs="+",
                        choices=["adding", "parity", "mqar"])
    parser.add_argument("--out", default="gated_write.json")
    args = parser.parse_args()
    if args.project:
        from chlu.project import ProjectManager

        pm = ProjectManager()
        config = pm.load(args.project)
        save_dir = str(pm.get_paths(args.project)["plots"])
    else:
        config = get_default_config()
        save_dir = "results"
        os.makedirs(save_dir, exist_ok=True)
    run_gated_write(config=config, save_dir=save_dir, items=args.items,
                    families=args.families, out_name=args.out, quick=args.quick)


if __name__ == "__main__":
    main()
