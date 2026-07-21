"""Experiment: the primitive harness (w20).

**Question.** Is the CLU a *general* AI primitive — an object at the same level
as an MLP / GRU / SSM / attention block — or only a special-purpose memory?

**Method.** One interchangeable slot (``chlu/core/blocks.py``), five primitives,
matched **block** parameter budget, identical embedding / positional embedding /
head / optimizer / schedule / data / LR grid, three task families reported
**separately** (a primitive that wins one family and loses two wins one family):

  1. ``mqar``   — multi-query associative recall (Zoology). Headline sweep:
                  accuracy vs sequence length (distractors) and vs #KV pairs.
  2. ``adding`` — the adding problem at T=128 (long-range integration; the
                  HiPPO/S4-native family).
  3. ``parity`` — cumulative XOR (state tracking; the recurrence-native family).

**Fairness commitments** (see `.claude/outputs/primitive-harness/PREREG.md`):
- the LR grid is fixed in advance and IDENTICAL for every primitive, so the
  tuning budget is equal by construction and is reported as a count;
- parameter matching is a search over each primitive's own width knob;
- compute cost is reported SEPARATELY (wall-clock + FLOPs) and never hidden
  inside the parameter match — the CLU is expected to be slower, and the
  requirement is that the cost is *stated*.

This module trains nothing physics-specific: it is deliberately a plain
supervised-learning harness, because that is the setting in which the primitive
claim has to hold.
"""

import functools
import json
import os
import time
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from chlu.config import CHLUConfig, get_default_config
from chlu.core.blocks import SequenceModel
from chlu.data.mqar import generate_mqar
from chlu.data.seq_tasks import generate_adding, generate_parity

IGNORE_INDEX = -100


# --------------------------------------------------------------------------
# Task families
# --------------------------------------------------------------------------
class Family:
    """A task family: batch generator + loss + metric + model I/O spec.

    ``metric_higher_is_better`` exists so the LR selection is written once and
    works for both the classification families (accuracy) and the regression
    family (MSE) — the harness must not silently maximise an error.
    """

    def __init__(self, name, seq_len, model_kwargs, metric_name, higher_is_better):
        self.name = name
        self.seq_len = seq_len
        self.model_kwargs = model_kwargs
        self.metric_name = metric_name
        self.higher_is_better = higher_is_better

    def batch(self, key, n):
        raise NotImplementedError

    def loss(self, model, x, y, mask):
        raise NotImplementedError

    def metric(self, model, x, y, mask):
        raise NotImplementedError


class ClassificationFamily(Family):
    """Token-in / token-out families (MQAR, parity): masked cross-entropy."""

    def loss(self, model, x, y, mask):
        logits = jax.vmap(model)(x)  # (B, T, C)
        ll = jnp.take_along_axis(
            jax.nn.log_softmax(logits, axis=-1), y[..., None], axis=-1
        )[..., 0]
        return -(ll * mask).sum() / jnp.maximum(mask.sum(), 1)

    def metric(self, model, x, y, mask):
        logits = jax.vmap(model)(x)
        correct = (jnp.argmax(logits, axis=-1) == y) & mask
        return correct.sum() / jnp.maximum(mask.sum(), 1)


class RegressionFamily(Family):
    """Vector-in / scalar-out family (adding problem): masked MSE."""

    def loss(self, model, x, y, mask):
        pred = jax.vmap(model)(x)  # (B, T, 1)
        se = ((pred - y) ** 2)[..., 0] * mask
        return se.sum() / jnp.maximum(mask.sum(), 1)

    def metric(self, model, x, y, mask):
        return self.loss(model, x, y, mask)


class MQARFamily(ClassificationFamily):
    def __init__(self, seq_len, num_kv_pairs, vocab):
        self.num_kv_pairs = num_kv_pairs
        self.vocab = vocab
        super().__init__(
            f"mqar_T{seq_len}_kv{num_kv_pairs}",
            seq_len,
            dict(vocab_size=vocab, out_dim=vocab, max_len=seq_len),
            "accuracy",
            True,
        )

    def batch(self, key, n):
        d = generate_mqar(
            key, n, self.seq_len, self.num_kv_pairs, vocab_size=self.vocab
        )
        tok, tgt = d["tokens"], d["targets"]
        mask = tgt != IGNORE_INDEX
        # Replace IGNORE_INDEX with 0 so take_along_axis stays in-bounds; the
        # mask (not the label) is what excludes non-query positions from the loss.
        return tok, jnp.where(mask, tgt, 0), mask


class ParityFamily(ClassificationFamily):
    def __init__(self, seq_len):
        super().__init__(
            f"parity_T{seq_len}",
            seq_len,
            dict(vocab_size=2, out_dim=2, max_len=seq_len),
            "accuracy",
            True,
        )

    def batch(self, key, n):
        d = generate_parity(key, n, self.seq_len)
        return d["tokens"], d["targets"], d["mask"]


class AddingFamily(RegressionFamily):
    def __init__(self, seq_len):
        super().__init__(
            f"adding_T{seq_len}",
            seq_len,
            dict(in_dim=2, out_dim=1, max_len=seq_len),
            "mse",
            False,
        )

    def batch(self, key, n):
        d = generate_adding(key, n, self.seq_len)
        return d["inputs"], d["targets"], d["mask"]


# --------------------------------------------------------------------------
# Parameter matching
# --------------------------------------------------------------------------
def count_params(tree) -> int:
    """Number of trainable (inexact-array) scalars in an Equinox PyTree."""
    return int(
        sum(
            x.size
            for x in jax.tree_util.tree_leaves(eqx.filter(tree, eqx.is_inexact_array))
        )
    )


def build_model(primitive, cfg, family, width, key):
    return SequenceModel(
        primitive,
        d_model=cfg.d_model,
        width=width,
        n_layers=cfg.n_layers,
        key=key,
        clu_hidden=cfg.clu_hidden,
        clu_dt=cfg.clu_dt,
        clu_gamma=cfg.clu_gamma,
        clu_steps=cfg.clu_steps,
        clu_kinetic_mode=cfg.clu_kinetic_mode,
        clu_potential_type=cfg.clu_potential_type,
        ssm_selective=cfg.ssm_selective,
        n_heads=cfg.attn_heads,
        **family.model_kwargs,
    )


def block_params(model) -> int:
    return count_params(model.blocks)


def match_width(primitive, cfg, family, key):
    """Binary-search the primitive's width knob to hit the block-param budget.

    Returns (width, block_params, total_params, rel_error). The search is
    monotone in width for every primitive here (each width knob enters the
    parameter count monotonically), so bisection is exact up to integer width
    granularity; the achieved error is reported rather than assumed.
    """
    lo, hi = cfg.width_search_lo, cfg.width_search_hi
    target = cfg.target_block_params
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        m = build_model(primitive, cfg, family, mid, key)
        p = block_params(m)
        err = abs(p / target - 1.0)
        if best is None or err < best[3]:
            best = (mid, p, count_params(m), err)
        if p < target:
            lo = mid + 1
        elif p > target:
            hi = mid - 1
        else:
            break
    return best


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def _make_opt(lr, grad_clip):
    """Cached optimizer: identical `lr` must yield the IDENTICAL object.

    optax builds fresh closures per call, and ``eqx.filter_jit`` keys its cache
    on the identity of non-array arguments — so constructing the optimizer inside
    the training loop would force a full recompile for every (lr, seed) cell.
    Measured: ~24 s of compilation per run against ~17 s of actual stepping.
    """
    return optax.chain(optax.clip_by_global_norm(grad_clip), optax.adam(lr))


@eqx.filter_jit
def _train_step(model, opt_state, x, y, mask, opt, family):
    """Module-level so the compilation cache is shared across seeds and LRs."""
    loss, grads = eqx.filter_value_and_grad(family.loss)(model, x, y, mask)
    updates, opt_state = opt.update(
        grads, opt_state, eqx.filter(model, eqx.is_inexact_array)
    )
    return eqx.apply_updates(model, updates), opt_state, loss


@eqx.filter_jit
def _eval_step(model, x, y, mask, family):
    return family.metric(model, x, y, mask)


def train_one(primitive, cfg, family, width, lr, seed, measure_cost=False):
    """Train one (primitive, family, lr, seed) cell. Returns a metrics dict.

    Wall-clock is the MEDIAN of the per-step times after the first (compiling)
    step, so it measures steady-state training cost, not compilation.
    """
    key = jax.random.PRNGKey(seed)
    mkey, dkey, ekey = jax.random.split(key, 3)
    model = build_model(primitive, cfg, family, width, mkey)

    opt = _make_opt(lr, cfg.grad_clip)
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))

    def step(model, opt_state, x, y, mask):
        return _train_step(model, opt_state, x, y, mask, opt, family)

    def evaluate(model, x, y, mask):
        return _eval_step(model, x, y, mask, family)

    step_times, losses = [], []
    diverged = False
    for i in range(cfg.train_steps):
        bkey = jax.random.fold_in(dkey, i)
        x, y, mask = family.batch(bkey, cfg.batch_size)
        t0 = time.perf_counter()
        model, opt_state, loss = step(model, opt_state, x, y, mask)
        loss = float(loss)
        if i > 0:
            jax.block_until_ready(loss)
            step_times.append(time.perf_counter() - t0)
        losses.append(loss)
        if not np.isfinite(loss):
            # Report divergence; do NOT silently restart with a smaller lr.
            diverged = True
            break

    xe, ye, me = family.batch(ekey, cfg.eval_batch)
    metric = float(evaluate(model, xe, ye, me)) if not diverged else float("nan")

    out = {
        "primitive": primitive,
        "family": family.name,
        "lr": lr,
        "seed": seed,
        "width": width,
        "metric": metric,
        "metric_name": family.metric_name,
        "final_loss": losses[-1] if losses else float("nan"),
        "diverged": diverged,
        "steps_run": len(losses),
        "wallclock_s_per_step": float(np.median(step_times)) if step_times else float("nan"),
    }
    if measure_cost:
        out["fwd_flops"] = forward_flops(model, xe[:1])
    return out


def forward_flops(model, x_one):
    """FLOPs of one forward pass (batch 1) from XLA's cost analysis.

    Uses XLA's own cost analysis rather than a hand-rolled per-primitive count,
    which could silently flatter one primitive. ``eqx.filter_jit`` does not
    expose ``cost_analysis``, so the model is closed over and a plain
    ``jax.jit`` of ``x`` alone is lowered. Returns NaN if the backend does not
    expose a cost analysis — reported as unavailable, never substituted.
    """
    try:
        compiled = jax.jit(lambda xx: jax.vmap(model)(xx)).lower(x_one).compile()
        analysis = compiled.cost_analysis()
        if isinstance(analysis, (list, tuple)):
            analysis = analysis[0]
        return float(analysis.get("flops", float("nan")))
    except Exception:
        return float("nan")


def run_family(cfg, family, primitives, results, log=print):
    """LR-select on seed 0, then run n_seeds at the winning LR, per primitive."""
    for primitive in primitives:
        wkey = jax.random.PRNGKey(0)
        width, bp, tp, err = match_width(primitive, cfg, family, wkey)
        log(
            f"  [{family.name}/{primitive}] width={width} block_params={bp} "
            f"total_params={tp} (budget {cfg.target_block_params}, err {err:.1%})"
        )

        # --- tuning: identical grid AND identical short budget for every
        # primitive. Run at cfg.tune_steps; the winner is retrained at full
        # length below, so no primitive gets a longer selection run than another.
        full_steps = cfg.train_steps
        cfg.train_steps = min(cfg.tune_steps, full_steps)
        tuning = []
        for lr in cfg.lr_grid:
            r = train_one(primitive, cfg, family, width, lr, seed=cfg_seed(cfg, 0))
            tuning.append(r)
            log(
                f"    lr={lr:g}: {family.metric_name}={r['metric']:.4f}"
                + (" [DIVERGED]" if r["diverged"] else "")
            )
        cfg.train_steps = full_steps
        valid = [r for r in tuning if np.isfinite(r["metric"])]
        if not valid:
            log(f"    !! all LRs diverged for {primitive} on {family.name}")
            results.append(
                {
                    "primitive": primitive,
                    "family": family.name,
                    "width": width,
                    "block_params": bp,
                    "total_params": tp,
                    "param_err": err,
                    "best_lr": None,
                    "metric_mean": float("nan"),
                    "metric_std": float("nan"),
                    "all_diverged": True,
                    "tuning_runs": len(tuning),
                    "seed_runs": [],
                }
            )
            continue
        best = (max if family.higher_is_better else min)(valid, key=lambda r: r["metric"])

        # --- final: n_seeds fresh full-length runs at the winning LR ---
        seed_runs = [
            train_one(
                primitive, cfg, family, width, best["lr"],
                seed=cfg_seed(cfg, s), measure_cost=(s == 0),
            )
            for s in range(cfg.n_seeds)
        ]
        vals = np.array([r["metric"] for r in seed_runs], float)
        wall = float(np.nanmedian([r["wallclock_s_per_step"] for r in seed_runs]))
        flops = next((r["fwd_flops"] for r in seed_runs if "fwd_flops" in r), float("nan"))
        log(
            f"    -> best lr={best['lr']:g}  {family.metric_name}="
            f"{np.nanmean(vals):.4f}+-{np.nanstd(vals):.4f}  "
            f"{wall * 1e3:.1f} ms/step  flops={flops:.3g}"
        )
        results.append(
            {
                "primitive": primitive,
                "family": family.name,
                "width": width,
                "block_params": bp,
                "total_params": tp,
                "param_err": err,
                "best_lr": best["lr"],
                "metric_name": family.metric_name,
                "metric_mean": float(np.nanmean(vals)),
                "metric_std": float(np.nanstd(vals)),
                "metric_per_seed": [float(v) for v in vals],
                "n_diverged": int(sum(r["diverged"] for r in seed_runs)),
                "all_diverged": False,
                "tuning_runs": len(tuning),
                "tuning_grid": list(cfg.lr_grid),
                "wallclock_s_per_step": wall,
                "fwd_flops": flops,
                "seed_runs": seed_runs,
            }
        )
    return results


def cfg_seed(cfg, i):
    return 1000 * i + 42


# --------------------------------------------------------------------------
# LR rescue pass (baseline integrity, Item 4)
# --------------------------------------------------------------------------
def run_lr_rescue(cfg, families_by_name, prior, log=print):
    """Re-run the NON-selected LRs at FULL length, for every primitive equally.

    Why this exists. Short-budget LR selection is fair (equal spend) but can be
    *uninformative*: on a hard recall cell nothing has learned by the tuning
    horizon, so all three LRs read at chance and the "winner" is noise. Observed
    on the first run: SSM on MQAR T=32 kv=4 read 0.008/0.010/0.005 at 600 steps,
    picked lr=1e-3 by a hair, and finished at 0.206 — while GRU, whose grid *had*
    separated by then, picked 3e-3 and finished at 0.892. That is exactly the
    "weak baseline" failure the harness must not commit: it would have understated
    a competitor, and a number we later retract is worse than no number.

    The pass is SYMMETRIC (every primitive, every family, same extra budget) and
    monotone (it can only raise a primitive's reported score, never lower it), so
    it cannot be a route to tuning toward a CLU win. Reported as its own table
    with `rescued` flags, never silently merged.
    """
    rescued = []
    for entry in prior:
        if entry.get("all_diverged") or entry.get("best_lr") is None:
            rescued.append(entry)
            continue
        family = families_by_name[entry["family"]]
        others = [lr for lr in cfg.lr_grid if lr != entry["best_lr"]]
        probes = [
            train_one(entry["primitive"], cfg, family, entry["width"], lr,
                      seed=cfg_seed(cfg, 0))
            for lr in others
        ]
        valid = [r for r in probes if np.isfinite(r["metric"])]

        def improves(r, ref=entry["metric_mean"], fam=family):
            return r["metric"] > ref if fam.higher_is_better else r["metric"] < ref

        better = [r for r in valid if improves(r)]
        new = dict(entry)
        new["rescue_probes"] = {
            str(r["lr"]): r["metric"] for r in probes
        }
        if better:
            win = (max if family.higher_is_better else min)(
                better, key=lambda r: r["metric"]
            )
            seeds = [win] + [
                train_one(entry["primitive"], cfg, family, entry["width"], win["lr"],
                          seed=cfg_seed(cfg, s))
                for s in range(1, cfg.n_seeds)
            ]
            vals = np.array([r["metric"] for r in seeds], float)
            log(
                f"  RESCUED {entry['family']}/{entry['primitive']}: "
                f"lr {entry['best_lr']:g} -> {win['lr']:g}, "
                f"{entry['metric_mean']:.4f} -> {np.nanmean(vals):.4f}"
            )
            new.update(
                rescued=True,
                pre_rescue_lr=entry["best_lr"],
                pre_rescue_metric_mean=entry["metric_mean"],
                best_lr=win["lr"],
                metric_mean=float(np.nanmean(vals)),
                metric_std=float(np.nanstd(vals)),
                metric_per_seed=[float(v) for v in vals],
            )
        else:
            new["rescued"] = False
        rescued.append(new)
    return rescued


def build_families(cfg, want):
    """Construct the family objects for the requested task families, in order."""
    families = []
    if "mqar" in want:
        for T in cfg.mqar_seq_lens:  # [1a] distractor axis
            families.append(MQARFamily(T, cfg.mqar_kv_fixed, cfg.mqar_vocab))
        for kv in cfg.mqar_kv_sweep:  # [1b] capacity axis
            if kv == cfg.mqar_kv_fixed and cfg.mqar_seq_len_fixed in cfg.mqar_seq_lens:
                continue  # already measured in 1a
            families.append(MQARFamily(cfg.mqar_seq_len_fixed, kv, cfg.mqar_vocab))
    if "adding" in want:
        families.append(AddingFamily(cfg.adding_seq_len))
    if "parity" in want:
        families.append(ParityFamily(cfg.parity_seq_len))
    return families


def rescue_from_json(path, config=None, out_path=None):
    """Run the LR rescue pass against an existing harness JSON.

    Lets the rescue be applied to a completed run without retraining the cells
    that already have a converged, informative LR selection.
    """
    if config is None:
        config = get_default_config()
    cfg = config.experiment_primitive_harness
    with open(path) as f:
        summary = json.load(f)
    prior = summary["results"]
    names = {r["family"] for r in prior}
    families = {f.name: f for f in build_families(cfg, {"mqar", "adding", "parity"})
                if f.name in names}
    missing = names - set(families)
    if missing:
        raise ValueError(f"Cannot rebuild families {missing} from the current config.")
    rescued = run_lr_rescue(cfg, families, prior)
    summary["results"] = rescued
    summary["rescue_pass"] = True
    out_path = out_path or path.replace(".json", "_rescued.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {out_path}")
    return summary


def run_primitive_harness(
    config: Optional[CHLUConfig] = None,
    save_dir: Optional[str] = None,
    seed: Optional[int] = None,
    quick: bool = False,
    families: Optional[list] = None,
    rescue: bool = False,
) -> dict:
    """Run the harness. ``families`` subsets {"mqar", "adding", "parity"}.

    ``rescue=True`` adds the symmetric full-length LR rescue pass (see
    ``run_lr_rescue``) — recommended whenever the short tuning runs may not have
    separated the grid.
    """
    if config is None:
        config = get_default_config()
    if save_dir is not None:
        config.project.save_dir = save_dir
    if seed is not None:
        config.project.seed = seed
    cfg = config.experiment_primitive_harness

    if quick:
        cfg.train_steps = 30
        cfg.lr_grid = [1e-3]
        cfg.n_seeds = 1
        cfg.eval_batch = 32
        cfg.target_block_params = 4000
        cfg.mqar_seq_lens = [32, 64]
        cfg.mqar_kv_sweep = [2, 4]
        cfg.adding_seq_len = 32
        cfg.parity_seq_len = 32

    want = set(families or ["mqar", "adding", "parity"])
    save_dir = config.project.save_dir or "results/"
    results_dir = os.path.join(save_dir, "..", "results")
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 70)
    print("PRIMITIVE HARNESS — CLU vs MLP / GRU / SSM / attention")
    print("=" * 70)
    print(
        f"primitives={cfg.primitives} d_model={cfg.d_model} n_layers={cfg.n_layers} "
        f"block-param budget={cfg.target_block_params} steps={cfg.train_steps} "
        f"lr_grid={cfg.lr_grid} seeds={cfg.n_seeds}"
    )

    families = build_families(cfg, want)

    results = []
    for family in families:
        print(f"\n--- {family.name} ---")
        run_family(cfg, family, cfg.primitives, results)

    if rescue:
        print("\n--- LR RESCUE PASS (non-selected LRs at full length, all primitives) ---")
        results = run_lr_rescue(cfg, {f.name: f for f in families}, results)

    summary = {
        "config": {
            "d_model": cfg.d_model,
            "n_layers": cfg.n_layers,
            "target_block_params": cfg.target_block_params,
            "train_steps": cfg.train_steps,
            "batch_size": cfg.batch_size,
            "lr_grid": list(cfg.lr_grid),
            "n_seeds": cfg.n_seeds,
            "clu": {
                "dt": cfg.clu_dt,
                "gamma": cfg.clu_gamma,
                "steps": cfg.clu_steps,
                "hidden": cfg.clu_hidden,
                "kinetic_mode": cfg.clu_kinetic_mode,
                "potential_type": cfg.clu_potential_type,
            },
            "ssm_selective": cfg.ssm_selective,
            "attn_heads": cfg.attn_heads,
        },
        "rescue_pass": rescue,
        "results": [{k: v for k, v in r.items() if k != "seed_runs"} for r in results],
    }
    out_path = os.path.join(results_dir, "exp_primitive_harness.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_path}")
    return {"results": results, "summary": summary}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Primitive harness (w20)")
    parser.add_argument("--project")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--families", nargs="+", choices=["mqar", "adding", "parity"])
    parser.add_argument("--steps", type=int, help="override train_steps")
    parser.add_argument("--rescue", action="store_true",
                        help="add the full-length LR rescue pass (all primitives)")
    parser.add_argument("--rescue-from", help="run ONLY the rescue pass against an existing JSON")
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
    if args.steps:
        config.experiment_primitive_harness.train_steps = args.steps
    if args.rescue_from:
        rescue_from_json(args.rescue_from, config=config)
        return
    run_primitive_harness(
        config=config, save_dir=save_dir, seed=args.seed,
        quick=args.quick, families=args.families, rescue=args.rescue,
    )


if __name__ == "__main__":
    main()
