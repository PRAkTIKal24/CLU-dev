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
        self._jit_batch = {}

    def batch(self, key, n):
        raise NotImplementedError

    def jbatch(self, key, n):
        """JIT-compiled, per-batch-size-cached batch generation.

        Fresh batches are drawn every step, and the generators (especially
        MQAR's without-replacement key/value sampling) cost more per step
        uncompiled than the training step itself. This is pure harness
        overhead: it sits OUTSIDE the timed region, so it never enters the
        reported wall-clock cost of any primitive.
        """
        fn = self._jit_batch.get(n)
        if fn is None:
            fn = jax.jit(lambda k: self.batch(k, n))
            self._jit_batch[n] = fn
        return fn(key)

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
        clu_read_mode=cfg.clu_read_mode,
        clu_write_mode=cfg.clu_write_mode,
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
        x, y, mask = family.jbatch(bkey, cfg.batch_size)
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

    xe, ye, me = family.jbatch(ekey, cfg.eval_batch)
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
            new_mean = float(np.nanmean(vals))
            # Winner's curse guard. The probe that triggered the rescue is a
            # SINGLE seed; the reported number is a 3-seed mean, which can come
            # out worse than the original (observed: adding_T128/mlp probed
            # better on seed 0 but averaged 0.1832 vs 0.1825 over three seeds).
            # Adopt the new LR only if it still wins on the 3-seed mean, so the
            # pass is genuinely monotone -- otherwise it could silently LOWER a
            # baseline, which is the opposite of what it exists to do.
            if improves({"metric": new_mean}):
                log(
                    f"  RESCUED {entry['family']}/{entry['primitive']}: "
                    f"lr {entry['best_lr']:g} -> {win['lr']:g}, "
                    f"{entry['metric_mean']:.4f} -> {new_mean:.4f}"
                )
                new.update(
                    rescued=True,
                    pre_rescue_lr=entry["best_lr"],
                    pre_rescue_metric_mean=entry["metric_mean"],
                    best_lr=win["lr"],
                    metric_mean=new_mean,
                    metric_std=float(np.nanstd(vals)),
                    metric_per_seed=[float(v) for v in vals],
                )
            else:
                log(
                    f"  rescue probe won on 1 seed but LOST on {cfg.n_seeds} "
                    f"({entry['family']}/{entry['primitive']}: "
                    f"{entry['metric_mean']:.4f} kept vs {new_mean:.4f}) — winner's curse"
                )
                new["rescued"] = False
                new["rescue_reverted_mean"] = new_mean
        else:
            new["rescued"] = False
        rescued.append(new)
    return rescued


# --------------------------------------------------------------------------
# w21: the (gamma x read-mode x clu_steps) sweep — CLU-INTERNAL knobs only
# --------------------------------------------------------------------------
def memory_half_life_tokens(gamma, clu_steps=1):
    """Tokens for a lightly-damped oscillator's AMPLITUDE to halve.

    ``velocity_verlet_step`` applies ``p <- (1-gamma) p`` once per Verlet
    sub-step, i.e. ``clu_steps`` times per token. Over a cycle a harmonic mode
    is half kinetic / half potential, so the *energy* decays like
    ``(1-gamma)^clu_steps`` per token and the amplitude like the square root of
    that. Hence half-life = ln2 / (-0.5 * clu_steps * ln(1-gamma))
    ~= 2 ln2 / (gamma * clu_steps) for small gamma -- the ``2 ln2 / gamma``
    figure quoted in the w20 report. Returns ``inf`` at gamma = 0.
    """
    if gamma <= 0:
        return float("inf")
    return float(np.log(2.0) / (-0.5 * clu_steps * np.log1p(-gamma)))


def _sweep_cell(cfg, family, cell, results, out_path, log=print):
    """Run one CLU sweep cell (a dict of CLU-internal overrides) and record it.

    The overrides are written onto ``cfg`` because ``build_model`` reads the CLU
    physics from there; they are restored afterwards so cells never leak into
    each other. Only ``clu_*`` fields are ever touched, so the shared slot
    (d_model, layers, budget, LR grid, steps, seeds) is byte-identical to the
    shipped harness in every cell.
    """
    saved = {k: getattr(cfg, k) for k in cell}
    for k, v in cell.items():
        setattr(cfg, k, v)
    try:
        label = " ".join(f"{k.replace('clu_', '')}={v}" for k, v in cell.items())
        log(f"\n  == CLU cell [{family.name}] {label} ==")
        before = len(results)
        run_family(cfg, family, ["clu"], results, log=log)
        for r in results[before:]:
            r["cell"] = dict(cell)
            r["half_life_tokens"] = memory_half_life_tokens(
                cfg.clu_gamma, cfg.clu_steps
            )
            r["seq_len"] = family.seq_len
    finally:
        for k, v in saved.items():
            setattr(cfg, k, v)
    # Checkpoint after every cell: a multi-hour sweep must never be
    # all-or-nothing (same rule as run_primitive_harness).
    _write_sweep(out_path, cfg, results, complete=False)
    return results


def _write_sweep(out_path, cfg, results, complete):
    summary = {
        "sweep": "gamma_read",
        "complete": complete,
        "shared_slot": {
            "d_model": cfg.d_model,
            "n_layers": cfg.n_layers,
            "target_block_params": cfg.target_block_params,
            "train_steps": cfg.train_steps,
            "tune_steps": cfg.tune_steps,
            "batch_size": cfg.batch_size,
            "eval_batch": cfg.eval_batch,
            "lr_grid": list(cfg.lr_grid),
            "n_seeds": cfg.n_seeds,
            "clu_dt": cfg.clu_dt,
            "clu_hidden": cfg.clu_hidden,
            "clu_kinetic_mode": cfg.clu_kinetic_mode,
            "clu_potential_type": cfg.clu_potential_type,
            "clu_read_mode": cfg.clu_read_mode,
            "clu_write_mode": cfg.clu_write_mode,
        },
        "results": [{k: v for k, v in r.items() if k != "seed_runs"} for r in results],
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def run_gamma_read_sweep(
    config: Optional[CHLUConfig] = None,
    save_dir: Optional[str] = None,
    families: Optional[list] = None,
    items: Optional[list] = None,
    out_name: str = "gamma_read_sweep.json",
    quick: bool = False,
) -> dict:
    """w21 Item 1-3: sweep the CLU's own dissipation / read / integration knobs.

    ``items`` subsets {"gamma", "read", "steps"}:

    - ``gamma``  Item 1: gamma over ``cfg.clu_gamma_sweep`` on every requested
      family, at the shipped read mode and clu_steps.
    - ``read``   Item 2: the 2-D (gamma x read_mode) table. Run at
      ``cfg.clu_steps`` **and** at the largest entry of ``clu_steps_sweep``,
      because at clu_steps=1 the two read modes are provably the same map.
    - ``steps``  Item 3: clu_steps over ``cfg.clu_steps_sweep`` at the best
      gamma found by Item 1 (or the config default if Item 1 was not run),
      adding problem only, with the cost multiple.

    Only CLU-internal knobs vary. The baselines are untouched and remain
    directly comparable to the shipped harness table.
    """
    if config is None:
        config = get_default_config()
    if save_dir is not None:
        config.project.save_dir = save_dir
    cfg = config.experiment_primitive_harness

    if quick:
        cfg.train_steps = 20
        cfg.tune_steps = 10
        cfg.lr_grid = [1e-3]
        cfg.n_seeds = 1
        cfg.eval_batch = 32
        cfg.target_block_params = 4000
        cfg.adding_seq_len = 32
        cfg.parity_seq_len = 32
        cfg.mqar_seq_len_fixed = 32
        cfg.clu_gamma_sweep = [0.0, 0.05]
        cfg.clu_steps_sweep = [1, 2]

    want_items = set(items or ["gamma", "read", "steps"])
    want_fams = set(families or ["adding", "parity", "mqar"])
    save_dir = config.project.save_dir or "results/"
    results_dir = os.path.join(save_dir, "..", "results")
    out_path = os.path.join(results_dir, out_name)

    # The three families at the shipped sweep points: adding T=128,
    # parity T=64, MQAR kv=4 T=128 (mqar_seq_len_fixed / mqar_kv_fixed).
    fams = []
    if "adding" in want_fams:
        fams.append(AddingFamily(cfg.adding_seq_len))
    if "parity" in want_fams:
        fams.append(ParityFamily(cfg.parity_seq_len))
    if "mqar" in want_fams:
        fams.append(MQARFamily(cfg.mqar_seq_len_fixed, cfg.mqar_kv_fixed, cfg.mqar_vocab))

    print("=" * 70)
    print("w21 GAMMA / READ-MODE / CLU_STEPS SWEEP (CLU-internal knobs only)")
    print("=" * 70)
    print(
        f"families={[f.name for f in fams]} items={sorted(want_items)} "
        f"gamma_grid={cfg.clu_gamma_sweep} steps_grid={cfg.clu_steps_sweep} "
        f"train_steps={cfg.train_steps} tune_steps={cfg.tune_steps} "
        f"seeds={cfg.n_seeds} lr_grid={cfg.lr_grid}"
    )

    results = []

    # ---- Item 1: the gamma sweep ----
    if "gamma" in want_items:
        print("\n########## ITEM 1 — gamma sweep ##########")
        for family in fams:
            for g in cfg.clu_gamma_sweep:
                _sweep_cell(cfg, family, {"clu_gamma": g}, results, out_path)

    # ---- Item 2: gamma x read_mode ----
    if "read" in want_items:
        print("\n########## ITEM 2 — gamma x read mode ##########")
        # clu_steps=1 is included so the degeneracy is *measured*, not asserted.
        steps_for_read = sorted({cfg.clu_steps, max(cfg.clu_steps_sweep)})
        for family in fams:
            for k in steps_for_read:
                for mode in cfg.clu_read_mode_sweep:
                    for g in cfg.clu_gamma_sweep:
                        if mode == "endpoint" and k == cfg.clu_steps and "gamma" in want_items:
                            continue  # already measured by Item 1
                        _sweep_cell(
                            cfg, family,
                            {"clu_gamma": g, "clu_read_mode": mode, "clu_steps": k},
                            results, out_path,
                        )

    # ---- Item 3: clu_steps at the best gamma (adding only) ----
    if "steps" in want_items:
        print("\n########## ITEM 3 — clu_steps ##########")
        adding = next((f for f in fams if f.name.startswith("adding")), None)
        if adding is None:
            print("  (skipped: adding family not requested)")
        else:
            best_g = _best_gamma(results, adding.name, cfg.clu_gamma)
            print(f"  best gamma from Item 1 on {adding.name}: {best_g}")
            for k in cfg.clu_steps_sweep:
                if k == cfg.clu_steps and "gamma" in want_items:
                    continue  # already measured at this gamma by Item 1
                _sweep_cell(
                    cfg, adding, {"clu_gamma": best_g, "clu_steps": k},
                    results, out_path,
                )

    summary = _write_sweep(out_path, cfg, results, complete=True)
    print(f"\nWrote {out_path}")
    return {"results": results, "summary": summary, "out_path": out_path}


def _best_gamma(results, family_name, default):
    """Best gamma for ``family_name`` among Item-1 cells (endpoint, clu_steps=1)."""
    cells = [
        r for r in results
        if r["family"] == family_name
        and set(r.get("cell", {})) == {"clu_gamma"}
        and np.isfinite(r.get("metric_mean", float("nan")))
    ]
    if not cells:
        return default
    higher = r_higher_is_better(family_name)
    best = (max if higher else min)(cells, key=lambda r: r["metric_mean"])
    return best["cell"]["clu_gamma"]


def r_higher_is_better(family_name):
    """Metric direction from a family name (accuracy up, MSE down)."""
    return not family_name.startswith("adding")


def benchmark_cost(cfg, family, primitives, n_steps=25, n_rounds=5, log=print):
    """Interleaved wall-clock + FLOPs benchmark at matched parameter budget.

    Wall-clock on a SHARED machine is not a property of the primitive alone: this
    wave ran alongside other agents' jobs (load average 277 at one point), which
    roughly doubled measured step times. Timing each primitive to completion in
    turn would therefore hand whichever primitive ran during a quiet period an
    artificial win.

    Mitigation: cycle through the primitives round-robin, timing a short burst of
    each per round, and report the MEDIAN over rounds plus the spread. Contention
    then hits every primitive in roughly the same proportion, and the ratios
    (which is what the report quotes) are far more stable than the absolutes.
    FLOPs are exact and load-independent, so they are the primary cost metric.
    """
    prepared = {}
    for primitive in primitives:
        width, bp, tp, err = match_width(primitive, cfg, family, jax.random.PRNGKey(0))
        model = build_model(primitive, cfg, family, width, jax.random.PRNGKey(0))
        opt = _make_opt(1e-3, cfg.grad_clip)
        prepared[primitive] = {
            "width": width, "block_params": bp, "total_params": tp, "param_err": err,
            "model": model, "opt": opt,
            "opt_state": opt.init(eqx.filter(model, eqx.is_inexact_array)),
            "times": [],
            "fwd_flops": forward_flops(model, family.jbatch(jax.random.PRNGKey(0), 1)[0]),
        }

    x, y, mask = family.jbatch(jax.random.PRNGKey(1), cfg.batch_size)
    for primitive in primitives:  # warm up compilation outside the timed region
        st = prepared[primitive]
        st["model"], st["opt_state"], loss = _train_step(
            st["model"], st["opt_state"], x, y, mask, st["opt"], family
        )
        jax.block_until_ready(loss)

    for rnd in range(n_rounds):
        for primitive in primitives:
            st = prepared[primitive]
            t0 = time.perf_counter()
            for _ in range(n_steps):
                st["model"], st["opt_state"], loss = _train_step(
                    st["model"], st["opt_state"], x, y, mask, st["opt"], family
                )
            jax.block_until_ready(loss)
            st["times"].append((time.perf_counter() - t0) / n_steps)
        log(f"  round {rnd + 1}/{n_rounds} done")

    out = {}
    for primitive in primitives:
        st = prepared[primitive]
        t = np.array(st["times"])
        out[primitive] = {
            "width": st["width"], "block_params": st["block_params"],
            "total_params": st["total_params"], "param_err": st["param_err"],
            "ms_per_step_median": float(np.median(t) * 1e3),
            "ms_per_step_min": float(t.min() * 1e3),
            "ms_per_step_iqr": float((np.percentile(t, 75) - np.percentile(t, 25)) * 1e3),
            "fwd_flops": st["fwd_flops"],
        }
    ref = out.get("gru", {}).get("ms_per_step_median")
    for primitive in primitives:
        if ref:
            out[primitive]["wallclock_x_gru"] = out[primitive]["ms_per_step_median"] / ref
        fl = out.get("gru", {}).get("fwd_flops")
        if fl and np.isfinite(fl):
            out[primitive]["flops_x_gru"] = out[primitive]["fwd_flops"] / fl
    return out


def benchmark_clu_cells(cfg, family, cells, n_steps=25, n_rounds=5, log=print):
    """Round-robin cost benchmark across CLU-internal configurations (w21 Item 3).

    Same interleaving rationale as ``benchmark_cost`` (this machine is shared, so
    sequential timing is not reproducible): cycle the cells, time a short burst of
    each per round, report the median over rounds. Cost multiples are quoted
    against the FIRST cell, which is by convention the shipped configuration.
    """
    prepared = {}
    order = []
    for cell in cells:
        name = " ".join(f"{k.replace('clu_', '')}={v}" for k, v in cell.items())
        order.append(name)
        saved = {k: getattr(cfg, k) for k in cell}
        for k, v in cell.items():
            setattr(cfg, k, v)
        try:
            width, bp, tp, err = match_width("clu", cfg, family, jax.random.PRNGKey(0))
            model = build_model("clu", cfg, family, width, jax.random.PRNGKey(0))
        finally:
            for k, v in saved.items():
                setattr(cfg, k, v)
        opt = _make_opt(1e-3, cfg.grad_clip)
        prepared[name] = {
            "cell": dict(cell), "width": width, "block_params": bp,
            "total_params": tp, "param_err": err, "model": model, "opt": opt,
            "opt_state": opt.init(eqx.filter(model, eqx.is_inexact_array)),
            "times": [],
            "fwd_flops": forward_flops(model, family.jbatch(jax.random.PRNGKey(0), 1)[0]),
        }

    x, y, mask = family.jbatch(jax.random.PRNGKey(1), cfg.batch_size)
    for name in order:  # warm up compilation outside the timed region
        st = prepared[name]
        st["model"], st["opt_state"], loss = _train_step(
            st["model"], st["opt_state"], x, y, mask, st["opt"], family
        )
        jax.block_until_ready(loss)

    for rnd in range(n_rounds):
        for name in order:
            st = prepared[name]
            t0 = time.perf_counter()
            for _ in range(n_steps):
                st["model"], st["opt_state"], loss = _train_step(
                    st["model"], st["opt_state"], x, y, mask, st["opt"], family
                )
            jax.block_until_ready(loss)
            st["times"].append((time.perf_counter() - t0) / n_steps)
        log(f"  round {rnd + 1}/{n_rounds} done")

    out = {}
    for name in order:
        st = prepared[name]
        t = np.array(st["times"])
        out[name] = {
            "cell": st["cell"], "width": st["width"],
            "block_params": st["block_params"], "total_params": st["total_params"],
            "param_err": st["param_err"],
            "ms_per_step_median": float(np.median(t) * 1e3),
            "ms_per_step_min": float(t.min() * 1e3),
            "ms_per_step_iqr": float((np.percentile(t, 75) - np.percentile(t, 25)) * 1e3),
            "fwd_flops": st["fwd_flops"],
        }
    ref = out[order[0]]
    for name in order:
        out[name]["wallclock_x_ref"] = (
            out[name]["ms_per_step_median"] / ref["ms_per_step_median"]
        )
        if np.isfinite(ref["fwd_flops"]) and ref["fwd_flops"]:
            out[name]["flops_x_ref"] = out[name]["fwd_flops"] / ref["fwd_flops"]
    return out


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
    out_path = os.path.join(results_dir, "exp_primitive_harness.json")

    results = []
    for family in families:
        print(f"\n--- {family.name} ---")
        run_family(cfg, family, cfg.primitives, results)
        # Checkpoint after every family: a multi-hour sweep must never be
        # all-or-nothing. A run interrupted at family k still yields k families
        # of usable, structured results.
        _write_summary(out_path, cfg, results, rescue=False, complete=False)

    if rescue:
        print("\n--- LR RESCUE PASS (non-selected LRs at full length, all primitives) ---")
        results = run_lr_rescue(cfg, {f.name: f for f in families}, results)

    summary = _write_summary(out_path, cfg, results, rescue=rescue, complete=True)
    print(f"\nWrote {out_path}")
    return {"results": results, "summary": summary}


def _write_summary(out_path, cfg, results, rescue, complete):
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
                "read_mode": cfg.clu_read_mode,
                "write_mode": cfg.clu_write_mode,
            },
            "ssm_selective": cfg.ssm_selective,
            "attn_heads": cfg.attn_heads,
        },
        "rescue_pass": rescue,
        "complete": complete,
        "results": [{k: v for k, v in r.items() if k != "seed_runs"} for r in results],
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


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
    parser.add_argument("--benchmark", action="store_true",
                        help="run ONLY the interleaved wall-clock/FLOPs cost benchmark")
    parser.add_argument("--gamma-sweep", action="store_true",
                        help="run ONLY the w21 CLU-internal gamma/read-mode/clu_steps sweep")
    parser.add_argument("--sweep-items", nargs="+", choices=["gamma", "read", "steps"],
                        help="which w21 sweep items to run (default: all)")
    parser.add_argument("--sweep-out", default="gamma_read_sweep.json",
                        help="filename for the w21 sweep JSON")
    parser.add_argument("--clu-steps-benchmark", action="store_true",
                        help="run ONLY the interleaved cost benchmark across clu_steps")
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
    if args.benchmark:
        cfg = config.experiment_primitive_harness
        fam = MQARFamily(cfg.mqar_seq_len_fixed, cfg.mqar_kv_fixed, cfg.mqar_vocab)
        out = benchmark_cost(cfg, fam, cfg.primitives)
        path = os.path.join(save_dir, "..", "results", "primitive_cost_benchmark.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"family": fam.name, "cost": out}, f, indent=2)
        for k, v in out.items():
            print(f"  {k:10s} {v['ms_per_step_median']:7.1f} ms/step "
                  f"(min {v['ms_per_step_min']:6.1f}, IQR {v['ms_per_step_iqr']:5.1f})  "
                  f"{v['fwd_flops']:.3g} FLOP  "
                  f"{v.get('wallclock_x_gru', float('nan')):.2f}x GRU wall / "
                  f"{v.get('flops_x_gru', float('nan')):.2f}x GRU FLOPs")
        print(f"Wrote {path}")
        return
    if args.clu_steps_benchmark:
        cfg = config.experiment_primitive_harness
        fam = AddingFamily(cfg.adding_seq_len)
        cells = [{"clu_steps": k} for k in cfg.clu_steps_sweep]
        cells += [{"clu_steps": k, "clu_read_mode": "trajectory"}
                  for k in cfg.clu_steps_sweep]
        out = benchmark_clu_cells(cfg, fam, cells)
        path = os.path.join(save_dir, "..", "results", "clu_steps_cost_benchmark.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"family": fam.name, "cost": out}, f, indent=2)
        for k, v in out.items():
            print(f"  {k:36s} d_clu={v['width']:4d} {v['ms_per_step_median']:7.1f} ms/step "
                  f"(IQR {v['ms_per_step_iqr']:5.1f})  {v['fwd_flops']:.3g} FLOP  "
                  f"{v['wallclock_x_ref']:.2f}x ref wall / "
                  f"{v.get('flops_x_ref', float('nan')):.2f}x ref FLOPs")
        print(f"Wrote {path}")
        return
    if args.gamma_sweep:
        run_gamma_read_sweep(
            config=config, save_dir=save_dir, families=args.families,
            items=args.sweep_items, out_name=args.sweep_out, quick=args.quick,
        )
        return
    run_primitive_harness(
        config=config, save_dir=save_dir, seed=args.seed,
        quick=args.quick, families=args.families, rescue=args.rescue,
    )


if __name__ == "__main__":
    main()
