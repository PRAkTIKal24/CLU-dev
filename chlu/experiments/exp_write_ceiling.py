"""Experiment WRITE-CEILING-BREAK (w24): can ANY write operator break the
d-independent learned-capacity ceiling ``K_ceiling ~= 32``?

**The law under attack.** w23 (``dimension-aware-budget``) pinned
``K_learned(d) = min(2^d, K_ceiling)`` with ``K_ceiling ~= 32`` on this exact
harness. Geometry is vindicated for ``d <= 5`` (capacity doubles per dimension,
the designed rate). The ceiling above it is NOT the terrain (a designed
``BallRegisterPotential`` reaches >= 256 under identical numerics), NOT the
parameter count (d=6 K=64 gets *worse* at 2x atoms: 0.855 -> 0.809) and NOT the
dimension (K=64 is unwritable at d=6 AND d=8 despite 4x different geometric
room). What is left is the **write operator**: one static GLOBAL gradient dig
asked to carve K disjoint valleys *jointly*, where the valleys fight over shared
atoms -- and whose write loss reaches ~0 while retrieval already fails, i.e. an
objective **blind to crowding**.

Three levers, one arm each, all on the SAME geometry / retrieval / atom budget as
w23 (read from ``config.experiment_designed_mechanism``, so ``baseline_global``
reproduces the w23 line by construction):

1. ``sequential_masked`` / ``sequential_free`` -- **locality**: write items one at
   a time, so no two valleys are dug by the same gradient step (masked = an item
   may only move its own contiguous atom block; free = one item at a time with all
   atoms movable, which separates "one gradient at a time" from "parameter-space
   masking").
2. ``scale_invariant`` -- **the Head's ablation**: make the per-item write signal
   size-independent (length scales tied to the site separation, which shrinks as
   ``K^{-1/d}``, plus ``sum`` item aggregation). If the ceiling moves under
   rescaling ALONE, the diagnosis is signal dilution, not interference.
3. ``crowding_aware`` -- **the objective**: worst-direction minimum violation,
   nearest-neighbour barrier (the all-pairs mean dilutes the crowding signal by
   ``1/K``) and an explicit atom-encroachment penalty at ``d_safe`` (the
   write-time analogue of the MVC-0 spacing gate).
4. ``combo`` -- all three.

⚠ **Fairness category (N46).** Every arm learns exactly what w22/w23 learned
(atom amplitudes/centers/widths, by gradient descent) and is supplied exactly what
w22/w23 supplied (the target sites, and the partition of the dictionary into K
contiguous blocks). No arm gets formula-placed centers or hand-set widths. A
ceiling that broke only by making the write more DESIGNED would be a scope
collapse, not a win.

Runnable directly:
    uv run python -m chlu.experiments.exp_write_ceiling --quick
or via the CLI: ``chlu exp-write-ceiling [--project N] [--seed I] [--quick]``.
"""

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
from chlu.core.memory_potentials import (
    DesignFreedomPotential,
    atom_write_mask_fn,
)
from chlu.experiments.exp_designed_mechanism import (
    _agg,
    _atoms_for,
    _n_params,
    _replace,
    ball_setup,
    score_cell,
)
from chlu.experiments.goldstone_harness import clu_with_potential
from chlu.training.train_memory import (
    train_memory_landscape,
    trainable_filter,
    write_loss,
)

#: Arms this module knows how to write. ``baseline_global`` is the w23 write.
ARMS = (
    "baseline_global",
    "sequential_masked",
    "sequential_free",
    "scale_invariant",
    "crowding_aware",
    "combo",
)


# ---------------------------------------------------------------------------
# Landscape construction (w23-identical, with an explicit atom count / width so
# the budget-adequacy re-check and the scale-invariant arm can vary them)
# ---------------------------------------------------------------------------


def build_V(d: int, K: int, n_atoms: int, dm, key, init_width: Optional[float] = None):
    """A LEARNED atom dictionary on the (d+1)-dim latent, K groups, ``n_atoms`` atoms.

    Identical to ``exp_designed_mechanism.build_learned_V`` except that the atom
    COUNT and the initial well WIDTH are explicit arguments (the w23 helper derives
    both from the config, which the 2x-atom adequacy re-check and the
    scale-invariant arm need to override).
    """
    return DesignFreedomPotential(
        rung="free_mlp",
        dim=d + 1,
        payloads=jnp.zeros((K,)),  # unused: the free_mlp rung has designed=None
        key=key,
        learned_family="atoms",
        n_atoms=int(n_atoms),
        rbf_init_width=dm.atom_init_width if init_width is None else float(init_width),
        confine=dm.learned_confine,
        atom_depth_init=dm.atom_depth_init,
        atom_groups=K,
        atom_init_scale=dm.atom_init_scale,
    )


# ---------------------------------------------------------------------------
# The write operators (one per arm)
# ---------------------------------------------------------------------------


def arm_write_spec(arm: str, dm, wc, d: int, K: int, sep: float, targets):
    """Resolve one arm into a concrete write specification.

    Returns a dict with ``sequential``/``masked`` (locality lever),
    ``init_width`` (scale lever) and ``loss_kwargs`` (objective levers), so every
    arm is a point in the SAME parameter space and the combination arm is just the
    union. Nothing here is item-specific -- no arm supplies placement.
    """
    lk = dict(
        n_perturb=dm.write_n_perturb,
        sigma_addr=dm.write_sigma_addr,
        sigma_pay=dm.write_sigma_pay,
        margin=dm.write_margin,
        barrier=dm.write_barrier,
        payload_index=d,
    )
    spec = {
        "sequential": False,
        "masked": False,
        "init_width": None,
        "steps": dm.write_steps,
        "loss_kwargs": lk,
    }
    seq = arm in ("sequential_masked", "sequential_free", "combo")
    if seq:
        spec["sequential"] = True
        spec["masked"] = arm != "sequential_free"
        spec["steps"] = wc.seq_steps_per_item
    if arm in ("scale_invariant", "combo"):
        # LENGTH scales tied to the site separation (which shrinks as K^{-1/d});
        # the ENERGY scales (margin, barrier) stay fixed, so the per-item signal
        # magnitude does not shrink with K.
        lk["sigma_addr"] = float(wc.scale_sigma_frac * sep)
        lk["item_agg"] = wc.scale_item_agg
        spec["init_width"] = float(wc.scale_width_frac * sep)
    if arm in ("crowding_aware", "combo"):
        lk["min_agg"] = wc.crowd_min_agg
        lk["barrier_pairs"] = wc.crowd_barrier_pairs
        lk["crowd_weight"] = float(wc.crowd_weight)
        lk["crowd_d_safe"] = float(
            min(
                wc.crowd_d_safe_frac * sep,
                wc.crowd_d_safe_mult
                * (dm.atom_init_width if spec["init_width"] is None else spec["init_width"]),
            )
        )
        # the crowding terms must see the FULL stored set even when the write
        # optimizes one item at a time
        lk["crowd_targets"] = jnp.asarray(targets)
    return spec


def sequential_write(V, targets, key, spec, dm, wc):
    """Write items ONE AT A TIME (the locality lever).

    Each item gets its own Adam state (no moment leakage between items) and, when
    ``spec["masked"]``, its own contiguous atom block -- every other block comes
    out of that item's write bit-identical. The step function is compiled ONCE
    (the target row and the row mask are traced arguments), which is what makes a
    K-item sequential write affordable at K=128.
    """
    spec_filter = trainable_filter(V)
    if spec_filter is None:
        return V, []
    K = int(targets.shape[0])
    n_atoms = int(V.learned.n_atoms)
    if spec["masked"]:
        masks = jnp.stack(
            [V.learned.group_rows(i).astype(jnp.float32) for i in range(K)]
        )
    else:
        masks = jnp.ones((K, n_atoms), dtype=jnp.float32)

    params, static = eqx.partition(V, spec_filter)
    opt = optax.adamw(dm.write_lr, weight_decay=dm.write_weight_decay)
    lk = spec["loss_kwargs"]

    @eqx.filter_jit
    def step_fn(params, state, k, tgt, mask):
        def loss_fn(p):
            return write_loss(eqx.combine(p, static), tgt, k, **lk)

        val, grads = eqx.filter_value_and_grad(loss_fn)(params)
        updates, state = opt.update(grads, state, params)
        updates = atom_write_mask_fn(mask)(updates)
        params = eqx.apply_updates(params, updates)
        return params, state, val

    hist = []
    for _ in range(int(wc.seq_passes)):
        for i in range(K):
            state = opt.init(params)  # fresh moments per item
            tgt = targets[i : i + 1]
            mask = masks[i]
            for _s in range(int(spec["steps"])):
                key, k = jax.random.split(key)
                params, state, val = step_fn(params, state, k, tgt, mask)
                hist.append(float(val))
    return eqx.combine(params, static), hist


def write_arm(arm: str, V, targets, key, dm, wc, d: int, sep: float, steps_mult=1.0):
    """Dispatch one arm's write. Returns ``(V_written, history)``."""
    spec = arm_write_spec(arm, dm, wc, d, int(targets.shape[0]), sep, targets)
    spec["steps"] = int(round(spec["steps"] * steps_mult))
    if spec["sequential"]:
        return sequential_write(V, targets, key, spec, dm, wc)
    return train_memory_landscape(
        V,
        targets,
        key,
        steps=spec["steps"],
        lr=dm.write_lr,
        weight_decay=dm.write_weight_decay,
        loss_kwargs=spec["loss_kwargs"],
    )


def _item_grad_budget(arm: str, K: int, dm, wc, steps_mult=1.0) -> int:
    """Optimizer budget in ITEM-gradient evaluations (the fair compute unit).

    A global step evaluates the loss on all K items; a sequential step on one. So
    ``global = K * write_steps`` and ``sequential = K * passes * steps_per_item``.
    The defaults give the sequential arms HALF the baseline's budget -- no arm can
    win the ceiling by spending more.
    """
    if arm in ("sequential_masked", "sequential_free", "combo"):
        return int(K * wc.seq_passes * round(wc.seq_steps_per_item * steps_mult))
    return int(K * round(dm.write_steps * steps_mult))


# ---------------------------------------------------------------------------
# One (arm, d, K, seed) cell: written + blank + pass/fail (the w23 criterion)
# ---------------------------------------------------------------------------


def evaluate_cell(
    arm: str,
    d: int,
    K: int,
    seed: int,
    dm,
    wc,
    atom_mult: float = 1.0,
    steps_mult: float = 1.0,
    with_blank=True,
):
    """Write + score one landscape, with the leak-immune value-blank control.

    The criterion is w23's, unchanged: mean strict >= ``dm.pass_strict`` AND the
    value blank (a landscape written with ZERO payloads) stays at/below the
    trivial ceiling. ``atom_mult`` / ``steps_mult`` drive the budget-adequacy
    re-checks (N92: a stall under an inadequate budget is not a ceiling).

    ``with_blank`` accepts ``True`` / ``False`` / ``"if_pass"``. The blank control
    exists to make a PASS leak-immune; a cell that misses the strict bar is a fail
    whatever the blank does. ``"if_pass"`` therefore skips the (equally expensive)
    blank write on failing cells — most cells above the ceiling — without weakening
    a single reported pass.
    """
    centers, payloads, targets, sep = ball_setup(d, K, dm)
    n_atoms = int(round(_atoms_for(dm, K, d) * atom_mult))
    k_w, k_b = jax.random.split(jax.random.PRNGKey(seed + 7919), 2)
    spec = arm_write_spec(arm, dm, wc, d, K, sep, targets)

    t0 = time.perf_counter()
    Vw = build_V(d, K, n_atoms, dm, k_w, init_width=spec["init_width"])
    Vw, hist = write_arm(arm, Vw, targets, k_w, dm, wc, d, sep, steps_mult=steps_mult)
    write_seconds = time.perf_counter() - t0
    mw = clu_with_potential(
        Vw, dim=d + 1, kinetic_mode="newtonian_learned", inertia=jnp.ones(d + 1)
    )
    written = score_cell(mw, centers, payloads, dm, d, seed)

    out = {
        "arm": arm,
        "d": d,
        "K": K,
        "seed": seed,
        "site_sep": float(sep),
        "n_atoms": n_atoms,
        "n_learned_params": _n_params(Vw),
        "atom_mult": float(atom_mult),
        "steps_mult": float(steps_mult),
        "item_grad_budget": _item_grad_budget(arm, K, dm, wc, steps_mult),
        "write_loss_final": float(np.mean(hist[-10:])) if hist else float("nan"),
        "write_seconds": float(write_seconds),
        "written": written,
    }
    if with_blank == "if_pass":
        with_blank = written["strict_success_rate"] >= dm.pass_strict
    if not with_blank:
        return out

    blank_pay = jnp.zeros_like(payloads)
    _, _, blank_targets, _ = ball_setup(d, K, dm, payloads=blank_pay)
    Vb = build_V(d, K, n_atoms, dm, k_b, init_width=spec["init_width"])
    Vb, _ = write_arm(
        arm, Vb, blank_targets, k_b, dm, wc, d, sep, steps_mult=steps_mult
    )
    mb = clu_with_potential(
        Vb, dim=d + 1, kinetic_mode="newtonian_learned", inertia=jnp.ones(d + 1)
    )
    blank = score_cell(mb, centers, payloads, dm, d, seed)
    # A blank landscape returns ~0 for every item and so legitimately "retrieves"
    # any item whose stored value happens to lie within payload_tol of 0 (w23).
    pay_np = np.asarray(payloads)
    trivial = float(np.mean(np.abs(pay_np) < dm.payload_tol))
    out["blank"] = blank
    out["value_blank_ok"] = bool(
        blank["strict_success_rate"] <= max(dm.blank_strict_max, trivial + 0.02)
    )
    return out


def _cell_record(cells, dm):
    strict = [c["written"]["strict_success_rate"] for c in cells]
    blanks = [c["value_blank_ok"] for c in cells if "value_blank_ok" in c]
    blanks_ok = all(blanks) if blanks else True
    mean_strict = float(np.mean(strict))
    return {
        "K": cells[0]["K"],
        "d": cells[0]["d"],
        "arm": cells[0]["arm"],
        "seeds": [c["seed"] for c in cells],
        "strict": _agg(strict),
        "selectivity": _agg([c["written"]["selectivity"] for c in cells]),
        "payload_abs_err": _agg([c["written"]["payload_abs_err_mean"] for c in cells]),
        "write_loss_final": _agg([c["write_loss_final"] for c in cells]),
        "write_seconds": _agg([c["write_seconds"] for c in cells]),
        "n_atoms": cells[0]["n_atoms"],
        "n_learned_params": cells[0]["n_learned_params"],
        "item_grad_budget": cells[0]["item_grad_budget"],
        "site_sep": cells[0]["site_sep"],
        "n_blank_seeds": len(blanks),
        "n_value_blank_pass": int(sum(blanks)),
        "passes": bool(mean_strict >= dm.pass_strict and blanks_ok),
    }


def run_cell(arm, d, K, dm, wc, seeds, atom_mult=1.0, steps_mult=1.0, verbose=True):
    """Evaluate one (arm, d, K) cell over ``seeds`` (blank control on the first
    ``wc.blank_seeds`` of them)."""
    cells = []
    for n, s in enumerate(seeds):
        cells.append(
            evaluate_cell(
                arm, d, K, s, dm, wc,
                atom_mult=atom_mult,
                steps_mult=steps_mult,
                with_blank="if_pass" if n < wc.blank_seeds else False,
            )
        )
    rec = _cell_record(cells, dm)
    if rec["passes"] and rec["n_blank_seeds"] == 0:
        # the cell passes on the seed MEAN while seed 0 individually missed the
        # bar, so the lazy blank never fired: no pass is reported without its
        # leak-immunity control.
        cells[0] = evaluate_cell(
            arm, d, K, seeds[0], dm, wc,
            atom_mult=atom_mult, steps_mult=steps_mult, with_blank=True,
        )
        rec = _cell_record(cells, dm)
    if verbose:
        print(
            f"  [{arm}] d={d} K={K:4d} strict={rec['strict']['mean']:.3f}"
            f" (n={len(seeds)}) wloss={rec['write_loss_final']['mean']:.2e}"
            f" atoms={rec['n_atoms']} sep={rec['site_sep']:.3f}"
            f" x{atom_mult:g}atoms x{steps_mult:g}steps"
            f" -> {'PASS' if rec['passes'] else 'fail'}"
            f" [{rec['write_seconds']['mean']:.0f}s]",
            flush=True,
        )
    return rec


def ladder_for(wc, d: int):
    """K rungs walked at dimension ``d`` (from the w23 last-pass rung upward)."""
    start = wc.k_start[wc.dims.index(d)] if d in wc.dims else min(wc.k_ladder)
    return [K for K in wc.k_ladder if start <= K <= wc.k_cap]


# ---------------------------------------------------------------------------
# Item 1/2/3 -- the ladders (one per arm x d), screened then confirmed
# ---------------------------------------------------------------------------


def arm_ladder(arm: str, d: int, dm, wc):
    """Walk the K ladder for one arm at one d; return ``K_learned`` + per-K rows.

    Screened at ``wc.screen_seeds`` seeds, then the decisive cells (the last PASS
    and the first FAIL) are re-run at the full ``wc.seeds`` list -- w23 found the
    write seed-fragile exactly at the 0.9 rung, so the verdict rungs must be
    multi-seed even when the walk is not.
    """
    rows, censored = [], False
    for K in ladder_for(wc, d):
        rec = run_cell(arm, d, K, dm, wc, wc.seeds[: wc.screen_seeds])
        rows.append(rec)
        if not rec["passes"]:
            break
    else:
        censored = True

    confirmed = {}
    if len(wc.seeds) > wc.screen_seeds:
        decisive = []
        passing = [r["K"] for r in rows if r["passes"]]
        if passing:
            decisive.append(max(passing))
        failing = [r["K"] for r in rows if not r["passes"]]
        if failing:
            decisive.append(min(failing))
        for K in decisive:
            rec = run_cell(arm, d, K, dm, wc, wc.seeds)
            confirmed[str(K)] = rec
            for n, r in enumerate(rows):
                if r["K"] == K:
                    rows[n] = rec

    passing = [r["K"] for r in rows if r["passes"]]
    k_star = max(passing, default=0)
    return {
        "arm": arm,
        "d": d,
        "k_learned": k_star,
        "start_rung_failed": bool(not passing),
        "censored": bool(censored and k_star == max(r["K"] for r in rows)),
        "first_fail_K": next((r["K"] for r in rows if not r["passes"]), None),
        "per_K": rows,
        "confirmed_cells": confirmed,
    }


def item1_arms(dm, wc):
    """K_learned(d) under every arm."""
    ladders = []
    for arm in wc.arms:
        for d in wc.dims:
            print(f"[ladder] arm={arm} d={d}", flush=True)
            ladders.append(arm_ladder(arm, d, dm, wc))
    table = {}
    for arm in wc.arms:
        table[arm] = {
            str(lad["d"]): {
                "k_learned": lad["k_learned"],
                "censored": lad["censored"],
                "first_fail_K": lad["first_fail_K"],
            }
            for lad in ladders
            if lad["arm"] == arm
        }
    return {"ladders": ladders, "k_learned_table": table, "dims": list(wc.dims)}


def item2_budget_adequacy(item1, dm, wc):
    """N92 protocol: every first-fail cell re-checked at 2x atoms and 2x steps.

    A stall under an inadequate budget is not a ceiling. The re-check is what
    turns "this arm failed at K" into "this arm has a wall below K".
    """
    out = []
    for lad in item1["ladders"]:
        K = lad["first_fail_K"]
        if K is None:
            continue
        base = next(r for r in lad["per_K"] if r["K"] == K)
        row = {
            "arm": lad["arm"],
            "d": lad["d"],
            "K": K,
            "base_strict": base["strict"]["mean"],
            "base_atoms": base["n_atoms"],
            "base_item_grad_budget": base["item_grad_budget"],
        }
        seeds = wc.seeds[: wc.adequacy_seeds]
        if wc.run_adequacy_recheck:
            rec = run_cell(
                lad["arm"], lad["d"], K, dm, wc, seeds, atom_mult=wc.adequacy_atom_mult
            )
            row["atoms_2x_strict"] = rec["strict"]["mean"]
            row["atoms_2x_passes"] = rec["passes"]
            row["atoms_2x_n_atoms"] = rec["n_atoms"]
        if wc.run_steps_recheck:
            rec = run_cell(
                lad["arm"], lad["d"], K, dm, wc, seeds, steps_mult=wc.steps_recheck_mult
            )
            row["steps_2x_strict"] = rec["strict"]["mean"]
            row["steps_2x_passes"] = rec["passes"]
        row["budget_adequate"] = bool(
            not row.get("atoms_2x_passes", False)
            and not row.get("steps_2x_passes", False)
        )
        out.append(row)
    return {"rows": out, "protocol": "N92: first-fail cell re-run at 2x atoms and 2x steps"}


# ---------------------------------------------------------------------------
# Item 4 -- the verdict
# ---------------------------------------------------------------------------


def item4_verdict(item1, wc):
    """Is the law unclamped, ceiling-raised, or does the ceiling survive?"""
    tab = item1["k_learned_table"]
    base = tab.get("baseline_global", {})
    best_arm, best_k = None, 0
    per_arm = {}
    for arm, by_d in tab.items():
        ks = [v["k_learned"] for v in by_d.values()]
        mx = max(ks, default=0)
        per_arm[arm] = {
            "k_max": mx,
            "beats_baseline_at": [
                d
                for d, v in by_d.items()
                if base.get(d) and v["k_learned"] > base[d]["k_learned"]
            ],
            "matches_2d_everywhere": all(
                v["k_learned"] >= min(2 ** int(d), wc.k_cap) for d, v in by_d.items()
            ),
        }
        if mx > best_k:
            best_arm, best_k = arm, mx
    baseline_max = max([v["k_learned"] for v in base.values()], default=0)
    if best_k > baseline_max and per_arm.get(best_arm, {}).get("matches_2d_everywhere"):
        verdict = "UNCLAMPED"
    elif best_k > baseline_max:
        verdict = "CEILING-RAISED"
    else:
        verdict = "CEILING-SURVIVES"
    return {
        "verdict": verdict,
        "baseline_k_max": baseline_max,
        "best_arm": best_arm,
        "best_k": best_k,
        "per_arm": per_arm,
        "note": (
            "UNCLAMPED = capacity tracks 2^d throughout; CEILING-RAISED = a higher "
            "but still d-independent K'_ceiling; CEILING-SURVIVES = no lever moved it."
        ),
    }


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def _plot(results, save_dir):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    it1 = results.get("item1_arms")
    if not it1:
        return []
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    dims = it1["dims"]
    for arm, by_d in it1["k_learned_table"].items():
        ks = [max(by_d.get(str(d), {}).get("k_learned", 0), 0.5) for d in dims]
        a1.semilogy(dims, ks, "o-", lw=1.8, label=arm)
    a1.semilogy(dims, [2**d for d in dims], "k--", alpha=0.5, label="$2^d$")
    a1.axhline(32, color="r", ls=":", lw=1.0, label="w23 ceiling $\\approx 32$")
    a1.set_xlabel("address dimension $d$")
    a1.set_ylabel("$K_{learned}$ (strict 0.9)")
    a1.set_title(f"write-ceiling: {results.get('item4_verdict', {}).get('verdict', '')}")
    a1.legend(fontsize=7)

    dfoc = dims[-1] if dims else None
    for lad in it1["ladders"]:
        if lad["d"] != dfoc:
            continue
        ks = [r["K"] for r in lad["per_K"]]
        st = [r["strict"]["mean"] for r in lad["per_K"]]
        a2.semilogx(ks, st, "o-", lw=1.8, base=2, label=lad["arm"])
    a2.axhline(0.9, color="k", ls=":", lw=0.9)
    a2.set_xlabel("K")
    a2.set_ylabel("strict success")
    a2.set_title(f"ladder at d={dfoc}")
    a2.legend(fontsize=7)
    fig.tight_layout()
    p = os.path.join(save_dir, "write_ceiling_fig1.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return [p]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_experiment_write_ceiling(
    config: Optional[CHLUConfig] = None,
    save_dir: str = "results",
    models_dir: Optional[str] = None,
    seed: Optional[int] = None,
):
    config = config or get_default_config()
    wc = config.experiment_write_ceiling
    # geometry / retrieval / atom budget are w23's, with the declared read-cost
    # reduction (queries per item) applied identically to every arm.
    dm = _replace(
        config.experiment_designed_mechanism, n_query_per_item=wc.n_query_per_item
    )
    seed = config.project.seed if seed is None else seed
    os.makedirs(save_dir, exist_ok=True)

    bad = [a for a in wc.arms if a not in ARMS]
    if bad:
        raise ValueError(f"unknown arm(s) {bad}; known: {ARMS}")

    results = {
        "seed": seed,
        "config": {
            "write_ceiling": {k: getattr(wc, k) for k in vars(wc)},
            "designed_mechanism": {
                k: getattr(dm, k)
                for k in (
                    "R", "well_width", "well_depth", "payload_kappa", "dt",
                    "gamma_address", "gamma_read", "address_steps", "read_steps",
                    "n_query_per_item", "query_sigma", "payload_tol", "pass_strict",
                    "atoms_per_item", "min_atoms", "min_atoms_base", "min_atoms_c",
                    "atom_init_scale", "atom_init_width", "atom_depth_init",
                    "learned_confine", "write_steps", "write_lr", "write_n_perturb",
                    "write_sigma_addr", "write_sigma_pay", "write_margin",
                    "write_barrier",
                )
            },
        },
    }
    print("[item 1] K_learned(d) per write arm", flush=True)
    results["item1_arms"] = item1_arms(dm, wc)
    print("[item 2] budget adequacy at every first-fail cell", flush=True)
    results["item2_budget_adequacy"] = item2_budget_adequacy(results["item1_arms"], dm, wc)
    results["item4_verdict"] = item4_verdict(results["item1_arms"], wc)

    results_dir = os.path.join(os.path.dirname(os.path.abspath(save_dir)), "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "exp_write_ceiling_metrics.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    try:
        results["figures"] = _plot(results, save_dir)
    except Exception as exc:  # pragma: no cover
        results["figures"] = []
        results["figure_error"] = repr(exc)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    results["metrics_path"] = out_path
    return results


def apply_quick(config: CHLUConfig) -> None:
    """Quick smoke: same code path, tiny sweeps (NOT a scientific run)."""
    wc = config.experiment_write_ceiling
    dm = config.experiment_designed_mechanism
    wc.arms = ["baseline_global", "sequential_masked", "crowding_aware", "combo"]
    wc.dims = [2]
    wc.k_ladder = [2, 4]
    wc.k_start = [2]
    wc.k_cap = 4
    wc.seeds = [0, 1]
    wc.screen_seeds = 1
    wc.blank_seeds = 1
    wc.n_query_per_item = 4
    wc.seq_steps_per_item = 20
    wc.run_adequacy_recheck = True
    wc.adequacy_seeds = 1
    wc.run_steps_recheck = True
    dm.atoms_per_item = 8
    dm.min_atoms = 32
    dm.min_atoms_base = 16
    dm.min_atoms_c = 2.0
    dm.address_steps = 300
    dm.read_steps = 200
    dm.write_steps = 40
    dm.write_n_perturb = 8
    dm.max_total_queries = 64


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Experiment WRITE-CEILING-BREAK: can any write operator break "
        "the d-independent K_ceiling~=32?"
    )
    parser.add_argument("--project", help="Project name to use (default: ./results)")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--quick", action="store_true", help="Quick smoke mode")
    parser.add_argument("--arms", nargs="+", help="Override the swept write arms")
    parser.add_argument("--dims", nargs="+", type=int, help="Override the swept dims")
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
    if args.arms:
        config.experiment_write_ceiling.arms = list(args.arms)
    if args.dims:
        wc = config.experiment_write_ceiling
        keep = [(d, k) for d, k in zip(wc.dims, wc.k_start, strict=False) if d in args.dims]
        wc.dims = [d for d, _ in keep] or list(args.dims)
        wc.k_start = [k for _, k in keep] or [min(wc.k_ladder)] * len(wc.dims)

    res = run_experiment_write_ceiling(
        config=config, save_dir=save_dir, models_dir=models_dir, seed=args.seed
    )
    print(json.dumps(res["item1_arms"]["k_learned_table"], indent=2))
    print(json.dumps(res["item4_verdict"], indent=2))
    print("metrics ->", res["metrics_path"])


if __name__ == "__main__":
    main()


# re-exported for tests / notebooks
__all__ = [
    "ARMS",
    "apply_quick",
    "arm_ladder",
    "arm_write_spec",
    "build_V",
    "evaluate_cell",
    "item1_arms",
    "item2_budget_adequacy",
    "item4_verdict",
    "ladder_for",
    "run_cell",
    "run_experiment_write_ceiling",
    "sequential_write",
    "write_arm",
]
