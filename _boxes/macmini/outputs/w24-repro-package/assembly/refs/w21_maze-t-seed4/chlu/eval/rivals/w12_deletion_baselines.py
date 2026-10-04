"""w12 spoke/deletion-baselines -- THE THREE COMPARATORS EVERY DELETION ROW NEEDS.

SCORED BUDGET: ZERO.  Nothing here touches a held-out test corner.  This module BUILDS
comparators and wires them into the rows-3/4 harness under names the crossover hub calls.
The hub spends the runs; this spoke spends none.

Why this module exists
----------------------
The Head's ruling on deletion language: never "no changes after deletion".  Say
**post-deletion adaptation is LIGHT and DATA-FREE** -- re-eigendecompose the ledger,
carry the address book, **no row of data read** -- against learned baselines that can
only delete by **retraining from scratch without the deleted rows**.  That sentence is
only worth printing if the baseline it is measured against is REAL.  So:

1. ``retrain_without``   the exact reference.  Refit the learned arm from scratch on the
                         fit pool MINUS the deleted rows.  Its cost is printed, never
                         netted: seconds, data rows read, bytes rewritten.
2. ``unlearn_neggrad`` / ``unlearn_finetune``
                         approximate unlearning, ONE PER LEARNED ARM, at **matched
                         compute** -- the same gradient row-visit budget the exact
                         retrain of the same arm was given.  A weakened comparator makes
                         the whole table worthless, so the budget is matched UPWARDS to
                         the exact reference and the realised budget is asserted equal
                         across arms rather than asserted in prose.
3. ``refuse_token``      a transformer with a REFUSE TOKEN in its output, trained on
                         exactly-certified impossible examples.  The ONLY permitted
                         comparator for refusal DISPLAY rows.

THE COST LINE IS PART OF THE DELIVERABLE, NOT A NICETY
------------------------------------------------------
:class:`CostLine` carries **seconds - data rows read - bytes rewritten** and every
comparator returns one.  A comparator with no cost line cannot appear in a table, and
:func:`assert_cost_line_complete` refuses one.

MATCHED COMPUTE, defined once, before any number
------------------------------------------------
There are two readings of "matched compute" for an approximate unlearner, and only one
of them is honest:

* match it to OUR adaptation (which reads no data at all) -- that STARVES the comparator
  and flatters us.  Refused.
* match it to the EXACT reference for the same arm -- the same number of gradient
  row-visits ``steps x batch`` that ``retrain_without`` spent on that arm.  Adopted.

So the unlearner cannot be accused of being under-resourced: it gets everything the exact
retrain got.  What differs is **which rows it may read** -- the deleted rows and a
size-matched retain anchor, never the whole surviving pool -- which is exactly the claim
approximate unlearning makes for itself.  Compute is matched; data access is not, and
both are printed.  ``PINS_W12["matched_compute_fraction"]`` is the dial, default 1.0.

WHAT WOULD PROVE THIS SPOKE WRONG, and what we do about it
----------------------------------------------------------
* An approximate unlearner that cannot be matched on compute => say so.
  :func:`matched_compute_budget` raises rather than silently shrinking a budget.
* The refuse-token transformer fails to separate possible from impossible => **that is a
  finding, not a bug.**  Wave 9's refuser separated at only **+0.047** and a ridge probe
  on the same observation scored at chance.  :func:`fit_refuse_token_transformer`
  reports its separation and AUROC and has no tuning loop at all -- its loss weight,
  learning rate and schedule are declared in :data:`PINS_W12` before any number.

Wiring: the crossover hub calls these BY NAME
---------------------------------------------
Every comparator is returned as a :class:`chlu.eval.rivals.w4_api.RivalRead` -- the same
class ``exp_gen_factor_wells.arm_features`` and ``w4_api.load_rival`` already hand the
rows-3/4 harness, with the same contract::

    values (N, F) float, nan = not requested  ->  (features (N, D) float32, oor (N,))

and every fitted comparator is SAVED under that contract, so::

    from chlu.eval.rivals.w4_api import load_rival
    r = load_rival(seed, "transformer__retrain_without__value-object_hue-0",
                   root="C:/chlu/outputs/w12-deletion-baselines/comparators")
    feats, oor = r(values)

loads it with no w12 import at all.  Nobody reimplements a comparator.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

__all__ = [
    "assert_chlu_in_worktree", "PINS_W12", "LEARNED_ARMS", "COMPARATOR_KINDS",
    "DeletionSpec", "value_spec", "conjunction_spec", "item_spec",
    "partition_fit_rows", "values_from_rows",
    "CostLine", "assert_cost_line_complete",
    "retrain_compute", "matched_compute_budget",
    "matched_arm", "fit_as_trained", "retrain_without",
    "approximate_unlearn", "fit_refuse_token_transformer",
    "certify_impossible", "refusal_row_applies", "build_refusal_sets",
    "separation_report",
    "COMPARATORS", "comparator_name", "save_comparator", "load_comparator",
]


# ---------------------------------------------------------------------------
# 0. The worktree assertion.  Before any number, always.
# ---------------------------------------------------------------------------


def assert_chlu_in_worktree(expected_root: Optional[str] = None) -> dict:
    """Assert ``chlu.__file__`` resolves inside the worktree that is running.

    The shared site-packages carries an editable ``.pth`` pointing at the Head's
    checkout.  If that wins the import, every number produced here belongs to a
    different tree and is void.  This is checked, not hoped for.
    """
    import chlu

    got = Path(chlu.__file__).resolve()
    root = Path(expected_root or os.environ.get("CHLU_ROOT") or Path.cwd()).resolve()
    if root not in got.parents:
        raise AssertionError(
            "chlu resolves to %s, which is NOT inside %s. The shared site-packages "
            ".pth has won the import; every number from this process would belong to "
            "another checkout. Refusing to continue." % (got, root))
    return {"chlu_file": str(got), "expected_root": str(root), "inside": True}


# ---------------------------------------------------------------------------
# 1. The pins.  Declared here, once, BEFORE any number.  Nothing below is tuned.
# ---------------------------------------------------------------------------

PINS_W12 = {
    # -- the exact reference -------------------------------------------------
    # 20 epochs is the LOWER end of the shipped EPOCH_GRID and 1e-3 the middle of the
    # shipped LR_GRID (w4_learners.py). They are INHERITED, not selected here: this
    # spoke runs no tuning loop, because a comparator whose strength we tuned is a
    # comparator whose strength we chose.
    "retrain_epochs": 20,
    "retrain_lr": 1e-3,
    "retrain_weight_decay": 0.0,
    "batch": 256,
    # -- matched compute -----------------------------------------------------
    # 1.0 = the approximate unlearner gets EVERY gradient row-visit the exact retrain
    # of the same arm got. Never below 1.0 by default: a starved comparator is the
    # failure mode this pin exists to make impossible.
    "matched_compute_fraction": 1.0,
    # The retain anchor is size-matched to the deleted set, so the unlearner's data
    # access is 2x the deletion and not a disguised full refit.
    "unlearn_retain_anchor_multiple": 1.0,
    # NegGrad+: ascend on the deleted rows only until their error reaches the level a
    # RANDOM-INIT model of the same architecture gives on the same rows -- a MEASURED
    # ceiling, not a chosen one -- then stop. A hinge, so it cannot diverge.
    "neggrad_alpha": 1.0,
    "neggrad_ceiling": ("measured: MSE of the random-init model of this architecture "
                        "on the deleted rows, before any training"),
    # -- the refuse-token transformer ---------------------------------------
    "refuse_loss_weight": 1.0,
    "refuse_epochs": 20,
    "refuse_lr": 1e-3,
    "refuse_weight_decay": 0.0,
    "refuse_threshold": 0.5,
    # -- the byte pin --------------------------------------------------------
    # 524,288 B at float32 = 131,072 parameters. Every learned comparator is matched
    # to it by w4_learners.match_width, which MEASURES the count by building the model
    # rather than computing it from a formula.
    "byte_pin": 524_288,
    "param_target": 131_072,
    "param_tolerance": 0.10,
    # -- the deletion pins ---------------------------------------------------
    # "red" = object_hue 0.0, factor 2 index 0 -- the value already pinned at
    # exp_gen_factor_wells.py:1348 and used by w8's single design. NOT re-chosen here.
    "value_deletion": {"factor": "object_hue", "value_index": 0},
    # The paper's illustration is "red AND cube". On the REALISED value-holdout fit
    # pool the cube class (shape index 0) is absent entirely -- it is the condition's
    # own held-out shape value -- so "red AND cube" would delete ZERO rows and the row
    # would be a no-op reported as a result. The conjunction pin is therefore
    # object_hue 0.0 AND shape 2.0, which the pool realises. The hub may override it;
    # partition_fit_rows REFUSES a no-op deletion either way.
    "conjunction_deletion": [{"factor": "object_hue", "value_index": 0},
                             {"factor": "shape", "value_index": 2}],
    "item_deletion_fraction": 0.01,
    "item_deletion_rng_seed": 20_260_921,
    # -- provenance ----------------------------------------------------------
    "feature_cache_generation": 2,
    "spoke": "w12/deletion-baselines",
}

#: Every LEARNED arm the wave-12 deletion rows carry. ``table`` is the floor and
#: already has a retrain comparator in ``w4_deletion``; these four are the learned arms.
LEARNED_ARMS = ("mlp", "cont_mlp", "transformer", "lstm")

#: The comparator kinds this module ships, in the order a table prints them.
COMPARATOR_KINDS = ("as_trained", "retrain_without",
                    "unlearn_neggrad", "unlearn_finetune", "refuse_token")


# ---------------------------------------------------------------------------
# 2. The deletion spec, and the partition it induces
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeletionSpec:
    """WHAT is deleted. Three kinds, exactly the three wave 12 prints."""

    kind: str                       # "value" | "conjunction" | "item"
    terms: tuple = ()               # ((factor_index, value_index), ...)
    fraction: float = 0.0           # item deletion only
    rng_seed: int = 0               # item deletion only
    label: str = ""
    human: str = ""

    def name(self) -> str:
        return self.label or self.kind

    def as_dict(self) -> dict:
        return {"kind": self.kind, "label": self.name(), "human": self.human,
                "terms": [[int(f), int(v)] for f, v in self.terms],
                "fraction": float(self.fraction), "rng_seed": int(self.rng_seed)}


def _fidx(grid, factor) -> int:
    return list(grid.names).index(str(factor))


def value_spec(grid, factor=None, value_index=None) -> DeletionSpec:
    """Delete every fit row carrying one (factor, value). The VALUE deletion row."""
    p = PINS_W12["value_deletion"]
    factor = p["factor"] if factor is None else factor
    value_index = p["value_index"] if value_index is None else int(value_index)
    f = _fidx(grid, factor)
    v = float(np.asarray(grid.values[f], np.float64)[int(value_index)])
    return DeletionSpec(kind="value", terms=((f, int(value_index)),),
                        label="value-%s-%d" % (factor, int(value_index)),
                        human="%s = %g" % (factor, v))


def conjunction_spec(grid, terms: Optional[Sequence[dict]] = None) -> DeletionSpec:
    """Delete every fit row carrying a CONJUNCTION. The CONJUNCTION deletion row."""
    terms = PINS_W12["conjunction_deletion"] if terms is None else terms
    tt = tuple((_fidx(grid, t["factor"]), int(t["value_index"])) for t in terms)
    lab = "conj-" + "-".join("%s%d" % (grid.names[f], v) for f, v in tt)
    human = " AND ".join(
        "%s = %g" % (grid.names[f], float(np.asarray(grid.values[f])[v]))
        for f, v in tt)
    return DeletionSpec(kind="conjunction", terms=tt, label=lab, human=human)


def item_spec(fraction=None, rng_seed=None) -> DeletionSpec:
    """Delete a random fraction of fit rows. The ITEM deletion row."""
    fraction = PINS_W12["item_deletion_fraction"] if fraction is None else fraction
    rng_seed = PINS_W12["item_deletion_rng_seed"] if rng_seed is None else rng_seed
    return DeletionSpec(kind="item", fraction=float(fraction), rng_seed=int(rng_seed),
                        label="item-%g" % float(fraction),
                        human="a random %.3g%% of fit rows" % (100.0 * fraction))


def partition_fit_rows(grid, fit_rows: np.ndarray, spec: DeletionSpec) -> dict:
    """``fit_rows`` -> kept / deleted, with the deletion REFUSED if it is a no-op.

    A deletion that removes zero rows is not a deletion, and a table row built on one
    prints a difference of exactly nothing and calls it exactness. Refused here with
    the realised counts in the message, never returned as a clean zero.
    """
    rows = np.asarray(fit_rows, np.int64)
    fi = grid.factor_indices_of(rows)
    if spec.kind == "item":
        rng = np.random.default_rng(int(spec.rng_seed))
        n_del = int(round(float(spec.fraction) * rows.size))
        pick = rng.choice(rows.size, size=max(n_del, 1), replace=False)
        mask = np.zeros(rows.size, bool)
        mask[pick] = True
    else:
        mask = np.ones(rows.size, bool)
        for f, v in spec.terms:
            mask &= (fi[:, int(f)] == int(v))
    if not mask.any():
        realised = {str(grid.names[f]): int((fi[:, f] == v).sum())
                    for f, v in spec.terms}
        raise AssertionError(
            "deletion spec %r removes 0 of %d fit rows -- it is a NO-OP, not a "
            "deletion. Per-term row counts in this pool: %s. Choose a spec the pool "
            "realises; a no-op row must never be printed as a deletion result."
            % (spec.name(), rows.size, realised))
    if mask.all():
        raise AssertionError(
            "deletion spec %r removes ALL %d fit rows; there is no surviving pool to "
            "retrain on." % (spec.name(), rows.size))
    return {
        "spec": spec.as_dict(),
        "kept_rows": rows[~mask], "deleted_rows": rows[mask],
        "n_fit_before": int(rows.size),
        "n_fit_after": int((~mask).sum()),
        "n_deleted": int(mask.sum()),
        "deleted_fraction": float(mask.mean()),
    }


def values_from_rows(grid, rows: np.ndarray) -> np.ndarray:
    """``(N, F)`` requested VALUES for these rows -- the harness's request format."""
    fi = grid.factor_indices_of(np.asarray(rows, np.int64))
    return np.stack([np.asarray(grid.values[k], np.float64)[fi[:, k]]
                     for k in range(len(grid.names))], axis=1)


# ---------------------------------------------------------------------------
# 3. The cost line.  Part of the deliverable, refused if incomplete.
# ---------------------------------------------------------------------------


@dataclass
class CostLine:
    """seconds - data rows read - bytes rewritten, for EVERY arm on EVERY row."""

    seconds: float
    data_rows_read: int             # DISTINCT rows of data the method touched
    bytes_rewritten: int            # bytes of the arm's own state written out
    gradient_row_visits: int = 0    # steps x batch -- the COMPUTE, matched across arms
    param_bytes_changed: Optional[int] = None   # MEASURED diff vs the as-trained arm
    passes_over_data: Optional[float] = None
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "seconds": round(float(self.seconds), 4),
            "data_rows_read": int(self.data_rows_read),
            "bytes_rewritten": int(self.bytes_rewritten),
            "gradient_row_visits": int(self.gradient_row_visits),
            "param_bytes_changed": (None if self.param_bytes_changed is None
                                    else int(self.param_bytes_changed)),
            "passes_over_data": (None if self.passes_over_data is None
                                 else round(float(self.passes_over_data), 4)),
            "note": self.note,
        }

    def one_line(self, who: str) -> str:
        return ("%s: %.2f s - %s data rows read - %s B rewritten (%s gradient "
                "row-visits)" % (who, self.seconds, format(self.data_rows_read, ","),
                                 format(self.bytes_rewritten, ","),
                                 format(self.gradient_row_visits, ",")))


COST_FIELDS = ("seconds", "data_rows_read", "bytes_rewritten")


def assert_cost_line_complete(cost: dict, who: str) -> dict:
    """A comparator with no cost line cannot appear in a table."""
    missing = [k for k in COST_FIELDS if cost.get(k) is None]
    if missing:
        raise AssertionError(
            "%s: cost line is missing %s. The Head's ruling makes the cost line part "
            "of the deliverable: a comparator without seconds, data rows read and "
            "bytes rewritten cannot appear in a table." % (who, missing))
    return cost


# ---------------------------------------------------------------------------
# 4. Matched compute
# ---------------------------------------------------------------------------


def retrain_compute(n_kept: int, epochs=None, batch=None) -> dict:
    """The exact reference's compute, in gradient row-visits. The thing to match."""
    epochs = int(PINS_W12["retrain_epochs"] if epochs is None else epochs)
    batch = int(PINS_W12["batch"] if batch is None else batch)
    steps_per_epoch = max(int(n_kept) // batch, 1)
    steps = steps_per_epoch * epochs
    return {"steps": int(steps), "batch": int(batch),
            "gradient_row_visits": int(steps * batch),
            "epochs": epochs, "steps_per_epoch": int(steps_per_epoch)}


def matched_compute_budget(n_kept: int, fraction=None, epochs=None,
                           batch=None) -> dict:
    """The approximate unlearner's budget, matched to the exact reference.

    Raises:
        AssertionError: if the matched budget rounds to fewer than one step. A budget
            that cannot be honoured is REPORTED, never silently shrunk.
    """
    fraction = float(PINS_W12["matched_compute_fraction"]
                     if fraction is None else fraction)
    ref = retrain_compute(n_kept, epochs, batch)
    steps = int(round(ref["steps"] * fraction))
    if steps < 1:
        raise AssertionError(
            "matched-compute budget rounds to %d steps at fraction %g against a "
            "reference of %d steps. This comparator CANNOT be matched on compute at "
            "this fraction; report that and stop -- do not ship a weakened one and "
            "call it matched." % (steps, fraction, ref["steps"]))
    return {"reference": ref, "fraction": fraction, "steps": steps,
            "batch": int(ref["batch"]),
            "gradient_row_visits": int(steps * ref["batch"]),
            "definition": (
                "matched compute = the SAME gradient row-visits (steps x batch) the "
                "exact retrain of the SAME arm was given. Matched UPWARDS to the exact "
                "reference, never downwards to our own data-free adaptation.")}


def assert_compute_matched(budgets: dict) -> dict:
    """Every arm's unlearner spent the SAME gradient row-visits. Checked, not claimed."""
    vals = {k: int(v["gradient_row_visits"]) for k, v in budgets.items()}
    uniq = sorted(set(vals.values()))
    if len(uniq) != 1:
        raise AssertionError(
            "matched compute FAILED: gradient row-visits differ across arms %s. "
            "Matched compute is the point; a table built on this would compare arms "
            "at different budgets." % vals)
    return {"matched": True, "gradient_row_visits": uniq[0], "per_arm": vals}


# ---------------------------------------------------------------------------
# 5. The learned arms, matched to the byte pin
# ---------------------------------------------------------------------------


def matched_arm(family: str, cardinalities, out_dim: int, key,
                param_target=None, log: Callable = print) -> dict:
    """Builder + width + MEASURED parameter count for one learned family."""
    from . import w4_learners as L

    target = int(PINS_W12["param_target"] if param_target is None else param_target)
    bs = L.builders(cardinalities, int(out_dim), key)
    if family not in bs:
        raise KeyError("no builder for family %r; have %s" % (family, sorted(bs)))
    build, kind = bs[family]
    w, n, err = L.match_width(build, target, tol=float(PINS_W12["param_tolerance"]))
    log("    [%s] matched width=%d, params=%d vs target %d (%+.3f%%), kind=%s"
        % (family, w, n, target, 100.0 * err, kind))
    return {"family": family, "build": build, "in_kind": kind, "width": int(w),
            "param_count": int(n), "param_error": float(err),
            "param_target": target, "param_bytes": int(n) * 4,
            "byte_pin": int(PINS_W12["byte_pin"]), "out_dim": int(out_dim)}


def _requests_for(grid, rows: np.ndarray):
    from ...core.gen_request import dense_request_arrays, requests_from_rows

    return dense_request_arrays(grid, requests_from_rows(grid, np.asarray(rows)))


def _encode(arm: dict, grid, rows: np.ndarray, cardinalities):
    from . import w4_learners as L

    return L.encode_inputs(arm["in_kind"], _requests_for(grid, rows), cardinalities)


def _param_bytes_changed(a, b) -> int:
    """MEASURED: how many BYTES of the arm's parameters actually differ."""
    import equinox as eqx
    import jax

    la = [np.asarray(x) for x in jax.tree_util.tree_leaves(eqx.filter(a, eqx.is_array))]
    lb = [np.asarray(x) for x in jax.tree_util.tree_leaves(eqx.filter(b, eqx.is_array))]
    if len(la) != len(lb):
        raise AssertionError("parameter trees differ in shape; not the same arm")
    n = 0
    for x, y in zip(la, lb):
        n += int((x != y).sum()) * int(x.dtype.itemsize)
    return n


# ---------------------------------------------------------------------------
# 6. COMPARATOR 1 (and the as-trained reference): retrain from scratch WITHOUT
# ---------------------------------------------------------------------------


def _fit_from_scratch(arm: dict, grid, rows: np.ndarray, Y: np.ndarray,
                      cardinalities, key, epochs=None, log: Callable = print):
    from . import w4_learners as L

    epochs = int(PINS_W12["retrain_epochs"] if epochs is None else epochs)
    Xin = _encode(arm, grid, rows, cardinalities)
    t0 = time.time()
    model, info = L.train(arm["build"](arm["width"]), Xin, Y, key, epochs=epochs,
                          lr=float(PINS_W12["retrain_lr"]),
                          wd=float(PINS_W12["retrain_weight_decay"]),
                          batch=int(PINS_W12["batch"]))
    return model, info, float(time.time() - t0)


def fit_as_trained(arm: dict, grid, fit_rows: np.ndarray, Y_fit: np.ndarray,
                   cardinalities, key, epochs=None, log: Callable = print) -> dict:
    """The arm BEFORE any deletion -- the reference both other comparators move from.

    This is also the "baselines as-trained" column the wave-12 plan asks for beside
    "baselines retrained-without".
    """
    model, info, secs = _fit_from_scratch(arm, grid, fit_rows, Y_fit, cardinalities,
                                          key, epochs, log)
    comp = retrain_compute(int(np.asarray(fit_rows).size), epochs)
    cost = CostLine(seconds=secs, data_rows_read=int(np.asarray(fit_rows).size),
                    bytes_rewritten=int(arm["param_count"]) * 4,
                    gradient_row_visits=comp["gradient_row_visits"],
                    param_bytes_changed=None,
                    passes_over_data=float(comp["epochs"]),
                    note=("the arm as trained on the FULL fit pool; it has not "
                          "deleted anything and its cost is the build cost, printed "
                          "so the deletion costs below are readable against it"))
    log("    " + cost.one_line("%s as-trained" % arm["family"]))
    return {"kind": "as_trained", "model": model, "arm": arm, "fit": info,
            "cost": assert_cost_line_complete(cost.as_dict(),
                                              "%s/as_trained" % arm["family"])}


def retrain_without(arm: dict, grid, part: dict, gather: Callable, cardinalities,
                    key, epochs=None, log: Callable = print) -> dict:
    """COMPARATOR 1 -- the EXACT reference: refit from scratch on the kept rows.

    Cost printed, never netted: seconds, data rows read (= every kept row, once per
    epoch), bytes rewritten (= every parameter byte of the arm).
    """
    kept = np.asarray(part["kept_rows"], np.int64)
    t_read = time.time()
    Y = gather(kept)
    read_s = float(time.time() - t_read)
    model, info, secs = _fit_from_scratch(arm, grid, kept, Y, cardinalities, key,
                                          epochs, log)
    comp = retrain_compute(int(kept.size), epochs)
    cost = CostLine(
        seconds=secs + read_s, data_rows_read=int(kept.size),
        bytes_rewritten=int(arm["param_count"]) * 4,
        gradient_row_visits=comp["gradient_row_visits"],
        param_bytes_changed=int(arm["param_count"]) * 4,
        passes_over_data=float(comp["epochs"]),
        note=("EXACT REFERENCE: the deleted rows were never in this pool. Every "
              "parameter byte is new, so bytes_rewritten is the whole arm; "
              "data_rows_read is every surviving row and it is read once per epoch "
              "(passes_over_data). %.2f s of the total is the feature read."
              % read_s))
    log("    " + cost.one_line("%s retrain_without %s"
                               % (arm["family"], part["spec"]["label"])))
    return {"kind": "retrain_without", "model": model, "arm": arm, "fit": info,
            "compute": comp,
            "cost": assert_cost_line_complete(
                cost.as_dict(), "%s/retrain_without" % arm["family"])}


# ---------------------------------------------------------------------------
# 7. COMPARATOR 2: approximate unlearning, at MATCHED COMPUTE
# ---------------------------------------------------------------------------


def _run_unlearn(model, batches, key, steps: int, lr: float, wd: float,
                 loss_fn: Callable, log: Callable = print, tag: str = ""):
    """Adam + warmup-cosine, the SAME optimiser ``w4_learners.train`` uses.

    ``batches`` is ``step -> dict`` of numpy arrays; ``loss_fn(model, batch)`` returns a
    scalar.  Written out here rather than reusing ``train`` because the unlearning
    objectives are not "MSE to a target on one array".
    """
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import optax

    warm = max(int(steps) // 20, 1)
    sched = optax.warmup_cosine_decay_schedule(
        init_value=lr * 0.1, peak_value=lr, warmup_steps=warm,
        decay_steps=max(int(steps), warm + 1), end_value=lr * 0.01)
    opt = optax.adamw(sched, weight_decay=wd)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def step_fn(m, st, batch):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(m, batch)
        upd, st = opt.update(grads, st, eqx.filter(m, eqx.is_array))
        return eqx.apply_updates(m, upd), st, loss

    first = last = None
    for i in range(int(steps)):
        b = {k: jnp.asarray(v) for k, v in batches(i).items()}
        model, opt_state, loss = step_fn(model, opt_state, b)
        if first is None:
            first = float(loss)
        last = float(loss)
        if log is not None and steps >= 20 and (i + 1) % max(steps // 4, 1) == 0:
            log("      [%s] step %d/%d loss %.5f" % (tag, i + 1, steps, last))
    return model, {"steps": int(steps), "lr": float(lr), "weight_decay": float(wd),
                   "loss_first": first, "loss_final": last}


def approximate_unlearn(base: dict, arm: dict, grid, part: dict, gather: Callable,
                        cardinalities, key, method: str = "neggrad",
                        fraction=None, epochs=None, log: Callable = print) -> dict:
    """COMPARATOR 2 -- approximate unlearning, ONE PER LEARNED ARM, MATCHED COMPUTE.

    ``method``:
      ``neggrad``   NegGrad+: descend on a size-matched RETAIN ANCHOR while ascending
                    on the deleted rows, the ascent hinged at the MEASURED random-init
                    error on those rows so it cannot diverge.
      ``finetune``  fine-tune on the retain anchor alone -- unlearning by catastrophic
                    forgetting, the cheapest thing anyone calls unlearning.

    Both are given the SAME gradient row-visits the exact retrain of the SAME arm was
    given.  What they do NOT get is the surviving pool: they read the deleted rows and
    a size-matched anchor and nothing else, and that is what the cost line prints.
    """
    import jax.numpy as jnp

    deleted = np.asarray(part["deleted_rows"], np.int64)
    kept = np.asarray(part["kept_rows"], np.int64)
    rng = np.random.default_rng(int(PINS_W12["item_deletion_rng_seed"]) + 17)
    n_anchor = min(int(round(float(PINS_W12["unlearn_retain_anchor_multiple"])
                             * deleted.size)), int(kept.size))
    anchor = kept[rng.choice(kept.size, size=n_anchor, replace=False)]

    budget = matched_compute_budget(int(kept.size), fraction, epochs)
    batch = int(budget["batch"])
    steps = int(budget["steps"])

    t_read = time.time()
    Yd = gather(deleted)
    Ya = gather(anchor)
    read_s = float(time.time() - t_read)
    Xd = _encode(arm, grid, deleted, cardinalities)
    Xa = _encode(arm, grid, anchor, cardinalities)

    # The MEASURED ceiling: what a random-init model of this architecture scores on
    # the deleted rows. Not a chosen number -- the arm is built and read.
    from . import w4_learners as L
    ceil = float(np.mean((L.predict(arm["build"](arm["width"]), Xd) - Yd) ** 2))

    half = max(batch // 2, 1)
    alpha = float(PINS_W12["neggrad_alpha"])

    def batches(i):
        r = np.random.default_rng(20_260_921 + i)
        ia = r.integers(0, Xa.shape[0], size=(batch if method == "finetune" else half))
        out = {"xa": Xa[ia], "ya": Ya[ia]}
        if method == "neggrad":
            idl = r.integers(0, Xd.shape[0], size=half)
            out["xd"] = Xd[idl]
            out["yd"] = Yd[idl]
        return out

    if method == "finetune":
        def loss_fn(m, b):
            import jax
            return jnp.mean((jax.vmap(m)(b["xa"]) - b["ya"]) ** 2)
    elif method == "neggrad":
        def loss_fn(m, b):
            import jax
            keep = jnp.mean((jax.vmap(m)(b["xa"]) - b["ya"]) ** 2)
            forget = jnp.mean((jax.vmap(m)(b["xd"]) - b["yd"]) ** 2)
            # hinge: push the deleted rows' error UP to the random-init level, no more
            return keep + alpha * jnp.maximum(ceil - forget, 0.0)
    else:
        raise ValueError("unknown unlearning method %r; have neggrad, finetune"
                         % method)

    t0 = time.time()
    model, info = _run_unlearn(base["model"], batches, key, steps,
                               float(PINS_W12["retrain_lr"]),
                               float(PINS_W12["retrain_weight_decay"]), loss_fn,
                               log=log, tag="%s/%s" % (arm["family"], method))
    secs = float(time.time() - t0)

    changed = _param_bytes_changed(base["model"], model)
    rows_read = int(anchor.size + (deleted.size if method == "neggrad" else 0))
    cost = CostLine(
        seconds=secs + read_s, data_rows_read=rows_read,
        bytes_rewritten=int(arm["param_count"]) * 4,
        gradient_row_visits=int(budget["gradient_row_visits"]),
        param_bytes_changed=int(changed),
        passes_over_data=float(budget["gradient_row_visits"]) / max(rows_read, 1),
        note=("APPROXIMATE UNLEARNING at MATCHED COMPUTE: the same %s gradient "
              "row-visits the exact retrain of this arm was given, spent on %d "
              "distinct rows instead of %d. bytes_rewritten is the whole arm; "
              "param_bytes_changed is MEASURED against the as-trained checkpoint. "
              "%.2f s of the total is the feature read."
              % (format(budget["gradient_row_visits"], ","), rows_read,
                 int(kept.size), read_s)))
    log("    " + cost.one_line("%s unlearn_%s %s"
                               % (arm["family"], method, part["spec"]["label"])))
    return {"kind": "unlearn_%s" % method, "model": model, "arm": arm, "fit": info,
            "compute": budget, "neggrad_ceiling_measured": ceil,
            "n_anchor": int(anchor.size), "n_deleted": int(deleted.size),
            "cost": assert_cost_line_complete(
                cost.as_dict(), "%s/unlearn_%s" % (arm["family"], method))}


# ---------------------------------------------------------------------------
# 8. COMPARATOR 3: the refuse-token transformer
# ---------------------------------------------------------------------------

#: A request is IMPOSSIBLE for one of exactly these reasons, and each is decided by an
#: integer / set-membership test on the factor grid. No tolerance, no model, no float
#: comparison other than exact grid-value equality.
IMPOSSIBLE_REASONS = ("off_grid_value", "out_of_range", "deleted_value",
                      "deleted_conjunction", "never_held")


def certify_impossible(values: np.ndarray, grid, held_bitmap, deleted_terms=(),
                       deleted_tuples=None) -> dict:
    """EXACT certification: which requests the post-deletion pool can never answer.

    ``held_bitmap``   per factor, a boolean over that factor's values: does the KEPT
                      pool hold that value at all (``exp_gen_factor_wells.
                      value_presence_bitmap`` on the kept rows).
    ``deleted_terms`` the (factor, value) terms of the deletion -- a VALUE deletion
                      makes every request naming that value impossible.
    ``deleted_tuples`` for a CONJUNCTION deletion, the set of deleted (f,v) term pairs
                      that must ALL be present for a request to be refused.

    Every test is set membership on integer indices or exact equality against the
    factor grid.  Certification is not a measurement and it is not a model.
    """
    V = np.asarray(values, np.float64)
    n, F = V.shape
    present = ~np.isnan(V)
    imp = np.zeros(n, bool)
    reason = np.array([""] * n, dtype=object)

    def _mark(m, why):
        fresh = m & ~imp
        reason[fresh] = why
        imp[fresh] = True

    idx = np.full((n, F), -1, np.int64)
    hits = np.zeros((n, F), bool)
    for k in range(F):
        gv = np.asarray(grid.values[k], np.float64)
        col = V[:, k]
        pk = present[:, k]
        hit = np.zeros(n, bool)
        pos = np.zeros(n, np.int64)
        if pk.any():
            eq = (col[pk][:, None] == gv[None, :])
            hit[pk] = eq.any(axis=1)
            pos[pk] = np.argmax(eq, axis=1)
        idx[:, k] = np.where(hit, pos, -1)
        hits[:, k] = hit

    # ORDER MATTERS and it is declared: after a VALUE deletion a request naming the
    # deleted value is impossible for BOTH reasons ("deleted_value" and "never_held").
    # The DELETION reason is marked first, because that is the reason the DISPLAY row
    # is about; reporting it as "never_held" would hide which mechanism refused it.
    if deleted_terms:
        dt = [(int(f), int(v)) for f, v in deleted_terms]
        m = np.ones(n, bool)
        for f, v in dt:
            m &= present[:, f] & (idx[:, f] == v)
        _mark(m, "deleted_value" if len(dt) == 1 else "deleted_conjunction")
    if deleted_tuples is not None:
        _mark(np.asarray(deleted_tuples, bool), "deleted_conjunction")

    for k in range(F):
        gv = np.asarray(grid.values[k], np.float64)
        col, pk, hit = V[:, k], present[:, k], hits[:, k]
        _mark(pk & ((col < gv[0]) | (col > gv[-1])), "out_of_range")
        _mark(pk & ~hit, "off_grid_value")
        bm = np.asarray(held_bitmap[k], bool)
        _mark(pk & hit & ~bm[np.clip(idx[:, k], 0, bm.size - 1)], "never_held")
    return {"impossible": imp, "reason": reason, "grid_index": idx,
            "certifier": ("exact: integer index membership against the factor grid and "
                          "the KEPT pool's value-presence bitmap, plus the deletion's "
                          "own terms. No tolerance and no model anywhere in it.")}


def refusal_row_applies(grid, part: dict, held_bitmap) -> dict:
    """Does this deletion leave a HOLE that a refusal row can be ABOUT? Measured.

    MEASURED, not assumed, and the answer is not always yes.  A VALUE deletion clears a
    bit in the value-presence bitmap and a CONJUNCTION deletion punches a hole in the
    coupling table, so requests naming them become exactly-certified impossible.  An
    ITEM deletion (a random fraction of rows) clears **nothing**: every deleted row's
    factor tuple is still composable from values the pool still holds, the bitmap is
    unchanged, and no request on earth becomes impossible.

    So an item-deletion row has NO refusal DISPLAY row, and that is a property of the
    deletion rather than a failure of the comparator.  Refusing a deleted ITEM would
    need a per-tuple deletion table -- which is a STORE bolted to a generator, exactly
    the objection ``exp_gen_refuse`` already makes one level coarser about the
    pair-deletion table.  The crossover hub reads ``applies`` and prints a DECLARED
    NOT-RUN with this measurement, never a zero.
    """
    terms = [(int(f), int(v)) for f, v in part["spec"]["terms"]]
    V = values_from_rows(grid, part["deleted_rows"])
    cert = certify_impossible(V, grid, held_bitmap, terms)
    n = int(cert["impossible"].sum())
    reasons = sorted(set(str(x) for x in cert["reason"][cert["impossible"]].tolist()))
    return {
        "applies": bool(n > 0),
        "n_deleted_requests_tested": int(V.shape[0]),
        "n_certified_impossible": n,
        "reasons": reasons,
        "why": ("a refusal row needs at least one request the post-deletion pool can "
                "never answer. %d of %d requests built from the deleted rows are "
                "certified impossible%s."
                % (n, int(V.shape[0]),
                   "" if n else (" -- a %s deletion clears no (factor, value) bit and "
                                 "punches no conjunction hole, so nothing becomes "
                                 "impossible and there is no DISPLAY row to print"
                                 % part["spec"]["kind"]))),
    }


def build_refusal_sets(grid, part: dict, held_bitmap, n_per_class: int = 4096,
                       rng_seed: int = 20_260_922,
                       include_out_of_range: bool = False) -> dict:
    """Train / eval sets for the refuser, DISJOINT by construction and checked.

    POSSIBLE   = requests the kept pool realises.
    IMPOSSIBLE = requests the exact certifier refuses.

    ``include_out_of_range`` is FALSE by default, and the reason is structural rather
    than convenient.  ``w4_api`` documents it for this family already: a ``tokens``
    input is a CLOSED vocabulary of grid classes, so an off-grid value is encoded as
    its nearest grid class and the transformer's input **cannot see** that the request
    was out of range at all.  Scoring the refuser on a leg its own encoding cannot
    represent is weakening a comparator through our own code, so the PRIMARY set is
    the post-deletion impossibles -- which is exactly what the DISPLAY rows ask -- and
    the out-of-range leg is run separately as a DECLARED DIAGNOSTIC so the structural
    limit is a number rather than a paragraph.  Out-of-range refusal already has its
    own exact comparator: the 8-byte value-presence bitmap plus bracket check
    (``exp_gen_factor_wells.accept_rule``), which scores it at 1.0000 for free.
    """
    rng = np.random.default_rng(int(rng_seed))
    kept = np.asarray(part["kept_rows"], np.int64)
    deleted = np.asarray(part["deleted_rows"], np.int64)
    terms = [(int(f), int(v)) for f, v in part["spec"]["terms"]]

    pos = values_from_rows(grid, kept[rng.choice(
        kept.size, size=min(2 * n_per_class, kept.size), replace=False)])
    neg_del = values_from_rows(grid, deleted[rng.choice(
        deleted.size, size=min(n_per_class, deleted.size), replace=False)])

    # out-of-range: push ONE requested factor past its grid end. Certified by the grid
    # alone, exactly as `accept_rule`'s out-of-range leg is.
    base = values_from_rows(grid, kept[rng.choice(
        kept.size, size=min(n_per_class, kept.size), replace=False)])
    oor = base.copy()
    which = rng.integers(0, len(grid.names), size=oor.shape[0])
    for k in range(len(grid.names)):
        gv = np.asarray(grid.values[k], np.float64)
        span = float(gv[-1] - gv[0])
        m = which == k
        hi = rng.random(int(m.sum())) < 0.5
        bump = np.where(hi, gv[-1] + 0.1 * span + 1e-6, gv[0] - 0.1 * span - 1e-6)
        oor[m, k] = bump

    if not include_out_of_range:
        oor = oor[:0]
    neg = np.concatenate([neg_del, oor], axis=0)
    cert_pos = certify_impossible(pos, grid, held_bitmap, terms)
    cert_neg = certify_impossible(neg, grid, held_bitmap, terms)
    if cert_pos["impossible"].any():
        raise AssertionError(
            "%d of %d POSSIBLE requests were certified impossible; the certifier and "
            "the kept pool disagree and no refusal number may be read off this."
            % (int(cert_pos["impossible"].sum()), pos.shape[0]))
    if not cert_neg["impossible"].all():
        raise AssertionError(
            "%d of %d IMPOSSIBLE requests were NOT certified impossible; they would "
            "train the refuser on a label the certifier does not support."
            % (int((~cert_neg["impossible"]).sum()), neg.shape[0]))

    def _split(A):
        k = A.shape[0] // 2
        p = rng.permutation(A.shape[0])
        return A[p[:k]], A[p[k:]]

    ptr, pev = _split(pos)
    ntr, nev = _split(neg)
    seen = set(map(tuple, np.round(np.concatenate([ptr, ntr]), 9).tolist()))
    leak = sum(1 for r in np.round(np.concatenate([pev, nev]), 9).tolist()
               if tuple(r) in seen)
    return {
        "train_values": np.concatenate([ptr, ntr]),
        "train_labels": np.concatenate([np.zeros(ptr.shape[0]),
                                        np.ones(ntr.shape[0])]).astype(np.float32),
        "eval_values": np.concatenate([pev, nev]),
        "eval_labels": np.concatenate([np.zeros(pev.shape[0]),
                                       np.ones(nev.shape[0])]).astype(np.float32),
        "eval_reason": np.concatenate([
            certify_impossible(pev, grid, held_bitmap, terms)["reason"],
            certify_impossible(nev, grid, held_bitmap, terms)["reason"]]),
        "n_possible": int(pos.shape[0]), "n_impossible": int(neg.shape[0]),
        "n_impossible_deleted": int(neg_del.shape[0]),
        "n_impossible_out_of_range": int(oor.shape[0]),
        "include_out_of_range": bool(include_out_of_range),
        "train_eval_overlap": int(leak),
        "certifier": cert_neg["certifier"],
    }


def _auroc(score: np.ndarray, label: np.ndarray) -> float:
    s = np.asarray(score, np.float64)
    y = np.asarray(label, np.float64) > 0.5
    if y.all() or not y.any():
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(s.size, np.float64)
    ranks[order] = np.arange(1, s.size + 1, dtype=np.float64)
    # average ranks over ties
    su = np.sort(s)
    i = 0
    while i < su.size:
        j = i
        while j + 1 < su.size and su[j + 1] == su[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    npos, nneg = int(y.sum()), int((~y).sum())
    return float((ranks[y].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def separation_report(score: np.ndarray, label: np.ndarray, reason=None,
                      threshold=None) -> dict:
    """The number w9 asks for by name: separation, AUROC, and the two rates.

    w9's refuser separated at **+0.047** on cmaze1 and a ridge probe on the same
    observation scored at chance.  Whatever this one gives is REPORTED; there is no
    tuning loop anywhere in this module to make it look better.
    """
    s = np.asarray(score, np.float64)
    y = np.asarray(label, np.float64) > 0.5
    thr = float(PINS_W12["refuse_threshold"] if threshold is None else threshold)
    out = {
        "n": int(s.size), "n_impossible": int(y.sum()), "n_possible": int((~y).sum()),
        "mean_score_impossible": float(s[y].mean()) if y.any() else None,
        "mean_score_possible": float(s[~y].mean()) if (~y).any() else None,
        "separation": (float(s[y].mean() - s[~y].mean())
                       if y.any() and (~y).any() else None),
        "auroc": _auroc(s, y),
        "threshold": thr,
        "refuse_rate_on_impossible": float((s[y] > thr).mean()) if y.any() else None,
        "false_refusal_on_possible": float((s[~y] > thr).mean()) if (~y).any() else None,
        "w9_reference": ("w9's refuser separated at +0.047 (cmaze1) / +0.020 (easy) "
                         "and a ridge probe on the same observation scored at chance. "
                         "A poor separation here is a FINDING, not a bug, and is not "
                         "tuned away."),
    }
    if reason is not None:
        r = np.asarray(reason, dtype=object)
        by = {}
        for why in sorted(set(r[y].tolist())):
            m = y & (r == why)
            by[str(why)] = {"n": int(m.sum()),
                            "mean_score": float(s[m].mean()),
                            "refuse_rate": float((s[m] > thr).mean())}
        out["by_reason"] = by
    return out


class RefuseTokenRead:
    """A transformer with a REFUSE TOKEN. Same request contract, plus a refusal score.

    ``__call__(values)`` returns ``(features (N, D), out_of_range (N,))`` exactly like
    every other arm, so it drops into the rows-3/4 harness unchanged.
    ``refuse_score(values)`` returns the refuse token's activation in ``[0, 1]``.
    """

    def __init__(self, read, out_dim: int, refuse_channel: int):
        self._read = read
        self.name = read.name
        self.seed = read.seed
        self.meta = read.meta
        self.names = read.names
        self.grid_values = read.grid_values
        self.cardinalities = read.cardinalities
        self.out_dim = int(out_dim)
        self.refuse_channel = int(refuse_channel)
        self.offgrid_ok = read.offgrid_ok
        self.partial_ok = read.partial_ok

    def __repr__(self):
        return "RefuseTokenRead(%r, seed=%d, refuse_channel=%d)" % (
            self.name, self.seed, self.refuse_channel)

    def _raw(self, values):
        return self._read(values)

    def __call__(self, values):
        full, oor = self._read(values)
        return np.asarray(full[:, :self.refuse_channel], np.float32), oor

    def refuse_score(self, values) -> np.ndarray:
        full, _ = self._read(values)
        z = np.asarray(full[:, self.refuse_channel], np.float64)
        return 1.0 / (1.0 + np.exp(-z))

    def refuses(self, values, threshold=None) -> np.ndarray:
        thr = float(PINS_W12["refuse_threshold"] if threshold is None else threshold)
        return self.refuse_score(values) > thr


def fit_refuse_token_transformer(grid, part: dict, gather: Callable, cardinalities,
                                 key, held_bitmap, D: int, seed: int = 0,
                                 epochs=None, n_per_class: int = 4096,
                                 include_out_of_range: bool = False,
                                 log: Callable = print) -> dict:
    """COMPARATOR 3 -- the refuse-token transformer, trained on CERTIFIED impossibles.

    The architecture is the shipped ``transformer`` family with ``out_dim = D + 1``:
    the extra output channel IS the refuse token.  It is matched to the same byte pin
    as every other arm, by the same measuring ``match_width``.

    Loss = MSE to the cached feature on POSSIBLE requests + ``refuse_loss_weight`` x
    logistic loss on the refuse channel over ALL requests.  Both pins are declared in
    ``PINS_W12`` before any number and there is NO tuning loop: a comparator whose
    strength we selected is a comparator whose strength we chose.
    """
    import jax
    import jax.numpy as jnp

    from . import w4_learners as L

    epochs = int(PINS_W12["refuse_epochs"] if epochs is None else epochs)
    arm = matched_arm("transformer", cardinalities, D + 1, key, log=log)
    sets = build_refusal_sets(grid, part, held_bitmap, n_per_class=n_per_class,
                              include_out_of_range=include_out_of_range)

    # Targets: the cached feature for a POSSIBLE request; for an impossible one there
    # is no ground-truth feature at all, so the feature term is MASKED there rather
    # than regressed towards a fabricated target.
    def _enc(values):
        V = np.asarray(values, np.float64)
        n, F = V.shape
        cls = np.zeros((n, F), np.int32)
        val = np.zeros((n, F), np.float32)
        for k in range(F):
            gv = np.asarray(grid.values[k], np.float64)
            cls[:, k] = np.argmin(np.abs(V[:, k][:, None] - gv[None, :]), axis=1)
            span = max(float(gv[-1] - gv[0]), 1e-12)
            val[:, k] = (V[:, k] - float(gv[0])) / span

        class _D:
            pass

        d = _D()
        d.cls_targets, d.val_targets = cls, val
        return L.encode_inputs(arm["in_kind"], d, cardinalities)

    Xtr = _enc(sets["train_values"])
    ytr = np.asarray(sets["train_labels"], np.float32)
    # ground-truth features: only the possible half has rows behind it
    kept = np.asarray(part["kept_rows"], np.int64)
    fi_kept = grid.factor_indices_of(kept)
    lookup = {}
    for i, t in enumerate(map(tuple, fi_kept.tolist())):
        lookup.setdefault(t, i)
    Vtr = np.asarray(sets["train_values"], np.float64)
    gidx = certify_impossible(Vtr, grid, held_bitmap,
                              [(int(f), int(v)) for f, v
                               in part["spec"]["terms"]])["grid_index"]
    tr_idx = np.full(Xtr.shape[0], -1, np.int64)
    for j in range(Vtr.shape[0]):
        if ytr[j] > 0.5 or (gidx[j] < 0).any():
            continue
        tr_idx[j] = lookup.get(tuple(int(x) for x in gidx[j]), -1)
    have = tr_idx >= 0
    Ytr = np.zeros((Xtr.shape[0], D), np.float32)
    if have.any():
        Ytr[have] = gather(kept[tr_idx[have]])
    fmask = have.astype(np.float32)
    log("    [refuse_token] train %d (possible %d / impossible %d); feature targets "
        "available for %d" % (Xtr.shape[0], int((ytr < 0.5).sum()),
                              int((ytr > 0.5).sum()), int(have.sum())))

    batch = int(PINS_W12["batch"])
    steps = max(Xtr.shape[0] // batch, 1) * epochs
    w = float(PINS_W12["refuse_loss_weight"])

    def batches(i):
        r = np.random.default_rng(20_260_923 + i)
        idx = r.integers(0, Xtr.shape[0], size=batch)
        return {"x": Xtr[idx], "y": Ytr[idx], "m": fmask[idx], "r": ytr[idx]}

    def loss_fn(m, b):
        out = jax.vmap(m)(b["x"])
        feat, logit = out[:, :D], out[:, D]
        se = jnp.mean((feat - b["y"]) ** 2, axis=1) * b["m"]
        mse = jnp.sum(se) / jnp.maximum(jnp.sum(b["m"]), 1.0)
        bce = jnp.mean(jnp.logaddexp(0.0, logit) - b["r"] * logit)
        return mse + w * bce

    t0 = time.time()
    model, info = _run_unlearn(arm["build"](arm["width"]), batches, key, steps,
                               float(PINS_W12["refuse_lr"]),
                               float(PINS_W12["refuse_weight_decay"]), loss_fn,
                               log=log, tag="refuse_token")
    secs = float(time.time() - t0)

    Xev = _enc(sets["eval_values"])
    z = L.predict(model, Xev)[:, D].astype(np.float64)
    score = 1.0 / (1.0 + np.exp(-z))
    sep = separation_report(score, sets["eval_labels"], sets["eval_reason"])

    rows_read = int(have.sum())
    cost = CostLine(
        seconds=secs, data_rows_read=rows_read,
        bytes_rewritten=int(arm["param_count"]) * 4,
        gradient_row_visits=int(steps * batch),
        param_bytes_changed=int(arm["param_count"]) * 4,
        passes_over_data=float(steps * batch) / max(rows_read, 1),
        note=("the ONLY permitted comparator for refusal DISPLAY rows. Trained on "
              "EXACTLY-CERTIFIED impossible examples (%d deleted-row requests + %d "
              "out-of-range), with NO tuning loop." % (
                  sets["n_impossible_deleted"], sets["n_impossible_out_of_range"])))
    log("    " + cost.one_line("refuse_token %s" % part["spec"]["label"]))
    log("    [refuse_token] separation %+.4f  AUROC %.4f  refuse@impossible %.4f  "
        "false-refusal@possible %.4f"
        % (sep["separation"], sep["auroc"], sep["refuse_rate_on_impossible"],
           sep["false_refusal_on_possible"]))
    return {"kind": "refuse_token", "model": model, "arm": arm, "fit": info,
            "separation": sep,
            "sets": {k: v for k, v in sets.items()
                     if not isinstance(v, np.ndarray)},
            "refuse_channel": int(D), "out_dim": int(D + 1),
            "cost": assert_cost_line_complete(cost.as_dict(), "refuse_token")}


# ---------------------------------------------------------------------------
# 9. The registry, and the by-name wiring the crossover hub calls
# ---------------------------------------------------------------------------

COMPARATORS = {
    "as_trained": {
        "role": "the learned arm BEFORE any deletion -- the reference the others move from",
        "entry_point": "chlu.eval.rivals.w12_deletion_baselines.fit_as_trained",
        "arms": list(LEARNED_ARMS),
        "reads_deleted_rows": True,
        "exactness": "n/a -- it has not deleted anything",
    },
    "retrain_without": {
        "role": "COMPARATOR 1 -- the EXACT reference: refit from scratch on the pool "
                "minus the deleted rows",
        "entry_point": "chlu.eval.rivals.w12_deletion_baselines.retrain_without",
        "arms": list(LEARNED_ARMS),
        "reads_deleted_rows": False,
        "exactness": "exact by construction: the deleted rows were never in the pool",
    },
    "unlearn_neggrad": {
        "role": "COMPARATOR 2a -- NegGrad+ approximate unlearning at MATCHED COMPUTE",
        "entry_point": "chlu.eval.rivals.w12_deletion_baselines.approximate_unlearn"
                       " (method='neggrad')",
        "arms": list(LEARNED_ARMS),
        "reads_deleted_rows": True,
        "exactness": "approximate -- its gap to retrain_without is the thing measured",
    },
    "unlearn_finetune": {
        "role": "COMPARATOR 2b -- fine-tune-on-retain unlearning at MATCHED COMPUTE",
        "entry_point": "chlu.eval.rivals.w12_deletion_baselines.approximate_unlearn"
                       " (method='finetune')",
        "arms": list(LEARNED_ARMS),
        "reads_deleted_rows": False,
        "exactness": "approximate -- its gap to retrain_without is the thing measured",
    },
    "refuse_token": {
        "role": "COMPARATOR 3 -- the refuse-token transformer, the ONLY permitted "
                "comparator for refusal DISPLAY rows",
        "entry_point": "chlu.eval.rivals.w12_deletion_baselines."
                       "fit_refuse_token_transformer",
        "arms": ["transformer"],
        "reads_deleted_rows": True,
        "exactness": "n/a -- it is a refusal comparator, scored by separation",
    },
}


def comparator_name(arm_family: str, kind: str, spec_label: str) -> str:
    """The ONE name the crossover hub calls. Stable, and it is the file name on disk."""
    if kind not in COMPARATOR_KINDS:
        raise KeyError("unknown comparator kind %r; have %s"
                       % (kind, list(COMPARATOR_KINDS)))
    if kind == "as_trained":
        return "%s__as_trained" % arm_family
    return "%s__%s__%s" % (arm_family, kind, spec_label)


def _pyify(o):
    """numpy scalars/arrays -> plain python, so a manifest is always JSON-writable."""
    if isinstance(o, dict):
        return {str(k): _pyify(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_pyify(v) for v in o]
    if isinstance(o, np.ndarray):
        return _pyify(o.tolist())
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def save_comparator(result: dict, seed: int, grid, cache_manifest: dict,
                    root, spec_label: str, part: Optional[dict] = None):
    """Write the comparator so ``w4_api.load_rival`` loads it with no w12 import.

    The manifest carries ``base_family`` (the architecture that rebuilds it), the
    deletion spec and the COST LINE, so a table can be built from the checkpoints
    alone.
    """
    from .w4_api import save_learner

    arm = result["arm"]
    name = comparator_name(arm["family"], result["kind"], spec_label)
    extra = {
        "base_family": arm["family"],
        "comparator_kind": result["kind"],
        "partial_ok": False,
        "w12_spoke": PINS_W12["spoke"],
        "deletion_spec": (part or {}).get("spec"),
        "deletion_counts": {k: v for k, v in (part or {}).items()
                            if isinstance(v, (int, float))},
        "cost_line": result["cost"],
        "param_target": arm["param_target"],
        "param_error": arm["param_error"],
        "byte_pin": arm["byte_pin"],
        "compute": result.get("compute"),
        "separation": result.get("separation"),
        "refuse_channel": result.get("refuse_channel"),
    }
    return save_learner(seed, name, result["model"], arm["width"], arm["in_kind"],
                        grid, arm["out_dim"], cache_manifest, root=root,
                        extra=_pyify(extra))


def load_comparator(seed: int, name: str, root):
    """Load a saved comparator by NAME. A refuse-token arm comes back wrapped."""
    from .w4_api import load_rival

    r = load_rival(seed, name, root=root)
    ch = r.meta.get("refuse_channel")
    if ch is not None:
        return RefuseTokenRead(r, int(r.meta["out_dim"]), int(ch))
    return r
