"""The rival arms' **outer loop** and the audit's five arms, one code path for all.

⭐ **The deliverable is protocol uniformity across families** (Hub ruling §6.1), so
every rival — TTT-Linear, TTT-MLP, DeltaNet, GDN, GDN-2 — goes through *this*
function, not its own. Any asymmetry between families is then a property of the
equations, not of who wrote the runner.

**The five arms** (`PREREG-Bprime.md` §2; the CLU's own column is banked, §7):

===================  ====================================================
``full``             the rival's learned test-time dynamics
``launder``          its **own byte-matched table** of ``(theta_K x,
                     theta_V x)`` pairs, arg-min read (**P5**)
``*_+0B``            the strongest **zero-extra-byte** readers of that same
                     table (2-NN mean, 2-NN IDW), signed margin reported
``same_keys_null``   the same table keys, payload column permuted
``blank``            the identical rival with **nothing written** (state =
                     the learned init ``W_0``/``S_0`` only)
===================  ====================================================

⭐ **F2a's binding guard** (`rival-recon` §F2a): the outer parameters are fitted on
**auxiliary streams built from different seeds** — different sites, different
payloads — and never on the cell's own stream. Without it a rival could memorise
the eval items in its parameters (pool 1) and the whole state-byte axis would be
meaningless. This is a *stricter* guard than any published rival baseline runs and
it is declared, not implied.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from chlu.eval.dividend import knn_mean_launder, same_keys_null, settle_deleted_launder
from chlu.eval.rivals.deltanet import DELTA_VARIANTS, DeltaMemory
from chlu.eval.rivals.ledger import TTT_MINI_BATCH, head_width_for_budget, matched_table_rows
from chlu.eval.rivals.mamba2 import Mamba2Memory
from chlu.eval.rivals.ttt import TTTMemory

#: The rival state types this task audits. ``gdn`` is the §A14.2 ablation;
#: ``gdn2`` is the **reference** delta-rule arm. ⭐ ``mamba2`` (C2W5, Head-funded
#: ruling 5) closes the referee's missing-experiment 5 — the B′ survey sentence
#: names SSMs and none was measured. ⛔ **APPEND ONLY:** the per-rival fit key is
#: ``RIVALS.index(name)``, so inserting a name anywhere but the end would silently
#: re-draw every later arm's initialisation and break reproduction of the banked
#: C2W4/C2W5 numbers (verified: the five incumbents reproduce bit-identically).
RIVALS: Tuple[str, ...] = ("ttt_linear", "ttt_mlp", "deltanet", "gdn", "gdn2",
                           "mamba2")

#: Which sizing law each rival's head width obeys (see ``head_width_for_budget``).
LEDGER_KIND: Dict[str, str] = {"ttt_linear": "ttt_linear", "ttt_mlp": "ttt_mlp",
                               "deltanet": "delta", "gdn": "delta", "gdn2": "delta",
                               "mamba2": "mamba2"}

#: ⭐ The declared budget: the CLU's **banked** ``aggregate@base`` full-byte figure,
#: 5456 B = 1364 float32 (`bprime-fb4-gate` §A3.4). Filed in the PREREG before any
#: run; ⛔ never re-tuned after seeing a score.
DEFAULT_BUDGET_FLOATS = 1364

#: F3-lite (declared as a budget choice, **not** presented as `rival-recon` F3
#: compliance, which asks for 6 lrs x 2 weight decays). ⭐ Kept as the default so
#: the C2W4 audit stays reproducible; the **full** F3 grid is opt-in below.
LR_GRID: Tuple[float, ...] = (1e-3, 3.16e-3, 1e-2)
#: ⭐ `rival-recon` **F3, verbatim** (= MAD App. B.4's 3x2 grid u Zoology's
#: ``np.logspace(-4, -2, 4)``): the standing baseline-tuning rule, the operational
#: form of N78. ⚠ Its *upper* edge is 1e-2 — the same as F3-lite's — so widening
#: the grid adds points only on the low-lr side.
LR_GRID_F3: Tuple[float, ...] = (1e-4, 3.16e-4, 5e-4, 1e-3, 3.16e-3, 1e-2)
#: F3's second axis. ``wd = 0`` keeps ``optax.adam`` (⛔ no optimiser change at
#: all, so the F3-lite control column is untouched); ``wd > 0`` uses decoupled
#: ``optax.adamw``.
WD_GRID_F3: Tuple[float, ...] = (0.0, 0.1)
#: TTT's mini-batch is part of the grid because the gym's stream is ~10-19 tokens
#: (see ``ttt.py``'s regime caveat) — the arm is audited at its own best.
TTT_B_GRID: Tuple[int, ...] = (1, TTT_MINI_BATCH)
OUTER_STEPS = 400


@dataclass
class FitExample:
    """One auxiliary (write-stream, query-set) pair for the outer loop."""

    xs: np.ndarray       # (T, d_in) write tokens [address | payload]
    mask: np.ndarray     # (T,) 1 = a real token
    xq: np.ndarray       # (n, d_in) launch buffers (payload block zeroed) = phi
    target: np.ndarray   # (n, m)


def make_rival(name: str, d_in: int, m: int, *, key,
               budget_floats: int = DEFAULT_BUDGET_FLOATS,
               mini_batch: int = TTT_MINI_BATCH,
               d_head: Optional[int] = None, **arm_kwargs):
    """Build a rival at the **iso-state** head width the PREREG's rule fixes.

    ``arm_kwargs`` are forwarded to the arm's constructor. ⛔ **Empty in every
    reported audit cell** — it exists so a declared *ablation* (e.g. Mamba-2's
    block-level ``use_D`` / ``gate_z``, dropped by default exactly as every other
    arm's block-level parts are) can be run through the SAME outer loop and scorer
    instead of a hand-rolled script.
    """
    if name not in RIVALS:
        raise ValueError(f"unknown rival {name!r}; known: {RIVALS}")
    d = int(d_head) if d_head is not None else head_width_for_budget(
        LEDGER_KIND[name], int(budget_floats), buffer_tokens=int(mini_batch))
    if name.startswith("ttt"):
        return TTTMemory(d_in, d, m, key=key,
                         kind=("linear" if name == "ttt_linear" else "mlp"),
                         mini_batch=int(mini_batch))
    if name == "mamba2":
        # ⚠ ``mini_batch`` is NOT forwarded as the SSD chunk: TTT's ``b`` changes
        # the function (and its state, via the in-flight buffer), while SSD's
        # chunk is an exact re-association. The chunk stays at the rig-matched
        # ``SSD_CHUNK`` and is asserted inert in the tests.
        return Mamba2Memory(d_in, d, m, key=key, **arm_kwargs)
    return DeltaMemory(d_in, d, m, key=key, variant=name, **arm_kwargs)


# --------------------------------------------------------------------------
# the outer loop
# --------------------------------------------------------------------------
def _mae(model, ex_xs, ex_mask, ex_xq, ex_tgt) -> jnp.ndarray:
    pred = model.read(model.write(ex_xs, ex_mask), ex_xq)
    return jnp.mean(jnp.abs(pred - ex_tgt))


@eqx.filter_jit
def _loss(model, xs, mask, xq, tgt) -> jnp.ndarray:
    """Mean over the auxiliary streams of the family's own error metric."""
    return jnp.mean(jax.vmap(lambda a, b, c, d: _mae(model, a, b, c, d))(
        xs, mask, xq, tgt))


def _stack(examples: Sequence[FitExample]):
    return (jnp.asarray(np.stack([e.xs for e in examples]), dtype=jnp.float32),
            jnp.asarray(np.stack([e.mask for e in examples]), dtype=jnp.float32),
            jnp.asarray(np.stack([e.xq for e in examples]), dtype=jnp.float32),
            jnp.asarray(np.stack([e.target for e in examples]), dtype=jnp.float32))


def fit_rival(model, examples: Sequence[FitExample], *, lr: float = 3.16e-3,
              wd: float = 0.0, steps: int = OUTER_STEPS,
              val_examples: Optional[Sequence[FitExample]] = None,
              verbose: bool = False):
    """Train the rival's **outer** parameters through its own test-time inner loop.

    That is what TTT's outer loop is (§2.2) and what a delta-rule layer's
    projections are trained by; doing anything else would audit an untrained rival,
    which would be a laundering in *our* favour.

    ``wd`` is `rival-recon` F3's second grid axis. ⛔ ``wd == 0`` keeps
    ``optax.adam`` **exactly** — the C2W4 configuration is not perturbed by the
    existence of the new axis; ``wd > 0`` switches to decoupled ``optax.adamw``.
    ``val_examples`` (if given) are scored on the fitted model and never entered
    into a gradient: they exist so best-of-grid can be selected on a **held-out**
    fit stream instead of on the objective it is optimising (see
    :func:`select_best`).
    """
    import optax

    xs, mask, xq, tgt = _stack(examples)
    opt = (optax.adamw(float(lr), weight_decay=float(wd)) if float(wd) > 0
           else optax.adam(float(lr)))
    params, static = eqx.partition(model, eqx.is_inexact_array)
    state = opt.init(params)

    @eqx.filter_jit
    def step(params, state):
        def f(p):
            return _loss(eqx.combine(p, static), xs, mask, xq, tgt)

        loss, grads = eqx.filter_value_and_grad(f)(params)
        upd, state = opt.update(grads, state, params)
        return eqx.apply_updates(params, upd), state, loss

    hist: List[float] = []
    for i in range(int(steps)):
        params, state, loss = step(params, state)
        hist.append(float(loss))
        if verbose and i % 100 == 0:
            print(f"    [fit lr={lr:g} wd={wd:g}] step {i:4d} loss {float(loss):.5f}")
    fitted = eqx.combine(params, static)
    diverged = not bool(np.all(np.isfinite(hist)))
    rec = {"lr": float(lr), "wd": float(wd), "steps": int(steps),
           "final": float("nan") if diverged else float(hist[-1]),
           "diverged": diverged, "history": hist}
    if val_examples is not None and not diverged:
        # held out: scored once, never differentiated
        rec["val_final"] = float(_loss(fitted, *_stack(val_examples)))
    elif val_examples is not None:
        rec["val_final"] = float("nan")
    # a diverged arm is REPORTED with its budget, never silently dropped
    return fitted, rec


def fit_grid(name: str, d_in: int, m: int, examples: Sequence[FitExample],
             *, key, budget_floats: int = DEFAULT_BUDGET_FLOATS,
             d_head: Optional[int] = None,
             lrs: Sequence[float] = LR_GRID,
             wds: Sequence[float] = (0.0,),
             b_grid: Optional[Sequence[int]] = None,
             steps: int = OUTER_STEPS,
             val_examples: Optional[Sequence[FitExample]] = None,
             arm_kwargs: Optional[dict] = None,
             verbose: bool = False) -> Tuple[List[dict], List[Any]]:
    """Fit **every** point of the ``lr x wd x b`` grid and return the whole surface.

    ⭐ **The init is drawn once per ``(arm, seed, b)`` and shared by every
    ``(lr, wd)``** — so the surface is a tuning surface and not a tuning-*and*-init
    surface, and so widening the grid cannot silently re-draw the incumbent points.
    ⚠ This is a **declared change** from C2W4, which split the key sequentially and
    therefore made every model depend on the grid's length and order; the
    consequence is priced by re-selecting the C2W4 sub-grid from these same fits
    (the *F3-lite control*, see :func:`select_best`).
    """
    bs = list(b_grid if b_grid is not None else
              (TTT_B_GRID if name.startswith("ttt") else (TTT_MINI_BATCH,)))
    grid: List[dict] = []
    models: List[Any] = []
    for b in bs:
        k_b = jax.random.fold_in(key, int(b))
        for wd in wds:
            for lr in lrs:
                model = make_rival(name, d_in, m, key=k_b,
                                   budget_floats=budget_floats,
                                   mini_batch=int(b), d_head=d_head,
                                   **(arm_kwargs or {}))
                model, rec = fit_rival(model, examples, lr=float(lr),
                                       wd=float(wd), steps=steps,
                                       val_examples=val_examples, verbose=verbose)
                rec["mini_batch"] = int(b)
                rec.pop("history")
                grid.append(rec)
                models.append(model)
    return grid, models


def select_best(grid: Sequence[dict], models: Sequence[Any], *,
                lrs: Optional[Sequence[float]] = None,
                wds: Optional[Sequence[float]] = None,
                on: str = "fit", label: str = "") -> Tuple[Any, dict]:
    """Pick the grid's best point, optionally restricted to a **sub-grid**.

    ``on="fit"`` selects on the fit split's own (optimised) loss — C2W4's rule.
    ``on="val"`` selects on the **held-out** auxiliary stream, which is the only
    way F3's ``wd`` axis can ever be chosen: a regulariser does not lower the
    objective it is not optimising. ⛔ Neither reader ever sees the eval split.
    """
    field = "val_final" if on == "val" else "final"
    tol = 1e-12
    sub = [(r, mo) for r, mo in zip(grid, models, strict=True)
           if (lrs is None or any(abs(r["lr"] - float(x)) <= tol for x in lrs))
           and (wds is None or any(abs(r.get("wd", 0.0) - float(x)) <= tol
                                   for x in wds))]
    if not sub:
        raise ValueError(f"empty sub-grid for selection {label!r}")
    best, best_rec = None, None
    for rec, model in sub:
        score = rec.get(field, float("nan"))
        score = float(score) if np.isfinite(score) else np.inf
        if best_rec is None or score < best_rec["_score"]:
            best, best_rec = model, dict(rec, _score=score)
    best_rec.pop("_score")
    return best, {"best": best_rec, "grid": list(grid),
                  "selection": {"label": label or ("f3" if lrs is None else "sub"),
                                "on": ("held-out fit-validation stream (never the "
                                       "eval split)" if on == "val" else
                                       "the FIT split's own loss (C2W4's rule)"),
                                "lrs": (None if lrs is None else [float(x)
                                                                  for x in lrs]),
                                "wds": (None if wds is None else [float(x)
                                                                  for x in wds]),
                                "n_points": len(sub)},
                  "note": ("best-of-grid on the FIT split (auxiliary streams from "
                           "different seeds), never on the eval split")}


def fit_best_of_grid(name: str, d_in: int, m: int, examples: Sequence[FitExample],
                     *, key, budget_floats: int = DEFAULT_BUDGET_FLOATS,
                     d_head: Optional[int] = None,
                     lrs: Sequence[float] = LR_GRID,
                     wds: Sequence[float] = (0.0,),
                     b_grid: Optional[Sequence[int]] = None,
                     steps: int = OUTER_STEPS,
                     val_examples: Optional[Sequence[FitExample]] = None,
                     arm_kwargs: Optional[dict] = None,
                     select_on: str = "fit", verbose: bool = False):
    """Best-of-grid on the **fit** split, never on the eval split."""
    grid, models = fit_grid(name, d_in, m, examples, key=key,
                            budget_floats=budget_floats, d_head=d_head, lrs=lrs,
                            wds=wds, b_grid=b_grid, steps=steps,
                            val_examples=val_examples, arm_kwargs=arm_kwargs,
                            verbose=verbose)
    return select_best(grid, models, on=select_on, label="best_of_grid")


# --------------------------------------------------------------------------
# the five arms
# --------------------------------------------------------------------------
def rival_arms(model, xs: np.ndarray, mask: np.ndarray, xq: np.ndarray, *,
               rng: Optional[np.random.Generator] = None,
               n_rows: Optional[int] = None) -> Dict[str, np.ndarray]:
    """Every arm's prediction, in the family's own payload space.

    All arms share the memory's ``theta_{K,Q,V,O}`` — the launder differs from the
    read **only** in that the dynamics have been replaced by a byte-equal table.
    That is the same discipline ``settle_deleted_launder`` applies to the CLU.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    xs_j = jnp.asarray(xs, dtype=jnp.float32)
    xq_j = jnp.asarray(xq, dtype=jnp.float32)
    msk = jnp.asarray(mask, dtype=jnp.float32)

    out: Dict[str, np.ndarray] = {}
    state = model.write(xs_j, msk)
    out["full"] = np.asarray(model.read(state, xq_j))
    out["blank"] = np.asarray(model.read(model.init_state(), xq_j))

    keys, vals = model.kv_table(xs_j)
    live = np.asarray(mask).astype(bool)
    keys = np.asarray(keys)[live]
    vals = np.asarray(vals)[live]
    rows = int(keys.shape[0] if n_rows is None else min(int(n_rows), keys.shape[0]))
    keys, vals = keys[:rows], vals[:rows]
    qk = np.asarray(model.query_keys(xq_j))

    def _dec(v):
        return np.asarray(model.decode_values(jnp.asarray(v, dtype=jnp.float32)))

    if rows == 0:
        # An **empty** table: nothing was written, so every table arm returns the
        # zero value — the same convention the gym's own launder uses when the
        # table cannot answer. This is the blank-store configuration and it must
        # not raise: it is a *control*, not a failure.
        z = np.zeros((qk.shape[0], int(model.d_v)), dtype=float)
        for nm in ("launder", "same_keys_null", "knn2_mean_+0B", "knn2_idw_+0B",
                   "table_mean_+0B"):
            out[nm] = _dec(z)
        return out

    out["launder"] = _dec(settle_deleted_launder(keys, vals, qk))
    out["same_keys_null"] = _dec(same_keys_null(keys, vals, qk, rng))
    out["knn2_mean_+0B"] = _dec(knn_mean_launder(keys, vals, qk, k=2))
    out["knn2_idw_+0B"] = _dec(knn_mean_launder(keys, vals, qk, k=2,
                                                weighting="inverse_distance"))
    # ⭐ the most trivial +0 B reader there is, and the one a partial-input /
    # trivial-baseline audit (Poliak et al. 2018; Feng, Wallace & Boyd-Graber
    # ACL 2019) demands: **ignore the query entirely and return the table's mean
    # value**. A "+0 B substitute audit" that omits the constant predictor is not
    # an audit. It costs nothing: the mean is a function of the table.
    out["table_mean_+0B"] = _dec(np.broadcast_to(vals.mean(axis=0, keepdims=True),
                                                 (qk.shape[0], vals.shape[1])))
    return out


def table_budget(model, *, budget_floats: Optional[int] = None) -> Dict[str, Any]:
    """How many ``(k, v)`` rows fit in **exactly** the memory's own state bytes."""
    st = int(budget_floats if budget_floats is not None
             else model.declared_state_floats())
    rows = matched_table_rows(st, model.d_k, model.d_v)
    return {"state_floats": st, "d_k": int(model.d_k), "d_v": int(model.d_v),
            "n_rows_affordable": int(rows),
            "row_floats": int(model.d_k + model.d_v)}


__all__ = [
    "RIVALS", "LEDGER_KIND", "DEFAULT_BUDGET_FLOATS", "LR_GRID", "LR_GRID_F3",
    "WD_GRID_F3", "TTT_B_GRID", "OUTER_STEPS", "FitExample", "make_rival",
    "fit_rival", "fit_grid", "select_best", "fit_best_of_grid", "rival_arms",
    "table_budget", "DELTA_VARIANTS",
]
