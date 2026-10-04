"""hub/refusal-2 -- COERCIVE-COMPONENT SCREENING at the LINEAR read point.

THE CRITERION (PREREG-1 section 2.1), and nothing else is reported as the result::

    S = E_store(x_read)          x_read[:, cols_k] = P[:, k, :] @ atoms[k]

WHAT IS SHIPPED AND WHAT IS MINE.
  * ``E_store``            = ``gen_landscape.store_energy``           -- CALLED, never re-implemented.
  * the linear read        = ``gen_landscape.launch_state(launch_mode="posterior_mean")``
                             (``gen_landscape.py:1028``; one of four ``LAUNCH_MODES``), reachable
                             through the shipped knob ``GenDynamicalReadConfig.launch_mode``
                             (``config.py:6196``).  NOT A NEW READ VERB.
  * the threshold          = ``gen_knn.threshold_from_valid`` at the shipped
                             ``refusal_valid_quantile = 0.95``   -- CALLED, not swept.
  * the term generalisation= ``refusal_lib.term_experts`` -- INHERITED from hub/refusal verbatim
                             (sha256 in ``inherited/SHA256SUMS.txt``), because the shipped
                             ``request_experts`` takes a dense ``(n, F)`` of grid class indices and
                             CANNOT EXPRESS EITHER INVALID KIND.
  * MINE, and the only new line: the posterior-weighted combination in place of the argmax, which is
    the shipped ``posterior_mean`` branch lifted from ``(n, F)`` requests to per-TERM requests.
    :func:`assert_is_shipped_linear_read` proves it equals ``launch_state``'s own output on valid
    row-derived requests, which is the proof that this is the shipped read and not a different arm.

NOT ONE MANIFEST-PINNED FILE IS EDITED.  Everything lives under
``$CHLU_HUB_OUTPUTS/hub-refusal-2/`` and imports the pin read-only.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE / "inherited"), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import refusal_lib as R  # noqa: E402  (the INHERITED library, byte-identical copy)

from chlu.core.gen_landscape import (  # noqa: E402
    AtomLandscape,
    _softmax_last,
    atom_posterior,
    atom_radii,
    launch_state,
    store_energy,
    total_energy,
)
from chlu.eval.gen_probe import binary_auroc  # noqa: E402
from chlu.eval.rivals import gen_knn as knn  # noqa: E402

#: The one path this hub may write.  The predecessor's directory is INHERITED FACT and
#: is opened read-only; nothing here ever writes to it.
OUT = Path(os.environ["CHLU_HUB_OUTPUTS"]) / "hub-refusal-2"

#: The shipped ``refusal_valid_quantile`` (``config.py:4594``).  Asserted against the
#: config at run time by :func:`assert_quantile_is_shipped` -- never a literal I chose.
Q_DEFAULT = 0.95

#: PREREG-1 section 4.2 -- the Certified Access section 4.1 relaxation residual.  ``S`` must beat the
#: better physics-free margin by MORE than this, or the landscape is decorative.
MARGIN_RESIDUAL = 0.024


def assert_quantile_is_shipped(config) -> dict:
    """``q`` is the harness's constant, read from the config -- not a number I picked."""
    q = float(config.gen_baselines.refusal_valid_quantile)
    if abs(q - Q_DEFAULT) > 0:
        raise AssertionError(
            f"the shipped refusal_valid_quantile moved to {q!r}; PREREG-1 section 2.4 pinned "
            f"{Q_DEFAULT!r}. STOP and report -- do not silently follow it.")
    return {"q": q, "source": "config.gen_baselines.refusal_valid_quantile",
            "swept": False, "chosen_by_this_hub": False}


# ---------------------------------------------------------------------------
# The linear read -- the posterior-weighted combination of each well's atoms
# ---------------------------------------------------------------------------


def _posterior_from_terms(book, log_w, n_terms, dtype32: bool = False) -> np.ndarray:
    """``(n, K, N)`` atom posterior from PER-TERM experts.

    Identical to ``atom_posterior`` for a valid one-term-per-factor request: that function
    is ``log_prior + sum_f (log_cond[f][v] - log_prior)``, and this is the same sum with
    one term per requested term.  ``dtype32`` reproduces ``atom_posterior``'s float32 cast
    exactly, which is what makes the bit-identity check in
    :func:`assert_is_shipped_linear_read` meaningful rather than approximate.
    """
    lp = np.asarray(book.log_prior, np.float64)
    n = int(log_w.shape[1])
    S = np.broadcast_to(lp, (n,) + lp.shape).copy()
    for t in range(int(log_w.shape[0])):
        active = (np.asarray(n_terms) > t)[:, None, None]
        S = S + np.where(active, np.asarray(log_w[t], np.float64), 0.0)
    if dtype32:
        S = S.astype(np.float32)
    return _softmax_last(S)


def linear_read(land, book, log_w, n_terms, dtype32: bool = False) -> dict:
    """``requests -> x_read``, the SHIPPED ``posterior_mean`` read, per TERM.

    ``x_read[:, cols_k] = P[:, k, :] @ atoms[k]`` -- ``gen_landscape.py:1029``, verbatim.
    Also returns the two PHYSICS-FREE readout margins (PREREG-1 section 4.2), which are functions
    of ``P`` ALONE: no store content, no distance, no energy.
    """
    P = _posterior_from_terms(book, log_w, n_terms, dtype32=dtype32)   # (n, K, N)
    store = land.store
    n = int(P.shape[0])
    x = np.zeros((n, store.dim), np.float32)
    for k, w in enumerate(store.well_channels):
        cols = np.asarray(w, np.int64)
        A = np.asarray(store.atoms[k], np.float32)
        x[:, cols] = (P[:, k, :] @ A).astype(np.float32)
    part = np.partition(P, -2, axis=-1)
    top1, top2 = part[..., -1], part[..., -2]
    return {
        "x_read": x,
        "posterior": P,
        "atom_index": P.argmax(axis=-1).astype(np.int64),
        # PREREG-1 section 4.2 -- negated so that, like S, HIGH means REFUSE.
        "M_conf": -top1.mean(axis=1),
        "M_margin": -(top1 - top2).mean(axis=1),
        "mean_top1": float(top1.mean()),
        "mean_entropy": float((-(P * np.log(np.maximum(P, 1e-300))).sum(axis=-1)).mean()),
    }


def assert_is_shipped_linear_read(land, book, requests, log_w, n_terms) -> dict:
    """PROOF that this is ``launch_state(..., "posterior_mean")`` and not a different arm.

    Runs on any VALID row-derived request set (one term per factor, every value on-grid),
    where the shipped entry point can express the request at all.  Reports BOTH the
    float32 path (which should be bit-identical to the shipped call) and the float64 path
    (this module's default), so the reader sees the dtype difference rather than a claim.
    """
    if any("class_indices" not in r for r in requests):
        return {"skipped": "not a row-derived valid request set"}
    Rq = np.asarray([[int(c) for c in r["class_indices"]] for r in requests], np.int64)
    ref = np.asarray(launch_state(land, book, Rq, launch_mode="posterior_mean")["x0"],
                     np.float32)
    got32 = linear_read(land, book, log_w, n_terms, dtype32=True)["x_read"][: ref.shape[0]]
    got64 = linear_read(land, book, log_w, n_terms, dtype32=False)["x_read"][: ref.shape[0]]
    scale = float(np.abs(ref).max()) or 1.0
    ref_post = np.asarray(atom_posterior(book, Rq), np.float32)
    mine_post = _posterior_from_terms(book, log_w, n_terms, dtype32=True)[: ref.shape[0]]
    return {
        "shipped_entry_point": "gen_landscape.launch_state(launch_mode='posterior_mean')",
        "bit_identical_float32_path": bool(np.array_equal(got32, ref)),
        "max_abs_delta_float32_path": float(np.max(np.abs(got32 - ref))),
        "max_abs_delta_float64_path": float(np.max(np.abs(got64 - ref))),
        "rel_delta_float64_path": float(np.max(np.abs(got64 - ref)) / scale),
        "max_abs_delta_posterior_vs_atom_posterior": float(
            np.max(np.abs(mine_post - _softmax_last(ref_post)))),
        "n_checked": int(ref.shape[0]),
        "state_scale_max_abs": scale,
    }


# ---------------------------------------------------------------------------
# The statistic, and the companions REJECTED IN ADVANCE
# ---------------------------------------------------------------------------


def nearest_atom_distance(store, X) -> dict:
    """PREREG-1 companion C1 -- REJECTED IN ADVANCE, computed and logged, never adopted.

    ``d(x_read, nearest atom)`` per well (the charter section C.4 statistic) and its ratio to the
    store's own RMS atom radius.  Also the FALSIFIER F7 check: if the linear read sits ON an
    atom, this is the argmax read again and section 2.2's hinge is gone.
    """
    Z = np.asarray(X, np.float32)
    radii = atom_radii(store)
    per_well = []
    for k, w in enumerate(store.well_channels):
        z = Z[:, np.asarray(w, np.int64)].astype(np.float64)
        a = np.asarray(store.atoms[k], np.float64)
        d2 = (np.sum(z * z, axis=1)[:, None] - 2.0 * (z @ a.T)
              + np.sum(a * a, axis=1)[None, :])
        per_well.append(np.sqrt(np.maximum(d2.min(axis=1), 0.0)))
    D = np.stack(per_well, axis=1)                       # (n, K)
    ref = float(np.sqrt(np.mean(radii ** 2))) or 1.0
    return {"d_nearest_per_well": D, "d_nearest_mean": D.mean(axis=1),
            "C1_d_over_spread": D.mean(axis=1) / ref,
            "atom_rms_radius": ref, "atom_radii": radii}


def linear_stats(land, book, requests, log_w, n_terms, sigma_ref: float,
                 dtype32: bool = False) -> dict:
    """Everything this hub measures at the linear read point, in one pass.

    ``S`` is THE criterion.  ``C1`` and ``C3`` are companions REJECTED IN ADVANCE.
    ``M_conf`` / ``M_margin`` are the physics-free CONTROLS, never candidates.
    """
    lr = linear_read(land, book, log_w, n_terms, dtype32=dtype32)
    x = lr["x_read"]
    S = np.asarray(store_energy(land, x, sigma_ref), np.float64)        # <-- THE CRITERION
    e_tot = np.asarray(total_energy(land, x, log_w, sigma_ref), np.float64)
    near = nearest_atom_distance(land.store, x)
    # C3 -- the predecessor's dead request-energy residuals, logged for continuity only.
    res = R.per_term_residuals(land, x, log_w, n_terms, sigma_ref)
    with np.errstate(invalid="ignore"):
        s_max = np.nanmax(res, axis=1)
        s_sum = np.nansum(res, axis=1)
    return {
        "S": S,                                   # PREREG-1 section 2.1 -- the ONLY reported result
        "M_conf": lr["M_conf"], "M_margin": lr["M_margin"],     # section 4.2 controls
        "C1_d_over_spread": near["C1_d_over_spread"],           # rejected in advance
        "d_nearest_mean": near["d_nearest_mean"],
        "C3_S_max": s_max, "C3_S_sum": s_sum,                   # rejected in advance
        "e_total": e_tot,
        "x_read": x, "posterior": lr["posterior"],
        "diag": {"mean_top1": lr["mean_top1"], "mean_entropy": lr["mean_entropy"],
                 "atom_rms_radius": near["atom_rms_radius"],
                 "mean_d_nearest": float(near["d_nearest_mean"].mean()),
                 "mean_C1_d_over_spread": float(near["C1_d_over_spread"].mean()),
                 "mean_S": float(S.mean()), "n": int(S.size)},
    }


def store_removed_landscape(land, seed: int):
    """Charter section C.2 -- the atoms' CONTENT destroyed, the address book untouched.

    Uses the INHERITED ``refusal_lib.shuffled_store`` verbatim: each channel column of each
    well's atom matrix permuted independently among the N atoms.  Byte count unchanged,
    every channel marginal unchanged, no atom a train row any more.  Exactly ONE declared
    quantity differs from the full arm -- and the posterior P is bit-identical between the
    two arms, which is precisely what makes this a test of the STORE and not of the book.
    """
    return AtomLandscape(R.shuffled_store(land.store, int(seed)),
                         np.asarray(land.sigma), land.coupling,
                         float(land.request_strength), float(land.coupling_strength))


# ---------------------------------------------------------------------------
# Scoring -- the shipped threshold rule, and the PAIR that is the result
# ---------------------------------------------------------------------------


def auroc(neg, pos) -> float:
    """Shipped ``binary_auroc``; ``pos`` is the set that SHOULD be refused (high score)."""
    s = np.concatenate([np.asarray(neg, np.float64), np.asarray(pos, np.float64)])
    y = np.concatenate([np.zeros(len(neg), bool), np.ones(len(pos), bool)])
    return float(binary_auroc(s, y))


def refuse_frac(scores, threshold) -> float:
    return float(knn.refusal_rates(np.asarray(scores), float(threshold))["refused_fraction"])


def score_one(stat_by_set: dict, q: float, valid_key: str = "val") -> dict:
    """The declared rule, applied once: threshold on ``val`` only, then every rate.

    Returns the (reject-invalid, accept-unseen-valid) PAIR -- which is the result --
    with AUROC as the companion, never the other way round.
    """
    thr = float(knn.threshold_from_valid(np.asarray(stat_by_set[valid_key]), float(q)))
    return {
        "threshold_from_val_only": thr, "q": float(q),
        "refused_fraction": {k: refuse_frac(v, thr) for k, v in stat_by_set.items()},
        "accepted_fraction": {k: 1.0 - refuse_frac(v, thr) for k, v in stat_by_set.items()},
        "mean": {k: float(np.mean(v)) for k, v in stat_by_set.items()},
        "n": {k: int(np.asarray(v).size) for k, v in stat_by_set.items()},
    }


def boot_ci_auroc(neg, pos, n_boot: int = 2000, seed: int = 20260917, alpha: float = 0.05):
    """Percentile bootstrap CI on a single AUROC (Head ruling R1 needs an interval)."""
    rng = np.random.default_rng(int(seed))
    a, b = np.asarray(neg, np.float64), np.asarray(pos, np.float64)
    if a.size == 0 or b.size == 0:
        return (float("nan"), float("nan"))
    vals = np.empty(int(n_boot), np.float64)
    for i in range(int(n_boot)):
        vals[i] = auroc(rng.choice(a, a.size, replace=True),
                        rng.choice(b, b.size, replace=True))
    return (float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2)))


def paired_delta_ci(neg_a, pos_a, neg_b, pos_b, n_boot: int = 2000,
                    seed: int = 20260917, alpha: float = 0.05):
    """PAIRED bootstrap CI on ``AUROC(a) - AUROC(b)`` -- the same resampled rows for both.

    This is the interval Head ruling R1 thresholds: tie if it includes 0 or |delta| <= 0.020.
    """
    rng = np.random.default_rng(int(seed))
    na, pa = np.asarray(neg_a, np.float64), np.asarray(pos_a, np.float64)
    nb, pb = np.asarray(neg_b, np.float64), np.asarray(pos_b, np.float64)
    if na.size != nb.size or pa.size != pb.size:
        raise ValueError("paired_delta_ci needs row-aligned scores for both statistics")
    d = np.empty(int(n_boot), np.float64)
    for i in range(int(n_boot)):
        ii = rng.integers(0, na.size, na.size)
        jj = rng.integers(0, pa.size, pa.size)
        d[i] = auroc(na[ii], pa[jj]) - auroc(nb[ii], pb[jj])
    lo, hi = float(np.quantile(d, alpha / 2)), float(np.quantile(d, 1 - alpha / 2))
    point = auroc(na, pa) - auroc(nb, pb)
    m = 0.020
    if lo <= 0.0 <= hi or abs(point) <= m:
        verdict = "tie"
    elif point > 0:
        verdict = "win"
    elif point < -m:
        verdict = "loss"
    else:
        verdict = "tie"
    return {"delta": point, "ci95": [lo, hi], "R1_verdict": verdict, "m": m}
