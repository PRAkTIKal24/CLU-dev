"""The factor probe and **the decoder-ceiling gate** for the generative phase.

⭐ Why this is a gate and not a control (R3).  The one shared decoder is fit on
train-split pairs only, so it may simply be unable to *render* a combination it
never saw.  If so, B accuracy is low for **every** arm, store-removed matches
CLU, and kill rule (i) fires on the decoder rather than on the store — we would
spend the phase's compute proving something about a decoder and read it as a
result about the primitive.  So the ceiling is measured **before any arm
exists**, by pushing the **real cached features** of held-out-combination rows
through the frozen decoder and probing what comes out.

⛔ ``ceiling_B`` alone means nothing.  It is only ever read as a ratio to
``ceiling_A``, the A-split equivalent measured with the same probe, the same
decoder and the same protocol.

**PROCEED iff ``ceiling_B >= ceiling_floor * ceiling_A``** (default floor 0.8,
R3.1).  A ``STOP`` is a legitimate result and returns to Head + Advisor; it is
⛔ not a licence to tune the decoder until it clears the floor.
"""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ..core.gen_bridge import model_dtype

#: R3.1 pre-registered floor: ceiling_B must be at least this fraction of ceiling_A.
DEFAULT_CEILING_FLOOR = 0.8


def factor_labels(grid, rows: np.ndarray) -> np.ndarray:
    """Per-factor class indices ``(N, n_factors)`` for ``rows`` (grid bijection)."""
    return np.asarray(grid.factor_indices_of(np.asarray(rows, np.int64)), np.int32)


def predict_factors(probe, images: np.ndarray, batch: int = 512) -> np.ndarray:
    """Top-1 factor predictions ``(N, n_factors)`` for a batch of images."""
    dtype = model_dtype(probe)
    fwd = eqx.filter_jit(jax.vmap(probe))
    preds = []
    for i in range(0, int(images.shape[0]), batch):
        logits = fwd(jnp.asarray(images[i : i + batch], dtype))
        preds.append(
            np.stack([np.asarray(jnp.argmax(lg, axis=-1)) for lg in logits], axis=1)
        )
    if not preds:
        return np.zeros((0, len(probe.heads.linears)), np.int32)
    return np.concatenate(preds).astype(np.int32)


def decode_features(decoder, features: np.ndarray, batch: int = 512) -> np.ndarray:
    """Render cached penultimate feature maps through the ONE frozen decoder."""
    dtype = model_dtype(decoder)
    fwd = eqx.filter_jit(jax.vmap(decoder))
    out = []
    for i in range(0, int(features.shape[0]), batch):
        out.append(np.asarray(fwd(jnp.asarray(features[i : i + batch], dtype)), np.float32))
    if not out:
        return np.zeros((0,) + tuple(decoder.out_shape), np.float32)
    return np.concatenate(out)


def probe_accuracy(
    probe, images: np.ndarray, labels: np.ndarray, factor_names, batch: int = 512
) -> dict:
    """Per-factor top-1 accuracy, the unweighted mean over factors, and exact match.

    ``aggregate`` (the number the gate reads) is the **unweighted mean over
    factors** — not a per-image exact match, which would be dominated by the
    single hardest head.
    """
    preds = predict_factors(probe, images, batch)
    labels = np.asarray(labels, np.int32)
    correct = preds == labels
    per_factor = {
        str(name): float(correct[:, j].mean()) if correct.size else float("nan")
        for j, name in enumerate(factor_names)
    }
    return {
        "n": int(labels.shape[0]),
        "per_factor": per_factor,
        "aggregate": float(np.mean(list(per_factor.values()))) if per_factor else float("nan"),
        "exact_match": float(correct.all(axis=1).mean()) if correct.size else float("nan"),
    }


def measure_ceiling(
    decoder,
    probe,
    features: np.ndarray,
    labels: np.ndarray,
    factor_names,
    batch: int = 512,
) -> dict:
    """Decode **real** cached features, probe the result: one ``ceiling_X``."""
    decoded = decode_features(decoder, features, batch)
    out = probe_accuracy(probe, decoded, labels, factor_names, batch)
    out["decoded_mean"] = float(decoded.mean())
    return out


def gate_verdict(
    ceiling_a: float, ceiling_b: float, floor: float = DEFAULT_CEILING_FLOOR
) -> dict:
    """⭐ The R3.1 gate.  ``PROCEED`` iff ``ceiling_B >= floor * ceiling_A``.

    ⛔ A ``STOP`` is a legitimate outcome and is reported as-is.
    """
    ratio = float(ceiling_b / ceiling_a) if ceiling_a > 0 else float("nan")
    # ``>=`` per R3.1. The ``rel_tol`` only resolves the float representation of an
    # EXACT tie (0.8 * 0.90 != 0.72 in binary); it is 1e-9 wide and cannot turn a
    # measured STOP into a PROCEED.
    threshold = floor * ceiling_a
    passed = bool(
        ceiling_a > 0
        and (ceiling_b >= threshold or math.isclose(ceiling_b, threshold, rel_tol=1e-9))
    )
    return {
        "ceiling_A": float(ceiling_a),
        "ceiling_B": float(ceiling_b),
        "ratio": ratio,
        "floor": float(floor),
        "verdict": "PROCEED" if passed else "STOP",
        "note": (
            "ceiling_B >= 0.8 x ceiling_A: the decoder can render held-out "
            "combinations, so kill rule (i) can be read on the store."
            if passed
            else "ceiling_B < 0.8 x ceiling_A: STOP — return to Head and Advisor. "
            "Running arms against this decoder would attribute a decoder failure "
            "to the store. ⛔ Do not tune the decoder to clear the floor."
        ),
    }


def summarize_seeds(per_seed: list[dict], key: str) -> dict:
    """Mean / std / min / max of ``rec[key]`` across seeds (std is population, ddof=0)."""
    vals = np.asarray([float(r[key]) for r in per_seed], np.float64)
    return {
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "min": float(vals.min()),
        "max": float(vals.max()),
        "n_seeds": int(vals.size),
        "values": [float(v) for v in vals],
    }


def summarize_per_factor(per_seed: list[dict], factor_names) -> dict:
    """Across-seed mean/std of each factor's accuracy."""
    out = {}
    for name in factor_names:
        vals = np.asarray([float(r["per_factor"][str(name)]) for r in per_seed])
        out[str(name)] = {"mean": float(vals.mean()), "std": float(vals.std())}
    return out


# ---------------------------------------------------------------------------
# ⭐ DISCRIMINATION (R20) — what kill rule (i) reads, and it is NOT detection
#
# Montero's held-out cell *is* ``shape == cylinder``, so it realises exactly ONE
# class: a probe that answers "cylinder" to everything in that colour corner
# scores 1.000 on it.  A bare held-out accuracy is therefore a **detection**
# rate, and ⛔ it may never be divided by a multi-class accuracy or put in the
# same column as one.  The like-for-like read is, **within the corner**:
#   positives = the held-out rows of the cell (all of its shape ``target``);
#   negatives = matched-seen rows in the SAME corner with a different, SEEN shape;
#   score     = balanced accuracy (argmax) and AUROC (the head's ``p(target)``).
# Balanced accuracy cannot go degenerate: answering ``target`` to everything
# gives TPR 1, TNR 0, i.e. exactly 0.5.
# ---------------------------------------------------------------------------

#: Floor on the ratio (held-out / seen normaliser) of the balanced metric (R3.1).
DEFAULT_DISC_FLOOR = 0.8
#: Absolute floor on the balanced metric itself (0.5 is chance).
DEFAULT_DISC_ABS_FLOOR = 0.75
#: ⛔ A balanced accuracy over a handful of negatives is not a measurement.
DEFAULT_DISC_MIN_NEGATIVES = 20


def factor_probs(probe, images: np.ndarray, factor_index: int, batch: int = 512) -> np.ndarray:
    """Softmax probabilities ``(N, C)`` of ONE factor head on ``images``.

    The argmax of this is exactly the column ``predict_factors`` returns for the
    same factor (asserted in ``tests/test_gen_bridge_wide.py``); the *margin* is
    what AUROC needs and what the argmax throws away.
    """
    dtype = model_dtype(probe)
    fwd = eqx.filter_jit(jax.vmap(probe))
    out = []
    for i in range(0, int(images.shape[0]), batch):
        logits = fwd(jnp.asarray(images[i : i + batch], dtype))
        out.append(np.asarray(jax.nn.softmax(logits[int(factor_index)], axis=-1), np.float64))
    if not out:
        return np.zeros((0, 0), np.float64)
    return np.concatenate(out)


def binary_auroc(scores, labels) -> float:
    """AUROC of ``scores`` against boolean ``labels`` (rank form, ties averaged).

    ⛔ No sklearn dependency; ties get the average rank, so a constant score
    returns exactly 0.5 rather than 0 or 1.
    """
    scores = np.asarray(scores, np.float64).ravel()
    labels = np.asarray(labels).astype(bool).ravel()
    if scores.size != labels.size:
        raise ValueError(f"scores {scores.size} != labels {labels.size}")
    n_pos = int(labels.sum())
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    srt = scores[order]
    ranks = np.empty(scores.size, np.float64)
    i = 0
    while i < srt.size:
        j = i
        while j + 1 < srt.size and srt[j + 1] == srt[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + 1 + j + 1)
        i = j + 1
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def balanced_accuracy(tpr: float, tnr: float) -> float:
    """``0.5 * (TPR + TNR)`` — the one metric that cannot go degenerate."""
    return float(0.5 * (float(tpr) + float(tnr)))


def discrimination_metrics(
    pos_probs: np.ndarray,
    neg_probs: np.ndarray,
    target: int,
    min_negatives: int = DEFAULT_DISC_MIN_NEGATIVES,
) -> dict:
    """⭐ One cell's DISCRIMINATION read: ``target`` vs matched-seen, in-corner.

    Args:
        pos_probs: ``(Np, C)`` probe softmax over the factor's classes for the
            **held-out** items (all of class ``target``), decoded.
        neg_probs: ``(Nn, C)`` the same for the **matched-seen negatives**
            (same corner, a different SEEN class), decoded.
        target: the class the held-out cell realises.
        min_negatives: below this the cell is reported ``sufficient=False`` and
            its numbers are ⛔ not poolable.

    Returns balanced accuracy, AUROC, TPR/TNR and the counts — plus
    ``detection_rate``, which is **the same number as TPR** and is labelled that
    way so it can never be mistaken for the balanced metric.
    """
    pos = np.asarray(pos_probs, np.float64)
    neg = np.asarray(neg_probs, np.float64)
    if pos.ndim != 2 or neg.ndim != 2 or (pos.size and neg.size and pos.shape[1] != neg.shape[1]):
        raise ValueError(f"expected (N, C) arrays with a common C, got {pos.shape} / {neg.shape}")
    t = int(target)
    n_pos, n_neg = int(pos.shape[0]), int(neg.shape[0])
    tpr = float((np.argmax(pos, axis=1) == t).mean()) if n_pos else float("nan")
    tnr = float((np.argmax(neg, axis=1) != t).mean()) if n_neg else float("nan")
    scores = np.concatenate([pos[:, t], neg[:, t]]) if (n_pos and n_neg) else np.zeros(0)
    labels = np.concatenate([np.ones(n_pos, bool), np.zeros(n_neg, bool)])
    return {
        "target": t,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "tpr": tpr,
        "tnr": tnr,
        "balanced_accuracy": balanced_accuracy(tpr, tnr),
        "auroc": binary_auroc(scores, labels) if (n_pos and n_neg) else float("nan"),
        # ⛔ DETECTION, not discrimination — the same number as TPR, kept only so
        # a reader can see the two quantities side by side and never conflated.
        "detection_rate": tpr,
        "sufficient": bool(n_neg >= int(min_negatives) and n_pos > 0),
    }


def multiclass_balanced_accuracy(y_true, y_pred, n_classes: int) -> dict:
    """Macro-averaged recall over the classes **present in** ``y_true``.

    ⭐ A different quantity from :func:`discrimination_metrics` — a 4-way read,
    reported in its own column and never divided into a binary one.
    """
    y_true = np.asarray(y_true, np.int64).ravel()
    y_pred = np.asarray(y_pred, np.int64).ravel()
    per_class, present = {}, []
    for c in range(int(n_classes)):
        m = y_true == c
        if not m.any():
            continue
        per_class[str(c)] = float((y_pred[m] == c).mean())
        present.append(c)
    return {
        "n": int(y_true.size),
        "classes_present": present,
        "per_class_recall": per_class,
        "balanced_accuracy": float(np.mean(list(per_class.values()))) if per_class else float("nan"),
        "accuracy": float((y_pred == y_true).mean()) if y_true.size else float("nan"),
    }


def pool_discrimination(per_cell: list, key: str = "balanced_accuracy") -> dict:
    """Macro-mean of ``key`` over the cells that have enough negatives."""
    used = [c for c in per_cell if c.get("sufficient", True)]
    vals = np.asarray([float(c[key]) for c in used], np.float64)
    return {
        "macro_mean": float(vals.mean()) if vals.size else float("nan"),
        "std": float(vals.std()) if vals.size else float("nan"),
        "min": float(vals.min()) if vals.size else float("nan"),
        "max": float(vals.max()) if vals.size else float("nan"),
        "n_cells": int(vals.size),
        "n_cells_dropped": int(len(per_cell) - len(used)),
        "values": [float(v) for v in vals],
    }


def discrimination_verdict(
    held_out: float,
    seen: float,
    floor: float = DEFAULT_DISC_FLOOR,
    abs_floor: float = DEFAULT_DISC_ABS_FLOOR,
) -> dict:
    """⭐ The pre-registered discrimination gate: BOTH floors must hold.

    ``PASS`` iff the pooled balanced metric is at least ``abs_floor`` absolute
    **and** at least ``floor`` of the seen-split normaliser.  ⛔ A ``FAIL`` is a
    legitimate result: it registers kill rule (i)'s ceiling at the measured
    level rather than at the 0.94 detection number.
    """
    ratio = float(held_out / seen) if seen > 0 else float("nan")
    passed = bool(
        np.isfinite(ratio)
        and float(held_out) >= float(abs_floor)
        and ratio >= float(floor)
    )
    return {
        "held_out": float(held_out),
        "seen": float(seen),
        "ratio": ratio,
        "floor": float(floor),
        "abs_floor": float(abs_floor),
        "verdict": "PASS" if passed else "FAIL",
    }
