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
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from ..core.gen_bridge import model_dtype
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


