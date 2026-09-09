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
