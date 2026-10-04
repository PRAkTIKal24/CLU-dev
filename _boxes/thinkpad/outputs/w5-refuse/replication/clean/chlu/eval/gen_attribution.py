"""⭐ THREE-POINT ATTRIBUTION — where does ``scale`` get lost: encoder, well, or code?

⭐ **DIAL: attribution.**  ⛔ No fix, no new mechanism, no arm.  This module locates
a **known** loss (R31/R34: the 30-bit read carries HUE and SHAPE and discards SCALE
and ORIENTATION, globally, not only inside Montero's corner); it does not repair it.

⛔ **What the path actually is** (R40.B, binding).  The phase-1 compose path is a
**product quantiser + count table**: a correlation-partitioned set of wells over the
2048 feature channels, a per-well **k-means** codebook, an address book of per-factor-
**value** mode histograms combined naive-Bayes and **argmax**'d per well, and a read
that is a **codebook copy-back**.  ⛔ There is no landscape, no particle, no settling
and no inter-well coupling on this path, and ⛔ the phrase "k-particle launch read"
names an ``argmax`` over a count table.  The working hypothesis for the loss is
therefore mundane and testable: **k-means keeps high-variance directions**, and
``scale``/``orientation`` are low-variance in these features.

**The three points, on the SAME rows with the SAME probe:**

1. ``encoder`` — the frozen 2048-d feature vector;
2. ``well``    — the slice of those features assigned to one well (``K`` fits);
3. ``code``    — the integer K-tuple after k-means + ``argmax``, one-hot encoded.

⭐ **Why point 2 must be per-well and never concatenated:** the wells *partition* the
coordinates (``sum_k d_k == D``), so the concatenation of every sub-vector **is** the
full feature vector up to a column permutation.  The only informative point-2
statistic is the best single well, and what it separates is *"the partition threw the
factor away"* from *"k-means threw it away"* — different repairs.

⛔ **Three live traps this module is written around.**

- **The same probe at all three points.**  :func:`probe_point` takes one probe
  config and one PRNG key and is the only fitting entry point here.  A linear probe
  on a 30-bit code and on a 2048-d vector are ⛔ **not** equally expressive, so two
  capacity bounds travel with every code number: the **random-code control**
  (:func:`random_codes`) and the non-parametric **tuple-lookup upper bound**
  (:func:`lookup_upper_bound`), which is strictly more expressive than any linear
  head on the one-hot.
- **A per-factor statistic is meaningless without ``n_values_realised``** (R31,
  fourth strike): :func:`n_values_realised` is computed for every row set and
  :func:`per_factor_rows` refuses to emit a number without it.
- **The loss is GLOBAL** (R34): the headline rows are the **spread** grid; the corner
  is reported separately and never as the global profile.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np

from ..core.gen_store import (
    WellProbe,
    train_well_probe,
    well_probe_accuracy,
    well_relevance,
)

#: The three points, in pipeline order, plus the control.  ⛔ Order is load-bearing:
#: retention is always read left to right.
POINTS = ("encoder", "well", "code")
CONTROL_POINT = "code_random"


# ---------------------------------------------------------------------------
# 1. Code representations
# ---------------------------------------------------------------------------


def onehot_codes(codes: np.ndarray, n_modes: int) -> np.ndarray:
    """``(N, K)`` integer codes -> ``(N, K*M)`` one-hot, wells concatenated.

    ⭐ This is the **most** a linear head can be given about the code without
    inventing structure: it makes every (well, mode) pair its own feature, so the
    probe is an additive lookup over modes.  ⛔ It still cannot represent
    interactions *between* wells — which is exactly why
    :func:`lookup_upper_bound` exists.
    """
    codes = np.asarray(codes, np.int64)
    if codes.ndim != 2:
        raise ValueError(f"codes must be (N, K), got {codes.shape}")
    n, k = codes.shape
    m = int(n_modes)
    if codes.size and (codes.min() < 0 or codes.max() >= m):
        raise ValueError(f"codes out of range [0, {m}) — saw [{codes.min()}, {codes.max()}]")
    out = np.zeros((n, k * m), np.float32)
    if n:
        rows = np.repeat(np.arange(n, dtype=np.int64), k)
        cols = (np.arange(k, dtype=np.int64)[None, :] * m + codes).ravel()
        out[rows, cols] = 1.0
    return out


def random_codes(n: int, n_wells: int, n_modes: int, seed: int) -> np.ndarray:
    """⭐ **The mandatory control** (task trap 1): a code of the SAME width carrying
    **no** information.  It bounds what the code-level probe can score with nothing.

    ⛔ If this scores materially above the majority baseline, every "the store drops
    it" conclusion drawn from a code-level probe is unsupported.
    """
    rng = np.random.default_rng(int(seed))
    return rng.integers(0, int(n_modes), size=(int(n), int(n_wells)), dtype=np.int64)


# ---------------------------------------------------------------------------
# 2. ⛔ THE ONE PROBE — the same class, hyper-parameters and key at every point
# ---------------------------------------------------------------------------


def probe_point(
    name: str,
    Z_train: np.ndarray,
    Y_train: np.ndarray,
    evals: dict,
    cardinalities: Sequence[int],
    factor_names: Sequence[str],
    cfg,
    key,
    log: Optional[Callable] = None,
) -> dict:
    """Fit ONE :class:`~chlu.core.gen_store.WellProbe` on ``Z_train`` and score it
    on every ``(name -> (Z, Y))`` in ``evals``.

    ⭐ ``train`` accuracy is reported beside every test accuracy, so an
    **under-trained** probe (which would look exactly like an information loss)
    is visible rather than inferred.  ⛔ Nothing in the signature lets a caller
    give one point more steps than another.
    """
    Z_train = np.asarray(Z_train, np.float32)
    probe, norm, hist = train_well_probe(
        Z_train, Y_train, list(cardinalities), cfg, key, log=None
    )
    out = {
        "point": str(name),
        "input_dim": int(Z_train.shape[1]),
        "n_train": int(Z_train.shape[0]),
        "probe_steps": int(cfg.probe_steps),
        "probe_batch": int(cfg.probe_batch),
        "probe_lr": float(cfg.probe_lr),
        "loss_first": float(hist[0]) if hist else float("nan"),
        "loss_last": float(hist[-1]) if hist else float("nan"),
        "train": well_probe_accuracy(probe, norm, Z_train, Y_train, factor_names),
        "eval": {
            k: well_probe_accuracy(probe, norm, np.asarray(z, np.float32), y, factor_names)
            for k, (z, y) in evals.items()
        },
    }
    if log is not None:
        log(
            f"      [{name:>11s}] d={out['input_dim']:>4d} loss "
            f"{out['loss_first']:.3f}->{out['loss_last']:.3f} "
            f"train_agg={out['train']['aggregate']:.4f} "
            + " ".join(f"{k}={v['aggregate']:.4f}" for k, v in out["eval"].items())
        )
    return out


# ---------------------------------------------------------------------------
# 3. Baselines, capacity bounds and retention
# ---------------------------------------------------------------------------


def chance_rates(cardinalities: Sequence[int], factor_names: Sequence[str]) -> dict:
    """Uniform chance ``1 / C`` per factor."""
    return {
        str(nm): float(1.0 / int(c))
        for nm, c in zip(factor_names, cardinalities, strict=True)
    }


def majority_baseline(
    Y_train: np.ndarray, Y_eval: np.ndarray, factor_names: Sequence[str]
) -> dict:
    """Train-majority class scored on ``Y_eval`` — the honest "no information" floor.

    ⛔ Uniform chance is the *nominal* floor; on an unbalanced row set the majority
    rate is higher, and the "≈ chance" clause of the pre-registered reading is
    adjudicated against the larger of the two.
    """
    Y_train = np.asarray(Y_train, np.int64)
    Y_eval = np.asarray(Y_eval, np.int64)
    out = {}
    for j, nm in enumerate(factor_names):
        if Y_train.shape[0] == 0 or Y_eval.shape[0] == 0:
            out[str(nm)] = float("nan")
            continue
        maj = int(np.bincount(Y_train[:, j]).argmax())
        out[str(nm)] = float((Y_eval[:, j] == maj).mean())
    return out


def lookup_upper_bound(
    codes_train: np.ndarray,
    Y_train: np.ndarray,
    codes_eval: np.ndarray,
    Y_eval: np.ndarray,
    factor_names: Sequence[str],
) -> dict:
    """⭐ The **capacity bound** on the code point: a non-parametric lookup on the
    FULL K-tuple (majority factor value per observed tuple, global majority for an
    unseen tuple).

    Strictly more expressive than a linear head on the one-hot — it represents
    arbitrary interactions between wells, which an additive linear probe cannot.
    ⇒ if this is also near chance, the code-level loss ⛔ cannot be blamed on probe
    capacity.  ``coverage`` is the fraction of eval rows whose tuple was seen in
    training; a low coverage means the bound is mostly the global majority and must
    be read as such.
    """
    ctr = np.asarray(codes_train, np.int64)
    cev = np.asarray(codes_eval, np.int64)
    Ytr = np.asarray(Y_train, np.int64)
    Yev = np.asarray(Y_eval, np.int64)
    key_tr = [tuple(int(x) for x in r) for r in ctr]
    key_ev = [tuple(int(x) for x in r) for r in cev]
    seen = {}
    for i, k in enumerate(key_tr):
        seen.setdefault(k, []).append(i)
    per_factor = {}
    for j, nm in enumerate(factor_names):
        if Ytr.shape[0] == 0 or Yev.shape[0] == 0:
            per_factor[str(nm)] = float("nan")
            continue
        glob = int(np.bincount(Ytr[:, j]).argmax())
        table = {
            k: int(np.bincount(Ytr[np.asarray(idx, np.int64), j]).argmax())
            for k, idx in seen.items()
        }
        pred = np.asarray([table.get(k, glob) for k in key_ev], np.int64)
        per_factor[str(nm)] = float((pred == Yev[:, j]).mean())
    cov = (
        float(np.mean([k in seen for k in key_ev])) if key_ev else float("nan")
    )
    return {
        "per_factor": per_factor,
        "aggregate": float(np.mean(list(per_factor.values()))) if per_factor else float("nan"),
        "n_distinct_tuples_train": int(len(seen)),
        "n_distinct_tuples_eval": int(len(set(key_ev))),
        "coverage": cov,
        "note": (
            "majority factor value per observed K-tuple; strictly more expressive "
            "than a linear head on the one-hot, so it BOUNDS the probe-capacity "
            "objection rather than arguing about it"
        ),
    }


def retention(acc_point: float, acc_encoder: float, chance: float) -> float:
    """``(acc_point - chance) / (acc_encoder - chance)`` — the fraction of the
    encoder's **above-chance** signal surviving to this point.

    ⛔ ``nan`` (not 0, not 1) when the encoder itself is at or below chance: a ratio
    against a zero-signal denominator is not a measurement.
    """
    denom = float(acc_encoder) - float(chance)
    if not np.isfinite(denom) or denom <= 1e-9:
        return float("nan")
    return float((float(acc_point) - float(chance)) / denom)


# ---------------------------------------------------------------------------
# 4. ⛔ n_values_realised — R31's fourth-strike rule, enforced structurally
# ---------------------------------------------------------------------------


def n_values_realised(
    labels: np.ndarray, factor_names: Sequence[str], cardinalities: Sequence[int]
) -> dict:
    """How many values of each factor the row set actually realises.

    ⚠ Montero's corner pins ``shape`` to ONE value; a statistic on a one-valued
    factor yields a confident-looking zero that reads exactly like a finding
    (R31's erratum).  Every per-factor number in this module carries this count.
    """
    Y = np.asarray(labels, np.int64)
    return {
        str(nm): {
            "n_values_realised": int(np.unique(Y[:, j]).size) if Y.size else 0,
            "cardinality": int(c),
        }
        for j, (nm, c) in enumerate(zip(factor_names, cardinalities, strict=True))
    }


def per_factor_rows(
    acc: dict, nvr: dict, factor_names: Sequence[str], **extra
) -> dict:
    """Zip an accuracy dict with ``n_values_realised``; mark 1-valued factors.

    ⛔ Refuses to emit a bare per-factor number: a factor realising one value is
    stamped ``NOT APPLICABLE`` and its accuracy is carried as ``degenerate``.
    """
    out = {}
    for nm in factor_names:
        nm = str(nm)
        n = int(nvr[nm]["n_values_realised"])
        row = {
            "accuracy": float(acc["per_factor"][nm]),
            "n_values_realised": n,
            "cardinality": int(nvr[nm]["cardinality"]),
            "degenerate": bool(n <= 1),
        }
        if n <= 1:
            row["status"] = "NOT APPLICABLE (1 realised value — R31)"
        for k, v in extra.items():
            row[k] = float(v[nm]) if isinstance(v, dict) else v
        out[nm] = row
    return out


# ---------------------------------------------------------------------------
# 5. ⭐ THE FREE INSTRUMENT — the store's own per-well MI, read out and tabulated
# ---------------------------------------------------------------------------


def mi_table(
    codes: np.ndarray,
    labels: np.ndarray,
    cardinalities: Sequence[int],
    factor_names: Sequence[str],
) -> dict:
    """``I(code_k ; factor_f)`` in bits per well, normalised by ``H(f)``.

    ⭐ This costs nothing — it is the matrix the store already computes for DCI —
    and it is **an independent view of the same fact from the address book's own
    side**: a flat ``scale`` row *is* the collapse.  ⛔ If it disagrees with the
    probe, the disagreement is reported; it is ⛔ not resolved in favour of either.
    """
    rel = well_relevance(np.asarray(codes, np.int64), np.asarray(labels, np.int64),
                         list(cardinalities))
    R = np.asarray(rel["mi_bits"], np.float64)
    H = np.asarray(rel["factor_entropy_bits"], np.float64)
    per_factor = {}
    for j, nm in enumerate(factor_names):
        col = R[:, j]
        per_factor[str(nm)] = {
            "mi_bits_per_well": [float(x) for x in col],
            "max_mi_bits": float(col.max()) if col.size else float("nan"),
            "sum_mi_bits": float(col.sum()),
            "entropy_bits": float(H[j]),
            "max_fraction_of_entropy": float(col.max() / H[j]) if H[j] > 0 else float("nan"),
        }
    return {
        "mi_bits": R.tolist(),
        "factor_entropy_bits": [float(x) for x in H],
        "per_factor": per_factor,
        "n_wells": int(R.shape[0]),
    }


# ---------------------------------------------------------------------------
# 6. ⭐ THE PRE-REGISTERED VERDICT RULE — mechanical, PREREG §4
# ---------------------------------------------------------------------------

#: The three words the spoke is allowed to return.  ⛔ Nothing else.
VERDICTS = ("World A", "World B", "in-between")


def attribution_verdict(
    acc_encoder: float,
    acc_code: float,
    chance: float,
    majority: float,
    acc_code_random: float,
    encoder_floor: float = 0.90,
    code_slack: float = 0.05,
) -> dict:
    """PREREG §4, applied mechanically.

    - **World A** iff ``acc_encoder >= encoder_floor`` **and**
      ``acc_code <= max(majority, acc_code_random) + code_slack``.
    - **World B** iff ``acc_encoder < (chance + encoder_floor) / 2`` — the
      operational reading of *"the encoder lacks it"*.
    - ⛔ **Otherwise IN-BETWEEN**, in those words, with the retention naming the
      direction.  ⛔ Never rounded to whichever world is more convenient.
    """
    floor = float(encoder_floor)
    b_thresh = 0.5 * (float(chance) + floor)
    code_floor = float(np.nanmax([float(majority), float(acc_code_random)]))
    code_at_chance = bool(float(acc_code) <= code_floor + float(code_slack))
    enc_has = bool(float(acc_encoder) >= floor)
    enc_lacks = bool(float(acc_encoder) < b_thresh)
    if enc_has and code_at_chance:
        verdict = "World A"
    elif enc_lacks:
        verdict = "World B"
    else:
        verdict = "in-between"
    return {
        "verdict": verdict,
        "acc_encoder": float(acc_encoder),
        "acc_code": float(acc_code),
        "chance": float(chance),
        "majority": float(majority),
        "acc_code_random": float(acc_code_random),
        "code_floor_used": code_floor,
        "encoder_floor": floor,
        "world_b_threshold": float(b_thresh),
        "encoder_has_it": enc_has,
        "encoder_lacks_it": enc_lacks,
        "code_at_chance": code_at_chance,
        "retention_code": retention(acc_code, acc_encoder, chance),
        "leaning": (
            "World A" if (code_at_chance and not enc_lacks) else
            "World B" if enc_lacks else "neither"
        ),
    }


def summarize_across_seeds(values: Sequence[float]) -> dict:
    """Mean / population sd (``ddof=0``) / per-seed values.

    ⚠ ``ddof=0`` is declared here because R31 carried forward a convention clash
    (a report's ``±`` was sample sd while its flag table declared population sd).
    """
    v = np.asarray([float(x) for x in values], np.float64)
    finite = v[np.isfinite(v)]
    return {
        "mean": float(finite.mean()) if finite.size else float("nan"),
        "std": float(finite.std()) if finite.size else float("nan"),
        "n_seeds": int(finite.size),
        "ddof": 0,
        "per_seed": [float(x) for x in v],
    }


__all__ = [
    "CONTROL_POINT",
    "POINTS",
    "VERDICTS",
    "WellProbe",
    "attribution_verdict",
    "chance_rates",
    "lookup_upper_bound",
    "majority_baseline",
    "mi_table",
    "n_values_realised",
    "onehot_codes",
    "per_factor_rows",
    "probe_point",
    "random_codes",
    "retention",
    "summarize_across_seeds",
]
