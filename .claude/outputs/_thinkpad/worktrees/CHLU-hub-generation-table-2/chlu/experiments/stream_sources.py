"""⭐ **C2W10 stream sources** — the synthetic regime-switcher, the frozen-stream
loader, and decimation.

Two sources, and they are **not** interchangeable:

**The synthetic regime-switcher** (:func:`make_regime_switcher`) is the wave's
second stream and the home of every designed negative: ``R`` hidden regimes over
a **shared input space** (the same X region maps to a different ``y`` per regime
— the INSECTS design in miniature), an **exact scripted revisit schedule** with
known change points, capacity pressure (more distinct items than well budget),
``k >= 3`` stream boundaries so L3's cross-stream criterion is computable, and a
**drift-free control condition**.
⛔ **Per §A14.8 the synthetic is a regression / mechanics instrument and NEVER a
claim venue.** Nothing measured on it is promotable, and every artifact derived
from it says so.

**The frozen real stream** (:func:`load_frozen_stream`) is loaded from **one
frozen file with one sha256**, named by
``.claude/outputs/c2w10-benchmark-gate/BENCHMARK-GATE.json``. ⭐ **The
reproduction gate is binding**: the loader recomputes the digest and refuses the
file when it does not match, *before* any number derived from it is consumed.
⛔ This module never downloads and never re-freezes a stream.

**Decimation** (:func:`decimate`, Head ruling 5, 2026-08-10) keeps every ``m``-th
instance from the registered ladder ``m in {1, 2, 5, 10}``. ⛔ Truncation is
REFUSED — it would delete the third cycle, i.e. the revisit, i.e. the benchmark.
⚠ **Decimation compresses the drift timeline**, so every stream carries
:attr:`Stream.since_change` and any adaptation-like quantity is reported **per
instance-since-change, never per stream-position**.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

#: the registered decimation ladder (PREREG-C2W10 §2.2b, Head ruling 5)
DECIMATION_LADDER = (1, 2, 5, 10)

#: the default scripted schedule: three regimes, each revisited once => 6
#: segments, 5 change points, and >= 3 stream boundaries for L3.
DEFAULT_SCHEDULE = (0, 1, 2, 0, 1, 2)


@dataclass
class Stream:
    """One prequential stream, with its hidden regime structure kept OUT of X.

    Attributes:
        X: ``(N, n_features)`` inputs. **The regime variable is not among them**
           (INSECTS withholds temperature by construction; so does this).
        y: ``(N,)`` labels.
        stream_id: ``(N,)`` which stream/segment each instance belongs to. The
            stream boundary is the unit L3's ``k`` counts.
        regime: ``(N,)`` the hidden regime — evaluation only, never a feature.
        change_points: instance indices at which the regime changes.
        since_change: ``(N,)`` instances since the last change point. ⭐ The only
            legal x-axis for an adaptation curve once decimation is on.
    """

    X: np.ndarray
    y: np.ndarray
    stream_id: np.ndarray
    regime: np.ndarray
    change_points: List[int]
    since_change: np.ndarray
    meta: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_streams(self) -> int:
        return int(len(np.unique(self.stream_id)))

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])

    def stream_slice(self, s: int):
        idx = np.nonzero(self.stream_id == int(s))[0]
        return self.X[idx], self.y[idx], idx


# ==========================================================================
# the synthetic regime-switcher (MECHANICS INSTRUMENT, never a claim venue)
# ==========================================================================
def regime_maps(n_regimes: int, n_features: int, n_classes: int,
                seed: int) -> np.ndarray:
    """``(R, n_classes, n_features)`` per-regime linear label maps.

    Each regime is an **independent** draw, so the same X region carries a
    different ``y`` per regime — which is what makes the store's memory of a
    regime worth anything and what makes a persistent store distinguishable from
    an episodic one at all.
    """
    rng = np.random.default_rng(int(seed) + 7717)
    W = rng.normal(size=(int(n_regimes), int(n_classes), int(n_features)))
    return W / np.linalg.norm(W, axis=-1, keepdims=True)


def label_of(X: np.ndarray, W_r: np.ndarray) -> np.ndarray:
    """``argmax_c <w_c, x>`` under one regime's map."""
    return np.asarray(np.argmax(np.asarray(X) @ np.asarray(W_r).T, axis=-1), dtype=int)


def make_regime_switcher(
    n_regimes: int = 3,
    n_per_stream: int = 64,
    n_features: int = 8,
    n_classes: int = 4,
    seed: int = 0,
    schedule: Optional[Sequence[int]] = None,
    drift_free: bool = False,
    n_anchors: int = 96,
    jitter: float = 0.02,
) -> Stream:
    """The scripted regime-switcher.

    Args:
        schedule: the exact revisit script, one regime index per stream. The
            default ``(0,1,2,0,1,2)`` gives **6 streams, 5 change points and one
            revisit per regime** — i.e. ``k = 3`` stream boundaries are available
            for L3, and the revisit the whole wave is about actually happens.
        n_anchors: ⭐ the number of **distinct addressable items**. Instances are
            drawn from a fixed anchor set (plus ``jitter``) that is **shared by
            every stream**, so a revisit returns to the *same addresses* and not
            merely to the same label map. Without this the store is asked to
            remember addresses no query ever visits again, its read lands in no
            basin (measured: 66 of 68 reads unassigned), the usage proxy is
            identically zero, and every lifecycle verb downstream of usage is
            unexercised for an instrument reason rather than a store reason.
            ``n_anchors > well_budget`` is this rig's **capacity pressure**.
        jitter: per-instance noise around the anchor, as a fraction of the
            anchor radius. It must stay well below half the admission spacing or
            coverage fails again; the rig reports the achieved coverage.
        drift_free: ⭐ **the control condition.** Every stream runs regime
            ``schedule[0]``, so there is nothing to re-learn: the *correct*
            behaviour of a persistent store here is **no benefit**, and a "win"
            on this condition falsifies the instrument rather than the baseline.
    """
    sched = list(DEFAULT_SCHEDULE if schedule is None else schedule)
    if drift_free:
        sched = [sched[0]] * len(sched)
    rng = np.random.default_rng(int(seed))
    W = regime_maps(int(n_regimes), int(n_features), int(n_classes), int(seed))
    anchors = rng.normal(size=(int(n_anchors), int(n_features)))
    anchors /= np.maximum(np.linalg.norm(anchors, axis=1, keepdims=True), 1e-12)
    Xs, ys, sid, reg = [], [], [], []
    for s, r in enumerate(sched):
        # ⭐ the SHARED input space: every stream visits the SAME anchors, so the
        # only thing that changes at a boundary is the map from address to label.
        pick = rng.integers(0, int(n_anchors), size=int(n_per_stream))
        Xt = anchors[pick] + float(jitter) * rng.normal(
            size=(int(n_per_stream), int(n_features)))
        Xt /= np.maximum(np.linalg.norm(Xt, axis=1, keepdims=True), 1e-12)
        Xs.append(Xt)
        # ⛔ the label follows the ANCHOR under the current regime, so the same
        # address genuinely carries a different y per regime.
        ys.append(label_of(anchors[pick], W[int(r) % int(n_regimes)]))
        sid.append(np.full((int(n_per_stream),), s, dtype=int))
        reg.append(np.full((int(n_per_stream),), int(r), dtype=int))
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    stream_id = np.concatenate(sid, axis=0)
    regime = np.concatenate(reg, axis=0)
    cps = _change_points(regime)
    return Stream(
        X=X, y=y, stream_id=stream_id, regime=regime, change_points=cps,
        since_change=_since_change(len(y), cps),
        meta={
            "source": "synthetic_regime_switcher",
            "role": ("MECHANICS INSTRUMENT — regression only; §A14.8 bars the "
                     "synthetic from being a claim venue"),
            "schedule": [int(v) for v in sched],
            "n_regimes": int(n_regimes), "n_classes": int(n_classes),
            "n_per_stream": int(n_per_stream), "seed": int(seed),
            "drift_free": bool(drift_free),
            "n_anchors": int(n_anchors), "jitter": float(jitter),
            "n_streams": len(sched), "n_boundaries": max(len(sched) - 1, 0),
            "decimation_m": 1,
        },
    )


def _change_points(regime: np.ndarray) -> List[int]:
    r = np.asarray(regime, dtype=int)
    return [int(i) for i in np.nonzero(r[1:] != r[:-1])[0] + 1]


def _since_change(n: int, change_points: Sequence[int]) -> np.ndarray:
    out = np.zeros((int(n),), dtype=int)
    last = 0
    cps = sorted(int(c) for c in change_points)
    j = 0
    for i in range(int(n)):
        if j < len(cps) and i == cps[j]:
            last = i
            j += 1
        out[i] = i - last
    return out


# ==========================================================================
# structure — asserted in a pytest, never claimed (Head ruling 5, condition 2)
# ==========================================================================
def structure_summary(stream: Stream) -> Dict[str, Any]:
    """The structure a decimated stream must still have, **with counts**."""
    r = np.asarray(stream.regime, dtype=int)
    runs: List[Dict[str, int]] = []
    start = 0
    for i in range(1, len(r) + 1):
        if i == len(r) or r[i] != r[start]:
            runs.append({"regime": int(r[start]), "start": int(start),
                         "n": int(i - start)})
            start = i
    seen: Dict[int, int] = {}
    for run in runs:
        seen[run["regime"]] = seen.get(run["regime"], 0) + 1
    return {
        "n_instances": int(len(r)),
        "n_runs": len(runs),
        "n_change_points": len(stream.change_points),
        "change_points": [int(c) for c in stream.change_points],
        "regime_sequence": [run["regime"] for run in runs],
        "counts_per_run": [run["n"] for run in runs],
        "n_revisits": int(sum(max(v - 1, 0) for v in seen.values())),
        "min_run": int(min((run["n"] for run in runs), default=0)),
    }


def structure_preserved(before: Dict[str, Any], after: Dict[str, Any]) -> bool:
    """Is the decimated stream still the same experiment?

    All three of: the **regime sequence** is unchanged (so every cycle and every
    revisit survives), the **change-point count** is unchanged, and **no segment
    is emptied**. A decimation that fails any of these has deleted a cycle, and
    the whole point of refusing truncation was to keep the revisit.
    """
    return bool(
        list(before["regime_sequence"]) == list(after["regime_sequence"])
        and int(before["n_change_points"]) == int(after["n_change_points"])
        and int(after["min_run"]) > 0
    )


def assert_structure_preserved(before: Dict[str, Any], after: Dict[str, Any],
                               m: int) -> None:
    if not structure_preserved(before, after):
        raise AssertionError(
            f"decimation m={m} does not preserve the stream's structure: "
            f"regimes {before['regime_sequence']} -> {after['regime_sequence']}, "
            f"change points {before['n_change_points']} -> "
            f"{after['n_change_points']}, min segment {after['min_run']}. "
            f"Truncation was refused for exactly this reason (Head ruling 5)."
        )


def decimate(stream: Stream, m: int) -> Stream:
    """Keep every ``m``-th instance. ⛔ Never truncate.

    ⚠ The drift timeline is **compressed** by ``m``: ``since_change`` is
    recomputed on the decimated index, and it is the only legal x-axis for an
    adaptation curve afterwards. ``meta['decimation_m']`` travels with the
    stream so the ledger cannot lose it.
    """
    m = int(m)
    if m < 1:
        raise ValueError(f"decimation m must be >= 1, got {m}")
    if m not in DECIMATION_LADDER:
        raise ValueError(
            f"m={m} is not on the registered ladder {DECIMATION_LADDER}; the "
            f"ladder is pre-registered and m is filed in PREREG-C2W10 §9 before "
            f"any claim cell runs"
        )
    idx = np.arange(0, len(stream), m)
    regime = np.asarray(stream.regime)[idx]
    cps = _change_points(regime)
    meta = dict(stream.meta)
    meta.update({"decimation_m": m, "n_instances_pre_decimation": len(stream)})
    return Stream(
        X=np.asarray(stream.X)[idx], y=np.asarray(stream.y)[idx],
        stream_id=np.asarray(stream.stream_id)[idx], regime=regime,
        change_points=cps, since_change=_since_change(len(idx), cps), meta=meta,
    )


def select_decimation(wall_s_at_m1: float, target_s: float,
                      ladder: Sequence[int] = DECIMATION_LADDER) -> Dict[str, Any]:
    """The registered selection rule: the **smallest** ``m`` on the ladder whose
    projected wall-clock meets the target (PREREG §P5).

    Reports the evidence, never the choice alone — the Hub files ``m`` into
    ``PREREG-C2W10.md`` §9 **before** any claim cell runs.
    """
    rows = [{"m": int(m), "projected_wall_s": float(wall_s_at_m1) / int(m),
             "meets_target": bool(float(wall_s_at_m1) / int(m) <= float(target_s))}
            for m in ladder]
    chosen = next((r["m"] for r in rows if r["meets_target"]), None)
    return {"rows": rows, "wall_s_at_m1": float(wall_s_at_m1),
            "target_s": float(target_s), "selected_m": chosen,
            "rule": "smallest m on the registered ladder meeting the wall-clock target"}


# ==========================================================================
# the frozen real stream — reproduce the digest FIRST, consume SECOND
# ==========================================================================
def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_benchmark_gate(path: str) -> Optional[Dict[str, Any]]:
    """The frozen file + its sha256, from ``BENCHMARK-GATE.json``.

    Returns ``None`` when the gate file does not exist — in which case the
    real-stream legs are a **declared NOT-RUN with that reason**, never a null,
    and this module does **not** download or freeze anything itself.
    """
    if not path or not os.path.exists(path):
        return None
    with open(path) as f:
        gate = json.load(f)
    for k_file in ("frozen_file", "file", "path", "stream_file"):
        if k_file in gate:
            break
    else:
        return gate
    return gate


def load_frozen_stream(path: str, sha256: Optional[str] = None, *,
                       feature_columns: Optional[Sequence[int]] = None,
                       label_column: int = -1,
                       change_points: Optional[Sequence[int]] = None,
                       stream_edges: Optional[Sequence[int]] = None,
                       delimiter: str = ",",
                       skip_header: int = 1) -> Stream:
    """Load ONE frozen stream file, **reproducing its digest first**.

    ⭐ The reproduction gate (PREREG-C2W10 §2.2b): the recorded sha256 is
    recomputed here and a mismatch **raises**, so no number derived from a
    different file than the one every other arm ran on can be consumed. One
    frozen file, one sha256, all arms.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"frozen stream {path!r} not present. ⛔ Do not re-download or "
            f"re-freeze it — one frozen file, one sha256, all arms."
        )
    got = sha256_of(path)
    if sha256 is not None and got.lower() != str(sha256).lower():
        raise ValueError(
            f"sha256 mismatch for {path!r}: recorded {sha256}, recomputed {got}. "
            f"Reproduce first, consume second — the arms are not comparable."
        )
    raw = np.genfromtxt(path, delimiter=delimiter, skip_header=int(skip_header))
    raw = np.atleast_2d(raw)
    cols = (list(range(raw.shape[1] - 1)) if feature_columns is None
            else [int(c) for c in feature_columns])
    X = np.asarray(raw[:, cols], dtype=float)
    y = np.asarray(raw[:, int(label_column)], dtype=int)
    n = X.shape[0]
    cps = [int(c) for c in (change_points or [])]
    edges = sorted({0, *[int(e) for e in (stream_edges or cps)], n})
    stream_id = np.zeros((n,), dtype=int)
    for s, (a, b) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        stream_id[a:b] = s
    regime = stream_id.copy()  # ⚠ OUR annotation unless the source publishes one
    return Stream(
        X=X, y=y, stream_id=stream_id, regime=regime, change_points=cps,
        since_change=_since_change(n, cps),
        meta={"source": "frozen_file", "path": path, "sha256": got,
              "n_instances": int(n), "decimation_m": 1,
              "regime_annotation": ("stream segments from the recorded change "
                                    "points; if the source does not publish them "
                                    "the annotation is OURS and every artifact "
                                    "must say so")},
    )


__all__ = [
    "DECIMATION_LADDER", "DEFAULT_SCHEDULE", "Stream", "regime_maps", "label_of",
    "make_regime_switcher", "structure_summary", "structure_preserved",
    "assert_structure_preserved", "decimate", "select_decimation", "sha256_of",
    "read_benchmark_gate", "load_frozen_stream",
]
