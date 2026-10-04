"""Window descriptors for the exemplar store.

Four descriptors, from cheapest to richest. Two are context-independent
(`chan_stats`, `last_patch`); two grow with the context rung (`patch_means`,
`raw_window`). That asymmetry is deliberate: it lets the tripwire spend a
growing byte budget on *more history* as well as on *more exemplars*, so a HEPA
win at a long rung cannot be attributed to the reference being frozen at 512.

Conventions, declared once:

* **Right-alignment.** Every descriptor is anchored at the prediction time t and
  padded with zeros at the FRONT for windows shorter than the rung. HEPA's
  PatchEmbedding instead anchors at the window start and right-pads. We choose
  recency-alignment because it is the sane choice for a nearest-neighbour
  reference and it can only make the tripwire stronger -- the direction we want
  an instrument built to kill a KEEP to err in.
* **Zero padding** is post-normalisation, so it is the (approximate) channel
  mean, not an out-of-range sentinel.
* **RevIN parity.** When the dataset's norm policy is 'revin', each window is
  per-channel z-scored before featurisation, matching
  hepa/model/encoder.py::RevIN (mean over time, std with torch's unbiased
  ddof=1, eps=1e-5). When it is 'none' (C-MAPSS) the loader has already applied
  a global z-score and we apply nothing, exactly as the encoder does.
"""

from __future__ import annotations

import os
from typing import Dict, Iterator, List, Sequence, Tuple

import numpy as np

from tw_config import PATCH_SIZE

REVIN_EPS = 1e-5

# --------------------------------------------------------------------------
# Wave-2 additions. BOTH ARE NUMERICALLY INERT -- they change how much memory a
# rung needs and how often a feature matrix is recomputed, never a value.
#   * CHUNK_TARGET_BYTES: the query block is sized so one block stays near this
#     many bytes. At L<=512 every descriptor keeps the wave-1 chunk of 512 rows
#     (checked: the largest wave-1 dim, raw_window on FD001, is 7 168 -> 2 232,
#     which clamps back to 512), so the wave-1 acceptance run is untouched. At
#     L=8192 raw_window would otherwise allocate ~0.4 GB per block.
#   * _QCACHE: query features do not depend on the byte budget or on the seed --
#     only the store's subsample does. Wave 1 recomputed them once per
#     (budget x seed x split); at 8192 that is the dominant cost.
# --------------------------------------------------------------------------
CHUNK_TARGET_BYTES: int = int(os.environ.get("TRIPWIRE_CHUNK_BYTES", 64_000_000))
QCACHE_MAX_BYTES: int = int(os.environ.get("TRIPWIRE_QCACHE_BYTES", 1_200_000_000))

_QCACHE: "Dict[Tuple, Tuple[object, np.ndarray]]" = {}
_QCACHE_BYTES: int = 0


def cache_clear() -> None:
    global _QCACHE_BYTES
    _QCACHE.clear()
    _QCACHE_BYTES = 0


def cache_stats() -> Dict[str, int]:
    return {"n_entries": len(_QCACHE), "bytes": int(_QCACHE_BYTES),
            "max_bytes": QCACHE_MAX_BYTES}


def feature_dim(descriptor: str, n_channels: int, context_len: int) -> int:
    """Dimension of `descriptor` at this rung. Pure arithmetic; no data."""
    C, L, P = n_channels, context_len, PATCH_SIZE
    n_tok = int(np.ceil(L / P))
    if descriptor == "chan_stats":
        return 4 * C
    if descriptor == "last_patch":
        return P * C
    if descriptor == "patch_means":
        return n_tok * C
    if descriptor == "raw_window":
        return L * C
    raise ValueError(f"unknown descriptor {descriptor!r}")


def _revin(x: np.ndarray) -> np.ndarray:
    """encoder.py::RevIN with no padding mask: (x - mean) / (std + eps)."""
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True, ddof=1) if x.shape[0] > 1 else np.zeros_like(mean)
    return (x - mean) / (std + REVIN_EPS)


def _right_align(x: np.ndarray, length: int) -> np.ndarray:
    """Pad-front / truncate-front so the last row of the result is the last row of x."""
    T, C = x.shape
    if T == length:
        return x
    if T > length:
        return x[-length:]
    out = np.zeros((length, C), dtype=np.float32)
    out[length - T:] = x
    return out


def _one(x: np.ndarray, descriptor: str, context_len: int) -> np.ndarray:
    T, C = x.shape
    if descriptor == "chan_stats":
        tt = np.arange(T, dtype=np.float32)
        tt = (tt - tt.mean()) / (tt.std() + 1e-8)
        slope = (tt[:, None] * (x - x.mean(axis=0, keepdims=True))).mean(axis=0)
        return np.concatenate(
            [x.mean(axis=0), x.std(axis=0), x[-1], slope]
        ).astype(np.float32)
    if descriptor == "last_patch":
        return _right_align(x, PATCH_SIZE).reshape(-1).astype(np.float32)
    if descriptor == "patch_means":
        n_tok = int(np.ceil(context_len / PATCH_SIZE))
        xa = _right_align(x, n_tok * PATCH_SIZE)
        return xa.reshape(n_tok, PATCH_SIZE, C).mean(axis=1).reshape(-1).astype(np.float32)
    if descriptor == "raw_window":
        return _right_align(x, context_len).reshape(-1).astype(np.float32)
    raise ValueError(f"unknown descriptor {descriptor!r}")


def build(
    contexts: Sequence[np.ndarray],
    descriptor: str,
    context_len: int,
    norm_mode: str,
) -> np.ndarray:
    """(N, feature_dim) float32 feature matrix."""
    if not contexts:
        return np.zeros((0, 0), dtype=np.float32)
    d = feature_dim(descriptor, contexts[0].shape[1], context_len)
    out = np.empty((len(contexts), d), dtype=np.float32)
    for i, x in enumerate(contexts):
        xx = _revin(x) if norm_mode == "revin" else x
        out[i] = _one(xx, descriptor, context_len)
    return out


def _chunk_rows(descriptor: str, n_channels: int, context_len: int, chunk: int) -> int:
    """Rows per query block, capped so one block stays near CHUNK_TARGET_BYTES."""
    d = feature_dim(descriptor, n_channels, context_len)
    cap = max(8, CHUNK_TARGET_BYTES // (4 * max(d, 1)))
    return int(max(1, min(chunk, cap)))


def iter_chunks(
    contexts: Sequence[np.ndarray],
    descriptor: str,
    context_len: int,
    norm_mode: str,
    chunk: int = 512,
) -> Iterator[tuple]:
    """Yield (start_index, feature_block) so long rungs never materialise (N, L*C).

    At L=8192 with 20k windows, `raw_window` is a 9 GB matrix; this keeps the
    peak near CHUNK_TARGET_BYTES. The store itself is always small by
    construction.

    If the FULL matrix fits QCACHE_MAX_BYTES it is built once and memoised on
    (identity of the context sequence, descriptor, rung, norm mode); subsequent
    passes -- the other byte budgets and the other seeds, which share the same
    query features by construction -- yield slices of it. Same floats either
    way; `build` is deterministic and depends on none of budget, seed or k.
    """
    global _QCACHE_BYTES
    n = len(contexts)
    if n == 0:
        return
    n_ch = int(contexts[0].shape[1])
    rows = _chunk_rows(descriptor, n_ch, context_len, chunk)
    total = n * feature_dim(descriptor, n_ch, context_len) * 4
    if total <= QCACHE_MAX_BYTES:
        key = (id(contexts), descriptor, int(context_len), norm_mode, n)
        hit = _QCACHE.get(key)
        if hit is None:
            blocks = [
                build(list(contexts[s : s + rows]), descriptor, context_len, norm_mode)
                for s in range(0, n, rows)
            ]
            M = np.concatenate(blocks, axis=0) if len(blocks) > 1 else blocks[0]
            while _QCACHE and _QCACHE_BYTES + M.nbytes > QCACHE_MAX_BYTES:
                _, ev = _QCACHE.pop(next(iter(_QCACHE)))
                _QCACHE_BYTES -= int(ev.nbytes)
            # hold a reference to `contexts` so its id() can never be recycled
            _QCACHE[key] = (contexts, M)
            _QCACHE_BYTES += int(M.nbytes)
        else:
            M = hit[1]
        for s in range(0, n, rows):
            yield s, M[s : s + rows]
        return
    for s in range(0, n, rows):
        block: List[np.ndarray] = list(contexts[s : s + rows])
        yield s, build(block, descriptor, context_len, norm_mode)
