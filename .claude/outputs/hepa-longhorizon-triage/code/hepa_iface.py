"""The only module that touches HEPA. Everything else is ours.

Contract (task §2): the tripwire reads via HEPA's own loader and scores with
HEPA's own metric code. Concretely this module imports and uses, unmodified:

  hepa.data.load_dataset                       -- splits, channels, labels
  hepa.data._common.global_zscore_bundle       -- the C-MAPSS norm_mode='none' path
  hepa.utils.config.get_horizons/get_norm_mode -- the horizon grid and norm policy
  hepa.training.finetune.EventDataset          -- cut points, contexts, time-to-event
  hepa.evaluation.surface.build_label_surface  -- y(t, dt)
  hepa.evaluation.metrics.h_auroc              -- the metric

HEPA is CC BY-NC-SA 4.0. It is imported from `.claude/hepa-src` at run time and
never vendored, copied, or committed.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Sequence as _ABCSequence
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from tw_config import (
    HEPA_DATA_DIR,
    HEPA_SRC,
    MIN_ENTITY_LEN,
    STRIDE_TEST,
    STRIDE_TRAIN,
)


def _bootstrap() -> None:
    """Put the HEPA clone on sys.path and point its loader at our staged data."""
    if not HEPA_SRC.exists():
        raise FileNotFoundError(
            f"HEPA source not found at {HEPA_SRC}. It is cloned by the hepa-lh-audit "
            "spoke; set HEPA_SRC to override."
        )
    os.environ.setdefault("HEPA_DATA_DIR", str(HEPA_DATA_DIR))
    if str(HEPA_SRC) not in sys.path:
        sys.path.insert(0, str(HEPA_SRC))


_bootstrap()

import torch  # noqa: E402  (after _bootstrap so HEPA_DATA_DIR is set first)

from hepa.data import load_dataset  # noqa: E402
from hepa.data._common import global_zscore_bundle  # noqa: E402
from hepa.evaluation.metrics import h_auroc, per_horizon_auroc  # noqa: E402
from hepa.evaluation.surface import build_label_surface  # noqa: E402
from hepa.training.finetune import EventDataset  # noqa: E402
from hepa.utils.config import get_norm_mode  # noqa: E402


def hepa_provenance() -> Dict[str, str]:
    """Commit + dirty state of the HEPA clone every number is relative to."""

    def _git(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", "-C", str(HEPA_SRC), *args],
                capture_output=True, text=True, check=True,
            ).stdout.strip()
        except Exception as exc:  # provenance must never crash a run
            return f"UNAVAILABLE: {exc}"

    return {
        "hepa_src": str(HEPA_SRC),
        "hepa_commit": _git("rev-parse", "HEAD"),
        "hepa_commit_date": _git("log", "-1", "--format=%cI"),
        "hepa_dirty": str(bool(_git("status", "--porcelain"))),
        "hepa_data_dir": str(HEPA_DATA_DIR),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "python": sys.version.split()[0],
    }


# ---------------------------------------------------------------------------
# Window extraction -- a faithful numpy mirror of scripts/train.py's loaders
# ---------------------------------------------------------------------------


class LazyContexts(_ABCSequence):
    """EventDataset's context slice, materialised on demand as a numpy VIEW.

    WHY (wave 2): the wave-1 harness eagerly copied every window into its own
    array. That is fine at L=512 but at L=8192 a dataset such as GECCO
    (34 563 test cuts x 9 channels) needs ~10 GB of copies on a 16 GB machine,
    and the 8192 rung -- the rung the whole long-horizon question turns on --
    cannot be run at all.

    BIT-IDENTICAL BY CONSTRUCTION, not by hope. ``EventDataset.__getitem__``
    is exactly

        t = self.starts[i]; ctx_start = max(0, t - self.max_context)
        return self.x[ctx_start:t], ...          # self.x = float32 view of e["test"]

    so ``arr[max(0, t - L):t]`` on ``arr = np.asarray(e["test"], np.float32)``
    is the same bytes. Nothing is recomputed, resampled, or re-padded here; the
    front zero-padding still happens downstream in ``features._right_align``,
    unchanged.
    """

    __slots__ = ("_arrays", "_ent", "_t", "context_len")

    def __init__(
        self,
        arrays: List[np.ndarray],
        ent_idx: np.ndarray,
        t_idx: np.ndarray,
        context_len: int,
    ):
        self._arrays = arrays
        self._ent = np.asarray(ent_idx, dtype=np.int64)
        self._t = np.asarray(t_idx, dtype=np.int64)
        self.context_len = int(context_len)

    def __len__(self) -> int:
        return int(self._t.shape[0])

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(len(self)))]
        i = int(i)
        if i < 0:
            i += len(self)
        t = int(self._t[i])
        return self._arrays[int(self._ent[i])][max(0, t - self.context_len) : t]

    @property
    def n_channels(self) -> int:
        return int(self._arrays[0].shape[1]) if self._arrays else 0

    @property
    def lengths(self) -> np.ndarray:
        """Realised context length of every window, without materialising one."""
        return np.minimum(self._t, self.context_len)


@dataclass
class Windows:
    """One split, materialised as HEPA's own (context, tte, t) triples."""

    contexts: LazyContexts         # each (T_i, C) view, T_i <= context_len
    tte: np.ndarray                # (N,) float, inf = no event within max_future
    t_index: np.ndarray            # (N,) int64, the cut point within the entity
    age: np.ndarray                # (N,) int64, steps since this entity's previous event
    entity: np.ndarray             # (N,) object, entity id
    context_len: int

    def __len__(self) -> int:
        return len(self.contexts)

    @property
    def effective_context(self) -> np.ndarray:
        return self.contexts.lengths


def _prev_event_index(labels: np.ndarray) -> np.ndarray:
    """prev[t] = index of the last event STRICTLY before t, or -1 if none.

    -1 stands for a virtual event one step before the record starts, so that
    `age = t - prev[t]` is well defined and >= 1 everywhere, and for an entity
    with no prior event it degenerates to "time since the entity began". On
    C-MAPSS (one event, at the final cycle, never before a cut point) this makes
    last-event-age exactly the cycle index + 1.
    """
    T = len(labels)
    prev = np.full(T, -1, dtype=np.int64)
    last = -1
    for t in range(T):
        prev[t] = last
        if labels[t] == 1:
            last = t
    return prev


def _windows_from_entities(
    entities: List[dict], context_len: int, max_future: int, stride: int
) -> Windows:
    """Exactly scripts/train.py::_build_event_loader, minus the DataLoader.

    Same entity filter (`len(test) <= 128+1` dropped), same EventDataset (so the
    same min_context=128, the same cut points, the same time-to-event), same
    context slice `x[max(0, t-L):t]`.
    """
    arrays: List[np.ndarray] = []
    ent_idx: List[int] = []
    ttes: List[float] = []
    ts: List[int] = []
    ages: List[int] = []
    eids: List[object] = []
    for e in entities:
        if len(e["test"]) <= MIN_ENTITY_LEN:
            continue
        ds = EventDataset(
            e["test"], e["labels"], max_context=context_len,
            stride=stride, max_future=max_future,
        )
        prev = _prev_event_index(np.asarray(e["labels"], dtype=np.int32))
        # EventDataset.x is a float32 torch view of this array; .numpy() shares it.
        a = ds.x.numpy()
        ai = len(arrays)
        arrays.append(a)
        for i in range(len(ds)):
            _, tte, t = ds[i]
            ti = int(t.item())
            ent_idx.append(ai)
            ttes.append(float(tte.item()))
            ts.append(ti)
            ages.append(ti - int(prev[ti]))
            eids.append(e.get("entity_id"))
    return Windows(
        contexts=LazyContexts(arrays, ent_idx, ts, context_len),
        tte=np.asarray(ttes, dtype=np.float64),
        t_index=np.asarray(ts, dtype=np.int64),
        age=np.asarray(ages, dtype=np.int64),
        entity=np.asarray(eids, dtype=object),
        context_len=context_len,
    )


@dataclass
class Bundle:
    """A dataset at one context rung: three splits + the protocol metadata."""

    name: str
    n_channels: int
    horizons: List[int]
    norm_mode: str
    context_len: int
    train: Windows
    val: Windows
    test: Windows
    entity_lengths: Dict[str, List[int]]


def load_bundle(dataset: str, context_len: int) -> Bundle:
    """Load `dataset` through HEPA's loader and cut it into windows at context L."""
    bundle = load_dataset(dataset)
    norm_mode = get_norm_mode(dataset)
    if norm_mode == "none":
        # scripts/train.py does exactly this before anything else.
        bundle = global_zscore_bundle(bundle)
    horizons = list(bundle["horizons"])
    max_future = max(horizons)
    return Bundle(
        name=dataset,
        n_channels=int(bundle["n_channels"]),
        horizons=horizons,
        norm_mode=norm_mode,
        context_len=context_len,
        train=_windows_from_entities(bundle["ft_train"], context_len, max_future, STRIDE_TRAIN),
        val=_windows_from_entities(bundle["ft_val"], context_len, max_future, STRIDE_TRAIN),
        test=_windows_from_entities(bundle["ft_test"], context_len, max_future, STRIDE_TEST),
        entity_lengths={
            split: [len(e["test"]) for e in bundle[f"ft_{split}"]]
            for split in ("train", "val", "test")
        },
    )


# ---------------------------------------------------------------------------
# Labels and metric -- HEPA's own functions, no reimplementation
# ---------------------------------------------------------------------------


def label_surface(tte: np.ndarray, horizons: List[int]) -> np.ndarray:
    """(N, K) binary surface via hepa.evaluation.surface.build_label_surface."""
    t = torch.tensor(np.asarray(tte, dtype=np.float32)).unsqueeze(0)      # (1, N)
    h = torch.tensor(np.asarray(horizons, dtype=np.float32))              # (K,)
    return build_label_surface(t, h).squeeze(0).numpy()                   # (N, K)


def score_h_auroc(
    p_surface: np.ndarray, y_surface: np.ndarray, horizons: List[int]
) -> float:
    """hepa.evaluation.metrics.h_auroc, imported, not reimplemented.

    Degenerate-horizon convention: whatever the SHIPPED CODE does, which is to
    drop a horizon only when it is EXACTLY degenerate (n_pos == 0 or
    n_pos == n_total). The paper and PROTOCOL.md instead threshold prevalence at
    0.001/0.999 (scout D4). We use the code's, per task §4.
    """
    return float(h_auroc(p_surface, y_surface, horizons))


def per_horizon(
    p_surface: np.ndarray, y_surface: np.ndarray, horizons: List[int]
) -> Dict[str, list]:
    return per_horizon_auroc(p_surface, y_surface, horizons)


def n_nondegenerate_horizons(y_surface: np.ndarray) -> int:
    n = y_surface.shape[0]
    pos = y_surface.sum(axis=0)
    return int(((pos > 0) & (pos < n)).sum())
