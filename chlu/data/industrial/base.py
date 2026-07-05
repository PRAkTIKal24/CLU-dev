"""Industrial-dataset loader protocol + download helpers (F3).

Every dataset exposes the same small surface so the evaluation harness
(``chlu.eval.harness``) is dataset-agnostic and new sources (e.g. FactoryWave)
can drop in later:

- ``unit_ids() / train_ids() / test_ids()``: physical **units** (a run, a
  bearing, a simulation seed) — the only legal split granularity.
- ``load_unit(uid) -> UnitRecord``: lazily load one unit's multichannel series
  with labels + metadata.
- ``label_kind``: "point" (per-timestep 0/1 labels — VUS-PR primary) or
  "episode" (one label per unit — episode AUROC primary).
- ``protocol``: "cross_unit" (fit on train units, score test units) or
  "per_unit_prefix" (TSB-AD style: each series carries its own normal training
  prefix; fit and score within the unit).

Downloads are fetch-or-point-at-path: constructors take ``root`` (defaulting
to ``$CHLU_DATA_ROOT`` or ``~/.cache/chlu/datasets/<name>``) and
``download=True`` invokes ``fetch()`` where automatic download is legal and
feasible; otherwise ``fetch()`` raises with manual instructions. Checksums are
verified whenever the upstream publishes or we have pinned them.
"""

import abc
import hashlib
import os
import shutil
import urllib.request
import zipfile
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def default_data_root() -> Path:
    """Dataset cache root: ``$CHLU_DATA_ROOT`` or ``~/.cache/chlu/datasets``."""
    env = os.environ.get("CHLU_DATA_ROOT")
    return Path(env).expanduser() if env else Path.home() / ".cache/chlu/datasets"


@dataclass(frozen=True)
class UnitRecord:
    """One physical unit's data (a run / bearing / simulation / episode).

    Attributes:
        unit_id: Unique unit identifier (the split key).
        data: (T, C) float32 multichannel series.
        channels: C channel names.
        point_labels: (T,) int8 0/1 per-timestep labels, or None for
            episode-labelled or unlabeled-normal units.
        episode_label: 0/1 unit-level label (episode datasets), else None.
        fault_class: Fault/anomaly class name if the dataset provides one.
        condition: Operating condition (cross-condition splits), if any.
        sampling_rate_hz: Sampling rate, if known.
        meta: Free-form extras (e.g. changepoint labels, train prefix length).
    """

    unit_id: str
    data: np.ndarray
    channels: tuple
    point_labels: np.ndarray | None = None
    episode_label: int | None = None
    fault_class: str | None = None
    condition: str | None = None
    sampling_rate_hz: float | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.data.ndim != 2:
            raise ValueError(f"data must be (T, C), got {self.data.shape}")
        if len(self.channels) != self.data.shape[1]:
            raise ValueError(
                f"{len(self.channels)} channel names for {self.data.shape[1]} columns"
            )
        if self.point_labels is not None and len(self.point_labels) != len(self.data):
            raise ValueError("point_labels length must match data length")


class IndustrialDataset(abc.ABC):
    """Abstract base for industrial datasets (see module docstring)."""

    name: str = "abstract"
    label_kind: str = "point"  # "point" | "episode"
    protocol: str = "cross_unit"  # "cross_unit" | "per_unit_prefix"
    license_note: str = ""
    citation: str = ""

    def __init__(self, root: str | Path | None = None, download: bool = False):
        self.root = Path(root).expanduser() if root else default_data_root() / self.name
        if download and not self.is_available():
            self.fetch()
        if not self.is_available():
            raise FileNotFoundError(
                f"{self.name}: no data found under {self.root}. "
                f"Pass download=True or fetch manually. {self.license_note}"
            )

    # -- required surface ---------------------------------------------------
    @abc.abstractmethod
    def unit_ids(self) -> tuple:
        """All unit ids (sorted, stable)."""

    @abc.abstractmethod
    def load_unit(self, unit_id: str) -> UnitRecord:
        """Load one unit lazily."""

    @abc.abstractmethod
    def is_available(self) -> bool:
        """True if the expected files exist under ``self.root``."""

    # -- canonical split (unit-level by construction) ------------------------
    def train_ids(self) -> tuple:
        """Canonical training units (normal-only where the dataset defines it)."""
        raise NotImplementedError(f"{self.name} defines no canonical train split")

    def test_ids(self) -> tuple:
        """Canonical test units."""
        raise NotImplementedError(f"{self.name} defines no canonical test split")

    def fetch(self) -> None:
        """Download into ``self.root`` (only where legal + feasible)."""
        raise NotImplementedError(
            f"{self.name}: automatic download not implemented. {self.license_note}"
        )

    # -- conveniences ---------------------------------------------------------
    def iter_units(self, unit_ids: Iterable[str] | None = None) -> Iterator[UnitRecord]:
        for uid in unit_ids if unit_ids is not None else self.unit_ids():
            yield self.load_unit(uid)

    def conditions(self) -> dict:
        """Mapping unit_id -> condition (for ``cross_condition_split``)."""
        return {rec.unit_id: rec.condition for rec in self.iter_units()}


# ---------------------------------------------------------------------------
# download helpers
# ---------------------------------------------------------------------------


def file_digest(path: Path, algo: str = "sha256", chunk: int = 1 << 20) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def download_file(
    url: str,
    dest: str | Path,
    sha256: str | None = None,
    md5: str | None = None,
    timeout: int = 60,
) -> Path:
    """Stream ``url`` to ``dest`` (via .part + rename) and verify checksums.

    Skips the download when ``dest`` already exists and passes verification.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    def _verify(p: Path) -> None:
        if sha256 and file_digest(p, "sha256") != sha256:
            raise IOError(f"sha256 mismatch for {p}")
        if md5 and file_digest(p, "md5") != md5:
            raise IOError(f"md5 mismatch for {p}")

    if dest.exists():
        _verify(dest)
        return dest

    part = dest.with_suffix(dest.suffix + ".part")
    print(f"downloading {url} -> {dest}")
    with urllib.request.urlopen(url, timeout=timeout) as resp, open(part, "wb") as out:
        shutil.copyfileobj(resp, out, length=1 << 20)
    _verify(part)
    part.rename(dest)
    return dest


def extract_zip(
    zip_path: str | Path,
    dest: str | Path,
    member_filter: str | None = None,
) -> Path:
    """Extract ``zip_path`` under ``dest``; optionally only members containing
    ``member_filter`` in their path (avoids exploding huge archives)."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        members = [
            m for m in zf.namelist() if member_filter is None or member_filter in m
        ]
        if not members:
            raise FileNotFoundError(
                f"no members matching {member_filter!r} in {zip_path}"
            )
        zf.extractall(dest, members=members)
    return dest


def require_pandas(feature: str):
    """Import pandas with an actionable error (loaders need the eval extra)."""
    try:
        import pandas  # noqa: PLC0415

        return pandas
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            f"{feature} requires pandas. Install the eval extra: uv sync --extra eval"
        ) from exc
