"""enwik8 — the byte-level real stream for the tier-iii pilot.

**Why this dataset and not a designed gym.** FB4 proved the designed gym families
measure our own constructions back at us (three of four saturated at <=4 B) and
`track2-admissibility` proved zero synthetic Track-2 candidates survive the
substitute audit. The tier-iii venue is therefore a *real* stream. enwik8 is the
standard byte-level long-range language-modelling benchmark (first 100 MB of an
English Wikipedia XML dump, Hutter Prize), scored in **bits per character**, and
it is metric-native in the sense intervention section 6 criterion 4 requires: the
metric is compression of a real distribution, not a constructed recall probe.

**The canonical split is 90/5/5 by BYTE POSITION, not by shuffling** — the first
90 MB train, the next 5 MB validation, the last 5 MB test. Shuffling would
destroy exactly the long-range structure a memory is supposed to exploit, so the
split is positional and deterministic and this module never permutes it.

**Staging is deterministic and download-once** (`csf3-download-race-and-sbatch`):
the archive is fetched through :func:`chlu.data.industrial.base.download_file`,
which is unique-temp + atomic-rename + check-final-first and therefore safe when
N array tasks share one cache on the cluster. Call :func:`stage_enwik8` **once,
serially** before launching a sweep; the jobs then hit the cached file.

Byte-level means the vocabulary is 256 and there is no tokenizer to match across
arms -- one fewer confound in the system-level swap.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np

from chlu.data.industrial.base import default_data_root, download_file

#: Upstream (Hutter Prize mirror). The DeepAI mirror is a byte-identical fallback.
ENWIK8_URL = "http://mattmahoney.net/dc/enwik8.zip"
ENWIK8_URL_FALLBACK = "https://data.deepai.org/enwik8.zip"

#: sha256 of `enwik8.zip` (36 445 475 B), measured from the mattmahoney.net
#: mirror on 2026-08-01. Checked only when ``stage_enwik8(verify=True)``: the
#: Hutter-Prize mirrors re-pack the archive from time to time, so the *payload*
#: length check (100 000 000 B) is the load-bearing guard, not this digest.
ENWIK8_SHA256 = "547994d9980ebed1288380d652999f38a14fe291a6247c157c3d33d4932534bc"

#: The canonical positional split, in bytes.
N_TRAIN = 90_000_000
N_VALID = 5_000_000
N_TEST = 5_000_000
N_TOTAL = N_TRAIN + N_VALID + N_TEST

#: Byte-level vocabulary. enwik8 uses ~205 distinct byte values; we keep the full
#: 256 so the embedding is identical no matter how much of the stream is staged.
VOCAB_SIZE = 256


@dataclass(frozen=True)
class Enwik8Split:
    """One positional split of the byte stream.

    Attributes:
        name: ``"train" | "valid" | "test"``.
        data: ``(n,)`` uint8 byte array, in stream order.
    """

    name: str
    data: np.ndarray

    def __len__(self) -> int:
        return int(self.data.shape[0])

    @property
    def n_bytes(self) -> int:
        return int(self.data.nbytes)


def enwik8_root(root: Optional[str | Path] = None) -> Path:
    """Cache directory for the raw archive and the extracted byte file."""
    if root is not None:
        return Path(root).expanduser()
    return default_data_root() / "enwik8"


def stage_enwik8(
    root: Optional[str | Path] = None,
    *,
    download: bool = True,
    verify: bool = False,
) -> Path:
    """Fetch + extract enwik8 **once, deterministically**; return the byte file.

    ``verify`` is off by default because the Hutter-Prize mirrors re-pack the zip
    (the *contents* are stable, the archive bytes are not). The extracted payload
    is checked instead: it must be exactly 100 000 000 bytes.

    ⚠ Run this serially before a cluster sweep (`csf3-download-race-and-sbatch`
    section A2's download-once pattern), then let the jobs read the cache.
    """
    d = enwik8_root(root)
    d.mkdir(parents=True, exist_ok=True)
    raw = d / "enwik8"
    if raw.exists() and raw.stat().st_size == N_TOTAL:
        return raw
    if not download:
        raise FileNotFoundError(
            f"enwik8 not staged at {raw}. Run stage_enwik8(download=True) once, "
            f"or fetch {ENWIK8_URL} manually and unzip it there."
        )
    zip_path = d / "enwik8.zip"
    try:
        download_file(ENWIK8_URL, zip_path, sha256=ENWIK8_SHA256 if verify else None)
    except Exception:  # pragma: no cover - network fallback
        download_file(ENWIK8_URL_FALLBACK, zip_path,
                      sha256=ENWIK8_SHA256 if verify else None)
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open("enwik8") as src, open(raw, "wb") as dst:
            while True:
                block = src.read(1 << 22)
                if not block:
                    break
                dst.write(block)
    if raw.stat().st_size != N_TOTAL:
        raise IOError(
            f"staged enwik8 is {raw.stat().st_size} B, expected {N_TOTAL} B"
        )
    return raw


def load_enwik8(
    root: Optional[str | Path] = None,
    *,
    download: bool = True,
    n_bytes: Optional[int] = None,
) -> Tuple[Enwik8Split, Enwik8Split, Enwik8Split]:
    """Load the canonical 90/5/5 **positional** split as uint8 arrays.

    Args:
        n_bytes: if given, take only the first ``n_bytes`` of the stream and split
            it 90/5/5 in the same proportions. This is the ``--quick`` / toy-scale
            path; it is a *prefix*, never a sample, so it is deterministic and the
            long-range structure inside the prefix is intact.
    """
    raw = stage_enwik8(root, download=download)
    total = N_TOTAL if n_bytes is None else int(min(n_bytes, N_TOTAL))
    data = np.fromfile(raw, dtype=np.uint8, count=total)
    if data.shape[0] != total:  # pragma: no cover - truncated cache
        raise IOError(f"read {data.shape[0]} B from {raw}, expected {total}")
    n_tr = int(round(total * N_TRAIN / N_TOTAL))
    n_va = int(round(total * N_VALID / N_TOTAL))
    return (
        Enwik8Split("train", data[:n_tr]),
        Enwik8Split("valid", data[n_tr: n_tr + n_va]),
        Enwik8Split("test", data[n_tr + n_va:]),
    )


def contiguous_batches(
    split: Enwik8Split,
    *,
    batch: int,
    seq_len: int,
    n_batches: Optional[int] = None,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """**Deterministic, order-preserving** batches — the evaluation iterator.

    The stream is cut into ``batch`` equal contiguous lanes and read left to
    right, so lane ``b`` at step ``t+1`` continues lane ``b`` at step ``t``. This
    is the only iterator an arm with a *persistent* memory may be evaluated with:
    a shuffled iterator would hand the CLU a store written from unrelated text.

    Yields ``(inputs, targets)``, both ``(batch, seq_len)`` uint8, with targets
    shifted by one byte (next-byte prediction).
    """
    n = len(split)
    lane = n // batch
    steps = (lane - 1) // seq_len
    if n_batches is not None:
        steps = min(steps, int(n_batches))
    if steps <= 0:
        raise ValueError(
            f"split '{split.name}' ({n} B) too short for batch={batch}, seq_len={seq_len}"
        )
    base = np.arange(batch, dtype=np.int64) * lane
    for t in range(steps):
        off = base + t * seq_len
        idx = off[:, None] + np.arange(seq_len + 1, dtype=np.int64)[None, :]
        window = split.data[idx]
        yield window[:, :-1], window[:, 1:]


def random_batches(
    split: Enwik8Split,
    *,
    batch: int,
    seq_len: int,
    n_batches: int,
    seed: int = 0,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Randomly-offset training batches from a **seeded** numpy Generator.

    Seeded explicitly (never global ``np.random``) so the data order is
    reproducible and, critically, **identical across arms at the same seed** —
    the system-level swap requires the same data order (task D2).
    """
    rng = np.random.default_rng(int(seed))
    hi = len(split) - seq_len - 1
    if hi <= 0:
        raise ValueError(f"split '{split.name}' too short for seq_len={seq_len}")
    for _ in range(int(n_batches)):
        off = rng.integers(0, hi, size=int(batch), dtype=np.int64)
        idx = off[:, None] + np.arange(seq_len + 1, dtype=np.int64)[None, :]
        window = split.data[idx]
        yield window[:, :-1], window[:, 1:]


def bits_per_character(mean_nats: float) -> float:
    """Convert a mean next-byte NLL in nats to bits per character.

    enwik8 is scored in bpc; every number in the pilot report uses this exact
    conversion so arms and published baselines are comparable.
    """
    return float(mean_nats) / float(np.log(2.0))
