"""WikiText-103 — the second real stream named by the tier-iii pilot.

⚠ **Declared status: BUILT, NOT THE PILOT'S VENUE.** The C2W4 pilot runs on
**enwik8** (:mod:`chlu.data.enwik8`) because byte-level scoring removes the
tokenizer as a confound from the system-level swap: every arm sees the identical
256-symbol alphabet and there is no vocabulary to match. This module exists so
the WT-103 leg named in §A14.3 is a configuration change rather than a build,
and so the loader is reviewed alongside the rest of the pilot.

Exposes the **same surface** as :mod:`chlu.data.enwik8` — ``load_wikitext103``
returns three ``Enwik8Split``-shaped records and the same
``contiguous_batches`` / ``random_batches`` iterators apply — so an arm can be
pointed at either stream without touching the trainer.

Two modes:

* ``level="byte"`` (default) — UTF-8 bytes, vocabulary 256, directly comparable
  with the enwik8 numbers and with the same iterators;
* ``level="word"`` — the published word-level protocol, with the vocabulary
  built **from the training split only** (a vocabulary built on the full corpus
  is a test-set leak) and everything outside it mapped to ``<unk>``.

Staging goes through the same concurrency-safe
:func:`~chlu.data.industrial.base.download_file` (unique-temp + atomic-rename +
check-final-first), so a cluster array sharing one cache is safe; still call
:func:`stage_wikitext103` **once, serially**, before launching a sweep.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from chlu.data.enwik8 import Enwik8Split
from chlu.data.industrial.base import default_data_root, download_file

#: The canonical raw-character distribution (Merity et al., 2016).
WIKITEXT103_URL = (
    "https://s3.amazonaws.com/research.metamind.io/wikitext/"
    "wikitext-103-raw-v1.zip"
)
WIKITEXT103_FALLBACK = (
    "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/"
    "wikitext-103-raw-v1.zip"
)

#: File names inside the archive, in split order.
SPLIT_FILES = {
    "train": "wikitext-103-raw/wiki.train.raw",
    "valid": "wikitext-103-raw/wiki.valid.raw",
    "test": "wikitext-103-raw/wiki.test.raw",
}

BYTE_VOCAB_SIZE = 256


def wikitext_root(root: Optional[str | Path] = None) -> Path:
    if root is not None:
        return Path(root).expanduser()
    return default_data_root() / "wikitext103"


def stage_wikitext103(root: Optional[str | Path] = None, *, download: bool = True
                      ) -> Path:
    """Fetch + extract WT-103-raw once; return the directory holding the splits."""
    d = wikitext_root(root)
    d.mkdir(parents=True, exist_ok=True)
    if all((d / Path(v).name).exists() for v in SPLIT_FILES.values()):
        return d
    if not download:
        raise FileNotFoundError(
            f"WikiText-103 not staged at {d}. Run stage_wikitext103(download=True) "
            f"once, or fetch {WIKITEXT103_URL} manually and unzip the three "
            f"wiki.*.raw files there."
        )
    zip_path = d / "wikitext-103-raw-v1.zip"
    try:
        download_file(WIKITEXT103_URL, zip_path)
    except Exception:  # pragma: no cover - network fallback
        download_file(WIKITEXT103_FALLBACK, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        for member in SPLIT_FILES.values():
            with zf.open(member) as src, open(d / Path(member).name, "wb") as dst:
                while True:
                    block = src.read(1 << 22)
                    if not block:
                        break
                    dst.write(block)
    return d


def _read(d: Path, split: str, n_bytes: Optional[int]) -> bytes:
    p = d / Path(SPLIT_FILES[split]).name
    raw = p.read_bytes()
    return raw if n_bytes is None else raw[: int(n_bytes)]


def load_wikitext103(
    root: Optional[str | Path] = None,
    *,
    download: bool = True,
    level: str = "byte",
    n_bytes: Optional[int] = None,
) -> Tuple[Enwik8Split, Enwik8Split, Enwik8Split]:
    """Load WT-103 as three splits with the enwik8 surface.

    ``n_bytes`` truncates **each** split to its first ``n_bytes`` (a prefix,
    never a sample), so the toy path stays deterministic and order-preserving.
    """
    if level not in ("byte", "word"):
        raise ValueError(f"level must be 'byte' or 'word', got {level!r}")
    d = stage_wikitext103(root, download=download)
    if level == "byte":
        return tuple(  # type: ignore[return-value]
            Enwik8Split(s, np.frombuffer(_read(d, s, n_bytes), dtype=np.uint8).copy())
            for s in ("train", "valid", "test")
        )
    vocab = build_word_vocab(_read(d, "train", n_bytes).decode("utf8", "replace"))
    return tuple(  # type: ignore[return-value]
        Enwik8Split(s, encode_words(_read(d, s, n_bytes).decode("utf8", "replace"),
                                    vocab))
        for s in ("train", "valid", "test")
    )


def build_word_vocab(train_text: str, *, min_count: int = 1) -> Dict[str, int]:
    """Vocabulary from the **training split only** — a corpus-wide vocab leaks.

    ``<unk>`` is index 0 and is always present (WT-103-raw contains no ``<unk>``
    of its own, which is the point of the raw distribution).
    """
    counts: Dict[str, int] = {}
    for w in train_text.split():
        counts[w] = counts.get(w, 0) + 1
    words: List[str] = ["<unk>"] + sorted(w for w, c in counts.items()
                                          if c >= int(min_count))
    return {w: i for i, w in enumerate(words)}


def encode_words(text: str, vocab: Dict[str, int]) -> np.ndarray:
    """Map whitespace-split tokens through ``vocab``; OOV -> ``<unk>`` (0)."""
    return np.fromiter((vocab.get(w, 0) for w in text.split()),
                       dtype=np.int32, count=-1)
