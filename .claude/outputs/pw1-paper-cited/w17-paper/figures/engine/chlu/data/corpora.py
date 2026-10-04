"""⭐ The CORPUS REGISTRY — a real stream is a *config value*, not a code path.

**What this module is for.** Track A (charter §3) runs the same block over several
real byte streams: enwik8 (primary), WikiText-103 (secondary), and — held as a
*seam*, deliberately unbuilt — PG-19 and FineWeb-Edu. Before this module the
trainer imported :func:`chlu.data.enwik8.load_enwik8` by name, so a second stream
was a code edit in the experiment body. Now it is one line of config.

⭐ **Corpus-GENERIC, not a two-way switch — this is load-bearing.** Adding a third
stream must cost **one loader module plus one :func:`register_corpus` call**, with
**no** change to the trainer, the iterators, the eval slices or the byte ledger.
The named consumers of that promise, both decided elsewhere and neither built here:

* **PG-19** — ⛔ DECLARED OUT OF SCOPE for this module (`c3-benchmark-scout` §1.6
  returned GO as an *internal long-horizon retention instrument*, **NO-GO as an
  external comparison venue**: the nearest published numbers are >=5x our params
  on a different tokenizer). Three engineering caveats are banked in
  :data:`SEAM_NOTES` so the follow-up cell is cheap when it is funded.
* **FineWeb-Edu** — a *priced option* for a paper-stage appendix arm (Advisor
  decision 5, 2026-08-13). ⛔ No loader is built here either.

**The surface every corpus returns is identical**: a 3-tuple of
:class:`~chlu.data.enwik8.Enwik8Split` (aliased :data:`CorpusSplit`), consumed by
the existing ``contiguous_batches`` / ``random_batches`` iterators. A corpus that
cannot present that surface does not belong in the registry.

**Staging discipline is ENFORCED here, not documented.** ``csf3-runbook`` §A2's
download-once pattern says: stage **serially, once, before** a sweep, then let the
array tasks hit the cache. :func:`load_corpus` therefore **refuses to download
from inside a Slurm array task** (:func:`in_array_task`) no matter what the caller
passed — an array of 15 seed-jobs racing one download is the failure that rule
exists to prevent, and a comment in a runbook does not stop it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from chlu.data.enwik8 import Enwik8Split

#: The corpus-agnostic name for the split record. Every loader returns three of
#: these; the trainer and the iterators know nothing else about a corpus.
CorpusSplit = Enwik8Split

#: Three-tuple ``(train, valid, test)``.
CorpusSplits = Tuple[CorpusSplit, CorpusSplit, CorpusSplit]


@dataclass(frozen=True)
class CorpusSpec:
    """Everything the harness needs to know about one real stream.

    Attributes:
        name: the config value, e.g. ``"enwik8"``.
        load: ``(root, download, n_bytes, **kw) -> (train, valid, test)``.
        stage: ``(root, download) -> Path`` — the serial download-once entry point.
        vocab_size: alphabet size of the returned arrays (256 for every byte stream).
        level: ``"byte"`` or ``"word"``; the metric name follows from it.
        metric: ``"bpc"`` for byte streams, ``"ppl"`` for word-level.
        doc_boundary: byte pattern that starts a new document, or ``None`` when
            the corpus has no document structure. ⚠ Load-bearing for
            :mod:`chlu.eval.text_slices`: "within-document" retention is
            meaningless without it, so a corpus registered with ``None`` is
            refused by the slice instrument rather than silently scored across
            document joins.
        citation: the dataset's paper/source, quoted in the artifact.
    """

    name: str
    load: Callable[..., CorpusSplits]
    stage: Callable[..., Path]
    vocab_size: int = 256
    level: str = "byte"
    metric: str = "bpc"
    doc_boundary: Optional[bytes] = None
    citation: str = ""
    notes: Dict[str, Any] = field(default_factory=dict)


_REGISTRY: Dict[str, CorpusSpec] = {}


def register_corpus(spec: CorpusSpec) -> CorpusSpec:
    """Add a stream to the registry. ⛔ Re-registering a name is an error."""
    if spec.name in _REGISTRY:
        raise ValueError(f"corpus {spec.name!r} is already registered")
    _REGISTRY[spec.name] = spec
    return spec


def available_corpora() -> Tuple[str, ...]:
    """Registered stream names, sorted — the legal values of the config field."""
    return tuple(sorted(_REGISTRY))


def get_corpus(name: str) -> CorpusSpec:
    """Look a stream up by its config value, with a listing on a miss."""
    try:
        return _REGISTRY[str(name)]
    except KeyError:
        raise KeyError(
            f"unknown corpus {name!r}; registered: {list(available_corpora())}. "
            f"A new stream is ONE loader module + one register_corpus() call — "
            f"see chlu/data/corpora.py's module docstring."
        ) from None


# --------------------------------------------------------------------------
# staging discipline — enforced, not documented
# --------------------------------------------------------------------------
def in_array_task() -> bool:
    """True inside a Slurm **array** task, where a download must never happen.

    N array tasks share one cache. :func:`chlu.data.industrial.base.download_file`
    is unique-temp + atomic-rename + check-final-first, so a race is *safe*, but it
    is still N redundant multi-GB fetches over the campus egress and it is the
    documented operator error (`csf3-runbook` §A2). Fail fast instead.
    """
    return bool(os.environ.get("SLURM_ARRAY_TASK_ID"))


class NotStagedError(FileNotFoundError):
    """A corpus was needed inside a job but the serial staging step never ran."""


def stage_corpus(name: str, root: Optional[str | Path] = None,
                 *, download: bool = True) -> Path:
    """⭐ The **serial, once, before-the-sweep** staging entry point.

    Call this from a dedicated single-task job (``STAGE_ONLY=1``), never from an
    array task; :func:`load_corpus` then hits the cache.
    """
    spec = get_corpus(name)
    if download and in_array_task():
        raise NotStagedError(_stage_first_message(name))
    return spec.stage(root, download=download)


def _stage_first_message(name: str) -> str:
    return (
        f"⛔ corpus {name!r} is not staged, and this process is a Slurm ARRAY TASK "
        f"(SLURM_ARRAY_TASK_ID={os.environ.get('SLURM_ARRAY_TASK_ID')!r}), where "
        f"downloading is FORBIDDEN: N array tasks would each fetch the same "
        f"multi-GB archive.\n"
        f"    Stage it ONCE, SERIALLY, before submitting the array:\n"
        f"        sbatch --export=ALL,CORPUS={name},STAGE_ONLY=1 -p serial -t 0:30:00 \\\n"
        f"               scripts/csf3/job_gpu_c3_seeds.sh\n"
        f"    then resubmit the array (it will hit the cache)."
    )


def load_corpus(name: str, root: Optional[str | Path] = None, *,
                download: bool = True, n_bytes: Optional[int] = None,
                **kw: Any) -> CorpusSplits:
    """Load ``name``'s canonical splits — the ONE call the trainer makes.

    ``download`` is honoured **except** inside a Slurm array task, where it is
    forced off and a miss raises :class:`NotStagedError` naming the staging
    command. That override is deliberate and cannot be configured away: it is the
    difference between "the runbook says stage first" and "the job cannot not".
    """
    spec = get_corpus(name)
    if download and in_array_task():
        download = False
        try:
            spec.stage(root, download=False)
        except FileNotFoundError as e:
            raise NotStagedError(_stage_first_message(name)) from e
    return spec.load(root, download=download, n_bytes=n_bytes, **kw)


# --------------------------------------------------------------------------
# the two built streams
# --------------------------------------------------------------------------
def _load_enwik8(root, *, download=True, n_bytes=None, **kw) -> CorpusSplits:
    from chlu.data.enwik8 import load_enwik8

    if kw:
        raise TypeError(f"enwik8 takes no extra options, got {sorted(kw)}")
    return load_enwik8(root, download=download, n_bytes=n_bytes)


def _stage_enwik8(root, *, download=True) -> Path:
    from chlu.data.enwik8 import stage_enwik8

    return stage_enwik8(root, download=download)


def _load_wikitext103(root, *, download=True, n_bytes=None, level="byte",
                      **kw) -> CorpusSplits:
    from chlu.data.wikitext import load_wikitext103

    if kw:
        raise TypeError(f"wikitext103 takes no extra options, got {sorted(kw)}")
    return load_wikitext103(root, download=download, level=level, n_bytes=n_bytes)


def _stage_wikitext103(root, *, download=True) -> Path:
    from chlu.data.wikitext import stage_wikitext103

    return stage_wikitext103(root, download=download)


register_corpus(CorpusSpec(
    name="enwik8",
    load=_load_enwik8,
    stage=_stage_enwik8,
    vocab_size=256,
    level="byte",
    metric="bpc",
    # ⭐ enwik8 is a raw MediaWiki XML dump: every article is one <page> element.
    # Measured on the first 2 MB: 268 <page> opens => ~7.5 kB per document, which
    # is several times the pilot's 1024-byte context — so a within-document
    # long-range slice is physically present rather than nominal.
    doc_boundary=b"<page>",
    citation="Hutter Prize / first 100 MB of an English Wikipedia XML dump.",
))

register_corpus(CorpusSpec(
    name="wikitext103",
    load=_load_wikitext103,
    stage=_stage_wikitext103,
    vocab_size=256,
    level="byte",
    metric="bpc",
    # WT-103-raw marks each article with a level-1 heading line ` = Title = `.
    # Level-2+ headings are ` = = Section = = `, so the boundary pattern must
    # anchor on the newline AND reject the double-equals continuation; see
    # chlu.eval.text_slices._document_starts, which does exactly that.
    doc_boundary=b"\n = ",
    citation="Merity, Xiong, Bradbury, Socher (2016), WikiText-103 (raw).",
))


# --------------------------------------------------------------------------
# ⛔ the seam — declared, costed, and deliberately NOT built
# --------------------------------------------------------------------------
#: What adding a stream actually touches (the audit of "the option is cheap").
#: ⭐ Verified by construction: nothing in this list is a trainer, iterator,
#: slice or ledger edit.
SEAM_COST = (
    "1. a new module chlu/data/<corpus>.py exposing stage_*/load_* with the "
    "Enwik8Split surface (the shape chlu/data/wikitext.py was written to serve);",
    "2. one register_corpus(CorpusSpec(...)) call in chlu/data/corpora.py, "
    "carrying vocab_size, level, metric, doc_boundary and citation;",
    "3. one config value: PilotConfig.corpus='<corpus>' (no code path).",
)

#: ⚠ `c3-benchmark-scout` §1.6's three engineering caveats, banked verbatim so the
#: PG-19 follow-up cell does not have to rediscover them. ⛔ Not a build order.
SEAM_NOTES: Dict[str, Tuple[str, ...]] = {
    "pg19": (
        "(i) 11 GB across ~28,752 individual files is an inode/many-small-files "
        "hazard on CSF3 — consolidate ONCE, SERIALLY, into a single memmap-able "
        "uint8 stream with a sha256 contract, exactly as chlu/data/enwik8.py does.",
        "(ii) PG-19's metric is WORD-LEVEL perplexity normalised by the raw word "
        "count computed from the TEXT, not from the tokenizer — which is why a "
        "byte-level model can be scored in the venue's own currency with no "
        "tokenizer confound. A PG-19 CorpusSpec therefore needs metric='ppl' and "
        "a word-count normaliser taken off the raw bytes.",
        "(iii) the validation set is 50 books / 3.0 M words — small enough that "
        "per-book variance is material. Report per-book spread, never the bare "
        "mean. ⛔ NO-GO as an external comparison venue (nearest published "
        "numbers are >=5x our params on a different tokenizer).",
    ),
    "fineweb_edu": (
        "Held as a PRICED OPTION for a paper-stage appendix arm (Advisor "
        "decision 5, 2026-08-13) to pre-empt the SSM reviewer's 'why only "
        "enwik8?'. Decided at the paper stage, not now. Cost = SEAM_COST.",
    ),
}
