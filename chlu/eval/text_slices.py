"""⭐ WITHIN-DOCUMENT RETENTION / REVISIT SLICES on REAL text.

**What this measures.** Not a scalar "does the model use long context", but
**bpc conditioned on distance-since-the-last-occurrence of the thing being
predicted**. A memory that actually retains shows up as a bpc gap that *widens*
with distance; a model that merely has good local statistics shows a flat gap.
That is the shape of the tier-iii claim, so it must be the shape of the
instrument.

--------------------------------------------------------------------------
⭐ ADOPTED, NOT INVENTED — and the borrowing is separable from the extension
--------------------------------------------------------------------------
**ADOPTED — Sun, Krishna, Mattarella-Micke, Iyyer (2021), *Do Long-Range Language
Models Actually Use Long-Range Context?*, EMNLP 2021, pp. 807–822,
arXiv:2109.09115.** Theirs is the definition of the slice and the bucket
structure: target positions bucketed by **distance to the last occurrence in the
prefix**, plus a **"never appears in the prefix"** bucket. Defined on PG-19. ⛔ We
do not rename it and we cite it.

**Precedent for the control — Khandelwal, He, Qi, Jurafsky (2018), *Sharp Nearby,
Fuzzy Far Away: How Neural Language Models Use Context*, ACL 2018,
aclanthology.org/P18-1027.** The perturbation-by-distance protocol (shuffle /
replace / drop prior context as a function of distance; effective context ≈200
tokens, word order matters only within ≈50). Our shuffled-position control is
that protocol applied to the slice itself.

**DECLARED AS OURS** (the field has no convention for these), one line each:

1. **The bin edges** — :data:`DEFAULT_BIN_EDGES`, chosen to straddle the model's
   context and run well past it (see the constant's own note).
2. **Computing it on enwik8 / WT-103 BYTES** rather than PG-19 subwords.
3. **Computing it for EVERY arm, including the dyn-eval substitute column** — a
   retention slice only our arm receives is not a comparison.
4. **The shuffled-position control** (Khandelwal et al. is the precedent for the
   *protocol*; applying it as a validity check *on the slice* is ours).

--------------------------------------------------------------------------
⚠⚠ TRAP 1 — THE REVISIT UNIT. This is the decision that makes or breaks it.
--------------------------------------------------------------------------
At **vocab 256**, "distance to the last occurrence of the same symbol" is **a few
bytes** for common characters: ``e`` recurs every ~10 bytes of English. Bucketed
that way the instrument degenerates into a **character-frequency count** wearing a
retention label — `c3-benchmark-scout` §1.3 calls this *"the single most likely
silent failure in text_slices.py"*, and it is right.

⇒ **The revisit unit here is the enclosing whitespace-delimited TOKEN, on a byte
stream.** Every byte position inherits the revisit distance of the *word* it sits
inside. And the degeneracy is not merely avoided by construction, it is
**asserted**: :func:`assert_non_degenerate` recomputes the same index at the raw
byte unit and requires the token unit's median revisit distance to exceed it by a
declared factor. A slice that fails that check **raises** rather than shipping.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

#: ASCII whitespace: space, tab, newline, CR, VT, FF. The token separator set.
WHITESPACE = np.array([32, 9, 10, 13, 11, 12], dtype=np.uint8)

#: ⭐ **DECLARED AS OURS.** Distances are in **BYTES**, because the model's
#: context is denominated in bytes and the point of the slice is what happens at
#: and beyond that horizon. The edges straddle the pilot's 1024-byte context
#: (``512`` and ``2048`` sit either side of it) and the top bin, ``>=8192``, is
#: 8x the context — the regime where only a memory can help. The first two edges
#: are deliberately tight so that a **degenerate**, character-frequency-like slice
#: would pile into them visibly rather than hide inside a coarse first bucket.
DEFAULT_BIN_EDGES: Tuple[int, ...] = (1, 8, 32, 128, 512, 2048, 8192)

#: Sun et al.'s bucket for target positions whose unit has **no prior occurrence
#: in the document**. ⛔ Not merged into the largest distance bin: "never seen"
#: and "seen 10 k bytes ago" are different questions, and this is where a memory
#: either shows up or does not.
NEVER_BIN = "never"

#: :func:`assert_non_degenerate` requires the token-unit median revisit distance
#: to be at least this many times the raw-byte-unit median. Measured on enwik8 the
#: real ratio is ~2 orders of magnitude; 10x is a floor that a genuinely
#: degenerate unit cannot clear.
NON_DEGENERACY_MIN_RATIO = 10.0

#: ...and an absolute floor, so a corpus with pathologically short lines cannot
#: satisfy the ratio while still measuring character frequency.
NON_DEGENERACY_MIN_MEDIAN_BYTES = 64.0


def bin_labels(edges: Sequence[int] = DEFAULT_BIN_EDGES) -> List[str]:
    """Human-readable bin names, in order, ending with :data:`NEVER_BIN`."""
    e = list(int(x) for x in edges)
    out = [f"[{e[i]},{e[i + 1]})" for i in range(len(e) - 1)]
    out.append(f"[{e[-1]},inf)")
    out.append(NEVER_BIN)
    return out


# ==========================================================================
# document boundaries — "within-document" is load-bearing
# ==========================================================================
def _document_starts(data: np.ndarray, pattern: Optional[bytes]) -> np.ndarray:
    """Byte offsets at which a new document begins (always includes 0).

    ⛔ **Explicit and tested per corpus**, because a revisit distance measured
    across a document join is not a revisit — it is a coincidence between two
    unrelated articles, and it would inflate exactly the long-distance bins the
    claim is read off.

    * **enwik8** — the raw MediaWiki XML dump: each article is one ``<page>``
      element, so the pattern is ``b"<page>"``.
    * **WikiText-103 (raw)** — articles open with a level-1 heading line
      ``\\n = Title = \\n``, while *sections* are ``\\n = = Section = = \\n``. The
      naive pattern therefore matches every section too, so a match whose next
      non-space byte is another ``=`` is rejected here.
    """
    if not pattern:
        return np.array([0], dtype=np.int64)
    pat = np.frombuffer(pattern, dtype=np.uint8)
    n, m = int(data.shape[0]), int(pat.shape[0])
    if n < m:
        return np.array([0], dtype=np.int64)
    # vectorised exact-match scan
    hits = np.ones(n - m + 1, dtype=bool)
    for i in range(m):
        hits &= data[i: n - m + 1 + i] == pat[i]
    idx = np.flatnonzero(hits).astype(np.int64)
    if pattern.endswith(b" = ") or pattern == b"\n = ":
        # ⚠ WT-103: reject ` = = ` (a SECTION heading, not a new article).
        nxt = idx + m
        keep = (nxt < n) & (data[np.minimum(nxt, n - 1)] != ord("="))
        idx = idx[keep]
    if idx.size == 0 or idx[0] != 0:
        idx = np.concatenate([[0], idx])
    return idx


# ==========================================================================
# the revisit index
# ==========================================================================
@dataclass
class RevisitIndex:
    """Per-byte-position revisit structure for one split.

    Attributes:
        bin_of_position: ``(n,)`` int8 bin id for every byte position; the last id
            is :data:`NEVER_BIN`, and ``-1`` marks positions that carry no unit
            (whitespace) and are excluded from every slice.
        distance: ``(n,)`` int64 revisit distance in bytes (``-1`` where absent).
        labels: bin names in id order.
        unit: ``"token"`` (the shipped unit) or ``"byte"`` (the degeneracy control).
    """

    bin_of_position: np.ndarray
    distance: np.ndarray
    labels: List[str]
    unit: str
    n_documents: int
    edges: Tuple[int, ...] = DEFAULT_BIN_EDGES
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_bins(self) -> int:
        return len(self.labels)

    def counts(self) -> Dict[str, int]:
        c = np.bincount(self.bin_of_position[self.bin_of_position >= 0],
                        minlength=self.n_bins)
        return {lab: int(c[i]) for i, lab in enumerate(self.labels)}

    def median_distance(self) -> float:
        d = self.distance[self.distance >= 0]
        return float(np.median(d)) if d.size else float("nan")


def _token_spans(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """``(starts, ends)`` of whitespace-delimited tokens, in stream order."""
    ws = np.isin(data, WHITESPACE)
    nonws = ~ws
    if not nonws.any():
        z = np.zeros(0, dtype=np.int64)
        return z, z
    prev = np.empty_like(nonws)
    prev[0] = False
    prev[1:] = nonws[:-1]
    starts = np.flatnonzero(nonws & ~prev).astype(np.int64)
    nxt = np.empty_like(nonws)
    nxt[-1] = False
    nxt[:-1] = nonws[1:]
    ends = (np.flatnonzero(nonws & ~nxt) + 1).astype(np.int64)
    return starts, ends


def build_revisit_index(
    data: np.ndarray,
    *,
    doc_boundary: Optional[bytes],
    unit: str = "token",
    edges: Sequence[int] = DEFAULT_BIN_EDGES,
) -> RevisitIndex:
    """⭐ The instrument. Bucket every byte position by its unit's revisit distance.

    Args:
        data: ``(n,)`` uint8 byte stream (one split).
        doc_boundary: the corpus's document-start pattern
            (:class:`~chlu.data.corpora.CorpusSpec.doc_boundary`). ⛔ ``None`` is
            refused: "within-document" cannot be honoured without it.
        unit: ``"token"`` — the shipped unit, the enclosing whitespace-delimited
            word (TRAP 1); or ``"byte"`` — the raw symbol, kept ONLY as the
            degeneracy control that :func:`assert_non_degenerate` compares against.

    Returns:
        A :class:`RevisitIndex`. Whitespace positions get bin ``-1`` and are
        excluded everywhere: they carry no unit, and scoring them would dilute
        every bucket with the easiest bytes in the corpus.
    """
    if doc_boundary is None:
        raise ValueError(
            "⛔ this corpus declares no doc_boundary, so 'within-document' "
            "retention cannot be computed. Register a CorpusSpec.doc_boundary "
            "(see chlu/data/corpora.py) rather than silently scoring across "
            "document joins — a revisit measured across two unrelated articles "
            "is a coincidence, and it lands in exactly the long bins the claim "
            "is read off.")
    data = np.asarray(data, dtype=np.uint8)
    n = int(data.shape[0])
    edges = tuple(int(x) for x in edges)
    labels = bin_labels(edges)
    never_id = len(labels) - 1

    docs = _document_starts(data, doc_boundary)
    doc_of = np.searchsorted(docs, np.arange(n, dtype=np.int64), side="right") - 1

    dist = np.full(n, -1, dtype=np.int64)
    has_unit = np.zeros(n, dtype=bool)

    if unit == "token":
        starts, ends = _token_spans(data)
        # key = (document id, token bytes) -> previous token start
        last: Dict[Tuple[int, bytes], int] = {}
        raw = data.tobytes()
        tok_dist = np.full(starts.shape[0], -1, dtype=np.int64)
        d_of_tok = doc_of[starts] if starts.size else np.zeros(0, dtype=np.int64)
        for i in range(starts.shape[0]):
            s, e, d = int(starts[i]), int(ends[i]), int(d_of_tok[i])
            k = (d, raw[s:e])
            p = last.get(k)
            if p is not None:
                tok_dist[i] = s - p
            last[k] = s
        # broadcast the token's distance onto every byte inside it
        for i in range(starts.shape[0]):
            s, e = int(starts[i]), int(ends[i])
            dist[s:e] = tok_dist[i]
            has_unit[s:e] = True
    elif unit == "byte":
        # ⚠ the DEGENERATE unit, retained only as the control.
        last_seen = np.full((docs.shape[0], 256), -1, dtype=np.int64)
        for i in range(n):
            d = int(doc_of[i])
            v = int(data[i])
            p = last_seen[d, v]
            if p >= 0:
                dist[i] = i - p
            last_seen[d, v] = i
        has_unit[:] = True
    else:
        raise ValueError(f"unit must be 'token' or 'byte', got {unit!r}")

    bins = np.full(n, -1, dtype=np.int16)
    seen = has_unit & (dist >= 0)
    bins[has_unit & (dist < 0)] = never_id            # never in the prefix
    if seen.any():
        bins[seen] = np.searchsorted(np.asarray(edges), dist[seen],
                                     side="right") - 1
        bins[seen] = np.clip(bins[seen], 0, len(edges) - 1)
    return RevisitIndex(
        bin_of_position=bins, distance=dist, labels=labels, unit=unit,
        n_documents=int(docs.shape[0]), edges=edges,
        meta={"n_bytes": n, "n_units_with_prior": int(seen.sum()),
              "n_units_never": int((bins == never_id).sum()),
              "n_positions_no_unit": int((bins < 0).sum())},
    )


# ==========================================================================
# ⚠ control (0): the non-degeneracy assertion — TRAP 1's tripwire
# ==========================================================================
def assert_non_degenerate(data: np.ndarray, *, doc_boundary: Optional[bytes],
                          edges: Sequence[int] = DEFAULT_BIN_EDGES,
                          sample_bytes: int = 400_000) -> Dict[str, Any]:
    """⛔ Fail rather than ship a slice that is really a character-frequency count.

    Recomputes the index at the **raw byte unit** — the degenerate definition the
    scout warned about — and requires the shipped **token unit** to have a median
    revisit distance both (a) at least
    :data:`NON_DEGENERACY_MIN_RATIO`x larger and (b) above
    :data:`NON_DEGENERACY_MIN_MEDIAN_BYTES` in absolute terms.

    Returns the measured comparison (so it can go in the artifact); raises
    ``AssertionError`` on failure.
    """
    d = np.asarray(data, dtype=np.uint8)[: int(sample_bytes)]
    tok = build_revisit_index(d, doc_boundary=doc_boundary, unit="token",
                              edges=edges)
    byt = build_revisit_index(d, doc_boundary=doc_boundary, unit="byte",
                              edges=edges)
    mt, mb = tok.median_distance(), byt.median_distance()
    ratio = float(mt / mb) if mb else float("inf")
    rep = {
        "median_distance_token_unit": mt,
        "median_distance_byte_unit": mb,
        "ratio": ratio,
        "min_ratio_required": NON_DEGENERACY_MIN_RATIO,
        "min_median_required": NON_DEGENERACY_MIN_MEDIAN_BYTES,
        "token_counts": tok.counts(),
        "byte_counts": byt.counts(),
        "n_bytes_sampled": int(d.shape[0]),
        "passed": bool(ratio >= NON_DEGENERACY_MIN_RATIO
                       and mt >= NON_DEGENERACY_MIN_MEDIAN_BYTES),
    }
    if not rep["passed"]:
        raise AssertionError(
            "⛔ DEGENERATE REVISIT SLICE (TRAP 1). The token-unit median revisit "
            f"distance is {mt} B against the raw-byte unit's {mb} B (ratio "
            f"{ratio:.2f}, need >={NON_DEGENERACY_MIN_RATIO} and median "
            f">={NON_DEGENERACY_MIN_MEDIAN_BYTES}). A slice whose distances "
            "collapse to single digits for the commonest units measures "
            "CHARACTER FREQUENCY, not retention. Do not tune the thresholds to "
            "pass — fix the unit.")
    return rep


# ==========================================================================
# ⚠ control (a): shuffled positions — the slice MUST move
# ==========================================================================
def shuffled_position_control(data: np.ndarray, *, doc_boundary: Optional[bytes],
                              edges: Sequence[int] = DEFAULT_BIN_EDGES,
                              seed: int = 0,
                              sample_bytes: int = 400_000) -> Dict[str, Any]:
    """Permute token ORDER within each document, preserving the content multiset.

    Khandelwal et al. (2018)'s perturbation-by-distance protocol, applied to the
    slice itself. Every token that existed still exists and every token's *text*
    is untouched — only **where** each one sits changes — so the distance
    structure is destroyed while the corpus's unigram statistics are exactly
    preserved.

    ⇒ **The bin populations must move.** If they do not, the slice is keyed on
    content frequency rather than distance, and the instrument is not measuring
    retention. ⛔ Per the task's kill conditions, a failure here is reported as a
    finding; the definition is not tuned until it passes.
    """
    d = np.asarray(data, dtype=np.uint8)[: int(sample_bytes)]
    base = build_revisit_index(d, doc_boundary=doc_boundary, unit="token",
                               edges=edges)
    rng = np.random.default_rng(int(seed))
    starts, ends = _token_spans(d)
    docs = _document_starts(d, doc_boundary)
    doc_of_tok = (np.searchsorted(docs, starts, side="right") - 1
                  if starts.size else np.zeros(0, dtype=np.int64))
    raw = d.tobytes()
    toks = [raw[int(s):int(e)] for s, e in zip(starts, ends, strict=True)]
    order = np.arange(len(toks), dtype=np.int64)
    for doc in np.unique(doc_of_tok):
        m = np.flatnonzero(doc_of_tok == doc)
        perm = rng.permutation(m.shape[0])
        order[m] = m[perm]
    # rebuild a stream of the SAME length: document markers stay put, token slots
    # keep their byte extents where possible by joining with single spaces.
    parts: List[bytes] = []
    for doc in range(docs.shape[0]):
        m = np.flatnonzero(doc_of_tok == doc)
        if m.size == 0:
            continue
        parts.append(raw[int(docs[doc]): int(starts[m[0]])])
        parts.append(b" ".join(toks[int(i)] for i in order[m]))
    shuf = np.frombuffer(b"".join(parts), dtype=np.uint8)
    perm_idx = build_revisit_index(shuf, doc_boundary=doc_boundary, unit="token",
                                   edges=edges)
    a, b = base.counts(), perm_idx.counts()
    ta = max(1, sum(a.values()))
    tb = max(1, sum(b.values()))
    tvd = 0.5 * sum(abs(a.get(k, 0) / ta - b.get(k, 0) / tb) for k in base.labels)
    return {
        "counts_ordered": a, "counts_shuffled": b,
        "median_distance_ordered": base.median_distance(),
        "median_distance_shuffled": perm_idx.median_distance(),
        "total_variation_distance": float(tvd),
        "seed": int(seed), "n_bytes_sampled": int(d.shape[0]),
        # a real permutation moves several percent of the mass between bins;
        # anything at machine-noise level means the slice is content-keyed.
        "slice_moved": bool(tvd > 0.01),
    }


# ==========================================================================
# ⚠ control (b): content relabelling — the slice must NOT move
# ==========================================================================
def content_relabel_control(data: np.ndarray, *, doc_boundary: Optional[bytes],
                            edges: Sequence[int] = DEFAULT_BIN_EDGES,
                            sample_bytes: int = 200_000) -> Dict[str, Any]:
    """Relabel token IDENTITIES bijectively, preserving the distance structure.

    The converse of the shuffled-position control and the dial declaration's
    falsifier: *"an eval slice that changes value when only the content (not the
    distance structure) is permuted"* kills the instrument. Here every token is
    replaced by a fresh, same-length symbol string under a consistent bijection,
    so each unit's revisit pattern is bit-identical while the text is unrecognisable.

    ⇒ **The bin populations must be UNCHANGED.**
    """
    d = np.asarray(data, dtype=np.uint8)[: int(sample_bytes)]
    base = build_revisit_index(d, doc_boundary=doc_boundary, unit="token",
                               edges=edges)
    starts, ends = _token_spans(d)
    raw = d.tobytes()
    out = bytearray(raw)

    # ⚠ Two constraints make this control non-trivial, and getting either wrong
    # silently breaks the invariance it is supposed to prove:
    #   (1) the map must be INJECTIVE and LENGTH-PRESERVING, or two distinct
    #       tokens merge and the revisit distances genuinely change;
    #   (2) it must not disturb the DOCUMENT MARKERS — `<page>` (enwik8) and
    #       ` = ` (WT-103) are themselves tokens, and relabelling them would move
    #       every document boundary and thus every distance.
    # So: structural tokens are left verbatim, and the replacement alphabet
    # excludes the bytes those markers are built from, which makes it impossible
    # for a relabelled token to forge a boundary.
    _STRUCT = set(b"<=&")
    alphabet = bytes(b for b in range(33, 127) if b not in _STRUCT)
    radix = len(alphabet)
    mapping: Dict[bytes, bytes] = {}
    per_len: Dict[int, int] = {}
    n_struct = 0
    for s, e in zip(starts, ends, strict=True):
        t = raw[int(s):int(e)]
        if t not in mapping:
            if _STRUCT & set(t):
                mapping[t] = t                      # structural: leave verbatim
                n_struct += 1
            else:
                L = len(t)
                k = per_len.get(L, 0)
                per_len[L] = k + 1
                if k >= radix ** L:                 # would collide -> leave alone
                    mapping[t] = t
                else:
                    mapping[t] = bytes(alphabet[(k // radix ** j) % radix]
                                       for j in range(L))
        out[int(s):int(e)] = mapping[t]
    rel = np.frombuffer(bytes(out), dtype=np.uint8)
    rel_idx = build_revisit_index(rel, doc_boundary=doc_boundary, unit="token",
                                  edges=edges)
    a, b = base.counts(), rel_idx.counts()
    injective = len(set(mapping.values())) == len(mapping)
    return {"counts_original": a, "counts_relabelled": b,
            "n_distinct_tokens": len(mapping),
            "n_structural_tokens_kept": n_struct,
            "relabelling_injective": bool(injective),
            "n_documents_original": base.n_documents,
            "n_documents_relabelled": rel_idx.n_documents,
            "n_bytes_sampled": int(d.shape[0]),
            "slice_invariant": bool(a == b and injective)}


# ==========================================================================
# scoring: per-bin bpc with per-bin n
# ==========================================================================
LN2 = float(np.log(2.0))


def contiguous_target_positions(n: int, *, batch: int, seq_len: int,
                                n_batches: Optional[int] = None
                                ) -> List[np.ndarray]:
    """Stream positions of the TARGETS that :func:`contiguous_batches` yields.

    ⛔ Mirrors ``chlu.data.enwik8.contiguous_batches``' arithmetic exactly (lane
    ``b`` starts at ``b * (n // batch)``; step ``t`` reads ``seq_len + 1`` bytes
    and targets are the window shifted by one). A test asserts
    ``split.data[positions] == targets`` for the real iterator — the alignment is
    the one thing in this module that cannot be allowed to drift.
    """
    lane = n // int(batch)
    steps = (lane - 1) // int(seq_len)
    if n_batches is not None:
        steps = min(steps, int(n_batches))
    base = np.arange(int(batch), dtype=np.int64) * lane
    out = []
    for t in range(max(0, steps)):
        off = base + t * int(seq_len)
        idx = off[:, None] + np.arange(int(seq_len) + 1, dtype=np.int64)[None, :]
        out.append(idx[:, 1:])
    return out


def slice_bpc(nll_nats: Sequence[np.ndarray], positions: Sequence[np.ndarray],
              index: RevisitIndex, *, min_n: int = 1) -> Dict[str, Any]:
    """Aggregate per-token NLL into per-bin bpc.

    Args:
        nll_nats: per-batch ``(batch, seq_len)`` next-byte NLL **in nats**.
        positions: the matching stream positions from
            :func:`contiguous_target_positions`.

    ⛔ **A bin with too few samples is reported WITH its ``n``, never silently
    averaged away.** Bins below ``min_n`` carry ``"bpc": None`` and keep their
    count, so a reader cannot mistake an empty bucket for a good score.
    """
    nb = index.n_bins
    tot = np.zeros(nb, dtype=np.float64)
    cnt = np.zeros(nb, dtype=np.int64)
    for v, p in zip(nll_nats, positions, strict=True):
        v = np.asarray(v, dtype=np.float64).reshape(-1)
        p = np.asarray(p, dtype=np.int64).reshape(-1)
        ok = (p >= 0) & (p < index.bin_of_position.shape[0])
        v, p = v[ok], p[ok]
        b = index.bin_of_position[p]
        m = b >= 0
        np.add.at(tot, b[m], v[m])
        np.add.at(cnt, b[m], 1)
    bins = {}
    for i, lab in enumerate(index.labels):
        n_i = int(cnt[i])
        bins[lab] = {
            "n": n_i,
            "bpc": (float(tot[i] / n_i / LN2) if n_i >= max(1, int(min_n))
                    else None),
            "nll_nats": (float(tot[i] / n_i) if n_i >= max(1, int(min_n))
                         else None),
            "sufficient": bool(n_i >= max(1, int(min_n))),
        }
    n_all = int(cnt.sum())
    return {
        "unit": index.unit,
        "edges": list(index.edges),
        "labels": list(index.labels),
        "n_documents": index.n_documents,
        "bins": bins,
        "n_scored": n_all,
        "bpc_all_binned": float(tot.sum() / n_all / LN2) if n_all else None,
        "min_n": int(min_n),
    }


@dataclass
class SliceReport:
    """The JSON artifact for one run: per-arm slices + the controls."""

    corpus: str
    split: str
    arms: Dict[str, Any] = field(default_factory=dict)
    controls: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        return {"corpus": self.corpus, "split": self.split,
                "citation": {
                    "slice_definition": "Sun, Krishna, Mattarella-Micke, Iyyer "
                                        "(2021), EMNLP, arXiv:2109.09115 — "
                                        "distance-to-last-occurrence buckets + "
                                        "the 'never appears in the prefix' bucket.",
                    "control_precedent": "Khandelwal, He, Qi, Jurafsky (2018), "
                                         "ACL, aclanthology.org/P18-1027 — "
                                         "perturbation-by-distance.",
                    "ours": ["the bin edges", "computed on enwik8/WT-103 BYTES",
                             "computed for EVERY arm incl. the dyn-eval column",
                             "the shuffled-position control applied to the slice"],
                },
                "arms": self.arms, "controls": self.controls, "meta": self.meta}

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_json(), indent=2, default=float))
        return p


def run_controls(data: np.ndarray, *, doc_boundary: Optional[bytes],
                 edges: Sequence[int] = DEFAULT_BIN_EDGES, seed: int = 0,
                 sample_bytes: int = 400_000) -> Dict[str, Any]:
    """All three validity checks in one call, for the artifact's ``controls``."""
    return {
        "non_degeneracy": assert_non_degenerate(
            data, doc_boundary=doc_boundary, edges=edges,
            sample_bytes=sample_bytes),
        "shuffled_position": shuffled_position_control(
            data, doc_boundary=doc_boundary, edges=edges, seed=seed,
            sample_bytes=sample_bytes),
        "content_relabel": content_relabel_control(
            data, doc_boundary=doc_boundary, edges=edges,
            sample_bytes=min(sample_bytes, 200_000)),
    }


def evaluate_slices(model, pcfg, split, *, corpus: str, doc_boundary: bytes,
                    n_batches: Optional[int] = None,
                    edges: Sequence[int] = DEFAULT_BIN_EDGES,
                    min_n: int = 30, index: Optional[RevisitIndex] = None,
                    ) -> Dict[str, Any]:
    """⭐ Score ONE arm's retention slice on a real split.

    Uses the deterministic, order-preserving ``contiguous_batches`` iterator — the
    only iterator an arm with a persistent memory may be evaluated with — and the
    per-token NLL from
    :func:`chlu.training.train_cluformer.eval_token_nll`.
    """
    from chlu.data.enwik8 import contiguous_batches
    from chlu.training.train_cluformer import eval_token_nll, plan_pass
    import jax.numpy as jnp

    idx = index if index is not None else build_revisit_index(
        split.data, doc_boundary=doc_boundary, unit="token", edges=edges)
    n = len(split)
    nb = pcfg.eval_batches if n_batches is None else int(n_batches)
    pos = contiguous_target_positions(n, batch=pcfg.batch, seq_len=pcfg.seq_len,
                                      n_batches=nb)
    nlls: List[np.ndarray] = []
    used: List[np.ndarray] = []
    for (x, y), p in zip(contiguous_batches(split, batch=pcfg.batch,
                                            seq_len=pcfg.seq_len, n_batches=nb),
                         pos, strict=False):
        tk = jnp.asarray(x, dtype=jnp.int32)
        tg = jnp.asarray(y, dtype=jnp.int32)
        plans, _ = plan_pass(model, tk, pcfg)
        nlls.append(np.asarray(eval_token_nll(model, tk, tg, plans, None, None)))
        used.append(p)
    out = slice_bpc(nlls, used, idx, min_n=min_n)
    out["corpus"] = corpus
    out["split"] = split.name
    return out


def evaluate_slices_dyneval(model, pcfg, split, *, corpus: str,
                            doc_boundary: bytes, lr: float,
                            n_batches: Optional[int] = None,
                            edges: Sequence[int] = DEFAULT_BIN_EDGES,
                            min_n: int = 30,
                            index: Optional[RevisitIndex] = None) -> Dict[str, Any]:
    """⭐ The retention slice **for the dynamic-evaluation substitute column**.

    ⛔ **This is the laundering control, and it is why the function exists.**
    Dynamic evaluation (Krause et al., ICML 2018 / arXiv:1904.08378) is SGD on the
    weights over the test stream — a substitute for exactly what a test-time
    memory sells. A retention slice computed only for our arm, and not for the
    substitute, is not a comparison; charter §5 makes the dyn-eval column
    mandatory on every LM table, and this makes it mandatory on every *slice*.

    Protocol is :func:`chlu.training.train_cluformer.dynamic_eval`'s, exactly:
    **strictly causal** — each batch is scored with the current weights first,
    then one SGD step is taken on it, so no position is ever scored by weights
    that have already seen it. ``lr`` is the arm's OWN best learning rate, taken
    from its ``dyneval`` phase, because a badly-tuned substitute is a weak
    substitute and a weak substitute flatters us.
    """
    import equinox as eqx
    import jax.numpy as jnp
    import optax

    from chlu.data.enwik8 import contiguous_batches
    from chlu.training.train_cluformer import (_accum_grads, eval_token_nll,
                                               loss_fn, plan_pass)

    idx = index if index is not None else build_revisit_index(
        split.data, doc_boundary=doc_boundary, unit="token", edges=edges)
    n = len(split)
    nb = pcfg.dyneval_batches if n_batches is None else int(n_batches)
    pos = contiguous_target_positions(n, batch=pcfg.batch, seq_len=pcfg.seq_len,
                                      n_batches=nb)
    n_micro = max(1, int(getattr(pcfg, "accum_steps", 1)))
    opt = optax.sgd(float(lr))
    m = model
    st = opt.init(eqx.filter(m, eqx.is_inexact_array))
    nlls: List[np.ndarray] = []
    used: List[np.ndarray] = []
    for (x, y), p in zip(contiguous_batches(split, batch=pcfg.batch,
                                            seq_len=pcfg.seq_len, n_batches=nb),
                         pos, strict=False):
        tk = jnp.asarray(x, dtype=jnp.int32)
        tg = jnp.asarray(y, dtype=jnp.int32)
        plans, _ = plan_pass(m, tk, pcfg)
        # scored BEFORE the update — strictly causal
        nlls.append(np.asarray(eval_token_nll(m, tk, tg, plans, None, None)))
        used.append(p)
        if n_micro > 1:
            _, grads = _accum_grads(m, tk, tg, plans, n_micro)
        else:
            _, grads = eqx.filter_value_and_grad(loss_fn)(m, tk, tg, plans)
        upd, st = opt.update(grads, st, eqx.filter(m, eqx.is_inexact_array))
        m = eqx.apply_updates(m, upd)
    out = slice_bpc(nlls, used, idx, min_n=min_n)
    out["corpus"] = corpus
    out["split"] = split.name
    out["dyneval_lr"] = float(lr)
    return out


def slice_gap(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Per-bin bpc gap ``b - a`` (positive = ``a`` is better), with both ``n``.

    ⭐ This is the shape the claim is read off: a memory that retains produces a
    gap that **widens with distance**, not a scalar. Bins where either side is
    under-sampled return ``None`` rather than a number.
    """
    out = {}
    for lab in a.get("labels", []):
        ra, rb = a["bins"].get(lab, {}), b["bins"].get(lab, {})
        va, vb = ra.get("bpc"), rb.get("bpc")
        out[lab] = {"gap_bpc": (float(vb - va) if (va is not None
                                                   and vb is not None) else None),
                    "n_a": ra.get("n"), "n_b": rb.get("n")}
    return out
