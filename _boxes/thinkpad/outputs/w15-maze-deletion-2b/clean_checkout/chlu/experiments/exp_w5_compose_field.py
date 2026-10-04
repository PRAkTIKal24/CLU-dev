"""w5 hub/compose-field -- THE GATE: the FIELD store's own content ceiling, in the
currency rows 3 and 4 are decided in, against the FULL RIVAL SET.

Why this module exists, and why it is additive
----------------------------------------------
Wave 4 measured that **no read of the WAVE-3 store beats a parameter-matched MLP, even a
perfect one**: ``oracle_span`` -- the true feature projected onto the span of the wells'
modes, the best answer *any* read of those atoms can give -- sits **6.966 distinct valid
novel tuples per corner BELOW ``mlp_180k``** on all three seeds.  That ceiling is a
property of the wave-3 store's 64 **FITTED** modes.

The field store **CHOOSES** its 64 atoms (greedy span selection against the address
manifold, under a basin-coverage constraint) and therefore spans a **different subspace
at the same 524,288 B**.  Whether some other 524,288 B of content clears the learner is
OPEN, and it is the single number that decides whether this hub's dial -- WHAT IS STORED
-- is alive at all:

* oracle above ``mlp_180k``  => the content dial is **OPEN**; the arms are worth running.
* oracle below ``mlp_180k``  => the content dial is **CLOSED**, as the read dial already
  is, and that is worth more than any arm.  Report it and STOP; do not spend scored
  runs chasing a ceiling measured as unreachable.

**Nothing in ``exp_gen_read`` is edited on disk.**  That module produced the -6.966 and
its arms, corners, judge, acceptance rule and paired estimand are exactly what the field
ceiling has to be measured against; editing it would put the two numbers on two
instruments.  So this module EXTENDS it at runtime -- it appends to ``ARMS``, updates the
three per-arm dicts and wraps ``build_arm``/``arm_features`` -- and every pre-existing
arm keeps its own code path, byte for byte.

**No field pin is re-selected here.**  Every one of them was selected on ``val`` by
``hub/field-store`` and is imported verbatim from ``exp_gen_field_store.PINS``.  This hub
chooses nothing on the held-out set; it measures one declared object once.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path
from typing import Callable

import numpy as np

from . import exp_gen_factor_wells as fw
from . import exp_gen_field_store as fs
from . import exp_gen_read as R
from ..core.gen_read import render_span, span_basis, span_coefficients

# ---------------------------------------------------------------------------
# 0. The pins this hub owns.  There are three, and none of them is a dial.
# ---------------------------------------------------------------------------

W5_PINS = {
    # the field store's own hyper-parameters are INHERITED, not chosen here
    "field_pins_from": "exp_gen_field_store.PINS (selected on val by hub/field-store)",
    # the atom seed follows the leg seed exactly as hub/field-store's did, so three
    # seeds are three genuinely different atom selections and not one object re-judged
    "atom_seed_stride": int(fs.PINS["atom_seed_stride"]),
    "field_atom_seed": int(fs.PINS["field_atom_seed"]),
    # the byte pin.  PRIMARY, and identical for both stores (w5-common section 1)
    "store_bytes": int(fw.PINS["store_bytes"]),
}

#: The legs this module adds.  The first is a DECLARED NON-ARM: it reads the requested
#: row's own feature, so charter B.3 disqualifies it as a mechanism forever.  It is a
#: ceiling, and a ceiling is the only thing that can close a dial.
FIELD_ARMS = (
    "oracle_span_field",            # NOT AN ARM: the FIELD store's content ceiling
    "field_lookup_argmax",          # FLOOR over the field atoms (charter C.1/C.4)
    "oracle_content_pool",          # NOT AN ARM: the BEST REALISABLE content, 524,288 B
    "oracle_content_oracle",        # NOT AN ARM: the ABSOLUTE content ceiling, 524,288 B
    "oracle_content_fine",          # NOT AN ARM: the same at FINE granularity, 524,288 B
    "oracle_true_feature",          # NOT AN ARM: the identity. Every store's ceiling.
    "oracle_content_items",         # NOT AN ARM: content = 64 REAL ITEMS, 524,288 B
    # -- THE ARM and its charter-C family, all on the SAME 524,288 B atom block --
    "pca_tensor",                   # THE ARM: principal content + the product address
    "pca_tensor_sum",               # ABLATION: same rank, address SUMMED not multiplied
    "pca_landscape_removed",        # CONTROL (C.2): atoms zeroed, must collapse
    "pca_landscape_shuffled",       # CONTROL (C.2): atom VALUES permuted, must collapse
    "pca_random_refit",             # CONTROL: random atoms, address book REFIT
    "pca_span_rotated",             # CONTROL: a PREDICTED no-op
    "pca_lookup_argmax",            # FLOOR (C.1/C.4): one atom per well, my address
    "pca_lookup_convex",            # FLOOR (C.1): the mixture over the same atoms
)

#: The arm's store, and every control's, is built from the SAME principal content and
#: differs only in what charter C says to do to the atoms.  ``None`` = the arm itself.
PCA_ARMS = {
    "pca_tensor": (None, "tensor"),
    "pca_tensor_sum": (None, "tensor_sum"),
    "pca_landscape_removed": ("zero", "tensor"),
    "pca_landscape_shuffled": ("value_permute", "tensor"),
    "pca_random_refit": ("random_span", "tensor"),
    "pca_span_rotated": ("rotate", "tensor"),
    "pca_lookup_argmax": (None, "tensor_argmax"),
    "pca_lookup_convex": (None, "tensor_convex"),
}
#: Applied BEFORE the address book is fitted -- the strong form of the C.2 question: if
#: the arm survives atoms it never saw, the answer was never in the store.
PCA_PRE_FIT = ("random_span", "rotate")

FIELD_ARM_CLASS = {
    "oracle_span_field": "NOT AN ARM (ceiling, FIELD store)",
    "field_lookup_argmax": "FLOOR (charter C.1/C.4, matched-bytes lookup, FIELD atoms)",
    "oracle_content_pool": "NOT AN ARM (ceiling, BEST REALISABLE content @524,288 B)",
    "oracle_content_oracle": "NOT AN ARM (ceiling, ABSOLUTE content @524,288 B, K=5)",
    "oracle_content_fine": "NOT AN ARM (ceiling, ABSOLUTE content @524,288 B, fine K)",
    "oracle_true_feature": "NOT AN ARM (ceiling of EVERY store, any budget)",
    "oracle_content_items": "NOT AN ARM (ceiling, 64 REAL ITEMS @524,288 B)",
    "pca_tensor": "ARM",
    "pca_tensor_sum": "ABLATION (same rank, address summed not multiplied)",
    "pca_landscape_removed": "CONTROL (charter C.2, atoms zeroed)",
    "pca_landscape_shuffled": "CONTROL (charter C.2, atom values permuted)",
    "pca_random_refit": "CONTROL (random atoms, address book REFIT)",
    "pca_span_rotated": "CONTROL (predicted NO-OP: the span is unchanged)",
    "pca_lookup_argmax": "FLOOR (charter C.1/C.4, matched-bytes lookup)",
    "pca_lookup_convex": "FLOOR (charter C.1, the mixture)",
}

#: The two optimal-content ceilings, and the one sentence that makes them load-bearing.
#:
#: The content a store of this layout can hold is ``A`` atoms in each of ``K`` wells, and
#: the byte pin fixes ``A * D * 4 = 524,288 B``, i.e. ``A = 64`` when ``D = 2048``.  The
#: best answer ANY read of such a store can give a request is the requested feature's
#: orthogonal projection onto the row space of those atoms, per well.  So the best answer
#: ANY 524,288 B store of this layout can give, over a set of rows, is the projection
#: onto the per-well rank-``A`` subspace that minimises squared reconstruction error on
#: those rows -- and by Eckart-Young that subspace is the leading ``A`` principal
#: directions of the rows' own features, well by well.
#:
#: ``oracle_content_pool`` fits that subspace on the FIT POOL: content chosen from
#: training data only, so it is the best content a store could actually be built to hold.
#: ``oracle_content_oracle`` fits it on the SCORED CORNER ROWS themselves: the content is
#: chosen knowing the rows it will be judged on, which no deployable store can do.  It is
#: therefore an upper bound that is not merely tight but unreachable, and a number below
#: the bar there closes the question for the byte pin rather than for one atom heuristic.
CONTENT_SOURCE = {"oracle_content_pool": "pool", "oracle_content_oracle": "corners",
                  "oracle_content_fine": "corners"}

#: Which legs re-partition, and how.  ``None`` keeps the field store's own inherited
#: ``factor`` wells; ``"fine"`` re-cuts the SAME channels into the coarsest partition
#: whose every well is at most ``A`` channels wide, so ``min(A, d_k) = d_k`` everywhere
#: and the per-well projection is the identity.  ⭐ The atom block is ``A x D`` floats
#: either way, so **both are 524,288 B**: this leg is what measures whether the byte pin
#: binds the content ceiling at all, or whether the WELL GRANULARITY does.
CONTENT_PARTITION = {"oracle_content_fine": "fine"}


# ---------------------------------------------------------------------------
# 1. The adapter: a FieldStore seen through the span interface
# ---------------------------------------------------------------------------


class FieldSpanStore:
    """A :class:`chlu.core.gen_field.FieldStore` at ``gen_read``'s span interface.

    ``gen_read.span_basis`` / ``span_coefficients`` / ``render_span`` need exactly three
    things off a store: ``modes`` (per well, the atom matrix whose ROW SPACE is what a
    read can reach), ``well_channels`` and ``dim``.  A FieldStore has all three; it just
    spells them differently.  Wrapping rather than converting is deliberate -- the bytes
    that are judged are the bytes in ``field.buf``, and this object holds a reference to
    that same store, so there is no second copy of the content to drift.

    Only ALIVE atoms are exposed.  A dead atom holds no content and including it would
    inflate the span with a row the store does not deploy, which would make the ceiling
    generous in the wrong direction.
    """

    def __init__(self, field):
        self.field = field
        alive = np.asarray(field.view("atom_alive"), np.uint8) > 0
        self.n_alive = int(alive.sum())
        self.well_channels = [np.asarray(c, np.int64)
                              for c in field.meta["well_channels"]]
        self.modes = [np.ascontiguousarray(
            np.asarray(field.atoms_of_well(k), np.float32)[alive])
            for k in range(field.K)]
        self.dim = int(field.D)

    @property
    def K(self) -> int:
        return len(self.modes)

    def span_ranks(self, tol: float = 1e-6) -> list:
        return [int(U.shape[1]) for U in span_basis(self, tol)]


def _assert_span_is_the_content(store: "FieldSpanStore", field) -> dict:
    """The ceiling must be a property of the DEPLOYED bytes and of nothing else.

    Two assertions, both cheap and both load-bearing:

    1. every exposed mode row is a byte-for-byte view of the ``atoms`` segment -- so the
       subspace being projected onto is the one in the store image, not a re-fit of it;
    2. the total number of exposed mode rows equals the number of live atoms -- so no
       dead atom widens the span.
    """
    A = np.asarray(field.view("atoms"), np.float32)
    alive = np.asarray(field.view("atom_alive"), np.uint8) > 0
    n_rows = int(sum(M.shape[0] for M in store.modes))
    if n_rows != store.K * int(alive.sum()):
        raise AssertionError(
            f"the span exposes {n_rows} mode rows but the store deploys "
            f"{store.K * int(alive.sum())} live atom rows")
    for k in range(store.K):
        ref = np.asarray(field.atoms_of_well(k), np.float32)[alive]
        if not np.array_equal(np.asarray(store.modes[k], np.float32), ref):
            raise AssertionError(f"well {k}'s modes are not the store's own atom bytes")
    return {"n_live_atoms": int(alive.sum()), "n_wells": int(store.K),
            "atoms_segment_bytes": int(A.size * 4),
            "span_ranks": store.span_ranks(),
            "what": ("the ceiling is the orthogonal projection onto the row space of "
                     "the DEPLOYED atom bytes; no refit, no dead atom")}


class PCASpanStore:
    """The L2-OPTIMAL content at the byte pin, at the same span interface.

    ``modes[k]`` is the leading ``A`` principal directions of the source rows' features
    restricted to well ``k``'s channels.  It occupies exactly the same bytes as a field
    or wave-3 store's atom block -- ``A x d_k`` float32 per well, ``A * D * 4`` in total
    -- and by Eckart-Young no other ``A``-row atom block for that well reconstructs the
    source rows better in squared error.

    It is a DECLARED NON-ARM twice over: the read that uses it is handed the true
    feature, and (for ``source="corners"``) the content itself was chosen knowing the
    scored rows.  ⛔ Neither can ever be quoted as a mechanism.  Its only job is to make
    a negative verdict a property of the BYTE BUDGET instead of a property of whichever
    atom-selection heuristic happened to be pinned.
    """

    def __init__(self, X: np.ndarray, well_channels, n_atoms: int, dim: int):
        self.well_channels = [np.asarray(c, np.int64) for c in well_channels]
        self.dim = int(dim)
        self.n_atoms = int(n_atoms)
        self.modes, self.energy = [], []
        Xf = np.asarray(X, np.float32)
        for ch in self.well_channels:
            M = np.asarray(Xf[:, ch], np.float64)
            mu = M.mean(axis=0)
            # The MEAN is one of the atoms, so the subspace is affine-capable exactly as
            # a store's is: a store writes items, and an item may be the mean item.
            Mc = M - mu
            _, s, Vt = np.linalg.svd(Mc, full_matrices=False)
            r = min(self.n_atoms - 1, Vt.shape[0])
            B = np.vstack([mu[None, :], Vt[:r]]).astype(np.float32)
            # ⛔ The BYTES are A x d_k whatever the span is.  A well narrower than A
            # cannot use all its atoms, and the honest way to say so is to deploy them
            # anyway as zero rows: they cost the same bytes and buy no span, which is
            # exactly what happens to a real store written into a narrow well.
            if B.shape[0] < self.n_atoms:
                B = np.vstack([B, np.zeros((self.n_atoms - B.shape[0], B.shape[1]),
                                           np.float32)])
            self.modes.append(np.ascontiguousarray(B.astype(np.float32)))
            tot = float((s ** 2).sum())
            self.energy.append(float((s[:r] ** 2).sum() / tot) if tot > 0 else 1.0)

    @property
    def K(self) -> int:
        return len(self.modes)

    def byte_ledger(self) -> dict:
        b = int(sum(M.size for M in self.modes) * 4)
        return {"deployed_bytes": b, "budget_bytes": int(W5_PINS["store_bytes"]),
                "matches_budget": bool(b == int(W5_PINS["store_bytes"])),
                "param_count": b // 4,
                "what": (f"{self.n_atoms} atoms x {self.dim} float32 -- the per-well "
                         "leading principal subspace, the SAME block a field or wave-3 "
                         "store's atoms occupy"),
                "variance_explained_per_well": [round(e, 6) for e in self.energy]}


class _ModeStore:
    """The minimal store at the span interface: modes, channels, dim, and a ledger.

    Used for the CONTROLS, which have to keep the shape and the byte ledger of the arm's
    store exactly and change only what is written in it (charter C.2).
    """

    def __init__(self, modes, well_channels, dim: int, what: str = ""):
        self.modes = [np.ascontiguousarray(np.asarray(M, np.float32)) for M in modes]
        self.well_channels = [np.asarray(c, np.int64) for c in well_channels]
        self.dim = int(dim)
        self.what = what

    @property
    def K(self) -> int:
        return len(self.modes)

    def byte_ledger(self) -> dict:
        b = int(sum(M.size for M in self.modes) * 4)
        return {"deployed_bytes": b, "budget_bytes": int(W5_PINS["store_bytes"]),
                "matches_budget": bool(b == int(W5_PINS["store_bytes"])),
                "param_count": b // 4, "what": self.what}


def basis_at_fixed_rank(store, ranks) -> list:
    """``span_basis`` but with each well's rank FORCED to the arm's own ``r_k``.

    ⚠ Why this exists, and it is a correctness fix rather than a convenience.  A
    POST-fit control keeps the ARM's address book and replaces only the atoms.  The
    book emits ``r_k`` coefficients per well, fixed when it was fitted against the
    unperturbed store.  ``span_basis`` re-derives the rank from the PERTURBED atoms'
    singular values, and a permutation of the atom values can move it — measured: the
    arm's well came out rank 63 and the permuted one rank 64, and ``render_span`` then
    raised ``matmul: size 64 is different from 63`` mid-leg.

    Truncating (or zero-padding) the perturbed basis to the book's rank is the only
    reading that keeps the control a control: same book, same rank, same bytes, atoms
    destroyed.  ⛔ It cannot flatter the control — a rank the book never learned to
    address carries no information it could use.
    """
    out = []
    for U, r in zip(span_basis(store), [int(r) for r in ranks]):
        U = np.asarray(U, np.float32)
        if U.shape[1] > r:
            U = U[:, :r]
        elif U.shape[1] < r:
            U = np.hstack([U, np.zeros((U.shape[0], r - U.shape[1]), np.float32)])
        out.append(np.ascontiguousarray(U))
    return out


def perturb_modes(store, kind: str, seed: int, what: str = ""):
    """charter C.2, at this hub's store interface.  Same shape, same ledger, atoms moved.

    ``zero`` the atoms hold nothing; ``value_permute`` the same numbers with the geometry
    destroyed; ``random_span`` random atoms at the same Frobenius norm (stronger than
    C.2 asks, because the address book is then REFIT against them); ``rotate`` a random
    orthogonal rotation INSIDE each well, which leaves the span mathematically unchanged
    and is therefore a PREDICTED NO-OP, reported as one.
    """
    rng = np.random.default_rng(int(seed) * 7919 + 20_260_919)
    out = []
    for G in store.modes:
        Gm = np.asarray(G, np.float32)
        if kind == "zero":
            out.append(np.zeros_like(Gm))
        elif kind == "value_permute":
            out.append(Gm.reshape(-1)[rng.permutation(Gm.size)].reshape(Gm.shape))
        elif kind == "random_span":
            Rn = rng.normal(size=Gm.shape).astype(np.float32)
            Rn *= float(np.linalg.norm(Gm)) / (float(np.linalg.norm(Rn)) + 1e-12)
            out.append(Rn)
        elif kind == "rotate":
            Q, _ = np.linalg.qr(rng.normal(size=(Gm.shape[0], Gm.shape[0])))
            out.append(Q.astype(np.float32) @ Gm)
        else:
            raise ValueError(kind)
    return _ModeStore(out, store.well_channels, store.dim,
                      what or f"{kind} control over the arm's own atom block")


class _ItemSpanStore:
    """Content that is WRITTEN ITEMS, at the span interface.  ``A`` rows, ``A x D``
    float32, the same block and the same semantics the field store deploys."""

    def __init__(self, X: np.ndarray, well_channels, dim: int):
        Xf = np.asarray(X, np.float32)
        self.well_channels = [np.asarray(c, np.int64) for c in well_channels]
        self.dim = int(dim)
        self.n_atoms = int(Xf.shape[0])
        self.modes = [np.ascontiguousarray(Xf[:, c]) for c in self.well_channels]

    @property
    def K(self) -> int:
        return len(self.modes)

    def byte_ledger(self) -> dict:
        b = int(sum(M.size for M in self.modes) * 4)
        return {"deployed_bytes": b, "budget_bytes": int(W5_PINS["store_bytes"]),
                "matches_budget": bool(b == int(W5_PINS["store_bytes"])),
                "param_count": b // 4,
                "what": (f"{self.n_atoms} written items x {self.dim} float32 -- one "
                         "item in every well, the field store's own layout")}


def _content_source_rows(out_dir: Path, source: str, log: Callable) -> np.ndarray:
    """The rows the optimal content is fitted from.  Two sources, both declared."""
    cond = fw.load_condition()
    if source == "pool":
        return np.asarray(np.load(out_dir / "partitions.npz")["pool"], np.int64)
    if source != "corners":
        raise ValueError(source)
    # EVERY row of EVERY corner this condition scores -- one content for the whole
    # held-out set, because a store is one object and does not get to be refitted per
    # request.  ⛔ This is the unreachable bound, and it is labelled so everywhere.
    if fw.PINS["condition"] == "val":
        cs = fw.val_corners(cond)
    else:
        cs = fw.nway_corners(cond) + fw.pair_subcorners(cond)
    rows = np.unique(np.concatenate([c["rows"] for c in cs]))
    log(f"  optimal content source = the {len(cs)} SCORED corners' own "
        f"{rows.size} rows (UNREACHABLE bound)")
    return rows


# ---------------------------------------------------------------------------
# 2. Building the field store inside THIS leg
# ---------------------------------------------------------------------------


def build_field_for_leg(out_dir: Path, log: Callable = print) -> dict:
    """Write the field store on the leg's own pool, seed and condition.  Resumable.

    The pool is ``partitions.npz['pool']`` -- the SAME rows, in the same order, that
    every other store arm in ``exp_gen_read`` is fitted from.  That is what makes the two
    ceilings like-for-like: the wave-3 store and the field store see one pool and differ
    only in what they choose to write into 524,288 B.

    The partition is also inherited (``factor_aligned_balanced``, the label-MI
    partition), because w5-common section 3.5 declares placement EXHAUSTED and this hub's
    dial is the CONTENT, not where the channels go.  Holding the partition fixed is what
    isolates the content.
    """
    from ..core.gen_field import (
        assert_read_request_only, build_field, field_byte_ledger,
    )

    out_dir = Path(out_dir)
    d = out_dir / "field"
    done = d / "field.pkl"
    if done.exists():
        log("  field store: already built for this leg, loading")
        with open(done, "rb") as fh:
            return pickle.load(fh)

    seed = int(fw.PINS["seed"])
    cond = fw.load_condition()
    grid = cond["grid"]
    parts = fw.build_partitions(out_dir, log=log)
    labels = np.asarray(parts["labels"]["factor"], np.int64)
    K = int(labels.max()) + 1
    hosts = fs._host_factors(parts["info"], K)
    pool = np.asarray(np.load(out_dir / "partitions.npz")["pool"], np.int64)
    F = fw.features_mmap(seed)
    t0 = time.time()
    X = fw.gather(F, pool)
    Y = grid.factor_indices_of(pool)
    cards = [int(c) for c in grid.cardinalities]
    log(f"  field store: pool {pool.size} rows, hosts {hosts}, seed {seed}")
    spec = fs.field_spec(seed)          # every field pin, inherited verbatim
    field = build_field(X, Y, pool, labels, hosts, cards, spec, log=log)
    del X
    led = field_byte_ledger(field, int(W5_PINS["store_bytes"]))
    if not led["matches_budget"]:
        raise AssertionError(
            f"the field store deploys {led['deployed_bytes']} B against the pinned "
            f"{W5_PINS['store_bytes']} B ledger -- the byte pin has moved")
    vals = grid.values_of(pool[:64])
    ro = assert_read_request_only(field, vals, grid.values)
    obj = {"field": field, "seed": seed, "condition": str(fw.PINS["condition"]),
           "byte_ledger": led, "request_only": ro, "manifest": field.manifest(),
           "pool_n_rows": int(pool.size), "fit_seconds": time.time() - t0,
           "spec": spec.to_dict() if hasattr(spec, "to_dict") else str(spec)}
    d.mkdir(parents=True, exist_ok=True)
    with open(done, "wb") as fh:
        pickle.dump(obj, fh)
    fw._jdump({k: v for k, v in obj.items() if k != "field"}, d / "field.json")
    log(f"  field store: built in {obj['fit_seconds']:.1f}s; atoms "
        f"{led['deployed_bytes']:,} B (image {led['total_image_bytes']:,} B) "
        f"sha {field.sha256()[:16]}")
    return obj


def build_pca_content(out_dir: Path, log: Callable = print) -> dict:
    """THE ARM'S CONTENT.  The per-well principal subspace of the FIT POOL.  Resumable.

    What is written, in one sentence: the ``A = 64`` leading principal directions of the
    fit pool's features, well by well, with the pool mean as the first atom -- exactly
    ``A x D`` float32, exactly 524,288 B, the same block the field store and the wave-3
    store deploy.

    Three properties worth stating because they are what make this content admissible.

    * ⭐ **No label enters it.**  The partition is inherited (labels once at fit time,
      w5-common section 4); the CONTENT itself is an unsupervised decomposition of
      training features.  It is strictly more label-free than the field store's item
      selection, which uses labels to decide what to write.
    * ⭐ **No held-out row enters it.**  It is fitted from ``partitions.npz['pool']``,
      the same pool every other store arm here is fitted from.
    * ⚠ **It gives up exact per-item deletion.**  A principal direction is not a written
      item, so charter B.4's "deletion is an exact operation on atoms" does not hold of
      this content the way it holds of the field store's.  ⛔ That is a declared cost of
      this content and it is why ``oracle_content_items`` is on the table beside it:
      rows 3 and 4 are this hub's target, and the item-store ceiling is reported so the
      price of keeping deletion is visible rather than hidden.
    """
    out_dir = Path(out_dir)
    cache = out_dir / "pca_content.pkl"
    if cache.exists():
        with open(cache, "rb") as fh:
            return pickle.load(fh)
    fw.build_partitions(out_dir, log=log)
    D = int(fw.PINS["D"])
    n_atoms = int(W5_PINS["store_bytes"]) // (4 * D)
    labels = np.asarray(dict(np.load(out_dir / "partitions.npz"))["factor"], np.int64)
    K = int(labels.max()) + 1
    wc = [np.flatnonzero(labels == k).astype(np.int64) for k in range(K)]
    pool = np.asarray(np.load(out_dir / "partitions.npz")["pool"], np.int64)
    X = fw.gather(fw.features_mmap(int(fw.PINS["seed"])), pool)
    store = PCASpanStore(X, wc, n_atoms, D)
    del X
    led = store.byte_ledger()
    if not led["matches_budget"]:
        raise AssertionError(
            f"the principal content deploys {led['deployed_bytes']} B against the "
            f"pinned {W5_PINS['store_bytes']} B ledger -- the byte pin has moved")
    obj = {"store": store, "byte_ledger": led, "well_sizes": [int(c.size) for c in wc],
           "n_atoms": n_atoms, "n_wells": K, "n_pool_rows": int(pool.size),
           "partition": "factor (K=5, inherited label-MI wells)",
           "label_use": ("labels align the wells ONCE at fit time and ground the "
                         "address book; the CONTENT is unsupervised (charter B.6)")}
    with open(cache, "wb") as fh:
        pickle.dump(obj, fh)
    fw._jdump({k: v for k, v in obj.items() if k != "store"},
              out_dir / "pca_content.json")
    log(f"  principal content: {n_atoms} atoms x {D} float32 = "
        f"{led['deployed_bytes']:,} B, wells {obj['well_sizes']}, "
        f"variance explained {[round(e, 4) for e in store.energy]}")
    return obj


def _build_pca_arm(arm: str, out_dir: Path, arm_dir: Path, done: Path, t0: float,
                   log: Callable) -> dict:
    """The arm and every one of its charter-C controls, off one content object."""
    from ..core.gen_read import TensorAddress, assert_read_request_only

    cond = fw.load_condition()
    grid = cond["grid"]
    cards = [int(c) for c in grid.cardinalities]
    base = build_pca_content(out_dir, log=log)
    store = base["store"]
    kind, verb = PCA_ARMS[arm]

    if kind in PCA_PRE_FIT:
        # the atoms are replaced BEFORE the book is fitted, so the book is fitted
        # against the perturbed store and the control is the strong form
        store = perturb_modes(store, kind, int(fw.PINS["seed"]))
    basis = span_basis(store)

    pool = np.asarray(np.load(out_dir / "partitions.npz")["pool"], np.int64)
    pool = pool[: int(R.READ_PINS["n_addr_train"])]
    tag = kind if kind in PCA_PRE_FIT else ""
    cache = out_dir / (
        f"address_pca_{'sum' if verb == 'tensor_sum' else 'prod'}"
        f"_R{int(R.READ_PINS['rank'])}_lam{R.READ_PINS['ridge']}"
        f"_sw{int(R.READ_PINS['sweeps'])}_n{int(R.READ_PINS['n_addr_train'])}"
        + (f"_{tag}" if tag else "") + ".pkl")
    if cache.exists():
        with open(cache, "rb") as fh:
            addr = pickle.load(fh)
        log(f"  address book: reused {cache.name}")
    else:
        Y, T = R._pool_span_targets(store, pool, basis, log=log)
        ntr = int(0.9 * Y.shape[0])
        addr = TensorAddress(cards, rank=int(R.READ_PINS["rank"]),
                             ridge=float(R.READ_PINS["ridge"]),
                             seed=int(R.READ_PINS["address_seed"]))
        addr.multiplicative = (verb != "tensor_sum")
        addr.fit(Y[:ntr], [t[:ntr] for t in T], sweeps=int(R.READ_PINS["sweeps"]),
                 log=log, Y_val=Y[ntr:], targets_val=[t[ntr:] for t in T])
        with open(cache, "wb") as fh:
            pickle.dump(addr, fh)

    obj = {"arm": arm, "read": verb, "kind": "span",
           "store": store, "basis": basis, "address": addr,
           "byte_ledger": dict(base["byte_ledger"]),
           "address_bytes": addr.byte_ledger(),
           "head_bytes": {"head_bytes": 0,
                          "note": "this read's address book IS the tensor book, "
                                  "declared beside the store and never netted in"},
           "partition": base["partition"], "label_use": base["label_use"],
           "content": "per-well principal subspace of the fit pool (unsupervised)"}

    # charter B.3: the read must be reachable from the REQUEST alone.  Asserted on the
    # deployed object, not argued for in a comment.
    vals0 = grid.values_of(pool[:64])
    obj["request_only"] = assert_read_request_only(
        addr, store, vals0, grid.values, basis, rng_seed=int(fw.PINS["seed"]))

    if kind is not None and kind not in PCA_PRE_FIT:
        # applied AFTER the book is fitted: the book is the arm's, the atoms are not
        obj["store"] = perturb_modes(store, kind, int(fw.PINS["seed"]))
        obj["perturbation"] = {"kind": kind, "when": "after the address book was fitted"}
        if kind == "zero":
            obj["zeroed"] = True
            obj["basis"] = basis
        else:
            # ⚠ FORCED to the arm's rank: the address book emits r_k coefficients and
            # the perturbed atoms' own rank can differ.  See basis_at_fixed_rank.
            obj["basis"] = basis_at_fixed_rank(obj["store"],
                                               [np.asarray(U).shape[1] for U in basis])
            obj["basis_rank_note"] = (
                "the control keeps the ARM's address book, so its basis is truncated "
                "or zero-padded to the arm's per-well rank; a rank the book never "
                "learned to address carries nothing it could use")
    elif kind is not None:
        obj["perturbation"] = {"kind": kind,
                               "when": "BEFORE the address book was fitted"}

    led = obj["store"].byte_ledger() if hasattr(obj["store"], "byte_ledger") else None
    if led is not None and not led["matches_budget"]:
        raise AssertionError(f"{arm} deploys {led['deployed_bytes']} B, not the pinned "
                             f"{W5_PINS['store_bytes']} B")
    obj["fit_seconds"] = time.time() - t0
    arm_dir.mkdir(parents=True, exist_ok=True)
    with open(done, "wb") as fh:
        pickle.dump(obj, fh)
    fw._jdump({k: v for k, v in obj.items()
               if k not in ("store", "basis", "address")}, arm_dir / "arm.json")
    log(f"  arm {arm}: fitted in {obj['fit_seconds']:.1f}s; store "
        f"{obj['byte_ledger']['deployed_bytes']:,} B + address "
        f"{obj['address_bytes'].get('address_bytes', 0):,} B")
    return obj


# ---------------------------------------------------------------------------
# 3. The two new legs, at exp_gen_read's arm interface
# ---------------------------------------------------------------------------


def build_arm(arm: str, out_dir: Path, log: Callable = print) -> dict:
    """Dispatch: the field legs here, every pre-existing leg to ``exp_gen_read``."""
    if arm not in FIELD_ARMS:
        return _ORIG_BUILD_ARM(arm, out_dir, log=log)

    out_dir = Path(out_dir)
    arm_dir = out_dir / "arms" / arm
    done = arm_dir / "arm.pkl"
    if done.exists():
        log(f"  arm {arm}: already fitted, loading")
        with open(done, "rb") as fh:
            return pickle.load(fh)

    t0 = time.time()

    # -- THE ARM: principal content, read by the product address book -------
    if arm in PCA_ARMS:
        return _build_pca_arm(arm, out_dir, arm_dir, done, t0, log)

    # -- the identity: the ceiling of EVERY store, at any budget -------------
    if arm == "oracle_true_feature":
        obj = {
            # ⚠ ``kind`` is REQUIRED by fw.delete_value_from_head, which fw.run_arm
            # calls for row 7 on every arm.  A ceiling has no address book, so the
            # non-"span" branch returns it unchanged and row 7 correctly measures
            # nothing for it -- which is the honest answer for a DECLARED NON-ARM.
            "arm": arm, "read": "true_feature", "kind": "oracle",
            # fw.run_arm fingerprints "the object that must not have been refitted
            # between the two halves of row 7" as o["protos"].protos.  This leg has NO
            # store at all, so the sequence is EMPTY and the check is declared as
            # proving nothing here rather than left to look like a passing check.
            "protos": R._FitFingerprint(()),
            "fingerprint_note": ("EMPTY: the identity read deploys no content, so "
                                 "there is nothing a refit could have moved"),
            "byte_ledger": {
                "deployed_bytes": None, "budget_bytes": int(W5_PINS["store_bytes"]),
                "param_count": None,
                "what": ("NO STORE. This leg returns the requested row's encoder "
                         "feature unchanged, so it is the ceiling of every store at "
                         "every byte budget and of the judge path itself.")},
            "address_bytes": {"address_bytes": 0},
            "head_bytes": {"head_bytes": 0},
            "declared": ("NOT AN ARM (charter B.3): the identity on the true feature. "
                         "It bounds this metric, not this store."),
            "fit_seconds": 0.0,
        }
        arm_dir.mkdir(parents=True, exist_ok=True)
        with open(done, "wb") as fh:
            pickle.dump(obj, fh)
        fw._jdump(obj, arm_dir / "arm.json")
        return obj

    # -- content = 64 REAL ITEMS, the store's own semantics ------------------
    if arm == "oracle_content_items":
        # ⭐ Why this rung exists.  ``oracle_content_pool`` writes DIRECTIONS, and a
        # direction is not an item: charter B.4 wants deletion to be an exact operation
        # on atoms, and that is only meaningful if an atom IS a written item.  This leg
        # keeps the store's own semantics -- 64 whole training rows, written into every
        # well exactly as the field store writes items -- and asks what their span is
        # worth.  ⛔ Still NOT AN ARM: the read is handed the true feature.  The gap
        # between this and ``oracle_content_pool`` is the price of insisting on items;
        # the gap between this and ``oracle_span_field`` is the field store's atom
        # SELECTION, and those are two different things to have measured.
        fw.build_partitions(out_dir, log=log)
        D = int(fw.PINS["D"])
        n_atoms = int(W5_PINS["store_bytes"]) // (4 * D)
        labels = np.asarray(dict(np.load(out_dir / "partitions.npz"))["factor"],
                            np.int64)
        K = int(labels.max()) + 1
        wc = [np.flatnonzero(labels == k).astype(np.int64) for k in range(K)]
        pool = np.asarray(np.load(out_dir / "partitions.npz")["pool"], np.int64)
        # A STRATIFIED draw, not a uniform one: the same idea the field store's
        # selection has, done without its basin-coverage constraint.  The seed is the
        # leg's, so three seeds are three genuinely different item sets.
        rng = np.random.default_rng(int(W5_PINS["field_atom_seed"])
                                    + int(W5_PINS["atom_seed_stride"])
                                    * int(fw.PINS["seed"]))
        pick = np.sort(pool[rng.permutation(pool.size)[:n_atoms]])
        Xi = fw.gather(fw.features_mmap(int(fw.PINS["seed"])), pick)
        store = _ItemSpanStore(Xi, wc, D)
        led = store.byte_ledger()
        if not led["matches_budget"]:
            raise AssertionError(
                f"{arm} deploys {led['deployed_bytes']} B against the pinned "
                f"{W5_PINS['store_bytes']} B ledger -- the byte pin has moved")
        basis = span_basis(store)
        ranks = [int(U.shape[1]) for U in basis]
        log(f"  arm {arm}: {n_atoms} real items x {D} float32, span ranks {ranks} "
            f"(sum {sum(ranks)}), {led['deployed_bytes']:,} B")
        obj = {"arm": arm, "read": "oracle_field", "kind": "oracle",
               "store": store, "basis": basis, "byte_ledger": led,
               "protos": R._FitFingerprint(list(store.modes)),
               "address_bytes": {"address_bytes": 0,
                                 "what": "a ceiling has no address; charter B.3"},
               "head_bytes": {"head_bytes": 0},
               "span_audit": {"span_ranks": ranks, "n_atoms_per_well": n_atoms,
                              "n_wells": K, "content_source": "64 real pool rows",
                              "item_rows": [int(r) for r in pick],
                              "total_reachable_dim": int(sum(ranks))},
               "declared": ("NOT AN ARM (charter B.3): the read is handed the true "
                            "feature. The CONTENT is 64 whole written items, so this "
                            "is what the store's own semantics can reach."),
               "fit_seconds": time.time() - t0}
        arm_dir.mkdir(parents=True, exist_ok=True)
        with open(done, "wb") as fh:
            pickle.dump(obj, fh)
        fw._jdump({k: v for k, v in obj.items() if k not in ("store", "basis")},
                  arm_dir / "arm.json")
        return obj

    # -- the optimal-content ceilings ---------------------------------------
    if arm in CONTENT_SOURCE:
        fw.build_partitions(out_dir, log=log)
        D = int(fw.PINS["D"])
        n_atoms = int(W5_PINS["store_bytes"]) // (4 * D)
        labels = np.asarray(dict(np.load(out_dir / "partitions.npz"))["factor"],
                            np.int64)
        K = int(labels.max()) + 1
        wc = [np.flatnonzero(labels == k).astype(np.int64) for k in range(K)]
        part_name = "factor (K=5, inherited: the field store's own wells)"
        if CONTENT_PARTITION.get(arm) == "fine":
            # ⭐ The SAME channels, re-cut so no well is wider than A.  Each well's own
            # A atoms then span it exactly, so the projection is the identity and the
            # ceiling is the true feature -- at the IDENTICAL 524,288 B atom block,
            # because that block is A x D floats whatever K is.  Channels are kept in
            # their inherited well order so the re-cut refines the label-MI partition
            # rather than replacing it (w5-common section 3.5: inherit the partition).
            order = np.concatenate(wc)
            wc = [np.ascontiguousarray(order[s:s + n_atoms])
                  for s in range(0, order.size, n_atoms)]
            K = len(wc)
            part_name = (f"fine (K={K}, the inherited factor wells REFINED into blocks "
                         f"of at most A={n_atoms} channels; same 524,288 B)")
        log(f"  arm {arm}: partition = {part_name}")
        rows = _content_source_rows(out_dir, CONTENT_SOURCE[arm], log)
        X = fw.gather(fw.features_mmap(int(fw.PINS["seed"])), rows)
        store = PCASpanStore(X, wc, n_atoms, D)
        del X
        led = store.byte_ledger()
        if not led["matches_budget"]:
            raise AssertionError(
                f"{arm} deploys {led['deployed_bytes']} B against the pinned "
                f"{W5_PINS['store_bytes']} B ledger -- the byte pin has moved")
        basis = span_basis(store)
        ranks = [int(U.shape[1]) for U in basis]
        log(f"  arm {arm}: {n_atoms} atoms/well, span ranks {ranks} (sum {sum(ranks)}), "
            f"{led['deployed_bytes']:,} B, variance explained "
            f"{[round(e, 4) for e in store.energy]}")
        obj = {
            "arm": arm, "read": "oracle_field", "kind": "oracle",
            "store": store, "basis": basis, "byte_ledger": led,
            # the fingerprint is the DEPLOYED atoms themselves, so row 7's
            # "was this refitted between the halves" check is real for this leg
            "protos": R._FitFingerprint(list(store.modes)),
            "address_bytes": {"address_bytes": 0,
                              "what": "a ceiling has no address; charter B.3"},
            "head_bytes": {"head_bytes": 0},
            "span_audit": {"span_ranks": ranks, "n_atoms_per_well": n_atoms,
                           "n_wells": K, "content_source": CONTENT_SOURCE[arm],
                           "partition": part_name,
                           "well_sizes": [int(c.size) for c in wc],
                           "total_reachable_dim": int(sum(ranks)),
                           "n_source_rows": int(rows.size),
                           "variance_explained_per_well": store.energy},
            "declared": (
                "NOT AN ARM (charter B.3): the read is handed the true feature. "
                + ("The CONTENT is also chosen from the SCORED CORNERS' OWN ROWS, which "
                   "no deployable store can do -- this is an UNREACHABLE upper bound."
                   if CONTENT_SOURCE[arm] == "corners" else
                   "The content is the L2-optimal 524,288 B chosen from TRAINING rows "
                   "only -- the best content a store could actually be built to hold.")),
            "fit_seconds": time.time() - t0,
        }
        arm_dir.mkdir(parents=True, exist_ok=True)
        with open(done, "wb") as fh:
            pickle.dump(obj, fh)
        fw._jdump({k: v for k, v in obj.items() if k not in ("store", "basis")},
                  arm_dir / "arm.json")
        return obj

    built = build_field_for_leg(out_dir, log=log)
    field = built["field"]
    store = FieldSpanStore(field)
    audit = _assert_span_is_the_content(store, field)
    basis = span_basis(store)
    log(f"  arm {arm}: {audit['n_live_atoms']} live atoms, span ranks "
        f"{audit['span_ranks']} (sum {sum(audit['span_ranks'])})")

    obj = {
        "arm": arm,
        "read": "oracle_field" if arm == "oracle_span_field" else "field_argmax",
        "kind": "oracle",
        "protos": R._FitFingerprint(list(store.modes)),
        "store": store, "basis": basis, "field": field,
        "byte_ledger": {
            "deployed_bytes": int(built["byte_ledger"]["deployed_bytes"]),
            "budget_bytes": int(W5_PINS["store_bytes"]),
            "param_count": int(built["byte_ledger"]["deployed_bytes"]) // 4,
            "what": ("the FIELD store's atoms, the SAME 524,288 B ledger the wave-3 "
                     "store deploys; the interface segments are declared beside it and "
                     "never netted into it"),
            "interface_bytes": int(built["byte_ledger"].get("interface_bytes", 0)),
            "total_image_bytes": int(built["byte_ledger"].get("total_image_bytes", 0)),
        },
        # A ceiling has NO address book: it is handed the answer.  Zero is the honest
        # number here, and it is the reason a ceiling can never be quoted as a mechanism.
        "address_bytes": {"address_bytes": 0,
                          "what": "a ceiling has no address; charter B.3 disqualifies it"},
        "head_bytes": {"head_bytes": 0},
        "span_audit": audit,
        "field_manifest": {"sha256": field.sha256(),
                           "n_atoms": int(field.A), "n_wells": int(field.K)},
        "request_only": built["request_only"],
        "fit_seconds": time.time() - t0,
    }
    if arm == "oracle_span_field":
        obj["declared"] = ("NOT AN ARM (charter B.3): it reads the requested row's "
                           "own feature. It is the CEILING of the CONTENT dial.")
    arm_dir.mkdir(parents=True, exist_ok=True)
    with open(done, "wb") as fh:
        pickle.dump(obj, fh)
    fw._jdump({k: v for k, v in obj.items()
               if k not in ("store", "basis", "field")}, arm_dir / "arm.json")
    return obj


def arm_features(obj: dict, values: np.ndarray, grid) -> tuple:
    """Dispatch: the two field verbs here, everything else to ``exp_gen_read``."""
    verb = obj.get("read", "head")
    if verb == "true_feature":
        # NOT AN ARM (charter B.3), and not even a store: the requested row's encoder
        # feature, returned unchanged.  It is the ceiling of every possible store.
        rows, oor = R.grid_rows_of(values, grid)
        F = fw.features_mmap(int(fw.PINS["seed"]))
        return fw.gather(F, rows), oor
    if verb == "oracle_field":
        # NOT AN ARM (charter B.3).  The requested row's OWN feature, projected onto
        # the span of the field store's deployed atoms: the best answer any read of
        # these bytes could give.
        rows, oor = R.grid_rows_of(values, grid)
        F = fw.features_mmap(int(fw.PINS["seed"]))
        X = fw.gather(F, rows)
        store, basis = obj["store"], obj["basis"]
        return render_span(store, span_coefficients(store, X, basis), basis), oor
    if verb == "field_argmax":
        # charter C.1/C.4: the matched-bytes lookup over the SAME atoms, driven by the
        # store's OWN request-only address.  If this matches the ceiling, the content is
        # a lookup table and is reported as one.
        from ..core.gen_field import field_address
        field = obj["field"]
        V = np.asarray(values, np.float64)
        a = field_address(field, V, grid.values)
        addr = np.asarray(a["mu"], np.float32)
        oor = np.asarray(a["out_of_range"], bool)
        out = np.zeros((V.shape[0], int(field.D)), np.float32)
        alive = np.asarray(field.view("atom_alive"), np.uint8) > 0
        for k in range(field.K):
            ch = np.asarray(field.meta["well_channels"][k], np.int64)
            Ak = np.asarray(field.atoms_of_well(k), np.float32)[alive]
            q = addr[:, ch]
            d2 = (np.sum(q * q, 1)[:, None] - 2.0 * (q @ Ak.T)
                  + np.sum(Ak * Ak, 1)[None, :])
            out[:, ch] = Ak[np.argmin(d2, axis=1)]
        return out, oor
    return _ORIG_ARM_FEATURES(obj, values, grid)


# ---------------------------------------------------------------------------
# 4. The runtime extension of exp_gen_read.  Additive, and asserted so.
# ---------------------------------------------------------------------------

_ORIG_BUILD_ARM = R.build_arm
_ORIG_ARM_FEATURES = R.arm_features
_ORIG_ARMS = tuple(R.ARMS)


def install() -> dict:
    """Append this hub's legs to ``exp_gen_read`` without editing it.

    Asserted additive: every pre-existing arm name, its store, its verb and its class
    are compared before and after and must be unchanged.  A hub that silently moved a
    rival's or a control's code path would void its own comparison, so the check is an
    assertion and not a comment.
    """
    before = {"arms": tuple(R.ARMS),
              "store": dict(R.ARM_STORE), "read": dict(R.ARM_READ),
              "cls": dict(R.ARM_CLASS)}
    R.ARMS = tuple(_ORIG_ARMS) + FIELD_ARMS
    R.ARM_STORE.update({a: "field" for a in FIELD_ARMS})
    R.ARM_READ.update({"oracle_span_field": "oracle_field",
                       "field_lookup_argmax": "field_argmax",
                       "oracle_content_pool": "oracle_field",
                       "oracle_content_oracle": "oracle_field",
                       "oracle_content_fine": "oracle_field",
                       "oracle_content_items": "oracle_field",
                       "oracle_true_feature": "true_feature"})
    R.ARM_READ.update({a: v for a, (_k, v) in PCA_ARMS.items()})
    R.ARM_STORE.update({a: "pca" for a in PCA_ARMS})
    R.ARM_CLASS.update(FIELD_ARM_CLASS)
    R.NOT_ARMS = tuple(R.NOT_ARMS) + ("oracle_span_field", "oracle_content_pool",
                                      "oracle_content_oracle", "oracle_content_fine",
                                      "oracle_content_items", "oracle_true_feature")
    R.build_arm = build_arm
    R.arm_features = arm_features
    for a in before["arms"]:
        if (R.ARM_STORE.get(a) != before["store"].get(a)
                or R.ARM_READ.get(a) != before["read"].get(a)
                or R.ARM_CLASS.get(a) != before["cls"].get(a)):
            raise AssertionError(f"installing this hub moved pre-existing arm {a!r}")
    if tuple(R.ARMS)[:len(_ORIG_ARMS)] != _ORIG_ARMS:
        raise AssertionError("installing this hub reordered exp_gen_read.ARMS")
    return {"added": list(FIELD_ARMS), "n_arms": len(R.ARMS),
            "pre_existing_unchanged": True}


# ---------------------------------------------------------------------------
# 5. The launcher's stages -- thin wrappers over exp_gen_read's own
# ---------------------------------------------------------------------------


def _make_artifacts_portable() -> dict:
    """Make this module's pickled artifacts readable from an ORDINARY import.

    ⚠ The defect this fixes, stated because it would have cost a replicator real time.
    The launcher runs ``python -m chlu.experiments.exp_w5_compose_field``, so while a
    stage is executing, this file's classes live in ``__main__`` and every artifact the
    stage pickles records them as ``__main__.PCASpanStore``.  Resume works, because the
    next stage is another ``-m`` run -- but anyone who loads the artifacts from a plain
    ``import`` gets ``AttributeError: Can't get attribute 'PCASpanStore' on module
    '__main__'``, which is exactly what a replicator does.

    ⛔ **The obvious fix is wrong and I tried it first.** Stamping
    ``cls.__module__ = "chlu.experiments.exp_w5_compose_field"`` on the ``__main__``
    copies makes pickle IMPORT that package module to verify the class -- which creates
    a SECOND module object with its own class objects -- and then refuse:
    ``PicklingError: it's not the same object as ...._ModeStore``.  ⚠ It broke a live
    leg. The lesson is that there must be exactly ONE copy of these classes in the
    process, not two with matching names.

    So: the ``__main__`` entry point **delegates** to the package module (see the bottom
    of this file), and this function runs at that module's import and **overwrites** the
    ``__main__`` copies with the package ones.  After it, every class in the process is
    the package's, new artifacts pickle under the package path, and artifacts written
    before the fix -- which name ``__main__.PCASpanStore`` -- resolve to the same object.
    """
    import sys
    names = ("FieldSpanStore", "PCASpanStore", "_ItemSpanStore", "_ModeStore")
    g = globals()
    m = sys.modules.get("__main__")
    if __name__ != "__main__" and m is not None and m is not sys.modules.get(__name__):
        for n in names:
            setattr(m, n, g[n])
        return {"mode": "package classes installed into __main__ (one copy in the "
                        "process; pre-fix artifacts resolve to it too)",
                "classes": names}
    return {"mode": "no action needed"}


_PORTABILITY = _make_artifacts_portable()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", required=True,
                   choices=("pins", "prepare", "fit", "run", "table"))
    p.add_argument("--arm", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--condition", default="val", choices=("val", "test"))
    p.add_argument("--max-seconds", type=float, default=2700.0)
    p.add_argument("--primary", default="oracle_span_field")
    p.add_argument("--margin", type=float, default=1.0)
    a = p.parse_args(argv)

    inst = install()
    R.set_condition(a.seed, a.condition)
    _bi = R.bind_instrument(log=lambda m: print(m, flush=True))
    out = R.out_leg(Path(a.out), a.seed, a.condition)
    out.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    log(f"stage={a.stage} arm={a.arm} out={out} seed={a.seed} cond={a.condition} "
        f"install={inst}")

    if a.stage == "pins":
        fw._jdump({"w5_pins": W5_PINS, "read_pins": R.READ_PINS, "fw_pins": fw.PINS,
                   "field_pins": {k: (list(v) if isinstance(v, tuple) else v)
                                  for k, v in fs.PINS.items()},
                   "install": inst, "instrument": _bi,
                   "bridge_root": os.environ.get("CHLU_GEN_BRIDGE_ROOT"),
                   "feature_cache": R._cache_manifest(int(a.seed))}, out / "pins.json")
        print(json.dumps({"w5_pins": W5_PINS, "install": inst}, indent=1, default=str))
        return 0
    if a.stage == "prepare":
        log(json.dumps(R.stage_prepare(out, log=log)))
        return 0
    if a.stage == "fit":
        log(json.dumps(R.stage_fit(a.arm, out, log=log), default=str))
        return 0
    if a.stage == "run":
        log(json.dumps(R.stage_run(a.arm, out, a.max_seconds, log=log), default=str))
        return 0
    if a.stage == "table":
        with R.read_dial():
            t = R.build_table(out, primary=a.primary, margin=a.margin, log=log)
        for name, blk in t["arms"].items():
            r4 = (blk.get("r4_comp_nway") or {})
            r3 = (blk.get("r3_comp_2way") or {})
            log(f"    {name:22s} {blk['class']:40s} "
                f"r4={r4.get('novel_tuples_per_corner', float('nan')):7.3f} "
                f"r3={r3.get('novel_tuples_per_corner', float('nan')):7.3f}")
        return 0
    raise SystemExit(f"stage {a.stage} not implemented")


if __name__ == "__main__":
    # ⛔ Delegate to the PACKAGE module instead of running this copy.  Running the
    # file as ``python -m`` defines its classes in ``__main__``; anything pickled from
    # here would name ``__main__.PCASpanStore`` and be unreadable to a replicator who
    # loads the artifacts from an ordinary import.  Importing the package module and
    # calling ITS ``main`` means there is exactly one copy of every class in the
    # process, so artifacts pickle under the package path and stay portable.
    from chlu.experiments.exp_w5_compose_field import main as _pkg_main

    raise SystemExit(_pkg_main())
