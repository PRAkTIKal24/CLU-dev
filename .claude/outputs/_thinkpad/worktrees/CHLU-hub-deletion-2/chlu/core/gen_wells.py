"""⭐ **WHERE THE WELLS GO** — well placement learned with the store in the loop.

⭐ DIAL DECLARATION — *placement*.  Every other stage of the generative cell is held
frozen: the encoder, the feature cache, the decoder, the read.  The **only** thing this
module learns is the ``D -> K`` assignment of feature coordinates to wells — the
``labels`` vector that :mod:`chlu.core.gen_store` turns into ``well_channels``.

⛔ **The contract this module lives under is not ours to renegotiate.**  The shipped
store plan states it in its own ``note`` field:

    "Every later factorisation-ladder rung loads this file VERBATIM.  K, n_modes, D and
    store_bytes are pinned here; **only the grouping (the `labels` field) is allowed to
    change between rungs.**"

So ``K = 5``, ``n_modes = 64``, ``D = 2048``, ``store_bytes = 524,288`` are **identical in
every arm**, and :func:`assert_byte_ledger` aborts the run if they ever are not.

⚠ **Terminology, corrected here because it was wrong in the task file and would have
propagated.**  The inherited partition is **hierarchical linkage** on the channel
correlation matrix — ``gen_store.correlation_grouping`` walks ``("average", "ward")`` and
the shipped Shapes3D/``recon`` plan records ``method_used: "ward"``.  **k-means++ seeds the
MODES** (``gen_store.init_store``), ⛔ not the partition.

**What is learned and how it is delivered.**  A soft assignment ``A (D, K)`` is optimised
jointly with a *relaxed* store under a label-free objective (§3), then **hardened to a
partition whose well-size histogram is EXACTLY the reference one** (:func:`harden_to_histogram`).
Every scored number then comes from the ordinary ``init_store`` + ``train_store`` path on
that hard partition, so an arm differs from another arm **only in its label vector** —
the discipline ``exp_gen_store.run_arm`` already enforces.  ⚠ The relaxation holds
``K·M·D`` floats while it is being optimised and the delivered store holds ``M·D``; that
is a **training** cost, it is reported as one, and it is ⛔ never netted against the store
budget.

⛔ **The control is not optional and it is not a footnote.**  :func:`dmatched_random_placement`
permutes the reference label vector, holding the size histogram exactly, so the only
thing that varies is which coordinate sits in which well.  ⭐ **Every placement number is
a MARGIN OVER THAT FLOOR, never an absolute** — this programme has already been burned
here (an untrained random-init encoder scored 0.9361/0.9865 against a trained encoder's
0.935/0.991 for 1,246x less compute).

Runnable through :mod:`chlu.experiments.exp_gen_wells_route_b`.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Callable, Optional, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from .gen_encoders import realised_values
from .gen_store import (
    GenStore,
    mode_usage,
    random_grouping,
    store_bytes,
    well_channels_from_labels,
)

#: ⛔ The arms, declared in code so neither can be swapped in later.  ``learned_weak_sup``
#: is a DECLARED arm: it is reported as a separate row and ⛔ **may never be quoted as the
#: label-free result**.  If the label-free arm fails and this one works, ⭐ *that is the
#: finding*, and it is stated in those words.
ARMS = ("random_dmatched", "learned_label_free", "learned_weak_sup", "correlation")

#: ⭐ The laundering control, named once so no caller can quietly pick another.  It runs
#: FIRST (`exp_gen_wells_route_b` orders the arms by this), because a floor measured after
#: the thing it is the floor for is a floor that can be talked about.
FLOOR_ARM = "random_dmatched"

#: Arms whose objective is label-free.  :func:`assert_label_free` must PASS on these and
#: ⛔ must FAIL on every other learned arm — an assertion that has never fired is not one.
LABEL_FREE_ARMS = ("learned_label_free",)

__all__ = [
    "ARMS",
    "FLOOR_ARM",
    "LABEL_FREE_ARMS",
    "CheckpointCfg",
    "LabelLeakError",
    "PlacementState",
    "address_sharpness",
    "assert_byte_ledger",
    "assert_cache_coverage",
    "assert_label_free",
    "assert_partition",
    "collapse_verdict",
    "config_fingerprint",
    "dmatched_random_placement",
    "harden_to_histogram",
    "labels_from_placement",
    "latest_checkpoint",
    "list_checkpoints",
    "load_checkpoint",
    "placement_loss",
    "placement_margin",
    "save_checkpoint",
    "sharpness_table",
    "soft_memberships",
    "train_placement",
    "weak_supervision_term",
]


class LabelLeakError(AssertionError):
    """Raised when a path declared label-free is shown not to be."""


# ---------------------------------------------------------------------------
# 1. Placement — the learned object, its floor, and the hardening
# ---------------------------------------------------------------------------


class PlacementState(eqx.Module):
    """The relaxation: soft assignment logits ``A (D, K)`` + a relaxed store.

    ``modes`` is ``(K, M, D)`` — full width, because a *soft* membership has no fixed
    ``d_k`` to live in.  ⛔ This is the training relaxation and **not** the delivered
    store; the delivered store is a plain :class:`chlu.core.gen_store.GenStore` over the
    hardened partition and is byte-identical to every other arm's.
    """

    logits: jnp.ndarray
    modes: jnp.ndarray

    def __init__(self, logits, modes):
        self.logits = jnp.asarray(logits, jnp.float32)
        self.modes = jnp.asarray(modes, jnp.float32)

    @property
    def dim(self) -> int:
        return int(self.logits.shape[0])

    @property
    def n_wells(self) -> int:
        return int(self.logits.shape[1])

    @property
    def n_modes(self) -> int:
        return int(self.modes.shape[1])


def soft_memberships(logits) -> jnp.ndarray:
    """``(D, K)`` logits -> ``(D, K)`` memberships with ``Σ_k a[d,k] = 1``.

    A coordinate is *distributed* over wells during training and belongs to exactly one
    after :func:`harden_to_histogram`.  The row-stochastic constraint is what makes the
    relaxation a relaxation *of a partition* rather than of an arbitrary mask.
    """
    return jax.nn.softmax(jnp.asarray(logits, jnp.float32), axis=-1)


def dmatched_random_placement(reference_labels: np.ndarray, seed: int) -> np.ndarray:
    """⭐⭐ **THE LAUNDERING CONTROL.**  Same ``K``, **same well-size histogram**, shuffled.

    Delegates to :func:`chlu.core.gen_store.random_grouping` verbatim — ⛔ deliberately
    not a second implementation of the same control.  A random partition drawn
    independently would differ from the reference in *two* ways (assignment and size
    profile) and any margin would be unattributable; permuting the reference label vector
    holds the histogram exactly, so the only thing that varies is which coordinate sits in
    which well.  That is what "D-matched" means and it is why the bytes match exactly.
    """
    return random_grouping(np.asarray(reference_labels), int(seed))


def harden_to_histogram(
    memberships: np.ndarray, target_sizes: Sequence[int]
) -> np.ndarray:
    """``(D, K)`` soft memberships -> hard labels realising ``target_sizes`` **exactly**.

    ⭐ **Why the sizes are constrained rather than read off.**  An unconstrained ``argmax``
    lets the learned arm drift to a different well-size histogram, and then it is no longer
    D-matched to the random floor: a margin could be bought with a size profile rather than
    with placement, and ⛔ the byte ledger would be renegotiated per seed.  Constraining the
    histogram makes the four arms differ in *exactly one* thing.

    **The rule (deterministic, and it is a rule, not a search).**  Take the ``(d, k)`` pairs
    in decreasing membership and assign ``d -> k`` whenever ``d`` is still unplaced and
    ``k`` is not yet full.  Ties break by ``(-a, d, k)``, so the result is a pure function
    of the inputs.  Every coordinate is placed because ``Σ_k target_sizes[k] == D``.
    """
    A = np.asarray(memberships, np.float64)
    d, k = A.shape
    tgt = np.asarray([int(x) for x in target_sizes], np.int64)
    if tgt.size != k:
        raise ValueError(f"target_sizes has {tgt.size} entries but there are {k} wells")
    if int(tgt.sum()) != d:
        raise ValueError(
            f"⛔ the target sizes must partition the {d} coordinates (they sum to "
            f"{int(tgt.sum())}) — a drifting histogram silently renegotiates the byte budget"
        )
    # lexsort is stable and gives the deterministic (-a, d, k) order without building
    # a Python sort key over D*K tuples.
    dd, kk = np.meshgrid(np.arange(d), np.arange(k), indexing="ij")
    flat_a, flat_d, flat_k = A.ravel(), dd.ravel(), kk.ravel()
    order = np.lexsort((flat_k, flat_d, -flat_a))
    labels = np.full(d, -1, np.int64)
    room = tgt.copy()
    placed = 0
    for idx in order:
        c, w = int(flat_d[idx]), int(flat_k[idx])
        if labels[c] >= 0 or room[w] <= 0:
            continue
        labels[c] = w
        room[w] -= 1
        placed += 1
        if placed == d:
            break
    if int((labels < 0).sum()):
        raise AssertionError(
            f"{int((labels < 0).sum())} coordinate(s) were left unplaced — this cannot "
            "happen when the targets sum to D and indicates a corrupted membership matrix"
        )
    return labels


def labels_from_placement(state: PlacementState, target_sizes: Sequence[int]) -> np.ndarray:
    """The delivered ``labels`` vector for a trained relaxation."""
    return harden_to_histogram(np.asarray(soft_memberships(state.logits)), target_sizes)


def assert_partition(labels: np.ndarray, d: int, k: int) -> dict:
    """⛔ The wells must PARTITION the ``D`` coordinates.  Raises, never warns."""
    lab = np.asarray(labels, np.int64)
    if lab.size != int(d):
        raise AssertionError(f"labels has {lab.size} entries, not D = {int(d)}")
    if lab.min() < 0 or lab.max() >= int(k):
        raise AssertionError(f"labels out of range [0, {int(k)}): [{lab.min()}, {lab.max()}]")
    sizes = np.bincount(lab, minlength=int(k))
    if int((sizes == 0).sum()):
        raise AssertionError(
            f"⛔ {int((sizes == 0).sum())} EMPTY well(s) in {sizes.tolist()} — a degenerate "
            "partition is reported, never silently merged away"
        )
    return {"well_sizes": [int(x) for x in sizes], "n_wells": int(k), "dim": int(d)}


# ---------------------------------------------------------------------------
# 2. ⛔ The byte ledger — asserted in code; the run ABORTS on a mismatch
# ---------------------------------------------------------------------------


def assert_byte_ledger(
    labels: np.ndarray,
    n_modes: int,
    expected_store_bytes: int,
    *,
    k: int,
    dim: int,
    rank: Optional[int] = None,
    arm: str = "",
    enforce: bool = True,
) -> dict:
    """⭐ ``Σ_k M·d_k·4 = M·D·4`` for **any** partition of ``D`` — so a mismatch is a
    structural fault, not a rounding difference, and the run stops.

    The coupling identity ``Σ_{k<l} r(d_k+d_l)·4 = r(K−1)D·4`` is checked alongside when a
    ``rank`` is given: it too holds for any partition, so ⛔ **a per-seed drift in well
    sizes may not silently renegotiate the budget.**
    """
    info = assert_partition(labels, dim, k)
    sizes = info["well_sizes"]
    m = int(n_modes)
    per_well = [m * int(s) * 4 for s in sizes]
    total = int(sum(per_well))
    closed_form = int(m * int(dim) * 4)
    led = {
        "arm": str(arm),
        "well_sizes": sizes,
        "n_modes": m,
        "dim": int(dim),
        "n_wells": int(k),
        "store_bytes": total,
        "store_bytes_per_well": per_well,
        "store_bytes_closed_form": closed_form,
        "expected_store_bytes": int(expected_store_bytes),
        "bytes_formula": "sum_k M*d_k*4 = M*D*4",
        "matches_expected": bool(total == int(expected_store_bytes)),
        "matches_closed_form": bool(total == closed_form),
        "bits_per_read": float(int(k) * np.log2(float(m))),
    }
    if rank is not None:
        r = int(rank)
        coup = sum(r * (sizes[a] + sizes[b]) * 4
                   for a in range(len(sizes)) for b in range(a + 1, len(sizes)))
        led["coupling"] = {
            "rank": r,
            "coupling_bytes": int(coup),
            "coupling_bytes_closed_form": int(r * (int(k) - 1) * int(dim) * 4),
            "identity": "sum_{k<l} r*(d_k+d_l)*4 = r*(K-1)*D*4 when the wells partition D",
            "matches_closed_form": bool(int(coup) == int(r * (int(k) - 1) * int(dim) * 4)),
        }
    ok = led["matches_expected"] and led["matches_closed_form"] and (
        rank is None or led["coupling"]["matches_closed_form"])
    led["ok"] = bool(ok)
    if enforce and not ok:
        raise AssertionError(
            f"⛔ BYTE LEDGER MISMATCH on arm {arm!r}: got {total} B "
            f"(closed form {closed_form} B, expected {int(expected_store_bytes)} B) from "
            f"well sizes {sizes}. Learned placement must cost what inherited placement "
            "costs; the run stops rather than renegotiate the budget."
        )
    return led


def store_byte_ledger_matches(store: GenStore, expected_store_bytes: int) -> dict:
    """The same gate applied to a **built** store — closes the loop on what shipped."""
    sb = store_bytes(store)
    sb["expected_store_bytes"] = int(expected_store_bytes)
    sb["matches_expected"] = bool(int(sb["store_bytes"]) == int(expected_store_bytes))
    return sb


# ---------------------------------------------------------------------------
# 3. ⛔ Preconditions that fail at STARTUP or not at all
# ---------------------------------------------------------------------------


def assert_cache_coverage(
    cache_rows: np.ndarray,
    needed_rows: np.ndarray,
    *,
    seed: int,
    scope: str = "",
    allow_reencode: bool = False,
    label: str = "",
) -> dict:
    """⛔⭐ **Assert the cache covers the rows this run needs — at job start.**

    ⚠ Measured on the staged set, not assumed: seed 0's feature cache holds **480,000**
    rows (``scope="full"``, 1.83 GiB of the 1.95 GiB staged) while seeds 1 and 2 hold
    **9,288 each** (``scope="used"``), and ``encoder_param_hash`` **DIFFERS per seed** —
    they are not interchangeable.  A run on seed 1 or 2 that asks for rows outside those
    9,288 falls into ``gen_store.gather_features``' extend-by-re-encode path, which
    ``gen-dynamical-read`` flagged **UNTESTED** — and on a cluster it would discover that
    **on the GPU, after the scheduler has already allocated it.**

    ⭐ So this names the seed and the shortfall and **dies immediately**.  ⚠ If the
    re-encode path is being taken deliberately, pass ``allow_reencode=True``: the run then
    reports an **exercised untested path**, ⛔ not a normal run.
    """
    have = np.asarray(cache_rows, np.int64)
    want = np.unique(np.asarray(needed_rows, np.int64))
    pos = np.searchsorted(have, want)
    safe = np.minimum(pos, max(have.size - 1, 0))
    hit = (pos < have.size) & (have[safe] == want) if have.size else np.zeros(want.size, bool)
    missing = want[~hit]
    rep = {
        "seed": int(seed),
        "label": str(label),
        "cache_scope": str(scope),
        "n_cache_rows": int(have.size),
        "n_rows_needed": int(want.size),
        "n_covered": int(hit.sum()),
        "n_shortfall": int(missing.size),
        "first_missing_rows": [int(x) for x in missing[:8]],
        "covered": bool(missing.size == 0),
        "allow_reencode": bool(allow_reencode),
        "exercised_untested_path": bool(missing.size > 0 and allow_reencode),
    }
    if missing.size and not allow_reencode:
        raise AssertionError(
            f"⛔ FEATURE-CACHE SHORTFALL on SEED {int(seed)} ({label or 'rows'}): the cache "
            f"holds {have.size} rows (scope={scope!r}) but this run needs {want.size}, of "
            f"which {missing.size} are NOT cached (first: "
            f"{[int(x) for x in missing[:5]]}).\n"
            "    Those rows would enter gather_features' extend-by-re-encode path, which is "
            "UNTESTED (gen-dynamical-read) — and on the cluster this would be discovered on "
            "the GPU, after the scheduler had already allocated it.\n"
            "    ⭐ Seed 0 is the only seed cached at scope='full' (480,000 rows). Either run "
            "seed 0, reduce n_store_train below the cached coverage, or pass "
            "--allow-reencode to take the untested path DELIBERATELY (it is then reported as "
            "an exercised untested path, not as a normal run)."
        )
    return rep


# ---------------------------------------------------------------------------
# 4. ⭐ The objective — label-free primary, and the declared weakly-supervised arm
# ---------------------------------------------------------------------------


def _soft_occupancy(d2, tau: float, eps: float = 1e-8):
    """``(N, K, M)`` squared distances -> soft occupancy at temperature ``tau``, ⭐ measured
    in units of **each row's own mean distance to that well's modes**.

    ⛔⭐ **This normalisation is not cosmetic, and the defect it fixes is on the record in
    this repo already.**  :func:`chlu.core.gen_dream.mode_distance_scale` says it outright:
    *"``exp(-d²/2τ²)`` is saturated one-hot whenever ``tau`` is small against this scale, and
    a saturated soft occupancy has vanishing gradient … Specifying ``tau`` in absolute units
    … is therefore a sampler defect, and it is the one that froze this module's first run."*

    ⚠ It froze this one too, and worse.  With an ABSOLUTE ``tau`` the first end-to-end run
    of this harness saturated by step 60 and **the occupancy entropy went to exactly
    0.0000** — every row picking one mode per well — where it stayed for the rest of
    training.  Both regularisers are **inert at that point** (the log showed
    ``loss == rec`` to five decimals), so the collapse is a *fixed point* of the absolute-
    ``tau`` objective: a collapsed store cannot climb back out.
    ⛔ That is the pre-registered collapse falsifier (`PREREG.md` §7) arriving as an
    optimiser artefact rather than as a property of placement.

    ⭐ Dividing by the row's own mean distance makes ``tau`` scale-free in ``D``, in ``M``
    and in the feature magnitude, so ``tau = 1`` means "genuinely soft" at every well and at
    every step, however far the modes have drifted.
    """
    # ⚠ ``tau`` is kept as a JAX scalar, ⛔ never coerced with ``float()``: it is annealed
    # per step and ``eqx.filter_jit`` treats a PYTHON float as **static**, so a float here
    # recompiles the whole training step on EVERY iteration.  Measured on this harness: it
    # is the difference between ~0.05 s and several seconds per step, and it is silent —
    # the run simply looks slow.  (⚠ ``gen_store.train_store`` still has this shape.)
    scale = jnp.mean(d2, axis=-1, keepdims=True) + eps      # (N, K, 1)
    return jax.nn.softmax(-d2 / (2.0 * jnp.asarray(tau) ** 2 * scale), axis=-1)


def _weighted_sq_dists(x, modes, a):
    """``(N, K, M)`` membership-weighted squared distances.

    ``d²_k(n,m) = Σ_d a[d,k] (x[n,d] − modes[k,m,d])²`` — the soft generalisation of the
    hard per-well gather.  At a hard ``a`` it reduces **exactly** to the distance
    ``gen_store.soft_weights`` computes on the well's own coordinates, which is what makes
    the relaxation and the delivered store the same objective.

    Expanded to ``Σ a·x² − 2 Σ a·x·m + Σ a·m²`` so the ``(N, K, M, D)`` broadcast is never
    materialised: at ``D = 2048``, ``K = 5``, ``M = 64`` and a 1,024-row batch that tensor
    is ~2.7 GB in float32 and ~4x that under reverse-mode AD.  ⛔ The run dies on a laptop
    and on an 80 GB card alike.
    """
    x2 = jnp.einsum("nd,dk->nk", x * x, a)                  # (N, K)
    xm = jnp.einsum("nd,dk,kmd->nkm", x, a, modes)          # (N, K, M)
    m2 = jnp.einsum("dk,kmd->km", a, modes * modes)         # (K, M)
    return x2[:, :, None] - 2.0 * xm + m2[None, :, :]


def placement_loss(
    state: PlacementState,
    x,
    tau: float,
    *,
    lambda_occ: float,
    lambda_fact: float,
    eps: float = 1e-8,
) -> tuple:
    """⛔⭐ **THE LABEL-FREE OBJECTIVE.**  ``(loss, parts)``.

    ``L = L_rec + λ_occ·L_occ + λ_fact·L_fact``, and **every one of the three is a
    function of the features and the store alone**:

    1. ``L_rec`` — **within-well fit**: ``Σ_k mean_n ‖z_k − π_k M_k‖²`` under the
       membership-weighted distance, ``π = softmax(−d²/2τ²)`` annealed ``τ_start → τ_end``
       so the soft read the gradient sees converges onto the hard read the metric uses.
    2. ``L_occ`` — **within-well MULTIMODALITY**: ``−Σ_k H(mean_n π_k)``.  A well whose
       mass sits on one mode is not multimodal, it is collapsed.  ⭐ This term is the
       direct counter-pressure to the collapse falsifier and it is why collapse is a
       *reported verdict* here rather than an accident.
    3. ``L_fact`` — **between-well FACTORISATION**: ``Σ_{k<l} I(π_k ; π_l)`` from the batch
       joint ``P̄(m,m') = mean_n π_k[n,m]·π_l[n,m']``.  Minimising it pushes the wells'
       codes towards independence — which is the whole claim "the wells are semantic" is
       standing on.

    ⛔ **No factor label enters any term, and the signature admits none.** That is checked
    structurally *and* numerically by :func:`assert_label_free`, which ⛔ must also FAIL on
    the weakly-supervised arm — an assertion that has never fired is not an assertion.
    """
    a = soft_memberships(state.logits)                      # (D, K)
    d2 = _weighted_sq_dists(jnp.asarray(x, jnp.float32), state.modes, a)
    pi = _soft_occupancy(d2, tau, eps)                      # (N, K, M)

    # 1. within-well fit.  ‖z_k − π M_k‖²_a expanded the same way as d² above.
    recon = jnp.einsum("nkm,kmd->nkd", pi, state.modes)      # (N, K, D)
    xr = jnp.asarray(x, jnp.float32)[:, None, :] - recon
    l_rec = jnp.mean(jnp.sum(jnp.einsum("nkd,dk->nkd", xr * xr, a), axis=-1))

    # 2. within-well multimodality: occupancy entropy, maximised.
    occ = jnp.mean(pi, axis=0)                              # (K, M)
    h_occ = -jnp.sum(occ * jnp.log(occ + eps), axis=-1)     # (K,)
    l_occ = -jnp.mean(h_occ)

    # 3. between-well factorisation: pairwise MI of the soft codes.
    n = pi.shape[0]
    joint = jnp.einsum("nkm,nlj->kmlj", pi, pi) / jnp.asarray(n, jnp.float32)
    marg = occ                                              # (K, M)
    outer = jnp.einsum("km,lj->kmlj", marg, marg)
    mi_full = jnp.sum(joint * (jnp.log(joint + eps) - jnp.log(outer + eps)), axis=(1, 3))
    k = state.n_wells
    pair = jnp.triu(jnp.ones((k, k), jnp.float32), 1)
    n_pairs = jnp.maximum(jnp.sum(pair), 1.0)
    l_fact = jnp.sum(mi_full * pair) / n_pairs

    loss = l_rec + float(lambda_occ) * l_occ + float(lambda_fact) * l_fact
    return loss, {
        "l_rec": l_rec,
        "l_occ": l_occ,
        "l_fact": l_fact,
        "mean_occupancy_entropy_nats": jnp.mean(h_occ),
        "mean_pairwise_mi_nats": l_fact,
    }


def weak_supervision_term(state: PlacementState, x, y, tau: float,
                          cardinalities: Optional[Sequence[int]] = None,
                          eps: float = 1e-8):
    """⛔ **THE DECLARED WEAKLY-SUPERVISED TERM — labels ENTER HERE, on purpose.**

    ``−mean_f max_k I(π_k ; f)``: reward each factor for being predictable from *one*
    well's code.  ⛔ This arm is reported as a **separate row** and ⛔ **may never be
    quoted as the label-free result.**  It exists for two reasons: it is a declared arm of
    the task, and it is the **leaky variant that makes** :func:`assert_label_free`
    **fire** — a check that has never failed on anything is not evidence.

    ⚠ ``cardinalities`` must be supplied **whenever this is called under ``jit``**: the
    one-hot width is a *static* shape, and inferring it with ``y.max()`` reads a **traced**
    array, which raises inside a jitted step.  ⛔ That is not a hypothetical — it is how
    this function first shipped, and the weakly-supervised arm died on it the moment the
    jitted step was reached (the un-jitted label-free gate had passed happily just above).
    """
    a = soft_memberships(state.logits)
    d2 = _weighted_sq_dists(jnp.asarray(x, jnp.float32), state.modes, a)
    pi = _soft_occupancy(d2, tau, eps)                               # (N, K, M)
    Y = jnp.asarray(y, jnp.int32)
    n = pi.shape[0]
    occ = jnp.mean(pi, axis=0)                                       # (K, M)
    if cardinalities is None:
        # ⚠ only reachable OUTSIDE jit (the gate, a test); under jit this would read a
        # tracer.  Kept so the un-jitted callers need not thread a static shape.
        cards = [int(np.asarray(y)[:, f].max()) + 1 for f in range(int(Y.shape[1]))]
    else:
        cards = [int(v) for v in cardinalities]
    per_factor = []
    for f in range(Y.shape[1]):
        v = int(cards[f])
        onehot = jax.nn.one_hot(Y[:, f], v)                          # (N, V)
        joint = jnp.einsum("nkm,nv->kmv", pi, onehot) / jnp.asarray(n, jnp.float32)
        pv = jnp.mean(onehot, axis=0)                                # (V,)
        outer = jnp.einsum("km,v->kmv", occ, pv)
        mi = jnp.sum(joint * (jnp.log(joint + eps) - jnp.log(outer + eps)), axis=(1, 2))
        per_factor.append(jnp.max(mi))
    return -jnp.mean(jnp.stack(per_factor))


def assert_label_free(
    loss_fn: Callable,
    state: PlacementState,
    x,
    labels: np.ndarray,
    *,
    tau: float,
    seed: int = 0,
    arm: str = "",
    expect_label_free: bool = True,
) -> dict:
    """⭐⭐ Prove (or disprove) that a placement objective is a function of the features
    and the store **alone**.

    Two independent checks, because either alone is weak:

    1. **Structural** — the objective's signature must not admit a labels argument.  A
       future edit that adds one is caught here rather than at cell time.
    2. **Numerical** — evaluate loss **and gradients** on **permuted factor labels**.  A
       label-free objective is **bit-identical**; if a label leaked through the partition,
       the bridge or the atom selection, they would move.

    ⛔ ``expect_label_free=False`` **inverts the gate**: the objective must then be shown to
    MOVE under the permutation.  That is how the weakly-supervised arm demonstrates this
    check has teeth — ⛔ an assertion that has never fired is not an assertion.
    """
    import inspect

    params = list(inspect.signature(loss_fn).parameters)
    label_like = [p for p in params if p in ("y", "labels", "targets", "factors")]
    structural_clean = not label_like

    rng = np.random.default_rng(int(seed) * 977 + 20_260_915)
    Y = np.asarray(labels, np.int64)
    Yp = Y[rng.permutation(Y.shape[0])]

    def _value_and_grad(yy):
        g = eqx.filter_value_and_grad(
            lambda st: loss_fn(st, x, yy, tau) if label_like else loss_fn(st, x, tau)
        )
        v, gr = g(state)
        leaves = [np.asarray(t) for t in jax.tree_util.tree_leaves(
            eqx.filter(gr, eqx.is_inexact_array))]
        return float(v), leaves

    v0, g0 = _value_and_grad(Y)
    v1, g1 = _value_and_grad(Yp)
    dv = abs(v0 - v1)
    dg = max((float(np.max(np.abs(p - q))) for p, q in zip(g0, g1, strict=True)), default=0.0)
    identical = bool(dv == 0.0 and dg == 0.0)

    rep = {
        "arm": str(arm),
        "expect_label_free": bool(expect_label_free),
        "structural_clean": bool(structural_clean),
        "label_like_parameters": label_like,
        "loss_on_true_labels": v0,
        "loss_on_permuted_labels": v1,
        "abs_loss_delta": dv,
        "max_abs_grad_delta": dg,
        "bit_identical_under_label_permutation": identical,
        "verdict": None,
    }
    if expect_label_free:
        rep["verdict"] = "LABEL-FREE" if (identical and structural_clean) else "LEAK"
        if not identical or not structural_clean:
            raise LabelLeakError(
                f"⛔ arm {arm!r} is declared LABEL-FREE but is not: structural_clean="
                f"{structural_clean} (label-like params {label_like}), "
                f"|Δloss| = {dv}, max|Δgrad| = {dg} under a label permutation. "
                "A label is reaching the objective through the partition, the bridge or "
                "the atom selection."
            )
    else:
        rep["verdict"] = "LEAKY-AS-DECLARED" if not identical else "CHECK-HAS-NO-TEETH"
        if identical:
            raise AssertionError(
                f"⛔ arm {arm!r} is the DELIBERATELY leaky variant, but the label-free "
                "check did not fire on it (loss and gradients were bit-identical under a "
                "label permutation). The check therefore proves nothing about the "
                "label-free arm either — fix the check, do not proceed."
            )
    return rep


# ---------------------------------------------------------------------------
# 5. ⭐ Training the relaxation — crash-resumable, and the resume is EXACT
# ---------------------------------------------------------------------------


class CheckpointCfg(eqx.Module):
    """Where checkpoints go and how often.  ⛔ ``~/scratch``, never ``$TMPDIR``, never ``$HOME``.

    ``$TMPDIR`` is node-local NVMe and is **wiped at job end**, so a checkpoint there
    protects nothing; ``$HOME`` is small, quota'd and backed up and is not a place to write
    a checkpoint every few minutes.
    """

    dir: str = eqx.field(static=True)
    every_seconds: float = eqx.field(static=True, default=300.0)
    every_steps: int = eqx.field(static=True, default=0)
    keep_last: int = eqx.field(static=True, default=3)

    def __init__(self, dir, every_seconds=300.0, every_steps=0, keep_last=3):
        self.dir = str(dir)
        self.every_seconds = float(every_seconds)
        self.every_steps = int(every_steps)
        self.keep_last = int(keep_last)


def config_fingerprint(payload: dict) -> str:
    """The config identity a resume must match.  ⛔ A different config is REFUSED."""
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _ckpt_stem(dir: Path, tag: str, step: int) -> Path:
    return Path(dir) / f"ckpt_{tag}_step{int(step):08d}"


def list_checkpoints(dir, tag: str) -> list:
    """Steps of the **good** checkpoints, ascending.

    ⭐ A checkpoint counts as good only when **both** its ``.eqx`` and its ``.json`` exist:
    the JSON is written second and is therefore the commit marker.  ⛔ A crash *during* a
    checkpoint write leaves a ``.eqx`` with no ``.json`` and is correctly ignored, so the
    last good checkpoint is never corrupted by the one that failed.
    """
    d = Path(dir)
    if not d.is_dir():
        return []
    out = []
    for j in sorted(d.glob(f"ckpt_{tag}_step*.json")):
        if j.with_suffix(".eqx").exists():
            try:
                out.append(int(j.stem.rsplit("step", 1)[1]))
            except (IndexError, ValueError):
                continue
    return sorted(out)


def latest_checkpoint(dir, tag: str) -> Optional[int]:
    steps = list_checkpoints(dir, tag)
    return steps[-1] if steps else None


def save_checkpoint(cfg: CheckpointCfg, tag: str, step: int, state, opt_state, meta: dict) -> Path:
    """⭐ Atomic (**temp file + ``os.replace``**), then rotate to ``keep_last``.

    ⛔ A crash during a checkpoint write must not corrupt the last good one, so nothing is
    ever written in place: the payload goes to ``.eqx.tmp`` and is renamed, and only then
    does the ``.json`` commit marker appear.
    """
    d = Path(cfg.dir)
    d.mkdir(parents=True, exist_ok=True)
    stem = _ckpt_stem(d, tag, step)
    eqx_tmp = stem.with_suffix(".eqx.tmp")
    eqx.tree_serialise_leaves(eqx_tmp, (state, opt_state))
    os.replace(eqx_tmp, stem.with_suffix(".eqx"))
    rec = dict(meta)
    rec["step"] = int(step)
    rec["written_at"] = time.time()
    json_tmp = stem.with_suffix(".json.tmp")
    json_tmp.write_text(json.dumps(rec, sort_keys=True, indent=2, default=str))
    os.replace(json_tmp, stem.with_suffix(".json"))
    keep = int(cfg.keep_last)
    if keep > 0:
        for old in list_checkpoints(d, tag)[:-keep]:
            s = _ckpt_stem(d, tag, old)
            s.with_suffix(".json").unlink(missing_ok=True)
            s.with_suffix(".eqx").unlink(missing_ok=True)
    return stem.with_suffix(".eqx")


def load_checkpoint(cfg: CheckpointCfg, tag: str, like_state, like_opt_state,
                    expect_fingerprint: str, log: Callable = print) -> Optional[tuple]:
    """``(state, opt_state, meta)`` from the newest good checkpoint, or ``None``.

    ⛔ **Refuses a checkpoint written under a different config.**  A resumed leg must be the
    SAME leg; silently continuing a different one would produce a run that no flag table
    describes.  ⭐ The recovery is **instrumented**: what was recovered, and from which step.
    """
    step = latest_checkpoint(cfg.dir, tag)
    if step is None:
        log(f"[resume] no checkpoint for {tag!r} in {cfg.dir} — starting from scratch")
        return None
    stem = _ckpt_stem(Path(cfg.dir), tag, step)
    meta = json.loads(stem.with_suffix(".json").read_text())
    got = str(meta.get("config_fingerprint", ""))
    if got != str(expect_fingerprint):
        raise SystemExit(
            f"⛔ refusing to resume {tag!r} from {stem.name}: it was written under a "
            f"DIFFERENT config (fingerprint {got[:12]}… vs {str(expect_fingerprint)[:12]}…).\n"
            "    A resumed leg must be the SAME leg. Give the other leg its own --ckpt-dir, "
            "or delete these checkpoints to start over."
        )
    state, opt_state = eqx.tree_deserialise_leaves(
        stem.with_suffix(".eqx"), (like_state, like_opt_state))
    log(f"[resume] ⭐ RECOVERED {tag!r} from {stem.name}: resuming at step {step} of "
        f"{meta.get('total_steps', '?')} (config fingerprint {got[:12]}… matches; "
        f"loss at checkpoint {meta.get('loss')}; {len(list_checkpoints(cfg.dir, tag))} "
        "good checkpoint(s) on disk)")
    return state, opt_state, meta


def train_placement(
    state: PlacementState,
    x: np.ndarray,
    *,
    steps: int,
    batch: int,
    lr: float,
    tau_start: float,
    tau_end: float,
    lambda_occ: float,
    lambda_fact: float,
    base_key,
    arm: str,
    logit_lr_mult: float = 1.0,
    cardinalities: Optional[Sequence[int]] = None,
    labels: Optional[np.ndarray] = None,
    lambda_sup: float = 0.0,
    ckpt: Optional[CheckpointCfg] = None,
    config_fp: str = "",
    resume: bool = False,
    log: Callable = print,
) -> tuple:
    """Fit the placement (and the relaxed store) with the store in the loop.

    ⭐ **The resume is EXACT, and here is why.**  The batch for step ``i`` is drawn from
    ``fold_in(base_key, i)`` — a pure function of ``(base_key, i)`` and ⛔ **not** of a key
    carried through the loop.  A run resumed at step ``i`` is therefore handed the
    identical batches in the identical order, so a resumed run and an uninterrupted one
    agree **bitwise**.  ⛔ Carrying the key would make the resume approximate and nobody
    would notice until the numbers disagreed.

    ⚠ ``labels``/``lambda_sup`` are used **only** by the declared weakly-supervised arm.
    For every label-free arm ``labels`` is ⛔ not touched — and that is not a promise, it is
    checked by :func:`assert_label_free`.
    """
    flat = jnp.asarray(np.asarray(x, np.float32).reshape((np.asarray(x).shape[0], -1)))
    n = int(flat.shape[0])
    bs = int(min(int(batch), n))
    total = int(steps)
    # ⭐ The logits get their OWN step size, and that is not a nicety.  The relaxation
    # carries K·M·D mode floats against D·K placement logits, and at a shared Adam rate
    # the modes absorb the gradient while the placement does not move at all: the first
    # smoke of this harness hardened with **0 of 2048 coordinates moved**, which looks
    # exactly like the finding "placement is not the lever" and is in fact an
    # optimisation artefact.  ⛔ A null must not be manufacturable by a learning rate.
    params = eqx.filter(state, eqx.is_inexact_array)
    _labels = jax.tree_util.tree_map(lambda _: "modes", params)
    _labels = eqx.tree_at(lambda t: t.logits, _labels, replace="logits")
    opt = optax.multi_transform(
        {"logits": optax.adam(float(lr) * float(logit_lr_mult)),
         "modes": optax.adam(float(lr))}, _labels)
    opt_state = opt.init(params)
    Y = None if labels is None else jnp.asarray(np.asarray(labels, np.int64))
    supervised = bool(lambda_sup) and Y is not None

    start = 0
    history: list = []
    if resume and ckpt is not None:
        got = load_checkpoint(ckpt, arm, state, opt_state, config_fp, log=log)
        if got is not None:
            state, opt_state, meta = got
            start = int(meta["step"])
            history = list(meta.get("history") or [])
    if start >= total:
        log(f"[resume] {arm!r}: checkpoint is at step {start} >= {total} — nothing to do")
        return state, history, {"steps_run": 0, "resumed_from": start, "total_steps": total}

    def _tau(i):
        return float(tau_start) * (float(tau_end) / float(tau_start)) ** (
            i / max(1, total - 1))

    def loss_fn(st, xb, tau):
        loss, parts = placement_loss(
            st, xb, tau, lambda_occ=lambda_occ, lambda_fact=lambda_fact)
        return loss, parts

    def loss_fn_sup(st, xb, yb, tau):
        loss, parts = placement_loss(
            st, xb, tau, lambda_occ=lambda_occ, lambda_fact=lambda_fact)
        sup = weak_supervision_term(st, xb, yb, tau, cardinalities=cardinalities)
        return loss + float(lambda_sup) * sup, parts

    fn = loss_fn_sup if supervised else loss_fn
    grad_fn = eqx.filter_value_and_grad(
        (lambda st, xb, yb, tau: fn(st, xb, yb, tau)) if supervised
        else (lambda st, xb, tau: fn(st, xb, tau)), has_aux=True)

    @eqx.filter_jit
    def step_sup(st, s, xb, yb, tau):
        (loss, parts), grads = grad_fn(st, xb, yb, tau)
        updates, s = opt.update(grads, s, st)
        return eqx.apply_updates(st, updates), s, loss, parts

    @eqx.filter_jit
    def step_plain(st, s, xb, tau):
        (loss, parts), grads = grad_fn(st, xb, tau)
        updates, s = opt.update(grads, s, st)
        return eqx.apply_updates(st, updates), s, loss, parts

    last_ckpt = time.time()
    t0 = time.time()
    n_ckpt = 0
    for i in range(start, total):
        # ⭐ pure function of (base_key, i): this is the whole resume guarantee.
        sk = jax.random.fold_in(base_key, i)
        idx = jax.random.choice(sk, n, (bs,), replace=False)
        # ⭐ a JAX scalar, so the jitted step is compiled ONCE rather than once per step.
        tau = jnp.asarray(_tau(i), jnp.float32)
        if supervised:
            state, opt_state, loss, parts = step_sup(
                state, opt_state, flat[idx], Y[idx], tau)
        else:
            state, opt_state, loss, parts = step_plain(state, opt_state, flat[idx], tau)
        history.append(float(loss))
        if i == start or (i + 1) % max(1, total // 5) == 0:
            log(f"    [{arm}] step {i + 1}/{total}  loss={history[-1]:.5f}  "
                f"tau={float(tau):.3f}  "
                f"rec={float(parts['l_rec']):.5f}  occ_H={float(parts['mean_occupancy_entropy_nats']):.4f}  "
                f"MI={float(parts['mean_pairwise_mi_nats']):.5f}")
        if ckpt is not None:
            due_t = (time.time() - last_ckpt) >= float(ckpt.every_seconds)
            due_s = ckpt.every_steps > 0 and ((i + 1) % int(ckpt.every_steps) == 0)
            if due_t or due_s or (i + 1) == total:
                save_checkpoint(ckpt, arm, i + 1, state, opt_state, {
                    "arm": arm, "total_steps": total, "loss": history[-1],
                    "config_fingerprint": config_fp, "history": history,
                    "reason": "time" if due_t else ("step" if due_s else "final"),
                })
                last_ckpt = time.time()
                n_ckpt += 1
    return state, history, {
        "steps_run": int(total - start),
        "resumed_from": int(start),
        "total_steps": int(total),
        "n_checkpoints_written": int(n_ckpt),
        "wall_seconds": float(time.time() - t0),
        "logit_lr_mult": float(logit_lr_mult),
    }


# ---------------------------------------------------------------------------
# 6. ⭐⭐ THE METRIC — "semantic" means ADDRESS-BOOK SHARPNESS PER FACTOR
# ---------------------------------------------------------------------------


def address_sharpness(
    codes: np.ndarray,
    labels: np.ndarray,
    factor_names: Sequence[str],
    cardinalities: Sequence[int],
    *,
    n_modes_per_well: Sequence[int],
    smoothing: float = 1.0,
) -> dict:
    """⭐⭐ ``S_fk = 1 − H(p)/log(n_values_realised(f))`` — ``0`` uniform, ``1`` a delta.

    ``p`` is the address book's posterior over factor ``f``'s values induced by well
    ``k``'s address, averaged over addresses under ``P(m|k)``; equivalently
    ``S_fk = 1 − H(f | code_k)/log V_realised``.  At a uniform realised marginal this is
    ``I(f ; code_k)/H(f)``.

    ⛔⛔ **``n_values_realised`` is on EVERY row.**  Montero's corner pins ``shape`` to
    **1 of 4** values, and a statistic on a one-valued factor yields a confident-looking
    number that reads exactly like a finding — that has produced a wrong headline in this
    programme once already.  ⭐ Here ``V_realised < 2`` makes ``log V = 0`` and the
    sharpness **UNDEFINED**: it is reported as ``None`` with ``degenerate: True``,
    ⛔ never as 0 and ⛔ never as 1.

    ⚠ **The second trap, closed here rather than by a referee.**  ``S`` is normalised by
    ``log V``, not by ``H(f)``, so a factor with a *skewed* marginal scores above zero
    while carrying no information at all.  ⭐ ``marginal_only_sharpness`` is therefore
    reported beside every row: it is exactly what ``S`` would be if the code said nothing,
    and ``sharpness_above_marginal`` is the difference.  (The D-matched random floor
    controls for this empirically as well, since it shares the marginal — but a number
    that needs another arm to be interpretable is a number that will be quoted alone.)
    """
    C = np.asarray(codes, np.int64)
    Y = np.asarray(labels, np.int64)
    if C.shape[0] != Y.shape[0]:
        raise ValueError(f"codes has {C.shape[0]} rows, labels {Y.shape[0]}")
    k_wells = int(C.shape[1])
    ms = [int(x) for x in n_modes_per_well]
    realised = realised_values(Y, list(factor_names))
    a = float(smoothing)
    per_factor = {}
    for f, (name, card) in enumerate(zip(factor_names, cardinalities, strict=True)):
        vals = np.unique(Y[:, f])
        v = int(vals.size)
        remap = {int(u): i for i, u in enumerate(vals)}
        yy = np.asarray([remap[int(t)] for t in Y[:, f]], np.int64)
        pv = np.bincount(yy, minlength=max(v, 1)).astype(np.float64)
        pv = pv / max(pv.sum(), 1.0)
        h_marg = float(-(pv[pv > 0] * np.log(pv[pv > 0])).sum())
        degenerate = v < 2
        logv = float(np.log(v)) if v >= 2 else 0.0
        rows = []
        for k in range(k_wells):
            m = ms[k] if k < len(ms) else int(C[:, k].max()) + 1
            joint = np.zeros((m, v), np.float64)
            np.add.at(joint, (C[:, k], yy), 1.0)
            joint += a
            joint /= joint.sum()
            pm = joint.sum(axis=1)                       # (M,)
            cond = joint / np.maximum(pm[:, None], 1e-300)
            with np.errstate(divide="ignore", invalid="ignore"):
                h_rows = -np.sum(np.where(cond > 0, cond * np.log(cond), 0.0), axis=1)
            h_cond = float(np.sum(pm * h_rows))
            rows.append({
                "well": int(k),
                "n_modes": int(m),
                "n_values_realised": int(v),          # ⛔ on EVERY row, no exceptions
                "cardinality": int(card),
                "degenerate": bool(degenerate),
                "conditional_entropy_nats": h_cond,
                "log_n_values_realised": logv,
                "sharpness": None if degenerate else float(1.0 - h_cond / logv),
                "marginal_only_sharpness": None if degenerate else float(1.0 - h_marg / logv),
                "sharpness_above_marginal": (
                    None if degenerate else float((h_marg - h_cond) / logv)),
                "mutual_information_nats": None if degenerate else float(h_marg - h_cond),
            })
        defined = [r["sharpness"] for r in rows if r["sharpness"] is not None]
        per_factor[str(name)] = {
            "n_values_realised": int(v),
            "cardinality": int(card),
            "degenerate": bool(degenerate),
            "marginal_entropy_nats": h_marg,
            "per_well": rows,
            "max_sharpness": float(max(defined)) if defined else None,
            "mean_sharpness": float(np.mean(defined)) if defined else None,
            "argmax_well": (int(max(range(len(rows)),
                                    key=lambda i: (rows[i]["sharpness"] or -np.inf)))
                            if defined else None),
        }
    ok = [d["max_sharpness"] for d in per_factor.values() if d["max_sharpness"] is not None]
    return {
        "metric": ("S_fk = 1 - H(f | code_k) / log n_values_realised(f); "
                   "0 uniform, 1 a delta"),
        "n_rows": int(C.shape[0]),
        "n_wells": k_wells,
        "smoothing": a,
        "realised_values": realised,
        "per_factor": per_factor,
        "mean_max_sharpness": float(np.mean(ok)) if ok else None,
        "n_degenerate_factors": int(sum(1 for d in per_factor.values() if d["degenerate"])),
    }


def placement_margin(arm_sharpness: dict, floor_sharpness: dict, *, arm: str = "",
                     floor_arm: str = FLOOR_ARM) -> dict:
    """⭐⭐ **THE ONLY FORM A PLACEMENT NUMBER MAY BE REPORTED IN: a margin over the floor.**

    ⛔ A sharpness number without its D-matched random-placement floor beside it is **void**
    — an untrained random-init encoder has already scored 0.9361/0.9865 against a trained
    encoder's 0.935/0.991 in this programme, for 1,246x less compute, so "high on seen
    data" measures nothing.  This function exists so a margin is the *only* thing the
    harness can hand a report.
    """
    out = {}
    for name, d in arm_sharpness["per_factor"].items():
        fl = floor_sharpness["per_factor"].get(name, {})
        a_max, f_max = d.get("max_sharpness"), fl.get("max_sharpness")
        a_mean, f_mean = d.get("mean_sharpness"), fl.get("mean_sharpness")
        out[name] = {
            "n_values_realised": d.get("n_values_realised"),
            "degenerate": d.get("degenerate"),
            "arm_max_sharpness": a_max,
            "floor_max_sharpness": f_max,
            "margin_max": (None if a_max is None or f_max is None else float(a_max - f_max)),
            "arm_mean_sharpness": a_mean,
            "floor_mean_sharpness": f_mean,
            "margin_mean": (None if a_mean is None or f_mean is None
                            else float(a_mean - f_mean)),
        }
    margins = [v["margin_max"] for v in out.values() if v["margin_max"] is not None]
    return {
        "arm": str(arm),
        "floor_arm": str(floor_arm),
        "rule": ("⭐ every placement number is a MARGIN OVER THE D-MATCHED RANDOM-PLACEMENT "
                 "FLOOR; ⛔ never an absolute"),
        "per_factor": out,
        "mean_margin_max": float(np.mean(margins)) if margins else None,
        "best_factor": (max(out, key=lambda kk: (out[kk]["margin_max"] is not None,
                                                 out[kk]["margin_max"] or -np.inf))
                        if margins else None),
        "max_margin": float(max(margins)) if margins else None,
    }


def collapse_verdict(arm_usage: dict, floor_usage: dict, *, frac: float = 0.80) -> dict:
    """⭐ The pre-registered collapse guard: **occupancy reported beside every sharpness.**

    Sharpness can rise by **collapsing wells** — one mode eats the mass and the address
    book looks sharp because there is nothing left to choose between.  ⛔ A gain whose
    ``mean_used_modes`` has fallen below ``frac`` of the floor arm's is reported as
    **COLLAPSE, not as a win**, in those words.
    """
    a = float(arm_usage["mean_used_modes"])
    f = float(floor_usage["mean_used_modes"])
    ratio = a / f if f > 0 else float("nan")
    collapsed = bool(f > 0 and ratio < float(frac))
    return {
        "arm_mean_used_modes": a,
        "floor_mean_used_modes": f,
        "ratio": ratio,
        "threshold": float(frac),
        "collapsed": collapsed,
        "arm_mean_entropy_bits": float(arm_usage["mean_entropy_bits"]),
        "floor_mean_entropy_bits": float(floor_usage["mean_entropy_bits"]),
        "verdict": (
            "⛔ COLLAPSE — the wells emptied; a sharpness gain here is not a win"
            if collapsed else "occupancy held"),
    }


def sharpness_table(per_arm: dict, factor_names: Sequence[str]) -> str:
    """A markdown table.  ⛔ Every row carries ``n_values_realised``; degenerate rows read
    ``undefined``, ⛔ never ``0.000``."""
    arms = list(per_arm)
    head = "| factor | V realised | " + " | ".join(arms) + " |"
    rule = "|---|---|" + "---|" * len(arms)
    lines = [head, rule]
    for nm in factor_names:
        v, cells = None, []
        for a in arms:
            d = per_arm[a]["per_factor"].get(str(nm), {})
            v = d.get("n_values_realised", v)
            s = d.get("max_sharpness")
            cells.append("**undefined**" if d.get("degenerate") or s is None
                         else f"{s:.4f}")
        lines.append(f"| `{nm}` | {v} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def usage_from_codes(codes: np.ndarray, n_modes_per_well: Sequence[int]) -> dict:
    """Per-well occupancy.  Reuses :func:`chlu.core.gen_store.mode_usage` verbatim."""
    ms = [int(x) for x in n_modes_per_well]
    return mode_usage(np.asarray(codes, np.int64), int(max(ms)))


def build_store_from_labels(labels, n_modes, feat_shape, x, key, seed):
    """The delivered store for an arm: hard partition -> the ordinary ``init_store`` path.

    ⛔ Deliberately a thin wrapper and not a second store constructor: every arm must go
    through the **same** ``init_store`` + ``train_store`` the inherited partition goes
    through, so the only thing that differs between arms is the label vector.
    """
    from .gen_store import init_store

    k = int(np.asarray(labels).max()) + 1
    wc = well_channels_from_labels(np.asarray(labels, np.int64), k)
    return init_store(np.asarray(x, np.float32), wc, int(n_modes), feat_shape, key,
                      seed=int(seed))
