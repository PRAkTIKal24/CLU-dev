"""w5 ``hub/refuse`` -- rows 5, 6 and refuse-after-deletion, on the FIELD STORE.

⛔⛔ READ THIS BEFORE READING A NUMBER THIS MODULE PRODUCES
===========================================================
**Refusal is winnable at 8 bytes.**  A per-factor value-presence bitmap plus a bracket
check scores ``(1.0000, 1.0000)`` on rows 1, 2, 5 and 6 against a 524,288 B store.  That
object already exists in the frozen harness as
``exp_gen_factor_wells.value_presence_bitmap`` + ``exp_gen_factor_wells.accept_rule``,
and this module imports it **verbatim** rather than writing its own, so the comparator
cannot drift.  ⭐ Its score is printed BESIDE every refusal number here, on every row,
without exception.  ⛔ Beating the prototype table on a refusal row proves nothing.

⭐⭐ WHERE THE CLAIM LIVES, and it is EXACT rather than measured
----------------------------------------------------------------
A per-factor gate **cannot express a conjunction deletion at all**.  Delete the
conjunction ``object_hue = h* AND shape = s*`` from a store that holds both values
separately and the per-factor bitmap's bits are *unchanged* -- there is no bit in it
whose meaning is "this pair".  Its ``delta_reject`` is ``0.0000`` by construction, not by
measurement, and :func:`bitmap8_bits_changed_by_a_conjunction_deletion` returns the
integer ``0`` to make that checkable rather than rhetorical.

The joint table that CAN express it pays for it on the other leg: a pair-PRESENCE table
refuses every conjunction the pool never held, and ⛔ **every row-3 and every row-4
request is an unseen conjunction**.  It therefore scores ``accept_unseen_conjunction =
0`` on exactly the rows the programme exists to win.

⚠ THE RESIDUAL, DECLARED HERE RATHER THAN LEFT FOR A REFEREE TO FIND: a *third* object
does match the field store on this row -- a pair-DELETION table kept SEPARATE from
presence, so "never held" and "deleted" are different arrays.  That is not a gap in the
argument, it is the argument: it is precisely what ``gen_field``'s ``coupling_hole``
segment IS.  This module builds that object as an explicit comparator
(:func:`pair_deletion_accept`), prices it, and reports it.  What it shows is that the
cheap refuser which follows a conjunction deletion is not the 8-byte rule plus nothing --
it is the 8-byte rule plus a table whose size is ``sum_{f<g} V_f V_g`` bits, and that
that table is a *store*, bolted to a generator it does not have.

The rows
--------
``r1_seen``   the accept-leg (AMENDMENT 1): an on-grid request every value of which the
              store holds.  ⛔ NOT row 2 -- that pairing was measured free.
``r2_unseen_valid``  an off-grid hue strictly between two held, grid-adjacent values.
``r5_comp_refusal``  an ON-GRID value the store never held.  Evidential, rank 2.
``r6_invalid_refusal``  ``out_of_range`` + ``duplicate_factor``.  ⛔ ``evidential: false``
              in ``gen_value_holdout.ROW_EVIDENCE_RANKING`` and it can never be anything
              else: out-of-range is decidable from the factor grid alone and the pin says
              so in ``definitional_features``.  Kept as the degenerate-arm guard.
``r7_hole``   ⭐ refuse-after-CONJUNCTION-deletion.  THE row.
``r7_evict``  refuse-after-VALUE-deletion.  Reported, and reported as the row where the
              8-byte rule CAN follow (clear one bit), so it is not where the claim lives.

The refusers, all scored on the same requests at the same setting
------------------------------------------------------------------
``field_geom``      ⭐ PRIMARY.  ``hole | empty_basin | out_of_range | duplicate``.
                    Exactly ONE threshold (``basin_live_tau``); the other three
                    mechanisms are threshold-free.
``field_geom_gap``  the pinned four-mechanism rule, ``+ in_gap`` at ``gap_tau = 4.0``.
                    A declared variant, scored beside the primary, ⛔ not instead of it.
``bitmap8``         ⭐ THE BAR.  The frozen 8-byte rule, imported verbatim.
``pair_presence``   the joint pair-presence table (see above).
``pair_deletion``   the 8-byte rule + a SEPARATE pair-deletion table (the residual).
``grid_syntax``     the zero-byte no-model laundering rule from the frozen scoreboard.
``refuse_everything`` / ``accept_everything``  the degenerate guards; both must score 0.
``field_removed`` / ``field_shuffled``  charter §C.2 landscape controls.

⛔ THRESHOLDS, WRITTEN DOWN BEFORE ANY INVALID ROW IS SCORED
------------------------------------------------------------
``basin_live_tau = 2.0`` and ``gap_tau = 4.0``, both **calibrated on the val condition by
the w4 field-store hub** and pinned in ``exp_gen_field_store.PINS`` before this hub
existed.  ⛔ Neither is moved here.  Both are in units of a **within-value spread**
(``basin_sigma`` / ``well_sigma``) -- ⛔ NOT the spacing of stored items, which is the
length scale the admission gate died on.  ``basin_live_tau`` sits on a measured plateau:
at tau in {2, 4, 8} the set of empty basins is *exactly* the two never-held values, on
all three seeds.

Runnable::

    python -m chlu.experiments.exp_gen_refuse --stage fit      --seed S --out DIR
    python -m chlu.experiments.exp_gen_refuse --stage queries  --seed S --out DIR
    python -m chlu.experiments.exp_gen_refuse --stage deletion --seed S --out DIR
    python -m chlu.experiments.exp_gen_refuse --stage report   --out DIR

Every stage writes one artifact per (stage, seed), skips a complete one, takes
``--max-seconds`` and honours it between corners.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from .exp_gen_factor_wells import (
    Judge,
    _jdump,
    _records_to_values,
    accept_rule,
    build_grid,
    build_partitions,
    fit_pool,
    gather,
    features_mmap,
    feature_cache_manifest,
    judge_valid,
    load_condition,
    nway_corners,
    pair_subcorners,
    value_presence_bitmap,
)
from .exp_gen_field_store import (
    PINS as FIELD_PINS,
    _host_factors,
    field_spec,
)

# ---------------------------------------------------------------------------
# 1. Pins.  Inherited from the field-store build hub; ⛔ no threshold moved.
# ---------------------------------------------------------------------------

PINS = dict(FIELD_PINS)
PINS.update({
    # ⭐ THE SCORED CONDITION.  The w4 build hub ran everything on `val` and spent no
    # scored run; this hub spends them, on the shipped query sets and the 109 scored
    # corners.
    "condition": "test",
    # ⛔ the fit pool excludes NOTHING beyond what the value-holdout train index already
    # excludes.  `cond["train_rows"]` is the complement of every held-out cell, so a
    # further exclusion would be a second, undeclared hold-out.
    "pool_exclude": None,
    # -- the deletion targets --------------------------------------------------
    # ⭐ THE HOLE: a conjunction the pool HOLDS (so the deletion is a real removal, not a
    # no-op), NAMED by the scored row-4 corners, with neither of its values held out, and
    # ⛔ disjoint from all three scored pair cells ((0,1), (3,2), (7,3)) -- which are the
    # conjunctions that must STAY ANSWERED.  Measured on the test-condition train index:
    # object_hue index 1 x shape index 1 holds 11,676 rows.
    "hole_pair": ((2, 1), (4, 1)),
    # The VALUE eviction target, inherited verbatim from the frozen harness's own
    # declaration (`exp_gen_factor_wells.DELETED_FACTOR/DELETED_VALUE_INDEX`): object_hue
    # index 0.  It is pair-000's hue, so the survivor set is exactly the sub-corners of
    # the other two pair cells plus every n-way corner.
    "delete_factor": 2, "delete_value": 0,
    # -- the rule --------------------------------------------------------------
    # ⭐ SELECTED ON VAL, BEFORE ANY INVALID ROW WAS SCORED (hub LOG 2026-09-20T00:22Z):
    # `in_gap` fires on 0.6-1.2 % of the store's OWN pool requests and on 0.0000 of
    # unseen two-way requests at this tau.  It is pure accept-leg cost on this encoder.
    # The primary rule drops it; the pinned four-mechanism rule is scored beside it.
    "primary_rule": "field_geom",
    "n_row1_requests": 512,
    # ⛔⛔ THE RIVAL'S THRESHOLD, SET ON VALID DATA ONLY and written down before any
    # invalid row was scored with it (hub LOG 2026-09-20T01:25Z, artifact
    # `knn_calibration_val.json`): the 99th percentile of the k=1 nearest-atom distance,
    # in units of the ATOMS' OWN median nearest-neighbour spacing, over the VAL field
    # store's own fit-pool requests, max over the three seeds, rounded up to 2 dp.
    # Measured p99 = 4.4972 / 3.8885 / 3.6813.
    # ⭐ This length scale is the one the target file forbids ME. It is here as a RIVAL,
    # to isolate basin geometry from item spacing. ⛔ No claim of this hub uses it.
    "knn_tau": 4.5,
    "knn_k": 1,
})

#: ⭐ The refusal mechanisms the target file names, and which rule carries each.
MECHANISMS = {
    "empty_basin": "the request names a value with NO basin -- zero LIVE atoms in it",
    "hole": "an asked term PAIR is flagged deleted in `coupling_hole`",
    "in_gap": "the address sits further than `gap_tau` widths from every live basin",
    "out_of_range": "the value is beyond the last basin of its factor",
    "duplicate_factor": "the request names one factor twice",
}

RULES_PRIMARY = ("hole", "empty_basin", "out_of_range", "duplicate_factor")
RULES_GAP = RULES_PRIMARY + ("in_gap",)

QUERY_ROWS = (
    ("r1_seen", "row1_seen"),
    ("r2_unseen_valid", "row2_unseen"),
    ("r5_comp_refusal", "row5_held_out"),
    ("r6_invalid_refusal", "c_invalid"),
)

SEEDS = (0, 1, 2)


def _now() -> float:
    return time.time()


def seed_dir(out: Path, seed: int) -> Path:
    return out / f"seed_{int(seed)}"


# ---------------------------------------------------------------------------
# 2. The fit pool and the field store, on the SCORED condition
# ---------------------------------------------------------------------------


def refuse_fit_pool(cond, seed: int = 0) -> np.ndarray:
    """The fit pool for one store seed, on the test condition.

    ⛔ Nothing is excluded beyond the value-holdout train index itself: that index is
    already the complement of every held-out cell (the two value cells, the three pair
    cells and the n-way cells), so an extra exclusion would be an undeclared hold-out.
    The store seed moves the DRAW, which is what makes the three-seed spread the
    object's spread rather than an optimiser's jitter.
    """
    rng_seed = int(PINS["pool_rng_seed"]) + int(PINS["pool_seed_stride"]) * int(seed)
    return fit_pool(cond["train_rows"], PINS["n_pool"], rng_seed, exclude=None)


def build_field_arm(out: Path, seed: int, log: Callable = print) -> dict:
    """Write the field store for one seed on the SCORED pool.  Resumable."""
    from ..core.gen_field import (
        assert_read_request_only, build_field, field_byte_ledger,
    )

    d = seed_dir(out, seed) / "arms" / "field"
    done = d / "arm.pkl"
    if done.exists():
        log(f"  field(seed {seed}): already built, loading")
        with open(done, "rb") as fh:
            return pickle.load(fh)

    cond = load_condition()
    grid = cond["grid"]
    parts = build_partitions(out / "partitions", log=log)
    labels = np.asarray(parts["labels"]["factor"], np.int64)
    K = int(labels.max()) + 1
    hosts = _host_factors(parts["info"], K)
    pool = refuse_fit_pool(cond, seed)
    F = features_mmap(int(PINS["bridge_seed"]))
    t0 = _now()
    X = gather(F, pool)
    Y = grid.factor_indices_of(pool)
    cards = [int(c) for c in grid.cardinalities]
    log(f"  field(seed {seed}): pool {pool.size} rows, hosts {hosts}")
    field = build_field(X, Y, pool, labels, hosts, cards, field_spec(seed), log=log)
    del X
    led = field_byte_ledger(field, int(PINS["store_bytes"]))
    if not led["matches_budget"]:
        raise AssertionError(
            f"the field store deploys {led['deployed_bytes']} B against the pinned "
            f"{PINS['store_bytes']} B ledger -- the byte ledger has moved")
    # charter §B.3 by SIGNATURE: no feature argument exists on the read.
    ro = assert_read_request_only(field, grid.values_of(pool[:64]), grid.values)
    obj = {"arm": "field", "kind": "field", "seed": int(seed), "field": field,
           "byte_ledger": led, "request_only": ro, "manifest": field.manifest(),
           "pool_n": int(pool.size), "fit_seconds": _now() - t0,
           "feature_cache": feature_cache_manifest(int(PINS["bridge_seed"]))}
    d.mkdir(parents=True, exist_ok=True)
    with open(done, "wb") as fh:
        pickle.dump(obj, fh)
    _jdump({k: v for k, v in obj.items() if k != "field"}, d / "arm.json")
    log(f"  field(seed {seed}): built in {obj['fit_seconds']:.1f}s; atoms "
        f"{led['deployed_bytes']:,} B + interface {led['interface_bytes']:,} B "
        f"(image {led['total_image_bytes']:,} B); coupling_hole "
        f"{led['per_segment']['coupling_hole']['bytes']:,} B")
    return obj


def landscape_control(obj: dict, kind: str, seed: int) -> dict:
    """Charter §C.2 controls: atoms ZEROED, or atom coordinates SHUFFLED.

    ⭐ Both must collapse.  `removed` zeroes the atom bytes, so every basin holds zero
    live atoms and the geometry rule refuses EVERYTHING -- which is the degenerate arm
    and scores 0 on the pair, exactly as it should.  `shuffled` keeps the atoms but
    permutes their coordinates within each well, so the landscape carries the same bytes
    in the wrong places.
    """
    out = dict(obj)
    fld = obj["field"].copy()
    A = fld.view("atoms")
    if kind == "removed":
        A[:] = 0.0
    elif kind == "shuffled":
        rng = np.random.default_rng(90_210 + int(seed))
        flat = np.asarray(A).reshape(-1)
        flat[:] = flat[rng.permutation(flat.size)]
    else:
        raise ValueError(kind)
    out["field"] = fld
    out["arm"] = f"field_landscape_{kind}"
    return out


# ---------------------------------------------------------------------------
# 3. The refusers.  Every one returns (accept, detail) on the SAME requests.
# ---------------------------------------------------------------------------


def field_accept(field, values: np.ndarray, dup: np.ndarray, grid,
                 rules: Sequence[str] = RULES_PRIMARY) -> tuple:
    """The field store's own geometry decides admissibility.  ⛔ No label, no feature.

    `gap_read` returns the three geometric mechanisms plus `out_of_range`; `dup` is the
    one syntactic term and it is declared as such rather than folded in silently.
    """
    from ..core.gen_field import gap_read

    g = gap_read(field, values, grid.values)
    ref = np.zeros(values.shape[0], bool)
    parts = {}
    for r in rules:
        m = (np.asarray(dup, bool) if r == "duplicate_factor"
             else np.asarray(g[r], bool))
        parts[r] = m
        ref |= m
    detail = {f"fires_{k}": float(v.mean()) for k, v in parts.items()}
    detail["gap_score_max_p99"] = float(np.quantile(g["gap_score"].max(axis=1), 0.99))
    detail["term_live_atoms"] = g["term_live_atoms"]
    return ~ref, detail


def bitmap8_accept(values: np.ndarray, dup: np.ndarray, oor: np.ndarray,
                   bitmap: list, grid) -> np.ndarray:
    """⭐ THE BAR: the frozen 8-byte rule, imported verbatim, not reimplemented."""
    return accept_rule(values, oor, dup, bitmap, grid)


def bitmap8_bytes(grid) -> int:
    """8 bytes: one bit per (factor, value), 57 terms -> 8 bytes packed."""
    return int(np.ceil(sum(int(c) for c in grid.cardinalities) / 8.0))


def bitmap8_bits_changed_by_a_conjunction_deletion() -> int:
    """⭐⭐ EXACT, not measured.  Returns 0, and that integer IS the claim.

    A per-factor value-presence bitmap has one bit per ``(factor, value)``.  Deleting the
    conjunction ``(f, v) AND (g, u)`` from a store that still holds ``v`` and still holds
    ``u`` leaves every one of those bits at its previous value: there is no bit in the
    object whose meaning is "this pair".  So the rule's ``delta_reject`` on the deleted
    conjunction is ``0.0000`` by construction.  ⛔ Clearing ``(f, v)``'s bit instead would
    refuse every request naming ``v`` with ANY partner, which is a value deletion and a
    different operation.
    """
    return 0


# -- the joint tables -------------------------------------------------------


def pair_layout(grid) -> tuple:
    """``(pairs, offsets, total_cells)`` -- literally ``gen_field``'s own pair layout."""
    from ..core.gen_field import coupling_layout

    cards = [int(c) for c in grid.cardinalities]
    lay = coupling_layout(cards)
    return lay["pairs"], lay["offsets"], int(lay["size"])


def pair_presence_table(grid, train_rows: np.ndarray) -> np.ndarray:
    """``(total_cells,) uint8`` -- did the fit pool ever hold this ``(f,v) x (g,u)``?"""
    cards = [int(c) for c in grid.cardinalities]
    pairs, offs, size = pair_layout(grid)
    fi = grid.factor_indices_of(np.asarray(train_rows, np.int64))
    tab = np.zeros(size, np.uint8)
    for (f, g), off in zip(pairs, offs):
        f, g, off = int(f), int(g), int(off)
        lin = fi[:, f] * cards[g] + fi[:, g]
        tab[off + np.unique(lin)] = 1
    return tab


def _candidate_indices(values: np.ndarray, grid) -> list:
    """Per factor, the grid indices a request's value implicates.

    ⛔ The SAME rule ``accept_rule`` uses: an on-grid value implicates itself; an
    off-grid value strictly inside the range implicates BOTH bracketing indices; an
    unasked factor implicates none.
    """
    n, F = values.shape
    out = []
    for f in range(F):
        gv = np.asarray(grid.values[f], np.float64)
        v = values[:, f]
        asked = ~np.isnan(v)
        vv = np.where(asked, v, gv[0])
        hi = np.clip(np.searchsorted(gv, vv, side="left"), 0, gv.size - 1)
        lo = np.clip(hi - 1, 0, gv.size - 1)
        exact = np.isclose(gv[hi], vv, rtol=0.0, atol=1e-12)
        out.append({"asked": asked, "hi": hi, "lo": np.where(exact, hi, lo),
                    "exact": exact})
    return out


def _pairs_all_flagged(values: np.ndarray, grid, flag: np.ndarray,
                       want: int) -> np.ndarray:
    """``(N,)`` -- does some asked pair have ``flag != want`` on any bracketing cell?

    With ``want = 1`` this is "some asked pair is NOT present" (the presence table's
    refusal); with ``want = 0`` it is "some asked pair IS flagged deleted".
    """
    cards = [int(c) for c in grid.cardinalities]
    pairs, offs, _ = pair_layout(grid)
    cand = _candidate_indices(values, grid)
    bad = np.zeros(values.shape[0], bool)
    for (f, g), off in zip(pairs, offs):
        f, g, off = int(f), int(g), int(off)
        both = cand[f]["asked"] & cand[g]["asked"]
        if not both.any():
            continue
        # every bracketing combination must satisfy the rule -- the same "all brackets"
        # convention `accept_rule` uses on the per-factor bitmap.
        hit = np.zeros(values.shape[0], bool)
        for a in ("hi", "lo"):
            for b in ("hi", "lo"):
                cell = off + cand[f][a] * cards[g] + cand[g][b]
                hit |= (flag[cell] != want)
        bad |= both & hit
    return bad


def pair_presence_accept(values: np.ndarray, dup: np.ndarray, oor: np.ndarray,
                         table: np.ndarray, grid) -> np.ndarray:
    """The JOINT pair-presence table: accept iff every asked PAIR was held.

    ⛔ This is the object that CAN express a conjunction deletion (clear the pair's bit)
    and it is exactly the object that refuses every unseen conjunction -- so it refuses
    every row-3 and every row-4 request.  Both halves are measured, not asserted.
    """
    ok = ~(np.asarray(dup, bool) | np.asarray(oor, bool))
    return ok & ~_pairs_all_flagged(values, grid, table, want=1)


def pair_deletion_accept(values: np.ndarray, dup: np.ndarray, oor: np.ndarray,
                         bitmap: list, deleted: np.ndarray, grid) -> np.ndarray:
    """⚠ THE DECLARED RESIDUAL: the 8-byte rule + a SEPARATE pair-deletion table.

    Presence and deletion in different arrays, so "never held" and "deleted" cannot be
    confused.  ⭐ This object DOES match the field store on the conjunction-deletion row,
    and saying so here is the point: the cheap refuser that follows a conjunction
    deletion is not the 8-byte rule, it is the 8-byte rule plus a table of
    ``sum_{f<g} V_f V_g`` bits -- and that table is a store with no generator attached.
    """
    ok = accept_rule(values, oor, dup, bitmap, grid)
    return ok & ~_pairs_all_flagged(values, grid, deleted, want=0)


def pair_table_bytes(grid) -> dict:
    """What a joint pair table costs, counted rather than guessed."""
    cards = [int(c) for c in grid.cardinalities]
    pairs, _, size = pair_layout(grid)
    return {"n_pairs": len(pairs), "n_cells": int(size),
            "bytes_packed_1bit": int(np.ceil(size / 8.0)),
            "bytes_1_per_cell": int(size),
            "what": "sum over f<g of V_f * V_g cells, one bit each when packed"}


# -- the rival refusers the target file names -------------------------------


def prototype_table_counts(grid, train_rows: np.ndarray) -> list:
    """Per ``(factor, value)``, how many fit-pool rows hold it.

    ⭐ THE PROTOTYPE TABLE REFUSES BY ABSENCE: its entry for ``(f, v)`` is the mean
    feature of the pool rows holding ``v``, so a value the pool never held has **no
    entry** and the table cannot answer a request naming it.  That is a real refusal
    mechanism and the target file names it as a rival.  ⛔ It is built here from the pool
    directly, not read off the field store.
    """
    fi = grid.factor_indices_of(np.asarray(train_rows, np.int64))
    return [np.bincount(fi[:, f], minlength=int(c)).astype(np.int64)
            for f, c in enumerate(grid.cardinalities)]


def table_absence_bytes(grid) -> int:
    """``(sum_k C_k + 1) x D`` float32 -- the prototype table's own ledger."""
    return int((sum(int(c) for c in grid.cardinalities) + 1) * int(PINS["D"]) * 4)


def table_absence_accept(values: np.ndarray, dup: np.ndarray, oor: np.ndarray,
                         counts: list, grid) -> np.ndarray:
    """The prototype table's rule: accept iff every asked value HAS a prototype.

    ⚠ MEASURED CONSEQUENCE, and it is the honest reading: this decides on exactly the
    same fact as the 8-byte bitmap -- which ``(factor, value)`` the pool holds -- and so
    returns exactly the same answers, at **475,136 B instead of 8**.  The byte multiple
    is printed beside the row (w5-common §1 / §3.9).
    """
    return accept_rule(values, oor, dup, [c > 0 for c in counts], grid)


def knn_density_score(field, values: np.ndarray, grid, k: int = 1) -> np.ndarray:
    """``(N,)`` distance from the request's address to its k-th nearest LIVE ATOM.

    ⛔⛔ THIS IS THE LENGTH SCALE THE TARGET FILE FORBIDS ME, and that is precisely why
    it is here as a RIVAL: the admission gate died on deriving a length scale from the
    spacing of stored items.  ``in_gap`` measures to **basin coordinates** in units of a
    **within-value spread**; this measures to **atoms** in units of the atoms' own
    spacing.  The pair isolates exactly that difference.  ⛔ No number this hub claims is
    computed with it.
    """
    from ..core.gen_field import LIFETIME_EPS, field_address

    mu = field_address(field, values, grid.values)["mu"]
    liv = (np.asarray(field.view("atom_alive"), np.uint8) > 0) & \
          (np.asarray(field.view("atom_life"), np.float32) > LIFETIME_EPS)
    worst = np.zeros(values.shape[0], np.float64)
    for kk in range(field.K):
        ch = np.asarray(field.meta["well_channels"][kk], np.int64)
        Ak = np.asarray(field.atoms_of_well(kk), np.float64)[liv]
        if Ak.size == 0:
            worst[:] = np.inf
            continue
        M = np.asarray(mu[:, ch], np.float64)
        d2 = ((M * M).sum(1)[:, None] - 2.0 * M @ Ak.T + (Ak * Ak).sum(1)[None, :])
        d = np.sqrt(np.maximum(d2, 0.0))
        kth = np.partition(d, min(k, d.shape[1]) - 1, axis=1)[:, min(k, d.shape[1]) - 1]
        # ⛔ the scale is the atoms' OWN nearest-neighbour spacing in this well
        aa = np.sqrt(np.maximum(
            (Ak * Ak).sum(1)[:, None] - 2.0 * Ak @ Ak.T + (Ak * Ak).sum(1)[None, :], 0.0))
        np.fill_diagonal(aa, np.inf)
        scale = float(np.median(aa.min(axis=1))) if Ak.shape[0] > 1 else 1.0
        worst = np.maximum(worst, kth / max(scale, 1e-9))
    return worst


# ---------------------------------------------------------------------------
# 4. The requests
# ---------------------------------------------------------------------------


def row1_request_rows(cond, pool: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(int(PINS["row1_rng_seed"]))
    n = int(min(int(PINS["n_row1_requests"]), pool.size))
    return np.sort(pool[rng.permutation(pool.size)[:n]])


def query_requests(cond, grid) -> dict:
    """The four shipped query sets as ``(values, dup, ids)``."""
    out = {}
    for row_id, qname in QUERY_ROWS:
        recs = cond["query_sets"][qname]["records"]
        vals, dup = _records_to_values(recs, grid)
        out[row_id] = {"values": vals, "dup": dup,
                       "ids": [r.get("id") for r in recs], "qname": qname,
                       "n": int(vals.shape[0])}
    return out


def scored_corners(cond) -> tuple:
    """The 60 row-3 pair sub-corners and the 49 row-4 n-way corners."""
    return pair_subcorners(cond), nway_corners(cond)


def corner_values(grid, corners) -> tuple:
    """Concatenate every corner's requests, keeping the ``(start, end)`` offsets."""
    vals, offs = [], []
    s = 0
    for c in corners:
        v = grid.values_of(np.asarray(c["rows"], np.int64))
        vals.append(v)
        offs.append((s, s + v.shape[0]))
        s += v.shape[0]
    return np.concatenate(vals, axis=0), offs


# ---------------------------------------------------------------------------
# 5. Stage: the query rows (1, 2, 5, 6)
# ---------------------------------------------------------------------------


def run_queries(out: Path, seed: int, max_seconds: float = 3600.0,
                log: Callable = print) -> dict:
    """⛔ SCORED.  Rows 1, 2, 5 and 6 for every refuser, at one seed."""
    art = seed_dir(out, seed) / "queries.npz"
    if art.exists():
        log(f"  queries(seed {seed}): already on disk")
        return {"skipped": True}

    t0 = _now()
    cond = load_condition()
    grid = cond["grid"]
    obj = build_field_arm(out, seed, log=log)
    fld = obj["field"]
    pool = refuse_fit_pool(cond, seed)
    bitmap = value_presence_bitmap(grid, pool)
    ptable = pair_presence_table(grid, pool)
    no_deletions = np.zeros(ptable.size, np.uint8)
    J = Judge(int(PINS["bridge_seed"]), log=log)

    Q = query_requests(cond, grid)
    # ⭐ row 1 is drawn from the fit pool, exactly as its rationale requires, but the
    # SHIPPED row1_seen query set is the accept-leg the scoreboard names, so that is what
    # is scored.  The pool-drawn set is kept as a declared diagnostic.
    arms = {}
    detail = {}
    for row_id, q in Q.items():
        vals, dup = q["values"], q["dup"]
        from ..core.gen_field import field_address, field_read
        addr = field_address(fld, vals, grid.values)
        oor = addr["out_of_range"]
        feats = field_read(fld, vals, grid.values)
        tup = J.read_back(feats, int(PINS["eval_batch"]))

        a_prim, d_prim = field_accept(fld, vals, dup, grid, RULES_PRIMARY)
        a_gap, d_gap = field_accept(fld, vals, dup, grid, RULES_GAP)
        a_b8 = bitmap8_accept(vals, dup, oor, bitmap, grid)
        a_pp = pair_presence_accept(vals, dup, oor, ptable, grid)
        a_pd = pair_deletion_accept(vals, dup, oor, bitmap, no_deletions, grid)

        # the landscape controls, on the same requests and the same rule
        a_rm, _ = field_accept(landscape_control(obj, "removed", seed)["field"],
                               vals, dup, grid, RULES_PRIMARY)
        a_sh, _ = field_accept(landscape_control(obj, "shuffled", seed)["field"],
                               vals, dup, grid, RULES_PRIMARY)

        for name, acc in (("field_geom", a_prim), ("field_geom_gap", a_gap),
                          ("bitmap8", a_b8), ("pair_presence", a_pp),
                          ("pair_deletion", a_pd),
                          ("field_landscape_removed", a_rm),
                          ("field_landscape_shuffled", a_sh)):
            val = judge_valid(vals, tup, grid) & acc
            arms.setdefault(name, {})[row_id] = {
                "accept": acc, "valid": val,
                "tuples": np.where(acc[:, None], tup, -1), "ids": q["ids"],
                "n": int(acc.size),
            }
        detail.setdefault("field_geom", {})[row_id] = {
            k: v for k, v in d_prim.items() if k != "term_live_atoms"}
        detail.setdefault("field_geom_gap", {})[row_id] = {
            k: v for k, v in d_gap.items() if k != "term_live_atoms"}
        log(f"  seed {seed} {row_id} (n={q['n']}): field_geom accept "
            f"{a_prim.mean():.4f} | bitmap8 {a_b8.mean():.4f} | "
            f"pair_presence {a_pp.mean():.4f}")
        if _now() - t0 > float(max_seconds):
            log(f"  queries(seed {seed}): wall-clock cap hit -- ⛔ nothing partial is "
                "written for this stage; rerun to redo it")
            return {"incomplete": True}

    tl = d_prim["term_live_atoms"]
    meta = {
        "seed": int(seed), "condition": PINS["condition"],
        "byte_ledger": obj["byte_ledger"],
        "bitmap8_bytes": bitmap8_bytes(grid),
        "pair_table": pair_table_bytes(grid),
        "term_live_atoms": tl.tolist(),
        "n_terms_with_zero_live_atoms": int((tl == 0).sum()),
        "judge": J.info,
        "feature_cache": obj["feature_cache"],
        "rules": {"field_geom": list(RULES_PRIMARY),
                  "field_geom_gap": list(RULES_GAP)},
        "thresholds": {"basin_live_tau": float(PINS["field_basin_live_tau"]),
                       "gap_tau": float(PINS["field_gap_tau"]),
                       "set_on": "the val condition, by the w4 field-store hub, before "
                                 "this hub existed and before any invalid row was scored",
                       "units": "within-value spread (basin_sigma / well_sigma) -- ⛔ NOT "
                                "the spacing of stored items"},
        "detail": detail,
        "seconds": _now() - t0,
    }
    seed_dir(out, seed).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(art, arms=np.array(arms, dtype=object))
    _jdump(meta, seed_dir(out, seed) / "queries.json")
    log(f"  queries(seed {seed}): done in {_now() - t0:.1f}s")
    return {"incomplete": False}


# ---------------------------------------------------------------------------
# 6. Stage: refuse-after-deletion -- the HOLE and the EVICTION
# ---------------------------------------------------------------------------


def _byte_check(before, after, ranges, label: str, log: Callable) -> dict:
    from ..core.gen_field import assert_changed_bytes_within

    acc = assert_changed_bytes_within(before, after, ranges, label=label)
    log(f"    {label}: {acc['bytes_changed']:,} byte(s) changed of "
        f"{acc['total_image_bytes']:,}; OUTSIDE the permitted range: "
        f"{acc['bytes_changed_outside_permitted']}")
    return {k: v for k, v in acc.items() if k != "outside_examples"}


def _judge_in_chunks(J, feats: np.ndarray, log: Callable, label: str,
                     chunk: int = 1024) -> np.ndarray:
    """Read back in bounded chunks so peak memory stays flat on a long request set.

    ⚠ The chunk is a MEMORY knob and ⛔ cannot move a number: ``Judge.read_back`` loops
    internally at ``PINS["eval_batch"] = 256``, and every chunk size used here is a
    multiple of 256, so the internal batch boundaries are identical at any of them.
    Lowered from 4096 to 1024 after a seed died silently mid-judge on a box shared with
    three other agents (hub LOG 2026-09-20T01:10Z).
    """
    import jax

    if chunk % int(PINS["eval_batch"]):
        raise AssertionError(
            f"judge chunk {chunk} is not a multiple of eval_batch "
            f"{PINS['eval_batch']} -- it would move the internal batch boundaries")
    out = np.empty((feats.shape[0], 6), np.int64)
    t = _now()
    for s in range(0, feats.shape[0], chunk):
        e = min(s + chunk, feats.shape[0])
        out[s:e] = J.read_back(feats[s:e], int(PINS["eval_batch"]))
        # ⛔⛔ THE FROZEN HARNESS LEAKS A COMPILED EXECUTABLE PER CALL, and this is the
        # only thing standing between a three-seed run and an OOM on a shared box.
        # `gen_probe.decode_features` and `gen_probe.predict_factors` each build a FRESH
        # `eqx.filter_jit(jax.vmap(model))` on every call, and `Judge.read_back` calls
        # both once per `eval_batch`-sized slice. JAX keys its compilation cache on the
        # wrapped function object, so each one is a new executable that is never reused
        # and never freed -- measured by `w5 hub/delete` at ~13 MB/s, ~20 MB per unit.
        # ⛔ `gen_probe` is harness code and charter §A.4 says editing it voids the
        # numbers, so it is NOT edited. `jax.clear_caches()` is a JAX API call in MY
        # code that frees exactly those orphaned executables. ⭐ It cannot move a number:
        # it evicts compiled artifacts, and the next call recompiles the same function
        # to the same code. The only cost is recompilation time.
        jax.clear_caches()
        log(f"    judge[{label}] {e}/{feats.shape[0]} ({_now() - t:.0f}s)")
    return out


def run_deletion(out: Path, seed: int, max_seconds: float = 14400.0,
                 log: Callable = print) -> dict:
    """⛔ SCORED.  ⭐ THE row: refuse-after-deletion, both flavours, at one seed.

    ``hole``   a CONJUNCTION deletion -- one byte of ``coupling_hole``.  ⭐ The row where
               a per-factor gate provably cannot follow.
    ``evict``  a VALUE deletion -- whole items zeroed.  Reported, and reported as the row
               where the 8-byte rule CAN follow by clearing one bit.
    """
    art = seed_dir(out, seed) / "deletion.npz"
    if art.exists():
        log(f"  deletion(seed {seed}): already on disk")
        return {"skipped": True}
    # ⭐ RESUMABLE, and this is not cosmetic: the two halves are independent measurements
    # and the eviction half costs a second judge pass.  A kill during the eviction used
    # to cost the hole as well.  Each half is checkpointed the moment it is complete, so
    # a kill costs at most one half.  (Three of my seed processes were killed mid-stage
    # on this shared box -- hub LOG 2026-09-20T01:30Z and 02:30Z.)
    ck_h = seed_dir(out, seed) / "_ck_hole.pkl"
    ck_e = seed_dir(out, seed) / "_ck_evict.pkl"

    from ..core.gen_field import (
        assert_no_refit, byte_diff, coupling_index, evict_term, field_address,
        field_read, punch_hole,
    )

    t0 = _now()
    cond = load_condition()
    grid = cond["grid"]
    cards = [int(c) for c in grid.cardinalities]
    obj = build_field_arm(out, seed, log=log)
    f0 = obj["field"]
    pool = refuse_fit_pool(cond, seed)
    bitmap = value_presence_bitmap(grid, pool)
    ptable = pair_presence_table(grid, pool)
    zero_del = np.zeros(ptable.size, np.uint8)
    J = Judge(int(PINS["bridge_seed"]), log=log)

    (hf, hv), (hg, hu) = PINS["hole_pair"]
    ef, ev_i = int(PINS["delete_factor"]), int(PINS["delete_value"])

    # -- the request sets ---------------------------------------------------
    c3, c4 = scored_corners(cond)
    corners = c3 + c4
    CV, offs = corner_values(grid, corners)
    Q = query_requests(cond, grid)
    R1 = Q["r1_seen"]
    log(f"  seed {seed}: {len(c3)} row-3 + {len(c4)} row-4 corners = {CV.shape[0]} "
        f"corner requests; accept-leg row1_seen n={R1['n']}")

    ci = grid.factor_indices_of(np.concatenate(
        [np.asarray(c["rows"], np.int64) for c in corners]))
    names_hole = (ci[:, hf] == hv) & (ci[:, hg] == hu)
    names_evict = (ci[:, ef] == ev_i)
    # ⭐ the never-held conjunctions among the scored corners: these MUST stay answered.
    fi_tr = grid.factor_indices_of(cond["train_rows"])
    held_pairs = set(map(tuple, np.unique(fi_tr[:, [hf, hg]], axis=0).tolist()))
    unseen_conj = np.array([tuple(r) not in held_pairs
                            for r in ci[:, [hf, hg]].tolist()], bool)
    log(f"    corner requests naming the HOLE conjunction "
        f"(f{hf}={hv} & f{hg}={hu}): {int(names_hole.sum())}")
    log(f"    corner requests naming the EVICTED value (f{ef}={ev_i}): "
        f"{int(names_evict.sum())}")
    log(f"    corner requests naming a NEVER-HELD (f{hf},f{hg}) conjunction: "
        f"{int(unseen_conj.sum())}  ⭐ these must stay ANSWERED")

    # -- the matched 4-way probe groups, built from the accept-leg requests --
    # ⛔ CONSTRUCTED, not filtered: they share one set of backgrounds and differ only in
    # the two factors, so the conjunction, each marginal alone and neither are matched.
    def _set(v, pairs):
        w = np.array(v, np.float64, copy=True)
        for ax, ix in pairs:
            w[:, ax] = float(grid.values[ax][ix])
        return w

    alt_f = (hv + 1) % cards[hf]
    alt_g = (hu + 1) % cards[hg]
    if alt_f == 5 and hf == 2:          # ⛔ never fall onto a HELD-OUT value
        alt_f = (alt_f + 1) % cards[hf]
    if alt_g == 0 and hg == 4:
        alt_g = (alt_g + 1) % cards[hg]
    GROUPS = {
        "both": _set(R1["values"], [(hf, hv), (hg, hu)]),
        "A_only": _set(R1["values"], [(hf, hv), (hg, alt_g)]),
        "B_only": _set(R1["values"], [(hf, alt_f), (hg, hu)]),
        "neither": _set(R1["values"], [(hf, alt_f), (hg, alt_g)]),
    }
    log(f"    matched probe groups built on row1_seen's backgrounds; alternates "
        f"f{hf}={alt_f}, f{hg}={alt_g}")

    # -- render + judge ONCE on the base store ------------------------------
    def _render(fld, V):
        return field_read(fld, V, grid.values), field_address(
            fld, V, grid.values)["out_of_range"]

    FC, oor_C = _render(f0, CV)
    F1, oor_1 = _render(f0, R1["values"])

    TC = _judge_in_chunks(J, FC, log, f"corners s{seed}")
    T1 = _judge_in_chunks(J, F1, log, f"row1 s{seed}")
    GF = {k: _render(f0, v) for k, v in GROUPS.items()}
    GT = {k: _judge_in_chunks(J, GF[k][0], log, f"grp-{k} s{seed}")
          for k in GROUPS}

    # =======================================================================
    # A. THE HOLE -- a conjunction deletion
    # =======================================================================
    fh_store = f0.copy()
    before = fh_store.copy()
    ho = punch_hole(fh_store, hf, hv, hg, hu, log=log)
    ba = _byte_check(before, fh_store, ho["permitted_ranges"],
                     f"HOLE (f{hf}={hv} & f{hg}={hu})", log)
    nr = assert_no_refit(before, fh_store)
    # ⛔ ASSERTED, not assumed: the hole changes no rendered feature anywhere.
    FC_post, _ = _render(fh_store, CV)
    F1_post, _ = _render(fh_store, R1["values"])
    feats_identical = bool(np.array_equal(FC, FC_post) and np.array_equal(F1, F1_post))
    if not feats_identical:
        raise AssertionError(
            "the hole changed a rendered feature -- the pre-hole judge pass may not be "
            "reused and this run is void")
    log(f"    hole: rendered features bit-identical on all {CV.shape[0]} corner and "
        f"{R1['n']} accept-leg requests: {feats_identical}")

    hole_tab = zero_del.copy()
    hole_tab[coupling_index(cards, hf, hv, hg, hu)] = 1
    ptable_post = ptable.copy()
    ptable_post[coupling_index(cards, hf, hv, hg, hu)] = 0

    def _accepts(fld, V, dup, oor, bm, pres, dele):
        a_prim, _ = field_accept(fld, V, dup, grid, RULES_PRIMARY)
        a_gap, _ = field_accept(fld, V, dup, grid, RULES_GAP)
        return {
            "field_geom": a_prim, "field_geom_gap": a_gap,
            "bitmap8": bitmap8_accept(V, dup, oor, bm, grid),
            "pair_presence": pair_presence_accept(V, dup, oor, pres, grid),
            "pair_deletion": pair_deletion_accept(V, dup, oor, bm, dele, grid),
        }

    zC = np.zeros(CV.shape[0], bool)
    hole_blk = {
        "pair": [[hf, hv], [hg, hu]],
        "coupling_index": int(coupling_index(cards, hf, hv, hg, hu)),
        "byte_assertion": ba, "no_refit": nr,
        "rendered_features_bit_identical": feats_identical,
        "bitmap8_bits_changed": bitmap8_bits_changed_by_a_conjunction_deletion(),
        "n_deleted_requests": int(names_hole.sum()),
        "n_unseen_conjunction_requests": int(unseen_conj.sum()),
        "names_hole": names_hole, "unseen_conj": unseen_conj,
        "corner_tuples": TC, "corner_values": CV, "corner_offsets": offs,
        "corner_ids": [c["corner_id"] for c in corners],
        "corner_family": [c["family"] for c in corners],
        "row1_tuples": T1, "row1_ids": R1["ids"],
        "pre": {"corners": _accepts(f0, CV, zC, oor_C, bitmap, ptable, zero_del),
                "row1": _accepts(f0, R1["values"], R1["dup"], oor_1, bitmap,
                                 ptable, zero_del)},
        "post": {"corners": _accepts(fh_store, CV, zC, oor_C, bitmap, ptable_post,
                                     hole_tab),
                 "row1": _accepts(fh_store, R1["values"], R1["dup"], oor_1, bitmap,
                                  ptable_post, hole_tab)},
        "groups": {},
    }
    for k, V in GROUPS.items():
        z = np.zeros(V.shape[0], bool)
        hole_blk["groups"][k] = {
            "tuples": GT[k], "values": V,
            "pre": _accepts(f0, V, z, GF[k][1], bitmap, ptable, zero_del),
            "post": _accepts(fh_store, V, z, GF[k][1], bitmap, ptable_post, hole_tab),
        }
    for nm in ("field_geom", "bitmap8", "pair_presence", "pair_deletion"):
        pre = hole_blk["pre"]["corners"][nm][names_hole].mean()
        post = hole_blk["post"]["corners"][nm][names_hole].mean()
        uns = hole_blk["post"]["corners"][nm][unseen_conj].mean()
        log(f"    HOLE {nm:15s} accept on the deleted conjunction "
            f"{pre:.4f} -> {post:.4f}   |   accept on UNSEEN conjunctions {uns:.4f}")

    # =======================================================================
    # B. THE EVICTION -- a value deletion.  Features DO change, so re-render.
    # =======================================================================
    fe = f0.copy()
    before_e = fe.copy()
    evr = evict_term(fe, ef, ev_i, log=log)
    ba_e = _byte_check(before_e, fe, evr["permitted_ranges"],
                       f"EVICT f{ef}={ev_i}", log)
    nr_e = assert_no_refit(before_e, fe)
    FC_e, oor_Ce = _render(fe, CV)
    F1_e, oor_1e = _render(fe, R1["values"])
    TC_e = _judge_in_chunks(J, FC_e, log, f"corners-evicted s{seed}")
    T1_e = _judge_in_chunks(J, F1_e, log, f"row1-evicted s{seed}")
    # ⭐ the 8-byte rule CAN follow a VALUE deletion: clear the value's bit.  Done here,
    # explicitly, so the comparator is given its best case on this row.
    bm_post = [b.copy() for b in bitmap]
    bm_post[ef][ev_i] = False
    ptable_e = ptable.copy()
    pairs, poffs, _ = pair_layout(grid)
    for (a, b), off in zip(pairs, poffs):
        a, b, off = int(a), int(b), int(off)
        if a == ef:
            ptable_e[off + ev_i * cards[b]: off + (ev_i + 1) * cards[b]] = 0
        elif b == ef:
            ptable_e[off + np.arange(cards[a]) * cards[b] + ev_i] = 0
    evict_blk = {
        "factor": ef, "value": ev_i,
        "byte_assertion": ba_e, "no_refit": nr_e,
        **{k: v for k, v in evr.items() if k != "permitted_ranges"},
        "n_deleted_requests": int(names_evict.sum()),
        "names_evict": names_evict,
        "read_changed_on_surviving_requests_frac": float(
            (np.abs(FC_e[~names_evict] - FC[~names_evict]) > 0).any(axis=1).mean()),
        "read_rel_change_on_surviving_requests": float(
            np.linalg.norm(FC_e[~names_evict] - FC[~names_evict])
            / max(np.linalg.norm(FC[~names_evict]), 1e-12)),
        "corner_tuples_post": TC_e, "row1_tuples_post": T1_e,
        "pre": {"corners": _accepts(f0, CV, zC, oor_C, bitmap, ptable, zero_del),
                "row1": _accepts(f0, R1["values"], R1["dup"], oor_1, bitmap,
                                 ptable, zero_del)},
        "post": {"corners": _accepts(fe, CV, zC, oor_Ce, bm_post, ptable_e, zero_del),
                 "row1": _accepts(fe, R1["values"], R1["dup"], oor_1e, bm_post,
                                  ptable_e, zero_del)},
        "bitmap8_bits_changed": 1,
        "bitmap8_note": ("⭐ the 8-byte rule CAN follow a VALUE deletion: one bit is "
                         "cleared, by hand, from outside. ⛔ This is NOT the row where "
                         "the claim lives."),
    }
    for nm in ("field_geom", "bitmap8"):
        pre = evict_blk["pre"]["corners"][nm][names_evict].mean()
        post = evict_blk["post"]["corners"][nm][names_evict].mean()
        sur = evict_blk["post"]["corners"][nm][~names_evict].mean()
        log(f"    EVICT {nm:15s} accept on the deleted value {pre:.4f} -> {post:.4f} "
            f"  |   accept on survivors {sur:.4f}")

    # -- the negative control: the instrument is not blind ------------------
    fn = f0.copy()
    bn = fn.copy()
    fn.view("term_mu")[0] += np.float32(1e-3)
    d4 = byte_diff(bn, fn)

    meta = {
        "seed": int(seed), "condition": PINS["condition"],
        "byte_ledger": obj["byte_ledger"],
        "hole_pair": [[hf, hv], [hg, hu]],
        "evict": {"factor": ef, "value": ev_i},
        "n_corners": {"row3": len(c3), "row4": len(c4)},
        "n_corner_requests": int(CV.shape[0]),
        "negative_control_refit": {
            "what": "a single 1e-3 nudge to ONE row of the address book",
            "bytes_changed": int(d4["n_changed"]),
            "per_segment": d4["per_segment"], "would_fail_assert_no_refit": True},
        "judge": J.info, "feature_cache": obj["feature_cache"],
        "bitmap8_bytes": bitmap8_bytes(grid), "pair_table": pair_table_bytes(grid),
        "seconds": _now() - t0,
    }
    seed_dir(out, seed).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(art, hole=np.array(hole_blk, dtype=object),
                        evict=np.array(evict_blk, dtype=object))
    _jdump(meta, seed_dir(out, seed) / "deletion.json")
    log(f"  deletion(seed {seed}): done in {_now() - t0:.1f}s")
    return {"incomplete": False}


# ---------------------------------------------------------------------------
# 6b. Stage: the rival refusers the target file names
# ---------------------------------------------------------------------------


def rival_eligibility(seed: int, log: Callable = print) -> dict:
    """⛔ Can the full rival set enter a refusal row AT ALL?  Established mechanically.

    Two separate questions, and they are not the same one:

    1. **Is there a refusal mechanism?**  Every learner in spoke A's set is a generator:
       ``r(values) -> (features, out_of_range)``.  It answers every request whose values
       are in range.  It has no mechanism by which it could decline, so on a refusal row
       it is the ``accept_everything`` degenerate arm and scores **0** by construction.
    2. **Is the fit like-for-like?**  ⛔ No.  Measured by spoke A and re-checked here with
       ``load_rival(...).like_for_like``: every rival is fitted on ``montero_combination``,
       which contains BOTH held-out values.  On rows 5 and 7 a rival saw the value the
       field store never did.

    ⭐ A rival that cannot enter a row takes a **dash and a footnote**, and ⛔ a dash is
    never a win for us (w5-common §1).
    """
    out = {"seed": int(seed), "rivals": {}}
    try:
        from ..eval.rivals.w4_api import RivalRefusal, available, load_rival
    except Exception as exc:                                    # pragma: no cover
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out
    try:
        avail = available(int(seed))
    except Exception as exc:
        out["error"] = f"available() raised {type(exc).__name__}: {exc}"
        return out
    for name, info in (avail or {}).items():
        row = {"callable": bool(info.get("callable")),
               "param_count": info.get("param_count"),
               "fit_population": info.get("fit_population"),
               "population_caveat": info.get("population_caveat")}
        try:
            r = load_rival(int(seed), name)
            row["like_for_like"] = r.like_for_like(
                {"object_hue": [0.5], "shape": [0.0]})
            row["has_refusal_mechanism"] = False
            row["can_enter_a_refusal_row"] = False
            row["footnote"] = (
                "⛔ DASH. This rival is a GENERATOR: its call returns features plus an "
                "out_of_range flag and it has no mechanism by which it could decline a "
                "request. On a refusal row it IS `accept_everything` and scores 0 by "
                "construction. ⛔ A dash is never a win for us.")
        except RivalRefusal as exc:
            row["has_refusal_mechanism"] = None
            row["can_enter_a_refusal_row"] = False
            row["NOT_RUN"] = str(exc)
            row["footnote"] = (
                "⛔ DECLARED NOT RUN, never a null: spoke A does not expose this arm as "
                "a callable (its output is a stochastic draw, not a function of the "
                "request), so its energy cannot be evaluated on my requests.")
        except Exception as exc:                                # pragma: no cover
            row["error"] = f"{type(exc).__name__}: {exc}"
        out["rivals"][name] = row
        log(f"    rival {name:18s} refusal-row: "
            f"{'NOT RUN' if 'NOT_RUN' in row else 'dash (no refusal mechanism)'}")
    return out


def run_rivals(out: Path, seed: int, log: Callable = print) -> dict:
    """⛔ SCORED (run 3/3).  The rival refusers of the target file, on the same rows.

    ``table_absence``  the prototype table, refit LIKE-FOR-LIKE on my value-holdout pool,
                       refusing by absence.  ⚠ Predicted from the mechanism alone to be
                       identical to the 8-byte rule at 475,136 B.
    ``knn_density``    ⛔ the forbidden length scale, as a rival.
    ``lace_energy``    ⛔ NOT RUN -- established mechanically, never a null.
    """
    art = seed_dir(out, seed) / "rivals.npz"
    if art.exists():
        log(f"  rivals(seed {seed}): already on disk")
        return {"skipped": True}

    from ..core.gen_field import coupling_index, field_address, punch_hole

    t0 = _now()
    cond = load_condition()
    grid = cond["grid"]
    cards = [int(c) for c in grid.cardinalities]
    obj = build_field_arm(out, seed, log=log)
    f0 = obj["field"]
    pool = refuse_fit_pool(cond, seed)
    counts = prototype_table_counts(grid, pool)
    tau = float(PINS["knn_tau"])
    k = int(PINS["knn_k"])
    Q = query_requests(cond, grid)

    arms = {}
    for row_id, q in Q.items():
        vals, dup = q["values"], q["dup"]
        oor = field_address(f0, vals, grid.values)["out_of_range"]
        a_tab = table_absence_accept(vals, dup, oor, counts, grid)
        s_knn = knn_density_score(f0, vals, grid, k=k)
        a_knn = ~(np.asarray(dup, bool) | np.asarray(oor, bool)) & (s_knn <= tau)
        for name, acc in (("table_absence", a_tab), ("knn_density", a_knn)):
            arms.setdefault(name, {})[row_id] = {
                "accept": acc, "n": int(acc.size), "ids": q["ids"]}
        arms.setdefault("knn_density", {})[row_id]["score_p50"] = float(
            np.median(s_knn))
        log(f"  seed {seed} {row_id}: table_absence accept {a_tab.mean():.4f} | "
            f"knn_density accept {a_knn.mean():.4f} (score p50 {np.median(s_knn):.3f})")

    # -- the conjunction deletion, on the same two rivals -------------------
    c3, c4 = scored_corners(cond)
    corners = c3 + c4
    CV, _ = corner_values(grid, corners)
    ci = grid.factor_indices_of(np.concatenate(
        [np.asarray(c["rows"], np.int64) for c in corners]))
    (hf, hv), (hg, hu) = PINS["hole_pair"]
    names_hole = (ci[:, hf] == hv) & (ci[:, hg] == hu)
    fi_tr = grid.factor_indices_of(cond["train_rows"])
    held_pairs = set(map(tuple, np.unique(fi_tr[:, [hf, hg]], axis=0).tolist()))
    unseen_conj = np.array([tuple(r) not in held_pairs
                            for r in ci[:, [hf, hg]].tolist()], bool)
    zC = np.zeros(CV.shape[0], bool)
    oor_C = field_address(f0, CV, grid.values)["out_of_range"]
    fh = f0.copy()
    punch_hole(fh, hf, hv, hg, hu)
    hole = {}
    for name in ("table_absence", "knn_density"):
        if name == "table_absence":
            pre = table_absence_accept(CV, zC, oor_C, counts, grid)
            post = pre                       # ⛔ no pair cell exists to change
            bits = 0
        else:
            s_pre = knn_density_score(f0, CV, grid, k=k)
            s_post = knn_density_score(fh, CV, grid, k=k)
            pre = ~oor_C & (s_pre <= tau)
            post = ~oor_C & (s_post <= tau)
            bits = 0
        hole[name] = {
            "accept_deleted_pre": float(pre[names_hole].mean()),
            "accept_deleted_post": float(post[names_hole].mean()),
            "delta_reject": float(pre[names_hole].mean() - post[names_hole].mean()),
            "accept_unseen_conjunction_post": float(post[unseen_conj].mean()),
            "bits_changed_by_the_conjunction_deletion": bits,
        }
        log(f"    HOLE {name:15s} accept on the deleted conjunction "
            f"{pre[names_hole].mean():.4f} -> {post[names_hole].mean():.4f}   |   "
            f"accept on UNSEEN conjunctions {post[unseen_conj].mean():.4f}")

    elig = rival_eligibility(seed, log=log)
    meta = {
        "seed": int(seed),
        "table_absence_bytes": table_absence_bytes(grid),
        "table_absence_byte_multiple_over_bitmap8": float(
            table_absence_bytes(grid) / bitmap8_bytes(grid)),
        "knn": {"tau": tau, "k": k,
                "calibration": "the 99th percentile over the VAL field store's own "
                               "fit-pool requests, max over seeds, rounded up to 2 dp; "
                               "⛔ set on VALID DATA ONLY, before any invalid row was "
                               "scored with it",
                "length_scale": "⛔ the ATOMS' OWN median nearest-neighbour spacing -- "
                                "the length scale the admission gate died on. It is "
                                "here as a RIVAL and no claim of this hub uses it."},
        "lace_energy": {
            "NOT_RUN": True,
            "why": "spoke A does not expose LACE as a callable: its output is a "
                   "stochastic draw, not a function of the request, so an energy "
                   "cannot be evaluated on my requests. ⛔ A declared NOT-RUN is never "
                   "a null.",
            "byte_multiple_over_the_store": 95.1},
        "rival_eligibility": elig,
        "hole": hole,
        "n_corner_requests": int(CV.shape[0]),
        "coupling_index": int(coupling_index(cards, hf, hv, hg, hu)),
        "seconds": _now() - t0,
    }
    seed_dir(out, seed).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(art, arms=np.array(arms, dtype=object))
    _jdump(meta, seed_dir(out, seed) / "rivals.json")
    log(f"  rivals(seed {seed}): done in {_now() - t0:.1f}s")
    return {"incomplete": False}


# ---------------------------------------------------------------------------
# 7. CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--stage", required=True,
                   choices=("fit", "queries", "deletion", "rivals", "report",
                            "declare"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--max-seconds", type=float, default=14400.0)
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = _now()

    def log(msg):
        print(f"[{_now() - t0:7.1f}s] {msg}", flush=True)

    if args.stage == "declare":
        cond = load_condition()
        grid = cond["grid"]
        c3, c4 = scored_corners(cond)
        _jdump({
            "pins": {k: (list(v) if isinstance(v, tuple) else v)
                     for k, v in PINS.items()},
            "rules": {"field_geom": list(RULES_PRIMARY),
                      "field_geom_gap": list(RULES_GAP)},
            "mechanisms": MECHANISMS,
            "bitmap8_bytes": bitmap8_bytes(grid),
            "pair_table": pair_table_bytes(grid),
            "bitmap8_bits_changed_by_a_conjunction_deletion":
                bitmap8_bits_changed_by_a_conjunction_deletion(),
            "n_corners": {"row3": len(c3), "row4": len(c4)},
        }, out / "declaration.json")
        log("declaration written")
    elif args.stage == "fit":
        build_field_arm(out, args.seed, log=log)
    elif args.stage == "queries":
        run_queries(out, args.seed, max_seconds=args.max_seconds, log=log)
    elif args.stage == "deletion":
        run_deletion(out, args.seed, max_seconds=args.max_seconds, log=log)
    elif args.stage == "rivals":
        run_rivals(out, args.seed, log=log)
    elif args.stage == "report":
        from .exp_gen_refuse_report import build_report
        build_report(out, log=log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
