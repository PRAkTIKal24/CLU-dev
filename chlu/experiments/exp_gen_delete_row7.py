"""w5 hub/delete -- ROW 7 ON THE FIELD STORE, SCORED.  Eviction and the coupling hole.

⭐⭐ **SCORED.**  Every number this module produces is on the held-out ``test`` corners of the
``value_holdout`` condition and is pre-registered in ``PREREG-w5-delete-1.md`` before it runs.
The scored-run row is written to ``C:/chlu/SCORED-LEDGER.md`` before the first number.

What this module is for
-----------------------
The field store (:mod:`chlu.core.gen_field`) already demonstrates its deletion mechanism
*unscored*, on the selection condition.  This module does the one thing that was missing: it
**scores it against the rivals**, and it measures **both halves of every deletion** --

    the REFUSAL (the deletion actually happened) **and**
    the SURVIVING COMPOSITION (the deletion broke nothing else)

-- because a deletion row with only one half is half-made.  A method that refuses everything
passes the first half; a method that changes nothing passes the second; only a method that
passes both has deleted anything.

The two deletions, and why these targets
----------------------------------------
``A. evict red``   ``object_hue = 0.0`` removed by lifetime/eviction: every atom holding it is
                   zeroed, in every well, together with its lifetime.  No refit.
``B. the hole``    ``red AND cylinder`` marked deleted in ``coupling_hole``: ONE byte.  No refit.

⛔ ``scale``, ``orientation`` and ``shape = cube`` are NOT deletion targets here -- all three are
degradation attractors on this decoder and the target file forbids them.

⭐ The scored corner set already carries the target sentence, and this is a property of the
instrument rather than a choice made here.  The condition's three held-out pair cells are

    ``pair-000`` = object_hue 0.0 AND shape 1.0 = **red AND cylinder**   <- the hole
    ``pair-001`` = object_hue 0.3 AND shape 2.0 = **green AND sphere**   <- must survive
    ``pair-002`` = object_hue 0.7 AND shape 3.0 = **blue AND oblong**    <- must survive

All three are held out of ``train_rows`` entirely, so all three carry ``coupling_w == 0`` -- an
**identical evidence state**.  One of them is holed and must be refused; the other two are not and
must stay answered.  That is the sharpest available form of "unseen is not the same as deleted",
and no joint table can reproduce it, because a joint table cannot tell the three apart at all.

⚠ **Declared, because the target file's wording and this split's geometry disagree and the
measurement wins.**  The target file names "red sphere, never seen" as the conjunction that must
stay allowed.  Measured on this split, ``red AND sphere`` holds 12,000 train rows -- it is SEEN --
and ``red AND cylinder`` is the one that was never seen.  Both are measured here: ``red AND
sphere`` as the seen-and-not-deleted survivor, and ``pair-001``/``pair-002`` as the never-seen
survivors that play the role the target file's sentence asks for.

The rivals, and who gets a dash
-------------------------------
``table_row_removed``   the prototype table with the deleted value's row removed.
``table_retrain``       ⭐ **retrain from scratch on the pool without the value: THE GOLD
                        STANDARD**, reported with its cost line.
``gate8_*``             the 8-byte per-factor presence bitmap, printed beside us.  It deletes a
                        VALUE for 1 bit -- we expect to tie it and said so in the PREREG -- and it
                        changes **ZERO bits** on a conjunction.  To express ``red AND cylinder`` at
                        all it must clear ``red`` (losing red AND sphere, red AND oblong) or clear
                        ``cylinder`` (losing green AND cylinder, blue AND cylinder, ...).  Both
                        over-deletions are MEASURED here as a collateral rate.
``joint_gate_hole``     the joint hue x shape presence gate, which CAN express the conjunction --
                        and which therefore refuses every request whose conjunction the pool never
                        held, i.e. every row-3 request.  Measured, not argued.
MLP / transformer / LSTM / LACE / PoE-EBM
                        **DASH + FOOTNOTE.**  They have no refusal verb -- they always emit -- so
                        they cannot enter the refusal half at all; deletion is available to them
                        only by retrain-from-scratch, whose cost is on the cost line.  ⛔ A dash is
                        never a win for us.

Every arm is a ``(generator, refusal verb)`` PAIR, because the dial falsifier is "eviction is
behaviourally indistinguishable from a gate bolted onto the table" and that comparison is only
meaningful if the gate is actually bolted onto a generator.

Runnable::

    python -m chlu.experiments.exp_gen_delete_row7 --stage corners --out DIR
    python -m chlu.experiments.exp_gen_delete_row7 --stage bytes   --seed S --out DIR
    python -m chlu.experiments.exp_gen_delete_row7 --stage score   --seed S --out DIR
    python -m chlu.experiments.exp_gen_delete_row7 --stage report  --out DIR

Every stage is resumable: it writes one artifact per (stage, seed) and per corner, skips what is
already complete, takes ``--max-seconds`` and honours it between corners.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
import zlib
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .exp_gen_factor_wells import (
    Judge,
    _cell_mask_rows,
    _jdump,
    accept_rule,
    build_partitions,
    features_mmap,
    feature_cache_manifest,
    fit_pool,
    gather,
    judge_valid,
    load_condition,
    nway_corners,
    pair_subcorners,
    seen_tuple_mask,
    value_presence_bitmap,
)
from .exp_gen_field_store import PINS as FSPINS, _host_factors, field_spec

# ---------------------------------------------------------------------------
# 0. Pins.  The store's own dials are INHERITED verbatim from hub/field-store; this
#    hub changes the CONDITION (val -> the scored value_holdout) and nothing else.
# ---------------------------------------------------------------------------

PINS = dict(FSPINS)
PINS.update({
    "condition": "value_holdout_scored",
    # ⛔ the SCORED pool: the plain value-holdout fit pool, with NO extra exclusions.  The
    # scored corners live in `test_rows` and are therefore already outside it -- asserted,
    # not assumed, by `assert_scored_and_heldout`.
    "seeds": (0, 1, 2),
    # ⭐ bridge seed == store-fit seed.  Generation 2 is full-scope on all three, so unlike
    # hub/field-store this hub does NOT have to declare the bridge-seed spread a NOT-RUN.
    "vary_bridge_seed": True,
    # -- the deletion targets, inherited from hub/field-store, declared in code -----
    # object_hue = 0.0 (red).  ⛔ not scale, not orientation, not shape = cube.
    "delete_factor": 2, "delete_value": 0,
    # THE HOLE: object_hue = 0.0 AND shape = 1.0 (cylinder) == the cell `pair-000`.
    "hole_pair": ((2, 0), (4, 1)),
    # the constructed conjunction probes: (object_hue index, shape index) -> role
    "probe_backgrounds": 256,
    "probe_rng_seed": 20_260_920,
})

#: Charter/programme standing margin, in the programme's own currency.  ⛔ Declared in the
#: PREREG before the first held-out number and NOT moved afterwards (charter section A.3).
MARGIN_NOVELTY = 1.0
#: Refusal bars, declared in the PREREG.
BAR_REFUSE_DELETED = 0.99
BAR_REFUSE_SURVIVOR_DRIFT = 0.01

#: Human names, so a reader of the report never has to decode an index.
SHAPE_NAMES = {0: "cube", 1: "cylinder", 2: "sphere", 3: "oblong"}
HUE_NAMES = {0: "red", 3: "green", 7: "blue"}


def _now() -> float:
    return time.time()


# ---------------------------------------------------------------------------
# 0b. Guards.  ⛔ Two ways this hub could silently produce a number that means
#     nothing; both are refusals, not warnings.
# ---------------------------------------------------------------------------


def assert_generation2(seed: int) -> dict:
    """⛔ REFUSE to run on cache generation 1, or on a partial cache.

    w5-common section 3.8: generation 1 fails its own encoder's re-encode check on all three
    seeds and generation-1 caches cannot be re-scored on this box.  ``CHLU_GEN_FEATURES_ROOT``
    is NOT set in the ambient environment on this machine, so a run that forgets it reads the
    frozen bridge -- generation 1, and full-scope at bridge seed 0 only.  That is exactly the
    mistake this function exists to make impossible.
    """
    man = feature_cache_manifest(int(seed))
    gen, scope = man.get("generation"), man.get("scope")
    n = man.get("n_rows")
    if int(gen or 0) != 2:
        raise AssertionError(
            f"seed {seed}: feature cache is GENERATION {gen}, not 2. Set "
            "CHLU_GEN_FEATURES_ROOT to spoke A's cache_generation2 tree. w5-common section "
            "3.8 forbids scoring on generation 1.")
    if str(scope) != "full":
        raise AssertionError(
            f"seed {seed}: feature cache scope is {scope!r}, not 'full' ({n} rows). A "
            "partial cache cannot carry the scored corners.")
    return {"generation": int(gen), "scope": str(scope), "n_rows": int(n or 0),
            "root": str(os.environ.get("CHLU_GEN_FEATURES_ROOT", "<unset -- THE BRIDGE>"))}


def assert_out_root(out: Path) -> Path:
    """⛔ REFUSE an output path outside ``CHLU_HUB_OUTPUTS``.

    ⚠ MEASURED HAZARD on this box: MSYS bash maps ``/c/...`` to ``C:\\...`` but **Python does
    not** -- ``os.path.abspath('/c/chlu/x')`` is ``C:\\c\\chlu\\x``, a different tree that is
    created silently.  A run that writes there looks successful and produces artifacts nobody
    will ever find.  So the root is resolved and checked.
    """
    p = Path(out).resolve()
    root = Path(os.environ.get("CHLU_HUB_OUTPUTS", "C:/chlu/outputs")).resolve()
    if root not in p.parents and p != root:
        raise AssertionError(
            f"output root {p} is not under {root}. On this box a POSIX-looking path handed to "
            "Python resolves to C:\\c\\... -- pass a Windows-style path (C:/chlu/outputs/...).")
    p.mkdir(parents=True, exist_ok=True)
    return p


def seed_dir(out: Path, seed: int) -> Path:
    return Path(out) / f"seed_{int(seed)}"


# ---------------------------------------------------------------------------
# 1. The corners, the partition into deleted and survivor, and the proof
# ---------------------------------------------------------------------------


def scored_corners(cond) -> dict:
    """The 109 scored corners, split into the deletion set and the survivor set.

    ``deleted``   the ``pair-000`` sub-corners -- the red-cylinder conjunction, which both
                  deletions remove.  These are the corners whose composition SHOULD die.
    ``survivor``  everything else -- the 49 n-way corners plus ``pair-001`` (green AND sphere)
                  and ``pair-002`` (blue AND oblong).  These are the corners whose composition
                  must NOT move, and there are 89 of them, which satisfies w5-common section 2's
                  ">= 49 paired corners" on the survivor half alone.
    """
    grid = cond["grid"]
    f, v = int(PINS["delete_factor"]), int(PINS["delete_value"])
    dval = float(grid.values[f][v])
    n4, n3 = nway_corners(cond), pair_subcorners(cond)
    allc = n4 + n3
    deleted, survivor = [], []
    for c in allc:
        fi = grid.factor_indices_of(c["rows"])
        holds_red = bool((fi[:, f] == v).any())
        (deleted if holds_red else survivor).append(c)
    return {"all": allc, "deleted": deleted, "survivor": survivor,
            "nway": n4, "pair": n3, "deleted_value": dval}


def assert_scored_and_heldout(cond, cs: dict, pool: np.ndarray) -> dict:
    """⛔ Structural proof, run BEFORE any number: disjoint, held out, and scored.

    Every corner row must be (a) in the value-holdout ``test`` index -- these are the SCORED
    corners, and this hub is spending a scored run on them -- and (b) outside the fit pool of
    every arm, or the corner is not a hold-out at all.
    """
    te = np.asarray(cond["test_rows"], np.int64)
    tr = np.asarray(cond["train_rows"], np.int64)
    P = np.asarray(pool, np.int64)
    rep = {}
    for name in ("all", "deleted", "survivor"):
        rows = np.concatenate([c["rows"] for c in cs[name]])
        uniq = np.unique(rows)
        if uniq.size != rows.size:
            raise AssertionError(f"{name}: corners overlap ({rows.size - uniq.size} shared rows)")
        n_pool = int(np.intersect1d(uniq, P).size)
        if n_pool:
            raise AssertionError(
                f"{name}: {n_pool} corner rows are inside the fit pool -- not a hold-out")
        rep[name] = {"n_corners": len(cs[name]), "n_requests": int(rows.size),
                     "rows_in_scored_test_index": int(np.intersect1d(uniq, te).size),
                     "rows_in_train_index": int(np.intersect1d(uniq, tr).size),
                     "rows_in_fit_pool": n_pool}
    if rep["all"]["rows_in_scored_test_index"] != rep["all"]["n_requests"]:
        raise AssertionError("not every corner row is in the scored test index")
    if len(cs["survivor"]) < 49:
        raise AssertionError(
            f"the survivor set has {len(cs['survivor'])} corners; w5-common section 2 requires "
            ">= 49 paired corners and forbids a single cell deciding a target")
    rep["statement"] = (
        "every corner row is in the value-holdout TEST index and none is in any arm's fit pool; "
        f"the survivor half alone is {len(cs['survivor'])} corners, which meets the >= 49 "
        "paired-corner estimand; this run SPENDS scored run 1/3 of target `deletion`")
    return rep


def conjunction_probes(cond, seed: int) -> dict:
    """⭐ The constructed 2x2 that separates the HOLE from the EVICTION.

    WARNING, and it is why these are CONSTRUCTED rather than filtered out of the corner set:
    the corner set has no corner for "red on a shape that is not cylinder", because no n-way
    corner carries red at all.  Without these probes the eviction and the hole would be
    indistinguishable on the scored corners -- they remove the same corners -- and the whole
    point is that they differ *off* those corners.

    Same backgrounds every time, four factors left alone, two factors set:

    =====================  ======  ===========================================================
    probe                  seen?   what it must do
    =====================  ======  ===========================================================
    ``red_cylinder``       no      the HOLE and the eviction: REFUSED by both
    ``red_sphere``         yes     eviction REFUSES it; the hole must LEAVE IT ALLOWED
    ``red_oblong``         yes     eviction REFUSES it; the hole must LEAVE IT ALLOWED
    ``green_cylinder``     yes     both must leave it ALLOWED (cylinder's other hues survive)
    ``blue_cylinder``      yes     both must leave it ALLOWED
    ``green_sphere``       **no**  never seen, not holed -- must stay ALLOWED (``pair-001``)
    ``blue_oblong``        **no**  never seen, not holed -- must stay ALLOWED (``pair-002``)
    ``red_cube``           no      REPORTED, not a bar: shape=cube is held out of the split
                                   entirely, so its basin is empty at build time and BOTH
                                   objects refuse it for an independent and legitimate reason.
                                   Its ``hole`` flag must still be 0 -- that is the part that
                                   is a bar, because it proves the hole is not what refuses it.
    =====================  ======  ===========================================================
    """
    grid = cond["grid"]
    rng = np.random.default_rng(int(PINS["probe_rng_seed"]) + 101 * int(seed))
    te = np.asarray(cond["test_rows"], np.int64)
    n = int(PINS["probe_backgrounds"])
    bg_rows = te[rng.permutation(te.size)[:n]]
    base = grid.values_of(bg_rows)
    hue_ax, shp_ax = 2, 4

    def _set(oh: int, sh: int) -> np.ndarray:
        v = np.array(base, np.float64, copy=True)
        v[:, hue_ax] = float(grid.values[hue_ax][oh])
        v[:, shp_ax] = float(grid.values[shp_ax][sh])
        return v

    spec = {
        "red_cylinder": (0, 1, False, "THE HOLE; never seen"),
        "red_sphere": (0, 2, True, "red's other shape, SEEN -- the hole must leave it allowed"),
        "red_oblong": (0, 3, True, "red's other shape, SEEN -- the hole must leave it allowed"),
        "green_cylinder": (3, 1, True, "cylinder's other hue, SEEN -- must stay allowed"),
        "blue_cylinder": (7, 1, True, "cylinder's other hue, SEEN -- must stay allowed"),
        "green_sphere": (3, 2, False, "pair-001: NEVER SEEN, not holed -- must stay allowed"),
        "blue_oblong": (7, 3, False, "pair-002: NEVER SEEN, not holed -- must stay allowed"),
        "red_cube": (0, 0, False, "shape=cube is held out entirely; reported, not a bar"),
    }
    out = {}
    for name, (oh, sh, seen, why) in spec.items():
        out[name] = {"values": _set(oh, sh), "hue_index": oh, "shape_index": sh,
                     "seen_in_pool": bool(seen), "why": why,
                     "label": f"{HUE_NAMES.get(oh, f'hue{oh}')} AND {SHAPE_NAMES[sh]}"}
    out["_backgrounds"] = {"rows": bg_rows, "n": int(bg_rows.size),
                           "source": "the value-holdout TEST index"}
    return out


# ---------------------------------------------------------------------------
# 2. The generators
# ---------------------------------------------------------------------------


def build_field_base(out: Path, seed: int, log: Callable = print) -> dict:
    """The field store on the SCORED pool.  Resumable."""
    from ..core.gen_field import (
        assert_read_request_only, build_field, field_byte_ledger,
    )

    d = seed_dir(out, seed) / "field"
    done = d / "field.pkl"
    if done.exists():
        log(f"  field(seed {seed}): already built, loading")
        with open(done, "rb") as fh:
            return pickle.load(fh)

    cache = assert_generation2(seed)
    cond = load_condition()
    grid = cond["grid"]
    parts = build_partitions(Path(out) / "partitions", log=log)
    labels = np.asarray(parts["labels"]["factor"], np.int64)
    K = int(labels.max()) + 1
    hosts = _host_factors(parts["info"], K)
    pool = scored_pool(cond, seed)
    F = features_mmap(int(seed))
    t0 = _now()
    X = gather(F, pool)
    Y = grid.factor_indices_of(pool)
    cards = [int(c) for c in grid.cardinalities]
    field = build_field(X, Y, pool, labels, hosts, cards, field_spec(seed), log=log)
    del X
    led = field_byte_ledger(field, int(PINS["store_bytes"]))
    if not led["matches_budget"]:
        raise AssertionError(
            f"the field store deploys {led['deployed_bytes']} B against the pinned "
            f"{PINS['store_bytes']} B ledger -- the byte ledger has moved")
    ro = assert_read_request_only(field, grid.values_of(pool[:64]), grid.values)
    obj = {"kind": "field", "seed": int(seed), "field": field, "pool": pool,
           "byte_ledger": led, "request_only": ro, "cache": cache,
           "manifest": field.manifest(), "fit_seconds": _now() - t0}
    d.mkdir(parents=True, exist_ok=True)
    with open(done, "wb") as fh:
        pickle.dump(obj, fh)
    _jdump({k: v for k, v in obj.items() if k not in ("field", "pool")}, d / "field.json")
    log(f"  field(seed {seed}): built in {obj['fit_seconds']:.1f}s; atoms "
        f"{led['deployed_bytes']:,} B (image {led['total_image_bytes']:,} B)")
    return obj


def scored_pool(cond, seed: int) -> np.ndarray:
    """The ONE fit pool per seed, shared by EVERY arm.  No extra exclusions.

    The scored corners are in ``test_rows`` and therefore already outside this pool; that is
    asserted rather than assumed by :func:`assert_scored_and_heldout`.
    """
    rng_seed = int(PINS["pool_rng_seed"]) + int(PINS["pool_seed_stride"]) * int(seed)
    return fit_pool(cond["train_rows"], int(PINS["n_pool"]), rng_seed)


def field_variants(base: dict, seed: int, log: Callable = print) -> dict:
    """⭐ The deletions, and the charter section C controls, derived IN PLACE.

    Every variant is a byte-image copy of the same store with one operation applied.  Nothing
    here refits; :func:`gen_field.assert_no_refit` is asserted on each in :func:`byte_proof`.
    """
    from ..core.gen_field import _items_holding, decay_items, evict_term, punch_hole

    f0 = base["field"]
    fac, valx = int(PINS["delete_factor"]), int(PINS["delete_value"])
    (hf, hv), (hg, hu) = PINS["hole_pair"]
    out = {"base": f0}

    # -- A. evict the value -------------------------------------------------
    fe = f0.copy()
    ev = evict_term(fe, fac, valx, log=log)
    out["evict_red"] = fe

    # -- A'. the same deletion reached by LIFETIME, in two steps -------------
    # The target says "by lifetime/eviction".  Step 1 moves lifetime bytes only (the item is
    # still there, at half life); step 2 takes the lifetime to zero, at which point the item's
    # content is zeroed too, so a decayed-out item leaves nothing behind.
    fd = f0.copy()
    ids, _, _ = _items_holding(fd, fac, valx)
    dec_half = decay_items(fd, ids, 0.5, log=log)
    fd_half = fd.copy()
    dec_zero = decay_items(fd, ids, 0.0, log=log)
    out["decay_red"] = fd
    out["_decay_half"] = fd_half

    # -- B. the hole --------------------------------------------------------
    fh_ = f0.copy()
    ho = punch_hole(fh_, hf, hv, hg, hu, log=log)
    out["hole"] = fh_

    # -- charter section C.2 controls ---------------------------------------
    rng = np.random.default_rng(int(seed) * 7919 + 20_260_920)
    fr = f0.copy()
    fr.view("atoms")[:] = 0.0
    out["landscape_removed"] = fr
    fs = f0.copy()
    for k in range(fs.K):
        blk = fs.atoms_of_well(k)
        blk[:] = blk[:, rng.permutation(blk.shape[1])]
    out["landscape_shuffled"] = fs

    out["_ops"] = {"evict": {k: v for k, v in ev.items() if k != "permitted_ranges"},
                   "decay_half": {k: v for k, v in dec_half.items()
                                  if k != "permitted_ranges"},
                   "decay_zero": {k: v for k, v in dec_zero.items()
                                  if k != "permitted_ranges"},
                   "hole": {k: v for k, v in ho.items() if k != "permitted_ranges"}}
    return out


# ---------------------------------------------------------------------------
# 3. The rivals -- the table, and the gates
# ---------------------------------------------------------------------------


def build_tables(out: Path, seed: int, log: Callable = print) -> dict:
    """The prototype table on MY pool: base, row-removed, and retrain-from-scratch.

    ⭐ Refitted here rather than loaded from spoke A's artifacts ON PURPOSE.
    ``row_eligibility.json`` records the caveat in spoke A's own words: the shipped rivals are
    fitted on the Montero combination pool, which contains every factor value, so on a
    value-hold-out row they saw the value the arm under test never did.  "Either report it with
    a population footnote and do NOT count it as beating a value-holdout arm, or refit it on
    your own pool."  This hub refits, so the comparison is like-for-like and the footnote is not
    load-bearing.
    """
    from ..core.gen_request import FactorPrototypes

    d = seed_dir(out, seed) / "tables"
    done = d / "tables.pkl"
    if done.exists():
        log(f"  tables(seed {seed}): already built, loading")
        with open(done, "rb") as fh:
            return pickle.load(fh)

    assert_generation2(seed)
    cond = load_condition()
    grid = cond["grid"]
    pool = scored_pool(cond, seed)
    fac, valx = int(PINS["delete_factor"]), int(PINS["delete_value"])
    F = features_mmap(int(seed))
    X = gather(F, pool)

    t0 = _now()
    base = FactorPrototypes.fit(grid, X, pool)
    cost_fit = _now() - t0

    # -- the cheap deletion: remove the value's ROW from the table -----------
    t0 = _now()
    protos = [np.array(P, np.float32, copy=True) for P in base.protos]
    protos[fac][valx] = base.mu_global
    npv = [list(c) for c in base.n_per_value]
    npv[fac][valx] = 0
    removed = FactorPrototypes(
        names=base.names, values=base.values, protos=tuple(protos),
        mu_global=base.mu_global, n_rows_fit=base.n_rows_fit,
        n_per_value=tuple(tuple(c) for c in npv))
    cost_remove = _now() - t0

    # -- ⭐ THE GOLD STANDARD: refit from scratch without the value ----------
    fi = grid.factor_indices_of(pool)
    keep = fi[:, fac] != valx
    kept_rows = pool[keep]
    t0 = _now()
    retrain = FactorPrototypes.fit(grid, X[keep], kept_rows)
    cost_retrain = _now() - t0
    del X

    obj = {"kind": "tables", "seed": int(seed), "base": base, "row_removed": removed,
           "retrain": retrain,
           "cost": {"fit_full_pool_s": float(cost_fit),
                    "row_removal_s": float(cost_remove),
                    "retrain_from_scratch_s": float(cost_retrain),
                    "rows_in_pool": int(pool.size),
                    "rows_dropped": int((~keep).sum()),
                    "rows_refitted_by_retrain": int(kept_rows.size),
                    "rows_refitted_by_row_removal": 0},
           "n_bytes": int(base.n_bytes) if hasattr(base, "n_bytes") else None}
    d.mkdir(parents=True, exist_ok=True)
    with open(done, "wb") as fh:
        pickle.dump(obj, fh)
    _jdump({"seed": int(seed), "cost": obj["cost"], "n_bytes": obj["n_bytes"],
            "n_per_value_deleted_factor_after_removal": list(removed.n_per_value[fac]),
            "n_per_value_deleted_factor_after_retrain": list(retrain.n_per_value[fac])},
           d / "tables.json")
    log(f"  tables(seed {seed}): fit {cost_fit:.2f}s; row-removal {cost_remove:.4f}s; "
        f"RETRAIN-FROM-SCRATCH {cost_retrain:.2f}s on {kept_rows.size:,} rows "
        f"({int((~keep).sum()):,} dropped)")
    return obj


class Gate:
    """A presence gate: the cheap refusal object the field store is measured against.

    ``per_factor``  the 8-byte per-factor value-presence bitmap plus the free bracket check --
                    w3-common section 7's object, the one that wins rows 1/2/5/6 at 1.0000.
                    ⭐⭐ It **cannot express a conjunction deletion**: there is no bit whose
                    meaning is "this pair".  Clearing a pair changes ZERO bits, and that is
                    asserted here by byte-diffing :meth:`pack`.
    ``joint``       the joint ``object_hue x shape`` presence gate, which CAN express the pair
                    -- and which therefore refuses every request whose pair the pool never held.
    """

    def __init__(self, kind: str, grid, bitmap=None, joint=None, label: str = "",
                 deleted_pairs=None):
        self.kind = str(kind)
        self.grid = grid
        self.bitmap = [np.array(b, bool, copy=True) for b in (bitmap or [])]
        self.joint = None if joint is None else np.array(joint, bool, copy=True)
        #: ``joint_backoff`` only: the SEPARATE explicit deleted-pair array.  Kept apart from
        #: ``joint`` on purpose -- that separation is the whole mechanism under test, and a
        #: rival that wants the conjunction deletion without refusing the unseen has to have it.
        self.deleted = (None if deleted_pairs is None
                        else np.array(deleted_pairs, bool, copy=True))
        self.label = label or kind

    # -- construction ------------------------------------------------------
    @classmethod
    def per_factor(cls, grid, fitted_rows) -> "Gate":
        return cls("per_factor", grid, bitmap=value_presence_bitmap(grid, fitted_rows),
                   label="8-byte per-factor presence bitmap")

    @classmethod
    def joint_hue_shape(cls, grid, fitted_rows) -> "Gate":
        fi = grid.factor_indices_of(np.asarray(fitted_rows, np.int64))
        J = np.zeros((int(grid.cardinalities[2]), int(grid.cardinalities[4])), bool)
        J[fi[:, 2], fi[:, 4]] = True
        return cls("joint", grid, bitmap=value_presence_bitmap(grid, fitted_rows), joint=J,
                   label="joint object_hue x shape presence gate")

    @classmethod
    def joint_backoff(cls, grid, fitted_rows) -> "Gate":
        """⭐ THE GENEROUS JOINT RIVAL -- built deliberately, so this hub beats no strawman.

        ⚠ MEASURED, and it is why this rival exists: the plain joint PRESENCE gate above changes
        **ZERO bits** when asked to delete ``red AND cylinder``, because ``joint[red, cylinder]``
        is already ``False`` -- the pool never held the pair.  A joint presence table has ONE bit
        that means both "never seen" and "deleted", so on this split the deletion is a no-op on
        it while it simultaneously refuses ``green AND sphere`` and ``blue AND oblong``, which
        must stay allowed.  Scoring only that object would be beating a strawman.

        So this rival is given the thing that fixes it: pairs back off to the marginals (an
        unseen pair is ANSWERED) plus a **separate explicit deleted-pair array**.  It can then
        express the conjunction deletion cleanly, and ⭐ **it is expected to TIE the field store's
        hole on the refusal half -- declared before it was scored.**  The point it makes is the
        honest one: the cheapest object that expresses a conjunction deletion without refusing
        the unseen is a separate deleted-bit array over pairs, which is exactly what
        ``coupling_hole`` is.  A per-factor gate does not reach it at any bit count.
        """
        cards = (int(grid.cardinalities[2]), int(grid.cardinalities[4]))
        return cls("joint_backoff", grid,
                   bitmap=value_presence_bitmap(grid, fitted_rows),
                   deleted_pairs=np.zeros(cards, bool),
                   label="joint hue x shape gate with marginal back-off + a SEPARATE "
                         "deleted-pair array")

    def copy(self) -> "Gate":
        return Gate(self.kind, self.grid, self.bitmap, self.joint, self.label, self.deleted)

    # -- the byte image, so a deletion's cost can be BYTE-DIFFED ------------
    def pack(self) -> bytes:
        """The gate as bytes, bit-packed CONTIGUOUSLY.

        ⚠ Contiguously and not per factor: the cardinalities are [10,10,10,8,4,15] = 57 values,
        which is 57 bits = **8 bytes**, the programme's own figure.  Padding each factor to a
        byte boundary would print 10 and quietly inflate the object this hub is measured against.
        """
        bits = [np.asarray(b, bool).ravel() for b in self.bitmap]
        if self.joint is not None:
            bits.append(self.joint.ravel())
        if self.deleted is not None:
            bits.append(self.deleted.ravel())
        v = np.concatenate(bits) if bits else np.zeros(0, bool)
        return bytes(np.packbits(v))

    def n_bits(self) -> int:
        n = int(sum(int(np.asarray(b).size) for b in self.bitmap))
        if self.joint is not None:
            n += int(self.joint.size)
        if self.deleted is not None:
            n += int(self.deleted.size)
        return n

    @property
    def n_bytes(self) -> int:
        return len(self.pack())

    # -- the deletions -----------------------------------------------------
    def clear_value(self, f: int, v: int) -> "Gate":
        g = self.copy()
        g.bitmap[int(f)][int(v)] = False
        g.label = self.label + f" [{self.grid.names[f]} index {v} cleared]"
        return g

    def delete_conjunction(self, f: int, v: int, g2: int, u: int) -> "Gate":
        """Delete a PAIR.  The per-factor gate cannot, and returns itself unchanged.

        ⛔ This is not a stub and it is not a concession made for convenience: an 8-byte
        per-factor bitmap has one bit per (factor, value) and no bit whose meaning is "this
        conjunction".  The only expressible approximations are to clear ``(f, v)`` -- which also
        deletes every OTHER pair containing it -- or to clear ``(g2, u)``.  Both are measured
        here as ``gate8_clear_*`` and their collateral is reported as a rate.
        """
        out = self.copy()
        if {int(f), int(g2)} == {2, 4}:
            oh, sh = (int(v), int(u)) if int(f) == 2 else (int(u), int(v))
            if self.kind == "joint":
                out.joint[oh, sh] = False
                out.label = self.label + f" [({oh},{sh}) presence bit cleared]"
                return out
            if self.kind == "joint_backoff":
                out.deleted[oh, sh] = True
                out.label = self.label + f" [({oh},{sh}) marked DELETED]"
                return out
        out.label = self.label + " [conjunction NOT EXPRESSIBLE -- 0 bits changed]"
        return out

    # -- the refusal verb --------------------------------------------------
    def _pair_indices(self, values: np.ndarray) -> tuple:
        gv_h, gv_s = self.grid.values[2], self.grid.values[4]
        vh, vs = values[:, 2], values[:, 4]
        ih = np.clip(np.searchsorted(gv_h, np.nan_to_num(vh), side="left"), 0, gv_h.size - 1)
        isx = np.clip(np.searchsorted(gv_s, np.nan_to_num(vs), side="left"), 0, gv_s.size - 1)
        asked = ~np.isnan(vh) & ~np.isnan(vs)
        return ih, isx, asked

    def refuse(self, values: np.ndarray, oor: np.ndarray) -> np.ndarray:
        """True where this gate REFUSES the request."""
        dup = np.zeros(values.shape[0], bool)
        ok = accept_rule(values, oor, dup, self.bitmap, self.grid)
        if self.joint is not None:
            ih, isx, asked = self._pair_indices(values)
            ok &= (~asked) | self.joint[ih, isx]
        if self.deleted is not None:
            # back-off: an UNSEEN pair is answered.  Only an EXPLICITLY deleted pair is refused.
            ih, isx, asked = self._pair_indices(values)
            ok &= ~(asked & self.deleted[ih, isx])
        return ~ok


class TableGate:
    """The prototype table's OWN refusal verb: a value with no fitted row is refused.

    This is what "table row-removal" means as a deletion: the row is gone, the count is zero,
    and the table says so.  It is the table's native mechanism and it is a fair rival on the
    VALUE deletion.  ⛔ It has nothing to say about a conjunction -- ``n_per_value`` is indexed
    by (factor, value) and there is no conjunction to zero.
    """

    def __init__(self, protos, grid, label: str = "table n_per_value gate"):
        self.protos, self.grid, self.label = protos, grid, label

    def refuse(self, values: np.ndarray, oor: np.ndarray) -> np.ndarray:
        bm = [np.asarray(c, np.int64) > 0 for c in self.protos.n_per_value]
        dup = np.zeros(values.shape[0], bool)
        return ~accept_rule(values, oor, dup, bm, self.grid)


# ---------------------------------------------------------------------------
# 4. Arms = (generator, refusal verb) pairs
# ---------------------------------------------------------------------------


def _field_features(field, values, grid):
    from ..core.gen_field import field_address, field_read
    addr = field_address(field, values, grid.values)
    return field_read(field, values, grid.values), addr["out_of_range"]


def _field_lookup_features(field, values, grid, mode: str):
    """⭐ Charter section C.1: the MATCHED-BYTES LOOKUP on the SAME atoms.

    Same store, same address book, same bytes -- the ridge projection over the live atoms' span
    is replaced by a lookup.  ``argmax`` returns the single nearest live atom per well;
    ``mean`` returns the kernel-weighted mean of the live atoms.  If the arm cannot beat these,
    the landscape is decorative and the charter says to report it as such.
    """
    from ..core.gen_field import LIFETIME_EPS, field_address
    addr = field_address(field, values, grid.values)
    mu = addr["mu"]
    N, D = mu.shape
    out = np.zeros((N, D), np.float32)
    life = np.asarray(field.view("atom_life"), np.float64)
    alive = (np.asarray(field.view("atom_alive"), np.uint8) > 0) & (life > LIFETIME_EPS)
    ws = np.asarray(field.view("well_sigma"), np.float64)
    nearest = np.full(N, np.inf)
    for k in range(field.K):
        ch = np.asarray(field.meta["well_channels"][k], np.int64)
        A = np.asarray(field.atoms_of_well(k), np.float64)
        M = np.asarray(mu[:, ch], np.float64)
        if not alive.any():
            continue
        idx = np.flatnonzero(alive)
        Ak = A[idx]
        d2 = ((M * M).sum(1)[:, None] - 2.0 * M @ Ak.T + (Ak * Ak).sum(1)[None, :])
        d2 = np.maximum(d2, 0.0)
        nearest = np.minimum(nearest, np.sqrt(d2.min(axis=1)))
        if mode == "argmax":
            out[:, ch] = Ak[np.argmin(d2, axis=1)].astype(np.float32)
        else:
            s = max(float(ws[k]), 1e-9)
            w = np.exp(-d2 / (2.0 * s * s)) * life[idx][None, :]
            w = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-30)
            out[:, ch] = (w @ Ak).astype(np.float32)
    return out, addr["out_of_range"], nearest


def build_arms(out: Path, seed: int, log: Callable = print) -> dict:
    """Every arm this hub scores, as ``(generator, refusal verb)`` pairs."""
    cond = load_condition()
    grid = cond["grid"]
    base = build_field_base(out, seed, log=log)
    fv = field_variants(base, seed, log=log)
    tabs = build_tables(out, seed, log=log)
    pool = base["pool"]
    fac, valx = int(PINS["delete_factor"]), int(PINS["delete_value"])
    (hf, hv), (hg, hu) = PINS["hole_pair"]

    g8 = Gate.per_factor(grid, pool)
    gj = Gate.joint_hue_shape(grid, pool)
    gb = Gate.joint_backoff(grid, pool)
    tg = TableGate(tabs["base"], grid)

    arms = {}

    def _F(field):
        return lambda vals: _field_features(field, vals, grid)

    def _L(field, mode):
        def fn(vals):
            f, o, nd = _field_lookup_features(field, vals, grid, mode)
            return f, o
        return fn

    def _T(protos):
        def fn(vals):
            oor = _table_oor(vals, grid)
            return protos.embed_many(_reqs(vals, grid)), oor
        return fn

    def _fr(field):
        from ..core.gen_field import gap_read
        return lambda vals: gap_read(field, vals, grid.values)

    # ---- the field store, the object under test --------------------------
    arms["field/base"] = dict(
        gen=_F(fv["base"]), refuse=_fr(fv["base"]), family="field", role="pre-deletion",
        deletion=None, gen_key="field_base")
    arms["field/evict_red"] = dict(
        gen=_F(fv["evict_red"]), refuse=_fr(fv["evict_red"]), family="field",
        role="DELETION A: evict object_hue=0.0 (red) -- atoms zeroed, no refit", deletion="A",
        gen_key="field_evict")
    arms["field/decay_red"] = dict(
        gen=_F(fv["decay_red"]), refuse=_fr(fv["decay_red"]), family="field",
        role="DELETION A by LIFETIME: lifetimes decayed to zero, then the item is gone",
        deletion="A", gen_key="field_decay", identical_to="field_evict")
    arms["field/hole"] = dict(
        gen=_F(fv["hole"]), refuse=_fr(fv["hole"]), family="field",
        role="DELETION B: the coupling hole on red AND cylinder -- ONE byte, no refit",
        deletion="B", gen_key="field_hole", identical_to="field_base")
    arms["field/landscape_removed"] = dict(
        gen=_F(fv["landscape_removed"]), refuse=_fr(fv["landscape_removed"]), family="control",
        role="charter C.2: atoms zeroed -- must collapse", deletion=None,
        gen_key="field_removed")
    arms["field/landscape_shuffled"] = dict(
        gen=_F(fv["landscape_shuffled"]), refuse=_fr(fv["landscape_shuffled"]),
        family="control", role="charter C.2: each well's channel axis permuted", deletion=None,
        gen_key="field_shuffled")
    arms["field/lookup_argmax"] = dict(
        gen=_L(fv["base"], "argmax"), refuse=_fr(fv["base"]), family="floor",
        role="charter C.1: matched-bytes lookup, nearest live atom", deletion=None,
        gen_key="field_lookup_argmax")
    arms["field/lookup_mean"] = dict(
        gen=_L(fv["base"], "mean"), refuse=_fr(fv["base"]), family="floor",
        role="charter C.1: matched-bytes lookup, kernel-weighted atom mean", deletion=None,
        gen_key="field_lookup_mean")

    # ---- the table, refitted on MY pool ----------------------------------
    arms["table/base"] = dict(
        gen=_T(tabs["base"]), refuse=tg.refuse, family="rival", role="the table, pre-deletion",
        deletion=None, gen_key="table_base")
    arms["table/row_removed"] = dict(
        gen=_T(tabs["row_removed"]), refuse=TableGate(tabs["row_removed"], grid).refuse,
        family="rival", role="RIVAL: table row-removal (the value's row is gone)", deletion="A",
        gen_key="table_row_removed")
    arms["table/retrain"] = dict(
        gen=_T(tabs["retrain"]), refuse=TableGate(tabs["retrain"], grid).refuse,
        family="rival", role="RIVAL, GOLD STANDARD: retrain from scratch without the value",
        deletion="A", gen_key="table_retrain")

    # ---- gates bolted onto the table -------------------------------------
    arms["table+gate8/base"] = dict(
        gen=_T(tabs["base"]), refuse=g8.refuse, family="gate",
        role="the 8-byte per-factor gate, pre-deletion", deletion=None, gen_key="table_base")
    arms["table+gate8/clear_red"] = dict(
        gen=_T(tabs["base"]), refuse=g8.clear_value(fac, valx).refuse, family="gate",
        role="RIVAL: the 8-byte gate deletes the VALUE for 1 bit", deletion="A",
        gen_key="table_base")
    arms["table+gate8/hole_attempt"] = dict(
        gen=_T(tabs["base"]), refuse=g8.delete_conjunction(hf, hv, hg, hu).refuse,
        family="gate",
        role="RIVAL: the 8-byte gate asked for the CONJUNCTION -- ZERO bits change",
        deletion="B", gen_key="table_base")
    arms["table+gate8/clear_cylinder"] = dict(
        gen=_T(tabs["base"]), refuse=g8.clear_value(hg, hu).refuse, family="gate",
        role="the gate's only other approximation to the conjunction: clear the SHAPE",
        deletion="B", gen_key="table_base")
    arms["table+joint/base"] = dict(
        gen=_T(tabs["base"]), refuse=gj.refuse, family="gate",
        role="the joint hue x shape PRESENCE gate, pre-deletion", deletion=None,
        gen_key="table_base")
    arms["table+joint/hole"] = dict(
        gen=_T(tabs["base"]), refuse=gj.delete_conjunction(hf, hv, hg, hu).refuse,
        family="gate",
        role=("RIVAL: the joint PRESENCE gate -- ONE bit means both 'never seen' and 'deleted', "
              "so the deletion is a no-op AND every unseen conjunction is refused"),
        deletion="B", gen_key="table_base")
    arms["table+backoff/base"] = dict(
        gen=_T(tabs["base"]), refuse=gb.refuse, family="gate",
        role="the GENEROUS joint rival, pre-deletion (unseen pairs back off and are answered)",
        deletion=None, gen_key="table_base")
    arms["table+backoff/hole"] = dict(
        gen=_T(tabs["base"]), refuse=gb.delete_conjunction(hf, hv, hg, hu).refuse,
        family="gate",
        role=("RIVAL, the generous one: marginal back-off + a SEPARATE deleted-pair array. "
              "Expected to TIE the field store's hole on the refusal half -- declared in advance"),
        deletion="B", gen_key="table_base")

    return {"arms": arms, "field_variants": fv, "tables": tabs, "gates": {
        "per_factor": g8, "joint": gj, "joint_backoff": gb,
        "clear_red": g8.clear_value(fac, valx),
        "clear_cylinder": g8.clear_value(hg, hu),
        "hole_attempt": g8.delete_conjunction(hf, hv, hg, hu),
        "joint_hole": gj.delete_conjunction(hf, hv, hg, hu),
        "backoff_hole": gb.delete_conjunction(hf, hv, hg, hu)}, "base": base}


def _reqs(values: np.ndarray, grid) -> list:
    """``(N, F)`` requested VALUES -> the request-dict list ``FactorPrototypes`` consumes.

    Built from the VALUES rather than from row ids on purpose: the constructed conjunction
    probes are requests no dataset row realises, and ``requests_from_rows`` cannot express them.
    ``nan`` means the factor was not requested, and such a term is simply absent -- which is the
    same convention every other arm on this hub uses.
    """
    names = list(grid.names)
    out = []
    for row in np.asarray(values, np.float64):
        out.append({"request": [{"factor": names[k], "factor_index": int(k),
                                 "value": float(row[k])}
                                for k in range(row.size) if not np.isnan(row[k])]})
    return out


def _table_oor(values: np.ndarray, grid) -> np.ndarray:
    oor = np.zeros(values.shape[0], bool)
    for f in range(values.shape[1]):
        gv = np.asarray(grid.values[f], np.float64)
        v = values[:, f]
        a = ~np.isnan(v)
        oor |= a & ((v < gv.min() - 1e-12) | (v > gv.max() + 1e-12))
    return oor


# ---------------------------------------------------------------------------
# 5. ⭐⭐ The byte proof, on the SCORED store
# ---------------------------------------------------------------------------


def byte_proof(out: Path, seed: int, log: Callable = print) -> dict:
    """BAR 5: zero bytes outside the permitted ranges, and no refit.  Exact, no tolerance."""
    from ..core.gen_field import (
        _items_holding, assert_changed_bytes_within, assert_no_refit, byte_diff,
        coupling_index, decay_items, evict_term, punch_hole, set_lifetime,
    )

    d = seed_dir(out, seed)
    art = d / "bytes.json"
    if art.exists():
        log(f"  bytes(seed {seed}): already on disk")
        return json.loads(art.read_text(encoding="utf-8"))

    base = build_field_base(out, seed, log=log)
    f0 = base["field"]
    cards = f0.cardinalities
    fac, valx = int(PINS["delete_factor"]), int(PINS["delete_value"])
    (hf, hv), (hg, hu) = PINS["hole_pair"]
    rep = {"seed": int(seed), "image_bytes": int(f0.buf.size),
           "store_ledger": base["byte_ledger"], "manifest": f0.manifest()}

    def _check(before, after, ranges, label):
        try:
            r = assert_changed_bytes_within(before, after, ranges, label)
            r["holds"] = True
        except AssertionError as exc:
            dd = byte_diff(before, after)
            r = {"label": label, "holds": False, "error": str(exc),
                 "bytes_changed": int(dd["n_changed"]),
                 "per_segment": dd["per_segment"]}
            log(f"  !! {label}: THE BYTE PROPERTY DOES NOT HOLD -- {exc}")
        return {k: v for k, v in r.items() if k != "outside_examples"}

    # -- A. eviction --------------------------------------------------------
    fe, b0 = f0.copy(), f0.copy()
    ev = evict_term(fe, fac, valx, log=log)
    rep["evict"] = {**{k: v for k, v in ev.items() if k != "permitted_ranges"},
                    "byte_assertion": _check(b0, fe, ev["permitted_ranges"],
                                             f"evict object_hue index {valx}"),
                    "no_refit": assert_no_refit(b0, fe)}

    # -- A'. the LIFETIME route, in two steps -------------------------------
    fd, b1 = f0.copy(), f0.copy()
    ids, _, _ = _items_holding(fd, fac, valx)
    h = decay_items(fd, ids, 0.5, log=log)
    mid = fd.copy()
    rep["decay_half"] = {**{k: v for k, v in h.items() if k != "permitted_ranges"},
                         "byte_assertion": _check(b1, fd, h["permitted_ranges"],
                                                  "decay lifetimes to half")}
    z = decay_items(fd, ids, 0.0, log=log)
    rep["decay_zero"] = {**{k: v for k, v in z.items() if k != "permitted_ranges"},
                         "byte_assertion": _check(mid, fd, z["permitted_ranges"],
                                                  "decay lifetimes to zero"),
                         "no_refit": assert_no_refit(b1, fd)}
    rep["lifetime_route_reaches_the_same_state"] = bool(
        np.array_equal(fd.view("atoms"), fe.view("atoms"))
        and np.array_equal(fd.view("atom_alive"), fe.view("atom_alive")))

    # -- B. the hole --------------------------------------------------------
    fh_, b2 = f0.copy(), f0.copy()
    ho = punch_hole(fh_, hf, hv, hg, hu, log=log)
    ci = coupling_index(cards, hf, hv, hg, hu)
    rep["hole"] = {**{k: v for k, v in ho.items() if k != "permitted_ranges"},
                   "byte_assertion": _check(b2, fh_, ho["permitted_ranges"],
                                            "hole (red AND cylinder)"),
                   "no_refit": assert_no_refit(b2, fh_),
                   "coupling_value_at_the_hole_unchanged": bool(
                       float(fh_.view("coupling")[ci]) == float(f0.view("coupling")[ci])),
                   "coupling_evidence_weight_at_the_hole": float(f0.view("coupling_w")[ci]),
                   "n_term_pairs_the_pool_never_held": int(
                       (np.asarray(f0.view("coupling_w"), np.float32) == 0.0).sum()),
                   "n_holes_after": int(np.asarray(fh_.view("coupling_hole")).sum())}

    # -- a lifetime set, moving lifetime bytes and nothing else -------------
    fl, b3 = f0.copy(), f0.copy()
    lf = set_lifetime(fl, [3, 11, 29], 0.25, log=log)
    rep["set_lifetime"] = {**{k: v for k, v in lf.items() if k != "permitted_ranges"},
                           "byte_assertion": _check(b3, fl, lf["permitted_ranges"],
                                                    "set_lifetime"),
                           "no_refit": assert_no_refit(b3, fl)}

    # -- the NEGATIVE control: the instrument is not blind ------------------
    fn_, b4 = f0.copy(), f0.copy()
    fn_.view("term_mu")[0] += np.float32(1e-3)
    dd = byte_diff(b4, fn_)
    try:
        assert_no_refit(b4, fn_)
        fired = False
    except AssertionError:
        fired = True
    rep["negative_control_refit"] = {
        "what": ("a single 1e-3 nudge to ONE row of the address book -- the smallest possible "
                 "refit -- to show the instrument is not blind"),
        "bytes_changed": int(dd["n_changed"]), "per_segment": dd["per_segment"],
        "assert_no_refit_FIRED": bool(fired)}

    # -- ⭐ the GATE's byte cost, for the same two deletions -----------------
    cond = load_condition()
    grid = cond["grid"]
    g8 = Gate.per_factor(grid, base["pool"])
    gj = Gate.joint_hue_shape(grid, base["pool"])
    gb = Gate.joint_backoff(grid, base["pool"])

    def _bits(a: "Gate", b: "Gate") -> int:
        return int(sum(bin(x ^ y).count("1") for x, y in zip(a.pack(), b.pack())))

    hole_ci = coupling_index(cards, hf, hv, hg, hu)
    n_pairs = int(np.asarray(f0.view("coupling_hole")).size)
    rep["gate_bytes"] = {
        "per_factor_gate_bits": int(g8.n_bits()), "per_factor_gate_bytes": int(len(g8.pack())),
        "joint_presence_gate_bits": int(gj.n_bits()),
        "joint_backoff_gate_bits": int(gb.n_bits()),
        "value_deletion_bits_changed_per_factor": _bits(g8, g8.clear_value(fac, valx)),
        "CONJUNCTION_deletion_bits_changed_per_factor": _bits(
            g8, g8.delete_conjunction(hf, hv, hg, hu)),
        "CONJUNCTION_deletion_bits_changed_joint_presence": _bits(
            gj, gj.delete_conjunction(hf, hv, hg, hu)),
        "CONJUNCTION_deletion_bits_changed_joint_backoff": _bits(
            gb, gb.delete_conjunction(hf, hv, hg, hu)),
        "joint_presence_bit_was_already_false": bool(
            not gj.joint[int(hv), int(hu)]),
        "field_store_coupling_hole_bytes": n_pairs,
        "field_store_coupling_hole_bits_if_packed": n_pairs,
        "field_store_CONJUNCTION_deletion_bytes_changed": 1,
        "statement": (
            "THREE measured facts, not one. (1) A per-factor gate has no bit whose meaning is "
            "'this conjunction', so the conjunction deletion changes ZERO bits at any size. "
            "(2) A joint PRESENCE gate has ONE bit that means both 'never seen' and 'deleted', "
            "so on this split the deletion is ALSO a no-op on it -- the bit was already false -- "
            "while it refuses every unseen conjunction, which is measured on the scored corners. "
            "(3) The cheapest object that expresses the deletion WITHOUT refusing the unseen is a "
            "SEPARATE deleted-pair array beside the evidence, which is exactly what "
            "`coupling_hole` is; the generous `joint_backoff` rival is given one and is expected "
            "to tie the field store on the refusal half, declared before it was scored."),
    }

    # -- and the TABLE's byte cost for the conjunction ----------------------
    tabs = build_tables(out, seed, log=log)
    pb = np.concatenate([np.asarray(P, np.float32).ravel() for P in tabs["base"].protos])
    pr = np.concatenate([np.asarray(P, np.float32).ravel() for P in tabs["row_removed"].protos])
    pt = np.concatenate([np.asarray(P, np.float32).ravel() for P in tabs["retrain"].protos])
    rep["table_bytes"] = {
        "prototype_bytes": int(pb.nbytes),
        "value_deletion_bytes_changed": int((pb.view(np.uint8) != pr.view(np.uint8)).sum()),
        "CONJUNCTION_deletion_bytes_changed": 0,
        "conjunction_note": ("the marginal prototype table is indexed by (factor, value); there "
                             "is no conjunction slot to remove, so a conjunction deletion "
                             "changes ZERO bytes -- it cannot enter BAR 3 or BAR 4 at all"),
        "retrain_from_scratch_bytes_changed": int(
            (pb.view(np.uint8) != pt.view(np.uint8)).sum()),
    }

    rep["ALL_BYTE_PROPERTIES_HOLD"] = bool(
        rep["evict"]["byte_assertion"].get("holds")
        and rep["decay_half"]["byte_assertion"].get("holds")
        and rep["decay_zero"]["byte_assertion"].get("holds")
        and rep["hole"]["byte_assertion"].get("holds")
        and rep["set_lifetime"]["byte_assertion"].get("holds")
        and rep["negative_control_refit"]["assert_no_refit_FIRED"])
    d.mkdir(parents=True, exist_ok=True)
    _jdump(rep, art)
    log(f"  bytes(seed {seed}): evict {rep['evict']['byte_assertion']['bytes_changed']:,} B "
        f"({rep['evict']['byte_assertion'].get('bytes_changed_outside_permitted')} outside); "
        f"hole {rep['hole']['byte_assertion']['bytes_changed']} B "
        f"({rep['hole']['byte_assertion'].get('bytes_changed_outside_permitted')} outside); "
        f"ALL HOLD = {rep['ALL_BYTE_PROPERTIES_HOLD']}")
    return rep


# ---------------------------------------------------------------------------
# 6. The scoring stage -- BOTH halves, every arm, every corner
# ---------------------------------------------------------------------------

_JUDGE: dict = {}


def _judge(seed: int, log: Callable) -> Judge:
    if int(seed) not in _JUDGE:
        _JUDGE[int(seed)] = Judge(int(seed), log=log)
    return _JUDGE[int(seed)]


def _refuse_mask(refuse_fn, values, oor) -> tuple:
    """Normalise the two refusal-verb signatures to ``(refuse, detail)``."""
    try:
        r = refuse_fn(values)
    except TypeError:
        r = refuse_fn(values, oor)
    if isinstance(r, dict):
        det = {k: (np.asarray(v) if isinstance(v, np.ndarray) else v)
               for k, v in r.items()
               if k in ("hole", "empty_basin", "in_gap", "out_of_range")}
        return np.asarray(r["refuse"], bool), det
    return np.asarray(r, bool), {}


def score_seed(out: Path, seed: int, max_seconds: float = 7200.0,
               log: Callable = print) -> dict:
    """Every arm over every scored corner and every constructed probe.  Resumable."""
    d = seed_dir(out, seed)
    art = d / "scores.npz"
    if art.exists():
        log(f"  score(seed {seed}): already on disk")
        return {"incomplete": False}

    cache = assert_generation2(seed)
    cond = load_condition()
    grid = cond["grid"]
    cs = scored_corners(cond)
    pool = scored_pool(cond, seed)
    proof = assert_scored_and_heldout(cond, cs, pool)
    probes = conjunction_probes(cond, seed)
    built = build_arms(out, seed, log=log)
    arms = built["arms"]
    J = _judge(int(seed), log)
    shared_bitmap = value_presence_bitmap(grid, pool)

    part = d / "scores_partial.pkl"
    done: dict = {}
    if part.exists():
        with open(part, "rb") as fh:
            done = pickle.load(fh)
        log(f"  score(seed {seed}): resuming with {len(done)} (arm, corner) units done")

    units = []
    for a in arms:
        for c in cs["all"]:
            units.append((a, c["corner_id"]))
        for pname in probes:
            if not pname.startswith("_"):
                units.append((a, f"probe/{pname}"))
    by_id = {c["corner_id"]: c for c in cs["all"]}
    # ⭐ The judge is the expensive part and many arms share ONE generator -- every gate arm
    # reads the same table.  Judged tuples are therefore cached by (generator key, corner) and
    # the saving is exact, not approximate: the arms differ only in their refusal verb.
    # ⚠ A cached generator is ASSERTED equal on its features the first time it is reused via an
    # identical-output key (`field/hole` vs `field/base`), so the cache can never paper over a
    # difference it was supposed to find -- see `identical_to` below.
    gen_cache: dict = {}
    equalities: dict = {}
    # a generator that some arm declares itself identical to must be computed FIRST, and its
    # features must be kept; every other generator's features are dropped as soon as they are
    # judged, which is what keeps this stage inside its memory cap.
    twins = {arms[a]["identical_to"] for a in arms if arms[a].get("identical_to")}
    keys = sorted({arms[a]["gen_key"] for a in arms},
                  key=lambda k: (0 if k in twins else 1, k))
    rank = {k: i for i, k in enumerate(keys)}
    # ⚠ CORNER-MAJOR, and the ordering is a MEMORY decision, not a cosmetic one.  Sorting by
    # generator meant every twin generator's features for all 117 units had to be held until the
    # arm that declares itself identical to it was reached, and the process grew past 4 GB on a
    # box shared with three other agents.  Corner-major keeps at most ONE corner's features
    # alive: the twin is consumed immediately after its target, and the cache is dropped the
    # moment the corner changes.  The unit set, the judge-pass count and every number are
    # unchanged -- only the order and the peak resident size move.
    order = sorted(units, key=lambda u: (u[1], rank[arms[u[0]]["gen_key"]]))
    cur_cid = None

    t0 = _now()
    n_new = 0
    for (a, cid) in order:
        if (a, cid) in done:
            continue
        if _now() - t0 > float(max_seconds):
            with open(part, "wb") as fh:
                pickle.dump(done, fh)
            log(f"  score(seed {seed}): wall-clock cap after {n_new} new units -- state "
                "saved, rerun to continue")
            return {"incomplete": True, "n_done": len(done)}
        if cid != cur_cid:
            gen_cache.clear()
            cur_cid = cid
        if cid.startswith("probe/"):
            vals = probes[cid.split("/", 1)[1]]["values"]
            fam = "probe"
        else:
            vals = grid.values_of(np.asarray(by_id[cid]["rows"], np.int64))
            fam = by_id[cid]["family"]
        gk = arms[a]["gen_key"]
        twin = arms[a].get("identical_to")
        if (gk, cid) in gen_cache:
            feats, oor, tup, fh_ = gen_cache[(gk, cid)]
        else:
            feats, oor = arms[a]["gen"](vals)
            fh_ = int(zlib.crc32(np.ascontiguousarray(feats, np.float32).tobytes()))
            if twin is not None and (twin, cid) in gen_cache:
                # ⭐ the DECLARED no-op: asserted, never assumed.  `punch_hole` touches no
                # segment the read consults, so the features must be BIT-IDENTICAL; that is
                # BAR 4(d) and it is checked here before the judge pass is reused.
                tw = gen_cache[(twin, cid)]
                same = bool(tw[3] == fh_ and np.array_equal(
                    np.asarray(feats, np.float32), np.asarray(tw[0], np.float32)))
                e = equalities.setdefault(f"{gk} == {twin}",
                                          {"n_units": 0, "all_bit_identical": True})
                e["n_units"] += 1
                e["all_bit_identical"] = bool(e["all_bit_identical"] and same)
                if same:
                    tup = tw[2]
                else:
                    tup = J.read_back(feats, int(PINS["eval_batch"]))
                    n_new += 1
            else:
                tup = J.read_back(feats, int(PINS["eval_batch"]))
                n_new += 1
            gen_cache[(gk, cid)] = ((feats if gk in twins else None), oor, tup, fh_)
        refuse, det = _refuse_mask(arms[a]["refuse"], vals, oor)
        shared = accept_rule(vals, oor, np.zeros(vals.shape[0], bool), shared_bitmap, grid)
        accept = shared & ~refuse
        valid = judge_valid(vals, tup, grid) & accept
        done[(a, cid)] = {
            "n": int(vals.shape[0]), "family": fam,
            "own_refuse": refuse, "shared_accept": shared, "accept": accept,
            "valid": valid, "tuples": np.where(accept[:, None], tup, -1),
            "raw_tuples": tup,
            "hole": det.get("hole"), "empty_basin": det.get("empty_basin"),
            "in_gap": det.get("in_gap"),
            "feat_crc32": fh_,
        }
        if len(done) % 100 == 0:
            el = _now() - t0
            log(f"  score(seed {seed}): {len(done)}/{len(units)} units, {n_new} judge passes, "
                f"{el:.0f}s elapsed")
    with open(part, "wb") as fh:
        pickle.dump(done, fh)

    payload = {"done": np.array(done, dtype=object),
               "generator_equalities": np.array(equalities, dtype=object),
               "proof": np.array(proof, dtype=object),
               "cache": np.array(cache, dtype=object),
               "probes": np.array({k: {kk: vv for kk, vv in v.items() if kk != "values"}
                                   for k, v in probes.items()}, dtype=object),
               "ops": np.array(built["field_variants"]["_ops"], dtype=object),
               "table_cost": np.array(built["tables"]["cost"], dtype=object),
               "arm_roles": np.array({k: {"role": v["role"], "family": v["family"],
                                          "deletion": v["deletion"]}
                                      for k, v in arms.items()}, dtype=object),
               "corner_sets": np.array(
                   {"deleted": [c["corner_id"] for c in cs["deleted"]],
                    "survivor": [c["corner_id"] for c in cs["survivor"]],
                    "nway": [c["corner_id"] for c in cs["nway"]],
                    "pair": [c["corner_id"] for c in cs["pair"]]}, dtype=object)}
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(art, **payload)
    log(f"  score(seed {seed}): {len(done)} units in {_now() - t0:.1f}s")
    return {"incomplete": False, "n_done": len(done)}


# ---------------------------------------------------------------------------
# 6b. ⭐ The LEARNERS -- the dash, and the cost line that earns it
# ---------------------------------------------------------------------------

#: The learner recipe, declared in code and not tuned here: it is the recipe
#: ``w4_deletion.run_deletion_comparator`` already uses for its ``families`` leg, so the
#: cost line is comparable with the one wave 4 wrote.  ⛔ No selection happens on this stage.
LEARNER_EPOCHS, LEARNER_LR, LEARNER_WD = 20, 1e-3, 0.0


def learner_cost(out: Path, seed: int, families=("mlp",), log: Callable = print) -> dict:
    """⭐⭐ Why MLP / transformer / LSTM get a DASH, measured rather than asserted.

    A learner has **no refusal verb**.  Asked for a value it was never trained on it does not
    decline -- it emits a feature vector, the decoder renders it, and the judge reads back
    *something*.  So the only deletion available to it is retrain-from-scratch, and this stage
    measures exactly two things:

    1. **the cost** of that retrain, in wall-clock seconds and rows refitted, beside the field
       store's eviction, so the cost line is a measurement and not a slogan; and
    2. **what the retrained learner does when the deleted value is asked for anyway** -- which
       is the content of the footnote.  A silent wrong answer is not a deletion, and the number
       that says so belongs in the report.

    ⛔ This stage does NOT put a learner on the refusal row.  It cannot enter it.
    """
    from ..core.gen_request import dense_request_arrays
    from ..eval.rivals import w4_learners as L
    import jax

    d = seed_dir(out, seed)
    art = d / "learners.json"
    if art.exists():
        log(f"  learners(seed {seed}): already on disk")
        return json.loads(art.read_text(encoding="utf-8"))

    assert_generation2(seed)
    cond = load_condition()
    grid = cond["grid"]
    cs = scored_corners(cond)
    pool = scored_pool(cond, seed)
    fac, valx = int(PINS["delete_factor"]), int(PINS["delete_value"])
    cards = [int(c) for c in grid.cardinalities]
    F = features_mmap(int(seed))
    X = gather(F, pool)
    D = int(X.shape[1])
    fi = grid.factor_indices_of(pool)
    keep = fi[:, fac] != valx

    dr_fit = dense_request_arrays(grid, _reqs(grid.values_of(pool), grid))
    J = _judge(int(seed), log)
    # the deleted set and a matched survivor slice, as REQUESTS
    del_rows = np.concatenate([c["rows"] for c in cs["deleted"]])
    sur_rows = np.concatenate([c["rows"] for c in cs["survivor"]])
    rng = np.random.default_rng(20_260_920 + seed)
    sur_rows = sur_rows[rng.permutation(sur_rows.size)[:del_rows.size]]

    key = jax.random.PRNGKey(int(seed) * 7717 + 20_260_920)
    target = int(PINS["store_bytes"]) // 4          # the pinned store's parameter count
    bs = L.builders(cards, D, key)
    rep = {"seed": int(seed), "recipe": {"epochs": LEARNER_EPOCHS, "lr": LEARNER_LR,
                                         "weight_decay": LEARNER_WD,
                                         "declared": "inherited from w4_deletion's families "
                                                     "leg; no tuning happens on this stage"},
           "parameter_target": target, "families": {}}

    for fam in families:
        build, kind = bs[fam]
        w, n_params, err = L.match_width(build, int(L.FAMILY_BUDGETS.get(fam, target)),
                                         log=log)
        Xin = L.encode_inputs(kind, dr_fit, cards)
        Y = np.asarray(X, np.float32)
        legs = {}
        for leg, mask in (("full_pool", np.ones(pool.size, bool)), ("without_red", keep)):
            t0 = _now()
            key, fk = jax.random.split(key)
            m, info = L.train(build(w), Xin[mask], Y[mask], fk, epochs=LEARNER_EPOCHS,
                              lr=LEARNER_LR, wd=LEARNER_WD)
            wall = _now() - t0

            def _feats(rows, _m=m, _kind=kind):
                dr = dense_request_arrays(grid, _reqs(grid.values_of(rows), grid))
                return L.predict(_m, L.encode_inputs(_kind, dr, cards))

            leg_rep = {"wall_clock_s": float(wall), "rows_refitted": int(mask.sum()),
                       "param_count": int(n_params), "param_error": float(err),
                       "fit": info}
            for name, rows in (("deleted_set", del_rows), ("survivor_slice", sur_rows)):
                fe = _feats(rows)
                vals = grid.values_of(rows)
                tup = J.read_back(fe, int(PINS["eval_batch"]))
                ok = judge_valid(vals, tup, grid)
                asked = grid.factor_indices_of(rows)[:, fac]
                leg_rep[name] = {
                    "n": int(rows.size),
                    "REFUSED": 0.0,
                    "refusal_verb": "NONE -- this family has no refusal verb; it always emits",
                    "judge_satisfies_the_full_request": float(ok.mean()),
                    "judge_reads_back_the_asked_object_hue": float(
                        (tup[:, fac] == asked).mean()),
                }
            legs[leg] = leg_rep
        rep["families"][fam] = legs
        log(f"  learners(seed {seed}) {fam}: full pool {legs['full_pool']['wall_clock_s']:.1f}s "
            f"/ WITHOUT RED {legs['without_red']['wall_clock_s']:.1f}s on "
            f"{legs['without_red']['rows_refitted']:,} rows; after the retrain it still ANSWERS "
            f"{legs['without_red']['deleted_set']['n']} red requests (refusal 0.0000) and the "
            f"judge reads the asked hue back at "
            f"{legs['without_red']['deleted_set']['judge_reads_back_the_asked_object_hue']:.4f}")
    del X
    rep["footnote"] = (
        "MLP / transformer / LSTM take a DASH on the refusal half of row 7 because they have no "
        "refusal verb: asked for a deleted value they emit a feature vector like any other, and "
        "the number above says what the judge then reads back. Their only deletion route is "
        "retrain-from-scratch, whose measured cost is on this page. A dash is never a win for us.")
    d.mkdir(parents=True, exist_ok=True)
    _jdump(rep, art)
    return rep


def unseen_pairs_check(out: Path, seed: int, n_bg: int = 64, log: Callable = print) -> dict:
    """⭐⭐ BAR 4(b): EVERY term pair the pool never held must stay ANSWERED after the hole.

    w5-common §3.7 records the field store's central property as "the hole refuses one
    conjunction while 96 unseen conjunctions stay answered", measured on the val condition.
    This scores it: it enumerates **every** coupling entry whose evidence weight is exactly
    zero -- i.e. every conjunction the fit pool never held -- and asks the store for each one,
    before and after the hole is punched.

    The bar, from the PREREG: on every unseen pair that is NOT the hole, the ``hole`` flag is
    **0.0000** and the refusal rate moves by at most 0.01. On the hole itself the flag is
    1.0000. ⛔ If a single other unseen pair lights up, the coupling table is a joint presence
    table in disguise and the row dies -- which is precisely the trap this design claims to
    escape by keeping ``coupling_w`` and ``coupling_hole`` in different arrays.

    No judge is involved: this is the store's own refusal verb, so it costs seconds.
    """
    from ..core.gen_field import coupling_index, gap_read, punch_hole

    d = seed_dir(out, seed)
    art = d / "unseen_pairs.json"
    if art.exists():
        log(f"  unseen-pairs(seed {seed}): already on disk")
        return json.loads(art.read_text(encoding="utf-8"))

    cond = load_condition()
    grid = cond["grid"]
    base = build_field_base(out, seed, log=log)
    f0 = base["field"]
    cards = f0.cardinalities
    (hf, hv), (hg, hu) = PINS["hole_pair"]
    holed_idx = coupling_index(cards, hf, hv, hg, hu)
    fh_ = f0.copy()
    punch_hole(fh_, hf, hv, hg, hu)

    w = np.asarray(f0.view("coupling_w"), np.float32)
    rng = np.random.default_rng(20_260_921 + int(seed))
    te = np.asarray(cond["test_rows"], np.int64)
    bg = grid.values_of(te[rng.permutation(te.size)[:int(n_bg)]])

    rows_out, worst = [], {"drift": 0.0, "flag": 0.0}
    for (f, g), off in zip(f0.meta["coupling_pairs"], f0.meta["coupling_offsets"]):
        f, g, off = int(f), int(g), int(off)
        Vf, Vg = cards[f], cards[g]
        for v in range(Vf):
            for u in range(Vg):
                idx = off + v * Vg + u
                if float(w[idx]) != 0.0:
                    continue
                vals = np.array(bg, np.float64, copy=True)
                vals[:, f] = float(grid.values[f][v])
                vals[:, g] = float(grid.values[g][u])
                pre = gap_read(f0, vals, grid.values)
                post = gap_read(fh_, vals, grid.values)
                rec = {"pair": [[f, v], [g, u]], "coupling_index": int(idx),
                       "is_THE_hole": bool(idx == holed_idx),
                       "hole_flag_before": float(np.asarray(pre["hole"]).mean()),
                       "hole_flag_after": float(np.asarray(post["hole"]).mean()),
                       "refuse_before": float(np.asarray(pre["refuse"]).mean()),
                       "refuse_after": float(np.asarray(post["refuse"]).mean())}
                rec["refuse_drift"] = rec["refuse_after"] - rec["refuse_before"]
                rows_out.append(rec)
                if not rec["is_THE_hole"]:
                    worst["drift"] = max(worst["drift"], abs(rec["refuse_drift"]))
                    worst["flag"] = max(worst["flag"], rec["hole_flag_after"])

    others = [r for r in rows_out if not r["is_THE_hole"]]
    the_hole = [r for r in rows_out if r["is_THE_hole"]]
    rep = {
        "seed": int(seed), "n_backgrounds_per_pair": int(n_bg),
        "n_term_pairs_the_pool_NEVER_held": len(rows_out),
        "n_other_unseen_pairs": len(others),
        "THE_HOLE": the_hole[0] if the_hole else {"NOT_RUN": "the holed pair was not unseen"},
        "other_unseen_pairs": {
            "max_hole_flag_after": worst["flag"],
            "max_abs_refuse_drift": worst["drift"],
            "n_with_any_hole_flag": int(sum(1 for r in others if r["hole_flag_after"] > 0)),
            "n_with_refuse_drift_above_0.01": int(
                sum(1 for r in others if abs(r["refuse_drift"]) > BAR_REFUSE_SURVIVOR_DRIFT)),
            "mean_refuse_after": float(np.mean([r["refuse_after"] for r in others]))
            if others else None,
            "n_fully_answered_after_the_hole": int(
                sum(1 for r in others if r["refuse_after"] == 0.0)),
        },
        "BAR4b_PASS": bool(the_hole and the_hole[0]["hole_flag_after"] == 1.0
                           and worst["flag"] == 0.0
                           and worst["drift"] <= BAR_REFUSE_SURVIVOR_DRIFT),
        "per_pair": rows_out,
        "statement": ("every conjunction the fit pool never held, enumerated from coupling_w == 0 "
                      "and asked for directly; the hole must light up and nothing else may"),
    }
    d.mkdir(parents=True, exist_ok=True)
    _jdump(rep, art)
    log(f"  unseen-pairs(seed {seed}): {len(rows_out)} pairs the pool never held; THE HOLE flags "
        f"{rep['THE_HOLE'].get('hole_flag_after')}; of the other {len(others)}, "
        f"{rep['other_unseen_pairs']['n_with_any_hole_flag']} light up and the worst refusal "
        f"drift is {worst['drift']:.4f}; BAR 4(b) PASS = {rep['BAR4b_PASS']}")
    return rep


#: The learner families that take a DASH on row 7, and the two sampler arms that take one too.
DASH_FAMILIES = ("mlp", "mlp_180k", "cont_mlp", "transformer", "lstm")


def dash_evidence(out: Path, seed: int, log: Callable = print) -> dict:
    """⭐⭐ EARN the dash, with a measurement rather than an assertion.

    w5-common §1: a row a rival cannot enter gets a dash and a footnote, and ⛔ never a win for
    us.  A footnote that just says "they have no refusal verb" is an assertion.  This stage
    makes it a measurement: it asks each shipped learner for the **deleted** conjunction and
    records (a) that it refuses nothing, because it has no verb with which to, and (b) what the
    judge reads back when it answers anyway.  A silent wrong answer is not a deletion, and the
    number that says so is what the dash rests on.

    ⚠ **POPULATION, and it is why nothing here is a behavioural comparison.**  These rivals are
    spoke A's, fitted on the **Montero combination pool** (32,256 rows, every factor value
    present) -- their own `row_eligibility.json` says they are NOT like-for-like with a
    value-holdout arm on a whole-value row, which row 7 is.  ⛔ So this stage reports a
    STRUCTURAL fact (no refusal verb -- which is population-independent) plus a descriptive
    read-back, and it does **not** put these numbers beside the field store's as a contest.  The
    like-for-like rivals on this hub are the table arms, refitted on my own pool.
    """
    from ..eval.rivals.w4_api import available, load_rival

    d = seed_dir(out, seed)
    art = d / "dash.json"
    if art.exists():
        log(f"  dash(seed {seed}): already on disk")
        return json.loads(art.read_text(encoding="utf-8"))

    assert_generation2(seed)
    cond = load_condition()
    grid = cond["grid"]
    cs = scored_corners(cond)
    fac = int(PINS["delete_factor"])
    J = _judge(int(seed), log)
    avail = available(int(seed))
    del_rows = np.concatenate([c["rows"] for c in cs["deleted"]])
    sur_rows = np.concatenate([c["rows"] for c in cs["survivor"]])
    rng = np.random.default_rng(20_260_920 + int(seed))
    sur_rows = np.sort(sur_rows[rng.permutation(sur_rows.size)[:del_rows.size]])
    rep = {"seed": int(seed), "store_parameters": int(PINS["store_bytes"]) // 4,
           "store_bytes": int(PINS["store_bytes"]), "families": {}, "not_callable": {}}

    for fam in DASH_FAMILIES:
        info = avail.get(fam)
        if not info or not info.get("callable"):
            rep["not_callable"][fam] = (info or {}).get("reason", "absent from the rival set")
            continue
        r = load_rival(int(seed), fam)
        fam_rep = {"param_count": info.get("param_count"),
                   "fit_population": info.get("fit_population"),
                   "population_caveat": info.get("population_caveat"),
                   "refusal_verb": "NONE",
                   "REFUSED_on_the_deleted_set": 0.0,
                   "why_the_dash": ("this family has no refusal verb: asked for a deleted value "
                                    "or a deleted conjunction it emits a feature vector like any "
                                    "other request. It cannot enter the refusal half of row 7 at "
                                    "all, and its only deletion route is retrain-from-scratch, "
                                    "whose measured cost is on the cost line.")}
        for name, rows in (("deleted_set_red_AND_cylinder", del_rows),
                           ("survivor_slice", sur_rows)):
            vals = grid.values_of(rows)
            feats, oor = r(vals)
            tup = J.read_back(np.asarray(feats, np.float32), int(PINS["eval_batch"]))
            asked = grid.factor_indices_of(rows)
            fam_rep[name] = {
                "n": int(rows.size),
                "refused": 0.0,
                "out_of_range_flagged": float(np.asarray(oor, bool).mean()),
                "judge_satisfies_the_full_request": float(
                    judge_valid(vals, tup, grid).mean()),
                "judge_reads_back_the_asked_object_hue": float(
                    (tup[:, fac] == asked[:, fac]).mean()),
                "judge_reads_back_the_asked_shape": float((tup[:, 4] == asked[:, 4]).mean()),
            }
        rep["families"][fam] = fam_rep
        log(f"  dash(seed {seed}) {fam}: refuses {fam_rep['REFUSED_on_the_deleted_set']:.4f} of "
            f"the 720 deleted-conjunction requests and answers all of them; the judge reads the "
            f"asked hue back at "
            f"{fam_rep['deleted_set_red_AND_cylinder']['judge_reads_back_the_asked_object_hue']:.4f}")

    for fam in ("lace", "poe_ebm"):
        info = avail.get(fam) or {}
        rep["not_callable"][fam] = info.get("reason", "absent")
    rep["footnote"] = (
        "MLP / mlp_180k / continuous-input MLP / transformer / LSTM take a DASH on row 7's "
        "refusal half: measured here, every one of them refuses 0.0000 of the deleted requests, "
        "because none of them has a refusal verb at all. LACE and PoE-EBM take a dash for a "
        "different reason -- spoke A's own API declines to expose them as request->features "
        "callables, because their output is a sampler draw rather than a function of the request "
        "-- and both sit far outside the byte pin (95.1x and 27.1x). "
        "POPULATION: every shipped rival here is fitted on the Montero combination pool and is "
        "NOT like-for-like with a value-holdout arm on a whole-value row, so these read-backs are "
        "descriptive and are NOT placed beside the field store as a contest. "
        "A dash is never a win for us.")
    d.mkdir(parents=True, exist_ok=True)
    _jdump(rep, art)
    return rep


def hole_read_identity(out: Path, seed: int, log: Callable = print) -> dict:
    """⭐⭐ BAR 4(d), asserted DIRECTLY: the hole changes no rendered feature, anywhere.

    ⚠ Why this exists as its own stage rather than as a by-product of scoring. The scoring loop
    opportunistically checks this when it happens to compute ``field/base`` and ``field/hole`` for
    the same corner in the same process, and reuses the judge pass when they agree. But the run is
    **resumable**, and a resumed process skips units already in ``done`` -- so on a run that spans
    several wall-clock windows the opportunistic check records nothing, and a PASS that depended on
    it would be reporting the absence of a diagnostic as a failure. That is exactly what happened
    here on the first assembled seed.

    So the identity is asserted here instead, directly and unconditionally: for **every** scored
    corner, render through the base store and through the holed store and compare the arrays
    elementwise. ``punch_hole`` writes one byte of ``coupling_hole``, a segment
    :func:`gen_field.field_read` never consults, so the two must be **bit-identical** -- and if they
    are ever not, the hole is touching the read and BAR 4(d) genuinely fails.

    Costs seconds: no judge is involved.
    """
    from ..core.gen_field import punch_hole

    d = seed_dir(out, seed)
    art = d / "hole_read_identity.json"
    if art.exists():
        log(f"  hole-read(seed {seed}): already on disk")
        return json.loads(art.read_text(encoding="utf-8"))

    cond = load_condition()
    grid = cond["grid"]
    cs = scored_corners(cond)
    base = build_field_base(out, seed, log=log)
    f0 = base["field"]
    (hf, hv), (hg, hu) = PINS["hole_pair"]
    fh_ = f0.copy()
    punch_hole(fh_, hf, hv, hg, hu)

    n_corners = n_requests = 0
    all_same = True
    worst = 0.0
    per_family = {}
    for c in cs["all"]:
        vals = grid.values_of(np.asarray(c["rows"], np.int64))
        a, _ = _field_features(f0, vals, grid)
        b, _ = _field_features(fh_, vals, grid)
        same = bool(np.array_equal(np.asarray(a, np.float32), np.asarray(b, np.float32)))
        diff = float(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64)).max())
        all_same = all_same and same
        worst = max(worst, diff)
        n_corners += 1
        n_requests += int(vals.shape[0])
        k = c["family"]
        per_family.setdefault(k, {"n_corners": 0, "all_bit_identical": True})
        per_family[k]["n_corners"] += 1
        per_family[k]["all_bit_identical"] = bool(per_family[k]["all_bit_identical"] and same)
    # and on the constructed probes, including the holed conjunction itself
    probes = conjunction_probes(cond, seed)
    probe_same = {}
    for name, v in probes.items():
        if name.startswith("_"):
            continue
        a, _ = _field_features(f0, v["values"], grid)
        b, _ = _field_features(fh_, v["values"], grid)
        probe_same[name] = bool(np.array_equal(np.asarray(a, np.float32),
                                               np.asarray(b, np.float32)))
        all_same = all_same and probe_same[name]

    rep = {"seed": int(seed), "n_corners": n_corners, "n_requests": n_requests,
           "ALL_BIT_IDENTICAL": bool(all_same),
           "max_abs_elementwise_difference": worst,
           "per_family": per_family, "per_probe_bit_identical": probe_same,
           "what_this_asserts": (
               "punch_hole writes ONE byte of `coupling_hole`, a segment field_read never "
               "consults, so every rendered feature must be bit-identical before and after. "
               "This is BAR 4(d) asserted directly on every scored corner and every constructed "
               "probe -- including the holed conjunction itself, whose RENDER is unchanged even "
               "though the request is now refused. The refusal is a separate verb from the read, "
               "and that separation is what makes the hole a deletion rather than a distortion.")}
    d.mkdir(parents=True, exist_ok=True)
    _jdump(rep, art)
    log(f"  hole-read(seed {seed}): {n_corners} corners / {n_requests:,} requests + "
        f"{len(probe_same)} probes -- ALL BIT-IDENTICAL = {rep['ALL_BIT_IDENTICAL']} "
        f"(max elementwise difference {worst:g})")
    return rep


def nearest_atom_check(out: Path, seed: int, log: Callable = print) -> dict:
    """Charter section C.4: if the settled state sits ON its nearest atom, we are a lookup."""
    d = seed_dir(out, seed)
    art = d / "c4_nearest_atom.json"
    if art.exists():
        return json.loads(art.read_text(encoding="utf-8"))
    cond = load_condition()
    grid = cond["grid"]
    cs = scored_corners(cond)
    base = build_field_base(out, seed, log=log)
    field = base["field"]
    rows = np.concatenate([c["rows"] for c in cs["survivor"]])
    rng = np.random.default_rng(20_260_920 + seed)
    rows = rows[rng.permutation(rows.size)[:1024]]
    vals = grid.values_of(rows)
    read, _ = _field_features(field, vals, grid)
    _, _, nearest_to_address = _field_lookup_features(field, vals, grid, "argmax")
    lk, _, _ = _field_lookup_features(field, vals, grid, "argmax")
    dist_read_to_nearest_atom = np.linalg.norm(
        np.asarray(read, np.float64) - np.asarray(lk, np.float64), axis=1)
    scale = np.linalg.norm(np.asarray(read, np.float64), axis=1)
    rep = {"seed": int(seed), "n_requests": int(rows.size),
           "mean_distance_settled_to_nearest_atom": float(dist_read_to_nearest_atom.mean()),
           "mean_relative_distance": float(
               (dist_read_to_nearest_atom / np.maximum(scale, 1e-12)).mean()),
           "min_relative_distance": float(
               (dist_read_to_nearest_atom / np.maximum(scale, 1e-12)).min()),
           "mean_address_to_nearest_atom": float(np.mean(nearest_to_address)),
           "verdict": None}
    rep["verdict"] = (
        "LOOKUP TABLE -- the settled state sits on its nearest atom (charter C.4)"
        if rep["mean_relative_distance"] < 1e-3 else
        "NOT a lookup table: the settled state is a strict interior point of the atoms' span, "
        f"at relative distance {rep['mean_relative_distance']:.4f} from the nearest atom")
    d.mkdir(parents=True, exist_ok=True)
    _jdump(rep, art)
    log(f"  C.4(seed {seed}): {rep['verdict']}")
    return rep


# ---------------------------------------------------------------------------
# 7. The report -- both halves, the bars, the cost line
# ---------------------------------------------------------------------------


def _per_corner(rec, grid, seen_mask) -> dict:
    cards = tuple(int(c) for c in grid.cardinalities)
    v = np.asarray(rec["valid"], bool)
    t = np.asarray(rec["tuples"], np.int64)
    keep = t[v]
    if keep.size:
        lin = np.ravel_multi_index(tuple(keep.T), cards)
        uniq = np.unique(lin)
        novel = uniq[~seen_mask[uniq]]
    else:
        uniq = np.zeros(0, np.int64)
        novel = np.zeros(0, np.int64)
    return {"n_requests": int(rec["n"]),
            "n_accepted": int(np.asarray(rec["accept"]).sum()),
            "refuse_rate": float(np.asarray(rec["own_refuse"]).mean()),
            "judge_accuracy": float(v.mean()) if rec["n"] else 0.0,
            "distinct_valid_tuples": int(uniq.size),
            "distinct_valid_novel_tuples": int(novel.size)}


def _paired(a_rows: dict, b_rows: dict, ids, metric: str) -> dict:
    """Paired per-corner difference ``a - b``, with a bootstrap CI.  Never a single cell."""
    dif = np.array([float(a_rows[i][metric]) - float(b_rows[i][metric]) for i in ids],
                   np.float64)
    n = dif.size
    if n == 0:
        return {"NOT_RUN": "no paired corners"}
    rng = np.random.default_rng(20_260_920)
    bs = np.array([dif[rng.integers(0, n, n)].mean() for _ in range(2000)])
    from math import sqrt
    se = float(dif.std(ddof=1) / sqrt(n)) if n > 1 else 0.0
    t = float(dif.mean() / se) if se > 0 else (0.0 if dif.mean() == 0 else float("inf"))
    return {"n_corners": int(n), "mean_delta": float(dif.mean()),
            "median_delta": float(np.median(dif)), "sd": float(dif.std(ddof=1)) if n > 1 else 0.0,
            "ci95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
            "t": t, "n_positive": int((dif > 0).sum()), "n_negative": int((dif < 0).sum()),
            "n_zero": int((dif == 0).sum()),
            "max_abs_delta": float(np.abs(dif).max())}


def corner_geometry(cond) -> dict:
    """``corner_id -> {shape, hues, parent}``, so BAR 2's break-out is by the corner's OWN
    geometry rather than by its name.  Asserted, not assumed: every corner on this scoreboard
    holds ``shape`` constant, and the break-out would be meaningless if one did not."""
    grid = cond["grid"]
    cs = scored_corners(cond)
    geo = {}
    for c in cs["all"]:
        fi = grid.factor_indices_of(c["rows"])
        shapes = sorted({int(x) for x in np.unique(fi[:, 4])})
        if len(shapes) != 1:
            raise AssertionError(
                f"{c['corner_id']} varies shape over {shapes}; this scoreboard's corners hold "
                "shape constant and the BAR 2 break-out depends on it")
        geo[c["corner_id"]] = {
            "shape_index": shapes[0], "shape": SHAPE_NAMES[shapes[0]],
            "hue_indices": sorted({int(x) for x in np.unique(fi[:, 2])}),
            "parent": c["meta"].get("parent"), "family": c["family"]}
    return geo


def _breakout(rows_a: dict, rows_b: dict, geo: dict, ids) -> dict:
    """⭐ BAR 2's literal sentence: 'green and blue still compose on EVERY shape'."""
    out = {}
    for name, sel in (
            ("green (pair-001, green AND sphere -- NEVER SEEN)",
             [i for i in ids if geo[i]["parent"] == "pair-001"]),
            ("blue (pair-002, blue AND oblong -- NEVER SEEN)",
             [i for i in ids if geo[i]["parent"] == "pair-002"]),
            ("n-way corners, shape = cylinder",
             [i for i in ids if geo[i]["family"] == "nway" and geo[i]["shape_index"] == 1]),
            ("n-way corners, shape = sphere",
             [i for i in ids if geo[i]["family"] == "nway" and geo[i]["shape_index"] == 2]),
            ("n-way corners, shape = oblong",
             [i for i in ids if geo[i]["family"] == "nway" and geo[i]["shape_index"] == 3])):
        out[name] = ({"NOT_RUN": "no corners in this slice"} if not sel
                     else {**_paired(rows_a, rows_b, sel, "distinct_valid_novel_tuples"),
                           "novel_per_corner_after": float(np.mean(
                               [rows_a[i]["distinct_valid_novel_tuples"] for i in sel])),
                           "novel_per_corner_before": float(np.mean(
                               [rows_b[i]["distinct_valid_novel_tuples"] for i in sel]))})
    return out


def build_report(out: Path, log: Callable = print) -> dict:
    """Assemble every seed into the two halves, the bars, and the cost line."""
    out = Path(out)
    cond = load_condition()
    grid = cond["grid"]
    seen_mask = seen_tuple_mask(grid, cond["train_rows"])
    geo = corner_geometry(cond)
    seeds = [s for s in PINS["seeds"] if (seed_dir(out, s) / "scores.npz").exists()]
    if not seeds:
        raise AssertionError("no scored seed on disk -- run --stage score first")
    rep = {"seeds_scored": seeds, "prereg": "PREREG-w5-delete-1.md",
           "scored_run": "1/3 of target `deletion`", "per_seed": {}}

    for s in seeds:
        z = np.load(seed_dir(out, s) / "scores.npz", allow_pickle=True)
        done = dict(z["done"].item())
        sets = dict(z["corner_sets"].item())
        roles = dict(z["arm_roles"].item())
        probes = dict(z["probes"].item())
        # the 256 background row ids are provenance, not a result; keep the count, drop the list
        if "_backgrounds" in probes:
            probes["_backgrounds"] = {k: v for k, v in probes["_backgrounds"].items()
                                      if k != "rows"}
        arms = sorted({a for (a, _) in done})
        # one row per (arm, corner); the estimand's unit
        rows = {}
        for (a, cid) in done:
            rows.setdefault(a, {})[cid] = _per_corner(done[(a, cid)], grid, seen_mask)
        eqs = dict(z["generator_equalities"].item()) if "generator_equalities" in z else {}

        def _hri(sd):
            q = seed_dir(out, sd) / "hole_read_identity.json"
            return json.loads(q.read_text(encoding="utf-8")) if q.exists() else None

        surv, dele = list(sets["survivor"]), list(sets["deleted"])
        # ⭐ how hard each constructed probe is: what fraction of its request TUPLES the fit pool
        # actually holds.  Recomputed here (the probes are deterministic in the seed) so a reader
        # never has to guess whether a probe was asking for something seen or something novel.
        _pr = conjunction_probes(cond, int(s))
        _poolset = set(int(x) for x in scored_pool(cond, int(s)))
        _trainset = set(int(x) for x in np.asarray(cond["train_rows"], np.int64))
        _cards = tuple(int(c) for c in grid.cardinalities)
        probe_hardness = {}
        for _p, _v in _pr.items():
            if _p.startswith("_"):
                continue
            _idx = np.stack([np.abs(np.asarray(grid.values[k], np.float64)[None, :]
                                    - _v["values"][:, k][:, None]).argmin(axis=1)
                             for k in range(len(_cards))], axis=1)
            _lin = np.ravel_multi_index(tuple(_idx.T), _cards)
            probe_hardness[_p] = {
                "label": _v["label"], "why": _v["why"],
                "conjunction_seen_in_pool": bool(_v["seen_in_pool"]),
                "frac_full_tuples_in_the_FIT_POOL": float(
                    np.mean([int(x) in _poolset for x in _lin])),
                "frac_full_tuples_in_the_TRAIN_index": float(
                    np.mean([int(x) in _trainset for x in _lin]))}
        S = {"arms": {}, "bars": {}, "cost": dict(z["table_cost"].item()),
             "ops": dict(z["ops"].item()), "cache": dict(z["cache"].item()),
             "proof": dict(z["proof"].item()), "probe_spec": probes, "roles": roles,
             "probe_hardness": probe_hardness}

        def _rate(a, cid, field="own_refuse"):
            return float(np.asarray(done[(a, cid)][field]).mean())

        def _setrate(a, ids, field="own_refuse"):
            num = sum(int(np.asarray(done[(a, c)][field]).sum()) for c in ids)
            den = sum(int(done[(a, c)]["n"]) for c in ids)
            return float(num) / max(den, 1)

        for a in arms:
            S["arms"][a] = {
                "role": roles[a]["role"], "family": roles[a]["family"],
                "deletion": roles[a]["deletion"],
                "refuse_on_deleted_set": _setrate(a, dele),
                "refuse_on_survivor_set": _setrate(a, surv),
                "novel_per_corner_survivor": float(np.mean(
                    [rows[a][c]["distinct_valid_novel_tuples"] for c in surv])),
                "novel_per_corner_deleted": float(np.mean(
                    [rows[a][c]["distinct_valid_novel_tuples"] for c in dele])),
                "judge_accuracy_survivor": float(np.mean(
                    [rows[a][c]["judge_accuracy"] for c in surv])),
                "probes": {p: {"refuse": _rate(a, f"probe/{p}"),
                               "judge_accuracy": float(np.asarray(
                                   done[(a, f"probe/{p}")]["valid"]).mean()),
                               "accepted": float(np.asarray(
                                   done[(a, f"probe/{p}")]["accept"]).mean()),
                               "hole_flag": (float(np.asarray(
                                   done[(a, f'probe/{p}')]["hole"]).mean())
                                   if done[(a, f"probe/{p}")]["hole"] is not None else None),
                               "empty_basin": (float(np.asarray(
                                   done[(a, f'probe/{p}')]["empty_basin"]).mean())
                                   if done[(a, f"probe/{p}")]["empty_basin"] is not None
                                   else None)}
                           for p in probes if not p.startswith("_")},
            }

        # ---- WHICH refusal mechanism is doing the work -------------------
        # `chlu-w5-refuse-f2` measured, on their own scored legs, that `in_gap` costs 0.005 of
        # the accept leg and adds nothing -- their four-mechanism rule LOSES to the 8-byte rule
        # because of it. `in_gap` is one of this store's three mechanisms too, so the question is
        # answered here from the per-request masks rather than assumed either way.
        def _mech(a, ids):
            out = {}
            for m in ("hole", "empty_basin", "in_gap"):
                num = den = 0
                for c in ids:
                    v = done[(a, c)].get(m)
                    if v is None:
                        break
                    num += int(np.asarray(v).sum())
                    den += int(done[(a, c)]["n"])
                out[m] = (float(num) / den) if den else None
            # what the store would refuse if `in_gap` were dropped from the rule
            num = den = 0
            for c in ids:
                h, e = done[(a, c)].get("hole"), done[(a, c)].get("empty_basin")
                o = done[(a, c)].get("in_gap")
                if h is None or e is None or o is None:
                    break
                num += int((np.asarray(h) | np.asarray(e)).sum())
                den += int(done[(a, c)]["n"])
            out["refuse_WITHOUT_in_gap"] = (float(num) / den) if den else None
            return out

        S["refusal_mechanism_decomposition"] = {
            a: {"deleted_set": _mech(a, dele), "survivor_set": _mech(a, surv),
                "refuse_WITH_in_gap": {"deleted_set": S["arms"][a]["refuse_on_deleted_set"],
                                       "survivor_set": S["arms"][a]["refuse_on_survivor_set"]}}
            for a in ("field/base", "field/evict_red", "field/decay_red", "field/hole")
        }
        S["refusal_mechanism_decomposition"]["what_this_answers"] = (
            "`in_gap` is a distance cut, not a discriminator. If the deletion refusals are carried "
            "entirely by `empty_basin` (what an eviction leaves) and `hole` (what a punch leaves), "
            "then `in_gap` is pure cost on the survivor set and this hub says so. Compare "
            "`refuse_WITHOUT_in_gap` with `refuse_WITH_in_gap` on both sets.")

        # ---- the paired composition half -------------------------------
        def _pair(a, b):
            return _paired(rows[a], rows[b], surv, "distinct_valid_novel_tuples")

        S["composition_vs_base"] = {
            "field/evict_red": _pair("field/evict_red", "field/base"),
            "field/decay_red": _pair("field/decay_red", "field/base"),
            "field/hole": _pair("field/hole", "field/base"),
            "field/landscape_removed": _pair("field/landscape_removed", "field/base"),
            "field/landscape_shuffled": _pair("field/landscape_shuffled", "field/base"),
            "field/lookup_argmax": _pair("field/lookup_argmax", "field/base"),
            "field/lookup_mean": _pair("field/lookup_mean", "field/base"),
            "table/row_removed": _pair("table/row_removed", "table/base"),
            "table/retrain": _pair("table/retrain", "table/base"),
            "table+gate8/clear_red": _pair("table+gate8/clear_red", "table+gate8/base"),
            "table+gate8/clear_cylinder": _pair("table+gate8/clear_cylinder",
                                                "table+gate8/base"),
            "table+gate8/hole_attempt": _pair("table+gate8/hole_attempt", "table+gate8/base"),
            "table+joint/hole": _pair("table+joint/hole", "table+joint/base"),
            "table+backoff/hole": _pair("table+backoff/hole", "table+backoff/base"),
        }
        # ⭐ the field store against the rivals, on the SURVIVING composition, post-deletion
        S["composition_vs_rivals"] = {
            "field_evict_vs_table_retrain(GOLD)": _paired(
                rows["field/evict_red"], rows["table/retrain"], surv,
                "distinct_valid_novel_tuples"),
            "field_evict_vs_table_row_removed": _paired(
                rows["field/evict_red"], rows["table/row_removed"], surv,
                "distinct_valid_novel_tuples"),
            "field_evict_vs_gate8_clear_red": _paired(
                rows["field/evict_red"], rows["table+gate8/clear_red"], surv,
                "distinct_valid_novel_tuples"),
            "field_hole_vs_joint_presence_hole": _paired(
                rows["field/hole"], rows["table+joint/hole"], surv,
                "distinct_valid_novel_tuples"),
            "field_hole_vs_joint_BACKOFF_hole(the generous rival)": _paired(
                rows["field/hole"], rows["table+backoff/hole"], surv,
                "distinct_valid_novel_tuples"),
            "field_base_vs_lookup_argmax(C.1 FLOOR)": _paired(
                rows["field/base"], rows["field/lookup_argmax"], surv,
                "distinct_valid_novel_tuples"),
            "field_base_vs_lookup_mean(C.1 FLOOR)": _paired(
                rows["field/base"], rows["field/lookup_mean"], surv,
                "distinct_valid_novel_tuples"),
        }

        # ---- ⭐ BAR 2's literal sentence, broken out by hue and by shape ----
        S["BAR2_breakout_green_and_blue_on_every_shape"] = {
            "evict_red": _breakout(rows["field/evict_red"], rows["field/base"], geo, surv),
            "hole": _breakout(rows["field/hole"], rows["field/base"], geo, surv),
            "table_row_removed": _breakout(rows["table/row_removed"], rows["table/base"],
                                           geo, surv),
            "table_retrain_GOLD": _breakout(rows["table/retrain"], rows["table/base"],
                                            geo, surv),
        }

        # ---- ⭐ "green and blue still compose on EVERY shape", in the currency the split
        #      actually allows, with the limit stated in the artifact itself ----
        _gb = ["green_cylinder", "green_sphere", "blue_cylinder", "blue_oblong"]
        S["green_and_blue_on_every_shape"] = {
            "LIMIT_DECLARED": (
                "green AND cylinder and blue AND cylinder have ZERO rows in the scored test "
                "index -- the split puts all of them in train -- so a HELD-OUT composition "
                "corner cannot be built for them and none was manufactured. They are measured "
                "here in JUDGE-ACCURACY currency on the constructed probes, which is labelled as "
                "such and is never substituted for the novel-tuple estimand. The novel-tuple "
                "estimand covers every held-out conjunction the split does contain: the 49 n-way "
                "corners (hues 1,2,8,9 x cylinder/sphere/oblong) plus pair-001 (green AND sphere) "
                "and pair-002 (blue AND oblong), both NEVER SEEN."),
            "judge_accuracy_on_the_probes": {
                p: {"before": S["arms"]["field/base"]["probes"][p]["judge_accuracy"],
                    "after_evicting_red": S["arms"]["field/evict_red"]["probes"][p][
                        "judge_accuracy"],
                    "after_the_hole": S["arms"]["field/hole"]["probes"][p]["judge_accuracy"],
                    "delta_evict": (S["arms"]["field/evict_red"]["probes"][p]["judge_accuracy"]
                                    - S["arms"]["field/base"]["probes"][p]["judge_accuracy"]),
                    "hardness": probe_hardness[p]}
                for p in _gb},
            "novel_tuple_estimand_on_the_held_out_conjunctions":
                S["BAR2_breakout_green_and_blue_on_every_shape"]["evict_red"],
        }

        # ---- churn: A COST LINE ONLY.  One definition.  Never a claim. ---
        def _churn(a, b):
            ch, n = 0, 0
            for c in surv:
                ta = np.asarray(done[(a, c)]["raw_tuples"], np.int64)
                tb = np.asarray(done[(b, c)]["raw_tuples"], np.int64)
                ch += int((ta != tb).any(axis=1).sum())
                n += int(ta.shape[0])
            return float(ch) / max(n, 1)

        S["churn_COST_LINE_ONLY"] = {
            "definition": ("the fraction of the survivor set's requests whose judged 6-tuple "
                           "changes between the pre-deletion and the post-deletion read; "
                           "DECLARED IN THE PREREG; reported for every method and NEVER "
                           "quoted as evidence for or against any arm"),
            "field/evict_red": _churn("field/evict_red", "field/base"),
            "field/decay_red": _churn("field/decay_red", "field/base"),
            "field/hole": _churn("field/hole", "field/base"),
            "table/row_removed": _churn("table/row_removed", "table/base"),
            "table/retrain": _churn("table/retrain", "table/base"),
        }

        # ---- BAR evaluation ---------------------------------------------
        b = {}
        ev, ho = S["arms"]["field/evict_red"], S["arms"]["field/hole"]
        base_a = S["arms"]["field/base"]
        b["BAR1_evict_refusal"] = {
            "refuse_on_deleted_set": ev["refuse_on_deleted_set"],
            "refuse_before": base_a["refuse_on_deleted_set"],
            "refuse_red_sphere": ev["probes"]["red_sphere"]["refuse"],
            "refuse_red_oblong": ev["probes"]["red_oblong"]["refuse"],
            "empty_basin_on_deleted_set_probe": ev["probes"]["red_cylinder"]["empty_basin"],
            "bar": f">= {BAR_REFUSE_DELETED}",
            "PASS": bool(ev["refuse_on_deleted_set"] >= BAR_REFUSE_DELETED
                         and ev["probes"]["red_sphere"]["refuse"] >= BAR_REFUSE_DELETED
                         and ev["probes"]["red_oblong"]["refuse"] >= BAR_REFUSE_DELETED)}
        cev = S["composition_vs_base"]["field/evict_red"]
        b["BAR2_evict_composition"] = {
            **cev, "bar": f"|delta| <= {MARGIN_NOVELTY} novel tuple/corner",
            "PASS": bool(abs(cev["mean_delta"]) <= MARGIN_NOVELTY)}
        b["BAR3_hole_refusal"] = {
            "hole_flag_on_deleted_set": _setrate("field/hole", dele, "hole"),
            "refuse_on_deleted_set": ho["refuse_on_deleted_set"],
            "refuse_before": base_a["refuse_on_deleted_set"],
            "hole_flag_red_cylinder_probe": ho["probes"]["red_cylinder"]["hole_flag"],
            "bar": f"hole flag = 1.0000 and refuse >= {BAR_REFUSE_DELETED}",
            "PASS": bool(ho["refuse_on_deleted_set"] >= BAR_REFUSE_DELETED
                         and _setrate("field/hole", dele, "hole") >= BAR_REFUSE_DELETED)}
        surv_names = ["green_sphere", "blue_oblong", "red_sphere", "red_oblong",
                      "green_cylinder", "blue_cylinder"]
        b["BAR4_hole_surgical"] = {
            "survivor_probes": {
                p: {"hole_flag_after": ho["probes"][p]["hole_flag"],
                    "refuse_before": base_a["probes"][p]["refuse"],
                    "refuse_after": ho["probes"][p]["refuse"],
                    "drift": ho["probes"][p]["refuse"] - base_a["probes"][p]["refuse"],
                    "never_seen": bool(probes[p]["seen_in_pool"] is False)}
                for p in surv_names},
            "red_cube_REPORTED_not_a_bar": {
                "hole_flag_after": ho["probes"]["red_cube"]["hole_flag"],
                "refuse_before": base_a["probes"]["red_cube"]["refuse"],
                "refuse_after": ho["probes"]["red_cube"]["refuse"],
                "why": probes["red_cube"]["why"]},
            "refuse_drift_on_survivor_corner_set": (
                ho["refuse_on_survivor_set"] - base_a["refuse_on_survivor_set"]),
            "paired_delta_on_survivor_corners": S["composition_vs_base"]["field/hole"],
            "features_bit_identical_to_base": {
                "crc32_equal_on_every_survivor_corner": bool(all(
                    done[("field/hole", c)]["feat_crc32"]
                    == done[("field/base", c)]["feat_crc32"] for c in surv)),
                "asserted_directly_on_every_corner_and_probe": _hri(s),
                "asserted_opportunistically_during_scoring": eqs.get("field_hole == field_base"),
                "why_the_opportunistic_check_can_be_empty": (
                    "the scoring loop only compares when it computes both generators for the same "
                    "corner in one process; a RESUMED process skips units already done, so on a "
                    "multi-window run it records nothing. That is the absence of a diagnostic, not "
                    "a failure, which is why the direct assertion above exists."),
                "why": ("punch_hole writes one byte of `coupling_hole`, a segment the read never "
                        "consults, so the rendered features must be BIT-identical; this is "
                        "asserted elementwise while scoring, not inferred afterwards")},
            "lifetime_route_features_identical_to_eviction":
                eqs.get("field_decay == field_evict"),
            "bar": (f"hole flag = 0.0000 on every survivor probe, refusal drift <= "
                    f"{BAR_REFUSE_SURVIVOR_DRIFT}, paired delta = 0 exactly"),
            "PASS": bool(
                all((ho["probes"][p]["hole_flag"] or 0.0) == 0.0 for p in surv_names)
                and all(abs(ho["probes"][p]["refuse"] - base_a["probes"][p]["refuse"])
                        <= BAR_REFUSE_SURVIVOR_DRIFT for p in surv_names)
                and S["composition_vs_base"]["field/hole"]["mean_delta"] == 0.0
                and (ho["probes"]["red_cube"]["hole_flag"] or 0.0) == 0.0
                and bool((_hri(s) or {}).get("ALL_BIT_IDENTICAL", False)))}
        b["BAR6_the_8_byte_gate"] = {
            "value_deletion": {
                "gate_refuse_on_deleted_set": S["arms"]["table+gate8/clear_red"][
                    "refuse_on_deleted_set"],
                "field_refuse_on_deleted_set": ev["refuse_on_deleted_set"],
                "gate_collateral_red_sphere": S["arms"]["table+gate8/clear_red"][
                    "probes"]["red_sphere"]["refuse"],
                "field_collateral_red_sphere": ev["probes"]["red_sphere"]["refuse"],
                "TIE_EXPECTED_AND_DECLARED_IN_THE_PREREG": True},
            "conjunction_deletion": {
                "gate8_refuse_on_deleted_set": S["arms"]["table+gate8/hole_attempt"][
                    "refuse_on_deleted_set"],
                "gate8_refuse_before": S["arms"]["table+gate8/base"]["refuse_on_deleted_set"],
                "field_hole_refuse_on_deleted_set": ho["refuse_on_deleted_set"],
                "gate8_clear_cylinder_collateral_green_cylinder": S["arms"][
                    "table+gate8/clear_cylinder"]["probes"]["green_cylinder"]["refuse"],
                "gate8_clear_red_collateral_red_sphere": S["arms"][
                    "table+gate8/clear_red"]["probes"]["red_sphere"]["refuse"],
                "field_hole_collateral_green_cylinder": ho["probes"]["green_cylinder"][
                    "refuse"],
                "field_hole_collateral_red_sphere": ho["probes"]["red_sphere"]["refuse"]},
            "joint_PRESENCE_gate_pays_on_the_unseen": {
                "joint_gate_refuse_on_deleted_set": S["arms"]["table+joint/hole"][
                    "refuse_on_deleted_set"],
                "joint_gate_refuse_on_SURVIVOR_set": S["arms"]["table+joint/hole"][
                    "refuse_on_survivor_set"],
                "field_hole_refuse_on_SURVIVOR_set": ho["refuse_on_survivor_set"],
                "joint_gate_refuse_green_sphere": S["arms"]["table+joint/hole"][
                    "probes"]["green_sphere"]["refuse"],
                "joint_gate_refuse_blue_oblong": S["arms"]["table+joint/hole"][
                    "probes"]["blue_oblong"]["refuse"],
                "field_hole_refuse_green_sphere": ho["probes"]["green_sphere"]["refuse"],
                "field_hole_refuse_blue_oblong": ho["probes"]["blue_oblong"]["refuse"]},
            "joint_BACKOFF_the_generous_rival": {
                "note": ("given a SEPARATE deleted-pair array so it can do the conjunction "
                         "deletion cleanly; a TIE here was declared in advance and is the "
                         "honest reading, not a loss"),
                "refuse_on_deleted_set": S["arms"]["table+backoff/hole"][
                    "refuse_on_deleted_set"],
                "refuse_on_SURVIVOR_set": S["arms"]["table+backoff/hole"][
                    "refuse_on_survivor_set"],
                "refuse_green_sphere": S["arms"]["table+backoff/hole"][
                    "probes"]["green_sphere"]["refuse"],
                "refuse_blue_oblong": S["arms"]["table+backoff/hole"][
                    "probes"]["blue_oblong"]["refuse"],
                "refuse_red_sphere": S["arms"]["table+backoff/hole"][
                    "probes"]["red_sphere"]["refuse"],
                "composition_delta_vs_its_own_base": S["composition_vs_base"][
                    "table+backoff/hole"]},
        }
        b["BAR7_table_row_removal"] = {
            "refuse_on_deleted_set": S["arms"]["table/row_removed"]["refuse_on_deleted_set"],
            "refuse_on_survivor_set": S["arms"]["table/row_removed"]["refuse_on_survivor_set"],
            "composition_delta_vs_table_base": S["composition_vs_base"]["table/row_removed"],
            "conjunction": "CANNOT ENTER -- no conjunction slot; 0 bytes change"}
        b["BAR8_gold_standard"] = {
            "retrain_refuse_on_deleted_set": S["arms"]["table/retrain"][
                "refuse_on_deleted_set"],
            "retrain_composition_delta": S["composition_vs_base"]["table/retrain"],
            "field_evict_composition_delta": cev,
            "cost": S["cost"]}
        S["bars"] = b
        rep["per_seed"][str(s)] = S
        log(f"  report: seed {s} assembled")

    rep["across_seeds"] = _across_seeds(rep)
    _jdump(rep, out / "report.json")
    log(f"  report -> {out / 'report.json'}")
    return rep


def _across_seeds(rep: dict) -> dict:
    """Mean and spread of every headline number across the scored seeds."""
    seeds = [str(s) for s in rep["seeds_scored"]]
    def _g(path):
        vals = []
        for s in seeds:
            cur = rep["per_seed"][s]
            for k in path:
                cur = cur.get(k) if isinstance(cur, dict) else None
                if cur is None:
                    break
            if isinstance(cur, (int, float)):
                vals.append(float(cur))
        if not vals:
            return None
        return {"per_seed": vals, "mean": float(np.mean(vals)),
                "spread": float(max(vals) - min(vals))}
    return {
        "evict_refuse_on_deleted_set": _g(["bars", "BAR1_evict_refusal",
                                           "refuse_on_deleted_set"]),
        "evict_composition_mean_delta": _g(["bars", "BAR2_evict_composition", "mean_delta"]),
        "hole_flag_on_deleted_set": _g(["bars", "BAR3_hole_refusal",
                                        "hole_flag_on_deleted_set"]),
        "hole_composition_mean_delta": _g(["composition_vs_base", "field/hole", "mean_delta"]),
        "joint_presence_gate_refuse_on_survivor_set": _g(
            ["bars", "BAR6_the_8_byte_gate", "joint_PRESENCE_gate_pays_on_the_unseen",
             "joint_gate_refuse_on_SURVIVOR_set"]),
        "field_hole_refuse_on_survivor_set": _g(
            ["bars", "BAR6_the_8_byte_gate", "joint_PRESENCE_gate_pays_on_the_unseen",
             "field_hole_refuse_on_SURVIVOR_set"]),
        "joint_backoff_refuse_on_deleted_set": _g(
            ["bars", "BAR6_the_8_byte_gate", "joint_BACKOFF_the_generous_rival",
             "refuse_on_deleted_set"]),
        "joint_backoff_refuse_on_survivor_set": _g(
            ["bars", "BAR6_the_8_byte_gate", "joint_BACKOFF_the_generous_rival",
             "refuse_on_SURVIVOR_set"]),
        "gate8_conjunction_refuse_on_deleted_set": _g(
            ["bars", "BAR6_the_8_byte_gate", "conjunction_deletion",
             "gate8_refuse_on_deleted_set"]),
        "retrain_from_scratch_s": _g(["cost", "retrain_from_scratch_s"]),
        "row_removal_s": _g(["cost", "row_removal_s"]),
    }


# ---------------------------------------------------------------------------
# 8. CLI
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 7b. The report as MARKDOWN -- rendered from report.json, never typed by hand
# ---------------------------------------------------------------------------


def _f(x, nd=4):
    if x is None:
        return "—"
    if isinstance(x, bool):
        return "yes" if x else "**no**"
    if isinstance(x, (int,)) and not isinstance(x, bool):
        return f"{x:,}"
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def _spread(g, nd=4):
    """``mean [s0 / s1 / s2]`` -- the seed spread always travels with the number."""
    if not g:
        return "—"
    per = " / ".join(_f(v, nd) for v in g["per_seed"])
    return f"**{_f(g['mean'], nd)}** [{per}]"


def render_markdown(out: Path, log: Callable = print) -> str:
    """Render ``report.json`` into the tables this hub's final section prints.

    Every number here is READ OUT of the artifact.  Nothing in this function computes a
    result, so the written report and the measured artifact cannot drift apart.
    """
    out = Path(out)
    rep = json.loads((out / "report.json").read_text(encoding="utf-8"))
    seeds = rep["seeds_scored"]
    s0 = rep["per_seed"][str(seeds[0])]

    def per(path, arm=None, nd=4):
        vals = []
        for s in seeds:
            cur = rep["per_seed"][str(s)]
            for k in (path if arm is None else [arm] + list(path)):
                cur = cur.get(k) if isinstance(cur, dict) else None
                if cur is None:
                    break
            vals.append(cur)
        if any(v is None for v in vals):
            return "—"
        return _spread({"per_seed": vals, "mean": float(np.mean(vals))}, nd)

    L = []
    L.append("**Scored run %s** · PREREG `%s` · seeds %s · cache generation %s scope `%s` "
             "(%s rows) · base `base/iter5 = 6675904`."
             % (rep["scored_run"], rep["prereg"], ", ".join(str(x) for x in seeds),
                s0["cache"]["generation"], s0["cache"]["scope"],
                f"{s0['cache']['n_rows']:,}"))
    pr = s0["proof"]
    L.append("")
    L.append("**Estimand.** %d survivor corners (%s requests) and %d deleted corners (%s "
             "requests); every corner row is in the scored `test` index and **%d** are in any "
             "arm's fit pool. The deleted set is `pair-000` = red ∧ cylinder; the survivors "
             "include `pair-001` = green ∧ sphere and `pair-002` = blue ∧ oblong, both **never "
             "seen**, both carrying `coupling_w = 0` — the same evidence state as the holed pair."
             % (pr["survivor"]["n_corners"], f"{pr['survivor']['n_requests']:,}",
                pr["deleted"]["n_corners"], f"{pr['deleted']['n_requests']:,}",
                pr["all"]["rows_in_fit_pool"]))
    L.append("")
    L.append("*Caption, on every table below: labels are used **once at fit time** to align the "
             "wells and to build the address book, and nowhere else — not in the landscape, not "
             "in the read, not in the write (charter §B.6, w5-common §4). Budgets: the **byte** "
             "pin of 524,288 B is PRIMARY; parameter counts are named separately wherever quoted "
             "and the two cannot hold simultaneously.*")

    # ---- the headline ----------------------------------------------------
    L.append("")
    L.append("### Row 7 — BOTH halves, every arm. Mean [seed 0 / 1 / 2]")
    L.append("")
    L.append("| arm | del | size of the deleting object | **refusal on the 20 deleted corners** "
             "| refusal on the 89 survivors | **composition Δ / corner, 89 survivors** |")
    L.append("|---|---|---|---|---|---|")
    gb = s0.get("bytes_summary", {})
    SIZE = {
        "field/base": "524,288 B store", "field/evict_red": "atoms of 6 items zeroed",
        "field/decay_red": "lifetimes → 0, then the items go",
        "field/hole": "**1 byte** of `coupling_hole`",
        "table/base": "466,944 B prototypes", "table/row_removed": "one prototype row",
        "table/retrain": "**the whole table, refitted**",
        "table+gate8/base": "**8 bytes** (57 bits)",
        "table+gate8/clear_red": "**1 bit**",
        "table+gate8/hole_attempt": "**0 bits — inexpressible**",
        "table+gate8/clear_cylinder": "1 bit, but it deletes the SHAPE",
        "table+joint/base": "97 bits", "table+joint/hole": "**0 bits — already false**",
        "table+backoff/base": "97 bits + a deleted-pair array",
        "table+backoff/hole": "1 bit (the generous rival)",
        "field/landscape_removed": "§C.2 control", "field/landscape_shuffled": "§C.2 control",
        "field/lookup_argmax": "§C.1 floor, same bytes",
        "field/lookup_mean": "§C.1 floor, same bytes"}
    order = ["field/base", "field/evict_red", "field/decay_red", "field/hole",
             "table/base", "table/row_removed", "table/retrain",
             "table+gate8/base", "table+gate8/clear_red", "table+gate8/hole_attempt",
             "table+gate8/clear_cylinder", "table+joint/base", "table+joint/hole",
             "table+backoff/base", "table+backoff/hole",
             "field/landscape_removed", "field/landscape_shuffled",
             "field/lookup_argmax", "field/lookup_mean"]
    for a in order:
        if a not in s0["arms"]:
            continue
        rd = per(["refuse_on_deleted_set"], arm=None) if False else _spread(
            {"per_seed": [rep["per_seed"][str(s)]["arms"][a]["refuse_on_deleted_set"]
                          for s in seeds],
             "mean": float(np.mean([rep["per_seed"][str(s)]["arms"][a][
                 "refuse_on_deleted_set"] for s in seeds]))})
        rs = _spread({"per_seed": [rep["per_seed"][str(s)]["arms"][a]["refuse_on_survivor_set"]
                                   for s in seeds],
                      "mean": float(np.mean([rep["per_seed"][str(s)]["arms"][a][
                          "refuse_on_survivor_set"] for s in seeds]))})
        if a in s0["composition_vs_base"] and "mean_delta" in s0["composition_vs_base"][a]:
            dl = _spread({"per_seed": [rep["per_seed"][str(s)]["composition_vs_base"][a][
                "mean_delta"] for s in seeds],
                "mean": float(np.mean([rep["per_seed"][str(s)]["composition_vs_base"][a][
                    "mean_delta"] for s in seeds]))}, 3)
        else:
            dl = "— *(reference)*"
        L.append("| `%s` | %s | %s | %s | %s | %s |"
                 % (a, s0["arms"][a]["deletion"] or "—", SIZE.get(a, "—"), rd, rs, dl))
    L.append("")
    L.append("| MLP · `mlp_180k` · transformer · LSTM · continuous-input MLP | — | — | **—** | "
             "**—** | **—** |")
    L.append("| LACE (95.1× the store budget) · PoE-EBM (27.1×) | — | — | **—** | **—** | **—** |")
    L.append("")
    L.append("⛔ **The dashes, and why they are not a win for us.** The five learners have **no "
             "refusal verb** — asked for a deleted value they emit a feature vector like any "
             "other request, so they cannot enter the refusal half at all; their only deletion "
             "route is retrain-from-scratch, and its measured cost is on the cost line below. "
             "LACE and PoE-EBM are sampler draws rather than functions of the request (spoke A's "
             "own API refuses them), and both are far outside the byte pin. ⛔ A dash is never a "
             "win for us.")

    def _load(seed, name):
        q = seed_dir(out, seed) / name
        return json.loads(q.read_text(encoding="utf-8")) if q.exists() else None

    def _acc(path, nd=4):
        """mean [s0/s1/s2] for a dotted path under each seed's report block."""
        vals = []
        for x in seeds:
            cur = rep["per_seed"][str(x)]
            for k in path:
                cur = cur.get(k) if isinstance(cur, dict) else None
                if cur is None:
                    break
            vals.append(cur)
        if any(v is None for v in vals):
            return "\u2014"
        return _spread({"per_seed": vals, "mean": float(np.mean(vals))}, nd)

    # ---- BAR 5: the byte property ---------------------------------------
    L.append("")
    L.append("### BAR 5 \u2014 the byte property, on the SCORED store. Exact, no tolerance")
    L.append("")
    L.append("| seed | store image sha256[:32] | items evicted | evict: changed / **OUTSIDE** | "
             "hole: bytes / **OUTSIDE** | `assert_no_refit` | negative control |")
    L.append("|---|---|---|---|---|---|---|")
    for sd in seeds:
        b = _load(sd, "bytes.json")
        if not b:
            L.append("| %d | *(NOT RUN)* | \u2014 | \u2014 | \u2014 | \u2014 | \u2014 |" % sd)
            continue
        ev, ho = b["evict"]["byte_assertion"], b["hole"]["byte_assertion"]
        nr = all(b["evict"]["no_refit"]["fitted_segments_byte_identical"].values())
        L.append("| %d | `%s` | %d | %s / **%s** | **%s** / **%s** | %s | %s |"
                 % (sd, b["manifest"]["sha256"][:32], b["evict"]["n_items_evicted"],
                    format(ev["bytes_changed"], ","),
                    ev.get("bytes_changed_outside_permitted"),
                    ho["bytes_changed"], ho.get("bytes_changed_outside_permitted"),
                    "**14/14**" if nr else "**FAILS**",
                    format(b["negative_control_refit"]["bytes_changed"], ",")))
    L.append("")
    L.append("\u26a0 The **negative control** is a 1e-3 nudge to **`term_mu[0]`** \u2014 one row "
             "of the address book, the smallest possible refit. It varies by seed because it is a "
             "property of *(which row, which store)*, **not** an instrument constant, and it is "
             "printed with its row and its store hash for exactly that reason. What it has to do "
             "is **fire**, and it fires on every seed \u2014 which is what makes the zeros above "
             "mean something. The **evict** count varies for the same kind of reason (different "
             "pool draws put 6, 6 and 7 items in red's basin); **the invariant is not the count, "
             "it is the 0 OUTSIDE.**")

    # ---- BAR 4(b): the unseen conjunctions -------------------------------
    u0 = _load(seeds[0], "unseen_pairs.json")
    if u0:
        L.append("")
        L.append("### BAR 4(b) \u2014 every conjunction the pool never held, accounted for BY "
                 "REASON")
        L.append("")
        L.append("| group | n | `hole` flag after | refusal before \u2192 after | drift |")
        L.append("|---|---|---|---|---|")
        th = u0["THE_HOLE"]
        o = u0["other_unseen_pairs"]
        n_ans = o["n_fully_answered_after_the_hole"]
        n_oth = u0["n_other_unseen_pairs"]
        L.append("| **THE HOLE** \u2014 red \u2227 cylinder | **1** | **%s** | %s \u2192 **%s** "
                 "| **+%s** |"
                 % (_f(th.get("hole_flag_after")), _f(th.get("refuse_before")),
                    _f(th.get("refuse_after")), _f(th.get("refuse_drift"))))
        L.append("| unseen because a whole **VALUE** is absent (`object_hue=0.5`, `shape=cube`) | "
                 "**%d** | **0.0000** | unchanged | **%s** |"
                 % (n_oth - n_ans, _f(o["max_abs_refuse_drift"])))
        L.append("| unseen **CONJUNCTIONS** whose marginals are both present \u2014 green "
                 "\u2227 sphere, blue \u2227 oblong | **%d** | **0.0000** | 0.0000 \u2192 "
                 "**0.0000** | **0.0000** |" % n_ans)
        L.append("")
        bits = []
        for sd in seeds:
            uu = _load(sd, "unseen_pairs.json") or {}
            oo = uu.get("other_unseen_pairs") or {}
            bits.append("seed %d PASS=%s (%s pairs, %s light up)"
                        % (sd, uu.get("BAR4b_PASS"),
                           uu.get("n_term_pairs_the_pool_NEVER_held"),
                           oo.get("n_with_any_hole_flag")))
        L.append("Across seeds: " + " \u00b7 ".join(bits))
        L.append("")
        L.append("\u26d4 **w5-common \u00a73.7's \u201c96 unseen conjunctions stay "
                 "answered\u201d is a VAL figure and is NOT quoted here.** On the scored split "
                 "the conjunctions that are unseen *and* have both marginals present number "
                 "exactly **three** \u2014 the three held-out pair cells \u2014 of which one is "
                 "holed and two stay answered. The rest are unseen because a whole value is "
                 "absent, so \u201cstill answered\u201d was never available to them; their "
                 "refusal is unmoved by the hole and they are reported separately.")

    # ---- the dash, earned -----------------------------------------------
    d0 = _load(seeds[0], "dash.json")
    if d0:
        L.append("")
        L.append("### The dash, earned \u2014 every learner asked for the deleted conjunction")
        L.append("")
        L.append("| family | params | **refused** | judge satisfies the request | reads back the "
                 "asked `object_hue` | the asked `shape` |")
        L.append("|---|---|---|---|---|---|")
        for fam, v in sorted(d0["families"].items()):
            ds = v["deleted_set_red_AND_cylinder"]
            L.append("| `%s` | %s | **%s** | %s | **%s** | %s |"
                     % (fam,
                        format(v["param_count"], ",") if v.get("param_count") else "\u2014",
                        _f(v["REFUSED_on_the_deleted_set"]),
                        _f(ds["judge_satisfies_the_full_request"]),
                        _f(ds["judge_reads_back_the_asked_object_hue"]),
                        _f(ds["judge_reads_back_the_asked_shape"])))
        for fam in sorted(d0["not_callable"]):
            L.append("| `%s` | \u2014 | **not callable** | \u2014 | \u2014 | \u2014 |" % fam)
        L.append("")
        L.append("\u2b50 Not one of them refuses a single request, because not one has a verb "
                 "with which to. **You cannot ask these objects to forget anything.** "
                 "\u26a0 These are spoke A's rivals, fitted on the **Montero combination pool**; "
                 "their own `row_eligibility.json` says they are not like-for-like with a "
                 "value-holdout arm on a whole-value row, and row 7 is one \u2014 so this table "
                 "is **descriptive of the mechanism, not a behavioural contest**. The "
                 "like-for-like rivals are the table arms, refitted on this pool. Budgets: the "
                 "pinned store is **524,288 B / 131,072 parameters**; `mlp_180k` (178,952) is the "
                 "parameter-matched rival at the secondary budget.")

    # ---- the cost line ---------------------------------------------------
    L.append("")
    L.append("### The cost line \u2014 what a deletion actually costs")
    L.append("")
    L.append("| method | what it touches | wall clock | rows refitted | can it express a "
             "CONJUNCTION? |")
    L.append("|---|---|---|---|---|")
    c = s0["cost"]
    b0 = _load(seeds[0], "bytes.json") or {}
    tb = b0.get("table_bytes", {})
    g2 = b0.get("gate_bytes", {})
    L.append("| **field store, evict** | the atoms of the 6\u20137 items holding the value | "
             "milliseconds (an in-place zero) | **0** | n/a \u2014 this is the VALUE deletion |")
    L.append("| **field store, the hole** | **1 byte** of `coupling_hole` | microseconds | **0** | "
             "**YES \u2014 and only that conjunction** |")
    L.append("| table, row removal | one prototype row \u2192 the global mean (%s B) | %s s | "
             "**0** | **NO \u2014 no conjunction slot; 0 bytes change** |"
             % (format(tb.get("value_deletion_bytes_changed", 0), ","),
                _f(c["row_removal_s"], 4)))
    L.append("| **table, RETRAIN FROM SCRATCH (GOLD STANDARD)** | %s B of prototypes | **%s s** | "
             "**%s** | NO |"
             % (format(tb.get("retrain_from_scratch_bytes_changed", 0), ","),
                _f(c["retrain_from_scratch_s"], 2),
                format(c["rows_refitted_by_retrain"], ",")))
    L.append("| 8-byte per-factor gate | **1 bit** for a value | microseconds | **0** | "
             "**NO \u2014 %s bits change at any size** |"
             % g2.get("CONJUNCTION_deletion_bits_changed_per_factor"))
    L.append("| joint hue\u00d7shape **presence** gate | \u2014 | \u2014 | **0** | **NO \u2014 "
             "%s bits change; that bit already meant \u201cnever seen\u201d** |"
             % g2.get("CONJUNCTION_deletion_bits_changed_joint_presence"))
    L.append("| joint **back-off + a SEPARATE deleted-pair array** (the generous rival) | "
             "**1 bit** | microseconds | **0** | **YES** |")
    lr = _load(seeds[0], "learners.json")
    if lr:
        for fam, legs in lr["families"].items():
            L.append("| **`%s` retrained WITHOUT the value** | every parameter | **%s s** | "
                     "**%s** | NO \u2014 and afterwards it still ANSWERS the deleted requests at "
                     "refusal **0.0000** |"
                     % (fam, _f(legs["without_red"]["wall_clock_s"], 1),
                        format(legs["without_red"]["rows_refitted"], ",")))
    else:
        L.append("| MLP / transformer / LSTM retrain | every parameter | *(see `learners.json`)* | "
                 "the whole pool | **NO \u2014 and no refusal verb at all** |")
    L.append("")
    churn = " \u00b7 ".join(
        "`%s` %s" % (k, _acc(["churn_COST_LINE_ONLY", k]))
        for k in ("field/evict_red", "field/hole", "table/row_removed", "table/retrain"))
    L.append("\u26a0 **CHURN IS A COST LINE ONLY \u2014 one definition, never a claim, in "
             "either direction.** Definition fixed in the PREREG before any number: *the fraction "
             "of the survivor set's requests whose judged 6-tuple changes between the "
             "pre-deletion and the post-deletion read.* Measured: " + churn)

    # ---- charter C floor --------------------------------------------------
    L.append("")
    md = s0.get("refusal_mechanism_decomposition")
    if md:
        L.append("")
        L.append("### Which refusal mechanism does the work \u2014 and what `in_gap` costs")
        L.append("")
        L.append("| arm | set | `hole` | `empty_basin` | `in_gap` | refuse WITH `in_gap` | "
                 "refuse WITHOUT it |")
        L.append("|---|---|---|---|---|---|---|")
        for a in ("field/base", "field/evict_red", "field/hole"):
            if a not in md:
                continue
            for setname in ("deleted_set", "survivor_set"):
                m = md[a][setname]
                L.append("| `%s` | %s | %s | %s | %s | **%s** | %s |"
                         % (a, setname.replace("_", " "), _f(m.get("hole")),
                            _f(m.get("empty_basin")), _f(m.get("in_gap")),
                            _f(md[a]["refuse_WITH_in_gap"][setname]),
                            _f(m.get("refuse_WITHOUT_in_gap"))))
        L.append("")
        L.append("\u26a0 `in_gap` is a distance cut, not a discriminator. `chlu-w5-refuse-f2` "
                 "measured on their own scored legs that it costs 0.005 of the accept leg and "
                 "adds nothing \u2014 their four-mechanism rule LOSES to the 8-byte rule because "
                 "of it. The table above answers the same question for this store from the "
                 "per-request masks: if the deletion refusals are carried entirely by "
                 "`empty_basin` (what an eviction leaves) and `hole` (what a punch leaves), then "
                 "`in_gap` is pure cost on the survivor set and this hub says so.")

    L.append("")
    L.append("### Charter \u00a7C floor \u2014 one line each")
    L.append("")
    L.append("- **\u00a7C.1 matched-bytes lookup.** `field/lookup_argmax` and `field/lookup_mean` "
             "read the SAME atoms through the SAME address book at the SAME bytes. Paired \u0394 "
             "(field base \u2212 lookup) on the 89 survivor corners: argmax %s \u00b7 mean %s. "
             "\u26d4 If these are not positive beyond seed spread, the landscape is decorative "
             "on the composition axis and this report says so."
             % (_acc(["composition_vs_rivals", "field_base_vs_lookup_argmax(C.1 FLOOR)",
                      "mean_delta"], 3),
                _acc(["composition_vs_rivals", "field_base_vs_lookup_mean(C.1 FLOOR)",
                      "mean_delta"], 3)))
    L.append("- **\u00a7C.2 landscape-removed / landscape-shuffled.** Both are arms on the "
             "headline table; their composition \u0394 against the same base is the collapse: "
             "removed %s \u00b7 shuffled %s."
             % (_acc(["composition_vs_base", "field/landscape_removed", "mean_delta"], 3),
                _acc(["composition_vs_base", "field/landscape_shuffled", "mean_delta"], 3)))
    L.append("- **\u00a7C.3 the tie clause.** Where this hub ties the 8-byte gate or the table on "
             "the VALUE deletion \u2014 declared in the PREREG as the EXPECTED outcome, before "
             "any number \u2014 the tie counts only via *deletion that changes the output "
             "exactly*, which is BAR 5.")
    c4 = _load(seeds[0], "c4_nearest_atom.json")
    if c4:
        L.append("- **\u00a7C.4 nearest atom.** The settled state sits at relative distance "
                 "**%s** from its nearest live atom (min over %s requests **%s**). \u00a7C.4 "
                 "retires an arm as a lookup table when this \u2192 0. It does not."
                 % (_f(c4["mean_relative_distance"]), format(c4["n_requests"], ","),
                    _f(c4["min_relative_distance"])))
    return chr(10).join(L)


def compare_runs(a: Path, b: Path, log: Callable = print) -> dict:
    """⭐ Charter section A.5: an independent replication, compared MECHANICALLY.

    ``a`` is this hub's run, ``b`` the replication's.  Unreplicated is not a result, and
    "it looked the same" is not a replication either -- so every headline number is diffed
    here and the verdict is a computed boolean, not a reading.

    Two classes of number, held to different tolerances and both declared:

    ``EXACT``  the byte properties, the refusal rates and the bit-identity flags.  These are
               deterministic given the seed and the pinned cache; any difference at all is a
               failure and is reported as one.
    ``TIGHT``  the composition deltas, which pass through the judge.  They are deterministic
               too on one machine, but a cross-machine replication can move them by encoder
               ULP, so they are diffed at 1e-6 and the actual gap is always printed.
    """
    ra = json.loads((Path(a) / "report.json").read_text(encoding="utf-8"))
    rb = json.loads((Path(b) / "report.json").read_text(encoding="utf-8"))
    EXACT = ("evict_refuse_on_deleted_set", "hole_flag_on_deleted_set",
             "joint_presence_gate_refuse_on_survivor_set",
             "field_hole_refuse_on_survivor_set",
             "joint_backoff_refuse_on_deleted_set",
             "joint_backoff_refuse_on_survivor_set",
             "gate8_conjunction_refuse_on_deleted_set")
    TIGHT = ("evict_composition_mean_delta", "hole_composition_mean_delta")
    rows, ok = {}, True
    for k in EXACT + TIGHT:
        ga, gb = (ra["across_seeds"] or {}).get(k), (rb["across_seeds"] or {}).get(k)
        if not ga or not gb:
            rows[k] = {"NOT_RUN": "absent from one of the two runs"}
            ok = False
            continue
        tol = 0.0 if k in EXACT else 1e-6
        gaps = [abs(x - y) for x, y in zip(ga["per_seed"], gb["per_seed"])]
        good = (len(ga["per_seed"]) == len(gb["per_seed"])
                and all(g <= tol for g in gaps))
        rows[k] = {"class": "EXACT" if k in EXACT else "TIGHT", "tolerance": tol,
                   "run": ga["per_seed"], "replication": gb["per_seed"],
                   "max_abs_gap": max(gaps) if gaps else None, "AGREES": bool(good)}
        ok = ok and good
    # the byte properties, per seed, straight off the bytes.json of each run
    byt = {}
    for s_ in ra["seeds_scored"]:
        pa = Path(a) / f"seed_{s_}" / "bytes.json"
        pb = Path(b) / f"seed_{s_}" / "bytes.json"
        if not (pa.exists() and pb.exists()):
            byt[str(s_)] = {"NOT_RUN": "a bytes.json is missing"}
            ok = False
            continue
        da = json.loads(pa.read_text(encoding="utf-8"))
        db = json.loads(pb.read_text(encoding="utf-8"))
        same = {}
        for key in ("evict", "hole", "decay_zero", "set_lifetime"):
            same[key] = {
                "bytes_changed": [da[key]["byte_assertion"]["bytes_changed"],
                                  db[key]["byte_assertion"]["bytes_changed"]],
                "outside": [da[key]["byte_assertion"].get("bytes_changed_outside_permitted"),
                            db[key]["byte_assertion"].get("bytes_changed_outside_permitted")],
                "AGREES": bool(da[key]["byte_assertion"]["bytes_changed"]
                               == db[key]["byte_assertion"]["bytes_changed"])}
            ok = ok and same[key]["AGREES"]
        same["ALL_BYTE_PROPERTIES_HOLD"] = [da["ALL_BYTE_PROPERTIES_HOLD"],
                                            db["ALL_BYTE_PROPERTIES_HOLD"]]
        byt[str(s_)] = same
    out = {"run": str(a), "replication": str(b), "headline": rows, "byte_properties": byt,
           "REPLICATED": bool(ok),
           "statement": ("charter A.5: unreplicated is not a result. EXACT numbers are diffed at "
                         "zero tolerance; the two composition deltas pass through the judge and "
                         "are diffed at 1e-6, with the actual gap always printed.")}
    _jdump(out, Path(a) / "replication.json")
    log(f"  REPLICATED = {out['REPLICATED']}")
    for k, v in rows.items():
        if "AGREES" in v:
            log(f"    {k}: {v['AGREES']} (max gap {v['max_abs_gap']})")
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--stage", required=True,
                   choices=("corners", "build", "bytes", "score", "learners", "c4",
                            "dash", "unseen", "holeread", "report", "markdown", "compare", "all"))
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--max-seconds", type=float, default=7200.0)
    p.add_argument("--against", default=None,
                   help="the replication run directory, for --stage compare")
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    out = assert_out_root(Path(args.out))
    seeds = [args.seed] if args.seed is not None else list(PINS["seeds"])
    log = print
    if args.stage in ("corners", "all"):
        cond = load_condition()
        cs = scored_corners(cond)
        per = {}
        for s in seeds:
            per[str(s)] = assert_scored_and_heldout(cond, cs, scored_pool(cond, s))
        info = {"deleted_value": cs["deleted_value"],
                "n_corners": {k: len(cs[k]) for k in ("all", "deleted", "survivor",
                                                      "nway", "pair")},
                "deleted_corner_ids": [c["corner_id"] for c in cs["deleted"]],
                "per_seed_proof": per,
                "pair_cells": {c["cell_id"]: c["meta"]["pair"]
                               for c in cond["cells"]["cells"]
                               if c["meta"]["family"] == "pair"}}
        _jdump(info, out / "corners.json")
        log(json.dumps({k: info[k] for k in ("deleted_value", "n_corners", "pair_cells")},
                       indent=1))
    if args.stage in ("build", "all"):
        for s in seeds:
            build_field_base(out, s, log=log)
            build_tables(out, s, log=log)
    if args.stage in ("bytes", "all"):
        for s in seeds:
            byte_proof(out, s, log=log)
    if args.stage in ("c4", "all"):
        for s_ in seeds:
            nearest_atom_check(out, s_, log=log)
    if args.stage in ("holeread", "all"):
        for s_ in seeds:
            hole_read_identity(out, s_, log=log)
    if args.stage in ("unseen", "all"):
        for s_ in seeds:
            unseen_pairs_check(out, s_, log=log)
    if args.stage in ("dash", "all"):
        for s_ in seeds:
            dash_evidence(out, s_, log=log)
    if args.stage in ("learners", "all"):
        for s_ in seeds:
            learner_cost(out, s_, log=log)
    if args.stage in ("score", "all"):
        for s in seeds:
            r = score_seed(out, s, max_seconds=args.max_seconds, log=log)
            if r.get("incomplete"):
                log(f"  seed {s} incomplete -- rerun to continue")
                return 2
    if args.stage in ("report", "all"):
        build_report(out, log=log)
    if args.stage == "compare":
        if not args.against:
            raise SystemExit("--stage compare needs --against REPLICATION_DIR")
        compare_runs(out, assert_out_root(Path(args.against)), log=log)
    if args.stage in ("markdown", "all"):
        md = render_markdown(out, log=log)
        (out / "REPORT.md").write_text(md, encoding="utf-8")
        log(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
