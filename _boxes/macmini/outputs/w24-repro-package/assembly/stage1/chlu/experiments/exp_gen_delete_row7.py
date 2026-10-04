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
import pickle
import time
import zlib
from pathlib import Path
from typing import Callable
import numpy as np
from .exp_gen_factor_wells import Judge, _jdump, accept_rule, build_partitions, features_mmap, feature_cache_manifest, fit_pool, gather, judge_valid, load_condition, nway_corners, pair_subcorners, seen_tuple_mask, value_presence_bitmap
from .exp_gen_field_store import PINS as FSPINS, _host_factors, field_spec
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
MARGIN_NOVELTY = 1.0
SHAPE_NAMES = {0: "cube", 1: "cylinder", 2: "sphere", 3: "oblong"}
HUE_NAMES = {0: "red", 3: "green", 7: "blue"}
def _now() -> float:
    return time.time()


def assert_generation2(seed: int) -> dict:
    """Refuse to run on a missing or truncated feature cache for ``seed``."""
    man = feature_cache_manifest(int(seed))
    if not man["present"]:
        raise AssertionError(
            f"seed {seed}: no feature cache at {man['file']}. Set NCLU_FEATURES_ROOT (README, Data).")
    if not man["size_ok"]:
        raise AssertionError(f"seed {seed}: feature cache {man['file']} has the wrong size.")
    return man


def assert_out_root(out: Path) -> Path:
    p = Path(out).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def seed_dir(out: Path, seed: int) -> Path:
    return Path(out) / f"seed_{int(seed)}"


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


LEARNER_EPOCHS, LEARNER_LR, LEARNER_WD = 20, 1e-3, 0.0
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


