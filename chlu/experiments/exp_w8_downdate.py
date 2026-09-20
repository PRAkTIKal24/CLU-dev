"""w8 spoke/downdate-build -- THE SINGLE DESIGN. Build only. ⛔ No scored run, no claim.

What this module does, in one sentence
--------------------------------------
It builds :class:`chlu.core.downdate_store.DowndatableStore` on the incumbent's own
partition and fit pool, discharges the **three assertions that are the deliverable**,
smokes the object through the shared frozen decoder and judge, and publishes
``DOWNDATE-READY.json`` -- which ``hub/single-design`` is blocked on.

⛔ Nothing here scores anything.  Every number this module prints is a property of the
object (bytes, deviations, ranks, counts) or a ``val`` smoke, and ``val`` is the
selection condition that charter A.2 makes free.  ``hub/single-design`` scores it.

THE THREE ASSERTIONS
--------------------
1. ⭐⭐ ``--stage assertions`` :: the downdated store **equals** the store fitted WITHOUT
   those rows, max absolute deviation printed per well, plus a **negative control** that
   nudges one atom and requires the same check to FIRE.  A check that has never failed
   is not evidence.
2. ⛔ ``--stage assertions`` :: **no pass over the data**, asserted mechanically by a
   tripwire on every route to the corpus AND a row counter that is invariant to corpus
   size -- run at two corpus sizes so "it read few rows" becomes "it read the same rows
   whatever the corpus is".
3. ⛔ ``--stage assertions`` :: **items are never kept**, asserted mechanically by an
   exhaustive inventory of every array in the object plus the object's byte count at two
   corpus sizes.

Stages
------
``fit``         build the store, print the ledgers, publish ``DOWNDATE-READY`` PARTIAL.
``assertions``  the three assertions and their controls; updates the readiness file.
``smoke``       fit this store's own address book, read the ``val`` corners through the
                shared frozen decoder and judge, beside ``pca_tensor`` on the same
                corners; updates the readiness file to ``ready``.
``sweep``       ⚠ arithmetic only, no fit: what the covariance would cost at other K.
                Requested by ``hub/single-design``; it changes nothing this spoke builds.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np

from ..core.downdate_store import (
    EIG_REL_TOL, ROWS_CONSUMED, DowndatableStore, assert_layout_matches_field_store,
    compare_stores, no_data_pass, retention_audit,
)

#: ⭐ The pins this spoke owns.  ⛔ Every one of them is fixed by the task file; none is
#: a dial this spoke is free to move.
W8_PINS = {
    "store_bytes": 524_288,          # the primary pin, tolerance 0 (task section 2)
    "n_directions": 64,              # A -- "a fixed 64 directions" (task section 2)
    "partition": "factor (K=5, the incumbent's own inherited label-MI wells)",
    "content_from": "exp_w5_compose_field.build_pca_content's partition and fit pool",
    "eig_rel_tol": EIG_REL_TOL,      # the well-posedness cut; see downdate_store's docstring
    # the deletion is NOT a choice: it is pinned in the programme's own code at
    # exp_gen_factor_wells.py:1348 -- object_hue = 0.0, pair-000's hue.
    "delete_factor": 2,
    "delete_value_index": 0,
    "small_delete_n": 64,            # the second, tiny deletion: exactness is not a
                                     # property of the deletion's size
    "halfsize_frac": 0.5,            # the second corpus size, for assertions 2 and 3
}

OUT_ROOT = Path(os.environ.get("CHLU_HUB_OUTPUTS", "/c/chlu/outputs")) / "w8-downdate"
READY = OUT_ROOT / "DOWNDATE-READY.json"


def _log(msg: str = "") -> None:
    """⚠ The console here is cp1252; the programme's markers are not. Never let a
    glyph kill a run -- degrade the character, never the line."""
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "ascii"
        print(line.encode(enc, "replace").decode(enc, "replace"), flush=True)


def _jdump(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_jsonable(obj), fh, indent=2, sort_keys=True)


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(float(o)) else float(o)
    if isinstance(o, float):
        return None if not np.isfinite(o) else o
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(x) for x in o]
    return o


# ---------------------------------------------------------------------------
# 0. The readiness file -- a hub is blocked on it, so it is written EARLY and often
# ---------------------------------------------------------------------------


def publish_ready(**fields) -> dict:
    """⭐⭐ Write/merge ``DOWNDATE-READY.json``.  PARTIAL the moment the store fits."""
    cur = {}
    if READY.exists():
        with open(READY, encoding="utf-8") as fh:
            cur = json.load(fh)
    cur.update(_jsonable(fields))
    cur["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _jdump(cur, READY)
    _log(f"  DOWNDATE-READY.json -> status={cur.get('status')!r}")
    return cur


# ---------------------------------------------------------------------------
# 1. The fit
# ---------------------------------------------------------------------------


def _setup(condition: str = "val"):
    from . import exp_gen_factor_wells as fw
    fw.PINS["condition"] = str(condition)
    return fw


def _pool_and_wells(fw, out_dir: Path):
    z = dict(np.load(out_dir / "partitions.npz"))
    lab = np.asarray(z["factor"], np.int64)
    K = int(lab.max()) + 1
    wc = [np.flatnonzero(lab == k).astype(np.int64) for k in range(K)]
    dead = np.asarray(z["dead"]).ravel().astype(bool)
    return np.asarray(z["pool"], np.int64), wc, dead


def build_store(fw, out_dir: Path, rows: np.ndarray, wc, cards, log=_log
                ) -> DowndatableStore:
    """Fit the store on ``rows``.  ⭐ The ONE place a corpus is read at all."""
    D = int(fw.PINS["D"])
    A = int(W8_PINS["store_bytes"]) // (4 * D)
    if A != int(W8_PINS["n_directions"]):
        raise AssertionError(
            f"the byte pin implies A = {A} but the task pins {W8_PINS['n_directions']}")
    grid, _ = fw.build_grid()
    st = DowndatableStore(wc, D, A, cards)
    X = fw.gather(fw.features_mmap(int(fw.PINS["seed"])), rows)
    st.absorb(X, grid.values_of(rows), grid.values)
    st.refresh()
    del X
    led = st.byte_ledger(int(W8_PINS["store_bytes"]))
    if not led["matches_budget"]:
        raise AssertionError(
            f"the store deploys {led['deployed_bytes']} B against the pinned "
            f"{W8_PINS['store_bytes']} B -- the primary pin has tolerance 0")
    log(f"  store: {A} atoms x {D} f32 = {led['deployed_bytes']:,} B on {rows.size:,} "
        f"rows; ranks {led['rank_kept_per_well']} rank/d "
        f"{led['rank_over_d_per_well']}")
    return st


def stage_fit(out_dir: Path, condition: str, log=_log) -> dict:
    fw = _setup(condition)
    out_dir.mkdir(parents=True, exist_ok=True)
    pool, wc, dead = _pool_and_wells(fw, out_dir)
    grid, _ = fw.build_grid()
    cards = [int(c) for c in grid.cardinalities]
    log(f"stage fit: condition={condition} pool={pool.size:,} wells={[c.size for c in wc]}")

    lay = assert_layout_matches_field_store(cards)
    log(f"  coupling layout == gen_field's: {lay['n_pairs']} pairs, "
        f"{lay['n_indices_checked']} indices checked")

    t0 = time.time()
    st = build_store(fw, out_dir, pool, wc, cards, log=log)
    fit_s = time.time() - t0

    # the hole, so the object ships with part 4 exercised rather than merely present.
    # pair-000 = object_hue 0.0 AND shape 1.0 -- w5-delete's own conjunction.
    hue_f = int(grid.index_of("object_hue"))
    shp_f = int(grid.index_of("shape"))
    shp_v = int(np.argmin(np.abs(np.asarray(grid.values[shp_f]) - 1.0)))
    hole = st.punch_hole(hue_f, int(W8_PINS["delete_value_index"]), shp_f, shp_v)
    log(f"  hole punched at pair-000 (object_hue=0.0 AND shape=1.0): "
        f"{hole['permitted_bytes']} B, value unchanged={hole['coupling_value_unchanged']}")

    store_led = st.byte_ledger(int(W8_PINS["store_bytes"]))
    cov_led = st.covariance_ledger()
    honest = st.honest_ledger(int(W8_PINS["store_bytes"]))
    ranks = st.rank_report()
    log(f"  ⛔ LEDGER, printed separately and never netted:")
    log(f"     store       {store_led['deployed_bytes']:>12,} B  (pin "
        f"{W8_PINS['store_bytes']:,} B, matches={store_led['matches_budget']})")
    log(f"     covariance  {cov_led['covariance_bytes']:>12,} B  float64 d_k^2 per well "
        f"{cov_led['d_k_squared_per_well']}")
    log(f"     counts      {st.count_ledger()['total_count_bytes']:>12,} B")
    log(f"     coupling    {st.coupling_ledger()['coupling_bytes']:>12,} B "
        f"+ {st.coupling_ledger()['coupling_hole_bytes']:,} B of holes")
    log(f"     HONEST SUM  {honest['honest_total_bytes']:>12,} B = "
        f"{honest['times_over_budget']}x the pin  ⛔ OVER, and printed not netted")

    with open(out_dir / "store.pkl", "wb") as fh:
        pickle.dump(st, fh)
    obj = {
        "pins": W8_PINS, "condition": condition,
        "well_sizes": st.well_sizes,
        "dead_per_well": [int(dead[c].sum()) for c in wc],
        "effective_well_sizes": [int(c.size - dead[c].sum()) for c in wc],
        "n_pool_rows": int(pool.size),
        "store_ledger": store_led, "covariance_ledger": cov_led,
        "count_ledger": st.count_ledger(), "coupling_ledger": st.coupling_ledger(),
        "honest_ledger": honest, "rank_report": ranks,
        "hole": hole, "coupling_layout_check": lay,
        "content_sha256": st.content_sha(), "fit_seconds": fit_s,
        "feature_cache": fw.feature_cache_manifest(int(fw.PINS["seed"])),
    }
    _jdump(obj, out_dir / "store.json")
    # ⛔ A hub is gated on this file, so a re-run of `fit` must not blank the
    # assertions a later stage already published. Only the placeholder fields that are
    # genuinely absent are written, and `status` is never downgraded.
    prior = {}
    if READY.exists():
        with open(READY, encoding="utf-8") as fh:
            prior = json.load(fh)
    keep = {k: prior[k] for k in
            ("status", "downdate_equals_refit", "no_pass_over_data",
             "no_pass_over_data_detail", "items_kept_asserted", "items_kept_detail",
             "all_three_assertions_passed", "smoke", "smoke_pinned", "gauge",
             "post_delete") if k in prior}
    publish_ready(
        status=keep.get("status", "partial"),
        store_bytes=int(store_led["deployed_bytes"]),
        cov_bytes=int(cov_led["covariance_bytes"]),
        honest_total_bytes=int(honest["honest_total_bytes"]),
        times_over_budget=honest["times_over_budget"],
        well_sizes=st.well_sizes, dead_per_well=obj["dead_per_well"],
        effective_well_sizes=obj["effective_well_sizes"],
        rank_kept_per_well=store_led["rank_kept_per_well"],
        rank_over_d_per_well=store_led["rank_over_d_per_well"],
        any_well_saturated=store_led["any_well_saturated"],
        eig_rel_tol=float(EIG_REL_TOL),
        downdate_equals_refit=keep.get(
            "downdate_equals_refit",
            {"max_abs_dev_per_well": [], "passed": False,
             "note": "not yet run at this publish"}),
        no_pass_over_data=keep.get("no_pass_over_data", False), items_kept=False,
        items_kept_asserted=keep.get("items_kept_asserted", False),
        rank_report={k: v for k, v in ranks.items() if k != "per_well"},
        seeds=[0], condition=condition,
        content_sha256=st.content_sha(),
        note=("PARTIAL: the store fits at the pin and the ledgers are final. The three "
              "assertions and the read smoke follow."),
    )
    return obj


# ---------------------------------------------------------------------------
# 2. The three assertions
# ---------------------------------------------------------------------------


def stage_assertions(out_dir: Path, condition: str, log=_log) -> dict:
    fw = _setup(condition)
    pool, wc, dead = _pool_and_wells(fw, out_dir)
    grid, _ = fw.build_grid()
    cards = [int(c) for c in grid.cardinalities]
    vals_all = grid.values_of(pool)
    F = fw.features_mmap(int(fw.PINS["seed"]))
    res: dict = {}

    f, v = int(W8_PINS["delete_factor"]), int(W8_PINS["delete_value_index"])
    dval = float(grid.values[f][v])
    big = np.isclose(vals_all[:, f], dval)
    rng = np.random.default_rng(int(fw.PINS["seed"]))
    small = np.zeros(pool.size, bool)
    small[rng.permutation(pool.size)[: int(W8_PINS["small_delete_n"])]] = True
    log(f"stage assertions: deletions -- pinned value {grid.names[f]}={dval} "
        f"({int(big.sum()):,} rows) and a random {int(small.sum())} rows")

    # =====================================================================
    # ASSERTION 1: the downdated store EQUALS the store fitted WITHOUT the rows
    # =====================================================================
    a1 = {}
    for tag, mask in (("pinned_value_object_hue_0.0", big), ("random_64_rows", small)):
        t0 = time.time()
        st_all = build_store(fw, out_dir, pool, wc, cards, log=lambda *_: None)
        X_del = fw.gather(F, pool[mask])
        st_all.delete(X_del, vals_all[mask], grid.values)
        t_down = time.time() - t0
        t1 = time.time()
        st_ref = build_store(fw, out_dir, pool[~mask], wc, cards, log=lambda *_: None)
        t_ref = time.time() - t1
        cmp = compare_stores(st_all, st_ref)
        a1[tag] = {
            "n_deleted": int(mask.sum()), "n_remaining": int((~mask).sum()),
            "downdate_seconds": t_down, "refit_from_scratch_seconds": t_ref,
            **cmp,
        }
        log(f"  [A1/{tag}] max abs dev per well "
            f"{[f'{d:.3e}' for d in cmp['max_abs_dev_per_well']]}")
        log(f"           worst {cmp['max_abs_dev']:.3e} "
            f"({cmp['max_dev_in_float32_ulp']:.3e} float32 ulp @1.0), "
            f"{cmp['n_wells_bit_identical']}/{len(wc)} wells bit-identical, "
            f"ranks agree={cmp['ranks_agree_in_every_well']}, "
            f"counts equal={cmp['counts_equal']}, "
            f"term/pair counts equal={cmp['term_counts_equal']}/{cmp['pair_counts_equal']}")

        # ⭐⭐ THE NEGATIVE CONTROL.  A check that has never failed is not evidence:
        # nudge ONE atom of the refit by 1e-3 and require the comparison to FIRE.
        if tag == "pinned_value_object_hue_0.0":
            st_bad = build_store(fw, out_dir, pool[~mask], wc, cards, log=lambda *_: None)
            st_bad.modes[0][1, 0] = np.float32(float(st_bad.modes[0][1, 0]) + 1e-3)
            bad = compare_stores(st_all, st_bad)
            fired = bool(bad["max_abs_dev"] > 1e-4)
            a1["negative_control"] = {
                "what": ("one atom of the without-rows refit nudged by 1e-3; the "
                         "equality check MUST fire"),
                "max_abs_dev": bad["max_abs_dev"],
                "max_abs_dev_per_well": bad["max_abs_dev_per_well"],
                "check_fired": fired,
            }
            if not fired:
                raise AssertionError(
                    "the negative control did NOT fire -- the equality check cannot "
                    "distinguish a perturbed store from an equal one, so its passes "
                    "mean nothing")
            log(f"  [A1/negative control] nudged one atom by 1e-3 -> "
                f"max abs dev {bad['max_abs_dev']:.3e}, check FIRED={fired} ✓")
        del X_del, st_ref
    worst = max(x["max_abs_dev"] for k, x in a1.items() if k != "negative_control")
    worst_ulp = max(x["max_dev_in_float32_ulp"] for k, x in a1.items()
                    if k != "negative_control")
    # "to float precision" on a float32 atom block means: within a float32 ulp at the
    # block's own scale.  A store cannot be closer to another store than that and still
    # be a different store.
    a1["passed"] = bool(worst_ulp <= 1.0 and all(
        x["ranks_agree_in_every_well"] and x["counts_equal"] and x["term_counts_equal"]
        and x["pair_counts_equal"] for k, x in a1.items() if k != "negative_control"))
    a1["worst_max_abs_dev"] = float(worst)
    a1["worst_max_abs_dev_in_float32_ulp"] = float(worst_ulp)
    a1["criterion"] = ("no atom differs by as much as ONE float32 ulp at unit scale "
                       "(1.19e-7, the finest distinction a float32 store can draw "
                       "between two unit directions); ranks agree in every well; and "
                       "every count -- rows, terms, pairs -- is equal")
    res["downdate_equals_refit"] = a1

    # =====================================================================
    # ASSERTION 2: NO PASS OVER THE DATA -- mechanically, at two corpus sizes
    # =====================================================================
    a2 = {"per_corpus_size": []}
    half = pool[: int(len(pool) * float(W8_PINS["halfsize_frac"]))]
    for name, rows in (("full_pool", pool), ("half_pool", half)):
        vals = grid.values_of(rows)
        m = np.isclose(vals[:, f], dval)
        if int(m.sum()) < 8:
            continue
        m = np.flatnonzero(m)[: int(W8_PINS["small_delete_n"])]
        st = build_store(fw, out_dir, rows, wc, cards, log=lambda *_: None)
        X_del = fw.gather(F, rows[m])
        vd = vals[m]
        out = no_data_pass(st.delete, X_del, vd, grid.values, modules=(fw,))
        a2["per_corpus_size"].append({
            "corpus": name, "n_corpus_rows": int(rows.size),
            "n_deleted": int(m.size),
            "rows_consumed_by_delete": int(out["rows_consumed"]),
            "rows_consumed_per_well": int(out["rows_consumed"]) // len(wc),
            "expected_rows_per_well": int(m.size),
            "n_tripwires_installed": int(out["n_tripwires"]),
            "tripwires": out["tripwires_installed"],
        })
        log(f"  [A2/{name}] corpus {rows.size:,} rows; delete({m.size}) consumed "
            f"{out['rows_consumed']} row-reads = {m.size} rows x {len(wc)} wells, "
            f"with {out['n_tripwires']} tripwires armed on the corpus")
        del X_del
    sizes = a2["per_corpus_size"]
    same = len({r["rows_consumed_by_delete"] for r in sizes}) == 1
    right = all(r["rows_consumed_per_well"] == r["expected_rows_per_well"] for r in sizes)
    a2["invariant_to_corpus_size"] = bool(same)
    a2["consumed_only_the_deleted_rows"] = bool(right)
    a2["passed"] = bool(same and right and len(sizes) >= 2)
    a2["criterion"] = ("the delete consumes exactly the rows it was handed, once per "
                       "well, and the SAME number at two corpus sizes -- and every "
                       "named route to the corpus raises while it runs")
    a2["what_this_does_not_claim"] = (
        "⛔ delete() is HANDED the rows to forget; it does not look them up. The store "
        "never held them. What is asserted is that the RETAINED corpus is never opened.")
    res["no_pass_over_data"] = a2

    # =====================================================================
    # ASSERTION 3: ITEMS ARE NEVER KEPT
    # =====================================================================
    st_full = build_store(fw, out_dir, pool, wc, cards, log=lambda *_: None)
    st_half = build_store(fw, out_dir, half, wc, cards, log=lambda *_: None)
    au_full = retention_audit(st_full, int(pool.size))
    au_half = retention_audit(st_half, int(half.size))
    grew = int(au_full["total_object_bytes"]) - int(au_half["total_object_bytes"])
    a3 = {
        "audit_at_full_pool": au_full,
        "object_bytes_at_full_pool": int(au_full["total_object_bytes"]),
        "object_bytes_at_half_pool": int(au_half["total_object_bytes"]),
        "n_rows_full": int(pool.size), "n_rows_half": int(half.size),
        "bytes_grown_with_corpus": int(grew),
        "size_independent_of_corpus": bool(grew == 0),
        "n_arrays_in_object": int(au_full["n_arrays"]),
        "min_n_over_d_margin": int(au_full["min_n_over_d_margin"]),
        "passed": bool(grew == 0 and not au_full["items_kept"]),
        "criterion": ("every array's shape is explained by (A, d_k, K, T, NC), the "
                      "inventory is complete against vars(store), no axis is the "
                      "corpus size, and the object's byte count is IDENTICAL at two "
                      "corpus sizes -- an object that keeps items grows with the corpus"),
    }
    log(f"  [A3] {au_full['n_arrays']} arrays, every shape explained by "
        f"(A, d_k, K, T, NC); object bytes {au_full['total_object_bytes']:,} at "
        f"{pool.size:,} rows and {au_half['total_object_bytes']:,} at {half.size:,} "
        f"rows -- grew by {grew} B")
    res["items_never_kept"] = a3

    res["all_three_passed"] = bool(a1["passed"] and a2["passed"] and a3["passed"])
    _jdump(res, out_dir / "assertions.json")

    publish_ready(
        status="partial",
        downdate_equals_refit={
            "max_abs_dev_per_well": a1["pinned_value_object_hue_0.0"][
                "max_abs_dev_per_well"],
            "max_abs_dev_per_well_random64": a1["random_64_rows"][
                "max_abs_dev_per_well"],
            "worst_max_abs_dev": a1["worst_max_abs_dev"],
            "worst_in_float32_ulp": a1["worst_max_abs_dev_in_float32_ulp"],
            "n_wells_bit_identical": a1["pinned_value_object_hue_0.0"][
                "n_wells_bit_identical"],
            "negative_control_fired": a1["negative_control"]["check_fired"],
            "passed": a1["passed"],
        },
        no_pass_over_data=bool(a2["passed"]),
        no_pass_over_data_detail={
            "rows_consumed_by_delete": [r["rows_consumed_by_delete"] for r in sizes],
            "corpus_sizes": [r["n_corpus_rows"] for r in sizes],
            "invariant_to_corpus_size": a2["invariant_to_corpus_size"],
        },
        items_kept=False, items_kept_asserted=bool(a3["passed"]),
        items_kept_detail={
            "object_bytes_full": a3["object_bytes_at_full_pool"],
            "object_bytes_half": a3["object_bytes_at_half_pool"],
            "size_independent_of_corpus": a3["size_independent_of_corpus"]},
        all_three_assertions_passed=bool(res["all_three_passed"]),
        note=("PARTIAL: the three assertions are discharged. The read smoke through the "
              "shared frozen decoder and judge follows."),
    )
    return res


# ---------------------------------------------------------------------------
# 3. The smoke -- it fits, it reads, it renders, and the judge reads it back
# ---------------------------------------------------------------------------


def _fit_address(fw, out_dir: Path, st, basis, log=_log, tag: str = ""):
    """This store's OWN address book.  ⛔ Not the incumbent's: different atoms."""
    from ..core.gen_read import TensorAddress
    from . import exp_gen_read as R

    cache = out_dir / (f"address_downdate{tag}_prod_R{int(R.READ_PINS['rank'])}"
                       f"_lam{R.READ_PINS['ridge']}_sw{int(R.READ_PINS['sweeps'])}"
                       f"_n{int(R.READ_PINS['n_addr_train'])}.pkl")
    if cache.exists():
        with open(cache, "rb") as fh:
            log(f"  address book: reused {cache.name}")
            return pickle.load(fh)
    grid, _ = fw.build_grid()
    pool = np.asarray(np.load(out_dir / "partitions.npz")["pool"], np.int64)
    pool = pool[: int(R.READ_PINS["n_addr_train"])]
    Y, T = R._pool_span_targets(st, pool, basis, log=log)
    ntr = int(0.9 * Y.shape[0])
    addr = TensorAddress([int(c) for c in grid.cardinalities],
                         rank=int(R.READ_PINS["rank"]),
                         ridge=float(R.READ_PINS["ridge"]),
                         seed=int(R.READ_PINS["address_seed"]))
    addr.multiplicative = True
    addr.fit(Y[:ntr], [t[:ntr] for t in T], sweeps=int(R.READ_PINS["sweeps"]),
             log=log, Y_val=Y[ntr:], targets_val=[t[ntr:] for t in T])
    with open(cache, "wb") as fh:
        pickle.dump(addr, fh)
    return addr


def stage_smoke(out_dir: Path, condition: str, log=_log, basis_kind: str = "svd") -> dict:
    """⛔ ``val`` only, and it claims NOTHING.  It shows the object reads.

    ``basis_kind`` selects the read's basis and it is not a cosmetic choice -- see
    ``--stage gauge``. ``svd`` is ``gen_read.span_basis``, the programme's incumbent
    basis, whose SVD of a maximally-degenerate atom block makes the post-deletion read
    move by O(1e-3) for gauge reasons alone. ``pinned`` is
    ``downdate_store.pinned_span_basis``, which costs the same zero bytes and moves
    with the atoms. Both are run and both are reported.
    """
    from ..core.downdate_store import pinned_basis_report, pinned_span_basis
    from ..core.gen_read import assert_read_request_only, span_basis
    from . import exp_gen_read as R
    from . import exp_w5_compose_field as W5

    # ⛔ MEASURED THE HARD WAY: ``fw.arm_features`` is wave 3's own dispatcher and
    # knows nothing of the ``tensor`` verb -- it fell straight through to the ridge
    # head and died on ``obj["spec"]`` after the address book had already been fitted.
    # The span read lives in ``exp_gen_read``; that is the one to call.
    arm_features = R.arm_features

    basis_fn = pinned_span_basis if basis_kind == "pinned" else span_basis
    tag = "_pinned" if basis_kind == "pinned" else ""
    fw = _setup(condition)
    pool, wc, dead = _pool_and_wells(fw, out_dir)
    cond = fw.load_condition()
    grid = cond["grid"]
    with open(out_dir / "store.pkl", "rb") as fh:
        st = pickle.load(fh)
    basis = basis_fn(st)
    log(f"stage smoke [{basis_kind} basis]: span ranks "
        f"{[int(np.asarray(U).shape[1]) for U in basis]}")
    addr = _fit_address(fw, out_dir, st, basis, log=log, tag=tag)

    ro = assert_read_request_only(addr, st, grid.values_of(pool[:64]), grid.values,
                                  basis, rng_seed=int(fw.PINS["seed"]))
    log(f"  charter B.3: the read is reachable from the request alone -- {ro}")

    obj = {
        "arm": "downdate_store", "read": "tensor", "kind": "span",
        "store": st, "basis": basis, "address": addr,
        "byte_ledger": st.byte_ledger(int(W8_PINS["store_bytes"])),
        "address_bytes": addr.byte_ledger(),
        "head_bytes": {"head_bytes": 0, "note": "the address book IS the head here"},
        "partition": W8_PINS["partition"],
        "protos": None, "request_only": ro,
    }
    # the judge and the corners, exactly as fw.run_arm drives them
    judge = fw.Judge(int(fw.PINS["seed"]), log=log)
    bitmap = fw.value_presence_bitmap(
        grid, np.setdiff1d(cond["train_rows"], fw.val_corner_rows(cond)))
    corners = fw.val_corners(cond)
    log(f"  {len(corners)} val corners through the SHARED FROZEN decoder and judge")

    def _run(o, tag):
        rows = []
        for c in corners:
            vals = fw._request_values_from_rows(grid, c["rows"])
            feats, oor = arm_features(o, vals, grid)
            acc = fw.accept_rule(vals, oor, np.zeros(vals.shape[0], bool), bitmap, grid)
            tup = judge.read_back(feats, int(fw.PINS["eval_batch"]))
            valid = fw.judge_valid(vals, tup, grid) & acc
            rows.append({"corner_id": c["corner_id"], "n": int(vals.shape[0]),
                         "accept": float(acc.mean()), "valid": float(valid.mean())})
        v = np.asarray([r["valid"] for r in rows], np.float64)
        log(f"  {tag:<16} valid mean {v.mean():.4f}  sd {v.std(ddof=1):.4f}  "
            f"min {v.min():.4f}  max {v.max():.4f}  over {len(rows)} corners")
        return {"per_corner": rows, "valid_mean": float(v.mean()),
                "valid_sd": float(v.std(ddof=1)), "valid_min": float(v.min()),
                "valid_max": float(v.max()), "n_corners": len(rows)}

    mine = _run(obj, "downdate_store")

    # ⭐ the incumbent, on the SAME corners and the SAME judge, for scale only.
    inc = W5.build_pca_content(out_dir, log=log)
    inc_store = inc["store"]
    inc_basis = span_basis(inc_store)
    inc_cache = out_dir / (
        "address_pca_prod_R128_lam0.01_sw40_n131072.pkl")
    with open(inc_cache, "rb") as fh:
        inc_addr = pickle.load(fh)
    inc_obj = {"arm": "pca_tensor", "read": "tensor", "kind": "span",
               "store": inc_store, "basis": inc_basis, "address": inc_addr,
               "byte_ledger": inc["byte_ledger"], "protos": None,
               "head_bytes": {"head_bytes": 0}}
    theirs = _run(inc_obj, "pca_tensor")

    d = np.asarray([r["valid"] for r in mine["per_corner"]]) - np.asarray(
        [r["valid"] for r in theirs["per_corner"]])
    within = bool(abs(float(d.mean())) <= float(theirs["valid_sd"]))
    smoke = {
        "downdate_store": mine, "pca_tensor": theirs,
        "paired_delta_mean": float(d.mean()), "paired_delta_sd": float(d.std(ddof=1)),
        "within_pca_tensor_spread": within,
        "condition": condition,
        "what": ("⛔ val only, charter A.2's free selection condition. This is a SMOKE: "
                 "it shows the object fits, reads, renders through the shared frozen "
                 "decoder and is read back by the shared judge, beside the incumbent "
                 "on the SAME corners. ⛔ NOTHING here is a claim; hub/single-design "
                 "scores this object."),
    }
    _jdump(smoke, out_dir / f"smoke{tag}.json")
    log(f"  paired delta (downdate_store - pca_tensor) mean {d.mean():+.4f}; "
        f"within pca_tensor's own corner spread ({theirs['valid_sd']:.4f}) = {within}")

    # ⭐ and the thing the object exists for, shown on the read path: delete, then read.
    X_del = None
    vals_all = grid.values_of(pool)
    f, v = int(W8_PINS["delete_factor"]), int(W8_PINS["delete_value_index"])
    m = np.isclose(vals_all[:, f], float(grid.values[f][v]))
    X_del = fw.gather(fw.features_mmap(int(fw.PINS["seed"])), pool[m])
    sha_before = st.content_sha()
    rec = st.delete(X_del, vals_all[m], grid.values)
    del X_del
    obj2 = dict(obj)
    obj2["store"] = st
    obj2["basis"] = basis_fn(st)            # ⚠ the basis moves; see the note below
    after = _run(obj2, "after delete")
    post = {
        "delete_receipt": rec, "content_sha_before": sha_before,
        "content_sha_after": st.content_sha(),
        "read_after_delete": after,
        "address_book_refitted": False,
        "⚠_declared": ("the address book is NOT refitted after the delete -- charter "
                       "B.4's whole point is that a deletion is not a retrain. But the "
                       "downdate MOVES the span, so the fixed book's coefficients now "
                       "name slightly different directions. That is a real property of "
                       "this design, reported here rather than hidden, and it is a "
                       "cost the item store does not pay."),
    }
    post["basis_kind"] = basis_kind
    post["pinned_basis_report"] = (pinned_basis_report(st)
                                   if basis_kind == "pinned" else None)
    _jdump(post, out_dir / f"smoke_after_delete{tag}.json")
    log(f"  after delete({rec['n_rows_deleted']:,} rows): valid mean "
        f"{after['valid_mean']:.4f} (was {mine['valid_mean']:.4f}), store sha moved="
        f"{sha_before != st.content_sha()}")

    publish_ready(
        status="ready",
        **{f"smoke{tag}": {"downdate_store_val_valid_mean": mine["valid_mean"],
               "pca_tensor_val_valid_mean": theirs["valid_mean"],
               "paired_delta_mean": float(d.mean()),
               "within_pca_tensor_spread": within,
               "n_val_corners": mine["n_corners"],
               "valid_mean_after_delete": after["valid_mean"],
               "basis_kind": basis_kind,
               "renders_through_shared_frozen_decoder_and_judge": True}},
        note=("READY: the store fits at the pin, reads through the shared frozen "
              "decoder and judge, and the three assertions are discharged. ⛔ No "
              "scored run and no claim -- hub/single-design scores this object."),
    )
    return smoke


# ---------------------------------------------------------------------------
# 4. Arithmetic only: what the covariance would cost at other K
# ---------------------------------------------------------------------------


def stage_sweep(out_dir: Path, condition: str, log=_log) -> dict:
    """⚠ ARITHMETIC, NOT A BUILD.  ⛔ Changes nothing this spoke ships.

    ``hub/single-design`` asked for the K-sweep on the true partition arithmetic: the
    covariance falls **quadratically** as wells get finer while the atom block falls
    only linearly, so a fine-enough partition brings the honest total under the pin.
    ⛔ This spoke does not take that trade -- its task pins K=5 at tolerance 0 and
    forbids netting the statistic into the store's budget -- but the hub is entitled to
    the corrected numbers, so they are computed here and handed over.
    """
    fw = _setup(condition)
    pool, wc, dead = _pool_and_wells(fw, out_dir)
    D = int(fw.PINS["D"])
    budget = int(W8_PINS["store_bytes"])
    n_live = int(D - dead.sum())
    rows = []
    for K in (5, 8, 12, 16, 20, 24, 28, 32, 37, 40, 48, 52, 64):
        d = [n_live // K + (1 if i < n_live % K else 0) for i in range(K)]
        sq = sum(x * x for x in d)
        tri = sum(x * (x + 1) // 2 for x in d)
        # largest A whose atoms + float64 triangle + means + counts fit the pin
        A = 0
        while True:
            cost = (A + 1) * D * 4 + tri * 8 + sum(d) * 8 + K * 8
            if cost > budget:
                break
            A += 1
        rows.append({
            "K": K, "d_k_equal": d[0], "A_max_under_pin": A,
            "rank_over_d": round(A / float(d[0]), 4) if d[0] else None,
            "saturated": bool(A >= d[0]),
            "deployed_span": int(sum(min(A, x) for x in d)),
            "atoms_bytes": int(A * D * 4),
            "cov_triangle_f64_bytes": int(tri * 8),
            "cov_square_f64_bytes": int(sq * 8),
            "honest_total_bytes": int(A * D * 4 + tri * 8 + sum(d) * 8 + K * 8),
            "times_pin": round((A * D * 4 + tri * 8 + sum(d) * 8 + K * 8) / budget, 4),
        })
        log(f"  K={K:<3d} d={d[0]:<4d} A<={A:<3d} rank/d={rows[-1]['rank_over_d']:<7} "
            f"span={rows[-1]['deployed_span']:<5d} cov(tri,f64)="
            f"{rows[-1]['cov_triangle_f64_bytes']:>9,} total="
            f"{rows[-1]['honest_total_bytes']:>9,} ({rows[-1]['times_pin']}x)"
            + ("  ⛔ SATURATED" if rows[-1]["saturated"] else ""))
    out = {
        "what": ("arithmetic only, on the TRUE partition: live channels "
                 f"{n_live} of {D} ({int(dead.sum())} dead), wells as equal as the "
                 "grid allows, float64 statistic held as a symmetric triangle"),
        "condition": condition, "D": D, "n_live_channels": n_live,
        "n_dead_channels": int(dead.sum()),
        "actual_partition_well_sizes": [int(c.size) for c in wc],
        "actual_dead_per_well": [int(dead[c].sum()) for c in wc],
        "budget_bytes": budget, "rows": rows,
        "⛔_this_spoke_does_not_take_this_trade": (
            "task section 2 pins 64 directions and 524,288 B at tolerance 0, and "
            "section 3 forbids netting the covariance into the store's budget. "
            "Re-pinning K to bring the statistic under the pin pays for it by deleting "
            "content. Handed to hub/single-design as arithmetic, not built."),
    }
    _jdump(out, out_dir / "k_sweep.json")
    return out


# ---------------------------------------------------------------------------
# 5. Entry point
# ---------------------------------------------------------------------------


def stage_gauge(out_dir: Path, condition: str, log=_log) -> dict:
    """⛔⭐ Is the READ stable under an exact deletion?  Measured, on both bases.

    ``hub/single-design`` measured a max render deviation of 2.1-5.4 at three matched
    byte partitions between a downdated store and a refit-without store whose atoms
    agree to 1e-15, and its Procrustes fix made things worse.  This stage reproduces
    the measurement at the pinned partition and tests the fix that does work:
    :func:`downdate_store.pinned_span_basis`, which never asks a degenerate SVD where
    the basis should point.

    ⛔ No address book is fitted here and nothing is scored: two stores that are the
    same store are handed the SAME coefficients, so every difference is gauge.
    """
    from ..core.downdate_store import gauge_report, pinned_span_basis
    from ..core.gen_read import span_basis

    fw = _setup(condition)
    pool, wc, dead = _pool_and_wells(fw, out_dir)
    cond = fw.load_condition()
    grid = cond["grid"]
    cards = [int(c) for c in grid.cardinalities]
    vals_all = grid.values_of(pool)
    f, v = int(W8_PINS["delete_factor"]), int(W8_PINS["delete_value_index"])
    mask = np.isclose(vals_all[:, f], float(grid.values[f][v]))
    log(f"stage gauge: deleting {int(mask.sum()):,} rows two ways and comparing the READ")

    st_down = build_store(fw, out_dir, pool, wc, cards, log=lambda *_: None)
    X_del = fw.gather(fw.features_mmap(int(fw.PINS["seed"])), pool[mask])
    st_down.delete(X_del, vals_all[mask], grid.values)
    del X_del
    st_ref = build_store(fw, out_dir, pool[~mask], wc, cards, log=lambda *_: None)
    cmp = compare_stores(st_down, st_ref)
    log(f"  the two stores agree to {cmp['max_abs_dev']:.3e} -- they ARE the same store")

    # arbitrary but fixed coefficients: the val corners' own requests through the
    # incumbent-shaped address book would need a book, so unit coefficients are used
    # instead. ⛔ The point is that IDENTICAL coefficients go through both stores.
    rng = np.random.default_rng(int(fw.PINS["seed"]))
    n = 256
    coeffs = [rng.normal(size=(n, int(np.asarray(U).shape[1]))).astype(np.float32)
              for U in span_basis(st_ref)]
    out = {"stores_agree_to": cmp["max_abs_dev"],
           "n_rows_deleted": int(mask.sum()),
           "svd_basis": gauge_report(st_down, st_ref, coeffs, span_basis),
           "pinned_basis": gauge_report(
               st_down, st_ref,
               [rng.normal(size=(n, int(np.asarray(U).shape[1]))).astype(np.float32)
                for U in pinned_span_basis(st_ref)],
               pinned_span_basis)}
    for k in ("svd_basis", "pinned_basis"):
        r = out[k]
        if r.get("comparable"):
            log(f"  {k:<13} max |render dev| {r['max_abs_render_dev']:.4e}  "
                f"({r['relative_to_render_scale']:.3e} of the render scale)  "
                f"ranks {r['span_ranks']}")
        else:
            log(f"  {k:<13} NOT COMPARABLE: {r}")
    sv = out["svd_basis"].get("max_abs_render_dev", float("nan"))
    pn = out["pinned_basis"].get("max_abs_render_dev", float("nan"))
    out["gauge_is_a_real_problem_on_the_svd_basis"] = bool(sv > 1e-3)
    out["pinned_basis_fixes_it"] = bool(pn < 1e-5)
    out["improvement_factor"] = float(sv / pn) if pn > 0 else float("inf")
    out["what"] = (
        "⛔ Two stores that agree to 1e-10 are handed IDENTICAL coefficients. Under "
        "gen_read.span_basis the rendered vectors differ by O(1), because that basis "
        "is the SVD of an atom block whose spectrum is maximally degenerate (r unit "
        "rows), so a 1e-10 atom perturbation rotates it by O(1). Under "
        "pinned_span_basis -- the store's own already-orthonormal rows, no SVD -- they "
        "do not. ⭐ The deletion is exact either way; what the pinned basis buys is "
        "that the READ survives it.")
    _jdump(out, out_dir / "gauge.json")
    # ⛔⭐ Publish it, because DOWNDATE-READY already carries
    # `smoke.valid_mean_after_delete = 0.0` and a reader who sees that number without
    # this one beside it will read a GAUGE ARTEFACT as a composition cost. The two
    # belong in the same file.
    publish_ready(gauge={
        "stores_agree_to": out["stores_agree_to"],
        "svd_basis_max_render_dev": out["svd_basis"].get("max_abs_render_dev"),
        "pinned_basis_max_render_dev": out["pinned_basis"].get("max_abs_render_dev"),
        "improvement_factor": out["improvement_factor"],
        "pinned_basis_fixes_it": out["pinned_basis_fixes_it"],
        "READ_THIS_BEFORE_QUOTING_valid_mean_after_delete": (
            "⛔ `smoke.valid_mean_after_delete` is measured with the PRE-delete address "
            "book on the default (SVD) basis and is therefore dominated by a GAUGE "
            "artefact, not by the deletion: gen_read.span_basis re-derives the basis "
            "from an atom block whose singular spectrum is maximally degenerate, so a "
            "1e-10 atom change rotates it by O(1). ⛔ DO NOT quote it as a composition "
            "cost. `smoke_pinned.valid_mean_after_delete` is the same measurement with "
            "the gauge pinned and is the number to read."),
    })
    return out


def stage_postdelete(out_dir: Path, condition: str, log=_log) -> dict:
    """⭐⭐ THE QUESTION THE OBJECT EXISTS TO ANSWER: delete, refit NOTHING, still read?

    Reads the ``val`` corners three ways through the shared frozen decoder and judge:
    before the deletion, after it, and after it with the address book's slots put back
    into correspondence with the directions they were fitted against.

    ⛔ The third is not a refit and not a rewrite of the book: it is a **re-indexing**,
    because the store orders directions by eigenvalue and a deletion moves eigenvalues,
    so nearly-degenerate pairs swap slots (``hub/single-design`` measured 11-35% of
    slots per well). Its cost is ``sum_k r_k`` indices and it is declared, not free.
    ⛔ Requires the pinned-basis address book: run ``--stage smoke --basis pinned``.
    """
    from ..core.downdate_store import (direction_correspondence,
                                       pinned_span_basis, pinned_span_basis_matched)
    from . import exp_gen_read as R

    fw = _setup(condition)
    pool, wc, dead = _pool_and_wells(fw, out_dir)
    cond = fw.load_condition()
    grid = cond["grid"]
    with open(out_dir / "store.pkl", "rb") as fh:
        st = pickle.load(fh)
    basis_pre = pinned_span_basis(st)
    addr = _fit_address(fw, out_dir, st, basis_pre, log=log, tag="_pinned")
    modes_pre = [np.asarray(M, np.float32).copy() for M in st.modes]

    judge = fw.Judge(int(fw.PINS["seed"]), log=log)
    bitmap = fw.value_presence_bitmap(
        grid, np.setdiff1d(cond["train_rows"], fw.val_corner_rows(cond)))
    corners = fw.val_corners(cond)

    def _run(store, basis, tag):
        o = {"arm": "downdate_store", "read": "tensor", "kind": "span",
             "store": store, "basis": basis, "address": addr, "protos": None}
        vals_v = []
        for c in corners:
            vals = fw._request_values_from_rows(grid, c["rows"])
            feats, oor = R.arm_features(o, vals, grid)
            acc = fw.accept_rule(vals, oor, np.zeros(vals.shape[0], bool), bitmap, grid)
            tup = judge.read_back(feats, int(fw.PINS["eval_batch"]))
            vals_v.append(float((fw.judge_valid(vals, tup, grid) & acc).mean()))
        v = np.asarray(vals_v)
        log(f"  {tag:<34} valid mean {v.mean():.4f}  sd {v.std(ddof=1):.4f}")
        return {"valid_mean": float(v.mean()), "valid_sd": float(v.std(ddof=1)),
                "per_corner": [float(x) for x in v]}

    before = _run(st, basis_pre, "before delete")

    f, v = int(W8_PINS["delete_factor"]), int(W8_PINS["delete_value_index"])
    vals_all = grid.values_of(pool)
    mask = np.isclose(vals_all[:, f], float(grid.values[f][v]))
    X_del = fw.gather(fw.features_mmap(int(fw.PINS["seed"])), pool[mask])
    rec = st.delete(X_del, vals_all[mask], grid.values)
    del X_del
    log(f"  deleted {rec['n_rows_deleted']:,} rows; refit NOTHING")

    corr = direction_correspondence(modes_pre, st.modes)
    log(f"  direction reordering: {corr['total_slots_reordered']}/{corr['total_slots']} "
        f"slots moved; per well "
        f"{[w['n_rank_order_changed'] for w in corr['per_well']]} of "
        f"{[w['r'] for w in corr['per_well']]}, max shift "
        f"{[w['max_shift'] for w in corr['per_well']]}, min matched cos "
        f"{min(w['min_matched_cosine'] for w in corr['per_well']):.4f}")

    after_plain = _run(st, pinned_span_basis(st), "after delete, pinned basis")
    basis_matched, _ = pinned_span_basis_matched(st, modes_pre)
    after_matched = _run(st, basis_matched, "after delete, pinned + re-indexed")

    out = {
        "before_delete": before, "after_delete_pinned": after_plain,
        "after_delete_pinned_reindexed": after_matched,
        "delete_receipt": rec, "reordering": corr,
        "drop_pinned": float(before["valid_mean"] - after_plain["valid_mean"]),
        "drop_pinned_reindexed": float(
            before["valid_mean"] - after_matched["valid_mean"]),
        "reindexing_recovered": float(
            after_matched["valid_mean"] - after_plain["valid_mean"]),
        "address_book_bytes_changed": 0,
        "what": ("⛔ val only, no claim. The store forgot 11,024 rows exactly and NO "
                 "address book was refitted -- the re-indexed leg changes zero bytes "
                 "of the book, it only puts its slots back opposite the directions "
                 "they were fitted against. Any remaining drop is the deletion."),
    }
    _jdump(out, out_dir / "post_delete.json")
    publish_ready(post_delete={
        "before": before["valid_mean"],
        "after_pinned_basis": after_plain["valid_mean"],
        "after_pinned_basis_reindexed": after_matched["valid_mean"],
        "slots_reordered": corr["total_slots_reordered"],
        "of_slots": corr["total_slots"],
        "address_book_refitted": False, "address_book_bytes_changed": 0,
    })
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", required=True,
                   choices=("fit", "assertions", "smoke", "sweep", "gauge", "postdelete", "all"))
    p.add_argument("--out", default=str(OUT_ROOT / "build" / "val_seed0"))
    p.add_argument("--condition", default="val", choices=("val", "test"))
    p.add_argument("--basis", default="svd", choices=("svd", "pinned"),
                   help="the read's basis; see --stage gauge for why it matters")
    return p


def main(argv=None) -> int:
    a = build_arg_parser().parse_args(argv)
    out = Path(a.out)
    stages = (("fit", "assertions", "smoke") if a.stage == "all" else (a.stage,))
    for s in stages:
        _log(f"===== stage {s} =====")
        if s == "smoke":
            stage_smoke(out, a.condition, basis_kind=a.basis)
        else:
            {"fit": stage_fit, "assertions": stage_assertions,
             "sweep": stage_sweep, "gauge": stage_gauge,
             "postdelete": stage_postdelete}[s](out, a.condition)
    _log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
