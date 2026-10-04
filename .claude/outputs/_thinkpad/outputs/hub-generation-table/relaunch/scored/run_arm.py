"""LEG 2 of the ONE scored run - the hub's arm and its controls on eval_b.

Leg 1 is the PINNED `exp_gen_cell --stage cell --seed 0`, run UNEDITED, which produces the
rival rows (clu, store_removed, knn_proto, lace, ebm_compose) on this machine.
Leg 2 is this file.  Both legs read the SAME eval_b rows, through the SAME frozen decoder
and the SAME hash-asserted external classifier, in the same sitting, so they are ONE
scored run (PREREG s2).

  - No pinned harness file is edited.  Every primitive here is IMPORTED from the harness:
    resolve_decoder, load_external_classifier, select_store_rows, split_fit_val,
    load_feature_cache, gather_features, render, score_images, distinct_counts,
    factor_labels, requests_from_labels, FactorPrototypes, store_bytes, clu_wide_vector.
  - No feature tensor reaches any read: the arm's coefficients come from the REQUEST.
  - NOTHING IS ENCODED.  n_shortfall is asserted == 0 before a single row is gathered; if
    a row this run needs is not already in the Mac-written cache the script STOPS and
    reports the shortfall.  It never encodes and it never passes will_extend=False with a
    shortfall > 0 (LAUNCH-COMMON s4).

Rows produced (every one at a byte ledger measured from the LIVE objects):
  clu_argmax_shipped  the pinned clu read, recomputed here - a cross-check against leg 1
  clu_linear          THE SHIPPED STORE, THE SHIPPED BYTES, only the read verb changed
  clu_linear_removed  its C.2 control
  arm                 the factored dictionary read at the frozen CONFIRM configuration
  arm controls        atoms_medoid (C.1 sharp) / atoms_kpp (C.1 adversarial) /
                      atoms_random / pca_affine (R4 control only) / random / mean (C.2)
  knn_proto_recheck   the ceiling, recomputed here against leg 1's own row

usage:  run_arm.py <out_dir> <config.json>
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dev"))


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    out_dir = Path(sys.argv[1])
    cfg_path = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    SPEC = json.loads(cfg_path.read_text(encoding="utf-8"))
    SEED = int(SPEC.get("seed", 0))
    T0 = time.time()

    from chlu.config import get_default_config
    from chlu.core.gen_bridge import dataset_module, load_feature_cache, param_hash
    from chlu.core.gen_dict_fit import (allocate_atoms, build_store_from_dicts,
                                        fit_factored_head_v2, well_dictionary)
    from chlu.core.gen_dream import requests_from_labels
    from chlu.config import gen_row_split_config
    from chlu.core.gen_compose import (compose_codes, deployed_byte_ledger, read_compose,
                                       request_posterior, store_removed_read)
    from chlu.core.gen_dream import fit_address_book
    from chlu.core.gen_request import FactorPrototypes, requests_from_rows
    from chlu.core.gen_store import (address_codes, assert_no_bypass, delete_store,
                                     load_store_plan, store_bytes,
                                     well_channels_from_labels)
    from chlu.core.gen_shared_basis_head import fit_shared_basis_head
    from chlu.core.gen_wide_read import wide_vector
    from chlu.data.gen_rowsplit import split_fit_val
    from chlu.eval.gen_probe import factor_labels, predict_factors
    from chlu.experiments.exp_gen_cell import (distinct_counts, load_external_classifier,
                                               render, resolve_store, score_images,
                                               store_plan_path)
    from chlu.experiments.exp_gen_compose import resolve_decoder
    from chlu.experiments.exp_gen_dream import resolve_grouping
    from chlu.experiments.exp_gen_store import (bridge_dir, gather_features,
                                                matched_seen_rows, select_store_rows)
    from chlu.utils.checkpoints import load_model
    import fast  # the DEV channel-geometry + balanced partition, adopted 11:35Z

    cfg = get_default_config()
    cfg.gen_cell.claude_root = os.environ["CHLU_CLAUDE_ROOT"]
    cfg.gen_compose.claude_root = os.environ["CHLU_CLAUDE_ROOT"]
    for k in ("decoder_source", "winner_root", "winner_decoder_pattern",
              "expected_decoder_recipe"):
        setattr(cfg.gen_compose, k, getattr(cfg.gen_cell, k))
    ccfg = cfg.gen_cell
    dataset, pretext = ccfg.dataset, ccfg.pretext

    bdir = bridge_dir(cfg, dataset, pretext, SEED)
    decoder, dec_info, bman = resolve_decoder(cfg, bdir, SEED, log=log)
    external, clf_info = load_external_classifier(cfg, SEED, log=log)
    encoder = load_model(bdir / "encoder.pkl")
    assert param_hash(encoder) == bman["encoder_param_hash"], "encoder hash mismatch"

    mod = dataset_module(dataset)
    split, values = mod.build_split(cfg.gen_data.data_root)
    grid = split.grid
    names, cards = list(grid.names), [int(x) for x in grid.cardinalities]

    rows = select_store_rows(split, cfg, SEED)
    pool = np.asarray(rows["store_train"], np.int64)
    rs = gen_row_split_config(cfg)
    lab_pool = factor_labels(grid, pool)
    fit_rows, val_rows, split_info = split_fit_val(
        pool, lab_pool, n_fit=int(ccfg.n_fit), n_val=int(ccfg.n_val), seed=SEED,
        mode=rs.mode, tiebreak_jitter=float(rs.tiebreak_jitter),
        seed_mult=int(rs.seed_mult), seed_offset=int(rs.seed_offset), log=log)
    assert split_info.get("mode") == "stratified" and not split_info.get("BROKEN"), \
        "the REPAIRED stratified split is required; head_tail is the broken one"
    row_sets = {"B": np.asarray(rows["eval_b"], np.int64),
                "MS": matched_seen_rows(dataset, split, values, rows["store_train"]),
                "fit": np.asarray(fit_rows, np.int64),
                "val": np.asarray(val_rows, np.int64)}

    # ---- DRY mode: exercise every line of this file WITHOUT touching a held-out row. --
    # B and MS are replaced by two Montero-SHAPED corners cut from inside the train split.
    # This is a WIRING TEST: the store is the shipped checkpoint, fitted on a pool that
    # CONTAINS these corners, so a dry number is leaky and optimistic by construction.
    # It is never reported as a result; it exists so that leg 2 is debugged before the
    # one scored run, not during it.
    DRY = os.environ.get("RUN_ARM_DRY") == "1"
    if DRY:
        import ctx as _ctx
        held = np.concatenate([np.asarray(rows["eval_b"], np.int64),
                               np.asarray(rows["eval_a"], np.int64)])
        picked = []
        for corner in (dict(hue_lo=0.2, scale_hi=True, shape_value=0.0, orient_hi=True),
                       dict(hue_lo=0.4, scale_hi=True, shape_value=2.0, orient_hi=True)):
            r = _ctx.montero_like_corner(np.asarray(values), grid, **corner)
            assert r.size == 72 and np.intersect1d(r, held).size == 0, \
                "a DRY corner must realise 72 rows and touch NO held-out row"
            picked.append(r)
        row_sets["B"], row_sets["MS"] = picked[0], picked[1]
        log("*** DRY RUN: B and MS are TRAIN-SPLIT corners. Every number below is a "
            "wiring check and is NOT a result. eval_a and eval_b were not read. ***")
    log(f"rows: pool={pool.size} B={row_sets['B'].size} MS={row_sets['MS'].size} "
        f"fit={row_sets['fit'].size} val={row_sets['val'].size}")

    crows, feats_cache, cman = load_feature_cache(bdir, verify=False)
    feat_shape = tuple(int(x) for x in cman["feat_shape"])
    D = int(np.prod(feat_shape))
    assert cman["encoder_param_hash"] == bman["encoder_param_hash"]

    # ---- THE NO-EXTENSION GUARD.  LAUNCH-COMMON s4: a shortfall > 0 means STOP. ----
    need = np.unique(np.concatenate([pool] + [v for v in row_sets.values()]))
    n_shortfall = int((~np.isin(need, crows)).sum())
    log(f"cache coverage: {need.size - n_shortfall}/{need.size} rows cached, "
        f"shortfall={n_shortfall} (cache scope={cman['scope']}, n_rows={cman['n_rows']})")
    if n_shortfall > 0:
        raise SystemExit(
            f"STOP: {n_shortfall} of {need.size} rows this run needs are NOT in the "
            f"Mac-written cache. This run would EXTEND the cache on this box, which "
            f"LAUNCH-COMMON s4 forbids. Nothing encoded, nothing scored, nothing written.")

    def gather(r):
        return gather_features(dataset, cfg, encoder, crows, feats_cache, cman,
                               np.asarray(r, np.int64), log=None)

    X_store = gather(pool)
    X = {k: gather(v) for k, v in row_sets.items()}
    lab = {k: factor_labels(grid, v) for k, v in row_sets.items()}
    lab_store = factor_labels(grid, pool)
    EV = ("B", "MS")
    req = {k: requests_from_labels(lab[k], "full", 0, SEED) for k in EV}
    batch = int(ccfg.eval_batch)

    results, artefacts = {}, {}

    def record(name, feats, ledger, note):
        row = {"name": name, "byte_ledger": ledger, "note": note}
        for k in EV:
            f = np.asarray(feats[k], np.float32)
            imgs = render(decoder, f, feat_shape, batch)
            sc = score_images(external, imgs, lab[k], names, batch)
            di = distinct_counts(external, f, imgs, batch)
            preds = np.asarray(predict_factors(external, imgs, batch), np.int64)
            row[f"composition_{k}"] = {"aggregate": float(sc["aggregate"]),
                                       "per_factor": sc["per_factor"],
                                       "exact_match": float(sc["exact_match"])}
            row[f"distinct_{k}"] = {"factor_tuples": int(di["factor_tuples"]),
                                    "feature_vectors": int(di["feature_vectors"]),
                                    "n": int(di["n"])}
            # PER-RENDER TUPLES, persisted for EVERY arm (the addendum requires this so
            # composition-2 reads them instead of spending its own scored run).
            artefacts[f"{name}__{k}"] = {"preds": preds, "labels": lab[k],
                                         "rows": row_sets[k], "requests": req[k]}
        results[name] = row
        log(f"  {name:22s} B={row['composition_B']['aggregate']:.4f} "
            f"tuples={row['distinct_B']['factor_tuples']:2d}/72  "
            f"MS={row['composition_MS']['aggregate']:.4f}  bytes={ledger.get('total_bytes')}")
        _flush()

    def _flush():
        (out_dir / "arm_results.json").write_text(
            json.dumps({"seed": SEED, "spec": SPEC, "rows": results,
                        "provenance": prov, "wall_clock_s": round(time.time() - T0, 1)},
                       indent=2, default=str), encoding="utf-8")

    prov = {
        "machine": "FORGIS_14 Windows 11 AMD64 (R3: every number carries its machine)",
        "branch": "hub/generation-table-2", "base": "945a3e9 (base/iter2)",
        "decoder": dec_info, "classifier": clf_info,
        "encoder_param_hash": bman["encoder_param_hash"],
        "cache": {"n_rows": int(cman["n_rows"]), "scope": cman["scope"],
                  "n_shortfall": n_shortfall,
                  "note": "shortfall 0 => this run cannot extend the cache => the "
                          "re-encode gate does not bind (LAUNCH-COMMON s4)"},
        "row_split": split_info, "rows": {k: int(v.size) for k, v in row_sets.items()},
    }
    _flush()

    # ================================================================= #
    # A. THE SHIPPED STORE - only the READ VERB changes.  Byte-identical.
    # ================================================================= #
    plan = load_store_plan(store_plan_path(cfg, dataset, pretext))
    labels_vec, ginfo = resolve_grouping(plan, X_store, cfg, SEED,
                                         bman["encoder_param_hash"], log=log)
    K_ship = int(plan["K"])
    wc = well_channels_from_labels(labels_vec, K_ship)
    store, store_prov = resolve_store(cfg, dataset, pretext, SEED, X_store, wc, plan,
                                      feat_shape, log=log)
    codes_train = address_codes(store, np.asarray(X_store, np.float32).reshape(-1, D))
    book = fit_address_book(codes_train, lab_store, cards, float(ccfg.address_smoothing))
    store_led = deployed_byte_ledger(store, book)
    prov["shipped_store"] = {"K": K_ship, "well_sizes": [len(w) for w in wc],
                             "n_modes_per_well": list(store_bytes(store)["n_modes_per_well"]),
                             "deployed_byte_ledger": store_led,
                             "store_provenance": store_prov, "grouping": ginfo}
    prov["no_bypass"] = assert_no_bypass(store, X["B"][:64], rng_seed=SEED)
    log(f"shipped store: K={K_ship} wells={[len(w) for w in wc]} "
        f"deployed={store_led['deployed_bytes']} B")

    post = {k: request_posterior(book, req[k]) for k in EV}
    codes = {k: compose_codes(post[k]) for k in EV}
    head_b = int(store_led["addressing_head"]["bytes"])
    ship_ledger = {"total_bytes": int(store_led["deployed_bytes"]) + head_b,
                   "store_bytes": int(store_led["deployed_bytes"]),
                   "addressing_head_bytes": head_b,
                   "addressing_head": store_led["addressing_head"],
                   "source": "gen_compose.deployed_byte_ledger + address_book_bytes, "
                             "LIVE objects. The addressing head is INSIDE total_bytes "
                             "here so the argmax row and the linear row are compared in "
                             "every column, not only in the store column."}
    record("clu_argmax_shipped",
           {k: np.asarray(read_compose(store, codes[k]["code"]), np.float32) for k in EV},
           ship_ledger,
           "the PINNED clu read (argmax over the address-book posterior), recomputed here "
           "as a cross-check against leg 1's own clu row")
    record("clu_argmax_store_removed",
           {k: np.asarray(store_removed_read(store, codes[k]["code"],
                                             str(ccfg.store_removed_delete_mode), SEED),
                          np.float32) for k in EV},
           ship_ledger, "C.2 on the shipped argmax read")

    # the LINEAR read on the SAME store: an additive request head whose byte count is the
    # shipped addressing head to the byte (n_add * K*M * 4 = 58 * 320 * 4 = 74,240).
    Ms_ship = [int(np.asarray(m).shape[0]) for m in store.modes]
    dicts_ship = [np.asarray(m, np.float64) for m in store.modes]
    perm_ship = [np.asarray(w, np.int64) for w in wc]
    Zs = np.asarray(X_store, np.float32).reshape(-1, D)
    head_lin = fit_factored_head_v2(dicts_ship, perm_ship, Zs, lab_store, cards,
                                    pairs=(), rank=0, ridge_add=float(SPEC["ridge_add"]),
                                    reduced=False, metric_correct=False)
    lin_led = head_lin.byte_ledger()
    lin_total = int(store_bytes(store)["store_bytes"] + lin_led["bytes"])
    record("clu_linear",
           {k: np.asarray(wide_vector(store, head_lin.coefficients(req[k])), np.float32)
            for k in EV},
           {"total_bytes": lin_total,
            "store_bytes": int(store_bytes(store)["store_bytes"]),
            "head_bytes": int(lin_led["bytes"]), "head_ledger": lin_led,
            "shipped_addressing_head_bytes_reference":
                int(ccfg.addressing_head_bytes_reference),
            "source": "gen_store.store_bytes + RaggedHead.byte_ledger on the LIVE objects"},
           "THE SHIPPED STORE, THE SHIPPED BYTES: the only change from clu_argmax_shipped "
           "is read_from_codes(argmax) -> clu_wide_vector(real-valued request coefficients)")
    # its C.2: the same coefficients against a store whose content is gone
    st_del = delete_store(store, str(ccfg.store_removed_delete_mode), seed=SEED)
    record("clu_linear_store_removed",
           {k: np.asarray(wide_vector(st_del, head_lin.coefficients(req[k])), np.float32)
            for k in EV},
           {"total_bytes": lin_total}, "C.2 on the linear read, same coefficients")

    # ================================================================= #
    # B. THE ARM at the frozen CONFIRM configuration, and its controls
    # ================================================================= #
    K, A = int(SPEC["K"]), int(SPEC["A"])
    _, dead, ow, _ = fast.channel_geometry(X_store, cfg, f"scored_s{SEED}", log=log)
    labels_arm = fast.balanced_labels(ow, dead, K, D)
    perm = np.stack([np.flatnonzero(labels_arm == k) for k in range(K)])
    views = [np.asarray(Zs[:, perm[k]], np.float64) for k in range(K)]
    Ms = allocate_atoms(views, A, method=SPEC.get("alloc", "uniform"))
    pairs = [tuple(p) for p in SPEC["pairs"]]

    for method in SPEC["methods"]:
        dicts = [well_dictionary(views[k], int(Ms[k]), method, SEED * 1000 + k)
                 for k in range(K)]
        st = build_store_from_dicts([tuple(int(i) for i in perm[k]) for k in range(K)],
                                    dicts, (D,))
        if SPEC.get("head_kind", "shared") == "shared":
            head = fit_shared_basis_head(dicts, perm, Zs, lab_store, cards, pairs,
                                         ridge_add=float(SPEC["ridge_add"]),
                                         ridge_pair=float(SPEC["ridge_pair"]),
                                         reduced=bool(SPEC["reduced"]))
        else:
            head = fit_factored_head_v2(dicts, perm, Zs, lab_store, cards, pairs,
                                        int(SPEC["rank"]),
                                        ridge_add=float(SPEC["ridge_add"]),
                                        ridge_pair=float(SPEC["ridge_pair"]),
                                        reduced=bool(SPEC["reduced"]),
                                        metric_correct=bool(SPEC["metric_correct"]))
        sb, hb = store_bytes(st), head.byte_ledger()
        led = {"total_bytes": int(sb["store_bytes"] + hb["bytes"]),
               "store_bytes": int(sb["store_bytes"]), "head_bytes": int(hb["bytes"]),
               "store_ledger": sb, "head_ledger": hb,
               "knn_proto_budget": 475136,
               "multiple_of_budget": (sb["store_bytes"] + hb["bytes"]) / 475136.0,
               "source": "gen_store.store_bytes + RaggedHead.byte_ledger, LIVE objects"}
        if method == SPEC["methods"][0] and "expect_total_bytes" in SPEC:
            assert led["total_bytes"] == int(SPEC["expect_total_bytes"]), (
                f"the LIVE byte ledger is {led['total_bytes']} but the PREREG says "
                f"{SPEC['expect_total_bytes']} -- STOP, do not report a mispriced arm")
        name = "arm" if method == SPEC["methods"][0] else f"control_{method}"
        record(name,
               {k: np.asarray(wide_vector(st, head.coefficients(req[k])), np.float32)
                for k in EV}, led,
               {"kmeans": "THE ARM - learned k-means atoms",
                "nmf": "the arm with non-negative (span-objective) atoms",
                "atoms_medoid": "C.1 SHARP: the real stored row nearest each learned centre",
                "atoms_kpp": "C.1 ADVERSARIAL: M maximally spread real stored rows",
                "atoms_random": "C.1: M uniformly random real stored rows",
                "pca_affine": "R4 CONTROL ONLY - mean + M-1 principal directions; NOT an arm",
                "random": "the D-matched Gaussian null",
                "mean": "C.2 - every atom replaced by its well mean",
                }.get(method, method))
        if method == SPEC["methods"][0]:
            # C.4 - does the read sit on its nearest atom? (a lookup table if it does)
            f = np.asarray(wide_vector(st, head.coefficients(req["B"])), np.float32)
            c4 = {}
            for k in range(K):
                Bk = np.asarray(dicts[k], np.float32)
                V = f.reshape(-1, D)[:, perm[k]]
                d2 = ((V[:, None, :] - Bk[None]) ** 2).sum(-1)
                nearest = np.sqrt(d2.min(1))
                spread = float(np.sqrt(((Bk - Bk.mean(0)) ** 2).sum(1)).mean())
                c4[k] = (float(nearest.mean()), spread)
            nn = float(np.mean([v[0] for v in c4.values()]))
            sp = float(np.mean([v[1] for v in c4.values()]))
            prov["C4_distance_to_nearest_atom"] = {
                "mean_distance_to_nearest_atom": nn, "mean_atom_spread": sp,
                "ratio": nn / max(sp, 1e-12),
                "rule": "charter C.4: a settled state ON its nearest atom (ratio -> 0) "
                        "IS a lookup table and is reported as one"}
            log(f"  C.4 nearest-atom/spread = {nn:.4f}/{sp:.4f} = {nn/max(sp,1e-12):.3f}")
            _flush()

    # ================================================================= #
    # C. the ceiling, recomputed here against leg 1's own row
    # ================================================================= #
    protos = FactorPrototypes.fit(grid, X["fit"], row_sets["fit"])
    pl = protos.byte_ledger()
    record("knn_proto_recheck",
           {k: np.asarray(protos.embed_many(requests_from_rows(grid, row_sets[k])),
                          np.float32) for k in EV},
           {"total_bytes": int(pl["bytes"]), "ledger": pl},
           "the ceiling, fitted on row_sets['fit'] exactly as exp_gen_cell:1840 does")

    # the structural best-constant floor on THESE rows (arithmetic, not a measurement)
    prov["best_constant_floor"] = {
        k: float(np.mean([1.0 / np.unique(lab[k][:, j]).size for j in range(len(cards))]))
        for k in EV}
    _flush()
    np.savez_compressed(out_dir / "per_render_tuples.npz",
                        **{f"{n}__{f}": v for n, d in artefacts.items()
                           for f, v in d.items()})
    log(f"done in {round(time.time()-T0,1)}s -> {out_dir}")


if __name__ == "__main__":
    main()
