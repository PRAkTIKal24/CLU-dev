"""EXP-R9 - the READ VERB on the SHIPPED store, powered.

The hub's clearest recommendation is measured at n = 1 on eval_b (+0.2014) and at n = 6 on
iteration 1's DEV corners (+0.1192).  This puts a PAIRED estimate on 43 held-out DEV
corners, which is the instrument that can resolve it.

  argmax : read_from_codes(store, compose_codes(request_posterior(book, req)))
  linear : clu_wide_vector(store, request-addressed REAL coefficients)
  both at 598,528 B = store 524,288 + addressing head 74,240 -- byte-identical.

DECLARED LIMITATION, and it is the reason this is DEV and not a scored row: the store and
the address book are the SHIPPED seed-0 objects, fitted on a pool that CONTAINS these DEV
corners.  Both rows are inflated by that leak.  They are inflated by the SAME leak, on the
SAME rows, through the SAME store -- so the PAIRED DIFFERENCE is the quantity this file
reports, and the absolute levels are not comparable with eval_b's.

usage:  r3_readverb.py <lo> <hi>
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ctx    # noqa: E402
import dev    # noqa: E402
import r1     # noqa: E402

SEED = 0
OUT3 = os.environ["CHLU_OUT3"].replace("/c/", "C:/", 1)
TAG = os.environ.get("R1_TAG", "_v")
LEDGER = os.path.join(OUT3, "runs", f"r1_ledger{TAG}.jsonl")


def main():
    lo = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    hi = int(sys.argv[2]) if len(sys.argv) > 2 else 43
    T0 = time.time()
    c = ctx.build_context(SEED)
    D = int(np.prod(c["feat_shape"]))
    cards = c["cards"]
    cfg = c["cfg"]

    from chlu.core.gen_bridge import param_hash  # noqa: F401
    from chlu.core.gen_compose import (compose_codes, deployed_byte_ledger, read_compose,
                                       request_posterior, store_removed_read)
    from chlu.core.gen_dict_fit import fit_factored_head_v2
    from chlu.core.gen_dream import fit_address_book, requests_from_labels
    from chlu.core.gen_store import (address_codes, load_store_plan, store_bytes,
                                     well_channels_from_labels)
    from chlu.core.gen_wide_read import wide_vector
    from chlu.experiments.exp_gen_cell import resolve_store, store_plan_path
    from chlu.experiments.exp_gen_dream import resolve_grouping

    # the SHIPPED objects, resolved ONCE: the store is loaded, never re-trained.
    X_pool = c["gather"](c["pool"])
    lab_pool = c["factor_labels"](c["grid"], c["pool"])
    plan = load_store_plan(store_plan_path(cfg, c["dataset"], c["pretext"]))
    labels_vec, _ = resolve_grouping(plan, X_pool, cfg, SEED,
                                     c["bman"]["encoder_param_hash"], log=ctx.log)
    K = int(plan["K"])
    wc = well_channels_from_labels(labels_vec, K)
    store, prov = resolve_store(cfg, c["dataset"], c["pretext"], SEED, X_pool, wc, plan,
                                c["feat_shape"], log=ctx.log)
    Zp = np.asarray(X_pool, np.float32).reshape(-1, D)
    book = fit_address_book(address_codes(store, Zp), lab_pool, cards,
                            float(cfg.gen_cell.address_smoothing))
    led = deployed_byte_ledger(store, book)
    TOTAL = int(led["deployed_bytes"] + led["addressing_head"]["bytes"])
    ctx.log(f"SHIPPED store K={K} wells={[len(w) for w in wc]} "
            f"store={led['deployed_bytes']} + head={led['addressing_head']['bytes']} "
            f"= {TOTAL} B  (loaded, NOT re-trained)")
    dicts = [np.asarray(m, np.float64) for m in store.modes]
    perm = [np.asarray(w, np.int64) for w in wc]
    head = fit_factored_head_v2(dicts, perm, Zp, lab_pool, cards, pairs=(), rank=0,
                                ridge_add=10.0, reduced=False, metric_correct=False)
    hb = head.byte_ledger()["bytes"]
    assert hb == int(cfg.gen_cell.addressing_head_bytes_reference), (
        f"the linear head is {hb} B but the SHIPPED addressing head is "
        f"{cfg.gen_cell.addressing_head_bytes_reference} B - the comparison is only "
        f"honest if they are byte-identical")
    ctx.log(f"linear head {hb} B == the shipped addressing head, to the byte")

    admitted = []
    for i, corner in enumerate(r1.CORNERS):
        try:
            dev.dev_setup(c, corner)
            admitted.append((i, corner))
        except (dev.HeldOutOverlap, AssertionError):
            pass
    got = r1.ledger_keys()
    for (ci, corner) in admitted[6:][lo:hi]:
        ct = r1.corner_tag(corner)
        if (ct, "VERB_linear", "verb") in got:
            continue
        s = dev.dev_setup(c, corner)
        req = requests_from_labels(s["lab_dev"], "full", 0, SEED)
        post = request_posterior(book, req)
        codes = compose_codes(post)
        rows = {
            "VERB_argmax": np.asarray(read_compose(store, codes["code"]), np.float32),
            "VERB_linear": np.asarray(wide_vector(store, head.coefficients(req)),
                                      np.float32),
            "VERB_argmax_removed": np.asarray(
                store_removed_read(store, codes["code"], "mean", SEED), np.float32),
        }
        scored = {}
        for name, feats in rows.items():
            t = time.time()
            out = dev.score_features(c, feats, s["lab_dev"])
            scored[name] = out
            out["total_bytes"] = TOTAL
            out["store_bytes"] = int(store_bytes(store)["store_bytes"])
            out["seconds"] = round(time.time() - t, 1)
            out["leak_declared"] = ("the SHIPPED store/book were fitted on a pool that "
                                    "CONTAINS this corner; both rows share the leak, so "
                                    "only the PAIRED difference is reported")
            with open(LEDGER, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"corner_tag": ct, "corner_index": ci,
                                     "corner": corner, "variant": name,
                                     "spec_hash": "verb", "seed": SEED,
                                     "table": "verb", **out}, default=str) + "\n")
        ctx.log(f"[c{ci} {ct}] " + "  ".join(
            f"{n.replace('VERB_',''):14s}={v['aggregate']:.4f}"
            f"({v['distinct_factor_tuples']:2d})" for n, v in scored.items()))
    ctx.log(f"r3_readverb done in {round(time.time()-T0,1)}s")


if __name__ == "__main__":
    main()
