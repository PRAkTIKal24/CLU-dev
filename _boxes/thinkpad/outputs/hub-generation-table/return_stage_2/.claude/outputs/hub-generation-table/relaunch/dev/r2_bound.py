"""EXP-R6 - the MATCHED-BYTES BOUND.  Is `knn_proto` beatable at 475,136 B at all?

Two reference rows, neither of them an arm, on the SAME 42 CONFIRM corners the arm was
measured on, so the comparison is paired:

  DENSE_ADD   52 dense D-dimensional prototypes, no interaction        425,984 B
              (= knn_proto's own model class, re-parameterised on the reduced design)
  DENSE_PAIR  the same + the shared-basis pair remap                   471,744 B

Both are storeless.  They are BOUNDS, not arms: no partition, no atoms, nothing to
delete atom-wise - exactly the "glorified lookup table" the programme's goal statement
rules out as a deliverable.  Their value is that they answer, at matched bytes, a
question the generation table has never had an answer to.

usage:  r2_bound.py <lo> <hi>
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arm2   # noqa: E402
import ctx    # noqa: E402
import dev    # noqa: E402
import r1     # noqa: E402
import specs  # noqa: E402

SEED = 0
OUT3 = os.environ["CHLU_OUT3"].replace("/c/", "C:/", 1)
TAG = os.environ.get("R1_TAG", "_b")
LEDGER = os.path.join(OUT3, "runs", f"r1_ledger{TAG}.jsonl")


def main():
    lo = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    hi = int(sys.argv[2]) if len(sys.argv) > 2 else 43
    T0 = time.time()
    c = ctx.build_context(SEED)
    D = int(np.prod(c["feat_shape"]))
    cards = c["cards"]
    from chlu.core.gen_dream import requests_from_labels

    P3 = specs.pair_index(specs.FACTORS, specs.PAIRS_BOUND)
    n_add = specs.n_add_of(cards, True)
    n_pair = specs.n_pair_of(cards, P3)
    VAR = {
        "DENSE_ADD":  dict(interaction=False),
        "DENSE_PAIR": dict(interaction=True),
    }
    ctx.log(f"n_add={n_add} n_pair={n_pair} D={D}  "
            f"DENSE_ADD={n_add*D*4} B  DENSE_PAIR={(n_add*D + n_pair*n_add)*4} B  "
            f"budget={arm2.KNN_BYTES} B")

    admitted = []
    for i, corner in enumerate(r1.CORNERS):
        try:
            dev.dev_setup(c, corner)
            admitted.append((i, corner))
        except (dev.HeldOutOverlap, AssertionError):
            pass
    pool_corners = admitted[6:][lo:hi]          # the CONFIRM set, same order as r1.py
    got = r1.ledger_keys()

    for (ci, corner) in pool_corners:
        ct = r1.corner_tag(corner)
        todo = [(n, v) for n, v in VAR.items() if (ct, n, "bound") not in got]
        if not todo:
            continue
        s = dev.dev_setup(c, corner)
        X_store = c["gather"](s["pool"])
        lab_store = c["factor_labels"](c["grid"], s["pool"])
        Zs = np.asarray(X_store, np.float32).reshape((-1, D))
        req = requests_from_labels(s["lab_dev"], "full", 0, SEED)
        for name, kw in todo:
            t = time.time()
            out = arm2.run_dense_reference(
                c, Zs, lab_store, req, s["lab_dev"], pairs=P3, n_add=n_add,
                n_pair=n_pair, ridge_add=10.0, ridge_pair=100.0, reduced=True, **kw)
            out["seconds"] = round(time.time() - t, 1)
            with open(LEDGER, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"corner_tag": ct, "corner_index": ci,
                                     "corner": corner, "variant": name,
                                     "spec_hash": "bound", "seed": SEED,
                                     "table": "bound", **out}, default=str) + "\n")
            ctx.log(f"[c{ci} {ct}] {name:12s} agg={out['aggregate']:.4f} "
                    f"tup={out['distinct_factor_tuples']:2d} B={out['total_bytes']} "
                    f"[{out['seconds']}s]")
    ctx.log(f"r2_bound done in {round(time.time()-T0,1)}s")


if __name__ == "__main__":
    main()
