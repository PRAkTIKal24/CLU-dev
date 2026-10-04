"""RELAUNCH driver - resumable per (corner, variant) DEV evaluation.

DEV ONLY.  ``dev.dev_setup`` REFUSES (never trims) any corner touching eval_a/eval_b.

Resumability is per UNIT = (corner, variant).  A kill costs one ~30-second unit, never
the run (charter 5 / ``scripts/csf3/job_cpu_gen.sh``).  The ledger is a JSONL append.

THE SELECTION DISCIPLINE, declared BEFORE any sweep (its absence produced iteration 1's
10-corner "tie" that became a 49-corner loss):
    SELECT  = the first 6 admitted corners.  Every mechanism choice is made here.
    CONFIRM = every other admitted corner.  Never looked at until a config is frozen.

usage:  r1.py <table> <lo> <hi> [set]      set in {select, confirm, all}
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arm2   # noqa: E402
import ctx    # noqa: E402
import dev    # noqa: E402
import fast   # noqa: E402
import specs  # noqa: E402

OUT3 = os.environ["CHLU_OUT3"].replace("/c/", "C:/", 1)
# One ledger PER WORKER: two processes at OMP=2 out-throughput one at OMP=4 here (the
# Lloyd centroid update is a single-threaded Python loop), and separate files remove
# any interleaved-append hazard entirely.  ``an.py`` globs them all.
TAG = os.environ.get("R1_TAG", "")
LEDGER = os.path.join(OUT3, "runs", f"r1_ledger{TAG}.jsonl")
os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
SEED = 0


def corner_list():
    """Every Montero-SHAPED corner the grid admits, in one fixed deterministic order."""
    out, seen = [], set()
    for h in (0.2, 0.4, 0.6, 0.0):
        for sh in (0.0, 1.0, 2.0, 3.0):
            for sc in (True, False):
                for o in (True, False):
                    d = dict(hue_lo=h, scale_hi=sc, shape_value=sh, orient_hi=o)
                    k = tuple(sorted(d.items()))
                    if k not in seen:
                        seen.add(k)
                        out.append(d)
    return out


CORNERS = corner_list()


def corner_tag(d):
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:10]


def ledger_keys():
    """Every unit already on disk, across EVERY worker's ledger."""
    import glob
    got = set()
    for path in sorted(glob.glob(os.path.join(OUT3, "runs", "r1_ledger*.jsonl"))):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    got.add((r["corner_tag"], r["variant"], r["spec_hash"]))
    return got


def append(rec):
    with open(LEDGER, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")


def spec_hash(spec):
    payload = {k: v for k, v in spec.items() if not k.startswith("_")}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:12]


def main():
    table = sys.argv[1]
    lo = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    hi = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    which = sys.argv[4] if len(sys.argv) > 4 else "select"
    T0 = time.time()

    c = ctx.build_context(SEED)
    D = int(np.prod(c["feat_shape"]))
    cards = c["cards"]
    from chlu.core.gen_dream import requests_from_labels
    from chlu.core.gen_request import FactorPrototypes, requests_from_rows

    VARIANTS = specs.table(table, cards, D)
    ctx.log(f"table={table} nvariants={len(VARIANTS)} corners[{lo}:{hi}] set={which}")
    for n, s_ in VARIANTS.items():
        ctx.log(f"  {n:18s} K={s_['K']:3d} A={s_['A']:5d} rank={s_['rank']:4d} "
                f"n_add={s_['n_add']} n_pair={s_['n_pair']} method={s_['method']:12s} "
                f"alloc={s_['alloc']:9s} bytes_pred={s_.get('_bytes_pred')}")

    admitted, skipped = [], []
    for i, corner in enumerate(CORNERS):
        try:
            dev.dev_setup(c, corner)
            admitted.append((i, corner))
        except (dev.HeldOutOverlap, AssertionError) as exc:
            skipped.append((i, corner, str(exc)[:90]))
    ctx.log(f"admitted {len(admitted)} corners, REFUSED {len(skipped)} "
            f"(held-out overlap / not 72 rows) - refused, never trimmed")
    sel = admitted[:6]
    conf = admitted[6:]
    pool_corners = {"select": sel, "confirm": conf, "all": admitted}[which][lo:hi]

    got = ledger_keys()
    for (ci, corner) in pool_corners:
        ct = corner_tag(corner)
        todo = [(n, sp) for n, sp in VARIANTS.items()
                if (ct, n, spec_hash(sp)) not in got]
        need_knn = (ct, "knn_proto", "rival") not in got
        if not todo and not need_knn:
            ctx.log(f"[c{ci} {ct}] all {len(VARIANTS)} variants already in the ledger")
            continue
        t0 = time.time()
        s = dev.dev_setup(c, corner)
        X_store = c["gather"](s["pool"])
        lab_store = c["factor_labels"](c["grid"], s["pool"])
        lab_dev = s["lab_dev"]
        Zs = np.asarray(X_store, np.float32).reshape((-1, D))
        req = requests_from_labels(lab_dev, "full", 0, SEED)
        best_const = float(np.mean([1.0 / np.unique(lab_dev[:, j]).size
                                    for j in range(len(cards))]))
        if need_knn:
            protos = FactorPrototypes.fit(c["grid"], X_store, s["pool"])
            kn = dev.score_features(c, np.asarray(protos.embed_many(
                requests_from_rows(c["grid"], s["dev_rows"])), np.float32), lab_dev)
            kn["total_bytes"] = int(protos.byte_ledger()["bytes"])
            append({"corner_tag": ct, "corner_index": ci, "corner": corner,
                    "variant": "knn_proto", "spec_hash": "rival", "seed": SEED,
                    "best_constant": best_const, "table": "rival", **kn})
            got.add((ct, "knn_proto", "rival"))
            ctx.log(f"[c{ci} {ct}] knn_proto        agg={kn['aggregate']:.4f} "
                    f"tup={kn['distinct_factor_tuples']:2d} B={kn['total_bytes']}")
        geom = {}
        for name, sp in todo:
            K = int(sp["K"])
            part = str(sp.get("partition", "equal"))
            if (K, part) not in geom:
                corr, dead, ow, _ = fast.channel_geometry(
                    X_store, c["cfg"], f"s{SEED}_{ct}", log=None)
                if part == "equal":
                    # EQUAL contiguous blocks of the ward dendrogram leaf order
                    labels = fast.balanced_labels(ow, dead, K, D)
                    perm = np.stack([np.flatnonzero(labels == k) for k in range(K)])
                elif part == "ward":
                    # the SHIPPED grouping: gen_store.correlation_grouping, UNEQUAL wells
                    from chlu.core.gen_store import correlation_grouping
                    labels, _ = correlation_grouping(
                        corr, K, "auto", dead=dead,
                        min_well_frac=float(c["cfg"].gen_store.min_well_frac),
                        return_info=True)
                    perm = [np.flatnonzero(np.asarray(labels) == k) for k in range(K)]
                    if min(len(x) for x in perm) == 0:
                        raise ValueError(f"ward grouping gave an EMPTY well at K={K}")
                else:
                    raise ValueError(f"unknown partition {part!r}")
                geom[(K, part)] = (perm, arm2.well_views(Zs, perm))
            perm, views = geom[(K, part)]
            tv = time.time()
            run = {k: v for k, v in sp.items()
                   if not k.startswith("_") and k != "partition"}
            try:
                out = arm2.run_variant(c, s, Zs, lab_store, perm, req, lab_dev,
                                       views=views, seed=SEED, **run)
            except Exception as exc:
                ctx.log(f"[c{ci}] {name} FAILED: {type(exc).__name__}: {exc}")
                append({"corner_tag": ct, "corner_index": ci, "corner": corner,
                        "variant": name, "spec_hash": spec_hash(sp), "seed": SEED,
                        "table": table, "spec": run,
                        "error": f"{type(exc).__name__}: {exc}"})
                continue
            out["seconds"] = round(time.time() - tv, 1)
            append({"corner_tag": ct, "corner_index": ci, "corner": corner,
                    "variant": name, "spec_hash": spec_hash(sp), "seed": SEED,
                    "best_constant": best_const, "table": table, "spec": run, **out})
            ctx.log(f"[c{ci} {ct}] {name:18s} agg={out['aggregate']:.4f} "
                    f"tup={out['distinct_factor_tuples']:2d} B={out['total_bytes']} "
                    f"r={out['rank_realised']} [{out['seconds']}s]")
        ctx.log(f"[c{ci}] corner done in {round(time.time()-t0,1)}s")
    ctx.log(f"r1 {table} done in {round(time.time()-T0,1)}s -> {LEDGER}")


if __name__ == "__main__":
    main()
