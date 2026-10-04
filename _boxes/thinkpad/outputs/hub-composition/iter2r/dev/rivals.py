"""hub/composition-2 (relaunch) — THE RIVALS AND THE §C CONTROLS, on MY corners.

⛔ SELECTION ONLY. `eval_a` / `eval_b` never enter (the guard is `dev.dev_setup`).

Why this module exists
----------------------
R2's bar is a PAIRED comparison of distinct-valid-novel tuples against `knn_proto`.
R3 forbids putting a Mac-published `knn_proto` row inside a paired comparison with a
Windows-produced arm row.  ⇒ the rival has to be RE-RUN on this box, on the SAME
corners, through the SAME instrument, or there is no bar to clear.  The stopped
relaunch measured the object but never measured a rival beside it.

Every arm below is scored by ONE function (`comp.headline`), so no arm is counted by a
different rule than another.  Every arm's bytes are declared.

The arms
--------
`clu_linear`        THE OBJECT.  K=32 correlation-partitioned wells, M=26 k-means modes,
                    request-addressed LINEAR read (`clu_wide_vector`), rank-16 pair head.
                    ⭐ Re-fitted through the PINNED `fit_factored_head`.
`knn_proto`         THE RIVAL.  `FactorPrototypes` (the pinned class), fitted on the SAME
                    pool the store is fitted on, `embed_many` on the same requests.
`store_removed`     §C.2 laundering control — identical read path, atoms ablated.
                    Two flavours: zeroed, and shuffled across wells (bytes unchanged).
`head_removed`      the other half of §C.2 — the atoms kept, the HEAD ablated to its
                    additive-only block (no pair interactions). Shows which half carries.
`lookup_feat`       §C.1 matched-bytes lookup: 57 real pool rows chosen as feature
                    k-medoids, addressed by nearest requested tuple.
`lookup_tuple`      §C.1 again, the STRONGER form: the 57 medoids chosen to cover the
                    FACTOR space, so the address book cannot be blamed.
                    ⭐ The §C.1 floor is the MAX of the two.
`const_mu`          the constant-output floor (one tuple, by construction).

§C.4 is reported per corner for `clu_linear`: distance from each render's feature to the
nearest row the store was fitted on, over the pool's own spread.  ⭐ Plus the fact that
makes it sharp here: the corner is REMOVED from the pool, so ⛔ no stored row carries a
target tuple — a lookup cannot be valid, whatever its distance.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

import comp
import ctx
import dev
import fast

OUT = Path(os.environ["CHLU_OUT"].replace("/c/", "C:/", 1))
LEDGER = OUT / "dev" / "rivals_ledger.jsonl"

# The reference object, from the addendum. ⛔ Not re-chosen here.
REF = dict(K=32, M=26, rank=16, ridge=10.0, pairs=["floor_huexscale", "object_huexscale"])


# ---------------------------------------------------------------------------
# resumable ledger
# ---------------------------------------------------------------------------
def ledger_rows() -> list[dict]:
    if not LEDGER.exists():
        return []
    out = []
    with open(LEDGER, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def ledger_has(tag: str, ci: int, arm: str) -> bool:
    return any(r.get("tag") == tag and r.get("corner_idx") == ci and r.get("arm") == arm
               for r in ledger_rows())


def ledger_put(rec: dict):
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")


# ---------------------------------------------------------------------------
# the matched-bytes lookup (§C.1)
# ---------------------------------------------------------------------------
def n_lookup_slots(total_bytes: int, D: int) -> int:
    """How many raw float32 feature vectors fit in the arm's WHOLE byte ledger."""
    return int(total_bytes // (int(D) * 4))


def _kmedoid_rows(F: np.ndarray, n: int, seed: int, iters: int = 25) -> np.ndarray:
    """Indices (into F's rows) of `n` medoids: k-means, then the nearest real row."""
    from chlu.core.gen_store import _kmeanspp_init
    X = np.asarray(F, np.float32)
    rng = np.random.default_rng(int(seed) * 7_919 + 13)
    C = np.asarray(_kmeanspp_init(X, int(n), rng), np.float32)
    for _ in range(int(iters)):
        a = _assign(X, C)
        Cn = np.stack([X[a == j].mean(0) if np.any(a == j) else C[j] for j in range(int(n))])
        if float(np.abs(Cn - C).mean()) < 1e-7:
            C = Cn
            break
        C = Cn.astype(np.float32)
    a = _assign(X, C)
    med = np.empty(int(n), np.int64)
    for j in range(int(n)):
        m = np.flatnonzero(a == j)
        if m.size == 0:                       # empty cluster -> nearest row overall
            m = np.arange(X.shape[0])
        d = ((X[m] - C[j][None]) ** 2).sum(1)
        med[j] = int(m[int(np.argmin(d))])
    return med


def _assign(X, C, batch: int = 8192) -> np.ndarray:
    cc = (np.asarray(C, np.float32) ** 2).sum(1)
    out = np.empty(X.shape[0], np.int64)
    for i in range(0, X.shape[0], batch):
        Xb = X[i:i + batch]
        out[i:i + batch] = np.argmin(cc[None] - 2.0 * (Xb @ np.asarray(C, np.float32).T), 1)
    return out


def _tuple_space(grid, lab: np.ndarray) -> np.ndarray:
    """Factor labels -> unit-scaled factor-VALUE coordinates, so a distance is meaningful."""
    Y = np.asarray(lab, np.int64)
    cols = []
    for k in range(Y.shape[1]):
        v = np.asarray(grid.values[k], np.float64)
        span = max(float(v[-1] - v[0]), 1e-12)
        cols.append((v[Y[:, k]] - v[0]) / span)
    return np.stack(cols, 1).astype(np.float32)


def lookup_read(Zpool, lab_pool, lab_req, med_idx, grid) -> np.ndarray:
    """Address the stored rows by the REQUEST only: nearest tuple, Hamming then value.

    ⭐ Request-only by construction: `lab_req` is the request, and no feature of the
    requested row is touched. ⛔ A tie is broken deterministically (lowest index).
    """
    Tm = np.asarray(lab_pool, np.int64)[np.asarray(med_idx, np.int64)]     # (S, F)
    Vm = _tuple_space(grid, Tm)                                           # (S, F)
    Tr = np.asarray(lab_req, np.int64)
    Vr = _tuple_space(grid, Tr)
    ham = (Tr[:, None, :] != Tm[None, :, :]).sum(-1).astype(np.float64)   # (N, S)
    l1 = np.abs(Vr[:, None, :] - Vm[None, :, :]).sum(-1).astype(np.float64)
    sel = np.argmin(ham + 1e-3 * l1, axis=1)
    return np.asarray(Zpool, np.float32)[np.asarray(med_idx, np.int64)[sel]]


# ---------------------------------------------------------------------------
# the per-corner driver
# ---------------------------------------------------------------------------
def run_corner(c, ci: int, *, tag: str, seed: int, cfgd=None, arms=None,
               log=ctx.log) -> dict:
    """Every arm on ONE corner. Resumable at (tag, corner, arm) granularity."""
    from chlu.core.gen_compose import clu_wide_vector
    from chlu.core.gen_dream import requests_from_labels
    from chlu.core.gen_factored_read import (assert_request_only_factored,
                                             default_pairs, fit_factored_head)
    from chlu.core.gen_request import FactorPrototypes, requests_from_rows
    from chlu.core.gen_store import assert_no_bypass, store_bytes

    cfgd = dict(REF if cfgd is None else cfgd)
    K, M, RANK, RIDGE = int(cfgd["K"]), int(cfgd["M"]), int(cfgd["rank"]), float(cfgd["ridge"])
    arms = list(arms) if arms else ["clu_linear", "knn_proto", "store_removed_zero",
                                    "store_removed_shuffle", "head_additive_only",
                                    "lookup_feat", "lookup_tuple", "const_mu",
                                    "ceiling"]
    todo = [a for a in arms if not ledger_has(tag, ci, a)]
    if not todo:
        log(f"[c{ci}] all {len(arms)} arms already in the ledger — skipped")
        return {}

    corner = comp.CORNERS[ci]
    D = int(np.prod(c["feat_shape"]))
    cards, grid = c["cards"], c["grid"]
    t0 = time.time()
    s = dev.dev_setup(c, corner, log=log)
    X_store = c["gather"](s["pool"])
    Zs = np.asarray(X_store, np.float32).reshape((-1, D))
    lab_store = c["factor_labels"](grid, s["pool"])
    lab_dev = s["lab_dev"]
    pool_tup = comp.pool_tuple_array(grid, s["pool"], c["factor_labels"])
    log(f"[c{ci}] pool {s['pool'].size} rows / {pool_tup.shape[0]} distinct tuples; "
        f"todo: {todo}")

    PAIRS = list(default_pairs(c["names"], list(cfgd["pairs"])))
    tot_b = comp.total_bytes(cards, PAIRS, K, M, RANK, D)
    knn_b = comp.knn_proto_bytes(cards, D)
    base = {"tag": tag, "corner_idx": ci, "corner": corner, "seed": int(seed),
            "config": cfgd, "n_pool": int(s["pool"].size),
            "n_pool_tuples": int(pool_tup.shape[0])}

    def emit(arm, feats, bytes_, note, extra=None):
        h = comp.headline(c, np.asarray(feats, np.float32), lab_dev, pool_tup)
        rec = {**base, "arm": arm, "bytes": int(bytes_),
               "bytes_vs_knn_proto": float(bytes_) / float(knn_b),
               "note": note, "headline": h}
        if extra:
            rec["extra"] = extra
        ledger_put(rec)
        log(f"[c{ci}] {arm:22s} tuples_valid_novel={h['tuples_valid_novel']:3d}  "
            f"agg={h['aggregate']:.4f}  distinct={h['distinct']:3d}  "
            f"exact={h['exact_match']:.4f}  bytes={int(bytes_):,}")
        return h

    # ---- the store + the head, built once and shared by every store-based arm ----
    need_store = any(a in todo for a in ("clu_linear", "store_removed_zero",
                                         "store_removed_shuffle", "head_additive_only"))
    st = head = a_ref = None
    if need_store:
        corr, dead, ow, oa = fast.channel_geometry(X_store, c["cfg"],
                                                   f"s{seed}_c{ci}_n{s['pool'].size}", log=None)
        labels = fast.balanced_labels(ow, dead, K, D)
        perm = np.stack([np.flatnonzero(labels == k) for k in range(K)])
        pq = fast.kmeans_pq(Zs, labels, K, M, seed, iters=25)
        st = fast.EqualPQ(perm, pq.modes.astype(np.float32), (D,)).to_gen_store()
        req = requests_from_labels(lab_dev, "full", 0, seed)
        head = fit_factored_head(st, Zs, lab_store, cards, PAIRS, RANK, RIDGE)
        a_ref = head.coefficients(req)

    if "clu_linear" in todo:
        feats = np.asarray(clu_wide_vector(st, a_ref), np.float32)
        ro = assert_request_only_factored(st, head, req, decoy_features=Zs[:64])
        nb = assert_no_bypass(st, Zs[:64], rng_seed=seed)
        # §C.4 — is the output a stored row?
        Zf = feats.reshape((feats.shape[0], -1))
        sub = Zs[:: max(1, Zs.shape[0] // 20000)]
        d_near = np.sqrt(np.maximum(
            ((Zf ** 2).sum(1)[:, None] - 2.0 * (Zf @ sub.T) + (sub ** 2).sum(1)[None]), 0.0
        )).min(1)
        mu = sub.mean(0)
        spread = float(np.sqrt(((sub - mu[None]) ** 2).sum(1).mean()))
        emit("clu_linear", feats,
             comp.store_bytes_of(M, D) + comp.head_bytes(cards, PAIRS, K, M, RANK),
             "THE OBJECT — request-addressed LINEAR read, pinned head",
             {"store_bytes": int(store_bytes(st)["store_bytes"]),
              "head_bytes": int(head.byte_ledger()["bytes"]),
              "assert_request_only_factored": {k: v for k, v in ro.items()
                                               if k != "signatures"},
              "assert_no_bypass": {k: v for k, v in nb.items() if k != "note"},
              "C4_distance_to_nearest_stored_row": {
                  "mean": float(d_near.mean()), "min": float(d_near.min()),
                  "max": float(d_near.max()),
                  "pool_rms_spread_about_its_mean": spread,
                  "mean_over_spread": float(d_near.mean() / max(spread, 1e-12)),
                  "n_pool_rows_compared": int(sub.shape[0]),
                  "note": ("⭐ the corner is REMOVED from the pool, so ⛔ NO stored row "
                           "carries a target tuple — a lookup cannot be VALID here, "
                           "whatever this distance is.")}})

    if "knn_proto" in todo:
        protos = FactorPrototypes.fit(grid, Zs, s["pool"])
        rq = requests_from_rows(grid, s["dev_rows"])
        feats = np.asarray(protos.embed_many(rq), np.float32).reshape(
            (-1,) + tuple(c["feat_shape"]))
        led = protos.byte_ledger()
        assert int(led["bytes"]) == knn_b, f"{led['bytes']} != {knn_b}"
        emit("knn_proto", feats, led["bytes"],
             "THE RIVAL — pinned FactorPrototypes, fitted on the SAME pool",
             {"n_prototypes": int(led["n_prototypes"]),
              "thin_values": {c["names"][k]: [int(x) for x in cnt if x < 50]
                              for k, cnt in enumerate(protos.n_per_value)}})

    if "store_removed_zero" in todo:
        import jax
        from chlu.core.gen_store import GenStore
        wc = tuple(tuple(int(i) for i in row) for row in st.well_channels)
        z = GenStore(wc, M, tuple(c["feat_shape"]), jax.random.PRNGKey(0),
                     init=[np.zeros_like(np.asarray(m)) for m in st.modes])
        emit("store_removed_zero", np.asarray(clu_wide_vector(z, a_ref), np.float32), tot_b,
             "§C.2 LAUNDERING CONTROL — identical read path, atoms ZEROED")

    if "store_removed_shuffle" in todo:
        import jax
        from chlu.core.gen_store import GenStore
        rng = np.random.default_rng(int(seed) * 104_729 + 7)
        wc = tuple(tuple(int(i) for i in row) for row in st.well_channels)
        allm = np.stack([np.asarray(m) for m in st.modes])          # (K, M, d)
        flat = allm.reshape(-1, allm.shape[-1])
        flat = flat[rng.permutation(flat.shape[0])]
        sh = GenStore(wc, M, tuple(c["feat_shape"]), jax.random.PRNGKey(0),
                      init=list(flat.reshape(allm.shape)))
        emit("store_removed_shuffle", np.asarray(clu_wide_vector(sh, a_ref), np.float32),
             tot_b, "§C.2 LAUNDERING CONTROL — atoms SHUFFLED across wells, bytes identical")

    if "head_additive_only" in todo:
        h0 = fit_factored_head(st, Zs, lab_store, cards, [], 0, RIDGE)
        a0 = h0.coefficients(requests_from_labels(lab_dev, "full", 0, seed))
        emit("head_additive_only", np.asarray(clu_wide_vector(st, a0), np.float32),
             comp.store_bytes_of(M, D) + comp.head_bytes(cards, [], K, M, 0),
             "the OTHER half of §C.2 — atoms kept, pair interactions removed")

    S = n_lookup_slots(tot_b, D)
    if "lookup_feat" in todo:
        stride = max(1, Zs.shape[0] // 20000)
        sub_rows = np.arange(0, Zs.shape[0], stride, dtype=np.int64)   # global indices
        med = sub_rows[_kmedoid_rows(Zs[sub_rows], S, seed)]
        emit("lookup_feat", lookup_read(Zs, lab_store, lab_dev, med, grid).reshape(
            (-1,) + tuple(c["feat_shape"])), int(S * D * 4),
            f"§C.1 MATCHED-BYTES LOOKUP — {S} real pool rows, FEATURE k-medoids, "
            "addressed by nearest requested tuple", {"n_slots": int(S)})

    if "lookup_tuple" in todo:
        V = _tuple_space(grid, lab_store)
        med = _kmedoid_rows(V, S, seed)
        emit("lookup_tuple", lookup_read(Zs, lab_store, lab_dev, med, grid).reshape(
            (-1,) + tuple(c["feat_shape"])), int(S * D * 4),
            f"§C.1 MATCHED-BYTES LOOKUP, STRONGER FORM — {S} real pool rows chosen to "
            "cover the FACTOR space", {"n_slots": int(S)})

    if "ceiling" in todo:
        # ⛔ NOT AN ARM and never reportable as one: it decodes the TRUE cached feature of
        # each held-out row, so it sees exactly what no arm may see. ⭐ R2 requires the
        # decoder ceiling reported BESIDE the headline; this measures it on MY corners
        # instead of inheriting 67.67 ± 2.49 from another cell.
        Xd = np.asarray(c["gather"](s["dev_rows"]), np.float32)
        emit("ceiling", Xd.reshape((-1,) + tuple(c["feat_shape"])), 0,
             "⛔ NOT AN ARM — the DECODER CEILING: the true cached feature of each "
             "held-out row, decoded. Reported beside the headline as R2 requires.")

    if "const_mu" in todo:
        mu = Zs.mean(0)
        emit("const_mu", np.repeat(mu[None], lab_dev.shape[0], 0).reshape(
            (-1,) + tuple(c["feat_shape"])), int(D * 4),
            "the CONSTANT-OUTPUT FLOOR — one tuple by construction; "
            "⚠ the bottom of the aggregate is free and this measures how much")

    log(f"[c{ci}] done in {time.time() - t0:.1f}s")
    return {}
