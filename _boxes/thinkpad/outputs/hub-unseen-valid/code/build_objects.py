"""Fit, ONCE and deterministically, every object row 2 needs. Resumable: each stage
checkpoints and is skipped if its file is already on disk.

Objects, all at MATCHED BYTES (64 atoms x 2048 float32 = 524,288 B = the deployed store):
  store        the CLU landscape: K=5 wells, M=64 modes, trained by gen_store.train_store
  basis        the fitted dictionary: the SAME k-means++ atoms, landscape training NOT run
  book         the address book over the store's codes
  protos       FactorPrototypes (knn_proto's embedding), fit on the same rows

The store fit pool is the VALUE-HOLDOUT train split, so the store never held object_hue
0.5 and never held shape 0.0. The encoder/decoder/judge are frozen and were fitted on a
split containing every hue -- that is the scope statement, carried in every artifact.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import hub_common as hc  # noqa: E402

CKPT = hc.OUT / "objects"
SEED = 0
N_STORE_TRAIN = 32_768
N_FIT, N_VAL = 32_256, 512
N_MODES = 64
K_WELLS = 5
# gen_store defaults, pinned here explicitly (w3-common s3: pin the store fit COMPLETELY)
TRAIN_STEPS, TRAIN_BATCH, TRAIN_LR = 1200, 128, 3e-3
COMMIT_BETA, TAU_START, TAU_END = 0.25, 1.0, 0.1
SMOOTHING = 1.0
RIDGE_PHI, RIDGE_ATOM = 1e-2, 1e-6


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


# ---------------------------------------------------------------------------
def split_and_rows():
    """The value-holdout train pool -> (fit, val) by the pinned stratified rule."""
    z = np.load(hc.VH / "factor_index.npz", allow_pickle=True)
    train_rows = np.asarray(z["train_rows"], np.int64)
    test_rows = np.asarray(z["test_rows"], np.int64)
    rng = np.random.default_rng(SEED * 7919 + 20_260_910)
    n = min(N_STORE_TRAIN, train_rows.size)
    pool = np.sort(rng.permutation(train_rows)[:n])
    from chlu.data.gen_rowsplit import split_fit_val

    g = hc.load_grid()
    labels = g.factor_indices_of(pool)
    fit, val, info = split_fit_val(pool, labels, N_FIT, N_VAL, SEED, mode="stratified")
    return {
        "train_rows": train_rows, "test_rows": test_rows,
        "pool": pool, "fit": np.sort(fit), "val": np.sort(val), "split_info": info,
    }


def features_for(rows):
    from chlu.core.gen_bridge import features_for_rows, load_feature_cache

    croot = hc.BUNDLE / "gen-feature-bridge" / "runs" / "shapes3d_recon" / f"seed_{SEED}"
    crows, feats, man = load_feature_cache(croot, verify=False)
    X = features_for_rows(crows, feats, np.asarray(rows, np.int64))
    return np.asarray(X, np.float32), man


def well_channels():
    from chlu.core.gen_store import well_channels_from_labels

    plan = json.load(open(hc.BUNDLE / "gen-store-grouping" / "runs" / "shapes3d_recon"
                          / "store_plan.json", encoding="utf-8"))
    lab = np.asarray(plan["labels"], np.int64)
    return well_channels_from_labels(lab, int(plan["K"])), plan


# ---------------------------------------------------------------------------
def main():
    CKPT.mkdir(parents=True, exist_ok=True)
    import equinox as eqx
    import jax

    sp = split_and_rows()
    log(f"value-holdout split: train {sp['train_rows'].size} test {sp['test_rows'].size}; "
        f"store pool {sp['pool'].size} -> fit {sp['fit'].size} val {sp['val'].size}")

    wc, plan = well_channels()
    log(f"well channels from the pinned plan: K={len(wc)} sizes={[len(w) for w in wc]}")

    Xfit, man = features_for(sp["fit"])
    log(f"fit features {Xfit.shape} from cache scope={man['scope']} n_rows={man['n_rows']}")
    feat_shape = tuple(man["feat_shape"])

    from chlu.core.gen_store import (
        address_codes, init_store, store_bytes, train_store,
    )

    class _Cfg:
        train_steps, train_batch, train_lr = TRAIN_STEPS, TRAIN_BATCH, TRAIN_LR
        commitment_beta, tau_start, tau_end = COMMIT_BETA, TAU_START, TAU_END

    # ---- basis (k-means++ atoms, landscape training NOT run) ---------------
    bpath = CKPT / "basis.pkl"
    key = jax.random.PRNGKey(SEED)
    s0 = init_store(Xfit, wc, N_MODES, feat_shape, key, SEED)
    if not bpath.exists():
        with open(bpath, "wb") as fh:
            pickle.dump({"modes": [np.asarray(m, np.float32) for m in s0.modes]}, fh)
        log(f"basis (k-means++ dictionary) written: {store_bytes(s0)['store_bytes']} B")
    else:
        log("basis: already on disk, skipped")

    # ---- store (the landscape: the SAME init, then trained) ----------------
    spath = CKPT / "store.pkl"
    if not spath.exists():
        t0 = time.time()
        store, hist, _ = train_store(s0, Xfit, _Cfg, key, log=log)
        with open(spath, "wb") as fh:
            pickle.dump({"modes": [np.asarray(m, np.float32) for m in store.modes],
                         "history": hist}, fh)
        log(f"store trained in {time.time()-t0:.1f}s; "
            f"{store_bytes(store)['store_bytes']} B, bits/read {store.bits_per_read}")
    else:
        log("store: already on disk, skipped")
    with open(spath, "rb") as fh:
        sd = pickle.load(fh)
    from chlu.core.gen_store import GenStore
    store = GenStore(wc, N_MODES, feat_shape, key, init=sd["modes"])

    # ---- address book over the store's codes -------------------------------
    apath = CKPT / "book.pkl"
    if not apath.exists():
        codes = address_codes(store, Xfit.reshape(Xfit.shape[0], -1))
        g = hc.load_grid()
        Y = g.factor_indices_of(sp["fit"])
        from chlu.core.gen_dream import fit_address_book
        book = fit_address_book(codes, Y, g.cardinalities, smoothing=SMOOTHING)
        with open(apath, "wb") as fh:
            pickle.dump({"log_prior": np.asarray(book.log_prior),
                         "log_cond": [np.asarray(c) for c in book.log_cond],
                         "codes": codes}, fh)
        log(f"address book fitted; codes {codes.shape}, "
            f"distinct configs {np.unique(codes, axis=0).shape[0]}")
    else:
        log("book: already on disk, skipped")

    # ---- knn_proto prototypes ---------------------------------------------
    ppath = CKPT / "protos.pkl"
    if not ppath.exists():
        from chlu.core.gen_request import FactorPrototypes
        g = hc.load_grid()
        pr = FactorPrototypes.fit(g, Xfit, sp["fit"])
        with open(ppath, "wb") as fh:
            pickle.dump({"protos": [np.asarray(p, np.float32) for p in pr.protos],
                         "mu_global": np.asarray(pr.mu_global, np.float32),
                         "n_per_value": pr.n_per_value, "n_rows_fit": pr.n_rows_fit}, fh)
        log(f"prototypes fitted: {pr.n_prototypes} protos, ledger "
            f"{pr.byte_ledger()['bytes']} B")
    else:
        log("protos: already on disk, skipped")

    np.savez_compressed(CKPT / "rows.npz", fit=sp["fit"], val=sp["val"],
                        pool=sp["pool"], train_rows=sp["train_rows"],
                        test_rows=sp["test_rows"])
    hc.jdump({
        "scope": hc.SCOPE,
        "seed": SEED,
        "pin": {"n_store_train": N_STORE_TRAIN, "n_fit": N_FIT, "n_val": N_VAL,
                "n_modes": N_MODES, "k_wells": K_WELLS,
                "train_steps": TRAIN_STEPS, "train_batch": TRAIN_BATCH,
                "train_lr": TRAIN_LR, "commitment_beta": COMMIT_BETA,
                "tau_start": TAU_START, "tau_end": TAU_END,
                "address_book_smoothing": SMOOTHING,
                "ridge_phi": RIDGE_PHI, "ridge_atom": RIDGE_ATOM,
                "pool_rng": "default_rng(seed*7919 + 20260910), permutation, first n",
                "fit_val_rule": "gen_rowsplit.split_fit_val(mode='stratified')",
                "well_partition": "the PINNED store_plan.json labels, verbatim",
                "grouping_method": plan["grouping_method"]},
        "split_info": sp["split_info"],
        "feature_cache": {k: man[k] for k in
                          ("D", "feat_shape", "dtype", "n_rows", "scope",
                           "encoder_param_hash", "features_sha256")},
        "well_sizes": [len(w) for w in wc],
        "store_bytes": store_bytes(store),
    }, CKPT / "manifest.json")
    log("wrote objects/manifest.json")


if __name__ == "__main__":
    main()
