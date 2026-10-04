#!/usr/bin/env python
"""ONE resumable unit: score one built store, REQUEST-ADDRESSED, through the frozen
decoder + external probe.  Reads `<out>/<tag>/modes.npz`, writes `eval.json` +
`preds.npz`.  Separate from `build_unit.py` so a scoring bug costs a 30 s re-score,
not a 7 min rebuild.

*** WHY REQUEST-ADDRESSED AND NOT FEATURE-ADDRESSED ***
`exp_gen_store`'s `read` metric addresses the store with the eval row's OWN cached
features.  That read cannot answer this hub's question: "is the deleted value
unrecoverable from every *request* naming it".  A feature-addressed read is handed
the answer in its input.  The deployed path is

    request (factor tuple) -> address_requests(book) -> argmax mode per well
                           -> read_from_codes(store) -> decoder -> probe

and NOTHING on it touches the feature cache (`address_requests`' own docstring
records this).  It is also the only frame in which `knn_proto` exists at all, so it
is the only frame in which the rival comparison the charter demands is possible.

Three reads are scored on every request set, because the difference between them is
where a deletion claim lives:
  clu           the arm.
  store_removed the LAUNDERING FLOOR (`delete_store(mode="mean")`) at identical
                bytes and identical codes: accuracy that survives it is accuracy
                the store never supplied.  ⛔ Not this hub's delete operator.
  feature_addr  the companion, for separating "the STORE lost v" from "the
                ADDRESSING lost v".  Reported beside, never instead.

⛔ No encoder is constructed here; features come from the frozen cache.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import jax  # noqa: E402

from chlu.config import CHLUConfig  # noqa: E402
from chlu.core.gen_bridge import (  # noqa: E402
    assert_decoder_hash,
    dataset_module,
    features_for_rows,
    load_feature_cache,
    param_hash,
)
from chlu.core.gen_compose import compose_codes, read_compose, request_posterior, store_removed_read  # noqa: E402
from chlu.core.gen_dream import AddressBook  # noqa: E402
from chlu.core.gen_store import GenStore, address_codes, read_from_codes  # noqa: E402
from chlu.eval.gen_probe import decode_features, factor_labels, predict_factors  # noqa: E402
from chlu.experiments.exp_gen_deletion import store_digest  # noqa: E402
from chlu.experiments.exp_gen_store import bridge_dir, select_store_rows  # noqa: E402
from chlu.utils.checkpoints import load_model  # noqa: E402


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def eval_row_sets(split, config, seed: int, n_dev: int,
                  targets: dict | None = None, n_target: int = 1024) -> dict:
    """The row sets every arm is scored on — fixed once, shared by every build.

    ⭐ `dev` is drawn from train rows the FULL store pool never held, so it stays a
    clean read after any deletion (a deletion only shrinks the pool).  ⛔ `b` is the
    72 held-out recombination rows: PREREG'd scored runs only.

    ⭐ `targets` adds one row set per DELETED THING — "requests that name `v`" —
    drawn from the same `free` rows.  Charter falsifier 2 needs the *pre-deletion*
    recoverability of exactly those requests, so every unit (deleted, placebo and
    undeleted alike) is scored on the **identical** targeted sets.  ⛔ They are
    built from the predicate and a fixed seed, never from a build's own artefact,
    so no unit can be handed a set the others did not see.
    """
    rows = select_store_rows(split, config, int(seed))
    pool = np.asarray(rows["store_train"], np.int64)
    train = np.asarray(split.train_rows, np.int64)
    free = np.setdiff1d(train, np.union1d(pool, np.asarray(rows["eval_a"], np.int64)))
    rng = np.random.default_rng(20_260_917)  # ⛔ fixed: the dev set never moves
    dev = np.sort(rng.choice(free, size=min(int(n_dev), free.size), replace=False))
    out = {"a": np.asarray(rows["eval_a"], np.int64),
           "b": np.asarray(rows["eval_b"], np.int64), "dev": dev}
    if targets:
        from chlu.experiments.exp_gen_deletion import factor_tuples, row_predicate
        tup = factor_tuples(split.grid, free)
        for name, spec in sorted(targets.items()):
            m = row_predicate(tup, [(int(f), int(v)) for f, v in spec])
            cand = free[m]
            # ⛔ a fixed per-target seed.  ⚠ NOT `hash(name)` — Python's str hash is
            # PYTHONHASHSEED-randomised, which would silently give two processes two
            # different targeted sets.  sha256 of the name is stable across processes.
            import hashlib as _hl
            _sd = int(_hl.sha256(name.encode()).hexdigest()[:8], 16)
            r = np.random.default_rng(20_260_917 + _sd)
            k = min(int(n_target), cand.size)
            out[name] = np.sort(r.choice(cand, size=k, replace=False)) if k else cand
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dataset", default="shapes3d")
    ap.add_argument("--pretext", default="recon")
    ap.add_argument("--n-dev", type=int, default=4096)
    ap.add_argument("--with-b", action="store_true",
                    help="also score the 72 HELD-OUT rows (PREREG'd scored runs only)")
    ap.add_argument("--targets", default=None,
                    help="JSON {name: [[factor,value],...]} — one targeted request set per "
                         "deleted thing; identical for every unit (falsifier 2)")
    ap.add_argument("--n-target", type=int, default=1024)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    targets = json.loads(Path(args.targets[1:]).read_text()) if (
        args.targets or "").startswith("@") else (
        json.loads(args.targets) if args.targets else None)

    t0 = time.perf_counter()
    out_dir = Path(args.out) / args.tag
    done = out_dir / "eval.json"
    if done.exists() and not args.force:
        log(f"SKIP {args.tag} — already scored")
        return 0
    with open(out_dir / "build.json") as fh:
        bman = json.load(fh)

    config = CHLUConfig()
    config.gen_store.dataset, config.gen_store.pretext = args.dataset, args.pretext
    bdir = bridge_dir(config, args.dataset, args.pretext, args.seed)
    crows, feats, cman = load_feature_cache(bdir, verify=False)
    feat_shape = tuple(int(x) for x in cman["feat_shape"])
    with open(bdir / "bridge_manifest.json") as fh:
        bridge_man = json.load(fh)
    decoder = load_model(bdir / "decoder.pkl")
    probe = load_model(bdir / "probe.pkl")
    # ⛔ THE shared decoder. An arm that silently retrained it fails right here.
    assert_decoder_hash(decoder, bridge_man["decoder_param_hash"])
    log(f"decoder hash OK ({bridge_man['decoder_param_hash'][:16]}…)")

    mod = dataset_module(args.dataset)
    split, values = mod.build_split(config.gen_data.data_root)
    grid = split.grid
    names = list(grid.names)

    # --- reload store + book from the unit's own artefact, re-assert the digest --
    z = np.load(out_dir / "modes.npz")
    nw = len([k for k in z.files if k.startswith("well_")])
    nf = len([k for k in z.files if k.startswith("cond_")])
    store = GenStore([z[f"chan_{j}"] for j in range(nw)], int(z["well_0"].shape[0]),
                     feat_shape, jax.random.PRNGKey(0),
                     init=[z[f"well_{j}"] for j in range(nw)])
    book = AddressBook(z["log_prior"], [z[f"cond_{f}"] for f in range(nf)])
    dg = store_digest(store, z["labels"], book)
    if dg["store_digest"] != bman["digest"]["store_digest"]:
        raise AssertionError(
            f"⛔ the store+book reloaded from modes.npz does not hash to the build's "
            f"digest:\n   build {bman['digest']['store_digest']}\n   load  {dg['store_digest']}")
    log(f"round-trip digest OK: {dg['store_digest'][:24]}…")

    sets = eval_row_sets(split, config, args.seed, args.n_dev, targets, args.n_target)
    want = ["a", "dev"] + (["b"] if args.with_b else []) + sorted(targets or {})

    results, saved = {}, {}
    for nm in want:
        r = sets[nm]
        labs = np.asarray(factor_labels(grid, r), np.int32)
        # ⭐ A VALID REQUEST **IS** THE ROW'S FACTOR TUPLE (`requests_from_rows`'
        # docstring). Nothing below reads a feature to build the request.
        R = labs.astype(np.int64)
        P = request_posterior(book, R)
        cc = compose_codes(P)
        codes_req = np.asarray(cc["code"], np.int64)

        reads = {
            "clu": np.asarray(read_compose(store, codes_req), np.float32),
            "store_removed": np.asarray(
                store_removed_read(store, codes_req, "mean", seed=args.seed), np.float32),
        }
        # companion: the feature-addressed read (exp_gen_store's `read` metric)
        X = features_for_rows(crows, feats, r).reshape((r.size, -1))
        codes_fea = np.asarray(address_codes(store, X), np.int64)
        reads["feature_addr"] = np.asarray(read_from_codes(store, codes_fea), np.float32)

        results[nm] = {"n": int(r.size), "reads": {},
                       "mean_confidence": float(cc["mean_confidence"]),
                       "n_distinct_code_configs": int(cc["n_distinct_code_configs"]),
                       "code_agreement_request_vs_feature": float(
                           (codes_req == codes_fea).mean())}
        for rn, feat in reads.items():
            dec = decode_features(decoder, feat, 256)
            preds = np.asarray(predict_factors(probe, dec, 256), np.int32)
            correct = (preds == labs)
            results[nm]["reads"][rn] = {
                "per_factor": {names[j]: float(correct[:, j].mean()) for j in range(len(names))},
                "aggregate": float(np.mean([correct[:, j].mean() for j in range(len(names))])),
                "exact_match": float(correct.all(axis=1).mean()),
                "decoded_mean": float(dec.mean()),
            }
            saved[f"{nm}_{rn}_preds"] = preds
        saved[f"{nm}_rows"] = r
        saved[f"{nm}_labels"] = labs
        saved[f"{nm}_codes_req"] = codes_req.astype(np.int32)
        saved[f"{nm}_codes_fea"] = codes_fea.astype(np.int32)
        log(f"  [{nm}] n={r.size} clu={results[nm]['reads']['clu']['aggregate']:.4f} "
            f"floor={results[nm]['reads']['store_removed']['aggregate']:.4f} "
            f"feataddr={results[nm]['reads']['feature_addr']['aggregate']:.4f} "
            f"conf={cc['mean_confidence']:.3f}")

    np.savez(out_dir / "preds.npz", **saved)
    payload = {
        "schema": "chlu-hub-deletion/eval/2",
        "tag": args.tag, "seed": args.seed, "factor_names": names,
        "cardinalities": [int(c) for c in grid.cardinalities],
        "build_digest": bman["digest"]["store_digest"],
        "n_live": bman["n_live"], "n_deleted": bman["n_deleted"],
        "predicate": bman.get("predicate"), "with_b": bool(args.with_b),
        "n_dev": int(args.n_dev), "targets": targets, "n_target": int(args.n_target), "sets": results,
        "store_bytes": int(64 * 2048 * 4),
        "decoder_param_hash": bridge_man["decoder_param_hash"],
        "probe_param_hash": param_hash(probe),
        "eval_wall_s": time.perf_counter() - t0,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(done, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=float)
        fh.write("\n")
    log(f"WROTE {done} ({payload['eval_wall_s']:.1f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
