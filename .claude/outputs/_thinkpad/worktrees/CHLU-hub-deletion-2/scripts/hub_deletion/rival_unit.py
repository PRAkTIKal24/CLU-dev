#!/usr/bin/env python
"""ONE resumable unit: the RIVAL arms on the SAME live set and the SAME predicate.

`knn_proto` is R8's composed per-factor prototype read:

    x(request) = mu_global + sum_terms ( mu_{factor, value} - mu_global )

then straight through the same frozen decoder and the same external probe.  It is
canonical by construction — the table is per-(factor, value) means of the live rows,
so `delete(v)` is "refit the means on live \\ rows(v)" and nothing else.

*** THE SYMMETRY THAT MAKES THIS A FAIR COMPARISON, AND IT IS NOT AN ACCIDENT ***
After the deletion neither arm can *refuse*; both degrade the deleted term to
"uninformative", by their own shipped code and for the same reason:
  knn_proto  `FactorPrototypes.fit` falls a value with no live rows back to
             `mu_global` (its own comment), so that term contributes
             `mu_global - mu_global = 0`.
  clu        `fit_address_book` Laplace-smooths `P(mode | f = v)` over ZERO rows to
             exactly uniform `1/M`, so that term contributes `log(1/M) - log P(m)`,
             which is constant in `m` and cannot move an argmax.
⭐ So the deleted factor drops out of the address in BOTH arms and the render is
driven by the surviving terms in both.  Neither arm is handed the deletion for free,
which is exactly the condition the charter puts on the rivals.

⛔ `lace` and `ebm_compose` are NOT run here — see the report's method section.
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

from chlu.config import CHLUConfig  # noqa: E402
from chlu.core.gen_bridge import (  # noqa: E402
    assert_decoder_hash,
    dataset_module,
    features_for_rows,
    load_feature_cache,
    param_hash,
)
from chlu.core.gen_request import FactorPrototypes, requests_from_rows  # noqa: E402
from chlu.eval.gen_probe import decode_features, factor_labels, predict_factors  # noqa: E402
from chlu.experiments.exp_gen_store import bridge_dir  # noqa: E402
from chlu.utils.checkpoints import load_model  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_unit import eval_row_sets  # noqa: E402


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True, help="the CLU build tag whose live set to mirror")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dataset", default="shapes3d")
    ap.add_argument("--pretext", default="recon")
    ap.add_argument("--n-dev", type=int, default=4096)
    ap.add_argument("--with-b", action="store_true")
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
    done = out_dir / "rival.json"
    if done.exists() and not args.force:
        log(f"SKIP {args.tag} — rival already scored")
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
    assert_decoder_hash(decoder, bridge_man["decoder_param_hash"])

    mod = dataset_module(args.dataset)
    split, values = mod.build_split(config.gen_data.data_root)
    grid = split.grid
    names = list(grid.names)

    # ⭐ THE SAME live row vector the CLU build used — read off its artefact, not
    # recomputed, so the two arms cannot drift apart.
    z = np.load(out_dir / "modes.npz")
    live = np.sort(np.asarray(z["live"], np.int64))
    import hashlib
    got = hashlib.sha256(live.tobytes()).hexdigest()
    if got != bman["live_sha256"]:
        raise AssertionError("⛔ live row set does not match the build manifest")
    log(f"live = {live.size} rows (sha256 {got[:16]}…) — identical to the CLU arm's")

    X_live = features_for_rows(crows, feats, live).reshape((live.size, -1))
    protos = FactorPrototypes.fit(grid, X_live, live)
    led = protos.byte_ledger()
    # a value emptied by the deletion is visible here, not silent
    empty = {names[k]: [v for v, n in enumerate(protos.n_per_value[k]) if n == 0]
             for k in range(len(names))}
    empty = {k: v for k, v in empty.items() if v}
    log(f"knn_proto table: {led['n_prototypes']} prototypes, {led['bytes']} B; "
        f"values with ZERO live rows: {empty or 'none'}")

    sets = eval_row_sets(split, config, args.seed, args.n_dev, targets, args.n_target)
    want = ["a", "dev"] + (["b"] if args.with_b else []) + sorted(targets or {})
    results, saved = {}, {}
    for nm in want:
        r = sets[nm]
        labs = np.asarray(factor_labels(grid, r), np.int32)
        reqs = requests_from_rows(grid, r)
        # ⚠ the prototype table is flat (N, D); the frozen decoder takes the feature
        # MAP (N, C, H, W).  `score_composition` reshapes the same way — ⛔ this is
        # the shipped convention, not a reinterpretation of the rival's output.
        E = np.asarray(protos.embed_many(reqs), np.float32).reshape((-1,) + feat_shape)
        dec = decode_features(decoder, E, 256)
        preds = np.asarray(predict_factors(probe, dec, 256), np.int32)
        correct = (preds == labs)
        results[nm] = {
            "n": int(r.size),
            "per_factor": {names[j]: float(correct[:, j].mean()) for j in range(len(names))},
            "aggregate": float(np.mean([correct[:, j].mean() for j in range(len(names))])),
            "exact_match": float(correct.all(axis=1).mean()),
            "decoded_mean": float(dec.mean()),
        }
        saved[f"{nm}_knn_proto_preds"] = preds
        saved[f"{nm}_rows"] = r
        saved[f"{nm}_labels"] = labs
        log(f"  [{nm}] knn_proto aggregate={results[nm]['aggregate']:.4f}")

    np.savez(out_dir / "rival_preds.npz", **saved)
    payload = {
        "schema": "chlu-hub-deletion/rival/1",
        "arm": "knn_proto", "tag": args.tag, "seed": args.seed, "factor_names": names,
        "n_live": int(live.size), "n_deleted": bman["n_deleted"],
        "predicate": bman.get("predicate"), "live_sha256": got,
        "byte_ledger": led,
        "clu_store_bytes": int(64 * 2048 * 4),
        "values_with_zero_live_rows": empty,
        "n_per_value": {names[k]: [int(x) for x in protos.n_per_value[k]]
                        for k in range(len(names))},
        "sets": results, "with_b": bool(args.with_b), "targets": targets, "n_target": int(args.n_target),
        "decoder_param_hash": bridge_man["decoder_param_hash"],
        "probe_param_hash": param_hash(probe),
        "wall_s": time.perf_counter() - t0,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(done, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=float)
        fh.write("\n")
    log(f"WROTE {done} ({payload['wall_s']:.1f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
