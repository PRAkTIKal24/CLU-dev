"""cl-encoder — the DECISIVE GATE harness.

kNN-in-φ, Class-IL, Split-CIFAR-10 reduced protocol, 200-item matched memory.
Gate: >= 0.35 (PCA-32 currently 0.219).

Everything scientific comes from production code (`build_cl_stream`, `build_read_in`,
`RingBufferKNN`, `cl_metrics`); this script only sequences arms/regimes/seeds and writes
JSON. No CLU settle is run — the gate is a property of the ADDRESS SPACE, not the store.

Usage:
  python gate_knn.py --arms pca:32,pca:64 --regimes task1_only --seeds 0 --out A.json
"""

import argparse
import ast
import json
import os
import time

import numpy as np

from chlu.config import get_default_config
from chlu.experiments.cl_baselines import cl_metrics  # read-only import (not my file)
from chlu.experiments.exp_cl_entry import (
    RingBufferKNN,
    apply_cifar10,
    build_cl_stream,
    build_phi,
)


def _nn_labels(fq, K, L):
    """1-NN labels via the gram identity (broadcasting blows memory at 5k keys)."""
    d2 = (fq**2).sum(1)[:, None] - 2.0 * fq @ K.T + (K**2).sum(1)[None, :]
    return L[d2.argmin(1)]


def knn_lines(cfg, stream, phi, budget):
    """Walk the stream; ring-buffer kNN-in-φ (the launder) + full-stream kNN (P4-1)."""
    T = cfg.n_tasks
    A_ring = np.zeros((T, T))
    A_full = np.zeros((T, T))
    ring = RingBufferKNN(budget)
    all_keys, all_labels = [], []
    test_feats = [np.asarray(phi(stream["test_X"][i]), float) for i in range(T)]
    for t in range(T):
        keys_t = np.asarray(phi(stream["train_X"][t]), float)
        yt = np.asarray(stream["train_y"][t]).astype(int)
        for i in range(len(keys_t)):
            ring.offer(keys_t[i], int(yt[i]))
        all_keys.append(keys_t)
        all_labels.append(yt)
        K = np.concatenate(all_keys)
        L = np.concatenate(all_labels)
        for i in range(t + 1):
            fq = test_feats[i]
            ye = np.asarray(stream["test_y"][i]).astype(int)
            A_ring[t, i] = float(np.mean(ring.predict(fq) == ye))
            A_full[t, i] = float(np.mean(_nn_labels(fq, K, L) == ye))
    return A_ring, A_full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="pca:32", help="comma list arm:phi_dim")
    ap.add_argument("--regimes", default="task1_only")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--n-fit-region", type=int, default=0)
    ap.add_argument("--n-fit-pool", type=int, default=0)
    ap.add_argument("--budget", type=int, default=0)
    ap.add_argument("--full-knn", action="store_true", help="also run the 10k-key kNN")
    ap.add_argument("--cfg", default="", help="k=v,... extra cfg overrides (literal)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    extra = {}
    for kv in [s for s in args.cfg.split(",") if s]:
        k, _, v = kv.partition("=")
        extra[k] = ast.literal_eval(v)

    rows = []
    for seed in [int(s) for s in args.seeds.split(",")]:
        # one stream per (seed, sizing) — shared by every arm/regime
        config = get_default_config()
        cfg = config.experiment_cl_entry
        if args.dataset == "cifar10":
            apply_cifar10(config)
        if args.n_fit_region:
            cfg.n_fit_region = args.n_fit_region
        if args.n_fit_pool:
            cfg.n_fit_pool = args.n_fit_pool
        for k, v in extra.items():
            if not hasattr(cfg, k):
                raise SystemExit(f"unknown cfg field {k!r}")
            setattr(cfg, k, v)
        budget = args.budget or cfg.memory_items
        t0 = time.time()
        stream = build_cl_stream(cfg, seed)
        print(f"[seed {seed}] stream built in {time.time()-t0:.1f}s; "
              f"fit_task1={len(stream['fit_pool_task1_only'])} "
              f"fit_generic={len(stream['fit_pool_generic_frozen'])}", flush=True)
        for spec in args.arms.split(","):
            arm, _, dim = spec.partition(":")
            cfg.phi_arm = arm
            if dim:
                cfg.phi_dim = int(dim)
            for regime in args.regimes.split(","):
                t0 = time.time()
                phi, prov = build_phi(regime, stream, cfg, seed)
                t_fit = time.time() - t0
                t0 = time.time()
                A_ring, A_full = knn_lines(cfg, stream, phi, budget)
                row = {
                    "seed": seed, "arm": arm, "phi_dim": int(cfg.phi_dim),
                    "regime": regime, "budget": int(budget),
                    "acc_ring": float(np.mean(A_ring[-1])),
                    "acc_full_stream": float(np.mean(A_full[-1])),
                    "A_ring": A_ring.tolist(), "A_full": A_full.tolist(),
                    "metrics_ring": cl_metrics(A_ring),
                    "metrics_full": cl_metrics(A_full),
                    "n_fit_pool": int(prov.get("n_fit_pool", prov.get("n_fit", 0))),
                    "overrides": {k: v for k, v in extra.items()},
                    "n_fit_region": int(cfg.n_fit_region),
                    "fit_seconds": t_fit, "eval_seconds": time.time() - t0,
                    "provenance": {k: v for k, v in prov.items()
                                   if isinstance(v, (int, float, str, bool, list))},
                }
                rows.append(row)
                print(json.dumps({k: row[k] for k in
                                  ("seed", "arm", "phi_dim", "regime", "acc_ring",
                                   "acc_full_stream", "fit_seconds")}), flush=True)
                os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
                with open(args.out, "w") as f:
                    json.dump(rows, f, indent=2, default=float)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
