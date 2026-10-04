"""cl-encoder — fit ONE conv trunk, then sweep the cheap read-out choices.

The trunk fit is the only expensive part; the h → φ head (dim, whitening, L2/cosine,
spatial pooling) and the memory budget are all post-hoc numpy. So: fit once, cache the
trunk features for the whole stream, then sweep. Every number is the SAME registered
gate metric (ring-buffer kNN-in-φ, Class-IL, end-of-stream ACC).

⚠ Sweeping the read-out on the gate metric is hyper-parameter selection ON the decision
metric — legitimate only because it is done on **seed 0** and the winner is re-measured
on held-out seeds 1,2 before any verdict. Reported as such.
"""

import argparse
import json
import os
import time

import numpy as np

from chlu.config import get_default_config
from chlu.experiments.cl_baselines import cl_metrics  # read-only import (not my file)
from chlu.experiments.exp_cl_entry import RingBufferKNN, apply_cifar10, build_cl_stream
from chlu.experiments.exp_phi_read_in import build_read_in
from chlu.experiments.phi_encoders import _PCAHead


def gate_lines(stream, feats_train, feats_test, budget, n_tasks):
    """Ring-buffer kNN-in-φ (the gate/launder) + full-stream kNN, Class-IL."""
    A_ring = np.zeros((n_tasks, n_tasks))
    A_full = np.zeros((n_tasks, n_tasks))
    ring = RingBufferKNN(budget)
    keys, labels = [], []
    for t in range(n_tasks):
        kt = feats_train[t]
        yt = np.asarray(stream["train_y"][t]).astype(int)
        for i in range(len(kt)):
            ring.offer(kt[i], int(yt[i]))
        keys.append(kt)
        labels.append(yt)
        K, L = np.concatenate(keys), np.concatenate(labels)
        for i in range(t + 1):
            fq = feats_test[i]
            ye = np.asarray(stream["test_y"][i]).astype(int)
            A_ring[t, i] = float(np.mean(ring.predict(fq) == ye))
            d2 = (fq**2).sum(1)[:, None] - 2 * fq @ K.T + (K**2).sum(1)[None, :]
            A_full[t, i] = float(np.mean(L[d2.argmin(1)] == ye))
    return A_ring, A_full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="simclr")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--regime", default="task1_only")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--n-fit-region", type=int, default=0)
    ap.add_argument("--n-fit-pool", type=int, default=0)
    ap.add_argument("--dims", default="32,64,128,256")
    ap.add_argument("--budget", type=int, default=200)
    ap.add_argument("--budgets", default="", help="extra budget sweep at the best head")
    ap.add_argument("--heads", default="pca,pca_whiten")
    ap.add_argument("--save-features", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    config = get_default_config()
    apply_cifar10(config)
    cfg = config.experiment_cl_entry
    if args.n_fit_region:
        cfg.n_fit_region = args.n_fit_region
    if args.n_fit_pool:
        cfg.n_fit_pool = args.n_fit_pool
    cfg.phi_arm = args.arm
    cfg.enc_steps = args.steps
    cfg.enc_batch = args.batch
    cfg.enc_head = "none"  # keep raw h; the head is swept below

    stream = build_cl_stream(cfg, args.seed)
    pool = stream[f"fit_pool_{args.regime}"]
    print(f"fit pool = {len(pool)} images", flush=True)
    t0 = time.time()
    phi, prov = build_read_in(args.arm, cfg.dataset, stream["train_X"][0], pool,
                              cfg, args.seed)
    fit_s = time.time() - t0
    print(f"fit {args.arm} in {fit_s:.0f}s: {prov}", flush=True)

    Htr = [np.asarray(phi(stream["train_X"][t]), np.float32) for t in range(cfg.n_tasks)]
    Hte = [np.asarray(phi(stream["test_X"][t]), np.float32) for t in range(cfg.n_tasks)]
    Hfit = np.asarray(phi(pool), np.float32)

    rows = []
    for spatial in ("keep", "gap"):
        def sp(H, spatial=spatial):
            if spatial == "keep":
                return H
            c = H.shape[1] // (cfg.enc_pool**2)
            return H.reshape(len(H), c, -1).mean(-1)
        Hfit_s, Htr_s, Hte_s = sp(Hfit), [sp(h) for h in Htr], [sp(h) for h in Hte]
        for head in args.heads.split(","):
            for dim in [int(d) for d in args.dims.split(",")]:
                if dim > Hfit_s.shape[1]:
                    continue
                H = _PCAHead(Hfit_s, dim, whiten=(head == "pca_whiten"))
                ftr, fte = [H(h) for h in Htr_s], [H(h) for h in Hte_s]
                for l2 in (False, True):
                    if l2:
                        ftr = [f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)
                               for f in ftr]
                        fte = [f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)
                               for f in fte]
                    A_ring, A_full = gate_lines(stream, ftr, fte, args.budget,
                                                cfg.n_tasks)
                    row = {
                        "arm": args.arm, "seed": args.seed, "regime": args.regime,
                        "steps": args.steps, "batch": args.batch,
                        "n_fit": int(len(pool)), "spatial": spatial, "head": head,
                        "phi_dim": dim, "l2": l2, "budget": args.budget,
                        "acc_ring": float(np.mean(A_ring[-1])),
                        "acc_full_stream": float(np.mean(A_full[-1])),
                        "metrics_ring": cl_metrics(A_ring),
                        "loss_first": prov.get("loss_first"),
                        "loss_final": prov.get("loss_final"),
                        "fit_seconds": fit_s,
                    }
                    rows.append(row)
                    print(f"{spatial:>4} {head:>10} d={dim:<4} l2={int(l2)} "
                          f"ring={row['acc_ring']:.4f} full={row['acc_full_stream']:.4f}",
                          flush=True)
    if args.save_features:
        np.savez_compressed(
            args.save_features,
            **{f"train{t}": Htr[t] for t in range(cfg.n_tasks)},
            **{f"test{t}": Hte[t] for t in range(cfg.n_tasks)},
            **{f"ytrain{t}": np.asarray(stream["train_y"][t]) for t in range(cfg.n_tasks)},
            **{f"ytest{t}": np.asarray(stream["test_y"][t]) for t in range(cfg.n_tasks)},
            fit=Hfit,
        )

    best = max(rows, key=lambda r: r["acc_ring"])
    for b in [int(x) for x in args.budgets.split(",") if x]:
        def sp2(H, spatial=best["spatial"]):
            if spatial == "keep":
                return H
            c = H.shape[1] // (cfg.enc_pool**2)
            return H.reshape(len(H), c, -1).mean(-1)
        Hh = _PCAHead(sp2(Hfit), best["phi_dim"], whiten=(best["head"] == "pca_whiten"))
        ftr = [Hh(sp2(h)) for h in Htr]
        fte = [Hh(sp2(h)) for h in Hte]
        if best["l2"]:
            ftr = [f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8) for f in ftr]
            fte = [f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8) for f in fte]
        A_ring, A_full = gate_lines(stream, ftr, fte, b, cfg.n_tasks)
        row = dict(best)
        row.update({"budget": b, "acc_ring": float(np.mean(A_ring[-1])),
                    "acc_full_stream": float(np.mean(A_full[-1])),
                    "metrics_ring": cl_metrics(A_ring), "budget_sweep": True})
        rows.append(row)
        print(f"budget={b:<6} ring={row['acc_ring']:.4f}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2, default=float)
    best = max([r for r in rows if not r.get("budget_sweep")],
               key=lambda r: r["acc_ring"])
    print("BEST:", json.dumps({k: best[k] for k in
                               ("spatial", "head", "phi_dim", "l2", "acc_ring",
                                "acc_full_stream")}))


if __name__ == "__main__":
    main()
