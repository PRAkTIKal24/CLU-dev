"""cl-encoder — store-side geometry + the same-keys launder at a new φ arm.

Walks the CIFAR Class-IL stream with the production `PhiStore` (MVC-0 controller:
admission gate, class-balanced eviction, `s = clu_s_frac · median-NN`) and reports the
required geometry — median-NN address spacing, `s`, `σ_q`, corrected packing slack
(⛔ never the retracted 1.08) — plus the store-keys kNN line. **No CLU settle is run**;
this is address-space geometry, not a retrieval claim.
"""

import argparse
import json

import numpy as np

from chlu.config import get_default_config
from chlu.experiments.exp_cl_entry import (
    PhiStore,
    _geometry_report,
    apply_cifar10,
    build_cl_stream,
    build_phi,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="simclr")
    ap.add_argument("--phi-dim", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--regime", default="task1_only")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--head", default="pca")
    ap.add_argument("--l2", type=int, default=1)
    ap.add_argument("--n-fit-region", type=int, default=0)
    ap.add_argument("--n-fit-pool", type=int, default=0)
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
    cfg.phi_dim = args.phi_dim
    cfg.enc_steps = args.steps
    cfg.enc_head = args.head
    cfg.enc_l2_normalize = bool(args.l2)

    stream = build_cl_stream(cfg, args.seed)
    phi, prov = build_phi(args.regime, stream, cfg, args.seed)
    store = PhiStore(cfg, cfg.phi_dim, args.seed)
    per_task = []
    A = np.zeros((cfg.n_tasks, cfg.n_tasks))
    for t in range(cfg.n_tasks):
        keys_t = np.asarray(phi(stream["train_X"][t]), float)
        yt = np.asarray(stream["train_y"][t]).astype(int)
        if t == 0:
            store.set_width(keys_t[: cfg.s_init_items])
        elif cfg.s_policy == "refit":
            _, centers, _ = store.live()
            store.set_width(centers)
        stats0 = dict(store.ctrl.stats)
        for i in range(len(keys_t)):
            store.offer(keys_t[i], int(yt[i]), t)
        stats = {k: store.ctrl.stats[k] - stats0.get(k, 0) for k in store.ctrl.stats}
        stats.update({"task": t, "offered": int(len(keys_t)),
                      "n_live": int(store.ctrl.n_live),
                      "well_width_s": float(store.s),
                      "d_safe": float(store.ctrl.d_safe),
                      "admitted_fraction": stats["admitted"] / max(1, len(keys_t))})
        per_task.append(stats)
        for i in range(t + 1):
            fq = np.asarray(phi(stream["test_X"][i]), float)
            ye = np.asarray(stream["test_y"][i]).astype(int)
            A[t, i] = float(np.mean(store.knn_predict(fq) == ye))

    _, centers, _ = store.live()
    fq_all = np.concatenate(
        [np.asarray(phi(stream["test_X"][i]), float) for i in range(cfg.n_tasks)]
    )
    out = {
        "arm": args.arm, "phi_dim": cfg.phi_dim, "seed": args.seed,
        "regime": args.regime, "head": args.head, "l2": bool(args.l2),
        "steps": args.steps, "n_fit_pool": int(prov.get("n_fit_pool", 0)),
        "geometry": _geometry_report(centers, fq_all, store.s),
        "per_task": per_task,
        "acc_knn_same_keys": float(np.mean(A[-1])),
        "memory_items": int(len(centers)),
        "memory_floats": int(len(centers) * cfg.phi_dim),
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(json.dumps(out, indent=2, default=float)[:3000])


if __name__ == "__main__":
    main()
