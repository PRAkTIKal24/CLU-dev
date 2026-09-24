"""w21 spoke/ladder-rivals-32k -- FIT one rival on the LADDER STORE'S OWN content pool.

Vendored from w20 `scripts/w20_fit_rival.py` (@7627afd).  w21 changes, and only these:
  * `--where` must be `rung:<k>`; the pool is truncated to `store_pool_n(rung, seed)` rows,
    READ FROM THE w12b ARTEFACTS (LEG.json n_fit_rows == partitions.json n_store_train), i.e.
    exactly `pool[:n_store_train]`, the rows the ladder store's content was fitted on.
    ⛔ no --npool override: the pool size is never typed in.
  * output tag `rung<k>_n<N>` (so labels can never collide with w20's 131,072-row set);
  * the spoke-A selection is RE-RUN on those rows (no --hp-from transfer from a
    different-size pool);
  * `--ownpool-sched` adds stage `ownpoolsched`: learned_ref_ownpool's schedule (20 epochs,
    lr 1e-3, wd 0) on the same rows -> the SAME-ROWS, SAME-STEPS cross-check;
  * `--steps-match N` adds stage `steps<N>`: the selected (lr, wd) at N optimiser steps
    (step-matched to a w20 131,072-row fit).

--- original w20 docstring ---
w20 hub/shapes-data-matched -- FIT one data-matched rival.  NO SCORING HERE.

Fits ONE family at the 131,072-parameter read-path pin (`w12_deletion_baselines.
matched_arm`, the pin T7's comparators carry) on the store's EXACT 131,072-row fit pool
for the named condition / rung, by the spoke-A (wave-4) protocol:

  stage 1  lr x weight-decay on the SHORT schedule (w4_learners.LR_GRID x WD_GRID,
           TUNE_EPOCHS = 8), 6 trials, held-out MSE on a fixed 10 % of the pool;
  stage 2  the epoch axis at FULL length (w4_learners.EPOCH_GRID = 20, 40) at the stage-1
           winner, held-out MSE recorded EVERY epoch -> the selection curve;
  DECLARED the selected (lr, wd, epochs) refitted on ALL 131,072 pool rows -> the rival
           that carries the margin.  Its training curve is written per epoch;
  TENX     the same (lr, wd) at 10x the epochs (10x the optimiser steps), on ALL rows ->
           the convergence DIAGNOSTIC column.  Its curve is written per epoch too.

With --hp-from <condition>, stages 1-2 are NOT re-run: the (lr, wd, epochs) selected for
the same seed and family on that condition's pool are transferred (used for the ladder
rungs, whose pools are the same size in the same feature space).  Declared where used.

Nothing is scored here, so nothing here touches a held-out corner.  Resumable: a stage
whose artefact exists is skipped.
"""
from __future__ import annotations

import argparse
import sys
import time
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from w21_lib import (D, OUT, SEL_RNG, TENX, assert_worktree, jdump, load_pool, log, store_pool_n,  # noqa: E402
                     selection_split, train_curve)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--where", required=True, help="val | test | rung:<k>")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--hp-from", default=None, help="transfer the selection from val|test")
    ap.add_argument("--no-tenx", action="store_true")
    ap.add_argument("--ownpool-sched", action="store_true")
    ap.add_argument("--steps-match", type=int, default=0)
    ap.add_argument("--only-extra", action="store_true", help="skip declared/tenx (already on disk)")
    ap.add_argument("--suffix", default="", help="separate output tree (DEV smoke, or a declared diagnostic)")
    ap.add_argument("--delete-scope", default=None,
                    help="value | conjunction_live | item: RETRAIN-WITHOUT on the store's pool "
                         "minus EXACTLY the rows w12_deletion_rows.py downdates (Head AMENDMENT-1 ii)")
    ap.add_argument("--epochs-mult", type=int, default=1, help="DIAGNOSTIC: multiply the declared epochs")
    a = ap.parse_args(argv)

    wt = assert_worktree()
    log("chlu.__file__ = %s" % wt)
    import jax
    from chlu.eval.rivals import w4_api, w4_learners as L
    from chlu.eval.rivals import w12_deletion_baselines as W12B
    from chlu.experiments import exp_gen_factor_wells as fw

    seed, fam = int(a.seed), a.family
    if not a.where.startswith("rung:"):
        raise SystemExit("w21 fits LADDER rungs only (rung:<k>); got %r" % a.where)
    if a.delete_scope:
        raise SystemExit("w21 does no deletion retrains")
    spn = store_pool_n(a.where.split(":", 1)[1], seed)
    npool = int(spn["n"])
    log("store content pool READ FROM ARTEFACTS: n_store_train = %d (%s ; %s)"
        % (npool, spn["LEG.json"], spn["partitions.json"]))
    tag = a.where.replace(":", "") + ("_n%d" % npool) + a.suffix
    d = OUT / "rivals" / tag / ("seed%d" % seed) / fam
    d.mkdir(parents=True, exist_ok=True)
    root = OUT / "rivals" / tag / "ckpt"

    fw.PINS["seed"] = seed
    fw.PINS["condition"] = "val" if a.where == "val" else "test"
    cond = fw.load_condition()
    grid = cond["grid"]
    cards = [int(c) for c in grid.cardinalities]
    pool, _dead, pfile = load_pool(a.where, seed)
    n_full = int(pool.size)
    pool = pool[:npool]
    assert pool.size == npool, (pool.size, npool)
    log("pool: first %d of the rung's %d-row pool (= the store's content rows)" % (npool, n_full))
    del_info = None
    if a.delete_scope:
        import w12_deletion_rows as DR
        idx = DR.factor_indices(pool, cards)
        red = idx[:, DR.RED_FACTOR] == DR.RED_INDEX
        if a.delete_scope == "value":
            m = red
        elif a.delete_scope == "conjunction_live":
            m = red & (idx[:, DR.SHAPE_FACTOR] == DR.SPHERE_INDEX)
        elif a.delete_scope == "item":
            rng = np.random.default_rng(20_260_921 + seed)
            m = np.zeros(pool.size, bool)
            m[rng.choice(pool.size, int(round(DR.ITEM_FRACTION * pool.size)), replace=False)] = True
        else:
            raise SystemExit("unknown scope %r" % a.delete_scope)
        del_info = {"scope": a.delete_scope, "n_deleted": int(m.sum()),
                    "n_survivors_read": int((~m).sum()),
                    "mask_definition": "w12_deletion_rows.py masks, same pool, same RNG"}
        pool = pool[~m]
        log("RETRAIN-WITHOUT %s: deleted %d rows, refit on %d survivors"
            % (a.delete_scope, int(m.sum()), pool.size))
    log("where=%s seed=%d family=%s  pool %s rows from %s" % (a.where, seed, fam, pool.size, pfile))
    man = fw.feature_cache_manifest(seed)
    from chlu.experiments.exp_gen_delete_row7 import assert_generation2
    log("cache %s" % assert_generation2(seed))
    F = fw.features_mmap(seed)
    t = time.time()
    Y = np.asarray(fw.gather(F, pool), np.float32)
    log("gathered %s in %.0fs" % (Y.shape, time.time() - t))

    key = jax.random.PRNGKey(seed * 7717 + SEL_RNG + zlib.crc32(fam.encode()) % 10_000)
    key, k0 = jax.random.split(key)
    arm = W12B.matched_arm(fam, cards, D, k0, log=log)
    X = W12B._encode(arm, grid, pool, cards)
    meta_arm = {k: v for k, v in arm.items() if k != "build"}

    # ---- selection -----------------------------------------------------------------
    sel_path = d / "selection.json"
    if a.hp_from:
        src = OUT / "rivals" / a.hp_from / ("seed%d" % seed) / fam / "selection.json"
        # (with --epochs-mult the transferred epochs are multiplied; recorded below)
        import json
        sel = json.loads(src.read_text(encoding="utf-8"))
        sel = {"TRANSFERRED_FROM": str(src), "selected": sel["selected"],
               "epochs_mult": int(a.epochs_mult),
               "why": ("the ladder rungs' pools are the same size (131,072) in the same "
                       "feature space; the spoke-A selection made on the rows 3-4 pool "
                       "for this seed and family is transferred, not re-run")}
        jdump(sel, sel_path)
    elif sel_path.exists():
        import json
        sel = json.loads(sel_path.read_text(encoding="utf-8"))
        log("selection already on disk: %s" % sel["selected"])
    else:
        fi, vi = selection_split(pool.size, seed)
        Xf, Yf, Xv, Yv = X[fi], Y[fi], X[vi], Y[vi]
        short, t0 = [], time.time()
        for lr in L.LR_GRID:
            for wd in L.WD_GRID:
                key, tk = jax.random.split(key)
                m, info = train_curve(arm["build"](arm["width"]), Xf, Yf, tk, L.TUNE_EPOCHS,
                                      lr, wd, Xv=Xv, Yv=Yv, val_every=L.TUNE_EPOCHS)
                v = info["curve"][-1]["val_mse"]
                short.append({"lr": lr, "weight_decay": wd, "epochs": L.TUNE_EPOCHS,
                              "short_val_mse": v, "curve": info["curve"]})
                log("  short %d/6 lr=%g wd=%g val %.5f" % (len(short), lr, wd, v))
        sb = short[int(np.argmin([s["short_val_mse"] for s in short]))]
        full = []
        for ep in L.EPOCH_GRID:
            key, fk = jax.random.split(key)
            m, info = train_curve(arm["build"](arm["width"]), Xf, Yf, fk, int(ep),
                                  sb["lr"], sb["weight_decay"], Xv=Xv, Yv=Yv, val_every=1,
                                  logf=log)
            v = info["curve"][-1]["val_mse"]
            full.append({"lr": sb["lr"], "weight_decay": sb["weight_decay"],
                         "epochs": int(ep), "full_val_mse": v, "curve": info["curve"]})
            log("  full epochs=%d val %.5f" % (ep, v))
        fb = full[int(np.argmin([f["full_val_mse"] for f in full]))]
        sel = {"protocol": "spoke-A / wave-4 (w4_learners.fit_family): 6 short trials "
                           "lr x wd at TUNE_EPOCHS, then EPOCH_GRID at full length",
               "LR_GRID": list(L.LR_GRID), "WD_GRID": list(L.WD_GRID),
               "TUNE_EPOCHS": L.TUNE_EPOCHS, "EPOCH_GRID": list(L.EPOCH_GRID),
               "selection_split": {"fit": int(fi.size), "heldout": int(vi.size),
                                   "rng": SEL_RNG + seed},
               "short": short, "full": full,
               "selected": {"lr": fb["lr"], "weight_decay": fb["weight_decay"],
                            "epochs": fb["epochs"], "heldout_mse": fb["full_val_mse"]},
               "seconds": round(time.time() - t0, 1), "arm": meta_arm}
        jdump(sel, sel_path)
        log("SELECTED %s" % sel["selected"])

    s = sel["selected"]
    lr_s, wd_s = float(s["lr"]), float(s["weight_decay"])
    stages = [] if a.only_extra else [("declared", int(s["epochs"]) * int(a.epochs_mult), lr_s, wd_s)]
    if not a.no_tenx and not a.only_extra:
        stages.append(("tenx", TENX * int(s["epochs"]), lr_s, wd_s))
    spe = max(pool.size // 256, 1)
    if a.ownpool_sched:
        # learned_ref_ownpool (w12 composition-ladder AMENDMENT 1): 20 epochs, lr 1e-3, wd 0
        stages.append(("ownpoolsched", 20, 1e-3, 0.0))
    if a.steps_match:
        if a.steps_match % spe:
            raise SystemExit("steps %d not a whole number of epochs at %d steps/epoch"
                             % (a.steps_match, spe))
        stages.append(("steps%d" % a.steps_match, a.steps_match // spe, lr_s, wd_s))
    for stage, E, lr_e, wd_e in stages:
        rp = d / ("%s.json" % stage)
        if rp.exists():
            log("%s already on disk" % stage)
            continue
        key, fk = jax.random.split(jax.random.PRNGKey(seed * 7717 + SEL_RNG + 1 + len(stage)
                                                       + zlib.crc32(fam.encode()) % 10_000))
        log("%s: %d epochs on ALL %d rows (lr=%g wd=%g)" % (stage, E, pool.size, lr_e, wd_e))
        m, info = train_curve(arm["build"](arm["width"]), X, Y, fk, E, lr_e, wd_e, logf=log)
        name = "%s__%s" % (fam, stage)
        w4_api.save_learner(seed, name, m, arm["width"], arm["in_kind"], grid, D, man,
                            root=str(root), extra={"base_family": fam, "partial_ok": False, "w20_stage": stage, "n_fit_rows": int(pool.size),
                                                   "where": a.where, "param_target": 131_072})
        c = info["curve"]
        tail = [r["train_mse"] for r in c[-max(2, len(c) // 10):]]
        jdump({"stage": stage, "where": a.where, "seed": seed, "family": fam,
               "n_fit_rows": int(pool.size), "pool_file": pfile,
               "param_count": arm["param_count"], "param_target": arm["param_target"],
               "param_error": arm["param_error"], "width": arm["width"],
               "in_kind": arm["in_kind"], "selected": s, "epochs": E,
               "stage_lr": lr_e, "stage_weight_decay": wd_e,
               "store_pool_from_artefacts": spn, "rung_pool_rows_total": n_full,
               "optimizer_steps": info["steps"], "batch": info["batch"],
               "train_info": {k: v for k, v in info.items() if k != "curve"},
               "curve": c,
               "last_decile_rel_change": (float((tail[0] - tail[-1]) / max(tail[0], 1e-12))),
               "deletion": del_info,
               "cost_line": {"seconds": info["seconds"], "data_rows_read": int(pool.size),
                             "bytes_rewritten": int(arm["param_count"]) * 4,
                             "gradient_row_visits": int(info["steps"]) * int(info["batch"])},
               "checkpoint": str(root / ("seed_%d" % seed) / name),
               "chlu_file": wt}, rp)
        log("%s done: final train mse %.5f in %.0fs" % (stage, c[-1]["train_mse"],
                                                         info["seconds"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
