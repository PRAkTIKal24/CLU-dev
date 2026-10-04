"""w21 -- X1 DIAGNOSTIC: replay `learned_ref_ownpool` with ITS OWN recipe AND PRNG keys.

A check on what differs between the two X1 arms, NOT a reconciliation: the X1 verdict was
pre-declared (LOG 13:58Z) and is computed from `mlp__ownpoolsched`, never from this arm.
Recipe copied from scripts/w12_ladder_1x.py `fit_one` (w12 composition-ladder AMENDMENT 1):
builders(cards, D, PRNGKey(seed))['mlp'] -> match_width(131,072) -> train(PRNGKey(seed+1),
20 epochs, lr 1e-3, wd 0, batch BATCH, rng=default_rng(seed)) on pool[:N_FIT_ROWS].
Saved as `mlp__refreplay` under this spoke's rung<k>_n32768 checkpoint tree.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from w21_lib import D, OUT, assert_worktree, jdump, load_pool, log, store_pool_n  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", required=True)
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args(argv)
    wt = assert_worktree()
    import jax
    from chlu.eval.rivals import w4_api, w4_learners as WL
    from chlu.experiments import exp_gen_factor_wells as fw

    seed, rung = int(a.seed), a.rung
    n = int(store_pool_n(rung, seed)["n"])
    fw.PINS["seed"] = seed
    fw.PINS["condition"] = "test"
    grid = fw.load_condition()["grid"]
    cards = [int(c) for c in grid.cardinalities]
    pool, _, pfile = load_pool("rung:" + rung, seed)
    build, kind = WL.builders(cards, int(D), jax.random.PRNGKey(seed))["mlp"]
    w, npar, err = WL.match_width(build, 131072, lo=1, hi=8192, log=lambda *x: None)
    model = build(w)
    rows = np.asarray(pool, np.int64)[:n]
    Y = fw.gather(fw.features_mmap(seed), rows)
    Xin = WL.onehot_inputs(grid.factor_indices_of(rows), cards)
    model, info = WL.train(model, Xin, Y, jax.random.PRNGKey(seed + 1), epochs=20, lr=1e-3, wd=0.0,
                           batch=WL.BATCH, log=None, rng=np.random.default_rng(seed))
    root = OUT / "rivals" / ("rung%s_n%d" % (rung, n)) / "ckpt"
    man = fw.feature_cache_manifest(seed)
    w4_api.save_learner(seed, "mlp__refreplay", model, w, kind, grid, D, man, root=str(root),
                        extra={"base_family": "mlp", "partial_ok": False, "w21_stage": "refreplay",
                               "n_fit_rows": int(rows.size), "param_target": 131072})
    jdump({"what": "X1 diagnostic replay of learned_ref_ownpool with its own recipe and PRNG keys",
           "rung": rung, "seed": seed, "width": int(w), "param_count": int(npar), "n_fit_rows": int(rows.size),
           "pool_file": pfile, "train_info": info, "chlu_file": wt},
          OUT / "rivals" / ("rung%s_n%d" % (rung, n)) / ("seed%d" % seed) / "mlp" / "refreplay.json")
    log("refreplay rung %s seed %d: width %d params %d steps %d loss_final %.5f"
        % (rung, seed, w, npar, info["steps"], info["loss_final"]))


if __name__ == "__main__":
    main()
