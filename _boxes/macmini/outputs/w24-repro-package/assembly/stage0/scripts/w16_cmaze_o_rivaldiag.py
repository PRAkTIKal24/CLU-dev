"""CMAZE-O C-field TRAINING-BUDGET DIAGNOSTIC (VAL only, never a matched rival row).

    python scripts/w16_cmaze_o_rivaldiag.py <seed> <cache_dir> <out_dir> [steps]

The matched rivals under C-field are trained with the w15 recipe (3,000 steps).  The
sequence rivals then read ~0.003 on unseen_nway, which invites the objection that they
were under-trained rather than out-composed.  This re-trains the SAME four rivals, at the
SAME matched parameter count and the SAME field output, for 10x the gradient steps, and
prints their VAL rows.  ⛔ NOT a rival row, NOT part of any bar: a diagnostic.
"""
import json
import os
import sys
import time

import jax
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import w16_cmaze_o as W                                            # noqa: E402
from chlu.exp_w15_cmaze_o_arms import (OCluRead, ORival, continuous_inputs,  # noqa: E402
                                       main_effect_design, make_grids, token_inputs)


def main(seed, cache, out, steps=30000):
    d, ae, _ = W.load(seed, cache)
    bank, pool = d["bank"], d["pool"]
    grids = make_grids(len(bank))
    Zf = W._items(bank, pool.train).reshape((len(pool.train), -1)).astype(np.float32)
    clu = OCluRead(Zf, pool.train, grids, K=8, n_atoms=64, ridge=1.0, pair_block=True,
                   seed=seed)
    target = int(clu.ledger()["read_floats"])
    Phi, mspec = main_effect_design(pool.train, grids)
    Tok, Cont = token_inputs(Phi, mspec), continuous_inputs(pool.train, grids)
    res = {"seed": seed, "steps": steps, "recipe_steps": W.KW["rival_steps"],
           "target_floats": target, "WHAT_THIS_IS": __doc__, "rivals": {}}
    for kind in ORival.KINDS:
        key = jax.random.PRNGKey(1000 + seed)
        if kind == "mlp":
            rv, X = ORival(kind, key, Phi.shape[1], Zf.shape[1], target), Phi
        elif kind == "cont_mlp":
            rv, X = ORival(kind, key, Cont.shape[1], Zf.shape[1], target), Cont
        else:
            rv, X = ORival(kind, key, None, Zf.shape[1], target, seq_in=Tok.shape[2],
                           n_tokens=Tok.shape[1]), Tok
        t = time.time()
        rv.fit(X, Zf, steps=steps, seed=seed)
        e = {"params": rv.n_params, "fit_seconds": time.time() - t,
             "heldin_val_mse": rv.val_mse, "rows": {}}
        for row in ("seen", "unseen_2way", "unseen_nway"):
            reqs = pool.rows[row]
            Pm, _ = main_effect_design(reqs, grids)
            xin = Pm if kind == "mlp" else (continuous_inputs(reqs, grids) if kind == "cont_mlp"
                                            else token_inputs(Pm, mspec))
            e["rows"][row] = W.judge_field(bank, reqs, rv.read(xin))
        res["rivals"][kind] = e
        print("[diag %d] %-12s %d steps  mse %.5f  seen %.4f 2way %.4f nway %.4f [%.0fs]"
              % (seed, kind, steps, rv.val_mse, e["rows"]["seen"]["valid"],
                 e["rows"]["unseen_2way"]["valid"], e["rows"]["unseen_nway"]["valid"],
                 e["fit_seconds"]), flush=True)
        json.dump(res, open(os.path.join(out, "C_rivaldiag_val_seed%d.json" % seed), "w"),
                  indent=1, default=str)


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2], sys.argv[3],
         int(sys.argv[4]) if len(sys.argv) > 4 else 30000)
