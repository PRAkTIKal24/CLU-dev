"""CMAZE-T TRAINING-BUDGET DIAGNOSTIC (VAL only; never a matched rival row, never a bar).

    python scripts/w16_cmaze_t_rivaldiag.py <seed> <out_dir> [mult]

The CMAZE-O C-field rivals, re-trained 10x longer, overturned that task's rival margin
(matched MLP 0.99 vs CLU 0.70 on VAL unseen_nway).  The same objection is put to the
CMAZE-T Config-A1 win here: the SAME three matched-parameter rivals (target = A1's read
floats), and the SAME trunks at 10x the read-path budget, each trained for ``mult`` x the
recipe's 8,000 gradient steps, SAME observation, SAME frozen encoder, SAME rows.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import chlu.exp_w15_cmaze_t as T                                    # noqa: E402
import chlu.exp_w15_cmaze_t_arms as A                               # noqa: E402
import w15_cmaze_t_run as R                                         # noqa: E402
import w16_cmaze_t as W                                             # noqa: E402

K = R.K_COLOURS


def main(seed, out, mult=10):
    rng = np.random.default_rng(10_000 + seed)
    base, ev_base, (tr_insts, tr_rows, _n, _m), (va_insts, va_rows, _vn, _vm) = \
        W.banks(seed, "val")
    ho = T.make_holdout(K, seed=seed)
    train_rows = T.split_rows(tr_rows, ho, K).get("seen", [])
    va = W.cap_split(T.split_rows(va_rows, ho, K), seed)
    enc, _ = A.train_state_encoder(tr_insts, steps=1500, seed=seed, verbose=False,
                                   restarts=3, accept_bce=0.010)
    trf, vaf = A.feature_maps(enc, tr_insts), A.feature_maps(enc, va_insts)
    from w9_harness.features import global_summary
    trg, vag = [global_summary(S) for S in trf], [global_summary(S) for S in vaf]
    # A1's read floats (latent 128, K 2, 6 atoms) -- from its own VAL JSON
    a1 = json.load(open(os.path.join(out, "A1_val_seed%d.json" % seed)))
    target = int(a1["clu_ledger"]["read_floats"])
    od = A.obs_dim(int(trf[0].shape[0]), K)
    Xs, Ys, Ms = R.single_sequences(train_rows, tr_insts, trf, trg)
    Xt, Yt, Mt = R.trajectory_sequences(tr_insts, trf, trg, rng, per_maze=6)
    steps = 8000 * mult
    res = {"seed": seed, "steps": steps, "recipe_steps": 8000, "target_floats": target,
           "WHAT_THIS_IS": __doc__, "arms": {}}
    for label, tgt in (("matched", target), ("10x read floats", 10 * target)):
        widths = A.match_widths(od, T.code_dim(K), tgt, tmax=R.SEQ_T,
                                **({"hi": 400} if tgt > target else {}))
        for kind, (w, p, kw) in widths.items():
            pol, rep = R.train_rival(kind, w, kw, Xs, Ys, Ms, Xt, Yt, Mt, seed, steps,
                                     "%s %s" % (label, kind))
            e = {"params": int(p), "final_loss": rep["final_loss"], "rows": {}}
            for row in ("seen", "pair", "nway", "value"):
                rs = va[row]
                O = A.build_obs(rs, va_insts, vaf, vag)
                e["rows"][row] = W.sc(va_insts, rs, A.policy_predict(pol, O, K))
            name = "diagnostic: %s at %s, %dx steps" % (kind, label, mult)
            res["arms"][name] = e
            print("[tdiag %d] %-45s loss %.4f  seen %.4f pair %.4f nway %.4f value %.4f"
                  % (seed, name, rep["final_loss"], e["rows"]["seen"], e["rows"]["pair"],
                     e["rows"]["nway"], e["rows"]["value"]), flush=True)
            json.dump(res, open(os.path.join(out, "A1_rivaldiag_val_seed%d.json" % seed),
                                "w"), indent=1, default=R._json_default)


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 10)
