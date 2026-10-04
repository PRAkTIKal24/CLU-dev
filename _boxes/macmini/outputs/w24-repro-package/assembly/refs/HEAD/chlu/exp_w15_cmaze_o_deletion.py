"""CMAZE-O's deletion rows: VALUE deletion and the CONJUNCTION deletion.

The two deletions the task file names
--------------------------------------
* **value**       -- delete every training state carrying colour ``c`` in ANY slot.
* **conjunction** -- delete colour ``c`` **and** tinted floor: every training state whose
  *tinted-floor* colour is ``c``.  The conjunction is narrower than the value deletion
  by construction, and both counts are printed.

What is deleted, and what is not (COMMON.md deletion language)
---------------------------------------------------------------
* **Store-scoped exactness** is the claim, and it is checked, not asserted: the store
  after deletion is compared, element by element, against a store **fitted from scratch
  without those rows** on the SAME partition.  The reported number is the maximum
  absolute difference over the retained statistic ``S, t, n`` and over the deployed
  atoms.
* The **frozen encoder and decoder are declared OUTSIDE the deletable object.**  They
  were trained by reconstruction on the whole training pool, they are shared by every
  arm, and deleting rows from the store does not and cannot remove them from the
  encoder.  The rival's retrain-without keeps the same frozen encoder, so the comparison
  is like for like.
* **Post-deletion adaptation is light and data-free** -- ⛔ never "no changes".  The head
  is re-solved in closed form against the statistics that were already retained
  (``A``, ``Mx`` downdated by the same subtraction).  No pool row is read, no item is
  kept, nothing is passed over.  Its wall-clock is printed.
* The **rival deletes only by retrain-without**, and its wall-clock is printed beside the
  CLU's.  The **as-trained** rival -- which has not deleted anything -- is printed beside
  both, because that is the row that shows what deletion costs.

The rival used for the deletion row is the matched-parameter **MLP**, declared here and
not selected: it is the cheapest of the four to retrain, and retraining is the whole cost
being measured.
"""

from __future__ import annotations

import time

import jax
import numpy as np

from chlu.exp_w15_cmaze_o_arms import (OCluRead, ORival, main_effect_design,
                                       make_grids)
from chlu.exp_w15_cmaze_o_encoder import decode_all
from chlu.exp_w15_cmaze_o_env import decode_state, judge_state, render_state

#: the colour whose states are deleted.  Pinned, not selected.
DELETE_COLOUR = 2


def _mask_value(reqs, c):
    return np.array([any(int(w) == c for w in r.words()[1:5]) for r in reqs], bool)


def _mask_conjunction(reqs, c):
    return np.array([int(r.ct) == c for r in reqs], bool)


def _store_gap(a, b) -> dict:
    """Max abs difference between two stores, well by well.  Exact arithmetic report."""
    gS = gt = gn = gM = 0.0
    for wa, wb in zip(a.store.stats, b.store.stats):
        gS = max(gS, float(np.max(np.abs(np.asarray(wa.S) - np.asarray(wb.S)))))
        gt = max(gt, float(np.max(np.abs(np.asarray(wa.t) - np.asarray(wb.t)))))
        gn = max(gn, abs(int(wa.n) - int(wb.n)))
    for Ma, Mb in zip(a.store.modes, b.store.modes):
        gM = max(gM, float(np.max(np.abs(np.asarray(Ma, np.float64)
                                         - np.asarray(Mb, np.float64)))))
    scale = max(1.0, float(np.max([np.max(np.abs(np.asarray(w.S)))
                                   for w in b.store.stats])))
    return {"max_abs_diff_S": gS, "max_abs_diff_t": gt, "max_abs_diff_n": int(gn),
            "max_abs_diff_atoms": gM, "S_scale": scale,
            "relative_S_gap": gS / scale,
            "note": ("store after deletion vs a store fitted from scratch WITHOUT the "
                     "deleted rows, on the SAME partition; atoms are eigenvectors and "
                     "are sign-ambiguous, so the atom gap is reported but the claim "
                     "rests on S, t, n")}


def _eval(ae, bank, reqs, Zr):
    LG = decode_all(ae, np.asarray(Zr, np.float32))
    ok = []
    for i, r in enumerate(reqs):
        j = judge_state(decode_state(LG[i]), render_state(bank[r.layout], r))
        ok.append(j["valid"])
    return {"n": len(reqs), "valid": float(np.mean(ok)) if ok else float("nan")}


def deletion_study(bank, pool, ae, Z, seed, config_a, rival_steps=2500,
                   rows=("seen", "unseen_2way", "unseen_nway"), verbose=True) -> dict:
    """Both deletions, the CLU and the rival, with every cost printed."""
    log = (lambda *a: print("[seed %d del]" % seed, *a, flush=True)) if verbose \
        else (lambda *a: None)
    grids = make_grids(len(bank))
    train = pool.train
    c = DELETE_COLOUR
    Phi_main, mspec = main_effect_design(train, grids)

    # the object as trained, once -- and its partition, which every comparison reuses
    clu_as_trained = OCluRead(Z, train, grids, K=config_a["K"],
                              n_atoms=config_a["n_atoms"], ridge=config_a["ridge"],
                              pair_block=config_a["pair_block"], seed=seed)
    wells = clu_as_trained.well_channels
    target = int(clu_as_trained.ledger()["read_floats"])

    t = time.time()
    mlp_as_trained = ORival("mlp", jax.random.PRNGKey(1000 + seed),
                            Phi_main.shape[1], Z.shape[1], target)
    mlp_as_trained.fit(Phi_main, Z, steps=rival_steps, seed=seed)
    mlp_train_secs = time.time() - t
    log("as-trained MLP fitted in %.0fs (%d params)"
        % (mlp_train_secs, mlp_as_trained.n_params))

    out = {"delete_colour": c, "n_train": len(train),
           "rival_used": "matched-parameter MLP",
           "as_trained_rival_fit_seconds": mlp_train_secs,
           "frozen_encoder_decoder": ("declared OUTSIDE the deletable object; trained "
                                      "on the whole training pool and NOT re-trained "
                                      "for either arm after deletion"),
           "deletions": {}}

    for kind, mask_fn in (("value", _mask_value), ("conjunction", _mask_conjunction)):
        m = mask_fn(train, c)
        keep = ~m
        del_reqs = [r for r, d in zip(train, m) if d]
        keep_reqs = [r for r, d in zip(train, m) if not d]
        log("%s deletion: %d of %d training states" % (kind, int(m.sum()), len(train)))

        # -- the CLU: exact downdate + a data-free closed-form re-solve
        clu = OCluRead(Z, train, grids, K=config_a["K"], n_atoms=config_a["n_atoms"],
                       ridge=config_a["ridge"], pair_block=config_a["pair_block"],
                       seed=seed, well_channels=wells)
        t = time.time()
        receipt = clu.delete_rows_exact(Z[m], del_reqs)
        clu_adapt_secs = time.time() - t

        # -- the reference object: fitted from scratch WITHOUT those rows
        t = time.time()
        clu_without = OCluRead(Z[keep], keep_reqs, grids, K=config_a["K"],
                               n_atoms=config_a["n_atoms"], ridge=config_a["ridge"],
                               pair_block=config_a["pair_block"], seed=seed,
                               well_channels=wells)
        clu_refit_secs = time.time() - t
        gap = _store_gap(clu, clu_without)
        log("  store-scoped gap: |dS|=%.3e |dt|=%.3e |dn|=%d rel=%.3e"
            % (gap["max_abs_diff_S"], gap["max_abs_diff_t"], gap["max_abs_diff_n"],
               gap["relative_S_gap"]))

        # -- the rival: retrain-without, the only deletion it has
        t = time.time()
        mlp_without = ORival("mlp", jax.random.PRNGKey(1000 + seed),
                             Phi_main.shape[1], Z.shape[1], target)
        mlp_without.fit(Phi_main[keep], Z[keep], steps=rival_steps, seed=seed)
        mlp_retrain_secs = time.time() - t
        log("  rival retrain-without: %.0fs (CLU adaptation %.2fs)"
            % (mlp_retrain_secs, clu_adapt_secs))

        # -- post-deletion composition ON THE OTHER COLOURS
        tab = {}
        for row in rows:
            reqs_all = pool.rows[row]
            others = [r for r in reqs_all if not mask_fn([r], c)[0]]
            deleted_side = [r for r in reqs_all if mask_fn([r], c)[0]]
            for label, rr in (("other_colours", others), ("deleted_colour",
                                                          deleted_side)):
                if not rr:
                    tab.setdefault(row, {})[label] = {"n": 0}
                    continue
                Pm, _ = main_effect_design(rr, grids)
                tab.setdefault(row, {})[label] = {
                    "CLU_as_trained": _eval(ae, bank, rr, clu_as_trained.read(rr)),
                    "CLU_after_deletion": _eval(ae, bank, rr, clu.read(rr)),
                    "CLU_refit_without": _eval(ae, bank, rr, clu_without.read(rr)),
                    "MLP_as_trained": _eval(ae, bank, rr, mlp_as_trained.read(Pm)),
                    "MLP_retrain_without": _eval(ae, bank, rr, mlp_without.read(Pm)),
                }

        out["deletions"][kind] = {
            "n_deleted": int(m.sum()),
            "fraction_of_train": float(m.mean()),
            "store_scoped_exactness": gap,
            "receipt": {k: (v if not isinstance(v, np.ndarray) else v.tolist())
                        for k, v in receipt.items()},
            "adaptation": {
                "CLU_adaptation_seconds": clu_adapt_secs,
                "CLU_adaptation_is_data_free": True,
                "CLU_adaptation_mechanism": (
                    "S,t,n and the head's A,Mx are downdated by the deleted rows' own "
                    "contribution and the same closed form is re-solved; no pool row is "
                    "read and no item is kept -- light and data-free, NOT 'no changes'"),
                "CLU_refit_from_scratch_seconds": clu_refit_secs,
                "rival_retrain_without_seconds": mlp_retrain_secs,
                "speedup_vs_rival_retrain": (mlp_retrain_secs / max(clu_adapt_secs,
                                                                    1e-9)),
            },
            "post_deletion_composition": tab,
        }
    out["tolerance_used"] = None
    return out
