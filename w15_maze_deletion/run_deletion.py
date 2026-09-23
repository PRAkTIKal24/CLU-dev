"""CMAZE-2b exact-deletion rows, one seed.  Free replication: NO scored run is spent.

    python -m w15_maze_deletion.run_deletion --seed S --encoder cache|train --tag T --out DIR

What is measured, and on what
-----------------------------
ENVIRONMENT: **CMAZE-2b** -- the w14 fixed request layer (``w14_request_layer.task2.
build_site_v2``), banks ``bank_after_seed{0,1,2}.pkl``, GRID_REBASED budget words, the
CMAZE-2b encoder pool (``w14_read_verb.pool``, 270 sites), K=8, n_atoms=31, ridge=0.01,
ae_steps=12000 -- i.e. exactly the configuration of the CMAZE-2b rows the paper prints
(``w14_read_verb.leg`` defaults).  Nothing in ``w14_request_layer`` / ``w14_read_verb`` /
``w9_maze`` is edited; all of it is imported.

The deletable object is the STORE (``CovContentStore`` sufficient statistic S,t,n and the
atoms derived from it) plus the address book's retained Gram (A, Mx).  The encoder and
its frozen decoder are declared OUTSIDE the deletable object.

The rows use only the TRAIN items (identical for the val and test splits -- asserted),
so no held-out row is read by the exactness measurement.  The adaptation-cost
composition rows read VAL rows only; the test-leg composition is DECLARED NOT-RUN.

PRIMARY row: delete every item written under target=green (w9's banked row).
SUPPLEMENTARY rows (labelled as such): every other target word, one maze word, one
agent word, and a random 10 % of rows.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

import equinox as eqx
import jax
import numpy as np

from chlu.experiments.exp_w8_single_design import CovContentStore
from w9_maze import arms as A
from w9_maze import evaluate, splits
from w9_maze.display import delete_word_content, downdate_vs_refit
from w9_maze.encoder import AE, LATENT_D, encode_all, train_ae
from w9_maze.pipeline import Renderer, build_items
from w14_read_verb import leg as RV
from w14_read_verb.pipeline import requests_for, row_sets_for

ENV = "CMAZE-2b"
GREEN = 1
N_ATOMS, K, RIDGE, AE_STEPS = 31, 8, 0.01, 12000


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def file_sha(p):
    with open(p, "rb") as fh:
        return sha(fh.read())


def git_head():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=os.environ.get("CHLU_ROOT", "."),
                                       text=True).strip()
    except Exception as e:                                      # pragma: no cover
        return "unknown (%s)" % e


def refit_without(clu, Z_keep, reqs_keep):
    """The rival's route: REFIT the deletable object from the surviving rows.

    Same well partition (fixed at write time, so it is part of the object's address,
    not something the refit may re-choose), a fresh ``CovContentStore`` on the
    survivors, and the head's closed form re-accumulated and re-solved on them.
    """
    ref = copy.deepcopy(clu)
    t0 = time.perf_counter()
    ref.store = CovContentStore(Z_keep, clu.well_channels, clu.n_atoms, clu.dim)
    ref.fit_head(Z_keep, reqs_keep)
    return ref, time.perf_counter() - t0


def rel(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    return float(np.abs(a - b).max() / max(1.0, float(np.abs(b).max())))


def one_deletion(clu, Ztrain, reqs_train, drop, probe_reqs, label, primary):
    keep = ~drop
    ex = downdate_vs_refit(Ztrain, keep, clu.well_channels, clu.n_atoms, clu.dim)
    # the whole deletable object: store + head Gram, downdated vs refit-without
    dd = copy.deepcopy(clu)
    Zd = np.asarray(Ztrain, np.float32)[drop]
    reqs_d = [r for r, d in zip(reqs_train, drop) if d]
    reqs_k = [r for r, k in zip(reqs_train, keep) if k]
    t0 = time.perf_counter()
    rec = dd.delete_rows_exact(Zd, reqs_d)
    t_down = time.perf_counter() - t0
    ref, t_refit = refit_without(clu, np.asarray(Ztrain, np.float32)[keep], reqs_k)
    Rd = dd.read(probe_reqs)
    Rr = ref.read(probe_reqs)
    R0 = clu.read(probe_reqs)
    S_moved = max(rel(w0.S, wr.S) for w0, wr in zip(clu.store.stats, ref.store.stats))
    row = {
        "environment": ENV, "label": label, "primary": bool(primary),
        "n_rows_removed": int(drop.sum()), "n_rows_kept": int(keep.sum()),
        "statistic_max_rel_residual": ex["statistic_max_rel_residual"],
        "subspace_max_projector_residual": ex["subspace_max_projector_residual"],
        "head_gram_A_max_rel_residual": rel(dd.A, ref.A),
        "head_gram_Mx_max_rel_residual": rel(dd.Mx, ref.Mx),
        "read_max_rel_residual_on_probe": rel(Rd, Rr),
        "n_probe_requests": len(probe_reqs),
        # SANITY: the deletion must MOVE the object; a zero residual is only meaningful
        # beside a non-zero change.  (before-deletion vs refit-without)
        "sanity_read_change_by_deletion": rel(R0, Rr),
        "sanity_statistic_change_by_deletion": S_moved,
        "sanity_head_A_change_by_deletion": rel(clu.A, ref.A),
        "receipt": {k: v for k, v in rec.items() if k != "basis_change_shapes"},
        "refit": False, "passes_over_retained_data": 0, "items_kept": False,
        "seconds_downdate_store_plus_head": t_down,
        "seconds_refit_without_store_plus_head": t_refit,
    }
    return row


def composition(bank, render, reader, rows, use_cert):
    if not rows:
        return {"n": 0}
    Zh = reader.read([r for (r, _m, _i, _j) in rows])
    plans, scenes = render.plans(Zh)
    agg = evaluate.aggregate(evaluate.answers(bank, rows, plans, scenes,
                                              use_certificate=use_cert))
    return {k: v for k, v in agg.items() if k != "recs"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--encoder", choices=("cache", "train"), required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--banks", default=RV.BANKS)
    ap.add_argument("--pool", default=RV.POOL)
    ap.add_argument("--rival", action="store_true")
    ap.add_argument("--bl-steps", type=int, default=4000)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    t_all = time.time()
    s = a.seed
    import chlu
    res = {"environment": ENV, "seed": s, "tag": a.tag, "encoder_mode": a.encoder,
           "scored_runs_spent": 0, "chlu_file": chlu.__file__,
           "checkout": os.environ.get("CHLU_ROOT"), "git_head": git_head(),
           "python": sys.version.split()[0], "platform": platform.platform(),
           "jax": jax.__version__, "numpy": np.__version__, "equinox": eqx.__version__,
           "config": {"K": K, "n_atoms": N_ATOMS, "ridge": RIDGE, "ae_steps": AE_STEPS,
                      "grid": "rebased", "latent_d": LATENT_D},
           "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    # ---- the environment: CMAZE-2b bank + pool, loaded via w14_read_verb's own paths
    bank_path = os.path.join(a.banks, "bank_after_seed%d.pkl" % s)
    import pickle
    with open(bank_path, "rb") as fh:
        blob = pickle.load(fh)
    bank = blob["sites"]
    RV.POOL = a.pool
    pool = RV.load_pool()
    res["bank"] = {"path": bank_path, "sha256": file_sha(bank_path),
                   "which": blob["which"], "seed0": blob["seed0"], "n_sites": len(bank)}
    assert blob["which"] == "after", "not a CMAZE-2b bank"
    res["pool"] = {"dir": a.pool, "n_sites": len(pool)}
    budgets, interp = RV.GRID_REBASED[s]
    res["config"]["budgets"], res["config"]["interp"] = list(budgets), list(interp)

    val, test = splits.make_split(bank, s)
    assert {m: sorted(v) for m, v in val.train_pairs.items()} == \
           {m: sorted(v) for m, v in test.train_pairs.items()}, "train set differs"
    res["train_set_identical_val_test"] = True
    train_rows = requests_for(bank, val.train_pairs, budgets)
    reqs_train = [r for (r, _m, _i, _j) in train_rows]
    res["n_train_requests"] = len(train_rows)

    # ---- the frozen encoder (OUTSIDE the deletable object)
    Xtr, itr, _ = build_items(bank, train_rows)
    pool_rows = [(None, m, i, j) for m, st in enumerate(pool) for (i, j) in st.pairs]
    Xp, _ip, _u = build_items(pool, pool_rows)
    X_ae = np.concatenate([Xtr, Xp], axis=0)
    res["ae_train_items"] = {"eval_bank_train": int(Xtr.shape[0]),
                             "pool": int(Xp.shape[0])}
    t0 = time.time()
    if a.encoder == "cache":
        cache = os.path.join(RV.OUT, "ae", "ae_seed%d_rebased_%d_%d.eqx"
                             % (s, AE_STEPS, len(pool)))
        ae = eqx.tree_deserialise_leaves(cache, AE(jax.random.PRNGKey(s),
                                                   int(X_ae.shape[-1])))
    else:
        ae, _hist = train_ae(X_ae, jax.random.PRNGKey(s), steps=AE_STEPS, verbose=True)
        cache = os.path.join(a.out, "ae_seed%d_rebased_%d_%d.eqx"
                             % (s, AE_STEPS, len(pool)))
        eqx.tree_serialise_leaves(cache, ae)
    res["encoder"] = {"path": cache, "sha256": file_sha(cache),
                      "seconds": time.time() - t0}
    print("seed %d encoder %s %s (%.0fs)" % (s, a.encoder, cache, time.time() - t0),
          flush=True)

    Ztr = encode_all(ae, Xtr)
    Zraw = Ztr[itr]
    mu = Zraw.mean(axis=0)
    sd = Zraw.std(axis=0) + 1e-6
    Ztrain = ((Zraw - mu) / sd).astype(np.float32)
    render = Renderer(ae, mu, sd)
    res["Ztrain_sha256"] = sha(np.ascontiguousarray(Ztrain).tobytes())
    grids = A.make_grids(len(bank), 6, 6, budgets)
    t0 = time.perf_counter()
    clu = A.CluRead(Ztrain, reqs_train, grids, K=K, n_atoms=N_ATOMS, ridge=RIDGE,
                    seed=s)
    res["seconds_initial_fit_store_plus_head"] = time.perf_counter() - t0
    led = clu.ledger()
    res["ledger"] = led

    # ---- the deletion rows
    keys = np.array([[r.m, r.a, r.t, r.b] for r in reqs_train], np.int64)
    rs_val = row_sets_for(bank, val, budgets, interp)
    grounded_val = rs_val["unseen_pair"] + rs_val["nway"]
    probe = reqs_train + [r for (r, _m, _i, _j) in grounded_val]
    rows = []
    rows.append(one_deletion(clu, Ztrain, reqs_train, keys[:, 2] == GREEN, probe,
                             "target=green (w9's banked row)", True))
    for t in range(6):
        if t != GREEN and (keys[:, 2] == t).any():
            rows.append(one_deletion(clu, Ztrain, reqs_train, keys[:, 2] == t, probe,
                                     "target=%d" % t, False))
    rows.append(one_deletion(clu, Ztrain, reqs_train, keys[:, 0] == 0, probe,
                             "maze=0", False))
    rows.append(one_deletion(clu, Ztrain, reqs_train, keys[:, 1] == 0, probe,
                             "agent=0", False))
    rng = np.random.default_rng(1000 + s)
    rnd = rng.random(len(reqs_train)) < 0.10
    rows.append(one_deletion(clu, Ztrain, reqs_train, rnd, probe,
                             "random 10%% of rows (rng %d)" % (1000 + s), False))
    res["deletion_rows"] = rows
    p = rows[0]
    print("seed %d PRIMARY %s: statistic %.3e projector %.3e | head A %.3e Mx %.3e "
          "read %.3e | removed %d kept %d | downdate %.3fs refit %.3fs"
          % (s, ENV, p["statistic_max_rel_residual"],
             p["subspace_max_projector_residual"], p["head_gram_A_max_rel_residual"],
             p["head_gram_Mx_max_rel_residual"], p["read_max_rel_residual_on_probe"],
             p["n_rows_removed"], p["n_rows_kept"],
             p["seconds_downdate_store_plus_head"],
             p["seconds_refit_without_store_plus_head"]), flush=True)

    # ---- adaptation cost (VAL rows only; the test-leg composition is NOT run)
    green_rows = [r for r in grounded_val if r[3] == GREEN]
    other_rows = [r for r in grounded_val if r[3] != GREEN]
    pre_g = composition(bank, render, clu, green_rows, True)
    pre_o = composition(bank, render, clu, other_rows, False)
    adapt = {"split": "val", "n_green_rows": len(green_rows),
             "n_other_rows": len(other_rows),
             "green_before": pre_g, "other_before": pre_o,
             "test_leg": "DECLARED NOT-RUN: reading test rows is a scored act; this "
                         "spoke spends zero scored runs"}
    for mode, exact in (("carry_W_R (w9 as pre-registered)", False),
                        ("exact_head_downdate (w9 Amendment 1)", True)):
        c = copy.deepcopy(clu)
        t0 = time.perf_counter()
        rec = delete_word_content(c, Ztrain, reqs_train, "target", GREEN,
                                  exact_head=exact)
        dt = time.perf_counter() - t0
        post_g = composition(bank, render, c, green_rows, True)
        post_o = composition(bank, render, c, other_rows, False)
        head_changed = int(sum(int((np.asarray(W1) != np.asarray(W0)).sum())
                               for W0, W1 in zip(clu.heads, c.heads)))
        atoms_changed = int(sum(int((np.asarray(M1) != np.asarray(M0)).sum())
                                for M0, M1 in zip(clu.store.modes, c.store.modes)))
        adapt[mode] = {
            "seconds": dt,
            "address_book_floats_changed": head_changed,
            "address_book_floats_total": int(led["head_floats"]),
            "atom_floats_changed": atoms_changed,
            "atom_floats_total": int(led["store_atom_floats"]),
            "address_book_columns_cleared": rec["address_book_columns_cleared"],
            "data_touched": "only the deleted rows handed in; 0 passes over retained data",
            "retrained": False,
            "green_after": post_g, "other_after": post_o,
            "validity_delta_other_targets":
                float(post_o["validity"] - pre_o["validity"]) if post_o.get("n") else None,
            "refusal_green_before_after": [pre_g.get("refusal"), post_g.get("refusal")],
        }
        print("seed %d adapt %-38s %.3fs  other validity %.4f -> %.4f  green refusal "
              "%.4f -> %.4f  head floats changed %d/%d"
              % (s, mode, dt, pre_o.get("validity", float("nan")),
                 post_o.get("validity", float("nan")),
                 pre_g.get("refusal", float("nan")), post_g.get("refusal", float("nan")),
                 head_changed, led["head_floats"]), flush=True)
    res["adaptation_cost"] = adapt

    # ---- the RIVAL's deletion: retrain-without, cost printed (never a downdate)
    if a.rival:
        from chlu.core.gen_request_head import request_design
        from w9_maze.baselines import LearnedArm, token_inputs
        main_spec = A.HeadSpec([int(g.size) for g in grids], pair_block=False)
        keep = keys[:, 2] != GREEN
        rk = [r for r, k in zip(reqs_train, keep) if k]
        Phi_k, _ = request_design(np.array([r.values() for r in rk], np.float64),
                                  grids, main_spec)
        Tok_k = token_inputs(Phi_k, main_spec.layout)
        arm = LearnedArm("transformer", jax.random.PRNGKey(s * 101 + 37),
                         Phi_k.shape[1], LATENT_D, led["read_floats"],
                         seq_in=Tok_k.shape[2])
        t0 = time.perf_counter()
        arm.fit(Tok_k, Ztrain[keep], steps=a.bl_steps, seed=s, verbose=False)
        dt = time.perf_counter() - t0
        res["rival_retrain_without"] = {
            "arm": "transformer (matched read-path params)", "params": arm.n_params,
            "matched": bool(arm.matched), "target_params": int(led["read_floats"]),
            "steps": a.bl_steps, "batch": 64,
            "row_reads_of_retained_data": int(a.bl_steps * 64),
            "n_rows_retained_and_required": int(keep.sum()),
            "seconds": dt, "items_kept": True,
            "note": "the rival can delete only by retraining on the kept rows, so it "
                    "must KEEP them; encoder is frozen and shared, cost excluded on "
                    "both sides"}
        print("seed %d RIVAL transformer retrain-without: %.1fs, %d steps, %d row reads"
              % (s, dt, a.bl_steps, a.bl_steps * 64), flush=True)
    else:
        res["rival_retrain_without"] = {"DECLARED_NOT_RUN": "--rival not given"}
    res["seconds"] = time.time() - t_all
    res["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    path = os.path.join(a.out, "deletion_%s_seed%d.json" % (a.tag, s))

    def d(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    with open(path, "w") as fh:
        json.dump(res, fh, indent=1, default=d)
    print("wrote", path, "%.0fs" % res["seconds"], flush=True)


if __name__ == "__main__":
    main()
