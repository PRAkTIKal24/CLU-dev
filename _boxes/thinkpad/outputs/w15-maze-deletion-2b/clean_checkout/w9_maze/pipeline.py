"""One end-to-end run: encoder, arms, rows, controls, columns.

The order is fixed and is the order the charter requires:

1. the bank and the split -- no arm exists yet;
2. the SHARED encoder, trained by reconstruction on the TRAIN items only;
3. the CEILING -- encode, decode, execute, judge, on training items: no arm can beat
   it, and every row is read against it;
4. the arms -- the CLU, its charter-C controls, the matched-bytes lookup floor, and
   the learned baselines at matched parameters;
5. the rows, each broken out by whether rule B bends that route;
6. the environment's own columns on THIS split.
"""

from __future__ import annotations

import json
import os
import time

import equinox as eqx
import jax
import numpy as np

from w9_maze import arms as A
from w9_maze import envcols, evaluate, splits
from w9_maze.baselines import LearnedArm, token_inputs
from w9_maze.encoder import AE, LATENT_D, decode_plans, encode_all, train_ae
from w9_maze.render import item_tensor, plan_fidelity
from w9_maze.task import BUDGET_GRID, BUDGET_INTERP

ROWS = ("seen", "unseen_pair", "nway", "budget_interp")


def build_items(bank, rows):
    """Item tensors for a list of ``(request, m, i, j)``, deduplicated by pair."""
    uniq, index = {}, []
    for (_r, m, i, j) in rows:
        k = (m, i, j)
        if k not in uniq:
            uniq[k] = len(uniq)
        index.append(uniq[k])
    X = np.zeros((len(uniq), 15, 15, 75), np.float32)
    for (m, i, j), n in uniq.items():
        X[n] = item_tensor(bank[m], i, j, bank[m].route[(i, j)])
    return X, np.asarray(index, np.int64), uniq


def ceiling(bank, rows, ae, X, index, tag=""):
    """encode -> decode -> execute -> exact judge.  The bound on every arm."""
    Z = encode_all(ae, X)
    plans, scenes = decode_plans(ae, Z)
    recs = evaluate.answers(bank, rows, plans[index], scenes[index])
    agg = evaluate.aggregate(recs)
    agg["tag"] = tag
    return agg, Z


def representation_ceiling(bank, rows):
    """The plan map itself, before any network: draw the optimum, execute the drawing."""
    ok, sc, coll = 0, [], 0
    for (req, m, i, j) in rows:
        o, s, c, _r = plan_fidelity(bank[m], i, j, req.b)
        ok += bool(o)
        sc.append(s)
        coll += c
    n = max(1, len(rows))
    return {"n": len(rows), "validity": ok / n, "score": float(np.mean(sc)),
            "collisions": int(coll)}


class Renderer:
    """The frozen decoder plus the latent standardisation, shared by every arm.

    Every arm predicts a STANDARDISED latent and is rendered through exactly this
    object, so the standardisation is a constant of the comparison.  It exists because
    the raw encoder latents have a per-dimension scale of order 50: the closed-form
    ridge is indifferent to that, but an Adam-trained baseline is not, and the first
    DEV leg measured the learned arms fitting targets at MSE ~1900 and scoring 0.0000
    across the board.  Standardising is the fix that makes the floor a real floor.
    """

    def __init__(self, ae, mu, sd):
        self.ae, self.mu, self.sd = ae, mu, sd

    def plans(self, Zs):
        return decode_plans(self.ae, np.asarray(Zs, np.float32) * self.sd + self.mu)


def run(bank, seed, n_atoms=8, K=8, ridge=1.0, ae_steps=6000, bl_steps=4000,
        which="val", verbose=True, do_baselines=True, do_controls=True,
        pool_sites=None, ae_cache=None, lowrank_atoms=None):
    t0 = time.time()
    val, test = splits.make_split(bank, seed)
    split = val if which == "val" else test
    M = len(bank)
    out = {"seed": int(seed), "which": which, "n_mazes": M, "n_atoms": int(n_atoms),
           "K": int(K), "ridge": float(ridge), "ae_steps": int(ae_steps),
           "bl_steps": int(bl_steps), "latent_d": LATENT_D}

    train_rows = splits.train_items(bank, split)
    row_sets = {r: splits.row_requests(bank, split, r) for r in ROWS}
    out["n_train_requests"] = len(train_rows)
    out["n_rows"] = {r: len(v) for r, v in row_sets.items()}
    if verbose:
        print("  train requests %d; rows %s" % (len(train_rows), out["n_rows"]),
              flush=True)

    # -- 2. the shared encoder ---------------------------------------------
    #: trained by reconstruction on the evaluation bank's TRAIN items plus the ENCODER
    #: POOL -- mazes from a disjoint seed range.  No held-out item is ever seen.
    Xtr, itr, _ = build_items(bank, train_rows)
    X_ae = Xtr
    if pool_sites:
        pool_rows = [(None, m, i, j) for m, st in enumerate(pool_sites)
                     for (i, j) in st.pairs]
        Xp, _ip, _u = build_items(pool_sites, pool_rows)
        X_ae = np.concatenate([Xtr, Xp], axis=0)
    out["ae_train_items"] = {"eval_bank_train": int(Xtr.shape[0]),
                             "pool": int(X_ae.shape[0] - Xtr.shape[0]),
                             "pool_sites": len(pool_sites) if pool_sites else 0}
    if verbose:
        print("  encoder trained on %d items (%d from the bank's train split, %d pool)"
              % (X_ae.shape[0], Xtr.shape[0], X_ae.shape[0] - Xtr.shape[0]), flush=True)
    #: the shared encoder is CACHED by (seed, steps, item count).  It is a constant of
    #: every arm comparison, so training it once and reusing it is not a shortcut --
    #: it is what makes a store-configuration sweep a sweep of the STORE.
    hist = []
    if ae_cache and os.path.exists(ae_cache):
        ae = eqx.tree_deserialise_leaves(
            ae_cache, AE(jax.random.PRNGKey(seed), int(X_ae.shape[-1])))
        if verbose:
            print("  encoder loaded from %s" % ae_cache, flush=True)
    else:
        ae, hist = train_ae(X_ae, jax.random.PRNGKey(seed), steps=ae_steps,
                            verbose=verbose)
        if ae_cache:
            eqx.tree_serialise_leaves(ae_cache, ae)
            if verbose:
                print("  encoder cached to %s" % ae_cache, flush=True)
    out["ae_loss_history"] = hist
    out["ae_cache"] = ae_cache

    # -- 3. the ceilings ----------------------------------------------------
    out["representation_ceiling"] = {r: representation_ceiling(bank, row_sets[r])
                                     for r in ROWS}
    ceil_tr, Ztr = ceiling(bank, train_rows, ae, Xtr, itr, "train")
    out["pipeline_ceiling_train"] = ceil_tr
    ceil_rows, zrows = {}, {}
    for r in ROWS:
        Xr, ir, _ = build_items(bank, row_sets[r])
        c, zr = ceiling(bank, row_sets[r], ae, Xr, ir, r)
        ceil_rows[r] = c
        zrows[r] = zr[ir]
    out["pipeline_ceiling_rows"] = ceil_rows
    if verbose:
        print("  ceiling: representation %s / pipeline train %.4f"
              % ({r: round(out["representation_ceiling"][r]["validity"], 4)
                  for r in ROWS}, ceil_tr["validity"]), flush=True)

    # -- 4. the arms --------------------------------------------------------
    Zraw = Ztr[itr]                                      # one latent per request
    mu = Zraw.mean(axis=0)
    sd = Zraw.std(axis=0) + 1e-6
    Ztrain = ((Zraw - mu) / sd).astype(np.float32)
    render = Renderer(ae, mu, sd)
    out["latent_scale"] = {"raw_abs_mean": float(np.abs(Zraw).mean()),
                           "raw_sd_mean": float(sd.mean())}
    #: what the STORE's span can render at all: project a true latent onto the
    #: deployed atom span, decode, execute, judge.  A rank-limited store cannot beat
    #: this, and it is the number that says whether the atom budget is the binding
    #: constraint rather than the read.
    reqs_train = [r for (r, _m, _i, _j) in train_rows]
    grids = A.make_grids(M, 6, 6, BUDGET_GRID)
    clu = A.CluRead(Ztrain, reqs_train, grids, K=K, n_atoms=n_atoms, ridge=ridge,
                    seed=seed)
    led = clu.ledger()
    out["ledger"] = led
    out["request_only_assertion"] = clu.assert_request_only(reqs_train)
    target_params = led["read_floats"]

    #: THE STORE-SPAN CEILING.  Project a TRUE latent onto the deployed atom span and
    #: render it: a rank-limited store cannot do better than this, whatever its address
    #: book.  If this is far below the pipeline ceiling, the binding constraint is the
    #: atom budget, not the read, and the row must be read that way.
    out["span_ceiling_rows"] = {}
    for r in ROWS:
        Zs_r = (zrows[r] - mu) / sd
        Zp = np.zeros_like(Zs_r)
        for ch, Mk in zip(clu.well_channels, clu.store.modes):
            U = np.linalg.qr(np.asarray(Mk, np.float64).T)[0]
            Zp[:, ch] = np.asarray((Zs_r[:, ch] @ U) @ U.T, np.float32)
        pl, sc = render.plans(Zp)
        out["span_ceiling_rows"][r] = evaluate.aggregate(
            evaluate.answers(bank, row_sets[r], pl, sc))
    if verbose:
        print("  span ceiling (true latent projected on the atoms): %s"
              % {r: round(v["validity"], 4)
                 for r, v in out["span_ceiling_rows"].items()}, flush=True)

    readers = {"clu": clu}
    #: the second, NON-SATURATED store, pre-registered beside the val-selected one.
    #: The val sweep showed that span fraction == saturation ratio on this content, so
    #: the configuration that can render a maze route at all sits at rank/d_k = 0.94.
    #: This arm is the charter-comfortable store, reported beside it and never instead.
    if lowrank_atoms:
        readers["clu_lowrank"] = A.CluRead(Ztrain, reqs_train, grids, K=K,
                                           n_atoms=int(lowrank_atoms), ridge=ridge,
                                           seed=seed)
        out["ledger_lowrank"] = readers["clu_lowrank"].ledger()
    if do_controls:
        for ctl in ("landscape_removed", "landscape_shuffled", "random_atoms"):
            readers[ctl] = A.CluRead(Ztrain, reqs_train, grids, K=K, n_atoms=n_atoms,
                                     ridge=ridge, control=ctl, seed=seed)
        readers["clu_nopair"] = A.CluRead(Ztrain, reqs_train, grids, K=K,
                                          n_atoms=n_atoms, ridge=ridge,
                                          pair_block=False, seed=seed)
    readers["lookup_matched"] = A.LookupRead(Ztrain, reqs_train,
                                             budget_floats=target_params, seed=seed)
    readers["lookup_full"] = A.LookupRead(Ztrain, reqs_train, seed=seed)
    out["lookup_stored"] = {"matched": readers["lookup_matched"].n_stored,
                            "matched_floats": readers["lookup_matched"].floats,
                            "full": readers["lookup_full"].n_stored,
                            "full_floats": readers["lookup_full"].floats}

    # -- the learned baselines ---------------------------------------------
    main_spec = A.HeadSpec([int(g.size) for g in grids], pair_block=False)
    def main_design(rows):
        return A.design_of([r for (r, _m, _i, _j) in rows], grids, main_spec)[0]
    Phi_tr = main_design(train_rows)
    Tok_tr = token_inputs(Phi_tr, main_spec.layout)
    learned = {}
    if do_baselines:
        #: fixed offsets, never ``hash(str)`` -- Python salts string hashes per
        #: process, which would make a replication run a different experiment.
        for kind, off in (("mlp", 11), ("lstm", 23), ("transformer", 37)):
            key = jax.random.PRNGKey(seed * 101 + off)
            arm = LearnedArm(kind, key, Phi_tr.shape[1], LATENT_D, target_params,
                             seq_in=Tok_tr.shape[2])
            X = Phi_tr if kind == "mlp" else Tok_tr
            arm.fit(X, Ztrain, steps=bl_steps, seed=seed, verbose=verbose)
            learned[kind] = arm
            if verbose:
                print("  %-12s width %3d params %7d (target %7d, matched %s) "
                      "heldin-val mse %.5f" % (kind, arm.width, arm.n_params,
                                               target_params, arm.matched,
                                               arm.val_mse), flush=True)
        out["baseline_params"] = {k: {"width": v.width, "params": v.n_params,
                                      "matched": bool(v.matched),
                                      "heldin_val_mse": v.val_mse}
                                  for k, v in learned.items()}

    # -- 5. the rows --------------------------------------------------------
    table = {}
    for row in ROWS:
        rows = row_sets[row]
        if not rows:
            continue
        reqs = [r for (r, _m, _i, _j) in rows]
        table[row] = {}
        for name, rd in readers.items():
            Zh = rd.read(reqs)
            plans, scenes = render.plans(Zh)
            recs = evaluate.answers(bank, rows, plans, scenes)
            table[row][name] = _row_record(recs, row)
        for name, arm in learned.items():
            Phi = main_design(rows)
            X = Phi if name == "mlp" else token_inputs(Phi, main_spec.layout)
            Zh = arm.read(X)
            plans, scenes = render.plans(Zh)
            recs = evaluate.answers(bank, rows, plans, scenes)
            table[row][name] = _row_record(recs, row)
    out["table"] = table

    # -- 6. the environment's own columns on THIS split --------------------
    #: charter C.4 -- does the read settle onto a written item?
    out["c4_nearest_item"] = {}
    for row in ("seen", "unseen_pair"):
        if row_sets.get(row):
            rq = [r for (r, _m, _i, _j) in row_sets[row]]
            out["c4_nearest_item"][row] = A.nearest_item_audit(
                readers["clu"].read(rq), Ztrain)

    out["env_columns"] = {}
    for row in ("unseen_pair", "nway"):
        rows = row_sets[row]
        out["env_columns"][row] = {
            "blind": envcols.blind_columns(bank, rows),
            "random": envcols.random_column(bank, rows),
            "bend": envcols.bend_stats(bank, rows),
        }
    out["env_columns"]["fragment"] = envcols.fragment_columns(
        bank, train_rows, row_sets["unseen_pair"] + row_sets["nway"])
    if pool_sites:
        out["env_columns"]["fragment_cross_maze"] = envcols.fragment_cross_maze(
            bank, pool_sites, row_sets["unseen_pair"] + row_sets["nway"])
    out["seconds"] = time.time() - t0
    return out, {"ae": ae, "render": render, "mu": mu, "sd": sd,
                 "readers": readers, "learned": learned, "split": split,
                 "row_sets": row_sets, "train_rows": train_rows, "Ztrain": Ztrain,
                 "grids": grids, "main_spec": main_spec, "Xtr": Xtr, "itr": itr}


def _row_record(recs, row):
    base = (evaluate.nway_aggregate(recs) if row == "nway"
            else evaluate.aggregate(recs))
    base["per_route"] = evaluate.aggregate(recs)
    base["bend"] = evaluate.aggregate(recs, lambda x: x["bends"])
    base["nobend"] = evaluate.aggregate(recs, lambda x: not x["bends"])
    return base


def to_json(obj, path):
    def d(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, default=d)
