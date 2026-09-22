"""One end-to-end CMAZE-2b leg with the READ VERB as the only dial.

Everything outside the verb is wave 9's, imported and not re-implemented: the split
(``w9_maze.splits``), the item (``w9_maze.render.item_tensor``), the shared encoder and
its frozen decoder (``w9_maze.encoder``), the store and its charter-C controls
(``w9_maze.arms``), the learned baselines at matched parameters
(``w9_maze.baselines``), the parameterless executor and the exact integer judge
(``w9_maze.render.execute``, ``w8_envscout.maze_colour.judge``).  The request layer is
CMAZE-2b (``w14_request_layer``), which is this hub's gate.

THE THREE ADDRESSES, AND THE TWO PLACES A COMPOSITION CAN HAPPEN
----------------------------------------------------------------
For every arm and every row the leg reads three addresses --

    FULL  = (m, a, t, b)          the wave-9 read, printed as the reference column
    A     = (m, a, NaN, b)        the agent half
    B     = (m, NaN, t, b)        the target half

-- and then composes A and B in two places:

    AVG    : decode( (Z_A + Z_B) / 2 )                 composition in LATENT space
    MERGE  : overlay( decode(Z_A), decode(Z_B) )       composition in PLAN space

Every arm gets both verbs and the identical decoder.  The merge is not the CLU's
privilege: the transformer, the LSTM, the MLP, the matched-bytes lookup and all three
charter-C controls are read at the same two half-addresses and composed by the same
fixed operator.  A merge only we get is not a result.

Nothing in the verb is fitted.  ``NaN`` is the head's own "not requested" encoding; the
overlay has no parameter, no threshold and no tolerance; ``tolerance_used`` is ``null``
because the judge is the wave-8 exact integer judge, imported.
"""

from __future__ import annotations

import json
import os
import time

import equinox as eqx
import jax
import numpy as np

from chlu.core.gen_request_head import head_read, request_design
from w9_maze import arms as A
from w9_maze import envcols, evaluate, splits
from w9_maze.baselines import LearnedArm, token_inputs
from w9_maze.encoder import AE, LATENT_D, decode_plans, encode_all, train_ae
from w9_maze.pipeline import Renderer, build_items, representation_ceiling
from w9_maze.task import Request
from w14_read_verb.merge import MERGE_RULES, half_values, overlay

ROWS = ("seen", "unseen_pair", "nway", "budget_interp")


# ------------------------------------------------------------------ rows
def requests_for(bank, pairs_by_maze, budgets):
    out = []
    for m, ps in pairs_by_maze.items():
        for (i, j) in ps:
            for b in budgets:
                if bank[m].opt.get((i, j), 10 ** 9) > b:
                    continue
                out.append((Request(m, i, j, int(b)), m, i, j))
    return out


def row_sets_for(bank, split, budgets, interp):
    held = {m: (split.unseen_pair.get(m, []) + split.nway.get(m, []))
            for m in split.train_pairs}
    return {
        "seen": requests_for(bank, split.train_pairs, budgets),
        "unseen_pair": requests_for(bank, split.unseen_pair, budgets),
        "nway": requests_for(bank, split.nway, budgets),
        "budget_interp": requests_for(bank, held, interp),
    }


# ------------------------------------------------------- reading an address
def read_values(reader, V, grids, main_spec):
    """Read ONE address table ``V`` (N, 4).  ``NaN`` = the word is not requested.

    The three arm families are dispatched here and nowhere else, so every arm sees the
    same ``V`` and the half-address is a property of the REQUEST, not of the arm.
    """
    if isinstance(reader, A.CluRead):
        Phi, _oor = request_design(V, grids, reader.spec)
        return head_read(reader.heads, reader.store, Phi)
    if isinstance(reader, A.LookupRead):
        # the matched-bytes floor answers with the mean of the stored items whose
        # REQUESTED words agree most.  A word that is not requested is not compared --
        # the same semantics ``NaN`` has in the head's design.
        Q = np.asarray(V, np.float64)
        w = np.array([4.0, 1.0, 1.0, 0.25], np.float64)
        out = np.zeros((Q.shape[0], reader.Z.shape[1]), np.float32)
        keys = reader.keys.astype(np.float64)
        for i in range(Q.shape[0]):
            asked = ~np.isnan(Q[i])
            if not asked.any():
                out[i] = reader.Z.mean(axis=0)
                continue
            agree = ((keys[:, asked] == Q[i][asked]) * w[asked]).sum(axis=1)
            best = agree.max()
            sel = np.flatnonzero(agree >= best - 1e-9)
            out[i] = reader.Z[sel].mean(axis=0)
        return out
    if isinstance(reader, LearnedArm):
        Phi, _oor = request_design(V, grids, main_spec)
        X = Phi if reader.kind == "mlp" else token_inputs(Phi, main_spec.layout)
        return reader.read(X)
    raise TypeError(type(reader))


def _cos(X, Y):
    X = np.asarray(X, np.float64)
    Y = np.asarray(Y, np.float64)
    nx = np.linalg.norm(X, axis=1) + 1e-12
    ny = np.linalg.norm(Y, axis=1) + 1e-12
    return (X * Y).sum(1) / (nx * ny)


def _rec(bank, rows, plans, scenes, nway=False):
    recs = evaluate.answers(bank, rows, plans, scenes)
    base = (evaluate.nway_aggregate(recs) if nway else evaluate.aggregate(recs))
    if nway:
        base["per_route"] = evaluate.aggregate(recs)
    base["bend"] = evaluate.aggregate(recs, lambda x: x["bends"])
    base["nobend"] = evaluate.aggregate(recs, lambda x: not x["bends"])
    return base


def verb_table(bank, rows, reader, render, grids, main_spec, nway=False):
    """Every verb for one arm on one row.  The two latents are read ONCE."""
    reqs = [r for (r, _m, _i, _j) in rows]
    V = np.array([r.values() for r in reqs], np.float64)
    Va, Vt = half_values(reqs)
    Zf = read_values(reader, V, grids, main_spec)
    Za = read_values(reader, Va, grids, main_spec)
    Zb = read_values(reader, Vt, grids, main_spec)

    out = {}
    pf, sf = render.plans(Zf)
    out["FULL"] = _rec(bank, rows, pf, sf, nway)
    zavg = ((np.asarray(Za, np.float64) + np.asarray(Zb, np.float64)) / 2.0
            ).astype(np.float32)
    pv, sv = render.plans(zavg)
    out["AVG"] = _rec(bank, rows, pv, sv, nway)
    pa, sa = render.plans(Za)
    pb, sb = render.plans(Zb)
    for rule in MERGE_RULES:
        pm = overlay(pa, pb, rule)
        # RESULT rows never refuse, so the scene is carried only to keep the record's
        # shape identical across verbs; nothing in a RESULT row reads it.
        out["MERGE_" + rule] = _rec(bank, rows, pm, sb, nway)
    out["_latent"] = {
        "cos_A_B": float(np.mean(_cos(Za, Zb))),
        "cos_AVG_FULL": float(np.mean(_cos(zavg, Zf))),
        "norm_A": float(np.linalg.norm(Za, axis=1).mean()),
        "norm_B": float(np.linalg.norm(Zb, axis=1).mean()),
        "norm_FULL": float(np.linalg.norm(Zf, axis=1).mean()),
    }
    return out


# ------------------------------------------------------------------- the leg
def run(bank, seed, budgets, interp, pool_sites=None, n_atoms=31, K=8, ridge=0.01,
        ae_steps=12000, bl_steps=4000, which="val", ae_cache=None,
        do_baselines=True, do_controls=True, verbose=True, lowrank_atoms=None):
    t0 = time.time()
    val, test = splits.make_split(bank, seed)
    split = val if which == "val" else test
    M = len(bank)
    out = {"seed": int(seed), "which": which, "n_mazes": M, "n_atoms": int(n_atoms),
           "K": int(K), "ridge": float(ridge), "ae_steps": int(ae_steps),
           "bl_steps": int(bl_steps), "latent_d": LATENT_D,
           "budgets": list(budgets), "interp": list(interp),
           "request_layer": "w14_request_layer.task2.build_site_v2 (CMAZE-2b)",
           "tolerance_used": None}

    train_rows = requests_for(bank, split.train_pairs, budgets)
    row_sets = row_sets_for(bank, split, budgets, interp)
    out["n_train_requests"] = len(train_rows)
    out["n_rows"] = {r: len(v) for r, v in row_sets.items()}
    if verbose:
        print("  train requests %d; rows %s" % (len(train_rows), out["n_rows"]),
              flush=True)

    # -- the shared encoder -------------------------------------------------
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
    out["ae_loss_history"] = hist
    out["ae_cache"] = ae_cache

    # -- the ceilings -------------------------------------------------------
    out["representation_ceiling"] = {r: representation_ceiling(bank, row_sets[r])
                                     for r in ROWS if row_sets[r]}
    Ztr = encode_all(ae, Xtr)
    ptr, sctr = decode_plans(ae, Ztr)
    out["pipeline_ceiling_train"] = _rec(bank, train_rows, ptr[itr], sctr[itr])
    ceil_rows, zrows = {}, {}
    for r in ROWS:
        if not row_sets[r]:
            continue
        Xr, ir, _ = build_items(bank, row_sets[r])
        Zr = encode_all(ae, Xr)
        pr, sr = decode_plans(ae, Zr)
        ceil_rows[r] = _rec(bank, row_sets[r], pr[ir], sr[ir])
        zrows[r] = Zr[ir]
    out["pipeline_ceiling_rows"] = ceil_rows
    if verbose:
        print("  ceiling: pipeline train %.4f  rows %s"
              % (out["pipeline_ceiling_train"]["validity"],
                 {r: round(v["validity"], 4) for r, v in ceil_rows.items()}),
              flush=True)

    # -- the arms -----------------------------------------------------------
    Zraw = Ztr[itr]
    mu = Zraw.mean(axis=0)
    sd = Zraw.std(axis=0) + 1e-6
    Ztrain = ((Zraw - mu) / sd).astype(np.float32)
    render = Renderer(ae, mu, sd)
    reqs_train = [r for (r, _m, _i, _j) in train_rows]
    grids = A.make_grids(M, 6, 6, budgets)
    clu = A.CluRead(Ztrain, reqs_train, grids, K=K, n_atoms=n_atoms, ridge=ridge,
                    seed=seed)
    led = clu.ledger()
    out["ledger"] = led
    out["request_only_assertion"] = clu.assert_request_only(reqs_train)
    target_params = led["read_floats"]

    # THE STORE-SPAN CEILING, printed on every store row beside rank/width.
    out["span_ceiling_rows"] = {}
    for r in ROWS:
        if r not in zrows:
            continue
        Zs_r = (zrows[r] - mu) / sd
        Zp = np.zeros_like(Zs_r)
        for ch, Mk in zip(clu.well_channels, clu.store.modes):
            U = np.linalg.qr(np.asarray(Mk, np.float64).T)[0]
            Zp[:, ch] = np.asarray((Zs_r[:, ch] @ U) @ U.T, np.float32)
        pl, sc = render.plans(Zp)
        out["span_ceiling_rows"][r] = _rec(bank, row_sets[r], pl, sc)

    readers = {"clu": clu}
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
    out["control_ledgers"] = {
        k: {"rank_over_width": v.ledger()["rank_over_width"],
            "deployed_rank_per_well": v.ledger()["deployed_rank_per_well"],
            "store_atom_floats": v.ledger()["store_atom_floats"],
            "read_floats": v.ledger()["read_floats"]}
        for k, v in readers.items() if isinstance(v, A.CluRead)}

    main_spec = A.HeadSpec([int(g.size) for g in grids], pair_block=False)
    if do_baselines:
        Phi_tr, _ = request_design(
            np.array([r.values() for r in reqs_train], np.float64), grids, main_spec)
        Tok_tr = token_inputs(Phi_tr, main_spec.layout)
        for kind, off in (("mlp", 11), ("lstm", 23), ("transformer", 37)):
            key = jax.random.PRNGKey(seed * 101 + off)
            arm = LearnedArm(kind, key, Phi_tr.shape[1], LATENT_D, target_params,
                             seq_in=Tok_tr.shape[2])
            X = Phi_tr if kind == "mlp" else Tok_tr
            arm.fit(X, Ztrain, steps=bl_steps, seed=seed, verbose=False)
            readers[kind] = arm
            if verbose:
                print("  %-12s width %3d params %7d (target %7d, matched %s)"
                      % (kind, arm.width, arm.n_params, target_params, arm.matched),
                      flush=True)
        out["baseline_params"] = {
            k: {"width": readers[k].width, "params": readers[k].n_params,
                "matched": bool(readers[k].matched),
                "heldin_val_mse": readers[k].val_mse}
            for k in ("mlp", "lstm", "transformer") if k in readers}

    # -- THE TABLE: arm x verb ---------------------------------------------
    table = {}
    for row in ROWS:
        rows = row_sets[row]
        if not rows:
            continue
        table[row] = {}
        for name, rd in readers.items():
            table[row][name] = verb_table(bank, rows, rd, render, grids, main_spec,
                                          nway=(row == "nway"))
        if verbose:
            print("  row %-14s n=%-5d clu FULL %.4f  AVG %.4f  MERGE %.4f"
                  % (row, len(rows), table[row]["clu"]["FULL"]["validity"],
                     table[row]["clu"]["AVG"]["validity"],
                     table[row]["clu"]["MERGE_target_wins"]["validity"]), flush=True)
    out["table"] = table

    # -- the environment's own columns on THIS split ------------------------
    out["env_columns"] = {}
    for row in ("unseen_pair", "nway"):
        if row_sets[row]:
            out["env_columns"][row] = {
                "blind": envcols.blind_columns(bank, row_sets[row]),
                "bend": envcols.bend_stats(bank, row_sets[row])}
    out["c4_nearest_item"] = {}
    for row in ("seen", "unseen_pair"):
        if row_sets.get(row):
            rq = [r for (r, _m, _i, _j) in row_sets[row]]
            out["c4_nearest_item"][row] = A.nearest_item_audit(clu.read(rq), Ztrain)
    out["seconds"] = time.time() - t0
    return out, {"ae": ae, "render": render, "readers": readers, "split": split,
                 "row_sets": row_sets, "Ztrain": Ztrain, "grids": grids,
                 "main_spec": main_spec, "clu": clu, "mu": mu, "sd": sd}


def to_json(obj, path):
    def d(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, default=d)
