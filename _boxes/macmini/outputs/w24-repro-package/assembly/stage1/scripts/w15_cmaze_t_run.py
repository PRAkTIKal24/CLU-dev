"""CMAZE-T VAL smoke, one seed.  Writes ``val_seed<N>.json`` into the output dir.

    python scripts/w15_cmaze_t_run.py <seed_index> <out_dir> [--mazes-train N]

Everything here is VAL.  ⛔ ZERO scored runs are spent by this spoke (task file), so no
row of the scored ledger is claimed and no test maze block is touched: the third maze
block is built and its seed range recorded, but it is NEVER evaluated -- it is wave 16's.
"""
from __future__ import annotations
import os
import sys
import time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chlu.exp_w15_cmaze_t as T                                    # noqa: E402
import chlu.exp_w15_cmaze_t_arms as A                               # noqa: E402
K_COLOURS = 4
SEQ_T = 8
def log(*a):
    print(*a, flush=True)


def trajectory_sequences(insts, feats, gsums, rng, per_maze=6, T_len=SEQ_T):
    """Random LEGAL trajectories, each a length-``T_len`` sequence of transitions.

    This is the rivals' natural form: the LSTM and the transformer are given something
    to combine across time.  ⚠ The transition itself is MARKOV by construction (the five
    words determine it), so recurrence has nothing to recover that the observation does
    not already carry -- that is a reported fact about the task, not a handicap applied
    to a rival.
    """
    from w9_harness.dynamics import initial_state, key_homes, step

    X, Y, M = [], [], []
    od = A.obs_dim(int(feats[0].shape[0]), K_COLOURS)
    cd = T.code_dim(K_COLOURS)
    for m, inst in enumerate(insts):
        homes = key_homes(inst)
        for _ in range(per_maze):
            st = initial_state(inst)
            xs = np.zeros((T_len, od), np.float32)
            ys = np.zeros((T_len, cd), np.float32)
            ms = np.zeros(T_len, np.float32)
            for t in range(T_len):
                a = T.ACTIONS[int(rng.integers(5))]
                xs[t] = A.obs_of(inst, feats[m], st, a, gsums[m])
                legal, disp, ev, pn, nxt = T.outcome_of(inst, st, a, homes)
                ys[t] = T.encode_outcome(K_COLOURS, legal, disp, ev, pn)
                ms[t] = 1.0
                if legal:
                    st = nxt
                    if st.pos == inst.goal:
                        break
            X.append(xs)
            Y.append(ys)
            M.append(ms)
    return np.stack(X), np.stack(Y), np.stack(M)


def single_sequences(rows, insts, feats, gsums):
    O = A.build_obs(rows, insts, feats, gsums)
    C = A.build_codes(rows, K_COLOURS)
    return O[:, None, :], C[:, None, :], np.ones((len(rows), 1), np.float32)


def train_rival(kind, width, kw, Xs, Ys, Ms, Xt, Yt, Mt, seed, steps, label):
    """Alternate batches between the stratified single transitions and the trajectory
    sequences, so every trunk is trained on both forms."""
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import optax

    pol = A.build_policy(kind, Xs.shape[-1], width, T.code_dim(K_COLOURS),
                         seed=seed, **kw)
    blocks = T.code_blocks(K_COLOURS)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))
    state = opt.init(eqx.filter(pol, eqx.is_array))

    @eqx.filter_jit
    def upd(p, st, xb, yb, mb):
        loss, g = eqx.filter_value_and_grad(A._seq_loss)(p, xb, yb, mb, blocks)
        u, st = opt.update(g, st, eqx.filter(p, eqx.is_array))
        return eqx.apply_updates(p, u), st, loss

    rng = np.random.default_rng(seed)
    t0 = time.time()
    hist = []
    for s in range(steps):
        if s % 2 == 0 or Xt.shape[0] == 0:
            idx = rng.integers(0, Xs.shape[0], size=min(96, Xs.shape[0]))
            xb, yb, mb = Xs[idx], Ys[idx], Ms[idx]
        else:
            idx = rng.integers(0, Xt.shape[0], size=min(48, Xt.shape[0]))
            xb, yb, mb = Xt[idx], Yt[idx], Mt[idx]
        pol, state, loss = upd(pol, state, jnp.asarray(xb), jnp.asarray(yb),
                               jnp.asarray(mb))
        if s % 400 == 0 or s == steps - 1:
            hist.append((s, float(loss)))
    rep = {"kind": kind, "width": int(width), "extra": {a: b for a, b in kw.items()
                                                        if a != "tmax"},
           "params": A.n_params(pol), "steps": int(steps),
           "final_loss": hist[-1][1], "loss_curve": hist,
           "seconds": time.time() - t0,
           "examples_seen": int(steps // 2 * 96 + steps // 2 * 48 * SEQ_T)}
    log("    TRAIN %-12s params %7d  loss %.4f  [%.1fs]"
        % (label, rep["params"], rep["final_loss"], rep["seconds"]))
    return pol, rep


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def deletion_block(train_rows, tr_insts, tr_feats, tr_g, Wtr, Ztr, Ctr, codec, grids,
                   cfgA, va_insts, va_split, va_feats, va_g, widths, Xt, Yt, Mt,
                   seed_index, rival_steps):
    """Two deletions, both measured against the surviving colours.

    VALUE:        every transition carrying colour c in ANY rule-bearing role.
    CONJUNCTION:  colour c AND a tinted floor -- rows whose TINT word is c.

    The CLU deletes by downdating the retained statistics (store and head), which is
    exact and data-free.  The rival can only delete by RETRAINING WITHOUT, and that cost
    is printed; the rival AS-TRAINED row is printed beside it.
    ⛔ The frozen codec and the frozen state encoder are OUTSIDE the deletable object and
    are not touched by either deletion.
    """
    k = K_COLOURS
    #: colour 1 is the deletion target -- it is an ordinary TRAIN colour, not the value
    #: hold-out colour (which is already absent from train).
    c = 1
    res = {"deleted_colour": c,
           "note": ("colour %d is an ordinary TRAIN colour; the value HOLD-OUT colour "
                    "%d is already absent from train and could not be deleted from it"
                    % (c, k)),
           "frozen_objects_declared_outside_the_deletable_object":
               ["Codec (next-state code <-> latent)",
                "w9_harness state encoder (conv AE on maze images)"],
           "post_deletion_adaptation": ("the head's retained statistics A and Mx are "
                                        "downdated and the SAME closed form is "
                                        "re-solved.  Light and DATA-FREE -- no pass over "
                                        "the retained pool, no item kept.  "
                                        "NOT 'no changes'.")}
    idx_all = np.arange(len(train_rows))
    for mode, sel in (("value", np.array([i for i in idx_all
                                          if T._has_colour(train_rows[i].words, k, c)])),
                      ("conjunction_colour_AND_tinted_floor",
                       np.array([i for i in idx_all
                                 if train_rows[i].words[3] == c]))):
        if sel.size == 0:
            res[mode] = {"DECLARED_NOT_RUN": "no train rows match"}
            continue
        keep = np.setdiff1d(idx_all, sel)
        block = {"n_rows_deleted": int(sel.size), "n_rows_kept": int(keep.size)}

        # ---- CLU: downdate ------------------------------------------------
        clu = A.TransitionClu(Ztr, Wtr, grids, K=cfgA["K"], n_atoms=cfgA["n_atoms"],
                              ridge=cfgA["ridge"], seed=seed_index)
        t0 = time.time()
        rec = clu.delete_rows_exact(Ztr[sel], Wtr[sel])
        block["clu_downdate_receipt"] = {kk: vv for kk, vv in rec.items()
                                         if kk != "basis_change_shapes"}
        block["clu_adaptation_cost"] = {
            "seconds": time.time() - t0,
            "passes_over_retained_data": 0, "items_kept": False,
            "gradient_steps": 0, "examples_seen": 0}

        # ---- store-scoped exactness: downdate vs fitted-without -----------
        #: ⭐ the SAME well partition, so the comparison is between two versions of the
        #: same object.  Recomputing the ward partition on the surviving rows would put
        #: different channels in each well and compare two different objects.
        clu_ref = A.TransitionClu(Ztr[keep], Wtr[keep], grids, K=cfgA["K"],
                                  n_atoms=cfgA["n_atoms"], ridge=cfgA["ridge"],
                                  seed=seed_index,
                                  well_channels=clu.well_channels)
        devs, scales = [], []
        for w1, w2 in zip(clu.store.stats, clu_ref.store.stats):
            d = np.abs(np.asarray(w1.S, np.float64) - np.asarray(w2.S, np.float64))
            devs.append(float(d.max()))
            scales.append(float(np.abs(np.asarray(w2.S, np.float64)).max()))
        rel = max(devs) / max(max(scales), 1e-12)
        atom_dev = max(float(np.abs(np.asarray(M1, np.float64)
                                    - np.asarray(M2, np.float64)).max())
                       for M1, M2 in zip(clu.store.modes, clu_ref.store.modes))
        atom_scale = max(float(np.abs(np.asarray(M, np.float64)).max())
                         for M in clu_ref.store.modes)
        block["store_scoped_exactness"] = {
            "statement": ("the store after deletion equals the store FITTED WITHOUT the "
                          "rows -- per-well sufficient statistic S_k, and the deployed "
                          "atoms derived from it -- at the SAME well partition"),
            "deployed_atoms_max_abs_delta": atom_dev,
            "deployed_atoms_scale": atom_scale,
            "deployed_atoms_max_relative_delta": atom_dev / max(atom_scale, 1e-12),
            "max_abs_delta": max(devs), "scale": max(scales),
            "max_relative_delta": rel, "bar": 1e-5, "pass": bool(rel <= 1e-5),
            "n_after_per_well_downdated": [int(w.n) for w in clu.store.stats],
            "n_after_per_well_fitted_without": [int(w.n) for w in clu_ref.store.stats]}

        # ---- composition on the SURVIVING colours -------------------------
        surv = {}
        for name in ("seen", "pair", "nway"):
            rs = [r for r in va_split.get(name, [])
                  if not T._has_colour(r.words, k, c)]
            if not rs:
                surv[name] = {"DECLARED_NOT_RUN": "no surviving-colour rows"}
                continue
            W = A.build_words(rs)
            surv[name] = {
                "n": len(rs),
                "CLU after deletion (downdate)": A.score_rows(
                    va_insts, rs, T.decode_codes(k, A.codec_decode(codec,
                                                                   clu.read(W))))["exact"],
                "CLU fitted WITHOUT the rows": A.score_rows(
                    va_insts, rs, T.decode_codes(k, A.codec_decode(
                        codec, clu_ref.read(W))))["exact"]}
        block["composition_on_surviving_colours"] = surv

        # ---- the rival: retrain-without, and as-trained -------------------
        kind = "mlp"
        w, p, kw = widths[kind]
        rows_keep = [train_rows[i] for i in keep]
        Xk, Yk, Mk = single_sequences(rows_keep, tr_insts, tr_feats, tr_g)
        t0 = time.time()
        pol_rt, rep_rt = train_rival(kind, w, kw, Xk, Yk, Mk, Xt, Yt, Mt,
                                     seed_index, rival_steps, "mlp retrain-without")
        rows_all = train_rows
        Xa, Ya, Ma = single_sequences(rows_all, tr_insts, tr_feats, tr_g)
        pol_as, rep_as = train_rival(kind, w, kw, Xa, Ya, Ma, Xt, Yt, Mt,
                                     seed_index, rival_steps, "mlp as-trained")
        block["rival_retrain_without_cost"] = {
            "seconds": rep_rt["seconds"], "gradient_steps": rep_rt["steps"],
            "examples_seen": rep_rt["examples_seen"],
            "passes_over_retained_data": "all of them -- the whole surviving pool is "
                                         "re-read every batch",
            "items_kept": True,
            "note": "the rival has no other way to delete"}
        rv = {}
        for name in ("seen", "pair", "nway"):
            rs = [r for r in va_split.get(name, [])
                  if not T._has_colour(r.words, k, c)]
            if not rs:
                rv[name] = {"DECLARED_NOT_RUN": "no surviving-colour rows"}
                continue
            O = A.build_obs(rs, va_insts, va_feats, va_g)
            rv[name] = {
                "n": len(rs),
                "rival retrain-without": A.score_rows(
                    va_insts, rs, A.policy_predict(pol_rt, O, k))["exact"],
                "rival as-trained": A.score_rows(
                    va_insts, rs, A.policy_predict(pol_as, O, k))["exact"]}
        block["rival_composition_on_surviving_colours"] = rv
        res[mode] = block
    return res


