"""CMAZE-T VAL smoke, one seed.  Writes ``val_seed<N>.json`` into the output dir.

    python scripts/w15_cmaze_t_run.py <seed_index> <out_dir> [--mazes-train N]

Everything here is VAL.  ⛔ ZERO scored runs are spent by this spoke (task file), so no
row of the scored ledger is claimed and no test maze block is touched: the third maze
block is built and its seed range recorded, but it is NEVER evaluated -- it is wave 16's.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chlu.exp_w15_cmaze_t as T                                    # noqa: E402
import chlu.exp_w15_cmaze_t_arms as A                               # noqa: E402

K_COLOURS = 4
SEQ_T = 8

#: the CLU val sweep.  ⭐ Selection is on the `seen` row of VAL and nothing else.
#: ⭐ The grid spans the SATURATION RATIO (span fraction = n_atoms / d_k).  A store at
#: high span nearly fills its wells, so RANDOM atoms span them too and the charter-C
#: controls stop collapsing -- which is exactly why the controls are part of the
#: selection bar and why low-span configs are in the grid.
CONFIG_GRID = [
    #  (latent D, K wells, n_atoms, ridge).  span fraction = n_atoms / (D / K).
    dict(latent=32, K=4, n_atoms=2, ridge=1.0),     # span 0.250
    dict(latent=48, K=4, n_atoms=2, ridge=1.0),     # span 0.167
    dict(latent=64, K=4, n_atoms=2, ridge=1.0),     # span 0.125
    dict(latent=64, K=4, n_atoms=3, ridge=1.0),     # span 0.188
    dict(latent=64, K=8, n_atoms=2, ridge=1.0),     # span 0.250
    dict(latent=64, K=8, n_atoms=3, ridge=1.0),     # span 0.375
    dict(latent=64, K=8, n_atoms=6, ridge=1.0),     # span 0.750
    dict(latent=64, K=8, n_atoms=2, ridge=0.1),
    dict(latent=96, K=8, n_atoms=2, ridge=1.0),     # span 0.167
    dict(latent=96, K=8, n_atoms=3, ridge=1.0),     # span 0.250
    dict(latent=128, K=4, n_atoms=3, ridge=1.0),    # span 0.094
    dict(latent=128, K=8, n_atoms=2, ridge=1.0),    # span 0.125
    dict(latent=128, K=8, n_atoms=3, ridge=1.0),    # span 0.188
    dict(latent=128, K=8, n_atoms=6, ridge=1.0),    # span 0.375
    dict(latent=128, K=8, n_atoms=3, ridge=0.1),
    dict(latent=256, K=8, n_atoms=3, ridge=1.0),    # span 0.094
]


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------ sequences
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


# ------------------------------------------------------------------- the seed run
def run(seed_index, out_dir, n_train_mazes=90, n_val_mazes=45, per_tuple=2,
        enc_steps=1500, codec_steps=2500, rival_steps=8000,
        max_rows_per_class=4000):
    t_start = time.time()
    rng = np.random.default_rng(10_000 + seed_index)
    base = 2_000_000 + 400_000 * seed_index
    out = {"seed_index": int(seed_index), "tolerance_used": None,
           "scored_runs_spent": 0, "k_colours": K_COLOURS}

    # ---------------------------------------------------------------- banks
    log("[seed %d] banks" % seed_index)
    t0 = time.time()
    tr_insts, tr_rows, tr_nat, tr_meta = T.build_bank(base, n_train_mazes,
                                                      per_tuple=per_tuple,
                                                      n_natural=150, verbose=True)
    va_insts, va_rows, va_nat, va_meta = T.build_bank(base + 200_000, n_val_mazes,
                                                      per_tuple=per_tuple,
                                                      n_natural=150)
    out["banks"] = {
        "train": dict(tr_meta, seed_block=[base, base + 200_000]),
        "val": dict(va_meta, seed_block=[base + 200_000, base + 300_000]),
        "test_RESERVED_NOT_EVALUATED": {
            "seed_block": [base + 300_000, base + 400_000],
            "note": ("reserved for wave 16; this spoke spends ZERO scored runs and "
                     "never builds or evaluates it")},
        "maze_disjointness": ("train and val mazes come from disjoint generator seed "
                              "blocks, so every val maze is a random maze never seen "
                              "at train and a within-maze lookup cannot transfer"),
        "seconds": time.time() - t0}
    log("  train %d mazes / %d rows, val %d mazes / %d rows  [%.1fs]"
        % (tr_meta["n_mazes"], tr_meta["n_rows"], va_meta["n_mazes"],
           va_meta["n_rows"], time.time() - t0))

    # ------------------------------------------------- acceptance 1, 2, 3, 4
    log("[seed %d] acceptance audits" % seed_index)
    t0 = time.time()
    all_insts = tr_insts + va_insts
    all_rows = list(tr_rows) + [T.TRow(r.maze + len(tr_insts), r.prefix, r.pos, r.ks,
                                       r.action, r.words, r.legal, r.disp, r.ev,
                                       r.pocket_next, r.bites_b) for r in va_rows]
    out["ACCEPT_1_independent_replay"] = T.replay_audit(all_insts, all_rows)
    out["ACCEPT_2_no_impossible_example"] = T.impossible_example_audit(all_insts,
                                                                      all_rows)
    out["ACCEPT_3b_words_determine_transition"] = T.words_determine_transition(all_rows)
    out["ACCEPT_4_rule_b_bites"] = {
        "stratified_bank_train": T.rule_b_column(tr_rows),
        "stratified_bank_val": T.rule_b_column(va_rows),
        "natural_distribution_train": T.rule_b_column(tr_nat),
        "natural_distribution_val": T.rule_b_column(va_nat),
        "which_is_which": ("the stratified bank caps rows per WORD TUPLE, which is what "
                           "gives the rare configurations enough rows to score; the "
                           "natural distribution is a uniform sample of (reachable "
                           "state, action) pairs.  Both are printed; neither is a gate.")}
    log("  replay %d rows, %d disagreements  [%.1fs]"
        % (out["ACCEPT_1_independent_replay"]["transitions_replayed"],
           out["ACCEPT_1_independent_replay"]["disagreements"], time.time() - t0))

    # ---------------------------------------------------------------- splits
    ho = T.make_holdout(K_COLOURS, seed=seed_index)
    tr_split = T.split_rows(tr_rows, ho, K_COLOURS)
    va_split_full = T.split_rows(va_rows, ho, K_COLOURS)
    #: ⭐ a DECLARED cap on the scored rows per class, uniform and seeded.  Every arm on
    #: a row sees the SAME rows; the cap exists so the Python-side exact judge stays
    #: inside the wall clock, and the n it produces is a column on every table.
    cap_rng = np.random.default_rng(777 + seed_index)
    va_split = {}
    for nm, rs in va_split_full.items():
        if len(rs) > max_rows_per_class:
            idx = cap_rng.permutation(len(rs))[:max_rows_per_class]
            va_split[nm] = [rs[int(i)] for i in sorted(idx)]
        else:
            va_split[nm] = rs
    train_rows = tr_split.get("seen", [])
    out["holdout"] = {"value_colour": ho.value_colour,
                      "pairs": [list(p) for p in ho.pairs],
                      "nway": [list(c) for c in ho.nway],
                      "n_train_rows_by_class": {k: len(v) for k, v in tr_split.items()},
                      "n_val_rows_by_class_BEFORE_CAP":
                          {k: len(v) for k, v in va_split_full.items()},
                      "n_val_rows_by_class_SCORED":
                          {k: len(v) for k, v in va_split.items()},
                      "max_rows_per_class": int(max_rows_per_class),
                      "cap_rule": ("uniform seeded subsample per class; every arm on a "
                                   "row sees the same rows")}
    out["ACCEPT_3_holdout_disjointness"] = T.disjointness_report(
        train_rows, {k: v for k, v in va_split.items() if k != "seen"}, ho, K_COLOURS)
    log("  train(seen) %d rows; val rows %s"
        % (len(train_rows), out["holdout"]["n_val_rows_by_class_SCORED"]))

    # -------------------------------------------------------- the frozen codec
    log("[seed %d] frozen codec (reconstruction only)" % seed_index)
    Ctr = A.build_codes(train_rows, K_COLOURS)
    #: ⭐ PREREG AMENDMENT 2.  The frozen codec is fitted on the TRAIN codes PLUS the
    #: enumerated emission ALPHABET -- a purely combinatorial set determined by the
    #: emission format, carrying no row, no word, no rule and no outcome.  Without it a
    #: codec fitted on train codes alone cannot round-trip a never-seen colour's code and
    #: the value hold-out row measures the codec instead of the arm (measured ceiling
    #: 0.4963 +- 0.0368 in the pre-amendment smoke).  The STORE is still built from the
    #: TRAIN codes only.
    Calpha = T.enumerate_code_alphabet(K_COLOURS)
    Ccodec = np.concatenate([Ctr, Calpha], axis=0)
    out["next_state_code"] = {
        "code_dim": T.code_dim(K_COLOURS),
        "blocks": {k: list(v) for k, v in T.code_blocks(K_COLOURS).items()},
        "n_distinct_codes_in_train": int(len(set(map(tuple, Ctr.tolist())))),
        "n_distinct_codes_in_val": int(len(set(
            map(tuple, A.build_codes(va_rows, K_COLOURS).tolist())))),
        "n_codes_in_the_enumerated_alphabet": int(Calpha.shape[0]),
        "codec_training_set": ("TRAIN codes + the enumerated emission alphabet "
                               "(PREREG AMENDMENT 2); the STORE is built from the TRAIN "
                               "codes only"),
        "why_this_matters": ("the emission target is a SMALL DISCRETE SET.  A random "
                             "projection of a few dozen well-separated points into a "
                             "handful of dimensions preserves their separation "
                             "(Johnson-Lindenstrauss), so a RANDOM-ATOM store can be "
                             "decodable even though its content was never fitted.  That "
                             "is why the charter-C controls are measured here rather "
                             "than assumed, and why they are part of the selection "
                             "bar.")}

    # -------------------------------------------------- the frozen state encoder
    log("[seed %d] frozen state encoder (reconstruction only)" % seed_index)
    enc, enc_rep = A.train_state_encoder(tr_insts, steps=enc_steps, seed=seed_index,
                                         verbose=True, restarts=3, accept_bce=0.010)
    enc_rep.pop("loss_curve", None)
    out["frozen_state_encoder"] = {k: v for k, v in enc_rep.items()
                                   if not isinstance(v, np.ndarray)}
    out["frozen_state_encoder"]["independence"] = {
        k: v for k, v in __import__("w9_harness.encoder", fromlist=["x"])
        .independence_report(enc).items() if k != "loss_source"}
    tr_feats = A.feature_maps(enc, tr_insts)
    va_feats = A.feature_maps(enc, va_insts)
    from w9_harness.features import global_summary
    tr_g = [global_summary(S) for S in tr_feats]
    va_g = [global_summary(S) for S in va_feats]

    # ------------------------------------------------------- the CLU val sweep
    log("[seed %d] CLU val sweep (selection on the `seen` row only)" % seed_index)
    Wtr = A.build_words(train_rows)
    grids = [np.arange(c, dtype=np.float64) for c in T.alphabets(K_COLOURS)]
    va_seen = va_split.get("seen", [])
    Wva_seen = A.build_words(va_seen)
    sweep = []
    codec_cache = {}
    best = None
    for cfg in CONFIG_GRID:
        lat = cfg["latent"]
        if lat not in codec_cache:
            codec_cache[lat] = A.train_codec(Ccodec, K_COLOURS, latent=lat,
                                             steps=codec_steps, seed=seed_index)
        codec, crep = codec_cache[lat]
        Ztr = A.codec_encode(codec, Ctr)
        try:
            clu = A.TransitionClu(Ztr, Wtr, grids, K=cfg["K"], n_atoms=cfg["n_atoms"],
                                  ridge=cfg["ridge"], seed=seed_index)
        except AssertionError as exc:
            sweep.append(dict(cfg, val_seen_exact=None,
                              DECLARED_NOT_RUN="saturated store: %s" % exc))
            log("    %s  DECLARED NOT-RUN (saturated)" % cfg)
            continue
        Zh = clu.read(Wva_seen)
        em = T.decode_codes(K_COLOURS, A.codec_decode(codec, Zh))
        sc = A.score_rows(va_insts, va_seen, em)["exact"]
        ctl_sc = {}
        for cname in A.TransitionClu.CONTROLS:
            cobj = A.TransitionClu(Ztr, Wtr, grids, K=cfg["K"], n_atoms=cfg["n_atoms"],
                                   ridge=cfg["ridge"], control=cname, seed=seed_index)
            ctl_sc[cname] = float(A.score_rows(
                va_insts, va_seen,
                T.decode_codes(K_COLOURS,
                               A.codec_decode(codec, cobj.read(Wva_seen))))["exact"])
        led = clu.ledger()
        worst_ctl = max(ctl_sc.values())
        row = dict(cfg, val_seen_exact=float(sc),
                   val_seen_exact_controls=ctl_sc,
                   worst_control=worst_ctl,
                   read_floats=led["read_floats"],
                   span_fraction=led["span_fraction_n_atoms_over_d_k"][0],
                   clears_bar=bool(sc >= 0.90 and worst_ctl <= sc - 0.15))
        sweep.append(row)
        log("    latent %3d K %2d atoms %2d ridge %.2g span %.3f -> seen %.4f "
            "(worst control %.4f) %s"
            % (lat, cfg["K"], cfg["n_atoms"], cfg["ridge"], row["span_fraction"], sc,
               worst_ctl, "CLEARS" if row["clears_bar"] else "below bar"))
        #: ⭐ ties on the `seen` row are broken by the FEWEST read floats -- the most
        #: parsimonious config that clears the bar, declared before the sweep is run.
        rank = (int(row["clears_bar"]), float(sc), -float(row["read_floats"]))
        if best is None or rank > best[1]:
            best = (cfg, rank, row)
    out["clu_val_sweep"] = sweep
    cfgA = dict(best[0])
    out["config_A"] = {"config": cfgA, "val_seen_exact": best[1][1],
                       "selection_rule": ("val `exact` on the `seen` row only; the "
                                          "held-out rows and the rollout row were not "
                                          "consulted.  Ties broken by the fewest "
                                          "read floats."),
                       "bar": ("seen >= 0.90 AND every charter-C control "
                               "collapses on `seen` (control <= CLU - 0.15)"),
                       "bar_met": bool(best[2]["clears_bar"])}
    log("  CONFIG-A %s  seen %.4f  bar met: %s"
        % (cfgA, best[1][1], out["config_A"]["bar_met"]))

    codec, crep = codec_cache[cfgA["latent"]]
    crep.pop("loss_curve", None)
    out["frozen_codec"] = crep
    Ztr = A.codec_encode(codec, Ctr)

    # pipeline ceiling: encode -> decode -> exact judge, on the TRAIN rows
    ceil_em = T.decode_codes(K_COLOURS, A.codec_decode(codec, Ztr))
    out["pipeline_ceiling_codec_roundtrip_on_train"] = A.score_rows(
        tr_insts, train_rows, ceil_em, "codec ceiling (train)")
    Zva_all = {name: A.codec_encode(codec, A.build_codes(rs, K_COLOURS))
               for name, rs in va_split.items() if rs}
    out["pipeline_ceiling_codec_roundtrip_on_val"] = {
        name: A.score_rows(va_insts, va_split[name],
                           T.decode_codes(K_COLOURS, A.codec_decode(codec, Z)),
                           "codec ceiling (%s)" % name)
        for name, Z in Zva_all.items()}
    log("  codec ceiling train %.4f"
        % out["pipeline_ceiling_codec_roundtrip_on_train"]["exact"])

    # -------------------------------------------------------- the CLU + controls
    log("[seed %d] CLU + charter-C controls" % seed_index)
    clu = A.TransitionClu(Ztr, Wtr, grids, K=cfgA["K"], n_atoms=cfgA["n_atoms"],
                          ridge=cfgA["ridge"], seed=seed_index)
    out["clu_ledger"] = clu.ledger()
    out["clu_request_only"] = {"assert_head_request_only": bool(
        clu.assert_request_only(Wva_seen) is not False)}
    controls = {}
    for c in A.TransitionClu.CONTROLS:
        controls[c] = A.TransitionClu(Ztr, Wtr, grids, K=cfgA["K"],
                                      n_atoms=cfgA["n_atoms"], ridge=cfgA["ridge"],
                                      control=c, seed=seed_index)

    # ------------------------------------------- the LEARNED composition-blind arms
    #: ⭐ The BLIND-* planner columns are rule-KNOWING relaxed oracles: they read the
    #: grid, never train, and so never have to generalise.  They are an upper bound on a
    #: composition-blind policy, not a policy.  These two ARE policies -- the same store,
    #: the same head machinery, the same rows, the same codec -- degraded in exactly one
    #: way each.  They are the falsifier the task file names.
    log("[seed %d] learned composition-blind arms" % seed_index)
    cb_grids = [np.arange(c, dtype=np.float64)
                for c in A.colour_blind_alphabets(K_COLOURS)]
    blind_arms = {
        "BLIND-COLOUR (learned: colour identity removed from the request)":
            (A.TransitionClu(Ztr, A.colour_blind_words(Wtr, K_COLOURS), cb_grids,
                             K=cfgA["K"], n_atoms=cfgA["n_atoms"],
                             ridge=cfgA["ridge"], seed=seed_index),
             lambda W: A.colour_blind_words(W, K_COLOURS)),
        "BLIND-SEQUENTIAL (learned: main effects only, no interaction terms)":
            (A.TransitionClu(Ztr, Wtr, grids, K=cfgA["K"], n_atoms=cfgA["n_atoms"],
                             ridge=cfgA["ridge"], pair_block=False, seed=seed_index),
             lambda W: W),
    }
    out["learned_blind_arms"] = {
        name: {"read_floats": obj.ledger()["read_floats"],
               "design_width": obj.ledger()["design_width"]}
        for name, (obj, _f) in blind_arms.items()}

    # ------------------------------------------------------------- the rivals
    log("[seed %d] matched-parameter rivals" % seed_index)
    target = int(out["clu_ledger"]["read_floats"])
    od = A.obs_dim(int(tr_feats[0].shape[0]), K_COLOURS)
    widths = A.match_widths(od, T.code_dim(K_COLOURS), target, tmax=SEQ_T)
    out["matched_parameters"] = {
        "target_floats": target,
        "target_definition": ("the CLU's READ-PATH float count = store atom floats + "
                              "address-book floats.  The float64 sufficient statistic "
                              "is printed separately and is NEVER matched."),
        "achieved": {k: {"width": int(w), "params": int(p),
                         "residual_pct": 100.0 * (p - target) / max(target, 1),
                         "extra": {a: b for a, b in kw.items() if a != "tmax"}}
                     for k, (w, p, kw) in widths.items()}}
    Xs, Ys, Ms = single_sequences(train_rows, tr_insts, tr_feats, tr_g)
    Xt, Yt, Mt = trajectory_sequences(tr_insts, tr_feats, tr_g, rng, per_maze=6)
    rivals, rival_reps = {}, {}
    for kind, (w, p, kw) in widths.items():
        rivals[kind], rival_reps[kind] = train_rival(
            kind, w, kw, Xs, Ys, Ms, Xt, Yt, Mt, seed_index, rival_steps, kind)
    out["rival_training"] = rival_reps

    # ---------------------------------------------- the CAPACITY DIAGNOSTIC
    #: ⚠ NOT a rival and NOT part of the naive-vs-naive table.  The matched rivals are
    #: small because the CLU's READ PATH is small (a few thousand floats), and the first
    #: question a referee asks is whether the CLU only wins because the rival was
    #: starved.  These are the same three trunks at TEN TIMES the read-path budget,
    #: same observation, same frozen encoder, same rows, same gradient steps.
    #: ⛔ This is not the forbidden "inflated-rival row": that prohibition is against
    #: matching a rival to the CLU's MEMORY LEDGER, which is never done here -- the
    #: ledger is printed separately and never matched.
    log("[seed %d] capacity diagnostic (rivals at 10x the read-path budget)"
        % seed_index)
    big_target = 10 * target
    big_widths = A.match_widths(od, T.code_dim(K_COLOURS), big_target, tmax=SEQ_T,
                                hi=400, verbose=True)
    big_rivals, big_reps = {}, {}
    for kind, (w, p, kw) in big_widths.items():
        big_rivals[kind], big_reps[kind] = train_rival(
            kind, w, kw, Xs, Ys, Ms, Xt, Yt, Mt, seed_index, rival_steps,
            "10x " + kind)
    out["capacity_diagnostic"] = {
        "target_floats": int(big_target),
        "achieved": {k: {"width": int(w), "params": int(p)}
                     for k, (w, p, kw) in big_widths.items()},
        "training": big_reps,
        "WHAT_THIS_IS": ("NOT a rival and NOT part of the naive-vs-naive table.  The "
                         "same three trunks at 10x the CLU's read-path budget, to show "
                         "whether the matched rivals were starved.  The CLU's memory "
                         "ledger is never matched by anything.")}

    # -------------------------------------------------------------- the floors
    log("[seed %d] lookup floors" % seed_index)
    fallback = A.majority_code(train_rows, K_COLOURS)
    raw_key_tr = A.make_raw_key(tr_insts)
    raw_key_va = A.make_raw_key(va_insts)
    frag_key_tr = A.make_frag_key(tr_insts)
    frag_key_va = A.make_frag_key(va_insts)
    tab_words = A.build_lookup(train_rows, A.words_key)
    tab_raw = A.build_lookup(train_rows, raw_key_tr)
    tab_frag = A.build_lookup(train_rows, frag_key_tr)
    #: the WITHIN-MAZE column: the table is built from the VAL mazes' OWN rows.
    #: ⛔ Declared NOT transfer -- it is the number a lookup gets when the maze repeats,
    #: printed so the cross-maze floor can be read against it.
    tab_raw_within = A.build_lookup(va_nat, raw_key_va)
    out["lookup_tables"] = {
        "cross_maze_words": {"size": len(tab_words)},
        "cross_maze_raw_r1": {"size": len(tab_raw)},
        "cross_maze_frag_r1": {"size": len(tab_frag)},
        "within_maze_raw_r1": {
            "size": len(tab_raw_within),
            "built_from": ("the VAL mazes' NATURAL-distribution rows -- a different "
                           "sample of the same mazes, so the scored row is not in the "
                           "table by construction"),
            "WARNING": ("this is NOT transfer.  It is printed only as the ceiling a "
                        "lookup reaches when the maze REPEATS, so the cross-maze floor "
                        "can be read against it.  Every scored maze in this harness is "
                        "a random maze never seen at train.")}}

    # ---------------------------------------------------------------- evaluate
    log("[seed %d] evaluate" % seed_index)
    t_eval = time.time()
    rows_out = {}
    for name in ("seen", "pair", "nway", "value"):
        rs = va_split.get(name, [])
        if not rs:
            rows_out[name] = {"DECLARED_NOT_RUN": "no val rows in this class"}
            continue
        W = A.build_words(rs)
        O = A.build_obs(rs, va_insts, va_feats, va_g)
        block = {}
        em = T.decode_codes(K_COLOURS, A.codec_decode(codec, clu.read(W)))
        block["CLU (single design)"] = A.score_rows(va_insts, rs, em)
        for cname, cobj in controls.items():
            emc = T.decode_codes(K_COLOURS, A.codec_decode(codec, cobj.read(W)))
            block["control: " + cname] = A.score_rows(va_insts, rs, emc)
        for bname, (bobj, fn) in blind_arms.items():
            emb = T.decode_codes(K_COLOURS, A.codec_decode(codec, bobj.read(fn(W))))
            block["arm: " + bname] = A.score_rows(va_insts, rs, emb)
        for kind, pol in rivals.items():
            block["rival: " + kind] = A.score_rows(
                va_insts, rs, A.policy_predict(pol, O, K_COLOURS))
        for kind, pol in big_rivals.items():
            block["diagnostic: %s at 10x read floats (NOT a matched rival)" % kind] =                 A.score_rows(va_insts, rs, A.policy_predict(pol, O, K_COLOURS))
        for lname, (keyfn, tab) in (
                ("floor: lookup cross-maze WORDS", (A.words_key, tab_words)),
                ("floor: lookup cross-maze raw r1", (raw_key_va, tab_raw)),
                ("floor: fragment cross-maze r1", (frag_key_va, tab_frag)),
                ("floor: lookup WITHIN-maze raw r1", (raw_key_va, tab_raw_within))):
            pred, info = A.lookup_predict(rs, keyfn, tab, fallback)
            block[lname] = dict(A.score_rows(va_insts, rs, pred), **info)
        for mode in T.BLIND_MODES:
            pred = [T.blind_outcome(va_insts[r.maze], r.state(), r.action, mode)
                    for r in rs]
            block["column: %s (planner)" % mode.upper()] = A.score_rows(
                va_insts, rs, pred)
        block["column: CODEC CEILING"] = out[
            "pipeline_ceiling_codec_roundtrip_on_val"].get(name, {})
        rows_out[name] = block
        log("  row %-6s n=%-5d CLU %.4f  blind-colour %.4f  blind-seq %.4f  "
            "lookup-words %.4f"
            % (name, len(rs), block["CLU (single design)"]["exact"],
               block["column: BLIND-COLOUR (planner)"]["exact"],
               block["column: BLIND-SEQUENTIAL (planner)"]["exact"],
               block["floor: lookup cross-maze WORDS"]["exact"]))
    out["ROWS"] = rows_out
    out["evaluate_seconds"] = time.time() - t_eval
    log("  evaluate done [%.1fs]" % out["evaluate_seconds"])

    # ------------------------------------------------------------ rollout row
    log("[seed %d] rollout row (FULL oracle routes, chained GENERATED states)"
        % seed_index)
    t0 = time.time()
    rollouts = [r for r in (T.rollouts_for_maze(i, m)
                            for m, i in enumerate(va_insts)) if r]

    def _clu_batch(obj, fn):
        def pb(mz, sts, acts):
            W = fn(np.array([T.words_of(va_insts[m], st, a)
                             for m, st, a in zip(mz, sts, acts)], np.float64))
            return T.decode_codes(K_COLOURS, A.codec_decode(codec, obj.read(W)))
        return pb

    def _rival_batch(pol):
        def pb(mz, sts, acts):
            O = np.stack([A.obs_of(va_insts[m], va_feats[m], st, a, va_g[m])
                          for m, st, a in zip(mz, sts, acts)]).astype(np.float32)
            return A.policy_predict(pol, O, K_COLOURS)
        return pb

    def _lookup_batch(keyfn, tab):
        def pb(mz, sts, acts):
            out_rows = []
            for m, st, a in zip(mz, sts, acts):
                row = T.TRow(m, "", st.pos, tuple(st.ks), a,
                             T.words_of(va_insts[m], st, a), 0, 0, 0, 0, False)
                out_rows.append(tab.get(keyfn(row), fallback))
            return np.array(out_rows, np.int64)
        return pb

    def _blind_batch(mode):
        def pb(mz, sts, acts):
            return np.array([T.blind_outcome(va_insts[m], st, a, mode)
                             for m, st, a in zip(mz, sts, acts)], np.int64)
        return pb

    roll = {}
    roll["CLU (single design)"] = A.rollout_validity(
        va_insts, rollouts, _clu_batch(clu, lambda W: W))
    for cname, cobj in controls.items():
        roll["control: " + cname] = A.rollout_validity(
            va_insts, rollouts, _clu_batch(cobj, lambda W: W))
    for bname, (bobj, fn) in blind_arms.items():
        roll["arm: " + bname] = A.rollout_validity(
            va_insts, rollouts, _clu_batch(bobj, fn))
    for kind, pol in rivals.items():
        roll["rival: " + kind] = A.rollout_validity(
            va_insts, rollouts, _rival_batch(pol))
    for kind, pol in big_rivals.items():
        roll["diagnostic: %s at 10x read floats (NOT a matched rival)" % kind] =             A.rollout_validity(va_insts, rollouts, _rival_batch(pol))
    for lname, (keyfn, tab) in (
            ("floor: lookup cross-maze WORDS", (A.words_key, tab_words)),
            ("floor: lookup cross-maze raw r1", (raw_key_va, tab_raw)),
            ("floor: fragment cross-maze r1", (frag_key_va, tab_frag)),
            ("floor: lookup WITHIN-maze raw r1", (raw_key_va, tab_raw_within))):
        roll[lname] = A.rollout_validity(va_insts, rollouts,
                                         _lookup_batch(keyfn, tab))
    for mode in T.BLIND_MODES:
        roll["column: %s (planner)" % mode.upper()] = A.rollout_validity(
            va_insts, rollouts, _blind_batch(mode))
    out["ROLLOUT"] = {
        "n_rollouts": len(rollouts),
        "route_length": "the FULL optimal route per maze (mean %.1f steps)"
                        % float(np.mean([len(r["actions"]) for r in rollouts])),
        "action_source": ("the engine's own optimal route; no arm chooses the actions, "
                          "so a rollout generated correctly at every step reaches the "
                          "goal within budget by construction"),
        "chaining": ("the state fed to step t+1 is the ARM'S OWN generated state, never "
                     "the engine's; the moment a step is wrong the rollout is dead and "
                     "nothing re-synchronises it"),
        "seconds": time.time() - t0,
        "arms": roll}
    roll8 = {}
    short = [dict(r, actions=r["actions"][:8]) for r in rollouts]
    for nm, pb in (("CLU (single design)", _clu_batch(clu, lambda W: W)),
                   ("rival: mlp", _rival_batch(rivals["mlp"])),
                   ("floor: lookup cross-maze WORDS",
                    _lookup_batch(A.words_key, tab_words)),
                   ("column: BLIND-COLOUR (planner)", _blind_batch("blind-colour")),
                   ("column: BLIND-SEQUENTIAL (planner)",
                    _blind_batch("blind-sequential"))):
        roll8[nm] = A.rollout_validity(va_insts, short, pb)
    out["ROLLOUT_L8_DIAGNOSTIC"] = {
        "L": 8, "arms": roll8,
        "note": ("a DIAGNOSTIC, never the scored rollout row: an 8-step prefix of a "
                 "~45-step optimal route cannot reach the goal, so `validity` is 0 for "
                 "every arm by construction and the readable number is "
                 "`all_steps_exact_rate`.")}
    log("  rollout CLU validity %.4f (per-step %.4f, mean len %.1f)  [%.1fs]"
        % (roll["CLU (single design)"]["validity"],
           roll["CLU (single design)"]["per_step_exactness"],
           roll["CLU (single design)"]["mean_rollout_length"], time.time() - t0))

    # ------------------------------------------------------------ deletion rows
    log("[seed %d] deletion rows" % seed_index)
    t_del = time.time()
    out["DELETION"] = deletion_block(train_rows, tr_insts, tr_feats, tr_g, Wtr, Ztr,
                                     Ctr, codec, grids, cfgA, va_insts, va_split,
                                     va_feats, va_g, widths, Xt, Yt, Mt, seed_index,
                                     rival_steps)

    out["DELETION"]["seconds"] = time.time() - t_del
    out["seconds_total"] = time.time() - t_start
    p = os.path.join(out_dir, "val_seed%d.json" % seed_index)
    with open(p, "w") as fh:
        json.dump(out, fh, indent=1, default=_json_default)
    log("[seed %d] wrote %s  [%.1f s]" % (seed_index, p, out["seconds_total"]))
    return p


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


# ---------------------------------------------------------------- deletion
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


if __name__ == "__main__":
    si = int(sys.argv[1])
    od = sys.argv[2]
    kwargs = {}
    for i in range(3, len(sys.argv), 2):
        kwargs[sys.argv[i].lstrip("-").replace("-", "_")] = int(sys.argv[i + 1])
    run(si, od, **kwargs)
