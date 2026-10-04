"""CMAZE-T with asymmetric inputs: configuration sweep on VAL and pinned full runs.

python scripts/cmazet_asymmetric.py sweep <seed> <out_dir>
python scripts/cmazet_asymmetric.py full <seed> <out_dir> --leg val|test --latent 128 --K 2 --atoms 6
"""

from __future__ import annotations
import json
import os
import sys
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import nclu.cmaze.task as T
import nclu.cmaze.arms as A
import cmazet_lib as R

K_COLOURS = R.K_COLOURS
SPAN_CAP = 0.3
SWEEP_LATENT = (64, 128, 256)
SWEEP_K = (2, 4, 8, 16)
SWEEP_ATOMS = (1, 2, 3, 4, 6, 8)


def log(*a):
    print(*a, flush=True)


def match_bits(words, k=K_COLOURS):
    W = np.asarray(words, np.int64)
    a, d, t = (W[:, 0], W[:, 2], W[:, 3])
    m_door = ((a > 0) & (d >= 1) & (d <= k) & (a == d)).astype(np.int64)
    m_tint = ((a > 0) & (t > 0) & (a == t)).astype(np.int64)
    return np.stack([m_door, m_tint], axis=1)


class Iface:
    """One interface object; every arm goes through it."""

    def __init__(self, kind):
        assert kind in ("none", "match"), kind
        self.kind = kind

    def words(self, W):
        W = np.asarray(W, np.float64)
        if self.kind == "none":
            return W
        return np.concatenate([W, match_bits(W).astype(np.float64)], axis=1)

    def grids(self, base_grids):
        if self.kind == "none":
            return base_grids
        return list(base_grids) + [np.arange(2, dtype=np.float64)] * 2

    def blind_words(self, W):
        return A.colour_blind_words(W, K_COLOURS)

    def obs(self, O, W):
        if self.kind == "none":
            return O
        return np.concatenate([O, match_bits(W).astype(np.float32)], axis=-1)

    def key(self, keyfn):
        if self.kind == "none":
            return keyfn

        def k2(r):
            mb = tuple(match_bits(np.array([r.words]))[0].tolist())
            return (keyfn(r), mb)

        return k2


def derangement(K, rng):
    while True:
        p = rng.permutation(K)
        if K < 2 or not np.any(p == np.arange(K)):
            return p


def make_control(Ztr, Wtr, grids, K, n_atoms, ridge, control, seed, pair_block=True):
    if control != "landscape_shuffled_across_wells":
        return A.TransitionClu(
            Ztr,
            Wtr,
            grids,
            K=K,
            n_atoms=n_atoms,
            ridge=ridge,
            control=control,
            seed=seed,
            pair_block=pair_block,
        )
    obj = A.TransitionClu(
        Ztr, Wtr, grids, K=K, n_atoms=n_atoms, ridge=ridge, seed=seed, pair_block=pair_block
    )
    rng = np.random.default_rng(seed + 991)
    pool = [np.asarray(M) for M in obj.store.modes]
    perm = derangement(obj.K, rng)
    new = []
    for j, ch in enumerate(obj.well_channels):
        src = pool[perm[j]]
        d = int(ch.size)
        M = np.zeros((obj.n_atoms, d), np.float32)
        w = min(d, src.shape[1])
        M[:, :w] = src[:, :w]
        new.append(M)
    obj.store.modes = new
    obj.control = control + " (deranged)"
    obj.fit_head(np.asarray(Ztr, np.float32), Wtr)
    obj.derangement = [int(x) for x in perm]
    return obj


def banks(seed_index, leg, n_train_mazes=90, n_eval_mazes=45, per_tuple=2):
    base = 2000000 + 400000 * seed_index
    tr = T.build_bank(base, n_train_mazes, per_tuple=per_tuple, n_natural=150)
    if leg == "val":
        ev_base = base + 200000
    elif leg == "test":
        ev_base = base + 300000
    else:
        raise ValueError(leg)
    ev = T.build_bank(ev_base, n_eval_mazes, per_tuple=per_tuple, n_natural=150)
    return (base, ev_base, tr, ev)


def cap_split(rows_split, seed_index, max_rows_per_class=4000):
    cap_rng = np.random.default_rng(777 + seed_index)
    out = {}
    for nm, rs in rows_split.items():
        if len(rs) > max_rows_per_class:
            idx = cap_rng.permutation(len(rs))[:max_rows_per_class]
            out[nm] = [rs[int(i)] for i in sorted(idx)]
        else:
            out[nm] = rs
    return out


def sc(insts, rows, em):
    return float(A.score_rows(insts, rows, em)["exact"])


def sweep(seed_index, out_dir, iface="none", codec_steps=2500, configC=False):
    t0 = time.time()
    IF = Iface(iface)
    base, ev_base, (tr_insts, tr_rows, tr_nat, tr_meta), (va_insts, va_rows, va_nat, va_meta) = (
        banks(seed_index, "val")
    )
    ho = T.make_holdout(K_COLOURS, seed=seed_index)
    train_rows = T.split_rows(tr_rows, ho, K_COLOURS).get("seen", [])
    va_split = cap_split(T.split_rows(va_rows, ho, K_COLOURS), seed_index)
    Ctr = A.build_codes(train_rows, K_COLOURS)
    Ccodec = np.concatenate([Ctr, T.enumerate_code_alphabet(K_COLOURS)], axis=0)
    Wtr_raw = A.build_words(train_rows)
    Wtr = IF.words(Wtr_raw)
    grids = IF.grids([np.arange(c, dtype=np.float64) for c in T.alphabets(K_COLOURS)])
    cb_grids = [np.arange(c, dtype=np.float64) for c in A.colour_blind_alphabets(K_COLOURS)]
    Wva = {n: A.build_words(rs) for n, rs in va_split.items() if rs}
    res = {
        "seed_index": seed_index,
        "iface": iface,
        "leg": "val",
        "n_val_rows": {n: len(rs) for n, rs in va_split.items()},
        "n_train_rows": len(train_rows),
        "grid": [],
    }
    log(
        "[sweep seed %d iface %s] train %d rows, val %s"
        % (seed_index, iface, len(train_rows), res["n_val_rows"])
    )
    spaces = [("code", None), (512, None)] if configC else [(L, None) for L in SWEEP_LATENT]
    for lat, _ in spaces:
        if lat == "code":
            dim = int(Ctr.shape[1])
            Ztr = Ctr.astype(np.float32)
            dec = lambda Zr: T.decode_codes(K_COLOURS, np.asarray(Zr))
            Ks, ATs = ((2, 3, 4, 7), (1, 2, 3))
        else:
            dim = lat
            codec, crep = A.train_codec(
                Ccodec, K_COLOURS, latent=lat, steps=codec_steps, seed=seed_index
            )
            Ztr = A.codec_encode(codec, Ctr)
            dec = (lambda c: lambda Zr: T.decode_codes(K_COLOURS, A.codec_decode(c, Zr)))(codec)
            Ks, ATs = (SWEEP_K, SWEEP_ATOMS + ((12, 16) if configC else ()))

        def ev(obj, wf):
            return {n: sc(va_insts, va_split[n], dec(obj.read(wf(W)))) for n, W in Wva.items()}

        for K in Ks:
            for na in ATs:
                span = na / (dim / K)
                if span > SPAN_CAP + 1e-09:
                    continue
                cfg = dict(latent=lat, K=K, n_atoms=na, ridge=1.0)
                if lat == "code":
                    cfg["latent"] = "code"
                try:
                    clu = A.TransitionClu(
                        Ztr, Wtr, grids, K=K, n_atoms=na, ridge=1.0, seed=seed_index
                    )
                except AssertionError as exc:
                    res["grid"].append(dict(cfg, DECLARED_NOT_RUN=str(exc)))
                    continue
                row = dict(
                    cfg,
                    span=span,
                    read_floats=clu.footprint()["read_floats"],
                    footprint=clu.footprint(),
                )
                row["CLU"] = ev(clu, IF.words)
                row["controls"] = {}
                for c in A.TransitionClu.CONTROLS:
                    cobj = make_control(Ztr, Wtr, grids, K, na, 1.0, c, seed_index)
                    row["controls"][c] = ev(cobj, IF.words)
                bc = A.TransitionClu(
                    Ztr,
                    A.colour_blind_words(Wtr_raw, K_COLOURS),
                    cb_grids,
                    K=K,
                    n_atoms=na,
                    ridge=1.0,
                    seed=seed_index,
                )
                bs = A.TransitionClu(
                    Ztr, Wtr, grids, K=K, n_atoms=na, ridge=1.0, pair_block=False, seed=seed_index
                )
                row["blind_colour"] = ev(bc, lambda W: A.colour_blind_words(W, K_COLOURS))
                row["blind_seq"] = ev(bs, IF.words)
                res["grid"].append(row)
                log(
                    "  lat %3d K %2d at %d span %.3f rf %6d | CLU seen %.3f pair %.3f nway %.3f value %.3f | worst ctl nway %.3f | bseq nway %.3f"
                    % (
                        dim,
                        K,
                        na,
                        span,
                        row["read_floats"],
                        row["CLU"].get("seen", -1),
                        row["CLU"].get("pair", -1),
                        row["CLU"].get("nway", -1),
                        row["CLU"].get("value", -1),
                        max((v.get("nway", -1) for v in row["controls"].values())),
                        row["blind_seq"].get("nway", -1),
                    )
                )
    res["seconds"] = time.time() - t0
    p = os.path.join(
        out_dir, "sweep_%s_am1%s_seed%d.json" % (iface, "_C" if configC else "", seed_index)
    )
    with open(p, "w") as fh:
        json.dump(res, fh, indent=1, default=R._json_default)
    log("wrote %s [%.1fs]" % (p, res["seconds"]))


def full(
    seed_index,
    out_dir,
    leg,
    latent,
    K,
    n_atoms,
    ridge=1.0,
    iface="none",
    tag="A",
    enc_steps=1500,
    codec_steps=2500,
    rival_steps=8000,
):
    t_start = time.time()
    IF = Iface(iface)
    rng = np.random.default_rng(10000 + seed_index)
    cfgA = dict(latent=int(latent), K=int(K), n_atoms=int(n_atoms), ridge=float(ridge))
    out = {
        "seed_index": seed_index,
        "leg": leg,
        "config": cfgA,
        "iface": iface,
        "tag": tag,
        "tolerance_used": None,
        "k_colours": K_COLOURS,
        "scored_run": leg == "test",
    }
    log("[%s seed %d leg %s] config %s iface %s" % (tag, seed_index, leg, cfgA, iface))
    base, ev_base, (tr_insts, tr_rows, tr_nat, tr_meta), (va_insts, va_rows, va_nat, va_meta) = (
        banks(seed_index, leg)
    )
    out["banks"] = {
        "train": dict(tr_meta, seed_block=[base, base + 200000]),
        leg: dict(va_meta, seed_block=[ev_base, ev_base + 100000]),
    }
    all_insts = tr_insts + va_insts
    all_rows = list(tr_rows) + [
        T.TRow(
            r.maze + len(tr_insts),
            r.prefix,
            r.pos,
            r.ks,
            r.action,
            r.words,
            r.legal,
            r.disp,
            r.ev,
            r.pocket_next,
            r.bites_b,
        )
        for r in va_rows
    ]
    out["ACCEPT_1_independent_replay"] = T.replay_audit(all_insts, all_rows)
    out["ACCEPT_2_no_impossible_example"] = T.impossible_example_audit(all_insts, all_rows)
    ho = T.make_holdout(K_COLOURS, seed=seed_index)
    tr_split = T.split_rows(tr_rows, ho, K_COLOURS)
    va_split_full = T.split_rows(va_rows, ho, K_COLOURS)
    va_split = cap_split(va_split_full, seed_index)
    train_rows = tr_split.get("seen", [])
    out["holdout"] = {
        "value_colour": ho.value_colour,
        "pairs": [list(p) for p in ho.pairs],
        "nway": [list(c) for c in ho.nway],
        "n_train_rows": len(train_rows),
        "n_eval_rows_before_cap": {k: len(v) for k, v in va_split_full.items()},
        "n_eval_rows_scored": {k: len(v) for k, v in va_split.items()},
    }
    out["ACCEPT_3_holdout_disjointness"] = T.disjointness_report(
        train_rows, {k: v for k, v in va_split.items() if k != "seen"}, ho, K_COLOURS
    )
    log(
        "  train(seen) %d rows; %s rows %s"
        % (len(train_rows), leg, out["holdout"]["n_eval_rows_scored"])
    )
    Ctr = A.build_codes(train_rows, K_COLOURS)
    Ccodec = np.concatenate([Ctr, T.enumerate_code_alphabet(K_COLOURS)], axis=0)
    codec, crep = A.train_codec(
        Ccodec, K_COLOURS, latent=cfgA["latent"], steps=codec_steps, seed=seed_index
    )
    crep.pop("loss_curve", None)
    out["frozen_codec"] = crep
    enc, enc_rep = A.train_state_encoder(
        tr_insts, steps=enc_steps, seed=seed_index, verbose=True, restarts=3, accept_bce=0.01
    )
    enc_rep.pop("loss_curve", None)
    out["frozen_state_encoder"] = {
        k: v for k, v in enc_rep.items() if not isinstance(v, np.ndarray)
    }
    tr_feats = A.feature_maps(enc, tr_insts)
    va_feats = A.feature_maps(enc, va_insts)
    from nclu.cmaze.harness.features import global_summary

    tr_g = [global_summary(S) for S in tr_feats]
    va_g = [global_summary(S) for S in va_feats]
    Ztr = A.codec_encode(codec, Ctr)
    Wtr_raw = A.build_words(train_rows)
    Wtr = IF.words(Wtr_raw)
    grids = IF.grids([np.arange(c, dtype=np.float64) for c in T.alphabets(K_COLOURS)])
    clu = A.TransitionClu(
        Ztr, Wtr, grids, K=cfgA["K"], n_atoms=cfgA["n_atoms"], ridge=cfgA["ridge"], seed=seed_index
    )
    out["clu_footprint"] = clu.footprint()
    out["rank_over_width_printed"] = {
        "per_well": clu.footprint()["span_fraction_n_atoms_over_d_k"],
        "max": max(clu.footprint()["span_fraction_n_atoms_over_d_k"]),
        "cap": SPAN_CAP,
    }
    out["clu_request_only"] = bool(
        clu.assert_request_only(IF.words(A.build_words(va_split["seen"]))) is not False
    )
    controls = {
        c: make_control(Ztr, Wtr, grids, cfgA["K"], cfgA["n_atoms"], cfgA["ridge"], c, seed_index)
        for c in A.TransitionClu.CONTROLS
    }
    out["across_wells_derangement"] = controls["landscape_shuffled_across_wells"].derangement
    cb_grids = [np.arange(c, dtype=np.float64) for c in A.colour_blind_alphabets(K_COLOURS)]
    blind_arms = {
        "BLIND-COLOUR (learned: colour identity removed from the request)": (
            A.TransitionClu(
                Ztr,
                IF.blind_words(Wtr_raw),
                cb_grids,
                K=cfgA["K"],
                n_atoms=cfgA["n_atoms"],
                ridge=cfgA["ridge"],
                seed=seed_index,
            ),
            IF.blind_words,
        ),
        "BLIND-SEQUENTIAL (learned: main effects only, no interaction terms)": (
            A.TransitionClu(
                Ztr,
                Wtr,
                grids,
                K=cfgA["K"],
                n_atoms=cfgA["n_atoms"],
                ridge=cfgA["ridge"],
                pair_block=False,
                seed=seed_index,
            ),
            IF.words,
        ),
    }
    target = int(out["clu_footprint"]["read_floats"])
    od = A.obs_dim(int(tr_feats[0].shape[0]), K_COLOURS) + (2 if iface == "match" else 0)
    widths = A.match_widths(od, T.code_dim(K_COLOURS), target, tmax=R.SEQ_T)
    out["matched_parameters"] = {
        "target_floats": target,
        "achieved": {
            k: {
                "width": int(w),
                "params": int(p),
                "residual_pct": 100.0 * (p - target) / max(target, 1),
            }
            for k, (w, p, kw) in widths.items()
        },
    }
    Xs, Ys, Ms = R.single_sequences(train_rows, tr_insts, tr_feats, tr_g)
    if iface == "match":
        Xs = np.concatenate([Xs, match_bits(Wtr_raw)[:, None, :].astype(np.float32)], axis=-1)
    Xt, Yt, Mt = R.trajectory_sequences(tr_insts, tr_feats, tr_g, rng, per_maze=6)
    if iface == "match":
        Xt = trajectory_with_match(tr_insts, tr_feats, tr_g, Xt, rng_seed=10000 + seed_index)
    rivals, reps = ({}, {})
    out["rival_steps_matched"] = int(rival_steps)
    out["rival_steps_10x_diagnostic"] = 8000
    for kind, (w, p, kw) in widths.items():
        rivals[kind], reps[kind] = R.train_rival(
            kind, w, kw, Xs, Ys, Ms, Xt, Yt, Mt, seed_index, rival_steps, kind
        )
    out["rival_training"] = reps
    big_widths = A.match_widths(od, T.code_dim(K_COLOURS), 10 * target, tmax=R.SEQ_T, hi=400)
    big = {}
    for kind, (w, p, kw) in big_widths.items():
        big[kind], _r = R.train_rival(
            kind, w, kw, Xs, Ys, Ms, Xt, Yt, Mt, seed_index, 8000, "10x " + kind
        )
    out["capacity_diagnostic"] = {
        k: {"width": int(w), "params": int(p)} for k, (w, p, kw) in big_widths.items()
    }
    fallback = A.majority_code(train_rows, K_COLOURS)
    raw_tr, raw_va = (IF.key(A.make_raw_key(tr_insts)), IF.key(A.make_raw_key(va_insts)))
    frag_tr, frag_va = (IF.key(A.make_frag_key(tr_insts)), IF.key(A.make_frag_key(va_insts)))
    wkey = IF.key(A.words_key)
    tabs = {
        "floor: lookup cross-maze WORDS": (wkey, A.build_lookup(train_rows, wkey)),
        "floor: lookup cross-maze raw r1": (raw_va, A.build_lookup(train_rows, raw_tr)),
        "floor: fragment cross-maze r1": (frag_va, A.build_lookup(train_rows, frag_tr)),
        "floor: lookup WITHIN-maze raw r1 (NOT transfer)": (raw_va, A.build_lookup(va_nat, raw_va)),
    }
    rows_out = {}
    for name in ("seen", "pair", "nway", "value"):
        rs = va_split.get(name, [])
        if not rs:
            rows_out[name] = {"DECLARED_NOT_RUN": "no rows in this class"}
            continue
        Wr = A.build_words(rs)
        O = IF.obs(A.build_obs(rs, va_insts, va_feats, va_g), Wr)
        b = {"n": len(rs)}

        def dec(obj, wf):
            return T.decode_codes(K_COLOURS, A.codec_decode(codec, obj.read(wf(Wr))))

        b["CLU"] = sc(va_insts, rs, dec(clu, IF.words))
        for c, obj in controls.items():
            b["control: " + c] = sc(va_insts, rs, dec(obj, IF.words))
        for nm, (obj, wf) in blind_arms.items():
            b["arm: " + nm] = sc(va_insts, rs, dec(obj, wf))
        for kind, pol in rivals.items():
            b["rival: " + kind] = sc(va_insts, rs, A.policy_predict(pol, O, K_COLOURS))
        for kind, pol in big.items():
            b["diagnostic: %s at 10x read floats (NOT a matched rival)" % kind] = sc(
                va_insts, rs, A.policy_predict(pol, O, K_COLOURS)
            )
        for ln, (kf, tab) in tabs.items():
            pred, _info = A.lookup_predict(rs, kf, tab, fallback)
            b[ln] = sc(va_insts, rs, pred)
        for mode in T.BLIND_MODES:
            pred = [T.blind_outcome(va_insts[r.maze], r.state(), r.action, mode) for r in rs]
            b["column: %s (planner)" % mode.upper()] = sc(va_insts, rs, pred)
        rows_out[name] = b
        best_r = max((v for k2, v in b.items() if k2.startswith("rival: ")))
        log(
            "  row %-5s n=%-5d CLU %.4f best-rival %.4f worst-ctl %.4f bseq %.4f"
            % (
                name,
                len(rs),
                b["CLU"],
                best_r,
                max((v for k2, v in b.items() if k2.startswith("control: "))),
                b["arm: BLIND-SEQUENTIAL (learned: main effects only, no interaction terms)"],
            )
        )
    out["ROWS"] = rows_out
    rollouts = [r for r in (T.rollouts_for_maze(i, m) for m, i in enumerate(va_insts)) if r]

    def clu_b(obj, wf):
        def pb(mz, sts, acts):
            W = np.array(
                [T.words_of(va_insts[m], st, a) for m, st, a in zip(mz, sts, acts)], np.float64
            )
            return T.decode_codes(K_COLOURS, A.codec_decode(codec, obj.read(wf(W))))

        return pb

    def riv_b(pol):
        def pb(mz, sts, acts):
            O = np.stack(
                [
                    A.obs_of(va_insts[m], va_feats[m], st, a, va_g[m])
                    for m, st, a in zip(mz, sts, acts)
                ]
            ).astype(np.float32)
            W = np.array(
                [T.words_of(va_insts[m], st, a) for m, st, a in zip(mz, sts, acts)], np.float64
            )
            return A.policy_predict(pol, IF.obs(O, W), K_COLOURS)

        return pb

    roll = {"CLU": A.rollout_validity(va_insts, rollouts, clu_b(clu, IF.words))}
    for c, obj in controls.items():
        roll["control: " + c] = A.rollout_validity(va_insts, rollouts, clu_b(obj, IF.words))
    for kind, pol in rivals.items():
        roll["rival: " + kind] = A.rollout_validity(va_insts, rollouts, riv_b(pol))
    for kind, pol in big.items():
        roll["diagnostic: %s at 10x" % kind] = A.rollout_validity(va_insts, rollouts, riv_b(pol))
    out["ROLLOUT"] = {
        "n_rollouts": len(rollouts),
        "mean_route_len": float(np.mean([len(r["actions"]) for r in rollouts])),
        "arms": roll,
    }
    log(
        "  rollout CLU validity %.4f per-step %.4f"
        % (roll["CLU"]["validity"], roll["CLU"]["per_step_exactness"])
    )
    if iface == "none":
        out["DELETION"] = R.deletion_block(
            train_rows,
            tr_insts,
            tr_feats,
            tr_g,
            Wtr,
            Ztr,
            Ctr,
            codec,
            grids,
            cfgA,
            va_insts,
            va_split,
            va_feats,
            va_g,
            widths,
            Xt,
            Yt,
            Mt,
            seed_index,
            rival_steps,
        )
    else:
        out["DELETION"] = {
            "DECLARED_NOT_RUN": "deletion rows are produced on the winning config only; see LOG.md"
        }
    out["seconds_total"] = time.time() - t_start
    p = os.path.join(out_dir, "%s_%s_seed%d.json" % (tag, leg, seed_index))
    with open(p, "w") as fh:
        json.dump(out, fh, indent=1, default=R._json_default)
    log("[%s seed %d leg %s] wrote %s [%.1fs]" % (tag, seed_index, leg, p, out["seconds_total"]))


def trajectory_with_match(insts, feats, gsums, Xt, rng_seed):
    """Append the two match bits to the rivals' trajectory observations.  The words are
    recomputed from the observation's own trajectory by replaying it with the SAME rng,
    so the bits are exactly the ones the transitions carried."""
    from nclu.cmaze.harness.dynamics import initial_state, key_homes

    rng = np.random.default_rng(rng_seed)
    out = np.zeros(Xt.shape[:2] + (Xt.shape[2] + 2,), np.float32)
    out[:, :, : Xt.shape[2]] = Xt
    i = 0
    for m, inst in enumerate(insts):
        homes = key_homes(inst)
        for _ in range(6):
            st = initial_state(inst)
            for t in range(Xt.shape[1]):
                a = T.ACTIONS[int(rng.integers(5))]
                w = np.array([T.words_of(inst, st, a, homes)], np.float64)
                out[i, t, -2:] = match_bits(w)[0]
                legal, disp, ev, pn, nxt = T.outcome_of(inst, st, a, homes)
                if legal:
                    st = nxt
                    if st.pos == inst.goal:
                        break
            i += 1
    return out


if __name__ == "__main__":
    mode, si, od = (sys.argv[1], int(sys.argv[2]), sys.argv[3])
    kw = {}
    for i in range(4, len(sys.argv), 2):
        kw[sys.argv[i].lstrip("-")] = sys.argv[i + 1]
    os.makedirs(od, exist_ok=True)
    if mode == "sweep":
        sweep(si, od, iface=kw.get("iface", "none"))
    elif mode == "sweepC":
        sweep(si, od, configC=True)
    else:
        full(
            si,
            od,
            kw["leg"],
            int(kw["latent"]),
            int(kw["K"]),
            int(kw["atoms"]),
            float(kw.get("ridge", 1.0)),
            kw.get("iface", "none"),
            kw.get("tag", "A"),
            rival_steps=int(kw.get("rival_steps", 8000)),
        )
