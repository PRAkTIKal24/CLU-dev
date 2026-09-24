"""w19 hub/maze-t-symmetric -- does the CMAZE-T win survive SYMMETRIC inputs?

    python scripts/w19_symmetric.py <seed> <out_dir> --cond WORDS-MAIN|WORDS-FULL|OBS|BOTH
        --leg val|test [--latent 128 --K 2 --atoms 6] [--rival_steps 8000]
        [--diag_mult 10] [--deletion 0|1] [--smoke 0|1]

⛔ NOTHING about the instrument changes.  Engine, solver, judge, banks, hold-outs, codec,
controls (across-wells DERANGED, w16 AMENDMENT 1), blind arms, floors, the rival recipe and
the store config (A1 = latent 128, K 2, 6 atoms) are w15/w16's, imported unmodified.
⭐ The ONLY thing that changes is the INPUT, and within a condition EVERY arm gets the
identical input (``Inputs`` below is the one object every arm goes through):

    WORDS-MAIN  CLU design [bias|main]; rivals the 27 main-effect one-hots.  The CLU is
                then the learned BLIND-SEQUENTIAL arm -- printed as such.
    WORDS-FULL  CLU design [bias|main|pair] (318); rivals the SAME 318-wide vector.
    OBS         rivals: the w9 observation (conv-AE patch + global ⊕ 29 dynamics scalars ⊕
                action), and a second row with the extractor's words appended.  The CLU's
                read stays request-only: a DECLARED word extractor (obs -> five words,
                trained on TRAIN rows only) feeds the request head; its accuracy is
                printed on every row.
    BOTH        rivals obs ⊕ the 318 design; the CLU reads the true words.
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

import chlu.exp_w15_cmaze_t as T                                    # noqa: E402
import chlu.exp_w15_cmaze_t_arms as A                               # noqa: E402
import w15_cmaze_t_run as R                                         # noqa: E402
import w16_cmaze_t as W16                                           # noqa: E402
from chlu.core.gen_request_head import HeadSpec, request_design     # noqa: E402

K_COLOURS = R.K_COLOURS
CONDS = ("WORDS-MAIN", "WORDS-FULL", "OBS", "BOTH")
BSEQ = "BLIND-SEQUENTIAL (learned: main effects only, no interaction terms)"
BCOL = "BLIND-COLOUR (learned: colour identity removed from the request)"

#: w21: the CLU class and its control builder.  Default = w19's A1 CLU, unchanged.  The
#: w21 runner (``--clu_cfg <json>``) swaps in ``w21_clu.CluFactory`` -- ONLY the CLU and
#: the four charter-C controls built from it change; blind arms, rivals, judge, codec,
#: banks, floors are untouched.
CLU_CLS = A.TransitionClu
MAKE_CONTROL = W16.make_control
CLU_CFG = None
#: w21 printed DIAGNOSTIC only: the candidate's colour-relational columns, appended to a
#: copy of each rival's shared input ('+rel').  Labelled 'diagnostic+rel@...', so it is
#: never inside R* (w19_bars reads only 'rival: ' / 'rival@10x-steps: ').
REL_FN = None


def log(*a):
    print(*a, flush=True)


# =================================================================== the inputs
class Inputs:
    """One input object per condition; every arm's input is produced here."""

    def __init__(self, cond):
        assert cond in CONDS, cond
        self.cond = cond
        self.grids = [np.arange(c, dtype=np.float64) for c in T.alphabets(K_COLOURS)]
        self.spec_main = HeadSpec([g.size for g in self.grids], pair_block=False, ridge=1.0)
        self.spec_full = HeadSpec([g.size for g in self.grids], pair_block=True, ridge=1.0)
        #: the CLU's design: main effects only in WORDS-MAIN, the full design otherwise
        self.clu_pair_block = cond != "WORDS-MAIN"
        self.uses_obs = cond in ("OBS", "BOTH")
        self.uses_extractor = cond == "OBS"

    def main27(self, W):
        Phi, _ = request_design(np.asarray(W, np.float64), self.grids, self.spec_main)
        assert Phi.shape[1] == 28
        return np.asarray(Phi[:, 1:], np.float32)

    def full318(self, W):
        Phi, _ = request_design(np.asarray(W, np.float64), self.grids, self.spec_full)
        assert Phi.shape[1] == 318
        return np.asarray(Phi, np.float32)

    def cont10(self, W):
        """w4 ContMLP convention: each word as a min-max-normalised NUMBER + a flag."""
        V = np.asarray(W, np.float64)
        out = np.zeros((V.shape[0], 2 * V.shape[1]), np.float32)
        for f, g in enumerate(self.grids):
            out[:, f] = (V[:, f] - g[0]) / max(g[-1] - g[0], 1e-9)
            out[:, V.shape[1] + f] = 1.0
        return out

    def variants(self):
        """(variant name, rival kinds).  Variant '' is the condition's shared input."""
        if self.cond in ("WORDS-MAIN", "WORDS-FULL"):
            return [("", ("mlp", "lstm", "transformer")), ("cont", ("mlp",))]
        if self.cond == "OBS":
            return [("", ("mlp", "lstm", "transformer")),
                    ("+extractor", ("mlp", "lstm", "transformer"))]
        return [("", ("mlp", "lstm", "transformer"))]

    def rival_x(self, variant, W, O=None, We=None):
        """The rival input for ``variant`` from true words W, obs O, extracted words We."""
        c = self.cond
        if variant == "+rel":
            base = self.rival_x("", W, O, We)
            return np.concatenate([base, REL_FN(W)], axis=1).astype(np.float32)
        if variant == "cont":
            return self.cont10(W)
        if c == "WORDS-MAIN":
            return self.main27(W)
        if c == "WORDS-FULL":
            return self.full318(W)
        if c == "OBS":
            if variant == "":
                return np.asarray(O, np.float32)
            return np.concatenate([O, self.main27(We)], axis=1).astype(np.float32)
        return np.concatenate([O, self.full318(W)], axis=1).astype(np.float32)

    @staticmethod
    def rival_name(kind, variant):
        if variant == "+rel":
            return kind + " +rel"
        if variant == "cont":
            return "cont_mlp"
        return kind + (" " + variant if variant else "")


# ============================================================ the word extractor
class WordExtractor:
    """DECLARED word extractor for OBS: obs -> the five words.  A small MLP classifier
    (obs_dim -> hidden relu -> 27 logits, one softmax per word), trained on TRAIN rows
    only.  Its output is the CLU's request in OBS and is appended to a second rival row,
    so it is not a private channel."""

    def __init__(self, obs_d, hidden=32, seed=0):
        import jax
        k1, k2 = jax.random.split(jax.random.PRNGKey(5150 + seed))
        self.cards = T.alphabets(K_COLOURS)
        self.offs = np.concatenate([[0], np.cumsum(self.cards)[:-1]]).astype(int)
        n_out = int(sum(self.cards))
        s1, s2 = 1.0 / np.sqrt(obs_d), 1.0 / np.sqrt(hidden)
        self.p = {"W1": np.asarray(jax.random.normal(k1, (obs_d, hidden))) * s1,
                  "b1": np.zeros(hidden, np.float32),
                  "W2": np.asarray(jax.random.normal(k2, (hidden, n_out))) * s2,
                  "b2": np.zeros(n_out, np.float32)}
        self.hidden = hidden

    def n_floats(self):
        return int(sum(np.asarray(v).size for v in self.p.values()))

    @staticmethod
    def _fwd(p, X):
        import jax.numpy as jnp
        h = jnp.maximum(X @ p["W1"] + p["b1"], 0.0)
        return h @ p["W2"] + p["b2"]

    def fit(self, O, W, steps=6000, batch=256, lr=3e-3, seed=0):
        import jax
        import jax.numpy as jnp
        import optax
        t0 = time.time()
        Y = np.zeros((len(W), int(sum(self.cards))), np.float32)
        Wi = np.asarray(W, np.int64)
        for f, (o, c) in enumerate(zip(self.offs, self.cards)):
            Y[np.arange(len(W)), o + Wi[:, f]] = 1.0
        offs, cards = [int(x) for x in self.offs], [int(x) for x in self.cards]

        def loss_fn(p, xb, yb):
            lg = self._fwd(p, xb)
            tot = 0.0
            for o, c in zip(offs, cards):
                tot = tot + optax.softmax_cross_entropy(lg[:, o:o + c], yb[:, o:o + c]).mean()
            return tot

        opt = optax.adam(lr)
        p = {k: jnp.asarray(v, jnp.float32) for k, v in self.p.items()}
        st = opt.init(p)

        @jax.jit
        def upd(p, st, xb, yb):
            l, g = jax.value_and_grad(loss_fn)(p, xb, yb)
            u, st = opt.update(g, st, p)
            return optax.apply_updates(p, u), st, l

        rng = np.random.default_rng(6000 + seed)
        O = np.asarray(O, np.float32)
        hist = []
        for s in range(steps):
            idx = rng.integers(0, len(O), size=min(batch, len(O)))
            p, st, l = upd(p, st, jnp.asarray(O[idx]), jnp.asarray(Y[idx]))
            if s % 1000 == 0 or s == steps - 1:
                hist.append((s, float(l)))
        self.p = {k: np.asarray(v) for k, v in p.items()}
        return {"steps": steps, "batch": batch, "lr": lr, "hidden": self.hidden,
                "final_loss": hist[-1][1], "loss_curve": hist, "floats": self.n_floats(),
                "seconds": time.time() - t0, "n_train_rows": int(len(O)),
                "trained_on": "TRAIN (seen) rows only"}

    def predict(self, O):
        O = np.asarray(O, np.float32)
        out = []
        for s in range(0, len(O), 4096):
            lg = np.asarray(self._fwd(self.p, O[s:s + 4096]))
            out.append(np.stack([lg[:, o:o + c].argmax(1)
                                 for o, c in zip(self.offs, self.cards)], axis=1))
        if not out:
            return np.zeros((0, 5), np.float64)
        return np.concatenate(out, 0).astype(np.float64)

    @staticmethod
    def accuracy(W_pred, W_true):
        P, Tt = np.asarray(W_pred, np.int64), np.asarray(W_true, np.int64)
        if len(Tt) == 0:
            return {"n": 0}
        return {"n": int(len(Tt)),
                "per_word": {nm: float((P[:, f] == Tt[:, f]).mean())
                             for f, nm in enumerate(T.WORD_NAMES)},
                "whole_tuple": float((P == Tt).all(1).mean())}


# ================================================================= persistence
class Persist:
    """Every trained object written to ``persist/<cond>_seed<N>/<leg>/`` with a param hash
    beside it, so every per-arm figure can be re-rendered later.  ``.eqx`` = equinox
    ``tree_serialise_leaves`` (rebuild args in MANIFEST.json); ``.pkl`` = pickle.
    ``param_sha256`` hashes the parameter arrays' bytes (order of the pytree leaves);
    ``file_sha256`` hashes the file."""

    def __init__(self, root):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self.manifest = {}

    @staticmethod
    def _leaf_hash(leaves):
        import hashlib
        h = hashlib.sha256()
        for x in leaves:
            a = np.ascontiguousarray(np.asarray(x))
            h.update(str(a.dtype).encode() + str(a.shape).encode())
            h.update(a.tobytes())
        return h.hexdigest()

    def _finish(self, name, param_hash, rebuild):
        import hashlib
        fp = os.path.join(self.root, name)
        fh = hashlib.sha256(open(fp, "rb").read()).hexdigest()
        with open(fp + ".sha256", "w") as f:
            f.write("param_sha256 %s\nfile_sha256 %s  %s\n" % (param_hash, fh, name))
        self.manifest[name] = {"param_sha256": param_hash, "file_sha256": fh,
                               "bytes": os.path.getsize(fp), "rebuild": rebuild}
        self.flush()

    def eqx(self, name, module, rebuild):
        import equinox as eqx
        import jax
        eqx.tree_serialise_leaves(os.path.join(self.root, name), module)
        leaves = jax.tree_util.tree_leaves(eqx.filter(module, eqx.is_array))
        self._finish(name, self._leaf_hash(leaves), rebuild)

    def pkl(self, name, obj, rebuild="pickle.load"):
        import pickle
        with open(os.path.join(self.root, name), "wb") as f:
            pickle.dump(obj, f, protocol=4)
        leaves = []
        for v in _array_leaves(obj):
            leaves.append(v)
        self._finish(name, self._leaf_hash(leaves), rebuild)

    def flush(self):
        with open(os.path.join(self.root, "MANIFEST.json"), "w") as f:
            json.dump(self.manifest, f, indent=1, default=R._json_default)


def _array_leaves(obj, depth=0):
    """Deterministic walk over the numpy arrays an object holds (for the param hash)."""
    if depth > 6:
        return
    if isinstance(obj, np.ndarray):
        yield obj
    elif isinstance(obj, dict):
        for k in sorted(obj, key=str):
            yield from _array_leaves(obj[k], depth + 1)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _array_leaves(v, depth + 1)
    elif hasattr(obj, "__dict__"):
        for k in sorted(vars(obj)):
            yield from _array_leaves(vars(obj)[k], depth + 1)
    elif hasattr(obj, "shape") and hasattr(obj, "dtype"):
        yield np.asarray(obj)


def _slug(name):
    name = (name.replace("diagnostic+rel@8k: ", "diagrel8k_")
            .replace("diagnostic+rel@80k: ", "diagrel80k_").replace(" +rel", "_plusrel"))
    return (name.replace("rival@10x-steps: ", "rival10xsteps_").replace(
        "diagnostic@10x-read-floats: ", "rival10xfloats_").replace("rival: ", "rival_")
        .replace(" +extractor", "_plusextractor").replace(" ", "_"))


def rival_rebuild(kind, od, width, kw, seed):
    return {"fn": "chlu.exp_w15_cmaze_t_arms.build_policy", "kind": kind, "obs_d": int(od),
            "width": int(width), "n_out": int(T.code_dim(K_COLOURS)), "seed": int(seed),
            "kw": {a: b for a, b in kw.items()}}


# ================================================================ trajectories
def trajectory_triples(insts, rng, per_maze=6, T_len=R.SEQ_T):
    """The SAME random trajectories as ``R.trajectory_sequences`` (same rng draws), kept
    as (maze, state, action, code) so every condition can featurise them its own way."""
    from w9_harness.dynamics import initial_state, key_homes
    out = []
    for m, inst in enumerate(insts):
        homes = key_homes(inst)
        for _ in range(per_maze):
            st = initial_state(inst)
            seq = []
            for t in range(T_len):
                a = T.ACTIONS[int(rng.integers(5))]
                w = T.words_of(inst, st, a, homes)
                legal, disp, ev, pn, nxt = T.outcome_of(inst, st, a, homes)
                seq.append((m, st, a, w, T.encode_outcome(K_COLOURS, legal, disp, ev, pn)))
                if legal:
                    st = nxt
                    if st.pos == inst.goal:
                        break
            out.append(seq)
    return out


# ===================================================================== the run
def run(seed_index, out_dir, cond, leg, latent=128, K=2, n_atoms=6, ridge=1.0,
        rival_steps=8000, diag_mult=10, deletion=None, smoke=False,
        enc_steps=1500, codec_steps=2500, ext_steps=6000, x10_diag=False, rel_diag=False):
    t_start = time.time()
    IN = Inputs(cond)
    if deletion is None:
        deletion = leg == "test"
    if smoke:
        rival_steps, enc_steps, codec_steps = 60, 60, 200
    diag_steps = rival_steps * diag_mult
    rng = np.random.default_rng(10_000 + seed_index)
    cfg = dict(latent=int(latent), K=int(K), n_atoms=int(n_atoms), ridge=float(ridge))
    #: PREREG AMENDMENT 1: OBS-X10 = the extractor trained 10x longer; a printed diagnostic
    #: only.  Raw-obs rivals are byte-identical to the OBS run's and are not re-trained.
    cname = cond + ("-X10" if x10_diag else "")
    tag = "%s_%s" % (cname, leg)
    out = {"seed_index": seed_index, "leg": leg, "condition": cond, "config": cfg,
           "tolerance_used": None, "k_colours": K_COLOURS, "scored_run": leg == "test",
           "smoke": bool(smoke),
           "input_per_arm": {
               "WORDS-MAIN": "CLU [bias|main] over the true words (== learned BLIND-SEQUENTIAL); "
                             "rivals the 27 main-effect one-hots; cont_mlp the words as numbers",
               "WORDS-FULL": "CLU [bias|main|pair] 318; rivals the SAME 318-wide vector; "
                             "cont_mlp the words as numbers",
               "OBS": "rivals: w9 obs (conv-AE patch+global, 29 dynamics scalars, action) and "
                      "obs + extractor one-hots; CLU reads the EXTRACTED words (request-only)",
               "BOTH": "rivals obs + 318 design; CLU the true words"}[cond],
           "rival_steps_matched": int(rival_steps),
           "rival_steps_10x_diagnostic": int(diag_steps),
           "w21_clu_cfg": CLU_CFG,
           "w21_rel_diagnostic": bool(rel_diag and REL_FN is not None)}
    PS = Persist(os.path.join(out_dir, "persist", "%s_seed%d" % (cname, seed_index),
                              leg + ("_SMOKE" if smoke else "")))
    out["persist_dir"] = PS.root
    log("[%s seed %d] config %s rival_steps %d / %d" % (tag, seed_index, cfg, rival_steps,
                                                        diag_steps))
    base, ev_base, (tr_insts, tr_rows, tr_nat, tr_meta), \
        (va_insts, va_rows, va_nat, va_meta) = W16.banks(seed_index, leg)
    out["banks"] = {"train": dict(tr_meta, seed_block=[base, base + 200_000]),
                    leg: dict(va_meta, seed_block=[ev_base, ev_base + 100_000])}
    if not smoke:
        all_rows = list(tr_rows) + [T.TRow(r.maze + len(tr_insts), r.prefix, r.pos, r.ks,
                                           r.action, r.words, r.legal, r.disp, r.ev,
                                           r.pocket_next, r.bites_b) for r in va_rows]
        out["ACCEPT_1_independent_replay"] = T.replay_audit(tr_insts + va_insts, all_rows)
        out["ACCEPT_2_no_impossible_example"] = T.impossible_example_audit(
            tr_insts + va_insts, all_rows)
    ho = T.make_holdout(K_COLOURS, seed=seed_index)
    tr_split = T.split_rows(tr_rows, ho, K_COLOURS)
    va_split_full = T.split_rows(va_rows, ho, K_COLOURS)
    va_split = W16.cap_split(va_split_full, seed_index)
    if smoke:
        va_split = {k: v[:150] for k, v in va_split.items()}
    train_rows = tr_split.get("seen", [])
    out["holdout"] = {"value_colour": ho.value_colour,
                      "pairs": [list(p) for p in ho.pairs],
                      "nway": [list(c) for c in ho.nway],
                      "n_train_rows": len(train_rows),
                      "n_eval_rows_scored": {k: len(v) for k, v in va_split.items()}}
    out["ACCEPT_3_holdout_disjointness"] = T.disjointness_report(
        train_rows, {k: v for k, v in va_split.items() if k != "seen"}, ho, K_COLOURS)

    # ---- frozen codec (+ frozen state encoder where an arm consumes the obs) ----
    Ctr = A.build_codes(train_rows, K_COLOURS)
    Ccodec = np.concatenate([Ctr, T.enumerate_code_alphabet(K_COLOURS)], axis=0)
    codec, crep = A.train_codec(Ccodec, K_COLOURS, latent=cfg["latent"],
                                steps=codec_steps, seed=seed_index)
    crep.pop("loss_curve", None)
    out["frozen_codec"] = crep
    PS.eqx("codec.eqx", codec, {"fn": "chlu.exp_w15_cmaze_t_arms.Codec",
                                "cdim": int(T.code_dim(K_COLOURS)), "latent": cfg["latent"]})
    Ztr = A.codec_encode(codec, Ctr)
    tr_feats = va_feats = tr_g = va_g = None
    if IN.uses_obs:
        enc, enc_rep = A.train_state_encoder(tr_insts, steps=enc_steps, seed=seed_index,
                                             verbose=False, restarts=3, accept_bce=0.010)
        enc_rep.pop("loss_curve", None)
        out["frozen_state_encoder"] = {k: v for k, v in enc_rep.items()
                                       if not isinstance(v, np.ndarray)}
        PS.eqx("state_encoder.eqx", enc, {"fn": "w9_harness.encoder.Encoder (from train())",
                                          "feat_dim": 12, "width": 32})
        tr_feats, va_feats = A.feature_maps(enc, tr_insts), A.feature_maps(enc, va_insts)
        from w9_harness.features import global_summary
        tr_g = [global_summary(S) for S in tr_feats]
        va_g = [global_summary(S) for S in va_feats]
    else:
        out["frozen_state_encoder"] = {"DECLARED_NOT_RUN": "no arm consumes the observation "
                                                           "in " + cond}

    def obs_rows(rows, insts, feats, gs):
        return A.build_obs(rows, insts, feats, gs) if IN.uses_obs else None

    def obs_triples(trip, insts, feats, gs):
        if not IN.uses_obs:
            return None
        return np.stack([A.obs_of(insts[m], feats[m], st, a, gs[m])
                         for (m, st, a) in trip]).astype(np.float32)

    Wtr_true = A.build_words(train_rows)
    Otr = obs_rows(train_rows, tr_insts, tr_feats, tr_g)

    # ---- the declared word extractor (OBS only) ----
    ext = None
    if IN.uses_extractor:
        ext = WordExtractor(Otr.shape[1], hidden=32, seed=seed_index)
        erep = ext.fit(Otr, Wtr_true, steps=200 if smoke else ext_steps, seed=seed_index)
        erep.pop("loss_curve", None)
        Wtr_fit = ext.predict(Otr)
        erep["accuracy_on_TRAIN_rows"] = ext.accuracy(Wtr_fit, Wtr_true)
        out["word_extractor"] = erep
        PS.pkl("word_extractor.pkl", ext.p, "dict of numpy params; WordExtractor._fwd")
        log("  extractor floats %d  train tuple-acc %.4f"
            % (erep["floats"], erep["accuracy_on_TRAIN_rows"]["whole_tuple"]))
    else:
        Wtr_fit = Wtr_true
        out["word_extractor"] = {"DECLARED_NOT_RUN": "words are given in " + cond}

    def words_for(rows, O):
        """The request every word-reading arm gets: true words, or the extractor's."""
        return ext.predict(O) if ext is not None else A.build_words(rows)

    # ---- the CLU, controls, learned blind arms (the condition's design) ----
    grids = IN.grids
    pb = IN.clu_pair_block
    clu = CLU_CLS(Ztr, Wtr_fit, grids, K=cfg["K"], n_atoms=cfg["n_atoms"],
                  ridge=cfg["ridge"], pair_block=pb, seed=seed_index)
    out["clu_ledger"] = clu.ledger()
    out["clu_design"] = "[bias|main|pair] 318" if pb else "[bias|main] 28 (== learned BLIND-SEQUENTIAL)"
    out["rank_over_width_printed"] = {"per_well": clu.ledger()["span_fraction_n_atoms_over_d_k"],
                                      "cap": W16.SPAN_CAP}
    out["clu_request_only"] = bool(clu.assert_request_only(Wtr_fit[:8]) is not False)
    controls = {c: MAKE_CONTROL(Ztr, Wtr_fit, grids, cfg["K"], cfg["n_atoms"],
                                    cfg["ridge"], c, seed_index, pair_block=pb)
                for c in A.TransitionClu.CONTROLS}
    der = controls["landscape_shuffled_across_wells"].derangement
    assert not np.any(np.asarray(der) == np.arange(len(der))), der
    out["across_wells_derangement"] = der
    out["across_wells_is_derangement"] = True
    PS.pkl("clu.pkl", clu)
    for c_, obj_ in controls.items():
        PS.pkl("control_%s.pkl" % c_, obj_)
    log("  across-wells derangement %s (no fixed point: asserted)" % der)
    cb_grids = [np.arange(c, dtype=np.float64) for c in A.colour_blind_alphabets(K_COLOURS)]
    cbw = lambda Wx: A.colour_blind_words(Wx, K_COLOURS)
    blind = {BCOL: (A.TransitionClu(Ztr, cbw(Wtr_fit), cb_grids, K=cfg["K"],
                                    n_atoms=cfg["n_atoms"], ridge=cfg["ridge"],
                                    pair_block=pb, seed=seed_index), cbw),
             BSEQ: (A.TransitionClu(Ztr, Wtr_fit, grids, K=cfg["K"], n_atoms=cfg["n_atoms"],
                                    ridge=cfg["ridge"], pair_block=False, seed=seed_index),
                    lambda Wx: Wx)}
    PS.pkl("blind_colour.pkl", blind[BCOL][0])
    PS.pkl("blind_sequential.pkl", blind[BSEQ][0])
    if not pb:
        out["NOTE_WORDS_MAIN"] = ("the CLU's design is main effects only, so the CLU IS the "
                                  "learned BLIND-SEQUENTIAL arm (identical object); VB3/SB3 "
                                  "are evaluated on BLIND-COLOUR only (PREREG)")

    # ---- rivals: identical input, matched on read-path floats ----
    ext_floats = ext.n_floats() if ext is not None else 0
    target = int(out["clu_ledger"]["read_floats"]) + ext_floats
    out["matched_target"] = {"clu_read_floats": int(out["clu_ledger"]["read_floats"]),
                             "extractor_floats_added": ext_floats, "target_floats": target}
    trips = trajectory_triples(tr_insts, rng)
    flat = [(m, st, a) for seq in trips for (m, st, a, w, y) in seq]
    Wtraj = np.array([w for seq in trips for (m, st, a, w, y) in seq], np.float64)
    Otraj = obs_triples(flat, tr_insts, tr_feats, tr_g)
    Wetraj = ext.predict(Otraj) if ext is not None else None
    Tlen = R.SEQ_T

    def seq_pack(Xflat):
        d = Xflat.shape[1]
        X = np.zeros((len(trips), Tlen, d), np.float32)
        Y = np.zeros((len(trips), Tlen, T.code_dim(K_COLOURS)), np.float32)
        M = np.zeros((len(trips), Tlen), np.float32)
        i = 0
        for s, seq in enumerate(trips):
            for t, (m, st, a, w, y) in enumerate(seq):
                X[s, t], Y[s, t], M[s, t] = Xflat[i], y, 1.0
                i += 1
        return X, Y, M

    Wetr = Wtr_fit if ext is not None else None
    rivals, reps, out["matched_parameters"], out["capacity_diagnostic"] = {}, {}, {}, {}
    out["rel_diagnostic_parameters"] = {}
    rival_inputs = {}
    variants = list(IN.variants())
    if rel_diag and REL_FN is not None:
        variants.append(("+rel", ("mlp", "lstm", "transformer")))
    for variant, kinds in variants:
        if x10_diag and variant == "":
            continue
        Xs1 = IN.rival_x(variant, Wtr_true, Otr, Wetr)
        Xs, Ys, Ms = Xs1[:, None, :], Ctr[:, None, :], np.ones((len(Xs1), 1), np.float32)
        Xt, Yt, Mt = seq_pack(IN.rival_x(variant, Wtraj, Otraj, Wetraj))
        rival_inputs[variant] = int(Xs1.shape[1])
        od = int(Xs1.shape[1])
        widths = A.match_widths(od, T.code_dim(K_COLOURS), target, kinds=kinds,
                                tmax=R.SEQ_T)
        big_w = A.match_widths(od, T.code_dim(K_COLOURS), 10 * target, kinds=kinds,
                               tmax=R.SEQ_T, hi=400)
        for kind in kinds:
            nm = IN.rival_name(kind, variant)
            w, p, kw = widths[kind]
            out["matched_parameters" if variant != "+rel" else "rel_diagnostic_parameters"][nm] = {"input_dim": od, "width": int(w), "params": int(p),
                                             "residual_pct": 100.0 * (p - target) / max(target, 1)}
            budgets = (("rival: ", rival_steps, widths[kind]),
                       ("rival@10x-steps: ", diag_steps, widths[kind]),
                       ("diagnostic@10x-read-floats: ", rival_steps, big_w[kind]))
            if variant == "+rel":
                budgets = (("diagnostic+rel@8k: ", rival_steps, widths[kind]),
                           ("diagnostic+rel@80k: ", diag_steps, widths[kind]))
            for lab, steps, wd in budgets:
                if x10_diag and lab.startswith("diagnostic"):
                    continue
                ww, pp, kkw = wd
                pol, rep = R.train_rival(kind, ww, kkw, Xs, Ys, Ms, Xt, Yt, Mt, seed_index,
                                         steps, lab + nm)
                rep.pop("loss_curve", None)
                rivals[lab + nm] = (pol, variant)
                PS.eqx(_slug(lab + nm) + ".eqx", pol,
                       dict(rival_rebuild(kind, od, ww, kkw, seed_index), steps=int(steps),
                            input_variant=variant or "shared"))
                reps[lab + nm] = rep
            bw, bp, _ = big_w[kind]
            out["capacity_diagnostic"][nm] = {"width": int(bw), "params": int(bp)}
    out["rival_input_dims"] = rival_inputs
    out["rival_training"] = reps

    # ---- floors ----
    fallback = A.majority_code(train_rows, K_COLOURS)
    raw_tr, raw_va = A.make_raw_key(tr_insts), A.make_raw_key(va_insts)
    frag_tr, frag_va = A.make_frag_key(tr_insts), A.make_frag_key(va_insts)
    ekey_map = {}

    def wkey(r):
        return ekey_map.get(id(r), r.words)
    if ext is not None:
        for r, wv in zip(train_rows, Wtr_fit):
            ekey_map[id(r)] = tuple(int(x) for x in wv)
    wname = "floor: lookup cross-maze WORDS" + (" (extracted words)" if ext is not None else "")
    tabs = {wname: (wkey, A.build_lookup(train_rows, wkey)),
            "floor: lookup cross-maze raw r1": (raw_va, A.build_lookup(train_rows, raw_tr)),
            "floor: fragment cross-maze r1": (frag_va, A.build_lookup(train_rows, frag_tr)),
            "floor: lookup WITHIN-maze raw r1 (NOT transfer)":
                (raw_va, A.build_lookup(va_nat, raw_va))}
    out["words_floor_name"] = wname

    def dec(obj, Wx):
        return T.decode_codes(K_COLOURS, A.codec_decode(codec, obj.read(Wx)))

    rows_out = {}
    for name in ("seen", "pair", "nway", "value"):
        rs = va_split.get(name, [])
        if not rs:
            rows_out[name] = {"DECLARED_NOT_RUN": "no rows in this class"}
            continue
        Wt = A.build_words(rs)
        O = obs_rows(rs, va_insts, va_feats, va_g)
        Wr = words_for(rs, O)
        b = {"n": len(rs)}
        if ext is not None:
            b["word_extractor_accuracy"] = WordExtractor.accuracy(Wr, Wt)
            for r, wv in zip(rs, Wr):
                ekey_map[id(r)] = tuple(int(x) for x in wv)
        b["CLU"] = W16.sc(va_insts, rs, dec(clu, Wr))
        for c, obj in controls.items():
            b["control: " + c] = W16.sc(va_insts, rs, dec(obj, Wr))
        for nm, (obj, wf) in blind.items():
            b["arm: " + nm] = W16.sc(va_insts, rs, dec(obj, wf(Wr)))
        for nm, (pol, variant) in rivals.items():
            b[nm] = W16.sc(va_insts, rs, A.policy_predict(
                pol, IN.rival_x(variant, Wt, O, Wr), K_COLOURS))
        for ln, (kf, tab) in tabs.items():
            pred, info = A.lookup_predict(rs, kf, tab, fallback)
            b[ln] = W16.sc(va_insts, rs, pred)
            b[ln + " :coverage"] = info["coverage"]
        for mode in T.BLIND_MODES:
            pred = [T.blind_outcome(va_insts[r.maze], r.state(), r.action, mode) for r in rs]
            b["column: %s (planner)" % mode.upper()] = W16.sc(va_insts, rs, pred)
        rows_out[name] = b
        best8 = max(v for k2, v in b.items() if k2.startswith("rival: "))
        best80 = max(v for k2, v in b.items() if k2.startswith("rival@10x-steps: "))
        if any(k2.startswith("diagnostic+rel") for k2 in b):
            log("  row %-5s diag+rel best %.4f (never a bar)" % (name, max(
                v for k2, v in b.items() if k2.startswith("diagnostic+rel"))))
        log("  row %-5s n=%-5d CLU %.4f best-rival8k %.4f best-rival80k %.4f worst-ctl %.4f "
            "bseq %.4f bcol %.4f%s" % (name, len(rs), b["CLU"], best8, best80,
                                       max(v for k2, v in b.items() if k2.startswith("control: ")),
                                       b["arm: " + BSEQ], b["arm: " + BCOL],
                                       ("  ext-tuple-acc %.4f" % b["word_extractor_accuracy"]
                                        ["whole_tuple"]) if ext is not None else ""))
    out["ROWS"] = rows_out

    # ---- rollout validity (limitation) + per-step exactness ----
    rollouts = [r for r in (T.rollouts_for_maze(i, m) for m, i in enumerate(va_insts)) if r]
    if smoke:
        rollouts = rollouts[:5]

    def batch_inputs(mz, sts, acts):
        Wt = np.array([T.words_of(va_insts[m], st, a) for m, st, a in zip(mz, sts, acts)],
                      np.float64)
        O = (np.stack([A.obs_of(va_insts[m], va_feats[m], st, a, va_g[m])
                       for m, st, a in zip(mz, sts, acts)]).astype(np.float32)
             if IN.uses_obs else None)
        Wr = ext.predict(O) if ext is not None else Wt
        return Wt, O, Wr

    def clu_b(obj):
        def pb_(mz, sts, acts):
            Wt, O, Wr = batch_inputs(mz, sts, acts)
            return dec(obj, Wr)
        return pb_

    def riv_b(pol, variant):
        def pb_(mz, sts, acts):
            Wt, O, Wr = batch_inputs(mz, sts, acts)
            return A.policy_predict(pol, IN.rival_x(variant, Wt, O, Wr), K_COLOURS)
        return pb_

    roll = {"CLU": A.rollout_validity(va_insts, rollouts, clu_b(clu))}
    for c, obj in controls.items():
        roll["control: " + c] = A.rollout_validity(va_insts, rollouts, clu_b(obj))
    for nm, (pol, variant) in rivals.items():
        roll[nm] = A.rollout_validity(va_insts, rollouts, riv_b(pol, variant))
    out["ROLLOUT"] = {"n_rollouts": len(rollouts),
                      "mean_route_len": float(np.mean([len(r["actions"]) for r in rollouts])),
                      "arms": roll}
    log("  rollout CLU validity %.4f per-step %.4f" % (roll["CLU"]["validity"],
                                                      roll["CLU"]["per_step_exactness"]))

    # ---- deletion (scored conditions) ----
    if deletion:
        out["DELETION"] = deletion_rows(IN, train_rows, tr_insts, Wtr_true, Wtr_fit, Otr,
                                        Ztr, Ctr, codec, grids, cfg, va_insts, va_split,
                                        va_feats, va_g, words_for, obs_rows, target, trips,
                                        seq_pack, Wtraj, Otraj, Wetraj, seed_index,
                                        rival_steps, PS)
    else:
        out["DELETION"] = {"DECLARED_NOT_RUN": "deletion rows run on scored TEST runs only"}
    out["seconds_total"] = time.time() - t_start
    p = os.path.join(out_dir, "%s_seed%d%s.json" % (tag, seed_index, "_SMOKE" if smoke else ""))
    with open(p, "w") as fh:
        json.dump(out, fh, indent=1, default=R._json_default)
    log("[%s seed %d] wrote %s [%.1fs]" % (tag, seed_index, p, out["seconds_total"]))


# ==================================================================== deletion
def deletion_rows(IN, train_rows, tr_insts, Wtr_true, Wtr_fit, Otr, Ztr, Ctr, codec, grids,
                  cfg, va_insts, va_split, va_feats, va_g, words_for, obs_rows, target,
                  trips, seq_pack, Wtraj, Otraj, Wetraj, seed_index, rival_steps, PS=None):
    """w15's deletion block (``R.deletion_block``), generalised to the condition's input.
    Rows to delete are chosen by their TRUE words; the CLU downdates the statistics with the
    words it was fitted on.  ⛔ Frozen codec, state encoder and word extractor are OUTSIDE
    the deletable object."""
    k, c = K_COLOURS, 1
    pb = IN.clu_pair_block
    res = {"deleted_colour": c, "condition": IN.cond,
           "frozen_objects_declared_outside_the_deletable_object":
               ["Codec", "w9 state encoder"] + (["word extractor"] if IN.uses_extractor else []),
           "post_deletion_adaptation": ("head statistics A, Mx and store statistics S_k, t_k, "
                                        "n_k downdated, the SAME closed form re-solved. Light "
                                        "and DATA-FREE. NOT 'no changes'.")}
    idx_all = np.arange(len(train_rows))
    for mode, sel in (("value", np.array([i for i in idx_all
                                          if T._has_colour(train_rows[i].words, k, c)])),
                      ("conjunction_colour_AND_tinted_floor",
                       np.array([i for i in idx_all if train_rows[i].words[3] == c]))):
        if sel.size == 0:
            res[mode] = {"DECLARED_NOT_RUN": "no train rows match"}
            continue
        keep = np.setdiff1d(idx_all, sel)
        block = {"n_rows_deleted": int(sel.size), "n_rows_kept": int(keep.size)}
        clu = CLU_CLS(Ztr, Wtr_fit, grids, K=cfg["K"], n_atoms=cfg["n_atoms"],
                      ridge=cfg["ridge"], pair_block=pb, seed=seed_index)
        t0 = time.time()
        rec = clu.delete_rows_exact(Ztr[sel], Wtr_fit[sel])
        block["clu_downdate_receipt"] = {kk: vv for kk, vv in rec.items()
                                         if kk != "basis_change_shapes"}
        block["clu_adaptation_cost"] = {"seconds": time.time() - t0,
                                        "passes_over_retained_data": 0, "items_kept": False,
                                        "gradient_steps": 0, "examples_seen": 0}
        ref = CLU_CLS(Ztr[keep], Wtr_fit[keep], grids, K=cfg["K"],
                              n_atoms=cfg["n_atoms"], ridge=cfg["ridge"], pair_block=pb,
                              seed=seed_index, well_channels=clu.well_channels)
        devs, scales = [], []
        for w1, w2 in zip(clu.store.stats, ref.store.stats):
            devs.append(float(np.abs(np.asarray(w1.S, np.float64)
                                     - np.asarray(w2.S, np.float64)).max()))
            scales.append(float(np.abs(np.asarray(w2.S, np.float64)).max()))
        atom_dev = max(float(np.abs(np.asarray(M1, np.float64) - np.asarray(M2, np.float64)).max())
                       for M1, M2 in zip(clu.store.modes, ref.store.modes))
        rel = max(devs) / max(max(scales), 1e-12)
        block["store_scoped_exactness"] = {
            "statement": "store after deletion == store FITTED WITHOUT the rows, same partition",
            "max_relative_delta_S": rel, "deployed_atoms_max_abs_delta": atom_dev,
            "bar": 1e-5, "pass": bool(rel <= 1e-5)}
        surv = {}
        for name in ("seen", "pair", "nway"):
            rs = [r for r in va_split.get(name, []) if not T._has_colour(r.words, k, c)]
            if not rs:
                surv[name] = {"DECLARED_NOT_RUN": "no surviving-colour rows"}
                continue
            Wr = words_for(rs, obs_rows(rs, va_insts, va_feats, va_g))
            d1 = T.decode_codes(k, A.codec_decode(codec, clu.read(Wr)))
            d2 = T.decode_codes(k, A.codec_decode(codec, ref.read(Wr)))
            surv[name] = {"n": len(rs),
                          "CLU after deletion (downdate)": W16.sc(va_insts, rs, d1),
                          "CLU fitted WITHOUT the rows": W16.sc(va_insts, rs, d2),
                          "reads_identical_rows_frac": float((d1 == d2).all(1).mean())}
        block["composition_on_surviving_colours"] = surv
        # rival: retrain-without vs as-trained (mlp, the condition's shared input)
        kind = "mlp"
        Wetr = Wtr_fit if IN.uses_extractor else None
        Xa = IN.rival_x("", Wtr_true, Otr, Wetr)
        Xt, Yt, Mt = seq_pack(IN.rival_x("", Wtraj, Otraj, Wetraj))
        w, p, kw = A.match_widths(Xa.shape[1], T.code_dim(k), target, kinds=(kind,),
                                  tmax=R.SEQ_T, verbose=False)[kind]
        pol_rt, rep_rt = R.train_rival(kind, w, kw, Xa[keep][:, None, :], Ctr[keep][:, None, :],
                                       np.ones((keep.size, 1), np.float32), Xt, Yt, Mt,
                                       seed_index, rival_steps, "mlp retrain-without")
        pol_as, rep_as = R.train_rival(kind, w, kw, Xa[:, None, :], Ctr[:, None, :],
                                       np.ones((len(Xa), 1), np.float32), Xt, Yt, Mt,
                                       seed_index, rival_steps, "mlp as-trained")
        if PS is not None:
            tagm = "value" if mode == "value" else "conjunction"
            PS.pkl("clu_after_%s.pkl" % tagm, clu)
            PS.pkl("clu_fitted_without_%s.pkl" % tagm, ref)
            rb = rival_rebuild(kind, Xa.shape[1], w, kw, seed_index)
            PS.eqx("rival_mlp_retrain_without_%s.eqx" % tagm, pol_rt, dict(rb, steps=rival_steps))
            PS.eqx("rival_mlp_as_trained_%s.eqx" % tagm, pol_as, dict(rb, steps=rival_steps))
        block["rival_retrain_without_cost"] = {
            "seconds": rep_rt["seconds"], "gradient_steps": rep_rt["steps"],
            "examples_seen": rep_rt["examples_seen"], "items_kept": True,
            "passes_over_retained_data": "the whole surviving pool is re-read every batch",
            "note": "the rival has no other way to delete"}
        rv = {}
        for name in ("seen", "pair", "nway"):
            rs = [r for r in va_split.get(name, []) if not T._has_colour(r.words, k, c)]
            if not rs:
                rv[name] = {"DECLARED_NOT_RUN": "no surviving-colour rows"}
                continue
            O = obs_rows(rs, va_insts, va_feats, va_g)
            X = IN.rival_x("", A.build_words(rs), O, words_for(rs, O))
            rv[name] = {"n": len(rs),
                        "rival retrain-without": W16.sc(va_insts, rs,
                                                        A.policy_predict(pol_rt, X, k)),
                        "rival as-trained": W16.sc(va_insts, rs, A.policy_predict(pol_as, X, k))}
        block["rival_composition_on_surviving_colours"] = rv
        res[mode] = block
    return res


if __name__ == "__main__":
    si, od = int(sys.argv[1]), sys.argv[2]
    kw = {}
    for i in range(3, len(sys.argv), 2):
        kw[sys.argv[i].lstrip("-")] = sys.argv[i + 1]
    os.makedirs(od, exist_ok=True)
    if "clu_cfg" in kw:
        import hashlib
        import w21_clu
        raw = open(kw["clu_cfg"], "rb").read()
        CLU_CFG = dict(json.loads(raw.decode("utf-8")), sha256=hashlib.sha256(raw).hexdigest(),
                       path=kw["clu_cfg"])
        fac = w21_clu.CluFactory(CLU_CFG)
        CLU_CLS, MAKE_CONTROL, REL_FN = fac, fac.make_control, fac.relations_block
        log("w21 CLU cfg %s sha256 %s" % (kw["clu_cfg"], CLU_CFG["sha256"]))
    run(si, od, kw["cond"], kw["leg"], ext_steps=int(kw.get("ext_steps", 6000)),
        x10_diag=bool(int(kw.get("x10_diag", 0))), latent=int(kw.get("latent", 128)),
        K=int(kw.get("K", 2)), n_atoms=int(kw.get("atoms", 6)),
        rival_steps=int(kw.get("rival_steps", 8000)), diag_mult=int(kw.get("diag_mult", 10)),
        deletion=(None if "deletion" not in kw else bool(int(kw["deletion"]))),
        smoke=bool(int(kw.get("smoke", 0))), rel_diag=bool(int(kw.get("rel_diag", 0))))
