"""w21 spoke/maze-t-seed4 -- ZERO-FIT per-tuple replay of PERSISTED CMAZE-T objects.

    python scripts/w21_pertuple_replay.py <out_json> --run_json <J> --persist <P>

Re-reads every object a w19/w21 run persisted (``persist/<cond>_seed<N>/<leg>/``), checks
each object's ``param_sha256`` against the run's MANIFEST, re-scores the n-way row on the
SAME rows the run scored (W16.banks + T.make_holdout + W16.cap_split, unchanged), ASSERTS the
aggregate equals the number the run printed (``ROWS.nway``), and prints the per-4-tuple
(w_agent, w_door, w_tint, w_dir) exact counts per arm.  ⛔ Nothing is fitted: the lookup
floors and planner columns are deterministic tables/solvers rebuilt exactly as the runner
builds them.  Used for seeds 0-2 (w19 did not print per-tuple scores).
"""

from __future__ import annotations

import json
import os
import pickle
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
import w21_symmetric as S                                           # noqa: E402

K = R.K_COLOURS


def slug_to_arm(fname):
    b = fname[:-4]
    if b.startswith("rival10xsteps_"):
        pre, rest = "rival@10x-steps: ", b[len("rival10xsteps_"):]
    elif b.startswith("rival10xfloats_"):
        pre, rest = "diagnostic@10x-read-floats: ", b[len("rival10xfloats_"):]
    else:
        pre, rest = "rival: ", b[len("rival_"):]
    rest = rest.replace("_plusextractor", " +extractor")
    return pre + rest


def main(op, run_json, pdir):
    import equinox as eqx
    import jax
    t0 = time.time()
    J = json.load(open(run_json))
    seed, leg, cond = int(J["seed_index"]), J["leg"], J["condition"]
    man = json.load(open(os.path.join(pdir, "MANIFEST.json")))
    IN = S.Inputs(cond)
    _b, _e, (tr_insts, tr_rows, _n, _m), (va_insts, va_rows, va_nat, _m2) = W16.banks(seed, leg)
    ho = T.make_holdout(K, seed=seed)
    train_rows = T.split_rows(tr_rows, ho, K).get("seen", [])
    rs = W16.cap_split(T.split_rows(va_rows, ho, K), seed).get("nway", [])
    assert len(rs) == J["ROWS"]["nway"]["n"], (len(rs), J["ROWS"]["nway"]["n"])
    hashchk = {}

    def check(name, obj, is_eqx):
        if is_eqx:
            leaves = jax.tree_util.tree_leaves(eqx.filter(obj, eqx.is_array))
        else:
            leaves = list(S._array_leaves(obj))
        h = S.Persist._leaf_hash(leaves)
        ok = h == man[name]["param_sha256"]
        hashchk[name] = {"param_sha256": h, "manifest": man[name]["param_sha256"], "match": ok}
        assert ok, name
        return obj

    def load_eqx(name, template):
        return check(name, eqx.tree_deserialise_leaves(os.path.join(pdir, name), template), True)

    def load_pkl(name):
        with open(os.path.join(pdir, name), "rb") as fh:
            return check(name, pickle.load(fh), False)

    codec = load_eqx("codec.eqx", A.Codec(jax.random.PRNGKey(0), int(T.code_dim(K)), 128))
    O = feats = gs = None
    if IN.uses_obs:
        import w9_harness.encoder as w9enc
        from w9_harness.features import global_summary
        X0 = A.maze_images(va_insts[:1])
        enc = load_eqx("state_encoder.eqx", w9enc.Encoder(int(X0.shape[1]), 32, 12,
                                                          jax.random.PRNGKey(0)))
        feats = A.feature_maps(enc, va_insts)
        gs = [global_summary(Sm) for Sm in feats]
        O = A.build_obs(rs, va_insts, feats, gs)
    Wt = A.build_words(rs)
    Wr = Wt
    if IN.uses_extractor:
        with open(os.path.join(pdir, "word_extractor.pkl"), "rb") as fh:
            p = check("word_extractor.pkl", pickle.load(fh), False)
        ext = S.WordExtractor(O.shape[1])
        ext.p = p
        Wr = ext.predict(O)

    def dec(obj, Wx):
        return T.decode_codes(K, A.codec_decode(codec, obj.read(Wx)))

    PT = {}
    got = {}
    got["CLU"] = S._sc(PT, "nway", "CLU", va_insts, rs, dec(load_pkl("clu.pkl"), Wr))
    for c in A.TransitionClu.CONTROLS:
        nm = "control: " + c
        got[nm] = S._sc(PT, "nway", nm, va_insts, rs, dec(load_pkl("control_%s.pkl" % c), Wr))
    got["arm: " + S.BCOL] = S._sc(PT, "nway", "arm: " + S.BCOL, va_insts, rs,
                                  dec(load_pkl("blind_colour.pkl"),
                                      A.colour_blind_words(Wr, K)))
    got["arm: " + S.BSEQ] = S._sc(PT, "nway", "arm: " + S.BSEQ, va_insts, rs,
                                  dec(load_pkl("blind_sequential.pkl"), Wr))
    for fn in sorted(man):
        if not fn.startswith(("rival_", "rival10x")) or not fn.endswith(".eqx"):
            continue
        if "retrain_without" in fn or "as_trained" in fn:
            continue
        rb = man[fn]["rebuild"]
        pol = load_eqx(fn, A.build_policy(rb["kind"], rb["obs_d"], rb["width"], rb["n_out"],
                                          rb["seed"], **rb["kw"]))
        variant = "" if rb.get("input_variant", "shared") == "shared" else rb["input_variant"]
        nm = slug_to_arm(fn)
        got[nm] = S._sc(PT, "nway", nm, va_insts, rs, A.policy_predict(
            pol, IN.rival_x(variant, Wt, O, Wr), K))
    # floors (deterministic tables, rebuilt as the runner builds them)
    fallback = A.majority_code(train_rows, K)
    raw_tr, raw_va = A.make_raw_key(tr_insts), A.make_raw_key(va_insts)
    frag_tr, frag_va = A.make_frag_key(tr_insts), A.make_frag_key(va_insts)
    ekey = {}
    wkey = lambda r: ekey.get(id(r), r.words)                               # noqa: E731
    if not IN.uses_extractor:
        tabs = {J["words_floor_name"]: (wkey, A.build_lookup(train_rows, wkey))}
    else:
        tabs = {}                     # extracted-word floor needs train obs: DECLARED skipped
    tabs.update({"floor: lookup cross-maze raw r1": (raw_va, A.build_lookup(train_rows, raw_tr)),
                 "floor: fragment cross-maze r1": (frag_va, A.build_lookup(train_rows, frag_tr)),
                 "floor: lookup WITHIN-maze raw r1 (NOT transfer)":
                     (raw_va, A.build_lookup(va_nat, raw_va))})
    for ln, (kf, tab) in tabs.items():
        pred, _info = A.lookup_predict(rs, kf, tab, fallback)
        got[ln] = S._sc(PT, "nway", ln, va_insts, rs, pred)
    for mode in T.BLIND_MODES:
        pred = [T.blind_outcome(va_insts[r.maze], r.state(), r.action, mode) for r in rs]
        nm = "column: %s (planner)" % mode.upper()
        got[nm] = S._sc(PT, "nway", nm, va_insts, rs, pred)
    ref = J["ROWS"]["nway"]
    repro = {k: {"replayed": v, "run_json": ref.get(k), "equal": ref.get(k) == v}
             for k, v in got.items()}
    res = {"script": "scripts/w21_pertuple_replay.py", "zero_fit": True, "run_json": run_json,
           "persist_dir": pdir, "seed": seed, "leg": leg, "condition": cond,
           "n_nway_rows": len(rs), "param_hash_checks": hashchk,
           "all_param_hashes_match": all(v["match"] for v in hashchk.values()),
           "reproduces_run_json": repro,
           "all_reproduce": all(v["equal"] for v in repro.values()),
           "PER_TUPLE_NWAY": PT.get("nway", {}), "seconds": time.time() - t0}
    with open(op, "w") as fh:
        json.dump(res, fh, indent=1, default=R._json_default)
    print("[replay %s %s seed %d] arms %d hashes-ok %s reproduce-all %s [%.0fs]"
          % (cond, leg, seed, len(got), res["all_param_hashes_match"], res["all_reproduce"],
             res["seconds"]), flush=True)
    for k, v in repro.items():
        print("   %-70s %.4f  run %s  %s" % (k, v["replayed"], v["run_json"],
                                            "OK" if v["equal"] else "MISMATCH"), flush=True)


if __name__ == "__main__":
    kw = {sys.argv[i].lstrip("-"): sys.argv[i + 1] for i in range(2, len(sys.argv), 2)}
    main(sys.argv[1], kw["run_json"], kw["persist"])
