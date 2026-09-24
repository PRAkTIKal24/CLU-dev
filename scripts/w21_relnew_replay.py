"""w21 spoke/rel-newseeds -- ZERO-FIT per-tuple replay of PERSISTED CLU-REL-A10 CMAZE-T runs.

    python scripts/w21_relnew_replay.py <out_json> --run_json <J> --persist <P> --clu_cfg <cfg>

``scripts/w21_pertuple_replay.py`` (maze-t-seed4) adapted to runs made by the VENDORED peer
runner ``peer_w21_clu_improve_e7fdb9b/w19_symmetric.py`` with ``--clu_cfg CLU-REL-A10.json
--rel_diag 1``.  The differences are ONLY what that runner adds:
  * the CLU and the four controls are ``w21_clu`` objects (unpickled; ``read`` unchanged);
  * the '+rel' diagnostic rivals (``diagrel8k_*`` / ``diagrel80k_*``) read the runner's own
    ``Inputs.rival_x('+rel', ...)`` with ``REL_FN = CluFactory(cfg).relations_block``.
Every persisted object's ``param_sha256`` is checked against the run's MANIFEST, the n-way
row is re-scored on the SAME rows (W16.banks + T.make_holdout + W16.cap_split, unchanged),
the aggregate is ASSERTED equal to the run JSON, and per-4-tuple exact counts are printed
(``w21_symmetric._sc``, which asserts its sum equals ``W16.sc``).  ⛔ Nothing is fitted.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import pickle
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
VEND = os.path.join(ROOT, "peer_w21_clu_improve_e7fdb9b")
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, VEND)                     # w21_clu (pickled CLU class lives there)

import chlu.exp_w15_cmaze_t as T                                    # noqa: E402
import chlu.exp_w15_cmaze_t_arms as A                               # noqa: E402
import w15_cmaze_t_run as R                                         # noqa: E402
import w16_cmaze_t as W16                                           # noqa: E402
import w21_clu                                                      # noqa: E402
import w21_symmetric as S21                                         # noqa: E402  (_sc only)

_spec = importlib.util.spec_from_file_location("peer_w19_symmetric_e7fdb9b",
                                               os.path.join(VEND, "w19_symmetric.py"))
V = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V)

K = R.K_COLOURS
FROZEN_SHA = "0d4ec5b9fba55bd6f943a91fc8afa20027d79c5edea3531a47117c86f08ee5c6"


def slug_to_arm(fname):
    b = fname[:-4]
    for pre_s, pre in (("diagrel80k_", "diagnostic+rel@80k: "), ("diagrel8k_", "diagnostic+rel@8k: "),
                       ("rival10xsteps_", "rival@10x-steps: "),
                       ("rival10xfloats_", "diagnostic@10x-read-floats: "), ("rival_", "rival: ")):
        if b.startswith(pre_s):
            rest = b[len(pre_s):]
            break
    rest = rest.replace("_plusrel", " +rel").replace("_plusextractor", " +extractor")
    return pre + rest


def main(op, run_json, pdir, cfg_path):
    import equinox as eqx
    import jax
    t0 = time.time()
    raw = open(cfg_path, "rb").read()
    sha = hashlib.sha256(raw).hexdigest()
    assert sha == FROZEN_SHA, sha
    J = json.load(open(run_json))
    assert J["w21_clu_cfg"]["sha256"] == FROZEN_SHA, J["w21_clu_cfg"]["sha256"]
    cfg = dict(json.loads(raw.decode("utf-8")), sha256=sha, path=cfg_path)
    fac = w21_clu.CluFactory(cfg)
    V.CLU_CFG, V.CLU_CLS, V.MAKE_CONTROL, V.REL_FN = cfg, fac, fac.make_control, fac.relations_block
    seed, leg, cond = int(J["seed_index"]), J["leg"], J["condition"]
    man = json.load(open(os.path.join(pdir, "MANIFEST.json")))
    IN = V.Inputs(cond)
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
            leaves = list(V._array_leaves(obj))
        h = V.Persist._leaf_hash(leaves)
        fh = hashlib.sha256(open(os.path.join(pdir, name), "rb").read()).hexdigest()
        ok = h == man[name]["param_sha256"] and fh == man[name]["file_sha256"]
        hashchk[name] = {"param_sha256": h, "file_sha256": fh,
                         "manifest_param_sha256": man[name]["param_sha256"],
                         "manifest_file_sha256": man[name]["file_sha256"], "match": ok}
        assert ok, name
        return obj

    def load_eqx(name, template):
        return check(name, eqx.tree_deserialise_leaves(os.path.join(pdir, name), template), True)

    def load_pkl(name):
        with open(os.path.join(pdir, name), "rb") as fh:
            return check(name, pickle.load(fh), False)

    codec = load_eqx("codec.eqx", A.Codec(jax.random.PRNGKey(0), int(T.code_dim(K)), 128))
    import w9_harness.encoder as w9enc
    from w9_harness.features import global_summary
    X0 = A.maze_images(va_insts[:1])
    enc = load_eqx("state_encoder.eqx", w9enc.Encoder(int(X0.shape[1]), 32, 12,
                                                      jax.random.PRNGKey(0)))
    feats = A.feature_maps(enc, va_insts)
    gs = [global_summary(Sm) for Sm in feats]
    O = A.build_obs(rs, va_insts, feats, gs)
    assert cond == "BOTH", cond
    Wt = A.build_words(rs)
    Wr = Wt

    def dec(obj, Wx):
        return T.decode_codes(K, A.codec_decode(codec, obj.read(Wx)))

    PT = {}
    got = {}
    got["CLU"] = S21._sc(PT, "nway", "CLU", va_insts, rs, dec(load_pkl("clu.pkl"), Wr))
    for c in A.TransitionClu.CONTROLS:
        nm = "control: " + c
        got[nm] = S21._sc(PT, "nway", nm, va_insts, rs, dec(load_pkl("control_%s.pkl" % c), Wr))
    got["arm: " + V.BCOL] = S21._sc(PT, "nway", "arm: " + V.BCOL, va_insts, rs,
                                    dec(load_pkl("blind_colour.pkl"),
                                        A.colour_blind_words(Wr, K)))
    got["arm: " + V.BSEQ] = S21._sc(PT, "nway", "arm: " + V.BSEQ, va_insts, rs,
                                    dec(load_pkl("blind_sequential.pkl"), Wr))
    for fn in sorted(man):
        if not fn.startswith(("rival_", "rival10x", "diagrel")) or not fn.endswith(".eqx"):
            continue
        if "retrain_without" in fn or "as_trained" in fn:
            continue
        rb = man[fn]["rebuild"]
        pol = load_eqx(fn, A.build_policy(rb["kind"], rb["obs_d"], rb["width"], rb["n_out"],
                                          rb["seed"], **rb["kw"]))
        variant = "" if rb.get("input_variant", "shared") == "shared" else rb["input_variant"]
        nm = slug_to_arm(fn)
        got[nm] = S21._sc(PT, "nway", nm, va_insts, rs, A.policy_predict(
            pol, IN.rival_x(variant, Wt, O, Wr), K))
    fallback = A.majority_code(train_rows, K)
    raw_tr, raw_va = A.make_raw_key(tr_insts), A.make_raw_key(va_insts)
    frag_tr, frag_va = A.make_frag_key(tr_insts), A.make_frag_key(va_insts)
    wkey = lambda r: r.words                                                # noqa: E731
    tabs = {J["words_floor_name"]: (wkey, A.build_lookup(train_rows, wkey)),
            "floor: lookup cross-maze raw r1": (raw_va, A.build_lookup(train_rows, raw_tr)),
            "floor: fragment cross-maze r1": (frag_va, A.build_lookup(train_rows, frag_tr)),
            "floor: lookup WITHIN-maze raw r1 (NOT transfer)":
                (raw_va, A.build_lookup(va_nat, raw_va))}
    for ln, (kf, tab) in tabs.items():
        pred, _info = A.lookup_predict(rs, kf, tab, fallback)
        got[ln] = S21._sc(PT, "nway", ln, va_insts, rs, pred)
    for mode in T.BLIND_MODES:
        pred = [T.blind_outcome(va_insts[r.maze], r.state(), r.action, mode) for r in rs]
        nm = "column: %s (planner)" % mode.upper()
        got[nm] = S21._sc(PT, "nway", nm, va_insts, rs, pred)
    ref = J["ROWS"]["nway"]
    repro = {k: {"replayed": v, "run_json": ref.get(k), "equal": ref.get(k) == v}
             for k, v in got.items()}
    missing = sorted(k for k, v in ref.items() if isinstance(v, (int, float)) and k != "n"
                     and not k.endswith(":coverage") and k not in got)
    res = {"script": "scripts/w21_relnew_replay.py", "zero_fit": True, "run_json": run_json,
           "persist_dir": pdir, "clu_cfg": cfg_path, "clu_cfg_sha256": sha, "seed": seed,
           "leg": leg, "condition": cond, "n_nway_rows": len(rs), "param_hash_checks": hashchk,
           "n_objects_checked": len(hashchk),
           "all_param_hashes_match": all(v["match"] for v in hashchk.values()),
           "reproduces_run_json": repro, "run_json_arms_not_replayed": missing,
           "all_reproduce": all(v["equal"] for v in repro.values()),
           "PER_TUPLE_NWAY": PT.get("nway", {}), "seconds": time.time() - t0}
    with open(op, "w") as fh:
        json.dump(res, fh, indent=1, default=R._json_default)
    print("[replay %s %s seed %d] arms %d objects %d hashes-ok %s reproduce-all %s not-replayed %s [%.0fs]"
          % (cond, leg, seed, len(got), len(hashchk), res["all_param_hashes_match"],
             res["all_reproduce"], missing, res["seconds"]), flush=True)
    for k, v in repro.items():
        print("   %-70s %.4f  run %s  %s" % (k, v["replayed"], v["run_json"],
                                            "OK" if v["equal"] else "MISMATCH"), flush=True)


if __name__ == "__main__":
    kw = {sys.argv[i].lstrip("-"): sys.argv[i + 1] for i in range(2, len(sys.argv), 2)}
    main(sys.argv[1], kw["run_json"], kw["persist"], kw["clu_cfg"])
