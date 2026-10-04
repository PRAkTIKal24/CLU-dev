"""Part C gate 2 + per-item extraction: re-read w19's persisted BOTH TEST objects on this machine.

NOTHING IS TRAINED. Every object is the staged w19 file, hash-checked immediately before it is
loaded (file_sha256 against MANIFEST.json and its .sha256; after loading, the parameter hash is
recomputed with w19's own Persist._leaf_hash and must equal param_sha256). The TEST bank, hold-out,
cap and split are re-derived from the deterministic engine exactly as w19's runner derives them.

THE GATE: every arm's unseen n-way score from these re-reads must equal the scored artefact
(w19-maze-t-symmetric/test/BOTH_test_seed{s}.json ROWS.nway) -- CLU, the four controls, both blind
arms and every rival -- and the deletion rows (DELETION.*.composition_on_surviving_colours.nway,
rival_composition_on_surviving_colours.nway). The printed check is the brief's: CLU
0.800 / 0.566 / 0.808 (mean 0.7246), controls 0.000 / 0.225 / 0.244 / 0.166. If any read
differs from the artefact, the per-item file is NOT written and the script exits 1.

usage: python cmazet_extract.py <stage root> <out json> [--seeds 0,1,2]
"""
import argparse
import hashlib
import json
import os
import pickle
import sys
import time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
import chlu                               # noqa: E402
assert os.path.dirname(os.path.dirname(os.path.abspath(chlu.__file__))) == REPO, \
    "chlu must resolve INSIDE this worktree: %s" % chlu.__file__
import chlu.exp_w15_cmaze_t as T          # noqa: E402
import chlu.exp_w15_cmaze_t_arms as A     # noqa: E402
import w16_cmaze_t as W16                 # noqa: E402
import cmazet_symmetric_runner as S     # noqa: E402
from chlu.paths import out_root as _out_root
EVID = str(_out_root())
ART = "w19-maze-t-symmetric/test/BOTH_test_seed{}.json"
K = W16.K_COLOURS
PRINTED = {"CLU": ["0.800", "0.566", "0.808"],
           "control: landscape_removed": "0.000", "control: landscape_shuffled": "0.225",
           "control: landscape_shuffled_across_wells": "0.244", "control: random_atoms": "0.166"}
def fsha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


class Stage:
    def __init__(self, root, seed):
        self.d = os.path.join(root, "BOTH_seed%d" % seed, "test")
        self.man = json.load(open(os.path.join(self.d, "MANIFEST.json")))
        self.loaded = {}

    def _check_file(self, name):
        p = os.path.join(self.d, name)
        got = fsha(p)
        m = self.man[name]
        lines = open(p + ".sha256").read().strip().split("\n")
        assert got == m["file_sha256"] == lines[1].split()[1], f"file hash mismatch {p}"
        assert os.path.getsize(p) == m["bytes"], f"byte count mismatch {p}"
        return p, m

    def pkl(self, name):
        p, m = self._check_file(name)
        with open(p, "rb") as f:
            obj = pickle.load(f)
        ph = S.Persist._leaf_hash(list(S._array_leaves(obj)))
        assert ph == m["param_sha256"], f"param hash mismatch {name}"
        self.loaded[name] = dict(file_sha256=m["file_sha256"], param_sha256=ph)
        return obj

    def eqx(self, name, skeleton):
        import equinox as eqx
        import jax
        p, m = self._check_file(name)
        mod = eqx.tree_deserialise_leaves(p, skeleton)
        leaves = jax.tree_util.tree_leaves(eqx.filter(mod, eqx.is_array))
        ph = S.Persist._leaf_hash(leaves)
        assert ph == m["param_sha256"], f"param hash mismatch {name}"
        self.loaded[name] = dict(file_sha256=m["file_sha256"], param_sha256=ph)
        return mod


def rival_skeleton(rb):
    kw = dict(rb.get("kw") or {})
    return A.build_policy(rb["kind"], rb["obs_d"], rb["width"], rb["n_out"], seed=rb["seed"], **kw)


def arm_of_file(fname):
    """persisted rival file -> the artefact's arm name."""
    stem = fname[:-4]
    for pre, lab in (("rival10xsteps_", "rival@10x-steps: "),
                     ("rival10xfloats_", "diagnostic@10x-read-floats: "), ("rival_", "rival: ")):
        if stem.startswith(pre):
            rest = stem[len(pre):]
            plus = rest.endswith("_plusextractor")
            rest = rest.replace("_plusextractor", "")
            return lab + rest + (" +extractor" if plus else "")
    raise ValueError(fname)


def family_of(value_colour, key4):
    fam = T._nway_candidates(K, value_colour)
    hits = [f for f in T.NWAY_FAMILIES if tuple(key4) in fam[f]]
    assert len(hits) == 1, (key4, hits)
    return hits[0]


def per_row(insts, rows, em):
    out = []
    for r, e in zip(rows, em):
        good, why, gen = T.judge_transition(insts[r.maze], r.state(), r.action, e)
        out.append((bool(good), why, [int(x) for x in e]))
    return out


def run_seed(root, seed):
    t0 = time.time()
    st = Stage(root, seed)
    art = json.load(open(os.path.join(EVID, ART.format(seed))))
    assert art["seed_index"] == seed and art["leg"] == "test" and art["condition"] == "BOTH"
    IN = S.Inputs("BOTH")
    base, ev_base, _tr, (va_insts, va_rows, va_nat, va_meta) = W16.banks(seed, "test")
    want = {k: v for k, v in art["banks"]["test"].items() if k != "seed_block"}
    assert va_meta == want, ("TEST bank meta differs", va_meta, want)
    ho = T.make_holdout(K, seed=seed)
    assert [list(p) for p in ho.pairs] == art["holdout"]["pairs"]
    assert [list(c) for c in ho.nway] == art["holdout"]["nway"]
    va_split = W16.cap_split(T.split_rows(va_rows, ho, K), seed)
    assert {k: len(v) for k, v in va_split.items()} == art["holdout"]["n_eval_rows_scored"]

    # ---- objects, each hash-checked before and after loading ----
    import jax
    codec = st.eqx("codec.eqx", A.Codec(jax.random.PRNGKey(0), T.code_dim(K),
                                        st.man["codec.eqx"]["rebuild"]["latent"]))
    imgs = A.maze_images(va_insts[:1])
    import w9_harness.encoder as w9enc
    rb = st.man["state_encoder.eqx"]["rebuild"]
    enc = st.eqx("state_encoder.eqx", w9enc.Encoder(int(imgs.shape[1]), rb["width"],
                                                     rb["feat_dim"], jax.random.PRNGKey(0)))
    va_feats = A.feature_maps(enc, va_insts)
    from w9_harness.features import global_summary
    va_g = [global_summary(Sx) for Sx in va_feats]
    clu = st.pkl("clu.pkl")
    controls = {c: st.pkl("control_%s.pkl" % c) for c in A.TransitionClu.CONTROLS}
    cbw = lambda Wx: A.colour_blind_words(Wx, K)  # noqa: E731
    blind = {"arm: " + S.BCOL: (st.pkl("blind_colour.pkl"), cbw),
             "arm: " + S.BSEQ: (st.pkl("blind_sequential.pkl"), lambda Wx: Wx)}
    rivals = {}
    for fname in sorted(st.man):
        if fname.startswith("rival") and "retrain_without" not in fname \
                and "as_trained" not in fname:
            rbd = st.man[fname]["rebuild"]
            variant = "" if rbd.get("input_variant", "shared") == "shared" else rbd["input_variant"]
            rivals[arm_of_file(fname)] = (st.eqx(fname, rival_skeleton(rbd)), variant)

    def dec(obj, Wx):
        return T.decode_codes(K, A.codec_decode(codec, obj.read(Wx)))

    def obs(rs):
        return A.build_obs(rs, va_insts, va_feats, va_g)

    rows = va_split["nway"]
    Wt = A.build_words(rows)
    O = obs(rows)
    Wr = Wt  # BOTH: the CLU reads the true words
    em = {"CLU": dec(clu, Wr)}
    for c, obj in controls.items():
        em["control: " + c] = dec(obj, Wr)
    for nm, (obj, wf) in blind.items():
        em[nm] = dec(obj, wf(Wr))
    for nm, (pol, variant) in rivals.items():
        em[nm] = A.policy_predict(pol, IN.rival_x(variant, Wt, O, Wr), K)
    verdicts = {nm: per_row(va_insts, rows, e) for nm, e in em.items()}
    got = {nm: sum(v[0] for v in vs) / len(vs) for nm, vs in verdicts.items()}

    # ---- THE GATE: every arm equals the scored artefact ----
    want_rows = art["ROWS"]["nway"]
    repro, fail = {}, []
    for nm, g in sorted(got.items()):
        w = want_rows[nm]
        repro[nm] = dict(reread=g, artefact=w, equal=(g == w), abs_diff=abs(g - w))
        if g != w:
            fail.append(nm)
    missing = sorted(k for k in want_rows if k not in got and not k.startswith(("floor", "column"))
                     and k != "n")
    printed = {"CLU": "%.3f" % got["CLU"]}
    for c in PRINTED:
        if c != "CLU":
            printed[c] = "%.3f" % got[c]

    # ---- deletion pair (value scope; conjunction too) on surviving-colour n-way rows ----
    dele = {}
    c_del = int(art["DELETION"]["deleted_colour"])
    for tagm, mode in (("value", "value"), ("conjunction", "conjunction_colour_AND_tinted_floor")):
        after = st.pkl("clu_after_%s.pkl" % tagm)
        without = st.pkl("clu_fitted_without_%s.pkl" % tagm)
        rbd = st.man["rival_mlp_retrain_without_%s.eqx" % tagm]["rebuild"]
        rt = st.eqx("rival_mlp_retrain_without_%s.eqx" % tagm, rival_skeleton(rbd))
        rbd2 = st.man["rival_mlp_as_trained_%s.eqx" % tagm]["rebuild"]
        at = st.eqx("rival_mlp_as_trained_%s.eqx" % tagm, rival_skeleton(rbd2))
        rs = [r for r in rows if not T._has_colour(r.words, K, c_del)]
        rs_idx = [i for i, r in enumerate(rows) if not T._has_colour(r.words, K, c_del)]
        Wd = A.build_words(rs)
        Od = obs(rs)
        Xd = IN.rival_x("", Wd, Od, Wd)
        e2 = {"CLU before deletion": dec(clu, Wd),
              "CLU after deletion (downdate)": dec(after, Wd),
              "CLU fitted WITHOUT the rows": dec(without, Wd),
              "rival mlp as-trained": A.policy_predict(at, Xd, K),
              "rival mlp retrain-without": A.policy_predict(rt, Xd, K)}
        v2 = {nm: per_row(va_insts, rs, e) for nm, e in e2.items()}
        g2 = {nm: sum(v[0] for v in vs) / len(vs) for nm, vs in v2.items()}
        wa = art["DELETION"][mode]
        checks = {
            "CLU after deletion (downdate)":
                wa["composition_on_surviving_colours"]["nway"]["CLU after deletion (downdate)"],
            "CLU fitted WITHOUT the rows":
                wa["composition_on_surviving_colours"]["nway"]["CLU fitted WITHOUT the rows"],
            "rival mlp as-trained": wa["rival_composition_on_surviving_colours"]["nway"]["rival as-trained"],
            "rival mlp retrain-without":
                wa["rival_composition_on_surviving_colours"]["nway"]["rival retrain-without"]}
        drep = {}
        for nm, w in checks.items():
            drep[nm] = dict(reread=g2[nm], artefact=w, equal=(g2[nm] == w))
            if g2[nm] != w:
                fail.append("DELETION %s: %s" % (tagm, nm))
        same = [bool((a == b).all()) for a, b in zip(e2["CLU after deletion (downdate)"],
                                                     e2["CLU fitted WITHOUT the rows"])]
        dele[tagm] = dict(deleted_colour=c_del, mode=mode, n_rows=len(rs), repro=drep,
                          before_reread=g2["CLU before deletion"],
                          downdate_equals_fitted_without_rows_frac=sum(same) / len(same),
                          row_index_in_nway=rs_idx,
                          verdicts={nm: [[int(v[0]), v[1], v[2]] for v in vs]
                                    for nm, vs in v2.items()})

    fam = [family_of(ho.value_colour, (r.words[0], r.words[2], r.words[3], r.words[4]))
           for r in rows]
    items = []
    for i, r in enumerate(rows):
        items.append(dict(i=i, maze=int(r.maze), maze_seed=int(va_insts[r.maze].seed),
                          pos=[int(x) for x in r.pos], ks=[int(x) for x in r.ks],
                          action=r.action, words=[int(x) for x in r.words], family=fam[i],
                          truth=[int(r.legal), int(r.disp), int(r.ev), int(r.pocket_next)]))
    return dict(seed=seed, seconds=time.time() - t0, artefact=ART.format(seed),
                artefact_sha256=fsha(os.path.join(EVID, ART.format(seed))),
                bank={"ev_base": ev_base, "meta": va_meta}, n_nway=len(rows),
                value_colour=ho.value_colour, objects_loaded=st.loaded,
                repro=repro, repro_missing_arms=missing, printed_3dp=printed,
                gate_failures=fail, deletion=dele, items=items,
                verdicts={nm: [[int(v[0]), v[1], v[2]] for v in vs]
                          for nm, vs in verdicts.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("out")
    ap.add_argument("--seeds", default="0,1,2")
    a = ap.parse_args()
    res = [run_seed(a.root, int(s)) for s in a.seeds.split(",")]
    fails = [(r["seed"], f) for r in res for f in r["gate_failures"]]
    summary = {
        "CLU_3dp": [r["printed_3dp"]["CLU"] for r in res],
        "CLU_mean_4dp": "%.4f" % (sum(r["repro"]["CLU"]["reread"] for r in res) / len(res)),
        "controls_3dp_per_seed": {c: [r["printed_3dp"][c] for r in res]
                                  for c in PRINTED if c != "CLU"},
        "controls_mean_3dp": {c: "%.3f" % (sum(r["repro"][c]["reread"] for r in res) / len(res))
                              for c in PRINTED if c != "CLU"},
        "all_arms_equal_artefact": not fails, "failures": fails}
    if len(res) == 3:
        assert summary["CLU_3dp"] == PRINTED["CLU"], summary["CLU_3dp"]
        assert summary["CLU_mean_4dp"] == "0.7246", summary["CLU_mean_4dp"]
        for c in PRINTED:
            if c != "CLU":
                assert summary["controls_mean_3dp"][c] == PRINTED[c], (c, summary)
    print(json.dumps(summary, indent=1))
    rep = os.path.splitext(a.out)[0] + ".REPRO.json"
    json.dump({"summary": summary,
               "per_seed": [{k: r[k] for k in ("seed", "seconds", "artefact", "artefact_sha256",
                                               "n_nway", "repro", "repro_missing_arms",
                                               "printed_3dp", "gate_failures",
                                               "objects_loaded")}
                            | {"deletion_repro": {m: {kk: vv for kk, vv in d.items()
                                                      if kk in ("repro", "before_reread",
                                                                "downdate_equals_fitted_without_rows_frac",
                                                                "n_rows")}
                                                  for m, d in r["deletion"].items()}}
                            for r in res]},
              open(rep, "w"), indent=1, sort_keys=True)
    if fails:
        print("GATE FAILED: per-item file NOT written;", rep)
        sys.exit(1)
    json.dump({"what": "per-item re-read of w19 BOTH TEST persisted objects (Part C); every "
                       "arm's aggregate equals the scored artefact (asserted)",
               "summary": summary, "seeds": res}, open(a.out, "w"), sort_keys=True)
    print("GATE PASSED; wrote", a.out, "and", rep)


if __name__ == "__main__":
    main()
