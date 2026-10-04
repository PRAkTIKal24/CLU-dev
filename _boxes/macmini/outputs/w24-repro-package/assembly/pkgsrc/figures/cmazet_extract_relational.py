"""As cmazet_extract.py, for the n-CLU-REL runs.   python figures/cmazet_extract_relational.py figures/data/cmazet_items_rel.json"""

import argparse
import hashlib
import importlib.util
import json
import os
import pickle
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for p in (os.path.join(REPO, "scripts"), REPO):
    if p not in sys.path:
        sys.path.insert(0, p)
import nclu

assert os.path.dirname(os.path.dirname(os.path.abspath(nclu.__file__))) == REPO, nclu.__file__
import nclu.cmaze.task as T
import nclu.cmaze.arms as A
import cmazet_asymmetric as ASYM
import cmazet_relational_nclu as cmazet_relational_nclu
import cmazet_symmetric as V
from nclu.paths import out_root as _out_root

EVID = str(_out_root())
RUNS = {
    0: ("cmaze_t/relational", "cmaze_t/relational/test/BOTH_test_seed0.json"),
    1: ("cmaze_t/relational", "cmaze_t/relational/test/BOTH_test_seed1.json"),
    2: ("cmaze_t/relational", "cmaze_t/relational/test/BOTH_test_seed2.json"),
    3: ("cmaze_t/relational", "cmaze_t/relational/test/BOTH_test_seed3.json"),
    4: ("cmaze_t/relational", "cmaze_t/relational/test/BOTH_test_seed4.json"),
    5: ("cmaze_t/relational", "cmaze_t/relational/test/BOTH_test_seed5.json"),
}
CFG = os.path.join(REPO, "scripts", "configs", "nclu_rel.json")
FROZEN_SHA = "0d4ec5b9fba55bd6f943a91fc8afa20027d79c5edea3531a47117c86f08ee5c6"
K = ASYM.K_COLOURS


def fsha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def arm_of_file(fname):
    b = fname[:-4]
    for pre_s, pre in (
        ("diagrel80k_", "diagnostic+rel@80k: "),
        ("diagrel8k_", "diagnostic+rel@8k: "),
        ("rival10xsteps_", "rival@10x-steps: "),
        ("rival10xfloats_", "diagnostic@10x-read-floats: "),
        ("rival_", "rival: "),
    ):
        if b.startswith(pre_s):
            rest = b[len(pre_s) :].replace("_plusrel", " +rel")
            return pre + rest
    raise ValueError(fname)


class Stage:
    def __init__(self, d):
        self.d = d
        self.man = json.load(open(os.path.join(d, "MANIFEST.json")))
        self.loaded = {}

    def _check_file(self, name):
        p = os.path.join(self.d, name)
        got = fsha(p)
        m = self.man[name]
        lines = open(p + ".sha256").read().strip().split("\n")
        assert got == m["file_sha256"] == lines[1].split()[1], "file hash mismatch %s" % p
        assert os.path.getsize(p) == m["bytes"], "byte count mismatch %s" % p
        return (p, m)

    def pkl(self, name):
        p, m = self._check_file(name)
        with open(p, "rb") as f:
            obj = pickle.load(f)
        ph = V.Persist._leaf_hash(list(V._array_leaves(obj)))
        assert ph == m["param_sha256"], "param hash mismatch %s" % name
        self.loaded[name] = dict(file_sha256=m["file_sha256"], param_sha256=ph)
        return obj

    def eqx(self, name, skeleton):
        import equinox as eqx
        import jax

        p, m = self._check_file(name)
        mod = eqx.tree_deserialise_leaves(p, skeleton)
        ph = V.Persist._leaf_hash(jax.tree_util.tree_leaves(eqx.filter(mod, eqx.is_array)))
        assert ph == m["param_sha256"], "param hash mismatch %s" % name
        self.loaded[name] = dict(file_sha256=m["file_sha256"], param_sha256=ph)
        return mod


def family_of(value_colour, key4):
    fam = T._nway_candidates(K, value_colour)
    hits = [f for f in T.NWAY_FAMILIES if tuple(key4) in fam[f]]
    assert len(hits) == 1, (key4, hits)
    return hits[0]


def per_row(insts, rows, em):
    out = []
    for r, e in zip(rows, em):
        good, why, _gen = T.judge_transition(insts[r.maze], r.state(), r.action, e)
        out.append([int(bool(good)), why, [int(x) for x in e]])
    return out


def run_seed(seed, cfg, fac):
    import jax

    t0 = time.time()
    branch, art_rel = RUNS[seed]
    art_p = os.path.join(EVID, art_rel)
    J = json.load(open(art_p))
    assert J["seed_index"] == seed and J["leg"] == "test" and (J["condition"] == "BOTH")
    assert J["rel_nclu_cfg"]["sha256"] == FROZEN_SHA, J["rel_nclu_cfg"]["sha256"]
    st = Stage(os.path.join(EVID, branch, "test", "persist", "BOTH_seed%d" % seed, "test"))
    IN = V.Inputs("BOTH")
    _b, ev_base, _tr, (va_insts, va_rows, _va_nat, va_meta) = ASYM.banks(seed, "test")
    want = {k: v for k, v in J["banks"]["test"].items() if k != "seed_block"}
    assert va_meta == want, ("TEST bank meta differs", va_meta, want)
    ho = T.make_holdout(K, seed=seed)
    assert [list(p) for p in ho.pairs] == J["holdout"]["pairs"]
    assert [list(c) for c in ho.nway] == J["holdout"]["nway"]
    va_split = ASYM.cap_split(T.split_rows(va_rows, ho, K), seed)
    assert {k: len(v) for k, v in va_split.items()} == J["holdout"]["n_eval_rows_scored"]
    rows = va_split["nway"]
    assert len(rows) == J["ROWS"]["nway"]["n"]
    codec = st.eqx(
        "codec.eqx",
        A.Codec(
            jax.random.PRNGKey(0), int(T.code_dim(K)), st.man["codec.eqx"]["rebuild"]["latent"]
        ),
    )
    import nclu.cmaze.harness.encoder as w9enc
    from nclu.cmaze.harness.features import global_summary

    rb = st.man["state_encoder.eqx"]["rebuild"]
    X0 = A.maze_images(va_insts[:1])
    enc = st.eqx(
        "state_encoder.eqx",
        w9enc.Encoder(int(X0.shape[1]), rb["width"], rb["feat_dim"], jax.random.PRNGKey(0)),
    )
    feats = A.feature_maps(enc, va_insts)
    gs = [global_summary(Sm) for Sm in feats]
    O = A.build_obs(rows, va_insts, feats, gs)
    Wt = A.build_words(rows)
    Wr = Wt

    def dec(obj, Wx):
        return T.decode_codes(K, A.codec_decode(codec, obj.read(Wx)))

    em = {"CLU": dec(st.pkl("clu.pkl"), Wr)}
    for c in A.TransitionClu.CONTROLS:
        em["control: " + c] = dec(st.pkl("control_%s.pkl" % c), Wr)
    em["arm: " + V.BCOL] = dec(st.pkl("blind_colour.pkl"), A.colour_blind_words(Wr, K))
    em["arm: " + V.BSEQ] = dec(st.pkl("blind_sequential.pkl"), Wr)
    for fn in sorted(st.man):
        if not fn.startswith(("rival_", "rival10x", "diagrel")) or not fn.endswith(".eqx"):
            continue
        if "retrain_without" in fn or "as_trained" in fn:
            continue
        rbd = st.man[fn]["rebuild"]
        pol = st.eqx(
            fn,
            A.build_policy(
                rbd["kind"], rbd["obs_d"], rbd["width"], rbd["n_out"], rbd["seed"], **rbd["kw"]
            ),
        )
        variant = "" if rbd.get("input_variant", "shared") == "shared" else rbd["input_variant"]
        em[arm_of_file(fn)] = A.policy_predict(pol, IN.rival_x(variant, Wt, O, Wr), K)
    verdicts = {nm: per_row(va_insts, rows, e) for nm, e in em.items()}
    got = {nm: sum((v[0] for v in vs)) / len(vs) for nm, vs in verdicts.items()}
    ref = J["ROWS"]["nway"]
    repro, fail = ({}, [])
    for nm, g in sorted(got.items()):
        w = ref.get(nm)
        repro[nm] = dict(reread=g, artefact=w, equal=w is not None and g == w)
        if not repro[nm]["equal"]:
            fail.append(nm)
    missing = sorted(
        (
            k
            for k, v in ref.items()
            if isinstance(v, (int, float))
            and k != "n"
            and (not k.startswith(("floor", "column")))
            and (k not in got)
        )
    )
    fail += ["NOT RE-READ: " + k for k in missing]
    fam = [
        family_of(ho.value_colour, (r.words[0], r.words[2], r.words[3], r.words[4])) for r in rows
    ]
    items = []
    for i, r in enumerate(rows):
        truth = [int(r.legal), int(r.disp), int(r.ev), int(r.pocket_next)]
        items.append(
            dict(
                i=i,
                maze=int(r.maze),
                maze_seed=int(va_insts[r.maze].seed),
                pos=[int(x) for x in r.pos],
                ks=[int(x) for x in r.ks],
                action=r.action,
                words=[int(x) for x in r.words],
                family=fam[i],
                truth=truth,
            )
        )
    return dict(
        seed=seed,
        config="CLU-REL-A10",
        seconds=time.time() - t0,
        artefact=art_rel,
        artefact_sha256=fsha(art_p),
        stage_dir=st.d,
        bank={"ev_base": ev_base, "meta": va_meta},
        n_nway=len(rows),
        value_colour=ho.value_colour,
        objects_loaded=st.loaded,
        repro=repro,
        repro_missing_arms=missing,
        gate_failures=fail,
        items=items,
        verdicts=verdicts,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--seeds", default="0,1,2,3,4,5")
    a = ap.parse_args()
    raw = open(CFG, "rb").read()
    sha = hashlib.sha256(raw).hexdigest()
    assert sha == FROZEN_SHA, sha
    cfg = dict(json.loads(raw.decode("utf-8")), sha256=sha, path=CFG)
    fac = cmazet_relational_nclu.CluFactory(cfg)
    V.CLU_CFG, V.CLU_CLS, V.MAKE_CONTROL, V.REL_FN = (
        cfg,
        fac,
        fac.make_control,
        fac.relations_block,
    )
    res = []
    for s in a.seeds.split(","):
        r = run_seed(int(s), cfg, fac)
        print(
            "[REL gate2 seed %d] arms %d objects %d equal %d/%d missing %s failures %s [%.0fs]"
            % (
                r["seed"],
                len(r["repro"]),
                len(r["objects_loaded"]),
                sum((v["equal"] for v in r["repro"].values())),
                len(r["repro"]),
                r["repro_missing_arms"],
                r["gate_failures"],
                r["seconds"],
            ),
            flush=True,
        )
        for k, v in r["repro"].items():
            print(
                "   %-72s %.16f  run %r  %s"
                % (k, v["reread"], v["artefact"], "OK" if v["equal"] else "MISMATCH")
            )
        res.append(r)
    fails = [(r["seed"], f) for r in res for f in r["gate_failures"]]
    summary = {
        "config": "CLU-REL-A10",
        "clu_cfg_sha256": sha,
        "seeds": [r["seed"] for r in res],
        "arms_checked": sum((len(r["repro"]) for r in res)),
        "arms_equal": sum((sum((v["equal"] for v in r["repro"].values())) for r in res)),
        "objects_hash_checked": sum((len(r["objects_loaded"]) for r in res)),
        "CLU_per_seed": {r["seed"]: r["repro"]["CLU"]["reread"] for r in res},
        "all_arms_equal_artefact": not fails,
        "failures": fails,
        "code": {
            "relational_nclu_sha256": fsha(cmazet_relational_nclu.__file__),
            "runner_sha256": fsha(V.__file__),
        },
    }
    print(json.dumps(summary, indent=1))
    rep = os.path.splitext(a.out)[0] + ".REPRO.json"
    json.dump(
        {
            "summary": summary,
            "per_seed": [
                {
                    k: r[k]
                    for k in (
                        "seed",
                        "seconds",
                        "artefact",
                        "artefact_sha256",
                        "stage_dir",
                        "n_nway",
                        "repro",
                        "repro_missing_arms",
                        "gate_failures",
                        "objects_loaded",
                    )
                }
                for r in res
            ],
        },
        open(rep, "w"),
        indent=1,
        sort_keys=True,
    )
    if fails:
        print("GATE 2 FAILED: per-item file NOT written;", rep)
        sys.exit(1)
    json.dump(
        {
            "what": "per-item re-read of the CLU-REL-A10 BOTH TEST persisted objects; every arm's aggregate equals the run JSON (asserted, float ==)",
            "summary": summary,
            "seeds": res,
        },
        open(a.out, "w"),
        sort_keys=True,
    )
    print("GATE 2 PASSED; wrote", a.out, "and", rep)


if __name__ == "__main__":
    main()
