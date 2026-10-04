"""w23 spoke/seed5-diagnosis -- QUESTION 1, part b.  ZERO-FIT content audit per OUTCOME CODE.

    python scripts/w23_seed5_content.py <out_dir>

For the persisted seed-5 BOTH TEST CLU and its four charter-C controls (same design, same A
and Mx, different deployed atoms; each file+param hash asserted against sidecar + MANIFEST),
for EVERY distinct outcome code in the fit pool and in the n-way eval rows:

  * z_true = codec.enc(code)  (frozen codec; identical for every row with that code),
  * the deployed span's relative residual  ||z_true - P_span z_true|| / ||z_true||  per well,
  * decode(P_span z_true)  -- the L2-best point the store can hold -- and its margin on the
    displacement block,
  * the CLU's in-sample exact on the fit rows carrying that code, and what it emits there.

⛔ Nothing is fitted.  Every object is read from its pickle.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import time
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import chlu.exp_w15_cmaze_t as T                                    # noqa: E402
import chlu.exp_w15_cmaze_t_arms as A                               # noqa: E402
import w16_cmaze_t as W16                                           # noqa: E402
import cmazet_symmetric_runner as S                                 # noqa: E402
from w23_seed5_trace import PD, K, SEED, verify_file                # noqa: E402

LINES = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LINES.append(s)


def main(out_dir):
    import equinox as eqx
    import jax
    t0 = time.time()
    man = json.load(open(os.path.join(PD, "MANIFEST.json")))
    names = ["clu.pkl"] + ["control_%s.pkl" % c for c in A.TransitionClu.CONTROLS]
    hc = {"codec.eqx": verify_file(PD, "codec.eqx", man)}
    codec = eqx.tree_deserialise_leaves(os.path.join(PD, "codec.eqx"),
                                        A.Codec(jax.random.PRNGKey(0), int(T.code_dim(K)), 128))
    assert S.Persist._leaf_hash(jax.tree_util.tree_leaves(
        eqx.filter(codec, eqx.is_array))) == man["codec.eqx"]["param_sha256"]
    objs = {}
    for nm in names:
        hc[nm] = verify_file(PD, nm, man)
        with open(os.path.join(PD, nm), "rb") as fh:
            o = pickle.load(fh)
        ph = S.Persist._leaf_hash(list(S._array_leaves(o)))
        assert ph == man[nm]["param_sha256"], nm
        hc[nm]["param_sha256_computed"] = ph
        hc[nm]["param_match"] = True
        objs[nm] = o
    P("[hash] codec + clu + 4 controls: file and param sha256 == sidecar == MANIFEST -> OK")

    _b, _e, (tr_insts, tr_rows, _n, _m), (va_insts, va_rows, _vn, _m2) = W16.banks(SEED, "test")
    ho = T.make_holdout(K, seed=SEED)
    train_rows = T.split_rows(tr_rows, ho, K).get("seen", [])
    rs = W16.cap_split(T.split_rows(va_rows, ho, K), SEED).get("nway", [])
    Wtr = A.build_words(train_rows)
    truth_tr = [tuple(int(x) for x in (r.legal, r.disp, r.ev, r.pocket_next)) for r in train_rows]
    truth_va = [tuple(int(x) for x in (r.legal, r.disp, r.ev, r.pocket_next)) for r in rs]
    clu = objs["clu.pkl"]
    em_tr = T.decode_codes(K, A.codec_decode(codec, clu.read(Wtr)))
    ok_tr = np.array([T.judge_transition(tr_insts[r.maze], r.state(), r.action, e)[0]
                      for r, e in zip(train_rows, em_tr)])
    # the n-way tuples each code serves
    serves = {}
    for r, c in zip(rs, truth_va):
        serves.setdefault(c, set()).add("%d,%d,%d,%d" % (r.words[0], r.words[2], r.words[3],
                                                         r.words[4]))
    codes = sorted(set(truth_tr) | set(truth_va))
    Z = A.codec_encode(codec, np.stack([T.encode_outcome(K, *c) for c in codes])).astype(np.float64)
    blocks = T.code_blocks(K)

    def dec(z):
        lg = np.asarray(A.codec_decode(codec, np.asarray(z, np.float32)))
        out = T.decode_codes(K, lg)
        o, c = blocks["disp"]
        s = np.sort(lg[:, o:o + c], axis=1)
        return out, s[:, -1] - s[:, -2]

    rt, _ = dec(Z)
    table = {}
    for nm, o in objs.items():
        Gs = [np.asarray(G, np.float64) for G in o.store.modes]
        chs = [np.asarray(c, np.int64) for c in o.store.well_channels]
        Zp = np.zeros_like(Z)
        res_w = np.zeros((len(codes), len(Gs)))
        for k, (G, ch) in enumerate(zip(Gs, chs)):
            if np.abs(G).sum() == 0:
                continue
            # orthonormal basis of the atom rows' span (SVD with a rank cut: a QR of a
            # rank-deficient 6-row block would invent a sixth direction)
            Uu, sv, _ = np.linalg.svd(G.T, full_matrices=False)
            Q = Uu[:, sv > sv.max() * 1e-8]
            Zp[:, ch] = (Z[:, ch] @ Q) @ Q.T
            res_w[:, k] = np.linalg.norm(Z[:, ch] - Zp[:, ch], axis=1) / \
                np.maximum(np.linalg.norm(Z[:, ch], axis=1), 1e-12)
        dp, mg = dec(Zp)
        rel = np.linalg.norm(Z - Zp, axis=1) / np.linalg.norm(Z, axis=1)
        table[nm] = {"rel_resid": rel, "rel_resid_per_well": res_w,
                     "oracle_decode": dp, "oracle_disp_margin": mg}

    rows = []
    cnt_tr = Counter(truth_tr)
    cnt_va = Counter(truth_va)
    for i, c in enumerate(codes):
        m = np.array([t == c for t in truth_tr])
        emit = Counter(tuple(int(x) for x in em_tr[j]) for j in np.flatnonzero(m))
        rec = {"code": list(c), "n_train": int(cnt_tr[c]), "n_nway_eval": int(cnt_va[c]),
               "serves_nway_tuples": sorted(serves.get(c, [])),
               "codec_roundtrip_ok": rt[i].tolist() == list(c),
               "clu_in_sample_exact": float(ok_tr[m].mean()) if m.any() else None,
               "clu_in_sample_emits_top3": [[str(a), b] for a, b in emit.most_common(3)]}
        for nm in objs:
            t = table[nm]
            rec[nm] = {"rel_resid": round(float(t["rel_resid"][i]), 4),
                       "rel_resid_per_well": np.round(t["rel_resid_per_well"][i], 4).tolist(),
                       "oracle_decode": t["oracle_decode"][i].tolist(),
                       "oracle_ok": t["oracle_decode"][i].tolist() == list(c),
                       "oracle_disp_margin": round(float(t["oracle_disp_margin"][i]), 4)}
        rows.append(rec)
    rows.sort(key=lambda r: -r["clu.pkl"]["rel_resid"])

    P("\n=== every outcome code (legal,disp,event,pocket) in the fit pool / n-way eval, sorted by "
      "the CLU span residual ===")
    P("| code | n train | n n-way eval | n-way tuples it serves | codec rt ok | CLU span rel-resid "
      "(well0, well1) | CLU L2-oracle decode (disp margin) | CLU in-sample exact | CLU in-sample "
      "emits | across_wells span rel-resid | across_wells L2-oracle |")
    P("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        c, x = r["clu.pkl"], r["control_landscape_shuffled_across_wells.pkl"]
        P("| %s | %d | %d | %s | %s | %.3f (%s) | %s%s (%.2f) | %s | %s | %.3f | %s%s |"
          % (tuple(r["code"]), r["n_train"], r["n_nway_eval"],
             " ".join("(%s)" % s for s in r["serves_nway_tuples"]) or "-",
             r["codec_roundtrip_ok"], c["rel_resid"],
             ", ".join("%.3f" % v for v in c["rel_resid_per_well"]),
             tuple(c["oracle_decode"]), "" if c["oracle_ok"] else " WRONG",
             c["oracle_disp_margin"],
             "%.3f" % r["clu_in_sample_exact"] if r["clu_in_sample_exact"] is not None else "-",
             r["clu_in_sample_emits_top3"], x["rel_resid"], tuple(x["oracle_decode"]),
             "" if x["oracle_ok"] else " WRONG"))
    n_bad = [r for r in rows if not r["clu.pkl"]["oracle_ok"]]
    P("\nCLU: codes whose L2-best in-span point decodes WRONG: %d of %d: %s"
      % (len(n_bad), len(rows), [tuple(r["code"]) for r in n_bad]))
    # correlation: in-sample exact vs span residual (codes with >= 20 train rows)
    sel = [r for r in rows if r["n_train"] >= 20]
    x = np.array([r["clu.pkl"]["rel_resid"] for r in sel])
    y = np.array([r["clu_in_sample_exact"] for r in sel])
    rho = float(np.corrcoef(x, y)[0, 1])
    P("codes with >=20 fit rows: %d ; Pearson(CLU span rel-resid, CLU in-sample exact) = %.3f"
      % (len(sel), rho))
    summ = {}
    for nm in objs:
        summ[nm] = {"codes_oracle_wrong": int(sum(not r[nm]["oracle_ok"] for r in rows)),
                    "rel_resid_1302": next(r[nm]["rel_resid"] for r in rows
                                           if r["code"] == [1, 3, 0, 2])}
    P("per object: #codes whose L2-best in-span point decodes wrong, and residual of (1,3,0,2):",
      json.dumps(summ))
    out = {"script": "scripts/w23_seed5_content.py", "zero_fit": True, "hash_checks": hc,
           "codes": rows, "pearson_resid_vs_insample_ge20": rho, "per_object": summ,
           "seconds": time.time() - t0}
    with open(os.path.join(out_dir, "Q1-CONTENT.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    with open(os.path.join(out_dir, "Q1-CONTENT.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(LINES) + "\n")
    P("[done %.0fs]" % out["seconds"])


if __name__ == "__main__":
    main(sys.argv[1])
