"""w23 spoke/seed5-diagnosis -- QUESTION 1, part c.  Is the seed-5 CONTENT signature general?

    python scripts/w23_content_allseeds.py <out_dir>

For the base n-CLU (A1) of BOTH TEST on every seed 0-5 (persisted objects; file + param sha256
asserted against sidecar and MANIFEST), for every held-out n-way tuple: its truth code(s), the
CLU's exact count, and for that code the deployed span's relative residual, the decode of the
L2-best in-span point (and its displacement/event margins), and the CLU's in-sample exact on
the FIT rows that carry the same code.  ⛔ Zero-fit: every object is read, nothing is solved.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import time
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import chlu.exp_w15_cmaze_t as T                                    # noqa: E402
import chlu.exp_w15_cmaze_t_arms as A                               # noqa: E402
import w16_cmaze_t as W16                                           # noqa: E402
import w21_symmetric as S                                           # noqa: E402
from w23_seed5_trace import K, verify_file                          # noqa: E402

OUT = "C:/chlu/outputs"
SRC = {s: (OUT + "/w19-maze-t-symmetric/persist/BOTH_seed%d/test" % s,
           OUT + "/w19-maze-t-symmetric/test/BOTH_test_seed%d.json" % s) for s in (0, 1, 2)}
SRC.update({s: (OUT + "/w21-maze-t-seed4/test/persist/BOTH_seed%d/test" % s,
                OUT + "/w21-maze-t-seed4/test/BOTH_test_seed%d.json" % s) for s in (3, 4, 5)})
LINES = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LINES.append(s)


def one_seed(seed):
    import equinox as eqx
    import jax
    pd, rj = SRC[seed]
    man = json.load(open(os.path.join(pd, "MANIFEST.json")))
    hc = {n: verify_file(pd, n, man) for n in ("codec.eqx", "clu.pkl")}
    codec = eqx.tree_deserialise_leaves(os.path.join(pd, "codec.eqx"),
                                        A.Codec(jax.random.PRNGKey(0), int(T.code_dim(K)), 128))
    assert S.Persist._leaf_hash(jax.tree_util.tree_leaves(
        eqx.filter(codec, eqx.is_array))) == man["codec.eqx"]["param_sha256"]
    with open(os.path.join(pd, "clu.pkl"), "rb") as fh:
        clu = pickle.load(fh)
    assert S.Persist._leaf_hash(list(S._array_leaves(clu))) == man["clu.pkl"]["param_sha256"]
    for v in hc.values():
        v["param_match"] = True
    J = json.load(open(rj))
    _b, _e, (tr_insts, tr_rows, _n, _m), (va_insts, va_rows, _vn, _m2) = W16.banks(seed, "test")
    ho = T.make_holdout(K, seed=seed)
    train_rows = T.split_rows(tr_rows, ho, K).get("seen", [])
    rs = W16.cap_split(T.split_rows(va_rows, ho, K), seed).get("nway", [])
    assert len(rs) == J["ROWS"]["nway"]["n"]
    em_va = T.decode_codes(K, A.codec_decode(codec, clu.read(A.build_words(rs))))
    ok_va = np.array([T.judge_transition(va_insts[r.maze], r.state(), r.action, e)[0]
                      for r, e in zip(rs, em_va)])
    assert float(ok_va.mean()) == J["ROWS"]["nway"]["CLU"], (seed, ok_va.mean())
    em_tr = T.decode_codes(K, A.codec_decode(codec, clu.read(A.build_words(train_rows))))
    ok_tr = np.array([T.judge_transition(tr_insts[r.maze], r.state(), r.action, e)[0]
                      for r, e in zip(train_rows, em_tr)])
    tr_code = [tuple(int(x) for x in (r.legal, r.disp, r.ev, r.pocket_next)) for r in train_rows]
    va_code = [tuple(int(x) for x in (r.legal, r.disp, r.ev, r.pocket_next)) for r in rs]
    codes = sorted(set(tr_code) | set(va_code))
    Z = A.codec_encode(codec, np.stack([T.encode_outcome(K, *c) for c in codes])).astype(np.float64)
    Zp = np.zeros_like(Z)
    for G, ch in zip(clu.store.modes, clu.store.well_channels):
        G = np.asarray(G, np.float64)
        Uu, sv, _ = np.linalg.svd(G.T, full_matrices=False)
        Q = Uu[:, sv > sv.max() * 1e-8]
        Zp[:, ch] = (Z[:, ch] @ Q) @ Q.T
    rel = np.linalg.norm(Z - Zp, axis=1) / np.linalg.norm(Z, axis=1)
    lg = np.asarray(A.codec_decode(codec, Zp.astype(np.float32)))
    dp = T.decode_codes(K, lg)
    bl = T.code_blocks(K)
    marg = {}
    for nm in ("legal", "disp", "event", "pocket"):
        o, c = bl[nm]
        s = np.sort(lg[:, o:o + c], axis=1)
        marg[nm] = s[:, -1] - s[:, -2]
    ci = {c: i for i, c in enumerate(codes)}
    tr_by = defaultdict(list)
    for j, c in enumerate(tr_code):
        tr_by[c].append(j)
    by_t = defaultdict(list)
    for i, r in enumerate(rs):
        by_t["%d,%d,%d,%d" % (r.words[0], r.words[2], r.words[3], r.words[4])].append(i)
    out = []
    for t in sorted(by_t):
        idx = by_t[t]
        cc = Counter(va_code[i] for i in idx)
        for c, n in cc.most_common():
            ii = [i for i in idx if va_code[i] == c]
            k = ci[c]
            trj = tr_by.get(c, [])
            out.append({"seed": seed, "tuple": t, "code": list(c), "n": n,
                        "clu_exact": int(ok_va[ii].sum()),
                        "n_fit_rows_with_code": len(trj),
                        "clu_in_sample_on_code": float(ok_tr[trj].mean()) if trj else None,
                        "span_rel_resid": float(rel[k]),
                        "l2_oracle_decode": dp[k].tolist(),
                        "l2_oracle_ok": dp[k].tolist() == list(c),
                        "l2_oracle_min_margin": float(min(marg[m][k] for m in marg))})
    return {"hash_checks": hc, "rows": out, "clu_nway": J["ROWS"]["nway"]["CLU"],
            "clu_in_sample_all": float(ok_tr.mean()), "persist": pd, "run_json": rj}


def main(out_dir):
    t0 = time.time()
    res = {"script": "scripts/w23_content_allseeds.py", "zero_fit": True, "seeds": {}}
    P("| seed | tuple | truth code | n | CLU exact | fit rows w/ code | CLU in-sample on code "
      "| span rel-resid | L2-oracle decode | min block margin |")
    P("|---|---|---|---|---|---|---|---|---|---|")
    allrows = []
    for s in range(6):
        r = one_seed(s)
        res["seeds"][s] = r
        for x in r["rows"]:
            allrows.append(x)
            P("| %d | (%s) | %s | %d | %d/%d | %d | %s | %.3f | %s%s | %.2f |"
              % (s, x["tuple"], tuple(x["code"]), x["n"], x["clu_exact"], x["n"],
                 x["n_fit_rows_with_code"],
                 "%.3f" % x["clu_in_sample_on_code"] if x["clu_in_sample_on_code"] is not None
                 else "-", x["span_rel_resid"], tuple(x["l2_oracle_decode"]),
                 "" if x["l2_oracle_ok"] else " WRONG", x["l2_oracle_min_margin"]))
        P("|  | seed %d: hashes OK, CLU nway %.4f reproduced; CLU in-sample (all fit rows) %.4f |"
          "  |  |  |  |  |  |  |  |" % (s, r["clu_nway"], r["clu_in_sample_all"]))
    # the partition that matters: code SEEN in fit (>= 1 row) -- is the CLU's failure
    # predicted by the content test?
    seen = [x for x in allrows if x["n_fit_rows_with_code"] > 0]
    cells = Counter()
    for x in seen:
        cont = x["l2_oracle_ok"] and (x["clu_in_sample_on_code"] or 0) == 1.0
        clu_ok = x["clu_exact"] / x["n"] > 0.5
        cells[(cont, clu_ok)] += x["n"]
    P("\nrows whose truth code HAS fit rows (n = %d rows): content-OK (L2-oracle right AND "
      "in-sample 1.000 on the code) x CLU right (>0.5 on the tuple-code group):"
      % sum(x["n"] for x in seen))
    for k in sorted(cells):
        P("   content_ok=%s clu_ok=%s : %d rows" % (k[0], k[1], cells[k]))
    res["contingency_rows"] = {"content_ok=%s,clu_ok=%s" % k: v for k, v in cells.items()}
    unseen = [x for x in allrows if x["n_fit_rows_with_code"] == 0]
    P("rows whose truth code has ZERO fit rows: %d rows, CLU exact %d"
      % (sum(x["n"] for x in unseen), sum(x["clu_exact"] for x in unseen)))
    res["unseen_code_rows"] = {"n": sum(x["n"] for x in unseen),
                               "clu_exact": sum(x["clu_exact"] for x in unseen)}
    res["seconds"] = time.time() - t0
    with open(os.path.join(out_dir, "Q1-CONTENT-ALLSEEDS.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    with open(os.path.join(out_dir, "Q1-CONTENT-ALLSEEDS.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(LINES) + "\n")
    P("[done %.0fs]" % res["seconds"])


if __name__ == "__main__":
    main(sys.argv[1])
