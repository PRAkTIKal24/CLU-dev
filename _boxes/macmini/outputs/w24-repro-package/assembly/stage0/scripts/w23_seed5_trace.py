"""w23 spoke/seed5-diagnosis -- QUESTION 1.  ZERO-FIT trace of the closed-form CLU read.

    python scripts/w23_seed5_trace.py <out_dir>

Seed 5, BOTH, TEST, base design (A1: latent 128, K 2, 6 atoms, ridge 1.0, [bias|main|pair]
318).  Loads the PERSISTED codec and clu.pkl (file + param sha256 asserted against the
.sha256 file AND MANIFEST.json before anything is read), rebuilds the eval rows exactly as the
runner did (deterministic banks + hold-out + cap_split, nothing fitted), reproduces the CLU's
per-tuple n-way counts, and then traces the read for every held-out n-way tuple:

  1. the design row (318 columns; the non-zero ones labelled),
  2. its projection on the fitted head  c_k = phi W_k  (6 coefficients per well, K = 2),
  3. the addressed wells/atoms and the read  z_hat = sum_k c_k G_k,
  4. the emitted four facts vs the truth,
  5. the locus tests:
       CONTENT  -- can the deployed atoms hold the right value?  oracle c*_k = argmin
                   ||c G_k - z_true[ch_k]||, decode(sum c*_k G_k) == truth ?
       CODEC    -- decode(encode(truth)) == truth ?  (frozen codec, outside the CLU)
       ADDRESS  -- ||c - c*||, and whether the head's c decodes right,
       DESIGN   -- estimability of phi(t) in the row space of the train design (full, and
                   restricted to the train rows that share this tuple's pair support), the
                   rank of that restricted design, per-column rank drop, ridge leverage;
                   in-sample CLU accuracy on the train rows carrying each of the tuple's
                   pair cells, and the train outcome mix of each pair cell,
       ISOLATION -- block-swap counterfactual between the failing tuple and a same-family
                   tuple the CLU gets right (the difference IS the finding).

⛔ Nothing is fitted, trained, or re-solved.  The head, store, A and Mx are read from the
pickle; ``A`` is asserted equal to Phi_train^T Phi_train rebuilt from the train words.
"""

from __future__ import annotations

import hashlib
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

K = 4
SEED = 5
OUT = "C:/chlu/outputs"
PD = OUT + "/w21-maze-t-seed4/test/persist/BOTH_seed5/test"
RUNJ = OUT + "/w21-maze-t-seed4/test/BOTH_test_seed5.json"
PTJ = OUT + "/w21-maze-t-seed4/replay/check_BOTH_test_seed5_PERTUPLE.json"
FACTS = ("legal", "disp", "event", "pocket")
WN = T.WORD_NAMES
LINES = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LINES.append(s)


def fsha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def verify_file(pdir, name, man):
    """.sha256 side file and MANIFEST must agree, and the file must hash to them."""
    side = open(os.path.join(pdir, name + ".sha256")).read().split()
    side_param, side_file = side[1], side[3]
    fh = fsha(os.path.join(pdir, name))
    rec = {"file_sha256_computed": fh, "file_sha256_sidecar": side_file,
           "file_sha256_manifest": man[name]["file_sha256"],
           "param_sha256_sidecar": side_param,
           "param_sha256_manifest": man[name]["param_sha256"]}
    ok = fh == side_file == man[name]["file_sha256"] and side_param == man[name]["param_sha256"]
    rec["file_match"] = ok
    assert ok, (name, rec)
    return rec


def col_labels(spec):
    lay = spec.layout
    cards = lay["cardinalities"]
    lab = ["bias"]
    for f, c in enumerate(cards):
        lab += ["%s=%d" % (WN[f], v) for v in range(c)]
    for (i, j) in lay["pairs"]:
        for a in range(cards[i]):
            for b in range(cards[j]):
                lab.append("%sx%s=(%d,%d)" % (WN[i], WN[j], a, b))
    assert len(lab) == spec.width
    return lab


def col_involves(spec):
    """For each column, the set of factor indices it involves."""
    lay = spec.layout
    out = [frozenset()]
    for f, c in enumerate(lay["cardinalities"]):
        out += [frozenset([f])] * c
    for (i, j) in lay["pairs"]:
        out += [frozenset([i, j])] * (lay["cardinalities"][i] * lay["cardinalities"][j])
    return out


def main(out_dir):
    import equinox as eqx
    import jax
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    res = {"script": "scripts/w23_seed5_trace.py", "zero_fit": True, "seed": SEED,
           "condition": "BOTH", "leg": "test", "persist_dir": PD, "run_json": RUNJ}

    # ------------------------------------------------------------ hashes FIRST
    man = json.load(open(os.path.join(PD, "MANIFEST.json")))
    hc = {}
    for nm in ("codec.eqx", "clu.pkl"):
        hc[nm] = verify_file(PD, nm, man)
    codec = eqx.tree_deserialise_leaves(os.path.join(PD, "codec.eqx"),
                                        A.Codec(jax.random.PRNGKey(0), int(T.code_dim(K)), 128))
    ph = S.Persist._leaf_hash(jax.tree_util.tree_leaves(eqx.filter(codec, eqx.is_array)))
    hc["codec.eqx"]["param_sha256_computed"] = ph
    assert ph == man["codec.eqx"]["param_sha256"]
    with open(os.path.join(PD, "clu.pkl"), "rb") as fh:
        clu = pickle.load(fh)
    ph = S.Persist._leaf_hash(list(S._array_leaves(clu)))
    hc["clu.pkl"]["param_sha256_computed"] = ph
    assert ph == man["clu.pkl"]["param_sha256"]
    for v in hc.values():
        v["param_match"] = True
    res["hash_checks"] = hc
    P("[hash] codec.eqx + clu.pkl: file sha256 == sidecar == MANIFEST, param sha256 == "
      "sidecar == MANIFEST == recomputed  -> OK")
    P("   clu.pkl param", hc["clu.pkl"]["param_sha256_computed"])

    # ------------------------------------------------------------ rows (no fit)
    J = json.load(open(RUNJ))
    _b, _e, (tr_insts, tr_rows, _n, _m), (va_insts, va_rows, _vn, _m2) = W16.banks(SEED, "test")
    ho = T.make_holdout(K, seed=SEED)
    assert [list(c) for c in ho.nway] == J["holdout"]["nway"]
    train_rows = T.split_rows(tr_rows, ho, K).get("seen", [])
    assert len(train_rows) == J["holdout"]["n_train_rows"], len(train_rows)
    rs = W16.cap_split(T.split_rows(va_rows, ho, K), SEED).get("nway", [])
    assert len(rs) == J["ROWS"]["nway"]["n"]
    Wtr = A.build_words(train_rows)
    Wva = A.build_words(rs)
    spec = clu.spec
    lab = col_labels(spec)
    inv = col_involves(spec)
    Phi_tr, _ = clu.design(Wtr)
    Phi_tr = np.asarray(Phi_tr, np.float64)
    A_re = Phi_tr.T @ Phi_tr
    dA = float(np.abs(A_re - clu.A).max())
    assert dA < 1e-6, dA
    P("[rows] train %d  nway eval %d ; A rebuilt from train words == clu.A (max|d| %.1e)"
      % (len(train_rows), len(rs), dA))
    res["A_rebuild_max_abs_diff"] = dA

    # ------------------------------------------------------------ reproduce
    Zhat_va = clu.read(Wva)
    em_va = T.decode_codes(K, A.codec_decode(codec, Zhat_va))
    exact = A.score_rows(va_insts, rs, em_va)["exact"]
    assert exact == J["ROWS"]["nway"]["CLU"], (exact, J["ROWS"]["nway"]["CLU"])
    PT = json.load(open(PTJ))["PER_TUPLE_NWAY"]["CLU"]
    key4 = lambda w: "%d,%d,%d,%d" % (w[0], w[2], w[3], w[4])        # noqa: E731
    by_t = defaultdict(list)
    for i, r in enumerate(rs):
        by_t[key4(r.words)].append(i)
    judged = [T.judge_transition(va_insts[r.maze], r.state(), r.action, e)
              for r, e in zip(rs, em_va)]
    for t, idx in by_t.items():
        n_ok = sum(judged[i][0] for i in idx)
        assert PT[t] == [len(idx), n_ok], (t, PT[t], len(idx), n_ok)
    P("[reproduce] CLU nway exact %.6f == run JSON; per-tuple counts == %s  -> OK"
      % (exact, os.path.basename(PTJ)))
    res["reproduce"] = {"CLU_nway_exact": exact, "run_json": J["ROWS"]["nway"]["CLU"],
                        "per_tuple_equal": True}

    # ------------------------------------------------------------ helpers
    heads = [np.asarray(W, np.float64) for W in clu.heads]
    Gs = [np.asarray(G, np.float64) for G in clu.store.modes]
    chs = [np.asarray(c, np.int64) for c in clu.store.well_channels]
    res["store"] = {"K": len(Gs), "n_atoms": [int(G.shape[0]) for G in Gs],
                    "well_widths": [int(c.size) for c in chs],
                    "deployed_rank": [int(r) for r in clu.store.deployed_rank],
                    "atom_row_norms": [np.linalg.norm(G, axis=1).round(4).tolist() for G in Gs]}
    P("[store] K=%d wells, widths %s, atoms %s, deployed rank %s"
      % (len(Gs), res["store"]["well_widths"], res["store"]["n_atoms"],
         res["store"]["deployed_rank"]))

    def read_phi(phi):
        phi = np.asarray(phi, np.float64).reshape(1, -1)
        z = np.zeros((1, clu.store.dim))
        cs = []
        for W, G, ch in zip(heads, Gs, chs):
            c = phi @ W
            cs.append(c[0])
            z[:, ch] = c @ G
        return cs, z

    def dec_logits(z):
        return np.asarray(A.codec_decode(codec, np.asarray(z, np.float32)))

    blocks = T.code_blocks(K)

    def facts_and_margins(lg):
        out, mar = [], []
        for nm in ("legal", "disp", "event", "pocket"):
            o, c = blocks[nm]
            v = lg[o:o + c]
            s = np.sort(v)
            out.append(int(np.argmax(v)))
            mar.append(float(s[-1] - s[-2]))
        return out, mar

    def oracle_coeffs(ztrue):
        cs, z = [], np.zeros(clu.store.dim)
        for G, ch in zip(Gs, chs):
            c, *_ = np.linalg.lstsq(G.T, ztrue[ch], rcond=None)
            cs.append(c)
            z[ch] = c @ G
        return cs, z

    # eigen-decomposition of the train Gram for estimability
    evA, U = np.linalg.eigh(clu.A)
    tol = evA.max() * 1e-10
    rankA = int((evA > tol).sum())
    Unull = U[:, evA <= tol]
    Hinv = np.linalg.inv(clu.A + spec.ridge * np.eye(spec.width))
    res["design_global"] = {"width": spec.width, "rank_train_design": rankA,
                            "null_dim": int(Unull.shape[1]), "ridge": spec.ridge}
    P("[design] width %d, rank(Phi_train) = %d, null-space dim %d (structural one-hot "
      "collinearity: bias = sum of each factor's main block, etc.)"
      % (spec.width, rankA, Unull.shape[1]))

    # train truth + CLU in-sample emissions
    Ztr_hat = clu.read(Wtr)
    em_tr = T.decode_codes(K, A.codec_decode(codec, Ztr_hat))
    tr_truth = np.array([[r.legal, r.disp, r.ev, r.pocket_next] for r in train_rows])
    tr_ok = np.array([T.judge_transition(tr_insts[r.maze], r.state(), r.action, e)[0]
                      for r, e in zip(train_rows, em_tr)])
    res["train_in_sample_exact"] = float(tr_ok.mean())
    P("[train] CLU in-sample exact on the %d fit rows: %.4f" % (len(train_rows), tr_ok.mean()))

    # ------------------------------------------------------------ per-tuple trace
    FREE = [t for t in by_t if t in ("2,0,1,2", "2,0,3,2", "2,0,3,3", "3,0,1,0")]
    SUMMARY = []
    TR = {}
    for t in sorted(by_t):
        idx = by_t[t]
        n_ok = sum(judged[i][0] for i in idx)
        fam = None
        tt = tuple(int(x) for x in t.split(","))
        for f, cands in T._nway_candidates(K, ho.value_colour).items():
            if tt in cands:
                fam = f
        reqs = Counter(tuple(int(x) for x in rs[i].words) for i in idx)
        truths = Counter(tuple(int(x) for x in (rs[i].legal, rs[i].disp, rs[i].ev,
                                                 rs[i].pocket_next)) for i in idx)
        emits = Counter(tuple(int(x) for x in em_va[i]) for i in idx)
        why = Counter(judged[i][1] for i in idx)
        rec = {"tuple": t, "family": fam, "n": len(idx), "clu_exact": n_ok,
               "requests": {str(k): v for k, v in reqs.items()},
               "truths": {str(k): v for k, v in truths.items()},
               "emitted": {str(k): v for k, v in emits.items()},
               "judge_reasons": dict(why), "per_request": []}
        # per distinct request
        for w, cnt in sorted(reqs.items()):
            ii = [i for i in idx if tuple(int(x) for x in rs[i].words) == w]
            phi = np.asarray(clu.design(np.array([w], np.float64))[0][0], np.float64)
            nz = np.flatnonzero(phi)
            cs, z = read_phi(phi)
            assert np.allclose(z[0], Zhat_va[ii[0]], atol=1e-4)
            lg = dec_logits(z)[0]
            f_emit, marg = facts_and_margins(lg)
            tru = [rs[ii[0]].legal, rs[ii[0]].disp, rs[ii[0]].ev, rs[ii[0]].pocket_next]
            assert all([rs[j].legal, rs[j].disp, rs[j].ev, rs[j].pocket_next] == tru for j in ii)
            code_true = T.encode_outcome(K, *tru)
            ztrue = A.codec_encode(codec, code_true[None])[0].astype(np.float64)
            codec_rt = facts_and_margins(dec_logits(ztrue[None])[0])[0]
            cstar, zor = oracle_coeffs(ztrue)
            or_facts, or_marg = facts_and_margins(dec_logits(zor[None])[0])
            # estimability
            null_frac = float(np.linalg.norm(Unull.T @ phi) / np.linalg.norm(phi))
            lev = float(phi @ Hinv @ phi)
            # restricted support design: train rows sharing >= 1 pair cell with phi
            pair_nz = [j for j in nz if len(inv[j]) == 2]
            sup = np.flatnonzero((Phi_tr[:, pair_nz] > 0).any(axis=1))
            Ps = Phi_tr[sup]
            rank_s = int(np.linalg.matrix_rank(Ps))
            coef, *_ = np.linalg.lstsq(Ps.T, phi, rcond=None)
            resid_s = float(np.linalg.norm(Ps.T @ coef - phi) / np.linalg.norm(phi))
            # per-column rank drop within the support design (is the column a combo of others?)
            coldrop = {}
            for j in nz:
                keep = [c for c in range(spec.width) if c != j]
                coldrop[lab[j]] = rank_s - int(np.linalg.matrix_rank(Ps[:, keep]))
            # per pair-cell support in train: count, outcome mix, CLU in-sample exact
            cells = []
            for j in nz:
                if len(inv[j]) != 2:
                    continue
                m = Phi_tr[:, j] > 0
                mix = Counter(tuple(int(x) for x in tr_truth[k]) for k in np.flatnonzero(m))
                cells.append({"col": int(j), "label": lab[j], "n_train": int(m.sum()),
                              "train_outcome_mix_top3": [[str(a), b] for a, b in mix.most_common(3)],
                              "frac_train_rows_with_this_truth": float(
                                  sum(v for a, v in mix.items() if list(a) == tru) / max(m.sum(), 1)),
                              "clu_in_sample_exact": float(tr_ok[m].mean()) if m.any() else None})
            # contributions of each non-zero column to c_k
            contrib = {lab[j]: [(phi[j] * heads[k][j]).round(4).tolist() for k in range(len(Gs))]
                       for j in nz}
            pr = {"request_words": list(w), "n_rows": cnt,
                  "design_nonzero_cols": [[int(j), lab[j], float(phi[j])] for j in nz],
                  "design_row_318": phi.astype(int).tolist(),
                  "c_head": [c.round(5).tolist() for c in cs],
                  "c_oracle": [c.round(5).tolist() for c in cstar],
                  "c_head_minus_oracle_norm": [float(np.linalg.norm(a - b)) for a, b in zip(cs, cstar)],
                  "c_oracle_norm": [float(np.linalg.norm(b)) for b in cstar],
                  "zhat_minus_ztrue_norm": float(np.linalg.norm(z[0] - ztrue)),
                  "zoracle_minus_ztrue_norm": float(np.linalg.norm(zor - ztrue)),
                  "ztrue_norm": float(np.linalg.norm(ztrue)),
                  "emitted": f_emit, "emitted_margins": [round(x, 4) for x in marg],
                  "truth": [int(x) for x in tru],
                  "wrong_facts": [FACTS[q] for q in range(4) if f_emit[q] != tru[q]],
                  "codec_roundtrip_of_truth": codec_rt,
                  "codec_roundtrip_ok": codec_rt == [int(x) for x in tru],
                  "oracle_atoms_decode": or_facts, "oracle_margins": [round(x, 4) for x in or_marg],
                  "CONTENT_ok_atoms_can_hold_truth": or_facts == [int(x) for x in tru],
                  "estimable_null_fraction": null_frac,
                  "ridge_leverage_phi_Hinv_phi": lev,
                  "support_rows": int(sup.size), "support_design_rank": rank_s,
                  "phi_in_rowspace_of_support_rel_resid": resid_s,
                  "support_rank_drop_per_nonzero_col": coldrop,
                  "pair_cells": cells, "contrib_to_c": contrib}
            rec["per_request"].append(pr)
        TR[t] = rec
        pr0 = rec["per_request"]
        SUMMARY.append({"tuple": t, "family": fam, "n": len(idx), "clu_exact": n_ok,
                        "wrong_facts": sorted({f for p in pr0 for f in p["wrong_facts"]}),
                        "codec_rt_ok": all(p["codec_roundtrip_ok"] for p in pr0),
                        "content_ok": all(p["CONTENT_ok_atoms_can_hold_truth"] for p in pr0),
                        "max_null_frac": max(p["estimable_null_fraction"] for p in pr0),
                        "max_leverage": max(p["ridge_leverage_phi_Hinv_phi"] for p in pr0),
                        "max_c_err": max(max(p["c_head_minus_oracle_norm"]) for p in pr0),
                        "min_pair_cell_clu_in_sample": min(
                            (c["clu_in_sample_exact"] for p in pr0 for c in p["pair_cells"]
                             if c["clu_in_sample_exact"] is not None), default=None)})
    res["SUMMARY"] = SUMMARY
    res["TRACE"] = TR

    # ------------------------------------------------------------ print the traces
    P("\n================ SUMMARY: every held-out n-way tuple, seed 5 BOTH TEST ================")
    P("| tuple | family | n | CLU exact | wrong facts | codec round-trip ok | atoms can hold truth "
      "(CONTENT ok) | phi null-space frac | ridge leverage | max ||c-c*|| | min in-sample exact "
      "over its pair cells |")
    P("|---|---|---|---|---|---|---|---|---|---|---|")
    for s in SUMMARY:
        P("| (%s) | %s | %d | %d/%d | %s | %s | %s | %.1e | %.4f | %.3f | %s |"
          % (s["tuple"], s["family"], s["n"], s["clu_exact"], s["n"],
             ",".join(s["wrong_facts"]) or "-", s["codec_rt_ok"], s["content_ok"],
             s["max_null_frac"], s["max_leverage"], s["max_c_err"],
             "%.3f" % s["min_pair_cell_clu_in_sample"]
             if s["min_pair_cell_clu_in_sample"] is not None else "-"))

    for t in FREE:
        rec = TR[t]
        P("\n================ TRACE tuple (%s) family %s n=%d  CLU %d/%d ================"
          % (t, rec["family"], rec["n"], rec["clu_exact"], rec["n"]))
        P("truths", rec["truths"], " emitted", rec["emitted"], " judge", rec["judge_reasons"])
        for pr in rec["per_request"]:
            P("-- request words %s (%s)  rows %d" % (pr["request_words"], ",".join(WN),
                                                       pr["n_rows"]))
            P("  1. design row: %d of 318 columns non-zero:" % len(pr["design_nonzero_cols"]))
            P("     ", "; ".join("[%d] %s" % (j, l) for j, l, _ in pr["design_nonzero_cols"]))
            for k in range(len(Gs)):
                P("  2. well %d: c_head   = %s" % (k, np.round(pr["c_head"][k], 4).tolist()))
                P("     well %d: c_oracle = %s   ||c-c*|| = %.4f  ||c*|| = %.4f"
                  % (k, np.round(pr["c_oracle"][k], 4).tolist(),
                     pr["c_head_minus_oracle_norm"][k], pr["c_oracle_norm"][k]))
            P("  3. ||z_hat - z_true|| = %.4f ; ||z_oracle - z_true|| = %.4f ; ||z_true|| = %.4f"
              % (pr["zhat_minus_ztrue_norm"], pr["zoracle_minus_ztrue_norm"], pr["ztrue_norm"]))
            P("  4. emitted %s (margins %s) vs truth %s -> wrong: %s"
              % (pr["emitted"], pr["emitted_margins"], pr["truth"], pr["wrong_facts"] or "none"))
            P("  5. codec(truth) round-trip %s ok=%s ; ORACLE atoms decode %s (margins %s) "
              "CONTENT-ok=%s" % (pr["codec_roundtrip_of_truth"], pr["codec_roundtrip_ok"],
                                 pr["oracle_atoms_decode"], pr["oracle_margins"],
                                 pr["CONTENT_ok_atoms_can_hold_truth"]))
            P("     estimability: null-space fraction of phi %.2e ; ridge leverage %.4f ; "
              "support rows %d, rank(support design) %d, phi in rowspace(support) rel-resid %.2e"
              % (pr["estimable_null_fraction"], pr["ridge_leverage_phi_Hinv_phi"],
                 pr["support_rows"], pr["support_design_rank"],
                 pr["phi_in_rowspace_of_support_rel_resid"]))
            P("     rank drop when each non-zero column is removed from the support design:",
              pr["support_rank_drop_per_nonzero_col"])
            P("     pair cells of this request in TRAIN: n / CLU in-sample exact / frac with "
              "this truth / top outcomes")
            for c in pr["pair_cells"]:
                P("       %-26s n=%5d  in-sample %.3f  truth-frac %.3f  %s"
                  % (c["label"], c["n_train"], c["clu_in_sample_exact"] or 0.0,
                     c["frac_train_rows_with_this_truth"], c["train_outcome_mix_top3"]))

    # ------------------------------------------------------------ ISOLATION: block swaps
    P("\n================ ISOLATION: failing vs same-triple correct tuple ================")
    iso = {}
    for bad, good in (("2,0,3,2", "2,0,3,3"), ("2,0,1,2", "3,0,1,0"), ("2,0,1,2", "2,0,3,3")):
        wb = TR[bad]["per_request"][0]["request_words"]
        wg = TR[good]["per_request"][0]["request_words"]
        pb_ = np.asarray(clu.design(np.array([wb], np.float64))[0][0], np.float64)
        pg_ = np.asarray(clu.design(np.array([wg], np.float64))[0][0], np.float64)
        diff_f = [f for f in range(5) if wb[f] != wg[f]]
        tru_b = TR[bad]["per_request"][0]["truth"]
        # blocks: main per factor, pair per (i,j)
        blk = {}
        for j in range(spec.width):
            if len(inv[j]) == 0:
                continue
            name = "main:" + WN[min(inv[j])] if len(inv[j]) == 1 else \
                "pair:" + "x".join(WN[q] for q in sorted(inv[j]))
            blk.setdefault(name, []).append(j)
        rows_ = []
        for name, cols in blk.items():
            if not any(f in diff_f for f in [WN.index(x) for x in
                                             name.split(":")[1].split("x")]):
                continue
            phi = pb_.copy()
            phi[cols] = pg_[cols]           # the bad request with ONE block taken from good
            f, m = facts_and_margins(dec_logits(read_phi(phi)[1])[0])
            # and the good request with that ONE block taken from bad
            phi2 = pg_.copy()
            phi2[cols] = pb_[cols]
            f2, m2 = facts_and_margins(dec_logits(read_phi(phi2)[1])[0])
            dc = [(phi[cols] - pb_[cols]) @ heads[k][cols] for k in range(len(Gs))]
            rows_.append({"block": name, "bad_with_good_block_emits": f,
                          "bad_with_good_block_margins": [round(x, 3) for x in m],
                          "fixes_bad": f == tru_b,
                          "good_with_bad_block_emits": f2,
                          "breaks_good": f2 != TR[good]["per_request"][0]["truth"],
                          "delta_c_norm_per_well": [float(np.linalg.norm(d)) for d in dc]})
        iso["%s vs %s" % (bad, good)] = {"bad_words": wb, "good_words": wg,
                                        "differing_words": [WN[f] for f in diff_f],
                                        "bad_truth": tru_b,
                                        "good_truth": TR[good]["per_request"][0]["truth"],
                                        "block_swaps": rows_}
        P("\n(%s) words %s  vs  (%s) words %s ; differ in %s"
          % (bad, wb, good, wg, [WN[f] for f in diff_f]))
        P("| block swapped | bad+good-block emits | fixes bad? | good+bad-block emits | breaks "
          "good? | ||delta c|| per well |")
        P("|---|---|---|---|---|---|")
        for r in rows_:
            P("| %s | %s | %s | %s | %s | %s |" % (r["block"], r["bad_with_good_block_emits"],
                                                   r["fixes_bad"], r["good_with_bad_block_emits"],
                                                   r["breaks_good"],
                                                   np.round(r["delta_c_norm_per_well"], 3).tolist()))
    res["ISOLATION"] = iso

    # ------------------------------------------------------------ dir-2 in-sample scan
    P("\n================ in-sample CLU exact on TRAIN, free-family cells (pocket c, no door, "
      "tint b!=c, b,c in 1..3) by direction ================")
    scan = {}
    for di in range(5):
        m = np.array([(w[0] in (1, 2, 3) and w[2] == 0 and w[3] in (1, 2, 3) and w[3] != w[0]
                       and w[4] == di) for w in Wtr])
        scan[di] = {"n": int(m.sum()), "clu_in_sample": float(tr_ok[m].mean()) if m.any() else None}
    P(json.dumps(scan))
    res["train_free_cells_by_dir"] = scan
    agent2 = {}
    for di in range(5):
        m = (Wtr[:, 0] == 2) & (Wtr[:, 4] == di)
        agent2[di] = {"n": int(m.sum()), "clu_in_sample": float(tr_ok[m].mean()) if m.any() else None,
                      "truth_legal_frac": float(tr_truth[m, 0].mean()) if m.any() else None}
    P("train rows with w_agent=2 by direction:", json.dumps(agent2))
    res["train_agent2_by_dir"] = agent2

    res["seconds"] = time.time() - t0
    with open(os.path.join(out_dir, "Q1-TRACE.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    with open(os.path.join(out_dir, "Q1-TRACE.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(LINES) + "\n")
    P("[done %.0fs]" % res["seconds"])


if __name__ == "__main__":
    main(sys.argv[1])
